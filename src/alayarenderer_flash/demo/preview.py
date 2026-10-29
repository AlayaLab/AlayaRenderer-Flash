#!/usr/bin/env python3
"""Serve SuperTuxKart source/render preview and G-buffer panels."""

from __future__ import annotations

import argparse
import array
import base64
import errno
import fcntl
import hashlib
import hmac
import io
import json
import math
import os
import secrets
import select
import shutil
import signal
import socket
import struct
import subprocess
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageFilter

FFMPEG_CANDIDATES = [path for path in (os.environ.get("ALAYA_FFMPEG", ""), shutil.which("ffmpeg") or "") if path]


VIDEO_PROFILES = {
    "fast": {
        "label": "Fast",
        "width": 640,
        "fps": 24,
        "bitrate_kbps": 1800,
        "x264_preset": "ultrafast",
        "x264_threads": 2,
        "x264_params": "sliced-threads=1:sync-lookahead=0:rc-lookahead=0",
    },
    "balanced": {
        "label": "Balanced",
        "width": 960,
        "fps": 30,
        "bitrate_kbps": 4200,
        "x264_preset": "ultrafast",
        "x264_threads": 4,
        "x264_params": "sliced-threads=1:sync-lookahead=0:rc-lookahead=0",
    },
    "quality": {
        "label": "Quality",
        "width": 1280,
        "fps": 30,
        "bitrate_kbps": 8000,
        "x264_preset": "ultrafast",
        "x264_threads": 2,
        "x264_params": "sliced-threads=1:sync-lookahead=0:rc-lookahead=0",
    },
}


MAX_INPUT_PAYLOAD_BYTES = 64 * 1024
MAX_WS_TEXT_BYTES = 64 * 1024
# Every POST body (input, prompt, track, recording) shares these limits.
MAX_POST_BODY_BYTES = MAX_INPUT_PAYLOAD_BYTES
POST_BODY_READ_TIMEOUT_S = 10.0
MAX_CONCURRENT_POST_BODIES = 16
# Recording start/stop are operator-only: disabled unless this token is set,
# and then each request must carry it in ADMIN_TOKEN_HEADER.
ADMIN_TOKEN_ENV = "ALAYA_PREVIEW_ADMIN_TOKEN"
ADMIN_TOKEN_FILE_ENV = "ALAYA_PREVIEW_ADMIN_TOKEN_FILE"
ADMIN_TOKEN_HEADER = "X-Alaya-Admin-Token"
# Comma-separated extra origins (e.g. https://demo.example.org) accepted for
# browser POST and WebSocket requests besides the server's own Host.
ALLOWED_ORIGINS_ENV = "ALAYA_PREVIEW_ALLOWED_ORIGINS"
MAX_PRIMARY_STREAM_FPS = 60.0
MODEL_FRAME_POLL_FPS = 120.0


def primary_stream_interval(max_fps: float) -> float:
    """Return the capped cadence for source preview transmission."""
    return 1.0 / max(1.0, min(MAX_PRIMARY_STREAM_FPS, float(max_fps)))


class RequestBodyError(ValueError):
    """A POST body rejected before or while reading it."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def parse_content_length(raw: str | None) -> int:
    """Return a validated Content-Length within MAX_POST_BODY_BYTES."""
    try:
        size = int(str(raw if raw is not None else "0").strip())
    except ValueError:
        raise RequestBodyError(400, "invalid_content_length") from None
    if size < 0:
        raise RequestBodyError(400, "invalid_content_length")
    if size > MAX_POST_BODY_BYTES:
        raise RequestBodyError(413, "payload_too_large")
    return size


def parse_json_object(raw: bytes) -> dict[str, object]:
    """Decode a JSON body that must be an object; an empty body is ``{}``."""
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RequestBodyError(400, "invalid_json") from None
    if not isinstance(data, dict):
        raise RequestBodyError(400, "json_object_required")
    return data


def admin_token_from_env() -> str:
    """Return the operator token from the environment or a token file; empty
    means recording endpoints stay disabled."""
    token = os.environ.get(ADMIN_TOKEN_ENV, "").strip()
    token_file = os.environ.get(ADMIN_TOKEN_FILE_ENV, "").strip()
    if not token and token_file:
        try:
            token = Path(token_file).expanduser().read_text(encoding="utf-8").strip()
        except OSError:
            token = ""
    return token


def allowed_origins_from_env() -> frozenset[str]:
    raw = os.environ.get(ALLOWED_ORIGINS_ENV, "")
    return frozenset(item.strip().rstrip("/").lower() for item in raw.split(",") if item.strip())


def origin_allowed(origin: str | None, host: str | None, extra_origins: frozenset[str]) -> bool:
    """Accept requests without an Origin (non-browser clients), same-host
    origins, and explicitly configured origins."""
    if origin is None:
        return True
    clean = origin.strip().rstrip("/").lower()
    if clean in extra_origins:
        return True
    parsed = urlparse(clean)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    return bool(host) and parsed.netloc == str(host).strip().lower()


def configured_presentation_fps(raw: str | None, max_fps: float, status_path: Path | None = None) -> float:
    """Use the actual producer cadence, bounded by the configured transport ceiling."""
    if status_path is not None:
        try:
            raw = json.loads(status_path.read_text()).get("publisher_nominal_fps", raw)
        except (OSError, ValueError, TypeError):
            pass
    try:
        nominal = float(raw) if raw is not None else 30.0
    except (TypeError, ValueError):
        nominal = 30.0
    if not math.isfinite(nominal) or nominal <= 0:
        nominal = 30.0
    return max(1.0, min(nominal, MAX_PRIMARY_STREAM_FPS, max_fps))


def render_game_page(body: bytes, nominal_fps: float) -> bytes:
    return body.replace(b"__MODEL_PRESENTATION_FPS__", str(nominal_fps).encode("ascii")).replace(
        b"__VIDEO_PROFILE_FPS__", str(max(12, int(round(nominal_fps)))).encode("ascii")
    )


def model_frame_poll_interval() -> float:
    """Poll model sequence changes faster than the model publication rate."""
    return 1.0 / MODEL_FRAME_POLL_FPS


def advance_stream_deadline(deadline: float, interval: float, now: float) -> float:
    """Advance a periodic deadline without changing phase after an overrun."""
    if deadline <= 0.0:
        return now + interval
    periods = max(1, math.floor((now - deadline) / interval) + 1)
    next_deadline = deadline + periods * interval
    if next_deadline <= now + interval * 1e-9:
        next_deadline += interval
    return next_deadline


def input_state_lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


def format_server_input_event_id(server_instance_id: str, reason: str, counter: int) -> str:
    return f"{server_instance_id}:server:{reason}:{counter}"


def coalesced_input_boundary(
    source_seq: int,
    source_frame: int,
    render_identity: tuple[object, ...],
    watermark: dict[str, object],
) -> tuple[int, int]:
    if source_seq < 0 or source_frame < 0:
        return -1, -1

    identity = tuple(render_identity)
    candidate_seq = source_seq + 1
    candidate_frame = source_frame + 1
    if tuple(watermark.get("identity", ())) != identity:
        watermark.clear()
    else:
        last_seq = int(watermark.get("source_seq", -1))
        last_frame = int(watermark.get("source_frame", -1))
        # Reuse an unconsumed target. Rapid input events are distinguished by
        # input_event_id and must not push the model boundary into the future.
        if last_seq >= 0 and source_seq < last_seq:
            candidate_seq = last_seq
        if last_frame >= 0 and source_frame < last_frame:
            candidate_frame = last_frame

    watermark.update(
        {
            "identity": identity,
            "source_seq": candidate_seq,
            "source_frame": candidate_frame,
        }
    )
    return candidate_seq, candidate_frame


@contextmanager
def input_state_commit_lock(path: Path) -> Iterator[None]:
    lock_path = input_state_lock_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


VIDEO_TEST_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>STK H.264 Stream Test</title>
  <style>
    * { box-sizing: border-box; }
    [hidden] { display: none !important; }
    [hidden] { display: none !important; }
    html, body {
      margin: 0;
      min-height: 100%;
      background: #090a0b;
      color: #eceff1;
      font: 13px/1.35 system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    body { min-height: 100vh; padding: 14px; }
    main { max-width: 1100px; margin: 0 auto; display: grid; gap: 12px; }
    .panel {
      overflow: hidden;
      background: #111312;
      border: 1px solid #303432;
      border-radius: 6px;
      display: grid;
      grid-template-rows: 34px minmax(0, 1fr);
    }
    .title {
      height: 34px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 10px;
      background: #1a1d1b;
      border-bottom: 1px solid #303432;
      font-weight: 900;
    }
    video {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #050607;
    }
    .status { color: #aeb5b0; font-weight: 800; }
  </style>
</head>
<body>
<main>
  <section class="panel">
    <div class="title"><span>Rendered Output · H.264 fMP4</span><span id="status">loading</span></div>
    <video id="video" autoplay muted playsinline controls></video>
  </section>
</main>
<script>
const video = document.getElementById('video');
const status = document.getElementById('status');
video.src = `/video/rendered.mp4?t=${Date.now()}`;
video.addEventListener('playing', () => { status.textContent = 'playing'; });
video.addEventListener('waiting', () => { status.textContent = 'buffering'; });
video.addEventListener('error', () => {
  const err = video.error ? `${video.error.code}` : 'unknown';
  status.textContent = `video error ${err}`;
});
video.play().catch(() => { status.textContent = 'click play'; });
</script>
</body>
</html>
"""


LANDING_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AlayaRenderer-Flash · Generative World Renderer at the Speed of Play</title>
  <meta name="description" content="A real-time generative forward world renderer for prompt-controllable interactive gameplay.">
  <style>
    :root {
      color-scheme: dark;
      --ink: #f1f0e9;
      --muted: #aaa9a2;
      --line: rgba(255,255,255,.17);
      --accent: #b8a6ff;
      --panel: #111310;
    }
    * { box-sizing: border-box; }
    html { min-height: 100%; background: #090a08; }
    body {
      min-height: 100vh;
      margin: 0;
      overflow-x: hidden;
      background:
        radial-gradient(circle at 75% 20%, rgba(142,173,92,.12), transparent 36%),
        linear-gradient(135deg, #080907 0%, #0d0f0c 52%, #090a08 100%);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    a { color: inherit; text-decoration: none; }
    .page {
      width: min(1500px, calc(100% - 48px));
      min-height: 100vh;
      margin: 0 auto;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }
    .nav {
      height: 86px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--line);
    }
    .wordmark {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .mark {
      width: 32px;
      height: 32px;
      display: grid;
      place-items: center;
    }
    .mark img {
      width: 31px;
      height: 29px;
      object-fit: contain;
      filter: invert(1);
    }
    .brand-text {
      display: grid;
      gap: 5px;
      align-content: center;
    }
    .brand-lab {
      color: var(--ink);
      font-size: 14px;
      line-height: 1;
      font-weight: 800;
      letter-spacing: .12em;
    }
    .brand-project {
      color: #a0a39b;
      font-size: 10.5px;
      line-height: 1;
      font-weight: 680;
      letter-spacing: .09em;
    }
    .nav-links { display: flex; gap: 30px; color: var(--muted); font-size: 13px; }
    .nav-links a { transition: color .18s ease; }
    .nav-links a:hover { color: var(--ink); }
    .hero {
      min-height: 700px;
      padding: clamp(54px, 8vh, 104px) 0;
      display: grid;
      grid-template-columns: minmax(300px, .78fr) minmax(560px, 1.35fr);
      gap: clamp(42px, 6vw, 100px);
      align-items: center;
    }
    .copy { width: 100%; max-width: 560px; min-width: 0; }
    .eyebrow {
      margin: 0 0 28px;
      color: var(--accent);
      font-size: 11px;
      font-weight: 850;
      letter-spacing: .2em;
      text-transform: uppercase;
    }
    h1 {
      margin: 0;
      max-width: 560px;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      font-size: clamp(44px, 4.05vw, 64px);
      line-height: .96;
      font-weight: 680;
      letter-spacing: -.052em;
      text-wrap: balance;
    }
    .lede {
      max-width: 470px;
      margin: 26px 0 0;
      color: #b8b9b2;
      font-size: clamp(15px, 1.15vw, 18px);
      line-height: 1.58;
    }
    .credits {
      margin-top: 24px;
      display: grid;
      gap: 7px;
    }
    .authors {
      max-width: 520px;
      color: #d4d5cf;
      font-size: 12.5px;
      line-height: 1.55;
      font-weight: 620;
      overflow-wrap: anywhere;
    }
    .affiliations {
      color: #7f827a;
      font-size: 10.5px;
      line-height: 1.4;
      font-weight: 700;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .actions { margin-top: 38px; display: flex; flex-wrap: wrap; gap: 12px; }
    .button {
      min-height: 48px;
      padding: 0 20px;
      border: 1px solid var(--line);
      border-radius: 999px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      font-size: 13px;
      font-weight: 760;
      transition: transform .18s ease, background .18s ease, color .18s ease;
    }
    .button.primary { background: var(--accent); border-color: var(--accent); color: #10120d; }
    .button:hover { transform: translateY(-2px); }
    .button.secondary:hover { background: rgba(255,255,255,.07); }
    .facts {
      margin-top: 48px;
      padding-top: 24px;
      border-top: 1px solid var(--line);
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
    }
    .fact strong { display: block; font-size: 14px; }
    .fact span { display: block; margin-top: 5px; color: #858780; font-size: 11px; }
    .demo-wrap { position: relative; min-width: 0; }
    .demo-card {
      position: relative;
      display: block;
      aspect-ratio: 16 / 9;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,.22);
      border-radius: 4px;
      background: #050604;
      box-shadow: 0 40px 100px rgba(0,0,0,.44);
      isolation: isolate;
    }
    .demo-card::after {
      content: "";
      position: absolute;
      inset: 0;
      z-index: 2;
      background: linear-gradient(180deg, rgba(0,0,0,.04), rgba(0,0,0,.42));
      transition: background .25s ease;
    }
    .demo-card:hover::after { background: linear-gradient(180deg, rgba(0,0,0,0), rgba(0,0,0,.23)); }
    .preview {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
      transform: scale(1.01);
    }
    .live-pill {
      position: absolute;
      z-index: 4;
      top: 20px;
      left: 20px;
      min-height: 30px;
      padding: 0 12px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid rgba(255,255,255,.28);
      border-radius: 999px;
      background: rgba(6,8,5,.58);
      backdrop-filter: blur(12px);
      font-size: 10px;
      font-weight: 800;
      letter-spacing: .12em;
    }
    .dot { width: 7px; height: 7px; border-radius: 50%; background: #8fff62; box-shadow: 0 0 12px #8fff62; }
    .play {
      position: absolute;
      z-index: 4;
      top: 50%;
      left: 50%;
      width: clamp(94px, 12vw, 142px);
      height: clamp(94px, 12vw, 142px);
      transform: translate(-50%, -50%);
      display: block;
      border-radius: 50%;
    }
    .play span {
      width: 100%;
      height: 100%;
      border: 1px solid rgba(255,255,255,.64);
      border-radius: 50%;
      background: rgba(7,9,6,.28);
      backdrop-filter: blur(8px);
      display: grid;
      place-items: center;
      font-size: 13px;
      font-weight: 900;
      letter-spacing: .18em;
      transition: transform .25s ease, background .25s ease;
    }
    .play:hover span { transform: scale(1.08); background: var(--accent); color: #10120d; border-color: var(--accent); }
    .caption {
      position: absolute;
      z-index: 4;
      right: 20px;
      bottom: 18px;
      left: 20px;
      display: flex;
      justify-content: space-between;
      gap: 18px;
      color: rgba(255,255,255,.88);
      font-size: 11px;
      font-weight: 720;
      letter-spacing: .04em;
    }
    .demo-index {
      position: absolute;
      top: -26px;
      right: 0;
      color: #73756f;
      font-size: 10px;
      letter-spacing: .15em;
      text-transform: uppercase;
    }
    .footer {
      min-height: 68px;
      border-top: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      color: #777972;
      font-size: 11px;
    }
    @media (min-width: 981px) and (max-height: 850px) {
      .hero {
        min-height: 610px;
        padding: 30px 0 34px;
      }
      .eyebrow { margin-bottom: 18px; }
      h1 { font-size: clamp(48px, 3.75vw, 60px); }
      .credits { margin-top: 18px; }
      .lede { margin-top: 20px; font-size: 16px; line-height: 1.48; }
      .actions { margin-top: 24px; }
    }
    @media (max-width: 980px) {
      .page { width: calc(100% - 28px); max-width: 760px; }
      .nav { height: 72px; }
      .nav-links { display: none; }
      .hero { width: 100%; grid-template-columns: minmax(0, 1fr); min-height: 0; padding: 52px 0 64px; gap: 52px; }
      .copy { width: 100%; max-width: 650px; }
      .demo-wrap { order: 2; }
    }
    @media (max-width: 560px) {
      .eyebrow { font-size: 9.5px; letter-spacing: .13em; }
      h1 { font-size: 42px; line-height: .98; }
      .credits { margin-top: 20px; }
      .authors { font-size: 11.5px; line-height: 1.5; }
      .affiliations { font-size: 9.5px; letter-spacing: .06em; }
      .lede { max-width: 100%; margin-top: 24px; font-size: 15.5px; line-height: 1.55; overflow-wrap: anywhere; }
      .actions { margin-top: 30px; }
      .facts { grid-template-columns: 1fr 1fr; }
      .caption span:last-child { display: none; }
    }
  </style>
</head>
<body>
  <main class="page">
    <nav class="nav">
      <a class="wordmark" href="/" aria-label="AlayaRenderer home">
        <span class="mark"><img src="/assets/alaya-renderer-mark.png" alt="Alaya Lab"></span>
        <span class="brand-text"><span class="brand-lab">ALAYA LAB</span><span class="brand-project">ALAYARENDERER-FLASH</span></span>
      </a>
    </nav>

    <section class="hero" id="about">
      <div class="copy">
        <p class="eyebrow">Real-time generative forward rendering</p>
        <h1>Generative world<br>renderer at the<br>speed of play</h1>
        <div class="credits">
          <div class="authors">Guixu Lin · Zheng-Hui Huang · Siqi Yang · Ming-Hsuan Yang · Kaipeng Zhang · Zhixiang Wang</div>
          <div class="affiliations">Alaya Lab · University of California, Merced</div>
        </div>
        <p class="lede">
          Players control gameplay through standard inputs while the physics
          engine evolves the world state and continuously exports synchronized
          G-buffers encoding geometry and material properties. Conditioned on
          these buffers and a text prompt, AlayaRenderer-Flash generates RGB
          frames at playback rate, enabling real-time visual restyling without
          altering the underlying physics, scene structure, or gameplay logic.
        </p>
        <div class="actions">
          <a class="button primary" href="https://arxiv.org/abs/2607.18703" target="_blank" rel="noopener">Technical report <span aria-hidden="true">↗</span></a>
        </div>
      </div>

      <div class="demo-wrap">
        <div class="demo-card" id="demo">
          <video class="preview" autoplay muted loop playsinline preload="metadata" poster="/assets/landing-demo-poster.png?v=paper-demo-20260720">
            <source src="/assets/landing-demo-first-60s.mp4?v=paper-demo-first60-20260720" type="video/mp4">
          </video>
          <span class="live-pill"><i class="dot"></i><span>LIVE SUPERTUXKART</span></span>
          <a class="play" href="/play" aria-label="Enter the interactive live demo"><span>PLAY</span></a>
          <span class="caption" style="justify-content:flex-end"><span>CLICK TO PLAY</span></span>
        </div>
      </div>
    </section>

    <footer class="footer"><span>ALAYA LAB · JULY 2026</span></footer>
  </main>
</body>
</html>
"""

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AlayaRenderer Flash · Live SuperTuxKart</title>
  <style>
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      min-height: 100%;
      background: #090a0b;
      color: #eceff1;
      font: 13px/1.35 system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    body {
      min-height: 100vh;
      overflow-y: auto;
      padding: 12px;
    }
    .dashboard {
      width: 100%;
      max-width: 1680px;
      margin: 0 auto;
      display: grid;
      gap: 12px;
    }
    .topbar {
      min-height: 68px;
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(520px, 1.35fr);
      gap: 12px;
      padding: 8px 14px;
      background: #151716;
      border: 1px solid #303432;
      border-radius: 6px;
      align-items: center;
    }
    .brand { min-width: 0; }
    .app-title {
      margin: 0;
      font-size: 19px;
      font-weight: 900;
      letter-spacing: 0;
      color: #f6f7f5;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .subtitle {
      margin-top: 3px;
      color: #aeb5b0;
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .status-strip {
      display: grid;
      grid-template-columns: repeat(5, minmax(82px, 1fr));
      gap: 8px;
      min-width: 0;
    }
    .metric {
      min-width: 0;
      padding: 8px 10px;
      background: #0e100f;
      border: 1px solid #2b302e;
      border-radius: 6px;
      display: grid;
      gap: 2px;
    }
    .metric .label {
      color: #88928d;
      font-size: 10px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: .04em;
    }
    .metric .value {
      min-width: 0;
      color: #f5f7f4;
      font-size: 13px;
      font-weight: 900;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .metric .value.live { color: #b6f36e; }
    .metric .value.stale { color: #f0be62; }
    .metric .value.offline { color: #f17575; }
    .viewport-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 12px;
      align-items: start;
    }
    .viewport-grid.single {
      max-width: 960px;
      width: 100%;
      margin: 0 auto;
    }
    .panel {
      min-width: 0;
      min-height: 0;
      overflow: hidden;
      background: #111312;
      border: 1px solid #303432;
      border-radius: 6px;
      display: grid;
      grid-template-rows: 34px minmax(0, 1fr);
      contain: layout paint;
    }
    .viewport-grid .panel {
      height: auto;
    }
    .title {
      height: 34px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 0 10px;
      color: #f6f7f5;
      background: #1a1d1b;
      border-bottom: 1px solid #303432;
      min-width: 0;
    }
    .title span:first-child {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 900;
    }
    .title span:last-child {
      color: #aeb5b0;
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
    }
    img,
    video,
    canvas {
      display: block;
      width: 100%;
      height: 100%;
      min-height: 0;
      object-fit: contain;
      background: #050607;
      image-rendering: auto;
    }
    .viewport-grid img,
    .viewport-grid video {
      aspect-ratio: 16 / 9;
      height: auto;
      object-fit: cover;
    }
    .rendered-stage {
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 9;
      overflow: hidden;
      background: #050607;
      outline: none;
    }
    .rendered-stage:focus-visible {
      box-shadow: inset 0 0 0 2px #cbedbd;
    }
    .rendered-stage .rendered-view {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      aspect-ratio: auto;
      object-fit: cover;
    }
    .rendered-stage .model-view { z-index: 1; }
    .rendered-stage .source-view {
      z-index: 2;
      opacity: 0;
      pointer-events: none;
      transition: opacity 240ms ease-out;
      will-change: opacity;
    }
    .rendered-stage.source-active .source-view {
      opacity: 1;
      transition-duration: 70ms;
    }
    .renderer-warmup {
      position: absolute;
      z-index: 5;
      right: 20px;
      bottom: 20px;
      left: 20px;
      width: min(420px, calc(100% - 40px));
      margin: 0 auto;
      padding: 12px 14px;
      display: grid;
      gap: 8px;
      background: rgba(10, 12, 11, .88);
      border: 1px solid #4b534e;
      border-radius: 6px;
      color: #f2f5ee;
      box-shadow: 0 12px 30px rgba(0, 0, 0, .28);
    }
    .renderer-warmup[hidden] { display: none; }
    .renderer-warmup-heading {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      font-size: 12px;
      font-weight: 850;
    }
    .renderer-warmup-stage { color: #dfffa4; }
    .renderer-warmup-detail {
      color: #aeb5b0;
      font-size: 11px;
      font-weight: 700;
    }
    .renderer-warmup-track {
      height: 4px;
      overflow: hidden;
      background: #252b27;
      border-radius: 2px;
    }
    .renderer-warmup-fill {
      width: 18%;
      height: 100%;
      background: #c5a7ff;
      border-radius: inherit;
      transition: width 220ms ease-out;
    }
    .rendered-stage .rendered-view[hidden] {
      display: none !important;
    }
    .model-inputs-pause-notice {
      position: absolute;
      z-index: 4;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      max-width: calc(100% - 32px);
      min-height: 38px;
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 7px 8px 7px 12px;
      border: 1px solid rgba(255, 255, 255, .2);
      border-radius: 6px;
      background: rgba(8, 11, 9, .88);
      box-shadow: 0 8px 24px rgba(0, 0, 0, .28);
      color: #f4f7f3;
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
    }
    .model-inputs-pause-notice[hidden] { display: none !important; }
    .model-inputs-resume-button {
      min-height: 26px;
      padding: 0 9px;
      border: 1px solid #b9f04c;
      border-radius: 4px;
      background: #b9f04c;
      color: #111509;
      font: inherit;
      font-size: 11px;
      cursor: pointer;
    }
    .model-inputs-resume-button:hover { background: #c9fa68; }
    .model-inputs-resume-button:focus-visible {
      outline: 2px solid #ffffff;
      outline-offset: 2px;
    }
    .title-controls {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }
    .title-controls span {
      color: #aeb5b0;
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
    }
    .buffer-section {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 8px;
      width: 100%;
      max-width: 960px;
      margin: 0 auto;
      min-width: 0;
    }
    .section-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      color: #f6f7f5;
      font-size: 15px;
      font-weight: 900;
    }
    .section-title h2 {
      margin: 0;
      font-size: 15px;
      line-height: 1.2;
    }
    .section-title span:last-child {
      color: #aeb5b0;
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
    }
    .gbuffer-single {
      width: 100%;
      min-width: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(180px, 240px);
      gap: 10px;
      align-items: center;
      padding: 10px;
      background: #111312;
      border: 1px solid #303432;
      border-radius: 6px;
    }
    .gbuffer-toolbar {
      min-width: 0;
      display: grid;
      gap: 8px;
      align-content: center;
    }
    .gbuffer-toolbar-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      min-width: 0;
    }
    .gbuffer-toolbar-head span:first-child {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: #f6f7f5;
      font-weight: 900;
    }
    .gbuffer-toolbar-head span:last-child {
      color: #aeb5b0;
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
    }
    .gbuffer-preview {
      width: 100%;
      aspect-ratio: 16 / 9;
      overflow: hidden;
      background: #050607;
      border: 1px solid #2b302e;
      border-radius: 4px;
    }
    .gbuffer-single img,
    .gbuffer-single canvas {
      aspect-ratio: 16 / 9;
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    .gbuffer-select,
    .gbuffer-rate-select,
    .gbuffer-size-select,
    .transport-select,
    .video-quality-select {
      min-width: 142px;
      height: 26px;
      padding: 0 8px;
      border: 1px solid #3b423d;
      border-radius: 4px;
      background: #0e100f;
      color: #f5f7f4;
      font: inherit;
      font-size: 12px;
      font-weight: 900;
    }
    .gbuffer-rate-select {
      min-width: 104px;
    }
    .gbuffer-size-select {
      min-width: 112px;
    }
    .footer-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(400px, .75fr);
      gap: 12px;
      align-items: stretch;
    }
    .telemetry,
    .keys {
      padding: 10px;
      background: #111312;
      border: 1px solid #303432;
      border-radius: 6px;
      min-width: 0;
    }
    .telemetry {
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 10px;
    }
    .telemetry-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      font-weight: 900;
    }
    .telemetry-head span:last-child {
      color: #aeb5b0;
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
    }
    .telemetry-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }
    .info-cell {
      min-width: 0;
      padding: 7px 8px;
      background: #0e100f;
      border: 1px solid #2b302e;
      border-radius: 5px;
    }
    .info-cell .label {
      display: block;
      color: #88928d;
      font-size: 10px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: .04em;
      margin-bottom: 3px;
    }
    .info-cell .value {
      display: block;
      min-width: 0;
      color: #f5f7f4;
      font-size: 12px;
      font-weight: 850;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .info-cell.wide { grid-column: span 2; }
    .buffer-files {
      grid-column: span 4;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 6px;
    }
    .record-controls {
      grid-column: span 4;
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 6px;
      align-items: center;
    }
    .record-controls input {
      min-width: 0;
      height: 28px;
      padding: 0 8px;
      border: 1px solid #3b423d;
      border-radius: 4px;
      background: #0e100f;
      color: #f5f7f4;
      font: inherit;
      font-size: 12px;
      font-weight: 850;
    }
    .record-button {
      min-width: 0;
      height: 28px;
      border: 1px solid #3b423d;
      border-radius: 4px;
      background: #202421;
      color: #f5f7f4;
      font: inherit;
      font-size: 12px;
      font-weight: 900;
      cursor: pointer;
    }
    .record-button.primary {
      background: #eff5e6;
      color: #080a09;
      border-color: #ffffff;
    }
    .record-button:disabled {
      opacity: .45;
      cursor: default;
    }
    .file-pill {
      min-width: 0;
      height: 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 0 8px;
      background: #171a18;
      border: 1px solid #303432;
      border-radius: 4px;
      color: #9ca6a0;
      font-size: 11px;
      font-weight: 850;
    }
    .file-pill.ok { border-color: #4e7044; color: #cbedbd; }
    .file-pill.bad { border-color: #714045; color: #f1b5b5; }
    .file-pill span {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .keys {
      display: grid;
      grid-template-rows: auto 1fr auto auto;
      align-items: center;
      gap: 10px;
    }
    .input-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      font-weight: 900;
      min-width: 0;
    }
    .input-title span:last-child {
      color: #aeb5b0;
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
    }
    .mode-switch {
      height: 28px;
      display: inline-grid;
      grid-template-columns: 1fr 1fr;
      gap: 0;
      padding: 2px;
      background: #090a0b;
      border: 1px solid #303432;
      border-radius: 5px;
    }
    .mode-button {
      min-width: 68px;
      padding: 0 10px;
      border: 0;
      border-radius: 3px;
      background: transparent;
      color: #8c9690;
      font: inherit;
      font-size: 12px;
      font-weight: 900;
      cursor: pointer;
    }
    .mode-button.active {
      background: #eff5e6;
      color: #080a09;
    }
    .keygrid {
      display: grid;
      grid-template-columns: repeat(5, minmax(42px, 1fr));
      gap: 6px;
      align-content: center;
    }
    .key {
      height: 34px;
      display: grid;
      place-items: center;
      background: #202421;
      color: #89938d;
      border: 1px solid #3a403c;
      border-radius: 4px;
      font-weight: 900;
    }
    .key.on { background: #eff5e6; color: #080a09; border-color: #ffffff; }
    .status { color: #aeb5b0; font-size: 12px; font-weight: 750; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .license-card {
      padding: 11px 12px;
      background: #111312;
      border: 1px solid #303432;
      border-radius: 6px;
      display: grid;
      gap: 5px;
      color: #c4cbc6;
    }
    .license-card h2 {
      margin: 0;
      color: #f6f7f5;
      font-size: 14px;
      line-height: 1.2;
      font-weight: 900;
    }
    .license-card p {
      margin: 0;
      max-width: 1180px;
      color: #aeb5b0;
      font-size: 12px;
      font-weight: 750;
    }
    .license-card a {
      color: #cbedbd;
      font-weight: 900;
      text-decoration: none;
    }
    .license-card a:hover { text-decoration: underline; }
    @media (max-width: 1180px) {
      .topbar { grid-template-columns: 1fr; }
      .status-strip { grid-template-columns: repeat(5, minmax(0, 1fr)); }
      .footer-grid { grid-template-columns: 1fr; }
      .telemetry-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .buffer-files { grid-column: span 2; grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .record-controls { grid-column: span 2; grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .info-cell.wide { grid-column: span 1; }
    }
    @media (max-width: 900px) {
      body { padding: 10px; }
      .viewport-grid { grid-template-columns: 1fr; }
      .viewport-grid .panel { height: auto; min-height: 0; }
      .viewport-grid img { aspect-ratio: 16 / 9; height: auto; }
      .gbuffer-single { grid-template-columns: minmax(0, 1fr) minmax(160px, 220px); }
      .status-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 620px) {
      body { padding: 8px; }
      .dashboard { gap: 10px; }
      .topbar { padding: 10px; }
      .app-title {
        white-space: normal;
        overflow: visible;
        text-overflow: clip;
        font-size: 18px;
      }
      .subtitle {
        white-space: normal;
        overflow: visible;
        text-overflow: clip;
      }
      .status-strip { grid-template-columns: 1fr; }
      .section-title {
        align-items: flex-start;
        flex-direction: column;
        gap: 4px;
      }
      .section-title span:last-child {
        white-space: normal;
      }
      .title span:last-child {
        max-width: 46%;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .gbuffer-single { grid-template-columns: 1fr; }
      .gbuffer-preview { max-width: 240px; }
      .gbuffer-toolbar-head {
        align-items: flex-start;
        flex-direction: column;
        gap: 4px;
      }
      .gbuffer-toolbar-head span:last-child {
        white-space: normal;
      }
      .gbuffer-toolbar .title-controls {
        flex-wrap: wrap;
      }
      .telemetry-grid { grid-template-columns: 1fr; }
      .buffer-files { grid-column: span 1; grid-template-columns: 1fr; }
      .record-controls { grid-column: span 1; grid-template-columns: 1fr; }
      .keygrid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .input-title {
        align-items: flex-start;
        flex-direction: column;
      }
    }

    /* Public game demo layout. */
    * { letter-spacing: 0; }
    html, body { background: #0a0c0b; color: #f4f6f2; }
    body { padding: 0; overflow-x: hidden; }
    button, select, textarea { font: inherit; }
    button:focus-visible, select:focus-visible, textarea:focus-visible, .rendered-stage:focus-visible {
      outline: 2px solid #b9f04c;
      outline-offset: 2px;
      box-shadow: none;
    }
    .dashboard.game-demo {
      width: 100%;
      max-width: 1440px;
      min-width: 0;
      min-height: 100vh;
      margin: 0 auto;
      padding: 18px 24px 24px;
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 14px;
    }
    .dashboard.game-demo > * {
      max-width: 100%;
      min-width: 0;
    }
    .game-demo .topbar {
      min-height: 54px;
      padding: 0 2px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      flex-wrap: nowrap;
      background: transparent;
      border: 0;
      border-radius: 0;
    }
    .game-demo .brand {
      flex: 1 1 auto;
      min-width: 0;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .alaya-mark {
      flex: 0 0 38px;
      width: 38px;
      height: 38px;
      overflow: hidden;
      padding: 6px;
      border: 1px solid #c8bcff;
      border-radius: 6px;
      background: #b7a5ff;
    }
    .alaya-mark img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: transparent;
    }
    .brand-copy {
      min-width: 0;
      display: grid;
      gap: 2px;
    }
    .brand-kicker,
    .section-kicker {
      color: #b9f04c;
      font-size: 10px;
      line-height: 1;
      font-weight: 900;
    }
    .game-demo .app-title {
      margin: 0;
      color: #ffffff;
      font-size: 22px;
      line-height: 1.1;
      font-weight: 900;
      white-space: nowrap;
      overflow: visible;
    }
    .game-demo .app-title span { margin-left: 4px; color: #ff7657; }
    .game-demo .subtitle {
      margin-top: 0;
      color: #929b95;
      font-size: 11px;
      font-weight: 700;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .header-status {
      flex: 0 0 auto;
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 10px;
      min-width: 0;
    }
    .live-badge,
    .session-detail {
      min-height: 32px;
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 0 10px;
      border: 1px solid #343936;
      border-radius: 6px;
      background: #131614;
      color: #d9ddd9;
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
    }
    .live-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #ff7657;
    }
    .live-badge .value.live + *, .live-badge .value.live { color: #dfffa4; }
    .live-badge:has(.value.live) .live-dot { background: #b9f04c; }
    .session-detail span { color: #7f8982; font-size: 10px; font-weight: 900; }
    .track-select {
      width: 138px;
      min-width: 0;
      border: 0;
      background: transparent;
      color: #ffffff;
      font-size: 12px;
      font-weight: 850;
      cursor: pointer;
    }
    .track-control {
      min-height: 38px;
      padding: 4px 10px 4px 12px;
      border-color: rgba(183, 165, 255, .72);
      background: rgba(183, 165, 255, .12);
      box-shadow: 0 0 0 1px rgba(183, 165, 255, .08), 0 8px 24px rgba(0, 0, 0, .18);
    }
    .track-control > span {
      color: #b7a5ff;
      font-size: 9px;
      letter-spacing: .09em;
      text-transform: uppercase;
    }
    .track-control .track-select {
      width: 152px;
      color: #ffffff;
      font-size: 14px;
      font-weight: 900;
    }
    .track-control:hover {
      border-color: #d0c4ff;
      background: rgba(183, 165, 255, .18);
    }
    .track-select:disabled {
      color: #909993;
      cursor: wait;
    }
    .game-demo .mode-switch {
      height: 34px;
      padding: 3px;
      border: 1px solid #343936;
      border-radius: 6px;
      background: #131614;
    }
    .mode-control {
      min-height: 38px;
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 3px 5px 3px 10px;
      border: 1px solid #343936;
      border-radius: 6px;
      background: #131614;
    }
    .mode-label {
      color: #7f8982;
      font-size: 9px;
      font-weight: 900;
      letter-spacing: .1em;
      text-transform: uppercase;
    }
    .mode-control .mode-switch {
      height: 30px;
      padding: 2px;
      border: 0;
      background: transparent;
    }
    .game-demo .mode-button {
      min-width: 66px;
      border-radius: 4px;
      color: #8e9891;
      font-size: 11px;
    }
    .game-demo .mode-button.active { background: #eef5e5; color: #0b0e0c; }
    .game-demo .mode-button:disabled {
      opacity: .38;
      cursor: not-allowed;
      filter: grayscale(1);
    }
    .coming-soon-badge {
      margin-left: 0;
      padding: 2px 4px;
      border: 1px solid currentColor;
      border-radius: 3px;
      font-size: 8px;
      font-weight: 900;
      letter-spacing: .06em;
      line-height: 1;
      text-transform: uppercase;
    }
    .game-demo #manualModeButton {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 92px;
      flex-direction: column;
      gap: 3px;
      line-height: 1;
      white-space: normal;
    }
    .play-area {
      width: min(100%, 1280px, calc((100vh - 285px) * 1.7778));
      min-width: 0;
      margin: 0 auto;
    }
    .game-frame {
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 9;
      overflow: hidden;
      border: 1px solid #3a403c;
      border-radius: 8px;
      background: #040605;
      box-shadow: 0 18px 48px rgba(0, 0, 0, .34);
    }
    .game-demo .rendered-stage {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      aspect-ratio: auto;
      cursor: crosshair;
    }
    .game-demo .rendered-stage .rendered-view { object-fit: cover; }
    .game-hud {
      position: absolute;
      z-index: 3;
      left: 12px;
      right: 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      pointer-events: none;
      color: #f7faf6;
      font-size: 11px;
      font-weight: 850;
    }
    .game-hud-top { top: 12px; }
    .game-hud-bottom { bottom: 12px; }
    .race-label,
    .game-metrics,
    .game-hud-bottom > span,
    .control-readout {
      min-height: 28px;
      display: inline-flex;
      align-items: center;
      padding: 0 9px;
      border: 1px solid rgba(255, 255, 255, .18);
      border-radius: 5px;
      background: rgba(8, 11, 9, .78);
      backdrop-filter: blur(8px);
    }
    .race-label { color: #f0f0f5; }
    .game-metrics { gap: 12px; color: #c8cfca; }
    .game-metrics strong { color: #ffffff; }
    .control-readout {
      gap: 5px;
      padding-right: 5px;
      color: #dfffa4;
    }
    .control-readout span {
      min-width: 56px;
      color: #dfffa4;
      font-size: 9px;
      text-align: right;
    }
    .control-readout kbd {
      --strength: 0;
      width: 20px;
      height: 20px;
      display: inline-grid;
      place-items: center;
      border: 1px solid rgba(255, 255, 255, .18);
      border-radius: 4px;
      background: rgba(255, 255, 255, .08);
      color: #8e9891;
      font: 900 11px/1 system-ui, sans-serif;
    }
    .control-readout kbd.active {
      border-color: #dfffa4;
      background: #b9f04c;
      color: #111509;
      opacity: calc(.58 + var(--strength) * .42);
    }
    .command-dock {
      width: min(100%, 1280px);
      margin: 0 auto;
      padding: 14px 0 16px;
      display: grid;
      gap: 10px;
      border-top: 1px solid #282d2a;
      border-bottom: 1px solid #282d2a;
    }
    .command-heading {
      min-width: 0;
      overflow: hidden;
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 14px;
    }
    .command-heading > div { display: flex; align-items: baseline; gap: 9px; }
    .command-heading > * { min-width: 0; }
    .command-heading h2 {
      margin: 0;
      color: #ffffff;
      font-size: 16px;
      line-height: 1.2;
      font-weight: 900;
    }
    .prompt-status {
      flex: 1 1 0;
      width: 0;
      min-width: 0;
      max-width: 68%;
      color: #98a19b;
      font-size: 12px;
      font-weight: 750;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .prompt-live-label {
      color: #dfffa4;
      font-weight: 850;
    }
    .prompt-status[data-state="queued"] { color: #f4c875; }
    .prompt-status[data-state="error"] { color: #ff9b85; }
    .prompt-presets {
      display: flex;
      align-items: center;
      gap: 7px;
      min-width: 0;
      overflow-x: auto;
      scrollbar-width: none;
    }
    .prompt-presets::-webkit-scrollbar { display: none; }
    .prompt-presets button {
      flex: 0 0 auto;
      min-height: 32px;
      padding: 0 11px;
      border: 1px solid #3a403c;
      border-radius: 6px;
      background: #151816;
      color: #d5dad6;
      font-size: 12px;
      font-weight: 800;
      cursor: pointer;
    }
    .prompt-presets button:hover { border-color: #76816f; color: #ffffff; }
    .prompt-form {
      min-width: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 96px;
      gap: 9px;
      align-items: stretch;
    }
    .prompt-form textarea {
      width: 100%;
      min-height: 54px;
      max-height: 110px;
      resize: vertical;
      padding: 10px 12px;
      border: 1px solid #3a403c;
      border-radius: 6px;
      background: #111412;
      color: #ffffff;
      font-size: 14px;
      line-height: 1.45;
    }
    .prompt-form textarea::placeholder { color: #69736c; }
    .execute-button {
      min-width: 0;
      border: 1px solid #ff947c;
      border-radius: 6px;
      background: #ff7657;
      color: #150b08;
      font-size: 14px;
      font-weight: 950;
      cursor: pointer;
    }
    .execute-button:hover { background: #ff896f; }
    .execute-button:disabled { opacity: .55; cursor: wait; }
    .advanced-panel {
      width: min(100%, 1280px);
      margin: 0 auto;
      border-bottom: 1px solid #282d2a;
      color: #dfe3df;
    }
    .advanced-panel > summary {
      min-height: 42px;
      display: flex;
      align-items: center;
      cursor: pointer;
      color: #9ba49e;
      font-size: 12px;
      font-weight: 850;
    }
    .advanced-panel[open] > summary { color: #ffffff; }
    .advanced-panel .buffer-section {
      max-width: none;
      padding: 4px 0 16px;
    }
    .advanced-panel .gbuffer-single {
      grid-template-columns: minmax(240px, .8fr) minmax(280px, 1.2fr);
      padding: 12px;
      border-radius: 6px;
      background: #111412;
    }
    .game-demo .license-card {
      width: min(100%, 1280px);
      margin: 0 auto;
      padding: 0;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      background: transparent;
      border: 0;
      border-radius: 0;
      color: #6f7972;
      font-size: 11px;
    }
    .game-demo .license-card a { color: #9ca69f; }
    .internal-state[hidden] { display: none !important; }
    @media (max-width: 900px) {
      .dashboard.game-demo { padding: 14px; }
      .play-area { width: 100%; }
      .game-demo .topbar { align-items: flex-start; flex-wrap: wrap; row-gap: 10px; }
      .header-status { width: 100%; justify-content: flex-start; }
      .advanced-panel .gbuffer-single { grid-template-columns: 1fr; }
    }
    @media (max-width: 620px) {
      .dashboard.game-demo { padding: 10px; gap: 11px; }
      .game-demo .topbar { min-height: 48px; gap: 8px; }
      .brand-kicker { display: block; font-size: 7px; }
      .game-demo .brand { display: flex; }
      .alaya-mark { flex-basis: 30px; width: 30px; height: 30px; padding: 4px; border-radius: 5px; }
      .game-demo .app-title { font-size: 17px; }
      .game-demo .subtitle { display: block; max-width: 190px; font-size: 8px; }
      .live-badge { display: none; }
      .header-status {
        flex: 1 1 100%;
        width: 100%;
        max-width: 100%;
        min-width: 0;
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 132px);
        gap: 7px;
      }
      .session-detail {
        width: 100%;
        min-width: 0;
        overflow: hidden;
        padding-inline: 8px;
      }
      .track-select {
        flex: 1 1 0;
        width: 0;
        max-width: 100%;
      }
      .game-demo .mode-switch {
        width: 132px;
        min-width: 0;
        max-width: 100%;
      }
      .game-demo .mode-button {
        min-width: 0;
        padding: 0 2px;
        font-size: 11px;
        white-space: nowrap;
      }
      .game-hud { left: 7px; right: 7px; font-size: 10px; }
      .game-hud-top { top: 7px; }
      .game-hud-bottom { bottom: 7px; }
      .game-metrics span:nth-child(2),
      .game-metrics span:nth-child(3),
      .game-hud-bottom > span:first-child { display: none; }
      .control-readout span { min-width: 48px; font-size: 8px; }
      .command-dock { padding: 12px 0 14px; }
      .command-heading { align-items: flex-start; flex-direction: column; gap: 4px; }
      .prompt-status {
        flex: 0 0 auto;
        width: 100%;
        max-width: 100%;
        display: -webkit-box;
        line-height: 1.35;
        white-space: normal;
        overflow-wrap: anywhere;
        -webkit-box-orient: vertical;
        -webkit-line-clamp: 2;
      }
      .prompt-form { grid-template-columns: minmax(0, 1fr); }
      .prompt-form textarea { min-height: 48px; font-size: 13px; }
      .execute-button { min-height: 42px; }
      .advanced-panel > summary { min-height: 44px; }
      .advanced-panel .title-controls { flex-wrap: wrap; }
      .game-demo .license-card {
        flex-direction: column;
        gap: 5px;
        padding-top: 2px;
        line-height: 1.45;
      }
    }

    /* Alaya Renderer visual language, adapted for the realtime game surface. */
    :root {
      --alaya-accent: #b7a5ff;
      --alaya-accent-strong: #c8bcff;
      --alaya-accent-soft: rgba(183, 165, 255, .10);
      --alaya-bg: #05020b;
      --alaya-surface: #08060b;
      --alaya-panel: rgba(255, 255, 255, .035);
      --alaya-panel-hover: rgba(255, 255, 255, .06);
      --alaya-border: rgba(255, 255, 255, .11);
      --alaya-border-strong: rgba(255, 255, 255, .17);
      --alaya-text: #f0f0f5;
      --alaya-muted: rgba(255, 255, 255, .58);
      --alaya-live: #b9f04c;
      --alaya-flash: #ff7657;
    }
    html,
    body {
      background: var(--alaya-bg);
      color: var(--alaya-text);
      font-family: "Inter Tight", Inter, ui-sans-serif, system-ui,
        -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button,
    select,
    textarea,
    kbd {
      font-family: "Space Grotesk", Inter, ui-sans-serif, system-ui,
        -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button:focus-visible,
    select:focus-visible,
    textarea:focus-visible,
    .rendered-stage:focus-visible { outline-color: var(--alaya-accent); }
    .dashboard.game-demo {
      max-width: 1480px;
      display: flex;
      flex-direction: column;
      background: var(--alaya-bg);
    }
    .game-demo .topbar,
    .command-dock,
    .advanced-panel,
    .game-demo .license-card {
      width: min(100%, 1280px);
      margin-inline: auto;
    }
    .play-area {
      width: min(100%, 1120px, 103.111vh);
      margin-inline: auto;
    }
    .alaya-mark {
      border-color: var(--alaya-accent-strong);
      background: var(--alaya-accent);
      box-shadow: 0 0 0 1px rgba(183, 165, 255, .08);
    }
    .brand-kicker,
    .section-kicker { color: var(--alaya-accent); }
    .game-demo .app-title span { color: var(--alaya-flash); }
    .game-demo .subtitle,
    .session-detail span,
    .prompt-status,
    .advanced-panel > summary,
    .game-demo .license-card { color: var(--alaya-muted); }
    .live-badge,
    .session-detail,
    .game-demo .mode-switch {
      border-color: var(--alaya-border);
      background: var(--alaya-panel);
    }
    .track-select { color: var(--alaya-text); }
    .game-demo .mode-button.active {
      background: var(--alaya-accent);
      color: #090611;
    }
    .game-frame {
      border-color: var(--alaya-border-strong);
      background: var(--alaya-surface);
      box-shadow: 0 24px 80px rgba(0, 0, 0, .42);
    }
    .race-label,
    .game-metrics,
    .game-hud-bottom > span,
    .control-readout {
      border-color: rgba(255, 255, 255, .16);
      background: rgba(5, 2, 11, .76);
    }
    .control-readout,
    .control-readout span { color: var(--alaya-live); }
    .race-label { color: var(--alaya-text); }
    .prompt-live-label {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      color: var(--alaya-live);
      font-weight: 850;
    }
    .prompt-live-label::before {
      content: "";
      flex: 0 0 5px;
      width: 5px;
      height: 5px;
      border-radius: 50%;
      background: currentColor;
    }
    .control-readout kbd.active {
      border-color: #dfffa4;
      background: var(--alaya-live);
    }
    .command-dock,
    .advanced-panel,
    .game-demo .license-card { border-color: var(--alaya-border); }
    .prompt-presets button {
      border-color: var(--alaya-border);
      background: var(--alaya-panel);
      color: rgba(255, 255, 255, .78);
    }
    .prompt-presets button:hover,
    .prompt-presets button.selected {
      border-color: var(--alaya-accent);
      background: var(--alaya-accent-soft);
      color: var(--alaya-accent-strong);
    }
    .prompt-form textarea,
    .advanced-panel .gbuffer-single,
    .gbuffer-select,
    .gbuffer-rate-select,
    .gbuffer-size-select {
      border-color: var(--alaya-border);
      background: var(--alaya-panel);
      color: var(--alaya-text);
    }
    .prompt-form textarea:focus { border-color: var(--alaya-accent); }
    .prompt-form textarea.custom-attention {
      border-color: var(--alaya-accent-strong);
      box-shadow: 0 0 0 3px rgba(183, 165, 255, .18), 0 0 24px rgba(183, 165, 255, .14);
    }
    .execute-button,
    .model-inputs-resume-button {
      border-color: var(--alaya-accent);
      background: var(--alaya-accent);
      color: #090611;
    }
    .execute-button:hover,
    .model-inputs-resume-button:hover { background: var(--alaya-accent-strong); }
    .model-inputs-pause-notice {
      border-color: rgba(183, 165, 255, .34);
      background: rgba(5, 2, 11, .90);
    }
    .advanced-panel[open] > summary { color: var(--alaya-accent-strong); }
    .gbuffer-preview {
      border-color: var(--alaya-border);
      background: var(--alaya-surface);
    }
    .game-demo .license-card { margin-top: auto; }
    .game-demo .license-card a { color: var(--alaya-accent); }
    @media (max-width: 900px) {
      .play-area { width: 100%; }
    }
    @media (max-width: 620px) {
      .game-demo .topbar,
      .play-area,
      .command-dock,
      .advanced-panel,
      .game-demo .license-card { width: 100%; }
    }
    /* Teaser-style centered content rail: every visible section follows the
       exact same width as the rendered frame. */
    .dashboard.game-demo {
      --teaser-content-width: min(100%, 1120px, calc((100vh - 250px) * 1.7778));
      width: 100%;
      max-width: none;
      padding: 12px 18px 16px;
      gap: 10px;
      align-items: center;
    }
    .game-demo .topbar,
    .game-demo .play-area,
    .game-demo .command-dock,
    .game-demo .advanced-panel,
    .game-demo .license-card {
      width: var(--teaser-content-width);
      margin-left: auto;
      margin-right: auto;
    }
    .game-demo .topbar {
      min-height: 58px;
      padding-left: 0;
      padding-right: 0;
    }
    .game-demo .brand-kicker { font-size: 11px; letter-spacing: .11em; }
    /* Title scales with viewport height so it stays one line on short laptops
       (the content rail width is itself derived from 100vh); big screens keep 30px. */
    .game-demo .app-title { font-size: clamp(20px, 3.0vh, 33px); line-height: 1.04; letter-spacing: -.035em; }
    .game-demo .subtitle { margin-top: 3px; font-size: 13px; }
    .game-demo .command-heading .section-kicker {
      font-size: 11px;
      letter-spacing: .12em;
    }
    .game-demo .command-heading h2 {
      margin: 0;
      font-size: 18px;
      line-height: 1.08;
      letter-spacing: -.02em;
    }
    .game-demo .command-heading {
      align-items: flex-start;
      justify-content: flex-start;
      flex-direction: column;
      gap: 6px;
    }
    .game-demo .prompt-status {
      flex: 1 1 auto;
      width: 100%;
      max-width: none;
      font-size: 13px;
      line-height: 1.45;
      white-space: normal;
      overflow: visible;
      text-overflow: clip;
      text-align: left;
    }
    .game-demo .advanced-panel > summary {
      padding: 10px 12px;
      border: 1px solid rgba(183, 165, 255, .34);
      border-radius: 7px;
      background: rgba(183, 165, 255, .07);
      color: rgba(255, 255, 255, .88);
      font-size: 14px;
      font-weight: 850;
    }
    .game-demo .advanced-panel > summary:hover {
      border-color: rgba(183, 165, 255, .62);
      background: rgba(183, 165, 255, .12);
    }
    @media (max-width: 620px) {
      .dashboard.game-demo { padding: 10px; }
      .game-demo .topbar,
      .game-demo .play-area,
      .game-demo .command-dock,
      .game-demo .advanced-panel,
      .game-demo .license-card { width: 100%; }
      .game-demo .app-title { font-size: 22px; }
    }
  </style>
</head>
<body>
<main class="dashboard game-demo">
  <header class="topbar">
    <div class="brand">
      <a class="alaya-mark" href="https://github.com/AlayaLab/AlayaRenderer" target="_blank" rel="noreferrer" aria-label="Alaya Lab AlayaRenderer">
        <img src="/assets/alaya-renderer-mark.png" alt="Alaya Lab">
      </a>
      <div class="brand-copy">
        <span class="brand-kicker">ALAYA LAB</span>
        <h1 class="app-title">AlayaRenderer <span>Flash</span></h1>
        <div class="subtitle" id="topStatus">Live SuperTuxKart</div>
      </div>
    </div>
    <div class="header-status">
      <div class="live-badge"><span class="live-dot"></span><span class="value offline" id="liveStatus">Offline</span></div>
      <label class="session-detail track-control" for="trackSelect">
        <span>Select track</span>
        <select class="track-select" id="trackSelect" aria-label="Select track">
          <option value="black_forest">Black Forest</option>
          <option value="candela_city">Candela City</option>
          <option value="cocoa_temple">Cocoa Temple</option>
          <option value="snowmountain">Northern Resort</option>
          <option value="volcano_island">Volcan Island</option>
          <option value="zengarden">Zen Garden</option>
        </select>
      </label>
      <div class="mode-control">
        <span class="mode-label">Mode</span>
        <div class="mode-switch" role="group" aria-label="Driving mode">
          <button class="mode-button active" id="autoModeButton" type="button">Auto</button>
          <button class="mode-button" id="manualModeButton" type="button" disabled aria-disabled="true" title="Manual mode is coming soon">Manual <span class="coming-soon-badge">Coming soon</span></button>
        </div>
      </div>
    </div>
  </header>

  <section class="play-area" aria-label="Live gameplay">
    <div class="game-frame">
      <div class="game-hud game-hud-top">
        <span class="race-label" id="raceLabel">BLACK FOREST</span>
        <div class="game-metrics">
          <span><strong id="fpsMetric">--</strong> FPS</span>
          <span id="ageMetric">--</span>
          <span id="resolutionMetric">--</span>
        </div>
      </div>
      <div class="rendered-stage" id="renderedStage" tabindex="0" role="application" aria-label="SuperTuxKart live gameplay">
        <video id="renderedVideo" class="rendered-view model-view" autoplay muted playsinline preload="none"></video>
        <img id="renderedImage" class="rendered-view model-view" src="/model.jpg?t=0" alt="SuperTuxKart live gameplay" hidden>
        <img id="sourceImage" class="rendered-view source-view" alt="" aria-hidden="true">
        <div class="model-inputs-pause-notice" id="modelInputsPauseNotice" role="status" hidden>
          <span>Live output paused</span>
          <button class="model-inputs-resume-button" id="modelInputsResumeButton" type="button">Resume</button>
        </div>
        <div class="renderer-warmup" id="rendererWarmup" role="status" aria-live="polite" hidden>
          <div class="renderer-warmup-heading">
            <span id="rendererWarmupStage" class="renderer-warmup-stage">Preparing renderer</span>
            <span id="rendererWarmupDetail" class="renderer-warmup-detail">Please wait</span>
          </div>
          <div class="renderer-warmup-track" role="progressbar" aria-label="Renderer startup progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
            <div id="rendererWarmupFill" class="renderer-warmup-fill"></div>
          </div>
        </div>
      </div>
      <div class="game-hud game-hud-bottom">
        <span id="frameStatus">Waiting for game stream</span>
        <div class="control-readout" id="controlReadout" aria-label="AI driving input">
          <span id="inputStatus">AI input</span>
          <kbd data-control="left">←</kbd>
          <kbd data-control="accel">↑</kbd>
          <kbd data-control="brake">↓</kbd>
          <kbd data-control="right">→</kbd>
        </div>
      </div>
    </div>
  </section>

  <section class="command-dock" aria-labelledby="commandTitle">
    <div class="command-heading">
      <div>
        <span class="section-kicker">TEXT PROMPT</span>
        <h2 id="commandTitle">Change the world</h2>
      </div>
      <span id="promptStatus" class="prompt-status" aria-live="polite">Ready</span>
    </div>
    <div class="prompt-presets" aria-label="Suggested visual styles">
      <button type="button" data-prompt="The render is dark and illuminated by an intense large-scale firestorm, with flickering orange-red key light, hot golden rim highlights, and hard, rapidly shifting high-contrast shadows. Dense smoke, glowing embers, and heat haze fill the air; the color design favors incandescent orange, crimson red, soot black, and molten gold.">Large Fire</button>
      <button type="button" data-prompt="The render has an apocalyptic volcanic-eruption grade, with powerful lava-red underlighting, intermittent orange-white flashes, and deep hard-edged shadows. Dense volcanic ash and smoke reduce visibility, while drifting embers and sulfurous haze support a saturated palette of molten orange, crimson, charcoal black, and sulfurous gold.">Volcanic Eruption</button>
      <button type="button" data-prompt="The lighting is bright and neutral daylight, with strong directional key light casting defined medium-contrast shadows. The atmosphere is clear air with little to no haze; dominant palette is saturated greens, warm amber tones, and cool blues, with clean stylized surfaces; no distinct weather is visible.">Clear Daylight</button>
      <button type="button" data-prompt="The render is in deep night, with very dark cool-blue ambient illumination, sparse silver moonlight, and crisp directional rim highlights creating deep, high-contrast shadows. The air is mostly clear with slight nocturnal haze; the palette favors navy blue, cyan-gray, near-black, and a few restrained warm highlights.">Deep Night</button>
      <button type="button" data-prompt="The render has a dark cyberpunk night grade, illuminated by saturated magenta and electric-cyan neon light, sharp colored rim highlights, glossy specular reflections, and deep high-contrast shadows. Light atmospheric haze catches the neon glow; the palette is dominated by hot pink, violet, cyan, cobalt blue, and near-black.">Cyberpunk</button>
      <button type="button" id="customPromptButton" class="custom-prompt-button" aria-pressed="false">Custom</button>
    </div>
    <form class="prompt-form" id="stylePromptForm">
      <textarea id="stylePrompt" rows="2" maxlength="1200" placeholder="Describe a visual style..." aria-label="Visual style prompt"></textarea>
      <button class="execute-button" id="executePromptButton" type="submit">Apply style</button>
    </form>
  </section>

  <details class="advanced-panel" id="modelInputsPanel">
    <summary>Model inputs</summary>
    <section class="buffer-section">
      <div class="section-title">
        <h2>G-buffer</h2>
        <span id="bufferStatus">Waiting for G-buffer preview</span>
      </div>
      <div class="gbuffer-single">
        <div class="gbuffer-toolbar">
          <div class="gbuffer-toolbar-head"><span id="gbufferTitle">Albedo</span></div>
          <div class="title-controls">
            <select id="gbufferSelect" class="gbuffer-select" aria-label="Select G-buffer">
              <option value="albedo">Albedo</option>
              <option value="normal">Normal</option>
              <option value="depth">Depth</option>
              <option value="metallic">Metallic</option>
              <option value="roughness">Roughness</option>
            </select>
            <select id="gbufferRateSelect" class="gbuffer-rate-select" aria-label="G-buffer frame rate">
              <option value="auto" selected>Auto FPS</option>
              <option value="8">8 FPS</option>
              <option value="4">4 FPS</option>
              <option value="2">2 FPS</option>
              <option value="0">Pause</option>
            </select>
            <select id="gbufferSizeSelect" class="gbuffer-size-select" aria-label="G-buffer size">
              <option value="auto" selected>Auto size</option>
              <option value="240">240p</option>
              <option value="360">360p</option>
              <option value="480">480p</option>
            </select>
          </div>
        </div>
        <div class="gbuffer-preview">
          <canvas id="gbufferCanvas" hidden aria-label="Selected G-buffer preview"></canvas>
          <img id="gbufferImage" alt="Selected G-buffer preview">
        </div>
      </div>
    </section>
  </details>

  <footer class="license-card" aria-label="Credits and licenses">
    <span>AlayaRenderer Flash · Alaya Lab</span>
    <span>Powered by SuperTuxKart · <a href="https://supertuxkart.net/Licensing" target="_blank" rel="noreferrer">Licensing</a> · <a href="https://github.com/supertuxkart/stk-code/blob/master/COPYING" target="_blank" rel="noreferrer">GPLv3+</a></span>
  </footer>

  <div class="internal-state" hidden aria-hidden="true">
    <span id="frameMetric">--</span><span id="captureStatus">--</span><span id="modeValue">--</span>
    <span id="captureEveryValue">--</span><span id="rawSizeValue">--</span><span id="videoProfileValue">--</span>
    <span id="playbackValue">--</span><span id="networkValue">--</span><span id="recordStatusValue">--</span>
    <span id="recordFramesValue">--</span><span id="recordDirValue">--</span><span id="recordRunValue">--</span>
    <span id="recordDetailValue">--</span><span id="captureDirValue">--</span><span id="renderedDirValue">--</span>
    <div id="bufferFiles"></div><span id="renderStatus">Live sync</span>
    <input id="recordDurationInput" type="number" value="7200"><input id="recordFpsInput" type="number" value="8">
    <button id="recordStartButton" type="button">Start</button><button id="recordStopButton" type="button">Stop</button>
    <select id="renderTransportSelect"><option value="websocket" selected>Live</option><option value="h264">H.264</option></select>
    <select id="videoQualitySelect"><option value="auto" selected>Auto</option><option value="fast">Fast</option><option value="balanced">Balanced</option><option value="quality">Quality</option></select>
  </div>
</main>
<script>
const frameStatus = document.getElementById('frameStatus');
const renderStatus = document.getElementById('renderStatus');
const topStatus = document.getElementById('topStatus');
const inputStatus = document.getElementById('inputStatus');
const controlReadout = document.getElementById('controlReadout');
const controlNodes = Array.from(document.querySelectorAll('[data-control]'));
const autoModeButton = document.getElementById('autoModeButton');
const manualModeButton = document.getElementById('manualModeButton');
const liveStatus = document.getElementById('liveStatus');
const resolutionMetric = document.getElementById('resolutionMetric');
const frameMetric = document.getElementById('frameMetric');
const fpsMetric = document.getElementById('fpsMetric');
const ageMetric = document.getElementById('ageMetric');
const captureStatus = document.getElementById('captureStatus');
const trackSelect = document.getElementById('trackSelect');
const raceLabel = document.getElementById('raceLabel');
const modeValue = document.getElementById('modeValue');
const captureEveryValue = document.getElementById('captureEveryValue');
const rawSizeValue = document.getElementById('rawSizeValue');
const videoProfileValue = document.getElementById('videoProfileValue');
const playbackValue = document.getElementById('playbackValue');
const networkValue = document.getElementById('networkValue');
const recordStatusValue = document.getElementById('recordStatusValue');
const recordFramesValue = document.getElementById('recordFramesValue');
const recordDirValue = document.getElementById('recordDirValue');
const recordDurationInput = document.getElementById('recordDurationInput');
const recordFpsInput = document.getElementById('recordFpsInput');
const recordStartButton = document.getElementById('recordStartButton');
const recordStopButton = document.getElementById('recordStopButton');
const recordRunValue = document.getElementById('recordRunValue');
const recordDetailValue = document.getElementById('recordDetailValue');
const captureDirValue = document.getElementById('captureDirValue');
const renderedDirValue = document.getElementById('renderedDirValue');
const bufferFiles = document.getElementById('bufferFiles');
const bufferStatus = document.getElementById('bufferStatus');
const gbufferTitle = document.getElementById('gbufferTitle');
const gbufferSelect = document.getElementById('gbufferSelect');
const gbufferRateSelect = document.getElementById('gbufferRateSelect');
const gbufferSizeSelect = document.getElementById('gbufferSizeSelect');
const gbufferImage = document.getElementById('gbufferImage');
const gbufferCanvas = document.getElementById('gbufferCanvas');
const modelInputsPanel = document.getElementById('modelInputsPanel');
const modelInputsPauseNotice = document.getElementById('modelInputsPauseNotice');
const modelInputsResumeButton = document.getElementById('modelInputsResumeButton');
const rendererWarmup = document.getElementById('rendererWarmup');
const rendererWarmupStage = document.getElementById('rendererWarmupStage');
const rendererWarmupDetail = document.getElementById('rendererWarmupDetail');
const rendererWarmupFill = document.getElementById('rendererWarmupFill');
const renderedStage = document.getElementById('renderedStage');
const renderedImage = document.getElementById('renderedImage');
const renderedVideo = document.getElementById('renderedVideo');
const sourceImage = document.getElementById('sourceImage');
const renderTransportSelect = document.getElementById('renderTransportSelect');
const videoQualitySelect = document.getElementById('videoQualitySelect');
const stylePromptForm = document.getElementById('stylePromptForm');
const stylePrompt = document.getElementById('stylePrompt');
const executePromptButton = document.getElementById('executePromptButton');
const promptStatus = document.getElementById('promptStatus');
const promptPresetButtons = Array.from(document.querySelectorAll('[data-prompt]'));
const customPromptButton = document.getElementById('customPromptButton');
const active = new Set();
const manualInputFeatureEnabled = false;
const interesting = new Set(['KeyW','KeyA','KeyS','KeyD','ArrowUp','ArrowDown','ArrowLeft','ArrowRight']);
const inputClientId = (window.crypto && typeof window.crypto.randomUUID === 'function')
  ? window.crypto.randomUUID()
  : `tab-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
const gbufferLabels = {
  albedo: 'Albedo',
  normal: 'Normal',
  depth: 'Depth',
  metallic: 'Metallic',
  roughness: 'Roughness'
};
const gbufferDetails = {
  albedo: 'RGB',
  normal: 'XYZ',
  depth: 'F32',
  metallic: 'A · enhanced',
  roughness: 'B · auto'
};
const videoProfiles = {
  fast: { label: 'Fast', width: 640, fps: 24, kbps: 1800 },
  balanced: { label: 'Balanced', width: 960, fps: __VIDEO_PROFILE_FPS__, kbps: 4200 },
  quality: { label: 'Quality', width: 1280, fps: __VIDEO_PROFILE_FPS__, kbps: 8000 }
};
const videoProfileOrder = ['fast', 'balanced', 'quality'];
let controlMode = 'auto';
let manualInputArmed = false;
let currentTrack = '';
let pendingTrack = '';
let latestControl = {};
let lastSeq = null;
let lastSeqAt = performance.now();
let previewFps = 0;
let workerOutputFps = 0;
let selectedGbuffer = 'albedo';
let gbufferRateMode = 'auto';
let effectiveGbufferFps = 4;
let lastGbufferRateUpdateAt = performance.now();
let gbufferSizeMode = 'auto';
let effectiveGbufferWidth = 360;
let lastGbufferSizeUpdateAt = performance.now();
let gbufferSerial = 0;
let gbufferPending = false;
let gbufferTimer = null;
let renderedSerial = 0;
let renderedPending = false;
let mainImageStreamGeneration = 0;
let sourceSerial = 0;
let sourcePending = false;
let sourceStreamWanted = false;
const mainViewSourceFallbackEnabled = false;
let sourceRequestGeneration = 0;
let sourceDecodeGeneration = 0;
let sourceDecoder = null;
let sourceDecoderUrl = null;
let sourceFrameGate = null;
let loadedSourceFrame = null;
const sourceFrameMetaBySeq = new Map();
const sourceStateAheadTolerance = 32;
let streamTransport = 'connecting';
let renderTransport = 'websocket';
let videoQualityMode = 'auto';
let videoQuality = 'balanced';
let streamSocket = null;
let streamSocketGeneration = 0;
let streamHelloSocketGeneration = 0;
let activeStreamConnectionGeneration = '';
let unfencedModelBinaryCount = 0;
let lastStreamClose = null;
let streamReconnectTimer = null;
let streamReconnectAttempt = 0;
let websocketRecoveryPending = false;
let lastStreamConfig = null;
const streamConfigHistory = [];
let renderedObjectUrl = null;
let renderedDecodeGeneration = 0;
const renderedDecoders = new Map();
const modelPresentationQueue = [];
const modelPresentationTargetDepth = 2;
const modelPresentationMaxDepth = 6;
const modelPresentationIntervalMs = 1000 / __MODEL_PRESENTATION_FPS__;
let modelPresentationTimer = null;
let modelPresentationNextAt = null;
let modelPresentationRunning = false;
let modelPresentationToken = 0;
let modelPresentationDropped = 0;
let modelPresentationPresented = 0;
let modelPresentationFastStartFrames = 0;
let modelPresentationBoundary = null;
let sourceObjectUrl = null;
let gbufferObjectUrl = null;
let streamOpened = false;
let renderedVideoStarted = false;
let gbufferDecodeBusy = false;
let queuedGbufferPayload = null;
let gbufferBitmapCtx = null;
let gbuffer2dCtx = null;
let gbufferWorker = null;
let gbufferWorkerReady = false;
let gbufferWorkerBusy = false;
let gbufferWorkerFailed = false;
let gbufferFrames = [];
let lastGbufferFrameAt = performance.now();
let lastGbufferPayloadAt = 0;
let gbufferDecodeStartedAt = 0;
let lastGbufferWatchdogAt = 0;
let lastGbufferSeq = null;
let lastRenderedJpegSeq = null;
let lastSourceJpegSeq = null;
let sourceLoadedCount = 0;
let lastSourceLoadedAt = 0;
let arrivedModelFrameSeq = null;
let arrivedModelFrameIdentity = null;
let loadedModelFrame = null;
let lastInputSentAt = 0;
let lastVideoQualitySample = null;
let currentPlaybackDelay = null;
let currentDropRatio = 0;
let autoQualityCooldownUntil = 0;
let autoQualityStableSince = performance.now();
let latestPageState = null;
let lastDriveSignature = 'auto:0:0:0';
let sourceReleaseTimer = null;
let modelFrameGateOpen = false;
let modelFrameGateSeq = null;
let modelFrameGateSourceSeq = null;
let modelFrameGateIdentity = null;
let modelFrameGateInputEventId = '';
let modelFrameGateVideoToken = null;
let localHandover = null;
let localInputAuthority = null;
let hardResetHandover = null;
let hardResetGeneration = 0;
let styleTransitionActive = false;
let styleTransitionPreviousIdentity = null;
let handoverInputSeqWatermark = -1;
let lastSourceIdentity = null;
let lastSourceSeq = null;
let lastRenderedIdentity = null;
let lastRenderedStateSeq = null;
let holdModelLoadCommit = false;
let heldModelLoad = null;
let pendingDecodedModelLoad = null;
const modelFrameMetaBySeq = new Map();
let h264EpochRequested = null;
let h264IdentityRequested = null;
let h264VideoTokenCounter = 0;
let h264ActiveVideoToken = null;
let h264FirstFrameVideoToken = null;
let h264VideoFrameCallbackId = null;
let h264RequestedVideoUrl = '';
let nextInputSeq = 0;
let latestInputIssuedSeq = -1;
let latestTransitionSeq = -1;
let lastInputAckSeq = -1;
let lastInputAck = null;
let serverInstanceId = '';
const retiredServerInstanceIds = new Set();
const suppressedInputKeys = new Set();
let lastServerInstanceObservation = null;
let lastInputAckTransport = '';
let serverInputSuspended = false;
let pendingInputTransition = null;
let inputRetryTimer = null;
let inputCoalesceTimer = null;
let lastInputTransitionAt = 0;
const inputCoalesceWindowMs = 90;
let ownershipReleaseSent = false;
const modelInputsIdleMs = 15000;
let modelInputsIdleTimer = null;
let modelInputsIdleDeadline = null;
let modelInputsStateApplied = Boolean(modelInputsPanel && modelInputsPanel.open);
if (gbufferCanvas) {
  gbufferBitmapCtx = gbufferCanvas.getContext('bitmaprenderer');
  if (!gbufferBitmapCtx) {
    gbuffer2dCtx = gbufferCanvas.getContext('2d', { alpha: false });
  }
}

function initGbufferWorker() {
  if (!gbufferCanvas || !('Worker' in window) || !('Blob' in window) || !('createImageBitmap' in window)) {
    return;
  }
  const workerSource = `
let ready = typeof createImageBitmap === 'function';
self.postMessage({ type: ready ? 'ready' : 'unsupported' });
self.onmessage = async (event) => {
  const data = event.data || {};
  if (data.type !== 'decode' || !ready) return;
  const payload = data.payload;
  try {
    const bitmap = await createImageBitmap(new Blob([payload], { type: 'image/jpeg' }));
    self.postMessage({
      type: 'bitmap',
      bitmap,
      width: bitmap.width,
      height: bitmap.height,
      bytes: payload.byteLength || 0
    }, [bitmap]);
  } catch (err) {
    self.postMessage({ type: 'error', message: err && err.message ? err.message : 'decode failed' });
  }
};`;
  try {
    const url = URL.createObjectURL(new Blob([workerSource], { type: 'text/javascript' }));
    const worker = new Worker(url);
    URL.revokeObjectURL(url);
    worker.onmessage = (event) => {
      const data = event.data || {};
      if (data.type === 'ready') {
        gbufferWorkerReady = true;
        return;
      }
      if (data.type === 'unsupported' || data.type === 'error') {
        gbufferWorkerReady = false;
        gbufferWorkerFailed = true;
        gbufferWorkerBusy = false;
        gbufferDecodeStartedAt = 0;
        return;
      }
      if (data.type === 'bitmap' && data.bitmap) {
        gbufferWorkerBusy = false;
        gbufferDecodeStartedAt = 0;
        if (renderGbufferBitmap(data.bitmap)) noteGbufferFrame(data.bytes || 0);
        const next = queuedGbufferPayload;
        queuedGbufferPayload = null;
        if (next) sendGbufferPayloadToWorker(next);
      }
    };
    worker.onerror = () => {
      gbufferWorkerReady = false;
      gbufferWorkerFailed = true;
      gbufferWorkerBusy = false;
      gbufferDecodeStartedAt = 0;
    };
    gbufferWorker = worker;
  } catch (_) {
    gbufferWorker = null;
    gbufferWorkerFailed = true;
  }
}

initGbufferWorker();

function normalizeKey(code) {
  return code;
}

function isFormTarget(target) {
  if (!target || !target.tagName) return false;
  return ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(target.tagName);
}

function focusGameStage() {
  if (renderedStage && renderedStage.focus) {
    renderedStage.focus({ preventScroll: true });
  }
}

function driveState(mode, keys) {
  const keySet = keys instanceof Set ? keys : new Set(keys || []);
  if (mode !== 'manual') {
    return { mode: 'auto', steer: 0, accel: false, brake: false };
  }
  const left = keySet.has('ArrowLeft') || keySet.has('KeyA');
  const right = keySet.has('ArrowRight') || keySet.has('KeyD');
  const accel = keySet.has('ArrowUp') || keySet.has('KeyW');
  const brake = keySet.has('ArrowDown') || keySet.has('KeyS');
  const steer = left === right ? 0 : (left ? -1 : 1);
  return { mode: 'manual', steer, accel, brake };
}

function driveSignature() {
  const state = driveState(controlMode, active);
  return `${state.mode}:${state.steer}:${state.accel ? 1 : 0}:${state.brake ? 1 : 0}`;
}

function renderedControlMatches(expected, control) {
  if (!expected) return false;
  const renderedMode = String(control && control.mode || '').toLowerCase();
  if (expected.mode !== 'manual') return renderedMode === 'auto';
  if (renderedMode && renderedMode !== 'manual') return false;
  const steer = Number(control && control.steer) || 0;
  const accel = Number(control && control.accel) || 0;
  const brake = Boolean(control && control.brake);
  const steerMatches = expected.steer < 0
    ? steer <= -0.5
    : (expected.steer > 0 ? steer >= 0.5 : Math.abs(steer) < 0.25);
  return steerMatches
    && (expected.accel ? accel >= 0.5 : accel < 0.5)
    && brake === expected.brake;
}

function identityNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? number : null;
}

function normalizeRenderIdentity(value = {}) {
  return {
    trackId: String(value.trackId ?? value.track_id ?? ''),
    sourceSessionId: String(value.sourceSessionId ?? value.source_session_id ?? ''),
    resetEpoch: identityNumber(value.resetEpoch ?? value.reset_epoch),
    sourceResetEpoch: identityNumber(value.sourceResetEpoch ?? value.source_reset_epoch)
  };
}

function renderIdentityFromState(state = latestPageState) {
  const rendered = state && state.rendered ? state.rendered : {};
  if (!rendered.external) return null;
  return normalizeRenderIdentity(rendered);
}

function sourceIdentityFromState(state = latestPageState) {
  const cfg = state && state.run_config ? state.run_config : {};
  return normalizeRenderIdentity({
    track_id: cfg.TRACK || '',
    source_session_id: cfg.SOURCE_SESSION_ID || ''
  });
}

function renderIdentityFromAck(ack = {}) {
  return normalizeRenderIdentity(ack);
}

function sameRenderIdentity(left, right) {
  if (!left || !right) return false;
  const a = normalizeRenderIdentity(left);
  const b = normalizeRenderIdentity(right);
  return a.trackId === b.trackId
    && a.sourceSessionId === b.sourceSessionId
    && a.resetEpoch === b.resetEpoch
    && a.sourceResetEpoch === b.sourceResetEpoch;
}

function renderIdentityMatches(expected, actual) {
  if (!expected || !actual) return false;
  const a = normalizeRenderIdentity(expected);
  const b = normalizeRenderIdentity(actual);
  return (!a.trackId || a.trackId === b.trackId)
    && (!a.sourceSessionId || a.sourceSessionId === b.sourceSessionId)
    && (a.resetEpoch === null || a.resetEpoch === b.resetEpoch)
    && (a.sourceResetEpoch === null || a.sourceResetEpoch === b.sourceResetEpoch);
}

function sameSourceIdentity(left, right) {
  if (!left || !right) return false;
  const a = normalizeRenderIdentity(left);
  const b = normalizeRenderIdentity(right);
  return a.trackId === b.trackId && a.sourceSessionId === b.sourceSessionId;
}

function sourceIdentityMatches(expected, actual) {
  if (!expected || !actual) return false;
  const a = normalizeRenderIdentity(expected);
  const b = normalizeRenderIdentity(actual);
  return (!a.trackId || a.trackId === b.trackId)
    && (!a.sourceSessionId || a.sourceSessionId === b.sourceSessionId);
}

function completeRenderIdentity(identity) {
  if (!identity) return false;
  const value = normalizeRenderIdentity(identity);
  return Boolean(value.trackId && value.sourceSessionId)
    && value.resetEpoch !== null
    && value.sourceResetEpoch !== null;
}

function clearModelGate() {
  modelFrameGateOpen = false;
  modelFrameGateSeq = null;
  modelFrameGateSourceSeq = null;
  modelFrameGateIdentity = null;
  modelFrameGateInputEventId = '';
  modelFrameGateVideoToken = null;
}

function releaseSupersededModelPresentationBoundary(inputEventId) {
  const authoritativeInputEventId = String(inputEventId || '');
  if (!modelPresentationBoundary
      || !authoritativeInputEventId
      || authoritativeInputEventId === modelPresentationBoundary.inputEventId) {
    return false;
  }
  modelPresentationBoundary = null;
  return true;
}

function invalidateSourceFrameFence() {
  sourceRequestGeneration += 1;
  sourceDecodeGeneration += 1;
  sourcePending = false;
  sourceFrameGate = null;
  loadedSourceFrame = null;
  sourceFrameMetaBySeq.clear();
  if (sourceDecoder) {
    sourceDecoder.onload = null;
    sourceDecoder.onerror = null;
    sourceDecoder.src = '';
    sourceDecoder = null;
  }
  if (sourceDecoderUrl) {
    URL.revokeObjectURL(sourceDecoderUrl);
    sourceDecoderUrl = null;
  }
}

function sourceFenceIdentity(value = null) {
  const identity = value ? normalizeRenderIdentity(value) : sourceIdentityFromState();
  return identity && (identity.trackId || identity.sourceSessionId)
    ? identity
    : null;
}

function armSourceFrameFence(reason, expectedIdentity = null, minSeq = null) {
  invalidateSourceFrameFence();
  const seq = Number(minSeq);
  const hasMinSeq = minSeq !== null
    && minSeq !== undefined
    && Number.isFinite(seq);
  sourceFrameGate = {
    generation: sourceRequestGeneration,
    reason: reason || 'source-fallback',
    expectedIdentity: sourceFenceIdentity(expectedIdentity),
    minSeq: hasMinSeq ? seq : null
  };
  renderedStage.classList.remove('source-active');
}

function sourceFrameMetaKey(socketGeneration, connectionGeneration, token, seq) {
  return `${socketGeneration}:${connectionGeneration}:${token}:${seq}`;
}

function rememberSourceFrameMeta(message, socketGeneration) {
  if (!sourceImageStreamSubscribed()) return;
  const seq = Number(message && message.seq);
  const token = Number(message && message.frame_token);
  const connectionGeneration = String(
    message && message.connection_generation || ''
  );
  const identity = normalizeRenderIdentity(
    message && message.identity ? message.identity : {}
  );
  if (!Number.isFinite(seq)
      || !Number.isInteger(token)
      || token <= 0
      || socketGeneration !== streamSocketGeneration
      || !connectionGeneration
      || connectionGeneration !== activeStreamConnectionGeneration
      || !identity.trackId
      || !identity.sourceSessionId) return;
  sourceFrameMetaBySeq.set(
    sourceFrameMetaKey(socketGeneration, connectionGeneration, token, seq),
    { seq, token, socketGeneration, connectionGeneration, identity }
  );
  while (sourceFrameMetaBySeq.size > 8) {
    sourceFrameMetaBySeq.delete(sourceFrameMetaBySeq.keys().next().value);
  }
}

function sourceFramePassesFence(frameMeta, requestGeneration) {
  if (!sourceImageStreamSubscribed()
      || requestGeneration !== sourceRequestGeneration) return false;
  const gate = sourceFrameGate;
  const seq = Number(frameMeta && frameMeta.seq);
  const identity = sourceFenceIdentity(frameMeta && frameMeta.identity);
  if (!gate || gate.generation !== requestGeneration || !Number.isFinite(seq)) return false;
  if (gate.minSeq !== null && seq < gate.minSeq) return false;
  if (gate.expectedIdentity
      && (!identity || !sourceIdentityMatches(gate.expectedIdentity, identity))) return false;
  if (frameMeta && frameMeta.transport === 'websocket') {
    if (!frameMeta.exactMeta
        || Number(frameMeta.socketGeneration) !== streamSocketGeneration
        || String(frameMeta.connectionGeneration || '')
          !== activeStreamConnectionGeneration
        || !Number.isInteger(Number(frameMeta.frameToken))
        || Number(frameMeta.frameToken) <= 0) return false;
  } else if (frameMeta && frameMeta.transport === 'websocket-legacy') {
    const stateSeq = Number(latestPageState && latestPageState.seq);
    const stateIdentity = sourceIdentityFromState();
    if (!Number.isFinite(stateSeq)
        || !identity
        || !sameSourceIdentity(identity, stateIdentity)
        || seq > stateSeq + sourceStateAheadTolerance) return false;
  }
  if (loadedSourceFrame
      && loadedSourceFrame.generation === requestGeneration
      && seq <= loadedSourceFrame.seq) return false;
  return true;
}

function commitSourceFrameLoaded(frameMeta, requestGeneration) {
  if (!sourceFramePassesFence(frameMeta, requestGeneration)) return false;
  const identity = sourceFenceIdentity(frameMeta && frameMeta.identity);
  const seq = Number(frameMeta.seq);
  loadedSourceFrame = { generation: requestGeneration, seq, identity };
  lastSourceJpegSeq = seq;
  sourceLoadedCount += 1;
  lastSourceLoadedAt = performance.now();
  renderedStage.classList.add('source-active');
  return true;
}

function revokeDecodedModelFrame(frame) {
  if (frame && frame.url && frame.url !== renderedObjectUrl) {
    URL.revokeObjectURL(frame.url);
  }
}

function clearModelPresentationPipeline(preserveBoundary = false) {
  renderedDecodeGeneration += 1;
  modelPresentationToken += 1;
  if (modelPresentationTimer !== null) clearTimeout(modelPresentationTimer);
  modelPresentationTimer = null;
  modelPresentationNextAt = null;
  modelPresentationRunning = false;
  if (!preserveBoundary) {
    modelPresentationFastStartFrames = 0;
    modelPresentationBoundary = null;
  }
  renderedImage.onload = null;
  renderedImage.onerror = null;
  for (const [url, decoder] of renderedDecoders) {
    decoder.onload = null;
    decoder.onerror = null;
    decoder.src = '';
    URL.revokeObjectURL(url);
  }
  renderedDecoders.clear();
  while (modelPresentationQueue.length) {
    revokeDecodedModelFrame(modelPresentationQueue.shift());
  }
}

function advanceModelPresentationDeadline(deadline, now) {
  if (!Number.isFinite(deadline)) return now + modelPresentationIntervalMs;
  let next = deadline + modelPresentationIntervalMs;
  if (next <= now) {
    const missed = Math.floor((now - next) / modelPresentationIntervalMs) + 1;
    next += missed * modelPresentationIntervalMs;
  }
  return next;
}

function presentDecodedModelFrame(frame) {
  if (!modelImageStreamSubscribed()
      || !frame
      || frame.generation !== renderedDecodeGeneration) {
    revokeDecodedModelFrame(frame);
    return false;
  }
  const previous = renderedObjectUrl;
  const presentationToken = ++modelPresentationToken;
  renderedObjectUrl = frame.url;
  renderedImage.onload = () => {
    const displayedUrl = renderedImage.currentSrc || renderedImage.src;
    commitRenderedLoadToken(
      frame.generation,
      frame.url,
      displayedUrl,
      frame.frameMeta,
      presentationToken
    );
  };
  renderedImage.onerror = () => {
    if (presentationToken !== modelPresentationToken) return;
    if (renderedObjectUrl === frame.url) renderedObjectUrl = previous;
    URL.revokeObjectURL(frame.url);
  };
  renderedImage.src = frame.url;
  modelPresentationPresented += 1;
  if (modelPresentationFastStartFrames > 0) {
    modelPresentationFastStartFrames -= 1;
  }
  if (previous && previous !== frame.url) {
    setTimeout(() => URL.revokeObjectURL(previous), 1000);
  }
  return true;
}

function runModelPresentation() {
  modelPresentationTimer = null;
  if (!modelImageStreamSubscribed()) {
    clearModelPresentationPipeline();
    return;
  }
  if (!modelPresentationRunning) return;
  const frame = modelPresentationQueue.shift();
  if (!frame) {
    modelPresentationRunning = false;
    modelPresentationNextAt = null;
    return;
  }
  presentDecodedModelFrame(frame);
  const now = performance.now();
  modelPresentationNextAt = advanceModelPresentationDeadline(
    modelPresentationNextAt,
    now
  );
  modelPresentationTimer = setTimeout(
    runModelPresentation,
    Math.max(0, modelPresentationNextAt - performance.now())
  );
}

function startModelPresentationIfReady() {
  const targetDepth = modelPresentationFastStartFrames > 0
    ? 1
    : modelPresentationTargetDepth;
  if (modelPresentationRunning
      || !modelImageStreamSubscribed()
      || modelPresentationQueue.length < targetDepth) return;
  modelPresentationRunning = true;
  modelPresentationNextAt = performance.now();
  modelPresentationTimer = setTimeout(runModelPresentation, 0);
}

function enqueueDecodedModelFrame(frame) {
  if (!modelImageStreamSubscribed()
      || !frame
      || frame.generation !== renderedDecodeGeneration
      || !frame.frameMeta) {
    revokeDecodedModelFrame(frame);
    return false;
  }
  const identity = normalizeRenderIdentity(frame.frameMeta.identity || {});
  const seq = Number(frame.frameMeta.seq);
  const sourceSeq = Number(frame.frameMeta.sourceSeq);
  const inputEventId = String(frame.frameMeta.inputEventId || '');
  if (!Number.isFinite(seq) || !completeRenderIdentity(identity)) {
    revokeDecodedModelFrame(frame);
    return false;
  }
  if (modelPresentationBoundary
      && (!sameRenderIdentity(modelPresentationBoundary.identity, identity)
        || !Number.isFinite(sourceSeq)
        || sourceSeq < modelPresentationBoundary.minSourceSeq
        || !inputEventId
        || inputEventId !== modelPresentationBoundary.inputEventId)) {
    revokeDecodedModelFrame(frame);
    modelPresentationDropped += 1;
    return false;
  }
  if (loadedModelFrame
      && sameRenderIdentity(loadedModelFrame.identity, identity)
      && seq <= loadedModelFrame.seq) {
    revokeDecodedModelFrame(frame);
    return false;
  }
  if (modelPresentationQueue.some(item => Number(item.frameMeta.seq) === seq)) {
    revokeDecodedModelFrame(frame);
    return false;
  }
  if (modelPresentationQueue.length
      && !sameRenderIdentity(
        modelPresentationQueue[0].frameMeta.identity,
        identity
      )) {
    revokeDecodedModelFrame(frame);
    return false;
  }
  modelPresentationQueue.push(frame);
  modelPresentationQueue.sort(
    (left, right) => Number(left.frameMeta.seq) - Number(right.frameMeta.seq)
  );
  const replacementPrimed = primeSoftHandoverReplacement(
    sourceSeq,
    identity,
    inputEventId
  );
  if (modelPresentationQueue.length > modelPresentationMaxDepth) {
    while (modelPresentationQueue.length > modelPresentationTargetDepth) {
      revokeDecodedModelFrame(modelPresentationQueue.shift());
      modelPresentationDropped += 1;
    }
  }
  if (!replacementPrimed) startModelPresentationIfReady();
  return true;
}

function primeModelPresentationBoundary(minSourceSeq, identity, inputEventId) {
  const boundary = Number(minSourceSeq);
  const normalizedIdentity = normalizeRenderIdentity(identity || {});
  const normalizedInputEventId = String(inputEventId || '');
  if (!Number.isFinite(boundary)
      || !completeRenderIdentity(normalizedIdentity)
      || !normalizedInputEventId) return false;
  modelPresentationBoundary = {
    minSourceSeq: boundary,
    identity: normalizedIdentity,
    inputEventId: normalizedInputEventId
  };
  for (let index = modelPresentationQueue.length - 1; index >= 0; index -= 1) {
    const frame = modelPresentationQueue[index];
    const frameIdentity = normalizeRenderIdentity(frame.frameMeta.identity || {});
    const frameSourceSeq = Number(frame.frameMeta.sourceSeq);
    const frameInputEventId = String(frame.frameMeta.inputEventId || '');
    if (sameRenderIdentity(normalizedIdentity, frameIdentity)
        && Number.isFinite(frameSourceSeq)
        && frameSourceSeq >= boundary
        && frameInputEventId === normalizedInputEventId) continue;
    modelPresentationQueue.splice(index, 1);
    revokeDecodedModelFrame(frame);
    modelPresentationDropped += 1;
  }
  modelPresentationFastStartFrames = 3;
  startModelPresentationIfReady();
  return true;
}

function primeSoftHandoverReplacement(sourceSeq, identity, inputEventId) {
  if (!localHandover || hardResetHandover || localHandover.optimistic
      || localHandover.presentationPrimed) return false;
  const boundary = Number(localHandover.boundarySourceSeq);
  const replacementSourceSeq = Number(sourceSeq);
  const replacementIdentity = normalizeRenderIdentity(identity || {});
  const replacementInputEventId = String(inputEventId || '');
  if (!Number.isFinite(boundary)
      || !Number.isFinite(replacementSourceSeq)
      || replacementSourceSeq < boundary
      || !completeRenderIdentity(replacementIdentity)
      || !renderIdentityMatches(localHandover.renderIdentity, replacementIdentity)
      || !replacementInputEventId
      || replacementInputEventId !== localHandover.inputEventId) {
    return false;
  }
  if (!primeModelPresentationBoundary(
    boundary,
    replacementIdentity,
    replacementInputEventId
  )) return false;
  localHandover.presentationPrimed = true;
  localHandover.replacementSourceSeq = replacementSourceSeq;
  return true;
}

function invalidatePendingModelFrames(clearArrival = true) {
  clearModelPresentationPipeline();
  if (localHandover) localHandover.presentationPrimed = false;
  heldModelLoad = null;
  pendingDecodedModelLoad = null;
  modelFrameMetaBySeq.clear();
  if (clearArrival) {
    arrivedModelFrameSeq = null;
    arrivedModelFrameIdentity = null;
  }
}

function invalidateLoadedModelFence(clearArrival = true) {
  invalidatePendingModelFrames(clearArrival);
  loadedModelFrame = null;
}

function beginHardReset(reason, expectedSourceIdentity = null) {
  styleTransitionActive = false;
  styleTransitionPreviousIdentity = null;
  closeModelInputsPanel('session-reset');
  localHandover = null;
  clearModelGate();
  invalidatePendingModelFrames(true);
  const inputSeqFloor = Math.max(
    handoverInputSeqWatermark,
    Number.isFinite(Number(latestInputIssuedSeq)) ? Number(latestInputIssuedSeq) : -1
  );
  handoverInputSeqWatermark = inputSeqFloor;
  hardResetHandover = {
    generation: ++hardResetGeneration,
    reason,
    expectedSourceIdentity: expectedSourceIdentity
      ? normalizeRenderIdentity(expectedSourceIdentity)
      : null,
    expectedDrive: driveState(controlMode, active),
    inputSeqFloor,
    inputSeqWatermark: inputSeqFloor,
    inputEventId: '',
    minSourceSeq: null,
    boundaryRenderIdentity: null,
    h264VideoToken: null,
    h264MinSourceSeq: null,
    startedAt: Date.now()
  };
  holdModelOutput(reason || 'source-reset');
}

function styleTransitionIdentityFromResponse(response = {}) {
  const previousIdentity = styleTransitionPreviousIdentity
    || (loadedModelFrame && loadedModelFrame.identity)
    || renderIdentityFromState()
    || lastRenderedIdentity;
  const resetEpoch = identityNumber(response.reset_epoch);
  if (!previousIdentity || resetEpoch === null) return null;
  const previous = normalizeRenderIdentity(previousIdentity);
  const sourceResetEpoch = identityNumber(response.source_reset_epoch);
  return normalizeRenderIdentity({
    ...previous,
    resetEpoch,
    sourceResetEpoch: sourceResetEpoch === null
      ? previous.sourceResetEpoch
      : sourceResetEpoch
  });
}

function beginStyleTransition(expectedRenderedIdentity = null) {
  const expected = expectedRenderedIdentity
    ? normalizeRenderIdentity(expectedRenderedIdentity)
    : null;
  if (!styleTransitionActive) {
    styleTransitionPreviousIdentity = normalizeRenderIdentity(
      (loadedModelFrame && loadedModelFrame.identity)
        || renderIdentityFromState()
        || lastRenderedIdentity
        || {}
    );
    clearModelGate();
    clearModelPresentationPipeline();
    if (renderTransport === 'h264' && renderedVideoStarted) {
      freezeRenderedVideoFrame();
      stopRenderedVideo();
    }
  }
  localHandover = null;
  hardResetHandover = null;
  modelFrameGateOpen = true;
  modelFrameGateIdentity = expected && completeRenderIdentity(expected)
    ? expected
    : null;
  modelFrameGateSeq = null;
  modelFrameGateSourceSeq = null;
  modelFrameGateInputEventId = '';
  styleTransitionActive = true;
  releaseSourceLayer();
  renderedStage.dataset.handover = 'style-transition';
  return true;
}

function cancelStyleTransition() {
  if (!styleTransitionActive) return false;
  styleTransitionActive = false;
  styleTransitionPreviousIdentity = null;
  clearModelGate();
  delete renderedStage.dataset.handover;
  if (streamTransport === 'polling') requestRenderedImage();
  else sendStreamConfig();
  return true;
}

function observeStateIdentity(state) {
  const sourceIdentity = sourceIdentityFromState(state);
  const sourceSeq = Number(state && state.seq);
  if (lastSourceIdentity && !sameSourceIdentity(lastSourceIdentity, sourceIdentity)) {
    beginHardReset('source-identity-change', sourceIdentity);
  } else if (lastSourceIdentity
      && Number.isFinite(sourceSeq)
      && Number.isFinite(lastSourceSeq)
      && sourceSeq < lastSourceSeq) {
    beginHardReset('source-seq-rollback', sourceIdentity);
  }
  lastSourceIdentity = sourceIdentity;
  if (Number.isFinite(sourceSeq)) lastSourceSeq = sourceSeq;

  const rendered = state && state.rendered ? state.rendered : {};
  const renderedIdentity = renderIdentityFromState(state);
  const renderedSeq = Number(rendered.seq);
  if (!renderedIdentity) return;
  if (lastRenderedIdentity && !sameRenderIdentity(lastRenderedIdentity, renderedIdentity)) {
    if (sameSourceIdentity(lastRenderedIdentity, renderedIdentity)) {
      beginStyleTransition(renderedIdentity);
    } else {
      beginHardReset('render-identity-change', sourceIdentity);
    }
  } else if (lastRenderedIdentity
      && sameRenderIdentity(lastRenderedIdentity, renderedIdentity)
      && Number.isFinite(renderedSeq)
      && Number.isFinite(lastRenderedStateSeq)
      && renderedSeq < lastRenderedStateSeq) {
    beginHardReset('render-seq-rollback', sourceIdentity);
  }
  lastRenderedIdentity = renderedIdentity;
  if (Number.isFinite(renderedSeq)) lastRenderedStateSeq = renderedSeq;
}

function noteModelFrameArrival(seq, identity) {
  const frameSeq = Number(seq);
  const frameIdentity = identity ? normalizeRenderIdentity(identity) : null;
  if (!Number.isFinite(frameSeq) || !frameIdentity) return;
  if (arrivedModelFrameIdentity
      && sameRenderIdentity(arrivedModelFrameIdentity, frameIdentity)
      && arrivedModelFrameSeq !== null
      && frameSeq < arrivedModelFrameSeq) {
    beginHardReset('model-payload-seq-rollback', sourceIdentityFromState());
  } else if (arrivedModelFrameIdentity
      && !sameRenderIdentity(arrivedModelFrameIdentity, frameIdentity)) {
    invalidateLoadedModelFence(false);
  }
  arrivedModelFrameSeq = frameSeq;
  arrivedModelFrameIdentity = frameIdentity;
}

function modelFrameMetaKey(socketGeneration, connectionGeneration, token, seq) {
  return `${socketGeneration}:${connectionGeneration}:${token}:${seq}`;
}

function rememberModelFrameMeta(message, socketGeneration) {
  if (!modelImageStreamSubscribed()) return;
  const seq = Number(message && message.seq);
  const sourceSeq = message && message.source_seq !== null
    && message.source_seq !== undefined
    ? Number(message.source_seq)
    : Number.NaN;
  const token = Number(message && message.frame_token);
  const connectionGeneration = String(
    message && message.connection_generation || ''
  );
  const identity = normalizeRenderIdentity(message && message.identity ? message.identity : {});
  const inputEventId = String(message && message.input_event_id || '');
  if (!Number.isFinite(seq)
      || !Number.isInteger(token)
      || token <= 0
      || socketGeneration !== streamSocketGeneration
      || !connectionGeneration
      || connectionGeneration !== activeStreamConnectionGeneration
      || !Number.isFinite(sourceSeq)
      || !completeRenderIdentity(identity)) return;
  if (lastRenderedIdentity
      && sameRenderIdentity(lastRenderedIdentity, identity)
      && Number.isFinite(lastRenderedStateSeq)
      && seq < lastRenderedStateSeq) return;
  modelFrameMetaBySeq.set(
    modelFrameMetaKey(socketGeneration, connectionGeneration, token, seq),
    {
      seq,
      sourceSeq,
      inputEventId,
      token,
      socketGeneration,
      connectionGeneration,
      identity
    }
  );
  while (modelFrameMetaBySeq.size > 8) {
    modelFrameMetaBySeq.delete(modelFrameMetaBySeq.keys().next().value);
  }
}

function commitModelFrameLoaded(seq, identity, fence = {}) {
  if (!mainImageStreamSubscribed()) return;
  const frameSeq = Number(seq);
  const frameIdentity = identity ? normalizeRenderIdentity(identity) : null;
  if (streamTransport === 'polling' && websocketRecoveryPending) return;
  if (renderTransport !== 'h264' && streamTransport === 'websocket') {
    if (!fence.exactMeta
        || Number(fence.socketGeneration) !== streamSocketGeneration
        || String(fence.connectionGeneration || '')
          !== activeStreamConnectionGeneration
        || !Number.isInteger(Number(fence.frameToken))
        || Number(fence.frameToken) <= 0) return;
  }
  if (!Number.isFinite(frameSeq)
      || !frameIdentity
      || !completeRenderIdentity(frameIdentity)) return;
  if (loadedModelFrame
      && sameRenderIdentity(loadedModelFrame.identity, frameIdentity)
      && frameSeq < loadedModelFrame.seq) {
    beginHardReset('loaded-model-seq-rollback', sourceIdentityFromState());
    return;
  }
  loadedModelFrame = {
    seq: frameSeq,
    sourceSeq: fence.sourceSeq !== null
      && fence.sourceSeq !== undefined
      && Number.isFinite(Number(fence.sourceSeq))
      ? Number(fence.sourceSeq)
      : null,
    inputEventId: String(fence.inputEventId || ''),
    identity: frameIdentity,
    exactMeta: Boolean(fence.exactMeta),
    socketGeneration: Number(fence.socketGeneration) || null,
    connectionGeneration: String(fence.connectionGeneration || ''),
    frameToken: Number(fence.frameToken) || null
  };
  noteModelFrame();
}

function acceptDecodedModelLoad(candidate) {
  if (!candidate) return;
  if (completeRenderIdentity(candidate.identity)) {
    pendingDecodedModelLoad = null;
    commitModelFrameLoaded(candidate.seq, candidate.identity, candidate);
    return;
  }
  const rendered = latestPageState && latestPageState.rendered
    ? latestPageState.rendered
    : {};
  const stateSeq = Number(rendered.seq);
  const currentIdentity = renderIdentityFromState();
  if (!Number.isFinite(stateSeq) || !currentIdentity) {
    pendingDecodedModelLoad = candidate;
    return;
  }
  if (stateSeq === candidate.seq) {
    pendingDecodedModelLoad = null;
    commitModelFrameLoaded(candidate.seq, currentIdentity, candidate);
  } else if (stateSeq < candidate.seq) {
    pendingDecodedModelLoad = candidate;
  }
}

function markModelFrameLoaded(seq, identity, fence = {}) {
  const candidate = {
    ...fence,
    seq: Number(seq),
    identity: normalizeRenderIdentity(identity || {})
  };
  if (holdModelLoadCommit) {
    heldModelLoad = candidate;
    return;
  }
  acceptDecodedModelLoad(candidate);
}

function reconcileDecodedModelLoad(state) {
  const renderedIdentity = renderIdentityFromState(state);
  const renderedSeq = Number(state && state.rendered && state.rendered.seq);
  if (renderedIdentity
      && Number.isFinite(renderedSeq)
      && arrivedModelFrameSeq === renderedSeq) {
    arrivedModelFrameIdentity = renderedIdentity;
  }
  if (!pendingDecodedModelLoad || holdModelLoadCommit) return;
  if (Number.isFinite(renderedSeq) && renderedSeq > pendingDecodedModelLoad.seq) {
    pendingDecodedModelLoad = null;
    return;
  }
  if (Number.isFinite(renderedSeq) && renderedSeq === pendingDecodedModelLoad.seq) {
    const candidate = pendingDecodedModelLoad;
    pendingDecodedModelLoad = null;
    commitModelFrameLoaded(candidate.seq, renderedIdentity, candidate);
  }
}

function requestSourceLayer(reason = '', expectedIdentity = null, minSeq = null) {
  if (!mainViewSourceFallbackEnabled) {
    if (sourceStreamWanted || renderedStage.classList.contains('source-active')) {
      releaseSourceLayer();
    }
    return false;
  }
  if (styleTransitionActive) return false;
  if (sourceReleaseTimer !== null) {
    clearTimeout(sourceReleaseTimer);
    sourceReleaseTimer = null;
  }
  const hasMinSeq = minSeq !== null
    && minSeq !== undefined
    && Number.isFinite(Number(minSeq));
  if (!sourceStreamWanted || expectedIdentity || hasMinSeq) {
    const stateSeq = Number(latestPageState && latestPageState.seq);
    const desiredIdentity = sourceFenceIdentity(expectedIdentity);
    const desiredMinSeq = hasMinSeq
      ? Number(minSeq)
      : (expectedIdentity ? null : (Number.isFinite(stateSeq) ? stateSeq : null));
    const existingIdentity = sourceFrameGate && sourceFrameGate.expectedIdentity;
    const sameFence = Boolean(
      sourceStreamWanted
      && sourceFrameGate
      && sourceFrameGate.generation === sourceRequestGeneration
      && sourceFrameGate.minSeq === desiredMinSeq
      && ((!desiredIdentity && !existingIdentity)
        || (desiredIdentity
          && existingIdentity
          && sameSourceIdentity(desiredIdentity, existingIdentity)))
    );
    if (!sameFence) {
      armSourceFrameFence(reason, desiredIdentity, desiredMinSeq);
    }
  }
  sourceStreamWanted = true;
  if (reason) renderedStage.dataset.handover = reason;
  sendStreamConfig();
  requestSourceImage();
  return true;
}

function releaseSourceLayer() {
  const wasWanted = sourceStreamWanted;
  sourceStreamWanted = false;
  invalidateSourceFrameFence();
  if (wasWanted) sendStreamConfig();
  renderedStage.classList.remove('source-active');
  delete renderedStage.dataset.handover;
  if (sourceReleaseTimer !== null) clearTimeout(sourceReleaseTimer);
  sourceReleaseTimer = setTimeout(() => {
    sourceReleaseTimer = null;
  }, 320);
}

function holdModelOutput(reason = 'renderer-unavailable') {
  const holdingVideoFrame = Boolean(
    renderTransport === 'h264'
    && renderedVideoStarted
    && renderedVideo.readyState >= 2
    && renderedVideo.videoWidth > 0
    && renderedVideo.videoHeight > 0
  );
  const hasModelFrame = Boolean(loadedModelFrame) || holdingVideoFrame;
  const alreadyHolding = renderedStage.dataset.handover === reason;
  if (!alreadyHolding) {
    clearModelPresentationPipeline();
    if (holdingVideoFrame) freezeRenderedVideoFrame();
    if (renderTransport === 'h264' && renderedVideoStarted) stopRenderedVideo();
  }
  releaseSourceLayer();
  if (hasModelFrame) {
    renderedImage.hidden = false;
    renderedVideo.hidden = true;
    text(renderStatus, 'model output · holding last frame');
  } else {
    renderedImage.hidden = true;
    renderedVideo.hidden = true;
    if (rendererWarmup.hidden) {
      rendererWarmupStage.textContent = 'Preparing renderer';
      rendererWarmupDetail.textContent = 'Waiting for live output';
      rendererWarmupFill.style.width = '18%';
      const track = rendererWarmupFill.parentElement;
      if (track) track.setAttribute('aria-valuenow', '18');
      rendererWarmup.hidden = false;
    }
    text(renderStatus, 'model output · loading');
  }
  renderedStage.dataset.handover = reason;
  return hasModelFrame;
}

function beginLocalHandover(ack) {
  const boundarySourceSeq = Number(ack && ack.boundary_source_seq);
  const inputSeq = Number(ack && ack.input_seq);
  const inputEventId = String((ack && ack.input_event_id) || '');
  const inputServerTime = Number(ack && ack.server_time) || 0;
  if (!Number.isFinite(inputSeq)
      || !Number.isFinite(boundarySourceSeq)
      || boundarySourceSeq < 0
      || !inputEventId) return;
  if (inputSeq <= handoverInputSeqWatermark) return;
  if (localHandover && inputSeq < localHandover.inputSeq) return;
  const expectedDrive = driveState(ack.mode, ack.keys);
  const renderIdentity = renderIdentityFromAck(ack);
  if (hardResetHandover) {
    if (inputSeq <= Number(hardResetHandover.inputSeqFloor)
        || inputSeq <= Number(hardResetHandover.inputSeqWatermark)) return;
    handoverInputSeqWatermark = inputSeq;
    hardResetHandover.inputSeqWatermark = inputSeq;
    hardResetHandover.inputEventId = inputEventId;
    hardResetHandover.inputServerTime = inputServerTime;
    hardResetHandover.minSourceSeq = boundarySourceSeq;
    hardResetHandover.boundaryRenderIdentity = renderIdentity;
    hardResetHandover.expectedDrive = expectedDrive;
    hardResetHandover.h264VideoToken = null;
    hardResetHandover.h264MinSourceSeq = null;
    if (renderTransport !== 'h264' && completeRenderIdentity(renderIdentity)) {
      primeModelPresentationBoundary(
        boundarySourceSeq,
        renderIdentity,
        inputEventId
      );
    }
    clearModelGate();
    return;
  }
  handoverInputSeqWatermark = inputSeq;
  // The previous event fence remains useful between transitions, but must be
  // released once a newer accepted event can produce its replacement frames.
  modelPresentationBoundary = null;
  localHandover = {
    inputSeq,
    inputEventId,
    inputServerTime,
    boundarySourceSeq,
    expectedDrive,
    renderIdentity,
    presentationPrimed: false,
    replacementSourceSeq: null,
    h264VideoToken: null,
    h264MinSourceSeq: null
  };
  // The model gate prevents premature handover completion. The presentation
  // queue remains live until a decoded replacement reaches the boundary.
  clearModelGate();
}

function beginOptimisticHandover(inputSeq) {
  if (!Number.isFinite(Number(inputSeq))
      || Number(inputSeq) <= handoverInputSeqWatermark) return;
  if (hardResetHandover) {
    clearModelGate();
    return;
  }
  if (localHandover && inputSeq < localHandover.inputSeq) return;
  localHandover = {
    inputSeq,
    inputEventId: '',
    boundarySourceSeq: null,
    optimistic: true,
    presentationPrimed: false,
    replacementSourceSeq: null,
    h264VideoToken: null,
    h264MinSourceSeq: null
  };
  clearModelGate();
}

function completeModelHandover() {
  clearModelGate();
  localHandover = null;
  hardResetHandover = null;
  styleTransitionActive = false;
  styleTransitionPreviousIdentity = null;
  releaseSourceLayer();
  rendererWarmup.hidden = true;
}

function noteModelFrame() {
  if (!modelFrameGateOpen) return false;
  const currentRendered = latestPageState && latestPageState.rendered
    ? latestPageState.rendered
    : {};
  const currentInputEventId = String(currentRendered.input_event_id || '');
  if (modelFrameGateInputEventId
      && currentInputEventId !== modelFrameGateInputEventId) return false;
  const activeHandover = localHandover || hardResetHandover;
  if (activeHandover && activeHandover.inputEventId
      && currentInputEventId !== activeHandover.inputEventId) return false;
  if (renderTransport === 'h264') {
    const currentIdentity = renderIdentityFromState();
    if (!modelFrameGateIdentity
        || !currentIdentity
        || !sameRenderIdentity(modelFrameGateIdentity, currentIdentity)) return false;
    if (!Number.isInteger(modelFrameGateVideoToken)
        || h264ActiveVideoToken !== modelFrameGateVideoToken
        || h264FirstFrameVideoToken !== modelFrameGateVideoToken) return false;
    const currentSourceSeq = Number(currentRendered.source_seq);
    if (modelFrameGateSourceSeq !== null
        && (!Number.isFinite(currentSourceSeq)
          || currentSourceSeq < modelFrameGateSourceSeq)) return false;
    if (activeHandover
        && Number(activeHandover.h264VideoToken) !== modelFrameGateVideoToken) return false;
  } else {
    if (!loadedModelFrame
        || !modelFrameGateIdentity
        || !sameRenderIdentity(loadedModelFrame.identity, modelFrameGateIdentity)) return false;
    if (modelFrameGateSeq !== null && loadedModelFrame.seq < modelFrameGateSeq) return false;
    if (modelFrameGateSourceSeq !== null
        && (loadedModelFrame.sourceSeq === null
          || loadedModelFrame.sourceSeq < modelFrameGateSourceSeq)) return false;
    if (modelFrameGateInputEventId
        && loadedModelFrame.inputEventId !== modelFrameGateInputEventId) return false;
  }
  if (renderTransport === 'h264') {
    renderedImage.hidden = true;
    renderedVideo.hidden = false;
  }
  completeModelHandover();
  return true;
}

function text(node, value) {
  node.textContent = value;
}

function setPromptStatus(message, state = '') {
  const value = String(message ?? '');
  const liveMatch = state === 'live' ? value.match(/^Live(?:\\s*·\\s*|\\s+|$)/) : null;
  promptStatus.replaceChildren();
  if (liveMatch) {
    const liveLabel = document.createElement('span');
    liveLabel.className = 'prompt-live-label';
    liveLabel.textContent = 'Live';
    promptStatus.append(liveLabel);
    const detail = value.slice(liveMatch[0].length).trim();
    if (detail) promptStatus.append(document.createTextNode(` · ${detail}`));
  } else {
    text(promptStatus, value);
  }
  promptStatus.dataset.state = state;
}

function setRendererWarmup(data) {
  const status = String(data.status || 'idle');
  const ready = data.ready === true;
  const warmed = data.warmed === true;
  let stage = '';
  let detail = '';
  let progress = 0;

  if (status === 'loading') {
    stage = 'Loading renderer';
    detail = 'Loading model weights';
    progress = 22;
  } else if (status === 'warming' && !ready) {
    stage = 'Preparing GPU';
    detail = 'Setting up the renderer';
    progress = 46;
  } else if (status === 'warming' && !warmed) {
    stage = 'Optimizing renderer';
    detail = 'Compiling and warming up the model';
    progress = 76;
  } else if (status === 'waiting_input' || status === 'waiting_for_input' || status.startsWith('waiting_')) {
    stage = 'Preparing live output';
    detail = 'Waiting for fresh game frames';
    progress = 90;
  } else if (status === 'retrying' || status === 'recovering' || status === 'needs_reset') {
    stage = 'Recovering renderer';
    detail = 'Starting a clean render session';
    progress = 60;
  } else {
    rendererWarmup.hidden = true;
    return;
  }

  rendererWarmupStage.textContent = stage;
  rendererWarmupDetail.textContent = detail;
  rendererWarmupFill.style.width = `${progress}%`;
  const track = rendererWarmupFill.parentElement;
  if (track) track.setAttribute('aria-valuenow', String(progress));
  rendererWarmup.hidden = false;
}

async function refreshStylePrompt() {
  try {
    const resp = await fetch('/api/prompt?t=' + performance.now(), { cache: 'no-store' });
    if (!resp.ok) return;
    const data = await resp.json();
    const reportedOutputFps = Number(data.output_fps);
    if (Number.isFinite(reportedOutputFps) && reportedOutputFps >= 0) {
      workerOutputFps = reportedOutputFps;
    }
    const prompt = String(data.prompt || '');
    const status = String(data.status || 'idle');
    const rendererError = String(data.error || 'Renderer error').split('\\n', 1)[0].slice(0, 120);
    setRendererWarmup(data);
    if (status === 'loading') setPromptStatus('Loading renderer', 'pending');
    else if (status === 'warming') setPromptStatus('Optimizing renderer', 'pending');
    else if (status === 'waiting_input' || status === 'waiting_for_input' || status.startsWith('waiting_')) setPromptStatus('Waiting for fresh game frames', 'queued');
    else if (status === 'rendering' || status === 'output_queued') setPromptStatus(`Rendering · ${prompt}`, 'pending');
    else if (status === 'retrying' || status === 'recovering' || status === 'needs_reset') setPromptStatus('Restarting from frame 0', 'queued');
    else if (status === 'error' || status === 'failed' || status === 'load_failed' || status === 'input_error') setPromptStatus(rendererError, 'error');
    else if (status === 'done' || status === 'streaming') setPromptStatus(`Live · ${prompt}`, 'live');
    else if (status === 'queued') setPromptStatus(`Queued · ${prompt}`, 'queued');
    else if (prompt) setPromptStatus(`Current style · ${prompt}`, data.renderer_connected ? 'live' : 'queued');
    else setPromptStatus('Ready', data.renderer_connected ? 'live' : '');
  } catch (_) {}
}

async function submitStylePrompt() {
  const prompt = stylePrompt.value.trim();
  if (prompt.length < 3) {
    setPromptStatus('Please enter a longer prompt', 'error');
    stylePrompt.focus();
    return;
  }
  executePromptButton.disabled = true;
  setPromptStatus('Submitting', 'pending');
  beginStyleTransition();
  try {
    const resp = await fetch('/api/prompt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt })
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok || !data.ok) {
      if (data.error === 'cooldown') {
        throw new Error(`Please wait ${Math.ceil(Number(data.retry_in) || 1)} seconds`);
      }
      throw new Error(data.error || 'Submission failed');
    }
    const expectedIdentity = styleTransitionIdentityFromResponse(data);
    if (!expectedIdentity || !completeRenderIdentity(expectedIdentity)) {
      throw new Error('Renderer did not return a complete style identity');
    }
    beginStyleTransition(expectedIdentity);
    setPromptStatus(
      data.renderer_connected ? 'Applied · Takes effect on the next segment' : 'Submitted · Renderer unavailable',
      data.renderer_connected ? 'live' : 'queued'
    );
  } catch (error) {
    cancelStyleTransition();
    setPromptStatus(error && error.message ? error.message : 'Submission failed', 'error');
  } finally {
    executePromptButton.disabled = false;
  }
}

for (const button of promptPresetButtons) {
  button.setAttribute('aria-pressed', 'false');
  button.addEventListener('click', () => {
    customPromptButton.classList.remove('selected');
    customPromptButton.setAttribute('aria-pressed', 'false');
    for (const candidate of promptPresetButtons) {
      const selected = candidate === button;
      candidate.classList.toggle('selected', selected);
      candidate.setAttribute('aria-pressed', selected ? 'true' : 'false');
    }
    stylePrompt.value = button.dataset.prompt || '';
    stylePrompt.focus();
  });
}

customPromptButton.addEventListener('click', () => {
  for (const button of promptPresetButtons) {
    button.classList.remove('selected');
    button.setAttribute('aria-pressed', 'false');
  }
  customPromptButton.classList.add('selected');
  customPromptButton.setAttribute('aria-pressed', 'true');
  stylePrompt.value = '';
  stylePrompt.classList.add('custom-attention');
  stylePrompt.focus();
  setTimeout(() => stylePrompt.classList.remove('custom-attention'), 1400);
});

stylePrompt.addEventListener('input', () => {
  let presetMatched = false;
  stylePrompt.classList.remove('custom-attention');
  for (const button of promptPresetButtons) {
    const selected = stylePrompt.value === (button.dataset.prompt || '');
    presetMatched ||= selected;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', selected ? 'true' : 'false');
  }
  const customSelected = stylePrompt.value.trim().length > 0 && !presetMatched;
  customPromptButton.classList.toggle('selected', customSelected);
  customPromptButton.setAttribute('aria-pressed', customSelected ? 'true' : 'false');
});

stylePromptForm.addEventListener('submit', (event) => {
  event.preventDefault();
  submitStylePrompt().catch(() => {});
});

function trackLabel(track) {
  const option = Array.from(trackSelect.options).find(item => item.value === track);
  return option ? option.textContent : track.replaceAll('_', ' ');
}

function pause(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function waitForTrackSwitch(track) {
  const deadline = Date.now() + 45000;
  while (Date.now() < deadline) {
    const resp = await fetch('/api/track?t=' + performance.now(), { cache: 'no-store' });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.error || 'Unable to read track status');
    if (data.status === 'error') {
      throw new Error(data.error || 'Track switch failed');
    }
    if (data.status === 'ready' && data.current === track) return;
    await pause(750);
  }
  throw new Error('Track switch timed out');
}

async function switchTrack(track) {
  if (!track || track === currentTrack || pendingTrack) return;
  const previousTrack = currentTrack;
  const label = trackLabel(track);
  pendingTrack = track;
  trackSelect.disabled = true;
  localInputAuthority = null;
  setControlMode('auto');
  beginHardReset('track-switch', { trackId: track });
  autoModeButton.disabled = true;
  manualModeButton.disabled = true;
  text(topStatus, `Loading ${label}...`);
  text(frameStatus, 'Restarting the live game stream');
  try {
    const resp = await fetch('/api/track', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ track })
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok || !data.ok) throw new Error(data.error || 'Track switch failed');
    await waitForTrackSwitch(track);
    currentTrack = track;
    pendingTrack = '';
    trackSelect.disabled = false;
    autoModeButton.disabled = false;
    manualModeButton.disabled = !manualInputFeatureEnabled;
    trackSelect.value = track;
    text(topStatus, 'Live SuperTuxKart');
  } catch (error) {
    pendingTrack = '';
    trackSelect.disabled = false;
    autoModeButton.disabled = false;
    manualModeButton.disabled = !manualInputFeatureEnabled;
    if (previousTrack) trackSelect.value = previousTrack;
    beginHardReset('track-switch-recovery', { trackId: previousTrack });
    text(topStatus, error && error.message ? error.message : 'Track switch failed');
    text(frameStatus, 'Track switch failed');
  }
}

trackSelect.addEventListener('change', () => {
  switchTrack(trackSelect.value).catch(() => {});
});

function formatAge(ms) {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return '--';
  if (ms < 1000) return `${Math.max(0, Math.round(ms))} ms`;
  return `${(ms / 1000).toFixed(ms < 10000 ? 1 : 0)} s`;
}

function formatBytes(bytes) {
  if (!bytes || bytes < 0) return '--';
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatSeconds(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value < 0) return '--';
  if (value < 60) return `${Math.round(value)} s`;
  const minutes = Math.floor(value / 60);
  const rest = Math.round(value % 60);
  if (minutes < 60) return `${minutes}m ${rest}s`;
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  return `${hours}h ${mins}m`;
}

function applyRecordStatus(status) {
  const state = status && status.status ? status.status : 'idle';
  const activeRecord = Boolean(status && status.active);
  text(recordStatusValue, state);
  text(recordFramesValue, status && status.frames ? String(status.frames) : '--');
  text(recordDirValue, status && status.final_dir ? status.final_dir : '--');
  text(recordRunValue, status && status.run_name ? status.run_name : '--');
  const elapsed = status && status.elapsed_sec !== undefined ? formatSeconds(status.elapsed_sec) : '--';
  const duration = status && status.duration_sec ? formatSeconds(status.duration_sec) : '--';
  text(recordDetailValue, `${elapsed} / ${duration}`);
  recordStartButton.disabled = activeRecord;
  recordStopButton.disabled = !activeRecord;
}

async function refreshRecordStatus() {
  try {
    const resp = await fetch('/record/status?t=' + performance.now(), { cache: 'no-store' });
    if (!resp.ok) throw new Error('record status unavailable');
    applyRecordStatus(await resp.json());
  } catch (_) {
    applyRecordStatus({ status: 'unknown', active: false });
  }
}

// Recording is operator-only: the server requires ALAYA_PREVIEW_ADMIN_TOKEN,
// which an operator stores once via sessionStorage.setItem('alayaAdminToken', ...).
function adminHeaders() {
  let token = '';
  try { token = sessionStorage.getItem('alayaAdminToken') || ''; } catch (_) {}
  return token ? { 'X-Alaya-Admin-Token': token } : {};
}

async function startRecording() {
  const duration = Math.max(1, Math.min(28800, Number(recordDurationInput.value) || 7200));
  const fps = Math.max(1, Math.min(30, Number(recordFpsInput.value) || 8));
  recordStartButton.disabled = true;
  try {
    const resp = await fetch('/record/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...adminHeaders() },
      body: JSON.stringify({ duration_sec: duration, fps })
    });
    if (!resp.ok) throw new Error(await resp.text());
    applyRecordStatus(await resp.json());
  } catch (_) {
    await refreshRecordStatus();
  }
}

async function stopRecording() {
  recordStopButton.disabled = true;
  try {
    const resp = await fetch('/record/stop', { method: 'POST', headers: adminHeaders() });
    if (!resp.ok) throw new Error(await resp.text());
    applyRecordStatus(await resp.json());
  } catch (_) {
    await refreshRecordStatus();
  }
}

function profileIndex(name) {
  const index = videoProfileOrder.indexOf(name);
  return index < 0 ? videoProfileOrder.indexOf('balanced') : index;
}

function profileText() {
  const profile = videoProfiles[videoQuality] || videoProfiles.balanced;
  if (videoQualityMode === 'auto') return `Auto -> ${profile.label}`;
  return profile.label;
}

function restartRenderedVideo(reason = '', options = {}) {
  if (renderTransport !== 'h264') return null;
  if (reason) text(renderStatus, reason);
  const preserveFrame = Boolean(options && options.preserveFrame);
  if (preserveFrame && renderedVideoStarted) freezeRenderedVideoFrame();
  stopRenderedVideo();
  const token = startRenderedVideo({
    minSourceSeq: options ? options.minSourceSeq : null,
    inputEventId: options ? options.inputEventId : '',
    // Keep the new video composited behind the frozen image so
    // requestVideoFrameCallback can observe its first real frame.
    reveal: true,
    requestSourceFallback: !preserveFrame
  });
  sendStreamConfig();
  return token;
}

function fileOk(info) {
  return Boolean(info && info.present && info.size_ok);
}

function updateFilePills(files) {
  const rows = [
    ['rgb', 'rgb'],
    ['albedo', 'albedo'],
    ['normal/depth', 'normal_depth'],
    ['depth', 'depth']
  ];
  bufferFiles.innerHTML = rows.map(([label, key]) => {
    const info = files && files[key];
    const ok = fileOk(info);
    const status = ok ? formatBytes(info.size) : 'missing';
    return `<div class="file-pill ${ok ? 'ok' : 'bad'}"><span>${label}</span><strong>${status}</strong></div>`;
  }).join('');
}

function gbufferFileOk(files, name) {
  if (name === 'albedo') return fileOk(files && files.albedo);
  if (name === 'depth') return fileOk(files && files.depth);
  return fileOk(files && files.normal_depth);
}

function updateGbufferTitle() {
  const label = gbufferLabels[selectedGbuffer] || 'Albedo';
  text(gbufferTitle, label);
  gbufferImage.alt = `${label} G-buffer stream`;
}

function showGbufferCanvas(enabled) {
  if (!gbufferCanvas) return;
  gbufferCanvas.hidden = !enabled;
  gbufferImage.hidden = enabled;
}

function renderGbufferBitmap(bitmap) {
  if (!gbufferCanvas || !bitmap) return false;
  if (!gbufferSubscribed()) {
    if (bitmap.close) bitmap.close();
    return false;
  }
  gbufferCanvas.width = bitmap.width;
  gbufferCanvas.height = bitmap.height;
  if (gbufferBitmapCtx && gbufferBitmapCtx.transferFromImageBitmap) {
    gbufferBitmapCtx.transferFromImageBitmap(bitmap);
  } else if (gbuffer2dCtx) {
    gbuffer2dCtx.drawImage(bitmap, 0, 0);
    if (bitmap.close) bitmap.close();
  }
  showGbufferCanvas(true);
  return true;
}

function noteGbufferFrame(bytes) {
  const now = performance.now();
  lastGbufferFrameAt = now;
  gbufferFrames.push({ t: now, bytes: Math.max(0, Number(bytes) || 0) });
  while (gbufferFrames.length && now - gbufferFrames[0].t > 5000) {
    gbufferFrames.shift();
  }
  updateNetworkText();
}

function currentGbufferStats() {
  const now = performance.now();
  const frames = gbufferFrames.filter(frame => now - frame.t <= 5000);
  if (!frames.length) {
    return { frames: 0, fps: 0, kbps: 0, decoder: gbufferWorkerReady ? 'worker' : 'main' };
  }
  const spanMs = Math.max(1000, frames[frames.length - 1].t - frames[0].t);
  const fps = frames.length * 1000 / spanMs;
  const kbps = frames.reduce((sum, frame) => sum + frame.bytes, 0) * 8 / spanMs;
  const decoder = gbufferWorkerReady ? 'worker' : 'main';
  return { frames: frames.length, fps, kbps, decoder };
}

function gbufferPerfText() {
  const stats = currentGbufferStats();
  if (!stats.frames) return 'waiting';
  return `${stats.fps.toFixed(1)} fps · ${Math.round(stats.kbps)} kbps · ${stats.decoder}`;
}

function updateNetworkText() {
  if (!networkValue) return;
  const profile = videoProfiles[videoQuality] || videoProfiles.balanced;
  const rgbText = renderTransport === 'h264'
    ? `RGB ${Math.round(profile.kbps / 100) / 10} Mbps`
    : 'RGB JPEG';
  const stats = currentGbufferStats();
  const gbufferText = !gbufferSubscribed() || effectiveGbufferFps <= 0
    ? 'G-buffer paused'
    : (stats.frames ? `G-buffer ${Math.round(stats.kbps)} kbps` : 'G-buffer waiting');
  text(networkValue, `${rgbText} · ${gbufferText}`);
}

function gbufferRateLabel() {
  if (gbufferRateMode === 'auto') return `Auto -> ${effectiveGbufferFps} fps`;
  if (effectiveGbufferFps <= 0) return 'Paused';
  return `${effectiveGbufferFps} fps`;
}

function gbufferSizeLabel() {
  if (gbufferSizeMode === 'auto') return `Auto -> ${effectiveGbufferWidth}px`;
  return `${effectiveGbufferWidth}px`;
}

function setEffectiveGbufferFps(fps, reason = '') {
  const value = Math.max(0, Math.min(8, Number(fps) || 0));
  if (value === effectiveGbufferFps && !reason) return;
  effectiveGbufferFps = value;
  lastGbufferRateUpdateAt = performance.now();
  if (reason) text(bufferStatus, reason);
  sendStreamConfig();
  if (streamTransport === 'polling') {
    gbufferPending = false;
    scheduleGbufferImage(0);
  }
}

function setEffectiveGbufferWidth(width, reason = '') {
  const value = Math.max(160, Math.min(640, Number(width) || 360));
  if (value === effectiveGbufferWidth && !reason) return;
  effectiveGbufferWidth = value;
  lastGbufferSizeUpdateAt = performance.now();
  queuedGbufferPayload = null;
  gbufferFrames = [];
  if (reason) text(bufferStatus, reason);
  sendStreamConfig();
  if (streamTransport === 'polling') {
    gbufferPending = false;
    scheduleGbufferImage(0);
  }
}

function gbufferIntervalMs() {
  if (effectiveGbufferFps <= 0) return null;
  return Math.max(90, Math.round(1000 / effectiveGbufferFps));
}

function mainImageStreamSubscribed() {
  return Boolean(
    !document.hidden
    && (!modelInputsPanel || !modelInputsPanel.open)
  );
}

function modelImageStreamSubscribed() {
  return mainImageStreamSubscribed() && renderTransport !== 'h264';
}

function sourceImageStreamSubscribed() {
  return mainImageStreamSubscribed() && sourceStreamWanted;
}

function gbufferSubscribed() {
  return Boolean(modelInputsPanel && modelInputsPanel.open && !document.hidden);
}

function scheduleGbufferImage(delay) {
  if (streamTransport !== 'polling') return;
  if (gbufferTimer !== null) clearTimeout(gbufferTimer);
  if (!gbufferSubscribed()) return;
  const interval = gbufferIntervalMs();
  if (interval === null) return;
  gbufferTimer = setTimeout(requestGbufferImage, delay === undefined ? interval : delay);
}

function requestGbufferImage() {
  if (streamTransport !== 'polling') return;
  if (!gbufferSubscribed()) return;
  if (gbufferPending) return;
  gbufferPending = true;
  const img = gbufferImage;
  const bufferName = selectedGbuffer;
  img.onload = () => {
    gbufferPending = false;
    scheduleGbufferImage(gbufferIntervalMs() || 1000);
  };
  img.onerror = () => {
    gbufferPending = false;
    scheduleGbufferImage(Math.max(350, gbufferIntervalMs() || 1000));
  };
  img.src = `/gbuffer/${bufferName}.jpg?w=${effectiveGbufferWidth}&i=${gbufferSerial++}&t=${Date.now()}`;
}

function requestRenderedImage() {
  if (!modelImageStreamSubscribed()) return;
  if (streamTransport !== 'polling') return;
  if (renderedPending) return;
  const requestGeneration = mainImageStreamGeneration;
  renderedPending = true;
  const rendered = latestPageState && latestPageState.rendered
    ? latestPageState.rendered
    : {};
  const frameSeq = Number(rendered.seq);
  const frameSourceSeq = rendered.source_seq !== null
    && rendered.source_seq !== undefined
    ? Number(rendered.source_seq)
    : Number.NaN;
  const frameIdentity = renderIdentityFromState();
  fetch(`/model.jpg?i=${renderedSerial++}&t=${Date.now()}`, { cache: 'no-store' })
    .then(async response => {
      if (!response.ok) throw new Error('model frame unavailable');
      const payload = await response.arrayBuffer();
      if (requestGeneration !== mainImageStreamGeneration
          || !modelImageStreamSubscribed()) return;
      const responseSeqHeader = response.headers.get('X-STK-Frame-Seq');
      const responseSourceSeqHeader = response.headers.get('X-STK-Source-Seq');
      const responseSeq = responseSeqHeader === null
        ? Number.NaN
        : Number(responseSeqHeader);
      const responseSourceSeq = responseSourceSeqHeader === null
        ? Number.NaN
        : Number(responseSourceSeqHeader);
      const responseIdentity = normalizeRenderIdentity({
        track_id: response.headers.get('X-STK-Track-Id') || '',
        source_session_id: response.headers.get('X-STK-Source-Session-Id') || '',
        reset_epoch: response.headers.get('X-STK-Reset-Epoch') ?? undefined,
        source_reset_epoch: response.headers.get('X-STK-Source-Reset-Epoch') ?? undefined
      });
      const responseInputEventId = String(
        response.headers.get('X-STK-Input-Event-Id') || ''
      );
      setJpegFromArrayBuffer(renderedImage, payload, 'rendered', {
        seq: Number.isFinite(responseSeq) ? responseSeq : frameSeq,
        identity: completeRenderIdentity(responseIdentity)
          ? responseIdentity
          : frameIdentity,
        transport: 'polling',
        sourceSeq: Number.isFinite(responseSourceSeq)
          ? responseSourceSeq
          : (Number.isFinite(frameSourceSeq) ? frameSourceSeq : null),
        inputEventId: responseInputEventId
      });
    })
    .catch(() => {})
    .finally(() => {
      if (requestGeneration !== mainImageStreamGeneration) return;
      renderedPending = false;
      if (modelImageStreamSubscribed() && streamTransport === 'polling') {
        setTimeout(requestRenderedImage, 70);
      }
    });
}

function requestSourceImage() {
  if (!sourceImageStreamSubscribed()) return;
  if (streamTransport === 'websocket') return;
  if (sourcePending) return;
  const requestGeneration = sourceRequestGeneration;
  const streamGeneration = mainImageStreamGeneration;
  sourcePending = true;
  fetch(`/jpeg?i=${sourceSerial++}&t=${Date.now()}`, { cache: 'no-store' })
    .then(async response => {
      if (!response.ok) throw new Error('source frame unavailable');
      const seqHeader = response.headers.get('X-STK-Frame-Seq');
      const seq = seqHeader === null ? Number.NaN : Number(seqHeader);
      const identity = normalizeRenderIdentity({
        track_id: response.headers.get('X-STK-Track-Id') || '',
        source_session_id: response.headers.get('X-STK-Source-Session-Id') || ''
      });
      const payload = await response.arrayBuffer();
      if (requestGeneration !== sourceRequestGeneration
          || streamGeneration !== mainImageStreamGeneration
          || !sourceImageStreamSubscribed()) return;
      setJpegFromArrayBuffer(sourceImage, payload, 'source', {
        seq,
        identity,
        transport: 'polling'
      });
    })
    .catch(() => {})
    .finally(() => {
      if (requestGeneration !== sourceRequestGeneration
          || streamGeneration !== mainImageStreamGeneration) return;
      sourcePending = false;
      if (sourceImageStreamSubscribed()) setTimeout(requestSourceImage, 34);
    });
}

function videoStreamUrl(videoToken = null, minSourceSeq = null, inputEventId = '') {
  const profile = videoProfiles[videoQuality] || videoProfiles.balanced;
  const rendered = latestPageState && latestPageState.rendered
    ? latestPageState.rendered
    : {};
  const expectedEpoch = Number(rendered.source_reset_epoch) || 0;
  const params = new URLSearchParams({
    profile: videoQuality,
    w: String(profile.width),
    fps: String(profile.fps),
    kbps: String(profile.kbps),
    model: '1',
    stream_generation: Number.isInteger(Number(videoToken))
      ? `${inputClientId}:${Number(videoToken)}`
      : inputClientId,
    t: String(Date.now())
  });
  if (expectedEpoch > 0) params.set('source_reset_epoch', String(expectedEpoch));
  if (minSourceSeq !== null
      && minSourceSeq !== undefined
      && Number.isFinite(Number(minSourceSeq))) {
    params.set('min_source_seq', String(Number(minSourceSeq)));
  }
  if (inputEventId) params.set('input_event_id', String(inputEventId));
  return `/video/rendered.mp4?${params.toString()}`;
}

function cancelRenderedVideoFirstFrameProbe() {
  if (h264VideoFrameCallbackId !== null
      && typeof renderedVideo.cancelVideoFrameCallback === 'function') {
    try { renderedVideo.cancelVideoFrameCallback(h264VideoFrameCallbackId); } catch (_) {}
  }
  h264VideoFrameCallbackId = null;
}

function commitRenderedVideoFirstFrame(videoToken) {
  if (!renderedVideoStarted || h264ActiveVideoToken !== videoToken) return false;
  h264FirstFrameVideoToken = videoToken;
  renderedVideo.hidden = false;
  noteModelFrame();
  return true;
}

function renderedVideoEventMatchesActiveRequest() {
  const currentUrl = renderedVideo.currentSrc || renderedVideo.src || '';
  return Number.isInteger(h264ActiveVideoToken)
    && Boolean(h264RequestedVideoUrl)
    && currentUrl === h264RequestedVideoUrl;
}

function armRenderedVideoFirstFrameProbe(videoToken) {
  cancelRenderedVideoFirstFrameProbe();
  if (typeof renderedVideo.requestVideoFrameCallback !== 'function') return;
  try {
    h264VideoFrameCallbackId = renderedVideo.requestVideoFrameCallback(() => {
      h264VideoFrameCallbackId = null;
      commitRenderedVideoFirstFrame(videoToken);
    });
  } catch (_) {
    h264VideoFrameCallbackId = null;
  }
}

function startRenderedVideo(options = {}) {
  if (!mainImageStreamSubscribed() || renderTransport !== 'h264') return null;
  const rendered = latestPageState && latestPageState.rendered
    ? latestPageState.rendered
    : {};
  if (!rendered.external) {
    requestSourceLayer('model-wait');
    return null;
  }
  const reveal = !options || options.reveal !== false;
  if (!renderedVideoStarted || !renderedVideo.src) {
    modelFrameGateOpen = true;
    if (!options || options.requestSourceFallback !== false) {
      requestSourceLayer('h264-start');
    }
  }
  if (reveal) renderedVideo.hidden = false;
  if (!renderedVideoStarted || !renderedVideo.src) {
    const videoToken = ++h264VideoTokenCounter;
    h264ActiveVideoToken = videoToken;
    h264FirstFrameVideoToken = null;
    h264EpochRequested = Number(rendered.source_reset_epoch) || 0;
    h264IdentityRequested = renderIdentityFromState();
    renderedVideo.src = videoStreamUrl(
      videoToken,
      options ? options.minSourceSeq : null,
      options ? options.inputEventId : ''
    );
    h264RequestedVideoUrl = renderedVideo.src;
    renderedVideoStarted = true;
    armRenderedVideoFirstFrameProbe(videoToken);
  }
  renderedVideo.play().catch(() => {});
  return h264ActiveVideoToken;
}

function sendGbufferPayloadToWorker(payload) {
  if (!gbufferWorker || !gbufferWorkerReady || gbufferWorkerFailed) return false;
  if (gbufferWorkerBusy) {
    queuedGbufferPayload = payload;
    return true;
  }
  gbufferWorkerBusy = true;
  gbufferDecodeStartedAt = performance.now();
  try {
    gbufferWorker.postMessage({ type: 'decode', payload }, [payload]);
    return true;
  } catch (_) {
    gbufferWorkerBusy = false;
    gbufferDecodeStartedAt = 0;
    gbufferWorkerFailed = true;
    return false;
  }
}

async function drawGbufferPayload(payload) {
  if (!gbufferSubscribed()) return;
  if (sendGbufferPayloadToWorker(payload)) return;
  if (!gbufferCanvas || !('createImageBitmap' in window)) {
    setJpegFromArrayBuffer(gbufferImage, payload, 'gbuffer');
    showGbufferCanvas(false);
    noteGbufferFrame(payload.byteLength || 0);
    return;
  }
  if (gbufferDecodeBusy) {
    queuedGbufferPayload = payload;
    return;
  }
  gbufferDecodeBusy = true;
  try {
    let nextPayload = payload;
    while (nextPayload) {
      const bytes = nextPayload.byteLength || 0;
      queuedGbufferPayload = null;
      const bitmap = await createImageBitmap(new Blob([nextPayload], { type: 'image/jpeg' }));
      if (renderGbufferBitmap(bitmap)) noteGbufferFrame(bytes);
      nextPayload = queuedGbufferPayload;
    }
  } catch (_) {
    setJpegFromArrayBuffer(gbufferImage, payload, 'gbuffer');
    showGbufferCanvas(false);
    noteGbufferFrame(payload.byteLength || 0);
  } finally {
    gbufferDecodeBusy = false;
  }
}

function stopRenderedVideo() {
  cancelRenderedVideoFirstFrameProbe();
  renderedVideoStarted = false;
  h264EpochRequested = null;
  h264IdentityRequested = null;
  h264ActiveVideoToken = null;
  h264FirstFrameVideoToken = null;
  h264RequestedVideoUrl = '';
  renderedVideo.playbackRate = 1.0;
  renderedVideo.pause();
  renderedVideo.removeAttribute('src');
  renderedVideo.load();
}

function freezeRenderedVideoFrame() {
  renderedVideo.pause();
  if (renderedVideo.readyState >= 2
      && renderedVideo.videoWidth > 0
      && renderedVideo.videoHeight > 0) {
    try {
      const canvas = document.createElement('canvas');
      canvas.width = renderedVideo.videoWidth;
      canvas.height = renderedVideo.videoHeight;
      const context = canvas.getContext('2d', { alpha: false });
      if (context) {
        context.drawImage(renderedVideo, 0, 0, canvas.width, canvas.height);
        renderedImage.onload = null;
        renderedImage.onerror = null;
        renderedImage.src = canvas.toDataURL('image/jpeg', 0.88);
      }
    } catch (_) {}
  }
  renderedImage.hidden = false;
  renderedVideo.hidden = true;
}

function invalidateMainImageRequests() {
  mainImageStreamGeneration += 1;
  renderedPending = false;
  sourcePending = false;
  modelFrameMetaBySeq.clear();
  sourceFrameMetaBySeq.clear();
  clearModelPresentationPipeline(true);
  sourceDecodeGeneration += 1;
  sourceImage.onload = null;
  sourceImage.onerror = null;
  if (sourceDecoder) {
    sourceDecoder.onload = null;
    sourceDecoder.onerror = null;
    sourceDecoder.src = '';
  }
  sourceDecoder = null;
  if (sourceDecoderUrl) URL.revokeObjectURL(sourceDecoderUrl);
  sourceDecoderUrl = null;
}

function pauseMainImageStream() {
  invalidateMainImageRequests();
  if (renderTransport === 'h264' && renderedVideoStarted) {
    freezeRenderedVideoFrame();
    stopRenderedVideo();
  }
}

function setModelInputsPausedUi(paused) {
  const isPaused = Boolean(paused);
  renderedStage.classList.toggle('model-inputs-paused', isPaused);
  if (modelInputsPauseNotice) modelInputsPauseNotice.hidden = !isPaused;
}

function clearModelInputsIdleTimer() {
  if (modelInputsIdleTimer !== null) clearTimeout(modelInputsIdleTimer);
  modelInputsIdleTimer = null;
  modelInputsIdleDeadline = null;
}

function armModelInputsIdleTimer() {
  clearModelInputsIdleTimer();
  if (!modelInputsPanel || !modelInputsPanel.open || document.hidden) return;
  modelInputsIdleDeadline = performance.now() + modelInputsIdleMs;
  modelInputsIdleTimer = setTimeout(() => {
    modelInputsIdleTimer = null;
    modelInputsIdleDeadline = null;
    if (!document.hidden && modelInputsPanel.open) {
      closeModelInputsPanel('idle-timeout');
    }
  }, modelInputsIdleMs);
}

function noteModelInputsActivity() {
  if (modelInputsPanel && modelInputsPanel.open && !document.hidden) {
    armModelInputsIdleTimer();
  }
}

function syncModelInputsPanel(reason = 'toggle') {
  if (!modelInputsPanel) return false;
  const open = Boolean(modelInputsPanel.open);
  setModelInputsPausedUi(open);
  if (open === modelInputsStateApplied) {
    if (!open) clearModelInputsIdleTimer();
    return false;
  }

  modelInputsStateApplied = open;
  gbufferPending = false;
  queuedGbufferPayload = null;
  gbufferFrames = [];
  if (gbufferTimer !== null) clearTimeout(gbufferTimer);
  gbufferTimer = null;
  lastGbufferPayloadAt = 0;
  lastGbufferSeq = null;
  text(bufferStatus, open ? 'Waiting for G-buffer preview' : 'Paused');

  if (open) {
    pauseMainImageStream();
    sendStreamConfig();
    scheduleGbufferImage(0);
    armModelInputsIdleTimer();
  } else {
    clearModelInputsIdleTimer();
    sendStreamConfig();
    resumeMainImageStream();
  }
  updateNetworkText();
  return true;
}

function closeModelInputsPanel(reason = 'resume') {
  clearModelInputsIdleTimer();
  if (!modelInputsPanel || !modelInputsPanel.open) {
    setModelInputsPausedUi(false);
    return false;
  }
  modelInputsPanel.open = false;
  return syncModelInputsPanel(reason);
}

function resumeMainImageStream() {
  mainImageStreamGeneration += 1;
  renderedPending = false;
  sourcePending = false;
  if (!mainImageStreamSubscribed()) return;
  if (renderTransport === 'h264') {
    startRenderedVideo();
  } else if (streamTransport === 'polling') {
    requestRenderedImage();
  }
  if (sourceStreamWanted) requestSourceImage();
}

function updatePlaybackStats() {
  let delay = null;
  let liveEdge = null;
  const ranges = renderedVideo.buffered;
  if (ranges && ranges.length > 0 && Number.isFinite(renderedVideo.currentTime)) {
    liveEdge = ranges.end(ranges.length - 1);
    delay = liveEdge - renderedVideo.currentTime;
    if (!Number.isFinite(delay) || delay < 0) delay = null;
  }

  let total = 0;
  let dropped = 0;
  if (renderedVideo.getVideoPlaybackQuality) {
    const quality = renderedVideo.getVideoPlaybackQuality();
    total = quality.totalVideoFrames || 0;
    dropped = quality.droppedVideoFrames || 0;
  } else {
    total = renderedVideo.webkitDecodedFrameCount || 0;
    dropped = renderedVideo.webkitDroppedFrameCount || 0;
  }

  const now = performance.now();
  let dropRatio = currentDropRatio;
  if (lastVideoQualitySample && total >= lastVideoQualitySample.total) {
    const deltaTotal = total - lastVideoQualitySample.total;
    const deltaDropped = Math.max(0, dropped - lastVideoQualitySample.dropped);
    if (deltaTotal >= 20) {
      dropRatio = deltaDropped / deltaTotal;
      lastVideoQualitySample = { total, dropped, t: now };
    }
  } else {
    lastVideoQualitySample = { total, dropped, t: now };
  }

  currentPlaybackDelay = delay;
  currentDropRatio = Number.isFinite(dropRatio) ? dropRatio : 0;
  const delayText = delay === null ? '--' : `${Math.round(delay * 1000)} ms`;
  const dropText = `${Math.round(currentDropRatio * 100)}% drop`;
  text(playbackValue, `${delayText} · ${dropText}`);
  return { delay, liveEdge, dropRatio: currentDropRatio, total, dropped };
}

function switchVideoProfile(profileName, reason) {
  if (!(profileName in videoProfiles) || profileName === videoQuality) return false;
  videoQuality = profileName;
  autoQualityCooldownUntil = performance.now() + 6500;
  autoQualityStableSince = performance.now();
  lastVideoQualitySample = null;
  restartRenderedVideo(reason);
  return true;
}

function tuneAutoVideoQuality(stats) {
  if (videoQualityMode !== 'auto' || renderTransport !== 'h264' || document.hidden) return;
  if (!stats || stats.delay === null) return;
  const now = performance.now();
  const bad = stats.delay > 0.9 || stats.dropRatio > 0.08;
  const severe = stats.delay > 1.35 || stats.dropRatio > 0.16;
  const good = stats.delay < 0.35 && stats.dropRatio < 0.025 && !renderedVideo.paused;

  if (bad) autoQualityStableSince = now;
  if (now < autoQualityCooldownUntil) return;

  const index = profileIndex(videoQuality);
  if ((severe || bad) && index > 0) {
    switchVideoProfile(videoProfileOrder[index - 1], `auto quality · ${videoProfiles[videoProfileOrder[index - 1]].label}`);
    return;
  }

  if (good) {
    if (now - autoQualityStableSince > 18000 && index < videoProfileOrder.length - 1) {
      switchVideoProfile(videoProfileOrder[index + 1], `auto quality · ${videoProfiles[videoProfileOrder[index + 1]].label}`);
    }
  } else {
    autoQualityStableSince = now;
  }
}

function tuneAutoGbufferRate(stats) {
  if ((gbufferRateMode !== 'auto' && gbufferSizeMode !== 'auto') || !gbufferSubscribed() || !stats) return;
  const now = performance.now();
  const delay = Number.isFinite(stats.delay) ? stats.delay : 0;
  const drop = Number.isFinite(stats.dropRatio) ? stats.dropRatio : 0;
  let targetFps = 4;
  let targetWidth = 360;
  if (delay > 0.75 || drop > 0.08) {
    targetFps = 2;
    targetWidth = 240;
  } else if (delay < 0.28 && drop < 0.025 && videoQuality !== 'fast') {
    targetFps = 8;
    targetWidth = 360;
  } else if (videoQuality === 'fast') {
    targetFps = 4;
    targetWidth = 240;
  }
  if (gbufferRateMode === 'auto' && now - lastGbufferRateUpdateAt >= 4000 && targetFps !== effectiveGbufferFps) {
    setEffectiveGbufferFps(targetFps);
  }
  if (gbufferSizeMode === 'auto' && now - lastGbufferSizeUpdateAt >= 6000 && targetWidth !== effectiveGbufferWidth) {
    setEffectiveGbufferWidth(targetWidth);
  }
}

function tuneVideoLatency() {
  if (renderTransport !== 'h264' || document.hidden || renderedVideo.paused) return;
  const stats = updatePlaybackStats();
  const delay = stats && stats.delay;
  if (Number.isFinite(delay)) {
    if (delay > 1.2) {
      // Progressive live fMP4 is not reliably seekable in Chrome. Seeking to a
      // buffered timestamp can reset playback to the same fragment repeatedly.
      renderedVideo.playbackRate = 1.1;
    } else if (delay > 0.55) {
      renderedVideo.playbackRate = 1.06;
    } else if (renderedVideo.playbackRate !== 1.0) {
      renderedVideo.playbackRate = 1.0;
    }
  }
  tuneAutoGbufferRate(stats);
  tuneAutoVideoQuality(stats);
}

function recoverGbufferIfStalled() {
  if (!gbufferSubscribed() || effectiveGbufferFps <= 0) return;
  const now = performance.now();
  if (now - lastGbufferWatchdogAt < 1000) return;
  lastGbufferWatchdogAt = now;

  if (gbufferWorkerBusy && gbufferDecodeStartedAt && now - gbufferDecodeStartedAt > 1800) {
    gbufferWorkerReady = false;
    gbufferWorkerFailed = true;
    gbufferWorkerBusy = false;
    gbufferDecodeStartedAt = 0;
    const next = queuedGbufferPayload;
    queuedGbufferPayload = null;
    if (next) {
      drawGbufferPayload(next).catch(() => {});
    }
  }

  const frameStalled = now - lastGbufferFrameAt > 2500;
  const payloadStalled = !lastGbufferPayloadAt || now - lastGbufferPayloadAt > 2500;
  if (!frameStalled && !payloadStalled) return;

  if (streamTransport === 'websocket' && streamSocket && streamSocket.readyState === WebSocket.OPEN) {
    sendStreamConfig();
    text(bufferStatus, `${gbufferLabels[selectedGbuffer] || 'G-buffer'} · recovering stream`);
    return;
  }

  if (streamTransport !== 'polling') {
    startPollingFallback();
    return;
  }

  gbufferPending = false;
  scheduleGbufferImage(0);
}

function setRenderTransport(mode, reason = '') {
  renderTransport = mode === 'h264' ? 'h264' : 'websocket';
  renderTransportSelect.value = renderTransport;
  lastRenderedJpegSeq = null;
  clearModelGate();
  invalidateLoadedModelFence(true);
  if (renderTransport === 'h264') {
    modelFrameGateOpen = true;
    requestSourceLayer('h264-start');
    startRenderedVideo();
  } else {
    stopRenderedVideo();
    renderedVideo.hidden = true;
    renderedImage.hidden = false;
    if (streamTransport === 'polling') requestRenderedImage();
  }
  if (sourceStreamWanted) requestSourceImage();
  if (reason) text(renderStatus, reason);
  sendStreamConfig();
}

function commitRenderedLoadToken(
  generation,
  url,
  displayedUrl,
  frameMeta,
  presentationToken = modelPresentationToken
) {
  if (generation !== renderedDecodeGeneration
      || presentationToken !== modelPresentationToken
      || renderedObjectUrl !== url
      || displayedUrl !== url
      || !frameMeta) return false;
  markModelFrameLoaded(
    frameMeta.seq,
    frameMeta.identity,
    frameMeta
  );
  return true;
}

function setJpegFromArrayBuffer(img, payload, kind, frameMeta = null) {
  if (kind === 'rendered' && !modelImageStreamSubscribed()) return;
  if (kind === 'source' && !sourceImageStreamSubscribed()) return;
  if (kind === 'gbuffer' && !gbufferSubscribed()) return;
  const url = URL.createObjectURL(new Blob([payload], { type: 'image/jpeg' }));
  if (kind === 'rendered') {
    if (!frameMeta) {
      URL.revokeObjectURL(url);
      return;
    }
    const generation = renderedDecodeGeneration;
    while (renderedDecoders.size >= modelPresentationMaxDepth) {
      const [staleUrl, staleDecoder] = renderedDecoders.entries().next().value;
      staleDecoder.onload = null;
      staleDecoder.onerror = null;
      staleDecoder.src = '';
      renderedDecoders.delete(staleUrl);
      URL.revokeObjectURL(staleUrl);
      modelPresentationDropped += 1;
    }
    const decoder = new Image();
    renderedDecoders.set(url, decoder);
    decoder.onload = () => {
      const decodedUrl = decoder.currentSrc || decoder.src;
      renderedDecoders.delete(url);
      if (generation !== renderedDecodeGeneration || decodedUrl !== url) {
        URL.revokeObjectURL(url);
        return;
      }
      enqueueDecodedModelFrame({ generation, url, frameMeta });
    };
    decoder.onerror = () => {
      renderedDecoders.delete(url);
      URL.revokeObjectURL(url);
    };
    decoder.src = url;
  } else if (kind === 'source') {
    const requestGeneration = sourceRequestGeneration;
    const candidate = {
      ...(frameMeta || {}),
      seq: Number(frameMeta && frameMeta.seq),
      identity: sourceFenceIdentity(frameMeta && frameMeta.identity)
    };
    if (!sourceFramePassesFence(candidate, requestGeneration)) {
      URL.revokeObjectURL(url);
      return;
    }
    const decodeGeneration = ++sourceDecodeGeneration;
    if (sourceDecoder) {
      sourceDecoder.onload = null;
      sourceDecoder.onerror = null;
      sourceDecoder.src = '';
    }
    if (sourceDecoderUrl) URL.revokeObjectURL(sourceDecoderUrl);
    const decoder = new Image();
    sourceDecoder = decoder;
    sourceDecoderUrl = url;
    decoder.onload = () => {
      if (decodeGeneration !== sourceDecodeGeneration
          || requestGeneration !== sourceRequestGeneration
          || !sourceFramePassesFence(candidate, requestGeneration)) {
        URL.revokeObjectURL(url);
        if (sourceDecoderUrl === url) sourceDecoderUrl = null;
        return;
      }
      sourceDecoder = null;
      sourceDecoderUrl = null;
      const previous = sourceObjectUrl;
      sourceObjectUrl = url;
      img.onload = () => {
        const displayedUrl = img.currentSrc || img.src;
        if (decodeGeneration === sourceDecodeGeneration
            && sourceObjectUrl === url
            && displayedUrl === url) {
          commitSourceFrameLoaded(candidate, requestGeneration);
        }
      };
      img.onerror = () => {
        if (sourceObjectUrl === url) sourceObjectUrl = previous;
        URL.revokeObjectURL(url);
      };
      img.src = url;
      if (previous) setTimeout(() => URL.revokeObjectURL(previous), 1000);
    };
    decoder.onerror = () => {
      if (decodeGeneration === sourceDecodeGeneration) sourceDecoder = null;
      if (sourceDecoderUrl === url) sourceDecoderUrl = null;
      URL.revokeObjectURL(url);
    };
    decoder.src = url;
  } else {
    const previous = gbufferObjectUrl;
    gbufferObjectUrl = url;
    img.src = url;
    if (previous) setTimeout(() => URL.revokeObjectURL(previous), 1000);
  }
}

function handleModelBinary(seq, payload, frameMeta) {
  if (!modelImageStreamSubscribed()) return;
  lastRenderedJpegSeq = seq;
  if (!frameMeta || !completeRenderIdentity(frameMeta.identity)) {
    unfencedModelBinaryCount += 1;
    if (styleTransitionActive) return;
    invalidateLoadedModelFence(false);
    requestSourceLayer('model-meta-missing');
    setJpegFromArrayBuffer(renderedImage, payload, 'rendered', null);
    return;
  }
  if (styleTransitionActive
      && (!modelFrameGateIdentity
        || !sameRenderIdentity(modelFrameGateIdentity, frameMeta.identity))) {
    return;
  }
  noteModelFrameArrival(seq, frameMeta.identity);
  setJpegFromArrayBuffer(
    renderedImage,
    payload,
    'rendered',
    {
      seq,
      sourceSeq: frameMeta.sourceSeq,
      inputEventId: frameMeta.inputEventId,
      identity: frameMeta.identity,
      exactMeta: true,
      socketGeneration: frameMeta.socketGeneration,
      connectionGeneration: frameMeta.connectionGeneration,
      frameToken: frameMeta.token
    }
  );
}

async function handleStreamBinary(data, socketGeneration) {
  const buffer = data instanceof Blob ? await data.arrayBuffer() : data;
  if (socketGeneration !== streamSocketGeneration) return;
  if (!buffer || buffer.byteLength < 5) return;
  const view = new DataView(buffer);
  const streamId = view.getUint8(0);
  const seq = view.getUint32(1, false);
  if (streamId === 1) {
    if (!modelImageStreamSubscribed()) return;
    handleModelBinary(seq, buffer.slice(5), null);
  } else if (streamId === 2) {
    if (!gbufferSubscribed()) return;
    lastGbufferSeq = seq;
    lastGbufferPayloadAt = performance.now();
    await drawGbufferPayload(buffer.slice(5));
  } else if (streamId === 3) {
    if (!modelImageStreamSubscribed()) return;
    handleModelBinary(seq, buffer.slice(5), null);
  } else if (streamId === 4) {
    if (!sourceImageStreamSubscribed()) return;
    setJpegFromArrayBuffer(sourceImage, buffer.slice(5), 'source', {
      seq,
      identity: sourceIdentityFromState(),
      transport: 'websocket-legacy'
    });
  } else if (streamId === 5) {
    if (!modelImageStreamSubscribed() || buffer.byteLength < 9) return;
    const token = view.getUint32(5, false);
    const key = modelFrameMetaKey(
      socketGeneration,
      activeStreamConnectionGeneration,
      token,
      seq
    );
    const frameMeta = modelFrameMetaBySeq.get(key) || null;
    modelFrameMetaBySeq.delete(key);
    handleModelBinary(seq, buffer.slice(9), frameMeta);
  } else if (streamId === 6) {
    if (!sourceImageStreamSubscribed() || buffer.byteLength < 9) return;
    const token = view.getUint32(5, false);
    const key = sourceFrameMetaKey(
      socketGeneration,
      activeStreamConnectionGeneration,
      token,
      seq
    );
    const frameMeta = sourceFrameMetaBySeq.get(key) || null;
    sourceFrameMetaBySeq.delete(key);
    if (!frameMeta) return;
    setJpegFromArrayBuffer(sourceImage, buffer.slice(9), 'source', {
      seq,
      identity: frameMeta.identity,
      exactMeta: true,
      transport: 'websocket',
      socketGeneration: frameMeta.socketGeneration,
      connectionGeneration: frameMeta.connectionGeneration,
      frameToken: frameMeta.token
    });
  }
}

function sendStreamConfig() {
  if (!streamSocket || streamSocket.readyState !== WebSocket.OPEN) return;
  const config = {
    type: 'config',
    gbuffer: selectedGbuffer,
    gbuffer_fps: gbufferSubscribed() ? effectiveGbufferFps : 0,
    gbuffer_width: effectiveGbufferWidth,
    rendered: false,
    model: modelImageStreamSubscribed(),
    source: sourceImageStreamSubscribed(),
    video_quality_mode: videoQualityMode,
    video_quality: videoQuality,
    visible: !document.hidden
  };
  lastStreamConfig = { ...config };
  streamConfigHistory.push({ at: performance.now(), ...config });
  while (streamConfigHistory.length > 32) streamConfigHistory.shift();
  streamSocket.send(JSON.stringify(config));
}

function clearInputRetry() {
  if (inputRetryTimer !== null) clearTimeout(inputRetryTimer);
  inputRetryTimer = null;
}

function armInputRetry(payload) {
  clearInputRetry();
  inputRetryTimer = setTimeout(() => {
    if (!pendingInputTransition || pendingInputTransition.payload.input_seq !== payload.input_seq) return;
    pendingInputTransition.attempts += 1;
    transmitInputPayload(payload);
    armInputRetry(payload);
  }, Math.min(1000, 220 + 120 * (pendingInputTransition ? pendingInputTransition.attempts : 0)));
}

function acceptedAuthorityFromAck(ack) {
  if (!ack || !ack.accepted || !ack.effective) return null;
  if (ack.mode !== 'manual' || ack.owner_client_id !== inputClientId) return null;
  return {
    eventId: String(ack.input_event_id || ''),
    serverTime: Number(ack.server_time) || 0,
    serverInstanceId: String(ack.server_instance_id || serverInstanceId || ''),
    inputSeq: Number(ack.input_seq),
    renderIdentity: renderIdentityFromAck(ack)
  };
}

function cancelLocalInputAuthority(
  reason,
  serverTime = 0,
  incomingServerInstanceId = '',
  force = false,
  suppressHeldKeys = false
) {
  const sameServerInstance = !incomingServerInstanceId
    || !localInputAuthority
    || !localInputAuthority.serverInstanceId
    || incomingServerInstanceId === localInputAuthority.serverInstanceId;
  if (localInputAuthority
      && !force
      && sameServerInstance
      && Number(serverTime) > 0
      && Number(serverTime) < Number(localInputAuthority.serverTime || 0)) {
    return false;
  }
  clearInputRetry();
  pendingInputTransition = null;
  localInputAuthority = null;
  manualInputArmed = false;
  localHandover = null;
  modelPresentationBoundary = null;
  clearModelGate();
  if (suppressHeldKeys) {
    for (const code of active) suppressedInputKeys.add(code);
  }
  active.clear();
  applyLocalControlMode('auto');
  if (reason) renderedStage.dataset.authority = reason;
  return true;
}

function resetInputSequenceForServerInstance() {
  clearInputRetry();
  pendingInputTransition = null;
  handoverInputSeqWatermark = -1;
  nextInputSeq = 0;
  latestInputIssuedSeq = -1;
  latestTransitionSeq = -1;
  lastInputAckSeq = -1;
  lastInputAck = null;
  lastInputAckTransport = '';
  lastDriveSignature = 'auto:0:0:0';
  lastInputSentAt = 0;
  ownershipReleaseSent = false;
}

function observeServerInstance(instanceId, state = null, source = 'unknown') {
  const nextInstanceId = String(instanceId || '');
  if (!nextInstanceId) return 'missing';
  if (retiredServerInstanceIds.has(nextInstanceId)) {
    lastServerInstanceObservation = {
      instanceId: nextInstanceId,
      status: 'retired',
      source,
      at: performance.now()
    };
    return 'retired';
  }
  if (!serverInstanceId) {
    serverInstanceId = nextInstanceId;
    lastServerInstanceObservation = {
      instanceId: nextInstanceId,
      status: 'initial',
      source,
      at: performance.now()
    };
    return 'initial';
  }
  if (serverInstanceId === nextInstanceId) return 'same';
  retiredServerInstanceIds.add(serverInstanceId);
  while (retiredServerInstanceIds.size > 32) {
    retiredServerInstanceIds.delete(retiredServerInstanceIds.values().next().value);
  }
  serverInstanceId = nextInstanceId;
  cancelLocalInputAuthority(
    'server-restart',
    0,
    nextInstanceId,
    true,
    true
  );
  resetInputSequenceForServerInstance();
  beginHardReset(
    'server-restart',
    state ? sourceIdentityFromState(state) : sourceIdentityFromState()
  );
  lastServerInstanceObservation = {
    instanceId: nextInstanceId,
    status: 'changed',
    source,
    at: performance.now()
  };
  return 'changed';
}

function sendServerInstanceAuto() {
  applyLocalControlMode('auto');
  ownershipReleaseSent = true;
  return sendInputState('release', true);
}

function reconcileInputAuthority(state) {
  const input = state && state.input ? state.input : {};
  const stateInstanceId = String(
    input.server_instance_id || state.server_instance_id || ''
  );
  if (input.input_suspended) {
    const cancelled = cancelLocalInputAuthority(
      input.input_suspended_reason || 'input-suspended',
      Number(input.server_time ?? input.t) || 0,
      stateInstanceId,
      true
    );
    beginHardReset('track-switch-server', sourceIdentityFromState(state));
    return cancelled;
  }
  if (!localInputAuthority) {
    const stateTime = Number(input.server_time ?? input.t) || 0;
    const acknowledgedAt = Number(localHandover && localHandover.inputServerTime)
      || Number(lastInputAck && lastInputAck.server_time)
      || 0;
    const eventId = String(input.input_event_id || '');
    if (localHandover
        && localHandover.inputEventId
        && eventId
        && eventId !== localHandover.inputEventId
        && stateTime >= acknowledgedAt) {
      localHandover = null;
      clearModelGate();
      return true;
    }
    return false;
  }
  if (localInputAuthority.serverInstanceId
      && stateInstanceId
      && localInputAuthority.serverInstanceId !== stateInstanceId) {
    return cancelLocalInputAuthority(
      'server-restart',
      0,
      stateInstanceId,
      true
    );
  }
  const stateTime = Number(input.server_time ?? input.t) || 0;
  if (stateTime < Number(localInputAuthority.serverTime || 0)) return false;
  const owner = String(input.owner_client_id || '');
  const eventId = String(input.input_event_id || '');
  const eventChanged = Boolean(eventId && eventId !== localInputAuthority.eventId);
  const writtenByOtherClient = Boolean(
    input.client_id && input.client_id !== inputClientId
  );
  const superseded = Boolean(owner && owner !== inputClientId)
    || (eventChanged && (writtenByOtherClient || input.mode !== 'manual'));
  if (superseded) {
    return cancelLocalInputAuthority(
      'superseded',
      stateTime,
      stateInstanceId
    );
  }
  return false;
}

function handleInputAck(ack, context = {}) {
  if (!ack || ack.type !== 'input_ack' || ack.client_id !== inputClientId) return;
  const ackInstanceId = String(ack.server_instance_id || '');
  const expectedInstanceId = String(context.expectedServerInstanceId || '');
  if (retiredServerInstanceIds.has(ackInstanceId)) return;
  if (serverInstanceId
      && ackInstanceId !== serverInstanceId
      && expectedInstanceId
      && ackInstanceId === expectedInstanceId) {
    return;
  }
  const instanceObservation = observeServerInstance(
    ackInstanceId,
    null,
    `input_ack:${String(context.transport || 'unknown')}`
  );
  if (instanceObservation === 'retired' || instanceObservation === 'missing') return;
  if (instanceObservation === 'changed') {
    sendServerInstanceAuto();
    return;
  }
  if (ackInstanceId !== serverInstanceId) return;
  const inputSeq = Number(ack.input_seq);
  if (!Number.isFinite(inputSeq)) return;
  lastInputAck = ack;
  lastInputAckTransport = String(context.transport || 'unknown');
  lastInputAckSeq = Math.max(lastInputAckSeq, inputSeq);
  const pending = pendingInputTransition
    && inputSeq === pendingInputTransition.payload.input_seq
    ? pendingInputTransition
    : null;
  if (pending && ack.accepted) {
    clearInputRetry();
    pendingInputTransition = null;
  }
  if (!ack.accepted) {
    if (ack.retryable) {
      if (pending && inputRetryTimer === null) armInputRetry(pending.payload);
      return;
    }
    if ([
      'owned_by_other_client',
      'ownership_required',
      'no_owner',
      'track_switch_in_progress',
      'server_instance_mismatch'
    ].includes(ack.reason)) {
      cancelLocalInputAuthority(
        ack.reason,
        Number(ack.server_time) || 0,
        String(ack.server_instance_id || ''),
        ack.reason === 'track_switch_in_progress'
          || ack.reason === 'server_instance_mismatch',
        ack.reason === 'server_instance_mismatch'
      );
      if (ack.reason === 'track_switch_in_progress') {
        beginHardReset('track-switch-server', sourceIdentityFromState());
      }
      if (ack.reason === 'server_instance_mismatch') sendServerInstanceAuto();
      return;
    }
    if (!pending) return;
    clearInputRetry();
    pendingInputTransition = null;
    if (ack.reason === 'stale_sequence' || ack.reason === 'sequence_conflict') {
      nextInputSeq = Math.max(
        nextInputSeq,
        Number(ack.last_input_seq) + 1 || 0,
        Number(ack.sequence_watermark) + 1 || 0
      );
      sendInputState(pending.payload.intent, true);
      return;
    }
    if (localHandover && localHandover.inputSeq === inputSeq) {
      localHandover = null;
      clearModelGate();
    }
    return;
  }
  if (!ack.effective || inputSeq < latestTransitionSeq) {
    if (localHandover && localHandover.inputSeq === inputSeq) {
      localHandover = null;
      clearModelGate();
    }
    return;
  }
  const authority = acceptedAuthorityFromAck(ack);
  if (authority) {
    localInputAuthority = authority;
  } else if (ack.mode === 'auto' && ack.owner_client_id === '') {
    localInputAuthority = null;
  }
  if (ack.transition || ack.authority_changed) {
    latestTransitionSeq = Math.max(latestTransitionSeq, inputSeq);
  }
  if (ack.transition
      || ack.authority_changed
      || (localHandover && localHandover.inputSeq === inputSeq)) {
    beginLocalHandover(ack);
  }
}

function transmitInputPayload(payload) {
  if (streamSocket && streamSocket.readyState === WebSocket.OPEN) {
    if (streamHelloSocketGeneration !== streamSocketGeneration) return false;
    try {
      streamSocket.send(JSON.stringify(payload));
      return true;
    } catch (_) {}
  }
  fetch('/input', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    keepalive: payload.intent === 'release'
  }).then(async response => {
    const ack = await response.json().catch(() => null);
    if (ack) handleInputAck(ack, {
      transport: 'http',
      expectedServerInstanceId: payload.server_instance_id
    });
  }).catch(() => {});
  return true;
}

function sendInputState(intent = 'heartbeat', forceTransition = false) {
  if (!serverInstanceId) return false;
  if (manualInputArmed && intent === 'heartbeat') return false;
  if (streamSocket
      && streamSocket.readyState === WebSocket.OPEN
      && streamHelloSocketGeneration !== streamSocketGeneration) {
    return false;
  }
  lastInputSentAt = performance.now();
  const release = intent === 'release';
  const payloadMode = release ? 'auto' : controlMode;
  const payloadKeys = release ? [] : Array.from(active);
  const nextDriveSignature = release ? 'auto:0:0:0' : driveSignature();
  const transitionRequested = forceTransition
    || (intent !== 'heartbeat' && nextDriveSignature !== lastDriveSignature);
  lastDriveSignature = nextDriveSignature;
  const payload = {
    type: 'input',
    client_id: inputClientId,
    input_seq: nextInputSeq++,
    server_instance_id: serverInstanceId,
    intent,
    mode: payloadMode,
    keys: payloadKeys,
    t: performance.now()
  };
  latestInputIssuedSeq = payload.input_seq;
  if (transitionRequested || release) {
    latestTransitionSeq = payload.input_seq;
    beginOptimisticHandover(payload.input_seq);
    clearInputRetry();
    pendingInputTransition = { payload, attempts: 0 };
    armInputRetry(payload);
  }
  transmitInputPayload(payload);
  return true;
}

function inputHeartbeat() {
  if (document.hidden) return;
  if (pendingInputTransition) return;
  const now = performance.now();
  const interval = controlMode === 'manual' ? 500 : 1200;
  if (now - lastInputSentAt >= interval) {
    sendInputState('heartbeat');
  }
}

function releaseInputOwnership() {
  if (ownershipReleaseSent) return;
  if (manualInputArmed && !localInputAuthority && active.size === 0) {
    manualInputArmed = false;
    ownershipReleaseSent = true;
    applyLocalControlMode('auto');
    return;
  }
  if (controlMode !== 'manual' && active.size === 0) return;
  ownershipReleaseSent = true;
  applyLocalControlMode('auto');
  sendInputState('release');
}

function startPollingFallback() {
  if (streamTransport === 'polling') return;
  streamTransport = 'polling';
  text(renderStatus, 'source fallback · polling');
  gbufferPending = false;
  renderedPending = false;
  sourcePending = false;
  showGbufferCanvas(false);
  scheduleGbufferImage(0);
  requestRenderedImage();
  requestSourceImage();
}

function clearStreamReconnectTimer() {
  if (streamReconnectTimer !== null) clearTimeout(streamReconnectTimer);
  streamReconnectTimer = null;
}

function scheduleStreamReconnect(immediate = false) {
  if (!('WebSocket' in window) || document.hidden) return;
  if (streamSocket
      && (streamSocket.readyState === WebSocket.OPEN
        || streamSocket.readyState === WebSocket.CONNECTING)) return;
  if (streamReconnectTimer !== null) return;
  const delay = immediate
    ? 0
    : Math.min(4000, 250 * (2 ** Math.min(streamReconnectAttempt, 4)));
  streamReconnectAttempt += 1;
  streamReconnectTimer = setTimeout(() => {
    streamReconnectTimer = null;
    if (document.hidden) return;
    startWebSocketStream();
  }, delay);
}

function handleStreamHello(message, socketGeneration) {
  if (socketGeneration !== streamSocketGeneration) return;
  const connectionGeneration = String(
    message && message.connection_generation || ''
  );
  if (!connectionGeneration) return;
  const instanceObservation = observeServerInstance(
    String(message && message.server_instance_id || ''),
    null,
    'stream_hello'
  );
  if (instanceObservation === 'retired' || instanceObservation === 'missing') {
    if (streamSocket) {
      try { streamSocket.close(); } catch (_) {}
    }
    return;
  }
  if (activeStreamConnectionGeneration
      && activeStreamConnectionGeneration !== connectionGeneration) {
    beginHardReset('stream-reconnect', sourceIdentityFromState());
  } else {
    modelFrameMetaBySeq.clear();
    sourceFrameMetaBySeq.clear();
  }
  activeStreamConnectionGeneration = connectionGeneration;
  streamHelloSocketGeneration = socketGeneration;
  websocketRecoveryPending = false;
  if (instanceObservation === 'changed') {
    sendServerInstanceAuto();
  } else if (pendingInputTransition) {
    transmitInputPayload(pendingInputTransition.payload);
    if (inputRetryTimer === null) {
      armInputRetry(pendingInputTransition.payload);
    }
  } else {
    sendInputState('heartbeat');
  }
}

function startWebSocketStream() {
  if (!('WebSocket' in window)) {
    startPollingFallback();
    return;
  }
  if (document.hidden) return;
  if (streamSocket
      && (streamSocket.readyState === WebSocket.OPEN
        || streamSocket.readyState === WebSocket.CONNECTING)) return;
  clearStreamReconnectTimer();
  const socketGeneration = ++streamSocketGeneration;
  let socketOpened = false;
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  const socketParams = new URLSearchParams({ client_id: inputClientId });
  if (new URLSearchParams(location.search).get('candidate_test') === '1') {
    socketParams.set('candidate_test', '1');
  }
  const socket = new WebSocket(
    `${scheme}://${location.host}/ws/stream?${socketParams.toString()}`
  );
  streamSocket = socket;
  socket.binaryType = 'arraybuffer';
  socket.onopen = () => {
    if (socketGeneration !== streamSocketGeneration || socket !== streamSocket) return;
    socketOpened = true;
    streamReconnectAttempt = 0;
    clearStreamReconnectTimer();
    streamOpened = true;
    streamTransport = 'websocket';
    text(renderStatus, 'source fallback · websocket');
    sendStreamConfig();
  };
  socket.onmessage = (event) => {
    if (socketGeneration !== streamSocketGeneration || socket !== streamSocket) return;
    if (typeof event.data === 'string') {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'stream_hello') {
          handleStreamHello(message, socketGeneration);
        }
        if (message.type === 'state' && message.state) {
          applyState(message.state, { transport: 'websocket' });
        }
        if (message.type === 'model_meta') {
          rememberModelFrameMeta(message, socketGeneration);
        }
        if (message.type === 'source_meta') {
          rememberSourceFrameMeta(message, socketGeneration);
        }
        if (message.type === 'input_ack') handleInputAck(message, {
          transport: 'websocket',
          expectedServerInstanceId: serverInstanceId
        });
        if (message.type === 'control' && message.control) {
          updateControlReadout(message.control);
        }
      } catch (_) {}
      return;
    }
    handleStreamBinary(event.data, socketGeneration).catch(() => {});
  };
  socket.onerror = () => {
    if (socketGeneration === streamSocketGeneration && !socketOpened) {
      startPollingFallback();
    }
  };
  socket.onclose = (event) => {
    if (socketGeneration !== streamSocketGeneration || socket !== streamSocket) return;
    lastStreamClose = {
      code: Number(event && event.code) || 0,
      reason: String(event && event.reason || ''),
      wasClean: Boolean(event && event.wasClean)
    };
    streamOpened = false;
    streamSocket = null;
    if (streamHelloSocketGeneration === socketGeneration) {
      streamHelloSocketGeneration = 0;
    }
    websocketRecoveryPending = true;
    if (!document.hidden) {
      beginHardReset('stream-disconnected', sourceIdentityFromState());
      startPollingFallback();
      scheduleStreamReconnect(false);
    }
  };
  setTimeout(() => {
    if (socketGeneration === streamSocketGeneration && !socketOpened) {
      if (socket === streamSocket) {
        streamSocket = null;
        try { socket.close(); } catch (_) {}
      }
      websocketRecoveryPending = true;
      startPollingFallback();
      scheduleStreamReconnect(false);
    }
  }, 1200);
}

function updateControlReadout(control = {}) {
  latestControl = control && typeof control === 'object' ? control : {};
  const displayControl = controlMode === 'manual'
    ? { ...latestControl, ...driveState(controlMode, active) }
    : latestControl;
  const steer = Math.max(-1, Math.min(1, Number(displayControl.steer) || 0));
  const accel = Math.max(0, Math.min(1, Number(displayControl.accel) || 0));
  const brake = displayControl.brake === true ? 1 : 0;
  const strengths = {
    left: Math.max(0, -steer),
    right: Math.max(0, steer),
    accel,
    brake
  };
  for (const node of controlNodes) {
    const strength = strengths[node.dataset.control] || 0;
    node.classList.toggle('active', strength > 0.04);
    node.style.setProperty('--strength', strength.toFixed(3));
  }
  const modeLabel = controlMode === 'manual' ? 'User input' : 'AI input';
  text(inputStatus, modeLabel);
  controlReadout.setAttribute(
    'aria-label',
    `${modeLabel}: steer ${steer.toFixed(2)}, throttle ${accel.toFixed(2)}, brake ${brake ? 'on' : 'off'}`
  );
}

function updateKeys(send = true, intent = 'heartbeat') {
  updateControlReadout(latestControl);
  if (!send) return;
  if (intent !== 'input') {
    if (inputCoalesceTimer !== null) clearTimeout(inputCoalesceTimer);
    inputCoalesceTimer = null;
    sendInputState(intent);
    return;
  }
  const now = performance.now();
  const remaining = inputCoalesceWindowMs - (now - lastInputTransitionAt);
  if (remaining <= 0 && inputCoalesceTimer === null) {
    lastInputTransitionAt = now;
    sendInputState('input');
    return;
  }
  if (inputCoalesceTimer !== null) clearTimeout(inputCoalesceTimer);
  inputCoalesceTimer = setTimeout(() => {
    inputCoalesceTimer = null;
    lastInputTransitionAt = performance.now();
    sendInputState('input');
  }, Math.max(0, remaining));
}

function applyLocalControlMode(mode) {
  const nextMode = mode === 'manual' ? 'manual' : 'auto';
  if (nextMode !== controlMode) latestControl = {};
  controlMode = nextMode;
  if (controlMode === 'auto') {
    manualInputArmed = false;
    active.clear();
  }
  if (controlMode === 'manual') delete renderedStage.dataset.authority;
  autoModeButton.classList.toggle('active', controlMode === 'auto');
  manualModeButton.classList.toggle('active', controlMode === 'manual');
  updateControlReadout(latestControl);
}

function setControlMode(mode, intent = '') {
  if (mode === 'manual' && !manualInputFeatureEnabled) {
    applyLocalControlMode('auto');
    return;
  }
  if (mode === 'manual' && (pendingTrack || serverInputSuspended)) {
    applyLocalControlMode('auto');
    return;
  }
  const armManual = mode === 'manual';
  applyLocalControlMode(mode);
  ownershipReleaseSent = false;
  if (armManual) {
    manualInputArmed = true;
    updateKeys(true, intent || 'manual');
    return;
  }
  updateKeys(true, intent || (controlMode === 'manual' ? 'manual' : 'auto'));
}

autoModeButton.addEventListener('click', () => setControlMode('auto'));
manualModeButton.addEventListener('click', () => {
  if (!manualInputFeatureEnabled) return;
  closeModelInputsPanel('manual-mode');
  setControlMode('manual');
  focusGameStage();
});
renderedStage.addEventListener('pointerdown', () => {
  focusGameStage();
});
recordStartButton.addEventListener('click', () => {
  startRecording().catch(() => {});
});
recordStopButton.addEventListener('click', () => {
  stopRecording().catch(() => {});
});
renderTransportSelect.addEventListener('change', () => {
  setRenderTransport(renderTransportSelect.value);
});
videoQualitySelect.addEventListener('change', () => {
  const selected = videoQualitySelect.value;
  if (selected === 'auto') {
    videoQualityMode = 'auto';
    if (!(videoQuality in videoProfiles)) videoQuality = 'balanced';
  } else {
    videoQualityMode = 'manual';
    videoQuality = selected in videoProfiles ? selected : 'balanced';
  }
  autoQualityStableSince = performance.now();
  autoQualityCooldownUntil = performance.now() + 2500;
  lastVideoQualitySample = null;
  if (renderTransport === 'h264') {
    restartRenderedVideo(`switching · ${profileText()}`);
  }
  text(videoProfileValue, profileText());
  updateNetworkText();
  sendStreamConfig();
});

renderedVideo.addEventListener('playing', () => {
  if (renderTransport === 'h264' && mainImageStreamSubscribed()) {
    if (renderedVideoEventMatchesActiveRequest()) {
      commitRenderedVideoFirstFrame(h264ActiveVideoToken);
    }
    renderedVideo.hidden = false;
    lastVideoQualitySample = null;
    text(renderStatus, `model output · h264 video · ${profileText()}`);
    noteModelFrame();
  }
});
renderedVideo.addEventListener('waiting', () => {
  if (renderTransport === 'h264' && mainImageStreamSubscribed()) {
    text(renderStatus, 'source fallback · h264 buffering');
  }
});
renderedVideo.addEventListener('error', () => {
  if (renderTransport === 'h264' && mainImageStreamSubscribed()) {
    stopRenderedVideo();
    requestSourceLayer('h264-wait');
    setTimeout(() => {
      if (renderTransport === 'h264' && mainImageStreamSubscribed()) {
        startRenderedVideo();
      }
    }, 600);
  }
});

gbufferSelect.addEventListener('change', () => {
  selectedGbuffer = gbufferSelect.value in gbufferLabels ? gbufferSelect.value : 'albedo';
  gbufferPending = false;
  queuedGbufferPayload = null;
  gbufferFrames = [];
  updateGbufferTitle();
  sendStreamConfig();
  scheduleGbufferImage(0);
});

gbufferRateSelect.addEventListener('change', () => {
  const selected = gbufferRateSelect.value;
  gbufferRateMode = selected === 'auto' ? 'auto' : 'manual';
  const nextFps = selected === 'auto' ? 4 : Number(selected);
  gbufferPending = false;
  queuedGbufferPayload = null;
  gbufferFrames = [];
  setEffectiveGbufferFps(nextFps);
});

gbufferSizeSelect.addEventListener('change', () => {
  const selected = gbufferSizeSelect.value;
  gbufferSizeMode = selected === 'auto' ? 'auto' : 'manual';
  const nextWidth = selected === 'auto' ? 360 : Number(selected);
  gbufferPending = false;
  queuedGbufferPayload = null;
  gbufferFrames = [];
  setEffectiveGbufferWidth(nextWidth);
});

modelInputsPanel.addEventListener('toggle', () => {
  syncModelInputsPanel('toggle');
});

for (const eventName of ['pointerdown', 'keydown', 'focusin', 'input', 'change']) {
  modelInputsPanel.addEventListener(eventName, noteModelInputsActivity, true);
}

modelInputsResumeButton.addEventListener('click', () => {
  closeModelInputsPanel('resume-button');
  focusGameStage();
});

document.addEventListener('visibilitychange', () => {
  sendStreamConfig();
  if (!document.hidden) {
    if (!streamSocket || streamSocket.readyState === WebSocket.CLOSED) {
      scheduleStreamReconnect(true);
    }
    ownershipReleaseSent = false;
    gbufferPending = false;
    renderedPending = false;
    sourcePending = false;
    scheduleGbufferImage(0);
    resumeMainImageStream();
    if (modelInputsPanel.open) armModelInputsIdleTimer();
    updateKeys(true, 'heartbeat');
  } else {
    clearModelInputsIdleTimer();
    clearStreamReconnectTimer();
    clearModelGate();
    invalidateMainImageRequests();
    invalidateLoadedModelFence(true);
    releaseInputOwnership();
    if (renderTransport === 'h264') stopRenderedVideo();
  }
});

addEventListener('keydown', (event) => {
  if (!interesting.has(event.code)) return;
  if (isFormTarget(event.target)) return;
  closeModelInputsPanel('player-input');
  if (pendingTrack || serverInputSuspended) {
    event.preventDefault();
    return;
  }
  const code = normalizeKey(event.code);
  event.preventDefault();
  if (!manualInputFeatureEnabled) return;
  if (suppressedInputKeys.has(code)) return;
  if (!serverInstanceId
      || (streamSocket
        && streamSocket.readyState === WebSocket.OPEN
        && streamHelloSocketGeneration !== streamSocketGeneration)) {
    suppressedInputKeys.add(code);
    return;
  }
  if (controlMode !== 'manual') {
    applyLocalControlMode('manual');
  }
  if (active.has(code)) return;
  ownershipReleaseSent = false;
  active.add(code);
  manualInputArmed = false;
  updateKeys(true, 'input');
});
addEventListener('keyup', (event) => {
  if (!interesting.has(event.code)) return;
  const code = normalizeKey(event.code);
  if (suppressedInputKeys.has(code)) {
    event.preventDefault();
    suppressedInputKeys.delete(code);
    return;
  }
  if (!active.has(code)) return;
  event.preventDefault();
  active.delete(code);
  updateKeys(true, 'input');
});
addEventListener('blur', () => {
  releaseInputOwnership();
});
addEventListener('pagehide', () => {
  clearModelInputsIdleTimer();
});
addEventListener('focus', () => {
  ownershipReleaseSent = false;
  if (!document.hidden) sendInputState('heartbeat');
});

function applyRenderHandover(state) {
  latestPageState = state;
  const rendered = state.rendered || {};
  const renderedIdentity = renderIdentityFromState(state);
  const renderedInputEventId = String(rendered.input_event_id || '');
  const hardResetMinSourceSeq = hardResetHandover
    && hardResetHandover.minSourceSeq !== null
    && hardResetHandover.minSourceSeq !== undefined
    && Number.isFinite(Number(hardResetHandover.minSourceSeq))
    ? Number(hardResetHandover.minSourceSeq)
    : null;
  if (styleTransitionActive
      && (!rendered.external
        || !renderedIdentity
        || !modelFrameGateIdentity
        || !sameRenderIdentity(modelFrameGateIdentity, renderedIdentity))) {
    return;
  }
  if (hardResetHandover && hardResetHandover.expectedSourceIdentity) {
    const sourceIdentity = sourceIdentityFromState(state);
    if (!sourceIdentityMatches(
      hardResetHandover.expectedSourceIdentity,
      sourceIdentity
    )) {
      clearModelGate();
      requestSourceLayer(
        hardResetHandover.reason || 'source-reset',
        hardResetHandover.expectedSourceIdentity,
        hardResetMinSourceSeq
      );
      return;
    }
  }
  if (hardResetHandover
      && hardResetHandover.expectedDrive
      && !renderedControlMatches(
        hardResetHandover.expectedDrive,
        state.control || {}
      )) {
    clearModelGate();
    requestSourceLayer(
      hardResetHandover.reason || 'source-reset',
      hardResetHandover.expectedSourceIdentity,
      hardResetMinSourceSeq
    );
    return;
  }
  if (hardResetHandover && hardResetMinSourceSeq !== null) {
    const renderedSourceSeq = Number(rendered.source_seq);
    const boundaryIdentity = hardResetHandover.boundaryRenderIdentity;
    if (!rendered.external
        || (boundaryIdentity
          && !renderIdentityMatches(boundaryIdentity, renderedIdentity))
        || (hardResetHandover.inputEventId
          && renderedInputEventId !== hardResetHandover.inputEventId)
        || !Number.isFinite(renderedSourceSeq)
        || renderedSourceSeq < hardResetMinSourceSeq) {
      clearModelGate();
      requestSourceLayer(
        hardResetHandover.reason || 'source-reset',
        hardResetHandover.expectedSourceIdentity,
        hardResetMinSourceSeq
      );
      return;
    }
  }
  if (localHandover) {
    const renderedSourceSeq = Number(rendered.source_seq);
    const boundarySourceSeq = localHandover.boundarySourceSeq;
    if (boundarySourceSeq === null
        || !Number.isFinite(Number(boundarySourceSeq))
        || !rendered.external
        || !renderIdentityMatches(localHandover.renderIdentity, renderedIdentity)
        || !renderedInputEventId
        || renderedInputEventId !== localHandover.inputEventId
        || !Number.isFinite(renderedSourceSeq)
        || renderedSourceSeq < Number(boundarySourceSeq)
        || !renderedControlMatches(localHandover.expectedDrive, state.control || {})) {
      clearModelGate();
      return;
    }
    if (renderTransport !== 'h264' && !localHandover.presentationPrimed) {
      clearModelGate();
      return;
    }
  } else if (!rendered.external) {
    holdModelOutput('renderer-unavailable');
    return;
  }

  modelFrameGateOpen = true;
  modelFrameGateSeq = Number.isFinite(Number(rendered.seq))
    ? Number(rendered.seq)
    : null;
  modelFrameGateSourceSeq = rendered.source_seq !== null
    && rendered.source_seq !== undefined
    && Number.isFinite(Number(rendered.source_seq))
    ? Number(rendered.source_seq)
    : null;
  modelFrameGateIdentity = renderedIdentity;
  modelFrameGateInputEventId = renderedInputEventId;
  modelFrameGateVideoToken = null;
  if (!mainImageStreamSubscribed()) return;
  if (renderTransport === 'h264') {
    const epoch = Number(rendered.source_reset_epoch) || 0;
    const activeHandover = localHandover || hardResetHandover;
    const handoverBoundary = localHandover
      ? Number(localHandover.boundarySourceSeq)
      : hardResetMinSourceSeq;
    const requiredSourceSeq = Number.isFinite(handoverBoundary)
      ? handoverBoundary
      : modelFrameGateSourceSeq;
    const handoverVideoToken = activeHandover
      && Number.isInteger(Number(activeHandover.h264VideoToken))
      ? Number(activeHandover.h264VideoToken)
      : null;
    const streamMatches = renderedVideoStarted
      && h264EpochRequested === epoch
      && sameRenderIdentity(h264IdentityRequested, renderedIdentity);
    const needsFreshHandoverStream = Boolean(activeHandover)
      && handoverVideoToken === null;
    const needsStreamRestart = !streamMatches
      || (handoverVideoToken !== null
        && h264ActiveVideoToken !== handoverVideoToken);
    if (needsFreshHandoverStream || needsStreamRestart) {
      const preserveFrame = Boolean(
        styleTransitionActive
        || (localHandover && activeHandover === localHandover)
      );
      const videoToken = restartRenderedVideo('model ready · switching', {
        minSourceSeq: requiredSourceSeq,
        inputEventId: activeHandover
          ? String(activeHandover.inputEventId || '')
          : renderedInputEventId,
        preserveFrame
      });
      if (!Number.isInteger(videoToken)) {
        clearModelGate();
        requestSourceLayer('h264-wait');
        return;
      }
      if (activeHandover) {
        activeHandover.h264VideoToken = videoToken;
        activeHandover.h264MinSourceSeq = requiredSourceSeq;
      }
      modelFrameGateVideoToken = videoToken;
      modelFrameGateSourceSeq = requiredSourceSeq;
      noteModelFrame();
      return;
    }
    modelFrameGateVideoToken = handoverVideoToken !== null
      ? handoverVideoToken
      : h264ActiveVideoToken;
    if (activeHandover
        && activeHandover.h264MinSourceSeq !== null
        && activeHandover.h264MinSourceSeq !== undefined
        && Number.isFinite(Number(activeHandover.h264MinSourceSeq))) {
      modelFrameGateSourceSeq = Number(activeHandover.h264MinSourceSeq);
    }
    noteModelFrame();
    return;
  }

  noteModelFrame();
  if (!modelFrameGateOpen) return;

  if (streamTransport === 'polling') {
    renderedPending = false;
    requestRenderedImage();
  } else {
    sendStreamConfig();
  }
}

function applyState(state, context = {}) {
    const incomingInstanceId = String(state.server_instance_id
      || (state.input && state.input.server_instance_id)
      || '');
    if (retiredServerInstanceIds.has(incomingInstanceId)) return;
    latestPageState = state;
    releaseSupersededModelPresentationBoundary(
      state.input && state.input.input_event_id
    );
    const readOnlyObserver = Boolean(state.read_only_observer);
    serverInputSuspended = Boolean(
      state.input && state.input.input_suspended
    );
    trackSelect.disabled = Boolean(pendingTrack || readOnlyObserver);
    autoModeButton.disabled = Boolean(
      pendingTrack || serverInputSuspended || readOnlyObserver
    );
    manualModeButton.disabled = Boolean(
      !manualInputFeatureEnabled || pendingTrack || serverInputSuspended || readOnlyObserver
    );
    const instanceObservation = observeServerInstance(
      incomingInstanceId,
      state,
      `state:${String(context.transport || 'unknown')}`
    );
    if (instanceObservation === 'retired' || instanceObservation === 'missing') return;
    if (context.transport === 'http') {
      if (instanceObservation === 'changed') sendServerInstanceAuto();
      else if (instanceObservation === 'initial') sendInputState('heartbeat');
    }
    observeStateIdentity(state);
    reconcileDecodedModelLoad(state);
    const authoritySuperseded = reconcileInputAuthority(state);
    if (!authoritySuperseded) applyRenderHandover(state);
    const now = performance.now();
    if (lastSeq !== null && state.seq !== lastSeq) {
      const dt = Math.max(1, now - lastSeqAt);
      previewFps = ((state.seq - lastSeq) * 1000) / dt;
      lastSeqAt = now;
    } else if (state.seq !== lastSeq) {
      lastSeqAt = now;
    } else {
      previewFps *= 0.7;
    }
    lastSeq = state.seq;

    const age = state.frame_age_ms ?? 0;
    const live = age < 3000;
    const sourceVisible = renderedStage.classList.contains('source-active');
    const renderBase = sourceVisible
      ? ((state.handover && state.handover.phase === 'waiting_for_model') ? 'source live · model catching up' : 'source live')
      : (state.rendered && state.rendered.external ? 'model output' : 'source fallback');
    const video = state.video || {};
    const videoLabel = video.encoder ? video.encoder : 'h264 video';
    const profileLabel = profileText();
    const cfg = state.run_config || {};
    const rawTrack = cfg.TRACK || '--';
    const formattedTrackLabel = rawTrack.replaceAll('_', ' ').split(' ')
      .map(part => part ? part[0].toUpperCase() + part.slice(1) : part)
      .join(' ');
    if (!pendingTrack && rawTrack !== currentTrack) {
      currentTrack = rawTrack;
      lastSeq = null;
      lastSeqAt = now;
      previewFps = 0;
      if (Array.from(trackSelect.options).some(option => option.value === rawTrack)) {
        trackSelect.value = rawTrack;
      }
    }
    const visibleTrack = pendingTrack && currentTrack ? currentTrack : rawTrack;
    const visibleTrackLabel = Array.from(trackSelect.options).some(option => option.value === visibleTrack)
      ? trackLabel(visibleTrack)
      : formattedTrackLabel;
    liveStatus.className = `value ${live ? 'live' : 'stale'}`;
    text(liveStatus, live ? 'Live' : 'Delayed');
    const renderedWidth = Number(state.rendered && state.rendered.width);
    const renderedHeight = Number(state.rendered && state.rendered.height);
    const displayWidth = renderedWidth > 0 ? renderedWidth : Number(state.width);
    const displayHeight = renderedHeight > 0 ? renderedHeight : Number(state.height);
    text(resolutionMetric, `${displayWidth}x${displayHeight}`);
    text(frameMetric, String(state.seq));
    const streamFps = workerOutputFps > 0.05 ? workerOutputFps : previewFps;
    text(fpsMetric, streamFps > 0.05 ? streamFps.toFixed(1) : '0');
    text(ageMetric, formatAge(age));
    text(renderStatus, `${renderBase} · ${renderTransport === 'h264' ? `${videoLabel} · ${profileLabel}` : streamTransport}`);
    renderStatus.title = video.reason || '';
    text(topStatus, pendingTrack
      ? `Loading ${trackLabel(pendingTrack)}...`
      : 'Live SuperTuxKart');
    text(frameStatus, pendingTrack
      ? 'Restarting the live game stream'
      : (live ? 'Live sync' : 'Delayed'));
    text(captureStatus, live ? 'writing latest buffers' : 'latest buffers paused');

    text(raceLabel, visibleTrackLabel.toUpperCase());
    const serverMode = state.input && state.input.mode ? state.input.mode : controlMode;
    updateControlReadout({ ...(state.control || {}), mode: serverMode });
    text(modeValue, serverMode === 'manual' ? 'user browser control' : (cfg.PROFILE_TIME ? `${cfg.PROFILE_TIME}s auto` : 'auto preview'));
    text(captureEveryValue, cfg.CAPTURE_EVERY || '--');
    text(rawSizeValue, state.raw_frame_bytes ? `${formatBytes(state.raw_frame_bytes)} / frame` : '--');
    text(videoProfileValue, profileLabel);
    updateNetworkText();
    text(captureDirValue, state.capture_dir || '--');
    text(renderedDirValue, state.rendered && state.rendered.external ? state.rendered.dir : 'source fallback');
    updateFilePills(state.files || {});

    const selectedReady = gbufferFileOk(state.files || {}, selectedGbuffer);
    const label = gbufferLabels[selectedGbuffer] || 'Albedo';
    const detail = gbufferDetails[selectedGbuffer] || 'RGB';
    const rate = gbufferRateLabel();
    const size = gbufferSizeLabel();
    const perf = gbufferPerfText();
    const seqText = lastGbufferSeq === null ? 'seq --' : `seq ${lastGbufferSeq}`;
    const syncText = renderTransport === 'h264'
      ? 'RGB buffered; Sync JPEG aligns'
      : (lastRenderedJpegSeq === null ? 'sync JPEG' : `RGB seq ${lastRenderedJpegSeq}`);
    text(bufferStatus, selectedReady ? `${label} · ${detail} · ${seqText} · ${rate} · ${size} · ${perf} · ${syncText}` : `${label} · waiting for buffer`);
}

async function tick() {
  if (streamTransport === 'websocket') {
    setTimeout(tick, 1000);
    return;
  }
  try {
    const stateResp = await fetch('/state?t=' + performance.now(), { cache: 'no-store' });
    if (!stateResp.ok) throw new Error('no frame');
    const state = await stateResp.json();
    applyState(state, { transport: 'http' });
  } catch (err) {
    liveStatus.className = 'value offline';
    text(liveStatus, 'Offline');
    text(topStatus, 'Waiting for live world');
    text(frameStatus, 'Waiting for game stream');
    text(captureStatus, 'not receiving state');
  } finally {
    const handover = latestPageState && latestPageState.handover
      ? latestPageState.handover
      : {};
    const waiting = Boolean(localHandover)
      || handover.phase === 'waiting_for_epoch'
      || handover.phase === 'waiting_for_model';
    setTimeout(tick, waiting ? 150 : 750);
  }
}
window.__alayaInputDebug = () => ({
  clientId: inputClientId,
  nextInputSeq,
  latestInputIssuedSeq,
  latestTransitionSeq,
  handoverInputSeqWatermark,
  hardResetGeneration,
  styleTransitionActive,
  styleTransitionPreviousIdentity: styleTransitionPreviousIdentity
    ? { ...styleTransitionPreviousIdentity }
    : null,
  lastInputAckSeq,
  lastInputAck,
  pendingInputSeq: pendingInputTransition
    ? pendingInputTransition.payload.input_seq
    : null,
  localHandover: localHandover ? { ...localHandover } : null,
  localInputAuthority: localInputAuthority ? { ...localInputAuthority } : null,
  hardResetHandover: hardResetHandover ? { ...hardResetHandover } : null,
  arrivedModelFrameSeq,
  arrivedModelFrameIdentity: arrivedModelFrameIdentity
    ? { ...arrivedModelFrameIdentity }
    : null,
  loadedModelFrame: loadedModelFrame
    ? { ...loadedModelFrame, identity: { ...loadedModelFrame.identity } }
    : null,
  heldModelLoad: heldModelLoad
    ? { ...heldModelLoad, identity: { ...heldModelLoad.identity } }
    : null,
  modelFrameGateOpen,
  modelFrameGateSeq,
  modelFrameGateSourceSeq,
  modelFrameGateIdentity: modelFrameGateIdentity
    ? { ...modelFrameGateIdentity }
    : null,
  modelFrameGateInputEventId,
  modelFrameGateVideoToken,
  sourceActive: renderedStage.classList.contains('source-active'),
  activeKeys: Array.from(active),
  suppressedInputKeys: Array.from(suppressedInputKeys),
  controlMode,
  manualInputArmed,
  autoModeActive: autoModeButton.classList.contains('active'),
  manualModeActive: manualModeButton.classList.contains('active'),
  inputStatus: inputStatus.textContent,
  renderedControlMode: latestPageState && latestPageState.control
    ? String(latestPageState.control.mode || '')
    : '',
  lastSourceJpegSeq,
  sourceLoadedCount,
  lastSourceLoadedAt,
  sourceRequestGeneration,
  sourceDecodeGeneration,
  sourceFrameGate: sourceFrameGate
    ? {
        ...sourceFrameGate,
        expectedIdentity: sourceFrameGate.expectedIdentity
          ? { ...sourceFrameGate.expectedIdentity }
          : null
      }
    : null,
  loadedSourceFrame: loadedSourceFrame
    ? {
        ...loadedSourceFrame,
        identity: loadedSourceFrame.identity ? { ...loadedSourceFrame.identity } : null
      }
    : null,
  renderedDecodeGeneration,
  modelPresentationQueueDepth: modelPresentationQueue.length,
  modelPresentationRunning,
  modelPresentationNextAt,
  modelPresentationDropped,
  modelPresentationPresented,
  modelPresentationFastStartFrames,
  modelPresentationBoundary: modelPresentationBoundary
    ? {
        minSourceSeq: modelPresentationBoundary.minSourceSeq,
        identity: { ...modelPresentationBoundary.identity },
        inputEventId: modelPresentationBoundary.inputEventId
      }
    : null,
  mainImageStreamSubscribed: mainImageStreamSubscribed(),
  modelImageStreamSubscribed: modelImageStreamSubscribed(),
  sourceImageStreamSubscribed: sourceImageStreamSubscribed(),
  mainImageStreamGeneration,
  renderedCurrentSrc: renderedImage.currentSrc || renderedImage.src || '',
  renderedVideoStarted,
  h264ActiveVideoToken,
  h264FirstFrameVideoToken,
  h264RequestedVideoUrl,
  gbufferSubscribed: gbufferSubscribed(),
  gbufferPanelOpen: Boolean(modelInputsPanel && modelInputsPanel.open),
  modelInputsIdleTimerArmed: modelInputsIdleTimer !== null,
  modelInputsIdleDeadline,
  lastGbufferSeq,
  lastGbufferPayloadAt,
  gbufferFrameSamples: gbufferFrames.length,
  streamSocketGeneration,
  streamHelloSocketGeneration,
  activeStreamConnectionGeneration,
  unfencedModelBinaryCount,
  serverInstanceId,
  retiredServerInstanceIds: Array.from(retiredServerInstanceIds),
  lastServerInstanceObservation: lastServerInstanceObservation
    ? { ...lastServerInstanceObservation }
    : null,
  lastInputAckTransport,
  serverInputSuspended,
  streamTransport,
  lastStreamClose,
  streamSocketReadyState: streamSocket ? streamSocket.readyState : null,
  streamReconnectAttempt,
  websocketRecoveryPending,
  lastStreamConfig: lastStreamConfig ? { ...lastStreamConfig } : null,
  streamConfigHistory: streamConfigHistory.map(item => ({ ...item }))
});
if (new URLSearchParams(location.search).get('candidate_test') === '1') {
  window.__alayaInputTest = {
    setModelLoadHold(value = true) {
      holdModelLoadCommit = Boolean(value);
    },
    flushHeldModelLoad() {
      const candidate = heldModelLoad;
      heldModelLoad = null;
      holdModelLoadCommit = false;
      if (candidate) {
        commitModelFrameLoaded(candidate.seq, candidate.identity, candidate);
      }
    },
    attemptStaleModelLoad(seq, identity) {
      return commitRenderedLoadToken(
        renderedDecodeGeneration - 1,
        renderedObjectUrl,
        renderedObjectUrl,
        {
          seq,
          identity,
          exactMeta: true,
          socketGeneration: streamSocketGeneration,
          connectionGeneration: activeStreamConnectionGeneration,
          frameToken: loadedModelFrame && loadedModelFrame.frameToken
            ? loadedModelFrame.frameToken
            : 1
        }
      );
    },
    attemptCurrentModelLoad(seq, identity) {
      return commitRenderedLoadToken(
        renderedDecodeGeneration,
        renderedObjectUrl,
        renderedObjectUrl,
        {
          seq,
          identity,
          exactMeta: true,
          socketGeneration: streamSocketGeneration,
          connectionGeneration: activeStreamConnectionGeneration,
          frameToken: loadedModelFrame && loadedModelFrame.frameToken
            ? loadedModelFrame.frameToken
            : 1
        }
      );
    },
    reconcileInputState(input) {
      return reconcileInputAuthority({ input });
    },
    attemptUnfencedModelLoad(seq, identity) {
      const before = loadedModelFrame;
      markModelFrameLoaded(seq, identity, {});
      return loadedModelFrame !== before;
    },
    reconnectStream() {
      if (streamSocket && streamSocket.readyState === WebSocket.OPEN) {
        streamSocket.send(JSON.stringify({ type: 'test_close' }));
      } else {
        scheduleStreamReconnect(true);
      }
    },
    requestManualHeartbeatWithoutOwner() {
      applyLocalControlMode('manual');
      sendInputState('heartbeat');
    },
    beginHardReset(reason = 'test-reset') {
      beginHardReset(reason, sourceIdentityFromState());
    }
  };
}
updateKeys(false);
videoQualityMode = videoQualitySelect.value === 'auto' ? 'auto' : 'manual';
videoQuality = videoQualityMode === 'auto'
  ? 'balanced'
  : (videoQualitySelect.value in videoProfiles ? videoQualitySelect.value : 'balanced');
gbufferRateMode = gbufferRateSelect.value === 'auto' ? 'auto' : 'manual';
effectiveGbufferFps = gbufferRateMode === 'auto' ? 4 : Math.max(0, Math.min(8, Number(gbufferRateSelect.value) || 0));
gbufferSizeMode = gbufferSizeSelect.value === 'auto' ? 'auto' : 'manual';
effectiveGbufferWidth = gbufferSizeMode === 'auto' ? 360 : Math.max(160, Math.min(640, Number(gbufferSizeSelect.value) || 360));
text(videoProfileValue, profileText());
setInterval(inputHeartbeat, 250);
setInterval(tuneVideoLatency, 500);
setInterval(recoverGbufferIfStalled, 1000);
setInterval(refreshRecordStatus, 3000);
setInterval(() => refreshStylePrompt().catch(() => {}), 1500);
updateGbufferTitle();
refreshRecordStatus().catch(() => {});
refreshStylePrompt().catch(() => {});
setRenderTransport('websocket');
startWebSocketStream();
tick();
</script>
</body>
</html>
"""


def make_handler(
    capture_dir: Path,
    rendered_dir: Path | None,
    main_width: int,
    gbuffer_width: int,
    jpeg_quality: int,
    max_fps: float,
    gbuffer_fps: float,
    input_state_path: Path | None = None,
    read_only_observer: bool = False,
):
    control_root = os.environ.get("STK_PROMPT_CONTROL_DIR")
    status_path = Path(control_root) / "worker_status.json" if control_root else None

    def current_nominal_fps():
        raw = os.environ.get("ALAYA_PUBLISH_SOURCE_FPS")
        runtime_root = os.environ.get("ALAYA_RUNTIME_ROOT")
        if runtime_root:
            try:
                manifest = json.loads((Path(runtime_root) / "deployment-manifest.json").read_text())
                raw = manifest["transport"]["target_fps"]
            except (OSError, ValueError, KeyError, TypeError):
                pass
        return configured_presentation_fps(raw, max_fps, status_path)

    def current_video_profiles():
        profiles = {name: dict(profile) for name, profile in VIDEO_PROFILES.items()}
        for name in ("balanced", "quality"):
            profiles[name]["fps"] = max(12, int(round(current_nominal_fps())))
        return profiles

    server_instance_id = secrets.token_hex(16)
    server_started_monotonic = time.monotonic()
    try:
        ws_startup_delay_s = max(
            0.0,
            float(os.environ.get("STK_WS_STARTUP_DELAY_S", "0") or 0),
        )
    except (TypeError, ValueError):
        ws_startup_delay_s = 0.0
    try:
        state_startup_delay_s = max(
            0.0,
            float(os.environ.get("STK_STATE_STARTUP_DELAY_S", "0") or 0),
        )
    except (TypeError, ValueError):
        state_startup_delay_s = 0.0
    input_audit_value = str(os.environ.get("STK_INPUT_AUDIT_PATH", "") or "").strip()
    input_audit_path = Path(input_audit_value) if input_audit_value else None
    cache: dict[str, dict[str, object]] = {}
    cache_lock = threading.Lock()
    input_state: dict[str, object] = {
        "mode": "auto",
        "keys": [],
        "t": 0,
        "source": "init",
        "client_id": "",
        "intent": "init",
        "input_seq": -1,
        "input_event_id": "",
        "boundary_source_seq": -1,
        "boundary_source_frame": -1,
        "boundary_pending": False,
        "server_instance_id": server_instance_id,
    }
    input_lock = threading.RLock()
    input_owner: dict[str, object] = {
        "client_id": "",
        "lease_until": 0.0,
        "timer": None,
    }
    input_clients: dict[str, dict[str, object]] = {}
    input_event_history: list[dict[str, object]] = [
        {
            "mode": "auto",
            "keys": [],
            "input_event_id": "init:auto",
            "boundary_source_seq": -1,
            "track_id": "",
            "source_session_id": "",
            "reset_epoch": -1,
            "source_reset_epoch": -1,
        }
    ]
    input_server_event_counter = 0
    input_boundary_watermark: dict[str, object] = {}
    input_owner_lease_s = 2.5
    input_path = (input_state_path or capture_dir / "input.json").resolve()
    observer_input_mtime_ns = -1
    shared_video_streams: dict[str, object] = {}
    shared_video_lock = threading.Lock()
    project_dir = Path(os.environ.get("ALAYA_PREVIEW_REFERENCE_ROOT", str(Path(__file__).resolve().parent))).resolve()
    alaya_mark_path = Path(
        os.environ.get(
            "STK_ALAYA_MARK_PATH",
            str(project_dir / "assets" / "alaya_renderer_mark.png"),
        )
    )
    record_status_path = project_dir / ".runtime" / "stk_pair_recording.status"
    record_launcher_pid_path = project_dir / ".runtime" / "stk_pair_recording_launcher.pid"
    record_script_path = project_dir / "scripts" / "start_stk_pair_recording_2h.sh"
    record_final_root = Path(os.environ.get("STK_RECORD_FINAL_ROOT", str(project_dir / ".runtime" / "recordings")))
    record_tmp_root = Path(os.environ.get("STK_RECORD_TMP_ROOT", str(project_dir / ".runtime" / "recordings_tmp")))
    prompt_control_dir = Path(
        os.environ.get(
            "STK_PROMPT_CONTROL_DIR",
            str(project_dir / ".runtime" / "style_control"),
        )
    )
    prompt_queue_path = prompt_control_dir / "prompt_queue.jsonl"
    prompt_state_path = prompt_control_dir / "style_prompt.json"
    prompt_live_control_path = prompt_control_dir / "live_control.json"
    worker_status_path = prompt_control_dir / "worker_status.json"
    window_dir = Path(os.environ.get("STK_ALAYA_WINDOW_DIR", "/dev/shm/stk_alaya_windows"))
    bridge_manifest_path = window_dir / "manifest.json"
    prompt_lock = threading.Lock()
    prompt_last_submit: dict[str, float] = {}
    prompt_cooldown_s = 3.0
    track_options = {
        "black_forest": "Black Forest",
        "candela_city": "Candela City",
        "cocoa_temple": "Cocoa Temple",
        "snowmountain": "Northern Resort",
        "volcano_island": "Volcan Island",
        "zengarden": "Zen Garden",
    }
    track_script_path = Path(
        os.environ.get(
            "STK_TRACK_SCRIPT_PATH",
            str(project_dir / "scripts" / "run_stk_capture.sh"),
        )
    )
    track_pid_path = Path(
        os.environ.get(
            "STK_TRACK_PID_PATH",
            str(project_dir / ".runtime" / "stk_live_preview_stk.pid"),
        )
    )
    track_log_path = Path(
        os.environ.get(
            "STK_TRACK_LOG_PATH",
            str(project_dir / "logs" / "stk_live_preview_stk.log"),
        )
    )
    track_switch_lock = threading.Lock()
    track_switch_state: dict[str, object] = {
        "status": "ready",
        "requested_track": "",
        "previous_track": "",
        "error": "",
        "renderer_reset_error": "",
        "started_at": 0.0,
        "finished_at": 0.0,
    }
    track_switch_cooldown_s = 5.0
    try:
        track_stop_grace_s = max(
            0.1,
            min(8.0, float(os.environ.get("STK_TRACK_STOP_GRACE_S", "0.5"))),
        )
    except (TypeError, ValueError):
        track_stop_grace_s = 0.5
    try:
        track_kill_wait_s = max(
            0.1,
            min(5.0, float(os.environ.get("STK_TRACK_KILL_WAIT_S", "2.0"))),
        )
    except (TypeError, ValueError):
        track_kill_wait_s = 2.0
    ffmpeg_path: str | None = None
    video_encoder: str | None = None
    video_encoder_reason = "ffmpeg not found"

    def ffmpeg_candidate_inputs() -> list[object]:
        return [
            os.environ.get("STK_VIDEO_FFMPEG"),
            shutil.which("ffmpeg"),
            project_dir / ".video_env" / "ffmpeg_cf" / "bin" / "ffmpeg",
            *FFMPEG_CANDIDATES,
        ]

    def find_ffmpeg_path() -> str | None:
        seen: set[str] = set()
        for candidate in ffmpeg_candidate_inputs():
            if not candidate:
                continue
            path = str(Path(candidate))
            if path in seen:
                continue
            seen.add(path)
            if Path(path).is_file() and os_access_executable(path):
                return path
        return None

    def ffmpeg_candidate_paths() -> list[str]:
        seen: set[str] = set()
        paths: list[str] = []
        for candidate in ffmpeg_candidate_inputs():
            if not candidate:
                continue
            path = str(Path(candidate))
            if path in seen:
                continue
            seen.add(path)
            if Path(path).is_file() and os_access_executable(path):
                paths.append(path)
        return paths

    def os_access_executable(path: str) -> bool:
        try:
            return Path(path).is_file() and bool(Path(path).stat().st_mode & 0o111)
        except Exception:
            return False

    def ffmpeg_supports_encoder(path: str, encoder: str) -> bool:
        try:
            proc = subprocess.run(
                [path, "-hide_banner", "-encoders"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=8,
                check=False,
            )
        except Exception:
            return False
        return encoder in (proc.stdout + proc.stderr)

    def ffmpeg_encoder_works(path: str, encoder: str) -> tuple[bool, str]:
        if encoder == "h264_nvenc":
            encode_args = [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p1",
                "-tune",
                "ull",
                "-rc",
                "cbr",
                "-b:v",
                "1M",
                "-g",
                "1",
                "-bf",
                "0",
            ]
        elif encoder == "libx264":
            encode_args = [
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-g",
                "1",
                "-bf",
                "0",
            ]
        else:
            encode_args = ["-c:v", encoder]
        cmd = [
            path,
            "-hide_banner",
            "-loglevel",
            "verbose",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=1",
            "-frames:v",
            "1",
            "-an",
            *encode_args,
            "-f",
            "null",
            "-",
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=8,
                check=False,
            )
        except Exception as exc:
            return False, repr(exc)
        output = (proc.stderr or "") + (proc.stdout or "")
        lines = output.strip().splitlines()
        lower_output = output.lower()
        interesting = [
            line
            for line in lines
            if any(
                marker in line.lower()
                for marker in (
                    "nvenc",
                    "openencodesessionex",
                    "unsupported",
                    "no capable",
                    "cannot load",
                    "permission",
                    "denied",
                    "error",
                )
            )
        ]
        if "unsupported device" in lower_output and not any(
            "unsupported device" in line.lower() for line in interesting[:3]
        ):
            interesting.insert(
                0, "OpenEncodeSessionEx failed: unsupported device; no capable NVENC encode session is available"
            )
        detail = interesting or lines
        if "unsupported device" in lower_output:
            head = [line for line in detail if "unsupported device" in line.lower() or "no capable" in line.lower()]
            tail = detail[-6:]
            merged: list[str] = []
            for line in [*head, *tail]:
                if line and line not in merged:
                    merged.append(line)
            return proc.returncode == 0, "\n".join(merged)
        return proc.returncode == 0, "\n".join(detail[-10:])

    def select_video_encoder() -> tuple[str | None, str | None, str]:
        fallback: tuple[str | None, str | None, str] = (None, None, "ffmpeg not found")
        for path in ffmpeg_candidate_paths():
            if ffmpeg_supports_encoder(path, "h264_nvenc"):
                ok, detail = ffmpeg_encoder_works(path, "h264_nvenc")
                if ok:
                    return path, "h264_nvenc", "h264_nvenc verified"
                if "unsupported device" in detail.lower():
                    fallback = (
                        path,
                        None,
                        "h264_nvenc unavailable: NVIDIA device reports unsupported NVENC encode session; using CPU H.264. "
                        + (detail or "encoder test failed"),
                    )
                else:
                    fallback = (path, None, f"h264_nvenc unavailable: {detail or 'encoder test failed'}")
            if ffmpeg_supports_encoder(path, "libx264"):
                ok, detail = ffmpeg_encoder_works(path, "libx264")
                if ok:
                    reason = (
                        f"{fallback[2]}; using libx264 verified"
                        if fallback[1] is None and "h264_nvenc" in fallback[2]
                        else "libx264 verified"
                    )
                    return path, "libx264", reason
                fallback = (path, None, f"libx264 unavailable: {detail or 'encoder test failed'}")
        return fallback

    ffmpeg_path, video_encoder, video_encoder_reason = select_video_encoder()

    def shared_video_payload() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        with shared_video_lock:
            for stale_key, stream in list(shared_video_streams.items()):
                is_alive = bool(getattr(stream, "is_alive", lambda: False)())
                if not is_alive and int(getattr(stream, "clients", 0)) <= 0:
                    shared_video_streams.pop(stale_key, None)
            streams = list(shared_video_streams.values())
        for stream in streams:
            profile = getattr(stream, "profile", {})
            rows.append(
                {
                    "key": getattr(stream, "key", ""),
                    "profile": profile.get("name", "") if isinstance(profile, dict) else "",
                    "width": profile.get("width", 0) if isinstance(profile, dict) else 0,
                    "fps": profile.get("fps", 0) if isinstance(profile, dict) else 0,
                    "bitrate_kbps": profile.get("bitrate_kbps", 0) if isinstance(profile, dict) else 0,
                    "x264_preset": profile.get("x264_preset", "") if isinstance(profile, dict) else "",
                    "x264_threads": profile.get("x264_threads", 0) if isinstance(profile, dict) else 0,
                    "clients": getattr(stream, "clients", 0),
                    "alive": bool(getattr(stream, "is_alive", lambda: False)()),
                    "init_ready": bool(getattr(stream, "init_ready", False)),
                    "media_boxes": len(getattr(stream, "media_boxes", [])),
                    "age_s": round(time.time() - float(getattr(stream, "started_at", time.time())), 1),
                    "last_error": getattr(stream, "last_error", ""),
                }
            )
        return rows

    def video_info_payload() -> dict[str, object]:
        return {
            "available": bool(ffmpeg_path and video_encoder),
            "ffmpeg_path": ffmpeg_path or "",
            "encoder": video_encoder or "",
            "reason": video_encoder_reason,
            "candidates": ffmpeg_candidate_paths(),
            "default_profile": "balanced",
            "profiles": current_video_profiles(),
            "max_fps": 1.0 / primary_stream_interval(max_fps),
            "shared_streams": shared_video_payload(),
        }

    def video_state_payload() -> dict[str, object]:
        return {
            "available": bool(ffmpeg_path and video_encoder),
            "encoder": video_encoder or "",
            "reason": video_encoder_reason[:240],
            "max_fps": 1.0 / primary_stream_interval(max_fps),
        }

    def int_query(params: dict[str, list[str]], key: str, default: int, lo: int, hi: int) -> int:
        try:
            value = int(float(params.get(key, [str(default)])[0]))
        except Exception:
            value = default
        return max(lo, min(hi, value))

    def video_profile_from_request(handler: BaseHTTPRequestHandler, source_width: int) -> dict[str, object]:
        query = parse_qs(urlparse(handler.path).query)
        profile_name = query.get("profile", ["balanced"])[0]
        video_profiles = current_video_profiles()
        profile = video_profiles.get(profile_name, video_profiles["balanced"])
        max_width = max(320, min(source_width, main_width if main_width > 0 else source_width, 1280))
        width = int_query(query, "w", int(profile["width"]), 320, max_width)
        fps = int_query(
            query, "fps", int(profile["fps"]), 12, max(12, min(int(MAX_PRIMARY_STREAM_FPS), int(round(max_fps))))
        )
        bitrate = int_query(query, "kbps", int(profile["bitrate_kbps"]), 800, 12000)
        return {
            "name": profile_name if profile_name in video_profiles else "balanced",
            "label": profile["label"],
            "width": width,
            "fps": fps,
            "bitrate_kbps": bitrate,
            "x264_preset": profile["x264_preset"],
            "x264_threads": profile.get("x264_threads", 0),
            "x264_params": profile.get("x264_params", ""),
        }

    def read_key_value_file(path: Path) -> dict[str, str]:
        data: dict[str, str] = {}
        try:
            for line in path.read_text().splitlines():
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                data[key.strip()] = value.strip()
        except Exception:
            pass
        return data

    def pid_is_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            stat_path = Path("/proc") / str(pid) / "stat"
            if stat_path.exists():
                parts = stat_path.read_text().split()
                if len(parts) > 2 and parts[2] == "Z":
                    return False
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return False

    def parse_iso_time(value: str) -> float | None:
        if not value:
            return None
        try:
            from datetime import datetime

            return datetime.fromisoformat(value).timestamp()
        except Exception:
            return None

    def latest_record_log_frame(tmp_dir: str) -> int | None:
        path = Path(tmp_dir) / "recorder.log"
        try:
            lines = path.read_text(errors="replace").splitlines()[-200:]
        except Exception:
            return None
        latest: int | None = None
        for line in lines:
            marker = "frame="
            if marker not in line:
                continue
            try:
                latest = int(line.split(marker, 1)[1].split()[0])
            except Exception:
                pass
        return latest

    def recording_status_payload() -> dict[str, object]:
        status = read_key_value_file(record_status_path)
        pid = 0
        for key in ("launcher_pid", "pid"):
            try:
                pid = int(status.get(key, "0") or "0")
                if pid:
                    break
            except Exception:
                pid = 0
        if not pid:
            try:
                pid = int(record_launcher_pid_path.read_text().strip())
            except Exception:
                pid = 0
        active = pid_is_alive(pid)

        frames: object = status.get("frames", "")
        done_exists = False
        failed_exists = False
        if active:
            log_frame = latest_record_log_frame(status.get("tmp_dir", ""))
            if log_frame is not None:
                frames = log_frame
        elif status.get("final_dir"):
            done_path = Path(status["final_dir"]) / "done.json"
            failed_exists = (Path(status["final_dir"]) / "FAILED.txt").exists()
            try:
                done = json.loads(done_path.read_text())
                done_exists = True
                frames = done.get("frames", frames)
            except Exception:
                pass

        started_ts = parse_iso_time(status.get("started_at", ""))
        ended_ts = parse_iso_time(status.get("ended_at", ""))
        now = time.time()
        elapsed = None
        if started_ts is not None:
            elapsed = (now if active or ended_ts is None else ended_ts) - started_ts

        state = status.get("status", "idle")
        if active and state not in {"recording", "stopping"}:
            state = "recording"
        if not active and state in {"recording", "stopping"}:
            if done_exists:
                state = "stopped"
            elif failed_exists:
                state = "record_failed"
        return {
            "status": state,
            "active": active,
            "pid": pid,
            "run_name": status.get("run_name", ""),
            "tmp_dir": status.get("tmp_dir", ""),
            "final_dir": status.get("final_dir", ""),
            "started_at": status.get("started_at", ""),
            "ended_at": status.get("ended_at", ""),
            "duration_sec": float(status.get("duration_sec", "0") or 0),
            "fps": float(status.get("fps", "0") or 0),
            "keep_tmp": status.get("keep_tmp", ""),
            "tmp_removed": status.get("tmp_removed", ""),
            "manifest": status.get("manifest", ""),
            "frames": frames,
            "elapsed_sec": elapsed,
        }

    def clean_run_name(value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value.strip())
        return cleaned[:96]

    def start_recording_job(data: dict[str, object]) -> tuple[int, dict[str, object]]:
        if read_only_observer:
            return 403, {
                "ok": False,
                "error": "Recording is disabled in read-only observer mode",
            }
        current = recording_status_payload()
        if bool(current.get("active")):
            return 409, current

        cfg = read_run_config()
        track = clean_run_name(cfg.get("TRACK", "black_forest")) or "black_forest"
        default_run_name = f"stk_pair_{track}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_name = clean_run_name(str(data.get("run_name", "") or default_run_name))
        if not run_name:
            run_name = default_run_name
        try:
            duration = float(data.get("duration_sec", 7200))
        except Exception:
            duration = 7200.0
        try:
            fps = float(data.get("fps", 8))
        except Exception:
            fps = 8.0
        duration = max(1.0, min(28800.0, duration))
        fps = max(1.0, min(30.0, fps))
        keep_tmp = "1" if bool(data.get("keep_tmp", False)) else "0"

        record_final_root.mkdir(parents=True, exist_ok=True)
        record_tmp_root.mkdir(parents=True, exist_ok=True)
        (project_dir / "logs").mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "PROJECT_DIR": str(project_dir),
                "CAPTURE_DIR": str(capture_dir),
                "TMP_ROOT": str(record_tmp_root),
                "FINAL_ROOT": str(record_final_root),
                "RUN_NAME": run_name,
                "DURATION_SEC": str(duration),
                "FPS": str(fps),
                "KEEP_TMP": keep_tmp,
                "EST_BYTES_PER_FRAME": str(data.get("est_bytes_per_frame", 4000000)),
                "SPACE_SAFETY": str(data.get("space_safety", 1.25)),
            }
        )
        log_path = project_dir / "logs" / f"{run_name}.launcher.log"
        log_file = log_path.open("ab")
        subprocess.Popen(
            ["bash", str(record_script_path)],
            cwd=str(project_dir),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        payload = recording_status_payload()
        deadline = time.time() + 2.0
        while time.time() < deadline:
            payload = recording_status_payload()
            if payload.get("run_name") == run_name:
                break
            time.sleep(0.1)
        payload["launcher_log"] = str(log_path)
        return 200, payload

    def stop_recording_job() -> tuple[int, dict[str, object]]:
        if read_only_observer:
            return 403, {
                "ok": False,
                "error": "Recording is disabled in read-only observer mode",
            }
        status = recording_status_payload()
        pid = int(status.get("pid", 0) or 0)
        if not bool(status.get("active")) or pid <= 0:
            return 200, status
        try:
            record_status_path.write_text(
                "\n".join(
                    [
                        "status=stopping",
                        f"run_name={status.get('run_name', '')}",
                        f"tmp_dir={status.get('tmp_dir', '')}",
                        f"final_dir={status.get('final_dir', '')}",
                        f"launcher_pid={pid}",
                        f"started_at={status.get('started_at', '')}",
                        f"duration_sec={status.get('duration_sec', 0)}",
                        f"fps={status.get('fps', 0)}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
        time.sleep(0.2)
        return 200, recording_status_payload()

    def style_renderer_connected() -> bool:
        worker = read_json_object(worker_status_path)
        try:
            worker_fresh = time.time() - float(worker.get("updated_at", 0)) < 10.0
        except (TypeError, ValueError):
            worker_fresh = False
        return bool(worker_fresh and worker.get("ready"))

    def read_json_object(path: Path) -> dict[str, object]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    def next_control_int(payload: dict[str, object], key: str) -> int:
        try:
            return int(payload.get(key, 0) or 0) + 1
        except (TypeError, ValueError):
            return 1

    def control_int(payload: dict[str, object], key: str, default: int = 0) -> int:
        try:
            return int(payload.get(key, default) or 0)
        except (TypeError, ValueError):
            return default

    def same_control_int(left: dict[str, object], right: dict[str, object], key: str) -> bool:
        return control_int(left, key, -1) == control_int(right, key, -2)

    def current_bridge_manifest(live_control: dict[str, object]) -> dict[str, object]:
        manifest = read_json_object(bridge_manifest_path)
        if not manifest:
            return {}
        for key in ("reset_epoch", "render_request_id", "prompt_revision"):
            if not same_control_int(manifest, live_control, key):
                return {}
        expected_session = str(
            read_run_config().get("SOURCE_SESSION_ID", "") or live_control.get("source_session_id", "")
        )
        manifest_session = str(manifest.get("source_session_id", "") or "")
        if expected_session and manifest_session != expected_session:
            return {}
        return manifest

    def current_render_identity_payload() -> dict[str, object]:
        cfg = read_run_config()
        live_control = read_json_object(prompt_live_control_path)
        worker = read_json_object(worker_status_path)
        manifest = current_bridge_manifest(live_control)
        track_id = str(cfg.get("TRACK", "") or live_control.get("track_id", "") or "")
        source_session_id = str(cfg.get("SOURCE_SESSION_ID", "") or live_control.get("source_session_id", "") or "")
        source_reset_epoch = control_int(
            manifest,
            "source_reset_epoch",
            control_int(worker, "source_reset_epoch", -1),
        )
        return {
            "track_id": track_id,
            "source_session_id": source_session_id,
            "reset_epoch": control_int(live_control, "reset_epoch", -1),
            "source_reset_epoch": source_reset_epoch,
        }

    def valid_rendered_state() -> dict[str, object]:
        if rendered_dir is None:
            return {}
        rgb_path = rendered_dir / "rgb.rgba8"
        latest_path = rendered_dir / "latest.json"
        if not rgb_path.exists() or not latest_path.exists():
            return {}

        live_control = read_json_object(prompt_live_control_path)
        prompt = str(live_control.get("prompt", "") or "").strip()
        if len(prompt) < 3:
            return {}

        worker = read_json_object(worker_status_path)
        try:
            worker_fresh = time.time() - float(worker.get("updated_at", 0)) < 10.0
        except (TypeError, ValueError):
            worker_fresh = False
        if not worker_fresh or not worker.get("ready"):
            return {}

        rendered = read_json_object(latest_path)
        for key in ("reset_epoch", "render_request_id", "prompt_revision"):
            if not same_control_int(rendered, live_control, key):
                return {}
        if "source_reset_epoch" in worker and not same_control_int(rendered, worker, "source_reset_epoch"):
            return {}

        expected_session = str(
            read_run_config().get("SOURCE_SESSION_ID", "") or live_control.get("source_session_id", "")
        )
        rendered_session = str(rendered.get("source_session_id", "") or "")
        if expected_session and rendered_session != expected_session:
            return {}
        expected_track = str(read_run_config().get("TRACK", "") or live_control.get("track_id", ""))
        rendered_track = str(rendered.get("track_id", "") or "")
        if expected_track and rendered_track != expected_track:
            return {}
        bridge_manifest = current_bridge_manifest(live_control)
        bridge_source_epoch = control_int(bridge_manifest, "source_reset_epoch", -1)
        if bridge_source_epoch >= 0 and control_int(rendered, "source_reset_epoch", -2) != bridge_source_epoch:
            return {}
        try:
            width = int(rendered["width"])
            height = int(rendered["height"])
            if width <= 0 or height <= 0 or rgb_path.stat().st_size != width * height * 4:
                return {}
        except (KeyError, OSError, TypeError, ValueError):
            return {}
        return rendered

    def render_handover_payload(
        rendered_state: dict[str, object],
        current_input: dict[str, object],
    ) -> dict[str, object]:
        event_id = str(current_input.get("input_event_id", "") or "")
        model_event_id = str(rendered_state.get("input_event_id", "") or "")
        boundary_seq = control_int(current_input, "boundary_source_seq", -1)
        boundary_frame = control_int(current_input, "boundary_source_frame", -1)
        model_source_seq = control_int(rendered_state, "source_seq", -1)
        model_epoch = control_int(rendered_state, "source_reset_epoch", 0)
        target_epoch = control_int(current_input, "source_reset_epoch", 0)
        target_track = str(current_input.get("track_id", "") or "")
        target_session = str(current_input.get("source_session_id", "") or "")
        model_track = str(rendered_state.get("track_id", "") or "")
        model_session = str(rendered_state.get("source_session_id", "") or "")
        source_identity_ready = bool(
            rendered_state
            and (not target_track or model_track == target_track)
            and (not target_session or model_session == target_session)
            and (target_epoch <= 0 or model_epoch == target_epoch)
        )
        boundary_available = boundary_seq >= 0 and boundary_frame >= 0
        ready = bool(
            event_id
            and rendered_state
            and boundary_available
            and source_identity_ready
            and model_source_seq >= boundary_seq
            and model_event_id == event_id
        )
        if event_id and boundary_available and rendered_state and not source_identity_ready:
            phase = "waiting_for_epoch"
        elif event_id and boundary_available and not ready:
            phase = "waiting_for_model"
        elif ready:
            phase = "model_ready"
        else:
            phase = "idle"
        return {
            "event_id": event_id,
            "model_input_event_id": model_event_id,
            "phase": phase,
            "pending": bool(event_id and boundary_available and not ready),
            "ready": ready,
            "handled_at": current_input.get("t", 0),
            "input_seq": current_input.get("input_seq", -1),
            "boundary_source_seq": boundary_seq,
            "boundary_source_frame": boundary_frame,
            "model_source_seq": model_source_seq,
            "target_track_id": target_track,
            "target_source_session_id": target_session,
            "model_track_id": model_track,
            "model_source_session_id": model_session,
            "source_identity_ready": source_identity_ready,
            "target_source_reset_epoch": target_epoch,
            "model_source_reset_epoch": model_epoch,
            "bridge_source_reset_epoch": model_epoch,
            "source_seq": model_source_seq,
            "boundary_seq": boundary_seq,
        }

    def capture_reset_boundary() -> tuple[int, int]:
        try:
            state = read_latest_state()
            seq_value = state.get("seq", -1)
            seq = int(-1 if seq_value is None else seq_value)
            source_frame_value = state.get("source_frame", seq)
            source_frame = int(seq if source_frame_value is None else source_frame_value)
            return seq, source_frame
        except Exception:
            return -1, -1

    def bump_renderer_reset(
        reason: str,
        track: str,
        trigger_if_prompt: bool = True,
        source_fence: dict[str, object] | None = None,
    ) -> dict[str, object]:
        now = time.time()
        if source_fence is None:
            reset_source_seq, reset_source_frame = capture_reset_boundary()
        else:
            reset_source_seq = control_int(source_fence, "source_seq", -1)
            reset_source_frame = control_int(source_fence, "source_frame", reset_source_seq)
        with prompt_lock:
            live_control = read_json_object(prompt_live_control_path)
            prompt = str(live_control.get("prompt", "") or "").strip()
            cfg = read_run_config()
            source_track = str((source_fence or {}).get("track_id", "") or cfg.get("TRACK", "") or track)
            source_session_id = str(
                (source_fence or {}).get("source_session_id", "") or cfg.get("SOURCE_SESSION_ID", "")
            )
            if source_track != track:
                raise RuntimeError(f"capture source track {source_track!r} does not match {track!r}")
            live_control.update(
                {
                    "schema_version": 2,
                    "reset_epoch": next_control_int(live_control, "reset_epoch"),
                    "reset_reason": reason,
                    "track_id": source_track,
                    "source_session_id": source_session_id,
                    "reset_source_seq": reset_source_seq,
                    "reset_source_frame": reset_source_frame,
                    "reset_at": now,
                }
            )
            should_render = trigger_if_prompt and len(prompt) >= 3
            if should_render:
                live_control.update(
                    {
                        "render_request_id": next_control_int(live_control, "render_request_id"),
                        "request_reason": reason,
                        "render_request_ts": now,
                    }
                )
            write_json_atomic(prompt_live_control_path, live_control)

            prompt_state = read_json_object(prompt_state_path)
            prompt_state.update(
                {
                    "prompt": prompt,
                    "status": "queued" if should_render else "idle",
                    "reset_epoch": live_control["reset_epoch"],
                    "reset_reason": reason,
                    "track_id": track,
                    "render_request_id": live_control.get("render_request_id", 0),
                    "ts": now,
                }
            )
            write_json_atomic(prompt_state_path, prompt_state)
            return live_control

    def prompt_status_payload() -> dict[str, object]:
        payload = read_json_object(prompt_state_path)
        live_control = read_json_object(prompt_live_control_path)
        worker = read_json_object(worker_status_path)
        try:
            worker_fresh = time.time() - float(worker.get("updated_at", 0)) < 10.0
        except (TypeError, ValueError):
            worker_fresh = False
        worker_status = str(worker.get("status", "") or "")
        worker_matches_request = same_control_int(worker, live_control, "reset_epoch") and control_int(
            worker, "accepted_request_id"
        ) >= control_int(live_control, "render_request_id")
        if worker_fresh and (worker_status in {"loading", "warming"} or worker_matches_request):
            for key in (
                "status",
                "error",
                "ready",
                "compile_mode",
                "reset_epoch",
                "accepted_request_id",
                "completed_request_id",
                "output_fps",
                "source_reset_epoch",
                "worker_instance_id",
                "updated_at",
                "warmed",
            ):
                if key in worker:
                    payload[key] = worker[key]
        payload.setdefault("prompt", "")
        payload.setdefault("status", "idle")
        payload["renderer_connected"] = style_renderer_connected()
        return payload

    def submit_style_prompt(prompt: object, client_ip: str) -> tuple[int, dict[str, object]]:
        if read_only_observer:
            return 403, {
                "ok": False,
                "error": "Prompt submission is disabled in read-only observer mode",
            }
        now = time.time()
        last_submit = prompt_last_submit.get(client_ip, 0.0)
        if now - last_submit < prompt_cooldown_s:
            return 429, {
                "ok": False,
                "error": "cooldown",
                "retry_in": round(prompt_cooldown_s - (now - last_submit), 1),
            }

        clean_prompt = "".join(ch for ch in str(prompt or "").strip() if ch.isprintable())[:1200]
        if len(clean_prompt) < 3:
            return 400, {"ok": False, "error": "prompt too short"}

        prompt_id = hashlib.sha1(f"{now}:{client_ip}:{clean_prompt}".encode("utf-8")).hexdigest()[:10]
        entry: dict[str, object] = {
            "id": prompt_id,
            "prompt": clean_prompt,
            "ts": now,
        }
        reset_source_seq, reset_source_frame = capture_reset_boundary()
        with prompt_lock:
            prompt_control_dir.mkdir(parents=True, exist_ok=True)
            with prompt_queue_path.open("a", encoding="utf-8") as queue_file:
                queue_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

            live_control = read_json_object(prompt_live_control_path)
            request_id = next_control_int(live_control, "render_request_id")
            reset_epoch = next_control_int(live_control, "reset_epoch")
            prompt_revision = next_control_int(live_control, "prompt_revision")
            cfg = read_run_config()
            live_control.update(
                {
                    "schema_version": 2,
                    "prompt": clean_prompt,
                    "prompt_revision": prompt_revision,
                    "render_request_id": request_id,
                    "request_reason": "prompt",
                    "render_request_ts": now,
                    "reset_epoch": reset_epoch,
                    "reset_reason": "prompt",
                    "track_id": cfg.get("TRACK", ""),
                    "source_session_id": cfg.get("SOURCE_SESSION_ID", ""),
                    "reset_source_seq": reset_source_seq,
                    "reset_source_frame": reset_source_frame,
                    "requested_at": now,
                }
            )
            write_json_atomic(prompt_live_control_path, live_control)

            state = {
                **entry,
                "status": "queued",
                "render_request_id": request_id,
                "prompt_revision": prompt_revision,
                "reset_epoch": reset_epoch,
                "reset_reason": "prompt",
            }
            write_json_atomic(prompt_state_path, state)

        prompt_last_submit[client_ip] = now
        return 200, {
            "ok": True,
            **state,
            "renderer_connected": style_renderer_connected(),
        }

    def clean_input_client_id(value: object, source: str) -> str:
        raw = str(value or "").strip()[:96]
        clean = "".join(char for char in raw if char.isalnum() or char in {"-", "_", ".", ":"})
        return clean or f"{source}:anonymous"

    def normalize_input_keys(keys: object) -> list[str]:
        if not isinstance(keys, list):
            return []
        return sorted({str(value)[:32] for value in keys if str(value)})[:32]

    def drive_input_signature(mode: str, keys: list[str]) -> tuple[object, ...]:
        key_set = set(keys)
        left = bool(key_set & {"ArrowLeft", "KeyA"})
        right = bool(key_set & {"ArrowRight", "KeyD"})
        steer = -1 if left and not right else 1 if right and not left else 0
        return (
            mode,
            steer,
            bool(key_set & {"ArrowUp", "KeyW"}),
            bool(key_set & {"ArrowDown", "KeyS"}),
            "Space" in key_set,
            bool(key_set & {"ShiftLeft", "ShiftRight"}),
            "KeyR" in key_set,
        )

    def next_server_input_event_id(reason: str) -> str:
        nonlocal input_server_event_counter
        input_server_event_counter += 1
        return format_server_input_event_id(server_instance_id, reason, input_server_event_counter)

    def next_input_boundary() -> tuple[int, int]:
        source_seq, source_frame = capture_reset_boundary()
        identity = current_render_identity_payload()
        return coalesced_input_boundary(
            source_seq,
            source_frame,
            (
                str(identity.get("track_id", "") or ""),
                str(identity.get("source_session_id", "") or ""),
                control_int(identity, "reset_epoch", -1),
                control_int(identity, "source_reset_epoch", -1),
            ),
            input_boundary_watermark,
        )

    def owner_payload_locked() -> dict[str, object]:
        owner_client_id = str(input_owner.get("client_id", "") or "")
        lease_until = float(input_owner.get("lease_until", 0.0) or 0.0)
        return {
            "owner_client_id": owner_client_id,
            "owner_lease_ms": (max(0.0, (lease_until - time.monotonic()) * 1000.0) if owner_client_id else 0.0),
        }

    def decorate_input_ack_locked(ack: dict[str, object], duplicate: bool = False) -> dict[str, object]:
        result = dict(ack)
        result.update(owner_payload_locked())
        result.setdefault("server_instance_id", server_instance_id)
        for key, value in current_render_identity_payload().items():
            result.setdefault(key, value)
        result["duplicate"] = duplicate
        result["effective"] = bool(
            result.get("accepted")
            and result.get("input_event_id")
            and result.get("input_event_id") == input_state.get("input_event_id")
        )
        return result

    def remember_input_ack_locked(
        client_id: str,
        input_seq: int,
        fingerprint: tuple[object, ...],
        ack: dict[str, object],
    ) -> None:
        stored_ack = dict(ack)
        for key, value in current_render_identity_payload().items():
            stored_ack.setdefault(key, value)
        previous = input_clients.get(client_id, {})
        ack_history = dict(previous.get("acks", {}))
        ack_history[input_seq] = {
            "fingerprint": fingerprint,
            "ack": stored_ack,
        }
        if len(ack_history) > 16:
            for stale_seq in sorted(ack_history)[:-16]:
                ack_history.pop(stale_seq, None)
        input_clients[client_id] = {
            "last_seq": input_seq,
            "fingerprint": fingerprint,
            "ack": stored_ack,
            "acks": ack_history,
            "updated_at": time.monotonic(),
        }
        if len(input_clients) <= 512:
            return
        stale_client = min(
            input_clients,
            key=lambda value: float(input_clients[value].get("updated_at", 0.0)),
        )
        if stale_client != client_id:
            input_clients.pop(stale_client, None)

    def write_input_state_locked(
        mode: str,
        keys: list[str],
        source: str,
        client_id: str,
        intent: str,
        input_seq: int,
        input_event_id: str,
        boundary_source_seq: int,
        boundary_source_frame: int,
        render_identity_override: dict[str, object] | None = None,
    ) -> None:
        now = time.time()
        render_identity = current_render_identity_payload()
        if render_identity_override:
            for key in (
                "track_id",
                "source_session_id",
                "reset_epoch",
                "source_reset_epoch",
            ):
                if key in render_identity_override:
                    render_identity[key] = render_identity_override[key]
        previous_signature = drive_input_signature(
            str(input_state.get("mode", "auto") or "auto"),
            normalize_input_keys(input_state.get("keys", [])),
        )
        next_signature = drive_input_signature(mode, keys)
        input_state.update(
            {
                "mode": mode,
                "keys": keys,
                "t": now,
                "server_time": now,
                "server_instance_id": server_instance_id,
                "source": source,
                "client_id": client_id,
                "intent": intent,
                "input_seq": input_seq,
                "input_event_id": input_event_id,
                "boundary_source_seq": boundary_source_seq,
                "boundary_source_frame": boundary_source_frame,
                "boundary_pending": False,
                **render_identity,
            }
        )
        if previous_signature != next_signature and input_event_id and boundary_source_seq >= 0:
            input_event_history.append(
                {
                    "mode": mode,
                    "keys": list(keys),
                    "input_event_id": input_event_id,
                    "boundary_source_seq": boundary_source_seq,
                    **render_identity,
                }
            )
            del input_event_history[:-128]
        payload = {
            "mode": mode,
            "keys": keys,
            "server_time": now,
            "server_instance_id": server_instance_id,
            "source": source,
            "client_id": client_id,
            "intent": intent,
            "input_seq": input_seq,
            "input_event_id": input_event_id,
            "boundary_source_seq": boundary_source_seq,
            "boundary_source_frame": boundary_source_frame,
            **render_identity,
        }
        with input_state_commit_lock(input_path):
            tmp_path = input_path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(payload), encoding="utf-8")
            tmp_path.replace(input_path)
        if input_audit_path is not None:
            try:
                input_audit_path.parent.mkdir(parents=True, exist_ok=True)
                with input_audit_path.open("a", encoding="utf-8") as audit_file:
                    audit_file.write(json.dumps(payload, separators=(",", ":")))
                    audit_file.write("\n")
            except Exception:
                pass

    def cancel_input_owner_timer_locked() -> None:
        timer = input_owner.get("timer")
        if isinstance(timer, threading.Timer):
            timer.cancel()
        input_owner["timer"] = None

    def clear_input_owner_locked() -> None:
        cancel_input_owner_timer_locked()
        input_owner["client_id"] = ""
        input_owner["lease_until"] = 0.0

    def expire_input_owner(expected_client_id: str, expected_lease_until: float) -> None:
        with input_lock:
            if input_owner.get("client_id") != expected_client_id:
                return
            lease_until = float(input_owner.get("lease_until", 0.0) or 0.0)
            if lease_until != expected_lease_until:
                return
            remaining = lease_until - time.monotonic()
            if remaining > 0.01:
                timer = threading.Timer(
                    remaining,
                    expire_input_owner,
                    args=(expected_client_id, expected_lease_until),
                )
                timer.daemon = True
                input_owner["timer"] = timer
                timer.start()
                return
            clear_input_owner_locked()
            boundary_source_seq, boundary_source_frame = next_input_boundary()
            write_input_state_locked(
                "auto",
                [],
                "lease-expired",
                expected_client_id,
                "timeout",
                int(input_state.get("input_seq", -1)),
                next_server_input_event_id("lease-expired"),
                boundary_source_seq,
                boundary_source_frame,
            )

    def schedule_input_owner_expiry_locked(client_id: str, lease_until: float) -> None:
        cancel_input_owner_timer_locked()
        timer = threading.Timer(
            max(0.01, lease_until - time.monotonic()),
            expire_input_owner,
            args=(client_id, lease_until),
        )
        timer.daemon = True
        input_owner["timer"] = timer
        timer.start()

    def track_input_gate_payload() -> dict[str, object]:
        with track_switch_lock:
            status = str(track_switch_state.get("status", "ready") or "ready")
        return {
            "input_suspended": status in {"switching", "restoring"},
            "input_suspended_reason": ("track_switch_in_progress" if status in {"switching", "restoring"} else ""),
            "track_switch_status": status,
        }

    def force_auto_input_locked(
        reason: str,
        *,
        reset_clients: bool = False,
        source_fence: dict[str, object] | None = None,
        clear_boundary: bool = False,
    ) -> dict[str, object]:
        clear_input_owner_locked()
        if reset_clients:
            input_clients.clear()
            input_event_history.clear()
        if clear_boundary:
            input_boundary_watermark.clear()
            boundary_source_seq, boundary_source_frame = -1, -1
        elif source_fence is None:
            boundary_source_seq, boundary_source_frame = next_input_boundary()
        else:
            source_seq = control_int(source_fence, "source_seq", -1)
            source_frame = control_int(source_fence, "source_frame", source_seq)
            boundary_source_seq = source_seq + 1 if source_seq >= 0 else -1
            boundary_source_frame = source_frame + 1 if source_frame >= 0 else -1
        input_event_id = next_server_input_event_id(reason)
        write_input_state_locked(
            "auto",
            [],
            reason,
            f"server:{reason}",
            "release",
            -1,
            input_event_id,
            boundary_source_seq,
            boundary_source_frame,
            source_fence,
        )
        identity = current_render_identity_payload()
        if source_fence:
            for key in (
                "track_id",
                "source_session_id",
                "reset_epoch",
                "source_reset_epoch",
            ):
                if key in source_fence:
                    identity[key] = source_fence[key]
        if not input_event_history or input_event_history[-1].get("input_event_id") != input_event_id:
            input_event_history.append(
                {
                    "mode": "auto",
                    "keys": [],
                    "input_event_id": input_event_id,
                    "boundary_source_seq": boundary_source_seq,
                    **identity,
                }
            )
        del input_event_history[:-128]
        return {
            "input_event_id": input_event_id,
            "boundary_source_seq": boundary_source_seq,
            "boundary_source_frame": boundary_source_frame,
            **identity,
        }

    def force_auto_input(
        reason: str,
        *,
        reset_clients: bool = False,
        source_fence: dict[str, object] | None = None,
        clear_boundary: bool = False,
    ) -> dict[str, object]:
        with input_lock:
            return force_auto_input_locked(
                reason,
                reset_clients=reset_clients,
                source_fence=source_fence,
                clear_boundary=clear_boundary,
            )

    def refresh_observer_input_locked() -> None:
        nonlocal observer_input_mtime_ns
        if not read_only_observer:
            return
        try:
            observed_mtime_ns = input_path.stat().st_mtime_ns
        except OSError:
            return
        if observed_mtime_ns == observer_input_mtime_ns:
            return
        payload = read_json_object(input_path)
        if not payload:
            return
        clean_mode = str(payload.get("mode", "auto") or "auto")
        if clean_mode not in {"auto", "manual"}:
            clean_mode = "auto"
        clean_keys = normalize_input_keys(payload.get("keys", []))
        if clean_mode == "auto":
            clean_keys = []
        mirrored_server_instance_id = str(payload.get("server_instance_id", "") or "")
        input_state.update(payload)
        input_state.update(
            {
                "mode": clean_mode,
                "keys": clean_keys,
                "server_instance_id": server_instance_id,
                "mirrored_server_instance_id": mirrored_server_instance_id,
                "read_only_observer": True,
            }
        )
        input_event_id = str(payload.get("input_event_id", "") or "")
        if input_event_id and (
            not input_event_history or input_event_history[-1].get("input_event_id") != input_event_id
        ):
            input_event_history.append(
                {
                    "mode": clean_mode,
                    "keys": list(clean_keys),
                    "input_event_id": input_event_id,
                    "boundary_source_seq": control_int(payload, "boundary_source_seq", -1),
                    "track_id": str(payload.get("track_id", "") or ""),
                    "source_session_id": str(payload.get("source_session_id", "") or ""),
                    "reset_epoch": control_int(payload, "reset_epoch", -1),
                    "source_reset_epoch": control_int(payload, "source_reset_epoch", -1),
                }
            )
            del input_event_history[:-128]
        observer_input_mtime_ns = observed_mtime_ns

    def update_input_state(
        mode: object,
        keys: object,
        source: str,
        client_id: object = "",
        intent: object = "heartbeat",
        input_seq: object = None,
        client_server_instance_id: object = "",
        force: bool = False,
    ) -> dict[str, object]:
        clean_mode = mode if mode in ("auto", "manual") else "auto"
        clean_keys = normalize_input_keys(keys)
        clean_client_id = clean_input_client_id(client_id, source)
        clean_intent = str(intent or "").strip().lower()
        if clean_intent not in {"heartbeat", "manual", "auto", "input", "release"}:
            clean_intent = "input" if clean_mode == "manual" and clean_keys else "heartbeat"
        if clean_intent == "release":
            clean_mode = "auto"
            clean_keys = []
        elif clean_mode == "auto":
            clean_keys = []

        with input_lock:
            refresh_observer_input_locked()
            client_record = input_clients.get(clean_client_id, {})
            last_input_seq = int(client_record.get("last_seq", -1))
            legacy_seq = False
            try:
                clean_input_seq = int(input_seq)
            except (TypeError, ValueError):
                clean_input_seq = last_input_seq + 1
                legacy_seq = True
            fingerprint = (
                clean_mode,
                tuple(clean_keys),
                clean_intent,
            )
            if read_only_observer and not force:
                return decorate_input_ack_locked(
                    {
                        "type": "input_ack",
                        "accepted": False,
                        "reason": "read_only_observer",
                        "retryable": False,
                        "client_id": clean_client_id,
                        "input_seq": clean_input_seq,
                        "last_input_seq": last_input_seq,
                        "sequence_watermark": last_input_seq,
                        "legacy_seq": legacy_seq,
                        "transition": False,
                        "authority_changed": False,
                        "input_event_id": str(input_state.get("input_event_id", "") or ""),
                        "boundary_source_seq": control_int(input_state, "boundary_source_seq", -1),
                        "boundary_source_frame": control_int(input_state, "boundary_source_frame", -1),
                        "boundary_available": control_int(input_state, "boundary_source_seq", -1) >= 0,
                        "mode": str(input_state.get("mode", "auto") or "auto"),
                        "keys": normalize_input_keys(input_state.get("keys", [])),
                        "intent": clean_intent,
                        "server_time": time.time(),
                    }
                )
            received_server_instance_id = str(client_server_instance_id or "")
            if not force and received_server_instance_id != server_instance_id:
                force_auto_input_locked("server-instance-mismatch")
                return decorate_input_ack_locked(
                    {
                        "type": "input_ack",
                        "accepted": False,
                        "reason": "server_instance_mismatch",
                        "retryable": False,
                        "client_id": clean_client_id,
                        "input_seq": clean_input_seq,
                        "last_input_seq": last_input_seq,
                        "sequence_watermark": last_input_seq,
                        "legacy_seq": legacy_seq,
                        "transition": False,
                        "authority_changed": False,
                        "input_event_id": str(input_state.get("input_event_id", "") or ""),
                        "boundary_source_seq": int(input_state.get("boundary_source_seq", -1)),
                        "boundary_source_frame": int(input_state.get("boundary_source_frame", -1)),
                        "boundary_available": bool(
                            int(input_state.get("boundary_source_seq", -1)) >= 0
                            and int(input_state.get("boundary_source_frame", -1)) >= 0
                        ),
                        "mode": "auto",
                        "keys": [],
                        "intent": clean_intent,
                        "received_server_instance_id": received_server_instance_id,
                        "server_time": time.time(),
                    }
                )
            gate = track_input_gate_payload()
            if gate["input_suspended"]:
                force_auto_input_locked("track-switch-gate", clear_boundary=True)
                return decorate_input_ack_locked(
                    {
                        "type": "input_ack",
                        "accepted": False,
                        "reason": "track_switch_in_progress",
                        "retryable": False,
                        "client_id": clean_client_id,
                        "input_seq": clean_input_seq,
                        "last_input_seq": last_input_seq,
                        "legacy_seq": legacy_seq,
                        "transition": False,
                        "authority_changed": False,
                        "input_event_id": str(input_state.get("input_event_id", "") or ""),
                        "boundary_source_seq": int(input_state.get("boundary_source_seq", -1)),
                        "boundary_source_frame": int(input_state.get("boundary_source_frame", -1)),
                        "mode": "auto",
                        "keys": [],
                        "server_time": time.time(),
                        **gate,
                    }
                )
            ack_history = client_record.get("acks", {})
            cached_entry = ack_history.get(clean_input_seq) if isinstance(ack_history, dict) else None
            if isinstance(cached_entry, dict):
                if fingerprint != cached_entry.get("fingerprint"):
                    return decorate_input_ack_locked(
                        {
                            "type": "input_ack",
                            "accepted": False,
                            "reason": "sequence_conflict",
                            "client_id": clean_client_id,
                            "input_seq": clean_input_seq,
                            "last_input_seq": last_input_seq,
                            "legacy_seq": legacy_seq,
                            "transition": False,
                            "authority_changed": False,
                            "input_event_id": "",
                            "boundary_source_seq": -1,
                            "boundary_source_frame": -1,
                            "server_time": time.time(),
                        }
                    )
                cached_ack = cached_entry.get("ack", {})
                if isinstance(cached_ack, dict):
                    return decorate_input_ack_locked(cached_ack, duplicate=True)
            if not force and clean_input_seq < last_input_seq:
                return decorate_input_ack_locked(
                    {
                        "type": "input_ack",
                        "accepted": False,
                        "reason": "stale_sequence",
                        "client_id": clean_client_id,
                        "input_seq": clean_input_seq,
                        "last_input_seq": last_input_seq,
                        "legacy_seq": legacy_seq,
                        "transition": False,
                        "authority_changed": False,
                        "input_event_id": "",
                        "boundary_source_seq": -1,
                        "boundary_source_frame": -1,
                        "server_time": time.time(),
                    }
                )
            if not force and clean_input_seq == last_input_seq:
                if fingerprint != client_record.get("fingerprint"):
                    return decorate_input_ack_locked(
                        {
                            "type": "input_ack",
                            "accepted": False,
                            "reason": "sequence_conflict",
                            "client_id": clean_client_id,
                            "input_seq": clean_input_seq,
                            "last_input_seq": last_input_seq,
                            "legacy_seq": legacy_seq,
                            "transition": False,
                            "authority_changed": False,
                            "input_event_id": "",
                            "boundary_source_seq": -1,
                            "boundary_source_frame": -1,
                            "server_time": time.time(),
                        }
                    )
                previous_ack = client_record.get("ack", {})
                if isinstance(previous_ack, dict):
                    return decorate_input_ack_locked(previous_ack, duplicate=True)

            now = time.monotonic()
            owner_client_id = str(input_owner.get("client_id", "") or "")
            lease_until = float(input_owner.get("lease_until", 0.0) or 0.0)
            owner_active = bool(owner_client_id and lease_until > now)
            if owner_client_id and not owner_active:
                clear_input_owner_locked()
                if input_state.get("mode") == "manual":
                    boundary_source_seq, boundary_source_frame = next_input_boundary()
                    write_input_state_locked(
                        "auto",
                        [],
                        "lease-expired",
                        owner_client_id,
                        "timeout",
                        int(input_state.get("input_seq", -1)),
                        next_server_input_event_id("lease-expired"),
                        boundary_source_seq,
                        boundary_source_frame,
                    )
                owner_client_id = ""
            previous_owner_client_id = owner_client_id if owner_active else ""

            rejection_reason = ""
            if force:
                if clean_mode == "auto":
                    clear_input_owner_locked()
            elif clean_mode == "manual":
                if owner_active and owner_client_id != clean_client_id:
                    can_take_ownership = clean_intent == "manual"
                else:
                    can_take_ownership = bool(
                        clean_intent == "manual"
                        or (clean_intent == "input" and clean_keys)
                        or (clean_intent == "heartbeat" and clean_keys and not owner_active)
                    )
                if owner_active and owner_client_id != clean_client_id and not can_take_ownership:
                    rejection_reason = "owned_by_other_client"
                if not owner_active and not can_take_ownership:
                    rejection_reason = "ownership_required"
            else:
                # Auto heartbeats from duplicate or stale tabs must not cancel an
                # active player. The Auto button is an explicit global release.
                if owner_active and clean_intent == "release" and owner_client_id != clean_client_id:
                    rejection_reason = "owned_by_other_client"
                elif owner_active and clean_intent not in {"auto", "release"}:
                    rejection_reason = "owned_by_other_client"

            if rejection_reason:
                ack = {
                    "type": "input_ack",
                    "accepted": False,
                    "reason": rejection_reason,
                    "client_id": clean_client_id,
                    "input_seq": clean_input_seq,
                    "last_input_seq": last_input_seq,
                    "sequence_watermark": clean_input_seq,
                    "legacy_seq": legacy_seq,
                    "transition": False,
                    "authority_changed": False,
                    "input_event_id": "",
                    "boundary_source_seq": -1,
                    "boundary_source_frame": -1,
                    "server_time": time.time(),
                }
                remember_input_ack_locked(clean_client_id, clean_input_seq, fingerprint, ack)
                return decorate_input_ack_locked(ack)

            previous_signature = drive_input_signature(
                str(input_state.get("mode", "auto") or "auto"),
                normalize_input_keys(input_state.get("keys", [])),
            )
            next_signature = drive_input_signature(clean_mode, clean_keys)
            boundary_retry = bool(
                input_state.get("boundary_pending")
                and str(input_state.get("client_id", "") or "") == clean_client_id
                and int(input_state.get("input_seq", -1)) == clean_input_seq
                and previous_signature == next_signature
            )
            transition = previous_signature != next_signature or boundary_retry
            boundary_source_seq = int(input_state.get("boundary_source_seq", -1))
            boundary_source_frame = int(input_state.get("boundary_source_frame", -1))
            input_event_id = str(input_state.get("input_event_id", "") or "")
            if transition:
                boundary_source_seq, boundary_source_frame = next_input_boundary()
                if not boundary_retry:
                    input_event_id = (
                        next_server_input_event_id(clean_intent) if force else f"{clean_client_id}:{clean_input_seq}"
                    )
                if boundary_source_seq < 0 or boundary_source_frame < 0:
                    safety_auto = clean_mode == "auto" and clean_intent in {"auto", "release"}
                    if safety_auto or force:
                        clear_input_owner_locked()
                        write_input_state_locked(
                            "auto",
                            [],
                            source,
                            clean_client_id,
                            clean_intent,
                            clean_input_seq,
                            input_event_id,
                            -1,
                            -1,
                        )
                        input_state["boundary_pending"] = not force
                    transient_ack = {
                        "type": "input_ack",
                        "accepted": bool(force),
                        "reason": ("forced_without_boundary" if force else "boundary_unavailable"),
                        "retryable": not force,
                        "client_id": clean_client_id,
                        "input_seq": clean_input_seq,
                        "last_input_seq": last_input_seq,
                        "sequence_watermark": last_input_seq,
                        "legacy_seq": legacy_seq,
                        "transition": True,
                        "authority_changed": bool(previous_owner_client_id and safety_auto),
                        "input_event_id": input_event_id,
                        "boundary_source_seq": -1,
                        "boundary_source_frame": -1,
                        "boundary_available": False,
                        "mode": clean_mode,
                        "keys": clean_keys,
                        "intent": clean_intent,
                        "server_time": time.time(),
                    }
                    return decorate_input_ack_locked(transient_ack)

            if clean_mode == "manual":
                lease_until = now + input_owner_lease_s
                input_owner["client_id"] = clean_client_id
                input_owner["lease_until"] = lease_until
                schedule_input_owner_expiry_locked(clean_client_id, lease_until)
            else:
                clear_input_owner_locked()

            next_owner_client_id = str(input_owner.get("client_id", "") or "")
            authority_changed = previous_owner_client_id != next_owner_client_id

            if not transition:
                input_event_id = (
                    (next_server_input_event_id(clean_intent) if force else f"{clean_client_id}:{clean_input_seq}")
                    if authority_changed
                    else str(input_state.get("input_event_id", "") or "")
                )
            write_input_state_locked(
                clean_mode,
                clean_keys,
                source,
                clean_client_id,
                clean_intent,
                clean_input_seq,
                input_event_id,
                boundary_source_seq,
                boundary_source_frame,
            )
            if (
                transition
                and boundary_source_seq >= 0
                and (not input_event_history or input_event_history[-1].get("input_event_id") != input_event_id)
            ):
                input_event_history.append(
                    {
                        "mode": clean_mode,
                        "keys": list(clean_keys),
                        "input_event_id": input_event_id,
                        "boundary_source_seq": boundary_source_seq,
                        **current_render_identity_payload(),
                    }
                )
                del input_event_history[:-128]
            ack = {
                "type": "input_ack",
                "accepted": True,
                "reason": "accepted",
                "client_id": clean_client_id,
                "input_seq": clean_input_seq,
                "last_input_seq": clean_input_seq,
                "legacy_seq": legacy_seq,
                "retryable": False,
                "transition": transition,
                "authority_changed": authority_changed,
                "input_event_id": input_event_id,
                "boundary_source_seq": boundary_source_seq,
                "boundary_source_frame": boundary_source_frame,
                "boundary_available": True,
                "mode": clean_mode,
                "keys": clean_keys,
                "intent": clean_intent,
                "server_time": time.time(),
            }
            if not force:
                remember_input_ack_locked(clean_client_id, clean_input_seq, fingerprint, ack)
            return decorate_input_ack_locked(ack)

    def input_state_snapshot() -> dict[str, object]:
        with input_lock:
            refresh_observer_input_locked()
            snapshot = dict(input_state)
            snapshot.update(owner_payload_locked())
            snapshot.update(track_input_gate_payload())
            if read_only_observer:
                snapshot["read_only_observer"] = True
            return snapshot

    def rendered_control_mode(
        source_seq: object,
        rendered_identity: dict[str, object],
    ) -> str:
        try:
            clean_source_seq = int(source_seq)
        except (TypeError, ValueError):
            return ""
        expected = current_render_identity_payload()
        expected.update(rendered_identity)
        selected = "auto"
        selected_specific_event = False
        with input_lock:
            for event in input_event_history:
                boundary = control_int(event, "boundary_source_seq", -1)
                event_mode = str(event.get("mode", "") or "")
                if event_mode == "auto" and boundary >= 0:
                    boundary += 1
                if boundary > clean_source_seq:
                    continue
                event_track = str(event.get("track_id", "") or "")
                event_session = str(event.get("source_session_id", "") or "")
                if event_track and event_track != str(expected.get("track_id", "") or ""):
                    continue
                if event_session and event_session != str(expected.get("source_session_id", "") or ""):
                    continue
                selected = event_mode
                selected_specific_event = bool(event_track or event_session)
            if not selected_specific_event:
                current_track = str(input_state.get("track_id", "") or "")
                current_session = str(input_state.get("source_session_id", "") or "")
                if (not current_track or current_track == str(expected.get("track_id", "") or "")) and (
                    not current_session or current_session == str(expected.get("source_session_id", "") or "")
                ):
                    selected = str(input_state.get("mode", "") or "")
        return selected if selected in {"auto", "manual"} else ""

    def interval_for(buffer_name: str) -> float:
        if buffer_name in {"albedo", "normal", "depth", "metallic", "roughness", "gbuffer_atlas"}:
            return 1.0 / max(1.0, gbuffer_fps)
        if buffer_name in {"model", "rendered"}:
            return model_frame_poll_interval()
        return primary_stream_interval(max_fps)

    def read_latest_state() -> dict[str, object]:
        state = json.loads((capture_dir / "latest.json").read_text())
        if not isinstance(state, dict):
            raise RuntimeError("latest frame state is invalid")
        return state

    def read_state() -> tuple[int, int, int]:
        state = read_latest_state()
        return int(state["seq"]), int(state["width"]), int(state["height"])

    def control_state_payload() -> dict[str, object]:
        state = read_latest_state()
        control = state.get("control")
        if not isinstance(control, dict):
            control = {}
        current_input = input_state_snapshot()
        return {
            "seq": int(state.get("seq", 0) or 0),
            "control": {
                **control,
                "mode": current_input.get("mode", "auto"),
            },
        }

    def read_rendered_state() -> tuple[int, int, int, Path]:
        state = valid_rendered_state()
        if state and rendered_dir is not None:
            return int(state["seq"]), int(state["width"]), int(state["height"]), rendered_dir
        seq, width, height = read_state()
        return seq, width, height, capture_dir

    def read_model_state() -> tuple[int, int, int, Path]:
        state = valid_rendered_state()
        if not state or rendered_dir is None:
            raise RuntimeError("fenced model frame is not ready")
        return int(state["seq"]), int(state["width"]), int(state["height"]), rendered_dir

    def read_run_config() -> dict[str, str]:
        config_path = capture_dir / "run_config.env"
        config: dict[str, str] = {}
        try:
            for line in config_path.read_text().splitlines():
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()
        except Exception:
            pass
        return config

    def source_frame_metadata(expected_seq: int) -> dict[str, object]:
        state = read_latest_state()
        if control_int(state, "seq", -1) != expected_seq:
            raise RuntimeError("source advanced while the JPEG was encoded")
        config_path = capture_dir / "run_config.env"
        latest_path = capture_dir / "latest.json"
        config = read_run_config()
        try:
            if latest_path.stat().st_mtime_ns < config_path.stat().st_mtime_ns:
                raise RuntimeError("source frame predates the active session")
        except FileNotFoundError as error:
            raise RuntimeError("source identity is not ready") from error
        track_id = config.get("TRACK", "")
        source_session_id = config.get("SOURCE_SESSION_ID", "")
        if not track_id or not source_session_id:
            raise RuntimeError("source identity is incomplete")
        return {
            "seq": expected_seq,
            "identity": {
                "track_id": track_id,
                "source_session_id": source_session_id,
            },
        }

    def current_track_id() -> str:
        return read_run_config().get("TRACK", "black_forest") or "black_forest"

    def is_managed_stk_pid(pid: int) -> bool:
        if pid <= 1:
            return False
        try:
            cmdline = (Path("/proc") / str(pid) / "cmdline").read_bytes()
            environ = (Path("/proc") / str(pid) / "environ").read_bytes()
        except Exception:
            return False
        expected_capture = f"STK_GBUFFER_CAPTURE_DIR={capture_dir}".encode("utf-8")
        return b"bin/supertuxkart" in cmdline and expected_capture in environ.split(b"\0")

    def managed_stk_pids() -> list[int]:
        candidates: set[int] = set()
        try:
            candidates.add(int(track_pid_path.read_text().strip()))
        except Exception:
            pass
        try:
            for entry in Path("/proc").iterdir():
                if entry.name.isdigit():
                    candidates.add(int(entry.name))
        except Exception:
            pass
        return sorted(pid for pid in candidates if is_managed_stk_pid(pid))

    def managed_stk_process_groups(pids: list[int]) -> tuple[set[int], set[int]]:
        groups: set[int] = set()
        individual_pids: set[int] = set()
        for pid in pids:
            if not is_managed_stk_pid(pid):
                continue
            try:
                pgid = os.getpgid(pid)
                session_id = os.getsid(pid)
            except ProcessLookupError:
                continue
            if pgid == pid and session_id == pid:
                groups.add(pgid)
            else:
                individual_pids.add(pid)
        return groups, individual_pids

    def process_group_alive(pgid: int) -> bool:
        try:
            entries = Path("/proc").iterdir()
        except Exception:
            return False
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
                state = fields[0]
                process_group = int(fields[2])
            except Exception:
                continue
            if process_group == pgid and state != "Z":
                return True
        return False

    def wait_for_stk_exit(
        groups: set[int],
        individual_pids: set[int],
        timeout: float,
    ) -> tuple[set[int], set[int]]:
        deadline = time.monotonic() + timeout
        remaining_groups = set(groups)
        remaining_pids = set(individual_pids)
        while time.monotonic() < deadline:
            remaining_groups = {pgid for pgid in remaining_groups if process_group_alive(pgid)}
            remaining_pids = {pid for pid in remaining_pids if is_managed_stk_pid(pid)}
            if not remaining_groups and not remaining_pids:
                break
            time.sleep(0.025)
        return remaining_groups, remaining_pids

    def signal_stk_processes(
        groups: set[int],
        individual_pids: set[int],
        signum: int,
    ) -> None:
        for pgid in groups:
            try:
                os.killpg(pgid, signum)
            except ProcessLookupError:
                pass
        for pid in individual_pids:
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                pass

    def stop_managed_stk() -> None:
        pids = managed_stk_pids()
        groups, individual_pids = managed_stk_process_groups(pids)
        signal_stk_processes(groups, individual_pids, signal.SIGTERM)
        groups, individual_pids = wait_for_stk_exit(
            groups,
            individual_pids,
            track_stop_grace_s,
        )
        signal_stk_processes(groups, individual_pids, signal.SIGKILL)
        groups, individual_pids = wait_for_stk_exit(
            groups,
            individual_pids,
            track_kill_wait_s,
        )
        if groups or individual_pids:
            raise RuntimeError("Managed SuperTuxKart process group did not terminate")
        try:
            track_pid_path.unlink()
        except FileNotFoundError:
            pass

    def launch_track(track: str) -> tuple[subprocess.Popen[bytes], str]:
        cfg = read_run_config()
        (project_dir / ".runtime").mkdir(parents=True, exist_ok=True)
        track_pid_path.parent.mkdir(parents=True, exist_ok=True)
        track_log_path.parent.mkdir(parents=True, exist_ok=True)
        source_session_id = f"{track}-{time.time_ns()}"
        env = os.environ.copy()
        env.pop("PROJECT_DIR", None)
        env.update(
            {
                "TRACK": track,
                "PROFILE_TIME": cfg.get("PROFILE_TIME", "86400") or "86400",
                "NUMKARTS": cfg.get("NUMKARTS", "8") or "8",
                "SCREENSIZE": cfg.get("SCREENSIZE", "1280x720") or "1280x720",
                "GFX_PRESET": cfg.get("GFX_PRESET", "5") or "5",
                "CAPTURE_EVERY": "1",
                "CAPTURE_WARMUP": "0",
                "CAPTURE_LIMIT": "0",
                "CAPTURE_VIS": "0",
                "CAPTURE_LATEST_ONLY": "1",
                "OUT_DIR": str(capture_dir),
                "RUN_NAME": "stk_live_preview",
                "SOURCE_SESSION_ID": source_session_id,
            }
        )
        log_file = track_log_path.open("ab")
        try:
            proc = subprocess.Popen(
                ["bash", str(track_script_path)],
                cwd=str(project_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        finally:
            log_file.close()
        pid_tmp = track_pid_path.with_suffix(".pid.tmp")
        pid_tmp.write_text(f"{proc.pid}\n", encoding="ascii")
        pid_tmp.replace(track_pid_path)
        return proc, source_session_id

    def current_source_reset_epoch() -> int:
        manifest = read_json_object(bridge_manifest_path)
        worker = read_json_object(worker_status_path)
        return max(
            control_int(manifest, "source_reset_epoch", -1),
            control_int(worker, "source_reset_epoch", -1),
            control_int(input_state, "source_reset_epoch", -1),
        )

    def wait_for_fresh_track_frame(
        track: str,
        source_session_id: str,
        proc: subprocess.Popen[bytes] | None,
        baseline_mtime_ns: int,
        minimum_source_reset_epoch: int,
        timeout: float = 30.0,
    ) -> dict[str, object]:
        latest_path = capture_dir / "latest.json"
        config_path = capture_dir / "run_config.env"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc is not None:
                return_code = proc.poll()
                if return_code is not None:
                    raise RuntimeError(f"SuperTuxKart exited with code {return_code}")
            try:
                latest_mtime_ns = latest_path.stat().st_mtime_ns
            except FileNotFoundError:
                latest_mtime_ns = 0
            cfg = read_run_config()
            config_matches = bool(
                cfg.get("TRACK", "") == track and cfg.get("SOURCE_SESSION_ID", "") == source_session_id
            )
            bridge = read_json_object(bridge_manifest_path)
            bridge_source_reset_epoch = control_int(bridge, "source_reset_epoch", -1)
            bridge_matches = bool(
                str(bridge.get("track_id", "") or "") == track
                and str(bridge.get("source_session_id", "") or "") == source_session_id
                and bridge_source_reset_epoch > minimum_source_reset_epoch
            )
            try:
                config_mtime_ns = config_path.stat().st_mtime_ns
            except FileNotFoundError:
                config_mtime_ns = 0
            if (
                latest_mtime_ns > baseline_mtime_ns
                and latest_mtime_ns >= config_mtime_ns
                and config_matches
                and bridge_matches
            ):
                try:
                    state = read_latest_state()
                    seq = int(state["seq"])
                    source_frame_value = state.get("source_frame", seq)
                    source_frame = int(seq if source_frame_value is None else source_frame_value)
                except (KeyError, TypeError, ValueError):
                    time.sleep(0.05)
                    continue
                return {
                    "track_id": track,
                    "source_session_id": source_session_id,
                    "source_seq": seq,
                    "source_frame": source_frame,
                    "source_reset_epoch": bridge_source_reset_epoch,
                    "source_generation": control_int(bridge, "source_generation", -1),
                    "latest_mtime_ns": latest_mtime_ns,
                }
            time.sleep(0.25)
        raise RuntimeError(f"No fresh game frames arrived within {timeout:.0f} seconds")

    def track_switch_worker(track: str, previous_track: str) -> None:
        latest_path = capture_dir / "latest.json"
        renderer_reset_error = ""
        previous_cfg = read_run_config()
        previous_source_session_id = str(previous_cfg.get("SOURCE_SESSION_ID", "") or "")
        previous_source_reset_epoch = current_source_reset_epoch()
        try:
            test_delay = max(
                0.0,
                min(5.0, float(os.environ.get("STK_TRACK_SWITCH_TEST_DELAY_S", "0"))),
            )
        except (TypeError, ValueError):
            test_delay = 0.0
        if test_delay > 0:
            time.sleep(test_delay)
        source_fence: dict[str, object] = {}
        try:
            initial_baseline = latest_path.stat().st_mtime_ns
        except FileNotFoundError:
            initial_baseline = 0
        try:
            force_auto_input("track-switch-active", clear_boundary=True)
            stop_managed_stk()
            proc, source_session_id = launch_track(track)
            source_fence = wait_for_fresh_track_frame(
                track,
                source_session_id,
                proc,
                initial_baseline,
                previous_source_reset_epoch,
            )
        except Exception as error:
            primary_error = str(error)
            rollback_error = ""
            rollback_renderer_error = ""
            restored = False
            supervisor_recovered = False
            with track_switch_lock:
                track_switch_state["status"] = "restoring"
                track_switch_state["error"] = primary_error
            force_auto_input("track-switch-restoring", clear_boundary=True)
            try:
                stop_managed_stk()
                if previous_track in track_options:
                    try:
                        rollback_baseline = latest_path.stat().st_mtime_ns
                    except FileNotFoundError:
                        rollback_baseline = 0
                    rollback_source_reset_epoch = current_source_reset_epoch()
                    rollback_proc, rollback_session_id = launch_track(previous_track)
                    rollback_fence = wait_for_fresh_track_frame(
                        previous_track,
                        rollback_session_id,
                        rollback_proc,
                        rollback_baseline,
                        rollback_source_reset_epoch,
                    )
                    try:
                        bump_renderer_reset(
                            "track_restore",
                            previous_track,
                            source_fence=rollback_fence,
                        )
                    except Exception as notify_error:
                        rollback_renderer_error = str(notify_error)
                        rollback_error += f"; renderer reset failed: {notify_error}"
                    force_auto_input(
                        "track-switch-restored",
                        source_fence=rollback_fence,
                    )
                    restored = not rollback_renderer_error
            except Exception as restore_error:
                rollback_error = f"; restore failed: {restore_error}"

            if not restored and previous_source_session_id:
                try:
                    recovered_fence = wait_for_fresh_track_frame(
                        previous_track,
                        previous_source_session_id,
                        None,
                        initial_baseline,
                        previous_source_reset_epoch,
                        timeout=20.0,
                    )
                    force_auto_input(
                        "track-switch-supervisor-recovered",
                        source_fence=recovered_fence,
                    )
                    supervisor_recovered = True
                except Exception as recovery_error:
                    rollback_error += f"; supervisor recovery failed: {recovery_error}"

            if not restored and not supervisor_recovered:
                force_auto_input("track-switch-failed", clear_boundary=True)
            with track_switch_lock:
                track_switch_state.update(
                    {
                        "status": ("ready" if restored or supervisor_recovered else "error"),
                        "error": f"{primary_error}{rollback_error}",
                        "renderer_reset_error": rollback_renderer_error,
                        "finished_at": time.time(),
                    }
                )
            return
        try:
            bump_renderer_reset(
                "track_switch",
                track,
                source_fence=source_fence,
            )
        except Exception as notify_error:
            renderer_reset_error = str(notify_error)
        with input_lock:
            with track_switch_lock:
                track_switch_state.update(
                    {
                        "status": "ready",
                        "error": "",
                        "renderer_reset_error": renderer_reset_error,
                        "finished_at": time.time(),
                    }
                )
            force_auto_input_locked(
                "track-switch-ready",
                source_fence=source_fence,
            )

    def renderer_identity_matches_track(track: str) -> bool:
        cfg = read_run_config()
        source_session_id = str(cfg.get("SOURCE_SESSION_ID", "") or "")
        live_control = read_json_object(prompt_live_control_path)
        bridge = read_json_object(bridge_manifest_path)
        return bool(
            source_session_id
            and str(cfg.get("TRACK", "") or "") == track
            and str(live_control.get("track_id", "") or "") == track
            and str(live_control.get("source_session_id", "") or "") == source_session_id
            and str(bridge.get("track_id", "") or "") == track
            and str(bridge.get("source_session_id", "") or "") == source_session_id
        )

    def repair_track_identity_worker(track: str) -> None:
        error_message = ""
        try:
            cfg = read_run_config()
            bridge = read_json_object(bridge_manifest_path)
            capture = read_latest_state()
            source_session_id = str(cfg.get("SOURCE_SESSION_ID", "") or "")
            if not source_session_id or str(cfg.get("TRACK", "") or "") != track:
                raise RuntimeError("current game track/session is not ready")
            if (
                str(bridge.get("track_id", "") or "") != track
                or str(bridge.get("source_session_id", "") or "") != source_session_id
            ):
                raise RuntimeError("bridge has not observed the current game session")
            source_seq = int(capture.get("seq", -1))
            source_frame_value = capture.get("source_frame", source_seq)
            source_frame = int(source_seq if source_frame_value is None else source_frame_value)
            source_fence = {
                "track_id": track,
                "source_session_id": source_session_id,
                "source_seq": source_seq,
                "source_frame": source_frame,
                "source_reset_epoch": control_int(bridge, "source_reset_epoch", current_source_reset_epoch()),
                "source_generation": control_int(bridge, "source_generation", -1),
            }
            bump_renderer_reset(
                "track_identity_repair",
                track,
                source_fence=source_fence,
            )
            force_auto_input(
                "track-identity-repaired",
                source_fence=source_fence,
            )
        except Exception as error:
            error_message = str(error)
        with track_switch_lock:
            track_switch_state.update(
                {
                    "status": "error" if error_message else "ready",
                    "error": error_message,
                    "renderer_reset_error": error_message,
                    "finished_at": time.time(),
                }
            )

    def track_status_payload(auto_repair: bool = True) -> dict[str, object]:
        current_track = current_track_id()
        start_identity_repair = False
        with track_switch_lock:
            status = str(track_switch_state.get("status", ""))
            if (
                auto_repair
                and current_track in track_options
                and status == "ready"
                and not renderer_identity_matches_track(current_track)
            ):
                track_switch_state.update(
                    {
                        "status": "switching",
                        "requested_track": current_track,
                        "previous_track": current_track,
                        "error": "",
                        "renderer_reset_error": "",
                        "started_at": time.time(),
                        "finished_at": 0.0,
                    }
                )
                start_identity_repair = True
            state = dict(track_switch_state)
        if start_identity_repair:
            threading.Thread(
                target=repair_track_identity_worker,
                args=(current_track,),
                name=f"stk-track-identity-{current_track}",
                daemon=True,
            ).start()
        return {
            "ok": True,
            "current": current_track,
            "options": [{"id": track, "label": label} for track, label in track_options.items()],
            **state,
        }

    def start_track_switch(track_value: object) -> tuple[int, dict[str, object]]:
        if read_only_observer:
            return 403, {
                "ok": False,
                "error": "Track switching is disabled in read-only observer mode",
            }
        track = str(track_value or "").strip()
        if track not in track_options:
            return 400, {"ok": False, "error": "Unsupported track"}
        previous_track = current_track_id()
        identity_matches = renderer_identity_matches_track(track)
        now = time.time()
        clean_same_track = False
        with track_switch_lock:
            status = str(track_switch_state.get("status", ""))
            if status in {"switching", "restoring"}:
                return 409, {"ok": False, "error": "A track switch is already in progress"}
            clean_same_track = bool(
                track == previous_track
                and identity_matches
                and status == "ready"
                and not track_switch_state.get("error")
                and not track_switch_state.get("renderer_reset_error")
            )
            if not clean_same_track:
                finished_at = float(track_switch_state.get("finished_at", 0.0) or 0.0)
                retry_in = track_switch_cooldown_s - (now - finished_at)
                if retry_in > 0:
                    return 429, {
                        "ok": False,
                        "error": "Please wait before switching tracks again",
                        "retry_in": retry_in,
                    }
                track_switch_state.update(
                    {
                        "status": "switching",
                        "requested_track": track,
                        "previous_track": previous_track,
                        "error": "",
                        "renderer_reset_error": "",
                        "started_at": now,
                        "finished_at": 0.0,
                    }
                )
        if clean_same_track:
            return 200, track_status_payload()
        force_auto_input(
            "track-switch-start",
            reset_clients=True,
            clear_boundary=True,
        )
        threading.Thread(
            target=track_switch_worker,
            args=(track, previous_track),
            name=f"stk-track-{track}",
            daemon=True,
        ).start()
        return 202, track_status_payload()

    def file_info(filename: str, expected_size: int) -> dict[str, object]:
        path = capture_dir / filename
        try:
            stat = path.stat()
        except FileNotFoundError:
            return {
                "present": False,
                "size": 0,
                "expected_size": expected_size,
                "size_ok": False,
                "age_ms": None,
            }
        age_ms = max(0.0, (time.time() - stat.st_mtime) * 1000.0)
        return {
            "present": True,
            "size": stat.st_size,
            "expected_size": expected_size,
            "size_ok": stat.st_size == expected_size,
            "age_ms": age_ms,
        }

    def state_payload() -> dict[str, object]:
        latest_state = read_latest_state()
        current_input = input_state_snapshot()
        seq = int(latest_state["seq"])
        width = int(latest_state["width"])
        height = int(latest_state["height"])
        source_control = latest_state.get("control")
        if not isinstance(source_control, dict):
            source_control = {}
        rgba_size = width * height * 4
        latest_path = capture_dir / "latest.json"
        try:
            latest_age_ms = max(0.0, (time.time() - latest_path.stat().st_mtime) * 1000.0)
        except Exception:
            latest_age_ms = None

        files = {
            "rgb": file_info("rgb.rgba8", rgba_size),
            "albedo": file_info("sp_diff_color.rgba8", rgba_size),
            "normal_depth": file_info("normal_depth.rgba8", rgba_size),
            "depth": file_info("depth.f32", rgba_size),
        }
        rendered_state = valid_rendered_state()
        rendered_external = bool(rendered_state)
        handover = render_handover_payload(rendered_state, current_input)
        rendered_control = rendered_state.get("control") if rendered_external else None
        control = rendered_control if isinstance(rendered_control, dict) else source_control
        explicit_rendered_mode = str(control.get("mode", "") or "").lower()
        if explicit_rendered_mode not in {"auto", "manual"}:
            explicit_rendered_mode = rendered_control_mode(
                rendered_state.get("source_seq") if rendered_external else seq,
                {
                    "track_id": rendered_state.get("track_id", ""),
                    "source_session_id": rendered_state.get("source_session_id", ""),
                    "reset_epoch": rendered_state.get("reset_epoch", -1),
                    "source_reset_epoch": rendered_state.get("source_reset_epoch", -1),
                }
                if rendered_external
                else current_render_identity_payload(),
            )
        run_config = read_run_config()
        current_identity = current_render_identity_payload()
        raw_frame_bytes = sum(int(info["expected_size"]) for info in files.values() if bool(info.get("present")))
        return {
            "server_instance_id": server_instance_id,
            "read_only_observer": read_only_observer,
            "seq": seq,
            "width": width,
            "height": height,
            "frame_age_ms": latest_age_ms,
            "capture_dir": str(capture_dir),
            "run_config": run_config,
            "files": files,
            "raw_frame_bytes": raw_frame_bytes,
            "source": {
                "seq": seq,
                "source_frame": latest_state.get("source_frame", seq),
                "input_event_id": latest_state.get("input_event_id", ""),
                "input_event_mode": latest_state.get("input_event_mode", ""),
                "control": source_control,
            },
            "rendered": {
                "external": rendered_external,
                "dir": str(rendered_dir) if rendered_dir is not None else "",
                "width": rendered_state.get("width") if rendered_external else None,
                "height": rendered_state.get("height") if rendered_external else None,
                "seq": rendered_state.get("seq") if rendered_external else None,
                "source_seq": rendered_state.get("source_seq") if rendered_external else None,
                "reset_epoch": rendered_state.get("reset_epoch") if rendered_external else None,
                "source_reset_epoch": rendered_state.get("source_reset_epoch") if rendered_external else None,
                "track_id": rendered_state.get("track_id") if rendered_external else None,
                "source_session_id": rendered_state.get("source_session_id") if rendered_external else None,
                "input_event_id": rendered_state.get("input_event_id") if rendered_external else None,
            },
            "handover": handover,
            "video": video_state_payload(),
            "control": {
                **control,
                "mode": explicit_rendered_mode,
            },
            "input": {
                "mode": current_input.get("mode", "auto"),
                "keys": current_input.get("keys", []),
                "t": current_input.get("t", 0),
                "server_time": current_input.get("server_time", current_input.get("t", 0)),
                "server_instance_id": current_input.get("server_instance_id", server_instance_id),
                "source": current_input.get("source", "unknown"),
                "client_id": current_input.get("client_id", ""),
                "intent": current_input.get("intent", ""),
                "input_seq": current_input.get("input_seq", -1),
                "input_event_id": current_input.get("input_event_id", ""),
                "boundary_source_seq": current_input.get("boundary_source_seq", -1),
                "boundary_source_frame": current_input.get("boundary_source_frame", -1),
                "owner_client_id": current_input.get("owner_client_id", ""),
                "owner_lease_ms": current_input.get("owner_lease_ms", 0.0),
                "input_suspended": current_input.get("input_suspended", False),
                "input_suspended_reason": current_input.get("input_suspended_reason", ""),
                "track_switch_status": current_input.get("track_switch_status", "ready"),
                "track_id": current_input.get("track_id", current_identity["track_id"]),
                "source_session_id": current_input.get("source_session_id", current_identity["source_session_id"]),
                "reset_epoch": current_input.get("reset_epoch", current_identity["reset_epoch"]),
                "source_reset_epoch": current_input.get("source_reset_epoch", current_identity["source_reset_epoch"]),
                "age_ms": (
                    max(0.0, (time.time() - float(current_input.get("t", 0))) * 1000.0)
                    if current_input.get("t", 0)
                    else None
                ),
            },
        }

    def rgba_image(path: Path, width: int, height: int) -> Image.Image:
        raw = path.read_bytes()
        expected = width * height * 4
        if len(raw) != expected:
            raise RuntimeError("latest frame is still being written")
        return Image.frombytes("RGBA", (width, height), raw)

    def needs_framebuffer_flip(root: Path) -> bool:
        return rendered_dir is None or root.resolve() != rendered_dir.resolve()

    def depth_image(path: Path, width: int, height: int) -> Image.Image:
        raw = path.read_bytes()
        expected = width * height * 4
        if len(raw) != expected:
            raise RuntimeError("latest depth is still being written")
        values = array.array("f")
        values.frombytes(raw)
        vals = [v for v in values if v < 1.0]
        lo = min(vals) if vals else 0.0
        hi = max(vals) if vals else 1.0
        scale = 255.0 / (hi - lo) if hi > lo else 1.0
        out = bytearray(width * height)
        for i, v in enumerate(values):
            x = int((v - lo) * scale)
            out[i] = 0 if x < 0 else 255 if x > 255 else x
        return Image.frombytes("L", (width, height), bytes(out)).convert("RGB")

    def channel_image(
        path: Path, depth_path: Path, width: int, height: int, channel: int, enhanced: bool = False
    ) -> Image.Image:
        raw = path.read_bytes()
        expected = width * height * 4
        if len(raw) != expected:
            raise RuntimeError("latest channel buffer is still being written")

        depth_raw = depth_path.read_bytes()
        if len(depth_raw) != width * height * 4:
            raise RuntimeError("latest depth is still being written")
        depth_values = array.array("f")
        depth_values.frombytes(depth_raw)

        source = raw[channel::4]
        hist = [0] * 256
        valid_count = 0
        nonzero_count = 0
        for i, depth in enumerate(depth_values):
            if depth >= 1.0:
                continue
            value = source[i]
            valid_count += 1
            if value > 0:
                hist[value] += 1
                nonzero_count += 1
        if nonzero_count == 0:
            for i, depth in enumerate(depth_values):
                if depth < 1.0:
                    hist[source[i]] += 1

        def percentile(q: float, count: int) -> int:
            if count <= 0:
                return 0
            target = max(1, int(count * q))
            running = 0
            for value, bucket_count in enumerate(hist):
                running += bucket_count
                if running >= target:
                    return value
            return 255

        sample_count = nonzero_count or valid_count
        lo = percentile(0.02, sample_count)
        hi = percentile(0.98, sample_count)
        scale = 237.0 / (hi - lo) if hi > lo else 1.0

        out = bytearray(width * height)
        for i, value in enumerate(source):
            if depth_values[i] >= 1.0:
                out[i] = 0
                continue
            mapped = int((value - lo) * scale) + 18
            out[i] = 18 if mapped < 18 else 255 if mapped > 255 else mapped
        if not enhanced:
            return Image.frombytes("L", (width, height), bytes(out)).convert("RGB")

        signal = bytearray(width * height)
        for i, value in enumerate(source):
            if depth_values[i] < 1.0 and value > 0:
                signal[i] = out[i]
        signal_image = Image.frombytes("L", (width, height), bytes(signal))
        signal_image = signal_image.filter(ImageFilter.MaxFilter(9))
        signal_raw = signal_image.tobytes()
        rgb = bytearray(width * height * 3)
        for i, value in enumerate(signal_raw):
            offset = i * 3
            if depth_values[i] >= 1.0:
                rgb[offset + 0] = 0
                rgb[offset + 1] = 0
                rgb[offset + 2] = 0
            elif value == 0:
                rgb[offset + 0] = 9
                rgb[offset + 1] = 10
                rgb[offset + 2] = 10
            else:
                boosted = 80 + int(value * 0.68)
                level = 255 if boosted > 255 else boosted
                rgb[offset + 0] = level
                rgb[offset + 1] = level
                rgb[offset + 2] = level
        return Image.frombytes("RGB", (width, height), bytes(rgb))

    def normal_image(path: Path, width: int, height: int) -> Image.Image:
        raw = path.read_bytes()
        expected = width * height * 4
        if len(raw) != expected:
            raise RuntimeError("latest normal buffer is still being written")
        out = bytearray(width * height * 3)
        for i in range(width * height):
            ex = raw[i * 4 + 0] / 255.0 * 2.0 - 1.0
            ey = raw[i * 4 + 1] / 255.0 * 2.0 - 1.0
            nx = ex
            ny = ey
            nz = 1.0 - abs(nx) - abs(ny)
            if nz < 0.0:
                old_x = nx
                nx = (1.0 - abs(ny)) * (1.0 if old_x >= 0.0 else -1.0)
                ny = (1.0 - abs(old_x)) * (1.0 if ny >= 0.0 else -1.0)
            length = math.sqrt(nx * nx + ny * ny + nz * nz)
            if length > 0.0:
                nx /= length
                ny /= length
                nz /= length
            out[i * 3 + 0] = int((nx * 0.5 + 0.5) * 255.0)
            out[i * 3 + 1] = int((ny * 0.5 + 0.5) * 255.0)
            out[i * 3 + 2] = int((nz * 0.5 + 0.5) * 255.0)
        return Image.frombytes("RGB", (width, height), bytes(out))

    def gbuffer_image(buffer_name: str, root: Path, width: int, height: int) -> Image.Image:
        if buffer_name == "depth":
            image = depth_image(root / "depth.f32", width, height)
        elif buffer_name == "normal":
            image = normal_image(root / "normal_depth.rgba8", width, height)
        elif buffer_name == "roughness":
            image = channel_image(root / "normal_depth.rgba8", root / "depth.f32", width, height, 2)
        elif buffer_name == "metallic":
            image = channel_image(root / "normal_depth.rgba8", root / "depth.f32", width, height, 3, True)
        elif buffer_name == "albedo":
            image = rgba_image(root / "sp_diff_color.rgba8", width, height)
            image = image.convert("RGB")
        else:
            raise RuntimeError(f"unknown gbuffer {buffer_name}")

        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if gbuffer_width > 0 and width > gbuffer_width:
            target_height = max(1, round(height * gbuffer_width / width))
            image = image.resize((gbuffer_width, target_height), Image.Resampling.BILINEAR)
        return image

    def rgb_preview_image(root: Path, width: int, height: int, target_width: int) -> Image.Image:
        image = rgba_image(root / "rgb.rgba8", width, height)
        image = image.convert("RGB")
        if needs_framebuffer_flip(root):
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if target_width > 0 and width > target_width:
            target_height = max(1, round(height * target_width / width))
            image = image.resize((target_width, target_height), Image.Resampling.BILINEAR)
        return image

    def preview_size(width: int, height: int, target_width: int) -> tuple[int, int]:
        target_width = max(1, target_width)
        target_height = max(1, round(height * target_width / width))
        return target_width, target_height

    def source_index_for_display(x: int, y: int, width: int, height: int, out_width: int, out_height: int) -> int:
        src_x = min(width - 1, int((x + 0.5) * width / out_width))
        display_y = min(height - 1, int((y + 0.5) * height / out_height))
        src_y = height - 1 - display_y
        return src_y * width + src_x

    def albedo_preview_image(root: Path, width: int, height: int, target_width: int) -> Image.Image:
        image = rgba_image(root / "sp_diff_color.rgba8", width, height)
        image = image.convert("RGB")
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        out_width, out_height = preview_size(width, height, target_width)
        if image.size != (out_width, out_height):
            image = image.resize((out_width, out_height), Image.Resampling.BILINEAR)
        return image

    def depth_preview_image(root: Path, width: int, height: int, target_width: int) -> Image.Image:
        out_width, out_height = preview_size(width, height, target_width)
        raw = (root / "depth.f32").read_bytes()
        if len(raw) != width * height * 4:
            raise RuntimeError("latest depth is still being written")
        values = array.array("f")
        values.frombytes(raw)
        sampled = array.array("f")
        for y in range(out_height):
            for x in range(out_width):
                sampled.append(values[source_index_for_display(x, y, width, height, out_width, out_height)])
        valid = [v for v in sampled if v < 1.0]
        lo = min(valid) if valid else 0.0
        hi = max(valid) if valid else 1.0
        scale = 255.0 / (hi - lo) if hi > lo else 1.0
        out = bytearray(out_width * out_height)
        for i, value in enumerate(sampled):
            mapped = int((value - lo) * scale)
            out[i] = 0 if mapped < 0 else 255 if mapped > 255 else mapped
        return Image.frombytes("L", (out_width, out_height), bytes(out)).convert("RGB")

    def normal_preview_image(root: Path, width: int, height: int, target_width: int) -> Image.Image:
        out_width, out_height = preview_size(width, height, target_width)
        raw = (root / "normal_depth.rgba8").read_bytes()
        if len(raw) != width * height * 4:
            raise RuntimeError("latest normal buffer is still being written")
        out = bytearray(out_width * out_height * 3)
        for y in range(out_height):
            for x in range(out_width):
                out_i = y * out_width + x
                src_i = source_index_for_display(x, y, width, height, out_width, out_height)
                ex = raw[src_i * 4 + 0] / 255.0 * 2.0 - 1.0
                ey = raw[src_i * 4 + 1] / 255.0 * 2.0 - 1.0
                nx = ex
                ny = ey
                nz = 1.0 - abs(nx) - abs(ny)
                if nz < 0.0:
                    old_x = nx
                    nx = (1.0 - abs(ny)) * (1.0 if old_x >= 0.0 else -1.0)
                    ny = (1.0 - abs(old_x)) * (1.0 if ny >= 0.0 else -1.0)
                length = math.sqrt(nx * nx + ny * ny + nz * nz)
                if length > 0.0:
                    nx /= length
                    ny /= length
                    nz /= length
                out[out_i * 3 + 0] = int((nx * 0.5 + 0.5) * 255.0)
                out[out_i * 3 + 1] = int((ny * 0.5 + 0.5) * 255.0)
                out[out_i * 3 + 2] = int((nz * 0.5 + 0.5) * 255.0)
        return Image.frombytes("RGB", (out_width, out_height), bytes(out))

    def channel_preview_image(
        root: Path, width: int, height: int, target_width: int, channel: int, enhanced: bool = False
    ) -> Image.Image:
        out_width, out_height = preview_size(width, height, target_width)
        raw = (root / "normal_depth.rgba8").read_bytes()
        if len(raw) != width * height * 4:
            raise RuntimeError("latest channel buffer is still being written")
        depth_raw = (root / "depth.f32").read_bytes()
        if len(depth_raw) != width * height * 4:
            raise RuntimeError("latest depth is still being written")
        depth_values = array.array("f")
        depth_values.frombytes(depth_raw)

        indices = [
            source_index_for_display(x, y, width, height, out_width, out_height)
            for y in range(out_height)
            for x in range(out_width)
        ]
        hist = [0] * 256
        valid_count = 0
        nonzero_count = 0
        for src_i in indices:
            if depth_values[src_i] >= 1.0:
                continue
            value = raw[src_i * 4 + channel]
            valid_count += 1
            if value > 0:
                hist[value] += 1
                nonzero_count += 1
        if nonzero_count == 0:
            for src_i in indices:
                if depth_values[src_i] < 1.0:
                    hist[raw[src_i * 4 + channel]] += 1

        def percentile(q: float, count: int) -> int:
            if count <= 0:
                return 0
            target = max(1, int(count * q))
            running = 0
            for value, bucket_count in enumerate(hist):
                running += bucket_count
                if running >= target:
                    return value
            return 255

        sample_count = nonzero_count or valid_count
        lo = percentile(0.02, sample_count)
        hi = percentile(0.98, sample_count)
        scale = 237.0 / (hi - lo) if hi > lo else 1.0
        out = bytearray(out_width * out_height)
        for i, src_i in enumerate(indices):
            if depth_values[src_i] >= 1.0:
                out[i] = 0
                continue
            value = raw[src_i * 4 + channel]
            mapped = int((value - lo) * scale) + 18
            out[i] = 18 if mapped < 18 else 255 if mapped > 255 else mapped
        if not enhanced:
            return Image.frombytes("L", (out_width, out_height), bytes(out)).convert("RGB")

        signal = bytearray(out_width * out_height)
        for i, src_i in enumerate(indices):
            if depth_values[src_i] < 1.0 and raw[src_i * 4 + channel] > 0:
                signal[i] = out[i]
        signal_image = Image.frombytes("L", (out_width, out_height), bytes(signal))
        signal_image = signal_image.filter(ImageFilter.MaxFilter(5))
        signal_raw = signal_image.tobytes()
        rgb = bytearray(out_width * out_height * 3)
        for i, value in enumerate(signal_raw):
            offset = i * 3
            if value == 0:
                rgb[offset + 0] = 9
                rgb[offset + 1] = 10
                rgb[offset + 2] = 10
            else:
                boosted = 80 + int(value * 0.68)
                level = 255 if boosted > 255 else boosted
                rgb[offset + 0] = level
                rgb[offset + 1] = level
                rgb[offset + 2] = level
        return Image.frombytes("RGB", (out_width, out_height), bytes(rgb))

    def gbuffer_preview_image(buffer_name: str, root: Path, width: int, height: int, target_width: int) -> Image.Image:
        if buffer_name == "albedo":
            return albedo_preview_image(root, width, height, target_width)
        if buffer_name == "normal":
            return normal_preview_image(root, width, height, target_width)
        if buffer_name == "depth":
            return depth_preview_image(root, width, height, target_width)
        if buffer_name == "metallic":
            return channel_preview_image(root, width, height, target_width, 3, True)
        if buffer_name == "roughness":
            return channel_preview_image(root, width, height, target_width, 2)
        raise RuntimeError(f"unknown gbuffer {buffer_name}")

    def get_gbuffer_atlas_jpeg() -> tuple[int, bytes]:
        seq, width, height = read_state()
        key = f"{capture_dir}:gbuffer_atlas"
        with cache_lock:
            item = cache.setdefault(key, {"seq": -1, "body": b""})
            if item["seq"] != seq:
                names = ("albedo", "normal", "depth", "metallic", "roughness")
                images = [gbuffer_image(name, capture_dir, width, height) for name in names]
                tile_width = max(image.width for image in images)
                tile_height = max(image.height for image in images)
                atlas = Image.new("RGB", (tile_width * len(images), tile_height), (5, 6, 7))
                for i, image in enumerate(images):
                    if image.size != (tile_width, tile_height):
                        image = image.resize((tile_width, tile_height), Image.Resampling.BILINEAR)
                    atlas.paste(image, (i * tile_width, 0))
                out = io.BytesIO()
                atlas.save(out, format="JPEG", quality=jpeg_quality, optimize=False)
                item["seq"] = seq
                item["body"] = out.getvalue()
            return seq, item["body"]  # type: ignore[return-value]

    def get_preview_atlas_jpeg() -> tuple[int, bytes]:
        seq, width, height = read_state()
        key = f"{capture_dir}:preview_atlas"
        with cache_lock:
            item = cache.setdefault(key, {"seq": -1, "body": b""})
            if item["seq"] != seq:
                gbuffer_tile_width = max(1, gbuffer_width)
                main_tile_width = max(1, gbuffer_tile_width * 5 // 2)
                source = rgb_preview_image(capture_dir, width, height, main_tile_width)

                _, rendered_width, rendered_height, rendered_root = read_rendered_state()
                rendered = rgb_preview_image(rendered_root, rendered_width, rendered_height, main_tile_width)
                main_tile_height = max(source.height, rendered.height)
                if source.size != (main_tile_width, main_tile_height):
                    source = source.resize((main_tile_width, main_tile_height), Image.Resampling.BILINEAR)
                if rendered.size != (main_tile_width, main_tile_height):
                    rendered = rendered.resize((main_tile_width, main_tile_height), Image.Resampling.BILINEAR)

                names = ("albedo", "normal", "depth", "metallic", "roughness")
                gbuffer_images = [
                    gbuffer_preview_image(name, capture_dir, width, height, gbuffer_tile_width) for name in names
                ]
                gbuffer_tile_height = max(image.height for image in gbuffer_images)
                for i, image in enumerate(gbuffer_images):
                    if image.size != (gbuffer_tile_width, gbuffer_tile_height):
                        gbuffer_images[i] = image.resize(
                            (gbuffer_tile_width, gbuffer_tile_height), Image.Resampling.BILINEAR
                        )

                atlas_width = gbuffer_tile_width * 5
                atlas_height = main_tile_height + gbuffer_tile_height
                atlas = Image.new("RGB", (atlas_width, atlas_height), (5, 6, 7))
                atlas.paste(source, (0, 0))
                atlas.paste(rendered, (main_tile_width, 0))
                for i, image in enumerate(gbuffer_images):
                    atlas.paste(image, (i * gbuffer_tile_width, main_tile_height))

                out = io.BytesIO()
                atlas.save(out, format="JPEG", quality=jpeg_quality, optimize=False)
                item["seq"] = seq
                item["body"] = out.getvalue()
            return seq, item["body"]  # type: ignore[return-value]

    def bounded_gbuffer_width(requested_width: int | None, source_width: int) -> int:
        fallback = gbuffer_width if gbuffer_width > 0 else min(source_width, 360)
        value = fallback if requested_width is None else requested_width
        return max(160, min(640, source_width, int(value)))

    def get_jpeg(buffer_name: str, requested_width: int | None = None) -> tuple[int, bytes]:
        if buffer_name == "preview_atlas":
            return get_preview_atlas_jpeg()

        if buffer_name == "gbuffer_atlas":
            return get_gbuffer_atlas_jpeg()

        if buffer_name == "model":
            seq, width, height, root = read_model_state()
            path = root / "rgb.rgba8"
        elif buffer_name == "rendered":
            seq, width, height, root = read_rendered_state()
            path = root / "rgb.rgba8"
        else:
            seq, width, height = read_state()
            root = capture_dir
            if buffer_name == "depth":
                path = root / "depth.f32"
            elif buffer_name == "albedo":
                path = root / "sp_diff_color.rgba8"
            elif buffer_name in {"normal", "metallic", "roughness"}:
                path = root / "normal_depth.rgba8"
            else:
                path = root / f"{buffer_name}.rgba8"

        target_gbuffer_width: int | None = None
        if buffer_name in {"albedo", "normal", "depth", "metallic", "roughness"}:
            target_gbuffer_width = bounded_gbuffer_width(requested_width, width)

        key = f"{path}:{buffer_name}:{target_gbuffer_width or 0}"
        with cache_lock:
            item = cache.setdefault(key, {"seq": -1, "body": b""})
            if item["seq"] != seq:
                if buffer_name in {"albedo", "normal", "depth", "metallic", "roughness"}:
                    if target_gbuffer_width is not None and width > target_gbuffer_width:
                        image = gbuffer_preview_image(buffer_name, root, width, height, target_gbuffer_width)
                    else:
                        image = gbuffer_image(buffer_name, root, width, height)
                else:
                    image = rgba_image(path, width, height)
                    image = image.convert("RGB")
                    if needs_framebuffer_flip(root):
                        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                    if main_width > 0 and width > main_width:
                        target_height = max(1, round(height * main_width / width))
                        image = image.resize((main_width, target_height), Image.Resampling.BILINEAR)
                out = io.BytesIO()
                image.save(out, format="JPEG", quality=jpeg_quality, optimize=False)
                item["seq"] = seq
                item["body"] = out.getvalue()
            return seq, item["body"]  # type: ignore[return-value]

    def write_ws_frame(conn: socket.socket, opcode: int, payload: bytes) -> None:
        length = len(payload)
        header = bytearray([0x80 | opcode])
        if length < 126:
            header.append(length)
        elif length < 65536:
            header.append(126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(127)
            header.extend(struct.pack("!Q", length))
        conn.sendall(bytes(header) + payload)

    def is_ws_connection_error(exc: OSError) -> bool:
        if isinstance(exc, socket.timeout):
            return True
        return exc.errno in {
            errno.EBADF,
            errno.EPIPE,
            errno.ECONNABORTED,
            errno.ECONNRESET,
            errno.ENETRESET,
            errno.ENOTCONN,
            errno.ESHUTDOWN,
            errno.ETIMEDOUT,
        }

    def read_exact(conn: socket.socket, size: int) -> bytes | None:
        chunks = bytearray()
        deadline = time.monotonic() + 0.25
        while len(chunks) < size:
            try:
                chunk = conn.recv(size - len(chunks))
            except socket.timeout:
                if time.monotonic() >= deadline:
                    return None
                continue
            except (ConnectionResetError, OSError):
                return None
            if not chunk:
                return None
            chunks.extend(chunk)
        return bytes(chunks)

    class WebSocketTextReader:
        def __init__(self) -> None:
            self.fragmented_text: bytearray | None = None
            self.closed = False

        def fail(self, conn: socket.socket, code: int, reason: bytes) -> str:
            if not self.closed:
                try:
                    write_ws_frame(
                        conn,
                        8,
                        struct.pack("!H", code) + reason[:123],
                    )
                except OSError:
                    pass
            self.closed = True
            return ""

        def decode_text(self, conn: socket.socket, payload: bytes) -> str:
            try:
                return payload.decode("utf-8")
            except UnicodeDecodeError:
                return self.fail(conn, 1007, b"invalid utf-8")

        def poll(self, conn: socket.socket) -> str | None:
            if self.closed:
                return ""
            readable, _, _ = select.select([conn], [], [], 0)
            if not readable:
                return None
            header = read_exact(conn, 2)
            if header is None:
                self.closed = True
                return ""
            first, second = header
            fin = bool(first & 0x80)
            if first & 0x70:
                return self.fail(conn, 1002, b"reserved bits set")
            opcode = first & 0x0F
            masked = bool(second & 0x80)
            length_marker = second & 0x7F
            control_frame = opcode >= 0x8
            if opcode not in {0x0, 0x1, 0x2, 0x8, 0x9, 0xA}:
                return self.fail(conn, 1002, b"unknown opcode")
            if not masked:
                return self.fail(conn, 1002, b"client frame must be masked")
            if control_frame and (not fin or length_marker > 125):
                return self.fail(conn, 1002, b"invalid control frame")

            length = length_marker
            if length_marker == 126:
                extended = read_exact(conn, 2)
                if extended is None:
                    self.closed = True
                    return ""
                length = struct.unpack("!H", extended)[0]
                if length < 126:
                    return self.fail(conn, 1002, b"non-canonical length")
            elif length_marker == 127:
                extended = read_exact(conn, 8)
                if extended is None:
                    self.closed = True
                    return ""
                length = struct.unpack("!Q", extended)[0]
                if length < 65536 or length & (1 << 63):
                    return self.fail(conn, 1002, b"invalid 64-bit length")

            if opcode == 0x0:
                if self.fragmented_text is None:
                    return self.fail(conn, 1002, b"unexpected continuation")
                cumulative_length = len(self.fragmented_text) + length
            elif opcode == 0x1:
                if self.fragmented_text is not None:
                    return self.fail(conn, 1002, b"new data during fragmented message")
                cumulative_length = length
            elif opcode == 0x2:
                return self.fail(conn, 1003, b"binary client messages unsupported")
            else:
                cumulative_length = 0
            if cumulative_length > MAX_WS_TEXT_BYTES:
                return self.fail(conn, 1009, b"text message too large")

            mask = read_exact(conn, 4)
            if mask is None:
                self.closed = True
                return ""
            payload = read_exact(conn, length) if length else b""
            if payload is None:
                self.closed = True
                return ""
            payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))

            if opcode == 0x8:
                if len(payload) == 1:
                    return self.fail(conn, 1002, b"invalid close payload")
                try:
                    write_ws_frame(conn, 0x8, payload or struct.pack("!H", 1000))
                except OSError:
                    pass
                self.closed = True
                return ""
            if opcode == 0x9:
                try:
                    write_ws_frame(conn, 0xA, payload)
                except OSError:
                    self.closed = True
                    return ""
                return None
            if opcode == 0xA:
                return None
            if opcode == 0x1:
                if fin:
                    return self.decode_text(conn, payload)
                self.fragmented_text = bytearray(payload)
                return None

            assert self.fragmented_text is not None
            self.fragmented_text.extend(payload)
            if not fin:
                return None
            complete = bytes(self.fragmented_text)
            self.fragmented_text = None
            return self.decode_text(conn, complete)

    def video_encode_args(profile: dict[str, object], fps: int, bitrate_kbps: int) -> list[str]:
        if video_encoder == "h264_nvenc":
            return [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p1",
                "-tune",
                "ull",
                "-rc",
                "cbr",
                "-b:v",
                f"{bitrate_kbps}k",
                "-maxrate",
                f"{bitrate_kbps}k",
                "-bufsize",
                f"{max(500, bitrate_kbps // 2)}k",
                "-g",
                str(fps),
                "-bf",
                "0",
                "-zerolatency",
                "1",
            ]

        args = [
            "-c:v",
            "libx264",
            "-preset",
            str(profile["x264_preset"]),
            "-tune",
            "zerolatency",
            "-b:v",
            f"{bitrate_kbps}k",
            "-maxrate",
            f"{bitrate_kbps}k",
            "-bufsize",
            f"{max(500, bitrate_kbps // 2)}k",
            "-g",
            str(fps),
            "-keyint_min",
            str(fps),
            "-bf",
            "0",
        ]
        threads = int(profile.get("x264_threads", 0) or 0)
        if threads > 0:
            args.extend(["-threads", str(threads)])
        x264_params = str(profile.get("x264_params", "") or "")
        if x264_params:
            args.extend(["-x264-params", x264_params])
        return args

    def build_video_cmd(width: int, height: int, root: Path, profile: dict[str, object]) -> list[str]:
        assert ffmpeg_path is not None
        fps = int(profile["fps"])
        target_width = int(profile["width"])
        bitrate_kbps = int(profile["bitrate_kbps"])
        filters = []
        if needs_framebuffer_flip(root):
            filters.append("vflip")
        filters.extend((f"scale={target_width}:-2", "format=yuv420p"))
        vf = ",".join(filters)
        return [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s:v",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-vf",
            vf,
            *video_encode_args(profile, fps, bitrate_kbps),
            "-movflags",
            "+empty_moov+default_base_moof+frag_every_frame",
            "-frag_duration",
            "33333",
            "-muxdelay",
            "0",
            "-muxpreload",
            "0",
            "-flush_packets",
            "1",
            "-f",
            "mp4",
            "pipe:1",
        ]

    class SharedVideoStream:
        def __init__(self, key: str, width: int, height: int, root: Path, profile: dict[str, object]):
            self.key = key
            self.width = width
            self.height = height
            self.root = root
            self.profile = dict(profile)
            self.expected = width * height * 4
            self.fps = int(profile["fps"])
            self.cmd = build_video_cmd(width, height, root, profile)
            self.condition = threading.Condition()
            self.proc: subprocess.Popen[bytes] | None = None
            self.stop_event = threading.Event()
            self.init_boxes: list[bytes] = []
            self.init_data = b""
            self.init_ready = False
            self.seen_media = False
            self.media_boxes: list[tuple[int, bytes, bytes]] = []
            self.next_seq = 0
            self.latest_sync_seq = 0
            self.clients = 0
            self.last_client_at = time.monotonic()
            self.started_at = time.time()
            self.last_error = ""

        def start(self) -> None:
            self.proc = subprocess.Popen(
                self.cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            threading.Thread(target=self._feed_frames, daemon=True).start()
            threading.Thread(target=self._read_boxes, daemon=True).start()
            threading.Thread(target=self._drain_stderr, daemon=True).start()

        def is_alive(self) -> bool:
            return not self.stop_event.is_set() and self.proc is not None and self.proc.poll() is None

        def add_client(self) -> None:
            with self.condition:
                self.clients += 1
                self.last_client_at = time.monotonic()
                self.condition.notify_all()

        def remove_client(self) -> None:
            with self.condition:
                self.clients = max(0, self.clients - 1)
                self.last_client_at = time.monotonic()
                self.condition.notify_all()

        def wait_init(self, timeout: float) -> bytes | None:
            deadline = time.monotonic() + timeout
            with self.condition:
                while not self.init_ready and self.is_alive():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self.condition.wait(min(0.2, remaining))
                return self.init_data if self.init_ready else None

        def start_seq(self) -> int:
            with self.condition:
                return self.latest_sync_seq

        def end_seq(self) -> int:
            with self.condition:
                return self.next_seq

        def wait_sync(self, seq: int, timeout: float) -> int | None:
            deadline = time.monotonic() + timeout
            with self.condition:
                while True:
                    for item_seq, box_type, _body in self.media_boxes:
                        if item_seq >= seq and box_type == b"moof":
                            return item_seq
                    if not self.is_alive():
                        return None
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self.condition.wait(min(0.2, remaining))

        def wait_media(self, seq: int, timeout: float) -> tuple[int, bytes] | None:
            deadline = time.monotonic() + timeout
            with self.condition:
                while True:
                    if self.media_boxes:
                        first_seq = self.media_boxes[0][0]
                        if seq < first_seq:
                            seq = self.latest_sync_seq
                        for item_seq, _box_type, body in self.media_boxes:
                            if item_seq >= seq:
                                return item_seq, body
                    if not self.is_alive():
                        return None
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self.condition.wait(min(0.2, remaining))

        def _should_stop_idle(self) -> bool:
            with self.condition:
                return self.clients == 0 and time.monotonic() - self.last_client_at > 6.0

        def _stop_proc(self) -> None:
            self.stop_event.set()
            proc = self.proc
            if proc is None:
                return
            for pipe in (proc.stdin, proc.stdout):
                try:
                    if pipe is not None:
                        pipe.close()
                except Exception:
                    pass
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass
            with self.condition:
                self.condition.notify_all()

        def _feed_frames(self) -> None:
            assert self.proc is not None
            assert self.proc.stdin is not None
            frame_interval = 1.0 / max(1, self.fps)
            next_at = time.monotonic()
            while not self.stop_event.is_set():
                if self._should_stop_idle():
                    break
                try:
                    frame = (self.root / "rgb.rgba8").read_bytes()
                    if len(frame) == self.expected:
                        self.proc.stdin.write(frame)
                        self.proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    break
                except Exception:
                    time.sleep(0.01)
                next_at += frame_interval
                delay = next_at - time.monotonic()
                if delay > 0:
                    time.sleep(min(delay, frame_interval))
                else:
                    next_at = time.monotonic()
            self._stop_proc()

        def _read_exact_pipe(self, size: int) -> bytes | None:
            assert self.proc is not None
            assert self.proc.stdout is not None
            chunks = bytearray()
            while len(chunks) < size and not self.stop_event.is_set():
                try:
                    chunk = self.proc.stdout.read(size - len(chunks))
                except Exception:
                    return None
                if not chunk:
                    return None
                chunks.extend(chunk)
            return bytes(chunks)

        def _read_boxes(self) -> None:
            try:
                while not self.stop_event.is_set():
                    header = self._read_exact_pipe(8)
                    if header is None:
                        break
                    size = struct.unpack(">I", header[:4])[0]
                    box_type = header[4:8]
                    if size == 0:
                        break
                    if size == 1:
                        extended = self._read_exact_pipe(8)
                        if extended is None:
                            break
                        full_size = struct.unpack(">Q", extended)[0]
                        body_size = full_size - 16
                        body = self._read_exact_pipe(body_size)
                        if body is None:
                            break
                        box = header + extended + body
                    else:
                        body_size = size - 8
                        body = self._read_exact_pipe(body_size)
                        if body is None:
                            break
                        box = header + body
                    self._store_box(box_type, box)
            finally:
                self._stop_proc()

        def _store_box(self, box_type: bytes, box: bytes) -> None:
            with self.condition:
                media_start = box_type in {b"styp", b"moof", b"mdat"}
                if not self.seen_media and not media_start:
                    self.init_boxes.append(box)
                    self.condition.notify_all()
                    return

                if media_start:
                    self.seen_media = True
                    if not self.init_ready:
                        self.init_data = b"".join(self.init_boxes)
                        self.init_ready = bool(self.init_data)
                    seq = self.next_seq
                    self.next_seq += 1
                    if box_type in {b"styp", b"moof"}:
                        self.latest_sync_seq = seq
                    self.media_boxes.append((seq, box_type, box))
                    if len(self.media_boxes) > 240:
                        self.media_boxes = self.media_boxes[-240:]
                    self.condition.notify_all()

        def _drain_stderr(self) -> None:
            assert self.proc is not None
            if self.proc.stderr is None:
                return
            try:
                for raw in iter(self.proc.stderr.readline, b""):
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="replace").strip()
                    if line:
                        self.last_error = line
                    if self.stop_event.is_set():
                        break
            except Exception:
                pass

    def shared_video_key(width: int, height: int, root: Path, profile: dict[str, object]) -> str:
        return ":".join(
            [
                str(profile.get("name", "profile")),
                str(profile.get("width", "")),
                str(profile.get("fps", "")),
                str(profile.get("bitrate_kbps", "")),
                f"{width}x{height}",
                str(root),
            ]
        )

    def get_shared_video_stream(width: int, height: int, root: Path, profile: dict[str, object]) -> SharedVideoStream:
        key = shared_video_key(width, height, root, profile)
        with shared_video_lock:
            for stale_key, stream in list(shared_video_streams.items()):
                if isinstance(stream, SharedVideoStream) and not stream.is_alive():
                    shared_video_streams.pop(stale_key, None)
            existing = shared_video_streams.get(key)
            if isinstance(existing, SharedVideoStream) and existing.is_alive():
                return existing
            stream = SharedVideoStream(key, width, height, root, profile)
            shared_video_streams[key] = stream
            stream.start()
            return stream

    def stream_h264_mp4(handler: BaseHTTPRequestHandler) -> None:
        if not ffmpeg_path or not video_encoder:
            handler.send_error(503, f"ffmpeg H.264 encoder is not available: {video_encoder_reason}")
            return

        query = parse_qs(urlparse(handler.path).query)
        model_only = query.get("model", ["0"])[0] == "1"
        requested_stream_generation = str(query.get("stream_generation", [""])[0] or "")
        stream_generation = "".join(
            character
            for character in requested_stream_generation[:96]
            if character.isalnum() or character in {"-", "_", ":"}
        )
        try:
            if model_only:
                _, width, height, root = read_model_state()
                expected_epoch = int(query.get("source_reset_epoch", ["0"])[0] or 0)
                expected_source_seq = int(query.get("min_source_seq", ["-1"])[0] or -1)
                expected_input_event_id = str(query.get("input_event_id", [""])[0] or "")[:160]
                model_state = valid_rendered_state()
                if expected_epoch > 0 and control_int(model_state, "source_reset_epoch", -1) != expected_epoch:
                    raise RuntimeError("requested model epoch is not ready")
                if expected_source_seq >= 0 and control_int(model_state, "source_seq", -1) < expected_source_seq:
                    raise RuntimeError("requested model boundary is not ready")
                if (
                    expected_input_event_id
                    and str(model_state.get("input_event_id", "") or "") != expected_input_event_id
                ):
                    raise RuntimeError("requested model input event is not ready")
            else:
                _, width, height, root = read_rendered_state()
        except Exception:
            handler.send_error(404, "latest rendered frame is not ready")
            return

        profile = video_profile_from_request(handler, width)
        stream = get_shared_video_stream(width, height, root, profile)
        stream.add_client()
        init_data = stream.wait_init(5.0)
        if not init_data:
            stream.remove_client()
            handler.send_error(503, "shared H.264 stream did not produce an init segment")
            return

        next_seq = stream.start_seq()
        if model_only and stream_generation:
            fresh_seq: int | None = stream.end_seq()
            fresh_fragment_count = 2 if expected_input_event_id else 1
            for _ in range(fresh_fragment_count):
                fresh_seq = stream.wait_sync(int(fresh_seq), 2.0)
                if fresh_seq is None:
                    stream.remove_client()
                    handler.send_error(503, "shared H.264 stream did not produce a fresh fragment")
                    return
                next_seq = fresh_seq
                fresh_seq += 1

        previous_timeout = handler.connection.gettimeout()
        handler.connection.settimeout(1.0)
        handler.send_response(200)
        handler.send_header("Content-Type", "video/mp4")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Connection", "close")
        handler.end_headers()

        try:
            handler.wfile.write(init_data)
            handler.wfile.flush()
            while True:
                item = stream.wait_media(next_seq, 2.0)
                if item is None:
                    if not stream.is_alive():
                        break
                    continue
                seq, chunk = item
                handler.wfile.write(chunk)
                handler.wfile.flush()
                next_seq = seq + 1
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            stream.remove_client()
            try:
                handler.connection.settimeout(previous_timeout)
            except Exception:
                pass

    admin_token = admin_token_from_env()
    extra_origins = allowed_origins_from_env()
    post_body_slots = threading.BoundedSemaphore(MAX_CONCURRENT_POST_BODIES)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def origin_ok(self) -> bool:
            return origin_allowed(self.headers.get("Origin"), self.headers.get("Host"), extra_origins)

        def admin_authorized(self) -> bool:
            if not admin_token:
                return False
            supplied = self.headers.get(ADMIN_TOKEN_HEADER, "")
            return hmac.compare_digest(supplied.encode("utf-8"), admin_token.encode("utf-8"))

        def send_json(self, status_code: int, payload: dict[str, object], ensure_ascii: bool = True) -> None:
            body = json.dumps(payload, ensure_ascii=ensure_ascii).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # The client hung up (e.g. a rejected half-sent body); nothing to report.
                self.close_connection = True

        def read_json_body(self) -> dict[str, object]:
            """Read a bounded JSON-object body under a timeout and a global
            concurrency cap; raises RequestBodyError."""
            size = parse_content_length(self.headers.get("Content-Length"))
            if not size:
                return {}
            if not post_body_slots.acquire(blocking=False):
                raise RequestBodyError(503, "server_busy")
            previous_timeout = self.connection.gettimeout()
            try:
                self.connection.settimeout(POST_BODY_READ_TIMEOUT_S)
                raw = self.rfile.read(size)
            except (TimeoutError, socket.timeout):
                raise RequestBodyError(408, "body_read_timeout") from None
            finally:
                try:
                    self.connection.settimeout(previous_timeout)
                except OSError:
                    pass
                post_body_slots.release()
            if len(raw) != size:
                raise RequestBodyError(400, "incomplete_body")
            return parse_json_object(raw)

        def handle_ws_stream(self) -> None:
            if not self.origin_ok():
                self.send_error(403, "origin not allowed")
                return
            try:
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
            key = self.headers.get("Sec-WebSocket-Key")
            if not key:
                self.send_error(400, "missing websocket key")
                return
            query = parse_qs(urlparse(self.path).query)
            requested_client_id = query.get("client_id", [""])[0]
            allow_test_controls = query.get("candidate_test", [""])[0] == "1"
            ws_client_id = clean_input_client_id(requested_client_id or f"ws:{id(self):x}", "websocket")
            accept = base64.b64encode(
                hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
            ).decode("ascii")
            self.send_response_only(101, "Switching Protocols")
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept)
            self.end_headers()

            conn = self.connection
            conn.settimeout(0.05)
            text_reader = WebSocketTextReader()
            connection_generation = secrets.token_hex(16)
            model_frame_token = 0
            source_frame_token = 0
            write_ws_frame(
                conn,
                1,
                json.dumps(
                    {
                        "type": "stream_hello",
                        "connection_generation": connection_generation,
                        "server_instance_id": server_instance_id,
                    },
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
            selected_gbuffer = "albedo"
            requested_gbuffer_fps = 0.0
            requested_gbuffer_width = bounded_gbuffer_width(None, 10_000)
            visible = True
            send_rendered = False
            send_model = False
            send_source = False
            last_render_seq = -1
            last_model_seq = -1
            last_source_seq = -1
            last_gbuffer_seq = -1
            next_render_at = 0.0
            next_model_at = 0.0
            next_source_at = 0.0
            next_gbuffer_at = 0.0
            next_state_at = 0.0
            next_control_at = 0.0
            valid_gbuffers = {"albedo", "normal", "depth", "metallic", "roughness"}

            def ws_gbuffer_interval() -> float:
                if requested_gbuffer_fps <= 0.0:
                    return 3600.0
                return 1.0 / max(1.0, requested_gbuffer_fps)

            while True:
                message = text_reader.poll(conn)
                if message == "":
                    return
                if message:
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        data = {}
                    if allow_test_controls and data.get("type") == "test_close":
                        write_ws_frame(
                            conn,
                            8,
                            struct.pack("!H", 1012) + b"candidate restart test",
                        )
                        return
                    requested = data.get("gbuffer")
                    if requested in valid_gbuffers and requested != selected_gbuffer:
                        selected_gbuffer = requested
                        last_gbuffer_seq = -1
                        next_gbuffer_at = 0.0
                    if "visible" in data:
                        visible = bool(data.get("visible"))
                    if "rendered" in data:
                        send_rendered = bool(data.get("rendered"))
                    if "model" in data:
                        send_model = bool(data.get("model"))
                        next_model_at = 0.0
                    if "source" in data:
                        send_source = bool(data.get("source"))
                        next_source_at = 0.0
                    if "gbuffer_fps" in data:
                        try:
                            requested_gbuffer_fps = max(
                                0.0, min(float(gbuffer_fps), float(data.get("gbuffer_fps", 4.0)))
                            )
                        except Exception:
                            requested_gbuffer_fps = min(4.0, max(0.0, float(gbuffer_fps)))
                        next_gbuffer_at = 0.0
                    if "gbuffer_width" in data:
                        try:
                            requested_gbuffer_width = bounded_gbuffer_width(
                                int(float(data.get("gbuffer_width", gbuffer_width or 360))),
                                10_000,
                            )
                        except Exception:
                            requested_gbuffer_width = bounded_gbuffer_width(None, 10_000)
                        last_gbuffer_seq = -1
                        next_gbuffer_at = 0.0
                    if "mode" in data or "keys" in data:
                        ack = update_input_state(
                            data.get("mode", "auto"),
                            data.get("keys", []),
                            "websocket",
                            ws_client_id,
                            data.get("intent", "heartbeat"),
                            data.get("input_seq"),
                            data.get("server_instance_id", ""),
                        )
                        write_ws_frame(
                            conn,
                            1,
                            json.dumps(ack, separators=(",", ":")).encode("utf-8"),
                        )
                        next_state_at = 0.0

                if not visible:
                    time.sleep(0.05)
                    continue

                now = time.monotonic()
                try:
                    if send_rendered and now >= next_render_at:
                        seq, body = get_jpeg("rendered")
                        if seq != last_render_seq:
                            payload = bytes([1]) + int(seq & 0xFFFFFFFF).to_bytes(4, "big") + body
                            write_ws_frame(conn, 2, payload)
                            last_render_seq = seq
                        next_render_at = advance_stream_deadline(
                            next_render_at,
                            interval_for("rendered"),
                            time.monotonic(),
                        )

                    if send_model and now >= next_model_at:
                        try:
                            seq, body = get_jpeg("model")
                            if seq != last_model_seq:
                                model_state = valid_rendered_state()
                                if control_int(model_state, "seq", -1) == seq:
                                    model_frame_token = (model_frame_token + 1) & 0xFFFFFFFF
                                    if model_frame_token == 0:
                                        model_frame_token = 1
                                    meta = json.dumps(
                                        {
                                            "type": "model_meta",
                                            "seq": seq,
                                            "source_seq": model_state.get("source_seq"),
                                            "input_event_id": model_state.get("input_event_id", ""),
                                            "frame_token": model_frame_token,
                                            "connection_generation": connection_generation,
                                            "identity": {
                                                "track_id": model_state.get("track_id", ""),
                                                "source_session_id": model_state.get("source_session_id", ""),
                                                "reset_epoch": model_state.get("reset_epoch"),
                                                "source_reset_epoch": model_state.get("source_reset_epoch"),
                                            },
                                        },
                                        separators=(",", ":"),
                                    ).encode("utf-8")
                                    write_ws_frame(conn, 1, meta)
                                    payload = (
                                        bytes([5])
                                        + int(seq & 0xFFFFFFFF).to_bytes(4, "big")
                                        + int(model_frame_token).to_bytes(4, "big")
                                        + body
                                    )
                                    write_ws_frame(conn, 2, payload)
                                    last_model_seq = seq
                        except Exception:
                            pass
                        next_model_at = advance_stream_deadline(
                            next_model_at,
                            interval_for("rendered"),
                            time.monotonic(),
                        )

                    if send_source and now >= next_source_at:
                        try:
                            seq, body = get_jpeg("rgb")
                            if seq != last_source_seq:
                                source_meta = source_frame_metadata(seq)
                                source_frame_token = (source_frame_token + 1) & 0xFFFFFFFF
                                if source_frame_token == 0:
                                    source_frame_token = 1
                                meta = json.dumps(
                                    {
                                        "type": "source_meta",
                                        "seq": seq,
                                        "frame_token": source_frame_token,
                                        "connection_generation": connection_generation,
                                        "identity": source_meta["identity"],
                                    },
                                    separators=(",", ":"),
                                ).encode("utf-8")
                                write_ws_frame(conn, 1, meta)
                                payload = (
                                    bytes([6])
                                    + int(seq & 0xFFFFFFFF).to_bytes(4, "big")
                                    + int(source_frame_token).to_bytes(4, "big")
                                    + body
                                )
                                write_ws_frame(conn, 2, payload)
                                last_source_seq = seq
                        except Exception:
                            pass
                        next_source_at = advance_stream_deadline(
                            next_source_at,
                            interval_for("rgb"),
                            time.monotonic(),
                        )

                    if requested_gbuffer_fps > 0.0 and now >= next_gbuffer_at:
                        seq, body = get_jpeg(selected_gbuffer, requested_gbuffer_width)
                        if seq != last_gbuffer_seq:
                            payload = bytes([2]) + int(seq & 0xFFFFFFFF).to_bytes(4, "big") + body
                            write_ws_frame(conn, 2, payload)
                            last_gbuffer_seq = seq
                        next_gbuffer_at = now + ws_gbuffer_interval()
                    elif requested_gbuffer_fps <= 0.0:
                        next_gbuffer_at = now + 0.5

                    if now >= next_state_at:
                        current_state = state_payload()
                        payload = json.dumps(
                            {
                                "type": "state",
                                "state": current_state,
                            }
                        ).encode("utf-8")
                        write_ws_frame(conn, 1, payload)
                        handover = current_state.get("handover", {})
                        handover_waiting = isinstance(handover, dict) and handover.get("phase") in {
                            "waiting_for_epoch",
                            "waiting_for_model",
                        }
                        next_state_at = now + (0.05 if handover_waiting else 0.75)

                    if now >= next_control_at:
                        payload = json.dumps(
                            {
                                "type": "control",
                                **control_state_payload(),
                            }
                        ).encode("utf-8")
                        write_ws_frame(conn, 1, payload)
                        next_control_at = now + 0.08
                except OSError as exc:
                    if is_ws_connection_error(exc):
                        return
                    time.sleep(0.03)
                except Exception:
                    time.sleep(0.03)

                sleep_targets = [next_gbuffer_at, next_state_at, next_control_at]
                if send_rendered:
                    sleep_targets.append(next_render_at)
                if send_model:
                    sleep_targets.append(next_model_at)
                if send_source:
                    sleep_targets.append(next_source_at)
                sleep_until = min(sleep_targets)
                time.sleep(max(0.002, min(0.02, sleep_until - time.monotonic())))

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/ws/stream":
                if ws_startup_delay_s > 0 and time.monotonic() - server_started_monotonic < ws_startup_delay_s:
                    self.send_error(503, "websocket startup delay")
                    return
                self.handle_ws_stream()
                return

            if path == "/video-test":
                body = VIDEO_TEST_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/video/rendered.mp4":
                stream_h264_mp4(self)
                return

            if path == "/video/info":
                body = json.dumps(video_info_payload()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/assets/alaya-renderer-mark.png":
                try:
                    body = alaya_mark_path.read_bytes()
                except Exception:
                    self.send_error(404, "AlayaRenderer mark is not available")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "public, max-age=86400")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/assets/alaya-renderer-mark-white.png":
                white_mark_path = project_dir / "assets" / "alaya-renderer-mark-white.png"
                try:
                    body = white_mark_path.read_bytes()
                except Exception:
                    self.send_error(404, "White AlayaRenderer mark is not available")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "public, max-age=86400")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/":
                try:
                    body = (project_dir / "assets" / "landing.html").read_bytes()
                except Exception:
                    body = LANDING_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path in ("/play", "/game"):
                try:
                    body = Path(
                        os.environ.get("ALAYA_PREVIEW_GAME_HTML", str(project_dir / "assets" / "game.html"))
                    ).read_bytes()
                except Exception:
                    body = HTML.encode("utf-8")
                body = render_game_page(body, current_nominal_fps())
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path in (
                "/assets/landing-demo.mp4",
                "/assets/landing-demo-first-60s.mp4",
            ):
                video_filename = "landing-demo-first-60s.mp4" if path.endswith("first-60s.mp4") else "landing-demo.mp4"
                video_path = project_dir / "assets" / video_filename
                try:
                    file_size = video_path.stat().st_size
                except Exception:
                    self.send_error(404, "Landing demo video is not available")
                    return
                start = 0
                end = file_size - 1
                range_header = self.headers.get("Range", "").strip()
                partial = False
                if range_header.startswith("bytes="):
                    try:
                        range_value = range_header[6:].split(",", 1)[0]
                        start_text, end_text = range_value.split("-", 1)
                        if start_text:
                            start = int(start_text)
                            end = int(end_text) if end_text else end
                        elif end_text:
                            suffix_size = int(end_text)
                            start = max(0, file_size - suffix_size)
                        start = max(0, min(start, file_size - 1))
                        end = max(start, min(end, file_size - 1))
                        partial = True
                    except Exception:
                        self.send_error(416, "Invalid byte range")
                        return
                content_length = end - start + 1
                self.send_response(206 if partial else 200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", "public, max-age=86400")
                if partial:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(content_length))
                self.end_headers()
                try:
                    with video_path.open("rb") as handle:
                        handle.seek(start)
                        remaining = content_length
                        while remaining > 0:
                            chunk = handle.read(min(1024 * 1024, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return

            if path == "/assets/landing-demo-poster.png":
                poster_path = project_dir / "assets" / "landing-demo-poster.png"
                try:
                    body = poster_path.read_bytes()
                except Exception:
                    self.send_error(404, "Landing demo poster is not available")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "public, max-age=86400")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/api/track":
                body = json.dumps(track_status_payload()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/api/prompt":
                body = json.dumps(prompt_status_payload(), ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/state":
                if state_startup_delay_s > 0 and time.monotonic() - server_started_monotonic < state_startup_delay_s:
                    self.send_error(503, "state startup delay")
                    return
                try:
                    state = state_payload()
                except Exception:
                    self.send_error(404, "latest frame is not ready")
                    return
                body = json.dumps(state).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/record/status":
                body = json.dumps(recording_status_payload()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/input":
                self.send_error(405)
                return

            if path == "/frame":
                frame_path = capture_dir / "rgb.rgba8"
                try:
                    body = frame_path.read_bytes()
                except Exception:
                    self.send_error(404, "latest frame is not ready")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/jpeg":
                try:
                    seq, body = get_jpeg("rgb")
                    source_meta = source_frame_metadata(seq)
                    source_identity = source_meta["identity"]
                except Exception:
                    self.send_error(404, "latest frame is not ready")
                    return

                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-STK-Frame-Seq", str(seq))
                self.send_header("X-STK-Track-Id", str(source_identity.get("track_id", "")))
                self.send_header(
                    "X-STK-Source-Session-Id",
                    str(source_identity.get("source_session_id", "")),
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/rendered.jpg":
                try:
                    _, body = get_jpeg("rendered")
                except Exception:
                    self.send_error(404, "latest rendered frame is not ready")
                    return

                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/model.jpg":
                try:
                    seq, body = get_jpeg("model")
                    model_state = valid_rendered_state()
                    if control_int(model_state, "seq", -1) != seq:
                        raise RuntimeError("model metadata does not match JPEG")
                except Exception:
                    self.send_error(404, "fenced model frame is not ready")
                    return

                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-STK-Frame-Seq", str(seq))
                self.send_header(
                    "X-STK-Source-Seq",
                    str(control_int(model_state, "source_seq", -1)),
                )
                self.send_header(
                    "X-STK-Input-Event-Id",
                    str(model_state.get("input_event_id", "") or "").replace("\r", "").replace("\n", "")[:160],
                )
                self.send_header("X-STK-Track-Id", str(model_state.get("track_id", "") or ""))
                self.send_header(
                    "X-STK-Source-Session-Id",
                    str(model_state.get("source_session_id", "") or ""),
                )
                self.send_header(
                    "X-STK-Reset-Epoch",
                    str(control_int(model_state, "reset_epoch", -1)),
                )
                self.send_header(
                    "X-STK-Source-Reset-Epoch",
                    str(control_int(model_state, "source_reset_epoch", -1)),
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/gbuffer/atlas.jpg":
                try:
                    _, body = get_jpeg("gbuffer_atlas")
                except Exception:
                    self.send_error(404, "latest G-buffer atlas is not ready")
                    return

                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            gbuffer_jpeg_map = {
                "/gbuffer/albedo.jpg": "albedo",
                "/gbuffer/normal.jpg": "normal",
                "/gbuffer/depth.jpg": "depth",
                "/gbuffer/metallic.jpg": "metallic",
                "/gbuffer/roughness.jpg": "roughness",
            }
            if path in gbuffer_jpeg_map:
                try:
                    query = parse_qs(urlparse(self.path).query)
                    try:
                        requested_width = int(float(query.get("w", ["0"])[0]))
                    except Exception:
                        requested_width = 0
                    _, body = get_jpeg(
                        gbuffer_jpeg_map[path],
                        requested_width if requested_width > 0 else None,
                    )
                except Exception:
                    self.send_error(404, "latest G-buffer frame is not ready")
                    return

                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/preview/atlas.jpg":
                try:
                    _, body = get_jpeg("preview_atlas")
                except Exception:
                    self.send_error(404, "latest preview atlas is not ready")
                    return

                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            stream_map = {
                "/source.mjpeg": "rgb",
                "/rendered.mjpeg": "rendered",
                "/gbuffer/atlas.mjpeg": "gbuffer_atlas",
                "/gbuffer/albedo.mjpeg": "albedo",
                "/gbuffer/normal.mjpeg": "normal",
                "/gbuffer/depth.mjpeg": "depth",
                "/gbuffer/metallic.mjpeg": "metallic",
                "/gbuffer/roughness.mjpeg": "roughness",
                "/mjpeg": "rgb",
            }
            if path in stream_map:
                buffer_name = stream_map[path]
                boundary = "stkframe"
                self.send_response(200)
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                last_seq = -1
                while True:
                    started = time.monotonic()
                    try:
                        seq, body = get_jpeg(buffer_name)
                        if seq != last_seq:
                            self.wfile.write(
                                f"--{boundary}\r\n"
                                "Content-Type: image/jpeg\r\n"
                                f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
                            )
                            self.wfile.write(body)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                            last_seq = seq
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    except Exception:
                        time.sleep(0.05)
                    elapsed = time.monotonic() - started
                    time.sleep(max(0.0, interval_for(buffer_name) - elapsed))

            self.send_error(404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path not in {"/api/track", "/api/prompt", "/record/start", "/record/stop", "/input"}:
                self.send_error(404)
                return
            if not self.origin_ok():
                self.send_json(403, {"ok": False, "error": "origin_not_allowed"})
                return
            if path in {"/record/start", "/record/stop"} and not self.admin_authorized():
                # Recording is an operator function, never exposed to visitors.
                error = "admin_token_required" if admin_token else "admin_disabled"
                self.send_json(403, {"ok": False, "error": error})
                return

            if path == "/input":
                try:
                    data = self.read_json_body()
                    ack = update_input_state(
                        data.get("mode", "auto"),
                        data.get("keys", []),
                        "http",
                        data.get("client_id", ""),
                        data.get("intent", "heartbeat"),
                        data.get("input_seq"),
                        data.get("server_instance_id", ""),
                    )
                    status_code = 200
                except RequestBodyError as error:
                    status_code = error.status
                    ack = {"type": "input_ack", "accepted": False, "reason": error.reason}
                    if error.status == 413:
                        ack["max_bytes"] = MAX_POST_BODY_BYTES
                    self.close_connection = True
                except Exception as error:
                    status_code = 400
                    ack = {
                        "type": "input_ack",
                        "accepted": False,
                        "reason": "invalid_request",
                        "error": str(error)[:240],
                    }
                body = json.dumps(ack, separators=(",", ":")).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/record/stop":
                status_code, payload = stop_recording_job()
                self.send_json(status_code, payload)
                return

            try:
                data = self.read_json_body()
            except RequestBodyError as error:
                self.close_connection = True
                self.send_json(error.status, {"ok": False, "error": error.reason})
                return

            if path == "/api/track":
                status_code, payload = start_track_switch(data.get("track", ""))
                self.send_json(status_code, payload)
            elif path == "/api/prompt":
                client_ip = self.client_address[0] if self.client_address else "?"
                status_code, payload = submit_style_prompt(data.get("prompt", ""), client_ip)
                self.send_json(status_code, payload, ensure_ascii=False)
            else:
                status_code, payload = start_recording_job(data)
                self.send_json(status_code, payload)

    if read_only_observer:
        with input_lock:
            refresh_observer_input_locked()
    else:
        force_auto_input("startup-reset", reset_clients=True)
    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--rendered-dir", type=Path)
    parser.add_argument(
        "--input-state",
        type=Path,
        help="Input-state path; defaults to CAPTURE_DIR/input.json",
    )
    parser.add_argument(
        "--read-only-observer",
        action="store_true",
        help=(
            "Never write input state or manage SuperTuxKart/recording processes; "
            "intended for an isolated shadow preview"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8091, type=int)
    parser.add_argument("--main-width", default=720, type=int)
    parser.add_argument("--gbuffer-width", default=360, type=int)
    parser.add_argument("--jpeg-quality", default=72, type=int)
    parser.add_argument("--max-fps", default=24.0, type=float)
    parser.add_argument("--gbuffer-fps", default=6.0, type=float)
    args = parser.parse_args()

    if args.read_only_observer:
        if not args.capture_dir.is_dir():
            parser.error("--capture-dir must already exist in read-only observer mode")
        if args.input_state is None:
            parser.error("--read-only-observer requires --input-state")
    else:
        args.capture_dir.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(
            args.capture_dir,
            args.rendered_dir,
            args.main_width,
            args.gbuffer_width,
            args.jpeg_quality,
            args.max_fps,
            args.gbuffer_fps,
            args.input_state,
            args.read_only_observer,
        ),
    )
    print(f"serving STK live preview from {args.capture_dir} on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
