#!/usr/bin/env python3
"""Bridge latest-only SuperTuxKart G-buffers into immutable model windows.

The STK capture writer replaces each raw buffer and advances ``latest.json``
last. This bridge treats that JSON file as the commit marker, rejects buffers
newer than the marker, and confirms the marker and source identity after every
read. Published windows therefore contain frames from exactly one logical
source generation.

Default model cadence is a non-overlapping 13-frame initial chunk followed by
non-overlapping 16-frame chunks. Production can select a two-latent 5/8 cadence
for lower input latency. A fixed rolling-window mode remains available for
other consumers that require at least 25 frames per request.

The manifest's immutable metadata file and, for NPY transport, its named
immutable sidecar are the publication authority. ``window.npz`` and
``window.gbuffers.npy`` are retained only as compatibility aliases and are not
an atomic pair.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import logging
import math
import os
import queue
import re
import signal
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Deque, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from alayarenderer_flash.transport import (
    TRANSPORT_NAME as SHM_SLOT_TRANSPORT,
)
from alayarenderer_flash.transport import (
    DoubleSlotWriter,
    SlotBusy,
    SlotReference,
    SlotReservation,
)
from alayarenderer_flash.transport import (
    run_self_check as run_shm_slot_self_check,
)

try:
    import cv2  # type: ignore[import-not-found]
except ImportError:  # Pillow remains a portable fallback.
    cv2 = None


SCHEMA_VERSION = 2
MODALITIES = ("albedo", "depth", "metallic", "normal", "roughness")
CONTROL_COLUMNS = ("steer", "accel", "brake", "speed", "position")
DEFAULT_LATENT_WINDOW = 4
SUPPORTED_LATENT_WINDOWS = (2, 4)
WINDOW_TRANSPORTS = ("npz", "npy", SHM_SLOT_TRANSPORT)
DEFAULT_WINDOW_TRANSPORT = "npz"
DEFAULT_CAPTURE_DIR = Path("/dev/shm/stk_browser_preview_live")
RAW_FILES = {
    "albedo": "sp_diff_color.rgba8",
    "normal_material": "normal_depth.rgba8",
    "depth": "depth.f32",
}
WINDOW_RE = re.compile(r"^window_(\d+)\.npz$")
INPUT_STEER_THRESHOLD = 0.10
FRESH_SOURCE_STATUSES = frozenset({"fresh_epoch_requested", "waiting_for_fresh_epoch"})


def cadence_for_latent_window(latent_window: int) -> tuple[int, int]:
    if latent_window not in SUPPORTED_LATENT_WINDOWS:
        supported = ", ".join(str(value) for value in SUPPORTED_LATENT_WINDOWS)
        raise ValueError(f"latent window must be one of: {supported}")
    return 4 * latent_window - 3, 4 * latent_window


class SourceNotReady(RuntimeError):
    """The source is absent, changing, stale, or between committed frames."""


class InconsistentCapture(SourceNotReady):
    """The latest-only files did not describe one committed frame."""


@dataclasses.dataclass(frozen=True)
class Snapshot:
    data: bytes
    device: int
    inode: int
    size: int
    mtime_ns: int

    @property
    def token(self) -> tuple[int, int, int, int]:
        return self.device, self.inode, self.size, self.mtime_ns


@dataclasses.dataclass(frozen=True)
class SourceIdentity:
    track: str
    source_session_id: str
    session_id_derived: bool
    token: str
    ready_after_ns: int
    run_config_mtime_ns: int
    metadata_mtime_ns: int


@dataclasses.dataclass(frozen=True)
class RawCapture:
    state: Mapping[str, Any]
    marker: Snapshot
    raw: Mapping[str, bytes]


@dataclasses.dataclass(frozen=True)
class CurrentMarkerState:
    identity: SourceIdentity
    marker: Snapshot
    seq: int
    source_frame: int
    width: int
    height: int


@dataclasses.dataclass
class Frame:
    sample_index: int
    seq: int
    source_frame: int
    captured_at_ns: int
    gbuffers: np.ndarray
    control: Mapping[str, Any]
    input_event_id: str = ""


@dataclasses.dataclass(frozen=True)
class PromptState:
    schema_version: int
    reset_epoch: int
    reset_reason: str
    render_request_id: int
    request_reason: str
    prompt_revision: int
    text: str
    active: bool
    track_id: str
    source_session_id: str
    reset_source_seq: int | None
    reset_source_frame: int | None
    requested_at: float

    @property
    def key(self) -> tuple[Any, ...]:
        return (
            self.schema_version,
            self.reset_epoch,
            self.render_request_id,
            self.prompt_revision,
            self.text,
            self.track_id,
            self.source_session_id,
            self.reset_source_seq,
            self.reset_source_frame,
        )


@dataclasses.dataclass(frozen=True)
class DirectionalInputState:
    requested_mode: str
    effective_mode: str
    steer: int
    accel: bool
    brake: bool
    direction_keys: tuple[str, ...]
    server_time: float
    source: str
    file_mtime_ns: int
    age_seconds: float
    fresh: bool
    token: tuple[int, int, int, int]
    input_event_id: str
    boundary_source_seq: int
    boundary_source_frame: int
    track_id: str
    source_session_id: str
    reset_epoch: int
    source_reset_epoch: int

    @property
    def signature(self) -> tuple[str, int, bool, bool]:
        return self.effective_mode, self.steer, self.accel, self.brake


@dataclasses.dataclass(frozen=True)
class WindowPlan:
    frames: Sequence[Frame]
    is_initial: bool
    mode: str


@dataclasses.dataclass(frozen=True)
class WindowFence:
    writer_epoch: int
    source_generation: int
    source_session_id: str
    source_identity_token: str
    track_id: str
    prompt_key: tuple[Any, ...]


@dataclasses.dataclass(frozen=True)
class WindowJob:
    fence: WindowFence
    window_id: int
    frames: tuple[Frame, ...]
    is_initial: bool
    mode: str
    prompt: PromptState
    previous_window_id: int
    previous_last_seq: int
    window_transport: str
    slot_reference: SlotReference | None
    compress: bool
    out_dir: Path
    staging_dir: Path
    stage_token: str
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    latent_window: int
    initial_pixel_frames: int
    resumed_pixel_frames: int
    created_at_ns: int

    @property
    def immutable_path(self) -> Path:
        return self.out_dir / f"window_{self.window_id:012d}.npz"

    @property
    def sidecar_path(self) -> Path | None:
        if self.window_transport != "npy":
            return None
        return gbuffer_sidecar_path(self.immutable_path)

    @property
    def staged_metadata_path(self) -> Path:
        return self.staging_dir / (f"job_{self.window_id:012d}_{self.stage_token}.metadata.npz")

    @property
    def staged_sidecar_path(self) -> Path | None:
        if self.window_transport != "npy":
            return None
        return self.staging_dir / (f"job_{self.window_id:012d}_{self.stage_token}.gbuffers.npy")

    @property
    def payload_file_name(self) -> str:
        if self.sidecar_path is not None:
            return self.sidecar_path.name
        if self.slot_reference is not None:
            return self.slot_reference.file_name
        return ""


@dataclasses.dataclass(frozen=True)
class WindowWriteResult:
    job: WindowJob
    metadata_size_bytes: int = 0
    sidecar_size_bytes: int = 0
    payload_nbytes: int = 0
    shape: tuple[int, ...] = ()
    completed_at_ns: int = 0
    stale: bool = False
    error: BaseException | None = None
    error_context: str = ""


class WindowPipelineFailure(RuntimeError):
    """The asynchronous immutable-window pipeline can no longer make progress."""


class AuthorityManifestReplaceError(RuntimeError):
    """Manifest replacement failed before or after its authority commit point."""

    def __init__(self, phase: str, committed: bool, detail: str) -> None:
        super().__init__(f"{phase}: {detail}")
        self.phase = phase
        self.committed = committed


class PromotionDeferred(RuntimeError):
    """The current source marker is transient; retry the oldest completion."""


class PromotionReset(RuntimeError):
    def __init__(
        self,
        reason: str,
        identity: SourceIdentity | None,
        boundary_seq: int | None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.identity = identity
        self.boundary_seq = boundary_seq


def read_snapshot(path: Path, expected_size: int | None = None) -> Snapshot:
    try:
        with path.open("rb", buffering=0) as handle:
            before = os.fstat(handle.fileno())
            data = handle.read()
            after = os.fstat(handle.fileno())
    except FileNotFoundError as exc:
        raise SourceNotReady(f"missing {path.name}") from exc
    except OSError as exc:
        raise SourceNotReady(f"cannot read {path}: {exc}") from exc

    before_token = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_token = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_token != after_token or len(data) != after.st_size:
        raise InconsistentCapture(f"{path.name} changed while being read")
    if expected_size is not None and len(data) != expected_size:
        raise InconsistentCapture(f"{path.name} has {len(data)} bytes, expected {expected_size}")
    return Snapshot(
        data=data,
        device=after.st_dev,
        inode=after.st_ino,
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
    )


def parse_json_object(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise InconsistentCapture(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise InconsistentCapture(f"{label} must contain a JSON object")
    return value


def parse_env(data: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceNotReady(f"run_config.env is not UTF-8: {exc}") from exc
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def read_source_identity(
    capture_dir: Path,
    settle_seconds: float,
    require_metadata: bool,
) -> SourceIdentity:
    config = read_snapshot(capture_dir / "run_config.env")
    config_values = parse_env(config.data)
    if not config_values.get("TRACK") or not config_values.get("SCREENSIZE"):
        raise SourceNotReady("run_config.env is incomplete")

    metadata_path = capture_dir / "metadata.txt"
    try:
        metadata = read_snapshot(metadata_path)
    except SourceNotReady:
        if require_metadata:
            raise
        metadata = Snapshot(b"", 0, 0, 0, 0)

    newest_identity_ns = max(config.mtime_ns, metadata.mtime_ns)
    if newest_identity_ns:
        age_ns = time.time_ns() - newest_identity_ns
        if age_ns < int(max(0.0, settle_seconds) * 1_000_000_000):
            raise SourceNotReady("source identity is still settling")

    track = config_values["TRACK"]
    digest = hashlib.sha256()
    digest.update(str(capture_dir.resolve()).encode("utf-8"))
    digest.update(b"\0")
    digest.update(track.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(config.mtime_ns).encode("ascii"))
    digest.update(b"\0")
    digest.update(config.data)
    source_session_id = config_values.get("SOURCE_SESSION_ID", "").strip()
    session_id_derived = not bool(source_session_id)
    if not source_session_id:
        source_session_id = f"legacy-{digest.hexdigest()[:20]}"
    return SourceIdentity(
        track=track,
        source_session_id=source_session_id,
        session_id_derived=session_id_derived,
        token=digest.hexdigest()[:20],
        ready_after_ns=newest_identity_ns,
        run_config_mtime_ns=config.mtime_ns,
        metadata_mtime_ns=metadata.mtime_ns,
    )


def read_committed_capture(
    capture_dir: Path,
    identity: SourceIdentity,
    expected_width: int,
    expected_height: int,
) -> RawCapture:
    marker_path = capture_dir / "latest.json"
    marker_before = read_snapshot(marker_path)
    state = parse_json_object(marker_before.data, "latest.json")
    try:
        seq = int(state["seq"])
        width = int(state["width"])
        height = int(state["height"])
    except Exception as exc:
        raise InconsistentCapture("latest.json is missing integer seq/width/height") from exc
    if seq < 0 or width <= 0 or height <= 0:
        raise InconsistentCapture("latest.json contains invalid dimensions or sequence")
    if expected_width > 0 and width != expected_width:
        raise SourceNotReady(f"source width is {width}, expected {expected_width}")
    if expected_height > 0 and height != expected_height:
        raise SourceNotReady(f"source height is {height}, expected {expected_height}")
    if marker_before.mtime_ns < identity.ready_after_ns:
        raise SourceNotReady("latest.json predates the current source identity")

    slot = str(state.get("slot", "") or "")
    if slot and re.fullmatch(r"slot_[01]", slot) is None:
        raise InconsistentCapture(f"latest.json contains invalid slot {slot!r}")
    raw_root = capture_dir / slot if slot else capture_dir

    expected_bytes = width * height * 4
    snapshots = {key: read_snapshot(raw_root / filename, expected_bytes) for key, filename in RAW_FILES.items()}
    for key, snapshot in snapshots.items():
        if snapshot.mtime_ns > marker_before.mtime_ns:
            raise InconsistentCapture(f"{RAW_FILES[key]} is newer than the latest.json commit marker")

    marker_after = read_snapshot(marker_path)
    if marker_after.token != marker_before.token or marker_after.data != marker_before.data:
        raise InconsistentCapture("latest.json advanced while buffers were read")

    identity_after = read_source_identity(
        capture_dir,
        settle_seconds=0.0,
        require_metadata=identity.metadata_mtime_ns > 0,
    )
    if identity_after.token != identity.token or identity_after.track != identity.track:
        raise InconsistentCapture("source identity changed while buffers were read")

    return RawCapture(
        state=state,
        marker=marker_before,
        raw={key: snapshot.data for key, snapshot in snapshots.items()},
    )


def read_stable_current_marker(
    capture_dir: Path,
    require_metadata: bool,
) -> CurrentMarkerState:
    identity = read_source_identity(
        capture_dir,
        settle_seconds=0.0,
        require_metadata=require_metadata,
    )
    marker = read_snapshot(capture_dir / "latest.json")
    state = parse_json_object(marker.data, "latest.json")
    try:
        seq = int(state["seq"])
        source_frame = int(state.get("source_frame", seq))
        width = int(state["width"])
        height = int(state["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InconsistentCapture("latest.json is missing stable seq/source_frame/width/height") from exc
    if seq < 0 or source_frame < 0 or width <= 0 or height <= 0:
        raise InconsistentCapture("latest.json contains invalid marker state")

    identity_after = read_source_identity(
        capture_dir,
        settle_seconds=0.0,
        require_metadata=require_metadata,
    )
    if (
        identity_after.token != identity.token
        or identity_after.track != identity.track
        or identity_after.source_session_id != identity.source_session_id
    ):
        raise InconsistentCapture("source identity changed during promotion fence")
    if marker.mtime_ns < max(identity.ready_after_ns, identity_after.ready_after_ns):
        raise SourceNotReady("latest.json predates the current source identity")
    return CurrentMarkerState(
        identity=identity,
        marker=marker,
        seq=seq,
        source_frame=source_frame,
        width=width,
        height=height,
    )


def _decode_octahedral_normal_reference(encoded_rg: np.ndarray) -> np.ndarray:
    enc = encoded_rg.astype(np.float32)
    enc *= 2.0 / 255.0
    enc -= 1.0
    nx = enc[:, :, 0]
    ny = enc[:, :, 1]
    nz = 1.0 - np.abs(nx) - np.abs(ny)
    folded = nz < 0.0
    old_x = nx[folded].copy()
    old_y = ny[folded].copy()
    nx[folded] = (1.0 - np.abs(old_y)) * np.where(old_x >= 0.0, 1.0, -1.0)
    ny[folded] = (1.0 - np.abs(old_x)) * np.where(old_y >= 0.0, 1.0, -1.0)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    np.maximum(length, 1e-12, out=length)
    decoded = np.empty((*nx.shape, 3), dtype=np.uint8)
    for channel, value in enumerate((nx, ny, nz)):
        normalized = value / length
        normalized += 1.0
        normalized *= 127.5
        np.clip(normalized, 0.0, 255.0, out=normalized)
        decoded[:, :, channel] = normalized
    return decoded


def _build_octahedral_normal_lut() -> np.ndarray:
    encoded = np.empty((256, 256, 2), dtype=np.uint8)
    encoded[:, :, 0] = np.arange(256, dtype=np.uint8)[:, None]
    encoded[:, :, 1] = np.arange(256, dtype=np.uint8)[None, :]
    return _decode_octahedral_normal_reference(encoded).reshape(65536, 3)


OCTAHEDRAL_NORMAL_LUT = _build_octahedral_normal_lut()


def decode_octahedral_normal(
    encoded_rg: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    indices = encoded_rg[:, :, 0].astype(np.uint16)
    indices <<= 8
    indices |= encoded_rg[:, :, 1]
    if out is None:
        return np.take(OCTAHEDRAL_NORMAL_LUT, indices, axis=0)
    if out.shape != (*indices.shape, 3) or out.dtype != np.uint8:
        raise ValueError("normal output must be uint8 [H,W,3]")
    np.take(OCTAHEDRAL_NORMAL_LUT, indices, axis=0, out=out)
    return out


def visualize_depth(
    depth: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    valid = np.isfinite(depth) & (depth < 1.0)
    if np.any(valid):
        if cv2 is not None:
            lo, hi, _, _ = cv2.minMaxLoc(depth, mask=valid.view(np.uint8))
        else:
            lo = float(np.min(depth[valid]))
            hi = float(np.max(depth[valid]))
    else:
        lo, hi = 0.0, 1.0
    scale = 255.0 / (hi - lo) if hi > lo else 1.0
    finite = np.nan_to_num(depth, nan=hi, posinf=hi, neginf=lo)
    finite -= lo
    finite *= scale
    np.clip(finite, 0, 255, out=finite)
    if out is None:
        out = np.empty((*depth.shape, 3), dtype=np.uint8)
    elif out.shape != (*depth.shape, 3) or out.dtype != np.uint8:
        raise ValueError("depth output must be uint8 [H,W,3]")
    gray = finite.astype(np.uint8)
    if cv2 is not None:
        cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB, dst=out)
    else:
        out[:, :, 0] = gray
        out[:, :, 1] = gray
        out[:, :, 2] = gray
    return out


def scale_to_cover_center_crop(
    rgb: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("scale input must be uint8 RGB [H,W,3]")
    source_height, source_width = rgb.shape[:2]
    scale = max(target_width / source_width, target_height / source_height)
    resized_width = max(target_width, int(math.ceil(source_width * scale)))
    resized_height = max(target_height, int(math.ceil(source_height * scale)))
    if cv2 is not None:
        resized = cv2.resize(
            rgb,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        left = max(0, (resized_width - target_width) // 2)
        top = max(0, (resized_height - target_height) // 2)
        result = resized[top : top + target_height, left : left + target_width]
        if result.shape != (target_height, target_width, 3):
            raise RuntimeError(f"unexpected fitted shape {result.shape}")
        return np.ascontiguousarray(result)

    image = Image.fromarray(rgb, mode="RGB")
    if (resized_width, resized_height) != (source_width, source_height):
        image = image.resize(
            (resized_width, resized_height),
            resample=Image.Resampling.BILINEAR,
        )
    left = max(0, (resized_width - target_width) // 2)
    top = max(0, (resized_height - target_height) // 2)
    image = image.crop((left, top, left + target_width, top + target_height))
    result = np.asarray(image, dtype=np.uint8)
    if result.shape != (target_height, target_width, 3):
        raise RuntimeError(f"unexpected fitted shape {result.shape}")
    return np.ascontiguousarray(result)


def fast_scale_to_cover_center_crop(
    image: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV fast path is unavailable")
    source_height, source_width = image.shape[:2]
    scale = max(target_width / source_width, target_height / source_height)
    resized_width = max(target_width, int(math.ceil(source_width * scale)))
    resized_height = max(target_height, int(math.ceil(source_height * scale)))
    top_left = np.flipud(image)
    resized = cv2.resize(
        top_left,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    left = max(0, (resized_width - target_width) // 2)
    top = max(0, (resized_height - target_height) // 2)
    return np.ascontiguousarray(resized[top : top + target_height, left : left + target_width])


def decode_gbuffers(
    raw: Mapping[str, bytes],
    width: int,
    height: int,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    pixels = width * height
    albedo_rgba = np.frombuffer(raw["albedo"], dtype=np.uint8).reshape(height, width, 4)
    packed = np.frombuffer(raw["normal_material"], dtype=np.uint8).reshape(height, width, 4)
    depth = np.frombuffer(raw["depth"], dtype="<f4", count=pixels).reshape(height, width)

    output = np.empty((len(MODALITIES), target_height, target_width, 3), dtype=np.uint8)
    if cv2 is not None:
        albedo = fast_scale_to_cover_center_crop(albedo_rgba, target_width, target_height)
        material = fast_scale_to_cover_center_crop(packed, target_width, target_height)
        fitted_depth = fast_scale_to_cover_center_crop(depth, target_width, target_height)
        cv2.mixChannels([albedo], [output[0]], [0, 0, 1, 1, 2, 2])
        visualize_depth(fitted_depth, out=output[1])
        cv2.mixChannels(
            [material],
            [output[2], output[4]],
            [3, 0, 3, 1, 3, 2, 2, 3, 2, 4, 2, 5],
        )
        decode_octahedral_normal(material[:, :, 0:2], out=output[3])
        return output

    source_modalities: Iterable[np.ndarray] = (
        albedo_rgba[:, :, :3],
        visualize_depth(depth),
        np.repeat(packed[:, :, 3:4], 3, axis=2),
        decode_octahedral_normal(packed[:, :, 0:2]),
        np.repeat(packed[:, :, 2:3], 3, axis=2),
    )
    for index, image in enumerate(source_modalities):
        # STK raw buffers have a bottom-left origin; model inputs are top-left.
        top_left = np.flipud(image).copy()
        output[index] = scale_to_cover_center_crop(top_left, target_width, target_height)
    return output


def number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clean_input_event_id(value: Any) -> str:
    return "".join(char for char in str(value or "") if char.isprintable())[:160]


def clean_control(value: Any) -> dict[str, Any]:
    control = value if isinstance(value, Mapping) else {}
    mode = str(control.get("mode", "") or "").lower()
    return {
        "kart_id": integer(control.get("kart_id"), -1),
        "steer": number(control.get("steer")),
        "accel": number(control.get("accel")),
        "brake": bool(control.get("brake", False)),
        "speed": number(control.get("speed")),
        "position": integer(control.get("position"), -1),
        "mode": mode if mode in {"auto", "manual"} else "",
    }


def control_matches_input_signature(
    control: Mapping[str, Any],
    signature: tuple[str, int, bool, bool],
) -> bool:
    expected_mode, expected_steer, expected_accel, expected_brake = signature
    actual_mode = str(control.get("mode", "") or "").lower()
    if actual_mode in {"auto", "manual"} and actual_mode != expected_mode:
        return False
    if expected_mode != "manual":
        return actual_mode != "manual"

    actual_steer = number(control.get("steer"))
    if expected_steer < 0:
        steer_matches = actual_steer <= -INPUT_STEER_THRESHOLD
    elif expected_steer > 0:
        steer_matches = actual_steer >= INPUT_STEER_THRESHOLD
    else:
        steer_matches = abs(actual_steer) < INPUT_STEER_THRESHOLD
    actual_accel = number(control.get("accel")) >= 0.5
    actual_brake = bool(control.get("brake", False))
    return bool(steer_matches and actual_accel == expected_accel and actual_brake == expected_brake)


def directional_signature_document(
    signature: tuple[str, int, bool, bool] | None,
) -> dict[str, Any]:
    if signature is None:
        return {}
    mode, steer, accel, brake = signature
    return {
        "mode": mode,
        "steer": steer,
        "accel": accel,
        "brake": brake,
    }


def directional_input_document(state: DirectionalInputState | None) -> dict[str, Any]:
    if state is None:
        return {}
    return {
        "requested_mode": state.requested_mode,
        "effective_mode": state.effective_mode,
        "steer": state.steer,
        "accel": state.accel,
        "brake": state.brake,
        "direction_keys": list(state.direction_keys),
        "server_time": state.server_time,
        "source": state.source,
        "file_mtime_ns": state.file_mtime_ns,
        "age_seconds": round(state.age_seconds, 3),
        "fresh": state.fresh,
        "input_event_id": state.input_event_id,
        "boundary_source_seq": state.boundary_source_seq,
        "boundary_source_frame": state.boundary_source_frame,
        "track_id": state.track_id,
        "source_session_id": state.source_session_id,
        "reset_epoch": state.reset_epoch,
        "source_reset_epoch": state.source_reset_epoch,
    }


def read_directional_input_state(
    path: Path,
    max_age_seconds: float,
    now_ns: int | None = None,
) -> DirectionalInputState | None:
    try:
        snapshot = read_snapshot(path)
        value = parse_json_object(snapshot.data, path.name)
    except SourceNotReady:
        return None

    requested_mode = str(value.get("mode", "auto") or "auto")
    requested_mode = "manual" if requested_mode == "manual" else "auto"
    raw_keys = value.get("keys", [])
    keys = {str(item) for item in raw_keys} if isinstance(raw_keys, list) else set()
    left = bool(keys & {"ArrowLeft", "KeyA"})
    right = bool(keys & {"ArrowRight", "KeyD"})
    accel = bool(keys & {"ArrowUp", "KeyW"})
    brake = bool(keys & {"ArrowDown", "KeyS"})
    steer = -1 if left and not right else 1 if right and not left else 0
    direction_keys = tuple(
        name
        for name, active in (
            ("left", left),
            ("right", right),
            ("accel", accel),
            ("brake", brake),
        )
        if active
    )
    current_ns = time.time_ns() if now_ns is None else now_ns
    age_seconds = max(0.0, (current_ns - snapshot.mtime_ns) / 1_000_000_000)
    fresh = max_age_seconds <= 0.0 or age_seconds <= max_age_seconds
    effective_mode = "manual" if requested_mode == "manual" and fresh else "auto"
    if effective_mode != "manual":
        steer = 0
        accel = False
        brake = False
        direction_keys = ()
    return DirectionalInputState(
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        steer=steer,
        accel=accel,
        brake=brake,
        direction_keys=direction_keys,
        server_time=number(value.get("server_time"), number(value.get("t"), 0.0)),
        source=str(value.get("source", "") or ""),
        file_mtime_ns=snapshot.mtime_ns,
        age_seconds=age_seconds,
        fresh=fresh,
        token=snapshot.token,
        input_event_id=clean_input_event_id(value.get("input_event_id")),
        boundary_source_seq=integer(value.get("boundary_source_seq"), -1),
        boundary_source_frame=integer(value.get("boundary_source_frame"), -1),
        track_id=str(value.get("track_id", "") or ""),
        source_session_id=str(value.get("source_session_id", "") or ""),
        reset_epoch=integer(value.get("reset_epoch"), -1),
        source_reset_epoch=integer(value.get("source_reset_epoch"), -1),
    )


def read_json_file(path: Path) -> dict[str, Any]:
    try:
        snapshot = read_snapshot(path)
        value = json.loads(snapshot.data.decode("utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def read_prompt_state(path: Path) -> PromptState:
    value = read_json_file(path)
    text = str(value.get("prompt", "") or "").strip()
    schema_version = integer(value.get("schema_version"), 0)
    render_request_id = max(0, integer(value.get("render_request_id"), 0))
    prompt_revision = max(0, integer(value.get("prompt_revision"), render_request_id))

    def optional_integer(key: str) -> int | None:
        if key not in value or value.get(key) is None:
            return None
        try:
            return int(value[key])
        except (TypeError, ValueError):
            return None

    return PromptState(
        schema_version=schema_version,
        reset_epoch=max(0, integer(value.get("reset_epoch"), 0)),
        reset_reason=str(value.get("reset_reason", "") or ""),
        render_request_id=render_request_id,
        request_reason=str(value.get("request_reason", "") or ""),
        prompt_revision=prompt_revision,
        text=text,
        active=schema_version == SCHEMA_VERSION and bool(text),
        track_id=str(value.get("track_id", "") or ""),
        source_session_id=str(value.get("source_session_id", "") or ""),
        reset_source_seq=optional_integer("reset_source_seq"),
        reset_source_frame=optional_integer("reset_source_frame"),
        requested_at=number(value.get("requested_at"), 0.0),
    )


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode("utf-8")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def prepare_json_temp(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.authority.tmp")
    payload = (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def replace_prepared_json(temporary: Path, destination: Path) -> None:
    try:
        os.replace(temporary, destination)
    except Exception as exc:
        raise AuthorityManifestReplaceError("pre-replace", False, f"{type(exc).__name__}: {exc}") from exc
    try:
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception as exc:
        raise AuthorityManifestReplaceError("post-replace-fsync", True, f"{type(exc).__name__}: {exc}") from exc


def atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray], compress: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            if compress:
                np.savez_compressed(handle, **arrays)
            else:
                np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_alias(source: Path, alias: Path) -> None:
    temporary = alias.with_name(f".{alias.name}.{os.getpid()}.tmp")
    try:
        # Immutable files and compatibility aliases share one output directory.
        # Never fall back to copying an 85+ MiB payload on the sampling thread.
        os.link(source, temporary)
        os.replace(temporary, alias)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def parse_window_filename(path: Path) -> tuple[int, int]:
    match = WINDOW_RE.match(path.name)
    if not match:
        return -1, -1
    return 0, int(match.group(1))


def gbuffer_sidecar_path(metadata_path: Path) -> Path:
    return metadata_path.with_name(f"{metadata_path.stem}.gbuffers.npy")


def discard_staged_job(job: WindowJob) -> None:
    for path in (job.staged_metadata_path, job.staged_sidecar_path):
        if path is None:
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def build_window_arrays(job: WindowJob) -> dict[str, np.ndarray]:
    frames = job.frames
    gbuffers = None if job.slot_reference is not None else np.stack([frame.gbuffers for frame in frames], axis=0)
    gbuffer_shape = (
        job.slot_reference.shape if job.slot_reference is not None else tuple(int(value) for value in gbuffers.shape)
    )
    gbuffer_payload_nbytes = (
        job.slot_reference.payload_nbytes if job.slot_reference is not None else int(gbuffers.nbytes)
    )
    gbuffer_file_size_bytes = job.slot_reference.file_size_bytes if job.slot_reference is not None else 0
    controls = np.asarray(
        [
            [
                number(frame.control.get("steer")),
                number(frame.control.get("accel")),
                1.0 if bool(frame.control.get("brake")) else 0.0,
                number(frame.control.get("speed")),
                number(frame.control.get("position"), -1.0),
            ]
            for frame in frames
        ],
        dtype=np.float32,
    )
    prompt = job.prompt
    fence = job.fence
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
        "window_id": np.asarray(job.window_id, dtype=np.int64),
        "generation": np.asarray(fence.source_generation, dtype=np.int64),
        "source_generation": np.asarray(fence.source_generation, dtype=np.int64),
        "source_reset_epoch": np.asarray(fence.source_generation, dtype=np.int64),
        "reset_epoch": np.asarray(prompt.reset_epoch, dtype=np.int64),
        "render_request_id": np.asarray(prompt.render_request_id, dtype=np.int64),
        "reset_reason": np.asarray(prompt.reset_reason),
        "request_reason": np.asarray(prompt.request_reason),
        "reset_source_seq": np.asarray(
            prompt.reset_source_seq if prompt.reset_source_seq is not None else -1,
            dtype=np.int64,
        ),
        "reset_source_frame": np.asarray(
            prompt.reset_source_frame if prompt.reset_source_frame is not None else -1,
            dtype=np.int64,
        ),
        "session_id": np.asarray(fence.source_session_id),
        "source_session_id": np.asarray(fence.source_session_id),
        "track": np.asarray(fence.track_id),
        "track_id": np.asarray(fence.track_id),
        "is_initial": np.asarray(job.is_initial, dtype=np.bool_),
        "window_mode": np.asarray(job.mode),
        "latent_window": np.asarray(job.latent_window, dtype=np.int32),
        "initial_pixel_frames": np.asarray(job.initial_pixel_frames, dtype=np.int32),
        "resumed_pixel_frames": np.asarray(job.resumed_pixel_frames, dtype=np.int32),
        "previous_window_id": np.asarray(job.previous_window_id, dtype=np.int64),
        "previous_last_seq": np.asarray(job.previous_last_seq, dtype=np.int64),
        "created_at_ns": np.asarray(job.created_at_ns, dtype=np.int64),
        "prompt_revision": np.asarray(prompt.prompt_revision, dtype=np.int64),
        "prompt": np.asarray(prompt.text if prompt.active else ""),
        "prompt_active": np.asarray(prompt.active, dtype=np.bool_),
        "modalities": np.asarray(MODALITIES),
        "control_columns": np.asarray(CONTROL_COLUMNS),
        "window_transport": np.asarray(job.window_transport),
        "gbuffers_file": np.asarray(job.payload_file_name),
        "gbuffers_shape": np.asarray(gbuffer_shape, dtype=np.int64),
        "gbuffers_dtype": np.asarray("uint8"),
        "gbuffers_payload_nbytes": np.asarray(gbuffer_payload_nbytes, dtype=np.int64),
        "gbuffers_file_size_bytes": np.asarray(gbuffer_file_size_bytes, dtype=np.int64),
        "sample_indices": np.asarray([frame.sample_index for frame in frames], dtype=np.int64),
        "frame_seqs": np.asarray([frame.seq for frame in frames], dtype=np.int64),
        "frame_input_event_ids": np.asarray([frame.input_event_id for frame in frames], dtype=np.str_),
        "source_frames": np.asarray([frame.source_frame for frame in frames], dtype=np.int64),
        "captured_at_ns": np.asarray([frame.captured_at_ns for frame in frames], dtype=np.int64),
        "controls": controls,
        "kart_ids": np.asarray(
            [integer(frame.control.get("kart_id"), -1) for frame in frames],
            dtype=np.int32,
        ),
        "frame_count": np.asarray(len(frames), dtype=np.int32),
        "width": np.asarray(job.target_width, dtype=np.int32),
        "height": np.asarray(job.target_height, dtype=np.int32),
        "first_seq": np.asarray(frames[0].seq, dtype=np.int64),
        "last_seq": np.asarray(frames[-1].seq, dtype=np.int64),
    }
    if gbuffers is not None:
        arrays["gbuffers"] = gbuffers
    if job.slot_reference is not None:
        for key, value in job.slot_reference.metadata().items():
            arrays[key] = np.asarray(value)
    return arrays


class WindowWriter:
    """Single-owner writer for immutable window pairs.

    The sampling thread only submits immutable jobs. This thread owns stacking
    and NPY/NPZ persistence. Publication authority remains with the bridge
    thread, which applies the live prompt/source fence before aliases and the
    manifest are advanced.
    """

    def __init__(
        self,
        max_pending: int,
        staging_dir: Path,
        initial_epoch: int = 0,
        write_delay: float = 0.0,
        save_npy_fn: Callable[[Path, np.ndarray], None] = atomic_save_npy,
        save_npz_fn: Callable[[Path, Mapping[str, np.ndarray], bool], None] = atomic_save_npz,
    ) -> None:
        self._jobs: queue.Queue[WindowJob | None] = queue.Queue(maxsize=max_pending)
        self._results: queue.SimpleQueue[WindowWriteResult] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._epoch = initial_epoch
        self._stopping = False
        self._fatal_error = ""
        self._active_jobs = 0
        self._submitted = 0
        self._completed = 0
        self._stale = 0
        self._errors = 0
        self._discarded_queued = 0
        self._max_pending_observed = 0
        self._write_delay = max(0.0, write_delay)
        self._staging_dir = staging_dir
        self._staging_dir.mkdir(parents=True, exist_ok=True)
        self._save_npy_fn = save_npy_fn
        self._save_npz_fn = save_npz_fn
        self._thread = threading.Thread(
            target=self._run,
            name="stk-window-writer",
            daemon=True,
        )
        self._thread.start()

    def _job_is_current(self, job: WindowJob) -> bool:
        with self._lock:
            return not self._stopping and job.fence.writer_epoch == self._epoch

    def submit(self, job: WindowJob) -> None:
        with self._lock:
            if self._fatal_error:
                raise WindowPipelineFailure(self._fatal_error)
            if self._stopping:
                raise WindowPipelineFailure("window writer is stopping")
            if job.fence.writer_epoch != self._epoch:
                raise WindowPipelineFailure("refused stale window-writer job")
        try:
            self._jobs.put_nowait(job)
        except queue.Full:
            raise
        with self._lock:
            self._submitted += 1
            self._max_pending_observed = max(self._max_pending_observed, self._jobs.qsize())

    def invalidate(self, epoch: int) -> None:
        with self._lock:
            if self._fatal_error:
                raise WindowPipelineFailure(self._fatal_error)
            if self._stopping:
                raise WindowPipelineFailure("window writer is stopping")
            if epoch <= self._epoch:
                raise WindowPipelineFailure("window writer epoch did not advance during reset")
            self._epoch = epoch
        discarded = 0
        while True:
            try:
                pending = self._jobs.get_nowait()
            except queue.Empty:
                break
            if pending is None:
                self._jobs.task_done()
                continue
            discard_staged_job(pending)
            discarded += 1
            self._jobs.task_done()
        if discarded:
            with self._lock:
                self._discarded_queued += discarded

    def _drain_pending(self) -> int:
        discarded = 0
        while True:
            try:
                pending = self._jobs.get_nowait()
            except queue.Empty:
                break
            if pending is not None:
                discard_staged_job(pending)
                discarded += 1
            self._jobs.task_done()
        if discarded:
            with self._lock:
                self._discarded_queued += discarded
        return discarded

    def discard_results(self) -> int:
        discarded = 0
        while True:
            result = self.poll()
            if result is None:
                break
            discard_staged_job(result.job)
            discarded += 1
        return discarded

    def purge_staging(self) -> None:
        try:
            paths = list(self._staging_dir.iterdir())
        except FileNotFoundError:
            return
        for path in paths:
            if not path.is_file():
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def poll(self) -> WindowWriteResult | None:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "thread_alive": self._thread.is_alive(),
                "epoch": self._epoch,
                "pending_jobs": self._jobs.qsize(),
                "active_jobs": self._active_jobs,
                "submitted": self._submitted,
                "completed": self._completed,
                "stale": self._stale,
                "errors": self._errors,
                "discarded_queued": self._discarded_queued,
                "max_pending_observed": self._max_pending_observed,
                "fatal_error": self._fatal_error,
            }

    def wait_idle(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            snapshot = self.summary()
            if snapshot["pending_jobs"] == 0 and snapshot["active_jobs"] == 0:
                return True
            time.sleep(0.001)
        snapshot = self.summary()
        return snapshot["pending_jobs"] == 0 and snapshot["active_jobs"] == 0

    def shutdown(self, timeout: float) -> bool:
        with self._lock:
            self._stopping = True
            self._epoch += 1
        self._drain_pending()
        if self._thread.is_alive():
            try:
                self._jobs.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join(timeout=max(0.0, timeout))
        self._drain_pending()
        self.discard_results()
        self.purge_staging()
        return not self._thread.is_alive()

    def _stale_result(self, job: WindowJob) -> WindowWriteResult:
        discard_staged_job(job)
        with self._lock:
            self._stale += 1
        return WindowWriteResult(
            job=job,
            completed_at_ns=time.time_ns(),
            stale=True,
        )

    def _write(self, job: WindowJob) -> WindowWriteResult:
        if not self._job_is_current(job):
            return self._stale_result(job)
        arrays = build_window_arrays(job)
        if self._write_delay > 0.0:
            time.sleep(self._write_delay)
        if not self._job_is_current(job):
            return self._stale_result(job)

        sidecar_path = job.staged_sidecar_path
        try:
            if sidecar_path is not None:
                self._save_npy_fn(sidecar_path, arrays["gbuffers"])
                if not self._job_is_current(job):
                    return self._stale_result(job)
                sidecar_stat = sidecar_path.stat()
                metadata_arrays = dict(arrays)
                metadata_arrays.pop("gbuffers")
                metadata_arrays["gbuffers_file_size_bytes"] = np.asarray(sidecar_stat.st_size, dtype=np.int64)
                self._save_npz_fn(job.staged_metadata_path, metadata_arrays, job.compress)
            else:
                self._save_npz_fn(job.staged_metadata_path, arrays, job.compress)
            if not self._job_is_current(job):
                return self._stale_result(job)
            metadata_stat = job.staged_metadata_path.stat()
            sidecar_stat = sidecar_path.stat() if sidecar_path is not None else None
            with self._lock:
                self._completed += 1
            return WindowWriteResult(
                job=job,
                metadata_size_bytes=metadata_stat.st_size,
                sidecar_size_bytes=(
                    sidecar_stat.st_size
                    if sidecar_stat
                    else job.slot_reference.file_size_bytes
                    if job.slot_reference is not None
                    else 0
                ),
                payload_nbytes=int(arrays["gbuffers_payload_nbytes"].item()),
                shape=tuple(int(value) for value in arrays["gbuffers_shape"].tolist()),
                completed_at_ns=time.time_ns(),
            )
        except BaseException as exc:
            discard_staged_job(job)
            with self._lock:
                self._errors += 1
                self._fatal_error = f"{type(exc).__name__}: {exc}"
            return WindowWriteResult(
                job=job,
                completed_at_ns=time.time_ns(),
                error=exc,
                error_context=f"writing staged job {job.stage_token}",
            )

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                self._jobs.task_done()
                return
            with self._lock:
                self._active_jobs += 1
            try:
                try:
                    result = self._write(job)
                except BaseException as exc:
                    discard_staged_job(job)
                    with self._lock:
                        self._errors += 1
                        self._fatal_error = f"{type(exc).__name__}: {exc}"
                    result = WindowWriteResult(
                        job=job,
                        completed_at_ns=time.time_ns(),
                        error=exc,
                        error_context=f"building staged job {job.stage_token}",
                    )
                self._results.put(result)
            finally:
                with self._lock:
                    self._active_jobs -= 1
                self._jobs.task_done()
            if result.error is not None:
                with self._lock:
                    self._stopping = True
                    self._epoch += 1
                self._drain_pending()
                return


class WindowScheduler:
    def __init__(
        self,
        initial_frames: int,
        subsequent_frames: int,
        fixed_window_frames: int,
        window_stride: int,
        ring_capacity: int,
    ) -> None:
        self.initial_frames = initial_frames
        self.subsequent_frames = subsequent_frames
        self.fixed_window_frames = fixed_window_frames
        self.window_stride = window_stride or fixed_window_frames
        required = fixed_window_frames if fixed_window_frames > 0 else max(initial_frames, subsequent_frames)
        capacity = ring_capacity or required
        if capacity < required:
            raise ValueError(f"ring capacity {capacity} is smaller than required {required}")
        self.frames: Deque[Frame] = collections.deque(maxlen=capacity)
        self.published = 0
        self.frames_since_publish = 0

    @property
    def mode(self) -> str:
        return "rolling" if self.fixed_window_frames > 0 else "chunks"

    def reset(self) -> None:
        self.frames.clear()
        self.published = 0
        self.frames_since_publish = 0

    def append(self, frame: Frame) -> None:
        if len(self.frames) == self.frames.maxlen:
            if self.fixed_window_frames > 0:
                self.frames.popleft()
            else:
                raise RuntimeError("frame ring would overflow before publication")
        self.frames.append(frame)
        self.frames_since_publish += 1

    def ready(self) -> bool:
        if self.fixed_window_frames > 0:
            if len(self.frames) < self.fixed_window_frames:
                return False
            return self.published == 0 or self.frames_since_publish >= self.window_stride
        required = self.initial_frames if self.published == 0 else self.subsequent_frames
        return len(self.frames) >= required

    def take(self) -> WindowPlan:
        if not self.ready():
            raise RuntimeError("window scheduler is not ready")
        initial = self.published == 0
        if self.fixed_window_frames > 0:
            selected = list(self.frames)[-self.fixed_window_frames :]
            self.frames_since_publish = 0
        else:
            required = self.initial_frames if initial else self.subsequent_frames
            selected = [self.frames.popleft() for _ in range(required)]
            self.frames_since_publish = 0
        self.published += 1
        return WindowPlan(frames=selected, is_initial=initial, mode=self.mode)


def advance_sample_deadline(
    deadline: float,
    sample_started_at: float,
    sample_finished_at: float,
    interval: float,
) -> float:
    if interval <= 0.0:
        return 0.0
    next_deadline = sample_started_at + interval if deadline <= 0.0 else deadline + interval
    max_catchup_lag = max(0.25, interval * 4.0)
    if sample_finished_at - next_deadline > max_catchup_lag:
        return sample_finished_at + interval
    return next_deadline


def wait_for_sample_deadline(
    deadline: float,
    now_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> float:
    now = now_fn()
    if deadline > now:
        sleep_fn(deadline - now)
        now = now_fn()
    return now


# A prompt boundary may run slightly ahead of the frames this bridge has read
# (the preview stamps it from the newest capture); anything further ahead was
# taken from an earlier capture session whose sequence numbers do not apply.
STALE_BOUNDARY_MARGIN_FRAMES = 300


def prompt_boundary_is_current(boundary: int | None, current_seq: int) -> bool:
    """Whether a prompt's explicit reset boundary belongs to the current capture session."""
    return boundary is None or boundary <= current_seq + STALE_BOUNDARY_MARGIN_FRAMES


class Bridge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.capture_dir = args.capture_dir.resolve()
        default_control_dir = Path(".runtime") / "alaya_live_control"
        self.control_dir = args.control_dir.resolve() if args.control_dir is not None else default_control_dir
        self.out_dir = args.out_dir.resolve() if args.out_dir is not None else self.control_dir / "upload"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for orphan_sidecar in self.out_dir.glob("window_*.gbuffers.npy"):
            metadata_name = orphan_sidecar.name.removesuffix(".gbuffers.npy") + ".npz"
            if not (self.out_dir / metadata_name).is_file():
                orphan_sidecar.unlink()
        self.staging_dir = self.out_dir / ".window_writer_staging"
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        for stale_stage_path in self.staging_dir.iterdir():
            if stale_stage_path.is_file():
                stale_stage_path.unlink()
        self.manifest_path = self.out_dir / "manifest.json"
        self.control_path = self.out_dir / "control.json"
        self.alias_path = self.out_dir / "window.npz"
        self.gbuffer_alias_path = self.out_dir / "window.gbuffers.npy"
        self.live_control_path = (
            args.live_control.resolve() if args.live_control is not None else self.control_dir / "live_control.json"
        )
        self.input_state_path = (
            args.input_state.resolve() if args.input_state is not None else self.capture_dir / "input.json"
        )
        self.worker_status_path = (
            args.worker_status.resolve() if args.worker_status is not None else self.control_dir / "worker_status.json"
        )

        initial, subsequent = args.cadence
        self.scheduler = WindowScheduler(
            initial_frames=initial,
            subsequent_frames=subsequent,
            fixed_window_frames=args.fixed_window_frames,
            window_stride=args.window_stride,
            ring_capacity=args.ring_capacity,
        )
        previous = read_json_file(self.manifest_path)
        self.source_generation = max(
            0,
            integer(
                previous.get("source_reset_epoch"),
                integer(previous.get("source_generation"), 0),
            ),
        )
        previous_window = previous.get("window")
        if not isinstance(previous_window, Mapping):
            previous_window = {}
        self.window_id = max(
            0,
            integer(previous_window.get("id"), integer(previous.get("window_id"), 0)),
        )
        retained_window_ids = [parse_window_filename(path)[1] for path in self.out_dir.glob("window_*.npz")]
        self.next_window_id = max([self.window_id, *retained_window_ids]) + 1
        previous_transport = str(previous.get("window_transport", "npz") or "npz")
        explicit_authority: dict[str, dict[str, str]] = {}

        def remember_authorized(value: Any) -> None:
            if isinstance(value, Mapping):
                name = str(value.get("immutable_file", "") or "")
                sidecar_name = str(value.get("gbuffers_file", "") or "")
            else:
                name = str(value or "")
                sidecar_name = ""
            if Path(name).name != name or WINDOW_RE.fullmatch(name) is None:
                return
            metadata_path = self.out_dir / name
            if not metadata_path.is_file():
                return
            if previous_transport == "npy":
                expected_sidecar = gbuffer_sidecar_path(metadata_path).name
                if sidecar_name and sidecar_name != expected_sidecar:
                    return
                sidecar_name = expected_sidecar
                if not (self.out_dir / sidecar_name).is_file():
                    return
            explicit_authority[name] = {
                "immutable_file": name,
                "gbuffers_file": sidecar_name,
            }

        remember_authorized(previous.get("window"))
        remember_authorized(previous.get("immutable_file"))
        for key in (
            "retained_window_files",
            "retained_immutable_files",
            "retained_windows",
        ):
            values = previous.get(key, [])
            if isinstance(values, list):
                for value in values:
                    remember_authorized(value)
        self.authorized_window_files: collections.OrderedDict[str, dict[str, str]] = collections.OrderedDict(
            (name, explicit_authority[name])
            for name in sorted(
                explicit_authority,
                key=lambda item: parse_window_filename(Path(item))[1],
            )
        )
        handled = previous.get("handled_worker_recovery_signatures", [])
        if not isinstance(handled, list):
            handled = []
        self.handled_worker_recoveries: Deque[str] = collections.deque(
            (str(value) for value in handled if value), maxlen=32
        )
        previous_recovery = previous.get("last_handled_worker_recovery")
        self.last_handled_worker_recovery = dict(previous_recovery) if isinstance(previous_recovery, Mapping) else {}
        handled_fresh_instances = previous.get("handled_fresh_epoch_worker_instances", [])
        if not isinstance(handled_fresh_instances, list):
            handled_fresh_instances = []
        self.handled_fresh_epoch_worker_instances: Deque[str] = collections.deque(
            (str(value) for value in handled_fresh_instances if value), maxlen=64
        )
        previous_fresh_epoch = previous.get("last_handled_fresh_epoch_request")
        self.last_handled_fresh_epoch_request = (
            dict(previous_fresh_epoch) if isinstance(previous_fresh_epoch, Mapping) else {}
        )
        self.fresh_epoch_request_count = max(0, integer(previous.get("fresh_epoch_request_count"), 0))
        raw_recovery_counts = previous.get("worker_recovery_counts", {})
        self.worker_recovery_counts: dict[str, dict[str, Any]] = {}
        if isinstance(raw_recovery_counts, Mapping):
            for key, value in raw_recovery_counts.items():
                if isinstance(value, Mapping):
                    self.worker_recovery_counts[str(key)] = dict(value)
        previous_cap = previous.get("last_worker_recovery_cap")
        self.last_worker_recovery_cap = dict(previous_cap) if isinstance(previous_cap, Mapping) else {}
        previous_backlog_resync = previous.get("last_worker_backlog_resync")
        self.last_worker_backlog_resync = (
            dict(previous_backlog_resync) if isinstance(previous_backlog_resync, Mapping) else {}
        )
        previous_manual_resync = previous.get("last_manual_input_resync")
        self.last_manual_input_resync = (
            dict(previous_manual_resync) if isinstance(previous_manual_resync, Mapping) else {}
        )
        self.manual_input_resync_count = max(0, integer(previous.get("manual_input_resync_count"), 0))
        self.manual_input_scope_key: tuple[Any, ...] | None = None
        self.manual_input_applied_signature: tuple[str, int, bool, bool] | None = None
        self.manual_input_observed_signature: tuple[str, int, bool, bool] | None = None
        self.manual_input_pending: dict[str, Any] = {}
        self.manual_input_last_token: tuple[int, int, int, int] | None = None
        self.manual_input_last_state: DirectionalInputState | None = None
        self.input_event_watermarks: Deque[dict[str, Any]] = collections.deque(maxlen=32)
        self.manual_input_transition_count = 0
        self.manual_input_coalesced_count = 0
        self.manual_input_duplicate_update_count = 0
        self.worker_lag_token_key: tuple[Any, ...] | None = None
        self.worker_lag_last_observation: tuple[Any, ...] | None = None
        self.worker_lag_observations = 0
        self.worker_lag_latest: dict[str, Any] = {}
        self.current_identity: SourceIdentity | None = None
        self.session_id = ""
        self.current_width = 0
        self.current_height = 0
        self.last_seq: int | None = None
        self.last_source_frame: int | None = None
        self.last_control: dict[str, Any] = clean_control({})
        self.sample_index = 0
        self.previous_window_id = -1
        self.previous_last_seq = -1
        self.enqueued_previous_window_id = -1
        self.enqueued_previous_last_seq = -1
        self.current_window: dict[str, Any] | None = None
        self.deferred_results: Deque[WindowWriteResult] = collections.deque()
        self.reset_reason = ""
        self.reset_at_ns = 0
        self.source_reset_reason = ""
        self.source_reset_at_ns = 0
        self.active_prompt = read_prompt_state(self.live_control_path)
        self.control_key: tuple[Any, ...] | None = None
        self.accept_after_seq: int | None = None
        self.force_current_control_boundary = False
        self.rebound_control_key: tuple[Any, ...] | None = None
        self.rebound_source_session_id = ""
        self.stop_requested = False
        self.pipeline_terminal = False
        self.pipeline_terminal_reason = ""
        self.published_windows = 0
        self.writer_epoch = 0
        self.writer_queue_full_resets = 0
        self.writer_stale_completions = 0
        self.alias_failure_count = 0
        self.last_alias_error: dict[str, Any] = {}
        self.alias_fn: Callable[[Path, Path], None] = atomic_alias
        self.marker_reader: Callable[[Path, bool], CurrentMarkerState] = read_stable_current_marker
        self.manifest_prepare_fn: Callable[[Path, Mapping[str, Any]], Path] = prepare_json_temp
        self.manifest_replace_fn: Callable[[Path, Path], None] = replace_prepared_json
        self.authority_commit_count = 0
        self.authority_durability_failure_count = 0
        self.last_authority_durability_failure: dict[str, Any] = {}
        self.effective_retain_windows = max(
            args.retain_windows,
            args.max_worker_lag_windows + 2,
        )
        self.window_writer = WindowWriter(
            max_pending=args.window_writer_queue_size,
            staging_dir=self.staging_dir,
            initial_epoch=self.writer_epoch,
            write_delay=args.window_writer_test_delay,
        )
        self.slot_writer: DoubleSlotWriter | None = None
        self.slot_reservation: SlotReservation | None = None
        self.slot_busy_observations = 0
        self.slot_last_busy_error = ""
        if args.window_transport == SHM_SLOT_TRANSPORT:
            self.slot_writer = DoubleSlotWriter(
                self.out_dir,
                max(self.scheduler.initial_frames, self.scheduler.subsequent_frames),
                len(MODALITIES),
                args.target_height,
                args.target_width,
            )
        self.last_status = ""
        self.last_error = ""
        self.last_status_write = 0.0
        self.reset_source("bridge_start", identity=None, boundary_seq=None, increment=True)

    def clear_current_window(self) -> None:
        self.current_window = None
        for path in (self.alias_path, self.gbuffer_alias_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def revoke_window_authority(self, status: str, error: str) -> None:
        previous_current_window = self.current_window
        previous_authority = self.authorized_window_files
        retired_authority = [dict(value) for value in previous_authority.values()]
        self.current_window = None
        self.authorized_window_files = collections.OrderedDict()
        self.clear_compatibility_aliases()
        manifest_temporary: Path | None = None
        try:
            prompt = read_prompt_state(self.live_control_path)
            atomic_write_json(self.control_path, self.control_document(prompt))
            manifest_temporary = self.manifest_prepare_fn(
                self.manifest_path,
                self.manifest_document(status, error, prompt),
            )
        except Exception as exc:
            self.current_window = previous_current_window
            self.authorized_window_files = previous_authority
            self.cleanup_authority_temps()
            raise WindowPipelineFailure(f"failed to prepare authority revocation: {type(exc).__name__}: {exc}") from exc
        try:
            self.manifest_replace_fn(manifest_temporary, self.manifest_path)
        except AuthorityManifestReplaceError as exc:
            manifest_temporary.unlink(missing_ok=True)
            if exc.committed:
                self.last_status = status
                self.last_error = error
                self.last_status_write = time.monotonic()
                self.record_authority_durability_failure("revoke", exc)
                self.terminate_window_pipeline("authority revoke committed-but-not-durable")
                raise WindowPipelineFailure(f"authority revoke committed-but-not-durable: {exc}") from exc
            self.current_window = previous_current_window
            self.authorized_window_files = previous_authority
            raise WindowPipelineFailure(f"failed to commit authority revocation: {exc}") from exc
        except Exception as exc:
            manifest_temporary.unlink(missing_ok=True)
            self.current_window = previous_current_window
            self.authorized_window_files = previous_authority
            raise WindowPipelineFailure(f"failed to commit authority revocation: {type(exc).__name__}: {exc}") from exc
        self.last_status = status
        self.last_error = error
        self.last_status_write = time.monotonic()
        self.delete_retired_authority(retired_authority)

    def record_authority_durability_failure(
        self,
        operation: str,
        failure: AuthorityManifestReplaceError,
    ) -> None:
        self.authority_durability_failure_count += 1
        self.last_authority_durability_failure = {
            "operation": operation,
            "phase": failure.phase,
            "committed": failure.committed,
            "error": str(failure),
            "at": time.time(),
        }

    def invalidate_window_jobs(self) -> None:
        self.writer_epoch += 1
        self.window_writer.invalidate(self.writer_epoch)
        if self.slot_writer is not None:
            self.slot_writer.invalidate(self.writer_epoch, "window_jobs_invalidated")
            self.slot_reservation = None
        while self.deferred_results:
            discard_staged_job(self.deferred_results.popleft().job)
        self.window_writer.discard_results()
        self.enqueued_previous_window_id = -1
        self.enqueued_previous_last_seq = -1

    def terminate_window_pipeline(self, reason: str) -> bool:
        if self.pipeline_terminal:
            return not self.window_writer.summary()["thread_alive"]
        self.pipeline_terminal = True
        self.pipeline_terminal_reason = reason
        self.stop_requested = True
        self.writer_epoch += 1
        if self.slot_writer is not None:
            self.slot_writer.invalidate(self.writer_epoch, reason)
            self.slot_reservation = None
        self.scheduler.reset()
        self.enqueued_previous_window_id = -1
        self.enqueued_previous_last_seq = -1
        while self.deferred_results:
            discard_staged_job(self.deferred_results.popleft().job)
        stopped = self.window_writer.shutdown(self.args.window_writer_shutdown_timeout)
        self.window_writer.discard_results()
        self.window_writer.purge_staging()
        return stopped

    def reset_source(
        self,
        reason: str,
        identity: SourceIdentity | None,
        boundary_seq: int | None,
        increment: bool = True,
    ) -> None:
        old_identity = self.current_identity
        allow_local_rebind = bool(
            reason == "source_session_changed"
            and old_identity is not None
            and identity is not None
            and not identity.session_id_derived
            and identity.track == old_identity.track
            and self.active_prompt.active
            and self.active_prompt.track_id == identity.track
        )
        if increment:
            self.source_generation += 1
        self.invalidate_window_jobs()
        self.clear_worker_lag_observation()
        self.scheduler.reset()
        self.current_identity = identity
        self.session_id = identity.source_session_id if identity else ""
        self.current_width = 0
        self.current_height = 0
        self.last_seq = None
        self.last_source_frame = None
        self.last_control = clean_control({})
        self.input_event_watermarks.clear()
        self.sample_index = 0
        self.previous_window_id = -1
        self.previous_last_seq = -1
        self.control_key = None
        self.accept_after_seq = boundary_seq
        self.force_current_control_boundary = bool(
            boundary_seq is None and identity is not None and reason in {"track_changed", "source_session_changed"}
        )
        if allow_local_rebind:
            self.rebound_control_key = self.active_prompt.key
            self.rebound_source_session_id = identity.source_session_id
        elif reason in {
            "bridge_start",
            "track_changed",
            "source_identity_unstable",
        }:
            self.rebound_control_key = None
            self.rebound_source_session_id = ""
        self.source_reset_reason = reason
        self.source_reset_at_ns = time.time_ns()
        self.reset_reason = reason
        self.reset_at_ns = self.source_reset_at_ns
        self.revoke_window_authority("waiting_source", "")
        logging.info(
            "source reset source_generation=%d reason=%s track=%s session=%s",
            self.source_generation,
            reason,
            identity.track if identity else "",
            self.session_id,
        )

    def adopt_source(self, identity: SourceIdentity) -> None:
        self.current_identity = identity
        self.session_id = identity.source_session_id
        self.write_metadata("waiting_control", "")
        logging.info("adopted source track=%s session=%s", identity.track, self.session_id)

    def control_matches_source(self, prompt: PromptState) -> bool:
        identity = self.current_identity
        exact_session = bool(identity is not None and prompt.source_session_id == identity.source_session_id)
        rebound_session = bool(
            identity is not None
            and self.rebound_control_key == prompt.key
            and self.rebound_source_session_id == identity.source_session_id
        )
        return bool(
            identity is not None
            and not identity.session_id_derived
            and prompt.schema_version == SCHEMA_VERSION
            and prompt.track_id == identity.track
            and (exact_session or rebound_session)
        )

    def reset_inference(
        self,
        reason: str,
        prompt: PromptState,
        boundary_seq: int | None,
    ) -> None:
        self.scheduler.reset()
        self.invalidate_window_jobs()
        self.sample_index = 0
        self.previous_window_id = -1
        self.previous_last_seq = -1
        self.active_prompt = prompt
        self.control_key = prompt.key
        self.accept_after_seq = boundary_seq
        self.force_current_control_boundary = False
        self.reset_reason = reason
        self.reset_at_ns = time.time_ns()
        status = "warming" if prompt.active and self.control_matches_source(prompt) else "idle"
        if prompt.active and not self.control_matches_source(prompt):
            status = "waiting_control"
        self.revoke_window_authority(status, "")
        logging.info(
            "inference reset reset_epoch=%d source_generation=%d reason=%s boundary_seq=%s active=%s",
            prompt.reset_epoch,
            self.source_generation,
            reason,
            boundary_seq,
            prompt.active,
        )

    def observe_control(self, prompt: PromptState, current_seq: int) -> bool:
        if self.current_identity and prompt.source_session_id == self.current_identity.source_session_id:
            self.rebound_control_key = None
            self.rebound_source_session_id = ""
        if self.control_key == prompt.key:
            self.active_prompt = prompt
            return False
        source_scope_changed = bool(
            self.control_matches_source(prompt)
            and (
                prompt.track_id != self.active_prompt.track_id
                or prompt.source_session_id != self.active_prompt.source_session_id
            )
        )
        explicit_boundary = prompt.reset_source_seq
        if source_scope_changed:
            # Sequence numbers are local to a capture session. A boundary copied
            # from the previous track/session is not comparable with this source.
            boundary = current_seq
            self.force_current_control_boundary = False
        elif self.force_current_control_boundary and self.control_matches_source(prompt):
            boundary = current_seq
            self.force_current_control_boundary = False
        elif self.control_key is None and self.accept_after_seq is not None:
            boundary = self.accept_after_seq
        else:
            boundary = current_seq if explicit_boundary is None else explicit_boundary
            if not prompt_boundary_is_current(explicit_boundary, current_seq):
                # A prompt keeps its reset boundary across track switches, but
                # sequence numbers restart with every capture session. Waiting
                # for an old session's frame number stalls the stream forever.
                logging.warning(
                    "ignoring stale prompt boundary %s (current_seq=%s); using current_seq",
                    explicit_boundary,
                    current_seq,
                )
                boundary = current_seq
        reason = prompt.reset_reason or prompt.request_reason or "live_control_changed"
        self.reset_inference(reason, prompt, boundary)
        return True

    def identity_reset_reason(self, identity: SourceIdentity) -> str:
        current = self.current_identity
        if current is None:
            return ""
        if identity.track != current.track:
            return "track_changed"
        if identity.token != current.token:
            return "source_session_changed"
        return ""

    def worker_recovery_signature(self, status: Mapping[str, Any]) -> str:
        payload = {
            "worker_instance_id": str(status.get("worker_instance_id", "") or ""),
            "status": str(status.get("status", "") or ""),
            "updated_at": number(status.get("updated_at"), 0.0),
            "error": str(status.get("error", "") or ""),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:24]

    def recovery_entry(self, reset_epoch: int) -> dict[str, Any]:
        key = str(reset_epoch)
        entry = self.worker_recovery_counts.setdefault(
            key,
            {
                "total": 0,
                "failed": 0,
                "needs_reset": 0,
                "needs_reset_instances": [],
                "capped": False,
                "cap_reason": "",
                "cap_error": "",
            },
        )
        if not isinstance(entry.get("needs_reset_instances"), list):
            entry["needs_reset_instances"] = []
        return entry

    def recovery_summary(self, prompt: PromptState) -> dict[str, Any]:
        entry = self.recovery_entry(prompt.reset_epoch)
        return {
            "reset_epoch": prompt.reset_epoch,
            "count": integer(entry.get("total"), 0),
            "failed_count": integer(entry.get("failed"), 0),
            "needs_reset_count": integer(entry.get("needs_reset"), 0),
            "capped": bool(entry.get("capped", False)),
            "cap_reason": str(entry.get("cap_reason", "") or ""),
            "cap_error": str(entry.get("cap_error", "") or ""),
        }

    def cap_worker_recovery(
        self,
        prompt: PromptState,
        status: Mapping[str, Any],
        signature: str,
        reason: str,
    ) -> None:
        entry = self.recovery_entry(prompt.reset_epoch)
        error = str(status.get("error", "") or "")
        entry.update(
            {
                "capped": True,
                "cap_reason": reason,
                "cap_error": error,
                "capped_at": time.time(),
            }
        )
        self.handled_worker_recoveries.append(signature)
        self.last_worker_recovery_cap = {
            "signature": signature,
            "worker_instance_id": str(status.get("worker_instance_id", "") or ""),
            "status": str(status.get("status", "") or ""),
            "reason": reason,
            "error_sha1": hashlib.sha1(error.encode("utf-8")).hexdigest()[:16],
            "capped_at": time.time(),
            "reset_epoch": prompt.reset_epoch,
            "source_reset_epoch": self.source_generation,
        }
        self.scheduler.reset()
        self.invalidate_window_jobs()
        self.sample_index = 0
        self.previous_window_id = -1
        self.previous_last_seq = -1
        self.reset_reason = "worker_recovery_capped"
        self.reset_at_ns = time.time_ns()
        self.revoke_window_authority("worker_failed", error)

    def recovery_is_capped(self, prompt: PromptState) -> bool:
        return bool(self.recovery_entry(prompt.reset_epoch).get("capped", False))

    def clear_worker_lag_observation(self) -> None:
        self.worker_lag_token_key = None
        self.worker_lag_last_observation = None
        self.worker_lag_observations = 0
        self.worker_lag_latest = {}

    def manual_input_scope(self, prompt: PromptState) -> tuple[Any, ...] | None:
        if not prompt.active or self.control_key != prompt.key or not self.control_matches_source(prompt):
            return None
        return prompt.key, self.source_generation, self.session_id

    def remember_input_event(self, state: DirectionalInputState) -> None:
        if not state.input_event_id or state.boundary_source_seq < 0:
            return
        key = (
            state.input_event_id,
            state.track_id,
            state.source_session_id,
            state.reset_epoch,
            state.source_reset_epoch,
        )
        for watermark in reversed(self.input_event_watermarks):
            if watermark["key"] != key:
                continue
            watermark["boundary_source_seq"] = max(
                integer(watermark.get("boundary_source_seq"), -1),
                state.boundary_source_seq,
            )
            watermark["boundary_source_frame"] = max(
                integer(watermark.get("boundary_source_frame"), -1),
                state.boundary_source_frame,
            )
            return
        self.input_event_watermarks.append(
            {
                "key": key,
                "input_event_id": state.input_event_id,
                "boundary_source_seq": state.boundary_source_seq,
                "boundary_source_frame": state.boundary_source_frame,
                "signature": state.signature,
                "track_id": state.track_id,
                "source_session_id": state.source_session_id,
                "reset_epoch": state.reset_epoch,
                "source_reset_epoch": state.source_reset_epoch,
                "observed_at_ns": state.file_mtime_ns,
            }
        )

    def input_event_for_capture(
        self,
        prompt: PromptState,
        source_seq: int,
        source_frame: int,
        captured_at_ns: int,
        control: Mapping[str, Any],
    ) -> str:
        identity = self.current_identity
        if identity is None:
            return ""
        for watermark in reversed(self.input_event_watermarks):
            if (
                watermark.get("track_id") != identity.track
                or watermark.get("source_session_id") != self.session_id
                or integer(watermark.get("reset_epoch"), -1) != prompt.reset_epoch
                or integer(watermark.get("source_reset_epoch"), -1) != self.source_generation
                or captured_at_ns < integer(watermark.get("observed_at_ns"), 0)
                or source_seq < integer(watermark.get("boundary_source_seq"), -1)
            ):
                continue
            boundary_source_frame = integer(watermark.get("boundary_source_frame"), -1)
            if boundary_source_frame >= 0 and source_frame < boundary_source_frame:
                continue
            signature = watermark.get("signature")
            if not isinstance(signature, tuple) or len(signature) != 4:
                continue
            if control_matches_input_signature(control, signature):
                return str(watermark.get("input_event_id", "") or "")
        return ""

    def capture_input_event_id(
        self,
        capture_state: Mapping[str, Any],
        prompt: PromptState,
        source_seq: int,
        source_frame: int,
        captured_at_ns: int,
        control: Mapping[str, Any],
    ) -> str:
        # New STK captures carry the event consumed by that exact game tick.
        # The input.json/control inference below is retained only for older builds.
        if "input_event_id" in capture_state:
            explicit_event_id = clean_input_event_id(capture_state.get("input_event_id"))
            if explicit_event_id.strip():
                return explicit_event_id
        return self.input_event_for_capture(
            prompt,
            source_seq,
            source_frame,
            captured_at_ns,
            control,
        )

    def manual_input_summary(self) -> dict[str, Any]:
        now = time.time()
        last_resync_at = number(self.last_manual_input_resync.get("handled_at"), 0.0)
        source_reset_at = self.source_reset_at_ns / 1_000_000_000
        cooldown_anchor = max(last_resync_at, source_reset_at)
        cooldown_remaining = (
            max(0.0, self.args.manual_input_resync_cooldown - (now - cooldown_anchor)) if cooldown_anchor > 0.0 else 0.0
        )
        pending = self.manual_input_pending
        pending_document: dict[str, Any] = {}
        if pending:
            changed_at = number(pending.get("changed_at"), now)
            pending_document = {
                "signature": directional_signature_document(pending.get("signature")),
                "previous_signature": directional_signature_document(pending.get("previous_signature")),
                "first_observed_at": number(pending.get("first_observed_at"), 0.0),
                "changed_at": changed_at,
                "source_seq": integer(pending.get("source_seq"), -1),
                "input_file_mtime_ns": integer(pending.get("input_file_mtime_ns"), 0),
                "coalesced_transitions": integer(pending.get("coalesced_transitions"), 0),
                "quiet_remaining": round(max(0.0, self.args.manual_input_coalesce - (now - changed_at)), 3),
            }
        return {
            "enabled": self.args.manual_input_resync,
            "coalesce_seconds": self.args.manual_input_coalesce,
            "cooldown_seconds": self.args.manual_input_resync_cooldown,
            "input_max_age_seconds": self.args.manual_input_max_age,
            "cooldown_remaining": round(cooldown_remaining, 3),
            "applied_signature": directional_signature_document(self.manual_input_applied_signature),
            "observed_signature": directional_signature_document(self.manual_input_observed_signature),
            "input": directional_input_document(self.manual_input_last_state),
            "pending": pending_document,
            "transition_count": self.manual_input_transition_count,
            "coalesced_transition_count": self.manual_input_coalesced_count,
            "duplicate_update_count": self.manual_input_duplicate_update_count,
            "resync_count": self.manual_input_resync_count,
            "last_resync": self.last_manual_input_resync,
        }

    def observe_manual_input(
        self,
        prompt: PromptState,
        current_seq: int,
        captured_at_ns: int,
        is_new_capture: bool,
        now: float | None = None,
    ) -> bool:
        observed_at = time.time() if now is None else now
        state = read_directional_input_state(
            self.input_state_path,
            self.args.manual_input_max_age,
            now_ns=int(observed_at * 1_000_000_000),
        )
        if state is None and self.manual_input_last_state is not None:
            previous = self.manual_input_last_state
            age = max(0.0, observed_at - previous.file_mtime_ns / 1_000_000_000)
            if (
                previous.requested_mode == "manual"
                and self.args.manual_input_max_age > 0.0
                and age > self.args.manual_input_max_age
            ):
                state = dataclasses.replace(
                    previous,
                    effective_mode="auto",
                    steer=0,
                    accel=False,
                    brake=False,
                    direction_keys=(),
                    age_seconds=age,
                    fresh=False,
                )
        if state is None:
            return False
        self.manual_input_last_state = state
        self.remember_input_event(state)

        scope = self.manual_input_scope(prompt)
        if not self.args.manual_input_resync or scope is None:
            self.manual_input_scope_key = None
            self.manual_input_applied_signature = None
            self.manual_input_observed_signature = None
            self.manual_input_pending = {}
            self.manual_input_last_token = state.token
            return False

        signature = state.signature
        if self.manual_input_scope_key != scope:
            self.manual_input_scope_key = scope
            self.manual_input_applied_signature = signature
            self.manual_input_observed_signature = signature
            self.manual_input_pending = {}
            self.manual_input_last_token = state.token
            return False

        token_changed = state.token != self.manual_input_last_token
        if signature != self.manual_input_observed_signature:
            self.manual_input_transition_count += 1
            self.manual_input_observed_signature = signature
            if signature == self.manual_input_applied_signature:
                if self.manual_input_pending:
                    self.manual_input_coalesced_count += 1
                self.manual_input_pending = {}
            else:
                previous_pending = self.manual_input_pending
                coalesced = integer(previous_pending.get("coalesced_transitions"), 0)
                if previous_pending:
                    coalesced += 1
                    self.manual_input_coalesced_count += 1
                self.manual_input_pending = {
                    "signature": signature,
                    "previous_signature": self.manual_input_applied_signature,
                    "first_observed_at": number(previous_pending.get("first_observed_at"), observed_at),
                    "changed_at": observed_at,
                    "source_seq": current_seq,
                    "input_file_mtime_ns": state.file_mtime_ns,
                    "coalesced_transitions": coalesced,
                }
        elif token_changed:
            # Heartbeats and alias-equivalent key updates must not move the debounce.
            self.manual_input_duplicate_update_count += 1
        self.manual_input_last_token = state.token

        pending = self.manual_input_pending
        if not pending or not is_new_capture or current_seq < 0:
            return False
        if pending.get("signature") != self.manual_input_observed_signature:
            return False
        if captured_at_ns < integer(pending.get("input_file_mtime_ns"), 0):
            return False
        changed_at = number(pending.get("changed_at"), observed_at)
        if observed_at - changed_at < self.args.manual_input_coalesce:
            return False

        last_resync_at = number(self.last_manual_input_resync.get("handled_at"), 0.0)
        source_reset_at = self.source_reset_at_ns / 1_000_000_000
        cooldown_anchor = max(last_resync_at, source_reset_at)
        if cooldown_anchor > 0.0 and observed_at - cooldown_anchor < self.args.manual_input_resync_cooldown:
            return False

        from_generation = self.source_generation
        boundary_seq = current_seq - 1
        previous_signature = self.manual_input_applied_signature
        self.manual_input_resync_count += 1
        self.last_manual_input_resync = {
            "status": "resynced",
            "reason": "manual_input",
            "handled_at": observed_at,
            "first_observed_at": number(pending.get("first_observed_at"), observed_at),
            "latency_ms": round(
                max(
                    0.0,
                    (observed_at - number(pending.get("first_observed_at"), observed_at)) * 1000.0,
                ),
                1,
            ),
            "source_seq": current_seq,
            "boundary_seq": boundary_seq,
            "from_source_reset_epoch": from_generation,
            "to_source_reset_epoch": from_generation + 1,
            "previous_signature": directional_signature_document(previous_signature),
            "signature": directional_signature_document(pending.get("signature")),
            "coalesced_transitions": integer(pending.get("coalesced_transitions"), 0),
            "reset_epoch": prompt.reset_epoch,
            "render_request_id": prompt.render_request_id,
            "prompt_revision": prompt.prompt_revision,
            "track_id": self.current_identity.track if self.current_identity else "",
            "source_session_id": self.session_id,
        }
        self.manual_input_applied_signature = pending.get("signature")
        self.manual_input_pending = {}
        logging.info(
            "manual input resync signature=%s source_generation=%d boundary_seq=%d",
            self.manual_input_applied_signature,
            from_generation,
            boundary_seq,
        )
        self.reset_source(
            "manual_input",
            identity=self.current_identity,
            boundary_seq=boundary_seq,
            increment=True,
        )
        self.manual_input_scope_key = (
            prompt.key,
            self.source_generation,
            self.session_id,
        )
        return True

    def worker_lag_summary(self) -> dict[str, Any]:
        last_resync_at = number(self.last_worker_backlog_resync.get("handled_at"), 0.0)
        last_resync_at = max(
            last_resync_at,
            number(self.last_manual_input_resync.get("handled_at"), 0.0),
        )
        cooldown_remaining = (
            max(
                0.0,
                self.args.worker_lag_resync_cooldown - (time.time() - last_resync_at),
            )
            if last_resync_at > 0.0
            else 0.0
        )
        return {
            "enabled": self.args.max_worker_lag_windows > 0,
            "max_windows": self.args.max_worker_lag_windows,
            "required_observations": self.args.worker_lag_observations,
            "cooldown_seconds": self.args.worker_lag_resync_cooldown,
            "cooldown_remaining": round(cooldown_remaining, 3),
            "consecutive_high_observations": self.worker_lag_observations,
            **self.worker_lag_latest,
        }

    def handle_worker_fresh_epoch(
        self,
        prompt: PromptState,
        current_seq: int,
    ) -> bool:
        """Honor one post-warmup fresh-source request per worker process."""
        if not prompt.active or not self.control_matches_source(prompt):
            return False
        status = read_json_file(self.worker_status_path)
        status_name = str(status.get("status", "") or "")
        if (
            status_name not in FRESH_SOURCE_STATUSES
            or not bool(status.get("needs_fresh_source", False))
            or not bool(status.get("fresh_epoch_warmup_completed", False))
            or not bool(status.get("warmed", False))
            or not bool(status.get("ready", False))
        ):
            return False

        worker_instance_id = str(status.get("worker_instance_id", "") or "")
        if not worker_instance_id:
            return False
        if worker_instance_id in self.handled_fresh_epoch_worker_instances:
            return False

        updated_at = number(status.get("updated_at"), 0.0)
        age = time.time() - updated_at
        requested_source_epoch = integer(
            status.get(
                "fresh_epoch_requested_source_reset_epoch",
                status.get("source_reset_epoch"),
            ),
            -1,
        )
        if (
            updated_at <= 0.0
            or age < -5.0
            or age > self.args.worker_status_max_age
            or integer(status.get("reset_epoch"), -1) != prompt.reset_epoch
            or requested_source_epoch != self.source_generation
            or integer(status.get("source_reset_epoch"), -1) != self.source_generation
            or integer(status.get("accepted_request_id"), -1) != prompt.render_request_id
            or integer(status.get("prompt_revision"), -1) != prompt.prompt_revision
            or str(status.get("track_id", "") or "") != (self.current_identity.track if self.current_identity else "")
            or str(status.get("source_session_id", "") or "") != self.session_id
        ):
            return False

        from_generation = self.source_generation
        handled_at = time.time()
        request_id = str(status.get("fresh_epoch_request_id", "") or "")
        self.handled_fresh_epoch_worker_instances.append(worker_instance_id)
        self.fresh_epoch_request_count += 1
        self.last_handled_fresh_epoch_request = {
            "status": "resynced",
            "reason": "worker_fresh_epoch",
            "request_id": request_id,
            "worker_instance_id": worker_instance_id,
            "worker_updated_at": updated_at,
            "requested_at": number(status.get("fresh_epoch_requested_at"), 0.0),
            "handled_at": handled_at,
            "latency_ms": round(
                max(
                    0.0,
                    (handled_at - number(status.get("fresh_epoch_requested_at"), handled_at)) * 1000.0,
                ),
                1,
            ),
            "source_seq": current_seq,
            "from_source_reset_epoch": from_generation,
            "to_source_reset_epoch": from_generation + 1,
            "reset_epoch": prompt.reset_epoch,
            "render_request_id": prompt.render_request_id,
            "prompt_revision": prompt.prompt_revision,
            "track_id": self.current_identity.track if self.current_identity else "",
            "source_session_id": self.session_id,
            "warmup_window_id": integer(status.get("fresh_epoch_warmup_window_id"), -1),
            "latest_window_id": integer(status.get("fresh_epoch_latest_window_id"), -1),
            "request_reason": str(status.get("fresh_epoch_reason", "") or ""),
        }
        logging.info(
            "worker fresh epoch request instance=%s source_generation=%d boundary_seq=%d",
            worker_instance_id,
            from_generation,
            current_seq,
        )
        self.reset_source(
            "worker_fresh_epoch",
            identity=self.current_identity,
            boundary_seq=current_seq,
            increment=True,
        )
        return True

    def handle_worker_backlog(
        self,
        prompt: PromptState,
        current_seq: int,
    ) -> bool:
        if self.args.max_worker_lag_windows <= 0:
            self.clear_worker_lag_observation()
            return False
        if not prompt.active or not self.control_matches_source(prompt):
            self.clear_worker_lag_observation()
            return False

        status = read_json_file(self.worker_status_path)
        status_name = str(status.get("status", "") or "")
        if status.get("backlog_eligible") is False:
            self.clear_worker_lag_observation()
            return False
        active_statuses = {
            "rendering",
            "output_queued",
            "streaming",
            "waiting_for_input",
        }
        if not bool(status.get("ready", False)) or status_name not in active_statuses:
            self.clear_worker_lag_observation()
            return False

        updated_at = number(status.get("updated_at"), 0.0)
        age = time.time() - updated_at
        accepted_window_id = integer(status.get("accepted_window_id"), -1)
        worker_instance_id = str(status.get("worker_instance_id", "") or "")
        if (
            not worker_instance_id
            or accepted_window_id < 0
            or updated_at <= 0.0
            or age < -5.0
            or age > self.args.worker_status_max_age
            or integer(status.get("reset_epoch"), -1) != prompt.reset_epoch
            or integer(status.get("source_reset_epoch"), -1) != self.source_generation
            or integer(status.get("accepted_request_id"), -1) != prompt.render_request_id
            or integer(status.get("prompt_revision"), -1) != prompt.prompt_revision
            or str(status.get("track_id", "") or "") != (self.current_identity.track if self.current_identity else "")
            or str(status.get("source_session_id", "") or "") != self.session_id
        ):
            self.clear_worker_lag_observation()
            return False

        lag_windows = max(0, self.window_id - accepted_window_id)
        stream_key = (
            prompt.key,
            self.source_generation,
            self.session_id,
            worker_instance_id,
        )
        if self.worker_lag_token_key != stream_key:
            self.clear_worker_lag_observation()
            self.worker_lag_token_key = stream_key

        observation = (
            worker_instance_id,
            updated_at,
            status_name,
            accepted_window_id,
        )
        self.worker_lag_latest = {
            "status": status_name,
            "worker_instance_id": worker_instance_id,
            "worker_updated_at": updated_at,
            "latest_window_id": self.window_id,
            "accepted_window_id": accepted_window_id,
            "completed_window_id": integer(status.get("completed_window_id"), -1),
            "lag_windows": lag_windows,
        }
        if observation == self.worker_lag_last_observation:
            return False
        self.worker_lag_last_observation = observation

        if lag_windows <= self.args.max_worker_lag_windows:
            self.worker_lag_observations = 0
            return False
        self.worker_lag_observations += 1
        if self.worker_lag_observations < self.args.worker_lag_observations:
            return False

        now = time.time()
        previous_resync_at = number(self.last_worker_backlog_resync.get("handled_at"), 0.0)
        previous_resync_at = max(
            previous_resync_at,
            number(self.last_manual_input_resync.get("handled_at"), 0.0),
        )
        if previous_resync_at > 0.0 and now - previous_resync_at < self.args.worker_lag_resync_cooldown:
            return False

        from_generation = self.source_generation
        self.last_worker_backlog_resync = {
            "status": "resynced",
            "reason": "worker_backlog",
            "handled_at": now,
            "source_seq": current_seq,
            "from_source_reset_epoch": from_generation,
            "to_source_reset_epoch": from_generation + 1,
            "latest_window_id": self.window_id,
            "accepted_window_id": accepted_window_id,
            "completed_window_id": integer(status.get("completed_window_id"), -1),
            "lag_windows": lag_windows,
            "observations": self.worker_lag_observations,
            "worker_instance_id": worker_instance_id,
            "worker_updated_at": updated_at,
        }
        logging.warning(
            "worker backlog resync lag_windows=%d latest_window=%d accepted_window=%d "
            "source_generation=%d boundary_seq=%d",
            lag_windows,
            self.window_id,
            accepted_window_id,
            from_generation,
            current_seq,
        )
        self.reset_source(
            "worker_backlog",
            identity=self.current_identity,
            boundary_seq=current_seq,
            increment=True,
        )
        return True

    def handle_worker_recovery(
        self,
        prompt: PromptState,
        current_seq: int,
    ) -> bool:
        if not prompt.active or not self.control_matches_source(prompt):
            return False
        status = read_json_file(self.worker_status_path)
        status_name = str(status.get("status", "") or "")
        if status_name not in {"needs_reset", "failed"}:
            return False
        updated_at = number(status.get("updated_at"), 0.0)
        age = time.time() - updated_at
        if updated_at <= 0.0 or age < -5.0 or age > self.args.worker_status_max_age:
            return False
        if integer(status.get("reset_epoch"), -1) != prompt.reset_epoch:
            return False
        if integer(status.get("source_reset_epoch"), -1) != self.source_generation:
            return False
        status_session = str(status.get("source_session_id", "") or "")
        if status_session and status_session != self.session_id:
            return False
        status_track = str(status.get("track_id", "") or "")
        if status_track and self.current_identity and status_track != self.current_identity.track:
            return False
        status_revision = integer(status.get("prompt_revision"), prompt.prompt_revision)
        if status_revision != prompt.prompt_revision:
            return False

        signature = self.worker_recovery_signature(status)
        if signature in self.handled_worker_recoveries:
            return False
        entry = self.recovery_entry(prompt.reset_epoch)
        worker_instance_id = str(status.get("worker_instance_id", "") or "")
        needs_instances = entry["needs_reset_instances"]
        if status_name == "needs_reset" and worker_instance_id in needs_instances:
            self.handled_worker_recoveries.append(signature)
            return False
        total = integer(entry.get("total"), 0)
        failed = integer(entry.get("failed"), 0)
        if bool(entry.get("capped", False)):
            self.handled_worker_recoveries.append(signature)
            return False
        if total >= self.args.max_worker_recoveries_per_reset:
            self.cap_worker_recovery(
                prompt,
                status,
                signature,
                "maximum automatic recoveries reached",
            )
            return False
        if status_name == "failed" and failed >= self.args.max_worker_failed_recoveries:
            self.cap_worker_recovery(
                prompt,
                status,
                signature,
                "maximum terminal-failure recoveries reached",
            )
            return False

        self.handled_worker_recoveries.append(signature)
        entry["total"] = total + 1
        if status_name == "failed":
            entry["failed"] = failed + 1
        else:
            entry["needs_reset"] = integer(entry.get("needs_reset"), 0) + 1
            needs_instances.append(worker_instance_id)
        self.last_handled_worker_recovery = {
            "signature": signature,
            "worker_instance_id": str(status.get("worker_instance_id", "") or ""),
            "status": status_name,
            "worker_updated_at": updated_at,
            "error_sha1": hashlib.sha1(str(status.get("error", "") or "").encode("utf-8")).hexdigest()[:16],
            "handled_at": time.time(),
            "reset_epoch": prompt.reset_epoch,
            "source_reset_epoch": self.source_generation,
            "source_seq": current_seq,
        }
        self.reset_source(
            f"worker_{status_name}",
            identity=self.current_identity,
            boundary_seq=current_seq,
            increment=True,
        )
        return True

    def prompt_dict(self, prompt: PromptState) -> dict[str, Any]:
        return {
            "schema_version": prompt.schema_version,
            "reset_epoch": prompt.reset_epoch,
            "reset_reason": prompt.reset_reason,
            "render_request_id": prompt.render_request_id,
            "request_reason": prompt.request_reason,
            "prompt_revision": prompt.prompt_revision,
            "text": prompt.text if prompt.active else "",
            "active": prompt.active,
            "track_id": prompt.track_id,
            "source_session_id": prompt.source_session_id,
            "reset_source_seq": prompt.reset_source_seq,
            "reset_source_frame": prompt.reset_source_frame,
            "requested_at": prompt.requested_at,
        }

    def window_matches_prompt(self, prompt: PromptState) -> bool:
        window = self.current_window
        return bool(
            window
            and prompt.active
            and self.control_matches_source(prompt)
            and integer(window.get("reset_epoch"), -1) == prompt.reset_epoch
            and integer(window.get("render_request_id"), -1) == prompt.render_request_id
            and integer(window.get("prompt_revision"), -1) == prompt.prompt_revision
            and integer(window.get("source_reset_epoch"), -1) == self.source_generation
            and str(window.get("source_session_id", "")) == self.session_id
        )

    def control_document(self, prompt: PromptState) -> dict[str, Any]:
        input_state = read_json_file(self.input_state_path)
        return {
            "schema_version": SCHEMA_VERSION,
            "updated_at_ns": time.time_ns(),
            "generation": self.source_generation,
            "source_generation": self.source_generation,
            "source_reset_epoch": self.source_generation,
            "reset_epoch": prompt.reset_epoch,
            "reset_reason": prompt.reset_reason,
            "request_reason": prompt.request_reason,
            "render_request_id": prompt.render_request_id,
            "prompt_revision": prompt.prompt_revision,
            "reset_source_seq": prompt.reset_source_seq,
            "reset_source_frame": prompt.reset_source_frame,
            "bridge_reset_reason": self.reset_reason,
            "session_id": self.session_id,
            "source_session_id": self.session_id,
            "source_session_id_derived": bool(self.current_identity and self.current_identity.session_id_derived),
            "local_session_rebind": bool(
                self.rebound_control_key == prompt.key and self.rebound_source_session_id == self.session_id
            ),
            "track": self.current_identity.track if self.current_identity else "",
            "track_id": self.current_identity.track if self.current_identity else "",
            "source_seq": self.last_seq,
            "source_frame": self.last_source_frame,
            "sample_index": self.sample_index - 1 if self.sample_index else None,
            "window_transport": self.args.window_transport,
            "kart": self.last_control,
            "input": {
                "mode": str(input_state.get("mode", "") or ""),
                "keys": input_state.get("keys", []) if isinstance(input_state.get("keys"), list) else [],
                "server_time": number(input_state.get("server_time"), 0.0),
                "source": str(input_state.get("source", "") or ""),
            },
            "prompt": self.prompt_dict(prompt),
            "live_control_file": str(self.live_control_path),
            "worker_status_file": str(self.worker_status_path),
            "handled_worker_recovery_signatures": list(self.handled_worker_recoveries),
            "last_handled_worker_recovery": self.last_handled_worker_recovery,
            "handled_fresh_epoch_worker_instances": list(self.handled_fresh_epoch_worker_instances),
            "fresh_epoch_request_count": self.fresh_epoch_request_count,
            "last_handled_fresh_epoch_request": (self.last_handled_fresh_epoch_request),
            "worker_recovery": self.recovery_summary(prompt),
            "worker_recovery_counts": self.worker_recovery_counts,
            "last_worker_recovery_cap": self.last_worker_recovery_cap,
            "worker_lag": self.worker_lag_summary(),
            "last_worker_backlog_resync": self.last_worker_backlog_resync,
            "manual_input_resync": self.manual_input_summary(),
            "manual_input_resync_count": self.manual_input_resync_count,
            "last_manual_input_resync": self.last_manual_input_resync,
            "window_writer": self.window_writer_summary(),
        }

    def committed_live_paths(self) -> list[Path]:
        return [self.out_dir / name for name in self.authorized_window_files]

    def retained_windows(self) -> list[str]:
        return [path.name for path in self.committed_live_paths()]

    def retained_window_files(self) -> list[dict[str, Any]]:
        return [dict(value) for value in self.authorized_window_files.values()]

    def manifest_document(
        self,
        status: str,
        error: str,
        prompt: PromptState,
    ) -> dict[str, Any]:
        immutable_file = str(self.current_window.get("immutable_file", "")) if self.current_window else ""
        immutable_gbuffers_file = str(self.current_window.get("gbuffers_file", "")) if self.current_window else ""
        first_seq = integer(self.current_window.get("first_seq"), -1) if self.current_window else -1
        last_seq = integer(self.current_window.get("last_seq"), -1) if self.current_window else -1
        frame_count = integer(self.current_window.get("frame_count"), 0) if self.current_window else 0
        published_at = number(self.current_window.get("published_at"), 0.0) if self.current_window else 0.0
        track_id = self.current_identity.track if self.current_identity else ""
        alias_status = self.current_window.get("alias_status", {}) if self.current_window else {}
        alias_metadata_ready = bool(alias_status.get("metadata", False))
        alias_gbuffers_ready = bool(alias_status.get("gbuffers", False))
        ready = status == "ready" and self.window_matches_prompt(prompt)
        return {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "ready": ready,
            "error": error,
            "updated_at_ns": time.time_ns(),
            "generation": self.source_generation,
            "source_generation": self.source_generation,
            "reset_epoch": prompt.reset_epoch,
            "render_request_id": prompt.render_request_id,
            "prompt_revision": prompt.prompt_revision,
            "reset_reason": prompt.reset_reason,
            "request_reason": prompt.request_reason,
            "reset_source_seq": prompt.reset_source_seq,
            "reset_source_frame": prompt.reset_source_frame,
            "source_reset_epoch": self.source_generation,
            "source_session_id": self.session_id,
            "source_session_id_derived": bool(self.current_identity and self.current_identity.session_id_derived),
            "local_session_rebind": bool(
                self.rebound_control_key == prompt.key and self.rebound_source_session_id == self.session_id
            ),
            "track_id": track_id,
            "first_seq": first_seq,
            "last_seq": last_seq,
            "window_id": integer(self.current_window.get("id"), 0) if self.current_window else 0,
            "frame_count": frame_count,
            "width": self.args.target_width,
            "height": self.args.target_height,
            "published_at": published_at,
            "reset": {
                "epoch": prompt.reset_epoch,
                "reason": prompt.reset_reason,
                "request_reason": prompt.request_reason,
                "source_seq": prompt.reset_source_seq,
                "source_frame": prompt.reset_source_frame,
                "at_ns": self.reset_at_ns,
            },
            "source_reset": {
                "epoch": self.source_generation,
                "reason": self.source_reset_reason,
                "at_ns": self.source_reset_at_ns,
            },
            "session_id": self.session_id,
            "track": track_id,
            "source": {
                "capture_dir": str(self.capture_dir),
                "seq": self.last_seq,
                "source_frame": self.last_source_frame,
                "width": self.current_width,
                "height": self.current_height,
            },
            "target": {
                "width": self.args.target_width,
                "height": self.args.target_height,
                "origin": "top_left",
                "layout": "T,M,H,W,RGB",
                "dtype": "uint8",
                "modalities": list(MODALITIES),
            },
            "window_transport": self.args.window_transport,
            "cadence": {
                "mode": self.scheduler.mode,
                "latent_window": self.args.latent_window,
                "initial_frames": self.scheduler.initial_frames,
                "subsequent_frames": self.scheduler.subsequent_frames,
                "fixed_window_frames": self.scheduler.fixed_window_frames,
                "window_stride": self.scheduler.window_stride,
            },
            "prompt": prompt.text if prompt.active else "",
            "prompt_active": prompt.active,
            "prompt_control": self.prompt_dict(prompt),
            "immutable_file": immutable_file,
            "window_authority": {
                "type": (
                    "immutable_metadata_plus_shared_slot"
                    if self.args.window_transport == SHM_SLOT_TRANSPORT
                    else "immutable_window_files"
                ),
                "metadata_file": immutable_file,
                "gbuffers_file": immutable_gbuffers_file,
                "aliases_authoritative": False,
            },
            "aliases": {
                "role": "compatibility_only",
                "authoritative": False,
                "metadata_file": self.alias_path.name if alias_metadata_ready else "",
                "gbuffers_file": (
                    self.gbuffer_alias_path.name if immutable_gbuffers_file and alias_gbuffers_ready else ""
                ),
                "failure_count": self.alias_failure_count,
                "last_error": self.last_alias_error,
            },
            "alias_file": self.alias_path.name if alias_metadata_ready else "",
            "gbuffers_file": immutable_gbuffers_file,
            "gbuffers_alias_file": (
                self.gbuffer_alias_path.name if immutable_gbuffers_file and alias_gbuffers_ready else ""
            ),
            "window": self.current_window,
            "retained_windows": self.retained_windows(),
            "retained_immutable_files": self.retained_windows(),
            "retained_window_files": self.retained_window_files(),
            "control_file": self.control_path.name,
            "live_control_file": str(self.live_control_path),
            "worker_status_file": str(self.worker_status_path),
            "handled_worker_recovery_signatures": list(self.handled_worker_recoveries),
            "last_handled_worker_recovery": self.last_handled_worker_recovery,
            "handled_fresh_epoch_worker_instances": list(self.handled_fresh_epoch_worker_instances),
            "fresh_epoch_request_count": self.fresh_epoch_request_count,
            "last_handled_fresh_epoch_request": (self.last_handled_fresh_epoch_request),
            "worker_recovery": self.recovery_summary(prompt),
            "worker_recovery_count": self.recovery_summary(prompt)["count"],
            "worker_recovery_capped": self.recovery_summary(prompt)["capped"],
            "worker_recovery_counts": self.worker_recovery_counts,
            "last_worker_recovery_cap": self.last_worker_recovery_cap,
            "worker_lag": self.worker_lag_summary(),
            "last_worker_backlog_resync": self.last_worker_backlog_resync,
            "manual_input_resync": self.manual_input_summary(),
            "manual_input_resync_count": self.manual_input_resync_count,
            "last_manual_input_resync": self.last_manual_input_resync,
            "window_writer": self.window_writer_summary(),
            "retention": {
                "requested": self.args.retain_windows,
                "effective": self.effective_retain_windows,
            },
        }

    def write_metadata(self, status: str, error: str) -> None:
        prompt = read_prompt_state(self.live_control_path)
        atomic_write_json(self.control_path, self.control_document(prompt))
        atomic_write_json(self.manifest_path, self.manifest_document(status, error, prompt))
        self.last_status = status
        self.last_error = error
        self.last_status_write = time.monotonic()

    def maybe_write_status(self, status: str, error: str = "") -> None:
        now = time.monotonic()
        changed = status != self.last_status or error != self.last_error
        if changed or now - self.last_status_write >= self.args.status_interval:
            self.write_metadata(status, error)

    def report_source_wait(self, status: str, error: str) -> bool:
        has_valid_window = bool(
            self.current_window is not None
            and self.active_prompt.active
            and self.control_matches_source(self.active_prompt)
        )
        if has_valid_window:
            return True
        else:
            self.maybe_write_status(status, error)
            return False

    def cleanup_windows(self) -> None:
        # Deletion is coupled to a successful manifest authority commit. Calling
        # this outside that transaction must never infer authority from disk.
        return

    def next_authorized_set(
        self,
        job: WindowJob,
    ) -> tuple[collections.OrderedDict[str, dict[str, str]], list[dict[str, str]]]:
        desired = collections.OrderedDict((name, dict(value)) for name, value in self.authorized_window_files.items())
        desired[job.immutable_path.name] = {
            "immutable_file": job.immutable_path.name,
            "gbuffers_file": job.sidecar_path.name if job.sidecar_path else "",
        }
        retired: list[dict[str, str]] = []
        while len(desired) > self.effective_retain_windows:
            _, entry = desired.popitem(last=False)
            retired.append(entry)
        return desired, retired

    def delete_retired_authority(self, retired: Sequence[Mapping[str, str]]) -> None:
        for entry in retired:
            metadata_name = str(entry.get("immutable_file", "") or "")
            sidecar_name = str(entry.get("gbuffers_file", "") or "")
            if Path(metadata_name).name != metadata_name:
                continue
            metadata_path = self.out_dir / metadata_name
            try:
                metadata_path.unlink(missing_ok=True)
            except OSError as exc:
                logging.warning("failed to delete retired metadata %s: %s", metadata_name, exc)
            if sidecar_name and Path(sidecar_name).name == sidecar_name:
                try:
                    (self.out_dir / sidecar_name).unlink(missing_ok=True)
                except OSError as exc:
                    logging.warning("failed to delete retired sidecar %s: %s", sidecar_name, exc)

    def confirm_source_identity(self, boundary_seq: int) -> bool:
        try:
            identity = read_source_identity(
                self.capture_dir,
                settle_seconds=0.0,
                require_metadata=not self.args.allow_missing_metadata,
            )
        except SourceNotReady:
            self.reset_source(
                "source_identity_unstable",
                identity=None,
                boundary_seq=boundary_seq,
                increment=True,
            )
            return False
        reason = self.identity_reset_reason(identity)
        if reason:
            self.reset_source(
                reason,
                identity=identity,
                boundary_seq=boundary_seq,
                increment=True,
            )
            return False
        return True

    def window_writer_summary(self) -> dict[str, Any]:
        return {
            **self.window_writer.summary(),
            "queue_capacity": self.args.window_writer_queue_size,
            "queue_full_resets": self.writer_queue_full_resets,
            "stale_completions": self.writer_stale_completions,
            "deferred_results": len(self.deferred_results),
            "staging_files": sum(1 for path in self.staging_dir.iterdir() if path.is_file()),
            "terminal": self.pipeline_terminal,
            "terminal_reason": self.pipeline_terminal_reason,
            "authority_commits": self.authority_commit_count,
            "authorized_windows": len(self.authorized_window_files),
            "authority_durability_failures": self.authority_durability_failure_count,
            "last_authority_durability_failure": self.last_authority_durability_failure,
            "shared_slot_busy_observations": self.slot_busy_observations,
            "shared_slot_last_busy_error": self.slot_last_busy_error,
        }

    def worker_accepted_window_id_for_slots(self) -> int:
        status = read_json_file(self.worker_status_path)
        prompt = self.active_prompt
        if not status or not prompt.active:
            return -1
        if integer(status.get("reset_epoch"), -1) != prompt.reset_epoch:
            return -1
        if integer(status.get("source_reset_epoch"), -1) != self.source_generation:
            return -1
        status_track = str(status.get("track_id", "") or "")
        status_session = str(status.get("source_session_id", "") or "")
        if status_track and self.current_identity and status_track != self.current_identity.track:
            return -1
        if status_session and status_session != self.session_id:
            return -1
        return max(
            integer(status.get("accepted_window_id"), -1),
            integer(status.get("completed_window_id"), -1),
        )

    def ensure_shared_slot_reservation(self) -> SlotReservation | None:
        if self.slot_writer is None:
            return None
        if self.slot_reservation is not None:
            return self.slot_reservation
        required = self.scheduler.initial_frames if self.scheduler.published == 0 else self.scheduler.subsequent_frames
        try:
            self.slot_reservation = self.slot_writer.begin(
                self.next_window_id,
                self.writer_epoch,
                required,
                self.worker_accepted_window_id_for_slots(),
            )
        except SlotBusy as exc:
            self.slot_busy_observations += 1
            error = str(exc)
            if error != self.slot_last_busy_error:
                logging.warning("shared-slot capture paused: %s", error)
                self.slot_last_busy_error = error
            return None
        return self.slot_reservation

    def commit_shared_slot(self, job: WindowJob) -> WindowJob:
        if self.slot_writer is None:
            return job
        reservation = self.slot_reservation
        if reservation is None:
            raise WindowPipelineFailure("shared-slot window has no active reservation")
        if reservation.window_id != job.window_id:
            raise WindowPipelineFailure(f"shared-slot window id {reservation.window_id} != job {job.window_id}")
        if reservation.frame_count != len(job.frames):
            raise WindowPipelineFailure(f"shared-slot frame count {reservation.frame_count} != job {len(job.frames)}")
        reference = self.slot_writer.commit(
            reservation,
            {
                "source_reset_epoch": job.fence.source_generation,
                "reset_epoch": job.prompt.reset_epoch,
                "render_request_id": job.prompt.render_request_id,
                "prompt_revision": job.prompt.prompt_revision,
                "track_id": job.fence.track_id,
                "source_session_id": job.fence.source_session_id,
                "first_seq": job.frames[0].seq,
                "last_seq": job.frames[-1].seq,
            },
        )
        self.slot_reservation = None
        return dataclasses.replace(job, slot_reference=reference)

    def make_window_job(self, plan: WindowPlan, prompt: PromptState) -> WindowJob:
        identity = self.current_identity
        if identity is None:
            raise WindowPipelineFailure("cannot enqueue a window without source identity")
        fence = WindowFence(
            writer_epoch=self.writer_epoch,
            source_generation=self.source_generation,
            source_session_id=self.session_id,
            source_identity_token=identity.token,
            track_id=identity.track,
            prompt_key=prompt.key,
        )
        return WindowJob(
            fence=fence,
            window_id=self.next_window_id,
            frames=tuple(plan.frames),
            is_initial=plan.is_initial,
            mode=plan.mode,
            prompt=prompt,
            previous_window_id=self.enqueued_previous_window_id,
            previous_last_seq=self.enqueued_previous_last_seq,
            window_transport=self.args.window_transport,
            slot_reference=None,
            compress=self.args.compress,
            out_dir=self.out_dir,
            staging_dir=self.staging_dir,
            stage_token=f"{os.getpid()}_{uuid.uuid4().hex}",
            source_width=self.current_width,
            source_height=self.current_height,
            target_width=self.args.target_width,
            target_height=self.args.target_height,
            latent_window=self.args.latent_window,
            initial_pixel_frames=self.scheduler.initial_frames,
            resumed_pixel_frames=self.scheduler.subsequent_frames,
            created_at_ns=time.time_ns(),
        )

    def completion_fence_matches(
        self,
        job: WindowJob,
        prompt: PromptState,
    ) -> bool:
        identity = self.current_identity
        fence = job.fence
        return bool(
            not self.pipeline_terminal
            and fence.writer_epoch == self.writer_epoch
            and fence.source_generation == self.source_generation
            and fence.source_session_id == self.session_id
            and identity is not None
            and fence.source_identity_token == identity.token
            and fence.track_id == identity.track
            and fence.prompt_key == self.control_key
            and prompt.key == fence.prompt_key
            and prompt.active
            and self.control_matches_source(prompt)
            and job.previous_window_id == self.previous_window_id
            and job.previous_last_seq == self.previous_last_seq
        )

    def discard_completion(self, result: WindowWriteResult) -> None:
        discard_staged_job(result.job)
        self.writer_stale_completions += 1

    def validate_current_marker_for_promotion(self, job: WindowJob) -> CurrentMarkerState:
        try:
            marker = self.marker_reader(
                self.capture_dir,
                not self.args.allow_missing_metadata,
            )
        except SourceNotReady as exc:
            raise PromotionDeferred(str(exc)) from exc

        identity = marker.identity
        reason = self.identity_reset_reason(identity)
        if reason:
            raise PromotionReset(reason, identity, marker.seq)
        if (
            identity.token != job.fence.source_identity_token
            or identity.source_session_id != job.fence.source_session_id
            or identity.track != job.fence.track_id
        ):
            raise PromotionReset("source_session_changed", identity, marker.seq)
        if (marker.width, marker.height) != (job.source_width, job.source_height):
            raise PromotionReset("resolution_changed", identity, marker.seq)

        last = job.frames[-1]
        if self.last_seq is not None and marker.seq < self.last_seq:
            raise PromotionReset("seq_rollback", identity, marker.seq)
        if self.last_source_frame is not None and marker.source_frame < self.last_source_frame:
            raise PromotionReset("source_frame_rollback", identity, marker.seq)
        if marker.seq < last.seq:
            raise PromotionReset("seq_rollback", identity, marker.seq)
        if marker.source_frame < last.source_frame:
            raise PromotionReset("source_frame_rollback", identity, marker.seq)
        return marker

    def promote_staged_pair(self, result: WindowWriteResult) -> bool:
        job = result.job
        staged_metadata = job.staged_metadata_path
        staged_sidecar = job.staged_sidecar_path
        live_metadata = job.immutable_path
        live_sidecar = job.sidecar_path
        if self.pipeline_terminal or self.stop_requested:
            discard_staged_job(job)
            return False
        if not staged_metadata.is_file():
            raise WindowPipelineFailure(f"staged metadata disappeared for window {job.window_id}")
        if staged_metadata.stat().st_size != result.metadata_size_bytes:
            raise WindowPipelineFailure(f"staged metadata size changed for window {job.window_id}")
        if live_metadata.exists() or (live_sidecar is not None and live_sidecar.exists()):
            raise WindowPipelineFailure(f"refusing to overwrite committed window {job.window_id}")

        sidecar_promoted = False
        try:
            if staged_sidecar is not None and live_sidecar is not None:
                if not staged_sidecar.is_file():
                    raise FileNotFoundError(staged_sidecar)
                if staged_sidecar.stat().st_size != result.sidecar_size_bytes:
                    raise RuntimeError("staged sidecar size changed before promotion")
                os.rename(staged_sidecar, live_sidecar)
                sidecar_promoted = True
                if self.pipeline_terminal or self.stop_requested:
                    live_sidecar.unlink(missing_ok=True)
                    discard_staged_job(job)
                    return False

            # Live placement remains unauthorized until the manifest transaction.
            os.rename(staged_metadata, live_metadata)
            return True
        except Exception as exc:
            if not live_metadata.exists():
                if sidecar_promoted and live_sidecar is not None:
                    live_sidecar.unlink(missing_ok=True)
                discard_staged_job(job)
            raise WindowPipelineFailure(
                f"failed to promote window {job.window_id}: {type(exc).__name__}: {exc}"
            ) from exc

    def advance_aliases_best_effort(self, job: WindowJob) -> dict[str, Any]:
        errors: list[dict[str, str]] = []
        sidecar_ok = job.sidecar_path is None and job.slot_reference is None
        metadata_ok = False
        if job.sidecar_path is not None:
            try:
                self.alias_fn(job.sidecar_path, self.gbuffer_alias_path)
                sidecar_ok = True
            except Exception as exc:
                errors.append({"alias": self.gbuffer_alias_path.name, "error": f"{type(exc).__name__}: {exc}"})
        else:
            try:
                self.gbuffer_alias_path.unlink()
            except FileNotFoundError:
                pass
            except Exception as exc:
                errors.append({"alias": self.gbuffer_alias_path.name, "error": f"{type(exc).__name__}: {exc}"})
        try:
            self.alias_fn(job.immutable_path, self.alias_path)
            metadata_ok = True
        except Exception as exc:
            errors.append({"alias": self.alias_path.name, "error": f"{type(exc).__name__}: {exc}"})
        if errors:
            for alias_path in (self.alias_path, self.gbuffer_alias_path):
                try:
                    alias_path.unlink()
                except FileNotFoundError:
                    pass
                except Exception as exc:
                    errors.append(
                        {
                            "alias": alias_path.name,
                            "error": f"cleanup {type(exc).__name__}: {exc}",
                        }
                    )
            metadata_ok = False
            sidecar_ok = False
            self.alias_failure_count += len(errors)
            self.last_alias_error = {
                "window_id": job.window_id,
                "at": time.time(),
                "errors": errors,
            }
            logging.warning("compatibility alias update failed: %s", errors)
        return {
            "metadata": metadata_ok,
            "gbuffers": sidecar_ok,
            "errors": errors,
        }

    def clear_compatibility_aliases(self) -> None:
        for alias_path in (self.alias_path, self.gbuffer_alias_path):
            try:
                alias_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                logging.warning("failed to clear compatibility alias %s: %s", alias_path, exc)

    def delete_unauthorized_pair(self, job: WindowJob) -> None:
        job.immutable_path.unlink(missing_ok=True)
        if job.sidecar_path is not None:
            job.sidecar_path.unlink(missing_ok=True)
        discard_staged_job(job)

    def cleanup_authority_temps(self) -> None:
        pattern = f".{self.manifest_path.name}.*.authority.tmp"
        for temporary in self.out_dir.glob(pattern):
            temporary.unlink(missing_ok=True)

    def apply_promotion_reset(self, reset: PromotionReset) -> None:
        self.reset_source(
            reset.reason,
            identity=reset.identity,
            boundary_seq=reset.boundary_seq,
            increment=True,
        )

    def window_metadata_from_result(
        self,
        result: WindowWriteResult,
        alias_status: Mapping[str, Any],
    ) -> dict[str, Any]:
        job = result.job
        frames = job.frames
        first = frames[0]
        last = frames[-1]
        published_at = time.time()
        return {
            "id": job.window_id,
            "window_transport": job.window_transport,
            "immutable_file": job.immutable_path.name,
            "alias_file": self.alias_path.name if alias_status["metadata"] else "",
            "metadata_size_bytes": result.metadata_size_bytes,
            "gbuffers_file": job.payload_file_name,
            "gbuffers_alias_file": (
                self.gbuffer_alias_path.name if job.sidecar_path and alias_status["gbuffers"] else ""
            ),
            "aliases_authoritative": False,
            "alias_status": dict(alias_status),
            "gbuffers_file_size_bytes": result.sidecar_size_bytes,
            "gbuffers_payload_nbytes": result.payload_nbytes,
            "is_initial": job.is_initial,
            "mode": job.mode,
            "latent_window": job.latent_window,
            "initial_pixel_frames": job.initial_pixel_frames,
            "resumed_pixel_frames": job.resumed_pixel_frames,
            "generation": job.fence.source_generation,
            "source_generation": job.fence.source_generation,
            "source_reset_epoch": job.fence.source_generation,
            "reset_epoch": job.prompt.reset_epoch,
            "render_request_id": job.prompt.render_request_id,
            "reset_reason": job.prompt.reset_reason,
            "request_reason": job.prompt.request_reason,
            "reset_source_seq": job.prompt.reset_source_seq,
            "reset_source_frame": job.prompt.reset_source_frame,
            "session_id": job.fence.source_session_id,
            "source_session_id": job.fence.source_session_id,
            "track": job.fence.track_id,
            "track_id": job.fence.track_id,
            "frame_count": len(frames),
            "sample_start": first.sample_index,
            "sample_end": last.sample_index,
            "expected_next_sample_index": last.sample_index + 1,
            "first_seq": first.seq,
            "last_seq": last.seq,
            "next_source_seq_must_exceed": last.seq,
            "first_source_frame": first.source_frame,
            "last_source_frame": last.source_frame,
            "previous_window_id": job.previous_window_id,
            "previous_last_seq": job.previous_last_seq,
            "shape": list(result.shape),
            "dtype": "uint8",
            "modalities": list(MODALITIES),
            "created_at_ns": job.created_at_ns,
            "writer_completed_at_ns": result.completed_at_ns,
            "writer_latency_ms": max(0.0, (result.completed_at_ns - job.created_at_ns) / 1_000_000),
            "published_at": published_at,
            "published_at_ns": int(published_at * 1_000_000_000),
            "size_bytes": result.metadata_size_bytes,
            "total_size_bytes": (result.metadata_size_bytes + result.sidecar_size_bytes),
            "prompt_revision": job.prompt.prompt_revision,
            "prompt": job.prompt.text if job.prompt.active else "",
            "prompt_active": job.prompt.active,
            **(job.slot_reference.metadata() if job.slot_reference is not None else {}),
        }

    def commit_window_result(self, result: WindowWriteResult) -> None:
        job = result.job
        if result.stale:
            self.discard_completion(result)
            return
        if result.error is not None:
            raise WindowPipelineFailure(
                f"window writer failed while {result.error_context}: {type(result.error).__name__}: {result.error}"
            ) from result.error

        latest_prompt = read_prompt_state(self.live_control_path)
        if not self.completion_fence_matches(job, latest_prompt):
            self.discard_completion(result)
            if latest_prompt.key != job.fence.prompt_key:
                self.observe_control(latest_prompt, job.frames[-1].seq)
            return
        try:
            self.validate_current_marker_for_promotion(job)
        except PromotionReset as reset:
            self.discard_completion(result)
            self.apply_promotion_reset(reset)
            return

        confirmed_prompt = read_prompt_state(self.live_control_path)
        if not self.completion_fence_matches(job, confirmed_prompt):
            self.discard_completion(result)
            if confirmed_prompt.key != job.fence.prompt_key:
                self.observe_control(confirmed_prompt, job.frames[-1].seq)
            return
        try:
            self.validate_current_marker_for_promotion(job)
        except PromotionReset as reset:
            self.discard_completion(result)
            self.apply_promotion_reset(reset)
            return
        if not self.promote_staged_pair(result):
            return

        first = job.frames[0]
        last = job.frames[-1]
        try:
            self.validate_current_marker_for_promotion(job)
        except PromotionDeferred:
            self.clear_compatibility_aliases()
            self.delete_unauthorized_pair(job)
            self.reset_source(
                "source_identity_unstable",
                identity=None,
                boundary_seq=last.seq,
                increment=True,
            )
            return
        except PromotionReset as reset:
            self.clear_compatibility_aliases()
            self.delete_unauthorized_pair(job)
            self.apply_promotion_reset(reset)
            return

        post_promote_prompt = read_prompt_state(self.live_control_path)
        if not self.completion_fence_matches(job, post_promote_prompt):
            self.clear_compatibility_aliases()
            self.delete_unauthorized_pair(job)
            if post_promote_prompt.key != job.fence.prompt_key:
                self.observe_control(post_promote_prompt, last.seq)
            return

        pending_alias_status = {
            "metadata": False,
            "gbuffers": False,
            "errors": [],
        }
        candidate_window = self.window_metadata_from_result(result, pending_alias_status)
        desired_authority, retired_authority = self.next_authorized_set(job)
        previous_current_window = self.current_window
        previous_authority = self.authorized_window_files
        previous_window_id = self.window_id
        self.current_window = candidate_window
        self.authorized_window_files = desired_authority
        self.window_id = job.window_id

        manifest_temporary: Path | None = None

        def restore_provisional_state() -> None:
            self.current_window = previous_current_window
            self.authorized_window_files = previous_authority
            self.window_id = previous_window_id

        try:
            atomic_write_json(self.control_path, self.control_document(job.prompt))
            manifest_document = self.manifest_document("ready", "", job.prompt)
            manifest_temporary = self.manifest_prepare_fn(self.manifest_path, manifest_document)
        except Exception as exc:
            restore_provisional_state()
            self.cleanup_authority_temps()
            self.clear_compatibility_aliases()
            self.delete_unauthorized_pair(job)
            raise WindowPipelineFailure(f"failed to prepare authority manifest: {type(exc).__name__}: {exc}") from exc

        final_prompt = read_prompt_state(self.live_control_path)
        if not self.completion_fence_matches(job, final_prompt):
            manifest_temporary.unlink(missing_ok=True)
            restore_provisional_state()
            self.clear_compatibility_aliases()
            self.delete_unauthorized_pair(job)
            if final_prompt.key != job.fence.prompt_key:
                self.observe_control(final_prompt, last.seq)
            return
        try:
            self.validate_current_marker_for_promotion(job)
            self.manifest_replace_fn(manifest_temporary, self.manifest_path)
        except PromotionDeferred:
            manifest_temporary.unlink(missing_ok=True)
            restore_provisional_state()
            self.clear_compatibility_aliases()
            self.delete_unauthorized_pair(job)
            self.reset_source(
                "source_identity_unstable",
                identity=None,
                boundary_seq=last.seq,
                increment=True,
            )
            return
        except PromotionReset as reset:
            manifest_temporary.unlink(missing_ok=True)
            restore_provisional_state()
            self.clear_compatibility_aliases()
            self.delete_unauthorized_pair(job)
            self.apply_promotion_reset(reset)
            return
        except AuthorityManifestReplaceError as exc:
            manifest_temporary.unlink(missing_ok=True)
            if exc.committed:
                self.last_status = "ready"
                self.last_error = ""
                self.last_status_write = time.monotonic()
                self.previous_window_id = job.window_id
                self.previous_last_seq = last.seq
                self.authority_commit_count += 1
                self.published_windows += 1
                self.clear_compatibility_aliases()
                self.record_authority_durability_failure("publish", exc)
                self.terminate_window_pipeline("authority publish committed-but-not-durable")
                raise WindowPipelineFailure(f"authority publish committed-but-not-durable: {exc}") from exc
            restore_provisional_state()
            self.clear_compatibility_aliases()
            self.delete_unauthorized_pair(job)
            raise WindowPipelineFailure(f"failed to commit authority manifest: {exc}") from exc
        except Exception as exc:
            manifest_temporary.unlink(missing_ok=True)
            restore_provisional_state()
            self.clear_compatibility_aliases()
            self.delete_unauthorized_pair(job)
            raise WindowPipelineFailure(f"failed to commit authority manifest: {type(exc).__name__}: {exc}") from exc

        self.last_status = "ready"
        self.last_error = ""
        self.last_status_write = time.monotonic()
        self.previous_window_id = job.window_id
        self.previous_last_seq = last.seq
        self.authority_commit_count += 1
        self.published_windows += 1
        self.delete_retired_authority(retired_authority)

        post_commit_prompt = read_prompt_state(self.live_control_path)
        if post_commit_prompt.key != job.fence.prompt_key:
            self.observe_control(post_commit_prompt, last.seq)
            return

        # Compatibility aliases are visible filesystem entries. Advance them
        # only after the immutable pair is authorized by manifest replacement.
        alias_status = self.advance_aliases_best_effort(job)
        candidate_window["alias_file"] = self.alias_path.name if alias_status["metadata"] else ""
        candidate_window["gbuffers_alias_file"] = (
            self.gbuffer_alias_path.name if job.sidecar_path is not None and alias_status["gbuffers"] else ""
        )
        candidate_window["alias_status"] = dict(alias_status)

        # The authority is already durable, so this compatibility-status
        # refresh is best effort. A later periodic status write converges it if
        # the immediate update fails.
        try:
            refreshed_prompt = read_prompt_state(self.live_control_path)
            if refreshed_prompt.key != job.fence.prompt_key:
                self.observe_control(refreshed_prompt, last.seq)
                return
            atomic_write_json(
                self.manifest_path,
                self.manifest_document("ready", "", refreshed_prompt),
            )
            self.last_status_write = time.monotonic()
        except Exception as exc:
            self.last_status_write = 0.0
            logging.warning(
                "failed to refresh compatibility alias status after authority commit: %s: %s",
                type(exc).__name__,
                exc,
            )
        logging.info(
            "published %s transport=%s frames=%d samples=%d..%d seq=%d..%d "
            "reset_epoch=%d source_generation=%d writer_ms=%.1f",
            job.immutable_path.name,
            job.window_transport,
            len(job.frames),
            first.sample_index,
            last.sample_index,
            first.seq,
            last.seq,
            job.prompt.reset_epoch,
            job.fence.source_generation,
            candidate_window["writer_latency_ms"],
        )

    def process_writer_completions(self) -> None:
        if self.pipeline_terminal:
            while self.deferred_results:
                discard_staged_job(self.deferred_results.popleft().job)
            self.window_writer.discard_results()
            return
        writer_error = str(self.window_writer.summary().get("fatal_error", "") or "")
        if writer_error:
            raise WindowPipelineFailure(writer_error)
        while True:
            result = self.window_writer.poll()
            if result is None:
                break
            self.deferred_results.append(result)
        while self.deferred_results:
            result = self.deferred_results.popleft()
            try:
                self.commit_window_result(result)
            except PromotionDeferred:
                self.deferred_results.appendleft(result)
                return
            except WindowPipelineFailure:
                raise
            except Exception as exc:
                raise WindowPipelineFailure(
                    f"window completion finalization failed: {type(exc).__name__}: {exc}"
                ) from exc

    def wait_for_writer_idle(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.process_writer_completions()
            summary = self.window_writer.summary()
            if summary["pending_jobs"] == 0 and summary["active_jobs"] == 0:
                self.process_writer_completions()
                return
            time.sleep(0.001)
        raise TimeoutError("window writer did not become idle")

    def publish(self, plan: WindowPlan, prompt: PromptState) -> None:
        if self.pipeline_terminal or self.stop_requested:
            return
        if not plan.frames:
            raise RuntimeError("cannot publish an empty window")
        latest_prompt = read_prompt_state(self.live_control_path)
        if (
            latest_prompt.key != self.control_key
            or not latest_prompt.active
            or not self.control_matches_source(latest_prompt)
        ):
            self.observe_control(latest_prompt, plan.frames[-1].seq)
            return
        prompt = latest_prompt
        if not self.confirm_source_identity(plan.frames[-1].seq):
            return
        job = self.make_window_job(plan, prompt)
        if self.slot_writer is not None:
            job = self.commit_shared_slot(job)
        try:
            self.window_writer.submit(job)
        except queue.Full:
            self.writer_queue_full_resets += 1
            logging.error(
                "window writer queue full at seq=%d; resetting causal history",
                plan.frames[-1].seq,
            )
            self.reset_source(
                "window_writer_queue_full",
                self.current_identity,
                boundary_seq=plan.frames[-1].seq,
                increment=True,
            )
            return
        self.next_window_id += 1
        self.enqueued_previous_window_id = job.window_id
        self.enqueued_previous_last_seq = job.frames[-1].seq

    def accept_capture(self, capture: RawCapture, prompt: PromptState) -> None:
        state = capture.state
        seq = integer(state.get("seq"), -1)
        source_frame = integer(state.get("source_frame"), seq)
        width = integer(state.get("width"), 0)
        height = integer(state.get("height"), 0)
        previous_seq = self.last_seq
        previous_source_frame = self.last_source_frame
        source_was_reset = False
        if self.current_width and (width, height) != (
            self.current_width,
            self.current_height,
        ):
            self.reset_source(
                "resolution_changed",
                self.current_identity,
                boundary_seq=seq,
                increment=True,
            )
            source_was_reset = True
        elif previous_seq is not None and seq < previous_seq:
            self.reset_source(
                "seq_rollback",
                self.current_identity,
                boundary_seq=seq,
                increment=True,
            )
            source_was_reset = True
        elif previous_source_frame is not None and source_frame < previous_source_frame:
            self.reset_source(
                "source_frame_rollback",
                self.current_identity,
                boundary_seq=seq,
                increment=True,
            )
            source_was_reset = True

        self.observe_control(prompt, seq)
        control = clean_control(state.get("control"))
        input_event_mode = str(state.get("input_event_mode", "") or "").lower()
        if not control.get("mode") and input_event_mode in {"auto", "manual"}:
            control["mode"] = input_event_mode
        duplicate = not source_was_reset and previous_seq is not None and seq == previous_seq
        manual_resynced = self.observe_manual_input(
            prompt,
            current_seq=seq,
            captured_at_ns=capture.marker.mtime_ns,
            is_new_capture=not duplicate,
        )
        if manual_resynced:
            # reset_source clears the prompt binding; rebind it to the same fenced
            # request before accepting the first frame of the new source epoch.
            self.observe_control(prompt, seq)
            source_was_reset = True
            duplicate = False
        self.current_width = width
        self.current_height = height
        self.last_seq = seq
        self.last_source_frame = source_frame
        self.last_control = control

        if prompt.schema_version != SCHEMA_VERSION:
            self.maybe_write_status("waiting_control", "live_control.json schema_version must be 2")
            return
        if not prompt.active:
            self.maybe_write_status("idle", "")
            return
        if not self.control_matches_source(prompt):
            identity = self.current_identity
            if identity is not None and identity.session_id_derived:
                error = "run_config.env is missing SOURCE_SESSION_ID"
            else:
                error = "live control track/session does not match the capture source"
            self.maybe_write_status("waiting_control", error)
            return
        if self.recovery_is_capped(prompt):
            recovery = self.recovery_summary(prompt)
            self.maybe_write_status("worker_failed", recovery["cap_error"])
            return
        if duplicate:
            self.maybe_write_status("warming" if not self.current_window else "ready")
            return
        if self.accept_after_seq is not None and seq <= self.accept_after_seq:
            self.maybe_write_status("warming", "waiting for a frame after reset boundary")
            return

        slot_reservation = None
        if self.slot_writer is not None:
            slot_reservation = self.ensure_shared_slot_reservation()
            if slot_reservation is None:
                return

        gbuffers = decode_gbuffers(
            capture.raw,
            width,
            height,
            self.args.target_width,
            self.args.target_height,
        )
        if slot_reservation is not None:
            self.slot_writer.write_frame(
                slot_reservation,
                len(self.scheduler.frames),
                gbuffers,
            )
        frame_gbuffers = np.empty((0,), dtype=np.uint8) if slot_reservation is not None else gbuffers
        frame = Frame(
            sample_index=self.sample_index,
            seq=seq,
            source_frame=source_frame,
            captured_at_ns=capture.marker.mtime_ns,
            gbuffers=frame_gbuffers,
            control=control,
            input_event_id=self.capture_input_event_id(
                state,
                prompt,
                seq,
                source_frame,
                capture.marker.mtime_ns,
                control,
            ),
        )
        self.scheduler.append(frame)
        self.sample_index += 1
        self.maybe_write_status("warming" if not self.current_window else "ready")
        if self.scheduler.ready():
            self.publish(self.scheduler.take(), prompt)

    def run(self) -> int:
        interval = 1.0 / self.args.target_fps if self.args.target_fps > 0 else 0.0
        next_sample_at = 0.0
        exit_code = 0
        while not self.stop_requested:
            try:
                self.process_writer_completions()
                identity = read_source_identity(
                    self.capture_dir,
                    self.args.identity_settle,
                    not self.args.allow_missing_metadata,
                )
                reason = self.identity_reset_reason(identity)
                if reason:
                    self.reset_source(
                        reason,
                        identity,
                        boundary_seq=None,
                        increment=True,
                    )
                elif self.current_identity is None:
                    self.adopt_source(identity)

                sample_started_at = wait_for_sample_deadline(next_sample_at)
                capture = read_committed_capture(
                    self.capture_dir,
                    identity,
                    self.args.expected_width,
                    self.args.expected_height,
                )
                marker_age = max(0.0, (time.time_ns() - capture.marker.mtime_ns) / 1e9)
                if self.args.stale_after > 0 and marker_age > self.args.stale_after:
                    self.report_source_wait("stale", f"latest source frame is {marker_age:.2f}s old")
                    time.sleep(self.args.poll_interval)
                    continue
                prompt = read_prompt_state(self.live_control_path)
                freshened = self.handle_worker_fresh_epoch(prompt, integer(capture.state.get("seq"), -1))
                recovered = False
                if not freshened:
                    recovered = self.handle_worker_recovery(prompt, integer(capture.state.get("seq"), -1))
                if not freshened and not recovered:
                    self.handle_worker_backlog(prompt, integer(capture.state.get("seq"), -1))
                before = self.sample_index
                self.accept_capture(capture, prompt)
                self.process_writer_completions()
                if self.sample_index != before:
                    next_sample_at = advance_sample_deadline(
                        next_sample_at,
                        sample_started_at,
                        time.monotonic(),
                        interval,
                    )
                else:
                    time.sleep(self.args.poll_interval)
            except SourceNotReady as exc:
                if not self.report_source_wait("waiting_source", str(exc)):
                    time.sleep(self.args.poll_interval)
            except WindowPipelineFailure as exc:
                logging.exception("window pipeline failed")
                if not self.terminate_window_pipeline(f"fatal: {type(exc).__name__}: {exc}"):
                    logging.error("window writer did not stop after fatal error")
                try:
                    self.write_metadata("error", f"{type(exc).__name__}: {exc}")
                except Exception:
                    logging.exception("failed to persist window-pipeline error status")
                exit_code = 1
                break
            except Exception as exc:
                logging.exception("bridge iteration failed")
                self.report_source_wait("error", f"{type(exc).__name__}: {exc}")
                time.sleep(max(self.args.poll_interval, 0.05))

            if self.args.once and self.published_windows >= 1:
                break
            if self.args.max_windows > 0 and self.published_windows >= self.args.max_windows:
                break

        if not self.pipeline_terminal and not self.terminate_window_pipeline("shutdown"):
            logging.error("window writer did not stop within %.1fs", self.args.window_writer_shutdown_timeout)
            exit_code = 1
        if exit_code == 0:
            self.maybe_write_status("stopped", "")
        return exit_code


def parse_cadence(value: str) -> tuple[int, int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cadence must contain integers") from exc
    if len(values) == 1:
        values.append(values[0])
    if len(values) != 2 or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("cadence must be FIRST,LATER with positive values")
    return values[0], values[1]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument(
        "--control-dir",
        type=Path,
        help="Shared control directory; defaults to project .runtime/alaya_live_control.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Upload directory; defaults to CONTROL_DIR/upload.",
    )
    parser.add_argument("--live-control", type=Path)
    parser.add_argument("--input-state", type=Path)
    parser.add_argument("--worker-status", type=Path)
    parser.add_argument("--worker-status-max-age", type=float, default=30.0)
    parser.add_argument("--max-worker-failed-recoveries", type=int, default=2)
    parser.add_argument("--max-worker-recoveries-per-reset", type=int, default=3)
    parser.add_argument("--max-worker-lag-windows", type=int, default=2)
    parser.add_argument("--worker-lag-observations", type=int, default=3)
    parser.add_argument("--worker-lag-resync-cooldown", type=float, default=15.0)
    parser.add_argument(
        "--manual-input-resync",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compatibility mode: start a fresh source epoch after coalesced "
            "manual drive-input changes. Disabled by default because normal "
            "drive input is temporally continuous."
        ),
    )
    parser.add_argument("--manual-input-coalesce", type=float, default=0.05)
    parser.add_argument("--manual-input-resync-cooldown", type=float, default=0.75)
    parser.add_argument(
        "--manual-input-max-age",
        type=float,
        default=2.0,
        help="Treat older manual input as auto control, matching the STK failsafe.",
    )
    parser.add_argument("--expected-width", type=int, default=1280)
    parser.add_argument("--expected-height", type=int, default=720)
    parser.add_argument("--target-width", type=int, default=832)
    parser.add_argument("--target-height", type=int, default=448)
    parser.add_argument("--target-fps", type=float, default=24.0)
    parser.add_argument(
        "--window-transport",
        choices=WINDOW_TRANSPORTS,
        default=DEFAULT_WINDOW_TRANSPORT,
        help=("Window payload: embedded NPZ, immutable mmap NPY, or preallocated double shared-memory slots"),
    )
    parser.add_argument(
        "--window-writer-queue-size",
        type=int,
        default=2,
        help="Maximum pending immutable-window jobs; one additional job may be active.",
    )
    parser.add_argument("--window-writer-shutdown-timeout", type=float, default=5.0)
    parser.add_argument(
        "--window-writer-test-delay",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--latent-window",
        type=int,
        choices=SUPPORTED_LATENT_WINDOWS,
        default=None,
        help="Derive production pixel cadence: 4 uses 13/16, 2 uses 5/8",
    )
    parser.add_argument(
        "--cadence",
        type=parse_cadence,
        default=None,
        help="Custom FIRST,LATER cadence; cannot conflict with --latent-window",
    )
    parser.add_argument(
        "--fixed-window-frames",
        type=int,
        default=0,
        help="Use rolling fixed-size windows instead of non-overlapping cadence chunks.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=0,
        help="New samples between rolling publications; defaults to fixed window size.",
    )
    parser.add_argument("--ring-capacity", type=int, default=0)
    parser.add_argument("--retain-windows", type=int, default=4)
    parser.add_argument("--poll-interval", type=float, default=0.005)
    parser.add_argument("--status-interval", type=float, default=1.0)
    parser.add_argument("--identity-settle", type=float, default=0.15)
    parser.add_argument("--stale-after", type=float, default=3.0)
    parser.add_argument("--allow-missing-metadata", action="store_true")
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def configure_temporal_cadence(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    if args.latent_window is None:
        if args.cadence is None:
            args.latent_window = DEFAULT_LATENT_WINDOW
            args.cadence = cadence_for_latent_window(DEFAULT_LATENT_WINDOW)
        else:
            # Custom cadence is retained for non-model bridge consumers and tests.
            args.latent_window = 0
        return

    expected = cadence_for_latent_window(args.latent_window)
    if args.cadence is not None and args.cadence != expected:
        parser.error(f"--latent-window {args.latent_window} requires --cadence {expected[0]},{expected[1]}")
    args.cadence = expected


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    positive = {
        "target width": args.target_width,
        "target height": args.target_height,
        "retain windows": args.retain_windows,
        "poll interval": args.poll_interval,
        "status interval": args.status_interval,
    }
    for label, value in positive.items():
        if value <= 0:
            parser.error(f"{label} must be positive")
    if args.fixed_window_frames < 0 or args.window_stride < 0 or args.ring_capacity < 0:
        parser.error("window and ring sizes cannot be negative")
    if args.worker_status_max_age <= 0:
        parser.error("--worker-status-max-age must be positive")
    if args.max_worker_failed_recoveries < 0:
        parser.error("--max-worker-failed-recoveries cannot be negative")
    if args.max_worker_recoveries_per_reset <= 0:
        parser.error("--max-worker-recoveries-per-reset must be positive")
    if args.max_worker_lag_windows < 0:
        parser.error("--max-worker-lag-windows cannot be negative")
    if args.worker_lag_observations <= 0:
        parser.error("--worker-lag-observations must be positive")
    if args.worker_lag_resync_cooldown < 0:
        parser.error("--worker-lag-resync-cooldown cannot be negative")
    if args.manual_input_coalesce < 0:
        parser.error("--manual-input-coalesce cannot be negative")
    if args.manual_input_resync_cooldown < 0:
        parser.error("--manual-input-resync-cooldown cannot be negative")
    if args.manual_input_max_age < 0:
        parser.error("--manual-input-max-age cannot be negative")
    if args.window_writer_queue_size <= 0:
        parser.error("--window-writer-queue-size must be positive")
    if args.window_writer_shutdown_timeout <= 0:
        parser.error("--window-writer-shutdown-timeout must be positive")
    if args.window_writer_test_delay < 0:
        parser.error("--window-writer-test-delay cannot be negative")
    if args.fixed_window_frames == 0 and args.window_stride:
        parser.error("--window-stride requires --fixed-window-frames")
    if args.fixed_window_frames and args.window_stride > args.fixed_window_frames:
        parser.error("--window-stride cannot exceed --fixed-window-frames")
    if args.window_transport == SHM_SLOT_TRANSPORT and args.fixed_window_frames:
        parser.error("shared-memory slots currently require non-overlapping chunk cadence")


def write_fake_capture(
    capture_dir: Path,
    seq: int,
    width: int,
    height: int,
    *,
    input_event_id: str | None = None,
    input_event_mode: str = "",
    control: Mapping[str, Any] | None = None,
) -> None:
    capture_dir.mkdir(parents=True, exist_ok=True)
    config_path = capture_dir / "run_config.env"
    if not config_path.exists():
        config_path.write_text(
            "\n".join(
                [
                    "TRACK=self_check",
                    "SOURCE_SESSION_ID=self-check-session",
                    f"SCREENSIZE={width}x{height}",
                    f"OUT_DIR={capture_dir}",
                    "",
                ]
            ),
            encoding="ascii",
        )
    metadata_path = capture_dir / "metadata.txt"
    if not metadata_path.exists():
        metadata_path.write_text(
            f"source=SuperTuxKart\nwidth={width}\nheight={height}\n",
            encoding="ascii",
        )
    y = np.arange(height, dtype=np.uint8)[:, None]
    x = np.arange(width, dtype=np.uint8)[None, :]
    albedo = np.zeros((height, width, 4), dtype=np.uint8)
    albedo[:, :, 0] = y
    albedo[:, :, 1] = x
    albedo[:, :, 2] = 64
    albedo[:, :, 3] = 255
    packed = np.zeros((height, width, 4), dtype=np.uint8)
    packed[:, :, 0] = 128
    packed[:, :, 1] = 128
    packed[:, :, 2] = 77
    packed[:, :, 3] = 33
    depth = np.linspace(0.0, 1.0, width * height, dtype="<f4").reshape(height, width)
    (capture_dir / RAW_FILES["albedo"]).write_bytes(albedo.tobytes())
    (capture_dir / RAW_FILES["normal_material"]).write_bytes(packed.tobytes())
    (capture_dir / RAW_FILES["depth"]).write_bytes(depth.tobytes())
    state = {
        "seq": seq,
        "source_frame": seq,
        "width": width,
        "height": height,
        "control": dict(control)
        if control is not None
        else {
            "kart_id": 7,
            "steer": 0.25,
            "accel": 1.0,
            "brake": False,
            "speed": 12.0,
            "position": 3,
        },
    }
    if input_event_id is not None:
        state["input_event_id"] = clean_input_event_id(input_event_id)
    if input_event_mode in {"auto", "manual"}:
        state["input_event_mode"] = input_event_mode
    # The marker is deliberately written last, matching the STK commit protocol.
    (capture_dir / "latest.json").write_text(json.dumps(state), encoding="ascii")


def run_self_check(window_transport: str) -> int:
    parser = make_parser()
    parser.set_defaults(window_transport=window_transport)
    assert parser.parse_args([]).manual_input_resync is False
    expected_modes = {2: (5, 8), 4: (13, 16)}
    for latent_window, expected_cadence in expected_modes.items():
        mode_args = parser.parse_args(["--latent-window", str(latent_window)])
        configure_temporal_cadence(mode_args, parser)
        validate_args(mode_args, parser)
        assert mode_args.cadence == expected_cadence

    fake_clock = [20.0]
    sleep_delays: list[float] = []

    def fake_now() -> float:
        return fake_clock[0]

    def fake_sleep(delay: float) -> None:
        sleep_delays.append(delay)
        fake_clock[0] += delay

    sample_started_at = wait_for_sample_deadline(
        20.05,
        now_fn=fake_now,
        sleep_fn=fake_sleep,
    )
    assert math.isclose(sample_started_at, 20.05)
    assert len(sleep_delays) == 1 and math.isclose(sleep_delays[0], 0.05)
    sample_started_at = wait_for_sample_deadline(
        20.04,
        now_fn=fake_now,
        sleep_fn=fake_sleep,
    )
    assert math.isclose(sample_started_at, 20.05)
    assert len(sleep_delays) == 1

    interval = 0.05
    deadline = advance_sample_deadline(0.0, 10.0, 10.01, interval)
    assert math.isclose(deadline, 10.05)
    deadline = advance_sample_deadline(deadline, 10.05, 10.18, interval)
    assert math.isclose(deadline, 10.10)  # Retain debt after normal serialization.
    deadline = advance_sample_deadline(deadline, 10.18, 11.0, interval)
    assert math.isclose(deadline, 11.05)  # Re-anchor after a long stall.
    simulated_now = 0.0
    simulated_deadline = 0.0
    for sample_number in range(160):
        if simulated_deadline > simulated_now:
            simulated_now = simulated_deadline
        simulated_start = simulated_now
        simulated_now += 0.13 if (sample_number + 1) % 16 == 0 else 0.02
        simulated_deadline = advance_sample_deadline(
            simulated_deadline,
            simulated_start,
            simulated_now,
            interval,
        )
    assert 160 / simulated_now > 19.5

    with tempfile.TemporaryDirectory(prefix="stk_bridge_check_") as temporary:
        root = Path(temporary)
        replace_check_path = root / "replace-check.json"
        replace_check_temp = prepare_json_temp(replace_check_path, {"authority": "committed"})
        original_fsync = os.fsync

        def fail_directory_fsync(_descriptor: int) -> None:
            raise OSError("injected directory fsync failure")

        os.fsync = fail_directory_fsync
        replace_failure: AuthorityManifestReplaceError | None = None
        try:
            replace_prepared_json(replace_check_temp, replace_check_path)
        except AuthorityManifestReplaceError as exc:
            replace_failure = exc
        finally:
            os.fsync = original_fsync
        assert replace_failure is not None
        assert replace_failure.committed is True
        assert replace_failure.phase == "post-replace-fsync"
        assert read_json_file(replace_check_path) == {"authority": "committed"}

        missing_replace_temp = root / ".missing-authority.tmp"
        pre_replace_failure: AuthorityManifestReplaceError | None = None
        try:
            replace_prepared_json(missing_replace_temp, replace_check_path)
        except AuthorityManifestReplaceError as exc:
            pre_replace_failure = exc
        assert pre_replace_failure is not None
        assert pre_replace_failure.committed is False
        assert pre_replace_failure.phase == "pre-replace"

        capture_dir = root / "capture"
        out_dir = root / "out"
        live_control_path = root / "control" / "live_control.json"
        worker_status_path = root / "control" / "worker_status.json"
        width, height = 64, 36
        write_fake_capture(capture_dir, seq=10, width=width, height=height)
        identity = read_source_identity(capture_dir, 0.0, True)
        capture = read_committed_capture(capture_dir, identity, width, height)
        decoded = decode_gbuffers(capture.raw, width, height, 32, 16)
        assert decoded.shape == (5, 16, 32, 3)
        assert decoded.dtype == np.uint8
        assert np.all(decoded[2, :, :, 0] == 33)
        assert np.all(decoded[2, :, :, 0] == decoded[2, :, :, 1])
        assert np.all(decoded[4, :, :, 0] == 77)
        assert decoded[0, 0, :, 0].mean() > decoded[0, -1, :, 0].mean()

        active_control = {
            "schema_version": SCHEMA_VERSION,
            "reset_epoch": 1,
            "reset_reason": "self_check_prompt",
            "render_request_id": 1,
            "request_reason": "prompt",
            "prompt_revision": 1,
            "prompt": "snowy cinematic racing",
            "track_id": "self_check",
            "source_session_id": "self-check-session",
            "reset_source_seq": 10,
            "reset_source_frame": 10,
            "requested_at": time.time(),
        }
        atomic_write_json(live_control_path, active_control)

        args = parser.parse_args(
            [
                "--capture-dir",
                str(capture_dir),
                "--out-dir",
                str(out_dir),
                "--live-control",
                str(live_control_path),
                "--worker-status",
                str(worker_status_path),
                "--expected-width",
                str(width),
                "--expected-height",
                str(height),
                "--target-width",
                "32",
                "--target-height",
                "16",
                "--cadence",
                "2,3",
                "--once",
                "--stale-after",
                "0",
            ]
        )
        configure_temporal_cadence(args, parser)
        validate_args(args, parser)
        bridge = Bridge(args)
        bridge.adopt_source(identity)
        bridge.accept_capture(capture, read_prompt_state(live_control_path))
        assert bridge.sample_index == 0  # The reset boundary frame is excluded.
        for seq in (11, 12):
            write_fake_capture(capture_dir, seq=seq, width=width, height=height)
            capture = read_committed_capture(capture_dir, identity, width, height)
            bridge.accept_capture(capture, read_prompt_state(live_control_path))
        bridge.wait_for_writer_idle()

        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["schema_version"] == SCHEMA_VERSION
        assert manifest["ready"] is True
        assert manifest["reset_epoch"] == 1
        assert manifest["source_reset_epoch"] == 1
        assert manifest["source_session_id"] == "self-check-session"
        assert manifest["window"]["frame_count"] == 2
        assert manifest["window_transport"] == window_transport
        assert manifest["window"]["window_transport"] == window_transport
        assert manifest["window_authority"]["metadata_file"] == manifest["immutable_file"]
        assert manifest["window_authority"]["aliases_authoritative"] is False
        assert manifest["aliases"]["role"] == "compatibility_only"
        assert manifest["aliases"]["authoritative"] is False
        assert manifest["window"]["aliases_authoritative"] is False
        immutable = out_dir / manifest["immutable_file"]
        assert immutable.is_file() and (out_dir / "window.npz").is_file()
        with np.load(immutable, allow_pickle=False) as archive:
            assert archive["window_transport"].item() == window_transport
            if window_transport == "npy":
                assert "gbuffers" not in archive
                sidecar = out_dir / archive["gbuffers_file"].item()
                assert sidecar.name == manifest["gbuffers_file"]
                assert sidecar.is_file() and (out_dir / "window.gbuffers.npy").is_file()
                gbuffers = np.load(sidecar, mmap_mode="r", allow_pickle=False)
                assert sidecar.stat().st_size == int(archive["gbuffers_file_size_bytes"].item())
            else:
                assert archive["gbuffers_file"].item() == ""
                assert not (out_dir / "window.gbuffers.npy").exists()
                gbuffers = archive["gbuffers"]
            assert gbuffers.shape == (2, 5, 16, 32, 3)
            assert gbuffers.dtype == np.uint8
            assert archive["gbuffers_shape"].tolist() == [2, 5, 16, 32, 3]
            assert archive["gbuffers_dtype"].item() == "uint8"
            assert int(archive["gbuffers_payload_nbytes"].item()) == gbuffers.nbytes
            assert archive["modalities"].tolist() == list(MODALITIES)
            assert archive["sample_indices"].tolist() == [0, 1]
            assert bool(archive["is_initial"].item())
            assert int(archive["reset_epoch"].item()) == 1
            assert int(archive["source_reset_epoch"].item()) == 1
            assert archive["source_session_id"].item() == "self-check-session"
            assert int(archive["latent_window"].item()) == 0
            assert int(archive["initial_pixel_frames"].item()) == 2
            assert int(archive["resumed_pixel_frames"].item()) == 3

        idle_control = {
            **active_control,
            "reset_epoch": 2,
            "reset_reason": "prompt_cleared",
            "render_request_id": 2,
            "prompt_revision": 2,
            "prompt": "",
            "reset_source_seq": 12,
            "reset_source_frame": 12,
        }
        atomic_write_json(live_control_path, idle_control)
        write_fake_capture(capture_dir, seq=13, width=width, height=height)
        capture = read_committed_capture(capture_dir, identity, width, height)
        bridge.accept_capture(capture, read_prompt_state(live_control_path))
        idle_manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert idle_manifest["ready"] is False
        assert idle_manifest["status"] == "idle"
        assert not (out_dir / "window.npz").exists()
        assert not (out_dir / "window.gbuffers.npy").exists()
        assert not bridge.scheduler.frames

        resumed_control = {
            **active_control,
            "reset_epoch": 3,
            "reset_reason": "prompt_resumed",
            "render_request_id": 3,
            "prompt_revision": 3,
            "prompt": "rainy night racing",
            "reset_source_seq": 13,
            "reset_source_frame": 13,
        }
        atomic_write_json(live_control_path, resumed_control)
        bridge.accept_capture(capture, read_prompt_state(live_control_path))
        for seq in (14, 15):
            write_fake_capture(capture_dir, seq=seq, width=width, height=height)
            capture = read_committed_capture(capture_dir, identity, width, height)
            bridge.accept_capture(capture, read_prompt_state(live_control_path))
        bridge.wait_for_writer_idle()
        assert bridge.current_window is not None
        assert bridge.current_window["reset_epoch"] == 3
        assert bridge.current_window["source_reset_epoch"] == 1

        worker_recovery = {
            "schema_version": SCHEMA_VERSION,
            "worker_instance_id": "self-check-worker",
            "status": "needs_reset",
            "error": "initial retained window is no longer available",
            "updated_at": time.time(),
            "reset_epoch": 3,
            "source_reset_epoch": 1,
            "prompt_revision": 3,
            "track_id": "self_check",
            "source_session_id": "self-check-session",
        }
        atomic_write_json(worker_status_path, worker_recovery)
        prompt = read_prompt_state(live_control_path)
        assert bridge.handle_worker_recovery(prompt, 15) is True
        assert bridge.handle_worker_recovery(prompt, 15) is False
        assert bridge.source_generation == 2
        bridge.accept_capture(capture, prompt)
        for seq in (16, 17):
            write_fake_capture(capture_dir, seq=seq, width=width, height=height)
            capture = read_committed_capture(capture_dir, identity, width, height)
            bridge.accept_capture(capture, prompt)
        bridge.wait_for_writer_idle()
        assert bridge.current_window is not None
        assert bridge.current_window["is_initial"] is True
        assert bridge.current_window["source_reset_epoch"] == 2
        recovery_manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert recovery_manifest["last_handled_worker_recovery"]["status"] == "needs_reset"

        for failure_index, (boundary, next_frames) in enumerate(((17, (18, 19)), (19, (20, 21))), start=1):
            failed_status = {
                **worker_recovery,
                "status": "failed",
                "error": f"terminal inference failure {failure_index}",
                "updated_at": time.time() + failure_index * 0.001,
                "source_reset_epoch": bridge.source_generation,
            }
            atomic_write_json(worker_status_path, failed_status)
            assert bridge.handle_worker_recovery(prompt, boundary) is True
            bridge.accept_capture(capture, prompt)
            for seq in next_frames:
                write_fake_capture(capture_dir, seq=seq, width=width, height=height)
                capture = read_committed_capture(capture_dir, identity, width, height)
                bridge.accept_capture(capture, prompt)
            bridge.wait_for_writer_idle()
            assert bridge.current_window is not None
            assert bridge.current_window["is_initial"] is True

        capped_status = {
            **worker_recovery,
            "status": "failed",
            "error": "terminal inference failure capped",
            "updated_at": time.time() + 0.01,
            "source_reset_epoch": bridge.source_generation,
        }
        atomic_write_json(worker_status_path, capped_status)
        assert bridge.handle_worker_recovery(prompt, 21) is False
        assert bridge.source_generation == 4
        assert bridge.recovery_is_capped(prompt)
        assert bridge.current_window is None
        capped_manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert capped_manifest["worker_recovery_count"] == 3
        assert capped_manifest["worker_recovery_capped"] is True

        reset_after_cap = {
            **resumed_control,
            "reset_epoch": 4,
            "reset_reason": "new_prompt_after_failure",
            "render_request_id": 4,
            "prompt_revision": 4,
            "prompt": "sunset mountain racing",
            "reset_source_seq": 21,
            "reset_source_frame": 21,
        }
        atomic_write_json(live_control_path, reset_after_cap)
        prompt = read_prompt_state(live_control_path)
        bridge.accept_capture(capture, prompt)
        for seq in (22, 23):
            write_fake_capture(capture_dir, seq=seq, width=width, height=height)
            capture = read_committed_capture(capture_dir, identity, width, height)
            bridge.accept_capture(capture, prompt)
        bridge.wait_for_writer_idle()
        assert bridge.current_window is not None
        assert bridge.current_window["reset_epoch"] == 4
        assert not bridge.recovery_is_capped(prompt)

        # An unsignaled game restart keeps the active prompt but starts new history.
        write_fake_capture(capture_dir, seq=5, width=width, height=height)
        capture = read_committed_capture(capture_dir, identity, width, height)
        bridge.accept_capture(capture, read_prompt_state(live_control_path))
        assert bridge.source_generation == 5
        assert bridge.sample_index == 0
        for seq in (6, 7):
            write_fake_capture(capture_dir, seq=seq, width=width, height=height)
            capture = read_committed_capture(capture_dir, identity, width, height)
            bridge.accept_capture(capture, read_prompt_state(live_control_path))
        bridge.wait_for_writer_idle()
        assert bridge.current_window is not None
        assert bridge.current_window["is_initial"] is True
        assert bridge.current_window["reset_epoch"] == 4
        assert bridge.current_window["source_reset_epoch"] == 5

        rebound_identity = dataclasses.replace(
            identity,
            source_session_id="self-check-session-restarted",
            token="self-check-restarted-token",
        )
        bridge.active_prompt = prompt
        bridge.reset_source(
            "source_session_changed",
            rebound_identity,
            boundary_seq=None,
            increment=True,
        )
        assert bridge.control_matches_source(prompt)
        bridge.observe_control(prompt, current_seq=8)
        assert bridge.accept_after_seq == 8

        # A control update for a newly bound session cannot reuse the previous
        # session's sequence boundary, even if the producer copied a large value.
        session_control = {
            **reset_after_cap,
            "reset_epoch": 5,
            "render_request_id": 5,
            "prompt_revision": 5,
            "source_session_id": rebound_identity.source_session_id,
            "reset_source_seq": 1_412_329,
            "reset_source_frame": 1_412_329,
        }
        atomic_write_json(live_control_path, session_control)
        session_prompt = read_prompt_state(live_control_path)
        assert bridge.observe_control(session_prompt, current_seq=8)
        assert bridge.accept_after_seq == 8
        assert bridge.force_current_control_boundary is False

        # The new control may arrive before the capture identity changes. Once the
        # new source is adopted, its current frame still becomes the local fence.
        track_control = {
            **session_control,
            "reset_epoch": 6,
            "render_request_id": 6,
            "prompt_revision": 6,
            "track_id": "zen_garden",
            "source_session_id": "zen-garden-session",
            "reset_source_seq": 1_412_330,
            "reset_source_frame": 1_412_330,
        }
        atomic_write_json(live_control_path, track_control)
        track_prompt = read_prompt_state(live_control_path)
        assert bridge.observe_control(track_prompt, current_seq=9)
        assert bridge.active_prompt == track_prompt
        assert bridge.accept_after_seq == 1_412_330
        assert not bridge.control_matches_source(track_prompt)

        (capture_dir / "run_config.env").write_text(
            "\n".join(
                [
                    "TRACK=zen_garden",
                    "SOURCE_SESSION_ID=zen-garden-session",
                    f"SCREENSIZE={width}x{height}",
                    f"OUT_DIR={capture_dir}",
                    "",
                ]
            ),
            encoding="ascii",
        )
        write_fake_capture(capture_dir, seq=0, width=width, height=height)
        track_identity = read_source_identity(capture_dir, 0.0, True)
        bridge.reset_source(
            "track_changed",
            track_identity,
            boundary_seq=None,
            increment=True,
        )
        assert bridge.force_current_control_boundary is True
        capture = read_committed_capture(capture_dir, track_identity, width, height)
        bridge.accept_capture(capture, track_prompt)
        assert bridge.accept_after_seq == 0
        assert bridge.force_current_control_boundary is False
        assert bridge.sample_index == 0

        write_fake_capture(capture_dir, seq=1, width=width, height=height)
        capture = read_committed_capture(capture_dir, track_identity, width, height)
        bridge.accept_capture(capture, track_prompt)
        assert bridge.sample_index == 1

        generation_before_rollback = bridge.source_generation
        write_fake_capture(capture_dir, seq=0, width=width, height=height)
        capture = read_committed_capture(capture_dir, track_identity, width, height)
        bridge.accept_capture(capture, track_prompt)
        assert bridge.source_generation == generation_before_rollback + 1
        assert bridge.source_reset_reason == "seq_rollback"
        assert bridge.accept_after_seq == 0
        assert bridge.sample_index == 0

        write_fake_capture(capture_dir, seq=1, width=width, height=height)
        capture = read_committed_capture(capture_dir, track_identity, width, height)
        bridge.accept_capture(capture, track_prompt)
        assert bridge.sample_index == 1

        # A sustained worker backlog starts a fresh fenced source generation.
        lag_capture_dir = root / "lag-capture"
        lag_out_dir = root / "lag-out"
        lag_control_path = root / "lag-control" / "live_control.json"
        lag_worker_status_path = root / "lag-control" / "worker_status.json"
        lag_control = {
            **active_control,
            "reset_source_seq": 100,
            "reset_source_frame": 100,
        }
        atomic_write_json(lag_control_path, lag_control)
        write_fake_capture(lag_capture_dir, seq=100, width=width, height=height)
        lag_identity = read_source_identity(lag_capture_dir, 0.0, True)
        lag_args = parser.parse_args(
            [
                "--capture-dir",
                str(lag_capture_dir),
                "--out-dir",
                str(lag_out_dir),
                "--live-control",
                str(lag_control_path),
                "--worker-status",
                str(lag_worker_status_path),
                "--expected-width",
                str(width),
                "--expected-height",
                str(height),
                "--target-width",
                "32",
                "--target-height",
                "16",
                "--cadence",
                "2,3",
                "--max-worker-lag-windows",
                "1",
                "--worker-lag-observations",
                "3",
                "--worker-lag-resync-cooldown",
                "15",
                "--stale-after",
                "0",
            ]
        )
        configure_temporal_cadence(lag_args, parser)
        validate_args(lag_args, parser)
        lag_bridge = Bridge(lag_args)
        lag_bridge.adopt_source(lag_identity)
        lag_prompt = read_prompt_state(lag_control_path)
        for seq in range(100, 106):
            write_fake_capture(lag_capture_dir, seq=seq, width=width, height=height)
            lag_capture = read_committed_capture(lag_capture_dir, lag_identity, width, height)
            lag_bridge.accept_capture(lag_capture, lag_prompt)
        lag_bridge.wait_for_writer_idle()
        assert lag_bridge.window_id == 2
        assert lag_bridge.source_generation == 1

        lag_status = {
            "schema_version": SCHEMA_VERSION,
            "worker_instance_id": "lag-self-check-worker",
            "status": "warming",
            "error": "",
            "ready": True,
            "updated_at": time.time(),
            "reset_epoch": 1,
            "source_reset_epoch": 1,
            "prompt_revision": 1,
            "track_id": "self_check",
            "source_session_id": "self-check-session",
            "accepted_request_id": 1,
            "accepted_window_id": 0,
            "completed_window_id": 0,
        }
        atomic_write_json(lag_worker_status_path, lag_status)
        assert lag_bridge.handle_worker_backlog(lag_prompt, 105) is False
        assert lag_bridge.worker_lag_observations == 0

        lag_status["status"] = "rendering"
        lag_status["ready"] = False
        atomic_write_json(lag_worker_status_path, lag_status)
        assert lag_bridge.handle_worker_backlog(lag_prompt, 105) is False
        lag_status["ready"] = True
        lag_status["source_session_id"] = "wrong-session"
        atomic_write_json(lag_worker_status_path, lag_status)
        assert lag_bridge.handle_worker_backlog(lag_prompt, 105) is False
        lag_status["source_session_id"] = "self-check-session"
        for observation_index in range(3):
            lag_status["updated_at"] = time.time() + observation_index * 0.001
            atomic_write_json(lag_worker_status_path, lag_status)
            triggered = lag_bridge.handle_worker_backlog(lag_prompt, 105)
            if observation_index == 0:
                assert lag_bridge.handle_worker_backlog(lag_prompt, 105) is False
                assert lag_bridge.worker_lag_observations == 1
            assert triggered is (observation_index == 2)
        assert lag_bridge.source_generation == 2
        assert lag_bridge.current_window is None
        assert lag_bridge.accept_after_seq == 105
        assert lag_bridge.source_reset_reason == "worker_backlog"
        assert lag_bridge.last_worker_backlog_resync["lag_windows"] == 2

        lag_bridge.accept_capture(lag_capture, lag_prompt)
        for seq in range(106, 114):
            write_fake_capture(lag_capture_dir, seq=seq, width=width, height=height)
            lag_capture = read_committed_capture(lag_capture_dir, lag_identity, width, height)
            lag_bridge.accept_capture(lag_capture, lag_prompt)
        lag_bridge.wait_for_writer_idle()
        assert lag_bridge.window_id == 5
        assert lag_bridge.current_window is not None
        assert lag_bridge.current_window["source_reset_epoch"] == 2

        lag_status.update(
            {
                "source_reset_epoch": 2,
                "accepted_window_id": 3,
                "completed_window_id": 3,
            }
        )
        for observation_index in range(3):
            lag_status["updated_at"] = time.time() + observation_index * 0.001
            atomic_write_json(lag_worker_status_path, lag_status)
            assert lag_bridge.handle_worker_backlog(lag_prompt, 113) is False
        assert lag_bridge.source_generation == 2
        lag_manifest = json.loads((lag_out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert lag_manifest["last_worker_backlog_resync"]["lag_windows"] == 2
        assert lag_bridge.worker_lag_summary()["cooldown_remaining"] > 0.0

        # Normal drive input is continuous and does not reset model history by default.
        no_resync_capture_dir = root / "no-resync-capture"
        no_resync_out_dir = root / "no-resync-out"
        no_resync_control_path = root / "no-resync-control" / "live_control.json"
        no_resync_worker_status_path = root / "no-resync-control" / "worker_status.json"
        no_resync_input_path = no_resync_capture_dir / "input.json"
        no_resync_control = {
            **active_control,
            "reset_source_seq": 299,
            "reset_source_frame": 299,
        }
        atomic_write_json(no_resync_control_path, no_resync_control)
        atomic_write_json(
            no_resync_input_path,
            {
                "mode": "auto",
                "keys": [],
                "server_time": time.time(),
                "source": "self-check",
            },
        )
        write_fake_capture(no_resync_capture_dir, seq=300, width=width, height=height)
        no_resync_identity = read_source_identity(no_resync_capture_dir, 0.0, True)
        no_resync_args = parser.parse_args(
            [
                "--capture-dir",
                str(no_resync_capture_dir),
                "--out-dir",
                str(no_resync_out_dir),
                "--live-control",
                str(no_resync_control_path),
                "--input-state",
                str(no_resync_input_path),
                "--worker-status",
                str(no_resync_worker_status_path),
                "--expected-width",
                str(width),
                "--expected-height",
                str(height),
                "--target-width",
                "32",
                "--target-height",
                "16",
                "--cadence",
                "2,3",
                "--stale-after",
                "0",
            ]
        )
        configure_temporal_cadence(no_resync_args, parser)
        validate_args(no_resync_args, parser)
        assert no_resync_args.manual_input_resync is False
        no_resync_bridge = Bridge(no_resync_args)
        no_resync_bridge.adopt_source(no_resync_identity)
        no_resync_prompt = read_prompt_state(no_resync_control_path)

        def accept_no_resync_capture(seq: int) -> None:
            write_fake_capture(no_resync_capture_dir, seq=seq, width=width, height=height)
            committed = read_committed_capture(no_resync_capture_dir, no_resync_identity, width, height)
            no_resync_bridge.accept_capture(committed, no_resync_prompt)

        accept_no_resync_capture(300)
        for seq, (mode, keys) in enumerate(
            (
                ("manual", ["ArrowLeft"]),
                ("manual", ["ArrowDown"]),
                ("manual", ["ArrowUp"]),
                ("auto", []),
            ),
            start=301,
        ):
            atomic_write_json(
                no_resync_input_path,
                {
                    "mode": mode,
                    "keys": keys,
                    "server_time": time.time(),
                    "source": "self-check",
                },
            )
            accept_no_resync_capture(seq)
        no_resync_bridge.wait_for_writer_idle()
        assert no_resync_bridge.source_generation == 1
        assert no_resync_bridge.manual_input_resync_count == 0

        # Explicit compatibility mode retains the previous resync behavior.
        input_capture_dir = root / "input-capture"
        input_out_dir = root / "input-out"
        input_control_path = root / "input-control" / "live_control.json"
        input_worker_status_path = root / "input-control" / "worker_status.json"
        input_state_path = input_capture_dir / "input.json"
        input_control = {
            **active_control,
            "reset_source_seq": 199,
            "reset_source_frame": 199,
        }
        atomic_write_json(input_control_path, input_control)
        atomic_write_json(
            input_state_path,
            {
                "mode": "auto",
                "keys": [],
                "server_time": time.time(),
                "source": "self-check",
            },
        )
        write_fake_capture(input_capture_dir, seq=200, width=width, height=height)
        input_identity = read_source_identity(input_capture_dir, 0.0, True)
        input_args = parser.parse_args(
            [
                "--capture-dir",
                str(input_capture_dir),
                "--out-dir",
                str(input_out_dir),
                "--live-control",
                str(input_control_path),
                "--input-state",
                str(input_state_path),
                "--worker-status",
                str(input_worker_status_path),
                "--expected-width",
                str(width),
                "--expected-height",
                str(height),
                "--target-width",
                "32",
                "--target-height",
                "16",
                "--cadence",
                "2,3",
                "--manual-input-resync",
                "--manual-input-coalesce",
                "0",
                "--manual-input-resync-cooldown",
                "0",
                "--stale-after",
                "0",
            ]
        )
        configure_temporal_cadence(input_args, parser)
        validate_args(input_args, parser)
        assert input_args.max_worker_lag_windows == 2
        input_bridge = Bridge(input_args)
        input_bridge.adopt_source(input_identity)
        input_prompt = read_prompt_state(input_control_path)

        def accept_input_capture(seq: int) -> RawCapture:
            write_fake_capture(input_capture_dir, seq=seq, width=width, height=height)
            committed = read_committed_capture(input_capture_dir, input_identity, width, height)
            input_bridge.accept_capture(committed, input_prompt)
            return committed

        current_input_capture = accept_input_capture(200)
        atomic_write_json(
            input_state_path,
            {
                "mode": "auto",
                "keys": [],
                "server_time": time.time() + 1.0,
                "source": "heartbeat",
            },
        )
        current_input_capture = accept_input_capture(201)
        assert input_bridge.source_generation == 1
        assert input_bridge.manual_input_resync_count == 0
        assert input_bridge.manual_input_duplicate_update_count >= 1

        atomic_write_json(
            input_state_path,
            {
                "mode": "manual",
                "keys": ["ArrowLeft"],
                "server_time": time.time(),
                "source": "keydown",
            },
        )
        current_input_capture = accept_input_capture(202)
        assert input_bridge.source_generation == 2
        assert input_bridge.manual_input_resync_count == 1
        assert input_bridge.last_manual_input_resync["signature"] == {
            "mode": "manual",
            "steer": -1,
            "accel": False,
            "brake": False,
        }
        assert input_bridge.current_window is None
        current_input_capture = accept_input_capture(203)
        input_bridge.wait_for_writer_idle()
        assert input_bridge.current_window is not None
        assert input_bridge.current_window["source_reset_epoch"] == 2

        # ArrowLeft and KeyA are the same effective state; Space is non-directional.
        for keys in (["KeyA"], ["KeyA", "Space"]):
            atomic_write_json(
                input_state_path,
                {
                    "mode": "manual",
                    "keys": keys,
                    "server_time": time.time(),
                    "source": "heartbeat",
                },
            )
            current_input_capture = accept_input_capture(integer(current_input_capture.state.get("seq"), -1) + 1)
        assert input_bridge.source_generation == 2
        assert input_bridge.manual_input_resync_count == 1

        # Multiple changes before a newly committed frame collapse to the latest state.
        stable_seq = integer(current_input_capture.state.get("seq"), -1)
        atomic_write_json(
            input_state_path,
            {
                "mode": "manual",
                "keys": ["ArrowLeft", "ArrowUp"],
                "server_time": time.time(),
                "source": "keydown",
            },
        )
        input_bridge.accept_capture(current_input_capture, input_prompt)
        atomic_write_json(
            input_state_path,
            {
                "mode": "manual",
                "keys": ["ArrowRight", "ArrowUp"],
                "server_time": time.time(),
                "source": "keydown",
            },
        )
        input_bridge.accept_capture(current_input_capture, input_prompt)
        assert input_bridge.manual_input_pending
        assert input_bridge.manual_input_coalesced_count >= 1
        current_input_capture = accept_input_capture(stable_seq + 1)
        assert input_bridge.source_generation == 3
        assert input_bridge.manual_input_resync_count == 2
        assert input_bridge.last_manual_input_resync["signature"] == {
            "mode": "manual",
            "steer": 1,
            "accel": True,
            "brake": False,
        }

        # A mismatched prompt/session cannot authorize a manual-input epoch reset.
        wrong_prompt = dataclasses.replace(
            input_prompt,
            source_session_id="wrong-session",
        )
        before_generation = input_bridge.source_generation
        atomic_write_json(
            input_state_path,
            {
                "mode": "manual",
                "keys": ["ArrowDown"],
                "server_time": time.time(),
                "source": "keydown",
            },
        )
        assert (
            input_bridge.observe_manual_input(
                wrong_prompt,
                current_seq=stable_seq + 2,
                captured_at_ns=time.time_ns(),
                is_new_capture=True,
            )
            is False
        )
        assert input_bridge.source_generation == before_generation

        # A recent manual reset also guards the backlog resync path from stacking.
        current_input_capture = accept_input_capture(stable_seq + 2)
        input_status = {
            "schema_version": SCHEMA_VERSION,
            "worker_instance_id": "input-self-check-worker",
            "status": "rendering",
            "error": "",
            "ready": True,
            "updated_at": time.time(),
            "reset_epoch": input_prompt.reset_epoch,
            "source_reset_epoch": input_bridge.source_generation,
            "prompt_revision": input_prompt.prompt_revision,
            "track_id": "self_check",
            "source_session_id": "self-check-session",
            "accepted_request_id": input_prompt.render_request_id,
            "accepted_window_id": 0,
            "completed_window_id": 0,
        }
        for observation_index in range(3):
            input_status["updated_at"] = time.time() + observation_index * 0.001
            atomic_write_json(input_worker_status_path, input_status)
            assert input_bridge.handle_worker_backlog(input_prompt, stable_seq + 2) is False
        assert input_bridge.source_generation == before_generation
        assert input_bridge.worker_lag_summary()["cooldown_remaining"] > 0.0

        stale_state = read_directional_input_state(
            input_state_path,
            max_age_seconds=2.0,
            now_ns=input_state_path.stat().st_mtime_ns + 3_000_000_000,
        )
        assert stale_state is not None
        assert stale_state.signature == ("auto", 0, False, False)

        def make_writer_test_bridge(
            label: str,
            queue_size: int,
            write_delay: float = 0.0,
            retain_windows: int = 4,
            max_worker_lag_windows: int = 2,
        ) -> tuple[Bridge, Path, Path, SourceIdentity, PromptState]:
            test_capture_dir = root / f"{label}-capture"
            test_out_dir = root / f"{label}-out"
            test_control_path = root / f"{label}-control" / "live_control.json"
            test_worker_status = root / f"{label}-control" / "worker_status.json"
            test_control = {
                **active_control,
                "reset_source_seq": 0,
                "reset_source_frame": 0,
            }
            atomic_write_json(test_control_path, test_control)
            write_fake_capture(test_capture_dir, seq=0, width=width, height=height)
            test_identity = read_source_identity(test_capture_dir, 0.0, True)
            test_args = parser.parse_args(
                [
                    "--capture-dir",
                    str(test_capture_dir),
                    "--out-dir",
                    str(test_out_dir),
                    "--live-control",
                    str(test_control_path),
                    "--worker-status",
                    str(test_worker_status),
                    "--expected-width",
                    str(width),
                    "--expected-height",
                    str(height),
                    "--target-width",
                    "32",
                    "--target-height",
                    "16",
                    "--cadence",
                    "1,1",
                    "--window-writer-queue-size",
                    str(queue_size),
                    "--window-writer-test-delay",
                    str(write_delay),
                    "--retain-windows",
                    str(retain_windows),
                    "--max-worker-lag-windows",
                    str(max_worker_lag_windows),
                    "--status-interval",
                    "1000",
                    "--stale-after",
                    "0",
                ]
            )
            configure_temporal_cadence(test_args, parser)
            validate_args(test_args, parser)
            test_bridge = Bridge(test_args)
            test_bridge.adopt_source(test_identity)
            initial_capture = read_committed_capture(test_capture_dir, test_identity, width, height)
            test_prompt = read_prompt_state(test_control_path)
            test_bridge.accept_capture(initial_capture, test_prompt)
            assert test_bridge.sample_index == 0
            return (
                test_bridge,
                test_capture_dir,
                test_control_path,
                test_identity,
                test_prompt,
            )

        def accept_writer_test_frame(
            test_bridge: Bridge,
            test_capture_dir: Path,
            test_identity: SourceIdentity,
            test_prompt: PromptState,
            seq: int,
        ) -> None:
            write_fake_capture(test_capture_dir, seq=seq, width=width, height=height)
            committed = read_committed_capture(test_capture_dir, test_identity, width, height)
            test_bridge.accept_capture(committed, test_prompt)

        from alayarenderer_flash.demo import worker as worker_module

        def strict_worker_reader(test_bridge: Bridge) -> Any:
            cadence = worker_module.TemporalCadence(
                latent_window=test_bridge.args.latent_window,
                initial_pixel_frames=test_bridge.scheduler.initial_frames,
                resumed_pixel_frames=test_bridge.scheduler.subsequent_frames,
            )
            return worker_module.InputReader(
                test_bridge.out_dir,
                test_bridge.live_control_path,
                test_bridge.args.target_height,
                test_bridge.args.target_width,
                cadence,
                test_bridge.args.window_transport,
            )

        random_normals = np.random.default_rng(7).integers(0, 256, size=(37, 53, 2), dtype=np.uint8)
        normal_reference = _decode_octahedral_normal_reference(random_normals)
        normal_fast = decode_octahedral_normal(random_normals)
        normal_out = np.empty_like(normal_reference)
        decode_octahedral_normal(random_normals, out=normal_out)
        assert np.array_equal(normal_fast, normal_reference)
        assert np.array_equal(normal_out, normal_reference)

        # A 200 ms writer delay must not appear in the 30 Hz sampling cadence.
        (
            slow_bridge,
            slow_capture_dir,
            _slow_control_path,
            slow_identity,
            slow_prompt,
        ) = make_writer_test_bridge("slow-writer", queue_size=32, write_delay=0.2)
        sampling_starts: list[float] = []
        sampling_durations: list[float] = []
        sample_deadline = time.monotonic()
        for seq in range(1, 13):
            delay = sample_deadline - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            sample_started = time.monotonic()
            sampling_starts.append(sample_started)
            accept_writer_test_frame(
                slow_bridge,
                slow_capture_dir,
                slow_identity,
                slow_prompt,
                seq,
            )
            sampling_durations.append(time.monotonic() - sample_started)
            sample_deadline += 1.0 / 30.0
        sampling_gaps = [right - left for left, right in zip(sampling_starts, sampling_starts[1:])]
        sampling_fps = (len(sampling_starts) - 1) / (sampling_starts[-1] - sampling_starts[0])
        assert max(sampling_durations) < 0.10
        assert max(sampling_gaps) < 0.10
        assert sampling_fps > 25.0
        slow_bridge.reset_source(
            "slow_writer_self_check_complete",
            slow_identity,
            boundary_seq=12,
            increment=True,
        )
        slow_bridge.wait_for_writer_idle()
        assert not list(slow_bridge.out_dir.glob("window_*.npz"))
        assert not list(slow_bridge.out_dir.glob("window_*.gbuffers.npy"))
        assert slow_bridge.terminate_window_pipeline("self_check_complete")

        # Writer output remains invisible to a direct worker until the main-thread
        # fence promotes the staged pair. Resetting while staged must discard it.
        (
            race_bridge,
            race_capture_dir,
            race_control_path,
            race_identity,
            race_prompt,
        ) = make_writer_test_bridge("reset-race", queue_size=2)
        pair_written = threading.Event()
        release_pair = threading.Event()
        original_race_save_npz = race_bridge.window_writer._save_npz_fn

        def held_pair_save_npz(
            path: Path,
            arrays: Mapping[str, np.ndarray],
            compress: bool,
        ) -> None:
            original_race_save_npz(path, arrays, compress)
            pair_written.set()
            if not release_pair.wait(2.0):
                raise TimeoutError("reset-race self-check did not release writer")

        race_bridge.window_writer._save_npz_fn = held_pair_save_npz
        accept_writer_test_frame(race_bridge, race_capture_dir, race_identity, race_prompt, 1)
        assert pair_written.wait(2.0)
        race_old_metadata = race_bridge.out_dir / "window_000000000001.npz"
        race_old_sidecar = gbuffer_sidecar_path(race_old_metadata)
        staged_metadata = list(race_bridge.staging_dir.glob("*.metadata.npz"))
        staged_sidecars = list(race_bridge.staging_dir.glob("*.gbuffers.npy"))
        assert len(staged_metadata) == 1
        assert not race_old_metadata.exists()
        assert not list(race_bridge.out_dir.glob("window_*.npz"))
        if window_transport == "npy":
            assert len(staged_sidecars) == 1
            assert not race_old_sidecar.exists()
        race_control = {
            **active_control,
            "reset_epoch": 2,
            "render_request_id": 2,
            "prompt_revision": 2,
            "reset_reason": "writer_reset_race",
            "prompt": "fresh epoch after writer race",
            "reset_source_seq": 1,
            "reset_source_frame": 1,
        }
        atomic_write_json(race_control_path, race_control)
        race_new_prompt = read_prompt_state(race_control_path)
        race_bridge.observe_control(race_new_prompt, current_seq=1)
        release_pair.set()
        race_bridge.wait_for_writer_idle()
        assert not race_old_metadata.exists()
        assert not race_old_sidecar.exists()
        assert not list(race_bridge.staging_dir.iterdir())
        assert not race_bridge.alias_path.exists()
        assert not race_bridge.gbuffer_alias_path.exists()
        race_bridge.window_writer._save_npz_fn = original_race_save_npz
        accept_writer_test_frame(race_bridge, race_capture_dir, race_identity, race_new_prompt, 2)
        race_bridge.wait_for_writer_idle()
        race_manifest = read_json_file(race_bridge.manifest_path)
        assert race_manifest["ready"] is True
        assert race_manifest["reset_epoch"] == 2
        assert race_manifest["window"]["first_seq"] == 2
        assert race_bridge.terminate_window_pipeline("self_check_complete")

        def marker_fence_reset_check(
            label: str,
            mutate: Callable[[dict[str, Any]], None],
            expected_reason: str,
        ) -> None:
            (
                marker_bridge,
                marker_capture_dir,
                _marker_control_path,
                marker_identity,
                marker_prompt,
            ) = make_writer_test_bridge(label, queue_size=2)
            accept_writer_test_frame(
                marker_bridge,
                marker_capture_dir,
                marker_identity,
                marker_prompt,
                1,
            )
            assert marker_bridge.window_writer.wait_idle(2.0)
            assert list(marker_bridge.staging_dir.iterdir())
            assert not list(marker_bridge.out_dir.glob("window_*.npz"))
            marker_path = marker_capture_dir / "latest.json"
            marker_state = read_json_file(marker_path)
            mutate(marker_state)
            atomic_write_json(marker_path, marker_state)
            generation_before = marker_bridge.source_generation
            marker_bridge.process_writer_completions()
            assert marker_bridge.source_generation == generation_before + 1
            assert marker_bridge.source_reset_reason == expected_reason
            assert not list(marker_bridge.staging_dir.iterdir())
            assert not list(marker_bridge.out_dir.glob("window_*.npz"))
            assert marker_bridge.terminate_window_pipeline("self_check_complete")

        marker_fence_reset_check(
            "promote-seq-rollback",
            lambda state: state.update({"seq": 0, "source_frame": 1}),
            "seq_rollback",
        )
        marker_fence_reset_check(
            "promote-source-frame-rollback",
            lambda state: state.update({"seq": 2, "source_frame": 0}),
            "source_frame_rollback",
        )
        marker_fence_reset_check(
            "promote-resolution-change",
            lambda state: state.update({"seq": 2, "source_frame": 2, "width": width + 1}),
            "resolution_changed",
        )

        def advance_latest_marker(capture_dir: Path, count: int = 1) -> None:
            marker_path = capture_dir / "latest.json"
            for _ in range(count):
                marker_state = read_json_file(marker_path)
                marker_state["seq"] = integer(marker_state.get("seq"), -1) + 1
                marker_state["source_frame"] = integer(marker_state.get("source_frame"), -1) + 1
                atomic_write_json(marker_path, marker_state)

        def read_marker_while_advancing(
            marker_reader: Callable[[Path, bool], CurrentMarkerState],
            capture_dir: Path,
            require_metadata: bool,
            advance_count: int = 1,
        ) -> CurrentMarkerState:
            source_reader = globals()["read_source_identity"]
            identity_reads = [0]

            def advancing_identity_reader(*reader_args: Any, **reader_kwargs: Any) -> SourceIdentity:
                identity = source_reader(*reader_args, **reader_kwargs)
                identity_reads[0] += 1
                if identity_reads[0] == 1:
                    advance_latest_marker(capture_dir, advance_count)
                return identity

            globals()["read_source_identity"] = advancing_identity_reader
            try:
                marker = marker_reader(capture_dir, require_metadata)
            finally:
                globals()["read_source_identity"] = source_reader
            assert identity_reads[0] == 2
            return marker

        def marker_advance_guard_check(
            label: str,
            guard_call: int,
            expect_manifest_temp: bool,
        ) -> None:
            (
                advance_bridge,
                advance_capture_dir,
                _advance_control_path,
                advance_identity,
                advance_prompt,
            ) = make_writer_test_bridge(label, queue_size=2)
            accept_writer_test_frame(
                advance_bridge,
                advance_capture_dir,
                advance_identity,
                advance_prompt,
                1,
            )
            assert advance_bridge.window_writer.wait_idle(2.0)
            original_marker_reader = advance_bridge.marker_reader
            marker_calls = [0]

            def guard_advancing_reader(path: Path, require_metadata: bool) -> CurrentMarkerState:
                marker_calls[0] += 1
                if marker_calls[0] != guard_call:
                    return original_marker_reader(path, require_metadata)
                live_path = advance_bridge.out_dir / "window_000000000001.npz"
                assert live_path.is_file()
                if window_transport == "npy":
                    assert gbuffer_sidecar_path(live_path).is_file()
                manifest_temps = list(advance_bridge.out_dir.glob(".manifest.json.*.authority.tmp"))
                assert bool(manifest_temps) is expect_manifest_temp
                return read_marker_while_advancing(
                    original_marker_reader,
                    path,
                    require_metadata,
                )

            advance_bridge.marker_reader = guard_advancing_reader
            generation_before = advance_bridge.source_generation
            advance_bridge.process_writer_completions()
            assert marker_calls[0] == 4
            assert advance_bridge.source_generation == generation_before
            assert advance_bridge.published_windows == 1
            assert not advance_bridge.deferred_results
            assert read_json_file(advance_bridge.manifest_path)["ready"] is True
            assert advance_bridge.terminate_window_pipeline("self_check_complete")

        marker_advance_guard_check(
            "post-promote-marker-advance",
            guard_call=3,
            expect_manifest_temp=False,
        )
        marker_advance_guard_check(
            "pre-manifest-marker-advance",
            guard_call=4,
            expect_manifest_temp=True,
        )

        # Every fence can observe several newer commits without treating normal
        # capture progress as a source failure.
        (
            continuous_bridge,
            continuous_capture_dir,
            _continuous_control_path,
            continuous_identity,
            continuous_prompt,
        ) = make_writer_test_bridge("continuous-marker-advance", queue_size=2)
        accept_writer_test_frame(
            continuous_bridge,
            continuous_capture_dir,
            continuous_identity,
            continuous_prompt,
            1,
        )
        assert continuous_bridge.window_writer.wait_idle(2.0)
        original_continuous_reader = continuous_bridge.marker_reader
        continuous_calls = [0]

        def continuous_advancing_reader(path: Path, require_metadata: bool) -> CurrentMarkerState:
            continuous_calls[0] += 1
            return read_marker_while_advancing(
                original_continuous_reader,
                path,
                require_metadata,
                advance_count=3,
            )

        continuous_bridge.marker_reader = continuous_advancing_reader
        continuous_generation = continuous_bridge.source_generation
        continuous_bridge.process_writer_completions()
        assert continuous_calls[0] == 4
        assert continuous_bridge.source_generation == continuous_generation
        assert continuous_bridge.published_windows == 1
        continuous_marker = read_json_file(continuous_capture_dir / "latest.json")
        assert continuous_marker["seq"] == 13
        assert continuous_marker["source_frame"] == 13
        assert continuous_bridge.terminate_window_pipeline("self_check_complete")

        # A normal post-promote advance may be followed by a real session switch;
        # the final guard must still revoke the live pair and reset the source.
        (
            advance_reset_bridge,
            advance_reset_capture_dir,
            _advance_reset_control_path,
            advance_reset_identity,
            advance_reset_prompt,
        ) = make_writer_test_bridge("marker-advance-then-reset", queue_size=2)
        accept_writer_test_frame(
            advance_reset_bridge,
            advance_reset_capture_dir,
            advance_reset_identity,
            advance_reset_prompt,
            1,
        )
        assert advance_reset_bridge.window_writer.wait_idle(2.0)
        original_advance_reset_reader = advance_reset_bridge.marker_reader
        advance_reset_calls = [0]

        def advance_then_reset_reader(path: Path, require_metadata: bool) -> CurrentMarkerState:
            advance_reset_calls[0] += 1
            if advance_reset_calls[0] == 3:
                return read_marker_while_advancing(
                    original_advance_reset_reader,
                    path,
                    require_metadata,
                )
            if advance_reset_calls[0] == 4:
                config_path = advance_reset_capture_dir / "run_config.env"
                config_text = config_path.read_text(encoding="ascii")
                config_path.write_text(
                    config_text.replace(
                        "SOURCE_SESSION_ID=self-check-session",
                        "SOURCE_SESSION_ID=self-check-session-reset",
                    ),
                    encoding="ascii",
                )
                advance_latest_marker(advance_reset_capture_dir)
            return original_advance_reset_reader(path, require_metadata)

        advance_reset_bridge.marker_reader = advance_then_reset_reader
        advance_reset_generation = advance_reset_bridge.source_generation
        advance_reset_bridge.process_writer_completions()
        assert advance_reset_calls[0] == 4
        assert advance_reset_bridge.source_generation == advance_reset_generation + 1
        assert advance_reset_bridge.source_reset_reason == "source_session_changed"
        assert not advance_reset_bridge.deferred_results
        assert not list(advance_reset_bridge.staging_dir.iterdir())
        assert not list(advance_reset_bridge.out_dir.glob("window_*.npz"))
        assert not list(advance_reset_bridge.out_dir.glob("window_*.gbuffers.npy"))
        assert not list(advance_reset_bridge.out_dir.glob(".manifest.json.*.authority.tmp"))
        assert advance_reset_bridge.terminate_window_pipeline("self_check_complete")

        # A run_config/session switch between the two identity reads invalidates
        # the promotion fence even when latest.json itself did not move.
        identity_race_capture = root / "identity-fence-capture"
        write_fake_capture(identity_race_capture, seq=1, width=width, height=height)
        original_identity_reader = read_source_identity
        identity_read_count = [0]

        def switching_identity_reader(*reader_args: Any, **reader_kwargs: Any) -> SourceIdentity:
            identity_read_count[0] += 1
            if identity_read_count[0] == 2:
                config_path = identity_race_capture / "run_config.env"
                config_text = config_path.read_text(encoding="ascii")
                config_path.write_text(
                    config_text.replace(
                        "SOURCE_SESSION_ID=self-check-session",
                        "SOURCE_SESSION_ID=self-check-session-switched",
                    ),
                    encoding="ascii",
                )
            return original_identity_reader(*reader_args, **reader_kwargs)

        globals()["read_source_identity"] = switching_identity_reader
        identity_change_rejected = False
        try:
            read_stable_current_marker(identity_race_capture, require_metadata=True)
        except InconsistentCapture as exc:
            identity_change_rejected = "identity changed" in str(exc)
        finally:
            globals()["read_source_identity"] = original_identity_reader
        assert identity_read_count[0] == 2
        assert identity_change_rejected

        # Switch session after the post-promote guard and manifest preparation.
        # The final guard must revoke the unauthorized live pair before reset.
        (
            outer_guard_bridge,
            outer_guard_capture_dir,
            _outer_guard_control_path,
            outer_guard_identity,
            outer_guard_prompt,
        ) = make_writer_test_bridge("outer-guard-session-switch", queue_size=2)
        original_manifest_prepare = outer_guard_bridge.manifest_prepare_fn

        def switch_session_after_manifest_prepare(
            path: Path,
            document: Mapping[str, Any],
        ) -> Path:
            temporary = original_manifest_prepare(path, document)
            config_path = outer_guard_capture_dir / "run_config.env"
            config_text = config_path.read_text(encoding="ascii")
            config_path.write_text(
                config_text.replace(
                    "SOURCE_SESSION_ID=self-check-session",
                    "SOURCE_SESSION_ID=self-check-session-outer-guard",
                ),
                encoding="ascii",
            )
            latest_path = outer_guard_capture_dir / "latest.json"
            atomic_write_json(latest_path, read_json_file(latest_path))
            return temporary

        outer_guard_bridge.manifest_prepare_fn = switch_session_after_manifest_prepare
        accept_writer_test_frame(
            outer_guard_bridge,
            outer_guard_capture_dir,
            outer_guard_identity,
            outer_guard_prompt,
            1,
        )
        outer_guard_bridge.wait_for_writer_idle()
        outer_manifest = read_json_file(outer_guard_bridge.manifest_path)
        assert outer_manifest["ready"] is False
        assert outer_guard_bridge.source_reset_reason == "source_session_changed"
        assert not list(outer_guard_bridge.out_dir.glob("window_*.npz"))
        assert not list(outer_guard_bridge.out_dir.glob("window_*.gbuffers.npy"))
        assert not outer_guard_bridge.alias_path.exists()
        assert not outer_guard_bridge.gbuffer_alias_path.exists()
        assert not outer_guard_bridge.authorized_window_files
        assert not list(outer_guard_bridge.out_dir.glob(".manifest.json.*.authority.tmp"))
        assert outer_guard_bridge.terminate_window_pipeline("self_check_complete")

        def manifest_failure_check(label: str, replace_failure: bool) -> None:
            (
                failure_bridge,
                failure_capture_dir,
                _failure_control_path,
                failure_identity,
                failure_prompt,
            ) = make_writer_test_bridge(label, queue_size=2)
            if replace_failure:

                def fail_manifest_replace(_temporary: Path, _destination: Path) -> None:
                    raise OSError("injected manifest replace failure")

                failure_bridge.manifest_replace_fn = fail_manifest_replace
            else:
                original_prepare = failure_bridge.manifest_prepare_fn

                def fail_manifest_prepare(
                    path: Path,
                    document: Mapping[str, Any],
                ) -> Path:
                    original_prepare(path, document)
                    raise OSError("injected manifest temp failure")

                failure_bridge.manifest_prepare_fn = fail_manifest_prepare
            accept_writer_test_frame(
                failure_bridge,
                failure_capture_dir,
                failure_identity,
                failure_prompt,
                1,
            )
            failure_seen = False
            try:
                failure_bridge.wait_for_writer_idle()
            except WindowPipelineFailure as exc:
                failure_seen = "manifest" in str(exc)
            assert failure_seen
            failure_manifest = read_json_file(failure_bridge.manifest_path)
            assert failure_manifest["ready"] is False
            assert not failure_bridge.authorized_window_files
            assert not list(failure_bridge.out_dir.glob("window_*.npz"))
            assert not list(failure_bridge.out_dir.glob("window_*.gbuffers.npy"))
            assert not list(failure_bridge.out_dir.glob(".manifest.json.*.authority.tmp"))
            assert not failure_bridge.alias_path.exists()
            assert not failure_bridge.gbuffer_alias_path.exists()
            assert failure_bridge.terminate_window_pipeline("manifest_failure")

        manifest_failure_check("manifest-temp-failure", replace_failure=False)
        manifest_failure_check("manifest-replace-failure", replace_failure=True)

        # os.replace is the authority commit point. A following directory-fsync
        # failure must preserve the committed pair and memory state, retain any
        # newly retired predecessor, and terminate without alias publication.
        (
            committed_publish_bridge,
            committed_publish_capture_dir,
            _committed_publish_control_path,
            committed_publish_identity,
            committed_publish_prompt,
        ) = make_writer_test_bridge(
            "manifest-post-replace-publish",
            queue_size=2,
            retain_windows=1,
            max_worker_lag_windows=0,
        )
        accept_writer_test_frame(
            committed_publish_bridge,
            committed_publish_capture_dir,
            committed_publish_identity,
            committed_publish_prompt,
            1,
        )
        accept_writer_test_frame(
            committed_publish_bridge,
            committed_publish_capture_dir,
            committed_publish_identity,
            committed_publish_prompt,
            2,
        )
        committed_publish_bridge.wait_for_writer_idle()
        publish_reader = strict_worker_reader(committed_publish_bridge)
        publish_authority = publish_reader.authority()
        assert publish_authority.token is not None
        publish_reader.set_active_token(publish_authority.token)
        publish_first = publish_reader.next_window_path(publish_authority.token, None)
        publish_reader.load_window(publish_first, publish_authority.token)
        publish_second = publish_reader.next_window_path(publish_authority.token, 1)
        publish_reader.load_window(publish_second, publish_authority.token)

        def publish_replace_then_fsync_failure(
            temporary: Path,
            destination: Path,
        ) -> None:
            os.replace(temporary, destination)
            raise AuthorityManifestReplaceError(
                "post-replace-fsync",
                True,
                "OSError: injected publish directory fsync failure",
            )

        committed_publish_bridge.manifest_replace_fn = publish_replace_then_fsync_failure
        accept_writer_test_frame(
            committed_publish_bridge,
            committed_publish_capture_dir,
            committed_publish_identity,
            committed_publish_prompt,
            3,
        )
        publish_durability_failure = False
        try:
            committed_publish_bridge.wait_for_writer_idle()
        except WindowPipelineFailure as exc:
            publish_durability_failure = "committed-but-not-durable" in str(exc)
        assert publish_durability_failure
        assert committed_publish_bridge.pipeline_terminal
        committed_publish_manifest = read_json_file(committed_publish_bridge.manifest_path)
        assert committed_publish_manifest["ready"] is True
        assert committed_publish_manifest["window_id"] == 3
        assert committed_publish_bridge.current_window is not None
        assert committed_publish_bridge.current_window["id"] == 3
        assert committed_publish_bridge.previous_window_id == 3
        assert committed_publish_bridge.published_windows == 3
        assert list(committed_publish_bridge.authorized_window_files) == [
            "window_000000000002.npz",
            "window_000000000003.npz",
        ]
        publish_paths = [committed_publish_bridge.out_dir / f"window_{window_id:012d}.npz" for window_id in (1, 2, 3)]
        assert all(path.is_file() for path in publish_paths)
        if window_transport == "npy":
            assert all(gbuffer_sidecar_path(path).is_file() for path in publish_paths)
        assert not committed_publish_bridge.alias_path.exists()
        assert not committed_publish_bridge.gbuffer_alias_path.exists()
        publish_third = publish_reader.next_window_path(publish_authority.token, 2)
        publish_third_window = publish_reader.load_window(publish_third, publish_authority.token)
        assert publish_third_window.header.window_id == 3
        assert committed_publish_bridge.authority_durability_failure_count == 1

        restarted_publish_bridge = Bridge(committed_publish_bridge.args)
        restarted_publish_manifest = read_json_file(restarted_publish_bridge.manifest_path)
        assert restarted_publish_manifest["ready"] is False
        assert restarted_publish_manifest["retained_windows"] == []
        assert not restarted_publish_bridge.authorized_window_files
        assert publish_paths[0].is_file()
        assert not publish_paths[1].exists()
        assert not publish_paths[2].exists()
        restarted_publish_bridge.adopt_source(committed_publish_identity)
        accept_writer_test_frame(
            restarted_publish_bridge,
            committed_publish_capture_dir,
            committed_publish_identity,
            committed_publish_prompt,
            4,
        )
        restarted_publish_bridge.wait_for_writer_idle()
        recovered_publish_manifest = read_json_file(restarted_publish_bridge.manifest_path)
        assert recovered_publish_manifest["window"]["is_initial"] is True
        assert recovered_publish_manifest["retained_windows"] == [recovered_publish_manifest["immutable_file"]]
        recovered_publish_reader = strict_worker_reader(restarted_publish_bridge)
        recovered_publish_authority = recovered_publish_reader.authority()
        assert recovered_publish_authority.token is not None
        recovered_publish_path = recovered_publish_reader.next_window_path(recovered_publish_authority.token, None)
        recovered_publish_reader.load_window(recovered_publish_path, recovered_publish_authority.token)
        assert restarted_publish_bridge.terminate_window_pipeline("self_check_complete")

        # The same post-replace failure while revoking authority keeps memory
        # empty and old files unmanifested. Restart must not re-authorize those
        # orphans, and its next initial window must be worker-readable.
        (
            committed_revoke_bridge,
            committed_revoke_capture_dir,
            committed_revoke_control_path,
            committed_revoke_identity,
            committed_revoke_prompt,
        ) = make_writer_test_bridge("manifest-post-replace-revoke", queue_size=2)
        accept_writer_test_frame(
            committed_revoke_bridge,
            committed_revoke_capture_dir,
            committed_revoke_identity,
            committed_revoke_prompt,
            1,
        )
        accept_writer_test_frame(
            committed_revoke_bridge,
            committed_revoke_capture_dir,
            committed_revoke_identity,
            committed_revoke_prompt,
            2,
        )
        committed_revoke_bridge.wait_for_writer_idle()
        revoked_paths = committed_revoke_bridge.committed_live_paths()
        assert len(revoked_paths) == 2

        def revoke_replace_then_fsync_failure(
            temporary: Path,
            destination: Path,
        ) -> None:
            os.replace(temporary, destination)
            raise AuthorityManifestReplaceError(
                "post-replace-fsync",
                True,
                "OSError: injected revoke directory fsync failure",
            )

        committed_revoke_bridge.manifest_replace_fn = revoke_replace_then_fsync_failure
        next_revoke_control = {
            **active_control,
            "reset_epoch": 2,
            "render_request_id": 2,
            "prompt_revision": 2,
            "reset_reason": "post_replace_revoke",
            "prompt": "authority revoked after replace",
            "reset_source_seq": 2,
            "reset_source_frame": 2,
        }
        atomic_write_json(committed_revoke_control_path, next_revoke_control)
        next_revoke_prompt = read_prompt_state(committed_revoke_control_path)
        revoke_durability_failure = False
        try:
            committed_revoke_bridge.observe_control(next_revoke_prompt, current_seq=2)
        except WindowPipelineFailure as exc:
            revoke_durability_failure = "committed-but-not-durable" in str(exc)
        assert revoke_durability_failure
        assert committed_revoke_bridge.pipeline_terminal
        committed_revoke_manifest = read_json_file(committed_revoke_bridge.manifest_path)
        assert committed_revoke_manifest["ready"] is False
        assert committed_revoke_manifest["window"] is None
        assert committed_revoke_manifest["retained_windows"] == []
        assert committed_revoke_manifest["retained_window_files"] == []
        assert committed_revoke_bridge.current_window is None
        assert not committed_revoke_bridge.authorized_window_files
        assert all(path.is_file() for path in revoked_paths)
        if window_transport == "npy":
            assert all(gbuffer_sidecar_path(path).is_file() for path in revoked_paths)
        revoke_reader = strict_worker_reader(committed_revoke_bridge)
        revoked_authority = revoke_reader.authority()
        assert revoked_authority.token is None
        assert revoked_authority.reason == "manifest_not_ready"
        assert committed_revoke_bridge.authority_durability_failure_count == 1

        restarted_revoke_bridge = Bridge(committed_revoke_bridge.args)
        restarted_revoke_manifest = read_json_file(restarted_revoke_bridge.manifest_path)
        assert restarted_revoke_manifest["retained_windows"] == []
        assert not restarted_revoke_bridge.authorized_window_files
        assert all(path.is_file() for path in revoked_paths)
        restarted_revoke_bridge.adopt_source(committed_revoke_identity)
        current_revoke_capture = read_committed_capture(
            committed_revoke_capture_dir,
            committed_revoke_identity,
            width,
            height,
        )
        restarted_revoke_bridge.accept_capture(current_revoke_capture, next_revoke_prompt)
        accept_writer_test_frame(
            restarted_revoke_bridge,
            committed_revoke_capture_dir,
            committed_revoke_identity,
            next_revoke_prompt,
            3,
        )
        restarted_revoke_bridge.wait_for_writer_idle()
        recovered_revoke_manifest = read_json_file(restarted_revoke_bridge.manifest_path)
        assert recovered_revoke_manifest["window"]["is_initial"] is True
        assert recovered_revoke_manifest["retained_windows"] == [recovered_revoke_manifest["immutable_file"]]
        recovered_revoke_reader = strict_worker_reader(restarted_revoke_bridge)
        recovered_revoke_authority = recovered_revoke_reader.authority()
        assert recovered_revoke_authority.token is not None
        recovered_revoke_path = recovered_revoke_reader.next_window_path(recovered_revoke_authority.token, None)
        recovered_revoke_reader.load_window(recovered_revoke_path, recovered_revoke_authority.token)
        assert restarted_revoke_bridge.terminate_window_pipeline("self_check_complete")

        # A transient marker read defers the oldest completion and preserves order.
        (
            defer_bridge,
            defer_capture_dir,
            _defer_control_path,
            defer_identity,
            defer_prompt,
        ) = make_writer_test_bridge("promote-defer", queue_size=2)
        accept_writer_test_frame(defer_bridge, defer_capture_dir, defer_identity, defer_prompt, 1)
        accept_writer_test_frame(defer_bridge, defer_capture_dir, defer_identity, defer_prompt, 2)
        assert defer_bridge.window_writer.wait_idle(2.0)
        original_marker_reader = defer_bridge.marker_reader
        defer_once = [True]

        def transient_marker_reader(path: Path, require_metadata: bool) -> CurrentMarkerState:
            if defer_once[0]:
                defer_once[0] = False
                raise SourceNotReady("injected unstable promotion marker")
            return original_marker_reader(path, require_metadata)

        defer_bridge.marker_reader = transient_marker_reader
        defer_bridge.process_writer_completions()
        assert len(defer_bridge.deferred_results) == 2
        assert not list(defer_bridge.out_dir.glob("window_*.npz"))
        defer_bridge.process_writer_completions()
        assert defer_bridge.published_windows == 2
        assert [path.name for path in defer_bridge.committed_live_paths()] == [
            "window_000000000001.npz",
            "window_000000000002.npz",
        ]
        assert defer_bridge.current_window is not None
        assert defer_bridge.current_window["previous_window_id"] == 1
        assert defer_bridge.terminate_window_pipeline("self_check_complete")

        # A prompt epoch reset revokes every prior window before deleting it.
        # The first replacement window is a fresh initial window and is the only
        # file exposed by the new manifest authority.
        (
            epoch_bridge,
            epoch_capture_dir,
            epoch_control_path,
            epoch_identity,
            epoch_prompt,
        ) = make_writer_test_bridge("epoch-authority-reset", queue_size=2)
        accept_writer_test_frame(epoch_bridge, epoch_capture_dir, epoch_identity, epoch_prompt, 1)
        epoch_bridge.wait_for_writer_idle()
        old_epoch_paths = epoch_bridge.committed_live_paths()
        assert len(old_epoch_paths) == 1
        assert old_epoch_paths[0].is_file()
        next_epoch_control = {
            **active_control,
            "reset_epoch": 2,
            "render_request_id": 2,
            "prompt_revision": 2,
            "reset_reason": "epoch_authority_reset",
            "prompt": "fresh authority epoch",
            "reset_source_seq": 1,
            "reset_source_frame": 1,
        }
        atomic_write_json(epoch_control_path, next_epoch_control)
        next_epoch_prompt = read_prompt_state(epoch_control_path)
        assert epoch_bridge.observe_control(next_epoch_prompt, current_seq=1)
        revoked_manifest = read_json_file(epoch_bridge.manifest_path)
        assert revoked_manifest["ready"] is False
        assert revoked_manifest["window"] is None
        assert revoked_manifest["retained_windows"] == []
        assert revoked_manifest["retained_immutable_files"] == []
        assert revoked_manifest["retained_window_files"] == []
        assert not epoch_bridge.authorized_window_files
        assert all(not path.exists() for path in old_epoch_paths)
        accept_writer_test_frame(
            epoch_bridge,
            epoch_capture_dir,
            epoch_identity,
            next_epoch_prompt,
            2,
        )
        epoch_bridge.wait_for_writer_idle()
        next_epoch_manifest = read_json_file(epoch_bridge.manifest_path)
        next_epoch_window = next_epoch_manifest["window"]
        assert next_epoch_manifest["ready"] is True
        assert next_epoch_window["is_initial"] is True
        assert next_epoch_window["previous_window_id"] == -1
        assert next_epoch_manifest["retained_windows"] == [next_epoch_window["immutable_file"]]
        assert next_epoch_manifest["retained_window_files"] == [
            {
                "immutable_file": next_epoch_window["immutable_file"],
                "gbuffers_file": next_epoch_window["gbuffers_file"],
            }
        ]
        assert epoch_bridge.terminate_window_pipeline("self_check_complete")

        # Restart authority is reconstructed only from explicit manifest entries.
        # A valid-looking live pair that was never listed remains an orphan.
        (
            restart_source_bridge,
            restart_capture_dir,
            _restart_control_path,
            restart_identity,
            restart_prompt,
        ) = make_writer_test_bridge("authority-restart", queue_size=2)
        accept_writer_test_frame(
            restart_source_bridge,
            restart_capture_dir,
            restart_identity,
            restart_prompt,
            1,
        )
        restart_source_bridge.wait_for_writer_idle()
        authorized_before_restart = restart_source_bridge.committed_live_paths()
        assert len(authorized_before_restart) == 1
        orphan_metadata = restart_source_bridge.out_dir / "window_000000000124.npz"
        os.link(authorized_before_restart[0], orphan_metadata)
        orphan_sidecar = gbuffer_sidecar_path(orphan_metadata)
        if window_transport == "npy":
            os.link(
                gbuffer_sidecar_path(authorized_before_restart[0]),
                orphan_sidecar,
            )
        restart_args = restart_source_bridge.args
        assert restart_source_bridge.terminate_window_pipeline("restart_self_check")
        restarted_bridge = Bridge(restart_args)
        restarted_manifest = read_json_file(restarted_bridge.manifest_path)
        assert restarted_manifest["ready"] is False
        assert restarted_manifest["retained_windows"] == []
        assert restarted_manifest["retained_window_files"] == []
        assert not restarted_bridge.authorized_window_files
        assert orphan_metadata.is_file()
        if window_transport == "npy":
            assert orphan_sidecar.is_file()
        assert all(not path.exists() for path in authorized_before_restart)
        orphan_metadata.unlink()
        orphan_sidecar.unlink(missing_ok=True)
        assert restarted_bridge.terminate_window_pipeline("self_check_complete")

        # Retention only sees committed live pairs. It cannot remove a staged job,
        # and an intentionally low setting still retains the current predecessor.
        (
            cleanup_bridge,
            cleanup_capture_dir,
            _cleanup_control_path,
            cleanup_identity,
            cleanup_prompt,
        ) = make_writer_test_bridge(
            "cleanup-overlap",
            queue_size=2,
            retain_windows=1,
            max_worker_lag_windows=0,
        )
        assert cleanup_bridge.effective_retain_windows == 2
        accept_writer_test_frame(cleanup_bridge, cleanup_capture_dir, cleanup_identity, cleanup_prompt, 1)
        cleanup_bridge.wait_for_writer_idle()
        cleanup_stage_ready = threading.Event()
        cleanup_stage_release = threading.Event()
        original_cleanup_save_npz = cleanup_bridge.window_writer._save_npz_fn

        def held_cleanup_save_npz(
            path: Path,
            arrays: Mapping[str, np.ndarray],
            compress: bool,
        ) -> None:
            original_cleanup_save_npz(path, arrays, compress)
            cleanup_stage_ready.set()
            if not cleanup_stage_release.wait(2.0):
                raise TimeoutError("cleanup-overlap self-check did not release writer")

        cleanup_bridge.window_writer._save_npz_fn = held_cleanup_save_npz
        accept_writer_test_frame(cleanup_bridge, cleanup_capture_dir, cleanup_identity, cleanup_prompt, 2)
        assert cleanup_stage_ready.wait(2.0)
        staged_before_cleanup = {path.name for path in cleanup_bridge.staging_dir.iterdir()}
        cleanup_bridge.cleanup_windows()
        assert staged_before_cleanup == {path.name for path in cleanup_bridge.staging_dir.iterdir()}
        assert [path.name for path in cleanup_bridge.committed_live_paths()] == ["window_000000000001.npz"]
        cleanup_stage_release.set()
        cleanup_bridge.wait_for_writer_idle()
        cleanup_bridge.window_writer._save_npz_fn = original_cleanup_save_npz
        accept_writer_test_frame(cleanup_bridge, cleanup_capture_dir, cleanup_identity, cleanup_prompt, 3)
        cleanup_bridge.wait_for_writer_idle()
        retained_chain = cleanup_bridge.committed_live_paths()
        assert [path.name for path in retained_chain] == [
            "window_000000000002.npz",
            "window_000000000003.npz",
        ]
        assert cleanup_bridge.current_window is not None
        predecessor = cleanup_bridge.out_dir / (
            f"window_{cleanup_bridge.current_window['previous_window_id']:012d}.npz"
        )
        assert predecessor.is_file()
        assert not list(cleanup_bridge.staging_dir.iterdir())
        assert cleanup_bridge.terminate_window_pipeline("self_check_complete")

        # A fatal writer error is terminal: pending jobs, results, and staging are
        # cleared, and no later completion can enter the live namespace.
        (
            error_bridge,
            error_capture_dir,
            _error_control_path,
            error_identity,
            error_prompt,
        ) = make_writer_test_bridge("writer-error", queue_size=2)

        fatal_started = threading.Event()
        fatal_release = threading.Event()

        def injected_writer_error(
            _path: Path,
            _arrays: Mapping[str, np.ndarray],
            _compress: bool,
        ) -> None:
            fatal_started.set()
            if not fatal_release.wait(2.0):
                raise TimeoutError("fatal self-check did not release writer")
            raise OSError("injected window writer failure")

        error_bridge.window_writer._save_npz_fn = injected_writer_error
        accept_writer_test_frame(error_bridge, error_capture_dir, error_identity, error_prompt, 1)
        assert fatal_started.wait(2.0)
        accept_writer_test_frame(error_bridge, error_capture_dir, error_identity, error_prompt, 2)
        assert error_bridge.window_writer.summary()["pending_jobs"] == 1
        fatal_release.set()
        assert error_bridge.window_writer.wait_idle(2.0)
        writer_error_propagated = False
        try:
            error_bridge.process_writer_completions()
        except WindowPipelineFailure as exc:
            writer_error_propagated = "injected window writer failure" in str(exc)
        assert writer_error_propagated
        assert error_bridge.window_writer.summary()["errors"] == 1
        assert error_bridge.terminate_window_pipeline("fatal_self_check")
        assert error_bridge.pipeline_terminal
        assert error_bridge.window_writer.summary()["pending_jobs"] == 0
        assert not list(error_bridge.out_dir.glob("window_*.npz"))
        assert not list(error_bridge.out_dir.glob("window_*.gbuffers.npy"))
        assert not list(error_bridge.staging_dir.iterdir())

        # A completed staged result followed by SIGTERM-style shutdown is deleted
        # without ever publishing a live metadata commit point.
        (
            stop_bridge,
            stop_capture_dir,
            _stop_control_path,
            stop_identity,
            stop_prompt,
        ) = make_writer_test_bridge("completed-then-stop", queue_size=2)
        accept_writer_test_frame(stop_bridge, stop_capture_dir, stop_identity, stop_prompt, 1)
        assert stop_bridge.window_writer.wait_idle(2.0)
        assert list(stop_bridge.staging_dir.iterdir())
        assert not list(stop_bridge.out_dir.glob("window_*.npz"))
        assert stop_bridge.terminate_window_pipeline("signal: SIGTERM")
        stop_bridge.process_writer_completions()
        assert not list(stop_bridge.staging_dir.iterdir())
        assert not list(stop_bridge.out_dir.glob("window_*.npz"))

        # With one active and one pending job, the next job must never block.
        # Queue saturation resets causal history and invalidates both old jobs.
        (
            full_bridge,
            full_capture_dir,
            _full_control_path,
            full_identity,
            full_prompt,
        ) = make_writer_test_bridge("writer-full", queue_size=1)
        full_pair_written = threading.Event()
        release_full_pair = threading.Event()
        original_full_save_npz = full_bridge.window_writer._save_npz_fn

        def held_full_save_npz(
            path: Path,
            arrays: Mapping[str, np.ndarray],
            compress: bool,
        ) -> None:
            original_full_save_npz(path, arrays, compress)
            full_pair_written.set()
            if not release_full_pair.wait(2.0):
                raise TimeoutError("queue-full self-check did not release writer")

        full_bridge.window_writer._save_npz_fn = held_full_save_npz
        accept_writer_test_frame(full_bridge, full_capture_dir, full_identity, full_prompt, 1)
        assert full_pair_written.wait(2.0)
        accept_writer_test_frame(full_bridge, full_capture_dir, full_identity, full_prompt, 2)
        generation_before_full = full_bridge.source_generation
        full_started = time.monotonic()
        accept_writer_test_frame(full_bridge, full_capture_dir, full_identity, full_prompt, 3)
        full_duration = time.monotonic() - full_started
        assert full_duration < 0.10
        assert full_bridge.writer_queue_full_resets == 1
        assert full_bridge.source_generation == generation_before_full + 1
        assert full_bridge.sample_index == 0
        assert not full_bridge.scheduler.frames
        release_full_pair.set()
        full_bridge.wait_for_writer_idle()
        assert not list(full_bridge.out_dir.glob("window_*.npz"))
        assert not list(full_bridge.out_dir.glob("window_*.gbuffers.npy"))
        assert not full_bridge.alias_path.exists()
        assert not full_bridge.gbuffer_alias_path.exists()
        assert not list(full_bridge.staging_dir.iterdir())
        assert full_bridge.terminate_window_pipeline("self_check_complete")

        # Alias creation must observe the new immutable pair in the already
        # committed manifest authority, never merely in the live namespace.
        (
            alias_order_bridge,
            alias_order_capture_dir,
            _alias_order_control_path,
            alias_order_identity,
            alias_order_prompt,
        ) = make_writer_test_bridge("alias-authority-order", queue_size=2)
        alias_authority_observations: list[bool] = []

        def authority_guarded_alias(source: Path, alias: Path) -> None:
            authority_manifest = read_json_file(alias_order_bridge.manifest_path)
            authorized_names = {
                str(entry.get("immutable_file", "") or "")
                for entry in authority_manifest.get("retained_window_files", [])
                if isinstance(entry, Mapping)
            }
            authorized_names.update(
                str(entry.get("gbuffers_file", "") or "")
                for entry in authority_manifest.get("retained_window_files", [])
                if isinstance(entry, Mapping)
            )
            alias_authority_observations.append(source.name in authorized_names)
            atomic_alias(source, alias)

        alias_order_bridge.alias_fn = authority_guarded_alias
        accept_writer_test_frame(
            alias_order_bridge,
            alias_order_capture_dir,
            alias_order_identity,
            alias_order_prompt,
            1,
        )
        alias_order_bridge.wait_for_writer_idle()
        alias_order_manifest = read_json_file(alias_order_bridge.manifest_path)
        assert alias_authority_observations
        assert all(alias_authority_observations)
        assert alias_order_manifest["window"]["alias_status"]["metadata"] is True
        assert alias_order_bridge.alias_path.is_file()
        if window_transport == "npy":
            assert alias_order_manifest["window"]["alias_status"]["gbuffers"] is True
            assert alias_order_bridge.gbuffer_alias_path.is_file()
        assert alias_order_bridge.terminate_window_pipeline("self_check_complete")

        # Alias publication is compatibility-only. Either alias may fail after
        # metadata commit without deleting the immutable committed pair.
        (
            alias_bridge,
            alias_capture_dir,
            _alias_control_path,
            alias_identity,
            alias_prompt,
        ) = make_writer_test_bridge("alias-failure", queue_size=2)
        if window_transport == "npy":

            def fail_sidecar_alias(source: Path, alias: Path) -> None:
                if alias == alias_bridge.gbuffer_alias_path:
                    raise OSError("injected sidecar alias failure")
                atomic_alias(source, alias)

            alias_bridge.alias_fn = fail_sidecar_alias
        else:

            def fail_first_metadata_alias(source: Path, alias: Path) -> None:
                if alias == alias_bridge.alias_path:
                    raise OSError("injected metadata alias failure")
                atomic_alias(source, alias)

            alias_bridge.alias_fn = fail_first_metadata_alias
        accept_writer_test_frame(alias_bridge, alias_capture_dir, alias_identity, alias_prompt, 1)
        alias_bridge.wait_for_writer_idle()
        alias_manifest = read_json_file(alias_bridge.manifest_path)
        alias_metadata = alias_bridge.out_dir / "window_000000000001.npz"
        assert alias_metadata.is_file()
        if window_transport == "npy":
            assert gbuffer_sidecar_path(alias_metadata).is_file()
            assert alias_manifest["window"]["alias_status"]["gbuffers"] is False
            assert alias_manifest["window"]["alias_status"]["metadata"] is False
            assert not alias_bridge.alias_path.exists()
            assert not alias_bridge.gbuffer_alias_path.exists()
            assert alias_bridge.alias_failure_count == 1

            def fail_metadata_alias(source: Path, alias: Path) -> None:
                if alias == alias_bridge.alias_path:
                    raise OSError("injected metadata alias failure")
                atomic_alias(source, alias)

            alias_bridge.alias_fn = fail_metadata_alias
            accept_writer_test_frame(alias_bridge, alias_capture_dir, alias_identity, alias_prompt, 2)
            alias_bridge.wait_for_writer_idle()
            alias_manifest = read_json_file(alias_bridge.manifest_path)
            second_metadata = alias_bridge.out_dir / "window_000000000002.npz"
            assert second_metadata.is_file()
            assert gbuffer_sidecar_path(second_metadata).is_file()
            assert alias_manifest["window"]["alias_status"]["metadata"] is False
            assert alias_manifest["immutable_file"] == second_metadata.name
            assert alias_manifest["aliases"]["metadata_file"] == ""
            assert not alias_bridge.alias_path.exists()
            assert not alias_bridge.gbuffer_alias_path.exists()
            assert alias_bridge.alias_failure_count == 2
        else:
            assert alias_manifest["window"]["alias_status"]["metadata"] is False
            assert alias_manifest["immutable_file"] == alias_metadata.name
            assert alias_bridge.alias_failure_count == 1
        assert alias_bridge.current_window is not None
        assert alias_bridge.current_window["id"] == alias_bridge.window_id
        assert alias_bridge.terminate_window_pipeline("self_check_complete")

        bridge.wait_for_writer_idle()
        lag_bridge.wait_for_writer_idle()
        input_bridge.wait_for_writer_idle()

        # Startup/restart recovery is a separate, non-failure handshake.
        fresh_capture_dir = root / "fresh-capture"
        fresh_out_dir = root / "fresh-out"
        fresh_control_path = root / "fresh-control" / "live_control.json"
        fresh_worker_status_path = root / "fresh-control" / "worker_status.json"
        fresh_rendered_dir = root / "fresh-rendered"
        fresh_rendered_dir.mkdir(parents=True)
        rendered_sentinel = b'{"seq":99,"source_reset_epoch":0}'
        (fresh_rendered_dir / "latest.json").write_bytes(rendered_sentinel)
        fresh_control = {
            **active_control,
            "reset_source_seq": 400,
            "reset_source_frame": 400,
        }
        atomic_write_json(fresh_control_path, fresh_control)
        write_fake_capture(fresh_capture_dir, seq=400, width=width, height=height)
        fresh_identity = read_source_identity(fresh_capture_dir, 0.0, True)
        fresh_args = parser.parse_args(
            [
                "--capture-dir",
                str(fresh_capture_dir),
                "--out-dir",
                str(fresh_out_dir),
                "--live-control",
                str(fresh_control_path),
                "--worker-status",
                str(fresh_worker_status_path),
                "--expected-width",
                str(width),
                "--expected-height",
                str(height),
                "--target-width",
                "32",
                "--target-height",
                "16",
                "--cadence",
                "2,3",
                "--max-worker-lag-windows",
                "1",
                "--worker-lag-observations",
                "1",
                "--stale-after",
                "0",
            ]
        )
        configure_temporal_cadence(fresh_args, parser)
        validate_args(fresh_args, parser)
        fresh_bridge = Bridge(fresh_args)
        fresh_bridge.adopt_source(fresh_identity)
        fresh_prompt = read_prompt_state(fresh_control_path)
        fresh_bridge.window_id = 100
        generation_before_startup = fresh_bridge.source_generation
        recovery_before_startup = fresh_bridge.recovery_summary(fresh_prompt)["count"]

        cold_loading = {
            "schema_version": SCHEMA_VERSION,
            "worker_instance_id": "cold-start-worker",
            "status": "loading",
            "ready": False,
            "backlog_eligible": False,
            "updated_at": time.time(),
            "startup_elapsed_seconds": 73.0,
            "reset_epoch": fresh_prompt.reset_epoch,
            "source_reset_epoch": fresh_bridge.source_generation,
            "prompt_revision": fresh_prompt.prompt_revision,
            "track_id": "self_check",
            "source_session_id": "self-check-session",
            "accepted_request_id": fresh_prompt.render_request_id,
            "accepted_window_id": 0,
            "completed_window_id": 0,
        }
        atomic_write_json(fresh_worker_status_path, cold_loading)
        assert fresh_bridge.handle_worker_backlog(fresh_prompt, 400) is False
        cold_loading.update(
            {
                "status": "rendering",
                "ready": True,
                "warmed": False,
                "updated_at": time.time(),
            }
        )
        atomic_write_json(fresh_worker_status_path, cold_loading)
        assert fresh_bridge.handle_worker_backlog(fresh_prompt, 400) is False
        assert fresh_bridge.source_generation == generation_before_startup
        assert fresh_bridge.worker_lag_observations == 0

        def fresh_request(instance_id: str) -> dict[str, Any]:
            requested_at = time.time() - 0.02
            return {
                **cold_loading,
                "worker_instance_id": instance_id,
                "status": "fresh_epoch_requested",
                "ready": True,
                "warmed": True,
                "backlog_eligible": False,
                "needs_fresh_source": True,
                "fresh_epoch_warmup_completed": True,
                "fresh_epoch_request_id": (f"{instance_id}:{fresh_bridge.source_generation}"),
                "fresh_epoch_requested_at": requested_at,
                "fresh_epoch_requested_source_reset_epoch": (fresh_bridge.source_generation),
                "fresh_epoch_reason": "initial_window_not_retained",
                "fresh_epoch_warmup_window_id": 99,
                "fresh_epoch_latest_window_id": 100,
                "updated_at": time.time(),
                "source_reset_epoch": fresh_bridge.source_generation,
            }

        first_request = fresh_request("cold-start-worker")
        atomic_write_json(fresh_worker_status_path, first_request)
        assert fresh_bridge.handle_worker_fresh_epoch(fresh_prompt, 400) is True
        assert fresh_bridge.source_generation == generation_before_startup + 1
        assert fresh_bridge.accept_after_seq == 400
        assert fresh_bridge.source_reset_reason == "worker_fresh_epoch"
        assert fresh_bridge.handle_worker_fresh_epoch(fresh_prompt, 400) is False
        assert fresh_bridge.recovery_summary(fresh_prompt)["count"] == recovery_before_startup
        assert not fresh_bridge.recovery_is_capped(fresh_prompt)
        assert (fresh_rendered_dir / "latest.json").read_bytes() == rendered_sentinel
        assert fresh_bridge.last_handled_fresh_epoch_request["latency_ms"] < 1000.0

        # A bridge restart preserves the per-instance dedupe set. Its own
        # bridge_start epoch also satisfies a worker still waiting for a token.
        generation_before_bridge_restart = fresh_bridge.source_generation
        assert fresh_bridge.terminate_window_pipeline("fresh_bridge_restart")
        restarted_bridge = Bridge(fresh_args)
        restarted_bridge.adopt_source(fresh_identity)
        fresh_bridge = restarted_bridge
        fresh_prompt = read_prompt_state(fresh_control_path)
        assert fresh_bridge.source_generation == generation_before_bridge_restart + 1
        assert "cold-start-worker" in fresh_bridge.handled_fresh_epoch_worker_instances
        atomic_write_json(fresh_worker_status_path, first_request)
        assert fresh_bridge.handle_worker_fresh_epoch(fresh_prompt, 401) is False

        # Three consecutive worker process restarts each get one fresh source
        # epoch without consuming the terminal-failure recovery budget.
        restart_generations: list[int] = []
        recovery_before_restarts = fresh_bridge.recovery_summary(fresh_prompt)["count"]
        for restart_index in range(1, 4):
            instance_id = f"hot-restart-worker-{restart_index}"
            request = fresh_request(instance_id)
            atomic_write_json(fresh_worker_status_path, request)
            boundary = 401 + restart_index
            assert fresh_bridge.handle_worker_fresh_epoch(fresh_prompt, boundary) is True
            restart_generations.append(fresh_bridge.source_generation)
            assert fresh_bridge.accept_after_seq == boundary
            assert fresh_bridge.handle_worker_fresh_epoch(fresh_prompt, boundary) is False
        assert restart_generations == sorted(set(restart_generations))
        assert fresh_bridge.fresh_epoch_request_count == 4
        assert fresh_bridge.recovery_summary(fresh_prompt)["count"] == recovery_before_restarts
        assert not fresh_bridge.recovery_is_capped(fresh_prompt)

        # A completion from an older worker/source epoch cannot reset the live stream.
        generation_before_stale_completion = fresh_bridge.source_generation
        stale_completion = {
            **first_request,
            "status": "streaming",
            "backlog_eligible": True,
            "needs_fresh_source": False,
            "updated_at": time.time(),
            "accepted_window_id": 0,
            "completed_window_id": 0,
            "source_reset_epoch": generation_before_startup,
        }
        atomic_write_json(fresh_worker_status_path, stale_completion)
        assert fresh_bridge.handle_worker_fresh_epoch(fresh_prompt, 405) is False
        assert fresh_bridge.handle_worker_recovery(fresh_prompt, 405) is False
        assert fresh_bridge.handle_worker_backlog(fresh_prompt, 405) is False
        assert fresh_bridge.source_generation == generation_before_stale_completion
        assert (fresh_rendered_dir / "latest.json").read_bytes() == rendered_sentinel

        retained_metadata = sorted(out_dir.glob("window_*.npz"))
        retained_sidecars = sorted(out_dir.glob("window_*.gbuffers.npy"))
        assert len(retained_metadata) <= args.retain_windows
        if window_transport == "npy":
            assert len(retained_sidecars) == len(retained_metadata)
            assert {gbuffer_sidecar_path(path).name for path in retained_metadata} == {
                path.name for path in retained_sidecars
            }
            orphan = out_dir / "window_999999999999.gbuffers.npy"
            atomic_save_npy(orphan, np.zeros((1,), dtype=np.uint8))
            bridge.cleanup_windows()
            assert orphan.exists()  # Retention never sweeps uncommitted/orphan files.
            orphan.unlink()
        else:
            assert not retained_sidecars
        for candidate_bridge in (
            bridge,
            lag_bridge,
            no_resync_bridge,
            input_bridge,
            fresh_bridge,
        ):
            assert candidate_bridge.terminate_window_pipeline("self_check_complete")
        print(
            "self-check passed: latent_window=2 cadence=5,8; "
            f"latent_window=4 cadence=13,16; transport={window_transport}; "
            f"normal input keeps history; manual input resync compatibility; "
            f"startup recovery handshake; "
            f"async writer 200ms sampling_fps={sampling_fps:.2f} "
            f"max_gap_ms={max(sampling_gaps) * 1000:.1f} "
            f"max_submit_ms={max(sampling_durations) * 1000:.1f}; "
            "staging fence; atomic current-marker promotion fence; "
            "marker rollback/resolution/identity race; strict manifest authority; "
            "post-replace publish/revoke recovery; epoch revocation; orphan restart isolation; "
            "cleanup/retention chain; fatal lifecycle; "
            "SIGTERM cleanup; post-authority aliases/failures; queue-full reset"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    configure_temporal_cadence(args, parser)
    validate_args(args, parser)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.self_check:
        if args.window_transport == SHM_SLOT_TRANSPORT:
            print(json.dumps(run_shm_slot_self_check(), sort_keys=True))
            return 0
        return run_self_check(args.window_transport)

    bridge = Bridge(args)

    def request_stop(_signum: int, _frame: Any) -> None:
        bridge.stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    return bridge.run()


if __name__ == "__main__":
    raise SystemExit(main())
