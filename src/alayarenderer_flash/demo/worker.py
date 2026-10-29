#!/usr/bin/env python3
"""Resident AlayaRenderer Flash worker for live SuperTuxKart G-buffers.

The bridge publishes immutable, non-overlapping input windows. Pixel cadence is
derived from the configured latent window: 13/16 frames for four latents or 5/8
for two. A resumed small-encoder call prepends the prior boundary frame, so its
actual causal encoder inputs are 17 or 9 frames with latent offset 1. This worker
keeps the Helios autoregressive state, small-encoder phase, tiny-decoder cache,
and RNG resident on the GPU until a generation/prompt/track fence changes. It
consumes only immutable metadata/sidecar pairs; compatibility aliases are never
input authority.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator

os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "yes")
os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np
import torch

from alayarenderer_flash.runtime import (
    CONTROL_COLUMNS,
    DEFAULT_NEGATIVE_PROMPT,
    MODALITIES,
    SUPPORTED_LATENT_WINDOWS,
    ContinuityError,
    FenceChanged,
    InputNotReady,
    ModelRuntime,
    OutputFrame,
    SessionToken,
    SharedSlotReadLease,
    TemporalCadence,
    WindowData,
    WindowHeader,
    cadence_for_latent_window,
    clean_input_event_id,
)
from alayarenderer_flash.transport import (
    TRANSPORT_NAME as SHM_SLOT_TRANSPORT,
)
from alayarenderer_flash.transport import (
    SlotChanged,
    SlotReference,
    snapshot_slot,
    state_matches_reference,
)
from alayarenderer_flash.transport import (
    read_state as read_slot_state,
)
from alayarenderer_flash.transport import (
    run_self_check as run_shm_slot_self_check,
)

SCHEMA_VERSION = 2
DEFAULT_LATENT_WINDOW = 4
WINDOW_TRANSPORTS = ("npz", "npy", SHM_SLOT_TRANSPORT)
DEFAULT_WINDOW_TRANSPORT = "npz"
DEFAULT_FRESH_EPOCH_TIMEOUT = 20.0
MAX_STARTUP_REPLAY_WINDOWS = 2
MAX_STARTUP_UNPUBLISHED_WINDOWS = 1
HEADER_CACHE_LIMIT = 256
PUBLISH_PROGRESS_TIMEOUT_SECONDS = 5.0
PUBLISH_PROGRESS_POLL_SECONDS = 0.25
PUBLISH_PROGRESS_WARNING_SECONDS = 0.25
PUBLISHER_NONEMPTY_WAIT_WARNING_SECONDS = 0.25
CYCLIC_GC_WARNING_SECONDS = 0.25
PUBLISHER_STAGE_NAMES = (
    "loop",
    "select",
    "reacquire_condition",
    "precommit",
    "stage_rgba",
    "stage_rgb_bytes",
    "input_lock_wait",
    "metadata_json",
    "promote",
    "postcommit",
)
WINDOW_RE = re.compile(r"^window_(?:(?P<generation>\d+)_)?(?P<window_id>\d+)\.npz$")
FRESH_SOURCE_STATUSES = frozenset({"fresh_epoch_requested", "waiting_for_fresh_epoch"})


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class CyclicGCTelemetry:
    def __init__(self) -> None:
        self._started_at: dict[int, float] = {}
        self._callback = self._on_collection
        self.steady_state_freeze_attempted = False
        self.steady_state_freeze_applied = False
        self.steady_state_freeze_apply_count = 0
        self.steady_state_frozen_object_count = 0
        self.pre_freeze_collection_seconds = 0.0
        self.steady_state_freeze_applied_at_unix = 0.0
        self.steady_state_freeze_error = ""
        self.reset()

    def reset(self) -> None:
        self._started_at.clear()
        self.last_generation = -1
        self.last_seconds = 0.0
        self.max_seconds = 0.0
        self.collection_count = 0
        self.over_250ms_count = 0

    def install(self) -> None:
        if self._callback not in gc.callbacks:
            gc.callbacks.append(self._callback)

    def apply_steady_state_freeze_once(self) -> bool:
        """Freeze resident warmup objects while leaving new objects collectable."""
        if self.steady_state_freeze_attempted:
            return False
        self.steady_state_freeze_attempted = True
        freeze = getattr(gc, "freeze", None)
        if not callable(freeze):
            self.steady_state_freeze_error = "gc.freeze is unavailable"
            return False

        collect_started_at = time.monotonic()
        try:
            gc.collect()
        except Exception as error:
            self.pre_freeze_collection_seconds = max(0.0, time.monotonic() - collect_started_at)
            self.steady_state_freeze_error = f"gc.collect failed: {error}"[:1000]
            return False
        self.pre_freeze_collection_seconds = max(0.0, time.monotonic() - collect_started_at)

        try:
            freeze()
        except Exception as error:
            self.steady_state_freeze_error = f"gc.freeze failed: {error}"[:1000]
            return False
        self.steady_state_freeze_applied = True
        self.steady_state_freeze_apply_count = 1
        self.steady_state_freeze_applied_at_unix = time.time()
        get_freeze_count = getattr(gc, "get_freeze_count", None)
        if callable(get_freeze_count):
            try:
                self.steady_state_frozen_object_count = max(0, as_int(get_freeze_count(), 0))
            except Exception:
                self.steady_state_frozen_object_count = 0
        return True

    def _on_collection(self, phase: str, info: dict[str, Any]) -> None:
        generation = as_int(info.get("generation"), -1)
        if phase == "start":
            self._started_at[generation] = time.monotonic()
            return
        if phase != "stop":
            return
        started_at = self._started_at.pop(generation, None)
        if started_at is None:
            return
        duration = max(0.0, time.monotonic() - started_at)
        self.last_generation = generation
        self.last_seconds = duration
        self.max_seconds = max(self.max_seconds, duration)
        self.collection_count += 1
        if duration <= CYCLIC_GC_WARNING_SECONDS:
            return
        self.over_250ms_count += 1
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            print(
                f"{timestamp} WARNING slow Python cyclic GC "
                f"generation={generation} duration_ms={duration * 1000.0:.2f} "
                f"collected={as_int(info.get('collected'), 0)} "
                f"uncollectable={as_int(info.get('uncollectable'), 0)}",
                file=sys.stderr,
                flush=True,
            )
        except (OSError, ValueError):
            pass

    def telemetry(self) -> dict[str, Any]:
        return {
            "python_gc_last_generation": self.last_generation,
            "python_gc_collection_ms": round(self.last_seconds * 1000.0, 2),
            "python_gc_collection_max_ms": round(self.max_seconds * 1000.0, 2),
            "python_gc_collection_count": self.collection_count,
            "python_gc_collection_over_250ms_count": self.over_250ms_count,
            "python_gc_freeze_attempted": self.steady_state_freeze_attempted,
            "python_gc_freeze_applied": self.steady_state_freeze_applied,
            "python_gc_freeze_apply_count": self.steady_state_freeze_apply_count,
            "python_gc_freeze_count": self.steady_state_frozen_object_count,
            "python_gc_pre_freeze_collection_ms": round(self.pre_freeze_collection_seconds * 1000.0, 2),
            "python_gc_freeze_applied_at_unix": round(self.steady_state_freeze_applied_at_unix, 6),
            "python_gc_freeze_error": self.steady_state_freeze_error,
        }


class WorkerLoopTelemetry:
    """Session-scoped wall-clock timings around the worker input loop."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.timing_window_id = -1
        self.timing_observed_at_unix = 0.0
        self.previous_enqueue_to_snapshot_seconds = 0.0
        self.previous_enqueue_to_snapshot_max_seconds = 0.0
        self.previous_enqueue_to_snapshot_max_window_id = -1
        self.previous_enqueue_to_snapshot_max_observed_at_unix = 0.0
        self.authority_seconds = 0.0
        self.authority_max_seconds = 0.0
        self.authority_max_window_id = -1
        self.authority_max_observed_at_unix = 0.0
        self.next_window_path_seconds = 0.0
        self.next_window_path_max_seconds = 0.0
        self.next_window_path_max_window_id = -1
        self.next_window_path_max_observed_at_unix = 0.0
        self.load_window_seconds = 0.0
        self.load_window_max_seconds = 0.0
        self.load_window_max_window_id = -1
        self.load_window_max_observed_at_unix = 0.0
        self.pre_model_status_update_seconds = 0.0
        self.pre_model_status_update_max_seconds = 0.0
        self.pre_model_status_update_max_window_id = -1
        self.pre_model_status_update_max_observed_at_unix = 0.0
        self.input_not_ready_count = 0
        self.input_not_ready_wait_total_seconds = 0.0
        self.input_not_ready_max_consecutive_wait_seconds = 0.0
        self.input_not_ready_max_consecutive_wait_window_id = -1
        self.input_not_ready_max_consecutive_wait_observed_at_unix = 0.0
        self._last_enqueue_completed_at: float | None = None
        self._input_not_ready_started_at: float | None = None
        self._authority_max_pending_window = False
        self._next_window_path_max_pending_window = False
        self._load_window_max_pending_window = False
        self._pre_model_status_update_max_pending_window = False

    @staticmethod
    def _duration(duration: float) -> float:
        return max(0.0, duration)

    def record_authority(self, duration: float, observed_at_unix: float) -> None:
        self.authority_seconds = self._duration(duration)
        if self.authority_seconds > self.authority_max_seconds:
            self.authority_max_seconds = self.authority_seconds
            self.authority_max_observed_at_unix = observed_at_unix
            self._authority_max_pending_window = True

    def record_next_window_path(
        self,
        duration: float,
        observed_at_unix: float,
    ) -> None:
        self.next_window_path_seconds = self._duration(duration)
        if self.next_window_path_seconds > self.next_window_path_max_seconds:
            self.next_window_path_max_seconds = self.next_window_path_seconds
            self.next_window_path_max_observed_at_unix = observed_at_unix
            self._next_window_path_max_pending_window = True

    def record_load_window(self, duration: float, observed_at_unix: float) -> None:
        self.load_window_seconds = self._duration(duration)
        if self.load_window_seconds > self.load_window_max_seconds:
            self.load_window_max_seconds = self.load_window_seconds
            self.load_window_max_observed_at_unix = observed_at_unix
            self._load_window_max_pending_window = True

    def record_pre_model_status_update(
        self,
        duration: float,
        observed_at_unix: float,
    ) -> None:
        self.pre_model_status_update_seconds = self._duration(duration)
        if self.pre_model_status_update_seconds > self.pre_model_status_update_max_seconds:
            self.pre_model_status_update_max_seconds = self.pre_model_status_update_seconds
            self.pre_model_status_update_max_observed_at_unix = observed_at_unix
            self._pre_model_status_update_max_pending_window = True

    def record_enqueue_completed(self, completed_at: float) -> None:
        self._last_enqueue_completed_at = completed_at

    def record_snapshot_start(
        self,
        started_at: float,
        window_id: int,
        observed_at_unix: float,
    ) -> None:
        self.timing_window_id = window_id
        self.timing_observed_at_unix = observed_at_unix
        pending_maxima = (
            ("authority_max", self._authority_max_pending_window),
            ("next_window_path_max", self._next_window_path_max_pending_window),
            ("load_window_max", self._load_window_max_pending_window),
            (
                "pre_model_status_update_max",
                self._pre_model_status_update_max_pending_window,
            ),
        )
        for name, pending in pending_maxima:
            if not pending:
                continue
            setattr(self, f"{name}_window_id", window_id)
        self._authority_max_pending_window = False
        self._next_window_path_max_pending_window = False
        self._load_window_max_pending_window = False
        self._pre_model_status_update_max_pending_window = False
        completed_at = self._last_enqueue_completed_at
        if completed_at is None:
            return
        self._last_enqueue_completed_at = None
        duration = self._duration(started_at - completed_at)
        self.previous_enqueue_to_snapshot_seconds = duration
        if duration > self.previous_enqueue_to_snapshot_max_seconds:
            self.previous_enqueue_to_snapshot_max_seconds = duration
            self.previous_enqueue_to_snapshot_max_window_id = window_id
            self.previous_enqueue_to_snapshot_max_observed_at_unix = observed_at_unix

    def record_input_not_ready(self, observed_at: float) -> None:
        self.input_not_ready_count += 1
        if self._input_not_ready_started_at is None:
            self._input_not_ready_started_at = observed_at

    def finish_input_not_ready_wait(
        self,
        observed_at: float,
        observed_at_unix: float,
        window_id: int = -1,
    ) -> None:
        started_at = self._input_not_ready_started_at
        if started_at is None:
            return
        self._input_not_ready_started_at = None
        duration = self._duration(observed_at - started_at)
        self.input_not_ready_wait_total_seconds += duration
        if duration > self.input_not_ready_max_consecutive_wait_seconds:
            self.input_not_ready_max_consecutive_wait_seconds = duration
            self.input_not_ready_max_consecutive_wait_window_id = window_id
            self.input_not_ready_max_consecutive_wait_observed_at_unix = observed_at_unix

    def telemetry(self) -> dict[str, Any]:
        return {
            "worker_loop_timing_window_id": self.timing_window_id,
            "worker_loop_timing_observed_at_unix": round(self.timing_observed_at_unix, 6),
            "previous_enqueue_to_snapshot_ms": round(self.previous_enqueue_to_snapshot_seconds * 1000.0, 2),
            "previous_enqueue_to_snapshot_max_ms": round(self.previous_enqueue_to_snapshot_max_seconds * 1000.0, 2),
            "previous_enqueue_to_snapshot_max_window_id": (self.previous_enqueue_to_snapshot_max_window_id),
            "previous_enqueue_to_snapshot_max_observed_at_unix": round(
                self.previous_enqueue_to_snapshot_max_observed_at_unix, 6
            ),
            "authority_ms": round(self.authority_seconds * 1000.0, 2),
            "authority_max_ms": round(self.authority_max_seconds * 1000.0, 2),
            "authority_max_window_id": self.authority_max_window_id,
            "authority_max_observed_at_unix": round(self.authority_max_observed_at_unix, 6),
            "next_window_path_ms": round(self.next_window_path_seconds * 1000.0, 2),
            "next_window_path_max_ms": round(self.next_window_path_max_seconds * 1000.0, 2),
            "next_window_path_max_window_id": self.next_window_path_max_window_id,
            "next_window_path_max_observed_at_unix": round(self.next_window_path_max_observed_at_unix, 6),
            "load_window_ms": round(self.load_window_seconds * 1000.0, 2),
            "load_window_max_ms": round(self.load_window_max_seconds * 1000.0, 2),
            "load_window_max_window_id": self.load_window_max_window_id,
            "load_window_max_observed_at_unix": round(self.load_window_max_observed_at_unix, 6),
            "pre_model_status_update_ms": round(self.pre_model_status_update_seconds * 1000.0, 2),
            "pre_model_status_update_max_ms": round(self.pre_model_status_update_max_seconds * 1000.0, 2),
            "pre_model_status_update_max_window_id": (self.pre_model_status_update_max_window_id),
            "pre_model_status_update_max_observed_at_unix": round(self.pre_model_status_update_max_observed_at_unix, 6),
            "input_not_ready_count": self.input_not_ready_count,
            "input_not_ready_wait_total_ms": round(self.input_not_ready_wait_total_seconds * 1000.0, 2),
            "input_not_ready_max_consecutive_wait_ms": round(
                self.input_not_ready_max_consecutive_wait_seconds * 1000.0, 2
            ),
            "input_not_ready_max_consecutive_wait_window_id": (self.input_not_ready_max_consecutive_wait_window_id),
            "input_not_ready_max_consecutive_wait_observed_at_unix": round(
                self.input_not_ready_max_consecutive_wait_observed_at_unix, 6
            ),
        }


class OutputPublishFailed(RuntimeError):
    """The bounded output writer exhausted its retries for this session."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", type=Path, default=Path(".runtime/alaya_live_control"))
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--rendered-dir", type=Path, default=Path("/dev/shm/stk_alaya_rendered"))
    parser.add_argument("--live-control", type=Path, default=None)
    parser.add_argument("--worker-status", type=Path, default=None)
    parser.add_argument(
        "--input-state",
        type=Path,
        default=None,
        help=(
            "Optional live input.json used to discard generated frames that predate "
            "a newly accepted player-input boundary"
        ),
    )
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--small-encoder", type=Path, required=True)
    parser.add_argument("--tiny-decoder", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument(
        "--window-transport",
        choices=WINDOW_TRANSPORTS,
        default=DEFAULT_WINDOW_TRANSPORT,
        help=("Window payload: embedded NPZ, immutable mmap NPY, or preallocated double shared-memory slots"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--latent-window",
        type=int,
        choices=SUPPORTED_LATENT_WINDOWS,
        default=DEFAULT_LATENT_WINDOW,
        help="Latent frames per request; 4 uses pixel cadence 13/16, 2 uses 5/8",
    )
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--few-step-shift", type=float, default=3.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--small-encoder-dim", type=int, default=128)
    parser.add_argument(
        "--small-encoder-mode",
        choices=("separate", "grouped", "joint"),
        default="grouped",
    )
    parser.add_argument("--compile-transformer", dest="compile_transformer", action="store_true")
    parser.add_argument("--no-compile-transformer", dest="compile_transformer", action="store_false")
    parser.add_argument("--compile-small-encoder", dest="compile_small_encoder", action="store_true")
    parser.add_argument("--no-compile-small-encoder", dest="compile_small_encoder", action="store_false")
    parser.add_argument("--compile-tiny-decoder", dest="compile_tiny_decoder", action="store_true")
    parser.add_argument("--no-compile-tiny-decoder", dest="compile_tiny_decoder", action="store_false")
    parser.set_defaults(
        compile_transformer=True,
        compile_small_encoder=True,
        compile_tiny_decoder=True,
    )
    parser.add_argument(
        "--compile-mode",
        choices=("eager", "transformer", "all"),
        default=None,
        help="Convenience override for the individual compile flags",
    )
    parser.add_argument("--eager-fallback", dest="eager_fallback", action="store_true")
    parser.add_argument("--no-eager-fallback", dest="eager_fallback", action="store_false")
    parser.set_defaults(eager_fallback=True)
    parser.add_argument("--warmup-chunks", type=int, default=2)
    parser.add_argument(
        "--fresh-epoch-timeout",
        type=float,
        default=DEFAULT_FRESH_EPOCH_TIMEOUT,
        help="Seconds before refreshing an unacknowledged fresh-source status handshake",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--publish-fps", type=float, default=24.0)
    parser.add_argument("--publish-queue-frames", type=int, default=24)
    parser.add_argument("--publish-attempts", type=int, default=3)
    parser.add_argument(
        "--adaptive-publish",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ALAYA_ADAPTIVE_PUBLISH", False),
        help="Pace frames from measured chunk arrivals; disabled by default",
    )
    parser.add_argument(
        "--publish-source-fps",
        type=float,
        default=0.0,
        help="Optional bridge FPS hint used before enough chunk arrivals are observed",
    )
    parser.add_argument(
        "--publish-buffer-frames",
        type=float,
        default=2.5,
        help="Adaptive jitter reserve at each chunk arrival",
    )
    parser.add_argument(
        "--publish-min-fps",
        type=float,
        default=12.0,
        help="Lower bound for adaptive recovery pacing",
    )
    parser.add_argument("--poll-interval", type=float, default=0.10)
    parser.add_argument("--once", action="store_true", help="Exit after one committed input window")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Validate the schema/fence/window reader with synthetic data and exit without loading models",
    )
    args = parser.parse_args(argv)
    args.input_dir = args.input_dir or args.control_dir / "upload"
    args.live_control = args.live_control or args.control_dir / "live_control.json"
    args.worker_status = args.worker_status or args.control_dir / "worker_status.json"
    if args.compile_mode is not None:
        args.compile_transformer = args.compile_mode in {"transformer", "all"}
        args.compile_small_encoder = args.compile_mode == "all"
        args.compile_tiny_decoder = args.compile_mode == "all"
    args.temporal_cadence = cadence_for_latent_window(args.latent_window)
    if not str(args.device).startswith("cuda"):
        parser.error("this worker currently requires a CUDA device, for example --device cuda:0")
    if args.height <= 0 or args.width <= 0 or args.height % 8 or args.width % 8:
        parser.error("height and width must be positive multiples of 8")
    if args.num_inference_steps < 1:
        parser.error("num-inference-steps must be at least 1")
    if args.max_attempts < 1:
        parser.error("max-attempts must be at least 1")
    if args.publish_queue_frames < 1 or args.publish_attempts < 1:
        parser.error("publish-queue-frames and publish-attempts must be at least 1")
    if args.publish_fps < 0 or args.publish_source_fps < 0:
        parser.error("publish-fps and publish-source-fps cannot be negative")
    if args.publish_buffer_frames < 0:
        parser.error("publish-buffer-frames cannot be negative")
    if args.publish_min_fps <= 0:
        parser.error("publish-min-fps must be positive")
    if args.adaptive_publish and args.publish_fps <= 0:
        parser.error("adaptive publishing requires a positive publish-fps cap")
    if args.publish_source_fps > args.publish_fps > 0:
        parser.error("publish-source-fps cannot exceed the publish-fps cap")
    if args.publish_min_fps > args.publish_fps > 0:
        parser.error("publish-min-fps cannot exceed the publish-fps cap")
    minimum_queue_frames = max(
        args.temporal_cadence.initial_pixel_frames,
        args.temporal_cadence.resumed_pixel_frames,
    )
    if args.publish_queue_frames < minimum_queue_frames:
        parser.error(
            f"publish-queue-frames must be at least {minimum_queue_frames} for latent-window {args.latent_window}"
        )
    adaptive_queue_frames = args.temporal_cadence.resumed_pixel_frames + int(np.ceil(args.publish_buffer_frames))
    if args.adaptive_publish and args.publish_queue_frames < adaptive_queue_frames:
        parser.error(
            f"adaptive publishing requires publish-queue-frames >= {adaptive_queue_frames} "
            "to retain one resumed chunk plus its jitter reserve"
        )
    if args.warmup_chunks not in (0, 2):
        parser.error("warmup-chunks must be 0 or 2; two chunks cover prepad and resumed shapes")
    if args.fresh_epoch_timeout <= 0:
        parser.error("fresh-epoch-timeout must be positive")
    return args


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def stage_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def promote_staged_file(temporary: Path, path: Path) -> None:
    os.replace(temporary, path)


def discard_staged_file(temporary: Path | None) -> None:
    if temporary is None:
        return
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = stage_bytes(path, payload)
    try:
        promote_staged_file(temporary, path)
    finally:
        discard_staged_file(temporary)


def json_payload_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_bytes(path, json_payload_bytes(payload))


def input_state_lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


@contextmanager
def input_state_commit_lock(path: Path | None) -> Iterator[None]:
    if path is None:
        yield
        return
    lock_path = input_state_lock_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def clean_prompt(value: Any) -> str:
    return "".join(char for char in str(value or "").strip() if char.isprintable())[:1200]


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def nested_window(manifest: dict[str, Any]) -> dict[str, Any]:
    value = manifest.get("window")
    return value if isinstance(value, dict) else manifest


def manifest_value(manifest: dict[str, Any], *keys: str, default: Any = None) -> Any:
    window = nested_window(manifest)
    for key in keys:
        if key in manifest:
            return manifest[key]
        if key in window:
            return window[key]
    return default


@dataclass(frozen=True)
class Authority:
    live: dict[str, Any]
    manifest: dict[str, Any]
    token: SessionToken | None
    reason: str


@dataclass(frozen=True)
class FreshSourceDecision:
    required: bool
    reason: str
    warmup_window_id: int
    oldest_window_id: int
    latest_window_id: int
    initial_window_id: int
    recovery_target_window_id: int = -1
    recovery_target_last_seq: int = -1
    replay_window_count: int = 0
    unpublished_window_count: int = 0


@dataclass(frozen=True)
class PreviousRecoveryTarget:
    window_id: int
    visible_source_seq: int
    status_window_id: int
    rendered_window_id: int
    previous_worker_instance_id: str


@dataclass
class FreshEpochHandshake:
    token_key: tuple[Any, ...]
    source_reset_epoch: int
    request_id: str
    requested_at: float
    requested_monotonic: float
    next_status_at: float
    decision: FreshSourceDecision
    retry_count: int = 0


def decide_fresh_source(
    warmup_header: WindowHeader,
    compatible: list[WindowHeader],
    recovery_target: PreviousRecoveryTarget | None = None,
) -> FreshSourceDecision:
    if not compatible:
        return FreshSourceDecision(
            required=True,
            reason="warmup_stream_no_longer_retained",
            warmup_window_id=warmup_header.window_id,
            oldest_window_id=-1,
            latest_window_id=-1,
            initial_window_id=-1,
            recovery_target_window_id=(recovery_target.window_id if recovery_target is not None else -1),
        )
    compatible = sorted(compatible, key=lambda header: header.window_id)
    oldest = compatible[0]
    latest = compatible[-1]
    initial = next((header for header in compatible if header.is_initial), None)
    recovery_header: WindowHeader | None = None
    replay_window_count = 0
    unpublished_window_count = 0
    if not warmup_header.is_initial or initial is None:
        reason = "initial_window_not_retained"
    elif recovery_target is not None:
        recovery_header = next(
            (header for header in compatible if header.window_id == recovery_target.window_id),
            None,
        )
        if recovery_header is None:
            reason = "previous_recovery_target_not_retained"
        else:
            replay_headers = [
                header for header in compatible if initial.window_id <= header.window_id <= recovery_header.window_id
            ]
            nearby_headers = [
                header for header in compatible if initial.window_id <= header.window_id <= latest.window_id
            ]
            replay_window_count = len(replay_headers)
            unpublished_window_count = max(0, latest.window_id - recovery_header.window_id)
            chain_is_contiguous = bool(
                nearby_headers
                and nearby_headers[0].is_initial
                and all(
                    current.previous_window_id == previous.window_id and current.previous_last_seq == previous.last_seq
                    for previous, current in zip(nearby_headers, nearby_headers[1:])
                )
            )
            if not chain_is_contiguous:
                reason = "previous_recovery_chain_not_contiguous"
            elif replay_window_count > MAX_STARTUP_REPLAY_WINDOWS:
                reason = "previous_recovery_replay_too_large"
            elif unpublished_window_count > MAX_STARTUP_UNPUBLISHED_WINDOWS:
                reason = "previous_recovery_backlog_too_large"
            elif (
                recovery_target.visible_source_seq >= 0
                and recovery_header.last_seq != recovery_target.visible_source_seq
            ):
                reason = "previous_recovery_visible_source_mismatch"
            else:
                reason = ""
    elif latest.window_id > warmup_header.window_id + 1:
        reason = "warmup_stream_not_adjacent_to_latest"
    else:
        reason = ""
    return FreshSourceDecision(
        required=bool(reason),
        reason=reason,
        warmup_window_id=warmup_header.window_id,
        oldest_window_id=oldest.window_id,
        latest_window_id=latest.window_id,
        initial_window_id=initial.window_id if initial is not None else -1,
        recovery_target_window_id=(recovery_target.window_id if recovery_target is not None else -1),
        recovery_target_last_seq=(recovery_header.last_seq if recovery_header is not None else -1),
        replay_window_count=replay_window_count,
        unpublished_window_count=unpublished_window_count,
    )


class WorkerStatus:
    REQUIRED_FIELDS = (
        "prompt",
        "status",
        "error",
        "ready",
        "compile_mode",
        "reset_epoch",
        "accepted_request_id",
        "completed_request_id",
        "output_fps",
        "worker_instance_id",
        "updated_at",
    )

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self.previous = read_json(path)
        self.instance_id = uuid.uuid4().hex[:16]
        self.payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "prompt": "",
            "status": "starting",
            "error": "",
            "ready": False,
            "compile_mode": "loading",
            "reset_epoch": 0,
            "source_reset_epoch": 0,
            "accepted_request_id": 0,
            "completed_request_id": 0,
            "accepted_window_id": -1,
            "completed_window_id": -1,
            "output_fps": 0.0,
            "output_seq": self._initial_output_seq(),
            "worker_instance_id": self.instance_id,
            "updated_at": time.time(),
        }
        self.write()
        self._heartbeat = threading.Thread(
            target=self._heartbeat_loop,
            name="alaya-worker-heartbeat",
            daemon=True,
        )
        self._heartbeat.start()

    def _initial_output_seq(self) -> int:
        return max(0, as_int(self.previous.get("output_seq"), 0))

    def _heartbeat_loop(self) -> None:
        while True:
            time.sleep(2.0)
            with self._lock:
                self.payload["updated_at"] = time.time()
                self.write()

    def update(self, **values: Any) -> None:
        with self._lock:
            self.payload.update(values)
            self.payload["schema_version"] = SCHEMA_VERSION
            self.payload["worker_instance_id"] = self.instance_id
            self.payload["updated_at"] = time.time()
            self.write()

    def write(self) -> None:
        with self._lock:
            for field in self.REQUIRED_FIELDS:
                self.payload.setdefault(
                    field,
                    "" if field in {"prompt", "status", "error", "compile_mode"} else 0,
                )
            atomic_write_json(self.path, dict(self.payload))

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self.payload.get(key, default)


class InputReader:
    def __init__(
        self,
        input_dir: Path,
        live_control_path: Path,
        height: int,
        width: int,
        cadence: TemporalCadence,
        window_transport: str,
    ):
        self.input_dir = input_dir
        self.live_control_path = live_control_path
        self.manifest_path = input_dir / "manifest.json"
        self.height = height
        self.width = width
        self.cadence = cadence
        self.window_transport = window_transport
        self._header_cache: OrderedDict[str, tuple[tuple[int, int, int, int], WindowHeader]] = OrderedDict()
        self._header_cache_lock = threading.Lock()
        self._header_cache_token_key: tuple[Any, ...] | None = None

    def set_active_token(self, token: SessionToken | None) -> None:
        token_key = token.key() if token is not None else None
        with self._header_cache_lock:
            if token_key == self._header_cache_token_key:
                return
            self._header_cache.clear()
            self._header_cache_token_key = token_key

    def _cached_header(
        self,
        cache_key: str,
        identity: tuple[int, int, int, int],
    ) -> WindowHeader | None:
        with self._header_cache_lock:
            cached = self._header_cache.get(cache_key)
            if cached is None:
                return None
            if cached[0] != identity:
                self._header_cache.pop(cache_key, None)
                return None
            self._header_cache.move_to_end(cache_key)
            return cached[1]

    def _remember_header(
        self,
        cache_key: str,
        identity: tuple[int, int, int, int],
        header: WindowHeader,
    ) -> None:
        with self._header_cache_lock:
            self._header_cache[cache_key] = (identity, header)
            self._header_cache.move_to_end(cache_key)
            while len(self._header_cache) > HEADER_CACHE_LIMIT:
                self._header_cache.popitem(last=False)

    def header_cache_size(self) -> int:
        with self._header_cache_lock:
            return len(self._header_cache)

    def authority(self) -> Authority:
        live = read_json(self.live_control_path)
        manifest = read_json(self.manifest_path)
        prompt = clean_prompt(live.get("prompt"))
        if len(prompt) < 3:
            return Authority(live, manifest, None, "no_prompt")
        if not manifest:
            return Authority(live, manifest, None, "waiting_for_manifest")
        if manifest.get("ready") is False or str(manifest.get("status", "ready")) in {"writing", "resetting"}:
            return Authority(live, manifest, None, "manifest_not_ready")

        manifest_transport = str(manifest_value(manifest, "window_transport", default="npz") or "npz")
        if manifest_transport not in WINDOW_TRANSPORTS:
            return Authority(
                live,
                manifest,
                None,
                f"unsupported_window_transport_{manifest_transport}",
            )
        if manifest_transport != self.window_transport:
            return Authority(
                live,
                manifest,
                None,
                f"window_transport_mismatch_{manifest_transport}_expected_{self.window_transport}",
            )

        input_schema = as_int(manifest_value(manifest, "schema_version"), -1)
        if input_schema != SCHEMA_VERSION:
            return Authority(live, manifest, None, f"unsupported_manifest_schema_{input_schema}")
        live_schema = as_int(live.get("schema_version"), 0)
        if live_schema != SCHEMA_VERSION:
            return Authority(live, manifest, None, f"unsupported_live_control_schema_{live_schema}")

        manifest_cadence = manifest.get("cadence", {})
        if isinstance(manifest_cadence, dict):
            expected_cadence = {
                "latent_window": self.cadence.latent_window,
                "initial_frames": self.cadence.initial_pixel_frames,
                "subsequent_frames": self.cadence.resumed_pixel_frames,
            }
            for field, expected in expected_cadence.items():
                actual = as_int(manifest_cadence.get(field), 0)
                if actual > 0 and actual != expected:
                    return Authority(
                        live,
                        manifest,
                        None,
                        f"temporal_cadence_mismatch_{field}_{actual}_expected_{expected}",
                    )

        reset_epoch = as_int(manifest_value(manifest, "reset_epoch"), -1)
        live_reset_epoch = as_int(live.get("reset_epoch"), -1)
        if reset_epoch < 0 or live_reset_epoch < 0:
            return Authority(live, manifest, None, "missing_reset_epoch")
        if live_reset_epoch != reset_epoch:
            return Authority(live, manifest, None, "waiting_for_reset_generation")
        source_reset_epoch = as_int(
            manifest_value(
                manifest,
                "source_reset_epoch",
                "source_generation",
                default=-1,
            ),
            -1,
        )
        if source_reset_epoch < 0:
            return Authority(live, manifest, None, "manifest_missing_source_reset_epoch")

        request_id = as_int(live.get("render_request_id"), 0)
        prompt_revision = as_int(live.get("prompt_revision"), request_id)
        manifest_request_id = as_int(
            manifest_value(manifest, "render_request_id"),
            -1,
        )
        manifest_prompt_revision = as_int(
            manifest_value(manifest, "prompt_revision"),
            -1,
        )
        prompt_active = manifest_value(manifest, "prompt_active", default=True)
        manifest_prompt = clean_prompt(manifest_value(manifest, "prompt", default=prompt))
        if prompt_active is False:
            return Authority(live, manifest, None, "waiting_for_active_prompt_window")
        if manifest_request_id != request_id:
            return Authority(live, manifest, None, "waiting_for_render_request")
        if manifest_prompt_revision != prompt_revision:
            return Authority(live, manifest, None, "waiting_for_prompt_revision")
        if manifest_prompt and manifest_prompt != prompt:
            return Authority(live, manifest, None, "waiting_for_prompt_text")

        track_id = str(manifest_value(manifest, "track_id", "track", default="") or "")
        live_track = str(live.get("track_id", live.get("track", "")) or "")
        if live_track and track_id and live_track != track_id:
            return Authority(live, manifest, None, "waiting_for_track")
        source_session_id = str(manifest_value(manifest, "source_session_id", "session_id", default="") or "")
        if not source_session_id:
            return Authority(live, manifest, None, "manifest_missing_source_session_id")

        token = SessionToken(
            reset_epoch=reset_epoch,
            source_reset_epoch=source_reset_epoch,
            request_id=request_id,
            prompt_revision=prompt_revision,
            prompt_hash=hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16],
            prompt=prompt,
            track_id=track_id or live_track,
            source_session_id=source_session_id,
        )
        return Authority(live, manifest, token, "ready")

    def assert_fence(self, expected: SessionToken) -> None:
        current = self.authority()
        if current.token is None or current.token.key() != expected.key():
            raise FenceChanged(
                f"session fence changed: expected={expected.key()} "
                f"current={current.token.key() if current.token else current.reason}"
            )

    def _candidate_paths(self, manifest: dict[str, Any]) -> list[Path]:
        paths: dict[str, Path] = {}
        current_name = manifest_value(manifest, "immutable_file")
        if current_name:
            current = self._safe_input_path(str(current_name))
            paths[current.name] = current
        retained = manifest.get("retained_immutable_files", manifest.get("retained", []))
        if isinstance(retained, list):
            for value in retained:
                name = value.get("immutable_file") if isinstance(value, dict) else value
                if name:
                    candidate = self._safe_input_path(str(name))
                    paths[candidate.name] = candidate
        return sorted(paths.values(), key=self._filename_sort_key)

    def _safe_input_path(self, name: str) -> Path:
        if Path(name).name != name or WINDOW_RE.fullmatch(name) is None:
            raise ContinuityError(f"manifest authorizes invalid immutable file: {name}")
        candidate = (self.input_dir / Path(name).name).resolve()
        if candidate.parent != self.input_dir.resolve():
            raise ContinuityError(f"immutable file escapes input directory: {name}")
        return candidate

    def _safe_sidecar_path(self, metadata_path: Path, name: str) -> Path:
        expected_name = f"{metadata_path.stem}.gbuffers.npy"
        if Path(name).name != name or name != expected_name:
            raise ValueError(
                f"window {metadata_path.name} has invalid gbuffers_file {name!r}; expected {expected_name!r}"
            )
        input_root = self.input_dir.resolve()
        candidate = input_root / name
        if candidate.parent != input_root:
            raise ValueError(f"gbuffer sidecar escapes input directory: {name}")
        return candidate

    def _safe_shared_slot_path(self, header: WindowHeader) -> tuple[Path, Path]:
        expected_file = f"gbuffer_slot_{header.gbuffers_slot_index}.u8"
        expected_state = f"gbuffer_slot_{header.gbuffers_slot_index}.json"
        if (
            header.gbuffers_slot_index not in (0, 1)
            or Path(header.gbuffers_file).name != header.gbuffers_file
            or header.gbuffers_file != expected_file
        ):
            raise ValueError(f"window {header.path.name} has invalid shared slot file {header.gbuffers_file!r}")
        if (
            Path(header.gbuffers_slot_state_file).name != header.gbuffers_slot_state_file
            or header.gbuffers_slot_state_file != expected_state
        ):
            raise ValueError(
                f"window {header.path.name} has invalid shared slot state file {header.gbuffers_slot_state_file!r}"
            )
        input_root = self.input_dir.resolve()
        slot_path = input_root / expected_file
        state_path = input_root / expected_state
        if slot_path.parent != input_root or state_path.parent != input_root:
            raise ValueError("shared slot path escapes input directory")
        return slot_path, state_path

    @staticmethod
    def _slot_reference(header: WindowHeader) -> SlotReference:
        return SlotReference(
            slot_index=header.gbuffers_slot_index,
            window_id=header.gbuffers_slot_window_id,
            writer_epoch=header.gbuffers_slot_writer_epoch,
            commit_token=header.gbuffers_slot_commit_token,
            shape=header.gbuffers_shape,
            dtype=header.gbuffers_dtype,
            file_name=header.gbuffers_file,
            state_file_name=header.gbuffers_slot_state_file,
            file_size_bytes=header.gbuffers_file_size_bytes,
            payload_nbytes=header.gbuffers_payload_nbytes,
        )

    def _shared_slot_ready(self, header: WindowHeader) -> bool:
        if header.window_transport != SHM_SLOT_TRANSPORT:
            return True
        try:
            _, state_path = self._safe_shared_slot_path(header)
            return state_matches_reference(read_slot_state(state_path), self._slot_reference(header))
        except (OSError, ValueError, TypeError):
            return False

    @staticmethod
    def _filename_sort_key(path: Path) -> tuple[int, int, str]:
        match = WINDOW_RE.match(path.name)
        if not match:
            return (sys.maxsize, sys.maxsize, path.name)
        generation = as_int(match.group("generation"), 0)
        return (generation, as_int(match.group("window_id"), 0), path.name)

    def next_window_path(self, token: SessionToken, last_window_id: int | None) -> Path:
        manifest = read_json(self.manifest_path)
        if last_window_id is not None:
            expected_window_id = last_window_id + 1
            expected_path = self._retained_path_for_window_id(manifest, expected_window_id)
            if expected_path is not None:
                try:
                    header = self.read_header(expected_path)
                except (FileNotFoundError, OSError, ValueError, KeyError):
                    return self._scan_next_window_path(token, last_window_id, manifest)
                if self._header_matches_token(header, token) and self._shared_slot_ready(header):
                    if header.previous_window_id != last_window_id:
                        raise ContinuityError(
                            f"window {header.window_id} points to "
                            f"{header.previous_window_id}, expected {last_window_id}"
                        )
                    return expected_path
                return self._scan_next_window_path(token, last_window_id, manifest)

            current_name = str(manifest_value(manifest, "immutable_file", default="") or "")
            current_match = WINDOW_RE.match(Path(current_name).name)
            current_window_id = as_int(current_match.group("window_id"), -1) if current_match is not None else -1
            if current_window_id <= last_window_id:
                raise InputNotReady("next immutable window has not been published")

        return self._scan_next_window_path(token, last_window_id, manifest)

    def _retained_path_for_window_id(
        self,
        manifest: dict[str, Any],
        window_id: int,
    ) -> Path | None:
        names: list[Any] = [manifest_value(manifest, "immutable_file")]
        retained = manifest.get("retained_immutable_files", manifest.get("retained", []))
        if isinstance(retained, list):
            names.extend(retained)
        for value in names:
            name = value.get("immutable_file") if isinstance(value, dict) else value
            if not name:
                continue
            match = WINDOW_RE.match(Path(str(name)).name)
            if match is None or as_int(match.group("window_id"), -1) != window_id:
                continue
            return self._safe_input_path(str(name))
        return None

    @staticmethod
    def _header_matches_token(header: WindowHeader, token: SessionToken) -> bool:
        return bool(
            header.reset_epoch == token.reset_epoch
            and header.source_reset_epoch == token.source_reset_epoch
            and header.render_request_id == token.request_id
            and header.prompt_revision == token.prompt_revision
            and (not token.track_id or not header.track_id or token.track_id == header.track_id)
            and (not token.source_session_id or token.source_session_id == header.source_session_id)
        )

    def _scan_next_window_path(
        self,
        token: SessionToken,
        last_window_id: int | None,
        manifest: dict[str, Any] | None = None,
    ) -> Path:
        compatible, current_error = self._compatible_headers(token, manifest)
        if not compatible:
            if current_error:
                raise ContinuityError(f"current immutable window is invalid: {current_error}")
            raise InputNotReady("no immutable window matches the active session")
        if last_window_id is None:
            first = compatible[0]
            if not first.is_initial:
                raise ContinuityError(
                    f"initial window for reset epochs "
                    f"({token.reset_epoch},{token.source_reset_epoch}) is no longer retained; "
                    f"oldest retained window is {first.window_id}"
                )
            return first.path
        for header in compatible:
            if header.window_id <= last_window_id:
                continue
            if header.previous_window_id != last_window_id:
                raise ContinuityError(
                    f"window gap after {last_window_id}: next retained window {header.window_id} "
                    f"points to {header.previous_window_id}"
                )
            return header.path
        raise InputNotReady("next immutable window has not been published")

    def _compatible_headers(
        self,
        token: SessionToken,
        manifest: dict[str, Any] | None = None,
    ) -> tuple[list[WindowHeader], str]:
        compatible: list[WindowHeader] = []
        manifest = manifest if manifest is not None else read_json(self.manifest_path)
        current_name = str(manifest_value(manifest, "immutable_file", default="") or "")
        current_error = ""
        for path in self._candidate_paths(manifest):
            try:
                header = self.read_header(path)
            except (FileNotFoundError, OSError, ValueError, KeyError) as error:
                if path.name == Path(current_name).name:
                    current_error = f"{path.name}: {error}"
                continue
            if header.reset_epoch != token.reset_epoch:
                continue
            if header.source_reset_epoch != token.source_reset_epoch:
                continue
            if header.render_request_id != token.request_id:
                continue
            if header.prompt_revision != token.prompt_revision:
                continue
            if token.track_id and header.track_id and token.track_id != header.track_id:
                continue
            if token.source_session_id and header.source_session_id != token.source_session_id:
                continue
            if not self._shared_slot_ready(header):
                continue
            compatible.append(header)
        compatible.sort(key=lambda item: item.window_id)
        return compatible, current_error

    def warmup_window_path(self, token: SessionToken) -> Path:
        """Return compile material without requiring the initial window to survive."""
        compatible, current_error = self._compatible_headers(token)
        if not compatible:
            if current_error:
                raise ContinuityError(f"current immutable window is invalid: {current_error}")
            raise InputNotReady("no immutable window matches the active session")
        initial = next((header for header in compatible if header.is_initial), None)
        return (initial or compatible[-1]).path

    def fresh_source_decision(
        self,
        token: SessionToken,
        warmup_header: WindowHeader,
        recovery_target: PreviousRecoveryTarget | None = None,
    ) -> FreshSourceDecision:
        compatible, _ = self._compatible_headers(token)
        return decide_fresh_source(warmup_header, compatible, recovery_target)

    def unpublished_window_count(
        self,
        token: SessionToken,
        completed_window_id: int,
    ) -> int:
        compatible, current_error = self._compatible_headers(token)
        if not compatible:
            if current_error:
                raise ContinuityError(f"current immutable window is invalid: {current_error}")
            raise InputNotReady("no immutable window matches the active session")
        return max(0, compatible[-1].window_id - completed_window_id)

    @staticmethod
    def _npz_scalar(data: Any, key: str, default: Any = None) -> Any:
        if key not in data:
            return default
        value = data[key]
        if np.asarray(value).size != 1:
            raise ValueError(f"NPZ key {key!r} must be scalar, got shape {np.asarray(value).shape}")
        return np.asarray(value).reshape(()).item()

    def read_header(self, path: Path) -> WindowHeader:
        before = path.stat()
        cache_key = str(path.resolve())
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        cached = self._cached_header(cache_key, identity)
        if cached is not None:
            return cached
        with np.load(path, allow_pickle=False) as data:
            window_transport = str(self._npz_scalar(data, "window_transport", "npz") or "npz")
            gbuffers_file = str(self._npz_scalar(data, "gbuffers_file", "") or "")
            gbuffers_shape = tuple(
                int(value)
                for value in np.asarray(
                    data["gbuffers_shape"] if "gbuffers_shape" in data else [],
                    dtype=np.int64,
                ).tolist()
            )
            gbuffers_dtype = str(self._npz_scalar(data, "gbuffers_dtype", "") or "")
            gbuffers_payload_nbytes = as_int(self._npz_scalar(data, "gbuffers_payload_nbytes", 0), 0)
            gbuffers_file_size_bytes = as_int(self._npz_scalar(data, "gbuffers_file_size_bytes", 0), 0)
            gbuffers_slot_index = as_int(self._npz_scalar(data, "gbuffers_slot_index", -1), -1)
            gbuffers_slot_window_id = as_int(self._npz_scalar(data, "gbuffers_slot_window_id", -1), -1)
            gbuffers_slot_writer_epoch = as_int(self._npz_scalar(data, "gbuffers_slot_writer_epoch", -1), -1)
            gbuffers_slot_commit_token = str(self._npz_scalar(data, "gbuffers_slot_commit_token", "") or "")
            gbuffers_slot_state_file = str(self._npz_scalar(data, "gbuffers_slot_state_file", "") or "")
            gbuffers_data_offset_bytes = as_int(self._npz_scalar(data, "gbuffers_data_offset_bytes", 0), 0)
            schema = as_int(self._npz_scalar(data, "schema_version", -1), -1)
            reset_epoch = as_int(
                self._npz_scalar(data, "reset_epoch", -1),
                -1,
            )
            source_reset_epoch = as_int(
                self._npz_scalar(
                    data,
                    "source_reset_epoch",
                    self._npz_scalar(data, "source_generation", -1),
                ),
                -1,
            )
            render_request_id = as_int(self._npz_scalar(data, "render_request_id", -1), -1)
            window_id = as_int(self._npz_scalar(data, "window_id", -1), -1)
            previous_window_id = as_int(self._npz_scalar(data, "previous_window_id", -1), -1)
            previous_last_seq = as_int(self._npz_scalar(data, "previous_last_seq", -1), -1)
            is_initial = bool(self._npz_scalar(data, "is_initial", False))
            prompt_revision = as_int(self._npz_scalar(data, "prompt_revision", 0), 0)
            track_id = str(self._npz_scalar(data, "track_id", self._npz_scalar(data, "track", "")) or "")
            source_session_id = str(
                self._npz_scalar(
                    data,
                    "source_session_id",
                    self._npz_scalar(data, "session_id", ""),
                )
                or ""
            )
            sample_indices = np.asarray(data["sample_indices"], dtype=np.int64)
            frame_seqs = np.asarray(data["frame_seqs"], dtype=np.int64)
        if schema != SCHEMA_VERSION:
            raise ValueError(f"window {path.name} has unsupported schema {schema}")
        if window_transport not in WINDOW_TRANSPORTS:
            raise ValueError(f"window {path.name} has unsupported transport {window_transport!r}")
        if window_transport != self.window_transport:
            raise ValueError(f"window {path.name} transport is {window_transport}, expected {self.window_transport}")
        if window_transport == "npy":
            self._safe_sidecar_path(path, gbuffers_file)
            if len(gbuffers_shape) != 5 or any(value <= 0 for value in gbuffers_shape):
                raise ValueError(f"window {path.name} has invalid gbuffers_shape {gbuffers_shape}")
            if gbuffers_dtype != "uint8":
                raise ValueError(f"window {path.name} has invalid gbuffers_dtype {gbuffers_dtype!r}")
            if gbuffers_payload_nbytes <= 0 or gbuffers_file_size_bytes <= 0:
                raise ValueError(f"window {path.name} is missing gbuffer sidecar size metadata")
        if window_transport == SHM_SLOT_TRANSPORT:
            if (
                gbuffers_slot_index not in (0, 1)
                or gbuffers_slot_window_id != window_id
                or gbuffers_slot_writer_epoch < 0
                or not gbuffers_slot_commit_token
                or gbuffers_data_offset_bytes != 0
            ):
                raise ValueError(f"window {path.name} has invalid shared-slot metadata")
            if len(gbuffers_shape) != 5 or any(value <= 0 for value in gbuffers_shape):
                raise ValueError(f"window {path.name} has invalid gbuffers_shape {gbuffers_shape}")
            if gbuffers_dtype != "uint8":
                raise ValueError(f"window {path.name} has invalid gbuffers_dtype {gbuffers_dtype!r}")
            if gbuffers_payload_nbytes <= 0 or gbuffers_file_size_bytes <= 0:
                raise ValueError(f"window {path.name} is missing shared-slot size metadata")
        if reset_epoch < 0 or source_reset_epoch < 0 or window_id < 0:
            raise ValueError(f"window {path.name} is missing reset_epoch/source_reset_epoch/window_id")
        if sample_indices.ndim != 1 or frame_seqs.ndim != 1 or len(sample_indices) != len(frame_seqs):
            raise ValueError(f"window {path.name} has invalid sample/frame index arrays")
        if len(sample_indices) == 0:
            raise ValueError(f"window {path.name} has no frames")
        header = WindowHeader(
            path=path,
            window_transport=window_transport,
            gbuffers_file=gbuffers_file,
            gbuffers_shape=gbuffers_shape,
            gbuffers_dtype=gbuffers_dtype,
            gbuffers_payload_nbytes=gbuffers_payload_nbytes,
            gbuffers_file_size_bytes=gbuffers_file_size_bytes,
            schema_version=schema,
            reset_epoch=reset_epoch,
            source_reset_epoch=source_reset_epoch,
            render_request_id=render_request_id,
            window_id=window_id,
            previous_window_id=previous_window_id,
            previous_last_seq=previous_last_seq,
            is_initial=is_initial,
            sample_start=int(sample_indices[0]),
            sample_end=int(sample_indices[-1]),
            frame_count=len(sample_indices),
            first_seq=int(frame_seqs[0]),
            last_seq=int(frame_seqs[-1]),
            prompt_revision=prompt_revision,
            track_id=track_id,
            source_session_id=source_session_id,
            gbuffers_slot_index=gbuffers_slot_index,
            gbuffers_slot_window_id=gbuffers_slot_window_id,
            gbuffers_slot_writer_epoch=gbuffers_slot_writer_epoch,
            gbuffers_slot_commit_token=gbuffers_slot_commit_token,
            gbuffers_slot_state_file=gbuffers_slot_state_file,
            gbuffers_data_offset_bytes=gbuffers_data_offset_bytes,
        )
        if window_transport == SHM_SLOT_TRANSPORT:
            self._safe_shared_slot_path(header)
        after = path.stat()
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if after_identity != identity:
            raise OSError(f"immutable window changed while reading header: {path.name}")
        self._remember_header(cache_key, identity, header)
        return header

    @staticmethod
    def _sidecar_stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
        # Hard-link compatibility aliases change ctime without changing payload bytes.
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
        )

    def _map_gbuffer_sidecar_fd(
        self,
        fd: int,
        path: Path,
        header: WindowHeader,
    ) -> np.memmap:
        before = os.fstat(fd)
        identity = self._sidecar_stat_identity(before)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"gbuffer sidecar {path.name} is not a regular file")
        if before.st_size != header.gbuffers_file_size_bytes:
            raise ValueError(
                f"gbuffer sidecar {path.name} has {before.st_size} bytes, expected {header.gbuffers_file_size_bytes}"
            )

        with os.fdopen(os.dup(fd), "rb") as handle:
            handle.seek(0)
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"gbuffer sidecar {path.name} uses unsupported NPY version {version}")
            data_offset = handle.tell()

        after_header = os.fstat(fd)
        if self._sidecar_stat_identity(after_header) != identity:
            raise OSError(f"gbuffer sidecar changed while reading header: {path.name}")
        if fortran_order:
            raise ValueError(f"gbuffer sidecar {path.name} must use C order")
        if tuple(shape) != header.gbuffers_shape:
            raise ValueError(f"gbuffer sidecar {path.name} shape is {shape}, expected {header.gbuffers_shape}")
        dtype = np.dtype(dtype)
        if dtype.hasobject or str(dtype) != header.gbuffers_dtype:
            raise ValueError(f"gbuffer sidecar {path.name} dtype is {dtype}, expected {header.gbuffers_dtype}")
        payload_nbytes = math.prod(shape) * dtype.itemsize
        if payload_nbytes != header.gbuffers_payload_nbytes:
            raise ValueError(
                f"gbuffer sidecar {path.name} payload is {payload_nbytes} bytes, "
                f"expected {header.gbuffers_payload_nbytes}"
            )
        if data_offset + payload_nbytes != before.st_size:
            raise ValueError(
                f"gbuffer sidecar {path.name} file layout is "
                f"{data_offset}+{payload_nbytes} bytes, expected {before.st_size}"
            )

        with os.fdopen(os.dup(fd), "rb") as handle:
            gbuffers = np.memmap(
                handle,
                dtype=dtype,
                mode="c",
                offset=data_offset,
                shape=shape,
                order="C",
            )
        after_map = os.fstat(fd)
        if self._sidecar_stat_identity(after_map) != identity:
            raise OSError(f"gbuffer sidecar changed while mapping: {path.name}")
        if not gbuffers.flags.c_contiguous:
            raise ValueError(f"gbuffer sidecar {path.name} is not C-contiguous")
        return gbuffers

    def _load_gbuffer_sidecar(self, header: WindowHeader) -> np.ndarray:
        path = self._safe_sidecar_path(header.path, header.gbuffers_file)
        if not hasattr(os, "O_NOFOLLOW"):
            raise OSError("O_NOFOLLOW is required for NPY sidecar transport")
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(path, flags)
        try:
            return self._map_gbuffer_sidecar_fd(fd, path, header)
        finally:
            os.close(fd)

    def _load_shared_slot(self, header: WindowHeader) -> np.ndarray:
        self._safe_shared_slot_path(header)
        try:
            return snapshot_slot(self.input_dir.resolve(), self._slot_reference(header))
        except SlotChanged as exc:
            raise InputNotReady(str(exc)) from exc

    def _open_shared_slot_lease(self, header: WindowHeader) -> SharedSlotReadLease:
        slot_path, state_path = self._safe_shared_slot_path(header)
        try:
            return SharedSlotReadLease.open(
                self._slot_reference(header),
                slot_path,
                state_path,
            )
        except SlotChanged as exc:
            raise InputNotReady(str(exc)) from exc

    def load_window(
        self,
        path: Path,
        token: SessionToken,
        direct_shared_slot: bool = False,
    ) -> WindowData:
        shared_slot_lease: SharedSlotReadLease | None = None
        with np.load(path, allow_pickle=False) as data:
            header = self.read_header(path)
            latent_window = as_int(
                self._npz_scalar(data, "latent_window", self.cadence.latent_window),
                self.cadence.latent_window,
            )
            initial_pixel_frames = as_int(
                self._npz_scalar(
                    data,
                    "initial_pixel_frames",
                    self.cadence.initial_pixel_frames,
                ),
                self.cadence.initial_pixel_frames,
            )
            resumed_pixel_frames = as_int(
                self._npz_scalar(
                    data,
                    "resumed_pixel_frames",
                    self.cadence.resumed_pixel_frames,
                ),
                self.cadence.resumed_pixel_frames,
            )
            modalities = tuple(str(value) for value in np.asarray(data["modalities"]).tolist())
            metadata_transport = str(self._npz_scalar(data, "window_transport", "npz") or "npz")
            if metadata_transport != header.window_transport:
                raise ValueError(
                    f"window transport changed while loading {path.name}: "
                    f"{metadata_transport} != {header.window_transport}"
                )
            if header.window_transport == "npz":
                if "gbuffers" not in data:
                    raise ValueError(f"NPZ window {path.name} has no embedded gbuffers")
                gbuffers = np.asarray(data["gbuffers"])
            else:
                if "gbuffers" in data:
                    raise ValueError(f"NPY metadata window {path.name} unexpectedly embeds gbuffers")
                gbuffers = None
            frame_seqs = np.asarray(data["frame_seqs"], dtype=np.int64)
            input_event_ids = np.asarray(
                data["frame_input_event_ids"]
                if "frame_input_event_ids" in data
                else np.full(frame_seqs.shape, "", dtype=np.str_),
                dtype=np.str_,
            )
            source_frames = np.asarray(data["source_frames"], dtype=np.int64)
            sample_indices = np.asarray(data["sample_indices"], dtype=np.int64)
            controls = np.asarray(data["controls"], dtype=np.float32)
            kart_ids = np.asarray(data["kart_ids"], dtype=np.int32)
            prompt = clean_prompt(self._npz_scalar(data, "prompt", token.prompt))

        if gbuffers is None:
            if header.window_transport == SHM_SLOT_TRANSPORT:
                if direct_shared_slot:
                    shared_slot_lease = self._open_shared_slot_lease(header)
                    gbuffers = shared_slot_lease.array
                else:
                    gbuffers = self._load_shared_slot(header)
            else:
                gbuffers = self._load_gbuffer_sidecar(header)

        if (
            header.reset_epoch != token.reset_epoch
            or header.source_reset_epoch != token.source_reset_epoch
            or header.render_request_id != token.request_id
            or header.prompt_revision != token.prompt_revision
        ):
            raise FenceChanged(
                "window fence mismatch: "
                f"reset_epoch={header.reset_epoch}, "
                f"source_reset_epoch={header.source_reset_epoch}, "
                f"render_request_id={header.render_request_id}, "
                f"prompt_revision={header.prompt_revision}"
            )
        if token.track_id and header.track_id and header.track_id != token.track_id:
            raise FenceChanged(f"window track changed to {header.track_id}")
        if token.source_session_id and header.source_session_id != token.source_session_id:
            raise FenceChanged(f"window source session changed to {header.source_session_id}")
        if prompt and prompt != token.prompt:
            raise FenceChanged("window prompt text does not match live_control.json")
        window_cadence = TemporalCadence(
            latent_window=latent_window,
            initial_pixel_frames=initial_pixel_frames,
            resumed_pixel_frames=resumed_pixel_frames,
        )
        if window_cadence != self.cadence:
            raise ValueError(f"window temporal cadence is {window_cadence}, expected {self.cadence}")
        if modalities != MODALITIES:
            raise ValueError(f"window modalities are {modalities}, expected {MODALITIES}")
        expected_shape = (header.frame_count, len(MODALITIES), self.height, self.width, 3)
        if header.gbuffers_shape and header.gbuffers_shape != expected_shape:
            raise ValueError(f"window gbuffer metadata shape is {header.gbuffers_shape}, expected {expected_shape}")
        if header.gbuffers_dtype and header.gbuffers_dtype != "uint8":
            raise ValueError(f"window gbuffer metadata dtype is {header.gbuffers_dtype}, expected uint8")
        if gbuffers.dtype != np.uint8 or gbuffers.shape != expected_shape:
            raise ValueError(f"gbuffers must be uint8 {expected_shape}, got {gbuffers.dtype} {gbuffers.shape}")
        if header.gbuffers_payload_nbytes and gbuffers.nbytes != header.gbuffers_payload_nbytes:
            raise ValueError(
                f"window gbuffer payload is {gbuffers.nbytes} bytes, expected {header.gbuffers_payload_nbytes}"
            )
        for name, value in (
            ("frame_seqs", frame_seqs),
            ("frame_input_event_ids", input_event_ids),
            ("source_frames", source_frames),
            ("sample_indices", sample_indices),
            ("kart_ids", kart_ids),
        ):
            if value.shape != (header.frame_count,):
                raise ValueError(f"{name} must have shape ({header.frame_count},), got {value.shape}")
        if controls.shape != (header.frame_count, len(CONTROL_COLUMNS)):
            raise ValueError(
                f"controls must have shape ({header.frame_count},{len(CONTROL_COLUMNS)}), got {controls.shape}"
            )
        if any(len(str(value)) > 160 for value in input_event_ids.tolist()):
            raise ValueError("frame_input_event_ids entries must be at most 160 characters")
        if not np.all(np.diff(frame_seqs) > 0):
            raise ValueError("frame_seqs must be strictly increasing")
        if not np.all(np.diff(source_frames) > 0):
            raise ValueError("source_frames must be strictly increasing")
        if not np.array_equal(sample_indices, np.arange(header.sample_start, header.sample_end + 1)):
            raise ValueError("sample_indices must be contiguous")
        expected_frames = self.cadence.initial_pixel_frames if header.is_initial else self.cadence.resumed_pixel_frames
        if header.frame_count != expected_frames:
            raise ValueError(
                f"{'initial' if header.is_initial else 'resumed'} window must contain "
                f"{expected_frames} frames, got {header.frame_count}"
            )
        if header.is_initial:
            if header.sample_start != 0 or header.previous_window_id != -1 or header.previous_last_seq != -1:
                raise ValueError("initial window must start at sample 0 with no predecessor")
        return WindowData(
            header=header,
            gbuffers=gbuffers,
            frame_seqs=frame_seqs,
            input_event_ids=input_event_ids,
            source_frames=source_frames,
            sample_indices=sample_indices,
            controls=np.nan_to_num(controls, nan=0.0, posinf=0.0, neginf=0.0),
            kart_ids=kart_ids,
            prompt=prompt or token.prompt,
            shared_slot_lease=shared_slot_lease,
        )


@dataclass(frozen=True)
class PublishItem:
    token: SessionToken
    header: WindowHeader
    frame: OutputFrame


@dataclass(frozen=True)
class CompletionStatusUpdate:
    token_key: tuple[Any, ...]
    window_id: int
    values: dict[str, Any]


class AdaptivePublishPacer:
    """Arrival-locked pacing with a small chunk-boundary jitter reserve."""

    ARRIVAL_ALPHA = 0.20
    JITTER_GUARD = 0.50
    LOW_BUFFER_FPS = 25.0
    RECOVERY_MARGIN_FRAMES = 4.0
    RAMP_UP_FPS_PER_CHUNK = 0.5
    RAMP_DOWN_FPS_PER_CHUNK = 2.0

    def __init__(
        self,
        fps_cap: float,
        min_fps: float,
        source_fps_hint: float,
        target_buffer_frames: float,
        steady_batch_frames: int,
    ):
        self.fps_cap = fps_cap
        self.min_fps = min(min_fps, fps_cap)
        self.source_fps_hint = source_fps_hint
        self.target_buffer_frames = target_buffer_frames
        self.steady_batch_frames = max(1, steady_batch_frames)
        hinted_fps = source_fps_hint if source_fps_hint > 0 else fps_cap
        self.nominal_fps = min(fps_cap, max(self.min_fps, self.LOW_BUFFER_FPS, hinted_fps))
        self.reset()

    def reset(self) -> None:
        initial_fps = self.source_fps_hint or self.fps_cap
        self.arrival_period = 1.0 / initial_fps
        self.arrival_jitter = 0.0
        self.arrival_samples = 0
        self.last_enqueue_at: float | None = None
        self.buffer_at_arrival = 0
        self.target_fps = min(initial_fps, self.fps_cap)
        self.pacing_state = "steady"
        self.startup_not_before: float | None = None
        self.startup_hold_seconds = 0.0
        self.next_publish_at: float | None = None
        self.last_publish_started_at: float | None = None
        self.publish_count = 0
        self.enqueue_count = 0
        self.underflow_started_at: float | None = None
        self.underflow_count = 0
        self.last_underflow_seconds = 0.0
        self.max_underflow_seconds = 0.0
        self.dropped_frames = 0
        self.commit_duration_ema = 0.0
        self.commit_duration_jitter = 0.0
        self.last_commit_duration = 0.0
        self.max_commit_duration = 0.0
        self.completion_correction = 0.0
        self.completion_correction_carry = 0.0

    def _sustainable_fps(self) -> float:
        guarded_period = self.arrival_period + self.JITTER_GUARD * self.arrival_jitter
        measured_fps = 1.0 / max(guarded_period, 1e-6)
        return min(self.fps_cap, measured_fps)

    def observe_enqueue(self, frame_count: int, queued_frames: int, now: float) -> None:
        if frame_count <= 0:
            return
        if self.underflow_started_at is not None:
            duration = max(0.0, now - self.underflow_started_at)
            self.last_underflow_seconds = duration
            self.max_underflow_seconds = max(self.max_underflow_seconds, duration)
            self.underflow_started_at = None

        if self.last_enqueue_at is not None:
            elapsed = now - self.last_enqueue_at
            if elapsed > 0:
                sample_period = elapsed / frame_count
                minimum_period = 1.0 / (self.fps_cap * 1.5)
                maximum_period = 1.0 / max(self.min_fps * 0.5, 1e-6)
                sample_period = min(max(sample_period, minimum_period), maximum_period)
                deviation = abs(sample_period - self.arrival_period)
                alpha = self.ARRIVAL_ALPHA
                self.arrival_period = (1.0 - alpha) * self.arrival_period + alpha * sample_period
                self.arrival_jitter = (1.0 - alpha) * self.arrival_jitter + alpha * deviation
                self.arrival_samples += 1

        self.last_enqueue_at = now
        self.buffer_at_arrival = max(0, queued_frames)
        sustainable_fps = self._sustainable_fps()
        if self.enqueue_count == 0:
            startup_frames = max(0, self.steady_batch_frames - frame_count) + self.target_buffer_frames
            self.startup_hold_seconds = startup_frames / max(sustainable_fps, 1e-6)
            self.startup_not_before = now + self.startup_hold_seconds
            self.target_fps = sustainable_fps
            self.pacing_state = "startup"
        else:
            low_water = self.target_buffer_frames
            high_water = low_water + self.RECOVERY_MARGIN_FRAMES
            low_buffer_fps = min(
                self.nominal_fps,
                max(self.min_fps, self.LOW_BUFFER_FPS),
            )
            if self.buffer_at_arrival <= low_water:
                desired_fps = min(sustainable_fps, low_buffer_fps)
                self.pacing_state = "protecting"
            elif self.buffer_at_arrival >= high_water:
                # Use the existing output ceiling to drain accumulated delay.
                # Returning only to the source rate cannot remove old backlog.
                desired_fps = self.fps_cap
                self.pacing_state = "recovering"
            else:
                desired_fps = min(self.nominal_fps, sustainable_fps)
                self.pacing_state = "steady"
            desired_fps = max(self.min_fps, min(self.fps_cap, desired_fps))
            if desired_fps >= self.target_fps:
                self.target_fps = min(
                    desired_fps,
                    self.target_fps + self.RAMP_UP_FPS_PER_CHUNK,
                )
            else:
                self.target_fps = max(
                    desired_fps,
                    self.target_fps - self.RAMP_DOWN_FPS_PER_CHUNK,
                )
        self.enqueue_count += 1

    def note_dropped(self, frame_count: int) -> None:
        self.dropped_frames += max(0, frame_count)

    def release_boundary(self, now: float) -> None:
        self.startup_not_before = None
        self.next_publish_at = now
        if self.underflow_started_at is not None:
            duration = max(0.0, now - self.underflow_started_at)
            self.last_underflow_seconds = duration
            self.max_underflow_seconds = max(self.max_underflow_seconds, duration)
            self.underflow_started_at = None

    def wait_delay(self, now: float) -> float:
        deadline = self.next_publish_at
        if self.publish_count == 0 and self.startup_not_before is not None:
            deadline = max(deadline or self.startup_not_before, self.startup_not_before)
        return max(0.0, (deadline or now) - now)

    def empty_wait_timeout(self, now: float) -> float | None:
        delay = self.wait_delay(now)
        if delay > 0:
            return delay
        return None

    def begin_underflow(self, now: float) -> None:
        if self.publish_count > 0 and self.underflow_started_at is None:
            self.underflow_started_at = now
            self.underflow_count += 1

    def note_publish_started(self, now: float) -> None:
        interval = 1.0 / max(self.target_fps, 1e-6)
        minimum_interval = 1.0 / self.fps_cap
        if self.next_publish_at is None:
            self.next_publish_at = now + interval
        else:
            self.next_publish_at = max(
                self.next_publish_at + interval,
                now + minimum_interval,
            )
        self.last_publish_started_at = now

    def note_published(self, started_at: float, now: float) -> None:
        duration = max(0.0, now - started_at)
        if self.publish_count == 0:
            self.commit_duration_ema = duration
        else:
            alpha = 0.10
            deviation = abs(duration - self.commit_duration_ema)
            self.commit_duration_ema = (1.0 - alpha) * self.commit_duration_ema + alpha * duration
            self.commit_duration_jitter = (1.0 - alpha) * self.commit_duration_jitter + alpha * deviation
        self.last_commit_duration = duration
        self.max_commit_duration = max(self.max_commit_duration, duration)
        self.completion_correction = duration - self.commit_duration_ema
        self.completion_correction_carry += self.completion_correction
        if self.next_publish_at is not None:
            scheduled_at = self.next_publish_at
            self.next_publish_at = max(
                started_at + 1.0 / self.fps_cap,
                scheduled_at + self.completion_correction_carry,
            )
            applied_correction = self.next_publish_at - scheduled_at
            # A negative residual can be clipped by the FPS cap. Preserve that
            # credit so it cancels later positive residuals instead of allowing
            # commit jitter to push every future deadline in one direction.
            self.completion_correction_carry -= applied_correction
        self.publish_count += 1

    def telemetry(self, queue_depth: int) -> dict[str, Any]:
        sustainable_fps = self._sustainable_fps()
        return {
            "publisher_mode": "adaptive",
            "publisher_queue_frames": queue_depth,
            "publisher_arrival_fps": round(1.0 / self.arrival_period, 3),
            "publisher_arrival_samples": self.arrival_samples,
            "publisher_sustainable_fps": round(sustainable_fps, 3),
            "publisher_target_fps": round(self.target_fps, 3),
            "publisher_nominal_fps": round(self.nominal_fps, 3),
            "publisher_pacing_state": self.pacing_state,
            "publisher_low_water_frames": self.target_buffer_frames,
            "publisher_high_water_frames": (self.target_buffer_frames + self.RECOVERY_MARGIN_FRAMES),
            "publisher_buffer_at_arrival": self.buffer_at_arrival,
            "publisher_target_buffer_frames": self.target_buffer_frames,
            "publisher_buffer_latency_ms": round(1000.0 * self.target_buffer_frames / max(sustainable_fps, 1e-6), 1),
            "publisher_startup_hold_ms": round(self.startup_hold_seconds * 1000.0, 1),
            "publisher_underflow_count": self.underflow_count,
            "publisher_last_underflow_ms": round(self.last_underflow_seconds * 1000.0, 1),
            "publisher_max_underflow_ms": round(self.max_underflow_seconds * 1000.0, 1),
            "publisher_dropped_frames": self.dropped_frames,
            "publisher_commit_ms": round(self.last_commit_duration * 1000.0, 2),
            "publisher_commit_ema_ms": round(self.commit_duration_ema * 1000.0, 2),
            "publisher_commit_jitter_ms": round(self.commit_duration_jitter * 1000.0, 2),
            "publisher_commit_max_ms": round(self.max_commit_duration * 1000.0, 2),
            "publisher_completion_correction_ms": round(self.completion_correction * 1000.0, 2),
            "publisher_completion_correction_carry_ms": round(self.completion_correction_carry * 1000.0, 2),
        }


class FramePublisher:
    """Paced latest-frame writer that never blocks the GPU inference loop."""

    def __init__(
        self,
        rendered_dir: Path,
        publish_fps: float,
        max_frames: int,
        publish_attempts: int,
        status: WorkerStatus,
        reader: InputReader,
        adaptive: bool = False,
        steady_batch_frames: int = 16,
        source_fps_hint: float = 0.0,
        target_buffer_frames: float = 2.5,
        min_fps: float = 12.0,
        input_state_path: Path | None = None,
        progress_timeout_seconds: float = PUBLISH_PROGRESS_TIMEOUT_SECONDS,
        watchdog_exit: Callable[[int], Any] = os._exit,
        gc_telemetry: CyclicGCTelemetry | None = None,
    ):
        self.rendered_dir = rendered_dir
        self.publish_fps = max(0.0, publish_fps)
        self.max_frames = max(1, max_frames)
        self.publish_attempts = max(1, publish_attempts)
        self.status = status
        self.reader = reader
        self.input_state_path = input_state_path
        self.queue: deque[PublishItem] = deque()
        self.condition = threading.Condition()
        latest = read_json(rendered_dir / "latest.json")
        self.output_seq = max(as_int(status.get("output_seq"), 0), as_int(latest.get("seq"), 0))
        self.frame_index = 0
        self.active_key: tuple[Any, ...] | None = None
        self.failed_key: tuple[Any, ...] | None = None
        self.completed_window: dict[tuple[Any, ...], int] = {}
        self.status_written_window: dict[tuple[Any, ...], int] = {}
        self.publish_times: deque[float] = deque(maxlen=120)
        self.dropped_frames = 0
        self.input_state_mtime_ns = -1
        self.input_event_id = ""
        self.input_boundary_seq = -1
        self.active_input_event_id = ""
        self.active_input_boundary_seq = -1
        self.pending_input_event_id = ""
        self.pending_input_boundary_seq = -1
        self.input_fast_forward_count = 0
        self.input_dropped_frames = 0
        self.publish_inflight_window_id = -1
        self.publish_inflight_source_seq = -1
        self.publish_inflight_input_event_id = ""
        self.progress_generation = 0
        self.progress_timeout_seconds = max(0.1, progress_timeout_seconds)
        self.watchdog_exit = watchdog_exit
        self.gc_telemetry = gc_telemetry
        self.publisher_stage_lock = threading.Lock()
        self._reset_publisher_stage_telemetry()
        self._reset_nonempty_wait_telemetry_locked()
        self.status_write_lock = threading.Lock()
        self.pending_completion_status: CompletionStatusUpdate | None = None
        self.status_inflight_window_id = -1
        self.status_coalesced_updates = 0
        self.status_write_errors = 0
        self.status_last_write_seconds = 0.0
        self.status_max_write_seconds = 0.0
        self.status_last_error = ""
        self.thread_error = ""
        self.thread_failed_at = 0.0
        self.pacer = (
            AdaptivePublishPacer(
                self.publish_fps,
                min_fps,
                source_fps_hint,
                target_buffer_frames,
                steady_batch_frames,
            )
            if adaptive
            else None
        )
        self.status_writer_thread = threading.Thread(
            target=self._status_writer_run,
            name="alaya-status-writer",
            daemon=True,
        )
        self.status_writer_thread.start()
        self.thread = threading.Thread(
            target=self._run,
            name="alaya-frame-publisher",
            daemon=True,
        )
        self.watchdog_thread = threading.Thread(
            target=self._progress_watchdog_run,
            name="alaya-publisher-watchdog",
            daemon=True,
        )
        self.thread.start()
        self.watchdog_thread.start()

    def reset(self, token: SessionToken | None) -> None:
        with self.status_write_lock:
            with self.condition:
                self.queue.clear()
                self.frame_index = 0
                self.active_key = token.key() if token is not None else None
                self.failed_key = self.active_key if self.thread_error and token is not None else None
                self.completed_window.clear()
                self.status_written_window.clear()
                self.pending_completion_status = None
                self.status_inflight_window_id = -1
                self.status_coalesced_updates = 0
                self.status_write_errors = 0
                self.status_last_write_seconds = 0.0
                self.status_max_write_seconds = 0.0
                self.status_last_error = ""
                self.publish_times.clear()
                self.dropped_frames = 0
                self.input_state_mtime_ns = -1
                self.input_event_id = ""
                self.input_boundary_seq = -1
                self.active_input_event_id = ""
                self.active_input_boundary_seq = -1
                self.pending_input_event_id = ""
                self.pending_input_boundary_seq = -1
                self.input_fast_forward_count = 0
                self.input_dropped_frames = 0
                self.publish_inflight_window_id = -1
                self.publish_inflight_source_seq = -1
                self.publish_inflight_input_event_id = ""
                self._reset_publisher_stage_telemetry()
                self._reset_nonempty_wait_telemetry_locked()
                self.progress_generation += 1
                if self.pacer is not None:
                    self.pacer.reset()
                self.condition.notify_all()

    def _reset_nonempty_wait_telemetry_locked(self) -> None:
        self.nonempty_wait_delay_seconds = 0.0
        self.nonempty_wait_delay_max_seconds = 0.0
        self.nonempty_wait_elapsed_seconds = 0.0
        self.nonempty_wait_elapsed_max_seconds = 0.0
        self.nonempty_wait_over_250ms_count = 0

    def _reset_publisher_stage_telemetry(self) -> None:
        started_monotonic = time.monotonic()
        started_unix = time.time()
        with self.publisher_stage_lock:
            self.publisher_stage = "reset"
            self.publisher_stage_started_monotonic = started_monotonic
            self.publisher_stage_started_unix = started_unix
            self.publisher_stage_window_id = -1
            self.publisher_stage_source_seq = -1
            self.publisher_stage_last_seconds = {name: 0.0 for name in PUBLISHER_STAGE_NAMES}
            self.publisher_stage_max_seconds = {name: 0.0 for name in PUBLISHER_STAGE_NAMES}
        self.publisher_progress_warning_active = False
        self.publisher_progress_warning_count = 0
        self.publisher_progress_warning_last_seconds = 0.0
        self.publisher_progress_warning_last_at_unix = 0.0

    def _start_publisher_stage(
        self,
        stage: str,
        item: PublishItem | None = None,
    ) -> None:
        started_monotonic = time.monotonic()
        started_unix = time.time()
        window_id = item.header.window_id if item is not None else -1
        source_seq = item.frame.source_seq if item is not None else -1
        with self.publisher_stage_lock:
            if self.publisher_stage == stage:
                self.publisher_stage_window_id = window_id
                self.publisher_stage_source_seq = source_seq
                return
            previous_stage = self.publisher_stage
            previous_started = self.publisher_stage_started_monotonic
            if previous_stage in self.publisher_stage_last_seconds:
                duration = max(0.0, started_monotonic - previous_started)
                self.publisher_stage_last_seconds[previous_stage] = duration
                self.publisher_stage_max_seconds[previous_stage] = max(
                    self.publisher_stage_max_seconds[previous_stage],
                    duration,
                )
            self.publisher_stage = stage
            self.publisher_stage_started_monotonic = started_monotonic
            self.publisher_stage_started_unix = started_unix
            self.publisher_stage_window_id = window_id
            self.publisher_stage_source_seq = source_seq

    def _retarget_publisher_stage(self, item: PublishItem | None) -> None:
        window_id = item.header.window_id if item is not None else -1
        source_seq = item.frame.source_seq if item is not None else -1
        with self.publisher_stage_lock:
            self.publisher_stage_window_id = window_id
            self.publisher_stage_source_seq = source_seq

    def _publisher_stage_telemetry(
        self,
        now: float | None = None,
    ) -> dict[str, Any]:
        observed_at = time.monotonic() if now is None else now
        with self.publisher_stage_lock:
            values: dict[str, Any] = {
                "publisher_stage": self.publisher_stage,
                "publisher_stage_started_monotonic": round(self.publisher_stage_started_monotonic, 6),
                "publisher_stage_started_unix": round(self.publisher_stage_started_unix, 6),
                "publisher_stage_age_ms": round(
                    max(
                        0.0,
                        observed_at - self.publisher_stage_started_monotonic,
                    )
                    * 1000.0,
                    2,
                ),
                "publisher_stage_window_id": self.publisher_stage_window_id,
                "publisher_stage_source_seq": self.publisher_stage_source_seq,
            }
            for name in PUBLISHER_STAGE_NAMES:
                values[f"publisher_stage_{name}_last_ms"] = round(self.publisher_stage_last_seconds[name] * 1000.0, 2)
                values[f"publisher_stage_{name}_max_ms"] = round(self.publisher_stage_max_seconds[name] * 1000.0, 2)
        return values

    def _clear_publisher_progress_warning(self) -> None:
        self.publisher_progress_warning_active = False

    def _warn_publisher_progress_stall(
        self,
        now: float,
        stalled_seconds: float,
        output_seq: int,
        queue_depth: int,
        inflight_window_id: int,
        inflight_source_seq: int,
    ) -> None:
        if self.publisher_progress_warning_active:
            return
        self.publisher_progress_warning_active = True
        self.publisher_progress_warning_count += 1
        self.publisher_progress_warning_last_seconds = stalled_seconds
        self.publisher_progress_warning_last_at_unix = time.time()
        stage = self._publisher_stage_telemetry(now)
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            print(
                f"{timestamp} WARNING publisher output stalled "
                f"stalled_ms={stalled_seconds * 1000.0:.2f} "
                f"output_seq={output_seq} queue_frames={queue_depth} "
                f"inflight_window_id={inflight_window_id} "
                f"inflight_source_seq={inflight_source_seq} "
                f"thread_stage={stage['publisher_stage']} "
                f"thread_stage_age_ms={stage['publisher_stage_age_ms']:.2f} "
                f"thread_stage_window_id={stage['publisher_stage_window_id']} "
                f"thread_stage_source_seq={stage['publisher_stage_source_seq']}",
                file=sys.stderr,
                flush=True,
            )
        except (OSError, ValueError):
            pass

    def _record_nonempty_wait_locked(
        self,
        requested_delay: float,
        elapsed: float,
        window_id: int,
        queue_depth: int,
    ) -> None:
        requested_delay = max(0.0, requested_delay)
        elapsed = max(0.0, elapsed)
        self.nonempty_wait_delay_seconds = requested_delay
        self.nonempty_wait_delay_max_seconds = max(
            self.nonempty_wait_delay_max_seconds,
            requested_delay,
        )
        self.nonempty_wait_elapsed_seconds = elapsed
        self.nonempty_wait_elapsed_max_seconds = max(
            self.nonempty_wait_elapsed_max_seconds,
            elapsed,
        )
        if elapsed <= PUBLISHER_NONEMPTY_WAIT_WARNING_SECONDS:
            return
        self.nonempty_wait_over_250ms_count += 1
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            print(
                f"{timestamp} WARNING slow publisher condition wait "
                f"window_id={window_id} queue_frames={queue_depth} "
                f"requested_ms={requested_delay * 1000.0:.2f} "
                f"elapsed_ms={elapsed * 1000.0:.2f}",
                file=sys.stderr,
                flush=True,
            )
        except (OSError, ValueError):
            pass

    @staticmethod
    def _input_matches_token(document: dict[str, Any], token: SessionToken) -> bool:
        return bool(
            str(document.get("track_id", "") or "") == token.track_id
            and str(document.get("source_session_id", "") or "") == token.source_session_id
            and as_int(document.get("reset_epoch"), -1) == token.reset_epoch
            and as_int(document.get("source_reset_epoch"), -1) == token.source_reset_epoch
        )

    def _record_input_drop_locked(self, frame_count: int) -> None:
        dropped = max(0, frame_count)
        if not dropped:
            return
        self.input_dropped_frames += dropped
        self.dropped_frames += dropped
        if self.pacer is not None:
            self.pacer.note_dropped(dropped)

    @staticmethod
    def _frame_matches_input_event(
        frame: OutputFrame,
        event_id: str,
        boundary_seq: int,
    ) -> bool:
        return bool(event_id and frame.input_event_id == event_id and frame.source_seq >= boundary_seq)

    def _frame_can_continue_while_input_pending(self, frame: OutputFrame) -> bool:
        if not self.pending_input_event_id:
            return True
        if self.active_input_event_id:
            return self._frame_matches_input_event(
                frame,
                self.active_input_event_id,
                self.active_input_boundary_seq,
            )
        return not frame.input_event_id

    def _activate_pending_input_boundary_locked(
        self,
        now: float,
        replacement_frame: OutputFrame | None = None,
    ) -> bool:
        boundary_seq = self.pending_input_boundary_seq
        event_id = self.pending_input_event_id
        if boundary_seq < 0 or not event_id:
            return False
        replacement_available = bool(
            replacement_frame is not None
            and self._frame_matches_input_event(
                replacement_frame,
                event_id,
                boundary_seq,
            )
        )
        if not replacement_available and not any(
            self._frame_matches_input_event(item.frame, event_id, boundary_seq) for item in self.queue
        ):
            return False

        retained = deque(
            item for item in self.queue if self._frame_matches_input_event(item.frame, event_id, boundary_seq)
        )
        dropped = len(self.queue) - len(retained)
        overflow = max(0, len(retained) - self.max_frames)
        for _ in range(overflow):
            retained.pop()
        self.queue = retained
        if event_id != self.active_input_event_id:
            self.input_fast_forward_count += 1
        self.active_input_event_id = event_id
        self.active_input_boundary_seq = boundary_seq
        self.pending_input_event_id = ""
        self.pending_input_boundary_seq = -1
        self.progress_generation += 1
        self._record_input_drop_locked(dropped)
        if overflow:
            self.dropped_frames += overflow
            if self.pacer is not None:
                self.pacer.note_dropped(overflow)
        if self.pacer is not None:
            self.pacer.release_boundary(now)
        self.condition.notify_all()
        return True

    def _refresh_input_boundary_locked(self, token: SessionToken) -> None:
        path = self.input_state_path
        if path is None:
            return
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            return
        if mtime_ns == self.input_state_mtime_ns:
            return
        document = read_json(path)
        if not document or not self._input_matches_token(document, token):
            return
        event_id = clean_input_event_id(document.get("input_event_id"))
        boundary_seq = as_int(document.get("boundary_source_seq"), -1)
        if not event_id or boundary_seq < 0:
            return
        self.input_state_mtime_ns = mtime_ns
        if event_id == self.input_event_id:
            previous_boundary_seq = self.input_boundary_seq
            self.input_boundary_seq = max(self.input_boundary_seq, boundary_seq)
            if self.input_boundary_seq <= previous_boundary_seq:
                return
        else:
            self.input_event_id = event_id
            self.input_boundary_seq = boundary_seq

        if event_id == self.active_input_event_id and self.input_boundary_seq <= self.active_input_boundary_seq:
            return
        self.pending_input_event_id = event_id
        self.pending_input_boundary_seq = max(
            self.input_boundary_seq,
            self.active_input_boundary_seq,
        )
        self.progress_generation += 1
        self._activate_pending_input_boundary_locked(time.monotonic())

    def _item_predates_input_locked(self, item: PublishItem) -> bool:
        self._refresh_input_boundary_locked(item.token)
        if self.pending_input_boundary_seq >= 0 and self._frame_matches_input_event(
            item.frame,
            self.pending_input_event_id,
            self.pending_input_boundary_seq,
        ):
            self._activate_pending_input_boundary_locked(
                time.monotonic(),
                item.frame,
            )
        elif not self._frame_can_continue_while_input_pending(item.frame):
            return True
        return bool(
            self.active_input_event_id
            and not self._frame_matches_input_event(
                item.frame,
                self.active_input_event_id,
                self.active_input_boundary_seq,
            )
        )

    def _record_current_input_drop_locked(self) -> None:
        self._record_input_drop_locked(1)

    def enqueue(self, token: SessionToken, window: WindowData, frames: list[OutputFrame]) -> None:
        with self.condition:
            if self.active_key != token.key():
                raise FenceChanged("publisher session changed before enqueue")
            if self.thread_error or not self.thread.is_alive():
                self.failed_key = token.key()
                raise OutputPublishFailed(self.thread_error or "publisher thread is not alive")
            if self.failed_key == token.key():
                raise OutputPublishFailed("publisher is failed until the bridge starts a fresh epoch")
            self._refresh_input_boundary_locked(token)
            items = [PublishItem(token, window.header, frame) for frame in frames]
            now = time.monotonic()
            pending_event_id = self.pending_input_event_id
            pending_boundary_seq = self.pending_input_boundary_seq
            if (
                pending_event_id
                and pending_boundary_seq >= 0
                and any(
                    self._frame_matches_input_event(
                        item.frame,
                        pending_event_id,
                        pending_boundary_seq,
                    )
                    for item in items
                )
            ):
                if self.pacer is not None:
                    self.pacer.observe_enqueue(len(items), len(self.queue), now)
                self.queue.extend(items)
                self._activate_pending_input_boundary_locked(now)
                self.condition.notify_all()
                return
            active_boundary_seq = self.active_input_boundary_seq
            active_event_id = self.active_input_event_id
            if pending_event_id:
                retained = [item for item in items if self._frame_can_continue_while_input_pending(item.frame)]
                self._record_input_drop_locked(len(items) - len(retained))
                items = retained
            elif active_event_id and active_boundary_seq >= 0:
                retained = [
                    item
                    for item in items
                    if self._frame_matches_input_event(
                        item.frame,
                        active_event_id,
                        active_boundary_seq,
                    )
                ]
                self._record_input_drop_locked(len(items) - len(retained))
                items = retained
            if not items:
                self.condition.notify_all()
                return
            if self.pacer is not None:
                self.pacer.observe_enqueue(len(items), len(self.queue), now)
            overflow = max(0, len(self.queue) + len(items) - self.max_frames)
            for _ in range(min(overflow, len(self.queue))):
                self.queue.popleft()
            if len(items) > self.max_frames:
                items = items[-self.max_frames :]
            if overflow > 0:
                self.dropped_frames += overflow
                if self.pacer is not None:
                    self.pacer.note_dropped(overflow)
            self.queue.extend(items)
            self.condition.notify_all()

    def failed_for(self, token: SessionToken) -> bool:
        with self.condition:
            return bool(self.thread_error) or self.failed_key == token.key()

    def fatal_error(self) -> str:
        with self.condition:
            if self.thread_error:
                return self.thread_error
            if not self.thread.is_alive():
                return "frame publisher thread exited without reporting an error"
            return ""

    def assert_healthy(self, token: SessionToken) -> None:
        with self.condition:
            if self.active_key != token.key():
                raise FenceChanged("publisher session changed before health check")
            error = self.thread_error
            if not error and not self.thread.is_alive():
                error = "frame publisher thread is not alive"
            if error:
                self.failed_key = token.key()
                raise OutputPublishFailed(error)

    def wait_for_window(self, token: SessionToken, window_id: int, timeout: float = 120.0) -> bool:
        deadline = time.monotonic() + timeout
        token_key = token.key()
        with self.condition:
            while time.monotonic() < deadline:
                if self.active_key != token_key or self.failed_key == token_key:
                    return False
                if (
                    self.completed_window.get(token_key, -1) >= window_id
                    and self.status_written_window.get(token_key, -1) >= window_id
                ):
                    return True
                self.condition.wait(timeout=min(0.25, deadline - time.monotonic()))
        return False

    def _status_writer_run(self) -> None:
        while True:
            with self.condition:
                while self.pending_completion_status is None:
                    self.condition.wait()
                update = self.pending_completion_status
                self.pending_completion_status = None
                self.status_inflight_window_id = update.window_id

            for attempt in range(1, self.publish_attempts + 1):
                stale = False
                error_text = ""
                elapsed = 0.0
                with self.status_write_lock:
                    with self.condition:
                        stale = (
                            self.active_key != update.token_key
                            or self.status_written_window.get(update.token_key, -1) >= update.window_id
                        )
                    if stale:
                        break
                    started_at = time.monotonic()
                    try:
                        self.status.update(**update.values)
                    except Exception as error:
                        error_text = f"{type(error).__name__}: {error}"[-1000:]
                    elapsed = time.monotonic() - started_at

                with self.condition:
                    if self.active_key != update.token_key:
                        stale = True
                    elif not error_text:
                        previous = self.status_written_window.get(update.token_key, -1)
                        self.status_written_window[update.token_key] = max(previous, update.window_id)
                        self.status_last_write_seconds = elapsed
                        self.status_max_write_seconds = max(self.status_max_write_seconds, elapsed)
                        self.status_last_error = ""
                        self.condition.notify_all()
                    else:
                        self.status_write_errors += 1
                        self.status_last_error = error_text
                    newer_pending = (
                        self.pending_completion_status is not None
                        and self.pending_completion_status.token_key == update.token_key
                        and self.pending_completion_status.window_id > update.window_id
                    )
                if stale or not error_text or newer_pending:
                    break
                if attempt < self.publish_attempts:
                    time.sleep(min(0.1 * attempt, 0.5))

            with self.condition:
                if self.status_inflight_window_id == update.window_id:
                    self.status_inflight_window_id = -1
                self.condition.notify_all()

    def _output_fps(self) -> float:
        now = time.monotonic()
        while self.publish_times and now - self.publish_times[0] > 2.0:
            self.publish_times.popleft()
        if len(self.publish_times) < 2:
            return 0.0
        elapsed = self.publish_times[-1] - self.publish_times[0]
        return (len(self.publish_times) - 1) / elapsed if elapsed > 0 else 0.0

    def _telemetry(self) -> dict[str, Any]:
        with self.condition:
            queue_depth = len(self.queue)
            if self.pacer is not None:
                values = self.pacer.telemetry(queue_depth)
            else:
                values = {
                    "publisher_mode": "fixed",
                    "publisher_queue_frames": queue_depth,
                    "publisher_target_fps": self.publish_fps,
                    "publisher_dropped_frames": self.dropped_frames,
                }
            values["publisher_queue_capacity_frames"] = self.max_frames
            values["publisher_status_pending_window_id"] = (
                self.pending_completion_status.window_id if self.pending_completion_status is not None else -1
            )
            values["publisher_status_inflight_window_id"] = self.status_inflight_window_id
            values["publisher_status_coalesced_updates"] = self.status_coalesced_updates
            values["publisher_status_write_errors"] = self.status_write_errors
            values["publisher_status_write_ms"] = round(self.status_last_write_seconds * 1000.0, 2)
            values["publisher_status_write_max_ms"] = round(self.status_max_write_seconds * 1000.0, 2)
            values["publisher_status_write_error"] = self.status_last_error
            values["publisher_thread_alive"] = bool(self.thread.is_alive() and not self.thread_error)
            values["publisher_thread_error"] = self.thread_error
            values["publisher_thread_failed_at"] = self.thread_failed_at
            values["publisher_status_writer_alive"] = self.status_writer_thread.is_alive()
            values["publisher_watchdog_alive"] = self.watchdog_thread.is_alive()
            values["publisher_progress_timeout_seconds"] = self.progress_timeout_seconds
            values["publisher_inflight_window_id"] = self.publish_inflight_window_id
            values["publisher_inflight_source_seq"] = self.publish_inflight_source_seq
            values["publisher_inflight_input_event_id"] = self.publish_inflight_input_event_id
            values["publisher_input_event_id"] = self.input_event_id
            values["publisher_input_boundary_seq"] = self.input_boundary_seq
            values["publisher_active_input_event_id"] = self.active_input_event_id
            values["publisher_active_input_boundary_seq"] = self.active_input_boundary_seq
            values["publisher_pending_input_event_id"] = self.pending_input_event_id
            values["publisher_pending_input_boundary_seq"] = self.pending_input_boundary_seq
            values["publisher_input_boundary_pending"] = bool(
                self.pending_input_event_id and self.pending_input_boundary_seq >= 0
            )
            values["publisher_input_fast_forward_count"] = self.input_fast_forward_count
            values["publisher_input_dropped_frames"] = self.input_dropped_frames
            values["publisher_nonempty_wait_delay_ms"] = round(self.nonempty_wait_delay_seconds * 1000.0, 2)
            values["publisher_nonempty_wait_delay_max_ms"] = round(self.nonempty_wait_delay_max_seconds * 1000.0, 2)
            values["publisher_nonempty_wait_elapsed_ms"] = round(self.nonempty_wait_elapsed_seconds * 1000.0, 2)
            values["publisher_nonempty_wait_elapsed_max_ms"] = round(self.nonempty_wait_elapsed_max_seconds * 1000.0, 2)
            values["publisher_nonempty_wait_over_250ms_count"] = self.nonempty_wait_over_250ms_count
            values["publisher_progress_warning_threshold_ms"] = round(PUBLISH_PROGRESS_WARNING_SECONDS * 1000.0, 2)
            values["publisher_progress_warning_active"] = self.publisher_progress_warning_active
            values["publisher_progress_warning_count"] = self.publisher_progress_warning_count
            values["publisher_progress_warning_last_ms"] = round(
                self.publisher_progress_warning_last_seconds * 1000.0, 2
            )
            values["publisher_progress_warning_last_at_unix"] = round(self.publisher_progress_warning_last_at_unix, 6)
            values.update(self._publisher_stage_telemetry())
            if self.gc_telemetry is not None:
                values.update(self.gc_telemetry.telemetry())
            return values

    def _metadata(
        self,
        item: PublishItem,
        output_seq: int,
        frame_index: int,
    ) -> dict[str, Any]:
        token = item.token
        frame = item.frame
        now = time.time()
        causal_metadata = frame.causal_trace.metadata() if frame.causal_trace is not None else {}
        return {
            "schema_version": SCHEMA_VERSION,
            "seq": output_seq,
            "frame_index": frame_index,
            "source_frame": frame.source_frame,
            "source_seq": frame.source_seq,
            "input_event_id": frame.input_event_id,
            "sample_index": frame.sample_index,
            "width": frame.rgb.shape[1],
            "height": frame.rgb.shape[0],
            "generation": token.reset_epoch,
            "reset_epoch": token.reset_epoch,
            "source_reset_epoch": token.source_reset_epoch,
            "source_generation": token.source_reset_epoch,
            "render_request_id": token.request_id,
            "prompt_revision": token.prompt_revision,
            "prompt_hash": token.prompt_hash,
            "track_id": token.track_id,
            "source_session_id": token.source_session_id,
            "window_id": item.header.window_id,
            "window_frame_offset": frame.window_offset,
            "window_frame_count": frame.window_frame_count,
            "window_complete": frame.window_offset + 1 == frame.window_frame_count,
            "kart_id": frame.kart_id,
            "control": frame.control,
            "published_at": now,
            "updated_at": now,
            **causal_metadata,
            **self._telemetry(),
        }

    def _commit(self, item: PublishItem) -> None:
        self.reader.assert_fence(item.token)
        with self.condition:
            if self.active_key != item.token.key():
                raise FenceChanged("publisher reset before frame commit")
            if self._item_predates_input_locked(item):
                self._record_current_input_drop_locked()
                raise FenceChanged("publisher frame predates the active input boundary")
        frame = item.frame
        self._start_publisher_stage("stage_rgba", item)
        rgba = np.empty((frame.rgb.shape[0], frame.rgb.shape[1], 4), dtype=np.uint8)
        rgba[..., :3] = frame.rgb
        rgba[..., 3] = 255
        rgb_path = self.rendered_dir / "rgb.rgba8"
        latest_path = self.rendered_dir / "latest.json"
        self._start_publisher_stage("stage_rgb_bytes", item)
        rgb_temporary = stage_bytes(rgb_path, rgba.tobytes(order="C"))
        latest_temporary: Path | None = None
        try:
            # Preview writes input.json under this same advisory lock. Staging the
            # large RGBA payload first keeps the serialized commit window tiny.
            self._start_publisher_stage("input_lock_wait", item)
            with input_state_commit_lock(self.input_state_path):
                self._start_publisher_stage("metadata_json", item)
                self.reader.assert_fence(item.token)
                with self.condition:
                    if self.active_key != item.token.key():
                        raise FenceChanged("publisher reset during frame commit")
                    if self._item_predates_input_locked(item):
                        self._record_current_input_drop_locked()
                        raise FenceChanged("publisher frame crossed a newer input boundary")
                    next_output_seq = self.output_seq + 1
                    next_frame_index = self.frame_index + 1
                    metadata = self._metadata(
                        item,
                        next_output_seq,
                        next_frame_index,
                    )
                    latest_temporary = stage_bytes(
                        latest_path,
                        json_payload_bytes(metadata),
                    )
                    self._start_publisher_stage("promote", item)
                    self.reader.assert_fence(item.token)
                    if self.active_key != item.token.key():
                        raise FenceChanged("publisher reset before output promotion")
                    if self._item_predates_input_locked(item):
                        self._record_current_input_drop_locked()
                        raise FenceChanged("publisher frame crossed the commit-time input boundary")
                    promote_staged_file(rgb_temporary, rgb_path)
                    promote_staged_file(latest_temporary, latest_path)
                    self.output_seq = next_output_seq
                    self.frame_index = next_frame_index
            self._start_publisher_stage("postcommit", item)
            self.reader.assert_fence(item.token)
            with self.condition:
                if self.active_key != item.token.key():
                    raise FenceChanged("publisher reset after frame commit")
        finally:
            discard_staged_file(rgb_temporary)
            discard_staged_file(latest_temporary)

    def _mark_completed(self, item: PublishItem) -> None:
        token = item.token
        header = item.header
        with self.condition:
            if self.active_key != token.key():
                return
            output_fps = round(self._output_fps(), 3)
            previous = self.completed_window.get(token.key(), -1)
            self.completed_window[token.key()] = max(previous, header.window_id)
            values = dict(
                prompt=token.prompt,
                prompt_hash=token.prompt_hash,
                status="streaming",
                error="",
                ready=True,
                backlog_eligible=True,
                needs_fresh_source=False,
                reset_epoch=token.reset_epoch,
                source_reset_epoch=token.source_reset_epoch,
                prompt_revision=token.prompt_revision,
                track_id=token.track_id,
                source_session_id=token.source_session_id,
                completed_request_id=token.request_id,
                acknowledged_request_id=token.request_id,
                completed_window_id=header.window_id,
                completed_immutable_file=header.path.name,
                completed_source_seq=header.last_seq,
                completed_at=time.time(),
                output_fps=output_fps,
                output_seq=self.output_seq,
                **(item.frame.causal_trace.metadata() if item.frame.causal_trace is not None else {}),
                **self._telemetry(),
            )
            update = CompletionStatusUpdate(token.key(), header.window_id, values)
            pending = self.pending_completion_status
            if pending is None or pending.token_key != update.token_key:
                self.pending_completion_status = update
            elif update.window_id > pending.window_id:
                self.pending_completion_status = update
                self.status_coalesced_updates += 1
            self.condition.notify_all()

    def _mark_failed(self, item: PublishItem, error: BaseException) -> None:
        token = item.token
        with self.condition:
            if self.active_key != token.key():
                return
            self.failed_key = token.key()
            self.queue.clear()
            self.condition.notify_all()
        self.status.update(
            prompt=token.prompt,
            status="failed",
            error=f"output publish failed: {error}\n{traceback.format_exc()}"[-4000:],
            ready=True,
            backlog_eligible=False,
            needs_fresh_source=False,
            reset_epoch=token.reset_epoch,
            source_reset_epoch=token.source_reset_epoch,
            prompt_revision=token.prompt_revision,
            track_id=token.track_id,
            source_session_id=token.source_session_id,
            accepted_request_id=token.request_id,
            failed_window_id=item.header.window_id,
            output_fps=0.0,
        )

    def _publish_item(self, item: PublishItem, started_at: float | None = None) -> bool:
        committed = False
        for attempt in range(1, self.publish_attempts + 1):
            self._start_publisher_stage("precommit", item)
            try:
                self._commit(item)
                committed = True
                break
            except FenceChanged:
                break
            except Exception as error:
                if attempt == self.publish_attempts:
                    self._mark_failed(item, error)
                else:
                    time.sleep(min(0.1 * attempt, 0.5))
        if not committed:
            return False
        published_at = time.monotonic()
        with self.condition:
            if self.active_key != item.token.key():
                return False
            self.publish_times.append(published_at)
            if self.pacer is not None and started_at is not None:
                self.pacer.note_published(started_at, published_at)
        if item.frame.window_offset + 1 == item.frame.window_frame_count:
            self._mark_completed(item)
        return True

    def _publish(self, item: PublishItem, started_at: float | None = None) -> bool:
        with self.condition:
            self.publish_inflight_window_id = item.header.window_id
            self.publish_inflight_source_seq = item.frame.source_seq
            self.publish_inflight_input_event_id = item.frame.input_event_id
        try:
            return self._publish_item(item, started_at)
        finally:
            with self.condition:
                self.publish_inflight_window_id = -1
                self.publish_inflight_source_seq = -1
                self.publish_inflight_input_event_id = ""
                self.condition.notify_all()

    def _run_fixed(self) -> None:
        next_deadline = time.monotonic()
        while True:
            self._start_publisher_stage("loop")
            self._start_publisher_stage("select")
            with self.condition:
                while not self.queue:
                    self._start_publisher_stage("idle")
                    self._retarget_publisher_stage(None)
                    self.condition.wait()
                    next_deadline = time.monotonic()
                self._start_publisher_stage("select")
                item = self.queue.popleft()
                self._retarget_publisher_stage(item)
                if self.active_key != item.token.key():
                    continue
                if self._item_predates_input_locked(item):
                    continue
            self._start_publisher_stage("reacquire_condition", item)
            if not self._publish(item):
                continue
            self._start_publisher_stage("loop")
            if self.publish_fps > 0:
                next_deadline += 1.0 / self.publish_fps
                delay = next_deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.25:
                    next_deadline = time.monotonic()

    def _run_adaptive(self) -> None:
        assert self.pacer is not None
        while True:
            self._start_publisher_stage("loop")
            self._start_publisher_stage("select")
            with self.condition:
                while True:
                    while not self.queue:
                        self._start_publisher_stage("idle")
                        self._retarget_publisher_stage(None)
                        now = time.monotonic()
                        timeout = self.pacer.empty_wait_timeout(now)
                        if timeout is None:
                            self.pacer.begin_underflow(now)
                        self.condition.wait(timeout=timeout)
                    self._start_publisher_stage("select")
                    item = self.queue[0]
                    self._retarget_publisher_stage(item)
                    if self.active_key != item.token.key():
                        self.queue.popleft()
                        continue
                    if self._item_predates_input_locked(item):
                        # A new boundary may have filtered the selected item out
                        # and replaced the queue while evaluating this predicate.
                        if self.queue and self.queue[0] is item:
                            self.queue.popleft()
                        continue
                    delay = self.pacer.wait_delay(time.monotonic())
                    if delay > 0:
                        wait_started_at = time.monotonic()
                        self.condition.wait(timeout=delay)
                        wait_elapsed = time.monotonic() - wait_started_at
                        self._record_nonempty_wait_locked(
                            delay,
                            wait_elapsed,
                            item.header.window_id,
                            len(self.queue),
                        )
                        continue
                    item = self.queue.popleft()
                    break
            self._start_publisher_stage("reacquire_condition", item)
            with self.condition:
                if self.active_key != item.token.key():
                    continue
                started_at = time.monotonic()
                self.pacer.note_publish_started(started_at)
            self._publish(item, started_at)

    def _progress_watchdog_run(self) -> None:
        last_output_seq = self.output_seq
        last_generation = self.progress_generation
        stalled_since: float | None = None
        warning_eligible_since: float | None = None
        while True:
            time.sleep(PUBLISH_PROGRESS_POLL_SECONDS)
            if self.thread_error:
                return
            now = time.monotonic()
            output_seq = self.output_seq
            generation = self.progress_generation
            queue_depth = len(self.queue)
            inflight_window_id = self.publish_inflight_window_id
            has_work = self.active_key is not None and (queue_depth > 0 or inflight_window_id >= 0)
            if output_seq != last_output_seq:
                last_output_seq = output_seq
                last_generation = generation
                stalled_since = now if has_work else None
                warning_eligible_since = stalled_since
                self._clear_publisher_progress_warning()
                if not has_work:
                    continue
            if generation != last_generation or not has_work:
                last_generation = generation
                self._clear_publisher_progress_warning()
                if not has_work:
                    stalled_since = None
                    warning_eligible_since = None
                    continue
                stalled_since = now
                warning_eligible_since = now
            if stalled_since is None:
                stalled_since = now
                warning_eligible_since = now
            stalled_seconds = now - stalled_since
            stage = self._publisher_stage_telemetry(now)
            planned_wait_seconds = 0.0
            if self.pacer is not None and inflight_window_id < 0 and stage["publisher_stage"] == "select":
                planned_wait_seconds = self.pacer.wait_delay(now)
            if planned_wait_seconds > 0.0:
                planned_wait_until = now + planned_wait_seconds
                warning_eligible_since = max(
                    warning_eligible_since or stalled_since,
                    planned_wait_until,
                )
            elif warning_eligible_since is None:
                warning_eligible_since = stalled_since
            warning_stalled_seconds = now - warning_eligible_since
            if (
                warning_stalled_seconds >= PUBLISH_PROGRESS_WARNING_SECONDS
                and self.output_seq == output_seq
                and self.progress_generation == generation
                and self.active_key is not None
                and (self.queue or self.publish_inflight_window_id >= 0)
            ):
                self._warn_publisher_progress_stall(
                    now,
                    stalled_seconds,
                    output_seq,
                    queue_depth,
                    inflight_window_id,
                    self.publish_inflight_source_seq,
                )
            if stalled_seconds < self.progress_timeout_seconds:
                continue
            if (
                self.output_seq != output_seq
                or self.progress_generation != generation
                or self.active_key is None
                or (not self.queue and self.publish_inflight_window_id < 0)
            ):
                continue
            error_text = (
                "frame publisher made no output progress while work remained "
                f"for {stalled_seconds:.1f}s: output_seq={output_seq} "
                f"queue_frames={queue_depth} inflight_window_id={inflight_window_id} "
                f"inflight_source_seq={self.publish_inflight_source_seq}"
            )
            failed_at = time.time()
            self.thread_error = error_text
            self.thread_failed_at = failed_at
            self.failed_key = self.active_key

            def write_failure_status() -> None:
                try:
                    self.status.update(
                        status="failed",
                        error=error_text,
                        ready=True,
                        backlog_eligible=False,
                        needs_fresh_source=False,
                        output_fps=0.0,
                        publisher_thread_alive=self.thread.is_alive(),
                        publisher_thread_error=error_text,
                        publisher_thread_failed_at=failed_at,
                        publisher_progress_stalled=True,
                        publisher_progress_stalled_seconds=round(stalled_seconds, 3),
                        publisher_progress_timeout_seconds=(self.progress_timeout_seconds),
                        publisher_queue_frames=queue_depth,
                        publisher_inflight_window_id=inflight_window_id,
                        publisher_inflight_source_seq=(self.publish_inflight_source_seq),
                        publisher_watchdog_fatal=True,
                        publisher_watchdog_alive=False,
                    )
                except BaseException:
                    pass

            try:
                status_thread = threading.Thread(
                    target=write_failure_status,
                    name="alaya-watchdog-status",
                    daemon=True,
                )
                status_thread.start()
                status_thread.join(timeout=PUBLISH_PROGRESS_POLL_SECONDS)
            finally:
                self.watchdog_exit(70)
            return

    def _record_thread_failure(self, error: BaseException) -> None:
        error_text = (f"frame publisher thread failed: {type(error).__name__}: {error}\n{traceback.format_exc()}")[
            -4000:
        ]
        failed_at = time.time()
        with self.condition:
            self.thread_error = error_text
            self.thread_failed_at = failed_at
            self.failed_key = self.active_key
            self.queue.clear()
            self.condition.notify_all()
        self.status.update(
            status="failed",
            error=error_text,
            ready=True,
            backlog_eligible=False,
            needs_fresh_source=False,
            output_fps=0.0,
            publisher_queue_frames=0,
            publisher_thread_alive=False,
            publisher_thread_error=error_text,
            publisher_thread_failed_at=failed_at,
            publisher_status_writer_alive=self.status_writer_thread.is_alive(),
        )
        print(error_text, file=sys.stderr, flush=True)

    def _run(self) -> None:
        try:
            if self.pacer is None:
                self._run_fixed()
            else:
                self._run_adaptive()
        except BaseException as error:
            self._record_thread_failure(error)


def _rendered_output_matches_token(
    rendered: dict[str, Any],
    token: SessionToken,
) -> bool:
    return bool(
        as_int(rendered.get("schema_version"), -1) == SCHEMA_VERSION
        and as_int(rendered.get("reset_epoch"), -1) == token.reset_epoch
        and as_int(rendered.get("source_reset_epoch"), -1) == token.source_reset_epoch
        and as_int(rendered.get("render_request_id"), -1) == token.request_id
        and as_int(rendered.get("prompt_revision"), -1) == token.prompt_revision
        and str(rendered.get("prompt_hash", "") or "") == token.prompt_hash
        and str(rendered.get("track_id", "") or "") == token.track_id
        and str(rendered.get("source_session_id", "") or "") == token.source_session_id
    )


def previous_recovery_target(
    status: WorkerStatus,
    token: SessionToken,
    rendered_dir: Path,
) -> PreviousRecoveryTarget | None:
    previous = status.previous
    previous_instance = str(previous.get("worker_instance_id", "") or "")
    if (
        not previous
        or not previous_instance
        or previous_instance == status.instance_id
        or as_int(previous.get("schema_version"), -1) != SCHEMA_VERSION
        or str(previous.get("status", "") or "")
        not in {
            "warming",
            "recovering",
            "rendering",
            "output_queued",
            "streaming",
            "waiting_for_input",
        }
        or not bool(previous.get("ready", False))
        or as_int(previous.get("reset_epoch"), -1) != token.reset_epoch
        or as_int(previous.get("source_reset_epoch"), -1) != token.source_reset_epoch
        or as_int(previous.get("prompt_revision"), -1) != token.prompt_revision
        or clean_prompt(previous.get("prompt")) != token.prompt
        or str(previous.get("track_id", "") or "") != token.track_id
        or str(previous.get("source_session_id", "") or "") != token.source_session_id
    ):
        return None

    rendered = read_json(rendered_dir / "latest.json")
    rendered_matches = _rendered_output_matches_token(rendered, token)
    completed_request_matches = as_int(previous.get("completed_request_id"), -1) == token.request_id
    accepted_request_matches = as_int(previous.get("accepted_request_id"), -1) == token.request_id
    rendered_window_id = as_int(rendered.get("window_id"), -1)
    rendered_source_seq = as_int(rendered.get("source_seq"), -1)
    legacy_status = "prompt_hash" not in previous or "completed_source_seq" not in previous
    if legacy_status:
        if not (
            rendered_matches
            and (accepted_request_matches or completed_request_matches)
            and rendered_window_id >= 0
            and rendered_source_seq >= 0
        ):
            return None
        return PreviousRecoveryTarget(
            window_id=rendered_window_id,
            visible_source_seq=rendered_source_seq,
            status_window_id=-1,
            rendered_window_id=rendered_window_id,
            previous_worker_instance_id=previous_instance,
        )

    if str(previous.get("prompt_hash", "") or "") != token.prompt_hash:
        return None
    status_window_id = as_int(previous.get("completed_window_id"), -1)
    status_source_seq = as_int(previous.get("completed_source_seq"), -1)
    status_authoritative = bool(completed_request_matches and status_window_id >= 0 and status_source_seq >= 0)
    rendered_authoritative = bool(
        rendered_matches
        and (accepted_request_matches or completed_request_matches)
        and rendered_window_id >= 0
        and rendered_source_seq >= 0
    )
    if not status_authoritative:
        status_window_id = -1
        status_source_seq = -1
    if not rendered_authoritative:
        rendered_window_id = -1
        rendered_source_seq = -1

    target_window_id = max(status_window_id, rendered_window_id)
    if target_window_id < 0:
        return None
    if target_window_id == rendered_window_id and rendered_source_seq >= 0:
        visible_source_seq = rendered_source_seq
    elif target_window_id == status_window_id and status_source_seq >= 0:
        visible_source_seq = status_source_seq
    else:
        return None
    return PreviousRecoveryTarget(
        window_id=target_window_id,
        visible_source_seq=visible_source_seq,
        status_window_id=status_window_id,
        rendered_window_id=rendered_window_id,
        previous_worker_instance_id=previous_instance,
    )


def is_hidden_replay_window(
    window_id: int,
    recovery_target: PreviousRecoveryTarget | None,
) -> bool:
    return bool(recovery_target is not None and window_id <= recovery_target.window_id)


def assert_visible_source_seq_advances(
    frames: list[OutputFrame],
    visible_source_seq: int,
) -> None:
    if visible_source_seq >= 0 and frames and min(frame.source_seq for frame in frames) <= visible_source_seq:
        raise ContinuityError(
            f"first unpublished window would move visible source_seq backward from {visible_source_seq}"
        )


def run_adaptive_pacer_self_check(cadence: TemporalCadence) -> dict[str, Any]:
    source_fps = 22.5 if cadence.latent_window == 4 else 22.0
    pacer = AdaptivePublishPacer(
        fps_cap=24.0,
        min_fps=12.0,
        source_fps_hint=source_fps,
        target_buffer_frames=2.5,
        steady_batch_frames=cadence.resumed_pixel_frames,
    )
    jitter_seconds = (
        0.000,
        0.008,
        -0.006,
        0.012,
        -0.008,
        0.020,
        -0.015,
        0.004,
        0.055,
        -0.040,
        0.006,
        -0.004,
        0.085,
        -0.065,
        0.003,
        -0.005,
    )
    arrivals: list[tuple[float, int]] = [(0.0, cadence.initial_pixel_frames)]
    arrival_at = 0.0
    nominal_interval = cadence.resumed_pixel_frames / source_fps
    simulation_chunks = int(660.0 * source_fps / cadence.resumed_pixel_frames) + 1
    for index in range(simulation_chunks):
        arrival_at += nominal_interval + jitter_seconds[index % len(jitter_seconds)]
        arrivals.append((arrival_at, cadence.resumed_pixel_frames))

    now = 0.0
    queue_depth = 0
    arrival_index = 0
    publish_times: list[float] = []
    publish_start_times: list[float] = []
    commit_durations: list[float] = []
    max_queue_depth = 0
    epsilon = 1e-9
    while arrival_index < len(arrivals) or queue_depth:
        next_arrival = arrivals[arrival_index][0] if arrival_index < len(arrivals) else float("inf")
        if queue_depth == 0:
            if next_arrival == float("inf"):
                break
            underflow_at = now + pacer.wait_delay(now)
            if pacer.publish_count > 0 and underflow_at < next_arrival - epsilon:
                pacer.begin_underflow(underflow_at)
            now = next_arrival
            frame_count = arrivals[arrival_index][1]
            pacer.observe_enqueue(frame_count, queue_depth, now)
            queue_depth += frame_count
            max_queue_depth = max(max_queue_depth, queue_depth)
            arrival_index += 1
            continue

        next_publish = now + pacer.wait_delay(now)
        if next_arrival <= next_publish + epsilon:
            now = next_arrival
            frame_count = arrivals[arrival_index][1]
            pacer.observe_enqueue(frame_count, queue_depth, now)
            queue_depth += frame_count
            max_queue_depth = max(max_queue_depth, queue_depth)
            arrival_index += 1
            continue

        started_at = next_publish
        queue_depth -= 1
        publish_index = len(publish_times)
        commit_duration = 0.004 + ((publish_index * 7) % 9) * 0.001
        if publish_index % 257 == 113:
            commit_duration = 0.035
        elif publish_index % 101 == 47:
            commit_duration = 0.024
        committed_at = started_at + commit_duration
        pacer.note_publish_started(started_at)
        while arrival_index < len(arrivals) and arrivals[arrival_index][0] <= committed_at:
            arrival_time, frame_count = arrivals[arrival_index]
            pacer.observe_enqueue(frame_count, queue_depth, arrival_time)
            queue_depth += frame_count
            max_queue_depth = max(max_queue_depth, queue_depth)
            arrival_index += 1
        now = committed_at
        publish_start_times.append(started_at)
        publish_times.append(committed_at)
        commit_durations.append(commit_duration)
        pacer.note_published(started_at, committed_at)

    gaps = [later - earlier for earlier, later in zip(publish_times, publish_times[1:])]
    if len(gaps) < 100:
        raise AssertionError("adaptive pacer simulation produced too few output gaps")
    simulated_seconds = arrivals[-1][0]
    if simulated_seconds < 600.0:
        raise AssertionError("adaptive pacer simulation did not cover ten minutes")
    ordered_gaps = sorted(gaps)
    p99_gap = ordered_gaps[int(0.99 * (len(ordered_gaps) - 1))]
    max_gap = max(gaps)
    min_gap = min(gaps)
    start_gaps = [later - earlier for earlier, later in zip(publish_start_times, publish_start_times[1:])]
    effective_fps = (len(publish_times) - 1) / (publish_times[-1] - publish_times[0])
    minimum_allowed_gap = 1.0 / 24.0
    if min(start_gaps) + 1e-9 < minimum_allowed_gap:
        raise AssertionError("adaptive pacer emitted a catch-up burst")
    if min_gap < 0.030:
        raise AssertionError("adaptive pacer emitted an excessively short visible gap")
    if p99_gap > 0.065 or max_gap > 0.100:
        raise AssertionError(f"adaptive pacer gap target missed: p99={p99_gap:.6f}s max={max_gap:.6f}s")
    minimum_effective_fps = source_fps - 0.5
    if effective_fps < minimum_effective_fps:
        raise AssertionError(
            f"adaptive pacer throughput target missed: {effective_fps:.3f} FPS, "
            f"expected at least {minimum_effective_fps:.3f}"
        )
    if min(commit_durations) > 0.004 or max(commit_durations) < 0.035:
        raise AssertionError("adaptive pacer simulation did not cover 4-35ms commits")
    if pacer.startup_hold_seconds > 0.300:
        raise AssertionError("adaptive startup reserve exceeds the 300ms latency budget")
    if max_queue_depth > cadence.resumed_pixel_frames + 8:
        raise AssertionError("adaptive pacer allowed excessive queue growth")
    if pacer.dropped_frames or pacer.underflow_count:
        raise AssertionError(
            f"adaptive pacer exhausted its reserve: underflows={pacer.underflow_count} drops={pacer.dropped_frames}"
        )

    pacer.reset()
    reset_state = pacer.telemetry(0)
    if (
        pacer.next_publish_at is not None
        or pacer.last_publish_started_at is not None
        or pacer.last_enqueue_at is not None
        or reset_state["publisher_commit_ema_ms"] != 0.0
        or reset_state["publisher_completion_correction_carry_ms"] != 0.0
        or reset_state["publisher_underflow_count"] != 0
        or reset_state["publisher_dropped_frames"] != 0
    ):
        raise AssertionError("adaptive pacer reset leaked history across a session fence")

    ramp_pacer = AdaptivePublishPacer(
        fps_cap=24.0,
        min_fps=12.0,
        source_fps_hint=22.5,
        target_buffer_frames=2.5,
        steady_batch_frames=cadence.resumed_pixel_frames,
    )
    ramp_at = 0.0
    ramp_pacer.observe_enqueue(cadence.initial_pixel_frames, 0, ramp_at)
    for _ in range(24):
        ramp_at += cadence.resumed_pixel_frames / 24.0
        ramp_pacer.observe_enqueue(cadence.resumed_pixel_frames, 3, ramp_at)
    if ramp_pacer.target_fps < 23.95:
        raise AssertionError("adaptive arrival EWMA did not ramp from its hint to 24 FPS")
    return {
        "adaptive_gap_p99_ms": round(p99_gap * 1000.0, 3),
        "adaptive_gap_max_ms": round(max_gap * 1000.0, 3),
        "adaptive_gap_min_ms": round(min_gap * 1000.0, 3),
        "adaptive_effective_fps": round(effective_fps, 3),
        "adaptive_commit_min_ms": round(min(commit_durations) * 1000.0, 3),
        "adaptive_commit_max_ms": round(max(commit_durations) * 1000.0, 3),
        "adaptive_max_queue_frames": max_queue_depth,
        "adaptive_startup_hold_ms": round(
            (max(0, cadence.resumed_pixel_frames - cadence.initial_pixel_frames) + 2.5) / source_fps * 1000.0,
            3,
        ),
        "adaptive_ramp_target_fps": round(ramp_pacer.target_fps, 3),
        "adaptive_simulated_seconds": round(simulated_seconds, 3),
    }


def run_self_check(cadence: TemporalCadence, window_transport: str) -> int:
    adaptive_results = run_adaptive_pacer_self_check(cadence)
    with tempfile.TemporaryDirectory(prefix="stk-smallenc-worker-") as temporary:
        root = Path(temporary)
        control_dir = root / "control"
        input_dir = control_dir / "upload"
        input_dir.mkdir(parents=True)
        prompt = "cinematic forest at golden hour"
        live = {
            "schema_version": SCHEMA_VERSION,
            "reset_epoch": 7,
            "render_request_id": 11,
            "prompt_revision": 5,
            "prompt": prompt,
            "track_id": "black_forest",
        }
        atomic_write_json(control_dir / "live_control.json", live)
        immutable_name = "window_000000000123.npz"
        sidecar_name = "window_000000000123.gbuffers.npy"
        frame_count = cadence.initial_pixel_frames
        frame_seqs = np.arange(100, 100 + frame_count, dtype=np.int64)
        gbuffers = np.zeros((frame_count, len(MODALITIES), 8, 8, 3), dtype=np.uint8)
        arrays = {
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
            "reset_epoch": np.asarray(7, dtype=np.int64),
            "source_reset_epoch": np.asarray(3, dtype=np.int64),
            "render_request_id": np.asarray(11, dtype=np.int64),
            "prompt_revision": np.asarray(5, dtype=np.int64),
            "latent_window": np.asarray(cadence.latent_window, dtype=np.int32),
            "initial_pixel_frames": np.asarray(cadence.initial_pixel_frames, dtype=np.int32),
            "resumed_pixel_frames": np.asarray(cadence.resumed_pixel_frames, dtype=np.int32),
            "window_id": np.asarray(123, dtype=np.int64),
            "previous_window_id": np.asarray(-1, dtype=np.int64),
            "previous_last_seq": np.asarray(-1, dtype=np.int64),
            "is_initial": np.asarray(True),
            "prompt": np.asarray(prompt),
            "track_id": np.asarray("black_forest"),
            "source_session_id": np.asarray("session-a"),
            "modalities": np.asarray(MODALITIES),
            "frame_seqs": frame_seqs,
            "frame_input_event_ids": np.asarray(["self-check:window"] * frame_count, dtype=np.str_),
            "source_frames": frame_seqs,
            "sample_indices": np.arange(frame_count, dtype=np.int64),
            "controls": np.zeros((frame_count, len(CONTROL_COLUMNS)), dtype=np.float32),
            "kart_ids": np.zeros(frame_count, dtype=np.int32),
        }
        if window_transport == "npy":
            sidecar_path = input_dir / sidecar_name
            with sidecar_path.open("wb") as handle:
                np.save(handle, gbuffers, allow_pickle=False)
            arrays.update(
                {
                    "window_transport": np.asarray("npy"),
                    "gbuffers_file": np.asarray(sidecar_name),
                    "gbuffers_shape": np.asarray(gbuffers.shape, dtype=np.int64),
                    "gbuffers_dtype": np.asarray("uint8"),
                    "gbuffers_payload_nbytes": np.asarray(gbuffers.nbytes, dtype=np.int64),
                    "gbuffers_file_size_bytes": np.asarray(sidecar_path.stat().st_size, dtype=np.int64),
                }
            )
        else:
            # Omitting the field deliberately covers legacy NPZ compatibility.
            arrays["gbuffers"] = gbuffers
        np.savez(input_dir / immutable_name, **arrays)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "ready": True,
            "window_transport": window_transport,
            "reset_epoch": 7,
            "source_reset_epoch": 3,
            "render_request_id": 11,
            "prompt_revision": 5,
            "prompt_active": True,
            "prompt": prompt,
            "track_id": "black_forest",
            "source_session_id": "session-a",
            "cadence": {
                "mode": "chunks",
                "latent_window": cadence.latent_window,
                "initial_frames": cadence.initial_pixel_frames,
                "subsequent_frames": cadence.resumed_pixel_frames,
            },
            "immutable_file": immutable_name,
            "window_id": 123,
        }
        atomic_write_json(input_dir / "manifest.json", manifest)
        reader = InputReader(
            input_dir,
            control_dir / "live_control.json",
            8,
            8,
            cadence,
            window_transport,
        )
        authority = reader.authority()
        if authority.token is None:
            raise AssertionError(f"self-check authority rejected: {authority.reason}")
        reader.set_active_token(authority.token)
        path = reader.next_window_path(authority.token, None)
        window = reader.load_window(path, authority.token)
        if window.header.window_id != 123 or window.header.source_reset_epoch != 3:
            raise AssertionError("self-check loaded the wrong immutable window")
        if window.header.window_transport != window_transport:
            raise AssertionError("self-check loaded the wrong window transport")
        if window.input_event_ids.tolist() != ["self-check:window"] * frame_count:
            raise AssertionError("self-check lost frame input event IDs in transport")
        orphan_path = input_dir / "window_000000000124.npz"
        orphan_path.write_bytes((input_dir / immutable_name).read_bytes())
        for strict_reader in (
            reader,
            InputReader(
                input_dir,
                control_dir / "live_control.json",
                8,
                8,
                cadence,
                window_transport,
            ),
        ):
            strict_reader.set_active_token(authority.token)
            try:
                strict_reader.next_window_path(authority.token, 123)
            except InputNotReady:
                pass
            else:
                raise AssertionError("unmanifested orphan metadata was adopted as authority")

        resumed_count = cadence.resumed_pixel_frames
        resumed_start = frame_count
        resumed_seqs = np.arange(
            int(frame_seqs[-1]) + 1,
            int(frame_seqs[-1]) + 1 + resumed_count,
            dtype=np.int64,
        )
        resumed_gbuffers = np.zeros((resumed_count, len(MODALITIES), 8, 8, 3), dtype=np.uint8)
        resumed_arrays = {
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
            "reset_epoch": np.asarray(7, dtype=np.int64),
            "source_reset_epoch": np.asarray(3, dtype=np.int64),
            "render_request_id": np.asarray(11, dtype=np.int64),
            "prompt_revision": np.asarray(5, dtype=np.int64),
            "latent_window": np.asarray(cadence.latent_window, dtype=np.int32),
            "initial_pixel_frames": np.asarray(cadence.initial_pixel_frames, dtype=np.int32),
            "resumed_pixel_frames": np.asarray(cadence.resumed_pixel_frames, dtype=np.int32),
            "window_id": np.asarray(124, dtype=np.int64),
            "previous_window_id": np.asarray(123, dtype=np.int64),
            "previous_last_seq": np.asarray(int(frame_seqs[-1]), dtype=np.int64),
            "is_initial": np.asarray(False),
            "prompt": np.asarray(prompt),
            "track_id": np.asarray("black_forest"),
            "source_session_id": np.asarray("session-a"),
            "modalities": np.asarray(MODALITIES),
            "frame_seqs": resumed_seqs,
            "frame_input_event_ids": np.asarray(["self-check:window"] * resumed_count, dtype=np.str_),
            "source_frames": resumed_seqs,
            "sample_indices": np.arange(resumed_start, resumed_start + resumed_count, dtype=np.int64),
            "controls": np.zeros((resumed_count, len(CONTROL_COLUMNS)), dtype=np.float32),
            "kart_ids": np.zeros(resumed_count, dtype=np.int32),
        }
        resumed_sidecar_name = "window_000000000124.gbuffers.npy"
        if window_transport == "npy":
            resumed_sidecar_path = input_dir / resumed_sidecar_name
            with resumed_sidecar_path.open("wb") as handle:
                np.save(handle, resumed_gbuffers, allow_pickle=False)
            resumed_arrays.update(
                {
                    "window_transport": np.asarray("npy"),
                    "gbuffers_file": np.asarray(resumed_sidecar_name),
                    "gbuffers_shape": np.asarray(resumed_gbuffers.shape, dtype=np.int64),
                    "gbuffers_dtype": np.asarray("uint8"),
                    "gbuffers_payload_nbytes": np.asarray(resumed_gbuffers.nbytes, dtype=np.int64),
                    "gbuffers_file_size_bytes": np.asarray(resumed_sidecar_path.stat().st_size, dtype=np.int64),
                }
            )
        else:
            resumed_arrays["gbuffers"] = resumed_gbuffers
        np.savez(orphan_path, **resumed_arrays)

        sequential_manifest = copy.deepcopy(manifest)
        sequential_manifest.update(
            {
                "immutable_file": orphan_path.name,
                "window_id": 124,
                "retained_immutable_files": [immutable_name, orphan_path.name],
                "retained_window_files": [
                    {
                        "immutable_file": immutable_name,
                        "gbuffers_file": sidecar_name if window_transport == "npy" else "",
                    },
                    {
                        "immutable_file": orphan_path.name,
                        "gbuffers_file": resumed_sidecar_name if window_transport == "npy" else "",
                    },
                ],
            }
        )
        atomic_write_json(input_dir / "manifest.json", sequential_manifest)
        sequential_reader = InputReader(
            input_dir,
            control_dir / "live_control.json",
            8,
            8,
            cadence,
            window_transport,
        )
        sequential_reader.set_active_token(authority.token)
        first_authorized_path = sequential_reader.next_window_path(authority.token, None)
        next_authorized_path = sequential_reader.next_window_path(authority.token, 123)
        if first_authorized_path.name != immutable_name:
            raise AssertionError("fresh reader skipped the retained initial window")
        if next_authorized_path.name != orphan_path.name:
            raise AssertionError("fresh reader did not recover the retained successor")

        atomic_write_json(input_dir / "manifest.json", manifest)
        try:
            sequential_reader.next_window_path(authority.token, 123)
        except InputNotReady:
            pass
        else:
            raise AssertionError("successor remained visible after manifest authority was revoked")
        transport_checks = 1
        sidecar_fd_swap_safe = window_transport != "npy"
        sidecar_hardlink_alias_safe = window_transport != "npy"
        if window_transport == "npy":
            if not isinstance(window.gbuffers, np.memmap):
                raise AssertionError("NPY window did not use mmap")
            original_sidecar = sidecar_path.read_bytes()
            missing_path = sidecar_path.with_name(f".{sidecar_path.name}.missing")
            os.replace(sidecar_path, missing_path)
            try:
                try:
                    reader.load_window(path, authority.token)
                except FileNotFoundError:
                    pass
                else:
                    raise AssertionError("missing NPY sidecar was accepted")
            finally:
                missing_path.unlink()
                sidecar_path.write_bytes(original_sidecar)
            if np.any(window.gbuffers):
                raise AssertionError("open NPY mmap did not survive sidecar retention unlink")

            malformed_shape = (frame_count, len(MODALITIES), 8, 4, 6)
            with sidecar_path.open("wb") as handle:
                np.save(
                    handle,
                    np.zeros(malformed_shape, dtype=np.uint8),
                    allow_pickle=False,
                )
            try:
                try:
                    reader.load_window(path, authority.token)
                except ValueError:
                    pass
                else:
                    raise AssertionError("malformed NPY sidecar was accepted")
            finally:
                sidecar_path.write_bytes(original_sidecar)
            try:
                reader._safe_sidecar_path(path, "../escape.gbuffers.npy")
            except ValueError:
                pass
            else:
                raise AssertionError("sidecar path traversal was accepted")

            outside_sidecar = root / "outside.gbuffers.npy"
            with outside_sidecar.open("wb") as handle:
                np.save(
                    handle,
                    np.full_like(gbuffers, 231),
                    allow_pickle=False,
                )
            if outside_sidecar.stat().st_size != sidecar_path.stat().st_size:
                raise AssertionError("sidecar race fixture does not preserve file size")

            symlink_backup = sidecar_path.with_name(f".{sidecar_path.name}.symlink-backup")
            os.replace(sidecar_path, symlink_backup)
            os.symlink(outside_sidecar, sidecar_path)
            try:
                try:
                    reader.load_window(path, authority.token)
                except OSError:
                    pass
                else:
                    raise AssertionError("O_NOFOLLOW accepted an NPY sidecar symlink")
            finally:
                sidecar_path.unlink()
                os.replace(symlink_backup, sidecar_path)

            real_os_open = os.open
            swap_backup = sidecar_path.with_name(f".{sidecar_path.name}.swap-backup")
            swap_state = {"done": False}

            def swap_after_open(
                target: Any,
                flags: int,
                mode: int = 0o777,
                *,
                dir_fd: int | None = None,
            ) -> int:
                if dir_fd is None:
                    fd = real_os_open(target, flags, mode)
                else:
                    fd = real_os_open(target, flags, mode, dir_fd=dir_fd)
                if Path(target) == sidecar_path and not swap_state["done"]:
                    os.replace(sidecar_path, swap_backup)
                    os.symlink(outside_sidecar, sidecar_path)
                    swap_state["done"] = True
                return fd

            os.open = swap_after_open
            try:
                raced_window = reader.load_window(path, authority.token)
            finally:
                os.open = real_os_open
                if sidecar_path.is_symlink():
                    sidecar_path.unlink()
                if swap_backup.exists():
                    os.replace(swap_backup, sidecar_path)
            if not swap_state["done"]:
                raise AssertionError("sidecar swap fixture did not intercept os.open")
            if np.any(raced_window.gbuffers):
                raise AssertionError("sidecar pathname swap replaced the already-open fd")
            sidecar_fd_swap_safe = True

            hardlink_alias = input_dir / "window.gbuffers.npy"
            hardlink_alias.unlink(missing_ok=True)
            real_np_memmap = np.memmap
            hardlink_state: dict[str, Any] = {"linked": False}

            def memmap_during_hardlink_alias(*args: Any, **kwargs: Any) -> np.memmap:
                if not hardlink_state["linked"]:
                    before_alias = sidecar_path.stat()
                    time.sleep(0.002)
                    os.link(sidecar_path, hardlink_alias)
                    after_alias = sidecar_path.stat()
                    hardlink_state.update(
                        {
                            "linked": True,
                            "samefile": os.path.samefile(sidecar_path, hardlink_alias),
                            "before_nlink": before_alias.st_nlink,
                            "after_nlink": after_alias.st_nlink,
                            "ctime_changed": (before_alias.st_ctime_ns != after_alias.st_ctime_ns),
                        }
                    )
                return real_np_memmap(*args, **kwargs)

            np.memmap = memmap_during_hardlink_alias
            try:
                hardlinked_window = reader.load_window(path, authority.token)
            finally:
                np.memmap = real_np_memmap
                hardlink_alias.unlink(missing_ok=True)
            if (
                not hardlink_state["linked"]
                or not hardlink_state["samefile"]
                or hardlink_state["after_nlink"] != hardlink_state["before_nlink"] + 1
                or not hardlink_state["ctime_changed"]
            ):
                raise AssertionError("hardlink alias race fixture was not effective")
            if np.any(hardlinked_window.gbuffers):
                raise AssertionError("hardlink alias changed the mmap payload")
            sidecar_hardlink_alias_safe = True

            window = reader.load_window(path, authority.token)
            if not isinstance(window.gbuffers, np.memmap):
                raise AssertionError("restored NPY sidecar did not mmap")
            transport_checks += 7

        class SlowCompletionStatus(WorkerStatus):
            def update(self, **values: Any) -> None:
                if values.get("status") == "streaming":
                    time.sleep(0.20)
                super().update(**values)

        status = SlowCompletionStatus(control_dir / "worker_status.json")
        publisher = FramePublisher(
            root / "rendered",
            1000.0,
            48,
            2,
            status,
            reader,
            adaptive=True,
            steady_batch_frames=cadence.resumed_pixel_frames,
            source_fps_hint=1000.0,
            target_buffer_frames=0.0,
            min_fps=1.0,
        )
        publisher.reset(authority.token)
        output_frames = [
            OutputFrame(
                rgb=np.zeros((8, 8, 3), dtype=np.uint8),
                source_seq=int(frame_seqs[index]),
                source_frame=int(frame_seqs[index]),
                sample_index=index,
                control={"steer": 0.0, "accel": 0.0, "brake": False, "speed": 0.0, "position": 0},
                kart_id=0,
                window_offset=index,
                window_frame_count=frame_count,
            )
            for index in range(frame_count)
        ]
        input_state_path = control_dir / "input.json"
        first_input_event_id = "self-check:input:1"
        first_input_boundary = int(frame_seqs[2])
        boundary_output_frames = [
            replace(
                frame,
                input_event_id=(
                    first_input_event_id if frame.source_seq >= first_input_boundary else "self-check:input:old"
                ),
            )
            for frame in output_frames
        ]
        atomic_write_json(
            input_state_path,
            {
                "input_event_id": first_input_event_id,
                "boundary_source_seq": first_input_boundary,
                "track_id": authority.token.track_id,
                "source_session_id": authority.token.source_session_id,
                "reset_epoch": authority.token.reset_epoch,
                "source_reset_epoch": authority.token.source_reset_epoch,
            },
        )
        boundary_publisher = FramePublisher(
            root / "boundary-rendered",
            0.0,
            48,
            2,
            status,
            reader,
            input_state_path=input_state_path,
        )
        boundary_publisher.reset(authority.token)
        with boundary_publisher.condition:
            boundary_publisher.queue.extend(
                PublishItem(authority.token, window.header, frame) for frame in boundary_output_frames
            )
            boundary_publisher._refresh_input_boundary_locked(authority.token)
            retained_source_seqs = [item.frame.source_seq for item in boundary_publisher.queue]
            if (
                retained_source_seqs != [int(value) for value in frame_seqs[2:]]
                or boundary_publisher.input_dropped_frames != 2
                or boundary_publisher.input_fast_forward_count != 1
            ):
                raise AssertionError("input boundary did not discard stale queued frames")
            time.sleep(0.001)
            atomic_write_json(
                input_state_path,
                {
                    "input_event_id": "self-check:input:2",
                    "boundary_source_seq": int(frame_seqs[-1]) + 1,
                    "track_id": authority.token.track_id,
                    "source_session_id": authority.token.source_session_id,
                    "reset_epoch": authority.token.reset_epoch,
                    "source_reset_epoch": authority.token.source_reset_epoch,
                },
            )
            boundary_publisher._refresh_input_boundary_locked(authority.token)
            if (
                [item.frame.source_seq for item in boundary_publisher.queue] != [int(value) for value in frame_seqs[2:]]
                or boundary_publisher.input_dropped_frames != 2
                or boundary_publisher.input_fast_forward_count != 1
                or boundary_publisher.active_input_boundary_seq != int(frame_seqs[2])
                or boundary_publisher.pending_input_boundary_seq != int(frame_seqs[-1]) + 1
            ):
                raise AssertionError("a pending input boundary stopped the replacement stream early")
            second_boundary_frame = replace(
                output_frames[-1],
                source_seq=int(frame_seqs[-1]) + 1,
                source_frame=int(frame_seqs[-1]) + 1,
                input_event_id="self-check:input:2",
            )
            boundary_publisher.queue.append(PublishItem(authority.token, window.header, second_boundary_frame))
            boundary_publisher._activate_pending_input_boundary_locked(time.monotonic())
            if (
                [item.frame.source_seq for item in boundary_publisher.queue] != [second_boundary_frame.source_seq]
                or boundary_publisher.input_dropped_frames != frame_count
                or boundary_publisher.input_fast_forward_count != 2
                or boundary_publisher.active_input_boundary_seq != second_boundary_frame.source_seq
                or boundary_publisher.pending_input_boundary_seq != -1
            ):
                raise AssertionError("replacement arrival did not activate the pending input boundary")

        adaptive_boundary_input_path = control_dir / "adaptive-boundary-input.json"
        adaptive_boundary_status = WorkerStatus(control_dir / "adaptive-boundary-worker-status.json")
        adaptive_boundary_publisher = FramePublisher(
            root / "adaptive-boundary-rendered",
            1000.0,
            48,
            2,
            adaptive_boundary_status,
            reader,
            adaptive=True,
            steady_batch_frames=cadence.resumed_pixel_frames,
            source_fps_hint=1000.0,
            target_buffer_frames=0.0,
            min_fps=1.0,
            input_state_path=adaptive_boundary_input_path,
        )
        adaptive_boundary_publisher.reset(authority.token)
        adaptive_output_seq_before = adaptive_boundary_publisher.output_seq
        adaptive_boundary = int(frame_seqs[-1]) + 1
        with adaptive_boundary_publisher.condition:
            adaptive_boundary_publisher.queue.extend(
                PublishItem(authority.token, window.header, frame) for frame in output_frames
            )
            atomic_write_json(
                adaptive_boundary_input_path,
                {
                    "input_event_id": "self-check:adaptive-boundary:1",
                    "boundary_source_seq": adaptive_boundary,
                    "track_id": authority.token.track_id,
                    "source_session_id": authority.token.source_session_id,
                    "reset_epoch": authority.token.reset_epoch,
                    "source_reset_epoch": authority.token.source_reset_epoch,
                },
            )
            adaptive_boundary_publisher.condition.notify_all()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with adaptive_boundary_publisher.condition:
                boundary_processed = (
                    adaptive_boundary_publisher.pending_input_boundary_seq == adaptive_boundary
                    and adaptive_boundary_publisher.active_input_boundary_seq == -1
                    and adaptive_boundary_publisher.input_fast_forward_count == 0
                    and not adaptive_boundary_publisher.queue
                    and adaptive_boundary_publisher.output_seq >= adaptive_output_seq_before + frame_count
                )
                publisher_error = adaptive_boundary_publisher.thread_error
            if boundary_processed or publisher_error:
                break
            time.sleep(0.01)
        with adaptive_boundary_publisher.condition:
            if (
                adaptive_boundary_publisher.thread_error
                or not adaptive_boundary_publisher.thread.is_alive()
                or adaptive_boundary_publisher.queue
                or adaptive_boundary_publisher.input_dropped_frames != 0
                or adaptive_boundary_publisher.input_fast_forward_count != 0
                or adaptive_boundary_publisher.pending_input_boundary_seq != adaptive_boundary
            ):
                raise AssertionError("adaptive publisher did not continue stale frames while replacement was pending")
        adaptive_replacement = replace(
            output_frames[-1],
            source_seq=adaptive_boundary,
            source_frame=adaptive_boundary,
            input_event_id="self-check:adaptive-boundary:1",
        )
        adaptive_boundary_publisher.enqueue(
            authority.token,
            window,
            [adaptive_replacement],
        )
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with adaptive_boundary_publisher.condition:
                replacement_published = (
                    adaptive_boundary_publisher.active_input_boundary_seq == adaptive_boundary
                    and adaptive_boundary_publisher.pending_input_boundary_seq == -1
                    and adaptive_boundary_publisher.input_fast_forward_count == 1
                    and adaptive_boundary_publisher.output_seq >= adaptive_output_seq_before + frame_count + 1
                )
                publisher_error = adaptive_boundary_publisher.thread_error
            if replacement_published or publisher_error:
                break
            time.sleep(0.01)
        with adaptive_boundary_publisher.condition:
            if (
                adaptive_boundary_publisher.thread_error
                or not adaptive_boundary_publisher.thread.is_alive()
                or adaptive_boundary_publisher.active_input_boundary_seq != adaptive_boundary
                or adaptive_boundary_publisher.pending_input_boundary_seq != -1
                or adaptive_boundary_publisher.input_fast_forward_count != 1
            ):
                raise AssertionError("adaptive publisher did not activate the first replacement frame")

        same_event_stale_frame = replace(
            output_frames[-1],
            source_seq=adaptive_boundary + 1,
            source_frame=adaptive_boundary + 1,
            input_event_id="self-check:adaptive-boundary:1",
        )
        same_event_boundary = same_event_stale_frame.source_seq + 1
        same_event_output_seq_before = adaptive_boundary_publisher.output_seq
        with adaptive_boundary_publisher.condition:
            adaptive_boundary_publisher.queue.append(
                PublishItem(authority.token, window.header, same_event_stale_frame)
            )
            atomic_write_json(
                adaptive_boundary_input_path,
                {
                    "input_event_id": "self-check:adaptive-boundary:1",
                    "boundary_source_seq": same_event_boundary,
                    "track_id": authority.token.track_id,
                    "source_session_id": authority.token.source_session_id,
                    "reset_epoch": authority.token.reset_epoch,
                    "source_reset_epoch": authority.token.source_reset_epoch,
                },
            )
            adaptive_boundary_publisher._refresh_input_boundary_locked(authority.token)
            if (
                adaptive_boundary_publisher.active_input_boundary_seq != adaptive_boundary
                or adaptive_boundary_publisher.pending_input_boundary_seq != same_event_boundary
                or [item.frame.source_seq for item in adaptive_boundary_publisher.queue]
                != [same_event_stale_frame.source_seq]
            ):
                raise AssertionError("same-event boundary advance did not preserve the old stream")
            adaptive_boundary_publisher.condition.notify_all()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with adaptive_boundary_publisher.condition:
                stale_frame_continued = (
                    adaptive_boundary_publisher.output_seq >= same_event_output_seq_before + 1
                    and not adaptive_boundary_publisher.queue
                )
                publisher_error = adaptive_boundary_publisher.thread_error
            if stale_frame_continued or publisher_error:
                break
            time.sleep(0.01)
        same_event_replacement = replace(
            same_event_stale_frame,
            source_seq=same_event_boundary,
            source_frame=same_event_boundary,
        )
        adaptive_boundary_publisher.enqueue(
            authority.token,
            window,
            [same_event_replacement],
        )
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with adaptive_boundary_publisher.condition:
                same_event_processed = (
                    adaptive_boundary_publisher.active_input_boundary_seq == same_event_boundary
                    and adaptive_boundary_publisher.pending_input_boundary_seq == -1
                    and adaptive_boundary_publisher.output_seq >= same_event_output_seq_before + 2
                )
                publisher_error = adaptive_boundary_publisher.thread_error
            if same_event_processed or publisher_error:
                break
            time.sleep(0.01)
        with adaptive_boundary_publisher.condition:
            if (
                adaptive_boundary_publisher.thread_error
                or not adaptive_boundary_publisher.thread.is_alive()
                or adaptive_boundary_publisher.queue
                or adaptive_boundary_publisher.active_input_boundary_seq != same_event_boundary
                or adaptive_boundary_publisher.pending_input_boundary_seq != -1
                or adaptive_boundary_publisher.input_fast_forward_count != 1
            ):
                raise AssertionError("adaptive publisher did not defer a same-event boundary until replacement")
        adaptive_boundary_queue_survived = True
        adaptive_same_event_boundary_survived = True
        adaptive_pending_boundary_continued = True
        adaptive_replacement_activated = True

        fatal_status = WorkerStatus(control_dir / "fatal-publisher-worker-status.json")
        fatal_publisher = FramePublisher(
            root / "fatal-publisher-rendered",
            1000.0,
            48,
            2,
            fatal_status,
            reader,
            adaptive=True,
            steady_batch_frames=cadence.resumed_pixel_frames,
            source_fps_hint=1000.0,
            target_buffer_frames=0.0,
            min_fps=1.0,
        )
        fatal_publisher.reset(authority.token)
        assert fatal_publisher.pacer is not None

        def fail_publisher_wait_delay(now: float) -> float:
            del now
            raise RuntimeError("synthetic publisher thread failure")

        with fatal_publisher.condition:
            fatal_publisher.pacer.wait_delay = fail_publisher_wait_delay
            fatal_publisher.queue.append(PublishItem(authority.token, window.header, output_frames[0]))
            fatal_publisher.condition.notify_all()
        fatal_publisher.thread.join(timeout=2.0)
        if fatal_publisher.thread.is_alive():
            raise AssertionError("synthetic publisher failure did not stop its thread")
        fatal_error = fatal_publisher.fatal_error()
        try:
            fatal_publisher.assert_healthy(authority.token)
        except OutputPublishFailed:
            fatal_health_rejected = True
        else:
            fatal_health_rejected = False
        try:
            fatal_publisher.enqueue(authority.token, window, output_frames)
        except OutputPublishFailed:
            fatal_enqueue_rejected = True
        else:
            fatal_enqueue_rejected = False
        if (
            "synthetic publisher thread failure" not in fatal_error
            or not fatal_publisher.failed_for(authority.token)
            or not fatal_health_rejected
            or not fatal_enqueue_rejected
            or fatal_status.get("status") != "failed"
            or fatal_status.get("backlog_eligible") is not False
            or fatal_status.get("publisher_thread_alive") is not False
            or "synthetic publisher thread failure" not in str(fatal_status.get("publisher_thread_error", ""))
        ):
            raise AssertionError("publisher thread failure was not made fatal and externally visible")
        publisher_fatal_guard_verified = True

        watchdog_exit_codes: list[int] = []
        watchdog_status = WorkerStatus(control_dir / "publisher-watchdog-worker-status.json")
        watchdog_publisher = FramePublisher(
            root / "publisher-watchdog-rendered",
            1000.0,
            48,
            2,
            watchdog_status,
            reader,
            adaptive=True,
            steady_batch_frames=cadence.resumed_pixel_frames,
            source_fps_hint=1000.0,
            target_buffer_frames=0.0,
            min_fps=1.0,
            progress_timeout_seconds=0.2,
            watchdog_exit=watchdog_exit_codes.append,
        )
        watchdog_publisher.reset(authority.token)
        assert watchdog_publisher.pacer is not None

        def hold_publisher_without_progress(now: float) -> float:
            del now
            return 60.0

        with watchdog_publisher.condition:
            watchdog_publisher.pacer.wait_delay = hold_publisher_without_progress
            watchdog_publisher.queue.append(PublishItem(authority.token, window.header, output_frames[0]))
            watchdog_publisher.condition.notify_all()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not watchdog_exit_codes:
            time.sleep(0.01)
        watchdog_publisher.watchdog_thread.join(timeout=1.0)
        watchdog_error = watchdog_publisher.fatal_error()
        if (
            watchdog_exit_codes != [70]
            or watchdog_publisher.watchdog_thread.is_alive()
            or "made no output progress" not in watchdog_error
            or watchdog_status.get("status") != "failed"
            or watchdog_status.get("publisher_watchdog_fatal") is not True
            or watchdog_status.get("publisher_thread_alive") is not True
            or watchdog_status.get("backlog_eligible") is not False
        ):
            raise AssertionError("publisher progress watchdog did not fail-fast on queued work without output")

        class FailingWatchdogStatus(WorkerStatus):
            def __init__(self, path: Path):
                self.watchdog_update_attempted = False
                super().__init__(path)

            def update(self, **values: Any) -> None:
                if values.get("publisher_watchdog_fatal"):
                    self.watchdog_update_attempted = True
                    raise RuntimeError("synthetic watchdog status write failure")
                super().update(**values)

        failing_watchdog_exit_codes: list[int] = []
        failing_watchdog_status = FailingWatchdogStatus(control_dir / "failing-watchdog-worker-status.json")
        failing_watchdog_publisher = FramePublisher(
            root / "failing-watchdog-rendered",
            1000.0,
            48,
            2,
            failing_watchdog_status,
            reader,
            adaptive=True,
            steady_batch_frames=cadence.resumed_pixel_frames,
            source_fps_hint=1000.0,
            target_buffer_frames=0.0,
            min_fps=1.0,
            progress_timeout_seconds=0.2,
            watchdog_exit=failing_watchdog_exit_codes.append,
        )
        failing_watchdog_publisher.reset(authority.token)
        assert failing_watchdog_publisher.pacer is not None
        with failing_watchdog_publisher.condition:
            failing_watchdog_publisher.pacer.wait_delay = hold_publisher_without_progress
            failing_watchdog_publisher.queue.append(PublishItem(authority.token, window.header, output_frames[0]))
            failing_watchdog_publisher.condition.notify_all()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not failing_watchdog_exit_codes:
            time.sleep(0.01)
        failing_watchdog_publisher.watchdog_thread.join(timeout=1.0)
        if (
            failing_watchdog_exit_codes != [70]
            or failing_watchdog_publisher.watchdog_thread.is_alive()
            or not failing_watchdog_status.watchdog_update_attempted
        ):
            raise AssertionError("watchdog status failure prevented the mandatory process exit")
        publisher_progress_watchdog_verified = True

        race_input_path = control_dir / "commit-race-input.json"
        race_rendered_dir = root / "commit-race-rendered"
        race_boundary = int(frame_seqs[2])
        atomic_write_json(
            race_input_path,
            {
                "input_event_id": "self-check:commit-race:1",
                "boundary_source_seq": race_boundary,
                "track_id": authority.token.track_id,
                "source_session_id": authority.token.source_session_id,
                "reset_epoch": authority.token.reset_epoch,
                "source_reset_epoch": authority.token.source_reset_epoch,
            },
        )
        race_publisher = FramePublisher(
            race_rendered_dir,
            0.0,
            48,
            2,
            status,
            reader,
            input_state_path=race_input_path,
        )
        race_publisher.reset(authority.token)
        race_output_seq_before = race_publisher.output_seq
        race_frame_index_before = race_publisher.frame_index
        race_item = PublishItem(
            authority.token,
            window.header,
            replace(
                output_frames[2],
                input_event_id="self-check:commit-race:1",
            ),
        )
        race_replacement_frame = replace(
            output_frames[3],
            source_seq=race_boundary + 1,
            source_frame=race_boundary + 1,
            input_event_id="self-check:commit-race:2",
        )
        original_stage_bytes = stage_bytes
        race_injected = False

        def stage_with_commit_race(path: Path, payload: bytes) -> Path:
            nonlocal race_injected
            temporary = original_stage_bytes(path, payload)
            if path == race_rendered_dir / "rgb.rgba8" and not race_injected:
                race_injected = True
                with input_state_commit_lock(race_input_path):
                    atomic_write_json(
                        race_input_path,
                        {
                            "input_event_id": "self-check:commit-race:2",
                            "boundary_source_seq": race_boundary + 1,
                            "track_id": authority.token.track_id,
                            "source_session_id": authority.token.source_session_id,
                            "reset_epoch": authority.token.reset_epoch,
                            "source_reset_epoch": authority.token.source_reset_epoch,
                        },
                    )
                with race_publisher.condition:
                    race_publisher.queue.append(
                        PublishItem(
                            authority.token,
                            window.header,
                            race_replacement_frame,
                        )
                    )
                    race_publisher._refresh_input_boundary_locked(authority.token)
                    race_publisher.queue.clear()
            return temporary

        globals()["stage_bytes"] = stage_with_commit_race
        commit_time_input_race_rejected = False
        try:
            try:
                race_publisher._commit(race_item)
            except FenceChanged:
                commit_time_input_race_rejected = True
        finally:
            globals()["stage_bytes"] = original_stage_bytes
        if (
            not race_injected
            or not commit_time_input_race_rejected
            or (race_rendered_dir / "rgb.rgba8").exists()
            or (race_rendered_dir / "latest.json").exists()
            or race_publisher.output_seq != race_output_seq_before
            or race_publisher.frame_index != race_frame_index_before
            or race_publisher.input_dropped_frames != 1
        ):
            raise AssertionError("commit-time input boundary did not reject the staged stale frame")
        publisher.enqueue(authority.token, window, output_frames)
        resumed_header = WindowHeader(
            path=input_dir / "window_000000000124.npz",
            window_transport=window_transport,
            gbuffers_file="",
            gbuffers_shape=(),
            gbuffers_dtype="",
            gbuffers_payload_nbytes=0,
            gbuffers_file_size_bytes=0,
            schema_version=SCHEMA_VERSION,
            reset_epoch=7,
            source_reset_epoch=3,
            render_request_id=11,
            window_id=124,
            previous_window_id=123,
            previous_last_seq=int(frame_seqs[-1]),
            is_initial=False,
            sample_start=resumed_start,
            sample_end=resumed_start + resumed_count - 1,
            frame_count=resumed_count,
            first_seq=int(resumed_seqs[0]),
            last_seq=int(resumed_seqs[-1]),
            prompt_revision=5,
            track_id="black_forest",
            source_session_id="session-a",
        )
        if reader.warmup_window_path(authority.token) != window.header.path:
            raise AssertionError("warmup did not prefer the retained initial window")
        adjacent = decide_fresh_source(window.header, [window.header, resumed_header])
        if adjacent.required:
            raise AssertionError("an initial window adjacent to latest requested a reset")
        far_header = replace(
            resumed_header,
            path=input_dir / "window_000000000125.npz",
            window_id=125,
            previous_window_id=124,
            previous_last_seq=resumed_header.last_seq,
            sample_start=resumed_header.sample_end + 1,
            sample_end=resumed_header.sample_end + cadence.resumed_pixel_frames,
            first_seq=resumed_header.last_seq + 1,
            last_seq=resumed_header.last_seq + cadence.resumed_pixel_frames,
        )
        stale_initial = decide_fresh_source(window.header, [window.header, resumed_header, far_header])
        if not stale_initial.required or stale_initial.reason != "warmup_stream_not_adjacent_to_latest":
            raise AssertionError("a 73s-style warmup backlog did not request a fresh epoch")
        evicted_initial = decide_fresh_source(resumed_header, [resumed_header, far_header])
        if not evicted_initial.required or evicted_initial.reason != "initial_window_not_retained":
            raise AssertionError("an evicted initial window did not request a fresh epoch")

        adjacent_initial_target = PreviousRecoveryTarget(
            window_id=window.header.window_id,
            visible_source_seq=window.header.last_seq,
            status_window_id=window.header.window_id,
            rendered_window_id=window.header.window_id,
            previous_worker_instance_id="previous-worker",
        )
        adjacent_initial_recovery = decide_fresh_source(
            window.header,
            [window.header, resumed_header],
            adjacent_initial_target,
        )
        if (
            adjacent_initial_recovery.required
            or adjacent_initial_recovery.replay_window_count != 1
            or adjacent_initial_recovery.unpublished_window_count != 1
        ):
            raise AssertionError("adjacent hot restart did not select bounded hidden replay")

        adjacent_resumed_target = replace(
            adjacent_initial_target,
            window_id=resumed_header.window_id,
            visible_source_seq=resumed_header.last_seq,
            status_window_id=resumed_header.window_id,
            rendered_window_id=resumed_header.window_id,
        )
        adjacent_resumed_recovery = decide_fresh_source(
            window.header,
            [window.header, resumed_header, far_header],
            adjacent_resumed_target,
        )
        if (
            adjacent_resumed_recovery.required
            or adjacent_resumed_recovery.replay_window_count != 2
            or adjacent_resumed_recovery.unpublished_window_count != 1
            or far_header.first_seq <= adjacent_resumed_target.visible_source_seq
        ):
            raise AssertionError("initial plus one resumed was not recovered monotonically")
        if not all(
            is_hidden_replay_window(window_id, adjacent_resumed_target)
            for window_id in (window.header.window_id, resumed_header.window_id)
        ) or is_hidden_replay_window(far_header.window_id, adjacent_resumed_target):
            raise AssertionError("hidden replay did not stop at the completed target")
        first_unpublished_frame = OutputFrame(
            rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            source_seq=far_header.first_seq,
            source_frame=far_header.first_seq,
            sample_index=far_header.sample_start,
            control={},
            kart_id=0,
            window_offset=0,
            window_frame_count=far_header.frame_count,
        )
        assert_visible_source_seq_advances([first_unpublished_frame], adjacent_resumed_target.visible_source_seq)
        try:
            assert_visible_source_seq_advances(
                [replace(first_unpublished_frame, source_seq=resumed_header.last_seq)],
                adjacent_resumed_target.visible_source_seq,
            )
        except ContinuityError:
            pass
        else:
            raise AssertionError("visible source_seq rollback was accepted")

        large_backlog_recovery = decide_fresh_source(
            window.header,
            [window.header, resumed_header, far_header],
            adjacent_initial_target,
        )
        if (
            not large_backlog_recovery.required
            or large_backlog_recovery.reason != "previous_recovery_backlog_too_large"
        ):
            raise AssertionError("a large unpublished backlog did not request a fresh epoch")
        original_unpublished_headers = reader._compatible_headers
        reader._compatible_headers = lambda _token, _manifest=None: (
            [window.header, resumed_header, far_header],
            "",
        )
        try:
            if reader.unpublished_window_count(authority.token, window.header.window_id) != 2:
                raise AssertionError("recovery backlog growth was not re-detected")
        finally:
            reader._compatible_headers = original_unpublished_headers
        large_replay_target = replace(
            adjacent_initial_target,
            window_id=far_header.window_id,
            visible_source_seq=far_header.last_seq,
            status_window_id=far_header.window_id,
            rendered_window_id=far_header.window_id,
        )
        large_replay_recovery = decide_fresh_source(
            window.header,
            [window.header, resumed_header, far_header],
            large_replay_target,
        )
        if not large_replay_recovery.required or large_replay_recovery.reason != "previous_recovery_replay_too_large":
            raise AssertionError("a three-window hidden replay was accepted")

        recovery_rendered_dir = root / "recovery-rendered"
        recovery_rendered_dir.mkdir()
        rendered_recovery = {
            "schema_version": SCHEMA_VERSION,
            "reset_epoch": authority.token.reset_epoch,
            "source_reset_epoch": authority.token.source_reset_epoch,
            "render_request_id": authority.token.request_id,
            "prompt_revision": authority.token.prompt_revision,
            "prompt_hash": authority.token.prompt_hash,
            "track_id": authority.token.track_id,
            "source_session_id": authority.token.source_session_id,
            "window_id": resumed_header.window_id,
            "source_seq": resumed_header.last_seq,
            "window_complete": True,
        }
        atomic_write_json(recovery_rendered_dir / "latest.json", rendered_recovery)
        fake_previous_status = object.__new__(WorkerStatus)
        fake_previous_status.instance_id = "new-worker"
        fake_previous_status.previous = {
            "schema_version": SCHEMA_VERSION,
            "worker_instance_id": "previous-worker",
            "status": "streaming",
            "ready": True,
            "prompt": authority.token.prompt,
            "prompt_hash": authority.token.prompt_hash,
            "reset_epoch": authority.token.reset_epoch,
            "source_reset_epoch": authority.token.source_reset_epoch,
            "completed_request_id": authority.token.request_id,
            "prompt_revision": authority.token.prompt_revision,
            "track_id": authority.token.track_id,
            "source_session_id": authority.token.source_session_id,
            "completed_window_id": window.header.window_id,
            "completed_source_seq": window.header.last_seq,
        }
        recovered_target = previous_recovery_target(fake_previous_status, authority.token, recovery_rendered_dir)
        if (
            recovered_target is None
            or recovered_target.window_id != resumed_header.window_id
            or recovered_target.visible_source_seq != resumed_header.last_seq
        ):
            raise AssertionError("rendered output did not fence the recovery target")
        new_schema_previous = dict(fake_previous_status.previous)
        legacy_previous = dict(new_schema_previous)
        legacy_previous.pop("prompt_hash")
        legacy_previous.pop("completed_source_seq")
        fake_previous_status.previous = legacy_previous
        legacy_rendered_target = previous_recovery_target(fake_previous_status, authority.token, recovery_rendered_dir)
        if (
            legacy_rendered_target is None
            or legacy_rendered_target.window_id != resumed_header.window_id
            or legacy_rendered_target.visible_source_seq != resumed_header.last_seq
            or legacy_rendered_target.status_window_id != -1
        ):
            raise AssertionError("baseline status did not use rendered-authoritative recovery")
        legacy_accepted_only = dict(legacy_previous)
        legacy_accepted_only["accepted_request_id"] = authority.token.request_id
        legacy_accepted_only["completed_request_id"] = 0
        fake_previous_status.previous = legacy_accepted_only
        if previous_recovery_target(fake_previous_status, authority.token, recovery_rendered_dir) is None:
            raise AssertionError("legacy accepted_request_id did not authorize rendered recovery")
        fake_previous_status.previous = legacy_previous

        mismatched_rendered = dict(rendered_recovery)
        mismatched_rendered["prompt_hash"] = "mismatched-rendered-token"
        atomic_write_json(recovery_rendered_dir / "latest.json", mismatched_rendered)
        if previous_recovery_target(fake_previous_status, authority.token, recovery_rendered_dir) is not None:
            raise AssertionError("legacy status authorized a mismatched rendered token")
        atomic_write_json(recovery_rendered_dir / "latest.json", rendered_recovery)

        request_mismatch_legacy = dict(legacy_previous)
        token_request_mismatch = authority.token.request_id + 100
        request_mismatch_legacy["accepted_request_id"] = token_request_mismatch
        request_mismatch_legacy["completed_request_id"] = token_request_mismatch
        fake_previous_status.previous = request_mismatch_legacy
        if previous_recovery_target(fake_previous_status, authority.token, recovery_rendered_dir) is not None:
            raise AssertionError("legacy status authorized a mismatched request")
        fake_previous_status.previous = new_schema_previous

        async_lag_previous = dict(fake_previous_status.previous)
        async_lag_previous.update(
            {
                "status": "output_queued",
                "accepted_request_id": authority.token.request_id,
                "completed_request_id": 0,
                "completed_window_id": -1,
                "completed_source_seq": -1,
            }
        )
        fake_previous_status.previous = async_lag_previous
        async_lag_target = previous_recovery_target(fake_previous_status, authority.token, recovery_rendered_dir)
        if async_lag_target is None or async_lag_target.window_id != resumed_header.window_id:
            raise AssertionError("async completion-status lag lost visible output")
        fake_previous_status.previous = {
            **async_lag_previous,
            "completed_request_id": authority.token.request_id,
            "completed_window_id": window.header.window_id,
            "completed_source_seq": window.header.last_seq,
        }
        partial_rendered = {
            **rendered_recovery,
            "source_seq": resumed_header.first_seq,
            "window_complete": False,
        }
        atomic_write_json(recovery_rendered_dir / "latest.json", partial_rendered)
        partial_target = previous_recovery_target(fake_previous_status, authority.token, recovery_rendered_dir)
        partial_decision = decide_fresh_source(
            window.header,
            [window.header, resumed_header],
            partial_target,
        )
        if (
            partial_target is None
            or not partial_decision.required
            or partial_decision.reason != "previous_recovery_visible_source_mismatch"
        ):
            raise AssertionError("partially visible old output did not force a fresh epoch")
        atomic_write_json(recovery_rendered_dir / "latest.json", rendered_recovery)
        rendered_only_previous = dict(fake_previous_status.previous)
        rendered_only_previous["status"] = "waiting_for_input"
        rendered_only_previous["completed_window_id"] = -1
        rendered_only_previous["completed_source_seq"] = -1
        fake_previous_status.previous = rendered_only_previous
        rendered_only_target = previous_recovery_target(fake_previous_status, authority.token, recovery_rendered_dir)
        if rendered_only_target is None or rendered_only_target.window_id != resumed_header.window_id:
            raise AssertionError("a second restart lost the visible recovery target")
        mismatched_previous = dict(fake_previous_status.previous)
        mismatched_previous["prompt"] = "wrong prompt"
        fake_previous_status.previous = mismatched_previous
        if previous_recovery_target(fake_previous_status, authority.token, recovery_rendered_dir) is not None:
            raise AssertionError("mismatched previous token authorized hidden replay")
        original_compatible_headers = reader._compatible_headers
        reader._compatible_headers = lambda _token, _manifest=None: ([far_header], "")
        try:
            if reader.warmup_window_path(authority.token) != far_header.path:
                raise AssertionError("warmup could not use the latest retained resumed window")
        finally:
            reader._compatible_headers = original_compatible_headers
        handshake = FreshEpochHandshake(
            token_key=authority.token.key(),
            source_reset_epoch=authority.token.source_reset_epoch,
            request_id="self-check-worker:3",
            requested_at=100.0,
            requested_monotonic=100.0,
            next_status_at=100.0 + DEFAULT_FRESH_EPOCH_TIMEOUT,
            decision=evicted_initial,
        )
        if handshake.next_status_at != 120.0:
            raise AssertionError("fresh epoch status handshake is not 20 seconds")
        resumed_window = WindowData(
            header=resumed_header,
            gbuffers=np.empty(0),
            frame_seqs=resumed_seqs,
            input_event_ids=np.full(resumed_count, "", dtype=np.str_),
            source_frames=resumed_seqs,
            sample_indices=np.arange(resumed_start, resumed_start + resumed_count, dtype=np.int64),
            controls=np.zeros((resumed_count, len(CONTROL_COLUMNS)), dtype=np.float32),
            kart_ids=np.zeros(resumed_count, dtype=np.int32),
            prompt=prompt,
        )

        class FakeWarmupRuntime:
            def __init__(self) -> None:
                self.warmed = False
                self.cadence = cadence
                self.device = torch.device("cpu")
                self.calls: list[tuple[int, bool, int]] = []
                self.reset_count = 0

            def reset_session(self, _token: SessionToken) -> None:
                self.reset_count += 1

            def _run_chunk(
                self,
                pixels: np.ndarray,
                _prompt: str,
                is_initial: bool,
                boundary_source_seq: int,
                _fence: Callable[[], None],
            ) -> None:
                self.calls.append((pixels.shape[0], is_initial, boundary_source_seq))

        fake_runtime = FakeWarmupRuntime()
        fake_warmup_window = WindowData(
            header=resumed_header,
            gbuffers=np.zeros((resumed_count, len(MODALITIES), 8, 8, 3), dtype=np.uint8),
            frame_seqs=resumed_window.frame_seqs,
            input_event_ids=resumed_window.input_event_ids,
            source_frames=resumed_window.source_frames,
            sample_indices=resumed_window.sample_indices,
            controls=resumed_window.controls,
            kart_ids=resumed_window.kart_ids,
            prompt=prompt,
        )
        real_synchronize = torch.cuda.synchronize
        torch.cuda.synchronize = lambda _device=None: None
        try:
            ModelRuntime.warmup(
                fake_runtime,
                fake_warmup_window,
                authority.token,
                lambda: None,
            )
        finally:
            torch.cuda.synchronize = real_synchronize
        if fake_runtime.calls != [
            (cadence.initial_pixel_frames, True, -1),
            (
                cadence.resumed_pixel_frames,
                False,
                int(resumed_seqs[cadence.initial_pixel_frames - 1]),
            ),
        ]:
            raise AssertionError("resumed-window warmup did not compile both fixed shapes")
        if not fake_runtime.warmed or fake_runtime.reset_count != 2:
            raise AssertionError("warmup did not leave a reset, resident compiled runtime")
        resumed_frames = [
            OutputFrame(
                rgb=np.zeros((8, 8, 3), dtype=np.uint8),
                source_seq=int(resumed_seqs[index]),
                source_frame=int(resumed_seqs[index]),
                sample_index=resumed_start + index,
                control={
                    "steer": 0.0,
                    "accel": 0.0,
                    "brake": False,
                    "speed": 0.0,
                    "position": 0,
                },
                kart_id=0,
                window_offset=index,
                window_frame_count=resumed_count,
            )
            for index in range(resumed_count)
        ]
        publisher.enqueue(authority.token, resumed_window, resumed_frames)
        if not publisher.wait_for_window(authority.token, resumed_header.window_id, timeout=2.0):
            raise AssertionError("self-check publisher did not persist the resumed window")
        with publisher.condition:
            publisher_times = list(publisher.publish_times)
            status_write_ms = publisher.status_max_write_seconds * 1000.0
        publisher_gaps = [later - earlier for earlier, later in zip(publisher_times, publisher_times[1:])]
        boundary_gap_ms = publisher_gaps[frame_count - 1] * 1000.0
        if status_write_ms < 180.0:
            raise AssertionError("self-check did not inject a 200ms completion-status write")
        if boundary_gap_ms >= 80.0:
            raise AssertionError(f"completion-status write blocked frame publishing for {boundary_gap_ms:.1f}ms")
        latest = read_json(root / "rendered/latest.json")
        worker_state = read_json(control_dir / "worker_status.json")
        if (
            latest.get("schema_version") != SCHEMA_VERSION
            or latest.get("source_reset_epoch") != 3
            or latest.get("window_id") != 124
            or latest.get("publisher_mode") != "adaptive"
            or worker_state.get("completed_request_id") != 11
            or worker_state.get("completed_window_id") != 124
        ):
            raise AssertionError("self-check publisher metadata/ack fence is invalid")
        other_latent_window = 2 if cadence.latent_window == 4 else 4
        other_cadence = cadence_for_latent_window(other_latent_window)
        mismatched_manifest = dict(manifest)
        mismatched_manifest["cadence"] = {
            "mode": "chunks",
            "latent_window": other_cadence.latent_window,
            "initial_frames": other_cadence.initial_pixel_frames,
            "subsequent_frames": other_cadence.resumed_pixel_frames,
        }
        atomic_write_json(input_dir / "manifest.json", mismatched_manifest)
        mismatched = reader.authority()
        if mismatched.token is not None or not mismatched.reason.startswith("temporal_cadence_mismatch_"):
            raise AssertionError("self-check accepted a mismatched temporal cadence")
        atomic_write_json(input_dir / "manifest.json", manifest)
        other_transport = "npz" if window_transport == "npy" else "npy"
        mismatched_transport_manifest = dict(manifest)
        mismatched_transport_manifest["window_transport"] = other_transport
        atomic_write_json(input_dir / "manifest.json", mismatched_transport_manifest)
        mismatched_transport = reader.authority()
        if mismatched_transport.token is not None or not mismatched_transport.reason.startswith(
            "window_transport_mismatch_"
        ):
            raise AssertionError("self-check accepted a mismatched window transport")
        atomic_write_json(input_dir / "manifest.json", manifest)
        transport_checks += 1

        cache_reader = InputReader(
            input_dir,
            control_dir / "live_control.json",
            8,
            8,
            cadence,
            window_transport,
        )
        cache_reader.set_active_token(authority.token)
        cache_identity = (1, 1, 1, 1)
        for index in range(50_000):
            cache_reader._remember_header(
                f"window_{index:012d}.npz",
                cache_identity,
                copy.copy(window.header),
            )
        if cache_reader.header_cache_size() != HEADER_CACHE_LIMIT:
            raise AssertionError("header LRU cache exceeded its fixed entry limit")
        header_cache_peak_entries = cache_reader.header_cache_size()
        changed_token = SessionToken(
            reset_epoch=authority.token.reset_epoch + 1,
            source_reset_epoch=authority.token.source_reset_epoch,
            request_id=authority.token.request_id,
            prompt_revision=authority.token.prompt_revision,
            prompt_hash=authority.token.prompt_hash,
            prompt=authority.token.prompt,
            track_id=authority.token.track_id,
            source_session_id=authority.token.source_session_id,
        )
        cache_reader.set_active_token(changed_token)
        if cache_reader.header_cache_size() != 0:
            raise AssertionError("header cache survived an active token change")
        cache_reader._remember_header("window_after_reset.npz", cache_identity, window.header)
        cache_reader.set_active_token(None)
        if cache_reader.header_cache_size() != 0:
            raise AssertionError("header cache survived an active token reset")

        live["reset_epoch"] = 8
        atomic_write_json(control_dir / "live_control.json", live)
        changed = reader.authority()
        if changed.token is not None or changed.reason != "waiting_for_reset_generation":
            raise AssertionError("self-check did not reject an obsolete input generation")
    print(
        json.dumps(
            {
                "ok": True,
                "schema_version": SCHEMA_VERSION,
                "checks": 27,
                "strict_manifest_sequence": True,
                "startup_recovery_checks": 24,
                "fresh_epoch_timeout_seconds": DEFAULT_FRESH_EPOCH_TIMEOUT,
                "transport_checks": transport_checks,
                "header_cache_stress_windows": 50_000,
                "header_cache_limit": HEADER_CACHE_LIMIT,
                "header_cache_peak_entries": header_cache_peak_entries,
                "header_cache_entries": cache_reader.header_cache_size(),
                "sidecar_fd_swap_safe": sidecar_fd_swap_safe,
                "sidecar_hardlink_alias_safe": sidecar_hardlink_alias_safe,
                "commit_time_input_race_rejected": commit_time_input_race_rejected,
                "adaptive_boundary_queue_survived": adaptive_boundary_queue_survived,
                "adaptive_same_event_boundary_survived": (adaptive_same_event_boundary_survived),
                "adaptive_pending_boundary_continued": (adaptive_pending_boundary_continued),
                "adaptive_replacement_activated": adaptive_replacement_activated,
                "publisher_fatal_guard_verified": publisher_fatal_guard_verified,
                "publisher_progress_watchdog_verified": (publisher_progress_watchdog_verified),
                "window_transport": window_transport,
                "latent_window": cadence.latent_window,
                "pixel_cadence": [
                    cadence.initial_pixel_frames,
                    cadence.resumed_pixel_frames,
                ],
                "async_status_boundary_gap_ms": round(boundary_gap_ms, 3),
                "async_status_write_ms": round(status_write_ms, 3),
                **adaptive_results,
            }
        )
    )
    return 0


def run() -> int:
    args = parse_args()
    if args.self_check:
        if args.window_transport == SHM_SLOT_TRANSPORT:
            print(json.dumps(run_shm_slot_self_check(), sort_keys=True))
            return 0
        return run_self_check(args.temporal_cadence, args.window_transport)
    args.control_dir.mkdir(parents=True, exist_ok=True)
    args.input_dir.mkdir(parents=True, exist_ok=True)
    args.rendered_dir.mkdir(parents=True, exist_ok=True)
    gc_telemetry = CyclicGCTelemetry()
    loop_telemetry = WorkerLoopTelemetry()
    status = WorkerStatus(args.worker_status)
    reader = InputReader(
        args.input_dir,
        args.live_control,
        args.height,
        args.width,
        args.temporal_cadence,
        args.window_transport,
    )
    publisher = FramePublisher(
        args.rendered_dir,
        args.publish_fps,
        args.publish_queue_frames,
        args.publish_attempts,
        status,
        reader,
        adaptive=args.adaptive_publish,
        steady_batch_frames=args.temporal_cadence.resumed_pixel_frames,
        source_fps_hint=args.publish_source_fps,
        target_buffer_frames=args.publish_buffer_frames,
        min_fps=args.publish_min_fps,
        input_state_path=args.input_state,
        gc_telemetry=gc_telemetry,
    )
    runtime = ModelRuntime(args)
    try:
        status.update(
            status="loading",
            ready=False,
            error="",
            backlog_eligible=False,
            needs_fresh_source=False,
            fresh_epoch_handshake_state="loading",
            latent_window=args.temporal_cadence.latent_window,
            window_transport=args.window_transport,
            initial_pixel_frames=args.temporal_cadence.initial_pixel_frames,
            resumed_pixel_frames=args.temporal_cadence.resumed_pixel_frames,
            publish_fps=args.publish_fps,
            publish_queue_frames=args.publish_queue_frames,
            adaptive_publish=args.adaptive_publish,
            publish_source_fps=args.publish_source_fps,
            publish_buffer_frames=args.publish_buffer_frames,
            publish_min_fps=args.publish_min_fps,
            input_state_path=str(args.input_state) if args.input_state else "",
        )
        runtime.load()
        gc_telemetry.install()
    except Exception as error:
        status.update(
            status="load_failed",
            ready=False,
            error=f"{error}\n{traceback.format_exc()}"[-4000:],
            backlog_eligible=False,
            needs_fresh_source=False,
            compile_mode=runtime.compile_mode,
        )
        raise

    status.update(
        status="idle",
        ready=True,
        error=runtime.compile_fallback_reason,
        backlog_eligible=False,
        needs_fresh_source=False,
        fresh_epoch_handshake_state="idle",
        fresh_epoch_timeout_seconds=args.fresh_epoch_timeout,
        compile_mode=runtime.compile_mode,
        checkpoint=str(args.checkpoint),
        window_transport=args.window_transport,
        small_encoder=str(args.small_encoder),
        tiny_decoder=str(args.tiny_decoder),
        device=str(runtime.device),
        eager_fallback_enabled=args.eager_fallback,
        warmed=runtime.warmed,
        **runtime.model_timing_telemetry(),
        **loop_telemetry.telemetry(),
        **gc_telemetry.telemetry(),
    )

    active_token: SessionToken | None = None
    blocked_token: tuple[Any, ...] | None = None
    fresh_handshake: FreshEpochHandshake | None = None
    fresh_epoch_requested_once = False
    recovery_target: PreviousRecoveryTarget | None = None
    recovered_target_pending_publish: PreviousRecoveryTarget | None = None
    recovery_visible_source_seq = -1
    startup_plan_checked = False
    startup_reference_header: WindowHeader | None = None
    stream_started = False
    last_idle_status = ""
    last_idle_write = 0.0

    while True:
        authority_started_at = time.monotonic()
        authority = reader.authority()
        authority_finished_at = time.monotonic()
        authority_observed_at_unix = time.time()
        authority_duration = authority_finished_at - authority_started_at
        loop_telemetry.record_authority(
            authority_duration,
            authority_observed_at_unix,
        )
        token = authority.token
        now = time.monotonic()
        if token is None:
            reader.set_active_token(None)
            if active_token is not None:
                active_token = None
                blocked_token = None
                recovery_target = None
                recovered_target_pending_publish = None
                recovery_visible_source_seq = -1
                startup_plan_checked = False
                startup_reference_header = None
                stream_started = False
                loop_telemetry.reset()
                publisher.reset(None)
            if authority.reason != last_idle_status or now - last_idle_write >= 2.0:
                if fresh_handshake is not None:
                    waited = max(0.0, now - fresh_handshake.requested_monotonic)
                    idle_name = "waiting_for_fresh_epoch"
                    needs_fresh_source = True
                    handshake_state = "waiting_for_manifest"
                else:
                    waited = 0.0
                    idle_name = "idle" if authority.reason == "no_prompt" else authority.reason
                    needs_fresh_source = False
                    handshake_state = "idle"
                status.update(
                    prompt=clean_prompt(authority.live.get("prompt")),
                    status=idle_name,
                    error="",
                    ready=True,
                    backlog_eligible=False,
                    needs_fresh_source=needs_fresh_source,
                    fresh_epoch_handshake_state=handshake_state,
                    fresh_epoch_wait_seconds=round(waited, 3),
                    compile_mode=runtime.compile_mode,
                    reset_epoch=as_int(authority.live.get("reset_epoch"), 0),
                    output_fps=0.0,
                    warmed=runtime.warmed,
                    **loop_telemetry.telemetry(),
                    **gc_telemetry.telemetry(),
                )
                last_idle_status = authority.reason
                last_idle_write = now
            time.sleep(args.poll_interval)
            continue

        reader.set_active_token(token)
        if active_token is None or active_token.key() != token.key():
            handshake_state = "idle"
            handshake_latency_ms = 0.0
            handshake_acknowledged = False
            if fresh_handshake is not None:
                same_request = (
                    token.reset_epoch == fresh_handshake.token_key[0]
                    and token.request_id == fresh_handshake.token_key[2]
                    and token.prompt_revision == fresh_handshake.token_key[3]
                    and token.track_id == fresh_handshake.token_key[5]
                    and token.source_session_id == fresh_handshake.token_key[6]
                )
                handshake_acknowledged = bool(
                    same_request and token.source_reset_epoch > fresh_handshake.source_reset_epoch
                )
                handshake_state = "acknowledged" if handshake_acknowledged else "superseded"
                handshake_latency_ms = round(
                    max(0.0, (time.time() - fresh_handshake.requested_at) * 1000.0),
                    1,
                )
                fresh_handshake = None
            active_token = token
            blocked_token = None
            stream_started = False
            runtime.reset_session(token)
            loop_telemetry.reset()
            loop_telemetry.record_authority(
                authority_duration,
                authority_observed_at_unix,
            )
            publisher.reset(token)
            recovery_target = previous_recovery_target(status, token, args.rendered_dir)
            recovered_target_pending_publish = None
            recovery_visible_source_seq = recovery_target.visible_source_seq if recovery_target is not None else -1
            startup_plan_checked = False
            startup_reference_header = None
            status.update(
                prompt=token.prompt,
                prompt_hash=token.prompt_hash,
                status=("recovering" if recovery_target is not None else "waiting_for_input"),
                error="",
                ready=True,
                backlog_eligible=False,
                needs_fresh_source=False,
                fresh_epoch_handshake_state=handshake_state,
                fresh_epoch_acknowledged=handshake_acknowledged,
                fresh_epoch_ack_latency_ms=handshake_latency_ms,
                compile_mode=runtime.compile_mode,
                reset_epoch=token.generation,
                source_reset_epoch=token.source_reset_epoch,
                prompt_revision=token.prompt_revision,
                track_id=token.track_id,
                source_session_id=token.source_session_id,
                accepted_request_id=0,
                completed_request_id=(token.request_id if recovery_target is not None else 0),
                accepted_window_id=-1,
                completed_window_id=(recovery_target.window_id if recovery_target is not None else -1),
                completed_source_seq=(recovery_target.visible_source_seq if recovery_target is not None else -1),
                output_fps=0.0,
                warmed=runtime.warmed,
                recovery_target_window_id=(recovery_target.window_id if recovery_target is not None else -1),
                recovery_previous_worker_instance_id=(
                    recovery_target.previous_worker_instance_id if recovery_target is not None else ""
                ),
                **loop_telemetry.telemetry(),
                **gc_telemetry.telemetry(),
            )

        if fresh_handshake is not None and fresh_handshake.token_key == token.key():
            if now >= fresh_handshake.next_status_at:
                fresh_handshake.retry_count += 1
                waited = max(0.0, now - fresh_handshake.requested_monotonic)
                status.update(
                    prompt=token.prompt,
                    status="waiting_for_fresh_epoch",
                    error="",
                    ready=True,
                    backlog_eligible=False,
                    needs_fresh_source=True,
                    fresh_epoch_handshake_state="awaiting_bridge",
                    fresh_epoch_request_id=fresh_handshake.request_id,
                    fresh_epoch_requested_source_reset_epoch=(fresh_handshake.source_reset_epoch),
                    fresh_epoch_wait_seconds=round(waited, 3),
                    fresh_epoch_retry_count=fresh_handshake.retry_count,
                    fresh_epoch_timed_out=True,
                    warmed=runtime.warmed,
                )
                fresh_handshake.next_status_at = now + args.fresh_epoch_timeout
            time.sleep(args.poll_interval)
            continue

        publisher_fatal_error = publisher.fatal_error()
        if publisher_fatal_error:
            loop_telemetry.finish_input_not_ready_wait(
                time.monotonic(),
                time.time(),
            )
            status.update(
                status="failed",
                error=publisher_fatal_error,
                ready=True,
                backlog_eligible=False,
                needs_fresh_source=False,
                output_fps=0.0,
                publisher_thread_alive=False,
                publisher_thread_error=publisher_fatal_error,
                **loop_telemetry.telemetry(),
                **gc_telemetry.telemetry(),
            )
            return 70
        if blocked_token == token.key():
            time.sleep(max(args.poll_interval, 0.25))
            continue
        if publisher.failed_for(token):
            loop_telemetry.finish_input_not_ready_wait(
                time.monotonic(),
                time.time(),
            )
            blocked_token = token.key()
            time.sleep(max(args.poll_interval, 0.25))
            continue

        try:
            next_window_path_started_at = time.monotonic()
            try:
                next_path = (
                    reader.warmup_window_path(token)
                    if not runtime.warmed
                    else reader.next_window_path(token, runtime.last_window_id)
                )
            finally:
                loop_telemetry.record_next_window_path(
                    time.monotonic() - next_window_path_started_at,
                    time.time(),
                )
            load_window_started_at = time.monotonic()
            try:
                window = reader.load_window(
                    next_path,
                    token,
                    direct_shared_slot=(runtime.warmed and args.window_transport == SHM_SLOT_TRANSPORT),
                )
            finally:
                loop_telemetry.record_load_window(
                    time.monotonic() - load_window_started_at,
                    time.time(),
                )
            runtime.stage_shared_slot_window(
                window,
                lambda: reader.assert_fence(token),
            )
            loop_telemetry.finish_input_not_ready_wait(
                time.monotonic(),
                time.time(),
                window.header.window_id,
            )
            reader.assert_fence(token)
            publisher.assert_healthy(token)
            if startup_reference_header is None:
                startup_reference_header = window.header
            if recovered_target_pending_publish is not None:
                unpublished_count = reader.unpublished_window_count(token, recovered_target_pending_publish.window_id)
                if unpublished_count > MAX_STARTUP_UNPUBLISHED_WINDOWS:
                    recovery_target = recovered_target_pending_publish
                    recovered_target_pending_publish = None
                    startup_plan_checked = False
        except InputNotReady:
            loop_telemetry.record_input_not_ready(time.monotonic())
            time.sleep(args.poll_interval)
            continue
        except ContinuityError as error:
            loop_telemetry.finish_input_not_ready_wait(
                time.monotonic(),
                time.time(),
            )
            blocked_token = token.key()
            status.update(
                prompt=token.prompt,
                status="needs_reset",
                error=str(error),
                ready=True,
                backlog_eligible=False,
                needs_fresh_source=False,
                compile_mode=runtime.compile_mode,
                reset_epoch=token.generation,
                source_reset_epoch=token.source_reset_epoch,
                **loop_telemetry.telemetry(),
                **gc_telemetry.telemetry(),
            )
            continue
        except FenceChanged:
            loop_telemetry.finish_input_not_ready_wait(
                time.monotonic(),
                time.time(),
            )
            continue
        except (ValueError, KeyError, OSError) as error:
            loop_telemetry.finish_input_not_ready_wait(
                time.monotonic(),
                time.time(),
            )
            blocked_token = token.key()
            status.update(
                prompt=token.prompt,
                status="needs_reset",
                error=f"immutable input rejected: {error}",
                ready=True,
                backlog_eligible=False,
                needs_fresh_source=False,
                compile_mode=runtime.compile_mode,
                reset_epoch=token.generation,
                source_reset_epoch=token.source_reset_epoch,
                prompt_revision=token.prompt_revision,
                track_id=token.track_id,
                source_session_id=token.source_session_id,
                **loop_telemetry.telemetry(),
                **gc_telemetry.telemetry(),
            )
            continue
        except Exception as error:
            loop_telemetry.finish_input_not_ready_wait(
                time.monotonic(),
                time.time(),
            )
            blocked_token = token.key()
            status.update(
                prompt=token.prompt,
                status="failed",
                error=f"{error}\n{traceback.format_exc()}"[-4000:],
                ready=True,
                backlog_eligible=False,
                needs_fresh_source=False,
                compile_mode=runtime.compile_mode,
                reset_epoch=token.generation,
                source_reset_epoch=token.source_reset_epoch,
                prompt_revision=token.prompt_revision,
                track_id=token.track_id,
                source_session_id=token.source_session_id,
                **loop_telemetry.telemetry(),
                **gc_telemetry.telemetry(),
            )
            continue

        pre_model_status_update_started_at = time.monotonic()
        try:
            status.update(
                prompt=token.prompt,
                status=(
                    "warming"
                    if not runtime.warmed
                    else "recovering"
                    if (recovery_target is not None or recovered_target_pending_publish is not None)
                    else "rendering"
                ),
                error="",
                ready=True,
                backlog_eligible=(
                    stream_started and recovery_target is None and recovered_target_pending_publish is None
                ),
                needs_fresh_source=False,
                fresh_epoch_handshake_state=(
                    "warming"
                    if not runtime.warmed
                    else "recovering"
                    if (recovery_target is not None or recovered_target_pending_publish is not None)
                    else "streaming"
                ),
                compile_mode=runtime.compile_mode,
                reset_epoch=token.generation,
                source_reset_epoch=token.source_reset_epoch,
                accepted_request_id=token.request_id,
                accepted_window_id=window.header.window_id,
                accepted_immutable_file=window.header.path.name,
                attempt=0,
            )
        finally:
            loop_telemetry.record_pre_model_status_update(
                time.monotonic() - pre_model_status_update_started_at,
                time.time(),
            )

        success = False
        fresh_requested_now = False
        hidden_replay = False
        last_error = ""
        for attempt in range(1, args.max_attempts + 1):
            window_cycle_started_at = time.monotonic()
            snapshot_started_at = time.monotonic()
            loop_telemetry.record_snapshot_start(
                snapshot_started_at,
                window.header.window_id,
                time.time(),
            )
            snapshot = runtime.snapshot()
            runtime.record_model_snapshot(time.monotonic() - snapshot_started_at)
            model_complete = False
            try:
                reader.assert_fence(token)
                if not runtime.warmed:
                    runtime.warmup(window, token, lambda: reader.assert_fence(token))
                    gc_telemetry.apply_steady_state_freeze_once()
                    gc_telemetry.reset()
                    window_cycle_started_at = time.monotonic()
                    snapshot_started_at = time.monotonic()
                    loop_telemetry.record_snapshot_start(
                        snapshot_started_at,
                        window.header.window_id,
                        time.time(),
                    )
                    snapshot = runtime.snapshot()
                    runtime.record_model_snapshot(time.monotonic() - snapshot_started_at)
                    reader.assert_fence(token)
                if not startup_plan_checked:
                    decision = reader.fresh_source_decision(
                        token,
                        startup_reference_header or window.header,
                        recovery_target,
                    )
                    if decision.required:
                        if fresh_epoch_requested_once:
                            raise ContinuityError(
                                "warmed runtime still lacks a current initial window after its "
                                f"one fresh-source request: {decision.reason}"
                            )
                        requested_at = time.time()
                        requested_monotonic = time.monotonic()
                        request_id = f"{status.instance_id}:{token.source_reset_epoch}"
                        fresh_handshake = FreshEpochHandshake(
                            token_key=token.key(),
                            source_reset_epoch=token.source_reset_epoch,
                            request_id=request_id,
                            requested_at=requested_at,
                            requested_monotonic=requested_monotonic,
                            next_status_at=requested_monotonic + args.fresh_epoch_timeout,
                            decision=decision,
                        )
                        fresh_epoch_requested_once = True
                        status.update(
                            prompt=token.prompt,
                            status="fresh_epoch_requested",
                            error="",
                            ready=True,
                            backlog_eligible=False,
                            needs_fresh_source=True,
                            fresh_epoch_handshake_state="awaiting_bridge",
                            fresh_epoch_request_id=request_id,
                            fresh_epoch_requested_at=requested_at,
                            fresh_epoch_requested_source_reset_epoch=(token.source_reset_epoch),
                            fresh_epoch_warmup_completed=True,
                            fresh_epoch_reason=decision.reason,
                            fresh_epoch_warmup_window_id=decision.warmup_window_id,
                            fresh_epoch_oldest_window_id=decision.oldest_window_id,
                            fresh_epoch_latest_window_id=decision.latest_window_id,
                            fresh_epoch_initial_window_id=decision.initial_window_id,
                            fresh_epoch_recovery_target_window_id=(decision.recovery_target_window_id),
                            fresh_epoch_replay_window_count=(decision.replay_window_count),
                            fresh_epoch_unpublished_window_count=(decision.unpublished_window_count),
                            fresh_epoch_retry_count=0,
                            fresh_epoch_timed_out=False,
                            compile_mode=runtime.compile_mode,
                            reset_epoch=token.generation,
                            source_reset_epoch=token.source_reset_epoch,
                            prompt_revision=token.prompt_revision,
                            track_id=token.track_id,
                            source_session_id=token.source_session_id,
                            accepted_request_id=token.request_id,
                            warmed=True,
                        )
                        fresh_requested_now = True
                        break
                    startup_plan_checked = True
                    if recovery_target is not None:
                        recovery_visible_source_seq = decision.recovery_target_last_seq
                    status.update(
                        status=("recovering" if recovery_target is not None else "rendering"),
                        backlog_eligible=False,
                        needs_fresh_source=False,
                        fresh_epoch_handshake_state=(
                            "recovering" if recovery_target is not None else "stream_starting"
                        ),
                        compile_mode=runtime.compile_mode,
                        warmed=True,
                        error=runtime.compile_fallback_reason,
                        recovery_replay_window_count=decision.replay_window_count,
                        recovery_unpublished_window_count=(decision.unpublished_window_count),
                    )
                hidden_replay = is_hidden_replay_window(window.header.window_id, recovery_target)
                frames = runtime.run_window(window, token, lambda: reader.assert_fence(token))
                model_complete = True
                reader.assert_fence(token)
                if not gc_telemetry.steady_state_freeze_attempted:
                    gc_telemetry.apply_steady_state_freeze_once()
                    gc_telemetry.reset()
                if hidden_replay:
                    success = True
                    break
                assert_visible_source_seq_advances(frames, recovery_visible_source_seq)
                publisher.enqueue(token, window, frames)
                loop_telemetry.record_enqueue_completed(time.monotonic())
                runtime.record_model_window_cycle(
                    window.header.window_id,
                    time.monotonic() - window_cycle_started_at,
                )
                stream_started = True
                recovered_target_pending_publish = None
                recovery_visible_source_seq = -1
                status.update(
                    prompt=token.prompt,
                    status="output_queued",
                    error=runtime.compile_fallback_reason,
                    ready=True,
                    backlog_eligible=True,
                    needs_fresh_source=False,
                    fresh_epoch_handshake_state="streaming",
                    compile_mode=runtime.compile_mode,
                    reset_epoch=token.generation,
                    source_reset_epoch=token.source_reset_epoch,
                    accepted_request_id=token.request_id,
                    accepted_window_id=window.header.window_id,
                    attempt=attempt,
                    warmed=runtime.warmed,
                    **runtime.causal_trace_telemetry(),
                    **runtime.model_timing_telemetry(),
                    **loop_telemetry.telemetry(),
                    **gc_telemetry.telemetry(),
                )
                success = True
                break
            except FenceChanged:
                runtime.restore(snapshot)
                break
            except OutputPublishFailed as error:
                runtime.restore(snapshot)
                last_error = str(error)
                break
            except Exception as error:
                runtime.restore(snapshot)
                last_error = f"{error}\n{traceback.format_exc()}"[-4000:]
                fell_back = (
                    runtime.fallback_to_eager(str(error)) if args.eager_fallback and not model_complete else False
                )
                status.update(
                    prompt=token.prompt,
                    status="retrying",
                    error=last_error,
                    ready=True,
                    backlog_eligible=False,
                    needs_fresh_source=False,
                    compile_mode=runtime.compile_mode,
                    reset_epoch=token.generation,
                    source_reset_epoch=token.source_reset_epoch,
                    accepted_request_id=token.request_id,
                    accepted_window_id=window.header.window_id,
                    attempt=attempt,
                    eager_fallback=fell_back,
                    **runtime.model_timing_telemetry(),
                    **loop_telemetry.telemetry(),
                    **gc_telemetry.telemetry(),
                )
                torch.cuda.empty_cache()
                if attempt < args.max_attempts:
                    time.sleep(min(1.0, 0.2 * attempt))

        if hidden_replay and success:
            if recovery_target is not None and runtime.last_window_id == recovery_target.window_id:
                recovered_target = recovery_target
                recovery_target = None
                recovered_target_pending_publish = recovered_target
                status.update(
                    prompt=token.prompt,
                    status="waiting_for_input",
                    error="",
                    ready=True,
                    backlog_eligible=False,
                    needs_fresh_source=False,
                    fresh_epoch_handshake_state="recovered_waiting_unpublished",
                    compile_mode=runtime.compile_mode,
                    reset_epoch=token.generation,
                    source_reset_epoch=token.source_reset_epoch,
                    accepted_request_id=token.request_id,
                    accepted_window_id=runtime.last_window_id,
                    completed_request_id=token.request_id,
                    completed_window_id=runtime.last_window_id,
                    completed_source_seq=recovery_visible_source_seq,
                    recovered_through_window_id=runtime.last_window_id,
                    recovery_visible_source_seq=recovery_visible_source_seq,
                    recovery_previous_worker_instance_id=(recovered_target.previous_worker_instance_id),
                    warmed=runtime.warmed,
                    **runtime.causal_trace_telemetry(),
                    **runtime.model_timing_telemetry(),
                    **loop_telemetry.telemetry(),
                    **gc_telemetry.telemetry(),
                )
            continue
        if fresh_requested_now:
            continue
        if success:
            if args.once:
                return 0 if publisher.wait_for_window(token, window.header.window_id) else 1
            continue

        current = reader.authority()
        if current.token is None or current.token.key() != token.key():
            continue
        blocked_token = token.key()
        status.update(
            prompt=token.prompt,
            status="failed",
            error=last_error or "session fence changed before the window could commit",
            ready=True,
            backlog_eligible=False,
            needs_fresh_source=False,
            compile_mode=runtime.compile_mode,
            reset_epoch=token.generation,
            source_reset_epoch=token.source_reset_epoch,
            accepted_request_id=token.request_id,
            accepted_window_id=window.header.window_id,
            failed_window_id=window.header.window_id,
            attempts=args.max_attempts,
            output_fps=0.0,
            **runtime.model_timing_telemetry(),
            **loop_telemetry.telemetry(),
            **gc_telemetry.telemetry(),
        )


def main() -> int:
    try:
        return run()
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
