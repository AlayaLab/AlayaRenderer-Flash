#!/usr/bin/env python3
"""Preallocated double-slot transport for live uint8 G-buffer windows.

The producer writes frames into a fixed MAP_SHARED slot as they are captured.
Publishing a window only replaces a small JSON slot-state document and the
normal immutable metadata manifest.  The consumer snapshots the mapped bytes
between two identical state reads, so a slot can never be accepted while it is
being reused.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TRANSPORT_NAME = "shm_slots"
SLOT_COUNT = 2
SLOT_FILE_RE = "gbuffer_slot_{slot}.u8"
SLOT_STATE_RE = "gbuffer_slot_{slot}.json"


class SlotBusy(RuntimeError):
    """A producer slot is still owned by a window the consumer has not copied."""


class SlotChanged(RuntimeError):
    """The shared slot changed before a stable consumer snapshot completed."""


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, ensure_ascii=True, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_state_snapshot(path: Path) -> tuple[dict[str, Any], tuple[int, int, int, int]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise SlotChanged(f"slot state {path.name} is not a regular file")
        with os.fdopen(os.dup(fd), "rb") as handle:
            payload = handle.read()
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise SlotChanged(f"slot state {path.name} changed while reading")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SlotChanged(f"slot state {path.name} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise SlotChanged(f"slot state {path.name} is not an object")
    return value, identity


@dataclass(frozen=True)
class SlotReservation:
    slot_index: int
    window_id: int
    writer_epoch: int
    commit_token: str
    frame_count: int
    shape: tuple[int, int, int, int, int]
    file_name: str
    state_file_name: str
    file_size_bytes: int


@dataclass(frozen=True)
class SlotReference:
    slot_index: int
    window_id: int
    writer_epoch: int
    commit_token: str
    shape: tuple[int, int, int, int, int]
    dtype: str
    file_name: str
    state_file_name: str
    file_size_bytes: int
    payload_nbytes: int

    def metadata(self) -> dict[str, Any]:
        return {
            "gbuffers_slot_index": self.slot_index,
            "gbuffers_slot_window_id": self.window_id,
            "gbuffers_slot_writer_epoch": self.writer_epoch,
            "gbuffers_slot_commit_token": self.commit_token,
            "gbuffers_slot_state_file": self.state_file_name,
            "gbuffers_data_offset_bytes": 0,
        }


class DoubleSlotWriter:
    """Single-thread-owned producer for two fixed raw uint8 slot files."""

    def __init__(
        self,
        root: Path,
        max_frames: int,
        modalities: int,
        height: int,
        width: int,
        channels: int = 3,
        slot_count: int = SLOT_COUNT,
    ) -> None:
        if min(max_frames, modalities, height, width, channels, slot_count) <= 0:
            raise ValueError("shared slot dimensions must be positive")
        if slot_count != SLOT_COUNT:
            raise ValueError(f"this transport requires exactly {SLOT_COUNT} slots")
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_shape = (max_frames, modalities, height, width, channels)
        self.file_size_bytes = int(np.prod(self.max_shape, dtype=np.int64))
        self._maps: list[np.memmap] = []
        self._active: SlotReservation | None = None
        self._written_offsets: set[int] = set()
        for slot in range(slot_count):
            path = self.root / SLOT_FILE_RE.format(slot=slot)
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
            fd = os.open(path, flags, 0o600)
            try:
                current_size = os.fstat(fd).st_size
                if current_size != self.file_size_bytes:
                    os.ftruncate(fd, self.file_size_bytes)
            finally:
                os.close(fd)
            self._maps.append(np.memmap(path, dtype=np.uint8, mode="r+", shape=self.max_shape, order="C"))

    def state_path(self, slot_index: int) -> Path:
        return self.root / SLOT_STATE_RE.format(slot=slot_index)

    def invalidate(self, writer_epoch: int, reason: str) -> None:
        self._active = None
        self._written_offsets.clear()
        for slot_index in range(SLOT_COUNT):
            _atomic_write_json(
                self.state_path(slot_index),
                {
                    "schema_version": 1,
                    "transport": TRANSPORT_NAME,
                    "status": "invalid",
                    "slot_index": slot_index,
                    "writer_epoch": writer_epoch,
                    "reason": reason,
                },
            )

    def begin(
        self,
        window_id: int,
        writer_epoch: int,
        frame_count: int,
        accepted_window_id: int,
    ) -> SlotReservation:
        if self._active is not None:
            return self._active
        if frame_count <= 0 or frame_count > self.max_shape[0]:
            raise ValueError(f"frame_count {frame_count} exceeds slot capacity {self.max_shape[0]}")
        slot_index = window_id % SLOT_COUNT
        state_path = self.state_path(slot_index)
        previous = read_state(state_path)
        previous_epoch = int(previous.get("writer_epoch", -1))
        previous_window = int(previous.get("window_id", -1))
        if (
            previous.get("status") == "ready"
            and previous_epoch == writer_epoch
            and previous_window > accepted_window_id
        ):
            raise SlotBusy(
                f"slot {slot_index} still holds window {previous_window}; worker accepted through {accepted_window_id}"
            )
        shape = (
            frame_count,
            self.max_shape[1],
            self.max_shape[2],
            self.max_shape[3],
            self.max_shape[4],
        )
        reservation = SlotReservation(
            slot_index=slot_index,
            window_id=window_id,
            writer_epoch=writer_epoch,
            commit_token=uuid.uuid4().hex,
            frame_count=frame_count,
            shape=shape,
            file_name=SLOT_FILE_RE.format(slot=slot_index),
            state_file_name=SLOT_STATE_RE.format(slot=slot_index),
            file_size_bytes=self.file_size_bytes,
        )
        _atomic_write_json(
            state_path,
            {
                "schema_version": 1,
                "transport": TRANSPORT_NAME,
                "status": "writing",
                "slot_index": slot_index,
                "window_id": window_id,
                "writer_epoch": writer_epoch,
                "commit_token": reservation.commit_token,
                "shape": list(shape),
                "dtype": "uint8",
            },
        )
        self._active = reservation
        self._written_offsets.clear()
        return reservation

    def write_frame(self, reservation: SlotReservation, offset: int, frame: np.ndarray) -> None:
        if reservation != self._active:
            raise RuntimeError("shared-slot reservation is no longer active")
        if offset < 0 or offset >= reservation.frame_count:
            raise IndexError(f"slot frame offset {offset} is out of range")
        expected = self.max_shape[1:]
        if frame.dtype != np.uint8 or frame.shape != expected:
            raise ValueError(f"shared-slot frame must be uint8 {expected}, got {frame.dtype} {frame.shape}")
        self._maps[reservation.slot_index][offset] = frame
        self._written_offsets.add(offset)

    def commit(
        self,
        reservation: SlotReservation,
        fence: Mapping[str, Any],
    ) -> SlotReference:
        if reservation != self._active:
            raise RuntimeError("shared-slot reservation is no longer active")
        expected_offsets = set(range(reservation.frame_count))
        if self._written_offsets != expected_offsets:
            missing = sorted(expected_offsets - self._written_offsets)
            raise RuntimeError(f"shared slot is missing frame offsets {missing}")
        reference = SlotReference(
            slot_index=reservation.slot_index,
            window_id=reservation.window_id,
            writer_epoch=reservation.writer_epoch,
            commit_token=reservation.commit_token,
            shape=reservation.shape,
            dtype="uint8",
            file_name=reservation.file_name,
            state_file_name=reservation.state_file_name,
            file_size_bytes=reservation.file_size_bytes,
            payload_nbytes=int(np.prod(reservation.shape, dtype=np.int64)),
        )
        document = {
            "schema_version": 1,
            "transport": TRANSPORT_NAME,
            "status": "ready",
            "slot_index": reference.slot_index,
            "window_id": reference.window_id,
            "writer_epoch": reference.writer_epoch,
            "commit_token": reference.commit_token,
            "shape": list(reference.shape),
            "dtype": reference.dtype,
            "file_name": reference.file_name,
            "file_size_bytes": reference.file_size_bytes,
            "payload_nbytes": reference.payload_nbytes,
            **dict(fence),
        }
        _atomic_write_json(self.state_path(reference.slot_index), document)
        self._active = None
        self._written_offsets.clear()
        return reference

    def abort(self, writer_epoch: int, reason: str) -> None:
        active = self._active
        self._active = None
        self._written_offsets.clear()
        if active is None:
            return
        _atomic_write_json(
            self.state_path(active.slot_index),
            {
                "schema_version": 1,
                "transport": TRANSPORT_NAME,
                "status": "invalid",
                "slot_index": active.slot_index,
                "window_id": active.window_id,
                "writer_epoch": writer_epoch,
                "reason": reason,
            },
        )


def state_matches_reference(state: Mapping[str, Any], reference: SlotReference) -> bool:
    return bool(
        state.get("transport") == TRANSPORT_NAME
        and state.get("status") == "ready"
        and int(state.get("slot_index", -1)) == reference.slot_index
        and int(state.get("window_id", -1)) == reference.window_id
        and int(state.get("writer_epoch", -1)) == reference.writer_epoch
        and str(state.get("commit_token", "") or "") == reference.commit_token
        and tuple(int(value) for value in state.get("shape", [])) == reference.shape
        and str(state.get("dtype", "") or "") == reference.dtype
        and str(state.get("file_name", "") or "") == reference.file_name
        and int(state.get("file_size_bytes", 0)) == reference.file_size_bytes
        and int(state.get("payload_nbytes", 0)) == reference.payload_nbytes
    )


def snapshot_slot(root: Path, reference: SlotReference) -> np.ndarray:
    """Return a private C-contiguous copy after a stable before/after state check."""
    state_path = root / reference.state_file_name
    before, before_state_identity = _read_state_snapshot(state_path)
    if not state_matches_reference(before, reference):
        raise SlotChanged(f"slot {reference.slot_index} no longer owns window {reference.window_id}")
    path = root / reference.file_name
    if path.parent.resolve() != root.resolve():
        raise SlotChanged("shared slot path escaped its input directory")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    stat_before = os.fstat(fd)
    if not stat.S_ISREG(stat_before.st_mode):
        os.close(fd)
        raise SlotChanged(f"slot file {path.name} is not a regular file")
    if stat_before.st_size != reference.file_size_bytes:
        os.close(fd)
        raise SlotChanged(f"slot file has {stat_before.st_size} bytes, expected {reference.file_size_bytes}")
    try:
        with os.fdopen(os.dup(fd), "rb") as handle:
            mapped = np.memmap(
                handle,
                dtype=np.uint8,
                mode="r",
                offset=0,
                shape=reference.shape,
                order="C",
            )
            try:
                snapshot = np.array(mapped, dtype=np.uint8, copy=True, order="C")
            finally:
                del mapped
        stat_after = os.fstat(fd)
    finally:
        os.close(fd)
    after, after_state_identity = _read_state_snapshot(state_path)
    if before != after or not state_matches_reference(after, reference):
        raise SlotChanged(f"slot {reference.slot_index} changed while copying window {reference.window_id}")
    if before_state_identity != after_state_identity:
        raise SlotChanged("shared slot state identity changed while copying")
    if (
        stat_after.st_dev,
        stat_after.st_ino,
        stat_after.st_size,
    ) != (
        stat_before.st_dev,
        stat_before.st_ino,
        stat_before.st_size,
    ):
        raise SlotChanged("shared slot file identity changed while copying")
    return snapshot


def run_self_check() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="alaya_shm_slots_") as temporary:
        root = Path(temporary)
        writer = DoubleSlotWriter(root, 3, 2, 4, 5)
        writer.invalidate(1, "self_check")
        reservations: list[SlotReservation] = []
        references: list[SlotReference] = []
        for window_id in (1, 2):
            reservation = writer.begin(window_id, 1, 3, accepted_window_id=window_id - 2)
            reservations.append(reservation)
            for offset in range(3):
                frame = np.full((2, 4, 5, 3), window_id * 10 + offset, dtype=np.uint8)
                writer.write_frame(reservation, offset, frame)
            references.append(writer.commit(reservation, {"source_reset_epoch": 7, "reset_epoch": 9}))
        first = snapshot_slot(root, references[0])
        second = snapshot_slot(root, references[1])
        assert first[:, 0, 0, 0, 0].tolist() == [10, 11, 12]
        assert second[:, 0, 0, 0, 0].tolist() == [20, 21, 22]
        try:
            writer.begin(3, 1, 3, accepted_window_id=0)
        except SlotBusy:
            busy_guard = True
        else:
            busy_guard = False
        assert busy_guard
        reused = writer.begin(3, 1, 3, accepted_window_id=1)
        for offset in range(3):
            writer.write_frame(
                reused,
                offset,
                np.full((2, 4, 5, 3), 30 + offset, dtype=np.uint8),
            )
        third = writer.commit(reused, {"source_reset_epoch": 7, "reset_epoch": 9})
        try:
            snapshot_slot(root, references[0])
        except SlotChanged:
            stale_guard = True
        else:
            stale_guard = False
        assert stale_guard
        assert snapshot_slot(root, third)[:, 0, 0, 0, 0].tolist() == [30, 31, 32]
        return {
            "ok": True,
            "transport": TRANSPORT_NAME,
            "slots": SLOT_COUNT,
            "busy_guard": busy_guard,
            "stale_guard": stale_guard,
            "slot_file_size_bytes": writer.file_size_bytes,
        }


if __name__ == "__main__":
    print(json.dumps(run_self_check(), sort_keys=True))
