"""Cap the size of the live services' log files.

The supervisors append child stdout/stderr (STK, bridge, worker, preview) to
``<log dir>/<service>.log`` through ``O_APPEND`` file descriptors, so a log can
be rotated in place: copy it to ``.1`` (shifting older backups) and truncate
the original; writers keep appending at the new end. Old worker hang dumps are
pruned as well.

    python -m alayarenderer_flash.demo.logcap LOG_DIR --max-bytes 52428800 --backups 3
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

DEFAULT_MAX_BYTES = 50 * 1024 * 1024
DEFAULT_BACKUPS = 3
DEFAULT_KEEP_HANG_DUMPS = 20


def rotate_if_needed(path: Path, max_bytes: int, backups: int) -> bool:
    """Rotate ``path`` into ``path.1 .. path.<backups>`` once it exceeds ``max_bytes``."""
    try:
        if path.stat().st_size <= max_bytes:
            return False
    except FileNotFoundError:
        return False
    for index in range(backups - 1, 0, -1):
        older = path.with_name(f"{path.name}.{index}")
        if older.exists():
            os.replace(older, path.with_name(f"{path.name}.{index + 1}"))
    if backups > 0:
        shutil.copyfile(path, path.with_name(f"{path.name}.1"))
    with open(path, "r+b") as handle:
        handle.truncate(0)
    return True


def prune_hang_dumps(log_dir: Path, keep: int) -> int:
    """Delete all but the newest ``keep`` ``worker_hang_*.txt`` dumps."""
    dumps = sorted(log_dir.glob("worker_hang_*.txt"))
    stale = dumps[: max(0, len(dumps) - keep)]
    for dump in stale:
        dump.unlink(missing_ok=True)
    return len(stale)


def sweep(log_dir: Path, max_bytes: int, backups: int, keep_hang_dumps: int) -> list[Path]:
    """Rotate every oversized ``*.log`` in ``log_dir``; return the rotated paths."""
    rotated = [path for path in sorted(log_dir.glob("*.log")) if rotate_if_needed(path, max_bytes, backups)]
    prune_hang_dumps(log_dir, keep_hang_dumps)
    return rotated


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--backups", type=int, default=DEFAULT_BACKUPS)
    parser.add_argument("--keep-hang-dumps", type=int, default=DEFAULT_KEEP_HANG_DUMPS)
    parser.add_argument("--interval", type=float, default=30.0, help="seconds between sweeps")
    parser.add_argument("--once", action="store_true", help="sweep once and exit")
    args = parser.parse_args(argv)
    while True:
        for path in sweep(args.log_dir, args.max_bytes, args.backups, args.keep_hang_dumps):
            print(f"rotated {path}", flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
