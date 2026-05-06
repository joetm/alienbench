"""Shared concurrency helpers for pipeline stages that write to a shared
``data/`` tree from multiple processes.

Both Stage 1 (``generate``) and Stage 2 (``extract``) rely on the same
duplication-prevention scheme: each unit of work is claimed atomically via
a per-key reservation file (``O_CREAT | O_EXCL``) under a ``_reservations/``
directory, and JSONL appends are guarded by an ``fcntl.flock``. Stale
reservations from crashed workers (dead PID, or older than
``RESERVATION_TTL_SECONDS``) are reaped on each acquisition pass.

Reservation keys are stored as filenames, so callers using integer indices
(generate) or UUID strings (extract) convert at the boundary.
"""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

logger = logging.getLogger(__name__)

# A reservation older than this is treated as abandoned even if the recorded
# PID still exists (e.g. cross-host runs where PID checks are not meaningful).
RESERVATION_TTL_SECONDS = 1800


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def reap_stale(res_dir: Path) -> None:
    """Remove reservation files whose owner is gone or whose age exceeds the TTL."""
    if not res_dir.exists():
        return
    now = time.time()
    for entry in res_dir.iterdir():
        try:
            stat = entry.stat()
        except FileNotFoundError:
            continue
        stale = (now - stat.st_mtime) > RESERVATION_TTL_SECONDS
        if not stale:
            try:
                pid_str = entry.read_text().split()[0]
                pid = int(pid_str)
            except (OSError, ValueError, IndexError):
                pid = -1
            if pid_alive(pid):
                continue
        try:
            entry.unlink()
            logger.info("Reaped stale reservation %s", entry)
        except FileNotFoundError:
            pass


def live_reservations(res_dir: Path) -> set[str]:
    if not res_dir.exists():
        return set()
    return {entry.name for entry in res_dir.iterdir()}


def try_reserve(
    res_dir: Path,
    candidates: Iterable[str],
    completed_fn: Callable[[], set[str]],
    rng: random.Random,
) -> str | None:
    """Atomically reserve one available key from ``candidates``.

    ``completed_fn`` returns the set of keys already written to the output
    JSONL. It is called both before and after the ``O_EXCL`` open, to close
    the race window where another worker appended for the same key and
    released its reservation between our snapshot and our claim. Returns the
    reserved key, or ``None`` if every candidate is either already completed
    or held by a live reservation.
    """
    res_dir.mkdir(parents=True, exist_ok=True)
    reap_stale(res_dir)
    done = completed_fn()
    claimed = live_reservations(res_dir)
    available = [c for c in candidates if c not in done and c not in claimed]
    rng.shuffle(available)
    payload = f"{os.getpid()} {datetime.now(timezone.utc).isoformat()}\n".encode()
    for key in available:
        path = res_dir / key
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        except OSError as e:
            if e.errno == errno.EEXIST:
                continue
            raise
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        if key in completed_fn():
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            continue
        return key
    return None


def release_reservation(res_dir: Path, key: str) -> None:
    path = res_dir / key
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def append_jsonl_locked(path: Path, record: dict) -> None:
    """Append ``record`` as a JSONL line under an exclusive ``fcntl`` lock."""
    line = json.dumps(record) + "\n"
    with open(path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
