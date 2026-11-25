"""Which process is running a job.

The web app and the command line share one database. On startup the web app
requeues jobs left ``running`` by a crash — but a job can also be ``running``
because a ``peaksift clip`` / ``peaksift live`` in another window is working on
it right now. Requeuing that one would run it twice at once, with two copies of
Whisper fighting over the GPU.

So whoever runs a job writes ``owner.json`` (process ID and start time) into the
job's work folder, and removes it when done. A job whose owner is still alive is
left alone. The start time guards against a recycled process ID.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path

from .. import paths

log = logging.getLogger(__name__)

OWNER_FILE = "owner.json"


def _owner_path(job_id: str) -> Path:
    return paths.job_work_dir(job_id) / OWNER_FILE


def _process_started(pid: int) -> float | None:
    """When ``pid`` started (seconds, any fixed epoch), or None if it isn't running."""
    if pid <= 0:
        return None
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        process_query_limited_information = 0x1000
        still_active = 259
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return None
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return None
            if code.value != still_active:
                return None
            created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
            if not kernel32.GetProcessTimes(
                handle,
                ctypes.byref(created),
                ctypes.byref(exited),
                ctypes.byref(kernel),
                ctypes.byref(user),
            ):
                return 0.0
            return ((created.dwHighDateTime << 32) | created.dwLowDateTime) / 1e7
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass  # exists, owned by someone else
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        with contextlib.suppress(OSError, IndexError, ValueError):
            return float(stat.read_text().rsplit(")", 1)[1].split()[19])
    return 0.0


def owned_elsewhere(job_id: str) -> bool:
    """True when another live process is running this job."""
    path = _owner_path(job_id)
    try:
        owner = json.loads(path.read_text(encoding="utf-8"))
        pid, started = int(owner["pid"]), float(owner["started"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if pid == os.getpid():
        return False
    now = _process_started(pid)
    return now is not None and (started == 0.0 or now == 0.0 or abs(now - started) < 1.0)


@contextlib.contextmanager
def claim(job_id: str) -> Iterator[None]:
    """Mark this process as running ``job_id`` for the duration of the block."""
    path = _owner_path(job_id)
    pid = os.getpid()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"pid": pid, "started": _process_started(pid) or 0.0}), encoding="utf-8"
        )
    except OSError as exc:  # never block a job over bookkeeping
        log.debug("Could not write %s: %s", path, exc)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            owner = json.loads(path.read_text(encoding="utf-8"))
            if owner.get("pid") == pid:
                path.unlink()
