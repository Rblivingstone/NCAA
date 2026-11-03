"""A cross-process file lock that works on Linux, macOS and Windows."""
from __future__ import annotations

import os
import time
from contextlib import contextmanager

try:
    import fcntl
except ImportError:                      # Windows
    fcntl = None
    import msvcrt


@contextmanager
def file_lock(path):
    """Hold an exclusive lock on ``path`` (created if missing) for the duration of the block."""
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a+") as fh:
        if fcntl is not None:
            fcntl.flock(fh, fcntl.LOCK_EX)
        else:
            fh.seek(0)
            while True:                  # msvcrt has no blocking-forever mode: retry
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(0.1)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(fh, fcntl.LOCK_UN)
            else:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
