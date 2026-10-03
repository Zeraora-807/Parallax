"""Process-owned run locks distinguish live work from abandoned database rows."""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import Self

from parallax.errors import ParallaxError


class RunAlreadyActiveError(ParallaxError):
    code = "run_already_active"


class RunLock:
    """Hold a nonblocking OS lock until exit; process death releases it as well.

    Lock files are intentionally retained: unlinking a locked file would let
    another process lock a different inode for the same run.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._descriptor: int | None = None

    def __enter__(self) -> Self:
        if self._descriptor is not None:
            raise RuntimeError("run lock is already held by this instance")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if os.name == "nt":
                import msvcrt

                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"\0")
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(descriptor)
            if error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise RunAlreadyActiveError("run is still active") from error
            raise
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        return self

    def __exit__(self, *exception_info: object) -> None:
        descriptor = self._descriptor
        if descriptor is not None:
            self._descriptor = None
            os.close(descriptor)
