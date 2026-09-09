"""Small cross-platform process identity helpers for crash recovery."""

from __future__ import annotations

import os
from pathlib import Path
import uuid


def process_start_token(pid: int) -> str | None:
    """Return a stable token that distinguishes PID reuse when available."""

    if pid <= 0:
        return None
    if os.name == "nt":
        import ctypes

        class FileTime(ctypes.Structure):
            _fields_ = (("low", ctypes.c_ulong), ("high", ctypes.c_ulong))

        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            0x1000,
            False,
            pid,
        )
        if not handle:
            return None
        try:
            creation = FileTime()
            exit_time = FileTime()
            kernel = FileTime()
            user = FileTime()
            if not ctypes.windll.kernel32.GetProcessTimes(  # type: ignore[attr-defined]
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel),
                ctypes.byref(user),
            ):
                return None
            value = (int(creation.high) << 32) | int(creation.low)
            return f"{value:016x}"
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]

    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        raw = stat_path.read_text(encoding="ascii")
        fields_after_name = raw[raw.rfind(")") + 2 :].split()
        start_ticks = int(fields_after_name[19])
    except (OSError, UnicodeDecodeError, ValueError, IndexError):
        return None
    return f"{start_ticks:016x}"


def candidate_temporary_name(*, suffix: str) -> str:
    if suffix not in {"tmp", "pending.json"}:
        raise ValueError(f"unsupported candidate temporary suffix: {suffix}")
    pid = os.getpid()
    token = process_start_token(pid) or "unknown"
    return f".pfc-{pid}-{token}-{uuid.uuid4().hex}.{suffix}"
