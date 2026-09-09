#!/usr/bin/env python3
"""Compatibility wrapper for the local EEX forward capture module."""

from __future__ import annotations

from pfc_shaping.data.eex_forward_local_capture import (
    capture_eex_forward_workbook,
    main,
)

__all__ = ["capture_eex_forward_workbook", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
