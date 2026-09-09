#!/usr/bin/env python3
"""Compatibility wrapper for the packaged monthly promotion capstone."""

from __future__ import annotations

import sys

from pfc_shaping.calibration import monthly_curve_capstone as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
