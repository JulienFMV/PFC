#!/usr/bin/env python3
"""Compatibility wrapper for the installed ``pfc-lt`` command."""

from __future__ import annotations

import sys

from pfc_shaping.cli import governed_release as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
