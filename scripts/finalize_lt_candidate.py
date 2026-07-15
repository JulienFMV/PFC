#!/usr/bin/env python3
"""Compatibility wrapper for the governed LT candidate finalizer."""

from __future__ import annotations

import sys

from pfc_shaping.pipeline import candidate_finalize as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
