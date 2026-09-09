#!/usr/bin/env python3
"""Compatibility wrapper for the packaged CH product-normalization audit."""

from __future__ import annotations

import sys

from pfc_shaping.validation import product_normalization as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
