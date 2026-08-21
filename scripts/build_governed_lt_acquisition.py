#!/usr/bin/env python3
"""Compatibility wrapper for the installed ``pfc-lt-build-acquisition`` command.

Production operators must use the sealed wheel entry point. Running this wrapper
as a module preserves developer ergonomics without mutating ``sys.path`` or
selecting a checkout as an import authority.
"""

from __future__ import annotations

from pfc_shaping.cli.governed_acquisition_builder import build_acquisition, main

__all__ = ["build_acquisition", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
