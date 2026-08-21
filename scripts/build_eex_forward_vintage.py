#!/usr/bin/env python3
"""Compatibility wrapper for the sealed EEX vintage builder module."""

from __future__ import annotations

from pfc_shaping.cli.eex_forward_vintage_builder import (
    build_eex_forward_vintage,
    main,
)

__all__ = ["build_eex_forward_vintage", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
