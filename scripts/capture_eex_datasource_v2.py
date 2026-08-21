#!/usr/bin/env python3
"""Compatibility wrapper for the governed EEX DataSource REST API v2 capture."""

from __future__ import annotations

from pfc_shaping.data.eex_datasource_v2_capture import (
    capture_eex_datasource_v2,
    main,
)

__all__ = ["capture_eex_datasource_v2", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
