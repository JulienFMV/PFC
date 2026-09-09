#!/usr/bin/env python3
"""Disabled legacy direct-publication entry point.

Use ``python -m pfc_shaping.cli.governed_release`` for candidate build,
finalization, registration, audit and promotion.
"""

from __future__ import annotations


def main() -> None:
    raise RuntimeError(
        "Legacy direct publication is disabled. Build an immutable LT candidate with "
        "pfc_shaping.cli.governed_release and use the governed audit/receipt/promotion flow."
    )


if __name__ == "__main__":
    main()
