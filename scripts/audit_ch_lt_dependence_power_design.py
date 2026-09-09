#!/usr/bin/env python3
"""Audit the outcome-blind CH LT dependence and power design draft."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pfc_shaping.validation.ch_lt_dependence_power_design import (
    verify_dependence_power_design,
)

ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--expected-design-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify_dependence_power_design(
            repo_root=ROOT,
            design_path=args.design,
            expected_design_sha256=args.expected_design_sha256,
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "ch_lt_dependence_power_design_error.v1",
                    "status": "INVALID_DEPENDENCE_POWER_DESIGN_NO_GO",
                    "error": str(exc),
                    "power_design_complete": False,
                    "evidence_slot_satisfied": False,
                    "scientific_admission": False,
                    "execution_authorized": False,
                    "production_authorization": False,
                    "promotion_gate": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
