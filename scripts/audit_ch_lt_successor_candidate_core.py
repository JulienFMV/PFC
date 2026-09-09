#!/usr/bin/env python3
"""Audit the outcome-blind CH LT successor candidate core v3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pfc_shaping.validation.ch_lt_successor_candidate_core_v3 import (
    verify_candidate_core_v3,
)

ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--expected-core-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify_candidate_core_v3(
            repo_root=ROOT,
            core_path=args.core,
            expected_core_sha256=args.expected_core_sha256,
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "ch_lt_successor_candidate_core_error.v3",
                    "status": "INVALID_CANDIDATE_CORE_NO_GO",
                    "error": str(exc),
                    "candidate_core_admitted": False,
                    "successor_exists": False,
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
