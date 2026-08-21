#!/usr/bin/env python3
"""Audit the outcome-blind CH LT external origin-registry protocol draft."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pfc_shaping.validation.ch_lt_origin_registry_protocol import (
    verify_origin_registry_protocol,
)

ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify_origin_registry_protocol(
            repo_root=ROOT,
            protocol_path=args.protocol,
            expected_protocol_sha256=args.expected_protocol_sha256,
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "ch_lt_origin_registry_protocol_error.v1",
                    "status": "INVALID_CH_LT_ORIGIN_REGISTRY_PROTOCOL_NO_GO",
                    "protocol_complete": False,
                    "registry_implemented": False,
                    "countable_prospective_origin_count": 0,
                    "evidence_slot_satisfied": False,
                    "truth_open_authorized": False,
                    "scientific_admission": False,
                    "execution_authorized": False,
                    "production_authorization": False,
                    "promotion_gate": False,
                    "error": str(exc),
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
