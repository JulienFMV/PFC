"""Audit the outcome-blind CH LT successor candidate core v3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from pfc_shaping.pipeline.governed_release_cli_contract import (
    ReleaseCliIdentityError,
    assert_installed_runtime_sealed,
)
from pfc_shaping.validation.ch_lt_successor_candidate_core_v3 import (
    verify_candidate_core_v3,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--expected-core-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        assert_installed_runtime_sealed()
    except ReleaseCliIdentityError as exc:
        print(
            json.dumps(
                {
                    "status": "INVALID_GOVERNED_LT_RUNTIME",
                    "command_id": "audit_ch_lt_successor_candidate_core",
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
    args = _parser().parse_args(argv)
    try:
        result = verify_candidate_core_v3(
            repo_root=args.evidence_root,
            core_path=args.core,
            expected_core_sha256=args.expected_core_sha256,
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "ch_lt_successor_candidate_core_error.v3",
                    "status": "INVALID_CANDIDATE_CORE_NO_GO",
                    "candidate_core_admitted": False,
                    "successor_exists": False,
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
