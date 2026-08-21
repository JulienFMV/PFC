"""Validate the structural Swiss LT compute policy without granting authority."""

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
from pfc_shaping.validation.ch_lt_compute_runtime import (
    assess_compute_runtime,
    load_compute_runtime_contract,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        assert_installed_runtime_sealed()
    except ReleaseCliIdentityError as exc:
        print(
            json.dumps(
                {
                    "status": "INVALID_GOVERNED_LT_RUNTIME",
                    "command_id": "audit_ch_lt_compute_runtime",
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
        document, payload = load_compute_runtime_contract(
            args.contract,
            expected_sha256=args.expected_contract_sha256,
        )
        result = assess_compute_runtime(document, document_bytes=payload).to_dict()
    except (OSError, UnicodeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "INVALID_CH_LT_COMPUTE_RUNTIME_POLICY",
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
