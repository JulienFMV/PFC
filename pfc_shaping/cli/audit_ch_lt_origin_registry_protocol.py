"""Audit the outcome-blind CH LT external origin-registry protocol draft."""

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
from pfc_shaping.validation.ch_lt_origin_registry_protocol import (
    verify_origin_registry_protocol,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        assert_installed_runtime_sealed()
    except ReleaseCliIdentityError as exc:
        _emit_error("INVALID_GOVERNED_LT_RUNTIME", exc)
        return 2
    args = _parser().parse_args(argv)
    try:
        result = verify_origin_registry_protocol(
            repo_root=args.evidence_root,
            protocol_path=args.protocol,
            expected_protocol_sha256=args.expected_protocol_sha256,
        )
    except (OSError, ValueError) as exc:
        _emit_error("INVALID_CH_LT_ORIGIN_REGISTRY_PROTOCOL_NO_GO", exc)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _emit_error(status: str, exc: BaseException) -> None:
    print(
        json.dumps(
            {
                "schema_version": "ch_lt_origin_registry_protocol_error.v1",
                "status": status,
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


if __name__ == "__main__":
    raise SystemExit(main())
