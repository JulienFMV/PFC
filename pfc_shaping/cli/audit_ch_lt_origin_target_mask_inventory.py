"""Build or audit an outcome-blind CH LT structural origin/target/mask inventory."""

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
from pfc_shaping.validation.ch_lt_origin_target_mask_inventory import (
    build_structural_inventory,
    verify_structural_inventory,
    write_structural_inventory,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="action", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--origin-as-of-utc", required=True)
    build.add_argument("--output", type=Path, required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--inventory", type=Path, required=True)
    audit.add_argument("--expected-inventory-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        assert_installed_runtime_sealed()
    except ReleaseCliIdentityError as exc:
        _emit_error("INVALID_GOVERNED_LT_RUNTIME", exc)
        return 2
    args = _parser().parse_args(argv)
    try:
        if args.action == "build":
            document = build_structural_inventory(
                repo_root=args.evidence_root,
                origin_as_of_utc=args.origin_as_of_utc,
            )
            result = write_structural_inventory(
                document=document,
                output_path=args.output,
                repo_root=args.evidence_root,
            )
        else:
            result = verify_structural_inventory(
                repo_root=args.evidence_root,
                inventory_path=args.inventory,
                expected_inventory_sha256=args.expected_inventory_sha256,
            )
    except (OSError, ValueError) as exc:
        _emit_error("INVALID_CH_LT_ORIGIN_TARGET_MASK_INVENTORY_NO_GO", exc)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _emit_error(status: str, exc: BaseException) -> None:
    print(
        json.dumps(
            {
                "schema_version": "ch_lt_origin_target_mask_inventory_error.v1",
                "status": status,
                "evidence_slot_satisfied": False,
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
