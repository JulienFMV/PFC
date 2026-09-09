"""Audit a Swiss LT PFC preregistration before any scientific execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.path_safety import assert_absolute_path_has_no_links
from pfc_shaping.validation.ch_lt_pit_preregistration import (
    DRAFT_STATE,
    assess_preregistration,
    canonical_json_bytes,
    load_preregistration,
)


def audit_preregistration(
    *,
    plan_path: Path,
    expected_plan_sha256: str,
    output_path: Path | None = None,
) -> dict[str, object]:
    document, payload = load_preregistration(
        plan_path,
        expected_sha256=expected_plan_sha256,
    )
    result = assess_preregistration(document, document_bytes=payload).to_dict()
    if output_path is not None:
        target = _workspace_output(output_path)
        if target.exists():
            raise ValueError("preregistration audit output already exists")
        write_durable_exact_bytes(
            target,
            canonical_json_bytes(result) + b"\n",
            label="CH LT PIT preregistration audit",
        )
    return result


def _workspace_output(path: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    allowed_root = root / "output" / "phase14" / "ch_lt_pit_preregistration_audits"
    candidate = path if path.is_absolute() else root / path
    candidate = assert_absolute_path_has_no_links(candidate)
    try:
        candidate.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(
            "preregistration audit output must stay in the canonical audit namespace"
        ) from exc
    if not candidate.parent.is_dir():
        raise ValueError("preregistration audit output directory is unavailable")
    return candidate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--mode",
        choices=("validate-draft", "admit-execution"),
        default="admit-execution",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = audit_preregistration(
            plan_path=args.plan,
            expected_plan_sha256=args.expected_plan_sha256,
            output_path=args.output,
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "INVALID_PREREGISTRATION",
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
    if args.mode == "validate-draft":
        return 0 if result["lifecycle_state"] == DRAFT_STATE else 4
    return 0 if result["execution_authorized"] is True else 3


if __name__ == "__main__":
    raise SystemExit(main())
