"""Reject the stale v1 dependence/power diagnostic and verify exact v2 selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from pfc_shaping.validation.dependence_power_supersession import (
    verify_dependence_power_selection,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--selected-artifact", type=Path, required=True)
    parser.add_argument("--expected-selected-artifact-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        result = verify_dependence_power_selection(
            repo_root=root,
            registry_path=args.registry,
            expected_registry_sha256=args.expected_registry_sha256,
            selected_artifact_path=args.selected_artifact,
            expected_selected_artifact_sha256=args.expected_selected_artifact_sha256,
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "INVALID_DEPENDENCE_POWER_SUPERSESSION_EVIDENCE",
                    "local_diagnostic_selection_current": False,
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
    return 0 if result["local_diagnostic_selection_current"] is True else 3


if __name__ == "__main__":
    raise SystemExit(main())
