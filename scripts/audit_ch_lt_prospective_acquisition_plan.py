"""Audit the blocked plan for fresh CH LT data acquisition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfc_shaping.validation.ch_lt_prospective_acquisition_plan import (
    assess_prospective_acquisition_plan,
    load_prospective_acquisition_plan,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        document, payload = load_prospective_acquisition_plan(
            args.plan, expected_sha256=args.expected_plan_sha256
        )
        result = assess_prospective_acquisition_plan(
            document, document_bytes=payload
        ).to_dict()
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "INVALID_PROSPECTIVE_ACQUISITION_PLAN_NO_GO",
                    "execution_authorized": False,
                    "scientific_admission": False,
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
