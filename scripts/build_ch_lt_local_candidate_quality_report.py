"""Build the evidence-bound non-T057 quality report for the current CH LT candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfc_shaping.validation.ch_lt_local_candidate_quality import (
    build_local_candidate_quality_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion-manifest", type=Path, required=True)
    parser.add_argument("--expected-completion-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        report = build_local_candidate_quality_report(
            repo_root=root,
            completion_manifest=args.completion_manifest,
            expected_completion_sha256=args.expected_completion_sha256,
            output_json=args.output_json,
            output_markdown=args.output_markdown,
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "ch_lt_local_candidate_quality_error.v1",
                    "status": "INVALID_LOCAL_CANDIDATE_EVIDENCE_NO_GO",
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
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
