#!/usr/bin/env python3
"""Build or replay policy-bound product evidence for one staged LT candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pfc_shaping.pipeline.candidate_product_evidence import (
    assemble_candidate_product_evidence,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-hierarchy-policy", type=Path, default=None)
    args = parser.parse_args(argv)
    result = assemble_candidate_product_evidence(
        args.staging,
        run_id=args.run_id,
        source_hierarchy_policy_path=args.source_hierarchy_policy,
    )
    print(json.dumps(result, sort_keys=True, default=str))
    return 0 if result.get("schema_version") == "candidate_product_normalization_evidence.v1" else 2


if __name__ == "__main__":
    raise SystemExit(main())
