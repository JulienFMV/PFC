"""Durably write one local-only ledger of contiguous CH hourly captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.validation.ch_lt_prospective_capture_ledger import (
    build_prospective_capture_ledger,
    canonical_json_bytes,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--expected-request-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        ledger = build_prospective_capture_ledger(
            repo_root=args.repo_root,
            request_path=args.request,
            expected_request_sha256=args.expected_request_sha256,
        )
        payload = canonical_json_bytes(ledger) + b"\n"
        output = args.output
        if not output.is_absolute():
            raise ValueError("ledger output path must be absolute")
        build_root = (args.repo_root / "build").resolve(strict=True)
        selected = output.resolve(strict=False)
        if build_root not in selected.parents:
            raise ValueError("ledger output must remain below repo build")
        write_durable_exact_bytes(output, payload, label="CH prospective capture ledger")
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "INVALID_PROSPECTIVE_CAPTURE_LEDGER_NO_GO",
                    "scientific_admission": False,
                    "production_authorization": False,
                    "promotion_gate": False,
                    "future_holdout_consumed": False,
                    "t057_consumed": False,
                    "error": str(exc),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": ledger["status"],
                "ledger_id": ledger["ledger_id"],
                "output": str(output),
                "output_sha256": hashlib.sha256(payload).hexdigest(),
                "scientific_admission": False,
                "production_authorization": False,
                "promotion_gate": False,
                "future_holdout_consumed": False,
                "t057_consumed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
