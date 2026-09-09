"""Audit one hash-bound CH LT compute-runtime manifest and its local receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from pfc_shaping.build_identity import SOURCE_REVISION
from pfc_shaping.path_safety import assert_absolute_path_has_no_links
from pfc_shaping.pipeline.governed_release_cli_contract import (
    ReleaseCliIdentityError,
    assert_installed_runtime_sealed,
)
from pfc_shaping.validation.ch_lt_compute_runtime import (
    canonical_json_bytes,
    load_compute_runtime_contract,
)
from pfc_shaping.validation.ch_lt_compute_runtime_manifest import (
    assess_compute_runtime_manifest,
    load_compute_runtime_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--evidence",
        action="append",
        default=[],
        metavar="ROLE=ABSOLUTE_PATH",
        help="Repeat once for every exact receipt/evidence role.",
    )
    return parser


def _parse_evidence_paths(entries: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError("--evidence must use ROLE=ABSOLUTE_PATH")
        role, raw_path = entry.split("=", 1)
        if not role or not raw_path or role in result:
            raise ValueError("--evidence roles and paths must be non-empty and unique")
        result[role] = Path(raw_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        assert_installed_runtime_sealed()
    except ReleaseCliIdentityError as exc:
        print(
            json.dumps(
                {
                    "status": "INVALID_GOVERNED_LT_RUNTIME",
                    "command_id": "audit_ch_lt_compute_runtime_manifest",
                    "structure_valid": False,
                    "receipt_and_bound_payload_bytes_verified": False,
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
    started_at = datetime.now(timezone.utc)
    started_ns = time.perf_counter_ns()
    try:
        evidence_paths = _parse_evidence_paths(args.evidence)
        contract, contract_bytes = load_compute_runtime_contract(
            args.contract,
            expected_sha256=args.expected_contract_sha256,
        )
        manifest, manifest_bytes = load_compute_runtime_manifest(
            args.manifest,
            expected_sha256=args.expected_manifest_sha256,
        )
        result = assess_compute_runtime_manifest(
            manifest,
            manifest_bytes=manifest_bytes,
            compute_contract=contract,
            compute_contract_bytes=contract_bytes,
            evidence_root=args.evidence_root,
            evidence_paths=evidence_paths,
        ).to_dict()
    except (OSError, UnicodeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "INVALID_CH_LT_COMPUTE_RUNTIME_MANIFEST",
                    "structure_valid": False,
                    "receipt_and_bound_payload_bytes_verified": False,
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
    completed_at = datetime.now(timezone.utc)
    runtime_identity = {
        "implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "source_revision": SOURCE_REVISION,
        "contract_path": str(assert_absolute_path_has_no_links(args.contract)),
        "manifest_path": str(assert_absolute_path_has_no_links(args.manifest)),
        "evidence_root": str(assert_absolute_path_has_no_links(args.evidence_root)),
        "evidence_paths": {
            role: str(assert_absolute_path_has_no_links(path))
            for role, path in sorted(evidence_paths.items())
        },
    }
    result.update(
        {
            "audit_started_at": started_at.isoformat().replace("+00:00", "Z"),
            "audit_completed_at": completed_at.isoformat().replace("+00:00", "Z"),
            "duration_ms": (time.perf_counter_ns() - started_ns) / 1_000_000,
            "runtime_identity": runtime_identity,
        }
    )
    result["audit_operation_id"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
