"""Build a sealed-runtime, local-only ledger of contiguous CH hourly captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from pfc_shaping.build_identity import SOURCE_REVISION
from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.pipeline.governed_release_cli_contract import (
    RUNTIME_RECEIPT_PATH_ENV,
    RUNTIME_RECEIPT_SHA256_ENV,
    ReleaseCliIdentityError,
    assert_installed_runtime_sealed,
)
from pfc_shaping.validation.ch_lt_prospective_capture_ledger import (
    build_prospective_capture_ledger,
    canonical_json_bytes,
)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        assert_installed_runtime_sealed()
    except ReleaseCliIdentityError as exc:
        _failure(exc, status="UNSEALED_RUNTIME_NO_GO")
        return 50
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--expected-request-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-receipt-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        root = Path(os.path.abspath(args.repo_root))
        build_root = root / "build"
        if os.path.normcase(os.path.abspath(Path.cwd())) != os.path.normcase(str(root)):
            raise ValueError("cwd must be exactly the canonical repo root")
        selected_output = Path(os.path.abspath(args.output))
        if not _within(selected_output, build_root) or selected_output == build_root:
            raise ValueError("ledger output must remain below repo build")
        selected_execution_receipt = Path(
            os.path.abspath(args.execution_receipt_output)
        )
        if (
            not _within(selected_execution_receipt, build_root)
            or selected_execution_receipt == build_root
        ):
            raise ValueError("execution receipt must remain below repo build")
        if os.path.normcase(str(selected_execution_receipt)) == os.path.normcase(
            str(selected_output)
        ):
            raise ValueError("execution receipt and ledger outputs must differ")
        ledger = build_prospective_capture_ledger(
            repo_root=root,
            request_path=args.request,
            expected_request_sha256=args.expected_request_sha256,
        )
        payload = canonical_json_bytes(ledger) + b"\n"
        write_durable_exact_bytes(
            selected_output,
            payload,
            label="sealed-runtime CH prospective capture ledger",
        )
        execution_receipt = {
            "schema_version": (
                "ch_lt_prospective_capture_ledger_execution_receipt.v1"
            ),
            "status": "PASS_LOCAL_CONSTRUCTION_OBSERVATION_NOT_AUTHORITY",
            "command": "build-ch-lt-prospective-capture-ledger",
            "project_source_revision": SOURCE_REVISION,
            "cwd": str(Path.cwd()),
            "sys_path": list(sys.path),
            "request": {
                "path": str(Path(os.path.abspath(args.request))),
                "sha256": args.expected_request_sha256,
            },
            "ledger": {
                "path": str(selected_output),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "ledger_id": ledger["ledger_id"],
                "status": ledger["status"],
            },
            "constructor_runtime_receipt": {
                "path": os.environ[RUNTIME_RECEIPT_PATH_ENV],
                "sha256": os.environ[RUNTIME_RECEIPT_SHA256_ENV],
                "admission_check": "assert_installed_runtime_sealed",
            },
            "runtime_authority": False,
            "scientific_admission": False,
            "production_authorization": False,
            "promotion_gate": False,
            "future_holdout_consumed": False,
            "t057_consumed": False,
        }
        execution_receipt_payload = canonical_json_bytes(execution_receipt) + b"\n"
        write_durable_exact_bytes(
            selected_execution_receipt,
            execution_receipt_payload,
            label="sealed-runtime CH prospective ledger execution receipt",
        )
    except (OSError, ValueError) as exc:
        _failure(exc, status="INVALID_PROSPECTIVE_CAPTURE_LEDGER_NO_GO")
        return 2
    print(
        json.dumps(
            {
                "schema_version": "ch_lt_prospective_capture_ledger_cli_result.v1",
                "status": ledger["status"],
                "ledger_id": ledger["ledger_id"],
                "output": str(selected_output),
                "output_sha256": hashlib.sha256(payload).hexdigest(),
                "constructor_runtime_receipt": {
                    "path": os.environ[RUNTIME_RECEIPT_PATH_ENV],
                    "sha256": os.environ[RUNTIME_RECEIPT_SHA256_ENV],
                },
                "execution_receipt": {
                    "path": str(selected_execution_receipt),
                    "sha256": hashlib.sha256(
                        execution_receipt_payload
                    ).hexdigest(),
                },
                "runtime_authority": False,
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


def _within(path: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath(
            (os.path.normcase(str(path)), os.path.normcase(str(parent)))
        ) == os.path.normcase(str(parent))
    except ValueError:
        return False


def _failure(exc: Exception, *, status: str) -> None:
    print(
        json.dumps(
            {
                "schema_version": "ch_lt_prospective_capture_ledger_cli_result.v1",
                "status": status,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "runtime_authority": False,
                "scientific_admission": False,
                "production_authorization": False,
                "promotion_gate": False,
                "future_holdout_consumed": False,
                "t057_consumed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
