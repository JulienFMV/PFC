"""Audit the Swiss market-time regime from a sealed launcherless runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Sequence

from pfc_shaping.build_identity import SOURCE_REVISION
from pfc_shaping.durable_artifact import (
    durable_artifact_durability_classification,
    write_durable_exact_bytes,
)
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.governed_release_cli_contract import (
    ReleaseCliIdentityError,
    assert_installed_runtime_sealed,
)
from pfc_shaping.validation import ch_market_time_regime as regime_module
from pfc_shaping.validation.ch_market_time_regime import (
    ChMarketTimeRegimeError,
    assess_market_time_regime,
    canonical_json_bytes,
    load_market_time_regime_contract,
)

COMMAND_ID = "audit_ch_market_time_regime"
AUDIT_NAMESPACE = Path("build") / "market-time-regime-audits"


class _StructuredArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(f"invalid arguments: {message}")


def audit_market_time_regime(
    *,
    contract_path: Path,
    expected_contract_sha256: str,
    repo_root: Path,
    runtime_receipt_path: Path,
    expected_runtime_receipt_sha256: str,
    output_path: Path,
) -> dict[str, object]:
    root = _canonical_repo_root(repo_root)
    output = _audit_output(output_path, repo_root=root)
    runtime_receipt = _absolute_regular_file(
        runtime_receipt_path, label="launcherless runtime receipt"
    )
    expected_runtime_sha256 = _canonical_sha256(
        expected_runtime_receipt_sha256, label="runtime receipt"
    )
    assert_installed_runtime_sealed(
        runtime_receipt_path=runtime_receipt,
        expected_runtime_receipt_sha256=expected_runtime_sha256,
    )

    document, contract_bytes = load_market_time_regime_contract(
        contract_path,
        expected_sha256=expected_contract_sha256,
        repo_root=root,
    )
    assessment = assess_market_time_regime(
        document,
        document_bytes=contract_bytes,
        repo_root=root,
        verify_evidence_files=True,
    ).to_dict()
    runtime_identity = _runtime_identity(
        runtime_receipt=runtime_receipt,
        expected_runtime_receipt_sha256=expected_runtime_sha256,
    )
    command_contract = {
        "command_id": COMMAND_ID,
        "repo_root": str(root),
        "cwd": str(Path.cwd()),
        "contract_path": str(_absolute_regular_file(contract_path, label="contract")),
        "expected_contract_sha256": expected_contract_sha256,
        "runtime_receipt_path": str(runtime_receipt),
        "expected_runtime_receipt_sha256": expected_runtime_sha256,
        "output_path": str(output),
        "verify_evidence_files": True,
    }
    assessment.update(
        {
            "receipt_schema_version": "ch_market_time_regime_audit_receipt.v1",
            "receipt_status": "LOCAL_DIAGNOSTIC_NOT_AUTHORITY",
            "command_id": COMMAND_ID,
            "command_contract": command_contract,
            "normalized_command_sha256": hashlib.sha256(
                canonical_json_bytes(command_contract)
            ).hexdigest(),
            "runtime_identity": runtime_identity,
            "cwd_equals_repo_root": True,
            "git_root_marker_bound": True,
            "command_exit_code": 0,
        }
    )
    operation = {
        "contract_id": assessment["contract_id"],
        "document_sha256": assessment["document_sha256"],
        "normalized_command_sha256": assessment["normalized_command_sha256"],
        "runtime_identity": runtime_identity,
        "blockers": assessment["blockers"],
    }
    assessment["audit_operation_id"] = hashlib.sha256(
        canonical_json_bytes(operation)
    ).hexdigest()
    payload = canonical_json_bytes(assessment) + b"\n"

    # Re-admit the exact runtime, contract and evidence immediately before the
    # durable write so a same-user mutation cannot become a successful receipt.
    assert_installed_runtime_sealed(
        runtime_receipt_path=runtime_receipt,
        expected_runtime_receipt_sha256=expected_runtime_sha256,
    )
    second_document, second_bytes = load_market_time_regime_contract(
        contract_path,
        expected_sha256=expected_contract_sha256,
        repo_root=root,
    )
    second = assess_market_time_regime(
        second_document,
        document_bytes=second_bytes,
        repo_root=root,
        verify_evidence_files=True,
    ).to_dict()
    if second != {key: assessment[key] for key in second}:
        raise ChMarketTimeRegimeError("market-time evidence changed before receipt write")

    write_durable_exact_bytes(
        output,
        payload,
        label="CH market-time regime audit receipt",
    )
    if read_stable_single_link_file(
        output,
        label="CH market-time regime audit receipt",
        max_bytes=len(payload),
    ) != payload:
        raise ChMarketTimeRegimeError("market-time audit receipt differs after write")
    return assessment


def _runtime_identity(
    *, runtime_receipt: Path, expected_runtime_receipt_sha256: str
) -> dict[str, object]:
    executable = _absolute_regular_file(Path(sys.executable), label="runtime Python")
    cli_path = _absolute_regular_file(Path(__file__), label="market-time CLI module")
    validator_path = _absolute_regular_file(
        Path(regime_module.__file__), label="market-time validator module"
    )
    return {
        "implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "source_revision": SOURCE_REVISION,
        "executable": str(executable),
        "executable_sha256": hashlib.sha256(
            read_stable_single_link_file(executable, label="runtime Python")
        ).hexdigest(),
        "sys_path": list(sys.path),
        "runtime_receipt_path": str(runtime_receipt),
        "runtime_receipt_sha256": expected_runtime_receipt_sha256,
        "cli_module_path": str(cli_path),
        "cli_module_sha256": hashlib.sha256(
            read_stable_single_link_file(cli_path, label="market-time CLI module")
        ).hexdigest(),
        "validator_module_path": str(validator_path),
        "validator_module_sha256": hashlib.sha256(
            read_stable_single_link_file(
                validator_path, label="market-time validator module"
            )
        ).hexdigest(),
        "durability_classification": durable_artifact_durability_classification(),
    }


def _canonical_repo_root(path: Path) -> Path:
    root = assert_absolute_path_has_no_links(path)
    if not root.is_dir():
        raise ValueError("repository root is unavailable")
    cwd = assert_absolute_path_has_no_links(Path.cwd())
    if os.path.normcase(str(cwd)) != os.path.normcase(str(root)):
        raise ValueError("current directory must equal the canonical repository root")
    git_marker = assert_absolute_path_has_no_links(root / ".git")
    if not git_marker.exists() or not (root / "AGENTS.md").is_file():
        raise ValueError("repository root markers are invalid")
    return root


def _audit_output(path: Path, *, repo_root: Path) -> Path:
    output = assert_absolute_path_has_no_links(path)
    allowed = repo_root / AUDIT_NAMESPACE
    if not allowed.is_dir():
        raise ValueError("market-time audit namespace is unavailable")
    try:
        output.relative_to(allowed)
    except ValueError as exc:
        raise ValueError("market-time audit output must stay below repo build namespace") from exc
    if output.parent != allowed or output.suffix != ".json":
        raise ValueError("market-time audit output must be one JSON file in its namespace")
    return output


def _absolute_regular_file(path: Path, *, label: str) -> Path:
    selected = assert_absolute_path_has_no_links(path)
    if not selected.is_file():
        raise ValueError(f"{label} is unavailable")
    return selected


def _canonical_sha256(value: str, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{label} SHA-256 is invalid")
    return normalized


def _parser() -> argparse.ArgumentParser:
    parser = _StructuredArgumentParser(description=__doc__)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--expected-runtime-receipt-sha256", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _failure(status: str, exc: Exception) -> None:
    print(
        json.dumps(
            {
                "receipt_schema_version": "ch_market_time_regime_audit_failure.v1",
                "status": status,
                "command_id": COMMAND_ID,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "scientific_admission": False,
                "execution_authorized": False,
                "production_authorization": False,
                "promotion_gate": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=sys.stderr,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except ValueError as exc:
        _failure("INVALID_CH_MARKET_TIME_REGIME_ARGUMENTS", exc)
        return 2
    try:
        result = audit_market_time_regime(
            contract_path=args.contract,
            expected_contract_sha256=args.expected_contract_sha256,
            repo_root=args.repo_root,
            runtime_receipt_path=args.runtime_receipt,
            expected_runtime_receipt_sha256=args.expected_runtime_receipt_sha256,
            output_path=args.output,
        )
    except ReleaseCliIdentityError as exc:
        _failure("INVALID_GOVERNED_LT_RUNTIME", exc)
        return 2
    except (ChMarketTimeRegimeError, OSError, ValueError) as exc:
        _failure("INVALID_CH_MARKET_TIME_REGIME", exc)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
