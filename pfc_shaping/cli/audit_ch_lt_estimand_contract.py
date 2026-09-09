"""Audit the fail-closed Swiss LT estimand/economic design contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Sequence

from pfc_shaping.build_identity import SOURCE_REVISION
from pfc_shaping.durable_artifact import (
    durable_artifact_durability_classification,
    write_durable_exact_bytes,
)
from pfc_shaping.path_safety import assert_absolute_path_has_no_links
from pfc_shaping.pipeline.governed_release_cli_contract import (
    ReleaseCliIdentityError,
    assert_installed_runtime_sealed,
)
from pfc_shaping.validation.ch_lt_estimand_contract import (
    STATUS,
    assess_estimand_contract,
    canonical_json_bytes,
    load_estimand_contract,
)

AUDIT_MODES = ("validate-draft", "admit-evaluation")


def audit_estimand_contract(
    *,
    contract_path: Path,
    expected_contract_sha256: str,
    mode: str = "admit-evaluation",
    output_path: Path | None = None,
    audit_root: Path | None = None,
) -> dict[str, object]:
    if mode not in AUDIT_MODES:
        raise ValueError("estimand contract audit mode is invalid")
    document, payload = load_estimand_contract(
        contract_path,
        expected_sha256=expected_contract_sha256,
    )
    result = assess_estimand_contract(document, document_bytes=payload).to_dict()
    source_path = contract_path.expanduser()
    if not source_path.is_absolute():
        source_path = Path.cwd() / source_path
    source_path = assert_absolute_path_has_no_links(source_path)
    result.update(
        {
            "audit_mode": mode,
            "command_exit_code": _result_exit_code(result, mode=mode),
            "source_contract_path": str(source_path),
            "runtime_identity": {
                "implementation": platform.python_implementation(),
                "python_version": platform.python_version(),
                "source_revision": SOURCE_REVISION,
                "durability_classification": durable_artifact_durability_classification(),
            },
        }
    )
    operation = {
        "audit_mode": mode,
        "source_document_sha256": result["source_document_sha256"],
        "contract_id": result["contract_id"],
        "validator_policy_sha256": result["validator_policy_sha256"],
        "runtime_identity": result["runtime_identity"],
    }
    result["audit_operation_id"] = hashlib.sha256(canonical_json_bytes(operation)).hexdigest()
    if output_path is not None:
        if audit_root is None:
            raise ValueError("--audit-root is required when --output is used")
        target = _workspace_output(output_path, audit_root=audit_root)
        if target.exists():
            raise ValueError("estimand contract audit output already exists")
        write_durable_exact_bytes(
            target,
            canonical_json_bytes(result) + b"\n",
            label="CH LT estimand contract audit",
        )
    return result


def _result_exit_code(result: dict[str, object], *, mode: str) -> int:
    if mode == "validate-draft":
        return 0 if result["status"] == STATUS else 4
    return 0 if result["execution_authorized"] is True else 3


def _workspace_output(path: Path, *, audit_root: Path) -> Path:
    if not audit_root.is_absolute():
        raise ValueError("estimand contract audit root must be absolute")
    root = assert_absolute_path_has_no_links(audit_root)
    if not root.is_dir():
        raise ValueError("estimand contract audit root is unavailable")
    allowed_root = root / "phase14" / "ch_lt_estimand_contract_audits"
    if not path.is_absolute():
        raise ValueError("estimand contract audit output must be absolute")
    candidate = path
    candidate = assert_absolute_path_has_no_links(candidate)
    try:
        candidate.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(
            "estimand contract audit output must stay in the canonical audit namespace"
        ) from exc
    if candidate.suffix != ".json":
        raise ValueError("estimand contract audit output must use the .json suffix")
    if not candidate.parent.is_dir():
        raise ValueError("estimand contract audit output directory is unavailable")
    return candidate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-root", type=Path)
    parser.add_argument("--mode", choices=AUDIT_MODES, default="admit-evaluation")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        assert_installed_runtime_sealed()
    except ReleaseCliIdentityError as exc:
        print(
            json.dumps(
                {
                    "status": "INVALID_GOVERNED_LT_RUNTIME",
                    "command_id": "audit_ch_lt_estimand_contract",
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
    args = _parser().parse_args(argv)
    try:
        result = audit_estimand_contract(
            contract_path=args.contract,
            expected_contract_sha256=args.expected_contract_sha256,
            mode=args.mode,
            output_path=args.output,
            audit_root=args.audit_root,
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "INVALID_ESTIMAND_CONTRACT",
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
    return int(result["command_exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
