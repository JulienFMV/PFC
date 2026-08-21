from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pfc_shaping.validation.entsoe_bounded_execution_receipt import (
    AUTHORITIES,
    CONTRACT_CONTENT_ID,
    PROPOSAL_SCHEMA,
    RECEIPT_SCHEMA,
    ROLE_BYTE_CAPS,
    ROLE_LIMITS,
    ROLE_ORDER,
    EntsoeBoundedExecutionReceiptError,
    canonical_sha256,
    validate_bounded_capture_proposal,
    validate_bounded_execution_contract,
    validate_synthetic_capture_receipt,
    verify_bounded_execution_paths,
)

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / ".planning" / "phases" / "14-lt-audit-remediation"
CONTRACT_PATH = (
    PHASE / "ENTSOE-BOUNDED-EXECUTION-PARAMETER-RECEIPT-CONTRACT-V1.json"
)
GOVERNED_CONTRACT_PATH = (
    PHASE / "EEX-ENTSOE-GOVERNED-ACQUISITION-PACKAGE-CONTRACT-V1.json"
)
MAPPING_CONTRACT_PATH = PHASE / "ENTSOE-PHYSICAL-MAPPING-COMPILER-CONTRACT-V1.json"
PROOF_ROOT = ROOT / "build" / "databricks-eex-daily" / "2026-08-05"
GOVERNED_PROOF_PATH = (
    PROOF_ROOT
    / "eex-entsoe-governed-acquisition-package-proofs"
    / "314ec85590c787874e2844d7db085236144c601681a5d1722f2735e6b1219d53"
    / "manifest.json"
)
MAPPING_PROOF_PATH = (
    PROOF_ROOT
    / "entsoe-physical-mapping-compiler-proofs"
    / "bb7ea1894463cb2c5fc30287d2239f0285cd6a74e901ab51a2f9de7e6794b766"
    / "manifest.json"
)
REFERENCE = datetime(2026, 8, 5, 13, 0, tzinfo=timezone.utc)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _proposal() -> dict[str, object]:
    return {
        "schema_version": PROPOSAL_SCHEMA,
        "evidence_class": "SYNTHETIC_TEST_ONLY",
        "capture_batch_id": "11111111-1111-4111-8111-111111111111",
        "created_at_utc": "2026-08-05T11:00:00Z",
        "snapshot_reference_time_utc": "2026-08-05T10:00:00Z",
        "challenge_id": "22222222-2222-4222-8222-222222222222",
        "challenge_nonce_sha256": _sha("nonce"),
        "mapping_content_id": (
            "5c69594f76e2a4a13ba091d7bf585c64961389f9bd143e99eef16f00d8f22d87"
        ),
        "metadata_schema_fingerprint": _sha("metadata schema"),
        "metadata_payload_sha256": _sha("metadata payload"),
        "metadata_statement_id": "33333333-3333-4333-8333-333333333333",
        "compiled_template_sha256": {
            "series_dimension": (
                "3423c4723fe9626370bf2168d41917c667cd85c99646245d15869437bfb59170"
            ),
            "latest_values": (
                "5e0c233af7ec385757a5f378e5afbbc84af66c914d4c7d6b72440ad2ec556fb6"
            ),
            "vintage_values": (
                "310a650dcc0e4acfc98414bb04a69fe4ca6a448e6745c62ce8c7aa7e51a72cdc"
            ),
        },
        "execution_parameters": {
            "series_dimension": {},
            "latest_values": {
                "start_utc": "2026-07-01T00:00:00Z",
                "end_utc": "2026-07-31T00:00:00Z",
            },
            "vintage_values": {
                "start_utc": "2026-07-01T00:00:00Z",
                "end_utc": "2026-07-31T00:00:00Z",
                "as_of_cutoff_utc": "2026-07-31T00:00:00Z",
            },
        },
        "cost_fence": {
            "currency": "CHF",
            "estimated_cost_amount": "0.50",
            "hard_cost_cap_amount": "1.00",
            "estimate_method": "owner_calculator_v1",
            "warehouse_size": "2XS",
            "warehouse_auto_stop_minutes": 5,
            "max_warehouse_starts": 1,
            "max_read_statements": 3,
            "statement_timeout_seconds": 300,
            "batch_timeout_seconds": 900,
            "max_retries": 0,
            "max_total_local_export_bytes": 68_719_476_736,
            "per_role_max_rows": dict(ROLE_LIMITS),
            "per_role_max_bytes": dict(ROLE_BYTE_CAPS),
            "limit_hit_requires_rejection": True,
            "cap_breach_requires_rejection": True,
            "retry_after_failure_authorized": False,
        },
        "authorization": {
            "explicit_user_authorization_present": False,
            "authorization_id": None,
            "execution_authorized": False,
        },
    }


def _receipt(proposal: dict[str, object]) -> dict[str, object]:
    templates = proposal["compiled_template_sha256"]
    parameters = proposal["execution_parameters"]
    assert isinstance(templates, dict)
    assert isinstance(parameters, dict)
    statements = []
    for index, role in enumerate(ROLE_ORDER, start=1):
        statements.append(
            {
                "role": role,
                "template_sha256": templates[role],
                "parameter_sha256": canonical_sha256(parameters[role]),
                "statement_id_sha256": _sha(f"statement {role}"),
                "statement_state": "SUCCEEDED",
                "statement_class": "SELECT",
                "statement_started_at_utc": "2026-08-05T12:00:00Z",
                "statement_completed_at_utc": "2026-08-05T12:00:00Z",
                "runtime_seconds": 0,
                "row_count": index,
                "size_bytes": index,
                "result_truncated": False,
                "limit_hit": False,
                "read_only_statement_verified": True,
                "remote_write_count": 0,
                "artifact_relative_path": (
                    f"build/governed-source-intake/synthetic/{role}.parquet"
                ),
                "artifact_sha256": _sha(f"artifact {role}"),
            }
        )
    return {
        "schema_version": RECEIPT_SCHEMA,
        "evidence_class": "SYNTHETIC_TEST_ONLY",
        "proposal_content_id": canonical_sha256(proposal),
        "capture_batch_id": proposal["capture_batch_id"],
        "snapshot_reference_time_utc": proposal["snapshot_reference_time_utc"],
        "capture_started_at_utc": "2026-08-05T12:00:00Z",
        "capture_completed_at_utc": "2026-08-05T12:00:00Z",
        "warehouse_start_count": 0,
        "read_statement_count": 3,
        "retry_count": 0,
        "total_runtime_seconds": 0,
        "total_local_export_bytes": 6,
        "statements": statements,
        "authorities": dict(AUTHORITIES),
    }


def test_contract_identity_and_zero_execution_authority() -> None:
    result = validate_bounded_execution_contract(
        json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    )
    assert result["contract_content_id"] == CONTRACT_CONTENT_ID
    assert result["databricks_connection_budget"] == 0
    assert result["databricks_statement_budget"] == 0
    assert result["warehouse_start_budget"] == 0
    assert result["execution_authorized"] is False


def test_proposal_is_valid_but_never_executable() -> None:
    result = validate_bounded_capture_proposal(_proposal(), reference_time_utc=REFERENCE)
    assert result["window_days"] == 30
    assert result["cost_fence_validated"] is True
    assert result["proposal_execution_authorized"] is False
    assert result["databricks_connection_count"] == 0
    assert result["warehouse_start_count"] == 0


def test_real_proposal_class_stays_unauthorized() -> None:
    proposal = _proposal()
    proposal["evidence_class"] = "REAL_UNAUTHORIZED_PROPOSAL"
    result = validate_bounded_capture_proposal(proposal, reference_time_utc=REFERENCE)
    assert result["evidence_class"] == "REAL_UNAUTHORIZED_PROPOSAL"
    assert result["proposal_execution_authorized"] is False
    with pytest.raises(EntsoeBoundedExecutionReceiptError, match="synthetic proposal"):
        validate_synthetic_capture_receipt(
            _receipt(proposal),
            proposal,
            reference_time_utc=REFERENCE,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda p: p.update(extra=True), "fields differ"),
        (
            lambda p: p["authorization"].update(execution_authorized=True),
            "authorization",
        ),
        (
            lambda p: p["cost_fence"].update(estimated_cost_amount="1.01"),
            "exceeds hard cap",
        ),
        (
            lambda p: p["cost_fence"].update(hard_cost_cap_amount="1e2"),
            "fixed decimal",
        ),
        (
            lambda p: p["cost_fence"].update(max_warehouse_starts=2),
            "max_warehouse_starts",
        ),
        (
            lambda p: p["cost_fence"].update(max_retries=1),
            "max_retries",
        ),
        (
            lambda p: p["compiled_template_sha256"].update(latest_values="A" * 64),
            "lowercase SHA-256",
        ),
        (
            lambda p: p["execution_parameters"]["latest_values"].update(
                end_utc="2026-08-02T00:00:00Z"
            ),
            "target window end",
        ),
        (
            lambda p: p["execution_parameters"]["latest_values"].update(
                end_utc="2026-08-02T00:00:00Z"
            )
            or p["execution_parameters"]["vintage_values"].update(
                end_utc="2026-08-02T00:00:00Z"
            ),
            "1..31 days",
        ),
        (
            lambda p: p["execution_parameters"]["vintage_values"].update(
                as_of_cutoff_utc="2026-08-05T10:00:01Z"
            ),
            "between target start and snapshot",
        ),
    ],
)
def test_proposal_rejects_unsafe_variants(mutation, match: str) -> None:
    proposal = _proposal()
    mutation(proposal)
    with pytest.raises(EntsoeBoundedExecutionReceiptError, match=match):
        validate_bounded_capture_proposal(proposal, reference_time_utc=REFERENCE)


def test_synthetic_receipt_structure_has_zero_execution_evidence() -> None:
    proposal = _proposal()
    result = validate_synthetic_capture_receipt(
        _receipt(proposal),
        proposal,
        reference_time_utc=REFERENCE,
    )
    assert result["synthetic_receipt_structure_verified"] is True
    assert result["synthetic_receipt_is_execution_evidence"] is False
    assert result["artifact_hashes_verified"] is False
    assert result["artifact_values_opened"] is False
    assert result["databricks_statement_count"] == 0
    assert result["warehouse_start_count"] == 0
    assert result["production_authorized"] is False


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda r: r.update(evidence_class="REAL_CAPTURE"),
            "receipt evidence",
        ),
        (
            lambda r: r.update(warehouse_start_count=1),
            "synthetic warehouse starts",
        ),
        (
            lambda r: r.update(retry_count=1),
            "retry count",
        ),
        (
            lambda r: r["statements"][0].update(row_count=ROLE_LIMITS[ROLE_ORDER[0]]),
            "row limit reached",
        ),
        (
            lambda r: r["statements"][1].update(
                size_bytes=ROLE_BYTE_CAPS[ROLE_ORDER[1]] + 1
            ),
            "byte cap exceeded",
        ),
        (
            lambda r: r["statements"][0].update(result_truncated=True),
            "truncation",
        ),
        (
            lambda r: r["statements"][0].update(limit_hit=True),
            "limit hit",
        ),
        (
            lambda r: r["statements"][0].update(read_only_statement_verified=False),
            "read-only",
        ),
        (
            lambda r: r["statements"][0].update(remote_write_count=1),
            "remote writes",
        ),
        (
            lambda r: r["statements"][0].update(
                artifact_relative_path="build/governed-source-intake/../secret"
            ),
            "artifact path is unsafe",
        ),
        (
            lambda r: r["statements"][2].update(parameter_sha256=_sha("tampered")),
            "parameter hash",
        ),
        (
            lambda r: r["statements"][0].update(runtime_seconds=1),
            "statement runtime differs",
        ),
    ],
)
def test_receipt_rejects_false_or_unbounded_evidence(mutation, match: str) -> None:
    proposal = _proposal()
    receipt = _receipt(proposal)
    mutation(receipt)
    with pytest.raises(EntsoeBoundedExecutionReceiptError, match=match):
        validate_synthetic_capture_receipt(
            receipt,
            proposal,
            reference_time_utc=REFERENCE,
        )


def test_receipt_rejects_totals_that_do_not_reconcile() -> None:
    proposal = _proposal()
    receipt = _receipt(proposal)
    receipt["total_local_export_bytes"] = 7
    with pytest.raises(EntsoeBoundedExecutionReceiptError, match="total export bytes"):
        validate_synthetic_capture_receipt(
            receipt,
            proposal,
            reference_time_utc=REFERENCE,
        )


def test_path_verifier_binds_d233_d234_and_d236(tmp_path: Path) -> None:
    proposal = _proposal()
    receipt = _receipt(proposal)
    proposal_path = tmp_path / "proposal.json"
    receipt_path = tmp_path / "receipt.json"
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    result = verify_bounded_execution_paths(
        contract_path=CONTRACT_PATH,
        governed_package_contract_path=GOVERNED_CONTRACT_PATH,
        governed_package_proof_manifest_path=GOVERNED_PROOF_PATH,
        mapping_compiler_contract_path=MAPPING_CONTRACT_PATH,
        mapping_compiler_proof_manifest_path=MAPPING_PROOF_PATH,
        proposal_path=proposal_path,
        receipt_path=receipt_path,
        reference_time_utc=REFERENCE,
    )
    assert result["contract_content_id"] == CONTRACT_CONTENT_ID
    assert result["synthetic_receipt_is_execution_evidence"] is False
    assert result["databricks_connection_count"] == 0


def test_path_verifier_rejects_duplicate_json_key(tmp_path: Path) -> None:
    proposal = _proposal()
    receipt = _receipt(proposal)
    proposal_path = tmp_path / "proposal.json"
    receipt_path = tmp_path / "receipt.json"
    proposal_path.write_text(
        json.dumps(proposal)[:-1] + ',"schema_version":"duplicate"}',
        encoding="utf-8",
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(EntsoeBoundedExecutionReceiptError, match="duplicate key"):
        verify_bounded_execution_paths(
            contract_path=CONTRACT_PATH,
            governed_package_contract_path=GOVERNED_CONTRACT_PATH,
            governed_package_proof_manifest_path=GOVERNED_PROOF_PATH,
            mapping_compiler_contract_path=MAPPING_CONTRACT_PATH,
            mapping_compiler_proof_manifest_path=MAPPING_PROOF_PATH,
            proposal_path=proposal_path,
            receipt_path=receipt_path,
            reference_time_utc=REFERENCE,
        )


def test_source_proof_manifest_hash_tamper_is_rejected(tmp_path: Path) -> None:
    proposal = _proposal()
    receipt = _receipt(proposal)
    proposal_path = tmp_path / "proposal.json"
    receipt_path = tmp_path / "receipt.json"
    proof_path = tmp_path / "mapping-proof.json"
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    proof = json.loads(MAPPING_PROOF_PATH.read_text(encoding="utf-8"))
    proof["execution"]["warehouse_start_count"] = 1
    proof_path.write_text(json.dumps(proof), encoding="utf-8")
    with pytest.raises(EntsoeBoundedExecutionReceiptError, match="SHA-256 differs"):
        verify_bounded_execution_paths(
            contract_path=CONTRACT_PATH,
            governed_package_contract_path=GOVERNED_CONTRACT_PATH,
            governed_package_proof_manifest_path=GOVERNED_PROOF_PATH,
            mapping_compiler_contract_path=MAPPING_CONTRACT_PATH,
            mapping_compiler_proof_manifest_path=proof_path,
            proposal_path=proposal_path,
            receipt_path=receipt_path,
            reference_time_utc=REFERENCE,
        )


def test_mutating_nested_cost_limits_does_not_mutate_module_constants() -> None:
    proposal = _proposal()
    proposal_copy = copy.deepcopy(proposal)
    proposal_copy["cost_fence"]["per_role_max_rows"]["series_dimension"] = 1
    assert ROLE_LIMITS["series_dimension"] == 100_001
