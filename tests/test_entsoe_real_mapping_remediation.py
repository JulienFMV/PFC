from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pfc_shaping.validation.entsoe_real_mapping_remediation import (
    CONTRACT_CONTENT_ID,
    CONTRACT_FILE_SHA256,
    EntsoeRealMappingRemediationError,
    assess_real_mapping_remediation,
    canonical_sha256,
    data_engineer_request_lines,
    validate_real_mapping_remediation_contract,
    verify_real_mapping_remediation_paths,
)

ROOT = Path(__file__).resolve().parents[1]
PLANNING = ROOT / ".planning" / "phases" / "14-lt-audit-remediation"
CONTRACT_PATH = PLANNING / "ENTSOE-REAL-MAPPING-REMEDIATION-CONTRACT-V1.json"
D241_ROOT = (
    ROOT
    / "build"
    / "databricks-control-plane"
    / "2026-08-06"
    / "entsoe-unity-catalog-schema-captures"
    / "d6c006609d881b51f08be6d60e01f68b59a40be8bdf2898ef0a98491f5771544"
)
D241_MANIFEST_PATH = D241_ROOT / "manifest.json"
D241_CAPTURE_PATH = D241_ROOT / "capture.json"
D241_ASSESSMENT_PATH = D241_ROOT / "assessment.json"
D241_VALIDATOR_PATH = (
    ROOT / "pfc_shaping" / "validation" / "entsoe_unity_catalog_schema_compatibility.py"
)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _contract() -> dict[str, object]:
    return _json(CONTRACT_PATH)


def _capture() -> dict[str, object]:
    return _json(D241_CAPTURE_PATH)


def _prior_assessment() -> dict[str, object]:
    return _json(D241_ASSESSMENT_PATH)


def _path_result() -> dict[str, object]:
    return verify_real_mapping_remediation_paths(
        contract_path=CONTRACT_PATH.resolve(),
        d241_manifest_path=D241_MANIFEST_PATH.resolve(),
        d241_capture_path=D241_CAPTURE_PATH.resolve(),
        d241_assessment_path=D241_ASSESSMENT_PATH.resolve(),
        d241_validator_path=D241_VALIDATOR_PATH.resolve(),
    )


def test_contract_identity_and_all_authority_stays_closed() -> None:
    contract = _contract()
    result = validate_real_mapping_remediation_contract(contract)
    assert canonical_sha256(contract) == CONTRACT_CONTENT_ID
    assert CONTRACT_FILE_SHA256 == "4a0d69a770fe0b45063bc4e5b4be4b9aca1e86c546b40d2bf8816a001cd06b8f"
    assert result["mapping_complete"] is False
    assert result["sql_generated"] is False
    assert result["model_input_authorized"] is False
    assert result["production_authorized"] is False


def test_exact_d241_bytes_produce_value_free_remediation_no_go() -> None:
    result = _path_result()
    assert result["status"] == "FAIL_REAL_MAPPING_REMEDIATION_REQUIRED_NO_MODEL_AUTHORITY"
    assert result["physical_column_counts"] == {
        "dimension": 11,
        "latest": 8,
        "vintages": 10,
    }
    assert result["target_timestamp_policy"] == {
        "physical_semantics": "RIGHT_EDGE",
        "normalized_semantics": "INTERVAL_START",
        "required_transform": "RIGHT_EDGE_MINUS_NATIVE_RESOLUTION",
        "ready": False,
    }
    assert all(value == 0 for value in result["execution"].values())
    assert result["authorities"]["real_schema_observed"] is True
    assert result["authorities"]["real_values_opened"] is False
    assert result["authorities"]["model_input_authorized"] is False


def test_right_edge_is_never_admitted_as_direct_target_timestamp() -> None:
    result = assess_real_mapping_remediation(_capture(), _prior_assessment())
    codes = {finding["code"] for finding in result["findings"]}
    assert "RIGHT_EDGE_CANNOT_BE_DIRECT_TARGET_TIMESTAMP" in codes
    assert "NATIVE_RESOLUTION_REQUIRED_FOR_INTERVAL_START" in codes
    for role in ("latest", "vintages"):
        assert "target_ts_utc" in result["blocked_normalized_fields"][role]
    plan = _contract()["mapping_plan"]
    assert plan["latest"]["fields"]["target_ts_utc"]["transform"] == (
        "RIGHT_EDGE_MINUS_NATIVE_RESOLUTION"
    )
    assert plan["vintages"]["fields"]["target_ts_utc"]["transform"] == (
        "RIGHT_EDGE_MINUS_NATIVE_RESOLUTION"
    )


def test_conservative_pit_candidate_cannot_self_approve() -> None:
    result = assess_real_mapping_remediation(_capture(), _prior_assessment())
    pit = result["point_in_time_policy"]
    assert pit["candidate"] == (
        "MAX_OF_PUBLICATION_PULL_AND_LOAD_AFTER_ALL_THREE_ARE_NON_NULL"
    )
    assert pit["all_components_non_null_required"] is True
    assert pit["owner_approved"] is False
    assert pit["latest_allowed_for_historical_replay"] is False
    assert result["readiness_gates"]["point_in_time_rule_owner_approved"] is False


def test_data_engineer_request_is_short_and_forbids_dangerous_defaults() -> None:
    lines = data_engineer_request_lines(_path_result())
    text = "\n".join(lines)
    assert len(lines) == 9
    assert "right edge minus native resolution" in text
    assert "as_of_utc=max(all three)" in text
    assert "DocumentType and VintageID are not substitutes" in text
    assert "never default unknown quality to OK or revision to zero" in text
    assert "No Databricks write is required or authorized" in text


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["normalized_time_semantics"].__setitem__(
                "direct_alias_of_right_edge_to_target_start_authorized", True
            ),
            "normalized time semantics",
        ),
        (
            lambda value: value["point_in_time_policy"].__setitem__(
                "publication_timestamp_alone_authorized", True
            ),
            "point-in-time policy",
        ),
        (
            lambda value: value["forbidden_substitutions"].__setitem__(
                "quality_flag_default_OK", True
            ),
            "forbidden substitution",
        ),
        (
            lambda value: value["readiness_gates"].__setitem__(
                "real_mapping_compilable_by_D234", True
            ),
            "readiness gates",
        ),
        (
            lambda value: value["authorities"].__setitem__(
                "model_input_authorized", True
            ),
            "authorities",
        ),
    ],
)
def test_contract_rejects_authority_or_semantic_relaxation(mutation, match: str) -> None:
    contract = _contract()
    mutation(contract)
    with pytest.raises(EntsoeRealMappingRemediationError, match=match):
        validate_real_mapping_remediation_contract(contract)


def test_capture_right_edge_declaration_drift_is_rejected() -> None:
    capture = _capture()
    for table in capture["tables"]:
        if table["full_name"].endswith(("latest", "vintages")):
            for column in table["columns"]:
                if column["name"] == "DateTimeUtc":
                    column["comment"] = "UTC timestamp."
                    break
            break
    prior = _prior_assessment()
    # Replay must fail first because the independently captured assessment no
    # longer corresponds to the mutated schema.
    with pytest.raises(EntsoeRealMappingRemediationError, match="assessment replay"):
        assess_real_mapping_remediation(capture, prior)


def test_prior_assessment_cannot_be_replaced_by_caller_claim() -> None:
    prior = copy.deepcopy(_prior_assessment())
    prior["status"] = "PASS"
    prior["model_input_authorized"] = True
    with pytest.raises(EntsoeRealMappingRemediationError, match="assessment replay"):
        assess_real_mapping_remediation(_capture(), prior)


def test_path_verifier_rejects_tampered_capture_before_semantic_use(tmp_path: Path) -> None:
    tampered = tmp_path / "capture.json"
    payload = D241_CAPTURE_PATH.read_bytes().replace(b'"SeriesID"', b'"SeriesIX"', 1)
    tampered.write_bytes(payload)
    with pytest.raises(EntsoeRealMappingRemediationError, match="capture SHA-256"):
        verify_real_mapping_remediation_paths(
            contract_path=CONTRACT_PATH.resolve(),
            d241_manifest_path=D241_MANIFEST_PATH.resolve(),
            d241_capture_path=tampered.resolve(),
            d241_assessment_path=D241_ASSESSMENT_PATH.resolve(),
            d241_validator_path=D241_VALIDATOR_PATH.resolve(),
        )


def test_mapping_plan_does_not_misuse_document_or_vintage_surrogates() -> None:
    contract = _contract()
    forbidden = contract["forbidden_substitutions"]
    assert forbidden["DocumentType_as_document_id"] is False
    assert forbidden["VintageID_as_source_document_or_revision"] is False
    assert forbidden["missing_revision_default_zero"] is False
    assert forbidden["directional_sign_inferred_from_column_names_only"] is False
