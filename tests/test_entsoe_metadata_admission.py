from __future__ import annotations

import hashlib
import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pfc_shaping.pipeline import production_phases
from pfc_shaping.validation import entsoe_metadata_admission as module
from pfc_shaping.validation.entsoe_metadata_admission import (
    CONTRACT_CONTENT_ID,
    CONTRACT_STATUS,
    EntsoeMetadataAdmissionError,
    admit_entsoe_metadata_capture,
    validate_metadata_admission_contract,
    verify_entsoe_metadata_capture_paths,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "ENTSOE-DATABRICKS-METADATA-ADMISSION-CONTRACT-V1.json"
)
CAPTURED_AT = "2026-08-05T12:00:00Z"
REFERENCE_TIME = datetime(2026, 8, 5, 12, 5, tzinfo=timezone.utc)


def _payload() -> bytes:
    return (
        "table_catalog,table_schema,table_name,column_name,ordinal_position,"
        "data_type,is_nullable\n"
        "dev,gold,dimentsoeseries,series_id,1,STRING,NO\n"
        "dev,gold,dimentsoeseries,group_name,2,STRING,YES\n"
        "dev,gold,factentsoetimeserieslatest,series_id,1,STRING,NO\n"
        "dev,gold,factentsoetimeserieslatest,target_ts_utc,2,TIMESTAMP,NO\n"
        "dev,gold,factentsoetimeserieslatest,value,3,DOUBLE,YES\n"
        "dev,gold,factentsoetimeseriesvintages,series_id,1,STRING,NO\n"
        "dev,gold,factentsoetimeseriesvintages,target_ts_utc,2,TIMESTAMP,NO\n"
        "dev,gold,factentsoetimeseriesvintages,as_of_utc,3,TIMESTAMP,NO\n"
        "dev,gold,factentsoetimeseriesvintages,value,4,DECIMAL,YES\n"
    ).encode("utf-8")


def _receipt(payload: bytes | None = None) -> dict[str, object]:
    content = _payload() if payload is None else payload
    return {
        "schema_version": "fmv_entsoe_databricks_metadata_capture_receipt.v1",
        "query_sha256": (
            "dec7e207603e3a8b69f5808b42454575f1b4a985a25fa8aca3e5c6a2c95b72fa"
        ),
        "statement_id": "01234567-89ab-4def-8123-456789abcdef",
        "statement_state": "SUCCEEDED",
        "captured_at_utc": CAPTURED_AT,
        "row_count": len(content.decode("utf-8").splitlines()) - 1,
        "result_truncated": False,
        "payload_sha256": hashlib.sha256(content).hexdigest(),
        "read_only_metadata_only": True,
        "remote_write_count": 0,
    }


def _admit(
    payload: bytes | None = None,
    *,
    receipt_changes: dict[str, object] | None = None,
    reference_time: datetime = REFERENCE_TIME,
) -> dict[str, object]:
    content = _payload() if payload is None else payload
    receipt = _receipt(content)
    if receipt_changes:
        receipt.update(receipt_changes)
    return admit_entsoe_metadata_capture(
        receipt,
        content,
        reference_time_utc=reference_time,
    )


def test_static_contract_is_exact_and_has_zero_current_budget() -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    result = validate_metadata_admission_contract(document)

    assert result == {
        "schema_version": "fmv_entsoe_databricks_metadata_admission_contract.v1",
        "status": CONTRACT_STATUS,
        "contract_content_id": CONTRACT_CONTENT_ID,
        "current_statement_budget": 0,
        "connector_code_allowed": False,
        "production_authorized": False,
    }


def test_valid_capture_profiles_schema_but_grants_no_model_authority() -> None:
    result = _admit()

    assert result["status"] == (
        "PASS_METADATA_SCHEMA_INTEGRITY_MAPPING_STILL_REQUIRED"
    )
    assert result["row_count"] == 9
    assert result["capture_age_seconds"] == 300
    assert len(result["schema_fingerprint"]) == 64
    assert result["table_profile"] == {
        "dimentsoeseries": {"column_count": 2, "nullable_count": 1},
        "factentsoetimeserieslatest": {"column_count": 3, "nullable_count": 1},
        "factentsoetimeseriesvintages": {
            "column_count": 4,
            "nullable_count": 1,
        },
    }
    assert result["metadata_schema_integrity_verified"] is True
    assert result["physical_column_mapping_verified"] is False
    assert result["entsoe_values_opened"] is False
    assert result["data_export_sql_generated"] is False
    assert result["training_authorized"] is False
    assert result["production_authorized"] is False
    assert result["validator_databricks_request_count"] == 0
    assert result["validator_warehouse_start_count"] == 0
    assert result["validator_network_call_count"] == 0
    assert result["validator_remote_write_count"] == 0


def test_path_verifier_binds_contract_receipt_and_payload(tmp_path: Path) -> None:
    payload_path = tmp_path / "metadata.csv"
    receipt_path = tmp_path / "receipt.json"
    payload_path.write_bytes(_payload())
    receipt_path.write_text(
        json.dumps(_receipt(), sort_keys=True),
        encoding="utf-8",
    )

    result = verify_entsoe_metadata_capture_paths(
        contract_path=CONTRACT_PATH.resolve(),
        receipt_path=receipt_path.resolve(),
        payload_path=payload_path.resolve(),
        reference_time_utc=REFERENCE_TIME,
    )

    assert result["contract_content_id"] == CONTRACT_CONTENT_ID
    assert result["payload_sha256"] == hashlib.sha256(_payload()).hexdigest()


@pytest.mark.parametrize(
    ("change", "value", "message"),
    [
        ("schema_version", "v2", "receipt schema"),
        ("query_sha256", "0" * 64, "query SHA-256"),
        ("statement_id", "NOT-A-UUID", "lowercase UUID"),
        ("statement_id", "01234567-89AB-4DEF-8123-456789ABCDEF", "lowercase UUID"),
        ("statement_state", "FAILED", "statement state"),
        ("captured_at_utc", "2026-08-05 12:00:00", "whole-second precision"),
        ("row_count", 8, "receipt row count"),
        ("row_count", True, "positive integer"),
        ("result_truncated", True, "result truncation"),
        ("payload_sha256", "0" * 64, "payload SHA-256"),
        ("read_only_metadata_only", False, "read-only metadata flag"),
        ("remote_write_count", 1, "remote write count"),
    ],
)
def test_receipt_mutations_fail_closed(
    change: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(EntsoeMetadataAdmissionError, match=message):
        _admit(receipt_changes={change: value})


def test_receipt_extra_or_missing_field_fails_closed() -> None:
    receipt = _receipt()
    receipt["warehouse_id"] = "invented"
    with pytest.raises(EntsoeMetadataAdmissionError, match="fields must match"):
        admit_entsoe_metadata_capture(
            receipt,
            _payload(),
            reference_time_utc=REFERENCE_TIME,
        )
    receipt = _receipt()
    receipt.pop("statement_state")
    with pytest.raises(EntsoeMetadataAdmissionError, match="fields must match"):
        admit_entsoe_metadata_capture(
            receipt,
            _payload(),
            reference_time_utc=REFERENCE_TIME,
        )


def test_future_capture_or_non_utc_reference_fails_closed() -> None:
    with pytest.raises(EntsoeMetadataAdmissionError, match="after reference time"):
        _admit(reference_time=REFERENCE_TIME - timedelta(minutes=10))
    with pytest.raises(EntsoeMetadataAdmissionError, match="timezone-aware UTC"):
        _admit(reference_time=REFERENCE_TIME.replace(tzinfo=None))
    offset = timezone(timedelta(hours=2))
    with pytest.raises(EntsoeMetadataAdmissionError, match="timezone-aware UTC"):
        _admit(reference_time=REFERENCE_TIME.astimezone(offset))


def test_stale_metadata_capture_fails_closed() -> None:
    with pytest.raises(EntsoeMetadataAdmissionError, match="metadata capture is stale"):
        _admit(reference_time=REFERENCE_TIME + timedelta(hours=2))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("bom", "BOM is forbidden"),
        ("crlf", "canonical LF"),
        ("no_final_newline", "canonical LF"),
        ("wrong_header", "header differs"),
        ("wrong_catalog", "catalog at row"),
        ("wrong_schema", "schema at row"),
        ("unknown_table", "unexpected target table"),
        ("missing_table", "target tables are missing"),
        ("unordered", "ordered by table"),
        ("ordinal_gap", "positive and contiguous"),
        ("ordinal_duplicate", "ordinal positions are duplicated"),
        ("case_duplicate", "case-insensitive column name is duplicated"),
        ("unsafe_identifier", "unsafe column identifier"),
        ("bad_type", "unsupported metadata data type"),
        ("bad_nullable", "is_nullable is invalid"),
        ("padded_field", "empty or padded field"),
    ],
)
def test_csv_mutations_fail_closed(mutation: str, message: str) -> None:
    text = _payload().decode("utf-8")
    if mutation == "bom":
        payload = b"\xef\xbb\xbf" + text.encode("utf-8")
    elif mutation == "crlf":
        payload = text.replace("\n", "\r\n").encode("utf-8")
    elif mutation == "no_final_newline":
        payload = text.rstrip("\n").encode("utf-8")
    elif mutation == "wrong_header":
        payload = text.replace("table_catalog", "catalog", 1).encode("utf-8")
    elif mutation == "wrong_catalog":
        payload = text.replace("dev,gold", "prd,gold", 1).encode("utf-8")
    elif mutation == "wrong_schema":
        payload = text.replace("dev,gold", "dev,silver", 1).encode("utf-8")
    elif mutation == "unknown_table":
        payload = text.replace("dimentsoeseries", "unknown", 1).encode("utf-8")
    elif mutation == "missing_table":
        payload = "\n".join(
            line
            for line in text.splitlines()
            if "factentsoetimeseriesvintages" not in line
        ).encode("utf-8") + b"\n"
    elif mutation == "unordered":
        lines = text.splitlines()
        lines[1], lines[3] = lines[3], lines[1]
        payload = "\n".join(lines).encode("utf-8") + b"\n"
    elif mutation == "ordinal_gap":
        payload = text.replace("group_name,2", "group_name,3", 1).encode("utf-8")
    elif mutation == "ordinal_duplicate":
        payload = text.replace("group_name,2", "group_name,1", 1).encode("utf-8")
    elif mutation == "case_duplicate":
        payload = text.replace("group_name,2", "SERIES_ID,2", 1).encode("utf-8")
    elif mutation == "unsafe_identifier":
        payload = text.replace("group_name,2", "group-name,2", 1).encode("utf-8")
    elif mutation == "bad_type":
        payload = text.replace("STRING,YES", "ARRAY<STRING>,YES", 1).encode("utf-8")
    elif mutation == "bad_nullable":
        payload = text.replace("DOUBLE,YES", "DOUBLE,MAYBE", 1).encode("utf-8")
    elif mutation == "padded_field":
        payload = text.replace("group_name,2", " group_name,2", 1).encode("utf-8")
    else:
        raise AssertionError(mutation)

    with pytest.raises(EntsoeMetadataAdmissionError, match=message):
        _admit(payload)


def test_limit_hit_is_rejected_as_potentially_truncated() -> None:
    header = _payload().decode("utf-8").splitlines()[0]
    lines = [header]
    for table_index, table in enumerate(module.TARGET_TABLES):
        count = 342 if table_index == 0 else 341
        for ordinal in range(1, count + 1):
            lines.append(
                f"dev,gold,{table},column_{ordinal},{ordinal},STRING,YES"
            )
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    assert len(lines) - 1 == 1_024

    with pytest.raises(EntsoeMetadataAdmissionError, match="outside bounds"):
        _admit(payload)


def test_contract_mutation_or_path_drift_fails_closed(tmp_path: Path) -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    document["execution_policy"]["current_databricks_statement_budget"] = 1
    with pytest.raises(EntsoeMetadataAdmissionError):
        validate_metadata_admission_contract(document)

    changed_contract = tmp_path / "contract.json"
    changed_contract.write_bytes(CONTRACT_PATH.read_bytes() + b" ")
    receipt = tmp_path / "receipt.json"
    payload = tmp_path / "metadata.csv"
    receipt.write_text(json.dumps(_receipt()), encoding="utf-8")
    payload.write_bytes(_payload())
    with pytest.raises(EntsoeMetadataAdmissionError, match="contract SHA differs"):
        verify_entsoe_metadata_capture_paths(
            contract_path=changed_contract.resolve(),
            receipt_path=receipt.resolve(),
            payload_path=payload.resolve(),
            reference_time_utc=REFERENCE_TIME,
        )


def test_all_paths_must_be_absolute() -> None:
    with pytest.raises(EntsoeMetadataAdmissionError, match="path must be absolute"):
        verify_entsoe_metadata_capture_paths(
            contract_path=Path("relative-contract.json"),
            receipt_path=Path("relative-receipt.json"),
            payload_path=Path("relative-metadata.csv"),
            reference_time_utc=REFERENCE_TIME,
        )


def test_duplicate_receipt_json_key_fails_closed(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    payload = tmp_path / "metadata.csv"
    receipt.write_text(
        '{"schema_version":"v1","schema_version":"v2"}',
        encoding="utf-8",
    )
    payload.write_bytes(_payload())

    with pytest.raises(EntsoeMetadataAdmissionError, match="duplicate key"):
        verify_entsoe_metadata_capture_paths(
            contract_path=CONTRACT_PATH.resolve(),
            receipt_path=receipt.resolve(),
            payload_path=payload.resolve(),
            reference_time_utc=REFERENCE_TIME,
        )


def test_validator_has_no_connector_execution_or_secret_surface() -> None:
    source = inspect.getsource(module)
    assert "databricks.sql" not in source
    assert "requests." not in source
    assert "Invoke-RestMethod" not in source
    assert "DATABRICKS_TOKEN" not in source
    assert "query_to_df" not in source
    assert "execute(" not in source


def test_admission_is_absent_from_lt_production() -> None:
    source = inspect.getsource(production_phases)
    assert "entsoe_metadata_admission" not in source
    assert "verify_entsoe_metadata_capture_paths" not in source
