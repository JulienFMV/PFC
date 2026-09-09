from __future__ import annotations

import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pfc_shaping.pipeline import production_phases
from pfc_shaping.validation import entsoe_physical_mapping_compiler as module
from pfc_shaping.validation.entsoe_metadata_admission import (
    admit_entsoe_metadata_capture,
)
from pfc_shaping.validation.entsoe_physical_mapping_compiler import (
    CONTRACT_CONTENT_ID,
    CONTRACT_STATUS,
    EntsoePhysicalMappingError,
    compile_entsoe_physical_mapping,
    validate_physical_mapping_compiler_contract,
    verify_entsoe_physical_mapping_paths,
)

ROOT = Path(__file__).resolve().parents[1]
PLANNING = ROOT / ".planning" / "phases" / "14-lt-audit-remediation"
CONTRACT_PATH = PLANNING / "ENTSOE-PHYSICAL-MAPPING-COMPILER-CONTRACT-V1.json"
GOVERNED_PACKAGE_PATH = (
    PLANNING / "EEX-ENTSOE-GOVERNED-ACQUISITION-PACKAGE-CONTRACT-V1.json"
)
INTAKE_PATH = PLANNING / "ENTSOE-DATABRICKS-INTAKE-CONTRACT-V1.json"
CAPTURED_AT = "2026-08-05T12:00:00Z"
MAPPED_AT = "2026-08-05T12:01:00Z"
REFERENCE_TIME = datetime(2026, 8, 5, 12, 5, tzinfo=timezone.utc)


def _metadata_payload() -> bytes:
    header = (
        "table_catalog,table_schema,table_name,column_name,ordinal_position,"
        "data_type,is_nullable"
    )
    rows = [
        "dev,gold,dimentsoeseries,series_key,1,BIGINT,NO",
        "dev,gold,dimentsoeseries,group_code,2,STRING,NO",
        "dev,gold,dimentsoeseries,field_code,3,STRING,NO",
        "dev,gold,dimentsoeseries,zone_from,4,STRING,YES",
        "dev,gold,dimentsoeseries,zone_to,5,STRING,YES",
        "dev,gold,dimentsoeseries,measure_unit,6,STRING,NO",
        "dev,gold,dimentsoeseries,resolution_iso,7,STRING,NO",
        "dev,gold,dimentsoeseries,endpoint_url,8,STRING,NO",
        "dev,gold,dimentsoeseries,doc_key,9,STRING,NO",
        "dev,gold,dimentsoeseries,unused_dimension_column,10,STRING,YES",
        "dev,gold,factentsoetimeserieslatest,series_key,1,BIGINT,NO",
        "dev,gold,factentsoetimeserieslatest,delivery_utc,2,TIMESTAMP,NO",
        "dev,gold,factentsoetimeserieslatest,measure_value,3,DOUBLE,YES",
        "dev,gold,factentsoetimeserieslatest,historical_flag,4,BOOLEAN,NO",
        "dev,gold,factentsoetimeserieslatest,quality_code,5,STRING,YES",
        "dev,gold,factentsoetimeserieslatest,revision_no,6,INT,NO",
        "dev,gold,factentsoetimeserieslatest,loaded_utc,7,TIMESTAMP,NO",
        "dev,gold,factentsoetimeseriesvintages,series_key,1,BIGINT,NO",
        "dev,gold,factentsoetimeseriesvintages,delivery_utc,2,TIMESTAMP,NO",
        "dev,gold,factentsoetimeseriesvintages,available_utc,3,TIMESTAMP,NO",
        "dev,gold,factentsoetimeseriesvintages,measure_value,4,DOUBLE,YES",
        "dev,gold,factentsoetimeseriesvintages,quality_code,5,STRING,YES",
        "dev,gold,factentsoetimeseriesvintages,revision_no,6,INT,NO",
        "dev,gold,factentsoetimeseriesvintages,loaded_utc,7,TIMESTAMP,NO",
    ]
    return ("\n".join([header, *rows]) + "\n").encode("utf-8")


def _receipt(payload: bytes) -> dict[str, object]:
    return {
        "schema_version": "fmv_entsoe_databricks_metadata_capture_receipt.v1",
        "query_sha256": (
            "dec7e207603e3a8b69f5808b42454575f1b4a985a25fa8aca3e5c6a2c95b72fa"
        ),
        "statement_id": "01234567-89ab-4def-8123-456789abcdef",
        "statement_state": "SUCCEEDED",
        "captured_at_utc": CAPTURED_AT,
        "row_count": len(payload.decode("utf-8").splitlines()) - 1,
        "result_truncated": False,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "read_only_metadata_only": True,
        "remote_write_count": 0,
    }


def _mapping(payload: bytes) -> dict[str, object]:
    receipt = _receipt(payload)
    admission = admit_entsoe_metadata_capture(
        receipt,
        payload,
        reference_time_utc=REFERENCE_TIME,
    )
    return {
        "schema_version": "fmv_entsoe_physical_column_mapping.v1",
        "evidence_class": "SYNTHETIC_TEST_ONLY",
        "mapping_owner": "SYNTHETIC_TEST_FIXTURE",
        "mapped_at_utc": MAPPED_AT,
        "metadata_schema_fingerprint": admission["schema_fingerprint"],
        "metadata_payload_sha256": admission["payload_sha256"],
        "metadata_statement_id": admission["statement_id"],
        "catalog": "dev",
        "schema": "gold",
        "table_mappings": {
            "series_dimension": {
                "physical_table": "dimentsoeseries",
                "columns": {
                    "series_id": "series_key",
                    "group_name": "group_code",
                    "field_name": "field_code",
                    "from_zone": "zone_from",
                    "to_zone": "zone_to",
                    "unit": "measure_unit",
                    "native_resolution": "resolution_iso",
                    "source_endpoint": "endpoint_url",
                    "document_id": "doc_key",
                },
            },
            "latest_values": {
                "physical_table": "factentsoetimeserieslatest",
                "columns": {
                    "series_id": "series_key",
                    "target_ts_utc": "delivery_utc",
                    "value": "measure_value",
                    "is_historical": "historical_flag",
                    "quality_flag": "quality_code",
                    "revision_number": "revision_no",
                    "meta_load_timestamp_utc": "loaded_utc",
                },
            },
            "vintage_values": {
                "physical_table": "factentsoetimeseriesvintages",
                "columns": {
                    "series_id": "series_key",
                    "target_ts_utc": "delivery_utc",
                    "as_of_utc": "available_utc",
                    "value": "measure_value",
                    "quality_flag": "quality_code",
                    "revision_number": "revision_no",
                    "meta_load_timestamp_utc": "loaded_utc",
                },
            },
        },
        "semantic_assertions": dict(module.SEMANTIC_ASSERTIONS),
        "authorities": dict(module.MAPPING_AUTHORITIES),
    }


def _compile(
    payload: bytes | None = None,
    *,
    mapping_mutator: object | None = None,
) -> dict[str, object]:
    content = _metadata_payload() if payload is None else payload
    mapping = _mapping(content)
    if mapping_mutator is not None:
        assert callable(mapping_mutator)
        mapping_mutator(mapping)
    return compile_entsoe_physical_mapping(
        _receipt(content),
        content,
        mapping,
        reference_time_utc=REFERENCE_TIME,
    )


def test_static_compiler_contract_is_exact_and_zero_budget() -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    result = validate_physical_mapping_compiler_contract(document)

    assert result == {
        "schema_version": "fmv_entsoe_physical_mapping_compiler_contract.v1",
        "status": CONTRACT_STATUS,
        "contract_content_id": CONTRACT_CONTENT_ID,
        "current_statement_budget": 0,
        "compiled_sql_execution_authorized": False,
        "production_authorized": False,
    }


def test_valid_synthetic_mapping_compiles_three_non_executable_templates() -> None:
    result = _compile()

    assert result["status"] == (
        "PASS_SYNTHETIC_MAPPING_FIXTURE_EXECUTION_NOT_AUTHORIZED"
    )
    assert result["mapping_evidence_class"] == "SYNTHETIC_TEST_ONLY"
    assert result["physical_mapping_matches_admitted_metadata"] is True
    assert result["real_metadata_evidence_asserted"] is False
    assert result["external_mapping_owner_identity_verified"] is False
    assert result["compiled_sql_execution_authorized"] is False
    assert result["entsoe_values_opened"] is False
    assert result["point_in_time_availability_proven"] is False
    assert result["model_input_authorized"] is False
    assert result["production_authorized"] is False
    assert result["validator_databricks_request_count"] == 0
    assert result["validator_warehouse_start_count"] == 0
    assert result["validator_network_call_count"] == 0
    assert result["validator_remote_write_count"] == 0
    assert set(result["sql_templates"]) == set(module.ROLE_SPECS)
    assert set(result["sql_template_sha256"]) == set(module.ROLE_SPECS)


def test_templates_are_explicit_parameterized_ordered_and_bounded() -> None:
    result = _compile()
    dimension = result["sql_templates"]["series_dimension"]
    latest = result["sql_templates"]["latest_values"]
    vintage = result["sql_templates"]["vintage_values"]

    for sql in (dimension, latest, vintage):
        assert sql.startswith("SELECT\n")
        assert "SELECT *" not in sql.upper()
        assert ";" not in sql
        assert sql.endswith("\n")
    assert "FROM `dev`.`gold`.`dimentsoeseries`" in dimension
    assert "`series_key` AS `series_id`" in dimension
    assert "ORDER BY `series_key`\nLIMIT 100001\n" in dimension
    assert ":start_utc" not in dimension
    assert "`delivery_utc` >= :start_utc" in latest
    assert "`delivery_utc` < :end_utc" in latest
    assert ":as_of_cutoff_utc" not in latest
    assert "LIMIT 1000001\n" in latest
    assert "`available_utc` <= :as_of_cutoff_utc" in vintage
    assert result["sql_parameters"] == {
        "series_dimension": [],
        "latest_values": ["start_utc", "end_utc"],
        "vintage_values": ["start_utc", "end_utc", "as_of_cutoff_utc"],
    }


def test_unused_physical_column_does_not_enter_projection() -> None:
    result = _compile()
    assert "unused_dimension_column" not in result["sql_templates"]["series_dimension"]


@pytest.mark.parametrize("physical", ["update", "selected_code"])
def test_delimited_safe_physical_names_do_not_trigger_structural_tokens(
    physical: str,
) -> None:
    payload = (
        _metadata_payload()
        .decode("utf-8")
        .replace(
            "dimentsoeseries,group_code,2,STRING",
            f"dimentsoeseries,{physical},2,STRING",
        )
        .encode("utf-8")
    )

    def mutate(mapping: dict[str, object]) -> None:
        mapping["table_mappings"]["series_dimension"]["columns"][
            "group_name"
        ] = physical

    result = _compile(payload, mapping_mutator=mutate)
    assert f"`{physical}` AS `group_name`" in result["sql_templates"][
        "series_dimension"
    ]


def test_path_verifier_binds_all_inputs(tmp_path: Path) -> None:
    payload = _metadata_payload()
    receipt_path = tmp_path / "receipt.json"
    payload_path = tmp_path / "metadata.csv"
    mapping_path = tmp_path / "mapping.json"
    receipt_path.write_text(json.dumps(_receipt(payload)), encoding="utf-8")
    payload_path.write_bytes(payload)
    mapping_path.write_text(json.dumps(_mapping(payload)), encoding="utf-8")

    result = verify_entsoe_physical_mapping_paths(
        compiler_contract_path=CONTRACT_PATH.resolve(),
        governed_package_contract_path=GOVERNED_PACKAGE_PATH.resolve(),
        intake_contract_path=INTAKE_PATH.resolve(),
        receipt_path=receipt_path.resolve(),
        metadata_payload_path=payload_path.resolve(),
        mapping_path=mapping_path.resolve(),
        reference_time_utc=REFERENCE_TIME,
    )

    assert result["compiler_contract_content_id"] == CONTRACT_CONTENT_ID
    assert result["compiled_sql_execution_authorized"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("extra_top", "fields must match"),
        ("bad_schema", "mapping schema"),
        ("bad_evidence", "evidence_class"),
        ("bad_owner", "owner format"),
        ("wrong_synthetic_owner", "synthetic mapping owner"),
        ("real_synthetic_owner", "real mapping owner is synthetic"),
        ("mapped_before_capture", "between capture and reference"),
        ("mapped_after_reference", "between capture and reference"),
        ("bad_fingerprint", "schema fingerprint"),
        ("bad_payload_hash", "payload SHA-256"),
        ("bad_statement_id", "statement ID"),
        ("bad_catalog", "mapping catalog"),
        ("bad_schema_name", "mapping schema name"),
        ("semantic_timezone", "semantic assertions"),
        ("semantic_cast", "semantic assertions"),
        ("semantic_latest_pit", "semantic assertions"),
        ("authority", "mapping authorities"),
    ],
)
def test_mapping_manifest_mutations_fail_closed(
    mutation: str,
    message: str,
) -> None:
    def mutate(mapping: dict[str, object]) -> None:
        if mutation == "extra_top":
            mapping["shortcut"] = True
        elif mutation == "bad_schema":
            mapping["schema_version"] = "v2"
        elif mutation == "bad_evidence":
            mapping["evidence_class"] = "UNVERIFIED"
        elif mutation == "bad_owner":
            mapping["mapping_owner"] = "x; DROP"
        elif mutation == "wrong_synthetic_owner":
            mapping["mapping_owner"] = "DATA_ENGINEER"
        elif mutation == "real_synthetic_owner":
            mapping["evidence_class"] = "REAL_METADATA_OWNER_ASSERTED"
        elif mutation == "mapped_before_capture":
            mapping["mapped_at_utc"] = "2026-08-05T11:59:59Z"
        elif mutation == "mapped_after_reference":
            mapping["mapped_at_utc"] = "2026-08-05T12:05:01Z"
        elif mutation == "bad_fingerprint":
            mapping["metadata_schema_fingerprint"] = "0" * 64
        elif mutation == "bad_payload_hash":
            mapping["metadata_payload_sha256"] = "0" * 64
        elif mutation == "bad_statement_id":
            mapping["metadata_statement_id"] = "11111111-1111-4111-8111-111111111111"
        elif mutation == "bad_catalog":
            mapping["catalog"] = "prd"
        elif mutation == "bad_schema_name":
            mapping["schema"] = "silver"
        elif mutation == "semantic_timezone":
            mapping["semantic_assertions"]["timestamp_timezone"] = "Europe/Zurich"
        elif mutation == "semantic_cast":
            mapping["semantic_assertions"]["implicit_casts_authorized"] = True
        elif mutation == "semantic_latest_pit":
            mapping["semantic_assertions"][
                "latest_values_point_in_time_authority"
            ] = True
        elif mutation == "authority":
            mapping["authorities"]["execution_authorized"] = True
        else:
            raise AssertionError(mutation)

    with pytest.raises(EntsoePhysicalMappingError, match=message):
        _compile(mapping_mutator=mutate)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_role", "table mapping roles"),
        ("extra_role_field", "mapping role fields differ"),
        ("wrong_table", "physical table"),
        ("missing_normalized", "normalized fields"),
        ("extra_normalized", "normalized fields"),
        ("duplicate_physical", "not injective"),
        ("absent_physical", "absent from admitted metadata"),
        ("unsafe_physical", "unsafe physical identifier"),
    ],
)
def test_table_mapping_mutations_fail_closed(
    mutation: str,
    message: str,
) -> None:
    def mutate(mapping: dict[str, object]) -> None:
        mappings = mapping["table_mappings"]
        if mutation == "missing_role":
            mappings.pop("latest_values")
        elif mutation == "extra_role_field":
            mappings["latest_values"]["shortcut"] = True
        elif mutation == "wrong_table":
            mappings["latest_values"]["physical_table"] = "dimentsoeseries"
        elif mutation == "missing_normalized":
            mappings["latest_values"]["columns"].pop("value")
        elif mutation == "extra_normalized":
            mappings["latest_values"]["columns"]["invented"] = "measure_value"
        elif mutation == "duplicate_physical":
            mappings["latest_values"]["columns"]["quality_flag"] = "measure_value"
        elif mutation == "absent_physical":
            mappings["latest_values"]["columns"]["quality_flag"] = "missing_column"
        elif mutation == "unsafe_physical":
            mappings["latest_values"]["columns"]["quality_flag"] = "quality;drop"
        else:
            raise AssertionError(mutation)

    with pytest.raises(EntsoePhysicalMappingError, match=message):
        _compile(mapping_mutator=mutate)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "dimentsoeseries,series_key,1,BIGINT",
            "dimentsoeseries,series_key,1,DOUBLE",
            "series_id=DOUBLE",
        ),
        (
            "factentsoetimeserieslatest,delivery_utc,2,TIMESTAMP",
            "factentsoetimeserieslatest,delivery_utc,2,STRING",
            "target_ts_utc=STRING",
        ),
        (
            "factentsoetimeserieslatest,measure_value,3,DOUBLE",
            "factentsoetimeserieslatest,measure_value,3,STRING",
            "value=STRING",
        ),
        (
            "factentsoetimeserieslatest,historical_flag,4,BOOLEAN",
            "factentsoetimeserieslatest,historical_flag,4,INT",
            "is_historical=INT",
        ),
        (
            "factentsoetimeserieslatest,revision_no,6,INT",
            "factentsoetimeserieslatest,revision_no,6,DOUBLE",
            "revision_number=DOUBLE",
        ),
    ],
)
def test_incompatible_physical_types_fail_closed(
    old: str,
    new: str,
    message: str,
) -> None:
    payload = _metadata_payload().decode("utf-8").replace(old, new, 1).encode("utf-8")

    with pytest.raises(EntsoePhysicalMappingError, match=message):
        _compile(payload)


@pytest.mark.parametrize(
    ("old", "new", "field"),
    [
        (
            "factentsoetimeserieslatest,series_key,1,BIGINT",
            "factentsoetimeserieslatest,series_key,1,INT",
            "series_id",
        ),
        (
            "factentsoetimeseriesvintages,measure_value,4,DOUBLE",
            "factentsoetimeseriesvintages,measure_value,4,DECIMAL",
            "value",
        ),
    ],
)
def test_individually_compatible_but_cross_table_type_drift_fails(
    old: str,
    new: str,
    field: str,
) -> None:
    payload = _metadata_payload().decode("utf-8").replace(old, new, 1).encode("utf-8")

    with pytest.raises(EntsoePhysicalMappingError, match=f"types differ for {field}"):
        _compile(payload)


def test_real_evidence_class_remains_owner_asserted_and_non_executable() -> None:
    def mutate(mapping: dict[str, object]) -> None:
        mapping["evidence_class"] = "REAL_METADATA_OWNER_ASSERTED"
        mapping["mapping_owner"] = "data.engineer@example.org"

    result = _compile(mapping_mutator=mutate)
    assert result["status"] == (
        "PASS_REAL_METADATA_OWNER_ASSERTED_MAPPING_EXECUTION_NOT_AUTHORIZED"
    )
    assert result["real_metadata_evidence_asserted"] is True
    assert result["external_mapping_owner_identity_verified"] is False
    assert result["compiled_sql_execution_authorized"] is False


def test_contract_or_intake_path_drift_fails_closed(tmp_path: Path) -> None:
    payload = _metadata_payload()
    changed_contract = tmp_path / "contract.json"
    changed_contract.write_bytes(CONTRACT_PATH.read_bytes() + b" ")
    receipt_path = tmp_path / "receipt.json"
    payload_path = tmp_path / "metadata.csv"
    mapping_path = tmp_path / "mapping.json"
    receipt_path.write_text(json.dumps(_receipt(payload)), encoding="utf-8")
    payload_path.write_bytes(payload)
    mapping_path.write_text(json.dumps(_mapping(payload)), encoding="utf-8")

    with pytest.raises(EntsoePhysicalMappingError, match="contract SHA-256 differs"):
        verify_entsoe_physical_mapping_paths(
            compiler_contract_path=changed_contract.resolve(),
            governed_package_contract_path=GOVERNED_PACKAGE_PATH.resolve(),
            intake_contract_path=INTAKE_PATH.resolve(),
            receipt_path=receipt_path.resolve(),
            metadata_payload_path=payload_path.resolve(),
            mapping_path=mapping_path.resolve(),
            reference_time_utc=REFERENCE_TIME,
        )


def test_d233_governed_package_path_drift_fails_closed(tmp_path: Path) -> None:
    payload = _metadata_payload()
    changed_package = tmp_path / "governed-package.json"
    changed_package.write_bytes(GOVERNED_PACKAGE_PATH.read_bytes() + b" ")
    receipt_path = tmp_path / "receipt.json"
    payload_path = tmp_path / "metadata.csv"
    mapping_path = tmp_path / "mapping.json"
    receipt_path.write_text(json.dumps(_receipt(payload)), encoding="utf-8")
    payload_path.write_bytes(payload)
    mapping_path.write_text(json.dumps(_mapping(payload)), encoding="utf-8")

    with pytest.raises(
        EntsoePhysicalMappingError,
        match="D233 governed acquisition package contract SHA-256 differs",
    ):
        verify_entsoe_physical_mapping_paths(
            compiler_contract_path=CONTRACT_PATH.resolve(),
            governed_package_contract_path=changed_package.resolve(),
            intake_contract_path=INTAKE_PATH.resolve(),
            receipt_path=receipt_path.resolve(),
            metadata_payload_path=payload_path.resolve(),
            mapping_path=mapping_path.resolve(),
            reference_time_utc=REFERENCE_TIME,
        )


def test_contract_content_mutation_fails_closed() -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    document["execution_policy"]["compiled_sql_execution_authorized"] = True
    with pytest.raises(EntsoePhysicalMappingError):
        validate_physical_mapping_compiler_contract(document)


def test_duplicate_mapping_json_key_fails_closed(tmp_path: Path) -> None:
    payload = _metadata_payload()
    receipt_path = tmp_path / "receipt.json"
    payload_path = tmp_path / "metadata.csv"
    mapping_path = tmp_path / "mapping.json"
    receipt_path.write_text(json.dumps(_receipt(payload)), encoding="utf-8")
    payload_path.write_bytes(payload)
    mapping_path.write_text(
        '{"schema_version":"v1","schema_version":"v2"}',
        encoding="utf-8",
    )

    with pytest.raises(EntsoePhysicalMappingError, match="duplicate key"):
        verify_entsoe_physical_mapping_paths(
            compiler_contract_path=CONTRACT_PATH.resolve(),
            governed_package_contract_path=GOVERNED_PACKAGE_PATH.resolve(),
            intake_contract_path=INTAKE_PATH.resolve(),
            receipt_path=receipt_path.resolve(),
            metadata_payload_path=payload_path.resolve(),
            mapping_path=mapping_path.resolve(),
            reference_time_utc=REFERENCE_TIME,
        )


def test_all_paths_must_be_absolute() -> None:
    with pytest.raises(EntsoePhysicalMappingError, match="path must be absolute"):
        verify_entsoe_physical_mapping_paths(
            compiler_contract_path=Path("contract.json"),
            governed_package_contract_path=Path("governed-package.json"),
            intake_contract_path=Path("intake.json"),
            receipt_path=Path("receipt.json"),
            metadata_payload_path=Path("metadata.csv"),
            mapping_path=Path("mapping.json"),
            reference_time_utc=REFERENCE_TIME,
        )


def test_compiler_has_no_connector_execution_or_secret_surface() -> None:
    source = inspect.getsource(module)
    assert "databricks.sql" not in source
    assert "requests." not in source
    assert "Invoke-RestMethod" not in source
    assert "DATABRICKS_TOKEN" not in source
    assert "query_to_df" not in source
    assert "execute(" not in source


def test_compiler_is_absent_from_lt_production() -> None:
    source = inspect.getsource(production_phases)
    assert "entsoe_physical_mapping_compiler" not in source
    assert "verify_entsoe_physical_mapping_paths" not in source
