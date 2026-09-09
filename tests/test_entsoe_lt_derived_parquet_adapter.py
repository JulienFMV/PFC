from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pfc_shaping.validation import entsoe_lt_derived_arithmetic_execution as d283
from pfc_shaping.validation import entsoe_lt_derived_parquet_adapter as module
from pfc_shaping.validation import entsoe_lt_derived_physical_compatibility as d282
from pfc_shaping.validation.entsoe_lt_derived_parquet_adapter import (
    EntsoeLtDerivedParquetAdapterError,
    execute_synthetic_parquet_package,
    validate_derived_parquet_adapter_contract,
    validate_derived_parquet_adapter_contract_path,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = module.CONTRACT_PATH
REFERENCE = datetime(2026, 8, 6, 3, 0, tzinfo=timezone.utc)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _profile(position: int) -> dict[str, object]:
    return {
        "transform_id_sha256": _hash(f"transform-{position}"),
        "output_feature_record_id": _hash(f"output-{position}"),
        "input_feature_record_ids_in_formula_order": [
            _hash(f"actual-record-{position}"),
            _hash(f"forecast-record-{position}"),
        ],
        "actual_series_id_sha256": _hash(f"actual-series-{position}"),
        "forecast_series_id_sha256": _hash(f"forecast-series-{position}"),
        "logical_zone_sha256": _hash("CH"),
        "canonical_unit": "MW",
        "effective_native_resolution": "PT1H",
        "target_duration_seconds": 3600,
        "temporal_zone_binding_complete": True,
        "derived_output_inherits_physical_profile": True,
        "raw_ids_persisted": False,
        "raw_or_feature_values_opened": False,
    }


def _d282_assessment(count: int = 2) -> dict[str, object]:
    return {
        "schema_version": d282.ASSESSMENT_SCHEMA,
        "status": d282.PASS_STATUS,
        "contract_content_id": d282.CONTRACT_CONTENT_ID,
        "reference_time_utc": "2026-08-06T02:00:00Z",
        "bindings": {
            name: _hash(name)
            for name in (
                "d280_assessment_content_id",
                "d281_assessment_content_id",
                "feature_records_content_id",
                "feature_selection_content_id",
                "family_mapping_content_id",
                "semantic_binding_content_id",
                "zone_registry_content_id",
                "zone_binding_content_id",
                "lineage_content_id",
            )
        },
        "transform_count": count,
        "all_primitives_selected_once": True,
        "all_temporal_zone_bindings_complete": True,
        "all_logical_zones_match": True,
        "all_canonical_units_are_mw": True,
        "all_effective_native_cadences_match": True,
        "d280_lineage_blocker_structurally_discharged": True,
        "profiles": [_profile(position) for position in range(count)],
        "clear_series_or_feature_ids_persisted": False,
        "real_or_feature_values_opened": False,
        "arithmetic_recomputed": False,
        "derived_model_input_authorized": False,
        "authorities": dict(d282.AUTHORITIES),
        **module.EXECUTION,
    }


def _rows(assessment: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for position, profile in enumerate(assessment["profiles"]):
        start = datetime(2026, 8, 1, position, tzinfo=timezone.utc)
        end = datetime(2026, 8, 1, position + 1, tzinfo=timezone.utc)
        for role, record_offset, series_field, value in (
            (d283.ACTUAL_ROLE, 0, "actual_series_id_sha256", "1250.25"),
            (d283.FORECAST_ROLE, 1, "forecast_series_id_sha256", "1248"),
        ):
            rows.append(
                {
                    "feature_record_id": profile["input_feature_record_ids_in_formula_order"][
                        record_offset
                    ],
                    "role": role,
                    "target_start_utc": start,
                    "target_end_utc": end,
                    "series_id_sha256": profile[series_field],
                    "logical_zone_sha256": profile["logical_zone_sha256"],
                    "canonical_unit": profile["canonical_unit"],
                    "effective_native_resolution": profile["effective_native_resolution"],
                    "value_decimal_mw": None if position == 1 else value,
                }
            )
    return rows


def _write_package(
    directory: Path,
    assessment: dict[str, object],
    *,
    rows: list[dict[str, object]] | None = None,
    schema: pa.Schema | None = None,
    compression: str = "zstd",
    row_group_size: int | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    parquet_path = directory / module.ARTIFACT_RELATIVE_PATH
    table = pa.Table.from_pylist(
        rows or _rows(assessment), schema=schema or module.EXPECTED_ARROW_SCHEMA
    )
    pq.write_table(
        table,
        parquet_path,
        compression=compression,
        row_group_size=row_group_size,
    )
    payload = parquet_path.read_bytes()
    metadata = pq.ParquetFile(parquet_path).metadata
    manifest = {
        "schema_version": module.MANIFEST_SCHEMA,
        "evidence_class": module.EVIDENCE_CLASS,
        "created_at_utc": "2026-08-06T02:30:00Z",
        "d282_assessment_content_id": module.canonical_sha256(assessment),
        "artifact": {
            "relative_path": module.ARTIFACT_RELATIVE_PATH,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "row_count": metadata.num_rows,
            "row_group_count": metadata.num_row_groups,
            "column_count": metadata.num_columns,
            "arrow_schema_content_id": module.ARROW_SCHEMA_CONTENT_ID,
            "compression": module.COMPRESSION,
        },
        "authorities": dict(module.MANIFEST_AUTHORITIES),
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rewrite_manifest(path: Path, document: dict[str, object]) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _execute(path: Path, assessment: dict[str, object]):
    return execute_synthetic_parquet_package(
        contract_path=CONTRACT_PATH,
        manifest_path=path,
        d282_assessment=assessment,
        reference_time_utc=REFERENCE,
    )


@pytest.fixture
def package(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    assessment = _d282_assessment()
    return _write_package(tmp_path, assessment), assessment


def test_contract_is_exact_bounded_zero_query_and_non_authoritative() -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    direct = validate_derived_parquet_adapter_contract(document)
    stable = validate_derived_parquet_adapter_contract_path(CONTRACT_PATH)

    assert direct == stable
    assert direct["contract_content_id"] == module.CONTRACT_CONTENT_ID
    assert direct["arrow_schema_content_id"] == module.ARROW_SCHEMA_CONTENT_ID
    assert direct["real_data_authorized"] is False
    assert direct["model_input_authorized"] is False
    assert all(value == 0 for value in module.EXECUTION.values())
    assert all(value is False for value in module.AUTHORITIES.values())


def test_valid_parquet_executes_d283_exact_decimal_and_null_policy(package) -> None:
    manifest_path, assessment = package

    result = _execute(manifest_path, assessment)

    assert result.assessment["status"] == module.PASS_STATUS
    assert result.assessment["row_count"] == 4
    assert result.assessment["actual_row_count"] == 2
    assert result.assessment["forecast_row_count"] == 2
    assert result.arithmetic_result.assessment["computed_value_count"] == 1
    assert result.arithmetic_result.assessment["null_propagated_count"] == 1
    values = [row["value_decimal_mw"] for row in result.arithmetic_result.output_artifact["rows"]]
    assert values == [None, "2.25"] or values == ["2.25", None]


def test_replay_is_content_deterministic(package) -> None:
    manifest_path, assessment = package

    first = _execute(manifest_path, assessment)
    second = _execute(manifest_path, assessment)

    assert first.assessment_content_id == second.assessment_content_id
    assert (
        first.arithmetic_result.output_artifact_content_id
        == second.arithmetic_result.output_artifact_content_id
    )


def test_assessment_contains_hashes_and_counts_but_no_values(package) -> None:
    manifest_path, assessment = package

    result = _execute(manifest_path, assessment)
    encoded = json.dumps(result.assessment, sort_keys=True)

    assert "1250.25" not in encoded
    assert "1248" not in encoded
    assert "2.25" not in encoded
    assert result.assessment["operand_or_output_values_persisted_in_assessment"] is False
    assert result.assessment["real_data_authorized"] is False
    assert result.assessment["model_input_authorized"] is False


def test_binary_float_value_column_is_rejected_before_coercion(tmp_path: Path) -> None:
    assessment = _d282_assessment()
    rows = _rows(assessment)
    for row in rows:
        row["value_decimal_mw"] = 1250.25
    fields = list(module.EXPECTED_ARROW_SCHEMA)
    fields[-1] = pa.field("value_decimal_mw", pa.float64(), nullable=True)
    manifest_path = _write_package(tmp_path, assessment, rows=rows, schema=pa.schema(fields))

    with pytest.raises(EntsoeLtDerivedParquetAdapterError, match="footer admission"):
        _execute(manifest_path, assessment)


@pytest.mark.parametrize(
    "schema",
    [
        pa.schema(list(module.EXPECTED_ARROW_SCHEMA)[::-1]),
        pa.schema(
            [
                *list(module.EXPECTED_ARROW_SCHEMA)[:2],
                pa.field("target_start_utc", pa.timestamp("ms", tz="UTC"), nullable=False),
                *list(module.EXPECTED_ARROW_SCHEMA)[3:],
            ]
        ),
        pa.schema(
            [
                pa.field(field.name, field.type, nullable=True)
                if field.name == "feature_record_id"
                else field
                for field in module.EXPECTED_ARROW_SCHEMA
            ]
        ),
    ],
    ids=["column-order", "timestamp-unit", "nullability"],
)
def test_exact_arrow_schema_rejects_order_precision_and_nullability(
    tmp_path: Path,
    schema: pa.Schema,
) -> None:
    assessment = _d282_assessment()
    manifest_path = _write_package(tmp_path, assessment, schema=schema)

    with pytest.raises(EntsoeLtDerivedParquetAdapterError, match="Arrow schema differs"):
        _execute(manifest_path, assessment)


@pytest.mark.parametrize("field", ["sha256", "size_bytes", "row_count", "row_group_count"])
def test_manifest_artifact_declarations_are_verified(package, field: str) -> None:
    manifest_path, assessment = package
    document = _manifest(manifest_path)
    artifact = document["artifact"]
    artifact[field] = _hash("wrong") if field == "sha256" else artifact[field] + 1
    _rewrite_manifest(manifest_path, document)

    with pytest.raises(EntsoeLtDerivedParquetAdapterError):
        _execute(manifest_path, assessment)


def test_fixed_sibling_path_rejects_traversal(package) -> None:
    manifest_path, assessment = package
    document = _manifest(manifest_path)
    document["artifact"]["relative_path"] = "../derived_operand_values.parquet"
    _rewrite_manifest(manifest_path, document)

    with pytest.raises(EntsoeLtDerivedParquetAdapterError, match="relative path"):
        _execute(manifest_path, assessment)


def test_duplicate_row_grain_is_rejected(tmp_path: Path) -> None:
    assessment = _d282_assessment()
    rows = _rows(assessment)
    rows.append(copy.deepcopy(rows[0]))
    manifest_path = _write_package(tmp_path, assessment, rows=rows)

    with pytest.raises(EntsoeLtDerivedParquetAdapterError, match="grain is duplicated"):
        _execute(manifest_path, assessment)


def test_unknown_role_is_rejected(tmp_path: Path) -> None:
    assessment = _d282_assessment()
    rows = _rows(assessment)
    rows[0]["role"] = "INFER_FROM_NAME"
    manifest_path = _write_package(tmp_path, assessment, rows=rows)

    with pytest.raises(EntsoeLtDerivedParquetAdapterError, match="role differs"):
        _execute(manifest_path, assessment)


@pytest.mark.parametrize("mutation", ["missing", "orphan", "noncanonical-decimal"])
def test_d283_rejects_semantic_and_decimal_mutations(tmp_path: Path, mutation: str) -> None:
    assessment = _d282_assessment()
    rows = _rows(assessment)
    if mutation == "missing":
        rows.pop()
    elif mutation == "orphan":
        rows[0]["feature_record_id"] = _hash("orphan")
    else:
        rows[0]["value_decimal_mw"] = "01250.25"
    manifest_path = _write_package(tmp_path, assessment, rows=rows)

    with pytest.raises(EntsoeLtDerivedParquetAdapterError, match="D283 execution rejected"):
        _execute(manifest_path, assessment)


def test_manifest_is_bound_to_exact_d282_assessment(package) -> None:
    manifest_path, assessment = package
    detached = copy.deepcopy(assessment)
    detached["profiles"][0]["canonical_unit"] = "kW"

    with pytest.raises(EntsoeLtDerivedParquetAdapterError, match="D282 assessment content ID"):
        _execute(manifest_path, detached)


def test_future_manifest_is_rejected(package) -> None:
    manifest_path, assessment = package
    document = _manifest(manifest_path)
    document["created_at_utc"] = "2026-08-06T03:00:01Z"
    _rewrite_manifest(manifest_path, document)

    with pytest.raises(EntsoeLtDerivedParquetAdapterError, match="future-dated"):
        _execute(manifest_path, assessment)


def test_hardlinked_manifest_is_rejected(tmp_path: Path) -> None:
    assessment = _d282_assessment()
    manifest_path = _write_package(tmp_path, assessment)
    os.link(manifest_path, tmp_path / "manifest-alias.json")

    with pytest.raises(EntsoeLtDerivedParquetAdapterError, match="manifest read failed"):
        _execute(manifest_path, assessment)


def test_mid_execution_manifest_mutation_is_rejected(package, monkeypatch) -> None:
    manifest_path, assessment = package
    original_read = module._read
    manifest_reads = 0

    def unstable_read(path: Path, label: str, max_bytes: int) -> bytes:
        nonlocal manifest_reads
        payload = original_read(path, label, max_bytes)
        if Path(path) == manifest_path:
            manifest_reads += 1
            if manifest_reads == 2:
                return payload + b" "
        return payload

    monkeypatch.setattr(module, "_read", unstable_read)

    with pytest.raises(EntsoeLtDerivedParquetAdapterError, match="manifest changed"):
        _execute(manifest_path, assessment)


def test_duplicate_manifest_json_key_is_rejected(package) -> None:
    manifest_path, assessment = package
    payload = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        payload.replace("{", '{"schema_version":"duplicate",', 1),
        encoding="utf-8",
    )

    with pytest.raises(EntsoeLtDerivedParquetAdapterError, match="duplicate keys"):
        _execute(manifest_path, assessment)


def test_contract_hash_rejects_mutated_copy(tmp_path: Path) -> None:
    mutated = tmp_path / "contract.json"
    mutated.write_bytes(CONTRACT_PATH.read_bytes() + b" ")

    with pytest.raises(EntsoeLtDerivedParquetAdapterError, match="SHA-256 differs"):
        validate_derived_parquet_adapter_contract_path(mutated)


def test_module_has_no_ct_databricks_or_h_drive_dependency() -> None:
    source = (ROOT / "pfc_shaping/validation/entsoe_lt_derived_parquet_adapter.py").read_text(
        encoding="utf-8"
    )

    assert "pfc_shaping.ct" not in source
    assert "databricks" not in source.lower()
    assert "H:\\" not in source
