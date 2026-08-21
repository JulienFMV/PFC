from __future__ import annotations

import hashlib
import json
from io import BytesIO

import pandas as pd
import pytest

from pfc_shaping.data.acquisition_contract import DATABRICKS_UPSTREAM_REPLAY_KIND
from pfc_shaping.data.databricks_lt_replay import (
    GOLD_SPOT_MODE,
    build_databricks_replay_package,
)
from pfc_shaping.data.databricks_lt_snapshot import (
    DATABRICKS_EXPORT_MANIFEST_SCHEMA,
    DATABRICKS_EXPORT_MODE,
    DatabricksLTSnapshotError,
    canonical_json_bytes,
    databricks_export_id,
    expected_databricks_quality_bindings,
    verify_databricks_snapshot_replay,
)


def _parquet_payload(frame: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    frame.to_parquet(buffer, index=True)
    return buffer.getvalue()


def _spot_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for hour in range(4):
        start = pd.Timestamp("2026-10-25T00:00:00Z") + pd.Timedelta(hours=hour)
        rows.append(
            {
                "SpotProductID": 1,
                "SourceProduct": "CH_DAY_AHEAD",
                "MarketZone": "CH",
                "DeliveryStartUtc": start,
                "DeliveryEndUtc": start + pd.Timedelta(hours=1),
                "FrequencyMinutes": 60,
                "Price": 50.0 + hour,
                "PriceUnit": "EUR/MWh",
                "ObservedAtUtc": pd.Timestamp("2026-10-24T12:00:00Z"),
            }
        )
    return pd.DataFrame(rows)


def _binding(path: str, payload: bytes) -> dict[str, object]:
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _case(
    *,
    environment: str = "prd",
) -> tuple[
    dict[str, object],
    bytes,
    dict[str, bytes],
    bytes,
    bytes,
    bytes,
]:
    source_frame = _spot_rows()
    source_payload = _parquet_payload(source_frame)
    package = build_databricks_replay_package(
        role="epex_ch",
        mode=GOLD_SPOT_MODE,
        as_of_utc="2026-10-25T05:00:00Z",
        source_payloads={"spot_price_interval": source_payload},
        source_tables={"spot_price_interval": f"{environment}.gold.factspotpriceinterval"},
        selection={"market_zone": "CH", "source_product": "CH_DAY_AHEAD"},
    )
    artifacts = dict(package.artifacts)
    query = {
        "source_table": f"{environment}.gold.factspotpriceinterval",
        "selected_columns": [str(column) for column in source_frame.columns],
        "predicate_sql": (
            "SELECT SpotProductID, SourceProduct, MarketZone, DeliveryStartUtc, "
            "DeliveryEndUtc, FrequencyMinutes, Price, PriceUnit, ObservedAtUtc "
            f"FROM {environment}.gold.factspotpriceinterval "
            "WHERE MarketZone = 'CH' AND ObservedAtUtc <= TIMESTAMP "
            "'2026-10-25T05:00:00Z'"
        ),
        "watermark_column": "ObservedAtUtc",
        "lower_watermark_exclusive_utc": None,
        "upper_watermark_inclusive_utc": "2026-10-25T05:00:00Z",
        "row_count": len(source_frame),
        "artifact_sha256": hashlib.sha256(source_payload).hexdigest(),
        "artifact_size_bytes": len(source_payload),
    }
    unsigned_export = {
        "schema_version": DATABRICKS_EXPORT_MANIFEST_SCHEMA,
        "role": "epex_ch",
        "source_environment": environment.upper(),
        "export_mode": DATABRICKS_EXPORT_MODE,
        "predecessor_generation_id": None,
        "as_of_utc": "2026-10-25T05:00:00+00:00",
        "exported_at_utc": "2026-10-25T05:30:00+00:00",
        "source_queries": {"spot_price_interval": query},
        "cost_evidence": {
            "schema_version": "fmv_databricks_lt_export_cost.v1",
            "evidence_basis": "CLIENT_OBSERVATION",
            "statement_count": 1,
            "warehouse_start_count": 0,
            "rows_exported": len(source_frame),
            "bytes_exported": len(source_payload),
            "billing_usage_quantity_dbcu": None,
            "estimated_cost_chf": None,
        },
        "authorities": {
            "source_authenticity_verified": False,
            "model_input_authorized": False,
            "calibration_authorized": False,
            "publication_authorized": False,
            "production_authorized": False,
        },
    }
    export = {
        **unsigned_export,
        "export_id": databricks_export_id(unsigned_export),
    }
    export_payload = canonical_json_bytes(export)
    bindings = {
        artifact_role: _binding(
            (
                "raw/epex_ch.parquet"
                if artifact_role == "model/raw.parquet"
                else "inputs/epex_ch.parquet"
                if artifact_role == "model/derived.parquet"
                else f"databricks/epex_ch/{artifact_role}"
            ),
            payload,
        )
        for artifact_role, payload in artifacts.items()
    }
    entry = {
        "source_system": "EPEX_SPOT",
        "source_receipt": {
            "received_at_utc": "2026-10-25T05:45:00+00:00",
            "source_locator": "databricks://prd.gold.factspotpriceinterval",
        },
        "sha256": bindings["model/derived.parquet"]["sha256"],
        "raw_artifact": {
            **bindings["model/raw.parquet"],
        },
        "derivation": {
            "parser_code_sha256": "1" * 64,
            "parser_config_sha256": "2" * 64,
        },
        "upstream_replay": {
            "kind": DATABRICKS_UPSTREAM_REPLAY_KIND,
            "replayed_at_utc": "2026-10-25T06:00:00+00:00",
            "manifest": _binding(
                "databricks/epex_ch/replay-manifest.json",
                package.manifest_payload,
            ),
            "artifacts": bindings,
            "export_manifest": _binding(
                "databricks/epex_ch/export-manifest.json",
                export_payload,
            ),
        },
    }
    return (
        entry,
        package.manifest_payload,
        artifacts,
        export_payload,
        artifacts["model/raw.parquet"],
        artifacts["model/derived.parquet"],
    )


def _verify(case: tuple[object, ...]) -> dict[str, object]:
    entry, manifest, artifacts, export, raw, derived = case
    assert isinstance(entry, dict)
    assert isinstance(manifest, bytes)
    assert isinstance(artifacts, dict)
    assert isinstance(export, bytes)
    assert isinstance(raw, bytes)
    assert isinstance(derived, bytes)
    return verify_databricks_snapshot_replay(
        role="epex_ch",
        source_system="EPEX_SPOT",
        entry=entry,
        manifest_payload=manifest,
        artifact_payloads=artifacts,
        export_manifest_payload=export,
        raw_payload=raw,
        derived_payload=derived,
    )


def test_prd_databricks_snapshot_replay_chain_is_exact() -> None:
    case = _case()
    result = _verify(case)
    entry = case[0]
    assert isinstance(entry, dict)

    assert result["status"] == "VERIFIED_SIGNABLE_PRD_DATABRICKS_REPLAY_CHAIN"
    bindings = expected_databricks_quality_bindings(
        entry=entry,
        raw=entry["raw_artifact"],
        derivation=entry["derivation"],
    )
    assert bindings["databricks_replay_manifest_sha256"] == hashlib.sha256(case[1]).hexdigest()


def test_dev_databricks_snapshot_cannot_be_calibration_evidence() -> None:
    with pytest.raises(DatabricksLTSnapshotError, match="requires PRD"):
        _verify(_case(environment="dev"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("predecessor_generation_id", "generation-previous", "incremental composition"),
        ("source_environment", "DEV", "requires PRD"),
    ],
)
def test_databricks_export_semantic_tampering_fails_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    case = list(_case())
    export = dict(json.loads(case[3]))
    export[field] = value
    identity = dict(export)
    identity.pop("export_id")
    export["export_id"] = databricks_export_id(identity)
    changed = canonical_json_bytes(export)
    entry = case[0]
    assert isinstance(entry, dict)
    upstream = entry["upstream_replay"]
    assert isinstance(upstream, dict)
    upstream["export_manifest"] = _binding(
        "databricks/epex_ch/export-manifest.json",
        changed,
    )
    case[3] = changed

    with pytest.raises(DatabricksLTSnapshotError, match=message):
        _verify(tuple(case))


def test_databricks_source_bytes_tampering_fails_closed() -> None:
    case = list(_case())
    artifacts = dict(case[2])
    artifacts["sources/spot_price_interval.parquet"] += b"tampered"
    case[2] = artifacts

    with pytest.raises(DatabricksLTSnapshotError, match="bytes differ"):
        _verify(tuple(case))
