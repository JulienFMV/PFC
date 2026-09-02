"""Governed bridge from bounded Silver exports to the day-ahead consumer.

The module has no Databricks connector.  It validates already exported rows,
keeps causal and realized uses separate, and refuses to manufacture original
publication or finality authority from technical vintage metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from numbers import Integral
from pathlib import Path

import pandas as pd

from pfc_shaping.data.governed_lt_acquisition import dataframe_semantic_sha256
from pfc_shaping.parquet_safety import validate_parquet_allocation_budget
from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.validation.entsoe_day_ahead_consumption import (
    FIELD_TO_ZONE,
    MARKET_TIMEZONES,
    RESOLUTION_SECONDS,
    SOURCE_COLUMNS,
    AvailabilityBasis,
    SpotUsage,
)

ROOT = Path(__file__).resolve().parents[2]
CAUSAL_SQL_PATH = ROOT / "docs/data/sql/databricks_prd_entsoe_day_ahead_causal_export_v2.sql"
REALIZED_SQL_PATH = ROOT / "docs/data/sql/databricks_prd_entsoe_day_ahead_realized_export_v2.sql"
CAUSAL_SQL_SHA256 = "f86cfeeec6bb2dc6d9c579426df091d7d45b01f0bca848b47aba62500cedbc6d"
REALIZED_SQL_SHA256 = "9f4104e4db8b16bfa21e0fce26afb47d364f983b95209424293356489fe6d81a"

EXPORT_SCHEMA = "fmv_entsoe_day_ahead_consumer_export.v2"
PREFLIGHT_SCHEMA = "fmv_entsoe_day_ahead_export_cost_preflight.v1"
REPLAY_SCHEMA = "fmv_entsoe_day_ahead_export_replay.v1"
RESULT_ROW_LIMIT = 20_001
MAX_WINDOW_DAYS = 32

RAW_COLUMNS = (
    "field_name",
    "series_key",
    "classification_sequence",
    "interval_start_utc",
    "date_time_utc",
    "interval_end_utc",
    "resolution",
    "price_eur_per_mwh",
    "publication_timestamp_utc",
    "first_seen_pull_ts_utc",
    "availability_basis",
    "availability_known",
    "availability_timestamp_utc",
    "is_historical",
    "dq_failed",
    "source_time_series_id",
    "source_document_mrid",
    "source_document_revision_number",
    "source_snapshot_id",
    "source_file_path",
    "vintage_id",
)

_SERIES_CANDIDATES = {
    "ch_price": frozenset({"day_ahead_prices||ch_price"}),
    "at_price": frozenset({"day_ahead_prices||at_price||1", "day_ahead_prices||at_price||2"}),
    "de_lu_price": frozenset(
        {"day_ahead_prices||de_lu_price||1", "day_ahead_prices||de_lu_price||2"}
    ),
    "fr_price": frozenset({"day_ahead_prices||fr_price"}),
    "it_nord_price": frozenset({"day_ahead_prices||it_nord_price"}),
}
_SELECTION_PARAMETER = {
    field: f"{field.removesuffix('_price')}_series_key" for field in FIELD_TO_ZONE
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CLEAN_TEXT = re.compile(r"^[^\x00-\x1f\x7f]+$")
_REPLAY_ARTIFACT_PATHS = frozenset(
    {
        "source/day-ahead-export.parquet",
        "consumer/day-ahead-source.parquet",
        "evidence/export-audit.json",
    }
)
_REPLAY_AUTHORITIES = {
    "model_input_authorized": False,
    "model_selection_authorized": False,
    "publication_authorized": False,
    "production_authorized": False,
}
_REPLAY_EXECUTION = {
    "databricks_connection_count": 0,
    "databricks_statement_count": 0,
    "business_row_count_opened": 0,
    "warehouse_start_count": 0,
    "databricks_write_count": 0,
    "network_call_count": 0,
    "remote_write_count": 0,
}
_PARQUET_PHYSICAL_TYPES = {
    "BOOLEAN",
    "INT32",
    "INT64",
    "FLOAT",
    "DOUBLE",
    "BYTE_ARRAY",
    "FIXED_LEN_BYTE_ARRAY",
}


class DayAheadMarketUse(str, Enum):
    """Business purpose, not trading or model authority."""

    VALUATION_HEDGE_SCOPE = "VALUATION_HEDGE_SCOPE"
    OBSERVATION_RISK = "OBSERVATION_RISK"
    FUTURE_MARKET_CANDIDATE = "FUTURE_MARKET_CANDIDATE"


class EvidenceKind(str, Enum):
    """Claims that Silver cannot make on its own."""

    REALIZED_FINALITY = "REALIZED_FINALITY"
    ORIGINAL_PUBLICATION = "ORIGINAL_PUBLICATION"


class EvidenceAuthority(str, Enum):
    """Admitted external evidence authorities."""

    PLATFORM_SIGNED_FINALITY_RECEIPT = "PLATFORM_SIGNED_FINALITY_RECEIPT"
    INDEPENDENT_SETTLEMENT_RECONCILIATION = "INDEPENDENT_SETTLEMENT_RECONCILIATION"
    PLATFORM_SIGNED_ORIGINAL_PUBLICATION_RECEIPT = "PLATFORM_SIGNED_ORIGINAL_PUBLICATION_RECEIPT"


class EntsoeDayAheadExportError(ValueError):
    """Fail-closed rejection of an unsafe export or evidence claim."""


@dataclass(frozen=True)
class ScopedDayAheadEvidence:
    """Hash-bound external evidence scoped to exact series and delivery bounds."""

    kind: EvidenceKind | str
    authority: EvidenceAuthority | str
    evidence_id: str
    evidence_document_sha256: str
    covered_frame_semantic_sha256: str
    asserted_at_utc: str | pd.Timestamp
    window_start_utc: str | pd.Timestamp
    window_end_utc: str | pd.Timestamp
    series_keys: Sequence[str]


@dataclass(frozen=True)
class DayAheadConsumerExport:
    """Consumer-ready rows plus an authority-negative audit."""

    frame: pd.DataFrame
    audit: Mapping[str, object]


@dataclass(frozen=True)
class DayAheadExportCostAssessment:
    """Offline cost fence; never execution authority."""

    assessment: Mapping[str, object]
    assessment_content_id: str


@dataclass(frozen=True)
class DayAheadExportReplayPackage:
    """Self-contained unsigned local replay package."""

    artifacts: Mapping[str, bytes]
    manifest_payload: bytes


def verify_export_sql_bindings() -> dict[str, str]:
    """Verify immutable v2 SQL templates without touching historical SQL."""

    observed = {
        "causal_sql_sha256": _file_sha256(CAUSAL_SQL_PATH, "causal export SQL"),
        "realized_sql_sha256": _file_sha256(REALIZED_SQL_PATH, "realized export SQL"),
    }
    expected = {
        "causal_sql_sha256": CAUSAL_SQL_SHA256,
        "realized_sql_sha256": REALIZED_SQL_SHA256,
    }
    if observed != expected:
        raise EntsoeDayAheadExportError("day-ahead export SQL binding differs")
    return observed


def build_causal_export_parameters(
    *,
    series_selection: Mapping[str, str],
    window_start_utc: str | pd.Timestamp,
    window_end_utc: str | pd.Timestamp,
    as_of_utc: str | pd.Timestamp,
) -> dict[str, object]:
    """Build parameters for a bounded, explicit-subset causal export."""

    verify_export_sql_bindings()
    selection = _series_selection(series_selection)
    window = _window_parameters(window_start_utc, window_end_utc)
    return {
        **window,
        "as_of_utc": _utc_text(_utc_timestamp(as_of_utc, "causal as-of")),
        **_selection_parameters(selection),
    }


def build_realized_export_parameters(
    *,
    series_selection: Mapping[str, str],
    window_start_utc: str | pd.Timestamp,
    window_end_utc: str | pd.Timestamp,
    assessed_at_utc: str | pd.Timestamp,
) -> dict[str, object]:
    """Build parameters for a bounded latest-revision candidate export."""

    verify_export_sql_bindings()
    selection = _series_selection(series_selection)
    window = _window_parameters(window_start_utc, window_end_utc)
    assessed = _utc_timestamp(assessed_at_utc, "realized assessment time")
    if assessed < _utc_timestamp(window["end_utc"], "realized window end"):
        raise EntsoeDayAheadExportError(
            "realized assessment must not precede delivery-window completion"
        )
    return {
        **window,
        "assessed_at_utc": _utc_text(assessed),
        **_selection_parameters(selection),
    }


def validate_causal_export(
    frame: pd.DataFrame,
    *,
    series_selection: Mapping[str, str],
    market_uses: Mapping[str, DayAheadMarketUse | str],
    window_start_utc: str | pd.Timestamp,
    window_end_utc: str | pd.Timestamp,
    as_of_utc: str | pd.Timestamp,
    query_sha256: str,
    publication_evidence: Sequence[ScopedDayAheadEvidence] = (),
) -> DayAheadConsumerExport:
    """Validate one consumer-complete causal export and adapt it exactly."""

    _expected_sha(query_sha256, CAUSAL_SQL_SHA256, "causal query")
    parameters = build_causal_export_parameters(
        series_selection=series_selection,
        window_start_utc=window_start_utc,
        window_end_utc=window_end_utc,
        as_of_utc=as_of_utc,
    )
    selection = _series_selection(series_selection)
    uses = _market_uses(market_uses, selection)
    raw = _normalize_export_frame(
        frame,
        selection=selection,
        window_start=_utc_timestamp(parameters["start_utc"], "causal start"),
        window_end=_utc_timestamp(parameters["end_utc"], "causal end"),
    )
    as_of = _utc_timestamp(parameters["as_of_utc"], "causal as-of")
    if (~raw["availability_known"]).any() or raw["availability_timestamp_utc"].isna().any():
        raise EntsoeDayAheadExportError("causal export has unknown availability")
    if raw["availability_timestamp_utc"].gt(as_of).any():
        raise EntsoeDayAheadExportError("causal export leaks a value after the as-of")
    if raw["availability_basis"].eq(AvailabilityBasis.UNKNOWN_BACKFILL.value).any():
        raise EntsoeDayAheadExportError("unknown backfill cannot enter a causal export")

    source_created_keys = frozenset(
        raw.loc[
            raw["availability_basis"].eq(AvailabilityBasis.SOURCE_DOCUMENT_CREATED.value),
            "series_key",
        ]
    )
    evidence = _validate_evidence_cover(
        publication_evidence,
        kind=EvidenceKind.ORIGINAL_PUBLICATION,
        expected_series_keys=source_created_keys,
        window_start=_utc_timestamp(parameters["start_utc"], "causal start"),
        window_end=_utc_timestamp(parameters["end_utc"], "causal end"),
        maximum_asserted_at=None,
        source_frame=raw,
    )
    source_created = raw["availability_basis"].eq(AvailabilityBasis.SOURCE_DOCUMENT_CREATED.value)
    if (
        raw.loc[source_created, "publication_timestamp_utc"]
        .ge(raw.loc[source_created, "interval_start_utc"])
        .any()
    ):
        raise EntsoeDayAheadExportError(
            "proven original day-ahead publication must precede delivery"
        )
    consumer = _consumer_frame(
        raw,
        usage=SpotUsage.CAUSAL_ASOF,
        original_publication_keys=source_created_keys,
    )
    return _result(
        raw=raw,
        consumer=consumer,
        usage=SpotUsage.CAUSAL_ASOF,
        selection=selection,
        market_uses=uses,
        window_start=str(parameters["start_utc"]),
        window_end=str(parameters["end_utc"]),
        temporal_cutoff=_utc_text(as_of),
        query_sha256=CAUSAL_SQL_SHA256,
        evidence=evidence,
    )


def validate_realized_export(
    frame: pd.DataFrame,
    *,
    series_selection: Mapping[str, str],
    market_uses: Mapping[str, DayAheadMarketUse | str],
    window_start_utc: str | pd.Timestamp,
    window_end_utc: str | pd.Timestamp,
    assessed_at_utc: str | pd.Timestamp,
    query_sha256: str,
    finality_evidence: Sequence[ScopedDayAheadEvidence],
) -> DayAheadConsumerExport:
    """Validate a latest-revision export only with exact finality evidence."""

    _expected_sha(query_sha256, REALIZED_SQL_SHA256, "realized query")
    parameters = build_realized_export_parameters(
        series_selection=series_selection,
        window_start_utc=window_start_utc,
        window_end_utc=window_end_utc,
        assessed_at_utc=assessed_at_utc,
    )
    selection = _series_selection(series_selection)
    uses = _market_uses(market_uses, selection)
    start = _utc_timestamp(parameters["start_utc"], "realized start")
    end = _utc_timestamp(parameters["end_utc"], "realized end")
    assessed = _utc_timestamp(parameters["assessed_at_utc"], "realized assessment")
    raw = _normalize_export_frame(
        frame,
        selection=selection,
        window_start=start,
        window_end=end,
    )
    if (
        raw["first_seen_pull_ts_utc"].isna().any()
        or raw["first_seen_pull_ts_utc"].gt(assessed).any()
    ):
        raise EntsoeDayAheadExportError(
            "realized export contains a value not observed by the assessment cutoff"
        )
    evidence = _validate_evidence_cover(
        finality_evidence,
        kind=EvidenceKind.REALIZED_FINALITY,
        expected_series_keys=frozenset(selection.values()),
        window_start=start,
        window_end=end,
        maximum_asserted_at=assessed,
        source_frame=raw,
    )
    consumer = _consumer_frame(
        raw,
        usage=SpotUsage.REALIZED_FINAL,
        original_publication_keys=frozenset(),
    )
    return _result(
        raw=raw,
        consumer=consumer,
        usage=SpotUsage.REALIZED_FINAL,
        selection=selection,
        market_uses=uses,
        window_start=str(parameters["start_utc"]),
        window_end=str(parameters["end_utc"]),
        temporal_cutoff=_utc_text(assessed),
        query_sha256=REALIZED_SQL_SHA256,
        evidence=evidence,
    )


def assess_export_cost_preflight(
    *,
    usage: SpotUsage | str,
    warehouse_state: str,
    warehouse_started_for_request: bool,
    partition_pruning_proven: bool,
    scan_upper_bound_is_hard: bool,
    estimated_scan_upper_bound_bytes: int | None,
    maximum_scan_bytes: int,
    maximum_runtime_seconds: int,
) -> DayAheadExportCostAssessment:
    """Assess an offline cost fence with zero execution authority."""

    try:
        selected_usage = SpotUsage(usage)
    except ValueError as exc:
        raise EntsoeDayAheadExportError("cost preflight usage is invalid") from exc
    state = _text(warehouse_state, "Warehouse state")
    for value, label in (
        (warehouse_started_for_request, "Warehouse-start flag"),
        (partition_pruning_proven, "partition-pruning flag"),
        (scan_upper_bound_is_hard, "hard-cap flag"),
    ):
        if type(value) is not bool:
            raise EntsoeDayAheadExportError(f"{label} must be boolean")
    if warehouse_started_for_request:
        raise EntsoeDayAheadExportError(
            "cost preflight cannot admit a Warehouse started for this request"
        )
    ceiling = _positive_int(maximum_scan_bytes, "maximum scan bytes")
    runtime = _positive_int(maximum_runtime_seconds, "maximum runtime seconds")
    if estimated_scan_upper_bound_bytes is not None:
        estimate = _nonnegative_int(estimated_scan_upper_bound_bytes, "estimated scan upper bound")
    else:
        estimate = None

    if state != "RUNNING_ALREADY_FOR_SEPARATE_AUTHORIZED_WORKLOAD":
        status = "STOP_NO_ACTIVE_WAREHOUSE"
    elif not partition_pruning_proven or not scan_upper_bound_is_hard or estimate is None:
        status = "STOP_SCAN_BOUND_UNPROVEN"
    elif estimate > ceiling:
        status = "STOP_SCAN_CAP_EXCEEDED"
    else:
        status = "READY_FOR_HUMAN_COST_REVIEW_NO_EXECUTION_AUTHORITY"
    query_sha = (
        CAUSAL_SQL_SHA256 if selected_usage is SpotUsage.CAUSAL_ASOF else REALIZED_SQL_SHA256
    )
    assessment = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": status,
        "usage": selected_usage.value,
        "query_sha256": query_sha,
        "warehouse_state": state,
        "warehouse_started_for_request": False,
        "partition_predicate": "AT_MOST_TWO_EXPLICIT_YEAR_MONTH_PARTITIONS",
        "partition_pruning_proven": partition_pruning_proven,
        "scan_upper_bound_is_hard": scan_upper_bound_is_hard,
        "estimated_scan_upper_bound_bytes": estimate,
        "maximum_scan_bytes": ceiling,
        "maximum_runtime_seconds": runtime,
        "human_cost_review_required": True,
        "execution_authorized": False,
        "execution": {
            "databricks_connection_count": 0,
            "databricks_statement_count": 0,
            "business_row_count_opened": 0,
            "warehouse_start_count": 0,
            "databricks_write_count": 0,
            "network_call_count": 0,
        },
    }
    return DayAheadExportCostAssessment(
        assessment=assessment,
        assessment_content_id=_canonical_sha256(assessment),
    )


def build_export_replay_package(
    *,
    raw_frame: pd.DataFrame,
    export: DayAheadConsumerExport,
) -> DayAheadExportReplayPackage:
    """Archive one validated synthetic/local export without filesystem writes."""

    if not isinstance(export, DayAheadConsumerExport):
        raise EntsoeDayAheadExportError("export replay input has the wrong type")
    audit = dict(export.audit)
    if audit.get("schema_version") != EXPORT_SCHEMA:
        raise EntsoeDayAheadExportError("export replay audit schema is invalid")
    if audit.get("raw_frame_semantic_sha256") != dataframe_semantic_sha256(raw_frame):
        raise EntsoeDayAheadExportError("raw export differs from its audit")
    if audit.get("consumer_frame_semantic_sha256") != dataframe_semantic_sha256(export.frame):
        raise EntsoeDayAheadExportError("consumer export differs from its audit")
    artifacts = {
        "source/day-ahead-export.parquet": _parquet_payload(raw_frame),
        "consumer/day-ahead-source.parquet": _parquet_payload(export.frame),
        "evidence/export-audit.json": _canonical_json_bytes(audit),
    }
    manifest_without_id = {
        "schema_version": REPLAY_SCHEMA,
        "usage": audit.get("usage"),
        "query_sha256": audit.get("query_sha256"),
        "artifacts": {
            path: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for path, payload in sorted(artifacts.items())
        },
        "status": "UNSIGNED_LOCAL_DAY_AHEAD_EXPORT_REPLAY_NOT_PUBLISHED",
        "execution": dict(_REPLAY_EXECUTION),
        "authorities": dict(_REPLAY_AUTHORITIES),
    }
    manifest = {
        **manifest_without_id,
        "build_id": _canonical_sha256(manifest_without_id),
    }
    return DayAheadExportReplayPackage(
        artifacts=artifacts,
        manifest_payload=_canonical_json_bytes(manifest),
    )


def verify_export_replay_package(
    *,
    artifacts: Mapping[str, bytes],
    manifest_payload: bytes,
) -> dict[str, object]:
    """Verify exact bytes and rerun the local export adaptation."""

    manifest = _strict_json_mapping(manifest_payload, "export replay manifest")
    expected_manifest_fields = {
        "schema_version",
        "build_id",
        "usage",
        "query_sha256",
        "artifacts",
        "status",
        "execution",
        "authorities",
    }
    if set(manifest) != expected_manifest_fields or manifest.get("schema_version") != REPLAY_SCHEMA:
        raise EntsoeDayAheadExportError("export replay manifest fields are not exact")
    identity = dict(manifest)
    build_id = _sha256(identity.pop("build_id", None), "export replay build ID")
    if _canonical_sha256(identity) != build_id:
        raise EntsoeDayAheadExportError("export replay build ID is invalid")
    if manifest.get("status") != "UNSIGNED_LOCAL_DAY_AHEAD_EXPORT_REPLAY_NOT_PUBLISHED":
        raise EntsoeDayAheadExportError("export replay status is invalid")
    if manifest.get("execution") != _REPLAY_EXECUTION:
        raise EntsoeDayAheadExportError("export replay execution declaration differs")
    if manifest.get("authorities") != _REPLAY_AUTHORITIES:
        raise EntsoeDayAheadExportError("export replay grants unsupported authority")
    if not isinstance(artifacts, Mapping) or set(artifacts) != _REPLAY_ARTIFACT_PATHS:
        raise EntsoeDayAheadExportError("export replay artifact inventory differs")
    declarations = manifest.get("artifacts")
    if not isinstance(declarations, Mapping) or set(declarations) != set(artifacts):
        raise EntsoeDayAheadExportError("export replay declarations differ")
    for path, payload in artifacts.items():
        declaration = declarations[path]
        if not isinstance(declaration, Mapping) or set(declaration) != {
            "sha256",
            "size_bytes",
        }:
            raise EntsoeDayAheadExportError(
                f"export replay artifact declaration is invalid: {path}"
            )
        if declaration != {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }:
            raise EntsoeDayAheadExportError(f"export replay artifact changed: {path}")

    raw = _read_replay_parquet(
        artifacts["source/day-ahead-export.parquet"],
        columns=RAW_COLUMNS,
        label="raw export",
    )
    consumer = _read_replay_parquet(
        artifacts["consumer/day-ahead-source.parquet"],
        columns=SOURCE_COLUMNS,
        label="consumer export",
    )
    audit = _strict_json_mapping(artifacts["evidence/export-audit.json"], "export replay audit")
    replayed = _replay_export(raw, audit)
    try:
        pd.testing.assert_frame_equal(
            replayed.frame,
            consumer,
            check_dtype=True,
            check_exact=True,
            check_like=False,
            check_freq=False,
        )
    except AssertionError as exc:
        raise EntsoeDayAheadExportError(
            "export replay does not reproduce archived consumer rows"
        ) from exc
    if dict(replayed.audit) != audit:
        raise EntsoeDayAheadExportError("export replay does not reproduce archived audit")
    if manifest.get("usage") != audit.get("usage") or manifest.get("query_sha256") != audit.get(
        "query_sha256"
    ):
        raise EntsoeDayAheadExportError("export replay manifest/audit binding differs")
    return {
        "schema_version": REPLAY_SCHEMA,
        "status": "VERIFIED_SELF_CONTAINED_DAY_AHEAD_EXPORT_REPLAY",
        "build_id": build_id,
        "usage": audit["usage"],
        "artifact_count": len(artifacts),
        "raw_row_count": len(raw),
        "consumer_row_count": len(consumer),
        "authorities": dict(_REPLAY_AUTHORITIES),
        **_REPLAY_EXECUTION,
    }


def _normalize_export_frame(
    frame: object,
    *,
    selection: Mapping[str, str],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise EntsoeDayAheadExportError("day-ahead export must be a DataFrame")
    if frame.columns.has_duplicates or tuple(frame.columns) != RAW_COLUMNS:
        raise EntsoeDayAheadExportError("day-ahead export columns are not exact")
    if frame.empty:
        raise EntsoeDayAheadExportError("day-ahead export is empty")
    if len(frame) >= RESULT_ROW_LIMIT:
        raise EntsoeDayAheadExportError("day-ahead export hit the rejection sentinel")
    result = frame.copy()
    for column in (
        "field_name",
        "series_key",
        "resolution",
        "availability_basis",
        "source_time_series_id",
        "source_document_mrid",
        "source_snapshot_id",
        "source_file_path",
        "vintage_id",
    ):
        result[column] = result[column].map(lambda value: _text(value, column))
    result["classification_sequence"] = result["classification_sequence"].map(
        lambda value: _optional_text(value, "classification sequence")
    )
    for column in (
        "interval_start_utc",
        "date_time_utc",
        "interval_end_utc",
        "publication_timestamp_utc",
        "first_seen_pull_ts_utc",
        "availability_timestamp_utc",
    ):
        result[column] = pd.to_datetime(result[column], errors="coerce", utc=True)
    for column in ("availability_known", "is_historical", "dq_failed"):
        if not result[column].map(lambda value: type(value) is bool).all():
            raise EntsoeDayAheadExportError(f"{column} must contain exact booleans")
    required_timestamps = ("interval_start_utc", "date_time_utc", "interval_end_utc")
    if result.loc[:, required_timestamps].isna().any().any():
        raise EntsoeDayAheadExportError("normalized interval bounds are missing")
    if result["date_time_utc"].ne(result["interval_end_utc"]).any():
        raise EntsoeDayAheadExportError("Date_Time_UTC differs from IntervalEndUtc")
    if (
        result["interval_start_utc"].lt(window_start).any()
        or result["interval_end_utc"].gt(window_end).any()
    ):
        raise EntsoeDayAheadExportError("export row lies outside the delivery window")

    expected_keys = result["field_name"].map(selection)
    if expected_keys.isna().any() or result["series_key"].ne(expected_keys).any():
        raise EntsoeDayAheadExportError("export differs from explicit series selection")
    if set(result["field_name"]) != set(selection):
        raise EntsoeDayAheadExportError("export does not contain every selected field")
    for row in result.itertuples(index=False):
        expected_classification = row.series_key.split("||")[2:]
        expected_sequence = expected_classification[0] if expected_classification else None
        if row.classification_sequence != expected_sequence:
            raise EntsoeDayAheadExportError(
                "classification sequence differs from the selected SeriesKey"
            )
        try:
            basis = AvailabilityBasis(row.availability_basis)
        except ValueError as exc:
            raise EntsoeDayAheadExportError("availability basis is invalid") from exc
        if basis is AvailabilityBasis.FMV_FIRST_SEEN:
            if (
                not row.availability_known
                or pd.isna(row.first_seen_pull_ts_utc)
                or row.availability_timestamp_utc != row.first_seen_pull_ts_utc
            ):
                raise EntsoeDayAheadExportError(
                    "FMV_FIRST_SEEN availability fields are inconsistent"
                )
        elif basis is AvailabilityBasis.SOURCE_DOCUMENT_CREATED:
            if (
                not row.availability_known
                or pd.isna(row.publication_timestamp_utc)
                or row.availability_timestamp_utc != row.publication_timestamp_utc
            ):
                raise EntsoeDayAheadExportError(
                    "SOURCE_DOCUMENT_CREATED availability fields are inconsistent"
                )
        elif row.availability_known or not pd.isna(row.availability_timestamp_utc):
            raise EntsoeDayAheadExportError("UNKNOWN_BACKFILL availability fields are inconsistent")
    if result["dq_failed"].any():
        raise EntsoeDayAheadExportError("day-ahead export contains failed DQ rows")

    seconds = result["resolution"].map(RESOLUTION_SECONDS)
    if seconds.isna().any():
        raise EntsoeDayAheadExportError("day-ahead export resolution is unsupported")
    resolution_ns = seconds.astype("int64") * 1_000_000_000
    start_ns = result["interval_start_utc"].map(lambda value: value.value)
    end_ns = result["interval_end_utc"].map(lambda value: value.value)
    duration_ns = end_ns - start_ns
    if duration_ns.le(0).any() or duration_ns.mod(resolution_ns).ne(0).any():
        raise EntsoeDayAheadExportError(
            "interval duration must be a positive native-resolution multiple"
        )
    if start_ns.mod(resolution_ns).ne(0).any() or end_ns.mod(resolution_ns).ne(0).any():
        raise EntsoeDayAheadExportError("interval bounds are off the native grid")
    prices = pd.to_numeric(result["price_eur_per_mwh"], errors="coerce")
    if prices.isna().any() or not prices.map(lambda value: math.isfinite(float(value))).all():
        raise EntsoeDayAheadExportError("day-ahead export price is not finite")
    result["price_eur_per_mwh"] = prices.astype(float)
    revisions = pd.to_numeric(result["source_document_revision_number"], errors="coerce")
    if revisions.isna().any() or revisions.lt(0).any() or revisions.mod(1).ne(0).any():
        raise EntsoeDayAheadExportError("source revision is not a nonnegative integer")
    result["source_document_revision_number"] = revisions.astype(int)

    grain = ["field_name", "series_key", "interval_start_utc", "interval_end_utc"]
    if result.duplicated(grain, keep=False).any() or result["vintage_id"].duplicated().any():
        raise EntsoeDayAheadExportError("day-ahead export grain is duplicated")
    result = result.sort_values(
        ["field_name", "interval_start_utc", "interval_end_utc"], kind="mergesort"
    ).reset_index(drop=True)
    previous_end = result.groupby(["field_name", "series_key"], sort=False)[
        "interval_end_utc"
    ].shift()
    if result["interval_start_utc"].lt(previous_end).any():
        raise EntsoeDayAheadExportError("day-ahead export intervals overlap")
    return result


def _consumer_frame(
    raw: pd.DataFrame,
    *,
    usage: SpotUsage,
    original_publication_keys: frozenset[str],
) -> pd.DataFrame:
    consumer = pd.DataFrame(
        {
            "field_name": raw["field_name"],
            "market_zone": raw["field_name"].map(FIELD_TO_ZONE),
            "market_timezone": raw["field_name"].map(
                lambda field: MARKET_TIMEZONES[FIELD_TO_ZONE[field]]
            ),
            "series_key": raw["series_key"],
            "classification_sequence": raw["classification_sequence"],
            "interval_start_utc": raw["interval_start_utc"],
            "interval_end_utc": raw["interval_end_utc"],
            "native_resolution": raw["resolution"],
            "price_eur_per_mwh": raw["price_eur_per_mwh"],
            "publication_timestamp_utc": raw["publication_timestamp_utc"],
            "first_seen_pull_ts_utc": raw["first_seen_pull_ts_utc"],
            "availability_basis": raw["availability_basis"],
            "original_publication_proven": raw["series_key"].isin(original_publication_keys)
            & raw["availability_basis"].eq(AvailabilityBasis.SOURCE_DOCUMENT_CREATED.value),
            "is_historical": raw["is_historical"],
            "is_final": usage is SpotUsage.REALIZED_FINAL,
            "dq_failed": raw["dq_failed"],
            "quality_status": "PASSED",
            "source_time_series_id": raw["source_time_series_id"],
            "source_document_mrid": raw["source_document_mrid"],
            "source_document_revision_number": raw["source_document_revision_number"],
            "source_snapshot_id": raw["source_snapshot_id"],
            "source_file_path": raw["source_file_path"],
        }
    )
    return consumer.loc[:, SOURCE_COLUMNS].reset_index(drop=True)


def _result(
    *,
    raw: pd.DataFrame,
    consumer: pd.DataFrame,
    usage: SpotUsage,
    selection: Mapping[str, str],
    market_uses: Mapping[str, str],
    window_start: str,
    window_end: str,
    temporal_cutoff: str,
    query_sha256: str,
    evidence: Sequence[Mapping[str, object]],
) -> DayAheadConsumerExport:
    audit = {
        "schema_version": EXPORT_SCHEMA,
        "status": f"PASS_{usage.value.upper()}_EXPORT_NOT_MODEL_AUTHORITY",
        "usage": usage.value,
        "query_sha256": query_sha256,
        "window_start_utc": window_start,
        "window_end_utc": window_end,
        "temporal_cutoff_utc": temporal_cutoff,
        "series_selection": dict(selection),
        "market_uses": dict(market_uses),
        "raw_row_count": len(raw),
        "consumer_row_count": len(consumer),
        "raw_frame_semantic_sha256": dataframe_semantic_sha256(raw),
        "consumer_frame_semantic_sha256": dataframe_semantic_sha256(consumer),
        "evidence": list(evidence),
        "evidence_content_id": _canonical_sha256(list(evidence)),
        "evidence_authentication_policy": (
            "UPSTREAM_GOVERNED_ADMISSION_REQUIRED_LOCAL_SCOPE_AND_HASH_BINDING_ONLY"
        ),
        "lineage_policy": {
            "intervals": "PRODUCER_NORMALIZED_HALF_OPEN_UTC",
            "curve_type": "NOT_EXPORTED_NOT_INFERRED",
            "silver_vintage_metadata": (
                "LATEST_RETAINED_LINEAGE_FOR_SEMANTIC_VINTAGE_NOT_A_CAUSAL_FEATURE"
            ),
            "quality_status": "PASSED_ONLY_AFTER_DQ_FAILED_FALSE",
            "finality": "EXTERNAL_EVIDENCE_ONLY",
            "original_publication": "EXTERNAL_EVIDENCE_ONLY",
        },
        "authorities": {
            "consumer_contract_authorized": True,
            "point_in_time_filter_validated": usage is SpotUsage.CAUSAL_ASOF,
            "realized_finality_evidence_validated": usage is SpotUsage.REALIZED_FINAL,
            "trading_execution_authorized": False,
            "monthly_level_authorized": False,
            "model_input_authorized": False,
            "model_selection_authorized": False,
            "production_authorized": False,
        },
    }
    return DayAheadConsumerExport(frame=consumer, audit=audit)


def _validate_evidence_cover(
    values: Sequence[ScopedDayAheadEvidence],
    *,
    kind: EvidenceKind,
    expected_series_keys: frozenset[str],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    maximum_asserted_at: pd.Timestamp | None,
    source_frame: pd.DataFrame,
) -> list[dict[str, object]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise EntsoeDayAheadExportError("evidence must be a sequence")
    if not expected_series_keys:
        if values:
            raise EntsoeDayAheadExportError("unexpected evidence has no scoped series")
        return []
    normalized: list[dict[str, object]] = []
    covered: set[str] = set()
    evidence_ids: set[str] = set()
    for item in values:
        if not isinstance(item, ScopedDayAheadEvidence):
            raise EntsoeDayAheadExportError("evidence item has the wrong type")
        try:
            item_kind = EvidenceKind(item.kind)
            authority = EvidenceAuthority(item.authority)
        except ValueError as exc:
            raise EntsoeDayAheadExportError("evidence kind or authority is invalid") from exc
        if item_kind is not kind:
            raise EntsoeDayAheadExportError("evidence kind differs from requested claim")
        if kind is EvidenceKind.ORIGINAL_PUBLICATION:
            if authority is not EvidenceAuthority.PLATFORM_SIGNED_ORIGINAL_PUBLICATION_RECEIPT:
                raise EntsoeDayAheadExportError(
                    "original publication requires a platform-signed receipt"
                )
        elif authority not in {
            EvidenceAuthority.PLATFORM_SIGNED_FINALITY_RECEIPT,
            EvidenceAuthority.INDEPENDENT_SETTLEMENT_RECONCILIATION,
        }:
            raise EntsoeDayAheadExportError("finality evidence authority is invalid")
        start = _utc_timestamp(item.window_start_utc, "evidence window start")
        end = _utc_timestamp(item.window_end_utc, "evidence window end")
        asserted = _utc_timestamp(item.asserted_at_utc, "evidence assertion time")
        if start != window_start or end != window_end:
            raise EntsoeDayAheadExportError("evidence delivery window differs")
        if maximum_asserted_at is not None and asserted > maximum_asserted_at:
            raise EntsoeDayAheadExportError("finality evidence postdates assessment cutoff")
        if kind is EvidenceKind.REALIZED_FINALITY and asserted < window_end:
            raise EntsoeDayAheadExportError("finality evidence predates delivery completion")
        if isinstance(item.series_keys, (str, bytes)):
            raise EntsoeDayAheadExportError("evidence SeriesKeys must be a sequence")
        keys = tuple(_text(value, "evidence SeriesKey") for value in item.series_keys)
        if not keys or len(keys) != len(set(keys)):
            raise EntsoeDayAheadExportError("evidence SeriesKeys are empty or duplicate")
        key_set = set(keys)
        if not key_set.issubset(expected_series_keys) or covered.intersection(key_set):
            raise EntsoeDayAheadExportError("evidence SeriesKey scope overlaps or differs")
        if authority is EvidenceAuthority.INDEPENDENT_SETTLEMENT_RECONCILIATION and any(
            "||it_nord_price" in key for key in key_set
        ):
            raise EntsoeDayAheadExportError(
                "independent LSEG settlement evidence cannot cover IT-North"
            )
        covered_frame = source_frame.loc[
            source_frame["series_key"].isin(key_set), RAW_COLUMNS
        ].reset_index(drop=True)
        if covered_frame.empty:
            raise EntsoeDayAheadExportError("evidence covers no exported row")
        expected_frame_sha256 = dataframe_semantic_sha256(covered_frame)
        observed_frame_sha256 = _sha256(
            item.covered_frame_semantic_sha256,
            "evidence covered-frame semantic SHA-256",
        )
        if observed_frame_sha256 != expected_frame_sha256:
            raise EntsoeDayAheadExportError("evidence does not bind the exact covered export rows")
        covered.update(key_set)
        evidence_id = _text(item.evidence_id, "evidence ID")
        if evidence_id in evidence_ids:
            raise EntsoeDayAheadExportError("evidence IDs repeat")
        evidence_ids.add(evidence_id)
        normalized.append(
            {
                "kind": item_kind.value,
                "authority": authority.value,
                "evidence_id": evidence_id,
                "evidence_document_sha256": _sha256(
                    item.evidence_document_sha256, "evidence document SHA-256"
                ),
                "covered_frame_semantic_sha256": observed_frame_sha256,
                "asserted_at_utc": _utc_text(asserted),
                "window_start_utc": _utc_text(start),
                "window_end_utc": _utc_text(end),
                "series_keys": sorted(key_set),
            }
        )
    if covered != set(expected_series_keys):
        raise EntsoeDayAheadExportError("evidence does not exactly cover selected SeriesKeys")
    return sorted(normalized, key=lambda item: str(item["evidence_id"]))


def _series_selection(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value or not set(value).issubset(FIELD_TO_ZONE):
        raise EntsoeDayAheadExportError(
            "series selection must be a non-empty subset of admitted market fields"
        )
    result: dict[str, str] = {}
    for field in FIELD_TO_ZONE:
        if field not in value:
            continue
        key = _text(value[field], f"selected SeriesKey for {field}")
        if key not in _SERIES_CANDIDATES[field]:
            raise EntsoeDayAheadExportError(
                f"selected SeriesKey is not an admitted candidate: {field}"
            )
        result[field] = key
    if len(set(result.values())) != len(result):
        raise EntsoeDayAheadExportError("series selection reuses a SeriesKey")
    return result


def _selection_parameters(selection: Mapping[str, str]) -> dict[str, str | None]:
    return {parameter: selection.get(field) for field, parameter in _SELECTION_PARAMETER.items()}


def _market_uses(
    value: Mapping[str, DayAheadMarketUse | str], selection: Mapping[str, str]
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(selection):
        raise EntsoeDayAheadExportError("market uses must exactly cover selected fields")
    result: dict[str, str] = {}
    for field in selection:
        try:
            result[field] = DayAheadMarketUse(value[field]).value
        except ValueError as exc:
            raise EntsoeDayAheadExportError(f"market use is invalid: {field}") from exc
    return result


def _window_parameters(window_start_utc: object, window_end_utc: object) -> dict[str, object]:
    start = _utc_timestamp(window_start_utc, "export window start")
    end = _utc_timestamp(window_end_utc, "export window end")
    if start >= end or end - start > pd.Timedelta(days=MAX_WINDOW_DAYS):
        raise EntsoeDayAheadExportError("export window is empty, inverted or exceeds 32 days")
    if any(value.second or value.microsecond or value.minute % 15 for value in (start, end)):
        raise EntsoeDayAheadExportError("export bounds must lie on a 15-minute UTC grid")
    last_instant = end - pd.Timedelta(nanoseconds=1)
    start_index = start.year * 12 + start.month
    end_index = last_instant.year * 12 + last_instant.month
    if end_index - start_index not in {0, 1}:
        raise EntsoeDayAheadExportError("export window spans more than two UTC month partitions")
    return {
        "start_utc": _utc_text(start),
        "end_utc": _utc_text(end),
        "partition_start_year": start.year,
        "partition_start_month": start.month,
        "partition_end_year": last_instant.year,
        "partition_end_month": last_instant.month,
    }


def _replay_export(raw: pd.DataFrame, audit: Mapping[str, object]) -> DayAheadConsumerExport:
    if set(audit) != {
        "schema_version",
        "status",
        "usage",
        "query_sha256",
        "window_start_utc",
        "window_end_utc",
        "temporal_cutoff_utc",
        "series_selection",
        "market_uses",
        "raw_row_count",
        "consumer_row_count",
        "raw_frame_semantic_sha256",
        "consumer_frame_semantic_sha256",
        "evidence",
        "evidence_content_id",
        "evidence_authentication_policy",
        "lineage_policy",
        "authorities",
    }:
        raise EntsoeDayAheadExportError("export replay audit fields are not exact")
    if audit.get("schema_version") != EXPORT_SCHEMA:
        raise EntsoeDayAheadExportError("export replay audit schema is invalid")
    selection = audit.get("series_selection")
    market_uses = audit.get("market_uses")
    evidence_items = audit.get("evidence")
    if (
        not isinstance(selection, Mapping)
        or not isinstance(market_uses, Mapping)
        or not isinstance(evidence_items, list)
    ):
        raise EntsoeDayAheadExportError("export replay audit content is invalid")
    evidence: list[ScopedDayAheadEvidence] = []
    for item in evidence_items:
        if not isinstance(item, Mapping) or set(item) != {
            "kind",
            "authority",
            "evidence_id",
            "evidence_document_sha256",
            "covered_frame_semantic_sha256",
            "asserted_at_utc",
            "window_start_utc",
            "window_end_utc",
            "series_keys",
        }:
            raise EntsoeDayAheadExportError("export replay evidence fields are not exact")
        series_keys = item["series_keys"]
        if not isinstance(series_keys, list):
            raise EntsoeDayAheadExportError("export replay evidence scope is invalid")
        evidence.append(
            ScopedDayAheadEvidence(
                kind=str(item["kind"]),
                authority=str(item["authority"]),
                evidence_id=str(item["evidence_id"]),
                evidence_document_sha256=str(item["evidence_document_sha256"]),
                covered_frame_semantic_sha256=str(item["covered_frame_semantic_sha256"]),
                asserted_at_utc=str(item["asserted_at_utc"]),
                window_start_utc=str(item["window_start_utc"]),
                window_end_utc=str(item["window_end_utc"]),
                series_keys=tuple(str(value) for value in series_keys),
            )
        )
    try:
        usage = SpotUsage(str(audit.get("usage", "")))
    except ValueError as exc:
        raise EntsoeDayAheadExportError("export replay usage is invalid") from exc
    common = {
        "series_selection": {str(key): str(value) for key, value in selection.items()},
        "market_uses": {str(key): str(value) for key, value in market_uses.items()},
        "window_start_utc": str(audit.get("window_start_utc", "")),
        "window_end_utc": str(audit.get("window_end_utc", "")),
        "query_sha256": str(audit.get("query_sha256", "")),
    }
    if usage is SpotUsage.CAUSAL_ASOF:
        return validate_causal_export(
            raw,
            as_of_utc=str(audit.get("temporal_cutoff_utc", "")),
            publication_evidence=tuple(evidence),
            **common,
        )
    return validate_realized_export(
        raw,
        assessed_at_utc=str(audit.get("temporal_cutoff_utc", "")),
        finality_evidence=tuple(evidence),
        **common,
    )


def _read_replay_parquet(payload: bytes, *, columns: Sequence[str], label: str) -> pd.DataFrame:
    try:
        validate_parquet_allocation_budget(
            payload,
            label=f"day-ahead {label}",
            max_rows=RESULT_ROW_LIMIT - 1,
            max_columns=32,
            max_cells=(RESULT_ROW_LIMIT - 1) * 32,
            max_row_groups=256,
            allowed_physical_types=_PARQUET_PHYSICAL_TYPES,
        )
        frame = pd.read_parquet(BytesIO(payload))
    except (OSError, ValueError) as exc:
        raise EntsoeDayAheadExportError(f"export replay {label} is not readable Parquet") from exc
    if frame.empty or frame.columns.has_duplicates or tuple(frame.columns) != tuple(columns):
        raise EntsoeDayAheadExportError(f"export replay {label} schema is invalid")
    return frame


def _parquet_payload(frame: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _strict_json_mapping(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntsoeDayAheadExportError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict) or _canonical_json_bytes(value) != payload:
        raise EntsoeDayAheadExportError(f"{label} is not a canonical mapping")
    return value


def _file_sha256(path: Path, label: str) -> str:
    try:
        payload = read_stable_single_link_file(path, label=label, max_bytes=200_000)
    except (OSError, ValueError) as exc:
        raise EntsoeDayAheadExportError(f"{label} read failed") from exc
    return hashlib.sha256(payload).hexdigest()


def _expected_sha(value: object, expected: str, label: str) -> str:
    observed = _sha256(value, label)
    if observed != expected:
        raise EntsoeDayAheadExportError(f"{label} SHA-256 differs")
    return observed


def _sha256(value: object, label: str) -> str:
    text = _text(value, label)
    if not _SHA256.fullmatch(text):
        raise EntsoeDayAheadExportError(f"{label} must be lowercase SHA-256")
    return text


def _text(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or _CLEAN_TEXT.fullmatch(value) is None
    ):
        raise EntsoeDayAheadExportError(f"{label} must be clean text")
    return value


def _optional_text(value: object, label: str) -> str | None:
    if value is None or pd.isna(value):
        return None
    return _text(value, label)


def _utc_timestamp(value: object, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise EntsoeDayAheadExportError(f"{label} is invalid") from exc
    if timestamp.tzinfo is None:
        raise EntsoeDayAheadExportError(f"{label} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _utc_text(value: pd.Timestamp) -> str:
    return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _positive_int(value: object, label: str) -> int:
    normalized = _nonnegative_int(value, label)
    if normalized == 0:
        raise EntsoeDayAheadExportError(f"{label} must be positive")
    return normalized


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise EntsoeDayAheadExportError(f"{label} must be a nonnegative integer")
    return int(value)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


__all__ = [
    "CAUSAL_SQL_SHA256",
    "EXPORT_SCHEMA",
    "PREFLIGHT_SCHEMA",
    "REPLAY_SCHEMA",
    "RAW_COLUMNS",
    "REALIZED_SQL_SHA256",
    "DayAheadConsumerExport",
    "DayAheadExportCostAssessment",
    "DayAheadExportReplayPackage",
    "DayAheadMarketUse",
    "EntsoeDayAheadExportError",
    "EvidenceAuthority",
    "EvidenceKind",
    "ScopedDayAheadEvidence",
    "assess_export_cost_preflight",
    "build_export_replay_package",
    "build_causal_export_parameters",
    "build_realized_export_parameters",
    "validate_causal_export",
    "validate_realized_export",
    "verify_export_replay_package",
    "verify_export_sql_bindings",
]
