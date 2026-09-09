"""Offline validation for a value-blind ENTSO-E temporal root-cause profile."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import pandas as pd

from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.validation.entsoe_day_ahead_prd import (
    EXPECTED_SERIES_IDENTITIES,
    PROFILE_RESULT_ROW_LIMIT,
    build_day_ahead_profile_parameters,
)

ROOT = Path(__file__).resolve().parents[2]
SQL_PATH = ROOT / "docs/data/sql/databricks_prd_entsoe_day_ahead_temporal_diagnostic.sql"
SQL_SHA256 = "f20b0ac5681404834132897d48b491532ae226a7cec81a76abae7043721e5f04"
SCHEMA_VERSION = "fmv_entsoe_day_ahead_temporal_diagnostic.v1"
EVIDENCE_STATUS = "PASS_RECONCILED_TEMPORAL_DIAGNOSTIC_EVIDENCE"
BLOCKED_SOURCE_STATUS = "BLOCKED_TEMPORAL_SOURCE_QUALITY"
PASS_SOURCE_STATUS = "PASS_TEMPORAL_SOURCE_QUALITY"

IDENTITY_COLUMNS = (
    "field_name",
    "series_key",
    "classification_sequence",
    "resolution",
)
AVAILABILITY_CAUSE_COLUMNS = (
    "publication_timestamp_null_count",
    "first_seen_null_count",
    "last_seen_null_count",
    "publication_after_first_seen_count",
    "first_seen_after_last_seen_count",
)
INCIDENT_CONTEXT_COLUMNS = ("publication_after_delivery_start_count",)
AVAILABILITY_UNION_COLUMN = "invalid_availability_order_count"
AVAILABILITY_LAG_COLUMNS = (
    "publication_to_first_seen_min_seconds",
    "publication_to_first_seen_p50_seconds",
    "publication_to_first_seen_p95_seconds",
    "publication_to_first_seen_max_seconds",
    "first_seen_to_last_seen_min_seconds",
    "first_seen_to_last_seen_p50_seconds",
    "first_seen_to_last_seen_p95_seconds",
    "first_seen_to_last_seen_max_seconds",
    "publication_to_delivery_min_seconds",
    "publication_to_delivery_p50_seconds",
    "publication_to_delivery_p95_seconds",
    "publication_to_delivery_max_seconds",
)
INTERVAL_CAUSE_COLUMNS = (
    "interval_start_null_count",
    "interval_end_null_count",
    "date_time_null_count",
    "interval_end_datetime_mismatch_count",
    "interval_nonpositive_count",
    "unsupported_resolution_count",
    "duration_mismatch_count",
)
INTERVAL_UNION_COLUMN = "invalid_interval_count"
INTERVAL_DURATION_COLUMNS = (
    "interval_duration_min_seconds",
    "interval_duration_p50_seconds",
    "interval_duration_p95_seconds",
    "interval_duration_max_seconds",
)
COUNT_COLUMNS = (
    "row_count",
    *AVAILABILITY_CAUSE_COLUMNS,
    *INCIDENT_CONTEXT_COLUMNS,
    AVAILABILITY_UNION_COLUMN,
    *INTERVAL_CAUSE_COLUMNS,
    INTERVAL_UNION_COLUMN,
)
SIGNED_METRIC_COLUMNS = (*AVAILABILITY_LAG_COLUMNS, *INTERVAL_DURATION_COLUMNS)
COLUMNS = (
    *IDENTITY_COLUMNS,
    "row_count",
    *AVAILABILITY_CAUSE_COLUMNS,
    *INCIDENT_CONTEXT_COLUMNS,
    AVAILABILITY_UNION_COLUMN,
    *AVAILABILITY_LAG_COLUMNS,
    *INTERVAL_CAUSE_COLUMNS,
    INTERVAL_UNION_COLUMN,
    *INTERVAL_DURATION_COLUMNS,
)
EXPECTED_PROFILE_COUNT_FIELDS = frozenset(
    {"row_count", AVAILABILITY_UNION_COLUMN, INTERVAL_UNION_COLUMN}
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RESOLUTIONS = frozenset({"PT15M", "PT30M", "PT60M", "PT1H"})


class TemporalDiagnosticError(ValueError):
    """Raised when temporal diagnostic evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class TemporalDiagnosticReport:
    """Reconciled causal counts with no extraction or model authority."""

    source_quality_status: str
    findings: tuple[dict[str, object], ...]
    metrics: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "evidence_status": EVIDENCE_STATUS,
            "source_quality_status": self.source_quality_status,
            "findings": [dict(item) for item in self.findings],
            "metrics": dict(self.metrics),
            "incident_interpretation": {
                "publication_delay_can_directly_explain_interval_structure": False,
                "publication_after_delivery_is_incident_compatible": True,
                "publication_after_first_seen_is_normal_late_publication_semantics": False,
                "causality_proven": False,
            },
            "authorities": {
                "bounded_pit_extraction_authorized": False,
                "model_input_authorized": False,
                "model_selection_authorized": False,
                "production_authorized": False,
            },
        }


def verify_sql_binding() -> str:
    payload = read_stable_single_link_file(
        SQL_PATH,
        label="day-ahead temporal diagnostic SQL",
        max_bytes=200_000,
    )
    observed = hashlib.sha256(payload).hexdigest()
    if observed != SQL_SHA256:
        raise TemporalDiagnosticError("Temporal diagnostic SQL binding differs")
    sql = payload.decode("utf-8").lower()
    if re.search(r"\b(insert|update|delete|merge|create|alter|drop|truncate|optimize)\b", sql):
        raise TemporalDiagnosticError("Temporal diagnostic SQL is not read-only")
    if "field_value" in sql or "price_eur_per_mwh" in sql:
        raise TemporalDiagnosticError("Temporal diagnostic SQL opens a business value")
    return observed


def assess_temporal_diagnostic(
    frame: pd.DataFrame,
    *,
    window_start_utc: str | pd.Timestamp,
    window_end_utc: str | pd.Timestamp,
    query_sha256: str,
    expected_profile_counts: Mapping[str, Mapping[str, object]],
) -> TemporalDiagnosticReport:
    """Validate and reconcile one bounded root-cause result."""

    verify_sql_binding()
    if not isinstance(query_sha256, str) or not _SHA256.fullmatch(query_sha256):
        raise TemporalDiagnosticError("Temporal diagnostic query SHA-256 is invalid")
    if query_sha256 != SQL_SHA256:
        raise TemporalDiagnosticError("Temporal diagnostic query SHA-256 differs")
    parameters = build_day_ahead_profile_parameters(
        window_start_utc=window_start_utc,
        window_end_utc=window_end_utc,
    )
    if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != COLUMNS:
        raise TemporalDiagnosticError("Temporal diagnostic columns differ")
    if frame.empty or len(frame) >= PROFILE_RESULT_ROW_LIMIT:
        raise TemporalDiagnosticError("Temporal diagnostic is empty or hit its row sentinel")

    normalized = frame.copy()
    for column in ("field_name", "series_key", "resolution"):
        normalized[column] = normalized[column].map(lambda value: _text(value, column))
    normalized["classification_sequence"] = normalized["classification_sequence"].map(
        lambda value: _optional_text(value, "classification_sequence")
    )
    for column in COUNT_COLUMNS:
        normalized[column] = normalized[column].map(
            lambda value, name=column: _nonnegative_int(value, name)
        )
    for column in SIGNED_METRIC_COLUMNS:
        normalized[column] = normalized[column].map(
            lambda value, name=column: _optional_int(value, name)
        )
    if normalized.duplicated(list(IDENTITY_COLUMNS)).any():
        raise TemporalDiagnosticError("Temporal diagnostic grain is duplicated")
    identities = set(
        zip(
            normalized["field_name"],
            normalized["classification_sequence"],
            strict=True,
        )
    )
    if identities != EXPECTED_SERIES_IDENTITIES:
        raise TemporalDiagnosticError("Temporal diagnostic series inventory differs")
    for row in normalized.itertuples(index=False):
        expected_key = f"day_ahead_prices||{row.field_name}"
        if row.classification_sequence is not None:
            expected_key += f"||{row.classification_sequence}"
        if row.series_key != expected_key:
            raise TemporalDiagnosticError("Temporal diagnostic SeriesKey is noncanonical")
        if row.resolution not in _RESOLUTIONS:
            raise TemporalDiagnosticError("Temporal diagnostic resolution is unsupported")
        _validate_cause_union(row, AVAILABILITY_CAUSE_COLUMNS, AVAILABILITY_UNION_COLUMN)
        _validate_cause_union(row, INTERVAL_CAUSE_COLUMNS, INTERVAL_UNION_COLUMN)
        _validate_ordered_metrics(row, AVAILABILITY_LAG_COLUMNS[0:4])
        _validate_ordered_metrics(row, AVAILABILITY_LAG_COLUMNS[4:8])
        _validate_ordered_metrics(row, AVAILABILITY_LAG_COLUMNS[8:12])
        _validate_ordered_metrics(row, INTERVAL_DURATION_COLUMNS)

    expected = _expected_profile_counts(expected_profile_counts)
    observed_keys = set(normalized["series_key"])
    if set(expected) != observed_keys:
        raise TemporalDiagnosticError("Profile reconciliation SeriesKeys differ")
    for row in normalized.itertuples(index=False):
        binding = expected[row.series_key]
        observed = {
            "row_count": row.row_count,
            AVAILABILITY_UNION_COLUMN: row.invalid_availability_order_count,
            INTERVAL_UNION_COLUMN: row.invalid_interval_count,
        }
        if observed != binding:
            raise TemporalDiagnosticError(
                f"Temporal diagnostic does not reconcile with profile: {row.series_key}"
            )

    cause_totals = {
        column: int(normalized[column].sum())
        for column in (
            *AVAILABILITY_CAUSE_COLUMNS,
            *INCIDENT_CONTEXT_COLUMNS,
            AVAILABILITY_UNION_COLUMN,
            *INTERVAL_CAUSE_COLUMNS,
            INTERVAL_UNION_COLUMN,
        )
    }
    total_rows = int(normalized["row_count"].sum())
    findings = tuple(
        {
            "severity": "CRITICAL"
            if column
            in {
                AVAILABILITY_UNION_COLUMN,
                INTERVAL_UNION_COLUMN,
            }
            else "HIGH",
            "code": f"TEMPORAL_{column.upper()}",
            "affected_count": count,
            "affected_rate": count / total_rows,
        }
        for column, count in cause_totals.items()
        if count
    )
    blocked = bool(cause_totals[AVAILABILITY_UNION_COLUMN] or cause_totals[INTERVAL_UNION_COLUMN])
    per_series = {
        row.series_key: {
            column: getattr(row, column) for column in (*COUNT_COLUMNS, *SIGNED_METRIC_COLUMNS)
        }
        for row in normalized.itertuples(index=False)
    }
    return TemporalDiagnosticReport(
        source_quality_status=BLOCKED_SOURCE_STATUS if blocked else PASS_SOURCE_STATUS,
        findings=findings,
        metrics={
            "query_sha256": SQL_SHA256,
            "window_parameters": parameters,
            "row_count": len(normalized),
            "source_row_count": total_rows,
            "cause_totals": cause_totals,
            "cause_rates": {column: count / total_rows for column, count in cause_totals.items()},
            "series": per_series,
            "profile_reconciliation": "EXACT",
        },
    )


def _validate_cause_union(row: object, causes: tuple[str, ...], union: str) -> None:
    row_count = int(getattr(row, "row_count"))
    cause_counts = [int(getattr(row, column)) for column in causes]
    union_count = int(getattr(row, union))
    if any(count > row_count for count in cause_counts) or union_count > row_count:
        raise TemporalDiagnosticError("Temporal cause count exceeds its series row count")
    if union_count < max(cause_counts, default=0) or union_count > sum(cause_counts):
        raise TemporalDiagnosticError("Temporal union count is inconsistent with causes")


def _validate_ordered_metrics(row: object, columns: tuple[str, ...]) -> None:
    values = [getattr(row, column) for column in columns]
    present = [value for value in values if value is not None]
    if present and (len(present) != len(values) or present != sorted(present)):
        raise TemporalDiagnosticError("Temporal lag/duration quantiles are inconsistent")


def _expected_profile_counts(
    value: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, int]]:
    if not isinstance(value, Mapping) or not value:
        raise TemporalDiagnosticError("Expected profile counts are required")
    result: dict[str, dict[str, int]] = {}
    for key, counts in value.items():
        series_key = _text(key, "profile SeriesKey")
        if not isinstance(counts, Mapping) or set(counts) != EXPECTED_PROFILE_COUNT_FIELDS:
            raise TemporalDiagnosticError("Expected profile count fields differ")
        result[series_key] = {
            name: _nonnegative_int(counts[name], name) for name in EXPECTED_PROFILE_COUNT_FIELDS
        }
    return result


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise TemporalDiagnosticError(f"{label} must be a nonnegative integer")
    return int(value)


def _optional_int(value: object, label: str) -> int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TemporalDiagnosticError(f"{label} must be an integer or null")
    return int(value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TemporalDiagnosticError(f"{label} must be clean text")
    return value


def _optional_text(value: object, label: str) -> str | None:
    if value is None or pd.isna(value):
        return None
    return _text(value, label)


__all__ = [
    "AVAILABILITY_CAUSE_COLUMNS",
    "AVAILABILITY_LAG_COLUMNS",
    "AVAILABILITY_UNION_COLUMN",
    "BLOCKED_SOURCE_STATUS",
    "COLUMNS",
    "COUNT_COLUMNS",
    "EVIDENCE_STATUS",
    "INCIDENT_CONTEXT_COLUMNS",
    "INTERVAL_CAUSE_COLUMNS",
    "INTERVAL_DURATION_COLUMNS",
    "INTERVAL_UNION_COLUMN",
    "SIGNED_METRIC_COLUMNS",
    "SQL_PATH",
    "SQL_SHA256",
    "TemporalDiagnosticError",
    "TemporalDiagnosticReport",
    "assess_temporal_diagnostic",
    "verify_sql_binding",
]
