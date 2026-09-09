from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pfc_shaping.validation.entsoe_day_ahead_prd import SERIES_INVENTORY_SPEC
from pfc_shaping.validation.entsoe_day_ahead_temporal_diagnostic import (
    BLOCKED_SOURCE_STATUS,
    COLUMNS,
    SQL_PATH,
    SQL_SHA256,
    TemporalDiagnosticError,
    assess_temporal_diagnostic,
    verify_sql_binding,
)

START = "2026-07-01T00:00:00Z"
END = "2026-08-01T00:00:00Z"


def _series_key(field: str, classification: str | None) -> str:
    base = f"day_ahead_prices||{field}"
    return f"{base}||{classification}" if classification is not None else base


def _frame() -> pd.DataFrame:
    rows = []
    for position, (_, field, classification) in enumerate(SERIES_INVENTORY_SPEC, start=1):
        resolution = "PT60M" if field == "ch_price" else "PT15M"
        duration = 3600 if resolution == "PT60M" else 900
        rows.append(
            {
                "field_name": field,
                "series_key": _series_key(field, classification),
                "classification_sequence": classification,
                "resolution": resolution,
                "row_count": 100 + position,
                "publication_timestamp_null_count": 0,
                "first_seen_null_count": 0,
                "last_seen_null_count": 0,
                "publication_after_first_seen_count": 100 + position,
                "first_seen_after_last_seen_count": 0,
                "publication_after_delivery_start_count": 0,
                "invalid_availability_order_count": 100 + position,
                "publication_to_first_seen_min_seconds": -7200,
                "publication_to_first_seen_p50_seconds": -3600,
                "publication_to_first_seen_p95_seconds": -1800,
                "publication_to_first_seen_max_seconds": -900,
                "first_seen_to_last_seen_min_seconds": 0,
                "first_seen_to_last_seen_p50_seconds": 0,
                "first_seen_to_last_seen_p95_seconds": 60,
                "first_seen_to_last_seen_max_seconds": 120,
                "publication_to_delivery_min_seconds": 3600,
                "publication_to_delivery_p50_seconds": 7200,
                "publication_to_delivery_p95_seconds": 10_800,
                "publication_to_delivery_max_seconds": 14_400,
                "interval_start_null_count": 0,
                "interval_end_null_count": 0,
                "date_time_null_count": 0,
                "interval_end_datetime_mismatch_count": position,
                "interval_nonpositive_count": 0,
                "unsupported_resolution_count": 0,
                "duration_mismatch_count": 0,
                "invalid_interval_count": position,
                "interval_duration_min_seconds": duration,
                "interval_duration_p50_seconds": duration,
                "interval_duration_p95_seconds": duration,
                "interval_duration_max_seconds": duration,
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def _profile_counts(frame: pd.DataFrame | None = None) -> dict[str, dict[str, int]]:
    source = _frame() if frame is None else frame
    return {
        row.series_key: {
            "row_count": row.row_count,
            "invalid_availability_order_count": row.invalid_availability_order_count,
            "invalid_interval_count": row.invalid_interval_count,
        }
        for row in source.itertuples(index=False)
    }


def _assess(frame: pd.DataFrame | None = None):
    source = _frame() if frame is None else frame
    return assess_temporal_diagnostic(
        source,
        window_start_utc=START,
        window_end_utc=END,
        query_sha256=SQL_SHA256,
        expected_profile_counts=_profile_counts(source),
    )


def test_temporal_diagnostic_reconciles_causes_and_preserves_no_authority() -> None:
    report = _assess().as_dict()

    assert report["evidence_status"] == "PASS_RECONCILED_TEMPORAL_DIAGNOSTIC_EVIDENCE"
    assert report["source_quality_status"] == BLOCKED_SOURCE_STATUS
    assert report["metrics"]["profile_reconciliation"] == "EXACT"
    assert report["metrics"]["cause_totals"]["duration_mismatch_count"] == 0
    assert report["incident_interpretation"] == {
        "publication_delay_can_directly_explain_interval_structure": False,
        "publication_after_delivery_is_incident_compatible": True,
        "publication_after_first_seen_is_normal_late_publication_semantics": False,
        "causality_proven": False,
    }
    assert not any(report["authorities"].values())


def test_temporal_diagnostic_rejects_union_or_profile_reconciliation_drift() -> None:
    inconsistent = _frame()
    inconsistent.loc[0, "invalid_interval_count"] = 0
    with pytest.raises(TemporalDiagnosticError, match="union count"):
        _assess(inconsistent)

    source = _frame()
    counts = _profile_counts(source)
    counts[source.loc[0, "series_key"]]["row_count"] += 1
    with pytest.raises(TemporalDiagnosticError, match="does not reconcile"):
        assess_temporal_diagnostic(
            source,
            window_start_utc=START,
            window_end_utc=END,
            query_sha256=SQL_SHA256,
            expected_profile_counts=counts,
        )


def test_temporal_diagnostic_rejects_bad_quantiles_and_inventory() -> None:
    bad_quantiles = _frame()
    bad_quantiles.loc[0, "publication_to_first_seen_p50_seconds"] = -9000
    with pytest.raises(TemporalDiagnosticError, match="quantiles"):
        _assess(bad_quantiles)

    bad_inventory = _frame()
    bad_inventory.loc[0, "classification_sequence"] = "9"
    with pytest.raises(TemporalDiagnosticError, match="inventory differs"):
        _assess(bad_inventory)


def test_temporal_diagnostic_sql_is_hash_bound_read_only_value_blind_and_pruned() -> None:
    assert verify_sql_binding() == SQL_SHA256
    sql = Path(SQL_PATH).read_text(encoding="utf-8").lower()
    assert "v._year = p.delivery_year" in sql
    assert "v._month = p.delivery_month" in sql
    assert "limit 101" in sql
    assert "field_value" not in sql
    assert "price_eur_per_mwh" not in sql
