from __future__ import annotations

import json
from numbers import Integral

import pytest

from pfc_shaping.validation.entsoe_day_ahead_prd import (
    PROFILE_COLUMNS,
    PROFILE_RESULT_ROW_LIMIT,
    PROFILE_SQL_SHA256,
    assess_day_ahead_prd_profile,
    build_day_ahead_profile_parameters,
)
from scripts.capture_entsoe_day_ahead_prd_profile import (
    INTEGER_COLUMNS,
    CaptureError,
    _canonical_sha256,
    _inventory_binding,
    _profile_frame,
    _resolve_output,
    verify_capture_path,
)


def _response() -> dict[str, object]:
    row = {
        "field_name": "ch_price",
        "series_key": "day_ahead_prices||ch_price",
        "classification_sequence": None,
        "unit": "EUR/MWh",
        "document_type": "A44",
        "series_id": "9",
        "resolution": "PT60M",
        "vintage_row_count": "744",
        "distinct_interval_count": "744",
        "min_interval_start_utc": "2026-07-01T00:00:00.000Z",
        "max_interval_end_utc": "2026-08-01T00:00:00.000Z",
        "null_value_count": "0",
        "dq_failed_count": "0",
        "unknown_availability_count": "0",
        "invalid_availability_order_count": "0",
        "invalid_interval_count": "0",
        "canonical_series_key_mismatch_count": "0",
        "profile_row_count": "1",
        "duplicate_vintage_key_count": "0",
        "orphan_series_key_count": "0",
        "gold_series_key_duplicate_count": "0",
        "latest_grain_duplicate_count": "0",
        "legacy_new_overlap_interval_count": "0",
    }
    return {
        "manifest": {
            "truncated": False,
            "schema": {"columns": [{"name": column} for column in PROFILE_COLUMNS]},
        },
        "result": {"data_array": [[row[column] for column in PROFILE_COLUMNS]]},
    }


def test_profile_frame_coerces_only_governed_integer_columns() -> None:
    frame = _profile_frame(_response())

    assert tuple(frame.columns) == PROFILE_COLUMNS
    assert frame.loc[0, "series_id"] == 9
    assert frame.loc[0, "vintage_row_count"] == 744
    assert all(isinstance(frame.loc[0, column], Integral) for column in INTEGER_COLUMNS)
    assert frame.loc[0, "classification_sequence"] is None


def test_profile_frame_rejects_truncation_and_schema_drift() -> None:
    truncated = _response()
    truncated["manifest"]["truncated"] = True
    with pytest.raises(CaptureError, match="truncated"):
        _profile_frame(truncated)

    drifted = _response()
    drifted["manifest"]["schema"]["columns"][0]["name"] = "price_eur_per_mwh"
    with pytest.raises(CaptureError, match="columns differ"):
        _profile_frame(drifted)


def test_output_must_be_new_and_below_repo_local_build(tmp_path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()

    with pytest.raises(CaptureError, match="already exists"):
        _resolve_output(str(existing))


def test_persisted_capture_replays_and_detects_tampering(tmp_path) -> None:
    frame = _profile_frame(_response())
    parameters = build_day_ahead_profile_parameters(
        window_start_utc="2026-07-01T00:00:00Z",
        window_end_utc="2026-08-01T00:00:00Z",
    )
    assessed_at = "2026-08-02T00:00:00Z"
    assessment = assess_day_ahead_prd_profile(
        frame,
        window_start_utc=parameters["start_utc"],
        window_end_utc=parameters["end_utc"],
        assessed_at_utc=assessed_at,
        profile_query_sha256=PROFILE_SQL_SHA256,
        series_inventory_binding=_inventory_binding(),
        rebuild_evidence=None,
    ).as_dict()
    document = {
        "schema_version": "fmv_entsoe_day_ahead_prd_live_capture.v1",
        "captured_at_utc": assessed_at,
        "query": {
            "relative_path": "docs/data/sql/databricks_prd_entsoe_day_ahead_profile.sql",
            "sha256": PROFILE_SQL_SHA256,
            "parameters": parameters,
            "value_columns_opened": 0,
            "result_row_limit": PROFILE_RESULT_ROW_LIMIT,
        },
        "warehouse": {"state_before": "RUNNING"},
        "execution": {
            "statement_id": "statement-1",
            "statement_status": "SUCCEEDED",
            "control_plane_get_count": 1,
            "databricks_statement_count": 1,
            "warehouse_start_count": 0,
            "databricks_write_count": 0,
            "retry_count": 0,
            "wait_timeout_seconds": 50,
            "on_wait_timeout": "CANCEL",
            "elapsed_seconds": 1.0,
            "result_row_count": 1,
            "result_truncated": False,
        },
        "profile_rows": frame.where(frame.notna(), None).to_dict(orient="records"),
        "assessment": assessment,
        "authorities": {
            "bounded_pit_extraction": False,
            "model_input": False,
            "model_selection": False,
            "production": False,
        },
    }
    document["content_id"] = _canonical_sha256(document)
    capture_path = tmp_path / "capture.json"
    capture_path.write_text(json.dumps(document), encoding="utf-8")

    replay = verify_capture_path(capture_path)

    assert replay["status"] == "PASS_CAPTURE_REPLAY"
    assert replay["databricks_statement_count"] == 0
    document["execution"]["result_row_count"] = 2
    capture_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CaptureError, match="content ID differs"):
        verify_capture_path(capture_path)
