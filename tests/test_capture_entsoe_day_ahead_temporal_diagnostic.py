from __future__ import annotations

import pytest

from pfc_shaping.validation.entsoe_day_ahead_temporal_diagnostic import (
    COLUMNS,
    COUNT_COLUMNS,
    SIGNED_METRIC_COLUMNS,
)
from scripts.capture_entsoe_day_ahead_prd_profile import CaptureError
from scripts.capture_entsoe_day_ahead_temporal_diagnostic import _diagnostic_frame


def _response() -> dict[str, object]:
    row: dict[str, object] = {column: "0" for column in COLUMNS}
    row.update(
        {
            "field_name": "ch_price",
            "series_key": "day_ahead_prices||ch_price",
            "classification_sequence": None,
            "resolution": "PT60M",
            "row_count": "744",
            "publication_to_first_seen_min_seconds": "-7200",
            "publication_to_first_seen_p50_seconds": "-3600",
            "publication_to_first_seen_p95_seconds": "-1800",
            "publication_to_first_seen_max_seconds": "-900",
            "interval_duration_min_seconds": "3600",
            "interval_duration_p50_seconds": "3600",
            "interval_duration_p95_seconds": "3600",
            "interval_duration_max_seconds": "3600",
        }
    )
    return {
        "manifest": {
            "truncated": False,
            "schema": {"columns": [{"name": column} for column in COLUMNS]},
        },
        "result": {"data_array": [[row[column] for column in COLUMNS]]},
    }


def test_diagnostic_frame_coerces_counts_and_signed_lags() -> None:
    frame = _diagnostic_frame(_response())

    assert tuple(frame.columns) == COLUMNS
    assert frame.loc[0, "row_count"] == 744
    assert frame.loc[0, "publication_to_first_seen_min_seconds"] == -7200
    assert all(int(frame.loc[0, column]) >= 0 for column in COUNT_COLUMNS)
    assert all(
        frame.loc[0, column] is None or int(frame.loc[0, column]) == frame.loc[0, column]
        for column in SIGNED_METRIC_COLUMNS
    )


def test_diagnostic_frame_rejects_truncation_schema_drift_and_negative_counts() -> None:
    truncated = _response()
    truncated["manifest"]["truncated"] = True
    with pytest.raises(CaptureError, match="truncated"):
        _diagnostic_frame(truncated)

    drifted = _response()
    drifted["manifest"]["schema"]["columns"][0]["name"] = "price_eur_per_mwh"
    with pytest.raises(CaptureError, match="columns differ"):
        _diagnostic_frame(drifted)

    negative = _response()
    row_count_position = COLUMNS.index("row_count")
    negative["result"]["data_array"][0][row_count_position] = "-1"
    with pytest.raises(CaptureError, match="nonnegative"):
        _diagnostic_frame(negative)
