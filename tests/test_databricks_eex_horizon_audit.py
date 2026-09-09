from __future__ import annotations

import pandas as pd
import pytest

from pfc_shaping.validation.databricks_eex_horizon_audit import (
    DatabricksEexHorizonAuditError,
    EXPECTED_COLUMNS,
    HORIZON_BUCKETS,
    audit_databricks_eex_horizon_coverage,
)


SOURCE_SHA256 = "a" * 64
NORMALIZATION_ID = "b" * 64


def _row(
    *,
    date: str,
    product: str,
    product_type: str,
    load_type: str = "BASE",
    price: float = 80.0,
) -> dict[str, object]:
    token = f"{date}-{product}-{load_type}"
    return {
        "date": date,
        "product": product,
        "load_type": load_type,
        "product_type": product_type,
        "market": "CH",
        "price": price,
        "unit": "EUR/MWH",
        "source": "EEX",
        "source_system": "DATABRICKS_PRD_GOLD",
        "product_id": f"product-{product}-{load_type}",
        "delivery_period_id": f"period-{product}-{load_type}",
        "fact_load_timestamp_utc": f"{date}T23:00:00Z",
        "source_snapshot_sha256": SOURCE_SHA256,
        "point_in_time_available_at_proven": False,
        "rolling_origin_selection_authorized": False,
        "production_promotion_authorized": False,
    }


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def _audit(frame: pd.DataFrame):
    return audit_databricks_eex_horizon_coverage(
        frame,
        source_snapshot_sha256=SOURCE_SHA256,
        source_normalization_content_id=NORMALIZATION_ID,
    )


def test_assigns_exact_horizon_bucket_boundaries() -> None:
    delivery_start = pd.Timestamp("2027-01-01")
    leads = (1, 30, 31, 90, 91, 180, 181, 365, 366, 730, 731)
    frame = _frame(
        [
            _row(
                date=(delivery_start - pd.Timedelta(days=lead)).date().isoformat(),
                product="2027-01",
                product_type="Month",
            )
            for lead in leads
        ]
    )

    result = _audit(frame)
    observed = result.daily_horizon_coverage.loc[
        result.daily_horizon_coverage["quote_count"].gt(0),
        ["horizon_bucket", "quote_count"],
    ]
    counts = observed.groupby("horizon_bucket")["quote_count"].sum().to_dict()

    assert counts == {
        "D01_30": 2,
        "D31_90": 2,
        "D91_180": 2,
        "D181_365": 2,
        "D366_730": 2,
        "D731_PLUS": 1,
    }
    assert set(result.horizon_summary["horizon_bucket"]) == set(HORIZON_BUCKETS)


def test_peak_lead_uses_first_weekday_of_delivery_period() -> None:
    # 2027-05-01 is Saturday; the PEAK month starts Monday 2027-05-03.
    result = _audit(
        _frame(
            [
                _row(
                    date="2027-05-02",
                    product="2027-05",
                    product_type="Month",
                    load_type="PEAK",
                )
            ]
        )
    )

    lifecycle = result.product_lifecycles.iloc[0]
    assert lifecycle["delivery_start_date"] == pd.Timestamp("2027-05-03")
    assert lifecycle["last_quote_lead_days"] == 1


@pytest.mark.parametrize(
    ("load_type", "date"),
    [("BASE", "2027-05-01"), ("PEAK", "2027-05-03")],
)
def test_rejects_quote_at_or_after_delivery_start(
    load_type: str,
    date: str,
) -> None:
    frame = _frame(
        [
            _row(
                date=date,
                product="2027-05",
                product_type="Month",
                load_type=load_type,
            )
        ]
    )

    with pytest.raises(
        DatabricksEexHorizonAuditError,
        match="does not predate delivery start",
    ):
        _audit(frame)


def test_daily_grid_includes_zero_coverage_without_fill_forward() -> None:
    frame = _frame(
        [
            _row(
                date="2026-10-01",
                product="2027-01",
                product_type="Month",
            ),
            _row(
                date="2026-10-02",
                product="2027-Q2",
                product_type="Quarter",
            ),
        ]
    )

    result = _audit(frame)
    month_bucket = result.horizon_summary.loc[
        result.horizon_summary["product_type"].eq("Month")
        & result.horizon_summary["horizon_bucket"].eq("D91_180")
    ].iloc[0]

    assert month_bucket["active_quotation_date_count"] == 2
    assert month_bucket["date_count_with_quotes"] == 1
    assert month_bucket["date_share_with_quotes"] == pytest.approx(0.5)
    assert result.summary["interpretation"]["per_product_fill_forward_used"] is False


def test_product_lifecycle_reports_weekday_gap_proxy() -> None:
    frame = _frame(
        [
            _row(
                date="2026-01-05",
                product="2027",
                product_type="Cal",
            ),
            _row(
                date="2026-01-07",
                product="2027",
                product_type="Cal",
            ),
        ]
    )

    lifecycle = _audit(frame).product_lifecycles.iloc[0]

    assert lifecycle["quotation_date_count"] == 2
    assert lifecycle["max_calendar_gap_days"] == 2
    assert lifecycle["max_missing_weekdays_between_quotes_proxy"] == 1
    assert lifecycle["weekday_denominator_proxy"] == 3
    assert lifecycle["weekday_quotation_count"] == 2
    assert lifecycle["weekday_coverage_proxy"] == pytest.approx(2 / 3)
    assert lifecycle["long_gap_proxy_count"] == 0


def test_long_gap_is_reported_without_claiming_market_inactivity() -> None:
    frame = _frame(
        [
            _row(
                date="2026-01-05",
                product="2027-Q1",
                product_type="Quarter",
            ),
            _row(
                date="2026-03-02",
                product="2027-Q1",
                product_type="Quarter",
            ),
            _row(
                date="2026-03-03",
                product="2027-Q1",
                product_type="Quarter",
            ),
        ]
    )

    result = _audit(frame)
    lifecycle = result.product_lifecycles.iloc[0]

    assert lifecycle["long_gap_proxy_count"] == 1
    assert lifecycle["largest_gap_after_date"] == pd.Timestamp("2026-01-05")
    assert lifecycle["largest_gap_before_date"] == pd.Timestamp("2026-03-02")
    assert lifecycle["quotation_count_before_largest_gap"] == 1
    assert lifecycle["quotation_count_after_largest_gap"] == 2
    assert result.summary["long_gap_lifecycle_count"] == 1
    assert result.summary["status"] == (
        "PASS_LOCAL_DESCRIPTIVE_HORIZON_COVERAGE_WITH_GAP_ANOMALIES_NOT_PIT"
    )
    assert result.summary["interpretation"][
        "long_gap_proves_market_inactivity"
    ] is False


def test_summary_is_descriptive_only_and_prices_do_not_change_coverage() -> None:
    rows = [
        _row(
            date="2026-06-30",
            product="2027-01",
            product_type="Month",
            load_type="BASE",
            price=10.0,
        ),
        _row(
            date="2026-06-28",
            product="2027-01",
            product_type="Month",
            load_type="PEAK",
            price=500.0,
        ),
    ]
    first = _audit(_frame(rows))
    changed = [dict(row) for row in rows]
    changed[0]["price"] = -100.0
    changed[1]["price"] = 0.0
    second = _audit(_frame(changed))

    pd.testing.assert_frame_equal(
        first.daily_horizon_coverage,
        second.daily_horizon_coverage,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        first.product_lifecycles,
        second.product_lifecycles,
        check_exact=True,
    )
    assert first.summary == second.summary
    assert first.summary["authority"]["model_selection_authorized"] is False
    assert first.summary["interpretation"][
        "quotation_date_is_authenticated_availability"
    ] is False
    assert first.summary["interpretation"]["prices_used_in_coverage_metrics"] is False


def test_delayed_fact_load_is_reported_but_not_treated_as_availability() -> None:
    row = _row(
        date="2020-01-02",
        product="2021",
        product_type="Cal",
    )
    row["fact_load_timestamp_utc"] = "2026-08-05T02:42:47Z"

    summary = _audit(_frame([row])).summary
    timing = summary["lineage_timing"]

    assert timing["quotation_to_fact_load_lag_days_p50"] > 365
    assert timing["row_share_loaded_more_than_365_days_after_quotation"] == 1.0
    assert timing["delayed_load_backfill_or_restatement_signature"] is True
    assert timing["fact_load_timestamp_is_provider_availability"] is False
    assert timing["historical_pit_inference_blocked"] is True


def test_rejects_duplicate_identity_product_mismatch_and_authority_overclaim() -> None:
    frame = _frame(
        [
            _row(
                date="2026-10-01",
                product="2027-Q1",
                product_type="Quarter",
            )
        ]
    )
    with pytest.raises(DatabricksEexHorizonAuditError, match="quote identity"):
        _audit(pd.concat([frame, frame], ignore_index=True))

    mismatch = frame.copy()
    mismatch.loc[0, "product_type"] = "Cal"
    with pytest.raises(
        DatabricksEexHorizonAuditError,
        match="product/product_type mismatch",
    ):
        _audit(mismatch)

    overclaim = frame.copy()
    overclaim.loc[0, "point_in_time_available_at_proven"] = True
    with pytest.raises(DatabricksEexHorizonAuditError, match="overclaims"):
        _audit(overclaim)
