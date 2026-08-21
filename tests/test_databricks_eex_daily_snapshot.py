from __future__ import annotations

import pandas as pd
import pytest

from pfc_shaping.data.databricks_eex_daily_snapshot import (
    DatabricksEexDailyNormalizationError,
    EXPECTED_COLUMNS,
    SHORT_TENOR_CONTRACT_SEMANTICS,
    normalize_databricks_eex_daily_snapshot,
    select_latest_quote_surface,
    split_solver_quote_maps,
)


SOURCE_SHA256 = "a" * 64


def _row(
    *,
    product_id: str,
    delivery_period_id: str,
    quotation_date: str,
    settlement_price: float,
    product_type: str,
    delivery_period_type: str,
    delivery_start: str,
    delivery_end: str,
) -> dict[str, object]:
    return {
        "ProductID": product_id,
        "DeliveryPeriodID": delivery_period_id,
        "QuotationDateID": quotation_date,
        "SettlementPriceEurMWh": settlement_price,
        "LastPriceEurMWh": None,
        "FactLoadTimestampUtc": f"{quotation_date}T23:00:00Z",
        "Country": "CH",
        "Commodity": "POWER",
        "ProductType": product_type,
        "DeliveryPeriodType": delivery_period_type,
        "DeliveryStartDate": delivery_start,
        "DeliveryEndDate": delivery_end,
    }


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def test_normalizes_base_and_weekday_bounded_peak_products() -> None:
    rows = [
        _row(
            product_id="m-base",
            delivery_period_id="m1",
            quotation_date="2026-08-04",
            settlement_price=0.0,
            product_type="BASE",
            delivery_period_type="MONTH",
            delivery_start="2026-09-01",
            delivery_end="2026-09-30",
        ),
        _row(
            product_id="m-peak",
            delivery_period_id="m2",
            quotation_date="2026-08-04",
            settlement_price=-5.0,
            product_type="PEAK",
            delivery_period_type="MONTH",
            delivery_start="2026-09-01",
            delivery_end="2026-09-30",
        ),
        _row(
            product_id="q-peak",
            delivery_period_id="q1",
            quotation_date="2026-08-04",
            settlement_price=80.0,
            product_type="PEAK",
            delivery_period_type="QUARTER",
            delivery_start="2028-01-03",
            delivery_end="2028-03-31",
        ),
        _row(
            product_id="y-peak",
            delivery_period_id="y1",
            quotation_date="2026-08-04",
            settlement_price=82.0,
            product_type="PEAK",
            delivery_period_type="YEAR",
            delivery_start="2028-01-03",
            delivery_end="2028-12-29",
        ),
        _row(
            product_id="day",
            delivery_period_id="d1",
            quotation_date="2026-08-04",
            settlement_price=90.0,
            product_type="BASE",
            delivery_period_type="DAY",
            delivery_start="2026-08-05",
            delivery_end="2026-08-05",
        ),
    ]

    result = normalize_databricks_eex_daily_snapshot(
        _frame(rows),
        source_snapshot_sha256=SOURCE_SHA256,
    )

    assert list(result.history["product"]) == [
        "2026-09",
        "2026-09",
        "2028",
        "2028-Q1",
    ]
    assert list(result.history["load_type"]) == ["BASE", "PEAK", "PEAK", "PEAK"]
    assert set(result.history["product_type"].astype(str)) == {
        "Month",
        "Quarter",
        "Cal",
    }
    assert list(result.history["price"])[:2] == [0.0, -5.0]
    assert "D:2026-08-05" in set(result.all_products_history["product"].astype(str))
    assert not result.history["point_in_time_available_at_proven"].any()
    assert result.audit["non_solver_live_row_count"] == 1
    assert result.audit["quarantined_row_count"] == 0
    assert result.audit["authority"]["rolling_origin_selection_authorized"] is False


def test_rejects_peak_product_with_calendar_instead_of_weekday_bounds() -> None:
    frame = _frame(
        [
            _row(
                product_id="good-base",
                delivery_period_id="m0",
                quotation_date="2026-08-04",
                settlement_price=74.0,
                product_type="BASE",
                delivery_period_type="MONTH",
                delivery_start="2026-09-01",
                delivery_end="2026-09-30",
            ),
            _row(
                product_id="bad-peak",
                delivery_period_id="m1",
                quotation_date="2026-08-04",
                settlement_price=75.0,
                product_type="PEAK",
                delivery_period_type="MONTH",
                delivery_start="2026-10-01",
                delivery_end="2026-10-31",
            )
        ]
    )

    result = normalize_databricks_eex_daily_snapshot(
        frame,
        source_snapshot_sha256=SOURCE_SHA256,
    )

    assert list(result.history["product"]) == ["2026-09"]
    assert list(result.quarantine["ProductID"]) == ["bad-peak"]
    assert list(result.quarantine["quarantine_reason"]) == [
        "DELIVERY_BOUNDARY_MISMATCH"
    ]


def test_rejects_unknown_delivery_type_and_duplicate_source_key() -> None:
    valid = _row(
        product_id="base",
        delivery_period_id="m1",
        quotation_date="2026-08-04",
        settlement_price=75.0,
        product_type="BASE",
        delivery_period_type="MONTH",
        delivery_start="2026-09-01",
        delivery_end="2026-09-30",
    )
    unknown = dict(valid, DeliveryPeriodType="SEASON")
    with pytest.raises(DatabricksEexDailyNormalizationError, match="values are invalid"):
        normalize_databricks_eex_daily_snapshot(
            _frame([unknown]),
            source_snapshot_sha256=SOURCE_SHA256,
        )

    with pytest.raises(DatabricksEexDailyNormalizationError, match="source quote key"):
        normalize_databricks_eex_daily_snapshot(
            _frame([valid, dict(valid)]),
            source_snapshot_sha256=SOURCE_SHA256,
        )


def test_selects_one_observed_surface_without_fill_forward() -> None:
    older = _row(
        product_id="older",
        delivery_period_id="m1",
        quotation_date="2026-08-03",
        settlement_price=71.0,
        product_type="BASE",
        delivery_period_type="MONTH",
        delivery_start="2026-09-01",
        delivery_end="2026-09-30",
    )
    newer = _row(
        product_id="newer",
        delivery_period_id="m2",
        quotation_date="2026-08-04",
        settlement_price=72.0,
        product_type="BASE",
        delivery_period_type="MONTH",
        delivery_start="2026-10-01",
        delivery_end="2026-10-31",
    )
    result = normalize_databricks_eex_daily_snapshot(
        _frame([older, newer]),
        source_snapshot_sha256=SOURCE_SHA256,
    )

    assert list(result.latest_surface["product"]) == ["2026-10"]
    prior = select_latest_quote_surface(result.history, as_of_date="2026-08-03")
    assert list(prior["product"]) == ["2026-09"]
    with pytest.raises(DatabricksEexDailyNormalizationError, match="timezone-naive"):
        select_latest_quote_surface(
            result.history,
            as_of_date=pd.Timestamp("2026-08-04", tz="Europe/Zurich"),
        )


def test_split_solver_quote_maps_keeps_base_and_peak_explicit() -> None:
    frame = _frame(
        [
            _row(
                product_id="base",
                delivery_period_id="m1",
                quotation_date="2026-08-04",
                settlement_price=0.0,
                product_type="BASE",
                delivery_period_type="MONTH",
                delivery_start="2026-09-01",
                delivery_end="2026-09-30",
            ),
            _row(
                product_id="peak",
                delivery_period_id="m2",
                quotation_date="2026-08-04",
                settlement_price=-3.0,
                product_type="PEAK",
                delivery_period_type="MONTH",
                delivery_start="2026-09-01",
                delivery_end="2026-09-30",
            ),
        ]
    )
    result = normalize_databricks_eex_daily_snapshot(
        frame,
        source_snapshot_sha256=SOURCE_SHA256,
    )

    surface_before = result.latest_surface.copy(deep=True)
    maps = split_solver_quote_maps(result.latest_surface)

    assert maps == {
        "BASE": {"2026-09": 0.0},
        "PEAK": {"2026-09": -3.0},
    }
    pd.testing.assert_frame_equal(result.latest_surface, surface_before, check_exact=True)


def test_retains_day_week_and_weekend_in_a_separate_live_layer() -> None:
    common = {
        "quotation_date": "2026-08-04",
        "settlement_price": 70.0,
    }
    rows = [
        _row(
            product_id="solver-month",
            delivery_period_id="m1",
            product_type="BASE",
            delivery_period_type="MONTH",
            delivery_start="2026-09-01",
            delivery_end="2026-09-30",
            **common,
        ),
        _row(
            product_id="day-base",
            delivery_period_id="d1",
            product_type="BASE",
            delivery_period_type="DAY",
            delivery_start="2026-08-05",
            delivery_end="2026-08-05",
            **common,
        ),
        _row(
            product_id="week-base",
            delivery_period_id="w1",
            product_type="BASE",
            delivery_period_type="WEEK",
            delivery_start="2026-08-10",
            delivery_end="2026-08-16",
            **common,
        ),
        _row(
            product_id="weekend-day-peak",
            delivery_period_id="d2",
            product_type="PEAK",
            delivery_period_type="DAY",
            delivery_start="2026-08-15",
            delivery_end="2026-08-15",
            **common,
        ),
        _row(
            product_id="week-peak",
            delivery_period_id="w2",
            product_type="PEAK",
            delivery_period_type="WEEK",
            delivery_start="2026-08-10",
            delivery_end="2026-08-14",
            **common,
        ),
        _row(
            product_id="weekend-base",
            delivery_period_id="we1",
            product_type="BASE",
            delivery_period_type="WEEKEND",
            delivery_start="2026-08-15",
            delivery_end="2026-08-16",
            **common,
        ),
        _row(
            product_id="weekend-peak",
            delivery_period_id="we2",
            product_type="PEAK",
            delivery_period_type="WEEKEND",
            delivery_start="2026-08-15",
            delivery_end="2026-08-16",
            **common,
        ),
    ]

    result = normalize_databricks_eex_daily_snapshot(
        _frame(rows), source_snapshot_sha256=SOURCE_SHA256
    )

    assert set(result.all_products_history["product"].astype(str)) == {
        "2026-09",
        "D:2026-08-05",
        "D:2026-08-15",
        "W:2026-W33",
        "WE:2026-W33",
    }
    assert set(result.all_products_history["product_type"].astype(str)) == {
        "Day",
        "Week",
        "Weekend",
        "Month",
    }
    assert list(result.history["product"]) == ["2026-09"]
    assert len(result.latest_all_products_surface) == 7
    assert result.audit["short_tenor_contract_semantics"] == (
        SHORT_TENOR_CONTRACT_SEMANTICS
    )
    with pytest.raises(DatabricksEexDailyNormalizationError, match="values are invalid"):
        split_solver_quote_maps(result.latest_all_products_surface)


def test_quarantines_non_live_quotes_and_week_peak_boundary_anomaly() -> None:
    rows = [
        _row(
            product_id="solver-month",
            delivery_period_id="m1",
            quotation_date="2026-08-04",
            settlement_price=70.0,
            product_type="BASE",
            delivery_period_type="MONTH",
            delivery_start="2026-09-01",
            delivery_end="2026-09-30",
        ),
        _row(
            product_id="expired-day",
            delivery_period_id="d1",
            quotation_date="2026-08-05",
            settlement_price=71.0,
            product_type="BASE",
            delivery_period_type="DAY",
            delivery_start="2026-08-05",
            delivery_end="2026-08-05",
        ),
        _row(
            product_id="bad-week-peak",
            delivery_period_id="w1",
            quotation_date="2026-08-04",
            settlement_price=72.0,
            product_type="PEAK",
            delivery_period_type="WEEK",
            delivery_start="2026-08-10",
            delivery_end="2026-08-16",
        ),
    ]

    result = normalize_databricks_eex_daily_snapshot(
        _frame(rows), source_snapshot_sha256=SOURCE_SHA256
    )

    reasons = dict(
        zip(
            result.quarantine["ProductID"].astype(str),
            result.quarantine["quarantine_reason"].astype(str),
            strict=True,
        )
    )
    assert reasons == {
        "expired-day": "QUOTATION_NOT_BEFORE_DELIVERY_START",
        "bad-week-peak": "DELIVERY_BOUNDARY_MISMATCH",
    }
    assert result.audit["quarantined_row_count"] == 2
    assert result.audit["quarantined_not_before_delivery_count"] == 1
    assert result.audit["quarantined_boundary_mismatch_count"] == 1
