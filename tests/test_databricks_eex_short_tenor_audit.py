from __future__ import annotations

import pandas as pd
import pytest

from pfc_shaping.validation.databricks_eex_short_tenor_audit import (
    DatabricksEexShortTenorAuditError,
    EXPECTED_COLUMNS,
    audit_databricks_eex_short_tenors,
)


SOURCE_SHA256 = "a" * 64
NORMALIZATION_ID = "b" * 64


def _row(
    *,
    date: str,
    product: str,
    product_type: str,
    load_type: str,
    price: float,
    product_id: str | None = None,
    delivery_period_id: str | None = None,
) -> dict[str, object]:
    token = f"{product}-{load_type}"
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
        "product_id": product_id or f"product-{token}",
        "delivery_period_id": delivery_period_id or f"period-{token}",
        "fact_load_timestamp_utc": f"{date}T23:00:00Z",
        "source_snapshot_sha256": SOURCE_SHA256,
        "point_in_time_available_at_proven": False,
        "rolling_origin_selection_authorized": False,
        "production_promotion_authorized": False,
    }


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def _complete_week() -> pd.DataFrame:
    quote_date = "2026-10-01"
    delivery_dates = pd.date_range("2026-10-26", periods=7, freq="D")
    base_prices = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    peak_prices = [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0]
    rows: list[dict[str, object]] = []
    for delivery_date, base_price, peak_price in zip(
        delivery_dates,
        base_prices,
        peak_prices,
        strict=True,
    ):
        product = f"D:{delivery_date.date().isoformat()}"
        rows.extend(
            [
                _row(
                    date=quote_date,
                    product=product,
                    product_type="Day",
                    load_type="BASE",
                    price=base_price,
                ),
                _row(
                    date=quote_date,
                    product=product,
                    product_type="Day",
                    load_type="PEAK",
                    price=peak_price,
                ),
            ]
        )
    rows.extend(
        [
            _row(
                date=quote_date,
                product="W:2026-W44",
                product_type="Week",
                load_type="BASE",
                price=sum(base_prices) / 7,
            ),
            _row(
                date=quote_date,
                product="W:2026-W44",
                product_type="Week",
                load_type="PEAK",
                price=sum(peak_prices[:5]) / 5,
            ),
            _row(
                date=quote_date,
                product="WE:2026-W44",
                product_type="Weekend",
                load_type="BASE",
                price=sum(base_prices[-2:]) / 2,
            ),
            _row(
                date=quote_date,
                product="WE:2026-W44",
                product_type="Weekend",
                load_type="PEAK",
                price=sum(peak_prices[-2:]) / 2,
            ),
        ]
    )
    return _frame(rows)


def _audit(frame: pd.DataFrame, *, quarantine: pd.DataFrame | None = None):
    return audit_databricks_eex_short_tenors(
        frame,
        source_snapshot_sha256=SOURCE_SHA256,
        source_normalization_content_id=NORMALIZATION_ID,
        quarantine=quarantine,
    )


def _quarantine_row() -> dict[str, object]:
    return {
        "ProductID": "quarantined-product",
        "DeliveryPeriodID": "same-period",
        "QuotationDateID": "2026-08-04",
        "SettlementPriceEurMWh": 10.5,
        "LastPriceEurMWh": None,
        "FactLoadTimestampUtc": "2026-08-04T23:00:00Z",
        "Country": "CH",
        "Commodity": "POWER",
        "ProductType": "PEAK",
        "DeliveryPeriodType": "WEEK",
        "DeliveryStartDate": "2026-08-10",
        "DeliveryEndDate": "2026-08-14",
        "canonical_product": "W:2026-W33",
        "quarantine_reason": "DELIVERY_BOUNDARY_MISMATCH",
    }


def test_audits_complete_week_weekend_day_strips_and_pairing() -> None:
    result = _audit(_complete_week())

    assert len(result.nesting_diagnostics) == 4
    assert set(result.nesting_diagnostics["status"]) == {"CONSISTENT"}
    assert len(result.offpeak_diagnostics) == 9
    assert set(result.offpeak_diagnostics["status"]) == {"PASS"}
    assert result.summary["base_peak_pair_count"] == 9
    assert result.summary["peak_only_date_product_count"] == 0
    assert result.summary["nesting_conflict_count"] == 0
    assert result.summary["authority"]["model_selection_authorized"] is False
    assert len(result.daily_coverage) == 3
    assert result.product_lifecycle["observation_count"].eq(1).all()


def test_classifies_same_vintage_parent_strip_difference_as_quote_conflict() -> None:
    frame = _complete_week()
    mask = frame["product"].eq("W:2026-W44") & frame["load_type"].eq("BASE")
    frame.loc[mask, "price"] += 1.0

    result = _audit(frame)

    conflicts = result.nesting_diagnostics.loc[
        result.nesting_diagnostics["status"].eq("QUOTE_CONFLICT")
    ]
    assert len(conflicts) == 1
    assert conflicts.iloc[0]["parent_product_type"] == "Week"
    assert conflicts.iloc[0]["load_type"] == "BASE"
    assert conflicts.iloc[0]["abs_residual_eur_mwh"] == pytest.approx(1.0)
    assert result.summary["status"] == (
        "PASS_LOCAL_INTEGRITY_WITH_MARKET_QUOTE_CONFLICTS"
    )
    assert result.summary["interpretation"][
        "quote_conflict_is_source_corruption"
    ] is False


def test_incomplete_day_strip_is_not_filled_or_reported_as_conflict() -> None:
    frame = _complete_week()
    missing = frame["product"].eq("D:2026-10-28") & frame["load_type"].eq("BASE")
    result = _audit(frame.loc[~missing].reset_index(drop=True))

    assert len(result.nesting_diagnostics) == 3
    assert set(result.nesting_diagnostics["status"]) == {"CONSISTENT"}
    assert result.summary["peak_only_date_product_count"] == 1
    assert result.summary["interpretation"]["per_product_fill_forward_used"] is False


def test_uses_europe_zurich_dst_hours_for_day_and_weekend() -> None:
    quote_date = "2026-03-20"
    frame = _frame(
        [
            _row(
                date=quote_date,
                product="D:2026-03-28",
                product_type="Day",
                load_type=load_type,
                price=10.0,
            )
            for load_type in ("BASE", "PEAK")
        ]
        + [
            _row(
                date=quote_date,
                product="D:2026-03-29",
                product_type="Day",
                load_type=load_type,
                price=10.0,
            )
            for load_type in ("BASE", "PEAK")
        ]
        + [
            _row(
                date=quote_date,
                product="WE:2026-W13",
                product_type="Weekend",
                load_type=load_type,
                price=10.0,
            )
            for load_type in ("BASE", "PEAK")
        ]
    )

    result = _audit(frame)

    offpeak = result.offpeak_diagnostics.set_index("product")
    assert offpeak.loc["D:2026-03-28", "total_hours"] == 24
    assert offpeak.loc["D:2026-03-29", "total_hours"] == 23
    assert offpeak.loc["WE:2026-W13", "total_hours"] == 47
    assert offpeak.loc["WE:2026-W13", "peak_hours"] == 24
    assert len(result.nesting_diagnostics) == 2
    assert set(result.nesting_diagnostics["status"]) == {"CONSISTENT"}


def test_allows_base_only_history_without_manufacturing_peak_pairs() -> None:
    frame = _frame(
        [
            _row(
                date="2026-08-01",
                product="D:2026-08-03",
                product_type="Day",
                load_type="BASE",
                price=10.0,
            )
        ]
    )

    result = _audit(frame)

    assert result.offpeak_diagnostics.empty
    assert result.summary["base_peak_pair_count"] == 0
    assert result.summary["base_only_date_product_count"] == 1
    assert result.summary["first_peak_quotation_date"] is None


def test_rejects_non_live_duplicate_mistyped_and_authority_overclaim() -> None:
    frame = _complete_week()
    with pytest.raises(DatabricksEexShortTenorAuditError, match="quote identity"):
        _audit(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))

    non_live = frame.iloc[[0]].copy()
    non_live.loc[:, "date"] = "2026-10-26"
    non_live.loc[:, "fact_load_timestamp_utc"] = "2026-10-26T23:00:00Z"
    with pytest.raises(DatabricksEexShortTenorAuditError, match="non-live"):
        _audit(non_live)

    mistyped = frame.copy()
    mistyped.loc[0, "product_type"] = "Week"
    with pytest.raises(DatabricksEexShortTenorAuditError, match="does not match"):
        _audit(mistyped)

    overclaim = frame.copy()
    overclaim.loc[0, "rolling_origin_selection_authorized"] = True
    with pytest.raises(DatabricksEexShortTenorAuditError, match="overclaims"):
        _audit(overclaim)


def test_rejects_one_identity_mapped_to_multiple_delivery_periods() -> None:
    first = _row(
        date="2026-08-01",
        product="D:2026-08-03",
        product_type="Day",
        load_type="BASE",
        price=10.0,
        delivery_period_id="period-a",
    )
    second = _row(
        date="2026-08-02",
        product="D:2026-08-03",
        product_type="Day",
        load_type="BASE",
        price=11.0,
        delivery_period_id="period-b",
    )

    with pytest.raises(DatabricksEexShortTenorAuditError, match="multiple delivery"):
        _audit(_frame([first, second]))


def test_reports_product_id_reuse_without_rejecting_valid_quote_identity() -> None:
    frame = _frame(
        [
            _row(
                date="2026-08-01",
                product="D:2026-08-03",
                product_type="Day",
                load_type="BASE",
                price=10.0,
                product_id="reused-id",
            ),
            _row(
                date="2026-08-01",
                product="D:2026-08-04",
                product_type="Day",
                load_type="BASE",
                price=11.0,
                product_id="reused-id",
            ),
        ]
    )

    result = _audit(frame)

    assert result.summary["mapping_diagnostics"][
        "product_id_reused_across_identity_count"
    ] == 1


def test_standard_eex_holidays_are_not_counted_as_candidate_exchange_gaps() -> None:
    shared = {
        "product": "D:2026-04-10",
        "product_type": "Day",
        "load_type": "BASE",
        "delivery_period_id": "same-period",
    }
    frame = _frame(
        [
            _row(date="2026-04-02", price=10.0, **shared),
            _row(date="2026-04-07", price=11.0, **shared),
        ]
    )

    result = _audit(frame)

    lifecycle = result.product_lifecycle.iloc[0]
    assert lifecycle["intervening_weekday_count_without_exchange_calendar"] == 2
    assert lifecycle["intervening_candidate_exchange_day_count"] == 0
    assert result.summary["public_holiday_calendar"]["as_of"] == "2025-06-19"


def test_reports_localized_open_exchange_day_gap_without_filling_it() -> None:
    shared = {
        "product": "D:2026-08-10",
        "product_type": "Day",
        "load_type": "BASE",
        "delivery_period_id": "same-period",
    }
    frame = _frame(
        [
            _row(date="2026-08-03", price=10.0, **shared),
            _row(date="2026-08-05", price=11.0, **shared),
        ]
    )

    result = _audit(frame)

    assert len(result.gap_diagnostics) == 1
    assert result.gap_diagnostics.loc[0, "missing_quotation_date"] == pd.Timestamp(
        "2026-08-04"
    )
    assert result.gap_diagnostics.loc[0, "classification"] == (
        "CANDIDATE_SOURCE_WIDE_GAP"
    )
    assert result.summary["status"] == (
        "PASS_LOCAL_INTEGRITY_WITH_LOCALIZED_TEMPORAL_GAPS"
    )
    assert result.summary["interpretation"]["per_product_fill_forward_used"] is False


def test_reconciles_gap_with_reason_coded_normalization_quarantine() -> None:
    shared = {
        "product": "W:2026-W33",
        "product_type": "Week",
        "load_type": "PEAK",
        "delivery_period_id": "same-period",
    }
    frame = _frame(
        [
            _row(date="2026-08-03", price=10.0, **shared),
            _row(date="2026-08-05", price=11.0, **shared),
        ]
    )
    quarantine = pd.DataFrame([_quarantine_row()])

    result = _audit(frame, quarantine=quarantine)

    assert result.gap_diagnostics.loc[0, "classification"] == (
        "EXPLAINED_NORMALIZATION_QUARANTINE"
    )
    assert result.summary["explained_normalization_quarantine_gap_count"] == 1
    assert result.summary["unexplained_candidate_gap_count"] == 0
    assert result.summary["status"] == (
        "PASS_LOCAL_INTEGRITY_NO_MARKET_QUOTE_CONFLICTS"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("Country", "DE", "Country values are invalid"),
        ("Commodity", "GAS", "Commodity values are invalid"),
        ("ProductType", "SHOULDER", "ProductType values are invalid"),
        ("quarantine_reason", "UNKNOWN_REASON", "reason is invalid"),
        (
            "FactLoadTimestampUtc",
            "2026-08-03T23:00:00Z",
            "quotation date follows",
        ),
        ("SettlementPriceEurMWh", float("nan"), "settlement price is invalid"),
    ],
)
def test_rejects_untrusted_quarantine_reconciliation_metadata(
    field: str,
    value: object,
    message: str,
) -> None:
    frame = _frame(
        [
            _row(
                date="2026-08-03",
                product="W:2026-W33",
                product_type="Week",
                load_type="PEAK",
                price=10.0,
                delivery_period_id="same-period",
            ),
            _row(
                date="2026-08-05",
                product="W:2026-W33",
                product_type="Week",
                load_type="PEAK",
                price=11.0,
                delivery_period_id="same-period",
            ),
        ]
    )
    quarantine_row = _quarantine_row()
    quarantine_row[field] = value

    with pytest.raises(DatabricksEexShortTenorAuditError, match=message):
        _audit(frame, quarantine=pd.DataFrame([quarantine_row]))


def test_rejects_inconsistent_non_boundary_quarantine_delivery_metadata() -> None:
    frame = _frame(
        [
            _row(
                date="2026-08-03",
                product="W:2026-W33",
                product_type="Week",
                load_type="PEAK",
                price=10.0,
                delivery_period_id="same-period",
            ),
            _row(
                date="2026-08-05",
                product="W:2026-W33",
                product_type="Week",
                load_type="PEAK",
                price=11.0,
                delivery_period_id="same-period",
            ),
        ]
    )
    quarantine_row = _quarantine_row()
    quarantine_row["quarantine_reason"] = "QUOTATION_NOT_BEFORE_DELIVERY_START"
    quarantine_row["DeliveryEndDate"] = "2026-08-16"

    with pytest.raises(
        DatabricksEexShortTenorAuditError,
        match="delivery metadata is inconsistent",
    ):
        _audit(frame, quarantine=pd.DataFrame([quarantine_row]))
