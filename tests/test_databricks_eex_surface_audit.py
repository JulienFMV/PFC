from __future__ import annotations

import pandas as pd
import pytest

from pfc_shaping.calibration.cascading import count_hours
from pfc_shaping.validation.databricks_eex_surface_audit import (
    DatabricksEexSurfaceAuditError,
    EXPECTED_COLUMNS,
    audit_databricks_eex_surfaces,
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
        "product_id": f"product-{token}",
        "delivery_period_id": f"period-{token}",
        "fact_load_timestamp_utc": f"{date}T23:00:00Z",
        "source_snapshot_sha256": SOURCE_SHA256,
        "point_in_time_available_at_proven": False,
        "rolling_origin_selection_authorized": False,
        "production_promotion_authorized": False,
    }


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def _quarter_strip(*, conflict: float = 0.0) -> pd.DataFrame:
    date = "2026-08-04"
    month_base = {"2027-01": 10.0, "2027-02": 20.0, "2027-03": 30.0}
    month_peak = {"2027-01": 12.0, "2027-02": 22.0, "2027-03": 32.0}
    base_energy = peak_energy = 0.0
    total_hours = peak_hours = 0
    rows: list[dict[str, object]] = []
    for product, base_price in month_base.items():
        month = int(product[-2:])
        total, peak, _ = count_hours(2027, month, month, country="CH")
        total_hours += total
        peak_hours += peak
        base_energy += base_price * total
        peak_energy += month_peak[product] * peak
        rows.extend(
            [
                _row(
                    date=date,
                    product=product,
                    product_type="Month",
                    load_type="BASE",
                    price=base_price,
                ),
                _row(
                    date=date,
                    product=product,
                    product_type="Month",
                    load_type="PEAK",
                    price=month_peak[product],
                ),
            ]
        )
    rows.extend(
        [
            _row(
                date=date,
                product="2027-Q1",
                product_type="Quarter",
                load_type="BASE",
                price=base_energy / total_hours + conflict,
            ),
            _row(
                date=date,
                product="2027-Q1",
                product_type="Quarter",
                load_type="PEAK",
                price=peak_energy / peak_hours,
            ),
        ]
    )
    return _frame(rows)


def _audit(frame: pd.DataFrame):
    return audit_databricks_eex_surfaces(
        frame,
        source_snapshot_sha256=SOURCE_SHA256,
        source_normalization_content_id=NORMALIZATION_ID,
    )


def test_audits_consistent_nested_strip_and_offpeak_recomposition() -> None:
    result = _audit(_quarter_strip())

    assert len(result.nesting_diagnostics) == 2
    assert set(result.nesting_diagnostics["status"]) == {"CONSISTENT"}
    assert result.summary["nesting_conflict_count"] == 0
    assert len(result.offpeak_diagnostics) == 4
    assert set(result.offpeak_diagnostics["status"]) == {"PASS"}
    assert result.summary["offpeak_recomposition_failure_count"] == 0
    assert result.daily_coverage.loc[0, "base_peak_pair_coverage_rate"] == 1.0
    assert result.summary["authority"]["model_selection_authorized"] is False


def test_classifies_market_nesting_conflict_without_failing_integrity() -> None:
    result = _audit(_quarter_strip(conflict=1.0))

    conflicts = result.nesting_diagnostics.loc[
        result.nesting_diagnostics["status"].eq("QUOTE_CONFLICT")
    ]
    assert len(conflicts) == 1
    assert conflicts.iloc[0]["load_type"] == "BASE"
    assert conflicts.iloc[0]["abs_residual_eur_mwh"] == pytest.approx(1.0)
    assert result.summary["status"] == (
        "PASS_LOCAL_INTEGRITY_WITH_MARKET_QUOTE_CONFLICTS"
    )
    assert result.summary["interpretation"][
        "quote_conflict_is_source_corruption"
    ] is False


def test_incomplete_child_strip_is_not_misreported_as_nested() -> None:
    frame = _quarter_strip()
    frame = frame.loc[~frame["product"].eq("2027-03")].reset_index(drop=True)

    result = _audit(frame)

    assert result.nesting_diagnostics.empty
    assert result.summary["fully_nested_parent_count"] == 0
    assert result.daily_coverage.loc[0, "unpaired_product_count"] == 0


def test_calendar_uses_whole_quarter_instead_of_splicing_partial_months() -> None:
    date = "2026-08-04"
    quarter_prices = {
        "2027-Q1": 100.0,
        "2027-Q2": 80.0,
        "2027-Q3": 70.0,
        "2027-Q4": 90.0,
    }
    rows = [
        _row(
            date=date,
            product=product,
            product_type="Quarter",
            load_type="BASE",
            price=price,
        )
        for product, price in quarter_prices.items()
    ]
    rows.extend(
        [
            _row(
                date=date,
                product="2027-01",
                product_type="Month",
                load_type="BASE",
                price=200.0,
            ),
            _row(
                date=date,
                product="2027-02",
                product_type="Month",
                load_type="BASE",
                price=200.0,
            ),
        ]
    )
    annual_energy = 0.0
    annual_hours = 0
    for quarter, price in quarter_prices.items():
        quarter_number = int(quarter[-1])
        first_month = (quarter_number - 1) * 3 + 1
        total, _, _ = count_hours(
            2027,
            first_month,
            first_month + 2,
            country="CH",
        )
        annual_hours += total
        annual_energy += price * total
    rows.append(
        _row(
            date=date,
            product="2027",
            product_type="Cal",
            load_type="BASE",
            price=annual_energy / annual_hours,
        )
    )

    result = _audit(_frame(rows))

    assert len(result.nesting_diagnostics) == 1
    calendar = result.nesting_diagnostics.iloc[0]
    assert calendar["parent_product"] == "2027"
    assert calendar["status"] == "CONSISTENT"
    assert calendar["child_products"] == (
        "2027-Q1|2027-Q2|2027-Q3|2027-Q4"
    )
    assert calendar["comparison_policy"] == (
        "FINEST_COMPLETE_NON_OVERLAPPING_CHILD_PARTITION"
    )


def test_negative_implied_offpeak_is_preserved_but_not_a_failure() -> None:
    frame = _frame(
        [
            _row(
                date="2026-08-04",
                product="2027-01",
                product_type="Month",
                load_type="BASE",
                price=0.0,
            ),
            _row(
                date="2026-08-04",
                product="2027-01",
                product_type="Month",
                load_type="PEAK",
                price=10.0,
            ),
        ]
    )

    result = _audit(frame)

    assert result.summary["negative_implied_offpeak_count"] == 1
    assert result.offpeak_diagnostics.loc[0, "status"] == "PASS"
    assert result.summary["interpretation"][
        "negative_implied_offpeak_is_automatically_invalid"
    ] is False


def test_rejects_duplicate_identity_and_authority_overclaim() -> None:
    frame = _quarter_strip()
    with pytest.raises(DatabricksEexSurfaceAuditError, match="quote identity"):
        _audit(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))

    overclaim = frame.copy()
    overclaim.loc[0, "rolling_origin_selection_authorized"] = True
    with pytest.raises(DatabricksEexSurfaceAuditError, match="overclaims"):
        _audit(overclaim)
