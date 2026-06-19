from __future__ import annotations

import pandas as pd
import pytest

from pfc_shaping.calibration.monthly_curve_priors import (
    build_fused_shape_prior,
    build_history_shape_prior,
    build_neighbor_panel_shape_prior,
    build_structural_monthly_shape_prior,
    build_structural_monthly_shape_prior_from_history,
    quote_coverage_by_horizon,
    recenter_shape_by_parent,
)
from pfc_shaping.calibration.monthly_forward_curve import build_monthly_constraint_system


def _constraints_2028_residual():
    months = pd.period_range("2028-01", "2028-12", freq="M")
    return build_monthly_constraint_system(months, {"2028": 80.40, "2028-Q1": 109.97})


def _constraints_2029_calendar():
    months = pd.period_range("2029-01", "2029-12", freq="M")
    return build_monthly_constraint_system(months, {"2029": 72.41})


def _assert_zero_mean_by_parent(shape: pd.Series, constraints) -> None:
    for bucket in constraints.month_buckets.dropna().drop_duplicates():
        mask = constraints.month_buckets.eq(bucket)
        idx = constraints.month_buckets.index[mask]
        hours = constraints.delivery_grid.month_hours.loc[idx]
        mean = float((shape.loc[idx].to_numpy(dtype=float) * hours.to_numpy(dtype=float)).sum() / hours.sum())
        assert mean == pytest.approx(0.0, abs=1e-10)


def test_recenter_shape_by_parent_preserves_zero_mean_per_active_bucket():
    constraints = _constraints_2028_residual()
    raw = pd.Series(
        {month: float(i * 3.0) for i, month in enumerate(constraints.delivery_grid.months, start=1)}
    )

    shape = recenter_shape_by_parent(raw, constraints)

    _assert_zero_mean_by_parent(shape, constraints)


def test_neighbor_panel_prior_is_invariant_to_absolute_neighbor_level_shift():
    constraints = _constraints_2028_residual()
    months = constraints.delivery_grid.months
    de_prices = {str(month): 100.0 - 2.5 * i for i, month in enumerate(months)}
    shifted = {product: price + 1000.0 for product, price in de_prices.items()}

    prior = build_neighbor_panel_shape_prior(
        constraints,
        {"DE": de_prices},
        neighbor_markets=("DE",),
        neighbor_shrinkage=0.25,
    )
    shifted_prior = build_neighbor_panel_shape_prior(
        constraints,
        {"DE": shifted},
        neighbor_markets=("DE",),
        neighbor_shrinkage=0.25,
    )

    assert prior.status == "DE_SINGLE_MARKET_MONTHLY"
    assert float((prior.shape - shifted_prior.shape).abs().max()) <= 1e-10
    _assert_zero_mean_by_parent(prior.shape, constraints)


def test_neighbor_panel_prior_uses_robust_median_not_outlier_market():
    constraints = _constraints_2029_calendar()
    months = constraints.delivery_grid.months
    base_values = {str(month): 90.0 + 10.0 * (month.month in (1, 2, 12)) - 8.0 * (month.month in (5, 6, 7)) for month in months}
    fr_shifted = {product: price + 50.0 for product, price in base_values.items()}
    outlier = dict(base_values)
    outlier["2029-04"] = 500.0
    expected = recenter_shape_by_parent(pd.Series({pd.Period(k, freq="M"): v for k, v in base_values.items()}), constraints)

    prior = build_neighbor_panel_shape_prior(
        constraints,
        {"DE": base_values, "FR": fr_shifted, "AT": outlier},
        neighbor_markets=("DE", "FR", "AT"),
        neighbor_shrinkage=0.0,
    )

    assert prior.status == "PANEL_MULTI_MARKET"
    assert prior.shape.loc[pd.Period("2029-04", freq="M")] == pytest.approx(
        expected.loc[pd.Period("2029-04", freq="M")]
    )
    _assert_zero_mean_by_parent(prior.shape, constraints)


def test_neighbor_panel_block_only_support_is_not_reported_as_multi_market_monthly_panel():
    constraints = _constraints_2029_calendar()
    prices = {
        "DE": {"2029": 80.0, "2029-Q1": 100.0},
        "FR": {"2029": 85.0, "2029-Q1": 105.0},
        "AT": {"2029": 75.0, "2029-Q1": 95.0},
    }

    prior = build_neighbor_panel_shape_prior(
        constraints,
        prices,
        neighbor_markets=("DE", "FR", "AT"),
        neighbor_shrinkage=0.5,
    )

    assert prior.status == "PANEL_BLOCK_SHAPE"
    assert set(prior.diagnostics["status"]) == {"BLOCK_SHAPE_USED"}
    assert prior.diagnostics["direct_month_quotes"].sum() == 0
    assert prior.diagnostics["block_shape_months"].sum() > 0
    _assert_zero_mean_by_parent(prior.shape, constraints)


def test_neighbor_panel_partial_monthly_support_is_not_reported_as_full_panel():
    constraints = _constraints_2029_calendar()
    prices = {
        "DE": {"2029": 80.0, "2029-01": 101.0, "2029-02": 100.0, "2029-03": 95.0},
        "FR": {"2029": 83.0, "2029-01": 103.0, "2029-02": 102.0, "2029-03": 97.0},
    }

    prior = build_neighbor_panel_shape_prior(
        constraints,
        prices,
        neighbor_markets=("DE", "FR"),
        neighbor_shrinkage=0.5,
    )
    fused = build_fused_shape_prior(constraints, panel_prior=prior)

    assert prior.status == "PARTIAL_MONTHLY_PANEL"
    assert fused.status == "PARTIAL_MONTHLY_PANEL"
    assert prior.diagnostics["direct_month_quote_share"].max() < 1.0
    assert set(prior.diagnostics["status"]) == {"MONTH_SHAPE_USED"}
    _assert_zero_mean_by_parent(prior.shape, constraints)


def test_neighbor_panel_reports_de_far_horizon_monthly_evidence_explicitly():
    constraints = _constraints_2028_residual()
    months = constraints.delivery_grid.months
    de_prices = {str(month): 100.0 + float(month.month) for month in months}

    prior = build_neighbor_panel_shape_prior(
        constraints,
        {"DE": de_prices},
        neighbor_markets=("DE", "FR"),
        neighbor_shrinkage=0.25,
        run_timestamp=pd.Timestamp("2026-06-17"),
    )

    de_diag = prior.diagnostics[prior.diagnostics["market"].eq("DE")].iloc[0]
    fr_diag = prior.diagnostics[prior.diagnostics["market"].eq("FR")].iloc[0]
    assert prior.status == "DE_SINGLE_MARKET_MONTHLY"
    assert de_diag["direct_month_quotes_h+2"] == 12
    assert de_diag["prior_far_horizon_monthly_evidence"] == "DE_FAR_HORIZON_MONTHLY_EVIDENCE"
    assert de_diag["market_far_horizon_monthly_evidence"] == "DE_FAR_HORIZON_MONTHLY_EVIDENCE"
    assert fr_diag["market_far_horizon_monthly_evidence"] == "NO_FAR_HORIZON_MONTHLY_EVIDENCE"
    _assert_zero_mean_by_parent(prior.shape, constraints)


def test_neighbor_panel_block_shape_does_not_claim_far_horizon_monthly_evidence():
    constraints = _constraints_2028_residual()
    prices = {
        "DE": {"2028": 80.0, "2028-Q1": 100.0},
        "FR": {"2028": 82.0, "2028-Q1": 101.0},
    }

    prior = build_neighbor_panel_shape_prior(
        constraints,
        prices,
        neighbor_markets=("DE", "FR"),
        neighbor_shrinkage=0.25,
        run_timestamp=pd.Timestamp("2026-06-17"),
    )

    assert prior.status == "PANEL_BLOCK_SHAPE"
    assert set(prior.diagnostics["prior_far_horizon_monthly_evidence"]) == {"NO_FAR_HORIZON_MONTHLY_EVIDENCE"}
    assert set(prior.diagnostics["market_far_horizon_monthly_evidence"]) == {"NO_FAR_HORIZON_MONTHLY_EVIDENCE"}
    assert int(prior.diagnostics["direct_month_quotes_h+2"].sum()) == 0


def test_history_shape_prior_computes_month_vs_calendar_deviations():
    constraints = _constraints_2029_calendar()
    deviations = {month: 12.0 - month for month in range(1, 13)}
    rows = []
    for date, hist_year, cal in [
        ("2026-06-01", 2027, 80.0),
        ("2026-07-01", 2028, 82.0),
    ]:
        rows.append(_history_row(date, str(hist_year), cal))
        for month, deviation in deviations.items():
            rows.append(_history_row(date, f"{hist_year}-{month:02d}", cal + deviation))
    history = pd.DataFrame(rows)

    prior = build_history_shape_prior(
        constraints,
        history,
        run_timestamp=pd.Timestamp("2026-12-31"),
        min_snapshots=2,
    )
    expected_raw = pd.Series(
        {pd.Period(f"2029-{month:02d}", freq="M"): deviation for month, deviation in deviations.items()}
    )
    expected = recenter_shape_by_parent(expected_raw, constraints)

    assert prior.status == "HISTORY_FORWARD"
    assert prior.shape.loc[pd.Period("2029-01", freq="M")] == pytest.approx(
        expected.loc[pd.Period("2029-01", freq="M")]
    )
    assert prior.diagnostics["n_history"].min() == 2
    _assert_zero_mean_by_parent(prior.shape, constraints)


def test_history_shape_prior_fails_closed_when_snapshot_support_is_insufficient():
    constraints = _constraints_2029_calendar()
    history = pd.DataFrame([_history_row("2026-06-01", "2027", 80.0)])

    prior = build_history_shape_prior(constraints, history, min_snapshots=2)

    assert prior.status == "UNSUPPORTED"
    assert float(prior.shape.abs().max()) == pytest.approx(0.0)


def test_template_structural_and_fused_prior_keep_zero_mean_and_status_label():
    constraints = _constraints_2028_residual()

    structural = build_structural_monthly_shape_prior(constraints, amplitude_eur_mwh=15.0)
    fused = build_fused_shape_prior(constraints, structural_prior=structural)

    assert structural.status == "STRUCTURAL_TEMPLATE"
    assert fused.status == "STRUCTURAL_TEMPLATE"
    _assert_zero_mean_by_parent(structural.shape, constraints)
    _assert_zero_mean_by_parent(fused.shape, constraints)


def test_structural_prior_can_be_derived_from_forward_history():
    constraints = _constraints_2029_calendar()
    rows = []
    deviations = {month: 6.0 - month for month in range(1, 13)}
    for snap in pd.date_range("2026-01-01", periods=3, freq="MS"):
        hist_year = 2027
        rows.append(_history_row(str(snap.date()), str(hist_year), 80.0))
        for month, deviation in deviations.items():
            rows.append(_history_row(str(snap.date()), f"{hist_year}-{month:02d}", 80.0 + deviation))
    history = pd.DataFrame(rows)

    structural = build_structural_monthly_shape_prior_from_history(
        constraints,
        history,
        run_timestamp=pd.Timestamp("2026-12-31"),
        min_snapshots=3,
    )
    expected = recenter_shape_by_parent(
        pd.Series(
            {pd.Period(f"2029-{month:02d}", freq="M"): deviation for month, deviation in deviations.items()}
        ),
        constraints,
    )

    assert structural.status == "STRUCTURAL_FORWARD_CLIMATOLOGY"
    assert structural.shape.loc[pd.Period("2029-01", freq="M")] == pytest.approx(
        expected.loc[pd.Period("2029-01", freq="M")]
    )
    _assert_zero_mean_by_parent(structural.shape, constraints)


def test_quote_coverage_by_horizon_counts_monthly_quotes_only():
    rows = [
        _history_row("2026-06-17", "2026-07", 60.0, market="CH"),
        _history_row("2026-06-17", "2027-01", 61.0, market="CH"),
        _history_row("2026-06-17", "2028-01", 62.0, market="CH"),
        _history_row("2026-06-17", "2030-01", 63.0, market="CH"),
        _history_row("2026-06-17", "2028-Q1", 64.0, market="CH", product_type="Quarter"),
        _history_row("2026-06-17", "2028-01", 70.0, market="DE"),
    ]

    coverage = quote_coverage_by_horizon(pd.DataFrame(rows))
    counts = {
        (row.market, row.horizon_bucket): int(row.monthly_quote_count)
        for row in coverage.itertuples(index=False)
    }

    assert counts[("CH", "h+0")] == 1
    assert counts[("CH", "h+1")] == 1
    assert counts[("CH", "h+2")] == 1
    assert counts[("CH", "h+3+")] == 1
    assert counts[("DE", "h+2")] == 1


def _history_row(
    date: str,
    product: str,
    price: float,
    *,
    market: str = "CH",
    load_type: str = "BASE",
    product_type: str | None = None,
) -> dict[str, object]:
    if product_type is None:
        product_type = "Month" if "-" in product and "-Q" not in product else "Cal"
    return {
        "date": pd.Timestamp(date),
        "product": product,
        "load_type": load_type,
        "product_type": product_type,
        "price": float(price),
        "market": market,
        "source": "TEST",
    }
