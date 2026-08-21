from __future__ import annotations

import numpy as np
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
from pfc_shaping.calibration.monthly_forward_curve import (
    MonthlyCurveConfig,
    solve_monthly_forward_curve_from_constraints,
)


def _constraints_2028_residual():
    months = pd.period_range("2028-01", "2028-12", freq="M")
    return build_monthly_constraint_system(months, {"2028": 80.40, "2028-Q1": 109.97})


def _constraints_2029_calendar():
    months = pd.period_range("2029-01", "2029-12", freq="M")
    return build_monthly_constraint_system(months, {"2029": 72.41})


def _constraints_2027_q2():
    months = pd.period_range("2027-04", "2027-06", freq="M")
    return build_monthly_constraint_system(months, {"2027-Q2": 71.95})


def _assert_zero_mean_by_parent(shape: pd.Series, constraints) -> None:
    for bucket in constraints.month_buckets.dropna().drop_duplicates():
        mask = constraints.month_buckets.eq(bucket)
        idx = constraints.month_buckets.index[mask]
        hours = constraints.delivery_grid.month_hours.loc[idx]
        mean = float((shape.loc[idx].to_numpy(dtype=float) * hours.to_numpy(dtype=float)).sum() / hours.sum())
        assert mean == pytest.approx(0.0, abs=1e-10)


def _bucket_mean(values: pd.Series, constraints, bucket: str) -> float:
    idx = constraints.month_buckets.index[constraints.month_buckets.eq(bucket)]
    hours = constraints.delivery_grid.month_hours.loc[idx]
    return float((values.loc[idx].to_numpy(dtype=float) * hours.to_numpy(dtype=float)).sum() / hours.sum())


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


def test_neighbor_calendar_only_quotes_do_not_create_shape_prior():
    constraints = _constraints_2029_calendar()
    prices = {
        "DE": {"2029": 80.0},
        "FR": {"2029": 85.0},
        "AT": {"2029": 75.0},
    }

    prior = build_neighbor_panel_shape_prior(
        constraints,
        prices,
        neighbor_markets=("DE", "FR", "AT"),
        neighbor_shrinkage=0.5,
    )

    assert prior.status == "UNSUPPORTED"
    assert set(prior.diagnostics["status"]) == {"CALENDAR_LEVEL_ONLY_UNSUPPORTED"}
    assert float(prior.shape.abs().max()) == pytest.approx(0.0)


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


def test_fused_prior_lets_direct_monthly_panel_dominate_structural_template_inside_parent():
    constraints = _constraints_2027_q2()
    de_prices = {
        "2027-04": 80.220000,
        "2027-05": 75.890000,
        "2027-06": 80.290000,
    }
    panel = build_neighbor_panel_shape_prior(
        constraints,
        {"DE": de_prices},
        neighbor_markets=("DE",),
        neighbor_shrinkage=0.0,
    )
    structural = build_structural_monthly_shape_prior(
        constraints,
        amplitude_eur_mwh=110.0,
    )

    fused = build_fused_shape_prior(
        constraints,
        panel_prior=panel,
        structural_prior=structural,
        weights={"panel": 1.0, "structural": 1.0},
    )
    result = solve_monthly_forward_curve_from_constraints(
        constraints,
        config=MonthlyCurveConfig(lambda_smooth_month=0.1, lambda_smooth_yoy=0.0, lambda_shape=25.0),
        shape_prior=fused,
    )

    assert "2027-Q2" in str(panel.diagnostics["direct_month_parent_buckets"].iloc[0])
    pd.testing.assert_series_equal(fused.shape, panel.shape, check_names=False)
    _assert_zero_mean_by_parent(fused.shape, constraints)
    assert result.residuals["abs_error"].max() <= 1e-8
    solved_dev = result.monthly_curve - _bucket_mean(result.monthly_curve, constraints, "2027-Q2")
    corr = float(np.corrcoef(solved_dev.to_numpy(dtype=float), panel.shape.to_numpy(dtype=float))[0, 1])
    assert corr >= 0.45


def test_fused_prior_does_not_suppress_structural_when_panel_weight_is_zero():
    constraints = _constraints_2027_q2()
    panel = build_neighbor_panel_shape_prior(
        constraints,
        {
            "DE": {
                "2027-04": 80.220000,
                "2027-05": 75.890000,
                "2027-06": 80.290000,
            }
        },
        neighbor_markets=("DE",),
        neighbor_shrinkage=0.0,
    )
    structural = build_structural_monthly_shape_prior(
        constraints,
        amplitude_eur_mwh=110.0,
    )

    fused = build_fused_shape_prior(
        constraints,
        panel_prior=panel,
        structural_prior=structural,
        weights={"panel": 0.0, "structural": 1.0},
    )

    pd.testing.assert_series_equal(fused.shape, structural.shape, check_names=False)
    structural_diag = fused.diagnostics.set_index("source").loc["structural"]
    assert structural_diag["suppressed_parent_buckets"] == ""
    assert structural_diag["effective_weight_min"] == pytest.approx(1.0)
    assert structural_diag["effective_weight_max"] == pytest.approx(1.0)


def test_template_structural_prior_reports_lambda_diagnostics_and_recenter_steps():
    constraints = _constraints_2028_residual()

    structural = build_structural_monthly_shape_prior(
        constraints,
        amplitude_eur_mwh=110.0,
        cap_eur_mwh=12.0,
        shrinkage=0.25,
    )
    diagnostics = structural.diagnostics.set_index("month")

    assert structural.status == "STRUCTURAL_TEMPLATE"
    assert {
        "source",
        "parent_bucket",
        "amplitude_eur_mwh",
        "raw_deviation_eur_mwh",
        "pre_recenter_parent_mean_eur_mwh",
        "recentered_deviation_eur_mwh",
        "pre_cap_deviation_eur_mwh",
        "cap_eur_mwh",
        "was_capped",
        "shrinkage",
        "recenter_adjustment_eur_mwh",
        "cap_adjustment_eur_mwh",
        "shape_deviation_eur_mwh",
        "shape_parent_mean_eur_mwh",
        "max_abs_parent_mean_residual",
        "zero_mean_parent_space",
        "fallback_reason",
        "n_history",
    }.issubset(structural.diagnostics.columns)
    assert diagnostics["source"].unique().tolist() == ["template_structural_monthly_ratios"]
    assert diagnostics["amplitude_eur_mwh"].unique().tolist() == [110.0]
    assert diagnostics["cap_eur_mwh"].unique().tolist() == [12.0]
    assert diagnostics["shrinkage"].unique().tolist() == [0.25]
    assert diagnostics.loc["2028-01", "raw_deviation_eur_mwh"] == pytest.approx(22.0)
    assert diagnostics.loc["2028-01", "recentered_deviation_eur_mwh"] != pytest.approx(22.0)
    assert diagnostics.loc["2028-01", "pre_cap_deviation_eur_mwh"] == pytest.approx(
        diagnostics.loc["2028-01", "recentered_deviation_eur_mwh"] * 0.75
    )
    assert diagnostics["was_capped"].any()
    assert bool(diagnostics.loc["2028-12", "was_capped"])
    for month in constraints.delivery_grid.months:
        row = diagnostics.loc[str(month)]
        assert row["shape_deviation_eur_mwh"] == pytest.approx(structural.shape.loc[month])
        assert row["recenter_adjustment_eur_mwh"] == pytest.approx(
            row["recentered_deviation_eur_mwh"] - row["raw_deviation_eur_mwh"]
        )
        assert row["shape_parent_mean_eur_mwh"] == pytest.approx(0.0, abs=1e-10)
    assert structural.contributions["shape_deviation_eur_mwh"].abs().max() <= 12.0
    assert diagnostics.loc["2028-12", "shape_deviation_eur_mwh"] == pytest.approx(12.0)
    assert diagnostics["max_abs_parent_mean_residual"].max() <= 1e-10
    _assert_zero_mean_by_parent(structural.shape, constraints)


def test_template_structural_fallback_reports_reason_and_history_counts():
    constraints = _constraints_2028_residual()
    history = pd.DataFrame(columns=["date", "market", "load_type", "product", "price"])

    structural = build_structural_monthly_shape_prior_from_history(
        constraints,
        history,
        fallback_to_template=True,
        fallback_amplitude_eur_mwh=110.0,
    )

    assert structural.status == "STRUCTURAL_TEMPLATE"
    diagnostics = structural.diagnostics
    assert set(diagnostics["fallback_reason"]) == {"empty_history"}
    assert diagnostics["n_history"].max() == 0
    assert diagnostics["amplitude_eur_mwh"].unique().tolist() == [110.0]
    assert diagnostics["max_abs_parent_mean_residual"].max() <= 1e-10
    _assert_zero_mean_by_parent(structural.shape, constraints)


def test_template_structural_prior_cap_is_invariant_to_ratio_level_offset():
    constraints = _constraints_2028_residual()
    ratios = {month: 1.0 + 0.02 * month for month in range(1, 13)}
    shifted = {month: ratio + 0.50 for month, ratio in ratios.items()}

    prior = build_structural_monthly_shape_prior(
        constraints,
        monthly_ratios=ratios,
        amplitude_eur_mwh=110.0,
        cap_eur_mwh=8.0,
        shrinkage=0.1,
    )
    shifted_prior = build_structural_monthly_shape_prior(
        constraints,
        monthly_ratios=shifted,
        amplitude_eur_mwh=110.0,
        cap_eur_mwh=8.0,
        shrinkage=0.1,
    )

    assert float((prior.shape - shifted_prior.shape).abs().max()) <= 1e-10
    assert prior.shape.abs().max() <= 8.0
    _assert_zero_mean_by_parent(prior.shape, constraints)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"amplitude_eur_mwh": float("nan")},
        {"cap_eur_mwh": -1.0},
        {"cap_eur_mwh": float("inf")},
        {"shrinkage": -0.1},
        {"shrinkage": 1.1},
        {"monthly_ratios": {1: float("nan")}},
    ],
)
def test_template_structural_prior_rejects_invalid_parameters(kwargs):
    constraints = _constraints_2028_residual()

    with pytest.raises(ValueError):
        build_structural_monthly_shape_prior(constraints, **kwargs)


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
