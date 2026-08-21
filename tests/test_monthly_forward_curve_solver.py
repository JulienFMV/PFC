from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from pfc_shaping.calibration.monthly_curve_priors import (
    build_neighbor_panel_shape_prior,
    build_structural_monthly_shape_prior,
)
from pfc_shaping.calibration.monthly_forward_curve import (
    MonthlyCurveConfig,
    build_monthly_constraint_system,
    solve_monthly_forward_curve,
    solve_monthly_forward_curve_from_constraints,
)
from pfc_shaping.calibration.monthly_forward_curve import (
    _parent_mean_operator,
    _second_difference_operator,
    _yoy_shape_operator,
)


def test_solver_reprices_sparse_calendar_q1_constraints_exactly():
    months = pd.period_range("2028-01", "2028-12", freq="M")
    constraints = build_monthly_constraint_system(months, {"2028": 80.40, "2028-Q1": 109.97})

    result = solve_monthly_forward_curve_from_constraints(
        constraints,
        config=MonthlyCurveConfig(lambda_smooth_month=1.0, lambda_smooth_yoy=0.0, lambda_shape=0.0),
    )

    assert result.kkt["max_abs_constraint_residual"] <= 1e-8
    assert result.kkt["stationarity_residual"] <= 1e-7
    assert result.residuals["abs_error"].max() <= 1e-8
    assert _bucket_mean(result.monthly_curve, constraints, "2028-Q1") == pytest.approx(109.97)
    assert _bucket_mean(result.monthly_curve, constraints, "2028-RESIDUAL") == pytest.approx(
        constraints.bucket_targets["2028-RESIDUAL"]
    )


def test_calendar_only_quote_moves_solution_through_hard_constraint():
    months = pd.period_range("2029-01", "2029-12", freq="M")
    low = solve_monthly_forward_curve(
        market="CH",
        delivery_months=months,
        own_quotes={"2029": 72.41},
        config=MonthlyCurveConfig(lambda_smooth_month=1.0, lambda_smooth_yoy=0.0, lambda_shape=0.0),
    )
    high = solve_monthly_forward_curve(
        market="CH",
        delivery_months=months,
        own_quotes={"2029": 82.41},
        config=MonthlyCurveConfig(lambda_smooth_month=1.0, lambda_smooth_yoy=0.0, lambda_shape=0.0),
    )

    assert float(low.monthly_curve.std()) <= 1e-8
    assert float(high.monthly_curve.std()) <= 1e-8
    assert (high.monthly_curve - low.monthly_curve).min() == pytest.approx(10.0)
    assert (high.monthly_curve - low.monthly_curve).max() == pytest.approx(10.0)


def test_solver_solution_is_invariant_to_absolute_neighbor_level_shift():
    months = pd.period_range("2028-01", "2028-12", freq="M")
    constraints = build_monthly_constraint_system(months, {"2028": 80.40, "2028-Q1": 109.97})
    de_prices = {str(month): 100.0 - 3.0 * i for i, month in enumerate(months)}
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
    config = MonthlyCurveConfig(lambda_smooth_month=1.0, lambda_smooth_yoy=0.0, lambda_shape=10.0)

    result = solve_monthly_forward_curve_from_constraints(constraints, config=config, shape_prior=prior)
    shifted_result = solve_monthly_forward_curve_from_constraints(
        constraints,
        config=config,
        shape_prior=shifted_prior,
    )

    assert float((result.monthly_curve - shifted_result.monthly_curve).abs().max()) <= 1e-8
    assert result.residuals["abs_error"].max() <= 1e-8
    assert shifted_result.residuals["abs_error"].max() <= 1e-8


def test_solver_reports_rank_nullspace_and_conditioning():
    months = pd.period_range("2028-01", "2028-12", freq="M")
    constraints = build_monthly_constraint_system(months, {"2028": 80.40, "2028-Q1": 109.97})

    result = solve_monthly_forward_curve_from_constraints(
        constraints,
        config=MonthlyCurveConfig(lambda_smooth_month=1.0, lambda_smooth_yoy=0.0, lambda_shape=0.0),
    )

    assert result.kkt["active_constraint_rank"] == 2
    assert result.kkt["nullspace_dimension"] == 10
    assert result.kkt["condition_number"] > 0.0
    assert result.kkt["solved_by_lstsq"] is False


def test_shape_prior_moves_unquoted_degrees_without_breaking_eex_quotes():
    months = pd.period_range("2028-01", "2028-12", freq="M")
    constraints = build_monthly_constraint_system(months, {"2028": 80.40, "2028-Q1": 109.97})
    structural = build_structural_monthly_shape_prior(constraints, amplitude_eur_mwh=20.0)
    config = MonthlyCurveConfig(lambda_smooth_month=0.1, lambda_smooth_yoy=0.0, lambda_shape=25.0)

    base = solve_monthly_forward_curve_from_constraints(
        constraints,
        config=MonthlyCurveConfig(lambda_smooth_month=1.0, lambda_smooth_yoy=0.0, lambda_shape=0.0),
    )
    shaped = solve_monthly_forward_curve_from_constraints(
        constraints,
        config=config,
        shape_prior=structural,
    )

    assert shaped.residuals["abs_error"].max() <= 1e-8
    assert float((shaped.monthly_curve - base.monthly_curve).abs().max()) > 0.1


def test_month_smoothness_does_not_cross_incompatible_parent_blocks():
    months = pd.period_range("2028-01", "2028-12", freq="M")
    constraints = build_monthly_constraint_system(months, {"2028": 80.40, "2028-Q1": 109.97})

    d2 = _second_difference_operator(constraints)

    march = months.get_loc(pd.Period("2028-03", freq="M"))
    april = months.get_loc(pd.Period("2028-04", freq="M"))
    crossing_rows = [
        row
        for row in d2
        if abs(row[march]) > 0.0 and abs(row[april]) > 0.0
    ]
    assert crossing_rows == []


def test_month_smoothness_does_not_cross_distinct_calendar_buckets():
    months = pd.period_range("2029-01", "2030-12", freq="M")
    constraints = build_monthly_constraint_system(months, {"2029": 73.25, "2030": 69.60})

    d2 = _second_difference_operator(constraints)

    dec_2029 = months.get_loc(pd.Period("2029-12", freq="M"))
    jan_2030 = months.get_loc(pd.Period("2030-01", freq="M"))
    crossing_rows = [
        row
        for row in d2
        if abs(row[dec_2029]) > 0.0 and abs(row[jan_2030]) > 0.0
    ]
    assert crossing_rows == []


def test_yoy_shape_operator_compares_residual_to_calendar_comparable_block():
    months = pd.period_range("2028-01", "2029-12", freq="M")
    constraints = build_monthly_constraint_system(
        months,
        {"2028": 80.40, "2028-Q1": 109.97, "2029": 82.67},
    )
    parent_operator = _parent_mean_operator(constraints)
    shape_operator = np.eye(len(months)) - parent_operator

    dyoy, dropped = _yoy_shape_operator(constraints, shape_operator)

    april_2028 = months.get_loc(pd.Period("2028-04", freq="M"))
    april_2029 = months.get_loc(pd.Period("2029-04", freq="M"))
    comparable_rows = [
        row
        for row in dyoy
        if abs(row[april_2028]) > 0.0 and abs(row[april_2029]) > 0.0
    ]
    assert comparable_rows
    assert dropped == 3


def _bucket_mean(curve: pd.Series, constraints, bucket: str) -> float:
    idx = constraints.month_buckets.index[constraints.month_buckets.eq(bucket)]
    hours = constraints.delivery_grid.month_hours.loc[idx]
    return float((curve.loc[idx].to_numpy(dtype=float) * hours.to_numpy(dtype=float)).sum() / hours.sum())
