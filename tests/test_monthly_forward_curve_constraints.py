from __future__ import annotations

import ast
import pathlib

import pandas as pd
import pytest

from pfc_shaping.calibration.monthly_forward_curve import (
    MarketQuote,
    MonthlyCurveConfig,
    build_monthly_constraint_system,
    month_delivery_hours,
    solve_monthly_forward_curve_from_constraints,
)


def test_month_hours_handle_leap_year_and_dst_transitions():
    hours = month_delivery_hours(pd.PeriodIndex(["2028-02", "2028-03", "2028-10"], freq="M"))

    assert hours.loc[pd.Period("2028-02", freq="M")] == pytest.approx(29 * 24)
    assert hours.loc[pd.Period("2028-03", freq="M")] == pytest.approx(31 * 24 - 1)
    assert hours.loc[pd.Period("2028-10", freq="M")] == pytest.approx(31 * 24 + 1)


def test_cal_plus_q1_implies_hour_weighted_apr_dec_residual_target():
    months = pd.period_range("2028-01", "2028-12", freq="M")
    system = build_monthly_constraint_system(months, {"2028": 80.40, "2028-Q1": 109.97})

    hours = system.delivery_grid.month_hours
    q1_hours = hours.loc[pd.period_range("2028-01", "2028-03", freq="M")].sum()
    year_hours = hours.sum()
    residual_hours = year_hours - q1_hours
    expected_residual = (80.40 * year_hours - 109.97 * q1_hours) / residual_hours

    assert system.bucket_targets["2028-Q1"] == pytest.approx(109.97)
    assert system.bucket_targets["2028-RESIDUAL"] == pytest.approx(expected_residual)
    assert system.bucket_targets["2028-RESIDUAL"] == pytest.approx(70.6209801545)
    assert system.feasibility_report().feasible


def test_residual_constraint_binds_parent_and_finer_quote_lineage() -> None:
    months = pd.period_range("2028-01", "2028-12", freq="M")
    quotes = (
        MarketQuote(
            market="CH", product="2028", load_type="BASE", price=80.40,
            quote_id="quote-cal", snapshot_id="snapshot", observation_id="observation",
        ),
        MarketQuote(
            market="CH", product="2028-Q1", load_type="BASE", price=109.97,
            quote_id="quote-q1", snapshot_id="snapshot", observation_id="observation",
        ),
    )

    system = build_monthly_constraint_system(months, quotes)
    residual = system.rows.loc[system.rows["product"].eq("2028-RESIDUAL")].iloc[0]

    assert set(residual["source_quote_ids"].split("|")) == {"quote-cal", "quote-q1"}
    assert residual["lineage_formula"] == "residual_parent_energy_less_known_buckets"
    assert len(residual["lineage_sha256"]) == 64


def test_high_q1_can_make_apr_dec_residual_below_lower_next_calendar():
    months = pd.period_range("2028-01", "2029-12", freq="M")
    system = build_monthly_constraint_system(
        months,
        {"2028": 80.99, "2028-Q1": 111.25, "2029": 73.25},
    )

    assert system.bucket_targets["2028-RESIDUAL"] < system.bucket_targets["2029"]
    assert system.bucket_targets["2028-RESIDUAL"] == pytest.approx(70.9827920012)


def test_manual_bucket_solution_reprices_active_constraints_exactly():
    months = pd.period_range("2028-01", "2028-12", freq="M")
    system = build_monthly_constraint_system(months, {"2028": 80.40, "2028-Q1": 109.97})
    values = system.month_buckets.map(lambda bucket: system.bucket_targets[str(bucket)]).to_numpy(dtype=float)

    residuals = system.residuals(values)

    assert residuals["abs_error"].max() <= 1e-10


def test_consistent_redundant_calendar_and_quarters_pass_and_report_calendar_drop():
    months = pd.period_range("2028-01", "2028-12", freq="M")
    quotes = {
        "2028": 80.0,
        "2028-Q1": 80.0,
        "2028-Q2": 80.0,
        "2028-Q3": 80.0,
        "2028-Q4": 80.0,
    }

    system = build_monthly_constraint_system(months, quotes)

    calendar_diag = system.quote_diagnostics.loc[system.quote_diagnostics["product"] == "2028"].iloc[0]
    assert bool(calendar_diag["active"]) is False
    assert calendar_diag["dropped_reason"] == "redundant_consistent"
    assert tuple(calendar_diag["active_row_indices"]) == (0, 1, 2, 3)
    assert system.names == ["BASE:2028-Q1", "BASE:2028-Q2", "BASE:2028-Q3", "BASE:2028-Q4"]


def test_conflicting_calendar_and_quarters_fail_instead_of_dropping_quote():
    months = pd.period_range("2028-01", "2028-12", freq="M")
    quotes = {
        "2028": 80.0,
        "2028-Q1": 90.0,
        "2028-Q2": 90.0,
        "2028-Q3": 90.0,
        "2028-Q4": 90.0,
    }

    with pytest.raises(ValueError, match="inconsistent quoted product 2028"):
        build_monthly_constraint_system(months, quotes)


def test_duplicate_product_conflict_fails_before_constraint_build():
    months = pd.period_range("2028-01", "2028-12", freq="M")
    quotes = (
        MarketQuote(market="CH", product="2028", load_type="BASE", price=80.0),
        MarketQuote(market="CH", product="2028", load_type="BASE", price=81.0),
    )

    with pytest.raises(ValueError, match="conflicting duplicate quote"):
        build_monthly_constraint_system(months, quotes)


def test_foreign_market_quote_in_own_constraints_fails_closed():
    months = pd.period_range("2028-01", "2028-12", freq="M")
    quotes = (
        MarketQuote(market="DE", product="2028", load_type="BASE", price=80.0),
    )

    with pytest.raises(ValueError, match="unexpected quote market"):
        build_monthly_constraint_system(months, quotes, market="CH")


def test_partial_product_grid_fails_closed():
    months = pd.period_range("2028-04", "2028-12", freq="M")

    with pytest.raises(ValueError, match="partial delivery grid for quoted product 2028"):
        build_monthly_constraint_system(months, {"2028": 80.40})


def test_solver_rejects_delivery_year_without_own_market_level_anchor():
    months = pd.period_range("2027-01", "2028-12", freq="M")
    constraints = build_monthly_constraint_system(months, {"2027": 80.0})
    shape = pd.Series(
        [float(month.month) for month in months],
        index=months,
        dtype=float,
    )

    with pytest.raises(ValueError, match="no own-market level anchor.*2028"):
        solve_monthly_forward_curve_from_constraints(
            constraints,
            shape_prior=shape,
        )


def test_solver_applies_relative_seasonal_prior_to_unrepresented_month():
    months = pd.period_range("2027-01", "2027-12", freq="M")
    quotes = {
        f"2027-{month:02d}": 80.0 + float(month - 6.5)
        for month in range(2, 13)
    }
    constraints = build_monthly_constraint_system(months, quotes)
    shape = pd.Series(
        [float(month.month - 6.5) for month in months],
        index=months,
        dtype=float,
    )

    result = solve_monthly_forward_curve_from_constraints(
        constraints,
        shape_prior=shape,
    )

    assert result.monthly_curve.loc[pd.Period("2027-01", freq="M")] == pytest.approx(
        74.5, abs=1e-5
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_solver_rejects_nonfinite_shape_prior(value: float):
    months = pd.period_range("2027-01", "2027-12", freq="M")
    constraints = build_monthly_constraint_system(months, {"2027": 80.0})
    shape = pd.Series(0.0, index=months, dtype=float)
    shape.iloc[0] = value

    with pytest.raises(ValueError, match="shape prior contains non-finite"):
        solve_monthly_forward_curve_from_constraints(
            constraints,
            shape_prior=shape,
        )


@pytest.mark.parametrize(
    "field",
    ["lambda_prior", "lambda_smooth_month", "lambda_smooth_yoy", "lambda_shape"],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, "1.0"])
def test_solver_rejects_invalid_objective_weights(field: str, value: object):
    months = pd.period_range("2027-01", "2027-12", freq="M")
    constraints = build_monthly_constraint_system(months, {"2027": 80.0})
    config = MonthlyCurveConfig(**{field: value})

    with pytest.raises(ValueError, match=f"monthly solver {field}.*finite non-negative"):
        solve_monthly_forward_curve_from_constraints(constraints, config=config)


def test_solver_normalizes_string_month_index_without_dropping_prior():
    months = pd.period_range("2027-01", "2027-12", freq="M")
    constraints = build_monthly_constraint_system(
        months,
        {
            f"2027-{month:02d}": 80.0 + float(month - 6.5)
            for month in range(2, 13)
        },
    )
    shape = pd.Series(
        [float(month - 6.5) for month in range(1, 13)],
        index=[f"2027-{month:02d}" for month in range(1, 13)],
        dtype=float,
    )

    result = solve_monthly_forward_curve_from_constraints(
        constraints,
        shape_prior=shape,
    )

    assert result.monthly_curve.loc[pd.Period("2027-01", freq="M")] == pytest.approx(
        74.5, abs=1e-5
    )


def test_solver_rejects_shape_index_collisions_after_month_normalization():
    months = pd.period_range("2027-01", "2027-12", freq="M")
    constraints = build_monthly_constraint_system(months, {"2027": 80.0})
    shape = pd.Series([1.0, 2.0], index=["2027-01-01", "2027-01-31"])

    with pytest.raises(ValueError, match="duplicate months"):
        solve_monthly_forward_curve_from_constraints(
            constraints,
            shape_prior=shape,
        )


@pytest.mark.parametrize("missing_month", [pd.NaT, "NaT"])
def test_solver_rejects_missing_shape_index_month(missing_month: object):
    months = pd.period_range("2027-01", "2027-12", freq="M")
    constraints = build_monthly_constraint_system(months, {"2027": 80.0})
    shape = pd.Series([1.0], index=[missing_month])

    with pytest.raises(ValueError, match="index contains missing months"):
        solve_monthly_forward_curve_from_constraints(
            constraints,
            shape_prior=shape,
        )


def test_monthly_forward_curve_module_does_not_import_ct_or_deprecated_model_path():
    source = pathlib.Path("pfc_shaping/calibration/monthly_forward_curve.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    assert all(not module.startswith("pfc_shaping.ct") for module in imported_modules)
    assert all(not module.startswith("pfc_shaping.model") for module in imported_modules)
