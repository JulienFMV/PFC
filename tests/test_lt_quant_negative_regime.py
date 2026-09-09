from __future__ import annotations

import numpy as np
import pandas as pd

from pfc_shaping.lt.model.negative_price_regime import (
    ScenarioBundle,
    build_physical_paths,
    build_risk_neutral_paths,
    marginal_quantiles_from_paths,
    reconcile_q_scenario_mean_to_quotes,
)


def test_physical_paths_are_marked_physical_and_use_risk_premium_bridge() -> None:
    index = pd.RangeIndex(3)
    p_curve = pd.Series([20.0, 10.0, 5.0], index=index)
    residuals = pd.DataFrame({"s1": [0.0, -15.0, -10.0], "s2": [2.0, -12.0, -8.0]}, index=index)
    risk_premium = pd.Series([1.0, 2.0, 3.0], index=index)

    bundle = build_physical_paths(p_curve, residuals, risk_premium=risk_premium)

    assert bundle.measure == "P"
    pd.testing.assert_series_equal(bundle.risk_premium, risk_premium.astype(float))
    assert bundle.paths.loc[1, "s1"] == -7.0


def test_risk_neutral_paths_require_explicit_measure_and_generate_quantile_fan() -> None:
    index = pd.RangeIndex(3)
    q_curve = pd.Series([20.0, 10.0, 5.0], index=index)
    residuals = pd.DataFrame({"s1": [0.0, -15.0, -10.0], "s2": [2.0, -12.0, -8.0]}, index=index)

    bundle = build_risk_neutral_paths(q_curve, residuals)
    fan = marginal_quantiles_from_paths(bundle, alphas=(0.1, 0.5, 0.9))

    assert bundle.measure == "Q"
    assert fan.diagnostics.loc[0, "measure"] == "Q"
    assert list(fan.quantiles.columns) == ["q10", "q50", "q90"]


def test_reconcile_q_scenario_mean_to_quotes_preserves_quote_means() -> None:
    index = pd.RangeIndex(4)
    paths = pd.DataFrame({"s1": [9.0, 9.0, 19.0, 19.0], "s2": [11.0, 11.0, 21.0, 21.0]}, index=index)
    bundle = ScenarioBundle(paths=paths, measure="Q", base_curve=paths.mean(axis=1))
    matrix = np.array([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]], dtype=float)
    rhs = np.array([12.0, 17.0], dtype=float)

    report = reconcile_q_scenario_mean_to_quotes(bundle, matrix, rhs)

    mean_curve = report.reconciled.paths.mean(axis=1).to_numpy()
    np.testing.assert_allclose(matrix @ mean_curve, rhs, atol=1e-12)
    assert report.verdict in {"PASS", "CONDITIONAL"}
    assert float(report.constraint_residuals["abs_error"].max()) < 1e-12
    pd.testing.assert_series_equal(report.reconciled.base_curve, report.reconciled.paths.mean(axis=1))


def test_reconcile_reports_negative_runs_by_scenario_over_time() -> None:
    paths = pd.DataFrame(
        {
            "s1": [5.0, -1.0, -2.0, -3.0, 4.0],
            "s2": [5.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    bundle = ScenarioBundle(paths=paths, measure="Q", base_curve=paths.mean(axis=1))
    matrix = np.ones((1, len(paths)), dtype=float) / len(paths)
    rhs = np.array([float(paths.mean(axis=1).mean())])

    report = reconcile_q_scenario_mean_to_quotes(bundle, matrix, rhs)
    run_row = report.tail_diagnostics.loc[report.tail_diagnostics["metric"].eq("max_negative_run")].iloc[0]

    assert run_row["before"] == 3.0


def test_reconcile_returns_no_go_when_quotes_are_irreconcilable() -> None:
    paths = pd.DataFrame({"s1": [1.0, 1.0], "s2": [1.0, 1.0]})
    bundle = ScenarioBundle(paths=paths, measure="Q", base_curve=paths.mean(axis=1))
    matrix = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=float)
    rhs = np.array([1.0, 2.0], dtype=float)

    report = reconcile_q_scenario_mean_to_quotes(bundle, matrix, rhs)

    assert report.verdict == "NO_GO"
    assert float(report.constraint_residuals["abs_error"].max()) > 1e-9
