from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pfc_shaping.lt.model.quant_shape_optimizer as optimizer_mod
from pfc_shaping.lt.model.quant_shape_optimizer import QuantShapeOptimizer
from pfc_shaping.lt.model.shape_constraints import ConstraintRow, ConstraintSystem


def _constraints(matrix: np.ndarray, rhs: np.ndarray, names: tuple[str, ...]) -> ConstraintSystem:
    rows = tuple(
        ConstraintRow(name=name, target=target, weights=weights, kind="TEST")
        for name, target, weights in zip(names, rhs, matrix)
    )
    return ConstraintSystem(rows=rows, n_variables=matrix.shape[1])


def test_optimizer_reprices_constraints_and_reports_kkt_residuals() -> None:
    matrix = np.array(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5],
        ],
        dtype=float,
    )
    rhs = np.array([10.0, 20.0], dtype=float)
    constraints = _constraints(matrix, rhs, ("m1", "m2"))
    prior = pd.Series([9.0, 11.0, 19.0, 21.0], dtype=float)

    result = QuantShapeOptimizer(epsilon_ridge=1e-8).solve(prior, constraints)

    np.testing.assert_allclose(matrix @ result.curve.to_numpy(), rhs, atol=1e-10)
    assert result.kkt.primal_inf < 1e-10
    assert result.kkt.stationarity_inf < 1e-8
    assert result.kkt.converged


def test_optimizer_fails_before_solving_infeasible_constraints() -> None:
    matrix = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=float)
    rhs = np.array([1.0, 2.0], dtype=float)
    constraints = _constraints(matrix, rhs, ("a", "b"))

    with pytest.raises(ValueError, match="infeasible"):
        QuantShapeOptimizer().solve(pd.Series([0.0, 0.0]), constraints)


def test_optimizer_preserves_exact_fit_for_redundant_consistent_rows() -> None:
    matrix = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=float)
    rhs = np.array([8.0, 16.0], dtype=float)
    constraints = _constraints(matrix, rhs, ("base", "base_x2"))

    result = QuantShapeOptimizer(epsilon_ridge=1e-7).solve(pd.Series([1.0, 1.0]), constraints)

    np.testing.assert_allclose(matrix @ result.curve.to_numpy(), rhs, atol=1e-10)
    assert result.kkt.converged


def test_optimizer_hard_fails_if_sparse_solver_returns_bad_finite_solution(monkeypatch: pytest.MonkeyPatch) -> None:
    matrix = np.array([[1.0, 0.0]], dtype=float)
    rhs = np.array([5.0], dtype=float)
    constraints = _constraints(matrix, rhs, ("quote",))

    def bad_solve(_kkt, solve_rhs):
        return np.zeros_like(solve_rhs)

    monkeypatch.setattr(optimizer_mod.spla, "spsolve", bad_solve)

    with pytest.raises(ValueError, match="did not converge"):
        QuantShapeOptimizer().solve(pd.Series([0.0, 0.0]), constraints)
