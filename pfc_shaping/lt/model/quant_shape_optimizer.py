"""Convex equality-constrained optimizer for LT quant shaping.

The first implementation deliberately solves only the deterministic market
curve ``p_t``.  Probabilistic scenario construction lives outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pfc_shaping.lt.model.shape_constraints import ConstraintSystem, FeasibilityReport


@dataclass(frozen=True)
class KKTReport:
    primal_inf: float
    stationarity_inf: float
    condition_number: float
    rank_a: int
    rank_augmented: int
    infeasibility_inf: float
    converged: bool


@dataclass(frozen=True)
class QuantShapeResult:
    curve: pd.Series
    lambdas: dict[str, float]
    feasibility: FeasibilityReport
    kkt: KKTReport
    constraint_residuals: pd.DataFrame
    objective_value: float


def second_difference_penalty(n: int) -> sp.csr_matrix:
    """Return ``D2.T @ D2`` for a one-dimensional curve."""

    if n < 3:
        return sp.csr_matrix((n, n), dtype=float)
    data = np.tile(np.array([1.0, -2.0, 1.0], dtype=float), n - 2)
    rows = np.repeat(np.arange(n - 2), 3)
    cols = np.concatenate([np.arange(i, i + 3) for i in range(n - 2)])
    d2 = sp.csr_matrix((data, (rows, cols)), shape=(n - 2, n))
    return d2.T @ d2


def _calendar_month_period(index: pd.DatetimeIndex) -> pd.PeriodIndex:
    local_index = index.tz_localize(None) if index.tz is not None else index
    return local_index.to_period("M")


def monthly_mean_second_difference_penalty(index: pd.Index) -> sp.csr_matrix:
    """Return a sparse penalty on second differences of calendar-month means."""

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("monthly mean smoothness requires a DatetimeIndex prior")
    n = len(index)
    if n < 3:
        return sp.csr_matrix((n, n), dtype=float)
    period = _calendar_month_period(index)
    month_codes, _ = pd.factorize(period, sort=False)
    n_months = int(month_codes.max()) + 1 if len(month_codes) else 0
    if n_months < 3:
        return sp.csr_matrix((n, n), dtype=float)
    month_counts = np.bincount(month_codes, minlength=n_months).astype(float)
    rows = np.arange(n, dtype=int)
    data = 1.0 / month_counts[month_codes]
    month_mean = sp.csr_matrix((data, (month_codes, rows)), shape=(n_months, n))
    d2 = sp.diags(
        diagonals=[np.ones(n_months - 2), -2.0 * np.ones(n_months - 2), np.ones(n_months - 2)],
        offsets=[0, 1, 2],
        shape=(n_months - 2, n_months),
        format="csr",
    )
    operator = d2 @ month_mean
    return operator.T @ operator


def monthly_boundary_jump_penalty(index: pd.Index) -> sp.csr_matrix:
    """Return a sparse penalty on jumps at calendar-month boundaries."""

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("monthly boundary smoothness requires a DatetimeIndex prior")
    n = len(index)
    if n < 2:
        return sp.csr_matrix((n, n), dtype=float)
    period = _calendar_month_period(index)
    boundary_positions = np.flatnonzero(period[1:].to_numpy() != period[:-1].to_numpy()) + 1
    if len(boundary_positions) == 0:
        return sp.csr_matrix((n, n), dtype=float)
    row_ids = np.repeat(np.arange(len(boundary_positions)), 2)
    col_ids = np.empty(2 * len(boundary_positions), dtype=int)
    col_ids[0::2] = boundary_positions - 1
    col_ids[1::2] = boundary_positions
    data = np.tile(np.array([-1.0, 1.0], dtype=float), len(boundary_positions))
    operator = sp.csr_matrix((data, (row_ids, col_ids)), shape=(len(boundary_positions), n))
    return operator.T @ operator


def _independent_constraint_rows(a: np.ndarray, rank: int) -> np.ndarray:
    if rank <= 0:
        return np.array([], dtype=int)
    if rank == a.shape[0]:
        return np.arange(a.shape[0], dtype=int)
    _, _, piv = la.qr(a.T, pivoting=True, mode="economic")
    return np.sort(np.asarray(piv[:rank], dtype=int))


class QuantShapeOptimizer:
    """Solve a convex QP ``min 1/2 p'Hp + f'p`` subject to ``A p = q``."""

    def __init__(
        self,
        *,
        lambda_prior: float = 1.0,
        lambda_smooth_h: float = 0.0,
        lambda_smooth_m: float = 0.0,
        lambda_seam: float = 0.0,
        epsilon_ridge: float = 1e-10,
        feasibility_tol: float = 1e-9,
        stationarity_tol: float = 1e-7,
    ) -> None:
        if lambda_prior <= 0.0 or not np.isfinite(lambda_prior):
            raise ValueError("lambda_prior must be positive and finite")
        if lambda_smooth_h < 0.0 or not np.isfinite(lambda_smooth_h):
            raise ValueError("lambda_smooth_h must be non-negative and finite")
        if lambda_smooth_m < 0.0 or not np.isfinite(lambda_smooth_m):
            raise ValueError("lambda_smooth_m must be non-negative and finite")
        if lambda_seam < 0.0 or not np.isfinite(lambda_seam):
            raise ValueError("lambda_seam must be non-negative and finite")
        if epsilon_ridge < 0.0 or not np.isfinite(epsilon_ridge):
            raise ValueError("epsilon_ridge must be non-negative and finite")
        self.lambda_prior = float(lambda_prior)
        self.lambda_smooth_h = float(lambda_smooth_h)
        self.lambda_smooth_m = float(lambda_smooth_m)
        self.lambda_seam = float(lambda_seam)
        self.epsilon_ridge = float(epsilon_ridge)
        self.feasibility_tol = float(feasibility_tol)
        self.stationarity_tol = float(stationarity_tol)

    def solve(
        self,
        prior: pd.Series,
        constraints: ConstraintSystem,
        *,
        prior_weights: np.ndarray | None = None,
    ) -> QuantShapeResult:
        if prior.empty:
            raise ValueError("prior curve is empty")
        if len(prior) != constraints.n_variables:
            raise ValueError(f"prior length {len(prior)} != constraint variables {constraints.n_variables}")
        p0 = prior.to_numpy(dtype=float)
        if not np.isfinite(p0).all():
            raise ValueError("prior contains non-finite values")
        n = len(p0)
        if prior_weights is None:
            w = np.ones(n, dtype=float)
        else:
            w = np.asarray(prior_weights, dtype=float)
            if w.shape != (n,):
                raise ValueError(f"prior_weights shape {w.shape} != ({n},)")
            if np.any(w <= 0.0) or not np.isfinite(w).all():
                raise ValueError("prior_weights must be strictly positive and finite")

        feasibility = constraints.feasibility_report()
        if not feasibility.feasible or feasibility.infeasibility_inf > self.feasibility_tol:
            raise ValueError(
                "hard constraints are infeasible: "
                f"rank(A)={feasibility.rank_a}, rank([A|q])={feasibility.rank_augmented}, "
                f"infeasibility={feasibility.infeasibility_inf:.3e}"
            )

        hessian = 2.0 * self.lambda_prior * sp.diags(w, format="csr")
        if self.lambda_smooth_h > 0.0:
            hessian = hessian + 2.0 * self.lambda_smooth_h * second_difference_penalty(n)
        if self.lambda_smooth_m > 0.0:
            hessian = hessian + 2.0 * self.lambda_smooth_m * monthly_mean_second_difference_penalty(prior.index)
        if self.lambda_seam > 0.0:
            hessian = hessian + 2.0 * self.lambda_seam * monthly_boundary_jump_penalty(prior.index)
        if self.epsilon_ridge > 0.0:
            hessian = hessian + 2.0 * self.epsilon_ridge * sp.eye(n, format="csr")
        linear = -2.0 * self.lambda_prior * w * p0

        a_sparse_full = constraints.sparse_matrix
        q = constraints.targets
        active_cols = np.unique(a_sparse_full.indices)
        a_rank = (
            a_sparse_full[:, active_cols].toarray()
            if len(active_cols)
            else np.zeros((a_sparse_full.shape[0], 0), dtype=float)
        )
        independent_rows = _independent_constraint_rows(a_rank, feasibility.rank_a)
        a_sparse = a_sparse_full[independent_rows, :] if len(independent_rows) else a_sparse_full[:0, :]
        q_work = q[independent_rows] if len(independent_rows) else q[:0]
        kkt = sp.bmat(
            [[hessian, a_sparse.T], [a_sparse, sp.csr_matrix((a_sparse.shape[0], a_sparse.shape[0]), dtype=float)]],
            format="csc",
        )
        rhs = np.concatenate([-linear, q_work])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", spla.MatrixRankWarning)
                solution = spla.spsolve(kkt, rhs)
        except (spla.MatrixRankWarning, RuntimeError, ValueError) as exc:
            raise ValueError(f"KKT solve failed: {exc}") from exc
        if not np.isfinite(solution).all():
            raise ValueError("KKT solve returned non-finite values")
        curve_values = solution[:n]
        lagrange = solution[n:]

        primal = a_sparse_full @ curve_values - q
        stationarity = hessian @ curve_values + linear + a_sparse.T @ lagrange
        primal_inf = float(np.max(np.abs(primal))) if len(primal) else 0.0
        stationarity_inf = float(np.max(np.abs(stationarity))) if len(stationarity) else 0.0
        cond = float(np.linalg.cond(kkt.toarray())) if kkt.shape[0] <= 2000 else float("nan")
        converged = bool(primal_inf <= self.feasibility_tol and stationarity_inf <= self.stationarity_tol)
        if not converged:
            raise ValueError(
                "KKT solve did not converge within hard tolerances: "
                f"primal_inf={primal_inf:.3e}, stationarity_inf={stationarity_inf:.3e}"
            )
        result_curve = pd.Series(curve_values, index=prior.index, name="price_quant_shape")
        objective = float(0.5 * curve_values @ (hessian @ curve_values) + linear @ curve_values)
        kkt_report = KKTReport(
            primal_inf=primal_inf,
            stationarity_inf=stationarity_inf,
            condition_number=cond,
            rank_a=feasibility.rank_a,
            rank_augmented=feasibility.rank_augmented,
            infeasibility_inf=feasibility.infeasibility_inf,
            converged=converged,
        )
        return QuantShapeResult(
            curve=result_curve,
            lambdas={
                "lambda_prior": self.lambda_prior,
                "lambda_smooth_h": self.lambda_smooth_h,
                "lambda_smooth_m": self.lambda_smooth_m,
                "lambda_seam": self.lambda_seam,
                "epsilon_ridge": self.epsilon_ridge,
            },
            feasibility=feasibility,
            kkt=kkt_report,
            constraint_residuals=constraints.residuals(curve_values),
            objective_value=objective,
        )
