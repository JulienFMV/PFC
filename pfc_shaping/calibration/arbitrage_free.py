"""
arbitrage_free.py
-----------------
Arbitrage-free calibration of a 15-minute HPFC (Hourly Price Forward Curve)
using Maximum Smoothness Forward Curve methodology.

The algorithm applies additive shifting functions to the raw shape curve so
that the average price over each futures delivery period exactly matches the
quoted futures price, while maximising smoothness (minimising the second
derivative of the correction function).

Mathematical formulation
~~~~~~~~~~~~~~~~~~~~~~~~
Given raw shape curve S(t) for t = 1..N (15-min timestamps) and M futures
contracts with delivery periods [a_i, b_i] and settlement prices F_i, find
correction function delta(t) that:

    Minimises:  sum (delta''(t))^2             (smoothness objective)
    Subject to: (1/n_i) * sum_{t in [a_i,b_i]} [S(t) + delta(t)] = F_i

The constrained QP is solved via the KKT system:

    | H   A^T |   | delta |   | 0 |
    |         | * |       | = |   |
    | A   0   |   | lambda|   | b |

where H is the second-difference (curvature) penalty matrix and b_i is the
residual F_i - mean(S over period i).

Reference: Biegler-Koenig & Pilz, 2015 — "Maximum Smoothness Forward Curves".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import holidays
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FuturesContract:
    """Represents an EEX power futures contract.

    Attributes:
        name: Human-readable contract label, e.g. ``"Cal-2026"``,
            ``"Q1-2026"``, ``"Mar-2026"``.
        price: Settlement price in EUR/MWh.
        start: Delivery start timestamp (inclusive, UTC).
        end: Delivery end timestamp (exclusive, UTC).
        product_type: ``'Base'`` (all hours) or ``'Peak'``
            (08:00-20:00 Mon-Fri, excl. Swiss public holidays).
    """

    name: str
    price: float
    start: pd.Timestamp
    end: pd.Timestamp
    product_type: str = "Base"

    def __post_init__(self) -> None:
        if self.product_type not in ("Base", "Peak"):
            raise ValueError(
                f"product_type must be 'Base' or 'Peak', got {self.product_type!r}"
            )
        if self.start >= self.end:
            raise ValueError(
                f"Contract {self.name}: start ({self.start}) must be before "
                f"end ({self.end})"
            )


@dataclass
class CalibrationResult:
    """Output of the arbitrage-free calibration.

    Attributes:
        calibrated_curve: Final P(t) = S(t) + delta(t). Index is a
            ``DatetimeIndex`` at 15-min frequency in UTC.
        delta: Additive correction applied to the raw curve.
        residuals: Per-contract comparison of target vs achieved average
            price, with columns ``['contract', 'target', 'achieved',
            'abs_error']``.
        max_abs_residual: Largest absolute repricing error across all
            contracts (EUR/MWh). Should be < 0.01 for a well-posed system.
        smoothness_cost: Value of the smoothness objective (delta^T H delta).
        converged: ``True`` if the KKT system was solved and the maximum
            residual is below tolerance.
    """

    calibrated_curve: pd.Series
    delta: pd.Series
    residuals: pd.DataFrame
    max_abs_residual: float
    smoothness_cost: float
    converged: bool


# ---------------------------------------------------------------------------
# Peak-hour mask builder
# ---------------------------------------------------------------------------

def _build_peak_mask(
    index: pd.DatetimeIndex,
    peak_hours: tuple[int, int],
    ch_holiday_dates: set,
) -> np.ndarray:
    """Return a boolean array indicating peak timestamps.

    Peak is defined as hours ``[peak_hours[0], peak_hours[1])`` on Monday
    through Friday in the Europe/Zurich timezone, excluding Swiss public
    holidays.

    Args:
        index: UTC ``DatetimeIndex`` at 15-min frequency.
        peak_hours: ``(start_hour, end_hour)`` in local Zurich time. The
            interval is ``[start_hour, end_hour)`` — e.g. ``(8, 20)`` means
            08:00-19:45 inclusive.
        ch_holiday_dates: Set of ``datetime.date`` objects for Swiss public
            holidays over the relevant years.

    Returns:
        Boolean numpy array of shape ``(len(index),)``.
    """
    idx_zurich = index.tz_convert("Europe/Zurich")
    hour = idx_zurich.hour
    dow = idx_zurich.dayofweek  # 0=Mon .. 6=Sun
    dates = idx_zurich.date

    is_weekday = dow < 5
    is_peak_hour = (hour >= peak_hours[0]) & (hour < peak_hours[1])
    is_not_holiday = np.array([d not in ch_holiday_dates for d in dates])

    return is_weekday & is_peak_hour & is_not_holiday


def _get_ch_holidays(years: Sequence[int]) -> set:
    """Collect Swiss public holidays for the given years.

    Uses the ``holidays`` library with all cantons aggregated (national
    holidays). This is consistent with EEX Peak definitions.

    Args:
        years: Calendar years to cover.

    Returns:
        Set of ``datetime.date`` objects.
    """
    result: set = set()
    for y in years:
        result |= set(holidays.Switzerland(years=y).keys())
    return result


# ---------------------------------------------------------------------------
# Sparse matrix builders
# ---------------------------------------------------------------------------

def _build_smoothness_matrix(n: int, weight: float = 1.0) -> sp.csc_matrix:
    """Build the second-difference smoothness penalty matrix H.

    H = weight * D2^T @ D2 where D2 is the ``(n-2) x n`` second-difference
    operator.  The resulting H is symmetric positive semi-definite, banded
    with bandwidth 5.

    Args:
        n: Number of timestamps.
        weight: Multiplicative weight for the smoothness penalty.

    Returns:
        Sparse ``n x n`` CSC matrix.
    """
    if n < 3:
        # Trivial case — no curvature penalty possible
        return sp.csc_matrix((n, n))

    # Second-difference operator D2: shape (n-2, n)
    # D2[i,:] has [1, -2, 1] at columns [i, i+1, i+2]
    m = n - 2
    diag_main = np.ones(m) * (-2.0)
    diag_off = np.ones(m)

    # Build D2 as a sparse matrix
    data = np.concatenate([diag_off[:m], diag_main, diag_off[:m]])
    row = np.concatenate([np.arange(m), np.arange(m), np.arange(m)])
    col = np.concatenate([np.arange(m), np.arange(1, m + 1), np.arange(2, m + 2)])

    D2 = sp.csc_matrix((data, (row, col)), shape=(m, n))

    H = weight * (D2.T @ D2)
    return H


def _build_constraint_matrix(
    n: int,
    index: pd.DatetimeIndex,
    contracts: list[FuturesContract],
    peak_mask: np.ndarray,
) -> tuple[sp.csc_matrix, np.ndarray, list[str]]:
    """Build the constraint matrix A and RHS target vector.

    Each row of A corresponds to one futures contract.  For a base contract,
    A[i, t] = 1/n_i for every timestamp t in the delivery period.  For a
    peak contract, only peak timestamps are included.

    Args:
        n: Total number of timestamps.
        index: The full UTC ``DatetimeIndex``.
        contracts: Futures contracts to calibrate against.
        peak_mask: Boolean array of peak timestamps.

    Returns:
        Tuple of ``(A, target_prices, contract_names)`` where ``A`` is a
        sparse ``(M, n)`` CSC matrix, ``target_prices`` is a 1-D array of
        futures prices, and ``contract_names`` is the list of contract names
        that were actually included (skipping degenerate ones).
    """
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    targets: list[float] = []
    names: list[str] = []

    row_idx = 0
    for contract in contracts:
        # Find timestamps within [start, end) using direct comparison
        # (avoids asi8 vs .value unit mismatch across pandas versions)
        in_period = (index >= contract.start) & (index < contract.end)

        if contract.product_type == "Peak":
            in_period = in_period & peak_mask

        ts_indices = np.where(in_period)[0]
        n_i = len(ts_indices)

        if n_i == 0:
            logger.warning(
                "Contract %s has no matching timestamps in the curve — skipped.",
                contract.name,
            )
            continue

        rows.append(np.full(n_i, row_idx, dtype=np.int32))
        cols.append(ts_indices.astype(np.int32))
        vals.append(np.full(n_i, 1.0 / n_i))
        targets.append(contract.price)
        names.append(contract.name)
        row_idx += 1

    m = row_idx
    if m == 0:
        return sp.csc_matrix((0, n)), np.array([]), []

    all_rows = np.concatenate(rows)
    all_cols = np.concatenate(cols)
    all_vals = np.concatenate(vals)

    A = sp.csc_matrix((all_vals, (all_rows, all_cols)), shape=(m, n))
    return A, np.array(targets), names


# ---------------------------------------------------------------------------
# Main calibrator
# ---------------------------------------------------------------------------

class ArbitrageFreeCalibrator:
    """Calibrates a raw shape curve to exactly reprice futures contracts.

    Uses an additive maximum-smoothness correction: the calibrated curve is
    ``P(t) = S(t) + delta(t)`` where ``delta`` minimises the integrated
    squared second derivative subject to average-price constraints from
    traded futures.

    Args:
        smoothness_weight: Relative weight of the smoothness penalty.
            Larger values produce smoother corrections at the potential
            expense of repricing accuracy in over-determined systems.
            Default ``1.0``.
        peak_hours: ``(start, end)`` hours in Europe/Zurich timezone
            defining the peak window. Default ``(8, 20)`` which maps to
            08:00-19:45 inclusive.
        regularisation: Small diagonal perturbation added to H to ensure
            numerical stability when the system is rank-deficient.
            Default ``1e-8``.
        tol: Convergence tolerance on the maximum absolute repricing
            residual (EUR/MWh). Default ``0.01``.
    """

    def __init__(
        self,
        smoothness_weight: float = 1.0,
        peak_hours: tuple[int, int] = (8, 20),
        regularisation: float = 1e-8,
        tol: float = 0.01,
    ) -> None:
        self.smoothness_weight = smoothness_weight
        self.peak_hours = peak_hours
        self.regularisation = regularisation
        self.tol = tol

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def calibrate(
        self,
        raw_curve: pd.Series,
        contracts: list[FuturesContract],
    ) -> CalibrationResult:
        """Calibrate a raw shape curve to match futures prices.

        Args:
            raw_curve: Raw (un-calibrated) price curve with a
                ``DatetimeIndex`` at 15-min frequency in UTC and values
                in EUR/MWh.
            contracts: List of futures contracts to reprice.  Contracts
                whose delivery period falls outside the curve's date range
                are silently skipped.

        Returns:
            A ``CalibrationResult`` containing the calibrated curve,
            correction term, per-contract residuals, and diagnostics.

        Raises:
            ValueError: If ``raw_curve`` is empty or has no timezone.
        """
        # ── Input validation ──────────────────────────────────────────
        if raw_curve.empty:
            raise ValueError("raw_curve is empty.")
        if raw_curve.index.tz is None:
            raise ValueError("raw_curve index must be timezone-aware (UTC).")

        if not contracts:
            logger.warning("No contracts supplied — returning raw curve unchanged.")
            return self._trivial_result(raw_curve)

        index = raw_curve.index
        n = len(index)
        S = raw_curve.values.astype(np.float64)

        logger.info(
            "Calibration: %d timestamps (%.1f days), %d contracts.",
            n,
            n / 96,
            len(contracts),
        )

        # ── Peak mask ─────────────────────────────────────────────────
        years = sorted(set(index.tz_convert("Europe/Zurich").year))
        ch_hols = _get_ch_holidays(years)
        peak_mask = _build_peak_mask(index, self.peak_hours, ch_hols)

        logger.debug(
            "Peak timestamps: %d / %d (%.1f%%)",
            peak_mask.sum(),
            n,
            100.0 * peak_mask.sum() / n,
        )

        # ── Build matrices ────────────────────────────────────────────
        H = _build_smoothness_matrix(n, weight=self.smoothness_weight)
        A, target_prices, contract_names = _build_constraint_matrix(
            n, index, contracts, peak_mask,
        )

        m = A.shape[0]
        if m == 0:
            logger.warning(
                "All contracts were skipped — returning raw curve unchanged."
            )
            return self._trivial_result(raw_curve)

        logger.info("Constraint matrix A: %d constraints x %d variables.", m, n)

        # ── RHS: b_i = F_i - mean(S over delivery period i) ──────────
        mean_S = np.array((A @ S).flat)  # A @ S gives weighted averages
        b = target_prices - mean_S

        logger.debug(
            "Target residuals (F - mean_S): min=%.4f, max=%.4f, mean=%.4f",
            b.min(),
            b.max(),
            b.mean(),
        )

        # ── Regularise H ──────────────────────────────────────────────
        H_reg = H + self.regularisation * sp.eye(n, format="csc")

        # ── Solve via Schur complement ────────────────────────────────
        # From KKT:  H δ + Aᵀ λ = 0,  A δ = b
        # =>  δ = −H⁻¹ Aᵀ λ
        # =>  −A H⁻¹ Aᵀ λ = b   (Schur complement, M×M system)
        # =>  λ = −(A H⁻¹ Aᵀ)⁻¹ b
        # =>  δ = H⁻¹ Aᵀ (A H⁻¹ Aᵀ)⁻¹ b
        #
        # Since H_reg is SPD and banded, we factorise once and solve
        # M right-hand sides to build the Schur complement.
        delta, converged = self._solve_schur(H_reg, A, b, n, m)

        # ── Calibrated curve ──────────────────────────────────────────
        P = S + delta

        # ── Residuals ─────────────────────────────────────────────────
        achieved = np.array((A @ P).flat)
        abs_errors = np.abs(achieved - target_prices)
        max_abs_residual = float(abs_errors.max()) if len(abs_errors) > 0 else 0.0

        residuals_df = pd.DataFrame(
            {
                "contract": contract_names,
                "target": target_prices,
                "achieved": achieved,
                "abs_error": abs_errors,
            }
        )

        if max_abs_residual > self.tol:
            converged = False
            logger.warning(
                "Max residual %.6f EUR/MWh exceeds tolerance %.4f.",
                max_abs_residual,
                self.tol,
            )
        else:
            logger.info(
                "Calibration converged. Max residual: %.6f EUR/MWh.",
                max_abs_residual,
            )

        # ── Smoothness cost ───────────────────────────────────────────
        smoothness_cost = float(delta @ (H @ delta))

        # ── Log summary ───────────────────────────────────────────────
        logger.info(
            "delta stats: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
            delta.min(),
            delta.max(),
            delta.mean(),
            delta.std(),
        )
        logger.info("Smoothness cost: %.6f", smoothness_cost)

        for _, row in residuals_df.iterrows():
            logger.debug(
                "  %-20s target=%8.3f  achieved=%8.3f  err=%8.6f",
                row["contract"],
                row["target"],
                row["achieved"],
                row["abs_error"],
            )

        return CalibrationResult(
            calibrated_curve=pd.Series(P, index=index, name="price_calibrated"),
            delta=pd.Series(delta, index=index, name="delta"),
            residuals=residuals_df,
            max_abs_residual=max_abs_residual,
            smoothness_cost=smoothness_cost,
            converged=converged,
        )

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------

    def _solve_schur(
        self,
        H_reg: sp.csc_matrix,
        A: sp.csc_matrix,
        b: np.ndarray,
        n: int,
        m: int,
    ) -> tuple[np.ndarray, bool]:
        """Solve the constrained QP via Schur complement reduction.

        The KKT conditions yield:

            ``delta = H^{-1} A^T (A H^{-1} A^T)^{-1} b``

        Since ``H_reg`` is SPD and banded, we factorise it once and solve
        ``M`` back-substitutions to build the ``M x M`` dense Schur
        complement ``S = A H^{-1} A^T``.

        When the constraint matrix ``A`` is rank-deficient (e.g.
        overlapping monthly + quarterly + annual base contracts), ``S``
        will be singular. We handle this by:

        1. Detecting rank via SVD of ``S``.
        2. If full rank: direct solve with iterative refinement.
        3. If rank-deficient: least-squares solve via pseudoinverse.
           Residuals on redundant constraints reflect inconsistency in
           the input prices (not a solver failure).

        Args:
            H_reg: Regularised smoothness matrix, SPD, sparse CSC.
            A: Constraint matrix, sparse CSC, shape ``(M, N)``.
            b: Constraint RHS vector, length ``M``.
            n: Number of timestamps.
            m: Number of constraints.

        Returns:
            Tuple of ``(delta, converged)``.
        """
        from scipy.sparse.linalg import splu

        converged = True

        try:
            # Factorise H_reg (SPD, banded -> fast sparse LU)
            logger.debug("Factorising H_reg (%d x %d)...", n, n)
            H_factor = splu(H_reg.tocsc())

            # Compute H^{-1} A^T column by column (M solves, each O(N))
            A_t_dense = A.T.toarray()  # N x M (M is small, typically < 100)
            H_inv_At = np.zeros((n, m))
            for j in range(m):
                H_inv_At[:, j] = H_factor.solve(A_t_dense[:, j])

            # Schur complement: S = A H^{-1} A^T (M x M dense)
            S_mat = np.asarray(A @ H_inv_At)

            # Detect rank
            U, sigma, Vt = np.linalg.svd(S_mat, full_matrices=False)
            rank_tol = max(m, n) * np.finfo(float).eps * sigma[0]
            rank = int(np.sum(sigma > rank_tol))

            if rank < m:
                logger.warning(
                    "Constraint matrix is rank-deficient: rank %d / %d. "
                    "Overlapping contracts may have inconsistent prices; "
                    "using least-squares fit for redundant constraints.",
                    rank,
                    m,
                )
                # Pseudoinverse solve: minimises ||S lam + b||_2
                sigma_inv = np.zeros_like(sigma)
                sigma_inv[:rank] = 1.0 / sigma[:rank]
                lam = Vt.T @ (sigma_inv * (U.T @ (-b)))
                delta = H_inv_At @ (-lam)

                # Iterative refinement within the column space
                for iteration in range(10):
                    r = b - np.asarray(A @ delta).ravel()
                    # Project residual onto column space of S
                    r_proj = U[:, :rank] @ (U[:, :rank].T @ r)
                    max_r_proj = np.max(np.abs(r_proj))
                    if max_r_proj < 1e-10:
                        logger.debug(
                            "Rank-deficient refinement converged at "
                            "iteration %d.",
                            iteration,
                        )
                        break
                    d_lam = Vt[:rank, :].T @ (
                        (1.0 / sigma[:rank]) * (U[:, :rank].T @ (-r_proj))
                    )
                    delta += H_factor.solve(A_t_dense @ (-d_lam))

            else:
                # Full-rank: direct solve with iterative refinement
                lam = np.linalg.solve(S_mat, -b)
                delta = H_factor.solve(A_t_dense @ (-lam))

                for iteration in range(10):
                    r = b - np.asarray(A @ delta).ravel()
                    max_r = np.max(np.abs(r))
                    if max_r < 1e-10:
                        logger.debug(
                            "Iterative refinement converged at iteration %d "
                            "(max residual %.2e).",
                            iteration,
                            max_r,
                        )
                        break
                    d_lam = np.linalg.solve(S_mat, -r)
                    delta += H_factor.solve(A_t_dense @ (-d_lam))

            if np.any(np.isnan(delta)) or np.any(np.isinf(delta)):
                raise ValueError("Solution contains NaN/Inf.")

            logger.debug("Schur complement solve succeeded.")
            return delta, converged

        except Exception as exc:
            logger.error("Schur complement solve failed: %s", exc)
            converged = False
            return np.zeros(n), converged

    def _trivial_result(self, raw_curve: pd.Series) -> CalibrationResult:
        """Return a no-op calibration result when there are no constraints.

        Args:
            raw_curve: The original raw shape curve.

        Returns:
            ``CalibrationResult`` with zero correction.
        """
        n = len(raw_curve)
        return CalibrationResult(
            calibrated_curve=raw_curve.copy().rename("price_calibrated"),
            delta=pd.Series(
                np.zeros(n), index=raw_curve.index, name="delta"
            ),
            residuals=pd.DataFrame(
                columns=["contract", "target", "achieved", "abs_error"]
            ),
            max_abs_residual=0.0,
            smoothness_cost=0.0,
            converged=True,
        )
