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
    is_hard: bool = True
    penalty_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.product_type not in ("Base", "Peak", "Offpeak"):
            raise ValueError(
                f"product_type must be 'Base', 'Peak', or 'Offpeak', got {self.product_type!r}"
            )
        if self.start >= self.end:
            raise ValueError(
                f"Contract {self.name}: start ({self.start}) must be before "
                f"end ({self.end})"
            )
        if self.penalty_weight <= 0:
            raise ValueError(
                f"Contract {self.name}: penalty_weight must be positive, got {self.penalty_weight!r}"
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
    holiday_dates: set,
    tz: str = "Europe/Zurich",
) -> np.ndarray:
    """Return a boolean array indicating peak timestamps.

    Peak is defined as hours ``[peak_hours[0], peak_hours[1])`` on Monday
    through Friday in the supplied local timezone, excluding the supplied
    set of national public holidays.

    Args:
        index: UTC ``DatetimeIndex`` at 15-min frequency.
        peak_hours: ``(start_hour, end_hour)`` in the local timezone. The
            interval is ``[start_hour, end_hour)`` — e.g. ``(8, 20)`` means
            08:00-19:45 inclusive.
        holiday_dates: Set of ``datetime.date`` objects for the relevant
            country's public holidays over the years spanned by ``index``.
        tz: IANA timezone for the local-time peak window. Defaults to
            ``Europe/Zurich`` (CH/AT) — pass ``Europe/Berlin`` for DE,
            ``Europe/Paris`` for FR, ``Europe/Rome`` for IT.

    Returns:
        Boolean numpy array of shape ``(len(index),)``.
    """
    idx_local = index.tz_convert(tz)
    hour = idx_local.hour
    dow = idx_local.dayofweek  # 0=Mon .. 6=Sun
    dates = idx_local.date

    is_weekday = dow < 5
    is_peak_hour = (hour >= peak_hours[0]) & (hour < peak_hours[1])
    is_not_holiday = np.array([d not in holiday_dates for d in dates])

    return is_weekday & is_peak_hour & is_not_holiday


# Map ISO-2 market code → (IANA tz, ``holidays`` constructor). Adding a
# new market only requires extending this table. CH and AT share
# Europe/Zurich (CET/CEST) but their holiday calendars differ; FR uses
# Paris, IT uses Rome, DE uses Berlin. Order matters only for logging.
_COUNTRY_TZ: dict[str, str] = {
    "CH": "Europe/Zurich",
    "DE": "Europe/Berlin",
    "AT": "Europe/Vienna",
    "FR": "Europe/Paris",
    "IT": "Europe/Rome",
}


def _get_country_holidays(years: Sequence[int], country: str = "CH") -> set:
    """Collect public holidays for the requested country and years.

    Used by the arbitrage-free calibrator to build the EEX-style Peak
    mask. EEX Peak excludes national-level public holidays of the
    delivery zone, so the country argument matters: a DE Peak forward
    must NOT exclude Bundesfeier (1 August, CH-only) and a CH Peak
    forward must NOT exclude Tag der Deutschen Einheit (3 October,
    DE-only).

    Args:
        years: Calendar years to cover.
        country: ISO-2 market code (CH / DE / AT / FR / IT).

    Returns:
        Set of ``datetime.date`` objects.
    """
    code = country.upper()
    constructors: dict[str, callable] = {
        "CH": holidays.Switzerland,
        "DE": holidays.Germany,
        "AT": holidays.Austria,
        "FR": holidays.France,
        "IT": holidays.Italy,
    }
    if code not in constructors:
        raise ValueError(
            f"Unsupported country {country!r} for arbitrage-free calibration; "
            f"expected one of {sorted(constructors)}"
        )
    ctor = constructors[code]
    result: set = set()
    for y in years:
        result |= set(ctor(years=y).keys())
    return result


# Backward-compat shim: legacy callers (and tests) still import
# _get_ch_holidays. Keep it as a thin alias around the country-aware
# helper so old call sites keep working unchanged.
def _get_ch_holidays(years: Sequence[int]) -> set:
    return _get_country_holidays(years, country="CH")


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
) -> tuple[sp.csc_matrix, np.ndarray, list[str], np.ndarray, np.ndarray]:
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
    weights: list[float] = []
    hard_flags: list[bool] = []

    row_idx = 0
    for contract in contracts:
        # Find timestamps within [start, end) using direct comparison
        # (avoids asi8 vs .value unit mismatch across pandas versions)
        in_period = (index >= contract.start) & (index < contract.end)

        if contract.product_type == "Peak":
            in_period = in_period & peak_mask
        elif contract.product_type == "Offpeak":
            in_period = in_period & ~peak_mask

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
        weights.append(float(contract.penalty_weight))
        hard_flags.append(bool(contract.is_hard))
        row_idx += 1

    m = row_idx
    if m == 0:
        return sp.csc_matrix((0, n)), np.array([]), [], np.array([]), np.array([])

    all_rows = np.concatenate(rows)
    all_cols = np.concatenate(cols)
    all_vals = np.concatenate(vals)

    A = sp.csc_matrix((all_vals, (all_rows, all_cols)), shape=(m, n))
    return A, np.array(targets), names, np.array(weights, dtype=float), np.array(hard_flags, dtype=bool)


# ---------------------------------------------------------------------------
# Main calibrator
# ---------------------------------------------------------------------------

class ArbitrageFreeCalibrator:
    """Calibrates a raw shape curve to exactly reprice futures contracts.

    Supports two modes:

    **Additive** (classic MSFC):
        ``P(t) = S(t) + delta(t)``

    **Multiplicative** (SOTA, Kiesel-Paraschiv consistent):
        ``P(t) = S(t) * m(t)``
        where m(t) is the multiplicative correction factor. This preserves
        the multiplicative decomposition structure (B × f_S × f_W × f_H × f_Q)
        and guarantees positivity. The constraint is linear in m:
        ``(1/n_i) * sum_{t in period_i} S(t)*m(t) = F_i``
        so the same QP structure applies with weighted constraints.

    Args:
        smoothness_weight: Relative weight of the smoothness penalty.
        peak_hours: ``(start, end)`` hours in Europe/Zurich timezone.
        regularisation: Diagonal perturbation for numerical stability.
        tol: Convergence tolerance (EUR/MWh).
        mode: ``'additive'`` or ``'multiplicative'``. Default ``'multiplicative'``.
    """

    def __init__(
        self,
        smoothness_weight: float = 1.0,
        peak_hours: tuple[int, int] = (8, 20),
        regularisation: float = 1e-8,
        tol: float = 0.01,
        mode: str = "multiplicative",
    ) -> None:
        if mode not in ("additive", "multiplicative"):
            raise ValueError(f"mode must be 'additive' or 'multiplicative', got {mode!r}")
        self.smoothness_weight = smoothness_weight
        self.peak_hours = peak_hours
        self.regularisation = regularisation
        self.tol = tol
        self.mode = mode

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def calibrate(
        self,
        raw_curve: pd.Series,
        contracts: list[FuturesContract],
        country: str = "CH",
    ) -> CalibrationResult:
        """Calibrate a raw shape curve to match futures prices.

        Args:
            raw_curve: Raw (un-calibrated) price curve with a
                ``DatetimeIndex`` at 15-min frequency in UTC and values
                in EUR/MWh.
            contracts: List of futures contracts to reprice.  Contracts
                whose delivery period falls outside the curve's date range
                are silently skipped.
            country: ISO-2 market code (CH / DE / AT / FR / IT). Drives
                the local timezone of the EEX Peak window and the set of
                national public holidays excluded from Peak. Calibrating
                a DE forward with the default ``"CH"`` would silently
                use Bundesfeier as a non-peak day (CH-only) and treat
                3 October as a Peak day (DE-only) — both wrong by
                EEX PHELIX-Peak convention. The pipeline routes the
                spec's country code through the assembler.

        Returns:
            A ``CalibrationResult`` containing the calibrated curve,
            correction term, per-contract residuals, and diagnostics.

        Raises:
            ValueError: If ``raw_curve`` is empty or has no timezone, or
                if ``country`` is unknown.
        """
        # ── Input validation ──────────────────────────────────────────
        # Validate ``country`` BEFORE the empty-contracts shortcut so a
        # typo never silently falls back to CH.
        local_tz = _COUNTRY_TZ.get(country.upper())
        if local_tz is None:
            raise ValueError(
                f"Unsupported country {country!r}; expected one of {sorted(_COUNTRY_TZ)}"
            )

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
            "Calibration (%s, country=%s, tz=%s): %d timestamps (%.1f days), %d contracts.",
            self.mode,
            country.upper(),
            local_tz,
            n,
            n / 96,
            len(contracts),
        )

        # ── Peak mask ─────────────────────────────────────────────────
        years = sorted(set(index.tz_convert(local_tz).year))
        country_hols = _get_country_holidays(years, country=country)
        peak_mask = _build_peak_mask(index, self.peak_hours, country_hols, tz=local_tz)

        logger.debug(
            "Peak timestamps: %d / %d (%.1f%%)",
            peak_mask.sum(),
            n,
            100.0 * peak_mask.sum() / n,
        )

        # ── Build matrices ────────────────────────────────────────────
        H = _build_smoothness_matrix(n, weight=self.smoothness_weight)
        H_raw = H  # keep unscaled H for smoothness cost logging

        # TODO(P1-01): The "multiplicative" mode solves an additive QP and
        # converts delta to a multiplicative factor post-hoc.  The smoothness
        # penalty H = D2^T D2 penalises curvature in absolute price space,
        # so a 1 EUR correction at 200 EUR/MWh is penalised the same as at
        # 20 EUR/MWh.  Ideally the QP should be formulated in log-price or
        # relative space for true proportional smoothness.  As a partial
        # mitigation we scale H by the inverse price level so that corrections
        # at high-price timestamps are penalised less in absolute terms.
        if self.mode == "multiplicative":
            price_scale = np.maximum(np.abs(S), 1.0)
            inv_price = 1.0 / price_scale
            # H_scaled = diag(1/|S|) @ H @ diag(1/|S|)  — proportional smoothness
            inv_diag = sp.diags(inv_price, format="csc")
            H = inv_diag @ H @ inv_diag

        # Both modes use the same additive QP solve (numerically stable).
        # Multiplicative mode converts the correction post-hoc.
        A, target_prices, contract_names, contract_weights, hard_flags = _build_constraint_matrix(
            n, index, contracts, peak_mask,
        )
        m_constr = A.shape[0]
        if m_constr == 0:
            logger.warning("All contracts skipped — returning raw curve unchanged.")
            return self._trivial_result(raw_curve)

        logger.info("Constraint matrix A: %d x %d (mode=%s).", m_constr, n, self.mode)
        mean_S = np.array((A @ S).flat)
        b = target_prices - mean_S

        logger.debug(
            "Target residuals: min=%.4f, max=%.4f, mean=%.4f",
            b.min(),
            b.max(),
            b.mean(),
        )

        # ── Regularise H ──────────────────────────────────────────────
        H_reg = H + self.regularisation * sp.eye(n, format="csc")

        # ── Solve via Schur complement ────────────────────────────────
        correction, converged = self._solve_mixed_system(
            H_reg=H_reg,
            A=A,
            b=b,
            weights=contract_weights,
            hard_flags=hard_flags,
            n=n,
            m=m_constr,
        )

        # ── Calibrated curve ──────────────────────────────────────────
        if self.mode == "multiplicative":
            # Additive delta is solved; convert to multiplicative:
            # P_add = S + delta_add (reprices exactly)
            # m(t) = P_add / S = 1 + delta_add / S
            # This preserves exact repricing while interpreting the
            # correction as multiplicative (Kiesel-Paraschiv consistent).
            P_add = S + correction
            # Use multiplicative where S is large enough, additive otherwise
            small_mask = np.abs(S) < 1.0
            safe_S = np.where(small_mask, 1.0, S)
            m_factor = P_add / safe_S
            m_factor = np.maximum(m_factor, 0.1)
            # For near-zero S, use additive result directly (no div-by-zero)
            P = np.where(small_mask, P_add, S * m_factor)
            delta_for_log = m_factor - 1.0
            logger.info(
                "Multiplicative factor m(t): min=%.4f, max=%.4f, mean=%.4f",
                m_factor.min(), m_factor.max(), m_factor.mean(),
            )
        else:
            P = S + correction
            delta_for_log = correction

        # ── Residuals ─────────────────────────────────────────────────
        # Re-verify with standard (unweighted) constraint matrix
        A_check, _, _, _, _ = _build_constraint_matrix(n, index, contracts, peak_mask)
        achieved = np.array((A_check @ P).flat)
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
        try:
            smoothness_cost = float(delta_for_log @ (H_raw @ delta_for_log))
            if not np.isfinite(smoothness_cost):
                smoothness_cost = 0.0
        except Exception:
            smoothness_cost = 0.0

        # ── Log summary ───────────────────────────────────────────────
        if self.mode == "multiplicative":
            logger.info(
                "m(t) stats: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                m_factor.min(), m_factor.max(), m_factor.mean(), m_factor.std(),
            )
        else:
            logger.info(
                "delta stats: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                delta_for_log.min(), delta_for_log.max(),
                delta_for_log.mean(), delta_for_log.std(),
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
            delta=pd.Series(delta_for_log, index=index, name="delta"),
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
        import warnings

        from scipy.sparse.linalg import splu

        converged = True

        try:
            # Suppress overflow/divide warnings during iterative refinement
            # — NaN/Inf are caught explicitly below
            warnings.filterwarnings("ignore", category=RuntimeWarning)
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
                    if np.any(np.isnan(delta)) or np.any(np.isinf(delta)):
                        logger.warning("NaN/Inf during rank-def refinement iter %d — stopping.", iteration)
                        break
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
                    if np.any(np.isnan(delta)) or np.any(np.isinf(delta)):
                        logger.warning("NaN/Inf during refinement iter %d — stopping.", iteration)
                        break
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
        finally:
            warnings.resetwarnings()

    def _solve_mixed_system(
        self,
        H_reg: sp.csc_matrix,
        A: sp.csc_matrix,
        b: np.ndarray,
        weights: np.ndarray,
        hard_flags: np.ndarray,
        n: int,
        m: int,
    ) -> tuple[np.ndarray, bool]:
        """Solve a mixed hard/soft calibration system.

        Hard constraints are enforced exactly via KKT.
        Soft constraints enter as quadratic penalties:
            0.5 x'Hx + 0.5 ||W (A_soft x - b_soft)||^2
        """
        if len(hard_flags) != m:
            return np.zeros(n), False

        hard_idx = np.where(hard_flags)[0]
        soft_idx = np.where(~hard_flags)[0]

        if len(soft_idx) == 0:
            return self._solve_schur(H_reg, A, b, n, m)

        import warnings

        from scipy.sparse.linalg import splu

        converged = True

        try:
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            n_hard = len(hard_idx)
            n_soft = len(soft_idx)
            logger.info(
                "Mixed calibration solve: %d hard constraints, %d soft constraints.",
                n_hard,
                n_soft,
            )

            A_soft = A[soft_idx]
            b_soft = np.asarray(b[soft_idx], dtype=float)
            w_soft = np.asarray(weights[soft_idx], dtype=float)
            W_diag = np.square(w_soft)

            # Factorise the smoothness operator once, then work only in
            # contract space via Woodbury / Schur complements.
            H_factor = splu(H_reg.tocsc())

            A_soft_t_dense = A_soft.T.toarray()
            H_inv_AsT = np.zeros((n, n_soft))
            for j in range(n_soft):
                H_inv_AsT[:, j] = H_factor.solve(A_soft_t_dense[:, j])

            S_ss = np.asarray(A_soft @ H_inv_AsT, dtype=float)
            W_inv = np.diag(1.0 / W_diag)
            R = W_inv + S_ss
            R_inv = np.linalg.pinv(R, rcond=1e-12)

            Wb = W_diag * b_soft
            H_inv_rhs = H_inv_AsT @ Wb
            soft_rhs_proj = S_ss @ Wb
            H_eff_inv_rhs = H_inv_rhs - H_inv_AsT @ (R_inv @ soft_rhs_proj)

            if n_hard == 0:
                delta = H_eff_inv_rhs
                if np.any(np.isnan(delta)) or np.any(np.isinf(delta)):
                    raise ValueError("Soft-only mixed solution contains NaN/Inf.")
                return delta, converged

            A_hard = A[hard_idx]
            b_hard = np.asarray(b[hard_idx], dtype=float)
            A_hard_t_dense = A_hard.T.toarray()
            H_inv_AhT = np.zeros((n, n_hard))
            for j in range(n_hard):
                H_inv_AhT[:, j] = H_factor.solve(A_hard_t_dense[:, j])

            S_sh = np.asarray(A_soft @ H_inv_AhT, dtype=float)
            H_eff_inv_AhT = H_inv_AhT - H_inv_AsT @ (R_inv @ S_sh)

            K = np.asarray(A_hard @ H_eff_inv_AhT, dtype=float)
            rhs_lambda = np.asarray(A_hard @ H_eff_inv_rhs, dtype=float).ravel() - b_hard
            lambda_vec = np.linalg.pinv(K, rcond=1e-12) @ rhs_lambda
            delta = H_eff_inv_rhs - H_eff_inv_AhT @ lambda_vec

            if np.any(np.isnan(delta)) or np.any(np.isinf(delta)):
                raise ValueError("Mixed hard/soft solution contains NaN/Inf.")
            return np.asarray(delta, dtype=float).ravel(), converged
        except Exception as exc:
            logger.error("Mixed hard/soft solve failed: %s", exc)
            return np.zeros(n), False
        finally:
            warnings.resetwarnings()

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
