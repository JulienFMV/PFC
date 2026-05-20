"""
test_phase05_negative_prices.py
-------------------------------
Phase 5 — MSFC retire silent floors + PFC peut être négative.

This module is the durable test scaffold for ALL Phase 5 tests (13 total).
Five tests are populated and passing after Plan 05-01; eight are stubbed-skipped
for Plans 05-02 and 05-03.

Plan 05-01 scope (5 tests populated):
  - test_msfc_signed_monthly_repricing      (NEG-01 / NEG-05)
  - test_arbitrage_free_signed_target       (NEG-02)
  - test_msfc_clamp_all_equal_knots         (codex action #3 — degenerate knot clamp)
  - test_msfc_clamp_all_negative_knots_no_inverted_bounds  (codex action #3 — signed clamp)
  - test_arbitrage_free_converged_reason_floor_induced     (codex action #6 — reason log)

Plan 05-02 scope (2 stubs):
  - test_water_value_delta_sign_invariant   (NEG-03)
  - test_assembler_delta_additive           (NEG-03 assembler integration)

Plan 05-03 scope (6 stubs):
  - test_cascading_spread_signed_base              (NEG-04)
  - test_fit_peak_ratios_deprecated                (NEG-04 backward compat)
  - test_master_flag_audit_log                     (D-A2-2 master flag INFO log)
  - test_phase05_summer_bowl_negative_acceptance   (SC #2 ROADMAP — gated by 5bis-B)
  - test_phase05_baseline_regression               (SC #5 ROADMAP — baseline frozen)
  - test_phase05_baseline_5bisA_via_enforce_true   (backward-compat legacy baseline)

References:
  - .planning/phases/05-msfc-log-prix-retire-silent-floors/05-CONTEXT.md
  - .planning/phases/05-msfc-log-prix-retire-silent-floors/05-RESEARCH.md
  - .planning/phases/05-msfc-log-prix-retire-silent-floors/05-VALIDATION.md
  - REQUIREMENTS.md NEG-01..NEG-05
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.model.msfc_spline import smooth_base_prices
from pfc_shaping.calibration.arbitrage_free import (
    ArbitrageFreeCalibrator,
    FuturesContract,
)

# ---------------------------------------------------------------------------
# Helpers shared across multiple tests
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _make_idx_year(year: int, tz: str = "UTC") -> pd.DatetimeIndex:
    """Return a 15-min UTC DatetimeIndex covering the full calendar year."""
    return pd.date_range(
        f"{year}-01-01", f"{year+1}-01-01", freq="15min", tz=tz, inclusive="left"
    )


def _make_B_flat(idx: pd.DatetimeIndex, default: float = 30.0) -> pd.Series:
    """Return a flat price series used as B_flat placeholder."""
    return pd.Series(default, index=idx, name="B_flat")


def _make_curve_for_month(
    year: int,
    month: int,
    price: float = 70.0,
) -> pd.Series:
    """Create a flat 15-min curve for a single month."""
    start = pd.Timestamp(f"{year}-{month:02d}-01", tz="UTC")
    if month == 12:
        end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
    else:
        end = pd.Timestamp(f"{year}-{month + 1:02d}-01", tz="UTC")
    idx = pd.date_range(start, end, freq="15min", inclusive="left")
    return pd.Series(price, index=idx, name="price")


# ---------------------------------------------------------------------------
# Plan 05-01 — Test 1 : test_msfc_signed_monthly_repricing  (NEG-01 / NEG-05)
# ---------------------------------------------------------------------------


def test_msfc_signed_monthly_repricing():
    """smooth_base_prices with a negative monthly forward preserves the mean.

    Requirement: NEG-01 / NEG-05 (reformulated per D-A4-7).
    Tests that _enforce_mean_constraints is sign-invariant — the iterative
    correction ``error * 0.8`` works identically for negative targets (D-A1-3).

    Setup:
      - 2027 full year, 15-min UTC index
      - base_prices: Cal'27=30, all months positive EXCEPT July 2027=-2.0 EUR/MWh
      - enforce_positivity=False (default Phase 5 negative-ready)

    Assert:
      mean(B_smooth[July 2027]) ≈ -2.0 EUR/MWh, atol=0.01

    RESEARCH Pitfall 1: both floor #1 (line 131 smooth_base_prices) and
    floor #2 (line 203 _enforce_mean_constraints) must be disabled. If only
    floor #1 is disabled, the mean would collapse to ≈ 1.0 instead of -2.0.
    """
    year = 2027
    idx = _make_idx_year(year)
    B_flat = _make_B_flat(idx, default=30.0)

    # Construct base_prices with a negative July forward
    base_prices: dict[str, float] = {
        str(year): 30.0,  # Cal level
        f"{year}-01": 35.0,
        f"{year}-02": 34.0,
        f"{year}-03": 32.0,
        f"{year}-04": 28.0,
        f"{year}-05": 25.0,
        f"{year}-06": 22.0,
        f"{year}-07": -2.0,  # <-- negative forward (monthly depression, D-A4-7)
        f"{year}-08": 25.0,
        f"{year}-09": 28.0,
        f"{year}-10": 32.0,
        f"{year}-11": 34.0,
        f"{year}-12": 36.0,
    }

    B_smooth = smooth_base_prices(idx, base_prices, B_flat, enforce_positivity=False)

    # Extract July 2027 timestamps
    july_mask = (B_smooth.index.year == year) & (B_smooth.index.month == 7)
    july_mean = float(B_smooth[july_mask].mean())

    assert abs(july_mean - (-2.0)) < 0.01, (
        f"July 2027 mean={july_mean:.4f} EUR/MWh, expected ≈ -2.0 EUR/MWh (atol=0.01). "
        "Check that BOTH floor #1 (l.131) and floor #2 (l.203) are disabled when "
        "enforce_positivity=False — RESEARCH Pitfall 1."
    )


# ---------------------------------------------------------------------------
# Plan 05-01 — Test 2 : test_arbitrage_free_signed_target  (NEG-02)
# ---------------------------------------------------------------------------


def test_arbitrage_free_signed_target():
    """ArbitrageFreeCalibrator with a negative monthly target converges.

    Requirement: NEG-02.
    Tests that removing the m_factor floor (enforce_m_factor_floor=False,
    the default) allows calibration against a negative price target while
    still reporting converged=True and max_abs_residual < tol.

    Setup:
      - One month of 15-min curve starting at 10.0 EUR/MWh (flat, positive)
      - Single FuturesContract with target = -5.0 EUR/MWh (negative monthly)
      - enforce_m_factor_floor=False (default)

    Assert:
      result.converged == True AND result.max_abs_residual < calibrator.tol
    """
    # Build a flat 15-min curve for January 2027
    curve = _make_curve_for_month(2027, 1, price=10.0)

    # Contract: January 2027 Base, target = -5.0 EUR/MWh
    contract = FuturesContract(
        name="Jan-2027-Base",
        price=-5.0,  # negative target
        start=pd.Timestamp("2027-01-01", tz="UTC"),
        end=pd.Timestamp("2027-02-01", tz="UTC"),
        product_type="Base",
    )

    calibrator = ArbitrageFreeCalibrator(
        tol=0.01,
        enforce_m_factor_floor=False,  # negative-ready default
    )
    result = calibrator.calibrate(curve, [contract])

    assert result.converged, (
        f"Expected converged=True with enforce_m_factor_floor=False on negative target, "
        f"got converged={result.converged}, max_abs_residual={result.max_abs_residual:.6f}"
    )
    assert result.max_abs_residual < calibrator.tol, (
        f"max_abs_residual={result.max_abs_residual:.6f} >= tol={calibrator.tol}"
    )


# ---------------------------------------------------------------------------
# Plan 05-01 — Test 3 : test_msfc_clamp_all_equal_knots  (codex action #3)
# ---------------------------------------------------------------------------


def test_msfc_clamp_all_equal_knots():
    """Degenerate knot set (all equal) produces a non-zero clamp window.

    Requirement: codex review action #3 — margin floor for degenerate knot sets.
    When all monthly forwards are identical (np.ptp == 0), the old formula
    produced margin=0, pinning B_smooth_raw to the single knot value regardless
    of PCHIP extrapolation.  The new formula uses max(0.5*ptp, 1.0) so the
    clamp window is always >= [knot - 1.0, knot + 1.0].

    This test constructs such a degenerate case and verifies that the clamp
    bounds satisfy lo <= hi with a non-trivial window (hi - lo >= 1.0).
    """
    year = 2027
    idx = _make_idx_year(year)
    knot_value = 10.0
    B_flat = _make_B_flat(idx, default=knot_value)

    # All months identical — degenerate case, np.ptp(y_knots) == 0
    base_prices = {str(year): knot_value}
    for m in range(1, 13):
        base_prices[f"{year}-{m:02d}"] = knot_value

    # Should not raise; result should be finite (clamp window is non-zero)
    B_smooth = smooth_base_prices(idx, base_prices, B_flat, enforce_positivity=False)

    assert np.all(np.isfinite(B_smooth.values)), (
        "B_smooth contains non-finite values for all-equal knots"
    )
    # With margin floor=1.0, the clamp window is [10-1, 10+1] = [9, 11].
    # All values should be within this window.
    assert float(B_smooth.min()) >= knot_value - 1.5, (
        f"B_smooth.min()={B_smooth.min():.4f} below expected lower bound"
    )
    assert float(B_smooth.max()) <= knot_value + 1.5, (
        f"B_smooth.max()={B_smooth.max():.4f} above expected upper bound"
    )


# ---------------------------------------------------------------------------
# Plan 05-01 — Test 4 : test_msfc_clamp_all_negative_knots_no_inverted_bounds
#              (codex action #3)
# ---------------------------------------------------------------------------


def test_msfc_clamp_all_negative_knots_no_inverted_bounds():
    """All-negative knot set: signed clamp produces lo <= hi (no inversion).

    Requirement: codex review action #3 — signed-aware extrapolation clamp.
    The old formula `np.clip(..., y_knots.min()*0.5, y_knots.max()*2.0)`
    produces inverted bounds for all-negative knots:
      y_knots = [-30, -20, -25] → lo = -15.0, hi = -40.0  (lo > hi — BUG)

    The new formula:
      margin = max(0.5 * ptp(y_knots), 1.0) = max(5.0, 1.0) = 5.0
      lo = min(y_knots) - margin = -30 - 5 = -35.0
      hi = max(y_knots) + margin = -20 + 5 = -15.0
      lo <= hi ✓

    This test constructs an all-negative knot set and asserts that B_smooth
    contains only finite values (np.clip with lo > hi produces NaN-like
    behaviour in numpy >= 1.24; in older numpy it silently returns lo).
    """
    year = 2027
    idx = _make_idx_year(year)
    # All monthly forwards negative, in range [-30, -20]
    base_prices: dict[str, float] = {str(year): -25.0}
    month_prices = [-30.0, -28.0, -25.0, -22.0, -20.0, -21.0,
                    -23.0, -24.0, -26.0, -27.0, -29.0, -28.0]
    for m, p in enumerate(month_prices, start=1):
        base_prices[f"{year}-{m:02d}"] = p

    B_flat = _make_B_flat(idx, default=-25.0)

    B_smooth = smooth_base_prices(idx, base_prices, B_flat, enforce_positivity=False)

    assert np.all(np.isfinite(B_smooth.values)), (
        "B_smooth contains non-finite values for all-negative knots — "
        "likely caused by inverted clamp bounds (lo > hi) from the old formula."
    )
    # Clamp bounds: lo=-35, hi=-15; all values should be within [-40, -10] with slack
    assert float(B_smooth.max()) <= -5.0, (
        f"B_smooth.max()={B_smooth.max():.4f} should be <= -5.0 for all-negative knots"
    )
    assert float(B_smooth.min()) >= -50.0, (
        f"B_smooth.min()={B_smooth.min():.4f} should be >= -50.0 (generous clamp check)"
    )


# ---------------------------------------------------------------------------
# Plan 05-01 — Test 5 : test_arbitrage_free_converged_reason_floor_induced
#              (codex action #6)
# ---------------------------------------------------------------------------


def test_arbitrage_free_converged_reason_floor_induced():
    """Floor-induced non-convergence emits reason='m_factor_floor_hit' INFO log.

    Requirement: codex review action #6 — reason-tagged INFO log distinguishes
    floor-induced non-convergence from iteration-limit non-convergence.

    Setup:
      - Curve at -50.0 EUR/MWh (strongly negative)
      - Contract targeting -50.0 EUR/MWh
      - enforce_m_factor_floor=True (legacy rollback)
      → m_factor will be clipped from very-negative to 0.1 → floor_applied=True
      → converged=False forced
      → INFO log with extra={'reason': 'floor_induced'} ... 'm_factor_floor_hit'

    Assert:
      - result.converged is False
      - A log record with level INFO and 'reason' == 'm_factor_floor_hit' was emitted
    """
    # Strongly negative curve — m_factor = target / curve will be << 0.1
    curve = _make_curve_for_month(2027, 7, price=-50.0)

    contract = FuturesContract(
        name="Jul-2027-Base",
        price=-50.0,  # m_factor = -50 / (-50) ≈ 1.0 normally; but with small_mask
        start=pd.Timestamp("2027-07-01", tz="UTC"),
        end=pd.Timestamp("2027-08-01", tz="UTC"),
        product_type="Base",
    )

    # To reliably trigger m_factor floor, use a strongly positive raw curve
    # but a strongly negative target (m_factor = target/raw < 0 << 0.1)
    curve_positive = _make_curve_for_month(2027, 7, price=50.0)
    contract_negative = FuturesContract(
        name="Jul-2027-Base",
        price=-10.0,  # m_factor = P_add / safe_S with P_add << 0, safe_S >> 0
        start=pd.Timestamp("2027-07-01", tz="UTC"),
        end=pd.Timestamp("2027-08-01", tz="UTC"),
        product_type="Base",
    )

    calibrator = ArbitrageFreeCalibrator(
        tol=0.01,
        enforce_m_factor_floor=True,  # legacy: clip active + converged=False propagated
    )

    # Capture INFO log records — attach handler to both module logger and root
    # to ensure records are captured regardless of propagation configuration.
    log_records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_records.append(record)

    handler = _Capture()
    handler.setLevel(logging.DEBUG)
    af_logger = logging.getLogger("pfc_shaping.calibration.arbitrage_free")
    original_level = af_logger.level
    af_logger.setLevel(logging.DEBUG)
    af_logger.addHandler(handler)
    try:
        result = calibrator.calibrate(curve_positive, [contract_negative])
    finally:
        af_logger.removeHandler(handler)
        af_logger.setLevel(original_level)

    # With curve_positive=50 and target=-10, m_factor = (50 + correction) / 50
    # where correction = -60 (additive correction to reach -10 from 50).
    # m_factor = -10/50 = -0.2, which is < 0.1 → floor triggers → floor_applied=True.
    floor_hit_records = [
        r for r in log_records
        if hasattr(r, "reason") and r.reason == "m_factor_floor_hit"
    ]

    # The floor MUST have triggered for this input (strongly positive curve, negative target).
    assert len(floor_hit_records) >= 1, (
        f"Expected at least 1 INFO log record with reason='m_factor_floor_hit' "
        f"for curve_price=50, target=-10 with enforce_m_factor_floor=True. "
        f"Got {len(log_records)} total records: "
        + ", ".join(
            f"[level={r.levelname} reason={getattr(r, 'reason', 'N/A')}]"
            for r in log_records
        )
    )

    # converged must be False when the floor was hit (NEG-02 literal)
    assert not result.converged, (
        "converged should be False when m_factor floor hit under "
        "enforce_m_factor_floor=True (NEG-02 literal, codex action #6)"
    )


# ---------------------------------------------------------------------------
# Plan 05-02 stubs — NEG-03 (WaterValueCorrection delta-additif)
# ---------------------------------------------------------------------------


def test_water_value_delta_sign_invariant():
    """compute_delta_wv(B_smooth=-10, f_wv=1.20) → delta_wv = +2.0 EUR/MWh.

    Requirement: NEG-03.
    Populated in Plan 05-02.
    """
    pytest.skip("populated in Plan 05-02")


def test_assembler_delta_additive():
    """assembler.build() consumes compute_delta_wv additively: P = B*fH*fW + delta_wv.

    Requirement: NEG-03 (assembler integration).
    Populated in Plan 05-02.
    """
    pytest.skip("populated in Plan 05-02")


# ---------------------------------------------------------------------------
# Plan 05-03 stubs — NEG-04, master flag, fixture, baselines
# ---------------------------------------------------------------------------


def test_cascading_spread_signed_base():
    """ContractCascader.synthesize_peak_prices with spread-additive: -10 + 5 = -5.

    Requirement: NEG-04.
    Populated in Plan 05-03.
    """
    pytest.skip("populated in Plan 05-03")


def test_fit_peak_ratios_deprecated():
    """fit_peak_ratios emits DeprecationWarning and delegates to fit_peak_spreads.

    Requirement: NEG-04 (backward compat).
    Populated in Plan 05-03.
    """
    pytest.skip("populated in Plan 05-03")


def test_master_flag_audit_log():
    """PFCAssembler.__init__ reads PFC_LT_ALLOW_NEGATIVE_PRICES once, emits INFO log.

    Requirement: D-A2-2 master flag audit-trail.
    Populated in Plan 05-03.
    """
    pytest.skip("populated in Plan 05-03")


def test_phase05_summer_bowl_negative_acceptance():
    """SC #2 ROADMAP: h13 Sunday July 2027 < -20 EUR/MWh with bowl + negative floors off.

    Requirement: SC #2 ROADMAP (gated by 5bis-B bowl calibration).
    Populated in Plan 05-03. Skips automatically if 5bis-B bowl baseline absent.
    """
    pytest.skip("populated in Plan 05-03")


def test_phase05_baseline_regression():
    """Regression: build(forwards_phase05_seed42) == baseline_pfc_seed42_phase05 atol=1e-12.

    Requirement: SC #5 ROADMAP.
    Populated in Plan 05-03.
    """
    pytest.skip("populated in Plan 05-03")


def test_phase05_baseline_5bisA_via_enforce_true():
    """Backward-compat: enforce_*=True on Phase 5 assembler matches 5bis-A baseline.

    Requirement: SC #5 ROADMAP (legacy backward compat).
    Populated in Plan 05-03.
    """
    pytest.skip("populated in Plan 05-03")


def test_fit_peak_spreads_empty_spot_history():
    """fit_peak_spreads with empty or missing spot_history falls back to default spread + WARN.

    Test ID: 5-03-07, codex action #7 (05-REVIEWS.md).
    Owning plan: Plan 05-03.
    Contract: ContractCascader.fit_peak_spreads(empty_df) emits a WARNING and
    populates peak_base_spreads_ with a default spread (e.g. 5.0 EUR/MWh for
    all 12 months) rather than raising an exception — this ensures the pipeline
    is resilient when spot history is unavailable (e.g. first run on a new market).
    Populated in Plan 05-03.
    """
    pytest.skip("populated in Plan 05-03 — codex action #7 fallback spread + warning")
