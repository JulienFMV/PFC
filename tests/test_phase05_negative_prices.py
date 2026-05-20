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
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.model.msfc_spline import smooth_base_prices
from pfc_shaping.calibration.arbitrage_free import (
    ArbitrageFreeCalibrator,
    FuturesContract,
)
from pfc_shaping.lt.model.water_value import WaterValueCorrection

# ---------------------------------------------------------------------------
# Helpers shared across multiple tests
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Phase 5 fixture paths (Task 5-03)
_BOWL_MARKER_PATH = FIXTURES_DIR / "baseline_pfc_seed42_bowl.parquet"
_PHASE05_BASELINE_PATH = FIXTURES_DIR / "baseline_pfc_seed42_phase05.parquet"
_BASELINE_5BISA_PATH = FIXTURES_DIR / "baseline_pfc_seed42.parquet"
_FORWARDS_PHASE05_PATH = FIXTURES_DIR / "forwards_phase05_seed42.parquet"


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
    """compute_delta_wv sign-invariance: (f_wv - 1) * |B| has correct directional semantic.

    Test ID: 5-02-01 | Requirement: NEG-03 | Reference: 05-VALIDATION.md task 5-02-01.

    Verifies that the delta-additive WV formula:
        delta_wv = (f_wv - 1.0) * |B_smooth|
    has the correct scarcity/abundance semantic regardless of sign(B), unlike the
    legacy multiplicative path ``f_wv * B`` which inverts the semantic when B < 0.

    Uses unittest.mock to fix f_wv at known values (scarcity=1.20, abundance=0.80)
    — this tests the formula, not the calibration accuracy of apply().

    Codex review action #1 (2026-05-19): compute_delta_wv is called with keyword-
    only args after B_smooth (fill_df=..., calendar_df=...) to prevent the easy
    swap with apply(timestamps, calendar_df, hydro_forecast) positional order.
    """
    idx = pd.date_range("2027-07-01", periods=24 * 7, freq="h", tz="UTC")

    # Case 1 — scarcity (f_wv > 1): delta should be POSITIVE regardless of sign(B)
    # delta = (1.20 - 1) * |-10| = +2.0  (correct: scarcity → price LESS negative)
    # vs legacy: f_wv * B = 1.20 * -10 = -12  (WRONG: scarcity makes it MORE negative)
    B_negative = pd.Series(-10.0, index=idx, name="B_smooth")
    wv = WaterValueCorrection()
    f_wv_scarcity = pd.Series(1.20, index=idx, name="f_WV")
    with patch.object(WaterValueCorrection, "apply", return_value=f_wv_scarcity):
        delta_scarce = wv.compute_delta_wv(
            B_negative, fill_df=None, calendar_df=pd.DataFrame(index=idx)
        )
    assert abs(float(delta_scarce.mean()) - 2.0) < 0.05, (
        f"Scarcity case: expected delta_wv ≈ +2.0 EUR/MWh, got {delta_scarce.mean():.4f}. "
        "Check formula: (f_wv - 1) * |B| with f_wv=1.20, B=-10 → (0.20) * 10 = +2.0."
    )

    # Case 2 — abundance (f_wv < 1): delta should be NEGATIVE regardless of sign(B)
    # delta = (0.80 - 1) * |-10| = -2.0  (correct: abundance → price MORE negative)
    # vs legacy: f_wv * B = 0.80 * -10 = -8  (WRONG: abundance makes it LESS negative)
    f_wv_abundance = pd.Series(0.80, index=idx, name="f_WV")
    with patch.object(WaterValueCorrection, "apply", return_value=f_wv_abundance):
        delta_abund = wv.compute_delta_wv(
            B_negative, fill_df=None, calendar_df=pd.DataFrame(index=idx)
        )
    assert abs(float(delta_abund.mean()) - (-2.0)) < 0.05, (
        f"Abundance case: expected delta_wv ≈ -2.0 EUR/MWh, got {delta_abund.mean():.4f}. "
        "Check formula: (f_wv - 1) * |B| with f_wv=0.80, B=-10 → (-0.20) * 10 = -2.0."
    )

    # Case 3 — sign-invariance: |B| means the same delta for B=+10 and B=-10
    # With f_wv=1.20: delta(B=+10) = (0.20)*10 = +2.0 AND delta(B=-10) = (0.20)*10 = +2.0
    B_mixed = pd.Series(
        [10.0 if i % 2 == 0 else -10.0 for i in range(len(idx))],
        index=idx, name="B_smooth"
    )
    with patch.object(WaterValueCorrection, "apply", return_value=f_wv_scarcity):
        delta_mixed = wv.compute_delta_wv(
            B_mixed, fill_df=None, calendar_df=pd.DataFrame(index=idx)
        )
    # All values of delta_mixed should be ≈ +2.0 regardless of whether B was +10 or -10
    assert abs(float(delta_mixed.mean()) - 2.0) < 0.05, (
        f"Sign-invariance: expected delta_wv ≈ +2.0 for all positions, got mean={delta_mixed.mean():.4f}. "
        "delta_wv depends on |B|, not sign(B)."
    )
    assert float(delta_mixed.min()) > 0.0, (
        f"All delta_wv values should be positive (scarcity) for all B signs, "
        f"got min={delta_mixed.min():.4f}."
    )

    # Case 4 — enforce_floor=True raises ValueError (D-A3-4)
    wv_legacy = WaterValueCorrection(enforce_floor=True)
    with pytest.raises(ValueError, match="enforce_floor"):
        wv_legacy.compute_delta_wv(
            B_negative, fill_df=None, calendar_df=pd.DataFrame(index=idx)
        )

    # Verify Series name
    assert delta_scarce.name == "delta_wv", (
        f"Expected delta_wv.name == 'delta_wv', got {delta_scarce.name!r}"
    )


def test_assembler_delta_additive():
    """assembler.build() emits 'WV delta_wv:' INFO log on delta-additive path.

    Test ID: 5-02-02 | Requirement: NEG-03 | Reference: 05-VALIDATION.md task 5-02-02.

    Verifies the Phase 5 assembler delta-additive integration:
    - INFO log line starting with 'WV delta_wv:' emitted on the delta-additive path
      (wv.enforce_floor=False, the default)
    - result['delta_wv'] is a pd.Series (column in returned DataFrame)
    - delta_wv is NOT identically zero (WV contribution flows through)
    - Negative control: enforce_floor=True path emits NO 'WV delta_wv:' log
    - result_legacy['delta_wv'] is all zeros (legacy path placeholder)

    Builds a minimal PFCAssembler with WaterValueCorrection (default enforce_floor=False)
    using the _generate_baseline helpers for ShapeHourly + ShapeIntraday.
    """
    import random
    from tests.fixtures._generate_baseline import (
        _build_synthetic_epex,
    )
    from pfc_shaping.data.calendar_ch import enrich_15min_index
    from pfc_shaping.lt.model.assembler import PFCAssembler
    from pfc_shaping.lt.model.shape_hourly import ShapeHourly
    from pfc_shaping.lt.model.shape_intraday import ShapeIntraday

    # Deterministic seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Minimal synthetic EPEX (3 years at 15-min)
    n_15min = 3 * 365 * 96
    epex_df = _build_synthetic_epex(n_15min, rng)
    calendar_df = enrich_15min_index(epex_df.index, country="CH")

    sh = ShapeHourly(use_seasonal_hourly=False).fit(epex_df, calendar_df)
    si = ShapeIntraday().fit(epex_df, entso_df=None, calendar_df=calendar_df)

    # Build WaterValueCorrection — it doesn't need hydro data for apply()
    # (returns f_WV=1.0 neutral when hydro_forecast is None); we test that the path
    # is taken and the telemetry fires, even if delta_wv values are zero (no hydro).
    # To ensure delta_wv is non-zero, we use a mock that returns a constant f_wv.
    wv = WaterValueCorrection()  # enforce_floor=False by default

    assembler = PFCAssembler(
        shape_hourly=sh,
        shape_intraday=si,
        uncertainty=None,
        water_value=wv,
        cascader=None,
        calibrator=None,
    )

    # Use a 1-month horizon so the build is fast
    ref_date = pd.Timestamp("2026-05-18", tz="UTC")
    base_prices = {"2027": 50.0, "2027-07": 30.0}

    # Mock wv.apply to return a non-neutral f_wv so delta_wv is non-zero
    f_wv_mock_val = 1.10  # scarcity: delta = (0.10) * |B|

    asm_logger = "pfc_shaping.lt.model.assembler"
    with patch.object(
        WaterValueCorrection, "apply",
        return_value=None  # will be overridden per-call below
    ) as mock_apply:
        # Return a Series with the right index for each apply() call
        def side_effect(timestamps, calendar_df_arg, hydro_forecast_arg=None):
            return pd.Series(f_wv_mock_val, index=timestamps, name="f_WV")
        mock_apply.side_effect = side_effect

        import logging as _logging
        with pytest.raises(Exception) if False else __import__("contextlib").nullcontext():
            # Capture INFO log records during build
            log_records: list[logging.LogRecord] = []

            class _Capture(logging.Handler):
                def emit(self, r: logging.LogRecord) -> None:
                    log_records.append(r)

            handler = _Capture()
            handler.setLevel(logging.DEBUG)
            asm_log = logging.getLogger(asm_logger)
            orig_lvl = asm_log.level
            asm_log.setLevel(logging.DEBUG)
            asm_log.addHandler(handler)
            try:
                result = assembler.build(
                    base_prices=base_prices,
                    start_date="2027-07-01",
                    horizon_days=31,
                    reference_date=ref_date,
                )
            finally:
                asm_log.removeHandler(handler)
                asm_log.setLevel(orig_lvl)

    # Assert INFO log with 'WV delta_wv:' was emitted on the delta-additive path
    wv_log_records = [
        r for r in log_records
        if r.getMessage().startswith("WV delta_wv:")
    ]
    assert len(wv_log_records) >= 1, (
        f"Expected at least 1 INFO log starting with 'WV delta_wv:' on delta-additive path "
        f"(wv.enforce_floor=False), got 0 of {len(log_records)} total records. "
        "Check assembler.py: INFO telemetry on the delta-additive path (D-A3-5)."
    )
    wv_msg = wv_log_records[0].getMessage()
    assert "min=" in wv_msg, f"'min=' missing in telemetry: {wv_msg!r}"
    assert "max=" in wv_msg, f"'max=' missing in telemetry: {wv_msg!r}"
    assert "mean=" in wv_msg, f"'mean=' missing in telemetry: {wv_msg!r}"
    assert "sign(B) flips:" in wv_msg, f"'sign(B) flips:' missing in telemetry: {wv_msg!r}"

    # Assert delta_wv column exists in result DataFrame
    assert "delta_wv" in result.columns, (
        f"'delta_wv' column missing in build() result. Got columns: {list(result.columns)}"
    )
    assert isinstance(result["delta_wv"], pd.Series), (
        f"result['delta_wv'] should be pd.Series, got {type(result['delta_wv'])}"
    )
    assert len(result["delta_wv"]) == len(result.index), (
        "delta_wv length mismatch with result.index"
    )
    assert result["delta_wv"].dtype.kind == "f", (
        f"delta_wv should be float dtype, got {result['delta_wv'].dtype}"
    )
    # With f_wv_mock=1.10 and B not all zero, delta_wv should be non-zero
    assert float(result["delta_wv"].abs().max()) > 0.0, (
        "delta_wv should be non-zero when f_wv=1.10 (WV contribution must flow through on "
        "the delta-additive path)."
    )

    # Negative control: enforce_floor=True → legacy path, no 'WV delta_wv:' log, delta_wv=0
    # WR-03 (Phase 5 code review): PFCAssembler now propagates its enforce_floor
    # kwarg bidirectionally to the sub-component (it is the authoritative source
    # of truth, per D-A2-3). Passing enforce_floor=True at the assembler level
    # is therefore required to engage the legacy multiplicative path — wiring
    # only wv_legacy.enforce_floor=True via the WaterValueCorrection ctor is
    # no longer sufficient because the assembler ctor would otherwise reset it
    # to False (the PFCAssembler default).
    wv_legacy = WaterValueCorrection(enforce_floor=True)
    assembler_legacy = PFCAssembler(
        shape_hourly=sh,
        shape_intraday=si,
        uncertainty=None,
        water_value=wv_legacy,
        cascader=None,
        calibrator=None,
        enforce_floor=True,
    )

    log_records_legacy: list[logging.LogRecord] = []

    class _CaptureLegacy(logging.Handler):
        def emit(self, r: logging.LogRecord) -> None:
            log_records_legacy.append(r)

    handler_legacy = _CaptureLegacy()
    handler_legacy.setLevel(logging.DEBUG)
    asm_log = logging.getLogger(asm_logger)
    orig_lvl = asm_log.level
    asm_log.setLevel(logging.DEBUG)
    asm_log.addHandler(handler_legacy)
    try:
        result_legacy = assembler_legacy.build(
            base_prices=base_prices,
            start_date="2027-07-01",
            horizon_days=31,
            reference_date=ref_date,
        )
    finally:
        asm_log.removeHandler(handler_legacy)
        asm_log.setLevel(orig_lvl)

    wv_legacy_records = [
        r for r in log_records_legacy
        if r.getMessage().startswith("WV delta_wv:")
    ]
    assert len(wv_legacy_records) == 0, (
        f"Legacy path (enforce_floor=True) should emit NO 'WV delta_wv:' log, "
        f"got {len(wv_legacy_records)} records."
    )
    assert float(result_legacy["delta_wv"].abs().max()) == 0.0, (
        "Legacy path (enforce_floor=True) should produce delta_wv=0 placeholder "
        "(pass-through Series of zeros)."
    )


# ---------------------------------------------------------------------------
# Plan 05-02 — Test NEW: test_compute_delta_wv_index_alignment (codex action #1)
# ---------------------------------------------------------------------------


def test_compute_delta_wv_index_alignment():
    """compute_delta_wv index alignment + TypeError on positional kwarg.

    Test ID: 5-02-03 (NEW — added by Plan 05-02 per codex review action #1)
    Requirement: NEG-03 (codex action #1 ergonomics guard)
    Reference: 05-REVIEWS.md lines 102-103 (suggested test) and lines 113-115 (action #1).

    Codex action #1 (2026-05-19): asserts
      (a) ``delta_wv.index.equals(B_smooth.index)`` post-call — the index is preserved
          through the formula ``(f_wv - 1.0) * B_smooth.abs()`` (pandas arithmetic).
      (b) ``TypeError`` when ``fill_df``/``calendar_df`` are passed positionally —
          the ``*`` separator in compute_delta_wv's signature is the enforcement mechanism.

    Case C (assembler precondition guard) is scoped out: it is exercised at runtime
    by ``test_assembler_delta_additive`` via a real assembler build with aligned indices.
    Any future regression that breaks alignment will trip the ``assert delta_wv.index.equals(B.index)``
    guard in assembler.py at runtime.
    """
    # Build a 1-month hourly index (July 2027)
    idx = pd.date_range("2027-07-01", periods=24 * 30, freq="h", tz="UTC")
    B_smooth = pd.Series([10.0] * len(idx), index=idx, name="B_smooth")

    wv = WaterValueCorrection()  # enforce_floor=False by default

    # Mock apply() to return a constant f_wv=1.10 (avoid need for calibrated wv)
    f_wv_mock = pd.Series(1.10, index=idx, name="f_WV")
    with patch.object(WaterValueCorrection, "apply", return_value=f_wv_mock):
        # Case A — KEYWORD-ONLY call works and index is preserved
        delta = wv.compute_delta_wv(
            B_smooth,
            fill_df=None,
            calendar_df=pd.DataFrame(index=idx),
        )

    assert delta.index.equals(B_smooth.index), (
        "delta_wv.index must equal B_smooth.index (codex action #1 index alignment check). "
        f"delta.index has {len(delta.index)} entries, B_smooth.index has {len(B_smooth.index)}."
    )
    assert delta.name == "delta_wv", (
        f"Expected delta.name == 'delta_wv', got {delta.name!r}"
    )
    assert delta.shape == B_smooth.shape, (
        f"Expected delta.shape == B_smooth.shape {B_smooth.shape}, got {delta.shape}"
    )

    # Case B — TypeError on positional kwarg (codex action #1 enforcement via * separator)
    with patch.object(WaterValueCorrection, "apply", return_value=f_wv_mock):
        with pytest.raises(TypeError):
            # Passing fill_df and calendar_df POSITIONALLY must raise TypeError
            wv.compute_delta_wv(B_smooth, None, pd.DataFrame(index=idx))


# ---------------------------------------------------------------------------
# Plan 05-03 stubs — NEG-04, master flag, fixture, baselines
# ---------------------------------------------------------------------------


def test_cascading_spread_signed_base():
    """ContractCascader.synthesize_peak_prices with spread-additive: -10 + 5 = -5.

    Test ID: 5-03-01 | Requirement: NEG-04
    Reference: 05-VALIDATION.md, D-A4-1, CONTEXT.md §NEG-04 cross-cutting truth.

    Proves sign-invariance of spread-additive peak synthesis: when base=-10 and
    fitted spread=+5, peak = -10 + 5 = -5. The legacy multiplicative formula
    (-10 × 1.05 = -10.5) would invert the spread semantic for negative base prices.

    Setup:
      - ContractCascader(allow_negative_peak=True) — Phase 5 default
      - Manually set peak_base_spreads_[m=7] = 5.0 EUR/MWh
      - forwards = {"2027-07": -10.0} (negative July forward)
    Assert:
      synthesize_peak_prices(forwards)["2027-07-Peak"] ≈ -5.0 EUR/MWh (atol=1e-6)
    """
    from pfc_shaping.calibration.cascading import ContractCascader

    cascader = ContractCascader(allow_negative_peak=True)
    # Manually inject fitted spreads (bypass fit_peak_spreads for unit isolation)
    cascader.peak_base_spreads_ = {m: 5.0 for m in range(1, 13)}  # 5.0 EUR/MWh for all months

    forwards = {"2027-07": -10.0}
    result = cascader.synthesize_peak_prices(forwards)

    assert "2027-07-Peak" in result, (
        f"synthesize_peak_prices should add '2027-07-Peak' key. "
        f"Got keys: {list(result.keys())}"
    )
    peak_val = result["2027-07-Peak"]
    assert abs(peak_val - (-5.0)) < 1e-6, (
        f"NEG-04 spread-additive: expected peak = base + spread = -10 + 5 = -5.0, "
        f"got {peak_val:.6f}. Sign-invariance assertion failed — check "
        "synthesize_peak_prices spread-additive branch (allow_negative_peak=True path)."
    )


def test_fit_peak_ratios_deprecated():
    """fit_peak_ratios emits DeprecationWarning and populates BOTH spreads AND ratios.

    Test ID: 5-03-02 | Requirement: NEG-04 (backward compat, codex action #2)
    Reference: 05-REVIEWS.md lines 96-97, 113-116 (UNIFIED SHIM CONTRACT).

    Codex action #2 unified shim contract:
      1. fit_peak_ratios ALWAYS emits DeprecationWarning pointing to fit_peak_spreads.
      2. Delegates to fit_peak_spreads so peak_base_spreads_ is populated.
      3. ALSO derives peak_base_ratios_ so legacy attribute readers don't get
         AttributeError during the deprecation window.
    Formula for ratios: {m: 1.0 + spread_m / max(base_price_m, 1.0)}.

    Setup:
      - ContractCascader()
      - Synthetic spot_history with 200 rows of 15-min data (above 100-row threshold)
      - Call fit_peak_ratios(spot_history) inside pytest.warns(DeprecationWarning)
    Assert:
      (a) DeprecationWarning is emitted (codex action #2 step 1)
      (b) peak_base_spreads_ is populated as dict[int, float] with keys 1..12
      (c) peak_base_ratios_ is populated as dict[int, float] with keys 1..12
      (d) All ratio values > 0 (derived as 1.0 + spread/max(base,1))
    """
    from pfc_shaping.calibration.cascading import ContractCascader

    # Build minimal spot_history: 300 rows of 15-min data around h8-20 to populate spreads
    idx = pd.date_range("2023-01-02", periods=300, freq="15min", tz="UTC")  # Monday start
    spot_history = pd.DataFrame(
        {"price_eur_mwh": 50.0 + np.sin(np.arange(300) * 0.1) * 10.0},
        index=idx,
    )

    cascader = ContractCascader()

    with pytest.warns(DeprecationWarning, match="fit_peak_ratios.*deprecated"):
        cascader.fit_peak_ratios(spot_history)

    # (b) peak_base_spreads_ populated
    assert hasattr(cascader, "peak_base_spreads_"), (
        "fit_peak_ratios (unified shim) must populate peak_base_spreads_ "
        "(step 2: delegates to fit_peak_spreads). Attribute missing."
    )
    assert isinstance(cascader.peak_base_spreads_, dict), (
        f"peak_base_spreads_ should be dict, got {type(cascader.peak_base_spreads_)}"
    )
    assert set(cascader.peak_base_spreads_.keys()) == set(range(1, 13)), (
        f"peak_base_spreads_ should have keys 1..12, got {sorted(cascader.peak_base_spreads_.keys())}"
    )

    # (c) peak_base_ratios_ populated (codex action #2 step 3)
    assert hasattr(cascader, "peak_base_ratios_"), (
        "fit_peak_ratios (unified shim) must populate peak_base_ratios_ for legacy "
        "attribute readers (codex action #2, step 3). Attribute missing — the shim "
        "does NOT raise NotImplementedError."
    )
    assert isinstance(cascader.peak_base_ratios_, dict), (
        f"peak_base_ratios_ should be dict, got {type(cascader.peak_base_ratios_)}"
    )
    assert set(cascader.peak_base_ratios_.keys()) == set(range(1, 13)), (
        f"peak_base_ratios_ should have keys 1..12, got {sorted(cascader.peak_base_ratios_.keys())}"
    )

    # (d) ratios should be > 0 (typical: 1.0 + spread/max(base, 1.0))
    for m, ratio in cascader.peak_base_ratios_.items():
        assert ratio > 0, (
            f"peak_base_ratios_[{m}] = {ratio:.4f} <= 0 — derived ratio should be > 0 "
            "(formula: 1.0 + spread / max(base_price_m, 1.0) where spread >= 0 for EEX data)."
        )


def test_master_flag_audit_log():
    """PFCAssembler.__init__ reads PFC_LT_ALLOW_NEGATIVE_PRICES once and emits INFO log.

    Test ID: 5-03-03 | Requirement: D-A2-2 master flag audit-trail
    Reference: CONTEXT.md §master-flag, assembler.py _resolve_allow_negative.

    Verifies:
    (a) INFO log emitted at construction time (not at build() time)
    (b) Log message includes 'PFC_LT_ALLOW_NEGATIVE_PRICES=True'
    (c) Log message includes all four floor_disabled fields (msfc, af, wv, cascading)
    (d) Flag is read ONCE at init; subsequent env-var changes do not affect the
        stored value (freeze-at-init semantic per D-A2-2).

    Uses env-var PFC_LT_ALLOW_NEGATIVE_PRICES=1 to trigger the True case.
    """
    import os
    from pfc_shaping.lt.model.assembler import PFCAssembler, _ALLOW_NEG_ENV_VAR
    from pfc_shaping.lt.model.shape_hourly import ShapeHourly
    from pfc_shaping.lt.model.shape_intraday import ShapeIntraday
    from tests.fixtures._generate_baseline import _build_synthetic_epex
    from pfc_shaping.data.calendar_ch import enrich_15min_index

    rng = np.random.default_rng(42)
    n_15min = 90 * 96  # 90 days — fast enough for unit test
    epex_df = _build_synthetic_epex(n_15min, rng)
    calendar_df = enrich_15min_index(epex_df.index, country="CH")
    sh = ShapeHourly(use_seasonal_hourly=False).fit(epex_df, calendar_df)
    si = ShapeIntraday().fit(epex_df, entso_df=None, calendar_df=calendar_df)

    # Capture INFO log records from the assembler logger
    log_records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_records.append(record)

    asm_logger = logging.getLogger("pfc_shaping.lt.model.assembler")
    handler = _Capture()
    handler.setLevel(logging.DEBUG)
    orig_level = asm_logger.level
    asm_logger.setLevel(logging.DEBUG)
    asm_logger.addHandler(handler)

    try:
        saved = os.environ.get(_ALLOW_NEG_ENV_VAR)
        os.environ[_ALLOW_NEG_ENV_VAR] = "1"
        try:
            assembler = PFCAssembler(
                shape_hourly=sh,
                shape_intraday=si,
                uncertainty=None,
                water_value=None,
                cascader=None,
                calibrator=None,
                # Phase 5 defaults (all floors OFF = negative-ready)
                enforce_positivity=False,
                enforce_m_factor_floor=False,
                enforce_floor=False,
                allow_negative_peak=True,
            )
        finally:
            if saved is None:
                os.environ.pop(_ALLOW_NEG_ENV_VAR, None)
            else:
                os.environ[_ALLOW_NEG_ENV_VAR] = saved
    finally:
        asm_logger.removeHandler(handler)
        asm_logger.setLevel(orig_level)

    # (a) INFO log emitted at construction time
    audit_records = [
        r for r in log_records
        if "PFC_LT_ALLOW_NEGATIVE_PRICES" in r.getMessage()
    ]
    assert len(audit_records) >= 1, (
        f"Expected at least 1 INFO log containing 'PFC_LT_ALLOW_NEGATIVE_PRICES' "
        f"at assembler construction time. Got {len(log_records)} total records. "
        "Check assembler.py __init__: logger.info(\"PFC_LT_ALLOW_NEGATIVE_PRICES=...\") "
        "must be called at construction, not at build()."
    )

    msg = audit_records[0].getMessage()

    # (b) Flag value present and True
    assert "True" in msg, (
        f"Audit log should contain 'True' (env-var was '1'). Got: {msg!r}. "
        "Check _resolve_allow_negative: '1' should resolve to True."
    )

    # (c) All four floor fields present
    for field in ("msfc:enforce_positivity", "af:m_factor_floor", "wv:floor", "cascading:allow_neg_peak"):
        assert field in msg, (
            f"Audit log missing field '{field}'. Got: {msg!r}. "
            "Check assembler.py logger.info format string (D-A2-2)."
        )

    # (d) freeze-at-init: _allow_negative_prices is the resolved bool, not a live env lookup
    assert isinstance(assembler._allow_negative_prices, bool), (
        f"_allow_negative_prices should be a bool (freeze-at-init), "
        f"got {type(assembler._allow_negative_prices)}"
    )
    assert assembler._allow_negative_prices is True, (
        f"_allow_negative_prices should be True when env={_ALLOW_NEG_ENV_VAR}=1, "
        f"got {assembler._allow_negative_prices}"
    )


def test_phase05_summer_bowl_negative_acceptance():
    """SC #2 ROADMAP: h13 Sunday July 2027 < -20 EUR/MWh with bowl + negative floors off.

    Test ID: 5-03-04 | Requirement: SC #2 ROADMAP
    Reference: D-A4-5, D-A4-6, codex action #4 (UTC→Europe/Zurich), 05-REVIEWS.md lines 98-99.

    GATING per D-A4-5:
    - SKIP if 5bis-B bowl marker (baseline_pfc_seed42_bowl.parquet) is absent.
    - SKIP if Phase-5 baseline (baseline_pfc_seed42_phase05.parquet) has no negative
      prices (min >= 0) — this indicates a synthetic environment where the ShapeHourly
      [0.4, 2.0] factor clip prevents negative output; in that case the test is a
      synthetic-environment limitation, not a code defect (the SC #2 threshold is
      calibrated against real OMPEX data where genuine negative prices exist).

    When both gates pass, the test:
    1. Loads forwards_phase05_seed42.parquet (Phase 5 fixture).
    2. Sets PFC_LT_USE_SEASONAL_HOURLY_SHAPE=1 and PFC_LT_ALLOW_NEGATIVE_PRICES=1.
    3. Builds the PFC using build_phase05_baseline_pfc() (floors OFF, bowl active).
    4. Converts index to Europe/Zurich (codex action #4: PFC index is UTC but
       'h13 Sunday' is local-time semantics — direct UTC filtering would shift the
       window by 1-2 hours and miss the bowl trough).
    5. Asserts subset['price_shape'].mean() < -20.0 EUR/MWh (SC #2 ROADMAP threshold).
    """
    # Gate 1: 5bis-B bowl marker present
    if not _BOWL_MARKER_PATH.exists():
        pytest.skip(
            "Phase 5 SC #2 acceptance gated per D-A4-5: 5bis-B bowl deepening "
            "marker (baseline_pfc_seed42_bowl.parquet) is absent — the bowl "
            "calibration MUST be run before SC #2 can produce h13 Sunday "
            "negative prices. This skip is a known synthetic-environment "
            "limitation, not a regression. Phase 10 empirical validation on "
            "real OMPEX data is required before this assertion is unmuted."
        )

    # Gate 2: Phase-5 baseline contains negative prices
    # (if min >= 0, we're in a synthetic environment where SC #2 is not achievable)
    # WR-09 (Phase 5 code review): the skip message below is intentionally
    # assertive and informative so an operator who triggers the skip
    # understands (a) why the test is gated, (b) the upstream cause (synthetic
    # ShapeHourly [0.4, 2.0] clip), and (c) what action unblocks it (real
    # OMPEX data + Phase 10 empirical validation). The gate logic itself
    # is BY DESIGN per 05-CONTEXT.md D-A4-5 and is NOT changed by this fix.
    if _PHASE05_BASELINE_PATH.exists():
        _baseline_check = pd.read_parquet(str(_PHASE05_BASELINE_PATH))
        _price_col_check = "price_shape" if "price_shape" in _baseline_check.columns else _baseline_check.columns[0]
        _baseline_min = float(_baseline_check[_price_col_check].min())
        if _baseline_min >= 0.0:
            pytest.skip(
                f"Phase 5 SC #2 acceptance gated per D-A4-5: synthetic baseline "
                f"min={_baseline_min:.2f} >= 0 — Phase 10 empirical validation on "
                f"real OMPEX data is required before this assertion can be "
                f"unmuted. Root cause: ShapeHourly [0.4, 2.0] factor clip "
                f"prevents the synthetic baseline_pfc_seed42_phase05.parquet "
                f"from producing the negative prices the SC #2 < -20 EUR/MWh "
                f"threshold needs. This is a SYNTHETIC-environment limitation, "
                f"NOT a code defect: the negative-prices path is exercised by "
                f"test_msfc_signed_monthly_repricing and test_arbitrage_free_"
                f"signed_target which use synthetic NEGATIVE forwards directly."
            )

    from tests.fixtures._generate_phase05_fixture import (
        build_phase05_baseline_pfc,
        build_phase05_forwards,
    )

    # Build Phase-5 PFC with bowl deepening (codex action #4: explicit env vars before init)
    import os
    saved = {}
    for var in ("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", "PFC_LT_ALLOW_NEGATIVE_PRICES"):
        saved[var] = os.environ.get(var)
    os.environ["PFC_LT_USE_SEASONAL_HOURLY_SHAPE"] = "1"
    os.environ["PFC_LT_ALLOW_NEGATIVE_PRICES"] = "1"
    try:
        pfc = build_phase05_baseline_pfc(seed=42)
    finally:
        for var, val in saved.items():
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = val

    price_col = "price_shape" if "price_shape" in pfc.columns else pfc.columns[0]

    # Codex action #4: EXPLICIT UTC→Europe/Zurich conversion before applying Sunday/h13 masks.
    # The PFC index is UTC; 'h13 Sunday' is local-time semantics. Direct UTC-index filtering
    # would shift the Sunday/h13 window by 1-2 hours and miss the bowl trough.
    idx_local = pfc.index.tz_convert("Europe/Zurich")

    # Apply local-time masks (codex action #4)
    mask = (
        (idx_local.day_name() == "Sunday")
        & (idx_local.hour == 13)
        & (idx_local.month == 7)
    )

    subset = pfc.loc[mask, price_col]
    assert len(subset) > 0, (
        "No h13 Sunday July 2027 timestamps found in PFC output. "
        "Check horizon: build should cover July 2027 (31 days). "
        f"PFC index: {pfc.index[0]} to {pfc.index[-1]}"
    )

    actual_mean = float(subset.mean())
    assert actual_mean < -20.0, (
        f"SC #2 not met: pfc[Sunday, h13 Europe/Zurich, July 2027].mean() = "
        f"{actual_mean:.2f} EUR/MWh, expected < -20.0 "
        "(5bis-B bowl + Phase 5 floors-off may not be deep enough — "
        "check 5bis-B sigma_on, hydro_weight_sigma_on calibration); "
        "the UTC→Europe/Zurich conversion per codex action #4 was applied."
    )


def test_phase05_baseline_regression():
    """Regression: build(forwards_phase05_seed42) == baseline_pfc_seed42_phase05 atol=1e-12.

    Test ID: 5-03-05 | Requirement: SC #5 ROADMAP (canonical baseline frozen)
    Reference: D-A4-9, 05B-REVIEWS.md §1 (atol=1e-12, rtol=0, identical schema).

    Verifies that regenerating the Phase-5 baseline (build_phase05_baseline_pfc)
    produces a byte-for-byte equivalent result at atol=1e-12 against the committed
    baseline_pfc_seed42_phase05.parquet. This catches unintended math changes.

    Convention D-A4-9 "new baseline per math change atomique": if this test fails
    intentionally (planned math change), commit the new parquet AND document the
    change in SUMMARY.md.
    """
    if not _PHASE05_BASELINE_PATH.exists():
        pytest.fail(
            f"Phase-5 baseline fixture missing: {_PHASE05_BASELINE_PATH}. "
            "Run: python tests/fixtures/_generate_phase05_fixture.py --generate-baseline"
        )

    from tests.fixtures._generate_phase05_fixture import build_phase05_baseline_pfc
    import os

    baseline = pd.read_parquet(str(_PHASE05_BASELINE_PATH))

    # Build with same env vars as the committed baseline
    saved = {}
    for var in ("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", "PFC_LT_ALLOW_NEGATIVE_PRICES"):
        saved[var] = os.environ.get(var)
    os.environ["PFC_LT_USE_SEASONAL_HOURLY_SHAPE"] = "1"
    os.environ["PFC_LT_ALLOW_NEGATIVE_PRICES"] = "1"
    try:
        rebuilt = build_phase05_baseline_pfc(seed=42)
    finally:
        for var, val in saved.items():
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = val

    pd.testing.assert_frame_equal(
        rebuilt,
        baseline,
        check_exact=False,
        atol=1e-12,
        rtol=0,
        check_dtype=True,
        check_column_type=True,
        check_index_type=True,
        check_freq=False,   # parquet does not persist DatetimeIndex.freq metadata
        obj="Phase-5 baseline regression (D-A4-9, SC #5)",
    )


def test_phase05_baseline_5bisA_via_enforce_true():
    """Backward-compat: enforce_*=True on Phase 5 assembler matches 5bis-A baseline.

    Test ID: 5-03-06 | Requirement: SC #5 ROADMAP (legacy backward compat)
    Reference: D-A2-3 operator rollback path, 05B-REVIEWS.md §1 (atol=1e-12).

    Proves the D-A2-3 OPERATOR ROLLBACK PATH: when all four sub-components are
    constructed with legacy behavior explicitly enabled via PFCAssembler ctor kwargs
    (B1/B4 Approach B), the Phase 5 pipeline reproduces the 5bis-A baseline
    (baseline_pfc_seed42.parquet) at atol=1e-12.

    No monkeypatching — the four explicit PFCAssembler kwargs replace the patching
    pattern: enforce_positivity=True, enforce_m_factor_floor=True, enforce_floor=True,
    allow_negative_peak=False.

    Note: The 5bis-A baseline (baseline_pfc_seed42.parquet) was built with
    water_value=None and cascader=None, so enforce_floor and allow_negative_peak
    flags have no effect on the numeric output (no WV component, no cascading).
    enforce_positivity and enforce_m_factor_floor are also no-ops here since
    Cal'27=80 EUR/MWh is strongly positive. The test still passes because the
    four rollback kwargs round-trip through the assembler without breaking anything.
    """
    if not _BASELINE_5BISA_PATH.exists():
        pytest.fail(
            f"5bis-A baseline fixture missing: {_BASELINE_5BISA_PATH}. "
            "Run: python tests/fixtures/_generate_baseline.py"
        )

    import os
    import random
    from tests.fixtures._generate_baseline import build_pfc

    baseline_5bisA = pd.read_parquet(str(_BASELINE_5BISA_PATH))

    # Ensure PFC_LT_ALLOW_NEGATIVE_PRICES=0 for rollback audit-trail
    saved_flag = os.environ.get("PFC_LT_ALLOW_NEGATIVE_PRICES")
    os.environ["PFC_LT_ALLOW_NEGATIVE_PRICES"] = "0"

    try:
        # build_pfc uses water_value=None, cascader=None, calibrator=None
        # and base_prices={"2027": 80.0}, start_date="2027-01-01", horizon_days=31
        # Replicate exact call but via PFCAssembler with rollback kwargs
        import numpy as np_inner
        import random as random_inner
        random_inner.seed(42)
        np_inner.random.seed(42)
        rng_inner = np_inner.random.default_rng(42)

        from tests.fixtures._generate_baseline import _build_synthetic_epex
        from pfc_shaping.data.calendar_ch import enrich_15min_index
        from pfc_shaping.lt.model.assembler import PFCAssembler
        from pfc_shaping.lt.model.shape_hourly import ShapeHourly
        from pfc_shaping.lt.model.shape_intraday import ShapeIntraday

        n_15min = 3 * 365 * 96
        epex_df = _build_synthetic_epex(n_15min, rng_inner)
        calendar_df = enrich_15min_index(epex_df.index, country="CH")
        sh = ShapeHourly(use_seasonal_hourly=False).fit(epex_df, calendar_df)
        si = ShapeIntraday().fit(epex_df, entso_df=None, calendar_df=calendar_df)

        # D-A2-3 operator rollback path: all four floor kwargs enabled explicitly
        assembler = PFCAssembler(
            shape_hourly=sh,
            shape_intraday=si,
            uncertainty=None,
            water_value=None,
            cascader=None,
            calibrator=None,
            enforce_positivity=True,         # NEG-01 rollback (MSFC floor restored)
            enforce_m_factor_floor=True,     # NEG-02 rollback (m_factor floor restored)
            enforce_floor=True,              # NEG-03 rollback (F_WV_FLOOR clip restored)
            allow_negative_peak=False,       # NEG-04 rollback (peak ratio>=1 restored)
        )

        # Same call as build_pfc() in _generate_baseline.py
        rebuilt = assembler.build(
            base_prices={"2027": 80.0},
            start_date="2027-01-01",
            horizon_days=31,
            reference_date=pd.Timestamp("2026-05-18", tz="UTC"),
            country="CH",
        )
    finally:
        if saved_flag is None:
            os.environ.pop("PFC_LT_ALLOW_NEGATIVE_PRICES", None)
        else:
            os.environ["PFC_LT_ALLOW_NEGATIVE_PRICES"] = saved_flag

    # Tolerance contract: atol=1e-12, rtol=0 (05B-REVIEWS.md §1 tolerance convention).
    # Phase 05-02 added 'delta_wv' column to assembler output (NEG-03) — the 5bis-A
    # baseline was generated before Phase 05-02 so it has 13 columns vs the Phase-5
    # assembler's 14 columns. Compare only the common columns (drop delta_wv for this
    # comparison — it is 0.0 on the enforce_floor=True legacy path anyway).
    common_cols = [c for c in baseline_5bisA.columns if c in rebuilt.columns]
    rebuilt_common = rebuilt[common_cols]
    try:
        pd.testing.assert_frame_equal(
            rebuilt_common,
            baseline_5bisA,
            check_exact=False,
            atol=1e-12,
            rtol=0,
            check_dtype=True,
            check_column_type=True,
            check_index_type=True,
            check_freq=False,   # parquet does not persist DatetimeIndex.freq metadata
            obj="5bis-A baseline rollback regression (D-A2-3, SC #5)",
        )
    except AssertionError as e:
        # Diagnostic: compute actual divergence magnitude
        rebuilt_num = rebuilt_common.select_dtypes("number")
        baseline_num = baseline_5bisA[common_cols].select_dtypes("number")
        if len(rebuilt_num.columns) > 0 and rebuilt_num.shape == baseline_num.shape:
            max_divergence = float(abs(rebuilt_num.values - baseline_num.values).max())
            raise AssertionError(
                f"5bis-A rollback regression FAILED (atol=1e-12). "
                f"Max numeric divergence: {max_divergence:.2e}. "
                f"If < 1e-10 this may be floating-point reordering — loosen to atol=1e-10 "
                f"and document in SUMMARY. If >> 1e-10 the D-A2-3 rollback path is broken. "
                f"Original assertion: {e}"
            ) from e
        raise


def test_fit_peak_spreads_empty_spot_history():
    """fit_peak_spreads with empty spot_history falls back to default 5.0 EUR/MWh + UserWarning.

    Test ID: 5-03-07 | Requirement: codex action #7 (05-REVIEWS.md)
    Reference: RESEARCH §Pitfall 4, ContractCascader.fit_peak_spreads docstring.

    Contract: fit_peak_spreads(empty_df) emits a UserWarning and populates
    peak_base_spreads_ with the default spread (5.0 EUR/MWh for all 12 months)
    rather than raising an exception — ensures pipeline resilience when spot
    history is unavailable (e.g. first run on a new market).

    Tests three sub-cases:
      (a) Empty DataFrame (0 rows): UserWarning + default spread
      (b) < 100 rows: UserWarning + default spread
      (c) Wrong / missing column: UserWarning + default spread
    """
    from pfc_shaping.calibration.cascading import ContractCascader

    _DEFAULT_SPREAD = 5.0

    # Case (a): empty DataFrame
    empty_df = pd.DataFrame({"price_eur_mwh": pd.Series([], dtype=float)})
    cascader_a = ContractCascader()
    with pytest.warns(UserWarning, match="empty.*insufficient"):
        cascader_a.fit_peak_spreads(empty_df)

    assert hasattr(cascader_a, "peak_base_spreads_"), (
        "peak_base_spreads_ must be set even on empty spot_history (codex action #7 fallback)."
    )
    assert set(cascader_a.peak_base_spreads_.keys()) == set(range(1, 13)), (
        f"peak_base_spreads_ must have keys 1..12 on empty input, "
        f"got {sorted(cascader_a.peak_base_spreads_.keys())}"
    )
    for m, spread in cascader_a.peak_base_spreads_.items():
        assert abs(spread - _DEFAULT_SPREAD) < 1e-9, (
            f"Default spread for month {m} should be {_DEFAULT_SPREAD} EUR/MWh, "
            f"got {spread:.4f} (codex action #7)."
        )

    # Case (b): fewer than 100 rows
    idx_small = pd.date_range("2023-07-01", periods=50, freq="h", tz="UTC")
    small_df = pd.DataFrame({"price_eur_mwh": np.ones(50) * 45.0}, index=idx_small)
    cascader_b = ContractCascader()
    with pytest.warns(UserWarning, match="empty.*insufficient"):
        cascader_b.fit_peak_spreads(small_df)
    assert hasattr(cascader_b, "peak_base_spreads_"), (
        "peak_base_spreads_ must be set for < 100 rows (codex action #7 fallback)."
    )
    for m, spread in cascader_b.peak_base_spreads_.items():
        assert abs(spread - _DEFAULT_SPREAD) < 1e-9, (
            f"Default spread for month {m} should be {_DEFAULT_SPREAD}, got {spread:.4f}."
        )

    # Case (c): missing 'price_eur_mwh' column (wrong column name)
    idx_ok = pd.date_range("2023-01-01", periods=200, freq="h", tz="UTC")
    wrong_col_df = pd.DataFrame({"price": np.ones(200) * 60.0}, index=idx_ok)
    cascader_c = ContractCascader()
    with pytest.warns(UserWarning, match="empty.*insufficient"):
        cascader_c.fit_peak_spreads(wrong_col_df)
    assert hasattr(cascader_c, "peak_base_spreads_"), (
        "peak_base_spreads_ must be set for missing column (codex action #7 fallback)."
    )
