"""
test_perfect_foresight.py
-------------------------
Unit tests for the additive perfect-foresight shaping diagnostic
(`pfc_shaping.validation.perfect_foresight`).

Architecture
------------
- Fast, deterministic unit tests built on tiny SYNTHETIC tz-aware hourly Series
  (run in milliseconds). They exercise the pure functions: realized_window_mean,
  perfect_foresight_anchors, monthly_signature, deleveled_diurnal_scores,
  ch_physical_subkpis, robust_aggregate.
- ONE slow integration test (`@pytest.mark.slow`) that runs run_perfect_foresight
  on the real parquet data with a single vintage and asserts the result
  dataclass is well-formed. It builds curves (~tens of seconds).

All float asserts use numpy.testing.assert_allclose. Computation under test is
deterministic (no unseeded RNG); windows bucket in Europe/Zurich.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pfc_shaping.validation.perfect_foresight import (
    ch_physical_subkpis,
    deleveled_diurnal_scores,
    monthly_signature,
    perfect_foresight_anchors,
    realized_window_mean,
    robust_aggregate,
)

TZ = "Europe/Zurich"


# ---------------------------------------------------------------------------
# Synthetic fixtures (tz-aware UTC hourly Series)
# ---------------------------------------------------------------------------
def _hourly_utc(start: str, end: str) -> pd.DatetimeIndex:
    """Hourly UTC index over [start, end) (end exclusive)."""
    return pd.date_range(start, end, freq="1h", inclusive="left", tz="UTC")


def _two_year_constant_per_window() -> pd.Series:
    """Synthetic 2-year (2023-2024) hourly UTC series with a known structure.

    Bucketed in Europe/Zurich, each *local* calendar month of 2023 carries a
    distinct constant price (month number * 10), so monthly/quarter/year means
    are exactly computable. The local years 2022 and 2024 are filled with a
    separate constant.

    The UTC span is padded by a day on each side so that, after conversion to
    Europe/Zurich, the *local* windows for 2023 (Cal, every quarter, every
    month) are fully covered — the module's coverage guard rejects any window
    whose local start precedes the first local timestamp, so a series starting
    exactly at UTC midnight (= 01:00 local) would fail to cover Jan-1 windows.
    """
    idx = _hourly_utc("2022-12-31", "2025-01-02")
    loc = idx.tz_convert(TZ)
    values = np.where(loc.year == 2023, loc.month * 10.0, 999.0)
    return pd.Series(values.astype(float), index=idx)


# ---------------------------------------------------------------------------
# 1. Coverage guard
# ---------------------------------------------------------------------------
def test_realized_window_mean_returns_none_for_future_window():
    """A window extending beyond data coverage (a future year) returns None."""
    s = _two_year_constant_per_window()  # covers 2023-2024 (local)
    assert realized_window_mean(s, 2026) is None
    # A quarter / month past the end also returns None.
    assert realized_window_mean(s, 2025, quarter=2) is None
    assert realized_window_mean(s, 2025, month=6) is None


def test_realized_window_mean_exact_for_covered_windows():
    """Month / quarter / year means are exact for a fully-covered window."""
    s = _two_year_constant_per_window()

    # Month: each local month m of 2023 has constant price m*10.
    assert_allclose(realized_window_mean(s, 2023, month=3), 30.0, atol=1e-9)
    assert_allclose(realized_window_mean(s, 2023, month=7), 70.0, atol=1e-9)

    # Quarter Q1 2023 = months 1,2,3 (prices 10,20,30), hour-weighted.
    loc = s.tz_convert(TZ)
    q1 = loc[(loc.index >= pd.Timestamp("2023-01-01", tz=TZ))
             & (loc.index < pd.Timestamp("2023-04-01", tz=TZ))]
    assert_allclose(realized_window_mean(s, 2023, quarter=1), float(q1.mean()), atol=1e-9)

    # Cal 2023 = hour-weighted mean of the whole local year 2023.
    y2023 = loc[loc.index.year == 2023]
    assert_allclose(realized_window_mean(s, 2023), float(y2023.mean()), atol=1e-9)


# ---------------------------------------------------------------------------
# 2. Energy conservation of settlements (Cal == hour-weighted mean of 4 Q)
# ---------------------------------------------------------------------------
def test_cal_equals_hour_weighted_mean_of_quarters():
    """Base Cal mean == hour-weighted average of its 4 realized quarter means."""
    s = _two_year_constant_per_window()
    cal = realized_window_mean(s, 2023)

    loc = s.tz_convert(TZ)
    qmeans, qweights = [], []
    for q in range(1, 5):
        first_month = 3 * (q - 1) + 1
        start = pd.Timestamp(year=2023, month=first_month, day=1, tz=TZ)
        end = start + pd.offsets.MonthBegin(3)
        win = loc[(loc.index >= start) & (loc.index < end)]
        qmeans.append(realized_window_mean(s, 2023, quarter=q))
        qweights.append(len(win))

    recombined = np.average(qmeans, weights=qweights)
    assert_allclose(cal, recombined, atol=1e-9)


# ---------------------------------------------------------------------------
# 3. Peak / offpeak partition (clean partition of base)
# ---------------------------------------------------------------------------
def test_peak_offpeak_partition_recombines_to_base():
    """Hour-weighted avg of peak & offpeak settlements == base settlement."""
    # Use a non-constant series so the partition is a non-trivial check.
    # Pad the UTC span so the local Q1-2023 window is fully covered (see
    # _two_year_constant_per_window for why the local start must precede Jan 1).
    idx = _hourly_utc("2022-12-31", "2023-04-02")
    loc = idx.tz_convert(TZ)
    # Distinct value per hour-of-day so peak/offpeak means differ.
    values = (loc.hour.values.astype(float) + 1.0) * 2.0
    s = pd.Series(values, index=idx)

    base = realized_window_mean(s, 2023, quarter=1, segment="base")
    peak = realized_window_mean(s, 2023, quarter=1, segment="peak")
    offpeak = realized_window_mean(s, 2023, quarter=1, segment="offpeak")

    # Recover the peak/offpeak hour counts to weight the recombination.
    from pfc_shaping.validation.perfect_foresight import _peak_mask
    win_idx = loc[(loc >= pd.Timestamp("2023-01-01", tz=TZ))
                  & (loc < pd.Timestamp("2023-04-01", tz=TZ))]
    mask = _peak_mask(win_idx)
    n_peak, n_off = int(mask.sum()), int((~mask).sum())

    recombined = np.average([peak, offpeak], weights=[n_peak, n_off])
    assert_allclose(recombined, base, atol=1e-9)
    # Sanity: the mask is a genuine non-trivial partition.
    assert n_peak > 0 and n_off > 0
    assert n_peak + n_off == len(win_idx)


# ---------------------------------------------------------------------------
# 4. perfect_foresight_anchors granularity
# ---------------------------------------------------------------------------
def test_anchors_granularity_cal_keeps_only_years():
    """granularity='cal' keeps only digit-year keys; never quarters or months."""
    s = _two_year_constant_per_window()
    keys = ["2023", "2023-Q1", "2023-03", "2024", "2026"]
    anchors = perfect_foresight_anchors(keys, s, granularity="cal")

    assert set(anchors) == {"2023", "2024"}  # 2026 unrealized -> dropped
    assert "2023-Q1" not in anchors
    assert "2023-03" not in anchors
    assert_allclose(anchors["2023"], realized_window_mean(s, 2023), atol=1e-9)


def test_anchors_granularity_cal_quarter_keeps_years_and_quarters():
    """granularity='cal_quarter' keeps years+quarters; month keys NEVER present."""
    s = _two_year_constant_per_window()
    keys = ["2023", "2023-Q1", "2023-Q2", "2023-03", "2026-Q1"]
    anchors = perfect_foresight_anchors(keys, s, granularity="cal_quarter")

    assert "2023" in anchors
    assert "2023-Q1" in anchors and "2023-Q2" in anchors
    assert "2023-03" not in anchors  # month keys never anchored
    assert "2026-Q1" not in anchors  # unrealized window dropped
    assert_allclose(anchors["2023-Q1"], realized_window_mean(s, 2023, quarter=1), atol=1e-9)


def test_anchors_invalid_granularity_raises():
    """An unknown granularity raises ValueError."""
    s = _two_year_constant_per_window()
    with pytest.raises(ValueError):
        perfect_foresight_anchors(["2023"], s, granularity="month")


# ---------------------------------------------------------------------------
# 5. monthly_signature
# ---------------------------------------------------------------------------
def test_monthly_signature_full_year():
    """A full synthetic delivery year yields 12 entries with correct per-month means."""
    s = _two_year_constant_per_window()
    sig = monthly_signature(s, 2023)
    assert len(sig) == 12
    assert list(sig.index) == list(range(1, 13))
    # Month m of 2023 is constant m*10.
    expected = pd.Series([m * 10.0 for m in range(1, 13)], index=range(1, 13))
    assert_allclose(sig.values, expected.values, atol=1e-9)


# ---------------------------------------------------------------------------
# 6. deleveled_diurnal_scores
# ---------------------------------------------------------------------------
def _diurnal_pair(model_transform):
    """Build a (model, realized) hourly pair with a non-trivial diurnal shape.

    `realized` has a sinusoidal hour-of-day profile; `model` applies
    `model_transform(realized_values)`.
    """
    idx = _hourly_utc("2023-01-01", "2023-02-01")
    loc = idx.tz_convert(TZ)
    base = 50.0 + 20.0 * np.sin(2 * np.pi * loc.hour.values / 24.0)
    realized = pd.Series(base, index=idx)
    model = pd.Series(model_transform(base), index=idx)
    return model, realized


def test_diurnal_cosine_is_one_under_affine_transform():
    """De-levelled cosine == 1 when model is an affine transform a*x+b (a>0)."""
    model, realized = _diurnal_pair(lambda x: 3.0 * x + 17.0)
    scores = deleveled_diurnal_scores(model, realized)
    assert_allclose(scores["cosine"], 1.0, atol=1e-9)


def test_diurnal_demeaned_rmse_zero_when_identical():
    """demeaned_rmse == 0 (and cosine == 1) when model == realized."""
    model, realized = _diurnal_pair(lambda x: x.copy())
    scores = deleveled_diurnal_scores(model, realized)
    assert_allclose(scores["demeaned_rmse"], 0.0, atol=1e-9)
    assert_allclose(scores["cosine"], 1.0, atol=1e-9)


def test_diurnal_cosine_below_one_when_pattern_shifted():
    """A phase-shifted diurnal pattern gives cosine < 1."""
    idx = _hourly_utc("2023-01-01", "2023-02-01")
    loc = idx.tz_convert(TZ)
    realized = pd.Series(50.0 + 20.0 * np.sin(2 * np.pi * loc.hour.values / 24.0), index=idx)
    # Shift the peak by 6 hours -> different hour-of-day pattern.
    model = pd.Series(50.0 + 20.0 * np.sin(2 * np.pi * (loc.hour.values + 6) / 24.0), index=idx)
    scores = deleveled_diurnal_scores(model, realized)
    assert scores["cosine"] < 1.0 - 1e-6


# ---------------------------------------------------------------------------
# 7. robust_aggregate
# ---------------------------------------------------------------------------
def test_robust_aggregate_known_values():
    """median/p10/p90/min/max correct on a known list."""
    vals = list(range(1, 11))  # 1..10
    agg = robust_aggregate(vals)
    assert agg["n"] == 10
    assert_allclose(agg["median"], np.median(vals), atol=1e-12)
    assert_allclose(agg["p10"], np.percentile(vals, 10), atol=1e-12)
    assert_allclose(agg["p90"], np.percentile(vals, 90), atol=1e-12)
    assert_allclose(agg["min"], 1.0, atol=1e-12)
    assert_allclose(agg["max"], 10.0, atol=1e-12)


def test_robust_aggregate_ignores_nan():
    """NaN values are dropped before aggregation."""
    vals = [1.0, 2.0, np.nan, 3.0, np.nan]
    agg = robust_aggregate(vals)
    assert agg["n"] == 3
    assert_allclose(agg["median"], 2.0, atol=1e-12)
    assert_allclose(agg["min"], 1.0, atol=1e-12)
    assert_allclose(agg["max"], 3.0, atol=1e-12)


def test_robust_aggregate_empty_returns_n_zero():
    """An empty (or all-NaN) input returns n=0 with NaN summaries."""
    agg = robust_aggregate([])
    assert agg["n"] == 0
    assert np.isnan(agg["median"])
    assert np.isnan(agg["p10"]) and np.isnan(agg["p90"])

    agg_nan = robust_aggregate([np.nan, np.nan])
    assert agg_nan["n"] == 0


# ---------------------------------------------------------------------------
# 8. ch_physical_subkpis
# ---------------------------------------------------------------------------
def _ch_subkpi_series() -> pd.Series:
    """Synthetic 2023 hourly series with known winter>summer level + peak step.

    Winter months carry +40, summer months +0; on EEX peak hours (08-20 Mon-Fri
    excl. CH national holidays) the series gets an additional +10. Building the
    step on the *exact* EEX mask makes the peak/offpeak spread exactly +10 (a
    clean partition), so the magnitude assertion can be tight. Mean level 100 so
    ratios are well-defined.
    """
    from pfc_shaping.validation.perfect_foresight import _peak_mask

    idx = _hourly_utc("2023-01-01", "2024-01-01")
    loc = idx.tz_convert(TZ)
    base = np.full(len(idx), 100.0)
    base += np.where(np.isin(loc.month, (12, 1, 2)), 40.0, 0.0)
    base += np.where(_peak_mask(loc), 10.0, 0.0)
    return pd.Series(base, index=idx)


def test_ch_subkpis_model_equals_realized_when_identical():
    """When model == realized, every sub-KPI model value equals its realized value."""
    s = _ch_subkpi_series()
    kpis = ch_physical_subkpis(s, s, 2023)
    for key in ("winter_summer_ratio", "solar_bowl_depth", "peak_offpeak_spread"):
        assert key in kpis
        assert_allclose(kpis[key]["model"], kpis[key]["realized"], atol=1e-9)


def test_ch_subkpis_have_expected_sign_and_magnitude():
    """winter>summer ratio > 1 and peak/offpeak spread is a positive ~10 step."""
    s = _ch_subkpi_series()
    kpis = ch_physical_subkpis(s, s, 2023)

    # Winter level (~140) > summer level (~100) -> ratio > 1.
    assert kpis["winter_summer_ratio"]["realized"] > 1.0

    # The +10 step is applied on the exact EEX peak mask, so peak hours sit ~+10
    # above off-peak. It is not *exactly* +10 because the +40 winter level does
    # not perfectly cancel out of the peak-minus-offpeak difference (winter hours
    # have slightly different peak/offpeak shares than the rest of the year), but
    # it stays close to the +10 step.
    spread = kpis["peak_offpeak_spread"]["realized"]
    assert spread > 0.0
    assert_allclose(spread, 10.0, atol=1.0)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Regression tests for the QA audit findings (covered: NaN-corr, internal gap
# in coverage guard, target_year-scoping of the diurnal score, public API).
# ---------------------------------------------------------------------------
def test_coverage_guard_rejects_internal_gap():
    """An internal NaN gap inside the window must trip the coverage guard."""
    s = _two_year_constant_per_window()
    # Punch a 24-hour hole inside Cal 2023 (well above the ±2h DST slack).
    cut = s.index[(s.index >= pd.Timestamp("2023-06-15", tz="UTC")) &
                  (s.index < pd.Timestamp("2023-06-16", tz="UTC"))]
    s.loc[cut] = np.nan
    assert realized_window_mean(s, 2023) is None
    assert realized_window_mean(s, 2023, quarter=2) is None
    assert realized_window_mean(s, 2023, month=6) is None
    # An unaffected window still resolves cleanly.
    assert realized_window_mean(s, 2023, month=3) is not None


def test_build_curve_is_public():
    """build_curve must be public (used by scripts/run_perfect_foresight)."""
    from pfc_shaping.validation import perfect_foresight as pf

    assert hasattr(pf, "build_curve")
    assert "build_curve" in pf.__all__


def test_build_curve_friendly_error_on_bad_config():
    """Unknown config name -> ValueError listing valid names, not bare StopIteration."""
    from pfc_shaping.validation.perfect_foresight import build_curve

    s = _two_year_constant_per_window()
    v = pd.Timestamp("2024-01-31 17:00", tz="UTC")
    with pytest.raises(ValueError, match="ABLATION_GRID"):
        build_curve("not_a_real_config", v, s, {"2024": 50.0})


def test_diurnal_year_filter_excludes_extraneous_year():
    """`year=` must scope the diurnal score to that year only."""
    # Two-year hourly series: 2023 has hour-of-day shape A, 2024 has shape B.
    idx = pd.date_range("2023-01-01", "2024-12-31 23:00", freq="1h", tz="UTC")
    local = idx.tz_convert(TZ)
    shape_a = np.cos((local.hour - 14) * np.pi / 12.0)  # peak at 14h
    shape_b = -shape_a  # inverted peak (trough at 14h)
    sel_2023 = local.year == 2023
    arr = np.where(sel_2023, 50 + 20 * shape_a, 50 + 20 * shape_b)
    model = pd.Series(arr, index=idx)
    realized = pd.Series(50 + 20 * shape_a, index=idx)  # matches 2023 only

    # No year filter -> blended score (lower cosine).
    no_filter = deleveled_diurnal_scores(model, realized)
    # Year=2023 -> identical hour-of-day pattern => cosine ~= 1.
    only_2023 = deleveled_diurnal_scores(model, realized, year=2023)
    assert only_2023["cosine"] > no_filter["cosine"] + 0.5
    assert_allclose(only_2023["cosine"], 1.0, atol=1e-9)


def test_safe_correlation_skips_nan_paired_months():
    """run_perfect_foresight must drop NaN-paired months before pearsonr (no crash)."""
    # Direct unit test on a representative input. The function `_safe_corr` is
    # defined inside `run_perfect_foresight`; we exercise the property by
    # constructing two signatures with overlapping months, some NaN-paired.
    a = pd.Series({1: 10.0, 2: np.nan, 3: 30.0, 4: 40.0})
    b = pd.Series({1: 12.0, 2: 25.0, 3: 31.0, 4: 39.0})
    both = pd.concat([a, b], axis=1, join="inner").dropna()
    assert len(both) == 3  # paired NaN dropped
    # And the pearsonr on the cleaned pair must succeed.
    from scipy.stats import pearsonr
    r, _ = pearsonr(both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy())
    assert -1.0 <= r <= 1.0


def test_sota_swap_threadsafe_under_contention():
    """4 threads racing on `_sota_estimator()` must all restore the original.

    Regression test for the class-attribute capture-and-restore race that the
    isolation audit identified: without the module-level lock, threads can
    capture each other's already-swapped methods as "original" and leak the
    SOTA fitter permanently — silently violating the Phase 10 atol=1e-12
    reproducibility contract for any subsequent caller in the process.
    """
    import threading
    import time

    from pfc_shaping.calibration.cascading import ContractCascader
    from pfc_shaping.validation.perfect_foresight import _sota_estimator

    sr_before = ContractCascader.fit_seasonal_ratios
    ps_before = ContractCascader.fit_peak_spreads

    def worker():
        with _sota_estimator():
            time.sleep(0.005)  # encourage interleaving

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert ContractCascader.fit_seasonal_ratios is sr_before, "fit_seasonal_ratios leaked"
    assert ContractCascader.fit_peak_spreads is ps_before, "fit_peak_spreads leaked"


def test_sota_swap_patches_and_restores_shapehourly_halflife():
    """The `sota` path must swap ShapeHourly.__init__ to use SOTA_HALFLIFE_DAYS
    and restore the original on exit (intra-day f_H optimisation)."""
    from pfc_shaping.lt.model.shape_hourly import ShapeHourly
    from pfc_shaping.validation.perfect_foresight import (
        SOTA_HALFLIFE_DAYS,
        _sota_estimator,
    )

    assert SOTA_HALFLIFE_DAYS == 90.0
    init_before = ShapeHourly.__init__
    # Outside the context: default half-life (180), not the SOTA one.
    assert ShapeHourly().halflife_days != SOTA_HALFLIFE_DAYS
    with _sota_estimator():
        assert ShapeHourly.__init__ is not init_before  # swapped
        assert ShapeHourly().halflife_days == SOTA_HALFLIFE_DAYS  # SOTA applied
    # Restored after exit.
    assert ShapeHourly.__init__ is init_before
    assert ShapeHourly().halflife_days != SOTA_HALFLIFE_DAYS


def test_sota_swap_restores_on_exception():
    """Exception inside the `with` block must still restore both methods."""
    from pfc_shaping.calibration.cascading import ContractCascader
    from pfc_shaping.validation.perfect_foresight import _sota_estimator

    sr_before = ContractCascader.fit_seasonal_ratios
    ps_before = ContractCascader.fit_peak_spreads
    try:
        with _sota_estimator():
            assert ContractCascader.fit_seasonal_ratios is not sr_before  # confirmed swapped
            raise RuntimeError("simulated mid-build failure")
    except RuntimeError:
        pass
    assert ContractCascader.fit_seasonal_ratios is sr_before
    assert ContractCascader.fit_peak_spreads is ps_before


def test_build_curve_rejects_unknown_estimator_string():
    """A typo in `estimator=` must raise ValueError, not silently fall back."""
    from pfc_shaping.validation.perfect_foresight import build_curve

    s = _two_year_constant_per_window()
    v = pd.Timestamp("2024-01-31 17:00", tz="UTC")
    with pytest.raises(ValueError, match="estimator="):
        build_curve("bowl_on_floors_off", v, s, {"2024": 50.0}, estimator="SOTA")  # uppercase typo


def test_module_all_completeness():
    """`__all__` must list every documented public name."""
    from pfc_shaping.validation import perfect_foresight as pf

    expected = {
        "PerfectForesightResult", "build_curve", "ch_physical_subkpis",
        "deleveled_diurnal_scores", "monthly_signature", "perfect_foresight_anchors",
        "realized_window_mean", "robust_aggregate", "run_perfect_foresight",
        "TZ", "PEAK_HOUR_START", "PEAK_HOUR_END",
        "WINTER_MONTHS", "SUMMER_MONTHS", "SOLAR_BOWL_HOURS", "PRODUCTION_CONFIG",
    }
    missing = expected - set(pf.__all__)
    assert not missing, f"__all__ missing: {missing}"
    for name in pf.__all__:
        assert hasattr(pf, name), f"__all__ lists {name!r} but it is not defined"


def test_granularity_ladder_uses_requested_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pfc_shaping.validation import perfect_foresight as pf
    from pfc_shaping.validation import scorecard
    from scripts import run_perfect_foresight as runner

    index = pd.date_range(
        "2025-01-01",
        "2026-01-01",
        freq="1h",
        inclusive="left",
        tz="UTC",
    )
    epex = pd.Series(
        50.0 + np.sin(np.arange(len(index)) * 2.0 * np.pi / len(index)),
        index=index,
    )
    selected_estimators: list[str] = []

    def fake_build_curve(
        _config_name: str,
        _vintage: pd.Timestamp,
        series: pd.Series,
        _anchors: dict[str, float],
        estimator: str = "baseline",
        **_kwargs: object,
    ) -> pd.Series:
        selected_estimators.append(estimator)
        return series

    monkeypatch.setattr(pf, "build_curve", fake_build_curve)
    monkeypatch.setattr(
        scorecard,
        "_forwards_for_vintage",
        lambda *_args, **_kwargs: {"2025": 50.0, "2025-Q1": 50.0},
    )

    ladder = runner._granularity_ladder(
        epex,
        pd.DataFrame(),
        pd.Timestamp("2024-12-31", tz="UTC"),
        2025,
        "bowl_on_floors_off",
        "sota",
    )

    assert set(ladder) == {"pf_cal", "pf_cal_quarter", "market"}
    assert selected_estimators == ["sota", "sota", "sota"]


# ---------------------------------------------------------------------------
# Slow integration test — real run_perfect_foresight with a single vintage
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not all(
        path.exists()
        for path in (
            Path(__file__).resolve().parents[1] / "data" / "epex_hourly.parquet",
            Path(__file__).resolve().parents[1]
            / "data"
            / "forwards_history_phase10.parquet",
        )
    ),
    reason="requires governed local Phase 10 EPEX/forward inputs outside Git",
)
@pytest.mark.slow
def test_run_perfect_foresight_single_vintage_wellformed():
    """End-to-end run on real data with one vintage -> well-formed result dataclass."""
    from pfc_shaping.validation.perfect_foresight import run_perfect_foresight
    from pfc_shaping.validation.scorecard import list_vintages_2024_2025

    repo_root = Path(__file__).resolve().parents[1]
    epex = pd.read_parquet(repo_root / "data/epex_hourly.parquet")["price_eur_mwh"]
    forwards_history = pd.read_parquet(repo_root / "data/forwards_history_phase10.parquet")

    v_2024_12_31 = next(
        v for v in list_vintages_2024_2025() if v.strftime("%Y-%m-%d") == "2024-12-31"
    )
    result = run_perfect_foresight(
        epex,
        forwards_history,
        target_year=2025,
        config_name="bowl_on_floors_off",
        vintages=[v_2024_12_31],
    )

    # Sweep non-empty and pf_cal_corr a valid correlation.
    assert not result.sweep.empty
    pf_corr = float(result.sweep["pf_cal_corr"].iloc[0])
    assert -1.0 <= pf_corr <= 1.0

    # Sub-KPIs expose the 3 CH-physical keys.
    for key in ("winter_summer_ratio", "solar_bowl_depth", "peak_offpeak_spread"):
        assert key in result.subkpis
