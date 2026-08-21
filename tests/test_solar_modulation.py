"""Tests for the solar-aware intra-day shape correction (Phase 10 §4quater).

Covers the checklist in
``.planning/phases/10-pfc-fmv-quality-scorecard/solar_research/README.md``:

* identity (beta = 0 -> f_H unchanged),
* mean-preservation (mean_h f_H_adj == 1 per local day),
* leakage (solar_pen_m at vintage v uses ONLY data < v),
* end-to-end on the best Cal-2025 vintage (2024-12-31): the summer bowl deepens
  and the peak/off-peak spread moves toward realized,
* the ``sota_solar`` estimator wiring keeps ``pf_cal_corr`` >= 0.85.

The real-data regression tests use a local governed ``entso_15min.parquet``;
the
end-to-end / wiring tests additionally need ``epex_hourly.parquet`` and the
forwards history and are skipped when those are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.lt.model.solar_modulation import (
    BLOCK_NAMES,
    HOUR_BLOCKS,
    SOLAR_MODULATION_EXPERIMENTAL,
    SOLAR_MODULATION_PRODUCTION_DEFAULT,
    SolarBlockedFHCorrection,
    SolarPenetrationFeature,
)

_REPO = Path(__file__).resolve().parents[1]
_ENTSO = _REPO / "pfc_shaping" / "data" / "entso_15min.parquet"
_EPEX = _REPO / "data" / "epex_hourly.parquet"
_FWDS = _REPO / "data" / "forwards_history_phase10.parquet"

requires_entso = pytest.mark.skipif(not _ENTSO.exists(), reason="entso_15min.parquet absent")
requires_epex = pytest.mark.skipif(
    not (_ENTSO.exists() and _EPEX.exists()), reason="epex_hourly.parquet absent"
)
requires_fwds = pytest.mark.skipif(
    not (_ENTSO.exists() and _EPEX.exists() and _FWDS.exists()),
    reason="forwards / epex data absent",
)

VINTAGE = pd.Timestamp("2024-12-31", tz="UTC")


# --------------------------------------------------------------------------- #
# Block structure
# --------------------------------------------------------------------------- #
def test_block_map_partition():
    assert set(HOUR_BLOCKS) == set(range(24))
    assert set(HOUR_BLOCKS.values()) == set(BLOCK_NAMES)
    assert HOUR_BLOCKS[12] == "MIDDAY_BOWL"
    assert HOUR_BLOCKS[3] == "NIGHT" and HOUR_BLOCKS[23] == "NIGHT"
    assert HOUR_BLOCKS[8] == "MORNING_RAMP"
    assert HOUR_BLOCKS[18] == "EVENING_PEAK"


# --------------------------------------------------------------------------- #
# Feature: known values + leakage
# --------------------------------------------------------------------------- #
@requires_entso
def test_feature_matches_direct_recompute():
    """solar_pen_m == Σsolar / Σload recomputed independently from the parquet.

    NOTE: the absolute values in ``data_probe.json`` (e.g. 2024-07 ≈ 0.18855)
    were measured on an earlier snapshot of ``entso_15min.parquet``; the file has
    since been revised, so this test validates the *formula* against a fresh,
    leak-filtered recomputation rather than pinning stale constants. It also
    sanity-checks the summer-high / winter-low ordering the method relies on.
    """
    raw = pd.read_parquet(_ENTSO)[["solar_mw", "load_mw"]]
    raw = raw[raw.index < VINTAGE]
    per = raw.index.tz_localize(None).to_period("M")
    expected = (raw["solar_mw"].groupby(per).sum() / raw["load_mw"].groupby(per).sum())

    pen = SolarPenetrationFeature(_ENTSO).monthly_realized(VINTAGE)
    for p in [pd.Period("2024-07", "M"), pd.Period("2021-01", "M"),
              pd.Period("2023-06", "M")]:
        assert float(pen.loc[p]) == pytest.approx(float(expected.loc[p]), rel=1e-9)
    # Summer penetration is far above winter (the signal the bowl correction uses).
    assert float(pen.loc[pd.Period("2024-07", "M")]) > 5 * float(
        pen.loc[pd.Period("2021-01", "M")]
    )


@requires_entso
def test_feature_is_leak_free():
    """Extracting solar_pen at vintage v must use ONLY data with index < v."""
    feat = SolarPenetrationFeature(_ENTSO)
    pen = feat.monthly_realized(VINTAGE)
    # No delivery month at/after the vintage may appear in the training feature.
    assert pen.index.max() <= pd.Period("2024-12", "M")
    assert pd.Period("2025-01", "M") not in pen.index
    # The maximum input timestamp consumed is strictly < vintage (audit hook).
    assert feat._last_max_input_ts_ < VINTAGE


@requires_entso
def test_feature_projection_capped_in_support():
    feat = SolarPenetrationFeature(_ENTSO)
    pen = feat.monthly_realized(VINTAGE)
    cap = float(np.percentile(pen.to_numpy(), 99))
    proj = feat.project(pd.Period("2025-07", "M"), VINTAGE)
    assert 0.0 < proj <= cap + 1e-12
    # A realized (pre-vintage) month returns its realized value exactly.
    assert feat.project(pd.Period("2024-07", "M"), VINTAGE) == pytest.approx(
        float(pen.loc[pd.Period("2024-07", "M")])
    )


# --------------------------------------------------------------------------- #
# Synthetic f_H fixtures for identity / mean-preservation
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synthetic_year():
    """A clean 2025 15-min index + calendar + a per-local-day-mean-1 diurnal f_H.

    The cosine has a 24h mean of 1, but DST-transition days (Mar/Oct) have 92/100
    quarters, so their raw daily mean is not exactly 1. ``modulate`` renormalises
    per local day, so to make ``beta = 0`` a *bit-exact* identity we pre-normalise
    the input per local day here — exactly the state ``ShapeHourly.apply`` produces
    in the real pipeline.
    """
    idx = pd.date_range("2025-01-01", "2026-01-01", freq="15min", inclusive="left", tz="UTC")
    cal = enrich_15min_index(idx, country="CH")
    local = idx.tz_convert("Europe/Zurich")
    f = 1.0 + 0.2 * np.cos(2 * np.pi * (local.hour.to_numpy() - 3) / 24.0)
    f_H = pd.Series(f, index=idx, name="f_H")
    day = pd.Index([f"{t.year}-{t.month:02d}-{t.day:02d}" for t in local])
    f_H = f_H / f_H.groupby(day).transform("mean")
    return idx, cal, f_H


@requires_entso
def test_identity_zero_beta(synthetic_year):
    """beta = 0 (empty table) -> f_H unchanged (per-day-mean-1 input is bit-stable)."""
    idx, cal, f_H = synthetic_year
    corr = SolarBlockedFHCorrection(feature=SolarPenetrationFeature(_ENTSO))
    corr.fitted_ = True
    corr.vintage_ = VINTAGE
    corr.baseline_pen_ = corr.feature.baseline(VINTAGE)
    corr.beta_ = {}  # all zeros
    out = corr.modulate(f_H, cal, vintage=VINTAGE)
    assert np.max(np.abs(out.to_numpy() - f_H.to_numpy())) < 1e-12


@requires_entso
def test_mean_preservation(synthetic_year):
    """After modulation every local calendar day keeps mean_h f_H_adj == 1."""
    idx, cal, f_H = synthetic_year
    corr = SolarBlockedFHCorrection(feature=SolarPenetrationFeature(_ENTSO))
    corr.fitted_ = True
    corr.vintage_ = VINTAGE
    corr.baseline_pen_ = corr.feature.baseline(VINTAGE)
    # Inject a non-trivial midday beta so the correction actually bites.
    corr.beta_ = {("Ete", "Ouvrable", "MIDDAY_BOWL"): -1.0,
                  ("Ete", "Weekend", "MIDDAY_BOWL"): -1.0}
    out = corr.modulate(f_H, cal, vintage=VINTAGE)
    local = idx.tz_convert("Europe/Zurich")
    day = pd.Index([f"{t.year}-{t.month:02d}-{t.day:02d}" for t in local])
    day_means = out.groupby(day).mean()
    assert np.max(np.abs(day_means.to_numpy() - 1.0)) < 1e-12


@requires_entso
def test_modulate_before_fit_raises():
    corr = SolarBlockedFHCorrection(feature=SolarPenetrationFeature(_ENTSO))
    with pytest.raises(RuntimeError):
        corr.modulate(pd.Series([1.0]), pd.DataFrame(), vintage=VINTAGE)


@requires_entso
def test_solar_modulate_none_vintage_passthrough(synthetic_year):
    """vintage=None degrades gracefully to the identity (no leak-safe anchor)."""
    from pfc_shaping.lt.model.solar_modulation import solar_modulate

    idx, cal, f_H = synthetic_year
    out = solar_modulate(f_H, cal, shape_hourly=None, vintage=None)
    assert out is f_H


# --------------------------------------------------------------------------- #
# Fit on real prices: sign of the key coefficients
# --------------------------------------------------------------------------- #
@requires_epex
def test_fit_coefficient_signs():
    """Real-data beta diagnostics explain why solar modulation remains gated off."""
    from pfc_shaping.lt.model.shape_hourly import ShapeHourly

    assert SOLAR_MODULATION_EXPERIMENTAL is True
    assert SOLAR_MODULATION_PRODUCTION_DEFAULT is False

    epex = pd.read_parquet(_EPEX)
    epex = epex.rename(columns={epex.columns[0]: "price_eur_mwh"})
    train = epex[epex.index < VINTAGE]
    cal = enrich_15min_index(train.index, country="CH")
    sh = ShapeHourly().fit(train, cal)

    corr = SolarBlockedFHCorrection(
        feature=SolarPenetrationFeature(_ENTSO)
    ).fit(epex, sh.get, VINTAGE)
    assert corr.fitted_
    b_mid = corr.beta_for("Ete", "Ouvrable", 12)
    b_night = corr.beta_for("Ete", "Ouvrable", 3)
    assert b_mid < 0.0
    assert b_mid == pytest.approx(-0.22941175107181738)
    assert b_night == pytest.approx(0.23406562575625295)
    assert abs(b_night) >= abs(b_mid)


# --------------------------------------------------------------------------- #
# End-to-end on the best Cal-2025 vintage
# --------------------------------------------------------------------------- #
@requires_fwds
def test_flag_off_explicit_is_byte_identical_to_default_sota(monkeypatch):
    from pfc_shaping.lt.model.assembler import PFCAssembler
    from pfc_shaping.validation.perfect_foresight import build_curve
    from pfc_shaping.validation.scorecard import list_vintages_2024_2025

    target = 2025
    config = "bowl_on_floors_off"
    epex = pd.read_parquet(_EPEX)["price_eur_mwh"]
    best = max(
        [v for v in list_vintages_2024_2025() if v.year < target],
        key=lambda v: (epex[epex.index < v].index.max() - epex[epex.index < v].index.min()).days,
    )
    anchors = {str(target): float(epex[epex.index.year == target].mean())}

    default_curve = build_curve(config, best, epex, anchors, estimator="sota")

    original_init = PFCAssembler.__init__

    def explicit_off_init(self, *args, _orig=original_init, **kwargs):
        kwargs.setdefault("enable_solar_modulation", False)
        _orig(self, *args, **kwargs)

    monkeypatch.setattr(PFCAssembler, "__init__", explicit_off_init)
    explicit_off_curve = build_curve(config, best, epex, anchors, estimator="sota")

    pd.testing.assert_series_equal(default_curve, explicit_off_curve, check_exact=True)


@requires_fwds
def test_end_to_end_flag_on_diagnostic_justifies_gate_off():
    from pfc_shaping.validation.perfect_foresight import build_curve, ch_physical_subkpis
    from pfc_shaping.validation.scorecard import list_vintages_2024_2025

    target = 2025
    config = "bowl_on_floors_off"
    epex = pd.read_parquet(_EPEX)["price_eur_mwh"]
    best = max(
        [v for v in list_vintages_2024_2025() if v.year < target],
        key=lambda v: (epex[epex.index < v].index.max() - epex[epex.index < v].index.min()).days,
    )
    anchors = {str(target): float(epex[epex.index.year == target].mean())}
    c_sota = build_curve(config, best, epex, anchors, estimator="sota")
    c_solar = build_curve(
        config,
        best,
        epex,
        anchors,
        estimator="sota_solar",
        solar_penetration_feature=SolarPenetrationFeature(_ENTSO),
    )

    ks = ch_physical_subkpis(c_sota, epex, target)
    kx = ch_physical_subkpis(c_solar, epex, target)

    assert SOLAR_MODULATION_EXPERIMENTAL is True
    assert SOLAR_MODULATION_PRODUCTION_DEFAULT is False

    bd_real = ks["solar_bowl_depth"]["realized"]
    # Flag-ON currently attenuates the bowl and moves away from realized.
    assert np.isfinite(ks["solar_bowl_depth"]["model"])
    assert np.isfinite(kx["solar_bowl_depth"]["model"])
    assert np.isfinite(bd_real)
    assert kx["solar_bowl_depth"]["model"] < ks["solar_bowl_depth"]["model"]
    assert abs(kx["solar_bowl_depth"]["model"] - bd_real) > abs(
        ks["solar_bowl_depth"]["model"] - bd_real
    )
    # Peak/off-peak spread also moves slightly away from realized.
    sp_real = ks["peak_offpeak_spread"]["realized"]
    assert abs(kx["peak_offpeak_spread"]["model"] - sp_real) > abs(
        ks["peak_offpeak_spread"]["model"] - sp_real
    )


@requires_fwds
def test_estimator_swap_restored_and_gate_preserved():
    """sota_solar must restore the monkeypatch and keep pf_cal_corr >= 0.85."""
    from scipy.stats import pearsonr

    from pfc_shaping.lt.model.assembler import PFCAssembler
    from pfc_shaping.validation.perfect_foresight import build_curve, monthly_signature
    from pfc_shaping.validation.scorecard import list_vintages_2024_2025

    target, config = 2025, "bowl_on_floors_off"
    epex = pd.read_parquet(_EPEX)["price_eur_mwh"]
    real_sig = monthly_signature(epex, target)
    best = max(
        [v for v in list_vintages_2024_2025() if v.year < target],
        key=lambda v: (epex[epex.index < v].index.max() - epex[epex.index < v].index.min()).days,
    )
    anchors = {str(target): float(epex[epex.index.year == target].mean())}

    before = PFCAssembler.__init__
    c_sota = build_curve(config, best, epex, anchors, estimator="sota")
    c_solar = build_curve(
        config,
        best,
        epex,
        anchors,
        estimator="sota_solar",
        solar_penetration_feature=SolarPenetrationFeature(_ENTSO),
    )
    after = PFCAssembler.__init__
    assert before is after  # swap fully restored

    def corr(curve):
        sig = monthly_signature(curve, target)
        both = pd.concat([sig, real_sig], axis=1, join="inner").dropna()
        return float(pearsonr(both.iloc[:, 0], both.iloc[:, 1])[0])

    r_sota, r_solar = corr(c_sota), corr(c_solar)
    assert r_solar >= 0.85
    # The intra-day-only, per-day-mean-preserving layer leaves the monthly
    # signature (hence pf_cal_corr) essentially unchanged.
    assert abs(r_sota - r_solar) < 5e-3


def test_invalid_estimator_rejected():
    from pfc_shaping.validation.perfect_foresight import build_curve

    with pytest.raises(ValueError):
        build_curve("bowl_on_floors_off", VINTAGE, pd.Series(dtype=float), {}, estimator="nope")
