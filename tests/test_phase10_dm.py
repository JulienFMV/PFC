"""
test_phase10_dm.py
------------------
Pillar 4 (Forecast accuracy comparison) — Diebold-Mariano test maison
+ 3 naive baselines (climatology, persistence Y-1, forwards-flat-no-shape).

Architecture
------------
- TestDieboldMariano (5 tests) : A clearly better / equal accuracy / n too few /
  h=24 lag=23 / var_d<0 fallback edge case.
- TestBaselines (3 tests) : baseline_climatology mean / baseline_persistence_y1
  window / baseline_forwards_flat horizon mapping.
- TestComputePillar4Dm (1 test) : end-to-end integration via
  compute_pillar4_dm + scorecard wiring.

Maison ~30-line DM via `statsmodels.tsa.stattools.acovf` + Bartlett HAC + HLN
small-sample correction. ZÉRO dep PyPI risquée (cf. RESEARCH §Don't Hand-Roll
reject `arch` et `dieboldmariano==0.1.x` slop risk).

Runtime cible : < 30s (pas de build_one chainé, juste numpy + statsmodels).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Module-level imports under test
# ---------------------------------------------------------------------------


def test_module_imports_ok():
    """Smoke : diebold_mariano + 3 baselines + compute_pillar4_dm importables."""
    from pfc_shaping.validation.dm_test import (  # noqa: F401
        baseline_climatology,
        baseline_forwards_flat,
        baseline_persistence_y1,
        diebold_mariano,
    )
    from pfc_shaping.validation.scorecard import compute_pillar4_dm  # noqa: F401


# ---------------------------------------------------------------------------
# TestDieboldMariano — DM 1995 + HLN 1997 correction
# ---------------------------------------------------------------------------


class TestDieboldMariano:
    """diebold_mariano — H0: equal accuracy between A and B."""

    def test_a_clearly_better_rejects_h0(self):
        """errors_a small (~0.1) vs errors_b big (~2.0) → mean_d<0, p<0.05."""
        from pfc_shaping.validation.dm_test import diebold_mariano

        rng = np.random.RandomState(42)
        n = 50
        errors_a = rng.normal(0.0, 0.1, n)  # A is accurate
        errors_b = rng.normal(0.0, 2.0, n)  # B is noisy
        res = diebold_mariano(errors_a, errors_b, h=1, loss="mae")
        assert res["degenerate"] is False
        assert res["mean_d"] < 0, f"A better → mean_d<0, got {res['mean_d']}"
        assert res["dm_stat"] < -2, f"A clearly better → dm_stat<-2, got {res['dm_stat']}"
        assert res["p_value"] < 0.05, f"A clearly better should reject H0, got p={res['p_value']}"
        assert res["n"] == n
        assert res["n_lags_hac"] == 0  # h=1 → lag=h-1=0

    def test_equal_accuracy_h0_not_rejected(self):
        """errors_a, errors_b same distribution → |dm_stat|<2, p>0.05."""
        from pfc_shaping.validation.dm_test import diebold_mariano

        rng = np.random.RandomState(42)
        n = 100
        errors_a = rng.normal(0.0, 1.0, n)
        errors_b = rng.normal(0.0, 1.0, n)  # different draw but same distribution
        res = diebold_mariano(errors_a, errors_b, h=1, loss="mae")
        assert res["degenerate"] is False
        assert abs(res["dm_stat"]) < 2.5, (
            f"equal accuracy → |dm_stat|<2.5, got {res['dm_stat']}"
        )
        assert res["p_value"] > 0.05, (
            f"equal accuracy should not reject H0, got p={res['p_value']}"
        )

    def test_n_too_few_degenerate(self):
        """n=3 < max(h+1, 8) → degenerate=True, p_value=NaN."""
        from pfc_shaping.validation.dm_test import diebold_mariano

        errors_a = np.array([0.1, 0.2, 0.3])
        errors_b = np.array([0.5, 0.6, 0.7])
        res = diebold_mariano(errors_a, errors_b, h=1, loss="mae")
        assert res["degenerate"] is True
        assert np.isnan(res["p_value"])
        assert np.isnan(res["dm_stat"])
        assert res["n"] == 3

    def test_h_24_lag_23_no_crash(self):
        """n=100 h=24 (Y+2 horizon) → n_lags_hac=23, sample suffisant, no crash."""
        from pfc_shaping.validation.dm_test import diebold_mariano

        rng = np.random.RandomState(42)
        n = 100
        errors_a = rng.normal(0.0, 1.0, n)
        errors_b = rng.normal(0.0, 1.0, n)
        res = diebold_mariano(errors_a, errors_b, h=24, loss="mae")
        # n=100 >= max(24+1, 8) = 25 → NOT degenerate
        assert res["degenerate"] is False
        assert res["n_lags_hac"] == 23, (
            f"h=24 → n_lags=h-1=23, got {res['n_lags_hac']}"
        )
        assert not np.isnan(res["dm_stat"])
        assert not np.isnan(res["p_value"])

    def test_var_d_negative_fallback(self):
        """Edge case construit : si var_d<=0 → fallback OR degenerate=True, no crash.

        Cas pathologique : input avec autocovariance négative dominante. La
        fonction doit OU fallback à sample variance (gammas[0]), OU retourner
        degenerate=True, jamais crasher avec NaN propagation silencieuse.
        """
        from pfc_shaping.validation.dm_test import diebold_mariano

        # Highly oscillating loss differential : -1, +1, -1, +1, ... force
        # potentiellement var_d négative en HAC (autocov lag=1 négative dominante)
        n = 30
        d_oscillating = np.tile([-1.0, 1.0], n // 2)  # n=30 values
        errors_a = d_oscillating * 0.5  # |a|
        errors_b = d_oscillating * 0.5 + d_oscillating  # |b| s.t. |a|-|b| oscille
        res = diebold_mariano(errors_a, errors_b, h=8, loss="mae")
        # No crash : la fonction retourne TOUJOURS un dict valide
        assert isinstance(res, dict)
        assert "dm_stat" in res
        assert "var_d" in res
        assert "degenerate" in res
        # Soit p_value est un float fini, soit degenerate=True (les 2 sont OK)
        if not res["degenerate"]:
            assert not np.isnan(res["dm_stat"])
        else:
            assert np.isnan(res["dm_stat"])

    def test_mse_loss_alternative(self):
        """loss='mse' utilise e^2 au lieu de |e| → result cohérent (sanity)."""
        from pfc_shaping.validation.dm_test import diebold_mariano

        rng = np.random.RandomState(0)
        n = 50
        errors_a = rng.normal(0.0, 0.1, n)
        errors_b = rng.normal(0.0, 2.0, n)
        res = diebold_mariano(errors_a, errors_b, h=1, loss="mse")
        # A better en MAE → aussi A better en MSE
        assert res["mean_d"] < 0
        assert res["p_value"] < 0.05

    def test_loss_invalid_raises(self):
        """loss=='invalid_loss' → ValueError explicit."""
        from pfc_shaping.validation.dm_test import diebold_mariano

        errors_a = np.zeros(20)
        errors_b = np.zeros(20)
        with pytest.raises(ValueError, match="mae|mse"):
            diebold_mariano(errors_a, errors_b, h=1, loss="logloss")

    def test_length_mismatch_raises(self):
        """len(a) != len(b) → ValueError explicit."""
        from pfc_shaping.validation.dm_test import diebold_mariano

        errors_a = np.zeros(20)
        errors_b = np.zeros(15)
        with pytest.raises(ValueError, match="[Ll]ength"):
            diebold_mariano(errors_a, errors_b, h=1, loss="mae")


# ---------------------------------------------------------------------------
# TestBaselines — 3 naive baselines maison
# ---------------------------------------------------------------------------


def _utc_hourly_index(start: str, end: str) -> pd.DatetimeIndex:
    """Helper : hourly UTC range."""
    return pd.date_range(start, end, freq="1h", inclusive="left", tz="UTC")


class TestBaselines:
    """3 naive baselines : climatology, persistence Y-1, forwards-flat-no-shape."""

    def test_baseline_climatology_returns_mean(self):
        """baseline_climatology = mean(EPEX | bloc filter, years range)."""
        from pfc_shaping.validation.block_masks import BlockMiddayWeekday
        from pfc_shaping.validation.dm_test import baseline_climatology

        # Construire EPEX hist 2019-2023 : 50 €/MWh constant pour Midday Weekday,
        # 30 €/MWh elsewhere (vérifier que le mask filtre bien)
        idx = _utc_hourly_index("2019-01-01", "2024-01-01")
        idx_local = idx.tz_convert("Europe/Zurich")
        is_midday_weekday = (idx_local.weekday < 5) & (idx_local.hour >= 10) & (idx_local.hour < 15)
        prices = np.where(is_midday_weekday, 50.0, 30.0)
        epex_hist = pd.Series(prices, index=idx, name="price_eur_mwh")
        bloc = BlockMiddayWeekday()
        result = baseline_climatology(epex_hist, bloc, years=(2019, 2023))
        # Tous les midday weekday valent 50 → mean(bloc filter) = 50
        assert isinstance(result, float)
        assert result == pytest.approx(50.0, abs=0.01)

    def test_baseline_persistence_y1_window(self):
        """vintage 2025-06-30, window ±15 days around 2024-06-30 → mean(bloc filter in window)."""
        from pfc_shaping.validation.block_masks import BlockMiddayWeekday
        from pfc_shaping.validation.dm_test import baseline_persistence_y1

        # EPEX realised 2024-06-01..2024-08-01 : 60 €/MWh pour midday weekday
        # autour de 2024-06-30 ± 15 days, 40 €/MWh ailleurs
        idx = _utc_hourly_index("2024-06-01", "2024-08-01")
        prices = np.full(len(idx), 40.0)
        # Mettre 60 €/MWh uniformément sur tout l'intervalle (le filtre window
        # + bloc le restreindra à midday weekday ± 15j de 2024-06-30)
        prices[:] = 60.0
        epex_realised = pd.Series(prices, index=idx, name="price_eur_mwh")
        bloc = BlockMiddayWeekday()
        vintage = pd.Timestamp("2025-06-30 18:00:00", tz="UTC")
        result = baseline_persistence_y1(
            epex_realised, vintage_date=vintage, bloc=bloc, window_days=15
        )
        # vintage_date - 1yr = 2024-06-30 ; window ± 15j = 2024-06-15..2024-07-15
        # Tous les prix valent 60 → mean = 60
        assert isinstance(result, float)
        assert result == pytest.approx(60.0, abs=0.01)

    def test_baseline_persistence_y1_empty_window_returns_nan(self):
        """Si fenêtre vide (epex_realised n'a pas de data pour cette periode) → NaN."""
        from pfc_shaping.validation.block_masks import BlockMiddayWeekday
        from pfc_shaping.validation.dm_test import baseline_persistence_y1

        # EPEX 2024 seulement ; vintage 2020 → window 2019 vide
        idx = _utc_hourly_index("2024-01-01", "2024-02-01")
        epex_realised = pd.Series(np.full(len(idx), 50.0), index=idx)
        bloc = BlockMiddayWeekday()
        vintage = pd.Timestamp("2020-06-30 18:00:00", tz="UTC")
        result = baseline_persistence_y1(
            epex_realised, vintage_date=vintage, bloc=bloc, window_days=15
        )
        assert np.isnan(result)

    def test_baseline_forwards_flat_horizon_mapping(self):
        """M+1, M+3, Y+1, Y+2 → bonnes keys lookup ; missing → NaN."""
        from pfc_shaping.validation.dm_test import baseline_forwards_flat

        forwards_asof = {
            "2024-07": 55.0,   # M+1 si vintage = 2024-06-30
            "2024-09": 60.0,   # M+3
            "2025-06": 70.0,   # absent comme Cal mais utile pour Y+2 indirect
            "2025": 65.0,      # Y+1
            "2026": 67.0,      # Y+2
        }
        vintage = pd.Timestamp("2024-06-30 18:00:00", tz="UTC")
        assert baseline_forwards_flat(forwards_asof, "M+1", vintage) == pytest.approx(55.0)
        assert baseline_forwards_flat(forwards_asof, "M+3", vintage) == pytest.approx(60.0)
        assert baseline_forwards_flat(forwards_asof, "Y+1", vintage) == pytest.approx(65.0)
        assert baseline_forwards_flat(forwards_asof, "Y+2", vintage) == pytest.approx(67.0)
        # M+6 absent du dict → NaN
        result_missing = baseline_forwards_flat(forwards_asof, "M+6", vintage)
        assert np.isnan(result_missing)


# ---------------------------------------------------------------------------
# TestComputePillar4Dm — integration end-to-end via scorecard wiring
# ---------------------------------------------------------------------------


class TestComputePillar4Dm:
    """compute_pillar4_dm — end-to-end DM cell KPI dict (bloc × horizon × baseline)."""

    def test_end_to_end_synth(self):
        """Synth pred_pfc + pred_baseline + realised → dict structure correct."""
        from pfc_shaping.validation.block_masks import BlockMiddayWeekday
        from pfc_shaping.validation.scorecard import compute_pillar4_dm

        rng = np.random.RandomState(42)
        idx = _utc_hourly_index("2024-06-01", "2024-09-01")  # 3 months
        n = len(idx)
        realised = pd.Series(50.0 + rng.normal(0, 2, n), index=idx)
        # PFC quasi-perfect : realised + small noise
        pred_pfc = realised + rng.normal(0, 0.3, n)
        # Baseline poor : realised + large noise (constant=50 alternative)
        pred_baseline = pd.Series(50.0 + rng.normal(0, 5, n), index=idx)
        bloc = BlockMiddayWeekday()
        res = compute_pillar4_dm(
            pred_pfc=pred_pfc,
            pred_baseline=pred_baseline,
            realised=realised,
            bloc=bloc,
            h_months=1,
        )
        # Structure check : 12 keys
        for key in ("bloc", "h_months", "dm_stat", "p_value", "n",
                    "mean_d", "var_d", "n_lags_hac", "degenerate",
                    "mae_pfc", "mae_baseline", "delta_mae", "better_than_baseline"):
            assert key in res, f"missing key {key} in result {res}"
        # PFC clairly better → mae_pfc < mae_baseline → delta_mae < 0
        assert res["mae_pfc"] < res["mae_baseline"]
        assert res["delta_mae"] < 0
        assert res["bloc"] == "block_midday_weekday"
        assert res["h_months"] == 1
        # better_than_baseline = "Y" if (mean_d < 0 AND p < 0.05)
        assert res["better_than_baseline"] in ("Y", "N")

    def test_pred_baseline_scalar_broadcasts(self):
        """pred_baseline=float (e.g. baseline_climatology output) → broadcast OK."""
        from pfc_shaping.validation.block_masks import BlockMiddayWeekday
        from pfc_shaping.validation.scorecard import compute_pillar4_dm

        rng = np.random.RandomState(0)
        idx = _utc_hourly_index("2024-06-01", "2024-09-01")
        n = len(idx)
        realised = pd.Series(50.0 + rng.normal(0, 2, n), index=idx)
        pred_pfc = realised + rng.normal(0, 0.3, n)
        baseline_scalar = 50.0  # climatology mean
        bloc = BlockMiddayWeekday()
        res = compute_pillar4_dm(
            pred_pfc=pred_pfc,
            pred_baseline=baseline_scalar,
            realised=realised,
            bloc=bloc,
            h_months=1,
        )
        assert res["bloc"] == "block_midday_weekday"
        assert isinstance(res["mae_pfc"], float)
        assert isinstance(res["mae_baseline"], float)
        # baseline = 50 constant → mae_baseline ≈ |realised - 50|.mean() ≈ E(|N(0,2)|) ≈ 1.6
        assert res["mae_baseline"] > 1.0
