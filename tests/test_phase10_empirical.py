"""
test_phase10_empirical.py
-------------------------
Pillar 2 (KYOS/Volue empirical accuracy) — MAE/RMSE/bias + Mincer-Zarnowitz
regression joint α=0 & β=1 tests.

Architecture
------------
- TestMzTest (4 tests) : mz_test sur cas trivial / biais constant / biais
  multiplicatif / dégeneré n<3.
- TestStatsmodelsFTestSignature (1 test, warning #6 fix) : pin statsmodels
  0.14.x API signature pour f_test('const = 0, predicted = 1') — détecte
  breaking change futur si statsmodels change la convention.
- TestComputeCellKpis (4 tests) : compute_cell_kpis sanity + empty mask +
  low_power_flag.
- TestConfig3SmokeTest (1 test, blocker #11 fix Q5 RESEARCH RESOLVED) :
  build_one(Config 3 = bowl_off_floors_off, vintage 2024-06-28) numerical
  stability sur real parquet OR pytest.skip si absent.
- TestPillar2Integration (1 test) : build_one + compute_cell_kpis end-to-end.

Runtime cible : < 5 min (≤1 build assembler pour integration, ≤1 build pour Config 3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Module-level imports under test
# ---------------------------------------------------------------------------


def test_module_imports_ok():
    """Smoke : mz_test + compute_cell_kpis sont importables sans erreur."""
    from pfc_shaping.validation.scorecard import compute_cell_kpis, mz_test  # noqa: F401


# ---------------------------------------------------------------------------
# TestMzTest — Mincer-Zarnowitz regression
# ---------------------------------------------------------------------------


class TestMzTest:
    """mz_test — Pattern 2 canonical statsmodels.OLS.f_test."""

    def test_perfect_forecast_alpha0_beta1(self):
        """Unbiased noisy forecast (n=100) → α≈0, β≈1, joint H0 not rejected (p>0.5).

        Note : un "vrai" perfect forecast (realised == predicted exactement) crée
        un edge case numérique statsmodels (RSS=1e-30 → f_test degenerate, p≈0.4
        artifact). On utilise donc un forecast "non-biased with noise" qui est
        statistiquement équivalent à perfect (alpha~0, beta~1) sans déclencher
        l'edge case OLS RSS=0.
        """
        from pfc_shaping.validation.scorecard import mz_test

        rng = np.random.RandomState(42)
        n = 100
        realised = pd.Series(50.0 + rng.normal(0, 5, n))
        predicted = realised + pd.Series(rng.normal(0, 1, n))
        res = mz_test(realised, predicted)
        assert abs(res["alpha"]) < 2.0, f"alpha should be small, got {res['alpha']}"
        assert abs(res["beta"] - 1.0) < 0.1, f"beta should be ~1, got {res['beta']}"
        # Joint H0 (alpha=0, beta=1) trivially NOT rejected on unbiased
        assert res["p_value_joint_unbiased"] > 0.5, (
            f"unbiased forecast should not reject H0, got p={res['p_value_joint_unbiased']}"
        )
        assert res["n_obs"] == 100
        assert res["degenerate"] is False

    def test_constant_offset_bias_detected(self):
        """realised = predicted + 1 (biais additif) → α≈1, β≈1, joint p<0.05."""
        from pfc_shaping.validation.scorecard import mz_test

        predicted = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        realised = predicted + 1.0  # biais constant +1
        res = mz_test(realised, predicted)
        assert abs(res["alpha"] - 1.0) < 1e-6, f"alpha should be ~1, got {res['alpha']}"
        assert abs(res["beta"] - 1.0) < 1e-6, f"beta should be ~1, got {res['beta']}"
        # Joint H0 (alpha=0, beta=1) REJECTED car alpha != 0
        assert res["p_value_joint_unbiased"] < 0.05

    def test_scaled_forecast_beta_detected(self):
        """realised = 10×predicted (biais multiplicatif) → β=10, joint p<0.05."""
        from pfc_shaping.validation.scorecard import mz_test

        predicted = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        realised = predicted * 10.0
        res = mz_test(realised, predicted)
        assert abs(res["beta"] - 10.0) < 1e-6, f"beta should be ~10, got {res['beta']}"
        assert res["p_value_joint_unbiased"] < 0.05

    def test_degenerate_n_lt_3(self):
        """n=1 (underdetermined OLS const+slope=2 params) → degenerate=True, no crash."""
        from pfc_shaping.validation.scorecard import mz_test

        realised = pd.Series([1.0])
        predicted = pd.Series([1.0])
        res = mz_test(realised, predicted)
        assert res["degenerate"] is True
        assert np.isnan(res["p_value_joint_unbiased"])
        assert res["n_obs"] == 1
        # alpha, beta, r_squared also NaN (or any value, no crash is the test)


# ---------------------------------------------------------------------------
# TestStatsmodelsFTestSignature — pin statsmodels 0.14.x API (warning #6 fix)
# ---------------------------------------------------------------------------


class TestStatsmodelsFTestSignature:
    """Pin statsmodels f_test API signature pour détecter breaking change."""

    def test_statsmodels_f_test_api_signature(self):
        """Pin : sm.OLS(...).fit().f_test('const = 0, predicted = 1') → pvalue/fvalue OK."""
        import statsmodels.api as sm

        assert sm.__version__.startswith("0.14"), (
            f"statsmodels version drift ({sm.__version__}) — "
            "verify f_test('const = 0, predicted = 1') API still valid"
        )

        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        X = sm.add_constant(x.rename("predicted"))
        fit = sm.OLS(y, X).fit()
        f_result = fit.f_test("const = 0, predicted = 1")

        assert hasattr(f_result, "pvalue"), "f_test result missing pvalue attribute"
        assert hasattr(f_result, "fvalue"), "f_test result missing fvalue attribute"
        # pvalue castable to float (NaN-safe on perfect fit)
        pval = f_result.pvalue
        assert pval is None or np.isnan(pval) or isinstance(float(pval), float), (
            f"f_test pvalue not float-castable: {pval}"
        )


# ---------------------------------------------------------------------------
# TestComputeCellKpis — MAE/RMSE/bias + MZ par cellule (bloc × horizon)
# ---------------------------------------------------------------------------


def _utc_hourly_2024() -> pd.DatetimeIndex:
    """Helper : un mois (juin 2024) hourly UTC."""
    return pd.date_range(
        "2024-06-01", "2024-07-01", freq="1h", inclusive="left", tz="UTC"
    )


class TestComputeCellKpis:
    """compute_cell_kpis — cellule unitaire (bloc × horizon)."""

    def test_sanity_on_deterministic_noise(self):
        """pred = realised + N(0, 0.5) → MAE ≈ 0.4, RMSE ≈ 0.5, bias ≈ 0."""
        from pfc_shaping.validation.block_masks import BlockMiddayWeekday
        from pfc_shaping.validation.scorecard import compute_cell_kpis

        idx = _utc_hourly_2024()
        rng = np.random.RandomState(42)
        realised = pd.Series(50.0 + rng.normal(0, 0.5, len(idx)), index=idx)
        # pred = realised + indep noise (~same RMSE)
        pred = realised + pd.Series(rng.normal(0, 0.5, len(idx)), index=idx)

        bloc = BlockMiddayWeekday()
        kpi = compute_cell_kpis(pred, realised, bloc, horizon_label="M+1")

        assert kpi["bloc"] == "block_midday_weekday"
        assert kpi["horizon"] == "M+1"
        assert kpi["n_obs"] > 0
        # MAE expected ≈ 0.5 * sqrt(2/pi) ≈ 0.4 ; RMSE ≈ sqrt(0.5²+0.5²) ≈ 0.707
        assert 0.2 < kpi["mae"] < 0.7
        assert 0.3 < kpi["rmse"] < 1.0
        assert abs(kpi["bias"]) < 0.5

    def test_empty_mask_returns_n_obs_0(self):
        """BlockWinterEveningPeak (nov-fév) sur juin → mask vide → degenerate=True."""
        from pfc_shaping.validation.block_masks import BlockWinterEveningPeak
        from pfc_shaping.validation.scorecard import compute_cell_kpis

        idx = _utc_hourly_2024()  # juin 2024 (out of nov-fév)
        pred = pd.Series(50.0, index=idx)
        realised = pd.Series(50.0, index=idx)

        bloc = BlockWinterEveningPeak()
        kpi = compute_cell_kpis(pred, realised, bloc, horizon_label="M+1")

        assert kpi["n_obs"] == 0
        assert np.isnan(kpi["mae"])
        assert np.isnan(kpi["rmse"])
        assert kpi["degenerate"] is True
        assert kpi["low_power_flag"] is True

    def test_low_power_flag_true_when_n_lt_30(self):
        """Tiny pred/realised (n<30 dans le bloc) → low_power_flag=True."""
        from pfc_shaping.validation.block_masks import BlockMiddayWeekday
        from pfc_shaping.validation.scorecard import compute_cell_kpis

        # 5 jours seulement → BlockMiddayWeekday (10-15h weekday) ≈ 5h × 5j ≈ 25 obs
        idx = pd.date_range(
            "2024-06-03", "2024-06-08", freq="1h", inclusive="left", tz="UTC"
        )
        pred = pd.Series(50.0, index=idx)
        realised = pd.Series(50.0, index=idx)

        kpi = compute_cell_kpis(pred, realised, BlockMiddayWeekday(), "M+1")
        assert kpi["n_obs"] < 30
        assert kpi["low_power_flag"] is True

    def test_low_power_flag_false_when_n_ge_30(self):
        """Mois complet → BlockMiddayWeekday (≈5h × 20wd ≈ 100 obs) → low_power_flag=False."""
        from pfc_shaping.validation.block_masks import BlockMiddayWeekday
        from pfc_shaping.validation.scorecard import compute_cell_kpis

        idx = _utc_hourly_2024()  # juin 2024 complet
        pred = pd.Series(50.0, index=idx)
        realised = pd.Series(50.0, index=idx)

        kpi = compute_cell_kpis(pred, realised, BlockMiddayWeekday(), "M+1")
        assert kpi["n_obs"] >= 30
        assert kpi["low_power_flag"] is False


# ---------------------------------------------------------------------------
# TestConfig3SmokeTest — Q5 RESEARCH RESOLVED, blocker #11 fix
# ---------------------------------------------------------------------------


class TestConfig3SmokeTest:
    """Config 3 (bowl_off_floors_off) numerical stability smoke-test."""

    def test_config_3_bowl_off_floors_off_numerical_stability(self):
        """build_one(Config 3, vintage 2024-06-28) → notna + |abs| < 1000."""
        from pathlib import Path

        from pfc_shaping.validation.scorecard import (
            ABLATION_GRID,
            build_one,
            derive_forwards_from_epex_hist,
        )

        config_3 = ABLATION_GRID[2]  # bowl_off_floors_off
        assert config_3.name == "bowl_off_floors_off", (
            f"Index drift: ABLATION_GRID[2] = {config_3.name}"
        )

        vintage = pd.Timestamp("2024-06-28 18:00", tz="Europe/Zurich").tz_convert("UTC")

        epex_path = Path("data/epex_hourly.parquet")
        if not epex_path.exists():
            pytest.skip(
                "data/epex_hourly.parquet not bootstrapped (Plan 10-01 Task 3 required)"
            )

        epex_hist_series = pd.read_parquet(epex_path)["price_eur_mwh"]
        forwards = derive_forwards_from_epex_hist(epex_hist_series, vintage)

        # build_one expects DataFrame with 'price_eur_mwh' column
        epex_hist_df = epex_hist_series.to_frame(name="price_eur_mwh")

        try:
            pfc = build_one(
                config_3,
                vintage,
                epex_hist_df,
                forwards,
                with_uncertainty=False,
            )
        except Exception as exc:
            pytest.fail(
                f"Config 3 (bowl OFF + floors OFF) raised {type(exc).__name__}: "
                f"{exc}. Q5 resolution path : Plan 10-03 should add "
                "--allow-config-3-failures escape hatch."
            )

        n_nan = pfc["price_shape"].isna().sum()
        assert n_nan == 0, (
            f"NaN detected in Config 3 PFC — {n_nan} NaNs. "
            "Q5 resolution : Plan 10-03 escape hatch needed."
        )
        max_abs = pfc["price_shape"].abs().max()
        assert max_abs < 1000.0, (
            f"Numerical explosion Config 3 — max abs={max_abs:.2f} €/MWh. "
            "Q5 resolution : Plan 10-03 escape hatch needed."
        )


# ---------------------------------------------------------------------------
# TestPillar2Integration — end-to-end build_one + compute_cell_kpis
# ---------------------------------------------------------------------------


class TestPillar2Integration:
    """End-to-end : build_one(Config 4) → compute_cell_kpis(BlockMiddayWeekday)."""

    def test_build_one_output_compatible_with_compute_cell_kpis(self):
        """build_one(Config 4) puis compute_cell_kpis : n_obs>0, bias<100 €/MWh."""
        from pfc_shaping.validation.block_masks import BlockMiddayWeekday
        from pfc_shaping.validation.scorecard import (
            ABLATION_GRID,
            build_one,
            compute_cell_kpis,
            derive_forwards_from_epex_hist,
            _synth_epex_hist_for_mock,
            last_business_day_of_month,
        )

        epex_hist_series = _synth_epex_hist_for_mock(seed=42)
        epex_hist_df = epex_hist_series.to_frame(name="price_eur_mwh")
        vintage = last_business_day_of_month(2024, 6).tz_convert("UTC")
        forwards = derive_forwards_from_epex_hist(epex_hist_series, vintage)

        pfc = build_one(
            ABLATION_GRID[3],  # bowl_on_floors_off (Config 4)
            vintage,
            epex_hist_df,
            forwards,
            with_uncertainty=False,
        )
        # "Realised" = synthétique = epex_hist_series projeté sur l'horizon Y+3
        # Pour le test, prendre les valeurs PFC comme realised proxy (sanity test)
        pred = pfc["price_shape"]
        # Realised = pred + tiny noise → sanity
        rng = np.random.RandomState(42)
        realised = pred + pd.Series(
            rng.normal(0, 1.0, len(pred)), index=pred.index
        )

        bloc = BlockMiddayWeekday()
        kpi = compute_cell_kpis(pred, realised, bloc, horizon_label="M+1")

        assert kpi["n_obs"] > 0
        assert abs(kpi["bias"]) < 100.0
        # Sanity : les 12 keys attendues sont présentes
        expected_keys = {
            "bloc", "horizon", "n_obs", "mae", "rmse", "bias",
            "mz_alpha", "mz_beta", "mz_p_value", "mz_r_squared",
            "low_power_flag", "degenerate",
        }
        assert set(kpi.keys()) == expected_keys, (
            f"Expected {expected_keys}, got {set(kpi.keys())}"
        )
        assert isinstance(kpi["low_power_flag"], (bool, np.bool_))
        assert isinstance(kpi["degenerate"], (bool, np.bool_))
