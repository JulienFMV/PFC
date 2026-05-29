"""
test_phase10_hildmann.py
------------------------
Pillar 1 (Hildmann 2013, ETH Zurich) — 4 tests structurels SC#1 gate Phase 10.

Architecture
------------
- Fixture module-scoped `pillar1_results` qui appelle `run_scorecard_pillar_1`
  une fois (Config 4 = bowl_on_floors_off, production target) ; les 4 tests
  SC#1 réutilisent ce résultat (~1 build assembler en mock mode).
- 4 tests SC#1 gate : arb-free / holiday-weekend / seasonal-profile / continuity.
- 12 unit tests des 4 fonctions Hildmann (cas pass, fail, dégenérés).

**IMPORTANT — MOCK CI mode (C4 REVIEWS)** :
  Ces tests utilisent `epex_source='mock'` avec une fixture synthétique
  déterministe (seed=42) CONSTRUITE pour PASS. Un mock-PASS prouve UNIQUEMENT
  que le code path est correct (4 fonctions importables, run_scorecard_pillar_1
  orchestrate le chain build_one → 4 tests sans crash, et les outputs sont
  TestResult.passed=True sur fixture conçue à cet effet).

  Le **vrai verdict SC#1** vient du **MOCK CI — Gate SC#1 réel =
  run_scorecard_pillar_1(epex_source='parquet') Plan 10-04 Task 2** sur les
  24 vintages réels, avec forwards_source == 'real_eex_xlsx'.

Runtime cible : < 5 min (≤4 vintages mock-mode, with_uncertainty=False).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Module-level imports under test
# ---------------------------------------------------------------------------


def _ensure_module_imports():
    from pfc_shaping.validation.structural_tests import (  # noqa: F401
        HOLIDAY_WEEKEND_RANGE,
        TestResult,
        test_arb_free,
        test_continuity,
        test_holiday_weekend,
        test_seasonal_profile,
    )
    from pfc_shaping.validation.scorecard import (  # noqa: F401
        run_scorecard_pillar_1,
    )


def test_module_imports_ok():
    """Smoke test : les 4 fonctions Hildmann + TestResult + HOLIDAY_WEEKEND_RANGE
    + run_scorecard_pillar_1 sont importables sans erreur."""
    _ensure_module_imports()


def test_holiday_weekend_range_is_frozen_from_notes():
    """HOLIDAY_WEEKEND_RANGE = (0.65, 0.95) per 10-01-NOTES Pitfall 1 IF branch."""
    from pfc_shaping.validation.structural_tests import HOLIDAY_WEEKEND_RANGE

    assert HOLIDAY_WEEKEND_RANGE == (0.65, 0.95), (
        f"HOLIDAY_WEEKEND_RANGE doit être (0.65, 0.95) ex-ante frozen IF branch "
        f"(C2 REVIEWS, ratio mesuré 0.8033 ∈ [0.65, 0.95]) — got {HOLIDAY_WEEKEND_RANGE}"
    )


# ---------------------------------------------------------------------------
# Module-scoped fixture — SC#1 gate evaluation (mock CI mode)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pillar1_results(tmp_path_factory):
    """Run Pillar 1 once for the entire module — SC#1 gate (MOCK CI evaluation).

    MOCK CI — Gate SC#1 réel = run_scorecard_pillar_1(epex_source='parquet')
    Plan 10-04 Task 2 sur les 24 vintages réels (forwards_source='real_eex_xlsx').

    Builds the PFC under Config 4 (bowl ON + floors OFF = production target),
    runs the 4 Hildmann tests on the aggregate of 4 mock vintages (Q1-Q4 2024).
    """
    from pfc_shaping.validation.scorecard import run_scorecard_pillar_1

    cache_dir = tmp_path_factory.mktemp("phase10_pillar1")
    results = run_scorecard_pillar_1(
        config_name="bowl_on_floors_off",
        cache_dir=cache_dir,
        epex_source="mock",
    )
    return results


# ---------------------------------------------------------------------------
# 4 SC#1 gate asserts (MOCK CI mode — couverture branche code, pas verdict gate)
# ---------------------------------------------------------------------------


def test_phase10_sc1_arb_free(pillar1_results):
    """SC#1.1 — arbitrage-freeness < 0.01 €/MWh on Config 4 aggregate.

    MOCK CI — Gate SC#1 réel = run_scorecard_pillar_1(epex_source='parquet')
    Plan 10-04 Task 2 sur les 24 vintages réels (forwards_source='real_eex_xlsx').
    """
    res = pillar1_results["arb_free"]
    assert res.passed, (
        f"Arb-free FAILED (MOCK CI): max dev = {res.observed:.4f} > {res.threshold}"
    )


def test_phase10_sc1_holiday_weekend(pillar1_results):
    """SC#1.2 — holiday-weekend ratio ∈ [0.65, 0.95] on Config 4.

    MOCK CI — Gate SC#1 réel = Plan 10-04 Task 2 real-run.
    """
    res = pillar1_results["holiday_weekend"]
    assert res.passed, (
        f"Holiday/weekend ratio FAILED (MOCK CI): observed = {res.observed:.3f}, "
        f"expected ∈ {res.threshold}. See Pitfall 1 in 10-01-NOTES.md."
    )


def test_phase10_sc1_seasonal_profile(pillar1_results):
    """SC#1.3 — Pearson(PFC_monthly, EPEX_hist_monthly) > 0.85 on Config 4.

    MOCK CI — Gate SC#1 réel = Plan 10-04 Task 2 real-run.
    """
    res = pillar1_results["seasonal_profile"]
    assert res.passed, (
        f"Seasonal corr FAILED (MOCK CI): r = {res.observed:.3f} <= 0.85"
    )


def test_phase10_sc1_continuity(pillar1_results):
    """SC#1.4 — max month-boundary jump < 2 €/MWh on Config 4.

    MOCK CI — Gate SC#1 réel = Plan 10-04 Task 2 real-run.
    """
    res = pillar1_results["continuity"]
    assert res.passed, (
        f"Continuity FAILED (MOCK CI): max jump = {res.observed:.3f} > 2.0 €/MWh"
    )


# ---------------------------------------------------------------------------
# Unit tests for the 4 Hildmann functions (12 tests, isolated synth data)
# ---------------------------------------------------------------------------


def _hourly_index_2024_2025_utc() -> pd.DatetimeIndex:
    """Helper : 2 ans 2024-2025 hourly UTC (pour les unit tests Hildmann)."""
    return pd.date_range(
        "2024-01-01", "2026-01-01", freq="1h", inclusive="left", tz="UTC"
    )


def _hourly_index_jan_2024_utc() -> pd.DatetimeIndex:
    """Helper : janvier 2024 hourly UTC (744 rows)."""
    return pd.date_range(
        "2024-01-01", "2024-02-01", freq="1h", inclusive="left", tz="UTC"
    )


class TestArbFreeFunction:
    """test_arb_free — Pillar 1.1 Hildmann arbitrage-freeness."""

    def test_pass_case_perfect_match(self):
        """Une PFC dont le mean(Jan 2024) = forward Jan 2024 exact → passed=True."""
        from pfc_shaping.validation.structural_tests import test_arb_free

        idx = _hourly_index_jan_2024_utc()
        pfc = pd.Series(50.0001, index=idx)
        forwards = {"2024-01": 50.0}
        res = test_arb_free(pfc, forwards, tol=0.01)
        assert res.passed
        assert abs(res.observed - 0.0001) < 1e-6

    def test_fail_case_large_deviation(self):
        """Une PFC dont le mean(Jan 2024) dévie de 0.5 du forward → passed=False."""
        from pfc_shaping.validation.structural_tests import test_arb_free

        idx = _hourly_index_jan_2024_utc()
        pfc = pd.Series(50.5, index=idx)
        forwards = {"2024-01": 50.0}
        res = test_arb_free(pfc, forwards, tol=0.01)
        assert not res.passed
        assert abs(res.observed - 0.5) < 1e-6

    def test_empty_forwards_returns_passed_true(self):
        """Un dict forwards vide → pas de violation possible → passed=True (max_dev=0)."""
        from pfc_shaping.validation.structural_tests import test_arb_free

        idx = _hourly_index_jan_2024_utc()
        pfc = pd.Series(50.0, index=idx)
        res = test_arb_free(pfc, {}, tol=0.01)
        assert res.passed
        assert res.observed == 0.0

    def test_cr01_local_calendar_boundary_bucketing(self):
        """CR-01 regression: `_period_mask` doit bucket en Europe/Zurich, pas UTC.

        Une PFC indexée UTC avec un step de +10 €/MWh à la frontière LOCALE
        2024→2025 (i.e. 2024-12-31 23:00 UTC = 2025-01-01 00:00 CET) doit voir
        sa moyenne Cal-2024 correspondre au mean des 8784 heures où LOCAL year
        == 2024. Avec la conv UTC buggée, l'heure 2024-12-31 23:00 UTC serait
        comptée dans Cal-2024 alors qu'elle appartient en local au Cal-2025 →
        biais de l'ordre de Δprice / 8760 dans la moyenne, déclenchant un
        faux fail à tol=0.01 €/MWh.
        """
        from pfc_shaping.validation.structural_tests import test_arb_free

        idx = pd.date_range(
            "2024-01-01", "2026-01-01", freq="1h", inclusive="left", tz="UTC"
        )
        idx_local = idx.tz_convert("Europe/Zurich")
        # Step +10 à la frontière locale 2024→2025
        values = np.where(idx_local.year == 2024, 50.0, 60.0).astype(float)
        pfc = pd.Series(values, index=idx)
        # Forward Cal-2024 = exactement 50.0 (mean local bucket année 2024)
        forwards = {"2024": 50.0, "2025": 60.0}
        res = test_arb_free(pfc, forwards, tol=0.01)
        assert res.passed, (
            f"CR-01 regression: Cal-2024 mean diverge (UTC bucketing bug) — "
            f"max_dev={res.observed:.6f}, details={res.details}"
        )
        # Dev sur les deux Cal doit être ~exactement 0 (pas ~10/8760)
        assert res.observed < 1e-9, (
            f"CR-01 regression: expected dev ~0, got {res.observed:.6e}"
        )


class TestHolidayWeekendFunction:
    """test_holiday_weekend — Pillar 1.2 Hildmann ratio weekend ∪ holidays / weekday."""

    def test_pass_case_synth_ratio_in_range(self):
        """Synth PFC weekday=100, weekend=80 → ratio 0.80 ∈ [0.65, 0.95]."""
        from pfc_shaping.validation.structural_tests import test_holiday_weekend

        idx = _hourly_index_2024_2025_utc()
        idx_local = idx.tz_convert("Europe/Zurich")
        is_weekend = idx_local.weekday >= 5
        values = np.where(is_weekend, 80.0, 100.0)
        pfc = pd.Series(values, index=idx)
        res = test_holiday_weekend(pfc)
        assert res.passed, f"observed={res.observed}, threshold={res.threshold}"
        # ratio ≈ 0.80, mais holidays VS rajoutent ~50 days/2yrs au numerator
        assert 0.65 <= res.observed <= 0.95

    def test_degenerate_near_zero_weekday_mean(self):
        """mean(weekday) près de zéro → degenerate=True, passed=False."""
        from pfc_shaping.validation.structural_tests import test_holiday_weekend

        idx = _hourly_index_2024_2025_utc()
        idx_local = idx.tz_convert("Europe/Zurich")
        is_weekend = idx_local.weekday >= 5
        # weekday=0.1 (near zero), weekend=10 → ratio undefined per |mean_on| < 0.5
        values = np.where(is_weekend, 10.0, 0.1)
        pfc = pd.Series(values, index=idx)
        res = test_holiday_weekend(pfc)
        assert not res.passed
        assert res.details.get("degenerate") is True

    def test_fail_case_ratio_out_of_range(self):
        """ratio = 0.30 (weekend très bas) → passed=False, ratio < 0.65."""
        from pfc_shaping.validation.structural_tests import test_holiday_weekend

        idx = _hourly_index_2024_2025_utc()
        idx_local = idx.tz_convert("Europe/Zurich")
        is_weekend = idx_local.weekday >= 5
        values = np.where(is_weekend, 30.0, 100.0)
        pfc = pd.Series(values, index=idx)
        res = test_holiday_weekend(pfc)
        assert not res.passed
        assert res.observed < 0.65


class TestSeasonalProfileFunction:
    """test_seasonal_profile — Pillar 1.3 Hildmann Pearson corr PFC vs EPEX hist."""

    def test_pass_case_high_corr_with_winter_peak(self):
        """Synth PFC + EPEX hist tous deux winter-peak → corr ≈ 1."""
        from pfc_shaping.validation.structural_tests import test_seasonal_profile

        # PFC 2024-2025 hourly avec saisonnalité hiver-pic
        idx_pfc = _hourly_index_2024_2025_utc()
        seasonal_pfc = 100 + 30 * np.cos(2 * np.pi * (idx_pfc.month - 1) / 12)
        pfc = pd.Series(seasonal_pfc, index=idx_pfc)

        # EPEX hist 2019-2023 hourly avec saisonnalité similaire
        idx_hist = pd.date_range(
            "2019-01-01", "2024-01-01", freq="1h", inclusive="left", tz="UTC"
        )
        seasonal_hist = 80 + 25 * np.cos(2 * np.pi * (idx_hist.month - 1) / 12)
        epex_hist = pd.Series(seasonal_hist, index=idx_hist)

        res = test_seasonal_profile(pfc, epex_hist)
        assert res.passed
        assert res.observed > 0.95  # cos × cos = parfait

    def test_fail_case_low_corr_inverted_seasonal(self):
        """PFC summer-peak vs hist winter-peak → corr fortement négative."""
        from pfc_shaping.validation.structural_tests import test_seasonal_profile

        idx_pfc = _hourly_index_2024_2025_utc()
        # PFC summer-peak (inverse)
        seasonal_pfc = 100 - 30 * np.cos(2 * np.pi * (idx_pfc.month - 1) / 12)
        pfc = pd.Series(seasonal_pfc, index=idx_pfc)

        idx_hist = pd.date_range(
            "2019-01-01", "2024-01-01", freq="1h", inclusive="left", tz="UTC"
        )
        seasonal_hist = 80 + 25 * np.cos(2 * np.pi * (idx_hist.month - 1) / 12)
        epex_hist = pd.Series(seasonal_hist, index=idx_hist)

        res = test_seasonal_profile(pfc, epex_hist)
        assert not res.passed
        assert res.observed < 0.85

    def test_partial_months_edge_case(self):
        """PFC ne couvrant que jan-juin → 6 mois communs, corr toujours computable."""
        from pfc_shaping.validation.structural_tests import test_seasonal_profile

        idx_pfc = pd.date_range(
            "2024-01-01", "2024-07-01", freq="1h", inclusive="left", tz="UTC"
        )
        seasonal_pfc = 100 + 30 * np.cos(2 * np.pi * (idx_pfc.month - 1) / 12)
        pfc = pd.Series(seasonal_pfc, index=idx_pfc)

        idx_hist = pd.date_range(
            "2019-01-01", "2024-01-01", freq="1h", inclusive="left", tz="UTC"
        )
        seasonal_hist = 80 + 25 * np.cos(2 * np.pi * (idx_hist.month - 1) / 12)
        epex_hist = pd.Series(seasonal_hist, index=idx_hist)

        res = test_seasonal_profile(pfc, epex_hist)
        # 6 mois communs jan-juin, corr toujours élevée (même shape cos)
        assert res.observed > 0.85
        assert res.details["n_months"] == 6

    def test_shape_diagnostics_and_affine_invariance(self):
        """details expose spearman + shape residuals ; Pearson est affine-invariant."""
        from pfc_shaping.validation.structural_tests import test_seasonal_profile

        idx_pfc = _hourly_index_2024_2025_utc()
        seasonal_pfc = 100 + 30 * np.cos(2 * np.pi * (idx_pfc.month - 1) / 12)
        pfc = pd.Series(seasonal_pfc, index=idx_pfc)

        idx_hist = pd.date_range(
            "2019-01-01", "2024-01-01", freq="1h", inclusive="left", tz="UTC"
        )
        seasonal_hist = (
            80
            + 25 * np.cos(2 * np.pi * (idx_hist.month - 1) / 12)
            + 4 * np.sin(2 * np.pi * (idx_hist.month - 1) / 12)
        )
        epex_hist = pd.Series(seasonal_hist, index=idx_hist)

        res = test_seasonal_profile(pfc, epex_hist)
        for key in (
            "spearman_r",
            "monthly_pfc",
            "monthly_hist",
            "shape_residuals_z",
            "top_divergent_months",
        ):
            assert key in res.details
        assert len(res.details["top_divergent_months"]) <= 3

        # Pearson r is invariant under affine transforms of either signature:
        # z-scoring the inputs must not change `observed` (normalisation = no-op).
        pfc_affine = pfc * 0.5 + 17.0
        hist_affine = epex_hist * 3.0 - 9.0
        res_affine = test_seasonal_profile(pfc_affine, hist_affine)
        assert res_affine.observed == pytest.approx(res.observed, abs=1e-9)


class TestContinuityFunction:
    """test_continuity — Pillar 1.4 Hildmann jumps aux frontières mensuelles."""

    def test_smooth_pfc_passes(self):
        """PFC parfaitement constante → max_jump = 0 < 2.0 €/MWh."""
        from pfc_shaping.validation.structural_tests import test_continuity

        idx = _hourly_index_2024_2025_utc()
        pfc = pd.Series(50.0, index=idx)
        res = test_continuity(pfc)
        assert res.passed
        assert res.observed == 0.0

    def test_5eur_jump_at_month_boundary_fails(self):
        """PFC avec saut de 5 €/MWh aux frontières mois local → passed=False."""
        from pfc_shaping.validation.structural_tests import test_continuity

        idx = _hourly_index_2024_2025_utc()
        idx_local = idx.tz_convert("Europe/Zurich")
        # Use LOCAL month (cohérent avec la sémantique de test_continuity)
        values = np.where(idx_local.month == 1, 50.0, 55.0).astype(float)
        pfc = pd.Series(values, index=idx)
        res = test_continuity(pfc, max_jump=2.0)
        assert not res.passed, (
            f"Expected fail with 5€ jump, got passed=True observed={res.observed}"
        )
        assert res.observed >= 5.0 - 1e-9, (
            f"Expected jump >= 5.0, got {res.observed}"
        )

    def test_single_month_edge_case_no_boundary(self):
        """PFC sur 1 seul mois local → aucune frontière → passed=True (max_jump=0)."""
        from pfc_shaping.validation.structural_tests import test_continuity

        # Construire un index strictement contenu dans 1 mois LOCAL CH
        # (= 2024-01-02 00:00 UTC à 2024-01-30 23:00 UTC pour éviter
        # le débordement UTC↔local au 31 janvier).
        idx = pd.date_range(
            "2024-01-02", "2024-01-31", freq="1h", inclusive="left", tz="UTC"
        )
        pfc = pd.Series(np.linspace(40, 60, len(idx)), index=idx)
        res = test_continuity(pfc, max_jump=2.0)
        assert res.passed
        # 0 boundary (1 month local only) → max_jump_obs = 0
        assert res.observed == 0.0, (
            f"Single-month should yield 0 jump, got {res.observed}"
        )
        assert res.details["n_boundaries"] == 0
