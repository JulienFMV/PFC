"""
test_phase10_infra.py
---------------------
Tests synthétiques (déterministes, pas d'I/O parquet) pour l'infrastructure
de la Phase 10 PFC FMV Quality Scorecard (Plan 10-01, wave 1).

Couvre :
    - 5 BlockMask : nommage + tz-safety + sémantique cross-midnight pour
      overnight_weekday + saisonnalité pour summer_solar_bowl + winter peak
    - scorecard.AblationConfig grid integrity (4 configs D-A6-1)
    - list_vintages_2024_2025 → 24 timestamps tz-aware UTC
    - last_business_day_of_month edge cases
    - derive_forwards_from_epex_hist skeleton callable (signature only)
    - constants FORWARDS_SOURCE_REAL / FORWARDS_SOURCE_FALLBACK_DIAGNOSTIC
    - conftest.py autouse `_pfc_lt_env_hygiene` snapshot des 2 flags
      Phase 10 par préfixe PFC_LT_*

Runtime cible : < 5s total (aucun I/O parquet, données synthétiques).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Block masks — module exists + ALL_BLOCKS contains the 5 D-A2-2 blocks
# ---------------------------------------------------------------------------


def test_all_blocks_count_and_names():
    """ALL_BLOCKS expose exactement les 5 blocs client D-A2-2."""
    from pfc_shaping.validation.block_masks import ALL_BLOCKS

    assert len(ALL_BLOCKS) == 5
    expected = {
        "block_overnight_weekday",
        "block_midday_weekday",
        "block_weekend_midday",
        "block_summer_solar_bowl",
        "block_winter_evening_peak",
    }
    got = {b.name for b in ALL_BLOCKS}
    assert got == expected, f"Expected {expected}, got {got}"


def test_block_mask_tz_naive_raises():
    """apply() raise sur un index tz-naive (UTC attendu)."""
    from pfc_shaping.validation.block_masks import ALL_BLOCKS

    naive_idx = pd.date_range("2024-06-01", "2024-06-02", freq="15min")
    assert naive_idx.tz is None  # sanity

    for block in ALL_BLOCKS:
        with pytest.raises(ValueError, match="tz-aware"):
            block.apply(naive_idx)


def test_block_mask_returns_ndarray_aligned():
    """apply() retourne np.ndarray[bool] de longueur identique à idx_utc."""
    from pfc_shaping.validation.block_masks import ALL_BLOCKS

    idx = pd.date_range("2024-06-01", "2024-06-08", freq="15min", tz="UTC")
    for block in ALL_BLOCKS:
        mask = block.apply(idx)
        assert isinstance(mask, np.ndarray), f"{block.name}: not ndarray"
        assert mask.dtype == bool, f"{block.name}: dtype {mask.dtype} != bool"
        assert len(mask) == len(idx), (
            f"{block.name}: len {len(mask)} != idx len {len(idx)}"
        )


def test_block_overnight_weekday_cross_midnight_and_weekend_exclusion():
    """BlockOvernightWeekday : (h>=18 | h<9) & weekday.

    Vérifie sur une fenêtre 7 jours (Lun 3 juin → Dim 9 juin 2024) :
    - weekday : ~22 quart-d'heure × (15 + 9) heures = couvert
    - samedi/dimanche : FALSE partout
    """
    from pfc_shaping.validation.block_masks import BlockOvernightWeekday

    # Lundi 3 juin 2024 = weekday=0, Dimanche 9 juin 2024 = weekday=6
    idx = pd.date_range("2024-06-03", "2024-06-10", freq="15min", tz="UTC")
    mask = BlockOvernightWeekday().apply(idx)

    idx_local = idx.tz_convert("Europe/Zurich")
    df = pd.DataFrame(
        {"mask": mask, "weekday": idx_local.weekday, "hour": idx_local.hour},
        index=idx_local,
    )

    # Weekend (Sam=5, Dim=6) : TRUE jamais
    weekend = df[df["weekday"] >= 5]
    assert not weekend["mask"].any(), "Weekend rows should be FALSE"

    # Weekday h=20 : TRUE
    wd_evening = df[(df["weekday"] < 5) & (df["hour"] == 20)]
    assert wd_evening["mask"].all(), "Weekday h=20 should be TRUE"

    # Weekday h=3 : TRUE (cross midnight)
    wd_night = df[(df["weekday"] < 5) & (df["hour"] == 3)]
    assert wd_night["mask"].all(), "Weekday h=3 (cross midnight) should be TRUE"

    # Weekday h=12 : FALSE
    wd_noon = df[(df["weekday"] < 5) & (df["hour"] == 12)]
    assert not wd_noon["mask"].any(), "Weekday h=12 should be FALSE"


def test_block_summer_solar_bowl_seasonality():
    """BlockSummerSolarBowl est FALSE en avril/septembre, TRUE juin h11-13."""
    from pfc_shaping.validation.block_masks import BlockSummerSolarBowl

    # Indices spécifiques :
    # April 15, 2024 12:00 local Europe/Zurich → 10:00 UTC (UTC+2 DST)
    # June 15, 2024 12:00 local → 10:00 UTC
    # September 15, 2024 12:00 local → 10:00 UTC
    idx_apr = pd.date_range("2024-04-01", "2024-04-30", freq="1h", tz="UTC")
    idx_sep = pd.date_range("2024-09-01", "2024-09-30", freq="1h", tz="UTC")
    idx_jun = pd.date_range("2024-06-01", "2024-06-30", freq="1h", tz="UTC")

    mask_apr = BlockSummerSolarBowl().apply(idx_apr)
    mask_sep = BlockSummerSolarBowl().apply(idx_sep)
    mask_jun = BlockSummerSolarBowl().apply(idx_jun)

    assert not mask_apr.any(), "April should be FALSE"
    assert not mask_sep.any(), "September should be FALSE"

    # June : at least some hours should be TRUE (h11-13 local sur juin → existe)
    idx_jun_local = idx_jun.tz_convert("Europe/Zurich")
    df_jun = pd.DataFrame({"mask": mask_jun, "hour": idx_jun_local.hour})
    h12_jun = df_jun[df_jun["hour"] == 12]
    assert h12_jun["mask"].all(), "June h=12 local should be TRUE"


def test_block_winter_evening_peak_seasonality():
    """BlockWinterEveningPeak : 17-21 weekday novembre-février."""
    from pfc_shaping.validation.block_masks import BlockWinterEveningPeak

    # Janvier 2024 (Hiver, contient bloc)
    idx_jan = pd.date_range("2024-01-01", "2024-01-31", freq="1h", tz="UTC")
    # Juin 2024 (été, pas dans bloc)
    idx_jun = pd.date_range("2024-06-01", "2024-06-30", freq="1h", tz="UTC")

    mask_jan = BlockWinterEveningPeak().apply(idx_jan)
    mask_jun = BlockWinterEveningPeak().apply(idx_jun)

    assert mask_jan.any(), "January should have TRUE entries (winter evening peak)"
    assert not mask_jun.any(), "June should be FALSE (not winter)"


# ---------------------------------------------------------------------------
# scorecard.py skeleton — AblationConfig + list_vintages + helpers + constants
# ---------------------------------------------------------------------------


def test_ablation_grid_has_4_configs_with_correct_flags():
    """ABLATION_GRID = 4 configs D-A6-1 avec les noms et flags exacts."""
    from pfc_shaping.validation.scorecard import ABLATION_GRID

    assert len(ABLATION_GRID) == 4

    names = {c.name for c in ABLATION_GRID}
    expected = {
        "bowl_off_floors_on",   # legacy
        "bowl_on_floors_on",    # 5bisB no Phase5
        "bowl_off_floors_off",  # Phase5 no 5bisB
        "bowl_on_floors_off",   # production target
    }
    assert names == expected, f"Expected {expected}, got {names}"

    # Spot-check production target = bowl_on_floors_off
    by_name = {c.name: c for c in ABLATION_GRID}
    prod = by_name["bowl_on_floors_off"]
    assert prod.use_seasonal_hourly is True
    assert prod.enforce_positivity is False
    assert prod.enforce_m_factor_floor is False
    assert prod.enforce_floor is False
    assert prod.allow_negative_peak is True

    # Spot-check legacy = bowl_off_floors_on
    legacy = by_name["bowl_off_floors_on"]
    assert legacy.use_seasonal_hourly is False
    assert legacy.enforce_positivity is True
    assert legacy.enforce_m_factor_floor is True
    assert legacy.enforce_floor is True
    assert legacy.allow_negative_peak is False


def test_list_vintages_2024_2025_24_timestamps_tz_aware_utc():
    """24 timestamps tz-aware UTC, premier 31 jan 2024 18:00 local (= 17:00 UTC hiver)."""
    from pfc_shaping.validation.scorecard import list_vintages_2024_2025

    vintages = list_vintages_2024_2025()
    assert len(vintages) == 24

    for v in vintages:
        assert isinstance(v, pd.Timestamp)
        assert v.tz is not None, "vintage must be tz-aware"
        # tz_convert UTC must succeed without exception
        v_utc = v.tz_convert("UTC")
        assert v_utc.tz is not None

    # Premier : 31 janvier 2024 (mercredi) — last business day of Jan 2024
    # 18:00 Europe/Zurich (UTC+1 en hiver) → 17:00 UTC
    first = vintages[0].tz_convert("UTC")
    assert first.year == 2024
    assert first.month == 1
    assert first.day == 31
    assert first.hour == 17  # 18:00 local CET = 17:00 UTC

    # Dernier : 31 décembre 2025 (mercredi) — last business day of Dec 2025
    # 18:00 Europe/Zurich (UTC+1 hiver) → 17:00 UTC
    last = vintages[-1].tz_convert("UTC")
    assert last.year == 2025
    assert last.month == 12
    assert last.day == 31
    assert last.hour == 17


def test_last_business_day_of_month_march_2024():
    """last_business_day_of_month(2024, 3) → 29 mars 2024 (vendredi).

    BMonthEnd(0) ignore Karfreitag — accepted per RESEARCH Pitfall 4.
    """
    from pfc_shaping.validation.scorecard import last_business_day_of_month

    ts = last_business_day_of_month(2024, 3)
    assert isinstance(ts, pd.Timestamp)
    assert ts.tz is not None

    ts_utc = ts.tz_convert("UTC")
    # 29 mars 2024 = vendredi (Karfreitag, mais BMonthEnd l'ignore)
    # 18:00 Europe/Zurich en mars = UTC+1 (avant DST end Mar) → 17:00 UTC
    # MAIS DST start = dernier dimanche mars 2024 = 31 mars → 29 mars est encore en CET
    assert ts_utc.year == 2024
    assert ts_utc.month == 3
    assert ts_utc.day == 29
    assert ts_utc.hour == 17  # 18:00 local CET (CEST commence le 31 mars 2024)


def test_derive_forwards_skeleton_callable():
    """derive_forwards_from_epex_hist exposé (signature only OK pour Plan 10-01)."""
    from pfc_shaping.validation.scorecard import derive_forwards_from_epex_hist

    assert callable(derive_forwards_from_epex_hist)


def test_forwards_source_constants_defined():
    """C3 REVIEWS : FORWARDS_SOURCE_REAL / _FALLBACK_DIAGNOSTIC string constants."""
    from pfc_shaping.validation.scorecard import (
        FORWARDS_SOURCE_REAL,
        FORWARDS_SOURCE_FALLBACK_DIAGNOSTIC,
    )

    assert FORWARDS_SOURCE_FALLBACK_DIAGNOSTIC == "fallback_diagnostic"
    assert FORWARDS_SOURCE_REAL == "real_eex_xlsx"


def test_run_scorecard_pillar_1_now_wired():
    """run_scorecard_pillar_1 est wired Plan 10-02 (n'est plus un stub).

    Plan 10-01 : stub raise NotImplementedError.
    Plan 10-02 : wired → couverture deep dans tests/test_phase10_hildmann.py
    (4 SC#1 + ≥12 unit tests). Ici on assert juste l'absence de
    NotImplementedError pour catch un éventuel revert accidentel.

    Le code path complet est testé par tests/test_phase10_hildmann.py et
    ne devrait pas être ré-exécuté ici (runtime ≈ 14s, lourd).
    """
    from pfc_shaping.validation.scorecard import run_scorecard_pillar_1

    # Smoke : la fonction est appelable avec un epex_source invalide → raise
    # ValueError (PAS NotImplementedError, qui prouverait un revert vers Plan 10-01).
    with pytest.raises(ValueError, match="epex_source"):
        run_scorecard_pillar_1(
            "bowl_on_floors_off",
            cache_dir=".",
            epex_source="invalid_source",
        )


# ---------------------------------------------------------------------------
# conftest._pfc_lt_env_hygiene : snapshot/restore les 2 flags PFC_LT_*
# ---------------------------------------------------------------------------


def test_conftest_env_hygiene_covers_use_seasonal_hourly_shape():
    """La fixture autouse snapshot/restore PFC_LT_USE_SEASONAL_HOURLY_SHAPE."""
    key = "PFC_LT_USE_SEASONAL_HOURLY_SHAPE"
    # Le test précédent peut avoir laissé l'env propre — set ici, vérifier
    # qu'il sera retiré par la fixture autouse en sortie de test.
    assert key not in os.environ, (
        f"sanity: {key} must be absent at test start (fixture autouse cleans previous tests)"
    )
    os.environ[key] = "1"
    assert os.environ.get(key) == "1"
    # Fin du test → fixture autouse remet l'env d'avant le yield (= absent)


def test_conftest_env_hygiene_restored_after_previous_test():
    """Si le test précédent a set PFC_LT_USE_SEASONAL_HOURLY_SHAPE, la fixture
    autouse l'a retiré avant ce test (preuve restore)."""
    key = "PFC_LT_USE_SEASONAL_HOURLY_SHAPE"
    assert key not in os.environ, (
        f"{key} should have been restored to absent by autouse fixture "
        "(set in previous test, must be cleaned)"
    )


def test_conftest_env_hygiene_covers_allow_negative_prices():
    """La fixture autouse snapshot/restore PFC_LT_ALLOW_NEGATIVE_PRICES."""
    key = "PFC_LT_ALLOW_NEGATIVE_PRICES"
    assert key not in os.environ, f"sanity: {key} must be absent at test start"
    os.environ[key] = "1"
    assert os.environ.get(key) == "1"


def test_conftest_env_hygiene_restored_allow_negative_prices():
    """Vérifie restore de PFC_LT_ALLOW_NEGATIVE_PRICES après test précédent."""
    key = "PFC_LT_ALLOW_NEGATIVE_PRICES"
    assert key not in os.environ, (
        f"{key} should have been restored to absent by autouse fixture"
    )
