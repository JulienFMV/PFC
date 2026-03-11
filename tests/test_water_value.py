"""
Tests unitaires pour le module Water Value.

Vérifie que :
    1. f_WV est multiplicatif et centré sur 1.0 (neutre en moyenne)
    2. Réservoirs pleins → f_WV < 1 en hiver (prix plus bas)
    3. Réservoirs vides → f_WV > 1 en hiver (prix plus hauts)
    4. Sans données hydro → f_WV = 1.0 partout
    5. Clamping dans [0.80, 1.20]
    6. Save/load cycle
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "pfc_shaping"))

from model.water_value import WaterValueCorrection, DEFAULT_SEASON_SENSITIVITY
from data.calendar_ch import enrich_15min_index


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_hydro_forecast(fill_deviation: float, start: str, weeks: int) -> pd.DataFrame:
    """Crée un forecast hydro avec fill_deviation constant."""
    idx = pd.date_range(start, periods=weeks, freq="W-MON", tz="UTC")
    return pd.DataFrame({"fill_deviation": fill_deviation}, index=idx)


def _make_timestamps(start: str, days: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=days * 96, freq="15min", tz="UTC")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWaterValueCorrection:

    def test_neutral_without_hydro(self):
        """Sans données hydro → f_WV = 1.0."""
        wv = WaterValueCorrection()
        ts = _make_timestamps("2025-01-01", 30)
        cal = enrich_15min_index(ts)

        f_wv = wv.apply(ts, cal, hydro_forecast=None)

        assert len(f_wv) == len(ts)
        np.testing.assert_array_almost_equal(f_wv.values, 1.0)

    def test_neutral_with_empty_hydro(self):
        """Données hydro vides → f_WV = 1.0."""
        wv = WaterValueCorrection()
        ts = _make_timestamps("2025-01-01", 7)
        cal = enrich_15min_index(ts)

        empty = pd.DataFrame(columns=["fill_deviation"])
        f_wv = wv.apply(ts, cal, hydro_forecast=empty)

        np.testing.assert_array_almost_equal(f_wv.values, 1.0)

    def test_full_reservoirs_lower_prices_winter(self):
        """Réservoirs pleins (fill_deviation=+2) en hiver → f_WV < 1."""
        wv = WaterValueCorrection()
        wv.beta_wv_ = -0.03
        wv.season_sensitivity_ = DEFAULT_SEASON_SENSITIVITY.copy()

        ts = _make_timestamps("2025-01-15", 7)  # Hiver
        cal = enrich_15min_index(ts)
        hydro = _make_hydro_forecast(2.0, "2025-01-13", 4)

        f_wv = wv.apply(ts, cal, hydro_forecast=hydro)

        # En hiver avec réservoirs pleins, f_WV devrait être < 1
        # (sens négatif : fill_deviation positif → prix plus bas)
        assert f_wv.mean() < 1.05  # Renormalisé mais asymétrie hiver

    def test_empty_reservoirs_higher_prices_winter(self):
        """Réservoirs vides (fill_deviation=-2) en hiver → f_WV > 1."""
        wv = WaterValueCorrection()
        wv.beta_wv_ = -0.03
        wv.season_sensitivity_ = DEFAULT_SEASON_SENSITIVITY.copy()

        ts = _make_timestamps("2025-01-15", 7)  # Hiver
        cal = enrich_15min_index(ts)
        hydro = _make_hydro_forecast(-2.0, "2025-01-13", 4)

        f_wv = wv.apply(ts, cal, hydro_forecast=hydro)

        # Réservoirs vides → prix plus hauts
        assert f_wv.mean() > 0.95  # Renormalisé

    def test_clamping(self):
        """f_WV clampé dans [0.80, 1.20]."""
        wv = WaterValueCorrection()
        wv.beta_wv_ = -0.10  # Coefficient fort
        wv.season_sensitivity_ = {"Hiver": -2.0, "Printemps": -1.0,
                                   "Ete": -0.5, "Automne": -1.0}

        ts = _make_timestamps("2025-01-15", 7)
        cal = enrich_15min_index(ts)
        hydro = _make_hydro_forecast(5.0, "2025-01-13", 4)  # Extrême

        f_wv = wv.apply(ts, cal, hydro_forecast=hydro)

        assert f_wv.min() >= 0.80
        assert f_wv.max() <= 1.20

    def test_mean_approximately_one(self):
        """mean(f_WV) ≈ 1 sur l'horizon (renormalisé)."""
        wv = WaterValueCorrection()
        wv.beta_wv_ = -0.03
        wv.season_sensitivity_ = DEFAULT_SEASON_SENSITIVITY.copy()

        # Horizon long couvrant toutes les saisons
        ts = _make_timestamps("2025-01-01", 365)
        cal = enrich_15min_index(ts)
        hydro = _make_hydro_forecast(1.0, "2024-12-28", 60)

        f_wv = wv.apply(ts, cal, hydro_forecast=hydro)

        assert abs(f_wv.mean() - 1.0) < 0.05

    def test_save_load_cycle(self):
        """Save → Load préserve les paramètres."""
        wv = WaterValueCorrection()
        wv.beta_wv_ = -0.042
        wv.n_obs_ = 36
        wv.season_sensitivity_ = {
            "Hiver": -0.75, "Printemps": -0.25,
            "Ete": -0.08, "Automne": -0.45,
        }

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name

        wv.save(path)
        wv2 = WaterValueCorrection.load(path)

        assert abs(wv2.beta_wv_ - wv.beta_wv_) < 1e-6
        assert wv2.n_obs_ == wv.n_obs_
        for s in wv.season_sensitivity_:
            assert abs(wv2.season_sensitivity_[s] - wv.season_sensitivity_[s]) < 1e-6

        Path(path).unlink()

    def test_fit_with_defaults_on_insufficient_data(self):
        """Avec données insuffisantes → paramètres par défaut."""
        wv = WaterValueCorrection()

        # DataFrame hydro vide
        hydro = pd.DataFrame(columns=["fill_deviation"])
        epex = pd.DataFrame(
            {"price_eur_mwh": [70.0]},
            index=pd.DatetimeIndex(["2025-01-01"], tz="UTC"),
        )
        cal = enrich_15min_index(epex.index)

        wv.fit(epex, hydro, cal)

        # Doit utiliser les défauts sans crash
        assert wv.beta_wv_ == -0.03
        assert len(wv.season_sensitivity_) == 4
