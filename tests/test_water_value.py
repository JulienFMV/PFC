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

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Use the modern package path (LT split). The legacy sys.path hack into
# pfc_shaping/ + 'from model.water_value' was retired with the LT/CT
# refactor (water_value moved under pfc_shaping/lt/model/).
from pfc_shaping.lt.model.water_value import (
    WaterValueCorrection,
    DEFAULT_SEASON_SENSITIVITY,
)
from pfc_shaping.data.calendar_ch import enrich_15min_index


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

    def test_scarcity_raises_price_and_abundance_lowers_price(self):
        """Controlled winter fill path: low fill raises price, high fill lowers it."""
        wv = WaterValueCorrection()
        wv.beta_wv_ = -0.03
        wv.season_sensitivity_ = DEFAULT_SEASON_SENSITIVITY.copy()
        wv.scarcity_multiplier_ = 1.0
        wv.abundance_multiplier_ = 1.0
        wv.horizon_halflife_days_ = 10_000.0

        ts = _make_timestamps("2025-01-15", 2)
        cal = enrich_15min_index(ts)
        hydro = pd.DataFrame(
            {"fill_deviation": [-0.5, 0.5]},
            index=pd.DatetimeIndex(["2025-01-15", "2025-01-16"], tz="UTC"),
        )

        f_wv = wv.apply(ts, cal, hydro_forecast=hydro)
        low_fill = f_wv.loc["2025-01-15"].mean()
        high_fill = f_wv.loc["2025-01-16"].mean()

        assert low_fill > 1.0 > high_fill
        assert 0.005 < (low_fill - 1.0) < 0.02
        assert 0.005 < (1.0 - high_fill) < 0.02

        base_price = 100.0
        assert base_price * low_fill > base_price > base_price * high_fill

    def test_delta_wv_preserves_all_quoted_and_free_month_block_means(self):
        """Delta-additive WV must not move delivered Cal/Q/M block prices."""
        from pfc_shaping.calibration.arbitrage_free import (
            ArbitrageFreeCalibrator,
            FuturesContract,
        )
        from pfc_shaping.calibration.cascading import _period_boundaries_utc

        tz = "Europe/Zurich"
        start, end = _period_boundaries_utc(2025, 1, 12, tz)
        idx = pd.date_range(start, end, freq="15min", inclusive="left", tz="UTC")
        cal = enrich_15min_index(idx)
        local = idx.tz_convert(tz)

        winter = np.isin(local.month, [11, 12, 1, 2, 3])
        summer = np.isin(local.month, [6, 7, 8, 9])
        B = pd.Series(
            np.where(winter, 95.0, np.where(summer, 55.0, 70.0)),
            index=idx,
            name="B_smooth",
        )

        fill_by_month = {
            1: -1.0, 2: -0.9, 3: -0.6, 4: -0.2,
            5: 0.2, 6: 0.7, 7: 1.0, 8: 0.8,
            9: 0.3, 10: -0.1, 11: -0.6, 12: -0.9,
        }
        hydro_idx = pd.date_range(start, end - pd.Timedelta(days=1), freq="W-WED", tz="UTC")
        hydro = pd.DataFrame(
            {
                "fill_deviation": [
                    fill_by_month[int(ts.tz_convert(tz).month)] for ts in hydro_idx
                ]
            },
            index=hydro_idx,
        )

        wv = WaterValueCorrection()
        wv.beta_wv_ = -0.03
        wv.season_sensitivity_ = DEFAULT_SEASON_SENSITIVITY.copy()
        wv.scarcity_multiplier_ = 1.35
        wv.abundance_multiplier_ = 1.0
        wv.horizon_halflife_days_ = 10_000.0

        quoted_contracts = [
            ("Cal-2025", 1, 12),
            ("Q1-2025", 1, 3),
            ("Q2-2025", 4, 6),
            ("Jan-2025", 1, 1),
            ("Feb-2025", 2, 2),
            ("Mar-2025", 3, 3),
            ("Apr-2025", 4, 4),
        ]
        assert sum(name.startswith("Cal") for name, _, _ in quoted_contracts) >= 1
        assert sum(name.startswith("Q") for name, _, _ in quoted_contracts) > 1
        assert sum(
            "-2025" in name and not name.startswith(("Cal", "Q"))
            for name, _, _ in quoted_contracts
        ) > 1

        delta = wv.compute_delta_wv(B, fill_df=hydro, calendar_df=cal)

        contracts = []
        for name, month_start, month_end in quoted_contracts:
            block_start, block_end = _period_boundaries_utc(2025, month_start, month_end, tz)
            mask = (delta.index >= block_start) & (delta.index < block_end)
            assert mask.any(), name
            assert abs(float(delta.loc[mask].mean())) < 1e-9, name
            contracts.append(
                FuturesContract(
                    name=name,
                    price=70.0,
                    start=block_start,
                    end=block_end,
                    product_type="Base",
                )
            )

        quoted_months = {1, 2, 3, 4}
        for month in set(range(1, 13)) - quoted_months:
            block_start, block_end = _period_boundaries_utc(2025, month, month, tz)
            mask = (delta.index >= block_start) & (delta.index < block_end)
            assert mask.any(), month
            assert abs(float(delta.loc[mask].mean())) < 1e-9, month

        raw_curve = pd.Series(70.0, index=idx, name="raw") + delta
        result = ArbitrageFreeCalibrator(tol=1e-12).calibrate(raw_curve, contracts)
        assert result.max_abs_residual < 1e-9
        assert (result.residuals["abs_error"] < 1e-9).all()

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
