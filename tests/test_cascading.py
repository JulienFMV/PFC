"""
Tests unitaires pour le module de cascading des contrats futures.

Vérifie que :
    1. La conservation d'énergie est respectée (Year→Q→M)
    2. Les heures sont comptées correctement (DST, leap years)
    3. Les contrats existants ne sont pas écrasés
    4. Les ratios saisonniers par défaut sont raisonnables
    5. Peak/Off-Peak decomposition est correcte
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "pfc_shaping"))

from calibration.cascading import (
    ContractCascader,
    ContractSpec,
    count_hours,
    parse_key,
    quarter_key,
    month_key,
)


# ---------------------------------------------------------------------------
# Tests parsing
# ---------------------------------------------------------------------------

class TestParseKey:
    def test_year(self):
        assert parse_key("2026") == ("Cal", 2026, None)

    def test_quarter(self):
        assert parse_key("2026-Q1") == ("Quarter", 2026, 1)
        assert parse_key("2026-Q4") == ("Quarter", 2026, 4)

    def test_month(self):
        assert parse_key("2026-03") == ("Month", 2026, 3)
        assert parse_key("2026-12") == ("Month", 2026, 12)

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_key("invalid")
        with pytest.raises(ValueError):
            parse_key("2026-W03")


# ---------------------------------------------------------------------------
# Tests comptage heures
# ---------------------------------------------------------------------------

class TestCountHours:
    def test_january_2025(self):
        """Janvier 2025 : 31 jours × 24h = 744h."""
        total, peak, offpeak = count_hours(2025, 1, 1)
        assert total == 744
        assert peak + offpeak == total
        assert peak > 0

    def test_february_leap_year(self):
        """Février 2024 (bissextile) : 29 jours."""
        total, _, _ = count_hours(2024, 2, 2)
        assert total == 29 * 24

    def test_february_non_leap(self):
        """Février 2025 (non bissextile) : 28 jours."""
        total, _, _ = count_hours(2025, 2, 2)
        assert total == 28 * 24

    def test_full_year_hours(self):
        """Un an = 8760h (non bissextile) ou 8784h (bissextile)."""
        total_2025, _, _ = count_hours(2025, 1, 12)
        assert total_2025 == 8760

        total_2024, _, _ = count_hours(2024, 1, 12)
        assert total_2024 == 8784

    def test_peak_is_subset(self):
        """Peak hours < total hours."""
        total, peak, offpeak = count_hours(2025, 1, 3)  # Q1
        assert 0 < peak < total
        assert peak + offpeak == total

    def test_dst_transition_march(self):
        """Mars 2025 : transition CET→CEST → 1 heure de moins."""
        total, _, _ = count_hours(2025, 3, 3)
        # Mars a 31 jours, mais perd 1h au changement d'heure
        assert total == 31 * 24 - 1

    def test_dst_transition_october(self):
        """Octobre 2025 : transition CEST→CET → 1 heure de plus."""
        total, _, _ = count_hours(2025, 10, 10)
        assert total == 31 * 24 + 1


# ---------------------------------------------------------------------------
# Tests cascading
# ---------------------------------------------------------------------------

class TestContractCascader:
    def test_year_to_quarters_energy_conservation(self):
        """Cal 2025 cascadé en 4 Q → conservation d'énergie."""
        cascader = ContractCascader()
        result = cascader.cascade({"2025": 75.0})

        # Tous les quarters doivent exister
        for q in range(1, 5):
            assert quarter_key(2025, q) in result

        # Conservation : weighted avg des quarters = year price
        total_hours = 0
        weighted_sum = 0.0
        for q in range(1, 5):
            ms, me = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}[q]
            total_h, _, _ = count_hours(2025, ms, me)
            weighted_sum += result[quarter_key(2025, q)] * total_h
            total_hours += total_h

        weighted_avg = weighted_sum / total_hours
        assert abs(weighted_avg - 75.0) < 0.001

    def test_quarter_to_months_energy_conservation(self):
        """Q1 2025 cascadé en 3 mois → conservation d'énergie."""
        cascader = ContractCascader()
        result = cascader.cascade({"2025-Q1": 80.0})

        for m in [1, 2, 3]:
            assert month_key(2025, m) in result

        total_hours = 0
        weighted_sum = 0.0
        for m in [1, 2, 3]:
            total_h, _, _ = count_hours(2025, m, m)
            weighted_sum += result[month_key(2025, m)] * total_h
            total_hours += total_h

        weighted_avg = weighted_sum / total_hours
        assert abs(weighted_avg - 80.0) < 0.001

    def test_full_cascade_year_to_months(self):
        """Cal 2025 → Q1..Q4 → all 12 months, conservation partout."""
        cascader = ContractCascader()
        result = cascader.cascade({"2025": 75.0})

        # 12 mois doivent exister
        for m in range(1, 13):
            assert month_key(2025, m) in result

    def test_existing_prices_not_overwritten(self):
        """Un prix existant n'est jamais écrasé."""
        cascader = ContractCascader()
        prices = {"2025": 75.0, "2025-Q1": 82.0}
        result = cascader.cascade(prices)

        # Q1 doit rester à 82.0 (pas écrasé)
        assert result["2025-Q1"] == 82.0

    def test_partial_quarters(self):
        """Si 3 quarters connus, le 4ème est déduit par résidu."""
        cascader = ContractCascader()
        prices = {
            "2025": 75.0,
            "2025-Q1": 80.0,
            "2025-Q2": 70.0,
            "2025-Q3": 65.0,
        }
        result = cascader.cascade(prices)

        # Q4 doit être calculé pour respecter la conservation
        assert "2025-Q4" in result
        assert result["2025-Q1"] == 80.0  # inchangé

    def test_offpeak_price_calculation(self):
        """Off-peak = (Base × total - Peak × peak_h) / offpeak_h."""
        total_h, peak_h, offpeak_h = count_hours(2025, 1, 1)
        offpeak = ContractCascader.offpeak_price(
            base_price=70.0,
            peak_price=90.0,
            year=2025,
            month_start=1,
            month_end=1,
        )
        # Vérification : Base × total = Peak × peak_h + OffPeak × offpeak_h
        recon = (90.0 * peak_h + offpeak * offpeak_h) / total_h
        assert abs(recon - 70.0) < 0.001

    def test_fit_seasonal_ratios(self):
        """Fitting de ratios saisonniers sur des données synthétiques."""
        # Données synthétiques : 3 ans de prix horaires
        rng = np.random.default_rng(42)
        idx = pd.date_range("2022-01-01", "2024-12-31 23:00", freq="h", tz="UTC")
        # Saisonnalité : hiver haut, été bas
        month = idx.tz_convert("Europe/Zurich").month
        seasonal = np.where(month.isin([11, 12, 1, 2, 3]), 1.15, 0.90)
        prices = 70.0 * seasonal + rng.normal(0, 5, len(idx))
        spot = pd.DataFrame({"price_eur_mwh": prices}, index=idx)

        cascader = ContractCascader()
        cascader.fit_seasonal_ratios(spot)

        assert cascader.seasonal_ratios_ is not None
        # Winter quarters should be > 1
        assert cascader.seasonal_ratios_["quarter"][1] > 1.0
        # Summer quarters should be < 1
        assert cascader.seasonal_ratios_["quarter"][3] < 1.0

    def test_build_contract_specs(self):
        """Conversion base_prices → ContractSpec objects."""
        cascader = ContractCascader()
        specs = cascader.build_contract_specs({
            "2025": 75.0,
            "2025-Q1": 80.0,
            "2025-01": 82.0,
        })

        assert len(specs) == 3
        types = {s.product_type for s in specs}
        assert types == {"Cal", "Quarter", "Month"}
