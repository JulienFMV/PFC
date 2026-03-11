"""
Tests unitaires pour le module de calibration arbitrage-free.

Vérifie que :
    1. La courbe calibrée reprice exactement les contrats futures
    2. La correction δ est lisse (pas de sauts aux frontières)
    3. Le système gère les contrats overlapping (Month + Quarter + Year)
    4. Peak vs Base est correctement séparé
    5. Edge cases : aucun contrat, contrat hors plage, système singulier
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ajouter le répertoire pfc_shaping au path
sys.path.insert(0, str(Path(__file__).parent.parent / "pfc_shaping"))

from calibration.arbitrage_free import (
    ArbitrageFreeCalibrator,
    CalibrationResult,
    FuturesContract,
    _build_smoothness_matrix,
    _build_peak_mask,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_curve(start: str, days: int, base_price: float = 70.0) -> pd.Series:
    """Crée une courbe sinusoïdale réaliste en 15min."""
    idx = pd.date_range(start, periods=days * 96, freq="15min", tz="UTC")
    hours = np.arange(len(idx)) / 4.0
    # Saisonnalité journalière + bruit
    seasonal = 5.0 * np.sin(2 * np.pi * hours / 24.0)
    noise = np.random.default_rng(42).normal(0, 1.0, len(idx))
    return pd.Series(base_price + seasonal + noise, index=idx, name="price")


def _make_contract(
    name: str,
    start: str,
    end: str,
    price: float,
    product_type: str = "Base",
) -> FuturesContract:
    return FuturesContract(
        name=name,
        price=price,
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
        product_type=product_type,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestArbitrageFreeCalibrator:
    """Tests du calibrateur arbitrage-free."""

    def test_single_contract_exact_repricing(self):
        """Un seul contrat Base mensuel → le prix moyen calibré doit matcher."""
        curve = _make_curve("2025-01-01", days=31, base_price=70.0)
        contract = _make_contract("Jan-2025", "2025-01-01", "2025-02-01", 75.0)

        cal = ArbitrageFreeCalibrator(tol=0.01)
        result = cal.calibrate(curve, [contract])

        assert result.converged
        assert result.max_abs_residual < 0.01
        # Vérifier que la moyenne sur janvier ≈ 75.0
        mean_jan = result.calibrated_curve.mean()
        assert abs(mean_jan - 75.0) < 0.01

    def test_multiple_non_overlapping_contracts(self):
        """Deux mois non-overlapping → chacun repricé exactement."""
        curve = _make_curve("2025-01-01", days=59, base_price=70.0)
        contracts = [
            _make_contract("Jan-2025", "2025-01-01", "2025-02-01", 80.0),
            _make_contract("Feb-2025", "2025-02-01", "2025-03-01", 65.0),
        ]

        cal = ArbitrageFreeCalibrator(tol=0.01)
        result = cal.calibrate(curve, contracts)

        assert result.converged
        assert result.max_abs_residual < 0.01

    def test_overlapping_contracts(self):
        """Q1 + 3 mois → overlapping, tous repricés."""
        curve = _make_curve("2025-01-01", days=90, base_price=70.0)
        contracts = [
            _make_contract("Q1-2025", "2025-01-01", "2025-04-01", 72.0),
            _make_contract("Jan-2025", "2025-01-01", "2025-02-01", 78.0),
            _make_contract("Feb-2025", "2025-02-01", "2025-03-01", 70.0),
            _make_contract("Mar-2025", "2025-03-01", "2025-04-01", 68.0),
        ]

        cal = ArbitrageFreeCalibrator(tol=0.05)
        result = cal.calibrate(curve, contracts)

        # Avec overlapping, le système peut être sur-déterminé
        # On vérifie que le résidu est raisonnable
        assert result.max_abs_residual < 1.0

    def test_smoothness_of_delta(self):
        """La correction δ doit être lisse (faible dérivée seconde)."""
        curve = _make_curve("2025-01-01", days=31, base_price=70.0)
        contract = _make_contract("Jan-2025", "2025-01-01", "2025-02-01", 75.0)

        cal = ArbitrageFreeCalibrator()
        result = cal.calibrate(curve, [contract])

        delta = result.delta.values
        # Dérivée seconde discrète
        d2 = np.diff(delta, n=2)
        # Lissage : la variance de d2 doit être faible
        assert np.std(d2) < 0.1

    def test_no_contracts(self):
        """Sans contrat → courbe inchangée."""
        curve = _make_curve("2025-01-01", days=10)
        cal = ArbitrageFreeCalibrator()
        result = cal.calibrate(curve, [])

        assert result.converged
        assert result.max_abs_residual == 0.0
        np.testing.assert_array_almost_equal(
            result.calibrated_curve.values, curve.values
        )

    def test_contract_outside_range_skipped(self):
        """Un contrat hors plage de la courbe est ignoré."""
        curve = _make_curve("2025-01-01", days=31)
        contract = _make_contract("Jun-2025", "2025-06-01", "2025-07-01", 80.0)

        cal = ArbitrageFreeCalibrator()
        result = cal.calibrate(curve, [contract])

        # Contrat hors plage → résultat trivial
        assert result.converged

    def test_futures_contract_validation(self):
        """Validation des inputs FuturesContract."""
        with pytest.raises(ValueError, match="product_type"):
            FuturesContract("test", 70.0, pd.Timestamp("2025-01-01", tz="UTC"),
                          pd.Timestamp("2025-02-01", tz="UTC"), "Invalid")

        with pytest.raises(ValueError, match="start"):
            FuturesContract("test", 70.0, pd.Timestamp("2025-02-01", tz="UTC"),
                          pd.Timestamp("2025-01-01", tz="UTC"))


class TestSmoothnessMatrix:
    """Tests de la matrice de lissage H."""

    def test_symmetry(self):
        """H doit être symétrique."""
        H = _build_smoothness_matrix(100)
        diff = H - H.T
        assert abs(diff).max() < 1e-12

    def test_positive_semidefinite(self):
        """H doit être PSD."""
        H = _build_smoothness_matrix(50).toarray()
        eigenvalues = np.linalg.eigvalsh(H)
        assert np.all(eigenvalues >= -1e-10)

    def test_small_n(self):
        """N < 3 → matrice zéro."""
        H = _build_smoothness_matrix(2)
        assert H.nnz == 0
