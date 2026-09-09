"""
Test the v2 Uncertainty (empirical residual quantile) calibration contract.

Reference: pfc_shaping/lt/model/uncertainty.py (v2 rewrite — empirical
residual quantiles replacing v1 multiplicative-ratio bootstrap + HORIZON_WIDENING
multiplier that produced observed IC80 coverage of 99% vs nominal 80%).

Contract under test:
    Calibration set in-sample coverage must approach nominal 20% by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.model.uncertainty import (
    FALLBACK_Q10_RES,
    FALLBACK_Q90_RES,
    MIN_CELL_OBS,
    NOMINAL_TAIL,
    Uncertainty,
)


def _synthetic_epex_and_calendar(
    n_years: int = 2, sigma: float = 30.0, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build deterministic synthetic EPEX with known per-cell mean + noise."""
    rng = np.random.default_rng(seed)
    n = 24 * 365 * n_years
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    months = idx.month
    saisons = np.where(np.isin(months, [6, 7, 8]), "ete", "hiver")
    type_jour = np.where(idx.dayofweek < 5, "semaine", "weekend")
    heure = idx.hour.values
    # Cell mean = 50 + 20·sin(2π·h/24) [€/MWh]
    base = 50.0 + 20.0 * np.sin(2.0 * np.pi * heure / 24.0)
    prices = base + rng.normal(0.0, sigma, size=n)
    epex = pd.DataFrame({"price_eur_mwh": prices}, index=idx)
    cal = pd.DataFrame(
        {
            "saison": saisons,
            "type_jour": type_jour,
            "heure_hce": heure,
            "quart": np.ones(n, dtype=int),
        },
        index=idx,
    )
    return epex, cal


class TestUncertaintyInSampleCoverage:
    """Empirical quantile method: in-sample coverage = nominal by construction."""

    def test_in_sample_coverage_within_2pct_of_nominal(self):
        """
        With perfect mean prediction, the empirical residual quantile bands
        achieve in-sample observed_p ≈ 20% to within sampling noise.
        For n ≈ 17520 and α=0.10, binomial 99% CI ≈ ±0.6%; tolerance ±2%
        is conservative.
        """
        epex, cal = _synthetic_epex_and_calendar(n_years=2, sigma=30.0)
        unc = Uncertainty().fit(epex, cal)

        # PFC = perfect cell-mean prediction
        base = 50.0 + 20.0 * np.sin(
            2.0 * np.pi * cal["heure_hce"].values / 24.0
        )
        pfc = pd.DataFrame({"price_shape": base}, index=epex.index)
        ic = unc.compute(pfc, cal)

        realised = epex["price_eur_mwh"].values
        violations = (realised < ic["p10"].values) | (realised > ic["p90"].values)
        observed_p = float(violations.mean())

        assert abs(observed_p - 0.20) < 0.02, (
            f"In-sample IC80 coverage {observed_p:.4f} deviates from "
            f"nominal 0.20 by more than 2%. Empirical quantile method should "
            f"approach the nominal coverage by construction."
        )

    def test_bandwidth_scales_with_residual_sigma(self):
        """Bandwidth ≈ 1.28σ each side ⇒ p90-p10 ≈ 2.56σ for Gaussian noise."""
        epex, cal = _synthetic_epex_and_calendar(n_years=2, sigma=30.0)
        unc = Uncertainty().fit(epex, cal)
        base = 50.0 + 20.0 * np.sin(
            2.0 * np.pi * cal["heure_hce"].values / 24.0
        )
        pfc = pd.DataFrame({"price_shape": base}, index=epex.index)
        ic = unc.compute(pfc, cal)
        bandwidth = (ic["p90"] - ic["p10"]).mean()
        # Theoretical: 2 * 1.2816 * 30 ≈ 76.9 €/MWh ; tolerance ±5 for finite-n noise.
        assert 70.0 < bandwidth < 85.0, (
            f"Mean bandwidth {bandwidth:.2f} €/MWh deviates from theoretical "
            f"2.56σ = 76.9 (σ=30) for empirical quantile method."
        )


class TestUncertaintyFallback:
    """Cells below MIN_CELL_OBS fall back to ±20 €/MWh symmetric residuals."""

    def test_sparse_cell_uses_fallback(self):
        """A cell with < MIN_CELL_OBS observations uses FALLBACK_Q10/Q90_RES."""
        n = MIN_CELL_OBS - 1  # one too few
        idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
        epex = pd.DataFrame({"price_eur_mwh": np.full(n, 50.0)}, index=idx)
        cal = pd.DataFrame(
            {
                "saison": ["hiver"] * n,
                "type_jour": ["semaine"] * n,
                "heure_hce": [0] * n,
                "quart": [1] * n,
            },
            index=idx,
        )
        unc = Uncertainty().fit(epex, cal)
        assert ("hiver", "semaine", 0, 1) in unc.boot_stats_
        stats = unc.boot_stats_[("hiver", "semaine", 0, 1)]
        assert stats["q10_res"] == FALLBACK_Q10_RES
        assert stats["q90_res"] == FALLBACK_Q90_RES


class TestUncertaintySaveLoad:
    """v2 parquet round-trip preserves residual quantiles exactly."""

    def test_round_trip_preserves_stats(self, tmp_path):
        epex, cal = _synthetic_epex_and_calendar(n_years=1, sigma=20.0)
        unc1 = Uncertainty().fit(epex, cal)
        path = tmp_path / "unc.parquet"
        unc1.save(path)
        unc2 = Uncertainty.load(path)
        assert set(unc1.boot_stats_.keys()) == set(unc2.boot_stats_.keys())
        for key in unc1.boot_stats_:
            s1 = unc1.boot_stats_[key]
            s2 = unc2.boot_stats_[key]
            assert s1["q10_res"] == pytest.approx(s2["q10_res"], abs=1e-12)
            assert s1["q90_res"] == pytest.approx(s2["q90_res"], abs=1e-12)

    def test_load_rejects_v1_schema(self, tmp_path):
        """Old v1 parquet (p10/p50/p90 ratio columns) raises explicit ValueError."""
        path = tmp_path / "v1.parquet"
        pd.DataFrame(
            [
                {
                    "saison": "hiver",
                    "type_jour": "semaine",
                    "heure": 0,
                    "quart": 1,
                    "p10": 0.85,
                    "p50": 1.0,
                    "p90": 1.15,
                }
            ]
        ).to_parquet(path, index=False)
        with pytest.raises(ValueError, match="v2 schema"):
            Uncertainty.load(path)


class TestUncertaintyNoHorizonWidening:
    """v2 ignores reference_date — bandwidth is horizon-invariant."""

    def test_bandwidth_invariant_across_horizons(self):
        """Same cell → same p90-p10 width regardless of reference_date."""
        epex, cal = _synthetic_epex_and_calendar(n_years=1, sigma=25.0)
        unc = Uncertainty().fit(epex, cal)
        base = 50.0 + 20.0 * np.sin(
            2.0 * np.pi * cal["heure_hce"].values / 24.0
        )
        pfc = pd.DataFrame({"price_shape": base}, index=epex.index)

        ic_near = unc.compute(pfc, cal, reference_date=epex.index.max())
        ic_far = unc.compute(
            pfc, cal, reference_date=pd.Timestamp("2020-01-01", tz="UTC")
        )
        # No horizon widening ⇒ bandwidths identical
        bw_near = (ic_near["p90"] - ic_near["p10"]).mean()
        bw_far = (ic_far["p90"] - ic_far["p10"]).mean()
        assert bw_near == pytest.approx(bw_far, abs=1e-9), (
            f"v2 must not apply horizon widening: bw_near={bw_near:.4f}, "
            f"bw_far={bw_far:.4f}"
        )
