"""
Phase 5ter — pfc_block_distribution (shape risk premium per client block).

Reference: pfc_shaping/lt/model/block_distribution.py

Contract under test:
    - Returns P10 <= P50 <= P90 of the block-average price.
    - Comonotonic aggregation == mean of the per-step Uncertainty v2 bands
      restricted to the block (the exact property the module relies on).
    - start/end restricts the delivery window; empty block raises.
    - The shape risk band is non-trivial on a summer-solar block.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.model.block_distribution import (
    BlockDistribution,
    pfc_block_distribution,
)
from pfc_shaping.lt.model.uncertainty import Uncertainty
from pfc_shaping.validation.block_masks import (
    BlockMiddayWeekday,
    BlockSummerSolarBowl,
)

_SAISON_BY_MONTH = {
    12: "Hiver", 1: "Hiver", 2: "Hiver",
    3: "Printemps", 4: "Printemps", 5: "Printemps",
    6: "Ete", 7: "Ete", 8: "Ete",
    9: "Automne", 10: "Automne", 11: "Automne",
}


def _calendar(idx_utc: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar enrichment derived from Europe/Zurich local time."""
    local = idx_utc.tz_convert("Europe/Zurich")
    saison = np.array([_SAISON_BY_MONTH[m] for m in local.month])
    type_jour = np.where(
        local.weekday < 5,
        "Ouvrable",
        np.where(local.weekday == 5, "Samedi", "Dimanche"),
    )
    return pd.DataFrame(
        {
            "saison": saison,
            "type_jour": type_jour,
            "heure_hce": local.hour.astype(int),
            "quart": (local.minute // 15 + 1).astype(int),
        },
        index=idx_utc,
    )


def _synthetic_epex(seed: int = 42, sigma: float = 30.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two years of hourly EPEX history with a known hourly shape + noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=24 * 365 * 2, freq="h", tz="UTC")
    cal = _calendar(idx)
    base = 60.0 + 25.0 * np.sin(2.0 * np.pi * cal["heure_hce"].to_numpy() / 24.0)
    prices = base + rng.normal(0.0, sigma, size=len(idx))
    return pd.DataFrame({"price_eur_mwh": prices}, index=idx), cal


def _synthetic_pfc() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A 15-min PFC over 2027 with the same deterministic hourly shape."""
    idx = pd.date_range("2027-01-01", "2027-12-31 23:45", freq="15min", tz="UTC")
    cal = _calendar(idx)
    price = 60.0 + 25.0 * np.sin(2.0 * np.pi * cal["heure_hce"].to_numpy() / 24.0)
    return pd.DataFrame({"price_shape": price}, index=idx), cal


@pytest.fixture(scope="module")
def fitted_uncertainty() -> Uncertainty:
    epex, epex_cal = _synthetic_epex()
    return Uncertainty().fit(epex, epex_cal)


@pytest.fixture(scope="module")
def pfc_and_calendar() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _synthetic_pfc()


class TestBlockDistributionContract:
    def test_quantiles_ordered(self, fitted_uncertainty, pfc_and_calendar):
        pfc, cal = pfc_and_calendar
        dist = pfc_block_distribution(
            pfc, cal, fitted_uncertainty, BlockMiddayWeekday()
        )
        assert isinstance(dist, BlockDistribution)
        assert dist.p10 <= dist.p50 <= dist.p90
        assert dist.n_steps > 0

    def test_deterministic(self, fitted_uncertainty, pfc_and_calendar):
        """No Monte-Carlo noise: repeated calls are bit-identical."""
        pfc, cal = pfc_and_calendar
        d1 = pfc_block_distribution(pfc, cal, fitted_uncertainty, BlockMiddayWeekday())
        d2 = pfc_block_distribution(pfc, cal, fitted_uncertainty, BlockMiddayWeekday())
        assert d1 == d2

    def test_comonotonic_equals_mean_of_per_step_bands(
        self, fitted_uncertainty, pfc_and_calendar
    ):
        """Block band == mean of the per-step Uncertainty v2 bands over the block."""
        pfc, cal = pfc_and_calendar
        block = BlockMiddayWeekday()
        dist = pfc_block_distribution(pfc, cal, fitted_uncertainty, block)

        ic = fitted_uncertainty.compute(pfc, cal)
        mask = block.apply(pfc.index)
        assert dist.p50 == pytest.approx(pfc["price_shape"].to_numpy()[mask].mean())
        assert dist.p10 == pytest.approx(ic["p10"].to_numpy()[mask].mean())
        assert dist.p90 == pytest.approx(ic["p90"].to_numpy()[mask].mean())

    def test_band_width_nontrivial_summer_solar(
        self, fitted_uncertainty, pfc_and_calendar
    ):
        """Inhedgeable shape band is a meaningful risk premium (> 5 €/MWh)."""
        pfc, cal = pfc_and_calendar
        dist = pfc_block_distribution(
            pfc, cal, fitted_uncertainty, BlockSummerSolarBowl()
        )
        assert dist.band_width > 5.0

    def test_start_end_window_restricts_steps(
        self, fitted_uncertainty, pfc_and_calendar
    ):
        pfc, cal = pfc_and_calendar
        block = BlockMiddayWeekday()
        full = pfc_block_distribution(pfc, cal, fitted_uncertainty, block)
        q1 = pfc_block_distribution(
            pfc, cal, fitted_uncertainty, block,
            start="2027-01-01", end="2027-04-01",
        )
        assert 0 < q1.n_steps < full.n_steps

    def test_empty_block_raises(self, fitted_uncertainty, pfc_and_calendar):
        pfc, cal = pfc_and_calendar
        with pytest.raises(ValueError, match="no timestamps"):
            pfc_block_distribution(
                pfc, cal, fitted_uncertainty, BlockSummerSolarBowl(),
                start="2027-01-01", end="2027-02-01",  # winter → no summer-solar steps
            )

    def test_requires_price_shape_column(self, fitted_uncertainty, pfc_and_calendar):
        pfc, cal = pfc_and_calendar
        bad = pfc.rename(columns={"price_shape": "price"})
        with pytest.raises(ValueError, match="price_shape"):
            pfc_block_distribution(bad, cal, fitted_uncertainty, BlockMiddayWeekday())
