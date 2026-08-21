"""
block_distribution.py — Phase 5ter (shape risk premium per client block)

Goal
----
Given an assembled PFC and a client block (e.g. weekday 10-15, overnight 18-9),
return the P10/P50/P90 distribution of the **block-average price** so a trader
can quote a shape risk premium on profile deals. The shape risk is the part of
the block average that is *not* hedgeable with standard base/peak forwards
(the PFC level is anchored arbitrage-free on EEX; the intra-block shape is not).

Method (comonotonic aggregation of the calibrated per-step IC80)
----------------------------------------------------------------
Phase 10 / Uncertainty v2 already calibrates, per cell
(saison, type_jour, heure, quart), additive residual quantiles q10_res/q90_res
(€/MWh). For a block B over a delivery window:

    P50(B) = mean_{t in B}  price_shape(t)
    P10(B) = mean_{t in B} (price_shape(t) + q10_res[cell(t)])
    P90(B) = mean_{t in B} (price_shape(t) + q90_res[cell(t)])

This is the **comonotonic** (perfectly-correlated) aggregation: the quantile of
the block average equals the average of the per-step quantiles. It is *exact*
under comonotonicity and is the conservative bound for an inhedgeable risk.

Why not independent-step Monte-Carlo? Sampling each 15-min step independently
would diversify the residuals at rate 1/sqrt(N) and collapse the band to a few
€/MWh — which understates a shape risk that is, by construction, correlated
within a delivery period (a high-price month lifts every hour together). The
comonotonic bound avoids assuming a diversification the trader cannot realise.
A diversified / empirical block-bootstrap variant (resampling whole historical
delivery periods to recover the true within-block correlation) is the natural
Phase 5ter follow-up; it is intentionally *not* implemented here.

This mirrors uncertainty.py's stated philosophy: empirical quantiles are exact
estimators (no bootstrap noise), and calibration-by-construction is preferred
over parametric bandwidth assumptions (Hildmann 2013, KYOS HPFC).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pfc_shaping.lt.model.uncertainty import Uncertainty
from pfc_shaping.validation.block_masks import BlockMask


@dataclass(frozen=True)
class BlockDistribution:
    """P10/P50/P90 of a client block average price (€/MWh)."""

    block: str
    n_steps: int
    p10: float
    p50: float
    p90: float

    @property
    def band_width(self) -> float:
        """P90 - P10 (€/MWh): the shape risk premium width."""
        return self.p90 - self.p10


def pfc_block_distribution(
    pfc_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    uncertainty: Uncertainty,
    block: BlockMask,
    *,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
) -> BlockDistribution:
    """Distribution of the average price over a client block.

    Args
    ----
    pfc_df       : assembled PFC, column ``price_shape``, tz-aware UTC index.
    calendar_df  : enrichment aligned on ``pfc_df.index`` with columns
                   ['saison', 'type_jour', 'heure_hce', 'quart'].
    uncertainty  : a *fitted* ``Uncertainty`` (v2) instance — the calibrated
                   per-cell residual quantiles are reused as the noise source.
    block        : a ``BlockMask`` (see ``pfc_shaping.validation.block_masks``).
    start, end   : optional delivery window restricting the block. Interpreted
                   in the index timezone (UTC); naive timestamps are localized
                   to UTC. ``start`` inclusive, ``end`` exclusive.

    Returns
    -------
    BlockDistribution with p10 <= p50 <= p90 (the per-cell quantiles satisfy
    q10_res <= 0 <= q90_res by construction, so ordering always holds).

    Raises
    ------
    ValueError
        If no timestamp falls in the block within [start, end), or if required
        columns are missing.
    """
    if "price_shape" not in pfc_df.columns:
        raise ValueError("pfc_df must have a 'price_shape' column.")
    if pfc_df.index.tz is None:
        raise ValueError("pfc_df index must be tz-aware (UTC expected).")

    mask = block.apply(pfc_df.index)

    if start is not None:
        start_ts = pd.Timestamp(start)
        if start_ts.tz is None:
            start_ts = start_ts.tz_localize("UTC")
        mask &= (pfc_df.index >= start_ts)
    if end is not None:
        end_ts = pd.Timestamp(end)
        if end_ts.tz is None:
            end_ts = end_ts.tz_localize("UTC")
        mask &= (pfc_df.index < end_ts)

    if not mask.any():
        raise ValueError(
            f"Block '{block.name}' has no timestamps in the requested window "
            f"[{start}, {end})."
        )

    # Per-step calibrated bands (additive residual quantiles around price_shape).
    ic = uncertainty.compute(pfc_df, calendar_df)

    p50 = float(pfc_df["price_shape"].to_numpy()[mask].mean())
    p10 = float(ic["p10"].to_numpy()[mask].mean())
    p90 = float(ic["p90"].to_numpy()[mask].mean())

    return BlockDistribution(
        block=block.name,
        n_steps=int(mask.sum()),
        p10=p10,
        p50=p50,
        p90=p90,
    )


__all__ = ["BlockDistribution", "pfc_block_distribution"]
