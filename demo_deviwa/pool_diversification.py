"""Pool diversification analytics for the Deviwa demo.

State-of-the-art utility-pool risk metrics, computed per delivery year :

  - **Individual actor VaR / ES** on the open price exposure
    (residual MWh × current Cal-Y forward × σ_price × √horizon).
  - **Consolidated pool VaR / ES** using the *net signed* exposure
    (= the natural offset between long actors who are under-hedged and
    short actors who are over-hedged for that year). This is the standard
    parametric approach for utility ERM ; the offset benefit emerges from
    sub-additivity of |·| over signed sums :
        Σ |e_i| ≥ |Σ e_i|     (equality only when all signs align).
  - **Diversification benefit** = Σ individual − pool, in EUR and %.

The ``correlation_matrix`` field on :class:`PoolRiskMetrics` is computed
from the historical daily net-load returns and is provided for the
educational heatmap on the dashboard. It does NOT drive the VaR
calculation here ; in the price-only single-factor model, all actors
share the same forward as the random factor, so the diversification
benefit comes purely from the signed offset of exposures, not from
load correlations. Future iterations can add a two-factor (price +
volume) model where the load correlation matters explicitly.

References :
  - JP Morgan / RiskMetrics : parametric VaR / ES with normal returns
  - EFET risk management guidelines, sub-additivity property
  - Volue / KYOS portfolio dashboards : "Diversifikations-Vorteil" headline
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ``demo_deviwa`` is a flat script directory rather than a proper Python
# package (no __init__.py — historical layout for the Streamlit pages).
# To stay import-safe in all execution contexts (Streamlit runtime, pytest,
# `python -c "import demo_deviwa.pool_diversification"`) we ensure the
# sibling-script directory is on sys.path before resolving the local imports.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from portfolio_yearly import (  # noqa: E402  (path injection above)
    VOLATILITY_ANNUAL_PCT,
    compute_hedge_by_year,
    get_forward_year_proxy,
)


# ---------------------------------------------------------------------------
# Statistical constants
# ---------------------------------------------------------------------------
# All hard-coded for transparency (no scipy import in this module — keeps
# the demo runtime light and avoids a soft dependency).

# z-scores for 1-tailed VaR confidence levels.
# Computed via scipy.stats.norm.ppf(α), rounded to 4 decimals.
Z_BY_CONF: dict[float, float] = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.975: 1.9600,
    0.99: 2.3263,
    0.999: 3.0902,  # extends the table so very-high CL doesn't silently clip to 0.99
}

# ES factor = φ(z) / (1 − α), where φ is the standard normal PDF.
# Computed once; valid only under the normal-return assumption.
def _es_factor(z: float, alpha: float) -> float:
    """Expected Shortfall factor for a normal distribution.

    ES = factor × σ, where factor = φ(z) / (1 − α), φ standard normal PDF.
    """
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return pdf / max(1.0 - alpha, 1e-12)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActorRiskMetrics:
    """Risk numbers for a single actor on a single delivery year.

    Attributes:
        actor: actor name (e.g. "RELL").
        delivery_year: e.g. 2027.
        open_mwh: signed open exposure (positive = under-hedged ;
            negative = over-hedged).
        forward_price_eur_mwh: best available year forward (Cal or proxy).
        exposure_eur: signed = open_mwh × forward_price.
        sigma_eur: scenario standard deviation in EUR (always ≥ 0).
        var_eur: parametric VaR at the chosen confidence level (always ≥ 0).
        es_eur: Expected Shortfall (a.k.a. CVaR), always ≥ var_eur.
    """
    actor: str
    delivery_year: int
    open_mwh: float
    forward_price_eur_mwh: float
    exposure_eur: float
    sigma_eur: float
    var_eur: float
    es_eur: float


@dataclass(frozen=True)
class PoolRiskMetrics:
    """Pool-level risk metrics + diversification breakdown.

    Attributes:
        delivery_year: e.g. 2027.
        confidence_level: e.g. 0.95.
        horizon_years: years from today to delivery midpoint (≥ 0.25 floor).
        sigma_assumed_pct: annualized vol assumption (typically 20%).
        actors: list of per-actor :class:`ActorRiskMetrics`.
        net_exposure_eur: signed sum of actor exposures (= the pool's
            residual price exposure after offset).
        pool_sigma_eur: |net_exposure| × σ × √T.
        sum_individual_var_eur: Σ_i ActorRiskMetrics.var_eur.
        pool_var_eur: z × pool_sigma_eur.
        diversification_benefit_eur: sum_individual − pool ≥ 0.
        diversification_pct: 100 × benefit / sum_individual (NaN if no
            individual risk).
        sum_individual_es_eur, pool_es_eur: same idea for Expected Shortfall.
        correlation_matrix: historical daily-return correlation of net loads
            across the actors, for the educational heatmap (does not drive
            the VaR computation in this single-factor model).
    """
    delivery_year: int
    confidence_level: float
    horizon_years: float
    sigma_assumed_pct: float
    actors: list[ActorRiskMetrics]
    net_exposure_eur: float
    pool_sigma_eur: float
    sum_individual_var_eur: float
    pool_var_eur: float
    diversification_benefit_eur: float
    diversification_pct: float
    sum_individual_es_eur: float
    pool_es_eur: float
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _z_score(confidence_level: float) -> float:
    """Return the z-score for a 1-tailed VaR confidence level.

    Linearly interpolates between :data:`Z_BY_CONF` entries (0.90, 0.95,
    0.975, 0.99, 0.999) for unsupported levels. Outside the table range
    the function clips to the table extremes ; under-/over-shooting these
    bounds would not be meaningful for a VaR anyway.
    """
    cl = float(confidence_level)
    if cl in Z_BY_CONF:
        return Z_BY_CONF[cl]
    keys = sorted(Z_BY_CONF.keys())
    if cl <= keys[0]:
        return Z_BY_CONF[keys[0]]
    if cl >= keys[-1]:
        return Z_BY_CONF[keys[-1]]
    # Linear interpolation between the two surrounding table entries
    for k1, k2 in zip(keys, keys[1:]):
        if k1 <= cl <= k2:
            w = (cl - k1) / (k2 - k1)
            return (1 - w) * Z_BY_CONF[k1] + w * Z_BY_CONF[k2]
    return Z_BY_CONF[0.95]  # unreachable fallback


def _years_to_delivery(year: int, today: pd.Timestamp) -> float:
    """Years from ``today`` to 1 July of delivery ``year``.

    Conventions :
      - Past delivery years (``year < today.year``) → 0.0 ; the year is
        fully settled, no remaining price uncertainty.
      - Current delivery year (``year == today.year``) → floored at 0.25,
        because there is still settlement-period uncertainty until the
        last day even when the mid-year anchor is in the past.
      - Future delivery years → exact (delivery_mid − today) in years.
    """
    if year < today.year:
        return 0.0
    delivery_mid = pd.Timestamp(year=year, month=7, day=1, tz=today.tz)
    seconds = (delivery_mid - today).total_seconds()
    years = seconds / (365.25 * 86400)
    return max(years, 0.25)


# ---------------------------------------------------------------------------
# Per-actor risk
# ---------------------------------------------------------------------------

def compute_actor_risk(
    actor: str,
    delivery_year: int,
    deals: pd.DataFrame,
    programme: pd.DataFrame,
    forwards: pd.DataFrame,
    today: pd.Timestamp | None = None,
    confidence_level: float = 0.95,
    vol_annual_pct: float = VOLATILITY_ANNUAL_PCT,
    *,
    extrapolate_missing_programme: bool = True,
) -> ActorRiskMetrics | None:
    """Compute VaR / ES for one actor on one delivery year.

    Returns ``None`` if the actor has no programme nor deals for that year
    (= no exposure to measure).
    """
    if today is None:
        today = pd.Timestamp.now(tz="Europe/Zurich")

    metrics_by_year = compute_hedge_by_year(
        deals, programme, actor=actor, today=today,
        extrapolate_missing_programme=extrapolate_missing_programme,
    )
    yh = metrics_by_year.get(delivery_year)
    if yh is None:
        return None

    forward_price = get_forward_year_proxy(forwards, delivery_year)
    if not np.isfinite(forward_price):
        # No forward → cannot value the exposure
        return ActorRiskMetrics(
            actor=actor,
            delivery_year=delivery_year,
            open_mwh=yh.open_mwh,
            forward_price_eur_mwh=float("nan"),
            exposure_eur=float("nan"),
            sigma_eur=float("nan"),
            var_eur=float("nan"),
            es_eur=float("nan"),
        )

    horizon = _years_to_delivery(delivery_year, today)
    z = _z_score(confidence_level)
    es_factor = _es_factor(z, confidence_level)

    exposure_eur = yh.open_mwh * forward_price
    sigma_eur = abs(exposure_eur) * (vol_annual_pct / 100.0) * math.sqrt(horizon)
    var_eur = z * sigma_eur
    es_eur = es_factor * sigma_eur

    return ActorRiskMetrics(
        actor=actor,
        delivery_year=delivery_year,
        open_mwh=yh.open_mwh,
        forward_price_eur_mwh=forward_price,
        exposure_eur=exposure_eur,
        sigma_eur=sigma_eur,
        var_eur=var_eur,
        es_eur=es_eur,
    )


# ---------------------------------------------------------------------------
# Load correlation matrix (educational viz)
# ---------------------------------------------------------------------------

def compute_load_correlation_matrix(
    programme: pd.DataFrame,
    freq: str = "D",
    actors: list[str] | None = None,
) -> pd.DataFrame:
    """Historical correlation of net-load returns between actors.

    Args:
        programme: long-format frame with ``actor, timestamp, programme_mw``.
        freq: pandas resample alias for the return frequency. Defaults to
            "D" (daily) — gives ~1460 obs over 4 years per actor, robust
            estimates without the autocorrelation noise of hourly data.
        actors: restrict to these actors (in order). ``None`` uses all
            actors present in the frame.

    Returns:
        Square DataFrame indexed and columned by actor name. Values in
        [-1, 1], NaN if an actor has < 30 valid return observations.
        Empty DataFrame if no programme data.
    """
    if programme is None or programme.empty:
        return pd.DataFrame()
    if "actor" not in programme.columns or "timestamp" not in programme.columns:
        return pd.DataFrame()

    if actors is None:
        actors = sorted(programme["actor"].dropna().unique().tolist())

    # Build a wide frame: index = resampled timestamp, columns = actors
    pivot = (
        programme.pivot_table(
            index="timestamp", columns="actor", values="programme_mw", aggfunc="mean"
        )
        .reindex(columns=actors)
    )
    # Aggregate to the requested frequency
    resampled = pivot.resample(freq).mean()
    # Compute returns (relative changes). fill_method=None disables the
    # deprecated default forward-fill (which would silently turn missing
    # daily values into 0% returns and bias correlations toward 0).
    returns = resampled.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="all")
    if returns.empty:
        return pd.DataFrame(index=actors, columns=actors, dtype=float)

    # Pearson correlation. min_periods filters out actors with too few obs.
    corr = returns.corr(method="pearson", min_periods=30)
    corr = corr.reindex(index=actors, columns=actors)

    # An actor with a constant programme (zero variance) has NaN correlation
    # with itself under Pearson — but for the dashboard heatmap we want the
    # diagonal to be 1.0 whenever the actor has enough non-null observations
    # (the off-diagonal entries against a flat series are genuinely undefined
    # and stay NaN, which is mathematically correct).
    for actor in corr.index:
        if actor not in returns.columns:
            continue
        n_obs = int(returns[actor].notna().sum())
        if n_obs >= 30:
            corr.loc[actor, actor] = 1.0

    return corr


# ---------------------------------------------------------------------------
# Pool-level diversification
# ---------------------------------------------------------------------------

def compute_pool_diversification(
    delivery_year: int,
    deals: pd.DataFrame,
    programme: pd.DataFrame,
    forwards: pd.DataFrame,
    actors: list[str] | None = None,
    today: pd.Timestamp | None = None,
    confidence_level: float = 0.95,
    vol_annual_pct: float = VOLATILITY_ANNUAL_PCT,
    *,
    extrapolate_missing_programme: bool = True,
) -> PoolRiskMetrics:
    """Compute the diversification breakdown for one delivery year.

    Args:
        delivery_year: e.g. 2027.
        deals, programme, forwards: cached parquet frames.
        actors: list of actor names to include in the pool. ``None`` = all
            actors present in either deals or programme.
        today: reference date for time-to-delivery and VaR horizon.
        confidence_level: 0.90 / 0.95 / 0.975 / 0.99 supported.
        vol_annual_pct: annualized vol assumption (default 20%).
    """
    if today is None:
        today = pd.Timestamp.now(tz="Europe/Zurich")

    # Determine the actor universe
    universe: set[str] = set()
    if deals is not None and not deals.empty and "actor" in deals.columns:
        universe.update(deals["actor"].dropna().astype(str).unique())
    if programme is not None and not programme.empty and "actor" in programme.columns:
        universe.update(programme["actor"].dropna().astype(str).unique())
    if actors is None:
        actors_list = sorted(universe)
    else:
        actors_list = list(actors)

    # Per-actor risk
    actor_metrics: list[ActorRiskMetrics] = []
    for a in actors_list:
        m = compute_actor_risk(
            a,
            delivery_year,
            deals,
            programme,
            forwards,
            today=today,
            confidence_level=confidence_level,
            vol_annual_pct=vol_annual_pct,
            extrapolate_missing_programme=extrapolate_missing_programme,
        )
        if m is not None:
            actor_metrics.append(m)

    # Aggregate
    horizon = _years_to_delivery(delivery_year, today)
    z = _z_score(confidence_level)
    es_factor = _es_factor(z, confidence_level)

    # An actor without a finite exposure (= no forward available for that
    # year) cannot be priced — we filter it out of the pool numerator and
    # denominator. If NO actor is priceable at all, we propagate NaN to
    # the pool risk fields rather than reporting "0 risk" — these are
    # semantically different states and the dashboard must distinguish them.
    valid = [m for m in actor_metrics if np.isfinite(m.exposure_eur)]
    pricing_available = len(valid) > 0

    if not pricing_available:
        net_exposure_eur = float("nan")
        pool_sigma_eur = float("nan")
        pool_var_eur = float("nan")
        pool_es_eur = float("nan")
        sum_var = float("nan")
        sum_es = float("nan")
        benefit_eur = float("nan")
        benefit_pct = float("nan")
    else:
        net_exposure_eur = sum(m.exposure_eur for m in valid)
        pool_sigma_eur = abs(net_exposure_eur) * (vol_annual_pct / 100.0) * math.sqrt(horizon)
        pool_var_eur = z * pool_sigma_eur
        pool_es_eur = es_factor * pool_sigma_eur

        sum_var = sum(m.var_eur for m in valid if np.isfinite(m.var_eur))
        sum_es = sum(m.es_eur for m in valid if np.isfinite(m.es_eur))

        benefit_eur = sum_var - pool_var_eur
        # Clamp tiny floating-point residuals so the dashboard doesn't
        # surface things like "−1.45e−11 EUR (0.0 %)".
        if abs(benefit_eur) < 1e-6:
            benefit_eur = 0.0
        benefit_pct = 100.0 * benefit_eur / sum_var if sum_var > 0 else 0.0
        if abs(benefit_pct) < 1e-9:
            benefit_pct = 0.0

    # Educational correlation matrix (does not drive VaR)
    corr = compute_load_correlation_matrix(programme, freq="D", actors=actors_list)

    return PoolRiskMetrics(
        delivery_year=delivery_year,
        confidence_level=confidence_level,
        horizon_years=horizon,
        sigma_assumed_pct=vol_annual_pct,
        actors=actor_metrics,
        net_exposure_eur=net_exposure_eur,
        pool_sigma_eur=pool_sigma_eur,
        sum_individual_var_eur=sum_var,
        pool_var_eur=pool_var_eur,
        diversification_benefit_eur=benefit_eur,
        diversification_pct=benefit_pct,
        sum_individual_es_eur=sum_es,
        pool_es_eur=pool_es_eur,
        correlation_matrix=corr,
    )
