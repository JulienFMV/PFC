"""
scorecard.py
------------
Skeleton du harness PFC FMV Quality Scorecard (Phase 10, 5-pillar SOTA replication).

Plan 10-01 ne livre que le skeleton :
    - dataclass `AblationConfig` + grille `ABLATION_GRID` (4 configs D-A6-1)
    - helpers vintages calendar (`list_vintages_2024_2025`, `last_business_day_of_month`)
    - `build_one(config, vintage, epex_hist, forwards_asof, with_uncertainty)`
      qui appelle `PFCAssembler.build(reference_date=vintage)` sans leakage
    - skeleton `derive_forwards_from_epex_hist` (Q2 fallback Mac Mini, body
      implémenté Plan 10-01 Task 3 sub-step 3c)
    - constantes module-level `FORWARDS_SOURCE_REAL`/`FORWARDS_SOURCE_FALLBACK_DIAGNOSTIC`
      (C3 REVIEWS : marker structuré propagé au scorecard parquet)
    - stub `run_scorecard_pillar_1` qui raise NotImplementedError (Plan 10-02 wires)

Les pillar maths (Hildmann, MAE/RMSE/MZ, Christoffersen, DM, Peer-review) sont
implémentés dans les Plans 10-02/03/04. Ce module expose les abstractions
consommables par ces plans.

Convention
----------
- Vintages : dernier jour ouvré (BMonthEnd(0)) de chaque mois 2024-01..2025-12,
  pinné à 18:00 local Europe/Zurich (post-market close), tz_convert vers UTC.
- Ablation grid 2×2 : (bowl OFF/ON) × (floors OFF/ON) = 4 configs, dont
  Config 4 (`bowl_on_floors_off`) = production target (SC#1 Hildmann gate).
- Forwards-as-of-vintage : 2 paths possibles documentés via la métadonnée
  `forwards_source` (C3 REVIEWS) propagée dans le scorecard parquet :
    * `real_eex_xlsx` (snapshot historique H:\\, FMV poste uniquement)
    * `fallback_diagnostic` (`derive_forwards_from_epex_hist` proxy non-leaky
      sur EPEX historique, Mac Mini default)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.tseries.offsets import BMonthEnd


# ---------------------------------------------------------------------------
# C3 REVIEWS — forwards_source structured flag (not just a log line)
# ---------------------------------------------------------------------------

#: Marker pour les runs qui consomment les forwards EEX réels (snapshot XLSX H:\).
#: Toute cellule Pillar 1/4 produite sous ce marker est gate-eligible pour SC#1.
FORWARDS_SOURCE_REAL: str = "real_eex_xlsx"

#: Marker pour les runs qui consomment le proxy `derive_forwards_from_epex_hist`
#: (fallback Mac Mini, EPEX hist same-period mean). Diagnostic only — toute
#: cellule produite sous ce marker est annotée "Diagnostic only — not gate-eligible"
#: par Plan 10-04 ; SC#1 ne peut PAS être satisfait par un run fallback.
FORWARDS_SOURCE_FALLBACK_DIAGNOSTIC: str = "fallback_diagnostic"


# ---------------------------------------------------------------------------
# Ablation grid (D-A6-1) — 4 configs explorées par le scorecard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationConfig:
    """Une config de l'ablation grid 2×2 (bowl OFF/ON × floors OFF/ON).

    Attributes
    ----------
    name
        Identifiant lisible (`bowl_off_floors_on`, `bowl_on_floors_off`, etc.).
    use_seasonal_hourly
        Bowl flag : passé à `ShapeHourly(use_seasonal_hourly=...)`. Aussi
        propagé via `PFC_LT_USE_SEASONAL_HOURLY_SHAPE` au besoin (mais
        l'arg explicit du ctor a priorité — freeze-at-init, cf. 5bis-A).
    enforce_positivity, enforce_m_factor_floor, enforce_floor, allow_negative_peak
        4 ctor args PFCAssembler (B1/B4 Approach B, cf. Phase 5 D-A2-3).
        Mode legacy : `enforce_*=True, allow_negative_peak=False`.
        Mode negative-ready : `enforce_*=False, allow_negative_peak=True`.
    """

    name: str
    use_seasonal_hourly: bool
    enforce_positivity: bool
    enforce_m_factor_floor: bool
    enforce_floor: bool
    allow_negative_peak: bool


#: Les 4 configs explorées par le scorecard (D-A6-1 CONTEXT Phase 10).
#:
#: - Config 1 (`bowl_off_floors_on`) : legacy pre-5bis-A.
#: - Config 2 (`bowl_on_floors_on`)  : 5bis-B livré sans Phase 5.
#: - Config 3 (`bowl_off_floors_off`) : Phase 5 livré sans 5bis-B.
#: - Config 4 (`bowl_on_floors_off`)  : **production target** (SC#1 Hildmann gate).
ABLATION_GRID: list[AblationConfig] = [
    AblationConfig(
        name="bowl_off_floors_on",
        use_seasonal_hourly=False,
        enforce_positivity=True,
        enforce_m_factor_floor=True,
        enforce_floor=True,
        allow_negative_peak=False,
    ),
    AblationConfig(
        name="bowl_on_floors_on",
        use_seasonal_hourly=True,
        enforce_positivity=True,
        enforce_m_factor_floor=True,
        enforce_floor=True,
        allow_negative_peak=False,
    ),
    AblationConfig(
        name="bowl_off_floors_off",
        use_seasonal_hourly=False,
        enforce_positivity=False,
        enforce_m_factor_floor=False,
        enforce_floor=False,
        allow_negative_peak=True,
    ),
    AblationConfig(
        name="bowl_on_floors_off",
        use_seasonal_hourly=True,
        enforce_positivity=False,
        enforce_m_factor_floor=False,
        enforce_floor=False,
        allow_negative_peak=True,
    ),
]


# ---------------------------------------------------------------------------
# Vintages calendar — last business day of each month 2024-01..2025-12
# ---------------------------------------------------------------------------


def last_business_day_of_month(year: int, month: int) -> pd.Timestamp:
    """Last business day (Mon-Fri, BMonthEnd(0)) of (year, month) à 18:00 local Europe/Zurich.

    BMonthEnd(0) n'honore PAS les jours fériés CH/VS — convention acceptée per
    RESEARCH §Pattern 5 / Pitfall 4 (commit acceptance A3). Pour 2024-2025 le
    rare clash férié est documenté ; aucune incidence sur les KPIs scorecard.

    Returns
    -------
    pd.Timestamp
        tz-aware Europe/Zurich (utiliser `.tz_convert("UTC")` pour l'index canonique).

    Examples
    --------
    >>> last_business_day_of_month(2024, 3)  # 29 mars 2024 (vendredi, Karfreitag)
    Timestamp('2024-03-29 18:00:00+0100', tz='Europe/Zurich')
    """
    first = pd.Timestamp(f"{year}-{month:02d}-01", tz="Europe/Zurich")
    last_bd_local = first + BMonthEnd(0)
    # Pin à 18:00 local Europe/Zurich (post-market close convention)
    last_bd_local = last_bd_local.replace(hour=18, minute=0, second=0, microsecond=0)
    return last_bd_local


def list_vintages_2024_2025() -> list[pd.Timestamp]:
    """24 vintages : last business day de chaque mois 2024-01..2025-12, tz UTC.

    Chaque vintage est pinné à 18:00 local Europe/Zurich puis converti UTC
    pour l'index canonique du modèle PFC LT (convention tz-strict, cf. CONTEXT).
    """
    out: list[pd.Timestamp] = []
    for year in (2024, 2025):
        for month in range(1, 13):
            last_bd_local = last_business_day_of_month(year, month)
            out.append(last_bd_local.tz_convert("UTC"))
    return out


# ---------------------------------------------------------------------------
# build_one — 1 PFC build (1 cellule du 96-grid)
# ---------------------------------------------------------------------------


def build_one(
    config: AblationConfig,
    vintage: pd.Timestamp,
    epex_hist: pd.DataFrame,
    forwards_asof: dict[str, float],
    with_uncertainty: bool = False,
) -> pd.DataFrame:
    """Build one PFC for (config, vintage). 1 of 96 grid points.

    Architecture
    ------------
    - Fit `ShapeHourly`, `ShapeIntraday` (et optionnellement `Uncertainty`)
      sur `epex_hist.loc[epex_hist.index < vintage]` (strict `<`, no leakage
      per RESEARCH Pitfall 2).
    - Instancie `PFCAssembler` avec les 4 ctor args floor de `config` +
      `ShapeHourly(use_seasonal_hourly=config.use_seasonal_hourly)`.
    - Appelle `assembler.build(base_prices=forwards_asof, reference_date=vintage,
      start_date=vintage.strftime("%Y-%m-%d"), horizon_days=3*365)`.

    Le module imports sont locaux (à l'intérieur de la fonction) pour éviter
    le coût d'import à l'import-time du package validation (Plan 10-01 garde
    le skeleton léger ; les Plans 10-02/03/04 wireront le contenu).
    """
    # Local imports — évite le coût à l'import time du package validation
    from pfc_shaping.data.calendar_ch import build_calendar
    from pfc_shaping.lt.model.assembler import PFCAssembler
    from pfc_shaping.lt.model.shape_hourly import ShapeHourly
    from pfc_shaping.lt.model.shape_intraday import ShapeIntraday
    from pfc_shaping.lt.model.uncertainty import Uncertainty

    # No leakage : strict `<` filter sur l'index historique vs vintage
    train_mask = epex_hist.index < vintage
    epex_train = epex_hist.loc[train_mask]

    # Calendar CH pour ShapeHourly/Intraday fit (build sur la fenêtre train)
    cal_start = epex_train.index.min().strftime("%Y-%m-%d")
    cal_end = epex_train.index.max().strftime("%Y-%m-%d")
    cal_train = build_calendar(cal_start, cal_end, country="CH")

    sh = ShapeHourly(use_seasonal_hourly=config.use_seasonal_hourly).fit(
        epex_train, cal_train
    )
    si = ShapeIntraday().fit(epex_train, None, cal_train)
    unc: Uncertainty | None = None
    if with_uncertainty:
        unc = Uncertainty(n_boot=500, seed=42).fit(epex_train, cal_train)

    assembler = PFCAssembler(
        shape_hourly=sh,
        shape_intraday=si,
        uncertainty=unc,
        enforce_positivity=config.enforce_positivity,
        enforce_m_factor_floor=config.enforce_m_factor_floor,
        enforce_floor=config.enforce_floor,
        allow_negative_peak=config.allow_negative_peak,
    )
    pfc = assembler.build(
        base_prices=forwards_asof,
        start_date=vintage.strftime("%Y-%m-%d"),
        horizon_days=3 * 365,  # Y+3 pour couvrir M+1..Y+2 horizons
        reference_date=vintage,
    )
    return pfc


# ---------------------------------------------------------------------------
# derive_forwards_from_epex_hist — Q2 fallback skeleton (body Plan 10-01 Task 3 sub-3c)
# ---------------------------------------------------------------------------


def derive_forwards_from_epex_hist(
    epex_hist: pd.Series,
    vintage: pd.Timestamp,
    horizon_days: int = 3 * 365,
) -> dict[str, float]:
    """Skeleton — body implémenté Plan 10-01 Task 3 sub-step 3c.

    Q2 fallback Mac Mini : quand le snapshot forwards XLSX H:\\ n'est pas
    accessible, on dérive des "would-be-forwards" proxy depuis l'EPEX
    historique en utilisant les fenêtres futures perçues = mean(EPEX_hist[
    same_period_last_N_years]) pour chaque Cal/Q/M key sur l'horizon `[vintage,
    vintage + horizon_days]`.

    Returns
    -------
    dict
        Mapping `{'YYYY': cal_proxy, 'YYYY-QN': quarter_proxy, 'YYYY-MM': month_proxy}`
        avec convention parser keys identique à `assembler.build(base_prices=...)`.

    Notes
    -----
    Cette fonction porte le marker `forwards_source = "fallback_diagnostic"`
    (cf. `FORWARDS_SOURCE_FALLBACK_DIAGNOSTIC`) propagé au scorecard parquet.
    Plan 10-04 SC#1 ne peut PAS être satisfait par un run fallback (gate
    eligible uniquement avec `forwards_source = "real_eex_xlsx"`).
    """
    raise NotImplementedError(
        "derive_forwards_from_epex_hist body implémenté Plan 10-01 Task 3 sub-step 3c"
    )


# ---------------------------------------------------------------------------
# run_scorecard_pillar_1 — Plan 10-02 wires the actual Hildmann tests
# ---------------------------------------------------------------------------


def run_scorecard_pillar_1(
    config_name: str,
    cache_dir: Any,
    epex_source: str = "mock",
) -> dict:
    """Stub — Pillar 1 (Hildmann SC#1 gate) wirée Plan 10-02.

    Cette signature est exposée Plan 10-01 pour permettre aux tests et au
    code consommateur d'importer la fonction. Le body (4 tests structurels
    arb-free / holiday-weekend / seasonal-corr / continuity) est implémenté
    dans Plan 10-02 (`structural_tests.py` + intégration ici).
    """
    raise NotImplementedError("Pillar 1 wiring deferred to Plan 10-02")


__all__ = [
    "AblationConfig",
    "ABLATION_GRID",
    "list_vintages_2024_2025",
    "last_business_day_of_month",
    "build_one",
    "derive_forwards_from_epex_hist",
    "FORWARDS_SOURCE_REAL",
    "FORWARDS_SOURCE_FALLBACK_DIAGNOSTIC",
    "run_scorecard_pillar_1",
]
