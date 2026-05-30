"""Perfect-Foresight Shaping diagnostic for the Swiss HPFC (Phase 10, additive).

This module isolates **shaping quality** (does the cascader distribute a *given*
price level correctly across months, days and hours?) from **forward-forecast
error** (did the market's forward curve match what actually happened?). It is a
strictly additive, non-gating diagnostic: it never mutates `build_one`, the
cascader, the assembler, or the four reproducibility-guarded scorecard pillars.

Methodology (grounded in the HPFC literature)
---------------------------------------------
The HPFC literature universally decomposes a forward curve into a **level**
(the arbitrage-free constraint that the hourly curve averages back to traded
futures over each delivery block) and a **shape** (the hourly x daily x weekly x
seasonal profile). See Fleten & Lemming (2003, *Energy Economics*), Benth,
Koekebakker & Ollmar (2007, *J. Derivatives*), Kiesel, Paraschiv & Saetherø
(2019, *Comput. Manag. Sci.*) and Caldana, Fusai & Roncoroni (2017, *EJOR*).
A shaping diagnostic must therefore measure the second component *conditional on*
the first being satisfied — i.e. it must remove the level/forecast error by
construction. We do this two complementary ways:

1. **Build-side perfect-foresight anchoring** (seasonal shape).
   Re-anchor `build_one` on the *realized* settlement of a delivery year (Cal,
   optionally Cal+Quarter) computed ex-post from EPEX, while keeping the real
   cascader trained only on data available before the vintage. The resulting
   monthly signature is compared to the realized monthly signature. Because the
   level is now perfect, any remaining monthly-shape error is pure seasonal
   profiling error. We deliberately anchor at the *coarse* (Cal / Cal+Quarter)
   granularity: anchoring at month level would make the monthly-signature test
   circular (MSFC forces monthly means to equal the anchors).

2. **Score-side de-levelling** (intra-period shape).
   For the realized delivery window, normalise *both* the model curve and the
   realized prices to the same block mean, then score the normalised diurnal /
   weekly profiles. This is the operational "settlement anchoring" used by
   practitioners (KYOS, Montel); it is invariant to the level forecast and
   measures intra-period shape only. We report two complementary axes
   (Lago et al. 2021 metric discipline): **cosine similarity** (pattern fidelity:
   right peak hour, right weekend dip) and **demeaned RMSE** (amplitude fidelity:
   right peak-trough spread, right solar-bowl depth).

Swiss-physical sub-KPIs (Bevilacqua et al. 2022; CH hydro-dominated mix) make the
shape error interpretable: winter/summer ratio (hydro-storage + heating signal),
midday solar-bowl depth (DE coupling, regime-drifting), peak/off-peak spread
(hydro flexibility flattening).

Reproducibility
---------------
All outputs are emitted under their own keys / files and never touch the
`pillar1/pillar2_df/pillar3_df/pillar4_df` objects guarded by
`tests/test_phase10_reproducibility.py` (atol=1e-12). Computation is
deterministic (no unseeded RNG).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pfc_shaping.validation.scorecard import (
    ABLATION_GRID,
    build_one,
    list_vintages_2024_2025,
)

# Europe/Zurich is the canonical delivery-block timezone across the codebase
# (`count_hours`, `_resolve_base`, `_check_energy_consistency` all bucket here).
TZ = "Europe/Zurich"

#: EEX peak convention (matches assembler `_is_peak_timestamp` / cascader
#: `count_hours`): 08:00-20:00 local, Mon-Fri, excluding CH *national* holidays.
#: NB: a known inconsistency exists in `cascading.fit_peak_spreads` (it omits the
#: holiday mask); we follow the canonical assembler convention here. Flagged in
#: 10-PERFECT-FORESIGHT-SHAPING.md §6 for the codebase owner.
PEAK_HOUR_START = 8
PEAK_HOUR_END = 20

#: Meteorological seasons (WMO convention, matches `calendar_ch.py` Hiver/Ete).
WINTER_MONTHS = (12, 1, 2)
SUMMER_MONTHS = (6, 7, 8)
#: Symmetric 5-hour window centred on solar noon (~13:30 CEST in central Europe);
#: captures the DE-coupled summer solar-bowl trough. The 11-15 window is the
#: midpoint of {10-14, 11-15, 10-16} CH/DE solar-profile conventions in the
#: literature (e.g. Bevilacqua et al. 2022, *Energy Economics*). Robust to ±1h.
SOLAR_BOWL_HOURS = (11, 12, 13, 14, 15)

PRODUCTION_CONFIG = "bowl_on_floors_off"  # Config 4 production target


# ---------------------------------------------------------------------------
# Realized settlements (perfect-foresight anchors)
# ---------------------------------------------------------------------------
def _to_local(series: pd.Series) -> pd.Series:
    if series.index.tz is None:
        raise ValueError("epex series must be tz-aware (UTC expected).")
    return series.tz_convert(TZ)


def _peak_mask(idx_local: pd.DatetimeIndex, country: str = "CH") -> np.ndarray:
    """EEX peak mask on a local-tz index: 08-20 Mon-Fri excl. CH national holidays."""
    import holidays as _holidays

    years = range(int(idx_local.year.min()), int(idx_local.year.max()) + 1)
    cal = _holidays.country_holidays(country, years=list(years))
    is_weekday = idx_local.weekday < 5
    is_peak_hour = (idx_local.hour >= PEAK_HOUR_START) & (idx_local.hour < PEAK_HOUR_END)
    is_holiday = np.isin(idx_local.normalize().date, list(cal.keys()))
    return np.asarray(is_weekday & is_peak_hour & ~is_holiday)


def realized_window_mean(
    epex: pd.Series,
    year: int,
    quarter: int | None = None,
    month: int | None = None,
    segment: str = "base",
) -> float | None:
    """Realized mean EPEX price over a delivery window, or None if not fully covered.

    The window is bucketed in Europe/Zurich (EEX delivery convention). Returns
    None when the data does not span the *entire* window, so partial settlements
    never leak into the diagnostic.

    Parameters
    ----------
    epex
        Hourly EPEX series, tz-aware UTC, column already extracted.
    year, quarter, month
        Delivery window. Supply ``year`` alone for a Cal settlement, ``year`` +
        ``quarter`` for a quarter, ``year`` + ``month`` for a month.
    segment
        ``"base"`` (all hours), ``"peak"`` or ``"offpeak"`` (EEX CH mask).
    """
    loc = _to_local(epex)
    data_end = loc.index.max()
    data_start = loc.index.min()

    if month is not None:
        start = pd.Timestamp(year=year, month=month, day=1, tz=TZ)
        end = start + pd.offsets.MonthBegin(1)
    elif quarter is not None:
        first_month = 3 * (quarter - 1) + 1
        start = pd.Timestamp(year=year, month=first_month, day=1, tz=TZ)
        end = start + pd.offsets.MonthBegin(3)
    else:
        start = pd.Timestamp(year=year, month=1, day=1, tz=TZ)
        end = pd.Timestamp(year=year + 1, month=1, day=1, tz=TZ)

    # Coverage guard: boundary check + internal-gap check. A window passes only
    # if (a) it lies within the data span AND (b) every expected hour is observed
    # and non-NaN. This rejects silent corruption from internal gaps (a missing
    # month inside the window would otherwise pass and bias the settlement).
    if start < data_start or (end - pd.Timedelta(hours=1)) > data_end:
        return None
    win = loc[(loc.index >= start) & (loc.index < end)]
    if win.empty:
        return None
    expected_hours = int((end - start) / pd.Timedelta(hours=1))
    actual_hours = int(win.notna().sum())
    # DST tolerance: a CH calendar year contains 8760 hours (one -1h and one +1h
    # cancel); months containing a DST transition have 23h or 25h days, already
    # in the hourly index. We allow a ±2h slack to absorb leap-seconds or other
    # rare quirks but reject larger gaps as data corruption.
    if actual_hours < expected_hours - 2:
        return None

    if segment == "base":
        return float(win.mean())
    mask = _peak_mask(win.index)
    sel = win[mask] if segment == "peak" else win[~mask]
    if sel.empty:
        return None
    val = float(sel.mean())
    return val if np.isfinite(val) else None


def perfect_foresight_anchors(
    vintage_keys: list[str],
    epex: pd.Series,
    granularity: str = "cal",
) -> dict[str, float]:
    """Build perfect-foresight anchors matching a vintage's tradeable key set.

    Keeps only Cal (``granularity="cal"``) or Cal+Quarter
    (``granularity="cal_quarter"``) keys from ``vintage_keys`` and replaces each
    value with the realized settlement of its delivery window. Keys whose window
    is not fully realized are dropped. Month keys are *never* anchored (see module
    docstring: monthly anchoring makes the monthly-signature test circular).
    """
    if granularity not in {"cal", "cal_quarter"}:
        raise ValueError(f"granularity={granularity!r} must be 'cal' or 'cal_quarter'")
    anchors: dict[str, float] = {}
    for key in vintage_keys:
        if key.isdigit():  # 'YYYY'
            val = realized_window_mean(epex, int(key))
        elif granularity == "cal_quarter" and "-Q" in key:  # 'YYYY-QN'
            y, q = key.split("-Q")
            val = realized_window_mean(epex, int(y), quarter=int(q))
        else:
            continue
        if val is not None:
            anchors[key] = val
    return anchors


# ---------------------------------------------------------------------------
# Shape scoring
# ---------------------------------------------------------------------------
def build_curve(config_name: str, vintage: pd.Timestamp, epex: pd.Series,
                anchors: dict[str, float]) -> pd.Series:
    """Build a single PFC and return the hourly `price_shape` series.

    Thin wrapper around `build_one` for the perfect-foresight diagnostic. The
    cascader is still trained on `epex[index < vintage]` (no leak); only the
    forwards anchors carry the perfect-foresight information.
    """
    try:
        config = next(c for c in ABLATION_GRID if c.name == config_name)
    except StopIteration as exc:
        raise ValueError(
            f"config_name={config_name!r} not in ABLATION_GRID "
            f"(expected one of {[c.name for c in ABLATION_GRID]})"
        ) from exc
    df = build_one(
        config=config,
        vintage=vintage,
        epex_hist=epex.to_frame("price_eur_mwh"),
        forwards_asof=anchors,
        with_uncertainty=False,
    )
    return df["price_shape"].resample("1h").mean()


# Backwards-compat alias for the historical private name; prefer `build_curve`.
_build_curve = build_curve


def monthly_signature(series_1h: pd.Series, year: int) -> pd.Series:
    """Month-of-year mean signature for a delivery year (local tz)."""
    loc = _to_local(series_1h)
    sub = loc[loc.index.year == year]
    return sub.groupby(sub.index.month).mean()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else float("nan")


def deleveled_diurnal_scores(
    model_1h: pd.Series,
    realized_1h: pd.Series,
    year: int | None = None,
    months: tuple[int, ...] | None = None,
) -> dict[str, float]:
    """Score the intra-day shape after de-levelling both curves to the same mean.

    Returns cosine similarity (pattern fidelity) and demeaned RMSE (amplitude
    fidelity) of the average hour-of-day profile.

    Parameters
    ----------
    model_1h, realized_1h
        Hourly tz-aware series. Both are bucketed in Europe/Zurich and
        intersected on index.
    year
        Optional delivery-year filter. **You should set this** to keep the score
        confined to the realized/anchored window — otherwise the intersection
        spans every overlapping hour, which can blend an anchored delivery year
        with extrapolated hours from `_resolve_base`'s year-offset fallback.
    months
        Optional season filter (e.g. ``SUMMER_MONTHS``).

    Returns
    -------
    dict
        ``cosine`` (additive- AND multiplicative-invariant on the profile),
        ``demeaned_rmse`` (additive-invariant only — responds to multiplicative
        rescale by design: amplitude fidelity), ``n_hours`` (sample size).
    """
    m = _to_local(model_1h)
    r = _to_local(realized_1h)
    common = m.index.intersection(r.index)
    m, r = m.loc[common], r.loc[common]
    if year is not None:
        sel = (m.index.year == year)
        m, r = m[sel], r[sel]
    if months is not None:
        sel = np.isin(m.index.month, months)
        m, r = m[sel], r[sel]
    if len(m) < 24:
        return {"cosine": float("nan"), "demeaned_rmse": float("nan"), "n_hours": int(len(m))}
    prof_m = m.groupby(m.index.hour).mean()
    prof_r = r.groupby(r.index.hour).mean()
    hours = prof_m.index.intersection(prof_r.index)
    dm = prof_m.loc[hours].values - prof_m.loc[hours].values.mean()
    dr = prof_r.loc[hours].values - prof_r.loc[hours].values.mean()
    rmse = float(np.sqrt(np.mean((dm - dr) ** 2)))
    return {"cosine": _cosine(dm, dr), "demeaned_rmse": rmse, "n_hours": int(len(m))}


def ch_physical_subkpis(model_1h: pd.Series, realized_1h: pd.Series, year: int) -> dict[str, Any]:
    """Interpretable Swiss-physical shape sub-KPIs (model vs realized) for a year."""
    m = _to_local(model_1h)
    r = _to_local(realized_1h)
    m = m[m.index.year == year]
    r = r[r.index.year == year]
    common = m.index.intersection(r.index)
    m, r = m.loc[common], r.loc[common]
    if m.empty:
        return {"degenerate": True}

    def winter_summer(s: pd.Series) -> float:
        w = s[np.isin(s.index.month, WINTER_MONTHS)].mean()
        su = s[np.isin(s.index.month, SUMMER_MONTHS)].mean()
        return float(w / su) if su else float("nan")

    def bowl_depth(s: pd.Series) -> float:
        summer = s[np.isin(s.index.month, SUMMER_MONTHS)]
        if summer.empty:
            return float("nan")
        midday = summer[np.isin(summer.index.hour, SOLAR_BOWL_HOURS)].mean()
        return float(1.0 - midday / summer.mean()) if summer.mean() else float("nan")

    def peak_offpeak(s: pd.Series) -> float:
        mask = _peak_mask(s.index)
        return float(s[mask].mean() - s[~mask].mean())

    return {
        "winter_summer_ratio": {"model": winter_summer(m), "realized": winter_summer(r)},
        "solar_bowl_depth": {"model": bowl_depth(m), "realized": bowl_depth(r)},
        "peak_offpeak_spread": {"model": peak_offpeak(m), "realized": peak_offpeak(r)},
    }


# ---------------------------------------------------------------------------
# Robust aggregation (addresses the min-order-statistic critique)
# ---------------------------------------------------------------------------
def robust_aggregate(values: list[float]) -> dict[str, float]:
    """Distributional summary (median, p10, p90, min, max, n). Robust to outliers."""
    arr = np.array([v for v in values if v == v], dtype=float)
    if arr.size == 0:
        return {"n": 0, "median": float("nan"), "p10": float("nan"),
                "p90": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclass
class PerfectForesightResult:
    """Container for the additive perfect-foresight diagnostic."""

    target_year: int
    config_name: str
    sweep: pd.DataFrame  # per-vintage seasonal-shape decomposition
    diurnal: pd.DataFrame  # per-vintage de-levelled intra-day shape scores
    subkpis: dict[str, Any]  # CH-physical sub-KPIs at best-trained vintage
    aggregates: dict[str, Any] = field(default_factory=dict)


def _train_years(epex: pd.Series, vintage: pd.Timestamp) -> float:
    sub = epex[epex.index < vintage]
    if sub.empty:
        return 0.0
    return round((sub.index.max() - sub.index.min()).days / 365.25, 2)


def run_perfect_foresight(
    epex: pd.Series,
    forwards_history: pd.DataFrame,
    target_year: int = 2025,
    config_name: str = PRODUCTION_CONFIG,
    vintages: list[pd.Timestamp] | None = None,
) -> PerfectForesightResult:
    """Run the full perfect-foresight shaping diagnostic for one delivery year.

    For every vintage strictly before ``target_year`` delivery, the model is
    re-anchored on the realized Cal settlement of ``target_year`` (perfect level
    foresight) and its monthly seasonal shape is compared to realized. The market
    baseline (real per-vintage forwards) is computed for the same months so the
    error decomposes into *forecast* (market->PF gap) and *shaping* (1 - PF corr).
    """
    from scipy.stats import pearsonr, spearmanr

    from pfc_shaping.validation.scorecard import _forwards_for_vintage

    if vintages is None:
        vintages = [v for v in list_vintages_2024_2025() if v.year < target_year]

    realized_cal = realized_window_mean(epex, target_year)
    if realized_cal is None:
        raise ValueError(
            f"Cal {target_year} is not fully realized in the EPEX data; "
            f"cannot run perfect-foresight for that delivery year."
        )
    realized_1h = epex  # already hourly
    real_sig = monthly_signature(realized_1h, target_year)

    def _safe_corr(a: pd.Series, b: pd.Series) -> tuple[float, float, int]:
        """Pearson + Spearman after dropping NaN-paired entries (degrade gracefully)."""
        both = pd.concat([a, b], axis=1, join="inner").dropna()
        if len(both) < 3:
            return float("nan"), float("nan"), int(len(both))
        x, y = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
        return float(pearsonr(x, y)[0]), float(spearmanr(x, y)[0]), int(len(both))

    sweep_rows: list[dict] = []
    diurnal_rows: list[dict] = []
    for v in vintages:
        # Perfect-foresight build (anchor realized Cal target_year only).
        pf_curve = build_curve(config_name, v, epex, {str(target_year): realized_cal})
        pf_sig = monthly_signature(pf_curve, target_year)
        pf_r, pf_rho, n_pairs = _safe_corr(pf_sig, real_sig)
        if n_pairs < 3:
            continue

        # Market baseline build (real per-vintage forwards) over the same months.
        mkt_fwds = _forwards_for_vintage(forwards_history, v, epex)
        market_r = float("nan")
        if mkt_fwds:
            mkt_curve = build_curve(config_name, v, epex, mkt_fwds)
            mkt_sig = monthly_signature(mkt_curve, target_year)
            market_r, _, _ = _safe_corr(mkt_sig, real_sig)

        sweep_rows.append({
            "vintage": v.strftime("%Y-%m-%d"),
            "train_years": _train_years(epex, v),
            "market_corr": round(market_r, 4),
            "pf_cal_corr": round(pf_r, 4),
            "pf_cal_spearman": round(pf_rho, 4),
            "forecast_gap": round(pf_r - market_r, 4) if market_r == market_r else float("nan"),
            "shaping_residual": round(1.0 - pf_r, 4),
            "n_months": n_pairs,
        })
        # CRITICAL: filter to the target year so the diurnal score scopes to the
        # realized/anchored delivery window — otherwise the intersection spans
        # 2025+2026Q1 (extrapolated via `_resolve_base` year-offset fallback) and
        # pollutes the score with un-anchored hours (~17% in the 2024-12 vintage).
        d = deleveled_diurnal_scores(pf_curve, realized_1h, year=target_year)
        d_summer = deleveled_diurnal_scores(
            pf_curve, realized_1h, year=target_year, months=SUMMER_MONTHS,
        )
        diurnal_rows.append({
            "vintage": v.strftime("%Y-%m-%d"),
            "train_years": _train_years(epex, v),
            "diurnal_cosine": round(d["cosine"], 4),
            "diurnal_demeaned_rmse": round(d["demeaned_rmse"], 3),
            "summer_cosine": round(d_summer["cosine"], 4),
            "summer_demeaned_rmse": round(d_summer["demeaned_rmse"], 3),
        })

    sweep = pd.DataFrame(sweep_rows)
    diurnal = pd.DataFrame(diurnal_rows)

    # CH-physical sub-KPIs at the best-trained (last) vintage.
    subkpis: dict[str, Any] = {}
    if vintages:
        best = max(vintages, key=lambda v: _train_years(epex, v))
        pf_best = build_curve(config_name, best, epex, {str(target_year): realized_cal})
        subkpis = {"vintage": best.strftime("%Y-%m-%d"),
                   **ch_physical_subkpis(pf_best, realized_1h, target_year)}

    # Effective regime count: `fit_seasonal_ratios` keeps only *full* calendar
    # years, so many adjacent vintages collapse to identical seasonal_ratios →
    # the per-vintage sweep over-states the diagnostic's effective sample size.
    # Surfaced here so callers can correctly interpret `pf_cal_corr` aggregates.
    n_distinct_pf_corr = int(sweep["pf_cal_corr"].round(4).nunique()) if not sweep.empty else 0

    aggregates = {
        "pf_cal_corr": robust_aggregate(sweep["pf_cal_corr"].tolist()) if not sweep.empty else {},
        "market_corr": robust_aggregate(sweep["market_corr"].tolist()) if not sweep.empty else {},
        "diurnal_cosine": robust_aggregate(diurnal["diurnal_cosine"].tolist()) if not diurnal.empty else {},
        "n_distinct_pf_corr_regimes": n_distinct_pf_corr,
    }
    return PerfectForesightResult(
        target_year=target_year, config_name=config_name, sweep=sweep,
        diurnal=diurnal, subkpis=subkpis, aggregates=aggregates,
    )


__all__ = [
    "PEAK_HOUR_END",
    "PEAK_HOUR_START",
    "PRODUCTION_CONFIG",
    "PerfectForesightResult",
    "SOLAR_BOWL_HOURS",
    "SUMMER_MONTHS",
    "TZ",
    "WINTER_MONTHS",
    "build_curve",
    "ch_physical_subkpis",
    "deleveled_diurnal_scores",
    "monthly_signature",
    "perfect_foresight_anchors",
    "realized_window_mean",
    "robust_aggregate",
    "run_perfect_foresight",
]
