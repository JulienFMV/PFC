"""
solar_modulation.py
-------------------
Solar-aware intra-day shape correction (Phase 10 — §4quater).

Motivation
==========
The SOTA shaping stack (regime-aware seasonal_ratios + hydro-aware peak_spreads
+ intra-day half-life=90d) clears the 0.85 SC#1 gate on all 12 Cal-2025
vintages, but a residual peak/off-peak amplitude gap (~14 EUR/MWh: model 20 vs
realized 6.4) is *invariant to the half-life* and was traced to a 2025
solar-regime shift that a past-fit ``f_H`` cannot anticipate. Empirically,
monthly CH solar penetration ``solar_pen_m = sum(solar_mw) / sum(load_mw)``
correlates with the realized midday-bowl depth at Pearson r = 0.914
(p ~ 5e-16, n = 39 months).

This module adds an exogenous, leak-free, *post-processing* correction to the
hourly shape factor ``f_H`` produced by :class:`ShapeHourly`, following
Method (A) of the research spec (``solar_research/method_research.json``):

    f_H_adj[t] = f_H[t] * (1 + beta[saison, type_jour, block(h)] *
                               (solar_pen_m(t) - solar_pen_baseline))

then re-normalised per local calendar day so that ``mean_h f_H_adj = 1`` is
preserved (the ``f_W`` / level layers downstream are untouched). The correction
is a small set of block-pooled per-hour coefficients (4 hour-blocks), fitted by
ridge regression against the de-levelled monthly midday residual — chosen over
GAM / residual-demand-stack / two-regime alternatives because n_months ~ 39
only supports a handful of free parameters (Harrell 2015 DoF rule).

Design contract (research §6)
=============================
* Pure post-processing layer between ``ShapeHourly.apply()`` and the assembler.
  It does NOT touch ``ShapeHourly`` internals, so the Phase-10 atol=1e-12
  reproducibility contract holds when the layer is disabled.
* Leakage is *intrinsic* to this module: every aggregation (feature extraction
  and beta estimation) is filtered to ``timestamp < vintage`` regardless of what
  the caller passes in. Forward delivery months (>= vintage) are projected from
  a climatology + linear-trend extrapolation of the training history — never
  from realized future data (research §3).

Citations: Karakatsani & Bunn (2008), Cludius et al. (2014), Wagner (2014),
Kiesel & Paraschiv (2017), Bevilacqua et al. (2022), Marcjasz et al. (2023),
Harrell (2015).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.lt.model.shape_hourly import SAISONS, TYPES_JOUR

logger = logging.getLogger(__name__)

__all__ = [
    "SolarPenetrationFeature",
    "SolarBlockedFHCorrection",
    "solar_modulate",
    "HOUR_BLOCKS",
    "BLOCK_NAMES",
    "DEFAULT_ENTSO_PATH",
    "DEFAULT_EPEX_PATH",
    "DEFAULT_RIDGE_LAMBDA",
]

# Repo root: pfc_shaping/lt/model/solar_modulation.py -> parents[3] == repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENTSO_PATH = _REPO_ROOT / "pfc_shaping" / "data" / "entso_15min.parquet"
DEFAULT_EPEX_PATH = _REPO_ROOT / "data" / "epex_hourly.parquet"

# 4 hour-blocks (research §4 block_structure). hour-of-day (local) -> block name.
#   NIGHT        : 00-06, 22-23   -> beta ~ 0 (no sun)
#   MORNING_RAMP : 07-09          -> beta slightly negative
#   MIDDAY_BOWL  : 10-15          -> beta < 0  (THE key coefficient: bowl deepens)
#   EVENING_PEAK : 16-21          -> beta 0..slightly positive (re-normalisation lift)
BLOCK_NAMES = ["NIGHT", "MORNING_RAMP", "MIDDAY_BOWL", "EVENING_PEAK"]
HOUR_BLOCKS: dict[int, str] = {}
for _h in range(24):
    if _h in (7, 8, 9):
        HOUR_BLOCKS[_h] = "MORNING_RAMP"
    elif _h in (10, 11, 12, 13, 14, 15):
        HOUR_BLOCKS[_h] = "MIDDAY_BOWL"
    elif _h in (16, 17, 18, 19, 20, 21):
        HOUR_BLOCKS[_h] = "EVENING_PEAK"
    else:  # 0..6, 22, 23
        HOUR_BLOCKS[_h] = "NIGHT"

# Ridge penalty on the (already mean-centred) single-slope regression. Small,
# stabilising; CV-tunable per research §4 (leave-one-year-out). The default is a
# modest absolute floor added to the design Gram scalar, which mainly damps
# blocks/cells with little penetration spread (e.g. NIGHT) towards beta = 0.
DEFAULT_RIDGE_LAMBDA = 1e-3

# Weekend / holiday day-types collapse to a single "Weekend" response group for
# beta pooling (research §2 temporal_treatment: the sun does not know the
# weekday, but the *response* differs between workday and weekend daytime load).
_WEEKEND_TYPES = {"Samedi", "Dimanche", "Ferie_CH", "Ferie_DE"}


def _response_group(type_jour: str) -> str:
    """Map a calendar ``type_jour`` to a beta-pooling response group."""
    return "Weekend" if type_jour in _WEEKEND_TYPES else "Ouvrable"


# ---------------------------------------------------------------------------
# Feature: leak-free monthly CH solar penetration
# ---------------------------------------------------------------------------
class SolarPenetrationFeature:
    """Leak-free monthly CH solar-penetration extractor.

    ``solar_pen_m = sum_{t in month} solar_mw_t / sum_{t in month} load_mw_t``,
    computed from ``entso_15min.parquet`` at 15-min resolution then aggregated
    to wall-clock months (DST-agnostic at monthly resolution, research §2).

    All public accessors take a ``vintage`` and use ONLY rows with
    ``index < vintage``. Forward delivery months (>= vintage) are projected via
    a same-calendar-month climatology plus a linear trend fitted on the training
    history, capped at the 99th percentile of the training distribution to avoid
    extrapolating beyond the fitted support (research §2 monotonicity_guard).
    """

    def __init__(self, entso_path: str | Path = DEFAULT_ENTSO_PATH) -> None:
        self.entso_path = Path(entso_path)
        self._raw: pd.DataFrame | None = None  # cached [solar_mw, load_mw]

    # -- internal -----------------------------------------------------------
    def _load(self) -> pd.DataFrame:
        if self._raw is None:
            # NOTE: pyarrow drops the (nameless) datetime index when read_parquet
            # is given an explicit ``columns=`` list — it returns a default
            # RangeIndex, which would silently break the ``index < vintage`` leak
            # filter. Read the frame whole, then subset, so the DatetimeIndex is
            # preserved. The result is cached, so the one-off cost is paid once.
            df = pd.read_parquet(self.entso_path)
            missing = {"solar_mw", "load_mw"} - set(df.columns)
            if missing:
                raise KeyError(f"{self.entso_path} missing columns {sorted(missing)}")
            df = df[["solar_mw", "load_mw"]]
            if not isinstance(df.index, pd.DatetimeIndex):
                raise TypeError(
                    f"{self.entso_path} index is {type(df.index).__name__}, "
                    "expected a DatetimeIndex (cannot apply the leak-free vintage filter)"
                )
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            self._raw = df
        return self._raw

    @staticmethod
    def _to_vintage_ts(vintage) -> pd.Timestamp:
        ts = pd.Timestamp(vintage)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts

    # -- public -------------------------------------------------------------
    def monthly_realized(self, vintage) -> pd.Series:
        """Realized ``solar_pen_m`` per month, using ONLY data ``index < vintage``.

        Returns a Series indexed by ``pandas.Period[M]`` (UTC wall-clock month).
        A trailing partial month (whose data is truncated by the vintage cut) is
        dropped so the ratio is never biased by a half-observed month.
        """
        v = self._to_vintage_ts(vintage)
        df = self._load()
        df = df[df.index < v]
        if df.empty:
            return pd.Series(dtype=float)
        # Wall-clock UTC months (DST-agnostic at monthly resolution, research §2).
        # tz_localize(None) keeps the UTC wall time and drops the tz so .to_period
        # does not emit the "dropping timezone information" UserWarning.
        per = df.index.tz_localize(None).to_period("M")
        num = df["solar_mw"].groupby(per).sum()
        den = df["load_mw"].groupby(per).sum()
        pen = (num / den).dropna()
        # Drop the trailing partial month (its last observed ts is < v but the
        # month itself extends to/after v) to avoid a truncation-biased ratio.
        if len(pen) > 0:
            last_period = df.index.max().tz_localize(None).to_period("M")
            if last_period.end_time >= v.tz_localize(None):
                pen = pen[pen.index < last_period]
        # Audit hook (research §3): the maximum input timestamp must be < vintage.
        self._last_max_input_ts_ = df.index.max()
        return pen.astype(float)

    def baseline(self, vintage) -> float:
        """Training-mean ``solar_pen`` (the centre of the linear response)."""
        pen = self.monthly_realized(vintage)
        return float(pen.mean()) if len(pen) else 0.0

    def _trend(self, pen: pd.Series) -> tuple[float, float]:
        """Linear (slope_per_month, intercept) of ``solar_pen`` vs a month index."""
        if len(pen) < 3:
            return 0.0, float(pen.mean()) if len(pen) else 0.0
        x = np.arange(len(pen), dtype=float)
        y = pen.to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        return float(slope), float(intercept)

    def project(self, period, vintage, realized: pd.Series | None = None) -> float:
        """Projected ``solar_pen_m`` for a delivery ``period`` (Period[M]).

        * If the month is fully realized before ``vintage`` -> realized value.
        * Otherwise -> same-calendar-month climatology (mean over training years)
          plus the linear-trend drift from the last training month to the target,
          capped at the 99th percentile of the training distribution.
        """
        period = pd.Period(period, freq="M")
        if realized is None:
            realized = self.monthly_realized(vintage)
        if period in realized.index:
            return float(realized.loc[period])
        if len(realized) == 0:
            return 0.0

        cap = float(np.percentile(realized.to_numpy(), 99))
        # Climatology: mean over training years for the same calendar month.
        same_month = realized[realized.index.month == period.month]
        clim = float(same_month.mean()) if len(same_month) else float(realized.mean())
        # Same-month climatology carries the seasonal level; the linear trend
        # adds the slow secular drift from the last training month to the target.
        slope, _ = self._trend(realized)
        last_period = realized.index.max()
        months_ahead = (period.year - last_period.year) * 12 + (period.month - last_period.month)
        proj = clim + slope * months_ahead
        return float(min(max(proj, 0.0), cap))

    def lookup(self, periods, vintage) -> pd.Series:
        """Projected ``solar_pen_m`` for an iterable of delivery periods."""
        realized = self.monthly_realized(vintage)
        idx = pd.PeriodIndex([pd.Period(p, freq="M") for p in periods], freq="M")
        vals = [self.project(p, vintage, realized=realized) for p in idx]
        return pd.Series(vals, index=idx, dtype=float, name="solar_pen_proj")


# ---------------------------------------------------------------------------
# Correction: block-pooled per-hour beta, fitted by ridge
# ---------------------------------------------------------------------------
class SolarBlockedFHCorrection:
    """Block-pooled multiplicative ``f_H`` correction driven by solar penetration.

    Fits, per ``(saison, response_group, hour-block)``, a single ridge slope
    ``beta`` of the de-levelled monthly midday residual on the centred solar
    penetration::

        f_H_realized[s, j, h, m] / f_H_baseline[s, j, h] - 1
            = beta[s, group(j), block(h)] * (solar_pen_m - solar_pen_baseline) + eps

    ``f_H_baseline`` is the model's fitted shape (``ShapeHourly.get``); the
    residual is therefore the part of the realized intra-day shape the past-fit
    model leaves on the table, regressed on the exogenous solar signal.
    """

    def __init__(
        self,
        feature: SolarPenetrationFeature | None = None,
        ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
        min_months: int = 6,
        blocks: dict[int, str] | None = None,
    ) -> None:
        self.feature = feature or SolarPenetrationFeature()
        self.ridge_lambda = float(ridge_lambda)
        self.min_months = int(min_months)
        self.blocks = dict(blocks) if blocks is not None else dict(HOUR_BLOCKS)
        # Fitted state
        self.fitted_: bool = False
        self.beta_: dict[tuple[str, str, str], float] = {}
        self.baseline_pen_: float = 0.0
        self.vintage_: pd.Timestamp | None = None
        self.n_months_: int = 0
        self.diagnostics_: dict = {}

    # -- fit ----------------------------------------------------------------
    def fit(
        self,
        epex_df: pd.DataFrame,
        baseline_provider,
        vintage,
        calendar_df: pd.DataFrame | None = None,
    ) -> "SolarBlockedFHCorrection":
        """Estimate the block betas (leak-free).

        Parameters
        ----------
        epex_df
            Realized price history with a ``price_eur_mwh`` column and a
            tz-aware DatetimeIndex (hourly or 15-min). Filtered to
            ``index < vintage`` internally — leak protection is intrinsic.
        baseline_provider
            Callable ``(saison, type_jour) -> np.ndarray[24]`` returning the
            model baseline hourly shape (typically ``ShapeHourly.get``).
        vintage
            Publication cut-off. Only data strictly before it is used.
        calendar_df
            Optional pre-enriched calendar for ``epex_df.index``; recomputed via
            :func:`enrich_15min_index` when omitted.
        """
        v = SolarPenetrationFeature._to_vintage_ts(vintage)
        self.vintage_ = v
        self.baseline_pen_ = self.feature.baseline(v)
        pen_m = self.feature.monthly_realized(v)  # Series[Period[M]]

        df = epex_df[["price_eur_mwh"]].copy()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df[df.index < v]
        if df.empty or len(pen_m) < self.min_months:
            logger.warning(
                "SolarBlockedFHCorrection.fit: insufficient data (rows=%d, months=%d) "
                "— correction disabled (beta=0)", len(df), len(pen_m),
            )
            self.fitted_ = True
            self.diagnostics_ = {"reason": "insufficient_data", "n_months": len(pen_m)}
            return self

        if calendar_df is None:
            calendar_df = enrich_15min_index(df.index, country="CH")
        cal = calendar_df.reindex(df.index)
        df = df.join(cal[["saison", "type_jour", "heure_hce"]])
        df = df.dropna(subset=["saison", "type_jour", "heure_hce", "price_eur_mwh"])
        df["heure_hce"] = df["heure_hce"].astype(int)
        df["period"] = df.index.to_period("M")

        # Mean realized price per (saison, type_jour, period, hour).
        grp = (
            df.groupby(["saison", "type_jour", "period", "heure_hce"])["price_eur_mwh"]
            .mean()
            .rename("price")
            .reset_index()
        )

        # Accumulate ridge design pairs (x, y) per (saison, group, block).
        acc: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
        n_cells = 0
        for (saison, type_jour, period), sub in grp.groupby(["saison", "type_jour", "period"]):
            if period not in pen_m.index:
                continue
            hourly = sub.set_index("heure_hce")["price"].reindex(range(24))
            if hourly.notna().sum() < 20:  # need most of the day present
                continue
            hourly = hourly.interpolate(limit_direction="both")
            day_mean = hourly.mean()
            if not np.isfinite(day_mean) or day_mean == 0:
                continue
            realized_fh = hourly.to_numpy() / day_mean  # mean_h = 1
            try:
                base_fh = np.asarray(baseline_provider(saison, type_jour), dtype=float)
            except Exception:  # pragma: no cover - defensive
                continue
            if base_fh.shape[0] != 24:
                continue
            x = float(pen_m.loc[period]) - self.baseline_pen_
            group = _response_group(type_jour)
            n_cells += 1
            for h in range(24):
                b = base_fh[h]
                if not np.isfinite(b) or b == 0:
                    continue
                y = realized_fh[h] / b - 1.0
                if not np.isfinite(y):
                    continue
                key = (saison, group, self.blocks[h])
                acc.setdefault(key, []).append((x, y))

        # Ridge slope (no intercept; residual is centred at baseline_pen).
        beta: dict[tuple[str, str, str], float] = {}
        for key, pairs in acc.items():
            arr = np.asarray(pairs, dtype=float)
            xs, ys = arr[:, 0], arr[:, 1]
            denom = float(np.dot(xs, xs)) + self.ridge_lambda
            beta[key] = float(np.dot(xs, ys) / denom) if denom > 0 else 0.0

        self.beta_ = beta
        self.n_months_ = int(len(pen_m))
        self.fitted_ = True
        self.diagnostics_ = {
            "n_months": self.n_months_,
            "n_cell_months": n_cells,
            "baseline_pen": self.baseline_pen_,
            "n_beta": len(beta),
            "beta_midday_summer_ouvrable": beta.get(("Ete", "Ouvrable", "MIDDAY_BOWL")),
        }
        logger.info(
            "SolarBlockedFHCorrection fitted: vintage=%s, months=%d, baseline_pen=%.4f, "
            "beta_midday(Ete,Ouvrable)=%s",
            v.date(), self.n_months_, self.baseline_pen_,
            self.diagnostics_["beta_midday_summer_ouvrable"],
        )
        return self

    def beta_for(self, saison: str, type_jour: str, hour: int) -> float:
        """Return the fitted beta for a cell/hour (0.0 if unfitted/absent)."""
        key = (saison, _response_group(type_jour), self.blocks.get(int(hour), "NIGHT"))
        return float(self.beta_.get(key, 0.0))

    # -- apply --------------------------------------------------------------
    def modulate(
        self,
        f_H_series: pd.Series,
        calendar_df: pd.DataFrame,
        vintage=None,
        tz: str = "Europe/Zurich",
    ) -> pd.Series:
        """Apply the multiplicative solar correction and re-normalise to mean_h=1.

        ``f_H_adj[t] = f_H[t] * (1 + beta[s, group(j), block(h)] *
                                     (solar_pen_m(t) - baseline_pen))``
        then divided by the per-local-day mean so each (saison, type_jour) day
        keeps ``mean_h f_H_adj = 1`` (the level / f_W layers stay intact).
        """
        if not self.fitted_:
            raise RuntimeError("SolarBlockedFHCorrection.modulate called before fit()")
        v = self.vintage_ if vintage is None else SolarPenetrationFeature._to_vintage_ts(vintage)

        idx = f_H_series.index
        cal = calendar_df.reindex(idx)
        local = idx.tz_convert(tz)
        periods = pd.PeriodIndex(local, freq="M")

        # Project solar_pen once per distinct delivery month.
        uniq = pd.PeriodIndex(sorted(set(periods)), freq="M")
        pen_lookup = self.feature.lookup(uniq, v)
        pen_map = {p: pen_lookup.loc[p] for p in uniq}
        pen_t = np.array([pen_map[p] for p in periods], dtype=float)

        saison = cal["saison"].to_numpy()
        type_jour = cal["type_jour"].to_numpy()
        heure = cal["heure_hce"].to_numpy()

        # Vectorised beta lookup via a per-row map (few distinct keys).
        beta_t = np.zeros(len(idx), dtype=float)
        for i in range(len(idx)):
            s, tj, h = saison[i], type_jour[i], heure[i]
            if s is None or (isinstance(s, float) and np.isnan(s)) or pd.isna(h):
                continue
            beta_t[i] = self.beta_.get(
                (s, _response_group(tj), self.blocks.get(int(h), "NIGHT")), 0.0
            )

        factor = 1.0 + beta_t * (pen_t - self.baseline_pen_)
        # Guard against pathological negatives (keeps f_H sign/positivity sane).
        factor = np.clip(factor, 0.5, 1.5)
        adj = f_H_series.to_numpy(dtype=float) * factor

        result = pd.Series(adj, index=idx, name=f_H_series.name or "f_H")
        # Re-normalise per local calendar day so mean_h f_H = 1 is preserved.
        day_key = pd.Index(
            [f"{t.year}-{t.month:02d}-{t.day:02d}" for t in local], name="day"
        )
        day_mean = result.groupby(day_key).transform("mean").replace(0, 1.0)
        result = result / day_mean
        return result


# ---------------------------------------------------------------------------
# Convenience wrapper used by the assembler hook
# ---------------------------------------------------------------------------
def solar_modulate(
    f_H_series: pd.Series,
    calendar_df: pd.DataFrame,
    shape_hourly,
    vintage,
    *,
    epex: pd.DataFrame | None = None,
    feature: SolarPenetrationFeature | None = None,
    cache: bool = True,
) -> pd.Series:
    """Fit-and-apply helper: the post-``ShapeHourly.apply()`` solar layer.

    Resolves (and caches on ``shape_hourly``) a :class:`SolarBlockedFHCorrection`
    for the given ``vintage``, then returns the modulated ``f_H`` series. When
    ``vintage`` is ``None`` the input is returned unchanged (no leak-safe anchor
    available) so the layer degrades gracefully.

    The baseline shape is ``shape_hourly.get``; the realized price history is
    ``epex`` if supplied, else a cached frame stashed on ``shape_hourly``
    (``_solar_epex_hist``), else the default ``epex_hourly.parquet`` — always
    filtered to ``< vintage`` inside the correction.
    """
    if vintage is None:
        logger.warning("solar_modulate: vintage is None — returning f_H unchanged")
        return f_H_series

    v = SolarPenetrationFeature._to_vintage_ts(vintage)
    cache_attr = "_solar_correction_cache"
    if cache and shape_hourly is not None:
        store = getattr(shape_hourly, cache_attr, None)
        if store is None:
            store = {}
            try:
                setattr(shape_hourly, cache_attr, store)
            except Exception:  # pragma: no cover - exotic sh objects
                store = {}
        key = v.isoformat()
        corr = store.get(key)
        if corr is None:
            corr = _fit_correction(shape_hourly, v, epex=epex, feature=feature)
            store[key] = corr
    else:
        corr = _fit_correction(shape_hourly, v, epex=epex, feature=feature)

    return corr.modulate(f_H_series, calendar_df, vintage=v)


def _fit_correction(shape_hourly, vintage, epex=None, feature=None):
    """Build and fit a correction for ``vintage`` from the available price history."""
    if epex is None and shape_hourly is not None:
        epex = getattr(shape_hourly, "_solar_epex_hist", None)
    if epex is None:
        epex = pd.read_parquet(DEFAULT_EPEX_PATH)
    if "price_eur_mwh" not in epex.columns:
        # Single-column frames from a Series are common; coerce by position.
        epex = epex.rename(columns={epex.columns[0]: "price_eur_mwh"})

    baseline_provider = (
        shape_hourly.get if shape_hourly is not None and hasattr(shape_hourly, "get")
        else None
    )
    if baseline_provider is None:
        # Neutral baseline (flat) — yields a pure-realized residual; only used
        # when no fitted ShapeHourly is available (degenerate/test path).
        def baseline_provider(_s, _tj):  # noqa: ANN001
            return np.ones(24)

    corr = SolarBlockedFHCorrection(feature=feature)
    corr.fit(epex, baseline_provider, vintage)
    return corr
