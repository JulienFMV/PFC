from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import yaml

# CT pipeline is imported lazily inside ``run_short_term_phase`` so that
# importing this module (LT-only) does not pull heavy CT dependencies
# (lightgbm, torch, …) into the interpreter. The names are still exposed
# to type checkers via the TYPE_CHECKING guard below.
if TYPE_CHECKING:
    from pfc_shaping.pipeline.swiss_short_term import (  # noqa: F401
        SwissShortTermArtifacts,
        SwissShortTermInputs,
    )


@dataclass
class LoadedInputs:
    epex_ch: pd.DataFrame
    epex_de: pd.DataFrame
    neighbor_prices_15min: dict[str, pd.DataFrame]
    entso: pd.DataFrame
    hydro: pd.DataFrame
    cal_ch: pd.DataFrame
    cal_de: pd.DataFrame
    commodities: pd.DataFrame | None
    outages_all: pd.DataFrame | None
    config: dict
    sh_mode: str
    eex_report_path: str | None


@dataclass
class SharedStructuralArtifacts:
    si: object
    unc: object
    calibrator: object | None
    entso_forecast: pd.DataFrame
    start_date: str
    horizon_days: int


@dataclass
class MarketSpec:
    """Per-market description used to drive ``_build_long_term_branch``.

    Captures everything that distinguishes one country's PFC build from
    another. Adding a new market (FR / AT / IT) means assembling one
    ``MarketSpec``; the rest of the pipeline is market-agnostic.
    """

    code: str                          # ISO-2 market code: 'CH', 'DE', 'AT', 'FR', 'IT'
    sheet: str                         # EEX workbook sheet name (today: same as code)
    tz: str                            # IANA timezone (e.g. 'Europe/Zurich')
    country: str                       # holidays / EEX peak country code
    epex_df: pd.DataFrame              # market-specific EPEX spot history
    cal_df: pd.DataFrame               # market-specific calendar enrichment
    pre_fitted_sh: object | None = None        # pre-fitted ShapeHourly (CH); None → fit inside branch
    water_value: object | None = None          # only for hydro markets (CH today)
    hydro_forecast: pd.DataFrame | None = None  # only for hydro markets
    outages_forecast: pd.DataFrame | None = None  # only when REMIT outages are wired
    out_base: str = ""                 # destination prefix for the {.parquet, .csv} pair


@dataclass
class MarketBranchArtifacts:
    """Output of ``_build_long_term_branch`` for one market.

    Every field is per-market; the LongTermArtifacts container keeps
    a dict keyed by market code plus convenience aliases ``swiss`` /
    ``german`` for backward compatibility.
    """

    code: str
    pfc: pd.DataFrame
    sh: object
    base_prices: dict
    cascaded_prices: dict
    fwd_source: str
    out_base: str
    wv: object | None = None
    hydro_forecast: pd.DataFrame | None = None
    outages_forecast: pd.DataFrame | None = None


# ── Backward-compat aliases ──────────────────────────────────────────────
# Older external code refers to ``SwissLongTermArtifacts`` /
# ``GermanLongTermArtifacts``. They are now the same object as
# ``MarketBranchArtifacts``; the per-country attribute names
# (``base_prices_ch``, ``cascaded_prices_ch`` etc.) are preserved as
# read-only properties on top of the generic dataclass for any caller
# that depends on the old shape.
SwissLongTermArtifacts = MarketBranchArtifacts
GermanLongTermArtifacts = MarketBranchArtifacts


def _legacy_alias(self: MarketBranchArtifacts, attr: str) -> object:
    """Return self.attr (used by legacy *_ch / *_de property aliases)."""
    return getattr(self, attr)


# Attach legacy attribute aliases on the dataclass so ``art.base_prices_ch``
# keeps returning ``art.base_prices`` on a CH branch (and similarly _de on
# a DE branch). This lets the dashboard pages and any leftover scripts
# read the old field names without modification.
for _legacy_name, _modern_name in [
    ("base_prices_ch", "base_prices"),
    ("cascaded_prices_ch", "cascaded_prices"),
    ("base_prices_de", "base_prices"),
    ("cascaded_prices_de", "cascaded_prices"),
]:
    setattr(
        MarketBranchArtifacts,
        _legacy_name,
        property(lambda self, _m=_modern_name: _legacy_alias(self, _m)),
    )


@dataclass
class LongTermArtifacts:
    """Top-level LT artifacts container.

    ``markets`` is the source of truth: one entry per active market, keyed
    by ISO-2 code. ``swiss`` and ``german`` properties are convenience
    accessors for the two markets currently wired in production.
    Activating FR / AT / IT only requires adding entries to ``markets`` —
    no field added on this dataclass.
    """

    shared: SharedStructuralArtifacts
    markets: dict[str, MarketBranchArtifacts]
    out_dir: str
    artifacts_dir: str
    today: str
    monthly_curve_manifests: dict[str, dict[str, object]] = field(default_factory=dict)

    @property
    def swiss(self) -> MarketBranchArtifacts:
        return self.markets["CH"]

    @property
    def german(self) -> MarketBranchArtifacts:
        return self.markets["DE"]


def _first_existing_path(*paths: str | None) -> str | None:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def _read_required_parquet(path: str, label: str, logger: logging.Logger) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required {label} parquet: {path}")
    try:
        df = pd.read_parquet(path)
    except Exception:
        logger.exception("  Failed to read required %s parquet: %s", label, path)
        raise
    if df.empty:
        raise ValueError(f"Required {label} parquet is empty: {path}")
    return df


def load_inputs(project_root: str, logger: logging.Logger) -> LoadedInputs:
    logger.info("=" * 70)
    logger.info("STEP 1: Loading data")
    logger.info("=" * 70)
    t0 = time.time()

    data_dir = os.path.join(project_root, "pfc_shaping", "data")
    epex_ch = _read_required_parquet(os.path.join(data_dir, "epex_15min.parquet"), "EPEX CH", logger)
    epex_de = _read_required_parquet(os.path.join(data_dir, "epex_de_15min.parquet"), "EPEX DE", logger)
    neighbor_prices_15min = {
        "de": epex_de,
    }
    for code in ["at", "fr", "it"]:
        path = os.path.join(data_dir, f"epex_{code}_15min.parquet")
        if os.path.exists(path):
            neighbor_prices_15min[code] = pd.read_parquet(path)
    entso = _read_required_parquet(os.path.join(data_dir, "entso_15min.parquet"), "ENTSO-E", logger)
    hydro = _read_required_parquet(os.path.join(data_dir, "hydro_reservoir.parquet"), "Hydro reservoir", logger)

    logger.info("  EPEX CH:  %d rows  [%s -> %s]", len(epex_ch), epex_ch.index.min().date(), epex_ch.index.max().date())
    logger.info("  EPEX DE:  %d rows  [%s -> %s]", len(epex_de), epex_de.index.min().date(), epex_de.index.max().date())
    logger.info("  ENTSO-E:  %d rows  [%s -> %s]", len(entso), entso.index.min().date(), entso.index.max().date())
    logger.info("  Hydro:    %d rows  [%s -> %s]", len(hydro), hydro.index.min().date(), hydro.index.max().date())
    logger.info("  Data loaded in %.1fs", time.time() - t0)

    logger.info("=" * 70)
    logger.info("STEP 2: Calendar enrichment")
    logger.info("=" * 70)
    t1 = time.time()
    from pfc_shaping.data.calendar_ch import enrich_15min_index

    cal_ch = enrich_15min_index(epex_ch.index, country="CH")
    cal_de = enrich_15min_index(epex_de.index, country="DE")

    logger.info("  CH calendar: %d rows, types: %s", len(cal_ch), dict(cal_ch["type_jour"].value_counts()))
    logger.info("  DE calendar: %d rows", len(cal_de))
    logger.info("  Calendar enriched in %.1fs", time.time() - t1)

    logger.info("=" * 70)
    logger.info("STEP 3: Feature engineering (solar_regime, load_deviation)")
    logger.info("=" * 70)
    logger.info("  solar_regime stats: mean=%.2f, std=%.2f", entso["solar_regime"].mean(), entso["solar_regime"].std())
    logger.info("  load_deviation stats: mean=%.2f, std=%.2f", entso["load_deviation"].mean(), entso["load_deviation"].std())

    with open(os.path.join(project_root, "pfc_shaping", "config.yaml"), encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model_cfg = config.get("model", {})
    forwards_cfg = config.get("forwards", {})
    sh_mode = model_cfg.get("shape_hourly_mode", "table")
    eex_report_path = _first_existing_path(
        forwards_cfg.get("eex_report_path"),
        forwards_cfg.get("eex_report_path_unc"),
    )
    if eex_report_path:
        logger.info("  EEX report path selected: %s", eex_report_path)
    else:
        logger.warning(
            "  No configured EEX report path found on filesystem; fallback loader will use repo-local/proxy source."
        )

    commodities_path = os.path.join(project_root, "data", "commodities_cache.parquet")
    commodities = pd.read_parquet(commodities_path) if os.path.exists(commodities_path) else None
    outages_path = os.path.join(project_root, "pfc_shaping", "data", "outages_15min.parquet")
    outages_all = pd.read_parquet(outages_path) if os.path.exists(outages_path) else None

    return LoadedInputs(
        epex_ch=epex_ch,
        epex_de=epex_de,
        neighbor_prices_15min=neighbor_prices_15min,
        entso=entso,
        hydro=hydro,
        cal_ch=cal_ch,
        cal_de=cal_de,
        commodities=commodities,
        outages_all=outages_all,
        config=config,
        sh_mode=sh_mode,
        eex_report_path=eex_report_path,
    )


def run_long_term_phase(
    project_root: str,
    inputs: LoadedInputs,
    peak_source_policy: str,
    logger: logging.Logger,
) -> LongTermArtifacts:
    logger.info("=" * 70)
    logger.info("STEP 4: Fitting ShapeHourly on CH EPEX (full history)")
    logger.info("=" * 70)
    t2 = time.time()

    if inputs.sh_mode == "mlp":
        from pfc_shaping.lt.model.shape_hourly_mlp import ShapeHourlyMLP
        sh = ShapeHourlyMLP()
        logger.info("  Using ShapeHourlyMLP (neural)")
    else:
        from pfc_shaping.lt.model.shape_hourly import ShapeHourly
        sh = ShapeHourly()
        logger.info("  Using ShapeHourly (table)")

    sh.fit(inputs.epex_ch, inputs.cal_ch, hydro_df=inputs.hydro)

    if inputs.sh_mode == "mlp":
        logger.info("  MLP fitted (neural shape function)")
    else:
        logger.info("  Fitted %d (saison, type_jour) cells", len(sh.factors_))
        logger.info("  f_W ratios: %s", {k: round(v, 4) for k, v in sh.f_W_.items()})
        logger.info("  Sample Hiver/Ouvrable peak h=12: f_H=%.4f", sh.get("Hiver", "Ouvrable")[12])
    logger.info("  ShapeHourly fitted in %.1fs", time.time() - t2)

    logger.info("=" * 70)
    logger.info("STEP 5: Fitting ShapeIntraday on DE-LU post Oct 2025")
    logger.info("=" * 70)
    t3 = time.time()
    from pfc_shaping.lt.model.shape_intraday import ShapeIntraday

    cutoff_de = pd.Timestamp("2025-10-01", tz="UTC")
    epex_de_post = inputs.epex_de[inputs.epex_de.index >= cutoff_de]
    cal_de_post = inputs.cal_de.loc[epex_de_post.index]
    entso_de_post = inputs.entso.reindex(epex_de_post.index)

    logger.info("  DE-LU post Oct 2025: %d rows", len(epex_de_post))

    si = ShapeIntraday()
    si.fit(epex_de_post, entso_de_post, cal_de_post)

    logger.info("  Fitted %d (saison, type_jour, heure) cells", len(si.base_factors_))
    logger.info("  Corrections (layer 2): %d cells", len(si.corrections_))
    logger.info("  ShapeIntraday fitted in %.1fs", time.time() - t3)

    logger.info("=" * 70)
    logger.info("STEP 6: Fitting WaterValue correction")
    logger.info("=" * 70)
    t4 = time.time()
    from pfc_shaping.lt.model.water_value import WaterValueCorrection

    wv = WaterValueCorrection()
    wv.fit(inputs.epex_ch, inputs.hydro, inputs.cal_ch)

    logger.info("  beta_WV = %.4f", wv.beta_wv_)
    logger.info("  Season sensitivities: %s", {k: f"{v:.3f}" for k, v in wv.season_sensitivity_.items()})
    logger.info("  Calibration obs: %d", wv.n_obs_)
    logger.info("  WaterValue fitted in %.1fs", time.time() - t4)

    logger.info("=" * 70)
    logger.info("STEP 7: Fitting Uncertainty (n_boot=500, production quality)")
    logger.info("=" * 70)
    t5 = time.time()
    from pfc_shaping.lt.model.uncertainty import Uncertainty

    unc = Uncertainty(n_boot=500, seed=42)
    unc.fit(epex_de_post, cal_de_post)

    logger.info("  Bootstrap cells: %d", len(unc.boot_stats_))
    logger.info("  Uncertainty fitted in %.1fs", time.time() - t5)

    logger.info("=" * 70)
    logger.info("STEP 8: Building base_prices (EEX forward levels)")
    logger.info("=" * 70)
    from pfc_shaping.data.forward_proxy import load_base_prices as load_fwd_prices
    from pfc_shaping.calibration.cascading import ContractCascader
    from pfc_shaping.pipeline.monthly_curve_authority import (
        delivery_months_from_prices,
        latest_base_prices_by_market,
        monthly_solver_enabled,
        monthly_solver_settings,
        solve_monthly_level_authority,
    )

    base_prices_ch, fwd_source_ch = load_fwd_prices(
        inputs.epex_ch,
        eex_report_path=inputs.eex_report_path,
        config=inputs.config,
    )
    logger.info("  Forward source: %s", fwd_source_ch)

    cascader_ch = ContractCascader()
    cascader_ch.fit_seasonal_ratios(inputs.epex_ch)
    # Phase 5 D-A4-2 migration (NEG-04): fit_peak_spreads calibrates spreads
    # in €/MWh (sign-invariant for negative forwards). The deprecation shim
    # still routes fit_peak_ratios() → fit_peak_spreads() but explicit migration
    # avoids the runtime DeprecationWarning.
    # Phase 5 D-A2-1 default (negative-ready): no explicit enforce_* kwargs.
    # Legacy rollback per D-A2-3: pass allow_negative_peak=False at ContractCascader construction.
    cascader_ch.fit_peak_spreads(inputs.epex_ch)
    cascaded_prices_ch = cascader_ch.cascade(base_prices_ch)
    cascaded_prices_ch = cascader_ch.synthesize_peak_prices(cascaded_prices_ch)
    quoted_keys_ch = set(base_prices_ch.keys())
    cascader_for_ch_branch: object | None = cascader_ch

    monthly_authority_ch = None
    if monthly_solver_enabled(inputs.config, market="CH"):
        settings = monthly_solver_settings(inputs.config)
        history_path = settings.get("eex_history_path")
        history = pd.DataFrame()
        neighbor_prices: dict[str, dict[str, float]] = {}
        if history_path and os.path.exists(str(history_path)):
            history = pd.read_parquet(str(history_path))
            history["date"] = pd.to_datetime(history["date"]).dt.tz_localize(None).dt.normalize()
            for neighbor in settings.get("markets", ("DE", "FR", "AT", "IT")):
                try:
                    _, prices = latest_base_prices_by_market(history, market=str(neighbor).upper())
                except ValueError:
                    continue
                neighbor_prices[str(neighbor).upper()] = prices
        monthly_authority_ch = solve_monthly_level_authority(
            market="CH",
            delivery_months=delivery_months_from_prices(base_prices_ch),
            own_base_prices=base_prices_ch,
            all_market_base_prices=neighbor_prices,
            eex_history=history,
            settings=settings,
            timezone="Europe/Zurich",
            original_forward_prices=base_prices_ch,
        )
        cascaded_prices_ch = monthly_authority_ch.assembler_base_prices
        quoted_keys_ch = monthly_authority_ch.quoted_keys
        cascader_for_ch_branch = None
        logger.info(
            "Monthly curve solver enabled for CH: monthly_solution_hash=%s active_constraints_hash=%s",
            monthly_authority_ch.monthly_solution_hash,
            monthly_authority_ch.active_constraints_hash,
        )

    logger.info("  Input keys: %d", len(base_prices_ch))
    logger.info("  Cascaded keys: %d", len(cascaded_prices_ch))
    for key in sorted(cascaded_prices_ch.keys()):
        logger.info("    %s: %.2f EUR/MWh", key, cascaded_prices_ch[key])

    latest_fill_dev = inputs.hydro["fill_deviation"].iloc[-1]
    logger.info("  Latest hydro fill_deviation: %.3f (as of %s)", latest_fill_dev, inputs.hydro.index[-1].date())

    start_date = (pd.Timestamp.utcnow() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    max_fwd_year = max(int(k[:4]) for k in cascaded_prices_ch.keys() if k[:4].isdigit() and len(k) >= 4)
    end_of_last_year = pd.Timestamp(f"{max_fwd_year}-12-31", tz="UTC")
    future_start_ts = pd.Timestamp(start_date, tz="UTC")
    horizon_days = (end_of_last_year - future_start_ts).days + 1
    logger.info(
        "  Horizon: %s -> 31/12/%d = %d days (%.1f years)",
        start_date,
        max_fwd_year,
        horizon_days,
        horizon_days / 365,
    )
    future_start = pd.Timestamp(start_date, tz="UTC")
    future_end = future_start + pd.Timedelta(days=horizon_days)
    hydro_idx = pd.date_range(future_start, future_end, freq="W-MON", tz="UTC")
    decay = np.linspace(latest_fill_dev, 0.0, len(hydro_idx))
    hydro_forecast = pd.DataFrame({"fill_deviation": decay}, index=hydro_idx)

    logger.info("  Building ENTSO-E climatology forecast for N+3 horizon...")
    future_idx = pd.date_range(future_start, future_end, freq="15min", inclusive="left", tz="UTC")
    future_zurich = future_idx.tz_convert("Europe/Zurich")

    entso_zurich = inputs.entso.copy()
    entso_zurich["month"] = inputs.entso.index.tz_convert("Europe/Zurich").month
    entso_zurich["hour"] = inputs.entso.index.tz_convert("Europe/Zurich").hour
    entso_zurich["qh"] = (inputs.entso.index.minute // 15) + 1

    agg_dict = {
        "solar_regime_median": ("solar_regime", "median"),
        "load_deviation_median": ("load_deviation", "median"),
    }
    if "flow_deviation" in entso_zurich.columns:
        agg_dict["flow_deviation_median"] = ("flow_deviation", "median")

    clim = entso_zurich.groupby(["month", "hour", "qh"]).agg(**agg_dict).reset_index()
    future_keys = pd.DataFrame(
        {
            "month": future_zurich.month,
            "hour": future_zurich.hour,
            "qh": (future_zurich.minute // 15) + 1,
        },
        index=future_idx,
    )
    entso_forecast = future_keys.merge(clim, on=["month", "hour", "qh"], how="left").set_index(future_idx)

    rename_map = {
        "solar_regime_median": "solar_regime",
        "load_deviation_median": "load_deviation",
    }
    keep_cols = ["solar_regime", "load_deviation"]
    if "flow_deviation_median" in entso_forecast.columns:
        rename_map["flow_deviation_median"] = "flow_deviation"
        keep_cols.append("flow_deviation")
    entso_forecast = entso_forecast.rename(columns=rename_map)[keep_cols]
    entso_forecast["solar_regime"] = entso_forecast["solar_regime"].fillna(1.0)
    entso_forecast["load_deviation"] = entso_forecast["load_deviation"].fillna(0.0)
    if "flow_deviation" in entso_forecast.columns:
        entso_forecast["flow_deviation"] = entso_forecast["flow_deviation"].fillna(0.0)

    logger.info("  ENTSO-E climatology forecast: %d rows", len(entso_forecast))
    logger.info("  solar_regime: mean=%.2f  std=%.2f", entso_forecast["solar_regime"].mean(), entso_forecast["solar_regime"].std())
    logger.info("  load_deviation: mean=%.3f  std=%.3f", entso_forecast["load_deviation"].mean(), entso_forecast["load_deviation"].std())

    outages_forecast = None
    if inputs.outages_all is not None:
        outages_forecast = inputs.outages_all[inputs.outages_all.index >= future_start]
        logger.info(
            "  Outages forecast: %d rows, max unavail=%.0f MW",
            len(outages_forecast),
            outages_forecast["unavailable_mw"].max() if len(outages_forecast) > 0 else 0,
        )

    out_dir = os.path.join("pfc_shaping", "output")
    artifacts_dir = os.path.join("pfc_shaping", "model", "artifacts")
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    out_base_ch = os.path.join(out_dir, f"pfc_15min_{today}")
    out_base_de = os.path.join(out_dir, f"pfc_de_15min_{today}")

    shared = _build_shared_long_term_artifacts(
        si=si,
        unc=unc,
        entso_forecast=entso_forecast,
        start_date=start_date,
        horizon_days=horizon_days,
        logger=logger,
    )

    # Generic per-market builds. Adding FR / AT / IT later only requires
    # appending one more MarketSpec to this list (Phase 3).
    swiss_spec = MarketSpec(
        code="CH",
        sheet="CH",
        tz="Europe/Zurich",
        country="CH",
        epex_df=inputs.epex_ch,
        cal_df=inputs.cal_ch,
        pre_fitted_sh=sh,
        water_value=wv,
        hydro_forecast=hydro_forecast,
        outages_forecast=outages_forecast,
        out_base=out_base_ch,
    )
    swiss = _build_long_term_branch(
        spec=swiss_spec,
        inputs=inputs,
        shared=shared,
        peak_source_policy=peak_source_policy,
        logger=logger,
        # CH reuses the up-front computed forwards / cascader to avoid
        # parsing the EEX XLSX twice on the same run.
        pre_loaded_base_prices=base_prices_ch,
        pre_loaded_fwd_source=fwd_source_ch,
        pre_loaded_cascaded_prices=cascaded_prices_ch,
        pre_loaded_cascader=cascader_for_ch_branch,
        pre_loaded_quoted_keys=quoted_keys_ch,
        monthly_level_authority="solver" if monthly_authority_ch is not None else "legacy",
        skip_legacy_level_cascade=monthly_authority_ch is not None,
        skip_legacy_base_smoothing=monthly_authority_ch is not None,
    )

    german_spec = MarketSpec(
        code="DE",
        sheet="DE",
        tz="Europe/Berlin",
        country="DE",
        epex_df=inputs.epex_de,
        cal_df=inputs.cal_de,
        pre_fitted_sh=None,        # German ShapeHourly is fit inside the branch
        water_value=None,
        hydro_forecast=None,
        outages_forecast=None,
        out_base=out_base_de,
    )
    german = _build_long_term_branch(
        spec=german_spec,
        inputs=inputs,
        shared=shared,
        peak_source_policy=peak_source_policy,
        logger=logger,
    )

    markets: dict[str, MarketBranchArtifacts] = {"CH": swiss, "DE": german}

    return LongTermArtifacts(
        shared=shared,
        markets=markets,
        out_dir=out_dir,
        artifacts_dir=artifacts_dir,
        today=today,
        monthly_curve_manifests={"CH": monthly_authority_ch.manifest} if monthly_authority_ch is not None else {},
    )


def run_short_term_phase(
    project_root: str,
    inputs: LoadedInputs,
    long_term: LongTermArtifacts,
    logger: logging.Logger,
):
    """Run the Swiss CT overlay on top of the LT PFC.

    The CT pipeline is imported lazily so the LT module remains
    importable in environments without CT dependencies.
    """
    from pfc_shaping.pipeline.swiss_short_term import (
        SwissShortTermInputs,
        run_swiss_short_term_overlay,
    )

    st_inputs = SwissShortTermInputs(
        epex_ch=inputs.epex_ch,
        epex_de=inputs.epex_de,
        neighbor_prices_15min=inputs.neighbor_prices_15min,
        entso=inputs.entso,
        hydro=inputs.hydro,
        commodities=inputs.commodities,
        outages_all=inputs.outages_all,
        base_pfc_ch=long_term.swiss.pfc,
        require_de_exogenous=os.getenv("PFC_CT_REQUIRE_DE_EXOGENOUS", "1") == "1",
        required_neighbor_codes=("de",),
    )
    return run_swiss_short_term_overlay(project_root=project_root, inputs=st_inputs, logger=logger)


def _build_shared_long_term_artifacts(
    si: object,
    unc: object,
    entso_forecast: pd.DataFrame,
    start_date: str,
    horizon_days: int,
    logger: logging.Logger,
) -> SharedStructuralArtifacts:
    logger.info("=" * 70)
    logger.info("STEP 9a: Finalizing shared LT structural context")
    logger.info("=" * 70)

    calibrator = None
    try:
        from pfc_shaping.calibration.arbitrage_free import ArbitrageFreeCalibrator
        calibrator = ArbitrageFreeCalibrator(smoothness_weight=1.0, tol=0.01)
        logger.info("  ArbitrageFreeCalibrator loaded OK")
    except Exception as exc:
        logger.warning("  ArbitrageFreeCalibrator unavailable: %s", exc)

    return SharedStructuralArtifacts(
        si=si,
        unc=unc,
        calibrator=calibrator,
        entso_forecast=entso_forecast,
        start_date=start_date,
        horizon_days=horizon_days,
    )


def _build_long_term_branch(
    spec: MarketSpec,
    inputs: LoadedInputs,
    shared: SharedStructuralArtifacts,
    peak_source_policy: str,
    logger: logging.Logger,
    *,
    pre_loaded_base_prices: dict | None = None,
    pre_loaded_fwd_source: str | None = None,
    pre_loaded_cascaded_prices: dict | None = None,
    pre_loaded_cascader: object | None = None,
    pre_loaded_quoted_keys: set[str] | None = None,
    monthly_level_authority: str = "legacy",
    skip_legacy_level_cascade: bool = False,
    skip_legacy_base_smoothing: bool = False,
) -> MarketBranchArtifacts:
    """Build one market's PFC from a MarketSpec.

    Centralises the previous Swiss / German branch logic so each new
    market only requires a MarketSpec and the up-front data wiring
    (EPEX history, calendar). The Swiss branch is the only one that
    reuses pre-fitted ShapeHourly / cascader / forward dict from
    ``run_long_term_phase`` to avoid recomputing them; other markets
    fit inside this function.

    Args:
        spec: per-market description.
        inputs: shared LoadedInputs (config, eex_report_path, etc.).
        shared: shared structural artifacts (ShapeIntraday, Uncertainty,
            ENTSO forecast, calibrator, horizon).
        peak_source_policy: see PFCAssembler.
        logger: pipeline logger.
        pre_loaded_*: optional precomputed forward dict / cascader /
            ShapeHourly. Used by the CH branch which builds them once
            in ``run_long_term_phase`` and reuses them; for other
            markets these are ``None`` and the function fits internally.

    Returns:
        MarketBranchArtifacts with the assembled 15-min PFC.
    """
    logger.info("=" * 70)
    logger.info("STEP 9.%s: Building %s PFC (LT branch, tz=%s)", spec.code, spec.code, spec.tz)
    logger.info("=" * 70)
    t0 = time.time()

    from pfc_shaping.calibration.cascading import ContractCascader
    from pfc_shaping.data.forward_proxy import load_base_prices as load_fwd_prices
    from pfc_shaping.lt.model.assembler import PFCAssembler

    # ── 1. ShapeHourly: reuse pre-fit if provided, else fit on the spot ──
    if spec.pre_fitted_sh is not None:
        sh = spec.pre_fitted_sh
        logger.info("  %s ShapeHourly: reusing pre-fitted instance", spec.code)
    else:
        if inputs.sh_mode == "mlp":
            from pfc_shaping.lt.model.shape_hourly_mlp import ShapeHourlyMLP
            sh = ShapeHourlyMLP()
        else:
            from pfc_shaping.lt.model.shape_hourly import ShapeHourly
            sh = ShapeHourly()
        sh.fit(spec.epex_df, spec.cal_df)
        logger.info("  %s ShapeHourly fitted (%s mode)", spec.code, inputs.sh_mode)

    # ── 2. Forwards (base_prices) ────────────────────────────────────────
    if pre_loaded_base_prices is not None:
        base_prices = pre_loaded_base_prices
        fwd_source = pre_loaded_fwd_source or "pre-loaded"
    else:
        base_prices, fwd_source = load_fwd_prices(
            spec.epex_df,
            eex_report_path=inputs.eex_report_path,
            config=inputs.config,
            market=spec.sheet,
        )
    logger.info("  %s forward source: %s", spec.code, fwd_source)

    # ── 3. Cascading ────────────────────────────────────────────────────
    if pre_loaded_cascader is not None and pre_loaded_cascaded_prices is not None:
        cascader = pre_loaded_cascader
        cascaded_prices = pre_loaded_cascaded_prices
    elif skip_legacy_level_cascade and pre_loaded_cascaded_prices is not None:
        cascader = None
        cascaded_prices = pre_loaded_cascaded_prices
    else:
        # Phase 5 D-A2-1 default (negative-ready): no explicit enforce_* kwargs.
        # Legacy rollback per D-A2-3: pass allow_negative_peak=False at ContractCascader construction.
        cascader = ContractCascader(tz=spec.tz)
        cascader.fit_seasonal_ratios(spec.epex_df)
        # Phase 5 D-A4-2 migration (NEG-04) — see comment at the analogous callsite above.
        cascader.fit_peak_spreads(spec.epex_df)
        cascaded_prices = cascader.cascade(base_prices)
        cascaded_prices = cascader.synthesize_peak_prices(cascaded_prices)

    logger.info("  %s cascaded keys: %d", spec.code, len(cascaded_prices))
    for key in sorted(cascaded_prices.keys()):
        logger.info("    %s %s: %.2f EUR/MWh", spec.code, key, cascaded_prices[key])

    # ── 4. Assemble ─────────────────────────────────────────────────────
    assembler = PFCAssembler(
        shape_hourly=sh,
        shape_intraday=shared.si,
        uncertainty=shared.unc,
        water_value=spec.water_value,
        cascader=cascader,
        calibrator=shared.calibrator,
        peak_source_policy=peak_source_policy,
        monthly_level_authority=monthly_level_authority,
        skip_legacy_level_cascade=skip_legacy_level_cascade,
        skip_legacy_base_smoothing=skip_legacy_base_smoothing,
    )
    build_kwargs = dict(
        base_prices=cascaded_prices,
        quoted_keys=set(pre_loaded_quoted_keys) if pre_loaded_quoted_keys is not None else set(base_prices.keys()),
        start_date=shared.start_date,
        horizon_days=shared.horizon_days,
        entso_forecast=shared.entso_forecast,
        hydro_forecast=spec.hydro_forecast,
    )
    # Only the Swiss branch (with country='CH' default) consumes outages today.
    # Other markets pass country=spec.country explicitly so the assembler
    # picks the right tz / holidays.
    if spec.outages_forecast is not None:
        build_kwargs["outages_forecast"] = spec.outages_forecast
    if spec.country != "CH":
        build_kwargs["country"] = spec.country

    pfc = assembler.build(**build_kwargs)
    logger.info("  %s PFC assembled: %d rows in %.1fs", spec.code, len(pfc), time.time() - t0)

    return MarketBranchArtifacts(
        code=spec.code,
        pfc=pfc,
        sh=sh,
        base_prices=base_prices,
        cascaded_prices=cascaded_prices,
        fwd_source=fwd_source,
        out_base=spec.out_base,
        wv=spec.water_value,
        hydro_forecast=spec.hydro_forecast,
        outages_forecast=spec.outages_forecast,
    )


# Per-market suffix used when writing artifacts. CH keeps the legacy
# unsuffixed names ("shape_hourly.parquet", "water_value.parquet") so
# the dashboard and any external loader keeps reading the historical
# paths. Other markets get an explicit lower-case suffix.
_ARTIFACT_SUFFIX: dict[str, str] = {
    "CH": "",
    "DE": "_de",
    "AT": "_at",
    "FR": "_fr",
    "IT": "_it",
}


def _save_market_artifacts(
    art: MarketBranchArtifacts,
    artifacts_dir: str,
    logger: logging.Logger,
) -> None:
    """Persist PFC + per-market shape / water value artifacts."""
    art.pfc.to_parquet(f"{art.out_base}.parquet")
    logger.info("  Saved: %s.parquet (%d rows)", art.out_base, len(art.pfc))
    art.pfc.to_csv(f"{art.out_base}.csv", sep=";", index_label="timestamp_local")
    logger.info("  Saved: %s.csv", art.out_base)

    suffix = _ARTIFACT_SUFFIX.get(art.code, f"_{art.code.lower()}")

    if hasattr(art.sh, "save"):
        if art.sh.__class__.__name__ == "ShapeHourlyMLP":
            art.sh.save(os.path.join(artifacts_dir, f"shape_hourly{suffix}_mlp.pkl"))
        else:
            art.sh.save(os.path.join(artifacts_dir, f"shape_hourly{suffix}.parquet"))

    if art.wv is not None and hasattr(art.wv, "save"):
        art.wv.save(os.path.join(artifacts_dir, f"water_value{suffix}.parquet"))


def _save_monthly_curve_manifests(
    manifests: dict[str, dict[str, object]],
    artifacts_dir: str,
    logger: logging.Logger,
) -> None:
    for market, manifest in sorted((manifests or {}).items()):
        suffix = "" if str(market).upper() == "CH" else f"_{str(market).lower()}"
        path = os.path.join(artifacts_dir, f"production_monthly_curve_manifest{suffix}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, default=str)
        logger.info("  Saved monthly curve manifest: %s", path)


def save_long_term_outputs(long_term: LongTermArtifacts, logger: logging.Logger) -> None:
    logger.info("=" * 70)
    logger.info("STEP 10: Saving output")
    logger.info("=" * 70)

    os.makedirs(long_term.out_dir, exist_ok=True)
    os.makedirs(long_term.artifacts_dir, exist_ok=True)

    for art in long_term.markets.values():
        _save_market_artifacts(art, long_term.artifacts_dir, logger)
    _save_monthly_curve_manifests(long_term.monthly_curve_manifests, long_term.artifacts_dir, logger)

    # Shared (cross-market) artifacts written once.
    long_term.shared.si.save(os.path.join(long_term.artifacts_dir, "shape_intraday.parquet"))
    long_term.shared.unc.save(os.path.join(long_term.artifacts_dir, "uncertainty.parquet"))


def print_pipeline_summary(long_term: LongTermArtifacts, total_time: float) -> None:
    pfc = long_term.swiss.pfc
    pfc_de = long_term.german.pfc
    idx_zurich = pfc.index.tz_convert("Europe/Zurich")

    print("\n" + "=" * 80)
    print(f"  PFC 15min PRODUCTION RUN - {long_term.today}")
    print("  Horizon: %s -> %s (%d days)" % (pfc.index.min().date(), pfc.index.max().date(), long_term.shared.horizon_days))
    print("  Timestamps: %d (15min intervals)" % len(pfc))
    print("=" * 80)

    print("\n--- PRICE DISTRIBUTION (EUR/MWh) ---")
    print("  Mean:    %.2f" % pfc["price_shape"].mean())
    print("  Median:  %.2f" % pfc["price_shape"].median())
    print("  Std:     %.2f" % pfc["price_shape"].std())
    print("  Min:     %.2f" % pfc["price_shape"].min())
    print("  Max:     %.2f" % pfc["price_shape"].max())
    print("  p5:      %.2f" % pfc["price_shape"].quantile(0.05))
    print("  p95:     %.2f" % pfc["price_shape"].quantile(0.95))

    print("\n--- FACTOR RANGES ---")
    for col in ["f_S", "f_W", "f_H", "f_Q", "f_WV"]:
        series = pfc[col]
        print("  %-5s: mean=%.4f  min=%.4f  max=%.4f  std=%.4f" % (col, series.mean(), series.min(), series.max(), series.std()))

    print("\n--- f_Q INTRADAY DETAIL ---")
    print("  f_Q range: [%.4f, %.4f]" % (pfc["f_Q"].min(), pfc["f_Q"].max()))
    print("  f_Q mean:  %.4f (should be ~1.0)" % pfc["f_Q"].mean())
    pfc_zurich = pfc.copy()
    pfc_zurich["hour"] = pfc.index.tz_convert("Europe/Zurich").hour
    for hour in [0, 6, 8, 12, 17, 20, 23]:
        mask = pfc_zurich["hour"] == hour
        print("    h=%02d: f_Q mean=%.4f  std=%.4f" % (hour, pfc_zurich.loc[mask, "f_Q"].mean(), pfc_zurich.loc[mask, "f_Q"].std()))

    print("\n--- CONFIDENCE INTERVAL WIDTH BY HORIZON ---")
    if pfc["p10"].notna().any():
        pfc_zurich["ic_width"] = pfc["p90"] - pfc["p10"]
        for profile_type in ["M+1..M+6", "M+7..M+12", "Y+2/Y+3"]:
            mask = pfc["profile_type"] == profile_type
            if mask.any():
                width = pfc_zurich.loc[mask, "ic_width"]
                price = pfc.loc[mask, "price_shape"]
                rel_width = (width / price.abs().clip(lower=1.0)).mean() * 100
                print(
                    "  %-12s: mean_width=%.2f EUR/MWh  mean_price=%.2f  rel_width=%.1f%%"
                    % (profile_type, width.mean(), price.mean(), rel_width)
                )
    else:
        print("  (No IC computed)")

    print("\n--- ENERGY CONSISTENCY (mean PFC vs forward) ---")
    for key in sorted(long_term.swiss.cascaded_prices_ch.keys()):
        base = long_term.swiss.cascaded_prices_ch[key]
        if len(key) == 4 and key.isdigit():
            mask = idx_zurich.year == int(key)
            label = f"Cal {key}"
        elif key.endswith("-Peak"):
            continue
        elif "Q" in key:
            year = int(key[:4])
            quarter = int(key.split("Q")[1][0])
            q_months = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}[quarter]
            mask = (idx_zurich.year == year) & (idx_zurich.month.isin(q_months))
            label = f"  {key}"
        elif len(key) == 7 and key[4] == "-":
            year = int(key[:4])
            month = int(key[5:])
            mask = (idx_zurich.year == year) & (idx_zurich.month == month)
            label = f"    {key}"
        else:
            continue

        n_pts = mask.sum()
        if n_pts == 0:
            continue
        mean_pfc = pfc.loc[mask, "price_shape"].mean()
        dev_pct = (mean_pfc - base) / abs(base) * 100
        marker = "OK" if abs(dev_pct) < 5.0 else "WARN"
        print("  %-14s: fwd=%.2f  pfc=%.2f  dev=%+.2f%%  n=%d  [%s]" % (label, base, mean_pfc, dev_pct, n_pts, marker))

    print("\n--- CALIBRATION STATUS ---")
    if pfc["calibrated"].any():
        print("  ArbitrageFree calibration: APPLIED")
    else:
        print("  ArbitrageFree calibration: NOT APPLIED (raw shape only)")

    print("\n--- ANNUAL AVERAGES ---")
    for year in sorted(idx_zurich.year.unique()):
        mask = idx_zurich.year == year
        if mask.sum() > 0:
            price = pfc.loc[mask, "price_shape"]
            print("  %d: mean=%.2f  min=%.2f  max=%.2f  n=%d" % (year, price.mean(), price.min(), price.max(), mask.sum()))

    print("\n--- PROFILE TYPE DISTRIBUTION ---")
    for profile_type, count in pfc["profile_type"].value_counts().items():
        pct = count / len(pfc) * 100
        print("  %-12s: %d rows (%.1f%%)" % (profile_type, count, pct))

    print("\n--- PEAK / OFF-PEAK ANALYSIS ---")
    hour = idx_zurich.hour
    dow = idx_zurich.dayofweek
    is_peak = (hour >= 8) & (hour < 20) & (dow < 5)
    for year in sorted(idx_zurich.year.unique()):
        yr_mask = idx_zurich.year == year
        if yr_mask.sum() == 0:
            continue
        peak = pfc.loc[yr_mask & is_peak, "price_shape"]
        offpeak = pfc.loc[yr_mask & ~is_peak, "price_shape"]
        if len(peak) > 0 and len(offpeak) > 0:
            spread = peak.mean() - offpeak.mean()
            ratio = peak.mean() / offpeak.mean() if offpeak.mean() != 0 else float("nan")
            print("  %d: peak=%.2f  offpeak=%.2f  spread=%.2f  ratio=%.3f" % (year, peak.mean(), offpeak.mean(), spread, ratio))

    print("\n" + "=" * 80)
    print("  DE PFC SUMMARY")
    print("=" * 80)
    print("  Timestamps: %d" % len(pfc_de))
    print("  Mean price: %.2f EUR/MWh" % pfc_de["price_shape"].mean())
    print("  Min: %.2f  Max: %.2f" % (pfc_de["price_shape"].min(), pfc_de["price_shape"].max()))

    idx_de_local = pfc_de.index.tz_convert("Europe/Berlin")
    print("\n--- DE ANNUAL AVERAGES ---")
    for year in sorted(idx_de_local.year.unique()):
        mask = idx_de_local.year == year
        if mask.sum() > 0:
            price = pfc_de.loc[mask, "price_shape"]
            print("  %d: mean=%.2f  min=%.2f  max=%.2f  n=%d" % (year, price.mean(), price.min(), price.max(), mask.sum()))

    print("\n--- DE PEAK / OFF-PEAK ---")
    hour_de = idx_de_local.hour
    dow_de = idx_de_local.dayofweek
    is_peak_de = (hour_de >= 8) & (hour_de < 20) & (dow_de < 5)
    for year in sorted(idx_de_local.year.unique()):
        yr_mask = idx_de_local.year == year
        if yr_mask.sum() == 0:
            continue
        peak = pfc_de.loc[yr_mask & is_peak_de, "price_shape"]
        offpeak = pfc_de.loc[yr_mask & ~is_peak_de, "price_shape"]
        if len(peak) > 0 and len(offpeak) > 0:
            spread = peak.mean() - offpeak.mean()
            ratio = peak.mean() / offpeak.mean() if offpeak.mean() != 0 else float("nan")
            print("  %d: peak=%.2f  offpeak=%.2f  spread=%.2f  ratio=%.3f" % (year, peak.mean(), offpeak.mean(), spread, ratio))

    print("\n" + "=" * 80)
    print("  TOTAL EXECUTION TIME: %.1f seconds" % total_time)
    print("  Output CH: %s.parquet + .csv" % long_term.swiss.out_base)
    print("  Output DE: %s.parquet + .csv" % long_term.german.out_base)
    print("=" * 80)
