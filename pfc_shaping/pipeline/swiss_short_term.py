from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from pfc_shaping.ct.model.lear_forecaster import LEARForecaster
from pfc_shaping.storage.local_duckdb import init_db, upsert_lear_backtest, upsert_lear_forecast

DEFAULT_PRICEFM_PYTHON = r"C:\Users\jbattaglia\.conda\pricefm_tf\python.exe"


@dataclass
class SwissShortTermInputs:
    epex_ch: pd.DataFrame
    epex_de: pd.DataFrame
    neighbor_prices_15min: dict[str, pd.DataFrame] | None
    entso: pd.DataFrame
    hydro: pd.DataFrame
    commodities: pd.DataFrame | None
    outages_all: pd.DataFrame | None
    base_pfc_ch: pd.DataFrame
    require_de_exogenous: bool = True
    required_neighbor_codes: tuple[str, ...] = ("de",)


@dataclass
class SwissShortTermInputHealth:
    epex_ch_rows: int
    epex_de_rows: int
    entso_rows: int
    hydro_rows: int
    outages_rows: int
    commodities_rows: int
    ch_de_overlap_hours: int
    entso_ch_overlap_hours: int
    has_de_price_support: bool
    neighbor_overlap_hours: dict[str, int]
    has_required_neighbor_support: bool


@dataclass
class SwissShortTermArtifacts:
    pfc_ch: pd.DataFrame
    lear_forecast: pd.DataFrame
    lear_run_id: str | None


def run_swiss_short_term_overlay(
    project_root: str,
    inputs: SwissShortTermInputs,
    logger: logging.Logger,
) -> SwissShortTermArtifacts:
    logger.info("=" * 70)
    logger.info("STEP 10: LEAR short-term forecast (Swiss CT branch)")
    logger.info("=" * 70)
    t_lear = time.time()

    input_health = _validate_swiss_short_term_inputs(inputs, logger)
    _save_input_health(input_health, logger)

    lear = LEARForecaster(tz="Europe/Zurich")
    lear.fit(
        epex_15min=inputs.epex_ch,
        entso_15min=inputs.entso,
        outages_15min=inputs.outages_all,
        commodities=inputs.commodities,
        hydro=inputs.hydro,
        epex_de_15min=inputs.epex_de,
        neighbor_price_15min=inputs.neighbor_prices_15min,
    )

    lear_forecast = lear.predict(horizon_days=10)
    logger.info("  LEAR forecast: %d hours, mean=%.1f EUR/MWh", len(lear_forecast), lear_forecast["price_lear"].mean())

    lear_forecast = _maybe_apply_experimental_pricefm(project_root, lear_forecast, logger)
    pfc_ch = lear.blend_with_pfc(inputs.base_pfc_ch.copy(), lear_forecast)

    lear_base = os.path.join("pfc_shaping", "output", f"lear_forecast_{pd.Timestamp.now().strftime('%Y-%m-%d')}")
    lear_forecast.to_parquet(f"{lear_base}.parquet", index=False)
    lear_forecast.to_csv(f"{lear_base}.csv", index=False)
    lear_forecast.to_parquet(os.path.join("pfc_shaping", "output", "lear_forecast_latest.parquet"), index=False)
    lear_forecast.to_csv(os.path.join("pfc_shaping", "output", "lear_forecast_latest.csv"), index=False)
    logger.info("  LEAR standalone saved: %s.parquet", lear_base)

    lear_run_id = _persist_lear_forecast(lear_forecast, logger)
    _maybe_run_lear_backtest(lear, lear_run_id, logger)

    logger.info("  LEAR completed in %.1fs", time.time() - t_lear)
    return SwissShortTermArtifacts(pfc_ch=pfc_ch, lear_forecast=lear_forecast, lear_run_id=lear_run_id)


def _validate_swiss_short_term_inputs(
    inputs: SwissShortTermInputs,
    logger: logging.Logger,
) -> SwissShortTermInputHealth:
    if inputs.epex_ch.empty:
        raise ValueError("Swiss CT requires non-empty CH EPEX input.")
    if inputs.entso.empty:
        raise ValueError("Swiss CT requires non-empty ENTSO input.")
    if inputs.hydro.empty:
        raise ValueError("Swiss CT requires non-empty hydro input.")
    if inputs.base_pfc_ch.empty:
        raise ValueError("Swiss CT requires non-empty Swiss LT base PFC.")

    ch_hourly_idx = inputs.epex_ch.index.to_series().resample("h").mean().dropna().index
    de_hourly_idx = inputs.epex_de.index.to_series().resample("h").mean().dropna().index if not inputs.epex_de.empty else pd.DatetimeIndex([])
    entso_hourly_idx = inputs.entso.index.to_series().resample("h").mean().dropna().index

    ch_de_overlap_hours = len(ch_hourly_idx.intersection(de_hourly_idx))
    entso_ch_overlap_hours = len(ch_hourly_idx.intersection(entso_hourly_idx))
    has_de_price_support = ch_de_overlap_hours > 0
    neighbor_overlap_hours: dict[str, int] = {}
    for code, df in (inputs.neighbor_prices_15min or {}).items():
        if df is None or df.empty:
            neighbor_overlap_hours[code] = 0
            continue
        nb_hourly_idx = df.index.to_series().resample("h").mean().dropna().index
        neighbor_overlap_hours[code] = len(ch_hourly_idx.intersection(nb_hourly_idx))

    has_required_neighbor_support = all(
        neighbor_overlap_hours.get(code, 0) > 0 for code in inputs.required_neighbor_codes
    )

    if inputs.require_de_exogenous and not has_required_neighbor_support:
        raise ValueError(
            f"Swiss CT is configured to require neighbor exogenous prices {inputs.required_neighbor_codes}, "
            f"but overlap was insufficient: {neighbor_overlap_hours}"
        )

    health = SwissShortTermInputHealth(
        epex_ch_rows=len(inputs.epex_ch),
        epex_de_rows=len(inputs.epex_de),
        entso_rows=len(inputs.entso),
        hydro_rows=len(inputs.hydro),
        outages_rows=0 if inputs.outages_all is None else len(inputs.outages_all),
        commodities_rows=0 if inputs.commodities is None else len(inputs.commodities),
        ch_de_overlap_hours=ch_de_overlap_hours,
        entso_ch_overlap_hours=entso_ch_overlap_hours,
        has_de_price_support=has_de_price_support,
        neighbor_overlap_hours=neighbor_overlap_hours,
        has_required_neighbor_support=has_required_neighbor_support,
    )
    logger.info(
        "  Swiss CT input health: CH=%d rows, DE=%d rows, ENTSO=%d rows, hydro=%d rows, CH/DE overlap=%d h, CH/ENTSO overlap=%d h, neighbors=%s",
        health.epex_ch_rows,
        health.epex_de_rows,
        health.entso_rows,
        health.hydro_rows,
        health.ch_de_overlap_hours,
        health.entso_ch_overlap_hours,
        health.neighbor_overlap_hours,
    )
    if not has_required_neighbor_support:
        logger.warning("  Swiss CT running without full required neighbor price overlap support.")
    return health


def _save_input_health(health: SwissShortTermInputHealth, logger: logging.Logger) -> None:
    path = os.path.join("pfc_shaping", "output", "swiss_ct_input_health_latest.json")
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(asdict(health), handle, indent=2)
        logger.info("  Swiss CT input health saved: %s", path)
    except Exception as exc:
        logger.warning("  Failed to persist Swiss CT input health: %s", exc)


def _maybe_apply_experimental_pricefm(
    project_root: str,
    lear_forecast: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    if os.getenv("PFC_ENABLE_PRICEFM_EXPERIMENT", "0") != "1":
        return lear_forecast

    try:
        from pfc_shaping.ct.model.futureboost_experimental import (
            DEFAULT_FUTUREBOOST_EXPERIMENT,
            apply_futureboost_experimental,
        )
        from pfc_shaping.ct.model.pricefm_experimental import (
            BEST_PRICEFM_EXPERIMENT,
            blend_lear_with_pricefm,
            load_pricefm_forecast,
        )

        pricefm_path = BEST_PRICEFM_EXPERIMENT.forecast_latest_path
        pricefm_meta_path = os.path.join("pfc_shaping", "output", "pricefm_forecast_latest_meta.json")
        generate_pricefm = os.getenv("PFC_GENERATE_PRICEFM_EXPERIMENT", "1") == "1"
        pricefm_blend_mode = os.getenv("PFC_PRICEFM_BLEND_MODE", "regime").strip().lower()

        if generate_pricefm:
            try:
                pricefm_python = os.getenv("PFC_PRICEFM_PYTHON", DEFAULT_PRICEFM_PYTHON)
                if not os.path.exists(pricefm_python):
                    pricefm_python = sys.executable
                pricefm_cmd = [
                    pricefm_python,
                    os.path.join(project_root, "scripts", "generate_pricefm_experimental_forecast.py"),
                    "--horizon-days",
                    "10",
                ]
                logger.info("  Generating experimental PriceFM forecast...")
                completed = subprocess.run(
                    pricefm_cmd,
                    cwd=project_root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if completed.stdout:
                    logger.info("  PriceFM generation stdout: %s", completed.stdout.strip())
            except Exception as gen_exc:
                if isinstance(gen_exc, subprocess.CalledProcessError) and gen_exc.stderr:
                    logger.warning("  Experimental PriceFM stderr: %s", gen_exc.stderr.strip())
                logger.warning("  Experimental PriceFM generation failed: %s", gen_exc)

        if not os.path.exists(pricefm_path):
            logger.info("  Experimental PriceFM forecast not found at %s", pricefm_path)
            return lear_forecast

        pricefm_forecast = load_pricefm_forecast(pricefm_path)
        if pricefm_blend_mode == "futureboost":
            use_cin = os.getenv("PFC_FUTUREBOOST_USE_CIN", "0") == "1"
            use_qra = os.getenv("PFC_FUTUREBOOST_USE_QRA", "1") == "1"
            futureboost_cfg = replace(
                DEFAULT_FUTUREBOOST_EXPERIMENT,
                use_causal_instance_norm=use_cin,
                use_qra_quantiles=use_qra,
            )
            lear_forecast_pricefm, futureboost_meta = apply_futureboost_experimental(
                lear_forecast,
                pricefm_forecast,
                config=futureboost_cfg,
            )
            logger.info(
                "  Experimental FutureBoost applied: %.0f rows, alpha=%s, train_rows=%d, use_cin=%s, use_qra=%s, qra_enabled=%s",
                lear_forecast_pricefm["futureboost_used"].sum(),
                futureboost_meta.get("alpha"),
                futureboost_meta.get("train_rows"),
                "1" if use_cin else "0",
                "1" if use_qra else "0",
                futureboost_meta.get("qra_enabled"),
            )
        else:
            lear_forecast_pricefm = blend_lear_with_pricefm(
                lear_forecast,
                pricefm_forecast,
                weight_pricefm=None if pricefm_blend_mode == "regime" else BEST_PRICEFM_EXPERIMENT.blend_weight,
            )

        logger.info(
            "  Experimental PriceFM blend applied: %.0f rows, mode=%s",
            (
                lear_forecast_pricefm["futureboost_used"].sum()
                if "futureboost_used" in lear_forecast_pricefm.columns
                else lear_forecast_pricefm["pricefm_used"].sum()
            ),
            pricefm_blend_mode,
        )
        if os.path.exists(pricefm_meta_path):
            logger.info("  Experimental PriceFM metadata: %s", pricefm_meta_path)

        pricefm_base = os.path.join("pfc_shaping", "output", f"lear_forecast_pricefm_experimental_{pd.Timestamp.now().strftime('%Y-%m-%d')}")
        lear_forecast_pricefm.to_parquet(f"{pricefm_base}.parquet", index=False)
        lear_forecast_pricefm.to_csv(f"{pricefm_base}.csv", index=False)
        lear_forecast_pricefm.to_parquet(
            os.path.join("pfc_shaping", "output", "lear_forecast_pricefm_experimental_latest.parquet"),
            index=False,
        )
        lear_forecast_pricefm.to_csv(
            os.path.join("pfc_shaping", "output", "lear_forecast_pricefm_experimental_latest.csv"),
            index=False,
        )

        if os.getenv("PFC_APPLY_PRICEFM_EXPERIMENT_TO_PFC", "0") == "1":
            logger.info("  Experimental PriceFM blend promoted into PFC overlay")
            return lear_forecast_pricefm.copy()
        return lear_forecast
    except Exception as pricefm_exc:
        logger.warning("  Experimental PriceFM blend failed: %s", pricefm_exc)
        return lear_forecast


def _persist_lear_forecast(lear_forecast: pd.DataFrame, logger: logging.Logger) -> str | None:
    lear_run_id = None
    try:
        db_path = init_db(os.path.join("pfc_shaping", "data", "pfc_local.duckdb"))
        lear_run_id = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S_LEAR")
        upsert_lear_forecast(db_path, lear_run_id, lear_forecast)
        logger.info("  LEAR forecast persisted to DuckDB: %s", lear_run_id)
    except Exception as db_exc:
        logger.warning("  LEAR DuckDB persistence failed: %s", db_exc)
    return lear_run_id


def _maybe_run_lear_backtest(lear: Any, lear_run_id: str | None, logger: logging.Logger) -> None:
    backtest_mode = os.getenv("PFC_LEAR_BACKTEST_MODE", "full").strip().lower()
    if backtest_mode not in {"full", "fast", "skip"}:
        logger.warning("  Unknown PFC_LEAR_BACKTEST_MODE=%s, falling back to 'full'", backtest_mode)
        backtest_mode = "full"

    if backtest_mode == "skip":
        logger.info("  LEAR backtest skipped (PFC_LEAR_BACKTEST_MODE=skip)")
        return

    default_horizons = [1, 2, 3, 5, 7, 10] if backtest_mode == "full" else [1, 3, 7]
    default_n_days = 30 if backtest_mode == "full" else 10
    horizons_env = os.getenv("PFC_LEAR_BACKTEST_HORIZONS", "").strip()
    n_days_env = os.getenv("PFC_LEAR_BACKTEST_DAYS", "").strip()
    horizons = [int(x) for x in horizons_env.split(",") if x.strip()] if horizons_env else default_horizons
    n_days = int(n_days_env) if n_days_env else default_n_days

    logger.info(
        "  Running LEAR backtest multi-horizon: mode=%s, n_days=%d, horizons=%s",
        backtest_mode,
        n_days,
        horizons,
    )
    t_bt = time.time()
    try:
        bt_frames = []
        for horizon in horizons:
            bt_h = lear.backtest(n_days=n_days, horizon=horizon)
            bt_h["horizon"] = horizon
            bt_frames.append(bt_h)
        bt = pd.concat(bt_frames, ignore_index=True)
        bt_path = os.path.join("pfc_shaping", "output", f"lear_backtest_{pd.Timestamp.now().strftime('%Y-%m-%d')}.parquet")
        bt.to_parquet(bt_path, index=False)
        bt.to_parquet(os.path.join("pfc_shaping", "output", "lear_backtest_latest.parquet"), index=False)
        summary = (
            bt.groupby("horizon")
            .agg(
                mae=("abs_error", "mean"),
                rmse=("error", lambda series: float(np.sqrt(np.mean(np.square(series))))),
                bias=("error", "mean"),
                corr=("forecast", lambda series: float(series.corr(bt.loc[series.index, "actual"]))),
            )
            .reset_index()
        )
        summary_path = os.path.join("pfc_shaping", "output", f"lear_backtest_summary_{pd.Timestamp.now().strftime('%Y-%m-%d')}.csv")
        summary.to_csv(summary_path, index=False)
        summary.to_csv(os.path.join("pfc_shaping", "output", "lear_backtest_summary_latest.csv"), index=False)
        try:
            db_path = init_db(os.path.join("pfc_shaping", "data", "pfc_local.duckdb"))
            if lear_run_id is not None:
                upsert_lear_backtest(db_path, lear_run_id, bt)
                logger.info("  LEAR backtest persisted to DuckDB: %s", lear_run_id)
        except Exception as db_exc:
            logger.warning("  LEAR backtest DuckDB persistence failed: %s", db_exc)
        logger.info("  Backtest saved: %s (%.1fs)", bt_path, time.time() - t_bt)
    except Exception as bt_exc:
        logger.warning("  Backtest failed: %s", bt_exc)
