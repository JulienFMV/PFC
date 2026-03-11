"""
rolling_update.py
-----------------
Weekly update cycle for PFC 15min with local-first execution.

Data sources (in order of priority):
  1. ENTSO-E Transparency API (entsoe.enabled: true) — for EPEX prices, load, generation
  2. Databricks (databricks.enabled: true) — for forwards and hydro
  3. Local Parquet cache — fallback offline
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def setup_logging(run_date: str) -> None:
    log_file = LOG_DIR / f"rolling_update_{run_date}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_config_path(value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (CONFIG_PATH.parent / p).resolve()


def _load_or_none(load_fn, path: Path, label: str):
    try:
        return load_fn(path)
    except Exception as e:
        logger.warning("Local %s unavailable (%s)", label, e)
        return None


def run_update(config: dict | None = None) -> Path:
    run_date = datetime.utcnow().strftime("%Y%m%d")
    setup_logging(run_date)
    logger.info("=== Start PFC 15min update - %s ===", run_date)

    if config is None:
        config = load_config()

    from pfc_shaping.calibration.arbitrage_free import ArbitrageFreeCalibrator
    from pfc_shaping.calibration.cascading import ContractCascader
    from pfc_shaping.data.calendar_ch import enrich_15min_index
    from pfc_shaping.data.ingest_entso import fetch_and_cache as fetch_entso
    from pfc_shaping.data.ingest_entso import load_parquet as load_entso
    from pfc_shaping.data.ingest_epex import fetch_and_cache as fetch_epex
    from pfc_shaping.data.ingest_epex import load_parquet as load_epex
    from pfc_shaping.data.ingest_forwards import load_base_prices
    from pfc_shaping.data.ingest_hydro import fetch_and_cache as fetch_hydro
    from pfc_shaping.data.ingest_hydro import load_parquet as load_hydro
    from pfc_shaping.model.assembler import PFCAssembler
    from pfc_shaping.model.shape_hourly import ShapeHourly
    from pfc_shaping.model.shape_intraday import ShapeIntraday
    from pfc_shaping.model.uncertainty import Uncertainty
    from pfc_shaping.model.water_value import WaterValueCorrection
    from pfc_shaping.pipeline.export_euler import export_both
    from pfc_shaping.pipeline.structural_break import detect_chow

    paths = config["paths"]
    params = config["model"]
    db_cfg = config.get("databricks", {})
    entsoe_cfg = config.get("entsoe", {})

    databricks_enabled = bool(db_cfg.get("enabled", False))
    entsoe_enabled = bool(entsoe_cfg.get("enabled", False))
    country_code = entsoe_cfg.get("country_code", "CH")

    epex_parquet_path = _resolve_config_path(paths["epex_parquet"])
    entso_parquet_path = _resolve_config_path(paths["entso_parquet"])
    hydro_parquet_path = _resolve_config_path(paths.get("hydro_parquet", "data/hydro_reservoir.parquet"))
    model_dir_path = _resolve_config_path(paths["model_dir"])
    output_dir_path = _resolve_config_path(paths["output_dir"])

    fetch_start = (pd.Timestamp.utcnow() - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    fetch_end = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    logger.info("ENTSO-E API enabled: %s | Databricks enabled: %s", entsoe_enabled, databricks_enabled)

    # 1) EPEX — priorité : ENTSO-E API > Databricks > cache local
    logger.info("1/8 Ingestion EPEX 15min")
    if entsoe_enabled:
        try:
            epex_df = fetch_epex(fetch_start, fetch_end, country_code=country_code, parquet_path=epex_parquet_path)
        except Exception as e:
            logger.warning("ENTSO-E API EPEX failed (%s) - fallback local cache", e)
            epex_df = load_epex(epex_parquet_path)
    elif databricks_enabled:
        try:
            from pfc_shaping.data.databricks_client import query_to_df, table_fqn
            epex_df = fetch_epex(fetch_start, fetch_end, parquet_path=epex_parquet_path)
        except Exception as e:
            logger.warning("Databricks EPEX failed (%s) - local cache", e)
            epex_df = load_epex(epex_parquet_path)
    else:
        epex_df = load_epex(epex_parquet_path)

    # 2) ENTSO (load + renewables) — priorité : ENTSO-E API > Databricks > cache local
    logger.info("2/8 Ingestion Swissgrid+renewables")
    if entsoe_enabled:
        try:
            entso_df = fetch_entso(fetch_start, fetch_end, country_code=country_code, parquet_path=entso_parquet_path)
        except Exception as e:
            logger.warning("ENTSO-E API load/gen failed (%s) - fallback local cache", e)
            entso_df = _load_or_none(load_entso, entso_parquet_path, "ENTSO parquet")
    elif databricks_enabled:
        try:
            entso_df = fetch_entso(fetch_start, fetch_end, parquet_path=entso_parquet_path)
        except Exception as e:
            logger.warning("Databricks ENTSO failed (%s) - local cache", e)
            entso_df = _load_or_none(load_entso, entso_parquet_path, "ENTSO parquet")
    else:
        entso_df = _load_or_none(load_entso, entso_parquet_path, "ENTSO parquet")

    if entso_df is None:
        logger.warning("No exogenous ENTSO data available - f_Q correction disabled")

    # 3) Hydro — Databricks ou cache local (pas dispo via ENTSO-E standard)
    logger.info("3/8 Ingestion hydro reservoir levels")
    if databricks_enabled:
        try:
            hydro_df = fetch_hydro(fetch_start, fetch_end, parquet_path=hydro_parquet_path, db_config=db_cfg)
        except Exception as e:
            logger.warning("Databricks hydro failed (%s) - local cache", e)
            hydro_df = _load_or_none(load_hydro, hydro_parquet_path, "hydro parquet")
    else:
        hydro_df = _load_or_none(load_hydro, hydro_parquet_path, "hydro parquet")

    # 4) Structural break
    logger.info("4/8 Structural break detection")
    cal_hist = enrich_15min_index(epex_df.index)
    break_result = detect_chow(
        epex_df,
        cal_hist,
        window_months=params.get("chow_window_months", 12),
        full_lookback_months=params.get("lookback_months", 36),
    )
    lookback_months = break_result.recommended_lookback_months
    logger.info("Break detected=%s | %s", break_result.detected, break_result.message)

    # 5) Fit window
    cutoff = pd.Timestamp.utcnow() - pd.DateOffset(months=lookback_months)
    epex_fit = epex_df[epex_df.index >= cutoff]
    entso_fit = entso_df[entso_df.index >= cutoff] if entso_df is not None else None
    hydro_fit = hydro_df[hydro_df.index >= cutoff] if hydro_df is not None else None
    cal_fit = enrich_15min_index(epex_fit.index)

    logger.info("Calibration window: %s -> %s (%d rows)", epex_fit.index.min().date(), epex_fit.index.max().date(), len(epex_fit))

    # 6) Models
    logger.info("5/8 Calibrate ShapeHourly")
    sh = ShapeHourly(sigma=params.get("gaussian_sigma", 0.5)).fit(epex_fit, cal_fit)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    sh.save(model_dir_path / "shape_hourly.parquet")

    logger.info("5/8 Calibrate ShapeIntraday")
    si = ShapeIntraday().fit(epex_fit, entso_fit, cal_fit)
    si.save(model_dir_path / "shape_intraday.parquet")

    logger.info("5/8 Calibrate Uncertainty")
    unc = Uncertainty(n_boot=params.get("n_boot", 500)).fit(epex_fit, cal_fit)
    unc.save(model_dir_path / "uncertainty.parquet")

    logger.info("5/8 Calibrate WaterValue")
    wv = WaterValueCorrection()
    if hydro_fit is not None:
        wv.fit(epex_fit, hydro_fit, cal_fit)
        wv.save(model_dir_path / "water_value.parquet")
    else:
        logger.info("No hydro dataset available - WaterValue defaults")

    # 7) Cascading + arbitrage-free calibrator
    logger.info("6/8 Prepare cascading and arbitrage-free calibration")
    cascader = ContractCascader()
    cascader.fit_seasonal_ratios(epex_fit)

    calibrator = ArbitrageFreeCalibrator(
        smoothness_weight=params.get("smoothness_weight", 1.0),
        tol=params.get("calibration_tol", 0.01),
    )

    # 8) Assemble
    logger.info("7/8 Assemble PFC 15min")
    assembler = PFCAssembler(
        shape_hourly=sh,
        shape_intraday=si,
        uncertainty=unc,
        water_value=wv,
        cascader=cascader,
        calibrator=calibrator,
    )

    if databricks_enabled:
        try:
            base_prices = load_base_prices(db_config=db_cfg)
            logger.info("Base prices loaded from Databricks (%d products)", len(base_prices))
        except Exception as e:
            logger.warning("Databricks forwards failed (%s) - fallback config", e)
            base_prices = config.get("base_prices_fallback", {})
    else:
        base_prices = config.get("base_prices_fallback", {})

    pfc_df = assembler.build(
        base_prices=base_prices,
        horizon_days=params.get("horizon_days", 3 * 365),
        hydro_forecast=hydro_fit,
    )

    logger.info("8/8 Export CSV + Parquet")
    exported = export_both(pfc_df, output_dir=output_dir_path, run_date=run_date)

    logger.info("=== Update done. Files: %s ===", exported)
    return exported["csv"]


if __name__ == "__main__":
    run_update()
