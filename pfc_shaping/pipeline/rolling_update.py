"""
rolling_update.py
-----------------
Daily update cycle for PFC 15min with local-first execution.
"""

from __future__ import annotations

import logging
import os
import sys
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOCK_DIR = Path(__file__).parent.parent / "run_locks"
LOCK_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def setup_logging(run_date: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

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


def _first_existing_path(*candidates: str | Path | None) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        p = Path(candidate)
        if p.exists():
            return p
    return None


def _load_or_none(load_fn, path: Path, label: str):
    try:
        return load_fn(path)
    except Exception as e:
        logger.warning("Local %s unavailable (%s)", label, e)
        return None


def _configure_ssl(config: dict) -> None:
    ssl_cfg = config.get("ssl", {})
    ca_bundle = ssl_cfg.get("ca_bundle")
    if ca_bundle:
        os.environ["REQUESTS_CA_BUNDLE"] = str(ca_bundle)
        logger.info("REQUESTS_CA_BUNDLE configured: %s", ca_bundle)


def _warn_missing_credentials(config: dict) -> None:
    db_cfg = config.get("databricks", {})
    if db_cfg.get("enabled", False):
        host = db_cfg.get("host") or os.getenv("DATABRICKS_HOST")
        http_path = db_cfg.get("http_path") or os.getenv("DATABRICKS_HTTP_PATH")
        token = db_cfg.get("token") or os.getenv("DATABRICKS_TOKEN")
        if not (host and http_path and token):
            logger.warning(
                "Databricks enabled but credentials are incomplete. "
                "Expected DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN."
            )

    entsoe_cfg = config.get("entsoe", {})
    if entsoe_cfg.get("enabled", False) and not os.getenv("ENTSOE_API_KEY"):
        logger.warning("ENTSO-E enabled but ENTSOE_API_KEY is not set.")


@contextmanager
def _run_lock(lock_name: str):
    lock_path = LOCK_DIR / f"{lock_name}.lock"
    if lock_path.exists():
        raise RuntimeError(f"Another run is active (lock exists: {lock_path})")
    lock_path.write_text(datetime.utcnow().isoformat(), encoding="utf-8")
    try:
        yield
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass


def _write_run_report(output_dir: Path, run_id: str, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"run_report_{run_id}.json"
    report_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return report_path


def run_update(config: dict | None = None) -> Path:
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

    run_date = datetime.utcnow().strftime("%Y%m%d")
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    setup_logging(run_date)
    logger.info("=== Start PFC 15min update - run_id=%s date=%s ===", run_id, run_date)

    if config is None:
        config = load_config()
    _configure_ssl(config)
    _warn_missing_credentials(config)
    teams_webhook = os.getenv("TEAMS_WEBHOOK_URL")

    from pfc_shaping.calibration.arbitrage_free import ArbitrageFreeCalibrator
    from pfc_shaping.calibration.cascading import ContractCascader
    from pfc_shaping.data.calendar_ch import enrich_15min_index
    from pfc_shaping.data.ingest_energy_charts import (
        fetch_and_cache_power as fetch_ec_power,
        fetch_and_cache_prices as fetch_ec_prices,
        load_epex_parquet as load_epex,
        load_power_parquet as load_entso,
    )
    from pfc_shaping.data.ingest_entso import fetch_and_cache as fetch_entso_api
    from pfc_shaping.data.ingest_epex import fetch_and_cache as fetch_epex_entsoe
    from pfc_shaping.data.ingest_forwards import (
        load_base_prices,
        load_base_prices_from_eex_report,
    )
    from pfc_shaping.data.ingest_hydro import fetch_and_cache as fetch_hydro
    from pfc_shaping.data.ingest_hydro import load_parquet as load_hydro
    from pfc_shaping.data.ingest_smard import fetch_and_cache as fetch_epex_smard
    from pfc_shaping.model.assembler import PFCAssembler
    from pfc_shaping.model.shape_hourly import ShapeHourly
    from pfc_shaping.model.shape_intraday import ShapeIntraday
    from pfc_shaping.model.uncertainty import Uncertainty
    from pfc_shaping.model.water_value import WaterValueCorrection
    from pfc_shaping.pipeline.export_euler import export_both
    from pfc_shaping.pipeline.quality_gate import (
        QualityGateError,
        validate_input_frame,
        validate_pfc_output,
    )
    from pfc_shaping.pipeline.structural_break import detect_chow
    from pfc_shaping.storage.local_duckdb import (
        benchmark_against_hfc,
        init_db,
        upsert_run_and_forecast,
    )
    from pfc_shaping.tools.notify_teams import send_teams_alert

    try:
        with _run_lock("rolling_update"):
            paths = config["paths"]
            params = config["model"]
            db_cfg = config.get("databricks", {})
            entsoe_cfg = config.get("entsoe", {})
        forwards_cfg = config.get("forwards", {})
        quality_cfg = config.get("quality", {})
        markets_cfg = forwards_cfg.get("eex_markets")
        if isinstance(markets_cfg, list) and markets_cfg:
            markets = [str(m).upper() for m in markets_cfg]
        else:
            markets = [str(forwards_cfg.get("eex_market", "CH")).upper()]
        markets = [m for m in markets if m in {"CH", "DE"}]
        if not markets:
            raise RuntimeError("No valid forwards markets configured. Use CH/DE.")

        databricks_enabled = bool(db_cfg.get("enabled", False))
        entsoe_enabled = bool(entsoe_cfg.get("enabled", False))
        smard_cfg = config.get("smard", {})
        smard_enabled = bool(smard_cfg.get("enabled", True))
        ec_cfg = config.get("energy_charts", {})
        ec_enabled = bool(ec_cfg.get("enabled", True))
        country_code = entsoe_cfg.get("country_code", "CH")

        epex_parquet_path = _resolve_config_path(paths["epex_parquet"])
        entso_parquet_path = _resolve_config_path(paths["entso_parquet"])
        hydro_parquet_path = _resolve_config_path(paths.get("hydro_parquet", "data/hydro_reservoir.parquet"))
        model_dir_path = _resolve_config_path(paths["model_dir"])
        output_dir_path = _resolve_config_path(paths["output_dir"])
        duckdb_path = _resolve_config_path(paths.get("duckdb_path", "data/pfc_local.duckdb"))
        hfc_dir = Path(paths.get("hfc_benchmark_dir", ""))

        fetch_start = (pd.Timestamp.utcnow() - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        fetch_end = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

        logger.info(
            "Sources - energy-charts: %s | SMARD: %s | ENTSO-E: %s | Databricks: %s",
            ec_enabled,
            smard_enabled,
            entsoe_enabled,
            databricks_enabled,
        )

        logger.info("1/8 Ingestion EPEX 15min")
        epex_df = None
        if ec_enabled:
            try:
                epex_df = fetch_ec_prices(fetch_start, fetch_end, bzn="CH", parquet_path=epex_parquet_path)
                logger.info("EPEX loaded from energy-charts (%d rows)", len(epex_df))
            except Exception as e:
                logger.warning("energy-charts prices failed (%s) - fallback SMARD", e)

        if epex_df is None and smard_enabled:
            try:
                epex_df = fetch_epex_smard(fetch_start, fetch_end, country_code=country_code, parquet_path=epex_parquet_path)
                logger.info("EPEX loaded from SMARD (%d rows)", len(epex_df))
            except Exception as e:
                logger.warning("SMARD EPEX failed (%s) - fallback ENTSO-E", e)

        if epex_df is None and entsoe_enabled:
            try:
                epex_df = fetch_epex_entsoe(fetch_start, fetch_end, country_code=country_code, parquet_path=epex_parquet_path)
                logger.info("EPEX loaded from ENTSO-E (%d rows)", len(epex_df))
            except Exception as e:
                logger.warning("ENTSO-E EPEX failed (%s) - fallback local cache", e)

        if epex_df is None:
            epex_df = load_epex(epex_parquet_path)

        try:
            rep = validate_input_frame(
                epex_df,
                name="EPEX",
                required_columns=["price_eur_mwh"],
                min_rows=96 * 5,
                max_age_days=5,
            )
            for w in rep.warnings:
                logger.warning(w)
        except QualityGateError as e:
            raise RuntimeError(f"Quality gate failed on EPEX: {e}") from e

        logger.info("2/8 Ingestion load + renewables CH")
        entso_df = None
        if ec_enabled:
            try:
                entso_df = fetch_ec_power(fetch_start, fetch_end, country="ch", parquet_path=entso_parquet_path)
                logger.info("Load/gen loaded from energy-charts (%d rows)", len(entso_df))
            except Exception as e:
                logger.warning("energy-charts power failed (%s) - fallback ENTSO-E", e)

        if entso_df is None and entsoe_enabled:
            try:
                entso_df = fetch_entso_api(fetch_start, fetch_end, country_code=country_code, parquet_path=entso_parquet_path)
                logger.info("Load/gen loaded from ENTSO-E (%d rows)", len(entso_df))
            except Exception as e:
                logger.warning("ENTSO-E load/gen failed (%s) - fallback local cache", e)

        if entso_df is None:
            entso_df = _load_or_none(load_entso, entso_parquet_path, "power parquet")

        if entso_df is None:
            logger.warning("No exogenous ENTSO data available - f_Q correction disabled")
        else:
            try:
                rep = validate_input_frame(
                    entso_df,
                    name="ENTSO",
                    required_columns=["load_mw", "solar_mw", "wind_mw"],
                    min_rows=96 * 3,
                    max_age_days=7,
                )
                for w in rep.warnings:
                    logger.warning(w)
            except QualityGateError as e:
                logger.warning("ENTSO quality gate degraded (non-blocking): %s", e)

        logger.info("3/8 Ingestion hydro reservoir levels")
        try:
            hydro_df = fetch_hydro(fetch_start, fetch_end, parquet_path=hydro_parquet_path, db_config=db_cfg)
        except Exception as e:
            logger.warning("Hydro fetch failed (%s) - local cache", e)
            hydro_df = _load_or_none(load_hydro, hydro_parquet_path, "hydro parquet")

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

        cutoff = pd.Timestamp.utcnow() - pd.DateOffset(months=lookback_months)
        epex_fit = epex_df[epex_df.index >= cutoff]
        entso_fit = entso_df[entso_df.index >= cutoff] if entso_df is not None else None
        hydro_fit = hydro_df[hydro_df.index >= cutoff] if hydro_df is not None else None
        cal_fit = enrich_15min_index(epex_fit.index)
        logger.info("Calibration window: %s -> %s (%d rows)", epex_fit.index.min().date(), epex_fit.index.max().date(), len(epex_fit))

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

        logger.info("6/8 Prepare cascading and arbitrage-free calibration")
        cascader = ContractCascader()
        cascader.fit_seasonal_ratios(epex_fit)
        calibrator = ArbitrageFreeCalibrator(
            smoothness_weight=params.get("smoothness_weight", 1.0),
            tol=params.get("calibration_tol", 0.01),
        )

        logger.info("7/8 Assemble PFC 15min")
        assembler = PFCAssembler(
            shape_hourly=sh,
            shape_intraday=si,
            uncertainty=unc,
            water_value=wv,
            cascader=cascader,
            calibrator=calibrator,
            calibration_fallback_to_raw=bool(params.get("calibration_fallback_to_raw", True)),
        )

        eex_as_of_date = forwards_cfg.get("eex_as_of_date")
        eex_unc = forwards_cfg.get("eex_report_path_unc")
        eex_primary = forwards_cfg.get("eex_report_path")
        eex_report_path = _first_existing_path(eex_unc, eex_primary)

        exported_by_market: dict[str, dict[str, Path]] = {}
        benchmark_by_market: dict[str, dict] = {}
        gate_by_market: dict[str, tuple[bool, str]] = {}
        source_by_market: dict[str, str] = {}

        for market in markets:
            logger.info("7.x Build market %s", market)
            base_prices = None
            source_forwards = "unknown"

            if eex_report_path is not None:
                try:
                    base_prices = load_base_prices_from_eex_report(
                        report_path=eex_report_path,
                        market=market,
                        as_of_date=eex_as_of_date,
                    )
                    source_forwards = "eex_xlsx"
                    logger.info("[%s] Base prices loaded from EEX XLSX (%d products)", market, len(base_prices))
                except Exception as e:
                    logger.warning("[%s] EEX XLSX forwards failed (%s)", market, e)
            else:
                logger.warning(
                    "No EEX XLSX path found (checked: %s | %s)",
                    eex_primary,
                    eex_unc,
                )

            if base_prices is None:
                if market == "CH":
                    try:
                        base_prices = load_base_prices(db_config=db_cfg)
                        source_forwards = "databricks"
                        logger.info("[%s] Base prices loaded from Databricks (%d products)", market, len(base_prices))
                    except Exception as e:
                        logger.warning("[%s] Databricks forwards failed (%s) - fallback config.yaml", market, e)
                        base_prices = config.get("base_prices_fallback", {})
                        source_forwards = "config_fallback"
                else:
                    raise RuntimeError(f"[{market}] Missing EEX forwards from XLSX; DE requires dedicated EEX sheet.")

            pfc_df = assembler.build(
                base_prices=base_prices,
                horizon_days=params.get("horizon_days", 3 * 365),
                hydro_forecast=hydro_fit,
            )

            expected_rows = int(params.get("horizon_days", 3 * 365) * 96 * 0.95)
            try:
                rep = validate_pfc_output(pfc_df, expected_min_rows=expected_rows)
                for w in rep.warnings:
                    logger.warning("[%s] %s", market, w)
            except QualityGateError as e:
                raise RuntimeError(f"[{market}] Quality gate failed on PFC output: {e}") from e

            filename_base = (
                f"pfc_15min_{run_id}"
                if market == "CH"
                else f"pfc_{market.lower()}_15min_{run_id}"
            )
            exported = export_both(
                pfc_df,
                output_dir=output_dir_path,
                filename_base=filename_base,
            )
            exported_by_market[market] = exported
            source_by_market[market] = source_forwards

            market_run_id = f"{run_id}_{market}"
            init_db(duckdb_path)
            upsert_run_and_forecast(
                db_path=duckdb_path,
                run_id=market_run_id,
                pfc_csv_path=exported["csv"],
                pfc_parquet_path=exported["parquet"],
                source_forwards=f"{source_forwards}:{market}",
            )

            benchmark_metrics: dict | None = None
            benchmark_gate_ok = True
            benchmark_gate_reason = "benchmark not evaluated for this market"
            if market == "CH":
                try:
                    hfc_files = sorted(hfc_dir.glob("HFC_Ompex_*.xlsx")) if hfc_dir.exists() else []
                    if hfc_files:
                        benchmark_metrics = benchmark_against_hfc(
                            db_path=duckdb_path,
                            run_id=market_run_id,
                            pfc_csv_path=exported["csv"],
                            hfc_file=hfc_files[-1],
                        )
                        mae_limit = float(quality_cfg.get("max_mae_eur_mwh", 20.0))
                        rmse_limit = float(quality_cfg.get("max_rmse_eur_mwh", 26.0))
                        bias_limit = float(quality_cfg.get("max_abs_bias_eur_mwh", 5.0))
                        benchmark_gate_ok = (
                            benchmark_metrics["mae"] <= mae_limit
                            and benchmark_metrics["rmse"] <= rmse_limit
                            and abs(benchmark_metrics["bias"]) <= bias_limit
                        )
                        benchmark_gate_reason = (
                            f"mae={benchmark_metrics['mae']:.2f}/{mae_limit:.2f}, "
                            f"rmse={benchmark_metrics['rmse']:.2f}/{rmse_limit:.2f}, "
                            f"|bias|={abs(benchmark_metrics['bias']):.2f}/{bias_limit:.2f}"
                        )
                        logger.info(
                            "[%s] Benchmark PFC vs HFC saved: n=%d MAE=%.4f RMSE=%.4f",
                            market,
                            benchmark_metrics["n_points"],
                            benchmark_metrics["mae"],
                            benchmark_metrics["rmse"],
                        )
                        if not benchmark_gate_ok:
                            logger.warning("[%s] Benchmark gate failed: %s", market, benchmark_gate_reason)
                    else:
                        logger.warning("[%s] No HFC benchmark file found in %s", market, hfc_dir)
                        benchmark_gate_reason = "no HFC file found"
                except Exception as e:
                    logger.warning("[%s] HFC benchmark persistence failed (%s)", market, e)
                    benchmark_gate_reason = f"benchmark failed: {e}"

            benchmark_by_market[market] = benchmark_metrics or {}
            gate_by_market[market] = (benchmark_gate_ok, benchmark_gate_reason)

            report_payload = {
                "run_id": market_run_id,
                "run_date": run_date,
                "market": market,
                "source_forwards": source_forwards,
                "output_csv": str(exported["csv"]),
                "output_parquet": str(exported["parquet"]),
                "row_count": int(len(pfc_df)),
                "calibrated": bool(pfc_df["calibrated"].any()) if "calibrated" in pfc_df.columns else False,
                "quality": {
                    "benchmark_gate_enabled": bool(quality_cfg.get("fail_on_benchmark", False) and market == "CH"),
                    "benchmark_gate_ok": bool(benchmark_gate_ok),
                    "benchmark_gate_reason": benchmark_gate_reason,
                },
                "benchmark": benchmark_metrics or {},
            }
            report_path = _write_run_report(output_dir_path, market_run_id, report_payload)
            logger.info("[%s] Run report written: %s", market, report_path)

            if market == "CH" and bool(quality_cfg.get("fail_on_benchmark", False)) and not benchmark_gate_ok:
                raise RuntimeError(f"[{market}] Benchmark gate failed in strict mode: {benchmark_gate_reason}")

        ch_csv = exported_by_market.get("CH", {}).get("csv")
        any_csv = next(iter(exported_by_market.values()))["csv"]
        success_csv = ch_csv if ch_csv is not None else any_csv

        logger.info(
            "=== Update done. run_id=%s markets=%s ===",
            run_id,
            ",".join(markets),
        )
        send_teams_alert(
            teams_webhook,
            title=f"PFC run SUCCESS [{run_id}]",
            facts={
                "run_id": run_id,
                "markets": ",".join(markets),
                "CH_benchmark": gate_by_market.get("CH", (True, "n/a"))[1],
                "CH_csv": exported_by_market.get("CH", {}).get("csv", "-"),
                "DE_csv": exported_by_market.get("DE", {}).get("csv", "-"),
            },
            ok=True,
        )
        return success_csv
    except Exception as exc:
        send_teams_alert(
            teams_webhook,
            title=f"PFC run FAILED [{run_id}]",
            facts={
                "run_id": run_id,
                "error": str(exc),
            },
            ok=False,
        )
        raise


if __name__ == "__main__":
    run_update()
