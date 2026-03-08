"""
rolling_update.py
-----------------
Cycle de mise à jour hebdomadaire de la PFC 15min N+3 ans.

Workflow (exécuté chaque lundi matin 06h00 HEC) :
    1. Ingestion nouvelles données EPEX (semaine écoulée)
    2. Ingestion nouvelles données ENTSO-E
    3. Test de rupture structurelle (Chow)
       → ajuste le lookback si nécessaire
    4. Recalibration ShapeHourly + ShapeIntraday + Uncertainty
    5. Assemblage PFC 15min sur N+3 ans
    6. Export CSV + Parquet → EULER

Entrée :
    config.yaml — paramètres de lookback, horizons, clés API, chemins

Journalisation :
    logs/rolling_update_YYYYMMDD.log

Usage :
    python -m pipeline.rolling_update
    # ou en planificateur : cron / APScheduler
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# Setup logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def setup_logging(run_date: str) -> None:
    log_file = LOG_DIR / f"rolling_update_{run_date}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_update(config: dict | None = None) -> Path:
    """
    Exécute le cycle complet de mise à jour.

    Args:
        config : dict de configuration (si None, charge config.yaml)

    Returns:
        Path du fichier CSV exporté
    """
    run_date = datetime.utcnow().strftime("%Y%m%d")
    setup_logging(run_date)
    logger.info("=== Démarrage du cycle de mise à jour PFC 15min — %s ===", run_date)

    if config is None:
        config = load_config()

    # Imports ici pour éviter les imports circulaires au niveau module
    from data.ingest_epex import fetch_and_cache, load_parquet as load_epex
    from data.ingest_entso import fetch_and_cache as fetch_entso, load_parquet as load_entso
    from data.ingest_forwards import load_base_prices
    from data.calendar_ch import enrich_15min_index
    from model.shape_hourly import ShapeHourly
    from model.shape_intraday import ShapeIntraday
    from model.uncertainty import Uncertainty
    from model.assembler import PFCAssembler
    from pipeline.structural_break import detect_chow
    from pipeline.export_euler import export_both

    paths  = config["paths"]
    params = config["model"]
    db_cfg = config.get("databricks")

    # Fenêtre de rafraîchissement : 14 derniers jours
    fetch_start = (pd.Timestamp.utcnow() - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    fetch_end   = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    # ── 1. Ingestion EPEX depuis Databricks ───────────────────────────────────
    logger.info("1/6 Ingestion EPEX Spot 15min (Databricks)...")
    try:
        epex_df = fetch_and_cache(
            start=fetch_start,
            end=fetch_end,
            parquet_path=paths["epex_parquet"],
            db_config=db_cfg,
        )
    except Exception as e:
        logger.warning("Databricks EPEX échoué (%s) — cache local utilisé", e)
        epex_df = load_epex(paths["epex_parquet"])

    # ── 2. Ingestion charge + renewables depuis Databricks ────────────────────
    logger.info("2/6 Ingestion Swissgrid load + renewables (Databricks)...")
    try:
        entso_df = fetch_entso(
            start=fetch_start,
            end=fetch_end,
            parquet_path=paths["entso_parquet"],
            db_config=db_cfg,
        )
    except Exception as e:
        logger.warning("Databricks ENTSO échoué (%s) — cache local utilisé", e)
        try:
            entso_df = load_entso(paths["entso_parquet"])
        except Exception:
            entso_df = None
            logger.warning("Pas de données exogènes disponibles — couche 2 f_Q désactivée")

    # ── 3. Détection rupture structurelle ─────────────────────────────────────
    logger.info("3/6 Test de rupture structurelle (Chow)...")
    cal_hist = enrich_15min_index(epex_df.index)
    break_result = detect_chow(
        epex_df,
        cal_hist,
        window_months=params.get("chow_window_months", 12),
        full_lookback_months=params.get("lookback_months", 36),
    )
    logger.info("Rupture : %s | %s", break_result.detected, break_result.message)
    lookback_months = break_result.recommended_lookback_months

    # ── 4. Filtrage sur la fenêtre lookback ───────────────────────────────────
    cutoff = pd.Timestamp.utcnow() - pd.DateOffset(months=lookback_months)
    epex_fit = epex_df[epex_df.index >= cutoff]
    entso_fit = entso_df[entso_df.index >= cutoff] if entso_df is not None else None
    cal_fit = enrich_15min_index(epex_fit.index)

    logger.info(
        "Fenêtre calibration : %s → %s (%d lignes EPEX)",
        epex_fit.index.min().date(), epex_fit.index.max().date(), len(epex_fit)
    )

    # ── 5. Recalibration des modèles ──────────────────────────────────────────
    logger.info("4/6 Recalibration ShapeHourly...")
    sh = ShapeHourly(sigma=params.get("gaussian_sigma", 0.5))
    sh.fit(epex_fit, cal_fit)
    sh.save(paths["model_dir"] + "/shape_hourly.parquet")

    logger.info("4/6 Recalibration ShapeIntraday...")
    si = ShapeIntraday()
    si.fit(epex_fit, entso_fit, cal_fit)
    si.save(paths["model_dir"] + "/shape_intraday.parquet")

    logger.info("4/6 Calibration Uncertainty (bootstrap)...")
    unc = Uncertainty(n_boot=params.get("n_boot", 500))
    unc.fit(epex_fit, cal_fit)
    unc.save(paths["model_dir"] + "/uncertainty.parquet")

    # ── 6. Assemblage PFC N+3 ─────────────────────────────────────────────────
    logger.info("5/6 Assemblage PFC 15min N+3 ans...")
    assembler = PFCAssembler(sh, si, unc)

    # Forwards EEX depuis Databricks (EULER) ; fallback sur config.yaml
    try:
        base_prices = load_base_prices(db_config=db_cfg)
        logger.info("Base prices chargés depuis Databricks : %d produits", len(base_prices))
    except Exception as e:
        logger.warning("Chargement forwards EEX échoué (%s) — fallback config.yaml", e)
        base_prices = config.get("base_prices_fallback", {})

    pfc_df = assembler.build(
        base_prices=base_prices,
        horizon_days=params.get("horizon_days", 3 * 365),
    )

    # ── 7. Export ──────────────────────────────────────────────────────────────
    logger.info("6/6 Export CSV + Parquet vers EULER...")
    exported = export_both(pfc_df, output_dir=paths["output_dir"], run_date=run_date)

    logger.info("=== Cycle terminé. Fichiers : %s ===", exported)
    return exported["csv"]


if __name__ == "__main__":
    run_update()
