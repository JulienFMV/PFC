#!/usr/bin/env python3
"""
run_daily.py — PFC Daily Production Runner
===========================================
Orchestre le pipeline PFC quotidien :

  1. Ingestion fraîche (EPEX spot J-1, ENTSO-E load/gen/outages, hydro)
  2. Réentraînement MLP rolling (fenêtre glissante 12 mois)
  3. Build PFC CH + DE (calibrée sur forwards EEX du jour)
  4. Quality gates (RMSE, MAE, bias vs seuils)
  5. Export (Parquet + CSV)

Usage:
    python3 scripts/run_daily.py                    # run complet
    python3 scripts/run_daily.py --skip-ingest      # skip data refresh
    python3 scripts/run_daily.py --dry-run          # ingestion seule, pas de PFC

Scheduling (crontab):
    0 7 * * 1-5  cd /path/to/PFC && python3 scripts/run_daily.py >> logs/daily_$(date +\\%Y\\%m\\%d).log 2>&1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

# ── Setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PFC_DAILY")

# ── Ensure logs dir exists ───────────────────────────────────────────────
(ROOT / "logs").mkdir(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="PFC Daily Production Runner")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip data ingestion, use cached data")
    parser.add_argument("--dry-run", action="store_true",
                        help="Ingest data only, don't build PFC")
    parser.add_argument("--days-back", type=int, default=7,
                        help="Days of history to refresh (default: 7)")
    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════
# STEP 1: DATA INGESTION
# ═════════════════════════════════════════════════════════════════════════

def ingest_data(days_back: int = 7) -> dict:
    """
    Rafraîchit toutes les sources de données.
    Retourne un dict de statuts.
    """
    today = date.today()
    start = (today - timedelta(days=days_back)).isoformat()
    end = (today + timedelta(days=1)).isoformat()

    status = {}

    # ── 1a. EPEX spot CH + DE (energy-charts) ──
    logger.info("=" * 60)
    logger.info("INGEST: EPEX spot CH + DE (%s → %s)", start, end)
    logger.info("=" * 60)
    try:
        from pfc_shaping.data.ingest_energy_charts import (
            fetch_and_cache_prices,
            fetch_and_cache_prices_de,
        )
        df_ch = fetch_and_cache_prices(start, end)
        df_de = fetch_and_cache_prices_de(start, end)
        status["epex"] = f"OK ({len(df_ch)} CH, {len(df_de)} DE rows)"
        logger.info("  EPEX: %s", status["epex"])
    except Exception as e:
        status["epex"] = f"FAILED: {e}"
        logger.error("  EPEX ingestion failed: %s", e)

    # ── 1b. Load + generation (energy-charts primary, ENTSO-E fallback) ──
    logger.info("=" * 60)
    logger.info("INGEST: Load/gen (%s → %s)", start, end)
    logger.info("=" * 60)
    try:
        from pfc_shaping.data.ingest_energy_charts import fetch_and_cache_power
        fetch_and_cache_power(start, end)
        status["load_gen"] = "OK (energy-charts)"
        logger.info("  Load/gen: OK (energy-charts)")
    except Exception as e:
        logger.warning("  energy-charts load/gen failed, trying ENTSO-E fallback: %s", e)
        try:
            from pfc_shaping.data.ingest_entso import fetch_and_cache
            fetch_and_cache(start, end, country_code="CH")
            status["load_gen"] = "OK (ENTSO-E fallback)"
            logger.info("  Load/gen: OK (ENTSO-E fallback)")
        except Exception as e2:
            status["load_gen"] = f"FAILED: {e2}"
            logger.warning("  Load/gen ingestion failed (non-critical): %s", e2)

    # ── 1c. ENTSO-E outages (REMIT UMM) ──
    logger.info("=" * 60)
    logger.info("INGEST: ENTSO-E outages/REMIT (%s → %s)", start, end)
    logger.info("=" * 60)
    try:
        from pfc_shaping.data.ingest_outages import fetch_and_cache as fetch_outages
        # Outages: look ahead 90 days for planned maintenance
        outage_end = (today + timedelta(days=90)).isoformat()
        fetch_outages(start, outage_end, country_code="CH")
        status["outages"] = "OK"
        logger.info("  Outages: OK")
    except Exception as e:
        status["outages"] = f"FAILED: {e}"
        logger.warning("  Outages ingestion failed (non-critical): %s", e)

    # ── 1d. Hydro reservoir levels (SFOE/BFE) ──
    logger.info("=" * 60)
    logger.info("INGEST: Hydro reservoir levels")
    logger.info("=" * 60)
    try:
        from pfc_shaping.data.ingest_hydro import fetch_and_cache as fetch_hydro
        fetch_hydro(start, end)
        status["hydro"] = "OK"
        logger.info("  Hydro: OK")
    except Exception as e:
        status["hydro"] = f"FAILED: {e}"
        logger.warning("  Hydro ingestion failed (non-critical): %s", e)

    return status


# ═════════════════════════════════════════════════════════════════════════
# STEP 2: RUN PFC PRODUCTION
# ═════════════════════════════════════════════════════════════════════════

def run_pfc_production() -> bool:
    """
    Lance run_pfc_production.py comme sous-processus.
    Retourne True si succès.
    """
    import subprocess

    logger.info("=" * 60)
    logger.info("BUILD: Running PFC production pipeline")
    logger.info("=" * 60)

    result = subprocess.run(
        [sys.executable, str(ROOT / "run_pfc_production.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=600,  # 10 min max
    )

    # Log stdout/stderr
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            logger.info("  [PROD] %s", line)
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            if "ERROR" in line or "FAIL" in line:
                logger.error("  [PROD] %s", line)
            else:
                logger.info("  [PROD] %s", line)

    if result.returncode != 0:
        logger.error("PFC production FAILED (exit code %d)", result.returncode)
        return False

    logger.info("PFC production completed successfully")
    return True


# ═════════════════════════════════════════════════════════════════════════
# STEP 3: QUALITY GATES
# ═════════════════════════════════════════════════════════════════════════

def check_quality_gates() -> bool:
    """
    Vérifie que la PFC du jour passe les quality gates.
    Compare avec le spot réalisé des derniers jours disponibles.
    """
    import yaml

    logger.info("=" * 60)
    logger.info("QUALITY: Checking quality gates")
    logger.info("=" * 60)

    cfg_path = ROOT / "pfc_shaping" / "config.yaml"
    with cfg_path.open() as f:
        config = yaml.safe_load(f)

    quality = config.get("quality", {})
    max_mae = quality.get("max_mae_eur_mwh", 20.0)
    max_rmse = quality.get("max_rmse_eur_mwh", 26.0)
    max_bias = quality.get("max_abs_bias_eur_mwh", 5.0)

    # Find today's PFC output
    import pandas as pd
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    pfc_path = ROOT / "pfc_shaping" / "output" / f"pfc_15min_{today_str}.parquet"

    if not pfc_path.exists():
        logger.warning("PFC output not found: %s", pfc_path)
        return False

    pfc = pd.read_parquet(pfc_path)

    # Sanity checks
    checks_passed = True

    # Check 1: No NaN in price_shape
    n_nan = pfc["price_shape"].isna().sum()
    if n_nan > 0:
        logger.error("FAIL: %d NaN in price_shape", n_nan)
        checks_passed = False
    else:
        logger.info("  price_shape NaN: 0 ✓")

    # Check 2: Price range reasonable
    p_min = pfc["price_shape"].min()
    p_max = pfc["price_shape"].max()
    if p_min < -50:
        logger.warning("  price_shape min=%.2f (< -50, unusual)", p_min)
    if p_max > 500:
        logger.warning("  price_shape max=%.2f (> 500, unusual)", p_max)
    logger.info("  price range: [%.2f, %.2f] EUR/MWh", p_min, p_max)

    # Check 3: Factor ranges
    for col in ["f_S", "f_W", "f_H", "f_Q"]:
        if col in pfc.columns:
            f_min, f_max = pfc[col].min(), pfc[col].max()
            if f_min < 0.1 or f_max > 5.0:
                logger.warning("  %s range [%.3f, %.3f] — outside expected bounds", col, f_min, f_max)
            else:
                logger.info("  %s range: [%.3f, %.3f] ✓", col, f_min, f_max)

    # Check 4: IC width (p90 - p10 should be positive)
    if "p10" in pfc.columns and "p90" in pfc.columns:
        ic_width = pfc["p90"] - pfc["p10"]
        if (ic_width < 0).any():
            logger.error("FAIL: %d timestamps with p90 < p10", (ic_width < 0).sum())
            checks_passed = False
        else:
            logger.info("  IC width: all positive ✓ (mean=%.2f)", ic_width.mean())

    # Check 5: Compare with recent spot (if available)
    epex_path = ROOT / "pfc_shaping" / "data" / "epex_15min.parquet"
    if epex_path.exists():
        import numpy as np
        epex = pd.read_parquet(epex_path)
        common = pfc.index.intersection(epex.index)
        if len(common) > 96:  # at least 1 day overlap
            pfc_spot = pfc.loc[common, "price_shape"]
            actual = epex.loc[common, "price_eur_mwh"]
            errors = pfc_spot.values - actual.values
            rmse = float(np.sqrt(np.mean(errors**2)))
            mae = float(np.mean(np.abs(errors)))
            bias = float(np.mean(errors))

            logger.info("  Backtest vs spot (%d pts):", len(common))
            logger.info("    RMSE: %.2f (max: %.1f) %s", rmse, max_rmse,
                        "✓" if rmse <= max_rmse else "✗ FAIL")
            logger.info("    MAE:  %.2f (max: %.1f) %s", mae, max_mae,
                        "✓" if mae <= max_mae else "✗ FAIL")
            logger.info("    Bias: %+.2f (max: ±%.1f) %s", bias, max_bias,
                        "✓" if abs(bias) <= max_bias else "✗ FAIL")

            if rmse > max_rmse or mae > max_mae or abs(bias) > max_bias:
                checks_passed = False

    return checks_passed


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    t0 = time.time()

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║           PFC DAILY PRODUCTION RUN                      ║")
    logger.info("║           %s                                    ║", date.today().isoformat())
    logger.info("╚══════════════════════════════════════════════════════════╝")

    # STEP 1: Ingest
    if not args.skip_ingest:
        ingest_status = ingest_data(days_back=args.days_back)
        logger.info("")
        logger.info("INGEST SUMMARY:")
        for source, status in ingest_status.items():
            logger.info("  %-10s: %s", source, status)

        if "FAILED" in ingest_status.get("epex", ""):
            logger.error("EPEX ingestion failed — cannot build PFC without prices")
            sys.exit(1)
    else:
        logger.info("Skipping data ingestion (--skip-ingest)")

    if args.dry_run:
        logger.info("Dry run — stopping after ingestion")
        sys.exit(0)

    # STEP 2: Build PFC
    success = run_pfc_production()
    if not success:
        logger.error("PFC production failed")
        sys.exit(1)

    # STEP 3: Quality gates
    quality_ok = check_quality_gates()

    elapsed = time.time() - t0
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║  DAILY RUN COMPLETE                                     ║")
    logger.info("║  Quality: %-44s ║", "PASS ✓" if quality_ok else "WARNINGS ⚠")
    logger.info("║  Time:    %-44s ║", f"{elapsed:.0f}s")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    sys.exit(0 if quality_ok else 2)


if __name__ == "__main__":
    main()
