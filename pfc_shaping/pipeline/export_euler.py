"""
export_euler.py
---------------
Export de la PFC 15min vers EULER (Phinergy ETRM).

Formats supportés :
    - CSV  : format plat, lisible directement par EULER
    - Parquet : format interne FMV (archivage + reporting)

Format CSV de sortie (à valider avec Phinergy) :
    timestamp_local;price_shape;p10;p90;profile_type;confidence
    2025-03-10 00:00;65.42;59.10;71.80;M+1..M+6;1.0
    2025-03-10 00:15;64.18;57.90;70.50;M+1..M+6;1.0
    ...

Convention :
    - Timestamps en heure locale Europe/Zurich (CET/CEST)
    - Prix en €/MWh
    - Séparateur point-virgule (standard Phinergy)
    - Encodage UTF-8

Usage :
    from pipeline.export_euler import export_csv, export_parquet
    export_csv(pfc_df, "output/pfc_20250310.csv")
    export_parquet(pfc_df, "output/pfc_20250310.parquet")
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Colonnes exportées vers EULER (dans cet ordre)
EULER_COLUMNS = ["price_shape", "p10", "p90", "profile_type", "confidence"]
INTERNAL_COLUMNS = EULER_COLUMNS + ["calibrated"]


def export_csv(
    pfc_df: pd.DataFrame,
    path: str | Path,
    tz_local: str = "Europe/Zurich",
) -> Path:
    """
    Exporte la PFC 15min au format CSV pour EULER.

    Args:
        pfc_df   : DataFrame PFC (index UTC, colonnes EULER_COLUMNS)
        path     : chemin de sortie
        tz_local : timezone locale pour les timestamps (défaut Europe/Zurich)

    Returns:
        Path du fichier créé
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = _prepare_export(pfc_df, tz_local, include_internal=False)

    df.to_csv(
        path,
        sep=";",
        decimal=".",
        float_format="%.4f",
        encoding="utf-8",
        date_format="%Y-%m-%d %H:%M",
    )

    logger.info(
        "Export CSV EULER : %s (%d lignes, %s → %s)",
        path,
        len(df),
        df.index.min(),
        df.index.max(),
    )
    return path


def export_parquet(
    pfc_df: pd.DataFrame,
    path: str | Path,
    tz_local: str = "Europe/Zurich",
) -> Path:
    """
    Exporte la PFC 15min en Parquet (archivage interne FMV).

    Returns:
        Path du fichier créé
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = _prepare_export(pfc_df, tz_local, include_internal=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy")

    logger.info("Export Parquet : %s (%d lignes)", path, len(df))
    return path


def export_both(
    pfc_df: pd.DataFrame,
    output_dir: str | Path,
    run_date: str | None = None,
    tz_local: str = "Europe/Zurich",
    filename_base: str | None = None,
) -> dict[str, Path]:
    """
    Exporte CSV + Parquet avec nommage automatique.

    Args:
        output_dir : répertoire de sortie
        run_date   : date du run au format 'YYYYMMDD' (défaut = aujourd'hui)

    Returns:
        dict {'csv': Path, 'parquet': Path}
    """
    if filename_base:
        base = Path(output_dir) / filename_base
    else:
        if run_date is None:
            run_date = pd.Timestamp.now().strftime("%Y%m%d")
        base = Path(output_dir) / f"pfc_15min_{run_date}"
    return {
        "csv":     export_csv(pfc_df, base.with_suffix(".csv"), tz_local),
        "parquet": export_parquet(pfc_df, base.with_suffix(".parquet"), tz_local),
    }


# ---------------------------------------------------------------------------
# Interne
# ---------------------------------------------------------------------------

def _prepare_export(pfc_df: pd.DataFrame, tz_local: str, include_internal: bool = False) -> pd.DataFrame:
    """
    Prépare le DataFrame pour l'export :
      - Convertit l'index UTC → tz_local
      - Sélectionne et ordonne les colonnes EULER
      - Renomme l'index
    """
    df = pfc_df.copy()

    # Conversion timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(tz_local)
    df.index.name = "timestamp_local"

    export_cols = INTERNAL_COLUMNS if include_internal else EULER_COLUMNS
    cols = [c for c in export_cols if c in df.columns]
    missing = [c for c in export_cols if c not in df.columns]
    if missing:
        logger.debug("Colonnes absentes de l'export : %s", missing)

    return df[cols]
