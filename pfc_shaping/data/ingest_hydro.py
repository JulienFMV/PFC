"""
ingest_hydro.py
---------------
Ingestion des niveaux de remplissage des réservoirs hydroélectriques suisses.

Sources (par ordre de priorité) :
    1. opendata.swiss (SFOE/BFE) — CSV public, données depuis 2000, hebdomadaire
       URL : https://www.uvek-gis.admin.ch/BFE/ogd/17/ogd17_fuellungsgrad_speicherseen.csv
    2. Databricks (table hydro_reservoir_levels) — fallback

Colonnes CSV SFOE :
    Datum, Wallis_speicherinhalt_gwh, Graubuenden_speicherinhalt_gwh,
    Tessin_speicherinhalt_gwh, UebrigCH_speicherinhalt_gwh,
    TotalCH_speicherinhalt_gwh, Wallis_max_speicherinhalt_gwh, ...

Variables dérivées :
    - fill_deviation : z-score vs moyenne historique (même semaine ISO, 10 ans glissants)
    - water_value_proxy : indicateur composite (= fill_deviation si snow_cover absent)

Format de sortie canonique (Parquet local) :
    index : DatetimeIndex UTC (fréquence hebdomadaire)
    colonnes :
        fill_pct          — niveau de remplissage en % (0-100)
        fill_gwh          — énergie stockée [GWh]
        max_capacity_gwh  — capacité maximale [GWh]
        fill_deviation    — z-score vs moyenne historique
        water_value_proxy — indicateur composite
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

DEFAULT_PARQUET = Path(__file__).parent.parent / "data" / "hydro_reservoir.parquet"

# Nombre d'années pour la fenêtre glissante du z-score historique
ROLLING_WINDOW_YEARS = 10

# URL opendata.swiss (SFOE/BFE)
SFOE_CSV_URL = (
    "https://www.uvek-gis.admin.ch/BFE/ogd/17/"
    "ogd17_fuellungsgrad_speicherseen.csv"
)

REQUEST_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Source 1 : opendata.swiss (SFOE)
# ---------------------------------------------------------------------------

def load_from_sfoe() -> pd.DataFrame:
    """
    Charge l'historique complet des niveaux de réservoirs depuis opendata.swiss.

    Le CSV contient toutes les données depuis 2000, mis à jour hebdomadairement.
    On utilise les colonnes TotalCH (agrégé national).

    Returns:
        DataFrame colonnes ['fill_pct', 'fill_gwh', 'max_capacity_gwh',
                            'wallis_gwh', 'graubuenden_gwh', 'tessin_gwh']
        index : DatetimeIndex UTC (fréquence hebdomadaire)
    """
    logger.info("Téléchargement réservoirs SFOE depuis opendata.swiss...")

    resp = requests.get(SFOE_CSV_URL, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    # Lire le CSV (séparateur virgule, encoding UTF-8)
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text), sep=",")

    if df.empty:
        raise ValueError("CSV SFOE vide")

    # Renommer les colonnes
    df = df.rename(columns={"Datum": "week_start"})
    df["week_start"] = pd.to_datetime(df["week_start"], utc=True)
    df = df.set_index("week_start").sort_index()

    # Construire le format canonique (Total CH)
    result = pd.DataFrame(index=df.index)
    result["fill_gwh"] = df["TotalCH_speicherinhalt_gwh"].astype(float)
    result["max_capacity_gwh"] = df["TotalCH_max_speicherinhalt_gwh"].astype(float)
    result["fill_pct"] = (result["fill_gwh"] / result["max_capacity_gwh"] * 100).round(2)

    # Colonnes régionales (utiles pour analyse Valais-spécifique)
    result["wallis_gwh"] = df.get("Wallis_speicherinhalt_gwh", np.nan)
    result["graubuenden_gwh"] = df.get("Graubuenden_speicherinhalt_gwh", np.nan)
    result["tessin_gwh"] = df.get("Tessin_speicherinhalt_gwh", np.nan)

    # Validation
    result["fill_pct"] = result["fill_pct"].clip(0, 100)
    result["fill_gwh"] = result["fill_gwh"].clip(lower=0)

    logger.info(
        "SFOE réservoirs chargés : %d semaines (%s → %s), "
        "fill_pct min=%.1f%% max=%.1f%%, capacité max=%.0f GWh",
        len(result),
        result.index.min().date(), result.index.max().date(),
        result["fill_pct"].min(), result["fill_pct"].max(),
        result["max_capacity_gwh"].iloc[-1],
    )
    return result


# ---------------------------------------------------------------------------
# Source 2 : Databricks (fallback)
# ---------------------------------------------------------------------------

def load_from_databricks(
    start: str,
    end: str,
    db_config: dict | None = None,
) -> pd.DataFrame:
    """
    Charge les niveaux de remplissage des réservoirs depuis Databricks.

    Args:
        start: date de début 'YYYY-MM-DD'
        end: date de fin 'YYYY-MM-DD' (exclu)
        db_config: config Databricks (si None, lit config.yaml)

    Returns:
        DataFrame colonnes ['fill_pct', 'fill_gwh', 'max_capacity_gwh']
        index : DatetimeIndex UTC (fréquence hebdomadaire)
    """
    from pfc_shaping.data.databricks_client import query_to_df, table_fqn

    fqn = table_fqn("hydro_reservoir_levels", db_config)

    # Charger les données avec un buffer historique pour le calcul du z-score
    start_with_buffer = (
        pd.Timestamp(start) - pd.DateOffset(years=ROLLING_WINDOW_YEARS)
    ).strftime("%Y-%m-%d")

    sql = f"""
        SELECT week_start, fill_pct, fill_gwh, max_capacity_gwh
        FROM {fqn}
        WHERE week_start >= ?
          AND week_start <  ?
        ORDER BY week_start
    """

    logger.info("Hydro réservoirs Databricks : %s → %s (buffer %s)", start, end, start_with_buffer)
    raw = query_to_df(sql, params=[start_with_buffer, end], config=db_config)

    if raw.empty:
        logger.warning("Aucune donnée hydro retournée pour %s → %s", start, end)
        return pd.DataFrame(
            columns=["fill_pct", "fill_gwh", "max_capacity_gwh"],
            index=pd.DatetimeIndex([], name="week_start", tz="UTC"),
        )

    raw["week_start"] = pd.to_datetime(raw["week_start"], utc=True)
    raw = raw.set_index("week_start").sort_index()

    # Validation basique des bornes
    invalid_pct = (raw["fill_pct"] < 0) | (raw["fill_pct"] > 100)
    if invalid_pct.any():
        logger.warning("%d valeurs fill_pct hors bornes [0,100] — clampées", invalid_pct.sum())
        raw["fill_pct"] = raw["fill_pct"].clip(0, 100)

    invalid_gwh = raw["fill_gwh"] < 0
    if invalid_gwh.any():
        logger.warning("%d valeurs fill_gwh négatives — mises à 0", invalid_gwh.sum())
        raw["fill_gwh"] = raw["fill_gwh"].clip(lower=0)

    logger.info(
        "Hydro chargé : %d semaines, fill_pct min=%.1f%% max=%.1f%%",
        len(raw), raw["fill_pct"].min(), raw["fill_pct"].max(),
    )
    return raw


# ---------------------------------------------------------------------------
# Water value features
# ---------------------------------------------------------------------------

def build_water_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit le DataFrame avec fill_deviation et water_value_proxy.

    fill_deviation :
        Z-score du remplissage actuel vs la moyenne historique pour la même
        semaine calendaire ISO, calculé sur une fenêtre glissante de 10 ans.
        Positif = réservoirs au-dessus de la moyenne → prix attendus plus bas.
        Négatif = réservoirs en-dessous → prix attendus plus hauts.

    water_value_proxy :
        Indicateur composite. En l'absence de données de couverture neigeuse,
        identique à fill_deviation.

    Args:
        df: DataFrame avec colonnes ['fill_pct', 'fill_gwh', 'max_capacity_gwh']
            et index DatetimeIndex UTC

    Returns:
        DataFrame enrichi avec colonnes supplémentaires ['fill_deviation', 'water_value_proxy']
    """
    df = df.copy()

    # Semaine ISO calendaire (1-53) pour le regroupement saisonnier
    df["iso_week"] = df.index.isocalendar().week.values.astype(int)

    # Calcul du z-score par semaine calendaire sur fenêtre glissante
    df["fill_deviation"] = np.nan

    for iso_week in df["iso_week"].unique():
        mask_week = df["iso_week"] == iso_week
        week_data = df.loc[mask_week, "fill_pct"]

        if len(week_data) < 3:
            logger.debug("Semaine ISO %d : %d obs (insuffisant pour z-score)", iso_week, len(week_data))
            continue

        # Fenêtre glissante : pour chaque observation, calculer mean/std
        # sur les ROLLING_WINDOW_YEARS années précédentes
        for idx in week_data.index:
            cutoff = idx - pd.DateOffset(years=ROLLING_WINDOW_YEARS)
            historical = week_data.loc[(week_data.index >= cutoff) & (week_data.index < idx)]

            if len(historical) < 3:
                # Pas assez d'historique — utiliser tout ce qui est disponible avant
                historical = week_data.loc[week_data.index < idx]

            if len(historical) < 2:
                continue

            hist_mean = historical.mean()
            hist_std = historical.std()

            if hist_std > 0:
                df.loc[idx, "fill_deviation"] = (week_data.loc[idx] - hist_mean) / hist_std
            else:
                df.loc[idx, "fill_deviation"] = 0.0

    # Remplir les NaN restants (premières années sans historique)
    n_missing = df["fill_deviation"].isna().sum()
    if n_missing > 0:
        logger.info(
            "%d semaines sans fill_deviation calculable (historique insuffisant) — rempli à 0.0",
            n_missing,
        )
        df["fill_deviation"] = df["fill_deviation"].fillna(0.0)

    # Water value proxy : fill_deviation seul (snow_cover non disponible)
    df["water_value_proxy"] = df["fill_deviation"]

    # Nettoyage colonne intermédiaire
    df.drop(columns=["iso_week"], inplace=True)

    logger.info(
        "Water value calculé : fill_deviation mean=%.2f std=%.2f",
        df["fill_deviation"].mean(),
        df["fill_deviation"].std(),
    )
    return df


# ---------------------------------------------------------------------------
# Cache Parquet
# ---------------------------------------------------------------------------

def load_parquet(path: str | Path = DEFAULT_PARQUET) -> pd.DataFrame:
    """Charge le cache Parquet local."""
    return pd.read_parquet(path)


def fetch_and_cache(
    start: str,
    end: str,
    parquet_path: str | Path = DEFAULT_PARQUET,
    db_config: dict | None = None,
) -> pd.DataFrame:
    """
    Télécharge les données hydro, fusionne avec le cache local et sauvegarde.
    Recalcule les features sur l'ensemble (le z-score peut changer avec de
    nouvelles données).

    Priorité : opendata.swiss (SFOE) > Databricks > cache local.

    Args:
        start: date de début 'YYYY-MM-DD'
        end: date de fin 'YYYY-MM-DD' (exclu)
        parquet_path: chemin du cache Parquet local
        db_config: config Databricks (si None, lit config.yaml)

    Returns:
        DataFrame canonique complet mis à jour
    """
    new_raw = None

    # Priorité 1 : opendata.swiss (SFOE) — historique complet depuis 2000
    try:
        new_raw = load_from_sfoe()
        logger.info("Hydro chargé depuis opendata.swiss (SFOE)")
    except Exception as e:
        logger.warning("opendata.swiss hydro échoué (%s) — fallback Databricks", e)

    # Priorité 2 : Databricks
    if new_raw is None:
        try:
            new_raw = load_from_databricks(start, end, db_config)
            logger.info("Hydro chargé depuis Databricks")
        except Exception as e:
            logger.warning("Databricks hydro échoué (%s) — fallback cache local", e)

    parquet_path = Path(parquet_path)

    if new_raw is not None and not new_raw.empty:
        if parquet_path.exists():
            existing = load_parquet(parquet_path)
            raw_cols = ["fill_pct", "fill_gwh", "max_capacity_gwh",
                        "wallis_gwh", "graubuenden_gwh", "tessin_gwh"]
            existing_raw = existing[[c for c in raw_cols if c in existing.columns]]
            combined = pd.concat([existing_raw, new_raw])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = new_raw
    elif parquet_path.exists():
        logger.info("Aucune source distante — utilisation du cache local")
        return load_parquet(parquet_path)
    else:
        raise ValueError("Aucune donnée hydro disponible (ni SFOE, ni Databricks, ni cache)")

    # Recalcul features sur l'ensemble complet
    combined = build_water_value(combined)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
    logger.info("Cache hydro mis à jour : %s (%d lignes)", parquet_path, len(combined))
    return combined
