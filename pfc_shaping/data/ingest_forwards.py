"""
ingest_forwards.py
------------------
Chargement des forwards EEX calibrÃ©s par EULER depuis Databricks.

Ces forwards constituent les niveaux de base B (â‚¬/MWh) utilisÃ©s par
l'assembleur PFC. EULER les calibre sur les prix EEX liquides ; notre
modÃ¨le applique par-dessus la forme (shape) 15min.

Table Databricks attendue (config.yaml â†’ databricks.tables.eex_forwards) :
    run_date        DATE        â€” date du run EULER (dernier run disponible)
    delivery_period STRING      â€” 'YYYY' | 'YYYY-QN' | 'YYYY-MM'
    price_eur_mwh   DOUBLE      â€” prix forward calibrÃ© [â‚¬/MWh]
    product_type    STRING      â€” 'Cal' | 'Quarter' | 'Month'

Format de sortie :
    dict[str, float] â€” clÃ©s : '2025', '2025-Q1', '2025-03', etc.
    Directement compatible avec PFCAssembler.build(base_prices=...)

Usage :
    from pfc_shaping.data.ingest_forwards import load_base_prices
    base_prices = load_base_prices(run_date='2025-03-10')
    # {'2025': 75.0, '2026': 72.0, '2025-Q2': 74.5, '2025-03': 82.0, ...}
"""

from __future__ import annotations

import logging

import pandas as pd

from pfc_shaping.data.databricks_client import query_to_df, table_fqn

logger = logging.getLogger(__name__)


def load_base_prices(
    run_date: str | None = None,
    db_config: dict | None = None,
) -> dict[str, float]:
    """
    Charge les forwards EEX depuis Databricks et retourne le dict base_prices.

    Args:
        run_date  : 'YYYY-MM-DD' du run EULER Ã  utiliser
                    Si None â†’ prend le run le plus rÃ©cent disponible
        db_config : config Databricks (si None, lit config.yaml)

    Returns:
        dict[str, float] â€” ex. {'2025': 75.0, '2025-Q1': 80.0, '2025-03': 82.0}
    """
    fqn = table_fqn("eex_forwards", db_config)

    if run_date is None:
        # Requête atomique : dernier run disponible + données en une seule passe
        sql = f"""
            SELECT delivery_period, price_eur_mwh, product_type
            FROM {fqn}
            WHERE run_date = (SELECT MAX(run_date) FROM {fqn})
            ORDER BY product_type, delivery_period
        """
        logger.info("Chargement forwards EEX (run le plus récent)...")
        df = query_to_df(sql, config=db_config)
        if not df.empty:
            run_date = "latest"
    else:
        # Validation format date
        import re
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", run_date):
            raise ValueError(f"run_date invalide (attendu YYYY-MM-DD) : {run_date}")

        sql = f"""
            SELECT delivery_period, price_eur_mwh, product_type
            FROM {fqn}
            WHERE run_date = ?
            ORDER BY product_type, delivery_period
        """
        logger.info("Chargement forwards EEX (run=%s)...", run_date)
        df = query_to_df(sql, params=[run_date], config=db_config)

    if df.empty:
        raise ValueError(f"Aucun forward EEX trouvÃ© pour run_date={run_date}")

    base_prices: dict[str, float] = {}
    for _, row in df.iterrows():
        key = str(row["delivery_period"]).strip()
        price = float(row["price_eur_mwh"])
        base_prices[key] = price

    logger.info(
        "Forwards EEX chargÃ©s : %d produits (Cal=%d, Quarter=%d, Month=%d)",
        len(base_prices),
        df[df["product_type"] == "Cal"].shape[0],
        df[df["product_type"] == "Quarter"].shape[0],
        df[df["product_type"] == "Month"].shape[0],
    )
    return base_prices


def latest_run_date(db_config: dict | None = None) -> str:
    """Retourne la date du dernier run EULER disponible dans Databricks."""
    fqn = table_fqn("eex_forwards", db_config)
    sql = f"SELECT MAX(run_date) AS latest FROM {fqn}"
    df = query_to_df(sql, config=db_config)
    return str(df["latest"].iloc[0])
