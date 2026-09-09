"""Source-only Databricks acquisition for upstream EEX forward materialization."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def load_base_prices(
    run_date: str | None = None,
    db_config: dict | None = None,
) -> dict[str, float]:
    """Load one explicit or latest EULER forward run from Databricks."""

    from pfc_shaping.data.databricks_client import query_to_df, table_fqn

    fqn = table_fqn("eex_forwards", db_config)
    if run_date is None:
        sql = f"""
            SELECT delivery_period, price_eur_mwh, product_type
            FROM {fqn}
            WHERE run_date = (SELECT MAX(run_date) FROM {fqn})
            ORDER BY product_type, delivery_period
        """
        logger.info("Loading latest EEX forwards from Databricks")
        frame = query_to_df(sql, config=db_config)
        if not frame.empty:
            run_date = "latest"
    else:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", run_date):
            raise ValueError(f"invalid run_date (expected YYYY-MM-DD): {run_date}")
        sql = f"""
            SELECT delivery_period, price_eur_mwh, product_type
            FROM {fqn}
            WHERE run_date = ?
            ORDER BY product_type, delivery_period
        """
        logger.info("Loading EEX forwards from Databricks (run=%s)", run_date)
        frame = query_to_df(sql, params=[run_date], config=db_config)
    if frame.empty:
        raise ValueError(f"no EEX forwards found for run_date={run_date}")
    prices = {
        str(row["delivery_period"]).strip(): float(row["price_eur_mwh"])
        for _, row in frame.iterrows()
    }
    logger.info("Loaded %d EEX forward products from Databricks", len(prices))
    return prices


def latest_run_date(db_config: dict | None = None) -> str:
    """Return the latest EULER run date available in Databricks."""

    from pfc_shaping.data.databricks_client import query_to_df, table_fqn

    fqn = table_fqn("eex_forwards", db_config)
    frame = query_to_df(f"SELECT MAX(run_date) AS latest FROM {fqn}", config=db_config)
    return str(frame["latest"].iloc[0])
