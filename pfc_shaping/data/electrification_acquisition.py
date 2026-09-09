"""Source-only acquisition helpers for electrification scenario inventories."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pfc_shaping.data.electrification_scenarios import (
    validate_electrification_scenarios,
)

DEFAULT_ELECTRIFICATION_SCENARIO_PATH = (
    Path.home() / "pfc_local_data" / "curated" / "electrification_scenarios.parquet"
)
DEFAULT_HPFC_SCENARIO_FEATURES_PATH = (
    Path.home() / "pfc_local_data" / "curated" / "hpfc_scenario_features.parquet"
)


def load_electrification_scenarios_from_databricks(
    *,
    table_key: str = "electrification_scenarios",
    config: dict | None = None,
    where_sql: str | None = None,
    require_recommended: bool = False,
) -> pd.DataFrame:
    """Acquire scenario rows from a configured Databricks table."""

    from pfc_shaping.data.databricks_client import query_to_df, table_fqn

    fqn = table_fqn(table_key, config=config)
    sql = f"SELECT * FROM {fqn}"
    if where_sql:
        sql += f" WHERE {where_sql}"
    return validate_electrification_scenarios(
        query_to_df(sql, config=config),
        require_recommended=require_recommended,
    )
