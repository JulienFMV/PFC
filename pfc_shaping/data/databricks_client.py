"""
databricks_client.py
--------------------
Connexion partagée au Databricks SQL Warehouse (databricks-sql-connector).

La connexion est gérée comme un singleton par session pour éviter d'ouvrir
un nouveau warehouse à chaque appel.

Paramètres de connexion (dans config.yaml sous clé 'databricks') :
    host        : ex. "adb-xxxx.azuredatabricks.net"
    http_path   : ex. "/sql/1.0/warehouses/xxxx"
    catalog     : ex. "fmv_prod"
    schema      : ex. "market_data"

Authentification (par ordre de priorité) :
    1. Variable d'environnement  DATABRICKS_TOKEN
    2. Clé 'token' dans config.yaml (déconseillé en production)

Usage :
    from data.databricks_client import get_connection, query_to_df

    df = query_to_df("SELECT * FROM epex_15min WHERE dt >= '2024-01-01'")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

# Singleton de connexion
_connection = None


def _load_db_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("databricks", {})


def get_connection(config: dict | None = None):
    """
    Retourne la connexion Databricks (réutilise le singleton existant).

    Args:
        config : dict de configuration Databricks (si None, charge config.yaml)

    Returns:
        databricks.sql.Connection
    """
    global _connection

    if _connection is not None:
        try:
            # Test de la connexion existante
            _connection.cursor().execute("SELECT 1")
            return _connection
        except Exception:
            logger.info("Connexion Databricks expirée — reconnexion...")
            _connection = None

    try:
        from databricks import sql as dbsql
    except ImportError as e:
        raise ImportError(
            "Installez databricks-sql-connector : pip install databricks-sql-connector"
        ) from e

    if config is None:
        config = _load_db_config()

    host = config.get("host") or os.environ.get("DATABRICKS_HOST")
    http_path = config.get("http_path") or os.environ.get("DATABRICKS_HTTP_PATH")
    token = config.get("token") or os.environ.get("DATABRICKS_TOKEN")

    if not all([host, http_path, token]):
        missing = [k for k, v in {"host": host, "http_path": http_path, "token": token}.items() if not v]
        raise EnvironmentError(
            f"Paramètres Databricks manquants : {missing}. "
            "Définissez DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN "
            "ou renseignez la section 'databricks' dans config.yaml."
        )

    logger.info("Connexion Databricks SQL Warehouse : %s%s", host, http_path)
    _connection = dbsql.connect(
        server_hostname=host,
        http_path=http_path,
        access_token=token,
    )
    return _connection


def query_to_df(
    sql: str,
    config: dict | None = None,
    params: list | None = None,
) -> pd.DataFrame:
    """
    Exécute une requête SQL Databricks et retourne un DataFrame pandas.

    Args:
        sql    : requête SQL (peut contenir des ? pour les paramètres)
        config : config Databricks (si None, charge config.yaml)
        params : paramètres de la requête (liste)

    Returns:
        DataFrame pandas avec les colonnes du résultat
    """
    conn = get_connection(config)
    with conn.cursor() as cursor:
        logger.debug("SQL : %s", sql[:200])
        cursor.execute(sql, params or [])
        cols = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=cols)
    logger.debug("Résultat : %d lignes × %d colonnes", len(df), len(cols))
    return df


def table_fqn(table_key: str, config: dict | None = None) -> str:
    """
    Retourne le nom complet (FQN) d'une table Databricks :
    catalog.schema.table_name

    Args:
        table_key : clé dans config.yaml → databricks.tables (ex. 'epex_15min')

    Returns:
        ex. "fmv_prod.market_data.epex_spot_15min"
    """
    if config is None:
        config = _load_db_config()

    catalog = config.get("catalog", "")
    schema = config.get("schema", "")
    tables = config.get("tables", {})

    table_name = tables.get(table_key)
    if not table_name:
        raise KeyError(
            f"Table '{table_key}' introuvable dans config.yaml → databricks.tables. "
            f"Tables connues : {list(tables.keys())}"
        )

    parts = [p for p in [catalog, schema, table_name] if p]
    return ".".join(parts)


def close() -> None:
    """Ferme la connexion singleton."""
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None
        logger.info("Connexion Databricks fermée")
