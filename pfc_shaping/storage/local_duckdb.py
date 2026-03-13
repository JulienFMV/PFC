"""
local_duckdb.py
---------------
Local analytical storage for production runs (DuckDB).

Stores:
  - run metadata
  - hourly forecast snapshots
  - PFC vs HFC benchmark metrics
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def init_db(db_path: str | Path) -> Path:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
              run_id VARCHAR PRIMARY KEY,
              run_ts_utc TIMESTAMP,
              source_forwards VARCHAR,
              pfc_csv_path VARCHAR,
              pfc_parquet_path VARCHAR,
              row_count BIGINT,
              calibrated BOOLEAN
            );
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS forecasts_hourly (
              run_id VARCHAR,
              ts_local TIMESTAMP,
              price_shape DOUBLE,
              p10 DOUBLE,
              p90 DOUBLE
            );
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS benchmarks (
              run_id VARCHAR,
              hfc_file VARCHAR,
              n_points BIGINT,
              mae DOUBLE,
              rmse DOUBLE,
              bias DOUBLE,
              p95_abs_error DOUBLE,
              window_start TIMESTAMP,
              window_end TIMESTAMP
            );
            """
        )
        # Indexes for query performance
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_forecasts_ts ON forecasts_hourly(ts_local)"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_forecasts_run ON forecasts_hourly(run_id)"
        )
    return db_path


def upsert_run_and_forecast(
    db_path: str | Path,
    run_id: str,
    pfc_csv_path: str | Path,
    pfc_parquet_path: str | Path,
    source_forwards: str,
) -> None:
    db_path = Path(db_path)
    pfc_csv_path = Path(pfc_csv_path)
    pfc_parquet_path = Path(pfc_parquet_path)

    df = None
    if pfc_parquet_path.exists():
        try:
            df_parquet = pd.read_parquet(pfc_parquet_path)
            if isinstance(df_parquet.index, pd.DatetimeIndex):
                df = df_parquet.reset_index().rename(columns={df_parquet.index.name or "index": "timestamp_local"})
            else:
                df = df_parquet
        except Exception:
            df = None
    if df is None:
        df = pd.read_csv(pfc_csv_path, sep=";")

    if "timestamp_local" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "timestamp_local"})

    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"], errors="coerce")
    df = df.dropna(subset=["timestamp_local"])
    row_count = int(len(df))
    calibrated = bool(df.get("calibrated", pd.Series([False] * max(len(df), 1))).astype(bool).any())

    hourly = (
        df.set_index("timestamp_local")
        .resample("1h")
        .agg(
            {
                "price_shape": "mean",
                "p10": "mean" if "p10" in df.columns else "first",
                "p90": "mean" if "p90" in df.columns else "first",
            }
        )
        .reset_index()
    )
    hourly = hourly.rename(columns={"timestamp_local": "ts_local"})
    hourly["run_id"] = run_id
    hourly = hourly[["run_id", "ts_local", "price_shape", "p10", "p90"]]

    with duckdb.connect(str(db_path)) as con:
        con.execute("DELETE FROM forecasts_hourly WHERE run_id = ?", [run_id])
        con.execute("DELETE FROM runs WHERE run_id = ?", [run_id])
        con.execute(
            """
            INSERT INTO runs(run_id, run_ts_utc, source_forwards, pfc_csv_path, pfc_parquet_path, row_count, calibrated)
            VALUES (?, now(), ?, ?, ?, ?, ?)
            """,
            [run_id, source_forwards, str(pfc_csv_path), str(pfc_parquet_path), row_count, calibrated],
        )
        con.register("hourly_df", hourly)
        con.execute("INSERT INTO forecasts_hourly SELECT * FROM hourly_df")


def benchmark_against_hfc(
    db_path: str | Path,
    run_id: str,
    pfc_csv_path: str | Path,
    hfc_file: str | Path,
) -> dict[str, float]:
    pfc_csv_path = Path(pfc_csv_path)
    hfc_file = Path(hfc_file)

    pfc_df = pd.read_csv(pfc_csv_path, sep=";")
    pfc_df["timestamp_local"] = pd.to_datetime(pfc_df["timestamp_local"], errors="coerce")
    pfc_df = pfc_df.dropna(subset=["timestamp_local"])
    pfc = pd.Series(pfc_df["price_shape"].astype(float).values, index=pfc_df["timestamp_local"])
    if pfc.index.has_duplicates:
        pfc = pfc.groupby(level=0).mean()

    hfc_df = pd.read_excel(hfc_file, sheet_name=0)
    hfc_ts = pd.to_datetime(hfc_df["Date"], dayfirst=True, errors="coerce")
    hfc = pd.Series(hfc_df["EUR/MWh"].astype(float).values, index=hfc_ts)
    hfc = hfc[~hfc.index.isna()]
    if hfc.index.has_duplicates:
        hfc = hfc.groupby(level=0).mean()

    pfc_h = pfc.resample("1h").mean()
    joined = pd.concat([pfc_h.rename("pfc"), hfc.rename("hfc")], axis=1, join="inner").dropna()
    if joined.empty:
        raise ValueError("No overlap between PFC and HFC timestamps")

    err = joined["pfc"] - joined["hfc"]
    metrics = {
        "n_points": int(len(joined)),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "bias": float(err.mean()),
        "p95_abs_error": float(err.abs().quantile(0.95)),
        "window_start": joined.index.min(),
        "window_end": joined.index.max(),
    }

    with duckdb.connect(str(db_path)) as con:
        con.execute("DELETE FROM benchmarks WHERE run_id = ?", [run_id])
        con.execute(
            """
            INSERT INTO benchmarks(run_id, hfc_file, n_points, mae, rmse, bias, p95_abs_error, window_start, window_end)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                str(hfc_file),
                metrics["n_points"],
                metrics["mae"],
                metrics["rmse"],
                metrics["bias"],
                metrics["p95_abs_error"],
                metrics["window_start"],
                metrics["window_end"],
            ],
        )
    return metrics
