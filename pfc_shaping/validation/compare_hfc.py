"""
compare_hfc.py
--------------
Quick benchmark utility: compare latest PFC export against latest OMPEX HFC file.

Usage:
    python -m pfc_shaping.validation.compare_hfc
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (CONFIG_PATH.parent / p).resolve()


def _latest_file(folder: Path, pattern: str) -> Path | None:
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _load_latest_pfc_csv(output_dir: Path) -> pd.Series:
    pfc_file = _latest_file(output_dir, "pfc_15min_*.csv")
    if pfc_file is None:
        raise FileNotFoundError(f"No PFC CSV found in {output_dir}")

    df = pd.read_csv(pfc_file, sep=";")
    if "timestamp_local" not in df.columns or "price_shape" not in df.columns:
        raise ValueError(f"Unexpected PFC format in {pfc_file}")

    ts = pd.to_datetime(df["timestamp_local"], errors="coerce")
    pfc = pd.Series(df["price_shape"].astype(float).values, index=ts, name="pfc")
    pfc = pfc[~pfc.index.isna()].sort_index()
    if pfc.empty:
        raise ValueError(f"Empty PFC series in {pfc_file}")
    return pfc


def _load_latest_hfc_xlsx(hfc_dir: Path) -> pd.Series:
    hfc_file = _latest_file(hfc_dir, "HFC_Ompex_*.xlsx")
    if hfc_file is None:
        raise FileNotFoundError(f"No HFC files found in {hfc_dir}")

    df = pd.read_excel(hfc_file, sheet_name=0)
    if "Date" not in df.columns or "EUR/MWh" not in df.columns:
        raise ValueError(f"Unexpected HFC format in {hfc_file}")

    ts = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    hfc = pd.Series(df["EUR/MWh"].astype(float).values, index=ts, name="hfc")
    hfc = hfc[~hfc.index.isna()].sort_index()
    if hfc.empty:
        raise ValueError(f"Empty HFC series in {hfc_file}")
    return hfc


def _align_for_comparison(pfc: pd.Series, hfc: pd.Series) -> pd.DataFrame:
    # Normalize duplicate timestamps (DST transitions / duplicate source rows).
    if pfc.index.has_duplicates:
        pfc = pfc.groupby(level=0).mean()
    if hfc.index.has_duplicates:
        hfc = hfc.groupby(level=0).mean()

    # If HFC is hourly and PFC is 15-min, compare at hourly granularity.
    pfc_step = pfc.index.to_series().diff().dropna().median()
    hfc_step = hfc.index.to_series().diff().dropna().median()

    if pd.notna(pfc_step) and pd.notna(hfc_step) and pfc_step < hfc_step:
        pfc_cmp = pfc.resample("1h").mean()
    else:
        pfc_cmp = pfc

    df = pd.concat([pfc_cmp, hfc], axis=1, join="inner").dropna()
    if df.empty:
        raise ValueError("No overlapping timestamps between PFC and HFC")
    return df


def _metrics(df: pd.DataFrame) -> dict[str, float]:
    err = df["pfc"] - df["hfc"]
    return {
        "n_points": float(len(df)),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "bias": float(err.mean()),
        "p95_abs_error": float(err.abs().quantile(0.95)),
    }


def main() -> None:
    cfg = _load_config()
    paths = cfg.get("paths", {})
    output_dir = _resolve_path(paths.get("output_dir", "output"))
    hfc_dir = Path(paths.get("hfc_benchmark_dir", ""))

    pfc = _load_latest_pfc_csv(output_dir)
    hfc = _load_latest_hfc_xlsx(hfc_dir)
    df = _align_for_comparison(pfc, hfc)
    m = _metrics(df)

    print("PFC vs HFC benchmark")
    print(f"points={int(m['n_points'])}")
    print(f"mae={m['mae']:.4f}")
    print(f"rmse={m['rmse']:.4f}")
    print(f"bias={m['bias']:.4f}")
    print(f"p95_abs_error={m['p95_abs_error']:.4f}")
    print(f"window={df.index.min()} -> {df.index.max()}")


if __name__ == "__main__":
    main()
