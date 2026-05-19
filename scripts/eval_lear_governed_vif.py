#!/usr/bin/env python3
"""Audit governed-feature multicollinearity for Swiss CT LEAR."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.model.lear_forecaster import LEARForecaster  # noqa: E402


def _read_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return None if df.empty else df


def main() -> None:
    data_root = ROOT / "pfc_shaping" / "data"
    out_root = ROOT / "pfc_shaping" / "output"
    out_root.mkdir(parents=True, exist_ok=True)

    lear = LEARForecaster(
        tz="Europe/Zurich",
        use_foundation_model=False,
        use_governed_forecast_features=True,
    )
    lear.fit(
        epex_15min=pd.read_parquet(data_root / "epex_15min.parquet"),
        entso_15min=pd.read_parquet(data_root / "entso_fundamentals_15min.parquet"),
        outages_15min=_read_optional(data_root / "outages_15min.parquet"),
        commodities=_read_optional(ROOT / "data" / "commodities_cache.parquet"),
        hydro=pd.read_parquet(data_root / "hydro_reservoir.parquet"),
        epex_de_15min=pd.read_parquet(data_root / "epex_de_15min.parquet"),
        de_renewable_forecast=_read_optional(data_root / "de_renewable_forecast.parquet"),
        multi_country_forecast=_read_optional(data_root / "multi_country_forecast_15min.parquet"),
        weather_forecast=_read_optional(data_root / "weather_forecast_hourly.parquet"),
    )
    report = lear.get_governed_vif_report()
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
        "report": report,
    }
    output_path = out_root / "governed_vif_latest.json"
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))
    print(f"output:{output_path.resolve()}")


if __name__ == "__main__":
    main()
