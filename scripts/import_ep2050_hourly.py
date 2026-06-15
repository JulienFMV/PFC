"""Import SFOE/BFE EP2050+ hourly electricity data locally.

This script keeps raw downloads on C: by default and writes normalized Parquet
outputs that can feed the Phase 13 HPFC scenario feature pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pfc_shaping.data.electrification_scenarios import derive_hpfc_scenario_features
from pfc_shaping.data.ep2050 import (
    DEFAULT_LOCAL_EP2050_DIR,
    build_ep2050_hourly,
    derive_ep2050_annual_scenarios,
    download_ep2050_hourly_zip,
    extract_ep2050_hourly_zip,
)


def _parse_csv_ints(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_csv_strs(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(DEFAULT_LOCAL_EP2050_DIR))
    parser.add_argument("--years", default="2025,2030", help="Comma-separated years, empty for all.")
    parser.add_argument("--scenarios", default="ZERO_Basis,WWB", help="Comma-separated scenarios.")
    parser.add_argument("--hourly-output", default="data/silver_ep2050_hourly.parquet")
    parser.add_argument("--annual-output", default="data/electrification_scenarios_ep2050.parquet")
    parser.add_argument("--features-output", default="data/hpfc_scenario_features_ep2050.parquet")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    if args.skip_download:
        zip_path = cache_dir / "ep2050_hourly_power_generation_consumption_11142.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"missing cached EP2050 ZIP: {zip_path}")
    else:
        zip_path = download_ep2050_hourly_zip(cache_dir)
    extracted = extract_ep2050_hourly_zip(zip_path, cache_dir / "hourly_11142")

    hourly = build_ep2050_hourly(
        extracted,
        years=_parse_csv_ints(args.years),
        scenarios=_parse_csv_strs(args.scenarios),
    )
    annual = derive_ep2050_annual_scenarios(hourly)
    features = derive_hpfc_scenario_features(annual)

    for output, frame in [
        (Path(args.hourly_output), hourly),
        (Path(args.annual_output), annual),
        (Path(args.features_output), features),
    ]:
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output, index=False)
        print(f"[ep2050] wrote {len(frame)} rows -> {output}")

    if len(hourly):
        print(f"[ep2050] scenarios={sorted(hourly['scenario'].astype(str).unique().tolist())}")
        print(f"[ep2050] years={sorted(hourly['year'].astype(int).unique().tolist())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
