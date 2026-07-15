"""Build gold HPFC scenario penetration features.

Examples
--------
Local:
    python scripts/build_hpfc_scenario_features.py --vintage 2026-01-15

Databricks:
    python scripts/build_hpfc_scenario_features.py --databricks --vintage 2026-01-15
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pfc_shaping.data.electrification_acquisition import (
    DEFAULT_ELECTRIFICATION_SCENARIO_PATH,
    DEFAULT_HPFC_SCENARIO_FEATURES_PATH,
    load_electrification_scenarios_from_databricks,
)
from pfc_shaping.data.electrification_scenarios import (
    asof_electrification_scenarios,
    derive_hpfc_scenario_features,
    load_electrification_scenarios,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(DEFAULT_ELECTRIFICATION_SCENARIO_PATH))
    parser.add_argument("--output", default=str(DEFAULT_HPFC_SCENARIO_FEATURES_PATH))
    parser.add_argument("--databricks", action="store_true", help="Load from Databricks config table.")
    parser.add_argument("--table-key", default="electrification_scenarios")
    parser.add_argument("--where-sql", default=None)
    parser.add_argument("--vintage", default=None, help="Optional as-of vintage, e.g. 2026-01-15.")
    parser.add_argument("--require-recommended", action="store_true")
    args = parser.parse_args(argv)

    if args.databricks:
        frame = load_electrification_scenarios_from_databricks(
            table_key=args.table_key,
            where_sql=args.where_sql,
            require_recommended=args.require_recommended,
        )
    else:
        frame = load_electrification_scenarios(
            args.path,
            require_recommended=args.require_recommended,
        )
    if args.vintage is not None:
        frame = asof_electrification_scenarios(frame, args.vintage)

    features = derive_hpfc_scenario_features(frame)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output, index=False)

    print(f"[features] rows={len(features)}")
    print(f"[features] output -> {output}")
    if len(features):
        print(f"[features] countries={sorted(features['country'].astype(str).unique().tolist())}")
        print(f"[features] scenarios={sorted(features['scenario'].astype(str).unique().tolist())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
