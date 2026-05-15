#!/usr/bin/env python3
"""Audit Swiss CT dataset readiness and governance state."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.model.lear_forecaster import LEARForecaster  # noqa: E402


@dataclass
class DatasetStatus:
    dataset: str
    path: str
    exists: bool
    rows: int
    min_ts: str | None
    max_ts: str | None
    freshness_status: str
    notes: str


@dataclass
class VariableGroupStatus:
    group: str
    priority: str
    kind: str
    source: str
    consumed_by_prod: bool
    historical_status: str
    freshness_status: str
    overall_status: str
    notes: str


DATASETS = {
    "epex_ch": ROOT / "pfc_shaping" / "data" / "epex_15min.parquet",
    "epex_de": ROOT / "pfc_shaping" / "data" / "epex_de_15min.parquet",
    "epex_fr": ROOT / "pfc_shaping" / "data" / "epex_fr_15min.parquet",
    "epex_at": ROOT / "pfc_shaping" / "data" / "epex_at_15min.parquet",
    "epex_it": ROOT / "pfc_shaping" / "data" / "epex_it_15min.parquet",
    "entso": ROOT / "pfc_shaping" / "data" / "entso_15min.parquet",
    "de_renewable_forecast": ROOT / "pfc_shaping" / "data" / "de_renewable_forecast.parquet",
    "hydro": ROOT / "pfc_shaping" / "data" / "hydro_reservoir.parquet",
    "outages": ROOT / "pfc_shaping" / "data" / "outages_15min.parquet",
    "commodities": ROOT / "data" / "commodities_cache.parquet",
}


VARIABLE_GROUPS = [
    {
        "group": "CH day-ahead price",
        "priority": "critical",
        "kind": "realized_price",
        "source": "epex_ch",
        "consumed_by_prod": True,
    },
    {
        "group": "DE day-ahead price",
        "priority": "critical",
        "kind": "realized_price",
        "source": "epex_de",
        "consumed_by_prod": True,
    },
    {
        "group": "FR/AT/IT day-ahead prices",
        "priority": "high",
        "kind": "realized_price",
        "source": "epex_neighbors",
        "consumed_by_prod": True,
    },
    {
        "group": "CH realized load/solar/wind/cross-border",
        "priority": "critical",
        "kind": "realized_fundamentals",
        "source": "entso",
        "consumed_by_prod": True,
    },
    {
        "group": "DE realized load/solar/wind/residual",
        "priority": "critical",
        "kind": "realized_fundamentals",
        "source": "entso",
        "consumed_by_prod": True,
    },
    {
        "group": "FR/AT/IT realized load/solar/wind",
        "priority": "high",
        "kind": "realized_fundamentals",
        "source": "entso",
        "consumed_by_prod": False,
    },
    {
        "group": "CH border capacities / balances / schedules",
        "priority": "high",
        "kind": "cross_border_fundamentals",
        "source": "entso",
        "consumed_by_prod": False,
    },
    {
        "group": "FR nuclear stress",
        "priority": "high",
        "kind": "cross_border_fundamentals",
        "source": "entso",
        "consumed_by_prod": False,
    },
    {
        "group": "DE renewable day-ahead forecast",
        "priority": "critical",
        "kind": "forecast",
        "source": "de_renewable_forecast",
        "consumed_by_prod": True,
    },
    {
        "group": "Multi-country load/solar/wind forecasts",
        "priority": "critical",
        "kind": "forecast",
        "source": "missing_multi_country_forecasts",
        "consumed_by_prod": False,
    },
    {
        "group": "Weather forecasts",
        "priority": "critical",
        "kind": "forecast",
        "source": "missing_weather_forecasts",
        "consumed_by_prod": False,
    },
    {
        "group": "Hydro reservoir / water value drivers",
        "priority": "high",
        "kind": "structural_fundamentals",
        "source": "hydro",
        "consumed_by_prod": True,
    },
    {
        "group": "Outages / generation unavailability",
        "priority": "high",
        "kind": "structural_fundamentals",
        "source": "outages",
        "consumed_by_prod": True,
    },
    {
        "group": "Fuel / CO2 commodities",
        "priority": "medium",
        "kind": "structural_fundamentals",
        "source": "commodities",
        "consumed_by_prod": True,
    },
]


def _iso(ts) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    if isinstance(ts, pd.Timestamp):
        return ts.isoformat()
    return str(ts)


def _read_df(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _freshness_from_max(max_ts: pd.Timestamp | None, current_ts: pd.Timestamp, max_age_days: int) -> str:
    if max_ts is None:
        return "missing"
    if max_ts.tz is None:
        max_ts = max_ts.tz_localize("UTC")
    age = current_ts - max_ts.tz_convert("UTC")
    if age <= pd.Timedelta(days=max_age_days):
        return "ok"
    if age <= pd.Timedelta(days=max_age_days * 14):
        return "partial"
    return "stale"


def _dataset_status(name: str, path: Path, current_ts: pd.Timestamp) -> DatasetStatus:
    df = _read_df(path)
    if df is None:
        return DatasetStatus(name, str(path), False, 0, None, None, "missing", "dataset file absent")

    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else None
    min_ts = idx.min() if idx is not None and len(df) else None
    max_ts = idx.max() if idx is not None and len(df) else None

    if name in {"epex_ch", "epex_de", "entso", "outages"}:
        freshness = _freshness_from_max(max_ts, current_ts, 2)
    elif name == "de_renewable_forecast":
        freshness = _freshness_from_max(max_ts, current_ts + pd.Timedelta(days=1), 2)
    elif name == "hydro":
        freshness = _freshness_from_max(max_ts, current_ts, 10)
    else:
        freshness = _freshness_from_max(max_ts, current_ts, 14)

    return DatasetStatus(
        dataset=name,
        path=str(path),
        exists=True,
        rows=len(df),
        min_ts=_iso(min_ts),
        max_ts=_iso(max_ts),
        freshness_status=freshness,
        notes="",
    )


def _build_prod_exog() -> set[str]:
    epex_ch = pd.read_parquet(DATASETS["epex_ch"])
    epex_de = pd.read_parquet(DATASETS["epex_de"])
    entso = pd.read_parquet(DATASETS["entso"])
    hydro = pd.read_parquet(DATASETS["hydro"])
    outages = _read_df(DATASETS["outages"])
    commodities = _read_df(DATASETS["commodities"])

    model = LEARForecaster(
        use_foundation_model=True,
        use_gbm_blend=True,
        use_mlp_blend=False,
        gbm_blend_max_horizon_days=1,
    )
    model.fit(
        epex_15min=epex_ch,
        entso_15min=entso,
        outages_15min=outages,
        commodities=commodities,
        hydro=hydro,
        epex_de_15min=epex_de,
    )
    return set(model.exog_.columns)


def _group_status(group: dict, datasets: dict[str, DatasetStatus], prod_exog: set[str], current_ts: pd.Timestamp) -> VariableGroupStatus:
    source = group["source"]
    notes = []

    if source == "epex_neighbors":
        fr = datasets["epex_fr"]
        at = datasets["epex_at"]
        it = datasets["epex_it"]
        hist = "ok" if all(ds.exists and ds.rows > 0 for ds in [fr, at, it]) else "missing"
        fresh = "ok" if all(ds.freshness_status in {"ok", "partial"} for ds in [fr, at, it]) else "partial"
    elif source == "missing_multi_country_forecasts":
        hist = "missing"
        fresh = "missing"
        notes.append("No governed J+1 load/solar/wind forecasts for FR/AT/IT/CH/DE except DE renewables.")
    elif source == "missing_weather_forecasts":
        hist = "missing"
        fresh = "missing"
        notes.append("No governed forecast weather layer in current CT pipeline.")
    else:
        ds = datasets.get(source)
        hist = "ok" if ds and ds.exists and ds.rows > 0 else "missing"
        fresh = ds.freshness_status if ds else "missing"

    if group["group"] == "DE realized load/solar/wind/residual":
        entso = pd.read_parquet(DATASETS["entso"])
        recent = entso.loc[entso.index >= current_ts.tz_convert("UTC") - pd.Timedelta(days=60)]
        cols = ["load_de_mw", "solar_de_mw", "wind_de_mw", "residual_load_de_mw"]
        cov = min(float(recent[c].notna().mean()) for c in cols if c in recent.columns)
        hist = "ok" if cov > 0.9 else "partial"
        notes.append(f"recent_60d_coverage={cov:.2%}")
    elif group["group"] == "FR/AT/IT realized load/solar/wind":
        entso = pd.read_parquet(DATASETS["entso"])
        recent = entso.loc[entso.index >= current_ts.tz_convert("UTC") - pd.Timedelta(days=60)]
        cols = [
            "load_fr_mw", "solar_fr_mw", "wind_fr_mw",
            "load_at_mw", "solar_at_mw", "wind_at_mw",
            "load_it_mw", "solar_it_mw", "wind_it_mw",
        ]
        cov = min(float(recent[c].notna().mean()) for c in cols if c in recent.columns)
        hist = "partial" if cov > 0.5 else "missing"
        fresh = "ok" if cov > 0.5 else "missing"
        notes.append(f"recent_60d_coverage={cov:.2%}")
    elif group["group"] == "CH border capacities / balances / schedules":
        entso = pd.read_parquet(DATASETS["entso"])
        recent = entso.loc[entso.index >= current_ts.tz_convert("UTC") - pd.Timedelta(days=60)]
        cols = [
            "ntc_balance_ch_de", "ntc_balance_ch_fr",
            "scheduled_net_export_ch_de_mw_zscore", "scheduled_net_export_ch_fr_mw_zscore",
            "ntc_total_ch_de_mw_zscore", "ntc_total_ch_fr_mw_zscore",
        ]
        cov = min(float(recent[c].notna().mean()) for c in cols if c in recent.columns)
        hist = "ok" if cov > 0.95 else "partial"
        notes.append(f"recent_60d_coverage={cov:.2%}")
    elif group["group"] == "FR nuclear stress":
        entso = pd.read_parquet(DATASETS["entso"])
        cols = ["fr_nuclear_unavailable_mw", "fr_nuclear_unavailability_ratio", "fr_nuclear_stress_flag"]
        cov = min(float(entso[c].notna().mean()) for c in cols if c in entso.columns)
        hist = "ok" if cov > 0.95 else "partial"
        notes.append(f"full_history_coverage={cov:.2%}")
    elif group["group"] == "Hydro reservoir / water value drivers":
        ds = datasets["hydro"]
        notes.append(f"max_ts={ds.max_ts}")
        if ds.freshness_status != "ok":
            hist = "partial"

    overall = "ok"
    if hist == "missing" or fresh == "missing":
        overall = "missing"
    elif hist != "ok" or fresh != "ok" or not group["consumed_by_prod"]:
        overall = "partial"

    consumed = group["consumed_by_prod"]
    if group["group"] == "CH border capacities / balances / schedules":
        consumed = False
    if group["group"] == "FR nuclear stress":
        consumed = False
    if group["group"] == "FR/AT/IT realized load/solar/wind":
        consumed = False
    if group["group"] == "DE renewable day-ahead forecast":
        consumed = all(c in prod_exog for c in ["forecast_wind_de_mw", "forecast_solar_de_mw"])
    if group["group"] == "DE realized load/solar/wind/residual":
        consumed = all(c in prod_exog for c in ["load_de_mw", "solar_de_mw", "wind_de_mw", "residual_load_de_mw"])

    if not consumed and overall == "ok":
        overall = "partial"
        notes.append("Available but not consumed by current production model.")

    return VariableGroupStatus(
        group=group["group"],
        priority=group["priority"],
        kind=group["kind"],
        source=source,
        consumed_by_prod=consumed,
        historical_status=hist,
        freshness_status=fresh,
        overall_status=overall,
        notes=" ".join(notes).strip(),
    )


def _render_markdown(dataset_statuses: list[DatasetStatus], group_statuses: list[VariableGroupStatus]) -> str:
    lines = [
        "# Swiss CT Dataset Audit",
        "",
        "## Dataset Files",
        "",
        "| Dataset | Freshness | Rows | Max Timestamp | Path |",
        "|---|---:|---:|---|---|",
    ]
    for ds in dataset_statuses:
        lines.append(f"| {ds.dataset} | {ds.freshness_status} | {ds.rows} | {ds.max_ts or ''} | `{ds.path}` |")

    lines.extend([
        "",
        "## Variable Groups",
        "",
        "| Group | Priority | Source | Consumed By Prod | Historical | Freshness | Overall | Notes |",
        "|---|---|---|---:|---|---|---|---|",
    ])
    for st in group_statuses:
        lines.append(
            f"| {st.group} | {st.priority} | {st.source} | "
            f"{'yes' if st.consumed_by_prod else 'no'} | {st.historical_status} | "
            f"{st.freshness_status} | {st.overall_status} | {st.notes} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    current_ts = pd.Timestamp.now(tz="Europe/Zurich")
    output_dir = ROOT / "pfc_shaping" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_statuses = [_dataset_status(name, path, current_ts) for name, path in DATASETS.items()]
    dataset_map = {ds.dataset: ds for ds in dataset_statuses}
    prod_exog = _build_prod_exog()
    group_statuses = [_group_status(group, dataset_map, prod_exog, current_ts) for group in VARIABLE_GROUPS]

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "current_local_time": current_ts.isoformat(),
        "dataset_files": [asdict(ds) for ds in dataset_statuses],
        "variable_groups": [asdict(gs) for gs in group_statuses],
        "production_exog_columns": sorted(prod_exog),
    }

    json_path = output_dir / "ct_dataset_audit_latest.json"
    md_path = output_dir / "ct_dataset_audit_latest.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_render_markdown(dataset_statuses, group_statuses), encoding="utf-8")

    print(json.dumps({
        "json": str(json_path),
        "markdown": str(md_path),
        "missing_groups": [g["group"] for g in payload["variable_groups"] if g["overall_status"] == "missing"],
        "partial_groups": [g["group"] for g in payload["variable_groups"] if g["overall_status"] == "partial"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
