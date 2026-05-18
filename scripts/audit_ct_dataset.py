#!/usr/bin/env python3
"""Audit Swiss CT dataset readiness and governance state."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.data.ct_datasets import (  # noqa: E402
    get_dataset_spec,
    load_local_dataset,
    resolve_local_cache_path,
)
from pfc_shaping.model.lear_forecaster import LEARForecaster  # noqa: E402


@dataclass
class DatasetStatus:
    dataset: str
    registry_name: str
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


DATASET_REGISTRY = {
    "epex_ch": "ct_price_15min_ch",
    "epex_de": "ct_price_15min_de",
    "epex_fr": "ct_price_15min_fr",
    "epex_at": "ct_price_15min_at",
    "epex_it": "ct_price_15min_it",
    "entso_fundamentals": "ct_entso_fundamentals_15min",
    "entso_border": "ct_entso_border_15min",
    "de_renewable_forecast": "ct_forecast_de_renewables_15min",
    "multi_country_forecast": "ct_forecast_multi_country_15min",
    "weather_forecast": "ct_weather_forecast_hourly",
    "hydro": "ct_hydro_daily",
    "outages": "ct_outages_15min",
    "commodities": "ct_commodities_daily",
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
        "source": "entso_fundamentals",
        "consumed_by_prod": True,
    },
    {
        "group": "DE realized load/solar/wind/residual",
        "priority": "critical",
        "kind": "realized_fundamentals",
        "source": "entso_fundamentals",
        "consumed_by_prod": True,
    },
    {
        "group": "FR/AT/IT realized load/solar/wind",
        "priority": "high",
        "kind": "realized_fundamentals",
        "source": "entso_fundamentals",
        "consumed_by_prod": False,
    },
    {
        "group": "CH border capacities / balances / schedules",
        "priority": "high",
        "kind": "cross_border_fundamentals",
        "source": "entso_border",
        "consumed_by_prod": False,
    },
    {
        "group": "FR nuclear stress",
        "priority": "high",
        "kind": "cross_border_fundamentals",
        "source": "entso_fundamentals",
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
        "source": "multi_country_forecast",
        "consumed_by_prod": False,
    },
    {
        "group": "Weather forecasts",
        "priority": "critical",
        "kind": "forecast",
        "source": "weather_forecast",
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


def _extract_time_bounds(df: pd.DataFrame, time_key: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if df.empty:
        return None, None
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index.min(), df.index.max()
    if time_key in df.columns:
        series = pd.to_datetime(df[time_key], utc=True, errors="coerce").dropna()
        if not series.empty:
            return series.min(), series.max()
    return None, None


def _freshness_from_max(max_ts: pd.Timestamp | None, current_ts: pd.Timestamp, max_age_days: int) -> str:
    if max_ts is None:
        return "missing"
    if max_ts.tz is None:
        max_ts = max_ts.tz_localize("UTC")
    age = current_ts.tz_convert("UTC") - max_ts.tz_convert("UTC")
    if age <= pd.Timedelta(days=max_age_days):
        return "ok"
    if age <= pd.Timedelta(days=max_age_days * 14):
        return "partial"
    return "stale"


def _dataset_status(alias: str, current_ts: pd.Timestamp) -> tuple[DatasetStatus, pd.DataFrame | None]:
    dataset_name = DATASET_REGISTRY[alias]
    spec = get_dataset_spec(dataset_name)
    path = resolve_local_cache_path(ROOT, dataset_name)
    notes: list[str] = []
    exists = bool(path and path.exists())

    if not exists and dataset_name in {"ct_entso_fundamentals_15min", "ct_entso_border_15min"}:
        legacy_path = ROOT / "pfc_shaping" / "data" / "entso_15min.parquet"
        if legacy_path.exists():
            notes.append("serving from legacy entso_15min fallback until split cache is materialized")

    df = load_local_dataset(ROOT, dataset_name, required=False)
    if df is None:
        status = DatasetStatus(
            dataset=alias,
            registry_name=dataset_name,
            path="" if path is None else str(path),
            exists=False,
            rows=0,
            min_ts=None,
            max_ts=None,
            freshness_status="missing",
            notes="; ".join(notes) if notes else "dataset file absent",
        )
        return status, None

    min_ts, max_ts = _extract_time_bounds(df, spec.time_key)
    freshness = _freshness_from_max(max_ts, current_ts, spec.freshness_sla_days)
    status = DatasetStatus(
        dataset=alias,
        registry_name=dataset_name,
        path="" if path is None else str(path),
        exists=exists,
        rows=len(df),
        min_ts=_iso(min_ts),
        max_ts=_iso(max_ts),
        freshness_status=freshness,
        notes="; ".join(notes),
    )
    return status, df


def _build_prod_exog(dataset_frames: dict[str, pd.DataFrame | None]) -> set[str]:
    epex_ch = dataset_frames["epex_ch"]
    epex_de = dataset_frames["epex_de"]
    entso = dataset_frames["entso_fundamentals"]
    hydro = dataset_frames["hydro"]
    outages = dataset_frames["outages"]
    commodities = dataset_frames["commodities"]
    de_renewable_forecast = dataset_frames["de_renewable_forecast"]

    if epex_ch is None or epex_de is None or entso is None or hydro is None:
        return set()

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
        de_renewable_forecast=de_renewable_forecast,
    )
    return set(model.exog_.columns)


def _group_status(
    group: dict,
    datasets: dict[str, DatasetStatus],
    dataset_frames: dict[str, pd.DataFrame | None],
    prod_exog: set[str],
    current_ts: pd.Timestamp,
) -> VariableGroupStatus:
    source = group["source"]
    notes: list[str] = []

    if source == "epex_neighbors":
        fr = datasets["epex_fr"]
        at = datasets["epex_at"]
        it = datasets["epex_it"]
        hist = "ok" if all(ds.rows > 0 for ds in [fr, at, it]) else "missing"
        fresh = "ok" if all(ds.freshness_status in {"ok", "partial"} for ds in [fr, at, it]) else "partial"
    else:
        ds = datasets.get(source)
        hist = "ok" if ds and ds.rows > 0 else "missing"
        fresh = ds.freshness_status if ds else "missing"

    entso_fundamentals = dataset_frames.get("entso_fundamentals")
    entso_border = dataset_frames.get("entso_border")

    if group["group"] == "DE realized load/solar/wind/residual" and entso_fundamentals is not None:
        recent = entso_fundamentals.loc[
            entso_fundamentals.index >= current_ts.tz_convert("UTC") - pd.Timedelta(days=60)
        ]
        cols = ["load_de_mw", "solar_de_mw", "wind_de_mw", "residual_load_de_mw"]
        cov = min(float(recent[c].notna().mean()) for c in cols if c in recent.columns)
        hist = "ok" if cov > 0.9 else "partial"
        notes.append(f"recent_60d_coverage={cov:.2%}")
    elif group["group"] == "FR/AT/IT realized load/solar/wind" and entso_fundamentals is not None:
        recent = entso_fundamentals.loc[
            entso_fundamentals.index >= current_ts.tz_convert("UTC") - pd.Timedelta(days=60)
        ]
        cols = [
            "load_fr_mw", "solar_fr_mw", "wind_fr_mw",
            "load_at_mw", "solar_at_mw", "wind_at_mw",
            "load_it_mw", "solar_it_mw", "wind_it_mw",
        ]
        cov = min(float(recent[c].notna().mean()) for c in cols if c in recent.columns)
        hist = "partial" if cov > 0.5 else "missing"
        fresh = "ok" if cov > 0.5 else "missing"
        notes.append(f"recent_60d_coverage={cov:.2%}")
    elif group["group"] == "CH border capacities / balances / schedules" and entso_border is not None:
        recent = entso_border.loc[
            entso_border.index >= current_ts.tz_convert("UTC") - pd.Timedelta(days=60)
        ]
        cols = [
            "ntc_balance_ch_de", "ntc_balance_ch_fr",
            "scheduled_net_export_ch_de_mw_zscore", "scheduled_net_export_ch_fr_mw_zscore",
            "ntc_total_ch_de_mw_zscore", "ntc_total_ch_fr_mw_zscore",
        ]
        cov = min(float(recent[c].notna().mean()) for c in cols if c in recent.columns)
        hist = "ok" if cov > 0.95 else "partial"
        notes.append(f"recent_60d_coverage={cov:.2%}")
    elif group["group"] == "FR nuclear stress" and entso_fundamentals is not None:
        cols = ["fr_nuclear_unavailable_mw", "fr_nuclear_unavailability_ratio", "fr_nuclear_stress_flag"]
        cov = min(float(entso_fundamentals[c].notna().mean()) for c in cols if c in entso_fundamentals.columns)
        hist = "ok" if cov > 0.95 else "partial"
        notes.append(f"full_history_coverage={cov:.2%}")
    elif group["group"] == "Multi-country load/solar/wind forecasts":
        forecast_df = dataset_frames.get("multi_country_forecast")
        if forecast_df is None or forecast_df.empty:
            hist = "missing"
            fresh = "missing"
            notes.append("No governed multi-country forecast cache present.")
        else:
            recent = forecast_df.loc[
                forecast_df.index >= current_ts.tz_convert("UTC") - pd.Timedelta(days=30)
            ]
            required_cols = [
                "forecast_load_ch_mw",
                "forecast_solar_ch_mw",
                "forecast_wind_ch_mw",
                "forecast_load_de_mw",
                "forecast_load_fr_mw",
                "forecast_load_at_mw",
                "forecast_load_it_mw",
                "forecast_solar_de_mw",
                "forecast_wind_de_mw",
                "forecast_solar_fr_mw",
                "forecast_wind_fr_mw",
                "forecast_solar_at_mw",
                "forecast_wind_at_mw",
                "forecast_solar_it_mw",
                "forecast_wind_it_mw",
            ]
            present_cols = [c for c in required_cols if c in forecast_df.columns]
            if not present_cols:
                hist = "missing"
                fresh = "missing"
                notes.append("Forecast cache exists but required forecast columns are absent.")
            else:
                cov = min(float(recent[c].notna().mean()) for c in present_cols) if not recent.empty else 0.0
                hist = "ok" if cov > 0.8 and len(present_cols) >= 8 else "partial"
                fresh = datasets["multi_country_forecast"].freshness_status
                notes.append(
                    f"present_cols={len(present_cols)}/{len(required_cols)} recent_30d_coverage={cov:.2%}"
                )
    elif group["group"] == "Weather forecasts":
        weather_df = dataset_frames.get("weather_forecast")
        if weather_df is None or weather_df.empty:
            hist = "missing"
            fresh = "missing"
            notes.append("No governed weather forecast cache present.")
        else:
            recent = weather_df.loc[
                weather_df.index >= current_ts.tz_convert("UTC") - pd.Timedelta(days=30)
            ]
            required_cols = [
                "wx_ch_zurich_temperature_2m",
                "wx_ch_zurich_cloud_cover",
                "wx_ch_zurich_shortwave_radiation",
                "wx_ch_zurich_wind_speed_10m",
                "wx_de_hamburg_wind_speed_10m",
                "wx_de_munich_shortwave_radiation",
                "wx_fr_lyon_cloud_cover",
                "wx_at_vienna_wind_speed_10m",
                "wx_it_milan_shortwave_radiation",
            ]
            present_cols = [c for c in required_cols if c in weather_df.columns]
            if not present_cols:
                hist = "missing"
                fresh = "missing"
                notes.append("Weather cache exists but required governed columns are absent.")
            else:
                cov = min(float(recent[c].notna().mean()) for c in present_cols) if not recent.empty else 0.0
                hist = "ok" if cov > 0.9 else "partial"
                fresh = datasets["weather_forecast"].freshness_status
                notes.append(
                    f"present_cols={len(present_cols)}/{len(required_cols)} recent_30d_coverage={cov:.2%}"
                )
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
        consumed = any(c in prod_exog for c in ["ntc_balance_ch_de", "ntc_balance_ch_fr"])
    elif group["group"] == "FR nuclear stress":
        consumed = "fr_nuclear_stress_flag" in prod_exog
    elif group["group"] == "FR/AT/IT realized load/solar/wind":
        consumed = any(c in prod_exog for c in ["load_fr_mw", "solar_fr_mw", "wind_fr_mw"])
    elif group["group"] == "DE renewable day-ahead forecast":
        consumed = all(c in prod_exog for c in ["forecast_wind_de_mw", "forecast_solar_de_mw"])
    elif group["group"] == "DE realized load/solar/wind/residual":
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
        "| Dataset | Registry | Freshness | Rows | Max Timestamp | Path | Notes |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for ds in dataset_statuses:
        lines.append(
            f"| {ds.dataset} | `{ds.registry_name}` | {ds.freshness_status} | {ds.rows} | "
            f"{ds.max_ts or ''} | `{ds.path}` | {ds.notes} |"
        )

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

    dataset_statuses: list[DatasetStatus] = []
    dataset_frames: dict[str, pd.DataFrame | None] = {}
    for alias in DATASET_REGISTRY:
        status, df = _dataset_status(alias, current_ts)
        dataset_statuses.append(status)
        dataset_frames[alias] = df

    dataset_map = {ds.dataset: ds for ds in dataset_statuses}
    prod_exog = _build_prod_exog(dataset_frames)
    group_statuses = [
        _group_status(group, dataset_map, dataset_frames, prod_exog, current_ts)
        for group in VARIABLE_GROUPS
    ]

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
