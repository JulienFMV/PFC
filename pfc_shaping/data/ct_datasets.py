from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REGISTRY_PATH = Path(__file__).resolve().parent / "ct_dataset_registry.yaml"

ENTSO_FUNDAMENTALS_COLUMNS = [
    "load_mw",
    "solar_mw",
    "wind_mw",
    "cross_border_mw",
    "nuclear_mw",
    "hydro_ror_mw",
    "hydro_reservoir_mw",
    "hydro_pumped_mw",
    "solar_regime",
    "load_deviation",
    "flow_deviation",
    "load_de_mw",
    "solar_de_mw",
    "wind_de_mw",
    "residual_load_de_mw",
    "load_fr_mw",
    "solar_fr_mw",
    "wind_fr_mw",
    "load_at_mw",
    "solar_at_mw",
    "wind_at_mw",
    "load_it_mw",
    "solar_it_mw",
    "wind_it_mw",
    "fr_nuclear_unavailable_mw",
    "fr_nuclear_unavailability_ratio",
    "fr_nuclear_stress_flag",
]

ENTSO_BORDER_COLUMNS = [
    "scheduled_net_export_ch_de_mw",
    "flow_net_export_ch_de_mw",
    "ntc_export_ch_de_mw",
    "ntc_import_ch_de_mw",
    "ntc_net_ch_de_mw",
    "ntc_total_ch_de_mw",
    "scheduled_net_export_ch_fr_mw",
    "flow_net_export_ch_fr_mw",
    "ntc_export_ch_fr_mw",
    "ntc_import_ch_fr_mw",
    "ntc_net_ch_fr_mw",
    "ntc_total_ch_fr_mw",
    "scheduled_net_export_ch_at_mw",
    "flow_net_export_ch_at_mw",
    "ntc_export_ch_at_mw",
    "ntc_import_ch_at_mw",
    "ntc_net_ch_at_mw",
    "ntc_total_ch_at_mw",
    "scheduled_net_export_ch_it_mw",
    "flow_net_export_ch_it_mw",
    "ntc_export_ch_it_mw",
    "ntc_import_ch_it_mw",
    "ntc_net_ch_it_mw",
    "ntc_total_ch_it_mw",
    "scheduled_net_export_ch_de_mw_zscore",
    "ntc_total_ch_de_mw_zscore",
    "ntc_net_ch_de_mw_zscore",
    "scheduled_net_export_ch_fr_mw_zscore",
    "ntc_total_ch_fr_mw_zscore",
    "ntc_net_ch_fr_mw_zscore",
    "scheduled_net_export_ch_at_mw_zscore",
    "ntc_total_ch_at_mw_zscore",
    "ntc_net_ch_at_mw_zscore",
    "scheduled_net_export_ch_it_mw_zscore",
    "ntc_total_ch_it_mw_zscore",
    "ntc_net_ch_it_mw_zscore",
    "ntc_balance_ch_de",
    "ntc_balance_ch_fr",
    "ntc_balance_ch_at",
    "ntc_balance_ch_it",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    layer: str
    domain: str
    source_system: str
    local_cache: str | None
    target_table: str
    grain: str
    time_key: str
    freshness_sla_days: int
    required: bool
    consumed_by_prod: bool
    metadata: dict[str, Any]


def load_registry() -> dict[str, Any]:
    with REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_dataset_spec(name: str) -> DatasetSpec:
    registry = load_registry()
    raw = registry.get("datasets", {}).get(name)
    if raw is None:
        raise KeyError(f"Unknown CT dataset '{name}' in registry {REGISTRY_PATH}")

    known = {
        "layer",
        "domain",
        "source_system",
        "local_cache",
        "target_table",
        "grain",
        "time_key",
        "freshness_sla_days",
        "required",
        "consumed_by_prod",
    }
    metadata = {k: v for k, v in raw.items() if k not in known}
    return DatasetSpec(
        name=name,
        layer=raw["layer"],
        domain=raw["domain"],
        source_system=raw["source_system"],
        local_cache=raw.get("local_cache"),
        target_table=raw["target_table"],
        grain=raw["grain"],
        time_key=raw["time_key"],
        freshness_sla_days=int(raw["freshness_sla_days"]),
        required=bool(raw["required"]),
        consumed_by_prod=bool(raw["consumed_by_prod"]),
        metadata=metadata,
    )


def resolve_local_cache_path(project_root: str | Path, dataset_name: str) -> Path | None:
    spec = get_dataset_spec(dataset_name)
    if not spec.local_cache:
        return None
    return Path(project_root) / spec.local_cache


def load_local_dataset(project_root: str | Path, dataset_name: str, required: bool | None = None) -> pd.DataFrame | None:
    spec = get_dataset_spec(dataset_name)
    path = resolve_local_cache_path(project_root, dataset_name)
    if path is None or not path.exists():
        if required if required is not None else spec.required:
            raise FileNotFoundError(f"Missing required dataset {dataset_name}: {path}")
        return None
    df = pd.read_parquet(path)
    if df.empty and (required if required is not None else spec.required):
        raise ValueError(f"Required dataset {dataset_name} is empty: {path}")
    return df


def load_neighbor_price_frames(project_root: str | Path) -> dict[str, pd.DataFrame]:
    mapping = {
        "de": "ct_price_15min_de",
        "fr": "ct_price_15min_fr",
        "at": "ct_price_15min_at",
        "it": "ct_price_15min_it",
    }
    frames: dict[str, pd.DataFrame] = {}
    for code, dataset_name in mapping.items():
        df = load_local_dataset(project_root, dataset_name, required=(code == "de"))
        if df is not None and not df.empty:
            frames[code] = df
    return frames


def split_entso_views(entso_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fundamentals = entso_df[[c for c in ENTSO_FUNDAMENTALS_COLUMNS if c in entso_df.columns]].copy()
    border = entso_df[[c for c in ENTSO_BORDER_COLUMNS if c in entso_df.columns]].copy()
    return fundamentals, border
