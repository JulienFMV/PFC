"""Governed electrification scenario data loader.

The LT electrification shape model consumes only vintage-safe scenario rows.
This module owns the production data contract: schema validation, date
normalisation, optional Databricks loading, and strict as-of filtering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from pfc_shaping.data.databricks_client import query_to_df, table_fqn

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ELECTRIFICATION_SCENARIO_PATH = _REPO_ROOT / "data" / "electrification_scenarios.parquet"
DEFAULT_HPFC_SCENARIO_FEATURES_PATH = _REPO_ROOT / "data" / "hpfc_scenario_features.parquet"

REQUIRED_COLUMNS = {
    "publication_date",
    "scenario",
    "country",
    "delivery_year",
}

RECOMMENDED_COLUMNS = {
    "source",
    "delivery_month",
    "scenario_weight",
    "quality_flag",
    "demand_twh",
    "peak_load_gw",
    "winter_demand_twh",
    "pv_gw",
    "pv_twh",
    "wind_gw",
    "wind_twh",
    "battery_power_gw",
    "battery_energy_gwh",
    "ev_twh",
    "heatpump_twh",
    "managed_charging_share",
    "hydro_twh",
    "hydro_capacity_gw",
    "hydro_reservoir_twh",
    "nuclear_gw",
    "dispatchable_gw",
    "gas_gw",
    "coal_gw",
    "import_twh",
    "export_twh",
    "net_import_twh",
    "ntc_ch_de_gw",
    "ntc_ch_fr_gw",
    "ntc_ch_it_gw",
}

PRODUCTION_COLUMNS = RECOMMENDED_COLUMNS | {
    "measurement_date",
    "track",
    "scenario_edition",
    "ingested_at_utc",
    "ntc_ch_at_gw",
    "gas_eur_mwh",
    "coal_eur_mwh",
    "co2_eur_t",
    "electrolysis_twh",
    "p2x_gw",
    "dsm_gw",
}

CROSS_BORDER_BALANCE_COLUMNS = {"import_twh", "export_twh", "net_import_twh"}

CONDITIONAL_PRODUCTION_FIELD_GROUPS = {
    # The model consumes net import dependency. Gross import/export are useful
    # diagnostics, but a governed net_import_twh is sufficient and avoids
    # inventing gross flows where public sources only publish net balance.
    "cross_border_balance": (("net_import_twh",), ("import_twh", "export_twh")),
}

COUNTRY_SCOPED_CRITICAL_VALUE_COLUMNS = {
    # Reservoir energy capacity is a CH hydro flexibility driver for the PFC.
    # For neighbouring countries, missing reservoir capacity is governed as an
    # explicit neutralisation unless a future source component supplies it.
    "hydro_reservoir_twh": {"CH"},
}

OPTIONAL_PRODUCTION_NULL_COLUMNS = {
    "delivery_month",
    "measurement_date",
} | CROSS_BORDER_BALANCE_COLUMNS

CRITICAL_PRODUCTION_VALUE_COLUMNS = PRODUCTION_COLUMNS - OPTIONAL_PRODUCTION_NULL_COLUMNS

UNACCEPTABLE_PRODUCTION_QUALITY_TOKENS = {
    "fallback",
    "internal",
    "partial",
    "proxy",
    "synthetic",
}

NONNEGATIVE_PREFIXES = ("ntc_",)
NONNEGATIVE_COLUMNS = {
    "scenario_weight",
    "demand_twh",
    "peak_load_gw",
    "winter_demand_twh",
    "pv_gw",
    "pv_twh",
    "wind_gw",
    "wind_twh",
    "battery_power_gw",
    "battery_energy_gwh",
    "ev_twh",
    "heatpump_twh",
    "hydro_twh",
    "hydro_capacity_gw",
    "hydro_reservoir_twh",
    "nuclear_gw",
    "dispatchable_gw",
    "gas_gw",
    "coal_gw",
    "import_twh",
    "export_twh",
    "gas_eur_mwh",
    "coal_eur_mwh",
    "co2_eur_t",
    "electrolysis_twh",
    "p2x_gw",
    "dsm_gw",
}
SIGNED_NUMERIC_COLUMNS = {
    "net_import_twh",
}
BOUNDED_SHARE_COLUMNS = {
    "managed_charging_share",
}
HPFC_SCENARIO_FEATURE_COLUMNS = [
    "pv_penetration",
    "wind_penetration",
    "battery_penetration",
    "battery_power_penetration",
    "battery_energy_cover_h",
    "ev_penetration",
    "heatpump_penetration",
    "hydro_flexibility",
    "hydro_energy_share",
    "hydro_capacity_penetration",
    "hydro_reservoir_cover",
    "hydro_reservoir_available",
    "import_dependency",
    "ntc_penetration",
    "firm_capacity_penetration",
    "managed_charging_share",
]

__all__ = [
    "DEFAULT_ELECTRIFICATION_SCENARIO_PATH",
    "DEFAULT_HPFC_SCENARIO_FEATURES_PATH",
    "REQUIRED_COLUMNS",
    "RECOMMENDED_COLUMNS",
    "PRODUCTION_COLUMNS",
    "CONDITIONAL_PRODUCTION_FIELD_GROUPS",
    "COUNTRY_SCOPED_CRITICAL_VALUE_COLUMNS",
    "CRITICAL_PRODUCTION_VALUE_COLUMNS",
    "UNACCEPTABLE_PRODUCTION_QUALITY_TOKENS",
    "HPFC_SCENARIO_FEATURE_COLUMNS",
    "validate_electrification_scenarios",
    "interpolate_electrification_scenario_years",
    "derive_hpfc_scenario_features",
    "write_hpfc_scenario_features",
    "load_electrification_scenarios",
    "load_electrification_scenarios_from_databricks",
    "asof_electrification_scenarios",
    "assert_scenario_coverage",
]


def _to_utc_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError("vintage/publication_date cannot be NaT")
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in NONNEGATIVE_COLUMNS if c in df.columns]
    cols.extend(c for c in df.columns if str(c).startswith(NONNEGATIVE_PREFIXES) and c not in cols)
    cols.extend(c for c in SIGNED_NUMERIC_COLUMNS if c in df.columns and c not in cols)
    return cols


def _safe_float(row: pd.Series, column: str, default: float = 0.0) -> float:
    if column not in row or pd.isna(row[column]):
        return default
    value = float(row[column])
    return value if np.isfinite(value) else default


def _optional_float(row: pd.Series, column: str) -> float:
    if column not in row or pd.isna(row[column]):
        return float("nan")
    value = float(row[column])
    return value if np.isfinite(value) else float("nan")


def _clip_feature(value: float, lo: float = -2.0, hi: float = 2.0) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, lo, hi))


def validate_electrification_scenarios(
    frame: pd.DataFrame,
    *,
    require_recommended: bool = False,
) -> pd.DataFrame:
    """Validate and normalize an electrification scenario table."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise KeyError(f"electrification scenarios missing required columns {sorted(missing)}")
    if require_recommended:
        missing_recommended = RECOMMENDED_COLUMNS - set(frame.columns)
        if missing_recommended:
            raise KeyError(
                "electrification scenarios missing recommended production columns "
                f"{sorted(missing_recommended)}"
            )

    df = frame.copy()
    df["publication_date"] = pd.to_datetime(df["publication_date"], utc=True)
    if df["publication_date"].isna().any():
        raise ValueError("publication_date contains NaT")
    if "measurement_date" in df.columns:
        df["measurement_date"] = pd.to_datetime(df["measurement_date"], utc=True, errors="coerce")
    if "ingested_at_utc" in df.columns:
        df["ingested_at_utc"] = pd.to_datetime(df["ingested_at_utc"], utc=True, errors="coerce")
        if df["ingested_at_utc"].isna().any():
            raise ValueError("ingested_at_utc contains NaT")
    if "track" in df.columns:
        df["track"] = df["track"].fillna("scenario").astype(str).str.lower()
        bad_track = ~df["track"].isin(["actual", "scenario"])
        if bad_track.any():
            raise ValueError("track must be one of ['actual', 'scenario']")
        if "measurement_date" in df.columns:
            actual_missing_measurement = (df["track"] == "actual") & df["measurement_date"].isna()
            if actual_missing_measurement.any():
                raise ValueError("actual rows require measurement_date")
    df["scenario"] = df["scenario"].astype(str)
    df["country"] = df["country"].astype(str)
    df["delivery_year"] = pd.to_numeric(df["delivery_year"], errors="raise").astype(int)
    if "delivery_month" in df.columns:
        month = pd.to_numeric(df["delivery_month"], errors="coerce")
        bad = month.notna() & ~month.between(1, 12)
        if bad.any():
            raise ValueError("delivery_month must be null or in [1, 12]")
        df["delivery_month"] = month

    for col in _numeric_columns(df):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if col in SIGNED_NUMERIC_COLUMNS:
            bad = df[col].notna() & ~np.isfinite(df[col])
            msg = f"{col} must be finite"
        else:
            bad = df[col].notna() & (~np.isfinite(df[col]) | (df[col] < 0.0))
            msg = f"{col} must be finite and non-negative"
        if bad.any():
            raise ValueError(msg)
    if "scenario_weight" in df.columns:
        weight = pd.to_numeric(df["scenario_weight"], errors="coerce")
        bad = weight.notna() & ~np.isfinite(weight)
        if bad.any():
            raise ValueError("scenario_weight must be finite when present")
        df["scenario_weight"] = weight
    for col in BOUNDED_SHARE_COLUMNS & set(df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad = df[col].notna() & (~np.isfinite(df[col]) | (df[col] < 0.0) | (df[col] > 1.0))
        if bad.any():
            raise ValueError(f"{col} must be finite and in [0, 1]")
    return df


def interpolate_electrification_scenario_years(
    frame: pd.DataFrame,
    years: Iterable[int],
    *,
    allow_clamp: bool = True,
) -> pd.DataFrame:
    """Expand scenario rows to requested delivery years by linear interpolation.

    EP2050+/TYNDP scenario tables are often published on a 5-year grid. The PFC
    consumes yearly delivery periods, so this function derives the intermediate
    yearly rows while preserving publication metadata and scenario provenance.
    Numeric physical columns are interpolated within each
    `(publication_date, source, scenario, country)` group. Years outside the
    published range are clamped to the nearest available row only when
    `allow_clamp=True`.
    """
    df = validate_electrification_scenarios(frame)
    target_years = sorted({int(y) for y in years})
    if not target_years:
        return df.iloc[0:0].copy()

    group_cols = [c for c in ["publication_date", "source", "scenario", "country"] if c in df.columns]
    numeric_cols = [
        c
        for c in df.columns
        if c not in set(group_cols) | {"delivery_year", "delivery_month"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    passthrough_cols = [
        c
        for c in df.columns
        if c not in set(group_cols) | {"delivery_year", "delivery_month"} | set(numeric_cols)
    ]

    rows: list[dict[str, object]] = []
    for group_key, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        meta = dict(zip(group_cols, group_key))
        sub = sub.sort_values("delivery_year").drop_duplicates("delivery_year", keep="last")
        x = sub["delivery_year"].astype(int).to_numpy()
        if not allow_clamp and (target_years[0] < int(x.min()) or target_years[-1] > int(x.max())):
            label = "/".join(str(meta.get(c, "")) for c in group_cols)
            raise ValueError(
                "scenario interpolation requires bracketing published years "
                f"for {label}: requested {target_years[0]}-{target_years[-1]}, "
                f"available {int(x.min())}-{int(x.max())}"
            )
        exact_years = set(int(y) for y in x)
        for year in target_years:
            out = dict(meta)
            out["delivery_year"] = int(year)
            out["delivery_month"] = pd.NA
            for col in numeric_cols:
                y = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(y)
                out[col] = float(np.interp(year, x[valid], y[valid])) if valid.any() else np.nan
            for col in passthrough_cols:
                values = sub[col].dropna()
                out[col] = values.iloc[-1] if len(values) else pd.NA
            if "quality_flag" in df.columns and year not in exact_years:
                base_quality = str(out.get("quality_flag", "official"))
                out["quality_flag"] = (
                    base_quality
                    if "interpolated" in base_quality
                    else f"{base_quality}_interpolated"
                )
            rows.append(out)
    return validate_electrification_scenarios(pd.DataFrame(rows))


def derive_hpfc_scenario_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive normalized HPFC structural indicators from scenario rows.

    The output is the model-facing "gold" layer: raw GW/TWh assumptions are
    preserved upstream, while the curve model consumes penetration-style
    features that are comparable across delivery years and scenarios.
    """
    df = validate_electrification_scenarios(frame)
    rows: list[dict[str, object]] = []
    meta_cols = [
        "publication_date",
        "source",
        "scenario",
        "country",
        "delivery_year",
        "delivery_month",
        "scenario_weight",
        "quality_flag",
    ]

    for _, row in df.iterrows():
        demand_twh = max(_safe_float(row, "demand_twh", 60.0), 1e-6)
        avg_load_gw = max(demand_twh * 1000.0 / 8760.0, 1e-6)
        peak_load_gw = max(_safe_float(row, "peak_load_gw", avg_load_gw), avg_load_gw)
        winter_demand_twh = max(_safe_float(row, "winter_demand_twh", demand_twh), 1e-6)

        pv_twh = _safe_float(row, "pv_twh", float("nan"))
        wind_twh = _safe_float(row, "wind_twh", float("nan"))
        pv_penetration = (
            pv_twh / demand_twh
            if np.isfinite(pv_twh)
            else _safe_float(row, "pv_gw") / avg_load_gw
        )
        wind_penetration = (
            wind_twh / demand_twh
            if np.isfinite(wind_twh)
            else _safe_float(row, "wind_gw") / avg_load_gw
        )

        battery_power_penetration = _safe_float(row, "battery_power_gw") / avg_load_gw
        battery_energy_cover_h = _safe_float(row, "battery_energy_gwh") / avg_load_gw
        battery_penetration = 0.5 * battery_power_penetration + 0.5 * min(
            battery_energy_cover_h / 4.0,
            2.0,
        )

        hydro_energy_share = _safe_float(row, "hydro_twh") / demand_twh
        hydro_capacity_penetration = _safe_float(row, "hydro_capacity_gw") / peak_load_gw
        reservoir_twh = _optional_float(row, "hydro_reservoir_twh")
        hydro_reservoir_available = float(np.isfinite(reservoir_twh))
        hydro_reservoir_cover = reservoir_twh / winter_demand_twh if np.isfinite(reservoir_twh) else 0.0
        hydro_terms = [
            (0.20, hydro_energy_share),
            (0.45, hydro_capacity_penetration),
        ]
        if np.isfinite(reservoir_twh):
            hydro_terms.append((0.35, hydro_reservoir_cover))
        hydro_weight = sum(weight for weight, _ in hydro_terms)
        hydro_flexibility = sum(weight * value for weight, value in hydro_terms) / max(hydro_weight, 1e-12)

        net_import_twh = _safe_float(
            row,
            "net_import_twh",
            _safe_float(row, "import_twh") - _safe_float(row, "export_twh"),
        )
        ntc_gw = sum(
            _safe_float(row, column)
            for column in row.index
            if str(column).startswith("ntc_") and str(column).endswith("_gw")
        )
        firm_capacity_gw = (
            _safe_float(row, "nuclear_gw")
            + _safe_float(row, "dispatchable_gw")
            + _safe_float(row, "gas_gw")
            + _safe_float(row, "coal_gw")
        )

        out = {column: row[column] for column in meta_cols if column in row.index}
        out.update(
            {
                "avg_load_gw": avg_load_gw,
                "peak_load_gw": peak_load_gw,
                "pv_penetration": _clip_feature(pv_penetration),
                "wind_penetration": _clip_feature(wind_penetration),
                "battery_penetration": _clip_feature(battery_penetration),
                "battery_power_penetration": _clip_feature(battery_power_penetration),
                "battery_energy_cover_h": _clip_feature(battery_energy_cover_h, hi=48.0),
                "ev_penetration": _clip_feature(_safe_float(row, "ev_twh") / demand_twh),
                "heatpump_penetration": _clip_feature(_safe_float(row, "heatpump_twh") / demand_twh),
                "hydro_flexibility": _clip_feature(hydro_flexibility),
                "hydro_energy_share": _clip_feature(hydro_energy_share),
                "hydro_capacity_penetration": _clip_feature(hydro_capacity_penetration),
                "hydro_reservoir_cover": _clip_feature(hydro_reservoir_cover),
                "hydro_reservoir_available": hydro_reservoir_available,
                "import_dependency": _clip_feature(net_import_twh / demand_twh, lo=-2.0, hi=2.0),
                "ntc_penetration": _clip_feature(ntc_gw / peak_load_gw),
                "firm_capacity_penetration": _clip_feature(firm_capacity_gw / peak_load_gw),
                "managed_charging_share": _clip_feature(
                    _safe_float(row, "managed_charging_share"),
                    lo=0.0,
                    hi=1.0,
                ),
            }
        )
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=meta_cols + ["avg_load_gw", "peak_load_gw"] + HPFC_SCENARIO_FEATURE_COLUMNS)
    return pd.DataFrame(rows)


def write_hpfc_scenario_features(
    frame: pd.DataFrame,
    path: str | Path = DEFAULT_HPFC_SCENARIO_FEATURES_PATH,
) -> pd.DataFrame:
    """Derive and write the gold HPFC scenario feature table."""
    features = derive_hpfc_scenario_features(frame)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path, index=False)
    return features


def load_electrification_scenarios(
    path: str | Path = DEFAULT_ELECTRIFICATION_SCENARIO_PATH,
    *,
    require_recommended: bool = False,
) -> pd.DataFrame:
    """Load scenario rows from parquet or csv and validate schema."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    return validate_electrification_scenarios(df, require_recommended=require_recommended)


def load_electrification_scenarios_from_databricks(
    *,
    table_key: str = "electrification_scenarios",
    config: dict | None = None,
    where_sql: str | None = None,
    require_recommended: bool = False,
) -> pd.DataFrame:
    """Load scenario rows from a configured Databricks table.

    `where_sql` is appended verbatim after `WHERE`; callers should keep it static
    or parameterized upstream.
    """
    fqn = table_fqn(table_key, config=config)
    sql = f"SELECT * FROM {fqn}"
    if where_sql:
        sql += f" WHERE {where_sql}"
    return validate_electrification_scenarios(
        query_to_df(sql, config=config),
        require_recommended=require_recommended,
    )


def asof_electrification_scenarios(frame: pd.DataFrame, vintage) -> pd.DataFrame:
    """Return rows known on or before `vintage`.

    Scenario rows are vintage-safe when `publication_date <= vintage`. Actual
    rows also require `measurement_date <= vintage`; a future-measured actual is
    not known at the pricing vintage even if the file was ingested/published.
    """
    v = _to_utc_timestamp(vintage)
    df = validate_electrification_scenarios(frame)
    mask = df["publication_date"] <= v
    if {"track", "measurement_date"}.issubset(df.columns):
        actual = df["track"].astype(str).str.lower() == "actual"
        mask = mask & (~actual | (df["measurement_date"] <= v))
    return df[mask].copy()


def assert_scenario_coverage(
    frame: pd.DataFrame,
    *,
    country: str,
    scenarios: Iterable[str],
    periods: Iterable[pd.Period],
) -> None:
    """Raise if country/scenario/year/month coverage is incomplete."""
    df = validate_electrification_scenarios(frame)
    missing: list[str] = []
    for scenario in scenarios:
        for period in periods:
            p = pd.Period(period, freq="M")
            mask = (
                (df["country"].astype(str) == str(country))
                & (df["scenario"].astype(str) == str(scenario))
                & (df["delivery_year"].astype(int) == int(p.year))
            )
            if "delivery_month" in df.columns:
                month = pd.to_numeric(df["delivery_month"], errors="coerce")
                mask = mask & (month.isna() | (month == int(p.month)))
            if not bool(mask.any()):
                missing.append(f"{country}/{scenario}/{p}")
    if missing:
        raise KeyError(f"missing electrification scenario coverage: {missing[:10]}")
