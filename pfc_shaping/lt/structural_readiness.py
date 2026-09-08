"""Read-only structural input diagnostics and signed hourly shape arithmetic.

These functions neither admit scenarios nor assemble a curve. Annual field
coverage cannot establish chronological feasibility or source authenticity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pfc_shaping.lt.curve_products import AuthorityNegative
from pfc_shaping.lt.model.shape_constraints import validate_utc_index


STRUCTURAL_GROUPS = {
    "demand": ("demand_twh", "ev_twh", "heatpump_twh"),
    "renewables": ("pv_gw", "wind_gw"),
    "battery_size": ("battery_power_gw", "battery_energy_gwh"),
    "battery_operation": (
        "battery_charge_efficiency", "battery_discharge_efficiency",
        "battery_initial_soc_share", "battery_terminal_soc_share",
    ),
    "flexibility": ("managed_charging_share", "dsm_gw"),
    "demand_shifting": ("dsm_max_shift_hours",),
    "hydro": ("hydro_capacity_gw", "hydro_reservoir_twh"),
    "firm_supply": ("nuclear_gw", "gas_gw", "coal_gw"),
    "costs": ("gas_eur_mwh", "coal_eur_mwh", "co2_eur_t"),
    "ch_interconnection": (
        "ntc_ch_de_gw", "ntc_ch_fr_gw", "ntc_ch_it_gw", "ntc_ch_at_gw",
    ),
}
_SHARES = {
    "managed_charging_share", "battery_initial_soc_share",
    "battery_terminal_soc_share",
}
_POSITIVE = {
    "demand_twh", "battery_charge_efficiency",
    "battery_discharge_efficiency", "dsm_max_shift_hours",
}


def _field_status(value: object, field: str) -> str:
    if value is None or pd.isna(value):
        return "MISSING"
    if isinstance(value, (bool, np.bool_)):
        return "INVALID"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "INVALID"
    if not np.isfinite(number) or number < 0 or (field in _POSITIVE and number == 0):
        return "INVALID"
    if (field in _SHARES or field.endswith("_efficiency")) and number > 1:
        return "INVALID"
    return "PRESENT_ZERO" if number == 0 else "PRESENT"


def audit_annual_inventory(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    countries: tuple[str, ...],
    years: tuple[int, ...],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Inventory exact annual rows without merging editions or filling gaps.

    Each source is audited separately. Labels are retained verbatim. Duplicate
    country/scenario/year rows are ambiguous, even with different release dates.
    Publication/ingestion fields are declarations, not verified availability.
    """
    required = {"country", "scenario", "delivery_year", "publication_date"}
    if required - set(frame):
        raise ValueError(f"missing inventory identity columns: {sorted(required - set(frame))}")
    origin = pd.Timestamp(as_of)
    if pd.isna(origin) or origin.tz is None:
        raise ValueError("as_of must be timezone-aware")
    if not countries or len(set(countries)) != len(countries):
        raise ValueError("countries must be nonempty and unique")
    if not years or any(type(year) is not int or year < 1900 for year in years):
        raise ValueError("years must be explicit integer delivery years")
    if len(set(years)) != len(years):
        raise ValueError("years must be unique")
    data = frame.copy(deep=True)
    for key in ("country", "scenario"):
        if data[key].isna().any() or data[key].astype(str).str.strip().eq("").any():
            raise ValueError(f"missing {key} identity")
    numeric_years = pd.to_numeric(data["delivery_year"], errors="coerce")
    if (numeric_years.isna().any() or not np.isfinite(numeric_years).all()
            or (numeric_years % 1 != 0).any()
            or data["delivery_year"].map(lambda v: isinstance(v, (bool, np.bool_))).any()):
        raise ValueError("delivery_year must contain finite integers")
    data["delivery_year"] = numeric_years.astype(int)
    published = pd.to_datetime(data["publication_date"], utc=True, errors="coerce", format="mixed")
    ingested = pd.to_datetime(
        data.get("ingested_at_utc", pd.Series(pd.NaT, index=data.index)),
        utc=True, errors="coerce", format="mixed",
    )
    annual = (data["delivery_month"].isna() if "delivery_month" in data
              else pd.Series(True, index=data.index))
    scenario_track = (data["track"].eq("scenario") if "track" in data
                      else pd.Series(False, index=data.index))
    available = published.notna() & published.le(origin) & (ingested.isna() | ingested.le(origin))
    selected = data.loc[annual & available].copy()
    selected["_ingestion_missing"] = ingested.loc[selected.index].isna()
    selected["_scenario_track"] = scenario_track.loc[selected.index]
    rows = []
    for scenario in sorted(data["scenario"].unique()):
        for country in countries:
            for year in years:
                cell = selected.loc[
                    selected["scenario"].eq(scenario) & selected["country"].eq(country)
                    & selected["delivery_year"].eq(year)
                ]
                identity = {"scenario": scenario, "country": country, "delivery_year": year}
                status = "PRESENT" if len(cell) == 1 else "NO_EXACT_ROW" if cell.empty else "AMBIGUOUS"
                row = cell.iloc[0] if len(cell) == 1 else None
                flags = str(row.get("quality_flag", "")) if row is not None else ""
                for group, fields in STRUCTURAL_GROUPS.items():
                    inactive = False
                    if row is not None:
                        # A declared zero asset needs no operational parameters.
                        inactive = (
                            group == "ch_interconnection" and country != "CH"
                            or group == "battery_operation"
                            and _field_status(row.get("battery_power_gw"), "battery_power_gw") == "PRESENT_ZERO"
                            and _field_status(row.get("battery_energy_gwh"), "battery_energy_gwh") == "PRESENT_ZERO"
                            or group == "demand_shifting"
                            and _field_status(row.get("dsm_gw"), "dsm_gw") == "PRESENT_ZERO"
                        )
                    for field in fields:
                        field_status = status if row is None else (
                            "NOT_APPLICABLE" if inactive else _field_status(row.get(field), field)
                        )
                        rows.append({
                            **identity, "group": group, "field": field,
                            "row_status": status, "field_status": field_status,
                            "quality_flag": flags,
                            "ingestion_timestamp_missing": bool(row["_ingestion_missing"]) if row is not None else True,
                            "scenario_track_declared": bool(row["_scenario_track"]) if row is not None else False,
                        })
    result = pd.DataFrame(rows)
    battery_conflicts = 0
    for _, row in selected.iterrows():
        power = _field_status(row.get("battery_power_gw"), "battery_power_gw")
        energy = _field_status(row.get("battery_energy_gwh"), "battery_energy_gwh")
        battery_conflicts += int({power, energy} == {"PRESENT", "PRESENT_ZERO"})
    report = {
        "status": "DIAGNOSTIC_ONLY_NOT_SCENARIO_ADMISSION",
        "rows_read": len(data), "annual_rows": int(annual.sum()),
        "publication_missing_or_invalid": int(published.isna().sum()),
        "publication_after_origin": int(published.gt(origin).sum()),
        "ingestion_missing_or_invalid": int(ingested.isna().sum()),
        "ingestion_after_origin": int(ingested.gt(origin).sum()),
        "battery_size_conflicts": battery_conflicts,
        "source_labels": sorted(data.get("source", pd.Series(dtype=str)).dropna().astype(str).unique()),
        "scenario_labels": sorted(data["scenario"].unique()),
        "quality_flags": sorted(data.get("quality_flag", pd.Series(dtype=str)).dropna().astype(str).unique()),
        "field_status_counts": {str(k): int(v) for k, v in result.get("field_status", pd.Series(dtype=str)).value_counts().items()},
        "unverified_requirements": [
            "source_authenticity_and_historical_availability",
            "source_to_fmv_scenario_mapping", "annual_to_hourly_weather_mapping",
            "chronological_hydro_and_storage_operation", "market_bidding_and_negative_prices",
        ],
        "authority": AuthorityNegative().to_manifest(),
    }
    return result, report


def center_signed_hourly_shape(prices: pd.Series) -> pd.Series:
    """Return signed EUR/MWh deviations over whole Swiss delivery months.

    Input must be native hourly observations or explicitly modelled hourly
    prices. This is arithmetic only: it neither grants a vendor calendar nor
    converts hourly observations into independent quarter-hour truth. When
    used to construct historical targets, the caller must close every training
    month before its origin; future realized monthly means are not predictors.
    """
    if not isinstance(prices, pd.Series) or prices.empty:
        raise ValueError("prices must be a nonempty Series")
    index = validate_utc_index(prices.index).as_unit("ns")
    local = index.tz_convert("Europe/Zurich")
    first_month = pd.Period(local[0].strftime("%Y-%m"), freq="M")
    last_month = pd.Period(local[-1].strftime("%Y-%m"), freq="M")
    start = first_month.start_time.tz_localize("Europe/Zurich")
    end = (last_month + 1).start_time.tz_localize("Europe/Zurich")
    expected = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
    if not index.equals(expected):
        raise ValueError("shape requires complete consecutive Swiss months at native hourly cadence")
    if pd.api.types.is_bool_dtype(prices.dtype):
        raise ValueError("prices must be finite numeric EUR/MWh values")
    try:
        values = prices.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("prices must be finite numeric EUR/MWh values") from exc
    if not np.isfinite(values).all():
        raise ValueError("prices must be finite numeric EUR/MWh values")
    raw = pd.Series(values, index=index)
    months = local.strftime("%Y-%m")
    shape = (raw - raw.groupby(months).transform("mean")).rename("shape_eur_mwh")
    if not np.isfinite(shape).all() or shape.groupby(months).mean().abs().max() > 1e-9:
        raise ValueError("signed shape failed monthly numerical neutrality")
    return shape
