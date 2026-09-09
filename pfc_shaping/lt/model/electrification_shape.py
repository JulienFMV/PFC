"""Scenario-driven electrification shape modulation for LT HPFC.

This module is an additive Phase 13 layer. It converts versioned long-term
electrification scenarios into a conservative block-level correction on ``f_H``.
All scenario access is as-of filtered by ``publication_date <= vintage``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.data.electrification_scenarios import (
    CONDITIONAL_PRODUCTION_FIELD_GROUPS,
    COUNTRY_SCOPED_CRITICAL_VALUE_COLUMNS,
    CRITICAL_PRODUCTION_VALUE_COLUMNS,
    PRODUCTION_COLUMNS,
    REQUIRED_COLUMNS,
    UNACCEPTABLE_PRODUCTION_QUALITY_TOKENS,
    validate_electrification_scenarios,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SCENARIO_PATH",
    "BLOCK_NAMES",
    "ElectrificationScenarioStore",
    "StructuralDriverProjector",
    "ElectrificationFHCorrection",
    "apply_electrification_scenarios",
    "assert_production_scenario_inventory",
    "structural_fan_chart",
    "electrification_modulate",
]

DEFAULT_SCENARIO_PATH = None

BLOCK_NAMES = [
    "NIGHT",
    "MORNING_RAMP",
    "MIDDAY_BOWL",
    "AFTERNOON_SHOULDER",
    "EVENING_RAMP",
]

def _to_utc_timestamp(value) -> pd.Timestamp:
    if value is None:
        raise ValueError("vintage is required for leak-free electrification scenario filtering")
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError("vintage is required for leak-free electrification scenario filtering")
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _local_day_key(index: pd.DatetimeIndex, tz: str) -> pd.Index:
    local = index.tz_convert(tz)
    return pd.Index([f"{t.year}-{t.month:02d}-{t.day:02d}" for t in local], name="day")


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "WINTER"
    if month in (6, 7, 8, 9):
        return "SUMMER"
    return "SHOULDER"


def _block(hour: int) -> str:
    if hour in (6, 7, 8, 9):
        return "MORNING_RAMP"
    if hour in (10, 11, 12, 13, 14, 15):
        return "MIDDAY_BOWL"
    if hour in (16, 17):
        return "AFTERNOON_SHOULDER"
    if hour in (18, 19, 20, 21):
        return "EVENING_RAMP"
    return "NIGHT"


def _day_type(value: str) -> str:
    return "WORKDAY" if str(value) == "Ouvrable" else "WEEKEND"


def _safe_float(row: pd.Series, name: str, default: float = 0.0) -> float:
    if name not in row or pd.isna(row[name]):
        return default
    value = float(row[name])
    return value if np.isfinite(value) else default


def _normalize_weights(labels: Sequence[str], weights: Mapping[str, float] | None) -> dict[str, float]:
    if not labels:
        return {}
    if weights is None:
        return {label: 1.0 / len(labels) for label in labels}
    raw = {label: max(float(weights.get(label, 0.0)), 0.0) for label in labels}
    total = sum(raw.values())
    if total <= 0.0:
        return {label: 1.0 / len(labels) for label in labels}
    return {label: value / total for label, value in raw.items()}


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    order = np.argsort(values)
    x = values[order]
    w = weights[order]
    total = float(w.sum())
    if total <= 0.0:
        return float(np.quantile(x, q))
    cdf = np.cumsum(w) / total
    return float(x[np.searchsorted(cdf, q, side="left")])


def assert_production_scenario_inventory(
    frame: pd.DataFrame,
    *,
    required_countries: Sequence[str] = ("CH", "DE", "FR", "IT", "AT"),
    scenarios: Sequence[str] | None = None,
    years: Sequence[int] | None = None,
) -> None:
    """Raise when a scenario inventory is not acceptable for final production.

    This is intentionally stricter than the smoke/prod-readiness schema gate.
    Local EP2050 proxy rows can be used for diagnostics, but production runs
    must use governed multi-country rows with full provenance and no proxy flags.
    """
    missing = sorted(PRODUCTION_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"production electrification scenarios missing columns {missing}")
    critical_value_columns = sorted(CRITICAL_PRODUCTION_VALUE_COLUMNS & set(frame.columns))
    country_s = frame["country"].astype(str) if "country" in frame.columns else pd.Series("", index=frame.index)
    missing_values: dict[str, int] = {}
    for column in critical_value_columns:
        missing_mask = frame[column].isna()
        if column in COUNTRY_SCOPED_CRITICAL_VALUE_COLUMNS:
            missing_mask = missing_mask & country_s.isin(COUNTRY_SCOPED_CRITICAL_VALUE_COLUMNS[column])
        count = int(missing_mask.sum())
        if count > 0:
            missing_values[column] = count
    if missing_values:
        raise ValueError(
            "production electrification scenarios contain missing critical values "
            f"{missing_values}"
        )
    missing_groups: list[dict[str, object]] = []
    for group_name, alternatives in CONDITIONAL_PRODUCTION_FIELD_GROUPS.items():
        for _, row in frame.iterrows():
            ok = any(
                all(field in frame.columns and pd.notna(row.get(field)) for field in option)
                for option in alternatives
            )
            if not ok:
                missing_groups.append(
                    {
                        "group": group_name,
                        "country": row.get("country", ""),
                        "scenario": row.get("scenario", ""),
                        "delivery_year": row.get("delivery_year", ""),
                    }
                )
    if missing_groups:
        raise ValueError(
            "production electrification scenarios contain incomplete conditional field groups "
            f"{missing_groups[:20]}"
        )
    countries = set(frame["country"].astype(str).unique().tolist()) if len(frame) else set()
    missing_countries = [country for country in required_countries if country not in countries]
    if missing_countries:
        raise ValueError(f"production electrification scenarios missing countries {missing_countries}")
    if scenarios is not None and years is not None:
        missing_coverage: list[dict[str, object]] = []
        country_s = frame["country"].astype(str)
        scenario_s = frame["scenario"].astype(str)
        delivery_year_s = frame["delivery_year"].astype(int)
        for country in required_countries:
            for scenario in scenarios:
                for year in years:
                    has_row = bool(
                        (
                            (country_s == str(country))
                            & (scenario_s == str(scenario))
                            & (delivery_year_s == int(year))
                        ).any()
                    )
                    if not has_row:
                        missing_coverage.append(
                            {
                                "country": str(country),
                                "scenario": str(scenario),
                                "delivery_year": int(year),
                            }
                        )
        if missing_coverage:
            examples = missing_coverage[:20]
            raise ValueError(
                "production electrification scenarios missing country/scenario/year coverage "
                f"{examples}"
            )
    flags = frame["quality_flag"].fillna("").astype(str).str.lower()
    bad = flags.map(lambda value: any(token in value for token in UNACCEPTABLE_PRODUCTION_QUALITY_TOKENS))
    if bool(bad.any()):
        examples = (
            frame.loc[bad, ["country", "scenario", "delivery_year", "quality_flag"]]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(f"production electrification scenarios contain non-production quality flags {examples}")


class ElectrificationScenarioStore:
    """Versioned scenario table with strict vintage filtering.

    The store accepts parquet or csv inputs. ``publication_date`` is normalized
    to UTC and every ``asof`` view keeps only rows published on or before the
    requested vintage.
    """

    def __init__(self, path: str | Path | None = DEFAULT_SCENARIO_PATH, frame: pd.DataFrame | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._frame = self._load(frame)

    def _load(self, frame: pd.DataFrame | None) -> pd.DataFrame:
        if frame is not None:
            df = frame.copy()
        elif self.path is None or not self.path.exists():
            df = pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))
        elif self.path.suffix.lower() == ".csv":
            df = pd.read_csv(self.path)
        else:
            df = pd.read_parquet(self.path)
        return validate_electrification_scenarios(df)

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame.copy()

    def asof(self, vintage) -> "ElectrificationScenarioStore":
        v = _to_utc_timestamp(vintage)
        df = self._frame[self._frame["publication_date"] <= v].copy()
        return ElectrificationScenarioStore(path=None, frame=df)

    def get(self, *, country: str, scenario: str, delivery_period) -> pd.Series:
        period = pd.Period(delivery_period, freq="M")
        df = self._frame
        mask = (
            (df["country"].astype(str) == str(country))
            & (df["scenario"].astype(str) == str(scenario))
            & (df["delivery_year"].astype(int) == int(period.year))
        )
        if "delivery_month" in df.columns:
            month = pd.to_numeric(df["delivery_month"], errors="coerce")
            mask = mask & (month.isna() | (month.astype("Int64") == int(period.month)))
        sub = df[mask].sort_values("publication_date")
        if sub.empty:
            raise KeyError(
                f"no electrification scenario row for country={country!r}, "
                f"scenario={scenario!r}, delivery_period={period}"
            )
        exact = sub
        if "delivery_month" in sub.columns:
            month = pd.to_numeric(sub["delivery_month"], errors="coerce")
            exact = sub[month == int(period.month)]
            if exact.empty:
                exact = sub[month.isna()]
        return exact.sort_values("publication_date").iloc[-1]


@dataclass(frozen=True)
class StructuralDriverProjector:
    """Convert raw scenario rows to bounded structural features."""

    max_abs_feature: float = 2.0

    def transform_row(self, row: pd.Series) -> dict[str, float]:
        demand_twh = max(_safe_float(row, "demand_twh", 60.0), 1.0)
        avg_load_gw = max(demand_twh * 1000.0 / 8760.0, 1e-6)
        pv_pen = (
            _safe_float(row, "pv_penetration")
            if "pv_penetration" in row.index and not pd.isna(row.get("pv_penetration"))
            else (
                _safe_float(row, "pv_twh") / demand_twh
                if "pv_twh" in row.index and not pd.isna(row.get("pv_twh"))
                else _safe_float(row, "pv_gw") / demand_twh
            )
        )
        wind_pen = (
            _safe_float(row, "wind_penetration")
            if "wind_penetration" in row.index and not pd.isna(row.get("wind_penetration"))
            else (
                _safe_float(row, "wind_twh") / demand_twh
                if "wind_twh" in row.index and not pd.isna(row.get("wind_twh"))
                else _safe_float(row, "wind_gw") / demand_twh
            )
        )
        battery_energy_cover_h = _safe_float(row, "battery_energy_gwh") / avg_load_gw
        battery_power_share = _safe_float(row, "battery_power_gw") / avg_load_gw
        ev_share = _safe_float(row, "ev_twh") / demand_twh
        hp_share = _safe_float(row, "heatpump_twh") / demand_twh
        managed_charging_share = _safe_float(row, "managed_charging_share")
        hydro_energy_share = _safe_float(row, "hydro_twh") / demand_twh
        hydro_capacity_share = _safe_float(row, "hydro_capacity_gw") / avg_load_gw
        hydro_reservoir_cover_h = _safe_float(row, "hydro_reservoir_twh") * 1000.0 / avg_load_gw
        nuclear_share = _safe_float(row, "nuclear_gw") / avg_load_gw
        dispatchable_share = (
            _safe_float(row, "dispatchable_gw")
            + _safe_float(row, "gas_gw")
            + _safe_float(row, "coal_gw")
        ) / avg_load_gw
        ntc_share = sum(
            _safe_float(row, c)
            for c in row.index
            if str(c).startswith("ntc_") and str(c).endswith("_gw")
        ) / avg_load_gw
        raw = {
            "pv_pen": pv_pen,
            "wind_pen": wind_pen,
            "battery_energy_cover_h": battery_energy_cover_h / 4.0,
            "battery_power_share": battery_power_share,
            "ev_share": ev_share,
            "heatpump_share": hp_share,
            "managed_charging_share": managed_charging_share,
            "hydro_energy_share": hydro_energy_share,
            "hydro_capacity_share": hydro_capacity_share,
            "hydro_reservoir_cover_h": hydro_reservoir_cover_h / 500.0,
            "nuclear_share": nuclear_share,
            "dispatchable_share": dispatchable_share,
            "ntc_share": ntc_share,
        }
        return {
            k: float(np.clip(v, -self.max_abs_feature, self.max_abs_feature))
            for k, v in raw.items()
        }

    def transform(self, row: pd.Series, calendar_df: pd.DataFrame) -> pd.DataFrame:
        features = self.transform_row(row)
        return pd.DataFrame({k: features[k] for k in features}, index=calendar_df.index)


class ElectrificationFHCorrection:
    """Conservative block-level structural modulation of ``f_H``."""

    def __init__(
        self,
        *,
        scenario_store: ElectrificationScenarioStore,
        scenario: str = "central",
        country: str = "CH",
        tz: str = "Europe/Zurich",
        projector: StructuralDriverProjector | None = None,
    ) -> None:
        self.scenario_store = scenario_store
        self.scenario = scenario
        self.country = country
        self.tz = tz
        self.projector = projector or StructuralDriverProjector()
        self.diagnostics_: dict[str, object] = {"applied": False}

    def fit(self, *_, **__) -> "ElectrificationFHCorrection":
        return self

    def _adjustment(self, features: dict[str, float], *, season: str, day_type: str, block: str) -> float:
        pv = features["pv_pen"]
        batt = 0.5 * features["battery_energy_cover_h"] + 0.5 * features["battery_power_share"]
        ev = features["ev_share"]
        hp = features["heatpump_share"]
        managed = features["managed_charging_share"]
        wind = features["wind_pen"]
        hydro = 0.35 * features["hydro_energy_share"] + 0.45 * features["hydro_capacity_share"] + 0.20 * features["hydro_reservoir_cover_h"]
        firm = features["nuclear_share"] + features["dispatchable_share"]
        ntc = features["ntc_share"]

        adj = 0.0
        if block == "MIDDAY_BOWL":
            adj += -0.32 * pv
            adj += 0.18 * batt
            adj += 0.05 * ev
            adj += 0.03 * hydro
        elif block == "MORNING_RAMP":
            adj += -0.08 * pv
            if season == "WINTER":
                adj += 0.12 * hp
                adj += -0.04 * wind
                adj += -0.06 * hydro
        elif block == "AFTERNOON_SHOULDER":
            adj += -0.04 * pv
            adj += 0.08 * batt
            adj += 0.03 * hydro
        elif block == "EVENING_RAMP":
            adj += 0.10 * pv
            adj += -0.18 * batt
            adj += 0.08 * ev * (1.0 - 0.5 * np.clip(managed, 0.0, 1.0))
            adj += -0.10 * hydro
            if season == "WINTER":
                adj += 0.12 * hp
                adj += -0.04 * wind
        elif block == "NIGHT":
            adj += 0.05 * ev * (1.0 - 0.5 * np.clip(managed, 0.0, 1.0))
            if season in {"WINTER", "SHOULDER"}:
                adj += -0.05 * wind

        if day_type == "WEEKEND":
            adj *= 0.8
        # Interconnection and firm/flexible capacity damp country-specific extremes.
        adj *= max(0.60, 1.0 - 0.03 * max(ntc, 0.0) - 0.04 * max(firm, 0.0))
        return float(np.clip(adj, -0.60, 0.60))

    def apply(self, f_H: pd.Series, calendar_df: pd.DataFrame, *, vintage) -> pd.Series:
        if f_H.empty:
            return f_H
        if f_H.index.tz is None:
            raise TypeError("f_H index must be tz-aware")
        cal = calendar_df.reindex(f_H.index)
        required_cal = {"heure_hce", "type_jour"}
        if required_cal - set(cal.columns):
            raise KeyError(f"calendar_df missing columns {sorted(required_cal - set(cal.columns))}")

        store = self.scenario_store.asof(vintage)
        local = f_H.index.tz_convert(self.tz)
        periods = pd.PeriodIndex([pd.Period(f"{t.year}-{t.month:02d}", freq="M") for t in local])
        out = f_H.astype(float).copy()
        sign = np.sign(out.to_numpy(dtype=float))
        sign[sign == 0.0] = 1.0
        log_base = np.log(np.clip(np.abs(out.to_numpy(dtype=float)), 1e-6, None))
        adj = np.zeros(len(out), dtype=float)

        cached: dict[pd.Period, dict[str, float]] = {}
        missing_periods: list[str] = []
        for period in pd.unique(periods):
            try:
                row = store.get(country=self.country, scenario=self.scenario, delivery_period=period)
            except KeyError:
                missing_periods.append(str(period))
                continue
            cached[period] = self.projector.transform_row(row)
        if missing_periods:
            self.diagnostics_ = {
                "applied": False,
                "reason": "missing_scenario_coverage",
                "missing_periods": missing_periods,
                "scenario": self.scenario,
                "country": self.country,
            }
            raise KeyError(
                "electrification scenario coverage missing for "
                f"country={self.country!r}, scenario={self.scenario!r}, "
                f"periods={missing_periods[:6]}"
            )

        hours = cal["heure_hce"].astype(int).to_numpy()
        types = cal["type_jour"].astype(str).to_numpy()
        for i, period in enumerate(periods):
            cell_season = _season(int(period.month))
            cell_day_type = _day_type(types[i])
            cell_block = _block(int(hours[i]))
            adj[i] = self._adjustment(
                cached[period],
                season=cell_season,
                day_type=cell_day_type,
                block=cell_block,
            )

        out = pd.Series(sign * np.exp(log_base + adj), index=out.index, name=out.name or "f_H")
        days = _local_day_key(out.index, self.tz)
        target_mean = f_H.astype(float).groupby(days).transform("mean")
        day_mean = out.groupby(days).transform("mean")
        safe_day_mean = day_mean.mask(day_mean.abs() < 1e-12, 1.0)
        out = (out / safe_day_mean * target_mean).rename(f_H.name or "f_H")
        final_mean = out.groupby(days).transform("mean")
        safe_final_mean = final_mean.mask(final_mean.abs() < 1e-12, 1.0)
        out = (out / safe_final_mean * target_mean).rename(f_H.name or "f_H")
        self.diagnostics_ = {
            "applied": True,
            "scenario": self.scenario,
            "country": self.country,
            "n_periods": len(cached),
            "max_abs_log_adjustment": float(np.max(np.abs(adj))) if len(adj) else 0.0,
        }
        return out


def apply_electrification_scenarios(
    f_H: pd.Series,
    calendar_df: pd.DataFrame,
    shape_hourly,
    *,
    vintage,
    scenarios: Sequence[str],
    scenario_store: ElectrificationScenarioStore,
    country: str = "CH",
    tz: str = "Europe/Zurich",
) -> dict[str, pd.Series]:
    """Apply the structural layer independently for several scenarios."""
    curves: dict[str, pd.Series] = {}
    for scenario in scenarios:
        correction = ElectrificationFHCorrection(
            scenario_store=scenario_store,
            scenario=str(scenario),
            country=country,
            tz=tz,
        )
        curves[str(scenario)] = correction.fit(shape_hourly=shape_hourly, vintage=vintage).apply(
            f_H,
            calendar_df,
            vintage=vintage,
        )
    return curves


def structural_fan_chart(
    scenario_curves: Mapping[str, pd.Series],
    *,
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Build an honest structural scenario bracket from scenario-specific curves.

    Returns a DataFrame indexed like the input curves with:

    * `weighted_mean`
    * `scenario_low`, `scenario_central`, `scenario_high`
    * `scenario_spread`, defined as high-low

    The slow/central/fast scenario set is a governed bracket, not a calibrated
    probability distribution. Do not label these three points as p10/p50/p90.
    """
    if not scenario_curves:
        raise ValueError("scenario_curves must contain at least one scenario")
    labels = list(scenario_curves)
    first_index = scenario_curves[labels[0]].index
    for label, curve in scenario_curves.items():
        if not curve.index.equals(first_index):
            raise ValueError(f"scenario curve {label!r} has a mismatched index")
    norm_weights = _normalize_weights(labels, weights)
    w = np.array([norm_weights[label] for label in labels], dtype=float)
    matrix = np.column_stack([scenario_curves[label].to_numpy(dtype=float) for label in labels])

    out = pd.DataFrame(index=first_index)
    out["weighted_mean"] = matrix @ w
    out["scenario_low"] = np.nanmin(matrix, axis=1)
    if "central" in scenario_curves:
        out["scenario_central"] = scenario_curves["central"].to_numpy(dtype=float)
    else:
        out["scenario_central"] = out["weighted_mean"]
    out["scenario_high"] = np.nanmax(matrix, axis=1)
    out["scenario_spread"] = out["scenario_high"] - out["scenario_low"]
    return out


def electrification_modulate(
    f_H: pd.Series,
    calendar_df: pd.DataFrame,
    shape_hourly,
    *,
    vintage,
    scenario: str = "central",
    scenario_store: ElectrificationScenarioStore | None = None,
    scenario_path: str | Path | None = DEFAULT_SCENARIO_PATH,
    require_scenario_data: bool = False,
    require_production_scenario_data: bool = False,
    production_required_countries: Sequence[str] = ("CH", "DE", "FR", "IT", "AT"),
    country: str = "CH",
    tz: str = "Europe/Zurich",
) -> pd.Series:
    """Apply leak-free scenario modulation to ``f_H``.

    Missing scenario data degrades to identity with a warning. This keeps an
    enabled diagnostic estimator from crashing in workspaces where the official
    scenario inventory has not yet been loaded.
    """
    store = scenario_store or ElectrificationScenarioStore(scenario_path)
    if store.frame.empty:
        if require_scenario_data:
            raise FileNotFoundError(
                "Electrification scenario data is required but no governed rows were loaded "
                f"(scenario_path={scenario_path!r})"
            )
        # Diagnostic no-op when no governed scenario inventory has been loaded.
        logger.warning("Electrification scenario store is empty; returning f_H unchanged")
        return f_H
    if require_production_scenario_data:
        local_years = sorted(set(int(year) for year in f_H.index.tz_convert(tz).year.unique()))
        assert_production_scenario_inventory(
            store.asof(vintage).frame,
            required_countries=production_required_countries,
            scenarios=[scenario],
            years=local_years,
        )
    correction = ElectrificationFHCorrection(
        scenario_store=store,
        scenario=scenario,
        country=country,
        tz=tz,
    )
    return correction.fit(shape_hourly=shape_hourly, vintage=vintage).apply(
        f_H,
        calendar_df,
        vintage=vintage,
    )
