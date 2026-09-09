"""Pure, portable transforms for governed LT input replay."""

from __future__ import annotations

import numpy as np
import pandas as pd

_SWISS_BORDER_KEYS = ("de", "fr", "at", "it")
_HYDRO_ROLLING_WINDOW_YEARS = 10
_DELIVERY_TIMEZONE = "Europe/Zurich"


def clean_epex(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the governed EPEX spike flag without cache or network I/O."""

    result = frame.copy()
    monthly_period = _delivery_month(result.index)
    absolute_price = result["price_eur_mwh"].abs()
    prior_p999 = absolute_price.groupby(monthly_period).transform(
        lambda values: values.shift(1).expanding(min_periods=4).quantile(0.999)
    )
    result["spike_flag"] = (absolute_price > prior_p999).fillna(False)
    return result


def build_entso_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce deterministic ENTSO-E feature engineering."""

    result = frame.copy()
    monthly_period = _delivery_month(result.index)
    prior_q33 = result["solar_mw"].groupby(monthly_period).transform(
        lambda values: values.shift(1).expanding(min_periods=3).quantile(0.33)
    )
    prior_q66 = result["solar_mw"].groupby(monthly_period).transform(
        lambda values: values.shift(1).expanding(min_periods=3).quantile(0.66)
    )
    solar = result["solar_mw"]
    valid_regime = prior_q33.notna() & prior_q66.notna() & (prior_q33 < prior_q66)
    result["solar_regime"] = np.select(
        [valid_regime & (solar <= prior_q33), valid_regime & (solar <= prior_q66)],
        [0.0, 1.0],
        default=2.0,
    )
    result.loc[~valid_regime, "solar_regime"] = 1.0
    result["load_deviation"] = _causal_monthly_zscore(
        result["load_mw"], monthly_period
    )

    if "cross_border_mw" in result.columns:
        result["flow_deviation"] = _causal_monthly_zscore(
            result["cross_border_mw"], monthly_period
        )
    else:
        result["flow_deviation"] = 0.0

    for border_key in _SWISS_BORDER_KEYS:
        for column in (
            f"scheduled_net_export_ch_{border_key}_mw",
            f"ntc_total_ch_{border_key}_mw",
            f"ntc_net_ch_{border_key}_mw",
        ):
            if column in result.columns:
                result[f"{column}_zscore"] = _causal_monthly_zscore(
                    result[column], monthly_period
                )

        export_column = f"ntc_export_ch_{border_key}_mw"
        import_column = f"ntc_import_ch_{border_key}_mw"
        if export_column in result.columns and import_column in result.columns:
            denominator = (
                result[export_column].fillna(0.0)
                + result[import_column].fillna(0.0)
            ).replace(0, np.nan)
            result[f"ntc_balance_ch_{border_key}"] = (
                (result[export_column].fillna(0.0) - result[import_column].fillna(0.0))
                / denominator
            ).fillna(0.0)

    if "fr_nuclear_unavailable_mw" in result.columns:
        unavailable = result["fr_nuclear_unavailable_mw"].fillna(0.0).clip(lower=0.0)
        prior = unavailable.shift(1).expanding(min_periods=3)
        q75 = prior.quantile(0.75).fillna(0.0)
        q90 = prior.quantile(0.90).fillna(1.0).clip(lower=1.0)
        stress_threshold = q75.clip(lower=5000.0)
        result["fr_nuclear_unavailable_mw"] = unavailable
        result["fr_nuclear_unavailability_ratio"] = (unavailable / q90).clip(upper=1.5)
        result["fr_nuclear_stress_flag"] = (
            unavailable >= stress_threshold
        ).astype(float)
    return result


def build_hydro_water_value(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce deterministic hydro water-value features."""

    result = frame.copy()
    seasonal_index = (
        result.index.tz_convert("Europe/Zurich")
        if result.index.tz is not None
        else result.index
    )
    result["iso_week"] = seasonal_index.isocalendar().week.values.astype(int)
    result["fill_deviation"] = np.nan
    result["water_value_supported"] = False
    result["water_value_history_count"] = 0
    result["water_value_reference_mean_pct"] = np.nan
    result["water_value_reference_std_pct"] = np.nan
    result["water_value_reference_se_pct"] = np.nan
    for iso_week in result["iso_week"].unique():
        mask = result["iso_week"] == iso_week
        week_data = result.loc[mask, "fill_pct"]
        if len(week_data) < 6:
            continue
        for index in week_data.index:
            cutoff = index - pd.DateOffset(years=_HYDRO_ROLLING_WINDOW_YEARS)
            historical = week_data.loc[
                (week_data.index >= cutoff) & (week_data.index < index)
            ]
            history_count = len(historical)
            result.loc[index, "water_value_history_count"] = history_count
            if history_count < 5:
                continue
            standard_deviation = historical.std()
            result.loc[index, "water_value_reference_mean_pct"] = historical.mean()
            if not np.isfinite(standard_deviation) or standard_deviation <= 1e-9:
                continue
            result.loc[index, "water_value_reference_std_pct"] = standard_deviation
            result.loc[index, "water_value_reference_se_pct"] = (
                standard_deviation / np.sqrt(history_count)
            )
            result.loc[index, "fill_deviation"] = (
                (week_data.loc[index] - historical.mean()) / standard_deviation
            )
            result.loc[index, "water_value_supported"] = True
    result["fill_deviation"] = result["fill_deviation"].fillna(0.0)
    for column in (
        "water_value_reference_mean_pct",
        "water_value_reference_std_pct",
        "water_value_reference_se_pct",
    ):
        result[column] = result[column].fillna(0.0)
    result["water_value_proxy"] = result["fill_deviation"]
    return result.drop(columns=["iso_week"])


def _delivery_month(index: pd.DatetimeIndex) -> pd.PeriodIndex:
    localized = (
        index.tz_convert(_DELIVERY_TIMEZONE).tz_localize(None)
        if index.tz is not None
        else index
    )
    return localized.to_period("M")


def _causal_monthly_zscore(
    values: pd.Series,
    monthly_period: pd.PeriodIndex,
) -> pd.Series:
    grouped = values.groupby(monthly_period)
    prior_mean = grouped.transform(
        lambda group: group.shift(1).expanding(min_periods=2).mean()
    )
    prior_std = grouped.transform(
        lambda group: group.shift(1).expanding(min_periods=2).std()
    )
    return ((values - prior_mean) / prior_std.replace(0, np.nan)).fillna(0.0)
