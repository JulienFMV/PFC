"""Export a local-test CH PFC as an hourly CSV over a local delivery window."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfc_shaping.calibration.eex_contract_selection import (  # noqa: E402
    calibration_buckets,
    finest_product_for_timestamp,
    quarter_product,
)
from pfc_shaping.lt.model.holiday_pressure import (  # noqa: E402
    ch_market_holiday_components as _ch_market_holiday_components,
    ch_market_holiday_pressure as _ch_market_holiday_pressure,
    ch_market_nonworking_pressure as _ch_market_nonworking_pressure,
    is_ch_nonworking_day as _is_ch_nonworking_day,
)
from pfc_shaping.lt.model.cross_year_seasonal_shape import (  # noqa: E402
    apply_cross_year_seasonal_shape_optimizer,
)
from pfc_shaping.lt.model.quote_aware_monthly_smoothing import (  # noqa: E402
    apply_quote_aware_monthly_smoothing,
)
from pfc_shaping.lt.model.quant_shape_optimizer import QuantShapeOptimizer  # noqa: E402
from pfc_shaping.lt.model.seam_nullspace_smoothing import (  # noqa: E402
    apply_seam_nullspace_smoothing,
)
from pfc_shaping.lt.model.shape_constraints import (  # noqa: E402
    build_base_peak_offpeak_constraint_system,
)
from scripts.build_local_test_ch_pfc import main as build_local_test_ch_pfc

LOCAL_TZ = "Europe/Zurich"
DEFAULT_LOCAL_START = "2026-06-13"
DEFAULT_LOCAL_END = "2030-12-31"
DEFAULT_OUTPUT = Path("output/local_test_ch_pfc_hourly_20260613_20301231.csv")
DEFAULT_REPORT = Path(".planning/phases/13-lt-electrification-scenario-shape/LOCAL-TEST-CH-PFC-HOURLY-CSV.md")
PRICE_COLUMNS = [
    "price_slow_eur_mwh",
    "price_central_eur_mwh",
    "price_fast_eur_mwh",
    "price_weighted_mean_eur_mwh",
    "structural_p10_eur_mwh",
    "structural_p50_eur_mwh",
    "structural_p90_eur_mwh",
    "structural_width_eur_mwh",
]
SCENARIO_PRICE_COLUMNS = {
    "slow": "price_slow_eur_mwh",
    "central": "price_central_eur_mwh",
    "fast": "price_fast_eur_mwh",
}

def _normalize_date(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).tz_localize(None).normalize()


def previous_business_day(valuation_date: str | pd.Timestamp) -> pd.Timestamp:
    """Return the previous Monday-Friday business day for EEX close checks."""
    day = _normalize_date(valuation_date) - pd.Timedelta(days=1)
    while day.weekday() >= 5:
        day -= pd.Timedelta(days=1)
    return day


def default_valuation_date() -> pd.Timestamp:
    return pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None).normalize()


def _local_window(local_start_date: str, local_end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(local_start_date).tz_localize(LOCAL_TZ)
    end = pd.Timestamp(local_end_date).tz_localize(LOCAL_TZ) + pd.Timedelta(hours=23, minutes=45)
    if end < start:
        raise ValueError("local_end_date must be >= local_start_date")
    return start, end


def _utc_build_window(local_start: pd.Timestamp, local_end_15min: pd.Timestamp) -> tuple[str, int]:
    start_utc = local_start.tz_convert("UTC")
    end_utc = local_end_15min.tz_convert("UTC")
    periods = int(((end_utc - start_utc) / pd.Timedelta(minutes=15))) + 1
    horizon_days = int(math.ceil(periods / 96.0))
    return start_utc.strftime("%Y-%m-%d %H:%M:%S"), horizon_days


def _parse_timestamp_ch(values: pd.Series, offsets: pd.Series | None = None) -> pd.Series:
    """Parse exported CH timestamps while preserving Europe/Zurich semantics."""
    text = values.astype(str)
    if offsets is not None:
        offset = offsets.astype(str).str.strip()
        offset = offset.str.replace("UTC", "", regex=False)
        numeric = offset.str.fullmatch(r"\d{1,4}", na=False)
        offset = offset.where(~numeric, "+" + offset.str.zfill(4))
        offset = offset.str.replace(":", "", regex=False)
        combined = text.str.strip() + " " + offset
        return pd.to_datetime(combined, format="%d.%m.%Y %H:%M %z", utc=True).dt.tz_convert(LOCAL_TZ)
    has_offset = text.str.contains(r"[+-]\d{4}$", regex=True, na=False)
    if bool(has_offset.all()):
        return pd.to_datetime(text, utc=True).dt.tz_convert(LOCAL_TZ)
    parsed = pd.to_datetime(text, format="%d.%m.%Y %H:%M", errors="raise")
    return parsed.dt.tz_localize(LOCAL_TZ, ambiguous="infer", nonexistent="shift_forward")


def to_hourly_csv_frame(fan_15min: pd.DataFrame, *, local_start_date: str, local_end_date: str) -> pd.DataFrame:
    local_start, local_end_15min = _local_window(local_start_date, local_end_date)
    hourly_start = local_start.floor("h")
    hourly_end = local_end_15min.floor("h")
    df = fan_15min.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("fan chart index must be a DatetimeIndex")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.tz_convert(LOCAL_TZ).sort_index()
    hourly = df.resample("h", label="left", closed="left").mean()
    hourly = hourly[(hourly.index >= hourly_start) & (hourly.index <= hourly_end)].copy()
    if hourly.empty:
        raise ValueError("hourly export is empty")

    out = pd.DataFrame(index=hourly.index)
    out["timestamp_ch"] = hourly.index.strftime("%d.%m.%Y %H:%M")
    offset = pd.Index(hourly.index.strftime("%z"))
    out["utc_offset_ch"] = "UTC" + offset.str.slice(0, 3) + ":" + offset.str.slice(3, 5)
    out["timestamp_utc"] = hourly.index.tz_convert("UTC").strftime("%d.%m.%Y %H:%M")
    column_map = {
        "curve_slow": "price_slow_eur_mwh",
        "curve_central": "price_central_eur_mwh",
        "curve_fast": "price_fast_eur_mwh",
        "weighted_mean": "price_weighted_mean_eur_mwh",
        "structural_scenario_low": "structural_p10_eur_mwh",
        "structural_scenario_central": "structural_p50_eur_mwh",
        "structural_scenario_high": "structural_p90_eur_mwh",
        "structural_scenario_spread": "structural_width_eur_mwh",
        "structural_p10": "structural_p10_eur_mwh",
        "structural_p50": "structural_p50_eur_mwh",
        "structural_p90": "structural_p90_eur_mwh",
        "structural_width": "structural_width_eur_mwh",
    }
    for source, target in column_map.items():
        if source in hourly.columns:
            out[target] = hourly[source].astype(float).round(6)
    out = out.reset_index(drop=True)
    expected_hours = len(pd.date_range(hourly_start, hourly_end, freq="h", tz=LOCAL_TZ))
    if len(out) != expected_hours:
        raise ValueError(f"hourly row count mismatch: expected {expected_hours}, got {len(out)}")
    numeric = out.select_dtypes(include="number")
    if numeric.isna().any().any():
        raise ValueError("hourly export contains NaN numeric values")
    return out


def _latest_eex_prices_by_load_type(
    path: Path,
    *,
    market: str = "CH",
) -> tuple[pd.Timestamp, dict[str, dict[str, float]]]:
    frame = pd.read_parquet(path)
    sub = frame[frame["market"].astype(str) == market].copy()
    if sub.empty:
        raise ValueError(f"no EEX prices for market={market}")
    base = sub[sub["load_type"].astype(str).str.upper() == "BASE"].copy()
    if base.empty:
        raise ValueError(f"no EEX BASE prices for market={market}")
    latest = pd.Timestamp(base["date"].max())
    snap = sub[sub["date"] == latest]
    prices: dict[str, dict[str, float]] = {}
    for load_type, group in snap.groupby(snap["load_type"].astype(str).str.upper()):
        prices[str(load_type)] = dict(zip(group["product"].astype(str), group["price"].astype(float)))
    return latest, prices


def _latest_eex_base_prices(path: Path, *, market: str = "CH") -> tuple[pd.Timestamp, dict[str, float]]:
    latest, prices = _latest_eex_prices_by_load_type(path, market=market)
    sub = prices.get("BASE", {})
    if not sub:
        raise ValueError(f"no EEX BASE prices for market={market}")
    return latest, sub


def require_forward_date(
    latest_forward_date: pd.Timestamp,
    *,
    required_forward_date: str | pd.Timestamp | None,
    valuation_date: str | pd.Timestamp | None,
) -> pd.Timestamp:
    """Fail fast when the EEX parquet is not fresh enough for a run."""
    if required_forward_date is None:
        if valuation_date is None:
            valuation_date = default_valuation_date()
        required = previous_business_day(valuation_date)
    else:
        required = _normalize_date(required_forward_date)

    latest = _normalize_date(latest_forward_date)
    if latest != required:
        raise ValueError(
            "stale EEX CH BASE forwards: "
            f"latest={latest.date()} required={required.date()}. "
            "Refresh data/eex_forwards_history.parquet from the desk EEX report "
            "or pass --allow-stale-forwards for an explicit diagnostic/backtest run."
        )
    return required


def _quarter_product(ts: pd.Timestamp) -> str:
    return quarter_product(ts)


def _calibration_product(ts: pd.Timestamp, prices: dict[str, float]) -> str | None:
    return finest_product_for_timestamp(ts, prices)


def _calibration_buckets(ts_ch: pd.Series, forward_prices: dict[str, float]) -> tuple[pd.Series, dict[str, float]]:
    return calibration_buckets(ts_ch, forward_prices)


def calibrate_hourly_to_eex(
    hourly: pd.DataFrame,
    *,
    forward_prices: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scale hourly scenario/fan columns so product means match EEX BASE."""
    out = hourly.copy()
    ts_ch = _parse_timestamp_ch(out["timestamp_ch"], out.get("utc_offset_ch"))
    out["_calibration_product"], calibration_targets = _calibration_buckets(ts_ch, forward_prices)
    rows: list[dict[str, object]] = []
    for product, idx in out.groupby("_calibration_product", dropna=True).groups.items():
        if product is None or product not in calibration_targets:
            continue
        current = float(out.loc[idx, "price_weighted_mean_eur_mwh"].mean())
        target = float(calibration_targets[str(product)])
        if current == 0.0 or not pd.notna(current):
            raise ValueError(f"cannot calibrate product {product}: current mean is {current}")
        factor = target / current
        if factor <= 0.0 or not np.isfinite(factor):
            raise ValueError(
                f"cannot calibrate product {product}: non-positive/non-finite scale factor "
                f"{factor} from current={current}, target={target}"
            )
        for column in PRICE_COLUMNS:
            if column in out.columns:
                out.loc[idx, column] = out.loc[idx, column].astype(float) * factor
        rows.append(
            {
                "product": product,
                "target_eex_base_eur_mwh": target,
                "pre_calibration_mean_eur_mwh": current,
                "post_calibration_mean_eur_mwh": float(out.loc[idx, "price_weighted_mean_eur_mwh"].mean()),
                "scale_factor": factor,
                "rows": len(idx),
            }
        )
    out = out.drop(columns=["_calibration_product"])
    for column in PRICE_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype(float).round(6)
    return out, pd.DataFrame(rows).sort_values("product").reset_index(drop=True)


def _eex_peak_mask(ts_ch: pd.Series, *, country: str = "CH") -> pd.Series:
    """Return EEX contractual peak hours for local CH timestamps.

    European EEX Peakload includes public holidays. Holiday pressure remains
    a separate spot-shaping feature.
    """
    ts = pd.Series(ts_ch, copy=False)
    if ts.empty:
        return pd.Series(dtype=bool, index=ts.index)
    if str(country).upper() not in {"CH", "DE", "AT", "FR", "IT"}:
        raise ValueError(f"unsupported EEX peak country: {country!r}")
    return (
        (ts.dt.weekday < 5)
        & (ts.dt.hour >= 8)
        & (ts.dt.hour < 20)
    )


def _price_level_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in PRICE_COLUMNS if column in frame.columns and column != "structural_width_eur_mwh"]


def _recompute_weighted_fan_columns(hourly: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = hourly.copy()
    labels = tuple(SCENARIO_PRICE_COLUMNS)
    matrix = np.column_stack([out[SCENARIO_PRICE_COLUMNS[label]].to_numpy(dtype=float) for label in labels])
    w = np.array([weights[label] for label in labels], dtype=float)
    out["price_weighted_mean_eur_mwh"] = matrix @ w
    out["structural_p10_eur_mwh"] = [_weighted_quantile_row(row, w, 0.10) for row in matrix]
    out["structural_p50_eur_mwh"] = [_weighted_quantile_row(row, w, 0.50) for row in matrix]
    out["structural_p90_eur_mwh"] = [_weighted_quantile_row(row, w, 0.90) for row in matrix]
    out["structural_width_eur_mwh"] = out["structural_p90_eur_mwh"] - out["structural_p10_eur_mwh"]
    return out


def _quarter_month_numbers(product: str) -> list[int]:
    if "-Q" not in str(product) or str(product).endswith("-RESIDUAL"):
        return []
    try:
        year_s, quarter_s = str(product).split("-Q", 1)
        int(year_s)
        quarter = int(quarter_s)
    except ValueError:
        return []
    if quarter < 1 or quarter > 4:
        return []
    first = (quarter - 1) * 3 + 1
    return [first, first + 1, first + 2]


def _quarter_year(product: str) -> int | None:
    if "-Q" not in str(product):
        return None
    try:
        return int(str(product).split("-Q", 1)[0])
    except ValueError:
        return None


def _month_key(year: int, month: int) -> str:
    return f"{int(year)}-{int(month):02d}"


def _guided_month_targets(
    *,
    parent_product: str,
    parent_target: float,
    months: list[int],
    month_counts: dict[int, int],
    primary_prices: dict[str, float],
    historical_deviations: dict[tuple[int, int], float] | None = None,
    primary_weight: float = 0.65,
) -> tuple[dict[int, float], str] | None:
    """Return month targets shaped by a guide but levelled to ``parent_target``."""
    year = _quarter_year(parent_product)
    if year is None:
        return None
    quarter = ((months[0] - 1) // 3) + 1

    def recenter(raw_targets: dict[int, float]) -> dict[int, float]:
        total = float(sum(month_counts[month] for month in months))
        avg = sum(float(raw_targets[month]) * float(month_counts[month]) for month in months) / total
        shift = float(parent_target) - avg
        return {month: float(raw_targets[month]) + shift for month in months}

    def targets_from_prices(
        *,
        guide_year: int,
        prices: dict[str, float],
    ) -> dict[int, float] | None:
        guide_parent = f"{guide_year}-Q{quarter}"
        guide_months = {_month_key(guide_year, month): prices.get(_month_key(guide_year, month)) for month in months}
        if guide_parent not in prices or any(value is None for value in guide_months.values()):
            return None
        return recenter({month: float(guide_months[_month_key(guide_year, month)]) for month in months})

    current = targets_from_prices(guide_year=year, prices=primary_prices)
    historical = None
    if historical_deviations:
        raw = {
            month: float(parent_target) + float(historical_deviations[(quarter, month)])
            for month in months
            if (quarter, month) in historical_deviations
        }
        if len(raw) == len(months):
            historical = recenter(raw)

    if current is not None and historical is not None:
        w = min(1.0, max(0.0, float(primary_weight)))
        blended = {
            month: w * float(current[month]) + (1.0 - w) * float(historical[month])
            for month in months
        }
        return recenter(blended), f"neighbor_current_{w:.2f}_historical_{1.0 - w:.2f}"
    if current is not None:
        return current, "neighbor_current"
    if historical is not None:
        return historical, "historical"
    return None


def _historical_month_deviations(
    forwards_path: Path,
    *,
    market: str,
    load_type: str,
    min_date: pd.Timestamp | None = None,
) -> dict[tuple[int, int], float]:
    """Median historical monthly deviations from the quoted quarter."""
    if not forwards_path.exists():
        return {}
    df = pd.read_parquet(forwards_path)
    if df.empty:
        return {}
    dates = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    sub = df[
        (df["market"].astype(str).str.upper() == str(market).upper())
        & (df["load_type"].astype(str).str.upper() == str(load_type).upper())
    ].copy()
    sub["date"] = dates.loc[sub.index]
    if min_date is not None:
        sub = sub[sub["date"] >= pd.Timestamp(min_date).tz_localize(None).normalize()]
    if sub.empty:
        return {}
    rows: list[dict[str, float]] = []
    for (date, product), quarter_row in sub[sub["product_type"].eq("Quarter")].groupby(["date", "product"]):
        product = str(product)
        months = _quarter_month_numbers(product)
        year = _quarter_year(product)
        if not months or year is None:
            continue
        quarter_price = float(quarter_row["price"].iloc[-1])
        same_date = sub[sub["date"].eq(date)]
        for month in months:
            month_product = _month_key(year, month)
            month_row = same_date[same_date["product"].astype(str).eq(month_product)]
            if month_row.empty:
                break
        else:
            quarter = ((months[0] - 1) // 3) + 1
            for month in months:
                month_product = _month_key(year, month)
                month_price = float(same_date[same_date["product"].astype(str).eq(month_product)]["price"].iloc[-1])
                rows.append({"quarter": quarter, "month": month, "deviation": month_price - quarter_price})
    if not rows:
        return {}
    dev = pd.DataFrame(rows)
    med = dev.groupby(["quarter", "month"])["deviation"].median()
    return {(int(q), int(m)): float(value) for (q, m), value in med.items()}


def _historical_annual_month_deviations(
    forwards_path: Path,
    *,
    market: str,
    load_type: str,
    min_date: pd.Timestamp | None = None,
) -> dict[int, float]:
    """Median historical monthly deviations from same-delivery Calendar."""
    if not forwards_path.exists():
        return {}
    df = pd.read_parquet(forwards_path)
    if df.empty:
        return {}
    dates = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    sub = df[
        (df["market"].astype(str).str.upper() == str(market).upper())
        & (df["load_type"].astype(str).str.upper() == str(load_type).upper())
    ].copy()
    sub["date"] = dates.loc[sub.index]
    if min_date is not None:
        sub = sub[sub["date"] >= pd.Timestamp(min_date).tz_localize(None).normalize()]
    if sub.empty:
        return {}
    rows: list[dict[str, float]] = []
    cal_rows = sub[sub["product_type"].astype(str).str.lower().isin(["cal", "year", "calendar"])]
    for (date, product), cal_row in cal_rows.groupby(["date", "product"]):
        product = str(product)
        if not product.isdigit():
            continue
        year = int(product)
        cal_price = float(cal_row["price"].iloc[-1])
        same_date = sub[sub["date"].eq(date)]
        for month in range(1, 13):
            month_product = _month_key(year, month)
            month_row = same_date[same_date["product"].astype(str).eq(month_product)]
            if month_row.empty:
                continue
            rows.append(
                {
                    "month": month,
                    "deviation": float(month_row["price"].iloc[-1]) - cal_price,
                }
            )
    if not rows:
        return {}
    dev = pd.DataFrame(rows)
    med = dev.groupby("month")["deviation"].median()
    return {int(month): float(value) for month, value in med.items()}


def _frontier_monthly_shape_guide(
    forwards_path: Path,
    *,
    markets: list[str],
    load_type: str,
    years: list[int],
    min_date: pd.Timestamp | None = None,
) -> dict[str, float]:
    """Build a multi-market monthly guide from current quotes plus history."""
    market_guides: list[dict[str, float]] = []
    for market in markets:
        market = str(market).upper().strip()
        if not market:
            continue
        try:
            _, prices_by_load = _latest_eex_prices_by_load_type(forwards_path, market=market)
        except ValueError:
            continue
        prices = prices_by_load.get(str(load_type).upper(), {})
        if not prices:
            continue
        quarter_dev = _historical_month_deviations(
            forwards_path,
            market=market,
            load_type=load_type,
            min_date=min_date,
        )
        annual_dev = _historical_annual_month_deviations(
            forwards_path,
            market=market,
            load_type=load_type,
            min_date=min_date,
        )
        guide: dict[str, float] = {}
        for year in years:
            for month in range(1, 13):
                month_key = _month_key(year, month)
                if month_key in prices:
                    guide[month_key] = float(prices[month_key])
                    continue
                quarter = ((month - 1) // 3) + 1
                quarter_key = f"{year}-Q{quarter}"
                if quarter_key in prices and (quarter, month) in quarter_dev:
                    guide[month_key] = float(prices[quarter_key]) + float(quarter_dev[(quarter, month)])
                    continue
                cal_key = str(year)
                if cal_key in prices and month in annual_dev:
                    guide[month_key] = float(prices[cal_key]) + float(annual_dev[month])
        if guide:
            market_guides.append(guide)
    if not market_guides:
        return {}
    keys = sorted(set().union(*(guide.keys() for guide in market_guides)))
    out: dict[str, float] = {}
    for key in keys:
        values = [guide[key] for guide in market_guides if key in guide]
        if values:
            out[key] = float(np.median(values))
    return out


def apply_neighbor_monthly_spread_anchor(
    hourly: pd.DataFrame,
    *,
    ts_ch: pd.Series,
    base_forward_prices: dict[str, float],
    peak_forward_prices: dict[str, float],
    neighbor_base_forward_prices: dict[str, float],
    neighbor_peak_forward_prices: dict[str, float],
    historical_base_deviations: dict[tuple[int, int], float] | None = None,
    historical_peak_deviations: dict[tuple[int, int], float] | None = None,
    neighbor_current_weight: float = 0.65,
    weights: dict[str, float],
    intensity: float = 1.0,
    max_month_delta_eur_mwh: float = 20.0,
    negative_price_floor: float = -30.0,
    max_weighted_negative_hours: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Anchor unquoted CH monthly splits to neighbor/current or historical shapes.

    The overlay is additive and local/test only. BASE adjustments are constant
    across all hours in the month. PEAK adjustments are applied to EEX peak
    hours and compensated on offpeak hours in the same month, so monthly BASE
    energy is unchanged while PEAK month shape can move.
    """
    if intensity < 0.0 or not np.isfinite(float(intensity)):
        raise ValueError("neighbor monthly anchor intensity must be finite and non-negative")
    if max_month_delta_eur_mwh < 0.0 or not np.isfinite(float(max_month_delta_eur_mwh)):
        raise ValueError("neighbor monthly max delta must be finite and non-negative")
    out = hourly.copy()
    if float(intensity) == 0.0:
        return out, pd.DataFrame()
    ts = pd.Series(np.asarray(ts_ch), index=out.index)
    month = ts.dt.month.astype(int)
    year = ts.dt.year.astype(int)
    peak_mask = _eex_peak_mask(ts, country="CH")
    scenario_cols = list(SCENARIO_PRICE_COLUMNS.values())
    audit_rows: list[dict[str, object]] = []

    def apply_base_anchor() -> None:
        base_buckets, base_targets = _calibration_buckets(ts, base_forward_prices)
        for product, idx in base_buckets.groupby(base_buckets, dropna=True).groups.items():
            product = str(product)
            months = _quarter_month_numbers(product)
            parent_year = _quarter_year(product)
            if not months or parent_year is None or product not in base_targets:
                continue
            idx = pd.Index(idx)
            month_counts = {m: int(((year.loc[idx] == parent_year) & (month.loc[idx] == m)).sum()) for m in months}
            if any(count <= 0 for count in month_counts.values()):
                continue
            guide = _guided_month_targets(
                parent_product=product,
                parent_target=float(base_targets[product]),
                months=months,
                month_counts=month_counts,
                primary_prices=neighbor_base_forward_prices,
                historical_deviations=historical_base_deviations,
                primary_weight=neighbor_current_weight,
            )
            if guide is None:
                continue
            targets, source = guide
            for m in months:
                m_idx = idx[(year.loc[idx].to_numpy(dtype=int) == parent_year) & (month.loc[idx].to_numpy(dtype=int) == m)]
                current = float(out.loc[m_idx, "price_weighted_mean_eur_mwh"].mean())
                raw_delta = float(targets[m]) - current
                delta = float(raw_delta) * float(intensity)
                if abs(delta) > float(max_month_delta_eur_mwh):
                    delta = math.copysign(float(max_month_delta_eur_mwh), delta)
                for col in scenario_cols:
                    out.loc[m_idx, col] = out.loc[m_idx, col].astype(float) + delta
                audit_rows.append(
                    {
                        "load_type": "BASE",
                        "product": product,
                        "month": _month_key(parent_year, m),
                        "source": source,
                        "current_mean_eur_mwh": current,
                        "guide_target_eur_mwh": float(targets[m]),
                        "delta_eur_mwh": delta,
                        "rows": len(m_idx),
                    }
                )

    def apply_peak_anchor() -> None:
        if not peak_forward_prices:
            return
        peak_ts = ts.loc[peak_mask]
        if peak_ts.empty:
            return
        peak_buckets, peak_targets = _calibration_buckets(peak_ts, peak_forward_prices)
        for product, peak_idx in peak_buckets.groupby(peak_buckets, dropna=True).groups.items():
            product = str(product)
            months = _quarter_month_numbers(product)
            parent_year = _quarter_year(product)
            if not months or parent_year is None or product not in peak_targets:
                continue
            peak_idx = pd.Index(peak_idx)
            month_counts = {m: int(((year.loc[peak_idx] == parent_year) & (month.loc[peak_idx] == m)).sum()) for m in months}
            if any(count <= 0 for count in month_counts.values()):
                continue
            guide = _guided_month_targets(
                parent_product=product,
                parent_target=float(peak_targets[product]),
                months=months,
                month_counts=month_counts,
                primary_prices=neighbor_peak_forward_prices,
                historical_deviations=historical_peak_deviations,
                primary_weight=neighbor_current_weight,
            )
            if guide is None:
                continue
            targets, source = guide
            for m in months:
                m_peak_idx = peak_idx[
                    (year.loc[peak_idx].to_numpy(dtype=int) == parent_year)
                    & (month.loc[peak_idx].to_numpy(dtype=int) == m)
                ]
                m_all_idx = out.index[(year == parent_year) & (month == m)]
                m_offpeak_idx = pd.Index(m_all_idx).difference(m_peak_idx)
                if len(m_peak_idx) == 0 or len(m_offpeak_idx) == 0:
                    continue
                current = float(out.loc[m_peak_idx, "price_weighted_mean_eur_mwh"].mean())
                raw_delta = float(targets[m]) - current
                peak_delta = float(raw_delta) * float(intensity)
                if abs(peak_delta) > float(max_month_delta_eur_mwh):
                    peak_delta = math.copysign(float(max_month_delta_eur_mwh), peak_delta)
                offpeak_delta = -peak_delta * float(len(m_peak_idx)) / float(len(m_offpeak_idx))
                for col in scenario_cols:
                    out.loc[m_peak_idx, col] = out.loc[m_peak_idx, col].astype(float) + peak_delta
                    out.loc[m_offpeak_idx, col] = out.loc[m_offpeak_idx, col].astype(float) + offpeak_delta
                audit_rows.append(
                    {
                        "load_type": "PEAK",
                        "product": product,
                        "month": _month_key(parent_year, m),
                        "source": source,
                        "current_mean_eur_mwh": current,
                        "guide_target_eur_mwh": float(targets[m]),
                        "delta_eur_mwh": peak_delta,
                        "offpeak_compensation_delta_eur_mwh": offpeak_delta,
                        "rows": len(m_peak_idx),
                    }
                )

    apply_base_anchor()
    out = _recompute_weighted_fan_columns(out, weights)
    apply_peak_anchor()
    out = _recompute_weighted_fan_columns(out, weights)
    min_price = float(out[[*scenario_cols, "price_weighted_mean_eur_mwh"]].min().min())
    if min_price < float(negative_price_floor) - 1e-9:
        raise ValueError(
            f"neighbor monthly anchor breached negative price floor: min={min_price:.6f}, "
            f"floor={float(negative_price_floor):.6f}"
        )
    weighted_negative_hours = int((out["price_weighted_mean_eur_mwh"] < 0.0).sum())
    if weighted_negative_hours > int(max_weighted_negative_hours):
        raise ValueError(
            "neighbor monthly anchor produced too many weighted-mean negative hours: "
            f"{weighted_negative_hours} > {int(max_weighted_negative_hours)}"
        )
    for column in PRICE_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype(float).round(6)
    audit = pd.DataFrame(audit_rows)
    if not audit.empty:
        audit = audit.sort_values(["load_type", "product", "month"]).reset_index(drop=True)
    return out, audit


def apply_synthetic_monthly_path_smoothing(
    hourly: pd.DataFrame,
    *,
    ts_ch: pd.Series,
    base_forward_prices: dict[str, float],
    peak_forward_prices: dict[str, float],
    weights: dict[str, float],
    intensity: float = 1.0,
    smoothness_lambda: float = 12.0,
    max_month_delta_eur_mwh: float = 12.0,
    negative_price_floor: float = -30.0,
    max_weighted_negative_hours: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Smooth monthly means inside annual/residual synthetic buckets only."""
    if intensity < 0.0 or not np.isfinite(float(intensity)):
        raise ValueError("synthetic monthly path smoothing intensity must be finite and non-negative")
    if smoothness_lambda < 0.0 or not np.isfinite(float(smoothness_lambda)):
        raise ValueError("synthetic monthly path smoothing lambda must be finite and non-negative")
    if max_month_delta_eur_mwh < 0.0 or not np.isfinite(float(max_month_delta_eur_mwh)):
        raise ValueError("synthetic monthly path max delta must be finite and non-negative")
    out = hourly.copy()
    if float(intensity) == 0.0 or float(smoothness_lambda) == 0.0 or float(max_month_delta_eur_mwh) == 0.0:
        return out, pd.DataFrame()
    ts = pd.Series(np.asarray(ts_ch), index=out.index)
    base_buckets, base_targets = _calibration_buckets(ts, base_forward_prices)
    if base_buckets.isna().any():
        missing = int(base_buckets.isna().sum())
        raise ValueError(f"synthetic monthly path smoothing found {missing} rows without an EEX BASE bucket")
    peak_mask = _eex_peak_mask(ts, country="CH")
    peak_buckets = pd.Series(dtype=object)
    if peak_forward_prices and bool(peak_mask.any()):
        peak_buckets, _ = _calibration_buckets(ts.loc[peak_mask], peak_forward_prices)
    month = ts.dt.month.astype(int)
    year = ts.dt.year.astype(int)
    scenario_cols = list(SCENARIO_PRICE_COLUMNS.values())
    audit_rows: list[dict[str, object]] = []

    for product, idx in base_buckets.groupby(base_buckets, dropna=True).groups.items():
        product = str(product)
        synthetic = product.endswith("-RESIDUAL") or product.isdigit()
        if not synthetic or product not in base_targets:
            continue
        idx = pd.Index(idx)
        months = sorted(int(value) for value in month.loc[idx].unique())
        if len(months) < 3:
            continue
        old_means = np.array(
            [float(out.loc[idx[month.loc[idx].to_numpy(dtype=int) == m], "price_weighted_mean_eur_mwh"].mean()) for m in months],
            dtype=float,
        )
        constraints: list[np.ndarray] = [
            np.array([float((month.loc[idx] == m).sum()) for m in months], dtype=float)
        ]
        peak_in_bucket = idx.intersection(peak_buckets.index) if not peak_buckets.empty else pd.Index([])
        if len(peak_in_bucket) > 0:
            for _, peak_idx in peak_buckets.loc[peak_in_bucket].groupby(peak_buckets.loc[peak_in_bucket], dropna=True).groups.items():
                peak_idx = pd.Index(peak_idx)
                coeff = np.array([float((month.loc[peak_idx] == m).sum()) for m in months], dtype=float)
                if coeff.sum() > 0.0:
                    constraints.append(coeff)
        delta = _solve_month_path_delta(
            old_means,
            constraints=np.vstack(constraints),
            smoothness_lambda=float(smoothness_lambda),
        )
        delta = delta * float(intensity)
        max_abs = float(np.max(np.abs(delta))) if len(delta) else 0.0
        if max_abs > float(max_month_delta_eur_mwh):
            delta = delta * (float(max_month_delta_eur_mwh) / max_abs)
        for pos, m in enumerate(months):
            m_idx = idx[month.loc[idx].to_numpy(dtype=int) == m]
            for col in scenario_cols:
                out.loc[m_idx, col] = out.loc[m_idx, col].astype(float) + float(delta[pos])
            audit_rows.append(
                {
                    "product": product,
                    "year": int(year.loc[m_idx].mode().iloc[0]) if len(m_idx) else None,
                    "month": int(m),
                    "mean_before_eur_mwh": float(old_means[pos]),
                    "delta_eur_mwh": float(delta[pos]),
                    "mean_after_eur_mwh": float(old_means[pos] + delta[pos]),
                    "rows": int(len(m_idx)),
                }
            )

    out = _recompute_weighted_fan_columns(out, weights)
    min_price = float(out[[*scenario_cols, "price_weighted_mean_eur_mwh"]].min().min())
    if min_price < float(negative_price_floor) - 1e-9:
        raise ValueError(
            f"synthetic monthly path smoothing breached negative price floor: min={min_price:.6f}, "
            f"floor={float(negative_price_floor):.6f}"
        )
    weighted_negative_hours = int((out["price_weighted_mean_eur_mwh"] < 0.0).sum())
    if weighted_negative_hours > int(max_weighted_negative_hours):
        raise ValueError(
            "synthetic monthly path smoothing produced too many weighted-mean negative hours: "
            f"{weighted_negative_hours} > {int(max_weighted_negative_hours)}"
        )
    for column in PRICE_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype(float).round(6)
    audit = pd.DataFrame(audit_rows)
    if not audit.empty:
        audit = audit.sort_values(["product", "month"]).reset_index(drop=True)
    return out, audit


def apply_neighbor_annual_residual_shape_anchor(
    hourly: pd.DataFrame,
    *,
    ts_ch: pd.Series,
    base_forward_prices: dict[str, float],
    neighbor_base_forward_prices: dict[str, float],
    historical_base_deviations: dict[tuple[int, int], float] | None = None,
    neighbor_current_weight: float = 0.65,
    weights: dict[str, float],
    intensity: float = 1.0,
    max_month_delta_eur_mwh: float = 25.0,
    negative_price_floor: float = -30.0,
    max_weighted_negative_hours: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Anchor annual residual monthly shape to neighbor forward structure.

    This handles products like ``2028-RESIDUAL`` created when a Calendar quote
    overlaps finer quoted products, e.g. CH Cal 2028 plus CH Q1 2028.
    Absolute levels are never copied from the neighbor.  We use neighbor
    month/quarter deviations and re-center them exactly to the CH residual
    bucket target.
    """
    if intensity < 0.0 or not np.isfinite(float(intensity)):
        raise ValueError("annual residual anchor intensity must be finite and non-negative")
    if max_month_delta_eur_mwh < 0.0 or not np.isfinite(float(max_month_delta_eur_mwh)):
        raise ValueError("annual residual anchor max delta must be finite and non-negative")
    out = hourly.copy()
    if float(intensity) == 0.0:
        return out, pd.DataFrame()
    ts = pd.Series(np.asarray(ts_ch), index=out.index)
    base_buckets, base_targets = _calibration_buckets(ts, base_forward_prices)
    if base_buckets.isna().any():
        missing = int(base_buckets.isna().sum())
        raise ValueError(f"annual residual anchor found {missing} rows without an EEX BASE bucket")
    month = ts.dt.month.astype(int)
    year = ts.dt.year.astype(int)
    scenario_cols = list(SCENARIO_PRICE_COLUMNS.values())
    audit_rows: list[dict[str, object]] = []

    for product, idx in base_buckets.groupby(base_buckets, dropna=True).groups.items():
        product = str(product)
        if not product.endswith("-RESIDUAL"):
            continue
        parent_year_text = product.removesuffix("-RESIDUAL")
        if not parent_year_text.isdigit() or product not in base_targets:
            continue
        parent_year = int(parent_year_text)
        idx = pd.Index(idx)
        months = sorted(int(value) for value in month.loc[idx].unique())
        if len(months) < 4:
            continue
        counts = {m: int((month.loc[idx] == m).sum()) for m in months}
        guide = _annual_residual_month_targets(
            parent_year=parent_year,
            parent_target=float(base_targets[product]),
            months=months,
            month_counts=counts,
            neighbor_base_forward_prices=neighbor_base_forward_prices,
            historical_base_deviations=historical_base_deviations,
            neighbor_current_weight=neighbor_current_weight,
        )
        if guide is None:
            continue
        targets, source = guide
        for m in months:
            m_idx = idx[(year.loc[idx].to_numpy(dtype=int) == parent_year) & (month.loc[idx].to_numpy(dtype=int) == m)]
            if len(m_idx) == 0:
                continue
            current = float(out.loc[m_idx, "price_weighted_mean_eur_mwh"].mean())
            raw_delta = float(targets[m]) - current
            delta = raw_delta * float(intensity)
            if abs(delta) > float(max_month_delta_eur_mwh):
                delta = math.copysign(float(max_month_delta_eur_mwh), delta)
            for col in scenario_cols:
                out.loc[m_idx, col] = out.loc[m_idx, col].astype(float) + delta
            audit_rows.append(
                {
                    "product": product,
                    "year": parent_year,
                    "month": int(m),
                    "source": source,
                    "current_mean_eur_mwh": current,
                    "guide_target_eur_mwh": float(targets[m]),
                    "delta_eur_mwh": delta,
                    "rows": int(len(m_idx)),
                }
            )

    out = _recompute_weighted_fan_columns(out, weights)
    min_price = float(out[[*scenario_cols, "price_weighted_mean_eur_mwh"]].min().min())
    if min_price < float(negative_price_floor) - 1e-9:
        raise ValueError(
            f"annual residual anchor breached negative price floor: min={min_price:.6f}, "
            f"floor={float(negative_price_floor):.6f}"
        )
    weighted_negative_hours = int((out["price_weighted_mean_eur_mwh"] < 0.0).sum())
    if weighted_negative_hours > int(max_weighted_negative_hours):
        raise ValueError(
            "annual residual anchor produced too many weighted-mean negative hours: "
            f"{weighted_negative_hours} > {int(max_weighted_negative_hours)}"
        )
    for column in PRICE_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype(float).round(6)
    audit = pd.DataFrame(audit_rows)
    if not audit.empty:
        audit = audit.sort_values(["product", "month"]).reset_index(drop=True)
    return out, audit


def apply_final_quant_annual_smoothness_calibration(
    hourly: pd.DataFrame,
    *,
    ts_ch: pd.Series,
    base_forward_prices: dict[str, float],
    peak_forward_prices: dict[str, float],
    weights: dict[str, float],
    lambda_smooth_h: float = 0.0,
    lambda_smooth_m: float = 10000.0,
    lambda_seam: float = 10000.0,
    negative_price_floor: float = -30.0,
    max_weighted_negative_hours: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Smooth complete annual-only synthetic years with hard BASE/PEAK constraints."""

    for name, value in {
        "lambda_smooth_h": lambda_smooth_h,
        "lambda_smooth_m": lambda_smooth_m,
        "lambda_seam": lambda_seam,
    }.items():
        if float(value) < 0.0 or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite and non-negative")
    out = hourly.copy()
    if float(lambda_smooth_h) == 0.0 and float(lambda_smooth_m) == 0.0 and float(lambda_seam) == 0.0:
        return out, pd.DataFrame()
    ts = pd.Series(np.asarray(ts_ch), index=out.index)
    base_buckets, base_targets = _calibration_buckets(ts, base_forward_prices)
    if base_buckets.isna().any():
        missing = int(base_buckets.isna().sum())
        raise ValueError(f"quant annual smoothness found {missing} rows without an EEX BASE bucket")
    month = ts.dt.month.astype(int)
    scenario_cols = list(SCENARIO_PRICE_COLUMNS.values())
    audit_rows: list[dict[str, object]] = []

    for product, idx in base_buckets.groupby(base_buckets, dropna=True).groups.items():
        product = str(product)
        if not product.isdigit() or product not in base_targets:
            continue
        idx = pd.Index(idx)
        months = sorted(int(value) for value in month.loc[idx].unique())
        if months != list(range(1, 13)):
            continue
        year_index = pd.DatetimeIndex(ts.loc[idx]).tz_convert("Europe/Zurich")
        constraints = build_base_peak_offpeak_constraint_system(
            year_index.tz_convert("UTC"),
            {product: float(base_targets[product])},
            {product: float(peak_forward_prices[product])} if product in peak_forward_prices else {},
            country="CH",
        )
        if not constraints.rows:
            continue
        for scenario, col in SCENARIO_PRICE_COLUMNS.items():
            prior = pd.Series(out.loc[idx, col].to_numpy(dtype=float), index=year_index, name=col)
            result = QuantShapeOptimizer(
                lambda_smooth_h=float(lambda_smooth_h),
                lambda_smooth_m=float(lambda_smooth_m),
                lambda_seam=float(lambda_seam),
                epsilon_ridge=1e-9,
                feasibility_tol=1e-6,
                stationarity_tol=1e-6,
            ).solve(prior, constraints)
            delta = result.curve.to_numpy(dtype=float) - prior.to_numpy(dtype=float)
            out.loc[idx, col] = result.curve.to_numpy(dtype=float)
            residuals = result.constraint_residuals
            audit_rows.append(
                {
                    "product": product,
                    "scenario": scenario,
                    "lambda_smooth_h": float(lambda_smooth_h),
                    "lambda_smooth_m": float(lambda_smooth_m),
                    "lambda_seam": float(lambda_seam),
                    "prior_rmse_eur_mwh": float(np.sqrt(np.mean(delta * delta))),
                    "max_abs_delta_eur_mwh": float(np.max(np.abs(delta))) if len(delta) else 0.0,
                    "max_constraint_abs_error_eur_mwh": (
                        float(residuals["abs_error"].max()) if not residuals.empty else 0.0
                    ),
                    "kkt_primal_inf": float(result.kkt.primal_inf),
                    "kkt_stationarity_inf": float(result.kkt.stationarity_inf),
                    "rows": int(len(idx)),
                }
            )

    out = _recompute_weighted_fan_columns(out, weights)
    min_price = float(out[[*scenario_cols, "price_weighted_mean_eur_mwh"]].min().min())
    if min_price < float(negative_price_floor) - 1e-9:
        raise ValueError(
            f"quant annual smoothness breached negative price floor: min={min_price:.6f}, "
            f"floor={float(negative_price_floor):.6f}"
        )
    weighted_negative_hours = int((out["price_weighted_mean_eur_mwh"] < 0.0).sum())
    if weighted_negative_hours > int(max_weighted_negative_hours):
        raise ValueError(
            "quant annual smoothness produced too many weighted-mean negative hours: "
            f"{weighted_negative_hours} > {int(max_weighted_negative_hours)}"
        )
    for column in PRICE_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype(float).round(6)
    audit = pd.DataFrame(audit_rows)
    if not audit.empty:
        audit = audit.sort_values(["product", "scenario"]).reset_index(drop=True)
    return out, audit


def _annual_residual_month_targets(
    *,
    parent_year: int,
    parent_target: float,
    months: list[int],
    month_counts: dict[int, int],
    neighbor_base_forward_prices: dict[str, float],
    historical_base_deviations: dict[tuple[int, int], float] | None,
    neighbor_current_weight: float,
) -> tuple[dict[int, float], str] | None:
    raw: dict[int, float] = {}
    sources: set[str] = set()
    for m in months:
        month_key = _month_key(parent_year, m)
        quarter = ((m - 1) // 3) + 1
        quarter_key = f"{parent_year}-Q{quarter}"
        if month_key in neighbor_base_forward_prices:
            raw[m] = float(neighbor_base_forward_prices[month_key])
            sources.add("neighbor_month")
            continue
        quarter_price = neighbor_base_forward_prices.get(quarter_key)
        if quarter_price is None:
            return None
        hist_dev = 0.0
        if historical_base_deviations:
            hist_dev = float(historical_base_deviations.get((quarter, m), 0.0))
        weight = min(max(float(neighbor_current_weight), 0.0), 1.0)
        raw[m] = float(quarter_price) + (1.0 - weight) * hist_dev
        sources.add("neighbor_quarter_hist_month")
    total = sum(month_counts[m] for m in months)
    if total <= 0:
        return None
    raw_mean = sum(raw[m] * month_counts[m] for m in months) / float(total)
    targets = {m: float(raw[m] - raw_mean + parent_target) for m in months}
    return targets, "+".join(sorted(sources))


def _solve_month_path_delta(
    monthly_means: np.ndarray,
    *,
    constraints: np.ndarray,
    smoothness_lambda: float,
) -> np.ndarray:
    n = int(len(monthly_means))
    if n < 3:
        return np.zeros(n, dtype=float)
    d2 = np.zeros((n - 2, n), dtype=float)
    for i in range(n - 2):
        d2[i, i] = 1.0
        d2[i, i + 1] = -2.0
        d2[i, i + 2] = 1.0
    penalty = d2.T @ d2
    hessian = np.eye(n, dtype=float) + float(smoothness_lambda) * penalty
    rhs = -float(smoothness_lambda) * penalty @ monthly_means.astype(float)
    a = _independent_rows(np.asarray(constraints, dtype=float))
    if a.size == 0:
        return np.linalg.solve(hessian, rhs)
    kkt = np.block([[hessian, a.T], [a, np.zeros((a.shape[0], a.shape[0]), dtype=float)]])
    kkt_rhs = np.concatenate([rhs, np.zeros(a.shape[0], dtype=float)])
    try:
        solution = np.linalg.solve(kkt, kkt_rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(kkt, kkt_rhs, rcond=None)[0]
    return solution[:n]


def _independent_rows(matrix: np.ndarray, *, tol: float = 1e-10) -> np.ndarray:
    rows: list[np.ndarray] = []
    current = np.empty((0, matrix.shape[1]), dtype=float)
    rank = 0
    for row in matrix:
        if not np.isfinite(row).all() or float(np.linalg.norm(row)) <= tol:
            continue
        candidate = np.vstack([current, row])
        candidate_rank = int(np.linalg.matrix_rank(candidate, tol=tol))
        if candidate_rank > rank:
            rows.append(row)
            current = candidate
            rank = candidate_rank
    if not rows:
        return np.empty((0, matrix.shape[1]), dtype=float)
    return np.vstack(rows)


def calibrate_hourly_to_eex_base_peak(
    hourly: pd.DataFrame,
    *,
    base_forward_prices: dict[str, float],
    peak_forward_prices: dict[str, float],
    weights: dict[str, float],
    negative_price_floor: float = -30.0,
    max_weighted_negative_hours: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Force direct EEX PEAK quotes while preserving BASE bucket means.

    The adjustment is additive and energy-conserving inside each quote-aware
    BASE bucket. PEAK targets are selected with the same Month > Quarter > Cal
    logic but only over contractual EEX peak hours.
    """
    if not peak_forward_prices:
        raise ValueError("EEX BASE+PEAK calibration requested but no PEAK prices are available")
    out = hourly.copy()
    ts_ch = _parse_timestamp_ch(out["timestamp_ch"], out.get("utc_offset_ch"))
    base_products, base_targets = _calibration_buckets(ts_ch, base_forward_prices)
    peak_mask = _eex_peak_mask(ts_ch, country="CH")
    peak_products, peak_targets = _calibration_buckets(ts_ch.loc[peak_mask], peak_forward_prices)
    if not peak_targets:
        raise ValueError("EEX BASE+PEAK calibration requested but no PEAK targets cover the delivery window")

    price_columns = _price_level_columns(out)
    rows: list[dict[str, object]] = []
    for base_product, base_idx in base_products.groupby(base_products, dropna=True).groups.items():
        if base_product is None or base_product not in base_targets:
            continue
        base_idx = pd.Index(base_idx)
        product_peak_idx = base_idx.intersection(peak_products.index)
        product_offpeak_idx = base_idx.difference(product_peak_idx)
        if len(product_peak_idx) == 0:
            continue
        if len(product_offpeak_idx) == 0:
            raise ValueError(f"cannot preserve BASE for {base_product}: no offpeak hours")

        target_peak_energy = 0.0
        peak_adjustments: list[tuple[pd.Index, str, float, float, float]] = []
        for peak_product, peak_idx in peak_products.loc[product_peak_idx].groupby(
            peak_products.loc[product_peak_idx],
            dropna=True,
        ).groups.items():
            if peak_product is None or peak_product not in peak_targets:
                continue
            peak_idx = pd.Index(peak_idx)
            current_peak = float(out.loc[peak_idx, "price_weighted_mean_eur_mwh"].mean())
            target_peak = float(peak_targets[str(peak_product)])
            if not np.isfinite(current_peak):
                raise ValueError(f"cannot calibrate PEAK {peak_product}: non-finite current mean")
            peak_delta = target_peak - current_peak
            for column in price_columns:
                out.loc[peak_idx, column] = out.loc[peak_idx, column].astype(float) + peak_delta
            target_peak_energy += target_peak * len(peak_idx)
            peak_adjustments.append((peak_idx, str(peak_product), target_peak, current_peak, peak_delta))

        if not peak_adjustments:
            continue
        current_offpeak_mean = float(out.loc[product_offpeak_idx, "price_weighted_mean_eur_mwh"].mean())
        target_base_energy = float(base_targets[str(base_product)]) * len(base_idx)
        target_offpeak_mean = (target_base_energy - target_peak_energy) / len(product_offpeak_idx)
        offpeak_delta = target_offpeak_mean - current_offpeak_mean
        for column in price_columns:
            out.loc[product_offpeak_idx, column] = out.loc[product_offpeak_idx, column].astype(float) + offpeak_delta

        out = _recompute_weighted_fan_columns(out, weights)
        min_price = float(out.loc[base_idx, _price_level_columns(out)].min().min())
        if min_price < float(negative_price_floor) - 1e-9:
            raise ValueError(
                f"EEX BASE+PEAK calibration breached floor for {base_product}: "
                f"min={min_price:.6f}, floor={float(negative_price_floor):.6f}"
            )
        weighted_negative_hours = int((out["price_weighted_mean_eur_mwh"] < 0.0).sum())
        if weighted_negative_hours > int(max_weighted_negative_hours):
            raise ValueError(
                "EEX BASE+PEAK calibration produced too many weighted-mean negative hours: "
                f"{weighted_negative_hours} > {int(max_weighted_negative_hours)}"
            )

        post_base = float(out.loc[base_idx, "price_weighted_mean_eur_mwh"].mean())
        for peak_idx, peak_product, target_peak, pre_peak, peak_delta in peak_adjustments:
            post_peak = float(out.loc[peak_idx, "price_weighted_mean_eur_mwh"].mean())
            rows.append(
                {
                    "base_product": str(base_product),
                    "peak_product": peak_product,
                    "target_eex_base_eur_mwh": float(base_targets[str(base_product)]),
                    "post_base_mean_eur_mwh": post_base,
                    "base_residual_eur_mwh": post_base - float(base_targets[str(base_product)]),
                    "target_eex_peak_eur_mwh": target_peak,
                    "pre_peak_mean_eur_mwh": pre_peak,
                    "post_peak_mean_eur_mwh": post_peak,
                    "peak_residual_eur_mwh": post_peak - target_peak,
                    "peak_delta_eur_mwh": peak_delta,
                    "offpeak_delta_eur_mwh": offpeak_delta,
                    "peak_hours": len(peak_idx),
                    "offpeak_hours": len(product_offpeak_idx),
                    "min_price_eur_mwh": min_price,
                }
            )

    out = _recompute_weighted_fan_columns(out, weights)
    for column in PRICE_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype(float).round(6)
    return out, pd.DataFrame(rows).sort_values(["base_product", "peak_product"]).reset_index(drop=True)


def _season_name(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _parse_weights(value: str, scenarios: tuple[str, ...] = ("slow", "central", "fast")) -> dict[str, float]:
    raw = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if len(raw) != len(scenarios):
        raise ValueError(f"expected {len(scenarios)} weights, got {len(raw)}")
    if any((not np.isfinite(weight)) or weight < 0.0 for weight in raw):
        raise ValueError("weights must be finite and non-negative")
    total = float(sum(raw))
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return {scenario: weight / total for scenario, weight in zip(scenarios, raw)}


def _swiss_hydro_base_delta(ts: pd.Timestamp, *, intensity: float) -> float:
    """Common CH hydro/PV shape overlay in EUR/MWh before product de-meaning."""
    hour = int(ts.hour)
    season = _season_name(int(ts.month))
    nonworking_pressure = min(1.0, _ch_market_nonworking_pressure(ts))
    delta = 0.0

    if season == "winter":
        if 7 <= hour <= 9:
            delta += 3.0
        elif 18 <= hour <= 21:
            delta += 7.0
        elif 10 <= hour <= 15:
            delta -= 2.0
        elif hour in {0, 1, 2, 3, 4, 5}:
            delta -= 2.5
    elif season == "spring":
        if 10 <= hour <= 16:
            delta -= 6.0
        elif 17 <= hour <= 21:
            delta += 4.5
        elif 7 <= hour <= 9:
            delta += 2.0
        if 10 <= hour <= 16:
            delta -= 2.0 * nonworking_pressure
    elif season == "summer":
        if 10 <= hour <= 16:
            delta -= 7.5
        elif 17 <= hour <= 22:
            delta += 5.5
        elif 7 <= hour <= 9:
            delta += 1.5
        if 10 <= hour <= 16:
            delta -= 2.5 * nonworking_pressure
    else:
        if 10 <= hour <= 15:
            delta -= 3.5
        elif 17 <= hour <= 21:
            delta += 5.0
        elif 7 <= hour <= 9:
            delta += 2.5

    if 18 <= hour <= 21:
        delta -= 1.0 * nonworking_pressure

    return float(np.clip(delta * float(intensity), -12.0, 12.0))


def _scenario_spread_delta(ts: pd.Timestamp, scenario: str, *, intensity: float) -> float:
    """Slow/fast structural dispersion in EUR/MWh before product de-meaning."""
    score = {"slow": -1.0, "central": 0.0, "fast": 1.0}[scenario]
    if score == 0.0 or intensity == 0.0:
        return 0.0

    hour = int(ts.hour)
    season = _season_name(int(ts.month))
    nonworking_pressure = min(1.0, _ch_market_nonworking_pressure(ts))
    delta = 0.0

    if 10 <= hour <= 15:
        delta += {"winter": -1.5, "spring": -6.0, "summer": -7.0, "autumn": -4.5}[season] * score
        if season in {"spring", "summer"}:
            delta += -1.5 * score * nonworking_pressure
    if 6 <= hour <= 8:
        delta += {"winter": 3.0, "spring": 1.8, "summer": 1.2, "autumn": 2.2}[season] * score
    if 17 <= hour <= 21:
        delta += {"winter": 4.0, "spring": 3.2, "summer": 3.0, "autumn": 3.5}[season] * score
    if hour in {22, 23, 0, 1}:
        delta += {"winter": 1.5, "spring": 0.8, "summer": 0.6, "autumn": 1.0}[season] * score

    return float(np.clip(delta * float(intensity), -10.0, 10.0))


def _negative_capture_delta(ts: pd.Timestamp, scenario: str, *, intensity: float) -> float:
    """PV capture stress that can create bounded local/test negative prices."""
    if intensity == 0.0 or int(ts.year) < 2028 or int(ts.month) not in {4, 5, 6, 7, 8, 9}:
        return 0.0
    if int(ts.hour) not in {10, 11, 12, 13, 14, 15}:
        return 0.0

    scenario_factor = {"slow": 0.15, "central": 0.55, "fast": 1.00}[scenario]
    year_factor = {2028: 0.70, 2029: 0.90, 2030: 1.10}.get(int(ts.year), 1.10)
    month_factor = 1.00 if int(ts.month) in {5, 6, 7, 8} else 0.75
    day_factor = 0.35 + 0.65 * min(1.0, _ch_market_nonworking_pressure(ts))
    hour_factor = 1.00 if int(ts.hour) in {12, 13, 14} else 0.65
    raw = -28.0 * scenario_factor * year_factor * month_factor * day_factor * hour_factor
    return float(np.clip(raw * float(intensity), -35.0, 0.0))


def _weighted_quantile_row(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    x = values[order]
    w = weights[order]
    cdf = np.cumsum(w) / float(w.sum())
    return float(x[np.searchsorted(cdf, q, side="left")])


def apply_local_test_structural_shape_upgrade(
    hourly: pd.DataFrame,
    *,
    forward_prices: dict[str, float],
    weights: dict[str, float],
    intensity: float = 1.0,
    spread_intensity: float = 1.0,
    enable_negative_price_capture: bool = False,
    negative_capture_intensity: float = 1.0,
    negative_price_floor: float = -30.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Widen local/test scenario shape while preserving product means.

    The overlay stresses Swiss hydro evening value, PV belly, winter
    electrification, weekend capture and, when explicitly enabled, bounded
    negative-price PV capture. It is explicitly non-production and additive;
    every scenario is de-meaned by calibration product before fan-chart
    recomputation.
    """
    out = hourly.copy()
    ts_ch = _parse_timestamp_ch(out["timestamp_ch"], out.get("utc_offset_ch"))
    products = pd.Series([_calibration_product(ts, forward_prices) for ts in ts_ch], index=out.index)
    if products.isna().any():
        missing = int(products.isna().sum())
        raise ValueError(f"cannot apply shape upgrade: {missing} rows have no calibration product")

    audit_rows: list[dict[str, object]] = []
    for scenario, column in SCENARIO_PRICE_COLUMNS.items():
        base = out[column].astype(float).copy()
        raw_delta = np.array(
            [
                _swiss_hydro_base_delta(ts, intensity=intensity)
                + _scenario_spread_delta(ts, scenario, intensity=spread_intensity)
                + (
                    _negative_capture_delta(ts, scenario, intensity=negative_capture_intensity)
                    if enable_negative_price_capture
                    else 0.0
                )
                for ts in ts_ch
            ],
            dtype=float,
        )
        delta_s = pd.Series(raw_delta, index=out.index)
        stressed_s = base.copy()
        for product, idx in products.groupby(products).groups.items():
            before = float(base.loc[idx].mean())
            demeaned = delta_s.loc[idx] - float(delta_s.loc[idx].mean())
            cap_ceiling = 24.0 if enable_negative_price_capture else 16.0
            cap_share = 0.28 if enable_negative_price_capture else 0.20
            cap_floor = 8.0 if enable_negative_price_capture else 3.0
            cap = min(cap_ceiling, max(cap_floor, abs(before) * cap_share))
            demeaned = demeaned.clip(lower=-cap, upper=cap)
            demeaned = demeaned - float(demeaned.mean())
            stressed_s.loc[idx] = base.loc[idx] + demeaned
            min_price = float(stressed_s.loc[idx].min())
            if min_price < float(negative_price_floor):
                raise ValueError(
                    f"shape upgrade breached negative price floor for {scenario}/{product}: "
                    f"min={min_price:.6f}, floor={float(negative_price_floor):.6f}"
                )
            audit_rows.append(
                {
                    "scenario": scenario,
                    "product": product,
                    "pre_mean_eur_mwh": before,
                    "post_mean_eur_mwh": float(stressed_s.loc[idx].mean()),
                    "max_abs_delta_eur_mwh": float(np.max(np.abs(demeaned))),
                    "min_price_eur_mwh": min_price,
                    "negative_hours": int((stressed_s.loc[idx] < 0.0).sum()),
                    "rows": len(idx),
                }
            )
        if (not enable_negative_price_capture) and (stressed_s <= 0.0).any():
            raise ValueError(f"shape upgrade produced non-positive prices for scenario={scenario}")
        out[column] = stressed_s.astype(float)

    labels = tuple(SCENARIO_PRICE_COLUMNS)
    matrix = np.column_stack([out[SCENARIO_PRICE_COLUMNS[label]].to_numpy(dtype=float) for label in labels])
    w = np.array([weights[label] for label in labels], dtype=float)
    out["price_weighted_mean_eur_mwh"] = matrix @ w
    out["structural_p10_eur_mwh"] = [_weighted_quantile_row(row, w, 0.10) for row in matrix]
    out["structural_p50_eur_mwh"] = [_weighted_quantile_row(row, w, 0.50) for row in matrix]
    out["structural_p90_eur_mwh"] = [_weighted_quantile_row(row, w, 0.90) for row in matrix]
    out["structural_width_eur_mwh"] = out["structural_p90_eur_mwh"] - out["structural_p10_eur_mwh"]
    for column in PRICE_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype(float).round(6)
    return out, pd.DataFrame(audit_rows).sort_values(["scenario", "product"]).reset_index(drop=True)


def _negative_capture_weight(ts: pd.Timestamp) -> float:
    """Rank hours where negative prices are economically plausible in CH LT."""
    year = int(ts.year)
    month = int(ts.month)
    hour = int(ts.hour)
    if year < 2028 or month not in {4, 5, 6, 7, 8, 9} or hour not in {10, 11, 12, 13, 14, 15}:
        return 0.0

    year_weight = {2028: 0.55, 2029: 0.80, 2030: 1.00}.get(year, 1.00)
    month_weight = 1.00 if month in {5, 6, 7, 8} else 0.65
    hour_weight = {10: 0.45, 11: 0.75, 12: 1.00, 13: 1.00, 14: 0.90, 15: 0.55}[hour]
    day_weight = 0.35 + 0.65 * min(1.0, _ch_market_nonworking_pressure(ts))
    return float(year_weight * month_weight * hour_weight * day_weight)


def _negative_capture_compensation_weight(ts: pd.Timestamp) -> float:
    """Rank same-bucket hours that can absorb a mean-preserving uplift."""
    hour = int(ts.hour)
    month = int(ts.month)
    weight = 0.0
    if 17 <= hour <= 21:
        weight += 1.00
    if 7 <= hour <= 9:
        weight += 0.35
    if month in {11, 12, 1, 2, 3} and hour in {6, 7, 8, 17, 18, 19, 20, 21, 22}:
        weight += 0.35
    if 17 <= hour <= 21:
        weight += 0.25 * max(0.0, 1.0 - min(1.0, _ch_market_nonworking_pressure(ts)))
    return float(weight)


def apply_post_calibration_negative_rebalancer(
    hourly: pd.DataFrame,
    *,
    forward_prices: dict[str, float],
    weights: dict[str, float],
    intensity: float = 1.0,
    weighted_negative_capture_intensity: float = 0.0,
    negative_price_floor: float = -30.0,
    max_weighted_negative_hours: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create bounded negative-price tails after EEX calibration.

    The rebalancer is local/test only.  It moves value inside each quote-aware
    EEX bucket from high-PV midday hours to compensation hours, with exactly
    zero mean delta per scenario and bucket before final rounding.
    """
    if intensity < 0.0 or not np.isfinite(float(intensity)):
        raise ValueError("post-calibration negative rebalancer intensity must be finite and non-negative")
    if weighted_negative_capture_intensity < 0.0 or not np.isfinite(float(weighted_negative_capture_intensity)):
        raise ValueError("weighted negative capture intensity must be finite and non-negative")
    out = hourly.copy()
    ts_ch = _parse_timestamp_ch(out["timestamp_ch"], out.get("utc_offset_ch"))
    products, calibration_targets = _calibration_buckets(ts_ch, forward_prices)
    if products.isna().any():
        missing = int(products.isna().sum())
        raise ValueError(f"cannot apply post-calibration negative rebalancer: {missing} rows have no EEX bucket")

    scenario_depth = {
        "slow": 0.0,
        "central": 7.0 * float(weighted_negative_capture_intensity),
        "fast": 14.0,
    }
    scenario_up_cap = {
        "slow": 0.0,
        "central": 14.0,
        "fast": 22.0,
    }
    low_weight = pd.Series([_negative_capture_weight(ts) for ts in ts_ch], index=out.index, dtype=float)
    comp_weight = pd.Series([_negative_capture_compensation_weight(ts) for ts in ts_ch], index=out.index, dtype=float)
    audit_rows: list[dict[str, object]] = []

    for scenario, column in SCENARIO_PRICE_COLUMNS.items():
        depth = float(scenario_depth[scenario]) * float(intensity)
        if depth <= 0.0:
            continue
        base = out[column].astype(float).copy()
        stressed = base.copy()
        for product, idx in products.groupby(products).groups.items():
            if product is None or product not in calibration_targets:
                continue
            idx = pd.Index(idx)
            low = low_weight.loc[idx].copy()
            if float(low.sum()) <= 0.0:
                continue
            low = low / float(low.max())
            down = -(depth * low)
            candidate = base.loc[idx] + down
            breach = candidate < float(negative_price_floor)
            if bool(breach.any()):
                allowed_down = float(negative_price_floor) - base.loc[idx]
                down = down.where(~breach, allowed_down)
            total_down = float(-down.sum())
            if total_down <= 0.0:
                continue

            comp = comp_weight.loc[idx].copy()
            comp = comp.where(low <= 0.0, 0.0)
            if float(comp.sum()) <= 0.0:
                comp = pd.Series(1.0, index=idx, dtype=float).where(low <= 0.0, 0.0)
            if float(comp.sum()) <= 0.0:
                comp = pd.Series(1.0, index=idx, dtype=float)
            up = comp / float(comp.sum()) * total_down
            max_up = float(up.max())
            cap = float(scenario_up_cap[scenario])
            if max_up > cap > 0.0:
                scale = cap / max_up
                down = down * scale
                total_down = float(-down.sum())
                up = comp / float(comp.sum()) * total_down

            delta = pd.Series(0.0, index=idx, dtype=float)
            delta = delta.add(down, fill_value=0.0).add(up, fill_value=0.0)
            delta = delta - float(delta.mean())
            stressed.loc[idx] = base.loc[idx] + delta
            min_price = float(stressed.loc[idx].min())
            if min_price < float(negative_price_floor) - 1e-9:
                raise ValueError(
                    f"post-calibration negative rebalancer breached floor for {scenario}/{product}: "
                    f"min={min_price:.6f}, floor={float(negative_price_floor):.6f}"
                )
            audit_rows.append(
                {
                    "scenario": scenario,
                    "product": product,
                    "target_eex_base_eur_mwh": float(calibration_targets[str(product)]),
                    "pre_mean_eur_mwh": float(base.loc[idx].mean()),
                    "post_mean_eur_mwh": float(stressed.loc[idx].mean()),
                    "max_down_eur_mwh": float(delta.min()),
                    "max_up_eur_mwh": float(delta.max()),
                    "min_price_eur_mwh": min_price,
                    "negative_hours": int((stressed.loc[idx] < 0.0).sum()),
                    "rows": len(idx),
                }
            )
        out[column] = stressed.astype(float)

    labels = tuple(SCENARIO_PRICE_COLUMNS)
    matrix = np.column_stack([out[SCENARIO_PRICE_COLUMNS[label]].to_numpy(dtype=float) for label in labels])
    w = np.array([weights[label] for label in labels], dtype=float)
    out["price_weighted_mean_eur_mwh"] = matrix @ w
    weighted_negative_hours = int((out["price_weighted_mean_eur_mwh"] < 0.0).sum())
    if weighted_negative_hours > int(max_weighted_negative_hours):
        raise ValueError(
            "post-calibration negative rebalancer produced too many weighted-mean negative hours: "
            f"{weighted_negative_hours} > {int(max_weighted_negative_hours)}"
        )
    out["structural_p10_eur_mwh"] = [_weighted_quantile_row(row, w, 0.10) for row in matrix]
    out["structural_p50_eur_mwh"] = [_weighted_quantile_row(row, w, 0.50) for row in matrix]
    out["structural_p90_eur_mwh"] = [_weighted_quantile_row(row, w, 0.90) for row in matrix]
    out["structural_width_eur_mwh"] = out["structural_p90_eur_mwh"] - out["structural_p10_eur_mwh"]
    for column in PRICE_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype(float).round(6)
    return out, pd.DataFrame(audit_rows).sort_values(["scenario", "product"]).reset_index(drop=True)


def _peak_shape_up_weight(ts: pd.Timestamp) -> float:
    workday_pressure = max(0.0, 1.0 - min(1.0, _ch_market_nonworking_pressure(ts)))
    hour = int(ts.hour)
    if hour in {17, 18, 19}:
        return 1.00 * workday_pressure
    if hour in {8, 9}:
        return 0.55 * workday_pressure
    if hour == 10:
        return 0.25 * workday_pressure
    return 0.0


def _peak_shape_down_weight(ts: pd.Timestamp) -> float:
    hour = int(ts.hour)
    weight = 0.0
    if hour in {20, 21, 22, 23, 0, 1, 2, 3, 4, 5}:
        weight += 1.00
    weight += 0.45 * min(1.0, _ch_market_nonworking_pressure(ts))
    return float(weight)


def apply_post_calibration_peak_shape_rebalancer(
    hourly: pd.DataFrame,
    *,
    forward_prices: dict[str, float],
    weights: dict[str, float],
    intensity: float = 1.0,
    negative_price_floor: float = -30.0,
    max_weighted_negative_hours: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Move value from offpeak/night into weekday peak shoulders by EEX bucket."""
    if intensity < 0.0 or not np.isfinite(float(intensity)):
        raise ValueError("post-calibration peak shape intensity must be finite and non-negative")
    out = hourly.copy()
    ts_ch = _parse_timestamp_ch(out["timestamp_ch"], out.get("utc_offset_ch"))
    products, calibration_targets = _calibration_buckets(ts_ch, forward_prices)
    up_weight = pd.Series([_peak_shape_up_weight(ts) for ts in ts_ch], index=out.index, dtype=float)
    down_weight = pd.Series([_peak_shape_down_weight(ts) for ts in ts_ch], index=out.index, dtype=float)
    audit_rows: list[dict[str, object]] = []
    target_total_by_scenario = {"slow": 2.0, "central": 3.0, "fast": 4.0}

    for scenario, column in SCENARIO_PRICE_COLUMNS.items():
        base = out[column].astype(float).copy()
        stressed = base.copy()
        target_total = float(target_total_by_scenario[scenario]) * float(intensity)
        if target_total <= 0.0:
            continue
        for product, idx in products.groupby(products).groups.items():
            if product is None or product not in calibration_targets:
                continue
            idx = pd.Index(idx)
            up = up_weight.loc[idx]
            down = down_weight.loc[idx].where(up_weight.loc[idx] <= 0.0, 0.0)
            if float(up.sum()) <= 0.0 or float(down.sum()) <= 0.0:
                continue
            up_delta = up / float(up.max()) * target_total
            total_up = float(up_delta.sum())
            down_delta = -(down / float(down.sum()) * total_up)
            delta = pd.Series(0.0, index=idx, dtype=float).add(up_delta, fill_value=0.0).add(
                down_delta, fill_value=0.0
            )
            delta = delta - float(delta.mean())
            stressed.loc[idx] = base.loc[idx] + delta
            min_price = float(stressed.loc[idx].min())
            if min_price < float(negative_price_floor) - 1e-9:
                raise ValueError(
                    f"post-calibration peak shape rebalancer breached floor for {scenario}/{product}: "
                    f"min={min_price:.6f}, floor={float(negative_price_floor):.6f}"
                )
            audit_rows.append(
                {
                    "scenario": scenario,
                    "product": product,
                    "target_eex_base_eur_mwh": float(calibration_targets[str(product)]),
                    "pre_mean_eur_mwh": float(base.loc[idx].mean()),
                    "post_mean_eur_mwh": float(stressed.loc[idx].mean()),
                    "max_down_eur_mwh": float(delta.min()),
                    "max_up_eur_mwh": float(delta.max()),
                    "min_price_eur_mwh": min_price,
                    "rows": len(idx),
                }
            )
        out[column] = stressed.astype(float)

    labels = tuple(SCENARIO_PRICE_COLUMNS)
    matrix = np.column_stack([out[SCENARIO_PRICE_COLUMNS[label]].to_numpy(dtype=float) for label in labels])
    w = np.array([weights[label] for label in labels], dtype=float)
    out["price_weighted_mean_eur_mwh"] = matrix @ w
    weighted_negative_hours = int((out["price_weighted_mean_eur_mwh"] < 0.0).sum())
    if weighted_negative_hours > int(max_weighted_negative_hours):
        raise ValueError(
            "post-calibration peak shape rebalancer produced too many weighted-mean negative hours: "
            f"{weighted_negative_hours} > {int(max_weighted_negative_hours)}"
        )
    out["structural_p10_eur_mwh"] = [_weighted_quantile_row(row, w, 0.10) for row in matrix]
    out["structural_p50_eur_mwh"] = [_weighted_quantile_row(row, w, 0.50) for row in matrix]
    out["structural_p90_eur_mwh"] = [_weighted_quantile_row(row, w, 0.90) for row in matrix]
    out["structural_width_eur_mwh"] = out["structural_p90_eur_mwh"] - out["structural_p10_eur_mwh"]
    for column in PRICE_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype(float).round(6)
    return out, pd.DataFrame(audit_rows).sort_values(["scenario", "product"]).reset_index(drop=True)


def _md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_none_"
    lines = [
        "| " + " | ".join(str(column) for column in frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    path: Path,
    *,
    csv_output: Path,
    fan_output: Path,
    hourly: pd.DataFrame,
    local_start: str,
    local_end: str,
    forward_date: str,
    required_forward_date: str | None,
    calibration: pd.DataFrame,
    shape_upgrade_enabled: bool = False,
    shape_upgrade_audit: pd.DataFrame | None = None,
    negative_price_capture_enabled: bool = False,
    post_calibration_negative_rebalancer_enabled: bool = False,
    weighted_negative_capture_intensity: float = 0.0,
    post_calibration_negative_rebalancer_audit: pd.DataFrame | None = None,
    post_calibration_peak_shape_rebalancer_enabled: bool = False,
    post_calibration_peak_shape_rebalancer_audit: pd.DataFrame | None = None,
    quote_aware_monthly_smoothing_enabled: bool = False,
    quote_aware_monthly_smoothing_audit: pd.DataFrame | None = None,
    final_monthly_path_smoothing_enabled: bool = False,
    final_monthly_path_smoothing_audit: pd.DataFrame | None = None,
    annual_residual_anchor_enabled: bool = False,
    annual_residual_anchor_audit: pd.DataFrame | None = None,
    final_quant_annual_smoothness_enabled: bool = False,
    final_quant_annual_smoothness_audit: pd.DataFrame | None = None,
    cross_year_seasonal_shape_enabled: bool = False,
    cross_year_seasonal_shape_audit: pd.DataFrame | None = None,
    seam_nullspace_smoothing_enabled: bool = False,
    seam_nullspace_smoothing_audit: pd.DataFrame | None = None,
    neighbor_monthly_anchor_enabled: bool = False,
    neighbor_monthly_anchor_audit: pd.DataFrame | None = None,
    eex_peak_calibration_enabled: bool = False,
    eex_peak_calibration_audit: pd.DataFrame | None = None,
    disable_cascade_trend_for_annual_only: bool = False,
) -> None:
    price = hourly["price_weighted_mean_eur_mwh"]
    if "structural_width_eur_mwh" in hourly.columns:
        width = hourly["structural_width_eur_mwh"]
    elif {"structural_p90_eur_mwh", "structural_p10_eur_mwh"}.issubset(hourly.columns):
        width = hourly["structural_p90_eur_mwh"] - hourly["structural_p10_eur_mwh"]
    elif {"price_fast_eur_mwh", "price_slow_eur_mwh"}.issubset(hourly.columns):
        width = hourly["price_fast_eur_mwh"] - hourly["price_slow_eur_mwh"]
    else:
        width = pd.Series(0.0, index=hourly.index)
    lines = [
        "# Local-Test CH PFC Hourly CSV",
        "",
        "* status: `agent-approved local/test only`",
        "* production approval from this local report: `NO - use manifest-backed capstone for promotion evidence`",
        f"* CSV: `{csv_output}`",
        f"* source fan chart: `{fan_output}`",
        f"* local window: `{local_start}` to `{local_end}` Europe/Zurich",
        f"* EEX CH forward date: `{forward_date}`",
        f"* required EEX CH forward date: `{required_forward_date or 'not enforced'}`",
        f"* local/test structural shape upgrade: `{'ON' if shape_upgrade_enabled else 'OFF'}`",
        f"* local/test negative price capture: `{'ON' if negative_price_capture_enabled else 'OFF'}`",
        (
            "* local/test post-calibration negative rebalancer: "
            f"`{'ON' if post_calibration_negative_rebalancer_enabled else 'OFF'}`"
        ),
        f"* local/test weighted negative capture intensity: `{weighted_negative_capture_intensity:.6f}`",
        (
            "* local/test post-calibration peak shape rebalancer: "
            f"`{'ON' if post_calibration_peak_shape_rebalancer_enabled else 'OFF'}`"
        ),
        (
            "* local/test quote-aware monthly smoothing: "
            f"`{'ON' if quote_aware_monthly_smoothing_enabled else 'OFF'}`"
        ),
        (
            "* local/test final monthly path smoothing: "
            f"`{'ON' if final_monthly_path_smoothing_enabled else 'OFF'}`"
        ),
        (
            "* local/test neighbor annual residual shape anchor: "
            f"`{'ON' if annual_residual_anchor_enabled else 'OFF'}`"
        ),
        (
            "* local/test final quant annual-only smoothness calibration: "
            f"`{'ON' if final_quant_annual_smoothness_enabled else 'OFF'}`"
        ),
        (
            "* local/test cross-year seasonal shape optimizer: "
            f"`{'ON' if cross_year_seasonal_shape_enabled else 'OFF'}`"
        ),
        (
            "* local/test final seam nullspace smoothing: "
            f"`{'ON' if seam_nullspace_smoothing_enabled else 'OFF'}`"
        ),
        (
            "* local/test neighbor monthly spread anchor: "
            f"`{'ON' if neighbor_monthly_anchor_enabled else 'OFF'}`"
        ),
        f"* local/test EEX BASE+PEAK calibration: `{'ON' if eex_peak_calibration_enabled else 'OFF'}`",
        f"* disable cascade trend for annual-only years: `{'ON' if disable_cascade_trend_for_annual_only else 'OFF'}`",
        f"* rows: `{len(hourly)}`",
        "",
        "## Price Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| mean | {price.mean():.6f} |",
        f"| min | {price.min():.6f} |",
        f"| p05 | {price.quantile(0.05):.6f} |",
        f"| p95 | {price.quantile(0.95):.6f} |",
        f"| max | {price.max():.6f} |",
        "",
        "## Structural Width",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| mean | {width.mean():.6f} |",
        f"| p95 | {width.quantile(0.95):.6f} |",
        f"| max | {width.max():.6f} |",
        "",
        "## Columns",
        "",
        ", ".join(f"`{column}`" for column in hourly.columns),
        "",
        "## EEX Calibration",
        "",
        _md_table(calibration),
        "",
        "## Local/Test Shape Upgrade Audit",
        "",
        _md_table(shape_upgrade_audit if shape_upgrade_audit is not None else pd.DataFrame()),
        "",
        "## Local/Test Post-Calibration Negative Rebalancer Audit",
        "",
        _md_table(
            post_calibration_negative_rebalancer_audit
            if post_calibration_negative_rebalancer_audit is not None
            else pd.DataFrame()
        ),
        "",
        "## Local/Test Post-Calibration Peak Shape Rebalancer Audit",
        "",
        _md_table(
            post_calibration_peak_shape_rebalancer_audit
            if post_calibration_peak_shape_rebalancer_audit is not None
            else pd.DataFrame()
        ),
        "",
        "## Local/Test Quote-Aware Monthly Smoothing Audit",
        "",
        _md_table(
            quote_aware_monthly_smoothing_audit
            if quote_aware_monthly_smoothing_audit is not None
            else pd.DataFrame()
        ),
        "",
        "## Local/Test Final Monthly Path Smoothing Audit",
        "",
        _md_table(
            final_monthly_path_smoothing_audit
            if final_monthly_path_smoothing_audit is not None
            else pd.DataFrame()
        ),
        "",
        "## Local/Test Neighbor Annual Residual Shape Anchor Audit",
        "",
        _md_table(
            annual_residual_anchor_audit
            if annual_residual_anchor_audit is not None
            else pd.DataFrame()
        ),
        "",
        "## Local/Test Final Quant Annual-Only Smoothness Calibration Audit",
        "",
        _md_table(
            final_quant_annual_smoothness_audit
            if final_quant_annual_smoothness_audit is not None
            else pd.DataFrame()
        ),
        "",
        "## Local/Test Cross-Year Seasonal Shape Optimizer Audit",
        "",
        _md_table(
            cross_year_seasonal_shape_audit
            if cross_year_seasonal_shape_audit is not None
            else pd.DataFrame()
        ),
        "",
        "## Local/Test Final Seam Nullspace Smoothing Audit",
        "",
        _md_table(
            seam_nullspace_smoothing_audit
            if seam_nullspace_smoothing_audit is not None
            else pd.DataFrame()
        ),
        "",
        "## Local/Test Neighbor Monthly Spread Anchor Audit",
        "",
        _md_table(
            neighbor_monthly_anchor_audit
            if neighbor_monthly_anchor_audit is not None
            else pd.DataFrame()
        ),
        "",
        "## Local/Test EEX BASE+PEAK Calibration Audit",
        "",
        _md_table(
            eex_peak_calibration_audit
            if eex_peak_calibration_audit is not None
            else pd.DataFrame()
        ),
        "",
        "## Limitations",
        "",
        "* Hourly values are arithmetic averages of the calibrated 15-minute local/test PFC.",
        "* The curve is calibrated to the required EEX CH forward snapshot for the run date.",
        "* The structural shape upgrade is local/test only and remains disabled unless explicitly requested.",
        "* Negative-price capture, when enabled, is local/test only and bounded by an explicit floor.",
        "* The post-calibration negative rebalancer, when enabled, is local/test only and mean-preserving by EEX bucket.",
        "* The post-calibration peak shape rebalancer, when enabled, is local/test only and mean-preserving by EEX bucket.",
        "* The quote-aware monthly smoothing, when enabled, is local/test only and preserves EEX BASE/PEAK buckets.",
        "* The final monthly path smoothing, when enabled, is local/test only and preserves EEX BASE/PEAK buckets.",
        "* The neighbor annual residual shape anchor, when enabled, is local/test only and preserves EEX BASE buckets.",
        "* The final quant annual-only smoothness calibration, when enabled, is local/test only and preserves annual BASE/PEAK buckets.",
        "* The final seam nullspace smoothing, when enabled, is local/test only and preserves active BASE/PEAK bucket means.",
        "* The neighbor monthly spread anchor, when enabled, is local/test only and preserves EEX BASE/PEAK buckets.",
        "* The EEX BASE+PEAK calibration, when enabled, is local/test only and uses quoted PEAK products from the same snapshot.",
        (
            "* This local/test report is not the promotion authority; use the selected-config "
            "artifact plus manifest-backed capstone for production promotion evidence."
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-start-date", default=DEFAULT_LOCAL_START)
    parser.add_argument("--local-end-date", default=DEFAULT_LOCAL_END)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--prefix", default="local_test_ch_pfc_20260613_20301231")
    parser.add_argument("--forwards", default="data/eex_forwards_history.parquet")
    parser.add_argument("--weights", default="0.25,0.50,0.25")
    parser.add_argument(
        "--valuation-date",
        default=str(default_valuation_date().date()),
        help="Run valuation date used to require previous-business-day EEX forwards.",
    )
    parser.add_argument(
        "--required-forward-date",
        default=None,
        help="Explicit required EEX CH BASE date (YYYY-MM-DD). Overrides --valuation-date.",
    )
    parser.add_argument(
        "--allow-stale-forwards",
        action="store_true",
        help="Do not enforce previous-business-day EEX freshness. Use only for diagnostics/backtests.",
    )
    parser.add_argument(
        "--enable-structural-shape-upgrade",
        action="store_true",
        help="Enable a local/test non-production overlay that widens structural scenario shape.",
    )
    parser.add_argument(
        "--disable-cascade-trend-for-annual-only",
        action="store_true",
        help=(
            "Local/test guard: do not extrapolate fitted seasonal trends when a delivery "
            "year has only a Cal quote. Quoted months/quarters remain untouched."
        ),
    )
    parser.add_argument(
        "--structural-shape-upgrade-intensity",
        type=float,
        default=1.0,
        help="Multiplier for the local/test structural shape overlay.",
    )
    parser.add_argument(
        "--structural-scenario-spread-intensity",
        type=float,
        default=1.0,
        help="Multiplier for the slow/fast local-test structural fan spread.",
    )
    parser.add_argument(
        "--enable-negative-price-capture",
        action="store_true",
        help="Enable bounded local/test PV capture that may create negative prices.",
    )
    parser.add_argument(
        "--negative-price-capture-intensity",
        type=float,
        default=1.0,
        help="Multiplier for bounded local/test negative-price capture.",
    )
    parser.add_argument(
        "--negative-price-floor",
        type=float,
        default=-30.0,
        help="Local/test hard floor for generated negative prices.",
    )
    parser.add_argument(
        "--enable-post-calibration-negative-rebalancer",
        action="store_true",
        help=(
            "Enable a local/test post-EEX-calibration rebalancer that creates bounded "
            "negative-price tails while preserving each quote-aware EEX bucket mean."
        ),
    )
    parser.add_argument(
        "--post-calibration-negative-rebalancer-intensity",
        type=float,
        default=1.0,
        help="Multiplier for the local/test post-calibration negative rebalancer.",
    )
    parser.add_argument(
        "--weighted-negative-capture-intensity",
        type=float,
        default=0.0,
        help=(
            "Opt-in local/test central-scenario negative capture intensity. "
            "A positive value can create bounded negative prices in "
            "price_weighted_mean_eur_mwh while preserving EEX bucket means."
        ),
    )
    parser.add_argument(
        "--max-weighted-negative-hours",
        type=int,
        default=0,
        help="Maximum allowed negative hours in price_weighted_mean_eur_mwh after post-calibration rebalancing.",
    )
    parser.add_argument(
        "--enable-post-calibration-peak-shape-rebalancer",
        action="store_true",
        help=(
            "Enable a local/test post-EEX-calibration rebalancer that shifts value "
            "from offpeak/night into weekday peak shoulders while preserving EEX bucket means."
        ),
    )
    parser.add_argument(
        "--post-calibration-peak-shape-intensity",
        type=float,
        default=1.0,
        help="Multiplier for the local/test post-calibration peak shape rebalancer.",
    )
    parser.add_argument(
        "--enable-eex-peak-calibration",
        action="store_true",
        help=(
            "Enable final local/test EEX BASE+PEAK calibration using quoted CH PEAK products "
            "from the same forward snapshot. Disabled by default."
        ),
    )
    parser.add_argument(
        "--enable-quote-aware-monthly-smoothing",
        action="store_true",
        help=(
            "Enable local/test constrained smoothing of implicit monthly means while preserving "
            "quote-aware EEX BASE and PEAK buckets. Disabled by default."
        ),
    )
    parser.add_argument(
        "--quote-aware-monthly-smoothing-intensity",
        type=float,
        default=1.0,
        help="Multiplier for local/test quote-aware monthly smoothing.",
    )
    parser.add_argument(
        "--quote-aware-monthly-smoothing-lambda",
        type=float,
        default=8.0,
        help="Curvature penalty for local/test quote-aware monthly smoothing.",
    )
    parser.add_argument(
        "--max-unquoted-month-delta-eur-mwh",
        type=float,
        default=18.0,
        help="Maximum absolute monthly delta introduced by quote-aware monthly smoothing.",
    )
    parser.add_argument(
        "--enable-neighbor-monthly-spread-anchor",
        action="store_true",
        help=(
            "Enable local/test CH monthly split anchoring to a neighbor market "
            "for unquoted quarterly months while preserving CH EEX BASE/PEAK buckets."
        ),
    )
    parser.add_argument(
        "--neighbor-monthly-market",
        default="DE",
        help="Neighbor market used for monthly spread anchoring (default: DE).",
    )
    parser.add_argument(
        "--neighbor-monthly-anchor-intensity",
        type=float,
        default=1.0,
        help="Multiplier for local/test neighbor monthly split anchoring.",
    )
    parser.add_argument(
        "--neighbor-monthly-current-weight",
        type=float,
        default=0.65,
        help="Blend weight on current neighbor monthly deviations; remaining weight uses historical CH deviations.",
    )
    parser.add_argument(
        "--neighbor-monthly-historical-market",
        default="CH",
        help="Market used for historical month-quarter deviations in neighbor monthly anchoring.",
    )
    parser.add_argument(
        "--neighbor-monthly-max-delta-eur-mwh",
        type=float,
        default=20.0,
        help="Maximum absolute hourly/monthly delta from neighbor monthly split anchoring.",
    )
    parser.add_argument(
        "--enable-neighbor-annual-residual-shape-anchor",
        action="store_true",
        help=(
            "Enable local/test anchoring of annual residual buckets (e.g. YYYY-RESIDUAL) "
            "to neighbor market quarter/month shape while preserving CH residual mean."
        ),
    )
    parser.add_argument(
        "--neighbor-annual-residual-anchor-intensity",
        type=float,
        default=1.0,
        help="Multiplier for local/test neighbor annual residual shape anchoring.",
    )
    parser.add_argument(
        "--neighbor-annual-residual-max-delta-eur-mwh",
        type=float,
        default=30.0,
        help="Maximum absolute hourly/monthly delta from neighbor annual residual anchoring.",
    )
    parser.add_argument(
        "--enable-final-monthly-path-smoothing",
        action="store_true",
        help=(
            "Enable a final local/test quote-aware monthly path smoothing pass after neighbor anchoring "
            "to reduce synthetic annual/residual month-to-month zigzags."
        ),
    )
    parser.add_argument(
        "--final-monthly-path-smoothing-intensity",
        type=float,
        default=1.0,
        help="Multiplier for the final local/test monthly path smoothing pass.",
    )
    parser.add_argument(
        "--final-monthly-path-smoothing-lambda",
        type=float,
        default=12.0,
        help="Curvature penalty for the final local/test monthly path smoothing pass.",
    )
    parser.add_argument(
        "--final-monthly-path-max-delta-eur-mwh",
        type=float,
        default=12.0,
        help="Maximum absolute hourly/monthly delta from the final monthly path smoothing pass.",
    )
    parser.add_argument(
        "--enable-final-quant-annual-smoothness",
        action="store_true",
        help=(
            "Enable local/test QuantShapeOptimizer smoothing for complete annual-only synthetic years "
            "while preserving annual BASE/PEAK constraints."
        ),
    )
    parser.add_argument(
        "--final-quant-lambda-smooth-h",
        type=float,
        default=0.0,
        help="Hourly curvature lambda for local/test annual-only quant smoothing.",
    )
    parser.add_argument(
        "--final-quant-lambda-smooth-m",
        type=float,
        default=10000.0,
        help="Monthly mean curvature lambda for local/test annual-only quant smoothing.",
    )
    parser.add_argument(
        "--final-quant-lambda-seam",
        type=float,
        default=10000.0,
        help="Month-boundary curvature lambda for local/test annual-only quant smoothing.",
    )
    parser.add_argument(
        "--enable-cross-year-seasonal-shape-optimizer",
        action="store_true",
        help=(
            "Enable local/test multi-year monthly seasonal reconciliation for synthetic "
            "annual/residual buckets while preserving active BASE/PEAK constraints."
        ),
    )
    parser.add_argument(
        "--cross-year-shape-markets",
        default="DE,AT,FR,IT",
        help="Comma-separated neighbor markets used to build the cross-year monthly shape guide.",
    )
    parser.add_argument(
        "--cross-year-shape-lambda-prior",
        type=float,
        default=1.0,
        help="Distance-to-prior lambda for local/test cross-year seasonal shaping.",
    )
    parser.add_argument(
        "--cross-year-shape-lambda-month-smooth",
        type=float,
        default=1.0,
        help="Monthly curvature lambda for local/test cross-year seasonal shaping.",
    )
    parser.add_argument(
        "--cross-year-shape-lambda-guide",
        type=float,
        default=2.0,
        help="Neighbor/historical guide lambda for local/test cross-year seasonal shaping.",
    )
    parser.add_argument(
        "--cross-year-shape-lambda-cross-month",
        type=float,
        default=8.0,
        help="Same-month parent-spread lambda for local/test cross-year seasonal shaping.",
    )
    parser.add_argument(
        "--cross-year-shape-lambda-cross-slope",
        type=float,
        default=20.0,
        help="Apr-Dec seasonal-slope stability lambda for local/test cross-year seasonal shaping.",
    )
    parser.add_argument(
        "--cross-year-shape-max-month-delta-eur-mwh",
        type=float,
        default=12.0,
        help="Maximum absolute monthly delta from local/test cross-year seasonal shaping.",
    )
    parser.add_argument(
        "--cross-year-shape-intensity",
        type=float,
        default=1.0,
        help="Multiplier for local/test cross-year seasonal shaping.",
    )
    parser.add_argument(
        "--enable-final-seam-nullspace-smoothing",
        action="store_true",
        help=(
            "Enable local/test additive seam smoothing projected into the nullspace of active "
            "BASE/PEAK EEX constraints."
        ),
    )
    parser.add_argument(
        "--final-seam-window-hours",
        type=int,
        default=168,
        help="Half-window in hours used around month-boundary seams.",
    )
    parser.add_argument(
        "--final-seam-target-jump-eur-mwh",
        type=float,
        default=18.0,
        help="Target absolute month-boundary jump after local/test seam smoothing.",
    )
    parser.add_argument(
        "--final-seam-min-jump-eur-mwh",
        type=float,
        default=20.0,
        help="Only month-boundary jumps above this threshold are smoothed.",
    )
    parser.add_argument(
        "--final-seam-max-delta-eur-mwh",
        type=float,
        default=12.0,
        help="Maximum absolute additive correction from final seam nullspace smoothing.",
    )
    parser.add_argument(
        "--final-seam-intensity",
        type=float,
        default=1.0,
        help="Multiplier for the final local/test seam nullspace smoothing correction.",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--enable-monthly-forward-curve-solver",
        action="store_true",
        help="Build the local-test PFC with the governed monthly BASE curve solver as level authority.",
    )
    parser.add_argument("--monthly-solver-constraint-tolerance", type=float, default=0.01)
    parser.add_argument("--monthly-solver-lambda-smooth-month", type=float, default=1.0)
    parser.add_argument("--monthly-solver-lambda-smooth-yoy", type=float, default=0.25)
    parser.add_argument("--monthly-solver-lambda-shape", type=float, default=1.0)
    parser.add_argument("--monthly-solver-neighbor-shrinkage", type=float, default=0.5)
    parser.add_argument(
        "--monthly-solver-allow-template-structural-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--monthly-solver-structural-amplitude", type=float, default=110.0)
    parser.add_argument("--fan-chart-output", default=None)
    parser.add_argument(
        "--skip-powerbi-refresh",
        action="store_true",
        help="Do not rebuild the Power BI CSV sidecars after writing the hourly CSV.",
    )
    parser.add_argument("--powerbi-output-dir", default="powerbi/data")
    parser.add_argument("--powerbi-spot", default="data/epex_hourly.parquet")
    args = parser.parse_args(argv)

    local_start, local_end_15min = _local_window(args.local_start_date, args.local_end_date)
    start_utc, horizon_days = _utc_build_window(local_start, local_end_15min)
    fan_output = Path(args.fan_chart_output or f"output/{args.prefix}_structural_fan_chart.parquet")
    expanded_output = Path(f"output/{args.prefix}_scenario_expanded.parquet")
    features_output = Path(f"data/hpfc_scenario_features_{args.prefix}.parquet")
    summary_output = Path(f".planning/phases/13-lt-electrification-scenario-shape/{args.prefix.upper()}-BUILD.md")
    governance_report = Path(f".planning/phases/13-lt-electrification-scenario-shape/{args.prefix.upper()}-GOVERNANCE.md")

    if args.enable_monthly_forward_curve_solver:
        if args.skip_build:
            raise ValueError("--enable-monthly-forward-curve-solver requires building a fresh fan chart")
        incompatible = {
            "--enable-quote-aware-monthly-smoothing": args.enable_quote_aware_monthly_smoothing,
            "--enable-neighbor-monthly-spread-anchor": args.enable_neighbor_monthly_spread_anchor,
            "--enable-neighbor-annual-residual-shape-anchor": args.enable_neighbor_annual_residual_shape_anchor,
            "--enable-final-monthly-path-smoothing": args.enable_final_monthly_path_smoothing,
            "--enable-final-quant-annual-smoothness": args.enable_final_quant_annual_smoothness,
            "--enable-cross-year-seasonal-shape-optimizer": args.enable_cross_year_seasonal_shape_optimizer,
            "--enable-final-seam-nullspace-smoothing": args.enable_final_seam_nullspace_smoothing,
        }
        enabled = [flag for flag, active in incompatible.items() if active]
        if enabled:
            raise ValueError(
                "--enable-monthly-forward-curve-solver is mutually exclusive with "
                f"mutating local-test post-processors: {', '.join(enabled)}"
            )

    if not args.skip_build:
        build_local_test_ch_pfc(
            [
                "--start-date",
                start_utc,
                "--horizon-days",
                str(horizon_days),
                "--expanded-output",
                str(expanded_output),
                "--features-output",
                str(features_output),
                "--output-prefix",
                args.prefix,
                "--fan-chart-output",
                str(fan_output),
                "--summary",
                str(summary_output),
                "--forwards",
                args.forwards,
                "--governance-report",
                str(governance_report),
                *(
                    ["--enable-monthly-forward-curve-solver"]
                    if args.enable_monthly_forward_curve_solver
                    else []
                ),
                *(
                    [
                        "--monthly-solver-constraint-tolerance",
                        str(args.monthly_solver_constraint_tolerance),
                        "--monthly-solver-lambda-smooth-month",
                        str(args.monthly_solver_lambda_smooth_month),
                        "--monthly-solver-lambda-smooth-yoy",
                        str(args.monthly_solver_lambda_smooth_yoy),
                        "--monthly-solver-lambda-shape",
                        str(args.monthly_solver_lambda_shape),
                        "--monthly-solver-neighbor-shrinkage",
                        str(args.monthly_solver_neighbor_shrinkage),
                        "--monthly-solver-structural-amplitude",
                        str(args.monthly_solver_structural_amplitude),
                        *(
                            [
                                "--monthly-solver-delivery-local-start-date",
                                args.local_start_date,
                                "--monthly-solver-delivery-local-end-date",
                                args.local_end_date,
                            ]
                            if args.enable_monthly_forward_curve_solver
                            else []
                        ),
                        *(
                            ["--monthly-solver-allow-template-structural-fallback"]
                            if args.monthly_solver_allow_template_structural_fallback
                            else ["--no-monthly-solver-allow-template-structural-fallback"]
                        ),
                    ]
                    if args.enable_monthly_forward_curve_solver
                    else []
                ),
                *(
                    ["--disable-cascade-trend-for-annual-only"]
                    if args.disable_cascade_trend_for_annual_only
                    else []
                ),
            ]
        )

    fan = pd.read_parquet(fan_output)
    latest_forward_date, forward_prices_by_load_type = _latest_eex_prices_by_load_type(Path(args.forwards), market="CH")
    forward_prices = forward_prices_by_load_type.get("BASE", {})
    if not forward_prices:
        raise ValueError("no EEX CH BASE prices in latest forward snapshot")
    peak_forward_prices = forward_prices_by_load_type.get("PEAK", {})
    neighbor_prices_by_load_type: dict[str, dict[str, float]] = {}
    historical_neighbor_deviations: dict[str, dict[tuple[int, int], float]] = {}
    cross_year_shape_guide: dict[str, float] = {}
    if args.enable_neighbor_monthly_spread_anchor or args.enable_neighbor_annual_residual_shape_anchor:
        _, neighbor_prices_by_load_type = _latest_eex_prices_by_load_type(
            Path(args.forwards),
            market=str(args.neighbor_monthly_market).upper(),
        )
        min_hist_date = pd.Timestamp(latest_forward_date).tz_localize(None).normalize() - pd.DateOffset(years=6)
        historical_market = str(args.neighbor_monthly_historical_market).upper()
        historical_neighbor_deviations = {
            "BASE": _historical_month_deviations(
                Path(args.forwards),
                market=historical_market,
                load_type="BASE",
                min_date=min_hist_date,
            ),
            "PEAK": _historical_month_deviations(
                Path(args.forwards),
                market=historical_market,
                load_type="PEAK",
                min_date=min_hist_date,
            ),
        }
    if args.enable_cross_year_seasonal_shape_optimizer:
        cross_year_markets = [
            market.strip().upper()
            for market in str(args.cross_year_shape_markets).split(",")
            if market.strip()
        ]
        min_hist_date = pd.Timestamp(latest_forward_date).tz_localize(None).normalize() - pd.DateOffset(years=6)
        cross_year_shape_guide = _frontier_monthly_shape_guide(
            Path(args.forwards),
            markets=cross_year_markets,
            load_type="BASE",
            years=list(range(int(local_start.year), int(local_end_15min.year) + 1)),
            min_date=min_hist_date,
        )
    weights = _parse_weights(args.weights)
    required_forward = None
    if not args.allow_stale_forwards:
        required_forward = require_forward_date(
            latest_forward_date,
            required_forward_date=args.required_forward_date,
            valuation_date=args.valuation_date,
        )
    hourly = to_hourly_csv_frame(
        fan,
        local_start_date=args.local_start_date,
        local_end_date=args.local_end_date,
    )
    shape_upgrade_audit = None
    if args.enable_structural_shape_upgrade:
        hourly, shape_upgrade_audit = apply_local_test_structural_shape_upgrade(
            hourly,
            forward_prices=forward_prices,
            weights=weights,
            intensity=args.structural_shape_upgrade_intensity,
            spread_intensity=args.structural_scenario_spread_intensity,
            enable_negative_price_capture=bool(args.enable_negative_price_capture),
            negative_capture_intensity=args.negative_price_capture_intensity,
            negative_price_floor=args.negative_price_floor,
        )
    hourly, calibration = calibrate_hourly_to_eex(hourly, forward_prices=forward_prices)
    post_negative_audit = None
    post_peak_audit = None
    if args.enable_post_calibration_negative_rebalancer:
        hourly, post_negative_audit = apply_post_calibration_negative_rebalancer(
            hourly,
            forward_prices=forward_prices,
            weights=weights,
            intensity=args.post_calibration_negative_rebalancer_intensity,
            weighted_negative_capture_intensity=args.weighted_negative_capture_intensity,
            negative_price_floor=args.negative_price_floor,
            max_weighted_negative_hours=args.max_weighted_negative_hours,
        )
        hourly, calibration = calibrate_hourly_to_eex(hourly, forward_prices=forward_prices)
    if args.enable_post_calibration_peak_shape_rebalancer:
        hourly, post_peak_audit = apply_post_calibration_peak_shape_rebalancer(
            hourly,
            forward_prices=forward_prices,
            weights=weights,
            intensity=args.post_calibration_peak_shape_intensity,
            negative_price_floor=args.negative_price_floor,
            max_weighted_negative_hours=args.max_weighted_negative_hours,
        )
        hourly, calibration = calibrate_hourly_to_eex(hourly, forward_prices=forward_prices)
    eex_peak_audit = None
    if args.enable_eex_peak_calibration:
        hourly, eex_peak_audit = calibrate_hourly_to_eex_base_peak(
            hourly,
            base_forward_prices=forward_prices,
            peak_forward_prices=peak_forward_prices,
            weights=weights,
            negative_price_floor=args.negative_price_floor,
            max_weighted_negative_hours=args.max_weighted_negative_hours,
        )
    quote_monthly_audit = None
    final_monthly_path_audit = None
    annual_residual_audit = None
    neighbor_monthly_audit = None
    final_quant_annual_audit = None
    cross_year_shape_audit = None
    seam_nullspace_audit = None
    if args.enable_quote_aware_monthly_smoothing:
        ts_ch = _parse_timestamp_ch(hourly["timestamp_ch"], hourly.get("utc_offset_ch"))
        hourly, quote_monthly_audit = apply_quote_aware_monthly_smoothing(
            hourly,
            ts_ch=ts_ch,
            base_forward_prices=forward_prices,
            peak_forward_prices=peak_forward_prices,
            peak_mask=_eex_peak_mask(ts_ch, country="CH"),
            weights=weights,
            intensity=args.quote_aware_monthly_smoothing_intensity,
            smoothness_lambda=args.quote_aware_monthly_smoothing_lambda,
            max_unquoted_month_delta_eur_mwh=args.max_unquoted_month_delta_eur_mwh,
            negative_price_floor=args.negative_price_floor,
            max_weighted_negative_hours=args.max_weighted_negative_hours,
        )
        hourly, calibration = calibrate_hourly_to_eex(hourly, forward_prices=forward_prices)
        if args.enable_eex_peak_calibration:
            hourly, eex_peak_audit = calibrate_hourly_to_eex_base_peak(
                hourly,
                base_forward_prices=forward_prices,
                peak_forward_prices=peak_forward_prices,
                weights=weights,
                negative_price_floor=args.negative_price_floor,
                max_weighted_negative_hours=args.max_weighted_negative_hours,
            )
    if args.enable_neighbor_monthly_spread_anchor:
        ts_ch = _parse_timestamp_ch(hourly["timestamp_ch"], hourly.get("utc_offset_ch"))
        hourly, neighbor_monthly_audit = apply_neighbor_monthly_spread_anchor(
            hourly,
            ts_ch=ts_ch,
            base_forward_prices=forward_prices,
            peak_forward_prices=peak_forward_prices,
            neighbor_base_forward_prices=neighbor_prices_by_load_type.get("BASE", {}),
            neighbor_peak_forward_prices=neighbor_prices_by_load_type.get("PEAK", {}),
            historical_base_deviations=historical_neighbor_deviations.get("BASE", {}),
            historical_peak_deviations=historical_neighbor_deviations.get("PEAK", {}),
            neighbor_current_weight=args.neighbor_monthly_current_weight,
            weights=weights,
            intensity=args.neighbor_monthly_anchor_intensity,
            max_month_delta_eur_mwh=args.neighbor_monthly_max_delta_eur_mwh,
            negative_price_floor=args.negative_price_floor,
            max_weighted_negative_hours=args.max_weighted_negative_hours,
        )
        hourly, calibration = calibrate_hourly_to_eex(hourly, forward_prices=forward_prices)
        if args.enable_eex_peak_calibration:
            hourly, eex_peak_audit = calibrate_hourly_to_eex_base_peak(
                hourly,
                base_forward_prices=forward_prices,
                peak_forward_prices=peak_forward_prices,
                weights=weights,
                negative_price_floor=args.negative_price_floor,
                max_weighted_negative_hours=args.max_weighted_negative_hours,
            )
    if args.enable_neighbor_annual_residual_shape_anchor:
        ts_ch = _parse_timestamp_ch(hourly["timestamp_ch"], hourly.get("utc_offset_ch"))
        hourly, annual_residual_audit = apply_neighbor_annual_residual_shape_anchor(
            hourly,
            ts_ch=ts_ch,
            base_forward_prices=forward_prices,
            neighbor_base_forward_prices=neighbor_prices_by_load_type.get("BASE", {}),
            historical_base_deviations=historical_neighbor_deviations.get("BASE", {}),
            neighbor_current_weight=args.neighbor_monthly_current_weight,
            weights=weights,
            intensity=args.neighbor_annual_residual_anchor_intensity,
            max_month_delta_eur_mwh=args.neighbor_annual_residual_max_delta_eur_mwh,
            negative_price_floor=args.negative_price_floor,
            max_weighted_negative_hours=args.max_weighted_negative_hours,
        )
        hourly, calibration = calibrate_hourly_to_eex(hourly, forward_prices=forward_prices)
        if args.enable_eex_peak_calibration:
            hourly, eex_peak_audit = calibrate_hourly_to_eex_base_peak(
                hourly,
                base_forward_prices=forward_prices,
                peak_forward_prices=peak_forward_prices,
                weights=weights,
                negative_price_floor=args.negative_price_floor,
                max_weighted_negative_hours=args.max_weighted_negative_hours,
            )
    if args.enable_final_monthly_path_smoothing:
        ts_ch = _parse_timestamp_ch(hourly["timestamp_ch"], hourly.get("utc_offset_ch"))
        hourly, final_monthly_path_audit = apply_synthetic_monthly_path_smoothing(
            hourly,
            ts_ch=ts_ch,
            base_forward_prices=forward_prices,
            peak_forward_prices=peak_forward_prices,
            weights=weights,
            intensity=args.final_monthly_path_smoothing_intensity,
            smoothness_lambda=args.final_monthly_path_smoothing_lambda,
            max_month_delta_eur_mwh=args.final_monthly_path_max_delta_eur_mwh,
            negative_price_floor=args.negative_price_floor,
            max_weighted_negative_hours=args.max_weighted_negative_hours,
        )
        hourly, calibration = calibrate_hourly_to_eex(hourly, forward_prices=forward_prices)
        if args.enable_eex_peak_calibration:
            hourly, eex_peak_audit = calibrate_hourly_to_eex_base_peak(
                hourly,
                base_forward_prices=forward_prices,
                peak_forward_prices=peak_forward_prices,
                weights=weights,
                negative_price_floor=args.negative_price_floor,
                max_weighted_negative_hours=args.max_weighted_negative_hours,
            )
        if args.enable_neighbor_annual_residual_shape_anchor:
            ts_ch = _parse_timestamp_ch(hourly["timestamp_ch"], hourly.get("utc_offset_ch"))
            hourly, annual_residual_audit = apply_neighbor_annual_residual_shape_anchor(
                hourly,
                ts_ch=ts_ch,
                base_forward_prices=forward_prices,
                neighbor_base_forward_prices=neighbor_prices_by_load_type.get("BASE", {}),
                historical_base_deviations=historical_neighbor_deviations.get("BASE", {}),
                neighbor_current_weight=args.neighbor_monthly_current_weight,
                weights=weights,
                intensity=args.neighbor_annual_residual_anchor_intensity,
                max_month_delta_eur_mwh=args.neighbor_annual_residual_max_delta_eur_mwh,
                negative_price_floor=args.negative_price_floor,
                max_weighted_negative_hours=args.max_weighted_negative_hours,
            )
            hourly, calibration = calibrate_hourly_to_eex(hourly, forward_prices=forward_prices)
            if args.enable_eex_peak_calibration:
                hourly, eex_peak_audit = calibrate_hourly_to_eex_base_peak(
                    hourly,
                    base_forward_prices=forward_prices,
                    peak_forward_prices=peak_forward_prices,
                    weights=weights,
                    negative_price_floor=args.negative_price_floor,
                    max_weighted_negative_hours=args.max_weighted_negative_hours,
                )
    if args.enable_final_quant_annual_smoothness:
        ts_ch = _parse_timestamp_ch(hourly["timestamp_ch"], hourly.get("utc_offset_ch"))
        hourly, final_quant_annual_audit = apply_final_quant_annual_smoothness_calibration(
            hourly,
            ts_ch=ts_ch,
            base_forward_prices=forward_prices,
            peak_forward_prices=peak_forward_prices,
            weights=weights,
            lambda_smooth_h=args.final_quant_lambda_smooth_h,
            lambda_smooth_m=args.final_quant_lambda_smooth_m,
            lambda_seam=args.final_quant_lambda_seam,
            negative_price_floor=args.negative_price_floor,
            max_weighted_negative_hours=args.max_weighted_negative_hours,
        )
    if args.enable_cross_year_seasonal_shape_optimizer:
        ts_ch = _parse_timestamp_ch(hourly["timestamp_ch"], hourly.get("utc_offset_ch"))
        hourly, cross_year_shape_audit = apply_cross_year_seasonal_shape_optimizer(
            hourly,
            ts_ch=ts_ch,
            base_forward_prices=forward_prices,
            peak_forward_prices=peak_forward_prices,
            peak_mask=_eex_peak_mask(ts_ch, country="CH"),
            guide_month_prices=cross_year_shape_guide,
            weights=weights,
            intensity=args.cross_year_shape_intensity,
            lambda_prior=args.cross_year_shape_lambda_prior,
            lambda_month_smooth=args.cross_year_shape_lambda_month_smooth,
            lambda_guide=args.cross_year_shape_lambda_guide,
            lambda_cross_month=args.cross_year_shape_lambda_cross_month,
            lambda_cross_slope=args.cross_year_shape_lambda_cross_slope,
            max_month_delta_eur_mwh=args.cross_year_shape_max_month_delta_eur_mwh,
            negative_price_floor=args.negative_price_floor,
            max_weighted_negative_hours=args.max_weighted_negative_hours,
        )
        hourly, calibration = calibrate_hourly_to_eex(hourly, forward_prices=forward_prices)
        if args.enable_eex_peak_calibration:
            hourly, eex_peak_audit = calibrate_hourly_to_eex_base_peak(
                hourly,
                base_forward_prices=forward_prices,
                peak_forward_prices=peak_forward_prices,
                weights=weights,
                negative_price_floor=args.negative_price_floor,
                max_weighted_negative_hours=args.max_weighted_negative_hours,
            )
    if args.enable_final_seam_nullspace_smoothing:
        ts_ch = _parse_timestamp_ch(hourly["timestamp_ch"], hourly.get("utc_offset_ch"))
        hourly, seam_nullspace_audit = apply_seam_nullspace_smoothing(
            hourly,
            ts_ch=ts_ch,
            base_forward_prices=forward_prices,
            peak_forward_prices=peak_forward_prices,
            weights=weights,
            seam_window_hours=args.final_seam_window_hours,
            target_boundary_jump_eur_mwh=args.final_seam_target_jump_eur_mwh,
            min_boundary_jump_eur_mwh=args.final_seam_min_jump_eur_mwh,
            max_abs_delta_eur_mwh=args.final_seam_max_delta_eur_mwh,
            intensity=args.final_seam_intensity,
            negative_price_floor=args.negative_price_floor,
            max_weighted_negative_hours=args.max_weighted_negative_hours,
        )
    if args.enable_eex_peak_calibration:
        hourly, eex_peak_audit = calibrate_hourly_to_eex_base_peak(
            hourly,
            base_forward_prices=forward_prices,
            peak_forward_prices=peak_forward_prices,
            weights=weights,
            negative_price_floor=args.negative_price_floor,
            max_weighted_negative_hours=args.max_weighted_negative_hours,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if "structural_p10_eur_mwh" not in hourly.columns and "price_slow_eur_mwh" in hourly.columns:
        scenario = hourly[["price_slow_eur_mwh", "price_central_eur_mwh", "price_fast_eur_mwh"]].astype(float)
        hourly["structural_p10_eur_mwh"] = scenario.min(axis=1)
    if "structural_p50_eur_mwh" not in hourly.columns and "price_central_eur_mwh" in hourly.columns:
        scenario = hourly[["price_slow_eur_mwh", "price_central_eur_mwh", "price_fast_eur_mwh"]].astype(float)
        hourly["structural_p50_eur_mwh"] = scenario.median(axis=1)
    if "structural_p90_eur_mwh" not in hourly.columns and "price_fast_eur_mwh" in hourly.columns:
        scenario = hourly[["price_slow_eur_mwh", "price_central_eur_mwh", "price_fast_eur_mwh"]].astype(float)
        hourly["structural_p90_eur_mwh"] = scenario.max(axis=1)
    if (
        "structural_width_eur_mwh" not in hourly.columns
        and {"structural_p90_eur_mwh", "structural_p10_eur_mwh"}.issubset(hourly.columns)
    ):
        hourly["structural_width_eur_mwh"] = (
            hourly["structural_p90_eur_mwh"] - hourly["structural_p10_eur_mwh"]
        )
    hourly.to_csv(output, index=False)
    _write_report(
        Path(args.report),
        csv_output=output,
        fan_output=fan_output,
        hourly=hourly,
        local_start=args.local_start_date,
        local_end=args.local_end_date,
        forward_date=str(latest_forward_date.date()),
        required_forward_date=str(required_forward.date()) if required_forward is not None else None,
        calibration=calibration,
        shape_upgrade_enabled=bool(args.enable_structural_shape_upgrade),
        shape_upgrade_audit=shape_upgrade_audit,
        negative_price_capture_enabled=bool(args.enable_negative_price_capture),
        post_calibration_negative_rebalancer_enabled=bool(args.enable_post_calibration_negative_rebalancer),
        weighted_negative_capture_intensity=float(args.weighted_negative_capture_intensity),
        post_calibration_negative_rebalancer_audit=post_negative_audit,
        post_calibration_peak_shape_rebalancer_enabled=bool(args.enable_post_calibration_peak_shape_rebalancer),
        post_calibration_peak_shape_rebalancer_audit=post_peak_audit,
        quote_aware_monthly_smoothing_enabled=bool(args.enable_quote_aware_monthly_smoothing),
        quote_aware_monthly_smoothing_audit=quote_monthly_audit,
        final_monthly_path_smoothing_enabled=bool(args.enable_final_monthly_path_smoothing),
        final_monthly_path_smoothing_audit=final_monthly_path_audit,
        annual_residual_anchor_enabled=bool(args.enable_neighbor_annual_residual_shape_anchor),
        annual_residual_anchor_audit=annual_residual_audit,
        final_quant_annual_smoothness_enabled=bool(args.enable_final_quant_annual_smoothness),
        final_quant_annual_smoothness_audit=final_quant_annual_audit,
        cross_year_seasonal_shape_enabled=bool(args.enable_cross_year_seasonal_shape_optimizer),
        cross_year_seasonal_shape_audit=cross_year_shape_audit,
        seam_nullspace_smoothing_enabled=bool(args.enable_final_seam_nullspace_smoothing),
        seam_nullspace_smoothing_audit=seam_nullspace_audit,
        neighbor_monthly_anchor_enabled=bool(args.enable_neighbor_monthly_spread_anchor),
        neighbor_monthly_anchor_audit=neighbor_monthly_audit,
        eex_peak_calibration_enabled=bool(args.enable_eex_peak_calibration),
        eex_peak_calibration_audit=eex_peak_audit,
        disable_cascade_trend_for_annual_only=bool(args.disable_cascade_trend_for_annual_only),
    )
    print(f"[hourly-csv] rows={len(hourly)}")
    print(f"[hourly-csv] output -> {output}")
    print(f"[hourly-csv] report -> {args.report}")
    if not args.skip_powerbi_refresh:
        from scripts.build_powerbi_exports import build_exports

        powerbi_output = Path(args.powerbi_output_dir)
        build_exports(output, Path(args.forwards), Path(args.powerbi_spot), powerbi_output)
        print(f"[powerbi] source csv -> {output}")
        print(f"[powerbi] exports -> {powerbi_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
