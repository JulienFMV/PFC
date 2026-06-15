"""Export a local-test CH PFC as an hourly CSV over a local delivery window."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfc_shaping.calibration.eex_contract_selection import (  # noqa: E402
    calibration_buckets,
    finest_product_for_timestamp,
    quarter_product,
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


def _latest_eex_base_prices(path: Path, *, market: str = "CH") -> tuple[pd.Timestamp, dict[str, float]]:
    frame = pd.read_parquet(path)
    sub = frame[(frame["market"].astype(str) == market) & (frame["load_type"].astype(str) == "BASE")].copy()
    if sub.empty:
        raise ValueError(f"no EEX BASE prices for market={market}")
    latest = pd.Timestamp(sub["date"].max())
    snap = sub[sub["date"] == latest]
    return latest, dict(zip(snap["product"].astype(str), snap["price"].astype(float)))


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
    weekend = ts.weekday() >= 5
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
        if weekend and 10 <= hour <= 16:
            delta -= 2.0
    elif season == "summer":
        if 10 <= hour <= 16:
            delta -= 7.5
        elif 17 <= hour <= 22:
            delta += 5.5
        elif 7 <= hour <= 9:
            delta += 1.5
        if weekend and 10 <= hour <= 16:
            delta -= 2.5
    else:
        if 10 <= hour <= 15:
            delta -= 3.5
        elif 17 <= hour <= 21:
            delta += 5.0
        elif 7 <= hour <= 9:
            delta += 2.5

    if weekend and 18 <= hour <= 21:
        delta -= 1.0

    return float(np.clip(delta * float(intensity), -12.0, 12.0))


def _scenario_spread_delta(ts: pd.Timestamp, scenario: str, *, intensity: float) -> float:
    """Slow/fast structural dispersion in EUR/MWh before product de-meaning."""
    score = {"slow": -1.0, "central": 0.0, "fast": 1.0}[scenario]
    if score == 0.0 or intensity == 0.0:
        return 0.0

    hour = int(ts.hour)
    season = _season_name(int(ts.month))
    weekend = ts.weekday() >= 5
    delta = 0.0

    if 10 <= hour <= 15:
        delta += {"winter": -1.5, "spring": -6.0, "summer": -7.0, "autumn": -4.5}[season] * score
        if weekend and season in {"spring", "summer"}:
            delta += -1.5 * score
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
    day_factor = 1.00 if ts.weekday() >= 5 else 0.35
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
    disable_cascade_trend_for_annual_only: bool = False,
) -> None:
    price = hourly["price_weighted_mean_eur_mwh"]
    width = hourly["structural_width_eur_mwh"]
    lines = [
        "# Local-Test CH PFC Hourly CSV",
        "",
        "* status: `agent-approved local/test only`",
        "* production approval: `NO`",
        f"* CSV: `{csv_output}`",
        f"* source fan chart: `{fan_output}`",
        f"* local window: `{local_start}` to `{local_end}` Europe/Zurich",
        f"* EEX CH forward date: `{forward_date}`",
        f"* required EEX CH forward date: `{required_forward_date or 'not enforced'}`",
        f"* local/test structural shape upgrade: `{'ON' if shape_upgrade_enabled else 'OFF'}`",
        f"* local/test negative price capture: `{'ON' if negative_price_capture_enabled else 'OFF'}`",
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
        "## Limitations",
        "",
        "* Hourly values are arithmetic averages of the calibrated 15-minute local/test PFC.",
        "* The curve is calibrated to the required EEX CH forward snapshot for the run date.",
        "* The structural shape upgrade is local/test only and remains disabled unless explicitly requested.",
        "* Negative-price capture, when enabled, is local/test only and bounded by an explicit floor.",
        "* This is not production FMV output; production governance remains NO-GO.",
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
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--fan-chart-output", default=None)
    args = parser.parse_args(argv)

    local_start, local_end_15min = _local_window(args.local_start_date, args.local_end_date)
    start_utc, horizon_days = _utc_build_window(local_start, local_end_15min)
    fan_output = Path(args.fan_chart_output or f"output/{args.prefix}_structural_fan_chart.parquet")
    expanded_output = Path(f"output/{args.prefix}_scenario_expanded.parquet")
    features_output = Path(f"data/hpfc_scenario_features_{args.prefix}.parquet")
    summary_output = Path(f".planning/phases/13-lt-electrification-scenario-shape/{args.prefix.upper()}-BUILD.md")
    governance_report = Path(f".planning/phases/13-lt-electrification-scenario-shape/{args.prefix.upper()}-GOVERNANCE.md")

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
                    ["--disable-cascade-trend-for-annual-only"]
                    if args.disable_cascade_trend_for_annual_only
                    else []
                ),
            ]
        )

    fan = pd.read_parquet(fan_output)
    latest_forward_date, forward_prices = _latest_eex_base_prices(Path(args.forwards), market="CH")
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
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
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
        disable_cascade_trend_for_annual_only=bool(args.disable_cascade_trend_for_annual_only),
    )
    print(f"[hourly-csv] rows={len(hourly)}")
    print(f"[hourly-csv] output -> {output}")
    print(f"[hourly-csv] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
