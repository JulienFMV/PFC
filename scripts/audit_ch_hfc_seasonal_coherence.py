"""Audit CH HFC/PFC seasonal coherence and duck-curve diagnostics.

The audit is intentionally additive: it does not change generated curves.  It
flags cases where the final hourly curve or the internal base component imply a
questionable monthly shape, especially for annual-only forward years where the
model must synthesize missing monthly products.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.export_local_test_ch_hourly_csv import _parse_timestamp_ch

PRICE = "price_weighted_mean_eur_mwh"
LOCAL_TZ = "Europe/Zurich"


def _load_hourly_csv(path: Path, *, price_column: str = PRICE) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp_ch", price_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df["ts"] = _parse_timestamp_ch(df["timestamp_ch"], df.get("utc_offset_ch"))
    df["year"] = df["ts"].dt.year.astype(int)
    df["month"] = df["ts"].dt.month.astype(int)
    df["quarter"] = df["ts"].dt.quarter.astype(int)
    df["hour"] = df["ts"].dt.hour.astype(int)
    df["weekday"] = df["ts"].dt.weekday.astype(int)
    df[price_column] = pd.to_numeric(df[price_column], errors="raise")
    return df


def _latest_forwards(path: Path, *, market: str) -> tuple[pd.Timestamp, pd.DataFrame]:
    df = pd.read_parquet(path)
    sub = df[(df["market"].astype(str) == market) & (df["load_type"].astype(str) == "BASE")].copy()
    if sub.empty:
        raise ValueError(f"no BASE forwards for market={market!r} in {path}")
    sub["date"] = pd.to_datetime(sub["date"])
    latest = pd.Timestamp(sub["date"].max())
    snap = sub[sub["date"] == latest].copy()
    snap["product"] = snap["product"].astype(str)
    snap["price"] = snap["price"].astype(float)
    return latest, snap


def _product_mask(df: pd.DataFrame, product: str) -> pd.Series:
    if "-Q" in product:
        year = int(product[:4])
        quarter = int(product[-1])
        return (df["year"] == year) & (df["quarter"] == quarter)
    if "-" in product:
        year = int(product[:4])
        month = int(product[5:7])
        return (df["year"] == year) & (df["month"] == month)
    if product.isdigit():
        return df["year"] == int(product)
    return pd.Series(False, index=df.index)


def quoted_product_residuals(
    hourly: pd.DataFrame,
    forwards: pd.DataFrame,
    *,
    price_column: str = PRICE,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in forwards.sort_values("product").iterrows():
        product = str(row["product"])
        mask = _product_mask(hourly, product)
        if not bool(mask.any()):
            continue
        mean = float(hourly.loc[mask, price_column].mean())
        target = float(row["price"])
        rows.append(
            {
                "product": product,
                "target_eex_base_eur_mwh": target,
                "curve_mean_eur_mwh": mean,
                "residual_eur_mwh": mean - target,
                "abs_residual_eur_mwh": abs(mean - target),
                "rows": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def monthly_means(
    hourly: pd.DataFrame,
    *,
    price_column: str = PRICE,
) -> pd.DataFrame:
    return (
        hourly.groupby(["year", "month"], as_index=False)
        .agg(
            mean_eur_mwh=(price_column, "mean"),
            min_eur_mwh=(price_column, "min"),
            max_eur_mwh=(price_column, "max"),
            rows=(price_column, "size"),
        )
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )


def hourly_month_matrix(
    hourly: pd.DataFrame,
    *,
    price_column: str = PRICE,
) -> pd.DataFrame:
    return (
        hourly.groupby(["year", "month", "hour"], as_index=False)
        .agg(mean_eur_mwh=(price_column, "mean"))
        .sort_values(["year", "month", "hour"])
        .reset_index(drop=True)
    )


def _coverage_for_year(forwards: pd.DataFrame, year: int) -> str:
    products = set(forwards["product"].astype(str))
    months = {f"{year}-{month:02d}" for month in range(1, 13)}
    quarters = {f"{year}-Q{quarter}" for quarter in range(1, 5)}
    if months & products:
        return "monthly_or_mixed"
    if quarters.issubset(products):
        return "full_quarter"
    if quarters & products:
        return "partial_quarter"
    if str(year) in products:
        return "annual_only"
    return "unquoted"


def _month_value(months: pd.DataFrame, year: int, month: int) -> float | None:
    sub = months[(months["year"] == year) & (months["month"] == month)]
    if sub.empty:
        return None
    return float(sub["mean_eur_mwh"].iloc[0])


def seasonal_coherence_checks(
    monthly: pd.DataFrame,
    forwards: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in sorted(monthly["year"].unique()):
        jan = _month_value(monthly, int(year), 1)
        feb = _month_value(monthly, int(year), 2)
        sep = _month_value(monthly, int(year), 9)
        octo = _month_value(monthly, int(year), 10)
        nov = _month_value(monthly, int(year), 11)
        dec = _month_value(monthly, int(year), 12)
        if jan is None or octo is None:
            continue

        q1_values = [value for value in [jan, feb, _month_value(monthly, int(year), 3)] if value is not None]
        q4_values = [value for value in [octo, nov, dec] if value is not None]
        autumn_values = [value for value in [sep, octo, nov] if value is not None]
        winter_values = [value for value in [jan, feb, dec] if value is not None]
        jan_minus_oct = jan - octo
        q1_minus_q4 = float(np.mean(q1_values) - np.mean(q4_values)) if q1_values and q4_values else np.nan
        winter_minus_autumn = (
            float(np.mean(winter_values) - np.mean(autumn_values)) if winter_values and autumn_values else np.nan
        )
        coverage = _coverage_for_year(forwards, int(year))

        severity = "ok"
        reason = "seasonal ordering plausible"
        if coverage == "annual_only" and jan_minus_oct < -5.0:
            severity = "critical"
            reason = "annual-only synthetic shape has January materially below October"
        elif coverage == "annual_only" and jan_minus_oct < 0.0:
            severity = "warning"
            reason = "annual-only synthetic shape has January below October"
        elif coverage == "partial_quarter" and q1_minus_q4 < -5.0:
            severity = "warning"
            reason = "partial-quarter year has Q1 materially below Q4 after completion"

        rows.append(
            {
                "year": int(year),
                "forward_coverage": coverage,
                "jan_mean_eur_mwh": jan,
                "oct_mean_eur_mwh": octo,
                "jan_minus_oct_eur_mwh": jan_minus_oct,
                "q1_minus_q4_eur_mwh": q1_minus_q4,
                "winter_minus_autumn_eur_mwh": winter_minus_autumn,
                "severity": severity,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def component_monthly_means(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("component parquet index must be a DatetimeIndex")
    if frame.index.tz is None:
        frame.index = pd.to_datetime(frame.index, utc=True)
    else:
        frame.index = frame.index.tz_convert("UTC")
    local = frame.tz_convert(LOCAL_TZ)
    numeric_cols = [col for col in ["price_shape", "B", "f_S", "f_W", "f_H", "f_Q", "f_WV", "f_bridge"] if col in local]
    hourly = local[numeric_cols].resample("h").mean()
    hourly["year"] = hourly.index.year.astype(int)
    hourly["month"] = hourly.index.month.astype(int)
    out = hourly.groupby(["year", "month"], as_index=False)[numeric_cols].mean()
    return out.sort_values(["year", "month"]).reset_index(drop=True)


def component_jan_oct_checks(component_monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    numeric_cols = [c for c in component_monthly.columns if c not in {"year", "month"}]
    for year in sorted(component_monthly["year"].unique()):
        jan = component_monthly[(component_monthly["year"] == year) & (component_monthly["month"] == 1)]
        octo = component_monthly[(component_monthly["year"] == year) & (component_monthly["month"] == 10)]
        if jan.empty or octo.empty:
            continue
        row: dict[str, object] = {"year": int(year)}
        for col in numeric_cols:
            row[f"{col}_jan_minus_oct"] = float(jan[col].iloc[0] - octo[col].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def _md_table(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_empty_"
    view = frame.head(max_rows).copy() if max_rows is not None else frame.copy()
    lines = [
        "| " + " | ".join(str(c) for c in view.columns) + " |",
        "|" + "|".join("---" for _ in view.columns) + "|",
    ]
    for _, row in view.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    if max_rows is not None and len(frame) > max_rows:
        lines.append(f"| ... | {' | '.join([''] * (len(view.columns) - 1))} |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    csv_path: Path,
    forwards_path: Path,
    latest_forward_date: pd.Timestamp,
    quoted_residuals: pd.DataFrame,
    seasonal_checks: pd.DataFrame,
    monthly: pd.DataFrame,
    hour_month: pd.DataFrame,
    component_checks: pd.DataFrame | None,
) -> None:
    critical = int((seasonal_checks["severity"] == "critical").sum()) if not seasonal_checks.empty else 0
    warnings = int((seasonal_checks["severity"] == "warning").sum()) if not seasonal_checks.empty else 0
    max_quoted_residual = (
        float(quoted_residuals["abs_residual_eur_mwh"].max()) if not quoted_residuals.empty else float("nan")
    )
    jan_oct_rows = seasonal_checks[seasonal_checks["jan_minus_oct_eur_mwh"] < 0.0]
    lines = [
        "# CH HFC Seasonal Coherence Audit",
        "",
        f"* CSV: `{csv_path}`",
        f"* forwards: `{forwards_path}`",
        f"* latest EEX CH BASE date: `{latest_forward_date.date()}`",
        "* scope: `local/test diagnostics`",
        "* production approval: `NO`",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| critical seasonal flags | {critical} |",
        f"| warning seasonal flags | {warnings} |",
        f"| max quoted-product residual EUR/MWh | {max_quoted_residual:.6f} |",
        f"| years with January below October | {len(jan_oct_rows)} |",
        "",
        "## Seasonal Checks",
        "",
        _md_table(seasonal_checks),
        "",
        "## Quoted EEX Product Residuals",
        "",
        _md_table(quoted_residuals),
        "",
        "## Monthly Means",
        "",
        _md_table(monthly),
        "",
        "## Hour x Month Duck Diagnostics",
        "",
        "_Full table is written to the CSV sidecar; first rows shown below._",
        "",
        _md_table(hour_month, max_rows=48),
        "",
    ]
    if component_checks is not None:
        lines.extend(
            [
                "## Component Jan-Oct Checks",
                "",
                _md_table(component_checks),
                "",
                "Interpretation: if `B_jan_minus_oct` is already negative, the inversion is in the "
                "forward cascade/base reconstruction, not in the intraday duck curve.",
                "",
            ]
        )
    lines.extend(
        [
            "## Gate Interpretation",
            "",
            "* `critical` means annual-only synthetic monthly completion contradicts the expected Swiss winter premium.",
            "* This is a model-quality gate, not a market override: if quoted monthly/quarterly EEX products imply the inversion, the gate should not force a correction.",
            "* The current local/test curve should be considered seasonally suspect if any critical flag remains.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def audit(
    csv_path: Path,
    forwards_path: Path,
    *,
    component_parquet: Path | None = None,
    market: str = "CH",
    price_column: str = PRICE,
) -> dict[str, pd.DataFrame | pd.Timestamp]:
    hourly = _load_hourly_csv(csv_path, price_column=price_column)
    latest, forwards = _latest_forwards(forwards_path, market=market)
    monthly = monthly_means(hourly, price_column=price_column)
    hour_month = hourly_month_matrix(hourly, price_column=price_column)
    residuals = quoted_product_residuals(hourly, forwards, price_column=price_column)
    seasonal = seasonal_coherence_checks(monthly, forwards)
    component_checks = None
    if component_parquet is not None:
        component_checks = component_jan_oct_checks(component_monthly_means(component_parquet))
    return {
        "latest_forward_date": latest,
        "quoted_residuals": residuals,
        "seasonal_checks": seasonal,
        "monthly": monthly,
        "hour_month": hour_month,
        "component_checks": component_checks if component_checks is not None else pd.DataFrame(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--forwards", default="data/eex_forwards_history.parquet")
    parser.add_argument("--component-parquet", default=None)
    parser.add_argument("--market", default="CH")
    parser.add_argument("--price-column", default=PRICE)
    parser.add_argument("--report", required=True)
    parser.add_argument("--monthly-output", default=None)
    parser.add_argument("--hour-month-output", default=None)
    parser.add_argument("--fail-on-critical", action="store_true")
    args = parser.parse_args(argv)

    result = audit(
        Path(args.csv),
        Path(args.forwards),
        component_parquet=Path(args.component_parquet) if args.component_parquet else None,
        market=args.market,
        price_column=args.price_column,
    )
    monthly = result["monthly"]
    hour_month = result["hour_month"]
    if args.monthly_output:
        Path(args.monthly_output).parent.mkdir(parents=True, exist_ok=True)
        monthly.to_csv(args.monthly_output, index=False)
    if args.hour_month_output:
        Path(args.hour_month_output).parent.mkdir(parents=True, exist_ok=True)
        hour_month.to_csv(args.hour_month_output, index=False)
    write_report(
        Path(args.report),
        csv_path=Path(args.csv),
        forwards_path=Path(args.forwards),
        latest_forward_date=result["latest_forward_date"],  # type: ignore[arg-type]
        quoted_residuals=result["quoted_residuals"],  # type: ignore[arg-type]
        seasonal_checks=result["seasonal_checks"],  # type: ignore[arg-type]
        monthly=monthly,  # type: ignore[arg-type]
        hour_month=hour_month,  # type: ignore[arg-type]
        component_checks=result["component_checks"],  # type: ignore[arg-type]
    )
    seasonal = result["seasonal_checks"]  # type: ignore[assignment]
    critical = int((seasonal["severity"] == "critical").sum()) if not seasonal.empty else 0
    warning = int((seasonal["severity"] == "warning").sum()) if not seasonal.empty else 0
    print(f"[seasonal-audit] critical={critical} warning={warning}")
    print(f"[seasonal-audit] report -> {args.report}")
    if args.fail_on_critical and critical:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
