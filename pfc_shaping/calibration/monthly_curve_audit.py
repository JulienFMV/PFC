"""Numerical audit gates for monthly forward curve candidates.

These gates are evidence generation only.  They never mutate the solved curve.
Shape gates are fail-closed: when calibrated historical thresholds are missing
or undersampled, the row is ``UNSUPPORTED`` rather than ``PASS``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from pfc_shaping.calibration.monthly_forward_curve import MonthlyConstraintSystem


def audit_monthly_curve_shape(
    monthly_curve: pd.Series,
    constraints: MonthlyConstraintSystem,
    *,
    year_pairs: Sequence[tuple[int, int]] | None = None,
    historical_thresholds: pd.DataFrame | None = None,
    neighbor_level_leakage_max_abs: float | None = None,
    repricing_tolerance: float = 1e-8,
    leakage_tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Return machine-readable monthly shape audit rows.

    ``historical_thresholds`` follows the Phase F threshold schema.  Shape
    gates look up P90/P97.5 by ``gate_id`` and metric name, preferring a
    month-specific ``delivery_bucket`` such as ``month_12`` and then ``all``.
    """

    curve = monthly_curve.reindex(constraints.delivery_grid.months).astype(float)
    if curve.isna().any():
        missing = [str(month) for month in curve[curve.isna()].index]
        raise ValueError(f"monthly curve is missing delivery months: {missing}")
    pairs = list(year_pairs or _adjacent_year_pairs(constraints.delivery_grid.months))
    rows: list[dict[str, object]] = []
    rows.extend(_active_repricing_rows(curve, constraints, repricing_tolerance))
    if neighbor_level_leakage_max_abs is not None:
        rows.append(
            _neighbor_level_leakage_row(
                leakage_max_abs=float(neighbor_level_leakage_max_abs),
                tolerance=float(leakage_tolerance),
            )
        )
    for year_a, year_b in pairs:
        rows.extend(
            _same_month_rows(
                curve,
                constraints,
                year_a=year_a,
                year_b=year_b,
                historical_thresholds=historical_thresholds,
            )
        )
        rows.extend(
            _residual_comparable_block_rows(
                curve,
                constraints,
                year_a=year_a,
                year_b=year_b,
                historical_thresholds=historical_thresholds,
            )
        )
    rows.append(_monthly_shape_regression_row(rows))
    return pd.DataFrame(rows)


def _active_repricing_rows(
    curve: pd.Series,
    constraints: MonthlyConstraintSystem,
    tolerance: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    residuals = constraints.residuals(curve.to_numpy(dtype=float))
    for row in residuals.itertuples(index=False):
        product = str(row.name).split(":", 1)[-1]
        error = float(row.achieved) - float(row.target)
        status = "PASS" if abs(error) <= tolerance else "CRITICAL"
        first_month = _first_month_for_bucket(constraints, product)
        rows.append(
            _gate_row(
                gate_id="hard_monthly_curve_repricing",
                status=status,
                severity="P0" if status == "CRITICAL" else "INFO",
                year=int(first_month.year) if first_month is not None else 0,
                month=None if first_month is None else int(first_month.month),
                product=product,
                parent_block_id=product,
                parent_block_type=_active_product_type(product),
                parent_hours=_parent_hours_for_bucket(constraints, product),
                parent_mean=float(row.target),
                month_price=np.nan,
                month_deviation=np.nan,
                metric_name="active_constraint_abs_error_eur_mwh",
                metric_value=abs(error),
                threshold_warning=tolerance,
                threshold_critical=tolerance,
                threshold_source="hard_constraint",
                n_history=np.nan,
                n_neighbors=np.nan,
                evidence=f"achieved={float(row.achieved):.12g}, target={float(row.target):.12g}",
                remediation_hint="Fix monthly solver constraints before inspecting shape.",
            )
        )
    return rows


def _neighbor_level_leakage_row(
    *,
    leakage_max_abs: float,
    tolerance: float,
) -> dict[str, object]:
    status = "PASS" if abs(float(leakage_max_abs)) <= float(tolerance) else "CRITICAL"
    return _gate_row(
        gate_id="neighbor_level_leakage",
        status=status,
        severity="P0" if status == "CRITICAL" else "INFO",
        year=0,
        month=None,
        product="neighbor_shift_invariance",
        parent_block_id="all",
        parent_block_type="all",
        parent_hours=np.nan,
        parent_mean=np.nan,
        month_price=np.nan,
        month_deviation=np.nan,
        metric_name="neighbor_shift_solution_delta_max_abs_eur_mwh",
        metric_value=float(abs(leakage_max_abs)),
        threshold_warning=float(tolerance),
        threshold_critical=float(tolerance),
        threshold_source="hard_invariance",
        n_history=np.nan,
        n_neighbors=np.nan,
        evidence=f"max_abs_solution_delta={float(leakage_max_abs):.12g}",
        remediation_hint="External market levels leaked into CH monthly solution; recenter neighbor priors.",
    )


def _same_month_rows(
    curve: pd.Series,
    constraints: MonthlyConstraintSystem,
    *,
    year_a: int,
    year_b: int,
    historical_thresholds: pd.DataFrame | None,
) -> list[dict[str, object]]:
    cal_a = _quote_target(constraints, str(year_a))
    cal_b = _quote_target(constraints, str(year_b))
    if cal_a is None or cal_b is None:
        return []
    calendar_spread = float(cal_a) - float(cal_b)
    rows: list[dict[str, object]] = []
    for month_number in range(1, 13):
        month_a = pd.Period(f"{year_a}-{month_number:02d}", freq="M")
        month_b = pd.Period(f"{year_b}-{month_number:02d}", freq="M")
        if month_a not in curve.index or month_b not in curve.index:
            continue
        price_a = float(curve.loc[month_a])
        price_b = float(curve.loc[month_b])
        parent_a = _parent_info(constraints, month_a)
        parent_b = _parent_info(constraints, month_b)
        if parent_a is None or parent_b is None:
            continue
        parent_spread = parent_a["target"] - parent_b["target"]
        month_spread = price_a - price_b
        shape_delta = (price_a - parent_a["target"]) - (price_b - parent_b["target"])
        direct_support = parent_a["type"] == "month" or parent_b["type"] == "month"
        metric = abs(float(shape_delta))
        threshold = _threshold_lookup(
            historical_thresholds,
            gate_id="same_month_rank_consistency",
            metric_name="same_month_shape_delta_abs_eur_mwh",
            month=month_number,
        )
        status, severity, threshold_source, warning, critical, n_history, reason = _status_from_threshold(
            metric_value=metric,
            threshold=threshold,
            direct_quote_support=direct_support,
            quote_support_reason="direct monthly quote support",
        )
        sign_reason = _sign_reason(
            calendar_spread=calendar_spread,
            parent_spread=parent_spread,
            month_spread=month_spread,
        )
        rows.append(
            _gate_row(
                gate_id="same_month_rank_consistency",
                status=status,
                severity=severity,
                year=year_a,
                month=month_number,
                product=f"{year_a}-{month_number:02d}_vs_{year_b}-{month_number:02d}",
                parent_block_id=f"{parent_a['bucket']}|{parent_b['bucket']}",
                parent_block_type=f"{parent_a['type']}|{parent_b['type']}",
                parent_hours=parent_a["hours"],
                parent_mean=parent_a["target"],
                month_price=price_a,
                month_deviation=price_a - parent_a["target"],
                metric_name="same_month_shape_delta_abs_eur_mwh",
                metric_value=metric,
                threshold_warning=warning,
                threshold_critical=critical,
                threshold_source=threshold_source,
                n_history=n_history,
                n_neighbors=np.nan,
                evidence=(
                    f"calendar_spread={calendar_spread:.4f}, "
                    f"parent_spread={parent_spread:.4f}, month_spread={month_spread:.4f}, "
                    f"shape_delta={shape_delta:.4f}, {sign_reason}, {reason}"
                ),
                remediation_hint="Review comparable-block shape prior or active quote support.",
                year_b=year_b,
                price_b=price_b,
                parent_mean_b=parent_b["target"],
                parent_hours_b=parent_b["hours"],
                parent_mix_adjustment=parent_spread - calendar_spread,
                expected_sign=float(np.sign(parent_spread)),
                actual_sign=float(np.sign(month_spread)),
                z_score_or_quantile=np.nan,
                quote_support_type="DIRECT_MONTH" if direct_support else "NONE",
                supporting_quote_keys="|".join(parent_a["source_quote_keys"] + parent_b["source_quote_keys"])
                if direct_support
                else "",
                quote_support_value=month_spread if direct_support else np.nan,
                quote_support_rule="active monthly hard constraint" if direct_support else "none",
            )
        )
    return rows


def _residual_comparable_block_rows(
    curve: pd.Series,
    constraints: MonthlyConstraintSystem,
    *,
    year_a: int,
    year_b: int,
    historical_thresholds: pd.DataFrame | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for month_number in range(1, 12 + 1):
        month_a = pd.Period(f"{year_a}-{month_number:02d}", freq="M")
        month_b = pd.Period(f"{year_b}-{month_number:02d}", freq="M")
        if month_a not in curve.index or month_b not in curve.index:
            continue
        parent_a = _parent_info(constraints, month_a)
        parent_b = _parent_info(constraints, month_b)
        if parent_a is None or parent_b is None:
            continue
        types = {str(parent_a["type"]), str(parent_b["type"])}
        if "residual" not in types or "calendar" not in types:
            continue
        price_a = float(curve.loc[month_a])
        price_b = float(curve.loc[month_b])
        dev_a = price_a - float(parent_a["target"])
        dev_b = price_b - float(parent_b["target"])
        metric = abs(float(dev_a - dev_b))
        threshold = _threshold_lookup(
            historical_thresholds,
            gate_id="residual_vs_implied_comparable_block",
            metric_name="comparable_block_shape_delta_abs_eur_mwh",
            month=month_number,
        )
        status, severity, threshold_source, warning, critical, n_history, reason = _status_from_threshold(
            metric_value=metric,
            threshold=threshold,
            direct_quote_support=False,
            quote_support_reason="",
        )
        parent_spread = float(parent_a["target"]) - float(parent_b["target"])
        rows.append(
            _gate_row(
                gate_id="residual_vs_implied_comparable_block",
                status=status,
                severity=severity,
                year=year_a,
                month=month_number,
                product=f"{year_a}-{month_number:02d}_vs_{year_b}-{month_number:02d}",
                parent_block_id=f"{parent_a['bucket']}|{parent_b['bucket']}",
                parent_block_type=f"{parent_a['type']}|{parent_b['type']}",
                parent_hours=parent_a["hours"],
                parent_mean=float(parent_a["target"]),
                month_price=price_a,
                month_deviation=dev_a,
                metric_name="comparable_block_shape_delta_abs_eur_mwh",
                metric_value=metric,
                threshold_warning=warning,
                threshold_critical=critical,
                threshold_source=threshold_source,
                n_history=n_history,
                n_neighbors=np.nan,
                evidence=(
                    f"parent_block_a={parent_a['bucket']}, parent_block_b={parent_b['bucket']}, "
                    f"parent_spread={parent_spread:.4f}, dev_a={dev_a:.4f}, dev_b={dev_b:.4f}, {reason}"
                ),
                remediation_hint=(
                    "Compare same-month deviations from comparable parent blocks; do not compare residual "
                    "Apr-Dec level directly with full calendar level."
                ),
                year_b=year_b,
                price_b=price_b,
                parent_mean_b=float(parent_b["target"]),
                parent_hours_b=parent_b["hours"],
                parent_mix_adjustment=parent_spread - _calendar_spread(constraints, year_a, year_b),
            )
        )
    return rows


def _monthly_shape_regression_row(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    targeted = [
        row
        for row in rows
        if row.get("gate_id") in {"same_month_rank_consistency", "residual_vs_implied_comparable_block"}
        and 2028 <= _safe_int(row.get("year"), default=0) <= 2030
    ]
    statuses = [str(row.get("status", "")) for row in targeted]
    if any(status == "CRITICAL" for status in statuses):
        status = "CRITICAL"
        severity = "P1"
        reason = "at least one targeted 2028-2030 monthly shape gate is CRITICAL"
    elif not targeted or any(status == "UNSUPPORTED" for status in statuses):
        status = "UNSUPPORTED"
        severity = "P2"
        reason = "targeted 2028-2030 monthly shape evidence is missing or undersampled"
    elif any(status == "WARNING" for status in statuses):
        status = "WARNING"
        severity = "P2"
        reason = "at least one targeted 2028-2030 monthly shape gate is WARNING"
    else:
        status = "PASS"
        severity = "INFO"
        reason = "targeted 2028-2030 monthly shape gates passed"
    return _gate_row(
        gate_id="monthly_shape_regression_2028_2030",
        status=status,
        severity=severity,
        year=2028,
        month=None,
        product="2028_2030_focus_population",
        parent_block_id="2028_2030",
        parent_block_type="focus_population",
        parent_hours=np.nan,
        parent_mean=np.nan,
        month_price=np.nan,
        month_deviation=np.nan,
        metric_name="targeted_gate_max_status",
        metric_value=float(_status_rank(status)),
        threshold_warning=float(_status_rank("WARNING")),
        threshold_critical=float(_status_rank("CRITICAL")),
        threshold_source="aggregate_targeted_gates",
        n_history=np.nan,
        n_neighbors=np.nan,
        evidence=f"{reason}; counts={dict(pd.Series(statuses, dtype=object).value_counts()) if statuses else {}}",
        remediation_hint="Investigate targeted same-month and comparable-block gate rows.",
    )


def _parent_info(constraints: MonthlyConstraintSystem, month: pd.Period) -> dict[str, object] | None:
    bucket = constraints.month_buckets.loc[month]
    if pd.isna(bucket):
        return None
    bucket_text = str(bucket)
    row = constraints.rows[constraints.rows["product"].astype(str).eq(bucket_text)]
    parent_product = bucket_text if row.empty else str(row["parent_product"].iloc[0])
    source_quote_keys: tuple[str, ...] = tuple()
    hours = np.nan
    if not row.empty:
        hours = float(row["hours"].iloc[0]) if "hours" in row else np.nan
        raw_sources = str(row["source_quote_keys"].iloc[0]) if "source_quote_keys" in row else ""
        source_quote_keys = tuple(part for part in raw_sources.split("|") if part)
    return {
        "bucket": bucket_text,
        "product": parent_product,
        "type": _parent_type(bucket_text, parent_product),
        "target": float(constraints.bucket_targets[bucket_text]),
        "hours": hours,
        "source_quote_keys": source_quote_keys,
    }


def _parent_hours_for_bucket(constraints: MonthlyConstraintSystem, bucket: str) -> float:
    row = constraints.rows[constraints.rows["product"].astype(str).eq(str(bucket))]
    if row.empty or "hours" not in row:
        return np.nan
    return float(row["hours"].iloc[0])


def _parent_type(bucket: str, parent_product: str) -> str:
    if bucket.endswith("-RESIDUAL"):
        return "residual"
    if "-Q" in parent_product:
        return "quarter"
    if parent_product.isdigit():
        return "calendar"
    if len(parent_product) == 7 and parent_product[4] == "-":
        return "month"
    return "unknown"


def _first_month_for_bucket(constraints: MonthlyConstraintSystem, bucket: str) -> pd.Period | None:
    mask = constraints.month_buckets.astype(str).eq(str(bucket))
    if not bool(mask.any()):
        return None
    return constraints.month_buckets.index[mask][0]


def _active_product_type(product: str) -> str:
    if product.endswith("-RESIDUAL"):
        return "residual"
    if "-Q" in product:
        return "quarter"
    if product.isdigit():
        return "calendar"
    if len(product) == 7 and product[4] == "-":
        return "month"
    return "unknown"


def _quote_target(constraints: MonthlyConstraintSystem, product: str) -> float | None:
    diag = constraints.quote_diagnostics
    if diag.empty:
        return None
    row = diag[diag["product"].astype(str).eq(str(product))]
    if row.empty:
        return None
    return float(row["target"].iloc[0])


def _calendar_spread(constraints: MonthlyConstraintSystem, year_a: int, year_b: int) -> float:
    cal_a = _quote_target(constraints, str(year_a))
    cal_b = _quote_target(constraints, str(year_b))
    if cal_a is None or cal_b is None:
        return np.nan
    return float(cal_a) - float(cal_b)


def _adjacent_year_pairs(months: pd.PeriodIndex) -> list[tuple[int, int]]:
    years = sorted({int(month.year) for month in months})
    return list(zip(years, years[1:]))


def _gate_row(
    *,
    gate_id: str,
    status: str,
    severity: str,
    year: int,
    month: int | None,
    product: str,
    parent_block_id: str,
    parent_block_type: str,
    parent_hours: float,
    parent_mean: float,
    month_price: float,
    month_deviation: float,
    metric_name: str,
    metric_value: float,
    threshold_warning: float,
    threshold_critical: float,
    threshold_source: str,
    n_history: float,
    n_neighbors: float,
    evidence: str,
    remediation_hint: str,
    **extra: object,
) -> dict[str, object]:
    row = {
        "gate_id": gate_id,
        "status": status,
        "severity": severity,
        "market": "CH",
        "load_type": "BASE",
        "year": year,
        "month": month,
        "product": product,
        "parent_block_id": parent_block_id,
        "parent_block_type": parent_block_type,
        "parent_hours": parent_hours,
        "parent_mean": parent_mean,
        "month_price": month_price,
        "month_deviation_from_parent": month_deviation,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "threshold_warning": threshold_warning,
        "threshold_critical": threshold_critical,
        "threshold_source": threshold_source,
        "n_history": n_history,
        "n_neighbors": n_neighbors,
        "evidence": evidence,
        "remediation_hint": remediation_hint,
    }
    row.update(extra)
    return row


def _threshold_lookup(
    thresholds: pd.DataFrame | None,
    *,
    gate_id: str,
    metric_name: str,
    month: int,
) -> dict[str, object] | None:
    if thresholds is None or thresholds.empty:
        return None
    required = {"gate_id", "metric", "market", "p90", "p975", "n_snapshots", "min_required_n", "status"}
    if not required <= set(thresholds.columns):
        return None
    frame = thresholds[
        thresholds["gate_id"].astype(str).eq(gate_id)
        & thresholds["metric"].astype(str).eq(metric_name)
        & thresholds["market"].astype(str).str.upper().eq("CH")
    ].copy()
    if frame.empty:
        return None
    delivery_bucket = frame.get("delivery_bucket", pd.Series("", index=frame.index)).astype(str)
    specific = frame[delivery_bucket.eq(f"month_{int(month):02d}")]
    if not specific.empty:
        return specific.iloc[0].to_dict()
    fallback = frame[delivery_bucket.eq("all")]
    if not fallback.empty:
        return fallback.iloc[0].to_dict()
    return None


def _status_from_threshold(
    *,
    metric_value: float,
    threshold: dict[str, object] | None,
    direct_quote_support: bool,
    quote_support_reason: str,
) -> tuple[str, str, str, float, float, float, str]:
    if direct_quote_support:
        return ("PASS", "INFO", "active_quote_support", np.nan, np.nan, np.nan, quote_support_reason)
    if threshold is None:
        return (
            "UNSUPPORTED",
            "P2",
            "missing_historical_threshold",
            np.nan,
            np.nan,
            0.0,
            "historical threshold row missing",
        )
    n_history = float(pd.to_numeric(pd.Series([threshold.get("n_snapshots")]), errors="coerce").iloc[0])
    min_required = float(pd.to_numeric(pd.Series([threshold.get("min_required_n")]), errors="coerce").iloc[0])
    warning = float(pd.to_numeric(pd.Series([threshold.get("p90")]), errors="coerce").iloc[0])
    critical = float(pd.to_numeric(pd.Series([threshold.get("p975")]), errors="coerce").iloc[0])
    threshold_status = str(threshold.get("status", "")).upper()
    if (
        not np.isfinite(n_history)
        or not np.isfinite(min_required)
        or n_history < min_required
        or threshold_status not in {"PASS", "OK", "USED"}
        or not np.isfinite(warning)
        or not np.isfinite(critical)
    ):
        return (
            "UNSUPPORTED",
            "P2",
            "insufficient_historical_threshold",
            warning,
            critical,
            n_history if np.isfinite(n_history) else 0.0,
            f"insufficient historical sample n={n_history}, min_required={min_required}",
        )
    value = abs(float(metric_value))
    if value > critical:
        return ("CRITICAL", "P1", "historical_p90_p975", warning, critical, n_history, "above historical P97.5")
    if value > warning:
        return ("WARNING", "P2", "historical_p90_p975", warning, critical, n_history, "above historical P90")
    return ("PASS", "INFO", "historical_p90_p975", warning, critical, n_history, "within historical envelope")


def _sign_reason(*, calendar_spread: float, parent_spread: float, month_spread: float) -> str:
    cal_sign = np.sign(calendar_spread)
    parent_sign = np.sign(parent_spread)
    month_sign = np.sign(month_spread)
    if parent_sign != 0.0 and month_sign not in (0.0, parent_sign):
        return "month sign contradicts comparable parent spread"
    if cal_sign != 0.0 and month_sign not in (0.0, cal_sign):
        return "month sign contradicts calendar spread after comparable-block check"
    return "month sign is compatible with comparable-block decomposition"


def _safe_int(value: object, *, default: int) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _status_rank(status: str) -> int:
    return {"PASS": 0, "WARNING": 1, "UNSUPPORTED": 2, "CRITICAL": 3}.get(str(status), 2)
