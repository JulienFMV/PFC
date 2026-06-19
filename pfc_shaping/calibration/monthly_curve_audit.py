"""Numerical audit gates for monthly forward curve candidates."""

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
    repricing_tolerance: float = 1e-8,
    material_calendar_spread_eur_mwh: float = 5.0,
    near_clone_spread_eur_mwh: float = 0.75,
) -> pd.DataFrame:
    """Return machine-readable monthly shape audit rows.

    These are governance gates, not model patches.  Historical threshold
    calibration will replace the provisional thresholds before production
    promotion, but the gates already fail known-bad sparse-year pathologies.
    """

    curve = monthly_curve.reindex(constraints.delivery_grid.months).astype(float)
    if curve.isna().any():
        missing = [str(month) for month in curve[curve.isna()].index]
        raise ValueError(f"monthly curve is missing delivery months: {missing}")
    pairs = list(year_pairs or _adjacent_year_pairs(constraints.delivery_grid.months))
    rows: list[dict[str, object]] = []
    rows.extend(_active_repricing_rows(curve, constraints, repricing_tolerance))
    for year_a, year_b in pairs:
        rows.extend(
            _same_month_rows(
                curve,
                constraints,
                year_a=year_a,
                year_b=year_b,
                material_calendar_spread_eur_mwh=material_calendar_spread_eur_mwh,
                near_clone_spread_eur_mwh=near_clone_spread_eur_mwh,
            )
        )
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
                parent_mean=float(row.target),
                month_price=np.nan,
                month_deviation=np.nan,
                metric_name="active_constraint_abs_error_eur_mwh",
                metric_value=abs(error),
                threshold_warning=tolerance,
                threshold_critical=tolerance,
                threshold_source="hard_constraint",
                evidence=f"achieved={float(row.achieved):.12g}, target={float(row.target):.12g}",
                remediation_hint="Fix monthly solver constraints before inspecting shape.",
            )
        )
    return rows


def _same_month_rows(
    curve: pd.Series,
    constraints: MonthlyConstraintSystem,
    *,
    year_a: int,
    year_b: int,
    material_calendar_spread_eur_mwh: float,
    near_clone_spread_eur_mwh: float,
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
        status, severity, reason = _same_month_status(
            month=month_number,
            calendar_spread=calendar_spread,
            parent_spread=parent_spread,
            month_spread=month_spread,
            direct_quote_support=parent_a["type"] == "month" or parent_b["type"] == "month",
            material_calendar_spread_eur_mwh=material_calendar_spread_eur_mwh,
            near_clone_spread_eur_mwh=near_clone_spread_eur_mwh,
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
                parent_mean=parent_a["target"],
                month_price=price_a,
                month_deviation=price_a - parent_a["target"],
                metric_name="month_spread_eur_mwh",
                metric_value=month_spread,
                threshold_warning=near_clone_spread_eur_mwh,
                threshold_critical=0.0,
                threshold_source="provisional_governance_fixture",
                evidence=(
                    f"calendar_spread={calendar_spread:.4f}, "
                    f"parent_spread={parent_spread:.4f}, shape_delta={shape_delta:.4f}, {reason}"
                ),
                remediation_hint="Review comparable-block shape prior or active quote support.",
                year_b=year_b,
                price_b=price_b,
                parent_mean_b=parent_b["target"],
                parent_mix_adjustment=parent_spread - calendar_spread,
            )
        )
    return rows


def _same_month_status(
    *,
    month: int,
    calendar_spread: float,
    parent_spread: float,
    month_spread: float,
    direct_quote_support: bool,
    material_calendar_spread_eur_mwh: float,
    near_clone_spread_eur_mwh: float,
) -> tuple[str, str, str]:
    if direct_quote_support:
        return ("PASS", "INFO", "direct monthly quote support")
    if abs(calendar_spread) < material_calendar_spread_eur_mwh:
        return ("PASS", "INFO", "calendar spread immaterial")
    cal_sign = np.sign(calendar_spread)
    parent_sign = np.sign(parent_spread)
    month_sign = np.sign(month_spread)
    if month in (11, 12) and cal_sign > 0 and month_spread < 0.0:
        return ("CRITICAL", "P1", "winter inversion against positive calendar spread")
    if parent_sign == cal_sign and month_sign not in (0.0, parent_sign):
        return ("CRITICAL", "P1", "month sign contradicts comparable parent spread")
    if abs(month_spread) <= near_clone_spread_eur_mwh:
        return ("WARNING", "P2", "near-clone month under material calendar spread")
    return ("PASS", "INFO", "spread supported by comparable parent decomposition")


def _parent_info(constraints: MonthlyConstraintSystem, month: pd.Period) -> dict[str, object] | None:
    bucket = constraints.month_buckets.loc[month]
    if pd.isna(bucket):
        return None
    bucket_text = str(bucket)
    row = constraints.rows[constraints.rows["product"].astype(str).eq(bucket_text)]
    parent_product = bucket_text if row.empty else str(row["parent_product"].iloc[0])
    return {
        "bucket": bucket_text,
        "product": parent_product,
        "type": _parent_type(bucket_text, parent_product),
        "target": float(constraints.bucket_targets[bucket_text]),
    }


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
    parent_mean: float,
    month_price: float,
    month_deviation: float,
    metric_name: str,
    metric_value: float,
    threshold_warning: float,
    threshold_critical: float,
    threshold_source: str,
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
        "parent_hours": np.nan,
        "parent_mean": parent_mean,
        "month_price": month_price,
        "month_deviation_from_parent": month_deviation,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "threshold_warning": threshold_warning,
        "threshold_critical": threshold_critical,
        "threshold_source": threshold_source,
        "n_history": np.nan,
        "n_neighbors": np.nan,
        "evidence": evidence,
        "remediation_hint": remediation_hint,
    }
    row.update(extra)
    return row
