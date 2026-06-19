from __future__ import annotations

import pandas as pd

from pfc_shaping.calibration.monthly_curve_audit import audit_monthly_curve_shape
from pfc_shaping.calibration.monthly_curve_priors import (
    recenter_shape_by_parent,
)
from pfc_shaping.calibration.monthly_forward_curve import build_monthly_constraint_system


def test_monthly_audit_passes_solver_candidate_without_critical_flags():
    constraints = _constraints()
    curve = _coherent_zero_mean_curve(constraints)

    audit = audit_monthly_curve_shape(curve, constraints, year_pairs=[(2028, 2029)])

    assert "CRITICAL" not in set(audit["status"])
    assert audit.loc[audit["gate_id"].eq("hard_monthly_curve_repricing"), "status"].eq("PASS").all()


def test_monthly_audit_flags_winter_inversion_without_direct_quote_support():
    constraints = _constraints()
    curve = _bad_curve_with_december_inversion(constraints)

    audit = audit_monthly_curve_shape(curve, constraints, year_pairs=[(2028, 2029)])
    dec = audit[
        audit["gate_id"].eq("same_month_rank_consistency")
        & audit["year"].eq(2028)
        & audit["month"].eq(12)
    ].iloc[0]

    assert dec["status"] == "CRITICAL"
    assert "winter inversion" in dec["evidence"]


def test_monthly_audit_flags_calendar_repricing_break():
    constraints = _constraints()
    curve = _bad_curve_with_december_inversion(constraints)
    curve.loc[pd.Period("2029-06", freq="M")] += 5.0

    audit = audit_monthly_curve_shape(curve, constraints, year_pairs=[(2028, 2029)])

    broken = audit[
        audit["gate_id"].eq("hard_monthly_curve_repricing")
        & audit["product"].eq("2029")
    ].iloc[0]
    assert broken["status"] == "CRITICAL"


def _constraints():
    months = pd.period_range("2028-01", "2029-12", freq="M")
    return build_monthly_constraint_system(
        months,
        {"2028": 80.40, "2028-Q1": 109.97, "2029": 72.41},
    )


def _bad_curve_with_december_inversion(constraints) -> pd.Series:
    values = {}
    for month in constraints.delivery_grid.months:
        if month.year == 2028 and month.month <= 3:
            values[month] = 109.97
        elif month.year == 2028:
            values[month] = constraints.bucket_targets["2028-RESIDUAL"]
        else:
            values[month] = 72.41
    values[pd.Period("2028-12", freq="M")] = 70.0
    values[pd.Period("2028-04", freq="M")] = 72.41
    return pd.Series(values, dtype=float, name="monthly_base_eur_mwh")


def _coherent_zero_mean_curve(constraints) -> pd.Series:
    raw = {}
    for month in constraints.delivery_grid.months:
        value = 0.0
        if month.year == 2028 and month.month == 12:
            value = 24.0
        elif month.year == 2028 and month.month == 11:
            value = 14.0
        elif month.year == 2028 and month.month in (5, 6, 7):
            value = -8.0
        elif month.year == 2029 and month.month == 12:
            value = 5.0
        elif month.year == 2029 and month.month in (5, 6, 7):
            value = -4.0
        raw[month] = value
    shape = recenter_shape_by_parent(pd.Series(raw), constraints)
    values = {}
    for month in constraints.delivery_grid.months:
        bucket = constraints.month_buckets.loc[month]
        values[month] = constraints.bucket_targets[str(bucket)] + float(shape.loc[month])
    return pd.Series(values, dtype=float, name="monthly_base_eur_mwh")
