from __future__ import annotations

import pandas as pd

from pfc_shaping.calibration.monthly_curve_audit import audit_monthly_curve_shape
from pfc_shaping.calibration.monthly_curve_priors import (
    recenter_shape_by_parent,
)
from pfc_shaping.calibration.monthly_forward_curve import build_monthly_constraint_system


def test_monthly_audit_passes_solver_candidate_without_critical_flags():
    constraints = _constraints()
    curve = _parent_flat_curve(constraints)

    audit = audit_monthly_curve_shape(
        curve,
        constraints,
        year_pairs=[(2028, 2029)],
        historical_thresholds=_thresholds(),
        neighbor_level_leakage_max_abs=1e-13,
    )

    assert "CRITICAL" not in set(audit["status"])
    assert audit.loc[audit["gate_id"].eq("hard_monthly_curve_repricing"), "status"].eq("PASS").all()
    assert audit.loc[audit["gate_id"].eq("neighbor_level_leakage"), "status"].iloc[0] == "PASS"
    assert audit.loc[audit["gate_id"].eq("monthly_shape_regression_2028_2030"), "status"].iloc[0] == "PASS"


def test_monthly_audit_flags_winter_inversion_without_direct_quote_support():
    constraints = _constraints()
    curve = _bad_repriced_curve_with_december_inversion(constraints)

    audit = audit_monthly_curve_shape(
        curve,
        constraints,
        year_pairs=[(2028, 2029)],
        historical_thresholds=_thresholds(),
    )
    dec = audit[
        audit["gate_id"].eq("same_month_rank_consistency")
        & audit["year"].eq(2028)
        & audit["month"].eq(12)
    ].iloc[0]

    assert dec["status"] == "CRITICAL"
    assert "above historical P97.5" in dec["evidence"]
    assert audit.loc[audit["gate_id"].eq("monthly_shape_regression_2028_2030"), "status"].iloc[0] == "CRITICAL"


def test_monthly_audit_marks_shape_gates_unsupported_when_threshold_history_is_missing():
    constraints = _constraints()
    curve = _parent_flat_curve(constraints)

    audit = audit_monthly_curve_shape(curve, constraints, year_pairs=[(2028, 2029)])

    shape = audit[audit["gate_id"].eq("same_month_rank_consistency")]
    comparable = audit[audit["gate_id"].eq("residual_vs_implied_comparable_block")]
    assert not shape.empty
    assert not comparable.empty
    assert set(shape["status"]) == {"UNSUPPORTED"}
    assert set(comparable["status"]) == {"UNSUPPORTED"}
    assert audit.loc[audit["gate_id"].eq("monthly_shape_regression_2028_2030"), "status"].iloc[0] == "UNSUPPORTED"


def test_monthly_audit_marks_shape_gates_unsupported_for_small_history_sample():
    constraints = _constraints()
    curve = _parent_flat_curve(constraints)

    audit = audit_monthly_curve_shape(
        curve,
        constraints,
        year_pairs=[(2028, 2029)],
        historical_thresholds=_thresholds(n_snapshots=5, min_required_n=24),
    )

    shape = audit[audit["gate_id"].eq("same_month_rank_consistency")]
    assert set(shape["status"]) == {"UNSUPPORTED"}
    assert set(shape["threshold_source"]) == {"insufficient_historical_threshold"}


def test_monthly_audit_required_phase_f_columns_are_present():
    constraints = _constraints()
    curve = _parent_flat_curve(constraints)

    audit = audit_monthly_curve_shape(
        curve,
        constraints,
        year_pairs=[(2028, 2029)],
        historical_thresholds=_thresholds(),
    )

    required = {
        "gate_id",
        "status",
        "severity",
        "market",
        "load_type",
        "year",
        "month",
        "product",
        "parent_block_id",
        "parent_block_type",
        "parent_hours",
        "parent_mean",
        "month_price",
        "month_deviation_from_parent",
        "metric_name",
        "metric_value",
        "threshold_warning",
        "threshold_critical",
        "threshold_source",
        "n_history",
        "n_neighbors",
        "evidence",
        "remediation_hint",
    }
    assert required <= set(audit.columns)


def test_monthly_audit_flags_calendar_repricing_break():
    constraints = _constraints()
    curve = _parent_flat_curve(constraints)
    curve.loc[pd.Period("2029-06", freq="M")] += 5.0

    audit = audit_monthly_curve_shape(
        curve,
        constraints,
        year_pairs=[(2028, 2029)],
        historical_thresholds=_thresholds(),
    )

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


def _bad_repriced_curve_with_december_inversion(constraints) -> pd.Series:
    raw = {}
    for month in constraints.delivery_grid.months:
        raw[month] = 0.0
        if month.year == 2028 and month.month == 12:
            raw[month] = -30.0
    shape = recenter_shape_by_parent(pd.Series(raw), constraints)
    values = {}
    for month in constraints.delivery_grid.months:
        bucket = constraints.month_buckets.loc[month]
        values[month] = constraints.bucket_targets[str(bucket)] + float(shape.loc[month])
    return pd.Series(values, dtype=float, name="monthly_base_eur_mwh")


def _parent_flat_curve(constraints) -> pd.Series:
    values = {}
    for month in constraints.delivery_grid.months:
        bucket = constraints.month_buckets.loc[month]
        values[month] = constraints.bucket_targets[str(bucket)]
    return pd.Series(values, dtype=float, name="monthly_base_eur_mwh")


def _thresholds(*, n_snapshots: int = 120, min_required_n: int = 24) -> pd.DataFrame:
    rows = []
    for gate_id, metric in [
        ("same_month_rank_consistency", "same_month_shape_delta_abs_eur_mwh"),
        ("residual_vs_implied_comparable_block", "comparable_block_shape_delta_abs_eur_mwh"),
    ]:
        rows.append(
            {
                "gate_id": gate_id,
                "metric": metric,
                "market": "CH",
                "delivery_bucket": "all",
                "lookback_start": "2020-01-01",
                "lookback_end": "2026-01-01",
                "n_snapshots": n_snapshots,
                "min_required_n": min_required_n,
                "p50": 2.0,
                "p90": 8.0,
                "p975": 15.0,
                "max_observed": 25.0,
                "regime_filter": "synthetic",
                "status": "PASS",
            }
        )
    return pd.DataFrame(rows)
