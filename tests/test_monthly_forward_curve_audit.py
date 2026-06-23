from __future__ import annotations

import pandas as pd

from pfc_shaping.calibration.monthly_curve_audit import (
    build_monthly_curve_historical_thresholds,
    build_monthly_curve_governance_gates,
    audit_monthly_curve_shape,
)
from pfc_shaping.calibration.monthly_curve_priors import (
    recenter_shape_by_parent,
)
from pfc_shaping.calibration.monthly_forward_curve import MarketQuote, build_monthly_constraint_system


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
    calendar_gate = audit[audit["gate_id"].eq("calendar_spread_seasonal_decomposition")].iloc[0]
    assert calendar_gate["status"] == "PASS"
    assert calendar_gate["metric_value"] <= 1e-8
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


def test_monthly_audit_flags_q4_vs_calendar_comparable_block_incoherence():
    constraints = _q4_constraints()
    curve = _bad_repriced_q4_curve_with_december_inversion(constraints)

    audit = audit_monthly_curve_shape(
        curve,
        constraints,
        year_pairs=[(2028, 2029)],
        historical_thresholds=_thresholds(),
    )
    dec = audit[
        audit["gate_id"].eq("residual_vs_implied_comparable_block")
        & audit["year"].eq(2028)
        & audit["month"].eq(12)
    ].iloc[0]

    assert dec["status"] == "CRITICAL"
    assert dec["parent_block_type"] == "quarter|calendar"
    assert dec["parent_type_pair"] == "quarter|calendar"
    assert "comparable_parent_types=quarter|calendar" in dec["evidence"]
    assert audit.loc[audit["gate_id"].eq("monthly_shape_regression_2028_2030"), "status"].iloc[0] == "CRITICAL"


def test_monthly_audit_q4_comparable_block_fails_closed_without_q4_threshold_history():
    constraints = _q4_constraints()
    curve = _parent_flat_curve(constraints)
    residual_only_thresholds = build_monthly_curve_historical_thresholds(
        _threshold_history(n_snapshots=30),
        run_timestamp=pd.Timestamp("2026-01-30"),
        min_required_n=24,
        lookback_years=None,
    )

    audit = audit_monthly_curve_shape(
        curve,
        constraints,
        year_pairs=[(2028, 2029)],
        historical_thresholds=residual_only_thresholds,
    )
    dec = audit[
        audit["gate_id"].eq("residual_vs_implied_comparable_block")
        & audit["year"].eq(2028)
        & audit["month"].eq(12)
    ].iloc[0]

    assert dec["status"] == "UNSUPPORTED"
    assert dec["parent_type_pair"] == "quarter|calendar"
    assert dec["threshold_source"] == "insufficient_historical_threshold"


def test_monthly_audit_normalizes_reversed_calendar_vs_q4_parent_type_pair():
    constraints = _q4_reversed_constraints()
    curve = _bad_repriced_curve_with_december_inversion(constraints)

    audit = audit_monthly_curve_shape(
        curve,
        constraints,
        year_pairs=[(2028, 2029)],
        historical_thresholds=_thresholds(),
    )
    dec = audit[
        audit["gate_id"].eq("residual_vs_implied_comparable_block")
        & audit["year"].eq(2028)
        & audit["month"].eq(12)
    ].iloc[0]

    assert dec["parent_block_type"] == "calendar|quarter"
    assert dec["parent_type_pair"] == "quarter|calendar"
    assert "comparable_parent_types=quarter|calendar" in dec["evidence"]


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
    calendar_gate = audit[audit["gate_id"].eq("calendar_spread_seasonal_decomposition")].iloc[0]
    assert broken["status"] == "CRITICAL"
    assert calendar_gate["status"] == "CRITICAL"
    assert calendar_gate["metric_name"] == "calendar_spread_decomposition_residual_abs_eur_mwh"


def test_historical_threshold_builder_emits_pass_rows_from_point_in_time_history():
    history = _threshold_history(n_snapshots=30)

    thresholds = build_monthly_curve_historical_thresholds(
        history,
        run_timestamp=pd.Timestamp("2026-01-30"),
        min_required_n=24,
        lookback_years=None,
    )

    same_all = thresholds[
        thresholds["gate_id"].eq("same_month_rank_consistency")
        & thresholds["delivery_bucket"].eq("all")
    ].iloc[0]
    comparable_apr = thresholds[
        thresholds["gate_id"].eq("residual_vs_implied_comparable_block")
        & thresholds["delivery_bucket"].eq("month_04")
    ].iloc[0]
    assert same_all["status"] == "PASS"
    assert comparable_apr["status"] == "PASS"
    assert same_all["n_snapshots"] == 30
    assert pd.notna(same_all["p90"])
    assert pd.notna(comparable_apr["p975"])


def test_historical_threshold_builder_calibrates_q4_vs_calendar_comparable_blocks():
    thresholds = build_monthly_curve_historical_thresholds(
        _q4_threshold_history(n_snapshots=30),
        run_timestamp=pd.Timestamp("2026-01-30"),
        min_required_n=24,
        lookback_years=None,
    )

    comparable_dec = thresholds[
        thresholds["gate_id"].eq("residual_vs_implied_comparable_block")
        & thresholds["delivery_bucket"].eq("month_12")
        & thresholds["parent_type_pair"].eq("quarter|calendar")
    ].iloc[0]

    assert comparable_dec["status"] == "PASS"
    assert comparable_dec["n_snapshots"] == 30
    assert comparable_dec["parent_type_pair"] == "quarter|calendar"
    assert pd.notna(comparable_dec["p975"])


def test_historical_threshold_builder_excludes_future_snapshot():
    history = pd.concat(
        [
            _threshold_history(n_snapshots=30),
            _threshold_history(n_snapshots=1, start="2026-03-01", shock=500.0),
        ],
        ignore_index=True,
    )

    thresholds = build_monthly_curve_historical_thresholds(
        history,
        run_timestamp=pd.Timestamp("2026-01-30"),
        min_required_n=24,
        lookback_years=None,
    )

    same_all = thresholds[
        thresholds["gate_id"].eq("same_month_rank_consistency")
        & thresholds["delivery_bucket"].eq("all")
    ].iloc[0]
    assert same_all["status"] == "PASS"
    assert float(same_all["max_observed"]) < 500.0


def test_historical_threshold_builder_emits_unsupported_when_sample_is_small():
    thresholds = build_monthly_curve_historical_thresholds(
        _threshold_history(n_snapshots=5),
        run_timestamp=pd.Timestamp("2026-01-30"),
        min_required_n=24,
        lookback_years=None,
    )

    assert set(thresholds["status"]) == {"UNSUPPORTED"}
    assert thresholds["p90"].isna().all()


def test_governance_gate_flags_future_available_quote():
    gates = build_monthly_curve_governance_gates(
        run_timestamp=pd.Timestamp("2026-06-17"),
        own_quotes=[
            MarketQuote(
                market="CH",
                product="2028",
                load_type="BASE",
                price=80.0,
                snapshot_date=pd.Timestamp("2026-06-18"),
                available_at=pd.Timestamp("2026-06-18"),
            )
        ],
    )

    row = gates[gates["gate_id"].eq("point_in_time_data_contract")].iloc[0]
    assert row["status"] == "CRITICAL"
    assert row["metric_value"] == 1.0


def test_governance_gate_passes_point_in_time_inputs():
    gates = build_monthly_curve_governance_gates(
        run_timestamp=pd.Timestamp("2026-06-17"),
        own_quotes=[
            MarketQuote(
                market="CH",
                product="2028",
                load_type="BASE",
                price=80.0,
                snapshot_date=pd.Timestamp("2026-06-17"),
                available_at=pd.Timestamp("2026-06-17"),
            )
        ],
        eex_history=pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-06-17"),
                    "market": "CH",
                    "load_type": "BASE",
                    "product": "2028",
                    "price": 80.0,
                }
            ]
        ),
    )

    row = gates[gates["gate_id"].eq("point_in_time_data_contract")].iloc[0]
    assert row["status"] == "PASS"
    assert row["metric_value"] == 0.0


def test_governance_gate_flags_lambda_hash_mismatch():
    gates = build_monthly_curve_governance_gates(
        run_timestamp=pd.Timestamp("2026-06-17"),
        active_config_hash="active",
        selected_config_hash="selected",
    )

    row = gates[gates["gate_id"].eq("lambda_calibration_artifact_present")].iloc[0]
    assert row["status"] == "CRITICAL"
    assert row["metric_name"] == "selected_config_hash_mismatch"


def test_governance_gate_flags_production_export_hash_mismatch():
    gates = build_monthly_curve_governance_gates(
        run_timestamp=pd.Timestamp("2026-06-17"),
        production_monthly_solution_hash="prod_solution",
        export_monthly_solution_hash="export_solution",
        production_active_constraints_hash="same_constraints",
        export_active_constraints_hash="same_constraints",
    )

    row = gates[gates["gate_id"].eq("production_export_path_parity")].iloc[0]
    assert row["status"] == "CRITICAL"
    assert row["metric_name"] == "prod_export_hash_mismatch"


def _constraints():
    months = pd.period_range("2028-01", "2029-12", freq="M")
    return build_monthly_constraint_system(
        months,
        {"2028": 80.40, "2028-Q1": 109.97, "2029": 72.41},
    )


def _q4_constraints():
    months = pd.period_range("2028-01", "2029-12", freq="M")
    return build_monthly_constraint_system(
        months,
        {"2028": 80.40, "2028-Q4": 70.00, "2029": 72.41},
    )


def _q4_reversed_constraints():
    months = pd.period_range("2028-01", "2029-12", freq="M")
    return build_monthly_constraint_system(
        months,
        {"2028": 80.40, "2029": 72.41, "2029-Q4": 70.00},
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


def _bad_repriced_q4_curve_with_december_inversion(constraints) -> pd.Series:
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


def _threshold_history(
    *,
    n_snapshots: int,
    start: str = "2026-01-01",
    shock: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for idx in range(n_snapshots):
        date = pd.Timestamp(start) + pd.Timedelta(days=idx)
        drift = float(idx % 7) * 0.2 + float(shock)
        base = [
            ("2028", 80.0 + 0.1 * drift),
            ("2028-Q1", 108.0 + 0.1 * drift),
            ("2029", 72.0 + 0.05 * drift),
        ]
        for product, price in base:
            rows.append(_history_row(date, product, price))
        for month in range(1, 13):
            seasonal = {1: 12, 2: 10, 3: 4, 4: -2, 5: -5, 6: -6, 7: -4, 8: -3, 9: 1, 10: 4, 11: 7, 12: 11}[month]
            rows.append(_history_row(date, f"2028-{month:02d}", 80.0 + seasonal + drift))
            rows.append(_history_row(date, f"2029-{month:02d}", 72.0 + 0.8 * seasonal + 0.5 * drift))
    return pd.DataFrame(rows)


def _q4_threshold_history(*, n_snapshots: int) -> pd.DataFrame:
    rows = []
    for idx in range(n_snapshots):
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=idx)
        drift = float(idx % 7) * 0.2
        base = [
            ("2028", 80.0 + 0.1 * drift),
            ("2028-Q4", 70.0 + 0.1 * drift),
            ("2029", 72.0 + 0.05 * drift),
        ]
        for product, price in base:
            rows.append(_history_row(date, product, price))
        for month in range(1, 13):
            seasonal = {1: 12, 2: 10, 3: 4, 4: -2, 5: -5, 6: -6, 7: -4, 8: -3, 9: 1, 10: 4, 11: 7, 12: 11}[month]
            rows.append(_history_row(date, f"2028-{month:02d}", 80.0 + seasonal + drift))
            rows.append(_history_row(date, f"2029-{month:02d}", 72.0 + 0.8 * seasonal + 0.5 * drift))
    return pd.DataFrame(rows)


def _history_row(date: pd.Timestamp, product: str, price: float) -> dict[str, object]:
    return {
        "date": date,
        "market": "CH",
        "load_type": "BASE",
        "product": product,
        "price": float(price),
        "source": "synthetic_threshold_fixture",
    }
