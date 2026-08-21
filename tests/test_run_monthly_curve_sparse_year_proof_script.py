from __future__ import annotations

import argparse

import pandas as pd

from pfc_shaping.calibration.monthly_curve_audit import audit_monthly_curve_shape
from pfc_shaping.calibration.monthly_forward_curve import MarketQuote
from pfc_shaping.calibration.monthly_forward_curve import build_monthly_constraint_system
from scripts.run_monthly_curve_sparse_year_proof import (
    _active_snapshot_quote_coverage,
    _build_sparse_proof_governance_gates,
    _monthly_curve_frame,
    _sparse_year_checks,
    _sparse_year_seam_checks,
)


def test_sparse_year_proof_emits_required_governance_gates_when_hashes_are_supplied() -> None:
    args = argparse.Namespace(
        require_lambda_artifact=True,
        require_path_parity=True,
        production_monthly_solution_hash="solution",
        export_monthly_solution_hash="solution",
        production_active_constraints_hash="constraints",
        export_active_constraints_hash="constraints",
    )

    gates = _build_sparse_proof_governance_gates(
        args=args,
        run_timestamp=pd.Timestamp("2026-06-17"),
        own_quotes=(_quote("2028"),),
        neighbor_quotes=(),
        eex_history=_history(),
        active_config_hash="config",
        selected_config_hash="config",
    )

    required = gates[
        gates["gate_id"].isin(
            {
                "lambda_calibration_artifact_present",
                "production_export_path_parity",
            }
        )
    ]
    assert set(required["gate_id"]) == {
        "lambda_calibration_artifact_present",
        "production_export_path_parity",
    }
    assert set(required["status"]) == {"PASS"}


def test_sparse_year_proof_does_not_emit_optional_governance_gates_by_default() -> None:
    args = argparse.Namespace(
        require_lambda_artifact=False,
        require_path_parity=False,
        production_monthly_solution_hash="",
        export_monthly_solution_hash="",
        production_active_constraints_hash="",
        export_active_constraints_hash="",
    )

    gates = _build_sparse_proof_governance_gates(
        args=args,
        run_timestamp=pd.Timestamp("2026-06-17"),
        own_quotes=(_quote("2028"),),
        neighbor_quotes=(),
        eex_history=_history(),
        active_config_hash=None,
        selected_config_hash=None,
    )

    assert set(gates["gate_id"]) == {"point_in_time_data_contract"}


def test_sparse_year_checks_carry_same_month_and_comparable_block_audit_context() -> None:
    constraints = build_monthly_constraint_system(
        pd.period_range("2028-01", "2029-12", freq="M"),
        {"2028": 80.40, "2028-Q1": 109.97, "2029": 72.41},
    )
    curve = pd.Series(
        {
            month: float(constraints.bucket_targets[str(constraints.month_buckets.loc[month])])
            for month in constraints.delivery_grid.months
        },
        dtype=float,
        name="monthly_base_eur_mwh",
    )
    monthly = _monthly_curve_frame(curve, constraints)
    audit_gates = audit_monthly_curve_shape(curve, constraints, year_pairs=[(2028, 2029)])

    checks = _sparse_year_checks(
        monthly,
        constraints,
        audit_gates=audit_gates,
        year_a=2028,
        year_b=2029,
    )

    april = checks[checks["month"].eq(4)].iloc[0]
    january = checks[checks["month"].eq(1)].iloc[0]

    assert april["comparable_block_gate_status"] == "UNSUPPORTED"
    assert april["comparable_block_gate_metric_name"] == "comparable_block_shape_delta_abs_eur_mwh"
    assert "parent_block_a=2028-RESIDUAL" in april["comparable_block_gate_evidence"]
    assert january["same_month_gate_status"] == "UNSUPPORTED"
    assert january["same_month_gate_metric_name"] == "same_month_shape_delta_abs_eur_mwh"


def test_sparse_year_seam_checks_surface_residual_boundary_rows() -> None:
    constraints = build_monthly_constraint_system(
        pd.period_range("2028-01", "2029-12", freq="M"),
        {"2028": 80.40, "2028-Q1": 109.97, "2028-Q2": 61.0, "2029": 72.41},
    )
    curve = pd.Series(
        {
            month: float(constraints.bucket_targets[str(constraints.month_buckets.loc[month])])
            for month in constraints.delivery_grid.months
        },
        dtype=float,
        name="monthly_base_eur_mwh",
    )
    monthly = _monthly_curve_frame(curve, constraints)

    seam_checks = _sparse_year_seam_checks(monthly, constraints)

    residual_boundary = seam_checks[
        seam_checks["bucket"].astype(str).eq("2028-Q2->2028-RESIDUAL")
        & seam_checks["check_type"].astype(str).eq("inter_bucket_seam")
    ].iloc[0]

    assert residual_boundary["from_month"] == 6
    assert residual_boundary["to_month"] == 7
    assert pd.notna(residual_boundary["residual_edge_deviation_eur_mwh"])


def test_active_snapshot_quote_coverage_marks_unassigned_years() -> None:
    constraints = build_monthly_constraint_system(
        pd.period_range("2027-01", "2030-12", freq="M"),
        {"2027": 97.54, "2028": 81.41},
    )
    curve = pd.Series(
        {
            month: float(constraints.bucket_targets[str(constraints.month_buckets.loc[month])])
            if pd.notna(constraints.month_buckets.loc[month])
            else 0.0
            for month in constraints.delivery_grid.months
        },
        dtype=float,
        name="monthly_base_eur_mwh",
    )
    monthly = _monthly_curve_frame(curve, constraints)

    coverage = _active_snapshot_quote_coverage(monthly)

    year_2028 = coverage[coverage["year"].eq(2028)].iloc[0]
    year_2029 = coverage[coverage["year"].eq(2029)].iloc[0]
    assert year_2028["status"] == "fully_assigned"
    assert year_2028["bucket_kinds"] == "calendar"
    assert year_2029["status"] == "unquoted_year_fallback"
    assert year_2029["unassigned_month_count"] == 12


def _quote(product: str) -> MarketQuote:
    return MarketQuote(
        market="CH",
        product=product,
        load_type="BASE",
        price=80.0,
        snapshot_date=pd.Timestamp("2026-06-17"),
        available_at=pd.Timestamp("2026-06-17"),
    )


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-06-17"),
                "market": "CH",
                "load_type": "BASE",
                "product": "2028",
                "price": 80.0,
            }
        ]
    )
