from __future__ import annotations

import pandas as pd

from pfc_shaping.calibration.monthly_curve_promotion import (
    evaluate_monthly_curve_promotion,
)


def test_promotion_accepts_documented_far_horizon_unsupported_shape_risk() -> None:
    decision = evaluate_monthly_curve_promotion(
        _audit_gates(),
        _thresholds(status="UNSUPPORTED", n_snapshots=13, min_required_n=24),
        run_timestamp=pd.Timestamp("2026-06-17"),
    )

    assert decision.approved
    assert decision.status == "PROMOTION_EVIDENCE_PASS"
    assert "ACCEPT" in set(decision.details["decision"])


def test_promotion_blocks_any_critical_gate() -> None:
    gates = _audit_gates()
    gates.loc[gates["gate_id"].eq("same_month_rank_consistency"), "status"] = "CRITICAL"

    decision = evaluate_monthly_curve_promotion(
        gates,
        _thresholds(status="PASS", n_snapshots=120, min_required_n=24),
        run_timestamp=pd.Timestamp("2026-06-17"),
    )

    assert not decision.approved
    assert decision.status == "BLOCKED"
    assert "CRITICAL audit gate present" in set(decision.details["reason"])


def test_promotion_blocks_near_horizon_unsupported_even_with_insufficient_threshold_sample() -> None:
    gates = _audit_gates(year=2027)

    decision = evaluate_monthly_curve_promotion(
        gates,
        _thresholds(status="UNSUPPORTED", n_snapshots=13, min_required_n=24),
        run_timestamp=pd.Timestamp("2026-06-17"),
    )

    assert not decision.approved
    unsupported = decision.details[
        decision.details["gate_id"].eq("residual_vs_implied_comparable_block")
    ].iloc[0]
    assert unsupported["decision"] == "BLOCK"
    assert "near-horizon" in unsupported["reason"]


def test_promotion_blocks_unsupported_without_threshold_evidence() -> None:
    decision = evaluate_monthly_curve_promotion(
        _audit_gates(),
        pd.DataFrame(columns=_thresholds().columns),
        run_timestamp=pd.Timestamp("2026-06-17"),
    )

    assert not decision.approved
    assert "missing threshold evidence for UNSUPPORTED row" in set(decision.details["reason"])


def test_promotion_blocks_missing_required_hard_gate() -> None:
    gates = _audit_gates()
    gates = gates[~gates["gate_id"].eq("neighbor_level_leakage")]

    decision = evaluate_monthly_curve_promotion(
        gates,
        _thresholds(status="UNSUPPORTED", n_snapshots=13, min_required_n=24),
        run_timestamp=pd.Timestamp("2026-06-17"),
    )

    assert not decision.approved
    row = decision.details[decision.details["gate_id"].eq("neighbor_level_leakage")].iloc[0]
    assert row["decision"] == "BLOCK"
    assert row["reason"] == "required hard gate is missing"


def _audit_gates(*, year: int = 2028) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gate_id": "hard_monthly_curve_repricing",
                "status": "PASS",
                "year": year,
                "month": 1,
                "product": str(year),
                "metric_name": "active_constraint_abs_error_eur_mwh",
            },
            {
                "gate_id": "neighbor_level_leakage",
                "status": "PASS",
                "year": 0,
                "month": 0,
                "product": "",
                "metric_name": "neighbor_level_leakage_max_abs_eur_mwh",
            },
            {
                "gate_id": "same_month_rank_consistency",
                "status": "PASS",
                "year": year,
                "month": 12,
                "product": f"{year}-12",
                "metric_name": "same_month_shape_delta_abs_eur_mwh",
            },
            {
                "gate_id": "residual_vs_implied_comparable_block",
                "status": "UNSUPPORTED",
                "year": year,
                "month": 4,
                "product": f"{year}-04",
                "metric_name": "comparable_block_shape_delta_abs_eur_mwh",
            },
            {
                "gate_id": "monthly_shape_regression_2028_2030",
                "status": "UNSUPPORTED",
                "year": year,
                "month": 0,
                "product": "2028_2030",
                "metric_name": "targeted_monthly_gate_status",
            },
        ]
    )


def _thresholds(
    *,
    status: str = "UNSUPPORTED",
    n_snapshots: int = 13,
    min_required_n: int = 24,
) -> pd.DataFrame:
    p90 = pd.NA if status == "UNSUPPORTED" else 8.0
    p975 = pd.NA if status == "UNSUPPORTED" else 15.0
    return pd.DataFrame(
        [
            {
                "gate_id": "residual_vs_implied_comparable_block",
                "metric": "comparable_block_shape_delta_abs_eur_mwh",
                "market": "CH",
                "delivery_bucket": "month_04",
                "lookback_start": "2020-06-17",
                "lookback_end": "2026-06-17",
                "n_snapshots": n_snapshots,
                "min_required_n": min_required_n,
                "p50": pd.NA,
                "p90": p90,
                "p975": p975,
                "max_observed": pd.NA,
                "regime_filter": "synthetic",
                "status": status,
            }
        ]
    )
