from __future__ import annotations

import json

from scripts.audit_epex_shape_lab_governance import audit_governance


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_audit_epex_shape_lab_governance_passes_lab_only_artifacts(tmp_path) -> None:
    lab = tmp_path / "lab.json"
    independent = tmp_path / "independent.json"
    ompex = tmp_path / "ompex.json"
    out = tmp_path / "decision.json"
    _write_json(
        lab,
        {
            "activation_status": "lab_only",
            "production_approved": False,
            "ompex_used_in_selection": False,
            "benchmark_policy": "advisory_only_ompex_forbidden_as_input_target_loss_or_gate",
            "constraint_summary": {
                "base_monthly_constraints": 2,
                "peak_monthly_constraints": 2,
                "max_after_abs_error_eur_mwh": 1e-8,
            },
            "audit": {"weighted_negative_hours": 0},
        },
    )
    _write_json(
        independent,
        {
            "benchmark_policy": "independent_no_ompex",
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
            "finite_adjusted_ok": True,
            "quantile_order_adjusted_ok": True,
            "max_abs_monthly_mean_delta_eur_mwh": 1e-8,
            "max_abs_width_delta_eur_mwh": 0.0,
            "weighted_negative_hours_adjusted": 0,
            "n_hours": 100,
        },
    )
    _write_json(
        ompex,
        {
            "benchmark_policy": "advisory",
            "read_only": True,
            "ompex_used_in_model": False,
            "n_points": 80,
        },
    )

    decision = audit_governance(
        lab_manifest=lab,
        independent_summary=independent,
        ompex_metrics=ompex,
        output_json=out,
    )

    assert decision["status"] == "PASS"
    assert decision["production_approval"] == "NO"
    assert decision["promotion_gate"] is False
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "PASS"


def test_audit_epex_shape_lab_governance_fails_if_ompex_was_used_for_selection(tmp_path) -> None:
    lab = tmp_path / "lab.json"
    independent = tmp_path / "independent.json"
    out = tmp_path / "decision.json"
    _write_json(
        lab,
        {
            "activation_status": "lab_only",
            "production_approved": False,
            "ompex_used_in_selection": True,
            "benchmark_policy": "advisory_only_ompex_forbidden_as_input_target_loss_or_gate",
            "constraint_summary": {
                "base_monthly_constraints": 2,
                "peak_monthly_constraints": 2,
                "max_after_abs_error_eur_mwh": 1e-8,
            },
            "audit": {"weighted_negative_hours": 0},
        },
    )
    _write_json(
        independent,
        {
            "benchmark_policy": "independent_no_ompex",
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
            "finite_adjusted_ok": True,
            "quantile_order_adjusted_ok": True,
            "max_abs_monthly_mean_delta_eur_mwh": 1e-8,
            "max_abs_width_delta_eur_mwh": 0.0,
            "weighted_negative_hours_adjusted": 0,
            "n_hours": 100,
        },
    )

    decision = audit_governance(lab_manifest=lab, independent_summary=independent, output_json=out)

    assert decision["status"] == "FAIL"
    assert any(row["name"] == "lab_ompex_not_selected" for row in decision["checks"] if row["status"] == "FAIL")
