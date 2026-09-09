from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from scripts import audit_ch_lt_laptop_model_pair as audit


def test_current_laptop_model_pair_is_exact_and_local_only() -> None:
    report = audit.audit_pair(audit.ROOT)

    assert report["status"] == audit.STATUS
    assert report["production_authorization"] is False
    assert report["promotion_gate"] is False
    assert report["scientific_admission"] is False
    assert report["countable_prospective_origin"] is False
    pair = report["deterministic_pair"]
    assert len(pair["byte_identical_material_roles"]) == 9
    assert pair["path_normalized_quality_summary_equal"] is True
    assert pair["runs"]["A"]["workspace_root_sys_path_count"] == 1
    assert pair["runs"]["B"]["active_processes_after_wait"] == 0
    structural = report["structural_quality"]
    assert structural["monthly_level_authority"] == "solver"
    assert structural["forward_source_kind"] == "TEST_FIXTURE"
    assert structural["maximum_monthly_mean_residual_eur_mwh"] <= 1e-9
    assert structural["maximum_hourly_quarter_factor_neutrality_residual"] <= 1e-12
    assert (
        structural["maximum_hourly_intraday_price_neutrality_residual_eur_mwh"]
        <= audit._INTRADAY_PRICE_NEUTRALITY_TOLERANCE_EUR_MWH
    )
    assert structural["maximum_final_no_q_parent_hour_range_eur_mwh"] <= 1e-12
    assert structural["final_product_replay"]["cal2030_base_hours"] == 8_760
    historical = report["historical_non_ch_non_confirmatory_diagnostic"]
    assert historical["candidate_accepted"] is False
    assert historical["target_is_full_ch_lt"] is False
    assert historical["usable_for_model_selection"] is False


def test_execution_receipt_authority_escalation_is_rejected(tmp_path) -> None:
    original = audit.ROOT / audit.RUNS[0].execution_receipt_relative
    document = json.loads(original.read_text(encoding="utf-8"))
    document["scientific_authorization"] = True
    payload = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    target = tmp_path / "execution-receipt.json"
    target.write_bytes(payload)
    relative = target.relative_to(audit.ROOT)

    # Rebind the independently hashed supervisor receipt to the mutated
    # execution receipt.  Otherwise the test correctly fails earlier at the
    # supervisor TOCTOU binding and never exercises the authority invariant.
    supervisor_original = audit.ROOT / audit.RUNS[0].supervisor_receipt_relative
    supervisor_document = json.loads(supervisor_original.read_text(encoding="utf-8"))
    supervisor_document["execution_receipt"].update(
        {
            "path": str(target),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    )
    supervisor_payload = (
        json.dumps(supervisor_document, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    supervisor_target = tmp_path / "supervisor-receipt.json"
    supervisor_target.write_bytes(supervisor_payload)
    supervisor_relative = supervisor_target.relative_to(audit.ROOT)
    spec = replace(
        audit.RUNS[0],
        execution_receipt_relative=relative,
        execution_receipt_sha256=hashlib.sha256(payload).hexdigest(),
        supervisor_receipt_relative=supervisor_relative,
        supervisor_receipt_sha256=hashlib.sha256(supervisor_payload).hexdigest(),
    )

    with pytest.raises(audit.LaptopModelPairError, match="scientific_authorization"):
        audit._audit_execution_receipt(audit.ROOT, spec)


def test_completion_hard_quote_escalation_is_rejected(tmp_path) -> None:
    original = audit.ROOT / audit.RUNS[0].completion_relative
    document = json.loads(original.read_text(encoding="utf-8"))
    document["hard_quote_eligible"] = True
    payload = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    target = tmp_path / "completion.json"
    target.write_bytes(payload)
    relative = target.relative_to(audit.ROOT)
    spec = replace(
        audit.RUNS[0],
        completion_relative=relative,
        completion_sha256=hashlib.sha256(payload).hexdigest(),
    )

    with pytest.raises(audit.LaptopModelPairError, match="hard_quote_eligible"):
        audit._audit_completion(audit.ROOT, spec)


def test_publish_rejects_shadow_output_directory(tmp_path) -> None:
    with pytest.raises(audit.LaptopModelPairError, match="canonical qualification path"):
        audit.publish(tmp_path / "shadow")
