"""Verify the canonical current CH LT local-quality selection v5."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json

ROOT = Path(r"C:\Users\jbattaglia\PFC_LT")
SELECTION_RELATIVE = Path(
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-LOCAL-QUALITY-CURRENT-SELECTION-V5-20260731.json"
)
SELECTION_SHA256 = "f61509834d692d64ee17dad6030f3672969788aac39d3cb3d56b7d5db9b7207b"
SELECTION_ID = "aa4233b4b42159bc4d0b6868fe3c3f0ec3e2449db593dbb14cd363c5e27336c6"
STATUS = "CURRENT_LOCAL_ENGINEERING_QUALITY_SELECTION_VALID_NO_GO"
_MAX_BYTES = 4 * 1024 * 1024


class CurrentLocalQualitySelectionError(ValueError):
    """Raised when the selected quality chain is incomplete or rebound."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CurrentLocalQualitySelectionError(f"{label} must be a mapping")
    return value


def _json(payload: bytes, *, label: str) -> Mapping[str, object]:
    try:
        value = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CurrentLocalQualitySelectionError(f"{label} is not strict JSON") from exc
    return _mapping(value, label=label)


def _inside(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath(
            (os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(root)))
        ) == os.path.normcase(os.path.abspath(root))
    except ValueError:
        return False


def _read_bound(
    root: Path,
    raw_path: object,
    expected_sha256: object,
    *,
    label: str,
    parse_json: bool = True,
) -> tuple[Mapping[str, object], bytes]:
    selected = Path(str(raw_path))
    if not selected.is_absolute():
        selected = root / selected
    selected = assert_absolute_path_has_no_links(Path(os.path.abspath(selected)))
    if not _inside(selected, root):
        raise CurrentLocalQualitySelectionError(f"{label} escapes the canonical repository")
    payload = read_stable_single_link_file(selected, label=label, max_bytes=_MAX_BYTES)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise CurrentLocalQualitySelectionError(f"{label} SHA-256 mismatch")
    return (_json(payload, label=label) if parse_json else {}), payload


def verify_current_selection(root: Path = ROOT) -> dict[str, object]:
    root = assert_absolute_path_has_no_links(root)
    if Path.cwd() != root:
        raise CurrentLocalQualitySelectionError("cwd must be the canonical repository root")
    selection_path = root / SELECTION_RELATIVE
    payload = read_stable_single_link_file(
        selection_path, label="current local quality selection", max_bytes=_MAX_BYTES
    )
    if hashlib.sha256(payload).hexdigest() != SELECTION_SHA256:
        raise CurrentLocalQualitySelectionError("current selection SHA-256 mismatch")
    selection = _json(payload, label="current local quality selection")
    core = dict(selection)
    selected_id = core.pop("selection_id", None)
    computed_id = hashlib.sha256(_canonical_json(core)).hexdigest()
    if selected_id != SELECTION_ID or computed_id != SELECTION_ID:
        raise CurrentLocalQualitySelectionError("current selection identity mismatch")
    expected_top = {
        "schema_version",
        "status",
        "as_of_date",
        "market",
        "laptop_qualification",
        "model_runs",
        "historical_shape_diagnostic",
        "current_data_boundary",
        "current_eex_boundary",
        "claim_boundary",
        "future_holdout_consumed",
        "t057_consumed",
        "countable_prospective_origin",
        "training_eligible",
        "model_selection_eligible",
        "scientific_admission",
        "production_authorization",
        "promotion_gate",
        "selection_id",
    }
    if set(selection) != expected_top:
        raise CurrentLocalQualitySelectionError("current selection fields are not exact")
    if (
        selection.get("schema_version") != "ch_lt_local_quality_current_selection.v1"
        or selection.get("status") != "CURRENT_LOCAL_ENGINEERING_QUALITY_SELECTION_NO_GO"
        or selection.get("market") != "CH"
    ):
        raise CurrentLocalQualitySelectionError("current selection classification mismatch")
    for field in (
        "future_holdout_consumed",
        "t057_consumed",
        "countable_prospective_origin",
        "training_eligible",
        "model_selection_eligible",
        "scientific_admission",
        "production_authorization",
        "promotion_gate",
    ):
        if selection.get(field) is not False:
            raise CurrentLocalQualitySelectionError(f"current selection {field} must be false")

    qualification = _mapping(
        selection.get("laptop_qualification"), label="laptop qualification"
    )
    quality, _ = _read_bound(
        root,
        qualification.get("quality_json_path"),
        qualification.get("quality_json_sha256"),
        label="laptop quality JSON",
    )
    terminal, _ = _read_bound(
        root,
        qualification.get("terminal_receipt_path"),
        qualification.get("terminal_receipt_sha256"),
        label="laptop qualification terminal receipt",
    )
    audit_receipt, _ = _read_bound(
        root,
        qualification.get("audit_execution_receipt_path"),
        qualification.get("audit_execution_receipt_sha256"),
        label="laptop qualification execution receipt",
    )
    audit_supervisor, _ = _read_bound(
        root,
        qualification.get("audit_supervisor_receipt_path"),
        qualification.get("audit_supervisor_receipt_sha256"),
        label="laptop qualification supervisor receipt",
    )
    if (
        qualification.get("qualification_id") != quality.get("qualification_id")
        or terminal.get("qualification_id") != quality.get("qualification_id")
        or quality.get("status") != "PASS_LOCAL_STRUCTURAL_REPRODUCIBILITY_ONLY_NO_GO"
        or terminal.get("status") != quality.get("status")
        or terminal.get("terminal_receipt_last_written") is not True
        or audit_receipt.get("status") != "TARGET_EXIT_ZERO_NOT_AUTHORITY"
        or audit_receipt.get("target_exit_code") != 0
    ):
        raise CurrentLocalQualitySelectionError("laptop qualification chain mismatch")
    quality_record = _mapping(terminal.get("quality_json"), label="terminal quality JSON")
    markdown_record = _mapping(
        terminal.get("quality_markdown"), label="terminal quality Markdown"
    )
    if (
        quality_record.get("sha256") != qualification.get("quality_json_sha256")
        or markdown_record.get("sha256") != qualification.get("quality_markdown_sha256")
    ):
        raise CurrentLocalQualitySelectionError("terminal receipt output binding mismatch")
    _read_bound(
        root,
        qualification.get("quality_markdown_path"),
        qualification.get("quality_markdown_sha256"),
        label="laptop quality Markdown",
        parse_json=False,
    )
    for document, label in ((quality, "quality"), (terminal, "terminal")):
        for field in ("production_authorization", "promotion_gate", "scientific_admission"):
            if document.get(field) is not False:
                raise CurrentLocalQualitySelectionError(f"{label} {field} must be false")
    for field in (
        "runtime_authority",
        "evaluation_authorization",
        "scientific_authorization",
        "production_authorization",
        "promotion_authorization",
    ):
        if audit_receipt.get(field) is not False:
            raise CurrentLocalQualitySelectionError(
                f"qualification execution receipt {field} must be false"
            )
    audit_supervisor_execution = _mapping(
        audit_supervisor.get("execution_receipt"),
        label="qualification supervisor execution receipt",
    )
    audit_supervisor_resources = _mapping(
        audit_supervisor.get("resource_usage"),
        label="qualification supervisor resources",
    )
    audit_supervisor_deadline = _mapping(
        audit_supervisor.get("deadline"),
        label="qualification supervisor deadline",
    )
    audit_worker_capability = _mapping(
        audit_supervisor.get("worker_capability"),
        label="qualification worker capability",
    )
    audit_worker_admission = _mapping(
        audit_supervisor.get("worker_admission"),
        label="qualification worker admission",
    )
    audit_process_tree = _mapping(
        audit_supervisor.get("process_tree"),
        label="qualification supervisor process tree",
    )
    if (
        audit_supervisor.get("status") != "WORKER_EXIT_ZERO_NOT_AUTHORITY"
        or audit_supervisor.get("worker_exit_code") != 0
        or audit_supervisor.get("runner_exit_code") != 0
        or audit_supervisor_execution.get("path")
        != str(root / qualification.get("audit_execution_receipt_path"))
        or audit_supervisor_execution.get("sha256")
        != qualification.get("audit_execution_receipt_sha256")
        or audit_supervisor_execution.get("status") != audit_receipt.get("status")
        or audit_supervisor_resources.get("active_processes") != 0
        or audit_supervisor.get("terminal_receipt_write_attempts") != 1
        or audit_supervisor_deadline.get("deadline_exceeded") is not False
        or float(
            audit_supervisor_deadline.get(
                "overshoot_before_terminal_write_seconds", float("inf")
            )
        )
        != 0.0
        or audit_worker_capability.get("one_shot") is not True
        or audit_worker_capability.get("consumed") is not True
        or audit_worker_capability.get("absent_after_worker") is not True
        or audit_worker_capability.get("revoked_after_worker") is not False
        or audit_worker_admission.get("status")
        != "CAPABILITY_CONSUMED_NO_AUTHORITY"
        or audit_process_tree.get("kill_on_supervisor_exit") is not True
        or audit_process_tree.get("mode")
        != "CREATE_SUSPENDED_ASSIGN_KILL_ON_JOB_CLOSE_THEN_RESUME"
    ):
        raise CurrentLocalQualitySelectionError(
            "laptop qualification supervisor binding mismatch"
        )
    for field in (
        "runtime_authority",
        "evaluation_authorization",
        "scientific_authorization",
        "production_authorization",
        "promotion_authorization",
    ):
        if audit_supervisor.get(field) is not False:
            raise CurrentLocalQualitySelectionError(
                f"qualification supervisor {field} must be false"
            )

    structural = _mapping(quality.get("structural_quality"), label="structural quality")
    deterministic = _mapping(
        quality.get("deterministic_pair"), label="deterministic pair"
    )
    deterministic_runs = _mapping(
        deterministic.get("runs"), label="deterministic runs"
    )
    final_replay = _mapping(
        structural.get("final_product_replay"), label="final product replay"
    )
    quote_counts = _mapping(
        final_replay.get("source_quote_counts"), label="final product quote counts"
    )
    if (
        float(
            structural.get(
                "maximum_hourly_intraday_price_neutrality_residual_eur_mwh",
                float("inf"),
            )
        )
        > 1e-12
        or deterministic.get("path_normalized_quality_summary_equal") is not True
        or float(
            structural.get(
                "maximum_final_price_counterfactual_hourly_mean_residual_eur_mwh",
                float("inf"),
            )
        )
        > 1e-12
        or float(
            structural.get(
                "maximum_final_no_q_parent_hour_range_eur_mwh", float("inf")
            )
        )
        > 1e-12
        or float(
            structural.get(
                "maximum_final_price_decomposition_residual_eur_mwh", float("inf")
            )
        )
        > 1e-12
        or final_replay.get("status")
        != "PASS_LOCAL_FINAL_PRODUCT_REPLAY_PRODUCTION_NO_GO"
        or final_replay.get("cal2030_base_hours") != 8760
        or float(
            final_replay.get(
                "maximum_supported_abs_residual_eur_mwh", float("inf")
            )
        )
        > 1e-9
        or quote_counts != {"BASE": 20, "OFFPEAK": 0, "PEAK": 20}
        or final_replay.get("source_quote_conflict_count") != 18
        or final_replay.get("production_authorization") is not False
        or len(deterministic.get("byte_identical_material_roles", [])) != 9
        or _mapping(deterministic_runs.get("A"), label="deterministic run A").get(
            "workspace_root_sys_path_count"
        )
        != 1
        or _mapping(deterministic_runs.get("B"), label="deterministic run B").get(
            "workspace_root_sys_path_count"
        )
        != 1
    ):
        raise CurrentLocalQualitySelectionError(
            "current structural reproducibility evidence is incomplete"
        )

    runs = selection.get("model_runs")
    if not isinstance(runs, list) or [item.get("role") for item in runs] != ["A", "B"]:
        raise CurrentLocalQualitySelectionError("model run roles are not exact")
    for raw in runs:
        run = _mapping(raw, label="model run")
        completion, _ = _read_bound(
            root,
            run.get("completion_manifest_path"),
            run.get("completion_manifest_sha256"),
            label=f"run {run.get('role')} completion",
        )
        receipt, _ = _read_bound(
            root,
            run.get("execution_receipt_path"),
            run.get("execution_receipt_sha256"),
            label=f"run {run.get('role')} execution receipt",
        )
        supervisor, _ = _read_bound(
            root,
            run.get("supervisor_receipt_path"),
            run.get("supervisor_receipt_sha256"),
            label=f"run {run.get('role')} supervisor receipt",
        )
        arguments = _mapping(completion.get("arguments"), label="model arguments")
        redacted_command = receipt.get("redacted_command")
        supervisor_execution = _mapping(
            supervisor.get("execution_receipt"), label="supervisor execution receipt"
        )
        supervisor_resource = _mapping(
            supervisor.get("resource_usage"), label="supervisor resource usage"
        )
        supervisor_deadline = _mapping(
            supervisor.get("deadline"), label="supervisor deadline"
        )
        worker_capability = _mapping(
            supervisor.get("worker_capability"), label="worker capability"
        )
        worker_admission = _mapping(
            supervisor.get("worker_admission"), label="worker admission"
        )
        if (
            completion.get("status") != "COMPLETE_LOCAL_TEST_NO_GO"
            or completion.get("production_authorization") is not False
            or completion.get("promotion_gate") is not False
            or receipt.get("status") != "TARGET_EXIT_ZERO_NOT_AUTHORITY"
            or receipt.get("target_exit_code") != 0
            or arguments.get("enable_experimental_sparse_intraday_regularization")
            is not True
            or not isinstance(redacted_command, list)
            or redacted_command.count(
                "--enable-experimental-sparse-intraday-regularization"
            )
            != 1
            or supervisor.get("status") != "WORKER_EXIT_ZERO_NOT_AUTHORITY"
            or supervisor.get("worker_exit_code") != 0
            or supervisor.get("runner_exit_code") != 0
            or supervisor.get("terminal_receipt_write_attempts") != 1
            or supervisor_execution.get("sha256")
            != run.get("execution_receipt_sha256")
            or supervisor_execution.get("status") != receipt.get("status")
            or supervisor_resource.get("active_processes") != 0
            or supervisor_deadline.get("deadline_exceeded") is not False
            or worker_capability.get("one_shot") is not True
            or worker_capability.get("consumed") is not True
            or worker_capability.get("absent_after_worker") is not True
            or worker_admission.get("status") != "CAPABILITY_CONSUMED_NO_AUTHORITY"
        ):
            raise CurrentLocalQualitySelectionError("selected model run is not local NO_GO")
        for field in (
            "runtime_authority",
            "evaluation_authorization",
            "scientific_authorization",
            "production_authorization",
            "promotion_authorization",
        ):
            if supervisor.get(field) is not False:
                raise CurrentLocalQualitySelectionError(
                    f"model supervisor {field} must be false"
                )

    claims = _mapping(selection.get("claim_boundary"), label="claim boundary")
    proven = claims.get("proven")
    required_proven = {
        "EXPERIMENTAL_SPARSE_INTRADAY_REGULARIZATION_IS_EXPLICITLY_OPTED_IN",
        "COMPLETE_LOCAL_MONTH_DELIVERY_WINDOW_AUG2026_THROUGH_DEC2030",
        "CAL2030_CONTAINS_EXACTLY_8760_LOCAL_HOURS",
        "FINAL_QUOTE_AWARE_BASE_PEAK_AND_IMPLIED_OFFPEAK_REPLAY_PASSES_WITHIN_1E_9_EUR_MWH",
        "FINAL_PRODUCT_PROJECTION_INTRODUCES_NO_NON_FQ_INTRAHOUR_STRUCTURE_WITHIN_1E_12_EUR_MWH",
        "INDIVIDUAL_FINAL_REPLAY_RESIDUALS_RETAIN_SUB_1E_10_PRECISION",
    }
    not_proven = claims.get("not_proven")
    if (
        not isinstance(proven, list)
        or not required_proven.issubset(proven)
        or not isinstance(not_proven, list)
        or "SOURCE_QUOTE_CONFLICT_POLICY_PRODUCTION_APPROVAL" not in not_proven
        or "PRODUCTION_READINESS_OR_PROMOTION" not in not_proven
    ):
        raise CurrentLocalQualitySelectionError("selection claim boundary is incomplete")

    historical = _mapping(
        selection.get("historical_shape_diagnostic"), label="historical diagnostic"
    )
    historical_report, _ = _read_bound(
        root,
        historical.get("path"),
        historical.get("sha256"),
        label="historical local diagnostic",
    )
    if (
        historical_report.get("status") != "LOCAL_STRUCTURAL_DIAGNOSTIC_ONLY_NO_GO"
        or historical.get("target_is_full_ch_lt") is not False
        or historical.get("candidate_accepted") is not False
        or historical.get("model_selection_eligible") is not False
    ):
        raise CurrentLocalQualitySelectionError("historical diagnostic boundary mismatch")

    data = _mapping(selection.get("current_data_boundary"), label="data boundary")
    ledger_selection, _ = _read_bound(
        root,
        data.get("ledger_selection_path"),
        data.get("ledger_selection_sha256"),
        label="current CH ledger selection",
    )
    current_registry, _ = _read_bound(
        root,
        data.get("current_evidence_registry_path"),
        data.get("current_evidence_registry_sha256"),
        label="current CH evidence registry",
    )
    ledger_selected = _mapping(
        ledger_selection.get("selected"), label="selected CH prospective ledger"
    )
    ledger_forbidden = ledger_selection.get("forbidden_claims")
    registry_current = _mapping(
        current_registry.get("current_prospective_capture_selection"),
        label="registry current prospective selection",
    )
    if (
        ledger_selection.get("schema_version")
        != "ch_lt_prospective_capture_ledger_selection.v3"
        or ledger_selection.get("status")
        != "SELECTED_LOCAL_DIAGNOSTIC_ONLY_NOT_PIT_AUTHORITY"
        or data.get("independent_rolling_origin_count") != 0
        or data.get("price_shaping_or_economic_inference") != "FORBIDDEN"
        or ledger_selection.get("independent_rolling_origin_count") != 0
        or ledger_selection.get("future_holdout_consumed") is not False
        or ledger_selection.get("t057_consumed") is not False
        or ledger_selection.get("promotion_gate") is not False
        or ledger_selection.get("production_authorization") is not False
        or ledger_selected.get("proxy_rows_add_independent_information") is not False
        or ledger_selected.get("scientific_admission") is not False
        or ledger_selected.get("production_authorization") is not False
        or not isinstance(ledger_forbidden, list)
        or "INDEPENDENT_ROLLING_ORIGIN_OR_HOLDOUT_EVIDENCE"
        not in ledger_forbidden
        or "TRAINING_MODEL_SELECTION_OR_CANDIDATE_AUTHORITY"
        not in ledger_forbidden
        or current_registry.get("schema_version") != "ch_lt_current_evidence_selection.v2"
        or current_registry.get("status")
        != "CURRENT_LOCAL_DIAGNOSTIC_SELECTION_ONLY_NOT_AUTHORITY"
        or current_registry.get("independent_rolling_origin_count") != 0
        or current_registry.get("future_holdout_consumed") is not False
        or current_registry.get("t057_consumed") is not False
        or current_registry.get("scientific_admission") is not False
        or current_registry.get("candidate_assembly_eligible") is not False
        or current_registry.get("publication_authorized") is not False
        or current_registry.get("promotion_gate") is not False
        or current_registry.get("production_authorization") is not False
        or registry_current.get("path") != data.get("ledger_selection_path")
        or registry_current.get("sha256") != data.get("ledger_selection_sha256")
        or registry_current.get("scientific_admission") is not False
        or registry_current.get("production_authorization") is not False
    ):
        raise CurrentLocalQualitySelectionError("current data authority boundary mismatch")

    eex = _mapping(selection.get("current_eex_boundary"), label="EEX boundary")
    eex_selection, _ = _read_bound(
        root,
        eex.get("selection_path"),
        eex.get("selection_sha256"),
        label="current CH EEX selection",
    )
    if (
        eex_selection.get("schema_version")
        != "ch_eex_current_local_capture_selection.v1"
        or eex_selection.get("status")
        != "CURRENT_LOCAL_QUARANTINE_SELECTION_NOT_PIT_AUTHORITY"
        or eex.get("hard_quote_eligible") is not False
        or eex.get("monthly_solver_authority") is not False
        or eex.get("scientific_admission") is not False
        or eex_selection.get("training_eligible") is not False
        or eex_selection.get("model_selection_eligible") is not False
        or eex_selection.get("t057_consumed") is not False
        or eex_selection.get("scientific_admission") is not False
        or eex_selection.get("candidate_assembly_eligible") is not False
        or eex_selection.get("publication_authorized") is not False
        or eex_selection.get("promotion_gate") is not False
        or eex_selection.get("production_authorization") is not False
        or eex_selection.get("monthly_level_authority") != "solver"
        or eex_selection.get("monthly_solver_hard_level_eligible") is not False
        or eex_selection.get("ompex_used") is not False
        or eex_selection.get("quote_values_disclosed") is not False
    ):
        raise CurrentLocalQualitySelectionError("current EEX authority boundary mismatch")

    return {
        "schema_version": "ch_lt_local_quality_current_selection_audit.v1",
        "status": STATUS,
        "selection_path": SELECTION_RELATIVE.as_posix(),
        "selection_sha256": SELECTION_SHA256,
        "selection_id": SELECTION_ID,
        "qualification_id": qualification.get("qualification_id"),
        "model_run_count": 2,
        "independent_rolling_origin_count": 0,
        "future_holdout_consumed": False,
        "t057_consumed": False,
        "training_eligible": False,
        "model_selection_eligible": False,
        "scientific_admission": False,
        "production_authorization": False,
        "promotion_gate": False,
    }


def main() -> int:
    result = verify_current_selection(ROOT)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
