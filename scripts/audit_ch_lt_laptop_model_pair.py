"""Audit the frozen CH LT laptop A/B smoke pair and publish local-only evidence.

This audit never opens T057 or future truth.  It proves deterministic local
execution and structural invariants only; it is not model selection, scientific
admission, production authorization, or promotion evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json

ROOT = Path(r"C:\Users\jbattaglia\PFC_LT")
OUTPUT_RELATIVE = Path(
    "build/local-model-qualification/ch-lt-laptop-model-pair-v5"
)
QUALITY_RELATIVE = Path(
    "output/phase14/ch_lt_local_candidate_quality_20260727_v3/quality.json"
)
QUALITY_SHA256 = "bbba2480cdc8ae7d08c6722f7d4a37680da40055f08aaacf2a83a48b6dab9d55"
MODEL_SOURCE_RELATIVE = Path("scripts/build_local_test_ch_pfc.py")
MODEL_SOURCE_SHA256 = "1cc746efaa286d8948fe5c3cb16a8b1bb6210656eafb6fffc9d1afb6d74aac30"
PROTECTED_FORWARD_SHA256 = (
    "21ba73e70b6a16e88ba4c7d21985eafbdbc8efa2641ebe5d97c74b33f64e4013"
)
EXPECTED_SOURCE_TREE_SHA256 = (
    "5ee25178ffa27028f0b532c7c75f12df64bda6001a528b908ba4f24ceaa5db01"
)
EXPECTED_INTERPRETER_SHA256 = (
    "50bfb90ee93bb0cb51175b546f133798dfe4b778677d95d81391e7bf6d85e5ac"
)


@dataclass(frozen=True)
class RunSpec:
    name: str
    run_id: str
    supervisor_receipt_relative: Path
    supervisor_receipt_sha256: str
    execution_receipt_relative: Path
    execution_receipt_sha256: str
    run_root_relative: Path
    completion_relative: Path
    completion_sha256: str


RUNS = (
    RunSpec(
        name="A",
        run_id="mdl31n1c",
        supervisor_receipt_relative=Path(
            "build/workspace-local-supervisors/mdl31n1c/supervisor-receipt.json"
        ),
        supervisor_receipt_sha256=(
            "1d37bd1191e5fcbfb435b663c1690e98e2ac84f407d268d74dc16aa8a31941c8"
        ),
        execution_receipt_relative=Path(
            "build/workspace-local-runs/mdl31n1c/execution-receipt.json"
        ),
        execution_receipt_sha256=(
            "409132c223419423d6940bbde8b474a73e36be1e324b40827a4595542ca11a80"
        ),
        run_root_relative=Path("build/local-model-runs/run31n1c"),
        completion_relative=Path("build/local-model-runs/run31n1c/completion_manifest.json"),
        completion_sha256=(
            "fbc8278e9ae51be2d2bb54387b939540c7da97577025026440d4eebaecf0202c"
        ),
    ),
    RunSpec(
        name="B",
        run_id="mdl31n2",
        supervisor_receipt_relative=Path(
            "build/workspace-local-supervisors/mdl31n2/supervisor-receipt.json"
        ),
        supervisor_receipt_sha256=(
            "b89f9cc3209667bc7341e1f1c57e5f7321d9507e5f7b8000308c213b7ebcee1b"
        ),
        execution_receipt_relative=Path(
            "build/workspace-local-runs/mdl31n2/execution-receipt.json"
        ),
        execution_receipt_sha256=(
            "ed68eaba3bf1b7607e298f653cf8a7d71dfdf4628fd5408786989c44e2ce0b84"
        ),
        run_root_relative=Path("build/local-model-runs/run31n2"),
        completion_relative=Path("build/local-model-runs/run31n2/completion_manifest.json"),
        completion_sha256=(
            "2404fc3ad59442b414ab90f951425e9d70d9b09cdb95e4603f620f2620674c04"
        ),
    ),
)

STATUS = "PASS_LOCAL_STRUCTURAL_REPRODUCIBILITY_ONLY_NO_GO"
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_EXPECTED_SUPERVISOR_WALL_BUDGET_SECONDS = 900.0
_OUTPUT_ROLES = {
    "expanded_scenarios",
    "final_product_replay",
    "governance_report",
    "monthly_curve_manifest",
    "pfc_central",
    "pfc_fast",
    "pfc_slow",
    "quality_summary",
    "scenario_features",
    "structural_fan_chart",
}
_BYTE_IDENTICAL_ROLES = _OUTPUT_ROLES - {"quality_summary"}
_PFC_ROLES = ("pfc_slow", "pfc_central", "pfc_fast")
_INTRADAY_PRICE_NEUTRALITY_TOLERANCE_EUR_MWH = 1e-12


class LaptopModelPairError(ValueError):
    """Raised when the frozen local qualification pair is not exact."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _stable(path: Path, *, label: str, max_bytes: int) -> bytes:
    selected = assert_absolute_path_has_no_links(path)
    return read_stable_single_link_file(selected, label=label, max_bytes=max_bytes)


def _json(payload: bytes, *, label: str) -> Mapping[str, object]:
    try:
        value = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise LaptopModelPairError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise LaptopModelPairError(f"{label} must be a mapping")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise LaptopModelPairError(f"{label} must be a mapping")
    return value


def _sequence(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise LaptopModelPairError(f"{label} must be a list")
    return value


def _inside(path: Path, boundary: Path) -> bool:
    try:
        return os.path.commonpath(
            (os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(boundary)))
        ) == os.path.normcase(os.path.abspath(boundary))
    except ValueError:
        return False


def _read_exact_json(
    root: Path, relative: Path, expected_sha256: str, *, label: str
) -> tuple[Mapping[str, object], bytes]:
    path = root / relative
    payload = _stable(path, label=label, max_bytes=_MAX_JSON_BYTES)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise LaptopModelPairError(f"{label} SHA-256 mismatch")
    return _json(payload, label=label), payload


def _audit_execution_receipt(
    root: Path, spec: RunSpec
) -> tuple[Mapping[str, object], dict[str, object]]:
    supervisor, _ = _read_exact_json(
        root,
        spec.supervisor_receipt_relative,
        spec.supervisor_receipt_sha256,
        label=f"run {spec.name} supervisor receipt",
    )
    supervisor_expected = {
        "schema_version": "pfc_lt_workspace_local_supervisor.v1",
        "status": "WORKER_EXIT_ZERO_NOT_AUTHORITY",
        "workspace": str(root),
        "run_id": spec.run_id,
            "global_wall_timeout_seconds": _EXPECTED_SUPERVISOR_WALL_BUDGET_SECONDS,
        "worker_exit_code": 0,
        "runner_exit_code": 0,
        "runtime_authority": False,
        "evaluation_authorization": False,
        "scientific_authorization": False,
        "production_authorization": False,
        "promotion_authorization": False,
    }
    for field, value in supervisor_expected.items():
        if supervisor.get(field) != value:
            raise LaptopModelPairError(f"run {spec.name} supervisor {field} mismatch")
    deadline = _mapping(supervisor.get("deadline"), label="supervisor deadline")
    if (
        deadline.get("budget_seconds") != _EXPECTED_SUPERVISOR_WALL_BUDGET_SECONDS
        or deadline.get("deadline_exceeded") is not False
        or float(deadline.get("overshoot_before_terminal_write_seconds", math.inf)) != 0.0
    ):
        raise LaptopModelPairError(f"run {spec.name} supervisor deadline failed")
    capability = _mapping(
        supervisor.get("worker_capability"), label="supervisor worker capability"
    )
    if (
        capability.get("one_shot") is not True
        or capability.get("consumed") is not True
        or capability.get("absent_after_worker") is not True
        or capability.get("revoked_after_worker") is not False
    ):
        raise LaptopModelPairError(f"run {spec.name} supervisor capability failed")
    process_tree = _mapping(supervisor.get("process_tree"), label="supervisor process tree")
    if (
        process_tree.get("kill_on_supervisor_exit") is not True
        or process_tree.get("mode")
        != "CREATE_SUSPENDED_ASSIGN_KILL_ON_JOB_CLOSE_THEN_RESUME"
    ):
        raise LaptopModelPairError(f"run {spec.name} supervisor process tree failed")
    supervisor_resources = _mapping(
        supervisor.get("resource_usage"), label="supervisor resource usage"
    )
    if (
        supervisor_resources.get("available") is not True
        or supervisor_resources.get("active_processes") != 0
    ):
        raise LaptopModelPairError(f"run {spec.name} supervisor accounting failed")
    bound_execution = _mapping(
        supervisor.get("execution_receipt"), label="supervisor execution receipt"
    )
    if (
        bound_execution.get("path") != str(root / spec.execution_receipt_relative)
        or bound_execution.get("sha256") != spec.execution_receipt_sha256
        or bound_execution.get("status") != "TARGET_EXIT_ZERO_NOT_AUTHORITY"
    ):
        raise LaptopModelPairError(f"run {spec.name} supervisor execution binding failed")

    receipt, _ = _read_exact_json(
        root,
        spec.execution_receipt_relative,
        spec.execution_receipt_sha256,
        label=f"run {spec.name} execution receipt",
    )
    expected = {
        "schema_version": "pfc_lt_workspace_local_execution.v6",
        "status": "TARGET_EXIT_ZERO_NOT_AUTHORITY",
        "target_exit_code": 0,
        "runner_exit_code": 0,
        "workspace": str(root),
        "runtime_authority": False,
        "evaluation_authorization": False,
        "scientific_authorization": False,
        "production_authorization": False,
        "promotion_authorization": False,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise LaptopModelPairError(f"run {spec.name} receipt {field} mismatch")
    supervision = _mapping(receipt.get("supervision"), label="supervision")
    if (
        supervision.get("timed_out") is not False
        or supervision.get("interrupted") is not False
        or supervision.get("tree_termination_confirmed") is not True
        or supervision.get("descendant_cleanup_triggered") is not False
    ):
        raise LaptopModelPairError(f"run {spec.name} supervision is not terminal-clean")
    resources = _mapping(receipt.get("resource_usage"), label="resource usage")
    if resources.get("available") is not True or resources.get("active_processes") != 0:
        raise LaptopModelPairError(f"run {spec.name} process accounting is incomplete")
    target_output = _mapping(receipt.get("target_output"), label="target output")
    if target_output.get("complete") is not True:
        raise LaptopModelPairError(f"run {spec.name} target output capture is incomplete")

    identity = _mapping(receipt.get("execution_identity"), label="execution identity")
    source_tree = _mapping(identity.get("source_tree"), label="source tree")
    if source_tree.get("tree_sha256") != EXPECTED_SOURCE_TREE_SHA256:
        raise LaptopModelPairError(f"run {spec.name} source tree mismatch")
    interpreter = _mapping(identity.get("target_interpreter"), label="interpreter")
    if (
        interpreter.get("sha256") != EXPECTED_INTERPRETER_SHA256
        or interpreter.get("path") != str(root / "build/pytest-runtime-v1/python.exe")
        or interpreter.get("link_count") != 1
    ):
        raise LaptopModelPairError(f"run {spec.name} interpreter identity mismatch")
    closure = _mapping(identity.get("target_import_closure"), label="import closure")
    if closure.get("status") != "BOUND_REPO_LOCAL_PTH":
        raise LaptopModelPairError(f"run {spec.name} import closure is not bound")
    sys_path = [str(item) for item in _sequence(closure.get("sys_path"), label="sys.path")]
    root_key = os.path.normcase(os.path.abspath(root))
    root_count = sum(os.path.normcase(os.path.abspath(item)) == root_key for item in sys_path)
    if root_count != 1 or any(not _inside(Path(item), root) for item in sys_path):
        raise LaptopModelPairError(
            f"run {spec.name} sys.path must contain exactly one workspace root and no external root"
        )
    module_origins = _sequence(closure.get("module_origins"), label="module origins")
    model_origins = [
        _mapping(item, label="model module origin")
        for item in module_origins
        if isinstance(item, Mapping)
        and item.get("module") == "scripts.build_local_test_ch_pfc"
    ]
    if len(model_origins) != 1 or (
        model_origins[0].get("sha256") != MODEL_SOURCE_SHA256
        or model_origins[0].get("path") != str(root / MODEL_SOURCE_RELATIVE)
        or model_origins[0].get("link_count") != 1
    ):
        raise LaptopModelPairError(f"run {spec.name} model module origin mismatch")
    return receipt, {
        "execution_receipt_path": spec.execution_receipt_relative.as_posix(),
        "execution_receipt_sha256": spec.execution_receipt_sha256,
        "supervisor_receipt_path": spec.supervisor_receipt_relative.as_posix(),
        "supervisor_receipt_sha256": spec.supervisor_receipt_sha256,
        "source_tree_sha256": source_tree["tree_sha256"],
        "model_source_sha256": model_origins[0]["sha256"],
        "interpreter_sha256": interpreter["sha256"],
        "import_closure_status": closure["status"],
        "sys_path": sys_path,
        "workspace_root_sys_path_count": root_count,
        "wall_duration_seconds": float(resources["wall_duration_seconds"]),
        "user_cpu_seconds": float(resources["user_cpu_seconds"]),
        "kernel_cpu_seconds": float(resources["kernel_cpu_seconds"]),
        "peak_job_memory_bytes": int(resources["peak_job_memory_bytes"]),
        "read_transfer_bytes": int(resources["read_transfer_bytes"]),
        "write_transfer_bytes": int(resources["write_transfer_bytes"]),
        "active_processes_after_wait": int(resources["active_processes"]),
    }


def _resolve_repo_file(root: Path, raw_path: object, *, label: str) -> Path:
    selected = Path(str(raw_path))
    if not selected.is_absolute():
        selected = root / selected
    selected = Path(os.path.abspath(selected))
    if not _inside(selected, root):
        raise LaptopModelPairError(f"{label} escapes the canonical repository")
    return selected


def _audit_completion(
    root: Path, spec: RunSpec
) -> tuple[Mapping[str, object], dict[str, bytes]]:
    completion, _ = _read_exact_json(
        root,
        spec.completion_relative,
        spec.completion_sha256,
        label=f"run {spec.name} completion manifest",
    )
    expected = {
        "schema_version": "local_test_ch_pfc_completion.v1",
        "status": "COMPLETE_LOCAL_TEST_NO_GO",
        "production_authorization": False,
        "promotion_gate": False,
        "scientific_admission": False,
        "training_eligible": False,
        "model_selection_eligible": False,
        "confirmatory_rolling_origin_eligible": False,
        "future_holdout_eligible": False,
        "t057_eligible": False,
        "candidate_assembly_eligible": False,
        "local_diagnostic_allowed": True,
        "completion_manifest_last_written": True,
        "monthly_level_authority": "solver",
        "forward_source_kind": "TEST_FIXTURE",
        "hard_quote_eligible": False,
        "monthly_manifest_promotion_eligible": False,
        "code_execution_attestation": (
            "NOT_PROVIDED_SOURCE_HASHES_ARE_OBSERVED_AFTER_IMPORT_ONLY"
        ),
    }
    for field, value in expected.items():
        if completion.get(field) != value:
            raise LaptopModelPairError(f"run {spec.name} completion {field} mismatch")
    inputs = _mapping(completion.get("inputs"), label="completion inputs")
    for role, raw_record in inputs.items():
        record = _mapping(raw_record, label=f"input {role}")
        path = _resolve_repo_file(root, record.get("path"), label=f"input {role}")
        payload = _stable(path, label=f"input {role}", max_bytes=_MAX_ARTIFACT_BYTES)
        if (
            hashlib.sha256(payload).hexdigest() != record.get("sha256")
            or len(payload) != record.get("size_bytes")
        ):
            raise LaptopModelPairError(f"run {spec.name} input {role} changed")
    if (
        _mapping(inputs.get("forward_history"), label="forward history").get("sha256")
        != PROTECTED_FORWARD_SHA256
    ):
        raise LaptopModelPairError("protected forward history hash mismatch")

    outputs = _mapping(completion.get("outputs"), label="completion outputs")
    if set(outputs) != _OUTPUT_ROLES:
        raise LaptopModelPairError(f"run {spec.name} output roles are not exact")
    run_root = root / spec.run_root_relative
    payloads: dict[str, bytes] = {}
    for role, raw_record in outputs.items():
        record = _mapping(raw_record, label=f"output {role}")
        path = _resolve_repo_file(root, record.get("path"), label=f"output {role}")
        if not _inside(path, run_root):
            raise LaptopModelPairError(f"run {spec.name} output {role} escapes its run root")
        payload = _stable(path, label=f"output {role}", max_bytes=_MAX_ARTIFACT_BYTES)
        if (
            hashlib.sha256(payload).hexdigest() != record.get("sha256")
            or len(payload) != record.get("size_bytes")
        ):
            raise LaptopModelPairError(f"run {spec.name} output {role} changed")
        payloads[str(role)] = payload
    return completion, payloads


def _normalize_run_bound(value: object, run_root: Path) -> object:
    if isinstance(value, Mapping):
        return {str(key): _normalize_run_bound(item, run_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_run_bound(item, run_root) for item in value]
    if isinstance(value, str):
        normalized = value.replace(str(run_root), "<RUN_ROOT>")
        normalized = normalized.replace(run_root.as_posix(), "<RUN_ROOT>")
        return normalized
    return value


def _monthly_structural_metrics(
    monthly_payload: bytes, pfc_payloads: Mapping[str, bytes], fan_payload: bytes
) -> dict[str, object]:
    monthly = _json(monthly_payload, label="monthly curve manifest")
    expected = {
        "market": "CH",
        "monthly_level_authority": "solver",
        "forward_source_kind": "TEST_FIXTURE",
        "hard_quote_eligible": False,
        "promotion_eligible": False,
        "structural_status": "STRUCTURAL_TEMPLATE",
    }
    for field, value in expected.items():
        if monthly.get(field) != value:
            raise LaptopModelPairError(f"monthly manifest {field} mismatch")
    kkt = _mapping(monthly.get("solver_kkt"), label="solver KKT")
    constraint_residual = float(kkt.get("max_abs_constraint_residual", math.inf))
    stationarity = float(kkt.get("stationarity_residual", math.inf))
    if (
        not math.isfinite(constraint_residual)
        or constraint_residual > 1e-9
        or not math.isfinite(stationarity)
        or stationarity > 1e-9
        or kkt.get("ridge_used") is not False
        or kkt.get("solved_by_lstsq") is not False
    ):
        raise LaptopModelPairError("monthly solver deterministic gate failed")
    solution_raw = _mapping(monthly.get("monthly_solution"), label="monthly solution")
    solution = pd.Series({str(key): float(value) for key, value in solution_raw.items()})
    scenarios: dict[str, object] = {}
    maximum_monthly_residual = 0.0
    maximum_hourly_qh_neutrality = 0.0
    maximum_hourly_intraday_price_neutrality = 0.0
    maximum_p99_hourly_intraday_price_neutrality = 0.0
    maximum_final_price_decomposition_residual = 0.0
    maximum_final_no_q_parent_hour_range = 0.0
    maximum_final_price_counterfactual_mean_residual = 0.0
    for role in _PFC_ROLES:
        frame = pd.read_parquet(io.BytesIO(pfc_payloads[role]))
        index = pd.DatetimeIndex(frame.index)
        if (
            len(frame) != 154_948
            or index.tz is None
            or not index.is_monotonic_increasing
            or not index.is_unique
            or not np.all(np.diff(index.asi8) == 900 * 1_000_000_000)
        ):
            raise LaptopModelPairError(f"{role} timestamp axis is invalid")
        required = {
            "price_shape",
            "B",
            "f_S",
            "f_W",
            "f_H",
            "f_Q",
            "f_WV",
            "delta_wv",
            "price_pre_final_projection",
            "delta_final_product_projection",
            "f_bridge",
            "calibrated",
            "p10",
            "p90",
        }
        if not required.issubset(frame.columns):
            raise LaptopModelPairError(f"{role} columns are incomplete")
        if (
            not np.isfinite(frame[["price_shape", "B", "f_Q"]].to_numpy(dtype=float)).all()
            or not bool(frame["calibrated"].all())
            or frame["p10"].notna().any()
            or frame["p90"].notna().any()
        ):
            raise LaptopModelPairError(f"{role} numeric or probability boundary failed")
        local = index.tz_convert("Europe/Zurich")
        months = pd.Index(local.strftime("%Y-%m"))
        observed = frame["price_shape"].groupby(months).mean()
        base = frame["B"].groupby(months).mean()
        if not observed.index.equals(solution.index) or not base.index.equals(solution.index):
            raise LaptopModelPairError(f"{role} monthly index differs from solver")
        price_residual = float(np.max(np.abs(observed.to_numpy() - solution.to_numpy())))
        base_residual = float(np.max(np.abs(base.to_numpy() - solution.to_numpy())))
        maximum_monthly_residual = max(
            maximum_monthly_residual, price_residual, base_residual
        )
        hourly = frame["f_Q"].groupby(index.floor("h")).mean()
        qh_residual = float(np.max(np.abs(hourly.to_numpy(dtype=float) - 1.0)))
        maximum_hourly_qh_neutrality = max(maximum_hourly_qh_neutrality, qh_residual)
        multiplicative_without_q = (
            frame["B"]
            * frame["f_S"]
            * frame["f_W"]
            * frame["f_H"]
            * frame["f_WV"]
            * frame["f_bridge"]
        )
        counterfactual_without_q = multiplicative_without_q + frame["delta_wv"]
        counterfactual_with_q = (
            multiplicative_without_q * frame["f_Q"] + frame["delta_wv"]
        )
        final_counterfactual_with_q = (
            frame["price_pre_final_projection"]
            + frame["delta_final_product_projection"]
        )
        final_decomposition_residual = float(
            np.max(
                np.abs(
                    frame["price_shape"].to_numpy(dtype=float)
                    - final_counterfactual_with_q.to_numpy(dtype=float)
                )
            )
        )
        pre_projection_hour_mean = frame["price_pre_final_projection"].groupby(
            index.floor("h")
        ).transform("mean")
        final_counterfactual_without_q = (
            pre_projection_hour_mean + frame["delta_final_product_projection"]
        )
        final_no_q_parent_hour_range = float(
            (
                final_counterfactual_without_q.groupby(index.floor("h")).max()
                - final_counterfactual_without_q.groupby(index.floor("h")).min()
            ).max()
        )
        maximum_final_price_decomposition_residual = max(
            maximum_final_price_decomposition_residual,
            final_decomposition_residual,
        )
        maximum_final_no_q_parent_hour_range = max(
            maximum_final_no_q_parent_hour_range,
            final_no_q_parent_hour_range,
        )
        final_price_counterfactual_mean_residual = float(
            (
                frame["price_shape"].groupby(index.floor("h")).mean()
                - final_counterfactual_without_q.groupby(index.floor("h")).mean()
            ).abs().max()
        )
        maximum_final_price_counterfactual_mean_residual = max(
            maximum_final_price_counterfactual_mean_residual,
            final_price_counterfactual_mean_residual,
        )
        hourly_price_residual = (
            counterfactual_with_q.groupby(index.floor("h")).mean()
            - counterfactual_without_q.groupby(index.floor("h")).mean()
        ).abs()
        if not np.isfinite(hourly_price_residual.to_numpy(dtype=float)).all():
            raise LaptopModelPairError(
                f"{role} intraday price-neutrality counterfactual is non-finite"
            )
        price_neutrality_max = float(hourly_price_residual.max())
        price_neutrality_p99 = float(
            np.quantile(hourly_price_residual.to_numpy(dtype=float), 0.99)
        )
        maximum_hourly_intraday_price_neutrality = max(
            maximum_hourly_intraday_price_neutrality,
            price_neutrality_max,
        )
        maximum_p99_hourly_intraday_price_neutrality = max(
            maximum_p99_hourly_intraday_price_neutrality,
            price_neutrality_p99,
        )
        price = frame["price_shape"].to_numpy(dtype=float)
        scenarios[role.removeprefix("pfc_")] = {
            "rows": int(len(frame)),
            "start_utc": index[0].isoformat(),
            "end_utc": index[-1].isoformat(),
            "mean_eur_mwh": float(np.mean(price)),
            "min_eur_mwh": float(np.min(price)),
            "max_eur_mwh": float(np.max(price)),
            "p05_eur_mwh": float(np.quantile(price, 0.05)),
            "p95_eur_mwh": float(np.quantile(price, 0.95)),
            "monthly_price_residual_eur_mwh": price_residual,
            "monthly_base_residual_eur_mwh": base_residual,
            "hourly_intraday_price_neutrality_max_abs_eur_mwh": (
                price_neutrality_max
            ),
            "hourly_intraday_price_neutrality_p99_abs_eur_mwh": (
                price_neutrality_p99
            ),
            "final_price_decomposition_max_abs_eur_mwh": (
                final_decomposition_residual
            ),
            "final_no_q_parent_hour_range_max_eur_mwh": (
                final_no_q_parent_hour_range
            ),
            "final_price_counterfactual_hourly_mean_max_abs_eur_mwh": (
                final_price_counterfactual_mean_residual
            ),
            "p10_non_null_rows": 0,
            "p90_non_null_rows": 0,
        }
    if maximum_monthly_residual > 1e-9 or maximum_hourly_qh_neutrality > 1e-12:
        raise LaptopModelPairError("PFC monthly or quarter-hour neutrality gate failed")
    if (
        maximum_hourly_intraday_price_neutrality
        > _INTRADAY_PRICE_NEUTRALITY_TOLERANCE_EUR_MWH
    ):
        raise LaptopModelPairError(
            "PFC parent-hour intraday price-neutrality gate failed"
        )
    if (
        maximum_final_price_decomposition_residual > 1e-12
        or maximum_final_no_q_parent_hour_range > 1e-12
        or maximum_final_price_counterfactual_mean_residual > 1e-12
    ):
        raise LaptopModelPairError("PFC final-price counterfactual gate failed")
    fan = pd.read_parquet(io.BytesIO(fan_payload))
    if (
        len(fan) != 154_948
        or not np.isfinite(fan.to_numpy(dtype=float)).all()
        or (fan["structural_scenario_low"] > fan["structural_scenario_high"]).any()
    ):
        raise LaptopModelPairError("structural fan chart is invalid")
    return {
        "monthly_level_authority": "solver",
        "forward_source_kind": "TEST_FIXTURE",
        "hard_quote_eligible": False,
        "solver_max_abs_constraint_residual_eur_mwh": constraint_residual,
        "solver_stationarity_residual": stationarity,
        "solver_condition_number": float(kkt["condition_number"]),
        "maximum_monthly_mean_residual_eur_mwh": maximum_monthly_residual,
        "maximum_hourly_quarter_factor_neutrality_residual": (
            maximum_hourly_qh_neutrality
        ),
        "maximum_hourly_intraday_price_neutrality_residual_eur_mwh": (
            maximum_hourly_intraday_price_neutrality
        ),
        "maximum_p99_hourly_intraday_price_neutrality_residual_eur_mwh": (
            maximum_p99_hourly_intraday_price_neutrality
        ),
        "hourly_intraday_price_neutrality_tolerance_eur_mwh": (
            _INTRADAY_PRICE_NEUTRALITY_TOLERANCE_EUR_MWH
        ),
        "maximum_final_price_decomposition_residual_eur_mwh": (
            maximum_final_price_decomposition_residual
        ),
        "maximum_final_no_q_parent_hour_range_eur_mwh": (
            maximum_final_no_q_parent_hour_range
        ),
        "maximum_final_price_counterfactual_hourly_mean_residual_eur_mwh": (
            maximum_final_price_counterfactual_mean_residual
        ),
        "scenario_interpretation": "STRUCTURAL_PATHS_NOT_CALIBRATED_PROBABILITIES",
        "scenarios": scenarios,
        "fan_chart": {
            "rows": int(len(fan)),
            "weighted_mean_eur_mwh": float(fan["weighted_mean"].mean()),
            "structural_spread_mean_eur_mwh": float(
                fan["structural_scenario_spread"].mean()
            ),
            "structural_spread_max_eur_mwh": float(
                fan["structural_scenario_spread"].max()
            ),
        },
    }


def _final_product_replay_metrics(payload: bytes) -> dict[str, object]:
    replay = _json(payload, label="final product replay")
    if (
        replay.get("schema_version") != "local_test_ch_final_product_replay.v1"
        or replay.get("status")
        != "PASS_LOCAL_FINAL_PRODUCT_REPLAY_PRODUCTION_NO_GO"
        or replay.get("production_authorization") is not False
        or replay.get("promotion_gate") is not False
        or float(replay.get("tolerance_eur_mwh", math.inf)) != 1e-9
    ):
        raise LaptopModelPairError("final product replay boundary failed")
    counts = _mapping(replay.get("source_quote_counts"), label="source quote counts")
    if {key: int(counts.get(key, -1)) for key in ("BASE", "PEAK", "OFFPEAK")} != {
        "BASE": 20,
        "PEAK": 20,
        "OFFPEAK": 0,
    }:
        raise LaptopModelPairError("final product replay source quote counts changed")
    scenarios = _mapping(replay.get("scenarios"), label="final replay scenarios")
    maximum_supported_residual = 0.0
    total_conflicts = 0
    for scenario in ("slow", "central", "fast"):
        evidence = _mapping(scenarios.get(scenario), label=f"final replay {scenario}")
        residual = float(
            evidence.get("maximum_supported_abs_residual_eur_mwh", math.inf)
        )
        if (
            evidence.get("status")
            != "PASS_WITH_SOURCE_QUOTE_CONFLICTS_PRODUCTION_NO_GO"
            or int(evidence.get("cal2030_base_rows", -1)) != 8_760
            or not math.isfinite(residual)
            or residual > 1e-9
        ):
            raise LaptopModelPairError(f"final product replay failed for {scenario}")
        maximum_supported_residual = max(maximum_supported_residual, residual)
        total_conflicts += int(evidence.get("source_quote_conflict_count", -1))
    if total_conflicts != 18:
        raise LaptopModelPairError("final product replay conflict inventory changed")
    return {
        "status": replay["status"],
        "source_quote_counts": dict(counts),
        "maximum_supported_abs_residual_eur_mwh": maximum_supported_residual,
        "source_quote_conflict_count": total_conflicts,
        "cal2030_base_hours": 8_760,
        "production_authorization": False,
    }


def audit_pair(root: Path = ROOT) -> dict[str, object]:
    root = assert_absolute_path_has_no_links(root)
    if Path.cwd() != root:
        raise LaptopModelPairError("cwd must be the canonical repository root")
    model_payload = _stable(
        root / MODEL_SOURCE_RELATIVE,
        label="current model source",
        max_bytes=_MAX_JSON_BYTES,
    )
    if hashlib.sha256(model_payload).hexdigest() != MODEL_SOURCE_SHA256:
        raise LaptopModelPairError("current model source no longer matches executed bytes")

    receipts: dict[str, Mapping[str, object]] = {}
    receipt_summaries: dict[str, object] = {}
    completions: dict[str, Mapping[str, object]] = {}
    payloads: dict[str, dict[str, bytes]] = {}
    for spec in RUNS:
        receipt, receipt_summary = _audit_execution_receipt(root, spec)
        completion, output_payloads = _audit_completion(root, spec)
        receipts[spec.name] = receipt
        receipt_summaries[spec.name] = receipt_summary
        completions[spec.name] = completion
        payloads[spec.name] = output_payloads

    identity_a = _mapping(receipts["A"].get("execution_identity"), label="identity A")
    identity_b = _mapping(receipts["B"].get("execution_identity"), label="identity B")
    closure_a = _mapping(identity_a.get("target_import_closure"), label="closure A")
    closure_b = _mapping(identity_b.get("target_import_closure"), label="closure B")
    for field in ("sys_path", "import_roots", "launcher_tree", "python_pth", "module_origins"):
        if closure_a.get(field) != closure_b.get(field):
            raise LaptopModelPairError(f"A/B import closure {field} mismatch")
    if completions["A"].get("inputs") != completions["B"].get("inputs"):
        raise LaptopModelPairError("A/B input manifests differ")
    for role in sorted(_BYTE_IDENTICAL_ROLES):
        if payloads["A"][role] != payloads["B"][role]:
            raise LaptopModelPairError(f"A/B material output differs: {role}")
    run_root_a = root / RUNS[0].run_root_relative
    run_root_b = root / RUNS[1].run_root_relative
    quality_a = payloads["A"]["quality_summary"].decode("utf-8")
    quality_b = payloads["B"]["quality_summary"].decode("utf-8")
    if quality_a.replace(str(run_root_a), "<RUN_ROOT>") != quality_b.replace(
        str(run_root_b), "<RUN_ROOT>"
    ):
        raise LaptopModelPairError("A/B path-normalized quality summaries differ")
    arguments_a = _normalize_run_bound(completions["A"].get("arguments"), run_root_a)
    arguments_b = _normalize_run_bound(completions["B"].get("arguments"), run_root_b)
    if arguments_a != arguments_b:
        raise LaptopModelPairError("A/B normalized command arguments differ")

    structural = _monthly_structural_metrics(
        payloads["A"]["monthly_curve_manifest"],
        {role: payloads["A"][role] for role in _PFC_ROLES},
        payloads["A"]["structural_fan_chart"],
    )
    structural["final_product_replay"] = _final_product_replay_metrics(
        payloads["A"]["final_product_replay"]
    )
    historical, _ = _read_exact_json(
        root, QUALITY_RELATIVE, QUALITY_SHA256, label="historical local quality report"
    )
    if (
        historical.get("status") != "LOCAL_STRUCTURAL_DIAGNOSTIC_ONLY_NO_GO"
        or historical.get("production_authorization") is not False
        or historical.get("promotion_gate") is not False
        or historical.get("future_holdout_consumed") is not False
        or historical.get("t057_consumed") is not False
    ):
        raise LaptopModelPairError("historical quality boundary changed")
    rolling = _mapping(
        _mapping(historical.get("intraday"), label="historical intraday").get(
            "rolling_origin"
        ),
        label="historical rolling diagnostic",
    )
    report: dict[str, object] = {
        "schema_version": "ch_lt_laptop_model_qualification.v1",
        "status": STATUS,
        "market": "CH",
        "qualification_scope": "STANDARD_USER_LAPTOP_LOCAL_ENGINEERING_ONLY",
        "production_authorization": False,
        "promotion_gate": False,
        "scientific_admission": False,
        "training_eligible": False,
        "model_selection_eligible": False,
        "confirmatory_rolling_origin_eligible": False,
        "future_holdout_eligible": False,
        "countable_prospective_origin": False,
        "future_holdout_consumed": False,
        "t057_consumed": False,
        "deterministic_pair": {
            "byte_identical_material_roles": sorted(_BYTE_IDENTICAL_ROLES),
            "path_normalized_quality_summary_equal": True,
            "path_normalized_arguments_equal": True,
            "inputs_equal_and_rehashed": True,
            "source_tree_equal": True,
            "import_closure_equal": True,
            "runs": receipt_summaries,
        },
        "structural_quality": structural,
        "historical_non_ch_non_confirmatory_diagnostic": {
            "path": QUALITY_RELATIVE.as_posix(),
            "sha256": QUALITY_SHA256,
            "target": rolling.get("target"),
            "origin_count": rolling.get("origin_count"),
            "candidate_mae_eur_mwh": rolling.get("candidate_mae_eur_mwh"),
            "incumbent_mae_eur_mwh": rolling.get("incumbent_mae_eur_mwh"),
            "relative_mae_improvement": rolling.get("relative_mae_improvement"),
            "wins": rolling.get("wins"),
            "losses": rolling.get("losses"),
            "ties": rolling.get("ties"),
            "latest_fold_noninferior": rolling.get("latest_fold_noninferior"),
            "candidate_accepted": rolling.get("candidate_accepted"),
            "target_is_full_ch_lt": False,
            "usable_for_model_selection": False,
        },
        "limitations": [
            "Forward levels remain TEST_FIXTURE and are not governed hard CH EEX quotes.",
            "Intraday shaping remains a DE-LU mixed-authority proxy not validated for CH.",
            "Slow/central/fast are structural paths, not calibrated probabilities.",
            "No direct governed CH 15-minute truth or calibrated P10/P90 is present.",
            "The supervisor binds live checkout bytes; this is not installed-wheel model qualification.",
            "Observed resource use has no prefrozen capacity SLO and is descriptive only.",
            "No future truth, T057 outcome, external CAS, trusted time, or production authority was used.",
        ],
    }
    report["qualification_id"] = hashlib.sha256(_canonical_json(report)).hexdigest()
    return report


def _markdown(report: Mapping[str, object]) -> bytes:
    pair = _mapping(report["deterministic_pair"], label="pair")
    runs = _mapping(pair["runs"], label="runs")
    structural = _mapping(report["structural_quality"], label="structural")
    central = _mapping(
        _mapping(structural["scenarios"], label="scenarios")["central"], label="central"
    )
    historical = _mapping(
        report["historical_non_ch_non_confirmatory_diagnostic"], label="historical"
    )
    lines = [
        "# CH LT laptop model qualification v1",
        "",
        f"- Status: `{report['status']}`",
        "- Production/promotion/scientific admission: `false / false / false`",
        "- Scope: standard-user laptop structural engineering only",
        f"- Qualification ID: `{report['qualification_id']}`",
        "",
        "## Deterministic A/B execution",
        "",
    ]
    for name in ("A", "B"):
        run = _mapping(runs[name], label=f"run {name}")
        lines.append(
            f"- Run {name}: receipt `{run['execution_receipt_sha256']}`, "
            f"wall `{run['wall_duration_seconds']:.3f}s`, peak job memory "
            f"`{int(run['peak_job_memory_bytes']) / (1024**3):.3f} GiB`, "
            "active processes after wait `0`."
        )
    lines.extend(
        [
            f"- Byte-identical material roles: `{len(pair['byte_identical_material_roles'])}`.",
            "- Path-normalized QUALITY and arguments: identical.",
            "- Import closure: `BOUND_REPO_LOCAL_PTH`; workspace root appears exactly once.",
            "",
            "## Structural curve checks",
            "",
            f"- Central rows: `{central['rows']}`.",
            f"- Central mean/min/max: `{central['mean_eur_mwh']:.6f}` / "
            f"`{central['min_eur_mwh']:.6f}` / `{central['max_eur_mwh']:.6f}` EUR/MWh.",
            f"- Maximum monthly mean residual: "
            f"`{structural['maximum_monthly_mean_residual_eur_mwh']:.3e}` EUR/MWh.",
            f"- Solver max constraint residual: "
            f"`{structural['solver_max_abs_constraint_residual_eur_mwh']:.3e}` EUR/MWh.",
            f"- Maximum hourly quarter-factor neutrality residual: "
            f"`{structural['maximum_hourly_quarter_factor_neutrality_residual']:.3e}`.",
            "- P10/P90: absent; structural paths are not probabilities.",
            "",
            "## Existing historical local diagnostic",
            "",
            f"- Target: `{historical['target']}` (not full CH LT).",
            f"- Candidate/incumbent MAE: `{historical['candidate_mae_eur_mwh']:.6f}` / "
            f"`{historical['incumbent_mae_eur_mwh']:.6f}` EUR/MWh.",
            f"- Relative MAE improvement: `{100 * float(historical['relative_mae_improvement']):.3f}%`; "
            f"wins/losses/ties `{historical['wins']}/{historical['losses']}/{historical['ties']}`.",
            f"- Latest-fold noninferiority/candidate accepted: "
            f"`{historical['latest_fold_noninferior']}/{historical['candidate_accepted']}`.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    return ("\n".join(lines) + "\n").encode("utf-8")


def publish(output_directory: Path) -> dict[str, object]:
    expected = ROOT / OUTPUT_RELATIVE
    selected = Path(os.path.abspath(output_directory))
    if selected != expected:
        raise LaptopModelPairError("output directory is not the canonical qualification path")
    if selected.exists():
        raise LaptopModelPairError("qualification output directory already exists")
    report = audit_pair(ROOT)
    selected.mkdir(parents=True, exist_ok=False)
    json_payload = _canonical_json(report) + b"\n"
    markdown_payload = _markdown(report)
    json_path = selected / "quality.json"
    markdown_path = selected / "QUALITY.md"
    write_durable_exact_bytes(json_path, json_payload, label="laptop qualification JSON")
    write_durable_exact_bytes(
        markdown_path, markdown_payload, label="laptop qualification Markdown"
    )
    receipt: dict[str, object] = {
        "schema_version": "ch_lt_laptop_model_qualification_receipt.v1",
        "status": STATUS,
        "qualification_id": report["qualification_id"],
        "quality_json": {
            "path": json_path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(json_payload).hexdigest(),
            "size_bytes": len(json_payload),
        },
        "quality_markdown": {
            "path": markdown_path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(markdown_payload).hexdigest(),
            "size_bytes": len(markdown_payload),
        },
        "terminal_receipt_last_written": True,
        "production_authorization": False,
        "promotion_gate": False,
        "scientific_admission": False,
    }
    receipt["receipt_id"] = hashlib.sha256(_canonical_json(receipt)).hexdigest()
    receipt_payload = _canonical_json(receipt) + b"\n"
    write_durable_exact_bytes(
        selected / "qualification-receipt.json",
        receipt_payload,
        label="laptop qualification terminal receipt",
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = publish(args.output_directory)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
