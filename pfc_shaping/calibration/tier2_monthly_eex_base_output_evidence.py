"""Path-only verification of one deterministic Tier 2 BASE model output."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import hashlib
import importlib
from importlib import metadata
import inspect
import os
from pathlib import Path
import platform
import re
import sys
from types import CodeType

import pandas as pd

import pfc_shaping.calibration.constraints as constraints_module
import pfc_shaping.calibration.cascading as cascading_module
import pfc_shaping.calibration.monthly_curve_priors as priors_module
import pfc_shaping.calibration.monthly_forward_curve as monthly_solver_module
import pfc_shaping.calibration.tier2_monthly_eex_base_replay as base_replay_module
import pfc_shaping.calibration.tier2_monthly_eex_evaluation as evaluation_module
import pfc_shaping.calibration.tier2_monthly_eex_fold_evidence as fold_module
import pfc_shaping.data.acquisition_contract as acquisition_module
import pfc_shaping.data.databricks_client as databricks_module
import pfc_shaping.data.eex_historical_vintage as vintage_module
import pfc_shaping.data.ingest_forwards as ingest_forwards_module
import pfc_shaping.pipeline.model_governance_contract as governance_module
from pfc_shaping.calibration.tier2_monthly_eex_evaluation import (
    EXECUTION_ATTESTATION_SCHEMA,
    FULL_TIER2_STATUS,
    TIMESTAMP_PUBLIC_KEY_ENV,
    TIMESTAMP_RECEIPT_SCHEMA,
    Tier2MonthlyEexEvaluationError,
    _canonical_json_bytes,
    _load_json_artifact,
    _require_equal,
    _require_exact_keys,
    _sha256_json,
    _verify_exact_artifact_authority_bytes,
)
from pfc_shaping.calibration.tier2_monthly_eex_fold_evidence import (
    _absolute_regular_file,
    _catalog_internal_paths,
    _file_set_snapshot,
    _file_snapshot,
    _verify_signed_selection_fold_evidence_context,
)
from pfc_shaping.data.acquisition_contract import (
    TRUSTED_ACQUISITION_PUBLIC_KEY_ENV,
    TRUSTED_TIME_PUBLIC_KEY_ENV,
)
from pfc_shaping.pipeline.model_governance_contract import (
    MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV,
)


BASE_OUTPUT_MANIFEST_SCHEMA = "tier2_monthly_eex_base_output_manifest.v1"
SOURCE_CLOSURE_SCHEMA = "tier2_monthly_eex_source_closure.v1"
RUNTIME_LOCK_SCHEMA = "tier2_monthly_eex_runtime_lock.v1"
BASE_OUTPUT_MANIFEST_ROLE = "tier2_monthly_eex_base_output_manifest"
BASE_OUTPUT_SCOPE = "MONTHLY_EEX_BASE_MASKED_QUOTE_ONLY"
AWAITING_REPLAY_STATUS = "GENERATED_AWAITING_INDEPENDENT_REPLAY"
VERIFIED_REPLAY_STATUS = "MODEL_OUTPUT_VERIFIED_BASE_ONLY"
MODEL_EXECUTION_PUBLIC_KEY_ENV = (
    "PFC_TIER2_BASE_MODEL_EXECUTION_TRUSTED_PUBLIC_KEY_PATH"
)

_MANIFEST_FILENAME = "base_output_manifest.v1.json"
_OUTPUT_FILENAME = "expected_base_output.v1.json"
_RUNTIME_PACKAGES = (
    "cryptography",
    "holidays",
    "numpy",
    "openpyxl",
    "pandas",
    "pyarrow",
    "python-dateutil",
    "pytz",
    "scipy",
    "six",
    "tzdata",
    "et-xmlfile",
)


@dataclass(frozen=True, init=False)
class VerifiedSelectionFoldBaseModelOutput:
    """Ephemeral path-verification result, never a durable downstream authority."""

    manifest_id: str
    manifest_sha256: str
    lineage_evidence_id: str
    fold_id: str
    candidate_config_sha256: str
    core_output_sha256: str
    source_closure_sha256: str
    runtime_lock_sha256: str
    model_execution_attestation_id: str
    model_execution_receipt_sha256: str
    timestamp_receipt_id: str
    timestamp_receipt_sha256: str
    model_execution_key_id: str
    model_execution_trusted_key_sha256: str
    timestamp_key_id: str
    timestamp_trusted_key_sha256: str
    timestamp_journal_id: str
    model_executed_at_utc: pd.Timestamp
    trusted_time_utc: pd.Timestamp
    scope: str = field(default=BASE_OUTPUT_SCOPE, init=False)
    status: str = field(default=VERIFIED_REPLAY_STATUS, init=False)
    data_lineage_verified: bool = field(default=True, init=False)
    model_output_verified: bool = field(default=True, init=False)
    metrics_verified: bool = field(default=False, init=False)
    configuration_selection_verified: bool = field(default=False, init=False)
    process_isolation_verified: bool = field(default=False, init=False)
    campaign_eligible: bool = field(default=False, init=False)
    production_approved: bool = field(default=False, init=False)
    peak_offpeak_verified: bool = field(default=False, init=False)
    full_tier2_pfc_status: str = field(default=FULL_TIER2_STATUS, init=False)

    def __init__(self, *_: object, **__: object) -> None:
        raise Tier2MonthlyEexEvaluationError(
            "VerifiedSelectionFoldBaseModelOutput requires path-only replay"
        )


def verify_signed_selection_fold_base_model_output(
    *,
    plan_path: str | Path,
    plan_governance_receipt_path: str | Path,
    plan_timestamp_receipt_path: str | Path,
    candidate_grid_path: str | Path,
    fold_manifest_path: str | Path,
    fold_execution_attestation_path: str | Path,
    fold_timestamp_receipt_path: str | Path,
    eex_catalog_path: str | Path,
    eex_history_path: str | Path,
    base_output_manifest_path: str | Path,
    model_execution_attestation_path: str | Path,
    base_output_timestamp_receipt_path: str | Path,
    source_closure_path: str | Path,
    runtime_lock_path: str | Path,
) -> VerifiedSelectionFoldBaseModelOutput:
    """Replay one BASE candidate/baseline and verify exact pre-existing bytes."""

    paths = {
        name: _absolute_regular_file(value, label=name)
        for name, value in {
            "plan": plan_path,
            "plan_governance": plan_governance_receipt_path,
            "plan_time": plan_timestamp_receipt_path,
            "candidate_grid": candidate_grid_path,
            "fold_manifest": fold_manifest_path,
            "fold_execution": fold_execution_attestation_path,
            "fold_time": fold_timestamp_receipt_path,
            "catalog": eex_catalog_path,
            "history": eex_history_path,
            "output_manifest": base_output_manifest_path,
            "model_execution": model_execution_attestation_path,
            "output_time": base_output_timestamp_receipt_path,
            "source_closure": source_closure_path,
            "runtime_lock": runtime_lock_path,
        }.items()
    }
    if paths["output_manifest"].name != _MANIFEST_FILENAME:
        raise Tier2MonthlyEexEvaluationError(
            f"BASE output manifest must be named {_MANIFEST_FILENAME}"
        )
    package_root = paths["output_manifest"].parent
    package_before = _base_package_snapshot(package_root)

    trust_policy = _capture_trust_policy()
    trust_paths = trust_policy["paths"]
    trust_before = trust_policy["snapshots"]
    outer_paths, outer_before = _outer_input_snapshot(paths)

    manifest_path, manifest, manifest_bytes = _load_json_artifact(
        paths["output_manifest"]
    )
    _require_equal(
        package_before[_MANIFEST_FILENAME][-1],
        hashlib.sha256(manifest_bytes).hexdigest(),
        label="BASE manifest initial package snapshot hash",
    )
    fold_module._require_canonical_json_bytes(
        manifest, manifest_bytes, label="BASE output manifest"
    )
    expected_path = _absolute_regular_file(
        package_root / _OUTPUT_FILENAME, label="expected BASE output"
    )
    expected_bytes = expected_path.read_bytes()
    _require_equal(
        package_before[_OUTPUT_FILENAME][-1],
        hashlib.sha256(expected_bytes).hexdigest(),
        label="expected BASE output initial package snapshot hash",
    )
    _, source_closure, source_closure_bytes = _load_json_artifact(
        paths["source_closure"]
    )
    _, runtime_lock, runtime_lock_bytes = _load_json_artifact(paths["runtime_lock"])
    _bind_read_bytes_to_snapshot(
        paths["source_closure"], source_closure_bytes, outer_before
    )
    _bind_read_bytes_to_snapshot(paths["runtime_lock"], runtime_lock_bytes, outer_before)

    context = _verify_signed_selection_fold_evidence_context(
        plan_path=paths["plan"],
        plan_governance_receipt_path=paths["plan_governance"],
        plan_timestamp_receipt_path=paths["plan_time"],
        candidate_grid_path=paths["candidate_grid"],
        fold_manifest_path=paths["fold_manifest"],
        execution_attestation_path=paths["fold_execution"],
        fold_timestamp_receipt_path=paths["fold_time"],
        eex_catalog_path=paths["catalog"],
        eex_history_path=paths["history"],
    )
    _validate_context_binding(
        context,
        paths=paths,
        outer_before=outer_before,
        trust_policy=trust_policy,
    )
    _validate_captured_journal_artifacts(
        paths=paths,
        outer_before=outer_before,
        trust_policy=trust_policy,
    )
    source_files, source_before, loaded_source_before = _verify_source_closure(
        source_closure,
        source_closure_bytes,
        expected_sha256=context.evaluation_source_closure_sha256,
    )
    runtime_before = _verify_runtime_lock(
        runtime_lock,
        runtime_lock_bytes,
        expected_sha256=context.runtime_lock_sha256,
    )
    valid_manifest = _validate_output_manifest(
        manifest,
        context=context,
        expected_bytes=expected_bytes,
    )

    _, execution_receipt, execution_receipt_bytes = _load_json_artifact(
        paths["model_execution"]
    )
    _, timestamp_receipt, timestamp_receipt_bytes = _load_json_artifact(
        paths["output_time"]
    )
    _bind_read_bytes_to_snapshot(
        paths["model_execution"], execution_receipt_bytes, outer_before
    )
    _bind_read_bytes_to_snapshot(
        paths["output_time"], timestamp_receipt_bytes, outer_before
    )
    public_keys = trust_policy["public_keys"]
    key_hashes = trust_policy["key_hashes"]
    key_paths = trust_policy["key_paths"]
    journals = trust_policy["journals"]
    execution = _verify_exact_artifact_authority_bytes(
        execution_receipt,
        artifact_path=manifest_path,
        artifact_bytes=manifest_bytes,
        expected_role=BASE_OUTPUT_MANIFEST_ROLE,
        expected_artifact_schema=BASE_OUTPUT_MANIFEST_SCHEMA,
        receipt_schema=EXECUTION_ATTESTATION_SCHEMA,
        signature_field="execution_attestation",
        trusted_key_env=MODEL_EXECUTION_PUBLIC_KEY_ENV,
        event_time_field="executed_at_utc",
        authority_id_field="authority_id",
        id_field="attestation_id",
        journal_required=False,
        trusted_public_key=public_keys[MODEL_EXECUTION_PUBLIC_KEY_ENV],
        trusted_key_path=key_paths[MODEL_EXECUTION_PUBLIC_KEY_ENV],
        trusted_key_sha256=key_hashes[MODEL_EXECUTION_PUBLIC_KEY_ENV],
    )
    trusted_time = _verify_exact_artifact_authority_bytes(
        timestamp_receipt,
        artifact_path=manifest_path,
        artifact_bytes=manifest_bytes,
        expected_role=BASE_OUTPUT_MANIFEST_ROLE,
        expected_artifact_schema=BASE_OUTPUT_MANIFEST_SCHEMA,
        receipt_schema=TIMESTAMP_RECEIPT_SCHEMA,
        signature_field="trusted_time_attestation",
        trusted_key_env=TIMESTAMP_PUBLIC_KEY_ENV,
        event_time_field="received_at_utc",
        authority_id_field="authority_id",
        id_field="receipt_id",
        journal_required=True,
        trusted_public_key=public_keys[TIMESTAMP_PUBLIC_KEY_ENV],
        trusted_key_path=key_paths[TIMESTAMP_PUBLIC_KEY_ENV],
        trusted_key_sha256=key_hashes[TIMESTAMP_PUBLIC_KEY_ENV],
        expected_journal_id=journals[evaluation_module.TIMESTAMP_JOURNAL_ID_ENV],
    )
    _validate_authorities_and_time(
        context, execution, trusted_time, trust_policy=trust_policy
    )

    candidate_config = _selected_candidate_config(context)
    retained, history, retained_hashes, history_hashes = _base_inputs(context)
    _require_equal(
        valid_manifest["base_retained_row_hashes"],
        list(retained_hashes),
        label="BASE retained input inventory",
    )
    _require_equal(
        valid_manifest["base_history_row_hashes"],
        list(history_hashes),
        label="BASE historical input inventory",
    )
    output = base_replay_module._replay_monthly_base_outputs(
        delivery_months=context.delivery_months,
        retained_quotes=retained,
        fold_historical_context=history,
        candidate_config=candidate_config,
    )
    generated_bytes = _canonical_json_bytes(output.canonical_payload())
    if generated_bytes != expected_bytes:
        raise Tier2MonthlyEexEvaluationError(
            "replayed BASE model output bytes differ from signed commitment"
        )

    _require_equal(_file_set_snapshot(source_files), source_before, label="source closure")
    _require_equal(
        _loaded_source_snapshot(),
        loaded_source_before,
        label="loaded source closure",
    )
    _require_equal(_runtime_snapshot(), runtime_before, label="runtime closure")
    _require_equal(_file_set_snapshot(outer_paths), outer_before, label="outer input closure")
    _require_equal(_file_set_snapshot(trust_paths), trust_before, label="trust anchor closure")
    _require_equal(
        _current_trust_environment(), trust_policy["environment"],
        label="trust environment closure",
    )
    if _base_package_snapshot(package_root) != package_before:
        raise Tier2MonthlyEexEvaluationError("BASE output package changed during replay")

    values = {
        "manifest_id": str(valid_manifest["manifest_id"]),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "lineage_evidence_id": context.proof.evidence_id,
        "fold_id": context.proof.fold_id,
        "candidate_config_sha256": context.proof.candidate_config_sha256,
        "core_output_sha256": hashlib.sha256(generated_bytes).hexdigest(),
        "source_closure_sha256": context.evaluation_source_closure_sha256,
        "runtime_lock_sha256": context.runtime_lock_sha256,
        "model_execution_attestation_id": execution.envelope_id,
        "model_execution_receipt_sha256": hashlib.sha256(
            execution_receipt_bytes
        ).hexdigest(),
        "timestamp_receipt_id": trusted_time.envelope_id,
        "timestamp_receipt_sha256": hashlib.sha256(
            timestamp_receipt_bytes
        ).hexdigest(),
        "model_execution_key_id": execution.authority_key_id,
        "model_execution_trusted_key_sha256": execution.trusted_key_sha256,
        "timestamp_key_id": trusted_time.authority_key_id,
        "timestamp_trusted_key_sha256": trusted_time.trusted_key_sha256,
        "timestamp_journal_id": str(trusted_time.journal_id),
        "model_executed_at_utc": execution.event_time_utc,
        "trusted_time_utc": trusted_time.event_time_utc,
        "scope": BASE_OUTPUT_SCOPE,
        "status": VERIFIED_REPLAY_STATUS,
        "data_lineage_verified": True,
        "model_output_verified": True,
        "metrics_verified": False,
        "configuration_selection_verified": False,
        "process_isolation_verified": False,
        "campaign_eligible": False,
        "production_approved": False,
        "peak_offpeak_verified": False,
        "full_tier2_pfc_status": FULL_TIER2_STATUS,
    }
    verified = object.__new__(VerifiedSelectionFoldBaseModelOutput)
    for name, value in values.items():
        object.__setattr__(verified, name, value)
    return verified


def _validate_output_manifest(
    manifest: Mapping[str, object],
    *,
    context: object,
    expected_bytes: bytes,
) -> dict[str, object]:
    required = {
        "schema_version", "manifest_id", "artifact_role", "scope",
        "full_tier2_pfc_status", "status", "data_lineage_verified",
        "model_output_verified", "metrics_verified", "campaign_eligible",
        "production_approved", "peak_offpeak_verified", "plan_id",
        "plan_artifact_sha256", "fold_id", "fold_spec_hash",
        "lineage_evidence_id", "fold_manifest_sha256", "catalog_id",
        "catalog_sha256", "history_sha256", "frame_semantic_sha256",
        "candidate_grid_sha256", "candidate_config_sha256",
        "source_closure_sha256", "runtime_lock_sha256", "delivery_months",
        "base_history_row_hashes", "base_retained_row_hashes",
        "base_input_set_sha256", "baseline_id", "core_output_schema",
        "core_output_status", "expected_output",
    }
    _require_exact_keys(manifest, required, label="BASE output manifest")
    expected = {
        "schema_version": BASE_OUTPUT_MANIFEST_SCHEMA,
        "artifact_role": BASE_OUTPUT_MANIFEST_ROLE,
        "scope": BASE_OUTPUT_SCOPE,
        "full_tier2_pfc_status": FULL_TIER2_STATUS,
        "status": AWAITING_REPLAY_STATUS,
        "data_lineage_verified": True,
        "model_output_verified": False,
        "metrics_verified": False,
        "campaign_eligible": False,
        "production_approved": False,
        "peak_offpeak_verified": False,
        "plan_id": context.plan_id,
        "plan_artifact_sha256": context.plan_artifact_sha256,
        "fold_id": context.proof.fold_id,
        "fold_spec_hash": context.proof.fold_spec_hash,
        "lineage_evidence_id": context.proof.evidence_id,
        "fold_manifest_sha256": context.proof.artifact_sha256,
        "catalog_id": context.proof.catalog_id,
        "catalog_sha256": context.catalog_sha256,
        "history_sha256": context.history_sha256,
        "frame_semantic_sha256": context.proof.frame_semantic_sha256,
        "candidate_grid_sha256": context.candidate_grid_sha256,
        "candidate_config_sha256": context.proof.candidate_config_sha256,
        "source_closure_sha256": context.evaluation_source_closure_sha256,
        "runtime_lock_sha256": context.runtime_lock_sha256,
        "delivery_months": list(context.delivery_months),
        "baseline_id": base_replay_module.BASELINE_ID,
        "core_output_schema": base_replay_module.BASE_REPLAY_OUTPUT_SCHEMA,
        "core_output_status": base_replay_module.BASE_REPLAY_STATUS,
        "expected_output": {
            "path": _OUTPUT_FILENAME,
            "sha256": hashlib.sha256(expected_bytes).hexdigest(),
            "size_bytes": len(expected_bytes),
        },
    }
    for field_name, value in expected.items():
        _require_equal(
            manifest.get(field_name), value, label=f"BASE manifest {field_name}"
        )
    for field_name in ("base_history_row_hashes", "base_retained_row_hashes"):
        values = manifest.get(field_name)
        if not isinstance(values, list) or any(type(value) is not str for value in values):
            raise Tier2MonthlyEexEvaluationError(f"{field_name} is invalid")
        if len(values) != len(set(values)):
            raise Tier2MonthlyEexEvaluationError(f"{field_name} contains duplicates")
    expected_input_hash = base_replay_module._base_input_set_sha256(
        manifest["base_history_row_hashes"],
        manifest["base_retained_row_hashes"],
    )
    _require_equal(
        manifest.get("base_input_set_sha256"), expected_input_hash,
        label="BASE input set hash",
    )
    unsigned = dict(manifest)
    manifest_id = str(unsigned.pop("manifest_id", ""))
    _require_equal(manifest_id, _sha256_json(unsigned), label="BASE manifest_id")
    return dict(manifest)


def _selected_candidate_config(context: object) -> dict[str, object]:
    configs = context.candidate_grid.get("configs")
    if not isinstance(configs, tuple):
        raise Tier2MonthlyEexEvaluationError("candidate grid is not deeply immutable")
    matches = [
        entry for entry in configs
        if isinstance(entry, Mapping)
        and entry.get("config_sha256") == context.proof.candidate_config_sha256
    ]
    if len(matches) != 1:
        raise Tier2MonthlyEexEvaluationError("selected BASE config is absent or ambiguous")
    config = fold_module._json_thaw(matches[0].get("config"))
    if type(config) is not dict:
        raise Tier2MonthlyEexEvaluationError("selected BASE config is not a JSON object")
    base_replay_module.validate_base_candidate_config(config)
    _require_equal(
        _sha256_json(config), context.proof.candidate_config_sha256,
        label="selected BASE config hash",
    )
    return config


def _base_inputs(
    context: object,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    if context.target_load_type != "BASE":
        raise Tier2MonthlyEexEvaluationError(
            "UNSUPPORTED_REQUIRES_JOINT_PEAK_OFFPEAK_SOLVER"
        )
    retained_records = tuple(
        row for row in context.retained_rows
        if row.market.upper() == "CH" and row.load_type.upper() == "BASE"
    )
    history_records = tuple(
        row for row in context.historical_rows
        if row.market.upper() == "CH" and row.load_type.upper() == "BASE"
    )
    retained = pd.DataFrame(asdict(row) for row in retained_records)
    history = pd.DataFrame(asdict(row) for row in history_records)
    return (
        retained,
        history,
        tuple(row.row_hash for row in retained_records),
        tuple(row.row_hash for row in history_records),
    )


def _verify_source_closure(
    payload: Mapping[str, object], raw: bytes, *, expected_sha256: str
) -> tuple[
    tuple[Path, ...],
    dict[str, tuple[int, int, int, int, str]],
    dict[str, str],
]:
    fold_module._require_canonical_json_bytes(payload, raw, label="source closure")
    _require_equal(hashlib.sha256(raw).hexdigest(), expected_sha256, label="source closure hash")
    _require_exact_keys(payload, {"schema_version", "closure_id", "files"}, label="source closure")
    _require_equal(payload.get("schema_version"), SOURCE_CLOSURE_SCHEMA, label="source closure schema")
    expected_paths = _expected_source_paths()
    entries = payload.get("files")
    if not isinstance(entries, list) or len(entries) != len(expected_paths):
        raise Tier2MonthlyEexEvaluationError("source closure inventory mismatch")
    paths = tuple(expected_paths[role] for role in sorted(expected_paths))
    before = _file_set_snapshot(paths)
    loaded_before = _loaded_source_snapshot()
    expected_entries = [
        {
            "role": role,
            "path": str(expected_paths[role]),
            "sha256": before[str(expected_paths[role])][-1],
            "size_bytes": before[str(expected_paths[role])][2],
            "loaded_code_sha256": loaded_before[role],
        }
        for role in sorted(expected_paths)
    ]
    _require_equal(entries, expected_entries, label="source closure files")
    unsigned = dict(payload)
    closure_id = str(unsigned.pop("closure_id", ""))
    _require_equal(closure_id, _sha256_json(unsigned), label="source closure_id")
    return paths, before, loaded_before


def _verify_runtime_lock(
    payload: Mapping[str, object], raw: bytes, *, expected_sha256: str
) -> dict[str, object]:
    fold_module._require_canonical_json_bytes(payload, raw, label="runtime lock")
    _require_equal(hashlib.sha256(raw).hexdigest(), expected_sha256, label="runtime lock hash")
    expected = _current_runtime_payload()
    _require_equal(payload, expected, label="runtime lock content")
    return _runtime_snapshot(expected)


def _current_runtime_payload() -> dict[str, object]:
    executable = _absolute_regular_file(sys.executable, label="Python executable")
    snapshot = _file_snapshot(executable)
    unsigned: dict[str, object] = {
        "schema_version": RUNTIME_LOCK_SCHEMA,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable_path": str(executable),
            "executable_sha256": snapshot[-1],
            "executable_size_bytes": snapshot[2],
            "platform": platform.platform(),
        },
        "packages": {
            package: _distribution_payload(package) for package in _RUNTIME_PACKAGES
        },
    }
    unsigned["runtime_id"] = _sha256_json(unsigned)
    return unsigned


def _distribution_payload(package: str) -> dict[str, object]:
    try:
        distribution = metadata.distribution(package)
    except metadata.PackageNotFoundError as exc:
        raise Tier2MonthlyEexEvaluationError(
            f"runtime distribution is missing: {package}"
        ) from exc
    files = distribution.files
    if files is None:
        raise Tier2MonthlyEexEvaluationError(
            f"runtime distribution has no file inventory: {package}"
        )
    digest = hashlib.sha256()
    file_count = 0
    for relative in sorted(files, key=lambda value: str(value).casefold()):
        if not _is_runtime_distribution_file(package, Path(str(relative))):
            continue
        path = Path(distribution.locate_file(relative)).resolve()
        if not path.is_file():
            continue
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise Tier2MonthlyEexEvaluationError(
                f"runtime distribution file is unreadable: {path}"
            ) from exc
        digest.update(str(relative).replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(hashlib.sha256(raw).digest())
        file_count += 1
    if file_count == 0:
        raise Tier2MonthlyEexEvaluationError(
            f"runtime distribution inventory is empty: {package}"
        )
    return {
        "version": distribution.version,
        "file_count": file_count,
        "files_sha256": digest.hexdigest(),
    }


def _is_runtime_distribution_file(package: str, relative: Path) -> bool:
    if package == "tzdata":
        return True
    return (
        relative.suffix.casefold() in {".py", ".pyd", ".dll", ".so", ".dylib"}
        or relative.name in {"METADATA", "RECORD"}
    )


def _runtime_snapshot(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if payload is None:
        payload = _current_runtime_payload()
    return {
        "runtime_id": payload["runtime_id"],
        "python_executable_sha256": payload["python"]["executable_sha256"],  # type: ignore[index]
        "packages": payload["packages"],
    }


def _expected_source_paths() -> dict[str, Path]:
    return {
        role: _absolute_regular_file(module.__file__, label=f"source {role}")
        for role, module in _expected_source_modules().items()
    }


def _expected_source_modules() -> dict[str, object]:
    modules = {
        "acquisition_contract": acquisition_module,
        "base_output_evidence": sys.modules[__name__],
        "base_replay": base_replay_module,
        "cascading": cascading_module,
        "constraints": constraints_module,
        "databricks_client": databricks_module,
        "evaluation": evaluation_module,
        "fold_evidence": fold_module,
        "monthly_curve_priors": priors_module,
        "monthly_forward_curve": monthly_solver_module,
        "model_governance_contract": governance_module,
        "ingest_forwards": ingest_forwards_module,
        "vintage_catalog": vintage_module,
    }
    for role, module_name in (
        (
            "selection_base_inputs",
            "pfc_shaping.calibration.tier2_monthly_eex_selection_base_inputs",
        ),
        (
            "selection_input_evidence",
            "pfc_shaping.calibration.tier2_monthly_eex_selection_input_evidence",
        ),
    ):
        modules[role] = importlib.import_module(module_name)
    return modules


def _loaded_source_snapshot() -> dict[str, str]:
    return {
        role: _loaded_module_code_sha256(module)
        for role, module in _expected_source_modules().items()
    }


def _loaded_module_code_sha256(module: object) -> str:
    functions: list[tuple[str, str, str, object]] = []
    classes: list[tuple[str, str, str]] = []
    modules: list[tuple[str, str, bool]] = []
    constants: list[tuple[str, str]] = []
    module_name = str(module.__name__)
    for name, value in vars(module).items():
        if inspect.isfunction(value):
            functions.append(
                (
                    name,
                    str(value.__module__),
                    str(value.__qualname__),
                    value,
                )
            )
        elif inspect.isclass(value):
            classes.append((name, str(value.__module__), str(value.__qualname__)))
            for member_name, member in vars(value).items():
                function = (
                    member.__func__
                    if isinstance(member, (classmethod, staticmethod))
                    else member
                )
                if inspect.isfunction(function):
                    functions.append(
                        (
                            f"{name}.{member_name}",
                            str(function.__module__),
                            str(function.__qualname__),
                            function,
                        )
                    )
        elif inspect.ismodule(value):
            provider = str(value.__name__)
            modules.append((name, provider, sys.modules.get(provider) is value))
        elif name.lstrip("_").isupper():
            representation = _stable_binding_repr(value)
            if representation is None:
                raise Tier2MonthlyEexEvaluationError(
                    f"loaded source constant is unsupported: {module_name}.{name}"
                )
            constants.append((name, representation))
    digest = hashlib.sha256()
    for name, provider, qualname, function in sorted(
        functions, key=lambda item: item[:3]
    ):
        digest.update(b"FUNCTION\0")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(provider.encode("utf-8"))
        digest.update(b"\0")
        digest.update(qualname.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            b"EXACT\0"
            if _provider_binding_is_exact(function, provider, qualname)
            else b"REBIND\0"
        )
        _update_code_digest(digest, function.__code__)
        digest.update(_stable_default_repr(function.__defaults__).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_stable_default_repr(function.__kwdefaults__).encode("utf-8"))
        digest.update(b"\0")
    for name, provider, qualname in sorted(classes):
        digest.update(b"CLASS\0")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(provider.encode("utf-8"))
        digest.update(b"\0")
        digest.update(qualname.encode("utf-8"))
        digest.update(b"\0")
        class_value = vars(module)[name]
        digest.update(
            b"EXACT\0"
            if _provider_binding_is_exact(class_value, provider, qualname)
            else b"REBIND\0"
        )
    for name, provider, is_exact in sorted(modules):
        digest.update(b"MODULE\0")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(provider.encode("utf-8"))
        digest.update(b"\0")
        digest.update(b"EXACT\0" if is_exact else b"REBIND\0")
    for name, representation in sorted(constants):
        digest.update(b"CONSTANT\0")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(representation.encode("utf-8"))
        digest.update(b"\0")
    if not functions:
        raise Tier2MonthlyEexEvaluationError(
            f"loaded source module has no Python code: {module_name}"
        )
    return digest.hexdigest()


def _stable_binding_repr(value: object) -> str | None:
    if value is None or type(value) in {bool, int, float, str}:
        return repr(value)
    if isinstance(value, (tuple, list)):
        items = [_stable_binding_repr(item) for item in value]
        if any(item is None for item in items):
            return None
        return f"{type(value).__name__}({','.join(str(item) for item in items)})"
    if isinstance(value, (set, frozenset)):
        items = [_stable_binding_repr(item) for item in value]
        if any(item is None for item in items):
            return None
        return f"{type(value).__name__}({','.join(sorted(str(item) for item in items))})"
    if isinstance(value, Mapping):
        items: list[tuple[str, str]] = []
        for key, item in value.items():
            key_repr = _stable_binding_repr(key)
            item_repr = _stable_binding_repr(item)
            if key_repr is None or item_repr is None:
                return None
            items.append((key_repr, item_repr))
        return "dict(" + ",".join(
            f"{key}:{item}" for key, item in sorted(items)
        ) + ")"
    if isinstance(value, re.Pattern):
        return f"REGEX({value.pattern!r},{value.flags})"
    if isinstance(value, Path):
        return f"PATH({value})"
    if type(value) is object:
        return "OBJECT_SENTINEL"
    return None


def _provider_binding_is_exact(
    value: object, provider: str, qualname: str
) -> bool:
    provider_module = sys.modules.get(provider)
    if provider_module is None or "<locals>" in qualname:
        return False
    resolved: object = provider_module
    try:
        for component in qualname.split("."):
            resolved = getattr(resolved, component)
    except AttributeError:
        return False
    return resolved is value


def _stable_default_repr(value: object) -> str:
    stable = _stable_binding_repr(value)
    if stable is not None:
        return stable
    if isinstance(value, (tuple, list)):
        return f"{type(value).__name__}(" + ",".join(
            _stable_default_repr(item) for item in value
        ) + ")"
    if isinstance(value, Mapping):
        return "dict(" + ",".join(
            sorted(
                f"{_stable_default_repr(key)}:{_stable_default_repr(item)}"
                for key, item in value.items()
            )
        ) + ")"
    value_type = type(value)
    return f"OBJECT({value_type.__module__}.{value_type.__qualname__})"


def _update_code_digest(digest: object, code: CodeType) -> None:
    for value in (
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_nlocals,
        code.co_stacksize,
        code.co_flags,
    ):
        digest.update(int(value).to_bytes(8, "big", signed=False))
    digest.update(code.co_code)
    for sequence in (code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars):
        digest.update(repr(tuple(sequence)).encode("utf-8"))
        digest.update(b"\0")
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            digest.update(b"CODE\0")
            _update_code_digest(digest, constant)
        else:
            digest.update(type(constant).__name__.encode("ascii", errors="backslashreplace"))
            digest.update(b"\0")
            digest.update(repr(constant).encode("utf-8", errors="backslashreplace"))
            digest.update(b"\0")


def _capture_trust_policy() -> dict[str, object]:
    env_names = (
        MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV,
        TRUSTED_ACQUISITION_PUBLIC_KEY_ENV,
        TRUSTED_TIME_PUBLIC_KEY_ENV,
        evaluation_module.EXECUTION_PUBLIC_KEY_ENV,
        TIMESTAMP_PUBLIC_KEY_ENV,
        MODEL_EXECUTION_PUBLIC_KEY_ENV,
    )
    paths: list[Path] = []
    key_ids: dict[str, str] = {}
    key_hashes: dict[str, str] = {}
    key_paths: dict[str, str] = {}
    public_keys: dict[str, object] = {}
    snapshots: dict[str, tuple[int, int, int, int, str]] = {}
    environment: dict[str, str] = {}
    for env_name in env_names:
        value = os.environ.get(env_name, "").strip()
        if not value:
            raise Tier2MonthlyEexEvaluationError(f"{env_name} is required")
        path = _absolute_regular_file(value, label=f"{env_name} trust anchor")
        key_bytes, snapshot = _read_stable_file(path, label=f"{env_name} trust anchor")
        public_key = evaluation_module._load_public_key_bytes(key_bytes)
        paths.append(path)
        environment[env_name] = str(path)
        key_paths[env_name] = str(path)
        key_hashes[env_name] = snapshot[-1]
        public_keys[env_name] = public_key
        key_ids[env_name] = evaluation_module._public_key_id(public_key)
        snapshots[str(path)] = snapshot
    if len(paths) != len(set(paths)):
        raise Tier2MonthlyEexEvaluationError("trust anchor paths must be pairwise distinct")
    non_time = {
        key_ids[MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV],
        key_ids[TRUSTED_ACQUISITION_PUBLIC_KEY_ENV],
        key_ids[evaluation_module.EXECUTION_PUBLIC_KEY_ENV],
        key_ids[MODEL_EXECUTION_PUBLIC_KEY_ENV],
    }
    time_keys = {
        key_ids[TRUSTED_TIME_PUBLIC_KEY_ENV],
        key_ids[TIMESTAMP_PUBLIC_KEY_ENV],
    }
    if len(non_time) != 4 or len(time_keys) != 2 or non_time.intersection(time_keys):
        raise Tier2MonthlyEexEvaluationError("trust authority key IDs are not independent")
    journals = _current_journal_environment()
    environment.update(journals)
    path_tuple = tuple(paths)
    return {
        "paths": path_tuple,
        "snapshots": snapshots,
        "key_ids": key_ids,
        "key_hashes": key_hashes,
        "key_paths": key_paths,
        "public_keys": public_keys,
        "journals": journals,
        "environment": environment,
    }


def _read_stable_file(
    path: Path, *, label: str
) -> tuple[bytes, tuple[int, int, int, int, str]]:
    before = path.stat()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise Tier2MonthlyEexEvaluationError(f"{label} is unreadable") from exc
    after = path.stat()
    before_identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
    )
    after_identity = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
    )
    if before_identity != after_identity or len(raw) != after_identity[2]:
        raise Tier2MonthlyEexEvaluationError(f"{label} changed while being read")
    return raw, (*after_identity, hashlib.sha256(raw).hexdigest())


def _current_journal_environment() -> dict[str, str]:
    values: dict[str, str] = {}
    for env_name in (
        acquisition_module.TRUSTED_TIME_JOURNAL_ID_ENV,
        evaluation_module.TIMESTAMP_JOURNAL_ID_ENV,
    ):
        value = os.environ.get(env_name, "").strip()
        if not value:
            raise Tier2MonthlyEexEvaluationError(f"{env_name} is required")
        values[env_name] = value
    return values


def _current_trust_environment() -> dict[str, str]:
    values: dict[str, str] = {}
    for env_name in (
        MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV,
        TRUSTED_ACQUISITION_PUBLIC_KEY_ENV,
        TRUSTED_TIME_PUBLIC_KEY_ENV,
        evaluation_module.EXECUTION_PUBLIC_KEY_ENV,
        TIMESTAMP_PUBLIC_KEY_ENV,
        MODEL_EXECUTION_PUBLIC_KEY_ENV,
    ):
        value = os.environ.get(env_name, "").strip()
        if not value:
            raise Tier2MonthlyEexEvaluationError(f"{env_name} is required")
        values[env_name] = str(
            _absolute_regular_file(value, label=f"{env_name} trust anchor")
        )
    values.update(_current_journal_environment())
    return values


def _validate_context_binding(
    context: object,
    *,
    paths: Mapping[str, Path],
    outer_before: Mapping[str, tuple[int, int, int, int, str]],
    trust_policy: Mapping[str, object],
) -> None:
    bindings = {
        "plan": context.plan_artifact_sha256,
        "plan_governance": context.plan_governance_receipt_sha256,
        "plan_time": context.plan_timestamp_receipt_sha256,
        "candidate_grid": context.candidate_grid_sha256,
        "fold_manifest": context.proof.artifact_sha256,
        "fold_execution": context.fold_execution_receipt_sha256,
        "fold_time": context.fold_timestamp_receipt_sha256,
        "catalog": context.catalog_sha256,
        "history": context.history_sha256,
    }
    for role, consumed_sha256 in bindings.items():
        _require_equal(
            consumed_sha256,
            outer_before[str(paths[role])][-1],
            label=f"same-snapshot consumed {role}",
        )
    key_ids = trust_policy["key_ids"]
    role_bindings = {
        "plan governance": (
            context.plan_governance_key_id,
            key_ids[MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV],
        ),
        "plan timestamp": (
            context.plan_timestamp_key_id,
            key_ids[TIMESTAMP_PUBLIC_KEY_ENV],
        ),
        "fold execution": (
            context.fold_execution_key_id,
            key_ids[evaluation_module.EXECUTION_PUBLIC_KEY_ENV],
        ),
        "fold timestamp": (
            context.fold_timestamp_key_id,
            key_ids[TIMESTAMP_PUBLIC_KEY_ENV],
        ),
        "acquisition": (
            context.acquisition_key_id,
            key_ids[TRUSTED_ACQUISITION_PUBLIC_KEY_ENV],
        ),
    }
    for role, (actual, expected) in role_bindings.items():
        _require_equal(actual, expected, label=f"same-snapshot {role} authority")
    _require_equal(
        tuple(context.source_timestamp_key_ids),
        (key_ids[TRUSTED_TIME_PUBLIC_KEY_ENV],),
        label="same-snapshot source timestamp authorities",
    )
    expected_non_time = {
        key_ids[MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV],
        key_ids[TRUSTED_ACQUISITION_PUBLIC_KEY_ENV],
        key_ids[evaluation_module.EXECUTION_PUBLIC_KEY_ENV],
    }
    expected_time = {
        key_ids[TRUSTED_TIME_PUBLIC_KEY_ENV],
        key_ids[TIMESTAMP_PUBLIC_KEY_ENV],
    }
    _require_equal(
        set(context.forbidden_non_time_authority_key_ids),
        expected_non_time,
        label="same-snapshot non-time authorities",
    )
    _require_equal(
        set(context.trusted_time_authority_key_ids),
        expected_time,
        label="same-snapshot time authorities",
    )


def _validate_captured_journal_artifacts(
    *,
    paths: Mapping[str, Path],
    outer_before: Mapping[str, tuple[int, int, int, int, str]],
    trust_policy: Mapping[str, object],
) -> None:
    journals = trust_policy["journals"]
    expected_tier2 = journals[evaluation_module.TIMESTAMP_JOURNAL_ID_ENV]
    for role in ("plan_time", "fold_time"):
        _, receipt, raw = _load_json_artifact(paths[role])
        _bind_read_bytes_to_snapshot(paths[role], raw, outer_before)
        _require_equal(
            receipt.get("journal_id"),
            expected_tier2,
            label=f"captured {role} journal",
        )

    _, catalog, catalog_bytes = _load_json_artifact(paths["catalog"])
    _bind_read_bytes_to_snapshot(paths["catalog"], catalog_bytes, outer_before)
    expected_source = journals[acquisition_module.TRUSTED_TIME_JOURNAL_ID_ENV]
    source_documents = catalog.get("source_documents")
    if not isinstance(source_documents, list) or not source_documents:
        raise Tier2MonthlyEexEvaluationError("catalog source documents are missing")
    for index, entry in enumerate(source_documents):
        if not isinstance(entry, Mapping):
            raise Tier2MonthlyEexEvaluationError("catalog source document is invalid")
        receipt = entry.get("trusted_time_receipt")
        if not isinstance(receipt, Mapping):
            raise Tier2MonthlyEexEvaluationError(
                "catalog trusted-time receipt is invalid"
            )
        _require_equal(
            receipt.get("journal_id"),
            expected_source,
            label=f"captured source journal {index}",
        )


def _validate_authorities_and_time(
    context: object,
    execution: object,
    trusted_time: object,
    *,
    trust_policy: Mapping[str, object],
) -> None:
    key_ids = trust_policy["key_ids"]
    _require_equal(
        execution.authority_key_id,
        key_ids[MODEL_EXECUTION_PUBLIC_KEY_ENV],
        label="captured model execution authority",
    )
    _require_equal(
        trusted_time.authority_key_id,
        key_ids[TIMESTAMP_PUBLIC_KEY_ENV],
        label="captured model timestamp authority",
    )
    for verified, env_name, label in (
        (execution, MODEL_EXECUTION_PUBLIC_KEY_ENV, "model execution"),
        (trusted_time, TIMESTAMP_PUBLIC_KEY_ENV, "model timestamp"),
    ):
        _require_equal(
            verified.trusted_key_path,
            trust_policy["key_paths"][env_name],
            label=f"captured {label} key path",
        )
        _require_equal(
            verified.trusted_key_sha256,
            trust_policy["key_hashes"][env_name],
            label=f"captured {label} key bytes",
        )
    _require_equal(execution.journal_id, None, label="model execution journal")
    _require_equal(
        trusted_time.journal_id,
        trust_policy["journals"][evaluation_module.TIMESTAMP_JOURNAL_ID_ENV],
        label="captured model timestamp journal",
    )
    if execution.authority_key_id in {
        *context.forbidden_non_time_authority_key_ids,
        *context.trusted_time_authority_key_ids,
    }:
        raise Tier2MonthlyEexEvaluationError("model execution authority is not independent")
    if trusted_time.authority_key_id not in context.trusted_time_authority_key_ids:
        raise Tier2MonthlyEexEvaluationError("model output trusted-time authority is unregistered")
    if not (
        context.proof.trusted_time_utc
        <= execution.event_time_utc
        <= trusted_time.event_time_utc
    ):
        raise Tier2MonthlyEexEvaluationError("model output causal time order is invalid")


def _outer_input_snapshot(
    paths: Mapping[str, Path]
) -> tuple[tuple[Path, ...], dict[str, tuple[int, int, int, int, str]]]:
    catalog_snapshot = _file_snapshot(paths["catalog"])
    _, catalog, catalog_bytes = _load_json_artifact(paths["catalog"])
    _require_equal(
        catalog_snapshot[-1], hashlib.sha256(catalog_bytes).hexdigest(),
        label="outer catalog snapshot hash",
    )
    internal = _catalog_internal_paths(catalog, paths["catalog"].parent, paths["history"])
    selected = tuple(sorted({*paths.values(), *internal}, key=lambda path: str(path).casefold()))
    return selected, _file_set_snapshot(selected)


def _bind_read_bytes_to_snapshot(
    path: Path,
    raw: bytes,
    snapshot: Mapping[str, tuple[int, int, int, int, str]],
) -> None:
    _require_equal(
        snapshot[str(path)][-1], hashlib.sha256(raw).hexdigest(),
        label=f"initial snapshot hash for {path.name}",
    )


def _base_package_snapshot(root: Path) -> dict[str, tuple[int, int, int, int, str]]:
    expected = {_MANIFEST_FILENAME, _OUTPUT_FILENAME}
    actual = {entry.name for entry in root.iterdir()}
    if actual != expected:
        raise Tier2MonthlyEexEvaluationError(
            f"BASE output package exact inventory mismatch: "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )
    return {
        name: _file_snapshot(
            _absolute_regular_file(root / name, label=f"BASE package {name}")
        )
        for name in sorted(expected)
    }


__all__ = [
    "VerifiedSelectionFoldBaseModelOutput",
    "verify_signed_selection_fold_base_model_output",
]
