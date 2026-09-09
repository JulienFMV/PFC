"""Path-only replay of the complete config-neutral Tier 2 SELECTION inputs."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path

import pandas as pd
from cryptography.exceptions import InvalidSignature

import pfc_shaping.calibration.tier2_monthly_eex_base_output_evidence as closure_module
import pfc_shaping.calibration.tier2_monthly_eex_evaluation as evaluation_module
import pfc_shaping.calibration.tier2_monthly_eex_fold_evidence as fold_module
import pfc_shaping.calibration.tier2_monthly_eex_selection_base_inputs as inputs_module
from pfc_shaping.calibration.tier2_monthly_eex_evaluation import (
    EXECUTION_ATTESTATION_SCHEMA,
    EXPECTED_SELECTION_INPUT_RESOURCE_PROFILE,
    FULL_TIER2_STATUS,
    SELECTION_INPUT_PLAN_SCHEMA,
    TIMESTAMP_PUBLIC_KEY_ENV,
    Tier2MonthlyEexEvaluationError,
    VerifiedArtifactAuthority,
    _aware_utc,
    _canonical_json_bytes,
    _load_json_artifact,
    _require_equal,
    _require_exact_keys,
    _require_sha256,
    _required_text,
    _sha256_json,
    _verify_exact_artifact_authority_bytes,
    verify_governed_plan_artifact,
)
from pfc_shaping.data.acquisition_contract import (
    TRUSTED_ACQUISITION_PUBLIC_KEY_ENV,
    TRUSTED_TIME_JOURNAL_ID_ENV,
    TRUSTED_TIME_PUBLIC_KEY_ENV,
)
from pfc_shaping.pipeline.model_governance_contract import (
    MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV,
)


SELECTION_INPUT_MANIFEST_SCHEMA = (
    "tier2_monthly_eex_selection_base_inputs_manifest.v1"
)
SELECTION_INPUT_MANIFEST_ROLE = (
    "tier2_monthly_eex_selection_base_inputs_manifest"
)
SELECTION_INPUT_SCOPE = "MONTHLY_EEX_BASE_SELECTION_INPUTS_ONLY"
SELECTION_INPUT_STATUS = "CONFIG_NEUTRAL_BASE_INPUTS_REPLAYED"
SELECTION_INPUT_EXECUTION_PUBLIC_KEY_ENV = (
    "PFC_TIER2_SELECTION_INPUT_EXECUTION_TRUSTED_PUBLIC_KEY_PATH"
)
TIMESTAMP_JOURNAL_WITNESS_PUBLIC_KEY_ENV = (
    "PFC_TIER2_TIMESTAMP_JOURNAL_WITNESS_TRUSTED_PUBLIC_KEY_PATH"
)
TIMESTAMP_JOURNAL_HEAD_PATH_ENV = "PFC_TIER2_TIMESTAMP_JOURNAL_HEAD_PATH"
TIMESTAMP_JOURNAL_CURRENT_HEAD_ID_ENV = (
    "PFC_TIER2_TIMESTAMP_JOURNAL_CURRENT_HEAD_ID"
)
LINKED_TIMESTAMP_RECEIPT_SCHEMA = (
    "tier2_selection_input_linked_timestamp_receipt.v1"
)
JOURNAL_HEAD_SCHEMA = "tier2_timestamp_journal_head_log.v1"

_MANIFEST_FILENAME = "selection_base_inputs_manifest.v1.json"
_INPUTS_FILENAME = "selection_base_inputs.v1.json"


@dataclass(frozen=True, init=False)
class VerifiedSelectionBaseInputEvidence:
    """Ephemeral result of a complete path-only D138 input replay."""

    manifest_id: str
    manifest_sha256: str
    input_set_id: str
    fold_count: int
    ordered_fold_ids: tuple[str, ...]
    execution_attestation_id: str
    linked_timestamp_receipt_id: str
    journal_head_id: str
    trusted_time_utc: pd.Timestamp
    scope: str = field(default=SELECTION_INPUT_SCOPE, init=False)
    status: str = field(default=SELECTION_INPUT_STATUS, init=False)
    data_lineage_verified: bool = field(default=True, init=False)
    configuration_neutral_verified: bool = field(default=True, init=False)
    model_output_verified: bool = field(default=False, init=False)
    configuration_selection_verified: bool = field(default=False, init=False)
    target_non_observation_verified: bool = field(default=False, init=False)
    target_reveal_order_verified: bool = field(default=False, init=False)
    process_isolation_verified: bool = field(default=False, init=False)
    metrics_verified: bool = field(default=False, init=False)
    holdout_verified: bool = field(default=False, init=False)
    campaign_eligible: bool = field(default=False, init=False)
    production_approved: bool = field(default=False, init=False)
    peak_offpeak_verified: bool = field(default=False, init=False)
    full_tier2_pfc_status: str = field(default=FULL_TIER2_STATUS, init=False)

    def __init__(self, *_: object, **__: object) -> None:
        raise Tier2MonthlyEexEvaluationError(
            "VerifiedSelectionBaseInputEvidence requires path-only replay"
        )


def verify_signed_selection_base_inputs(
    *,
    plan_path: str | Path,
    plan_governance_receipt_path: str | Path,
    plan_timestamp_receipt_path: str | Path,
    eex_catalog_path: str | Path,
    eex_history_path: str | Path,
    selection_input_manifest_path: str | Path,
    execution_attestation_path: str | Path,
    linked_timestamp_receipt_path: str | Path,
    journal_head_path: str | Path,
    source_closure_path: str | Path,
    runtime_lock_path: str | Path,
) -> VerifiedSelectionBaseInputEvidence:
    """Authenticate and replay all preregistered SELECTION BASE inputs."""

    paths = {
        name: fold_module._absolute_regular_file(value, label=name)
        for name, value in {
            "plan": plan_path,
            "plan_governance": plan_governance_receipt_path,
            "plan_time": plan_timestamp_receipt_path,
            "catalog": eex_catalog_path,
            "history": eex_history_path,
            "manifest": selection_input_manifest_path,
            "execution": execution_attestation_path,
            "linked_time": linked_timestamp_receipt_path,
            "journal_head": journal_head_path,
            "source_closure": source_closure_path,
            "runtime_lock": runtime_lock_path,
        }.items()
    }
    if paths["manifest"].name != _MANIFEST_FILENAME:
        raise Tier2MonthlyEexEvaluationError(
            f"selection input manifest must be named {_MANIFEST_FILENAME}"
        )
    package_root = paths["manifest"].parent
    inputs_path = fold_module._absolute_regular_file(
        package_root / _INPUTS_FILENAME, label="selection input payload"
    )
    _preflight_file_sizes(paths, inputs_path=inputs_path)
    package_before = _package_snapshot(package_root)
    _, captured_catalog, captured_catalog_bytes = _load_json_artifact(paths["catalog"])
    _preflight_catalog_internal_sizes(
        captured_catalog,
        catalog_root=paths["catalog"].parent,
        selected_history=paths["history"],
    )
    outer_paths = _outer_paths(paths, captured_catalog)
    outer_before = fold_module._file_set_snapshot(outer_paths)
    trust = _capture_trust_policy(expected_journal_head=paths["journal_head"])
    _, captured_plan, captured_plan_bytes = _load_json_artifact(paths["plan"])
    _, captured_governance, captured_governance_bytes = _load_json_artifact(
        paths["plan_governance"]
    )
    _, captured_plan_time, captured_plan_time_bytes = _load_json_artifact(
        paths["plan_time"]
    )
    del captured_governance
    for path, raw in (
        (paths["catalog"], captured_catalog_bytes),
        (paths["plan"], captured_plan_bytes),
        (paths["plan_governance"], captured_governance_bytes),
        (paths["plan_time"], captured_plan_time_bytes),
    ):
        _bind_bytes(path, raw, outer_before)

    manifest_path, manifest, manifest_bytes = _load_json_artifact(paths["manifest"])
    fold_module._require_canonical_json_bytes(
        manifest, manifest_bytes, label="selection input manifest"
    )
    _bind_bytes(paths["manifest"], manifest_bytes, package_before)
    inputs_bytes = inputs_path.read_bytes()
    _bind_bytes(inputs_path, inputs_bytes, package_before)

    _, source_closure, source_closure_bytes = _load_json_artifact(
        paths["source_closure"]
    )
    _, runtime_lock, runtime_lock_bytes = _load_json_artifact(paths["runtime_lock"])
    _bind_bytes(paths["source_closure"], source_closure_bytes, outer_before)
    _bind_bytes(paths["runtime_lock"], runtime_lock_bytes, outer_before)

    plan = verify_governed_plan_artifact(
        paths["plan"],
        governance_receipt_path=paths["plan_governance"],
        timestamp_receipt_path=paths["plan_time"],
    )
    for actual, expected, label in (
        (plan.artifact_sha256, hashlib.sha256(captured_plan_bytes).hexdigest(), "plan"),
        (
            plan.governance_receipt_sha256,
            hashlib.sha256(captured_governance_bytes).hexdigest(),
            "plan governance receipt",
        ),
        (
            plan.timestamp_receipt_sha256,
            hashlib.sha256(captured_plan_time_bytes).hexdigest(),
            "plan timestamp receipt",
        ),
    ):
        _require_equal(actual, expected, label=f"captured {label} hash")
    _require_equal(
        _sha256_json({key: value for key, value in captured_plan.items() if key != "plan_id"}),
        plan.plan_id,
        label="captured plan identity",
    )
    _require_equal(
        plan.payload.get("schema_version"),
        SELECTION_INPUT_PLAN_SCHEMA,
        label="D138 plan schema",
    )
    _require_equal(
        _thaw(plan.payload.get("selection_input_resource_profile")),
        EXPECTED_SELECTION_INPUT_RESOURCE_PROFILE,
        label="D138 resource profile",
    )
    _validate_plan_authorities(plan, trust)
    try:
        frame, catalog, source_authorities = fold_module._verify_eex_paths(
            paths["catalog"], paths["history"]
        )
    except ValueError as exc:
        raise Tier2MonthlyEexEvaluationError(
            "EEX source verification failed"
        ) from exc
    _require_equal(
        catalog.catalog_sha256,
        outer_before[str(paths["catalog"])][-1],
        label="captured catalog hash",
    )
    _require_equal(
        catalog.history_sha256,
        outer_before[str(paths["history"])][-1],
        label="captured history hash",
    )
    _validate_source_authorities(source_authorities, trust)
    frame_hash = fold_module._frame_semantic_sha256(frame)
    source_files, source_before, loaded_before = closure_module._verify_source_closure(
        source_closure,
        source_closure_bytes,
        expected_sha256=str(plan.payload["evaluation_source_closure_sha256"]),
    )
    runtime_before = closure_module._verify_runtime_lock(
        runtime_lock,
        runtime_lock_bytes,
        expected_sha256=str(plan.payload["runtime_lock_sha256"]),
    )

    derived = inputs_module._derive_selection_base_input_set(frame, plan=plan)
    derived_bytes = _canonical_json_bytes(derived.canonical_payload())
    if derived_bytes != inputs_bytes:
        raise Tier2MonthlyEexEvaluationError(
            "replayed selection input bytes differ from signed commitment"
        )
    valid_manifest = _validate_manifest(
        manifest,
        plan=plan,
        catalog=catalog,
        source_authorities=source_authorities,
        frame_semantic_sha256=frame_hash,
        derived=derived,
        inputs_bytes=inputs_bytes,
    )

    _, execution_payload, execution_bytes = _load_json_artifact(paths["execution"])
    _, time_payload, time_bytes = _load_json_artifact(paths["linked_time"])
    _, head_payload, head_bytes = _load_json_artifact(paths["journal_head"])
    for label, payload, raw in (
        ("selection input execution attestation", execution_payload, execution_bytes),
        ("selection input linked timestamp receipt", time_payload, time_bytes),
        ("selection input timestamp journal", head_payload, head_bytes),
    ):
        fold_module._require_canonical_json_bytes(payload, raw, label=label)
    for path, raw in (
        (paths["execution"], execution_bytes),
        (paths["linked_time"], time_bytes),
        (paths["journal_head"], head_bytes),
    ):
        _bind_bytes(path, raw, outer_before)

    execution = _verify_exact_artifact_authority_bytes(
        execution_payload,
        artifact_path=manifest_path,
        artifact_bytes=manifest_bytes,
        expected_role=SELECTION_INPUT_MANIFEST_ROLE,
        expected_artifact_schema=SELECTION_INPUT_MANIFEST_SCHEMA,
        receipt_schema=EXECUTION_ATTESTATION_SCHEMA,
        signature_field="execution_attestation",
        trusted_key_env=SELECTION_INPUT_EXECUTION_PUBLIC_KEY_ENV,
        event_time_field="executed_at_utc",
        authority_id_field="authority_id",
        id_field="attestation_id",
        journal_required=False,
        trusted_public_key=trust["public_keys"][
            SELECTION_INPUT_EXECUTION_PUBLIC_KEY_ENV
        ],
        trusted_key_path=trust["key_paths"][
            SELECTION_INPUT_EXECUTION_PUBLIC_KEY_ENV
        ],
        trusted_key_sha256=trust["key_hashes"][
            SELECTION_INPUT_EXECUTION_PUBLIC_KEY_ENV
        ],
    )
    linked_time = _verify_linked_timestamp_receipt(
        time_payload,
        manifest_bytes=manifest_bytes,
        execution=execution,
        execution_bytes=execution_bytes,
        plan_timestamp_receipt=captured_plan_time,
        plan_timestamp_receipt_bytes=captured_plan_time_bytes,
        trust=trust,
    )
    journal_head_id, witnessed_at = _verify_journal_head(
        head_payload,
        linked_receipt=time_payload,
        linked_receipt_bytes=time_bytes,
        plan_timestamp_receipt=captured_plan_time,
        plan_timestamp_receipt_bytes=captured_plan_time_bytes,
        trust=trust,
    )
    _validate_causal_order(
        plan_time=plan.trusted_time_utc,
        execution_time=execution.event_time_utc,
        receipt_time=linked_time.event_time_utc,
        witnessed_at=witnessed_at,
    )

    _require_equal(
        fold_module._file_set_snapshot(source_files),
        source_before,
        label="selection input source closure",
    )
    _require_equal(
        closure_module._loaded_source_snapshot(),
        loaded_before,
        label="selection input loaded source closure",
    )
    _require_equal(
        closure_module._runtime_snapshot(),
        runtime_before,
        label="selection input runtime closure",
    )
    _require_equal(
        fold_module._file_set_snapshot(outer_paths),
        outer_before,
        label="selection input outer closure",
    )
    _require_equal(
        fold_module._file_set_snapshot(trust["paths"]),
        trust["snapshots"],
        label="selection input trust closure",
    )
    _require_equal(
        _current_trust_environment(),
        trust["environment"],
        label="selection input trust environment",
    )
    if _package_snapshot(package_root) != package_before:
        raise Tier2MonthlyEexEvaluationError(
            "selection input package changed during replay"
        )

    values = {
        "manifest_id": str(valid_manifest["manifest_id"]),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "input_set_id": derived.input_set_id,
        "fold_count": derived.fold_count,
        "ordered_fold_ids": derived.ordered_fold_ids,
        "execution_attestation_id": execution.envelope_id,
        "linked_timestamp_receipt_id": linked_time.envelope_id,
        "journal_head_id": journal_head_id,
        "trusted_time_utc": linked_time.event_time_utc,
    }
    verified = object.__new__(VerifiedSelectionBaseInputEvidence)
    for name, value in values.items():
        object.__setattr__(verified, name, value)
    return verified


def _validate_manifest(
    manifest: Mapping[str, object],
    *,
    plan: object,
    catalog: object,
    source_authorities: Mapping[str, object],
    frame_semantic_sha256: str,
    derived: object,
    inputs_bytes: bytes,
) -> dict[str, object]:
    claims = {
        "data_lineage_verified": True,
        "configuration_neutral": True,
        "model_output_verified": False,
        "configuration_selection_verified": False,
        "target_non_observation_verified": False,
        "target_reveal_order_verified": False,
        "process_isolation_verified": False,
        "metrics_verified": False,
        "holdout_verified": False,
        "campaign_eligible": False,
        "production_approved": False,
        "peak_offpeak_verified": False,
    }
    required = {
        "schema_version",
        "manifest_id",
        "artifact_role",
        "scope",
        "status",
        "full_tier2_pfc_status",
        *claims,
        "plan_id",
        "plan_artifact_sha256",
        "plan_governance_receipt_sha256",
        "plan_timestamp_receipt_sha256",
        "catalog_root",
        "frame_semantic_sha256",
        "source_authorities",
        "evaluation_source_closure_sha256",
        "runtime_lock_sha256",
        "resource_profile",
        "fold_count",
        "ordered_fold_ids",
        "input_set_id",
        "inputs_artifact",
    }
    _require_exact_keys(manifest, required, label="selection input manifest")
    expected = {
        "schema_version": SELECTION_INPUT_MANIFEST_SCHEMA,
        "artifact_role": SELECTION_INPUT_MANIFEST_ROLE,
        "scope": SELECTION_INPUT_SCOPE,
        "status": SELECTION_INPUT_STATUS,
        "full_tier2_pfc_status": FULL_TIER2_STATUS,
        **claims,
        "plan_id": plan.plan_id,
        "plan_artifact_sha256": plan.artifact_sha256,
        "plan_governance_receipt_sha256": plan.governance_receipt_sha256,
        "plan_timestamp_receipt_sha256": plan.timestamp_receipt_sha256,
        "catalog_root": {
            "catalog_id": catalog.catalog_id,
            "catalog_sha256": catalog.catalog_sha256,
            "history_sha256": catalog.history_sha256,
            "root_sha256": plan.payload["selection_eex_root"]["root_sha256"],
        },
        "frame_semantic_sha256": frame_semantic_sha256,
        "source_authorities": dict(source_authorities),
        "evaluation_source_closure_sha256": plan.payload[
            "evaluation_source_closure_sha256"
        ],
        "runtime_lock_sha256": plan.payload["runtime_lock_sha256"],
        "resource_profile": EXPECTED_SELECTION_INPUT_RESOURCE_PROFILE,
        "fold_count": derived.fold_count,
        "ordered_fold_ids": list(derived.ordered_fold_ids),
        "input_set_id": derived.input_set_id,
        "inputs_artifact": {
            "path": _INPUTS_FILENAME,
            "schema_version": inputs_module.SELECTION_BASE_INPUT_SET_SCHEMA,
            "sha256": hashlib.sha256(inputs_bytes).hexdigest(),
            "size_bytes": len(inputs_bytes),
        },
    }
    for name, value in expected.items():
        _require_equal(manifest.get(name), value, label=f"selection input {name}")
    _require_equal(
        manifest.get("catalog_root"),
        _thaw(plan.payload["selection_eex_root"]),
        label="selection input plan catalog root",
    )
    unsigned = dict(manifest)
    manifest_id = str(unsigned.pop("manifest_id", ""))
    _require_equal(manifest_id, _sha256_json(unsigned), label="selection input manifest_id")
    return dict(manifest)


def _verify_linked_timestamp_receipt(
    payload: Mapping[str, object],
    *,
    manifest_bytes: bytes,
    execution: object,
    execution_bytes: bytes,
    plan_timestamp_receipt: Mapping[str, object],
    plan_timestamp_receipt_bytes: bytes,
    trust: Mapping[str, object],
) -> object:
    plan_time = plan_timestamp_receipt
    unsigned = dict(payload)
    signature = unsigned.pop("trusted_time_attestation", None)
    required = {
        "schema_version",
        "artifact_role",
        "artifact_schema_version",
        "artifact_sha256",
        "artifact_size_bytes",
        "authority_id",
        "received_at_utc",
        "journal_id",
        "journal_sequence",
        "previous_receipt_id",
        "execution_attestation_id",
        "execution_attestation_sha256",
        "plan_timestamp_receipt_sha256",
        "receipt_id",
    }
    _require_exact_keys(unsigned, required, label="linked timestamp receipt")
    _require_equal(
        unsigned.get("schema_version"),
        LINKED_TIMESTAMP_RECEIPT_SCHEMA,
        label="linked timestamp schema",
    )
    for name, expected in (
        ("artifact_role", SELECTION_INPUT_MANIFEST_ROLE),
        ("artifact_schema_version", SELECTION_INPUT_MANIFEST_SCHEMA),
        ("artifact_sha256", hashlib.sha256(manifest_bytes).hexdigest()),
        ("artifact_size_bytes", len(manifest_bytes)),
    ):
        _require_equal(unsigned.get(name), expected, label=f"linked timestamp {name}")
    _required_text(unsigned.get("authority_id"), label="linked timestamp authority")
    _require_equal(
        unsigned.get("execution_attestation_id"),
        execution.envelope_id,
        label="linked execution attestation ID",
    )
    _require_equal(
        unsigned.get("execution_attestation_sha256"),
        hashlib.sha256(execution_bytes).hexdigest(),
        label="linked execution attestation hash",
    )
    _require_equal(
        unsigned.get("plan_timestamp_receipt_sha256"),
        hashlib.sha256(plan_timestamp_receipt_bytes).hexdigest(),
        label="linked plan timestamp receipt hash",
    )
    _require_equal(
        unsigned.get("journal_id"),
        plan_time.get("journal_id"),
        label="linked timestamp journal",
    )
    _require_equal(
        unsigned.get("journal_id"),
        trust["journal_id"],
        label="captured linked timestamp journal",
    )
    plan_sequence = _exact_nonnegative_int(
        plan_time.get("journal_sequence"), label="plan journal sequence"
    )
    _require_equal(
        unsigned.get("journal_sequence"),
        plan_sequence + 1,
        label="linked timestamp sequence",
    )
    _require_equal(
        unsigned.get("previous_receipt_id"),
        plan_time.get("receipt_id"),
        label="linked previous receipt",
    )
    receipt_id = _verify_signed_payload(
        unsigned,
        signature,
        id_field="receipt_id",
        signature_label="trusted_time_attestation",
        public_key=trust["public_keys"][TIMESTAMP_PUBLIC_KEY_ENV],
    )
    return VerifiedArtifactAuthority(
        artifact_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        artifact_role=SELECTION_INPUT_MANIFEST_ROLE,
        artifact_schema_version=SELECTION_INPUT_MANIFEST_SCHEMA,
        envelope_id=receipt_id,
        authority_key_id=trust["key_ids"][TIMESTAMP_PUBLIC_KEY_ENV],
        trusted_key_path=trust["key_paths"][TIMESTAMP_PUBLIC_KEY_ENV],
        trusted_key_sha256=trust["key_hashes"][TIMESTAMP_PUBLIC_KEY_ENV],
        journal_id=str(unsigned["journal_id"]),
        event_time_utc=_aware_utc(
            unsigned["received_at_utc"], label="received_at_utc"
        ),
    )


def _verify_journal_head(
    payload: Mapping[str, object],
    *,
    linked_receipt: Mapping[str, object],
    linked_receipt_bytes: bytes,
    plan_timestamp_receipt: Mapping[str, object],
    plan_timestamp_receipt_bytes: bytes,
    trust: Mapping[str, object],
) -> tuple[str, pd.Timestamp]:
    plan_receipt = plan_timestamp_receipt
    plan_receipt_bytes = plan_timestamp_receipt_bytes
    unsigned = dict(payload)
    signature = unsigned.pop("witness_attestation", None)
    required = {
        "schema_version",
        "journal_id",
        "entries",
        "current_head_sequence",
        "current_head_id",
        "witness_authority_id",
        "witnessed_at_utc",
        "journal_head_id",
    }
    _require_exact_keys(unsigned, required, label="timestamp journal head")
    _require_equal(
        unsigned.get("schema_version"), JOURNAL_HEAD_SCHEMA, label="journal head schema"
    )
    _require_equal(
        unsigned.get("journal_id"),
        linked_receipt.get("journal_id"),
        label="journal head journal_id",
    )
    entries = unsigned.get("entries")
    max_entries = int(
        EXPECTED_SELECTION_INPUT_RESOURCE_PROFILE["max_timestamp_journal_entries"]
    )
    if not isinstance(entries, list) or not 2 <= len(entries) <= max_entries:
        raise Tier2MonthlyEexEvaluationError("timestamp journal entries are invalid")
    previous_head_id = ""
    first_receipt_sequence: int | None = None
    normalized_entries: list[dict[str, object]] = []
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, Mapping):
            raise Tier2MonthlyEexEvaluationError("timestamp journal entry is invalid")
        entry = dict(raw_entry)
        _require_exact_keys(
            entry,
            {
                "head_sequence",
                "receipt_id",
                "receipt_sha256",
                "receipt_journal_sequence",
                "previous_head_id",
                "head_id",
            },
            label="timestamp journal entry",
        )
        _require_equal(entry.get("head_sequence"), index, label="journal head sequence")
        _require_equal(
            entry.get("previous_head_id"),
            previous_head_id,
            label="journal previous head",
        )
        receipt_sequence = _exact_nonnegative_int(
            entry.get("receipt_journal_sequence"),
            label="receipt journal sequence",
        )
        if first_receipt_sequence is None:
            first_receipt_sequence = receipt_sequence
        _require_equal(
            receipt_sequence,
            first_receipt_sequence + index,
            label="contiguous receipt journal sequence",
        )
        _require_sha256(entry.get("receipt_id"), label="journal receipt ID")
        _require_sha256(entry.get("receipt_sha256"), label="journal receipt hash")
        identity = dict(entry)
        head_id = str(identity.pop("head_id", ""))
        _require_equal(head_id, _sha256_json(identity), label="journal entry head_id")
        previous_head_id = head_id
        normalized_entries.append(entry)
    _require_equal(
        unsigned.get("current_head_sequence"),
        len(entries) - 1,
        label="current journal head sequence",
    )
    _require_equal(
        unsigned.get("current_head_id"),
        previous_head_id,
        label="current journal head ID",
    )
    _require_equal(
        unsigned.get("current_head_id"),
        trust["current_head_id"],
        label="external current journal head ID",
    )
    expected_plan = {
        "receipt_id": plan_receipt.get("receipt_id"),
        "receipt_sha256": hashlib.sha256(plan_receipt_bytes).hexdigest(),
        "receipt_journal_sequence": plan_receipt.get("journal_sequence"),
    }
    expected_linked = {
        "receipt_id": linked_receipt.get("receipt_id"),
        "receipt_sha256": hashlib.sha256(linked_receipt_bytes).hexdigest(),
        "receipt_journal_sequence": linked_receipt.get("journal_sequence"),
    }
    plan_indices = [
        index
        for index, entry in enumerate(normalized_entries)
        if all(entry.get(name) == value for name, value in expected_plan.items())
    ]
    if len(plan_indices) != 1 or plan_indices[0] + 1 >= len(normalized_entries):
        raise Tier2MonthlyEexEvaluationError(
            "timestamp journal does not contain one plan anchor"
        )
    linked_entry = normalized_entries[plan_indices[0] + 1]
    if any(linked_entry.get(name) != value for name, value in expected_linked.items()):
        raise Tier2MonthlyEexEvaluationError(
            "timestamp journal does not include the linked D138 receipt"
        )
    _required_text(unsigned.get("witness_authority_id"), label="witness authority")
    witnessed_at = _aware_utc(unsigned.get("witnessed_at_utc"), label="witnessed_at_utc")
    journal_head_id = _verify_signed_payload(
        unsigned,
        signature,
        id_field="journal_head_id",
        signature_label="witness_attestation",
        public_key=trust["public_keys"][TIMESTAMP_JOURNAL_WITNESS_PUBLIC_KEY_ENV],
    )
    return journal_head_id, witnessed_at


def _verify_signed_payload(
    unsigned: Mapping[str, object],
    signature: object,
    *,
    id_field: str,
    signature_label: str,
    public_key: object,
) -> str:
    identity = dict(unsigned)
    envelope_id = str(identity.pop(id_field, ""))
    _require_equal(envelope_id, _sha256_json(identity), label=id_field)
    if not isinstance(signature, Mapping):
        raise Tier2MonthlyEexEvaluationError(f"{signature_label} is missing")
    _require_exact_keys(
        signature,
        {"algorithm", "key_id", "payload_sha256", "value_base64"},
        label=signature_label,
    )
    _require_equal(signature.get("algorithm"), "Ed25519", label="signature algorithm")
    _require_equal(
        signature.get("key_id"),
        evaluation_module._public_key_id(public_key),
        label=f"{signature_label} key",
    )
    signed = {**identity, id_field: envelope_id}
    canonical = _canonical_json_bytes(signed)
    _require_equal(
        signature.get("payload_sha256"),
        hashlib.sha256(canonical).hexdigest(),
        label=f"{signature_label} payload hash",
    )
    try:
        raw = base64.b64decode(str(signature.get("value_base64", "")), validate=True)
        public_key.verify(raw, canonical)
    except (TypeError, ValueError, InvalidSignature) as exc:
        raise Tier2MonthlyEexEvaluationError(
            f"{signature_label} signature is invalid"
        ) from exc
    return envelope_id


def _capture_trust_policy(*, expected_journal_head: Path) -> dict[str, object]:
    env_names = (
        MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV,
        TRUSTED_ACQUISITION_PUBLIC_KEY_ENV,
        TRUSTED_TIME_PUBLIC_KEY_ENV,
        TIMESTAMP_PUBLIC_KEY_ENV,
        SELECTION_INPUT_EXECUTION_PUBLIC_KEY_ENV,
        TIMESTAMP_JOURNAL_WITNESS_PUBLIC_KEY_ENV,
    )
    paths: list[Path] = []
    snapshots: dict[str, object] = {}
    key_ids: dict[str, str] = {}
    key_paths: dict[str, str] = {}
    key_hashes: dict[str, str] = {}
    public_keys: dict[str, object] = {}
    environment: dict[str, str] = {}
    for env_name in env_names:
        value = os.environ.get(env_name, "").strip()
        if not value:
            raise Tier2MonthlyEexEvaluationError(f"{env_name} is required")
        path = fold_module._absolute_regular_file(value, label=f"{env_name} trust anchor")
        raw, snapshot = closure_module._read_stable_file(
            path, label=f"{env_name} trust anchor"
        )
        key = evaluation_module._load_public_key_bytes(raw)
        paths.append(path)
        snapshots[str(path)] = snapshot
        key_ids[env_name] = evaluation_module._public_key_id(key)
        key_paths[env_name] = str(path)
        key_hashes[env_name] = snapshot[-1]
        public_keys[env_name] = key
        environment[env_name] = str(path)
    if len(set(paths)) != len(paths) or len(set(key_ids.values())) != len(key_ids):
        raise Tier2MonthlyEexEvaluationError(
            "selection input trust authorities must be pairwise independent"
        )
    journal_id = os.environ.get(evaluation_module.TIMESTAMP_JOURNAL_ID_ENV, "").strip()
    if not journal_id:
        raise Tier2MonthlyEexEvaluationError(
            f"{evaluation_module.TIMESTAMP_JOURNAL_ID_ENV} is required"
        )
    environment[evaluation_module.TIMESTAMP_JOURNAL_ID_ENV] = journal_id
    source_journal_id = os.environ.get(TRUSTED_TIME_JOURNAL_ID_ENV, "").strip()
    if not source_journal_id:
        raise Tier2MonthlyEexEvaluationError(
            f"{TRUSTED_TIME_JOURNAL_ID_ENV} is required"
        )
    environment[TRUSTED_TIME_JOURNAL_ID_ENV] = source_journal_id
    head_value = os.environ.get(TIMESTAMP_JOURNAL_HEAD_PATH_ENV, "").strip()
    if not head_value:
        raise Tier2MonthlyEexEvaluationError(
            f"{TIMESTAMP_JOURNAL_HEAD_PATH_ENV} is required"
        )
    trusted_head = fold_module._absolute_regular_file(
        head_value, label="trusted timestamp journal head"
    )
    _require_equal(
        trusted_head, expected_journal_head, label="trusted timestamp journal head path"
    )
    environment[TIMESTAMP_JOURNAL_HEAD_PATH_ENV] = str(trusted_head)
    current_head_id = os.environ.get(
        TIMESTAMP_JOURNAL_CURRENT_HEAD_ID_ENV, ""
    ).strip()
    _require_sha256(current_head_id, label="external current journal head ID")
    environment[TIMESTAMP_JOURNAL_CURRENT_HEAD_ID_ENV] = current_head_id
    return {
        "paths": tuple(paths),
        "snapshots": snapshots,
        "key_ids": key_ids,
        "key_paths": key_paths,
        "key_hashes": key_hashes,
        "public_keys": public_keys,
        "environment": environment,
        "journal_id": journal_id,
        "source_journal_id": source_journal_id,
        "current_head_id": current_head_id,
    }


def _validate_plan_authorities(plan: object, trust: Mapping[str, object]) -> None:
    _require_equal(
        plan.governance_key_id,
        trust["key_ids"][MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV],
        label="captured plan governance authority",
    )
    _require_equal(
        plan.timestamp_key_id,
        trust["key_ids"][TIMESTAMP_PUBLIC_KEY_ENV],
        label="captured plan timestamp authority",
    )


def _validate_source_authorities(
    source: Mapping[str, object], trust: Mapping[str, object]
) -> None:
    _require_equal(
        source.get("acquisition_key_id"),
        trust["key_ids"][TRUSTED_ACQUISITION_PUBLIC_KEY_ENV],
        label="captured acquisition authority",
    )
    _require_equal(
        source.get("timestamp_key_ids"),
        [trust["key_ids"][TRUSTED_TIME_PUBLIC_KEY_ENV]],
        label="captured source timestamp authority",
    )
    _require_equal(
        source.get("timestamp_journal_ids"),
        [trust["source_journal_id"]],
        label="captured source timestamp journal",
    )


def _validate_causal_order(
    *,
    plan_time: pd.Timestamp,
    execution_time: pd.Timestamp,
    receipt_time: pd.Timestamp,
    witnessed_at: pd.Timestamp,
) -> None:
    if not plan_time <= execution_time <= receipt_time <= witnessed_at:
        raise Tier2MonthlyEexEvaluationError(
            "selection input causal time order is invalid"
        )


def _package_snapshot(root: Path) -> dict[str, object]:
    expected = {_MANIFEST_FILENAME, _INPUTS_FILENAME}
    actual = {entry.name for entry in root.iterdir()}
    if actual != expected:
        raise Tier2MonthlyEexEvaluationError(
            "selection input package exact inventory mismatch"
        )
    return {
        str(path): fold_module._file_snapshot(path)
        for path in (
            fold_module._absolute_regular_file(root / name, label=name)
            for name in sorted(expected)
        )
    }


def _preflight_file_sizes(
    paths: Mapping[str, Path], *, inputs_path: Path
) -> None:
    profile = EXPECTED_SELECTION_INPUT_RESOURCE_PROFILE
    limits = {
        "plan": int(profile["max_selection_input_manifest_bytes"]),
        "plan_governance": int(profile["max_evidence_envelope_bytes"]),
        "plan_time": int(profile["max_evidence_envelope_bytes"]),
        "catalog": int(profile["max_eex_catalog_bytes"]),
        "history": int(profile["max_eex_history_bytes"]),
        "manifest": int(profile["max_selection_input_manifest_bytes"]),
        "execution": int(profile["max_evidence_envelope_bytes"]),
        "linked_time": int(profile["max_evidence_envelope_bytes"]),
        "journal_head": int(profile["max_evidence_envelope_bytes"]),
        "source_closure": int(profile["max_source_closure_bytes"]),
        "runtime_lock": int(profile["max_runtime_lock_bytes"]),
    }
    for name, path in paths.items():
        if path.stat().st_size > limits[name]:
            raise Tier2MonthlyEexEvaluationError(
                f"{name} exceeds the preregistered byte-size bound"
            )
    if inputs_path.stat().st_size > int(profile["max_selection_input_payload_bytes"]):
        raise Tier2MonthlyEexEvaluationError(
            "selection input payload exceeds the preregistered byte-size bound"
        )


def _preflight_catalog_internal_sizes(
    catalog: Mapping[str, object],
    *,
    catalog_root: Path,
    selected_history: Path,
) -> None:
    limit = int(
        EXPECTED_SELECTION_INPUT_RESOURCE_PROFILE["max_eex_internal_file_bytes"]
    )
    for path in fold_module._catalog_internal_paths(
        catalog, catalog_root, selected_history
    ):
        if path == selected_history:
            continue
        if path.stat().st_size > limit:
            raise Tier2MonthlyEexEvaluationError(
                "EEX catalog internal file exceeds the preregistered byte-size bound"
            )


def _outer_paths(
    paths: Mapping[str, Path], catalog: Mapping[str, object]
) -> tuple[Path, ...]:
    internal = fold_module._catalog_internal_paths(
        catalog, paths["catalog"].parent, paths["history"]
    )
    return tuple(
        sorted(
            {*paths.values(), *internal},
            key=lambda path: str(path).casefold(),
        )
    )


def _bind_bytes(
    path: Path, raw: bytes, snapshot: Mapping[str, object]
) -> None:
    _require_equal(
        snapshot[str(path)][-1],
        hashlib.sha256(raw).hexdigest(),
        label=f"initial snapshot hash for {path.name}",
    )


def _current_trust_environment() -> dict[str, str]:
    values: dict[str, str] = {}
    for env_name in (
        MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV,
        TRUSTED_ACQUISITION_PUBLIC_KEY_ENV,
        TRUSTED_TIME_PUBLIC_KEY_ENV,
        TIMESTAMP_PUBLIC_KEY_ENV,
        SELECTION_INPUT_EXECUTION_PUBLIC_KEY_ENV,
        TIMESTAMP_JOURNAL_WITNESS_PUBLIC_KEY_ENV,
    ):
        value = os.environ.get(env_name, "").strip()
        if not value:
            raise Tier2MonthlyEexEvaluationError(f"{env_name} is required")
        values[env_name] = str(
            fold_module._absolute_regular_file(value, label=f"{env_name} trust anchor")
        )
    journal_id = os.environ.get(evaluation_module.TIMESTAMP_JOURNAL_ID_ENV, "").strip()
    if not journal_id:
        raise Tier2MonthlyEexEvaluationError(
            f"{evaluation_module.TIMESTAMP_JOURNAL_ID_ENV} is required"
        )
    values[evaluation_module.TIMESTAMP_JOURNAL_ID_ENV] = journal_id
    source_journal_id = os.environ.get(TRUSTED_TIME_JOURNAL_ID_ENV, "").strip()
    if not source_journal_id:
        raise Tier2MonthlyEexEvaluationError(
            f"{TRUSTED_TIME_JOURNAL_ID_ENV} is required"
        )
    values[TRUSTED_TIME_JOURNAL_ID_ENV] = source_journal_id
    head_value = os.environ.get(TIMESTAMP_JOURNAL_HEAD_PATH_ENV, "").strip()
    if not head_value:
        raise Tier2MonthlyEexEvaluationError(
            f"{TIMESTAMP_JOURNAL_HEAD_PATH_ENV} is required"
        )
    values[TIMESTAMP_JOURNAL_HEAD_PATH_ENV] = str(
        fold_module._absolute_regular_file(
            head_value, label="trusted timestamp journal head"
        )
    )
    current_head_id = os.environ.get(
        TIMESTAMP_JOURNAL_CURRENT_HEAD_ID_ENV, ""
    ).strip()
    _require_sha256(current_head_id, label="external current journal head ID")
    values[TIMESTAMP_JOURNAL_CURRENT_HEAD_ID_ENV] = current_head_id
    return values


def _exact_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise Tier2MonthlyEexEvaluationError(f"{label} must be a non-negative integer")
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


__all__ = [
    "VerifiedSelectionBaseInputEvidence",
    "verify_signed_selection_base_inputs",
]
