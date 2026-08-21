"""Path-only replay of one governed monthly EEX SELECTION data lineage.

The proof emitted by this boundary is deliberately not campaign eligible.  It
authenticates and replays source selection and row lineage only. Model outputs
remain unverified diagnostics until deterministic solver/baseline replay exists;
HOLDOUT remains unsupported until candidate freeze and campaign identity close.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass
import hashlib
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.calibration.cascading import count_hours
from pfc_shaping.calibration.monthly_forward_curve import product_periods
from pfc_shaping.calibration.tier2_monthly_eex_evaluation import (
    EXECUTION_ATTESTATION_SCHEMA,
    EXECUTION_PUBLIC_KEY_ENV,
    FULL_TIER2_STATUS,
    LOCAL_TIMEZONE,
    SCOPE,
    TIMESTAMP_PUBLIC_KEY_ENV,
    TIMESTAMP_RECEIPT_SCHEMA,
    ProductConstraint,
    Tier2MonthlyEexEvaluationError,
    _aware_utc,
    _canonical_json_bytes,
    _deep_freeze_json,
    _finite_number,
    _load_json_artifact,
    _mapping,
    _regular_file,
    _require_equal,
    _require_exact_keys,
    _require_sha256,
    _required_text,
    _sha256_json,
    _verify_exact_artifact_authority_bytes,
    assess_target_identifiability,
    product_constraint_vector,
    verify_governed_plan_artifact,
)
from pfc_shaping.data.acquisition_contract import (
    TRUSTED_ACQUISITION_PUBLIC_KEY_ENV,
    TRUSTED_TIME_PUBLIC_KEY_ENV,
)
from pfc_shaping.data.eex_historical_vintage import (
    VerifiedEexHistoricalVintageCatalog,
    verify_eex_historical_vintage_catalog,
)


FOLD_MANIFEST_SCHEMA = "tier2_monthly_eex_fold_evidence_manifest.v1"
FOLD_ROWS_SCHEMA = "tier2_monthly_eex_fold_rows.v1"
FOLD_RESULT_SCHEMA = "tier2_monthly_eex_fold_result_replay.v1"
CANDIDATE_GRID_SCHEMA = "tier2_monthly_eex_candidate_grid.v1"
FOLD_MANIFEST_ROLE = "tier2_monthly_eex_fold_evidence_manifest"
FOLD_STATUS = "DATA_LINEAGE_REPLAYED_MODEL_OUTPUT_UNVERIFIED"
HOLDOUT_UNSUPPORTED = "UNSUPPORTED_HOLDOUT_REQUIRES_VERIFIED_FREEZE"
REPRICING_TOLERANCE = 1e-9
NUMERIC_TOLERANCE = 1e-12

_ROWS_FILENAME = "fold_rows.v1.json"
_RESULT_FILENAME = "fold_result.v1.json"
_MANIFEST_FILENAME = "fold_manifest.v1.json"
_VERIFIED_FOLD_TOKEN = object()
_VERIFIED_REPLAY_CONTEXT_TOKEN = object()

_ROLE_NAMES = (
    "selection_training_context",
    "fold_historical_context",
    "fold_evaluation_snapshot_retained",
    "removed_revealing_quotes",
    "masked_target",
    "outside_grid_quotes",
    "diagnostic_later_quotes",
)


@dataclass(frozen=True)
class VerifiedSelectionFoldDataLineage:
    evidence_id: str
    artifact_sha256: str
    fold_id: str
    fold_spec_hash: str
    candidate_config_sha256: str
    catalog_id: str
    frame_semantic_sha256: str
    execution_key_id: str
    timestamp_key_id: str
    executed_at_utc: pd.Timestamp
    trusted_time_utc: pd.Timestamp
    diagnostic_result_sha256: str
    _verification_token: InitVar[object | None] = None

    def __post_init__(self, _verification_token: object | None) -> None:
        if _verification_token is not _VERIFIED_FOLD_TOKEN:
            raise Tier2MonthlyEexEvaluationError(
                "VerifiedSelectionFoldDataLineage requires path-based replay"
            )


@dataclass(frozen=True)
class _FoldReplayRow:
    date: object
    available_at: object
    market: str
    load_type: str
    product: str
    price: float
    snapshot_id: str
    row_hash: str
    quote_id: str


@dataclass(frozen=True)
class _VerifiedSelectionFoldReplayContext:
    proof: VerifiedSelectionFoldDataLineage
    plan_id: str
    plan_artifact_sha256: str
    candidate_grid_sha256: str
    evaluation_source_closure_sha256: str
    runtime_lock_sha256: str
    plan_trusted_time_utc: pd.Timestamp
    plan_governance_receipt_sha256: str
    plan_timestamp_receipt_sha256: str
    fold_execution_receipt_sha256: str
    fold_timestamp_receipt_sha256: str
    target_load_type: str
    plan_governance_key_id: str
    plan_timestamp_key_id: str
    fold_execution_key_id: str
    fold_timestamp_key_id: str
    acquisition_key_id: str
    source_timestamp_key_ids: tuple[str, ...]
    forbidden_non_time_authority_key_ids: tuple[str, ...]
    trusted_time_authority_key_ids: tuple[str, ...]
    historical_rows: tuple[_FoldReplayRow, ...]
    retained_rows: tuple[_FoldReplayRow, ...]
    delivery_months: tuple[str, ...]
    candidate_grid: Mapping[str, object]
    catalog_sha256: str
    history_sha256: str
    _verification_token: InitVar[object | None] = None

    def __post_init__(self, _verification_token: object | None) -> None:
        if _verification_token is not _VERIFIED_REPLAY_CONTEXT_TOKEN:
            raise Tier2MonthlyEexEvaluationError(
                "fold replay context requires same-snapshot path verification"
            )


def verify_signed_selection_fold_evidence(
    *,
    plan_path: str | Path,
    plan_governance_receipt_path: str | Path,
    plan_timestamp_receipt_path: str | Path,
    candidate_grid_path: str | Path,
    fold_manifest_path: str | Path,
    execution_attestation_path: str | Path,
    fold_timestamp_receipt_path: str | Path,
    eex_catalog_path: str | Path,
    eex_history_path: str | Path,
) -> VerifiedSelectionFoldDataLineage:
    """Authenticate and replay one exact SELECTION data-lineage package."""

    return _verify_signed_selection_fold_evidence_context(
        plan_path=plan_path,
        plan_governance_receipt_path=plan_governance_receipt_path,
        plan_timestamp_receipt_path=plan_timestamp_receipt_path,
        candidate_grid_path=candidate_grid_path,
        fold_manifest_path=fold_manifest_path,
        execution_attestation_path=execution_attestation_path,
        fold_timestamp_receipt_path=fold_timestamp_receipt_path,
        eex_catalog_path=eex_catalog_path,
        eex_history_path=eex_history_path,
    ).proof


def _verify_signed_selection_fold_evidence_context(
    *,
    plan_path: str | Path,
    plan_governance_receipt_path: str | Path,
    plan_timestamp_receipt_path: str | Path,
    candidate_grid_path: str | Path,
    fold_manifest_path: str | Path,
    execution_attestation_path: str | Path,
    fold_timestamp_receipt_path: str | Path,
    eex_catalog_path: str | Path,
    eex_history_path: str | Path,
) -> _VerifiedSelectionFoldReplayContext:
    """Return private consumable records from the same authenticated snapshot."""

    paths = {
        name: _absolute_regular_file(value, label=name)
        for name, value in {
            "plan_path": plan_path,
            "plan_governance_receipt_path": plan_governance_receipt_path,
            "plan_timestamp_receipt_path": plan_timestamp_receipt_path,
            "candidate_grid_path": candidate_grid_path,
            "fold_manifest_path": fold_manifest_path,
            "execution_attestation_path": execution_attestation_path,
            "fold_timestamp_receipt_path": fold_timestamp_receipt_path,
            "eex_catalog_path": eex_catalog_path,
            "eex_history_path": eex_history_path,
        }.items()
    }
    if paths["fold_manifest_path"].name != _MANIFEST_FILENAME:
        raise Tier2MonthlyEexEvaluationError(
            f"fold manifest must be named {_MANIFEST_FILENAME}"
        )

    plan = verify_governed_plan_artifact(
        paths["plan_path"],
        governance_receipt_path=paths["plan_governance_receipt_path"],
        timestamp_receipt_path=paths["plan_timestamp_receipt_path"],
    )
    package_root = paths["fold_manifest_path"].parent
    package_snapshot = _package_snapshot(package_root)
    manifest_path, manifest, manifest_bytes = _load_json_artifact(
        paths["fold_manifest_path"]
    )
    _require_equal(
        package_snapshot[_MANIFEST_FILENAME][-1],
        hashlib.sha256(manifest_bytes).hexdigest(),
        label="manifest initial package snapshot hash",
    )
    _require_canonical_json_bytes(manifest, manifest_bytes, label="fold manifest")

    _, execution_receipt, execution_receipt_bytes = _load_json_artifact(
        paths["execution_attestation_path"]
    )
    _, timestamp_receipt, timestamp_receipt_bytes = _load_json_artifact(
        paths["fold_timestamp_receipt_path"]
    )
    fold_time = _verify_exact_artifact_authority_bytes(
        timestamp_receipt,
        artifact_path=manifest_path,
        artifact_bytes=manifest_bytes,
        expected_role=FOLD_MANIFEST_ROLE,
        expected_artifact_schema=FOLD_MANIFEST_SCHEMA,
        receipt_schema=TIMESTAMP_RECEIPT_SCHEMA,
        signature_field="trusted_time_attestation",
        trusted_key_env=TIMESTAMP_PUBLIC_KEY_ENV,
        event_time_field="received_at_utc",
        authority_id_field="authority_id",
        id_field="receipt_id",
        journal_required=True,
    )
    execution = _verify_exact_artifact_authority_bytes(
        execution_receipt,
        artifact_path=manifest_path,
        artifact_bytes=manifest_bytes,
        expected_role=FOLD_MANIFEST_ROLE,
        expected_artifact_schema=FOLD_MANIFEST_SCHEMA,
        receipt_schema=EXECUTION_ATTESTATION_SCHEMA,
        signature_field="execution_attestation",
        trusted_key_env=EXECUTION_PUBLIC_KEY_ENV,
        event_time_field="executed_at_utc",
        authority_id_field="authority_id",
        id_field="attestation_id",
        journal_required=False,
    )
    if execution.authority_key_id in {
        plan.governance_key_id,
        plan.timestamp_key_id,
        fold_time.authority_key_id,
    }:
        raise Tier2MonthlyEexEvaluationError(
            "execution authority is not independent of governance/trusted time"
        )
    if not (plan.trusted_time_utc <= execution.event_time_utc <= fold_time.event_time_utc):
        raise Tier2MonthlyEexEvaluationError(
            "fold evidence violates preregistration/execution/trusted-time order"
        )

    grid = _verify_candidate_grid(paths["candidate_grid_path"], plan.payload)
    frame, catalog_evidence, source_authorities = _verify_eex_paths(
        paths["eex_catalog_path"], paths["eex_history_path"]
    )
    _verify_authority_separation(
        plan=plan,
        execution_key_id=execution.authority_key_id,
        fold_timestamp_key_id=fold_time.authority_key_id,
        source_authorities=source_authorities,
    )
    frame_hash = _frame_semantic_sha256(frame)

    manifest_valid = _validate_manifest(
        manifest,
        manifest_bytes=manifest_bytes,
        plan=plan,
        grid=grid,
        catalog=catalog_evidence,
        frame_semantic_sha256=frame_hash,
        source_authorities=source_authorities,
    )
    fold_spec = _find_selection_fold(plan.payload, str(manifest_valid["fold_id"]))
    _bind_manifest_to_fold(manifest_valid, fold_spec)

    rows_path, rows_payload, _ = _load_bound_artifact(
        manifest_path.parent,
        manifest_valid,
        role="fold_rows",
        expected_name=_ROWS_FILENAME,
        expected_schema=FOLD_ROWS_SCHEMA,
    )
    _, result_payload, result_bytes = _load_bound_artifact(
        manifest_path.parent,
        manifest_valid,
        role="fold_result",
        expected_name=_RESULT_FILENAME,
        expected_schema=FOLD_RESULT_SCHEMA,
    )
    if rows_path.parent != manifest_path.parent:
        raise Tier2MonthlyEexEvaluationError("fold rows escaped package root")

    expected_roles, target_row, retained_rows, delivery_months = _derive_fold_rows(
        frame,
        plan=plan,
        fold_spec=fold_spec,
    )
    _validate_rows_payload(
        rows_payload,
        expected_roles=expected_roles,
        frame_semantic_sha256=frame_hash,
    )
    _validate_result_payload(
        result_payload,
        plan=plan,
        grid=grid,
        manifest=manifest_valid,
        target_row=target_row,
        retained_rows=retained_rows,
        delivery_months=delivery_months,
    )
    proof = VerifiedSelectionFoldDataLineage(
        evidence_id=str(manifest_valid["evidence_id"]),
        artifact_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        fold_id=str(manifest_valid["fold_id"]),
        fold_spec_hash=str(manifest_valid["fold_spec_hash"]),
        candidate_config_sha256=str(manifest_valid["candidate_config_sha256"]),
        catalog_id=catalog_evidence.catalog_id,
        frame_semantic_sha256=frame_hash,
        execution_key_id=execution.authority_key_id,
        timestamp_key_id=fold_time.authority_key_id,
        executed_at_utc=execution.event_time_utc,
        trusted_time_utc=fold_time.event_time_utc,
        diagnostic_result_sha256=hashlib.sha256(result_bytes).hexdigest(),
        _verification_token=_VERIFIED_FOLD_TOKEN,
    )
    historical_rows = _replay_rows(
        frame,
        expected_roles["fold_historical_context"]["row_hashes"],
        label="fold historical context",
    )
    retained_replay_rows = _replay_rows(
        frame,
        expected_roles["fold_evaluation_snapshot_retained"]["row_hashes"],
        label="fold retained snapshot",
    )
    frozen_grid = _deep_freeze_json(grid)
    if _package_snapshot(package_root) != package_snapshot:
        raise Tier2MonthlyEexEvaluationError(
            "fold package changed during verification"
        )
    return _VerifiedSelectionFoldReplayContext(
        proof=proof,
        plan_id=plan.plan_id,
        plan_artifact_sha256=plan.artifact_sha256,
        candidate_grid_sha256=hashlib.sha256(grid["_artifact_bytes"]).hexdigest(),
        evaluation_source_closure_sha256=str(
            manifest_valid["evaluation_source_closure_sha256"]
        ),
        runtime_lock_sha256=str(manifest_valid["runtime_lock_sha256"]),
        plan_trusted_time_utc=plan.trusted_time_utc,
        plan_governance_receipt_sha256=plan.governance_receipt_sha256,
        plan_timestamp_receipt_sha256=plan.timestamp_receipt_sha256,
        fold_execution_receipt_sha256=hashlib.sha256(
            execution_receipt_bytes
        ).hexdigest(),
        fold_timestamp_receipt_sha256=hashlib.sha256(
            timestamp_receipt_bytes
        ).hexdigest(),
        target_load_type=str(target_row["load_type"]),
        plan_governance_key_id=plan.governance_key_id,
        plan_timestamp_key_id=plan.timestamp_key_id,
        fold_execution_key_id=proof.execution_key_id,
        fold_timestamp_key_id=proof.timestamp_key_id,
        acquisition_key_id=str(source_authorities["acquisition_key_id"]),
        source_timestamp_key_ids=tuple(
            sorted(str(value) for value in source_authorities["timestamp_key_ids"])
        ),
        forbidden_non_time_authority_key_ids=tuple(
            sorted(
                {
                    plan.governance_key_id,
                    proof.execution_key_id,
                    str(source_authorities["acquisition_key_id"]),
                }
            )
        ),
        trusted_time_authority_key_ids=tuple(
            sorted(
                {
                    plan.timestamp_key_id,
                    proof.timestamp_key_id,
                    *(
                        str(value)
                        for value in source_authorities["timestamp_key_ids"]
                    ),
                }
            )
        ),
        historical_rows=historical_rows,
        retained_rows=retained_replay_rows,
        delivery_months=delivery_months,
        candidate_grid=frozen_grid,  # type: ignore[arg-type]
        catalog_sha256=catalog_evidence.catalog_sha256,
        history_sha256=catalog_evidence.history_sha256,
        _verification_token=_VERIFIED_REPLAY_CONTEXT_TOKEN,
    )


def _validate_manifest(
    manifest: Mapping[str, object],
    *,
    manifest_bytes: bytes,
    plan: object,
    grid: Mapping[str, object],
    catalog: VerifiedEexHistoricalVintageCatalog,
    frame_semantic_sha256: str,
    source_authorities: Mapping[str, object],
) -> dict[str, object]:
    required = {
        "schema_version",
        "evidence_id",
        "scope",
        "full_tier2_pfc_status",
        "data_lineage_verified",
        "model_output_verified",
        "campaign_eligible",
        "production_approved",
        "status",
        "plan_id",
        "plan_artifact_sha256",
        "fold_id",
        "fold_spec_hash",
        "campaign_type",
        "origin_id",
        "fold_as_of_utc",
        "candidate_grid_sha256",
        "candidate_config_sha256",
        "evaluation_source_closure_sha256",
        "runtime_lock_sha256",
        "catalog_root",
        "frame_semantic_sha256",
        "source_authorities",
        "artifacts",
    }
    _require_exact_keys(manifest, required, label="fold manifest")
    _require_equal(manifest.get("schema_version"), FOLD_MANIFEST_SCHEMA, label="manifest schema")
    _require_equal(manifest.get("scope"), SCOPE, label="fold scope")
    _require_equal(
        manifest.get("full_tier2_pfc_status"),
        FULL_TIER2_STATUS,
        label="full Tier-2 status",
    )
    _require_equal(
        manifest.get("data_lineage_verified"), True, label="data lineage verification"
    )
    _require_equal(
        manifest.get("model_output_verified"), False, label="model output verification"
    )
    _require_equal(manifest.get("campaign_eligible"), False, label="campaign eligibility")
    _require_equal(manifest.get("production_approved"), False, label="production approval")
    _require_equal(manifest.get("status"), FOLD_STATUS, label="fold status")
    if manifest.get("campaign_type") == "HOLDOUT":
        raise Tier2MonthlyEexEvaluationError(HOLDOUT_UNSUPPORTED)
    _require_equal(manifest.get("campaign_type"), "SELECTION", label="fold campaign")
    _require_equal(manifest.get("plan_id"), plan.plan_id, label="fold plan_id")
    _require_equal(
        manifest.get("plan_artifact_sha256"),
        plan.artifact_sha256,
        label="fold plan artifact hash",
    )
    _require_equal(
        manifest.get("candidate_grid_sha256"),
        hashlib.sha256(grid["_artifact_bytes"]).hexdigest(),
        label="candidate grid hash",
    )
    config_hash = _require_sha256(
        manifest.get("candidate_config_sha256"), label="candidate config hash"
    )
    if config_hash not in grid["_config_hashes"]:
        raise Tier2MonthlyEexEvaluationError("candidate config is not in governed grid")
    _require_equal(
        manifest.get("evaluation_source_closure_sha256"),
        plan.payload["evaluation_source_closure_sha256"],
        label="evaluation source closure",
    )
    _require_equal(
        manifest.get("runtime_lock_sha256"),
        plan.payload["runtime_lock_sha256"],
        label="runtime lock",
    )
    expected_root = {
        "catalog_id": catalog.catalog_id,
        "catalog_sha256": catalog.catalog_sha256,
        "history_sha256": catalog.history_sha256,
    }
    expected_root["root_sha256"] = _sha256_json(
        {
            "schema_version": "tier2_monthly_eex_catalog_root.v1",
            **expected_root,
        }
    )
    _require_equal(manifest.get("catalog_root"), expected_root, label="catalog root")
    _require_equal(
        manifest.get("catalog_root"),
        _json_thaw(plan.payload["selection_eex_root"]),
        label="selection catalog root",
    )
    _require_equal(
        manifest.get("frame_semantic_sha256"),
        frame_semantic_sha256,
        label="frame semantic hash",
    )
    _require_equal(
        manifest.get("source_authorities"),
        source_authorities,
        label="source authorities",
    )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or [
        item.get("role") if isinstance(item, Mapping) else None for item in artifacts
    ] != ["fold_rows", "fold_result"]:
        raise Tier2MonthlyEexEvaluationError(
            "manifest artifact roles must be exactly fold_rows then fold_result"
        )
    unsigned = dict(manifest)
    evidence_id = str(unsigned.pop("evidence_id", ""))
    if evidence_id != _sha256_json(unsigned):
        raise Tier2MonthlyEexEvaluationError("fold evidence_id mismatch")
    return dict(manifest)


def _find_selection_fold(
    plan_payload: Mapping[str, object], fold_id: str
) -> Mapping[str, object]:
    selection = plan_payload.get("selection_folds")
    if not isinstance(selection, Sequence):
        raise Tier2MonthlyEexEvaluationError("verified plan selection folds are unavailable")
    matches = [item for item in selection if isinstance(item, Mapping) and item.get("fold_id") == fold_id]
    if len(matches) != 1:
        holdout = plan_payload.get("holdout_folds")
        if isinstance(holdout, Sequence) and any(
            isinstance(item, Mapping) and item.get("fold_id") == fold_id for item in holdout
        ):
            raise Tier2MonthlyEexEvaluationError(HOLDOUT_UNSUPPORTED)
        raise Tier2MonthlyEexEvaluationError("fold is not preregistered in SELECTION")
    return matches[0]


def _bind_manifest_to_fold(
    manifest: Mapping[str, object], fold: Mapping[str, object]
) -> None:
    for field in (
        "fold_id",
        "fold_spec_hash",
        "campaign_type",
        "origin_id",
        "fold_as_of_utc",
    ):
        _require_equal(manifest.get(field), fold.get(field), label=f"fold binding {field}")


def _verify_candidate_grid(
    path: Path, plan_payload: Mapping[str, object]
) -> dict[str, object]:
    _, grid, raw = _load_json_artifact(path)
    _require_canonical_json_bytes(grid, raw, label="candidate grid")
    _require_equal(
        hashlib.sha256(raw).hexdigest(),
        plan_payload["candidate_grid_sha256"],
        label="candidate grid bytes",
    )
    required = {"schema_version", "grid_id", "market", "timezone", "configs"}
    _require_exact_keys(grid, required, label="candidate grid")
    _require_equal(grid.get("schema_version"), CANDIDATE_GRID_SCHEMA, label="grid schema")
    _require_equal(grid.get("market"), "CH", label="grid market")
    _require_equal(grid.get("timezone"), LOCAL_TIMEZONE, label="grid timezone")
    configs = grid.get("configs")
    if not isinstance(configs, list) or not configs:
        raise Tier2MonthlyEexEvaluationError("candidate grid configs are empty")
    hashes: list[str] = []
    for entry in configs:
        if not isinstance(entry, Mapping):
            raise Tier2MonthlyEexEvaluationError("candidate grid entry is not an object")
        _require_exact_keys(entry, {"config_sha256", "config"}, label="candidate grid entry")
        config = _mapping(entry.get("config"), label="candidate config")
        config_hash = _require_sha256(entry.get("config_sha256"), label="candidate config hash")
        _require_equal(config_hash, _sha256_json(config), label="candidate config content hash")
        hashes.append(config_hash)
    if hashes != sorted(set(hashes)):
        raise Tier2MonthlyEexEvaluationError("candidate grid configs are not sorted and unique")
    _require_equal(
        hashes,
        list(plan_payload["candidate_config_sha256_allowlist"]),
        label="candidate grid allowlist",
    )
    unsigned = dict(grid)
    grid_id = str(unsigned.pop("grid_id", ""))
    if grid_id != _sha256_json(unsigned):
        raise Tier2MonthlyEexEvaluationError("candidate grid_id mismatch")
    return {**grid, "_artifact_bytes": raw, "_config_hashes": tuple(hashes)}


def _verify_eex_paths(
    catalog_path: Path, history_path: Path
) -> tuple[pd.DataFrame, VerifiedEexHistoricalVintageCatalog, dict[str, object]]:
    for env_name, label in (
        (TRUSTED_ACQUISITION_PUBLIC_KEY_ENV, "EEX acquisition trust anchor"),
        (TRUSTED_TIME_PUBLIC_KEY_ENV, "EEX trusted-time trust anchor"),
    ):
        value = os.environ.get(env_name, "").strip()
        if not value:
            raise Tier2MonthlyEexEvaluationError(f"{env_name} is required")
        _absolute_regular_file(value, label=label)
    _, catalog, catalog_bytes = _load_json_artifact(catalog_path)
    history_file = _absolute_regular_file(history_path, label="eex_history_path")
    history_bytes = history_file.read_bytes()
    internal_paths = _catalog_internal_paths(
        catalog, catalog_path.parent, history_file
    )
    internal_snapshot = _file_set_snapshot(internal_paths)
    frame, evidence = verify_eex_historical_vintage_catalog(
        catalog,
        catalog_path=catalog_path,
        history_path=history_file,
    )
    if _file_set_snapshot(internal_paths) != internal_snapshot:
        raise Tier2MonthlyEexEvaluationError(
            "EEX catalog inputs changed during verification"
        )
    _require_equal(
        evidence.catalog_sha256,
        hashlib.sha256(catalog_bytes).hexdigest(),
        label="catalog single-snapshot hash",
    )
    _require_equal(
        evidence.history_sha256,
        hashlib.sha256(history_bytes).hexdigest(),
        label="history single-snapshot hash",
    )
    acquisition = _mapping(
        catalog.get("acquisition_attestation"), label="catalog acquisition attestation"
    )
    acquisition_key = _require_sha256(
        acquisition.get("key_id"), label="catalog acquisition key"
    )
    source_documents = catalog.get("source_documents")
    if not isinstance(source_documents, list) or not source_documents:
        raise Tier2MonthlyEexEvaluationError("catalog source documents are missing")
    timestamp_keys: set[str] = set()
    timestamp_journals: set[str] = set()
    for entry in source_documents:
        receipt = _mapping(
            _mapping(entry, label="catalog source document").get("trusted_time_receipt"),
            label="source trusted-time receipt",
        )
        attestation = _mapping(
            receipt.get("trusted_time_attestation"), label="source time attestation"
        )
        timestamp_keys.add(
            _require_sha256(attestation.get("key_id"), label="source timestamp key")
        )
        timestamp_journals.add(
            _required_text(receipt.get("journal_id"), label="source timestamp journal")
        )
    authorities = {
        "acquisition_key_id": acquisition_key,
        "timestamp_key_ids": sorted(timestamp_keys),
        "timestamp_journal_ids": sorted(timestamp_journals),
    }
    return frame, evidence, authorities


def _catalog_internal_paths(
    catalog: Mapping[str, object], root: Path, selected_history: Path
) -> tuple[Path, ...]:
    history = _mapping(catalog.get("history_parquet"), label="catalog history")
    resolved_history = _resolve_catalog_regular(root, history.get("path"), label="catalog history")
    if resolved_history != selected_history:
        raise Tier2MonthlyEexEvaluationError("catalog history path differs from selected history")
    resolved = {resolved_history}
    documents = catalog.get("source_documents")
    if not isinstance(documents, list):
        raise Tier2MonthlyEexEvaluationError("catalog source documents are invalid")
    for raw in documents:
        entry = _mapping(raw, label="catalog source document")
        resolved.add(
            _resolve_catalog_regular(
                root, entry.get("path"), label="catalog source document"
            )
        )
        resolved.add(
            _resolve_catalog_regular(
                root, entry.get("parser_code_path"), label="catalog parser code"
            )
        )
    return tuple(sorted(resolved, key=lambda path: str(path).casefold()))


def _resolve_catalog_regular(root: Path, value: object, *, label: str) -> Path:
    text = _required_text(value, label=label)
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts:
        raise Tier2MonthlyEexEvaluationError(f"{label} path is not confined")
    resolved = _regular_file(root / relative)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise Tier2MonthlyEexEvaluationError(f"{label} escaped catalog root") from exc
    return resolved


def _derive_fold_rows(
    frame: pd.DataFrame,
    *,
    plan: object,
    fold_spec: Mapping[str, object],
) -> tuple[dict[str, dict[str, object]], Mapping[str, object], pd.DataFrame, tuple[str, ...]]:
    origin = _aware_utc(fold_spec["fold_as_of_utc"], label="fold_as_of_utc")
    ch = frame[frame["market"].astype(str).eq("CH")].copy()
    eligible_snapshot_rows = ch[ch["available_at"] <= origin]
    if eligible_snapshot_rows.empty:
        raise Tier2MonthlyEexEvaluationError("UNSUPPORTED_SNAPSHOT_OR_TARGET_UNAVAILABLE")
    latest_available = eligible_snapshot_rows["available_at"].max()
    latest = eligible_snapshot_rows[
        eligible_snapshot_rows["available_at"].eq(latest_available)
    ]
    snapshot_ids = sorted(set(latest["snapshot_id"].astype(str)))
    if len(snapshot_ids) != 1:
        raise Tier2MonthlyEexEvaluationError("UNSUPPORTED_SNAPSHOT_OR_TARGET_UNAVAILABLE")
    snapshot = ch[ch["snapshot_id"].astype(str).eq(snapshot_ids[0])].copy()
    target_matches = snapshot[
        snapshot["product"].astype(str).eq(str(fold_spec["target_product"]))
        & snapshot["load_type"].astype(str).eq(str(fold_spec["target_load_type"]))
    ]
    if len(target_matches) != 1:
        raise Tier2MonthlyEexEvaluationError("UNSUPPORTED_SNAPSHOT_OR_TARGET_UNAVAILABLE")
    target = target_matches.iloc[0].to_dict()
    target_months = set(product_periods(str(target["product"])))
    overlap = snapshot["product"].astype(str).map(
        lambda product: bool(target_months.intersection(product_periods(product)))
    )
    target_hash = str(target["row_hash"])
    retained = snapshot[~overlap].copy()
    removed = snapshot[overlap & ~snapshot["row_hash"].astype(str).eq(target_hash)].copy()
    outside = snapshot.iloc[0:0].copy()
    later = frame[
        frame["market"].astype(str).eq("CH")
        & frame["load_type"].astype(str).eq(str(target["load_type"]))
        & frame["product"].astype(str).eq(str(target["product"]))
        & (frame["available_at"] > origin)
    ].copy()

    selection_cutoff = min(plan.trusted_time_utc, origin)
    selection_context = _context_rows(
        frame,
        cutoff=selection_cutoff,
        evaluation_snapshot_id=snapshot_ids[0],
        evaluation_document_id=str(target["source_document_id"]),
    )
    fold_context = _context_rows(
        frame,
        cutoff=origin,
        evaluation_snapshot_id=snapshot_ids[0],
        evaluation_document_id=str(target["source_document_id"]),
    )
    role_frames = {
        "selection_training_context": selection_context,
        "fold_historical_context": fold_context,
        "fold_evaluation_snapshot_retained": retained,
        "removed_revealing_quotes": removed,
        "masked_target": target_matches,
        "outside_grid_quotes": outside,
        "diagnostic_later_quotes": later,
    }
    inventories = {
        role: _inventory(role, role_frames[role]) for role in _ROLE_NAMES
    }
    _validate_role_disjointness(role_frames)
    retained_same_load = retained[
        retained["market"].astype(str).eq("CH")
        & retained["load_type"].astype(str).eq(str(target["load_type"]))
    ]
    delivery_periods = sorted(
        {
            month
            for product in [
                str(target["product"]),
                *retained_same_load["product"].astype(str).tolist(),
            ]
            for month in product_periods(product)
        }
    )
    if not delivery_periods:
        raise Tier2MonthlyEexEvaluationError("fold delivery grid is empty")
    contiguous = pd.period_range(delivery_periods[0], delivery_periods[-1], freq="M")
    return inventories, target, retained, tuple(str(month) for month in contiguous)


def _context_rows(
    frame: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    evaluation_snapshot_id: str,
    evaluation_document_id: str,
) -> pd.DataFrame:
    local = cutoff.tz_convert(LOCAL_TIMEZONE)
    local_month_start = pd.Timestamp(year=local.year, month=local.month, day=1)
    upper = local_month_start - pd.DateOffset(months=1)
    lower = upper - pd.DateOffset(months=72)
    date_values = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    eligible = frame[
        (frame["available_at"] <= cutoff)
        & (date_values >= lower)
        & (date_values < upper)
        & ~frame["snapshot_id"].astype(str).eq(evaluation_snapshot_id)
        & ~frame["source_document_id"].astype(str).eq(evaluation_document_id)
    ].copy()
    if eligible.empty:
        return eligible
    identity = ["date", "market", "load_type", "product"]
    ordered = eligible.sort_values(
        [*identity, "revision_sequence", "available_at", "quote_id"],
        kind="mergesort",
    )
    return ordered.groupby(identity, sort=True, as_index=False).tail(1)


def _validate_role_disjointness(role_frames: Mapping[str, pd.DataFrame]) -> None:
    context_roles = {"selection_training_context", "fold_historical_context"}
    non_context = [role for role in _ROLE_NAMES if role not in context_roles]
    seen: dict[str, str] = {}
    for role in non_context:
        for column in ("row_hash", "quote_id", "revision_id"):
            for value in role_frames[role][column].astype(str):
                key = f"{column}:{value}"
                if key in seen:
                    raise Tier2MonthlyEexEvaluationError(
                        f"fold roles {seen[key]} and {role} overlap on {column}"
                    )
                seen[key] = role
    evaluation_snapshot_ids = set(
        pd.concat(
            [
                role_frames["fold_evaluation_snapshot_retained"],
                role_frames["removed_revealing_quotes"],
                role_frames["masked_target"],
            ],
            ignore_index=True,
        )["snapshot_id"].astype(str)
    )
    evaluation_documents = set(
        pd.concat(
            [
                role_frames["fold_evaluation_snapshot_retained"],
                role_frames["removed_revealing_quotes"],
                role_frames["masked_target"],
            ],
            ignore_index=True,
        )["source_document_id"].astype(str)
    )
    for role in context_roles:
        if evaluation_snapshot_ids.intersection(
            role_frames[role]["snapshot_id"].astype(str)
        ) or evaluation_documents.intersection(
            role_frames[role]["source_document_id"].astype(str)
        ):
            raise Tier2MonthlyEexEvaluationError(
                f"{role} overlaps evaluation snapshot/document"
            )


def _inventory(role: str, frame: pd.DataFrame) -> dict[str, object]:
    ordered = frame.sort_values(
        ["available_at", "snapshot_id", "market", "load_type", "product", "row_hash"],
        kind="mergesort",
    )
    inventory: dict[str, object] = {
        "row_count": len(ordered),
        "row_hashes": ordered["row_hash"].astype(str).tolist(),
        "quote_ids": ordered["quote_id"].astype(str).tolist(),
        "revision_ids": ordered["revision_id"].astype(str).tolist(),
        "snapshot_ids": ordered["snapshot_id"].astype(str).tolist(),
        "source_document_ids": ordered["source_document_id"].astype(str).tolist(),
        "source_document_sha256": ordered["source_document_sha256"].astype(str).tolist(),
        "acquisition_ids": ordered["acquisition_id"].astype(str).tolist(),
        "available_at_utc": [pd.Timestamp(value).isoformat() for value in ordered["available_at"]],
    }
    inventory["inventory_sha256"] = _sha256_json(
        {
            "schema_version": "tier2_monthly_eex_role_inventory.v1",
            "role": role,
            **inventory,
        }
    )
    return inventory


def _replay_rows(
    frame: pd.DataFrame,
    row_hashes: object,
    *,
    label: str,
) -> tuple[_FoldReplayRow, ...]:
    if not isinstance(row_hashes, list) or any(
        type(value) is not str or not value for value in row_hashes
    ):
        raise Tier2MonthlyEexEvaluationError(f"{label} row hashes are invalid")
    if len(row_hashes) != len(set(row_hashes)):
        raise Tier2MonthlyEexEvaluationError(f"{label} row hashes contain duplicates")
    indexed = frame.set_index(frame["row_hash"].astype(str), drop=False)
    if indexed.index.has_duplicates:
        raise Tier2MonthlyEexEvaluationError("verified frame contains duplicate row hashes")
    missing = [row_hash for row_hash in row_hashes if row_hash not in indexed.index]
    if missing:
        raise Tier2MonthlyEexEvaluationError(f"{label} rows escaped the verified frame")
    records: list[_FoldReplayRow] = []
    for row_hash in row_hashes:
        row = indexed.loc[row_hash]
        records.append(
            _FoldReplayRow(
                date=row["date"],
                available_at=row["available_at"],
                market=str(row["market"]),
                load_type=str(row["load_type"]),
                product=str(row["product"]),
                price=float(row["price"]),
                snapshot_id=str(row["snapshot_id"]),
                row_hash=str(row["row_hash"]),
                quote_id=str(row["quote_id"]),
            )
        )
    return tuple(records)


def _validate_rows_payload(
    payload: Mapping[str, object],
    *,
    expected_roles: Mapping[str, Mapping[str, object]],
    frame_semantic_sha256: str,
) -> None:
    required = {
        "schema_version",
        "frame_semantic_sha256",
        "roles",
        "permitted_candidate_input_row_hashes",
    }
    _require_exact_keys(payload, required, label="fold rows")
    _require_equal(payload.get("schema_version"), FOLD_ROWS_SCHEMA, label="fold rows schema")
    _require_equal(
        payload.get("frame_semantic_sha256"),
        frame_semantic_sha256,
        label="fold rows frame hash",
    )
    _require_equal(payload.get("roles"), expected_roles, label="fold role inventories")
    expected_inputs = [
        *expected_roles["fold_historical_context"]["row_hashes"],
        *expected_roles["fold_evaluation_snapshot_retained"]["row_hashes"],
    ]
    if len(expected_inputs) != len(set(expected_inputs)):
        raise Tier2MonthlyEexEvaluationError(
            "permitted candidate input inventory contains duplicates"
        )
    _require_equal(
        payload.get("permitted_candidate_input_row_hashes"),
        expected_inputs,
        label="permitted candidate input row inventory",
    )


def _validate_result_payload(
    payload: Mapping[str, object],
    *,
    plan: object,
    grid: Mapping[str, object],
    manifest: Mapping[str, object],
    target_row: Mapping[str, object],
    retained_rows: pd.DataFrame,
    delivery_months: tuple[str, ...],
) -> dict[str, object]:
    required = {
        "schema_version",
        "status",
        "data_lineage_verified",
        "model_output_verified",
        "campaign_eligible",
        "production_approved",
        "candidate_config_sha256",
        "baseline_config_sha256",
        "delivery_months",
        "candidate_segments",
        "baseline_segments",
        "reported_metrics",
    }
    _require_exact_keys(payload, required, label="fold result")
    _require_equal(payload.get("schema_version"), FOLD_RESULT_SCHEMA, label="fold result schema")
    _require_equal(payload.get("status"), FOLD_STATUS, label="fold result status")
    _require_equal(
        payload.get("data_lineage_verified"), True, label="result data lineage verification"
    )
    _require_equal(
        payload.get("model_output_verified"), False, label="result model output verification"
    )
    _require_equal(payload.get("campaign_eligible"), False, label="result campaign eligibility")
    _require_equal(payload.get("production_approved"), False, label="result production approval")
    _require_equal(
        payload.get("candidate_config_sha256"),
        manifest["candidate_config_sha256"],
        label="result candidate config",
    )
    _require_equal(
        payload.get("baseline_config_sha256"),
        _sha256_json(_json_thaw(plan.payload["baseline"])),
        label="baseline config hash",
    )
    _require_equal(payload.get("delivery_months"), list(delivery_months), label="delivery grid")
    candidate_values, candidate_conservation = _segment_values(
        payload.get("candidate_segments"), delivery_months=delivery_months, label="candidate"
    )
    baseline_values, baseline_conservation = _segment_values(
        payload.get("baseline_segments"), delivery_months=delivery_months, label="baseline"
    )
    target = ProductConstraint(str(target_row["product"]), str(target_row["load_type"]))
    retained_constraints = [
        ProductConstraint(str(row.product), str(row.load_type))
        for row in retained_rows.itertuples(index=False)
    ]
    assessment = assess_target_identifiability(target=target, retained=retained_constraints)
    if assessment.status != "IDENTIFIABLE":
        raise Tier2MonthlyEexEvaluationError(assessment.status)
    target_vector = product_constraint_vector(target, delivery_months=delivery_months)
    candidate_prediction = float(np.dot(target_vector, candidate_values))
    baseline_prediction = float(np.dot(target_vector, baseline_values))
    target_price = float(target_row["price"])
    candidate_error = candidate_prediction - target_price
    baseline_error = baseline_prediction - target_price
    candidate_residual = _repricing_max(
        retained_rows, candidate_values, delivery_months=delivery_months
    )
    baseline_residual = _repricing_max(
        retained_rows, baseline_values, delivery_months=delivery_months
    )
    if max(candidate_residual, baseline_residual) > REPRICING_TOLERANCE:
        raise Tier2MonthlyEexEvaluationError("fold retained quote repricing exceeds tolerance")
    retained_matrix = [
        product_constraint_vector(item, delivery_months=delivery_months).tolist()
        for item in retained_constraints
    ]
    expected = {
        "target_quote_id": str(target_row["quote_id"]),
        "target_row_hash": str(target_row["row_hash"]),
        "target_product": str(target_row["product"]),
        "target_load_type": str(target_row["load_type"]),
        "target_price": target_price,
        "target_weight": float(_product_delivery_hours(target)),
        "candidate_prediction": candidate_prediction,
        "baseline_prediction": baseline_prediction,
        "candidate_signed_error": candidate_error,
        "baseline_signed_error": baseline_error,
        "candidate_abs_error": abs(candidate_error),
        "baseline_abs_error": abs(baseline_error),
        "candidate_squared_error": candidate_error**2,
        "baseline_squared_error": baseline_error**2,
        "candidate_repricing_max_abs_error": candidate_residual,
        "baseline_repricing_max_abs_error": baseline_residual,
        "candidate_conservation_max_abs_error": candidate_conservation,
        "baseline_conservation_max_abs_error": baseline_conservation,
        "target_vector_sha256": _sha256_json(target_vector.tolist()),
        "retained_matrix_sha256": _sha256_json(retained_matrix),
        "identifiability_status": assessment.status,
        "retained_rank": assessment.retained_rank,
        "augmented_rank": assessment.augmented_rank,
        "retained_condition_number": assessment.retained_condition_number,
        "augmented_condition_number": assessment.augmented_condition_number,
    }
    _require_numeric_mapping_equal(
        payload.get("reported_metrics"), expected, label="reported fold metrics"
    )
    return expected


def _segment_values(
    value: object,
    *,
    delivery_months: tuple[str, ...],
    label: str,
) -> tuple[np.ndarray, float]:
    if not isinstance(value, list) or len(value) != len(delivery_months):
        raise Tier2MonthlyEexEvaluationError(f"{label} segment grid mismatch")
    output: list[float] = []
    max_conservation = 0.0
    for expected_month, raw in zip(delivery_months, value, strict=True):
        entry = _mapping(raw, label=f"{label} segment")
        _require_exact_keys(
            entry,
            {"month", "peak_price", "offpeak_price", "base_price"},
            label=f"{label} segment",
        )
        _require_equal(entry.get("month"), expected_month, label=f"{label} segment month")
        peak = _finite_number(entry.get("peak_price"), label=f"{label} peak price")
        offpeak = _finite_number(entry.get("offpeak_price"), label=f"{label} offpeak price")
        base = _finite_number(entry.get("base_price"), label=f"{label} base price")
        period = pd.Period(expected_month, freq="M")
        total_hours, peak_hours, offpeak_hours = count_hours(
            int(period.year), int(period.month), int(period.month), tz=LOCAL_TIMEZONE, country="CH"
        )
        implied_base = (peak * peak_hours + offpeak * offpeak_hours) / total_hours
        max_conservation = max(max_conservation, abs(base - implied_base))
        output.extend([peak, offpeak])
    if max_conservation > REPRICING_TOLERANCE:
        raise Tier2MonthlyEexEvaluationError(f"{label} monthly conservation exceeds tolerance")
    return np.asarray(output, dtype=np.float64), float(max_conservation)


def _repricing_max(
    rows: pd.DataFrame,
    values: np.ndarray,
    *,
    delivery_months: tuple[str, ...],
) -> float:
    residuals = []
    for row in rows.itertuples(index=False):
        vector = product_constraint_vector(
            ProductConstraint(str(row.product), str(row.load_type)),
            delivery_months=delivery_months,
        )
        residuals.append(abs(float(np.dot(vector, values)) - float(row.price)))
    return float(max(residuals, default=0.0))


def _product_delivery_hours(constraint: ProductConstraint) -> int:
    total = 0
    for month in product_periods(constraint.product):
        base, peak, offpeak = count_hours(
            int(month.year), int(month.month), int(month.month), tz=LOCAL_TIMEZONE, country="CH"
        )
        total += {"BASE": base, "PEAK": peak, "OFFPEAK": offpeak}[constraint.load_type.upper()]
    return total


def _load_bound_artifact(
    root: Path,
    manifest: Mapping[str, object],
    *,
    role: str,
    expected_name: str,
    expected_schema: str,
) -> tuple[Path, dict[str, object], bytes]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise Tier2MonthlyEexEvaluationError("manifest artifact inventory is invalid")
    matches = [item for item in artifacts if isinstance(item, Mapping) and item.get("role") == role]
    if len(matches) != 1:
        raise Tier2MonthlyEexEvaluationError(f"manifest must contain one {role} artifact")
    entry = matches[0]
    _require_exact_keys(
        entry,
        {"role", "path", "schema_version", "sha256", "size_bytes"},
        label=f"{role} artifact entry",
    )
    _require_equal(entry.get("path"), expected_name, label=f"{role} artifact path")
    _require_equal(entry.get("schema_version"), expected_schema, label=f"{role} artifact schema")
    path, payload, raw = _load_json_artifact(root / expected_name)
    _require_canonical_json_bytes(payload, raw, label=role)
    _require_equal(entry.get("sha256"), hashlib.sha256(raw).hexdigest(), label=f"{role} hash")
    _require_equal(entry.get("size_bytes"), len(raw), label=f"{role} size")
    _require_equal(payload.get("schema_version"), expected_schema, label=f"{role} payload schema")
    return path, payload, raw


def _package_snapshot(root: Path) -> dict[str, tuple[int, int, int, int, str]]:
    expected = {_MANIFEST_FILENAME, _ROWS_FILENAME, _RESULT_FILENAME}
    actual: set[str] = set()
    paths: list[Path] = []
    for entry in root.iterdir():
        regular = _regular_file(entry)
        actual.add(entry.name)
        paths.append(regular)
    if actual != expected:
        raise Tier2MonthlyEexEvaluationError(
            f"fold package exact inventory mismatch: expected={sorted(expected)}, actual={sorted(actual)}"
        )
    return {
        path.name: _file_snapshot(path)
        for path in sorted(paths, key=lambda item: item.name)
    }


def _file_set_snapshot(
    paths: Sequence[Path],
) -> dict[str, tuple[int, int, int, int, str]]:
    return {str(path): _file_snapshot(path) for path in paths}


def _file_snapshot(path: Path) -> tuple[int, int, int, int, str]:
    regular = _regular_file(path)
    raw = regular.read_bytes()
    stat_result = regular.stat()
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
        hashlib.sha256(raw).hexdigest(),
    )


def _verify_authority_separation(
    *,
    plan: object,
    execution_key_id: str,
    fold_timestamp_key_id: str,
    source_authorities: Mapping[str, object],
) -> None:
    source_timestamp_keys = source_authorities.get("timestamp_key_ids")
    if not isinstance(source_timestamp_keys, list) or not source_timestamp_keys:
        raise Tier2MonthlyEexEvaluationError(
            "source trusted-time authorities are unavailable"
        )
    time_keys = {
        plan.timestamp_key_id,
        fold_timestamp_key_id,
        *source_timestamp_keys,
    }
    non_time_keys = [
        plan.governance_key_id,
        str(source_authorities.get("acquisition_key_id", "")),
        execution_key_id,
    ]
    if any(not key for key in non_time_keys):
        raise Tier2MonthlyEexEvaluationError(
            "non-time authority identities are unavailable"
        )
    if len(set(non_time_keys)) != len(non_time_keys):
        raise Tier2MonthlyEexEvaluationError(
            "governance, acquisition and execution authorities must be distinct"
        )
    if set(non_time_keys).intersection(time_keys):
        raise Tier2MonthlyEexEvaluationError(
            "non-time authorities must be independent of trusted-time authorities"
        )


def _frame_semantic_sha256(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(
        ["available_at", "snapshot_id", "market", "load_type", "product", "row_hash"],
        kind="mergesort",
    )
    return _sha256_json(
        {
            "schema_version": "tier2_monthly_eex_frame_semantic.v1",
            "row_count": len(ordered),
            "ordered_row_hashes": ordered["row_hash"].astype(str).tolist(),
        }
    )


def _require_numeric_mapping_equal(
    actual: object, expected: Mapping[str, object], *, label: str
) -> None:
    if not isinstance(actual, Mapping) or set(actual) != set(expected):
        raise Tier2MonthlyEexEvaluationError(f"{label} fields mismatch")
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, float):
            number = _finite_number(actual_value, label=f"{label}.{key}")
            if not math.isclose(number, expected_value, rel_tol=0.0, abs_tol=NUMERIC_TOLERANCE):
                raise Tier2MonthlyEexEvaluationError(f"{label}.{key} mismatch")
        else:
            _require_equal(actual_value, expected_value, label=f"{label}.{key}")


def _absolute_regular_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise Tier2MonthlyEexEvaluationError(f"{label} must be absolute")
    return _regular_file(path)


def _require_canonical_json_bytes(
    payload: Mapping[str, object], raw: bytes, *, label: str
) -> None:
    if raw != _canonical_json_bytes(payload):
        raise Tier2MonthlyEexEvaluationError(
            f"{label} must use canonical UTF-8 JSON bytes"
        )


def _json_thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_thaw(item) for item in value]
    return value


__all__ = ["verify_signed_selection_fold_evidence"]
