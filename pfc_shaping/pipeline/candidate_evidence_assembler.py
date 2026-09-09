"""Assemble reproducible post-solve evidence from one staged LT candidate."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import pandas as pd

import pfc_shaping.calibration.monthly_curve_audit as monthly_curve_audit_module
from pfc_shaping.calibration.monthly_curve_audit import audit_monthly_curve_shape
from pfc_shaping.calibration.monthly_curve_config_identity import (
    config_hash,
    solver_numerical_diagnostic_errors,
)
from pfc_shaping.calibration.monthly_forward_curve import (
    MarketQuote,
    MonthlyConstraintSystem,
    build_monthly_constraint_system,
)
from pfc_shaping.data.acquisition_contract import (
    GOVERNED_LT_INPUT_SNAPSHOT_SCHEMAS,
    AcquisitionAuthenticationError,
    verify_acquisition_contract,
)
from pfc_shaping.data.forward_proxy import verify_forward_manifest_binding
from pfc_shaping.data.lt_input_sources import (
    LTInputPaths,
    consumed_lt_input_roles,
    validate_governed_source_system,
    validate_lt_input_consumption,
    validate_lt_input_contract_semantics,
)
from pfc_shaping.data.snapshot_publication_state import (
    SnapshotPublicationStateError,
    verify_external_publication_evidence,
    verify_publication_authority_separation,
)
from pfc_shaping.path_safety import path_is_link, read_stable_single_link_file
from pfc_shaping.pipeline.candidate_evidence import (
    CandidateEvidenceError,
    verify_pre_run_governance_evidence,
)
from pfc_shaping.pipeline.model_governance_contract import (
    ModelGovernanceAuthenticationError,
    validate_selected_lambda_decision,
)
from pfc_shaping.pipeline.monthly_curve_authority import (
    active_monthly_curve_config_payload,
    monthly_solver_settings,
)
from pfc_shaping.pipeline.probabilistic_output_governance import (
    deterministic_only_frame,
)
from pfc_shaping.pipeline.strict_structured_data import (
    load_strict_json,
    load_strict_yaml,
)

ASSEMBLY_SCHEMA = "lt_candidate_derived_evidence_assembly.v1"
EXPORT_SCHEMA = "lt_candidate_hourly_export.v1"
PARITY_SCHEMA = "monthly_curve_selected_config_run_parity.v1"
ASSEMBLY_MANIFEST = Path("manifests") / "candidate_derived_evidence_assembly.json"
HOURLY_EXPORT = Path("exports") / "CH" / "pfc_hourly.csv"
HOURLY_EXPORT_MANIFEST = Path("manifests") / "hourly_export_manifest_ch.json"
SELECTED_CONFIG_PARITY = Path("manifests") / "selected_config_run_parity.json"
MONTHLY_AUDIT_GATES = Path("audits") / "monthly_curve_gates.csv"
MONTHLY_AUDIT_MANIFEST = Path("manifests") / "monthly_curve_audit_manifest.json"
LOCK_FILE = ".candidate-evidence-assembly.lock"
LOCAL_TZ = "Europe/Zurich"
DEPENDENCY_GRAPH = {
    "hourly_export_ch": ["pfc_ch_15min"],
    "export_manifest": ["pfc_ch_15min", "hourly_export_ch"],
    "selected_config_run_parity": [
        "active_config",
        "production_manifest",
        "selected_lambda_decision",
    ],
    "monthly_audit_gates": [
        "historical_thresholds",
        "pfc_ch_15min",
        "production_manifest",
    ],
    "monthly_audit_manifest": [
        "historical_thresholds",
        "monthly_audit_gates",
        "pfc_ch_15min",
        "production_manifest",
    ],
}


class CandidateEvidenceAssemblyError(RuntimeError):
    """Raised when staged candidate bytes cannot produce deterministic evidence."""


def assemble_candidate_derived_evidence(
    staging_path: str | Path,
    *,
    run_id: str,
) -> dict[str, object]:
    """Create the first reproducible evidence layer without running the model again."""

    staging = Path(staging_path).resolve()
    if not staging.is_dir():
        raise CandidateEvidenceAssemblyError(f"candidate staging is unavailable: {staging}")
    with _assembly_lock(staging, run_id=run_id):
        if (staging / ASSEMBLY_MANIFEST).is_file():
            return verify_candidate_derived_evidence(staging, expected_run_id=run_id)

        run_manifest_path = staging / "manifests" / "candidate_run_manifest.json"
        run_manifest = _load_json(run_manifest_path)
        if str(run_manifest.get("run_id", "")) != str(run_id):
            raise CandidateEvidenceAssemblyError("candidate run_id mismatch")
        if run_manifest.get("candidate_evidence_manifest") is not None:
            raise CandidateEvidenceAssemblyError("candidate evidence is already sealed")
        _validate_deterministic_run_manifest(run_manifest)

        sources = _source_artifacts(staging, run_manifest)
        pfc = _load_pfc(staging / str(sources["pfc_ch_15min"]["path"]))
        hourly = _hourly_export_frame(pfc)
        hourly_path = staging / HOURLY_EXPORT
        _write_csv_exclusive(hourly_path, hourly)
        hourly_ref = _artifact_ref(staging, hourly_path, role="hourly_export_ch")

        export_manifest = _expected_export_manifest(
            run_id=run_id,
            hourly=hourly,
            hourly_ref=hourly_ref,
            pfc_ref=sources["pfc_ch_15min"],
            production_ref=sources["production_manifest"],
        )
        export_manifest_path = staging / HOURLY_EXPORT_MANIFEST
        _write_json_exclusive(export_manifest_path, export_manifest)
        export_manifest_ref = _artifact_ref(
            staging,
            export_manifest_path,
            role="export_manifest",
        )

        parity, monthly_curve, constraints = _selected_config_parity(
            staging,
            run_id=run_id,
            sources=sources,
        )
        parity_path = staging / SELECTED_CONFIG_PARITY
        _write_json_exclusive(parity_path, parity)
        parity_ref = _artifact_ref(staging, parity_path, role="selected_config_run_parity")
        historical_thresholds = pd.read_csv(staging / str(sources["historical_thresholds"]["path"]))
        monthly_gates = audit_monthly_curve_shape(
            monthly_curve,
            constraints,
            historical_thresholds=historical_thresholds,
        )
        monthly_gates_path = staging / MONTHLY_AUDIT_GATES
        _write_csv_exclusive(monthly_gates_path, monthly_gates)
        monthly_gates_ref = _artifact_ref(
            staging,
            monthly_gates_path,
            role="monthly_audit_gates",
        )
        monthly_audit_manifest = _expected_monthly_audit_manifest(
            run_id=run_id,
            gates=monthly_gates,
            gates_ref=monthly_gates_ref,
            sources=sources,
        )
        monthly_audit_manifest_path = staging / MONTHLY_AUDIT_MANIFEST
        _write_json_exclusive(monthly_audit_manifest_path, monthly_audit_manifest)
        monthly_audit_manifest_ref = _artifact_ref(
            staging,
            monthly_audit_manifest_path,
            role="monthly_audit_manifest",
        )

        payload: dict[str, object] = {
            "schema_version": ASSEMBLY_SCHEMA,
            "run_id": str(run_id),
            "evidence_reference_timestamp": str(run_manifest.get("reference_timestamp", "")),
            "promotion_eligible": False,
            "sources": sources,
            "derived": {
                "hourly_export_ch": hourly_ref,
                "export_manifest": export_manifest_ref,
                "selected_config_run_parity": parity_ref,
                "monthly_audit_gates": monthly_gates_ref,
                "monthly_audit_manifest": monthly_audit_manifest_ref,
            },
            "dependency_graph": DEPENDENCY_GRAPH,
        }
        _write_json_exclusive(staging / ASSEMBLY_MANIFEST, payload)
        return verify_candidate_derived_evidence(staging, expected_run_id=run_id)


def verify_candidate_derived_evidence(
    staging_path: str | Path,
    *,
    expected_run_id: str,
) -> dict[str, object]:
    """Replay the assembly from its staged parents and reject coherent JSON claims."""

    staging = Path(staging_path).resolve()
    manifest = _load_json(staging / ASSEMBLY_MANIFEST)
    errors: list[str] = []
    if set(manifest) != {
        "schema_version",
        "run_id",
        "evidence_reference_timestamp",
        "promotion_eligible",
        "sources",
        "derived",
        "dependency_graph",
    }:
        errors.append("top_level_keys")
    if manifest.get("schema_version") != ASSEMBLY_SCHEMA:
        errors.append("schema_version")
    if str(manifest.get("run_id", "")) != str(expected_run_id):
        errors.append("run_id")
    if manifest.get("promotion_eligible") is not False:
        errors.append("promotion_eligible")
    if manifest.get("dependency_graph") != DEPENDENCY_GRAPH:
        errors.append("dependency_graph")

    run_manifest = _load_json(staging / "manifests" / "candidate_run_manifest.json")
    try:
        _validate_deterministic_run_manifest(run_manifest)
    except CandidateEvidenceAssemblyError:
        errors.append("probabilistic_output_governance")
    if str(manifest.get("evidence_reference_timestamp", "")) != str(
        run_manifest.get("reference_timestamp", "")
    ):
        errors.append("evidence_reference_timestamp")
    try:
        expected_sources = _source_artifacts(staging, run_manifest)
    except CandidateEvidenceAssemblyError:
        errors.append("source_resolution")
        expected_sources = {}
    if manifest.get("sources") != expected_sources:
        errors.append("sources")

    derived = manifest.get("derived")
    if not isinstance(derived, Mapping) or set(derived) != {
        "hourly_export_ch",
        "export_manifest",
        "selected_config_run_parity",
        "monthly_audit_gates",
        "monthly_audit_manifest",
    }:
        errors.append("derived_roles")
    else:
        for role, entry in derived.items():
            try:
                _verify_artifact_ref(staging, entry, expected_role=str(role))
            except CandidateEvidenceAssemblyError:
                errors.append(f"{role}.artifact")
        expected_paths = {
            "hourly_export_ch": HOURLY_EXPORT.as_posix(),
            "export_manifest": HOURLY_EXPORT_MANIFEST.as_posix(),
            "selected_config_run_parity": SELECTED_CONFIG_PARITY.as_posix(),
            "monthly_audit_gates": MONTHLY_AUDIT_GATES.as_posix(),
            "monthly_audit_manifest": MONTHLY_AUDIT_MANIFEST.as_posix(),
        }
        for role, expected_path in expected_paths.items():
            entry = derived.get(role)
            if not isinstance(entry, Mapping) or entry.get("path") != expected_path:
                errors.append(f"{role}.canonical_path")

    if expected_sources:
        try:
            expected_hourly = _hourly_export_frame(
                _load_pfc(staging / str(expected_sources["pfc_ch_15min"]["path"]))
            )
            expected_csv = _csv_bytes(expected_hourly)
            hourly_path = _regular_file_within(staging, HOURLY_EXPORT, role="hourly_export_ch")
            if hourly_path.read_bytes() != expected_csv:
                errors.append("hourly_export_ch.replay")

            export_manifest = _load_json(staging / HOURLY_EXPORT_MANIFEST)
            expected_export_manifest = _expected_export_manifest(
                run_id=expected_run_id,
                hourly=expected_hourly,
                hourly_ref=_artifact_ref(staging, hourly_path, role="hourly_export_ch"),
                pfc_ref=expected_sources["pfc_ch_15min"],
                production_ref=expected_sources["production_manifest"],
            )
            if export_manifest != expected_export_manifest:
                errors.append("export_manifest.replay")

            expected_parity, monthly_curve, constraints = _selected_config_parity(
                staging,
                run_id=expected_run_id,
                sources=expected_sources,
            )
            if _load_json(staging / SELECTED_CONFIG_PARITY) != expected_parity:
                errors.append("selected_config_run_parity.replay")
            historical_thresholds = pd.read_csv(
                staging / str(expected_sources["historical_thresholds"]["path"])
            )
            expected_gates = audit_monthly_curve_shape(
                monthly_curve,
                constraints,
                historical_thresholds=historical_thresholds,
            )
            gates_path = _regular_file_within(
                staging,
                MONTHLY_AUDIT_GATES,
                role="monthly_audit_gates",
            )
            if gates_path.read_bytes() != _csv_bytes(expected_gates):
                errors.append("monthly_audit_gates.replay")
            expected_gates_ref = _artifact_ref(
                staging,
                gates_path,
                role="monthly_audit_gates",
            )
            expected_monthly_manifest = _expected_monthly_audit_manifest(
                run_id=expected_run_id,
                gates=expected_gates,
                gates_ref=expected_gates_ref,
                sources=expected_sources,
            )
            if _load_json(staging / MONTHLY_AUDIT_MANIFEST) != expected_monthly_manifest:
                errors.append("monthly_audit_manifest.replay")
            if _source_artifacts(staging, run_manifest) != expected_sources:
                errors.append("sources.changed_during_replay")
        except (CandidateEvidenceAssemblyError, ModelGovernanceAuthenticationError, ValueError):
            errors.append("derived_replay")

    if errors:
        raise CandidateEvidenceAssemblyError(
            "candidate derived evidence mismatch: " + ",".join(sorted(set(errors)))
        )
    return manifest


def _source_artifacts(
    staging: Path,
    run_manifest: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    markets = run_manifest.get("markets")
    source_evidence = run_manifest.get("source_evidence")
    pre_run_path_value = run_manifest.get("pre_run_governance_manifest")
    data_contract = run_manifest.get("data_contract")
    data_pointer = run_manifest.get("data_pointer")
    if not isinstance(markets, Mapping) or not isinstance(markets.get("CH"), Mapping):
        raise CandidateEvidenceAssemblyError("CH candidate market outputs are missing")
    if not isinstance(source_evidence, Mapping):
        raise CandidateEvidenceAssemblyError("candidate source evidence is missing")
    if not isinstance(data_contract, Mapping):
        raise CandidateEvidenceAssemblyError("governed LT input snapshot is missing")
    if not isinstance(data_pointer, Mapping):
        raise CandidateEvidenceAssemblyError("governed LT input pointer is missing")
    if not pre_run_path_value:
        raise CandidateEvidenceAssemblyError("pre-run governance manifest is missing")
    if str(pre_run_path_value) != "manifests/pre_run_governance_manifest.json":
        raise CandidateEvidenceAssemblyError("pre-run governance path is not canonical")
    declared_pre_run_hash = str(run_manifest.get("pre_run_governance_manifest_sha256", ""))
    pre_run_path = _regular_file_within(
        staging,
        str(pre_run_path_value),
        role="pre_run_governance",
    )
    if _sha256_file(pre_run_path) != declared_pre_run_hash:
        raise CandidateEvidenceAssemblyError("pre-run governance hash mismatch")
    try:
        pre_run = verify_pre_run_governance_evidence(
            staging,
            expected_run_id=str(run_manifest.get("run_id", "")),
            expected_valuation_timestamp=str(run_manifest.get("reference_timestamp", "")),
        )
    except CandidateEvidenceError as exc:
        raise CandidateEvidenceAssemblyError("pre-run governance authentication failed") from exc
    independent = pre_run.get("artifacts")
    if not isinstance(independent, Mapping) or not isinstance(
        independent.get("selected_lambda_decision"), Mapping
    ):
        raise CandidateEvidenceAssemblyError("selected lambda decision binding is missing")
    eex_pre_run = independent.get("eex_forward_source")
    if not isinstance(eex_pre_run, Mapping) or not isinstance(
        eex_pre_run.get("acquisition_contract"), Mapping
    ):
        raise CandidateEvidenceAssemblyError("authenticated pre-run EEX source binding is missing")
    selected_path = independent["selected_lambda_decision"].get("path")
    thresholds_path = independent["historical_thresholds"].get("path")
    ch = markets["CH"]
    values = {
        "active_config": run_manifest.get("config_path"),
        "input_snapshot_manifest": data_contract.get("archive_path"),
        "input_snapshot_pointer": data_pointer.get("archive_path"),
        "forward_source": source_evidence.get("forward_source_archive"),
        "eex_acquisition_contract": eex_pre_run["acquisition_contract"].get("path"),
        "forward_quote_snapshot": source_evidence.get("forward_quote_snapshot"),
        "production_manifest": ch.get("monthly_curve_manifest"),
        "pfc_ch_15min": ch.get("pfc_parquet"),
        "selected_lambda_decision": selected_path,
        "historical_thresholds": thresholds_path,
    }
    sources: dict[str, dict[str, object]] = {}
    for role, value in values.items():
        if not value:
            raise CandidateEvidenceAssemblyError(f"candidate source path is missing: {role}")
        path = _regular_file_within(staging, str(value), role=role)
        sources[role] = _artifact_ref(staging, path, role=role)
    declared_hashes = {
        "active_config": run_manifest.get("config_sha256"),
        "input_snapshot_manifest": data_contract.get("sha256"),
        "input_snapshot_pointer": data_pointer.get("sha256"),
        "forward_source": source_evidence.get("forward_source_sha256"),
        "forward_quote_snapshot": source_evidence.get("forward_quote_snapshot_sha256"),
        "production_manifest": ch.get("monthly_curve_manifest_sha256"),
        "pfc_ch_15min": ch.get("pfc_parquet_sha256"),
    }
    for role, declared in declared_hashes.items():
        if str(declared or "") != str(sources[role]["sha256"]):
            raise CandidateEvidenceAssemblyError(f"candidate declared source hash mismatch: {role}")
    if str(sources["forward_source"]["sha256"]) != str(eex_pre_run.get("sha256", "")):
        raise CandidateEvidenceAssemblyError(
            "candidate EEX source differs from authenticated pre-run source"
        )
    try:
        input_contract = _load_json(staging / str(sources["input_snapshot_manifest"]["path"]))
        verified_contract = verify_acquisition_contract(input_contract)
        validate_lt_input_contract_semantics(input_contract)
    except (
        CandidateEvidenceAssemblyError,
        AcquisitionAuthenticationError,
        ValueError,
    ) as exc:
        raise CandidateEvidenceAssemblyError(
            "governed LT input snapshot authentication failed"
        ) from exc
    pointer = _load_json(staging / str(sources["input_snapshot_pointer"]["path"]))
    pointer_schema = pointer.get("schema_version")
    if pointer_schema != "lt_data_pointer.v2":
        raise CandidateEvidenceAssemblyError(
            "promotion-grade LT candidate requires an external CAS v2 pointer"
        )
    if pointer_schema == "lt_data_pointer.v2":
        publication_entries = {
            "publication_intent": run_manifest.get("data_publication_intent"),
            "publication_anchor_receipt": run_manifest.get("data_publication_anchor_receipt"),
            "publication_head_observation": run_manifest.get("data_publication_head_observation"),
        }
        publication_payloads: dict[str, bytes] = {}
        for role, entry in publication_entries.items():
            if not isinstance(entry, Mapping):
                raise CandidateEvidenceAssemblyError(
                    f"candidate external publication evidence is missing: {role}"
                )
            path = _regular_file_within(
                staging,
                str(entry.get("archive_path", "")),
                role=role,
            )
            payload = read_stable_single_link_file(
                path,
                label=f"candidate {role}",
            )
            publication_payloads[role] = payload
            sources[role] = {
                "role": role,
                "path": path.relative_to(staging).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            if str(entry.get("sha256", "")) != str(sources[role]["sha256"]):
                raise CandidateEvidenceAssemblyError(
                    f"candidate declared source hash mismatch: {role}"
                )
        try:
            run_spec = pre_run.get("governed_run_spec")
            snapshot_binding = (
                run_spec.get("input_snapshot") if isinstance(run_spec, Mapping) else None
            )
            expected_nonce = (
                snapshot_binding.get("publication_head_challenge_nonce")
                if isinstance(snapshot_binding, Mapping)
                else None
            )
            intent_document = _load_json_bytes(
                publication_payloads["publication_intent"],
                label="publication intent",
            )
            receipt_document = _load_json_bytes(
                publication_payloads["publication_anchor_receipt"],
                label="publication anchor receipt",
            )
            if intent_document.get("transition_type") != "PUBLISH":
                raise CandidateEvidenceAssemblyError(
                    "promotion-grade candidate cannot consume a BOOTSTRAP publication"
                )
            observation = _load_json_bytes(
                publication_payloads["publication_head_observation"],
                label="publication HEAD observation",
            )
            if observation.get("challenge_nonce") != expected_nonce:
                raise CandidateEvidenceAssemblyError(
                    "candidate HEAD observation nonce differs from run spec"
                )
            verify_external_publication_evidence(
                intent_payload=publication_payloads["publication_intent"],
                receipt_payload=publication_payloads["publication_anchor_receipt"],
                observation_payload=publication_payloads["publication_head_observation"],
                pointer=pointer,
                require_current=False,
            )
            verify_publication_authority_separation(
                input_contract,
                intent=intent_document,
                receipt=receipt_document,
                observation=observation,
            )
            run_started_at = pd.Timestamp(run_manifest.get("run_started_at_utc"))
            observed_time = pd.Timestamp(observation.get("observed_at_utc"))
            expires_time = pd.Timestamp(observation.get("expires_at_utc"))
            if not observed_time <= run_started_at < expires_time:
                raise CandidateEvidenceAssemblyError(
                    "candidate HEAD observation was not fresh at run start"
                )
        except SnapshotPublicationStateError as exc:
            raise CandidateEvidenceAssemblyError(
                "candidate external publication evidence is invalid"
            ) from exc
    generation_id = str(run_manifest.get("data_generation_id", "")).strip()
    if not generation_id:
        raise CandidateEvidenceAssemblyError("candidate data generation_id is missing")
    if (
        str(pointer.get("generation_id", "")) != generation_id
        or str(verified_contract.get("generation_id", "")) != generation_id
    ):
        raise CandidateEvidenceAssemblyError("LT pointer/contract generation mismatch")
    if str(pointer.get("contract_sha256", "")) != str(sources["input_snapshot_manifest"]["sha256"]):
        raise CandidateEvidenceAssemblyError("LT pointer contract hash mismatch")
    if (
        verified_contract.get("calibration_eligible") is not True
        or verified_contract.get("source_class") != "GOVERNED_ACQUISITION"
        or verified_contract.get("layout") != "external_v2"
    ):
        raise CandidateEvidenceAssemblyError(
            "governed LT input snapshot is not calibration eligible"
        )
    _validate_input_consumption(staging, run_manifest, verified_contract)
    return sources


def _validate_input_consumption(
    staging: Path,
    run_manifest: Mapping[str, object],
    contract: Mapping[str, object],
) -> None:
    files = contract.get("files")
    receipts = run_manifest.get("input_sources")
    if not isinstance(files, Mapping) or not isinstance(receipts, Mapping):
        raise CandidateEvidenceAssemblyError("candidate input consumption evidence is missing")
    config_path = str(run_manifest.get("config_path", ""))
    try:
        config = load_strict_yaml((staging / config_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        config = None
    if not isinstance(config, Mapping):
        raise CandidateEvidenceAssemblyError("candidate config is unavailable for input replay")
    try:
        consumed = consumed_lt_input_roles(
            config,
            available_roles={str(role) for role in files},
        )
        paths = LTInputPaths(
            root=Path("."),
            snapshot_root=Path("."),
            layout=str(contract.get("layout", "")),
            generation_id=str(contract.get("generation_id", "")),
            calibration_eligible=contract.get("calibration_eligible") is True,
            available_at_utc=str(contract.get("available_at_utc", "")),
            files={str(role): Path(str(entry.get("path", ""))) for role, entry in files.items()},
            expected_files={str(role): entry for role, entry in files.items()},
            schema_version=str(contract.get("schema_version", "")),
        )
        validate_lt_input_consumption(
            paths,
            roles=consumed,
            reference_timestamp=str(run_manifest.get("reference_timestamp", "")),
        )
    except (TypeError, ValueError) as exc:
        raise CandidateEvidenceAssemblyError("LT input PIT consumption replay failed") from exc
    receipt_roles = {str(role) for role in receipts}
    if receipt_roles != consumed:
        raise CandidateEvidenceAssemblyError(
            "candidate consumed LT input receipt roles are not exact"
        )
    for role in sorted(consumed):
        entry = files.get(role)
        receipt = receipts.get(role)
        if not isinstance(entry, Mapping) or not isinstance(receipt, Mapping):
            raise CandidateEvidenceAssemblyError(f"consumed LT input receipt is missing: {role}")
        expected = {
            "role": role,
            "logical_path": str(entry.get("path", "")),
            "size_bytes": int(entry.get("size_bytes", -1)),
            "sha256": str(entry.get("sha256", "")),
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise CandidateEvidenceAssemblyError(
                    f"consumed LT input receipt mismatch: {role}.{key}"
                )


def _selected_config_parity(
    staging: Path,
    *,
    run_id: str,
    sources: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], pd.Series, MonthlyConstraintSystem]:
    try:
        config = load_strict_yaml(
            (staging / str(sources["active_config"]["path"])).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise CandidateEvidenceAssemblyError("candidate config is unreadable") from exc
    if not isinstance(config, Mapping):
        raise CandidateEvidenceAssemblyError("candidate config must contain a mapping")
    active = active_monthly_curve_config_payload(monthly_solver_settings(config))
    active_settings = monthly_solver_settings(config)
    active_hash = config_hash(active)
    selected = _load_json(staging / str(sources["selected_lambda_decision"]["path"]))
    validate_selected_lambda_decision(selected, expected_config=active)
    production = _load_json(staging / str(sources["production_manifest"]["path"]))
    monthly_curve, constraints = _validate_production_manifest(
        staging,
        production=production,
        sources=sources,
        expected_solver_settings=active_settings,
    )
    production_hash = str(production.get("active_config_hash", ""))
    parity = production_hash == active_hash == str(selected.get("config_hash", ""))
    if not parity:
        raise CandidateEvidenceAssemblyError("selected/config/production hash parity failed")
    payload = {
        "schema_version": PARITY_SCHEMA,
        "run_id": str(run_id),
        "status": "PASS",
        "selection_status": str(selected["selection_status"]),
        "production_approved": bool(selected["production_approved"]),
        "canonical_config": dict(selected["canonical_config"]),
        "config_hash": str(selected["config_hash"]),
        "active_config_hash": active_hash,
        "selected_lambda_config_hash": str(selected["config_hash"]),
        "production_active_config_hash": production_hash,
        "monthly_solution_hash": str(production["monthly_solution_hash"]),
        "active_constraints_hash": str(production["active_constraints_hash"]),
        "ompex_training_selection_exclusion": _ompex_exclusion_evidence(
            staging,
            sources=sources,
            selected=selected,
        ),
        "parents": {
            role: sources[role]
            for role in (
                "active_config",
                "production_manifest",
                "selected_lambda_decision",
            )
        },
        "producer_contract": "pfc.pipeline.selected_config_run_parity.v1",
    }
    return payload, monthly_curve, constraints


def _load_pfc(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise CandidateEvidenceAssemblyError("staged CH PFC is unreadable") from exc
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise CandidateEvidenceAssemblyError("staged CH PFC index must be datetime")
    if "price_shape" not in frame.columns:
        raise CandidateEvidenceAssemblyError("staged CH PFC has no price_shape")
    try:
        frame = deterministic_only_frame(frame, context="staged CH PFC")
    except ValueError as exc:
        raise CandidateEvidenceAssemblyError(str(exc)) from exc
    if (
        frame.index.tz is None
        or not frame.index.is_monotonic_increasing
        or not frame.index.is_unique
    ):
        raise CandidateEvidenceAssemblyError("staged CH PFC index contract failed")
    utc_index = frame.index.tz_convert("UTC")
    if len(utc_index) < 2 or not bool(
        (utc_index.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all()
    ):
        raise CandidateEvidenceAssemblyError("staged CH PFC cadence must be exactly 15 minutes")
    local_index = frame.index.tz_convert(LOCAL_TZ)
    if not bool(
        ((local_index.minute % 15) == 0).all()
        and (local_index.second == 0).all()
        and (local_index.microsecond == 0).all()
    ):
        raise CandidateEvidenceAssemblyError("staged CH PFC is not quarter-hour aligned")
    price = pd.to_numeric(frame["price_shape"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(price).all():
        raise CandidateEvidenceAssemblyError("staged CH PFC price_shape is not finite")
    return frame


def _hourly_export_frame(pfc: pd.DataFrame) -> pd.DataFrame:
    local = pfc[["price_shape"]].tz_convert(LOCAL_TZ)
    counts = local["price_shape"].resample("h", label="left", closed="left").count()
    if not bool((counts == 4).all()):
        raise CandidateEvidenceAssemblyError("CH PFC does not contain four quarters per local hour")
    hourly = local["price_shape"].resample("h", label="left", closed="left").mean()
    if hourly.empty:
        raise CandidateEvidenceAssemblyError("CH hourly export is empty")
    out = pd.DataFrame(index=hourly.index)
    out["timestamp_ch"] = hourly.index.strftime("%d.%m.%Y %H:%M")
    offset = pd.Index(hourly.index.strftime("%z"))
    out["utc_offset_ch"] = "UTC" + offset.str.slice(0, 3) + ":" + offset.str.slice(3, 5)
    out["timestamp_utc"] = hourly.index.tz_convert("UTC").strftime("%d.%m.%Y %H:%M")
    out["price_weighted_mean_eur_mwh"] = hourly.to_numpy(dtype=float)
    return out.reset_index(drop=True)


def _expected_export_manifest(
    *,
    run_id: str,
    hourly: pd.DataFrame,
    hourly_ref: Mapping[str, object],
    pfc_ref: Mapping[str, object],
    production_ref: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": EXPORT_SCHEMA,
        "run_id": str(run_id),
        "market": "CH",
        "timezone": LOCAL_TZ,
        "source_price_column": "price_shape",
        "delivered_price_column": "price_weighted_mean_eur_mwh",
        "delivered_curve_kind": "deterministic_price_shape",
        "probabilistic_status": "DETERMINISTIC_ONLY",
        "interval_columns": [],
        "intervals_permitted_for_pricing_or_risk": False,
        "scenario_weights": {"price_shape": 1.0},
        "rows": int(len(hourly)),
        "first_timestamp_utc": str(hourly["timestamp_utc"].iloc[0]),
        "last_timestamp_utc": str(hourly["timestamp_utc"].iloc[-1]),
        "columns": list(hourly.columns),
        "finite_price_values": bool(
            np.isfinite(hourly["price_weighted_mean_eur_mwh"].to_numpy(dtype=float)).all()
        ),
        "parents": {
            "pfc_ch_15min": dict(pfc_ref),
            "production_manifest": dict(production_ref),
        },
        "artifact": dict(hourly_ref),
        "producer_contract": "pfc.pipeline.candidate_hourly_export.v1",
    }


def _validate_deterministic_run_manifest(run_manifest: Mapping[str, object]) -> None:
    if run_manifest.get("probabilistic_status") != "DETERMINISTIC_ONLY":
        raise CandidateEvidenceAssemblyError(
            "candidate probabilistic_status must be DETERMINISTIC_ONLY"
        )
    if run_manifest.get("interval_columns") != []:
        raise CandidateEvidenceAssemblyError("candidate interval_columns must be empty")
    if run_manifest.get("intervals_permitted_for_pricing_or_risk") is not False:
        raise CandidateEvidenceAssemblyError(
            "candidate intervals must be forbidden for pricing and risk"
        )
    markets = run_manifest.get("markets")
    if not isinstance(markets, Mapping):
        raise CandidateEvidenceAssemblyError("candidate market outputs are missing")
    for market, output in markets.items():
        if not isinstance(output, Mapping):
            raise CandidateEvidenceAssemblyError(f"candidate market output is invalid: {market}")
        if output.get("probabilistic_status") != "DETERMINISTIC_ONLY":
            raise CandidateEvidenceAssemblyError(
                f"candidate market is not deterministic-only: {market}"
            )
        columns = output.get("columns")
        if not isinstance(columns, list) or any(column in {"p10", "p90"} for column in columns):
            raise CandidateEvidenceAssemblyError(
                f"candidate market interval columns are not excluded: {market}"
            )


def _ompex_exclusion_evidence(
    staging: Path,
    *,
    sources: Mapping[str, Mapping[str, object]],
    selected: Mapping[str, object],
) -> dict[str, object]:
    """Recompute the benchmark exclusion from sealed input and selection lineage."""

    inspected: list[dict[str, str]] = []
    matches: list[str] = []
    input_contract = verify_acquisition_contract(
        _load_json(staging / str(sources["input_snapshot_manifest"]["path"]))
    )
    input_files = input_contract.get("files")
    if not isinstance(input_files, Mapping):
        raise CandidateEvidenceAssemblyError("candidate input source lineage is missing")
    for role, entry in sorted(input_files.items()):
        if not isinstance(entry, Mapping):
            raise CandidateEvidenceAssemblyError(f"candidate input source is invalid: {role}")
        try:
            source_system = validate_governed_source_system(str(role), entry)
        except ValueError as exc:
            raise CandidateEvidenceAssemblyError(
                f"candidate input source system is inadmissible: {role}"
            ) from exc
        inspected.append({"role": str(role), "source_system": source_system})

    eex_contract = verify_acquisition_contract(
        _load_json(staging / str(sources["eex_acquisition_contract"]["path"]))
    )
    if eex_contract.get("schema_version") not in GOVERNED_LT_INPUT_SNAPSHOT_SCHEMAS:
        raise CandidateEvidenceAssemblyError(
            "EEX source lineage requires a governed v2/v3 contract"
        )
    eex_files = eex_contract.get("files")
    if not isinstance(eex_files, Mapping) or set(eex_files) != {"eex_forward_source"}:
        raise CandidateEvidenceAssemblyError(
            "authenticated EEX source lineage must contain only the forward source"
        )
    eex_entry = eex_files.get("eex_forward_source") if isinstance(eex_files, Mapping) else None
    if not isinstance(eex_entry, Mapping):
        raise CandidateEvidenceAssemblyError("authenticated EEX source lineage is missing")
    try:
        eex_source_system = validate_governed_source_system("eex_forward_source", eex_entry)
    except ValueError as exc:
        raise CandidateEvidenceAssemblyError("EEX source system is inadmissible") from exc
    inspected.append({"role": "eex_forward_source", "source_system": eex_source_system})

    config = load_strict_yaml(
        (staging / str(sources["active_config"]["path"])).read_text(encoding="utf-8")
    )
    if not isinstance(config, Mapping):
        raise CandidateEvidenceAssemblyError("candidate config is unavailable for OMPEX replay")
    config_for_model = json.loads(json.dumps(config))
    paths = config_for_model.get("paths")
    benchmark_path = None
    if isinstance(paths, dict):
        benchmark_path = paths.pop("hfc_benchmark_dir", None)
    quality = config_for_model.get("quality")
    if not isinstance(quality, Mapping) or quality.get("benchmark_policy") != "advisory":
        matches.append("active_config:benchmark_policy")
    if isinstance(quality, Mapping) and quality.get("fail_on_benchmark") is not False:
        matches.append("active_config:fail_on_benchmark")
    model_config_identity = json.dumps(
        config_for_model,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    if _contains_ompex_benchmark_marker(model_config_identity):
        matches.append("active_config:model_consumed_sections")
    for role, entry in sorted(sources.items()):
        identity = f"{role}|{entry.get('path', '')}"
        if _contains_ompex_benchmark_marker(identity):
            matches.append(f"candidate_source:{role}")
    selection_identity = json.dumps(
        selected.get("canonical_config", {}),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    if _contains_ompex_benchmark_marker(selection_identity):
        matches.append("selected_lambda_decision:canonical_config")
    if matches:
        raise CandidateEvidenceAssemblyError(
            "OMPEX/HFC is present in model, selection, or promotion lineage: "
            + ",".join(sorted(matches))
        )
    return {
        "schema_version": "ompex_training_selection_exclusion.v1",
        "status": "PASS",
        "prohibited_uses": ["MODEL_INPUT", "TARGET", "LOSS", "SELECTION", "PROMOTION_GATE"],
        "inspected_input_sources": inspected,
        "authenticated_source_systems": sorted({entry["source_system"] for entry in inspected}),
        "benchmark_config": {
            "path_configured": benchmark_path is not None,
            "policy": quality.get("benchmark_policy") if isinstance(quality, Mapping) else None,
            "fail_on_benchmark": quality.get("fail_on_benchmark")
            if isinstance(quality, Mapping)
            else None,
            "usage": "ADVISORY_ONLY",
        },
        "inspected_selected_config_sha256": str(selected.get("config_hash", "")),
        "matches": matches,
    }


def _contains_ompex_benchmark_marker(value: str) -> bool:
    normalized = str(value).casefold().replace("-", "_").replace(" ", "_")
    return any(marker in normalized for marker in ("ompex", "hfc_ompex", "hfc_benchmark"))


def _expected_monthly_audit_manifest(
    *,
    run_id: str,
    gates: pd.DataFrame,
    gates_ref: Mapping[str, object],
    sources: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    statuses = (
        {str(key): int(value) for key, value in gates["status"].value_counts().items()}
        if "status" in gates.columns
        else {}
    )
    numeric = gates.select_dtypes(include="number")
    finite_numeric = bool(
        np.isfinite(numeric.to_numpy(dtype=float)[~pd.isna(numeric.to_numpy(dtype=float))]).all()
    )
    return {
        "schema_version": "monthly_curve_candidate_audit.v1",
        "run_id": str(run_id),
        "market": "CH",
        "rows": int(len(gates)),
        "status_counts": dict(sorted(statuses.items())),
        "critical_count": int(statuses.get("CRITICAL", 0)),
        "unsupported_count": int(statuses.get("UNSUPPORTED", 0)),
        "finite_numeric_values": finite_numeric,
        "promotion_eligible": False,
        "parents": {
            role: dict(sources[role])
            for role in (
                "historical_thresholds",
                "pfc_ch_15min",
                "production_manifest",
            )
        },
        "artifact": dict(gates_ref),
        "producer_contract": "pfc.pipeline.monthly_curve_candidate_audit.v1",
        "audit_implementation": {
            "module": "pfc_shaping.calibration.monthly_curve_audit",
            "module_sha256": _sha256_file(Path(monthly_curve_audit_module.__file__)),
            "pandas_version": pd.__version__,
        },
    }


def _validate_production_manifest(
    staging: Path,
    *,
    production: Mapping[str, object],
    sources: Mapping[str, Mapping[str, object]],
    expected_solver_settings: Mapping[str, object],
) -> tuple[pd.Series, MonthlyConstraintSystem]:
    if str(production.get("market", "")).upper() != "CH":
        raise CandidateEvidenceAssemblyError("production manifest market must be CH")
    if production.get("monthly_level_authority") != "solver":
        raise CandidateEvidenceAssemblyError("monthly solver is not the level authority")
    if production.get("solver_config") != dict(expected_solver_settings):
        raise CandidateEvidenceAssemblyError("production solver config differs from active config")
    if str(production.get("solver_config_hash", "")) != _sha256_json(
        dict(expected_solver_settings)
    ):
        raise CandidateEvidenceAssemblyError("production solver config hash mismatch")
    if not production.get("monthly_solution_hash") or not production.get("active_constraints_hash"):
        raise CandidateEvidenceAssemblyError("production monthly solution evidence is incomplete")
    if production.get("promotion_eligible") is not True:
        raise CandidateEvidenceAssemblyError("production monthly manifest is not eligible")
    forward = production.get("forward_snapshot")
    eligibility = production.get("forward_eligibility")
    if not isinstance(forward, Mapping):
        raise CandidateEvidenceAssemblyError("production forward snapshot is missing")
    if not isinstance(eligibility, Mapping):
        raise CandidateEvidenceAssemblyError("production forward eligibility is missing")
    if str(forward.get("source_sha256", "")) != str(sources["forward_source"]["sha256"]):
        raise CandidateEvidenceAssemblyError("production forward source hash mismatch")
    run_manifest = _load_json(staging / "manifests" / "candidate_run_manifest.json")
    source_evidence = run_manifest.get("source_evidence")
    if not isinstance(source_evidence, Mapping):
        raise CandidateEvidenceAssemblyError("candidate source evidence is missing")
    for key in ("snapshot_id", "observation_id"):
        if str(forward.get(key, "")) != str(source_evidence.get(f"forward_{key}", "")):
            raise CandidateEvidenceAssemblyError(f"production forward {key} mismatch")
    portable_forward = dict(forward)
    portable_forward["source_path"] = str(staging / str(sources["forward_source"]["path"]))
    try:
        verify_forward_manifest_binding(
            portable_forward,
            eligibility,
            verify_source=True,
            expected_reference_timestamp=str(run_manifest.get("reference_timestamp", "")),
        )
    except ValueError as exc:
        raise CandidateEvidenceAssemblyError(
            "production forward snapshot/eligibility replay failed"
        ) from exc
    _validate_forward_quote_snapshot(
        staging / str(sources["forward_quote_snapshot"]["path"]),
        forward=forward,
    )
    months = production.get("delivery_months")
    if not isinstance(months, list) or not months:
        raise CandidateEvidenceAssemblyError("production delivery months are missing")
    pfc = _load_pfc(staging / str(sources["pfc_ch_15min"]["path"]))
    expected_months = pd.PeriodIndex([str(value) for value in months], freq="M")
    expected_index = _delivery_quarter_hour_grid(expected_months)
    actual_index = pfc.index.tz_convert("UTC")
    if not actual_index.equals(expected_index):
        raise CandidateEvidenceAssemblyError(
            "PFC quarter-hour grid does not exactly cover solver delivery months"
        )
    ch_outputs = run_manifest.get("markets", {}).get("CH", {})
    if not isinstance(ch_outputs, Mapping) or int(ch_outputs.get("rows", -1)) != len(pfc):
        raise CandidateEvidenceAssemblyError("candidate run CH row count mismatch")
    local_periods = pfc.index.tz_convert(LOCAL_TZ).tz_localize(None).to_period("M")
    monthly = pfc["price_shape"].groupby(local_periods).mean().sort_index()
    monthly = monthly.reindex(expected_months)
    if monthly.isna().any():
        raise CandidateEvidenceAssemblyError("PFC does not cover production delivery months")
    payload = {str(key): round(float(value), 10) for key, value in monthly.items()}
    if production.get("monthly_solution") != payload:
        raise CandidateEvidenceAssemblyError("PFC monthly means differ from solver vector")
    if _sha256_json(payload) != str(production.get("monthly_solution_hash", "")):
        raise CandidateEvidenceAssemblyError("PFC monthly means do not match solver solution")
    constraints = _validate_hard_base_constraints(
        monthly,
        production=production,
        forward=forward,
    )
    return monthly, constraints


def _validate_hard_base_constraints(
    monthly: pd.Series,
    *,
    production: Mapping[str, object],
    forward: Mapping[str, object],
) -> MonthlyConstraintSystem:
    quotes_payload = forward.get("quotes")
    if not isinstance(quotes_payload, list):
        raise CandidateEvidenceAssemblyError("production forward quotes are missing")
    quotes: list[MarketQuote] = []
    for item in quotes_payload:
        if not isinstance(item, Mapping) or str(item.get("load_type", "")).upper() != "BASE":
            continue
        quotes.append(
            MarketQuote(
                market="CH",
                product=str(item.get("product", "")),
                load_type="BASE",
                price=float(item.get("price_eur_mwh")),
                snapshot_date=pd.Timestamp(forward.get("snapshot_date")),
                source=str(forward.get("source_kind", "")),
                available_at=pd.Timestamp(forward.get("available_at")),
                quote_id=str(item.get("quote_id", "")),
                snapshot_id=str(forward.get("snapshot_id", "")),
                observation_id=str(forward.get("observation_id", "")),
                source_kind=str(forward.get("source_kind", "")),
                source_sha256=str(forward.get("source_sha256", "")),
            )
        )
    if not quotes:
        raise CandidateEvidenceAssemblyError("production has no hard CH BASE quotes")
    policy = production.get("product_hierarchy_policy")
    if not isinstance(policy, Mapping):
        raise CandidateEvidenceAssemblyError("production quote hierarchy policy is missing")
    solver_config = production.get("solver_config")
    if not isinstance(solver_config, Mapping):
        raise CandidateEvidenceAssemblyError("production solver config is missing")
    numerical_errors = solver_numerical_diagnostic_errors(
        production.get("solver_kkt"),
        solver_config=solver_config,
    )
    if numerical_errors:
        raise CandidateEvidenceAssemblyError(
            f"production solver numerical diagnostics failed: {numerical_errors}"
        )
    expected_policy = {
        "schema_version": "monthly_quote_hierarchy_policy.v1",
        "priority": ["MONTH", "QUARTER", "CALENDAR"],
        "quote_conflict_tolerance_eur_mwh": float(
            solver_config.get("quote_conflict_tolerance", 0.01)
        ),
        "hard_repricing_tolerance_eur_mwh": float(solver_config.get("constraint_tolerance", 1e-9)),
        "stationarity_tolerance": float(solver_config.get("stationarity_tolerance", 1e-7)),
        "residual_lineage_required": True,
    }
    if dict(policy) != expected_policy:
        raise CandidateEvidenceAssemblyError("production quote hierarchy policy mismatch")
    if str(production.get("product_hierarchy_policy_sha256", "")) != _sha256_json(expected_policy):
        raise CandidateEvidenceAssemblyError("production quote hierarchy policy hash mismatch")
    try:
        constraints = build_monthly_constraint_system(
            monthly.index,
            quotes,
            timezone=LOCAL_TZ,
            market="CH",
            load_type="BASE",
            constraint_tolerance=float(policy.get("hard_repricing_tolerance_eur_mwh", 1e-9)),
            quote_conflict_tolerance=float(policy.get("quote_conflict_tolerance_eur_mwh", 0.01)),
        )
    except (TypeError, ValueError) as exc:
        raise CandidateEvidenceAssemblyError(
            "hard CH BASE constraint reconstruction failed"
        ) from exc
    if not constraints.names:
        raise CandidateEvidenceAssemblyError(
            "no hard CH BASE constraint overlaps the delivered monthly curve"
        )
    if str(production.get("active_constraints_hash", "")) != _hash_frame(constraints.rows):
        raise CandidateEvidenceAssemblyError("production active constraints hash mismatch")
    if str(production.get("quote_diagnostics_hash", "")) != _hash_frame(
        constraints.quote_diagnostics
    ):
        raise CandidateEvidenceAssemblyError("production quote diagnostics hash mismatch")
    provenance_columns = [
        "constraint_name",
        "product",
        "parent_product",
        "source_quote_ids",
        "lineage_formula",
        "lineage_sha256",
        "lineage_payload",
        "target",
        "hours",
        "n_months",
        "is_residual",
        "load_type",
    ]
    expected_provenance = _canonical_frame_records(constraints.rows[provenance_columns])
    if production.get("constraint_provenance_rows") != expected_provenance:
        raise CandidateEvidenceAssemblyError("production constraint provenance mismatch")
    if str(production.get("constraint_provenance_hash", "")) != _sha256_json(expected_provenance):
        raise CandidateEvidenceAssemblyError("production constraint provenance hash mismatch")
    expected_hard_quotes = [
        dict(item)
        for item in quotes_payload
        if isinstance(item, Mapping) and str(item.get("load_type", "")).upper() == "BASE"
    ]
    if production.get("hard_quotes") != expected_hard_quotes:
        raise CandidateEvidenceAssemblyError("production hard quote set mismatch")
    if str(production.get("hard_quote_set_hash", "")) != _sha256_json(expected_hard_quotes):
        raise CandidateEvidenceAssemblyError("production hard quote set hash mismatch")
    residuals = constraints.residuals(monthly.to_numpy(dtype=float))
    tolerance = float(policy.get("hard_repricing_tolerance_eur_mwh", 1e-9))
    if bool((residuals["abs_error"].astype(float) > tolerance).any()):
        raise CandidateEvidenceAssemblyError(
            "PFC monthly means do not reprice hard CH BASE constraints"
        )
    return constraints


def _validate_forward_quote_snapshot(
    path: Path,
    *,
    forward: Mapping[str, object],
) -> None:
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise CandidateEvidenceAssemblyError("forward quote snapshot is unreadable") from exc
    required = {
        "date",
        "product",
        "load_type",
        "product_type",
        "price",
        "market",
        "source",
        "quote_id",
    }
    if required.difference(frame.columns):
        raise CandidateEvidenceAssemblyError("forward quote snapshot columns are incomplete")
    actual = sorted(
        (
            str(row.market).upper(),
            str(row.load_type).upper(),
            str(row.product),
            str(row.product_type),
            round(float(row.price), 10),
            str(pd.Timestamp(row.date).date()),
            str(row.source),
            str(row.quote_id),
        )
        for row in frame.itertuples(index=False)
    )
    snapshot_date = str(pd.Timestamp(forward.get("snapshot_date")).date())
    expected: list[tuple[str, str, str, str, float, str, str, str]] = []
    quotes = forward.get("quotes")
    if not isinstance(quotes, list) or not quotes:
        raise CandidateEvidenceAssemblyError("production forward quotes are missing")
    for quote in quotes:
        if not isinstance(quote, Mapping):
            raise CandidateEvidenceAssemblyError("production forward quote is invalid")
        product = str(quote.get("product", ""))
        load_type = str(quote.get("load_type", "")).upper()
        if load_type == "PEAK" and product.lower().endswith("-peak"):
            product = product[:-5]
        elif load_type == "OFFPEAK" and product.lower().endswith("-offpeak"):
            product = product[:-8]
        product_type = (
            "Cal"
            if len(product) == 4 and product.isdigit()
            else "Quarter"
            if "-Q" in product
            else "Month"
            if len(product) == 7 and product[4] == "-"
            else "Unknown"
        )
        expected.append(
            (
                str(forward.get("market", "")).upper(),
                load_type,
                product,
                product_type,
                round(float(quote.get("price_eur_mwh")), 10),
                snapshot_date,
                "candidate_forward_snapshot",
                str(quote.get("quote_id", "")),
            )
        )
    if actual != sorted(expected):
        raise CandidateEvidenceAssemblyError(
            "forward quote snapshot does not match authenticated workbook replay"
        )


def _delivery_quarter_hour_grid(months: pd.PeriodIndex) -> pd.DatetimeIndex:
    if months.empty:
        raise CandidateEvidenceAssemblyError("solver delivery month grid is empty")
    expected = pd.period_range(months.min(), months.max(), freq="M")
    if not months.equals(expected):
        raise CandidateEvidenceAssemblyError("solver delivery months are not contiguous")
    start = months.min().to_timestamp(how="start").tz_localize(LOCAL_TZ)
    end = (months.max() + 1).to_timestamp(how="start").tz_localize(LOCAL_TZ)
    return pd.date_range(
        start.tz_convert("UTC"),
        end.tz_convert("UTC"),
        freq="15min",
        inclusive="left",
    )


def _canonical_frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    payload = frame.copy().reindex(sorted(frame.columns), axis=1)
    payload = payload.sort_values(list(payload.columns)).reset_index(drop=True)
    return payload.to_dict(orient="records")


def _hash_frame(frame: pd.DataFrame) -> str:
    return _sha256_json(_canonical_frame_records(frame))


def _artifact_ref(staging: Path, path: Path, *, role: str) -> dict[str, object]:
    regular = _regular_file_within(staging, path, role=role)
    return {
        "role": role,
        "path": regular.relative_to(staging).as_posix(),
        "sha256": _sha256_file(regular),
        "size_bytes": regular.stat().st_size,
    }


def _verify_artifact_ref(
    staging: Path,
    entry: object,
    *,
    expected_role: str,
) -> None:
    if not isinstance(entry, Mapping) or entry.get("role") != expected_role:
        raise CandidateEvidenceAssemblyError(f"invalid artifact reference: {expected_role}")
    path = _regular_file_within(staging, str(entry.get("path", "")), role=expected_role)
    if entry != _artifact_ref(staging, path, role=expected_role):
        raise CandidateEvidenceAssemblyError(f"artifact reference mismatch: {expected_role}")


def _regular_file_within(staging: Path, value: str | Path, *, role: str) -> Path:
    lexical = Path(value)
    if not lexical.is_absolute():
        lexical = staging / lexical
    if path_is_link(lexical):
        raise CandidateEvidenceAssemblyError(f"{role} cannot be a link")
    resolved = lexical.resolve()
    try:
        resolved.relative_to(staging)
    except ValueError as exc:
        raise CandidateEvidenceAssemblyError(f"{role} escapes candidate staging") from exc
    if not resolved.is_file():
        raise CandidateEvidenceAssemblyError(f"{role} is not a regular file")
    return resolved


@contextmanager
def _assembly_lock(staging: Path, *, run_id: str) -> Iterator[None]:
    lock = staging / LOCK_FILE
    payload = json.dumps({"run_id": str(run_id), "pid": os.getpid()}, sort_keys=True).encode()
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise CandidateEvidenceAssemblyError("candidate evidence assembly is locked") from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        lock.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, object]:
    try:
        raw = read_stable_single_link_file(path, label="candidate JSON artifact")
        payload = load_strict_json(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise CandidateEvidenceAssemblyError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise CandidateEvidenceAssemblyError(f"JSON artifact must contain an object: {path}")
    return payload


def _load_json_bytes(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CandidateEvidenceAssemblyError(f"invalid JSON artifact: {label}") from exc
    if not isinstance(value, dict):
        raise CandidateEvidenceAssemblyError(f"JSON artifact must contain an object: {label}")
    return value


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    raw = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _write_bytes_exclusive(path, raw)


def _write_csv_exclusive(path: Path, frame: pd.DataFrame) -> None:
    _write_bytes_exclusive(path, _csv_bytes(frame))


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise CandidateEvidenceAssemblyError(f"refusing to overwrite artifact: {path}") from exc
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
