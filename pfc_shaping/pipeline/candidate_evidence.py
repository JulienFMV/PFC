from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd

from pfc_shaping.build_identity import SOURCE_REVISION
from pfc_shaping.data.acquisition_contract import (
    GOVERNED_LT_INPUT_SNAPSHOT_SCHEMAS,
    AcquisitionAuthenticationError,
    verify_acquisition_contract,
)
from pfc_shaping.data.lt_input_sources import validate_governed_source_system
from pfc_shaping.path_safety import path_is_link, read_stable_single_link_file
from pfc_shaping.pipeline.model_governance_contract import (
    ModelGovernanceAuthenticationError,
    validate_selected_lambda_decision,
    verify_model_governance_artifact_receipt,
)
from pfc_shaping.pipeline.process_identity import candidate_temporary_name
from pfc_shaping.pipeline.strict_structured_data import (
    load_strict_json,
    load_strict_yaml,
)
from pfc_shaping.version import __version__

EVIDENCE_SCHEMA = "candidate_evidence.v1"
PRE_RUN_SCHEMA = "pre_run_model_governance.v2"
EVIDENCE_MANIFEST = Path("manifests") / "candidate_evidence_manifest.json"
CANDIDATE_RUN_MANIFEST = Path("manifests") / "candidate_run_manifest.json"
PRE_RUN_EVIDENCE_MANIFEST = Path("manifests") / "pre_run_governance_manifest.json"
PRE_RUN_EEX_ROLE = "eex_forward_source"
PRE_RUN_CONFIG_ROLE = "runtime_config"
GOVERNED_RUN_SPEC_SCHEMA = "governed_run_spec.v1"

INDEPENDENT_EVIDENCE_ROLES = frozenset(
    {"historical_thresholds", "selected_lambda_decision"}
)
CANDIDATE_DERIVED_EVIDENCE_ROLES = frozenset(
    {
        "production_manifest",
        "export_manifest",
        "audit_gates",
        "product_normalization_summary",
        "selected_config_run_parity",
    }
)
REQUIRED_EVIDENCE_ROLES = (
    INDEPENDENT_EVIDENCE_ROLES | CANDIDATE_DERIVED_EVIDENCE_ROLES
)
ASSEMBLED_EXTRA_EVIDENCE_ROLES = frozenset({"product_normalization_evidence"})
ROLE_DEPENDENCIES: Mapping[str, tuple[str, ...]] = {
    "historical_thresholds": (),
    "selected_lambda_decision": (),
    "production_manifest": (),
    "export_manifest": ("production_manifest",),
    "audit_gates": (
        "historical_thresholds",
        "production_manifest",
        "selected_lambda_decision",
    ),
    "product_normalization_summary": (
        "export_manifest",
        "production_manifest",
    ),
    "product_normalization_evidence": (
        "product_normalization_summary",
        "selected_config_run_parity",
    ),
    "selected_config_run_parity": (
        "production_manifest",
        "selected_lambda_decision",
    ),
}


class CandidateEvidenceError(RuntimeError):
    """Raised when promotion evidence is incomplete, mutable, or inconsistent."""


@dataclass(frozen=True)
class EvidenceArtifactInput:
    path: str | Path
    producer_contract: str
    authority_receipt_path: str | Path | None = None


def capture_pre_run_governance_evidence(
    staging_path: str | Path,
    *,
    run_id: str,
    valuation_timestamp: str,
    build_timestamp: str,
    historical_thresholds_path: str | Path,
    historical_thresholds_receipt_path: str | Path,
    selected_lambda_decision_path: str | Path,
    selected_lambda_decision_receipt_path: str | Path,
    runtime_config_path: str | Path,
    expected_runtime_config_sha256: str,
    source_revision: str,
    input_snapshot_sha256: str,
    input_pointer_sha256: str,
    input_generation_id: str,
    peak_source_policy: str,
    use_seasonal_hourly_shape: bool,
    publication_head_observation_sha256: str | None = None,
    publication_head_challenge_nonce: str | None = None,
    eex_forward_source_path: str | Path | None = None,
    eex_acquisition_contract_path: str | Path | None = None,
) -> dict[str, object]:
    """Capture authenticated independent inputs before any candidate solve."""

    source_revision = str(source_revision).strip().lower()
    input_snapshot_sha256 = str(input_snapshot_sha256).strip().lower()
    input_pointer_sha256 = str(input_pointer_sha256).strip().lower()
    if publication_head_observation_sha256 is not None:
        publication_head_observation_sha256 = str(
            publication_head_observation_sha256
        ).strip().lower()
    if publication_head_challenge_nonce is not None:
        publication_head_challenge_nonce = str(
            publication_head_challenge_nonce
        ).strip().lower()
    input_generation_id = str(input_generation_id).strip()
    peak_source_policy = str(peak_source_policy).strip()
    try:
        valuation_dt = datetime.fromisoformat(
            str(valuation_timestamp).strip().replace("Z", "+00:00")
        )
        build_dt = datetime.fromisoformat(
            str(build_timestamp).strip().replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise CandidateEvidenceError("run timestamps must be ISO-8601") from exc
    if valuation_dt.tzinfo is None or build_dt.tzinfo is None:
        raise CandidateEvidenceError("run timestamps must be timezone-aware")
    valuation_timestamp = valuation_dt.astimezone(timezone.utc).isoformat()
    build_timestamp = build_dt.astimezone(timezone.utc).isoformat()
    if build_dt < valuation_dt:
        raise CandidateEvidenceError("build timestamp precedes valuation timestamp")
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", source_revision):
        raise CandidateEvidenceError("source revision is not a canonical digest")
    if not _is_canonical_sha256(input_snapshot_sha256):
        raise CandidateEvidenceError("input snapshot SHA-256 is not canonical")
    if not _is_canonical_sha256(input_pointer_sha256):
        raise CandidateEvidenceError("input pointer SHA-256 is not canonical")
    if (
        publication_head_observation_sha256 is not None
        and not _is_canonical_sha256(publication_head_observation_sha256)
    ):
        raise CandidateEvidenceError(
            "publication HEAD observation SHA-256 is not canonical"
        )
    if (
        publication_head_challenge_nonce is not None
        and not _is_canonical_sha256(publication_head_challenge_nonce)
    ):
        raise CandidateEvidenceError("publication HEAD challenge nonce is not canonical")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", input_generation_id):
        raise CandidateEvidenceError("input generation id is not portable")
    if peak_source_policy not in {"same_first", "strict_same", "any"}:
        raise CandidateEvidenceError("peak source policy is invalid")
    if not isinstance(use_seasonal_hourly_shape, bool):
        raise CandidateEvidenceError("seasonal hourly shape control must be boolean")
    staging = Path(staging_path).resolve()
    if not staging.is_dir() or any(staging.iterdir()):
        raise CandidateEvidenceError("pre-run governance requires an empty staging directory")
    sources = {
        "historical_thresholds": (
            _pre_run_source(historical_thresholds_path, role="historical_thresholds"),
            _pre_run_source(
                historical_thresholds_receipt_path,
                role="historical_thresholds.authority_receipt",
            ),
        ),
        "selected_lambda_decision": (
            _pre_run_source(selected_lambda_decision_path, role="selected_lambda_decision"),
            _pre_run_source(
                selected_lambda_decision_receipt_path,
                role="selected_lambda_decision.authority_receipt",
            ),
        ),
    }
    source_paths = [path for pair in sources.values() for path in pair]
    if (eex_forward_source_path is None) != (eex_acquisition_contract_path is None):
        raise CandidateEvidenceError(
            "EEX source and acquisition contract must be provided together"
        )
    eex_source = (
        _pre_run_source(eex_forward_source_path, role=PRE_RUN_EEX_ROLE)
        if eex_forward_source_path is not None
        else None
    )
    eex_contract = (
        _pre_run_source(
            eex_acquisition_contract_path,
            role=f"{PRE_RUN_EEX_ROLE}.acquisition_contract",
        )
        if eex_acquisition_contract_path is not None
        else None
    )
    if eex_source is not None and eex_contract is not None:
        source_paths.extend([eex_source, eex_contract])
    if len(set(source_paths)) != len(source_paths):
        raise CandidateEvidenceError("pre-run governance source paths must be unique")
    entries: dict[str, object] = {}
    for role, (source, receipt_source) in sources.items():
        if source == receipt_source:
            raise CandidateEvidenceError(f"{role} artifact and receipt must be distinct")
        if not source.is_file() or not receipt_source.is_file():
            raise CandidateEvidenceError(f"{role} pre-run source evidence is unavailable")
        source_bytes = read_stable_single_link_file(
            source,
            label=f"{role} pre-run artifact",
        )
        receipt_bytes = read_stable_single_link_file(
            receipt_source,
            label=f"{role} pre-run receipt",
        )
        try:
            receipt_payload = load_strict_json(receipt_bytes.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise CandidateEvidenceError(f"{role} authority receipt is invalid JSON") from exc
        try:
            verified = verify_model_governance_artifact_receipt(
                receipt_payload,
                artifact_path=source,
                expected_role=role,
                valuation_timestamp=valuation_timestamp,
            )
        except ModelGovernanceAuthenticationError as exc:
            raise CandidateEvidenceError(f"{role} authority receipt is not authentic") from exc
        destination_root = staging / "evidence" / "independent" / role
        destination_root.mkdir(parents=True, exist_ok=False)
        artifact_destination = destination_root / source.name
        receipt_destination = destination_root / "authority_receipt.json"
        _write_bytes_exclusive(artifact_destination, source_bytes)
        _write_bytes_exclusive(receipt_destination, receipt_bytes)
        verify_model_governance_artifact_receipt(
            _load_json(receipt_destination),
            artifact_path=artifact_destination,
            expected_role=role,
            valuation_timestamp=valuation_timestamp,
        )
        entries[role] = {
            "path": artifact_destination.relative_to(staging).as_posix(),
            "sha256": _sha256_file(artifact_destination),
            "size_bytes": artifact_destination.stat().st_size,
            "authority_receipt": {
                "path": receipt_destination.relative_to(staging).as_posix(),
                "sha256": _sha256_file(receipt_destination),
                "receipt_id": verified["receipt_id"],
                "authority_id": verified["authority_id"],
                "issued_at_utc": verified["issued_at_utc"],
                "data_cutoff_utc": verified["data_cutoff_utc"],
            },
        }
    if eex_source is not None and eex_contract is not None:
        eex_entry = _verified_eex_acquisition(
            eex_source,
            eex_contract,
            valuation_timestamp=valuation_timestamp,
        )
        source_bytes = eex_source.read_bytes()
        contract_bytes = eex_contract.read_bytes()
        destination_root = staging / "evidence" / "independent" / PRE_RUN_EEX_ROLE
        destination_root.mkdir(parents=True, exist_ok=False)
        artifact_destination = destination_root / eex_source.name
        contract_destination = destination_root / "acquisition_contract.json"
        _write_bytes_exclusive(artifact_destination, source_bytes)
        _write_bytes_exclusive(contract_destination, contract_bytes)
        _verified_eex_acquisition(
            artifact_destination,
            contract_destination,
            valuation_timestamp=valuation_timestamp,
        )
        entries[PRE_RUN_EEX_ROLE] = {
            "path": artifact_destination.relative_to(staging).as_posix(),
            "sha256": _sha256_file(artifact_destination),
            "size_bytes": artifact_destination.stat().st_size,
            "acquisition_contract": {
                "path": contract_destination.relative_to(staging).as_posix(),
                "sha256": _sha256_file(contract_destination),
                "acquisition_id": eex_entry["acquisition_id"],
                "available_at_utc": eex_entry["available_at_utc"],
            },
        }
    config_source = _pre_run_source(runtime_config_path, role=PRE_RUN_CONFIG_ROLE)
    if config_source in source_paths or not config_source.is_file():
        raise CandidateEvidenceError("runtime config pre-run source is unavailable or reused")
    config_bytes = config_source.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if config_sha256 != str(expected_runtime_config_sha256).strip().lower():
        raise CandidateEvidenceError("runtime config SHA-256 does not match build contract")
    try:
        config_payload = load_strict_yaml(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CandidateEvidenceError("runtime config is invalid YAML") from exc
    if not isinstance(config_payload, Mapping):
        raise CandidateEvidenceError("runtime config must contain a mapping")
    config_root = staging / "evidence" / "independent" / PRE_RUN_CONFIG_ROLE
    config_root.mkdir(parents=True, exist_ok=False)
    config_destination = config_root / "config.yaml"
    _write_bytes_exclusive(config_destination, config_bytes)
    entries[PRE_RUN_CONFIG_ROLE] = {
        "path": config_destination.relative_to(staging).as_posix(),
        "sha256": config_sha256,
        "size_bytes": len(config_bytes),
        "semantic_sha256": hashlib.sha256(
            json.dumps(
                config_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest(),
    }
    input_snapshot_binding: dict[str, object] = {
        "generation_id": input_generation_id,
        "contract_logical_path": (
            Path("snapshots") / input_generation_id / "lt_input_snapshot.json"
        ).as_posix(),
        "contract_sha256": input_snapshot_sha256,
        "pointer_logical_path": "views/pfc_lt/current.json",
        "pointer_sha256": input_pointer_sha256,
    }
    if publication_head_observation_sha256 is not None:
        input_snapshot_binding["publication_head_observation_sha256"] = (
            publication_head_observation_sha256
        )
    if publication_head_challenge_nonce is not None:
        input_snapshot_binding["publication_head_challenge_nonce"] = (
            publication_head_challenge_nonce
        )
    run_spec: dict[str, object] = {
        "schema_version": GOVERNED_RUN_SPEC_SCHEMA,
        "run_id": str(run_id),
        "as_of_utc": str(valuation_timestamp),
        "run_started_at_utc": build_timestamp,
        "build_timestamp_utc": build_timestamp,
        "runtime": {
            "distribution": "fmv-pfc-lt",
            "version": __version__,
            "source_revision": source_revision,
        },
        "config": {
            "path": str(entries[PRE_RUN_CONFIG_ROLE]["path"]),
            "sha256": config_sha256,
            "semantic_sha256": str(
                entries[PRE_RUN_CONFIG_ROLE]["semantic_sha256"]
            ),
        },
        "input_snapshot": input_snapshot_binding,
        "model_controls": {
            "peak_source_policy": peak_source_policy,
            "use_seasonal_hourly_shape": use_seasonal_hourly_shape,
        },
        "pre_run_artifacts": _run_spec_artifact_bindings(entries),
    }
    run_spec_sha256 = _canonical_mapping_sha256(run_spec)
    manifest: dict[str, object] = {
        "schema_version": PRE_RUN_SCHEMA,
        "run_id": str(run_id),
        "valuation_timestamp": str(valuation_timestamp),
        "captured_at_utc": build_timestamp,
        "artifacts": entries,
        "governed_run_spec": run_spec,
        "governed_run_spec_sha256": run_spec_sha256,
        "promotion_eligible": False,
    }
    _write_json_exclusive(staging / PRE_RUN_EVIDENCE_MANIFEST, manifest)
    return manifest


def verify_pre_run_governance_evidence(
    staging_path: str | Path,
    *,
    expected_run_id: str,
    expected_valuation_timestamp: str,
    require_only_pre_run_files: bool = False,
) -> dict[str, object]:
    staging = Path(staging_path).resolve()
    manifest = _load_json(staging / PRE_RUN_EVIDENCE_MANIFEST)
    if manifest.get("schema_version") != PRE_RUN_SCHEMA:
        raise CandidateEvidenceError("pre-run governance schema mismatch")
    if str(manifest.get("run_id", "")) != str(expected_run_id):
        raise CandidateEvidenceError("pre-run governance run_id mismatch")
    if str(manifest.get("valuation_timestamp", "")) != str(expected_valuation_timestamp):
        raise CandidateEvidenceError("pre-run governance valuation mismatch")
    entries = manifest.get("artifacts")
    allowed_roles = set(INDEPENDENT_EVIDENCE_ROLES) | {
        PRE_RUN_EEX_ROLE,
        PRE_RUN_CONFIG_ROLE,
    }
    if (
        not isinstance(entries, Mapping)
        or not (set(INDEPENDENT_EVIDENCE_ROLES) | {PRE_RUN_CONFIG_ROLE}).issubset(
            entries
        )
        or not set(entries).issubset(allowed_roles)
    ):
        raise CandidateEvidenceError("pre-run governance artifact roles mismatch")
    for role in sorted(INDEPENDENT_EVIDENCE_ROLES):
        entry = entries[role]
        if not isinstance(entry, Mapping):
            raise CandidateEvidenceError(f"pre-run governance {role} entry is invalid")
        receipt = entry.get("authority_receipt")
        if not isinstance(receipt, Mapping):
            raise CandidateEvidenceError(f"pre-run governance {role} receipt is missing")
        artifact, _ = _regular_file_within(staging, str(entry.get("path", "")), role=role)
        receipt_path, _ = _regular_file_within(
            staging,
            str(receipt.get("path", "")),
            role=f"{role}.authority_receipt",
        )
        if _sha256_file(artifact) != entry.get("sha256"):
            raise CandidateEvidenceError(f"pre-run governance {role} hash mismatch")
        verified = verify_model_governance_artifact_receipt(
            _load_json(receipt_path),
            artifact_path=artifact,
            expected_role=role,
            valuation_timestamp=expected_valuation_timestamp,
        )
        if receipt.get("receipt_id") != verified.get("receipt_id"):
            raise CandidateEvidenceError(f"pre-run governance {role} receipt mismatch")
    if PRE_RUN_EEX_ROLE in entries:
        eex_entry = entries[PRE_RUN_EEX_ROLE]
        if not isinstance(eex_entry, Mapping):
            raise CandidateEvidenceError("pre-run EEX entry is invalid")
        contract_entry = eex_entry.get("acquisition_contract")
        if not isinstance(contract_entry, Mapping):
            raise CandidateEvidenceError("pre-run EEX acquisition contract is missing")
        artifact, _ = _regular_file_within(
            staging,
            str(eex_entry.get("path", "")),
            role=PRE_RUN_EEX_ROLE,
        )
        contract, _ = _regular_file_within(
            staging,
            str(contract_entry.get("path", "")),
            role=f"{PRE_RUN_EEX_ROLE}.acquisition_contract",
        )
        if _sha256_file(artifact) != eex_entry.get("sha256"):
            raise CandidateEvidenceError("pre-run EEX source hash mismatch")
        if _sha256_file(contract) != contract_entry.get("sha256"):
            raise CandidateEvidenceError("pre-run EEX contract hash mismatch")
        verified = _verified_eex_acquisition(
            artifact,
            contract,
            valuation_timestamp=expected_valuation_timestamp,
        )
        for key in ("acquisition_id", "available_at_utc"):
            if contract_entry.get(key) != verified.get(key):
                raise CandidateEvidenceError(f"pre-run EEX {key} mismatch")
    config_entry = entries[PRE_RUN_CONFIG_ROLE]
    if not isinstance(config_entry, Mapping):
        raise CandidateEvidenceError("pre-run runtime config entry is invalid")
    config_path, _ = _regular_file_within(
        staging,
        str(config_entry.get("path", "")),
        role=PRE_RUN_CONFIG_ROLE,
    )
    config_bytes = config_path.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != config_entry.get("sha256"):
        raise CandidateEvidenceError("pre-run runtime config hash mismatch")
    try:
        config_payload = load_strict_yaml(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CandidateEvidenceError("pre-run runtime config is invalid") from exc
    semantic_sha256 = hashlib.sha256(
        json.dumps(
            config_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    if semantic_sha256 != config_entry.get("semantic_sha256"):
        raise CandidateEvidenceError("pre-run runtime config semantic hash mismatch")
    run_spec = manifest.get("governed_run_spec")
    if not isinstance(run_spec, Mapping):
        raise CandidateEvidenceError("pre-run governed run spec is missing")
    if run_spec.get("schema_version") != GOVERNED_RUN_SPEC_SCHEMA:
        raise CandidateEvidenceError("pre-run governed run spec schema mismatch")
    if str(run_spec.get("run_id", "")) != str(expected_run_id):
        raise CandidateEvidenceError("pre-run governed run spec run_id mismatch")
    if str(run_spec.get("as_of_utc", "")) != str(expected_valuation_timestamp):
        raise CandidateEvidenceError("pre-run governed run spec as_of mismatch")
    try:
        as_of = datetime.fromisoformat(str(run_spec.get("as_of_utc", "")))
        build_timestamp = datetime.fromisoformat(
            str(run_spec.get("build_timestamp_utc", ""))
        )
        run_started_at = datetime.fromisoformat(
            str(run_spec.get("run_started_at_utc", ""))
        )
    except ValueError as exc:
        raise CandidateEvidenceError("pre-run governed timestamps are invalid") from exc
    if (
        as_of.tzinfo is None
        or build_timestamp.tzinfo is None
        or run_started_at.tzinfo is None
        or run_started_at != build_timestamp
        or build_timestamp < as_of
        or manifest.get("captured_at_utc") != run_spec.get("build_timestamp_utc")
    ):
        raise CandidateEvidenceError("pre-run governed timestamps are inconsistent")
    runtime = run_spec.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("distribution") != "fmv-pfc-lt"
        or runtime.get("version") != __version__
        or not re.fullmatch(
            r"(?:[0-9a-f]{40}|[0-9a-f]{64})",
            str(runtime.get("source_revision", "")),
        )
        or SOURCE_REVISION is None
        or runtime.get("source_revision") != SOURCE_REVISION
    ):
        raise CandidateEvidenceError("pre-run governed runtime identity is invalid")
    config_binding = run_spec.get("config")
    expected_config_binding = {
        key: config_entry.get(key)
        for key in ("path", "sha256", "semantic_sha256")
    }
    if config_binding != expected_config_binding:
        raise CandidateEvidenceError("pre-run governed config binding mismatch")
    snapshot_binding = run_spec.get("input_snapshot")
    if not isinstance(snapshot_binding, Mapping):
        raise CandidateEvidenceError("pre-run governed input snapshot binding is missing")
    generation_id = str(snapshot_binding.get("generation_id", ""))
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", generation_id)
        or snapshot_binding.get("contract_logical_path")
        != (Path("snapshots") / generation_id / "lt_input_snapshot.json").as_posix()
        or snapshot_binding.get("pointer_logical_path") != "views/pfc_lt/current.json"
        or not _is_canonical_sha256(snapshot_binding.get("contract_sha256"))
        or not _is_canonical_sha256(snapshot_binding.get("pointer_sha256"))
        or (
            "publication_head_observation_sha256" in snapshot_binding
            and not _is_canonical_sha256(
                snapshot_binding.get("publication_head_observation_sha256")
            )
        )
        or (
            "publication_head_challenge_nonce" in snapshot_binding
            and not _is_canonical_sha256(
                snapshot_binding.get("publication_head_challenge_nonce")
            )
        )
    ):
        raise CandidateEvidenceError("pre-run governed input snapshot binding is invalid")
    if run_spec.get("pre_run_artifacts") != _run_spec_artifact_bindings(entries):
        raise CandidateEvidenceError("pre-run governed artifact bindings mismatch")
    model_controls = run_spec.get("model_controls")
    if (
        not isinstance(model_controls, Mapping)
        or set(model_controls) != {
            "peak_source_policy",
            "use_seasonal_hourly_shape",
        }
        or model_controls.get("peak_source_policy")
        not in {"same_first", "strict_same", "any"}
        or not isinstance(model_controls.get("use_seasonal_hourly_shape"), bool)
    ):
        raise CandidateEvidenceError("pre-run governed model controls are invalid")
    if manifest.get("governed_run_spec_sha256") != _canonical_mapping_sha256(
        run_spec
    ):
        raise CandidateEvidenceError("pre-run governed run spec hash mismatch")
    if require_only_pre_run_files:
        expected_files = {PRE_RUN_EVIDENCE_MANIFEST.as_posix()}
        for entry in entries.values():
            expected_files.add(str(entry["path"]))
            supporting = entry.get("authority_receipt") or entry.get(
                "acquisition_contract"
            )
            if supporting is not None:
                expected_files.add(str(supporting["path"]))
        actual_files = {
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file()
        }
        if actual_files != expected_files:
            raise CandidateEvidenceError(
                "pre-run staging contains files outside authenticated governance evidence"
            )
    return manifest


def _run_spec_artifact_bindings(
    entries: Mapping[str, object],
) -> dict[str, object]:
    bindings: dict[str, object] = {}
    for role, raw_entry in sorted(entries.items()):
        if not isinstance(raw_entry, Mapping):
            continue
        binding: dict[str, object] = {
            "path": raw_entry.get("path"),
            "sha256": raw_entry.get("sha256"),
        }
        support = raw_entry.get("authority_receipt") or raw_entry.get(
            "acquisition_contract"
        )
        if isinstance(support, Mapping):
            binding["supporting_evidence_sha256"] = support.get("sha256")
        bindings[str(role)] = binding
    return bindings


def _canonical_mapping_sha256(payload: Mapping[str, object]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _is_canonical_sha256(value: object) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value)))


def _verified_eex_acquisition(
    artifact_path: Path,
    contract_path: Path,
    *,
    valuation_timestamp: str,
) -> dict[str, object]:
    if not artifact_path.is_file() or not contract_path.is_file():
        raise CandidateEvidenceError("EEX acquisition evidence is unavailable")
    try:
        contract = _load_json(contract_path)
        unsigned = verify_acquisition_contract(contract)
    except (CandidateEvidenceError, AcquisitionAuthenticationError) as exc:
        raise CandidateEvidenceError("EEX acquisition contract is not authentic") from exc
    if unsigned.get("schema_version") not in GOVERNED_LT_INPUT_SNAPSHOT_SCHEMAS:
        raise CandidateEvidenceError(
            "EEX acquisition requires a governed v2/v3 source contract"
        )
    files = unsigned.get("files")
    if not isinstance(files, Mapping) or set(files) != {PRE_RUN_EEX_ROLE}:
        raise CandidateEvidenceError("EEX acquisition contract must contain only the forward source")
    entry = files.get(PRE_RUN_EEX_ROLE) if isinstance(files, Mapping) else None
    if not isinstance(entry, Mapping):
        raise CandidateEvidenceError("EEX acquisition contract has no forward source")
    if unsigned.get("calibration_eligible") is not True or entry.get(
        "calibration_eligible"
    ) is not True:
        raise CandidateEvidenceError("EEX acquisition source is not calibration eligible")
    if unsigned.get("source_class") != "GOVERNED_ACQUISITION":
        raise CandidateEvidenceError("EEX acquisition source class is not governed")
    try:
        source_system = validate_governed_source_system(PRE_RUN_EEX_ROLE, entry)
    except ValueError as exc:
        raise CandidateEvidenceError("EEX acquisition source system is not admissible") from exc
    acquisition_id = str(unsigned.get("acquisition_id", "")).strip()
    entry_acquisition_id = str(entry.get("acquisition_id", "")).strip()
    if not acquisition_id or not entry_acquisition_id:
        raise CandidateEvidenceError("EEX acquisition identity is missing")
    if entry_acquisition_id != acquisition_id:
        raise CandidateEvidenceError("EEX acquisition identity is divergent")
    if str(entry.get("sha256", "")) != _sha256_file(artifact_path):
        raise CandidateEvidenceError("EEX acquisition artifact hash mismatch")
    if int(entry.get("size_bytes", -1)) != artifact_path.stat().st_size:
        raise CandidateEvidenceError("EEX acquisition artifact size mismatch")
    raw_artifact = entry.get("raw_artifact")
    if not isinstance(raw_artifact, Mapping):
        raise CandidateEvidenceError("EEX acquisition raw artifact binding is missing")
    if str(raw_artifact.get("sha256", "")) != _sha256_file(artifact_path):
        raise CandidateEvidenceError("EEX acquisition raw artifact hash mismatch")
    if int(raw_artifact.get("size_bytes", -1)) != artifact_path.stat().st_size:
        raise CandidateEvidenceError("EEX acquisition raw artifact size mismatch")
    available = pd.Timestamp(entry.get("available_at_utc"))
    contract_available = pd.Timestamp(unsigned.get("available_at_utc"))
    valuation = pd.Timestamp(valuation_timestamp)
    if (
        available.tzinfo is None
        or contract_available.tzinfo is None
        or valuation.tzinfo is None
        or available > valuation
        or contract_available > valuation
    ):
        raise CandidateEvidenceError("EEX acquisition availability is not point-in-time")
    return {
        "acquisition_id": acquisition_id,
        "available_at_utc": available.tz_convert("UTC").isoformat(),
        "source_system": source_system,
    }


def seal_candidate_evidence(
    staging_path: str | Path,
    *,
    run_id: str,
    artifacts: Mapping[str, EvidenceArtifactInput],
    _allow_assembled_extra: bool = False,
) -> dict[str, object]:
    """Seal the exact pre-finalization evidence DAG and bind it to the run."""

    staging = Path(staging_path).resolve()
    if not staging.is_dir():
        raise CandidateEvidenceError(f"candidate staging is unavailable: {staging}")
    manifest_path = staging / EVIDENCE_MANIFEST
    _require_exact_roles(artifacts, allow_assembled_extra=_allow_assembled_extra)
    entries: dict[str, object] = {}
    occupied_paths: dict[str, str] = {}
    run_manifest = _load_json(staging / CANDIDATE_RUN_MANIFEST)
    try:
        from pfc_shaping.pipeline.candidate_bundle import (
            verify_deterministic_artifact_inventory,
        )

        verify_deterministic_artifact_inventory(staging, run_manifest)
    except RuntimeError as exc:
        raise CandidateEvidenceError(
            "candidate deterministic artifacts are incomplete or invalid"
        ) from exc
    valuation_timestamp = run_manifest.get("reference_timestamp")
    pre_run = _verified_bound_pre_run(
        staging,
        run_manifest=run_manifest,
        expected_run_id=run_id,
    )
    pre_run_entries = pre_run["artifacts"]
    for role in sorted(artifacts):
        item = artifacts[role]
        producer_contract = str(item.producer_contract).strip()
        if not producer_contract:
            raise CandidateEvidenceError(f"{role} producer_contract is missing")
        path, relative = _regular_file_within(staging, item.path, role=role)
        _claim_unique_path(occupied_paths, relative, role=role)
        entry: dict[str, object] = {
            "path": relative,
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
            "trust_class": (
                "independent_external"
                if role in INDEPENDENT_EVIDENCE_ROLES
                else "candidate_derived"
            ),
            "producer_contract": producer_contract,
            "dependencies": list(ROLE_DEPENDENCIES[role]),
        }
        if role in INDEPENDENT_EVIDENCE_ROLES:
            if item.authority_receipt_path is None:
                raise CandidateEvidenceError(f"{role} authority receipt is missing")
            receipt, receipt_relative = _regular_file_within(
                staging,
                item.authority_receipt_path,
                role=f"{role}_authority_receipt",
            )
            pre_run_entry = pre_run_entries[role]
            pre_run_receipt = pre_run_entry["authority_receipt"]
            if relative != pre_run_entry.get("path"):
                raise CandidateEvidenceError(
                    f"{role} is not the artifact captured before the solve"
                )
            if receipt_relative != pre_run_receipt.get("path"):
                raise CandidateEvidenceError(
                    f"{role} receipt is not the receipt captured before the solve"
                )
            _claim_unique_path(
                occupied_paths,
                receipt_relative,
                role=f"{role}_authority_receipt",
            )
            receipt_payload = _load_json(receipt)
            try:
                verified_receipt = verify_model_governance_artifact_receipt(
                    receipt_payload,
                    artifact_path=path,
                    expected_role=role,
                    valuation_timestamp=valuation_timestamp,
                )
            except ModelGovernanceAuthenticationError as exc:
                raise CandidateEvidenceError(
                    f"{role} authority receipt is not authentic"
                ) from exc
            entry["authority_receipt"] = {
                "path": receipt_relative,
                "sha256": _sha256_file(receipt),
                "size_bytes": receipt.stat().st_size,
                "receipt_id": verified_receipt["receipt_id"],
                "authority_id": verified_receipt["authority_id"],
                "issued_at_utc": verified_receipt["issued_at_utc"],
                "data_cutoff_utc": verified_receipt["data_cutoff_utc"],
            }
        elif item.authority_receipt_path is not None:
            raise CandidateEvidenceError(
                f"{role} cannot claim an independent authority receipt"
            )
        entries[role] = entry

    payload: dict[str, object] = {
        "schema_version": EVIDENCE_SCHEMA,
        "run_id": str(run_id),
        "promotion_eligible": False,
        "evidence_reference_timestamp": str(valuation_timestamp),
        "artifacts": entries,
        "dependency_order": _dependency_order(entries),
        "candidate_run_preseal_sha256": _candidate_run_preseal_sha256(run_manifest),
        "pre_run_governance_manifest_sha256": str(
            run_manifest["pre_run_governance_manifest_sha256"]
        ),
    }
    _materialize_exact_json_atomic(manifest_path, payload)
    manifest_hash = _sha256_file(manifest_path)
    _bind_evidence_to_candidate_run(
        staging,
        run_id=run_id,
        evidence_manifest_sha256=manifest_hash,
    )
    verify_candidate_evidence(staging, expected_run_id=run_id)
    return payload


def seal_assembled_candidate_evidence(
    staging_path: str | Path,
    *,
    run_id: str,
) -> dict[str, object]:
    """Seal only canonical artifacts replayed from the staged LT candidate."""

    staging = Path(staging_path).resolve()
    artifacts = _canonical_assembled_artifacts(staging, run_id=run_id)
    payload = seal_candidate_evidence(
        staging,
        run_id=run_id,
        artifacts=artifacts,
        _allow_assembled_extra=True,
    )
    verify_assembled_candidate_evidence(staging, expected_run_id=run_id)
    return payload


def verify_assembled_candidate_evidence(
    bundle_path: str | Path,
    *,
    expected_run_id: str,
) -> dict[str, object]:
    """Replay both evidence assemblers and enforce the canonical seal inventory."""

    root = Path(bundle_path).resolve()
    manifest = verify_candidate_evidence(root, expected_run_id=expected_run_id)
    expected_inputs = _canonical_assembled_artifacts(root, run_id=expected_run_id)
    entries = manifest.get("artifacts")
    if not isinstance(entries, Mapping):
        raise CandidateEvidenceError("assembled candidate evidence inventory is invalid")
    for role, item in expected_inputs.items():
        path, relative = _regular_file_within(root, item.path, role=role)
        entry = entries.get(role)
        if not isinstance(entry, Mapping):
            raise CandidateEvidenceError(f"assembled candidate evidence role is missing: {role}")
        if (
            entry.get("path") != relative
            or entry.get("sha256") != _sha256_file(path)
            or entry.get("size_bytes") != path.stat().st_size
            or entry.get("producer_contract") != item.producer_contract
        ):
            raise CandidateEvidenceError(
                f"assembled candidate evidence role is not canonical: {role}"
            )
    return manifest


def _canonical_assembled_artifacts(
    staging: Path,
    *,
    run_id: str,
) -> dict[str, EvidenceArtifactInput]:
    from pfc_shaping.pipeline.candidate_evidence_assembler import (
        verify_candidate_derived_evidence,
    )
    from pfc_shaping.pipeline.candidate_product_evidence import (
        PRODUCT_EVIDENCE_MANIFEST,
        verify_candidate_product_evidence,
    )

    assembly = verify_candidate_derived_evidence(staging, expected_run_id=run_id)
    product = verify_candidate_product_evidence(staging, expected_run_id=run_id)
    sources = assembly.get("sources")
    derived = assembly.get("derived")
    product_artifacts = product.get("artifacts")
    if not all(
        isinstance(value, Mapping)
        for value in (sources, derived, product_artifacts)
    ):
        raise CandidateEvidenceError("canonical assembled evidence manifests are invalid")
    pre_run = verify_pre_run_governance_evidence(
        staging,
        expected_run_id=run_id,
        expected_valuation_timestamp=str(
            _load_json(staging / CANDIDATE_RUN_MANIFEST).get("reference_timestamp", "")
        ),
    )
    independent = pre_run.get("artifacts")
    if not isinstance(independent, Mapping):
        raise CandidateEvidenceError("canonical independent evidence is missing")

    def artifact_path(container: Mapping[str, object], role: str) -> str:
        entry = container.get(role)
        if not isinstance(entry, Mapping) or not entry.get("path"):
            raise CandidateEvidenceError(f"canonical evidence path is missing: {role}")
        return str(entry["path"])

    def authority_path(role: str) -> str:
        entry = independent.get(role)
        receipt = entry.get("authority_receipt") if isinstance(entry, Mapping) else None
        if not isinstance(receipt, Mapping) or not receipt.get("path"):
            raise CandidateEvidenceError(f"canonical authority receipt is missing: {role}")
        return str(receipt["path"])

    return {
        "historical_thresholds": EvidenceArtifactInput(
            path=artifact_path(sources, "historical_thresholds"),
            producer_contract="pfc.model_governance.historical_thresholds.v1",
            authority_receipt_path=authority_path("historical_thresholds"),
        ),
        "selected_lambda_decision": EvidenceArtifactInput(
            path=artifact_path(sources, "selected_lambda_decision"),
            producer_contract="pfc.model_governance.selected_lambda_decision.v1",
            authority_receipt_path=authority_path("selected_lambda_decision"),
        ),
        "production_manifest": EvidenceArtifactInput(
            path=artifact_path(sources, "production_manifest"),
            producer_contract="pfc.pipeline.monthly_curve_authority.v1",
        ),
        "export_manifest": EvidenceArtifactInput(
            path=artifact_path(derived, "export_manifest"),
            producer_contract="pfc.pipeline.hourly_export.v1",
        ),
        "audit_gates": EvidenceArtifactInput(
            path=artifact_path(derived, "monthly_audit_gates"),
            producer_contract="pfc.pipeline.monthly_curve_audit.v1",
        ),
        "product_normalization_summary": EvidenceArtifactInput(
            path=artifact_path(product_artifacts, "product_normalization_summary"),
            producer_contract="pfc.pipeline.product_normalization_evidence.v1",
        ),
        "product_normalization_evidence": EvidenceArtifactInput(
            path=PRODUCT_EVIDENCE_MANIFEST.as_posix(),
            producer_contract="pfc.pipeline.product_normalization_evidence.v1",
        ),
        "selected_config_run_parity": EvidenceArtifactInput(
            path=artifact_path(derived, "selected_config_run_parity"),
            producer_contract="pfc.pipeline.selected_config_run_parity.v1",
        ),
    }


def verify_candidate_evidence(
    bundle_path: str | Path,
    *,
    expected_run_id: str,
) -> dict[str, object]:
    bundle = Path(bundle_path).resolve()
    manifest_path = bundle / EVIDENCE_MANIFEST
    manifest = _load_json(manifest_path)
    errors: list[str] = []
    if set(manifest) != {
        "schema_version",
        "run_id",
        "promotion_eligible",
        "evidence_reference_timestamp",
        "artifacts",
        "dependency_order",
        "candidate_run_preseal_sha256",
        "pre_run_governance_manifest_sha256",
    }:
        errors.append("top_level_keys")
    pre_run: Mapping[str, object] = {"artifacts": {}}
    if manifest.get("schema_version") != EVIDENCE_SCHEMA:
        errors.append("schema_version")
    if str(manifest.get("run_id", "")) != str(expected_run_id):
        errors.append("run_id")
    if manifest.get("promotion_eligible") is not False:
        errors.append("candidate_evidence_must_not_self_approve")
    entries = manifest.get("artifacts")
    if not isinstance(entries, Mapping):
        raise CandidateEvidenceError("candidate evidence artifacts are invalid")
    entry_roles = set(entries)
    expected_roles = set(REQUIRED_EVIDENCE_ROLES)
    if entry_roles == expected_roles | set(ASSEMBLED_EXTRA_EVIDENCE_ROLES):
        expected_roles = entry_roles
    elif entry_roles != expected_roles:
        errors.append("artifact_roles")
    if entry_roles == expected_roles:
        occupied_paths: dict[str, str] = {}
        run_manifest = _load_json(bundle / CANDIDATE_RUN_MANIFEST)
        if str(manifest.get("evidence_reference_timestamp", "")) != str(
            run_manifest.get("reference_timestamp", "")
        ):
            errors.append("evidence_reference_timestamp")
        try:
            pre_run = _verified_bound_pre_run(
                bundle,
                run_manifest=run_manifest,
                expected_run_id=expected_run_id,
            )
        except CandidateEvidenceError:
            errors.append("pre_run_governance_binding")
            pre_run = {"artifacts": {}}
        pre_run_entries = pre_run["artifacts"]
        valuation_timestamp = run_manifest.get("reference_timestamp")
        for role in sorted(expected_roles):
            entry = entries[role]
            if not isinstance(entry, Mapping):
                errors.append(f"{role}.entry")
                continue
            expected_trust = (
                "independent_external"
                if role in INDEPENDENT_EVIDENCE_ROLES
                else "candidate_derived"
            )
            if entry.get("trust_class") != expected_trust:
                errors.append(f"{role}.trust_class")
            if list(entry.get("dependencies", [])) != list(ROLE_DEPENDENCIES[role]):
                errors.append(f"{role}.dependencies")
            if not str(entry.get("producer_contract", "")).strip():
                errors.append(f"{role}.producer_contract")
            _verify_entry_file(bundle, entry, role=role, errors=errors)
            _record_entry_path(entry, occupied_paths, role=role, errors=errors)
            receipt = entry.get("authority_receipt")
            if role in INDEPENDENT_EVIDENCE_ROLES:
                if not isinstance(receipt, Mapping):
                    errors.append(f"{role}.authority_receipt")
                else:
                    pre_run_entry = pre_run_entries.get(role)
                    if not isinstance(pre_run_entry, Mapping):
                        errors.append(f"{role}.pre_run_entry")
                    else:
                        pre_run_receipt = pre_run_entry.get("authority_receipt")
                        if entry.get("path") != pre_run_entry.get("path"):
                            errors.append(f"{role}.pre_run_artifact_path")
                        if not isinstance(pre_run_receipt, Mapping) or receipt.get(
                            "path"
                        ) != pre_run_receipt.get("path"):
                            errors.append(f"{role}.pre_run_receipt_path")
                    _verify_entry_file(
                        bundle,
                        receipt,
                        role=f"{role}.authority_receipt",
                        errors=errors,
                    )
                    _record_entry_path(
                        receipt,
                        occupied_paths,
                        role=f"{role}.authority_receipt",
                        errors=errors,
                    )
                    try:
                        artifact_path, _ = _regular_file_within(
                            bundle,
                            str(entry.get("path", "")),
                            role=role,
                        )
                        receipt_path, _ = _regular_file_within(
                            bundle,
                            str(receipt.get("path", "")),
                            role=f"{role}.authority_receipt",
                        )
                        verified_receipt = verify_model_governance_artifact_receipt(
                            _load_json(receipt_path),
                            artifact_path=artifact_path,
                            expected_role=role,
                            valuation_timestamp=valuation_timestamp,
                        )
                    except (CandidateEvidenceError, ModelGovernanceAuthenticationError):
                        errors.append(f"{role}.authority_authentication")
                    else:
                        for key in (
                            "receipt_id",
                            "authority_id",
                            "issued_at_utc",
                            "data_cutoff_utc",
                        ):
                            if receipt.get(key) != verified_receipt.get(key):
                                errors.append(f"{role}.authority_{key}")
            elif receipt is not None:
                errors.append(f"{role}.unexpected_authority_receipt")
        try:
            order = _dependency_order(entries)
        except CandidateEvidenceError:
            errors.append("dependency_cycle")
        else:
            if manifest.get("dependency_order") != order:
                errors.append("dependency_order")

    run_manifest = _load_json(bundle / CANDIDATE_RUN_MANIFEST)
    if str(run_manifest.get("run_id", "")) != str(expected_run_id):
        errors.append("candidate_run_id")
    expected_path = EVIDENCE_MANIFEST.as_posix()
    if run_manifest.get("candidate_evidence_manifest") != expected_path:
        errors.append("candidate_run_evidence_path")
    if run_manifest.get("candidate_evidence_manifest_sha256") != _sha256_file(
        manifest_path
    ):
        errors.append("candidate_run_evidence_hash")
    if manifest.get("candidate_run_preseal_sha256") != _candidate_run_preseal_sha256(
        run_manifest
    ):
        errors.append("candidate_run_preseal_sha256")
    if manifest.get("pre_run_governance_manifest_sha256") != run_manifest.get(
        "pre_run_governance_manifest_sha256"
    ):
        errors.append("pre_run_governance_manifest_sha256")
    selected_entry = pre_run.get("artifacts", {}).get("selected_lambda_decision")
    if not isinstance(selected_entry, Mapping):
        errors.append("selected_lambda_decision.pre_run_entry")
    else:
        try:
            selected_path, _ = _regular_file_within(
                bundle,
                str(selected_entry.get("path", "")),
                role="selected_lambda_decision",
            )
            selected_decision = _load_json(selected_path)
            expected_config = _candidate_active_monthly_config(bundle, run_manifest)
            validate_selected_lambda_decision(
                selected_decision,
                expected_config=expected_config,
            )
        except (CandidateEvidenceError, ModelGovernanceAuthenticationError, ValueError):
            errors.append("selected_lambda_decision.semantic_validation")
    if errors:
        raise CandidateEvidenceError(
            "candidate evidence contract mismatch: " + ",".join(sorted(set(errors)))
        )
    return manifest


def _bind_evidence_to_candidate_run(
    staging: Path,
    *,
    run_id: str,
    evidence_manifest_sha256: str,
) -> None:
    path = staging / CANDIDATE_RUN_MANIFEST
    manifest = _load_json(path)
    if str(manifest.get("run_id", "")) != str(run_id):
        raise CandidateEvidenceError("candidate run/evidence run_id mismatch")
    manifest["candidate_evidence_manifest"] = EVIDENCE_MANIFEST.as_posix()
    manifest["candidate_evidence_manifest_sha256"] = evidence_manifest_sha256
    _atomic_write_json(path, manifest)


def _verified_bound_pre_run(
    bundle: Path,
    *,
    run_manifest: Mapping[str, object],
    expected_run_id: str,
) -> dict[str, object]:
    declared_path = str(run_manifest.get("pre_run_governance_manifest", ""))
    declared_hash = str(run_manifest.get("pre_run_governance_manifest_sha256", ""))
    if declared_path != PRE_RUN_EVIDENCE_MANIFEST.as_posix() or len(declared_hash) != 64:
        raise CandidateEvidenceError("candidate run pre-run governance binding is missing")
    path = bundle / PRE_RUN_EVIDENCE_MANIFEST
    if _sha256_file(path) != declared_hash:
        raise CandidateEvidenceError("candidate run pre-run governance hash mismatch")
    return verify_pre_run_governance_evidence(
        bundle,
        expected_run_id=expected_run_id,
        expected_valuation_timestamp=str(run_manifest.get("reference_timestamp", "")),
    )


def _candidate_run_preseal_sha256(run_manifest: Mapping[str, object]) -> str:
    payload = dict(run_manifest)
    payload.pop("candidate_evidence_manifest", None)
    payload.pop("candidate_evidence_manifest_sha256", None)
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _candidate_active_monthly_config(
    bundle: Path,
    run_manifest: Mapping[str, object],
) -> Mapping[str, object] | None:
    config_value = str(run_manifest.get("config_path", "")).strip()
    if not config_value:
        return None
    config_path, _ = _regular_file_within(bundle, config_value, role="candidate_config")
    try:
        config = load_strict_yaml(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise CandidateEvidenceError("candidate config is unreadable") from exc
    if not isinstance(config, Mapping):
        raise CandidateEvidenceError("candidate config must contain a mapping")
    from pfc_shaping.pipeline.monthly_curve_authority import (
        active_monthly_curve_config_payload,
        monthly_solver_settings,
    )

    return active_monthly_curve_config_payload(monthly_solver_settings(config))


def _require_exact_roles(
    artifacts: Mapping[str, EvidenceArtifactInput],
    *,
    allow_assembled_extra: bool = False,
) -> None:
    expected = set(REQUIRED_EVIDENCE_ROLES)
    if allow_assembled_extra:
        expected.update(ASSEMBLED_EXTRA_EVIDENCE_ROLES)
    missing = sorted(expected.difference(artifacts))
    unexpected = sorted(set(artifacts).difference(expected))
    if missing or unexpected:
        raise CandidateEvidenceError(
            f"candidate evidence roles mismatch: missing={missing}, unexpected={unexpected}"
        )


def _claim_unique_path(paths: dict[str, str], relative: str, *, role: str) -> None:
    previous = paths.get(relative)
    if previous is not None:
        raise CandidateEvidenceError(
            f"candidate evidence path is reused by {previous} and {role}: {relative}"
        )
    paths[relative] = role


def _record_entry_path(
    entry: Mapping[str, object],
    paths: dict[str, str],
    *,
    role: str,
    errors: list[str],
) -> None:
    relative = str(entry.get("path", ""))
    previous = paths.get(relative)
    if previous is not None:
        errors.append(f"{role}.path_reused_with_{previous}")
    else:
        paths[relative] = role


def _pre_run_source(value: str | Path, *, role: str) -> Path:
    lexical = Path(value).expanduser()
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    lexical = Path(os.path.abspath(lexical))
    for component in (lexical, *lexical.parents):
        try:
            linked = path_is_link(component)
        except OSError as exc:
            raise CandidateEvidenceError(f"cannot validate {role} source path") from exc
        if linked:
            raise CandidateEvidenceError(f"{role} source path cannot contain a link")
    if not lexical.is_file():
        raise CandidateEvidenceError(f"{role} pre-run source evidence is unavailable")
    return lexical.resolve()


def _regular_file_within(
    root: Path,
    value: str | Path,
    *,
    role: str,
) -> tuple[Path, str]:
    lexical = Path(value)
    if not lexical.is_absolute():
        lexical = root / lexical
    if path_is_link(lexical):
        raise CandidateEvidenceError(f"{role} cannot be a link")
    resolved = lexical.resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise CandidateEvidenceError(f"{role} escapes candidate staging") from exc
    if not resolved.is_file():
        raise CandidateEvidenceError(f"{role} is not a regular file: {resolved}")
    return resolved, relative


def _verify_entry_file(
    root: Path,
    entry: Mapping[str, object],
    *,
    role: str,
    errors: list[str],
) -> None:
    try:
        path, relative = _regular_file_within(root, str(entry.get("path", "")), role=role)
    except CandidateEvidenceError:
        errors.append(f"{role}.path")
        return
    if relative != str(entry.get("path", "")):
        errors.append(f"{role}.noncanonical_path")
    if _sha256_file(path) != str(entry.get("sha256", "")):
        errors.append(f"{role}.sha256")
    if path.stat().st_size != entry.get("size_bytes"):
        errors.append(f"{role}.size_bytes")


def _dependency_order(entries: Mapping[str, object]) -> list[str]:
    pending = {
        role: set(entry.get("dependencies", []))
        for role, entry in entries.items()
        if isinstance(entry, Mapping)
    }
    if set(pending) != set(entries):
        raise CandidateEvidenceError("candidate evidence contains invalid dependency entries")
    order: list[str] = []
    while pending:
        ready = sorted(role for role, dependencies in pending.items() if not dependencies)
        if not ready:
            raise CandidateEvidenceError("candidate evidence dependency graph contains a cycle")
        order.extend(ready)
        for role in ready:
            pending.pop(role)
        for dependencies in pending.values():
            dependencies.difference_update(ready)
    return order


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = load_strict_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise CandidateEvidenceError(f"invalid JSON evidence file: {path}") from exc
    if not isinstance(payload, dict):
        raise CandidateEvidenceError(f"evidence JSON must contain an object: {path}")
    return payload


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _json_bytes(payload)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise CandidateEvidenceError(f"refusing to overwrite evidence: {path}") from exc
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise CandidateEvidenceError(f"refusing to overwrite evidence: {path}") from exc
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(candidate_temporary_name(suffix="tmp"))
    try:
        _write_json_exclusive(temporary, payload)
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def _materialize_exact_json_atomic(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    raw = _json_bytes(payload)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != raw:
            raise CandidateEvidenceError(f"sealed evidence differs from replay: {path}")
        return
    temporary = path.with_name(candidate_temporary_name(suffix="tmp"))
    try:
        _write_bytes_exclusive(temporary, raw)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise CandidateEvidenceError(f"sealed evidence publish collision: {path}")
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
