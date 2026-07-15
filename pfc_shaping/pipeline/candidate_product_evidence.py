"""Build policy-bound delivered-product evidence from a staged LT candidate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import pandas as pd

from pfc_shaping.pipeline.candidate_evidence_assembler import (
    ASSEMBLY_MANIFEST,
    CandidateEvidenceAssemblyError,
    verify_candidate_derived_evidence,
)
from pfc_shaping.pipeline.process_identity import candidate_temporary_name
from pfc_shaping.pipeline.quote_conflict_policy_contract import (
    QuoteConflictPolicyAuthenticationError,
    verify_quote_conflict_policy,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json
from pfc_shaping.validation.product_normalization import run_audit

INVENTORY_SCHEMA = "candidate_quote_conflict_inventory.v1"
PRODUCT_EVIDENCE_SCHEMA = "candidate_product_normalization_evidence.v1"
DRAFT_GATES = Path("audits") / "product_normalization_gates_prepolicy.csv"
CONFLICT_INVENTORY = Path("manifests") / "quote_conflict_inventory.json"
STAGED_POLICY = (
    Path("evidence") / "independent" / "source_hierarchy_policy" / "policy.json"
)
FINAL_GATES = Path("audits") / "product_normalization_gates.csv"
FINAL_SUMMARY = Path("manifests") / "product_normalization_summary.json"
PRODUCT_EVIDENCE_MANIFEST = Path("manifests") / "product_normalization_evidence.json"


class CandidateProductEvidenceError(RuntimeError):
    """Raised when delivered-product evidence is incomplete or inconsistent."""


def assemble_candidate_product_evidence(
    staging_path: str | Path,
    *,
    run_id: str,
    source_hierarchy_policy_path: str | Path | None = None,
) -> dict[str, object]:
    staging = Path(staging_path).resolve()
    try:
        assembly = verify_candidate_derived_evidence(
            staging,
            expected_run_id=run_id,
        )
    except CandidateEvidenceAssemblyError as exc:
        raise CandidateProductEvidenceError("base evidence assembly is invalid") from exc
    paths = _bound_paths(staging, assembly)
    expected_gates, expected_inventory = _draft_inventory(
        staging,
        run_id=run_id,
        paths=paths,
        assembly=assembly,
    )
    draft_path = staging / DRAFT_GATES
    inventory_path = staging / CONFLICT_INVENTORY
    _materialize_exact_bytes(
        draft_path,
        _csv_bytes(expected_gates),
        label="product pre-policy gates",
    )
    _materialize_exact_json(
        inventory_path,
        expected_inventory,
        label="quote-conflict inventory",
    )

    if source_hierarchy_policy_path is None:
        return {
            "schema_version": "candidate_product_evidence_status.v1",
            "run_id": str(run_id),
            "status": "SOURCE_HIERARCHY_POLICY_REQUIRED",
            "promotion_eligible": False,
            "quote_conflict_inventory": CONFLICT_INVENTORY.as_posix(),
            "required_policy_payload": expected_inventory["required_policy_payload"],
        }

    if (staging / PRODUCT_EVIDENCE_MANIFEST).is_file():
        return verify_candidate_product_evidence(staging, expected_run_id=run_id)
    external_policy = Path(source_hierarchy_policy_path).resolve()
    try:
        policy_bytes = external_policy.read_bytes()
        signed_policy = _load_json_bytes(policy_bytes, path=external_policy)
        verify_quote_conflict_policy(signed_policy)
    except (OSError, CandidateProductEvidenceError, QuoteConflictPolicyAuthenticationError) as exc:
        raise CandidateProductEvidenceError("source hierarchy policy is not authentic") from exc
    policy_path = staging / STAGED_POLICY
    if policy_path.is_file() and policy_path.read_bytes() != policy_bytes:
        raise CandidateProductEvidenceError("staged source hierarchy policy changed")
    pending_policy_path = policy_path.with_name(
        candidate_temporary_name(suffix="pending.json")
    )
    audit_policy_path = policy_path
    if not policy_path.is_file():
        _materialize_exact_bytes(
            pending_policy_path,
            policy_bytes,
            label="pending source hierarchy policy",
        )
        audit_policy_path = pending_policy_path
    try:
        gates, summary = _run_product_audit(paths, policy_path=audit_policy_path)
    except Exception:
        if audit_policy_path == pending_policy_path:
            pending_policy_path.unlink(missing_ok=True)
        raise
    policy_eval = summary.get("source_hierarchy_policy")
    if (
        not isinstance(policy_eval, Mapping)
        or policy_eval.get("status") != "ACCEPTED_PRODUCTION_APPROVED"
        or int(summary.get("blocking_quote_conflict_count", -1)) != 0
        or summary.get("all_gates_pass") is not True
    ):
        if audit_policy_path == pending_policy_path:
            pending_policy_path.unlink(missing_ok=True)
        raise CandidateProductEvidenceError(
            "signed source hierarchy policy or delivered-product gates did not pass"
        )
    _materialize_exact_bytes(
        policy_path,
        policy_bytes,
        label="source hierarchy policy",
    )
    pending_policy_path.unlink(missing_ok=True)
    gates, summary = _run_product_audit(paths, policy_path=policy_path)
    summary = _portable_summary(staging, summary)
    policy_eval = summary.get("source_hierarchy_policy")
    if (
        not isinstance(policy_eval, Mapping)
        or policy_eval.get("status") != "ACCEPTED_PRODUCTION_APPROVED"
        or int(summary.get("blocking_quote_conflict_count", -1)) != 0
        or summary.get("all_gates_pass") is not True
    ):
        raise CandidateProductEvidenceError(
            "signed source hierarchy policy or delivered-product gates did not pass"
        )
    gates_path = staging / FINAL_GATES
    summary_path = staging / FINAL_SUMMARY
    _materialize_exact_bytes(
        gates_path,
        _csv_bytes(gates),
        label="product normalization gates",
    )
    _materialize_exact_json(
        summary_path,
        summary,
        label="product normalization summary",
    )
    evidence = _expected_product_evidence_manifest(
        staging,
        run_id=run_id,
        assembly=assembly,
        paths=paths,
        policy_path=policy_path,
        gates_path=gates_path,
        summary_path=summary_path,
    )
    _materialize_exact_json(
        staging / PRODUCT_EVIDENCE_MANIFEST,
        evidence,
        label="product normalization evidence manifest",
    )
    return verify_candidate_product_evidence(staging, expected_run_id=run_id)


def verify_candidate_product_evidence(
    staging_path: str | Path,
    *,
    expected_run_id: str,
) -> dict[str, object]:
    staging = Path(staging_path).resolve()
    try:
        assembly = verify_candidate_derived_evidence(
            staging,
            expected_run_id=expected_run_id,
        )
    except CandidateEvidenceAssemblyError as exc:
        raise CandidateProductEvidenceError("base evidence assembly is invalid") from exc
    paths = _bound_paths(staging, assembly)
    expected_draft, expected_inventory = _draft_inventory(
        staging,
        run_id=expected_run_id,
        paths=paths,
        assembly=assembly,
    )
    draft_path = _regular_file(staging, DRAFT_GATES, role="product_prepolicy_gates")
    inventory_path = _regular_file(
        staging,
        CONFLICT_INVENTORY,
        role="quote_conflict_inventory",
    )
    if draft_path.read_bytes() != _csv_bytes(expected_draft):
        raise CandidateProductEvidenceError("product pre-policy gates replay mismatch")
    if _load_json(inventory_path) != expected_inventory:
        raise CandidateProductEvidenceError("quote-conflict inventory replay mismatch")
    policy_path = _regular_file(staging, STAGED_POLICY, role="source_hierarchy_policy")
    try:
        verify_quote_conflict_policy(_load_json(policy_path))
    except QuoteConflictPolicyAuthenticationError as exc:
        raise CandidateProductEvidenceError("staged source hierarchy policy is invalid") from exc
    gates, summary = _run_product_audit(paths, policy_path=policy_path)
    summary = _portable_summary(staging, summary)
    policy_eval = summary.get("source_hierarchy_policy")
    if (
        not isinstance(policy_eval, Mapping)
        or policy_eval.get("status") != "ACCEPTED_PRODUCTION_APPROVED"
        or int(summary.get("blocking_quote_conflict_count", -1)) != 0
        or summary.get("all_gates_pass") is not True
    ):
        raise CandidateProductEvidenceError("product evidence no longer passes its hard gates")
    gates_path = _regular_file(staging, FINAL_GATES, role="product_normalization_gates")
    summary_path = _regular_file(
        staging,
        FINAL_SUMMARY,
        role="product_normalization_summary",
    )
    if gates_path.read_bytes() != _csv_bytes(gates):
        raise CandidateProductEvidenceError("product normalization gates replay mismatch")
    if _load_json(summary_path) != summary:
        raise CandidateProductEvidenceError("product normalization summary replay mismatch")
    expected = _expected_product_evidence_manifest(
        staging,
        run_id=expected_run_id,
        assembly=assembly,
        paths=paths,
        policy_path=policy_path,
        gates_path=gates_path,
        summary_path=summary_path,
    )
    manifest = _load_json(staging / PRODUCT_EVIDENCE_MANIFEST)
    if manifest != expected:
        raise CandidateProductEvidenceError("product evidence manifest mismatch")
    return manifest


def _draft_inventory(
    staging: Path,
    *,
    run_id: str,
    paths: Mapping[str, Path],
    assembly: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    gates, summary = _run_product_audit(paths, policy_path=None)
    conflicts = list(summary.get("quote_conflict_identities", []))
    required_policy = {
        "schema_version": "ch_quote_conflict_source_hierarchy_policy.v1",
        "market": "CH",
        "forward_snapshot_date": str(summary["forward_snapshot_date"]),
        "source_hierarchy": "quote_aware_finer_buckets_over_redundant_parent",
        "accept_quote_conflict": True,
        "expected_quote_conflict_count": int(summary["quote_conflict_count"]),
        "expected_quote_conflicts": conflicts,
        "quote_conflict_identity_hash": str(summary["quote_conflict_identity_hash"]),
        "input_csv_sha256": str(summary["input_csv_sha256"]),
        "forwards_sha256": str(summary["forwards_sha256"]),
        "production_approved": True,
    }
    inventory = {
        "schema_version": INVENTORY_SCHEMA,
        "run_id": str(run_id),
        "promotion_eligible": False,
        "quote_conflict_count": int(summary["quote_conflict_count"]),
        "quote_conflict_identity_hash": str(summary["quote_conflict_identity_hash"]),
        "quote_conflict_identities": conflicts,
        "parents": {
            "base_assembly": _file_ref(staging, staging / ASSEMBLY_MANIFEST),
            "hourly_export_ch": _file_ref(staging, paths["hourly"]),
            "forward_quote_snapshot": _file_ref(staging, paths["forwards"]),
            "production_manifest": _file_ref(staging, paths["production"]),
        },
        "required_policy_payload": required_policy,
        "producer_contract": "pfc.pipeline.quote_conflict_inventory.v1",
    }
    return gates, inventory


def _run_product_audit(
    paths: Mapping[str, Path],
    *,
    policy_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    production = _load_json(paths["production"])
    snapshot = production.get("forward_snapshot")
    if not isinstance(snapshot, Mapping):
        raise CandidateProductEvidenceError("production forward snapshot is missing")
    try:
        return run_audit(
            csv_path=paths["hourly"],
            forwards_path=paths["forwards"],
            market="CH",
            required_forward_date=str(pd.Timestamp(snapshot["snapshot_date"]).date()),
            source_hierarchy_policy_path=policy_path,
            monthly_candidate_manifest_path=paths["production"],
            monthly_forward_source_path=paths["forward_source"],
            require_signed_source_hierarchy_policy=policy_path is not None,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise CandidateProductEvidenceError("delivered-product audit failed") from exc


def _portable_summary(staging: Path, summary: Mapping[str, object]) -> dict[str, object]:
    out = dict(summary)
    out["audit_script"] = "scripts/audit_ch_product_normalization.py"
    for key in ("input_csv", "forwards_path", "monthly_candidate_manifest"):
        value = out.get(key)
        if value:
            out[key] = _relative(staging, Path(str(value)), role=key)
    policy = out.get("source_hierarchy_policy")
    if isinstance(policy, Mapping):
        portable_policy = dict(policy)
        if portable_policy.get("path"):
            portable_policy["path"] = _relative(
                staging,
                Path(str(portable_policy["path"])),
                role="source_hierarchy_policy",
            )
        out["source_hierarchy_policy"] = portable_policy
    return out


def _expected_product_evidence_manifest(
    staging: Path,
    *,
    run_id: str,
    assembly: Mapping[str, object],
    paths: Mapping[str, Path],
    policy_path: Path,
    gates_path: Path,
    summary_path: Path,
) -> dict[str, object]:
    return {
        "schema_version": PRODUCT_EVIDENCE_SCHEMA,
        "run_id": str(run_id),
        "promotion_eligible": False,
        "parents": {
            "base_assembly": _file_ref(staging, staging / ASSEMBLY_MANIFEST),
            "product_prepolicy_gates": _file_ref(staging, staging / DRAFT_GATES),
            "quote_conflict_inventory": _file_ref(staging, staging / CONFLICT_INVENTORY),
            "hourly_export_ch": _file_ref(staging, paths["hourly"]),
            "forward_quote_snapshot": _file_ref(staging, paths["forwards"]),
            "production_manifest": _file_ref(staging, paths["production"]),
            "export_manifest": _file_ref(
                staging,
                staging / "manifests" / "hourly_export_manifest_ch.json",
            ),
            "selected_config_run_parity": _file_ref(
                staging,
                staging / "manifests" / "selected_config_run_parity.json",
            ),
            "selected_lambda_decision": _file_ref(
                staging,
                _entry_path(
                    staging,
                    assembly["sources"]["selected_lambda_decision"],
                ),
            ),
            "source_hierarchy_policy": _file_ref(staging, policy_path),
        },
        "artifacts": {
            "product_normalization_gates": _file_ref(staging, gates_path),
            "product_normalization_summary": _file_ref(staging, summary_path),
        },
        "producer_contract": "pfc.pipeline.product_normalization_evidence.v1",
    }


def _bound_paths(
    staging: Path,
    assembly: Mapping[str, object],
) -> dict[str, Path]:
    sources = assembly.get("sources")
    derived = assembly.get("derived")
    if not isinstance(sources, Mapping) or not isinstance(derived, Mapping):
        raise CandidateProductEvidenceError("base assembly inventory is invalid")
    return {
        "hourly": _entry_path(staging, derived["hourly_export_ch"]),
        "forward_source": _entry_path(staging, sources["forward_source"]),
        "forwards": _entry_path(staging, sources["forward_quote_snapshot"]),
        "production": _entry_path(staging, sources["production_manifest"]),
    }


def _entry_path(staging: Path, entry: object) -> Path:
    if not isinstance(entry, Mapping):
        raise CandidateProductEvidenceError("assembly artifact entry is invalid")
    return _regular_file(staging, str(entry.get("path", "")), role=str(entry.get("role", "")))


def _file_ref(staging: Path, path: Path) -> dict[str, object]:
    regular = _regular_file(staging, path, role="evidence_artifact")
    return {
        "path": regular.relative_to(staging).as_posix(),
        "sha256": _sha256_file(regular),
        "size_bytes": regular.stat().st_size,
    }


def _regular_file(staging: Path, value: str | Path, *, role: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = staging / path
    resolved = path.resolve()
    try:
        resolved.relative_to(staging)
    except ValueError as exc:
        raise CandidateProductEvidenceError(f"{role} escapes candidate staging") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise CandidateProductEvidenceError(f"{role} is not a regular staged file")
    return resolved


def _relative(staging: Path, path: Path, *, role: str) -> str:
    return _regular_file(staging, path, role=role).relative_to(staging).as_posix()


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = load_strict_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise CandidateProductEvidenceError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise CandidateProductEvidenceError(f"JSON artifact must be an object: {path}")
    return payload


def _load_json_bytes(payload: bytes, *, path: Path) -> dict[str, object]:
    try:
        value = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CandidateProductEvidenceError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise CandidateProductEvidenceError(f"JSON artifact must be an object: {path}")
    return value


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    raw = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
            default=str,
        )
        + "\n"
    ).encode("utf-8")
    _write_bytes_exclusive(path, raw)


def _materialize_exact_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    label: str,
) -> None:
    raw = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
            default=str,
        )
        + "\n"
    ).encode("utf-8")
    _materialize_exact_bytes(path, raw, label=label)


def _materialize_exact_bytes(path: Path, payload: bytes, *, label: str) -> None:
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise CandidateProductEvidenceError(f"{label} changed")
        return
    _write_bytes_exclusive(path, payload)


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(candidate_temporary_name(suffix="tmp"))
    try:
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise CandidateProductEvidenceError(
            f"temporary evidence write already exists: {temporary}"
        ) from exc
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise CandidateProductEvidenceError(f"refusing to overwrite artifact: {path}") from exc
    except OSError as exc:
        raise CandidateProductEvidenceError(f"atomic evidence publish failed: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
