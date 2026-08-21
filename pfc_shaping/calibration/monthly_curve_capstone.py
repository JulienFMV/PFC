"""Check monthly curve promotion using real prod/export manifests.

This is the promotion capstone for the monthly BASE solver.  Unlike
``check_monthly_curve_promotion.py``, it does not accept raw parity hashes on
the command line.  It reads them from the production/export monthly solver
manifests and reads the selected lambda hash from the calibration artifact,
then appends the required governance gates before evaluating promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Mapping

import pandas as pd

from pfc_shaping.calibration.cascading import count_hours
from pfc_shaping.calibration.monthly_curve_audit import build_monthly_curve_governance_gates
from pfc_shaping.calibration.monthly_curve_config_identity import (
    config_hash,
    solver_numerical_diagnostic_errors,
)
from pfc_shaping.calibration.monthly_curve_promotion import evaluate_monthly_curve_promotion
from pfc_shaping.calibration.monthly_forward_curve import product_periods
from pfc_shaping.data.acquisition_contract import (
    GOVERNED_LT_INPUT_SNAPSHOT_SCHEMAS,
    verify_acquisition_contract,
)
from pfc_shaping.data.forward_proxy import (
    PRODUCTION_EEX_MAX_AGE_BUSINESS_DAYS,
    verify_forward_manifest_binding,
)
from pfc_shaping.data.lt_input_sources import (
    consumed_lt_input_roles,
    dataframe_sha256,
    validate_governed_forward_history_frame,
    validate_governed_lt_snapshot_bundle,
)
from pfc_shaping.data.shared_data_root import (
    LT_DATA_VIEW_ID,
)
from pfc_shaping.data.snapshot_anchor_client import (
    SnapshotAnchorClientError,
    verify_external_operation_receipt,
)
from pfc_shaping.data.snapshot_publication_state import (
    SnapshotPublicationStateError,
    verify_external_publication_evidence,
    verify_publication_authority_separation,
)
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.atomic_promotion import (
    PromotionError,
    verify_candidate_bundle_manifest,
)
from pfc_shaping.pipeline.candidate_evidence import (
    CandidateEvidenceError,
    verify_assembled_candidate_evidence,
    verify_pre_run_governance_evidence,
)
from pfc_shaping.pipeline.governed_release_cli_contract import (
    ReleaseCliIdentityError,
    absolute_path,
    assert_installed_runtime_sealed,
    assert_phase_private_key_scope,
)
from pfc_shaping.pipeline.monthly_curve_authority import monthly_curve_config_from_settings
from pfc_shaping.pipeline.promotion_contract import (
    PROMOTION_POLICY_VERSION,
    REQUIRED_PROMOTION_GATE_IDS,
    sign_promotion_receipt,
)
from pfc_shaping.pipeline.quality_gate import (
    QualityGateError,
    input_grid_errors,
    validate_input_frame,
    weekly_local_grid_errors,
)
from pfc_shaping.pipeline.quote_conflict_policy_contract import (
    QuoteConflictPolicyAuthenticationError,
    verify_quote_conflict_policy,
)
from pfc_shaping.pipeline.release_request_contract import (
    ReleaseRequestAuthenticationError,
    load_registered_release_request_evidence,
)
from pfc_shaping.pipeline.strict_structured_data import (
    StrictStructuredDataError,
    load_strict_json,
    load_strict_yaml,
)
from pfc_shaping.validation.product_normalization import (
    DEFAULT_PRICE_COLUMN,
    load_source_hierarchy_policy,
    run_audit,
)

PRODUCT_NORMALIZATION_EVIDENCE_KEY = "delivered_product_normalization_evidence"
REQUIRED_PRODUCT_NORMALIZATION_EVIDENCE_FIELDS = (
    "summary_sha256",
    "input_csv_sha256",
    "forwards_sha256",
    "forward_snapshot_date",
    "market",
)
REQUIRED_PRODUCT_GATE_FAMILIES = (
    "hard_base_product_repricing",
    "hard_peak_product_repricing",
    "implied_offpeak_identity",
)

REQUIRED_GOVERNANCE_GATES = set(REQUIRED_PROMOTION_GATE_IDS)

REQUIRED_CANDIDATE_FRESHNESS_COLUMNS = {
    "epex_ch": ("price_eur_mwh",),
    "epex_de": ("price_eur_mwh",),
    "entso": ("load_mw", "solar_mw", "wind_mw"),
    "hydro": ("fill_deviation", "water_value_supported"),
}
OPTIONAL_CANDIDATE_FRESHNESS_COLUMNS = {
    "outages": ("unavailable_mw", "n_outages"),
    "epex_at": ("price_eur_mwh",),
    "epex_fr": ("price_eur_mwh",),
    "epex_it": ("price_eur_mwh",),
}
CANDIDATE_FRESHNESS_COLUMNS = {
    **REQUIRED_CANDIDATE_FRESHNESS_COLUMNS,
    **OPTIONAL_CANDIDATE_FRESHNESS_COLUMNS,
}
SANCTIONED_FRESHNESS_MAX_AGE_DAYS = {
    "epex_ch": 2.0,
    "epex_de": 2.0,
    "entso": 3.0,
    "hydro": 10.0,
    "outages": 2.0,
    "epex_at": 2.0,
    "epex_fr": 2.0,
    "epex_it": 2.0,
}
SANCTIONED_FRESHNESS_MAX_GAP_DAYS = {
    "epex_ch": 1.0 / 24.0,
    "epex_de": 1.0 / 24.0,
    "entso": 2.0 / 24.0,
    "hydro": 14.0,
    "outages": 2.0 / 24.0,
    "epex_at": 1.0 / 24.0,
    "epex_fr": 1.0 / 24.0,
    "epex_it": 1.0 / 24.0,
}
SANCTIONED_MIN_FINITE_FRACTION = 0.95
SANCTIONED_MIN_RECENT_FINITE_FRACTION = 0.95
SANCTIONED_EXPECTED_CADENCE_SECONDS = {
    "epex_ch": 15 * 60,
    "epex_de": 15 * 60,
    "entso": 15 * 60,
    "hydro": 7 * 24 * 60 * 60,
    "outages": 15 * 60,
    "epex_at": 15 * 60,
    "epex_fr": 15 * 60,
    "epex_it": 15 * 60,
}
SANCTIONED_EXPECTED_PHASE_OFFSET_SECONDS = {
    "hydro": 4 * 24 * 60 * 60,
}
SANCTIONED_MIN_GRID_COVERAGE = 0.95


@dataclass(frozen=True)
class CapturedArtifact:
    path: Path
    raw: bytes | None
    sha256: str


def _capture_artifact(path: Path) -> CapturedArtifact:
    resolved = Path(path).resolve()
    raw = read_stable_single_link_file(resolved, label="capstone input artifact")
    return CapturedArtifact(resolved, raw, hashlib.sha256(raw).hexdigest())


def _captured_raw(artifact: CapturedArtifact) -> bytes:
    if artifact.raw is None:
        raise ValueError(f"captured artifact bytes are unavailable: {artifact.path}")
    return artifact.raw


def _promotion_now_utc() -> pd.Timestamp:
    """Return the trusted promotion clock; tests may monkeypatch this function."""

    return pd.Timestamp.now("UTC")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    promotion_evaluated_at = _aware_utc_timestamp(
        _promotion_now_utc(),
        label="promotion clock",
    )
    captured: dict[str, CapturedArtifact] = {
        "audit_gates": _capture_artifact(args.audit_gates),
        "historical_thresholds": _capture_artifact(args.historical_thresholds),
        "production_manifest": _capture_artifact(args.production_manifest),
        "export_manifest": _capture_artifact(args.export_manifest),
        "selected_config_artifact": _capture_artifact(args.selected_config_artifact),
    }
    candidate_bundle_path = (
        args.candidate_bundle_manifest
        if args.candidate_bundle_manifest is not None
        else args.production_manifest.with_name("candidate_bundle_manifest.json")
    )
    try:
        verified_candidate_bundle = verify_candidate_bundle_manifest(candidate_bundle_path)
    except PromotionError as exc:
        raise ValueError(f"candidate bundle verification failed: {exc}") from exc
    captured["candidate_bundle_manifest"] = _capture_artifact(candidate_bundle_path)
    if captured["candidate_bundle_manifest"].sha256 != verified_candidate_bundle.manifest_sha256:
        raise ValueError("candidate bundle manifest changed during initial verification")
    candidate_bundle_manifest = _load_mapping_bytes(
        _captured_raw(captured["candidate_bundle_manifest"]),
        captured["candidate_bundle_manifest"].path,
    )
    if candidate_bundle_manifest.get("schema_version") != "candidate_bundle.v1":
        raise ValueError("candidate bundle manifest schema is not candidate_bundle.v1")
    metadata = candidate_bundle_manifest.get("metadata")
    assembled_mode = (
        isinstance(metadata, Mapping)
        and metadata.get("evidence_contract") == "assembled_lt_candidate.v1"
    )
    output_path = args.output or (
        verified_candidate_bundle.path.parent
        / (
            f"{verified_candidate_bundle.run_id}."
            f"{verified_candidate_bundle.manifest_sha256}.promotion_receipt.json"
        )
    )
    for role, path in (
        ("promotion_receipt", output_path),
        ("augmented_audit_gates", args.augmented_audit_gates),
        ("details_output", args.details_output),
    ):
        if path is not None:
            _require_output_outside_bundle(path, verified_candidate_bundle.path, role=role)
    candidate_run_path = (
        verified_candidate_bundle.path / "manifests" / "candidate_run_manifest.json"
    )
    try:
        captured["candidate_run_manifest"] = _capture_artifact(candidate_run_path)
    except OSError as exc:
        raise ValueError("candidate bundle is missing candidate_run_manifest.json") from exc
    _require_artifact_in_bundle(
        captured["candidate_run_manifest"],
        verified_candidate_bundle.path,
        candidate_bundle_manifest,
        role="candidate_run_manifest",
    )
    candidate_run_manifest = _load_mapping_bytes(
        _captured_raw(captured["candidate_run_manifest"]),
        captured["candidate_run_manifest"].path,
    )
    governed_run_spec = _capture_candidate_pre_run_evidence(
        candidate_run_manifest,
        bundle_path=verified_candidate_bundle.path,
        bundle_manifest=candidate_bundle_manifest,
        captured=captured,
    )
    candidate_config = _capture_candidate_config_evidence(
        candidate_run_manifest,
        bundle_path=verified_candidate_bundle.path,
        bundle_manifest=candidate_bundle_manifest,
        captured=captured,
    )
    candidate_data_evidence = _capture_candidate_data_evidence(
        candidate_run_manifest,
        bundle_path=verified_candidate_bundle.path,
        bundle_manifest=candidate_bundle_manifest,
        captured=captured,
    )
    valuation_timestamp, candidate_freshness_gate = _verify_candidate_run_evidence(
        candidate_run_manifest,
        run_id=verified_candidate_bundle.run_id,
        promotion_evaluated_at=promotion_evaluated_at,
        data_contract=candidate_data_evidence["contract"],
        data_pointer=candidate_data_evidence["pointer"],
        publication_head_observation=candidate_data_evidence.get("publication_head_observation"),
        governed_run_spec=governed_run_spec,
        active_config=candidate_config,
        data_root=args.data_root,
        candidate_finalized_at=candidate_bundle_manifest.get(
            "candidate_finalized_at_utc",
            candidate_bundle_manifest.get("created_at_utc"),
        ),
    )
    if args.product_normalization_summary is not None:
        captured["product_normalization_summary"] = _capture_artifact(
            args.product_normalization_summary
        )
    if args.manifest is not None:
        captured["audit_manifest"] = _capture_artifact(args.manifest)
    if assembled_mode:
        _capture_canonical_assembled_artifacts(
            captured,
            bundle_path=verified_candidate_bundle.path,
            run_id=verified_candidate_bundle.run_id,
        )
    _validate_promotion_authorization_context(
        args,
        assembled_mode=assembled_mode,
        run_id=verified_candidate_bundle.run_id,
        candidate_bundle_manifest_sha256=verified_candidate_bundle.manifest_sha256,
        captured=captured,
    )
    _require_artifact_in_bundle(
        captured["production_manifest"],
        verified_candidate_bundle.path,
        candidate_bundle_manifest,
        role="production_manifest",
    )
    audit_gates = pd.read_csv(BytesIO(_captured_raw(captured["audit_gates"])))
    historical_thresholds = pd.read_csv(BytesIO(_captured_raw(captured["historical_thresholds"])))
    production_manifest = _load_mapping_bytes(
        _captured_raw(captured["production_manifest"]),
        captured["production_manifest"].path,
    )
    contract_files = candidate_data_evidence["contract"].get("files")
    available_roles = set(contract_files) if isinstance(contract_files, Mapping) else set()
    solver_consumes_history = "eex_forwards_history" in consumed_lt_input_roles(
        candidate_config,
        available_roles=available_roles,
    )
    _verify_governed_monthly_solver_history_binding(
        production_manifest=production_manifest,
        candidate_run_manifest=candidate_run_manifest,
        data_contract=candidate_data_evidence["contract"],
        active_config=candidate_config,
        governed_history=(
            _load_governed_forward_history_frame(
                data_root=args.data_root,
                data_contract=candidate_data_evidence["contract"],
                data_pointer=candidate_data_evidence["pointer"],
                publication_head_observation=candidate_data_evidence.get(
                    "publication_head_observation"
                ),
                valuation_timestamp=valuation_timestamp,
            )
            if solver_consumes_history
            else None
        ),
        valuation_timestamp=valuation_timestamp,
    )
    _resolve_forward_source_path(
        production_manifest,
        captured["production_manifest"].path,
    )
    _verify_forward_source_in_bundle(
        production_manifest,
        verified_candidate_bundle.path,
        candidate_bundle_manifest,
    )
    forward_workbook_path = Path(
        str(production_manifest["forward_snapshot"]["source_path"])
    ).resolve()
    captured["forward_workbook"] = _capture_artifact(forward_workbook_path)
    _require_artifact_in_bundle(
        captured["forward_workbook"],
        verified_candidate_bundle.path,
        candidate_bundle_manifest,
        role="forward_workbook",
    )
    export_manifest = _load_mapping_bytes(
        _captured_raw(captured["export_manifest"]),
        captured["export_manifest"].path,
    )
    if assembled_mode:
        export_manifest = _assembled_export_manifest_view(
            export_manifest,
            production_manifest=production_manifest,
            export_capture=captured["export_manifest"],
            production_capture=captured["production_manifest"],
            bundle_path=verified_candidate_bundle.path,
        )
    _resolve_forward_source_path(
        export_manifest,
        (
            captured["production_manifest"].path
            if assembled_mode
            else captured["export_manifest"].path
        ),
    )
    _verify_forward_source_in_bundle(
        export_manifest,
        verified_candidate_bundle.path,
        candidate_bundle_manifest,
    )
    selected_config = _load_mapping_bytes(
        _captured_raw(captured["selected_config_artifact"]),
        captured["selected_config_artifact"].path,
    )
    product_capture = captured.get("product_normalization_summary")
    product_normalization_summary = (
        _load_mapping_bytes(_captured_raw(product_capture), product_capture.path)
        if product_capture is not None
        else None
    )
    if assembled_mode:
        assembled_expected = _assembled_product_normalization_contract(
            captured=captured,
            product_summary=product_normalization_summary,
            bundle_path=verified_candidate_bundle.path,
            run_id=verified_candidate_bundle.run_id,
        )
        selected_config = {
            **selected_config,
            PRODUCT_NORMALIZATION_EVIDENCE_KEY: assembled_expected,
        }
        production_manifest = {
            **production_manifest,
            PRODUCT_NORMALIZATION_EVIDENCE_KEY: assembled_expected,
        }
        export_manifest = {
            **export_manifest,
            PRODUCT_NORMALIZATION_EVIDENCE_KEY: assembled_expected,
        }
    monthly_candidate_capture: CapturedArtifact | None = None
    monthly_candidate_manifest: Mapping[str, object] | None = None
    product_normalization_replay: Mapping[str, object] | None = None
    product_normalization_replay_errors: list[str] = []
    if isinstance(product_normalization_summary, Mapping):
        candidate_value = product_normalization_summary.get("monthly_candidate_manifest")
        if candidate_value:
            candidate_path = Path(str(candidate_value))
            if not candidate_path.is_absolute():
                candidate_path = (
                    verified_candidate_bundle.path / candidate_path
                    if assembled_mode
                    else product_capture.path.parent / candidate_path
                    if product_capture is not None
                    else candidate_path
                )
            monthly_candidate_capture = _capture_artifact(candidate_path)
            captured["monthly_candidate_manifest"] = monthly_candidate_capture
            _require_artifact_in_bundle(
                monthly_candidate_capture,
                verified_candidate_bundle.path,
                candidate_bundle_manifest,
                role="monthly_candidate_manifest",
            )
            monthly_candidate_manifest = _load_mapping_bytes(
                _captured_raw(monthly_candidate_capture),
                monthly_candidate_capture.path,
            )
            _resolve_forward_source_path(
                monthly_candidate_manifest,
                monthly_candidate_capture.path,
            )
            _verify_forward_source_in_bundle(
                monthly_candidate_manifest,
                verified_candidate_bundle.path,
                candidate_bundle_manifest,
            )
        for role, path_key, hash_key in (
            ("delivered_product_csv", "input_csv", "input_csv_sha256"),
            ("delivered_forwards", "forwards_path", "forwards_sha256"),
        ):
            value = product_normalization_summary.get(path_key)
            if not value:
                raise ValueError(f"product normalization summary missing {path_key}")
            artifact_path = Path(str(value))
            if not artifact_path.is_absolute():
                artifact_path = (
                    verified_candidate_bundle.path / artifact_path
                    if assembled_mode
                    else product_capture.path.parent / artifact_path
                    if product_capture is not None
                    else artifact_path
                )
            artifact = _capture_artifact(artifact_path)
            captured[role] = artifact
            if artifact.sha256 != str(product_normalization_summary.get(hash_key, "")):
                raise ValueError(f"product normalization summary {hash_key} mismatch")
            try:
                relative = artifact.path.relative_to(verified_candidate_bundle.path).as_posix()
            except ValueError as exc:
                raise ValueError(
                    f"audited artifact is outside candidate bundle: {artifact.path}"
                ) from exc
            bundle_files = candidate_bundle_manifest.get("files")
            if (
                not isinstance(bundle_files, Mapping)
                or bundle_files.get(relative) != artifact.sha256
            ):
                raise ValueError(
                    f"audited artifact is not hash-bound in candidate bundle: {relative}"
                )
        source_policy_path: Path | None = None
        source_policy = product_normalization_summary.get("source_hierarchy_policy")
        if isinstance(source_policy, Mapping) and source_policy.get("path"):
            source_policy_path = Path(str(source_policy["path"]))
            if not source_policy_path.is_absolute():
                source_policy_path = (
                    verified_candidate_bundle.path / source_policy_path
                    if assembled_mode
                    else product_capture.path.parent / source_policy_path
                    if product_capture is not None
                    else source_policy_path
                )
            source_policy_capture = _capture_artifact(source_policy_path)
            captured["source_hierarchy_policy"] = source_policy_capture
            _require_artifact_in_bundle(
                source_policy_capture,
                verified_candidate_bundle.path,
                candidate_bundle_manifest,
                role="source_hierarchy_policy",
            )
            if source_policy_capture.sha256 != str(source_policy.get("sha256", "")):
                raise ValueError("source hierarchy policy sha256 mismatch")
            raw_policy = load_source_hierarchy_policy(
                source_policy_capture.path,
                payload=_captured_raw(source_policy_capture),
            )
            if isinstance(raw_policy, Mapping) and raw_policy.get("production_approved") is True:
                try:
                    verify_quote_conflict_policy(raw_policy)
                except QuoteConflictPolicyAuthenticationError as exc:
                    raise ValueError(
                        f"source hierarchy policy authorization failed: {exc}"
                    ) from exc
        product_normalization_replay, product_normalization_replay_errors = (
            _replay_product_normalization_audit(
                declared_summary=product_normalization_summary,
                delivered_csv=captured["delivered_product_csv"],
                delivered_forwards=captured["delivered_forwards"],
                monthly_candidate_manifest=monthly_candidate_capture,
                source_hierarchy_policy=captured.get("source_hierarchy_policy"),
                forward_workbook=captured["forward_workbook"],
                bundle_path=verified_candidate_bundle.path,
                expected_evidence=_product_normalization_evidence(
                    production_manifest=production_manifest,
                    export_manifest=export_manifest,
                    selected_config=selected_config,
                    promotion_timestamp=valuation_timestamp,
                ),
            )
        )
    if (
        assembled_mode
        and isinstance(product_normalization_summary, Mapping)
        and monthly_candidate_capture is not None
    ):
        product_normalization_summary = {
            **product_normalization_summary,
            "monthly_candidate_manifest": str(monthly_candidate_capture.path),
        }
    selected_hash = _selected_config_hash(selected_config, args.selected_config_artifact)
    active_hash = _active_config_hash(production_manifest)

    governance = build_manifest_governance_gates(
        run_timestamp=promotion_evaluated_at,
        production_manifest=production_manifest,
        export_manifest=export_manifest,
        selected_config=selected_config,
        active_config_hash=active_hash,
        selected_config_hash=selected_hash,
        product_normalization_summary=product_normalization_summary,
        product_normalization_summary_path=(product_capture.path if product_capture else None),
        product_normalization_summary_sha256=(product_capture.sha256 if product_capture else None),
        product_normalization_replay=product_normalization_replay,
        product_normalization_replay_errors=product_normalization_replay_errors,
        monthly_candidate_manifest=monthly_candidate_manifest,
        monthly_candidate_manifest_path=(
            monthly_candidate_capture.path if monthly_candidate_capture else None
        ),
        monthly_candidate_manifest_sha256=(
            monthly_candidate_capture.sha256 if monthly_candidate_capture else None
        ),
        product_normalization_evidence=_product_normalization_evidence(
            production_manifest=production_manifest,
            export_manifest=export_manifest,
            selected_config=selected_config,
            promotion_timestamp=valuation_timestamp,
        ),
        production_manifest_path=captured["production_manifest"].path,
        production_manifest_sha256=captured["production_manifest"].sha256,
        selected_config_path=captured["selected_config_artifact"].path,
    )
    governance = pd.concat(
        [governance, pd.DataFrame([candidate_freshness_gate])],
        ignore_index=True,
    )
    augmented = _replace_gate_rows(audit_gates, governance)
    manifest = _promotion_manifest(
        audit_manifest=(
            _load_mapping_bytes(
                _captured_raw(captured["audit_manifest"]), captured["audit_manifest"].path
            )
            if "audit_manifest" in captured
            else {}
        ),
        production_manifest=production_manifest,
        export_manifest=export_manifest,
        selected_config=selected_config,
        product_normalization_summary=product_normalization_summary,
        product_normalization_summary_path=(product_capture.path if product_capture else None),
        product_normalization_summary_sha256=(product_capture.sha256 if product_capture else None),
        governance_gates=governance,
    )

    decision = evaluate_monthly_curve_promotion(
        augmented,
        historical_thresholds,
        run_timestamp=promotion_evaluated_at,
        far_horizon_min_years=int(args.far_horizon_min_years),
        required_governance_gates=REQUIRED_GOVERNANCE_GATES,
        manifest=manifest,
    )

    if args.augmented_audit_gates is not None:
        _require_output_outside_bundle(
            args.augmented_audit_gates,
            verified_candidate_bundle.path,
            role="augmented_audit_gates",
        )
        args.augmented_audit_gates.parent.mkdir(parents=True, exist_ok=True)
        augmented.to_csv(args.augmented_audit_gates, index=False)
    _assert_captured_artifacts_unchanged(captured)
    try:
        verified_candidate_bundle_end = verify_candidate_bundle_manifest(candidate_bundle_path)
    except PromotionError as exc:
        raise ValueError(f"candidate bundle changed during evaluation: {exc}") from exc
    if verified_candidate_bundle_end.manifest_sha256 != verified_candidate_bundle.manifest_sha256:
        raise ValueError("candidate bundle manifest changed during evaluation")
    receipt = _promotion_receipt(
        run_id=verified_candidate_bundle.run_id,
        approved=decision.approved,
        decision_summary=decision.summary,
        valuation_timestamp=valuation_timestamp,
        promotion_evaluated_at=promotion_evaluated_at,
        input_artifacts=captured,
        governance_gates=governance,
        augmented_audit_gates=augmented,
        promotion_manifest=manifest,
        candidate_bundle_manifest_sha256=captured["candidate_bundle_manifest"].sha256,
        candidate_run_manifest_sha256=captured["candidate_run_manifest"].sha256,
        release_request_id=args.release_request_id,
        release_operation_id=args.release_operation_id,
        expected_current_event_id=(
            None if args.expected_current_event_id == "NONE" else args.expected_current_event_id
        ),
        operation_created_at_utc=args.operation_created_at_utc,
        release_root_sha256=args.release_root_sha256,
        evidence_inventory_sha256=args.evidence_inventory_sha256,
        workflow_domain_sha256=getattr(args, "workflow_domain_sha256", None),
        release_request_document_sha256=getattr(args, "release_request_document_sha256", None),
        signing_private_key_path=args.signing_private_key,
    )
    _atomic_write_json(output_path, receipt)
    if args.details_output is not None:
        _require_output_outside_bundle(
            args.details_output,
            verified_candidate_bundle.path,
            role="details_output",
        )
        args.details_output.parent.mkdir(parents=True, exist_ok=True)
        decision.details.to_csv(args.details_output, index=False)

    print(json.dumps(decision.summary, sort_keys=True, default=str))
    return 0 if decision.approved else 1


def _capture_candidate_data_evidence(
    candidate_run_manifest: Mapping[str, object],
    *,
    bundle_path: Path,
    bundle_manifest: Mapping[str, object],
    captured: dict[str, CapturedArtifact],
) -> dict[str, Mapping[str, object]]:
    evidence: dict[str, Mapping[str, object]] = {}
    for name in ("contract", "pointer"):
        entry = candidate_run_manifest.get(f"data_{name}")
        if not isinstance(entry, Mapping):
            raise ValueError(f"candidate run manifest data_{name} evidence is missing")
        archive_path = str(entry.get("archive_path", "")).strip()
        declared_hash = str(entry.get("sha256", "")).strip()
        if not archive_path or not _is_sha256(declared_hash):
            raise ValueError(f"candidate run manifest data_{name} evidence is incomplete")
        path = (bundle_path / archive_path).resolve()
        try:
            path.relative_to(bundle_path.resolve())
        except ValueError as exc:
            raise ValueError(f"candidate data {name} escapes candidate bundle") from exc
        artifact = _capture_artifact(path)
        if artifact.sha256 != declared_hash:
            raise ValueError(f"candidate data {name} hash does not match run manifest")
        captured[f"candidate_data_{name}"] = artifact
        _require_artifact_in_bundle(
            artifact,
            bundle_path,
            bundle_manifest,
            role=f"candidate_data_{name}",
        )
        evidence[name] = _load_mapping_bytes(_captured_raw(artifact), artifact.path)
    if evidence["pointer"].get("schema_version") == "lt_data_pointer.v2":
        publication_names = (
            "publication_intent",
            "publication_anchor_receipt",
            "publication_head_observation",
        )
        publication_payloads: dict[str, bytes] = {}
        for name in publication_names:
            entry = candidate_run_manifest.get(f"data_{name}")
            if not isinstance(entry, Mapping):
                raise ValueError(f"candidate run manifest data_{name} evidence is missing")
            archive_path = str(entry.get("archive_path", "")).strip()
            declared_hash = str(entry.get("sha256", "")).strip()
            if not archive_path or not _is_sha256(declared_hash):
                raise ValueError(f"candidate run manifest data_{name} evidence is incomplete")
            path = (bundle_path / archive_path).resolve()
            try:
                path.relative_to(bundle_path.resolve())
            except ValueError as exc:
                raise ValueError(f"candidate data {name} escapes candidate bundle") from exc
            artifact = _capture_artifact(path)
            if artifact.sha256 != declared_hash:
                raise ValueError(f"candidate data {name} hash does not match run manifest")
            captured[f"candidate_data_{name}"] = artifact
            _require_artifact_in_bundle(
                artifact,
                bundle_path,
                bundle_manifest,
                role=f"candidate_data_{name}",
            )
            publication_payloads[name] = _captured_raw(artifact)
            evidence[name] = _load_mapping_bytes(
                publication_payloads[name],
                artifact.path,
            )
        if evidence["publication_intent"].get("transition_type") != "PUBLISH":
            raise ValueError(
                "promotion-grade capstone cannot consume a BOOTSTRAP publication"
            )
        try:
            verify_external_publication_evidence(
                intent_payload=publication_payloads["publication_intent"],
                receipt_payload=publication_payloads["publication_anchor_receipt"],
                observation_payload=publication_payloads["publication_head_observation"],
                pointer=evidence["pointer"],
                require_current=False,
            )
            verify_publication_authority_separation(
                evidence["contract"],
                intent=evidence["publication_intent"],
                receipt=evidence["publication_anchor_receipt"],
                observation=evidence["publication_head_observation"],
            )
            verify_external_operation_receipt(publication_payloads["publication_anchor_receipt"])
        except (SnapshotAnchorClientError, SnapshotPublicationStateError) as exc:
            raise ValueError(
                "candidate external publication evidence is invalid or not authoritative"
            ) from exc
    return evidence


def _capture_candidate_pre_run_evidence(
    candidate_run_manifest: Mapping[str, object],
    *,
    bundle_path: Path,
    bundle_manifest: Mapping[str, object],
    captured: dict[str, CapturedArtifact],
) -> Mapping[str, object]:
    archive_path = str(candidate_run_manifest.get("pre_run_governance_manifest", "")).strip()
    declared_hash = str(
        candidate_run_manifest.get("pre_run_governance_manifest_sha256", "")
    ).strip()
    if archive_path != "manifests/pre_run_governance_manifest.json" or not _is_sha256(
        declared_hash
    ):
        raise ValueError("candidate pre-run governance evidence is incomplete")
    artifact = _capture_artifact(bundle_path / archive_path)
    if artifact.sha256 != declared_hash:
        raise ValueError("candidate pre-run governance hash does not match run manifest")
    captured["candidate_pre_run_governance"] = artifact
    _require_artifact_in_bundle(
        artifact,
        bundle_path,
        bundle_manifest,
        role="candidate_pre_run_governance",
    )
    try:
        pre_run = verify_pre_run_governance_evidence(
            bundle_path,
            expected_run_id=str(candidate_run_manifest.get("run_id", "")),
            expected_valuation_timestamp=str(candidate_run_manifest.get("reference_timestamp", "")),
        )
    except CandidateEvidenceError as exc:
        raise ValueError("candidate pre-run governance evidence is invalid") from exc
    run_spec = pre_run.get("governed_run_spec")
    if not isinstance(run_spec, Mapping):
        raise ValueError("candidate governed run spec is missing")
    return run_spec


def _capture_candidate_config_evidence(
    candidate_run_manifest: Mapping[str, object],
    *,
    bundle_path: Path,
    bundle_manifest: Mapping[str, object],
    captured: dict[str, CapturedArtifact],
) -> Mapping[str, object]:
    archive_path = str(candidate_run_manifest.get("config_path", "")).strip()
    declared_hash = str(candidate_run_manifest.get("config_sha256", "")).strip()
    if not archive_path or not _is_sha256(declared_hash):
        raise ValueError("candidate run manifest config evidence is incomplete")
    path = (bundle_path / archive_path).resolve()
    try:
        path.relative_to(bundle_path.resolve())
    except ValueError as exc:
        raise ValueError("candidate config escapes candidate bundle") from exc
    artifact = _capture_artifact(path)
    if artifact.sha256 != declared_hash:
        raise ValueError("candidate config hash does not match run manifest")
    captured["candidate_config"] = artifact
    _require_artifact_in_bundle(
        artifact,
        bundle_path,
        bundle_manifest,
        role="candidate_config",
    )
    config = load_strict_yaml(_captured_raw(artifact).decode("utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("candidate config must contain a mapping")
    return config


def _verify_governed_monthly_solver_history_binding(
    *,
    production_manifest: Mapping[str, object],
    candidate_run_manifest: Mapping[str, object],
    data_contract: Mapping[str, object],
    active_config: Mapping[str, object],
    governed_history: pd.DataFrame | None = None,
    valuation_timestamp: pd.Timestamp | None = None,
) -> None:
    contract_files = data_contract.get("files")
    if not isinstance(contract_files, Mapping):
        raise ValueError("monthly solver governed data contract files are missing")
    expected_roles = consumed_lt_input_roles(
        active_config,
        available_roles=set(contract_files),
    )
    solver_expected = "eex_forwards_history" in expected_roles
    manifest_authority = str(production_manifest.get("monthly_level_authority", "")).strip().lower()
    if manifest_authority not in {"solver", "legacy"}:
        raise ValueError("production manifest monthly_level_authority must be solver or legacy")
    manifest_solver = manifest_authority == "solver"
    if solver_expected != manifest_solver:
        raise ValueError(
            "candidate config monthly solver activation does not match "
            "production manifest monthly_level_authority"
        )
    if not solver_expected:
        return
    source_hashes = production_manifest.get("source_hashes")
    input_sources = candidate_run_manifest.get("input_sources")
    if not isinstance(source_hashes, Mapping):
        raise ValueError("monthly solver manifest source_hashes are missing")
    if not isinstance(input_sources, Mapping) or not isinstance(contract_files, Mapping):
        raise ValueError("monthly solver governed input evidence is missing")
    solver_hash = str(source_hashes.get("eex_forwards_history", ""))
    receipt = input_sources.get("eex_forwards_history")
    contract = contract_files.get("eex_forwards_history")
    if not _is_sha256(solver_hash):
        raise ValueError("monthly solver EEX forward history source hash is missing")
    if not isinstance(receipt, Mapping) or not isinstance(contract, Mapping):
        raise ValueError("monthly solver EEX forward history receipt is missing")
    if contract.get("calibration_eligible") is not True:
        raise ValueError("monthly solver EEX forward history is not calibration eligible")
    receipt_hash = str(receipt.get("sha256", ""))
    contract_hash = str(contract.get("sha256", ""))
    if solver_hash != receipt_hash or solver_hash != contract_hash:
        raise ValueError("monthly solver EEX forward history hash binding mismatch")
    catalog_binding = contract.get("vintage_catalog")
    if not isinstance(catalog_binding, Mapping):
        raise ValueError("monthly solver signed EEX vintage catalog binding is missing")
    catalog_hash = str(source_hashes.get("eex_historical_vintage_catalog", ""))
    if not _is_sha256(catalog_hash) or catalog_hash != str(catalog_binding.get("sha256", "")):
        raise ValueError("monthly solver EEX vintage catalog hash binding mismatch")
    if governed_history is None or valuation_timestamp is None:
        raise ValueError("monthly solver EEX forward history semantic evidence is missing")
    verified_vintage = governed_history.attrs.get("verified_vintage_catalog")
    declared_vintage = production_manifest.get("verified_vintage_catalog")
    if not isinstance(verified_vintage, Mapping) or dict(verified_vintage) != declared_vintage:
        raise ValueError("monthly solver EEX vintage catalog evidence mismatch")
    if (
        verified_vintage.get("status") != "VERIFIED_SIGNED_IMMUTABLE_VINTAGES"
        or str(verified_vintage.get("catalog_sha256", "")) != catalog_hash
        or str(verified_vintage.get("history_sha256", "")) != solver_hash
    ):
        raise ValueError("monthly solver EEX vintage catalog evidence is invalid")
    consumed_frame_hash = str(source_hashes.get("eex_forwards_history_consumed_frame", ""))
    if not _is_sha256(consumed_frame_hash) or consumed_frame_hash != dataframe_sha256(
        governed_history
    ):
        raise ValueError("monthly solver consumed EEX history frame hash mismatch")
    pit_selection = verified_vintage.get("pit_selection")
    if not isinstance(pit_selection, Mapping):
        raise ValueError("monthly solver EEX PIT selection evidence is missing")
    pit_valuation = _aware_utc_timestamp(
        pit_selection.get("valuation_timestamp"),
        label="monthly solver EEX PIT valuation timestamp",
    )
    if pit_valuation != valuation_timestamp:
        raise ValueError("monthly solver EEX PIT valuation timestamp mismatch")
    pit_available = _aware_utc_timestamp(
        pit_selection.get("max_available_at"),
        label="monthly solver EEX PIT maximum availability",
    )
    if pit_available > valuation_timestamp:
        raise ValueError("monthly solver EEX PIT selection uses future information")
    observation_dates = pd.to_datetime(governed_history.get("date"), utc=True, errors="coerce")
    summary_reference = valuation_timestamp
    if not observation_dates.isna().all():
        summary_reference = max(summary_reference, observation_dates.max())
    validated = validate_governed_forward_history_frame(
        governed_history,
        reference_timestamp=summary_reference,
    )
    source_summary = list(validated.attrs.get("eex_source_summary", []))
    declared_summary = production_manifest.get("eex_forwards_history_source_summary")
    if declared_summary != source_summary:
        raise ValueError("monthly solver EEX forward history source summary mismatch")
    declared_summary_hash = str(source_hashes.get("eex_forwards_history_source_summary", ""))
    if declared_summary_hash != _sha256_json(source_summary):
        raise ValueError("monthly solver EEX forward history source summary hash mismatch")


def _load_governed_forward_history_frame(
    *,
    data_root: Path,
    data_contract: Mapping[str, object],
    data_pointer: Mapping[str, object],
    publication_head_observation: Mapping[str, object] | None,
    valuation_timestamp: pd.Timestamp,
) -> pd.DataFrame | None:
    files = data_contract.get("files")
    if not isinstance(files, Mapping) or "eex_forwards_history" not in files:
        return None
    entry = files["eex_forwards_history"]
    if not isinstance(entry, Mapping):
        raise ValueError("governed EEX forward history contract entry is invalid")
    root = Path(str(data_root).strip()).expanduser()
    if not root.is_absolute():
        raise ValueError("data_root must be absolute for governed EEX forward history")
    root = root.resolve()
    contract_rel = str(data_pointer.get("contract_path", "")).strip()
    expected_contract_rel = (
        Path("snapshots") / str(data_contract.get("generation_id", "")) / "lt_input_snapshot.json"
    ).as_posix()
    if Path(contract_rel).as_posix() != expected_contract_rel:
        raise ValueError("governed EEX data contract path is not canonical")
    try:
        snapshot_root = (root / contract_rel).resolve().parent
        snapshot_root.relative_to(root)
        source = (snapshot_root / str(entry.get("path", ""))).resolve()
        source.relative_to(snapshot_root)
    except (OSError, ValueError) as exc:
        raise ValueError("governed EEX forward history path escapes data root") from exc
    catalog_binding = entry.get("vintage_catalog")
    if not isinstance(catalog_binding, Mapping):
        raise ValueError("governed EEX forward history vintage catalog is missing")
    try:
        catalog_path = (snapshot_root / str(catalog_binding.get("path", ""))).resolve()
        catalog_path.relative_to(snapshot_root)
        catalog_payload = read_stable_single_link_file(
            catalog_path,
            label="governed EEX vintage catalog",
        )
        catalog = load_strict_json(catalog_payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, StrictStructuredDataError, ValueError) as exc:
        raise ValueError("governed EEX vintage catalog is unreadable") from exc
    if not isinstance(catalog, Mapping):
        raise ValueError("governed EEX vintage catalog must be a mapping")
    if hashlib.sha256(catalog_payload).hexdigest() != str(catalog_binding.get("sha256", "")) or len(
        catalog_payload
    ) != catalog_binding.get("size_bytes"):
        raise ValueError("governed EEX vintage catalog bytes do not match data contract")
    from pfc_shaping.data.eex_historical_vintage import (
        materialize_eex_forward_history_as_of,
        verify_eex_historical_vintage_catalog,
    )

    vintages, evidence = verify_eex_historical_vintage_catalog(
        catalog,
        catalog_path=catalog_path,
        history_path=source,
    )
    if evidence.history_sha256 != str(entry.get("sha256", "")):
        raise ValueError("governed EEX vintage history does not match data contract")
    selected = materialize_eex_forward_history_as_of(
        vintages,
        valuation_timestamp=valuation_timestamp,
    )
    selected.attrs["verified_vintage_catalog"] = {
        "catalog_id": evidence.catalog_id,
        "catalog_sha256": evidence.catalog_sha256,
        "history_sha256": evidence.history_sha256,
        "snapshot_count": evidence.snapshot_count,
        "source_document_count": evidence.source_document_count,
        "data_cutoff_utc": evidence.data_cutoff_utc,
        "status": "VERIFIED_SIGNED_IMMUTABLE_VINTAGES",
        "pit_selection": dict(selected.attrs.get("pit_selection", {})),
    }
    return selected


def _verify_candidate_run_evidence(
    manifest: Mapping[str, object],
    *,
    run_id: str,
    promotion_evaluated_at: pd.Timestamp,
    data_contract: Mapping[str, object],
    data_pointer: Mapping[str, object],
    publication_head_observation: Mapping[str, object] | None,
    governed_run_spec: Mapping[str, object],
    active_config: Mapping[str, object],
    data_root: Path,
    candidate_finalized_at: object,
) -> tuple[pd.Timestamp, dict[str, object]]:
    structural_errors: list[str] = []
    if manifest.get("schema_version") != "lt_candidate_run.v1":
        structural_errors.append("schema_version")
    if str(manifest.get("run_id", "")) != run_id:
        structural_errors.append("run_id")
    if manifest.get("pipeline_scope") != "LT_ONLY":
        structural_errors.append("pipeline_scope")
    if manifest.get("promotion_eligible") is not False:
        structural_errors.append("candidate_must_not_self_approve")
    if manifest.get("reference_timestamp_is_explicit") is not True:
        structural_errors.append("reference_timestamp_not_explicit")
    try:
        valuation = _aware_utc_timestamp(
            manifest.get("reference_timestamp"),
            label="candidate reference_timestamp",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"candidate run manifest has invalid reference timestamp: {exc}") from exc
    if valuation > promotion_evaluated_at:
        structural_errors.append("reference_timestamp_after_promotion_clock")
    try:
        run_started_at = _aware_utc_timestamp(
            manifest.get("run_started_at_utc"),
            label="run_started_at_utc",
        )
        serialized_at = _aware_utc_timestamp(
            manifest.get("candidate_serialized_at_utc"),
            label="candidate_serialized_at_utc",
        )
        finalized_at = _aware_utc_timestamp(
            candidate_finalized_at,
            label="candidate bundle created_at_utc",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"candidate runtime timestamp is invalid: {exc}") from exc
    if run_started_at > serialized_at:
        structural_errors.append("run_started_after_candidate_serialized")
    if serialized_at > finalized_at:
        structural_errors.append("candidate_serialized_after_finalized")
    if valuation > run_started_at:
        structural_errors.append("candidate_valuation_after_run_started")
    if finalized_at > promotion_evaluated_at:
        structural_errors.append("candidate_finalized_after_promotion_clock")
    if manifest.get("data_layout") != "external_v2":
        structural_errors.append("data_layout_not_external_v2")
    if manifest.get("data_root_id") != LT_DATA_VIEW_ID:
        structural_errors.append("data_root_id")
    generation_id = str(manifest.get("data_generation_id", "")).strip()
    if not generation_id:
        structural_errors.append("data_generation_id_missing")

    if data_contract.get("schema_version") not in GOVERNED_LT_INPUT_SNAPSHOT_SCHEMAS:
        structural_errors.append("data_contract_schema")
    if data_contract.get("layout") != "external_v2":
        structural_errors.append("data_contract_layout")
    if str(data_contract.get("generation_id", "")) != generation_id:
        structural_errors.append("data_contract_generation_id")
    if data_contract.get("calibration_eligible") is not True:
        structural_errors.append("data_contract_not_calibration_eligible")
    if data_contract.get("source_class") != "GOVERNED_ACQUISITION":
        structural_errors.append("data_contract_source_class")
    else:
        try:
            verify_acquisition_contract(data_contract)
        except ValueError:
            structural_errors.append("data_contract_acquisition_attestation")
    acquisition_id = str(data_contract.get("acquisition_id", "")).strip()
    if not acquisition_id:
        structural_errors.append("data_contract_acquisition_id")
    contract_available_at: pd.Timestamp | None = None
    try:
        contract_available_at = _aware_utc_timestamp(
            data_contract.get("available_at_utc"),
            label="data contract available_at_utc",
        )
    except (TypeError, ValueError):
        structural_errors.append("data_contract_available_at")
    else:
        if contract_available_at > valuation:
            structural_errors.append("data_contract_available_after_valuation")

    contract_receipt = manifest.get("data_contract")
    pointer_receipt = manifest.get("data_pointer")
    if not isinstance(contract_receipt, Mapping) or not _is_sha256(
        str(contract_receipt.get("sha256", ""))
    ):
        structural_errors.append("data_contract_receipt")
    if not isinstance(pointer_receipt, Mapping) or not _is_sha256(
        str(pointer_receipt.get("sha256", ""))
    ):
        structural_errors.append("data_pointer_receipt")
    elif pointer_receipt.get("logical_path") != "views/pfc_lt/current.json":
        structural_errors.append("data_pointer_not_canonical_lt_view")
    pointer_schema = data_pointer.get("schema_version")
    if pointer_schema != "lt_data_pointer.v2":
        structural_errors.append("data_pointer_schema")
    if str(data_pointer.get("generation_id", "")) != generation_id:
        structural_errors.append("data_pointer_generation_id")
    if isinstance(contract_receipt, Mapping) and str(
        data_pointer.get("contract_sha256", "")
    ) != str(contract_receipt.get("sha256", "")):
        structural_errors.append("data_pointer_contract_hash")
    if pointer_schema == "lt_data_pointer.v2":
        publication_receipts = {
            "intent": manifest.get("data_publication_intent"),
            "anchor_receipt": manifest.get("data_publication_anchor_receipt"),
            "head_observation": manifest.get("data_publication_head_observation"),
        }
        for name, publication_receipt in publication_receipts.items():
            if not isinstance(publication_receipt, Mapping) or not _is_sha256(
                str(publication_receipt.get("sha256", ""))
            ):
                structural_errors.append(f"data_publication_{name}_receipt")
        snapshot_binding = governed_run_spec.get("input_snapshot")
        observation_receipt = publication_receipts["head_observation"]
        if (
            not isinstance(snapshot_binding, Mapping)
            or not isinstance(observation_receipt, Mapping)
            or snapshot_binding.get("publication_head_observation_sha256")
            != observation_receipt.get("sha256")
        ):
            structural_errors.append("data_publication_head_observation_run_spec_binding")
        if (
            not isinstance(snapshot_binding, Mapping)
            or not _is_sha256(str(snapshot_binding.get("publication_head_challenge_nonce", "")))
            or not isinstance(publication_head_observation, Mapping)
            or snapshot_binding.get("publication_head_challenge_nonce")
            != publication_head_observation.get("challenge_nonce")
        ):
            structural_errors.append("data_publication_head_challenge_run_spec_binding")

    contract_files = data_contract.get("files")
    input_sources = manifest.get("input_sources")
    if not isinstance(contract_files, Mapping):
        structural_errors.append("data_contract_files")
        contract_files = {}
    if not isinstance(input_sources, Mapping):
        structural_errors.append("input_sources")
        input_sources = {}
    expected_consumed_roles = _expected_consumed_input_roles(
        active_config,
        contract_files=contract_files,
    )
    missing_consumed_roles = expected_consumed_roles.difference(input_sources)
    unexpected_consumed_roles = set(input_sources).difference(expected_consumed_roles)
    if missing_consumed_roles:
        structural_errors.append(
            f"input_sources_missing_consumed_roles={sorted(missing_consumed_roles)}"
        )
    if unexpected_consumed_roles:
        structural_errors.append(
            f"input_sources_unexpected_roles={sorted(unexpected_consumed_roles)}"
        )
    for role in REQUIRED_CANDIDATE_FRESHNESS_COLUMNS:
        expected = contract_files.get(role)
        actual = input_sources.get(role)
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
            structural_errors.append(f"input_source_{role}_missing")
            continue
        if expected.get("calibration_eligible") is not True:
            structural_errors.append(f"input_source_{role}_not_calibration_eligible")
        if str(expected.get("acquisition_id", "")) != acquisition_id:
            structural_errors.append(f"input_source_{role}_acquisition_id")
        try:
            declared_cadence = int(expected.get("expected_cadence_seconds", 0))
        except (TypeError, ValueError):
            declared_cadence = 0
        if declared_cadence != SANCTIONED_EXPECTED_CADENCE_SECONDS[role]:
            structural_errors.append(f"input_source_{role}_expected_cadence")
        try:
            role_available_at = _aware_utc_timestamp(
                expected.get("available_at_utc"),
                label=f"input source {role} available_at_utc",
            )
        except (TypeError, ValueError):
            structural_errors.append(f"input_source_{role}_available_at")
        else:
            if role_available_at > valuation:
                structural_errors.append(f"input_source_{role}_available_after_valuation")
            if contract_available_at is not None and role_available_at > contract_available_at:
                structural_errors.append(f"input_source_{role}_acquisition_cutoff")
        comparisons = {
            "logical_path": expected.get("path"),
            "size_bytes": expected.get("size_bytes"),
            "sha256": expected.get("sha256"),
        }
        for key, expected_value in comparisons.items():
            if str(actual.get(key, "")) != str(expected_value):
                structural_errors.append(f"input_source_{role}_{key}")
        if not _is_sha256(str(actual.get("frame_sha256", ""))):
            structural_errors.append(f"input_source_{role}_frame_sha256")
        try:
            if int(actual.get("rows", 0)) <= 0:
                structural_errors.append(f"input_source_{role}_rows")
        except (TypeError, ValueError):
            structural_errors.append(f"input_source_{role}_rows")
    for role in sorted(set(input_sources).difference(REQUIRED_CANDIDATE_FRESHNESS_COLUMNS)):
        expected = contract_files.get(role)
        actual = input_sources.get(role)
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
            structural_errors.append(f"input_source_{role}_missing")
            continue
        if expected.get("calibration_eligible") is not True:
            structural_errors.append(f"input_source_{role}_not_calibration_eligible")
        if str(expected.get("acquisition_id", "")) != acquisition_id:
            structural_errors.append(f"input_source_{role}_acquisition_id")
        try:
            available = _aware_utc_timestamp(
                expected.get("available_at_utc"),
                label=f"input source {role} available_at_utc",
            )
        except (TypeError, ValueError):
            structural_errors.append(f"input_source_{role}_available_at")
        else:
            if available > valuation:
                structural_errors.append(f"input_source_{role}_available_after_valuation")
            if contract_available_at is not None and available > contract_available_at:
                structural_errors.append(f"input_source_{role}_acquisition_cutoff")
        for key, expected_value in {
            "logical_path": expected.get("path"),
            "size_bytes": expected.get("size_bytes"),
            "sha256": expected.get("sha256"),
        }.items():
            if str(actual.get(key, "")) != str(expected_value):
                structural_errors.append(f"input_source_{role}_{key}")
        if role in SANCTIONED_EXPECTED_CADENCE_SECONDS:
            try:
                cadence = int(expected.get("expected_cadence_seconds", 0))
            except (TypeError, ValueError):
                cadence = 0
            if cadence != SANCTIONED_EXPECTED_CADENCE_SECONDS[role]:
                structural_errors.append(f"input_source_{role}_expected_cadence")

    if structural_errors:
        raise ValueError(
            "candidate run manifest/data contract is not promotion-grade: "
            + ",".join(sorted(set(structural_errors)))
        )

    freshness_errors: list[str] = []
    reports = manifest.get("freshness_reports")
    if not isinstance(reports, Mapping):
        reports = {}
        freshness_errors.append("freshness_reports_missing")
    missing_reports = set(REQUIRED_CANDIDATE_FRESHNESS_COLUMNS).difference(reports)
    if missing_reports:
        freshness_errors.append(f"missing_reports={sorted(missing_reports)}")
    for role, columns in REQUIRED_CANDIDATE_FRESHNESS_COLUMNS.items():
        report = reports.get(role)
        if not isinstance(report, Mapping):
            continue
        if str(report.get("dataset", "")) != role:
            freshness_errors.append(f"{role}.dataset")
        try:
            if int(report.get("checks_passed", 0)) <= 0:
                freshness_errors.append(f"{role}.checks_passed")
        except (TypeError, ValueError):
            freshness_errors.append(f"{role}.checks_passed")
        if list(report.get("warnings", []) or []):
            freshness_errors.append(f"{role}.warnings")
        if list(report.get("errors", []) or []):
            freshness_errors.append(f"{role}.errors")
        metrics = report.get("metrics")
        if not isinstance(metrics, Mapping):
            freshness_errors.append(f"{role}.metrics")
            continue
        for column in columns:
            prefix = f"{column}."
            try:
                finite_fraction = _finite_metric(metrics, prefix + "finite_fraction")
                recent_fraction = _finite_metric(metrics, prefix + "recent_finite_fraction")
                recent_window = _finite_metric(metrics, prefix + "recent_window_days")
                declared_gap = _finite_metric(metrics, prefix + "max_finite_gap_days")
                declared_age = _finite_metric(metrics, prefix + "age_days")
                latest = _aware_utc_timestamp(
                    metrics.get(prefix + "latest_finite_timestamp"),
                    label=f"{role}.{column}.latest_finite_timestamp",
                )
            except (TypeError, ValueError) as exc:
                freshness_errors.append(f"{role}.{column}.invalid_metrics:{exc}")
                continue
            if latest > valuation:
                freshness_errors.append(f"{role}.{column}.latest_after_valuation")
                continue
            age_at_valuation = (valuation - latest).total_seconds() / 86400.0
            age_at_promotion = (promotion_evaluated_at - latest).total_seconds() / 86400.0
            if abs(declared_age - age_at_valuation) > 1e-8:
                freshness_errors.append(f"{role}.{column}.declared_age_mismatch")
            if finite_fraction < SANCTIONED_MIN_FINITE_FRACTION:
                freshness_errors.append(f"{role}.{column}.finite_fraction")
            if recent_fraction < SANCTIONED_MIN_RECENT_FINITE_FRACTION:
                freshness_errors.append(f"{role}.{column}.recent_finite_fraction")
            if abs(recent_window - 7.0) > 1e-8:
                freshness_errors.append(f"{role}.{column}.recent_window_days")
            if age_at_promotion > SANCTIONED_FRESHNESS_MAX_AGE_DAYS[role]:
                freshness_errors.append(f"{role}.{column}.stale_at_promotion")
            if max(declared_gap, age_at_promotion) > SANCTIONED_FRESHNESS_MAX_GAP_DAYS[role]:
                freshness_errors.append(f"{role}.{column}.gap_at_promotion")

    freshness_errors.extend(
        _recompute_candidate_input_freshness(
            data_root=data_root,
            data_contract=data_contract,
            data_pointer=data_pointer,
            candidate_manifest=manifest,
            valuation_timestamp=valuation,
            promotion_evaluated_at=promotion_evaluated_at,
        )
    )

    status = "PASS" if not freshness_errors else "CRITICAL"
    evidence = (
        "candidate run inputs remain fresh at promotion evaluation; "
        f"valuation_timestamp={valuation.isoformat()}; "
        f"promotion_evaluated_at={promotion_evaluated_at.isoformat()}"
        if not freshness_errors
        else (
            "candidate run freshness replay failed; "
            f"errors={freshness_errors}; valuation_timestamp={valuation.isoformat()}; "
            f"promotion_evaluated_at={promotion_evaluated_at.isoformat()}"
        )
    )
    return valuation, _governance_gate_row(
        gate_id="candidate_run_freshness",
        status=status,
        severity="INFO" if not freshness_errors else "P0",
        product="candidate_run",
        parent_block_id="candidate_run_manifest",
        metric_name="candidate_run_freshness_error",
        metric_value=0.0 if not freshness_errors else 1.0,
        threshold_source="code_sanctioned_production_freshness_policy",
        evidence=evidence,
        remediation_hint="Regenerate from one fresh governed external_v2 acquisition generation.",
    )


def _expected_consumed_input_roles(
    config: Mapping[str, object],
    *,
    contract_files: Mapping[str, object],
) -> set[str]:
    return consumed_lt_input_roles(config, available_roles=set(contract_files))


def _aware_utc_timestamp(value: object, *, label: str) -> pd.Timestamp:
    if value is None or value == "":
        raise ValueError(f"{label} is missing")
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _recompute_candidate_input_freshness(
    *,
    data_root: Path,
    data_contract: Mapping[str, object],
    data_pointer: Mapping[str, object],
    candidate_manifest: Mapping[str, object],
    valuation_timestamp: pd.Timestamp,
    promotion_evaluated_at: pd.Timestamp,
) -> list[str]:
    errors: list[str] = []
    raw_root = str(data_root).strip()
    if not raw_root:
        return ["data_root_missing"]
    selected_root = Path(raw_root).expanduser()
    if not selected_root.is_absolute():
        return [f"data_root_not_absolute:{selected_root}"]
    root = selected_root.resolve()
    if not root.is_dir():
        return [f"data_root_unavailable:{root}"]
    contract_rel = str(data_pointer.get("contract_path", "")).strip()
    try:
        live_contract_path = (root / contract_rel).resolve()
        live_contract_path.relative_to(root)
    except (OSError, ValueError):
        return ["data_contract_path_escapes_data_root"]
    try:
        live_contract_bytes = read_stable_single_link_file(
            live_contract_path,
            label="live LT input contract",
        )
    except OSError as exc:
        return [f"data_contract_unavailable:{exc}"]
    archived_contract = candidate_manifest.get("data_contract")
    expected_contract_hash = (
        str(archived_contract.get("sha256", "")) if isinstance(archived_contract, Mapping) else ""
    )
    if hashlib.sha256(live_contract_bytes).hexdigest() != expected_contract_hash:
        return ["data_contract_live_hash_mismatch"]
    try:
        live_contract = load_strict_json(live_contract_bytes.decode("utf-8"))
    except (UnicodeDecodeError, StrictStructuredDataError):
        return ["data_contract_live_invalid_json"]
    if live_contract != dict(data_contract):
        return ["data_contract_live_semantic_mismatch"]
    try:
        validate_governed_lt_snapshot_bundle(live_contract_path.parent, live_contract)
    except (OSError, TypeError, ValueError) as exc:
        return [f"data_contract_bundle_invalid:{exc}"]

    contract_files = data_contract.get("files")
    receipts = candidate_manifest.get("input_sources")
    declared_reports = candidate_manifest.get("freshness_reports")
    if not isinstance(contract_files, Mapping) or not isinstance(receipts, Mapping):
        return ["candidate_input_contract_or_receipts_missing"]
    if not isinstance(declared_reports, Mapping):
        return ["candidate_freshness_reports_missing"]
    snapshot_root = live_contract_path.parent
    for role, columns in REQUIRED_CANDIDATE_FRESHNESS_COLUMNS.items():
        contract_entry = contract_files.get(role)
        receipt = receipts.get(role)
        if not isinstance(contract_entry, Mapping) or not isinstance(receipt, Mapping):
            errors.append(f"{role}.source_evidence_missing")
            continue
        source_rel = str(contract_entry.get("path", "")).strip()
        try:
            source = (snapshot_root / source_rel).resolve()
            source.relative_to(snapshot_root)
        except (OSError, ValueError):
            errors.append(f"{role}.source_path_escapes_generation")
            continue
        try:
            payload = read_stable_single_link_file(
                source,
                label=f"live LT input {role}",
            )
            frame = pd.read_parquet(BytesIO(payload))
        except (OSError, ValueError) as exc:
            errors.append(f"{role}.source_unreadable:{exc}")
            continue
        errors.extend(_candidate_cadence_errors(frame, role=role))
        if len(payload) != int(contract_entry.get("size_bytes", -1)):
            errors.append(f"{role}.source_size_mismatch")
        source_hash = hashlib.sha256(payload).hexdigest()
        if source_hash != str(contract_entry.get("sha256", "")):
            errors.append(f"{role}.source_hash_mismatch")
        if source_hash != str(receipt.get("sha256", "")):
            errors.append(f"{role}.receipt_hash_mismatch")
        if len(frame) != int(receipt.get("rows", -1)):
            errors.append(f"{role}.receipt_rows_mismatch")
        if dataframe_sha256(frame) != str(receipt.get("frame_sha256", "")):
            errors.append(f"{role}.receipt_frame_hash_mismatch")
        kwargs = {
            "name": role,
            "required_columns": list(columns),
            "min_rows": 1,
            "max_age_days": SANCTIONED_FRESHNESS_MAX_AGE_DAYS[role],
            "fail_on_stale": True,
            "min_finite_fraction": SANCTIONED_MIN_FINITE_FRACTION,
            "recent_window_days": 7.0,
            "min_recent_finite_fraction": SANCTIONED_MIN_RECENT_FINITE_FRACTION,
            "max_finite_gap_days": SANCTIONED_FRESHNESS_MAX_GAP_DAYS[role],
        }
        try:
            valuation_report = validate_input_frame(
                frame,
                reference_timestamp=valuation_timestamp,
                **kwargs,
            )
        except QualityGateError as exc:
            errors.append(f"{role}.valuation_recompute_failed:{exc}")
            continue
        recomputed_report = {
            "dataset": valuation_report.dataset,
            "checks_passed": int(valuation_report.checks_passed),
            "warnings": list(valuation_report.warnings),
            "errors": list(valuation_report.errors),
            "metrics": dict(valuation_report.metrics),
        }
        if _json_clean(recomputed_report) != _json_clean(declared_reports.get(role)):
            errors.append(f"{role}.declared_freshness_report_mismatch")
        try:
            validate_input_frame(
                frame,
                reference_timestamp=promotion_evaluated_at,
                **kwargs,
            )
        except QualityGateError as exc:
            errors.append(f"{role}.promotion_recompute_failed:{exc}")
    for role in sorted(set(receipts).difference(REQUIRED_CANDIDATE_FRESHNESS_COLUMNS)):
        contract_entry = contract_files.get(role)
        receipt = receipts.get(role)
        if not isinstance(contract_entry, Mapping) or not isinstance(receipt, Mapping):
            errors.append(f"{role}.source_evidence_missing")
            continue
        try:
            source = (snapshot_root / str(contract_entry.get("path", ""))).resolve()
            source.relative_to(snapshot_root)
            payload = read_stable_single_link_file(
                source,
                label=f"live LT optional input {role}",
            )
            frame = pd.read_parquet(BytesIO(payload))
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{role}.source_unreadable:{exc}")
            continue
        if role in SANCTIONED_EXPECTED_CADENCE_SECONDS:
            errors.extend(_candidate_cadence_errors(frame, role=role))
        if role == "eex_forwards_history":
            try:
                validate_governed_forward_history_frame(
                    frame,
                    reference_timestamp=valuation_timestamp,
                )
            except ValueError as exc:
                errors.append(f"{role}.semantic_pit_failed:{exc}")
        if len(payload) != int(contract_entry.get("size_bytes", -1)):
            errors.append(f"{role}.source_size_mismatch")
        source_hash = hashlib.sha256(payload).hexdigest()
        if source_hash != str(contract_entry.get("sha256", "")):
            errors.append(f"{role}.source_hash_mismatch")
        if source_hash != str(receipt.get("sha256", "")):
            errors.append(f"{role}.receipt_hash_mismatch")
        if len(frame) != int(receipt.get("rows", -1)):
            errors.append(f"{role}.receipt_rows_mismatch")
        if dataframe_sha256(frame) != str(receipt.get("frame_sha256", "")):
            errors.append(f"{role}.receipt_frame_hash_mismatch")
        columns = OPTIONAL_CANDIDATE_FRESHNESS_COLUMNS.get(role)
        if columns is None:
            continue
        kwargs = {
            "name": role,
            "required_columns": list(columns),
            "min_rows": 1,
            "max_age_days": SANCTIONED_FRESHNESS_MAX_AGE_DAYS[role],
            "fail_on_stale": True,
            "min_finite_fraction": SANCTIONED_MIN_FINITE_FRACTION,
            "recent_window_days": 7.0,
            "min_recent_finite_fraction": SANCTIONED_MIN_RECENT_FINITE_FRACTION,
            "max_finite_gap_days": SANCTIONED_FRESHNESS_MAX_GAP_DAYS[role],
        }
        try:
            valuation_report = validate_input_frame(
                frame,
                reference_timestamp=valuation_timestamp,
                **kwargs,
            )
        except QualityGateError as exc:
            errors.append(f"{role}.valuation_recompute_failed:{exc}")
            continue
        recomputed_report = {
            "dataset": valuation_report.dataset,
            "checks_passed": int(valuation_report.checks_passed),
            "warnings": list(valuation_report.warnings),
            "errors": list(valuation_report.errors),
            "metrics": dict(valuation_report.metrics),
        }
        if _json_clean(recomputed_report) != _json_clean(declared_reports.get(role)):
            errors.append(f"{role}.declared_freshness_report_mismatch")
        try:
            validate_input_frame(
                frame,
                reference_timestamp=promotion_evaluated_at,
                **kwargs,
            )
        except QualityGateError as exc:
            errors.append(f"{role}.promotion_recompute_failed:{exc}")
    return errors


def _candidate_cadence_errors(frame: pd.DataFrame, *, role: str) -> list[str]:
    if role == "hydro":
        return weekly_local_grid_errors(
            frame,
            name=role,
            timezone="Europe/Zurich",
            weekday=0,
            min_grid_coverage=SANCTIONED_MIN_GRID_COVERAGE,
        )
    return input_grid_errors(
        frame,
        name=role,
        expected_cadence_seconds=SANCTIONED_EXPECTED_CADENCE_SECONDS[role],
        expected_phase_offset_seconds=SANCTIONED_EXPECTED_PHASE_OFFSET_SECONDS.get(role, 0),
        min_grid_coverage=SANCTIONED_MIN_GRID_COVERAGE,
    )


def _finite_metric(metrics: Mapping[str, object], key: str) -> float:
    value = float(metrics.get(key))
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _require_output_outside_bundle(path: Path, bundle_path: Path, *, role: str) -> None:
    resolved = Path(path).resolve()
    bundle = bundle_path.resolve()
    if resolved == bundle or bundle in resolved.parents:
        raise ValueError(f"{role} output must be outside finalized candidate bundle: {resolved}")


def build_manifest_governance_gates(
    *,
    run_timestamp: pd.Timestamp | None,
    production_manifest: Mapping[str, object],
    export_manifest: Mapping[str, object],
    selected_config: Mapping[str, object],
    active_config_hash: str,
    selected_config_hash: str,
    product_normalization_summary: Mapping[str, object] | None = None,
    product_normalization_summary_path: Path | None = None,
    product_normalization_summary_sha256: str | None = None,
    product_normalization_replay: Mapping[str, object] | None = None,
    product_normalization_replay_errors: list[str] | None = None,
    monthly_candidate_manifest: Mapping[str, object] | None = None,
    monthly_candidate_manifest_path: Path | None = None,
    monthly_candidate_manifest_sha256: str | None = None,
    product_normalization_evidence: Mapping[str, object] | None = None,
    production_manifest_path: Path | None = None,
    production_manifest_sha256: str | None = None,
    selected_config_path: Path | None = None,
) -> pd.DataFrame:
    """Return strict governance gates backed by manifest-derived hashes."""

    if product_normalization_evidence is None:
        product_normalization_evidence = _product_normalization_evidence(
            production_manifest=production_manifest,
            export_manifest=export_manifest,
            selected_config=selected_config,
            promotion_timestamp=run_timestamp,
        )

    base = (
        build_monthly_curve_governance_gates(
            run_timestamp=pd.Timestamp(run_timestamp)
            if run_timestamp is not None
            else pd.Timestamp.utcnow(),
            active_config_hash=active_config_hash,
            selected_config_hash=selected_config_hash,
            production_monthly_solution_hash=_required_manifest_hash(
                production_manifest,
                "monthly_solution_hash",
                role="production",
            ),
            export_monthly_solution_hash=_required_manifest_hash(
                export_manifest,
                "monthly_solution_hash",
                role="export",
            ),
            production_active_constraints_hash=_required_manifest_hash(
                production_manifest,
                "active_constraints_hash",
                role="production",
            ),
            export_active_constraints_hash=_required_manifest_hash(
                export_manifest,
                "active_constraints_hash",
                role="export",
            ),
            require_lambda_artifact=True,
            require_path_parity=True,
        )
        .loc[lambda frame: frame["gate_id"].isin(REQUIRED_GOVERNANCE_GATES)]
        .reset_index(drop=True)
    )
    extra = pd.DataFrame(
        [
            _delivered_product_normalization_row(
                product_normalization_summary,
                product_normalization_summary_path=product_normalization_summary_path,
                product_normalization_summary_sha256=product_normalization_summary_sha256,
                product_normalization_replay=product_normalization_replay,
                product_normalization_replay_errors=product_normalization_replay_errors,
                monthly_candidate_manifest=monthly_candidate_manifest,
                monthly_candidate_manifest_path=monthly_candidate_manifest_path,
                monthly_candidate_manifest_sha256=monthly_candidate_manifest_sha256,
                expected_evidence=product_normalization_evidence,
            ),
            _lt_probabilistic_output_governance_row(export_manifest),
            _ompex_training_selection_exclusion_row(selected_config),
            _selected_config_production_approval_row(selected_config),
            _selected_config_manifest_parity_row(
                selected_config=selected_config,
                production_manifest=production_manifest,
                export_manifest=export_manifest,
                active_config_hash=active_config_hash,
                selected_config_hash=selected_config_hash,
                production_manifest_path=production_manifest_path,
                production_manifest_sha256=production_manifest_sha256,
                selected_config_path=selected_config_path,
            ),
        ]
    )
    return pd.concat([base, extra], ignore_index=True)


def _replace_gate_rows(audit_gates: pd.DataFrame, governance: pd.DataFrame) -> pd.DataFrame:
    base = audit_gates[~audit_gates["gate_id"].astype(str).isin(REQUIRED_GOVERNANCE_GATES)]
    columns = list(dict.fromkeys([*base.columns, *governance.columns]))
    nonempty_columns = [
        frame.dropna(axis=1, how="all") for frame in (base, governance) if not frame.empty
    ]
    if not nonempty_columns:
        return pd.DataFrame(columns=columns)
    return pd.concat(nonempty_columns, ignore_index=True).reindex(columns=columns)


def _load_mapping(path: Path) -> dict[str, object]:
    captured = _capture_artifact(path)
    return _load_mapping_bytes(_captured_raw(captured), captured.path)


def _load_mapping_bytes(raw: bytes, path: Path) -> dict[str, object]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        value = load_strict_json(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"structured artifact is not an object: {path}")
        return value
    if suffix in {".yaml", ".yml"}:
        value = load_strict_yaml(raw.decode("utf-8")) or {}
        if not isinstance(value, dict):
            raise ValueError(f"structured artifact is not an object: {path}")
        return value
    raise ValueError(f"unsupported structured file type for {path}")


def _capture_canonical_assembled_artifacts(
    captured: dict[str, CapturedArtifact],
    *,
    bundle_path: Path,
    run_id: str,
) -> None:
    seal = verify_assembled_candidate_evidence(bundle_path, expected_run_id=run_id)
    entries = seal.get("artifacts")
    if not isinstance(entries, Mapping):
        raise ValueError("assembled candidate seal inventory is invalid")
    role_map = {
        "audit_gates": "audit_gates",
        "historical_thresholds": "historical_thresholds",
        "production_manifest": "production_manifest",
        "export_manifest": "export_manifest",
        "selected_config_artifact": "selected_config_run_parity",
        "product_normalization_summary": "product_normalization_summary",
    }
    for argument_role, seal_role in role_map.items():
        provided = captured.get(argument_role)
        entry = entries.get(seal_role)
        if provided is None or not isinstance(entry, Mapping):
            raise ValueError(f"assembled candidate role is missing: {seal_role}")
        if provided.sha256 != str(entry.get("sha256", "")):
            raise ValueError(f"release artifact differs from assembled seal: {argument_role}")
        canonical = _capture_artifact(bundle_path / str(entry.get("path", "")))
        _require_artifact_in_bundle(
            canonical,
            bundle_path,
            _load_mapping(bundle_path / "candidate_bundle_manifest.json"),
            role=seal_role,
        )
        if canonical.sha256 != str(entry.get("sha256", "")) or len(_captured_raw(canonical)) != int(
            entry.get("size_bytes", -1)
        ):
            raise ValueError(f"assembled candidate role changed: {seal_role}")
        captured[argument_role] = canonical
    for seal_role in (
        "product_normalization_evidence",
        "selected_lambda_decision",
    ):
        entry = entries.get(seal_role)
        if not isinstance(entry, Mapping):
            raise ValueError(f"assembled candidate role is missing: {seal_role}")
        captured[seal_role] = _capture_artifact(bundle_path / str(entry.get("path", "")))


def _assembled_export_manifest_view(
    export_manifest: Mapping[str, object],
    *,
    production_manifest: Mapping[str, object],
    export_capture: CapturedArtifact,
    production_capture: CapturedArtifact,
    bundle_path: Path,
) -> dict[str, object]:
    if export_manifest.get("schema_version") != "lt_candidate_hourly_export.v1":
        raise ValueError("assembled export manifest schema mismatch")
    parents = export_manifest.get("parents")
    parent = parents.get("production_manifest") if isinstance(parents, Mapping) else None
    _verify_assembled_ref(
        parent,
        artifact=production_capture,
        bundle_path=bundle_path,
        role="export.production_manifest",
    )
    merged = dict(production_manifest)
    merged.update(export_manifest)
    merged["assembled_export_manifest_sha256"] = export_capture.sha256
    return merged


def _assembled_product_normalization_contract(
    *,
    captured: dict[str, CapturedArtifact],
    product_summary: Mapping[str, object] | None,
    bundle_path: Path,
    run_id: str,
) -> dict[str, object]:
    if not isinstance(product_summary, Mapping):
        raise ValueError("assembled product normalization summary is missing")
    evidence_capture = captured.get("product_normalization_evidence")
    selected_decision = captured.get("selected_lambda_decision")
    if evidence_capture is None or selected_decision is None:
        raise ValueError("assembled product evidence parents are missing")
    evidence = _load_mapping_bytes(
        _captured_raw(evidence_capture),
        evidence_capture.path,
    )
    if evidence.get("schema_version") != "candidate_product_normalization_evidence.v1" or str(
        evidence.get("run_id", "")
    ) != str(run_id):
        raise ValueError("assembled product evidence identity mismatch")
    parents = evidence.get("parents")
    artifacts = evidence.get("artifacts")
    if not isinstance(parents, Mapping) or not isinstance(artifacts, Mapping):
        raise ValueError("assembled product evidence inventory is invalid")
    for role, capture_key in (
        ("production_manifest", "production_manifest"),
        ("export_manifest", "export_manifest"),
        ("selected_config_run_parity", "selected_config_artifact"),
        ("selected_lambda_decision", "selected_lambda_decision"),
    ):
        _verify_assembled_ref(
            parents.get(role),
            artifact=captured[capture_key],
            bundle_path=bundle_path,
            role=f"product_evidence.{role}",
        )
    _verify_assembled_ref(
        artifacts.get("product_normalization_summary"),
        artifact=captured["product_normalization_summary"],
        bundle_path=bundle_path,
        role="product_evidence.product_normalization_summary",
    )
    parity = _load_mapping_bytes(
        _captured_raw(captured["selected_config_artifact"]),
        captured["selected_config_artifact"].path,
    )
    if (
        parity.get("schema_version") != "monthly_curve_selected_config_run_parity.v1"
        or parity.get("status") != "PASS"
        or parity.get("production_approved") is not True
        or parity.get("selection_status") != "PRODUCTION_APPROVED"
    ):
        raise ValueError("assembled selected-config parity is not production approved")
    policy = product_summary.get("source_hierarchy_policy")
    expected = {
        "summary_sha256": captured["product_normalization_summary"].sha256,
        "input_csv_sha256": str(product_summary.get("input_csv_sha256", "")),
        "forwards_sha256": str(product_summary.get("forwards_sha256", "")),
        "forward_snapshot_date": str(product_summary.get("forward_snapshot_date", "")),
        "market": str(product_summary.get("market", "")),
        "quote_conflict_identity_hash": str(
            product_summary.get("quote_conflict_identity_hash", "")
        ),
        "source_hierarchy_policy_sha256": str(
            policy.get("sha256", "") if isinstance(policy, Mapping) else ""
        ),
    }
    missing = sorted(
        key for key in REQUIRED_PRODUCT_NORMALIZATION_EVIDENCE_FIELDS if not expected[key]
    )
    if missing:
        raise ValueError(f"assembled product evidence fields are missing: {missing}")
    return expected


def _verify_assembled_ref(
    entry: object,
    *,
    artifact: CapturedArtifact,
    bundle_path: Path,
    role: str,
) -> None:
    if not isinstance(entry, Mapping):
        raise ValueError(f"assembled artifact reference is missing: {role}")
    try:
        relative = artifact.path.relative_to(bundle_path.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"assembled artifact escapes bundle: {role}") from exc
    if (
        entry.get("path") != relative
        or entry.get("sha256") != artifact.sha256
        or int(entry.get("size_bytes", -1)) != len(_captured_raw(artifact))
    ):
        raise ValueError(f"assembled artifact reference mismatch: {role}")


def _resolve_forward_source_path(manifest: dict[str, object], manifest_path: Path) -> None:
    snapshot = manifest.get("forward_snapshot")
    if not isinstance(snapshot, Mapping):
        return
    source_value = snapshot.get("source_path")
    if not source_value:
        return
    source_path = Path(str(source_value))
    if source_path.is_absolute():
        return
    resolved_snapshot = dict(snapshot)
    resolved_snapshot["source_path"] = str((manifest_path.resolve().parent / source_path).resolve())
    manifest["forward_snapshot"] = resolved_snapshot


def _verify_forward_source_in_bundle(
    manifest: Mapping[str, object],
    bundle_path: Path,
    bundle_manifest: Mapping[str, object],
) -> None:
    snapshot = manifest.get("forward_snapshot")
    if not isinstance(snapshot, Mapping) or not snapshot.get("source_path"):
        raise ValueError("forward snapshot source_path is missing from candidate manifest")
    source_path = Path(str(snapshot["source_path"])).resolve()
    try:
        relative = source_path.relative_to(bundle_path.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"forward workbook is outside candidate bundle: {source_path}") from exc
    files = bundle_manifest.get("files")
    expected_hash = str(snapshot.get("source_sha256", ""))
    if not isinstance(files, Mapping) or str(files.get(relative, "")) != expected_hash:
        raise ValueError(f"forward workbook is not hash-bound in candidate bundle: {relative}")


def _require_artifact_in_bundle(
    artifact: CapturedArtifact,
    bundle_path: Path,
    bundle_manifest: Mapping[str, object],
    *,
    role: str,
) -> None:
    try:
        relative = artifact.path.relative_to(bundle_path.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{role} is outside candidate bundle: {artifact.path}") from exc
    files = bundle_manifest.get("files")
    if not isinstance(files, Mapping) or str(files.get(relative, "")) != artifact.sha256:
        raise ValueError(f"{role} is not hash-bound in candidate bundle: {relative}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_captured_artifacts_unchanged(
    artifacts: Mapping[str, CapturedArtifact],
) -> None:
    changed: list[str] = []
    for role, captured in artifacts.items():
        try:
            current_hash = _sha256_file(captured.path)
        except OSError:
            changed.append(f"{role}:missing")
            continue
        if current_hash != captured.sha256:
            changed.append(f"{role}:sha256_mismatch")
    if changed:
        raise ValueError(f"promotion inputs changed during evaluation: {changed}")


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(_json_clean(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite promotion receipt: {destination}"
            ) from exc
    finally:
        temp.unlink(missing_ok=True)


def _selected_config_hash(config: Mapping[str, object], path: Path) -> str:
    canonical = config.get("canonical_config")
    if not isinstance(canonical, Mapping):
        raise ValueError(f"selected config artifact missing canonical_config: {path}")
    return config_hash(canonical)


def _product_normalization_evidence(
    *,
    production_manifest: Mapping[str, object],
    export_manifest: Mapping[str, object],
    selected_config: Mapping[str, object],
    promotion_timestamp: pd.Timestamp | None = None,
) -> dict[str, object]:
    selected = selected_config.get(PRODUCT_NORMALIZATION_EVIDENCE_KEY)
    production = production_manifest.get(PRODUCT_NORMALIZATION_EVIDENCE_KEY)
    export = export_manifest.get(PRODUCT_NORMALIZATION_EVIDENCE_KEY)
    if not isinstance(selected, Mapping):
        return {
            "expected": {},
            "errors": [f"selected_config_missing_{PRODUCT_NORMALIZATION_EVIDENCE_KEY}"],
        }
    expected = {str(key): value for key, value in selected.items()}
    errors: list[str] = []
    for key in REQUIRED_PRODUCT_NORMALIZATION_EVIDENCE_FIELDS:
        if not expected.get(key):
            errors.append(f"{key}_missing")
    for role, evidence in (("production", production), ("export", export)):
        if evidence is None:
            errors.append(f"{role}_{PRODUCT_NORMALIZATION_EVIDENCE_KEY}_missing")
            continue
        if not isinstance(evidence, Mapping):
            errors.append(f"{role}_{PRODUCT_NORMALIZATION_EVIDENCE_KEY}_invalid")
            continue
        for key in REQUIRED_PRODUCT_NORMALIZATION_EVIDENCE_FIELDS:
            value = evidence.get(key)
            if value is None or value == "":
                errors.append(f"{role}_{key}_missing")
            elif str(value) != str(expected.get(key, "")):
                errors.append(f"{role}_{key}_mismatch")
        for key in ("quote_conflict_identity_hash", "source_hierarchy_policy_sha256"):
            value = evidence.get(key)
            expected_value = expected.get(key)
            if expected_value not in (None, "") and value in (None, ""):
                errors.append(f"{role}_{key}_missing")
            elif value is not None and str(value) != str(expected.get(key, "")):
                errors.append(f"{role}_{key}_mismatch")
    forward_binding: dict[str, object] = {}
    solver_binding: dict[str, object] = {}
    constraint_governance_binding: dict[str, object] = {}
    for role, manifest in (("production", production_manifest), ("export", export_manifest)):
        role_constraint_binding, role_constraint_errors = _verify_constraint_governance(manifest)
        errors.extend(f"{role}_{error}" for error in role_constraint_errors)
        if role_constraint_binding:
            if not constraint_governance_binding:
                constraint_governance_binding = role_constraint_binding
            elif role_constraint_binding != constraint_governance_binding:
                errors.append("production_export_constraint_governance_mismatch")
        snapshot = manifest.get("forward_snapshot")
        eligibility = manifest.get("forward_eligibility")
        if not isinstance(snapshot, Mapping) or not isinstance(eligibility, Mapping):
            errors.append(f"{role}_forward_binding_missing")
            continue
        try:
            verified = verify_forward_manifest_binding(
                snapshot,
                eligibility,
                expected_max_age_business_days=PRODUCTION_EEX_MAX_AGE_BUSINESS_DAYS,
                verify_source=True,
                expected_reference_timestamp=promotion_timestamp,
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"{role}_forward_binding_invalid:{exc}")
            continue
        binding = {
            "forward_snapshot_id": verified["snapshot_id"],
            "forward_observation_id": verified["observation_id"],
            "forward_source_sha256": verified["source_sha256"],
            "forward_quote_set_sha256": verified["quote_set_sha256"],
            "forward_quote_lineage_sha256": verified["quote_lineage_sha256"],
            "forward_policy_sha256": verified["policy_sha256"],
            "forward_eligibility_id": verified["eligibility_id"],
        }
        if not forward_binding:
            forward_binding = binding
        elif binding != forward_binding:
            errors.append("production_export_forward_binding_mismatch")
        role_solver_binding = {
            "monthly_solution_hash": manifest.get("monthly_solution_hash"),
            "active_constraints_hash": manifest.get("active_constraints_hash"),
            "active_config_hash": _active_config_hash(manifest),
        }
        if any(value in (None, "") for value in role_solver_binding.values()):
            errors.append(f"{role}_solver_binding_incomplete")
        elif not solver_binding:
            solver_binding = role_solver_binding
        elif role_solver_binding != solver_binding:
            errors.append("production_export_solver_binding_mismatch")
    return {
        "expected": expected,
        "errors": errors,
        "forward_binding": forward_binding,
        "solver_binding": solver_binding,
        "constraint_governance_binding": constraint_governance_binding,
    }


def _verify_constraint_governance(
    manifest: Mapping[str, object],
) -> tuple[dict[str, object], list[str]]:
    errors: list[str] = []
    solver_config = manifest.get("solver_config") or manifest.get("config")
    if not isinstance(solver_config, Mapping):
        return {}, ["solver_config_missing"]
    errors.extend(
        solver_numerical_diagnostic_errors(
            manifest.get("solver_kkt"),
            solver_config=solver_config,
        )
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
    policy = manifest.get("product_hierarchy_policy")
    if not isinstance(policy, Mapping) or dict(policy) != expected_policy:
        errors.append("product_hierarchy_policy_semantic_mismatch")
    if str(manifest.get("product_hierarchy_policy_sha256", "")) != _sha256_json(expected_policy):
        errors.append("product_hierarchy_policy_sha256_mismatch")

    hard_quotes = manifest.get("hard_quotes")
    if not isinstance(hard_quotes, list) or not hard_quotes:
        errors.append("hard_quotes_missing")
        hard_quotes = []
    if str(manifest.get("hard_quote_set_hash", "")) != _sha256_json(hard_quotes):
        errors.append("hard_quote_set_hash_mismatch")
    forward_snapshot = manifest.get("forward_snapshot")
    snapshot_quotes = (
        forward_snapshot.get("quotes") if isinstance(forward_snapshot, Mapping) else None
    )
    if not isinstance(snapshot_quotes, list):
        errors.append("forward_snapshot_quotes_missing")
        snapshot_quotes = []
    expected_hard_quotes = [
        dict(quote)
        for quote in snapshot_quotes
        if isinstance(quote, Mapping) and str(quote.get("load_type", "")).upper() == "BASE"
    ]
    if hard_quotes != expected_hard_quotes:
        errors.append("hard_quotes_forward_snapshot_mismatch")
    hard_quote_products = [
        (str(quote.get("product", "")), str(quote.get("load_type", "")).upper())
        for quote in hard_quotes
        if isinstance(quote, Mapping)
    ]
    if len(hard_quote_products) != len(set(hard_quote_products)):
        errors.append("hard_quote_product_duplicate")
    quote_id_list = [
        str(quote.get("quote_id"))
        for quote in hard_quotes
        if isinstance(quote, Mapping) and quote.get("quote_id")
    ]
    quote_ids = set(quote_id_list)
    if len(quote_id_list) != len(hard_quotes):
        errors.append("hard_quote_id_missing")
    if len(quote_ids) != len(quote_id_list):
        errors.append("hard_quote_id_duplicate")
    quote_by_id = {
        str(quote.get("quote_id")): quote
        for quote in hard_quotes
        if isinstance(quote, Mapping) and quote.get("quote_id")
    }

    provenance_rows = manifest.get("constraint_provenance_rows")
    if not isinstance(provenance_rows, list) or not provenance_rows:
        errors.append("constraint_provenance_rows_missing")
        provenance_rows = []
    if str(manifest.get("constraint_provenance_hash", "")) != _sha256_json(provenance_rows):
        errors.append("constraint_provenance_hash_mismatch")
    provenance_products = [
        str(row.get("product", ""))
        for row in provenance_rows
        if isinstance(row, Mapping) and row.get("product")
    ]
    if len(provenance_products) != len(set(provenance_products)):
        errors.append("constraint_provenance_product_duplicate")
    provenance_by_product = {
        str(row.get("product", "")): row
        for row in provenance_rows
        if isinstance(row, Mapping) and row.get("product")
    }
    for index, row in enumerate(provenance_rows):
        if not isinstance(row, Mapping):
            errors.append(f"constraint_provenance_row_{index}_invalid")
            continue
        raw_payload = row.get("lineage_payload")
        try:
            payload = (
                load_strict_json(str(raw_payload))
                if isinstance(raw_payload, str)
                else dict(raw_payload)
                if isinstance(raw_payload, Mapping)
                else None
            )
        except StrictStructuredDataError:
            payload = None
        if not isinstance(payload, dict):
            errors.append(f"constraint_provenance_row_{index}_payload_invalid")
            continue
        if str(row.get("lineage_sha256", "")) != _sha256_json(payload):
            errors.append(f"constraint_provenance_row_{index}_lineage_hash_mismatch")
        source_ids = [value for value in str(row.get("source_quote_ids", "")).split("|") if value]
        payload_source_ids = [str(value) for value in payload.get("source_quote_ids", [])]
        if source_ids != payload_source_ids or not source_ids:
            errors.append(f"constraint_provenance_row_{index}_source_quote_ids_mismatch")
        if len(source_ids) != len(set(source_ids)):
            errors.append(f"constraint_provenance_row_{index}_source_quote_ids_duplicate")
        if any(source_id not in quote_ids for source_id in source_ids):
            errors.append(f"constraint_provenance_row_{index}_unknown_quote_id")
        parent_quote_id = str(payload.get("parent_quote_id", ""))
        parent_product = str(row.get("parent_product", ""))
        parent_quote = quote_by_id.get(parent_quote_id)
        if parent_quote is None or str(parent_quote.get("product", "")) != parent_product:
            errors.append(f"constraint_provenance_row_{index}_parent_quote_mismatch")
        if str(payload.get("parent_product", "")) != parent_product:
            errors.append(f"constraint_provenance_row_{index}_parent_product_mismatch")
        row_load_type = str(row.get("load_type", "")).upper()
        if (
            parent_quote is not None
            and str(parent_quote.get("load_type", "")).upper() != row_load_type
        ):
            errors.append(f"constraint_provenance_row_{index}_parent_load_type_mismatch")
        try:
            if abs(float(payload.get("target")) - float(row.get("target"))) > 1e-12:
                errors.append(f"constraint_provenance_row_{index}_target_mismatch")
        except (TypeError, ValueError):
            errors.append(f"constraint_provenance_row_{index}_target_invalid")
        is_residual = bool(row.get("is_residual"))
        expected_formula = (
            "residual_parent_energy_less_known_buckets" if is_residual else "direct_parent_mean"
        )
        payload_formula = (
            "(parent_price*parent_hours-known_bucket_energy)/free_hours"
            if is_residual
            else "direct_parent_mean"
        )
        if str(row.get("lineage_formula", "")) != expected_formula:
            errors.append(f"constraint_provenance_row_{index}_row_formula_mismatch")
        if str(payload.get("formula", "")) != payload_formula:
            errors.append(f"constraint_provenance_row_{index}_payload_formula_mismatch")
        if is_residual != str(row.get("product", "")).endswith("-RESIDUAL"):
            errors.append(f"constraint_provenance_row_{index}_residual_flag_mismatch")
        try:
            row_target = float(row.get("target"))
            free_hours = float(row.get("hours"))
            parent_price = float(parent_quote.get("price_eur_mwh"))
        except (AttributeError, TypeError, ValueError):
            errors.append(f"constraint_provenance_row_{index}_energy_inputs_invalid")
            continue
        if not all(math.isfinite(value) for value in (row_target, free_hours, parent_price)):
            errors.append(f"constraint_provenance_row_{index}_energy_inputs_nonfinite")
            continue
        if free_hours <= 0.0:
            errors.append(f"constraint_provenance_row_{index}_hours_nonpositive")
            continue
        try:
            expected_free_hours = _expected_constraint_row_hours(
                row,
                provenance_by_product=provenance_by_product,
                stack=set(),
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"constraint_provenance_row_{index}_calendar_hours_invalid:{exc}")
            continue
        if abs(free_hours - expected_free_hours) > 1e-9:
            errors.append(f"constraint_provenance_row_{index}_calendar_hours_mismatch")
        if not is_residual:
            if abs(row_target - parent_price) > 1e-12:
                errors.append(f"constraint_provenance_row_{index}_direct_target_quote_mismatch")
            if source_ids != [parent_quote_id]:
                errors.append(f"constraint_provenance_row_{index}_source_quote_closure_mismatch")
            continue
        known_bucket_names = payload.get("known_buckets")
        if not isinstance(known_bucket_names, list) or not known_bucket_names:
            errors.append(f"constraint_provenance_row_{index}_known_buckets_missing")
            continue
        normalized_known_buckets = [str(value) for value in known_bucket_names]
        if len(normalized_known_buckets) != len(set(normalized_known_buckets)):
            errors.append(f"constraint_provenance_row_{index}_known_buckets_duplicate")
        known_energy = 0.0
        known_hours = 0.0
        residual_inputs_valid = True
        parent_periods = _contract_product_period_keys(parent_product)
        covered_periods: set[tuple[int, int]] = set()
        expected_source_ids = [parent_quote_id]
        for known_bucket in known_bucket_names:
            known_row = provenance_by_product.get(str(known_bucket))
            if not isinstance(known_row, Mapping):
                errors.append(
                    f"constraint_provenance_row_{index}_known_bucket_unknown:{known_bucket}"
                )
                residual_inputs_valid = False
                continue
            if str(known_row.get("load_type", "")).upper() != row_load_type:
                errors.append(
                    f"constraint_provenance_row_{index}_known_bucket_load_type_mismatch:{known_bucket}"
                )
                residual_inputs_valid = False
            try:
                bucket_periods = _constraint_row_delivery_periods(
                    known_row,
                    provenance_by_product=provenance_by_product,
                    stack={str(row.get("product", ""))},
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(
                    f"constraint_provenance_row_{index}_known_bucket_periods_invalid:"
                    f"{known_bucket}:{exc}"
                )
                residual_inputs_valid = False
                continue
            if not bucket_periods or not bucket_periods < parent_periods:
                errors.append(
                    f"constraint_provenance_row_{index}_known_bucket_not_strict_child:{known_bucket}"
                )
                residual_inputs_valid = False
            if covered_periods.intersection(bucket_periods):
                errors.append(
                    f"constraint_provenance_row_{index}_known_bucket_overlap:{known_bucket}"
                )
                residual_inputs_valid = False
            covered_periods.update(bucket_periods)
            expected_source_ids.extend(
                value for value in str(known_row.get("source_quote_ids", "")).split("|") if value
            )
            try:
                bucket_target = float(known_row.get("target"))
                bucket_hours = _expected_constraint_row_hours(
                    known_row,
                    provenance_by_product=provenance_by_product,
                    stack=set(),
                )
            except (TypeError, ValueError):
                errors.append(
                    f"constraint_provenance_row_{index}_known_bucket_invalid:{known_bucket}"
                )
                residual_inputs_valid = False
                continue
            if (
                not math.isfinite(bucket_target)
                or not math.isfinite(bucket_hours)
                or bucket_hours <= 0
            ):
                errors.append(
                    f"constraint_provenance_row_{index}_known_bucket_nonfinite:{known_bucket}"
                )
                residual_inputs_valid = False
                continue
            known_energy += bucket_target * bucket_hours
            known_hours += bucket_hours
        if residual_inputs_valid:
            expected_source_ids = list(dict.fromkeys(expected_source_ids))
            if source_ids != expected_source_ids:
                errors.append(f"constraint_provenance_row_{index}_source_quote_closure_mismatch")
            parent_hours = expected_free_hours + known_hours
            recomputed_target = (parent_price * parent_hours - known_energy) / free_hours
            if abs(row_target - recomputed_target) > 1e-10:
                errors.append(
                    f"constraint_provenance_row_{index}_residual_energy_identity_mismatch"
                )
    errors.extend(
        _constraint_partition_errors(
            manifest=manifest,
            hard_quotes=hard_quotes,
            provenance_rows=provenance_rows,
            provenance_by_product=provenance_by_product,
            repricing_tolerance=float(expected_policy["hard_repricing_tolerance_eur_mwh"]),
        )
    )
    binding = {
        "product_hierarchy_policy": expected_policy,
        "hard_quotes": hard_quotes,
        "constraint_provenance_rows": provenance_rows,
    }
    return (binding if not errors else {}), errors


def _expected_constraint_row_hours(
    row: Mapping[str, object],
    *,
    provenance_by_product: Mapping[str, object],
    stack: set[str],
) -> float:
    product = str(row.get("product", ""))
    if not product or product in stack:
        raise ValueError(f"invalid or cyclic constraint product {product!r}")
    parent_product = str(row.get("parent_product", ""))
    load_type = str(row.get("load_type", "BASE")).upper()
    parent_hours = _contract_product_hours(parent_product, load_type=load_type)
    if not bool(row.get("is_residual")):
        return parent_hours
    raw_payload = row.get("lineage_payload")
    payload = load_strict_json(raw_payload) if isinstance(raw_payload, str) else dict(raw_payload)
    known_buckets = payload.get("known_buckets")
    if not isinstance(known_buckets, list) or not known_buckets:
        raise ValueError(f"residual product {product!r} has no known buckets")
    next_stack = {*stack, product}
    known_hours = 0.0
    for known_bucket in known_buckets:
        known_row = provenance_by_product.get(str(known_bucket))
        if not isinstance(known_row, Mapping):
            raise KeyError(str(known_bucket))
        known_hours += _expected_constraint_row_hours(
            known_row,
            provenance_by_product=provenance_by_product,
            stack=next_stack,
        )
    free_hours = parent_hours - known_hours
    if not math.isfinite(free_hours) or free_hours <= 0.0:
        raise ValueError(f"residual product {product!r} has invalid free hours")
    return free_hours


def _constraint_row_delivery_periods(
    row: Mapping[str, object],
    *,
    provenance_by_product: Mapping[str, object],
    stack: set[str],
) -> set[tuple[int, int]]:
    product = str(row.get("product", ""))
    if not product or product in stack:
        raise ValueError(f"invalid or cyclic constraint product {product!r}")
    parent_periods = _contract_product_period_keys(str(row.get("parent_product", "")))
    if not bool(row.get("is_residual")):
        return parent_periods
    raw_payload = row.get("lineage_payload")
    payload = load_strict_json(raw_payload) if isinstance(raw_payload, str) else dict(raw_payload)
    known_buckets = payload.get("known_buckets")
    if not isinstance(known_buckets, list) or not known_buckets:
        raise ValueError(f"residual product {product!r} has no known buckets")
    covered: set[tuple[int, int]] = set()
    next_stack = {*stack, product}
    for known_bucket in known_buckets:
        known_row = provenance_by_product.get(str(known_bucket))
        if not isinstance(known_row, Mapping):
            raise KeyError(str(known_bucket))
        bucket_periods = _constraint_row_delivery_periods(
            known_row,
            provenance_by_product=provenance_by_product,
            stack=next_stack,
        )
        if not bucket_periods <= parent_periods:
            raise ValueError(f"known bucket {known_bucket!r} escapes parent delivery period")
        if covered.intersection(bucket_periods):
            raise ValueError(f"known bucket {known_bucket!r} overlaps another known bucket")
        covered.update(bucket_periods)
    residual_periods = parent_periods.difference(covered)
    if not residual_periods:
        raise ValueError(f"residual product {product!r} has no free delivery period")
    return residual_periods


def _contract_product_period_keys(product: str) -> set[tuple[int, int]]:
    return {(int(period.year), int(period.month)) for period in product_periods(product)}


def _constraint_partition_errors(
    *,
    manifest: Mapping[str, object],
    hard_quotes: list[object],
    provenance_rows: list[object],
    provenance_by_product: Mapping[str, object],
    repricing_tolerance: float,
) -> list[str]:
    errors: list[str] = []
    raw_delivery_months = manifest.get("delivery_months")
    if not isinstance(raw_delivery_months, list) or not raw_delivery_months:
        return ["delivery_months_missing"]
    try:
        canonical_months = [str(pd.Period(str(value), freq="M")) for value in raw_delivery_months]
    except (TypeError, ValueError) as exc:
        return [f"delivery_months_invalid:{exc}"]
    if canonical_months != [str(value) for value in raw_delivery_months]:
        errors.append("delivery_months_not_canonical")
    if canonical_months != sorted(set(canonical_months)):
        errors.append("delivery_months_not_unique_sorted")
    delivery_periods = {(int(month[:4]), int(month[5:7])) for month in canonical_months}

    row_partitions: list[tuple[int, Mapping[str, object], set[tuple[int, int]]]] = []
    occupied: dict[tuple[int, int], int] = {}
    for index, row in enumerate(provenance_rows):
        if not isinstance(row, Mapping):
            continue
        try:
            periods = _constraint_row_delivery_periods(
                row,
                provenance_by_product=provenance_by_product,
                stack=set(),
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"constraint_provenance_row_{index}_partition_invalid:{exc}")
            continue
        if not periods <= delivery_periods:
            errors.append(f"constraint_provenance_row_{index}_escapes_delivery_grid")
        for period in sorted(periods):
            previous = occupied.get(period)
            if previous is not None:
                errors.append(
                    f"constraint_provenance_global_overlap:{period[0]:04d}-{period[1]:02d}:"
                    f"rows={previous},{index}"
                )
            else:
                occupied[period] = index
        row_partitions.append((index, row, periods))

    for quote in hard_quotes:
        if not isinstance(quote, Mapping):
            continue
        quote_id = str(quote.get("quote_id", ""))
        load_type = str(quote.get("load_type", "")).upper()
        try:
            quote_periods = _contract_product_period_keys(str(quote.get("product", "")))
            quote_price = float(quote.get("price_eur_mwh"))
        except (TypeError, ValueError) as exc:
            errors.append(f"hard_quote_partition_invalid:{quote_id}:{exc}")
            continue
        in_grid = quote_periods.intersection(delivery_periods)
        if not in_grid:
            continue
        if in_grid != quote_periods:
            errors.append(f"hard_quote_partially_intersects_delivery_grid:{quote_id}")
            continue
        covering = [
            (row, periods)
            for _, row, periods in row_partitions
            if str(row.get("load_type", "")).upper() == load_type
            and periods
            and periods <= quote_periods
        ]
        covered = set().union(*(periods for _, periods in covering)) if covering else set()
        if covered != quote_periods:
            missing = sorted(quote_periods.difference(covered))
            errors.append(f"hard_quote_partition_incomplete:{quote_id}:missing={missing}")
            continue
        quote_hours = _contract_period_keys_hours(quote_periods, load_type=load_type)
        delivered_energy = 0.0
        try:
            for row, periods in covering:
                delivered_energy += float(row.get("target")) * _contract_period_keys_hours(
                    periods,
                    load_type=load_type,
                )
        except (TypeError, ValueError) as exc:
            errors.append(f"hard_quote_partition_target_invalid:{quote_id}:{exc}")
            continue
        repricing_error = abs((delivered_energy / quote_hours) - quote_price)
        if not math.isfinite(repricing_error) or repricing_error > repricing_tolerance:
            errors.append(
                f"hard_quote_partition_repricing_mismatch:{quote_id}:error={repricing_error:.12g}"
            )
    return errors


def _contract_period_keys_hours(
    periods: set[tuple[int, int]],
    *,
    load_type: str,
) -> float:
    hours = 0.0
    for year, month in sorted(periods):
        total, peak, offpeak = count_hours(year, month, month, country="CH")
        if load_type == "BASE":
            hours += total
        elif load_type == "PEAK":
            hours += peak
        elif load_type == "OFFPEAK":
            hours += offpeak
        else:
            raise ValueError(f"unsupported constraint load type {load_type!r}")
    if hours <= 0.0:
        raise ValueError("constraint partition has no delivery hours")
    return float(hours)


def _contract_product_hours(product: str, *, load_type: str) -> float:
    periods = product_periods(product)
    hours = 0
    for period in periods:
        total, peak, offpeak = count_hours(
            int(period.year),
            int(period.month),
            int(period.month),
            country="CH",
        )
        if load_type == "BASE":
            hours += total
        elif load_type == "PEAK":
            hours += peak
        elif load_type == "OFFPEAK":
            hours += offpeak
        else:
            raise ValueError(f"unsupported constraint load type {load_type!r}")
    return float(hours)


def _allocate_product_replay_root() -> Path:
    scratch_value = os.environ.get("TEMP") or os.environ.get("TMP")
    if scratch_value is None:
        raise ValueError("product replay requires an explicit scratch root")
    scratch_parent = assert_absolute_path_has_no_links(Path(scratch_value))
    if not scratch_parent.is_dir():
        raise ValueError("product replay scratch root is unavailable")
    for _ in range(32):
        selected = scratch_parent / f"pfc-lt-product-replay-{uuid.uuid4().hex}"
        try:
            selected.mkdir()
        except FileExistsError:
            continue
        return selected
    raise ValueError("product replay scratch root cannot be allocated")


def _replay_product_normalization_audit(
    *,
    declared_summary: Mapping[str, object],
    delivered_csv: CapturedArtifact,
    delivered_forwards: CapturedArtifact,
    monthly_candidate_manifest: CapturedArtifact | None,
    source_hierarchy_policy: CapturedArtifact | None,
    forward_workbook: CapturedArtifact,
    bundle_path: Path,
    expected_evidence: Mapping[str, object],
) -> tuple[Mapping[str, object] | None, list[str]]:
    """Recompute the delivered-product audit from hash-bound candidate inputs."""

    errors: list[str] = []
    expected = expected_evidence.get("expected")
    expected = expected if isinstance(expected, Mapping) else {}
    market = str(expected.get("market", "")).upper()
    forward_date = str(expected.get("forward_snapshot_date", ""))
    if market != "CH":
        errors.append("product_replay_expected_market_not_ch")
    if not forward_date:
        errors.append("product_replay_expected_forward_snapshot_date_missing")
    if str(declared_summary.get("price_column", "")) != DEFAULT_PRICE_COLUMN:
        errors.append("product_replay_price_column_not_sanctioned")
    try:
        declared_tolerance = float(declared_summary.get("hard_tolerance_eur_mwh"))
    except (TypeError, ValueError):
        errors.append("product_replay_hard_tolerance_invalid")
    else:
        if not math.isclose(declared_tolerance, 1e-6, rel_tol=0.0, abs_tol=0.0):
            errors.append("product_replay_hard_tolerance_not_sanctioned")
    if errors:
        return None, errors

    try:
        replay_root = _allocate_product_replay_root()
        try:
            replay_csv = _materialize_captured_bundle_artifact(
                delivered_csv,
                bundle_path=bundle_path,
                replay_root=replay_root,
            )
            replay_forwards = _materialize_captured_bundle_artifact(
                delivered_forwards,
                bundle_path=bundle_path,
                replay_root=replay_root,
            )
            replay_monthly = (
                _materialize_captured_bundle_artifact(
                    monthly_candidate_manifest,
                    bundle_path=bundle_path,
                    replay_root=replay_root,
                )
                if monthly_candidate_manifest is not None
                else None
            )
            replay_policy = (
                _materialize_captured_bundle_artifact(
                    source_hierarchy_policy,
                    bundle_path=bundle_path,
                    replay_root=replay_root,
                )
                if source_hierarchy_policy is not None
                else None
            )
            _materialize_captured_bundle_artifact(
                forward_workbook,
                bundle_path=bundle_path,
                replay_root=replay_root,
            )
            _, replayed = run_audit(
                csv_path=replay_csv,
                forwards_path=replay_forwards,
                market="CH",
                price_column=DEFAULT_PRICE_COLUMN,
                required_forward_date=forward_date,
                hard_tolerance=1e-6,
                peak_country="CH",
                command_argv=(
                    list(declared_summary["command_argv"])
                    if isinstance(declared_summary.get("command_argv"), list)
                    else None
                ),
                required_load_types=("BASE", "PEAK"),
                source_hierarchy_policy_path=replay_policy,
                monthly_candidate_manifest_path=replay_monthly,
            )
        finally:
            shutil.rmtree(replay_root)
    except (OSError, TypeError, ValueError) as exc:
        return None, [f"product_replay_failed:{type(exc).__name__}:{exc}"]

    declared_core = _product_replay_comparable_summary(declared_summary)
    replayed_core = _product_replay_comparable_summary(replayed)
    if declared_core != replayed_core:
        errors.append(
            "product_replay_summary_mismatch:"
            f"declared={_sha256_json(declared_core)}:replayed={_sha256_json(replayed_core)}"
        )
    return replayed, errors


def _materialize_captured_bundle_artifact(
    artifact: CapturedArtifact,
    *,
    bundle_path: Path,
    replay_root: Path,
) -> Path:
    if artifact.raw is None:
        raise ValueError(f"captured bytes are unavailable: {artifact.path}")
    try:
        relative = artifact.path.resolve().relative_to(bundle_path.resolve())
    except ValueError as exc:
        raise ValueError(
            f"captured replay artifact is outside candidate bundle: {artifact.path}"
        ) from exc
    destination = (replay_root / relative).resolve()
    try:
        destination.relative_to(replay_root)
    except ValueError as exc:
        raise ValueError(f"captured replay artifact escapes replay root: {relative}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(artifact.raw)
    if _sha256_file(destination) != artifact.sha256:
        raise ValueError(f"captured replay artifact hash mismatch: {relative.as_posix()}")
    return destination


def _product_replay_comparable_summary(summary: Mapping[str, object]) -> dict[str, object]:
    excluded = {"audit_script", "command_argv", "input_csv", "forwards_path", "note"}
    comparable = {
        str(key): _json_clean(value) for key, value in summary.items() if str(key) not in excluded
    }
    policy = comparable.get("source_hierarchy_policy")
    if isinstance(policy, Mapping):
        comparable["source_hierarchy_policy"] = {
            str(key): _json_clean(value) for key, value in policy.items() if str(key) != "path"
        }
    comparable.pop("monthly_candidate_manifest", None)
    return comparable


def _delivered_product_normalization_row(
    summary: Mapping[str, object] | None,
    *,
    product_normalization_summary_path: Path | None,
    product_normalization_summary_sha256: str | None,
    product_normalization_replay: Mapping[str, object] | None,
    product_normalization_replay_errors: list[str] | None,
    monthly_candidate_manifest: Mapping[str, object] | None,
    monthly_candidate_manifest_path: Path | None,
    monthly_candidate_manifest_sha256: str | None,
    expected_evidence: Mapping[str, object],
) -> dict[str, object]:
    if summary is None:
        return _governance_gate_row(
            gate_id="delivered_product_normalization_audit",
            status="CRITICAL",
            severity="P0",
            product="delivered_ch_product_normalization",
            parent_block_id="delivered_product_normalization",
            metric_name="delivered_product_normalization_not_approved",
            metric_value=1.0,
            threshold_source="delivered_product_normalization_summary",
            evidence="missing delivered product normalization summary artifact",
            remediation_hint="Run scripts/audit_ch_product_normalization.py on the delivered CSV and pass its summary JSON.",
        )

    errors: list[str] = []
    errors.extend(str(error) for error in (product_normalization_replay_errors or []))
    if product_normalization_replay is None and not product_normalization_replay_errors:
        errors.append("product_normalization_replay_missing")
    evidence_errors = list(expected_evidence.get("errors", []))
    errors.extend(str(error) for error in evidence_errors)
    expected = expected_evidence.get("expected")
    expected = expected if isinstance(expected, Mapping) else {}
    actual_summary_hash = str(product_normalization_summary_sha256 or "")
    _compare_expected(errors, expected, "summary_sha256", actual_summary_hash)
    _compare_expected(errors, expected, "input_csv_sha256", summary.get("input_csv_sha256"))
    _compare_expected(errors, expected, "forwards_sha256", summary.get("forwards_sha256"))
    _compare_expected(
        errors, expected, "forward_snapshot_date", summary.get("forward_snapshot_date")
    )
    forward_binding = expected_evidence.get("forward_binding")
    forward_binding = forward_binding if isinstance(forward_binding, Mapping) else {}
    for key in (
        "forward_snapshot_id",
        "forward_observation_id",
        "forward_source_sha256",
        "forward_quote_set_sha256",
        "forward_quote_lineage_sha256",
        "forward_policy_sha256",
        "forward_eligibility_id",
    ):
        _compare_value(errors, forward_binding.get(key), summary.get(key), key)
    solver_binding = expected_evidence.get("solver_binding")
    solver_binding = solver_binding if isinstance(solver_binding, Mapping) else {}
    for key in ("monthly_solution_hash", "active_constraints_hash", "active_config_hash"):
        _compare_value(errors, solver_binding.get(key), summary.get(key), key)
    candidate_path_value = summary.get("monthly_candidate_manifest")
    candidate_hash_value = summary.get("monthly_candidate_manifest_sha256")
    if not candidate_path_value or not candidate_hash_value:
        errors.append("monthly_candidate_manifest_binding_missing")
    else:
        candidate_path = Path(str(candidate_path_value))
        if not candidate_path.is_absolute() and product_normalization_summary_path is not None:
            candidate_path = product_normalization_summary_path.resolve().parent / candidate_path
        candidate_path = candidate_path.resolve()
        if (
            monthly_candidate_manifest_path is None
            or candidate_path != monthly_candidate_manifest_path.resolve()
        ):
            errors.append("monthly_candidate_manifest_missing")
        elif str(monthly_candidate_manifest_sha256 or "") != str(candidate_hash_value):
            errors.append("monthly_candidate_manifest_sha256_mismatch")
        else:
            candidate_manifest = monthly_candidate_manifest
            if not isinstance(candidate_manifest, Mapping):
                errors.append("monthly_candidate_manifest_invalid")
                candidate_manifest = {}
            candidate_snapshot = candidate_manifest.get("forward_snapshot")
            candidate_eligibility = candidate_manifest.get("forward_eligibility")
            if not isinstance(candidate_snapshot, Mapping) or not isinstance(
                candidate_eligibility, Mapping
            ):
                errors.append("monthly_candidate_forward_binding_missing")
            else:
                try:
                    candidate_verified = verify_forward_manifest_binding(
                        candidate_snapshot,
                        candidate_eligibility,
                        expected_max_age_business_days=PRODUCTION_EEX_MAX_AGE_BUSINESS_DAYS,
                        verify_source=True,
                    )
                except (TypeError, ValueError) as exc:
                    errors.append(f"monthly_candidate_forward_binding_invalid:{exc}")
                else:
                    candidate_forward_binding = {
                        "forward_snapshot_id": candidate_verified["snapshot_id"],
                        "forward_observation_id": candidate_verified["observation_id"],
                        "forward_source_sha256": candidate_verified["source_sha256"],
                        "forward_quote_set_sha256": candidate_verified["quote_set_sha256"],
                        "forward_quote_lineage_sha256": candidate_verified["quote_lineage_sha256"],
                        "forward_policy_sha256": candidate_verified["policy_sha256"],
                        "forward_eligibility_id": candidate_verified["eligibility_id"],
                    }
                    if candidate_forward_binding != dict(forward_binding):
                        errors.append("monthly_candidate_forward_binding_mismatch")
            for key, expected_value in solver_binding.items():
                actual_value = (
                    _active_config_hash(candidate_manifest)
                    if key == "active_config_hash"
                    else candidate_manifest.get(key)
                )
                if str(actual_value or "") != str(expected_value or ""):
                    errors.append(f"monthly_candidate_{key}_mismatch")
    _compare_expected(errors, expected, "market", str(summary.get("market", "")).upper())

    if str(summary.get("schema_version", "")) != "ch_product_normalization_audit.v3":
        errors.append("schema_version_mismatch")
    if summary.get("monthly_candidate_binding_status") != "BOUND":
        errors.append("monthly_candidate_binding_not_bound")
    for key in (
        "all_gates_pass",
        "covered_hard_gates_pass",
        "supported_hard_gates_no_critical",
    ):
        if key not in summary:
            errors.append(f"{key}_missing")
        elif summary.get(key) is not True:
            errors.append(f"{key}_false")
    for key in (
        "critical_count",
        "unsupported_count",
        "blocking_quote_conflict_count",
        "delivered_curve_drift_count",
    ):
        value = _int_or_none(summary.get(key))
        if value is None:
            errors.append(f"{key}_missing_or_invalid")
        elif value != 0:
            errors.append(f"{key}_nonzero")
    gate_id_counts = summary.get("gate_id_counts")
    if not isinstance(gate_id_counts, Mapping):
        errors.append("gate_id_counts_missing")
        gate_id_counts = {}
    for gate_id in REQUIRED_PRODUCT_GATE_FAMILIES:
        if (
            _int_or_none(gate_id_counts.get(gate_id)) is None
            or int(gate_id_counts.get(gate_id, 0)) <= 0
        ):
            errors.append(f"{gate_id}_missing")

    policy = summary.get("source_hierarchy_policy")
    quote_conflict_count = _int_or_none(summary.get("quote_conflict_count"))
    if quote_conflict_count is None:
        errors.append("quote_conflict_count_missing_or_invalid")
        quote_conflict_count = 0
    if quote_conflict_count > 0:
        if not isinstance(policy, Mapping):
            errors.append("source_hierarchy_policy_missing")
        elif policy.get("status") != "ACCEPTED_PRODUCTION_APPROVED":
            errors.append("source_hierarchy_policy_not_accepted_production_approved")
        elif policy.get("production_approved") is not True:
            errors.append("source_hierarchy_policy_not_production_approved")
        elif _int_or_none(policy.get("blocking_quote_conflict_count")) != 0:
            errors.append("source_hierarchy_policy_blocking_conflicts_nonzero")
        else:
            if _int_or_none(policy.get("accepted_quote_conflict_count")) != quote_conflict_count:
                errors.append("source_hierarchy_policy_accepted_conflict_count_mismatch")
            _compare_value(
                errors,
                policy.get("input_csv_sha256"),
                summary.get("input_csv_sha256"),
                "policy_input_csv_sha256",
            )
            _compare_value(
                errors,
                policy.get("forwards_sha256"),
                summary.get("forwards_sha256"),
                "policy_forwards_sha256",
            )
            _compare_value(
                errors,
                policy.get("quote_conflict_identity_hash"),
                summary.get("quote_conflict_identity_hash"),
                "policy_quote_conflict_identity_hash",
            )
            _compare_expected(
                errors,
                expected,
                "quote_conflict_identity_hash",
                summary.get("quote_conflict_identity_hash"),
            )
            _compare_expected(
                errors,
                expected,
                "source_hierarchy_policy_sha256",
                policy.get("sha256"),
            )

    status = "PASS" if not errors else "CRITICAL"
    severity = "INFO" if not errors else "P0"
    metric_value = 0.0 if not errors else 1.0
    path_text = (
        str(product_normalization_summary_path)
        if product_normalization_summary_path is not None
        else ""
    )
    hash_text = (
        _sha256_file(product_normalization_summary_path)
        if product_normalization_summary_path is not None
        and product_normalization_summary_path.exists()
        else ""
    )
    evidence = (
        "delivered product normalization audit passed; "
        f"summary={path_text}; sha256={hash_text}; "
        f"status_counts={summary.get('status_counts', {})}"
        if not errors
        else (
            "delivered product normalization audit is not production-ready; "
            f"errors={errors}; summary={path_text}; sha256={hash_text}"
        )
    )
    return _governance_gate_row(
        gate_id="delivered_product_normalization_audit",
        status=status,
        severity=severity,
        product="delivered_ch_product_normalization",
        parent_block_id="delivered_product_normalization",
        metric_name="delivered_product_normalization_not_approved",
        metric_value=metric_value,
        threshold_source="delivered_product_normalization_summary",
        evidence=evidence,
        remediation_hint="Fix delivered BASE/PEAK/OFFPEAK repricing or provide a production-approved source hierarchy policy for quote conflicts.",
    )


def _compare_expected(
    errors: list[str],
    expected: Mapping[str, object],
    key: str,
    actual: object,
) -> None:
    expected_value = expected.get(key)
    if expected_value is None or expected_value == "":
        errors.append(f"expected_{key}_missing")
        return
    _compare_value(errors, expected_value, actual, key)


def _compare_value(errors: list[str], expected: object, actual: object, label: str) -> None:
    if actual is None or actual == "":
        errors.append(f"{label}_missing")
    elif str(actual) != str(expected):
        errors.append(f"{label}_mismatch")


def _selected_config_production_approval_row(
    selected_config: Mapping[str, object],
) -> dict[str, object]:
    approved = selected_config.get("production_approved") is True
    assembled_parity = (
        selected_config.get("schema_version") == "monthly_curve_selected_config_run_parity.v1"
    )
    promotion_approval = selected_config.get("production_promotion_approved")
    promotion_approved = promotion_approval is True or assembled_parity
    selection_status = str(selected_config.get("selection_status", ""))
    status_is_prod = selection_status == "PRODUCTION_APPROVED" and (
        not assembled_parity or selected_config.get("status") == "PASS"
    )
    if approved and promotion_approved and status_is_prod:
        status = "PASS"
        severity = "INFO"
        metric_value = 0.0
        evidence = (
            "selected config is production approved; "
            f"production_approved={selected_config.get('production_approved')!r}, "
            f"production_promotion_approved={promotion_approval!r}, "
            f"selection_status={selection_status}"
        )
    else:
        status = "CRITICAL"
        severity = "P0"
        metric_value = 1.0
        evidence = (
            "selected config is not production-approved; "
            f"production_approved={selected_config.get('production_approved')!r}, "
            f"production_promotion_approved={promotion_approval!r}, "
            f"selection_status={selection_status!r}"
        )
    return _governance_gate_row(
        gate_id="selected_config_production_approval",
        status=status,
        severity=severity,
        product="monthly_curve_selected_config",
        parent_block_id="selected_config_approval",
        metric_name="selected_config_not_production_approved",
        metric_value=metric_value,
        threshold_source="selected_config_artifact",
        evidence=evidence,
        remediation_hint="Use a production-approved selected config artifact before promotion.",
    )


def _lt_probabilistic_output_governance_row(
    export_manifest: Mapping[str, object],
) -> dict[str, object]:
    errors: list[str] = []
    expected = {
        "delivered_curve_kind": "deterministic_price_shape",
        "probabilistic_status": "DETERMINISTIC_ONLY",
        "interval_columns": [],
        "intervals_permitted_for_pricing_or_risk": False,
        "source_price_column": "price_shape",
        "scenario_weights": {"price_shape": 1.0},
        "producer_contract": "pfc.pipeline.candidate_hourly_export.v1",
    }
    for key, value in expected.items():
        if export_manifest.get(key) != value:
            errors.append(f"{key}_mismatch")
    columns = export_manifest.get("columns")
    if not isinstance(columns, list):
        errors.append("columns_missing")
    elif any(str(column).lower() in {"p10", "p90"} for column in columns):
        errors.append("interval_columns_exported")
    if export_manifest.get("finite_price_values") is not True:
        errors.append("finite_price_values_not_true")
    status = "PASS" if not errors else "CRITICAL"
    return _governance_gate_row(
        gate_id="lt_probabilistic_output_governance",
        status=status,
        severity="INFO" if not errors else "P0",
        product="lt_candidate_probabilistic_output",
        parent_block_id="probabilistic_output_governance",
        metric_name="uncalibrated_intervals_exposed",
        metric_value=0.0 if not errors else 1.0,
        threshold_source="sealed_candidate_hourly_export_manifest",
        evidence=(
            "sealed candidate is deterministic-only and exports no interval columns"
            if not errors
            else f"probabilistic output governance failed; errors={errors}"
        ),
        remediation_hint=(
            "Remove p10/p90 from candidate and industrial exports, or provide a "
            "separately approved CH rolling-origin probabilistic evidence contract."
        ),
    )


def _ompex_training_selection_exclusion_row(
    selected_config: Mapping[str, object],
) -> dict[str, object]:
    evidence = selected_config.get("ompex_training_selection_exclusion")
    errors: list[str] = []
    if not isinstance(evidence, Mapping):
        errors.append("evidence_missing")
        evidence = {}
    if evidence.get("schema_version") != "ompex_training_selection_exclusion.v1":
        errors.append("schema_version_mismatch")
    if evidence.get("status") != "PASS":
        errors.append("status_not_pass")
    if evidence.get("matches") != []:
        errors.append("prohibited_source_match")
    required_uses = {"MODEL_INPUT", "TARGET", "LOSS", "SELECTION", "PROMOTION_GATE"}
    uses = evidence.get("prohibited_uses")
    if not isinstance(uses, list) or set(str(value) for value in uses) != required_uses:
        errors.append("prohibited_uses_mismatch")
    inspected = evidence.get("inspected_input_sources")
    if not isinstance(inspected, list) or not inspected:
        errors.append("input_source_replay_missing")
    if str(evidence.get("inspected_selected_config_sha256", "")) != str(
        selected_config.get("config_hash", "")
    ):
        errors.append("selected_config_hash_mismatch")
    status = "PASS" if not errors else "CRITICAL"
    return _governance_gate_row(
        gate_id="ompex_training_selection_exclusion",
        status=status,
        severity="INFO" if not errors else "P0",
        product="lt_model_input_and_selection_lineage",
        parent_block_id="benchmark_exclusion_governance",
        metric_name="ompex_used_for_model_or_selection",
        metric_value=0.0 if not errors else 1.0,
        threshold_source="sealed_candidate_input_and_selection_lineage",
        evidence=(
            "OMPEX/HFC benchmark is absent from sealed model inputs and lambda selection"
            if not errors
            else f"OMPEX/HFC exclusion evidence failed; errors={errors}"
        ),
        remediation_hint=(
            "Remove OMPEX/HFC from model inputs, targets, losses, selection and promotion gates; "
            "retain it only as an external post-model benchmark."
        ),
    )


def _selected_config_manifest_parity_row(
    *,
    selected_config: Mapping[str, object],
    production_manifest: Mapping[str, object],
    export_manifest: Mapping[str, object],
    active_config_hash: str,
    selected_config_hash: str,
    production_manifest_path: Path | None,
    production_manifest_sha256: str | None,
    selected_config_path: Path | None,
) -> dict[str, object]:
    missing: list[str] = []
    mismatches: list[str] = []
    assembled_parity = (
        selected_config.get("schema_version") == "monthly_curve_selected_config_run_parity.v1"
    )

    expected = {
        "config_hash": active_config_hash,
        (
            "active_config_hash"
            if assembled_parity
            else "active_config_hash_from_candidate_manifest"
        ): active_config_hash,
        "monthly_solution_hash": _required_manifest_hash(
            production_manifest,
            "monthly_solution_hash",
            role="production",
        ),
        "active_constraints_hash": _required_manifest_hash(
            production_manifest,
            "active_constraints_hash",
            role="production",
        ),
    }
    for key, expected_value in expected.items():
        value = selected_config.get(key)
        if not value:
            missing.append(key)
        elif str(value) != str(expected_value):
            mismatches.append(f"{key}: selected={value} expected={expected_value}")

    if str(export_manifest.get("monthly_solution_hash", "")) != str(
        expected["monthly_solution_hash"]
    ):
        mismatches.append("export monthly_solution_hash does not match production")
    if str(export_manifest.get("active_constraints_hash", "")) != str(
        expected["active_constraints_hash"]
    ):
        mismatches.append("export active_constraints_hash does not match production")
    if selected_config_hash != active_config_hash:
        mismatches.append(
            f"selected_config_hash={selected_config_hash} active_config_hash={active_config_hash}"
        )
    if (
        not assembled_parity
        and str(selected_config.get("schema_version", "")) != "monthly_curve_selected_config.v1"
    ):
        mismatches.append("schema_version is not monthly_curve_selected_config.v1")
    if assembled_parity:
        parents = selected_config.get("parents")
        production_parent = (
            parents.get("production_manifest") if isinstance(parents, Mapping) else None
        )
        if not isinstance(production_parent, Mapping):
            missing.append("parents.production_manifest")
        elif production_parent.get("sha256") != str(production_manifest_sha256 or ""):
            mismatches.append("selected parity production parent hash mismatch")
        if selected_config.get("status") != "PASS":
            mismatches.append("selected parity status is not PASS")
        candidate_manifest = None
        candidate_manifest_sha256 = None
    else:
        candidate_manifest = selected_config.get("candidate_manifest")
        candidate_manifest_sha256 = selected_config.get("candidate_manifest_sha256")
    if not assembled_parity and not candidate_manifest:
        missing.append("candidate_manifest")
    if not assembled_parity and not candidate_manifest_sha256:
        missing.append("candidate_manifest_sha256")
    if candidate_manifest and candidate_manifest_sha256:
        candidate_path = Path(str(candidate_manifest))
        if not candidate_path.is_absolute() and selected_config_path is not None:
            candidate_path = selected_config_path.parent / candidate_path
        candidate_path = candidate_path.resolve()
        if not candidate_path.is_file():
            mismatches.append(f"candidate_manifest does not exist: {candidate_path}")
        else:
            actual_candidate_hash = str(production_manifest_sha256 or "")
            if str(candidate_manifest_sha256) != actual_candidate_hash:
                mismatches.append("candidate_manifest_sha256 does not match candidate bytes")
            if production_manifest_path is not None:
                production_path = production_manifest_path.resolve()
                if candidate_path != production_path:
                    mismatches.append(
                        "candidate_manifest path does not match production manifest path"
                    )
                if not actual_candidate_hash:
                    mismatches.append("production manifest captured hash is missing")

    if missing or mismatches:
        status = "CRITICAL"
        severity = "P0"
        metric_value = 1.0
        evidence = f"missing={missing}; mismatches={mismatches}"
    else:
        status = "PASS"
        severity = "INFO"
        metric_value = 0.0
        evidence = (
            "selected config hashes match production/export active config, "
            "monthly solution and active constraints"
        )
    return _governance_gate_row(
        gate_id="selected_config_manifest_parity",
        status=status,
        severity=severity,
        product="monthly_curve_selected_config",
        parent_block_id="selected_prod_export_triad",
        metric_name="selected_manifest_hash_mismatch",
        metric_value=metric_value,
        threshold_source="selected_config_prod_export_manifest_triad",
        evidence=evidence,
        remediation_hint="Regenerate or approve the selected config artifact from the same production/export manifest triad.",
    )


def _governance_gate_row(
    *,
    gate_id: str,
    status: str,
    severity: str,
    product: str,
    parent_block_id: str,
    metric_name: str,
    metric_value: float,
    threshold_source: str,
    evidence: str,
    remediation_hint: str,
) -> dict[str, object]:
    return {
        "gate_id": gate_id,
        "status": status,
        "severity": severity,
        "year": 0,
        "month": None,
        "product": product,
        "parent_block_id": parent_block_id,
        "parent_block_type": "governance",
        "parent_hours": float("nan"),
        "parent_mean": float("nan"),
        "month_price": float("nan"),
        "month_deviation": float("nan"),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "threshold_warning": 0.0,
        "threshold_critical": 0.0,
        "threshold_source": threshold_source,
        "n_history": float("nan"),
        "n_neighbors": float("nan"),
        "evidence": evidence,
        "remediation_hint": remediation_hint,
    }


def _active_config_hash(manifest: Mapping[str, object]) -> str:
    payload = _active_config_payload(manifest)
    computed = config_hash(payload)
    explicit = manifest.get("active_config_hash")
    if explicit and str(explicit) != computed:
        raise ValueError("production manifest active_config_hash does not match solver_config")
    return computed


def _active_config_payload(manifest: Mapping[str, object]) -> dict[str, object]:
    solver_config = manifest.get("solver_config") or manifest.get("config")
    if not isinstance(solver_config, Mapping):
        raise ValueError("production manifest missing active_config_hash or solver_config/config")
    monthly_config = monthly_curve_config_from_settings(solver_config)
    payload = dict(monthly_config.__dict__)
    payload.update(
        {
            "markets": sorted(str(m).upper() for m in solver_config.get("markets", [])),
            "history_lookback_years": solver_config.get("history_lookback_years"),
            "min_structural_snapshots": solver_config.get("min_structural_snapshots"),
            "allow_template_structural_fallback": bool(
                solver_config.get("allow_template_structural_fallback", False)
            ),
            "structural_amplitude_eur_mwh": float(
                solver_config.get("structural_amplitude_eur_mwh", 110.0)
            ),
            "panel_weight": float(solver_config.get("panel_weight", 1.0)),
            "history_weight": float(solver_config.get("history_weight", 0.5)),
            "structural_weight": float(solver_config.get("structural_weight", 1.0)),
        }
    )
    return payload


def _required_manifest_hash(manifest: Mapping[str, object], key: str, *, role: str) -> str:
    value = manifest.get(key)
    if not value:
        raise ValueError(f"{role} manifest missing required {key}")
    return str(value)


def _promotion_manifest(
    *,
    audit_manifest: Mapping[str, object],
    production_manifest: Mapping[str, object],
    export_manifest: Mapping[str, object],
    selected_config: Mapping[str, object],
    product_normalization_summary: Mapping[str, object] | None,
    product_normalization_summary_path: Path | None,
    product_normalization_summary_sha256: str | None,
    governance_gates: pd.DataFrame,
) -> dict[str, object]:
    manifest = dict(audit_manifest)
    manifest["promotion_evidence_source"] = "prod_export_manifests"
    manifest["production_monthly_solution_hash"] = production_manifest.get(
        "monthly_solution_hash", ""
    )
    manifest["export_monthly_solution_hash"] = export_manifest.get("monthly_solution_hash", "")
    manifest["production_active_constraints_hash"] = production_manifest.get(
        "active_constraints_hash", ""
    )
    manifest["export_active_constraints_hash"] = export_manifest.get("active_constraints_hash", "")
    manifest["active_config_hash"] = _active_config_hash(production_manifest)
    manifest["selected_config_hash"] = _selected_config_hash(
        selected_config, Path("<selected_config_artifact>")
    )
    manifest["selected_config_production_approved"] = selected_config.get(
        "production_approved", False
    )
    manifest["selected_config_production_promotion_approved"] = selected_config.get(
        "production_promotion_approved",
        False,
    )
    manifest["selected_config_selection_status"] = selected_config.get("selection_status", "")
    manifest["selected_config_monthly_solution_hash"] = selected_config.get(
        "monthly_solution_hash", ""
    )
    manifest["selected_config_active_constraints_hash"] = selected_config.get(
        "active_constraints_hash", ""
    )
    manifest["delivered_product_normalization_summary"] = (
        str(product_normalization_summary_path)
        if product_normalization_summary_path is not None
        else ""
    )
    manifest["delivered_product_normalization_summary_sha256"] = str(
        product_normalization_summary_sha256 or ""
    )
    manifest["delivered_product_normalization_all_gates_pass"] = dict(
        product_normalization_summary or {}
    ).get("all_gates_pass", False)
    manifest["governance_gate_summary"] = governance_gates["status"].value_counts().to_dict()
    return manifest


def _int_or_none(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _promotion_receipt(
    *,
    run_id: str,
    approved: bool,
    decision_summary: Mapping[str, object],
    valuation_timestamp: pd.Timestamp,
    promotion_evaluated_at: pd.Timestamp,
    input_artifacts: Mapping[str, CapturedArtifact],
    governance_gates: pd.DataFrame,
    augmented_audit_gates: pd.DataFrame,
    promotion_manifest: Mapping[str, object],
    candidate_bundle_manifest_sha256: str,
    candidate_run_manifest_sha256: str,
    release_request_id: str | None,
    release_operation_id: str | None = None,
    expected_current_event_id: str | None = None,
    operation_created_at_utc: str | None = None,
    release_root_sha256: str | None = None,
    evidence_inventory_sha256: str | None,
    signing_private_key_path: Path,
    workflow_domain_sha256: str | None = None,
    release_request_document_sha256: str | None = None,
) -> dict[str, object]:
    artifacts: dict[str, dict[str, object]] = {}
    for role, captured in input_artifacts.items():
        artifacts[role] = {
            "path": str(captured.path),
            "sha256": captured.sha256,
            "present": True,
        }
    payload: dict[str, object] = {
        "schema_version": "promotion_receipt.v2",
        "promotion_policy_version": PROMOTION_POLICY_VERSION,
        "authorization_mode": (
            "REGISTERED_PROMOTION"
            if all(
                str(value or "")
                for value in (
                    release_request_id,
                    release_operation_id,
                    operation_created_at_utc,
                    release_root_sha256,
                    evidence_inventory_sha256,
                    workflow_domain_sha256,
                    release_request_document_sha256,
                )
            )
            else "DIAGNOSTIC_ONLY"
        ),
        "run_id": str(run_id),
        "valuation_timestamp": pd.Timestamp(valuation_timestamp).isoformat(),
        "promotion_evaluated_at": pd.Timestamp(promotion_evaluated_at).isoformat(),
        "approved": bool(approved),
        "decision_summary": dict(decision_summary),
        "input_artifacts": artifacts,
        "governance_gates": governance_gates.to_dict(orient="records"),
        "augmented_audit_gates_sha256": _sha256_json(
            augmented_audit_gates.to_dict(orient="records")
        ),
        "promotion_manifest": dict(promotion_manifest),
        "candidate_bundle_manifest_sha256": str(candidate_bundle_manifest_sha256),
        "candidate_run_manifest_sha256": str(candidate_run_manifest_sha256),
        "release_request_id": str(release_request_id or ""),
        "release_operation_id": str(release_operation_id or ""),
        "expected_current_event_id": expected_current_event_id,
        "operation_created_at_utc": str(operation_created_at_utc or ""),
        "release_root_sha256": str(release_root_sha256 or ""),
        "evidence_inventory_sha256": str(evidence_inventory_sha256 or ""),
        "workflow_domain_sha256": str(workflow_domain_sha256 or ""),
        "release_request_document_sha256": str(release_request_document_sha256 or ""),
    }
    return sign_promotion_receipt(
        _json_clean(payload),
        private_key_path=signing_private_key_path,
    )


def _sha256_json(payload: object) -> str:
    raw = json.dumps(
        _json_clean(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _json_clean(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def _load_canonical_registered_release_request(
    request_path: Path,
    *,
    workflow_root: Path,
    run_id: str,
) -> dict[str, object]:
    """Load one content-addressed request only from its governed workflow slot."""

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(run_id)):
        raise ValueError("registered release request run_id is invalid")
    request, _ = _load_canonical_registered_release_request_evidence(
        request_path,
        workflow_root=workflow_root,
        run_id=run_id,
    )
    return request


def _load_canonical_registered_release_request_evidence(
    request_path: Path,
    *,
    workflow_root: Path,
    run_id: str,
) -> tuple[dict[str, object], str]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(run_id)):
        raise ValueError("registered release request run_id is invalid")
    try:
        return load_registered_release_request_evidence(
            request_path,
            workflow_root=workflow_root,
            expected_run_id=run_id,
        )
    except ReleaseRequestAuthenticationError as exc:
        raise ValueError(f"registered release request authentication failed: {exc}") from exc


def _validate_promotion_authorization_context(
    args: argparse.Namespace,
    *,
    assembled_mode: bool,
    run_id: str,
    candidate_bundle_manifest_sha256: str,
    captured: Mapping[str, CapturedArtifact],
) -> None:
    args.workflow_domain_sha256 = None
    args.release_request_document_sha256 = None
    operation_fields = {
        "release_request_id": args.release_request_id,
        "release_operation_id": args.release_operation_id,
        "expected_current_event_id": args.expected_current_event_id,
        "operation_created_at_utc": args.operation_created_at_utc,
        "release_root_sha256": args.release_root_sha256,
        "evidence_inventory_sha256": args.evidence_inventory_sha256,
    }
    populated = {key for key, value in operation_fields.items() if value is not None}
    if not populated:
        if args.registered_release_request is not None or args.workflow_root is not None:
            raise ValueError(
                "registered release request/workflow cannot be supplied without operation identity"
            )
        return
    missing = sorted(set(operation_fields).difference(populated))
    if missing:
        raise ValueError(f"promotion authorization context is incomplete: {missing}")
    if not assembled_mode:
        raise ValueError("promotion-authorizing capstone requires assembled candidate evidence")
    if args.registered_release_request is None:
        raise ValueError("promotion-authorizing capstone requires a registered release request")
    if args.workflow_root is None:
        raise ValueError("promotion-authorizing capstone requires the governed workflow root")
    try:
        assert_installed_runtime_sealed()
        assert_phase_private_key_scope("audit")
    except ReleaseCliIdentityError as exc:
        raise ValueError(str(exc)) from exc

    request, request_document_sha256 = _load_canonical_registered_release_request_evidence(
        args.registered_release_request,
        workflow_root=args.workflow_root,
        run_id=run_id,
    )
    args.workflow_domain_sha256 = str(request.get("workflow_domain_sha256", ""))
    args.release_request_document_sha256 = request_document_sha256
    request_id = str(request.get("request_id", ""))
    if request.get("registration_contract") != "assembled_candidate_seal.v1":
        raise ValueError("registered release request is not assembled-candidate evidence")
    if str(request.get("run_id", "")) != str(run_id):
        raise ValueError("registered release request run_id mismatch")
    if str(request.get("candidate_bundle_manifest_sha256", "")) != str(
        candidate_bundle_manifest_sha256
    ):
        raise ValueError("registered release request candidate bundle hash mismatch")
    expected_current = request.get("expected_current")
    expected_event = (
        None if args.expected_current_event_id == "NONE" else str(args.expected_current_event_id)
    )
    expected_contract = (
        {"kind": "NONE"}
        if expected_event is None
        else {"kind": "EVENT", "event_id": expected_event}
    )
    bindings = {
        "request_id": args.release_request_id,
        "created_at_utc": args.operation_created_at_utc,
        "release_root_sha256": args.release_root_sha256,
        "evidence_inventory_sha256": args.evidence_inventory_sha256,
    }
    for key, expected in bindings.items():
        if request.get(key) != expected:
            raise ValueError(f"registered release request {key} mismatch")
    if args.release_operation_id != request_id:
        raise ValueError("release operation_id must equal registered request_id")
    if expected_current != expected_contract:
        raise ValueError("registered release request expected_current mismatch")
    artifacts = request.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("registered release request artifact inventory is invalid")
    if _sha256_json(artifacts) != request.get("evidence_inventory_sha256"):
        raise ValueError("registered release request artifact inventory hash mismatch")
    for role, entry in artifacts.items():
        artifact = captured.get(str(role))
        if artifact is None or not isinstance(entry, Mapping):
            raise ValueError(f"registered release request artifact is absent: {role}")
        if str(entry.get("sha256", "")) != artifact.sha256 or int(
            entry.get("size_bytes", -1)
        ) != len(_captured_raw(artifact)):
            raise ValueError(f"registered release request artifact mismatch: {role}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-gates", type=absolute_path, required=True)
    parser.add_argument("--historical-thresholds", type=absolute_path, required=True)
    parser.add_argument("--manifest", type=absolute_path, default=None)
    parser.add_argument("--production-manifest", type=absolute_path, required=True)
    parser.add_argument("--export-manifest", type=absolute_path, required=True)
    parser.add_argument("--selected-config-artifact", type=absolute_path, required=True)
    parser.add_argument(
        "--candidate-bundle-manifest",
        type=absolute_path,
        default=None,
    )
    parser.add_argument(
        "--data-root",
        type=absolute_path,
        required=True,
        help="Governed shared LT data root used to replay candidate freshness",
    )
    parser.add_argument(
        "--product-normalization-summary",
        type=absolute_path,
        default=None,
    )
    parser.add_argument(
        "--signing-private-key",
        type=absolute_path,
        default=os.environ.get("PFC_PROMOTION_SIGNING_PRIVATE_KEY_PATH"),
        required=os.environ.get("PFC_PROMOTION_SIGNING_PRIVATE_KEY_PATH") is None,
        help="Mounted Ed25519 private key used to authenticate the external receipt",
    )
    parser.add_argument("--far-horizon-min-years", type=int, default=2)
    parser.add_argument("--augmented-audit-gates", type=absolute_path, default=None)
    parser.add_argument("--output", type=absolute_path, default=None)
    parser.add_argument("--details-output", type=absolute_path, default=None)
    parser.add_argument("--release-request-id", default=None)
    parser.add_argument("--release-operation-id", default=None)
    parser.add_argument("--expected-current-event-id", default=None)
    parser.add_argument("--operation-created-at-utc", default=None)
    parser.add_argument("--release-root-sha256", default=None)
    parser.add_argument("--evidence-inventory-sha256", default=None)
    parser.add_argument("--registered-release-request", type=absolute_path, default=None)
    parser.add_argument("--workflow-root", type=absolute_path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
