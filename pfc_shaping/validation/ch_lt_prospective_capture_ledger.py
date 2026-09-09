"""Build a hash-bound ledger of contiguous local CH hourly captures.

The ledger is deliberately negative-authority evidence.  It proves only that
the referenced local capture and Builder artifacts are replay-consistent and
temporally contiguous, and that exact local-verifier receipt bytes make only
unsigned claims about their runtime.  It cannot manufacture trusted
``available_at`` time, product semantics, independence, CAS admission or any
scientific/production authority.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pfc_shaping.data.governed_lt_acquisition import (
    validate_raw_envelope,
    verify_provider_raw_replay,
)
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json

REQUEST_SCHEMA = "ch_lt_prospective_capture_ledger_request.v1"
LEDGER_SCHEMA = "ch_lt_prospective_capture_ledger.v1"
LEDGER_STATUS = "CONTIGUOUS_LOCAL_QUARANTINE_NOT_PIT_AUTHORITY"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_JSON_BYTES = 2 * 1024 * 1024
_REQUEST_KEYS = {"schema_version", "market", "source_role", "entries"}
_ENTRY_KEYS = {
    "capture_summary_path",
    "capture_summary_sha256",
    "acquisition_manifest_path",
    "acquisition_manifest_sha256",
    "audit_path",
    "audit_sha256",
    "audit_runtime_receipt_path",
    "audit_runtime_receipt_sha256",
}
_BLOCKERS = (
    "TRUSTED_AVAILABLE_AT_AND_REVISION_LINEAGE",
    "PROVIDER_AUTHENTICATED_PRODUCT_AUCTION_SESSION_AND_SETTLEMENT_SEMANTICS",
    "CAPTURE_TIME_CA_TLS_IDENTITY_ATTESTATION",
    "INDEPENDENT_ACQUISITION_SIGNATURE",
    "BUILDER_INACCESSIBLE_IMMUTABLE_FREEZE",
    "EXTERNAL_CAS_WORM_RECEIPT_AND_FRESH_MONOTONE_HEAD",
    "MULTI_SEASON_MULTI_REGIME_INDEPENDENT_ROLLING_ORIGINS",
    "DEPENDENCE_AWARE_POWER_AND_PREREGISTERED_UNIQUE_ORIGINS",
    "SEALED_FUTURE_HOLDOUT",
    "CALIBRATED_PROBABILISTIC_SCENARIOS",
)
_AUTHORITY_KEY_MARKERS = (
    "admission",
    "admitted",
    "authoritative",
    "authority",
    "authorization",
    "authorized",
    "eligible",
    "holdout_consumed",
    "immutable",
    "official",
    "production",
    "promotion",
    "published",
    "signature",
    "signed",
    "t057_consumed",
    "trusted",
    "verified",
)
_ALLOWED_TRUE_AUTHORITY_LIKE_KEYS = {
    "hourly_aggregation_eligible",
    "local_diagnostic_allowed",
    "native_hourly_cadence_eligible",
    "next_authority",
}
_NEGATIVE_STATUS_MARKERS = (
    "absent",
    "blocked",
    "false",
    "local",
    "missing",
    "never",
    "no_go",
    "not_",
    "requires",
    "unsigned",
    "untrusted",
    "unverified",
)


class ChLtProspectiveCaptureLedgerError(ValueError):
    """Raised when local prospective evidence cannot form one exact ledger."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def build_prospective_capture_ledger(
    *,
    repo_root: str | Path,
    request_path: str | Path,
    expected_request_sha256: str,
) -> dict[str, object]:
    """Validate exact local evidence and return a canonical negative ledger."""

    root = _absolute_no_links(repo_root)
    build_root = _absolute_no_links(root / "build")
    selected_request_path = _path(
        request_path,
        label="prospective capture ledger request",
    )
    request = _load_hash_bound_mapping(
        selected_request_path,
        expected_request_sha256,
        label="prospective capture ledger request",
        build_root=build_root,
    )
    _exact_keys(request, _REQUEST_KEYS, label="ledger request")
    _equal(request.get("schema_version"), REQUEST_SCHEMA, "request schema")
    _equal(request.get("market"), "CH", "market")
    _equal(request.get("source_role"), "epex_ch", "source role")
    raw_entries = request.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ChLtProspectiveCaptureLedgerError("ledger request entries are missing")

    ledger_entries: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_file_identities: set[tuple[int, int]] = set()
    seen_bodies: set[str] = set()
    previous_end: datetime | None = None
    total_source = 0
    total_output = 0

    for index, raw_entry in enumerate(raw_entries):
        entry = _mapping(raw_entry, label=f"entry {index}")
        _exact_keys(entry, _ENTRY_KEYS, label=f"entry {index}")
        verified = _verify_entry(
            entry,
            index=index,
            root=root,
            build_root=build_root,
        )
        acquisition_id = str(verified["acquisition_id"])
        if acquisition_id in seen_ids:
            raise ChLtProspectiveCaptureLedgerError("acquisition IDs are not unique")
        seen_ids.add(acquisition_id)
        for path in verified.pop("_input_paths"):
            key = os.path.normcase(os.path.abspath(path))
            if key in seen_paths:
                raise ChLtProspectiveCaptureLedgerError("ledger evidence paths are reused")
            seen_paths.add(key)
            metadata = Path(path).stat(follow_symlinks=False)
            identity = (int(metadata.st_dev), int(metadata.st_ino))
            if identity in seen_file_identities:
                raise ChLtProspectiveCaptureLedgerError(
                    "ledger evidence files are reused through a physical alias"
                )
            seen_file_identities.add(identity)
        body_hash = str(verified.pop("_provider_body_sha256"))
        if body_hash in seen_bodies:
            raise ChLtProspectiveCaptureLedgerError("provider body bytes are reused")
        seen_bodies.add(body_hash)

        start = _utc_datetime(verified["start_utc"], label=f"entry {index} start")
        end = _utc_datetime(verified["end_utc"], label=f"entry {index} end")
        if previous_end is not None and start != previous_end:
            relation = "overlap" if start < previous_end else "gap"
            raise ChLtProspectiveCaptureLedgerError(
                f"prospective capture sequence contains a {relation}"
            )
        previous_end = end
        total_source += int(verified["source_observation_count"])
        total_output += int(verified["output_observation_count"])
        ledger_entries.append(verified)

    first_start = str(ledger_entries[0]["start_utc"])
    final_end = str(ledger_entries[-1]["end_utc"])
    duration_hours = int(
        (_utc_datetime(final_end, label="ledger end") - _utc_datetime(first_start, label="ledger start"))
        .total_seconds()
        // 3600
    )
    if duration_hours != total_source:
        raise ChLtProspectiveCaptureLedgerError(
            "contiguous duration does not equal native hourly observation count"
        )
    if total_output != total_source * 4:
        raise ChLtProspectiveCaptureLedgerError(
            "15-minute transport row count does not equal four times native hourly rows"
        )

    ledger: dict[str, object] = {
        "schema_version": LEDGER_SCHEMA,
        "ledger_id": "",
        "status": LEDGER_STATUS,
        "market": "CH",
        "source_role": "epex_ch",
        "construction_request": _reference(
            root,
            selected_request_path,
            expected_request_sha256,
        ),
        "window": {
            "start_utc": first_start,
            "end_utc_exclusive": final_end,
            "duration_hours": duration_hours,
        },
        "inventory": {
            "capture_count": len(ledger_entries),
            "native_hourly_observation_count": total_source,
            "transport_quarter_hour_row_count": total_output,
            "native_cadence_seconds": 3600,
            "transport_cadence_seconds": 900,
            "transport_sampling_kind": "UPSAMPLED_STEPWISE_PROXY",
            "proxy_rows_add_independent_information": False,
        },
        "entries": ledger_entries,
        "scientific_scope": {
            "structural_cadence_and_exact_replay_check_allowed": True,
            "price_level_shaping_or_economic_inference_allowed": False,
            "native_quarter_hour_truth_eligible": False,
            "independent_rolling_origin_count": 0,
            "seasonal_regime_coverage": "NOT_ESTABLISHED",
            "model_selection_eligible": False,
            "confirmatory_rolling_origin_eligible": False,
            "future_holdout_eligible": False,
        },
        "holdout_access_evidence": {
            "constructor_read_scope": "EXACT_HASH_BOUND_CH_CAPTURE_CHAIN_ONLY",
            "forbidden_path_markers": ["t057", "holdout"],
            "forbidden_path_marker_match_count": 0,
            "future_holdout_or_t057_input_role_present": False,
        },
        "missing_external_authorities": list(_BLOCKERS),
        "scientific_admission": False,
        "training_eligible": False,
        "candidate_assembly_eligible": False,
        "publication_authorized": False,
        "promotion_gate": False,
        "production_authorization": False,
        "future_holdout_consumed": False,
        "t057_consumed": False,
    }
    unsigned = dict(ledger)
    unsigned.pop("ledger_id")
    ledger["ledger_id"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    return ledger


def _verify_entry(
    entry: Mapping[str, object],
    *,
    index: int,
    root: Path,
    build_root: Path,
) -> dict[str, object]:
    documents: dict[str, tuple[Path, str, Mapping[str, object]]] = {}
    for prefix in (
        "capture_summary",
        "acquisition_manifest",
        "audit",
        "audit_runtime_receipt",
    ):
        path = _path(entry.get(f"{prefix}_path"), label=f"entry {index} {prefix}")
        expected = _sha256(entry.get(f"{prefix}_sha256"), label=f"entry {index} {prefix}")
        document = _load_hash_bound_mapping(
            path,
            expected,
            label=f"entry {index} {prefix}",
            build_root=build_root,
        )
        documents[prefix] = (path, expected, document)

    summary_path, summary_hash, summary = documents["capture_summary"]
    manifest_path, manifest_hash, manifest = documents["acquisition_manifest"]
    audit_path, audit_hash, audit = documents["audit"]
    receipt_path, receipt_hash, receipt = documents["audit_runtime_receipt"]
    for label, document in (
        ("capture summary", summary),
        ("Builder manifest", manifest),
        ("local isolated audit", audit),
        ("local verifier runtime receipt", receipt),
    ):
        _reject_smuggled_positive_authority(document, label=label)

    _equal(summary.get("schema_version"), "local_public_provider_capture.v2", "capture schema")
    _equal(summary.get("status"), "COMPLETE_LOCAL_UNTRUSTED_CAPTURE", "capture status")
    _equal(summary.get("source_role"), "epex_ch", "capture role")
    _equal(summary.get("raw_cadence_minutes"), 60, "capture cadence")
    _equal(summary.get("timestamp_authority"), "LOCAL_WORKSTATION_CLOCK_UNTRUSTED", "capture time authority")
    _equal(summary.get("publication_status"), "NOT_PUBLISHED", "capture publication")
    _equal(
        summary.get("next_authority"),
        "NETWORKLESS_INSTALLED_ACQUISITION_BUILDER",
        "capture next authority",
    )
    for field in ("production_authorization", "promotion_gate"):
        _equal(summary.get(field), False, f"capture {field}")

    acquisition_id = _text(summary.get("acquisition_id"), label="acquisition ID")
    start = _utc_datetime(summary.get("start_utc"), label="capture start")
    end = _utc_datetime(summary.get("end_utc"), label="capture end")
    received = _utc_datetime(summary.get("received_at_utc"), label="capture received time")
    if not start < end or received < end:
        raise ChLtProspectiveCaptureLedgerError("capture chronology is invalid")
    duration_hours = int((end - start).total_seconds() // 3600)
    if (end - start).total_seconds() != duration_hours * 3600:
        raise ChLtProspectiveCaptureLedgerError("capture window is not whole-hour aligned")
    _equal(summary.get("validated_bronze_start_utc"), _utc_text(start), "validated bronze start")
    _equal(summary.get("validated_bronze_end_utc"), _utc_text(end), "validated bronze end")
    _equal(summary.get("validated_bronze_rows"), duration_hours * 4, "validated bronze rows")

    resolution = _mapping(summary.get("resolution_provenance"), label="capture resolution")
    _verify_resolution(resolution, duration_hours=duration_hours)

    spec_ref = _mapping(summary.get("capture_spec"), label="capture spec reference")
    body_ref = _mapping(summary.get("provider_body"), label="provider body reference")
    spec_path = _path(spec_ref.get("path"), label="capture spec")
    body_path = _path(body_ref.get("path"), label="provider body")
    spec_hash = _sha256(spec_ref.get("sha256"), label="capture spec")
    body_hash = _sha256(body_ref.get("sha256"), label="provider body")
    spec_payload = _read_hash_bound_file(
        spec_path,
        spec_hash,
        expected_size=spec_ref.get("size_bytes"),
        label="capture spec",
        build_root=build_root,
    )
    body_payload = _read_hash_bound_file(
        body_path,
        body_hash,
        expected_size=body_ref.get("size_bytes"),
        label="provider body",
        build_root=build_root,
    )
    spec = _strict_mapping(spec_payload, label="capture spec")
    _reject_smuggled_positive_authority(spec, label="capture spec")
    _equal(spec.get("schema_version"), "lt_provider_capture_spec.v1", "provider capture spec schema")
    _equal(spec.get("acquisition_id"), acquisition_id, "spec acquisition ID")
    _equal(spec.get("source_role"), "epex_ch", "spec role")
    _equal(spec.get("source_system"), "ENERGY_CHARTS_API", "spec source system")
    _equal(spec.get("source_locator"), "https://api.energy-charts.info/price", "spec source locator")
    _equal(spec.get("provider_id"), "ENERGY_CHARTS_API", "spec provider")
    _equal(spec.get("received_at_utc"), _utc_text(received), "spec received time")
    _equal(spec.get("start_utc"), _utc_text(start), "spec start")
    _equal(spec.get("end_utc"), _utc_text(end), "spec end")
    spec_documents = spec.get("documents")
    if not isinstance(spec_documents, list) or len(spec_documents) != 1:
        raise ChLtProspectiveCaptureLedgerError("capture spec document inventory is not exact")
    spec_document = _mapping(spec_documents[0], label="capture spec document")
    if os.path.normcase(os.path.abspath(_path(spec_document.get("body_path"), label="spec body"))) != os.path.normcase(os.path.abspath(body_path)):
        raise ChLtProspectiveCaptureLedgerError("capture spec body path mismatch")
    _equal(spec_document.get("document_id"), "day_ahead_price", "spec document ID")
    _equal(
        spec_document.get("request_url"),
        "https://api.energy-charts.info/price",
        "spec request URL",
    )
    _equal(
        spec_document.get("request_parameters"),
        {
            "bzn": "CH",
            "start": start.strftime("%Y-%m-%dT%H:%MZ"),
            "end": (end - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%MZ"),
        },
        "spec request parameters",
    )
    _equal(spec_document.get("credential_reference"), None, "spec credential")
    _equal(spec_document.get("response_status_code"), 200, "spec response status")
    _equal(spec_document.get("response_media_type"), "application/json", "spec media type")
    _equal(spec_document.get("response_content_encoding"), "identity", "spec content encoding")

    _equal(manifest.get("schema_version"), "lt_provider_acquisition_build.v2", "Builder manifest schema")
    _equal(manifest.get("acquisition_id"), acquisition_id, "Builder acquisition ID")
    _equal(manifest.get("source_role"), "epex_ch", "Builder role")
    _equal(manifest.get("source_system"), "ENERGY_CHARTS_API", "Builder source system")
    _equal(manifest.get("received_at_utc"), _utc_text(received), "Builder received time")
    _equal(manifest.get("capture_spec"), {"sha256": spec_hash, "size_bytes": len(spec_payload)}, "Builder capture spec")
    _equal(manifest.get("resolution_provenance"), dict(resolution), "Builder resolution")
    _equal(manifest.get("signing_status"), "UNSIGNED_REQUIRES_INDEPENDENT_AUTHORITIES", "Builder signing")
    _equal(manifest.get("publication_status"), "NOT_PUBLISHED", "Builder publication")
    _equal(manifest.get("namespace_security_status"), "BUILDER_MUTABLE_UNTRUSTED_QUARANTINE", "Builder namespace")
    _equal(
        manifest.get("authority_handoff_status"),
        "REQUIRES_BUILDER_INACCESSIBLE_COPY_AND_INDEPENDENT_SIGNATURE",
        "Builder authority handoff",
    )
    artifacts = _mapping(manifest.get("artifacts"), label="Builder artifacts")
    expected_artifact_names = {
        "bronze.parquet",
        "provider-parser.py",
        "provider-raw-envelope.json",
        "provider-transform-config.json",
    }
    if set(artifacts) != expected_artifact_names:
        raise ChLtProspectiveCaptureLedgerError("Builder artifact inventory is not exact")
    artifact_references: dict[str, dict[str, object]] = {}
    artifact_payloads: dict[str, bytes] = {}
    artifact_paths: list[Path] = []
    for name in sorted(expected_artifact_names):
        artifact = _mapping(artifacts.get(name), label=f"Builder artifact {name}")
        artifact_hash = _sha256(artifact.get("sha256"), label=f"Builder artifact {name}")
        artifact_path = manifest_path.parent / name
        payload = _read_hash_bound_file(
            artifact_path,
            artifact_hash,
            expected_size=artifact.get("size_bytes"),
            label=f"Builder artifact {name}",
            build_root=build_root,
        )
        artifact_references[name] = {
            **_reference(root, artifact_path, artifact_hash),
            "size_bytes": len(payload),
        }
        artifact_payloads[name] = payload
        artifact_paths.append(artifact_path)
    parser_config = _strict_mapping(
        artifact_payloads["provider-transform-config.json"],
        label="provider transform config",
    )
    replay = verify_provider_raw_replay(
        role="epex_ch",
        source_system="ENERGY_CHARTS_API",
        envelope_payload=artifact_payloads["provider-raw-envelope.json"],
        bronze_payload=artifact_payloads["bronze.parquet"],
        parser_payload=artifact_payloads["provider-parser.py"],
        parser_code_sha256=str(artifacts["provider-parser.py"]["sha256"]),
        parser_config=parser_config,
    )
    _equal(replay.get("status"), "VERIFIED_EXACT_PROVIDER_REPLAY", "ledger provider replay")
    _equal(replay.get("resolution_provenance"), dict(resolution), "ledger replay resolution")
    _equal(
        replay.get("bronze_frame_sha256"),
        manifest.get("bronze_frame_sha256"),
        "ledger replay semantic frame",
    )
    envelope = validate_raw_envelope(artifact_payloads["provider-raw-envelope.json"])
    envelope_documents = envelope.get("documents")
    if not isinstance(envelope_documents, list) or len(envelope_documents) != 1:
        raise ChLtProspectiveCaptureLedgerError("provider envelope document inventory is not exact")
    envelope_document = _mapping(envelope_documents[0], label="provider envelope document")
    try:
        embedded_body = base64.b64decode(
            _text(envelope_document.get("body_base64"), label="provider envelope body"),
            validate=True,
        )
    except (ValueError, TypeError) as exc:
        raise ChLtProspectiveCaptureLedgerError("provider envelope body is invalid") from exc
    if embedded_body != body_payload:
        raise ChLtProspectiveCaptureLedgerError(
            "provider envelope does not contain the exact captured body"
        )

    _equal(audit.get("schema_version"), "local_provider_acquisition_audit.v4", "audit schema")
    _equal(audit.get("status"), "VERIFIED_LOCAL_QUARANTINE_NOT_PRODUCTION", "audit status")
    _equal(audit.get("manifest_sha256"), manifest_hash, "audit manifest hash")
    _equal(audit.get("source_role"), "epex_ch", "audit role")
    _equal(audit.get("source_system"), "ENERGY_CHARTS_API", "audit source system")
    _equal(audit.get("received_at_utc"), _utc_text(received), "audit received time")
    _equal(audit.get("resolution_provenance"), dict(resolution), "audit resolution")
    _equal(audit.get("artifacts"), dict(artifacts), "audit artifacts")
    _equal(audit.get("replay"), replay, "audit replay")
    for field in (
        "scientific_admission",
        "training_eligible",
        "model_selection_eligible",
        "candidate_assembly_eligible",
        "confirmatory_rolling_origin_eligible",
        "future_holdout_eligible",
        "t057_eligible",
        "production_authorization",
        "promotion_gate",
    ):
        _equal(audit.get(field), False, f"audit {field}")
    evidence_scope = _mapping(audit.get("evidence_scope"), label="audit evidence scope")
    _equal(evidence_scope.get("source_observation_count"), duration_hours, "audit source count")
    _equal(evidence_scope.get("output_observation_count"), duration_hours * 4, "audit output count")
    _equal(evidence_scope.get("proxy_rows_add_independent_information"), False, "audit proxy independence")

    _equal(receipt.get("schema_version"), "fmv_lt_provider_verifier_audit_receipt.v1", "runtime receipt schema")
    _equal(receipt.get("command"), "audit-acquisition", "runtime receipt command")
    _equal(receipt.get("status"), "LOCAL_RUNTIME_OBSERVATION_NOT_AUTHORITY", "runtime receipt status")
    _equal(receipt.get("runtime_authority"), False, "runtime authority")
    _equal(receipt.get("production_authorization"), False, "runtime production authority")
    audit_result = _mapping(receipt.get("audit_result"), label="runtime audit result")
    _equal(audit_result.get("sha256"), audit_hash, "runtime audit hash")
    _equal(audit_result.get("status"), "VERIFIED_LOCAL_QUARANTINE_NOT_PRODUCTION", "runtime audit status")
    if os.path.normcase(os.path.abspath(_path(audit_result.get("path"), label="runtime audit path"))) != os.path.normcase(os.path.abspath(audit_path)):
        raise ChLtProspectiveCaptureLedgerError("runtime receipt audit path mismatch")
    runtime = _mapping(receipt.get("runtime_observation"), label="runtime observation")
    source_revision = _sha256(runtime.get("source_revision"), label="verifier source revision")
    _equal(
        runtime.get("supervisor_attestation"),
        "UNSIGNED_LOCAL_PROCESS_OBSERVATION",
        "verifier supervisor attestation",
    )
    admission = _mapping(runtime.get("receipt"), label="runtime admission")
    expected_admission = {
        "captured_artifact_sys_path_count": 1,
        "captured_dependency_root_sys_path_count": 1,
        "source_artifact_sys_path_count": 0,
        "source_dependency_root_sys_path_count": 0,
        "dependency_import_source": "PROCESS_PRIVATE_EXACT_BYTE_COPY",
        "schema_version": "fmv_lt_provider_verifier_runtime_admission.v1",
        "status": "PASS",
    }
    for field, value in expected_admission.items():
        _equal(admission.get(field), value, f"runtime admission {field}")
    verifier_artifact_hash = _sha256(
        admission.get("captured_artifact_sha256"), label="verifier artifact"
    )
    dependency_tree_hash = _sha256(
        admission.get("dependency_tree_sha256"), label="verifier dependency tree"
    )

    acquisition_dir = _path(audit.get("acquisition_directory"), label="audit acquisition directory")
    if os.path.normcase(os.path.abspath(manifest_path.parent)) != os.path.normcase(os.path.abspath(acquisition_dir)):
        raise ChLtProspectiveCaptureLedgerError("audit acquisition directory mismatch")

    return {
        "acquisition_id": acquisition_id,
        "start_utc": _utc_text(start),
        "end_utc": _utc_text(end),
        "received_at_utc_untrusted": _utc_text(received),
        "source_observation_count": duration_hours,
        "output_observation_count": duration_hours * 4,
        "capture_summary": _reference(root, summary_path, summary_hash),
        "capture_spec": _reference(root, spec_path, spec_hash),
        "provider_body": _reference(root, body_path, body_hash),
        "acquisition_manifest": _reference(root, manifest_path, manifest_hash),
        "acquisition_artifacts": artifact_references,
        "local_isolated_audit": _reference(root, audit_path, audit_hash),
        "audit_runtime_receipt": _reference(root, receipt_path, receipt_hash),
        "verifier_runtime": {
            "receipt_claimed_artifact_sha256": verifier_artifact_hash,
            "receipt_claimed_dependency_tree_sha256": dependency_tree_hash,
            "receipt_claimed_source_revision": source_revision,
            "receipt_claimed_dependency_import_source": (
                "PROCESS_PRIVATE_EXACT_BYTE_COPY"
            ),
            "receipt_claimed_source_sys_path_count": 0,
            "ledger_rehashed_runtime_payloads": False,
            "attestation": "UNSIGNED_LOCAL_PROCESS_OBSERVATION_NOT_AUTHORITY",
        },
        "native_hourly_cadence_observed": True,
        "native_hourly_scientific_truth_eligible": False,
        "native_quarter_hour_truth_eligible": False,
        "scientific_admission": False,
        "production_authorization": False,
        "_provider_body_sha256": body_hash,
        "_input_paths": [
            summary_path,
            spec_path,
            body_path,
            manifest_path,
            audit_path,
            receipt_path,
            *artifact_paths,
        ],
    }


def _verify_resolution(value: Mapping[str, object], *, duration_hours: int) -> None:
    expected = {
        "schema_version": "lt_observation_resolution_provenance.v2",
        "role": "epex_ch",
        "source_cadence_seconds": 3600,
        "output_cadence_seconds": 900,
        "source_observation_count": duration_hours,
        "output_observation_count": duration_hours * 4,
        "output_sampling_kind": "UPSAMPLED_STEPWISE_PROXY",
        "resampling_method": "FORWARD_FILL_WITHIN_SOURCE_INTERVAL",
        "native_hourly_cadence_eligible": True,
        "hourly_aggregation_eligible": True,
        "native_quarter_hour_cadence_eligible": False,
        "native_quarter_hour_truth_eligible": False,
        "quarter_hour_truth_blocker": "SOURCE_CADENCE_IS_HOURLY",
        "scientific_use_class": "NATIVE_HOURLY_PRICE_QH_TRANSPORT_PROXY_ONLY",
        "product_identity_status": "EXTERNAL_PRODUCT_AUCTION_SESSION_IDENTITY_UNVERIFIED",
    }
    _equal(dict(value), expected, "resolution provenance")


def _load_hash_bound_mapping(
    path: str | Path,
    expected_sha256: object,
    *,
    label: str,
    build_root: Path,
) -> Mapping[str, object]:
    selected = _path(path, label=label)
    expected = _sha256(expected_sha256, label=label)
    payload = _read_hash_bound_file(
        selected,
        expected,
        expected_size=None,
        label=label,
        build_root=build_root,
    )
    return _strict_mapping(payload, label=label)


def _read_hash_bound_file(
    path: Path,
    expected_sha256: str,
    *,
    expected_size: object | None,
    label: str,
    build_root: Path,
) -> bytes:
    selected = _absolute_no_links(path)
    _require_within(selected, build_root, label=label)
    _reject_holdout_path_marker(selected, label=label)
    payload = read_stable_single_link_file(selected, label=label, max_bytes=_MAX_JSON_BYTES if selected.suffix.lower() == ".json" else 64 * 1024 * 1024)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ChLtProspectiveCaptureLedgerError(f"{label} SHA-256 mismatch")
    if expected_size is not None and (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size != len(payload)
    ):
        raise ChLtProspectiveCaptureLedgerError(f"{label} size mismatch")
    return payload


def _strict_mapping(payload: bytes, *, label: str) -> Mapping[str, object]:
    try:
        value = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChLtProspectiveCaptureLedgerError(f"{label} is not strict JSON") from exc
    return _mapping(value, label=label)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ChLtProspectiveCaptureLedgerError(f"{label} must be a string-keyed mapping")
    return value


def _reject_smuggled_positive_authority(value: object, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if (
                not isinstance(item, (Mapping, list, tuple))
                and normalized not in _ALLOWED_TRUE_AUTHORITY_LIKE_KEYS
                and (
                    normalized == "valid"
                    or any(
                        marker in normalized for marker in _AUTHORITY_KEY_MARKERS
                    )
                )
                and not _is_explicitly_negative_authority_value(item)
            ):
                raise ChLtProspectiveCaptureLedgerError(
                    f"{label} contains a positive authority-like claim: {key}"
                )
            _reject_smuggled_positive_authority(item, label=label)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_smuggled_positive_authority(item, label=label)


def _is_explicitly_negative_authority_value(value: object) -> bool:
    if value is False or value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        return any(marker in normalized for marker in _NEGATIVE_STATUS_MARKERS)
    return False


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ChLtProspectiveCaptureLedgerError(f"{label} fields are not exact")


def _equal(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        raise ChLtProspectiveCaptureLedgerError(f"{label} mismatch")


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ChLtProspectiveCaptureLedgerError(f"{label} is invalid")
    return value


def _sha256(value: object, *, label: str) -> str:
    text = _text(value, label=f"{label} SHA-256")
    if _SHA256.fullmatch(text) is None:
        raise ChLtProspectiveCaptureLedgerError(f"{label} SHA-256 is invalid")
    return text


def _path(value: object, *, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ChLtProspectiveCaptureLedgerError(f"{label} path is invalid")
    raw = os.fspath(value)
    if not raw or raw.strip() != raw:
        raise ChLtProspectiveCaptureLedgerError(f"{label} path is invalid")
    return _absolute_no_links(raw)


def _absolute_no_links(path: str | Path) -> Path:
    raw = os.fspath(path)
    if os.name == "nt":
        _, tail = os.path.splitdrive(raw)
        if ":" in tail:
            raise ChLtProspectiveCaptureLedgerError(
                "Windows alternate data streams are forbidden"
            )
    try:
        return assert_absolute_path_has_no_links(raw)
    except ValueError as exc:
        raise ChLtProspectiveCaptureLedgerError(str(exc)) from exc


def _require_within(path: Path, parent: Path, *, label: str) -> None:
    try:
        within = os.path.commonpath((os.path.normcase(str(path)), os.path.normcase(str(parent)))) == os.path.normcase(str(parent))
    except ValueError:
        within = False
    if not within:
        raise ChLtProspectiveCaptureLedgerError(f"{label} must remain below repo build")


def _reject_holdout_path_marker(path: Path, *, label: str) -> None:
    lowered = tuple(part.lower() for part in path.parts)
    if any("t057" in part or "holdout" in part for part in lowered):
        raise ChLtProspectiveCaptureLedgerError(
            f"{label} path contains a forbidden future-holdout marker"
        )


def _utc_datetime(value: object, *, label: str) -> datetime:
    text = _text(value, label=label)
    if not text.endswith("Z"):
        raise ChLtProspectiveCaptureLedgerError(f"{label} must use canonical UTC Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ChLtProspectiveCaptureLedgerError(f"{label} is invalid") from exc
    if parsed.tzinfo != timezone.utc or _utc_text(parsed) != text:
        raise ChLtProspectiveCaptureLedgerError(f"{label} is not canonical")
    return parsed


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _reference(root: Path, path: Path, sha256: str) -> dict[str, object]:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ChLtProspectiveCaptureLedgerError("ledger evidence is outside repo") from exc
    return {"path": relative, "sha256": sha256}


__all__ = [
    "ChLtProspectiveCaptureLedgerError",
    "LEDGER_SCHEMA",
    "LEDGER_STATUS",
    "REQUEST_SCHEMA",
    "build_prospective_capture_ledger",
    "canonical_json_bytes",
]
