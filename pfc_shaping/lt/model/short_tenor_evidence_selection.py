"""Fail-closed selection of the local EEX short-tenor evidence chain.

The selection reconciles content-addressed local artifacts only.  It never
loads market values, learns a coefficient, authenticates PIT availability or
grants model/production authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any


SCHEMA_VERSION = "fmv-eex-ch-short-tenor-evidence-selection-v1"
SELECTION_STATUS = "PASS_LOCAL_EVIDENCE_CHAIN_RECONCILED_NO_MODEL_AUTHORITY"
SELECTION_DECISION = "D-20260805-225"
SELECTION_CONTENT_ID = (
    "792149a7d46834b17507ea4ec6a15dbfeb2ab9233b058b9783673a58a57c26aa"
)

SELECTED_CONTRAST_CONTENT_ID = (
    "309b9f07236d1cfb32b3d92c1ed5413bde6966fa383629882a539ccbd60e9cb5"
)
SELECTED_COMBINATION_CONTENT_ID = (
    "122afd7292549b00a946a2893487d344e459c420a8bbdf99c11714c7308a8cfd"
)
SELECTED_POLICY_CONTENT_ID = (
    "189e7ab9d460daefbaa689f1dc8ae4c3bb269f7b5bcb5eae95e77b6d0d4d3e43"
)
NORMALIZATION_CONTENT_ID = (
    "2837dc4849dc4b573c441059574973e0b8cc0fbb5023203509cb2929dd636a3f"
)
SHORT_TENOR_AUDIT_CONTENT_ID = (
    "b09eb3250df5a3c0616eb169c512319c514ddf540251b405023d9351bd5d8bde"
)
SUPERSEDED_CONTRAST_IDS = {
    "ae7b962ca5f85ff75223ad45fd578c29b49a1d66dfbdd418b941c337dda38147",
    "7a4c2a601e169d8c471e0939919f12cf56b03d132d8691a82d6bfef5b392bd03",
}
SUPERSEDED_COMBINATION_IDS = {
    "21c557df75260ce162c15af6fbe0c91de4c8a6fcd233564403a04e7b1fc53ad0",
    "c06dfcf5fbda16cd5bab04005d4581e35216116c30c6c9df6cef94f2d949f67b",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_KEYS = {
    "schema_version",
    "status",
    "decision",
    "purpose",
    "selected_chain",
    "superseded_evidence",
    "scientific_payload_identity",
    "execution",
    "authorities",
    "invariants",
}
_FALSE_AUTHORITIES = {
    "point_in_time_availability_proven",
    "rolling_origin_selection_authorized",
    "model_input_authorized",
    "candidate_assembly_authorized",
    "promotion_authorized",
    "production_authorized",
}
_ZERO_EXECUTION_FIELDS = {
    "databricks_request_count",
    "warehouse_start_count",
    "network_call_count",
    "remote_write_count",
}


def validate_short_tenor_evidence_selection(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate the frozen selection document without reading any artifact."""

    _require_exact_keys(document, _TOP_LEVEL_KEYS, "selection document")
    _require_equal(document.get("schema_version"), SCHEMA_VERSION, "schema")
    _require_equal(document.get("status"), SELECTION_STATUS, "status")
    _require_equal(document.get("decision"), SELECTION_DECISION, "decision")

    selected = _require_mapping(document.get("selected_chain"), "selected chain")
    _require_equal(
        selected.get("normalization_content_id"),
        NORMALIZATION_CONTENT_ID,
        "normalization content id",
    )
    _require_equal(
        selected.get("short_tenor_audit_content_id"),
        SHORT_TENOR_AUDIT_CONTENT_ID,
        "short-tenor audit content id",
    )
    contrast = _require_mapping(selected.get("contrast_bundle"), "contrast bundle")
    proof = _require_mapping(selected.get("combination_proof"), "combination proof")
    policy = _require_mapping(
        selected.get("combination_policy"), "combination policy"
    )
    _require_equal(
        contrast.get("content_id"),
        SELECTED_CONTRAST_CONTENT_ID,
        "selected contrast content id",
    )
    _require_equal(
        proof.get("content_id"),
        SELECTED_COMBINATION_CONTENT_ID,
        "selected combination content id",
    )
    _require_equal(
        policy.get("content_id"),
        SELECTED_POLICY_CONTENT_ID,
        "selected policy content id",
    )
    _require_sha_fields(contrast, "selected contrast")
    _require_sha_fields(proof, "selected combination")
    _require_sha_fields(policy, "selected policy")

    superseded = document.get("superseded_evidence")
    if not isinstance(superseded, list) or len(superseded) != 4:
        raise ValueError("selection must name exactly four superseded artifacts")
    observed_contrasts: set[str] = set()
    observed_proofs: set[str] = set()
    for raw_entry in superseded:
        entry = _require_mapping(raw_entry, "superseded entry")
        _require_exact_keys(
            entry,
            {
                "kind",
                "content_id",
                "manifest_sha256",
                "bound_source_contrast_id",
                "reason",
            },
            "superseded entry",
        )
        content_id = _require_sha(entry.get("content_id"), "superseded content id")
        _require_sha(entry.get("manifest_sha256"), "superseded manifest")
        kind = entry.get("kind")
        if kind == "CONTRAST_BUNDLE":
            observed_contrasts.add(content_id)
            _require_equal(
                entry.get("bound_source_contrast_id"),
                None,
                "superseded contrast source binding",
            )
            _require_equal(
                entry.get("reason"),
                "STALE_TEST_BYTES_SAME_SCIENTIFIC_PAYLOAD",
                "superseded contrast reason",
            )
        elif kind == "COMBINATION_PROOF":
            observed_proofs.add(content_id)
            bound = _require_sha(
                entry.get("bound_source_contrast_id"),
                "superseded proof contrast binding",
            )
            if bound not in SUPERSEDED_CONTRAST_IDS:
                raise ValueError("superseded proof must bind a superseded contrast")
            _require_equal(
                entry.get("reason"),
                "BINDS_SUPERSEDED_CONTRAST_BUNDLE",
                "superseded proof reason",
            )
        else:
            raise ValueError("unsupported superseded evidence kind")
    _require_equal(
        observed_contrasts,
        SUPERSEDED_CONTRAST_IDS,
        "superseded contrast identities",
    )
    _require_equal(
        observed_proofs,
        SUPERSEDED_COMBINATION_IDS,
        "superseded combination identities",
    )

    identity = _require_mapping(
        document.get("scientific_payload_identity"), "scientific payload identity"
    )
    for field in ("contrast_rows_sha256", "diagnostic_rows_sha256", "summary_sha256"):
        _require_sha(identity.get(field), f"scientific payload {field}")
    _require_equal(
        identity.get("selected_and_superseded_contrast_payloads_identical"),
        True,
        "scientific payload equivalence",
    )
    _require_equal(
        identity.get("selection_used_market_performance"),
        False,
        "market-performance selection",
    )

    execution = _require_mapping(document.get("execution"), "execution")
    _require_equal(execution.get("repo_local_replay_count"), 2, "replay count")
    for field in _ZERO_EXECUTION_FIELDS:
        _require_equal(execution.get(field), 0, f"execution {field}")
    authorities = _require_mapping(document.get("authorities"), "authorities")
    _require_exact_keys(authorities, _FALSE_AUTHORITIES, "authorities")
    for field in _FALSE_AUTHORITIES:
        _require_equal(authorities.get(field), False, f"authority {field}")

    canonical = _canonical_json_bytes(document)
    content_id = hashlib.sha256(canonical).hexdigest()
    _require_equal(content_id, SELECTION_CONTENT_ID, "selection content id")
    return {
        "status": SELECTION_STATUS,
        "selection_content_id": content_id,
        "selected_contrast_content_id": SELECTED_CONTRAST_CONTENT_ID,
        "selected_combination_content_id": SELECTED_COMBINATION_CONTENT_ID,
        "superseded_artifact_count": len(superseded),
        "model_input_authorized": False,
        "production_authorized": False,
    }


def validate_short_tenor_evidence_manifest_chain(
    document: Mapping[str, object],
    manifest_payloads: Mapping[str, bytes],
) -> dict[str, object]:
    """Verify the exact selected and superseded local manifest bytes."""

    selection = validate_short_tenor_evidence_selection(document)
    selected = _require_mapping(document.get("selected_chain"), "selected chain")
    contrast = _require_mapping(selected.get("contrast_bundle"), "contrast bundle")
    proof = _require_mapping(selected.get("combination_proof"), "combination proof")
    superseded = document.get("superseded_evidence")
    assert isinstance(superseded, list)
    expected_sha = {
        str(contrast["content_id"]): str(contrast["manifest_sha256"]),
        str(proof["content_id"]): str(proof["manifest_sha256"]),
    }
    for raw_entry in superseded:
        entry = _require_mapping(raw_entry, "superseded entry")
        expected_sha[str(entry["content_id"])] = str(entry["manifest_sha256"])
    _require_equal(set(manifest_payloads), set(expected_sha), "manifest identity set")

    manifests: dict[str, Mapping[str, Any]] = {}
    for content_id, expected in expected_sha.items():
        payload = manifest_payloads[content_id]
        if not isinstance(payload, bytes) or not payload or len(payload) > 64 * 1024:
            raise ValueError("manifest payload must be non-empty bounded bytes")
        _require_equal(
            hashlib.sha256(payload).hexdigest(), expected, f"manifest hash {content_id}"
        )
        manifest = _parse_json_object(payload, f"manifest {content_id}")
        _require_equal(manifest.get("content_id"), content_id, "manifest content id")
        _require_zero_remote_execution(manifest)
        _require_no_manifest_authority(manifest)
        manifests[content_id] = manifest

    selected_contrast = manifests[SELECTED_CONTRAST_CONTENT_ID]
    _require_equal(
        selected_contrast.get("schema_version"),
        "fmv_eex_ch_short_tenor_contrast_bundle.v1",
        "contrast manifest schema",
    )
    source = _require_mapping(selected_contrast.get("source"), "contrast source")
    _require_equal(
        source.get("normalization_content_id"),
        NORMALIZATION_CONTENT_ID,
        "contrast normalization binding",
    )
    _require_equal(
        source.get("short_tenor_audit_content_id"),
        SHORT_TENOR_AUDIT_CONTENT_ID,
        "contrast audit binding",
    )
    _validate_contrast_artifacts(document, selected_contrast)
    _validate_implementation(contrast, selected_contrast, include_contract=True)

    for content_id in SUPERSEDED_CONTRAST_IDS:
        superseded_manifest = manifests[content_id]
        _validate_contrast_artifacts(document, superseded_manifest)
        superseded_source = _require_mapping(
            superseded_manifest.get("source"), "superseded contrast source"
        )
        _require_equal(
            superseded_source.get("normalization_content_id"),
            NORMALIZATION_CONTENT_ID,
            "superseded contrast normalization binding",
        )
        _require_equal(
            superseded_source.get("short_tenor_audit_content_id"),
            SHORT_TENOR_AUDIT_CONTENT_ID,
            "superseded contrast audit binding",
        )

    selected_proof = manifests[SELECTED_COMBINATION_CONTENT_ID]
    _require_equal(
        selected_proof.get("schema_version"),
        "fmv_eex_ch_short_tenor_combination_proof_bundle.v1",
        "combination manifest schema",
    )
    boundaries = _require_mapping(
        selected_proof.get("source_boundaries"), "combination source boundaries"
    )
    _require_equal(
        boundaries.get("contrast_bundle_content_id"),
        SELECTED_CONTRAST_CONTENT_ID,
        "combination contrast binding",
    )
    _require_equal(
        boundaries.get("combination_policy_content_id"),
        SELECTED_POLICY_CONTENT_ID,
        "combination policy binding",
    )
    _validate_implementation(proof, selected_proof, include_contract=False)
    artifacts = _require_mapping(selected_proof.get("artifacts"), "proof artifacts")
    _require_equal(
        _artifact_hash(artifacts, "summary"),
        proof.get("summary_sha256"),
        "proof summary hash",
    )
    _require_equal(
        _artifact_hash(artifacts, "residuals"),
        proof.get("residuals_sha256"),
        "proof residual hash",
    )

    superseded_by_id = {
        str(_require_mapping(item, "superseded entry")["content_id"]): _require_mapping(
            item, "superseded entry"
        )
        for item in superseded
    }
    for content_id in SUPERSEDED_COMBINATION_IDS:
        old_boundaries = _require_mapping(
            manifests[content_id].get("source_boundaries"),
            "superseded combination source boundaries",
        )
        _require_equal(
            old_boundaries.get("contrast_bundle_content_id"),
            superseded_by_id[content_id].get("bound_source_contrast_id"),
            "superseded combination contrast binding",
        )

    return {
        **selection,
        "manifest_count": len(manifests),
        "manifest_chain_verified": True,
        "databricks_request_count": 0,
        "warehouse_start_count": 0,
    }


def _validate_contrast_artifacts(
    document: Mapping[str, object], manifest: Mapping[str, Any]
) -> None:
    identity = _require_mapping(
        document.get("scientific_payload_identity"), "scientific payload identity"
    )
    artifacts = _require_mapping(manifest.get("artifacts"), "contrast artifacts")
    for artifact, field in (
        ("contrasts", "contrast_rows_sha256"),
        ("diagnostics", "diagnostic_rows_sha256"),
        ("summary", "summary_sha256"),
    ):
        _require_equal(
            _artifact_hash(artifacts, artifact),
            identity.get(field),
            f"contrast artifact {artifact}",
        )


def _validate_implementation(
    expected: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    include_contract: bool,
) -> None:
    implementation = _require_mapping(manifest.get("implementation"), "implementation")
    fields = ["module_sha256", "tests_sha256"]
    if include_contract:
        fields.append("contract_sha256")
    for field in fields:
        _require_equal(
            implementation.get(field), expected.get(field), f"implementation {field}"
        )


def _artifact_hash(artifacts: Mapping[str, Any], name: str) -> object:
    return _require_mapping(artifacts.get(name), f"artifact {name}").get("sha256")


def _require_zero_remote_execution(manifest: Mapping[str, Any]) -> None:
    execution = _require_mapping(manifest.get("execution"), "manifest execution")
    for field in _ZERO_EXECUTION_FIELDS:
        _require_equal(execution.get(field), 0, f"manifest execution {field}")


def _require_no_manifest_authority(manifest: Mapping[str, Any]) -> None:
    authority = _require_mapping(manifest.get("authority"), "manifest authority")
    for field, value in authority.items():
        if field.endswith("authorized") or field.endswith("proven"):
            _require_equal(value, False, f"manifest authority {field}")


def _parse_json_object(payload: bytes, label: str) -> Mapping[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    return _require_mapping(value, label)


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys do not match the frozen contract")


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _require_sha_fields(value: Mapping[str, Any], label: str) -> None:
    for field, item in value.items():
        if field == "content_id" or field.endswith("sha256"):
            _require_sha(item, f"{label} {field}")


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_equal(observed: object, expected: object, label: str) -> None:
    if observed != expected or type(observed) is not type(expected):
        raise ValueError(f"{label} does not match the frozen selection")


__all__ = [
    "SCHEMA_VERSION",
    "SELECTION_CONTENT_ID",
    "SELECTION_DECISION",
    "SELECTION_STATUS",
    "SELECTED_COMBINATION_CONTENT_ID",
    "SELECTED_CONTRAST_CONTENT_ID",
    "SELECTED_POLICY_CONTENT_ID",
    "validate_short_tenor_evidence_manifest_chain",
    "validate_short_tenor_evidence_selection",
]
