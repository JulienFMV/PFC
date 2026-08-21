"""Build an immutable EEX vintage catalog-signing request.

The builder has no network, signing, CAS, calibration, publication, or
production authority.  It turns one independently timestamped EEX workbook
into a deterministic, immutable handoff bundle for separate acquisition and
CAS authorities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path, PurePosixPath

from pfc_shaping.cli.governed_acquisition_builder import (
    _admit_output_directory,
    _materialize_output,
)
from pfc_shaping.data import ingest_forwards
from pfc_shaping.data.eex_forward_vintage_intake import (
    EEX_FORWARD_REVISION_POLICY,
    EEX_FORWARD_SOURCE_CLASS,
    append_eex_forward_vintage_intake,
    canonical_json_bytes,
)
from pfc_shaping.data.eex_historical_vintage import (
    EEX_HISTORICAL_CATALOG_SCHEMA,
    verify_eex_historical_vintage_catalog,
)
from pfc_shaping.durable_artifact import durable_artifact_durability_classification
from pfc_shaping.exit_codes import EXIT_INTEGRITY_FAILURE
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
    register_process_immutable_file_bytes,
)
from pfc_shaping.pipeline.governed_release_cli_contract import (
    ReleaseCliIdentityError,
    absolute_path,
    assert_installed_runtime_sealed,
    canonical_sha256,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json

SIGNING_REQUEST_SCHEMA = "eex_historical_vintage_signing_request.v1"
BUILD_MANIFEST_SCHEMA = "eex_forward_vintage_build.v1"
HISTORY_NAME = "history.parquet"
SIGNING_REQUEST_NAME = "catalog-signing-request.json"
BUILD_MANIFEST_NAME = "eex-vintage-build-manifest.json"
MAX_CATALOG_BYTES = 4 * 1024 * 1024
MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_PARSER_BYTES = 2 * 1024 * 1024
_SIGNING_REQUEST_KEYS = {
    "schema_version",
    "request_id",
    "generated_at_utc",
    "intake_id",
    "previous_catalog",
    "verification_trust",
    "proposed_unsigned_catalog",
    "required_external_actions",
    "local_authority",
}
_SOURCE_DOCUMENT_ENTRY_KEYS = {
    "snapshot_ids",
    "acquisition_id",
    "observed_at",
    "available_at",
    "source_document_id",
    "path",
    "sha256",
    "size_bytes",
    "parser_code_path",
    "parser_code_sha256",
    "parser_config",
    "parser_config_sha256",
    "trusted_time_receipt",
}
_LOCAL_AUTHORITY = {
    "catalog_signed": False,
    "external_cas_admitted": False,
    "calibration_eligible": False,
    "production_authorization": False,
    "promotion_gate": False,
}
_REQUIRED_EXTERNAL_ACTIONS = [
    "INDEPENDENT_ACQUISITION_SIGNATURE",
    "BUILDER_INACCESSIBLE_EXTERNAL_CAS_ADMISSION",
    "INDEPENDENT_PRODUCTION_MANIFEST_GATE",
]


def build_eex_forward_vintage(
    *,
    intake_spec_path: Path,
    expected_spec_sha256: str,
    source_document_path: Path,
    trusted_time_receipt_path: Path,
    trusted_time_public_key_path: Path,
    trusted_time_journal_id: str,
    output_directory: Path,
    previous_catalog_path: Path | None = None,
    previous_history_path: Path | None = None,
    previous_catalog_public_key_path: Path | None = None,
) -> Path:
    """Create or exactly replay one immutable unsigned authority handoff."""

    if (previous_catalog_path is None) != (previous_history_path is None):
        raise ValueError(
            "previous EEX catalog and history paths must be supplied together"
        )
    if previous_catalog_path is not None and previous_catalog_public_key_path is None:
        raise ValueError("previous EEX catalog public key path is required")
    time_key_path, time_key_payload = _load_public_trust_anchor(
        trusted_time_public_key_path,
        label="EEX trusted-time public key",
    )
    journal_id = str(trusted_time_journal_id).strip()
    if not journal_id or len(journal_id) > 128:
        raise ValueError("EEX trusted-time journal ID is invalid")
    acquisition_key_path: Path | None = None
    acquisition_key_payload: bytes | None = None
    if previous_catalog_public_key_path is not None:
        acquisition_key_path, acquisition_key_payload = _load_public_trust_anchor(
            previous_catalog_public_key_path,
            label="previous EEX catalog public key",
        )

    output, output_parent_identity = _admit_output_directory(output_directory)
    artifacts: dict[str, bytes] = {}
    history, result = append_eex_forward_vintage_intake(
        spec_path=intake_spec_path,
        expected_spec_sha256=expected_spec_sha256,
        source_document_path=source_document_path,
        trusted_time_receipt_path=trusted_time_receipt_path,
        previous_catalog_path=previous_catalog_path,
        previous_history_path=previous_history_path,
        trusted_time_public_key_path=time_key_path,
        trusted_time_journal_id=journal_id,
        previous_catalog_public_key_path=acquisition_key_path,
    )
    prior_entries: list[dict[str, object]] = []
    previous_evidence: dict[str, object] | None = None
    if previous_catalog_path is not None and previous_history_path is not None:
        prior_entries, previous_evidence = _archive_previous_catalog_inputs(
            catalog_path=previous_catalog_path,
            history_path=previous_history_path,
            artifacts=artifacts,
            acquisition_public_key_path=acquisition_key_path,
            trusted_time_public_key_path=time_key_path,
            trusted_time_journal_id=journal_id,
        )

    time_key_hash = _sha256(time_key_payload)
    time_key_name = f"trusted-time-public-key-{time_key_hash}.pem"
    _add_artifact(artifacts, time_key_name, time_key_payload)
    verification_trust: dict[str, object] = {
        "trusted_time_public_key_path": time_key_name,
        "trusted_time_public_key_sha256": time_key_hash,
        "trusted_time_journal_id": journal_id,
        "previous_catalog_public_key_path": None,
        "previous_catalog_public_key_sha256": None,
    }
    if acquisition_key_payload is not None:
        acquisition_key_hash = _sha256(acquisition_key_payload)
        acquisition_key_name = (
            f"previous-catalog-public-key-{acquisition_key_hash}.pem"
        )
        _add_artifact(artifacts, acquisition_key_name, acquisition_key_payload)
        verification_trust["previous_catalog_public_key_path"] = acquisition_key_name
        verification_trust["previous_catalog_public_key_sha256"] = acquisition_key_hash

    source_payload = _read_bound_file(
        source_document_path,
        label="EEX source document",
        max_bytes=MAX_SOURCE_BYTES,
        expected_sha256=result.source_document_sha256,
        expected_size=result.source_document_size_bytes,
    )
    source_name = f"source-{result.source_document_sha256}.xlsx"
    _add_artifact(artifacts, source_name, source_payload)

    parser_path = assert_absolute_path_has_no_links(Path(ingest_forwards.__file__).resolve())
    parser_payload = _read_bound_file(
        parser_path,
        label="EEX parser code",
        max_bytes=MAX_PARSER_BYTES,
        expected_sha256=result.parser_code_sha256,
    )
    parser_name = f"parser-{result.parser_code_sha256}.py"
    _add_artifact(artifacts, parser_name, parser_payload)

    new_entry = result.source_catalog_entry(
        source_document_path=source_name,
        parser_code_path=parser_name,
    )
    source_documents = [*prior_entries, new_entry]
    source_documents.sort(
        key=lambda entry: int(
            _mapping(entry.get("trusted_time_receipt"), label="trusted-time receipt")[
                "journal_sequence"
            ]
        )
    )

    history_buffer = BytesIO()
    history.to_parquet(history_buffer, index=False)
    history_payload = history_buffer.getvalue()
    _add_artifact(artifacts, HISTORY_NAME, history_payload)

    proposed_catalog: dict[str, object] = {
        "schema_version": EEX_HISTORICAL_CATALOG_SCHEMA,
        "created_at_utc": result.available_at_utc,
        "data_cutoff_utc": result.available_at_utc,
        "calibration_eligible": True,
        "source_class": EEX_FORWARD_SOURCE_CLASS,
        "revision_policy": EEX_FORWARD_REVISION_POLICY,
        "ompex_used": False,
        "history_parquet": {
            "path": HISTORY_NAME,
            "sha256": _sha256(history_payload),
            "size_bytes": len(history_payload),
            "row_count": len(history),
        },
        "source_documents": source_documents,
    }
    proposed_catalog["catalog_id"] = _sha256(canonical_json_bytes(proposed_catalog))

    signing_request: dict[str, object] = {
        "schema_version": SIGNING_REQUEST_SCHEMA,
        "generated_at_utc": result.available_at_utc,
        "intake_id": result.intake_id,
        "previous_catalog": previous_evidence,
        "verification_trust": verification_trust,
        "proposed_unsigned_catalog": proposed_catalog,
        "required_external_actions": [
            *_REQUIRED_EXTERNAL_ACTIONS,
        ],
        "local_authority": dict(_LOCAL_AUTHORITY),
    }
    signing_request["request_id"] = _sha256(canonical_json_bytes(signing_request))
    validate_eex_forward_vintage_signing_request(signing_request, artifacts=artifacts)
    request_payload = canonical_json_bytes(signing_request)
    _add_artifact(artifacts, SIGNING_REQUEST_NAME, request_payload)

    spec_payload = _read_bound_file(
        intake_spec_path,
        label="EEX intake specification",
        max_bytes=1024 * 1024,
        expected_sha256=expected_spec_sha256,
    )
    _add_artifact(artifacts, "intake-spec.json", spec_payload)
    receipt_payload = _read_bound_file(
        trusted_time_receipt_path,
        label="EEX trusted-time receipt",
        max_bytes=1024 * 1024,
        expected_sha256=result.trusted_time_receipt_sha256,
    )
    _add_artifact(artifacts, "trusted-time-receipt.json", receipt_payload)

    manifest: dict[str, object] = {
        "schema_version": BUILD_MANIFEST_SCHEMA,
        "intake_id": result.intake_id,
        "request_id": signing_request["request_id"],
        "catalog_id": proposed_catalog["catalog_id"],
        "data_cutoff_utc": result.available_at_utc,
        "status": "COMPLETED_UNSIGNED_EXTERNAL_AUTHORITY_HANDOFF",
        "artifacts": {
            name: {"sha256": _sha256(payload), "size_bytes": len(payload)}
            for name, payload in sorted(artifacts.items())
        },
        "new_row_count": result.new_row_count,
        "total_row_count": result.total_row_count,
        "signing_status": "UNSIGNED_REQUIRES_INDEPENDENT_ACQUISITION_AUTHORITY",
        "cas_status": "NOT_ADMITTED_REQUIRES_BUILDER_INACCESSIBLE_EXTERNAL_CAS",
        "calibration_eligible": False,
        "production_authorization": False,
        "promotion_gate": False,
        "durability_classification": durable_artifact_durability_classification(),
    }
    manifest_payload_without_id = canonical_json_bytes(manifest)
    build_id = _sha256(manifest_payload_without_id)
    manifest["build_id"] = build_id
    manifest_payload = canonical_json_bytes(manifest)
    expected = {**artifacts, BUILD_MANIFEST_NAME: manifest_payload}
    return _materialize_output(
        output=output,
        output_parent_identity=output_parent_identity,
        build_id=build_id,
        artifacts=artifacts,
        manifest_payload=manifest_payload,
        expected=expected,
        manifest_name=BUILD_MANIFEST_NAME,
    )


def _archive_previous_catalog_inputs(
    *,
    catalog_path: Path,
    history_path: Path,
    artifacts: dict[str, bytes],
    acquisition_public_key_path: Path | None,
    trusted_time_public_key_path: Path,
    trusted_time_journal_id: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    catalog_file = assert_absolute_path_has_no_links(Path(catalog_path))
    catalog_payload = read_stable_single_link_file(
        catalog_file,
        label="previous EEX vintage catalog",
        max_bytes=MAX_CATALOG_BYTES,
    )
    try:
        catalog = load_strict_json(catalog_payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("previous EEX vintage catalog is invalid") from exc
    if not isinstance(catalog, Mapping):
        raise ValueError("previous EEX vintage catalog must be a mapping")
    _, evidence = verify_eex_historical_vintage_catalog(
        catalog,
        catalog_path=catalog_file,
        history_path=history_path,
        acquisition_public_key_path=acquisition_public_key_path,
        trusted_time_public_key_path=trusted_time_public_key_path,
        trusted_time_journal_id=trusted_time_journal_id,
    )
    raw_entries = catalog.get("source_documents")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("previous EEX source inventory is missing")
    archived: list[dict[str, object]] = []
    for raw_entry in raw_entries:
        entry = dict(_mapping(raw_entry, label="previous EEX source entry"))
        source_hash = str(entry.get("sha256", ""))
        source_size = entry.get("size_bytes")
        source_path = _resolve_relative_catalog_file(
            catalog_file.parent,
            entry.get("path"),
            label="previous EEX source document",
        )
        source_payload = _read_bound_file(
            source_path,
            label="previous EEX source document",
            max_bytes=MAX_SOURCE_BYTES,
            expected_sha256=source_hash,
            expected_size=source_size,
        )
        source_name = f"source-{source_hash}.xlsx"
        _add_artifact(artifacts, source_name, source_payload)
        entry["path"] = source_name

        parser_hash = str(entry.get("parser_code_sha256", ""))
        parser_path = _resolve_relative_catalog_file(
            catalog_file.parent,
            entry.get("parser_code_path"),
            label="previous EEX parser code",
        )
        parser_payload = _read_bound_file(
            parser_path,
            label="previous EEX parser code",
            max_bytes=MAX_PARSER_BYTES,
            expected_sha256=parser_hash,
        )
        parser_name = f"parser-{parser_hash}.py"
        _add_artifact(artifacts, parser_name, parser_payload)
        entry["parser_code_path"] = parser_name
        archived.append(entry)
    return archived, {
        "catalog_id": evidence.catalog_id,
        "catalog_sha256": evidence.catalog_sha256,
        "history_sha256": evidence.history_sha256,
        "data_cutoff_utc": evidence.data_cutoff_utc,
    }


def validate_eex_forward_vintage_signing_request(
    request: Mapping[str, object],
    *,
    artifacts: Mapping[str, bytes],
) -> None:
    """Reject claim smuggling and unbound files in an unsigned handoff."""

    if set(request) != _SIGNING_REQUEST_KEYS:
        raise ValueError("EEX signing request fields are not exact")
    if request.get("schema_version") != SIGNING_REQUEST_SCHEMA:
        raise ValueError("EEX signing request schema is unsupported")
    identity = dict(request)
    request_id = str(identity.pop("request_id", ""))
    if request_id != _sha256(canonical_json_bytes(identity)):
        raise ValueError("EEX signing request_id mismatch")
    if request.get("local_authority") != _LOCAL_AUTHORITY:
        raise ValueError("EEX signing request local authority claims are invalid")
    if request.get("required_external_actions") != _REQUIRED_EXTERNAL_ACTIONS:
        raise ValueError("EEX signing request external actions are invalid")
    proposed = _mapping(
        request.get("proposed_unsigned_catalog"),
        label="proposed EEX catalog",
    )
    expected_catalog_fields = {
        "schema_version",
        "catalog_id",
        "created_at_utc",
        "data_cutoff_utc",
        "calibration_eligible",
        "source_class",
        "revision_policy",
        "ompex_used",
        "history_parquet",
        "source_documents",
    }
    if set(proposed) != expected_catalog_fields:
        raise ValueError("proposed EEX catalog fields are not exact")
    if (
        proposed.get("schema_version") != EEX_HISTORICAL_CATALOG_SCHEMA
        or proposed.get("calibration_eligible") is not True
        or proposed.get("source_class") != EEX_FORWARD_SOURCE_CLASS
        or proposed.get("revision_policy") != EEX_FORWARD_REVISION_POLICY
        or proposed.get("ompex_used") is not False
    ):
        raise ValueError("proposed EEX catalog claims are invalid")
    catalog_identity = dict(proposed)
    catalog_id = str(catalog_identity.pop("catalog_id", ""))
    if catalog_id != _sha256(canonical_json_bytes(catalog_identity)):
        raise ValueError("proposed EEX catalog_id mismatch")
    history = _mapping(proposed.get("history_parquet"), label="EEX history entry")
    if set(history) != {"path", "sha256", "size_bytes", "row_count"}:
        raise ValueError("proposed EEX history fields are not exact")
    _verify_artifact_binding(
        artifacts,
        path=history.get("path"),
        expected_sha256=history.get("sha256"),
        expected_size=history.get("size_bytes"),
        label="proposed EEX history",
    )
    if (
        not isinstance(history.get("row_count"), int)
        or isinstance(history.get("row_count"), bool)
        or history["row_count"] < 1
    ):
        raise ValueError("proposed EEX history row count is invalid")
    documents = proposed.get("source_documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("proposed EEX source inventory is missing")
    for raw in documents:
        document = _mapping(raw, label="proposed EEX source entry")
        if set(document) != _SOURCE_DOCUMENT_ENTRY_KEYS:
            raise ValueError("proposed EEX source entry fields are not exact")
        _verify_artifact_binding(
            artifacts,
            path=document.get("path"),
            expected_sha256=document.get("sha256"),
            expected_size=document.get("size_bytes"),
            label="proposed EEX source document",
        )
        parser_path = document.get("parser_code_path")
        parser_name = str(parser_path)
        parser_payload = artifacts.get(parser_name)
        if parser_payload is None or _sha256(parser_payload) != str(
            document.get("parser_code_sha256", "")
        ):
            raise ValueError("proposed EEX parser artifact binding mismatch")
    trust = _mapping(request.get("verification_trust"), label="EEX verification trust")
    if set(trust) != {
        "trusted_time_public_key_path",
        "trusted_time_public_key_sha256",
        "trusted_time_journal_id",
        "previous_catalog_public_key_path",
        "previous_catalog_public_key_sha256",
    }:
        raise ValueError("EEX verification trust fields are not exact")
    _verify_artifact_binding(
        artifacts,
        path=trust.get("trusted_time_public_key_path"),
        expected_sha256=trust.get("trusted_time_public_key_sha256"),
        expected_size=None,
        label="EEX trusted-time public key",
    )
    previous_path = trust.get("previous_catalog_public_key_path")
    previous_hash = trust.get("previous_catalog_public_key_sha256")
    if (previous_path is None) != (previous_hash is None):
        raise ValueError("previous EEX catalog trust binding is incomplete")
    if previous_path is not None:
        _verify_artifact_binding(
            artifacts,
            path=previous_path,
            expected_sha256=previous_hash,
            expected_size=None,
            label="previous EEX catalog public key",
        )
    previous_catalog = request.get("previous_catalog")
    if (previous_catalog is None) != (previous_path is None):
        raise ValueError("previous EEX catalog evidence/trust mismatch")
    if previous_catalog is not None and set(
        _mapping(previous_catalog, label="previous EEX catalog evidence")
    ) != {"catalog_id", "catalog_sha256", "history_sha256", "data_cutoff_utc"}:
        raise ValueError("previous EEX catalog evidence fields are not exact")


def _verify_artifact_binding(
    artifacts: Mapping[str, bytes],
    *,
    path: object,
    expected_sha256: object,
    expected_size: object | None,
    label: str,
) -> None:
    name = str(path)
    if not name or Path(name).name != name:
        raise ValueError(f"{label} path is invalid")
    payload = artifacts.get(name)
    if payload is None or _sha256(payload) != str(expected_sha256):
        raise ValueError(f"{label} artifact binding mismatch")
    if expected_size is not None and (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or len(payload) != expected_size
    ):
        raise ValueError(f"{label} artifact size mismatch")


def _resolve_relative_catalog_file(root: Path, raw: object, *, label: str) -> Path:
    value = str(raw)
    portable = PurePosixPath(value)
    if (
        not value
        or portable.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in portable.parts)
    ):
        raise ValueError(f"{label} path is not portable and relative")
    selected = assert_absolute_path_has_no_links(root.joinpath(*portable.parts))
    try:
        selected.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the catalog root") from exc
    return selected


def _read_bound_file(
    path: Path,
    *,
    label: str,
    max_bytes: int,
    expected_sha256: object,
    expected_size: object | None = None,
) -> bytes:
    selected = assert_absolute_path_has_no_links(Path(path))
    payload = read_stable_single_link_file(selected, label=label, max_bytes=max_bytes)
    if _sha256(payload) != str(expected_sha256):
        raise ValueError(f"{label} SHA-256 mismatch")
    if expected_size is not None and (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or len(payload) != expected_size
    ):
        raise ValueError(f"{label} size mismatch")
    return payload


def _load_public_trust_anchor(
    path: Path,
    *,
    label: str,
) -> tuple[Path, bytes]:
    selected = assert_absolute_path_has_no_links(Path(path))
    payload = read_stable_single_link_file(selected, label=label, max_bytes=64 * 1024)
    if not payload:
        raise ValueError(f"{label} is empty")
    register_process_immutable_file_bytes(selected, payload, label=label)
    return selected, payload


def _add_artifact(artifacts: dict[str, bytes], name: str, payload: bytes) -> None:
    if not name or Path(name).name != name:
        raise ValueError("EEX signing-request artifact name is invalid")
    previous = artifacts.get(name)
    if previous is not None and previous != payload:
        raise ValueError("EEX signing-request artifact identity collision")
    artifacts[name] = payload


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-receipt", type=absolute_path, required=True)
    parser.add_argument(
        "--expected-runtime-receipt-sha256",
        type=canonical_sha256,
        required=True,
    )
    parser.add_argument("--intake-spec", type=Path, required=True)
    parser.add_argument("--expected-spec-sha256", required=True)
    parser.add_argument("--source-document", type=Path, required=True)
    parser.add_argument("--trusted-time-receipt", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--previous-catalog", type=Path)
    parser.add_argument("--previous-history", type=Path)
    parser.add_argument("--trusted-time-public-key", type=Path, required=True)
    parser.add_argument("--trusted-time-journal-id", required=True)
    parser.add_argument("--previous-catalog-public-key", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        assert_installed_runtime_sealed(
            runtime_receipt_path=args.runtime_receipt,
            expected_runtime_receipt_sha256=args.expected_runtime_receipt_sha256,
        )
    except ReleaseCliIdentityError as exc:
        _emit_failure(exc, retry_disposition="DO_NOT_RETRY_UNSEALED_RUNTIME")
        return EXIT_INTEGRITY_FAILURE
    try:
        manifest = build_eex_forward_vintage(
            intake_spec_path=args.intake_spec,
            expected_spec_sha256=args.expected_spec_sha256,
            source_document_path=args.source_document,
            trusted_time_receipt_path=args.trusted_time_receipt,
            output_directory=args.output_directory,
            previous_catalog_path=args.previous_catalog,
            previous_history_path=args.previous_history,
            trusted_time_public_key_path=args.trusted_time_public_key,
            trusted_time_journal_id=args.trusted_time_journal_id,
            previous_catalog_public_key_path=args.previous_catalog_public_key,
        )
    except (OSError, ValueError) as exc:
        _emit_failure(exc, retry_disposition="QUARANTINE_OR_EXACT_RETRY_AFTER_REPAIR")
        return EXIT_INTEGRITY_FAILURE
    print(manifest)
    return 0


def _emit_failure(exc: Exception, *, retry_disposition: str) -> None:
    print(
        json.dumps(
            {
                "command": "python -I -B -m pfc_shaping.cli.eex_forward_vintage_builder",
                "status": "INTEGRITY_FAILURE",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "retry_disposition": retry_disposition,
                "catalog_signed": False,
                "external_cas_admitted": False,
                "production_authorization": False,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
