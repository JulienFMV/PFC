"""Fail-closed local capture for one official EEX DataSource REST API v2 query.

The capture preserves provider bytes and transport metadata for a later
trusted-time, acquisition-signature and external-CAS handoff.  It never turns
workstation time or a successful HTTPS request into scientific authority.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
import re
import ssl
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote_to_bytes, urlencode

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization

from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json

if os.name == "nt":
    import msvcrt
else:  # pragma: no cover - exercised by non-Windows CI only
    import fcntl

SPEC_SCHEMA = "eex_datasource_v2_capture_spec.v1"
MANIFEST_SCHEMA = "eex_datasource_v2_local_capture.v1"
ATTEMPT_SCHEMA = "eex_datasource_v2_local_capture_attempt.v1"
FAILURE_SCHEMA = "eex_datasource_v2_local_capture_failure.v1"
SPEC_STATUS = "FROZEN_EEX_DATASOURCE_V2_CAPTURE_SPEC_NO_AUTHORITY"
CAPTURE_STATUS = "LOCAL_EEX_DATASOURCE_V2_CAPTURE_NOT_PIT_AUTHORITY"
ATTEMPT_STATUS = "ATTEMPTED_LOCAL_EEX_DATASOURCE_V2_CAPTURE"
FAILURE_STATUS = "TERMINAL_LOCAL_EEX_DATASOURCE_V2_CAPTURE_FAILED_NO_AUTHORITY"
API_ROOT = "https://api.eex-group.com/v2"
TOKEN_ENV = "PFC_EEX_DATASOURCE_ACCESS_TOKEN"
GUIDE_RELEASE = "006"
GUIDE_PUBLISHED_DATE = "2026-07-16"
GUIDE_SHA256 = "d24cc35c7600622cba44d00dd988045be20ec01ad325bbb85552626bfcc7ad81"
# Empty until Security/IT independently approves one exact egress trust anchor.
# A request spec is never allowed to select trust by itself.
APPROVED_TLS_CA_SHA256S: frozenset[str] = frozenset()

_MAX_SPEC_BYTES = 512 * 1024
_MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_PROVIDER_RECORD_LIMIT = 60_000
_MIN_REQUEST_INTERVAL_SECONDS = 1.0
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PORTABLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHORT_CODE = re.compile(r"[A-Z0-9]{2,16}")
_MATURITY = re.compile(r"[0-9]{6}")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_ISIN = re.compile(r"[A-Z]{2}[A-Z0-9]{9}[0-9]")
_SAFE_QUERY_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}")
_ACCESS_TOKEN = re.compile(r"[A-Za-z0-9._~+/=-]{16,4096}")
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_SENSITIVE_QUERY_NAMES = frozenset(
    {"authorization", "access_token", "token", "api_key", "apikey", "password", "secret"}
)
_AUTHORITY_FIELDS = (
    "scientific_admission",
    "monthly_solver_hard_level_eligible",
    "training_eligible",
    "model_selection_eligible",
    "holdout_eligible",
    "candidate_assembly_eligible",
    "publication_authorized",
    "production_authorization",
    "promotion_gate",
    "future_holdout_consumed",
    "t057_consumed",
)
_SPEC_KEYS = {
    "schema_version",
    "spec_id",
    "status",
    "capture_id",
    "planned_at_utc",
    "request",
    "documentation",
    "tls",
    "product_identity",
    "reference_capture",
    "execution",
    "local_capture_execution_authorized",
    *_AUTHORITY_FIELDS,
}
_REQUEST_KEYS = {
    "method",
    "api_root",
    "api_version",
    "data_role",
    "endpoint_path",
    "query",
    "accept",
    "commodity",
    "area",
    "rolling_symbols_allowed",
}
_DOCUMENTATION_KEYS = {
    "user_guide_release",
    "user_guide_published_date",
    "user_guide_path",
    "user_guide_sha256",
    "openapi_path",
    "openapi_sha256",
}
_PRODUCT_KEYS = {"trade_date", "short_code", "maturity", "instrument_isin"}
_TLS_KEYS = {"ca_bundle_path", "ca_bundle_sha256", "minimum_tls_version"}
_REFERENCE_KEYS = {"manifest_path", "manifest_sha256", "response_sha256"}
_EXECUTION_KEYS = {
    "access_token_env_var",
    "timeout_seconds",
    "max_response_bytes",
    "records_path",
}
_RESPONSE_HEADER_ALLOWLIST = frozenset(
    {
        "cache-control",
        "content-encoding",
        "content-length",
        "content-type",
        "date",
        "etag",
        "last-modified",
        "retry-after",
        "x-correlation-id",
        "x-request-id",
    }
)
_NEXT_REQUIRED_AUTHORITIES = (
    "INDEPENDENT_TRUSTED_TIME_RECEIPT",
    "INDEPENDENT_ACQUISITION_SIGNATURE",
    "BUILDER_INACCESSIBLE_IMMUTABLE_COPY",
    "EXTERNAL_CAS_WORM_RECEIPT_AND_FRESH_HEAD",
    "FMV_LICENSE_AND_ENTITLEMENT_ATTESTATION",
    "INDEPENDENT_OPENAPI_RESPONSE_CONFORMANCE",
    "SECURITY_APPROVED_TLS_INSPECTION_AND_TOKEN_HANDLING_OR_GOVERNED_DIRECT_EGRESS",
)
_NEGATIVE_EVIDENCE_FIELDS = (
    "trusted_time_receipt_present",
    "independent_acquisition_signature_present",
    "builder_inaccessible_immutable_copy_present",
    "external_cas_admitted",
    "source_license_attested",
    "provider_direct_identity_attested",
)
_MANIFEST_KEYS = {
    "schema_version",
    "status",
    "capture_id",
    "spec",
    "source_system",
    "request",
    "product_identity",
    "documentation",
    "tls_trust",
    "reference_capture",
    "capture_started_utc_untrusted",
    "capture_completed_utc_untrusted",
    "workstation_clock_trusted",
    "transport",
    "response",
    *_NEGATIVE_EVIDENCE_FIELDS,
    *_AUTHORITY_FIELDS,
    "next_required_authorities",
}


class EexDatasourceV2CaptureError(ValueError):
    """Raised when an official v2 request cannot be captured unambiguously."""


@dataclass(frozen=True)
class ProductIdentity:
    trade_date: str
    short_code: str
    maturity: str
    instrument_isin: str


@dataclass(frozen=True)
class VerifiedCaptureSpec:
    capture_id: str
    spec_id: str
    spec_sha256: str
    spec_path: Path
    planned_at_utc: str
    data_role: str
    endpoint_path: str
    query: tuple[tuple[str, str], ...]
    product_identity: ProductIdentity
    user_guide_path: Path
    user_guide_sha256: str
    openapi_path: Path
    openapi_sha256: str
    ca_bundle_path: Path
    ca_bundle_sha256: str
    ca_bundle_bytes: bytes
    ca_bundle_encoding: str
    reference_capture: Mapping[str, str] | None
    timeout_seconds: int
    max_response_bytes: int
    records_path: tuple[str, ...]

    @property
    def request_target(self) -> str:
        query = urlencode(self.query, quote_via=quote, safe="")
        return self.endpoint_path if not query else f"{self.endpoint_path}?{query}"


@dataclass(frozen=True)
class HttpsExchange:
    status: int
    reason: str
    headers: tuple[tuple[str, str], ...]
    body: bytes
    tls_version: str
    tls_cipher: str
    peer_leaf_certificate_sha256: str


def canonical_json_bytes(value: object) -> bytes:
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


def compute_spec_id(document: Mapping[str, object]) -> str:
    identity = dict(document)
    identity.pop("spec_id", None)
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def load_eex_datasource_v2_capture_spec(
    *,
    repo_root: str | Path,
    spec_path: str | Path,
    expected_spec_sha256: str,
) -> VerifiedCaptureSpec:
    """Verify one caller-hash-bound request specification and all local inputs."""

    root = _root(repo_root)
    _require_sha256(expected_spec_sha256, label="caller-held capture spec")
    selected_spec = _repo_file(
        root, spec_path, label="capture spec", allow_absolute=True
    )
    payload = read_stable_single_link_file(
        selected_spec,
        label="EEX DataSource v2 capture spec",
        max_bytes=_MAX_SPEC_BYTES,
    )
    if hashlib.sha256(payload).hexdigest() != expected_spec_sha256:
        raise EexDatasourceV2CaptureError("capture spec SHA-256 mismatch")
    document = _strict_mapping(payload, label="capture spec")
    if canonical_json_bytes(document) != payload:
        raise EexDatasourceV2CaptureError("capture spec is not canonical JSON")
    _exact_keys(document, _SPEC_KEYS, label="capture spec")
    _equal(document.get("schema_version"), SPEC_SCHEMA, label="schema version")
    _equal(document.get("status"), SPEC_STATUS, label="spec status")
    capture_id = _portable_id(document.get("capture_id"), label="capture ID")
    spec_id = _require_sha256(document.get("spec_id"), label="spec ID")
    if compute_spec_id(document) != spec_id:
        raise EexDatasourceV2CaptureError("spec ID does not bind content")
    planned_at = _utc(document.get("planned_at_utc"), label="planned-at")
    if document.get("local_capture_execution_authorized") is not True:
        raise EexDatasourceV2CaptureError("local capture execution is not authorized")
    for field in _AUTHORITY_FIELDS:
        if document.get(field) is not False:
            raise EexDatasourceV2CaptureError(f"capture spec {field} must be false")

    request = _mapping(document.get("request"), label="request")
    _exact_keys(request, _REQUEST_KEYS, label="request")
    _equal(request.get("method"), "GET", label="HTTP method")
    _equal(request.get("api_root"), API_ROOT, label="API root")
    _equal(request.get("api_version"), "v2", label="API version")
    _equal(request.get("accept"), "application/json", label="response media type")
    _equal(request.get("commodity"), "POWER", label="commodity")
    _equal(request.get("area"), "CH", label="market area")
    if request.get("rolling_symbols_allowed") is not False:
        raise EexDatasourceV2CaptureError("rolling symbols must be disabled")
    role = str(request.get("data_role", ""))
    if role not in {"REFERENCE_DATA", "SETTLEMENT"}:
        raise EexDatasourceV2CaptureError("data role is unsupported")
    endpoint = _endpoint_path(request.get("endpoint_path"), role=role)
    query = _query_pairs(request.get("query"))

    documentation = _mapping(document.get("documentation"), label="documentation")
    _exact_keys(documentation, _DOCUMENTATION_KEYS, label="documentation")
    _equal(documentation.get("user_guide_release"), GUIDE_RELEASE, label="guide release")
    _equal(
        documentation.get("user_guide_published_date"),
        GUIDE_PUBLISHED_DATE,
        label="guide publication date",
    )
    guide_hash = _require_sha256(
        documentation.get("user_guide_sha256"), label="user guide"
    )
    if guide_hash != GUIDE_SHA256:
        raise EexDatasourceV2CaptureError("user guide is not the pinned release 006")
    guide_path = _verified_repo_document(
        root,
        documentation.get("user_guide_path"),
        expected_sha256=guide_hash,
        label="EEX REST v2 user guide",
    )
    openapi_hash = _require_sha256(
        documentation.get("openapi_sha256"), label="OpenAPI specification"
    )
    openapi_path = _verified_repo_document(
        root,
        documentation.get("openapi_path"),
        expected_sha256=openapi_hash,
        label="EEX REST v2 OpenAPI specification",
    )

    tls = _mapping(document.get("tls"), label="TLS")
    _exact_keys(tls, _TLS_KEYS, label="TLS")
    _equal(tls.get("minimum_tls_version"), "TLSv1.2", label="minimum TLS version")
    ca_hash = _require_sha256(tls.get("ca_bundle_sha256"), label="TLS CA bundle")
    ca_path = _repo_file(root, tls.get("ca_bundle_path"), label="TLS CA bundle")
    ca_bytes = read_stable_single_link_file(
        ca_path,
        label="TLS CA bundle",
        max_bytes=4 * 1024 * 1024,
    )
    if hashlib.sha256(ca_bytes).hexdigest() != ca_hash:
        raise EexDatasourceV2CaptureError("TLS CA bundle SHA-256 mismatch")
    if ca_hash not in APPROVED_TLS_CA_SHA256S:
        raise EexDatasourceV2CaptureError(
            "TLS CA bundle is not admitted by the independent security policy"
        )
    ca_encoding = _validate_ca_bundle(ca_bytes, planned_at_utc=planned_at)

    identity_doc = _mapping(document.get("product_identity"), label="product identity")
    _exact_keys(identity_doc, _PRODUCT_KEYS, label="product identity")
    identity = ProductIdentity(
        trade_date=_matched(identity_doc.get("trade_date"), _DATE, label="trade date"),
        short_code=_matched(identity_doc.get("short_code"), _SHORT_CODE, label="short code"),
        maturity=_matched(identity_doc.get("maturity"), _MATURITY, label="maturity"),
        instrument_isin=_matched(
            identity_doc.get("instrument_isin"), _ISIN, label="instrument ISIN"
        ),
    )
    endpoint_parts = endpoint[1:].split("/")
    product_parts = [
        "derivatives",
        "POWER",
        "CH",
        identity.trade_date,
        identity.short_code,
        identity.maturity,
    ]
    expected_parts = (
        ["rd", *product_parts]
        if role == "REFERENCE_DATA"
        else [endpoint_parts[0], *product_parts]
    )
    if endpoint_parts != expected_parts:
        raise EexDatasourceV2CaptureError(
            "endpoint does not bind the exact product identity segments"
        )

    reference_value = document.get("reference_capture")
    reference: Mapping[str, str] | None
    if role == "REFERENCE_DATA":
        if reference_value is not None:
            raise EexDatasourceV2CaptureError("reference-data capture cannot cite itself")
        reference = None
    else:
        reference = _verify_reference_capture(
            root,
            reference_value,
            product_identity=identity,
        )

    execution = _mapping(document.get("execution"), label="execution")
    _exact_keys(execution, _EXECUTION_KEYS, label="execution")
    _equal(
        execution.get("access_token_env_var"), TOKEN_ENV, label="access-token environment"
    )
    timeout = _bounded_int(execution.get("timeout_seconds"), 1, 120, label="timeout")
    response_limit = _bounded_int(
        execution.get("max_response_bytes"),
        1,
        _MAX_RESPONSE_BYTES,
        label="response byte limit",
    )
    records_path = _records_path(execution.get("records_path"))
    return VerifiedCaptureSpec(
        capture_id=capture_id,
        spec_id=spec_id,
        spec_sha256=expected_spec_sha256,
        spec_path=selected_spec,
        planned_at_utc=planned_at,
        data_role=role,
        endpoint_path=endpoint,
        query=query,
        product_identity=identity,
        user_guide_path=guide_path,
        user_guide_sha256=guide_hash,
        openapi_path=openapi_path,
        openapi_sha256=openapi_hash,
        ca_bundle_path=ca_path,
        ca_bundle_sha256=ca_hash,
        ca_bundle_bytes=ca_bytes,
        ca_bundle_encoding=ca_encoding,
        reference_capture=reference,
        timeout_seconds=timeout,
        max_response_bytes=response_limit,
        records_path=records_path,
    )


def capture_eex_datasource_v2(
    *,
    repo_root: str | Path,
    spec_path: str | Path,
    expected_spec_sha256: str,
    environment: Mapping[str, str] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> Path:
    """Execute one exact GET and return its local non-authoritative manifest."""

    root = _root(repo_root)
    spec = load_eex_datasource_v2_capture_spec(
        repo_root=root,
        spec_path=spec_path,
        expected_spec_sha256=expected_spec_sha256,
    )
    token = str((environment if environment is not None else os.environ).get(TOKEN_ENV, ""))
    if _ACCESS_TOKEN.fullmatch(token) is None:
        raise EexDatasourceV2CaptureError(
            f"{TOKEN_ENV} must contain one opaque printable access token"
        )
    spec_payload = read_stable_single_link_file(
        spec.spec_path,
        label="EEX DataSource v2 capture spec",
        max_bytes=_MAX_SPEC_BYTES,
    )
    if hashlib.sha256(spec_payload).hexdigest() != spec.spec_sha256:
        raise EexDatasourceV2CaptureError(
            "capture spec changed before credential material scan"
        )
    if _contains_credential(spec.request_target.encode("ascii"), token=token) or (
        _contains_credential(spec_payload, token=token)
    ):
        raise EexDatasourceV2CaptureError(
            "capture spec or request target contains credential material"
        )
    _reject_decoded_credential(
        _strict_mapping(spec_payload, label="capture spec"), token=token
    )
    now = clock or (lambda: datetime.now(timezone.utc))
    started = _clock(now, label="capture start")
    capture_root = root / "build" / "eex-datasource-v2-captures"
    capture_root.mkdir(exist_ok=True)
    capture_root = assert_absolute_path_has_no_links(capture_root)
    output = capture_root / spec.capture_id
    staging = capture_root / f"_incomplete-{spec.capture_id}"
    if output.exists():
        raise EexDatasourceV2CaptureError("capture output already exists")
    staging.mkdir(exist_ok=False)
    attempt = {
        "schema_version": ATTEMPT_SCHEMA,
        "status": ATTEMPT_STATUS,
        "capture_id": spec.capture_id,
        "spec_id": spec.spec_id,
        "spec_sha256": spec.spec_sha256,
        "capture_started_utc_untrusted": _iso(started),
        "access_token_source": f"ENV:{TOKEN_ENV}",
        "access_token_persisted": False,
        "scientific_admission": False,
        "production_authorization": False,
        "promotion_gate": False,
    }
    _write_json(staging / "capture-attempt.json", attempt, label="EEX v2 capture attempt")

    phase = "RATE_LIMIT_ADMISSION"
    exchange: HttpsExchange | None = None
    try:
        with _request_rate_slot(capture_root, timeout_seconds=spec.timeout_seconds):
            phase = "HTTPS_REQUEST"
            exchange = _https_transport(spec, token)
        phase = "CREDENTIAL_ECHO_VALIDATION"
        _reject_credential_echo(exchange, token=token)
        phase = "RESPONSE_HEADER_VALIDATION"
        response_headers = _validated_response_headers(exchange.headers)
        phase = "RESPONSE_CONTRACT_VALIDATION"
        records, parsed = _validate_response(spec, exchange, response_headers)
        _reject_decoded_credential(parsed, token=token)
        completed = _clock(now, label="capture completion")
        phase = "RESPONSE_DURABLE_WRITE"
        body_path = write_durable_exact_bytes(
            staging / "response.json", exchange.body, label="EEX v2 exact response"
        )
        headers_document = {
            "schema_version": "eex_datasource_v2_response_headers.v1",
            "status_code": exchange.status,
            "reason": exchange.reason,
            "headers": response_headers,
        }
        headers_path = _write_json(
            staging / "response-headers.json",
            headers_document,
            label="EEX v2 response headers",
        )
        phase = "INPUT_FINALIZATION_REVERIFY"
        _reverify_capture_inputs(spec)
    except Exception as exc:
        _write_capture_failure_receipt(
            staging,
            spec=spec,
            started=started,
            phase=phase,
            exchange=exchange,
            error=exc,
        )
        raise
    response_hash = hashlib.sha256(exchange.body).hexdigest()
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": CAPTURE_STATUS,
        "capture_id": spec.capture_id,
        "spec": {
            "path": _relative(root, spec.spec_path),
            "spec_id": spec.spec_id,
            "sha256": spec.spec_sha256,
        },
        "source_system": "EEX_GROUP_DATASOURCE_REST_API_V2",
        "request": {
            "method": "GET",
            "api_root": API_ROOT,
            "request_target": spec.request_target,
            "data_role": spec.data_role,
            "commodity": "POWER",
            "area": "CH",
            "authorization_scheme": "BEARER_ENVIRONMENT_ONLY",
            "access_token_persisted": False,
        },
        "product_identity": {
            "trade_date": spec.product_identity.trade_date,
            "short_code": spec.product_identity.short_code,
            "maturity": spec.product_identity.maturity,
            "instrument_isin": spec.product_identity.instrument_isin,
        },
        "documentation": {
            "user_guide_release": GUIDE_RELEASE,
            "user_guide_published_date": GUIDE_PUBLISHED_DATE,
            "user_guide_path": _relative(root, spec.user_guide_path),
            "user_guide_sha256": spec.user_guide_sha256,
            "openapi_path": _relative(root, spec.openapi_path),
            "openapi_sha256": spec.openapi_sha256,
        },
        "tls_trust": {
            "ca_bundle_path": _relative(root, spec.ca_bundle_path),
            "ca_bundle_sha256": spec.ca_bundle_sha256,
            "ca_bundle_encoding": spec.ca_bundle_encoding,
            "minimum_tls_version": "TLSv1.2",
            "ambient_windows_certificate_store_used": False,
            "ca_loaded_from_hash_verified_captured_bytes": True,
            "tls_key_logging_disabled": True,
            "trust_classification": (
                "CALLER_PINNED_ROOT_TLS_SERVER_AUTHENTICATION_NOT_DIRECT_PROVIDER_IDENTITY"
            ),
            "tls_interception_security_approved": False,
        },
        "reference_capture": dict(spec.reference_capture) if spec.reference_capture else None,
        "capture_started_utc_untrusted": _iso(started),
        "capture_completed_utc_untrusted": _iso(completed),
        "workstation_clock_trusted": False,
        "transport": {
            "http_status": exchange.status,
            "response_headers_path": _relative(staging, headers_path),
            "response_headers_sha256": hashlib.sha256(
                canonical_json_bytes(headers_document)
            ).hexdigest(),
            "tls_version": exchange.tls_version,
            "tls_cipher": exchange.tls_cipher,
            "peer_leaf_certificate_sha256": _require_sha256(
                exchange.peer_leaf_certificate_sha256,
                label="peer leaf certificate",
            ),
            "provider_date_header": response_headers["date"],
            "provider_date_header_trusted_time": False,
        },
        "response": {
            "path": _relative(staging, body_path),
            "sha256": response_hash,
            "size_bytes": len(exchange.body),
            "strict_json": True,
            "top_level_type": type(parsed).__name__,
            "records_path": list(spec.records_path),
            "record_count": len(records),
            "provider_record_limit": _PROVIDER_RECORD_LIMIT,
            "provider_limit_reached": False,
            "corrections_preserved_without_collapse": spec.data_role == "SETTLEMENT",
            "settlement_revision_ordering": (
                "TM_CHRONOLOGY_VALIDATED_RAW_PROVIDER_ORDER_PRESERVED"
                if spec.data_role == "SETTLEMENT"
                else "NOT_APPLICABLE"
            ),
        },
        "trusted_time_receipt_present": False,
        "independent_acquisition_signature_present": False,
        "builder_inaccessible_immutable_copy_present": False,
        "external_cas_admitted": False,
        "source_license_attested": False,
        "provider_direct_identity_attested": False,
        "scientific_admission": False,
        "monthly_solver_hard_level_eligible": False,
        "training_eligible": False,
        "model_selection_eligible": False,
        "holdout_eligible": False,
        "candidate_assembly_eligible": False,
        "publication_authorized": False,
        "production_authorization": False,
        "promotion_gate": False,
        "future_holdout_consumed": False,
        "t057_consumed": False,
        "next_required_authorities": list(_NEXT_REQUIRED_AUTHORITIES),
    }
    try:
        _write_json(
            staging / "capture-manifest.json", manifest, label="EEX v2 capture manifest"
        )
        if output.exists():
            raise EexDatasourceV2CaptureError(
                "capture output appeared before atomic finalization"
            )
        os.rename(staging, output)
    except Exception as exc:
        _write_capture_failure_receipt(
            staging,
            spec=spec,
            started=started,
            phase="ATOMIC_FINALIZATION",
            exchange=exchange,
            error=exc,
        )
        raise
    return output / "capture-manifest.json"


def _write_capture_failure_receipt(
    staging: Path,
    *,
    spec: VerifiedCaptureSpec,
    started: datetime,
    phase: str,
    exchange: HttpsExchange | None,
    error: Exception,
) -> Path:
    failed = datetime.now(timezone.utc)
    http_status = exchange.status if exchange is not None else None
    headers_safe = phase != "CREDENTIAL_ECHO_VALIDATION"
    retry_after = (
        _safe_failure_header(exchange, "retry-after")
        if exchange is not None and headers_safe
        else None
    )
    correlation_id = None
    if exchange is not None and headers_safe:
        correlation_id = _safe_failure_header(exchange, "x-correlation-id")
        if correlation_id is None:
            correlation_id = _safe_failure_header(exchange, "x-request-id")
    failure_class, retry_disposition = _failure_classification(
        phase=phase,
        http_status=http_status,
        error=error,
    )
    attempt_path = staging / "capture-attempt.json"
    attempt_bytes = read_stable_single_link_file(
        attempt_path,
        label="EEX v2 capture attempt",
        max_bytes=_MAX_SPEC_BYTES,
    )
    receipt = {
        "schema_version": FAILURE_SCHEMA,
        "status": FAILURE_STATUS,
        "capture_id": spec.capture_id,
        "spec_id": spec.spec_id,
        "spec_sha256": spec.spec_sha256,
        "supersedes_attempt_path": "capture-attempt.json",
        "supersedes_attempt_sha256": hashlib.sha256(attempt_bytes).hexdigest(),
        "capture_started_utc_untrusted": _iso(started),
        "capture_failed_utc_untrusted": _iso(failed),
        "duration_milliseconds_untrusted": max(
            0, int((failed - started).total_seconds() * 1000)
        ),
        "terminal_for_capture_id": True,
        "failure_phase": phase,
        "failure_class": failure_class,
        "http_status": http_status,
        "retry_after_observed": retry_after,
        "correlation_id_observed": correlation_id,
        "response_bytes_observed": len(exchange.body) if exchange is not None else 0,
        "retry_disposition": retry_disposition,
        "automatic_retry_attempted": False,
        "same_capture_id_retry_allowed": False,
        "new_hash_bound_spec_and_capture_id_required": True,
        "staging_directory_finalized": False,
        "process_crash_indeterminate": False,
        "access_token_persisted": False,
        "scientific_admission": False,
        "monthly_solver_hard_level_eligible": False,
        "training_eligible": False,
        "model_selection_eligible": False,
        "holdout_eligible": False,
        "candidate_assembly_eligible": False,
        "publication_authorized": False,
        "production_authorization": False,
        "promotion_gate": False,
        "future_holdout_consumed": False,
        "t057_consumed": False,
    }
    return _write_json(
        staging / "capture-failure.json",
        receipt,
        label="EEX v2 terminal capture failure",
    )


def _failure_classification(
    *, phase: str, http_status: int | None, error: Exception
) -> tuple[str, str]:
    if http_status in {401, 403}:
        return (
            "AUTHENTICATION_OR_ENTITLEMENT_REJECTED",
            "NO_RETRY_PENDING_CREDENTIAL_OR_ENTITLEMENT_REVIEW",
        )
    if http_status == 429:
        return (
            "PROVIDER_RATE_LIMITED",
            "NEW_ID_ONLY_AFTER_RETRY_AFTER_AND_OPERATOR_REVIEW",
        )
    if http_status is not None and 500 <= http_status <= 599:
        return "PROVIDER_SERVER_ERROR", "NEW_ID_ONLY_AFTER_BOUNDED_BACKOFF"
    if phase == "RATE_LIMIT_ADMISSION":
        return "LOCAL_RATE_LIMIT_UNAVAILABLE", "NO_RETRY_PENDING_LOCK_INVENTORY"
    if phase == "HTTPS_REQUEST":
        return "HTTPS_TRANSPORT_FAILED", "NEW_ID_ONLY_AFTER_TRANSIENT_REVIEW"
    if phase == "CREDENTIAL_ECHO_VALIDATION":
        return "CREDENTIAL_ECHO_DETECTED", "NO_RETRY_PENDING_SECURITY_REVIEW"
    if isinstance(error, (OSError, ssl.SSLError, http.client.HTTPException)):
        return "LOCAL_IO_FAILURE", "NO_RETRY_PENDING_OPERATOR_REVIEW"
    return "CONTRACT_OR_FINALIZATION_FAILED", "NO_RETRY_PENDING_EVIDENCE_REVIEW"


def _safe_failure_header(exchange: HttpsExchange, name: str) -> str | None:
    values = [value.strip() for key, value in exchange.headers if key.lower() == name]
    if len(values) != 1:
        return None
    value = values[0]
    if not value or len(value) > 256 or any(ord(character) < 0x20 for character in value):
        return None
    if name == "retry-after":
        if value.isdigit() and int(value) <= 86_400:
            return value
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        return value if parsed.tzinfo is not None else None
    return value if re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", value) else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--expected-spec-sha256", required=True)
    arguments = parser.parse_args(argv)
    try:
        result = capture_eex_datasource_v2(
            repo_root=arguments.repo_root,
            spec_path=arguments.spec,
            expected_spec_sha256=arguments.expected_spec_sha256,
        )
    except (EexDatasourceV2CaptureError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "LOCAL_EEX_DATASOURCE_V2_CAPTURE_FAILED_NOT_AUTHORITY",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "production_authorization": False,
                    "promotion_gate": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(result)
    return 0


def _https_transport(spec: VerifiedCaptureSpec, token: str) -> HttpsExchange:
    ca_data = (
        spec.ca_bundle_bytes.decode("ascii")
        if spec.ca_bundle_encoding == "PEM"
        else x509.load_der_x509_certificate(spec.ca_bundle_bytes)
        .public_bytes(serialization.Encoding.PEM)
        .decode("ascii")
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.keylog_filename = None
    context.load_verify_locations(cadata=ca_data)
    connection = http.client.HTTPSConnection(
        "api.eex-group.com", 443, timeout=spec.timeout_seconds, context=context
    )
    response: http.client.HTTPResponse | None = None
    try:
        connection.request(
            "GET",
            spec.request_target,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "Authorization": f"Bearer {token}",
                "Cache-Control": "no-cache",
                "User-Agent": "FMV-PFC-LT-EEX-DataSource-v2/1",
            },
        )
        socket = connection.sock
        if socket is None:
            raise EexDatasourceV2CaptureError("EEX TLS socket is unavailable")
        certificate = socket.getpeercert(binary_form=True)
        if not certificate:
            raise EexDatasourceV2CaptureError("EEX TLS peer certificate is unavailable")
        cipher = socket.cipher()
        response = connection.getresponse()
        body = response.read(spec.max_response_bytes + 1)
        if len(body) > spec.max_response_bytes:
            raise EexDatasourceV2CaptureError("EEX response exceeds the byte limit")
        return HttpsExchange(
            status=int(response.status),
            reason=str(response.reason or ""),
            headers=tuple((str(name), str(value)) for name, value in response.getheaders()),
            body=body,
            tls_version=str(socket.version() or ""),
            tls_cipher=str(cipher[0] if cipher else ""),
            peer_leaf_certificate_sha256=hashlib.sha256(certificate).hexdigest(),
        )
    except EexDatasourceV2CaptureError:
        raise
    except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
        raise EexDatasourceV2CaptureError("EEX HTTPS request failed closed") from exc
    finally:
        connection.close()


@contextmanager
def _request_rate_slot(capture_root: Path, *, timeout_seconds: int):
    """Serialize in-flight requests and enforce EEX's one-request/second limit."""

    lock = capture_root / "request-rate-limit.lock"
    state = capture_root / "request-rate-limit.json"
    deadline = time.monotonic() + timeout_seconds
    with _exclusive_file_lock(lock, deadline=deadline):
        last_request_ns = 0
        if state.exists():
            payload = read_stable_single_link_file(
                state,
                label="EEX request rate-limit state",
                max_bytes=16 * 1024,
            )
            document = _strict_mapping(payload, label="EEX request rate-limit state")
            if canonical_json_bytes(document) != payload:
                raise EexDatasourceV2CaptureError(
                    "EEX request rate-limit state is not canonical JSON"
                )
            _exact_keys(
                document,
                {"schema_version", "last_request_epoch_ns_untrusted"},
                label="EEX request rate-limit state",
            )
            if document.get("schema_version") != "eex_datasource_v2_rate_limit.v1":
                raise EexDatasourceV2CaptureError(
                    "EEX request rate-limit state schema is unsupported"
                )
            value = document.get("last_request_epoch_ns_untrusted")
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise EexDatasourceV2CaptureError(
                    "EEX request rate-limit state timestamp is invalid"
                )
            last_request_ns = value
        current_ns = time.time_ns()
        if current_ns < last_request_ns:
            raise EexDatasourceV2CaptureError(
                "workstation clock moved backwards across EEX requests"
            )
        wait_seconds = max(
            0.0,
            _MIN_REQUEST_INTERVAL_SECONDS
            - ((current_ns - last_request_ns) / 1_000_000_000),
        )
        if time.monotonic() + wait_seconds > deadline:
            raise EexDatasourceV2CaptureError(
                "EEX request rate-limit wait exceeds request timeout"
            )
        if wait_seconds:
            time.sleep(wait_seconds)
        admitted_ns = time.time_ns()
        state_document = {
            "schema_version": "eex_datasource_v2_rate_limit.v1",
            "last_request_epoch_ns_untrusted": admitted_ns,
        }
        temporary = capture_root / f".request-rate-limit-{os.getpid()}.tmp"
        write_durable_exact_bytes(
            temporary,
            canonical_json_bytes(state_document),
            label="EEX request rate-limit state",
        )
        os.replace(temporary, state)
        yield


@contextmanager
def _exclusive_file_lock(path: Path, *, deadline: float):
    """Hold one crash-released OS lock until the guarded request completes."""

    handle = _open_governed_lock_file(path)
    acquired = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:  # pragma: no cover - exercised by non-Windows CI only
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise EexDatasourceV2CaptureError(
                        "EEX request rate-limit lock is unavailable"
                    ) from None
                time.sleep(min(0.05, remaining))
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover - exercised by non-Windows CI only
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _open_governed_lock_file(path: Path):
    selected = assert_absolute_path_has_no_links(path)
    before: os.stat_result | None
    try:
        before = selected.stat(follow_symlinks=False)
    except FileNotFoundError:
        before = None
    if before is not None and not _is_single_link_regular_file(before):
        raise EexDatasourceV2CaptureError(
            "EEX request rate-limit lock is not a single-link regular file"
        )
    flags = os.O_RDWR | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    if before is None:
        flags |= os.O_CREAT | os.O_EXCL
    descriptor: int | None = None
    try:
        descriptor = os.open(selected, flags, 0o600)
        opened = os.fstat(descriptor)
        if not _is_single_link_regular_file(opened):
            raise EexDatasourceV2CaptureError(
                "EEX request rate-limit lock descriptor is not single-link regular"
            )
        if before is not None and _file_object_identity(before) != _file_object_identity(
            opened
        ):
            raise EexDatasourceV2CaptureError(
                "EEX request rate-limit lock identity changed before open"
            )
        if opened.st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        elif opened.st_size != 1:
            raise EexDatasourceV2CaptureError(
                "EEX request rate-limit lock has invalid content size"
            )
        after = selected.stat(follow_symlinks=False)
        opened_after = os.fstat(descriptor)
        if (
            not _is_single_link_regular_file(after)
            or _file_object_identity(after) != _file_object_identity(opened_after)
        ):
            raise EexDatasourceV2CaptureError(
                "EEX request rate-limit lock identity changed during open"
            )
        return os.fdopen(descriptor, "r+b", buffering=0)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _is_single_link_regular_file(value: os.stat_result) -> bool:
    attributes = int(getattr(value, "st_file_attributes", 0))
    return bool(
        stat.S_ISREG(value.st_mode)
        and int(value.st_nlink) == 1
        and not (
            os.name == "nt"
            and attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        )
    )


def _file_object_identity(value: os.stat_result) -> tuple[int, int, int]:
    return int(value.st_dev), int(value.st_ino), int(value.st_nlink)


def _validate_response(
    spec: VerifiedCaptureSpec,
    exchange: HttpsExchange,
    headers: Mapping[str, str],
) -> tuple[list[object], object]:
    if exchange.tls_version not in {"TLSv1.2", "TLSv1.3"}:
        raise EexDatasourceV2CaptureError("EEX TLS version is below the contract minimum")
    if not exchange.tls_cipher:
        raise EexDatasourceV2CaptureError("EEX TLS cipher is unavailable")
    _require_sha256(exchange.peer_leaf_certificate_sha256, label="peer leaf certificate")
    if exchange.status != 200:
        raise EexDatasourceV2CaptureError(f"EEX returned HTTP {exchange.status}")
    content_type = headers["content-type"].split(";", 1)[0].strip().lower()
    if content_type != "application/json":
        raise EexDatasourceV2CaptureError("EEX response content type is not JSON")
    encoding = headers.get("content-encoding", "identity").strip().lower()
    if encoding not in {"", "identity"}:
        raise EexDatasourceV2CaptureError("EEX response content encoding is not identity")
    if len(exchange.body) < 1 or len(exchange.body) > spec.max_response_bytes:
        raise EexDatasourceV2CaptureError("EEX response body size is invalid")
    declared = headers.get("content-length")
    if declared is not None and (not declared.isdigit() or int(declared) != len(exchange.body)):
        raise EexDatasourceV2CaptureError("EEX response Content-Length mismatch")
    try:
        parsed = load_strict_json(exchange.body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise EexDatasourceV2CaptureError("EEX response is not strict UTF-8 JSON") from exc
    selected = parsed
    for part in spec.records_path:
        if not isinstance(selected, Mapping) or part not in selected:
            raise EexDatasourceV2CaptureError("EEX records path is absent")
        selected = selected[part]
    if isinstance(selected, Mapping):
        records: list[object] = [selected]
    elif isinstance(selected, list):
        records = list(selected)
    else:
        raise EexDatasourceV2CaptureError("EEX records payload is not an object or list")
    if not records:
        raise EexDatasourceV2CaptureError("EEX response contains no records")
    if len(records) >= _PROVIDER_RECORD_LIMIT:
        raise EexDatasourceV2CaptureError(
            "EEX response reaches the provider record limit and may be truncated"
        )
    _validate_product_records(spec, records)
    return records, parsed


def _reject_credential_echo(exchange: HttpsExchange, *, token: str) -> None:
    candidates = [exchange.body, exchange.reason.encode("utf-8", errors="replace")]
    for name, value in exchange.headers:
        candidates.append(str(name).encode("utf-8", errors="replace"))
        candidates.append(str(value).encode("utf-8", errors="replace"))
    if any(_contains_credential(candidate, token=token) for candidate in candidates):
        raise EexDatasourceV2CaptureError(
            "EEX response reflects credential material; no response artifact was written"
        )


def _credential_encodings(token: str) -> frozenset[str]:
    token_bytes = token.encode("ascii")
    return frozenset(
        {
            token,
            quote(token, safe=""),
            base64.b64encode(token_bytes).decode("ascii"),
            base64.b64encode(token_bytes).decode("ascii").rstrip("="),
            base64.urlsafe_b64encode(token_bytes).decode("ascii"),
            base64.urlsafe_b64encode(token_bytes).decode("ascii").rstrip("="),
        }
    )


def _contains_credential(value: bytes, *, token: str) -> bool:
    encodings = tuple(candidate.encode("ascii") for candidate in _credential_encodings(token))
    current = value
    for _ in range(4):
        compact = current.translate(None, b" \t\r\n")
        if any(candidate in current or candidate in compact for candidate in encodings):
            return True
        decoded = unquote_to_bytes(current)
        if decoded == current:
            return False
        current = decoded
    compact = current.translate(None, b" \t\r\n")
    if any(candidate in current or candidate in compact for candidate in encodings):
        return True
    return unquote_to_bytes(current) != current


def _reject_decoded_credential(value: object, *, token: str) -> None:
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            if _contains_credential(current.encode("utf-8"), token=token):
                raise EexDatasourceV2CaptureError(
                    "EEX response reflects credential material; no response artifact was written"
                )
        elif isinstance(current, Mapping):
            pending.extend(current.keys())
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)


def _reverify_capture_inputs(spec: VerifiedCaptureSpec) -> None:
    expected = (
        (spec.spec_path, spec.spec_sha256, _MAX_SPEC_BYTES, "capture spec"),
        (
            spec.user_guide_path,
            spec.user_guide_sha256,
            _MAX_DOCUMENT_BYTES,
            "EEX REST v2 user guide",
        ),
        (
            spec.openapi_path,
            spec.openapi_sha256,
            _MAX_DOCUMENT_BYTES,
            "EEX REST v2 OpenAPI specification",
        ),
        (spec.ca_bundle_path, spec.ca_bundle_sha256, 4 * 1024 * 1024, "TLS CA bundle"),
    )
    for path, expected_sha256, max_bytes, label in expected:
        payload = read_stable_single_link_file(path, label=label, max_bytes=max_bytes)
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise EexDatasourceV2CaptureError(
                f"{label} changed after request and before capture finalization"
            )


def _validate_product_records(spec: VerifiedCaptureSpec, records: Sequence[object]) -> None:
    expected = spec.product_identity
    revisions: list[tuple[datetime, str]] = []
    for item in records:
        if not isinstance(item, Mapping):
            raise EexDatasourceV2CaptureError("EEX product record is not a mapping")
        exact_identity = (
            item.get("ShortCode") == expected.short_code
            and item.get("Maturity") == expected.maturity
            and item.get("TrdDate") == expected.trade_date
            and item.get("InstrumentISIN") == expected.instrument_isin
        )
        if not exact_identity:
            raise EexDatasourceV2CaptureError(
                "EEX response mixes or omits the exact product identity"
            )
        if spec.data_role == "SETTLEMENT":
            action_fields = [name for name in ("UpdtAct", "UpdAct") if name in item]
            if len(action_fields) != 1 or item[action_fields[0]] not in {"New", "Change"}:
                raise EexDatasourceV2CaptureError(
                    "settlement update action is absent or unsupported"
                )
            if not isinstance(item.get("Px"), (int, float)) or isinstance(
                item.get("Px"), bool
            ):
                raise EexDatasourceV2CaptureError("settlement price is not numeric")
            if item.get("Currency") != "EUR" or item.get("UOM") != "MWh":
                raise EexDatasourceV2CaptureError(
                    "settlement currency or unit is not EUR/MWh"
                )
            timestamp = _utc(item.get("Tm"), label="settlement update time", as_dt=True)
            if not isinstance(timestamp, datetime):
                raise EexDatasourceV2CaptureError("settlement update time is invalid")
            revisions.append((timestamp, str(item[action_fields[0]])))
    chronological = sorted(revisions, key=lambda value: value[0])
    if chronological and (
        chronological[0][1] != "New"
        or any(action != "Change" for _, action in chronological[1:])
        or len({timestamp for timestamp, _ in chronological}) != len(chronological)
    ):
        raise EexDatasourceV2CaptureError(
            "settlement revision lifecycle is not one New followed by unique Changes"
        )


def _verify_reference_capture(
    root: Path,
    value: object,
    *,
    product_identity: ProductIdentity,
) -> Mapping[str, str]:
    reference = _mapping(value, label="reference capture")
    _exact_keys(reference, _REFERENCE_KEYS, label="reference capture")
    expected_manifest_hash = _require_sha256(
        reference.get("manifest_sha256"), label="reference capture manifest"
    )
    expected_response_hash = _require_sha256(
        reference.get("response_sha256"), label="reference capture response"
    )
    manifest_path = _repo_file(
        root, reference.get("manifest_path"), label="reference capture manifest"
    )
    expected_parent = root / "build" / "eex-datasource-v2-captures"
    if (
        manifest_path.name != "capture-manifest.json"
        or manifest_path.parent.parent != expected_parent
    ):
        raise EexDatasourceV2CaptureError(
            "reference manifest is not one finalized EEX v2 capture"
        )
    payload = read_stable_single_link_file(
        manifest_path, label="reference capture manifest", max_bytes=_MAX_SPEC_BYTES
    )
    if hashlib.sha256(payload).hexdigest() != expected_manifest_hash:
        raise EexDatasourceV2CaptureError("reference capture manifest SHA-256 mismatch")
    manifest = _strict_mapping(payload, label="reference capture manifest")
    if canonical_json_bytes(manifest) != payload:
        raise EexDatasourceV2CaptureError("reference capture manifest is not canonical JSON")
    _exact_keys(manifest, _MANIFEST_KEYS, label="reference capture manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA or manifest.get("status") != CAPTURE_STATUS:
        raise EexDatasourceV2CaptureError("reference capture manifest is unsupported")
    if manifest_path.parent.name != manifest.get("capture_id"):
        raise EexDatasourceV2CaptureError("reference capture directory binding is inconsistent")
    if (
        manifest.get("source_system") != "EEX_GROUP_DATASOURCE_REST_API_V2"
        or manifest.get("workstation_clock_trusted") is not False
        or manifest.get("reference_capture") is not None
        or manifest.get("next_required_authorities") != list(_NEXT_REQUIRED_AUTHORITIES)
    ):
        raise EexDatasourceV2CaptureError(
            "reference capture source, clock or authority requirements are inconsistent"
        )
    for field in _NEGATIVE_EVIDENCE_FIELDS:
        if manifest.get(field) is not False:
            raise EexDatasourceV2CaptureError(
                f"reference capture {field} is not explicitly false"
            )
    for field in _AUTHORITY_FIELDS:
        if manifest.get(field) is not False:
            raise EexDatasourceV2CaptureError(
                f"reference capture {field} is not explicitly false"
            )
    spec_binding = _mapping(manifest.get("spec"), label="reference spec binding")
    _exact_keys(
        spec_binding,
        {"path", "spec_id", "sha256"},
        label="reference spec binding",
    )
    reference_spec = load_eex_datasource_v2_capture_spec(
        repo_root=root,
        spec_path=spec_binding.get("path"),
        expected_spec_sha256=_require_sha256(
            spec_binding.get("sha256"), label="reference spec"
        ),
    )
    if (
        reference_spec.capture_id != manifest.get("capture_id")
        or reference_spec.spec_id != spec_binding.get("spec_id")
        or reference_spec.data_role != "REFERENCE_DATA"
    ):
        raise EexDatasourceV2CaptureError("reference spec binding is inconsistent")
    request = _mapping(manifest.get("request"), label="reference capture request")
    if dict(request) != {
        "method": "GET",
        "api_root": API_ROOT,
        "request_target": reference_spec.request_target,
        "data_role": "REFERENCE_DATA",
        "commodity": "POWER",
        "area": "CH",
        "authorization_scheme": "BEARER_ENVIRONMENT_ONLY",
        "access_token_persisted": False,
    }:
        raise EexDatasourceV2CaptureError(
            "reference capture request claims differ from its verified spec"
        )
    documentation = _mapping(
        manifest.get("documentation"), label="reference documentation"
    )
    if dict(documentation) != {
        "user_guide_release": GUIDE_RELEASE,
        "user_guide_published_date": GUIDE_PUBLISHED_DATE,
        "user_guide_path": _relative(root, reference_spec.user_guide_path),
        "user_guide_sha256": reference_spec.user_guide_sha256,
        "openapi_path": _relative(root, reference_spec.openapi_path),
        "openapi_sha256": reference_spec.openapi_sha256,
    }:
        raise EexDatasourceV2CaptureError(
            "reference capture documentation differs from its verified spec"
        )
    tls_trust = _mapping(manifest.get("tls_trust"), label="reference TLS trust")
    if dict(tls_trust) != {
        "ca_bundle_path": _relative(root, reference_spec.ca_bundle_path),
        "ca_bundle_sha256": reference_spec.ca_bundle_sha256,
        "ca_bundle_encoding": reference_spec.ca_bundle_encoding,
        "minimum_tls_version": "TLSv1.2",
        "ambient_windows_certificate_store_used": False,
        "ca_loaded_from_hash_verified_captured_bytes": True,
        "tls_key_logging_disabled": True,
        "trust_classification": (
            "CALLER_PINNED_ROOT_TLS_SERVER_AUTHENTICATION_NOT_DIRECT_PROVIDER_IDENTITY"
        ),
        "tls_interception_security_approved": False,
    }:
        raise EexDatasourceV2CaptureError(
            "reference capture TLS claims differ from its verified spec"
        )
    identity = _mapping(manifest.get("product_identity"), label="reference product identity")
    if dict(identity) != {
        "trade_date": product_identity.trade_date,
        "short_code": product_identity.short_code,
        "maturity": product_identity.maturity,
        "instrument_isin": product_identity.instrument_isin,
    }:
        raise EexDatasourceV2CaptureError("reference capture product identity mismatch")
    response = _mapping(manifest.get("response"), label="reference response")
    _exact_keys(
        response,
        {
            "path",
            "sha256",
            "size_bytes",
            "strict_json",
            "top_level_type",
            "records_path",
            "record_count",
            "provider_record_limit",
            "provider_limit_reached",
            "corrections_preserved_without_collapse",
            "settlement_revision_ordering",
        },
        label="reference response",
    )
    if response.get("sha256") != expected_response_hash:
        raise EexDatasourceV2CaptureError("reference response binding mismatch")
    body_path = _child_file(
        manifest_path.parent, response.get("path"), label="reference response body"
    )
    body = read_stable_single_link_file(
        body_path, label="reference response body", max_bytes=_MAX_RESPONSE_BYTES
    )
    if hashlib.sha256(body).hexdigest() != expected_response_hash:
        raise EexDatasourceV2CaptureError("reference response SHA-256 mismatch")
    transport = _mapping(manifest.get("transport"), label="reference transport")
    _exact_keys(
        transport,
        {
            "http_status",
            "response_headers_path",
            "response_headers_sha256",
            "tls_version",
            "tls_cipher",
            "peer_leaf_certificate_sha256",
            "provider_date_header",
            "provider_date_header_trusted_time",
        },
        label="reference transport",
    )
    headers_path = _child_file(
        manifest_path.parent,
        transport.get("response_headers_path"),
        label="reference response headers",
    )
    headers_payload = read_stable_single_link_file(
        headers_path,
        label="reference response headers",
        max_bytes=_MAX_SPEC_BYTES,
    )
    expected_headers_hash = _require_sha256(
        transport.get("response_headers_sha256"),
        label="reference response headers",
    )
    if hashlib.sha256(headers_payload).hexdigest() != expected_headers_hash:
        raise EexDatasourceV2CaptureError("reference response headers SHA-256 mismatch")
    headers_document = _strict_mapping(
        headers_payload, label="reference response headers"
    )
    if canonical_json_bytes(headers_document) != headers_payload:
        raise EexDatasourceV2CaptureError(
            "reference response headers are not canonical JSON"
        )
    _exact_keys(
        headers_document,
        {"schema_version", "status_code", "reason", "headers"},
        label="reference response headers",
    )
    stored_headers = _mapping(
        headers_document.get("headers"), label="reference stored headers"
    )
    validated_headers = _validated_response_headers(
        tuple((str(name), str(value)) for name, value in stored_headers.items())
    )
    exchange = HttpsExchange(
        status=_bounded_int(transport.get("http_status"), 100, 599, label="HTTP status"),
        reason=str(headers_document.get("reason", "")),
        headers=tuple(validated_headers.items()),
        body=body,
        tls_version=str(transport.get("tls_version", "")),
        tls_cipher=str(transport.get("tls_cipher", "")),
        peer_leaf_certificate_sha256=_require_sha256(
            transport.get("peer_leaf_certificate_sha256"),
            label="reference peer leaf certificate",
        ),
    )
    records, parsed = _validate_response(
        reference_spec,
        exchange,
        validated_headers,
    )
    if (
        headers_document.get("status_code") != exchange.status
        or response.get("size_bytes") != len(body)
        or response.get("record_count") != len(records)
        or response.get("top_level_type") != type(parsed).__name__
        or response.get("records_path") != list(reference_spec.records_path)
        or response.get("strict_json") is not True
        or response.get("provider_record_limit") != _PROVIDER_RECORD_LIMIT
        or response.get("provider_limit_reached") is not False
        or response.get("corrections_preserved_without_collapse") is not False
        or response.get("settlement_revision_ordering") != "NOT_APPLICABLE"
        or transport.get("provider_date_header") != validated_headers["date"]
        or transport.get("provider_date_header_trusted_time") is not False
    ):
        raise EexDatasourceV2CaptureError("reference response replay differs from manifest")
    return {
        "manifest_path": _relative(root, manifest_path),
        "manifest_sha256": expected_manifest_hash,
        "response_sha256": expected_response_hash,
    }


def _validated_response_headers(values: Sequence[tuple[str, str]]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for raw_name, raw_value in values:
        name = raw_name.strip().lower()
        if name not in _RESPONSE_HEADER_ALLOWLIST:
            continue
        if name in selected:
            raise EexDatasourceV2CaptureError(f"EEX response repeats {name} header")
        value = raw_value.strip()
        if not value or any(ord(character) < 0x20 and character != "\t" for character in value):
            raise EexDatasourceV2CaptureError("EEX response header is invalid")
        selected[name] = value
    for required in ("content-type", "date"):
        if required not in selected:
            raise EexDatasourceV2CaptureError(f"EEX response lacks {required} header")
    try:
        provider_date = parsedate_to_datetime(selected["date"])
    except (InvalidSignature, TypeError, ValueError) as exc:
        raise EexDatasourceV2CaptureError("EEX response Date header is invalid") from exc
    if provider_date.tzinfo is None:
        raise EexDatasourceV2CaptureError("EEX response Date header lacks timezone")
    return dict(sorted(selected.items()))


def _endpoint_path(value: object, *, role: str) -> str:
    endpoint = str(value or "")
    if (
        not endpoint.startswith("/")
        or "?" in endpoint
        or "#" in endpoint
        or "\\" in endpoint
        or "%" in endpoint
        or "//" in endpoint
        or any(part in {"", ".", ".."} for part in endpoint[1:].split("/"))
    ):
        raise EexDatasourceV2CaptureError("endpoint path is not canonical and literal")
    prefix = "/rd/derivatives/POWER/CH/" if role == "REFERENCE_DATA" else "/SPR"
    if role == "REFERENCE_DATA" and not endpoint.startswith(prefix):
        raise EexDatasourceV2CaptureError("reference endpoint is not CH power derivatives /rd")
    if role == "SETTLEMENT" and not endpoint.startswith(
        ("/spr/derivatives/POWER/CH/", "/sprs/derivatives/POWER/CH/")
    ):
        raise EexDatasourceV2CaptureError("settlement endpoint is not CH power derivatives")
    if "+" in endpoint:
        raise EexDatasourceV2CaptureError("rolling-symbol endpoint is forbidden")
    return endpoint


def _query_pairs(value: object) -> tuple[tuple[str, str], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EexDatasourceV2CaptureError("query must be a canonical list")
    pairs: list[tuple[str, str]] = []
    for item in value:
        mapping = _mapping(item, label="query item")
        _exact_keys(mapping, {"name", "value"}, label="query item")
        name = str(mapping.get("name", ""))
        query_value = str(mapping.get("value", ""))
        if _SAFE_QUERY_NAME.fullmatch(name) is None or name.lower() in _SENSITIVE_QUERY_NAMES:
            raise EexDatasourceV2CaptureError("query parameter name is unsafe")
        if not query_value or any(ord(character) < 0x20 for character in query_value):
            raise EexDatasourceV2CaptureError("query parameter value is invalid")
        if "bearer " in query_value.lower():
            raise EexDatasourceV2CaptureError("query parameter contains credential material")
        pairs.append((name, query_value))
    if pairs != sorted(pairs) or len({name for name, _ in pairs}) != len(pairs):
        raise EexDatasourceV2CaptureError("query parameters must be sorted and unique")
    return tuple(pairs)


def _records_path(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EexDatasourceV2CaptureError("records path must be a list")
    parts = tuple(str(item) for item in value)
    if len(parts) > 8 or any(
        not part or len(part) > 64 or any(ord(character) < 0x20 for character in part)
        for part in parts
    ):
        raise EexDatasourceV2CaptureError("records path is invalid")
    return parts


def _verified_repo_document(
    root: Path,
    path: object,
    *,
    expected_sha256: str,
    label: str,
) -> Path:
    selected = _repo_file(root, path, label=label)
    payload = read_stable_single_link_file(selected, label=label, max_bytes=_MAX_DOCUMENT_BYTES)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise EexDatasourceV2CaptureError(f"{label} SHA-256 mismatch")
    return selected


def _validate_ca_bundle(payload: bytes, *, planned_at_utc: str) -> str:
    try:
        if payload.startswith(b"-----BEGIN CERTIFICATE-----"):
            certificates = x509.load_pem_x509_certificates(payload)
            encoding = "PEM"
        else:
            certificates = [x509.load_der_x509_certificate(payload)]
            encoding = "DER"
    except ValueError as exc:
        raise EexDatasourceV2CaptureError("TLS CA bundle is not valid X.509") from exc
    if len(certificates) != 1:
        raise EexDatasourceV2CaptureError("TLS trust must contain exactly one pinned root CA")
    certificate = certificates[0]
    if certificate.subject != certificate.issuer:
        raise EexDatasourceV2CaptureError("TLS trust anchor is not self-issued")
    try:
        basic_constraints = certificate.extensions.get_extension_for_class(
            x509.BasicConstraints
        ).value
    except x509.ExtensionNotFound as exc:
        raise EexDatasourceV2CaptureError("TLS trust anchor lacks CA constraints") from exc
    if not basic_constraints.ca:
        raise EexDatasourceV2CaptureError("TLS trust anchor is not a CA")
    try:
        key_usage = certificate.extensions.get_extension_for_class(x509.KeyUsage).value
    except x509.ExtensionNotFound as exc:
        raise EexDatasourceV2CaptureError("TLS trust anchor lacks key usage") from exc
    if not key_usage.key_cert_sign:
        raise EexDatasourceV2CaptureError("TLS trust anchor cannot sign certificates")
    try:
        certificate.verify_directly_issued_by(certificate)
    except (TypeError, ValueError) as exc:
        raise EexDatasourceV2CaptureError(
            "TLS trust anchor self-signature is invalid"
        ) from exc
    planned = datetime.fromisoformat(planned_at_utc.replace("Z", "+00:00"))
    if not certificate.not_valid_before_utc <= planned <= certificate.not_valid_after_utc:
        raise EexDatasourceV2CaptureError("TLS trust anchor is not valid at planned capture time")
    return encoding


def _repo_file(
    root: Path,
    value: object,
    *,
    label: str,
    allow_absolute: bool = False,
) -> Path:
    candidate = Path(os.fspath(value)) if isinstance(value, (str, os.PathLike)) else Path("")
    if allow_absolute and candidate.is_absolute():
        selected = assert_absolute_path_has_no_links(candidate)
    else:
        relative = _portable_relative(value, label=label)
        selected = assert_absolute_path_has_no_links(root / Path(*relative.parts))
    try:
        selected.relative_to(root)
    except ValueError as exc:
        raise EexDatasourceV2CaptureError(f"{label} is outside the repository") from exc
    if not selected.is_file():
        raise EexDatasourceV2CaptureError(f"{label} is unavailable")
    return selected


def _child_file(parent: Path, value: object, *, label: str) -> Path:
    relative = _portable_relative(value, label=label)
    selected = assert_absolute_path_has_no_links(parent / Path(*relative.parts))
    try:
        selected.relative_to(parent)
    except ValueError as exc:
        raise EexDatasourceV2CaptureError(f"{label} is outside its capture") from exc
    if not selected.is_file():
        raise EexDatasourceV2CaptureError(f"{label} is unavailable")
    return selected


def _portable_relative(value: object, *, label: str) -> PurePosixPath:
    raw = str(value or "")
    parts = raw.split("/")
    if (
        not raw
        or "\\" in raw
        or ":" in raw
        or raw.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
        or any(character in raw for character in '<>"|?*')
        or any(ord(character) < 0x20 for character in raw)
        or any(part.endswith((".", " ")) for part in parts)
        or any(part.split(".", 1)[0].upper() in _WINDOWS_RESERVED for part in parts)
    ):
        raise EexDatasourceV2CaptureError(f"{label} path is not portable relative")
    return PurePosixPath(*parts)


def _strict_mapping(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        document = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise EexDatasourceV2CaptureError(f"{label} is not strict JSON") from exc
    if not isinstance(document, dict):
        raise EexDatasourceV2CaptureError(f"{label} must be a mapping")
    return document


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EexDatasourceV2CaptureError(f"{label} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise EexDatasourceV2CaptureError(f"{label} fields are not exact")


def _equal(value: object, expected: object, *, label: str) -> None:
    if value != expected or (isinstance(expected, bool) and value is not expected):
        raise EexDatasourceV2CaptureError(f"{label} is invalid")


def _matched(value: object, pattern: re.Pattern[str], *, label: str) -> str:
    text = str(value or "")
    if pattern.fullmatch(text) is None:
        raise EexDatasourceV2CaptureError(f"{label} is invalid")
    return text


def _portable_id(value: object, *, label: str) -> str:
    selected = _matched(value, _PORTABLE_ID, label=label)
    if (
        selected.endswith((".", " "))
        or selected.split(".", 1)[0].upper() in _WINDOWS_RESERVED
    ):
        raise EexDatasourceV2CaptureError(f"{label} is not portable")
    return selected


def _require_sha256(value: object, *, label: str) -> str:
    return _matched(value, _SHA256, label=f"{label} SHA-256")


def _bounded_int(value: object, minimum: int, maximum: int, *, label: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or value > maximum
    ):
        raise EexDatasourceV2CaptureError(f"{label} is outside its allowed range")
    return value


def _root(value: str | Path) -> Path:
    root = assert_absolute_path_has_no_links(value)
    if not root.is_dir() or not (root / ".git").exists() or not (root / "build").is_dir():
        raise EexDatasourceV2CaptureError("canonical repository root is unavailable")
    return root


def _utc(value: object, *, label: str, as_dt: bool = False) -> str | datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise EexDatasourceV2CaptureError(f"{label} is not an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise EexDatasourceV2CaptureError(f"{label} lacks timezone")
    normalized = parsed.astimezone(timezone.utc)
    return normalized if as_dt else _iso(normalized)


def _clock(clock: Callable[[], datetime], *, label: str) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EexDatasourceV2CaptureError(f"{label} clock is not timezone-aware")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _relative(parent: Path, child: Path) -> str:
    return child.relative_to(parent).as_posix()


def _write_json(path: Path, value: object, *, label: str) -> Path:
    return write_durable_exact_bytes(path, canonical_json_bytes(value), label=label)


__all__ = [
    "API_ROOT",
    "ATTEMPT_SCHEMA",
    "CAPTURE_STATUS",
    "EexDatasourceV2CaptureError",
    "GUIDE_PUBLISHED_DATE",
    "GUIDE_RELEASE",
    "GUIDE_SHA256",
    "HttpsExchange",
    "MANIFEST_SCHEMA",
    "SPEC_SCHEMA",
    "SPEC_STATUS",
    "TOKEN_ENV",
    "capture_eex_datasource_v2",
    "compute_spec_id",
    "load_eex_datasource_v2_capture_spec",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - exercised through the installed module
    raise SystemExit(main())
