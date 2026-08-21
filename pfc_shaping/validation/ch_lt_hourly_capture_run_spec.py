"""Frozen checkout-only contract for one CH hourly day-ahead capture attempt."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json

SCHEMA_VERSION = "ch_lt_hourly_day_ahead_capture_run_spec.v1"
RUN_SPEC_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-HOURLY-DAY-AHEAD-CAPTURE-RUN-SPEC-20260727.json"
)
RUN_SPEC_ID = "c6389dfa8c948fc55558ff2d6acd01aeab488b6bce228d129be9e022cfcb0cb7"
DOCUMENT_SHA256 = "47cb42616f6cda629b725b05159db0cd1dc3c3e4ce054d857a356e130851deb7"
CA_BUNDLE_SHA256 = "e68ca0a61d7e000d0c10fb3a712be09c533378b99af5fa4c30a34c7aa8f36f25"
OUTPUT_RELATIVE_PATH = "output/phase14/ch_da_hourly_capture_20260727_attempt1"
STATUS = "FROZEN_LOCAL_CAPTURE_ATTEMPT_NOT_ADMISSION"
RETRY_SCHEMA_VERSION = "ch_lt_hourly_day_ahead_capture_curl_retry.v1"
RETRY_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-HOURLY-DAY-AHEAD-CAPTURE-CURL-RETRY-20260727.json"
)
RETRY_ID = "3ad869f51d0e225c60beea4ce10667bc0267441f44fae36c5f41994899c46c0e"
RETRY_DOCUMENT_SHA256 = (
    "2b981cae15b301c177e9768a13505b9bb5221c384d1ee20ee4040a602ac2e36f"
)
RETRY_OUTPUT_RELATIVE_PATH = (
    "output/phase14/ch_da_hourly_capture_20260727_attempt2_curl"
)
PRIOR_ATTEMPT_RELATIVE_PATH = f"{OUTPUT_RELATIVE_PATH}/capture-attempt.json"
PRIOR_ATTEMPT_SHA256 = (
    "39b39ba5e044a992ba28349519a7341b103a5602a302c692671dd71106f0e27b"
)
RETRY_STATUS = "FROZEN_LOCAL_TRANSPORT_RETRY_NOT_ADMISSION"

_MAX_SPEC_BYTES = 256 * 1024
_MAX_CA_BYTES = 2 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_KEYS = {
    "schema_version",
    "run_spec_id",
    "lifecycle_state",
    "as_of_utc",
    "market",
    "timezone",
    "source",
    "window",
    "execution",
    "tls",
    "intended_use",
    "failure_policy",
    "external_authorities",
    "local_capture_execution_authorized",
    "scientific_execution_authorized",
    "scientific_admission",
    "model_selection_authorized",
    "publication_authorized",
    "production_authorization",
    "promotion_gate",
    "future_holdout_consumed",
    "t057_consumed",
}


class ChLtHourlyCaptureRunSpecError(ValueError):
    """Raised when the frozen local capture attempt can be rebound or weakened."""


@dataclass(frozen=True)
class VerifiedHourlyCaptureRunSpec:
    run_spec_id: str
    document_sha256: str
    role: str
    start_utc: str
    end_utc: str
    raw_cadence_minutes: int
    acquisition_id: str
    output_directory: Path
    ca_bundle_path: Path
    expected_source_observation_count: int
    expected_bronze_observation_count: int

    def audit_dict(self) -> dict[str, object]:
        return {
            "schema_version": "ch_lt_hourly_capture_run_spec_audit.v1",
            "status": STATUS,
            "run_spec_id": self.run_spec_id,
            "document_sha256": self.document_sha256,
            "output_directory": str(self.output_directory),
            "local_capture_execution_authorized": True,
            "scientific_execution_authorized": False,
            "scientific_admission": False,
            "model_selection_authorized": False,
            "publication_authorized": False,
            "production_authorization": False,
            "promotion_gate": False,
            "future_holdout_consumed": False,
            "t057_consumed": False,
        }


@dataclass(frozen=True)
class VerifiedHourlyCurlRetry:
    retry_id: str
    document_sha256: str
    base: VerifiedHourlyCaptureRunSpec
    acquisition_id: str
    output_directory: Path
    response_body_path: Path
    response_headers_path: Path
    curl_arguments: tuple[str, ...]

    def audit_dict(self) -> dict[str, object]:
        return {
            "schema_version": "ch_lt_hourly_capture_curl_retry_audit.v1",
            "status": RETRY_STATUS,
            "retry_id": self.retry_id,
            "document_sha256": self.document_sha256,
            "prior_attempt_sha256": PRIOR_ATTEMPT_SHA256,
            "output_directory": str(self.output_directory),
            "local_capture_execution_authorized": True,
            "scientific_execution_authorized": False,
            "scientific_admission": False,
            "model_selection_authorized": False,
            "publication_authorized": False,
            "production_authorization": False,
            "promotion_gate": False,
            "future_holdout_consumed": False,
            "t057_consumed": False,
        }


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def compute_run_spec_id(document: Mapping[str, object]) -> str:
    unsigned = dict(document)
    unsigned.pop("run_spec_id", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def compute_retry_id(document: Mapping[str, object]) -> str:
    unsigned = dict(document)
    unsigned.pop("retry_id", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def verify_hourly_capture_run_spec(
    *,
    repo_root: str | Path,
    run_spec_path: str | Path,
    expected_document_sha256: str,
) -> VerifiedHourlyCaptureRunSpec:
    """Verify exact spec and CA bytes, returning only frozen execution values."""

    _require_sha256(expected_document_sha256, label="caller-held run spec")
    if expected_document_sha256 != DOCUMENT_SHA256:
        raise ChLtHourlyCaptureRunSpecError(
            "caller-held run spec SHA-256 does not match frozen identity"
        )
    root = assert_absolute_path_has_no_links(_absolute(repo_root))
    canonical_spec = _repo_path(root, RUN_SPEC_RELATIVE_PATH)
    selected_spec = _absolute(run_spec_path)
    if _path_key(selected_spec) != _path_key(canonical_spec):
        raise ChLtHourlyCaptureRunSpecError("run spec path is not canonical")
    payload = read_stable_single_link_file(
        canonical_spec,
        label="CH LT hourly capture run spec",
        max_bytes=_MAX_SPEC_BYTES,
    )
    if hashlib.sha256(payload).hexdigest() != DOCUMENT_SHA256:
        raise ChLtHourlyCaptureRunSpecError("run spec SHA-256 mismatch")
    document = _strict_mapping(payload, label="run spec")
    _exact_keys(document, _TOP_LEVEL_KEYS, label="run spec")
    _equal(document.get("schema_version"), SCHEMA_VERSION, "schema version")
    _equal(document.get("run_spec_id"), RUN_SPEC_ID, "run spec ID")
    if compute_run_spec_id(document) != RUN_SPEC_ID:
        raise ChLtHourlyCaptureRunSpecError("run spec ID does not bind content")
    _equal(document.get("lifecycle_state"), STATUS, "lifecycle")
    _equal(document.get("as_of_utc"), "2026-07-27T00:00:00Z", "as-of")
    _equal(document.get("market"), "CH", "market")
    _equal(document.get("timezone"), "Europe/Zurich", "timezone")

    source = _mapping(document.get("source"), label="source")
    _equal(
        dict(source),
        {
            "role": "epex_ch",
            "source_system": "ENERGY_CHARTS_API",
            "endpoint": "https://api.energy-charts.info/price",
            "bidding_zone": "CH",
            "product": "DAY_AHEAD_PRICE_CH",
            "product_cadence_seconds": 3600,
            "product_identity_status": (
                "EXTERNAL_PRODUCT_AUCTION_SESSION_IDENTITY_UNVERIFIED"
            ),
            "license_status": "PROVIDER_RESPONSE_LICENSE_FIELD_REQUIRED_BY_PARSER",
        },
        "source contract",
    )
    window = _mapping(document.get("window"), label="window")
    _equal(
        dict(window),
        {
            "start_utc": "2026-06-26T22:00:00Z",
            "end_utc_exclusive": "2026-07-26T22:00:00Z",
            "duration_hours": 720,
            "expected_source_observation_count": 720,
            "dst_interpretation": (
                "UTC_INTERVAL_MAPPED_TO_EUROPE_ZURICH_DELIVERY_HOURS"
            ),
        },
        "window contract",
    )
    execution = _mapping(document.get("execution"), label="execution")
    _equal(
        dict(execution),
        {
            "wrapper": "scripts/run_ch_lt_hourly_capture_from_spec.py",
            "capture_implementation": "scripts/capture_public_energy_charts_lt.py",
            "acquisition_id": (
                "ch-da-hourly-20260626T2200Z-20260726T2200Z-attempt1"
            ),
            "raw_cadence_minutes": 60,
            "expected_bronze_observation_count": 2880,
            "expected_bronze_cadence_seconds": 900,
            "bronze_sampling_kind": "UPSAMPLED_STEPWISE_PROXY",
            "native_hourly_truth_eligible_after_external_admission": True,
            "native_quarter_hour_truth_eligible": False,
            "output_directory_relative": OUTPUT_RELATIVE_PATH,
            "output_namespace_reuse_forbidden": True,
        },
        "execution contract",
    )
    _verify_intended_use(document.get("intended_use"))
    _verify_failure_policy(document.get("failure_policy"))
    external = _mapping(document.get("external_authorities"), label="authorities")
    expected_external_keys = {
        "trusted_time_receipt_present",
        "independent_signature_present",
        "builder_inaccessible_immutable_namespace_present",
        "external_cas_receipt_present",
        "fresh_authenticated_head_present",
    }
    _exact_keys(external, expected_external_keys, label="external authorities")
    if any(value is not False for value in external.values()):
        raise ChLtHourlyCaptureRunSpecError("external authority must remain absent")
    _equal(document.get("local_capture_execution_authorized"), True, "local capture")
    for field in (
        "scientific_execution_authorized",
        "scientific_admission",
        "model_selection_authorized",
        "publication_authorized",
        "production_authorization",
        "promotion_gate",
        "future_holdout_consumed",
        "t057_consumed",
    ):
        _equal(document.get(field), False, field)

    tls = _mapping(document.get("tls"), label="TLS")
    _equal(tls.get("mode"), "EXPLICIT_PINNED_CA_BUNDLE", "TLS mode")
    _equal(tls.get("ca_bundle_sha256"), CA_BUNDLE_SHA256, "CA SHA-256")
    _equal(tls.get("ca_bundle_size_bytes"), 258962, "CA size")
    ca_bundle = assert_absolute_path_has_no_links(_absolute(str(tls.get("ca_bundle_path"))))
    ca_payload = read_stable_single_link_file(
        ca_bundle,
        label="pinned FMV CA bundle",
        max_bytes=_MAX_CA_BYTES,
    )
    if len(ca_payload) != 258962 or hashlib.sha256(ca_payload).hexdigest() != CA_BUNDLE_SHA256:
        raise ChLtHourlyCaptureRunSpecError("pinned CA bundle identity changed")

    output = _repo_path(root, OUTPUT_RELATIVE_PATH)
    if not _is_within(output, root):
        raise ChLtHourlyCaptureRunSpecError("capture output escapes repository")
    return VerifiedHourlyCaptureRunSpec(
        run_spec_id=RUN_SPEC_ID,
        document_sha256=DOCUMENT_SHA256,
        role="epex_ch",
        start_utc="2026-06-26T22:00:00Z",
        end_utc="2026-07-26T22:00:00Z",
        raw_cadence_minutes=60,
        acquisition_id="ch-da-hourly-20260626T2200Z-20260726T2200Z-attempt1",
        output_directory=output,
        ca_bundle_path=ca_bundle,
        expected_source_observation_count=720,
        expected_bronze_observation_count=2880,
    )


def verify_hourly_curl_retry(
    *,
    repo_root: str | Path,
    base_run_spec_path: str | Path,
    expected_base_sha256: str,
    retry_path: str | Path,
    expected_retry_sha256: str,
) -> VerifiedHourlyCurlRetry:
    """Verify the closed failed attempt and an exact curl transport retry."""

    base = verify_hourly_capture_run_spec(
        repo_root=repo_root,
        run_spec_path=base_run_spec_path,
        expected_document_sha256=expected_base_sha256,
    )
    _require_sha256(expected_retry_sha256, label="caller-held retry")
    if expected_retry_sha256 != RETRY_DOCUMENT_SHA256:
        raise ChLtHourlyCaptureRunSpecError(
            "caller-held retry SHA-256 does not match frozen identity"
        )
    root = assert_absolute_path_has_no_links(_absolute(repo_root))
    canonical_retry = _repo_path(root, RETRY_RELATIVE_PATH)
    if _path_key(_absolute(retry_path)) != _path_key(canonical_retry):
        raise ChLtHourlyCaptureRunSpecError("retry path is not canonical")
    payload = read_stable_single_link_file(
        canonical_retry,
        label="CH LT hourly curl retry",
        max_bytes=_MAX_SPEC_BYTES,
    )
    if hashlib.sha256(payload).hexdigest() != RETRY_DOCUMENT_SHA256:
        raise ChLtHourlyCaptureRunSpecError("retry SHA-256 mismatch")
    document = _strict_mapping(payload, label="retry")
    _exact_keys(
        document,
        {
            "schema_version",
            "retry_id",
            "lifecycle_state",
            "as_of_utc",
            "prior_attempt",
            "request_identity",
            "retry_execution",
            "curl_transport",
            "completion_contract",
            "external_authorities",
            "local_capture_execution_authorized",
            "scientific_execution_authorized",
            "scientific_admission",
            "model_selection_authorized",
            "publication_authorized",
            "production_authorization",
            "promotion_gate",
            "future_holdout_consumed",
            "t057_consumed",
        },
        label="retry",
    )
    _equal(document.get("schema_version"), RETRY_SCHEMA_VERSION, "retry schema")
    _equal(document.get("retry_id"), RETRY_ID, "retry ID")
    if compute_retry_id(document) != RETRY_ID:
        raise ChLtHourlyCaptureRunSpecError("retry ID does not bind content")
    _equal(document.get("lifecycle_state"), RETRY_STATUS, "retry lifecycle")
    _equal(document.get("as_of_utc"), "2026-07-27T00:00:00Z", "retry as-of")
    _equal(
        dict(_mapping(document.get("prior_attempt"), label="prior attempt")),
        {
            "run_spec_path": RUN_SPEC_RELATIVE_PATH,
            "run_spec_sha256": DOCUMENT_SHA256,
            "run_spec_id": RUN_SPEC_ID,
            "output_directory_relative": OUTPUT_RELATIVE_PATH,
            "attempt_record_relative": PRIOR_ATTEMPT_RELATIVE_PATH,
            "attempt_record_sha256": PRIOR_ATTEMPT_SHA256,
            "failure_classification": "SANDBOX_REQUESTS_PROXY_CONNECTION_REFUSED",
            "failure_stderr_durably_captured": False,
            "prior_namespace_closed_to_retry": True,
        },
        "prior attempt binding",
    )
    prior_directory = _repo_path(root, OUTPUT_RELATIVE_PATH)
    if not prior_directory.is_dir() or {path.name for path in prior_directory.iterdir()} != {
        "capture-attempt.json"
    }:
        raise ChLtHourlyCaptureRunSpecError("prior failed namespace is not exact")
    prior_payload = read_stable_single_link_file(
        _repo_path(root, PRIOR_ATTEMPT_RELATIVE_PATH),
        label="prior failed capture attempt",
        max_bytes=_MAX_SPEC_BYTES,
    )
    if hashlib.sha256(prior_payload).hexdigest() != PRIOR_ATTEMPT_SHA256:
        raise ChLtHourlyCaptureRunSpecError("prior attempt SHA-256 mismatch")
    _equal(
        dict(_mapping(document.get("request_identity"), label="request identity")),
        {
            "endpoint": "https://api.energy-charts.info/price",
            "bidding_zone": "CH",
            "start_utc": base.start_utc,
            "end_utc_exclusive": base.end_utc,
            "provider_end_parameter_inclusive": "2026-07-26T21:00Z",
            "expected_source_observation_count": 720,
            "source_cadence_seconds": 3600,
        },
        "request identity",
    )
    acquisition_id = "ch-da-hourly-20260626T2200Z-20260726T2200Z-attempt2-curl"
    _equal(
        dict(_mapping(document.get("retry_execution"), label="retry execution")),
        {
            "wrapper": "scripts/run_ch_lt_hourly_curl_retry_from_spec.py",
            "acquisition_id": acquisition_id,
            "output_directory_relative": RETRY_OUTPUT_RELATIVE_PATH,
            "prepare_subcommand_required": True,
            "curl_subcommand_required": True,
            "finalize_subcommand_required": True,
            "output_namespace_reuse_forbidden": True,
        },
        "retry execution",
    )
    output = _repo_path(root, RETRY_OUTPUT_RELATIVE_PATH)
    body_path = output / "provider-response.json"
    headers_path = output / "response-headers.txt"
    root_token = root.as_posix()
    arguments = (
        "-L",
        "--ssl-no-revoke",
        "--max-redirs",
        "0",
        "--proto",
        "=https",
        "--tlsv1.2",
        "--fail-with-body",
        "--silent",
        "--show-error",
        "--header",
        "Accept: application/json",
        "--header",
        "Accept-Encoding: identity",
        "--cacert",
        "C:/certs/git-ca-plus-fmv.crt",
        "--dump-header",
        headers_path.as_posix(),
        "--output",
        body_path.as_posix(),
        (
            "https://api.energy-charts.info/price?bzn=CH&"
            "start=2026-06-26T22%3A00Z&end=2026-07-26T21%3A00Z"
        ),
    )
    transport = _mapping(document.get("curl_transport"), label="curl transport")
    expected_template = [
        argument.replace(root_token, "{repo_root}")
        if argument.startswith(root_token)
        else argument
        for argument in arguments
    ]
    _equal(
        dict(transport),
        {
            "executable": "curl.exe",
            "arguments_template": expected_template,
            "redirect_limit": 0,
            "response_encoding_requested": "identity",
            "ca_bundle_sha256": CA_BUNDLE_SHA256,
            "response_body_relative": (
                f"{RETRY_OUTPUT_RELATIVE_PATH}/provider-response.json"
            ),
            "response_headers_relative": (
                f"{RETRY_OUTPUT_RELATIVE_PATH}/response-headers.txt"
            ),
        },
        "curl transport",
    )
    _equal(
        dict(_mapping(document.get("completion_contract"), label="completion")),
        {
            "http_status_required": 200,
            "media_type_required": "application/json",
            "content_encoding_allowed": ["", "identity"],
            "provider_body_max_bytes": 8388608,
            "expected_bronze_observation_count": 2880,
            "expected_bronze_cadence_seconds": 900,
            "bronze_sampling_kind": "UPSAMPLED_STEPWISE_PROXY",
            "native_hourly_truth_eligible_after_external_admission": True,
            "native_quarter_hour_truth_eligible": False,
            "receipt_last_written_required": True,
        },
        "completion contract",
    )
    external = _mapping(document.get("external_authorities"), label="retry authorities")
    if set(external) != {
        "trusted_time_receipt_present",
        "independent_signature_present",
        "builder_inaccessible_immutable_namespace_present",
        "external_cas_receipt_present",
        "fresh_authenticated_head_present",
    } or any(value is not False for value in external.values()):
        raise ChLtHourlyCaptureRunSpecError("retry external authorities are invalid")
    _equal(document.get("local_capture_execution_authorized"), True, "retry local capture")
    for field in (
        "scientific_execution_authorized",
        "scientific_admission",
        "model_selection_authorized",
        "publication_authorized",
        "production_authorization",
        "promotion_gate",
        "future_holdout_consumed",
        "t057_consumed",
    ):
        _equal(document.get(field), False, f"retry {field}")
    return VerifiedHourlyCurlRetry(
        retry_id=RETRY_ID,
        document_sha256=RETRY_DOCUMENT_SHA256,
        base=base,
        acquisition_id=acquisition_id,
        output_directory=output,
        response_body_path=body_path,
        response_headers_path=headers_path,
        curl_arguments=arguments,
    )


def _verify_intended_use(value: object) -> None:
    intended = _mapping(value, label="intended use")
    _equal(
        dict(intended),
        {
            "allowed": [
                "LOCAL_HOURLY_SHAPE_DIAGNOSTIC",
                "EXTERNAL_ADMISSION_INPUT_CANDIDATE",
            ],
            "forbidden": [
                "MONTHLY_LEVEL_AUTHORITY",
                "NATIVE_QUARTER_HOUR_TRUTH",
                "MODEL_SELECTION_BEFORE_EXTERNAL_ADMISSION",
                "FUTURE_HOLDOUT_TRUTH_OPEN",
                "SNAPSHOT_PUBLICATION",
                "CANDIDATE_PROMOTION",
                "PRODUCTION_PROMOTION",
            ],
        },
        "intended use",
    )


def _verify_failure_policy(value: object) -> None:
    policy = _mapping(value, label="failure policy")
    _equal(
        dict(policy),
        {
            "http_redirect": "FAIL_CLOSED",
            "non_200_status": "FAIL_CLOSED_RETAIN_ATTEMPT_ONLY",
            "unexpected_media_or_encoding": "FAIL_CLOSED_RETAIN_ATTEMPT_ONLY",
            "response_budget_exceeded": "FAIL_CLOSED_RETAIN_ATTEMPT_ONLY",
            "source_count_mismatch": "FAIL_CLOSED_NO_RUN_RECEIPT",
            "cadence_or_window_mismatch": "FAIL_CLOSED_NO_RUN_RECEIPT",
            "existing_output_directory": "FAIL_BEFORE_NETWORK",
            "retry_policy": "NEW_FROZEN_RUN_SPEC_AND_NEW_OUTPUT_NAMESPACE_ONLY",
        },
        "failure policy",
    )


def _strict_mapping(payload: bytes, *, label: str) -> Mapping[str, object]:
    try:
        parsed = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChLtHourlyCaptureRunSpecError(f"{label} is invalid JSON") from exc
    return _mapping(parsed, label=label)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ChLtHourlyCaptureRunSpecError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ChLtHourlyCaptureRunSpecError(f"{label} keys are not exact")


def _equal(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ChLtHourlyCaptureRunSpecError(f"{label} is invalid")


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ChLtHourlyCaptureRunSpecError(f"{label} SHA-256 is invalid")


def _repo_path(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != relative_path:
        raise ChLtHourlyCaptureRunSpecError("repository relative path is unsafe")
    return root.joinpath(*pure.parts)


def _absolute(path: str | Path) -> Path:
    selected = Path(path).expanduser()
    return selected if selected.is_absolute() else Path(os.path.abspath(selected))


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


__all__ = [
    "CA_BUNDLE_SHA256",
    "DOCUMENT_SHA256",
    "OUTPUT_RELATIVE_PATH",
    "PRIOR_ATTEMPT_SHA256",
    "RETRY_DOCUMENT_SHA256",
    "RETRY_ID",
    "RETRY_OUTPUT_RELATIVE_PATH",
    "RETRY_RELATIVE_PATH",
    "RUN_SPEC_ID",
    "RUN_SPEC_RELATIVE_PATH",
    "ChLtHourlyCaptureRunSpecError",
    "VerifiedHourlyCaptureRunSpec",
    "VerifiedHourlyCurlRetry",
    "compute_retry_id",
    "compute_run_spec_id",
    "verify_hourly_capture_run_spec",
    "verify_hourly_curl_retry",
]
