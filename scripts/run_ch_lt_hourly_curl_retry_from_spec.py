"""Prepare/finalize the frozen curl retry for the CH hourly day-ahead capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfc_shaping.data.governed_lt_acquisition import (
    ENERGY_CHARTS_PRICE_URL,
    ENERGY_CHARTS_SOURCE_SYSTEM,
    CapturedProviderDocument,
    build_provider_transform_config,
    build_raw_envelope,
    energy_price_resolution_provenance,
    transform_provider_raw,
)
from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.pipeline.strict_structured_data import load_strict_json
from pfc_shaping.validation.ch_lt_hourly_capture_contract_v2 import (
    VerifiedHourlyCurlCaptureV2,
    verify_hourly_curl_capture_contract_v2,
)
from pfc_shaping.validation.ch_lt_hourly_capture_run_spec import (
    DOCUMENT_SHA256,
    RETRY_DOCUMENT_SHA256,
    ChLtHourlyCaptureRunSpecError,
    VerifiedHourlyCurlRetry,
    verify_hourly_curl_retry,
)

ATTEMPT_SCHEMA_VERSION = "ch_lt_hourly_curl_capture_attempt.v1"
ATTEMPT_SCHEMA_VERSION_V2 = "ch_lt_hourly_curl_capture_attempt.v2"
COMMAND_SCHEMA_VERSION = "ch_lt_hourly_curl_command.v1"
COMMAND_SCHEMA_VERSION_V2 = "ch_lt_hourly_curl_command.v2"
SUMMARY_SCHEMA_VERSION = "ch_lt_hourly_curl_capture_summary.v1"
SUMMARY_SCHEMA_VERSION_V2 = "ch_lt_hourly_curl_capture_summary.v2"
RECEIPT_SCHEMA_VERSION = "ch_lt_hourly_curl_capture_receipt.v1"
RECEIPT_SCHEMA_VERSION_V2 = "ch_lt_hourly_curl_capture_receipt.v2"
LOCAL_TRANSFORM_SCHEMA_VERSION_V2 = "ch_lt_hourly_local_transform_record.v2"
RECEIPT_STATUS = "COMPLETE_LOCAL_UNTRUSTED_CURL_CAPTURE_NOT_ADMITTED"
_MAX_BODY_BYTES = 8 * 1024 * 1024
_MAX_HEADERS_BYTES = 256 * 1024
_MAX_JSON_BYTES = 2 * 1024 * 1024
_STATUS_RE = re.compile(r"^HTTP/(?:1(?:\.0|\.1)?|2)\s+(\d{3})(?:\s|$)")
_ALLOWED_HEADERS = {
    "content-length",
    "content-type",
    "date",
    "etag",
    "last-modified",
    "x-request-id",
}


VerifiedCurlCapture = VerifiedHourlyCurlRetry | VerifiedHourlyCurlCaptureV2


def prepare_retry(retry: VerifiedCurlCapture) -> Path:
    """Create the one-shot output namespace and exact rendered curl command."""

    output = retry.output_directory
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    attempt_path = output / "capture-attempt.json"
    command_path = output / "curl-command.json"
    is_v2 = isinstance(retry, VerifiedHourlyCurlCaptureV2)
    attempt = {
        "schema_version": (
            ATTEMPT_SCHEMA_VERSION_V2 if is_v2 else ATTEMPT_SCHEMA_VERSION
        ),
        "status": "PREPARED_LOCAL_UNTRUSTED_CURL_CAPTURE",
        "acquisition_id": retry.acquisition_id,
        "source_role": retry.base.role,
        "start_utc": retry.base.start_utc,
        "end_utc_exclusive": retry.base.end_utc,
        "local_workstation_clock_trusted": False,
        "scientific_admission": False,
        "publication_authorized": False,
        "production_authorization": False,
        "promotion_gate": False,
    }
    if is_v2:
        attempt["capture_contract_id"] = retry.retry_id
        attempt["capture_contract_sha256"] = retry.document_sha256
        attempt["builder_input_authorized"] = False
        attempt["external_admission_input_candidate"] = False
        attempt["transport_attestation"] = False
        attempt["certificate_revocation_check_performed"] = False
    else:
        attempt["retry_id"] = retry.retry_id
        attempt["retry_document_sha256"] = retry.document_sha256
        attempt["base_run_spec_id"] = retry.base.run_spec_id
        attempt["base_run_spec_sha256"] = retry.base.document_sha256
    command = {
        "schema_version": (
            COMMAND_SCHEMA_VERSION_V2 if is_v2 else COMMAND_SCHEMA_VERSION
        ),
        "executable": "curl.exe",
        "arguments": list(retry.curl_arguments),
        "shell_interpolation_allowed": False,
        "redirect_limit": 0,
        "expected_body_path": str(retry.response_body_path),
        "expected_headers_path": str(retry.response_headers_path),
        "scientific_admission": False,
        "production_authorization": False,
        "promotion_gate": False,
    }
    if is_v2:
        command["capture_contract_id"] = retry.retry_id
        command["capture_contract_sha256"] = retry.document_sha256
        command["builder_input_authorized"] = False
        command["external_admission_input_candidate"] = False
        command["transport_attestation"] = False
        command["certificate_revocation_check_performed"] = False
    else:
        command["retry_id"] = retry.retry_id
    _write_json(attempt_path, attempt, label="CH hourly curl attempt")
    _write_json(command_path, command, label="CH hourly exact curl command")
    return command_path


def finalize_retry(retry: VerifiedCurlCapture) -> Path:
    """Validate captured curl bytes and emit summary/receipt without admission."""

    output = retry.output_directory
    if not output.is_dir():
        raise ChLtHourlyCaptureRunSpecError("retry output directory is unavailable")
    expected_before = {
        "capture-attempt.json",
        "curl-command.json",
        "provider-response.json",
        "response-headers.txt",
    }
    if {path.name for path in output.iterdir()} != expected_before:
        raise ChLtHourlyCaptureRunSpecError("curl retry pre-finalize inventory is not exact")
    attempt_payload = _stable(output / "capture-attempt.json", label="curl attempt")
    command_payload = _stable(output / "curl-command.json", label="curl command")
    attempt = _strict_mapping(attempt_payload, label="curl attempt")
    command = _strict_mapping(command_payload, label="curl command")
    _verify_prepared_metadata(attempt, command, retry=retry)
    body = read_stable_single_link_file(
        retry.response_body_path,
        label="curl provider body",
        max_bytes=_MAX_BODY_BYTES,
    )
    headers_payload = read_stable_single_link_file(
        retry.response_headers_path,
        label="curl response headers",
        max_bytes=_MAX_HEADERS_BYTES,
    )
    response_headers = _parse_response_headers(headers_payload)
    content_type = response_headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type != "application/json":
        raise ChLtHourlyCaptureRunSpecError("curl response media type is not JSON")
    content_encoding = response_headers.get("content-encoding", "").strip().lower()
    if content_encoding not in {"", "identity"}:
        raise ChLtHourlyCaptureRunSpecError("curl response encoding is not identity")
    if not body:
        raise ChLtHourlyCaptureRunSpecError("curl provider body is empty")
    received_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    start = datetime.fromisoformat(retry.base.start_utc.replace("Z", "+00:00"))
    end = datetime.fromisoformat(retry.base.end_utc.replace("Z", "+00:00"))
    last_source_interval = end - timedelta(minutes=retry.base.raw_cadence_minutes)
    request_parameters = {
        "bzn": "CH",
        "start": start.strftime("%Y-%m-%dT%H:%MZ"),
        "end": last_source_interval.strftime("%Y-%m-%dT%H:%MZ"),
    }
    admitted_headers = {
        key: value for key, value in response_headers.items() if key in _ALLOWED_HEADERS
    }
    document = CapturedProviderDocument(
        document_id="day_ahead_price",
        request_url=ENERGY_CHARTS_PRICE_URL,
        request_parameters=request_parameters,
        response_media_type=content_type,
        body=body,
        response_headers=admitted_headers,
        credential_reference=None,
        response_status_code=200,
        response_content_encoding="identity",
    )
    envelope = build_raw_envelope(
        acquisition_id=retry.acquisition_id,
        source_role="epex_ch",
        source_system=ENERGY_CHARTS_SOURCE_SYSTEM,
        source_locator=ENERGY_CHARTS_PRICE_URL,
        provider_id=ENERGY_CHARTS_SOURCE_SYSTEM,
        received_at_utc=received_at,
        documents=[document],
    )
    transform_config = build_provider_transform_config(
        role="epex_ch",
        start_utc=retry.base.start_utc,
        end_utc=retry.base.end_utc,
    )
    bronze = transform_provider_raw(
        envelope_payload=envelope,
        parser_config=transform_config,
    )
    resolution = energy_price_resolution_provenance(bronze, role="epex_ch")
    _verify_resolution(resolution, bronze_rows=len(bronze), retry=retry)
    is_v2 = isinstance(retry, VerifiedHourlyCurlCaptureV2)
    transform_record = {
        "schema_version": (
            LOCAL_TRANSFORM_SCHEMA_VERSION_V2 if is_v2 else "lt_provider_capture_spec.v1"
        ),
        "acquisition_id": retry.acquisition_id,
        "source_role": "epex_ch",
        "source_system": ENERGY_CHARTS_SOURCE_SYSTEM,
        "source_locator": ENERGY_CHARTS_PRICE_URL,
        "provider_id": ENERGY_CHARTS_SOURCE_SYSTEM,
        "received_at_utc": received_at,
        "start_utc": retry.base.start_utc,
        "end_utc": retry.base.end_utc,
        "documents": [
            {
                "document_id": "day_ahead_price",
                "body_path": str(retry.response_body_path),
                "request_url": ENERGY_CHARTS_PRICE_URL,
                "request_parameters": request_parameters,
                "credential_reference": None,
                "response_status_code": 200,
                "response_media_type": content_type,
                "response_content_encoding": "identity",
                "response_headers": admitted_headers,
            }
        ],
    }
    if is_v2:
        transform_record["capture_contract_id"] = retry.retry_id
        transform_record["capture_contract_sha256"] = retry.document_sha256
        transform_record["provider_body"] = _record(retry.response_body_path, body)
        transform_record["response_headers_record"] = _record(
            retry.response_headers_path, headers_payload
        )
        transform_record["builder_input_authorized"] = False
        transform_record["external_admission_input_candidate"] = False
        transform_record["transport_attestation"] = False
        transform_record["certificate_revocation_check_performed"] = False
    transform_path = output / (
        "local-transform-record.json" if is_v2 else "capture-spec.json"
    )
    transform_payload = _canonical_json(transform_record)
    _write_bytes(
        transform_path,
        transform_payload,
        label="CH hourly curl local transform record" if is_v2 else "CH hourly curl capture spec",
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION_V2 if is_v2 else SUMMARY_SCHEMA_VERSION,
        "status": "COMPLETE_LOCAL_UNTRUSTED_CURL_CAPTURE",
        "acquisition_id": retry.acquisition_id,
        "source_role": "epex_ch",
        "source_product": "DAY_AHEAD_PRICE_CH",
        "received_at_utc": received_at,
        "start_utc": retry.base.start_utc,
        "end_utc_exclusive": retry.base.end_utc,
        "source_cadence_seconds": retry.base.raw_cadence_minutes * 60,
        "source_observation_count": retry.base.expected_source_observation_count,
        "resolution_provenance": resolution,
        "provider_body": _record(retry.response_body_path, body),
        "response_headers": _record(retry.response_headers_path, headers_payload),
        "native_quarter_hour_truth_eligible": False,
        "timestamp_authority": "LOCAL_WORKSTATION_CLOCK_UNTRUSTED",
        "publication_status": "NOT_PUBLISHED",
        "scientific_admission": False,
        "model_selection_authorized": False,
        "publication_authorized": False,
        "production_authorization": False,
        "promotion_gate": False,
        "future_holdout_consumed": False,
        "t057_consumed": False,
    }
    if is_v2:
        summary["capture_contract_id"] = retry.retry_id
        summary["capture_contract_sha256"] = retry.document_sha256
        summary["local_transform_record"] = _record(transform_path, transform_payload)
        summary["validated_in_memory_proxy_rows"] = int(len(bronze))
        summary["derived_proxy_materialized"] = False
        summary["builder_input_authorized"] = False
        summary["external_admission_input_candidate"] = False
        summary["transport_attestation"] = False
        summary["certificate_revocation_check_performed"] = False
        summary["as_of_interpretation"] = (
            "CONTRACT_WINDOW_ANCHOR_NOT_CAPTURE_OR_AVAILABLE_AT_TIMESTAMP"
        )
        summary["native_hourly_truth_eligibility_status"] = (
            "BLOCKED_PENDING_PRODUCT_IDENTITY_REVISION_POLICY_"
            "TRUSTED_AVAILABILITY_AND_EXTERNAL_ADMISSION"
        )
    else:
        summary["retry_id"] = retry.retry_id
        summary["capture_spec"] = _record(transform_path, transform_payload)
        summary["validated_bronze_rows"] = int(len(bronze))
        summary["native_hourly_truth_eligible_after_external_admission"] = True
    summary_path = output / "capture-summary.json"
    summary_payload = _canonical_json(summary)
    _write_bytes(summary_path, summary_payload, label="CH hourly curl summary")
    is_v2 = isinstance(retry, VerifiedHourlyCurlCaptureV2)
    receipt = {
        "schema_version": (
            RECEIPT_SCHEMA_VERSION_V2 if is_v2 else RECEIPT_SCHEMA_VERSION
        ),
        "status": RECEIPT_STATUS,
        "capture_attempt": _record(output / "capture-attempt.json", attempt_payload),
        "curl_command": _record(output / "curl-command.json", command_payload),
        "provider_body": _record(retry.response_body_path, body),
        "response_headers": _record(retry.response_headers_path, headers_payload),
        "capture_summary": _record(summary_path, summary_payload),
        "source_observation_count": retry.base.expected_source_observation_count,
        "native_quarter_hour_truth_eligible": False,
        "local_capture_completed": True,
        "local_workstation_clock_trusted": False,
        "trusted_time_receipt_present": False,
        "independent_signature_present": False,
        "builder_inaccessible_immutable_namespace_present": False,
        "external_cas_receipt_present": False,
        "fresh_authenticated_head_present": False,
        "scientific_execution_authorized": False,
        "scientific_admission": False,
        "model_selection_authorized": False,
        "publication_authorized": False,
        "production_authorization": False,
        "promotion_gate": False,
        "future_holdout_consumed": False,
        "t057_consumed": False,
        "receipt_last_written": True,
    }
    if is_v2:
        receipt["capture_contract_id"] = retry.retry_id
        receipt["capture_contract_sha256"] = retry.document_sha256
        receipt["local_transform_record"] = _record(transform_path, transform_payload)
        receipt["validated_in_memory_proxy_rows"] = (
            retry.base.expected_bronze_observation_count
        )
        receipt["derived_proxy_materialized"] = False
        receipt["builder_input_authorized"] = False
        receipt["external_admission_input_candidate"] = False
        receipt["transport_attestation"] = False
        receipt["certificate_revocation_check_performed"] = False
        receipt["native_hourly_truth_eligibility_status"] = (
            "BLOCKED_PENDING_PRODUCT_IDENTITY_REVISION_POLICY_"
            "TRUSTED_AVAILABILITY_AND_EXTERNAL_ADMISSION"
        )
    else:
        receipt["retry_id"] = retry.retry_id
        receipt["retry_document_sha256"] = retry.document_sha256
        receipt["base_run_spec_id"] = retry.base.run_spec_id
        receipt["base_run_spec_sha256"] = retry.base.document_sha256
        receipt["prior_failed_attempt_sha256"] = (
            "39b39ba5e044a992ba28349519a7341b103a5602a302c692671dd71106f0e27b"
        )
        receipt["capture_spec"] = _record(transform_path, transform_payload)
        receipt["bronze_observation_count"] = (
            retry.base.expected_bronze_observation_count
        )
        receipt["native_hourly_truth_eligible_after_external_admission"] = True
    receipt_path = output / "run-receipt.json"
    _write_json(receipt_path, receipt, label="CH hourly curl run receipt")
    expected_after = expected_before | {
        "local-transform-record.json" if is_v2 else "capture-spec.json",
        "capture-summary.json",
        "run-receipt.json",
    }
    if {path.name for path in output.iterdir()} != expected_after:
        raise ChLtHourlyCaptureRunSpecError("curl retry final inventory is not exact")
    return receipt_path


def _verify_prepared_metadata(
    attempt: Mapping[str, object],
    command: Mapping[str, object],
    *,
    retry: VerifiedCurlCapture,
) -> None:
    is_v2 = isinstance(retry, VerifiedHourlyCurlCaptureV2)
    attempt_keys = {
        "schema_version",
        "status",
        "acquisition_id",
        "source_role",
        "start_utc",
        "end_utc_exclusive",
        "local_workstation_clock_trusted",
        "scientific_admission",
        "publication_authorized",
        "production_authorization",
        "promotion_gate",
    }
    command_keys = {
        "schema_version",
        "executable",
        "arguments",
        "shell_interpolation_allowed",
        "redirect_limit",
        "expected_body_path",
        "expected_headers_path",
        "scientific_admission",
        "production_authorization",
        "promotion_gate",
    }
    if is_v2:
        v2_keys = {
            "capture_contract_id",
            "capture_contract_sha256",
            "builder_input_authorized",
            "external_admission_input_candidate",
            "transport_attestation",
            "certificate_revocation_check_performed",
        }
        attempt_keys |= v2_keys
        command_keys |= v2_keys
    else:
        attempt_keys |= {
            "retry_id",
            "retry_document_sha256",
            "base_run_spec_id",
            "base_run_spec_sha256",
        }
        command_keys.add("retry_id")
    _exact_keys(attempt, attempt_keys, label="curl attempt")
    _exact_keys(command, command_keys, label="curl command")
    expected_attempt_schema = (
        ATTEMPT_SCHEMA_VERSION_V2 if is_v2 else ATTEMPT_SCHEMA_VERSION
    )
    expected_command_schema = (
        COMMAND_SCHEMA_VERSION_V2 if is_v2 else COMMAND_SCHEMA_VERSION
    )
    _equal(attempt.get("schema_version"), expected_attempt_schema, "attempt schema")
    if is_v2:
        _equal(
            attempt.get("capture_contract_id"), retry.retry_id, "attempt contract ID"
        )
        _equal(
            attempt.get("capture_contract_sha256"),
            retry.document_sha256,
            "attempt contract hash",
        )
    else:
        _equal(attempt.get("retry_id"), retry.retry_id, "attempt retry ID")
    _equal(
        attempt.get("status"),
        "PREPARED_LOCAL_UNTRUSTED_CURL_CAPTURE",
        "attempt status",
    )
    _equal(attempt.get("acquisition_id"), retry.acquisition_id, "attempt acquisition")
    _equal(attempt.get("source_role"), retry.base.role, "attempt source role")
    _equal(attempt.get("start_utc"), retry.base.start_utc, "attempt start")
    _equal(attempt.get("end_utc_exclusive"), retry.base.end_utc, "attempt end")
    _equal(attempt.get("local_workstation_clock_trusted"), False, "attempt clock")
    _equal(attempt.get("scientific_admission"), False, "attempt admission")
    _equal(attempt.get("publication_authorized"), False, "attempt publication")
    _equal(attempt.get("production_authorization"), False, "attempt production")
    _equal(attempt.get("promotion_gate"), False, "attempt promotion")
    _equal(command.get("schema_version"), expected_command_schema, "command schema")
    if is_v2:
        _equal(
            command.get("capture_contract_id"), retry.retry_id, "command contract ID"
        )
        _equal(
            command.get("capture_contract_sha256"),
            retry.document_sha256,
            "command contract hash",
        )
    else:
        _equal(command.get("retry_id"), retry.retry_id, "command retry ID")
    _equal(command.get("executable"), "curl.exe", "curl executable")
    _equal(command.get("arguments"), list(retry.curl_arguments), "curl arguments")
    _equal(command.get("shell_interpolation_allowed"), False, "curl interpolation")
    _equal(command.get("redirect_limit"), 0, "curl redirects")
    _equal(command.get("scientific_admission"), False, "command admission")
    _equal(command.get("production_authorization"), False, "command production")
    _equal(command.get("promotion_gate"), False, "command promotion")
    _equal(
        command.get("expected_body_path"),
        str(retry.response_body_path),
        "command body path",
    )
    _equal(
        command.get("expected_headers_path"),
        str(retry.response_headers_path),
        "command headers path",
    )
    if is_v2:
        for field in (
            "builder_input_authorized",
            "external_admission_input_candidate",
            "transport_attestation",
            "certificate_revocation_check_performed",
        ):
            _equal(attempt.get(field), False, f"attempt {field}")
            _equal(command.get(field), False, f"command {field}")


def _parse_response_headers(payload: bytes) -> dict[str, str]:
    try:
        text = payload.decode("iso-8859-1")
    except UnicodeDecodeError as exc:
        raise ChLtHourlyCaptureRunSpecError("curl headers are not decodable") from exc
    lines = [line.rstrip("\r") for line in text.split("\n")]
    status_lines = [line for line in lines if line.startswith("HTTP/")]
    if len(status_lines) != 1:
        raise ChLtHourlyCaptureRunSpecError("curl response chain is not exactly one hop")
    match = _STATUS_RE.match(status_lines[0])
    if match is None or int(match.group(1)) != 200:
        raise ChLtHourlyCaptureRunSpecError("curl HTTP status is not 200")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        if ":" not in line:
            raise ChLtHourlyCaptureRunSpecError("curl response header is malformed")
        name, raw_value = line.split(":", 1)
        key = name.strip().lower()
        if not key or key in headers:
            raise ChLtHourlyCaptureRunSpecError("curl response header identity is invalid")
        headers[key] = raw_value.strip()
    return headers


def _verify_resolution(
    value: Mapping[str, object], *, bronze_rows: int, retry: VerifiedCurlCapture
) -> None:
    expected = {
        "schema_version": "lt_observation_resolution_provenance.v2",
        "role": "epex_ch",
        "source_cadence_seconds": 3600,
        "output_cadence_seconds": 900,
        "source_observation_count": retry.base.expected_source_observation_count,
        "output_observation_count": retry.base.expected_bronze_observation_count,
        "resampling_method": "FORWARD_FILL_WITHIN_SOURCE_INTERVAL",
        "output_sampling_kind": "UPSAMPLED_STEPWISE_PROXY",
        "native_quarter_hour_cadence_eligible": False,
        "native_hourly_cadence_eligible": True,
        "hourly_aggregation_eligible": True,
        "native_quarter_hour_truth_eligible": False,
        "product_identity_status": "EXTERNAL_PRODUCT_AUCTION_SESSION_IDENTITY_UNVERIFIED",
        "quarter_hour_truth_blocker": "SOURCE_CADENCE_IS_HOURLY",
        "scientific_use_class": "NATIVE_HOURLY_PRICE_QH_TRANSPORT_PROXY_ONLY",
    }
    _equal(dict(value), expected, "resolution provenance")
    if bronze_rows != retry.base.expected_bronze_observation_count:
        raise ChLtHourlyCaptureRunSpecError("bronze observation count is invalid")


def _load_retry(args: argparse.Namespace) -> VerifiedCurlCapture:
    repo_root = Path(__file__).resolve().parents[1]
    if args.contract_v2 is not None:
        if args.expected_contract_sha256 is None:
            raise ChLtHourlyCaptureRunSpecError("v2 contract SHA-256 is required")
        if args.run_spec is not None or args.retry_spec is not None:
            raise ChLtHourlyCaptureRunSpecError("v1 and v2 capture contracts cannot be mixed")
        return verify_hourly_curl_capture_contract_v2(
            repo_root=repo_root,
            contract_path=args.contract_v2,
            expected_contract_sha256=args.expected_contract_sha256,
        )
    if args.run_spec is None or args.retry_spec is None:
        raise ChLtHourlyCaptureRunSpecError("both v1 run and retry specs are required")
    return verify_hourly_curl_retry(
        repo_root=repo_root,
        base_run_spec_path=args.run_spec,
        expected_base_sha256=args.expected_run_spec_sha256,
        retry_path=args.retry_spec,
        expected_retry_sha256=args.expected_retry_spec_sha256,
    )


def _strict_mapping(payload: bytes, *, label: str) -> Mapping[str, object]:
    try:
        parsed = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChLtHourlyCaptureRunSpecError(f"{label} is invalid JSON") from exc
    if not isinstance(parsed, Mapping) or not all(isinstance(key, str) for key in parsed):
        raise ChLtHourlyCaptureRunSpecError(f"{label} must be an object")
    return parsed


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ChLtHourlyCaptureRunSpecError(f"{label} keys are invalid")


def _stable(path: Path, *, label: str) -> bytes:
    return read_stable_single_link_file(path, label=label, max_bytes=_MAX_JSON_BYTES)


def _equal(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ChLtHourlyCaptureRunSpecError(f"{label} is invalid")


def _record(path: Path, payload: bytes) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _canonical_json(value: object) -> bytes:
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


def _write_bytes(path: Path, payload: bytes, *, label: str) -> None:
    write_durable_exact_bytes(path, payload, label=label)


def _write_json(path: Path, value: object, *, label: str) -> None:
    _write_bytes(path, _canonical_json(value), label=label)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "finalize"))
    parser.add_argument("--run-spec", type=Path)
    parser.add_argument("--expected-run-spec-sha256", default=DOCUMENT_SHA256)
    parser.add_argument("--retry-spec", type=Path)
    parser.add_argument("--expected-retry-spec-sha256", default=RETRY_DOCUMENT_SHA256)
    parser.add_argument("--contract-v2", type=Path)
    parser.add_argument("--expected-contract-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        retry = _load_retry(args)
        result = prepare_retry(retry) if args.action == "prepare" else finalize_retry(retry)
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "ch_lt_hourly_curl_retry_failure.v1",
                    "status": "LOCAL_CURL_RETRY_FAILED_NO_ADMISSION",
                    "scientific_admission": False,
                    "publication_authorized": False,
                    "production_authorization": False,
                    "promotion_gate": False,
                    "error": str(exc),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    if args.action == "prepare":
        command = _strict_mapping(_stable(result, label="curl command"), label="curl command")
        print(json.dumps(command, sort_keys=True, separators=(",", ":")))
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
