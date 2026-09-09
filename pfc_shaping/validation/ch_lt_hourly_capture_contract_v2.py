"""Hash-bound reusable contract for local CH hourly day-ahead curl captures."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlencode

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json

SCHEMA_VERSION = "ch_lt_hourly_day_ahead_curl_capture.v2"
LIFECYCLE_STATE = "FROZEN_LOCAL_CURL_CAPTURE_NOT_ADMISSION"
_ENDPOINT = "https://api.energy-charts.info/price"
_MAX_CONTRACT_BYTES = 256 * 1024
_MAX_CA_BYTES = 2 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PORTABLE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_TOP_LEVEL_KEYS = {
    "schema_version",
    "contract_id",
    "lifecycle_state",
    "as_of_utc",
    "market",
    "timezone",
    "source",
    "window",
    "execution",
    "tls",
    "transport",
    "intended_use",
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
_FALSE_AUTHORITY_FIELDS = (
    "scientific_execution_authorized",
    "scientific_admission",
    "model_selection_authorized",
    "publication_authorized",
    "production_authorization",
    "promotion_gate",
    "future_holdout_consumed",
    "t057_consumed",
)


class ChLtHourlyCaptureContractV2Error(ValueError):
    """Raised when a reusable frozen capture contract is ambiguous or unsafe."""


@dataclass(frozen=True)
class VerifiedHourlyCaptureBaseV2:
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


@dataclass(frozen=True)
class VerifiedHourlyCurlCaptureV2:
    retry_id: str
    document_sha256: str
    base: VerifiedHourlyCaptureBaseV2
    acquisition_id: str
    output_directory: Path
    response_body_path: Path
    response_headers_path: Path
    curl_arguments: tuple[str, ...]


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def compute_contract_id(document: Mapping[str, object]) -> str:
    unsigned = dict(document)
    unsigned.pop("contract_id", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def verify_hourly_curl_capture_contract_v2(
    *,
    repo_root: str | Path,
    contract_path: str | Path,
    expected_contract_sha256: str,
) -> VerifiedHourlyCurlCaptureV2:
    root = _absolute_no_links(repo_root)
    selected_contract = _absolute_no_links(contract_path)
    _require_contract_location(selected_contract, root)
    _require_sha256(expected_contract_sha256, label="caller-held contract")
    payload = read_stable_single_link_file(
        selected_contract,
        label="CH hourly curl capture v2 contract",
        max_bytes=_MAX_CONTRACT_BYTES,
    )
    observed_hash = hashlib.sha256(payload).hexdigest()
    if observed_hash != expected_contract_sha256:
        raise ChLtHourlyCaptureContractV2Error("capture contract SHA-256 mismatch")
    document = _strict_mapping(payload, label="capture contract")
    _exact_keys(document, _TOP_LEVEL_KEYS, label="capture contract")
    _equal(document.get("schema_version"), SCHEMA_VERSION, "schema version")
    _equal(document.get("lifecycle_state"), LIFECYCLE_STATE, "lifecycle state")
    contract_id = _text(document.get("contract_id"), label="contract ID")
    _require_sha256(contract_id, label="contract ID")
    if compute_contract_id(document) != contract_id:
        raise ChLtHourlyCaptureContractV2Error("contract ID does not bind content")
    _equal(document.get("market"), "CH", "market")
    _equal(document.get("timezone"), "Europe/Zurich", "timezone")
    as_of = _utc_datetime(document.get("as_of_utc"), label="as-of")

    source = _mapping(document.get("source"), label="source")
    _equal(
        dict(source),
        {
            "role": "epex_ch",
            "source_system": "ENERGY_CHARTS_API",
            "endpoint": _ENDPOINT,
            "bidding_zone": "CH",
            "product": "DAY_AHEAD_PRICE_CH",
            "product_cadence_seconds": 3600,
            "product_identity_status": "EXTERNAL_PRODUCT_AUCTION_SESSION_IDENTITY_UNVERIFIED",
        },
        "source contract",
    )

    window = _mapping(document.get("window"), label="window")
    _exact_keys(
        window,
        {
            "start_utc",
            "end_utc_exclusive",
            "duration_hours",
            "expected_source_observation_count",
            "dst_interpretation",
        },
        label="window",
    )
    start = _utc_datetime(window.get("start_utc"), label="window start")
    end = _utc_datetime(window.get("end_utc_exclusive"), label="window end")
    if start.minute or start.second or start.microsecond or end.minute or end.second or end.microsecond:
        raise ChLtHourlyCaptureContractV2Error("capture window is not hour-aligned")
    duration_seconds = int((end - start).total_seconds())
    if duration_seconds <= 0 or duration_seconds > 32 * 24 * 3600:
        raise ChLtHourlyCaptureContractV2Error("capture window duration is invalid")
    duration_hours = duration_seconds // 3600
    if duration_seconds % 3600:
        raise ChLtHourlyCaptureContractV2Error("capture window duration is fractional")
    _equal(window.get("duration_hours"), duration_hours, "window duration")
    _equal(
        window.get("expected_source_observation_count"),
        duration_hours,
        "source observation count",
    )
    _equal(
        window.get("dst_interpretation"),
        "UTC_INTERVAL_MAPPED_TO_EUROPE_ZURICH_DELIVERY_HOURS",
        "DST interpretation",
    )
    if end > as_of:
        raise ChLtHourlyCaptureContractV2Error("capture truth extends beyond contract as-of")

    execution = _mapping(document.get("execution"), label="execution")
    _exact_keys(
        execution,
        {
            "wrapper",
            "acquisition_id",
            "raw_cadence_minutes",
            "expected_bronze_observation_count",
            "expected_bronze_cadence_seconds",
            "bronze_sampling_kind",
            "native_hourly_truth_eligibility_status",
            "native_quarter_hour_truth_eligible",
            "builder_input_authorized",
            "external_admission_input_candidate",
            "transport_attestation",
            "output_directory_relative",
            "output_namespace_reuse_forbidden",
        },
        label="execution",
    )
    _equal(
        execution.get("wrapper"),
        "scripts/run_ch_lt_hourly_curl_retry_from_spec.py",
        "wrapper",
    )
    acquisition_id = _text(execution.get("acquisition_id"), label="acquisition ID")
    if _PORTABLE_ID_RE.fullmatch(acquisition_id) is None:
        raise ChLtHourlyCaptureContractV2Error("acquisition ID is not portable")
    _equal(execution.get("raw_cadence_minutes"), 60, "raw cadence")
    expected_bronze = duration_hours * 4
    _equal(
        execution.get("expected_bronze_observation_count"),
        expected_bronze,
        "bronze observation count",
    )
    _equal(execution.get("expected_bronze_cadence_seconds"), 900, "bronze cadence")
    _equal(
        execution.get("bronze_sampling_kind"),
        "UPSAMPLED_STEPWISE_PROXY",
        "bronze sampling kind",
    )
    _equal(
        execution.get("native_hourly_truth_eligibility_status"),
        (
            "BLOCKED_PENDING_PRODUCT_IDENTITY_REVISION_POLICY_"
            "TRUSTED_AVAILABILITY_AND_EXTERNAL_ADMISSION"
        ),
        "hourly eligibility",
    )
    _equal(execution.get("builder_input_authorized"), False, "builder input")
    _equal(
        execution.get("external_admission_input_candidate"),
        False,
        "external admission input",
    )
    _equal(execution.get("transport_attestation"), False, "transport attestation")
    _equal(
        execution.get("native_quarter_hour_truth_eligible"),
        False,
        "quarter-hour eligibility",
    )
    _equal(execution.get("output_namespace_reuse_forbidden"), True, "output reuse")
    output = _repo_relative_path(
        root,
        execution.get("output_directory_relative"),
        label="output directory",
        required_prefix=("output", "phase14"),
    )

    tls = _mapping(document.get("tls"), label="TLS")
    _exact_keys(tls, {"ca_bundle_path", "ca_bundle_sha256", "ca_bundle_size_bytes"}, label="TLS")
    ca_bundle = _absolute_no_links(_text(tls.get("ca_bundle_path"), label="CA bundle path"))
    ca_payload = read_stable_single_link_file(
        ca_bundle,
        label="capture CA bundle",
        max_bytes=_MAX_CA_BYTES,
    )
    ca_hash = _text(tls.get("ca_bundle_sha256"), label="CA bundle SHA-256")
    _require_sha256(ca_hash, label="CA bundle SHA-256")
    if hashlib.sha256(ca_payload).hexdigest() != ca_hash:
        raise ChLtHourlyCaptureContractV2Error("CA bundle SHA-256 mismatch")
    _equal(tls.get("ca_bundle_size_bytes"), len(ca_payload), "CA bundle size")

    transport = _mapping(document.get("transport"), label="transport")
    _equal(
        dict(transport),
        {
            "executable": "curl.exe",
            "redirect_limit": 0,
            "protocol": "https",
            "minimum_tls_version": "1.2",
            "response_encoding_requested": "identity",
            "provider_body_max_bytes": 8388608,
            "certificate_revocation_check": "DISABLED_LOCAL_UNTRUSTED_CAPTURE_ONLY",
        },
        "transport contract",
    )
    _verify_intended_use(document.get("intended_use"))
    _verify_external_authorities(document.get("external_authorities"))
    _equal(
        document.get("local_capture_execution_authorized"),
        True,
        "local execution authority",
    )
    for field in _FALSE_AUTHORITY_FIELDS:
        _equal(document.get(field), False, field)

    body_path = output / "provider-response.json"
    headers_path = output / "response-headers.txt"
    provider_end = end.replace(tzinfo=timezone.utc).timestamp() - 3600
    provider_end_text = datetime.fromtimestamp(provider_end, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%MZ"
    )
    query = urlencode(
        {
            "bzn": "CH",
            "start": start.strftime("%Y-%m-%dT%H:%MZ"),
            "end": provider_end_text,
        }
    )
    request_url = f"{_ENDPOINT}?{query}"
    curl_arguments = (
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
        str(ca_bundle),
        "--dump-header",
        str(headers_path),
        "--output",
        str(body_path),
        request_url,
    )
    base = VerifiedHourlyCaptureBaseV2(
        run_spec_id=contract_id,
        document_sha256=observed_hash,
        role="epex_ch",
        start_utc=_iso(start),
        end_utc=_iso(end),
        raw_cadence_minutes=60,
        acquisition_id=acquisition_id,
        output_directory=output,
        ca_bundle_path=ca_bundle,
        expected_source_observation_count=duration_hours,
        expected_bronze_observation_count=expected_bronze,
    )
    return VerifiedHourlyCurlCaptureV2(
        retry_id=contract_id,
        document_sha256=observed_hash,
        base=base,
        acquisition_id=acquisition_id,
        output_directory=output,
        response_body_path=body_path,
        response_headers_path=headers_path,
        curl_arguments=curl_arguments,
    )


def _verify_intended_use(value: object) -> None:
    _equal(
        dict(_mapping(value, label="intended use")),
        {
            "allowed": ["LOCAL_HOURLY_SHAPE_DIAGNOSTIC"],
            "forbidden": [
                "MONTHLY_LEVEL_AUTHORITY",
                "NATIVE_QUARTER_HOUR_TRUTH",
                "EXTERNAL_ADMISSION_INPUT_CANDIDATE",
                "MODEL_SELECTION_BEFORE_EXTERNAL_ADMISSION",
                "FUTURE_HOLDOUT_TRUTH_OPEN",
                "SNAPSHOT_PUBLICATION",
                "CANDIDATE_PROMOTION",
                "PRODUCTION_PROMOTION",
            ],
        },
        "intended use",
    )


def _verify_external_authorities(value: object) -> None:
    expected = {
        "trusted_time_receipt_present": False,
        "independent_signature_present": False,
        "builder_inaccessible_immutable_namespace_present": False,
        "external_cas_receipt_present": False,
        "fresh_authenticated_head_present": False,
    }
    _equal(dict(_mapping(value, label="external authorities")), expected, "external authorities")


def _repo_relative_path(
    root: Path,
    value: object,
    *,
    label: str,
    required_prefix: tuple[str, ...],
) -> Path:
    text = _text(value, label=label)
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts or pure.parts[: len(required_prefix)] != required_prefix:
        raise ChLtHourlyCaptureContractV2Error(f"{label} is outside its governed prefix")
    selected = _absolute_no_links(root.joinpath(*pure.parts))
    if os.path.commonpath((str(selected), str(root))) != str(root):
        raise ChLtHourlyCaptureContractV2Error(f"{label} escaped the repo")
    return selected


def _require_contract_location(contract: Path, root: Path) -> None:
    expected_parent = root / ".planning" / "phases" / "14-lt-audit-remediation"
    if contract.parent != expected_parent or not contract.name.startswith(
        "CH-LT-HOURLY-DAY-AHEAD-CURL-CAPTURE-"
    ):
        raise ChLtHourlyCaptureContractV2Error("capture contract path is not canonical")


def _absolute_no_links(path: str | Path) -> Path:
    selected = Path(os.path.abspath(Path(path).expanduser()))
    return assert_absolute_path_has_no_links(selected)


def _strict_mapping(payload: bytes, *, label: str) -> Mapping[str, object]:
    try:
        parsed = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChLtHourlyCaptureContractV2Error(f"{label} is invalid JSON") from exc
    return _mapping(parsed, label=label)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ChLtHourlyCaptureContractV2Error(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ChLtHourlyCaptureContractV2Error(f"{label} keys are invalid")


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ChLtHourlyCaptureContractV2Error(f"{label} must be non-empty text")
    return value


def _require_sha256(value: str, *, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise ChLtHourlyCaptureContractV2Error(f"{label} is not lowercase SHA-256")


def _utc_datetime(value: object, *, label: str) -> datetime:
    text = _text(value, label=label)
    if not text.endswith("Z"):
        raise ChLtHourlyCaptureContractV2Error(f"{label} must use UTC Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ChLtHourlyCaptureContractV2Error(f"{label} is invalid") from exc
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _equal(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ChLtHourlyCaptureContractV2Error(f"{label} is invalid")


__all__ = [
    "ChLtHourlyCaptureContractV2Error",
    "LIFECYCLE_STATE",
    "SCHEMA_VERSION",
    "VerifiedHourlyCaptureBaseV2",
    "VerifiedHourlyCurlCaptureV2",
    "canonical_json_bytes",
    "compute_contract_id",
    "verify_hourly_curl_capture_contract_v2",
]
