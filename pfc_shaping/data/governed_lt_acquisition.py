"""Exact-byte, deterministic acquisition boundary for governed LT inputs.

This module deliberately contains no network, signing, publication, or cache I/O.
It turns an immutable provider response envelope into the canonical bronze frame
that the existing LT replay layer can independently transform.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.resources
import json
import math
import platform
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from io import BytesIO, StringIO
from pathlib import Path
from typing import Callable, Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from pfc_shaping.parquet_safety import validate_parquet_allocation_budget
from pfc_shaping.path_safety import read_stable_single_link_file

RAW_ENVELOPE_SCHEMA = "lt_raw_envelope.v1"
PROVIDER_TRANSFORM_CONFIG_SCHEMA = "lt_provider_transform_config.v1"
OBSERVATION_RESOLUTION_PROVENANCE_SCHEMA = (
    "lt_observation_resolution_provenance.v2"
)
OBSERVATION_RESOLUTION_PROVENANCE_ATTR = "lt_observation_resolution_provenance"
ENERGY_CHARTS_PRICE_TRANSFORM = "pfc.provider.energy_charts_price.v1"
ENTSOE_XML_BUNDLE_TRANSFORM = "pfc.provider.entsoe_xml_bundle.v1"
SFOE_OGD17_CSV_TRANSFORM = "pfc.provider.sfoe_ogd17_csv.v1"

ENERGY_CHARTS_SOURCE_SYSTEM = "ENERGY_CHARTS_API"
ENTSOE_SOURCE_SYSTEM = "ENTSOE_TRANSPARENCY_PLATFORM"
SFOE_SOURCE_SYSTEM = "SFOE_BFE_OGD17"

ENERGY_CHARTS_PRICE_URL = "https://api.energy-charts.info/price"
ENTSOE_API_URL = "https://web-api.tp.entsoe.eu/api"
SFOE_OGD17_URL = (
    "https://www.uvek-gis.admin.ch/BFE/ogd/17/"
    "ogd17_fuellungsgrad_speicherseen.csv"
)

_RAW_ENVELOPE_KEYS = {
    "schema_version",
    "acquisition_id",
    "source_role",
    "source_system",
    "source_locator",
    "provider_id",
    "received_at_utc",
    "documents",
}
_DOCUMENT_KEYS = {
    "document_id",
    "ordinal",
    "request",
    "response",
    "body_base64",
    "body_sha256",
    "body_size_bytes",
}
_REQUEST_KEYS = {"method", "url", "parameters", "credential_reference"}
_RESPONSE_KEYS = {"status_code", "media_type", "content_encoding", "headers"}
_CONFIG_KEYS = {
    "schema_version",
    "transform_id",
    "role",
    "source_system",
    "start_utc",
    "end_utc",
    "output_frequency_seconds",
    "runtime_fingerprint",
}
_RESOLUTION_PROVENANCE_KEYS = {
    "schema_version",
    "role",
    "source_cadence_seconds",
    "output_cadence_seconds",
    "source_observation_count",
    "output_observation_count",
    "resampling_method",
    "output_sampling_kind",
    "native_quarter_hour_cadence_eligible",
    "native_hourly_cadence_eligible",
    "hourly_aggregation_eligible",
    "native_quarter_hour_truth_eligible",
    "product_identity_status",
    "quarter_hour_truth_blocker",
    "scientific_use_class",
}
_PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE_PARAMETER = re.compile(
    r"(?:api.?key|authorization|password|secret|security.?token|token)", re.IGNORECASE
)
_ENV_CREDENTIAL_REFERENCE = re.compile(r"^env:[A-Z][A-Z0-9_]{0,127}$")
_MEDIA_TYPE = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")
_ALLOWED_RESPONSE_HEADERS = frozenset(
    {"content-length", "content-type", "date", "etag", "last-modified", "x-request-id"}
)
_ENERGY_CHARTS_ZONE_BY_ROLE = {
    "epex_ch": "CH",
    "epex_de": "DE-LU",
    "epex_at": "AT",
    "epex_fr": "FR",
    "epex_it": "IT-North",
}
_ENTSOE_CH_DOMAIN = "10YCH-SWISSGRIDZ"
_ENTSOE_DOCUMENT_IDS = ("actual_load_ch", "actual_generation_ch")
MAX_PROVIDER_DOCUMENT_BYTES = 32 * 1024 * 1024
MAX_PROVIDER_DOCUMENT_TOTAL_BYTES = 64 * 1024 * 1024
MAX_RAW_ENVELOPE_BYTES = 96 * 1024 * 1024
MAX_PROVIDER_CAPTURE_SPEC_BYTES = 1024 * 1024
MAX_ENERGY_CHARTS_DAY_AHEAD_LEAD = pd.Timedelta(hours=48)
_MAX_PROVIDER_WINDOW = {
    ENERGY_CHARTS_PRICE_TRANSFORM: pd.Timedelta(days=32),
    ENTSOE_XML_BUNDLE_TRANSFORM: pd.Timedelta(days=366 * 15),
    SFOE_OGD17_CSV_TRANSFORM: pd.Timedelta(days=366 * 30),
}
_MAX_PROVIDER_DOCUMENT_BYTES_BY_ROLE = {
    "epex_ch": 2 * 1024 * 1024,
    "epex_de": 2 * 1024 * 1024,
    "epex_at": 2 * 1024 * 1024,
    "epex_fr": 2 * 1024 * 1024,
    "epex_it": 2 * 1024 * 1024,
    "entso": 16 * 1024 * 1024,
    "hydro": 8 * 1024 * 1024,
}
_PROVIDER_BRONZE_MAX_ROWS = {
    "epex_ch": 4_096,
    "epex_de": 4_096,
    "epex_at": 4_096,
    "epex_fr": 4_096,
    "epex_it": 4_096,
    "entso": 600_000,
    "hydro": 2_000,
}
_PROVIDER_BRONZE_PHYSICAL_TYPES = {
    "BOOLEAN",
    "INT32",
    "INT64",
    "FLOAT",
    "DOUBLE",
}
_ENTSOE_DOCUMENT_IDENTITY = {
    "load_mw": {
        "namespace": "urn:iec62325.351:tc57wg16:451-6:load-document:3:0",
        "document_type": "A65",
        "domain_field": "outBiddingZone_Domain.mRID",
    },
    "generation": {
        "namespace": "urn:iec62325.351:tc57wg16:451-6:generation-document:3:0",
        "document_type": "A75",
        "domain_field": "in_Domain.mRID",
    },
}


class GovernedLTAcquisitionError(ValueError):
    """Raised when exact provider bytes cannot produce an eligible LT input."""


@dataclass(frozen=True)
class CapturedProviderDocument:
    """One exact HTTP response captured by an authority outside this module."""

    document_id: str
    request_url: str
    request_parameters: Mapping[str, str]
    response_media_type: str
    body: bytes
    response_headers: Mapping[str, str] = field(default_factory=dict)
    credential_reference: str | None = None
    response_status_code: int = 200
    response_content_encoding: str = "identity"
    request_method: str = "GET"


@dataclass(frozen=True)
class _ProviderSpec:
    roles: frozenset[str]
    source_system: str
    provider_id: str
    source_locator: str
    document_ids: tuple[str, ...]
    output_frequency_seconds: int
    transform: Callable[[Mapping[str, object], Mapping[str, object]], pd.DataFrame]


def build_raw_envelope(
    *,
    acquisition_id: str,
    source_role: str,
    source_system: str,
    source_locator: str,
    provider_id: str,
    received_at_utc: str,
    documents: Sequence[CapturedProviderDocument],
) -> bytes:
    """Build canonical JSON that contains the exact provider response bytes."""

    document_budget = provider_document_max_bytes_for_role(source_role)
    body_sizes = [len(bytes(document.body)) for document in documents]
    if any(size < 1 or size > document_budget for size in body_sizes):
        raise GovernedLTAcquisitionError(
            "LT raw envelope provider document budget is exceeded"
        )
    total_body_bytes = sum(body_sizes)
    if total_body_bytes > MAX_PROVIDER_DOCUMENT_TOTAL_BYTES:
        raise GovernedLTAcquisitionError(
            "LT raw envelope cumulative document budget is exceeded"
        )
    encoded_documents: list[dict[str, object]] = []
    for ordinal, document in enumerate(documents, start=1):
        body = bytes(document.body)
        encoded_documents.append(
            {
                "document_id": document.document_id,
                "ordinal": ordinal,
                "request": {
                    "method": document.request_method,
                    "url": document.request_url,
                    "parameters": dict(document.request_parameters),
                    "credential_reference": document.credential_reference,
                },
                "response": {
                    "status_code": document.response_status_code,
                    "media_type": document.response_media_type,
                    "content_encoding": document.response_content_encoding,
                    "headers": dict(document.response_headers),
                },
                "body_base64": base64.b64encode(body).decode("ascii"),
                "body_sha256": hashlib.sha256(body).hexdigest(),
                "body_size_bytes": len(body),
            }
        )
    envelope = {
        "schema_version": RAW_ENVELOPE_SCHEMA,
        "acquisition_id": acquisition_id,
        "source_role": source_role,
        "source_system": source_system,
        "source_locator": source_locator,
        "provider_id": provider_id,
        "received_at_utc": received_at_utc,
        "documents": encoded_documents,
    }
    payload = _canonical_json_bytes(envelope)
    validate_raw_envelope(payload)
    return payload


def validate_raw_envelope(payload: bytes) -> dict[str, object]:
    """Validate a canonical envelope and every embedded byte/hash binding."""

    if not payload or len(payload) > MAX_RAW_ENVELOPE_BYTES:
        raise GovernedLTAcquisitionError("LT raw envelope size is invalid")
    try:
        parsed = _strict_json_loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GovernedLTAcquisitionError("LT raw envelope is not strict UTF-8 JSON") from exc
    if not isinstance(parsed, dict) or set(parsed) != _RAW_ENVELOPE_KEYS:
        raise GovernedLTAcquisitionError("LT raw envelope fields are not exact")
    if _canonical_json_bytes(parsed) != payload:
        raise GovernedLTAcquisitionError("LT raw envelope is not canonical JSON")
    if parsed.get("schema_version") != RAW_ENVELOPE_SCHEMA:
        raise GovernedLTAcquisitionError("LT raw envelope schema is unsupported")
    for field_name in ("acquisition_id", "source_role", "source_system", "provider_id"):
        _portable_id(parsed.get(field_name), label=f"LT raw envelope {field_name}")
    locator = str(parsed.get("source_locator", ""))
    _validate_locator(locator)
    _utc_timestamp(parsed.get("received_at_utc"), label="LT raw envelope received_at_utc")

    documents = parsed.get("documents")
    if not isinstance(documents, list) or not documents:
        raise GovernedLTAcquisitionError("LT raw envelope documents are missing")
    document_ids: set[str] = set()
    document_budget = provider_document_max_bytes_for_role(str(parsed["source_role"]))
    for expected_ordinal, raw_document in enumerate(documents, start=1):
        if not isinstance(raw_document, dict) or set(raw_document) != _DOCUMENT_KEYS:
            raise GovernedLTAcquisitionError("LT raw envelope document fields are not exact")
        document_id = _portable_id(
            raw_document.get("document_id"), label="LT raw envelope document_id"
        )
        if document_id in document_ids:
            raise GovernedLTAcquisitionError("LT raw envelope repeats a document_id")
        document_ids.add(document_id)
        if raw_document.get("ordinal") != expected_ordinal:
            raise GovernedLTAcquisitionError("LT raw envelope document order is not contiguous")
        _validate_request(raw_document.get("request"))
        _validate_response(raw_document.get("response"))
        body = _decode_body(raw_document)
        if len(body) > min(MAX_PROVIDER_DOCUMENT_BYTES, document_budget):
            raise GovernedLTAcquisitionError("LT raw envelope document is too large")
        _validate_response_body_headers(raw_document["response"], body)
    return parsed


def build_provider_transform_config(
    *,
    role: str,
    start_utc: str,
    end_utc: str,
) -> dict[str, object]:
    """Bind one provider parser to its role, time window, and exact runtime."""

    transform_id = provider_transform_id_for_role(role)
    spec = _PROVIDER_REGISTRY[transform_id]
    config = {
        "schema_version": PROVIDER_TRANSFORM_CONFIG_SCHEMA,
        "transform_id": transform_id,
        "role": str(role),
        "source_system": spec.source_system,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "output_frequency_seconds": spec.output_frequency_seconds,
        "runtime_fingerprint": provider_runtime_fingerprint(),
    }
    _validate_provider_config(config)
    return config


def provider_transform_id_for_role(role: str) -> str:
    if str(role) in _ENERGY_CHARTS_ZONE_BY_ROLE:
        return ENERGY_CHARTS_PRICE_TRANSFORM
    if str(role) == "entso":
        return ENTSOE_XML_BUNDLE_TRANSFORM
    if str(role) == "hydro":
        return SFOE_OGD17_CSV_TRANSFORM
    raise GovernedLTAcquisitionError(f"provider-raw role is not allow-listed: {role}")


def provider_document_ids_for_role(role: str) -> tuple[str, ...]:
    transform_id = provider_transform_id_for_role(role)
    return _PROVIDER_REGISTRY[transform_id].document_ids


def provider_document_max_bytes_for_role(role: str) -> int:
    provider_transform_id_for_role(role)
    return min(MAX_PROVIDER_DOCUMENT_BYTES, _MAX_PROVIDER_DOCUMENT_BYTES_BY_ROLE[str(role)])


def validate_provider_document_inventory(
    role: str,
    document_ids: Sequence[str],
) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in document_ids)
    if str(role) == "entso" and normalized != _ENTSOE_DOCUMENT_IDS:
        if len(normalized) < 2 or len(normalized) > 32 or len(normalized) % 2:
            raise GovernedLTAcquisitionError(
                "provider transform document inventory mismatch"
            )
        expected = tuple(
            document_id
            for sequence in range(1, len(normalized) // 2 + 1)
            for document_id in (
                f"actual_load_ch_{sequence:04d}",
                f"actual_generation_ch_{sequence:04d}",
            )
        )
    else:
        expected = provider_document_ids_for_role(role)
    if normalized != expected:
        raise GovernedLTAcquisitionError("provider transform document inventory mismatch")
    return normalized


def approved_provider_parser_path_for_role(role: str) -> Path:
    provider_transform_id_for_role(role)
    return Path(__file__).resolve()


def approved_provider_parser_payload_for_role(role: str) -> bytes:
    """Read the installed parser from its active filesystem or zipimport package."""

    provider_transform_id_for_role(role)
    return _installed_module_payload(
        module_file=__file__,
        package=__package__,
        resource_name=Path(__file__).name,
        label=f"installed provider parser for {role}",
    )


def transform_provider_raw(
    *,
    envelope_payload: bytes,
    parser_config: Mapping[str, object],
) -> pd.DataFrame:
    """Replay exact provider responses through the installed allow-listed parser."""

    _validate_provider_config(parser_config)
    transform_id = str(parser_config["transform_id"])
    spec = _PROVIDER_REGISTRY[transform_id]
    envelope = validate_raw_envelope(envelope_payload)
    role = str(parser_config["role"])
    if role not in spec.roles or str(envelope["source_role"]) != role:
        raise GovernedLTAcquisitionError("provider transform role mismatch")
    if (
        str(parser_config["source_system"]) != spec.source_system
        or str(envelope["source_system"]) != spec.source_system
        or str(envelope["provider_id"]) != spec.provider_id
        or str(envelope["source_locator"]) != spec.source_locator
    ):
        raise GovernedLTAcquisitionError("provider transform source identity mismatch")
    _validate_received_at_policy(
        role=role,
        envelope=envelope,
        config=parser_config,
    )
    document_ids = tuple(str(item["document_id"]) for item in envelope["documents"])
    validate_provider_document_inventory(role, document_ids)
    try:
        frame = spec.transform(envelope, parser_config)
    except GovernedLTAcquisitionError:
        raise
    except Exception as exc:
        raise GovernedLTAcquisitionError("provider transform failed") from exc
    _validate_bronze_frame(role, frame)
    return frame


def _validate_received_at_policy(
    *,
    role: str,
    envelope: Mapping[str, object],
    config: Mapping[str, object],
) -> None:
    received_at = _utc_timestamp(
        envelope.get("received_at_utc"),
        label="LT raw envelope received_at_utc",
    )
    if role == "entso":
        _, end = _config_window(config)
        if end > received_at:
            raise GovernedLTAcquisitionError(
                "ENTSO-E actual window extends beyond received_at_utc"
            )
        return
    if role in _ENERGY_CHARTS_ZONE_BY_ROLE:
        _, end = _config_window(config)
        if end > received_at + MAX_ENERGY_CHARTS_DAY_AHEAD_LEAD:
            raise GovernedLTAcquisitionError(
                "Energy Charts day-ahead window exceeds the received_at_utc lead ceiling"
            )
        return
    if role == "hydro":
        # Hydro observation dates are checked against receipt after CSV parsing.
        return
    raise GovernedLTAcquisitionError("provider temporal availability policy is missing")


def verify_provider_raw_replay(
    *,
    role: str,
    source_system: str,
    envelope_payload: bytes,
    bronze_payload: bytes,
    parser_payload: bytes,
    parser_code_sha256: str,
    parser_config: Mapping[str, object],
) -> dict[str, object]:
    """Prove that exact provider bytes reproduce the archived bronze frame."""

    installed_payload = approved_provider_parser_payload_for_role(role)
    installed_hash = hashlib.sha256(installed_payload).hexdigest()
    if (
        installed_hash != hashlib.sha256(parser_payload).hexdigest()
        or installed_hash != str(parser_code_sha256)
    ):
        raise GovernedLTAcquisitionError(
            f"provider parser is not the installed allow-listed version: {role}"
        )
    if str(parser_config.get("source_system", "")) != str(source_system):
        raise GovernedLTAcquisitionError("provider replay source system mismatch")
    replayed = transform_provider_raw(
        envelope_payload=envelope_payload,
        parser_config=parser_config,
    )
    try:
        validate_parquet_allocation_budget(
            bronze_payload,
            label=f"provider bronze artifact {role}",
            max_rows=_PROVIDER_BRONZE_MAX_ROWS[role],
            max_columns=16,
            max_cells=4_800_000,
            max_row_groups=2_048,
            allowed_physical_types=_PROVIDER_BRONZE_PHYSICAL_TYPES,
        )
    except ValueError as exc:
        raise GovernedLTAcquisitionError(str(exc)) from exc
    try:
        bronze = pd.read_parquet(BytesIO(bronze_payload))
    except (OSError, ValueError) as exc:
        raise GovernedLTAcquisitionError("provider bronze artifact is not parquet") from exc
    _validate_bronze_frame(role, bronze)
    try:
        pd.testing.assert_frame_equal(
            replayed,
            bronze,
            check_dtype=True,
            check_exact=True,
            check_like=False,
            check_freq=False,
        )
    except AssertionError as exc:
        raise GovernedLTAcquisitionError(
            f"provider bytes do not reproduce the bronze frame: {role}"
        ) from exc
    if replayed.attrs != bronze.attrs:
        raise GovernedLTAcquisitionError(
            f"provider bytes do not reproduce bronze metadata: {role}"
        )
    resolution_provenance = (
        energy_price_resolution_provenance(bronze, role=role)
        if role in _ENERGY_CHARTS_ZONE_BY_ROLE
        else None
    )
    return {
        "schema_version": PROVIDER_TRANSFORM_CONFIG_SCHEMA,
        "transform_id": str(parser_config["transform_id"]),
        "role": role,
        "envelope_sha256": hashlib.sha256(envelope_payload).hexdigest(),
        "bronze_frame_sha256": dataframe_semantic_sha256(bronze),
        "resolution_provenance": resolution_provenance,
        "status": "VERIFIED_EXACT_PROVIDER_REPLAY",
    }


def dataframe_semantic_sha256(frame: pd.DataFrame) -> str:
    """Hash frame semantics independently from Parquet serialization details."""

    metadata = {
        "columns": [str(value) for value in frame.columns],
        "dtypes": [str(value) for value in frame.dtypes],
        "index_names": [str(value) for value in frame.index.names],
        "index_dtype": str(frame.index.dtype),
        "rows": len(frame),
    }
    if OBSERVATION_RESOLUTION_PROVENANCE_ATTR in frame.attrs:
        metadata["resolution_provenance"] = frame.attrs[
            OBSERVATION_RESOLUTION_PROVENANCE_ATTR
        ]
    digest = hashlib.sha256(_canonical_json_bytes(metadata))
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def provider_runtime_fingerprint() -> dict[str, object]:
    """Return the runtime identity that affects byte-to-frame semantics."""

    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": str(sys.implementation.cache_tag),
        },
        "packages": {
            name: _package_fingerprint(name)
            for name in ("numpy", "pandas", "pyarrow", "tzdata")
        },
    }


def _transform_energy_charts_price(
    envelope: Mapping[str, object], config: Mapping[str, object]
) -> pd.DataFrame:
    document = envelope["documents"][0]
    request = document["request"]
    role = str(config["role"])
    start, end = _config_window(config)
    _require_media_type(document, {"application/json"})
    body = _decode_body(document)
    try:
        data = _strict_json_loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GovernedLTAcquisitionError("Energy Charts body is not strict UTF-8 JSON") from exc
    if not isinstance(data, dict) or set(data) != {
        "deprecated",
        "license_info",
        "price",
        "unit",
        "unix_seconds",
    }:
        raise GovernedLTAcquisitionError("Energy Charts price body must be a mapping")
    if data.get("deprecated") is not False:
        raise GovernedLTAcquisitionError("Energy Charts price endpoint is deprecated")
    if data.get("unit") != "EUR / MWh":
        raise GovernedLTAcquisitionError("Energy Charts price unit is not EUR/MWh")
    if not isinstance(data.get("license_info"), str) or not str(
        data["license_info"]
    ).strip():
        raise GovernedLTAcquisitionError("Energy Charts price license is missing")
    timestamps = data.get("unix_seconds")
    prices = data.get("price")
    if not isinstance(timestamps, list) or not isinstance(prices, list) or not timestamps:
        raise GovernedLTAcquisitionError("Energy Charts price arrays are missing")
    if len(timestamps) != len(prices):
        raise GovernedLTAcquisitionError("Energy Charts price arrays have different lengths")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in timestamps):
        raise GovernedLTAcquisitionError("Energy Charts timestamps are not integer epochs")
    numeric_prices = [_finite_float(value, label="Energy Charts price") for value in prices]
    index = pd.to_datetime(timestamps, unit="s", utc=True)
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise GovernedLTAcquisitionError("Energy Charts timestamps are not sorted and unique")
    if len(index) < 2:
        raise GovernedLTAcquisitionError("Energy Charts price series is too short")
    deltas = np.diff(index.asi8) // 1_000_000_000
    unique_deltas = set(int(value) for value in deltas)
    if len(unique_deltas) != 1 or unique_deltas.pop() not in {900, 3600}:
        raise GovernedLTAcquisitionError("Energy Charts price cadence is unsupported or incomplete")
    raw_cadence = int(deltas[0])
    expected_parameters = {
        "bzn": _ENERGY_CHARTS_ZONE_BY_ROLE[role],
        "start": start.strftime("%Y-%m-%dT%H:%MZ"),
        "end": (end - pd.Timedelta(seconds=raw_cadence)).strftime(
            "%Y-%m-%dT%H:%MZ"
        ),
    }
    _require_request(
        request,
        url=ENERGY_CHARTS_PRICE_URL,
        parameters=expected_parameters,
        credential_reference=None,
    )
    expected_raw = pd.date_range(
        start,
        end - pd.Timedelta(seconds=raw_cadence),
        freq=pd.Timedelta(seconds=raw_cadence),
    )
    if not index.equals(expected_raw):
        raise GovernedLTAcquisitionError("Energy Charts price window is incomplete")
    frame = pd.DataFrame({"price_eur_mwh": numeric_prices}, index=index)
    target = pd.date_range(start, end - pd.Timedelta(minutes=15), freq="15min")
    result = frame.reindex(target).ffill()
    result.attrs[OBSERVATION_RESOLUTION_PROVENANCE_ATTR] = (
        _energy_price_resolution_provenance(
            role=role,
            raw_cadence_seconds=raw_cadence,
            source_observation_count=len(frame),
            output_observation_count=len(result),
        )
    )
    return result


def _transform_entsoe_xml_bundle(
    envelope: Mapping[str, object], config: Mapping[str, object]
) -> pd.DataFrame:
    start, end = _config_window(config)
    chunks: list[pd.DataFrame] = []
    cursor = start
    documents = list(envelope["documents"])
    for load_document, generation_document in zip(
        documents[::2],
        documents[1::2],
        strict=True,
    ):
        load_request = load_document["request"]
        load_parameters = load_request.get("parameters")
        if not isinstance(load_parameters, Mapping):
            raise GovernedLTAcquisitionError("ENTSO-E chunk parameters are invalid")
        chunk_start = _entsoe_request_timestamp(
            load_parameters.get("periodStart"),
            label="ENTSO-E chunk periodStart",
        )
        chunk_end = _entsoe_request_timestamp(
            load_parameters.get("periodEnd"),
            label="ENTSO-E chunk periodEnd",
        )
        if (
            chunk_start != cursor
            or chunk_end <= chunk_start
            or chunk_end - chunk_start > pd.Timedelta(days=366)
            or chunk_end > end
        ):
            raise GovernedLTAcquisitionError(
                "ENTSO-E chunk coverage has a gap, overlap, or oversized window"
            )
        period_start = chunk_start.strftime("%Y%m%d%H%M")
        period_end = chunk_end.strftime("%Y%m%d%H%M")
        _require_request(
            load_request,
            url=ENTSOE_API_URL,
            parameters={
                "documentType": "A65",
                "processType": "A16",
                "outBiddingZone_Domain": _ENTSOE_CH_DOMAIN,
                "periodStart": period_start,
                "periodEnd": period_end,
            },
            credential_reference="env:ENTSOE_API_KEY",
        )
        _require_request(
            generation_document["request"],
            url=ENTSOE_API_URL,
            parameters={
                "documentType": "A75",
                "processType": "A16",
                "in_Domain": _ENTSOE_CH_DOMAIN,
                "periodStart": period_start,
                "periodEnd": period_end,
            },
            credential_reference="env:ENTSOE_API_KEY",
        )
        components = {
            "load_mw": _parse_entsoe_document(
                load_document,
                component="load_mw",
                start=chunk_start,
                end=chunk_end,
            ),
            **_parse_entsoe_generation(
                generation_document,
                start=chunk_start,
                end=chunk_end,
            ),
        }
        chunk_target = pd.date_range(
            chunk_start,
            chunk_end - pd.Timedelta(minutes=15),
            freq="15min",
        )
        chunk = pd.DataFrame(index=chunk_target)
        for column in ("load_mw", "solar_mw", "wind_mw"):
            series = components.get(column)
            if series is None:
                raise GovernedLTAcquisitionError(f"ENTSO-E bundle is missing {column}")
            aligned = series.reindex(chunk_target)
            if aligned.isna().any():
                raise GovernedLTAcquisitionError(
                    f"ENTSO-E bundle has incomplete {column}"
                )
            chunk[column] = aligned.astype(float)
        chunks.append(chunk)
        cursor = chunk_end
    if cursor != end:
        raise GovernedLTAcquisitionError("ENTSO-E chunk coverage is incomplete")
    frame = pd.concat(chunks)
    target = pd.date_range(start, end - pd.Timedelta(minutes=15), freq="15min")
    if not frame.index.equals(target):
        raise GovernedLTAcquisitionError("ENTSO-E chunk coverage is incomplete")
    return frame


def _entsoe_request_timestamp(value: object, *, label: str) -> pd.Timestamp:
    raw = str(value or "")
    if re.fullmatch(r"[0-9]{12}", raw) is None:
        raise GovernedLTAcquisitionError(f"{label} is invalid")
    try:
        return pd.to_datetime(raw, format="%Y%m%d%H%M", utc=True)
    except (TypeError, ValueError) as exc:
        raise GovernedLTAcquisitionError(f"{label} is invalid") from exc


def _transform_sfoe_csv(
    envelope: Mapping[str, object], config: Mapping[str, object]
) -> pd.DataFrame:
    document = envelope["documents"][0]
    _require_request(
        document["request"],
        url=SFOE_OGD17_URL,
        parameters={},
        credential_reference=None,
    )
    _require_media_type(document, {"text/csv", "application/csv"})
    body = _decode_body(document)
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise GovernedLTAcquisitionError("SFOE OGD17 body is not UTF-8 CSV") from exc
    reader = csv.DictReader(StringIO(text, newline=""))
    required = (
        "Datum",
        "Wallis_speicherinhalt_gwh",
        "Graubuenden_speicherinhalt_gwh",
        "Tessin_speicherinhalt_gwh",
        "UebrigCH_speicherinhalt_gwh",
        "TotalCH_speicherinhalt_gwh",
        "Wallis_max_speicherinhalt_gwh",
        "Graubuenden_max_speicherinhalt_gwh",
        "Tessin_max_speicherinhalt_gwh",
        "UebrigCH_max_speicherinhalt_gwh",
        "TotalCH_max_speicherinhalt_gwh",
    )
    if reader.fieldnames is None or tuple(reader.fieldnames) != required:
        raise GovernedLTAcquisitionError("SFOE OGD17 CSV schema is not exact")
    records: list[dict[str, float]] = []
    timestamps: list[pd.Timestamp] = []
    for row_number, row in enumerate(reader, start=1):
        if row_number > 2_000:
            raise GovernedLTAcquisitionError("SFOE OGD17 CSV row budget is exceeded")
        if None in row:
            raise GovernedLTAcquisitionError("SFOE OGD17 CSV rows are malformed")
        try:
            timestamp = pd.Timestamp(str(row["Datum"])).tz_localize(
                "Europe/Zurich",
                ambiguous="raise",
                nonexistent="raise",
            ).tz_convert("UTC")
        except (TypeError, ValueError) as exc:
            raise GovernedLTAcquisitionError("SFOE OGD17 date is invalid") from exc
        fill = _finite_float(row["TotalCH_speicherinhalt_gwh"], label="SFOE fill")
        capacity = _finite_float(
            row["TotalCH_max_speicherinhalt_gwh"], label="SFOE capacity"
        )
        if capacity <= 0 or fill < 0 or fill > capacity:
            raise GovernedLTAcquisitionError("SFOE OGD17 storage values are out of bounds")
        regional_fill = [
            _finite_float(row[column], label=f"SFOE {column}")
            for column in (
                "Wallis_speicherinhalt_gwh",
                "Graubuenden_speicherinhalt_gwh",
                "Tessin_speicherinhalt_gwh",
                "UebrigCH_speicherinhalt_gwh",
            )
        ]
        regional_capacity = [
            _finite_float(row[column], label=f"SFOE {column}")
            for column in (
                "Wallis_max_speicherinhalt_gwh",
                "Graubuenden_max_speicherinhalt_gwh",
                "Tessin_max_speicherinhalt_gwh",
                "UebrigCH_max_speicherinhalt_gwh",
            )
        ]
        if (
            any(value < 0 for value in (*regional_fill, *regional_capacity))
            or not math.isclose(sum(regional_fill), fill, abs_tol=0.01)
            or not math.isclose(sum(regional_capacity), capacity, abs_tol=0.01)
        ):
            raise GovernedLTAcquisitionError(
                "SFOE OGD17 regional totals do not reconcile"
            )
        record = {
            "fill_pct": round(fill / capacity * 100.0, 2),
            "fill_gwh": fill,
            "max_capacity_gwh": capacity,
            "uebrig_ch_gwh": regional_fill[3],
        }
        for source, destination in (
            ("Wallis_speicherinhalt_gwh", "wallis_gwh"),
            ("Graubuenden_speicherinhalt_gwh", "graubuenden_gwh"),
            ("Tessin_speicherinhalt_gwh", "tessin_gwh"),
        ):
            if source in row and str(row[source]).strip():
                regional_value = _finite_float(row[source], label=f"SFOE {source}")
                if regional_value < 0:
                    raise GovernedLTAcquisitionError(
                        f"SFOE {source} is negative"
                    )
                record[destination] = regional_value
        timestamps.append(timestamp)
        records.append(record)
    if not records:
        raise GovernedLTAcquisitionError("SFOE OGD17 CSV rows are missing")
    index = pd.DatetimeIndex(timestamps)
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise GovernedLTAcquisitionError("SFOE OGD17 dates are not sorted and unique")
    received_at = _utc_timestamp(
        envelope.get("received_at_utc"),
        label="LT raw envelope received_at_utc",
    )
    if index.max() > received_at:
        raise GovernedLTAcquisitionError(
            "SFOE OGD17 observation date extends beyond received_at_utc"
        )
    start, end = _config_window(config)
    frame = pd.DataFrame(records, index=index)
    frame = frame.loc[(frame.index >= start) & (frame.index < end)]
    if frame.empty:
        raise GovernedLTAcquisitionError("SFOE OGD17 window is empty")
    if len(frame) < 2:
        raise GovernedLTAcquisitionError("SFOE OGD17 window has insufficient observations")
    local_index = frame.index.tz_convert("Europe/Zurich")
    if any(
        timestamp.weekday() != 0
        or timestamp.hour != 0
        or timestamp.minute != 0
        or timestamp.second != 0
        for timestamp in local_index
    ):
        raise GovernedLTAcquisitionError(
            "SFOE OGD17 weekly phase is not Monday 00:00 Europe/Zurich"
        )
    deltas = local_index.tz_localize(None).to_series().diff().dropna()
    if not (deltas == pd.Timedelta(days=7)).all():
        raise GovernedLTAcquisitionError("SFOE OGD17 weekly cadence is incomplete")
    if (
        frame.index.min() - start >= pd.Timedelta(days=7)
        or end - frame.index.max() > pd.Timedelta(days=7)
    ):
        raise GovernedLTAcquisitionError("SFOE OGD17 window coverage is incomplete")
    return frame


def _parse_entsoe_document(
    document: Mapping[str, object], *, component: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    _require_media_type(document, {"application/xml", "text/xml"})
    root = _safe_xml_root(_decode_body(document))
    _validate_entsoe_document_identity(root, component="load_mw")
    time_series_rows = _children(root, "TimeSeries")
    if len(time_series_rows) != 1:
        raise GovernedLTAcquisitionError(
            f"ENTSO-E XML must contain exactly one {component} TimeSeries"
        )
    values: dict[pd.Timestamp, float] = {}
    for time_series in time_series_rows:
        _validate_entsoe_time_series_identity(time_series, component="load_mw")
        if not _first_text(time_series, "mRID"):
            raise GovernedLTAcquisitionError("ENTSO-E TimeSeries mRID is missing")
        series = _entsoe_period_values(time_series, start=start, end=end)
        for timestamp, value in series.items():
            values[timestamp] = values.get(timestamp, 0.0) + value
    if not values:
        raise GovernedLTAcquisitionError(f"ENTSO-E XML contains no {component} values")
    return pd.Series(values, name=component, dtype=float).sort_index()


def _parse_entsoe_generation(
    document: Mapping[str, object], *, start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, pd.Series]:
    _require_media_type(document, {"application/xml", "text/xml"})
    root = _safe_xml_root(_decode_body(document))
    _validate_entsoe_document_identity(root, component="generation")
    aggregated: dict[str, dict[pd.Timestamp, float]] = {"solar_mw": {}, "wind_mw": {}}
    series_ids: set[str] = set()
    psr_types: set[str] = set()
    for time_series in _children(root, "TimeSeries"):
        _validate_entsoe_time_series_identity(time_series, component="generation")
        series_id = _first_text(time_series, "mRID")
        if not series_id or series_id in series_ids:
            raise GovernedLTAcquisitionError("ENTSO-E repeats a TimeSeries mRID")
        series_ids.add(series_id)
        psr_type = _required_path_text(time_series, ("MktPSRType", "psrType"))
        if psr_type == "B16":
            component = "solar_mw"
        elif psr_type in {"B18", "B19"}:
            component = "wind_mw"
        else:
            continue
        if psr_type in psr_types:
            raise GovernedLTAcquisitionError("ENTSO-E repeats a generation PSR type")
        psr_types.add(psr_type)
        for timestamp, value in _entsoe_period_values(
            time_series, start=start, end=end
        ).items():
            aggregated[component][timestamp] = (
                aggregated[component].get(timestamp, 0.0) + value
            )
    return {
        component: pd.Series(values, name=component, dtype=float).sort_index()
        for component, values in aggregated.items()
        if values
    }


def _entsoe_period_values(
    time_series: ET.Element, *, start: pd.Timestamp, end: pd.Timestamp
) -> dict[pd.Timestamp, float]:
    values: dict[pd.Timestamp, float] = {}
    for period in _children(time_series, "Period"):
        period_start_raw = _required_path_text(period, ("timeInterval", "start"))
        period_end_raw = _required_path_text(period, ("timeInterval", "end"))
        resolution_raw = _first_text(period, "resolution")
        try:
            period_start = pd.Timestamp(period_start_raw).tz_convert("UTC")
            period_end = pd.Timestamp(period_end_raw).tz_convert("UTC")
        except (TypeError, ValueError) as exc:
            raise GovernedLTAcquisitionError("ENTSO-E period interval is invalid") from exc
        resolution = {"PT15M": pd.Timedelta(minutes=15), "PT60M": pd.Timedelta(hours=1)}.get(
            resolution_raw
        )
        if resolution is None:
            raise GovernedLTAcquisitionError("ENTSO-E resolution is unsupported")
        if (
            period_end <= period_start
            or period_start < start
            or period_end > end
        ):
            raise GovernedLTAcquisitionError(
                "ENTSO-E period interval is outside the requested window"
            )
        positions: set[int] = set()
        for point in _children(period, "Point"):
            try:
                position = int(_first_text(point, "position"))
            except (TypeError, ValueError) as exc:
                raise GovernedLTAcquisitionError("ENTSO-E point position is invalid") from exc
            if position < 1 or position in positions:
                raise GovernedLTAcquisitionError("ENTSO-E point positions are invalid")
            positions.add(position)
            quantity = _finite_float(_first_text(point, "quantity"), label="ENTSO-E quantity")
            if quantity < 0:
                raise GovernedLTAcquisitionError("ENTSO-E quantity is negative")
            point_start = period_start + (position - 1) * resolution
            for timestamp in pd.date_range(
                point_start,
                point_start + resolution - pd.Timedelta(minutes=15),
                freq="15min",
            ):
                if start <= timestamp < end:
                    if timestamp in values:
                        raise GovernedLTAcquisitionError("ENTSO-E periods overlap")
                    values[timestamp] = quantity
        if not positions or positions != set(range(1, len(positions) + 1)):
            raise GovernedLTAcquisitionError("ENTSO-E point sequence is incomplete")
        if period_start + len(positions) * resolution != period_end:
            raise GovernedLTAcquisitionError(
                "ENTSO-E point sequence does not match the period interval"
            )
    return values


def _safe_xml_root(body: bytes) -> ET.Element:
    upper = body.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise GovernedLTAcquisitionError("ENTSO-E XML declarations are unsafe")
    try:
        return ET.fromstring(body)
    except ET.ParseError as exc:
        raise GovernedLTAcquisitionError("ENTSO-E body is not well-formed XML") from exc


def _validate_entsoe_document_identity(root: ET.Element, *, component: str) -> None:
    identity = _ENTSOE_DOCUMENT_IDENTITY[component]
    expected_root = f"{{{identity['namespace']}}}GL_MarketDocument"
    if root.tag != expected_root:
        raise GovernedLTAcquisitionError("ENTSO-E document namespace/root mismatch")
    for item in root.iter():
        if not isinstance(item.tag, str) or not item.tag.startswith(
            f"{{{identity['namespace']}}}"
        ):
            raise GovernedLTAcquisitionError(
                "ENTSO-E document contains a foreign-namespace element"
            )
    if _first_text(root, "type") != identity["document_type"]:
        raise GovernedLTAcquisitionError("ENTSO-E document type mismatch")
    if _first_text(root, "process.processType") != "A16":
        raise GovernedLTAcquisitionError("ENTSO-E process type mismatch")
    if _first_text(root, str(identity["domain_field"])) != _ENTSOE_CH_DOMAIN:
        raise GovernedLTAcquisitionError("ENTSO-E Swiss domain mismatch")


def _validate_entsoe_time_series_identity(
    time_series: ET.Element,
    *,
    component: str,
) -> None:
    expected_business_type, expected_aggregation = {
        "load_mw": ("A04", "A01"),
        "generation": ("A01", "A08"),
    }[component]
    if _first_text(time_series, "businessType") != expected_business_type:
        raise GovernedLTAcquisitionError("ENTSO-E business type mismatch")
    if _first_text(time_series, "objectAggregation") != expected_aggregation:
        raise GovernedLTAcquisitionError("ENTSO-E object aggregation mismatch")
    if _first_text(time_series, "curveType") != "A01":
        raise GovernedLTAcquisitionError("ENTSO-E curve type mismatch")
    if _first_text(time_series, "quantity_Measure_Unit.name") != "MAW":
        raise GovernedLTAcquisitionError("ENTSO-E quantity unit is not MW")


def _children(element: ET.Element, local_name: str) -> list[ET.Element]:
    namespace = _element_namespace(element)
    expected_tag = f"{{{namespace}}}{local_name}"
    return [item for item in element if item.tag == expected_tag]


def _first_text(element: ET.Element, local_name: str) -> str:
    matches = _children(element, local_name)
    if len(matches) != 1 or matches[0].text is None or not matches[0].text.strip():
        raise GovernedLTAcquisitionError(
            f"ENTSO-E XML requires exactly one {local_name}"
        )
    return matches[0].text.strip()


def _required_path_text(element: ET.Element, path: tuple[str, ...]) -> str:
    selected = element
    for local_name in path[:-1]:
        matches = _children(selected, local_name)
        if len(matches) != 1:
            raise GovernedLTAcquisitionError(
                f"ENTSO-E XML requires exactly one {'/'.join(path)}"
            )
        selected = matches[0]
    return _first_text(selected, path[-1])


def _element_namespace(element: ET.Element) -> str:
    if not isinstance(element.tag, str) or not element.tag.startswith("{"):
        raise GovernedLTAcquisitionError("ENTSO-E XML element namespace is missing")
    namespace, separator, _ = element.tag[1:].partition("}")
    if not separator or not namespace:
        raise GovernedLTAcquisitionError("ENTSO-E XML element namespace is invalid")
    return namespace


def _validate_provider_config(config: Mapping[str, object]) -> None:
    if set(config) != _CONFIG_KEYS:
        raise GovernedLTAcquisitionError("provider transform config fields are not exact")
    if config.get("schema_version") != PROVIDER_TRANSFORM_CONFIG_SCHEMA:
        raise GovernedLTAcquisitionError("provider transform config schema is unsupported")
    transform_id = str(config.get("transform_id", ""))
    spec = _PROVIDER_REGISTRY.get(transform_id)
    if spec is None:
        raise GovernedLTAcquisitionError("provider transform is not allow-listed")
    role = str(config.get("role", ""))
    if role not in spec.roles or config.get("source_system") != spec.source_system:
        raise GovernedLTAcquisitionError("provider transform config identity mismatch")
    if config.get("output_frequency_seconds") != spec.output_frequency_seconds:
        raise GovernedLTAcquisitionError("provider transform cadence mismatch")
    start, end = _config_window(config)
    if end <= start:
        raise GovernedLTAcquisitionError("provider transform window is empty")
    if transform_id != SFOE_OGD17_CSV_TRANSFORM:
        cadence = pd.Timedelta(minutes=15)
        if (
            start.value % cadence.value
            or end.value % cadence.value
            or (end - start).value % cadence.value
        ):
            raise GovernedLTAcquisitionError(
                "provider transform window is not aligned to the 15-minute UTC grid"
            )
    if end - start > _MAX_PROVIDER_WINDOW[transform_id]:
        raise GovernedLTAcquisitionError("provider transform window is too large")
    if config.get("runtime_fingerprint") != provider_runtime_fingerprint():
        raise GovernedLTAcquisitionError("provider transform runtime fingerprint mismatch")


def _validate_request(value: object) -> None:
    if not isinstance(value, dict) or set(value) != _REQUEST_KEYS:
        raise GovernedLTAcquisitionError("LT raw envelope request fields are not exact")
    if value.get("method") != "GET":
        raise GovernedLTAcquisitionError("LT raw envelope request method is unsupported")
    url = str(value.get("url", ""))
    parsed = urlsplit(url)
    if (
        len(url) > 2048
        or
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise GovernedLTAcquisitionError("LT raw envelope request URL is unsafe")
    parameters = value.get("parameters")
    if not isinstance(parameters, dict) or len(parameters) > 64:
        raise GovernedLTAcquisitionError("LT raw envelope request parameters are invalid")
    for key, parameter in parameters.items():
        if (
            not isinstance(key, str)
            or not key
            or len(key) > 128
            or _has_unsafe_http_text(key)
            or _SENSITIVE_PARAMETER.search(key)
        ):
            raise GovernedLTAcquisitionError("LT raw envelope contains a sensitive parameter")
        if (
            not isinstance(parameter, str)
            or len(parameter) > 2048
            or _has_unsafe_http_text(parameter)
        ):
            raise GovernedLTAcquisitionError("LT raw envelope request parameter is not text")
    credential = value.get("credential_reference")
    if credential is not None and (
        not isinstance(credential, str)
        or _ENV_CREDENTIAL_REFERENCE.fullmatch(credential) is None
    ):
        raise GovernedLTAcquisitionError("LT raw envelope credential reference is invalid")


def _validate_response(value: object) -> None:
    if not isinstance(value, dict) or set(value) != _RESPONSE_KEYS:
        raise GovernedLTAcquisitionError("LT raw envelope response fields are not exact")
    if value.get("status_code") != 200:
        raise GovernedLTAcquisitionError("LT raw envelope response status is not successful")
    media_type = value.get("media_type")
    if (
        not isinstance(media_type, str)
        or _MEDIA_TYPE.fullmatch(media_type) is None
    ):
        raise GovernedLTAcquisitionError("LT raw envelope media type is not normalized")
    if value.get("content_encoding") != "identity":
        raise GovernedLTAcquisitionError("LT raw envelope body is not decoded to identity bytes")
    headers = value.get("headers")
    if not isinstance(headers, dict) or len(headers) > len(_ALLOWED_RESPONSE_HEADERS):
        raise GovernedLTAcquisitionError("LT raw envelope response headers are invalid")
    for key, header_value in headers.items():
        if (
            key not in _ALLOWED_RESPONSE_HEADERS
            or not isinstance(header_value, str)
            or len(header_value) > 4096
            or _has_unsafe_http_text(header_value)
        ):
            raise GovernedLTAcquisitionError(
                "LT raw envelope response header is not allow-listed"
            )


def _validate_response_body_headers(response: Mapping[str, object], body: bytes) -> None:
    headers = response.get("headers")
    if not isinstance(headers, Mapping):
        raise GovernedLTAcquisitionError("LT raw envelope response headers are invalid")
    content_length = headers.get("content-length")
    if content_length is not None:
        raw_content_length = str(content_length)
        if re.fullmatch(r"0|[1-9][0-9]*", raw_content_length) is None:
            raise GovernedLTAcquisitionError(
                "LT raw envelope Content-Length is invalid"
            )
        try:
            declared_length = int(raw_content_length)
        except ValueError as exc:
            raise GovernedLTAcquisitionError(
                "LT raw envelope Content-Length is invalid"
            ) from exc
        if declared_length != len(body):
            raise GovernedLTAcquisitionError(
                "LT raw envelope Content-Length does not match body bytes"
            )
    content_type = headers.get("content-type")
    if content_type is not None:
        normalized = str(content_type).split(";", 1)[0].strip().lower()
        if normalized != str(response.get("media_type", "")).lower():
            raise GovernedLTAcquisitionError(
                "LT raw envelope Content-Type does not match media_type"
            )


def _has_unsafe_http_text(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _decode_body(document: Mapping[str, object]) -> bytes:
    try:
        body = base64.b64decode(str(document.get("body_base64", "")), validate=True)
    except (TypeError, ValueError) as exc:
        raise GovernedLTAcquisitionError("LT raw envelope body is not valid base64") from exc
    size = document.get("body_size_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size < 1 or size != len(body):
        raise GovernedLTAcquisitionError("LT raw envelope body size mismatch")
    expected_hash = str(document.get("body_sha256", ""))
    if not _SHA256.fullmatch(expected_hash) or hashlib.sha256(body).hexdigest() != expected_hash:
        raise GovernedLTAcquisitionError("LT raw envelope body hash mismatch")
    return body


def _require_request(
    request: Mapping[str, object],
    *,
    url: str,
    parameters: Mapping[str, str],
    credential_reference: str | None,
) -> None:
    if (
        request.get("method") != "GET"
        or request.get("url") != url
        or request.get("parameters") != dict(parameters)
        or request.get("credential_reference") != credential_reference
    ):
        raise GovernedLTAcquisitionError("provider request does not match the allow-list")


def _require_media_type(document: Mapping[str, object], allowed: set[str]) -> None:
    response = document.get("response")
    if not isinstance(response, Mapping) or response.get("media_type") not in allowed:
        raise GovernedLTAcquisitionError("provider response media type is unsupported")


def _config_window(config: Mapping[str, object]) -> tuple[pd.Timestamp, pd.Timestamp]:
    return (
        _utc_timestamp(config.get("start_utc"), label="provider transform start_utc"),
        _utc_timestamp(config.get("end_utc"), label="provider transform end_utc"),
    )


def _utc_timestamp(value: object, *, label: str) -> pd.Timestamp:
    raw = str(value or "")
    if not raw.endswith("Z"):
        raise GovernedLTAcquisitionError(f"{label} must use canonical UTC Z notation")
    try:
        parsed = pd.Timestamp(raw)
    except (TypeError, ValueError) as exc:
        raise GovernedLTAcquisitionError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != pd.Timedelta(0):
        raise GovernedLTAcquisitionError(f"{label} is not UTC")
    return parsed.tz_convert("UTC")


def _portable_id(value: object, *, label: str) -> str:
    raw = str(value or "")
    if not _PORTABLE_ID.fullmatch(raw) or raw in {".", ".."}:
        raise GovernedLTAcquisitionError(f"{label} is not a portable identifier")
    return raw


def _validate_locator(value: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme not in {"https", "fixture"} or not parsed.netloc:
        raise GovernedLTAcquisitionError("LT raw envelope source_locator is invalid")
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        raise GovernedLTAcquisitionError("LT raw envelope source_locator is unsafe")


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise GovernedLTAcquisitionError(f"{label} is not numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GovernedLTAcquisitionError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise GovernedLTAcquisitionError(f"{label} is not finite")
    return number


def energy_price_resolution_provenance(
    frame: pd.DataFrame,
    *,
    role: str,
) -> dict[str, object]:
    """Return and validate the cadence semantics bound to an EPEX bronze frame."""

    if role not in _ENERGY_CHARTS_ZONE_BY_ROLE:
        raise GovernedLTAcquisitionError(
            "resolution provenance is only defined for Energy Charts price roles"
        )
    if set(frame.attrs) != {OBSERVATION_RESOLUTION_PROVENANCE_ATTR}:
        raise GovernedLTAcquisitionError(
            "provider EPEX bronze resolution provenance is missing or ambiguous"
        )
    provenance = frame.attrs.get(OBSERVATION_RESOLUTION_PROVENANCE_ATTR)
    if not isinstance(provenance, Mapping) or set(provenance) != _RESOLUTION_PROVENANCE_KEYS:
        raise GovernedLTAcquisitionError(
            "provider EPEX bronze resolution provenance fields are not exact"
        )
    source_cadence = provenance.get("source_cadence_seconds")
    source_count = provenance.get("source_observation_count")
    output_count = provenance.get("output_observation_count")
    for value, label in (
        (source_cadence, "source cadence"),
        (source_count, "source observation count"),
        (output_count, "output observation count"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise GovernedLTAcquisitionError(
                f"provider EPEX bronze resolution {label} is invalid"
            )
    if source_cadence not in {900, 3600}:
        raise GovernedLTAcquisitionError(
            "provider EPEX bronze source cadence is unsupported"
        )
    expected = _energy_price_resolution_provenance(
        role=role,
        raw_cadence_seconds=source_cadence,
        source_observation_count=source_count,
        output_observation_count=output_count,
    )
    if dict(provenance) != expected:
        raise GovernedLTAcquisitionError(
            "provider EPEX bronze resolution provenance is inconsistent"
        )
    if output_count != len(frame):
        raise GovernedLTAcquisitionError(
            "provider EPEX bronze resolution output count is inconsistent"
        )
    expected_output_count = source_count * (source_cadence // 900)
    if output_count != expected_output_count:
        raise GovernedLTAcquisitionError(
            "provider EPEX bronze resolution expansion is inconsistent"
        )
    if not isinstance(frame.index, pd.DatetimeIndex) or len(frame.index) < 2:
        raise GovernedLTAcquisitionError(
            "provider EPEX bronze resolution index is invalid"
        )
    deltas = np.diff(frame.index.asi8) // 1_000_000_000
    if set(int(value) for value in deltas) != {900}:
        raise GovernedLTAcquisitionError(
            "provider EPEX bronze output cadence is not 15 minutes"
        )
    return dict(provenance)


def require_native_quarter_hour_price_truth(
    frame: pd.DataFrame,
    *,
    role: str,
) -> dict[str, object]:
    """Fail closed unless cadence and external product identity prove QH truth.

    Energy Charts price bytes alone can establish observation cadence, but do
    not bind the auction, session, or product identity needed to call a
    15-minute series native quarter-hour market truth.
    """

    provenance = energy_price_resolution_provenance(frame, role=role)
    if provenance["native_quarter_hour_truth_eligible"] is not True:
        raise GovernedLTAcquisitionError(
            f"{role} price observations are not native quarter-hour truth; "
            "admission blocker: "
            f"{provenance['quarter_hour_truth_blocker']}"
        )
    return provenance


def _energy_price_resolution_provenance(
    *,
    role: str,
    raw_cadence_seconds: int,
    source_observation_count: int,
    output_observation_count: int,
) -> dict[str, object]:
    native_quarter_hour_cadence = raw_cadence_seconds == 900
    native_hourly_cadence = raw_cadence_seconds == 3600
    return {
        "schema_version": OBSERVATION_RESOLUTION_PROVENANCE_SCHEMA,
        "role": role,
        "source_cadence_seconds": raw_cadence_seconds,
        "output_cadence_seconds": 900,
        "source_observation_count": source_observation_count,
        "output_observation_count": output_observation_count,
        "resampling_method": (
            "NONE"
            if native_quarter_hour_cadence
            else "FORWARD_FILL_WITHIN_SOURCE_INTERVAL"
        ),
        "output_sampling_kind": (
            "DIRECT_15_MINUTE_OBSERVATIONS_PRODUCT_IDENTITY_UNVERIFIED"
            if native_quarter_hour_cadence
            else "UPSAMPLED_STEPWISE_PROXY"
        ),
        "native_quarter_hour_cadence_eligible": native_quarter_hour_cadence,
        "native_hourly_cadence_eligible": native_hourly_cadence,
        "hourly_aggregation_eligible": True,
        "native_quarter_hour_truth_eligible": False,
        "product_identity_status": (
            "EXTERNAL_PRODUCT_AUCTION_SESSION_IDENTITY_UNVERIFIED"
        ),
        "quarter_hour_truth_blocker": (
            "EXTERNAL_PRODUCT_AUCTION_SESSION_IDENTITY_REQUIRED"
            if native_quarter_hour_cadence
            else "SOURCE_CADENCE_IS_HOURLY"
        ),
        "scientific_use_class": (
            "DIRECT_15_MINUTE_OBSERVATIONS_PRODUCT_IDENTITY_UNVERIFIED"
            if native_quarter_hour_cadence
            else "NATIVE_HOURLY_PRICE_QH_TRANSPORT_PROXY_ONLY"
        ),
    }


def _validate_bronze_frame(role: str, frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise GovernedLTAcquisitionError("provider transform produced an empty frame")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise GovernedLTAcquisitionError("provider transform index is not timezone-aware")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise GovernedLTAcquisitionError("provider transform index is not sorted and unique")
    if role.startswith("epex_") and list(frame.columns) != ["price_eur_mwh"]:
        raise GovernedLTAcquisitionError("provider EPEX bronze schema is invalid")
    if role.startswith("epex_"):
        energy_price_resolution_provenance(frame, role=role)
    if role == "entso" and not {"load_mw", "solar_mw", "wind_mw"}.issubset(frame.columns):
        raise GovernedLTAcquisitionError("provider ENTSO-E bronze schema is invalid")
    if role == "hydro" and not {
        "fill_pct",
        "fill_gwh",
        "max_capacity_gwh",
    }.issubset(frame.columns):
        raise GovernedLTAcquisitionError("provider hydro bronze schema is invalid")
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise GovernedLTAcquisitionError("provider bronze values are not finite")


def _package_fingerprint(name: str) -> dict[str, object]:
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return {"version": "NOT_INSTALLED", "module_sha256": None}
    try:
        module = importlib.import_module(name)
        module_path = Path(str(module.__file__)).resolve()
        module_hash = hashlib.sha256(
            read_stable_single_link_file(
                module_path, label=f"provider runtime package {name}"
            )
        ).hexdigest()
    except (AttributeError, ImportError, OSError, TypeError, ValueError):
        module_hash = None
    return {"version": version, "module_sha256": module_hash}


def _installed_module_payload(
    *,
    module_file: str,
    package: str,
    resource_name: str,
    label: str,
) -> bytes:
    path = Path(module_file)
    try:
        if path.is_file():
            payload = read_stable_single_link_file(path, label=label)
        else:
            payload = importlib.resources.files(package).joinpath(resource_name).read_bytes()
    except (AttributeError, ImportError, OSError, TypeError, ValueError) as exc:
        raise GovernedLTAcquisitionError(f"{label} cannot be read") from exc
    if not payload or len(payload) > 2 * 1024 * 1024:
        raise GovernedLTAcquisitionError(f"{label} size is invalid")
    return payload


def _strict_json_loads(value: str) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise json.JSONDecodeError("duplicate key", value, 0)
            result[key] = item
        return result

    return json.loads(
        value,
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda constant: (_ for _ in ()).throw(
            json.JSONDecodeError(f"invalid constant {constant}", value, 0)
        ),
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


_PROVIDER_REGISTRY = {
    ENERGY_CHARTS_PRICE_TRANSFORM: _ProviderSpec(
        roles=frozenset(_ENERGY_CHARTS_ZONE_BY_ROLE),
        source_system=ENERGY_CHARTS_SOURCE_SYSTEM,
        provider_id=ENERGY_CHARTS_SOURCE_SYSTEM,
        source_locator=ENERGY_CHARTS_PRICE_URL,
        document_ids=("day_ahead_price",),
        output_frequency_seconds=900,
        transform=_transform_energy_charts_price,
    ),
    ENTSOE_XML_BUNDLE_TRANSFORM: _ProviderSpec(
        roles=frozenset({"entso"}),
        source_system=ENTSOE_SOURCE_SYSTEM,
        provider_id=ENTSOE_SOURCE_SYSTEM,
        source_locator=ENTSOE_API_URL,
        document_ids=_ENTSOE_DOCUMENT_IDS,
        output_frequency_seconds=900,
        transform=_transform_entsoe_xml_bundle,
    ),
    SFOE_OGD17_CSV_TRANSFORM: _ProviderSpec(
        roles=frozenset({"hydro"}),
        source_system=SFOE_SOURCE_SYSTEM,
        provider_id=SFOE_SOURCE_SYSTEM,
        source_locator=SFOE_OGD17_URL,
        document_ids=("ogd17_reservoir_levels",),
        output_frequency_seconds=604800,
        transform=_transform_sfoe_csv,
    ),
}
