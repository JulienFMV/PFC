"""Capture exact public Energy Charts price bytes for local LT replay.

This is a network-capture utility, not a publisher or signing authority. It
writes an immutable provider body and an exact ``lt_provider_capture_spec.v1``
for the separately installed, networkless acquisition Builder. Workstation
timestamps and paths are explicitly non-production evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import requests

from pfc_shaping.data.governed_lt_acquisition import (
    ENERGY_CHARTS_PRICE_URL,
    ENERGY_CHARTS_SOURCE_SYSTEM,
    CapturedProviderDocument,
    build_provider_transform_config,
    build_raw_envelope,
    energy_price_resolution_provenance,
    provider_document_max_bytes_for_role,
    transform_provider_raw,
)
from pfc_shaping.data.ingest_energy_charts import DEFAULT_VERIFY_BUNDLE
from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.path_safety import assert_absolute_path_has_no_links

CAPTURE_SPEC_SCHEMA = "lt_provider_capture_spec.v1"
CAPTURE_SUMMARY_SCHEMA = "local_public_provider_capture.v2"
ENERGY_CHARTS_ZONE_BY_ROLE = {
    "epex_ch": "CH",
    "epex_de": "DE-LU",
    "epex_at": "AT",
    "epex_fr": "FR",
    "epex_it": "IT-North",
}
_PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ALLOWED_HEADERS = {
    "content-length",
    "content-type",
    "date",
    "etag",
    "last-modified",
    "x-request-id",
}


def capture_energy_charts_price(
    *,
    role: str,
    start_utc: str,
    end_utc: str,
    raw_cadence_minutes: int,
    acquisition_id: str,
    output_directory: Path,
    session: requests.Session | None = None,
    ca_bundle: str | Path | None = None,
) -> Path:
    """Capture one exact response and return the last-written summary path."""

    if role not in ENERGY_CHARTS_ZONE_BY_ROLE:
        raise ValueError("Energy Charts role is unsupported")
    if _PORTABLE_ID.fullmatch(acquisition_id) is None:
        raise ValueError("acquisition_id must be a portable identifier")
    if raw_cadence_minutes not in {15, 60}:
        raise ValueError("raw cadence must be 15 or 60 minutes")
    start = _utc_timestamp(start_utc, label="start_utc")
    end = _utc_timestamp(end_utc, label="end_utc")
    cadence = pd.Timedelta(minutes=raw_cadence_minutes)
    if end <= start or end - start > pd.Timedelta(days=32):
        raise ValueError("capture window must be positive and at most 32 days")
    if start != start.floor(cadence) or end != end.floor(cadence):
        raise ValueError("capture window is not aligned to the raw cadence")

    output = _absolute_no_links(output_directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    output = assert_absolute_path_has_no_links(output)
    output.mkdir(exist_ok=False)
    attempt_path = output / "capture-attempt.json"
    body_path = output / "provider-response.json"
    spec_path = output / "capture-spec.json"
    summary_path = output / "capture-summary.json"
    attempt = {
        "schema_version": CAPTURE_SUMMARY_SCHEMA,
        "status": "ATTEMPTED_LOCAL_UNTRUSTED_CAPTURE",
        "production_authorization": False,
        "promotion_gate": False,
        "acquisition_id": acquisition_id,
        "source_role": role,
        "start_utc": _iso(start),
        "end_utc": _iso(end),
        "workstation_clock_trusted": False,
    }
    _write_json(attempt_path, attempt, label="local capture attempt")

    request_parameters = {
        "bzn": ENERGY_CHARTS_ZONE_BY_ROLE[role],
        "start": start.strftime("%Y-%m-%dT%H:%MZ"),
        "end": (end - cadence).strftime("%Y-%m-%dT%H:%MZ"),
    }
    verify = _ca_bundle(ca_bundle)
    client = session or requests.Session()
    response = client.get(
        ENERGY_CHARTS_PRICE_URL,
        params=request_parameters,
        headers={"Accept": "application/json", "Accept-Encoding": "identity"},
        timeout=(10, 60),
        allow_redirects=False,
        verify=verify,
        stream=True,
    )
    if response.status_code != 200:
        raise ValueError(f"Energy Charts returned HTTP {response.status_code}")
    final_url = urlsplit(response.url)
    expected_url = urlsplit(ENERGY_CHARTS_PRICE_URL)
    if (
        final_url.scheme != expected_url.scheme
        or final_url.netloc != expected_url.netloc
        or final_url.path != expected_url.path
    ):
        raise ValueError("Energy Charts response URL identity changed")
    encoding = str(response.headers.get("content-encoding", "identity")).strip().lower()
    if encoding not in {"", "identity"}:
        raise ValueError("Energy Charts response ignored identity encoding")
    content_type = str(response.headers.get("content-type", "")).split(";", 1)[0].lower()
    if content_type != "application/json":
        raise ValueError("Energy Charts response media type is not application/json")
    maximum = provider_document_max_bytes_for_role(role)
    declared_length = response.headers.get("content-length")
    if declared_length is not None and int(declared_length) > maximum:
        raise ValueError("Energy Charts response exceeds the role budget")
    chunks: list[bytes] = []
    body_size = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        body_size += len(chunk)
        if body_size > maximum:
            raise ValueError("Energy Charts response exceeds the role budget")
        chunks.append(bytes(chunk))
    body = b"".join(chunks)
    if not body:
        raise ValueError("Energy Charts response body is empty")
    received_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    response_headers = {
        key.lower(): str(value)
        for key, value in response.headers.items()
        if key.lower() in _ALLOWED_HEADERS
    }
    document = CapturedProviderDocument(
        document_id="day_ahead_price",
        request_url=ENERGY_CHARTS_PRICE_URL,
        request_parameters=request_parameters,
        response_media_type=content_type,
        body=body,
        response_headers=response_headers,
        credential_reference=None,
        response_status_code=200,
        response_content_encoding="identity",
    )
    envelope = build_raw_envelope(
        acquisition_id=acquisition_id,
        source_role=role,
        source_system=ENERGY_CHARTS_SOURCE_SYSTEM,
        source_locator=ENERGY_CHARTS_PRICE_URL,
        provider_id=ENERGY_CHARTS_SOURCE_SYSTEM,
        received_at_utc=received_at,
        documents=[document],
    )
    transform_config = build_provider_transform_config(
        role=role,
        start_utc=_iso(start),
        end_utc=_iso(end),
    )
    bronze = transform_provider_raw(
        envelope_payload=envelope,
        parser_config=transform_config,
    )
    resolution_provenance = energy_price_resolution_provenance(bronze, role=role)
    spec = {
        "schema_version": CAPTURE_SPEC_SCHEMA,
        "acquisition_id": acquisition_id,
        "source_role": role,
        "source_system": ENERGY_CHARTS_SOURCE_SYSTEM,
        "source_locator": ENERGY_CHARTS_PRICE_URL,
        "provider_id": ENERGY_CHARTS_SOURCE_SYSTEM,
        "received_at_utc": received_at,
        "start_utc": _iso(start),
        "end_utc": _iso(end),
        "documents": [
            {
                "document_id": "day_ahead_price",
                "body_path": str(body_path),
                "request_url": ENERGY_CHARTS_PRICE_URL,
                "request_parameters": request_parameters,
                "credential_reference": None,
                "response_status_code": 200,
                "response_media_type": content_type,
                "response_content_encoding": "identity",
                "response_headers": response_headers,
            }
        ],
    }
    spec_payload = _canonical_json(spec)
    _write_bytes(body_path, body, label="local provider response")
    _write_bytes(
        spec_path,
        spec_payload,
        label="local provider capture specification",
    )
    summary = {
        "schema_version": CAPTURE_SUMMARY_SCHEMA,
        "status": "COMPLETE_LOCAL_UNTRUSTED_CAPTURE",
        "production_authorization": False,
        "promotion_gate": False,
        "publication_status": "NOT_PUBLISHED",
        "timestamp_authority": "LOCAL_WORKSTATION_CLOCK_UNTRUSTED",
        "next_authority": "NETWORKLESS_INSTALLED_ACQUISITION_BUILDER",
        "acquisition_id": acquisition_id,
        "source_role": role,
        "received_at_utc": received_at,
        "start_utc": _iso(start),
        "end_utc": _iso(end),
        "raw_cadence_minutes": raw_cadence_minutes,
        "resolution_provenance": resolution_provenance,
        "provider_body": _record(body_path, body),
        "capture_spec": _record(spec_path, spec_payload),
        "validated_bronze_rows": int(len(bronze)),
        "validated_bronze_start_utc": _iso(bronze.index.min()),
        "validated_bronze_end_utc": _iso(bronze.index.max() + pd.Timedelta(minutes=15)),
    }
    _write_json(summary_path, summary, label="local provider capture summary")
    return summary_path


def _absolute_no_links(path: str | Path) -> Path:
    selected = Path(path).expanduser()
    if not selected.is_absolute():
        raise ValueError("capture output directory must be absolute")
    return assert_absolute_path_has_no_links(Path(os.path.abspath(selected)))


def _ca_bundle(value: str | Path | None) -> bool | str:
    selected = value or os.environ.get("PFC_REQUESTS_CA_BUNDLE")
    if selected is None and DEFAULT_VERIFY_BUNDLE.exists():
        selected = DEFAULT_VERIFY_BUNDLE
    if selected is None:
        return True
    path = _absolute_no_links(selected)
    if not path.is_file():
        raise ValueError("CA bundle is unavailable")
    return str(path)


def _utc_timestamp(value: str, *, label: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed.tz_convert("UTC")


def _iso(value: pd.Timestamp) -> str:
    return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes, *, label: str) -> None:
    write_durable_exact_bytes(path, payload, label=label)


def _write_json(path: Path, payload: object, *, label: str) -> None:
    _write_bytes(path, _canonical_json(payload), label=label)


def _record(path: Path, payload: bytes) -> dict[str, object]:
    import hashlib

    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=sorted(ENERGY_CHARTS_ZONE_BY_ROLE), required=True)
    parser.add_argument("--start-utc", required=True)
    parser.add_argument("--end-utc", required=True)
    parser.add_argument("--raw-cadence-minutes", type=int, choices=(15, 60), required=True)
    parser.add_argument("--acquisition-id", default=f"local-{uuid.uuid4()}")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--ca-bundle", type=Path)
    args = parser.parse_args(argv)
    summary = capture_energy_charts_price(
        role=args.role,
        start_utc=args.start_utc,
        end_utc=args.end_utc,
        raw_cadence_minutes=args.raw_cadence_minutes,
        acquisition_id=args.acquisition_id,
        output_directory=args.output_directory,
        ca_bundle=args.ca_bundle,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
