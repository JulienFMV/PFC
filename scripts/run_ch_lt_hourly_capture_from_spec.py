"""Execute the one frozen local CH hourly Energy Charts capture attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.pipeline.strict_structured_data import load_strict_json
from pfc_shaping.validation.ch_lt_hourly_capture_run_spec import (
    DOCUMENT_SHA256,
    ChLtHourlyCaptureRunSpecError,
    VerifiedHourlyCaptureRunSpec,
    verify_hourly_capture_run_spec,
)
from scripts.capture_public_energy_charts_lt import capture_energy_charts_price

RECEIPT_SCHEMA_VERSION = "ch_lt_hourly_day_ahead_local_capture_receipt.v1"
RECEIPT_STATUS = "COMPLETE_LOCAL_UNTRUSTED_HOURLY_CAPTURE_NOT_ADMITTED"
_MAX_CAPTURE_FILE_BYTES = 8 * 1024 * 1024


def execute_frozen_hourly_capture(
    *, repo_root: str | Path, run_spec_path: str | Path, expected_run_spec_sha256: str
) -> Path:
    """Run the frozen request once and emit a last-written negative-authority receipt."""

    verified = verify_hourly_capture_run_spec(
        repo_root=repo_root,
        run_spec_path=run_spec_path,
        expected_document_sha256=expected_run_spec_sha256,
    )
    summary_path = capture_energy_charts_price(
        role=verified.role,
        start_utc=verified.start_utc,
        end_utc=verified.end_utc,
        raw_cadence_minutes=verified.raw_cadence_minutes,
        acquisition_id=verified.acquisition_id,
        output_directory=verified.output_directory,
        ca_bundle=verified.ca_bundle_path,
    )
    summary_payload = _stable(summary_path, label="capture summary")
    summary = _strict_mapping(summary_payload, label="capture summary")
    records = _verify_summary(summary, verified=verified, summary_path=summary_path)
    receipt_path = verified.output_directory / "run-receipt.json"
    if receipt_path.exists():
        raise ChLtHourlyCaptureRunSpecError("run receipt already exists")
    inventory_before_receipt = {path.name for path in verified.output_directory.iterdir()}
    if inventory_before_receipt != {
        "capture-attempt.json",
        "provider-response.json",
        "capture-spec.json",
        "capture-summary.json",
    }:
        raise ChLtHourlyCaptureRunSpecError("capture output inventory is not exact")
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": RECEIPT_STATUS,
        "run_spec": {
            "path": str(Path(run_spec_path).resolve()),
            "sha256": DOCUMENT_SHA256,
            "run_spec_id": verified.run_spec_id,
        },
        "ca_bundle": {
            "path": str(verified.ca_bundle_path),
            "sha256": (
                "e68ca0a61d7e000d0c10fb3a712be09c533378b99af5fa4c30a34c7aa8f36f25"
            ),
            "size_bytes": 258962,
        },
        "acquisition_id": verified.acquisition_id,
        "source_role": verified.role,
        "source_product": "DAY_AHEAD_PRICE_CH",
        "source_cadence_seconds": 3600,
        "source_observation_count": verified.expected_source_observation_count,
        "bronze_cadence_seconds": 900,
        "bronze_observation_count": verified.expected_bronze_observation_count,
        "bronze_sampling_kind": "UPSAMPLED_STEPWISE_PROXY",
        "native_hourly_truth_eligible_after_external_admission": True,
        "native_quarter_hour_truth_eligible": False,
        "start_utc": verified.start_utc,
        "end_utc_exclusive": verified.end_utc,
        "capture_summary": _record(summary_path, summary_payload),
        "provider_body": records["provider_body"],
        "capture_spec": records["capture_spec"],
        "local_capture_completed": True,
        "local_workstation_clock_trusted": False,
        "trusted_time_receipt_present": False,
        "independent_signature_present": False,
        "external_cas_receipt_present": False,
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
    write_durable_exact_bytes(
        receipt_path,
        _canonical_json(receipt),
        label="CH LT hourly local capture receipt",
    )
    return receipt_path


def _verify_summary(
    summary: Mapping[str, object], *, verified: VerifiedHourlyCaptureRunSpec, summary_path: Path
) -> dict[str, dict[str, object]]:
    expected = {
        "schema_version": "local_public_provider_capture.v2",
        "status": "COMPLETE_LOCAL_UNTRUSTED_CAPTURE",
        "production_authorization": False,
        "promotion_gate": False,
        "publication_status": "NOT_PUBLISHED",
        "timestamp_authority": "LOCAL_WORKSTATION_CLOCK_UNTRUSTED",
        "next_authority": "NETWORKLESS_INSTALLED_ACQUISITION_BUILDER",
        "acquisition_id": verified.acquisition_id,
        "source_role": verified.role,
        "start_utc": verified.start_utc,
        "end_utc": verified.end_utc,
        "raw_cadence_minutes": 60,
        "validated_bronze_rows": verified.expected_bronze_observation_count,
        "validated_bronze_start_utc": verified.start_utc,
        "validated_bronze_end_utc": verified.end_utc,
    }
    for field, value in expected.items():
        _equal(summary.get(field), value, f"summary {field}")
    resolution = _mapping(summary.get("resolution_provenance"), label="resolution")
    _equal(
        dict(resolution),
        {
            "schema_version": "lt_observation_resolution_provenance.v2",
            "role": "epex_ch",
            "source_cadence_seconds": 3600,
            "output_cadence_seconds": 900,
            "source_observation_count": 720,
            "output_observation_count": 2880,
            "resampling_method": "FORWARD_FILL_WITHIN_SOURCE_INTERVAL",
            "output_sampling_kind": "UPSAMPLED_STEPWISE_PROXY",
            "native_quarter_hour_cadence_eligible": False,
            "native_hourly_cadence_eligible": True,
            "hourly_aggregation_eligible": True,
            "native_quarter_hour_truth_eligible": False,
            "product_identity_status": (
                "EXTERNAL_PRODUCT_AUCTION_SESSION_IDENTITY_UNVERIFIED"
            ),
            "quarter_hour_truth_blocker": "SOURCE_CADENCE_IS_HOURLY",
            "scientific_use_class": "NATIVE_HOURLY_PRICE_QH_TRANSPORT_PROXY_ONLY",
        },
        "resolution provenance",
    )
    if _path_key(summary_path) != _path_key(verified.output_directory / "capture-summary.json"):
        raise ChLtHourlyCaptureRunSpecError("capture summary path is not canonical")
    result: dict[str, dict[str, object]] = {}
    for role, filename in (
        ("provider_body", "provider-response.json"),
        ("capture_spec", "capture-spec.json"),
    ):
        record = _mapping(summary.get(role), label=role)
        path = Path(str(record.get("path")))
        expected_path = verified.output_directory / filename
        if _path_key(path) != _path_key(expected_path):
            raise ChLtHourlyCaptureRunSpecError(f"{role} path is not canonical")
        payload = _stable(path, label=role)
        if len(payload) != record.get("size_bytes"):
            raise ChLtHourlyCaptureRunSpecError(f"{role} size mismatch")
        if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
            raise ChLtHourlyCaptureRunSpecError(f"{role} SHA-256 mismatch")
        result[role] = _record(path, payload)
    return result


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


def _equal(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ChLtHourlyCaptureRunSpecError(f"{label} is invalid")


def _stable(path: Path, *, label: str) -> bytes:
    return read_stable_single_link_file(
        path,
        label=label,
        max_bytes=_MAX_CAPTURE_FILE_BYTES,
    )


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


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-spec", type=Path, required=True)
    parser.add_argument("--expected-run-spec-sha256", required=True)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        receipt = execute_frozen_hourly_capture(
            repo_root=repo_root,
            run_spec_path=args.run_spec,
            expected_run_spec_sha256=args.expected_run_spec_sha256,
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "ch_lt_hourly_capture_failure.v1",
                    "status": "LOCAL_CAPTURE_FAILED_NO_ADMISSION",
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
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
