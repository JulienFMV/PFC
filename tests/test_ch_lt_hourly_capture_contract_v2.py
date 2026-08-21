from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pfc_shaping.cli.governed_acquisition_builder import build_acquisition
from pfc_shaping.validation.ch_lt_hourly_capture_contract_v2 import (
    ChLtHourlyCaptureContractV2Error,
    canonical_json_bytes,
    compute_contract_id,
    verify_hourly_curl_capture_contract_v2,
)
from scripts.run_ch_lt_hourly_curl_retry_from_spec import finalize_retry, prepare_retry


def _contract(root: Path, ca_bundle: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "ch_lt_hourly_day_ahead_curl_capture.v2",
        "contract_id": "",
        "lifecycle_state": "FROZEN_LOCAL_CURL_CAPTURE_NOT_ADMISSION",
        "as_of_utc": "2026-07-29T00:00:00Z",
        "market": "CH",
        "timezone": "Europe/Zurich",
        "source": {
            "role": "epex_ch",
            "source_system": "ENERGY_CHARTS_API",
            "endpoint": "https://api.energy-charts.info/price",
            "bidding_zone": "CH",
            "product": "DAY_AHEAD_PRICE_CH",
            "product_cadence_seconds": 3600,
            "product_identity_status": (
                "EXTERNAL_PRODUCT_AUCTION_SESSION_IDENTITY_UNVERIFIED"
            ),
        },
        "window": {
            "start_utc": "2026-06-28T22:00:00Z",
            "end_utc_exclusive": "2026-07-28T22:00:00Z",
            "duration_hours": 720,
            "expected_source_observation_count": 720,
            "dst_interpretation": (
                "UTC_INTERVAL_MAPPED_TO_EUROPE_ZURICH_DELIVERY_HOURS"
            ),
        },
        "execution": {
            "wrapper": "scripts/run_ch_lt_hourly_curl_retry_from_spec.py",
            "acquisition_id": "test-ch-hourly-curl-v2",
            "raw_cadence_minutes": 60,
            "expected_bronze_observation_count": 2880,
            "expected_bronze_cadence_seconds": 900,
            "bronze_sampling_kind": "UPSAMPLED_STEPWISE_PROXY",
            "native_hourly_truth_eligibility_status": (
                "BLOCKED_PENDING_PRODUCT_IDENTITY_REVISION_POLICY_"
                "TRUSTED_AVAILABILITY_AND_EXTERNAL_ADMISSION"
            ),
            "native_quarter_hour_truth_eligible": False,
            "builder_input_authorized": False,
            "external_admission_input_candidate": False,
            "transport_attestation": False,
            "output_directory_relative": "output/phase14/test-hourly-v2",
            "output_namespace_reuse_forbidden": True,
        },
        "tls": {
            "ca_bundle_path": ca_bundle.as_posix(),
            "ca_bundle_sha256": hashlib.sha256(ca_bundle.read_bytes()).hexdigest(),
            "ca_bundle_size_bytes": ca_bundle.stat().st_size,
        },
        "transport": {
            "executable": "curl.exe",
            "redirect_limit": 0,
            "protocol": "https",
            "minimum_tls_version": "1.2",
            "response_encoding_requested": "identity",
            "provider_body_max_bytes": 8388608,
            "certificate_revocation_check": (
                "DISABLED_LOCAL_UNTRUSTED_CAPTURE_ONLY"
            ),
        },
        "intended_use": {
            "allowed": [
                "LOCAL_HOURLY_SHAPE_DIAGNOSTIC",
            ],
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
        "external_authorities": {
            "trusted_time_receipt_present": False,
            "independent_signature_present": False,
            "builder_inaccessible_immutable_namespace_present": False,
            "external_cas_receipt_present": False,
            "fresh_authenticated_head_present": False,
        },
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
    payload["contract_id"] = compute_contract_id(payload)
    return payload


def _write_contract(root: Path, payload: dict[str, object]) -> tuple[Path, str]:
    parent = root / ".planning" / "phases" / "14-lt-audit-remediation"
    parent.mkdir(parents=True)
    path = parent / "CH-LT-HOURLY-DAY-AHEAD-CURL-CAPTURE-TEST.json"
    encoded = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    path.write_bytes(encoded)
    return path, hashlib.sha256(encoded).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    root = tmp_path / "repo"
    root.mkdir()
    ca_bundle = tmp_path / "ca.pem"
    ca_bundle.write_bytes(b"test-ca-bundle\n")
    return root, ca_bundle, _contract(root, ca_bundle)


def test_v2_contract_verifies_and_prepares_exact_curl_command(tmp_path: Path) -> None:
    root, _, payload = _fixture(tmp_path)
    path, document_hash = _write_contract(root, payload)

    verified = verify_hourly_curl_capture_contract_v2(
        repo_root=root,
        contract_path=path,
        expected_contract_sha256=document_hash,
    )
    command_path = prepare_retry(verified)
    command = json.loads(command_path.read_text(encoding="utf-8"))

    assert verified.base.expected_source_observation_count == 720
    assert verified.base.expected_bronze_observation_count == 2880
    assert verified.output_directory.is_dir()
    assert command["arguments"][-1].endswith(
        "bzn=CH&start=2026-06-28T22%3A00Z&end=2026-07-28T21%3A00Z"
    )
    assert command["scientific_admission"] is False
    assert command["production_authorization"] is False
    assert command["builder_input_authorized"] is False
    assert command["external_admission_input_candidate"] is False
    assert command["transport_attestation"] is False
    assert command["certificate_revocation_check_performed"] is False


def test_v2_contract_finalizes_its_own_window_without_v1_constants(
    tmp_path: Path,
) -> None:
    root, _, payload = _fixture(tmp_path)
    path, document_hash = _write_contract(root, payload)
    verified = verify_hourly_curl_capture_contract_v2(
        repo_root=root,
        contract_path=path,
        expected_contract_sha256=document_hash,
    )
    prepare_retry(verified)
    provider = {
        "license_info": "CC BY 4.0",
        "unix_seconds": [1782684000 + 3600 * index for index in range(720)],
        "price": [float(index % 24) for index in range(720)],
        "unit": "EUR / MWh",
        "deprecated": False,
    }
    verified.response_body_path.write_text(
        json.dumps(provider, separators=(",", ":")), encoding="utf-8"
    )
    verified.response_headers_path.write_bytes(
        (
            "HTTP/1.1 200 OK\r\n"
            "content-type: application/json\r\n"
            f"content-length: {verified.response_body_path.stat().st_size}\r\n"
            "date: Wed, 29 Jul 2026 08:45:09 GMT\r\n\r\n"
        ).encode("ascii")
    )

    receipt_path = finalize_retry(verified)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert receipt["schema_version"] == "ch_lt_hourly_curl_capture_receipt.v2"
    assert receipt["capture_contract_id"] == verified.retry_id
    assert receipt["capture_contract_sha256"] == document_hash
    assert "prior_failed_attempt_sha256" not in receipt
    assert "base_run_spec_id" not in receipt
    assert receipt["source_observation_count"] == 720
    assert receipt["validated_in_memory_proxy_rows"] == 2880
    assert receipt["derived_proxy_materialized"] is False
    assert receipt["builder_input_authorized"] is False
    assert receipt["external_admission_input_candidate"] is False
    assert receipt["transport_attestation"] is False
    assert receipt["certificate_revocation_check_performed"] is False
    assert receipt["scientific_admission"] is False
    assert receipt["production_authorization"] is False
    assert "capture_spec" not in receipt
    assert "bronze_observation_count" not in receipt
    assert (verified.output_directory / "local-transform-record.json").is_file()
    assert not (verified.output_directory / "capture-spec.json").exists()
    builder_parent = root / "builder-output"
    builder_parent.mkdir()
    with pytest.raises(ValueError, match="fields are not exact"):
        build_acquisition(
            capture_spec_path=(
                verified.output_directory / "local-transform-record.json"
            ).resolve(),
            output_directory=(builder_parent / "rejected").resolve(),
        )


@pytest.mark.parametrize(
    ("metadata_name", "field", "value", "error"),
    [
        (
            "capture-attempt.json",
            "scientific_execution_authorized",
            True,
            "keys are invalid",
        ),
        (
            "capture-attempt.json",
            "publication_authorized",
            True,
            "attempt publication is invalid",
        ),
        (
            "curl-command.json",
            "promotion_gate",
            True,
            "command promotion is invalid",
        ),
    ],
)
def test_v2_finalize_rejects_injected_or_weakened_prepared_claims(
    tmp_path: Path,
    metadata_name: str,
    field: str,
    value: object,
    error: str,
) -> None:
    root, _, payload = _fixture(tmp_path)
    path, document_hash = _write_contract(root, payload)
    verified = verify_hourly_curl_capture_contract_v2(
        repo_root=root,
        contract_path=path,
        expected_contract_sha256=document_hash,
    )
    prepare_retry(verified)
    metadata_path = verified.output_directory / metadata_name
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    verified.response_body_path.write_bytes(b"{}")
    verified.response_headers_path.write_bytes(
        b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n"
    )

    with pytest.raises(ValueError, match=error):
        finalize_retry(verified)


def test_v2_contract_rejects_caller_hash_mismatch(tmp_path: Path) -> None:
    root, _, payload = _fixture(tmp_path)
    path, _ = _write_contract(root, payload)

    with pytest.raises(ChLtHourlyCaptureContractV2Error, match="SHA-256 mismatch"):
        verify_hourly_curl_capture_contract_v2(
            repo_root=root,
            contract_path=path,
            expected_contract_sha256="0" * 64,
        )


def test_v2_contract_rejects_output_escape_even_when_rehashed(tmp_path: Path) -> None:
    root, _, payload = _fixture(tmp_path)
    execution = dict(payload["execution"])
    execution["output_directory_relative"] = "output/phase14/../../outside"
    payload["execution"] = execution
    payload["contract_id"] = compute_contract_id(payload)
    path, document_hash = _write_contract(root, payload)

    with pytest.raises(ChLtHourlyCaptureContractV2Error, match="governed prefix"):
        verify_hourly_curl_capture_contract_v2(
            repo_root=root,
            contract_path=path,
            expected_contract_sha256=document_hash,
        )


def test_v2_contract_rejects_truth_after_asof_even_when_rehashed(tmp_path: Path) -> None:
    root, _, payload = _fixture(tmp_path)
    payload["as_of_utc"] = "2026-07-28T21:00:00Z"
    payload["contract_id"] = compute_contract_id(payload)
    path, document_hash = _write_contract(root, payload)

    with pytest.raises(ChLtHourlyCaptureContractV2Error, match="beyond contract as-of"):
        verify_hourly_curl_capture_contract_v2(
            repo_root=root,
            contract_path=path,
            expected_contract_sha256=document_hash,
        )


def test_contract_id_uses_canonical_content_not_formatting(tmp_path: Path) -> None:
    _, _, payload = _fixture(tmp_path)
    expected = payload["contract_id"]

    assert compute_contract_id(payload) == expected
    assert canonical_json_bytes(payload).startswith(b'{"as_of_utc"')
