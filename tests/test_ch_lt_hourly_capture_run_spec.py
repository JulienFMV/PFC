from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import pfc_shaping.validation.ch_lt_hourly_capture_run_spec as contract
import scripts.run_ch_lt_hourly_capture_from_spec as runner
from pfc_shaping.validation.ch_lt_hourly_capture_run_spec import (
    DOCUMENT_SHA256,
    RETRY_DOCUMENT_SHA256,
    RETRY_ID,
    RETRY_RELATIVE_PATH,
    RUN_SPEC_ID,
    RUN_SPEC_RELATIVE_PATH,
    ChLtHourlyCaptureRunSpecError,
    compute_run_spec_id,
    verify_hourly_capture_run_spec,
    verify_hourly_curl_retry,
)
from scripts.run_ch_lt_hourly_curl_retry_from_spec import finalize_retry, prepare_retry

ROOT = Path(__file__).resolve().parents[1]
RUN_SPEC = ROOT.joinpath(*RUN_SPEC_RELATIVE_PATH.split("/"))
RETRY_SPEC = ROOT.joinpath(*RETRY_RELATIVE_PATH.split("/"))


def test_canonical_run_spec_binds_exact_request_ca_and_negative_authorities() -> None:
    verified = verify_hourly_capture_run_spec(
        repo_root=ROOT,
        run_spec_path=RUN_SPEC,
        expected_document_sha256=DOCUMENT_SHA256,
    )

    assert verified.run_spec_id == RUN_SPEC_ID
    assert verified.expected_source_observation_count == 720
    assert verified.expected_bronze_observation_count == 2880
    audit = verified.audit_dict()
    assert audit["local_capture_execution_authorized"] is True
    assert audit["scientific_execution_authorized"] is False
    assert audit["scientific_admission"] is False
    assert audit["production_authorization"] is False
    assert audit["promotion_gate"] is False


def test_run_spec_id_recomputes_from_exact_unsigned_content() -> None:
    document = json.loads(RUN_SPEC.read_text(encoding="utf-8"))
    assert compute_run_spec_id(document) == document["run_spec_id"] == RUN_SPEC_ID
    assert hashlib.sha256(RUN_SPEC.read_bytes()).hexdigest() == DOCUMENT_SHA256


def test_caller_rehashed_shadow_spec_is_rejected_before_semantics() -> None:
    document = json.loads(RUN_SPEC.read_text(encoding="utf-8"))
    document["publication_authorized"] = True
    document["run_spec_id"] = compute_run_spec_id(document)
    shadow = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()

    with pytest.raises(ChLtHourlyCaptureRunSpecError, match="frozen identity"):
        verify_hourly_capture_run_spec(
            repo_root=ROOT,
            run_spec_path=RUN_SPEC,
            expected_document_sha256=hashlib.sha256(shadow).hexdigest(),
        )


def test_noncanonical_run_spec_path_is_rejected() -> None:
    with pytest.raises(ChLtHourlyCaptureRunSpecError, match="not canonical"):
        verify_hourly_capture_run_spec(
            repo_root=ROOT,
            run_spec_path=ROOT / "shadow-run-spec.json",
            expected_document_sha256=DOCUMENT_SHA256,
        )


def test_changed_ca_bundle_bytes_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    original = contract.read_stable_single_link_file

    def _reader(path: Path, **kwargs: object) -> bytes:
        payload = original(path, **kwargs)
        if str(path).lower().endswith("git-ca-plus-fmv.crt"):
            return payload + b"changed"
        return payload

    monkeypatch.setattr(contract, "read_stable_single_link_file", _reader)
    with pytest.raises(ChLtHourlyCaptureRunSpecError, match="CA bundle identity"):
        verify_hourly_capture_run_spec(
            repo_root=ROOT,
            run_spec_path=RUN_SPEC,
            expected_document_sha256=DOCUMENT_SHA256,
        )


def test_executor_emits_last_written_local_no_go_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    canonical = verify_hourly_capture_run_spec(
        repo_root=ROOT,
        run_spec_path=RUN_SPEC,
        expected_document_sha256=DOCUMENT_SHA256,
    )
    verified = replace(canonical, output_directory=tmp_path / "capture")
    monkeypatch.setattr(runner, "verify_hourly_capture_run_spec", lambda **_: verified)

    def _capture(**kwargs: object) -> Path:
        output = Path(str(kwargs["output_directory"]))
        output.mkdir()
        body = b'{"license_info":{"name":"fixture"}}\n'
        capture_spec = b'{"schema_version":"fixture"}\n'
        (output / "capture-attempt.json").write_bytes(b"{}\n")
        (output / "provider-response.json").write_bytes(body)
        (output / "capture-spec.json").write_bytes(capture_spec)
        summary = {
            "schema_version": "local_public_provider_capture.v2",
            "status": "COMPLETE_LOCAL_UNTRUSTED_CAPTURE",
            "production_authorization": False,
            "promotion_gate": False,
            "publication_status": "NOT_PUBLISHED",
            "timestamp_authority": "LOCAL_WORKSTATION_CLOCK_UNTRUSTED",
            "next_authority": "NETWORKLESS_INSTALLED_ACQUISITION_BUILDER",
            "acquisition_id": verified.acquisition_id,
            "source_role": "epex_ch",
            "received_at_utc": "2026-07-27T00:00:01Z",
            "start_utc": verified.start_utc,
            "end_utc": verified.end_utc,
            "raw_cadence_minutes": 60,
            "resolution_provenance": {
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
                "scientific_use_class": (
                    "NATIVE_HOURLY_PRICE_QH_TRANSPORT_PROXY_ONLY"
                ),
            },
            "provider_body": _record(output / "provider-response.json", body),
            "capture_spec": _record(output / "capture-spec.json", capture_spec),
            "validated_bronze_rows": 2880,
            "validated_bronze_start_utc": verified.start_utc,
            "validated_bronze_end_utc": verified.end_utc,
        }
        summary_path = output / "capture-summary.json"
        summary_path.write_text(
            json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return summary_path

    monkeypatch.setattr(runner, "capture_energy_charts_price", _capture)
    receipt_path = runner.execute_frozen_hourly_capture(
        repo_root=ROOT,
        run_spec_path=RUN_SPEC,
        expected_run_spec_sha256=DOCUMENT_SHA256,
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == runner.RECEIPT_STATUS
    assert receipt["source_observation_count"] == 720
    assert receipt["native_hourly_truth_eligible_after_external_admission"] is True
    assert receipt["native_quarter_hour_truth_eligible"] is False
    assert receipt["scientific_admission"] is False
    assert receipt["publication_authorized"] is False
    assert receipt["production_authorization"] is False
    assert receipt["receipt_last_written"] is True
    assert {path.name for path in receipt_path.parent.iterdir()} == {
        "capture-attempt.json",
        "provider-response.json",
        "capture-spec.json",
        "capture-summary.json",
        "run-receipt.json",
    }


def test_canonical_curl_retry_binds_failed_attempt_and_exact_argv() -> None:
    retry = verify_hourly_curl_retry(
        repo_root=ROOT,
        base_run_spec_path=RUN_SPEC,
        expected_base_sha256=DOCUMENT_SHA256,
        retry_path=RETRY_SPEC,
        expected_retry_sha256=RETRY_DOCUMENT_SHA256,
    )

    assert retry.retry_id == RETRY_ID
    assert retry.curl_arguments[:4] == (
        "-L",
        "--ssl-no-revoke",
        "--max-redirs",
        "0",
    )
    assert retry.curl_arguments[-1].startswith("https://api.energy-charts.info/price?")
    assert retry.audit_dict()["scientific_admission"] is False


def test_caller_rehashed_curl_retry_is_rejected() -> None:
    document = json.loads(RETRY_SPEC.read_text(encoding="utf-8"))
    document["publication_authorized"] = True
    document["retry_id"] = contract.compute_retry_id(document)
    shadow = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()

    with pytest.raises(ChLtHourlyCaptureRunSpecError, match="frozen identity"):
        verify_hourly_curl_retry(
            repo_root=ROOT,
            base_run_spec_path=RUN_SPEC,
            expected_base_sha256=DOCUMENT_SHA256,
            retry_path=RETRY_SPEC,
            expected_retry_sha256=hashlib.sha256(shadow).hexdigest(),
        )


def test_curl_retry_prepare_and_finalize_emit_exact_local_receipt(tmp_path: Path) -> None:
    canonical = verify_hourly_curl_retry(
        repo_root=ROOT,
        base_run_spec_path=RUN_SPEC,
        expected_base_sha256=DOCUMENT_SHA256,
        retry_path=RETRY_SPEC,
        expected_retry_sha256=RETRY_DOCUMENT_SHA256,
    )
    output = tmp_path / "curl-capture"
    body_path = output / "provider-response.json"
    headers_path = output / "response-headers.txt"
    old_root = canonical.output_directory.as_posix()
    new_root = output.as_posix()
    arguments = tuple(value.replace(old_root, new_root) for value in canonical.curl_arguments)
    retry = replace(
        canonical,
        output_directory=output,
        response_body_path=body_path,
        response_headers_path=headers_path,
        curl_arguments=arguments,
    )

    command_path = prepare_retry(retry)
    command = json.loads(command_path.read_text(encoding="utf-8"))
    assert command["arguments"] == list(arguments)
    provider = {
        "license_info": "CC BY 4.0",
        "unix_seconds": [1782511200 + 3600 * index for index in range(720)],
        "price": [float(index % 24) for index in range(720)],
        "unit": "EUR / MWh",
        "deprecated": False,
    }
    body_path.write_text(
        json.dumps(provider, separators=(",", ":")), encoding="utf-8"
    )
    headers_path.write_bytes(
        (
            "HTTP/1.1 200 OK\r\n"
            "content-type: application/json\r\n"
            f"content-length: {body_path.stat().st_size}\r\n"
            "date: Mon, 27 Jul 2026 12:45:32 GMT\r\n\r\n"
        ).encode("ascii")
    )

    receipt_path = finalize_retry(retry)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "COMPLETE_LOCAL_UNTRUSTED_CURL_CAPTURE_NOT_ADMITTED"
    assert receipt["source_observation_count"] == 720
    assert receipt["bronze_observation_count"] == 2880
    assert receipt["scientific_admission"] is False
    assert receipt["production_authorization"] is False
    assert receipt["receipt_last_written"] is True
    assert {path.name for path in output.iterdir()} == {
        "capture-attempt.json",
        "curl-command.json",
        "provider-response.json",
        "response-headers.txt",
        "capture-spec.json",
        "capture-summary.json",
        "run-receipt.json",
    }


def _record(path: Path, payload: bytes) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
