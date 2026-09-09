from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import pytest

from pfc_shaping.cli import build_ch_lt_prospective_capture_ledger as packaged_cli
from pfc_shaping.data.governed_lt_acquisition import (
    CapturedProviderDocument,
    approved_provider_parser_payload_for_role,
    build_provider_transform_config,
    build_raw_envelope,
    energy_price_resolution_provenance,
    transform_provider_raw,
    verify_provider_raw_replay,
)
from pfc_shaping.pipeline.governed_release_cli_contract import ReleaseCliIdentityError
from pfc_shaping.validation.ch_lt_prospective_capture_ledger import (
    LEDGER_STATUS,
    REQUEST_SCHEMA,
    ChLtProspectiveCaptureLedgerError,
    build_prospective_capture_ledger,
    canonical_json_bytes,
)
from scripts.build_ch_lt_prospective_capture_ledger import main


def _write_json(path: Path, value: object) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def _utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _provider_body(start: datetime, hours: int, number: int) -> bytes:
    return canonical_json_bytes(
        {
            "license_info": "CC BY 4.0 test fixture",
            "unix_seconds": [
                int((start + timedelta(hours=offset)).timestamp())
                for offset in range(hours)
            ],
            "price": [float(number + offset) for offset in range(hours)],
            "unit": "EUR / MWh",
            "deprecated": False,
        }
    )


def _entry(
    root: Path,
    *,
    number: int,
    start: datetime,
    hours: int = 24,
    body: bytes | None = None,
    received_offset_hours: int = 1,
) -> dict[str, str]:
    build = root / "build"
    capture = build / "captures" / f"capture-{number}"
    acquisition = build / "acquisitions" / f"acquisition-{number}"
    audit_dir = build / "audits"
    capture.mkdir(parents=True)
    acquisition.mkdir(parents=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    acquisition_id = f"ch-hourly-{number}"
    end = start + timedelta(hours=hours)
    received = end + timedelta(hours=received_offset_hours)
    body_path = capture / "provider-response.json"
    body_payload = body if body is not None else _provider_body(start, hours, number)
    body_path.write_bytes(body_payload)
    body_hash = hashlib.sha256(body_payload).hexdigest()
    config = build_provider_transform_config(
        role="epex_ch",
        start_utc=_utc(start),
        end_utc=_utc(end),
    )
    parser_payload = approved_provider_parser_payload_for_role("epex_ch")
    envelope_payload = build_raw_envelope(
        acquisition_id=acquisition_id,
        source_role="epex_ch",
        source_system="ENERGY_CHARTS_API",
        source_locator="https://api.energy-charts.info/price",
        provider_id="ENERGY_CHARTS_API",
        received_at_utc=_utc(received),
        documents=(
            CapturedProviderDocument(
                document_id="day_ahead_price",
                request_url="https://api.energy-charts.info/price",
                request_parameters={
                    "bzn": "CH",
                    "start": start.strftime("%Y-%m-%dT%H:%MZ"),
                    "end": (end - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%MZ"),
                },
                response_media_type="application/json",
                body=body_payload,
                response_headers={
                    "content-length": str(len(body_payload)),
                    "content-type": "application/json",
                },
            ),
        ),
    )
    frame = transform_provider_raw(
        envelope_payload=envelope_payload,
        parser_config=config,
    )
    resolution = energy_price_resolution_provenance(frame, role="epex_ch")
    bronze_buffer = BytesIO()
    frame.to_parquet(bronze_buffer, index=True)
    bronze_payload = bronze_buffer.getvalue()
    config_payload = canonical_json_bytes(config)
    artifact_payloads = {
        "bronze.parquet": bronze_payload,
        "provider-parser.py": parser_payload,
        "provider-raw-envelope.json": envelope_payload,
        "provider-transform-config.json": config_payload,
    }
    replay = verify_provider_raw_replay(
        role="epex_ch",
        source_system="ENERGY_CHARTS_API",
        envelope_payload=envelope_payload,
        bronze_payload=bronze_payload,
        parser_payload=parser_payload,
        parser_code_sha256=hashlib.sha256(parser_payload).hexdigest(),
        parser_config=config,
    )

    spec = {
        "schema_version": "lt_provider_capture_spec.v1",
        "acquisition_id": acquisition_id,
        "source_role": "epex_ch",
        "source_system": "ENERGY_CHARTS_API",
        "source_locator": "https://api.energy-charts.info/price",
        "provider_id": "ENERGY_CHARTS_API",
        "received_at_utc": _utc(received),
        "start_utc": _utc(start),
        "end_utc": _utc(end),
        "documents": [
            {
                "document_id": "day_ahead_price",
                "body_path": str(body_path),
                "request_url": "https://api.energy-charts.info/price",
                "request_parameters": {
                    "bzn": "CH",
                    "start": start.strftime("%Y-%m-%dT%H:%MZ"),
                    "end": (end - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%MZ"),
                },
                "credential_reference": None,
                "response_status_code": 200,
                "response_media_type": "application/json",
                "response_content_encoding": "identity",
                "response_headers": {
                    "content-length": str(len(body_payload)),
                    "content-type": "application/json",
                },
            }
        ],
    }
    spec_path, spec_hash = _write_json(capture / "capture-spec.json", spec)
    summary = {
        "schema_version": "local_public_provider_capture.v2",
        "status": "COMPLETE_LOCAL_UNTRUSTED_CAPTURE",
        "source_role": "epex_ch",
        "raw_cadence_minutes": 60,
        "timestamp_authority": "LOCAL_WORKSTATION_CLOCK_UNTRUSTED",
        "publication_status": "NOT_PUBLISHED",
        "next_authority": "NETWORKLESS_INSTALLED_ACQUISITION_BUILDER",
        "production_authorization": False,
        "promotion_gate": False,
        "acquisition_id": acquisition_id,
        "start_utc": _utc(start),
        "end_utc": _utc(end),
        "received_at_utc": _utc(received),
        "validated_bronze_start_utc": _utc(start),
        "validated_bronze_end_utc": _utc(end),
        "validated_bronze_rows": hours * 4,
        "resolution_provenance": resolution,
        "capture_spec": {
            "path": str(spec_path),
            "sha256": spec_hash,
            "size_bytes": spec_path.stat().st_size,
        },
        "provider_body": {
            "path": str(body_path),
            "sha256": body_hash,
            "size_bytes": len(body_payload),
        },
    }
    summary_path, summary_hash = _write_json(capture / "capture-summary.json", summary)
    artifacts: dict[str, dict[str, object]] = {}
    for name, payload in artifact_payloads.items():
        path = acquisition / name
        path.write_bytes(payload)
        artifacts[name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    manifest = {
        "schema_version": "lt_provider_acquisition_build.v2",
        "acquisition_id": acquisition_id,
        "source_role": "epex_ch",
        "source_system": "ENERGY_CHARTS_API",
        "received_at_utc": _utc(received),
        "capture_spec": {
            "sha256": spec_hash,
            "size_bytes": spec_path.stat().st_size,
        },
        "resolution_provenance": resolution,
        "signing_status": "UNSIGNED_REQUIRES_INDEPENDENT_AUTHORITIES",
        "publication_status": "NOT_PUBLISHED",
        "namespace_security_status": "BUILDER_MUTABLE_UNTRUSTED_QUARANTINE",
        "authority_handoff_status": (
            "REQUIRES_BUILDER_INACCESSIBLE_COPY_AND_INDEPENDENT_SIGNATURE"
        ),
        "bronze_frame_sha256": replay["bronze_frame_sha256"],
        "artifacts": artifacts,
    }
    manifest_path, manifest_hash = _write_json(
        acquisition / "acquisition-build-manifest.json", manifest
    )
    false_fields = {
        field: False
        for field in (
            "scientific_admission",
            "training_eligible",
            "model_selection_eligible",
            "candidate_assembly_eligible",
            "confirmatory_rolling_origin_eligible",
            "future_holdout_eligible",
            "t057_eligible",
            "production_authorization",
            "promotion_gate",
        )
    }
    audit = {
        "schema_version": "local_provider_acquisition_audit.v4",
        "status": "VERIFIED_LOCAL_QUARANTINE_NOT_PRODUCTION",
        "manifest_sha256": manifest_hash,
        "source_role": "epex_ch",
        "source_system": "ENERGY_CHARTS_API",
        "received_at_utc": _utc(received),
        "resolution_provenance": resolution,
        "artifacts": artifacts,
        "replay": replay,
        "evidence_scope": {
            "source_observation_count": hours,
            "output_observation_count": hours * 4,
            "proxy_rows_add_independent_information": False,
        },
        "acquisition_directory": str(acquisition),
        **false_fields,
    }
    audit_path, audit_hash = _write_json(audit_dir / f"audit-{number}.json", audit)
    receipt = {
        "schema_version": "fmv_lt_provider_verifier_audit_receipt.v1",
        "command": "audit-acquisition",
        "status": "LOCAL_RUNTIME_OBSERVATION_NOT_AUTHORITY",
        "runtime_authority": False,
        "production_authorization": False,
        "audit_result": {
            "path": str(audit_path),
            "sha256": audit_hash,
            "status": "VERIFIED_LOCAL_QUARANTINE_NOT_PRODUCTION",
        },
        "runtime_observation": {
            "source_revision": "3" * 64,
            "supervisor_attestation": "UNSIGNED_LOCAL_PROCESS_OBSERVATION",
            "receipt": {
                "captured_artifact_sha256": "1" * 64,
                "dependency_tree_sha256": "2" * 64,
                "captured_artifact_sys_path_count": 1,
                "captured_dependency_root_sys_path_count": 1,
                "source_artifact_sys_path_count": 0,
                "source_dependency_root_sys_path_count": 0,
                "dependency_import_source": "PROCESS_PRIVATE_EXACT_BYTE_COPY",
                "schema_version": "fmv_lt_provider_verifier_runtime_admission.v1",
                "status": "PASS",
            }
        },
    }
    receipt_path, receipt_hash = _write_json(
        audit_dir / f"audit-{number}-runtime-receipt.json", receipt
    )
    return {
        "capture_summary_path": str(summary_path),
        "capture_summary_sha256": summary_hash,
        "acquisition_manifest_path": str(manifest_path),
        "acquisition_manifest_sha256": manifest_hash,
        "audit_path": str(audit_path),
        "audit_sha256": audit_hash,
        "audit_runtime_receipt_path": str(receipt_path),
        "audit_runtime_receipt_sha256": receipt_hash,
    }


def _request(root: Path, entries: list[dict[str, str]]) -> tuple[Path, str]:
    return _write_json(
        root / "build" / "ledger-request.json",
        {
            "schema_version": REQUEST_SCHEMA,
            "market": "CH",
            "source_role": "epex_ch",
            "entries": entries,
        },
    )


def test_builds_contiguous_negative_authority_ledger(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    entries = [
        _entry(root, number=1, start=start),
        _entry(root, number=2, start=start + timedelta(hours=24)),
    ]
    request, request_hash = _request(root, entries)

    ledger = build_prospective_capture_ledger(
        repo_root=root,
        request_path=request,
        expected_request_sha256=request_hash,
    )

    assert ledger["status"] == LEDGER_STATUS
    assert ledger["inventory"] == {
        "capture_count": 2,
        "native_hourly_observation_count": 48,
        "transport_quarter_hour_row_count": 192,
        "native_cadence_seconds": 3600,
        "transport_cadence_seconds": 900,
        "transport_sampling_kind": "UPSAMPLED_STEPWISE_PROXY",
        "proxy_rows_add_independent_information": False,
    }
    assert ledger["scientific_admission"] is False
    assert ledger["production_authorization"] is False
    assert ledger["t057_consumed"] is False
    assert ledger["entries"][0]["native_hourly_cadence_observed"] is True
    assert ledger["entries"][0]["native_hourly_scientific_truth_eligible"] is False
    assert str(ledger["entries"][0]["capture_summary"]["path"]).startswith("build/")


@pytest.mark.parametrize("delta_hours, relation", [(23, "overlap"), (25, "gap")])
def test_rejects_noncontiguous_sequence(
    tmp_path: Path, delta_hours: int, relation: str
) -> None:
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    request, request_hash = _request(
        root,
        [
            _entry(root, number=1, start=start),
            _entry(root, number=2, start=start + timedelta(hours=delta_hours)),
        ],
    )

    with pytest.raises(ChLtProspectiveCaptureLedgerError, match=relation):
        build_prospective_capture_ledger(
            repo_root=root,
            request_path=request,
            expected_request_sha256=request_hash,
        )


def test_rejects_capture_received_before_delivery_window_end(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    entry = _entry(
        root,
        number=1,
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        received_offset_hours=-1,
    )
    request, request_hash = _request(root, [entry])

    with pytest.raises(
        ChLtProspectiveCaptureLedgerError,
        match="capture chronology is invalid",
    ):
        build_prospective_capture_ledger(
            repo_root=root,
            request_path=request,
            expected_request_sha256=request_hash,
        )


def test_rejects_reused_provider_body_bytes(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    body = _provider_body(start, 24, 1)
    request, request_hash = _request(
        root,
        [
            _entry(root, number=1, start=start, body=body),
            _entry(root, number=2, start=start, body=body),
        ],
    )

    with pytest.raises(ChLtProspectiveCaptureLedgerError, match="body bytes are reused"):
        build_prospective_capture_ledger(
            repo_root=root,
            request_path=request,
            expected_request_sha256=request_hash,
        )


def test_rejects_positive_authority_claim(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    entry = _entry(
        root,
        number=1,
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    audit_path = Path(entry["audit_path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["production_authorization"] = True
    _, entry["audit_sha256"] = _write_json(audit_path, audit)
    request, request_hash = _request(root, [entry])

    with pytest.raises(
        ChLtProspectiveCaptureLedgerError,
        match="positive authority-like claim: production_authorization",
    ):
        build_prospective_capture_ledger(
            repo_root=root,
            request_path=request,
            expected_request_sha256=request_hash,
        )


def test_rejects_unknown_positive_authority_claim(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    entry = _entry(
        root,
        number=1,
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    audit_path = Path(entry["audit_path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["production_status"] = "GO"
    _, entry["audit_sha256"] = _write_json(audit_path, audit)
    request, request_hash = _request(root, [entry])

    with pytest.raises(
        ChLtProspectiveCaptureLedgerError,
        match="positive authority-like claim: production_status",
    ):
        build_prospective_capture_ledger(
            repo_root=root,
            request_path=request,
            expected_request_sha256=request_hash,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("independent_signature", True),
        ("official_product_semantics_verified", True),
        ("authoritative_source", True),
        ("valid", True),
    ],
)
def test_rejects_additional_authority_vocabulary(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    entry = _entry(
        root,
        number=1,
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    audit_path = Path(entry["audit_path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit[field] = value
    _, entry["audit_sha256"] = _write_json(audit_path, audit)
    request, request_hash = _request(root, [entry])

    with pytest.raises(
        ChLtProspectiveCaptureLedgerError,
        match=f"positive authority-like claim: {field}",
    ):
        build_prospective_capture_ledger(
            repo_root=root,
            request_path=request,
            expected_request_sha256=request_hash,
        )


def test_rehashes_current_builder_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    entry = _entry(
        root,
        number=1,
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    manifest_path = Path(entry["acquisition_manifest_path"])
    (manifest_path.parent / "bronze.parquet").write_bytes(b"changed-after-audit")
    request, request_hash = _request(root, [entry])

    with pytest.raises(
        ChLtProspectiveCaptureLedgerError,
        match="Builder artifact bronze.parquet SHA-256 mismatch",
    ):
        build_prospective_capture_ledger(
            repo_root=root,
            request_path=request,
            expected_request_sha256=request_hash,
        )


def test_rejects_caller_hash_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    entry = _entry(
        root,
        number=1,
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    request, _ = _request(root, [entry])

    with pytest.raises(ChLtProspectiveCaptureLedgerError, match="SHA-256 mismatch"):
        build_prospective_capture_ledger(
            repo_root=root,
            request_path=request,
            expected_request_sha256="0" * 64,
        )


def test_rejects_any_future_holdout_path_marker(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    entry = _entry(
        root,
        number=1,
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    request, request_hash = _request(root, [entry])
    forbidden = request.with_name("t057-ledger-request.json")
    request.rename(forbidden)

    with pytest.raises(
        ChLtProspectiveCaptureLedgerError,
        match="forbidden future-holdout marker",
    ):
        build_prospective_capture_ledger(
            repo_root=root,
            request_path=forbidden,
            expected_request_sha256=request_hash,
        )


def test_cli_writes_exact_local_only_ledger(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    entry = _entry(
        root,
        number=1,
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    request, request_hash = _request(root, [entry])
    output = root / "build" / "ledger.json"

    assert (
        main(
            [
                "--repo-root",
                str(root),
                "--request",
                str(request),
                "--expected-request-sha256",
                request_hash,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["status"] == LEDGER_STATUS
    assert document["production_authorization"] is False
    first_payload = output.read_bytes()

    assert (
        main(
            [
                "--repo-root",
                str(root),
                "--request",
                str(request),
                "--expected-request-sha256",
                request_hash,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.read_bytes() == first_payload


def test_cli_refuses_divergent_existing_target(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    request, request_hash = _request(
        root,
        [_entry(root, number=1, start=start)],
    )
    output = root / "build" / "ledger.json"
    arguments = [
        "--repo-root",
        str(root),
        "--request",
        str(request),
        "--expected-request-sha256",
        request_hash,
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    first_payload = output.read_bytes()

    request, request_hash = _request(
        root,
        [_entry(root, number=2, start=start)],
    )
    arguments[3] = str(request)
    arguments[5] = request_hash
    assert main(arguments) == 2
    assert output.read_bytes() == first_payload


def test_packaged_cli_requires_seal_then_writes_local_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    entry = _entry(
        root,
        number=1,
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    request, request_hash = _request(root, [entry])
    output = root / "build" / "packaged-ledger.json"
    execution_receipt = root / "build" / "packaged-ledger-receipt.json"
    monkeypatch.chdir(root)
    monkeypatch.setattr(packaged_cli, "assert_installed_runtime_sealed", lambda: None)
    monkeypatch.setenv(
        "PFC_LT_RUNTIME_RECEIPT_PATH",
        str(root / "build" / "runtime-receipt.json"),
    )
    monkeypatch.setenv("PFC_LT_RUNTIME_RECEIPT_SHA256", "a" * 64)

    assert (
        packaged_cli.main(
            [
                "--repo-root",
                str(root),
                "--request",
                str(request),
                "--expected-request-sha256",
                request_hash,
                "--output",
                str(output),
                "--execution-receipt-output",
                str(execution_receipt),
            ]
        )
        == 0
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["status"] == LEDGER_STATUS
    assert document["scientific_admission"] is False
    assert document["production_authorization"] is False
    receipt = json.loads(execution_receipt.read_text(encoding="utf-8"))
    assert receipt["ledger"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert receipt["constructor_runtime_receipt"]["sha256"] == "a" * 64
    assert receipt["production_authorization"] is False


def test_packaged_cli_checks_runtime_before_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _reject() -> None:
        raise ReleaseCliIdentityError("unsealed")

    monkeypatch.setattr(packaged_cli, "assert_installed_runtime_sealed", _reject)

    assert packaged_cli.main([]) == 50
