from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pfc_shaping.cli import audit_provider_acquisition_quarantine as audit_script
from pfc_shaping.cli.audit_provider_acquisition_quarantine import audit_quarantine
from pfc_shaping.cli.governed_acquisition_builder import build_acquisition
from pfc_shaping.data.governed_lt_acquisition import (
    ENERGY_CHARTS_PRICE_URL,
    ENERGY_CHARTS_SOURCE_SYSTEM,
)
from scripts import audit_provider_acquisition_quarantine as checkout_stub

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "governed_lt_acquisition"
    / "energy_charts_price_ch.json"
)


def _build(tmp_path: Path) -> tuple[Path, str]:
    spec = {
        "schema_version": "lt_provider_capture_spec.v1",
        "acquisition_id": "audit-fixture-001",
        "source_role": "epex_ch",
        "source_system": ENERGY_CHARTS_SOURCE_SYSTEM,
        "source_locator": ENERGY_CHARTS_PRICE_URL,
        "provider_id": ENERGY_CHARTS_SOURCE_SYSTEM,
        "received_at_utc": "2026-07-02T00:00:00Z",
        "start_utc": "2026-07-01T00:00:00Z",
        "end_utc": "2026-07-02T00:00:00Z",
        "documents": [
            {
                "document_id": "day_ahead_price",
                "body_path": str(FIXTURE.resolve()),
                "request_url": ENERGY_CHARTS_PRICE_URL,
                "request_parameters": {
                    "bzn": "CH",
                    "start": "2026-07-01T00:00Z",
                    "end": "2026-07-01T23:00Z",
                },
                "credential_reference": None,
                "response_status_code": 200,
                "response_media_type": "application/json",
                "response_content_encoding": "identity",
                "response_headers": {"etag": '"fixture"'},
            }
        ],
    }
    spec_path = tmp_path / "capture-spec.json"
    spec_path.write_text(
        json.dumps(spec, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    directory = tmp_path / "quarantine"
    manifest_path = build_acquisition(
        capture_spec_path=spec_path,
        output_directory=directory,
    )
    return directory, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def test_audit_replays_exact_quarantine_and_preserves_negative_authority(tmp_path):
    directory, manifest_sha256 = _build(tmp_path)
    output = tmp_path / "audit.json"

    audit = audit_quarantine(
        acquisition_directory=directory,
        expected_manifest_sha256=manifest_sha256,
        output_json=output,
    )

    assert audit["status"] == "VERIFIED_LOCAL_QUARANTINE_NOT_PRODUCTION"
    assert audit["schema_version"] == "local_provider_acquisition_audit.v4"
    assert audit["production_authorization"] is False
    assert audit["promotion_gate"] is False
    assert audit["scientific_admission"] is False
    assert audit["training_eligible"] is False
    assert audit["model_selection_eligible"] is False
    assert audit["confirmatory_rolling_origin_eligible"] is False
    assert audit["future_holdout_eligible"] is False
    assert audit["t057_eligible"] is False
    assert audit["candidate_assembly_eligible"] is False
    assert audit["local_diagnostic_allowed"] is True
    assert audit["authority_gates"] == {
        "security": {
            "status": "MISSING_EXTERNAL_AUTHORITIES",
            "admitted": False,
            "blocking_controls": list(
                audit_script._SECURITY_BLOCKING_CONTROLS
            ),
        },
        "source_scientific_admission": {
            "status": "MISSING_SOURCE_AUTHORITIES",
            "admitted": False,
            "blocking_evidence": list(
                audit_script._SOURCE_SCIENTIFIC_BLOCKING_EVIDENCE
            ),
        },
        "candidate_readiness": {
            "status": "INSUFFICIENT_CONFIRMATORY_EVIDENCE",
            "admitted": False,
            "blocking_evidence": list(
                audit_script._CANDIDATE_READINESS_BLOCKING_EVIDENCE
            ),
        },
    }
    assert audit["evidence_scope"] == {
        "capture_episode_count": 1,
        "source_observation_count": 24,
        "output_observation_count": 96,
        "independent_information_unit_upper_bound": 24,
        "proxy_rows_add_independent_information": False,
        "seasonal_regime_coverage": "NOT_ESTABLISHED_BY_SINGLE_CAPTURE_EPISODE",
    }
    assert audit["replay"]["status"] == "VERIFIED_EXACT_PROVIDER_REPLAY"
    assert audit["negative_authority"]["publication_status"] == "NOT_PUBLISHED"
    assert audit["resolution_provenance"]["source_cadence_seconds"] == 3600
    assert audit["resolution_provenance"][
        "native_quarter_hour_truth_eligible"
    ] is False
    assert audit["audit_runtime"] == {
        "source_revision": audit_script.SOURCE_REVISION,
        "admission_status": "DIRECT_FUNCTION_CALL_NOT_RUNTIME_AUTHORITY",
        "dependency_versions": {},
        "runtime_admission": {},
        "provider_transform_config_sha256": audit["artifacts"][
            "provider-transform-config.json"
        ]["sha256"],
    }
    assert json.loads(output.read_text(encoding="utf-8")) == audit


def test_direct_audit_cannot_accept_forged_runtime_claim(tmp_path: Path) -> None:
    directory, manifest_sha256 = _build(tmp_path)

    with pytest.raises(TypeError, match="runtime_dependency_versions"):
        audit_quarantine(
            acquisition_directory=directory,
            expected_manifest_sha256=manifest_sha256,
            output_json=tmp_path / "audit.json",
            runtime_dependency_versions={},  # type: ignore[call-arg]
        )

    assert not hasattr(checkout_stub, "audit_quarantine")
    assert checkout_stub.main() == 50


def test_audit_rejects_wrong_caller_held_manifest_hash(tmp_path):
    directory, _ = _build(tmp_path)

    with pytest.raises(ValueError, match="caller-held SHA-256"):
        audit_quarantine(
            acquisition_directory=directory,
            expected_manifest_sha256="0" * 64,
            output_json=tmp_path / "audit.json",
        )

    assert not (tmp_path / "audit.json").exists()


@pytest.mark.parametrize("claim", (True, 1))
def test_audit_rejects_rehashed_native_quarter_hour_relabelling(
    tmp_path: Path,
    claim: object,
) -> None:
    directory, _ = _build(tmp_path)
    manifest_path = directory / "acquisition-build-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["resolution_provenance"]["native_quarter_hour_truth_eligible"] = claim
    unsigned = dict(manifest)
    unsigned.pop("build_id")
    unsigned_payload = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    manifest["build_id"] = hashlib.sha256(unsigned_payload).hexdigest()
    payload = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    manifest_path.write_bytes(payload)

    with pytest.raises(ValueError, match="resolution provenance differs"):
        audit_quarantine(
            acquisition_directory=directory,
            expected_manifest_sha256=hashlib.sha256(payload).hexdigest(),
            output_json=tmp_path / "audit.json",
        )

    assert not (tmp_path / "audit.json").exists()


def test_audit_exact_retry_is_idempotent_and_divergent_retry_is_rejected(tmp_path):
    directory, manifest_sha256 = _build(tmp_path)
    output = tmp_path / "audit.json"

    first = audit_quarantine(
        acquisition_directory=directory,
        expected_manifest_sha256=manifest_sha256,
        output_json=output,
    )
    first_payload = output.read_bytes()
    second = audit_quarantine(
        acquisition_directory=directory,
        expected_manifest_sha256=manifest_sha256,
        output_json=output,
    )

    assert second == first
    assert output.read_bytes() == first_payload
    output.write_bytes(b"{}\n")
    with pytest.raises(FileExistsError, match="divergent"):
        audit_quarantine(
            acquisition_directory=directory,
            expected_manifest_sha256=manifest_sha256,
            output_json=output,
        )


def test_audit_rejects_output_inside_protected_quarantine(tmp_path):
    directory, manifest_sha256 = _build(tmp_path)

    with pytest.raises(ValueError, match="overlaps protected evidence"):
        audit_quarantine(
            acquisition_directory=directory,
            expected_manifest_sha256=manifest_sha256,
            output_json=directory / "audit.json",
        )

    assert {path.name for path in directory.iterdir()} == audit_script._INVENTORY


def test_audit_detects_quarantine_mutation_during_publication(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    directory, manifest_sha256 = _build(tmp_path)
    original_write = audit_script._write_exact_retry

    def _mutating_write(path, payload, *, label):
        original_write(path, payload, label=label)
        parser_path = directory / "provider-parser.py"
        parser_path.write_bytes(parser_path.read_bytes() + b"\n")

    monkeypatch.setattr(audit_script, "_write_exact_retry", _mutating_write)
    with pytest.raises(ValueError, match="changed during audit"):
        audit_quarantine(
            acquisition_directory=directory,
            expected_manifest_sha256=manifest_sha256,
            output_json=tmp_path / "audit.json",
        )


def test_packaged_audit_main_reports_structured_integrity_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    directory, _ = _build(tmp_path)

    code = audit_script.main(
        [
            "--acquisition-directory",
            str(directory),
            "--expected-manifest-sha256",
            "0" * 64,
            "--output-json",
            str(tmp_path / "audit.json"),
        ]
    )

    failure = json.loads(capsys.readouterr().err)
    assert code == 50
    assert failure["status"] == "INTEGRITY_FAILURE"
    assert failure["production_authorization"] is False
    assert failure["retry_disposition"] == (
        "DO_NOT_RETRY_USE_ISOLATED_VERIFIER_ZIPAPP"
    )
