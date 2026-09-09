from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from pfc_shaping.cli import audit_legacy_provider_resolution as audit_script
from pfc_shaping.cli.audit_legacy_provider_resolution import audit_legacy_resolution
from pfc_shaping.cli.governed_acquisition_builder import build_acquisition
from pfc_shaping.data.governed_lt_acquisition import (
    ENERGY_CHARTS_PRICE_URL,
    ENERGY_CHARTS_SOURCE_SYSTEM,
)
from scripts import audit_legacy_provider_resolution as checkout_stub

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "governed_lt_acquisition"
    / "energy_charts_price_ch.json"
)


def _legacy_fixture(
    tmp_path: Path,
    *,
    role: str = "epex_ch",
) -> tuple[Path, str, Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    spec = {
        "schema_version": "lt_provider_capture_spec.v1",
        "acquisition_id": "legacy-resolution-001",
        "source_role": role,
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
                    "bzn": "CH" if role == "epex_ch" else "DE-LU",
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

    bronze_path = directory / "bronze.parquet"
    bronze = pd.read_parquet(bronze_path)
    bronze.attrs.clear()
    bronze.to_parquet(bronze_path, index=True)
    bronze_payload = bronze_path.read_bytes()

    manifest = json.loads(manifest_path.read_bytes())
    manifest["schema_version"] = "lt_provider_acquisition_build.v1"
    manifest.pop("resolution_provenance")
    manifest["artifacts"]["bronze.parquet"] = {
        "sha256": _sha256(bronze_payload),
        "size_bytes": len(bronze_payload),
    }
    manifest["bronze_frame_sha256"] = _legacy_frame_sha256(bronze)
    unsigned = dict(manifest)
    unsigned.pop("build_id")
    manifest["build_id"] = _sha256(_canonical_json(unsigned))
    manifest_payload = _canonical_json(manifest)
    manifest_path.write_bytes(manifest_payload)
    manifest_sha = _sha256(manifest_payload)

    artifacts = manifest["artifacts"]
    prior = {
        "acquisition_directory": str(directory.resolve()),
        "artifacts": artifacts,
        "manifest_sha256": manifest_sha,
        "negative_authority": {
            "authority_handoff_status": manifest["authority_handoff_status"],
            "namespace_security_status": manifest["namespace_security_status"],
            "publication_status": manifest["publication_status"],
            "signing_status": manifest["signing_status"],
        },
        "production_authorization": False,
        "promotion_gate": False,
        "received_at_utc": manifest["received_at_utc"],
        "remaining_authority": (
            "BUILDER_INACCESSIBLE_COPY_ACL_FREEZE_TRUSTED_TIME_AND_INDEPENDENT_SIGNATURE"
        ),
        "replay": {
            "bronze_frame_sha256": manifest["bronze_frame_sha256"],
            "envelope_sha256": artifacts["provider-raw-envelope.json"]["sha256"],
            "role": role,
            "schema_version": "lt_provider_transform_config.v1",
            "status": "VERIFIED_EXACT_PROVIDER_REPLAY",
            "transform_id": "pfc.provider.energy_charts_price.v1",
        },
        "schema_version": "local_provider_acquisition_audit.v1",
        "source_role": role,
        "source_system": ENERGY_CHARTS_SOURCE_SYSTEM,
        "status": "VERIFIED_LOCAL_QUARANTINE_NOT_PRODUCTION",
    }
    prior_path = tmp_path / "legacy-audit.json"
    prior_payload = _canonical_json(prior) + b"\n"
    prior_path.write_bytes(prior_payload)
    return directory, manifest_sha, prior_path, _sha256(prior_payload)


def test_legacy_hourly_quarantine_is_reclassified_without_qh_authority(
    tmp_path: Path,
) -> None:
    directory, manifest_sha, prior_path, prior_sha = _legacy_fixture(tmp_path)
    output = tmp_path / "resolution-audit.json"

    result = audit_legacy_resolution(
        acquisition_directory=directory,
        expected_manifest_sha256=manifest_sha,
        prior_audit_path=prior_path,
        expected_prior_audit_sha256=prior_sha,
        output_json=output,
    )

    assert result["status"] == "LEGACY_RESOLUTION_RECLASSIFIED_LOCAL_NO_GO"
    assert result["production_authorization"] is False
    assert result["scientific_admission"] is False
    assert result["legacy_audit"]["resolution_authority"] is False
    assert result["resolution_provenance"]["source_cadence_seconds"] == 3600
    assert result["resolution_provenance"][
        "native_quarter_hour_truth_eligible"
    ] is False
    assert result["eligibility"]["native_hourly_price_evidence"] == (
        "LOCAL_UNSIGNED_ONLY"
    )
    assert result["eligibility"]["native_quarter_hour_price_truth"] == "INELIGIBLE"
    assert result["audit_runtime"]["admission_status"] == (
        "DIRECT_FUNCTION_CALL_NOT_RUNTIME_AUTHORITY"
    )
    assert result["audit_runtime"]["dependency_versions"] == {}
    assert result["audit_runtime"]["runtime_admission"] == {}
    assert json.loads(output.read_bytes()) == result


def test_direct_legacy_audit_cannot_accept_forged_runtime_claim(
    tmp_path: Path,
) -> None:
    directory, manifest_sha, prior_path, prior_sha = _legacy_fixture(tmp_path)

    with pytest.raises(TypeError, match="runtime_dependency_versions"):
        audit_legacy_resolution(
            acquisition_directory=directory,
            expected_manifest_sha256=manifest_sha,
            prior_audit_path=prior_path,
            expected_prior_audit_sha256=prior_sha,
            output_json=tmp_path / "resolution-audit.json",
            runtime_dependency_versions={},  # type: ignore[call-arg]
        )

    assert not hasattr(checkout_stub, "audit_legacy_resolution")
    assert checkout_stub.main() == 50


def test_legacy_resolution_audit_rejects_rebound_prior_audit(tmp_path: Path) -> None:
    directory, manifest_sha, prior_path, _ = _legacy_fixture(tmp_path)
    prior = json.loads(prior_path.read_bytes())
    prior["manifest_sha256"] = "0" * 64
    payload = _canonical_json(prior) + b"\n"
    prior_path.write_bytes(payload)

    with pytest.raises(ValueError, match="binding is invalid"):
        audit_legacy_resolution(
            acquisition_directory=directory,
            expected_manifest_sha256=manifest_sha,
            prior_audit_path=prior_path,
            expected_prior_audit_sha256=_sha256(payload),
            output_json=tmp_path / "resolution-audit.json",
        )


def test_legacy_resolution_exact_retry_and_output_boundaries(tmp_path: Path) -> None:
    directory, manifest_sha, prior_path, prior_sha = _legacy_fixture(tmp_path)
    output = tmp_path / "resolution-audit.json"

    first = audit_legacy_resolution(
        acquisition_directory=directory,
        expected_manifest_sha256=manifest_sha,
        prior_audit_path=prior_path,
        expected_prior_audit_sha256=prior_sha,
        output_json=output,
    )
    payload = output.read_bytes()
    second = audit_legacy_resolution(
        acquisition_directory=directory,
        expected_manifest_sha256=manifest_sha,
        prior_audit_path=prior_path,
        expected_prior_audit_sha256=prior_sha,
        output_json=output,
    )
    assert second == first
    assert output.read_bytes() == payload

    output.write_bytes(b"{}\n")
    with pytest.raises(FileExistsError, match="divergent"):
        audit_legacy_resolution(
            acquisition_directory=directory,
            expected_manifest_sha256=manifest_sha,
            prior_audit_path=prior_path,
            expected_prior_audit_sha256=prior_sha,
            output_json=output,
        )

    with pytest.raises(ValueError, match="overlaps protected evidence"):
        audit_legacy_resolution(
            acquisition_directory=directory,
            expected_manifest_sha256=manifest_sha,
            prior_audit_path=prior_path,
            expected_prior_audit_sha256=prior_sha,
            output_json=directory / "resolution-audit.json",
        )


def test_legacy_resolution_detects_prior_audit_mutation_during_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, manifest_sha, prior_path, prior_sha = _legacy_fixture(tmp_path)
    original_write = audit_script._write_exact_retry

    def _mutating_write(path, payload, *, label):
        original_write(path, payload, label=label)
        prior_path.write_bytes(prior_path.read_bytes() + b" ")

    monkeypatch.setattr(audit_script, "_write_exact_retry", _mutating_write)
    with pytest.raises(ValueError, match="changed during audit"):
        audit_legacy_resolution(
            acquisition_directory=directory,
            expected_manifest_sha256=manifest_sha,
            prior_audit_path=prior_path,
            expected_prior_audit_sha256=prior_sha,
            output_json=tmp_path / "resolution-audit.json",
        )


def _legacy_frame_sha256(frame: pd.DataFrame) -> str:
    metadata = {
        "columns": [str(value) for value in frame.columns],
        "dtypes": [str(value) for value in frame.dtypes],
        "index_names": [str(value) for value in frame.index.names],
        "index_dtype": str(frame.index.dtype),
        "rows": len(frame),
    }
    digest = hashlib.sha256(_canonical_json(metadata))
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
