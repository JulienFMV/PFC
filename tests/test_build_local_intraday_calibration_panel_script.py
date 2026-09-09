from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from pfc_shaping.cli.audit_legacy_provider_resolution import (
    audit_legacy_resolution,
)
from pfc_shaping.cli.audit_provider_acquisition_quarantine import audit_quarantine
from pfc_shaping.cli.governed_acquisition_builder import build_acquisition
from pfc_shaping.data.governed_lt_acquisition import (
    ENERGY_CHARTS_PRICE_URL,
    ENERGY_CHARTS_SOURCE_SYSTEM,
)
from scripts import build_local_intraday_calibration_panel as panel_script
from scripts.build_local_intraday_calibration_panel import build_panel
from tests.test_audit_legacy_provider_resolution_script import _legacy_fixture


@pytest.fixture(autouse=True)
def _audit_synthetic_verifier_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _audit(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "status": "PASS",
            "production_authorization": False,
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
            "source_revision": "e" * 64,
            "dependency_tree_sha256": "b" * 64,
        }

    monkeypatch.setattr(panel_script, "audit_verifier_zipapp", _audit)


def _resolution_provenance(*, role: str, periods: int) -> dict[str, object]:
    return {
        "schema_version": "lt_observation_resolution_provenance.v2",
        "role": role,
        "source_cadence_seconds": 900,
        "output_cadence_seconds": 900,
        "source_observation_count": periods,
        "output_observation_count": periods,
        "resampling_method": "NONE",
        "output_sampling_kind": "DIRECT_15_MINUTE_OBSERVATIONS_PRODUCT_IDENTITY_UNVERIFIED",
        "native_quarter_hour_cadence_eligible": True,
        "native_hourly_cadence_eligible": False,
        "hourly_aggregation_eligible": True,
        "native_quarter_hour_truth_eligible": False,
        "product_identity_status": "EXTERNAL_PRODUCT_AUCTION_SESSION_IDENTITY_UNVERIFIED",
        "quarter_hour_truth_blocker": "EXTERNAL_PRODUCT_AUCTION_SESSION_IDENTITY_REQUIRED",
        "scientific_use_class": "DIRECT_15_MINUTE_OBSERVATIONS_PRODUCT_IDENTITY_UNVERIFIED",
    }


def _parquet(path: Path, start: str, periods: int, *, role: str | None = None) -> str:
    frame = pd.DataFrame(
        {"price_eur_mwh": range(periods)},
        index=pd.date_range(start, periods=periods, freq="15min", tz="UTC"),
    )
    if role is not None:
        frame.attrs["lt_observation_resolution_provenance"] = _resolution_provenance(
            role=role, periods=periods
        )
    frame.to_parquet(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit(
    tmp_path: Path,
    *,
    start: str,
    periods: int,
) -> tuple[Path, str, Path, str, Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    if periods != 4:
        raise ValueError("fresh audit fixture expects one downstream 15-minute hour")
    start_timestamp = pd.Timestamp(start, tz="UTC")
    raw = {
        "deprecated": False,
        "license_info": "CC BY 4.0",
        "price": [50.0, 51.0],
        "unit": "EUR / MWh",
        "unix_seconds": [
            int(start_timestamp.timestamp()),
            int((start_timestamp + pd.Timedelta(hours=1)).timestamp()),
        ],
    }
    body_path = tmp_path / "provider.json"
    body_path.write_text(json.dumps(raw), encoding="utf-8")
    end_timestamp = start_timestamp + pd.Timedelta(hours=2)
    spec = {
        "schema_version": "lt_provider_capture_spec.v1",
        "acquisition_id": f"panel-{start_timestamp.strftime('%Y%m%d%H%M')}",
        "source_role": "epex_de",
        "source_system": ENERGY_CHARTS_SOURCE_SYSTEM,
        "source_locator": ENERGY_CHARTS_PRICE_URL,
        "provider_id": ENERGY_CHARTS_SOURCE_SYSTEM,
        "received_at_utc": end_timestamp.isoformat().replace("+00:00", "Z"),
        "start_utc": start_timestamp.isoformat().replace("+00:00", "Z"),
        "end_utc": end_timestamp.isoformat().replace("+00:00", "Z"),
        "documents": [
            {
                "document_id": "day_ahead_price",
                "body_path": str(body_path),
                "request_url": ENERGY_CHARTS_PRICE_URL,
                "request_parameters": {
                    "bzn": "DE-LU",
                    "start": start_timestamp.strftime("%Y-%m-%dT%H:%MZ"),
                    "end": (start_timestamp + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%MZ"),
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
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    directory = tmp_path / "quarantine"
    manifest_path = build_acquisition(
        capture_spec_path=spec_path,
        output_directory=directory,
    )
    audit_path = tmp_path / "audit.json"
    audit = audit_quarantine(
        acquisition_directory=directory,
        expected_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        output_json=audit_path,
    )
    audit["audit_runtime"]["source_revision"] = "e" * 64
    audit_path.write_text(
        json.dumps(audit, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    audit_sha256 = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    receipt_path, receipt_sha256, artifact_path, artifact_sha256 = _write_runtime_receipt(
        audit_path=audit_path,
        audit=audit,
        audit_sha256=audit_sha256,
        command="audit-acquisition",
    )
    return (
        audit_path,
        audit_sha256,
        receipt_path,
        receipt_sha256,
        artifact_path,
        artifact_sha256,
    )


def _write_runtime_receipt(
    *,
    audit_path: Path,
    audit: dict[str, object],
    audit_sha256: str,
    command: str,
) -> tuple[Path, str, Path, str]:
    artifact_path = audit_path.with_name(f"{audit_path.stem}-verifier.pyz")
    if not artifact_path.exists():
        artifact_path.write_bytes(b"synthetic-verifier-artifact\n")
    artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    receipt = {
        "schema_version": "fmv_lt_provider_verifier_audit_receipt.v1",
        "status": "LOCAL_RUNTIME_OBSERVATION_NOT_AUTHORITY",
        "command": command,
        "production_authorization": False,
        "runtime_authority": False,
        "audit_result": {
            "path": str(audit_path),
            "sha256": audit_sha256,
            "schema_version": audit["schema_version"],
            "status": audit["status"],
        },
        "runtime_observation": {
            "source_revision": audit["audit_runtime"]["source_revision"],
            "dependency_versions": {"pandas": "2.3.3"},
            "receipt": {
                "schema_version": "fmv_lt_provider_verifier_runtime_admission.v1",
                "status": "PASS",
                "dependency_tree_sha256": "b" * 64,
                "dependency_import_source": "PROCESS_PRIVATE_EXACT_BYTE_COPY",
                "captured_artifact_sha256": artifact_sha256,
                "captured_artifact_sys_path_count": 1,
                "source_artifact_sys_path_count": 0,
                "captured_dependency_root_sys_path_count": 1,
                "source_dependency_root_sys_path_count": 0,
            },
            "supervisor_attestation": "UNSIGNED_LOCAL_PROCESS_OBSERVATION",
        },
    }
    receipt_path = audit_path.with_name(f"{audit_path.stem}-runtime-receipt.json")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return (
        receipt_path,
        hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        artifact_path,
        artifact_sha256,
    )


def _evidence_kwargs(
    evidence: tuple[Path, str, Path, str, Path, str],
) -> dict[str, object]:
    (
        audit_path,
        audit_hash,
        receipt_path,
        receipt_hash,
        artifact_path,
        artifact_hash,
    ) = evidence
    return {
        "audit_json_paths": [audit_path],
        "expected_audit_sha256": [audit_hash],
        "audit_runtime_receipt_json_paths": [receipt_path],
        "expected_audit_runtime_receipt_sha256": [receipt_hash],
        "verifier_artifact_paths": [artifact_path],
        "expected_verifier_artifact_sha256": [artifact_hash],
    }


def test_panel_binds_audits_and_writes_completion_manifest_last(tmp_path):
    base = tmp_path / "base.parquet"
    base_hash = _parquet(base, "2026-01-01T00:00:00", 4)
    evidence = _audit(
        tmp_path / "fresh",
        start="2026-01-01T01:00:00",
        periods=4,
    )
    output = tmp_path / "panel.parquet"
    manifest_path = tmp_path / "panel-manifest.json"

    manifest = build_panel(
        base_parquet=base,
        expected_base_sha256=base_hash,
        **_evidence_kwargs(evidence),
        start_utc="2026-01-01T00:00:00Z",
        end_utc="2026-01-01T02:00:00Z",
        output_parquet=output,
        output_manifest=manifest_path,
    )

    assert manifest["status"] == "COMPLETE_LOCAL_MIXED_AUTHORITY_NO_GO"
    assert manifest["production_authorization"] is False
    assert manifest["promotion_gate"] is False
    assert manifest["rows"] == 8
    assert manifest["schema_version"] == "local_intraday_calibration_panel.v3"
    assert manifest["scientific_admission"] is False
    assert manifest["confirmatory_rolling_origin_eligible"] is False
    assert manifest["local_diagnostic_allowed"] is True
    assert manifest["resolution_provenance"]["native_quarter_hour_truth_eligible"] is False
    assert (
        manifest["sources"][1]["resolution_provenance"]["native_quarter_hour_cadence_eligible"]
        is False
    )
    assert manifest["sources"][1]["resolution_provenance"]["source_observation_count"] == 2
    assert manifest["sources"][1]["resolution_provenance"]["output_observation_count"] == 8
    assert (
        manifest["sources"][1]["resolution_provenance"]["native_quarter_hour_truth_eligible"]
        is False
    )
    assert manifest["season_row_counts"] == {"Hiver": 8}
    assert hashlib.sha256(output.read_bytes()).hexdigest() == manifest["panel"]["sha256"]
    assert manifest_path.stat().st_mtime_ns >= output.stat().st_mtime_ns


@pytest.mark.parametrize(
    "field",
    (
        "production_authorization",
        "promotion_gate",
        "scientific_admission",
        "training_eligible",
        "model_selection_eligible",
        "confirmatory_rolling_origin_eligible",
        "future_holdout_eligible",
        "t057_eligible",
        "candidate_assembly_eligible",
    ),
)
def test_panel_rejects_every_overclaimed_fresh_audit_gate(
    tmp_path: Path,
    field: str,
) -> None:
    base = tmp_path / "base.parquet"
    base_hash = _parquet(base, "2026-01-01T00:00:00", 4)
    evidence = _audit(
        tmp_path / "fresh",
        start="2026-01-01T01:00:00",
        periods=4,
    )
    audit_path, _, _, _, artifact_path, artifact_hash = evidence
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit[field] = True
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    audit_hash = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    receipt_path, receipt_hash, artifact_path, artifact_hash = _write_runtime_receipt(
        audit_path=audit_path,
        audit=audit,
        audit_sha256=audit_hash,
        command="audit-acquisition",
    )
    evidence = (
        audit_path,
        audit_hash,
        receipt_path,
        receipt_hash,
        artifact_path,
        artifact_hash,
    )

    with pytest.raises(ValueError, match="not eligible for the local DE"):
        build_panel(
            base_parquet=base,
            expected_base_sha256=base_hash,
            **_evidence_kwargs(evidence),
            start_utc="2026-01-01T00:00:00Z",
            end_utc="2026-01-01T02:00:00Z",
            output_parquet=tmp_path / "panel.parquet",
            output_manifest=tmp_path / "panel-manifest.json",
        )


@pytest.mark.parametrize(
    "mutation",
    ("missing_gate", "security_admitted", "security_admitted_int", "v3"),
)
def test_panel_rejects_ambiguous_or_downgraded_fresh_audit_authority(
    tmp_path: Path,
    mutation: str,
) -> None:
    base = tmp_path / "base.parquet"
    base_hash = _parquet(base, "2026-01-01T00:00:00", 4)
    evidence = _audit(
        tmp_path / "fresh",
        start="2026-01-01T01:00:00",
        periods=4,
    )
    audit_path, _, _, _, artifact_path, artifact_hash = evidence
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if mutation == "missing_gate":
        audit["authority_gates"].pop("source_scientific_admission")
    elif mutation == "security_admitted":
        audit["authority_gates"]["security"]["admitted"] = True
    elif mutation == "security_admitted_int":
        audit["authority_gates"]["security"]["admitted"] = 0
    else:
        audit["schema_version"] = "local_provider_acquisition_audit.v3"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    audit_hash = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    receipt_path, receipt_hash, artifact_path, artifact_hash = _write_runtime_receipt(
        audit_path=audit_path,
        audit=audit,
        audit_sha256=audit_hash,
        command="audit-acquisition",
    )
    evidence = (
        audit_path,
        audit_hash,
        receipt_path,
        receipt_hash,
        artifact_path,
        artifact_hash,
    )

    with pytest.raises(
        ValueError,
        match="not eligible for the local DE|unsupported|receipt command",
    ):
        build_panel(
            base_parquet=base,
            expected_base_sha256=base_hash,
            **_evidence_kwargs(evidence),
            start_utc="2026-01-01T00:00:00Z",
            end_utc="2026-01-01T02:00:00Z",
            output_parquet=tmp_path / "panel.parquet",
            output_manifest=tmp_path / "panel-manifest.json",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("captured_artifact_sys_path_count", True),
        ("source_artifact_sys_path_count", False),
        ("captured_artifact_sha256", "d" * 64),
    ),
)
def test_panel_rejects_forged_or_bool_ambiguous_verifier_runtime_receipt(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    base = tmp_path / "base.parquet"
    base_hash = _parquet(base, "2026-01-01T00:00:00", 4)
    evidence = _audit(
        tmp_path / "fresh",
        start="2026-01-01T01:00:00",
        periods=4,
    )
    (
        audit_path,
        audit_hash,
        receipt_path,
        _,
        artifact_path,
        artifact_hash,
    ) = evidence
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["runtime_observation"]["receipt"][field] = value
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_hash = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    evidence = (
        audit_path,
        audit_hash,
        receipt_path,
        receipt_hash,
        artifact_path,
        artifact_hash,
    )

    with pytest.raises(ValueError, match="isolated runtime binding"):
        build_panel(
            base_parquet=base,
            expected_base_sha256=base_hash,
            **_evidence_kwargs(evidence),
            start_utc="2026-01-01T00:00:00Z",
            end_utc="2026-01-01T02:00:00Z",
            output_parquet=tmp_path / "panel.parquet",
            output_manifest=tmp_path / "panel-manifest.json",
        )


def test_panel_rejects_verifier_artifact_manifest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "base.parquet"
    base_hash = _parquet(base, "2026-01-01T00:00:00", 4)
    evidence = _audit(
        tmp_path / "fresh",
        start="2026-01-01T01:00:00",
        periods=4,
    )

    def _mismatched_artifact(path: Path) -> dict[str, object]:
        return {
            "status": "PASS",
            "production_authorization": False,
            "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_revision": "d" * 64,
            "dependency_tree_sha256": "b" * 64,
        }

    monkeypatch.setattr(
        panel_script,
        "audit_verifier_zipapp",
        _mismatched_artifact,
    )
    with pytest.raises(ValueError, match="artifact manifest binding"):
        build_panel(
            base_parquet=base,
            expected_base_sha256=base_hash,
            **_evidence_kwargs(evidence),
            start_utc="2026-01-01T00:00:00Z",
            end_utc="2026-01-01T02:00:00Z",
            output_parquet=tmp_path / "panel.parquet",
            output_manifest=tmp_path / "panel-manifest.json",
        )


def test_panel_rejects_manifest_semantic_hash_divergent_from_replay(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.parquet"
    base_hash = _parquet(base, "2026-01-01T00:00:00", 4)
    evidence = _audit(
        tmp_path / "fresh",
        start="2026-01-01T01:00:00",
        periods=4,
    )
    audit_path, _, _, _, artifact_path, _ = evidence
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    directory = Path(audit["acquisition_directory"])
    manifest_path = directory / "acquisition-build-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["bronze_frame_sha256"] = "d" * 64
    unsigned = dict(manifest)
    unsigned.pop("build_id")
    manifest["build_id"] = hashlib.sha256(panel_script._canonical_json(unsigned)).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    audit["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    audit_hash = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    receipt_path, receipt_hash, artifact_path, artifact_hash = _write_runtime_receipt(
        audit_path=audit_path,
        audit=audit,
        audit_sha256=audit_hash,
        command="audit-acquisition",
    )
    rebound = (
        audit_path,
        audit_hash,
        receipt_path,
        receipt_hash,
        artifact_path,
        artifact_hash,
    )

    with pytest.raises(ValueError, match="fresh acquisition manifest binding"):
        build_panel(
            base_parquet=base,
            expected_base_sha256=base_hash,
            **_evidence_kwargs(rebound),
            start_utc="2026-01-01T00:00:00Z",
            end_utc="2026-01-01T02:00:00Z",
            output_parquet=tmp_path / "panel.parquet",
            output_manifest=tmp_path / "panel-manifest.json",
        )


def test_panel_rejects_gap_without_writing_outputs(tmp_path):
    base = tmp_path / "base.parquet"
    base_hash = _parquet(base, "2026-01-01T00:00:00", 4)
    evidence = _audit(
        tmp_path / "fresh",
        start="2026-01-01T01:15:00",
        periods=4,
    )
    output = tmp_path / "panel.parquet"
    manifest = tmp_path / "panel-manifest.json"

    with pytest.raises(ValueError, match="complete contiguous"):
        build_panel(
            base_parquet=base,
            expected_base_sha256=base_hash,
            **_evidence_kwargs(evidence),
            start_utc="2026-01-01T00:00:00Z",
            end_utc="2026-01-01T02:00:00Z",
            output_parquet=output,
            output_manifest=manifest,
        )

    assert not output.exists()
    assert not manifest.exists()


def test_panel_exact_retry_is_idempotent_and_divergent_retry_is_rejected(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.parquet"
    base_hash = _parquet(base, "2026-01-01T00:00:00", 4)
    evidence = _audit(
        tmp_path / "fresh",
        start="2026-01-01T01:00:00",
        periods=4,
    )
    output = tmp_path / "panel.parquet"
    manifest_path = tmp_path / "panel-manifest.json"
    kwargs = {
        "base_parquet": base,
        "expected_base_sha256": base_hash,
        **_evidence_kwargs(evidence),
        "start_utc": "2026-01-01T00:00:00Z",
        "end_utc": "2026-01-01T02:00:00Z",
        "output_parquet": output,
        "output_manifest": manifest_path,
    }

    first = build_panel(**kwargs)
    first_panel = output.read_bytes()
    first_manifest = manifest_path.read_bytes()
    assert build_panel(**kwargs) == first
    assert output.read_bytes() == first_panel
    assert manifest_path.read_bytes() == first_manifest

    manifest_path.write_bytes(b"{}\n")
    with pytest.raises(FileExistsError, match="divergent"):
        build_panel(**kwargs)


def test_panel_rejects_output_inside_audited_acquisition(tmp_path: Path) -> None:
    base = tmp_path / "base.parquet"
    base_hash = _parquet(base, "2026-01-01T00:00:00", 4)
    evidence = _audit(
        tmp_path / "fresh",
        start="2026-01-01T01:00:00",
        periods=4,
    )
    directory = evidence[0].parent / "quarantine"
    before = {path.name for path in directory.iterdir()}

    with pytest.raises(ValueError, match="overlaps protected source evidence"):
        build_panel(
            base_parquet=base,
            expected_base_sha256=base_hash,
            **_evidence_kwargs(evidence),
            start_utc="2026-01-01T00:00:00Z",
            end_utc="2026-01-01T02:00:00Z",
            output_parquet=directory / "panel.parquet",
            output_manifest=tmp_path / "panel-manifest.json",
        )

    assert {path.name for path in directory.iterdir()} == before


def test_panel_detects_source_mutation_between_panel_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "base.parquet"
    base_hash = _parquet(base, "2026-01-01T00:00:00", 4)
    evidence = _audit(
        tmp_path / "fresh",
        start="2026-01-01T01:00:00",
        periods=4,
    )
    parser_path = evidence[0].parent / "quarantine" / "provider-parser.py"
    original_write = panel_script._write_exact_retry
    calls = 0

    def _mutating_write(path, payload, *, label):
        nonlocal calls
        original_write(path, payload, label=label)
        calls += 1
        if calls == 1:
            parser_path.write_bytes(parser_path.read_bytes() + b"# mutation\n")

    monkeypatch.setattr(panel_script, "_write_exact_retry", _mutating_write)
    with pytest.raises(ValueError, match="changed during build"):
        build_panel(
            base_parquet=base,
            expected_base_sha256=base_hash,
            **_evidence_kwargs(evidence),
            start_utc="2026-01-01T00:00:00Z",
            end_utc="2026-01-01T02:00:00Z",
            output_parquet=tmp_path / "panel.parquet",
            output_manifest=tmp_path / "panel-manifest.json",
        )
    assert not (tmp_path / "panel-manifest.json").exists()


def test_panel_accepts_hash_bound_legacy_de_reclassification_as_local_no_go(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.parquet"
    base_hash = _parquet(base, "2026-06-30T23:00:00", 4)
    directory, manifest_sha, prior_path, prior_sha = _legacy_fixture(
        tmp_path / "legacy",
        role="epex_de",
    )
    reclassification_path = tmp_path / "legacy-resolution.json"
    result = audit_legacy_resolution(
        acquisition_directory=directory,
        expected_manifest_sha256=manifest_sha,
        prior_audit_path=prior_path,
        expected_prior_audit_sha256=prior_sha,
        output_json=reclassification_path,
    )
    result["audit_runtime"]["source_revision"] = "e" * 64
    reclassification_path.write_text(json.dumps(result), encoding="utf-8")
    audit_sha = hashlib.sha256(reclassification_path.read_bytes()).hexdigest()
    receipt_path, receipt_sha, artifact_path, artifact_sha = _write_runtime_receipt(
        audit_path=reclassification_path,
        audit=result,
        audit_sha256=audit_sha,
        command="audit-legacy-resolution",
    )
    evidence = (
        reclassification_path,
        audit_sha,
        receipt_path,
        receipt_sha,
        artifact_path,
        artifact_sha,
    )

    manifest = build_panel(
        base_parquet=base,
        expected_base_sha256=base_hash,
        **_evidence_kwargs(evidence),
        start_utc="2026-06-30T23:00:00Z",
        end_utc="2026-07-02T00:00:00Z",
        output_parquet=tmp_path / "panel.parquet",
        output_manifest=tmp_path / "panel-manifest.json",
    )

    source = manifest["sources"][1]
    assert source["kind"] == "LEGACY_RESOLUTION_RECLASSIFICATION_LOCAL_NO_GO"
    assert source["resolution_provenance"]["native_quarter_hour_truth_eligible"] is False
    assert source["resolution_provenance"]["source_cadence_seconds"] == 3600


def test_panel_rejects_legacy_reclassification_that_claims_scientific_admission(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.parquet"
    base_hash = _parquet(base, "2026-06-30T23:00:00", 4)
    directory, manifest_sha, prior_path, prior_sha = _legacy_fixture(
        tmp_path / "legacy",
        role="epex_de",
    )
    reclassification_path = tmp_path / "legacy-resolution.json"
    result = audit_legacy_resolution(
        acquisition_directory=directory,
        expected_manifest_sha256=manifest_sha,
        prior_audit_path=prior_path,
        expected_prior_audit_sha256=prior_sha,
        output_json=reclassification_path,
    )
    result["audit_runtime"]["source_revision"] = "e" * 64
    result["scientific_admission"] = True
    reclassification_path.write_text(json.dumps(result), encoding="utf-8")
    audit_sha = hashlib.sha256(reclassification_path.read_bytes()).hexdigest()
    receipt_path, receipt_sha, artifact_path, artifact_sha = _write_runtime_receipt(
        audit_path=reclassification_path,
        audit=result,
        audit_sha256=audit_sha,
        command="audit-legacy-resolution",
    )
    evidence = (
        reclassification_path,
        audit_sha,
        receipt_path,
        receipt_sha,
        artifact_path,
        artifact_sha,
    )

    with pytest.raises(ValueError, match="not eligible for local migration"):
        build_panel(
            base_parquet=base,
            expected_base_sha256=base_hash,
            **_evidence_kwargs(evidence),
            start_utc="2026-06-30T23:00:00Z",
            end_utc="2026-07-02T00:00:00Z",
            output_parquet=tmp_path / "panel.parquet",
            output_manifest=tmp_path / "panel-manifest.json",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("current_reclassification_parser_sha256", "d" * 64),
        ("supersession_scope", "tampered supersession claim"),
    ),
)
def test_panel_rejects_unbound_legacy_reclassification_claims(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    base = tmp_path / "base.parquet"
    base_hash = _parquet(base, "2026-06-30T23:00:00", 4)
    directory, manifest_sha, prior_path, prior_sha = _legacy_fixture(
        tmp_path / "legacy",
        role="epex_de",
    )
    reclassification_path = tmp_path / "legacy-resolution.json"
    result = audit_legacy_resolution(
        acquisition_directory=directory,
        expected_manifest_sha256=manifest_sha,
        prior_audit_path=prior_path,
        expected_prior_audit_sha256=prior_sha,
        output_json=reclassification_path,
    )
    result["audit_runtime"]["source_revision"] = "e" * 64
    result[field] = value
    reclassification_path.write_text(json.dumps(result), encoding="utf-8")
    audit_sha = hashlib.sha256(reclassification_path.read_bytes()).hexdigest()
    receipt_path, receipt_sha, artifact_path, artifact_sha = _write_runtime_receipt(
        audit_path=reclassification_path,
        audit=result,
        audit_sha256=audit_sha,
        command="audit-legacy-resolution",
    )
    evidence = (
        reclassification_path,
        audit_sha,
        receipt_path,
        receipt_sha,
        artifact_path,
        artifact_sha,
    )

    with pytest.raises(ValueError, match="not eligible for local migration"):
        build_panel(
            base_parquet=base,
            expected_base_sha256=base_hash,
            **_evidence_kwargs(evidence),
            start_utc="2026-06-30T23:00:00Z",
            end_utc="2026-07-02T00:00:00Z",
            output_parquet=tmp_path / "panel.parquet",
            output_manifest=tmp_path / "panel-manifest.json",
        )


def test_direct_panel_cli_rejects_paths_outside_repo_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    build = root / "build"
    build.mkdir(parents=True)
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        panel_script.subprocess,
        "run",
        lambda *_args, **_kwargs: Namespace(stdout=str(root)),
    )
    args = Namespace(
        base_parquet=tmp_path / "AppData" / "base.parquet",
        audit_json=[build / "audit.json"],
        audit_runtime_receipt_json=[build / "receipt.json"],
        verifier_artifact=[build / "verifier.pyz"],
        output_parquet=build / "panel.parquet",
        output_manifest=build / "panel.json",
    )

    with pytest.raises(ValueError, match="base-parquet.*below canonical repo build"):
        panel_script._require_cli_workspace_and_paths(args, root=root)
