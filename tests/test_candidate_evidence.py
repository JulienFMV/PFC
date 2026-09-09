from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from pfc_shaping.calibration.monthly_curve_lambda_calibration import config_hash
from pfc_shaping.pipeline.candidate_evidence import (
    EVIDENCE_MANIFEST,
    GOVERNED_RUN_SPEC_SCHEMA,
    REQUIRED_EVIDENCE_ROLES,
    CandidateEvidenceError,
    EvidenceArtifactInput,
    _canonical_mapping_sha256,
    _run_spec_artifact_bindings,
    capture_pre_run_governance_evidence,
    seal_candidate_evidence,
    verify_candidate_evidence,
    verify_pre_run_governance_evidence,
)
from pfc_shaping.pipeline.model_governance_contract import (
    sign_model_governance_artifact_receipt,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_yaml
from pfc_shaping.version import __version__


@pytest.fixture(autouse=True)
def _governance_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key = Ed25519PrivateKey.generate()
    private = tmp_path / "governance-private.pem"
    public = tmp_path / "governance-public.pem"
    private.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv("PFC_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH", str(private))
    monkeypatch.setenv("PFC_MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_PATH", str(public))


def _staging(tmp_path: Path, *, run_id: str = "run-001") -> Path:
    staging = tmp_path / "staging"
    manifests = staging / "manifests"
    manifests.mkdir(parents=True)
    artifacts = {
        "markets/CH/pfc.csv": b"timestamp;price\n2026-07-13T00:00:00Z;80\n",
        "models/shared/shape.bin": b"deterministic-model",
    }
    inventory: dict[str, str] = {}
    for relative, payload in artifacts.items():
        path = staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        inventory[relative] = hashlib.sha256(payload).hexdigest()
    inventory_sha256 = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (manifests / "candidate_run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "reference_timestamp": "2026-07-13T00:00:00+00:00",
                "deterministic_artifacts": inventory,
                "deterministic_artifacts_sha256": inventory_sha256,
            }
        ),
        encoding="utf-8",
    )
    return staging


def _artifacts(staging: Path) -> dict[str, EvidenceArtifactInput]:
    root = staging / "evidence"
    root.mkdir()
    artifacts: dict[str, EvidenceArtifactInput] = {}
    for role in sorted(REQUIRED_EVIDENCE_ROLES):
        path = root / f"{role}.json"
        if role == "historical_thresholds":
            _write_historical_threshold_fixture(path)
        elif role == "selected_lambda_decision":
            canonical = {"lambda_prior": 1.0}
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "monthly_curve_selected_lambda_decision.v1",
                        "decision_id": "decision-test",
                        "calibration_campaign_id": "campaign-test",
                        "selection_reason": "test fixture",
                        "selection_status": "PRODUCTION_APPROVED",
                        "production_approved": True,
                        "canonical_config": canonical,
                        "config_hash": config_hash(canonical),
                    }
                ),
                encoding="utf-8",
            )
        else:
            path.write_text(json.dumps({"role": role}), encoding="utf-8")
        receipt = None
        if role in {"historical_thresholds", "selected_lambda_decision"}:
            receipt = root / f"{role}.authority.json"
            receipt.write_text(
                json.dumps(
                    sign_model_governance_artifact_receipt(
                        path,
                        artifact_role=role,
                        authority_id="FMV_MODEL_GOVERNANCE_TEST",
                        issued_at_utc="2026-07-12T00:00:00Z",
                        data_cutoff_utc="2026-07-11T00:00:00Z",
                        private_key_path=os.environ[
                            "PFC_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH"
                        ],
                    )
                ),
                encoding="utf-8",
            )
        artifacts[role] = EvidenceArtifactInput(
            path=path,
            producer_contract=f"{role}.producer.v1",
            authority_receipt_path=receipt,
        )
    _bind_pre_run_fixture(staging, artifacts)
    return artifacts


def _write_historical_threshold_fixture(path: Path) -> None:
    columns = [
        "gate_id", "metric", "market", "delivery_bucket", "lookback_start",
        "lookback_end", "n_snapshots", "min_required_n", "p50", "p90",
        "p975", "max_observed", "regime_filter", "status",
    ]
    lines = [",".join(columns)]
    for gate, metric in (
        ("same_month_rank_consistency", "same_month_shape_delta_abs_eur_mwh"),
        ("residual_vs_implied_comparable_block", "comparable_block_shape_delta_abs_eur_mwh"),
    ):
        for bucket in ["all", *(f"month_{month:02d}" for month in range(1, 13))]:
            lines.append(
                f"{gate},{metric},CH,{bucket},,,0,24,,,,,test_fixture,UNSUPPORTED"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bind_pre_run_fixture(
    staging: Path,
    artifacts: dict[str, EvidenceArtifactInput],
) -> None:
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    entries: dict[str, object] = {}
    for role in ("historical_thresholds", "selected_lambda_decision"):
        item = artifacts[role]
        artifact = Path(item.path)
        receipt = Path(item.authority_receipt_path)
        receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
        entries[role] = {
            "path": artifact.relative_to(staging).as_posix(),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "size_bytes": artifact.stat().st_size,
            "authority_receipt": {
                "path": receipt.relative_to(staging).as_posix(),
                "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                "receipt_id": receipt_payload["receipt_id"],
                "authority_id": receipt_payload["authority_id"],
                "issued_at_utc": receipt_payload["issued_at_utc"],
                "data_cutoff_utc": receipt_payload["data_cutoff_utc"],
            },
        }
    config_path_value = run.get(
        "config_path",
        "evidence/independent/runtime_config/config.yaml",
    )
    config_path = staging / str(config_path_value)
    if not config_path.is_file():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("model: {}\n", encoding="utf-8")
    config_payload = load_strict_yaml(config_path.read_text(encoding="utf-8"))
    entries["runtime_config"] = {
        "path": config_path.relative_to(staging).as_posix(),
        "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "size_bytes": config_path.stat().st_size,
        "semantic_sha256": hashlib.sha256(
            json.dumps(
                config_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest(),
    }
    run_spec = {
        "schema_version": GOVERNED_RUN_SPEC_SCHEMA,
        "run_id": run["run_id"],
        "as_of_utc": run["reference_timestamp"],
        "build_timestamp_utc": "2026-07-13T00:01:00+00:00",
        "run_started_at_utc": "2026-07-13T00:01:00+00:00",
        "runtime": {
            "distribution": "fmv-pfc-lt",
            "version": __version__,
            "source_revision": "1" * 40,
        },
        "config": {
            key: entries["runtime_config"][key]
            for key in ("path", "sha256", "semantic_sha256")
        },
        "input_snapshot": {
            "generation_id": "generation-001",
            "contract_logical_path": "snapshots/generation-001/lt_input_snapshot.json",
            "contract_sha256": "2" * 64,
            "pointer_logical_path": "views/pfc_lt/current.json",
            "pointer_sha256": "3" * 64,
        },
        "model_controls": {
            "peak_source_policy": "same_first",
            "use_seasonal_hourly_shape": True,
        },
        "pre_run_artifacts": _run_spec_artifact_bindings(entries),
    }
    pre_path = staging / "manifests" / "pre_run_governance_manifest.json"
    pre_path.write_text(
        json.dumps(
            {
                "schema_version": "pre_run_model_governance.v2",
                "run_id": run["run_id"],
                "valuation_timestamp": run["reference_timestamp"],
                "captured_at_utc": "2026-07-13T00:01:00+00:00",
                "artifacts": entries,
                "governed_run_spec": run_spec,
                "governed_run_spec_sha256": _canonical_mapping_sha256(run_spec),
                "promotion_eligible": False,
            }
        ),
        encoding="utf-8",
    )
    run["pre_run_governance_manifest"] = "manifests/pre_run_governance_manifest.json"
    run["pre_run_governance_manifest_sha256"] = hashlib.sha256(
        pre_path.read_bytes()
    ).hexdigest()
    run_path.write_text(json.dumps(run), encoding="utf-8")


def test_seal_candidate_evidence_binds_exact_dag_to_run_manifest(tmp_path: Path) -> None:
    staging = _staging(tmp_path)

    manifest = seal_candidate_evidence(
        staging,
        run_id="run-001",
        artifacts=_artifacts(staging),
    )

    assert set(manifest["artifacts"]) == set(REQUIRED_EVIDENCE_ROLES)
    assert manifest["promotion_eligible"] is False
    verified = verify_candidate_evidence(staging, expected_run_id="run-001")
    assert verified == manifest
    run = json.loads(
        (staging / "manifests" / "candidate_run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert run["candidate_evidence_manifest"] == EVIDENCE_MANIFEST.as_posix()
    assert len(run["candidate_evidence_manifest_sha256"]) == 64


def test_sealing_rejects_model_artifact_mutated_after_inventory(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    (staging / "markets" / "CH" / "pfc.csv").write_text(
        "mutated-before-seal",
        encoding="utf-8",
    )

    with pytest.raises(
        CandidateEvidenceError,
        match="deterministic artifacts are incomplete or invalid",
    ):
        seal_candidate_evidence(
            staging,
            run_id="run-001",
            artifacts=_artifacts(staging),
        )


def test_seal_candidate_evidence_rejects_missing_role(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    artifacts = _artifacts(staging)
    artifacts.pop("audit_gates")

    with pytest.raises(CandidateEvidenceError, match="roles mismatch"):
        seal_candidate_evidence(staging, run_id="run-001", artifacts=artifacts)


def test_seal_candidate_evidence_rejects_same_file_for_multiple_roles(
    tmp_path: Path,
) -> None:
    staging = _staging(tmp_path)
    artifacts = _artifacts(staging)
    production = artifacts["production_manifest"]
    artifacts["export_manifest"] = EvidenceArtifactInput(
        path=production.path,
        producer_contract="export.producer.v1",
    )

    with pytest.raises(CandidateEvidenceError, match="path is reused"):
        seal_candidate_evidence(staging, run_id="run-001", artifacts=artifacts)


def test_seal_candidate_evidence_requires_independent_authority_receipt(
    tmp_path: Path,
) -> None:
    staging = _staging(tmp_path)
    artifacts = _artifacts(staging)
    selected = artifacts["selected_lambda_decision"]
    artifacts["selected_lambda_decision"] = EvidenceArtifactInput(
        path=selected.path,
        producer_contract=selected.producer_contract,
    )

    with pytest.raises(CandidateEvidenceError, match="authority receipt is missing"):
        seal_candidate_evidence(staging, run_id="run-001", artifacts=artifacts)


def test_seal_candidate_evidence_rejects_signed_invalid_selected_decision(
    tmp_path: Path,
) -> None:
    staging = _staging(tmp_path)
    artifacts = _artifacts(staging)
    selected = Path(artifacts["selected_lambda_decision"].path)
    receipt = Path(artifacts["selected_lambda_decision"].authority_receipt_path)
    selected.write_text(json.dumps({"garbage_selected_after_the_fact": True}), encoding="utf-8")
    receipt.write_text(
        json.dumps(
            sign_model_governance_artifact_receipt(
                selected,
                artifact_role="selected_lambda_decision",
                authority_id="FMV_MODEL_GOVERNANCE_TEST",
                issued_at_utc="2026-07-12T00:00:00Z",
                data_cutoff_utc="2026-07-11T00:00:00Z",
                private_key_path=os.environ[
                    "PFC_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH"
                ],
            )
        ),
        encoding="utf-8",
    )
    _bind_pre_run_fixture(staging, artifacts)

    with pytest.raises(CandidateEvidenceError, match="semantic_validation"):
        seal_candidate_evidence(staging, run_id="run-001", artifacts=artifacts)


def test_verify_candidate_evidence_rejects_artifact_mutation(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    artifacts = _artifacts(staging)
    seal_candidate_evidence(staging, run_id="run-001", artifacts=artifacts)
    Path(artifacts["audit_gates"].path).write_text("tampered", encoding="utf-8")

    with pytest.raises(CandidateEvidenceError, match="audit_gates.sha256"):
        verify_candidate_evidence(staging, expected_run_id="run-001")


def test_verify_candidate_evidence_rejects_manifest_role_relabel(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    seal_candidate_evidence(staging, run_id="run-001", artifacts=_artifacts(staging))
    manifest_path = staging / EVIDENCE_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["audit_gates"]["trust_class"] = "independent_external"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CandidateEvidenceError, match="trust_class"):
        verify_candidate_evidence(staging, expected_run_id="run-001")


def test_verify_candidate_evidence_rejects_post_seal_run_mutation(
    tmp_path: Path,
) -> None:
    staging = _staging(tmp_path)
    seal_candidate_evidence(staging, run_id="run-001", artifacts=_artifacts(staging))
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["config_sha256"] = "f" * 64
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceError, match="candidate_run_preseal_sha256"):
        verify_candidate_evidence(staging, expected_run_id="run-001")


def test_pre_run_governance_captures_authenticated_inputs_before_solve(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "pre-run-staging"
    staging.mkdir()
    sources = tmp_path / "sources"
    sources.mkdir()
    thresholds = sources / "historical_thresholds.csv"
    decision = sources / "selected_lambda_decision.json"
    _write_historical_threshold_fixture(thresholds)
    decision.write_text(json.dumps({"decision": "lambda"}), encoding="utf-8")
    runtime_config = sources / "config.yaml"
    runtime_config.write_text("model: {}\n", encoding="utf-8")
    threshold_receipt = sources / "thresholds.receipt.json"
    decision_receipt = sources / "decision.receipt.json"
    for role, artifact, receipt in (
        ("historical_thresholds", thresholds, threshold_receipt),
        ("selected_lambda_decision", decision, decision_receipt),
    ):
        receipt.write_text(
            json.dumps(
                sign_model_governance_artifact_receipt(
                    artifact,
                    artifact_role=role,
                    authority_id="FMV_MODEL_GOVERNANCE_TEST",
                    issued_at_utc="2026-07-12T00:00:00Z",
                    data_cutoff_utc="2026-07-11T00:00:00Z",
                    private_key_path=os.environ[
                        "PFC_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH"
                    ],
                )
            ),
            encoding="utf-8",
        )

    manifest = capture_pre_run_governance_evidence(
        staging,
        run_id="run-001",
        valuation_timestamp="2026-07-13T00:00:00+00:00",
        build_timestamp="2026-07-13T00:01:00+00:00",
        historical_thresholds_path=thresholds,
        historical_thresholds_receipt_path=threshold_receipt,
        selected_lambda_decision_path=decision,
        selected_lambda_decision_receipt_path=decision_receipt,
        runtime_config_path=runtime_config,
        expected_runtime_config_sha256=hashlib.sha256(
            runtime_config.read_bytes()
        ).hexdigest(),
        source_revision="1" * 40,
        input_snapshot_sha256="2" * 64,
        input_pointer_sha256="3" * 64,
        input_generation_id="generation-001",
        peak_source_policy="same_first",
        use_seasonal_hourly_shape=True,
    )

    assert set(manifest["artifacts"]) == {
        "historical_thresholds",
        "selected_lambda_decision",
        "runtime_config",
    }
    assert verify_pre_run_governance_evidence(
        staging,
        expected_run_id="run-001",
        expected_valuation_timestamp="2026-07-13T00:00:00+00:00",
        require_only_pre_run_files=True,
    ) == manifest


def test_pre_run_governance_rejects_reused_source_paths(tmp_path: Path) -> None:
    staging = tmp_path / "pre-run-staging"
    staging.mkdir()
    artifact = tmp_path / "same.json"
    artifact.write_text("same", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    runtime_config = tmp_path / "config.yaml"
    runtime_config.write_text("model: {}\n", encoding="utf-8")

    with pytest.raises(CandidateEvidenceError, match="source paths must be unique"):
        capture_pre_run_governance_evidence(
            staging,
            run_id="run-001",
            valuation_timestamp="2026-07-13T00:00:00+00:00",
            build_timestamp="2026-07-13T00:01:00+00:00",
            historical_thresholds_path=artifact,
            historical_thresholds_receipt_path=receipt,
            selected_lambda_decision_path=artifact,
            selected_lambda_decision_receipt_path=receipt,
            runtime_config_path=runtime_config,
            expected_runtime_config_sha256=hashlib.sha256(
                runtime_config.read_bytes()
            ).hexdigest(),
            source_revision="1" * 40,
            input_snapshot_sha256="2" * 64,
            input_pointer_sha256="3" * 64,
            input_generation_id="generation-001",
            peak_source_policy="same_first",
            use_seasonal_hourly_shape=True,
        )


def test_pre_run_governance_rejects_invalid_source_revision_before_io(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "pre-run-staging"
    staging.mkdir()

    with pytest.raises(CandidateEvidenceError, match="source revision"):
        capture_pre_run_governance_evidence(
            staging,
            run_id="run-001",
            valuation_timestamp="2026-07-13T00:00:00+00:00",
            build_timestamp="2026-07-13T00:01:00+00:00",
            historical_thresholds_path=tmp_path / "missing-thresholds.csv",
            historical_thresholds_receipt_path=tmp_path / "missing-thresholds.json",
            selected_lambda_decision_path=tmp_path / "missing-decision.json",
            selected_lambda_decision_receipt_path=tmp_path / "missing-receipt.json",
            runtime_config_path=tmp_path / "missing-config.yaml",
            expected_runtime_config_sha256="a" * 64,
            source_revision="branch-name",
            input_snapshot_sha256="2" * 64,
            input_pointer_sha256="3" * 64,
            input_generation_id="generation-001",
            peak_source_policy="same_first",
            use_seasonal_hourly_shape=True,
        )

    assert list(staging.iterdir()) == []


def test_pre_run_governance_rejects_rehashed_invalid_runtime_identity(
    tmp_path: Path,
) -> None:
    staging = _staging(tmp_path)
    _artifacts(staging)
    manifest_path = staging / "manifests" / "pre_run_governance_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_spec = manifest["governed_run_spec"]
    run_spec["runtime"]["source_revision"] = "branch-name"
    manifest["governed_run_spec_sha256"] = _canonical_mapping_sha256(run_spec)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CandidateEvidenceError, match="runtime identity"):
        verify_pre_run_governance_evidence(
            staging,
            expected_run_id="run-001",
            expected_valuation_timestamp="2026-07-13T00:00:00+00:00",
        )
