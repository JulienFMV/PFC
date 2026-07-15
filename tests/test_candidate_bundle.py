from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from pfc_shaping.calibration.monthly_curve_lambda_calibration import config_hash
from pfc_shaping.data.lt_input_sources import InputSourceReceipt, dataframe_sha256
from pfc_shaping.pipeline.atomic_promotion import (
    create_candidate_staging,
    finalize_candidate_staging,
    verify_candidate_bundle,
)
from pfc_shaping.pipeline.candidate_bundle import (
    CandidateSerializationError,
    write_long_term_candidate,
)
from pfc_shaping.pipeline.candidate_evidence import (
    GOVERNED_RUN_SPEC_SCHEMA,
    REQUIRED_EVIDENCE_ROLES,
    EvidenceArtifactInput,
    _canonical_mapping_sha256,
    _run_spec_artifact_bindings,
    seal_candidate_evidence,
)
from pfc_shaping.pipeline.model_governance_contract import (
    sign_model_governance_artifact_receipt,
)
from pfc_shaping.pipeline.monthly_curve_authority import (
    active_monthly_curve_config_payload,
    monthly_solver_settings,
)
from pfc_shaping.pipeline.production_phases import (
    LoadedInputs,
    LongTermArtifacts,
    MarketBranchArtifacts,
    SharedStructuralArtifacts,
)
from pfc_shaping.pipeline.quality_gate import QualityReport
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


class _Serializable:
    def __init__(self, content: str) -> None:
        self.content = content

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.content, encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[LongTermArtifacts, LoadedInputs, Path]:
    project = tmp_path / "project"
    data_dir = project / "pfc_shaping" / "data"
    data_dir.mkdir(parents=True)
    config_path = project / "pfc_shaping" / "config.yaml"
    config_path.write_text("model: {}\n", encoding="utf-8")
    receipts: dict[str, InputSourceReceipt] = {}
    roles = {
        "epex_15min.parquet": "epex_ch",
        "epex_de_15min.parquet": "epex_de",
        "entso_15min.parquet": "entso",
        "hydro_reservoir.parquet": "hydro",
    }
    for name in (
        "epex_15min.parquet",
        "epex_de_15min.parquet",
        "entso_15min.parquet",
        "hydro_reservoir.parquet",
    ):
        path = data_dir / name
        payload = name.encode("ascii")
        path.write_bytes(payload)
        role = roles[name]
        receipts[role] = InputSourceReceipt(
            role=role,
            path=str(path.resolve()),
            logical_path=name,
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            rows=0,
            frame_sha256=dataframe_sha256(pd.DataFrame()),
        )
    config_payload = config_path.read_bytes()
    config_receipt = InputSourceReceipt(
        role="config",
        path=str(config_path.resolve()),
        logical_path="config.yaml",
        size_bytes=len(config_payload),
        sha256=hashlib.sha256(config_payload).hexdigest(),
    )
    source = tmp_path / "Price_Report_EEX.xlsx"
    source.write_bytes(b"exact-eex-workbook")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    snapshot = {
        "source_path": str(source.resolve()),
        "source_sha256": source_hash,
        "snapshot_date": "2026-07-13T00:00:00+00:00",
        "market": "CH",
        "snapshot_id": "snapshot",
        "observation_id": "observation",
        "quotes": [
            {
                "product": "2027",
                "load_type": "BASE",
                "price_eur_mwh": 80.0,
                "quote_id": "quote-2027",
            }
        ],
    }
    index = pd.date_range("2027-01-01", periods=4, freq="15min", tz="UTC")
    pfc = pd.DataFrame(
        {
            "price_shape": [80.0, 81.0, 82.0, 83.0],
            "profile_type": ["BASE"] * 4,
            "confidence": [1.0] * 4,
        },
        index=index,
    )
    forbidden_output = tmp_path / "shared-output" / "pfc"
    branch = MarketBranchArtifacts(
        code="CH",
        pfc=pfc,
        sh=_Serializable("shape"),
        base_prices={"2027": 80.0},
        cascaded_prices={"2027": 80.0},
        fwd_source="fixture",
        out_base=str(forbidden_output),
        forward_snapshot=snapshot,
        final_product_projection_report={"status": "PASS"},
        wv=_Serializable("water"),
    )
    manifest = {
        "monthly_solution_hash": "solution",
        "active_constraints_hash": "constraints",
        "active_config_hash": "config",
        "forward_snapshot": snapshot,
    }
    long_term = LongTermArtifacts(
        shared=SharedStructuralArtifacts(
            si=_Serializable("intraday"),
            unc=_Serializable("uncertainty"),
            calibrator=None,
            entso_forecast=pd.DataFrame(),
            start_date="2027-01-01",
            horizon_days=365,
        ),
        markets={"CH": branch},
        out_dir=str(tmp_path / "shared-output"),
        artifacts_dir=str(tmp_path / "shared-models"),
        today="2026-07-13",
        monthly_curve_manifests={"CH": manifest},
    )
    inputs = LoadedInputs(
        epex_ch=pd.DataFrame(),
        epex_de=pd.DataFrame(),
        neighbor_prices_15min={},
        entso=pd.DataFrame(),
        hydro=pd.DataFrame(),
        cal_ch=pd.DataFrame(),
        cal_de=pd.DataFrame(),
        commodities=None,
        outages_all=None,
        config={},
        sh_mode="table",
        eex_report_path=str(source),
        reference_timestamp=pd.Timestamp("2026-07-13T12:00:00Z"),
        reference_timestamp_is_explicit=True,
        freshness_reports={
            "epex_ch": QualityReport("epex_ch", 5, [], [], {"age_days": 0.0})
        },
        input_source_receipts=receipts,
        config_source_receipt=config_receipt,
        data_root=str(data_dir),
        data_layout="legacy_repo",
        config_semantic_sha256=hashlib.sha256(b"{}").hexdigest(),
    )
    return long_term, inputs, project


def test_lt_candidate_is_serialized_only_inside_atomic_staging(tmp_path: Path) -> None:
    long_term, inputs, project = _fixture(tmp_path)
    release_root = tmp_path / "releases"
    staging = create_candidate_staging(release_root, run_id="run-001")

    run_manifest = write_long_term_candidate(
        staging,
        run_id="run-001",
        long_term=long_term,
        inputs=inputs,
        project_root=project,
    )
    _seal_evidence(staging, "run-001")
    bundle = finalize_candidate_staging(
        release_root,
        run_id="run-001",
        staging_path=staging,
    )

    verified = verify_candidate_bundle(release_root, run_id="run-001")
    assert verified.manifest_sha256 == bundle.manifest_sha256
    assert run_manifest["pipeline_scope"] == "LT_ONLY"
    assert run_manifest["promotion_eligible"] is False
    assert run_manifest["probabilistic_status"] == "DETERMINISTIC_ONLY"
    assert run_manifest["interval_columns"] == []
    assert run_manifest["intervals_permitted_for_pricing_or_risk"] is False
    assert (bundle.path / "markets" / "CH" / "pfc_15min.parquet").is_file()
    assert (bundle.path / "models" / "shared" / "shape_intraday.parquet").is_file()
    assert not (bundle.path / "models" / "shared" / "uncertainty.parquet").exists()
    evidence = run_manifest["source_evidence"]
    archive = bundle.path / str(evidence["forward_source_archive"])
    assert archive.read_bytes() == b"exact-eex-workbook"
    quote_snapshot = bundle.path / str(evidence["forward_quote_snapshot"])
    assert pd.read_parquet(quote_snapshot).iloc[0]["product"] == "2027"
    portable_manifest = json.loads(
        (
            bundle.path
            / "manifests"
            / "production_monthly_curve_manifest_ch.json"
        ).read_text(encoding="utf-8")
    )
    portable_source = (
        bundle.path
        / "manifests"
        / portable_manifest["forward_snapshot"]["source_path"]
    ).resolve()
    assert portable_source.is_relative_to(bundle.path)
    assert portable_source == archive.resolve()
    assert not (tmp_path / "shared-output" / "pfc.parquet").exists()


def test_lt_candidate_rejects_finite_uncalibrated_intervals(tmp_path: Path) -> None:
    long_term, inputs, project = _fixture(tmp_path)
    long_term.markets["CH"].pfc["p10"] = 70.0
    long_term.markets["CH"].pfc["p90"] = 90.0
    staging = create_candidate_staging(tmp_path / "releases", run_id="run-intervals")

    with pytest.raises(CandidateSerializationError, match="uncalibrated interval columns"):
        write_long_term_candidate(
            staging,
            run_id="run-intervals",
            long_term=long_term,
            inputs=inputs,
            project_root=project,
        )


def _seal_evidence(staging: Path, run_id: str) -> None:
    evidence = staging / "evidence" / "promotion"
    evidence.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, EvidenceArtifactInput] = {}
    run = json.loads(
        (staging / "manifests" / "candidate_run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    config = yaml.safe_load((staging / run["config_path"]).read_text(encoding="utf-8"))
    canonical = active_monthly_curve_config_payload(monthly_solver_settings(config))
    for role in sorted(REQUIRED_EVIDENCE_ROLES):
        artifact = evidence / f"{role}.json"
        if role == "historical_thresholds":
            _write_historical_threshold_fixture(artifact)
        elif role == "selected_lambda_decision":
            artifact.write_text(
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
            artifact.write_text(json.dumps({"role": role}), encoding="utf-8")
        receipt = None
        if role in {"historical_thresholds", "selected_lambda_decision"}:
            receipt = evidence / f"{role}.authority.json"
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
        artifacts[role] = EvidenceArtifactInput(
            path=artifact,
            producer_contract="test.fixture.v1",
            authority_receipt_path=receipt,
        )
    _bind_pre_run_fixture(staging, run_id, artifacts)
    seal_candidate_evidence(staging, run_id=run_id, artifacts=artifacts)


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
    run_id: str,
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
    config_path = staging / str(run["config_path"])
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
    generation_id = str(run.get("data_generation_id") or "generation-001")
    contract = run.get("data_contract") or {}
    pointer = run.get("data_pointer") or {}
    run_spec = {
        "schema_version": GOVERNED_RUN_SPEC_SCHEMA,
        "run_id": run_id,
        "as_of_utc": run["reference_timestamp"],
        "build_timestamp_utc": run["reference_timestamp"],
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
            "generation_id": generation_id,
            "contract_logical_path": (
                f"snapshots/{generation_id}/lt_input_snapshot.json"
            ),
            "contract_sha256": contract.get("sha256", "2" * 64),
            "pointer_logical_path": "views/pfc_lt/current.json",
            "pointer_sha256": pointer.get("sha256", "3" * 64),
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
                "run_id": run_id,
                "valuation_timestamp": run["reference_timestamp"],
                "captured_at_utc": run["reference_timestamp"],
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


def test_lt_candidate_blocks_forward_source_changed_after_solve(tmp_path: Path) -> None:
    long_term, inputs, project = _fixture(tmp_path)
    source = Path(inputs.eex_report_path)
    source.write_bytes(b"changed-after-solve")
    staging = create_candidate_staging(tmp_path / "releases", run_id="run-001")

    with pytest.raises(CandidateSerializationError, match="changed before candidate serialization"):
        write_long_term_candidate(
            staging,
            run_id="run-001",
            long_term=long_term,
            inputs=inputs,
            project_root=project,
        )


def test_candidate_run_manifest_is_json_and_binds_input_hashes(tmp_path: Path) -> None:
    long_term, inputs, project = _fixture(tmp_path)
    staging = create_candidate_staging(tmp_path / "releases", run_id="run-001")
    write_long_term_candidate(
        staging,
        run_id="run-001",
        long_term=long_term,
        inputs=inputs,
        project_root=project,
    )

    manifest = json.loads(
        (staging / "manifests" / "candidate_run_manifest.json").read_text(encoding="utf-8")
    )
    expected = hashlib.sha256(
        (project / "pfc_shaping" / "data" / "epex_15min.parquet").read_bytes()
    ).hexdigest()
    assert manifest["input_sources"]["epex_ch"]["sha256"] == expected
    assert manifest["reference_timestamp_is_explicit"] is True
    assert manifest["data_root_id"] == "legacy_repo"
    assert "data_root" not in manifest
    assert manifest["config_path"] == "evidence/config/config.yaml"


def test_deterministic_artifact_hash_is_run_and_path_independent(tmp_path: Path) -> None:
    manifests: list[dict[str, object]] = []
    for index in (1, 2):
        case = tmp_path / f"case-{index}"
        long_term, inputs, project = _fixture(case)
        staging = create_candidate_staging(
            case / "releases",
            run_id=f"run-{index:03d}",
        )
        manifests.append(
            write_long_term_candidate(
                staging,
                run_id=f"run-{index:03d}",
                long_term=long_term,
                inputs=inputs,
                project_root=project,
            )
        )

    assert manifests[0]["deterministic_artifacts"] == manifests[1][
        "deterministic_artifacts"
    ]
    assert manifests[0]["deterministic_artifacts_sha256"] == manifests[1][
        "deterministic_artifacts_sha256"
    ]


def test_legacy_external_pointer_is_not_labelled_as_canonical_view(tmp_path: Path) -> None:
    long_term, inputs, project = _fixture(tmp_path)
    pointer = tmp_path / "current.json"
    pointer_payload = b'{"schema_version":"lt_data_pointer.v1"}'
    pointer.write_bytes(pointer_payload)
    legacy_receipt = InputSourceReceipt(
        role="lt_data_pointer",
        path=str(pointer.resolve()),
        logical_path="current.json",
        size_bytes=len(pointer_payload),
        sha256=hashlib.sha256(pointer_payload).hexdigest(),
    )
    inputs = replace(
        inputs,
        data_layout="external_v2",
        data_generation_id="generation-001",
        data_pointer_receipt=legacy_receipt,
        data_pointer_bytes=pointer_payload,
    )
    staging = create_candidate_staging(tmp_path / "releases", run_id="run-001")

    write_long_term_candidate(
        staging,
        run_id="run-001",
        long_term=long_term,
        inputs=inputs,
        project_root=project,
    )

    manifest = json.loads(
        (staging / "manifests" / "candidate_run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["data_root_id"] == "legacy_external_pointer"


def test_candidate_blocks_core_input_changed_after_consumption(tmp_path: Path) -> None:
    long_term, inputs, project = _fixture(tmp_path)
    (project / "pfc_shaping" / "data" / "entso_15min.parquet").write_bytes(b"replaced")
    staging = create_candidate_staging(tmp_path / "releases", run_id="run-001")

    with pytest.raises(CandidateSerializationError, match="changed after it was consumed: entso"):
        write_long_term_candidate(
            staging,
            run_id="run-001",
            long_term=long_term,
            inputs=inputs,
            project_root=project,
        )


def test_candidate_blocks_consumed_frame_mutation(tmp_path: Path) -> None:
    long_term, inputs, project = _fixture(tmp_path)
    inputs.entso.loc[pd.Timestamp("2026-07-13T00:00:00Z"), "load_mw"] = 9999.0
    staging = create_candidate_staging(tmp_path / "releases", run_id="run-001")

    with pytest.raises(CandidateSerializationError, match="input frame changed.*entso"):
        write_long_term_candidate(
            staging,
            run_id="run-001",
            long_term=long_term,
            inputs=inputs,
            project_root=project,
        )
