from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import pfc_shaping.cli.audit_ch_lt_compute_runtime_manifest as cli_module
from pfc_shaping.cli.audit_ch_lt_compute_runtime_manifest import main
from pfc_shaping.pipeline.governed_release_cli_contract import ReleaseCliIdentityError
from pfc_shaping.validation.ch_lt_compute_runtime import canonical_json_bytes
from pfc_shaping.validation.ch_lt_compute_runtime_manifest import (
    DESIGN_FREEZE_RECEIPT_ROLE,
    DESIGN_FREEZE_RECEIPT_SCHEMA_VERSION,
    MONTE_CARLO_STUDY_ROLE,
    MONTE_CARLO_STUDY_SCHEMA_VERSION,
    RAW_BOUND_PAYLOAD_ROLES,
    RECEIPT_HASH_FIELDS,
    RECEIPT_PAYLOAD_ROLES,
    RECEIPT_SCHEMA_VERSION,
    RECEIPT_STATUS,
    SCENARIO_DESIGN_ROLE,
    SCENARIO_DESIGN_SCHEMA_VERSION,
    STATUS,
    ChLtComputeRuntimeManifestError,
    assess_compute_runtime_manifest,
    compute_execution_context_sha256,
    compute_scenario_design_core_sha256,
    load_compute_runtime_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "CH-LT-COMPUTE-RUNTIME-DRAFT-20260727.json"
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> bytes:
    payload = canonical_json_bytes(value) + b"\n"
    path.write_bytes(payload)
    return payload


def _receipt(
    role: str,
    *,
    contract_id: str,
    contract_sha256: str,
    workload_scope: str,
    execution_context_sha256: str,
    bound_payload_hashes: dict[str, str],
) -> dict[str, object]:
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "role": role,
        "compute_runtime_contract_id": contract_id,
        "compute_runtime_document_sha256": contract_sha256,
        "plan_id": _sha("plan"),
        "attempt_id": "attempt-001",
        "origin_id": "origin-001",
        "candidate_id": "candidate-001",
        "workload_scope": workload_scope,
        "execution_context_sha256": execution_context_sha256,
        "status": RECEIPT_STATUS[role],
        "bound_payload_hashes": bound_payload_hashes,
    }


def _fixture(
    tmp_path: Path,
    *,
    workload_scope: str = "SCENARIO_SCORING",
    backend: str = "GPU",
    positive_margin_half_width: float = 0.01,
) -> tuple[dict[str, object], bytes, dict[str, Path], dict[str, object], bytes]:
    contract_bytes = CONTRACT.read_bytes()
    contract = json.loads(contract_bytes)
    contract_sha256 = hashlib.sha256(contract_bytes).hexdigest()
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    evidence_paths: dict[str, Path] = {}
    receipt_hashes = {role: "0" * 64 for role in RECEIPT_HASH_FIELDS}
    frozen_weights: str
    if workload_scope == "SHAPING_INFERENCE_FROM_FROZEN_WEIGHTS":
        weights_path = evidence_root / "frozen_weights.bin"
        weights_payload = b"frozen-shaping-weights-v1\n"
        weights_path.write_bytes(weights_payload)
        evidence_paths["frozen_shaping_weights"] = weights_path
        frozen_weights = hashlib.sha256(weights_payload).hexdigest()
    else:
        frozen_weights = "NOT_APPLICABLE"
    monte_carlo_study = {
        "schema_version": MONTE_CARLO_STUDY_SCHEMA_VERSION,
        "compute_runtime_contract_id": contract["contract_id"],
        "compute_runtime_document_sha256": contract_sha256,
        "plan_id": _sha("plan"),
        "comparison_family_id": "comparison-family-001",
        "qualification_data": "ADMITTED_NON_HOLDOUT_ONLY",
        "estimator": "PAIRED_COMMON_RANDOM_NUMBER_CONTRAST",
        "common_random_numbers": True,
        "candidate_scenario_counts": [50, 100, 200],
        "selected_scenario_count": 200,
        "selected_chunk_count": 4,
        "selected_chunk_plan_sha256": _sha("chunks"),
        "shock_stream_inventory_sha256": _sha("shock streams"),
        "reference_scale_eur_mwh": 12.5,
        "positive_margin_eur_mwh": 1.0,
        "positive_margin_contrast_half_width_eur_mwh": positive_margin_half_width,
        "zero_margin_one_sided_half_width_eur_mwh": 0.01,
        "confidence_level": 0.95,
        "future_holdout_used": False,
        "frozen_before_holdout": True,
        "status": "PASS_FROZEN_DESIGN_SUPPORTED",
        "execution_authorized": False,
        "production_authorization": False,
        "promotion_gate": False,
    }
    monte_carlo_study_path = evidence_root / "monte_carlo_error_study.json"
    monte_carlo_study_payload = _write_json(monte_carlo_study_path, monte_carlo_study)
    evidence_paths[MONTE_CARLO_STUDY_ROLE] = monte_carlo_study_path
    monte_carlo_study_sha256 = hashlib.sha256(monte_carlo_study_payload).hexdigest()
    scenario_design: dict[str, object] = {
        "schema_version": SCENARIO_DESIGN_SCHEMA_VERSION,
        "compute_runtime_contract_id": contract["contract_id"],
        "compute_runtime_document_sha256": contract_sha256,
        "plan_id": _sha("plan"),
        "comparison_family_id": "comparison-family-001",
        "scenario_count": 200,
        "chunk_count": 4,
        "chunk_plan_sha256": _sha("chunks"),
        "shock_stream_inventory_sha256": _sha("shock streams"),
        "monte_carlo_error_study_sha256": monte_carlo_study_sha256,
        "design_freeze_receipt_sha256": "0" * 64,
        "frozen_positive_reference_scale": 12.5,
        "qualification_data": "ADMITTED_NON_HOLDOUT_ONLY",
        "frozen_before_holdout": True,
        "execution_authorized": False,
        "production_authorization": False,
        "promotion_gate": False,
    }
    design_freeze_receipt = {
        "schema_version": DESIGN_FREEZE_RECEIPT_SCHEMA_VERSION,
        "compute_runtime_contract_id": contract["contract_id"],
        "compute_runtime_document_sha256": contract_sha256,
        "plan_id": _sha("plan"),
        "comparison_family_id": "comparison-family-001",
        "scenario_design_core_sha256": compute_scenario_design_core_sha256(
            scenario_design
        ),
        "monte_carlo_error_study_sha256": monte_carlo_study_sha256,
        "status": "LOCAL_FREEZE_RECEIPT_NOT_TRUSTED_TIME",
        "frozen_before_holdout": True,
        "external_signature_verified": False,
        "trusted_time_verified": False,
        "ledger_sequence_verified": False,
        "execution_authorized": False,
        "production_authorization": False,
        "promotion_gate": False,
    }
    design_freeze_path = evidence_root / "design_freeze_receipt.json"
    design_freeze_payload = _write_json(design_freeze_path, design_freeze_receipt)
    evidence_paths[DESIGN_FREEZE_RECEIPT_ROLE] = design_freeze_path
    scenario_design["design_freeze_receipt_sha256"] = hashlib.sha256(
        design_freeze_payload
    ).hexdigest()
    scenario_design_path = evidence_root / "probabilistic_scenario_mc_design.json"
    scenario_design_payload = _write_json(scenario_design_path, scenario_design)
    evidence_paths[SCENARIO_DESIGN_ROLE] = scenario_design_path
    scenario_design_sha256 = hashlib.sha256(scenario_design_payload).hexdigest()
    manifest: dict[str, object] = {
        "schema_version": "ch_lt_compute_runtime_manifest.v1",
        "compute_runtime_contract_id": contract["contract_id"],
        "compute_runtime_document_sha256": contract_sha256,
        **{
            field: receipt_hashes[role]
            for role, field in RECEIPT_HASH_FIELDS.items()
        },
        "frozen_shaping_weights_sha256": frozen_weights,
        "plan_id": _sha("plan"),
        "attempt_id": "attempt-001",
        "origin_id": "origin-001",
        "candidate_id": "candidate-001",
        "input_hashes": {
            "origin_target_mask_and_inner_fold_inventory": _sha("origin inventory"),
            "input_manifests_and_available_at_times": _sha("input manifests"),
            "candidate_baseline_and_hyperparameter_grid": _sha("candidates"),
            "training_cache_manifest": _sha("cache"),
            SCENARIO_DESIGN_ROLE: scenario_design_sha256,
        },
        "code_and_wheel_hashes": {
            "governed_lt_wheel": _sha("wheel"),
            "dependency_lock": _sha("lock"),
            "evaluation_code": _sha("evaluation code"),
            "runtime_environment": _sha("runtime environment"),
            "numpy_wheel": _sha("numpy wheel"),
            "torch_wheel": _sha("torch wheel"),
        },
        "lock_hash": _sha("lock"),
        "python_numpy_torch_versions": {
            "python": "3.11.14",
            "numpy": "2.0.2",
            "torch": "2.11.0+cu128",
        },
        "cuda_build_runtime_cudnn_driver": {
            "cuda_build": "12.8",
            "cuda_runtime": "12.8",
            "cudnn": "91900",
            "nvidia_driver": "595.95",
        },
        "os_gpu_uuid_model_compute_capability": {
            "os": "Windows-11",
            "gpu_uuid": "GPU-test-uuid",
            "gpu_model": "RTX 3000 Ada Laptop",
            "compute_capability": "8.9",
        },
        "dtype_and_determinism_flags": {
            "dtype": "FLOAT64",
            "amp": False,
            "tf32": False,
            "torch_compile": False,
            "cublas_workspace_config": ":4096:8",
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "matmul_tf32": False,
            "cudnn_tf32": False,
            "dataloader_workers": 0,
            "clean_process_repeats": 3,
            "fresh_process_bootstrap_required": True,
            "cublas_env_set_before_torch_import_and_cuda_initialization": True,
            "no_cuda_capability_query_before_bootstrap": True,
            "flags_attested_after_cuda_initialization": True,
            "abort_on_nondeterministic_warning_or_error": True,
        },
        "seed_tree_and_shock_hashes": {
            "generator": "NUMPY_PCG64_ON_CPU_ONLY",
            "bit_generator": "PCG64",
            "seed_preimage_encoding": "UTF8_U32BE_LENGTH_PREFIXED_FIELDS_V1",
            "seed_digest_to_integer": "SHA256_FIRST_128_BITS_BIG_ENDIAN_UNSIGNED",
            "shock_dtype": "LITTLE_ENDIAN_FLOAT64",
            "shock_array_layout": "C_CONTIGUOUS",
            "shock_semantic_hash_scheme": "SHA256_DTYPE_ENDIAN_SHAPE_AND_C_ORDER_BYTES_V1",
            "shock_stream_inventory_sha256": _sha("shock streams"),
            "shock_bytes_sha256": _sha("shock bytes"),
            "cuda_rng_used": False,
            "candidate_id_in_standardized_shock_seed": False,
            "common_random_numbers": True,
        },
        "row_scenario_and_chunk_order": {
            "row_ids_sha256": _sha("rows"),
            "scenario_ids_sha256": _sha("scenarios"),
            "target_mask_sha256": _sha("mask"),
            "weights_sha256": _sha("weights"),
            "chunk_plan_sha256": _sha("chunks"),
            "row_count": 100,
            "scenario_count": 200,
            "chunk_count": 4,
            "serialization": "DTYPE_ENDIAN_SHAPE_AND_C_ORDER_BYTES_V1",
        },
        "backend_and_fallback_reason": {
            "workload_scope": workload_scope,
            "backend": backend,
            "fallback_used": False,
            "fallback_reason": "NOT_APPLICABLE",
            "gpu_fit_attempted": False,
            "mixed_backend": False,
            "rng_started_before_fallback": False,
            "output_started_before_fallback": False,
        },
    }
    execution_context_sha256 = compute_execution_context_sha256(manifest)
    bound_payload_hashes_by_receipt: dict[str, dict[str, str]] = {}
    for receipt_role, payload_roles in RECEIPT_PAYLOAD_ROLES.items():
        bound_payload_hashes: dict[str, str] = {}
        for payload_role in payload_roles:
            if payload_role in {"prediction_seal_receipt", "scenario_seal_receipt"}:
                continue
            payload_path = evidence_root / f"{payload_role}.bin"
            payload_bytes = f"bound-payload:{payload_role}\n".encode("utf-8")
            payload_path.write_bytes(payload_bytes)
            evidence_paths[payload_role] = payload_path
            bound_payload_hashes[payload_role] = hashlib.sha256(payload_bytes).hexdigest()
        bound_payload_hashes_by_receipt[receipt_role] = bound_payload_hashes
    for role in RECEIPT_HASH_FIELDS:
        if role == "truth_open_receipt":
            continue
        receipt = _receipt(
            role,
            contract_id=contract["contract_id"],
            contract_sha256=contract_sha256,
            workload_scope=workload_scope,
            execution_context_sha256=execution_context_sha256,
            bound_payload_hashes=bound_payload_hashes_by_receipt[role],
        )
        path = evidence_root / f"{role}.json"
        payload = _write_json(path, receipt)
        evidence_paths[role] = path
        receipt_hashes[role] = hashlib.sha256(payload).hexdigest()
    truth_payload_hashes = bound_payload_hashes_by_receipt["truth_open_receipt"]
    truth_payload_hashes.update(
        {
            "prediction_seal_receipt": receipt_hashes["prediction_seal"],
            "scenario_seal_receipt": receipt_hashes["scenario_seal"],
        }
    )
    truth_receipt = _receipt(
        "truth_open_receipt",
        contract_id=contract["contract_id"],
        contract_sha256=contract_sha256,
        workload_scope=workload_scope,
        execution_context_sha256=execution_context_sha256,
        bound_payload_hashes=truth_payload_hashes,
    )
    truth_path = evidence_root / "truth_open_receipt.json"
    truth_payload = _write_json(truth_path, truth_receipt)
    evidence_paths["truth_open_receipt"] = truth_path
    receipt_hashes["truth_open_receipt"] = hashlib.sha256(truth_payload).hexdigest()
    for role, field in RECEIPT_HASH_FIELDS.items():
        manifest[field] = receipt_hashes[role]
    manifest_bytes = canonical_json_bytes(manifest) + b"\n"
    return manifest, manifest_bytes, evidence_paths, contract, contract_bytes


def _assess(
    manifest: dict[str, object],
    manifest_bytes: bytes,
    evidence_paths: dict[str, Path],
    contract: dict[str, object],
    contract_bytes: bytes,
) -> object:
    return assess_compute_runtime_manifest(
        manifest,
        manifest_bytes=manifest_bytes,
        compute_contract=contract,
        compute_contract_bytes=contract_bytes,
        evidence_root=next(iter(evidence_paths.values())).parent,
        evidence_paths=evidence_paths,
    )


def test_runtime_manifest_is_closed_hash_bound_and_non_authoritative(tmp_path: Path) -> None:
    manifest, payload, evidence, contract, contract_payload = _fixture(tmp_path)

    result = _assess(manifest, payload, evidence, contract, contract_payload).to_dict()

    assert result["status"] == STATUS
    assert result["structure_valid"] is True
    assert result["receipt_and_bound_payload_bytes_verified"] is True
    assert result["execution_authorized"] is False
    assert result["production_authorization"] is False
    assert result["promotion_gate"] is False
    assert len(result["validator_implementation_sha256"]) == 64
    assert set(result["verified_evidence_bytes_sha256"]) == (
        set(RECEIPT_HASH_FIELDS)
        | set(RAW_BOUND_PAYLOAD_ROLES)
        | {SCENARIO_DESIGN_ROLE, MONTE_CARLO_STUDY_ROLE, DESIGN_FREEZE_RECEIPT_ROLE}
    )


def test_runtime_manifest_rejects_unknown_nested_role(tmp_path: Path) -> None:
    manifest, _, evidence, contract, contract_payload = _fixture(tmp_path)
    assert isinstance(manifest["input_hashes"], dict)
    manifest["input_hashes"]["unknown"] = _sha("unknown")
    payload = canonical_json_bytes(manifest)

    with pytest.raises(ChLtComputeRuntimeManifestError, match="input hashes keys"):
        _assess(manifest, payload, evidence, contract, contract_payload)


def test_runtime_manifest_rejects_backend_fallback_after_rng(tmp_path: Path) -> None:
    manifest, _, evidence, contract, contract_payload = _fixture(
        tmp_path,
        backend="CPU",
    )
    backend = manifest["backend_and_fallback_reason"]
    assert isinstance(backend, dict)
    backend.update(
        {
            "fallback_used": True,
            "fallback_reason": "GPU_OOM",
            "rng_started_before_fallback": True,
        }
    )
    payload = canonical_json_bytes(manifest)

    with pytest.raises(ChLtComputeRuntimeManifestError, match="after RNG or output"):
        _assess(manifest, payload, evidence, contract, contract_payload)


def test_receipts_cannot_be_reused_after_execution_context_mutation(tmp_path: Path) -> None:
    manifest, _, evidence, contract, contract_payload = _fixture(tmp_path)
    code_hashes = manifest["code_and_wheel_hashes"]
    backend = manifest["backend_and_fallback_reason"]
    assert isinstance(code_hashes, dict)
    assert isinstance(backend, dict)
    code_hashes["governed_lt_wheel"] = _sha("different wheel")
    backend["backend"] = "CPU"
    payload = canonical_json_bytes(manifest)

    with pytest.raises(ChLtComputeRuntimeManifestError, match="execution context"):
        _assess(manifest, payload, evidence, contract, contract_payload)


def test_runtime_manifest_rejects_gpu_fallback_for_cpu_oracle_scope(tmp_path: Path) -> None:
    manifest, _, evidence, contract, contract_payload = _fixture(
        tmp_path,
        workload_scope="MONTHLY_SOLVER",
        backend="CPU",
    )
    backend = manifest["backend_and_fallback_reason"]
    assert isinstance(backend, dict)
    backend.update({"fallback_used": True, "fallback_reason": "GPU_OOM"})
    payload = canonical_json_bytes(manifest)

    with pytest.raises(ChLtComputeRuntimeManifestError, match="fallback relation"):
        _assess(manifest, payload, evidence, contract, contract_payload)


def test_frozen_weights_are_required_only_for_frozen_shaping_inference(
    tmp_path: Path,
) -> None:
    manifest, payload, evidence, contract, contract_payload = _fixture(
        tmp_path,
        workload_scope="SHAPING_INFERENCE_FROM_FROZEN_WEIGHTS",
    )

    result = _assess(manifest, payload, evidence, contract, contract_payload)

    assert "frozen_shaping_weights" in result.verified_evidence_bytes_sha256


def test_runtime_manifest_rejects_scenario_count_changed_after_design_freeze(
    tmp_path: Path,
) -> None:
    manifest, _, evidence, contract, contract_payload = _fixture(tmp_path)
    order = manifest["row_scenario_and_chunk_order"]
    assert isinstance(order, dict)
    order["scenario_count"] = 100
    payload = canonical_json_bytes(manifest)

    with pytest.raises(ChLtComputeRuntimeManifestError, match="scenario_count"):
        _assess(manifest, payload, evidence, contract, contract_payload)


def test_runtime_manifest_rejects_failed_prefreeze_monte_carlo_precision(
    tmp_path: Path,
) -> None:
    manifest, payload, evidence, contract, contract_payload = _fixture(
        tmp_path,
        positive_margin_half_width=0.02,
    )

    with pytest.raises(ChLtComputeRuntimeManifestError, match="precision fails"):
        _assess(manifest, payload, evidence, contract, contract_payload)


def test_runtime_manifest_rejects_tampered_receipt_bytes(tmp_path: Path) -> None:
    manifest, payload, evidence, contract, contract_payload = _fixture(tmp_path)
    evidence["parity_qualification_report"].write_bytes(b"tampered\n")

    with pytest.raises(ChLtComputeRuntimeManifestError, match="evidence SHA-256 mismatch"):
        _assess(manifest, payload, evidence, contract, contract_payload)


def test_runtime_manifest_rejects_tampered_bound_payload_bytes(tmp_path: Path) -> None:
    manifest, payload, evidence, contract, contract_payload = _fixture(tmp_path)
    evidence["predictions"].write_bytes(b"different prediction bytes\n")

    with pytest.raises(ChLtComputeRuntimeManifestError, match="bound payload SHA-256"):
        _assess(manifest, payload, evidence, contract, contract_payload)


def test_runtime_manifest_rejects_physical_path_reuse_across_roles(tmp_path: Path) -> None:
    manifest, payload, evidence, contract, contract_payload = _fixture(tmp_path)
    evidence["scenarios"] = evidence["predictions"]

    with pytest.raises(ChLtComputeRuntimeManifestError, match="physically role-distinct"):
        _assess(manifest, payload, evidence, contract, contract_payload)


def test_truth_open_receipt_must_bind_exact_prediction_and_scenario_seals(
    tmp_path: Path,
) -> None:
    manifest, _, evidence, contract, contract_payload = _fixture(tmp_path)
    receipt = json.loads(evidence["truth_open_receipt"].read_bytes())
    receipt["bound_payload_hashes"]["prediction_seal_receipt"] = _sha("wrong seal")
    receipt_payload = _write_json(evidence["truth_open_receipt"], receipt)
    manifest["truth_open_receipt_sha256"] = hashlib.sha256(receipt_payload).hexdigest()
    payload = canonical_json_bytes(manifest)

    with pytest.raises(ChLtComputeRuntimeManifestError, match="prediction seal binding"):
        _assess(manifest, payload, evidence, contract, contract_payload)


def test_manifest_bytes_must_rebind_to_assessed_mapping(tmp_path: Path) -> None:
    manifest, _, evidence, contract, contract_payload = _fixture(tmp_path)

    with pytest.raises(ChLtComputeRuntimeManifestError, match="do not bind"):
        _assess(manifest, b"{}", evidence, contract, contract_payload)


def test_runtime_manifest_loader_and_cli_fail_closed(
    tmp_path: Path,
    capsys: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module, "assert_installed_runtime_sealed", lambda: None)
    manifest, payload, evidence, _, _ = _fixture(tmp_path)
    manifest_path = tmp_path / "runtime_manifest.json"
    manifest_path.write_bytes(payload)
    manifest_hash = hashlib.sha256(payload).hexdigest()

    loaded, loaded_bytes = load_compute_runtime_manifest(
        manifest_path,
        expected_sha256=manifest_hash,
    )
    assert loaded == manifest
    assert loaded_bytes == payload
    args = [
        "--contract",
        str(CONTRACT),
        "--expected-contract-sha256",
        hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "--manifest",
        str(manifest_path),
        "--expected-manifest-sha256",
        manifest_hash,
        "--evidence-root",
        str(next(iter(evidence.values())).parent),
    ]
    for role, path in evidence.items():
        args.extend(["--evidence", f"{role}={path}"])

    code = main(args)

    assert code == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    result = json.loads(captured.out)
    assert result["status"] == STATUS
    assert result["execution_authorized"] is False
    assert result["runtime_identity"]["contract_path"] == str(CONTRACT)
    assert len(result["audit_operation_id"]) == 64


def test_manifest_cli_rejects_unsealed_runtime_before_argument_parsing(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject() -> None:
        raise ReleaseCliIdentityError("runtime is not isolated")

    monkeypatch.setattr(cli_module, "assert_installed_runtime_sealed", reject)

    assert main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    failure = json.loads(captured.err)
    assert failure["status"] == "INVALID_GOVERNED_LT_RUNTIME"
    assert failure["command_id"] == "audit_ch_lt_compute_runtime_manifest"
    assert failure["production_authorization"] is False
