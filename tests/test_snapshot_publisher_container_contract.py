from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from pfc_shaping.publisher_package_contract import PUBLISHER_RUNTIME_PYTHON_FILES
from scripts import check_snapshot_publisher_container_context as container_contract
from scripts import stage_snapshot_publisher_container_wheelhouse as wheel_staging

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "deploy" / "publisher" / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
WORKFLOW = ROOT / ".github" / "workflows" / "publisher-runtime-v6.yml"
OPERATIONS_CONTRACT = ROOT / "deploy" / "publisher" / "operations-contract.json"


def test_dockerfile_has_one_fail_closed_windows_runtime_path() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "ARG PFC_PUBLISHER_BASE_IMAGE\n" in dockerfile
    assert "ARG PFC_PUBLISHER_BASE_IMAGE=" not in dockerfile
    assert dockerfile.count("FROM ${PFC_PUBLISHER_BASE_IMAGE}") == 2
    assert "COPY --from=wheelhouse" not in dockerfile
    assert "COPY build\\publisher-container-wheelhouse C:\\wheelhouse" in dockerfile
    assert "pip install" not in dockerfile
    assert "Invoke-WebRequest" not in dockerfile
    assert "curl " not in dockerfile
    assert "ADD http" not in dockerfile
    assert "USER ContainerUser" in dockerfile
    assert "HEALTHCHECK NONE" in dockerfile
    assert "CMD []" in dockerfile
    assert "VOLUME" not in dockerfile
    assert "TEMP=\"C:\\pfc\\scratch\"" in dockerfile
    assert "TMP=\"C:\\pfc\\scratch\"" in dockerfile
    assert dockerfile.count("if ($LASTEXITCODE -ne 0) { throw '") >= 5
    assert "write unexpectedly admitted" in dockerfile
    assert "fmv-lt-snapshot-publisher.pyz prepare --help" in dockerfile
    assert "fmv-lt-snapshot-publisher.pyz cas --help" in dockerfile
    assert "fmv-lt-snapshot-publisher.pyz finalize --help" in dockerfile
    assert 'org.fmv.pfc.production-authorization="false"' in dockerfile
    assert dockerfile.rstrip().endswith(
        'ENTRYPOINT ["python.exe", "-I", "-S", "-B", '
        '"C:\\\\pfc\\\\runtime\\\\fmv-lt-snapshot-publisher.pyz"]'
    )


def test_docker_context_is_positive_listed_and_excludes_sensitive_scopes() -> None:
    dockerignore = DOCKERIGNORE.read_text(encoding="utf-8").splitlines()

    assert dockerignore[0] == "**"
    assert "!uv.lock" in dockerignore
    assert "!build/publisher-container-wheelhouse/*.whl" not in dockerignore
    allowed_wheels = {
        line.removeprefix("!build/publisher-container-wheelhouse/")
        for line in dockerignore
        if line.startswith("!build/publisher-container-wheelhouse/")
        and line.endswith(".whl")
    }
    assert allowed_wheels == {
        "cffi-2.1.0-cp311-cp311-win_amd64.whl",
        "cryptography-47.0.0-cp311-abi3-win_amd64.whl",
        "numpy-2.0.2-cp311-cp311-win_amd64.whl",
        "pandas-2.3.3-cp311-cp311-win_amd64.whl",
        "pyarrow-21.0.0-cp311-cp311-win_amd64.whl",
        "pycparser-3.0-py3-none-any.whl",
        "python_dateutil-2.9.0.post0-py2.py3-none-any.whl",
        "pytz-2026.2-py2.py3-none-any.whl",
        "pyyaml-6.0.3-cp311-cp311-win_amd64.whl",
        "six-1.17.0-py2.py3-none-any.whl",
        "tzdata-2026.3-py2.py3-none-any.whl",
    }
    assert "!pfc_shaping/**/*.py" not in dockerignore
    assert "!pfc_shaping/publisher_runtime_admission.py" in dockerignore
    assert "!pfc_shaping/data/snapshot_publisher.py" in dockerignore
    assert "!pfc_shaping/pipeline/strict_structured_data.py" in dockerignore
    assert "!scripts/check_snapshot_publisher_container_context.py" in dockerignore
    allowed_files = {line.removeprefix("!") for line in dockerignore if line.startswith("!")}
    assert set(PUBLISHER_RUNTIME_PYTHON_FILES).issubset(allowed_files)
    assert all("pfc_shaping/ct/" not in line for line in dockerignore if line.startswith("!"))
    assert all("powerbi" not in line.lower() for line in dockerignore if line.startswith("!"))
    assert all(".env" not in line.lower() for line in dockerignore if line.startswith("!"))


def test_ci_is_non_promoting_pinned_and_container_build_is_manually_gated() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "permissions:\n  contents: read" in workflow
    assert "persist-credentials: false" in workflow
    assert "actions/checkout@11d5960a326750d5838078e36cf38b85af677262" in workflow
    assert "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065" in workflow
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in workflow
    assert "github.ref == 'refs/heads/main'" in workflow
    assert "runs-on: [self-hosted, Windows, X64, pfc-publisher-windows-docker]" in workflow
    assert "environment: publisher-container-proof" in workflow
    assert "docker build --network none --pull=false" in workflow
    assert "--isolation process" in workflow
    assert "--build-context" not in workflow
    assert "Stage only approved wheel archives" in workflow
    assert "scripts/stage_snapshot_publisher_container_wheelhouse.py" in workflow
    assert "-m scripts.stage_snapshot_publisher_container_wheelhouse" in workflow
    assert "docker run --rm --network none --read-only" in workflow
    assert "docker push" not in workflow
    assert "--iidfile" in workflow
    assert "PFC_CONTAINER_PROOF_IMAGE_ID" in workflow
    assert "PFC_CONTAINER_PROOF_TAG" not in workflow
    assert "Remove-Item -LiteralPath $staging -Recurse" not in workflow
    assert "publisher wheel staging contains a linked or nested entry" in workflow
    assert "publisher base image must not declare inherited volumes" in workflow
    assert "final image declares an implicit volume" in workflow
    assert "final image label differs from the contract" in workflow
    assert "org.opencontainers.image.base.name" in workflow
    assert "unowned publisher build workspace residue exists" in workflow
    assert "[IO.Directory]::CreateDirectory($buildRoot)" in workflow
    assert "PFC_PUBLISHER_BUILD_PYTHON_SHA256" in workflow
    assert "production_authorization = $false" in workflow
    assert "registry_image_digest = 'MISSING_NOT_PUSHED'" in workflow


def test_operations_contract_governs_observability_and_forward_only_rollback() -> None:
    contract = json.loads(OPERATIONS_CONTRACT.read_bytes())

    assert contract["workload_model"] == "ONE_SHOT_GOVERNED_JOB"
    assert contract["runtime"]["root_filesystem_requirement"] == "DOCKER_RUN_READ_ONLY"
    assert contract["runtime"]["isolation_mode"] == (
        "PROCESS_EXPLICIT_HOST_BASE_OSVERSION_COMPATIBILITY_REQUIRED"
    )
    assert contract["runtime"]["network_requirement"] == (
        "DENY_BY_DEFAULT_PHASE_ENDPOINT_ALLOWLIST_ONLY"
    )
    assert contract["runtime"]["scratch_mount"] == (
        "EXPLICIT_RW_NAMED_OR_BIND_MOUNT_NO_ANONYMOUS_FALLBACK"
    )
    assert contract["runtime"]["data_mount"] == (
        "PREPARE_AND_FINALIZE_EXPLICIT_RW_FMV_DATA_ROOT"
    )
    assert contract["runtime"]["evidence_mount"] == (
        "CAS_AND_FINALIZE_EXPLICIT_RW_RECEIPT_OBSERVATION_ROOT"
    )
    assert contract["runtime"]["secrets"] == (
        "READ_ONLY_DIRECTORY_MOUNTS_PRESENT_AT_START_PATHS_DEFERRED_POST_ADMISSION"
    )
    assert "pfc_publisher_admission_duration_seconds" in contract["observability"][
        "required_metrics"
    ]
    assert "operation_id" in contract["observability"][
        "log_only_high_cardinality_fields"
    ]
    assert contract["rollback"]["automatic_rollback"] is False
    assert contract["rollback"]["data_head"] == "NEVER_REWIND_EXTERNAL_CAS_HEAD"
    assert contract["rollback"]["data_correction"] == (
        "PUBLISH_NEW_SIGNED_COMPENSATING_GENERATION"
    )
    assert contract["rollback"]["compatibility_gate"] == (
        "SIGNED_TARGET_DIGEST_PROTOCOL_SCHEMA_TRUST_REGISTRY_DOMAIN_AND_FRESH_CAS_HEAD_BINDING_REQUIRED"
    )
    assert contract["lifecycle"]["restart_policy"] == "NEVER"
    assert contract["lifecycle"]["retryable_exit_codes"] == [41, 52]
    assert contract["lifecycle"]["maximum_retry_count"] == 5
    assert contract["lifecycle"]["maximum_total_attempts"] == 6
    assert contract["production_authorization"] is False


def test_operations_contract_rejects_ambiguous_or_extended_policy() -> None:
    contract = json.loads(OPERATIONS_CONTRACT.read_bytes())
    contract["runtime"]["quietly_weaken_read_only"] = True
    with pytest.raises(ValueError, match="runtime operations policy"):
        container_contract._strict_operations_contract(json.dumps(contract).encode())

    contract = json.loads(OPERATIONS_CONTRACT.read_bytes())
    contract["observability"]["required_metrics"] = {"unexpected": "mapping"}
    with pytest.raises(ValueError, match="observability policy"):
        container_contract._strict_operations_contract(json.dumps(contract).encode())

    contract = json.loads(OPERATIONS_CONTRACT.read_bytes())
    contract["observability"]["required_alerts"] = ["DUMMY_ALERT"]
    with pytest.raises(ValueError, match="contract bytes differ"):
        container_contract._strict_operations_contract(json.dumps(contract).encode())

    contract = json.loads(OPERATIONS_CONTRACT.read_bytes())
    contract["lifecycle"]["retryable_exit_codes"] = [40, 41, 52]
    with pytest.raises(ValueError, match="lifecycle policy"):
        container_contract._strict_operations_contract(json.dumps(contract).encode())

    duplicate = OPERATIONS_CONTRACT.read_bytes().replace(
        b'{\n  "schema_version":',
        b'{\n  "schema_version": "duplicate",\n  "schema_version":',
        1,
    )
    with pytest.raises(ValueError, match="duplicate key"):
        container_contract._strict_operations_contract(duplicate)


@pytest.mark.parametrize(
    "reference",
    [
        "python:3.11-windowsservercore-ltsc2022",
        "python:latest@sha256:" + "a" * 64,
        "python@sha256:" + "A" * 64,
        "python@sha512:" + "a" * 64,
        "sha256:" + "a" * 64,
        "python@sha256:" + "a" * 64,
        "registry.fmv.local//python@sha256:" + "a" * 64,
        "registry.fmv.local/python:@sha256:" + "a" * 64,
        " python@sha256:" + "a" * 64,
        "python@sha256:" + "a" * 64 + " ",
    ],
)
def test_base_image_rejects_mutable_or_noncanonical_references(reference: str) -> None:
    with pytest.raises(ValueError, match="exact lowercase sha256 reference"):
        container_contract._validated_base_image(reference)


def test_base_image_accepts_registry_repository_and_sha256() -> None:
    digest = "a" * 64
    assert (
        container_contract._validated_base_image(
            f"registry.fmv.local/pfc/python:3.11.14-windows@sha256:{digest}"
        )
        == digest
    )


def test_context_manifest_binds_artifact_closure_base_and_dockerfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = tmp_path / "context"
    context.mkdir()
    artifact = context / container_contract.ARTIFACT_NAME
    artifact_buffer = io.BytesIO()
    with zipfile.ZipFile(
        artifact_buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("__main__.py", b"raise SystemExit(0)\n")
    artifact_payload = artifact_buffer.getvalue()
    artifact.write_bytes(artifact_payload)
    dependency_root = context / container_contract.DEPENDENCY_ROOT_NAME
    dependency_root.mkdir()
    (dependency_root / "payload.py").write_bytes(b"payload")
    receipt = {
        "production_authorization": False,
        "target": {"platform_tag": "win_amd64"},
        "source_lock": {"path": "uv.lock", "sha256": "e" * 64},
        "tree_sha256": "b" * 64,
        "file_count": 1,
        "wheels": [
            {
                "name": "Demo_Pkg",
                "version": "1.2.3",
                "filename": "demo_pkg-1.2.3-py3-none-any.whl",
                "sha256": "f" * 64,
                "size": 123,
            }
        ],
    }
    receipt_payload = _canonical_json(receipt)
    receipt_path = context / container_contract.DEPENDENCY_RECEIPT_NAME
    receipt_path.write_bytes(receipt_payload)
    operations_path = context / container_contract.OPERATIONS_CONTRACT_NAME
    operations_path.write_bytes(OPERATIONS_CONTRACT.read_bytes())
    staging = {
        "schema_version": "fmv_lt_snapshot_publisher_wheelhouse_staging.v1",
        "status": "PASS",
        "source_lock": receipt["source_lock"],
        "wheel_count": 1,
        "wheels": [
            {
                "name": "Demo_Pkg",
                "version": "1.2.3",
                "filename": "demo_pkg-1.2.3-py3-none-any.whl",
                "sha256": "f" * 64,
                "size": 123,
            }
        ],
        "production_authorization": False,
    }
    (context / container_contract.WHEELHOUSE_STAGING_MANIFEST_NAME).write_bytes(
        _canonical_json(staging)
    )
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_bytes(b"FROM exact\n")

    monkeypatch.setattr(
        container_contract,
        "audit_publisher_zipapp",
        lambda _path: {
            "artifact_sha256": hashlib.sha256(artifact_payload).hexdigest(),
            "dependency_closure_receipt_sha256": hashlib.sha256(
                receipt_payload
            ).hexdigest(),
            "dependency_tree_sha256": "b" * 64,
            "runtime_contract_sha256": "c" * 64,
            "runtime_executable_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(
        container_contract,
        "dependency_tree_inventory",
        lambda _path: {"tree_sha256": "b" * 64, "file_count": 1},
    )
    monkeypatch.setattr(
        container_contract,
        "dependency_distribution_inventory",
        lambda _path: {"demo-pkg": "1.2.3"},
    )
    monkeypatch.setattr(
        container_contract,
        "_artifact_runtime_contract",
        lambda _path, *, expected_sha256: {
            "python": {"platform": "win32", "pointer_bits": 64, "abi": "cp311"},
            "production_authorization": False,
        },
    )

    digest = "a" * 64
    output = context / container_contract.CONTAINER_MANIFEST_NAME
    result = container_contract.build_container_context_manifest(
        context,
        base_image=f"registry.fmv.local/python@sha256:{digest}",
        output=output,
        dockerfile=dockerfile,
    )

    assert result["base_image_digest"] == digest
    assert result["artifact_sha256"] == hashlib.sha256(artifact_payload).hexdigest()
    assert result["dependency_tree_sha256"] == "b" * 64
    assert result["runtime_user"] == "ContainerUser"
    assert result["operations_contract_sha256"] == hashlib.sha256(
        OPERATIONS_CONTRACT.read_bytes()
    ).hexdigest()
    assert len(result["wheelhouse_staging_manifest_sha256"]) == 64
    assert result["publisher_bundle_private_key_block_scan_status"] == (
        "PASS_NO_SYNTACTIC_PEM_OR_OPENSSH_PRIVATE_KEY_BLOCKS"
    )
    assert result["base_image_secret_scan_status"] == "MISSING_REQUIRED_EXTERNAL"
    assert result["production_authorization"] is False
    assert output.read_bytes() == _canonical_json(result)


def test_context_manifest_rejects_extra_or_secret_like_entry(tmp_path: Path) -> None:
    context = tmp_path / "context"
    context.mkdir()
    (context / container_contract.ARTIFACT_NAME).write_bytes(b"artifact")
    (context / container_contract.DEPENDENCY_ROOT_NAME).mkdir()
    (context / container_contract.DEPENDENCY_RECEIPT_NAME).write_bytes(b"{}")
    (context / container_contract.OPERATIONS_CONTRACT_NAME).write_bytes(
        OPERATIONS_CONTRACT.read_bytes()
    )
    (context / container_contract.WHEELHOUSE_STAGING_MANIFEST_NAME).write_bytes(b"{}")
    (context / "request-signing-private-key.pem").write_bytes(b"secret")

    with pytest.raises(ValueError, match="inventory is not exact"):
        container_contract.build_container_context_manifest(
            context,
            base_image="registry.fmv.local/python@sha256:" + "a" * 64,
            output=context / container_contract.CONTAINER_MANIFEST_NAME,
        )


def test_private_key_block_scan_rejects_pem_payload(tmp_path: Path) -> None:
    payload = tmp_path / "apparently-benign.bin"
    encoded = base64.b64encode(b"secret-private-key-material" * 4)
    payload.write_bytes(
        b"prefix\n-----BEGIN PRIVATE KEY-----\n"
        + encoded
        + b"\n-----END PRIVATE KEY-----\n"
    )

    with pytest.raises(ValueError, match="private-key block"):
        container_contract._reject_private_key_blocks([payload])


def test_private_key_block_scan_decompresses_zipapp_members(tmp_path: Path) -> None:
    artifact = tmp_path / "publisher.pyz"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        encoded = base64.b64encode(b"secret-private-key-material" * 4)
        archive.writestr(
            "pfc_shaping/benign_name.py",
            b"-----BEGIN PRIVATE KEY-----\n"
            + encoded
            + b"\n-----END PRIVATE KEY-----\n",
        )
    payload = buffer.getvalue()
    assert b"-----BEGIN PRIVATE KEY-----" not in payload
    artifact.write_bytes(payload)

    with pytest.raises(ValueError, match="zipapp contains a private-key block"):
        container_contract._reject_zipapp_private_key_blocks(
            artifact,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def test_wheel_stager_rejects_extras_and_copies_only_stable_lock_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    wheel = source / "demo-1.0-py3-none-any.whl"
    wheel.write_bytes(b"exact-wheel")
    lock = tmp_path / "uv.lock"
    lock.write_bytes(b"lock")
    runtime = tmp_path / "runtime.json"
    runtime.write_bytes(b"{}")
    selected = [
        {
            "name": "demo",
            "version": "1.0",
            "filename": wheel.name,
            "sha256": hashlib.sha256(b"exact-wheel").hexdigest(),
            "size": len(b"exact-wheel"),
        }
    ]
    monkeypatch.setattr(wheel_staging, "_assert_build_runtime", lambda: None)
    monkeypatch.setattr(wheel_staging, "_strict_json_mapping", lambda _payload: {})
    monkeypatch.setattr(wheel_staging, "_expected_distributions", lambda _value: {})
    monkeypatch.setattr(
        wheel_staging,
        "_select_locked_wheels",
        lambda **_kwargs: selected,
    )

    output = tmp_path / "staged"
    manifest = tmp_path / wheel_staging.STAGING_MANIFEST_NAME
    result = wheel_staging.stage_container_wheelhouse(
        source,
        output,
        manifest,
        lock_path=lock,
        runtime_contract_path=runtime,
    )

    assert result["wheel_count"] == 1
    assert (output / wheel.name).read_bytes() == b"exact-wheel"
    assert manifest.read_bytes() == _canonical_json(result)
    assert (
        wheel_staging.audit_staged_container_wheelhouse(
            output,
            manifest,
            lock_path=lock,
            runtime_contract_path=runtime,
        )
        == result
    )

    (output / wheel.name).write_bytes(b"torn-wheel!")
    with pytest.raises(ValueError, match="differs from uv.lock"):
        wheel_staging.audit_staged_container_wheelhouse(
            output,
            manifest,
            lock_path=lock,
            runtime_contract_path=runtime,
        )

    extra_parent = tmp_path / "extra"
    extra_parent.mkdir()
    extra_source = extra_parent / "source"
    extra_source.mkdir()
    (extra_source / wheel.name).write_bytes(b"exact-wheel")
    (extra_source / "secret.pem.whl").write_bytes(b"secret")
    with pytest.raises(ValueError, match="inventory is not exact"):
        wheel_staging.stage_container_wheelhouse(
            extra_source,
            extra_parent / "staged",
            extra_parent / wheel_staging.STAGING_MANIFEST_NAME,
            lock_path=lock,
            runtime_contract_path=runtime,
        )
