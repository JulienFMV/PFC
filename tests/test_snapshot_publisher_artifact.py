from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from pfc_shaping import path_safety, publisher_runtime_admission
from pfc_shaping.data.snapshot_publisher import (
    SnapshotPublicationRepairRequired,
    committed_snapshot_repair_result,
)
from pfc_shaping.durable_artifact import DurableArtifactRepairRequired
from pfc_shaping.exit_codes import (
    EXIT_INTEGRITY_FAILURE,
    EXIT_PROJECTION_REPAIR_REQUIRED,
)
from pfc_shaping.publisher_package_contract import (
    PUBLISHER_EMBEDDED_CONTRACT_FILES,
    PUBLISHER_ENVIRONMENT_CONTRACT_PATH,
    PUBLISHER_FORBIDDEN_FILES,
    PUBLISHER_MANIFEST_PATH,
    PUBLISHER_RUNTIME_CONTRACT_PATH,
    PUBLISHER_RUNTIME_PYTHON_FILES,
)
from pfc_shaping.publisher_runtime_admission import (
    RUNTIME_DEPENDENCY_ROOT_ENV,
    SUPERVISOR_ADMISSION_SLO_ENV,
    SUPERVISOR_SCRATCH_ROOT_ENV,
    SUPERVISOR_WORKER_TIMEOUT_ENV,
)
from scripts import build_snapshot_publisher_zipapp as publisher_zipapp
from scripts import publish_governed_lt_input_snapshot as publisher_cli
from scripts.build_snapshot_publisher_runtime_closure import (
    build_runtime_dependency_closure,
)
from scripts.build_snapshot_publisher_zipapp import (
    audit_publisher_zipapp,
    build_publisher_zipapp,
)
from tests.test_governed_lt_input_snapshot_v2 import (
    _governed_snapshot,
    _write_external_predecessor,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def governed_publisher_supervision_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Path]:
    scratch = ROOT / "build" / "publisher-test-scratch" / uuid.uuid4().hex[:12]
    scratch.mkdir(parents=True)
    monkeypatch.setenv(SUPERVISOR_SCRATCH_ROOT_ENV, str(scratch))
    monkeypatch.setenv(SUPERVISOR_WORKER_TIMEOUT_ENV, "900")
    monkeypatch.setenv(SUPERVISOR_ADMISSION_SLO_ENV, "300")
    try:
        yield scratch
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


@pytest.fixture(scope="session")
def publisher_wheel_directory() -> Path:
    wheelhouse = os.environ.get("PFC_TEST_PUBLISHER_WHEELHOUSE", "").strip()
    if not wheelhouse:
        pytest.skip("a uv.lock-approved publisher wheelhouse is required")
    return Path(wheelhouse)


@pytest.fixture(scope="session")
def publisher_dependency_bundle(
    tmp_path_factory: pytest.TempPathFactory,
    publisher_wheel_directory: Path,
) -> tuple[Path, Path]:
    prebuilt = os.environ.get("PFC_TEST_PUBLISHER_DEPENDENCY_ROOT", "").strip()
    prebuilt_receipt = os.environ.get(
        "PFC_TEST_PUBLISHER_DEPENDENCY_RECEIPT", ""
    ).strip()
    if bool(prebuilt) != bool(prebuilt_receipt):
        pytest.fail("prebuilt publisher closure and receipt must be supplied together")
    if prebuilt:
        return Path(prebuilt), Path(prebuilt_receipt)
    build_root = tmp_path_factory.mktemp("publisher-runtime")
    output = build_root / "site-packages"
    receipt = build_root / "dependency-closure-receipt.json"
    result = build_runtime_dependency_closure(
        publisher_wheel_directory,
        output,
        receipt,
    )
    assert result["distribution_count"] == 11
    return output, receipt


@pytest.fixture(scope="session")
def publisher_dependency_root(
    publisher_dependency_bundle: tuple[Path, Path],
) -> Path:
    return publisher_dependency_bundle[0]


@pytest.fixture(scope="session")
def publisher_dependency_receipt(
    publisher_dependency_bundle: tuple[Path, Path],
) -> Path:
    return publisher_dependency_bundle[1]


@pytest.fixture(scope="session")
def publisher_artifact(
    tmp_path_factory: pytest.TempPathFactory,
    publisher_dependency_root: Path,
    publisher_dependency_receipt: Path,
    publisher_wheel_directory: Path,
) -> Path:
    artifact = tmp_path_factory.mktemp("publisher-artifact") / "publisher.pyz"
    build_publisher_zipapp(
        artifact,
        publisher_dependency_root,
        publisher_dependency_receipt,
        publisher_wheel_directory,
    )
    return artifact


def test_publisher_zipapp_is_reproducible_runnable_and_signer_free(
    tmp_path: Path,
    publisher_dependency_root: Path,
    publisher_dependency_receipt: Path,
    publisher_wheel_directory: Path,
) -> None:
    first = tmp_path / "publisher-a.pyz"
    second = tmp_path / "publisher-b.pyz"

    first_audit = build_publisher_zipapp(
        first,
        publisher_dependency_root,
        publisher_dependency_receipt,
        publisher_wheel_directory,
    )
    second_audit = build_publisher_zipapp(
        second,
        publisher_dependency_root,
        publisher_dependency_receipt,
        publisher_wheel_directory,
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_audit["artifact_sha256"] == second_audit["artifact_sha256"]
    assert first_audit["foreign_authority_signer_present"] is False
    assert first_audit["release_attestation_status"] == "MISSING_REQUIRED_EXTERNAL"
    with zipfile.ZipFile(first) as archive:
        names = set(archive.namelist())
    assert PUBLISHER_MANIFEST_PATH in names
    assert set(PUBLISHER_EMBEDDED_CONTRACT_FILES).issubset(names)
    assert set(PUBLISHER_RUNTIME_PYTHON_FILES).issubset(names)
    assert PUBLISHER_FORBIDDEN_FILES.isdisjoint(names)

    environment = os.environ.copy()
    environment[RUNTIME_DEPENDENCY_ROOT_ENV] = str(publisher_dependency_root)
    temporary_root = Path(environment[SUPERVISOR_SCRATCH_ROOT_ENV])
    capture_residue_before = {
        path.name
        for pattern in (
            "fmv-pfc-publisher-supervisor-*",
            "fmv-pfc-publisher-runtime-*",
        )
        for path in temporary_root.glob(pattern)
    }
    help_outputs = []
    for command in ([], ["prepare"], ["cas"], ["finalize"]):
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                str(first),
                *command,
                "--help",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=960,
        )
        assert result.returncode == 0, result.stderr
        help_outputs.append(result.stdout)
    capture_residue_after = {
        path.name
        for pattern in (
            "fmv-pfc-publisher-supervisor-*",
            "fmv-pfc-publisher-runtime-*",
        )
        for path in temporary_root.glob(pattern)
    }
    assert capture_residue_after == capture_residue_before
    assert "{prepare,cas,finalize}" in help_outputs[0]

    with zipfile.ZipFile(first) as archive:
        runtime = json.loads(archive.read(PUBLISHER_RUNTIME_CONTRACT_PATH))
        environment = json.loads(archive.read(PUBLISHER_ENVIRONMENT_CONTRACT_PATH))
    assert runtime["delivery_model"] == (
        "ISOLATED_ZIPAPP_WITH_CONTENT_ADDRESSED_DEPENDENCY_CLOSURE"
    )
    assert runtime["artifact_contains_dependencies"] is False
    assert runtime["launcher_policy"]["required_invocation"] == (
        "python.exe -I -S -B artifact.pyz"
    )
    assert runtime["dependency_closure"]["tree_sha256"] == first_audit[
        "dependency_tree_sha256"
    ]
    assert runtime["dependency_closure"]["distribution_count"] == 11
    assert runtime["dependency_closure"]["receipt_sha256"] == first_audit[
        "dependency_closure_receipt_sha256"
    ]
    assert runtime["dependency_closure"]["source_lock_sha256"] == runtime[
        "source_lock"
    ]["sha256"]
    assert runtime["dependency_closure"]["wheel_provenance"] == (
        "VERIFIED_ARCHIVES_AND_RECORD_AGAINST_UV_LOCK"
    )
    assert runtime["dependency_closure"]["extraction_policy"] == {
        "pth_files": "FORBIDDEN",
        "recorded_bytecode": "VERIFIED_THEN_EXCLUDED",
        "recorded_nested_wheels": "VERIFIED_THEN_EXCLUDED",
    }
    assert len(runtime["dependency_closure"]["wheels"]) == 11
    assert all(
        len(item["sha256"]) == 64 and len(item["record_sha256"]) == 64
        for item in runtime["dependency_closure"]["wheels"]
    )
    assert runtime["installation_policy"]["network_access"] == "DENIED"
    assert environment["production_authorization"] is False


def test_real_publisher_bundle_seals_a_non_authorizing_container_context(
    tmp_path: Path,
    publisher_artifact: Path,
    publisher_dependency_root: Path,
    publisher_dependency_receipt: Path,
) -> None:
    from scripts.check_snapshot_publisher_container_context import (
        ARTIFACT_NAME,
        CONTAINER_MANIFEST_NAME,
        DEPENDENCY_RECEIPT_NAME,
        DEPENDENCY_ROOT_NAME,
        OPERATIONS_CONTRACT_NAME,
        WHEELHOUSE_STAGING_MANIFEST_NAME,
        build_container_context_manifest,
    )

    context = tmp_path / "container-context"
    context.mkdir()
    shutil.copy2(publisher_artifact, context / ARTIFACT_NAME)
    shutil.copy2(publisher_dependency_receipt, context / DEPENDENCY_RECEIPT_NAME)
    shutil.copy2(
        ROOT / "deploy" / "publisher" / OPERATIONS_CONTRACT_NAME,
        context / OPERATIONS_CONTRACT_NAME,
    )
    shutil.copytree(publisher_dependency_root, context / DEPENDENCY_ROOT_NAME)
    receipt = json.loads((context / DEPENDENCY_RECEIPT_NAME).read_bytes())
    wheelhouse_staging = {
        "schema_version": "fmv_lt_snapshot_publisher_wheelhouse_staging.v1",
        "status": "PASS",
        "source_lock": receipt["source_lock"],
        "wheel_count": len(receipt["wheels"]),
        "wheels": [
            {
                "name": item["name"],
                "version": item["version"],
                "filename": item["filename"],
                "sha256": item["sha256"],
                "size": item["size"],
            }
            for item in receipt["wheels"]
        ],
        "production_authorization": False,
    }
    (context / WHEELHOUSE_STAGING_MANIFEST_NAME).write_bytes(
        json.dumps(
            wheelhouse_staging,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest = "a" * 64

    result = build_container_context_manifest(
        context,
        base_image=f"registry.fmv.local/pfc/python@sha256:{digest}",
        output=context / CONTAINER_MANIFEST_NAME,
    )

    assert result["status"] == "PASS"
    assert result["base_image_digest"] == digest
    assert result["artifact_sha256"] == hashlib.sha256(
        (context / ARTIFACT_NAME).read_bytes()
    ).hexdigest()
    assert result["production_authorization"] is False


@pytest.mark.slow
def test_optimized_publisher_zipapp_prepares_real_provider_raw_v3_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publisher_dependency_root: Path,
    publisher_artifact: Path,
) -> None:
    isolated_cwd = tmp_path / "isolated-cwd"
    isolated_cwd.mkdir()
    locked_provider_runtime = _provider_runtime_fingerprint_from_artifact(
        publisher_artifact,
        publisher_dependency_root,
        cwd=isolated_cwd,
    )
    monkeypatch.setattr(
        "pfc_shaping.data.governed_lt_acquisition.provider_runtime_fingerprint",
        lambda: locked_provider_runtime,
    )
    source_root, contract = _governed_snapshot(tmp_path / "source", monkeypatch)
    assert contract["schema_version"] == "lt_input_snapshot.v3"
    source = source_root / "snapshots" / "generation-001"
    target = tmp_path / "target"
    target.mkdir()
    predecessor_payload = _write_external_predecessor(target)

    request_private_name = "PFC_DATA_PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_PATH"
    request_private = os.environ[request_private_name]
    environment = {
        name: value
        for name, value in os.environ.items()
        if "PRIVATE_KEY" not in name and not name.startswith("_PFC_SNAPSHOT_PUBLISHER_")
    }
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    for compatibility_root in (
        "PFC_DATA_ROOT",
        "PFC_LT_DATA_ROOT",
        "PFC_SHARED_DATA_ROOT",
    ):
        environment.pop(compatibility_root, None)
    environment[request_private_name] = request_private
    environment[RUNTIME_DEPENDENCY_ROOT_ENV] = str(publisher_dependency_root)
    environment["FMV_DATA_ROOT"] = str(target)

    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-I",
            "-S",
            "-B",
            str(publisher_artifact),
            "prepare",
            "--source-bundle",
            str(source),
            "--data-root",
            str(target),
            "--operation-id",
            "00000000-0000-4000-8000-000000000061",
            "--operation-created-at-utc",
            "2026-07-16T09:00:00+00:00",
            "--expected-anchor-sequence",
            "1",
            "--expected-anchor-receipt-id",
            "0" * 64,
        ],
        cwd=isolated_cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=960,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "PREPARED_NOT_COMMITTED"
    assert payload["commit_status"] == "NOT_COMMITTED"
    assert payload["production_authorization"] is False
    assert Path(payload["intent_path"]).is_file()
    assert (target / "views" / "pfc_lt" / "current.json").read_bytes() == predecessor_payload


def _provider_runtime_fingerprint_from_artifact(
    artifact: Path,
    dependency_root: Path,
    *,
    cwd: Path,
) -> dict[str, object]:
    code = (
        "import json,sys;"
        "sys.path[:0]=[sys.argv[1],sys.argv[2]];"
        "from pfc_shaping.data.governed_lt_acquisition import "
        "provider_runtime_fingerprint;"
        "print(json.dumps(provider_runtime_fingerprint(),sort_keys=True))"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-I",
            "-S",
            "-B",
            "-c",
            code,
            str(artifact),
            str(dependency_root),
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    return payload


def test_publisher_zipapp_auditor_rejects_extra_authority_module(
    tmp_path: Path,
    publisher_artifact: Path,
) -> None:
    artifact = tmp_path / "publisher.pyz"
    artifact.write_bytes(publisher_artifact.read_bytes())
    with zipfile.ZipFile(artifact, "a") as archive:
        archive.writestr(
            "pfc_shaping/data/snapshot_bootstrap_authority.py",
            b"FORBIDDEN = True\n",
        )

    with pytest.raises(ValueError, match="positive contract"):
        audit_publisher_zipapp(artifact)


def test_publisher_builder_refuses_to_replace_different_existing_artifact(
    tmp_path: Path,
    publisher_dependency_root: Path,
    publisher_dependency_receipt: Path,
    publisher_wheel_directory: Path,
) -> None:
    artifact = tmp_path / "publisher.pyz"
    artifact.write_bytes(b"previous admitted artifact")

    with pytest.raises(ValueError, match="already exists with other bytes"):
        build_publisher_zipapp(
            artifact,
            publisher_dependency_root,
            publisher_dependency_receipt,
            publisher_wheel_directory,
        )

    assert artifact.read_bytes() == b"previous admitted artifact"


def test_runtime_contract_binding_rejects_tree_changed_after_receipt_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = json.loads(
        (ROOT / "deploy" / "publisher" / "runtime-contract.json").read_text(
            encoding="utf-8"
        )
    )
    monkeypatch.setattr(
        publisher_zipapp,
        "dependency_tree_inventory",
        lambda _root: {"tree_sha256": "b" * 64, "file_count": 2},
    )

    with pytest.raises(ValueError, match="changed after receipt validation"):
        publisher_zipapp._bind_runtime_contract(
            template,
            tmp_path,
            {"tree_sha256": "a" * 64, "file_count": 1},
        )


def test_publisher_audit_hashes_the_same_stable_bytes_it_audits(
    tmp_path: Path,
    publisher_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "publisher.pyz"
    artifact.write_bytes(publisher_artifact.read_bytes())
    captured = artifact.read_bytes()

    def capture_then_replace(_path: Path) -> bytes:
        artifact.write_bytes(b"replacement after stable capture")
        return captured

    monkeypatch.setattr(
        publisher_zipapp,
        "_stable_artifact_payload",
        capture_then_replace,
    )

    audit = publisher_zipapp.audit_publisher_zipapp(artifact)

    assert audit["artifact_sha256"] == hashlib.sha256(captured).hexdigest()
    assert audit["artifact_sha256"] != hashlib.sha256(artifact.read_bytes()).hexdigest()


def test_stable_artifact_capture_rejects_same_size_mtime_restored_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "publisher.pyz"
    original = b"original-publisher-image"
    replacement = b"replacement-image-bytes!"
    assert len(original) == len(replacement)
    artifact.write_bytes(original)
    initial = artifact.stat()
    real_open = os.open
    opens = 0

    def rewrite_before_verification_read(
        selected: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        nonlocal opens
        opens += 1
        if opens == 2:
            artifact.write_bytes(replacement)
            os.utime(
                artifact,
                ns=(initial.st_atime_ns, initial.st_mtime_ns),
            )
        return real_open(selected, flags, mode)

    # Deterministically rewrite between the primary and verification descriptor
    # opens. The second read must observe different bytes even when size and
    # mtime are restored.
    monkeypatch.setattr(
        path_safety,
        "os",
        SimpleNamespace(
            **{
                name: getattr(os, name)
                for name in dir(os)
                if not name.startswith("__") and name != "open"
            },
            open=rewrite_before_verification_read,
        ),
    )

    with pytest.raises(ValueError, match="changed during stable verification"):
        publisher_zipapp._stable_artifact_payload(artifact)


def test_stable_artifact_capture_rejects_link_count_change(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "publisher.pyz"
    alias = tmp_path / "publisher-hardlink.pyz"
    artifact.write_bytes(b"publisher-image")
    os.link(artifact, alias)

    with pytest.raises(ValueError, match="single-link regular file"):
        publisher_zipapp._stable_artifact_payload(artifact)


def test_publisher_environment_contract_separates_phase_private_keys() -> None:
    contract = json.loads(
        (ROOT / "deploy" / "publisher" / "environment-contract.json").read_text(
            encoding="utf-8"
        )
    )

    assert (
        publisher_runtime_admission._validated_environment_contract_mapping(contract)
        == contract
    )

    assert contract["schema_version"] == "fmv_lt_snapshot_publisher_environment.v1"
    assert "PFC_DATA_PUBLICATION_BOOTSTRAP_SIGNING_PRIVATE_KEY_PATH" in contract[
        "forbidden_in_all_phases"
    ]
    assert {
        "_PFC_SNAPSHOT_PUBLISHER_SUPERVISED_WORKER",
        "_PFC_SNAPSHOT_PUBLISHER_CAPTURE_ROOT",
        "_PFC_SNAPSHOT_PUBLISHER_WORKER_TOKEN",
    }.issubset(contract["forbidden_in_all_phases"])
    assert contract["forbidden_outside_prepare"] == [
        "PFC_DATA_PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_PATH"
    ]
    assert "PFC_DATA_PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_PATH" in contract["phases"][
        "prepare"
    ]["required"]
    assert "PFC_DATA_PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_PATH" not in contract[
        "phases"
    ]["cas"]["required"]
    assert "FMV_DATA_ROOT" in contract["phases"]["prepare"]["required"]
    assert "FMV_DATA_ROOT" in contract["phases"]["finalize"]["required"]
    assert "PFC_DATA_ROOT" not in json.dumps(contract, sort_keys=True)
    assert contract["path_authority"] == {
        "FMV_DATA_ROOT": (
            "REQUIRED_AND_EXPLICIT_DATA_ROOT_MUST_RESOLVE_EQUAL_AFTER_"
            "LEXICAL_NO_REPARSE_VALIDATION"
        ),
        "PFC_SNAPSHOT_PUBLISHER_SCRATCH_ROOT": (
            "REQUIRED_EXISTING_NO_REPARSE_WRITABLE_WITH_MINIMUM_FREE_CAPACITY"
        ),
    }
    assert contract["runtime_supervision"] == {
        "worker_timeout_seconds": "REQUIRED_FINITE_POSITIVE_MAX_3600",
        "admission_slo_seconds": (
            "REQUIRED_FINITE_POSITIVE_NOT_ABOVE_WORKER_TIMEOUT"
        ),
        "admission_metric": (
            "STRUCTURED_STDERR_ON_SUCCESS_STDOUT_ON_ADMITTED_FAILURE_"
            "FMV_LT_SNAPSHOT_PUBLISHER_ADMISSION_METRIC_V1"
        ),
    }
    assert contract["production_authorization"] is False
    assert set(contract["host_passthrough"]) == set(
        publisher_runtime_admission._SAFE_HOST_ENVIRONMENT_NAMES
    )
    assert contract["worker_runtime_overrides"] == {
        "TEMP": "SUPERVISOR_ROOT",
        "TMP": "SUPERVISOR_ROOT",
    }
    assert RUNTIME_DEPENDENCY_ROOT_ENV in contract["common_required"]
    assert SUPERVISOR_SCRATCH_ROOT_ENV in contract["common_required"]
    assert SUPERVISOR_WORKER_TIMEOUT_ENV in contract["common_required"]
    assert SUPERVISOR_ADMISSION_SLO_ENV in contract["common_required"]
    assert "PFC_DATA_TIMESTAMP_JOURNAL_ID" in contract["common_required"]


def test_supervised_worker_environment_is_allowlisted_and_defers_prepare_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = json.loads(
        (ROOT / "deploy" / "publisher" / "environment-contract.json").read_text(
            encoding="utf-8"
        )
    )
    private_name = "PFC_DATA_PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_PATH"
    monkeypatch.setenv(private_name, str(tmp_path / "request-private.pem"))
    monkeypatch.setenv("PFC_UNRELATED_DIAGNOSTIC", "must-not-pass")
    monkeypatch.setenv("TEMP", str(tmp_path / "untrusted-host-temp"))
    monkeypatch.setenv("TMP", str(tmp_path / "untrusted-host-tmp"))

    worker_environment, deferred = (
        publisher_runtime_admission._supervised_worker_environment(
            contract,
            phase="prepare",
            supervisor_root=tmp_path,
            worker_token="a" * 64,
        )
    )

    assert private_name not in worker_environment
    assert deferred == {private_name: str(tmp_path / "request-private.pem")}
    assert "PFC_UNRELATED_DIAGNOSTIC" not in worker_environment
    assert "PYTHONPATH" not in worker_environment
    assert worker_environment["TEMP"] == str(tmp_path)
    assert worker_environment["TMP"] == str(tmp_path)


def test_supervisor_rejects_forbidden_authority_before_worker_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = tmp_path / "publisher.pyz"
    artifact.write_bytes(b"publisher")
    contract = json.loads(
        (ROOT / "deploy" / "publisher" / "environment-contract.json").read_text(
            encoding="utf-8"
        )
    )
    monkeypatch.setenv(
        "PFC_DATA_PUBLICATION_BOOTSTRAP_SIGNING_PRIVATE_KEY_PATH",
        str(tmp_path / "forbidden.pem"),
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_assert_isolated_launcher_flags",
        lambda: None,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_publisher_artifact_path",
        lambda: artifact,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_verified_environment_contract",
        lambda _artifact: contract,
    )
    monkeypatch.setattr(
        publisher_runtime_admission.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("worker must not spawn")
        ),
    )

    result = publisher_runtime_admission._run_supervised_publisher_worker()

    assert result == 50
    payload = json.loads(capsys.readouterr().err)
    assert "forbidden authority" in payload["message"]


def test_publisher_cli_requires_authoritative_fmv_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in ("FMV_DATA_ROOT", "PFC_LT_DATA_ROOT", "PFC_SHARED_DATA_ROOT"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(SystemExit) as exc_info:
        publisher_cli._parse_args(
            [
                "prepare",
                "--data-root",
                str(tmp_path),
                "--operation-id",
                "00000000-0000-4000-8000-000000000001",
                "--operation-created-at-utc",
                "2026-07-17T00:00:00Z",
            ]
        )

    assert exc_info.value.code == 2
    assert "FMV_DATA_ROOT is required" in capsys.readouterr().err


def test_publisher_cli_rejects_data_root_divergent_from_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    authoritative = tmp_path / "authoritative"
    divergent = tmp_path / "divergent"
    monkeypatch.setenv("FMV_DATA_ROOT", str(authoritative))
    monkeypatch.delenv("PFC_LT_DATA_ROOT", raising=False)
    monkeypatch.delenv("PFC_SHARED_DATA_ROOT", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        publisher_cli._parse_args(
            [
                "prepare",
                "--data-root",
                str(divergent),
                "--operation-id",
                "00000000-0000-4000-8000-000000000001",
                "--operation-created-at-utc",
                "2026-07-17T00:00:00Z",
            ]
        )

    assert exc_info.value.code == 2
    assert "must match the authoritative FMV_DATA_ROOT" in capsys.readouterr().err


def test_publisher_cli_accepts_data_root_equal_to_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FMV_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("PFC_LT_DATA_ROOT", raising=False)
    monkeypatch.delenv("PFC_SHARED_DATA_ROOT", raising=False)

    args = publisher_cli._parse_args(
        [
            "prepare",
            "--data-root",
            str(tmp_path),
            "--operation-id",
            "00000000-0000-4000-8000-000000000001",
            "--operation-created-at-utc",
            "2026-07-17T00:00:00Z",
        ]
    )

    assert args.data_root == tmp_path


def test_publisher_cli_rejects_linked_authoritative_data_root_before_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    authoritative = tmp_path / "authoritative"
    alias = tmp_path / "authoritative-alias"
    authoritative.mkdir()
    try:
        alias.symlink_to(authoritative, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink capability is unavailable")
    monkeypatch.setenv("FMV_DATA_ROOT", str(alias))
    monkeypatch.delenv("PFC_LT_DATA_ROOT", raising=False)
    monkeypatch.delenv("PFC_SHARED_DATA_ROOT", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        publisher_cli._parse_args(
            [
                "prepare",
                "--data-root",
                str(alias),
                "--operation-id",
                "00000000-0000-4000-8000-000000000001",
                "--operation-created-at-utc",
                "2026-07-17T00:00:00Z",
            ]
        )

    assert exc_info.value.code == 2
    assert "link or reparse point" in capsys.readouterr().err


def test_publisher_runtime_contract_closes_exact_dependency_set_and_source_lock() -> None:
    contract_path = ROOT / "deploy" / "publisher" / "runtime-contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert contract["schema_version"] == "fmv_lt_snapshot_publisher_runtime.v6"
    assert contract["artifact"] == "fmv_lt_snapshot_publisher_artifact.v2"
    assert contract["python"] == {
        "implementation": "CPython",
        "version": "3.11.*",
        "abi": "cp311",
        "binding": "INJECT_EXACT_RUNTIME_FINGERPRINT_AT_BUILD",
    }
    assert {item["name"]: item["version"] for item in contract["locked_distributions"]} == {
        "cffi": "2.1.0",
        "cryptography": "47.0.0",
        "numpy": "2.0.2",
        "pandas": "2.3.3",
        "pyarrow": "21.0.0",
        "pycparser": "3.0",
        "python-dateutil": "2.9.0.post0",
        "pytz": "2026.2",
        "PyYAML": "6.0.3",
        "six": "1.17.0",
        "tzdata": "2026.3",
    }
    assert contract["source_lock"]["sha256"] == hashlib.sha256(
        (ROOT / "uv.lock").read_bytes()
    ).hexdigest()
    assert contract["dependency_closure"]["provenance_binding"] == (
        "INJECT_VERIFIED_UV_LOCK_WHEEL_RECEIPT_AT_BUILD"
    )
    assert contract["dependency_loading_policy"] == (
        publisher_runtime_admission.PUBLISHER_DEPENDENCY_LOADING_POLICY
    )
    assert contract["production_authorization"] is False


def test_publisher_runtime_fingerprint_rejects_non_cp311(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    version_info = SimpleNamespace(major=3, minor=12)
    monkeypatch.setattr(publisher_runtime_admission.sys, "version_info", version_info)

    with pytest.raises(
        publisher_runtime_admission.PublisherRuntimeAdmissionError,
        match="require CPython 3.11",
    ):
        publisher_runtime_admission.python_runtime_fingerprint()


def test_publisher_runtime_fingerprint_rejects_wrong_cache_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation = SimpleNamespace(cache_tag="cpython-311-mutated")
    monkeypatch.setattr(publisher_runtime_admission.sys, "implementation", implementation)

    with pytest.raises(
        publisher_runtime_admission.PublisherRuntimeAdmissionError,
        match="cache tag cpython-311",
    ):
        publisher_runtime_admission.python_runtime_fingerprint()


def _synthetic_dependency_root(tmp_path: Path) -> Path:
    root = tmp_path / "source-site-packages"
    package = root / "demo"
    metadata = root / "demo-1.0.dist-info"
    package.mkdir(parents=True)
    metadata.mkdir()
    (package / "__init__.py").write_bytes(b'VALUE = "EXPECTED"\n')
    (metadata / "METADATA").write_bytes(
        b"Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n"
    )
    return root


def test_private_dependency_capture_is_independent_from_source_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _synthetic_dependency_root(tmp_path)
    expected_tree = publisher_runtime_admission.dependency_tree_inventory(source)
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_CAPTURED_DEPENDENCY_RUNTIME",
        None,
    )

    captured = publisher_runtime_admission._capture_verified_dependency_closure(
        source,
        expected_tree=expected_tree,
        expected_distributions={"demo": "1.0"},
    )
    runtime = publisher_runtime_admission._CAPTURED_DEPENDENCY_RUNTIME
    assert runtime is not None
    try:
        (source / "demo" / "__init__.py").write_bytes(b'VALUE = "ATTACK"\n')

        assert (captured / "demo" / "__init__.py").read_bytes() == (
            b'VALUE = "EXPECTED"\n'
        )
        assert publisher_runtime_admission.dependency_tree_inventory(
            captured
        ) == expected_tree
        assert captured != source
    finally:
        runtime.cleanup()


def test_private_dependency_capture_rejects_source_mutation_during_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _synthetic_dependency_root(tmp_path)
    expected_tree = publisher_runtime_admission.dependency_tree_inventory(source)
    original_copy = publisher_runtime_admission._copy_stable_runtime_file
    calls = 0

    def copy_then_attack(source_path: Path, destination: Path) -> None:
        nonlocal calls
        original_copy(source_path, destination)
        calls += 1
        if calls == 1:
            (source / "demo" / "__init__.py").write_bytes(b'VALUE = "ATTACK"\n')

    monkeypatch.setattr(
        publisher_runtime_admission,
        "_CAPTURED_DEPENDENCY_RUNTIME",
        None,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_copy_stable_runtime_file",
        copy_then_attack,
    )

    with pytest.raises(
        publisher_runtime_admission.PublisherRuntimeAdmissionError,
        match="changed during private capture|private dependency capture diverges",
    ):
        publisher_runtime_admission._capture_verified_dependency_closure(
            source,
            expected_tree=expected_tree,
            expected_distributions={"demo": "1.0"},
        )

    assert publisher_runtime_admission._CAPTURED_DEPENDENCY_RUNTIME is None


def test_runtime_admission_appends_only_private_captured_dependency_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "publisher.pyz"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("placeholder", b"runtime-admission-test")
    source = tmp_path / "verified-source-site-packages"
    captured = tmp_path / "private-capture" / "site-packages"
    source.mkdir()
    captured.mkdir(parents=True)
    (captured / "demo.py").write_text("VALUE = 'captured'\n", encoding="utf-8")
    dist_info = captured / "demo-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (dist_info / "top_level.txt").write_text("demo\n", encoding="utf-8")
    stdlib = tmp_path / "stdlib"
    stdlib.mkdir()
    (stdlib / "demo.py").write_text("VALUE = 'attacker'\n", encoding="utf-8")
    baseline = [str(artifact), str(stdlib)]
    contract = {
        "schema_version": "fmv_lt_snapshot_publisher_runtime.v6",
        "artifact": "fmv_lt_snapshot_publisher_artifact.v2",
        "dependency_loading_policy": (
            publisher_runtime_admission.PUBLISHER_DEPENDENCY_LOADING_POLICY
        ),
        "python": {"runtime": "test"},
        "dependency_closure": {
            "tree_sha256": "a" * 64,
            "file_count": 1,
        },
        "locked_distributions": [{"name": "demo", "version": "1.0"}],
    }

    monkeypatch.setattr(sys, "path", baseline.copy())
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_verified_artifact_member_payload",
        lambda _artifact, **_kwargs: b"{}",
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_strict_json_mapping",
        lambda _payload: contract,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_verify_isolated_launcher",
        lambda _contract: None,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "python_runtime_fingerprint",
        lambda: {"runtime": "test"},
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_verify_dependency_closure",
        lambda _contract, **_kwargs: source,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_verify_distribution_inventory",
        lambda _contract, _root: {"demo": "1.0"},
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_capture_verified_dependency_closure",
        lambda _root, **_kwargs: captured,
    )

    monkeypatch.delitem(sys.modules, "demo", raising=False)
    result = publisher_runtime_admission.verify_publisher_runtime(artifact)

    assert result["dependency_import_source"] == "PROCESS_PRIVATE_EXACT_BYTE_COPY"
    assert sys.path == [str(artifact), str(captured), str(tmp_path / "stdlib")]
    assert str(source) not in sys.path
    assert Path(sys.modules["demo"].__file__).parent == captured


def test_runtime_admission_rejects_lexical_alias_of_dependency_source_on_sys_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "publisher.pyz"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("placeholder", b"runtime-admission-test")
    source = tmp_path / "verified-source"
    source.mkdir()
    alias = os.path.join(str(source.parent), ".", source.name)
    contract = {
        "schema_version": "fmv_lt_snapshot_publisher_runtime.v6",
        "artifact": "fmv_lt_snapshot_publisher_artifact.v2",
        "dependency_loading_policy": (
            publisher_runtime_admission.PUBLISHER_DEPENDENCY_LOADING_POLICY
        ),
        "python": {"runtime": "test"},
        "dependency_closure": {
            "tree_sha256": "a" * 64,
            "file_count": 1,
        },
        "locked_distributions": [{"name": "demo", "version": "1.0"}],
    }
    capture_called = False

    def unexpected_capture(_root: Path, **_kwargs: object) -> Path:
        nonlocal capture_called
        capture_called = True
        return tmp_path / "unexpected-capture"

    monkeypatch.setattr(sys, "path", [alias])
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_verified_artifact_member_payload",
        lambda _artifact, **_kwargs: b"{}",
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_strict_json_mapping",
        lambda _payload: contract,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_verify_isolated_launcher",
        lambda _contract: None,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "python_runtime_fingerprint",
        lambda: {"runtime": "test"},
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_verify_dependency_closure",
        lambda _contract, **_kwargs: source,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_verify_distribution_inventory",
        lambda _contract, _root: {"demo": "1.0"},
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_capture_verified_dependency_closure",
        unexpected_capture,
    )

    with pytest.raises(
        publisher_runtime_admission.PublisherRuntimeAdmissionError,
        match="source dependency root is already importable",
    ):
        publisher_runtime_admission.verify_publisher_runtime(artifact)

    assert capture_called is False


def test_dependency_source_sys_path_guard_rejects_physical_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "verified-source"
    alias = tmp_path / "source-alias"
    source.mkdir()
    try:
        alias.symlink_to(source, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink capability is unavailable")
    monkeypatch.setattr(sys, "path", [str(alias)])

    with pytest.raises(
        publisher_runtime_admission.PublisherRuntimeAdmissionError,
        match="source dependency root is already importable",
    ):
        publisher_runtime_admission._assert_dependency_source_not_on_sys_path(source)


def test_isolated_sys_path_rejects_distinct_base_relative_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "runtime"
    artifact = tmp_path / "publisher.pyz"
    shadow = base / "shadow"
    shadow.mkdir(parents=True)
    artifact.write_bytes(b"zipapp")
    monkeypatch.setattr(
        sys,
        "path",
        [
            str(artifact),
            str(base / "python311.zip"),
            str(base / "DLLs"),
            str(base / "Lib"),
            str(base),
            str(shadow),
        ],
    )

    with pytest.raises(
        publisher_runtime_admission.PublisherRuntimeAdmissionError,
        match="sys.path is not isolated",
    ):
        publisher_runtime_admission._isolate_runtime_sys_path(
            artifact=artifact,
            base=base,
        )


def test_isolated_sys_path_removes_runtime_root_from_import_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "runtime"
    artifact = tmp_path / "publisher.pyz"
    monkeypatch.setattr(
        sys,
        "path",
        [
            str(artifact),
            str(base / "python311.zip"),
            str(base / "DLLs"),
            str(base / "Lib"),
            str(base),
        ],
    )

    publisher_runtime_admission._isolate_runtime_sys_path(
        artifact=artifact,
        base=base,
    )

    assert str(base) not in sys.path
    assert sys.path == [
        str(artifact),
        str(base / "python311.zip"),
        str(base / "DLLs"),
        str(base / "Lib"),
    ]


def test_dependency_import_sys_path_mutation_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "publisher.pyz"
    captured = tmp_path / "captured" / "site-packages"
    captured.mkdir(parents=True)
    admitted = (str(artifact), str(captured))
    monkeypatch.setattr(sys, "path", [*admitted, str(tmp_path / "hostile")])

    with pytest.raises(
        publisher_runtime_admission.PublisherRuntimeAdmissionError,
        match="mutated the admitted sys.path",
    ):
        publisher_runtime_admission._assert_sys_path_unchanged_after_dependency_imports(
            admitted,
            captured_root=captured,
        )


def test_public_entrypoint_rejects_unbound_internal_worker_marker(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(
        publisher_runtime_admission._SUPERVISED_WORKER_ENV,
        "1",
    )
    monkeypatch.delenv(
        publisher_runtime_admission._SUPERVISED_CAPTURE_ROOT_ENV,
        raising=False,
    )
    monkeypatch.delenv(
        publisher_runtime_admission._SUPERVISED_WORKER_TOKEN_ENV,
        raising=False,
    )

    result = publisher_runtime_admission.run_admitted_publisher_zipapp()

    assert result == 50
    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "RUNTIME_ADMISSION_FAILED"
    assert "capability is missing" in payload["message"]


def test_public_entrypoint_rejects_forged_prefixed_capture_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    capture_root = tmp_path / "fmv-pfc-publisher-supervisor-forged"
    capture_root.mkdir()
    monkeypatch.setenv(publisher_runtime_admission._SUPERVISED_WORKER_ENV, "1")
    monkeypatch.setenv(
        publisher_runtime_admission._SUPERVISED_CAPTURE_ROOT_ENV,
        str(capture_root),
    )
    monkeypatch.setenv(
        publisher_runtime_admission._SUPERVISED_WORKER_TOKEN_ENV,
        "a" * 64,
    )

    result = publisher_runtime_admission.run_admitted_publisher_zipapp()

    assert result == 50
    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "RUNTIME_ADMISSION_FAILED"
    assert "outside governed scratch" in payload["message"]


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_windows_supervisor_contains_worker_before_code_and_kills_descendant(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "worker-and-child-pids.txt"
    worker_program = (
        "import os,pathlib,subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        f"pathlib.Path({str(marker)!r}).write_text(f'{{os.getpid()}},{{child.pid}}');"
        "time.sleep(60)"
    )
    worker = subprocess.Popen(
        [sys.executable, "-c", worker_program],
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP
            | publisher_runtime_admission._WINDOWS_CREATE_SUSPENDED
        ),
    )
    job_handle: int | None = None
    try:
        time.sleep(0.2)
        assert not marker.exists()
        job_handle = publisher_runtime_admission._create_windows_kill_on_close_job()
        publisher_runtime_admission._assign_process_to_windows_job(job_handle, worker)
        publisher_runtime_admission._resume_windows_process(worker)
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists()
        child_pid = int(marker.read_text(encoding="utf-8").split(",")[1])
        started = time.monotonic()
        publisher_runtime_admission._close_windows_handle(job_handle)
        job_handle = None
        worker.wait(timeout=5)
        assert time.monotonic() - started < 5

        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        def child_is_active() -> bool:
            handle = kernel32.OpenProcess(0x1000, False, child_pid)
            if not handle:
                return False
            try:
                exit_code = wintypes.DWORD()
                return bool(
                    kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                    and exit_code.value == 259
                )
            finally:
                kernel32.CloseHandle(handle)

        deadline = time.monotonic() + 5
        while child_is_active() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not child_is_active()
    finally:
        if job_handle is not None:
            publisher_runtime_admission._close_windows_handle(job_handle)
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=5)


def test_supervisor_normalizes_scratch_creation_failure_to_json_exit_50(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_assert_isolated_launcher_flags",
        lambda: None,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_allocate_inherited_acl_runtime_directory",
        lambda **_kwargs: (_ for _ in ()).throw(
            OSError("injected supervisor scratch failure")
        ),
    )

    result = publisher_runtime_admission._run_supervised_publisher_worker()

    assert result == 50
    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "RUNTIME_ADMISSION_FAILED"
    assert payload["error_type"] == "OSError"
    assert payload["message"] == "injected supervisor scratch failure"


def test_supervisor_cleanup_retry_clears_transient_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor_root = tmp_path / "fmv-pfc-publisher-supervisor-retry"
    supervisor_root.mkdir()
    real_rmtree = shutil.rmtree
    attempts = 0

    def transient_rmtree(path: Path, *args, **kwargs) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("transient sharing violation")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(publisher_runtime_admission.shutil, "rmtree", transient_rmtree)
    monkeypatch.setattr(publisher_runtime_admission.time, "sleep", lambda _seconds: None)

    assert publisher_runtime_admission._cleanup_supervisor_root(supervisor_root) is None
    assert attempts == 2
    assert not supervisor_root.exists()


def test_worker_failure_is_not_followed_by_second_cleanup_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = tmp_path / "publisher.pyz"
    artifact.write_bytes(b"publisher")
    contract = json.loads(
        (ROOT / "deploy" / "publisher" / "environment-contract.json").read_text(
            encoding="utf-8"
        )
    )

    class FailedWorker:
        pid = 12347

        def wait(self, timeout: float | None = None) -> int:
            assert timeout is not None and 0 < timeout <= 900.0
            return 7

        def poll(self) -> int:
            return 7

    monkeypatch.setattr(
        publisher_runtime_admission,
        "_assert_isolated_launcher_flags",
        lambda: None,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_publisher_artifact_path",
        lambda: artifact,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_verified_environment_contract",
        lambda _artifact: contract,
    )
    monkeypatch.setattr(
        publisher_runtime_admission.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FailedWorker(),
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_create_windows_kill_on_close_job",
        lambda: 1,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_assign_process_to_windows_job",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_resume_windows_process",
        lambda _worker: None,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_close_windows_handle",
        lambda _handle: None,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_cleanup_supervisor_root",
        lambda _root: OSError("persistent cleanup failure"),
    )
    monkeypatch.setattr(sys, "argv", ["publisher.pyz", "prepare"])
    monkeypatch.setenv(
        "PFC_DATA_PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_PATH",
        str(tmp_path / "request-private.pem"),
    )
    metric = {
        "schema_version": "fmv_lt_snapshot_publisher_admission_metric.v1",
        "metric": "admission_duration_seconds",
        "value": 0.25,
        "slo_seconds": 300.0,
        "deadline_seconds": 900.0,
        "slo_met": True,
        "status": "PASS",
        "worker_pid": 12347,
        "production_authorization": False,
    }
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_wait_for_admission_then_deliver_environment",
        lambda **_kwargs: metric,
    )

    result = publisher_runtime_admission._run_supervised_publisher_worker()

    assert result == 7
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == metric


def test_supervisor_rejects_linked_scratch_root_before_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "real-scratch"
    alias = tmp_path / "scratch-alias"
    scratch.mkdir()
    try:
        alias.symlink_to(scratch, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink capability is unavailable")
    monkeypatch.setenv(SUPERVISOR_SCRATCH_ROOT_ENV, str(alias))

    with pytest.raises(
        publisher_runtime_admission.PublisherRuntimeAdmissionError,
        match="scratch root is invalid",
    ):
        publisher_runtime_admission._validated_runtime_supervision_config()


@pytest.mark.skipif(os.name != "nt", reason="Windows path headroom contract")
def test_capture_rejects_windows_path_without_operational_headroom() -> None:
    captured = Path("C:/") / ("a" * 180) / "site-packages"
    entries = [(f"package/{'b' * 80}.py", Path("unused"))]

    with pytest.raises(
        publisher_runtime_admission.PublisherRuntimeAdmissionError,
        match="insufficient Windows path headroom",
    ):
        publisher_runtime_admission._assert_capture_path_headroom(captured, entries)


def test_supervisor_enforces_worker_deadline_and_terminates_process_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = tmp_path / "publisher.pyz"
    artifact.write_bytes(b"publisher")
    monkeypatch.setenv(SUPERVISOR_WORKER_TIMEOUT_ENV, "0.01")
    monkeypatch.setenv(SUPERVISOR_ADMISSION_SLO_ENV, "0.005")
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_assert_isolated_launcher_flags",
        lambda: None,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_publisher_artifact_path",
        lambda: artifact,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_verified_environment_contract",
        lambda _artifact: json.loads(
            (ROOT / "deploy" / "publisher" / "environment-contract.json").read_text(
                encoding="utf-8"
            )
        ),
    )

    class TimedOutWorker:
        pid = 12345
        terminated = False

        def wait(self, timeout: float | None = None) -> int:
            raise subprocess.TimeoutExpired("publisher-worker", timeout)

        def poll(self) -> int | None:
            return -9 if self.terminated else None

    worker = TimedOutWorker()
    monkeypatch.setattr(
        publisher_runtime_admission.subprocess,
        "Popen",
        lambda *_args, **_kwargs: worker,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_terminate_supervised_worker",
        lambda selected: setattr(selected, "terminated", True),
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_create_windows_kill_on_close_job",
        lambda: 1,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_assign_process_to_windows_job",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_resume_windows_process",
        lambda _worker: None,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_close_windows_handle",
        lambda _handle: None,
    )

    result = publisher_runtime_admission._run_supervised_publisher_worker()

    assert result == 50
    assert worker.terminated is True
    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "RUNTIME_ADMISSION_FAILED"
    assert "governed runtime deadline" in payload["message"]


def test_supervisor_emits_structured_admission_duration_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = tmp_path / "publisher.pyz"
    artifact.write_bytes(b"publisher")
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_assert_isolated_launcher_flags",
        lambda: None,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_publisher_artifact_path",
        lambda: artifact,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_verified_environment_contract",
        lambda _artifact: json.loads(
            (ROOT / "deploy" / "publisher" / "environment-contract.json").read_text(
                encoding="utf-8"
            )
        ),
    )

    class SuccessfulWorker:
        pid = 12346

        def wait(self, timeout: float | None = None) -> int:
            assert timeout == 900.0
            return 0

        def poll(self) -> int:
            return 0

    monkeypatch.setattr(
        publisher_runtime_admission.subprocess,
        "Popen",
        lambda *_args, **_kwargs: SuccessfulWorker(),
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_load_supervised_admission_metric",
        lambda _root, **_kwargs: {
            "schema_version": "fmv_lt_snapshot_publisher_admission_metric.v1",
            "metric": "admission_duration_seconds",
            "value": 0.25,
            "slo_seconds": 300.0,
            "deadline_seconds": 900.0,
            "slo_met": True,
            "status": "PASS",
            "worker_pid": 12346,
            "production_authorization": False,
        },
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_create_windows_kill_on_close_job",
        lambda: 1,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_assign_process_to_windows_job",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_resume_windows_process",
        lambda _worker: None,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_close_windows_handle",
        lambda _handle: None,
    )

    result = publisher_runtime_admission._run_supervised_publisher_worker()

    assert result == 0
    metric = json.loads(capsys.readouterr().err)
    assert metric["schema_version"] == (
        "fmv_lt_snapshot_publisher_admission_metric.v1"
    )
    assert metric["metric"] == "admission_duration_seconds"
    assert metric["deadline_seconds"] == 900.0
    assert metric["slo_seconds"] == 300.0
    assert metric["status"] == "PASS"


def test_supervised_admission_metric_round_trip_is_token_bound(
    governed_publisher_supervision_environment: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "a" * 64
    monkeypatch.setenv(
        publisher_runtime_admission._SUPERVISED_WORKER_TOKEN_ENV,
        token,
    )

    publisher_runtime_admission._write_supervised_admission_metric(
        governed_publisher_supervision_environment,
        admission_duration_seconds=0.25,
    )
    metric = publisher_runtime_admission._load_supervised_admission_metric(
        governed_publisher_supervision_environment,
        expected_token=token,
        expected_deadline_seconds=900.0,
        expected_slo_seconds=300.0,
        expected_worker_pid=os.getpid(),
    )

    assert metric["metric"] == "admission_duration_seconds"
    assert metric["value"] == 0.25
    assert metric["status"] == "PASS"
    assert "token" not in metric

    with pytest.raises(
        publisher_runtime_admission.PublisherRuntimeAdmissionError,
        match="metric is invalid",
    ):
        publisher_runtime_admission._load_supervised_admission_metric(
            governed_publisher_supervision_environment,
            expected_token=token,
            expected_deadline_seconds=900.0,
            expected_slo_seconds=300.0,
            expected_worker_pid=os.getpid() + 1,
        )

    with pytest.raises(
        publisher_runtime_admission.PublisherRuntimeAdmissionError,
        match="metric is invalid",
    ):
        publisher_runtime_admission._load_supervised_admission_metric(
            governed_publisher_supervision_environment,
            expected_token="b" * 64,
            expected_deadline_seconds=900.0,
            expected_slo_seconds=300.0,
            expected_worker_pid=os.getpid(),
        )


def test_post_admission_delivery_error_retains_already_sealed_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric = {
        "schema_version": "fmv_lt_snapshot_publisher_admission_metric.v1",
        "metric": "admission_duration_seconds",
        "value": 0.25,
        "slo_seconds": 300.0,
        "deadline_seconds": 900.0,
        "slo_met": True,
        "status": "PASS",
        "worker_pid": 12348,
        "production_authorization": False,
    }
    (tmp_path / ".runtime-admission-metric.json").write_bytes(b"sealed")

    class AdmittedWorker:
        pid = 12348

        def poll(self) -> None:
            return None

    monkeypatch.setattr(
        publisher_runtime_admission,
        "_load_supervised_admission_metric",
        lambda _root, **_kwargs: metric,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_write_post_admission_environment_capability",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("injected post-admission delivery failure")
        ),
    )

    with pytest.raises(
        publisher_runtime_admission.PublisherPostAdmissionEnvironmentDeliveryError
    ) as captured:
        publisher_runtime_admission._wait_for_admission_then_deliver_environment(
            worker=AdmittedWorker(),  # type: ignore[arg-type]
            supervisor_root=tmp_path,
            deferred_environment={"PFC_SECRET_PATH": "C:/secret.pem"},
            expected_token="a" * 64,
            expected_deadline_seconds=900.0,
            expected_slo_seconds=300.0,
            timeout_seconds=1.0,
        )

    assert captured.value.admission_metric == metric


@pytest.mark.parametrize(
    ("duration", "expected_value", "expected_status"),
    [
        (300.0000004, 300.0, "PASS"),
        (300.0000006, 300.000001, "SLO_BREACH"),
    ],
)
def test_supervised_admission_metric_uses_serialized_boundary_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    duration: float,
    expected_value: float,
    expected_status: str,
) -> None:
    token = "c" * 64
    metric_root = tmp_path / expected_status.lower()
    metric_root.mkdir()
    monkeypatch.setenv(
        publisher_runtime_admission._SUPERVISED_WORKER_TOKEN_ENV,
        token,
    )

    publisher_runtime_admission._write_supervised_admission_metric(
        metric_root,
        admission_duration_seconds=duration,
    )
    metric = publisher_runtime_admission._load_supervised_admission_metric(
        metric_root,
        expected_token=token,
        expected_deadline_seconds=900.0,
        expected_slo_seconds=300.0,
        expected_worker_pid=os.getpid(),
    )

    assert metric["value"] == expected_value
    assert metric["status"] == expected_status


def test_supervisor_normalizes_artifact_path_failure_to_json_exit_50(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_assert_isolated_launcher_flags",
        lambda: None,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_publisher_artifact_path",
        lambda: (_ for _ in ()).throw(OSError("injected cwd failure")),
    )

    result = publisher_runtime_admission._run_supervised_publisher_worker()

    assert result == 50
    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "RUNTIME_ADMISSION_FAILED"
    assert payload["error_type"] == "OSError"
    assert payload["message"] == "injected cwd failure"


def test_runtime_admission_normalizes_operational_io_failure_to_json_exit_50(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        publisher_runtime_admission,
        "verify_publisher_runtime",
        lambda _artifact: (_ for _ in ()).throw(OSError("injected SMB disconnect")),
    )

    result = publisher_runtime_admission._run_admitted_publisher_worker(
        supervisor_root=tmp_path,
        admission_started=0.0,
    )

    assert result == 50
    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "RUNTIME_ADMISSION_FAILED"
    assert payload["error_type"] == "OSError"
    assert payload["message"] == "injected SMB disconnect"


def test_prepare_private_key_is_activated_only_after_runtime_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_name = "PFC_DATA_PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_PATH"
    private_value = str(tmp_path / "request-private.pem")
    monkeypatch.delenv(private_name, raising=False)
    observations: list[tuple[str, bool]] = []

    def verify(_artifact: Path) -> dict[str, object]:
        caller_locals = sys._getframe(1).f_locals
        assert "deferred_environment" not in caller_locals
        assert private_value not in repr(caller_locals)
        observations.append(("verify", private_name in os.environ))
        return {}

    def metric(*_args, **_kwargs) -> None:
        observations.append(("metric", private_name in os.environ))

    def consume(_root: Path) -> dict[str, str]:
        observations.append(("consume", private_name in os.environ))
        return {private_name: private_value}

    def business_main() -> int:
        observations.append(("business", os.environ.get(private_name) == private_value))
        return 0

    monkeypatch.setattr(publisher_runtime_admission, "verify_publisher_runtime", verify)
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_write_supervised_admission_metric",
        metric,
    )
    monkeypatch.setattr(
        publisher_runtime_admission,
        "_consume_post_admission_environment",
        consume,
    )
    monkeypatch.setattr(publisher_cli, "main", business_main)
    try:
        result = publisher_runtime_admission._run_admitted_publisher_worker(
            supervisor_root=tmp_path,
            admission_started=time.monotonic(),
            deferred_environment_expected=True,
        )
    finally:
        os.environ.pop(private_name, None)

    assert result == 0
    assert observations == [
        ("verify", False),
        ("metric", False),
        ("consume", False),
        ("business", True),
    ]


def test_post_admission_environment_capability_is_one_shot_and_pid_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_name = "PFC_DATA_PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_PATH"
    private_value = str(tmp_path / "request-private.pem")
    token = "a" * 64
    monkeypatch.setenv("_PFC_SNAPSHOT_PUBLISHER_WORKER_TOKEN", token)
    monkeypatch.setattr(sys, "argv", [str(tmp_path / "publisher.pyz"), "prepare"])
    capability = tmp_path / ".post-admission-environment.json"
    assert not capability.exists()

    publisher_runtime_admission._write_post_admission_environment_capability(
        tmp_path,
        deferred_environment={private_name: private_value},
        token=token,
        worker_pid=os.getpid(),
    )
    environment = publisher_runtime_admission._consume_post_admission_environment(tmp_path)

    assert environment == {private_name: private_value}
    assert not capability.exists()


def test_atomic_json_publication_hides_partial_file_until_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_path = tmp_path / "sealed.json"
    entered_fsync = threading.Event()
    release_fsync = threading.Event()
    real_fsync = os.fsync

    def blocking_fsync(descriptor: int) -> None:
        real_fsync(descriptor)
        entered_fsync.set()
        assert release_fsync.wait(timeout=5)

    monkeypatch.setattr(publisher_runtime_admission.os, "fsync", blocking_fsync)
    writer = threading.Thread(
        target=publisher_runtime_admission._write_atomic_exclusive_json,
        args=(final_path, {"sealed": True}),
    )
    writer.start()
    assert entered_fsync.wait(timeout=5)
    assert not final_path.exists()
    assert list(tmp_path.glob(".*.tmp"))

    release_fsync.set()
    writer.join(timeout=5)

    assert not writer.is_alive()
    assert json.loads(final_path.read_text(encoding="ascii")) == {"sealed": True}


def test_atomic_json_reader_retries_transient_two_link_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_path = tmp_path / "sealed.json"
    linked = threading.Event()
    release_link = threading.Event()
    real_link = os.link

    def blocking_link(source: Path, target: Path) -> None:
        real_link(source, target)
        linked.set()
        assert release_link.wait(timeout=5)

    monkeypatch.setattr(publisher_runtime_admission.os, "link", blocking_link)
    writer = threading.Thread(
        target=publisher_runtime_admission._write_atomic_exclusive_json,
        args=(final_path, {"sealed": True}),
    )
    writer.start()
    assert linked.wait(timeout=5)
    assert final_path.stat().st_nlink == 2
    result: list[bytes] = []
    reader = threading.Thread(
        target=lambda: result.append(
            publisher_runtime_admission._read_after_atomic_link_publication(
                final_path,
                label="test sealed JSON",
                max_bytes=1024,
                retry_seconds=5.0,
            )
        )
    )
    reader.start()
    time.sleep(0.02)
    assert reader.is_alive()

    release_link.set()
    writer.join(timeout=5)
    reader.join(timeout=5)

    assert result == [b'{"sealed":true}']


def test_publisher_runtime_admission_rejects_nonisolated_launcher(
    publisher_dependency_root: Path,
    publisher_artifact: Path,
) -> None:
    environment = os.environ.copy()
    environment[RUNTIME_DEPENDENCY_ROOT_ENV] = str(publisher_dependency_root)
    result = subprocess.run(
        [sys.executable, str(publisher_artifact), "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 50
    payload = json.loads(result.stderr)
    assert payload["status"] == "RUNTIME_ADMISSION_FAILED"
    assert "-I -S -B" in payload["message"]


def test_publisher_runtime_admission_rejects_dependency_tree_attacks(
    tmp_path: Path,
    publisher_dependency_root: Path,
    publisher_artifact: Path,
) -> None:
    attack_root = tmp_path / "private-dependency-attack-root"
    shutil.copytree(publisher_dependency_root, attack_root)
    environment = os.environ.copy()
    environment[RUNTIME_DEPENDENCY_ROOT_ENV] = str(attack_root)

    attacks = [
        (attack_root / "foreign-1.0.dist-info" / "METADATA", b"Name: foreign\nVersion: 1.0\n"),
        (attack_root / "startup-hook.pth", b"import os\n"),
    ]
    metadata_path = next(attack_root.glob("numpy-*.dist-info/METADATA"))
    original_metadata = metadata_path.read_bytes()
    attacks.append((metadata_path, original_metadata + b"\nWheel-Tampered: true\n"))
    for attacked_path, attacked_payload in attacks:
        existed = attacked_path.exists()
        original = attacked_path.read_bytes() if existed else None
        attacked_path.parent.mkdir(parents=True, exist_ok=True)
        attacked_path.write_bytes(attacked_payload)
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    str(publisher_artifact),
                    "--help",
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 50
            assert json.loads(result.stderr)["status"] == "RUNTIME_ADMISSION_FAILED"
        finally:
            if original is None:
                attacked_path.unlink()
                if attacked_path.parent.name == "foreign-1.0.dist-info":
                    attacked_path.parent.rmdir()
            else:
                attacked_path.write_bytes(original)

    hardlink = attack_root / "foreign-hardlink.bin"
    os.link(metadata_path, hardlink)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                str(publisher_artifact),
                "--help",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 50
        assert json.loads(result.stderr)["status"] == "RUNTIME_ADMISSION_FAILED"
    finally:
        hardlink.unlink()


def test_publisher_runtime_admission_binds_exact_python_launcher(
    publisher_dependency_root: Path,
    publisher_artifact: Path,
) -> None:
    base_executable = Path(getattr(sys, "_base_executable", sys.executable))
    if base_executable.read_bytes() == Path(sys.executable).read_bytes():
        pytest.skip("the test environment does not expose a distinct base launcher")
    environment = os.environ.copy()
    environment[RUNTIME_DEPENDENCY_ROOT_ENV] = str(publisher_dependency_root)

    result = subprocess.run(
        [
            str(base_executable),
            "-I",
            "-S",
            "-B",
            str(publisher_artifact),
            "--help",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 50
    assert "fingerprint diverges" in json.loads(result.stderr)["message"]


def test_publisher_runtime_admission_rejects_tampered_zipapp_member(
    tmp_path: Path,
    publisher_dependency_root: Path,
    publisher_artifact: Path,
) -> None:
    artifact = tmp_path / "publisher.pyz"
    artifact.write_bytes(publisher_artifact.read_bytes())
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(artifact, "a") as archive:
            archive.writestr(
                "pfc_shaping/durable_artifact.py",
                b"raise RuntimeError('tampered')\n",
            )
    environment = os.environ.copy()
    environment[RUNTIME_DEPENDENCY_ROOT_ENV] = str(publisher_dependency_root)

    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(artifact), "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 50
    assert "inventory is ambiguous" in json.loads(result.stderr)["message"]


def test_offline_bootstrap_signer_has_a_separate_command_surface() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.sign_snapshot_bootstrap_authorization", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--input" in result.stdout
    assert "--output" in result.stdout


def test_publisher_cli_preserves_precommit_local_repair_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    local_repair = {
        "schema_version": "durable_artifact_repair.v2",
        "status": "LOCAL_ARTIFACT_REPAIR_REQUIRED",
        "reason": "UNOWNED_TEMPORARY_RESIDUE",
        "path": r"C:\evidence\.pfc-deadbeef.tmp",
        "operator_action": "VERIFY_WRITER_STOPPED_THEN_QUARANTINE_RESIDUE",
        "automatic_deletion_performed": False,
        "durability_classification": (
            "PROCESS_CRASH_RECOVERABLE_POWER_LOSS_UNPROVEN_ON_WINDOWS"
        ),
    }
    args = SimpleNamespace(
        command="prepare",
        source_bundle=Path(r"C:\source"),
        data_root=Path(r"C:\target"),
        operation_id="00000000-0000-4000-8000-000000000001",
        operation_created_at_utc="2026-07-16T09:00:00+00:00",
        bootstrap_authorization=None,
    )
    monkeypatch.setattr(publisher_cli, "_parse_args", lambda _: args)
    monkeypatch.setattr(
        publisher_cli,
        "assert_snapshot_publisher_runtime_identity",
        lambda **_: None,
    )
    monkeypatch.setattr(
        publisher_cli,
        "_prepare_head_expectation",
        lambda _: ("a" * 64, 1),
    )
    monkeypatch.setattr(
        publisher_cli,
        "prepare_governed_lt_snapshot",
        lambda **_: (_ for _ in ()).throw(DurableArtifactRepairRequired(local_repair)),
    )

    assert publisher_cli.main([]) == EXIT_INTEGRITY_FAILURE
    payload = json.loads(capsys.readouterr().err)
    assert payload == {
        "schema_version": "lt_snapshot_publication_cli_result.v1",
        "status": "LOCAL_ARTIFACT_REPAIR_REQUIRED",
        "command": "prepare",
        "commit_status": "NOT_COMMITTED",
        "projection_status": "LOCAL_ARTIFACT_REPAIR_REQUIRED",
        "retry_disposition": "FOLLOW_OPERATOR_ACTION_THEN_RETRY_SAME_COMMAND",
        "local_repair": local_repair,
        "production_authorization": False,
    }


def test_publisher_cli_preserves_postcommit_local_repair_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    local_repair_error = DurableArtifactRepairRequired(
        {
            "schema_version": "durable_artifact_repair.v2",
            "status": "LOCAL_ARTIFACT_REPAIR_REQUIRED",
            "reason": "REPAIR_LOCK_HELD_OR_UNOWNED",
            "path": str(tmp_path / ".pfc-repair-deadbeef.lock"),
            "operator_action": "RETRY_AFTER_WRITER_EXIT_OR_QUARANTINE_STALE_LOCK",
            "automatic_deletion_performed": False,
            "durability_classification": "PROCESS_CRASH_AND_DIRECTORY_FSYNC_DURABLE",
        }
    )
    receipt_payload = b'{"receipt_id":"receipt-1","sequence":2}'
    result = committed_snapshot_repair_result(
        command="cas",
        intent={"operation_id": "operation-1", "intent_id": "intent-1"},
        receipt={"receipt_id": "receipt-1", "sequence": 2},
        receipt_payload=receipt_payload,
        receipt_path=tmp_path / "receipt.json",
        projection_status="RECEIPT_ARCHIVE_REPAIR_REQUIRED",
        retry_disposition="RETRY_EXACT_OPERATION_LOOKUP",
        error=local_repair_error,
    )
    args = SimpleNamespace(command="cas")
    monkeypatch.setattr(publisher_cli, "_parse_args", lambda _: args)
    monkeypatch.setattr(
        publisher_cli,
        "assert_snapshot_publisher_runtime_identity",
        lambda **_: None,
    )
    monkeypatch.setattr(
        publisher_cli,
        "_commit_external_cas",
        lambda _: (_ for _ in ()).throw(SnapshotPublicationRepairRequired(result)),
    )

    assert publisher_cli.main([]) == EXIT_PROJECTION_REPAIR_REQUIRED
    payload = json.loads(capsys.readouterr().err)
    assert payload["commit_status"] == "COMMITTED"
    assert payload["local_repair"] == local_repair_error.result
    assert payload["error"] == {
        "type": "DurableArtifactRepairRequired",
        "message": "durable artifact directory requires explicit repair",
    }
