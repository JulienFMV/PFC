from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from pfc_shaping import verifier_runtime_admission
from pfc_shaping.publisher_runtime_admission import ProcessRuntimeDirectory
from pfc_shaping.verifier_package_contract import (
    VERIFIER_COMMANDS,
    VERIFIER_FORBIDDEN_FILES,
    VERIFIER_RUNTIME_PYTHON_FILES,
)
from pfc_shaping.verifier_runtime_admission import (
    VERIFIER_DEPENDENCY_ROOT_ENV,
    VERIFIER_SCRATCH_ROOT_ENV,
)
from scripts.build_lt_provider_verifier_zipapp import (
    audit_verifier_zipapp,
    build_verifier_zipapp,
)


def test_verifier_inventory_is_read_only_and_command_bounded() -> None:
    assert VERIFIER_COMMANDS == {
        "audit-acquisition",
        "audit-current-eex",
        "audit-legacy-resolution",
        "runtime-check",
    }
    assert not VERIFIER_FORBIDDEN_FILES.intersection(VERIFIER_RUNTIME_PYTHON_FILES)
    assert "pfc_shaping/pipeline/atomic_promotion.py" not in VERIFIER_RUNTIME_PYTHON_FILES
    assert "pfc_shaping/data/acquisition_signing.py" not in VERIFIER_RUNTIME_PYTHON_FILES


def test_current_eex_output_pair_requires_dedicated_governed_root(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    governed = repository / "build" / "provider-verifier-eex-results"
    output_root = governed / "run-1"
    output_root.mkdir(parents=True)

    audit, receipt = verifier_runtime_admission._admit_current_eex_output_pair(
        repository_root=repository,
        output_root=output_root,
        audit_output=output_root / "current-eex-audit.json",
        receipt_output=output_root / "runtime-receipt.json",
        protected_paths=(),
    )

    assert audit == output_root / "current-eex-audit.json"
    assert receipt == output_root / "runtime-receipt.json"


@pytest.mark.parametrize(
    "relative_root",
    (
        "build/eex-forward-local-captures/eex-ch-20260730-v1",
        "build/eex-forward-local-captures/eex-ch-20260729-v2",
        "build/workspace-local-runs/eexcap30v1",
        "build/workspace-local-runs/eexcap03",
    ),
)
def test_current_eex_output_pair_rejects_evidence_roots_without_residue(
    tmp_path: Path,
    relative_root: str,
) -> None:
    repository = tmp_path / "repo"
    (repository / "build" / "provider-verifier-eex-results").mkdir(parents=True)
    evidence_root = repository.joinpath(*relative_root.split("/"))
    evidence_root.mkdir(parents=True)
    audit = evidence_root / "current-eex-audit.json"
    receipt = evidence_root / "runtime-receipt.json"

    with pytest.raises(
        verifier_runtime_admission.PublisherRuntimeAdmissionError,
        match="output root is not governed",
    ):
        verifier_runtime_admission._admit_current_eex_output_pair(
            repository_root=repository,
            output_root=evidence_root,
            audit_output=audit,
            receipt_output=receipt,
            protected_paths=(),
        )

    assert not audit.exists()
    assert not receipt.exists()


def test_process_runtime_directory_is_writable_then_exactly_cleaned(
    tmp_path: Path,
) -> None:
    runtime = ProcessRuntimeDirectory(prefix="runtime-", parent=tmp_path)
    root = Path(runtime.name)
    payload = root / "payload.bin"
    payload.write_bytes(b"payload")
    payload.chmod(stat.S_IREAD)

    runtime.cleanup()

    assert not root.exists()


def test_verifier_worker_environment_is_minimal_and_drops_caller_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency_root = tmp_path / "dependencies"
    scratch = tmp_path / "scratch"
    supervisor = scratch / "vv-test"
    dependency_root.mkdir()
    supervisor.mkdir(parents=True)
    monkeypatch.setenv(VERIFIER_DEPENDENCY_ROOT_ENV, str(dependency_root))
    monkeypatch.setenv(VERIFIER_SCRATCH_ROOT_ENV, str(scratch))
    monkeypatch.setenv("ENTSOE_API_KEY", "must-not-cross")
    monkeypatch.setenv("DATABRICKS_TOKEN", "must-not-cross")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-cross")

    environment = verifier_runtime_admission._supervised_worker_environment(
        supervisor_root=supervisor,
        token="a" * 64,
    )

    assert environment[VERIFIER_DEPENDENCY_ROOT_ENV] == str(dependency_root)
    assert environment[VERIFIER_SCRATCH_ROOT_ENV] == str(scratch)
    assert environment["_PFC_LT_VERIFIER_SUPERVISED_WORKER"] == "1"
    assert environment["TEMP"] == str(supervisor)
    assert environment["TMP"] == str(supervisor)
    assert "ENTSOE_API_KEY" not in environment
    assert "DATABRICKS_TOKEN" not in environment
    assert "AWS_SECRET_ACCESS_KEY" not in environment


def test_verifier_cleanup_retries_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Runtime:
        def __init__(self) -> None:
            self.calls = 0

        def cleanup(self) -> None:
            self.calls += 1
            if self.calls < 3:
                raise OSError("transient cleanup failure")

    runtime = Runtime()
    monkeypatch.setattr(verifier_runtime_admission.time, "sleep", lambda _value: None)

    assert verifier_runtime_admission._cleanup_runtime_directory(runtime) is None
    assert runtime.calls == 3


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object ordering")
def test_verifier_supervisor_assigns_job_before_resuming_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    runtime_root = tmp_path / "vv-test"
    runtime_root.mkdir()

    class Runtime:
        name = str(runtime_root)

        def cleanup(self) -> None:
            events.append("cleanup")

    class Worker:
        pid = 12345

        def __init__(self) -> None:
            self.returncode: int | None = None

        def wait(self, *, timeout: float) -> int:
            events.append(("wait", timeout))
            self.returncode = 0
            return 0

        def poll(self) -> int | None:
            return self.returncode

    worker = Worker()

    def popen(command, **kwargs):
        events.append(("popen", command, kwargs))
        return worker

    monkeypatch.setattr(verifier_runtime_admission, "_assert_isolated_launcher_flags", lambda: None)
    monkeypatch.setattr(verifier_runtime_admission, "_command_name", lambda _argv: "runtime-check")
    monkeypatch.setattr(verifier_runtime_admission, "_validated_scratch_root", lambda: tmp_path)
    monkeypatch.setattr(verifier_runtime_admission, "ProcessRuntimeDirectory", lambda **_kwargs: Runtime())
    monkeypatch.setattr(verifier_runtime_admission, "read_stable_single_link_file", lambda *_args, **_kwargs: b"artifact")
    monkeypatch.setattr(verifier_runtime_admission, "_write_captured_artifact", lambda *_args: None)
    monkeypatch.setattr(verifier_runtime_admission, "_write_worker_capability", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(verifier_runtime_admission, "_supervised_worker_environment", lambda **_kwargs: {})
    monkeypatch.setattr(verifier_runtime_admission.subprocess, "Popen", popen)
    monkeypatch.setattr(verifier_runtime_admission.runtime_core, "_create_windows_kill_on_close_job", lambda: events.append("create-job") or 99)
    monkeypatch.setattr(verifier_runtime_admission.runtime_core, "_assign_process_to_windows_job", lambda job, selected: events.append(("assign", job, selected.pid)))
    monkeypatch.setattr(verifier_runtime_admission.runtime_core, "_resume_windows_process", lambda selected: events.append(("resume", selected.pid)))
    monkeypatch.setattr(verifier_runtime_admission.runtime_core, "_close_windows_handle", lambda job: events.append(("close", job)))
    monkeypatch.setattr(verifier_runtime_admission.sys, "argv", [str(tmp_path / "source.pyz"), "runtime-check"])

    assert verifier_runtime_admission._run_supervisor() == 0
    popen_event = events[0]
    assert popen_event[0] == "popen"
    assert popen_event[2]["creationflags"] & verifier_runtime_admission.runtime_core._WINDOWS_CREATE_SUSPENDED
    assert events[1:5] == [
        "create-job",
        ("assign", 99, 12345),
        ("resume", 12345),
        ("wait", verifier_runtime_admission._WORKER_TIMEOUT_SECONDS),
    ]
    assert events[-2:] == [("close", 99), "cleanup"]


def test_verifier_supervisor_timeout_terminates_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    runtime_root = tmp_path / "vv-timeout"
    runtime_root.mkdir()

    class Runtime:
        name = str(runtime_root)

        def cleanup(self) -> None:
            events.append("cleanup")

    class Worker:
        pid = 23456

        def __init__(self) -> None:
            self.returncode: int | None = None

        def wait(self, *, timeout: float) -> int:
            raise subprocess.TimeoutExpired("worker", timeout)

        def poll(self) -> int | None:
            return self.returncode

    worker = Worker()

    def terminate(selected) -> None:
        events.append("terminate")
        selected.returncode = 50

    monkeypatch.setattr(verifier_runtime_admission, "_assert_isolated_launcher_flags", lambda: None)
    monkeypatch.setattr(verifier_runtime_admission, "_command_name", lambda _argv: "runtime-check")
    monkeypatch.setattr(verifier_runtime_admission, "_validated_scratch_root", lambda: tmp_path)
    monkeypatch.setattr(verifier_runtime_admission, "ProcessRuntimeDirectory", lambda **_kwargs: Runtime())
    monkeypatch.setattr(verifier_runtime_admission, "read_stable_single_link_file", lambda *_args, **_kwargs: b"artifact")
    monkeypatch.setattr(verifier_runtime_admission, "_write_captured_artifact", lambda *_args: None)
    monkeypatch.setattr(verifier_runtime_admission, "_write_worker_capability", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(verifier_runtime_admission, "_supervised_worker_environment", lambda **_kwargs: {})
    monkeypatch.setattr(verifier_runtime_admission.subprocess, "Popen", lambda *_args, **_kwargs: worker)
    monkeypatch.setattr(verifier_runtime_admission.runtime_core, "_terminate_supervised_worker", terminate)
    if os.name == "nt":
        monkeypatch.setattr(verifier_runtime_admission.runtime_core, "_create_windows_kill_on_close_job", lambda: 88)
        monkeypatch.setattr(verifier_runtime_admission.runtime_core, "_assign_process_to_windows_job", lambda *_args: None)
        monkeypatch.setattr(verifier_runtime_admission.runtime_core, "_resume_windows_process", lambda *_args: None)
        monkeypatch.setattr(verifier_runtime_admission.runtime_core, "_close_windows_handle", lambda _job: events.append("close-job"))
    monkeypatch.setattr(verifier_runtime_admission.sys, "argv", [str(tmp_path / "source.pyz"), "runtime-check"])

    assert verifier_runtime_admission._run_supervisor() == 50
    assert events.index("terminate") < events.index("cleanup")
    if os.name == "nt":
        assert events.index("close-job") < events.index("cleanup")


def test_verifier_supervisor_keyboard_interrupt_terminates_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    runtime_root = tmp_path / "vv-interrupt"
    runtime_root.mkdir()

    class Runtime:
        name = str(runtime_root)

        def cleanup(self) -> None:
            events.append("cleanup")

    class Worker:
        pid = 34567

        def __init__(self) -> None:
            self.returncode: int | None = None

        def wait(self, *, timeout: float) -> int:
            raise KeyboardInterrupt

        def poll(self) -> int | None:
            return self.returncode

    worker = Worker()

    def terminate(selected) -> None:
        events.append("terminate")
        selected.returncode = 50

    monkeypatch.setattr(verifier_runtime_admission, "_assert_isolated_launcher_flags", lambda: None)
    monkeypatch.setattr(verifier_runtime_admission, "_command_name", lambda _argv: "runtime-check")
    monkeypatch.setattr(verifier_runtime_admission, "_validated_scratch_root", lambda: tmp_path)
    monkeypatch.setattr(verifier_runtime_admission, "ProcessRuntimeDirectory", lambda **_kwargs: Runtime())
    monkeypatch.setattr(verifier_runtime_admission, "read_stable_single_link_file", lambda *_args, **_kwargs: b"artifact")
    monkeypatch.setattr(verifier_runtime_admission, "_write_captured_artifact", lambda *_args: None)
    monkeypatch.setattr(verifier_runtime_admission, "_write_worker_capability", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(verifier_runtime_admission, "_supervised_worker_environment", lambda **_kwargs: {})
    monkeypatch.setattr(verifier_runtime_admission.subprocess, "Popen", lambda *_args, **_kwargs: worker)
    monkeypatch.setattr(verifier_runtime_admission.runtime_core, "_terminate_supervised_worker", terminate)
    if os.name == "nt":
        monkeypatch.setattr(verifier_runtime_admission.runtime_core, "_create_windows_kill_on_close_job", lambda: 77)
        monkeypatch.setattr(verifier_runtime_admission.runtime_core, "_assign_process_to_windows_job", lambda *_args: None)
        monkeypatch.setattr(verifier_runtime_admission.runtime_core, "_resume_windows_process", lambda *_args: None)
        monkeypatch.setattr(verifier_runtime_admission.runtime_core, "_close_windows_handle", lambda _job: events.append("close-job"))
    monkeypatch.setattr(verifier_runtime_admission.sys, "argv", [str(tmp_path / "source.pyz"), "runtime-check"])

    with pytest.raises(KeyboardInterrupt):
        verifier_runtime_admission._run_supervisor()

    assert events.index("terminate") < events.index("cleanup")
    if os.name == "nt":
        assert events.index("close-job") < events.index("cleanup")


def test_worker_rejects_supervisor_root_outside_governed_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "scratch"
    attacker_root = tmp_path / "attacker" / "vv-forged"
    scratch.mkdir()
    attacker_root.mkdir(parents=True)
    monkeypatch.setenv(VERIFIER_SCRATCH_ROOT_ENV, str(scratch))
    monkeypatch.setenv("_PFC_LT_VERIFIER_SUPERVISOR_ROOT", str(attacker_root))
    monkeypatch.setenv("_PFC_LT_VERIFIER_SUPERVISOR_TOKEN", "a" * 64)

    with pytest.raises(
        verifier_runtime_admission.PublisherRuntimeAdmissionError,
        match="outside governed scratch",
    ):
        verifier_runtime_admission._consume_worker_capability()


def test_runtime_receipt_is_hash_bound_but_never_runtime_authority(
    tmp_path: Path,
) -> None:
    audit_path = tmp_path / "audit.json"
    receipt_path = tmp_path / "runtime-receipt.json"
    supervisor = tmp_path / "vv-supervisor"
    acquisition = tmp_path / "acquisition"
    dependency = tmp_path / "dependency"
    scratch = tmp_path / "scratch"
    supervisor.mkdir()
    acquisition.mkdir()
    dependency.mkdir()
    scratch.mkdir()
    evidence_payload = b"provider evidence"
    manifest_payload = b'{"schema_version":"fixture"}'
    (acquisition / "evidence.bin").write_bytes(evidence_payload)
    (acquisition / "acquisition-build-manifest.json").write_bytes(manifest_payload)
    audit = {
        "schema_version": "local_provider_acquisition_audit.v4",
        "status": "VERIFIED_LOCAL_QUARANTINE_NOT_PRODUCTION",
        "production_authorization": False,
        "acquisition_directory": str(acquisition),
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "artifacts": {
            "evidence.bin": {
                "sha256": hashlib.sha256(evidence_payload).hexdigest(),
                "size_bytes": len(evidence_payload),
            }
        },
    }
    audit_payload = json.dumps(
        audit,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii") + b"\n"
    audit_path.write_bytes(audit_payload)

    receipt = verifier_runtime_admission._write_runtime_observation_receipt(
        command="audit-acquisition",
        audit_result=audit,
        audit_output=audit_path,
        receipt_output=receipt_path,
        source_revision="a" * 64,
        dependency_versions={"pandas": "2.3.3"},
        runtime_observation={"status": "PASS"},
        supervisor_root=supervisor,
        protected_paths=(acquisition, dependency, scratch, supervisor),
    )

    assert receipt["status"] == "LOCAL_RUNTIME_OBSERVATION_NOT_AUTHORITY"
    assert receipt["runtime_authority"] is False
    assert receipt["production_authorization"] is False
    assert receipt["audit_result"]["sha256"] == hashlib.sha256(audit_payload).hexdigest()
    assert json.loads(receipt_path.read_bytes()) == receipt


@pytest.mark.parametrize(
    ("audit_location", "receipt_location"),
    [
        ("supervisor", "external"),
        ("external", "acquisition"),
        ("external", "prior"),
        ("dependency", "external"),
        ("external", "scratch"),
    ],
)
def test_verifier_rejects_outputs_inside_evidence_or_runtime_roots(
    tmp_path: Path,
    audit_location: str,
    receipt_location: str,
) -> None:
    roots = {
        name: tmp_path / name
        for name in (
            "supervisor",
            "acquisition",
            "prior",
            "dependency",
            "scratch",
            "external",
        )
    }
    for root in roots.values():
        root.mkdir()

    with pytest.raises(
        verifier_runtime_admission.PublisherRuntimeAdmissionError,
        match="overlaps protected",
    ):
        verifier_runtime_admission._admit_business_output_pair(
            audit_output=roots[audit_location] / "audit.json",
            receipt_output=roots[receipt_location] / "receipt.json",
            protected_paths=(
                roots["supervisor"],
                roots["acquisition"],
                roots["prior"],
                roots["dependency"],
                roots["scratch"],
            ),
        )


def test_runtime_receipt_rejects_evidence_mutated_after_business_audit(
    tmp_path: Path,
) -> None:
    acquisition = tmp_path / "acquisition"
    supervisor = tmp_path / "supervisor"
    acquisition.mkdir()
    supervisor.mkdir()
    evidence_path = acquisition / "evidence.bin"
    manifest_path = acquisition / "acquisition-build-manifest.json"
    evidence_payload = b"original"
    manifest_payload = b"manifest"
    evidence_path.write_bytes(evidence_payload)
    manifest_path.write_bytes(manifest_payload)
    audit = {
        "schema_version": "local_provider_acquisition_audit.v4",
        "status": "VERIFIED_LOCAL_QUARANTINE_NOT_PRODUCTION",
        "production_authorization": False,
        "acquisition_directory": str(acquisition),
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "artifacts": {
            "evidence.bin": {
                "sha256": hashlib.sha256(evidence_payload).hexdigest(),
                "size_bytes": len(evidence_payload),
            }
        },
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_bytes(
        json.dumps(audit, sort_keys=True, separators=(",", ":")).encode("ascii")
        + b"\n"
    )
    evidence_path.write_bytes(b"mutated!")

    with pytest.raises(
        verifier_runtime_admission.PublisherRuntimeAdmissionError,
        match="changed after business audit",
    ):
        verifier_runtime_admission._write_runtime_observation_receipt(
            command="audit-acquisition",
            audit_result=audit,
            audit_output=audit_path,
            receipt_output=tmp_path / "receipt.json",
            source_revision="a" * 64,
            dependency_versions={"pandas": "2.3.3"},
            runtime_observation={"status": "PASS"},
            supervisor_root=supervisor,
            protected_paths=(acquisition, supervisor),
        )

    assert not (tmp_path / "receipt.json").exists()


def test_verifier_auditor_rejects_decompression_bomb(tmp_path: Path) -> None:
    artifact = tmp_path / "bomb.pyz"
    with zipfile.ZipFile(artifact, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "oversized.py",
            b"x" * (16 * 1024 * 1024 + 1),
        )

    with pytest.raises(ValueError, match="decompression budget"):
        audit_verifier_zipapp(artifact)


@pytest.mark.slow
def test_real_verifier_zipapp_is_reproducible_and_has_exact_sys_path(
    tmp_path: Path,
) -> None:
    wheelhouse, dependency_root, receipt = _real_closure()
    first = tmp_path / "provider-verifier-a.pyz"
    second = tmp_path / "provider-verifier-b.pyz"
    first_audit = build_verifier_zipapp(
        first,
        dependency_root,
        receipt,
        wheelhouse,
    )
    second_audit = build_verifier_zipapp(
        second,
        dependency_root,
        receipt,
        wheelhouse,
    )
    assert first.read_bytes() == second.read_bytes()
    assert first_audit["artifact_sha256"] == second_audit["artifact_sha256"]
    assert audit_verifier_zipapp(first) == first_audit

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    environment = dict(os.environ)
    environment[VERIFIER_DEPENDENCY_ROOT_ENV] = str(dependency_root)
    environment[VERIFIER_SCRATCH_ROOT_ENV] = str(scratch)
    for name in tuple(environment):
        if name.upper().startswith(("PFC_", "FMV_")) and (
            "SIGNING_PRIVATE_KEY" in name.upper()
            or "ANCHOR_MTLS_KEY" in name.upper()
        ):
            environment.pop(name)
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(first), "runtime-check"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stderr
    admission = json.loads(completed.stdout)
    assert admission["status"] == "PASS"
    assert admission["runtime_authority"] is False
    assert admission["supervisor_attestation"] == (
        "UNSIGNED_LOCAL_PROCESS_OBSERVATION_NOT_AUTHORITY"
    )
    assert admission["captured_artifact_sys_path_count"] == 1
    assert admission["source_artifact_sys_path_count"] == 0
    assert admission["captured_dependency_root_sys_path_count"] == 1
    assert admission["source_dependency_root_sys_path_count"] == 0
    assert admission["captured_artifact_sha256"] == hashlib.sha256(
        first.read_bytes()
    ).hexdigest()
    assert list(scratch.iterdir()) == []


def _real_closure() -> tuple[Path, Path, Path]:
    names = {
        "wheelhouse": "PFC_TEST_PUBLISHER_WHEELHOUSE",
        "dependency_root": "PFC_TEST_PUBLISHER_DEPENDENCY_ROOT",
        "receipt": "PFC_TEST_PUBLISHER_DEPENDENCY_RECEIPT",
    }
    selected = {
        label: Path(os.environ.get(environment_name, "")).expanduser()
        for label, environment_name in names.items()
    }
    if not all(path.is_absolute() and path.exists() for path in selected.values()):
        pytest.skip("real locked dependency closure is not configured")
    return (
        selected["wheelhouse"],
        selected["dependency_root"],
        selected["receipt"],
    )
