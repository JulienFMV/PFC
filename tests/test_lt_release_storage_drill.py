from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

import pfc_shaping.pipeline.storage_drill as storage_drill
import scripts.run_lt_release_storage_drill as storage_drill_script
from pfc_shaping.path_safety import path_is_link
from pfc_shaping.pipeline.atomic_promotion import PromotionError
from pfc_shaping.pipeline.storage_drill import (
    DRILL_NAMESPACE,
    REPORT_FILE,
    StorageDrillError,
    run_storage_drill,
)
from scripts.run_lt_release_storage_drill import main


def test_storage_drill_exercises_core_semantics_without_promotion(tmp_path: Path) -> None:
    result = run_storage_drill(
        drill_root=tmp_path,
        run_id="unit-core",
        timeout_seconds=180,
    )

    assert result["status"] == "PASS"
    assert result["promotion_ready"] is False
    assert result["production_authorization"] is False
    assert result["evidence_scope"] == "STORAGE_DRILL_ONLY"
    assert result["non_promotional"] is True
    checks = {item["name"]: item for item in result["checks"]}
    assert checks["crash_after_hardlink_recovery"]["status"] == "PASS"
    assert checks["multi_process_exclusive_hardlink_contention"]["status"] == "PASS"
    assert checks["transition_lock_contention"]["status"] == "PASS"
    assert checks["atomic_replace_visibility"]["status"] == "PASS"
    assert checks["directory_finalize_replace_visibility"]["status"] == "PASS"
    assert checks["alias_samefile_and_visibility"]["status"] == "UNSUPPORTED"
    assert checks["distributed_multi_host_smb_semantics"]["status"] == "UNSUPPORTED"
    assert checks["external_control_attestations"]["status"] == "UNSUPPORTED"
    assert set(result["tested_code_sha256"]) == {
        "atomic_promotion.py",
        "storage_drill.py",
        "path_safety.py",
        "run_lt_release_storage_drill.py",
    }

    report_path = tmp_path / DRILL_NAMESPACE / "unit-core" / REPORT_FILE
    archived = json.loads(report_path.read_text(encoding="utf-8"))
    assert archived["status"] == "PASS"
    assert archived["promotion_ready"] is False
    assert report_path.stat().st_nlink == 1
    actual_files = {
        path.relative_to(report_path.parent).as_posix()
        for path in report_path.parent.rglob("*")
        if path.is_file()
    }
    reported_files = {item["path"] for item in archived["artifact_inventory"]}
    assert actual_files == reported_files | set(archived["artifact_inventory_exclusions"])


def test_storage_drill_rejects_a_trivial_same_spelling_alias(tmp_path: Path) -> None:
    result = run_storage_drill(
        drill_root=tmp_path,
        run_id="unit-alias",
        alias_roots=[tmp_path],
        timeout_seconds=20,
    )

    alias = next(
        item for item in result["checks"] if item["name"] == "alias_samefile_and_visibility"
    )
    assert result["status"] == "FAIL"
    assert alias["status"] == "FAIL"
    assert "lexically distinct" in alias["details"]["error"]


def test_storage_drill_requires_configured_it_alias_classes(tmp_path: Path) -> None:
    result = run_storage_drill(
        drill_root=tmp_path,
        run_id="unit-required-alias",
        required_alias_classes=["drive", "unc_short", "unc_fqdn"],
        timeout_seconds=20,
    )

    alias = next(
        item for item in result["checks"] if item["name"] == "alias_samefile_and_visibility"
    )
    assert result["status"] == "FAIL"
    assert alias["required"] is True
    assert alias["status"] == "FAIL"


def test_storage_drill_lock_probe_ignores_large_production_wait_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PFC_PROMOTION_LOCK_WAIT_SECONDS", "300")

    result = run_storage_drill(
        drill_root=tmp_path,
        run_id="unit-lock-wait",
        timeout_seconds=20,
    )

    lock = next(
        item for item in result["checks"] if item["name"] == "transition_lock_contention"
    )
    assert lock["status"] == "PASS"


def test_storage_drill_refuses_relative_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(StorageDrillError, match="absolute"):
        run_storage_drill(drill_root=Path("relative"), run_id="unit-relative")


def test_storage_drill_refuses_reusing_a_run_directory(tmp_path: Path) -> None:
    (tmp_path / DRILL_NAMESPACE / "existing").mkdir(parents=True)

    with pytest.raises(StorageDrillError, match="already exists"):
        run_storage_drill(drill_root=tmp_path, run_id="existing")


def test_storage_drill_refuses_a_governed_release_root(tmp_path: Path) -> None:
    (tmp_path / "release_domain.json").write_text("{}", encoding="utf-8")

    with pytest.raises(StorageDrillError, match="inside a governed release"):
        run_storage_drill(drill_root=tmp_path, run_id="release-root")


def test_storage_drill_refuses_any_descendant_of_a_governed_release(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    nested = release / "candidates" / "real-run" / "evidence"
    nested.mkdir(parents=True)
    (release / "release_domain.json").write_text("{}", encoding="utf-8")

    with pytest.raises(StorageDrillError, match="inside a governed release"):
        run_storage_drill(drill_root=nested, run_id="nested-release")
    assert not (nested / DRILL_NAMESPACE).exists()


@pytest.mark.parametrize("run_id", ["CON", "LPT1.txt", "trailing."])
def test_storage_drill_refuses_windows_reserved_run_ids(
    tmp_path: Path,
    run_id: str,
) -> None:
    with pytest.raises(StorageDrillError, match="Windows"):
        run_storage_drill(drill_root=tmp_path, run_id=run_id)


def test_storage_drill_refuses_exposed_private_key_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PFC_PROMOTION_EVENT_SIGNING_PRIVATE_KEY_PATH", "secret.pem")

    with pytest.raises(StorageDrillError, match="private-key"):
        run_storage_drill(drill_root=tmp_path, run_id="private-key")
    assert not (tmp_path / DRILL_NAMESPACE).exists()


def test_exclusive_worker_does_not_misclassify_storage_failure_as_contention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / DRILL_NAMESPACE / "worker-failure"
    directory = run_dir / "exclusive"
    directory.mkdir(parents=True)
    barrier = directory / "start.barrier"
    barrier.write_bytes(b"start\n")
    ready = directory / "writer.ready"

    def fail_storage(_path: Path, _payload: bytes) -> None:
        raise PromotionError("fsync failed")

    monkeypatch.setattr(storage_drill, "_write_bytes_exclusive_fsync", fail_storage)
    args = argparse.Namespace(
        operation="exclusive-write",
        run_dir=str(run_dir),
        target=str(directory / "winner.bin"),
        barrier=str(barrier),
        payload_hex=b"payload".hex(),
        ready=str(ready),
        release=None,
        result=None,
        allowed_hex=[],
    )

    assert storage_drill._worker_main(args) == 18


def test_worker_and_inventory_reject_internal_reparse_path(tmp_path: Path) -> None:
    run_dir = tmp_path / DRILL_NAMESPACE / "linked-worker"
    outside = tmp_path / "outside"
    run_dir.mkdir(parents=True)
    outside.mkdir()
    linked = run_dir / "linked"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory reparse points are unavailable: {exc}")

    with pytest.raises(StorageDrillError, match="cannot contain a link"):
        storage_drill._worker_path(run_dir, linked / "escaped.bin")
    with pytest.raises(StorageDrillError, match="cannot be a link"):
        storage_drill._artifact_inventory(run_dir)
    assert list(outside.iterdir()) == []


def test_worker_and_inventory_reject_external_hardlink(tmp_path: Path) -> None:
    run_dir = tmp_path / DRILL_NAMESPACE / "hardlinked-worker"
    run_dir.mkdir(parents=True)
    outside = tmp_path / "outside-result.json"
    outside.write_bytes(b"outside\n")
    linked = run_dir / "result.json"
    os.link(outside, linked)

    with pytest.raises(StorageDrillError, match="cannot have hardlinks"):
        storage_drill._worker_path(run_dir, linked)
    with pytest.raises(StorageDrillError, match="forbidden hardlinks"):
        storage_drill._artifact_inventory(run_dir)
    assert outside.read_bytes() == b"outside\n"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (r"H:\\release", "drive"),
        (r"\\server\share\release", "unc_short"),
        (r"\\server.example.ch\share\release", "unc_fqdn"),
        (r"\\127.0.0.1\share\release", "invalid"),
        (r"\\[::1]\share\release", "invalid"),
        (r"\\server.\share\release", "invalid"),
        (r"\\?\C:\release", "invalid"),
        (r"\\.\C:\release", "invalid"),
    ],
)
def test_alias_classifier_requires_genuine_windows_spellings(
    value: str,
    expected: str,
) -> None:
    assert storage_drill._alias_class(value) == expected


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse-point contract")
def test_path_safety_detects_python311_windows_reparse_attribute() -> None:
    class ReparsePath:
        @staticmethod
        def lstat() -> object:
            return type(
                "Metadata",
                (),
                {
                    "st_mode": stat.S_IFDIR,
                    "st_file_attributes": 0x400,
                },
            )()

    assert path_is_link(ReparsePath()) is True  # type: ignore[arg-type]


def test_supervised_probe_kills_exact_worker_on_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / DRILL_NAMESPACE / "timeout-worker"
    run_dir.mkdir(parents=True)

    def hanging_command(_operation: str, _run_dir: Path, *_arguments: str) -> list[str]:
        return [sys.executable, "-c", "import time; time.sleep(60)"]

    monkeypatch.setattr(storage_drill, "_worker_command", hanging_command)
    started = time.monotonic()

    check = storage_drill._capture_supervised_probe(
        "atomic_replace_visibility",
        run_dir=run_dir,
        timeout_seconds=0.1,
        required=True,
    )

    assert check["status"] == "FAIL"
    assert "supervised deadline" in check["details"]["error"]
    assert time.monotonic() - started < 5


def test_supervised_probe_records_process_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / DRILL_NAMESPACE / "cleanup-failure"
    run_dir.mkdir(parents=True)

    def completed_command(_operation: str, _run_dir: Path, *_arguments: str) -> list[str]:
        return [sys.executable, "-c", "pass"]

    def fail_cleanup(_worker: object) -> None:
        raise StorageDrillError("process group did not drain")

    monkeypatch.setattr(storage_drill, "_worker_command", completed_command)
    monkeypatch.setattr(storage_drill, "_terminate_worker_tree", fail_cleanup)

    check = storage_drill._capture_supervised_probe(
        "atomic_replace_visibility",
        run_dir=run_dir,
        timeout_seconds=1,
        required=True,
    )

    assert check["status"] == "FAIL"
    assert check["details"]["process_cleanup_status"] == "FAIL"


def test_supervised_probe_cleans_up_after_communication_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / DRILL_NAMESPACE / "communication-failure"
    run_dir.mkdir(parents=True)
    cleanup_calls: list[object] = []

    class BrokenWorker:
        returncode = None

        @staticmethod
        def communicate(*, timeout: float) -> tuple[bytes, bytes]:
            raise OSError(f"pipe failed after {timeout}")

    worker = BrokenWorker()
    monkeypatch.setattr(storage_drill, "_start_worker", lambda *_a, **_k: worker)
    monkeypatch.setattr(
        storage_drill,
        "_terminate_worker_tree",
        lambda selected: cleanup_calls.append(selected),
    )

    check = storage_drill._capture_supervised_probe(
        "atomic_replace_visibility",
        run_dir=run_dir,
        timeout_seconds=1,
        required=True,
    )

    assert check["status"] == "FAIL"
    assert "communication failed" in check["details"]["error"]
    assert cleanup_calls == [worker]


@pytest.mark.parametrize("cancellation_source", ["communicate", "cleanup"])
def test_supervised_probe_preserves_operator_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancellation_source: str,
) -> None:
    run_dir = tmp_path / DRILL_NAMESPACE / f"cancel-{cancellation_source}"
    run_dir.mkdir(parents=True)

    class Worker:
        returncode = 0

        @staticmethod
        def communicate(*, timeout: float) -> tuple[bytes, bytes]:
            if cancellation_source == "communicate":
                raise KeyboardInterrupt
            return b"", b""

    worker = Worker()
    monkeypatch.setattr(storage_drill, "_start_worker", lambda *_a, **_k: worker)

    def cleanup(_worker: object) -> None:
        if cancellation_source == "cleanup":
            raise KeyboardInterrupt
        raise StorageDrillError("secondary cleanup failure")

    monkeypatch.setattr(storage_drill, "_terminate_worker_tree", cleanup)

    with pytest.raises(KeyboardInterrupt):
        storage_drill._capture_supervised_probe(
            "atomic_replace_visibility",
            run_dir=run_dir,
            timeout_seconds=1,
            required=True,
        )


def test_cleanup_failure_aborts_before_following_probes_or_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def failed_probe(name: str, **_kwargs: object) -> dict[str, object]:
        calls.append(name)
        return {
            "name": name,
            "required": True,
            "status": "FAIL",
            "details": {
                "error": "group did not drain",
                "process_cleanup_status": "FAIL",
            },
        }

    monkeypatch.setattr(storage_drill, "_capture_supervised_probe", failed_probe)

    with pytest.raises(StorageDrillError, match="fatal process cleanup failure"):
        run_storage_drill(
            drill_root=tmp_path,
            run_id="fatal-cleanup",
            timeout_seconds=20,
        )

    assert calls == ["crash_after_hardlink_recovery"]
    run_dir = tmp_path / DRILL_NAMESPACE / "fatal-cleanup"
    assert not (run_dir / REPORT_FILE).exists()


def test_supervised_process_tree_stops_grandchild_heartbeat(tmp_path: Path) -> None:
    run_dir = tmp_path / DRILL_NAMESPACE / "grandchild-worker"
    run_dir.mkdir(parents=True)
    heartbeat = run_dir / "heartbeat.bin"
    worker = storage_drill._start_worker(
        "test-hang-tree",
        run_dir,
        "--result",
        str(heartbeat),
        supervised_tree=True,
    )
    try:
        storage_drill._wait_for_file(heartbeat, worker=worker, timeout_seconds=10)
        time.sleep(0.1)
        storage_drill._terminate_worker_tree(worker)
        size_after_stop = heartbeat.stat().st_size
        time.sleep(0.2)
        assert heartbeat.stat().st_size == size_after_stop
    finally:
        storage_drill._terminate_worker_tree(worker)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group escalation contract")
def test_posix_supervision_kills_grandchild_ignoring_sigterm(tmp_path: Path) -> None:
    run_dir = tmp_path / DRILL_NAMESPACE / "posix-grandchild-worker"
    run_dir.mkdir(parents=True)
    heartbeat = run_dir / "heartbeat.bin"
    worker = storage_drill._start_worker(
        "test-hang-tree-ignore-term",
        run_dir,
        "--result",
        str(heartbeat),
        supervised_tree=True,
    )
    try:
        storage_drill._wait_for_file(heartbeat, worker=worker, timeout_seconds=10)
        storage_drill._terminate_worker_tree(worker)
        size_after_stop = heartbeat.stat().st_size
        time.sleep(0.2)
        assert heartbeat.stat().st_size == size_after_stop
    finally:
        storage_drill._terminate_worker_tree(worker)


def test_storage_drill_cli_preserves_non_promotional_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "--drill-root",
            str(tmp_path),
            "--run-id",
            "unit-cli",
            "--timeout-seconds",
            "180",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "PASS"
    assert output["promotion_ready"] is False


def test_storage_drill_cli_emits_versioned_emergency_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_before_archive(**_kwargs: object) -> dict[str, object]:
        raise PromotionError("target volume unavailable")

    monkeypatch.setattr(storage_drill_script, "run_storage_drill", fail_before_archive)
    token = "unit-parent-handshake"
    monkeypatch.setenv(storage_drill_script.SUPERVISION_TOKEN_ENV, token)

    exit_code = storage_drill_script.main(
        [
            "--drill-root",
            str(tmp_path),
            "--_run-child",
            "--_supervision-token",
            token,
        ]
    )

    assert exit_code == 2
    output = json.loads(capsys.readouterr().out)
    assert output["schema_version"] == "lt_release_storage_drill_error.v1"
    assert output["promotion_ready"] is False
    assert output["production_authorization"] is False


def test_storage_drill_rejects_direct_internal_child_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = storage_drill_script.main(
        ["--drill-root", str(tmp_path), "--_run-child"]
    )

    assert exit_code == 2
    output = json.loads(capsys.readouterr().out)
    assert "parent handshake" in output["error"]
    assert not (tmp_path / DRILL_NAMESPACE).exists()


def test_public_cli_does_not_relay_pass_when_cleanup_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CompletedChild:
        returncode = 0

        @staticmethod
        def communicate(*, timeout: float) -> tuple[bytes, bytes]:
            assert timeout == 10
            return b'{"status":"PASS"}\n', b""

    def fail_cleanup(_worker: object) -> None:
        raise StorageDrillError("process group did not drain")

    monkeypatch.setattr(
        storage_drill_script.subprocess,
        "Popen",
        lambda *_args, **_kwargs: CompletedChild(),
    )
    monkeypatch.setattr(storage_drill_script, "_terminate_worker_tree", fail_cleanup)
    args = argparse.Namespace(global_timeout_seconds=10, timeout_seconds=1)

    exit_code = storage_drill_script._run_supervised_cli(
        ["--drill-root", str(tmp_path)],
        args,
    )

    assert exit_code == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "FAIL"
    assert output["process_cleanup_status"] == "FAIL"


def test_public_cli_cleans_up_after_communication_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_calls: list[object] = []

    class BrokenChild:
        returncode = None

        @staticmethod
        def communicate(*, timeout: float) -> tuple[bytes, bytes]:
            raise OSError(f"pipe failed after {timeout}")

    child = BrokenChild()
    monkeypatch.setattr(
        storage_drill_script.subprocess,
        "Popen",
        lambda *_args, **_kwargs: child,
    )
    monkeypatch.setattr(
        storage_drill_script,
        "_terminate_worker_tree",
        lambda selected: cleanup_calls.append(selected),
    )
    args = argparse.Namespace(global_timeout_seconds=10, timeout_seconds=1)

    exit_code = storage_drill_script._run_supervised_cli(
        ["--drill-root", str(tmp_path)],
        args,
    )

    assert exit_code == 2
    output = json.loads(capsys.readouterr().out)
    assert "communication failed" in output["error"]
    assert cleanup_calls == [child]


@pytest.mark.parametrize("cancellation_source", ["communicate", "cleanup"])
def test_public_cli_preserves_operator_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancellation_source: str,
) -> None:
    class Child:
        returncode = 0

        @staticmethod
        def communicate(*, timeout: float) -> tuple[bytes, bytes]:
            if cancellation_source == "communicate":
                raise KeyboardInterrupt
            return b"", b""

    monkeypatch.setattr(
        storage_drill_script.subprocess,
        "Popen",
        lambda *_args, **_kwargs: Child(),
    )

    def cleanup(_worker: object) -> None:
        if cancellation_source == "cleanup":
            raise KeyboardInterrupt
        raise StorageDrillError("secondary cleanup failure")

    monkeypatch.setattr(storage_drill_script, "_terminate_worker_tree", cleanup)
    args = argparse.Namespace(global_timeout_seconds=10, timeout_seconds=1)

    with pytest.raises(KeyboardInterrupt):
        storage_drill_script._run_supervised_cli(
            ["--drill-root", str(tmp_path)],
            args,
        )
