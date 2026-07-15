from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import pfc_shaping.path_safety as path_safety
import pfc_shaping.pipeline.governed_release_cli_contract as cli_contract
from pfc_shaping.pipeline.atomic_promotion import (
    PromotionBusyError,
    PromotionConflictError,
    PromotionProjectionRepairRequired,
)
from pfc_shaping.pipeline.governed_release_cli_contract import (
    ALLOWED_PRIVATE_KEYS_BY_COMMAND,
    PRIVATE_KEY_ENVIRONMENTS,
    ReleaseCliIdentityError,
    assert_phase_private_key_scope,
)
from scripts import run_governed_lt_release as release_cli

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_governed_lt_release.py"
REQUEST_ID = "a" * 64


@pytest.fixture(autouse=True)
def _sealed_test_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_contract, "SOURCE_REVISION", "1" * 64)


def _common(tmp_path: Path, *, include_failure_root: bool = True) -> list[str]:
    arguments = [
        "--release-root",
        str(tmp_path / "release"),
        "--workflow-root",
        str(tmp_path / "workflow"),
        "--evidence-root",
        str(tmp_path / "evidence"),
        "--run-id",
        "run-001",
    ]
    if include_failure_root:
        arguments.extend(["--failure-root", str(tmp_path / "failures")])
    return arguments


def _provision_failure_root(tmp_path: Path) -> Path:
    root = tmp_path / "failures"
    root.mkdir(exist_ok=True)
    return root


def test_status_does_not_resolve_irrelevant_data_root_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FMV_DATA_ROOT", str(tmp_path / "first"))
    monkeypatch.setenv("PFC_LT_DATA_ROOT", str(tmp_path / "second"))
    monkeypatch.setattr(cli_contract, "SOURCE_REVISION", None)

    args = release_cli._parse_args(
        ["status", *_common(tmp_path, include_failure_root=False)]
    )

    assert args.command == "status"


@pytest.mark.parametrize(
    "argv",
    [
        ["register", "--expect-no-current"],
        ["audit", "--request-id", REQUEST_ID, "--data-root", "{data}"],
        ["promote", "--request-id", REQUEST_ID],
    ],
)
def test_release_mutations_require_external_failure_root(
    tmp_path: Path,
    argv: list[str],
) -> None:
    common = _common(tmp_path, include_failure_root=False)
    expanded = [value.format(data=tmp_path / "data") for value in argv]

    with pytest.raises(SystemExit) as exc:
        release_cli._parse_args([expanded[0], *common, *expanded[1:]])

    assert exc.value.code == 2


def test_rollback_requires_external_failure_root(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        release_cli._parse_args(
            [
                "rollback",
                "--release-root",
                str(tmp_path / "release"),
                "--rollback-authorization",
                str(tmp_path / "authorization.json"),
            ]
        )

    assert exc.value.code == 2


def test_failure_root_must_be_dedicated_from_workflow_root(tmp_path: Path) -> None:
    argv = _common(tmp_path)
    failure_index = argv.index("--failure-root") + 1
    argv[failure_index] = str(tmp_path / "workflow")

    with pytest.raises(ReleaseCliIdentityError, match="dedicated.*workflow_root"):
        release_cli._parse_args(["register", *argv, "--expect-no-current"])


def test_failure_root_must_be_dedicated_from_environment_key_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_PATH",
        str(tmp_path / "failures" / "trusted.pem"),
    )

    with pytest.raises(ReleaseCliIdentityError, match="environment:.*KEY_PATH"):
        release_cli._parse_args(
            ["register", *_common(tmp_path), "--expect-no-current"]
        )


def test_missing_preprovisioned_failure_root_blocks_phase_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("transition executed before failure-root readiness")

    monkeypatch.setattr(release_cli, "promote_release_request", forbidden)

    assert release_cli.main(
        ["promote", *_common(tmp_path), "--request-id", REQUEST_ID]
    ) == 50
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "INTEGRITY_FAILURE"
    assert payload["failure_manifest_status"] == "UNAVAILABLE"
    assert payload["error_type"] == "ReleaseCliIdentityError"
    assert not (tmp_path / "failures").exists()


def test_non_finite_result_is_replaced_by_strict_json_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _provision_failure_root(tmp_path)
    monkeypatch.setattr(
        release_cli,
        "promote_release_request",
        lambda **_kwargs: {"status": "PROMOTED", "invalid": float("nan")},
    )

    assert release_cli.main(
        ["promote", *_common(tmp_path), "--request-id", REQUEST_ID]
    ) == 50
    raw = capsys.readouterr().out
    payload = json.loads(raw)

    assert "NaN" not in raw
    assert payload["status"] == "INTEGRITY_FAILURE"
    assert payload["failure_manifest_status"] == "RECORDED"
    manifest = json.loads(Path(payload["failure_manifest"]).read_text(encoding="utf-8"))
    assert manifest["operation_id"] == payload["operation_id"]
    assert manifest["process_id"] == payload["process_id"] == os.getpid()


def test_arbitrary_result_object_is_rejected_without_string_coercion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class UnsafeResult:
        def __str__(self) -> str:
            return "token=must-not-reach-stdout"

    _provision_failure_root(tmp_path)
    monkeypatch.setattr(
        release_cli,
        "promote_release_request",
        lambda **_kwargs: {"status": "PROMOTED", "invalid": UnsafeResult()},
    )

    assert release_cli.main(
        ["promote", *_common(tmp_path), "--request-id", REQUEST_ID]
    ) == 50
    raw = capsys.readouterr().out
    payload = json.loads(raw)

    assert "must-not-reach-stdout" not in raw
    assert payload["status"] == "INTEGRITY_FAILURE"
    assert payload["error"] == "result is not strict-JSON serializable"
    assert payload["failure_manifest_status"] == "RECORDED"


def test_non_mapping_phase_result_becomes_auditable_integrity_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _provision_failure_root(tmp_path)
    monkeypatch.setattr(
        release_cli,
        "promote_release_request",
        lambda **_kwargs: ["not", "a", "mapping"],
    )

    assert release_cli.main(
        ["promote", *_common(tmp_path), "--request-id", REQUEST_ID]
    ) == 50
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "INTEGRITY_FAILURE"
    assert payload["error"] == "phase result must be an exact dict"
    assert payload["failure_manifest_status"] == "RECORDED"
    manifest = json.loads(Path(payload["failure_manifest"]).read_text(encoding="utf-8"))
    assert manifest["operation_id"] == payload["operation_id"]
    assert manifest["process_id"] == payload["process_id"] == os.getpid()


def test_primary_conflict_reports_lock_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _provision_failure_root(tmp_path)

    def conflict(**_kwargs: object) -> dict[str, object]:
        failure = PromotionConflictError("CAS mismatch")
        failure.lock_cleanup_status = "FAIL"
        failure.lock_cleanup_error_type = "PermissionError"
        raise failure

    monkeypatch.setattr(release_cli, "promote_release_request", conflict)

    assert release_cli.main(
        ["promote", *_common(tmp_path), "--request-id", REQUEST_ID]
    ) == 40
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "CAS_CONFLICT"
    assert payload["lock_cleanup_status"] == "FAIL"
    assert payload["lock_cleanup_error_type"] == "PermissionError"
    manifest = json.loads(Path(payload["failure_manifest"]).read_text(encoding="utf-8"))
    assert manifest["lock_cleanup_status"] == "FAIL"
    assert manifest["lock_cleanup_error_type"] == "PermissionError"


def test_error_messages_are_bounded_and_secret_assignments_are_redacted() -> None:
    message = release_cli._safe_error_message(
        RuntimeError("token=super-secret " + "x" * 5000)
    )

    assert "super-secret" not in message
    assert len(message) <= release_cli.MAX_ERROR_MESSAGE_CHARS


def test_failure_probe_requires_run_phase_directory_acl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    failure_root = _provision_failure_root(tmp_path)
    forbidden_directory = failure_root / "run-001" / "promote"
    original_mkdir = Path.mkdir

    def reject_phase_directory(path: Path, *args: object, **kwargs: object) -> None:
        if path == forbidden_directory:
            raise PermissionError("create-folders ACL denied")
        original_mkdir(path, *args, **kwargs)

    def forbidden(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("transition executed before directory ACL probe")

    monkeypatch.setattr(Path, "mkdir", reject_phase_directory)
    monkeypatch.setattr(release_cli, "promote_release_request", forbidden)

    assert release_cli.main(
        ["promote", *_common(tmp_path), "--request-id", REQUEST_ID]
    ) == 50
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "INTEGRITY_FAILURE"
    assert payload["failure_manifest_status"] == "UNAVAILABLE"
    assert not forbidden_directory.exists()


def test_audit_requires_explicit_data_root_even_with_historical_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = tmp_path / "shared"
    monkeypatch.delenv("FMV_DATA_ROOT", raising=False)
    monkeypatch.delenv("PFC_LT_DATA_ROOT", raising=False)
    monkeypatch.setenv("PFC_SHARED_DATA_ROOT", str(shared))

    with pytest.raises(SystemExit) as exc:
        release_cli._parse_args(
            [
                "audit",
                *_common(tmp_path),
                "--request-id",
                REQUEST_ID,
                "--signing-private-key",
                str(tmp_path / "private.pem"),
            ]
        )

    assert exc.value.code == 2


def test_audit_retry_parser_allows_missing_private_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in PRIVATE_KEY_ENVIRONMENTS:
        monkeypatch.delenv(name, raising=False)

    args = release_cli._parse_args(
        [
            "audit",
            *_common(tmp_path),
            "--request-id",
            REQUEST_ID,
            "--data-root",
            str(tmp_path / "data"),
        ]
    )

    assert args.signing_private_key is None


def test_register_requires_an_explicit_expected_current_contract(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        release_cli._parse_args(["register", *_common(tmp_path)])
    assert exc.value.code == 2

    no_current = release_cli._parse_args(
        ["register", *_common(tmp_path), "--expect-no-current"]
    )
    current = release_cli._parse_args(
        [
            "register",
            *_common(tmp_path),
            "--expect-current-event-id",
            "d" * 64,
        ]
    )

    assert no_current.expected_current_event_id is None
    assert current.expected_current_event_id == "d" * 64
    renewed = release_cli._parse_args(
        [
            "register",
            *_common(tmp_path),
            "--expect-no-current",
            "--request-nonce",
            "receipt-renewal-001",
        ]
    )
    assert renewed.request_nonce == "receipt-renewal-001"


def test_promote_reports_cas_conflict_with_dedicated_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def conflict(**_: object) -> dict[str, object]:
        raise PromotionConflictError("expected old head, observed new head")

    _provision_failure_root(tmp_path)
    monkeypatch.setattr(release_cli, "promote_release_request", conflict)
    exit_code = release_cli.main(
        ["promote", *_common(tmp_path), "--request-id", REQUEST_ID]
    )

    assert exit_code == 40
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "CAS_CONFLICT"


def test_promote_reports_busy_lock_with_dedicated_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def busy(**_: object) -> dict[str, object]:
        raise PromotionBusyError("transition lock is busy")

    _provision_failure_root(tmp_path)
    monkeypatch.setattr(release_cli, "promote_release_request", busy)
    exit_code = release_cli.main(
        ["promote", *_common(tmp_path), "--request-id", REQUEST_ID]
    )

    assert exit_code == 41
    assert json.loads(capsys.readouterr().out)["status"] == "BUSY"


def test_promote_reports_committed_projection_repair_with_exit_51(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    event_id = "e" * 64

    def committed(**_: object) -> dict[str, object]:
        raise PromotionProjectionRepairRequired(event_id)

    _provision_failure_root(tmp_path)
    monkeypatch.setattr(release_cli, "promote_release_request", committed)
    exit_code = release_cli.main(
        ["promote", *_common(tmp_path), "--request-id", REQUEST_ID]
    )

    assert exit_code == 51
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "POINTER_REPAIR_REQUIRED"
    assert payload["event_id"] == event_id


def test_promote_rejects_reused_transition_and_receipt_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_path, public_path = _write_keypair(tmp_path, "shared")
    monkeypatch.setenv(
        "PFC_PROMOTION_EVENT_SIGNING_PRIVATE_KEY_PATH",
        str(private_path),
    )
    monkeypatch.setenv("PFC_PROMOTION_EVENT_TRUSTED_PUBLIC_KEY_PATH", str(public_path))
    monkeypatch.setenv("PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH", str(public_path))

    with pytest.raises(ReleaseCliIdentityError, match="authority keyrings overlap"):
        assert_phase_private_key_scope("promote")


def test_promote_rejects_relative_authority_key_path_before_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in PRIVATE_KEY_ENVIRONMENTS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(
        "PFC_PROMOTION_EVENT_SIGNING_PRIVATE_KEY_PATH",
        "relative/event-private.pem",
    )
    monkeypatch.setenv(
        "PFC_PROMOTION_EVENT_TRUSTED_PUBLIC_KEY_PATH",
        str(tmp_path / "event-public.pem"),
    )

    with pytest.raises(ReleaseCliIdentityError, match="must be absolute"):
        assert_phase_private_key_scope("promote")


def test_rollback_is_a_separate_phase_with_only_event_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    authorization = tmp_path / "rollback-authorization.json"
    captured: dict[str, object] = {}

    def fake_rollback(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"schema_version": "governed_lt_rollback_result.v1", "status": "ROLLED_BACK"}

    _provision_failure_root(tmp_path)
    monkeypatch.setattr(release_cli, "rollback_release", fake_rollback)
    exit_code = release_cli.main(
        [
            "rollback",
            "--release-root",
            str(tmp_path / "release"),
            "--failure-root",
            str(tmp_path / "failures"),
            "--rollback-authorization",
            str(authorization),
        ]
    )

    assert exit_code == 0
    assert captured == {
        "release_root": tmp_path / "release",
        "rollback_authorization_path": authorization,
    }
    assert json.loads(capsys.readouterr().out)["status"] == "ROLLED_BACK"


def _write_keypair(root: Path, name: str) -> tuple[Path, Path]:
    key = Ed25519PrivateKey.generate()
    private_path = root / f"{name}-private.pem"
    public_path = root / f"{name}-public.pem"
    private_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_path, public_path


def test_cli_identity_rejects_historical_cross_authority_key_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in PRIVATE_KEY_ENVIRONMENTS:
        monkeypatch.delenv(name, raising=False)
    event_private, event_public = _write_keypair(tmp_path, "event")
    _, receipt_public = _write_keypair(tmp_path, "receipt")
    receipt_keyring = tmp_path / "receipt-keyring"
    receipt_keyring.mkdir()
    (receipt_keyring / "event.pem").write_bytes(event_public.read_bytes())
    monkeypatch.setenv("PFC_PROMOTION_EVENT_SIGNING_PRIVATE_KEY_PATH", str(event_private))
    monkeypatch.setenv("PFC_PROMOTION_EVENT_TRUSTED_PUBLIC_KEY_PATH", str(event_public))
    monkeypatch.setenv("PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH", str(receipt_public))
    monkeypatch.setenv("PFC_PROMOTION_TRUSTED_PUBLIC_KEY_DIR", str(receipt_keyring))

    with pytest.raises(ReleaseCliIdentityError, match="authority keyrings overlap"):
        assert_phase_private_key_scope("promote")


def _build_args(tmp_path: Path) -> list[str]:
    return [
        "--run-id",
        "run-001",
        "--release-root",
        str(tmp_path / "release"),
        "--failure-root",
        str(tmp_path / "failures"),
        "--reference-timestamp",
        "2026-07-14T08:00:00+00:00",
        "--build-timestamp",
        "2026-07-14T08:01:00+00:00",
        "--source-revision",
        "1" * 64,
        "--config",
        str(tmp_path / "config.yaml"),
        "--config-sha256",
        "b" * 64,
        "--input-snapshot-contract",
        str(tmp_path / "data" / "snapshots" / "generation-001" / "lt_input_snapshot.json"),
        "--input-snapshot-sha256",
        "c" * 64,
        "--input-pointer-contract",
        str(tmp_path / "data" / "views" / "pfc_lt" / "current.json"),
        "--input-pointer-sha256",
        "d" * 64,
        "--input-generation-id",
        "generation-001",
        "--data-root",
        str(tmp_path / "data"),
        "--historical-thresholds",
        str(tmp_path / "thresholds.csv"),
        "--historical-thresholds-receipt",
        str(tmp_path / "thresholds.receipt.json"),
        "--selected-lambda-decision",
        str(tmp_path / "selected-lambda.json"),
        "--selected-lambda-decision-receipt",
        str(tmp_path / "selected-lambda.receipt.json"),
        "--eex-report-path",
        str(tmp_path / "Price_Report_EEX.xlsx"),
        "--eex-acquisition-contract",
        str(tmp_path / "eex-acquisition.json"),
        "--peak-source-policy",
        "same_first",
        "--use-seasonal-hourly-shape",
    ]


def test_operations_runbook_documents_every_required_build_argument() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    cli_contract.add_build_arguments(parser)
    required_actions = [
        action
        for action in parser._actions
        if action.required and action.option_strings
    ]
    runbook = (ROOT / "pfc_shaping" / "tools" / "OPERATIONS.md").read_text(
        encoding="utf-8"
    )
    build_block = runbook.split("pfc-lt build `", maxsplit=1)[1].split(
        "```", maxsplit=1
    )[0]
    documented_options = set(re.findall(r"--[a-z0-9-]+", build_block))

    missing = [
        action.dest
        for action in required_actions
        if documented_options.isdisjoint(action.option_strings)
    ]
    assert missing == []


def test_controller_build_runs_only_builder_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []

    def fake_build(args: object) -> dict[str, object]:
        calls.append("build")
        return {
            "schema_version": "governed_lt_build_result.v1",
            "status": "CANDIDATE_STAGED_EVIDENCE_PENDING",
            "state": "CANDIDATE_STAGED_EVIDENCE_PENDING",
            "next_action": "SOURCE_HIERARCHY_POLICY_REQUIRED",
            "run_id": "run-001",
            "promotion_eligible": False,
        }

    def forbidden(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("another release phase was invoked")

    monkeypatch.setattr(release_cli, "_run_build_phase", fake_build)
    monkeypatch.setattr(release_cli, "_run_finalize_phase", forbidden)
    monkeypatch.setattr(release_cli, "_run_release_transition", forbidden)

    _provision_failure_root(tmp_path)
    exit_code = release_cli.main(["build", *_build_args(tmp_path)])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == release_cli.BUILD_POLICY_REQUIRED_EXIT
    assert calls == ["build"]
    assert payload["schema_version"] == "governed_lt_build_result.v1"
    assert payload["promotion_eligible"] is False


@pytest.mark.parametrize("phase", ["build", "finalize"])
def test_build_and_finalize_failures_are_json_and_durably_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    phase: str,
) -> None:
    def injected(_args: object) -> dict[str, object]:
        raise OSError(f"injected {phase} failure")

    if phase == "build":
        monkeypatch.setattr(release_cli, "_run_build_phase", injected)
        argv = ["build", *_build_args(tmp_path)]
    else:
        monkeypatch.setattr(release_cli, "_run_finalize_phase", injected)
        argv = [
            "finalize",
            "--release-root",
            str(tmp_path / "release"),
            "--failure-root",
            str(tmp_path / "failures"),
            "--run-id",
            "run-001",
            "--source-hierarchy-policy",
            str(tmp_path / "policy.json"),
        ]

    _provision_failure_root(tmp_path)
    exit_code = release_cli.main(argv)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    manifest_path = Path(payload["failure_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert exit_code == 50
    assert captured.err == ""
    assert payload["status"] == "INTEGRITY_FAILURE"
    assert payload["failure_manifest_status"] == "RECORDED"
    assert manifest_path.is_relative_to(tmp_path / "failures")
    assert manifest["schema_version"] == release_cli.FAILURE_SCHEMA
    assert manifest["phase"] == phase
    assert manifest["run_id"] == "run-001"
    assert manifest["exit_code"] == 50
    assert manifest["error_type"] == "OSError"
    assert manifest["promotion_eligible"] is False
    assert manifest_path.stat().st_nlink == 1


def test_failure_root_must_be_external_to_release_root(tmp_path: Path) -> None:
    args = _build_args(tmp_path)
    failure_index = args.index("--failure-root") + 1
    args[failure_index] = str(tmp_path / "release" / "failures")

    with pytest.raises(ReleaseCliIdentityError, match="external"):
        release_cli._parse_args(["build", *args])


def test_failure_root_identity_alias_overlap_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(release_cli, "paths_overlap_by_identity", lambda *_: True)

    with pytest.raises(ReleaseCliIdentityError, match="external"):
        release_cli._parse_args(["build", *_build_args(tmp_path)])


def test_release_root_link_validation_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = release_cli.assert_absolute_path_has_no_links

    def reject_release(path: str | Path) -> Path:
        if Path(path) == tmp_path / "release":
            raise ValueError("simulated release-root junction")
        return original(path)

    monkeypatch.setattr(release_cli, "assert_absolute_path_has_no_links", reject_release)

    with pytest.raises(ReleaseCliIdentityError, match="without links"):
        release_cli._parse_args(["build", *_build_args(tmp_path)])


def test_path_identity_error_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*_args: object) -> bool:
        raise OSError("simulated SMB identity outage")

    monkeypatch.setattr(path_safety.os.path, "samefile", unavailable)

    with pytest.raises(ValueError, match="cannot establish path identity"):
        path_safety.paths_overlap_by_identity(
            tmp_path / "release",
            tmp_path / "failures",
        )


@pytest.mark.parametrize(
    "path",
    [r"\\.\C:\pfc\failure", r"\\?\GLOBALROOT\Device\HarddiskVolume1\pfc"],
)
def test_windows_device_namespaces_are_rejected(path: str) -> None:
    with pytest.raises(ValueError, match="device namespaces"):
        path_safety.assert_absolute_path_has_no_links(path)


def test_failure_root_is_revalidated_before_manifest_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    failure_root = _provision_failure_root(tmp_path)
    args = SimpleNamespace(
        failure_root=failure_root,
        release_root=tmp_path / "release",
        run_id="run-001",
        command="promote",
    )
    monkeypatch.setattr(release_cli, "paths_overlap_by_identity", lambda *_: True)
    result: dict[str, object] = {"status": "CAS_CONFLICT"}

    release_cli._attach_failure_manifest(
        args,
        result=result,
        exit_code=40,
        failure=PromotionConflictError("injected conflict"),
        operation_id="a" * 32,
        process_id=os.getpid(),
    )

    assert result["failure_manifest_status"] == "UNAVAILABLE"
    assert "failure_manifest" not in result
    assert list(failure_root.iterdir()) == []
    assert not (tmp_path / "release").exists()


def test_failure_manifest_retry_recovers_prior_crash_hardlink(tmp_path: Path) -> None:
    failure_root = tmp_path / "failures"
    directory = failure_root / "run-001" / "promote"
    directory.mkdir(parents=True)
    temporary = directory / ".pfc-999-abcdef123456.tmp"
    prior = directory / "prior.json"
    temporary.write_text('{"prior":true}\n', encoding="utf-8")
    os.link(temporary, prior)

    current = release_cli._write_failure_manifest(
        failure_root=failure_root,
        release_root=tmp_path / "release",
        run_id="run-001",
        phase="promote",
        status="CAS_CONFLICT",
        exit_code=40,
        failure=PromotionConflictError("retry"),
        operation_id="a" * 32,
        process_id=os.getpid(),
    )

    assert not temporary.exists()
    assert prior.stat().st_nlink == 1
    assert current.stat().st_nlink == 1


def test_failure_manifest_unlink_error_never_reports_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure_root = tmp_path / "failures"
    directory = failure_root / "run-001" / "promote"
    directory.mkdir(parents=True)
    temporary = directory / ".pfc-999-abcdef123456.tmp"
    prior = directory / "prior.json"
    temporary.write_text('{"prior":true}\n', encoding="utf-8")
    os.link(temporary, prior)
    original_unlink = Path.unlink

    def fail_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path == temporary:
            raise OSError("injected unlink failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    result: dict[str, object] = {"status": "CAS_CONFLICT"}
    release_cli._attach_failure_manifest(
        SimpleNamespace(
            failure_root=failure_root,
            release_root=tmp_path / "release",
            run_id="run-001",
            command="promote",
        ),
        result=result,
        exit_code=40,
        failure=PromotionConflictError("retry"),
        operation_id="a" * 32,
        process_id=os.getpid(),
    )

    assert result["failure_manifest_status"] == "UNAVAILABLE"
    assert "failure_manifest" not in result


@pytest.mark.parametrize("phase", ["register", "audit", "promote", "rollback"])
def test_each_release_mutation_records_its_failure_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    phase: str,
) -> None:
    def injected(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise OSError(f"injected {phase} failure")

    if phase == "rollback":
        monkeypatch.setattr(release_cli, "rollback_release", injected)
        argv = [
            "rollback",
            "--release-root",
            str(tmp_path / "release"),
            "--failure-root",
            str(tmp_path / "failures"),
            "--rollback-authorization",
            str(tmp_path / "authorization.json"),
        ]
    else:
        monkeypatch.setattr(
            release_cli,
            {
                "register": "register_assembled_release_request",
                "audit": "audit_release_request",
                "promote": "promote_release_request",
            }[phase],
            injected,
        )
        argv = [phase, *_common(tmp_path)]
        if phase == "register":
            argv.extend(["--expect-no-current"])
        elif phase == "audit":
            argv.extend(
                ["--request-id", REQUEST_ID, "--data-root", str(tmp_path / "data")]
            )
        else:
            argv.extend(["--request-id", REQUEST_ID])

    _provision_failure_root(tmp_path)
    assert release_cli.main(argv) == 50
    payload = json.loads(capsys.readouterr().out)
    manifest = json.loads(Path(payload["failure_manifest"]).read_text(encoding="utf-8"))

    assert payload["failure_manifest_status"] == "RECORDED"
    assert manifest["phase"] == phase
    assert manifest["promotion_eligible"] is False
    assert manifest["operation_id"] == payload["operation_id"]
    assert manifest["process_id"] == payload["process_id"] == os.getpid()


def test_controller_build_rejects_source_revision_different_from_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_contract, "SOURCE_REVISION", "a" * 64)
    monkeypatch.setattr(
        cli_contract,
        "_installed_runtime_source_revision",
        lambda: "a" * 64,
    )

    with pytest.raises(SystemExit) as exc:
        release_cli._parse_args(["build", *_build_args(tmp_path)])

    assert exc.value.code == 2


def test_sealed_runtime_rejects_installed_source_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_contract, "SOURCE_REVISION", "a" * 64)
    monkeypatch.setattr(
        cli_contract,
        "_installed_runtime_source_revision",
        lambda: "b" * 64,
    )

    with pytest.raises(ReleaseCliIdentityError, match="sources do not match"):
        cli_contract.assert_installed_runtime_sealed()


def test_sealed_runtime_accepts_exact_installed_source_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_contract, "SOURCE_REVISION", "a" * 64)
    monkeypatch.setattr(
        cli_contract,
        "_installed_runtime_source_revision",
        lambda: "a" * 64,
    )

    cli_contract.assert_installed_runtime_sealed()


def test_controller_build_rejects_unsealed_source_checkout_before_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_contract, "SOURCE_REVISION", None)

    with pytest.raises(ReleaseCliIdentityError, match="sealed identity"):
        release_cli._parse_args(["build", *_build_args(tmp_path)])

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "argv",
    [
        [
            "finalize",
            "--release-root",
            "{release}",
            "--failure-root",
            "{failure}",
            "--run-id",
            "run-001",
            "--source-hierarchy-policy",
            "{policy}",
        ],
        ["register", "{common}", "--expect-no-current"],
        ["audit", "{common}", "--request-id", REQUEST_ID, "--data-root", "{data}"],
        ["promote", "{common}", "--request-id", REQUEST_ID],
        [
            "rollback",
            "--release-root",
            "{release}",
            "--failure-root",
            "{failure}",
            "--rollback-authorization",
            "{authorization}",
        ],
    ],
)
def test_mutating_commands_reject_unsealed_source_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    monkeypatch.setattr(cli_contract, "SOURCE_REVISION", None)
    expanded: list[str] = []
    for value in argv:
        if value == "{common}":
            expanded.extend(_common(tmp_path))
        else:
            expanded.append(
                value.format(
                    release=tmp_path / "release",
                    failure=tmp_path / "failures",
                    policy=tmp_path / "policy.json",
                    data=tmp_path / "data",
                    authorization=tmp_path / "authorization.json",
                )
            )

    with pytest.raises(ReleaseCliIdentityError, match="sealed identity"):
        release_cli._parse_args(expanded)


def test_controller_finalize_runs_only_finalizer_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []

    def fake_finalize(args: object) -> dict[str, object]:
        calls.append("finalize")
        return {
            "schema_version": "governed_lt_finalize_result.v1",
            "status": "CANDIDATE_FINALIZED_NOT_PROMOTED",
            "run_id": "run-001",
            "promotion_eligible": False,
        }

    def forbidden(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("another release phase was invoked")

    monkeypatch.setattr(release_cli, "_run_build_phase", forbidden)
    monkeypatch.setattr(release_cli, "_run_finalize_phase", fake_finalize)
    monkeypatch.setattr(release_cli, "_run_release_transition", forbidden)

    _provision_failure_root(tmp_path)
    exit_code = release_cli.main(
        [
            "finalize",
            "--release-root",
            str(tmp_path / "release"),
            "--failure-root",
            str(tmp_path / "failures"),
            "--run-id",
            "run-001",
            "--source-hierarchy-policy",
            str(tmp_path / "signed-policy.json"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert calls == ["finalize"]
    assert payload["schema_version"] == "governed_lt_finalize_result.v1"
    assert payload["promotion_eligible"] is False


@pytest.mark.parametrize("forbidden_command", ["all", "run", "release", "auto-promote"])
def test_controller_has_no_multi_phase_command(forbidden_command: str) -> None:
    with pytest.raises(SystemExit) as raised:
        release_cli._parse_args([forbidden_command])

    assert raised.value.code == 2


@pytest.mark.parametrize("arguments", [["--help"], ["build", "--help"]])
def test_controller_help_is_process_safe_with_conflicting_data_aliases(
    tmp_path: Path,
    arguments: list[str],
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["FMV_DATA_ROOT"] = str(tmp_path / "first")
    env["PFC_LT_DATA_ROOT"] = str(tmp_path / "second")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "build" in result.stdout
    assert "finalize" in result.stdout or arguments == ["build", "--help"]
    assert list(tmp_path.iterdir()) == []


def test_controller_build_requires_explicit_data_root_before_io(
    tmp_path: Path,
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["FMV_DATA_ROOT"] = str(tmp_path / "first")
    env["PFC_LT_DATA_ROOT"] = str(tmp_path / "second")
    arguments = _build_args(tmp_path)
    data_root_index = arguments.index("--data-root")
    del arguments[data_root_index : data_root_index + 2]

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "build", *arguments],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--data-root" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_controller_import_does_not_load_builder_runtime(tmp_path: Path) -> None:
    code = (
        "import sys; import scripts.run_governed_lt_release; "
        "assert 'pfc_shaping.pipeline.production_phases' not in sys.modules; "
        "assert 'scripts.build_lt_candidate' not in sys.modules; "
        "assert 'scripts.finalize_lt_candidate' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("command", sorted(ALLOWED_PRIVATE_KEYS_BY_COMMAND))
@pytest.mark.parametrize("private_key_env", sorted(PRIVATE_KEY_ENVIRONMENTS))
def test_phase_private_key_matrix_is_fail_closed(
    command: str,
    private_key_env: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in PRIVATE_KEY_ENVIRONMENTS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(private_key_env, str(tmp_path / "secret.pem"))

    if private_key_env in ALLOWED_PRIVATE_KEYS_BY_COMMAND[command]:
        assert_phase_private_key_scope(command)
    else:
        with pytest.raises(ReleaseCliIdentityError, match="forbidden private keys"):
            assert_phase_private_key_scope(command)


def test_controller_rejects_builder_with_authority_private_key_before_io(
    tmp_path: Path,
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    for name in PRIVATE_KEY_ENVIRONMENTS:
        env.pop(name, None)
    env["PFC_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH"] = str(
        tmp_path / "model-authority-private.pem"
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "build", *_build_args(tmp_path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 50
    payload = json.loads(result.stdout)
    assert payload["status"] == "INTEGRITY_FAILURE"
    assert "forbidden private keys" in payload["error"]
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module_name", ["build", "finalize"])
def test_direct_builder_and_finalizer_apis_enforce_identity_before_io(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    monkeypatch.setenv(
        "PFC_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH",
        "C:/forbidden/model-private.pem",
    )
    if module_name == "build":
        from scripts import build_lt_candidate as module

        call = module.build_candidate
    else:
        from scripts import finalize_lt_candidate as module

        call = module.finalize_candidate

    with pytest.raises(ReleaseCliIdentityError, match="forbidden private keys"):
        call(argparse.Namespace())


def test_direct_finalizer_api_rejects_unsealed_checkout_before_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import finalize_lt_candidate as finalizer

    monkeypatch.setattr(cli_contract, "SOURCE_REVISION", None)

    with pytest.raises(ReleaseCliIdentityError, match="installed runtime"):
        finalizer.finalize_candidate(argparse.Namespace())


@pytest.mark.parametrize("phase", ["build", "finalize"])
def test_direct_build_and_finalize_apis_reject_relative_release_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    for name in PRIVATE_KEY_ENVIRONMENTS:
        monkeypatch.delenv(name, raising=False)
    if phase == "build":
        from scripts import build_lt_candidate as module

        args = argparse.Namespace(
            source_revision="1" * 64,
            release_root=Path("relative-release"),
            failure_root=tmp_path / "failures",
        )
    else:
        from scripts import finalize_lt_candidate as module

        args = argparse.Namespace(
            release_root=Path("relative-release"),
            failure_root=tmp_path / "failures",
            source_hierarchy_policy=tmp_path / "policy.json",
        )

    with pytest.raises(ReleaseCliIdentityError, match="release_root path must be absolute"):
        module.build_candidate(args) if phase == "build" else module.finalize_candidate(args)

    assert not (tmp_path / "failures").exists()


def test_finalize_retry_replays_exact_sealed_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pfc_shaping.pipeline.atomic_promotion import CandidateBundle
    from pfc_shaping.pipeline.candidate_product_evidence import STAGED_POLICY
    from scripts import finalize_lt_candidate as finalizer

    release = tmp_path / "release"
    final = release / "candidates" / "run-001"
    sealed_policy = final / STAGED_POLICY
    sealed_policy.parent.mkdir(parents=True)
    sealed_policy.write_bytes(b'{"signed":"policy"}\n')
    supplied_policy = tmp_path / "signed-policy.json"
    supplied_policy.write_bytes(sealed_policy.read_bytes())
    bundle = CandidateBundle(
        run_id="run-001",
        path=final,
        manifest_path=final / "candidate_bundle_manifest.json",
        manifest_sha256="a" * 64,
    )
    calls: list[str] = []

    def fake_finalize(release_root: Path, *, run_id: str) -> CandidateBundle:
        calls.append(run_id)
        return bundle

    monkeypatch.setattr(finalizer, "finalize_assembled_candidate_staging", fake_finalize)
    args = argparse.Namespace(
        release_root=release,
        failure_root=tmp_path / "failures",
        run_id="run-001",
        source_hierarchy_policy=supplied_policy,
    )

    first = finalizer.finalize_candidate(args)
    second = finalizer.finalize_candidate(args)

    assert first == second
    assert calls == ["run-001", "run-001"]
    assert first["candidate_bundle_manifest_sha256"] == "a" * 64


def test_finalize_retry_rejects_different_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pfc_shaping.pipeline.atomic_promotion import CandidateBundle
    from pfc_shaping.pipeline.candidate_product_evidence import (
        STAGED_POLICY,
        CandidateProductEvidenceError,
    )
    from scripts import finalize_lt_candidate as finalizer

    release = tmp_path / "release"
    final = release / "candidates" / "run-001"
    sealed_policy = final / STAGED_POLICY
    sealed_policy.parent.mkdir(parents=True)
    sealed_policy.write_bytes(b'{"signed":"first"}\n')
    supplied_policy = tmp_path / "signed-policy.json"
    supplied_policy.write_bytes(b'{"signed":"different"}\n')
    bundle = CandidateBundle(
        run_id="run-001",
        path=final,
        manifest_path=final / "candidate_bundle_manifest.json",
        manifest_sha256="a" * 64,
    )
    monkeypatch.setattr(
        finalizer,
        "finalize_assembled_candidate_staging",
        lambda *args, **kwargs: bundle,
    )
    args = argparse.Namespace(
        release_root=release,
        failure_root=tmp_path / "failures",
        run_id="run-001",
        source_hierarchy_policy=supplied_policy,
    )

    with pytest.raises(CandidateProductEvidenceError, match="differs from sealed"):
        finalizer.finalize_candidate(args)


def test_real_builder_operation_separates_state_from_policy_next_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import build_lt_candidate as builder

    staging = tmp_path / "release" / "candidates" / ".run-001.staging"
    selected_relative = Path("evidence/independent/selected-lambda.json")
    (staging / selected_relative).parent.mkdir(parents=True)
    (staging / selected_relative).write_text("{}", encoding="utf-8")
    inputs = SimpleNamespace(config={})
    load_call: dict[str, object] = {}
    run_call: dict[str, object] = {}

    def fake_load_inputs(*args: object, **kwargs: object) -> object:
        load_call.update(kwargs)
        return inputs

    monkeypatch.setattr(builder, "create_candidate_staging", lambda *a, **k: staging)
    monkeypatch.setattr(
        builder,
        "capture_pre_run_governance_evidence",
        lambda *a, **k: {
            "artifacts": {
                "selected_lambda_decision": {"path": selected_relative.as_posix()},
                "runtime_config": {"path": "evidence/independent/runtime_config/config.yaml"},
                    "eex_forward_source": {
                        "path": "evidence/independent/eex/Price_Report_EEX.xlsx",
                        "acquisition_contract": {
                            "available_at_utc": "2026-07-14T07:59:00+00:00"
                        },
                    },
            },
            "governed_run_spec_sha256": "e" * 64,
        },
    )
    monkeypatch.setattr(builder, "load_inputs", fake_load_inputs)
    monkeypatch.setattr(builder, "monthly_solver_settings", lambda config: {})
    monkeypatch.setattr(
        builder,
        "active_monthly_curve_config_payload",
        lambda settings: {},
    )
    monkeypatch.setattr(
        builder,
        "validate_selected_lambda_decision",
        lambda *a, **k: None,
    )
    def fake_run_long_term(*args: object, **kwargs: object) -> object:
        run_call.update(kwargs)
        return object()

    monkeypatch.setattr(builder, "run_long_term_phase", fake_run_long_term)
    monkeypatch.setattr(
        builder,
        "write_long_term_candidate",
        lambda *a, **k: {"reference_timestamp": "2026-07-14T08:00:00+00:00"},
    )
    monkeypatch.setattr(
        builder,
        "assemble_candidate_derived_evidence",
        lambda *a, **k: {"derived": {"audit": {}}},
    )
    monkeypatch.setattr(
        builder,
        "assemble_candidate_product_evidence",
        lambda *a, **k: {
            "status": "SOURCE_HIERARCHY_POLICY_REQUIRED",
            "required_policy_payload": {"inventory_sha256": "a" * 64},
        },
    )
    args = argparse.Namespace(
        release_root=tmp_path / "release",
        failure_root=tmp_path / "failures",
        run_id="run-001",
        reference_timestamp="2026-07-14T08:00:00+00:00",
        build_timestamp="2026-07-14T08:01:00+00:00",
        source_revision="1" * 64,
        config=tmp_path / "config.yaml",
        config_sha256="b" * 64,
        input_snapshot_contract=(
            tmp_path / "data" / "snapshots" / "generation-001" / "lt_input_snapshot.json"
        ),
        input_snapshot_sha256="c" * 64,
        input_pointer_contract=(
            tmp_path / "data" / "views" / "pfc_lt" / "current.json"
        ),
        input_pointer_sha256="d" * 64,
        input_generation_id="generation-001",
        historical_thresholds=tmp_path / "thresholds.csv",
        historical_thresholds_receipt=tmp_path / "thresholds.receipt.json",
        selected_lambda_decision=tmp_path / "selected-lambda.json",
        selected_lambda_decision_receipt=tmp_path / "selected-lambda.receipt.json",
        eex_report_path=tmp_path / "Price_Report_EEX.xlsx",
        eex_acquisition_contract=tmp_path / "eex-acquisition.json",
        data_root=tmp_path / "data",
        peak_source_policy="same_first",
        use_seasonal_hourly_shape=True,
    )

    result = builder.build_candidate(args)

    assert result["status"] == "CANDIDATE_STAGED_EVIDENCE_PENDING"
    assert result["state"] == "CANDIDATE_STAGED_EVIDENCE_PENDING"
    assert result["next_action"] == "SOURCE_HIERARCHY_POLICY_REQUIRED"
    assert result["promotion_eligible"] is False
    assert load_call["config_path"] == (
        staging / "evidence/independent/runtime_config/config.yaml"
    )
    assert load_call["eex_report_path"] == (
        staging / "evidence/independent/eex/Price_Report_EEX.xlsx"
    )
    assert load_call["expected_input_generation_id"] == "generation-001"
    assert load_call["input_pointer_contract"] == args.input_pointer_contract
    assert run_call["peak_source_policy"] == "same_first"
    assert run_call["use_seasonal_hourly_shape"] is True
