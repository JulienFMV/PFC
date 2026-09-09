from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_workspace_local as runner

_CANONICAL_WINDOWS_E2E = (
    os.name == "nt"
    and Path.cwd().resolve() == runner._EXPECTED_ROOT.resolve()
    and Path(sys.executable).is_file()
)


def _fresh_e2e_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:9]}"


def _supervisor_command(run_id: str, *arguments: str) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-m",
        "scripts.run_workspace_local",
        "--run-id",
        run_id,
        *arguments,
    ]


def _read_json(path: Path) -> dict[str, object]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(decoded, dict)
    return decoded


def _wait_for(
    predicate,
    *,
    timeout_seconds: float = 30.0,
    interval_seconds: float = 0.02,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval_seconds)
    raise AssertionError("timed out waiting for governed E2E state")


@pytest.fixture(autouse=True)
def _stub_main_execution_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner,
        "_execution_identity",
        lambda _root, _command: {
            "runner_source": {"sha256": "a" * 64},
            "target_interpreter": {"sha256": "b" * 64},
            "git_head": "c" * 40,
            "git_branch": "test",
        },
    )


def test_prepare_run_and_environment_are_repo_local(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    run_root, paths = runner.prepare_run(root, "test-run")
    env, removed = runner.build_environment({}, run_root, paths)

    assert runner._is_within(run_root, root)
    assert all(runner._is_within(path, root) for path in paths.values())
    assert paths["pytest-root"] == root / "build" / "wpt-test-run"
    assert Path(env["TEMP"]) == paths["temp"]
    assert Path(env["HOME"]) == paths["home"]
    assert Path(env["USERPROFILE"]) == paths["home"]
    assert Path(env["APPDATA"]) == paths["appdata"]
    assert Path(env["LOCALAPPDATA"]) == paths["local-appdata"]
    assert Path(env["PROGRAMDATA"]) == paths["programdata"]
    assert Path(env["PIP_CACHE_DIR"]) == paths["pip-cache"]
    assert Path(env["CONDA_PKGS_DIRS"]) == paths["conda-pkgs"]
    assert Path(env["CONDARC"]).is_file()
    assert "register_envs: false" in Path(env["CONDARC"]).read_text(encoding="utf-8")
    assert removed == []


def test_canonical_repo_root_is_literal_c_workspace() -> None:
    assert runner._EXPECTED_ROOT == Path(r"C:\Users\jbattaglia\PFC_LT")


def test_prepare_run_rejects_namespace_reuse(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    runner.prepare_run(root, "single-use")

    with pytest.raises(runner.WorkspaceLocalError, match="already exists"):
        runner.prepare_run(root, "single-use")


def test_prepare_run_rejects_long_windows_namespace(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    with pytest.raises(runner.WorkspaceLocalError, match="at most 16"):
        runner.prepare_run(root, "publication-cas-candidate-long-label")


def test_prepare_run_records_preflight_failure(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(
        runner,
        "_probe_directory",
        lambda _path: (_ for _ in ()).throw(PermissionError("simulated ACL failure")),
    )

    with pytest.raises(PermissionError, match="simulated ACL failure"):
        runner.prepare_run(root, "acl-failure")

    receipt_path = root / "build" / "workspace-local-runs" / "acl-failure"
    receipt = (receipt_path / "execution-receipt.json").read_text(encoding="utf-8")
    assert '"status": "PREFLIGHT_FAILED"' in receipt


def test_pytest_receives_fresh_repo_local_basetemp(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    basetemp = root / "build" / "fresh-pytest"

    command = runner.prepare_command(
        ["python", "-B", "-m", "pytest", "tests/test_example.py"],
        root,
        basetemp,
    )

    assert command[command.index("--basetemp") + 1] == str(basetemp)
    assert command[command.index("--junitxml") + 1] == str(
        basetemp.parent / "pytest-results.xml"
    )
    assert command[command.index("--pfc-workspace-result-json") + 1] == str(
        basetemp.parent / "pytest-native-result.json"
    )
    assert command[command.index("-p", command.index("--junitxml")) + 1] == (
        "scripts.run_workspace_local"
    )


def test_pytest_rejects_any_caller_supplied_basetemp(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    stale = root / "build" / "stale"

    with pytest.raises(runner.WorkspaceLocalError, match="caller-supplied"):
        runner.prepare_command(
            ["python", "-m", "pytest", "--basetemp", str(stale)],
            root,
            root / "build" / "unused",
        )


@pytest.mark.parametrize("option", ["--junitxml", "--junit-xml"])
def test_pytest_rejects_any_caller_supplied_junit_output(
    tmp_path: Path,
    option: str,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    with pytest.raises(runner.WorkspaceLocalError, match="caller-supplied"):
        runner.prepare_command(
            ["python", "-m", "pytest", option, str(root / "build" / "forged.xml")],
            root,
            root / "build" / "unused",
        )


@pytest.mark.parametrize(
    "arguments",
    [
        ["-p", "attacker.plugin"],
        ["-pattacker.plugin"],
        ["-qpattacker.plugin"],
    ],
)
def test_pytest_rejects_arbitrary_explicit_plugin(
    tmp_path: Path,
    arguments: list[str],
) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    with pytest.raises(runner.WorkspaceLocalError, match="plugin is not allowlisted"):
        runner.prepare_command(
            ["python", "-m", "pytest", *arguments],
            root,
            root / "build" / "unused",
        )


def test_pytest_rejects_caller_supplied_native_result_path(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    with pytest.raises(runner.WorkspaceLocalError, match="native result"):
        runner.prepare_command(
            [
                "python",
                "-m",
                "pytest",
                "--pfc-workspace-result-json",
                str(root / "build" / "forged.json"),
            ],
            root,
            root / "build" / "unused",
        )


@pytest.mark.parametrize(
    "arguments",
    [
        ["-c", "attacker.ini"],
        ["-cattacker.ini"],
        ["-qcattacker.ini"],
        ["--config-file", "attacker.ini"],
        ["--config-file=attacker.ini"],
        ["--inifile=attacker.ini"],
        ["-o", "addopts=-p attacker.plugin"],
        ["-qoaddopts=-p attacker.plugin"],
        ["--override-ini=addopts=-p attacker.plugin"],
        ["--rootdir", "outside"],
        ["--confcutdir=outside"],
    ],
)
def test_pytest_rejects_caller_configuration_injection(
    tmp_path: Path,
    arguments: list[str],
) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    with pytest.raises(runner.WorkspaceLocalError, match="configuration override"):
        runner.prepare_command(
            ["python", "-m", "pytest", *arguments],
            root,
            root / "build" / "unused",
        )


def test_pytest_rejects_argument_file_expansion(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    argument_file = tmp_path / "attacker.args"
    argument_file.write_text("-p attacker.plugin\n", encoding="utf-8")

    with pytest.raises(runner.WorkspaceLocalError, match="argument-file"):
        runner.prepare_command(
            ["python", "-m", "pytest", f"@{argument_file}"],
            root,
            root / "build" / "unused",
        )


def test_project_executable_and_playwright_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    with pytest.raises(runner.WorkspaceLocalError, match="Python interpreter"):
        runner.prepare_command([str(root / "build" / "pfc-lt.exe")], root, root / "tmp")
    with pytest.raises(runner.WorkspaceLocalError, match="Python interpreter"):
        runner.prepare_command(["playwright.exe"], root, root / "tmp")
    with pytest.raises(runner.WorkspaceLocalError, match="not allowlisted: playwright"):
        runner.prepare_command(["python", "-m", "playwright"], root, root / "tmp")
    with pytest.raises(runner.WorkspaceLocalError, match="Python interpreter"):
        runner.prepare_command(["npx", "playwright"], root, root / "tmp")
    with pytest.raises(runner.WorkspaceLocalError, match="followed immediately"):
        runner.prepare_command(
            ["python", "-c", "import playwright", "-m", "pytest"],
            root,
            root / "tmp",
        )
    with pytest.raises(runner.WorkspaceLocalError, match="followed immediately"):
        runner.prepare_command(
            ["python", "script.py", "-m", "pytest"],
            root,
            root / "tmp",
        )


def test_pytest_marker_argument_after_module_remains_allowed(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    command = runner.prepare_command(
        ["python", "-B", "-m", "pytest", "-m", "not slow"],
        root,
        root / "build" / "fresh",
    )

    assert command[4:6] == ["-m", "not slow"]


def test_repo_local_python_runtime_is_an_allowed_tool(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    interpreter = root / "build" / "conda-runtime-v19" / "python.exe"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()

    command = runner.prepare_command(
        [str(interpreter), "-I", "-m", "scripts.check_lt_wheel_contract", "--help"],
        root,
        root / "tmp",
    )

    assert command[0] == str(interpreter)


def test_eex_datasource_v2_packaged_command_is_isolated_and_path_governed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    spec = root / "build" / "eex-datasource-v2-specs" / "capture.json"
    spec.parent.mkdir(parents=True)
    spec.write_text("{}\n", encoding="utf-8")
    interpreter = root / "build" / "conda-runtime-eex-test" / "python.exe"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"python-fixture")
    command = [
        str(interpreter),
        "-I",
        "-B",
        "-m",
        runner._EEX_DATASOURCE_MODULE,
        "--repo-root",
        str(root),
        "--spec",
        str(spec),
        "--expected-spec-sha256",
        "a" * 64,
    ]

    prepared = runner.prepare_command(command, root, root / "build" / "pytest-root")

    assert prepared[1:] == command[1:]


def test_eex_datasource_v2_packaged_command_rejects_nonisolated_flags(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    spec = root / "build" / "eex-datasource-v2-specs" / "capture.json"
    spec.parent.mkdir(parents=True)
    spec.write_text("{}\n", encoding="utf-8")

    with pytest.raises(runner.WorkspaceLocalError, match="exact isolated"):
        runner.prepare_command(
            [
                "python",
                "-B",
                "-m",
                runner._EEX_DATASOURCE_MODULE,
                "--repo-root",
                str(root),
                "--spec",
                str(spec),
                "--expected-spec-sha256",
                "a" * 64,
            ],
            root,
            root / "build" / "pytest-root",
        )


def test_eex_datasource_token_is_forwarded_only_for_exact_module() -> None:
    token = "local-test-eex-token"
    environment = {runner._EEX_DATASOURCE_TOKEN_ENV: token}
    command = ["python", "-I", "-B", "-m", runner._EEX_DATASOURCE_MODULE]

    assert runner._selected_eex_datasource_token(command, environment) == token
    assert (
        runner._selected_eex_datasource_token(
            ["python", "-I", "-B", "-m", "ruff"], environment
        )
        is None
    )
    assert (
        runner._selected_eex_datasource_token(
            command, {runner._EEX_DATASOURCE_TOKEN_ENV: "too-short"}
        )
        is None
    )


def test_supervisor_rejects_eex_token_before_workspace_or_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "local-test-eex-token"
    monkeypatch.setenv(runner._EEX_DATASOURCE_TOKEN_ENV, token)

    with pytest.raises(runner.WorkspaceLocalError, match="forwarding is disabled"):
        runner.main(
            [
                "--run-id",
                "eex-disabled",
                "--",
                "python",
                "-I",
                "-B",
                "-m",
                runner._EEX_DATASOURCE_MODULE,
            ]
        )

    assert runner._EEX_DATASOURCE_TOKEN_ENV not in runner.os.environ


def test_supervisor_rejects_invalid_eex_token_before_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(runner._EEX_DATASOURCE_TOKEN_ENV, "too-short")

    with pytest.raises(runner.WorkspaceLocalError, match="forwarding is disabled"):
        runner.main(
            [
                "--run-id",
                "eex-invalid",
                "--",
                "python",
                "-I",
                "-B",
                "-m",
                runner._EEX_DATASOURCE_MODULE,
            ]
        )

    assert runner._EEX_DATASOURCE_TOKEN_ENV not in runner.os.environ


def _launcherless_command(root: Path) -> list[str]:
    build = root / "build"
    return [
        "python",
        "-B",
        "-m",
        "scripts.build_launcherless_local_runtime",
        "--runtime-prefix",
        str(build / "conda-runtime-v9"),
        "--project-wheel",
        str(build / "wheel-dist" / "fmv_pfc_lt-0.14.0-py3-none-any.whl"),
        "--publisher-wheelhouse",
        str(build / "runtime-inputs" / "publisher-wheelhouse"),
        "--publisher-dependency-root",
        str(build / "runtime-inputs" / "publisher-closure"),
        "--publisher-receipt",
        str(build / "runtime-inputs" / "publisher-receipt.json"),
        "--additional-wheel-directory",
        str(build / "runtime-inputs" / "additional-wheels"),
        "--python-runtime-manifest",
        str(build / "runtime-inputs" / "python-runtime-manifest.json"),
        "--conda-prefix-build-receipt",
        str(build / "runtime-inputs" / "conda-prefix-build-receipt.json"),
        "--receipt-output",
        str(build / "runtime-receipts" / "runtime-v9.json"),
        "--expected-python-runtime-manifest-sha256",
        "a" * 64,
        "--expected-conda-prefix-build-receipt-sha256",
        "b" * 64,
        "--lock-path",
        str(root / "uv.lock"),
    ]


def test_launcherless_runtime_paths_must_remain_below_repo_build(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = _launcherless_command(root)
    appdata = tmp_path / "AppData" / "Local" / "pfc-lt-build" / "wheelhouse"
    command[command.index("--publisher-wheelhouse") + 1] = str(appdata)

    with pytest.raises(runner.WorkspaceLocalError, match="below repo build"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def test_launcherless_runtime_repo_local_paths_are_allowed(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    command = runner.prepare_command(
        _launcherless_command(root),
        root,
        root / "build" / "pytest",
    )

    assert command[command.index("--publisher-wheelhouse") + 1].startswith(str(root / "build"))


def _successor_core_command(root: Path, *, packaged: bool = False) -> list[str]:
    core = (
        root
        / ".planning"
        / "phases"
        / "14-lt-audit-remediation"
        / "CH-LT-SUCCESSOR-CANDIDATE-CORE-V3-20260730.json"
    )
    if packaged:
        interpreter = root / "build" / "conda-runtime-test" / "python.exe"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_bytes(b"test interpreter identity")
        return [
            str(interpreter),
            "-I",
            "-B",
            "-m",
            "pfc_shaping.cli.audit_ch_lt_successor_candidate_core",
            "--evidence-root",
            str(root),
            "--core",
            str(core),
            "--expected-core-sha256",
            runner._SUCCESSOR_CANDIDATE_CORE_SHA256,
        ]
    return [
        "python",
        "-B",
        "-m",
        "scripts.audit_ch_lt_successor_candidate_core",
        "--core",
        str(core),
        "--expected-core-sha256",
        runner._SUCCESSOR_CANDIDATE_CORE_SHA256,
    ]


def test_successor_core_local_command_is_exactly_bounded(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    command = runner.prepare_command(
        _successor_core_command(root),
        root,
        root / "build" / "pytest",
    )

    assert command[command.index("--core") + 1].startswith(str(root))


@pytest.mark.parametrize("mutation", ["wrong_core", "wrong_hash", "duplicate", "extra"])
def test_successor_core_local_command_rejects_grammar_mutations(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = _successor_core_command(root)
    if mutation == "wrong_core":
        command[command.index("--core") + 1] = str(root / "build" / "forged.json")
    elif mutation == "wrong_hash":
        command[command.index("--expected-core-sha256") + 1] = "0" * 64
    elif mutation == "duplicate":
        command.extend(["--core", command[command.index("--core") + 1]])
    else:
        command.append("--verbose")

    with pytest.raises(runner.WorkspaceLocalError):
        runner.prepare_command(command, root, root / "build" / "pytest")


def test_successor_core_packaged_command_is_not_runner_allowlisted(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = _successor_core_command(root, packaged=True)

    with pytest.raises(runner.WorkspaceLocalError, match="not allowlisted"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def test_successor_core_command_rejects_extra_python_flags(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = _successor_core_command(root)
    command.insert(command.index("-m"), "-E")

    with pytest.raises(runner.WorkspaceLocalError, match="exact Python flags"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def _dependence_power_design_command(root: Path) -> list[str]:
    design = (
        root
        / ".planning"
        / "phases"
        / "14-lt-audit-remediation"
        / "CH-LT-DEPENDENCE-POWER-DESIGN-DRAFT-V1-20260730.json"
    )
    return [
        "python",
        "-B",
        "-m",
        "scripts.audit_ch_lt_dependence_power_design",
        "--design",
        str(design),
        "--expected-design-sha256",
        runner._DEPENDENCE_POWER_DESIGN_SHA256,
    ]


def test_dependence_power_design_command_is_exactly_bounded(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    command = runner.prepare_command(
        _dependence_power_design_command(root),
        root,
        root / "build" / "pytest",
    )

    assert command[command.index("--design") + 1].startswith(str(root))


@pytest.mark.parametrize("mutation", ["wrong_design", "wrong_hash", "duplicate", "extra"])
def test_dependence_power_design_command_rejects_grammar_mutations(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = _dependence_power_design_command(root)
    if mutation == "wrong_design":
        command[command.index("--design") + 1] = str(root / "build" / "forged.json")
    elif mutation == "wrong_hash":
        command[command.index("--expected-design-sha256") + 1] = "0" * 64
    elif mutation == "duplicate":
        command.extend(["--design", command[command.index("--design") + 1]])
    else:
        command.append("--verbose")

    with pytest.raises(runner.WorkspaceLocalError):
        runner.prepare_command(command, root, root / "build" / "pytest")


def _origin_registry_protocol_command(
    root: Path, *, packaged: bool = False
) -> list[str]:
    protocol = root / runner._ORIGIN_REGISTRY_PROTOCOL
    if packaged:
        interpreter = root / "build" / "conda-runtime-test" / "python.exe"
        interpreter.parent.mkdir(parents=True, exist_ok=True)
        interpreter.write_bytes(b"test interpreter identity")
        prefix = [
            str(interpreter),
            "-I",
            "-B",
            "-m",
            "pfc_shaping.cli.audit_ch_lt_origin_registry_protocol",
            "--evidence-root",
            str(root),
        ]
    else:
        prefix = [
            "python",
            "-B",
            "-m",
            "scripts.audit_ch_lt_origin_registry_protocol",
        ]
    return prefix + [
        "--protocol",
        str(protocol),
        "--expected-protocol-sha256",
        runner._ORIGIN_REGISTRY_PROTOCOL_SHA256,
    ]


@pytest.mark.parametrize("packaged", [False, True])
def test_origin_registry_protocol_command_is_exactly_bounded(
    tmp_path: Path, packaged: bool
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = runner.prepare_command(
        _origin_registry_protocol_command(root, packaged=packaged),
        root,
        root / "build" / "pytest",
    )
    assert command[command.index("--protocol") + 1] == str(
        root / runner._ORIGIN_REGISTRY_PROTOCOL
    )


@pytest.mark.parametrize(
    "mutation", ["wrong_protocol", "wrong_hash", "duplicate", "extra", "extra_flag"]
)
def test_origin_registry_protocol_command_rejects_grammar_mutations(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = _origin_registry_protocol_command(root)
    if mutation == "wrong_protocol":
        command[command.index("--protocol") + 1] = str(root / "build" / "forged.json")
    elif mutation == "wrong_hash":
        command[command.index("--expected-protocol-sha256") + 1] = "0" * 64
    elif mutation == "duplicate":
        command.extend(["--protocol", command[command.index("--protocol") + 1]])
    elif mutation == "extra":
        command.append("--verbose")
    else:
        command.insert(command.index("-m"), "-E")

    with pytest.raises(runner.WorkspaceLocalError):
        runner.prepare_command(command, root, root / "build" / "pytest")


def _origin_target_mask_command(
    root: Path, *, action: str = "build", packaged: bool = False
) -> list[str]:
    output = root / "build" / "ch-lt-origin-target-mask-inventories" / "inventory.json"
    if packaged:
        interpreter = root / "build" / "conda-runtime-test" / "python.exe"
        interpreter.parent.mkdir(parents=True, exist_ok=True)
        interpreter.write_bytes(b"test interpreter identity")
        prefix = [
            str(interpreter),
            "-I",
            "-B",
            "-m",
            "pfc_shaping.cli.audit_ch_lt_origin_target_mask_inventory",
            "--evidence-root",
            str(root),
            action,
        ]
    else:
        prefix = [
            "python",
            "-B",
            "-m",
            "scripts.audit_ch_lt_origin_target_mask_inventory",
            action,
        ]
    if action == "build":
        return prefix + [
            "--origin-as-of-utc",
            "2026-07-30T12:00:00Z",
            "--output",
            str(output),
        ]
    return prefix + [
        "--inventory",
        str(output),
        "--expected-inventory-sha256",
        "a" * 64,
    ]


@pytest.mark.parametrize("action", ["build", "audit"])
def test_origin_target_mask_command_is_workspace_bounded(
    tmp_path: Path, action: str
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = runner.prepare_command(
        _origin_target_mask_command(root, action=action),
        root,
        root / "build" / "pytest",
    )
    path_option = "--output" if action == "build" else "--inventory"
    assert command[command.index(path_option) + 1].startswith(str(root / "build"))


@pytest.mark.parametrize("action", ["build", "audit"])
def test_packaged_origin_target_mask_command_is_workspace_bounded(
    tmp_path: Path, action: str
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = runner.prepare_command(
        _origin_target_mask_command(root, action=action, packaged=True),
        root,
        root / "build" / "pytest",
    )
    assert command[command.index("--evidence-root") + 1] == str(root)


@pytest.mark.parametrize(
    "mutation",
    ["outside", "relative", "duplicate", "extra", "bad_origin", "bad_hash", "extra_flag"],
)
def test_origin_target_mask_command_rejects_grammar_mutations(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    action = "audit" if mutation == "bad_hash" else "build"
    command = _origin_target_mask_command(root, action=action)
    if mutation == "outside":
        command[command.index("--output") + 1] = str(tmp_path / "outside.json")
    elif mutation == "relative":
        command[command.index("--output") + 1] = "build/inventory.json"
    elif mutation == "duplicate":
        command.extend(["--output", command[command.index("--output") + 1]])
    elif mutation == "extra":
        command.append("--truth-path")
    elif mutation == "bad_origin":
        command[command.index("--origin-as-of-utc") + 1] = "2026-07-30"
    elif mutation == "bad_hash":
        command[command.index("--expected-inventory-sha256") + 1] = "INVALID"
    else:
        command.insert(command.index("-m"), "-E")

    with pytest.raises(runner.WorkspaceLocalError):
        runner.prepare_command(command, root, root / "build" / "pytest")


def _local_panel_command(root: Path) -> list[str]:
    build = root / "build"
    return [
        "python",
        "-B",
        "-m",
        "scripts.build_local_intraday_calibration_panel",
        "--base-parquet",
        str(build / "panel" / "base.parquet"),
        "--expected-base-sha256",
        "a" * 64,
        "--audit-json",
        str(build / "panel" / "audit.json"),
        "--expected-audit-sha256",
        "b" * 64,
        "--audit-runtime-receipt-json",
        str(build / "panel" / "runtime-receipt.json"),
        "--expected-audit-runtime-receipt-sha256",
        "c" * 64,
        "--verifier-artifact",
        str(build / "panel" / "verifier.pyz"),
        "--expected-verifier-artifact-sha256",
        "d" * 64,
        "--start-utc",
        "2026-01-01T00:00:00Z",
        "--end-utc",
        "2026-01-02T00:00:00Z",
        "--output-parquet",
        str(build / "panel" / "output.parquet"),
        "--output-manifest",
        str(build / "panel" / "output.json"),
    ]


def test_local_panel_paths_must_remain_below_repo_build(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = _local_panel_command(root)
    command[command.index("--audit-json") + 1] = str(tmp_path / "AppData" / "audit.json")

    with pytest.raises(runner.WorkspaceLocalError, match="below repo build"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def test_local_panel_repo_build_paths_are_allowed(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    command = runner.prepare_command(
        _local_panel_command(root),
        root,
        root / "build" / "pytest",
    )

    assert command[command.index("--output-manifest") + 1].startswith(str(root / "build"))


def _local_model_command(root: Path) -> list[str]:
    output = root / "build" / "local-model-runs" / "run-a"
    return [
        "python",
        "-B",
        "-m",
        "scripts.build_local_test_ch_pfc",
        "--inventory",
        str(root / "data" / "electrification_scenarios_prod_candidate_neutralized_2030.parquet"),
        "--manifest",
        str(
            root
            / ".planning"
            / "phases"
            / "13-lt-electrification-scenario-shape"
            / "SCENARIO-GOVERNANCE-LOCAL-TEST-MANIFEST.yaml"
        ),
        "--governance-report",
        str(output / "LOCAL-TEST-GOVERNANCE-GATE.md"),
        "--vintage",
        "2026-06-12",
        "--scenarios",
        "slow,central,fast",
        "--weights",
        "0.25,0.50,0.25",
        "--market",
        "CH",
        "--start-date",
        "2026-07-31T22:00:00Z",
        "--horizon-days",
        "1614",
        "--delivery-local-end-exclusive",
        "2031-01-01T00:00:00",
        "--epex-hourly",
        str(root / "data" / "epex_hourly.parquet"),
        "--intraday-epex",
        str(
            root
            / "output"
            / "phase14"
            / "prospective_intraday_panel_20260724"
            / "epex_de_15min_complete.parquet"
        ),
        "--intraday-market",
        "DE",
        "--intraday-cutoff",
        "2025-10-01T00:00:00Z",
        "--intraday-provenance-manifest",
        str(
            root
            / "output"
            / "phase14"
            / "prospective_intraday_panel_20260724"
            / "panel-manifest.json"
        ),
        "--expected-intraday-provenance-sha256",
        "af8232e7045eb9aa685d946770a56a108e438a6a71e9cbdab47614abcc5a9e4f",
        "--require-nonflat-intraday-shape",
        "--require-direct-intraday-seasons",
        "Hiver,Printemps,Ete,Automne",
        "--forwards",
        str(root / "data" / "eex_forwards_history.parquet"),
        "--expanded-output",
        str(output / "scenario_expanded.parquet"),
        "--features-output",
        str(output / "hpfc_scenario_features.parquet"),
        "--output-dir",
        str(output),
        "--output-prefix",
        "local-pfc-smoke",
        "--fan-chart-output",
        str(output / "structural_fan_chart.parquet"),
        "--summary",
        str(output / "QUALITY.md"),
        "--completion-manifest",
        str(output / "completion_manifest.json"),
        "--enable-monthly-forward-curve-solver",
        "--monthly-solver-constraint-tolerance",
        "1e-9",
        "--monthly-solver-lambda-smooth-month",
        "1.0",
        "--monthly-solver-lambda-smooth-yoy",
        "0.25",
        "--monthly-solver-lambda-shape",
        "1.0",
        "--monthly-solver-neighbor-shrinkage",
        "0.5",
        "--monthly-solver-allow-template-structural-fallback",
        "--monthly-solver-structural-amplitude",
        "110.0",
        "--disable-cascade-trend-for-annual-only",
        "--enable-experimental-sparse-intraday-regularization",
    ]


def test_local_model_route_accepts_only_frozen_repo_local_run(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    command = runner.prepare_command(
        _local_model_command(root), root, root / "build" / "pytest"
    )

    assert command[command.index("--market") + 1] == "CH"
    assert command[command.index("--output-dir") + 1].startswith(
        str(root / "build" / "local-model-runs")
    )


def test_local_model_route_rejects_output_outside_governed_run(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = _local_model_command(root)
    command[command.index("--summary") + 1] = str(root / "QUALITY.md")

    with pytest.raises(runner.WorkspaceLocalError, match="inside --output-dir"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def test_local_model_route_rejects_rebound_protected_forward_input(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = _local_model_command(root)
    command[command.index("--forwards") + 1] = str(root / "build" / "shadow.parquet")

    with pytest.raises(runner.WorkspaceLocalError, match="canonical read-only input"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def test_local_model_route_rejects_solver_knob_drift(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = _local_model_command(root)
    command[command.index("--monthly-solver-lambda-shape") + 1] = "2.0"

    with pytest.raises(runner.WorkspaceLocalError, match="must remain frozen"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def test_laptop_model_qualification_route_is_exact(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    output = root / runner._LAPTOP_MODEL_QUALIFICATION_OUTPUT
    command = [
        "python",
        "-B",
        "-m",
        "scripts.audit_ch_lt_laptop_model_pair",
        "--output-directory",
        str(output),
    ]

    prepared = runner.prepare_command(command, root, root / "build" / "pytest")

    assert prepared[-1] == str(output)
    command[-1] = str(root / "build" / "shadow")
    with pytest.raises(runner.WorkspaceLocalError, match="canonical laptop qualification"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def test_current_local_quality_selection_audit_route_is_argument_free(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = [
        "python",
        "-B",
        "-m",
        "scripts.audit_ch_lt_local_quality_current_selection",
    ]

    prepared = runner.prepare_command(command, root, root / "build" / "pytest")
    assert prepared[0] == str(Path(runner.sys.executable).resolve())
    assert prepared[1:] == command[1:]
    with pytest.raises(runner.WorkspaceLocalError, match="no arguments"):
        runner.prepare_command(
            [*command, "--shadow"], root, root / "build" / "pytest"
        )


def test_structural_prediction_commitment_route_is_exact() -> None:
    root = runner._EXPECTED_ROOT
    output = root / runner._STRUCTURAL_PREDICTION_COMMITMENT_OUTPUT
    command = [
        "python",
        "-B",
        "-m",
        "scripts.build_ch_lt_structural_prediction_commitment",
        "--inventory",
        str(root / runner._STRUCTURAL_PREDICTION_COMMITMENT_INVENTORY),
        "--expected-inventory-sha256",
        runner._STRUCTURAL_PREDICTION_COMMITMENT_INVENTORY_SHA256,
        "--output-directory",
        str(output),
    ]

    prepared = runner.prepare_command(command, root, root / "build" / "pytest")

    assert prepared[-1] == str(output)
    command[-1] = str(root / "build" / "shadow")
    with pytest.raises(runner.WorkspaceLocalError, match="canonical structural commitment"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def test_structural_prediction_commitment_requires_only_no_bytecode_flag(
) -> None:
    root = runner._EXPECTED_ROOT
    output = root / runner._STRUCTURAL_PREDICTION_COMMITMENT_OUTPUT
    command = [
        "python",
        "-I",
        "-B",
        "-m",
        "scripts.build_ch_lt_structural_prediction_commitment",
        "--inventory",
        str(root / runner._STRUCTURAL_PREDICTION_COMMITMENT_INVENTORY),
        "--expected-inventory-sha256",
        runner._STRUCTURAL_PREDICTION_COMMITMENT_INVENTORY_SHA256,
        "--output-directory",
        str(output),
    ]

    with pytest.raises(runner.WorkspaceLocalError, match="exact Python flags"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def test_local_future_origin_selection_route_is_exact() -> None:
    root = runner._EXPECTED_ROOT
    registry = root / runner._LOCAL_FUTURE_ORIGIN_SELECTION_REGISTRY
    command = [
        "python",
        "-B",
        "-m",
        "scripts.audit_ch_lt_local_future_origin_selection",
        "--registry",
        str(registry),
        "--expected-registry-sha256",
        runner._LOCAL_FUTURE_ORIGIN_SELECTION_SHA256,
    ]

    prepared = runner.prepare_command(command, root, root / "build" / "pytest")

    assert prepared[1:] == command[1:]
    command[-1] = "0" * 64
    with pytest.raises(runner.WorkspaceLocalError, match="must match"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def test_packaged_local_future_origin_selection_route_requires_isolation(
    tmp_path: Path,
) -> None:
    root = tmp_path
    interpreter = root / "build" / "conda-runtime-future" / "python.exe"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"python-fixture")
    registry = root / runner._LOCAL_FUTURE_ORIGIN_SELECTION_REGISTRY
    command = [
        str(interpreter),
        "-I",
        "-B",
        "-m",
        "pfc_shaping.validation.ch_lt_local_future_origin_selection",
        "--registry",
        str(registry),
        "--expected-registry-sha256",
        runner._LOCAL_FUTURE_ORIGIN_SELECTION_SHA256,
    ]

    prepared = runner.prepare_command(command, root, root / "build" / "pytest")

    assert prepared[1:] == command[1:]
    command.remove("-I")
    with pytest.raises(runner.WorkspaceLocalError, match="exact Python flags"):
        runner.prepare_command(command, root, root / "build" / "pytest")


@pytest.mark.parametrize("external", [False, True])
def test_packaged_local_future_origin_selection_rejects_non_repo_runtime(
    tmp_path: Path,
    external: bool,
) -> None:
    root = tmp_path
    executable = str(Path(sys.executable).resolve()) if external else "python"
    command = [
        executable,
        "-I",
        "-B",
        "-m",
        runner._LOCAL_FUTURE_ORIGIN_PACKAGED_MODULE,
        "--registry",
        str(root / runner._LOCAL_FUTURE_ORIGIN_SELECTION_REGISTRY),
        "--expected-registry-sha256",
        runner._LOCAL_FUTURE_ORIGIN_SELECTION_SHA256,
    ]

    with pytest.raises(runner.WorkspaceLocalError, match="repo-local runtime"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def test_packaged_local_future_origin_execution_identity_is_exact(
    tmp_path: Path,
) -> None:
    root = tmp_path
    runtime = root / "build" / "conda-runtime-future"
    interpreter = runtime / "python.exe"
    origin = (
        runtime
        / "Lib"
        / "site-packages"
        / "pfc_shaping"
        / "validation"
        / "ch_lt_local_future_origin_selection.py"
    )
    origin.parent.mkdir(parents=True)
    interpreter.write_bytes(b"python-fixture")
    origin.write_bytes(b"module-fixture")
    package_parent = runtime / "Lib" / "site-packages"
    command = [
        str(interpreter),
        "-I",
        "-B",
        "-m",
        runner._LOCAL_FUTURE_ORIGIN_PACKAGED_MODULE,
    ]
    identity = {
        "target_interpreter": {"path": str(interpreter)},
        "target_import_closure": {
            "status": "BOUND_REPO_LOCAL_PTH",
            "sys_path": [str(package_parent)],
            "module_origins": [
                {
                    "module": runner._LOCAL_FUTURE_ORIGIN_PACKAGED_MODULE,
                    "path": str(origin),
                }
            ],
        },
    }

    runner._require_local_future_origin_execution_identity(command, identity, root)

    identity["target_import_closure"]["status"] = "UNBOUND_EXTERNAL_INTERPRETER"
    with pytest.raises(runner.WorkspaceLocalError, match="not repo-local and bound"):
        runner._require_local_future_origin_execution_identity(command, identity, root)


def test_explicit_relative_repo_runtime_is_not_substituted(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "build" / "conda-runtime-test" / "python.exe"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"python-fixture")
    command = [
        str(Path("build") / "conda-runtime-test" / "python.exe"),
        "-B",
        "-m",
        "scripts.audit_ch_lt_local_future_origin_selection",
    ]

    runner._validate_executable(command, tmp_path)

    assert Path(command[0]) == runtime.resolve(strict=True)


def _eex_vintage_command(root: Path) -> list[str]:
    build = root / "build"
    return [
        "python",
        "-I",
        "-B",
        "-m",
        "pfc_shaping.cli.eex_forward_vintage_builder",
        "--runtime-receipt",
        str(build / "runtime-receipt.json"),
        "--expected-runtime-receipt-sha256",
        "b" * 64,
        "--intake-spec",
        str(build / "eex" / "intake.json"),
        "--expected-spec-sha256",
        "a" * 64,
        "--source-document",
        str(build / "eex" / "source.xlsx"),
        "--trusted-time-receipt",
        str(build / "eex" / "receipt.json"),
        "--trusted-time-public-key",
        str(build / "eex" / "time-public.pem"),
        "--trusted-time-journal-id",
        "eex-journal-v1",
        "--output-directory",
        str(build / "eex" / "request"),
    ]


def test_eex_vintage_workflow_paths_must_remain_below_repo_build(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = _eex_vintage_command(root)
    command[command.index("--source-document") + 1] = str(tmp_path / "AppData" / "source.xlsx")

    with pytest.raises(runner.WorkspaceLocalError, match="below repo build"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def test_eex_vintage_workflow_repo_local_paths_are_allowed(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    command = runner.prepare_command(
        _eex_vintage_command(root),
        root,
        root / "build" / "pytest",
    )

    assert command[4] == "pfc_shaping.cli.eex_forward_vintage_builder"


def test_eex_vintage_workflow_requires_isolated_python(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = _eex_vintage_command(root)
    command.remove("-I")

    with pytest.raises(runner.WorkspaceLocalError, match="requires isolated Python"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def _eex_local_capture_command() -> list[str]:
    return [
        "python",
        "-B",
        "-m",
        "scripts.capture_eex_forward_local",
        "--source-document",
        str(runner._FMV_EEX_WORKBOOK_SOURCE),
        "--expected-source-sha256",
        "a" * 64,
        "--capture-id",
        "eex-ch-20260729-v1",
    ]


def test_eex_local_capture_allows_only_canonical_read_only_source(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    command = runner.prepare_command(
        _eex_local_capture_command(),
        root,
        root / "build" / "pytest",
    )

    assert command[command.index("--source-document") + 1] == str(
        runner._FMV_EEX_WORKBOOK_SOURCE
    )


@pytest.mark.parametrize(
    "source",
    [
        r"C:\Users\jbattaglia\AppData\Local\Price_Report_EEX.xlsx",
        r"C:\Users\jbattaglia\PFC_LT\build\Price_Report_EEX.xlsx",
        r"\\other-server\share\Price_Report_EEX.xlsx",
    ],
)
def test_eex_local_capture_rejects_noncanonical_source(
    tmp_path: Path,
    source: str,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = _eex_local_capture_command()
    command[command.index("--source-document") + 1] = source

    with pytest.raises(runner.WorkspaceLocalError, match="canonical read-only FMV"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def test_eex_local_capture_rejects_unknown_argument(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = [*_eex_local_capture_command(), "--output-directory", str(root / "build")]

    with pytest.raises(runner.WorkspaceLocalError, match="argument is not permitted"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def test_eex_local_capture_rejects_duplicate_option(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = [*_eex_local_capture_command(), "--capture-id", "second-id"]

    with pytest.raises(runner.WorkspaceLocalError, match="exactly once"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def test_eex_local_capture_requires_lowercase_sha256_and_portable_id(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    bad_hash = _eex_local_capture_command()
    bad_hash[bad_hash.index("--expected-source-sha256") + 1] = "A" * 64
    with pytest.raises(runner.WorkspaceLocalError, match="lowercase SHA-256"):
        runner.prepare_command(bad_hash, root, root / "build" / "pytest")

    bad_id = _eex_local_capture_command()
    bad_id[bad_id.index("--capture-id") + 1] = "../escape"
    with pytest.raises(runner.WorkspaceLocalError, match="portable identifier"):
        runner.prepare_command(bad_id, root, root / "build" / "pytest")


def test_eex_local_capture_requires_python_no_bytecode(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = _eex_local_capture_command()
    command.remove("-B")

    with pytest.raises(runner.WorkspaceLocalError, match="requires Python -B"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def _eex_current_selection_command(
    root: Path,
    *,
    module: str = "scripts.audit_ch_eex_current_local_capture",
) -> list[str]:
    return [
        "python",
        "-B",
        "-m",
        module,
        "--registry",
        str(root / runner._EEX_CURRENT_SELECTION_REGISTRY),
        "--expected-registry-sha256",
        runner._EEX_CURRENT_SELECTION_SHA256,
    ]


def test_eex_current_selection_route_is_exact_and_read_only(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    command = runner.prepare_command(
        _eex_current_selection_command(root),
        root,
        root / "build" / "pytest",
    )

    assert command[3] == "scripts.audit_ch_eex_current_local_capture"
    assert command[command.index("--expected-registry-sha256") + 1] == (
        runner._EEX_CURRENT_SELECTION_SHA256
    )

    repricing_command = runner.prepare_command(
        _eex_current_selection_command(
            root,
            module="scripts.audit_ch_eex_current_repricing_sensitivity",
        ),
        root,
        root / "build" / "pytest",
    )
    assert repricing_command[3] == (
        "scripts.audit_ch_eex_current_repricing_sensitivity"
    )


def test_eex_current_selection_rejects_shadow_registry(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = _eex_current_selection_command(root)
    command[command.index("--registry") + 1] = str(root / "build" / "shadow.json")

    with pytest.raises(runner.WorkspaceLocalError, match="canonical current CH EEX"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def test_eex_current_selection_rejects_hash_or_extra_argument(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    bad_hash = _eex_current_selection_command(root)
    bad_hash[bad_hash.index("--expected-registry-sha256") + 1] = "0" * 64
    with pytest.raises(runner.WorkspaceLocalError, match="must match"):
        runner.prepare_command(bad_hash, root, root / "build" / "pytest")

    extra = [*_eex_current_selection_command(root), "--output", "shadow.json"]
    with pytest.raises(runner.WorkspaceLocalError, match="not permitted"):
        runner.prepare_command(extra, root, root / "build" / "pytest")


def test_eex_current_selection_requires_python_no_bytecode(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = _eex_current_selection_command(root)
    command.remove("-B")

    with pytest.raises(runner.WorkspaceLocalError, match="requires Python -B"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def _preregistration_supersession_command(root: Path) -> list[str]:
    return [
        "python",
        "-B",
        "-m",
        "scripts.verify_ch_lt_preregistration_supersession",
        "--registry",
        str(root / runner._PREREGISTRATION_SUPERSESSION_REGISTRY),
        "--expected-registry-sha256",
        runner._PREREGISTRATION_SUPERSESSION_SHA256,
    ]


def test_preregistration_supersession_route_is_exact_and_read_only(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    command = runner.prepare_command(
        _preregistration_supersession_command(root),
        root,
        root / "build" / "pytest",
    )

    assert command[command.index("--expected-registry-sha256") + 1] == (
        runner._PREREGISTRATION_SUPERSESSION_SHA256
    )


def test_preregistration_supersession_rejects_external_registry(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = _preregistration_supersession_command(root)
    command[command.index("--registry") + 1] = str(tmp_path / "shadow.json")

    with pytest.raises(runner.WorkspaceLocalError, match="canonical D164 registry"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def test_preregistration_supersession_rejects_wrong_hash(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = _preregistration_supersession_command(root)
    command[command.index("--expected-registry-sha256") + 1] = "0" * 64

    with pytest.raises(runner.WorkspaceLocalError, match="canonical D164 registry"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def test_preregistration_supersession_rejects_unknown_or_duplicate_option(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    unknown = [*_preregistration_supersession_command(root), "--output", "result.json"]
    with pytest.raises(runner.WorkspaceLocalError, match="argument is not permitted"):
        runner.prepare_command(unknown, root, root / "build" / "pytest")

    duplicate = [
        *_preregistration_supersession_command(root),
        "--registry",
        str(root / runner._PREREGISTRATION_SUPERSESSION_REGISTRY),
    ]
    with pytest.raises(runner.WorkspaceLocalError, match="exactly once"):
        runner.prepare_command(duplicate, root, root / "build" / "pytest")


def _packaged_preregistration_supersession_command(root: Path) -> list[str]:
    return [
        "python",
        "-I",
        "-B",
        "-m",
        "pfc_shaping.cli.verify_ch_lt_preregistration_supersession",
        "--evidence-root",
        str(root),
        "--registry",
        str(root / runner._PREREGISTRATION_SUPERSESSION_REGISTRY),
        "--expected-registry-sha256",
        runner._PREREGISTRATION_SUPERSESSION_SHA256,
    ]


def test_packaged_preregistration_supersession_route_requires_isolation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = runner.prepare_command(
        _packaged_preregistration_supersession_command(root),
        root,
        root / "build" / "pytest",
    )
    assert "-I" in command
    assert command[command.index("--evidence-root") + 1] == str(root)

    without_isolation = _packaged_preregistration_supersession_command(root)
    without_isolation.remove("-I")
    with pytest.raises(runner.WorkspaceLocalError, match="requires Python -I"):
        runner.prepare_command(without_isolation, root, root / "build" / "pytest")


def test_packaged_preregistration_supersession_rejects_rebound_evidence_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = _packaged_preregistration_supersession_command(root)
    command[command.index("--evidence-root") + 1] = str(tmp_path)

    with pytest.raises(runner.WorkspaceLocalError, match="canonical workspace"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def _energy_charts_capture_command(root: Path) -> list[str]:
    ca_bundle = root / "build" / "test-ca.pem"
    ca_bundle.parent.mkdir(parents=True, exist_ok=True)
    ca_bundle.write_text("test CA bytes\n", encoding="utf-8")
    return [
        "python",
        "-B",
        "-m",
        "scripts.capture_public_energy_charts_lt",
        "--role",
        "epex_ch",
        "--start-utc",
        "2026-06-28T22:00:00Z",
        "--end-utc",
        "2026-07-28T22:00:00Z",
        "--raw-cadence-minutes",
        "60",
        "--acquisition-id",
        "ch-da-hourly-20260729-a",
        "--output-directory",
        str(root / "build" / "prospective-captures" / "ch-da-hourly-20260729-a"),
        "--ca-bundle",
        str(ca_bundle),
    ]


def test_energy_charts_ch_hourly_capture_is_repo_local(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    command = runner.prepare_command(
        _energy_charts_capture_command(root),
        root,
        root / "build" / "pytest",
    )

    assert command[command.index("--role") + 1] == "epex_ch"
    assert command[command.index("--raw-cadence-minutes") + 1] == "60"


def test_energy_charts_capture_rejects_external_output(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = _energy_charts_capture_command(root)
    command[command.index("--output-directory") + 1] = str(tmp_path / "AppData" / "capture")

    with pytest.raises(runner.WorkspaceLocalError, match="below repo build"):
        runner.prepare_command(command, root, root / "build" / "pytest")


def test_energy_charts_capture_requires_explicit_ca_bundle(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = _energy_charts_capture_command(root)
    index = command.index("--ca-bundle")
    del command[index : index + 2]

    with pytest.raises(runner.WorkspaceLocalError, match="--ca-bundle"):
        runner.prepare_command(command, root, root / "build" / "pytest")


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--role", "epex_de", "restricted to epex_ch"),
        ("--raw-cadence-minutes", "15", "native hourly cadence 60"),
    ],
)
def test_energy_charts_workspace_capture_rejects_wrong_product(
    tmp_path: Path,
    option: str,
    value: str,
    message: str,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    command = _energy_charts_capture_command(root)
    command[command.index(option) + 1] = value

    with pytest.raises(runner.WorkspaceLocalError, match=message):
        runner.prepare_command(command, root, root / "build" / "pytest")


def test_rejected_command_writes_terminal_receipt(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    monkeypatch.setattr(runner, "canonical_repo_root", lambda: root)
    monkeypatch.setattr(runner, "verify_canonical_workspace", lambda _root: None)

    with pytest.raises(runner.WorkspaceLocalError, match="Python interpreter"):
        runner._worker_main(["--run-id", "rejected", "--", "playwright.exe"])

    receipt = root / "build" / "workspace-local-runs" / "rejected" / "execution-receipt.json"
    payload = receipt.read_text(encoding="utf-8")
    assert '"status": "REJECTED_BEFORE_EXECUTION"' in payload
    assert '"playwright.exe"' in payload


@pytest.mark.parametrize(
    ("return_code", "expected_status"),
    [(0, "TARGET_EXIT_ZERO_NOT_AUTHORITY"), (7, "TARGET_EXIT_NONZERO")],
)
def test_target_exit_transition_is_non_authoritative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    return_code: int,
    expected_status: str,
) -> None:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    monkeypatch.setattr(runner, "canonical_repo_root", lambda: root)
    monkeypatch.setattr(runner, "verify_canonical_workspace", lambda _root: None)
    monkeypatch.setattr(
        runner,
        "_execute_command_with_capture",
        lambda _command, *, cwd, env, run_root, wall_timeout_seconds, on_started: _fake_captured_execution(
            run_root,
            return_code=return_code,
            wall_timeout_seconds=wall_timeout_seconds,
            on_started=on_started,
        ),
    )

    observed = runner._worker_main(
        ["--run-id", f"exit-{return_code}", "--", "python", "-m", "ruff"]
    )

    receipt_path = root / "build" / "workspace-local-runs" / f"exit-{return_code}"
    receipt = (receipt_path / "execution-receipt.json").read_text(encoding="utf-8")
    assert observed == return_code
    assert '"schema_version": "pfc_lt_workspace_local_execution.v6"' in receipt
    assert f'"status": "{expected_status}"' in receipt
    assert '"production_authorization": false' in receipt
    assert '"promotion_authorization": false' in receipt
    assert '"scientific_authorization": false' in receipt
    assert '"evaluation_authorization": false' in receipt
    assert '"runtime_authority": false' in receipt


def test_target_wall_timeout_is_terminal_and_non_authoritative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    monkeypatch.setattr(runner, "canonical_repo_root", lambda: root)
    monkeypatch.setattr(runner, "verify_canonical_workspace", lambda _root: None)
    monkeypatch.setattr(
        runner,
        "_execute_command_with_capture",
        lambda _command, *, cwd, env, run_root, wall_timeout_seconds, on_started: _fake_captured_execution(
            run_root,
            return_code=124,
            wall_timeout_seconds=wall_timeout_seconds,
            on_started=on_started,
            timed_out=True,
        ),
    )

    observed = runner._worker_main(
        [
            "--run-id",
            "timeout",
            "--wall-timeout-seconds",
            "0.25",
            "--",
            "python",
            "-m",
            "ruff",
        ]
    )

    receipt_path = root / "build" / "workspace-local-runs" / "timeout"
    receipt = json.loads((receipt_path / "execution-receipt.json").read_text())
    assert observed == 124
    assert receipt["status"] == "TARGET_WALL_TIMEOUT_TREE_TERMINATED_NO_AUTHORITY"
    assert receipt["runner_exit_code"] == 124
    assert receipt["supervision"] == {
        "global_wall_timeout_seconds": 0.25,
        "timed_out": True,
        "interrupted": False,
        "descendant_cleanup_triggered": False,
        "tree_termination_confirmed": True,
    }
    assert receipt["resource_usage"]["available"] is True
    assert receipt["production_authorization"] is False


def test_target_interrupt_is_terminal_and_non_authoritative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    monkeypatch.setattr(runner, "canonical_repo_root", lambda: root)
    monkeypatch.setattr(runner, "verify_canonical_workspace", lambda _root: None)
    monkeypatch.setattr(
        runner,
        "_execute_command_with_capture",
        lambda _command, *, cwd, env, run_root, wall_timeout_seconds, on_started: _fake_captured_execution(
            run_root,
            return_code=124,
            wall_timeout_seconds=wall_timeout_seconds,
            on_started=on_started,
            interrupted=True,
        ),
    )

    observed = runner._worker_main(
        ["--run-id", "interrupt", "--", "python", "-m", "ruff"]
    )

    receipt_path = root / "build" / "workspace-local-runs" / "interrupt"
    receipt = json.loads((receipt_path / "execution-receipt.json").read_text())
    assert observed == 130
    assert receipt["status"] == "INTERRUPTED_TREE_TERMINATED_NO_AUTHORITY"
    assert receipt["runner_exit_code"] == 130
    assert receipt["supervision"]["interrupted"] is True
    assert receipt["supervision"]["tree_termination_confirmed"] is True
    assert receipt["production_authorization"] is False


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "86401"])
def test_wall_timeout_must_be_positive_finite_and_bounded(value: str) -> None:
    with pytest.raises(SystemExit):
        runner._parser().parse_args(["--wall-timeout-seconds", value, "--preflight-only"])


def _fake_captured_execution(
    run_root: Path,
    *,
    return_code: int,
    stdout: bytes = b"",
    stderr: bytes = b"",
    truncated: bool = False,
    wall_timeout_seconds: float = runner._DEFAULT_WALL_TIMEOUT_SECONDS,
    on_started=None,
    timed_out: bool = False,
    interrupted: bool = False,
    descendant_cleanup_triggered: bool = False,
) -> runner.CapturedExecution:
    if on_started is not None:
        on_started(
            {
                "target_process": {"pid": 12345, "start_token": "1" * 16},
                "process_tree": {
                    "mode": "TEST_CONTAINMENT",
                    "kill_on_supervisor_exit": True,
                    "global_wall_timeout_seconds": wall_timeout_seconds,
                },
            }
        )
    streams: dict[str, dict[str, object]] = {}
    for label, payload in (("stdout", stdout), ("stderr", stderr)):
        path = run_root / f"target-{label}.log"
        path.write_bytes(payload)
        streams[label] = {
            "path": str(path),
            "captured_bytes": len(payload),
            "total_bytes": len(payload) + (1 if truncated else 0),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "full_stream_sha256": hashlib.sha256(
                payload + (b"x" if truncated else b"")
            ).hexdigest(),
            "truncated": truncated,
        }
    return runner.CapturedExecution(
        returncode=return_code,
        streams=streams,
        supervision={
            "global_wall_timeout_seconds": wall_timeout_seconds,
            "timed_out": timed_out,
            "interrupted": interrupted,
            "descendant_cleanup_triggered": descendant_cleanup_triggered,
            "tree_termination_confirmed": True,
        },
        resource_usage={
            "platform": "test",
            "wall_duration_seconds": 0.01,
        },
        timed_out=timed_out,
        interrupted=interrupted,
    )


def _write_native_result(
    command: list[str],
    *,
    exit_status: int = 0,
    selected: int = 0,
    deselected: int = 0,
    passed: int = 0,
    failed: int = 0,
    errors: int = 0,
    skipped: int = 0,
    xfailed: int = 0,
    xpassed: int = 0,
) -> None:
    path = Path(command[command.index("--pfc-workspace-result-json") + 1])
    path.write_text(
        json.dumps(
            {
                "schema_version": runner._PYTEST_NATIVE_RESULT_SCHEMA,
                "exit_status": exit_status,
                "selected": selected,
                "deselected": deselected,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "skipped": skipped,
                "xfailed": xfailed,
                "xpassed": xpassed,
                "production_authorization": False,
                "promotion_authorization": False,
                "scientific_authorization": False,
                "evaluation_authorization": False,
                "runtime_authority": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_successful_pytest_receipt_persists_hash_bound_structured_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    monkeypatch.setattr(runner, "canonical_repo_root", lambda: root)
    monkeypatch.setattr(runner, "verify_canonical_workspace", lambda _root: None)

    def execute(command, *, cwd, env, run_root, wall_timeout_seconds, on_started):
        junit_path = Path(command[command.index("--junitxml") + 1])
        junit_path.write_text(
            '<testsuites><testsuite name="pytest" errors="0" failures="0" '
            'skipped="1" tests="3" time="1.25" /></testsuites>',
            encoding="utf-8",
        )
        _write_native_result(
            command,
            selected=3,
            deselected=2,
            passed=2,
            skipped=1,
        )
        return _fake_captured_execution(
            run_root,
            return_code=0,
            stdout=b"2 passed, 1 skipped\n",
            wall_timeout_seconds=wall_timeout_seconds,
            on_started=on_started,
        )

    monkeypatch.setattr(runner, "_execute_command_with_capture", execute)

    observed = runner._worker_main(
        ["--run-id", "junit-pass", "--", "python", "-B", "-m", "pytest", "tests"]
    )

    receipt_path = root / "build" / "workspace-local-runs" / "junit-pass"
    receipt = json.loads((receipt_path / "execution-receipt.json").read_text(encoding="utf-8"))
    assert observed == 0
    assert receipt["status"] == "TARGET_EXIT_ZERO_NOT_AUTHORITY"
    assert receipt["pytest_result"]["applicable"] is True
    assert receipt["pytest_result"]["duration_seconds"] == 1.25
    assert receipt["pytest_result"]["tests"] == 3
    assert receipt["pytest_result"]["passed"] == 2
    assert receipt["pytest_result"]["failures"] == 0
    assert receipt["pytest_result"]["errors"] == 0
    assert receipt["pytest_result"]["skipped"] == 1
    assert receipt["pytest_result"]["deselected"] == 2
    assert receipt["pytest_result"]["native_result"]["selected"] == 3
    assert receipt["pytest_result"]["native_result"]["deselected"] == 2
    assert receipt["pytest_result"]["path"] == str(receipt_path / "pytest-results.xml")
    assert receipt["pytest_result"]["sha256"] == hashlib.sha256(
        (receipt_path / "pytest-results.xml").read_bytes()
    ).hexdigest()
    assert receipt["target_output"]["complete"] is True
    assert receipt["target_output"]["streams"]["stdout"]["sha256"] == hashlib.sha256(
        b"2 passed, 1 skipped\n"
    ).hexdigest()
    assert receipt["scientific_authorization"] is False
    assert receipt["production_authorization"] is False


def test_successful_pytest_without_junit_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    monkeypatch.setattr(runner, "canonical_repo_root", lambda: root)
    monkeypatch.setattr(runner, "verify_canonical_workspace", lambda _root: None)
    monkeypatch.setattr(
        runner,
        "_execute_command_with_capture",
        lambda _command, *, cwd, env, run_root, wall_timeout_seconds, on_started: _fake_captured_execution(
            run_root,
            return_code=0,
            wall_timeout_seconds=wall_timeout_seconds,
            on_started=on_started,
        ),
    )

    observed = runner._worker_main(
        ["--run-id", "junit-missing", "--", "python", "-B", "-m", "pytest", "tests"]
    )

    receipt_path = root / "build" / "workspace-local-runs" / "junit-missing"
    receipt = json.loads((receipt_path / "execution-receipt.json").read_text(encoding="utf-8"))
    assert observed == 2
    assert receipt["status"] == "TARGET_EVIDENCE_INVALID"
    assert receipt["target_exit_code"] == 0
    assert receipt["runner_exit_code"] == 2
    assert "JUnit output is missing" in receipt["failure_message"]


def test_output_capture_truncation_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    monkeypatch.setattr(runner, "canonical_repo_root", lambda: root)
    monkeypatch.setattr(runner, "verify_canonical_workspace", lambda _root: None)
    monkeypatch.setattr(
        runner,
        "_execute_command_with_capture",
        lambda _command, *, cwd, env, run_root, wall_timeout_seconds, on_started: _fake_captured_execution(
            run_root,
            return_code=0,
            stdout=b"retained",
            truncated=True,
            wall_timeout_seconds=wall_timeout_seconds,
            on_started=on_started,
        ),
    )

    observed = runner._worker_main(
        ["--run-id", "truncated", "--", "python", "-m", "ruff"]
    )

    receipt_path = root / "build" / "workspace-local-runs" / "truncated"
    receipt = json.loads((receipt_path / "execution-receipt.json").read_text(encoding="utf-8"))
    assert observed == 2
    assert receipt["status"] == "TARGET_OUTPUT_CAPTURE_LIMIT_EXCEEDED"
    assert receipt["target_output"]["complete"] is False
    assert receipt["production_authorization"] is False


def test_bounded_stream_capture_records_full_and_retained_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_OUTPUT_CAPTURE_LIMIT_BYTES", 5)
    destination = tmp_path / "stdout.log"
    results: dict[str, dict[str, object]] = {}
    errors: list[BaseException] = []

    runner._pump_bounded_stream(
        io.BytesIO(b"abcdef"),
        destination,
        "stdout",
        results,
        errors,
    )

    assert errors == []
    assert destination.read_bytes() == b"abcde"
    assert results["stdout"]["captured_bytes"] == 5
    assert results["stdout"]["total_bytes"] == 6
    assert results["stdout"]["sha256"] == hashlib.sha256(b"abcde").hexdigest()
    assert results["stdout"]["full_stream_sha256"] == hashlib.sha256(b"abcdef").hexdigest()
    assert results["stdout"]["truncated"] is True


@pytest.mark.parametrize("tests", ["True", "01", "10000001"])
def test_pytest_junit_rejects_noncanonical_or_unbounded_counts(
    tmp_path: Path,
    tests: str,
) -> None:
    path = tmp_path / "pytest-results.xml"
    path.write_text(
        f'<testsuites><testsuite errors="0" failures="0" skipped="0" '
        f'tests="{tests}" time="1.0" /></testsuites>',
        encoding="utf-8",
    )

    with pytest.raises(runner.WorkspaceLocalError, match="pytest tests"):
        runner._pytest_junit_evidence(path, root=tmp_path)


def test_pytest_junit_rejects_multiple_suites(tmp_path: Path) -> None:
    path = tmp_path / "pytest-results.xml"
    path.write_text(
        '<testsuites><testsuite errors="0" failures="0" skipped="0" tests="1" '
        'time="1.0" /><testsuite errors="0" failures="0" skipped="0" tests="1" '
        'time="1.0" /></testsuites>',
        encoding="utf-8",
    )

    with pytest.raises(runner.WorkspaceLocalError, match="exactly one"):
        runner._pytest_junit_evidence(path, root=tmp_path)


def test_target_output_hash_mismatch_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    monkeypatch.setattr(runner, "canonical_repo_root", lambda: root)
    monkeypatch.setattr(runner, "verify_canonical_workspace", lambda _root: None)

    def execute(
        _command, *, cwd, env, run_root, wall_timeout_seconds, on_started
    ):
        captured = _fake_captured_execution(
            run_root,
            return_code=0,
            stdout=b"real",
            wall_timeout_seconds=wall_timeout_seconds,
            on_started=on_started,
        )
        captured.streams["stdout"]["sha256"] = "0" * 64
        return captured

    monkeypatch.setattr(runner, "_execute_command_with_capture", execute)

    observed = runner._worker_main(
        ["--run-id", "hash-mismatch", "--", "python", "-m", "ruff"]
    )

    receipt_path = root / "build" / "workspace-local-runs" / "hash-mismatch"
    receipt = json.loads((receipt_path / "execution-receipt.json").read_text(encoding="utf-8"))
    assert observed == 2
    assert receipt["status"] == "TARGET_EVIDENCE_INVALID"
    assert "capture hash is inconsistent" in receipt["failure_message"]
    assert receipt["production_authorization"] is False


def test_execution_identity_change_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    monkeypatch.setattr(runner, "canonical_repo_root", lambda: root)
    monkeypatch.setattr(runner, "verify_canonical_workspace", lambda _root: None)
    calls = 0

    def identity(_root, _command):
        nonlocal calls
        calls += 1
        return {
            "runner_source": {"sha256": ("a" if calls == 1 else "b") * 64},
            "target_interpreter": {"sha256": "c" * 64},
            "git_head": "d" * 40,
            "git_branch": "test",
        }

    monkeypatch.setattr(runner, "_execution_identity", identity)
    monkeypatch.setattr(
        runner,
        "_execute_command_with_capture",
        lambda _command, *, cwd, env, run_root, wall_timeout_seconds, on_started: _fake_captured_execution(
            run_root,
            return_code=0,
            wall_timeout_seconds=wall_timeout_seconds,
            on_started=on_started,
        ),
    )

    observed = runner._worker_main(
        ["--run-id", "identity-swap", "--", "python", "-m", "ruff"]
    )

    receipt_path = root / "build" / "workspace-local-runs" / "identity-swap"
    receipt = json.loads((receipt_path / "execution-receipt.json").read_text(encoding="utf-8"))
    assert observed == 2
    assert receipt["status"] == "TARGET_EVIDENCE_INVALID"
    assert "execution identity changed" in receipt["failure_message"]


def test_source_tree_identity_changes_with_dirty_source_bytes(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    source = root / "pfc_shaping" / "lt" / "model.py"
    test_source = root / "tests" / "test_model.py"
    source.parent.mkdir(parents=True)
    test_source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    test_source.write_text("def test_value():\n    assert True\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname = 'example'\n", encoding="utf-8")

    before = runner._source_tree_identity(root)
    test_source.write_text("def test_value():\n    assert False\n", encoding="utf-8")
    after = runner._source_tree_identity(root)

    assert before["schema_version"] == "pfc_lt_workspace_source_tree.v1"
    assert before["file_count"] == 3
    assert before["tree_sha256"] != after["tree_sha256"]
    assert before["total_bytes"] != after["total_bytes"]


def test_repo_local_import_closure_binds_pth_roots_and_module_origins(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    launcher = root / "build" / "pytest-runtime"
    runtime_lib = root / "build" / "runtime" / "Lib"
    test_site = launcher / "test-site-packages"
    source = root / "pfc_shaping"
    for directory in (launcher, runtime_lib, test_site / "pytest", source):
        directory.mkdir(parents=True, exist_ok=True)
    interpreter = launcher / "python.exe"
    interpreter.write_bytes(b"python")
    (launcher / "python311._pth").write_text(
        "..\\runtime\\Lib\ntest-site-packages\n..\\..\n",
        encoding="utf-8",
    )
    (runtime_lib / "stdlib.py").write_text("VALUE = 1\n", encoding="utf-8")
    (test_site / "pytest" / "__init__.py").write_text(
        "__version__ = 'test'\n",
        encoding="utf-8",
    )
    (source / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    source_tree = {
        "schema_version": "pfc_lt_workspace_source_tree.v1",
        "file_count": 1,
        "total_bytes": 10,
        "tree_sha256": "a" * 64,
    }

    before = runner._target_import_closure_identity(
        root,
        interpreter,
        [str(interpreter), "-I", "-B", "-m", "pytest"],
        source_tree=source_tree,
    )
    (test_site / "pytest" / "__init__.py").write_text(
        "__version__ = 'changed'\n",
        encoding="utf-8",
    )
    after = runner._target_import_closure_identity(
        root,
        interpreter,
        [str(interpreter), "-I", "-B", "-m", "pytest"],
        source_tree=source_tree,
    )

    assert before["status"] == "BOUND_REPO_LOCAL_PTH"
    assert before["python_pth"]["sha256"] == hashlib.sha256(
        (launcher / "python311._pth").read_bytes()
    ).hexdigest()
    assert before["launcher_tree"]["tree_sha256"] != after["launcher_tree"]["tree_sha256"]
    assert [item["module"] for item in before["module_origins"]] == [
        "pfc_shaping",
        "pytest",
    ]


def test_repo_local_import_closure_rejects_pth_escape(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    launcher = root / "build" / "pytest-runtime"
    launcher.mkdir(parents=True)
    interpreter = launcher / "python.exe"
    interpreter.write_bytes(b"python")
    (launcher / "python311._pth").write_text("..\\..\\..\n", encoding="utf-8")

    with pytest.raises(runner.WorkspaceLocalError, match="escaped the workspace"):
        runner._target_import_closure_identity(
            root,
            interpreter,
            [str(interpreter), "-I", "-B", "-m", "pytest"],
            source_tree={
                "schema_version": "pfc_lt_workspace_source_tree.v1",
                "file_count": 1,
                "total_bytes": 1,
                "tree_sha256": "a" * 64,
            },
        )


def test_successful_pytest_exit_cannot_hide_junit_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    monkeypatch.setattr(runner, "canonical_repo_root", lambda: root)
    monkeypatch.setattr(runner, "verify_canonical_workspace", lambda _root: None)

    def execute(command, *, cwd, env, run_root, wall_timeout_seconds, on_started):
        Path(command[command.index("--junitxml") + 1]).write_text(
            '<testsuites><testsuite errors="0" failures="1" skipped="0" '
            'tests="1" time="0.1" /></testsuites>',
            encoding="utf-8",
        )
        _write_native_result(command, selected=1, passed=1)
        return _fake_captured_execution(
            run_root,
            return_code=0,
            wall_timeout_seconds=wall_timeout_seconds,
            on_started=on_started,
        )

    monkeypatch.setattr(runner, "_execute_command_with_capture", execute)

    observed = runner._worker_main(
        ["--run-id", "exit-conflict", "--", "python", "-B", "-m", "pytest", "tests"]
    )

    receipt_path = root / "build" / "workspace-local-runs" / "exit-conflict"
    receipt = json.loads((receipt_path / "execution-receipt.json").read_text(encoding="utf-8"))
    assert observed == 2
    assert receipt["status"] == "TARGET_EVIDENCE_INVALID"
    assert "JUnit/native passed counts disagree" in receipt["failure_message"]


def test_pytest_native_result_rejects_boolean_count(tmp_path: Path) -> None:
    path = tmp_path / "pytest-native-result.json"
    payload = {
        "schema_version": runner._PYTEST_NATIVE_RESULT_SCHEMA,
        "exit_status": 0,
        "selected": True,
        "deselected": 0,
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "xfailed": 0,
        "xpassed": 0,
        "production_authorization": False,
        "promotion_authorization": False,
        "scientific_authorization": False,
        "evaluation_authorization": False,
        "runtime_authority": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(runner.WorkspaceLocalError, match="strict integer"):
        runner._pytest_native_evidence(path, root=tmp_path)


def test_pytest_native_result_rejects_authority_smuggling(tmp_path: Path) -> None:
    path = tmp_path / "pytest-native-result.json"
    payload = {
        "schema_version": runner._PYTEST_NATIVE_RESULT_SCHEMA,
        "exit_status": 0,
        "selected": 1,
        "deselected": 0,
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "xfailed": 0,
        "xpassed": 0,
        "production_authorization": True,
        "promotion_authorization": False,
        "scientific_authorization": False,
        "evaluation_authorization": False,
        "runtime_authority": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(runner.WorkspaceLocalError, match="must be false"):
        runner._pytest_native_evidence(path, root=tmp_path)


def test_pytest_native_result_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "pytest-native-result.json"
    path.write_text('{"schema_version":"x","schema_version":"y"}', encoding="utf-8")

    with pytest.raises(runner.WorkspaceLocalError, match="duplicate JSON key"):
        runner._pytest_native_evidence(path, root=tmp_path)


def test_stable_reader_detects_descriptor_identity_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "evidence.bin"
    path.write_bytes(b"evidence")
    original_fstat = runner.os.fstat
    calls = 0

    def changing_fstat(file_descriptor: int):
        nonlocal calls
        calls += 1
        observed = original_fstat(file_descriptor)
        if calls == 2:
            return SimpleNamespace(
                st_dev=observed.st_dev,
                st_ino=observed.st_ino + 1,
                st_size=observed.st_size,
                st_nlink=observed.st_nlink,
            )
        return observed

    monkeypatch.setattr(runner.os, "fstat", changing_fstat)

    with pytest.raises(runner.WorkspaceLocalError, match="changed during capture"):
        runner._stable_bounded_regular_file_bytes(
            path,
            root=tmp_path,
            limit_bytes=1024,
            label="test evidence",
        )


def test_inherited_descendant_pipe_is_terminated_without_unbounded_join(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "spawn_descendant.py"
    script.write_text(
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(1.0)'])\n"
        "print('direct child complete')\n",
        encoding="utf-8",
    )
    run_root = tmp_path / "run"
    run_root.mkdir()
    monkeypatch.setattr(runner, "_PIPE_DRAIN_GRACE_SECONDS", 0.05)
    started = time.monotonic()

    completed = runner._execute_command_with_capture(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=dict(runner.os.environ),
        run_root=run_root,
        wall_timeout_seconds=5.0,
        on_started=lambda _started: None,
    )

    assert time.monotonic() - started < 0.8
    assert completed.supervision["descendant_cleanup_triggered"] is True
    assert completed.supervision["tree_termination_confirmed"] is True


def test_global_wall_timeout_terminates_job_and_records_resource_usage(
    tmp_path: Path,
) -> None:
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    run_root = tmp_path / "run"
    run_root.mkdir()
    started: list[dict[str, object]] = []
    began = time.monotonic()

    completed = runner._execute_command_with_capture(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=dict(runner.os.environ),
        run_root=run_root,
        wall_timeout_seconds=0.2,
        on_started=started.append,
    )

    assert time.monotonic() - began < 3.0
    assert completed.timed_out is True
    assert completed.supervision["tree_termination_confirmed"] is True
    assert completed.resource_usage["wall_duration_seconds"] >= 0.2
    assert completed.resource_usage["user_cpu_seconds"] >= 0.0
    assert completed.resource_usage["kernel_cpu_seconds"] >= 0.0
    if runner.os.name == "nt":
        assert completed.resource_usage["platform"] == "windows_job_object"
        assert completed.resource_usage["active_processes"] == 0
        assert completed.resource_usage["peak_job_memory_bytes"] > 0
        assert completed.resource_usage["write_transfer_bytes"] >= 0
        assert started[0]["process_tree"]["mode"] == (
            "CREATE_SUSPENDED_ASSIGN_KILL_ON_JOB_CLOSE_THEN_RESUME"
        )


def _stale_execution_receipt(
    root: Path,
    run_root: Path,
    *,
    runner_process: dict[str, object],
    target_process: dict[str, object] | None,
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema_version": runner._RECEIPT_SCHEMA,
        "status": "EXECUTION_STARTED" if target_process else "EXECUTION_PENDING",
        "production_authorization": False,
        "promotion_authorization": False,
        "scientific_authorization": False,
        "evaluation_authorization": False,
        "runtime_authority": False,
        "workspace": str(root),
        "run_root": str(run_root),
        "runner_process": runner_process,
    }
    if target_process is not None:
        receipt["target_process"] = target_process
        receipt["process_tree"] = {
            "mode": "CREATE_SUSPENDED_ASSIGN_KILL_ON_JOB_CLOSE_THEN_RESUME",
            "kill_on_supervisor_exit": True,
            "global_wall_timeout_seconds": 1800.0,
        }
    return receipt


def test_stale_run_reconciliation_is_exclusive_and_preserves_original(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    run_root = root / "build" / "workspace-local-runs" / "stale"
    run_root.mkdir(parents=True)
    receipt_path = run_root / "execution-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _stale_execution_receipt(
                root,
                run_root,
                runner_process={"pid": 999_991, "start_token": "1" * 16},
                target_process={"pid": 999_992, "start_token": "2" * 16},
            ),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    original = receipt_path.read_bytes()

    reconciliation = runner._reconcile_stale_run(root, "stale")

    sidecar = run_root / "reconciliation-receipt.json"
    assert receipt_path.read_bytes() == original
    assert sidecar.is_file()
    assert reconciliation["status"] == "ABANDONED_RUN_CONFIRMED_NO_AUTHORITY"
    assert reconciliation["execution_receipt"]["sha256"] == hashlib.sha256(
        original
    ).hexdigest()
    assert reconciliation["runner_observation"]["exact_identity_active"] is False
    assert reconciliation["target_observation"]["exact_identity_active"] is False
    assert reconciliation["retry_disposition"] == "ROTATE_TO_A_FRESH_RUN_ID_ONLY"
    assert reconciliation["production_authorization"] is False

    with pytest.raises(runner.WorkspaceLocalError, match="already has"):
        runner._reconcile_stale_run(root, "stale")


def test_stale_run_reconciliation_refuses_live_exact_runner(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    run_root = root / "build" / "workspace-local-runs" / "live"
    run_root.mkdir(parents=True)
    current = runner._new_process_identity(runner.os.getpid(), label="test runner")
    (run_root / "execution-receipt.json").write_text(
        json.dumps(
            _stale_execution_receipt(
                root,
                run_root,
                runner_process=current,
                target_process=None,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(runner.WorkspaceLocalError, match="still active"):
        runner._reconcile_stale_run(root, "live")


def test_stale_run_reconciliation_refuses_legacy_schema(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    run_root = root / "build" / "workspace-local-runs" / "legacy"
    run_root.mkdir(parents=True)
    receipt = _stale_execution_receipt(
        root,
        run_root,
        runner_process={"pid": 999_993, "start_token": "3" * 16},
        target_process=None,
    )
    receipt["schema_version"] = "pfc_lt_workspace_local_execution.v5"
    (run_root / "execution-receipt.json").write_text(
        json.dumps(receipt) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(runner.WorkspaceLocalError, match="schema v6"):
        runner._reconcile_stale_run(root, "legacy")


def test_internal_worker_capability_is_parent_bound_and_one_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    supervisor_root = root / "build" / "workspace-local-supervisors" / "capability"
    supervisor_root.mkdir(parents=True)
    captured = supervisor_root / "run_workspace_local_worker.py"
    captured.write_bytes(b"# captured worker\n")
    source = {
        "source_path": str(root / "scripts" / "run_workspace_local.py"),
        "captured_path": str(captured),
        "size_bytes": captured.stat().st_size,
        "sha256": hashlib.sha256(captured.read_bytes()).hexdigest(),
    }
    worker_argv = [
        "--run-id",
        "capability",
        "--wall-timeout-seconds",
        "30",
        "--preflight-only",
    ]
    capability_path, token, capability = runner._write_internal_worker_capability(
        root=root,
        supervisor_root=supervisor_root,
        run_id="capability",
        worker_source=source,
        worker_argv_sha256=runner._command_sha256(worker_argv),
        wall_timeout_seconds=30.0,
    )
    monkeypatch.setattr(runner, "__file__", str(captured))
    monkeypatch.setattr(
        runner,
        "_current_parent_pid",
        lambda: int(capability["parent_process"]["pid"]),
    )
    monkeypatch.setenv(runner._INTERNAL_WORKER_TOKEN_ENV, token)
    monkeypatch.setenv(runner._INTERNAL_WORKER_CAPABILITY_ENV, str(capability_path))
    monkeypatch.setenv(
        runner._INTERNAL_WORKER_PARENT_PID_ENV,
        str(capability["parent_process"]["pid"]),
    )
    monkeypatch.setenv(
        runner._INTERNAL_WORKER_PARENT_TOKEN_ENV,
        str(capability["parent_process"]["start_token"]),
    )

    with pytest.raises(runner.WorkspaceLocalError, match="command binding"):
        runner._consume_internal_worker_capability(
            root=root,
            args=SimpleNamespace(run_id="capability", wall_timeout_seconds=30.0),
            worker_argv=[*worker_argv, "--unexpected"],
        )
    assert capability_path.is_file()
    with pytest.raises(runner.WorkspaceLocalError, match="wall timeout binding"):
        runner._consume_internal_worker_capability(
            root=root,
            args=SimpleNamespace(run_id="capability", wall_timeout_seconds=31.0),
            worker_argv=worker_argv,
        )
    assert capability_path.is_file()

    consumed = runner._consume_internal_worker_capability(
        root=root,
        args=SimpleNamespace(run_id="capability", wall_timeout_seconds=30.0),
        worker_argv=worker_argv,
    )

    assert consumed == capability
    assert not capability_path.exists()
    admission_path = supervisor_root / "worker-admission.json"
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    assert admission["status"] == "CAPABILITY_CONSUMED_NO_AUTHORITY"
    assert admission["capability_sha256"] == hashlib.sha256(
        json.dumps(capability, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    ).hexdigest()
    evidence = runner._worker_admission_evidence(
        root=root,
        supervisor_root=supervisor_root,
        expected_run_id="capability",
        expected_capability_sha256=admission["capability_sha256"],
        expected_worker_argv_sha256=runner._command_sha256(worker_argv),
        expected_worker_process=admission["worker_process"],
        expected_parent_process=capability["parent_process"],
    )
    assert evidence is not None
    assert evidence["status"] == "CAPABILITY_CONSUMED_NO_AUTHORITY"
    with pytest.raises(runner.WorkspaceLocalError, match="binding is inconsistent"):
        runner._worker_admission_evidence(
            root=root,
            supervisor_root=supervisor_root,
            expected_run_id="wrong-run",
            expected_capability_sha256=admission["capability_sha256"],
            expected_worker_argv_sha256=runner._command_sha256(worker_argv),
            expected_worker_process=admission["worker_process"],
            expected_parent_process=capability["parent_process"],
        )
    assert runner._INTERNAL_WORKER_TOKEN_ENV not in runner.os.environ
    with pytest.raises(runner.WorkspaceLocalError, match="incomplete"):
        runner._consume_internal_worker_capability(
            root=root,
            args=SimpleNamespace(run_id="capability", wall_timeout_seconds=30.0),
            worker_argv=worker_argv,
        )


def test_internal_worker_flag_without_capability_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_EXPECTED_ROOT", tmp_path)
    monkeypatch.setattr(runner, "verify_canonical_workspace", lambda _root: None)
    for key in (
        runner._INTERNAL_WORKER_TOKEN_ENV,
        runner._INTERNAL_WORKER_CAPABILITY_ENV,
        runner._INTERNAL_WORKER_PARENT_PID_ENV,
        runner._INTERNAL_WORKER_PARENT_TOKEN_ENV,
    ):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(runner.WorkspaceLocalError, match="incomplete"):
        runner.main(
            [
                runner._INTERNAL_WORKER_FLAG,
                "--run-id",
                "forged",
                "--preflight-only",
            ]
        )


@pytest.mark.skipif(
    not _CANONICAL_WINDOWS_E2E,
    reason="literal-root Windows workstation supervisor contract",
)
def test_supervisor_e2e_zero_binds_admission_and_revokes_capability() -> None:
    run_id = _fresh_e2e_id("e2ez")
    completed = subprocess.run(
        _supervisor_command(
            run_id,
            "--wall-timeout-seconds",
            "30",
            "--preflight-only",
        ),
        cwd=runner._EXPECTED_ROOT,
        check=False,
        capture_output=True,
        timeout=40,
    )

    supervisor_root = (
        runner._EXPECTED_ROOT / "build" / "workspace-local-supervisors" / run_id
    )
    receipt = _read_json(supervisor_root / "supervisor-receipt.json")
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert receipt["status"] == "WORKER_EXIT_ZERO_NOT_AUTHORITY"
    assert receipt["worker_admission"]["status"] == (
        "CAPABILITY_CONSUMED_NO_AUTHORITY"
    )
    assert receipt["worker_capability"]["consumed"] is True
    assert receipt["worker_capability"]["absent_after_worker"] is True
    assert not (supervisor_root / "worker-capability.json").exists()
    assert receipt["resource_usage"]["active_processes"] == 0
    assert receipt["resource_usage"]["read_operation_count"] > 0
    assert receipt["resource_usage"]["write_operation_count"] > 0
    assert receipt["resource_usage"]["read_transfer_bytes"] > 0
    assert receipt["resource_usage"]["write_transfer_bytes"] > 0
    assert Path(receipt["mutable_paths"]["appdata"]).is_relative_to(
        runner._EXPECTED_ROOT / "build"
    )
    assert Path(receipt["mutable_paths"]["local-appdata"]).is_relative_to(
        runner._EXPECTED_ROOT / "build"
    )
    for authority in (
        "production_authorization",
        "promotion_authorization",
        "scientific_authorization",
        "evaluation_authorization",
        "runtime_authority",
    ):
        assert receipt[authority] is False


@pytest.mark.skipif(
    not _CANONICAL_WINDOWS_E2E,
    reason="literal-root Windows workstation supervisor contract",
)
def test_supervisor_e2e_timeout_before_admission_revokes_capability() -> None:
    run_id = _fresh_e2e_id("e2eb")
    completed = subprocess.run(
        _supervisor_command(
            run_id,
            "--wall-timeout-seconds",
            "0.000001",
            "--preflight-only",
        ),
        cwd=runner._EXPECTED_ROOT,
        check=False,
        capture_output=True,
        timeout=15,
    )

    supervisor_root = (
        runner._EXPECTED_ROOT / "build" / "workspace-local-supervisors" / run_id
    )
    receipt = _read_json(supervisor_root / "supervisor-receipt.json")
    assert completed.returncode == 124
    assert receipt["status"] == "SUPERVISOR_PREFLIGHT_WALL_TIMEOUT_NO_AUTHORITY"
    assert receipt["worker_capability"]["consumed"] is False
    assert receipt["worker_capability"]["absent_after_worker"] is True
    assert not (supervisor_root / "worker-admission.json").exists()
    assert not (supervisor_root / "worker-capability.json").exists()


@pytest.mark.skipif(
    not _CANONICAL_WINDOWS_E2E,
    reason="literal-root Windows workstation supervisor contract",
)
def test_supervisor_e2e_timeout_after_admission_terminates_tree() -> None:
    run_id = _fresh_e2e_id("e2ea")
    target_root = (
        runner._EXPECTED_ROOT / "build" / "workspace-local-e2e-targets" / run_id
    )
    target_root.mkdir(parents=True, exist_ok=False)
    target = target_root / "test_hold_for_supervisor_timeout.py"
    target.write_text(
        "import time\n\n"
        "def test_hold_for_supervisor_timeout():\n"
        "    time.sleep(30)\n",
        encoding="utf-8",
        newline="\n",
    )
    completed = subprocess.run(
        _supervisor_command(
            run_id,
            "--wall-timeout-seconds",
            "5",
            "--",
            sys.executable,
            "-B",
            "-m",
            "pytest",
            str(target),
            "-q",
            "-p",
            "no:cacheprovider",
        ),
        cwd=runner._EXPECTED_ROOT,
        check=False,
        capture_output=True,
        timeout=20,
    )

    supervisor_root = (
        runner._EXPECTED_ROOT / "build" / "workspace-local-supervisors" / run_id
    )
    receipt = _read_json(supervisor_root / "supervisor-receipt.json")
    execution = _read_json(
        runner._EXPECTED_ROOT
        / "build"
        / "workspace-local-runs"
        / run_id
        / "execution-receipt.json"
    )
    assert completed.returncode == 124
    assert receipt["status"] == "SUPERVISOR_WALL_TIMEOUT_TREE_TERMINATED_NO_AUTHORITY"
    assert receipt["worker_admission"]["status"] == (
        "CAPABILITY_CONSUMED_NO_AUTHORITY"
    )
    assert receipt["worker_capability"]["consumed"] is True
    assert receipt["worker_capability"]["absent_after_worker"] is True
    assert receipt["resource_usage"]["active_processes"] == 0
    assert execution["status"] == "SUPERVISOR_WALL_TIMEOUT_TREE_TERMINATED_NO_AUTHORITY"
    assert execution["runner_exit_code"] == 124
    assert execution["supervisor_termination"]["tree_termination_confirmed"] is True


@pytest.mark.skipif(
    not _CANONICAL_WINDOWS_E2E,
    reason="literal-root Windows workstation supervisor contract",
)
def test_supervisor_e2e_abrupt_parent_death_kills_worker_and_reconciles() -> None:
    run_id = _fresh_e2e_id("e2ec")
    target_root = (
        runner._EXPECTED_ROOT / "build" / "workspace-local-e2e-targets" / run_id
    )
    target_root.mkdir(parents=True, exist_ok=False)
    target = target_root / "test_hold_for_supervisor_crash.py"
    target.write_text(
        "import time\n\n"
        "def test_hold_for_supervisor_crash():\n"
        "    time.sleep(120)\n",
        encoding="utf-8",
        newline="\n",
    )
    supervisor = subprocess.Popen(
        _supervisor_command(
            run_id,
            "--wall-timeout-seconds",
            "60",
            "--",
            sys.executable,
            "-B",
            "-m",
            "pytest",
            str(target),
            "-q",
            "-p",
            "no:cacheprovider",
        ),
        cwd=runner._EXPECTED_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    supervisor_root = (
        runner._EXPECTED_ROOT / "build" / "workspace-local-supervisors" / run_id
    )
    execution_path = (
        runner._EXPECTED_ROOT
        / "build"
        / "workspace-local-runs"
        / run_id
        / "execution-receipt.json"
    )
    try:
        _wait_for(
            lambda: (supervisor_root / "worker-admission.json").is_file()
            and execution_path.is_file(),
            timeout_seconds=15,
        )
        receipt = _read_json(supervisor_root / "supervisor-receipt.json")
        worker_identity = receipt["worker_process"]
        supervisor.kill()
        supervisor.wait(timeout=10)

        _wait_for(
            lambda: runner._observe_expected_process(
                worker_identity,
                label="E2E worker",
            )["exact_identity_active"]
            is False,
            timeout_seconds=10,
        )
        assert not (supervisor_root / "worker-capability.json").exists()
        assert (supervisor_root / "worker-admission.json").is_file()
        reconciliation = runner._reconcile_stale_run(runner._EXPECTED_ROOT, run_id)
        assert reconciliation["status"] == "ABANDONED_RUN_CONFIRMED_NO_AUTHORITY"
        assert reconciliation["runner_observation"]["exact_identity_active"] is False
        assert reconciliation["target_observation"] is None
        with pytest.raises(
            runner.WorkspaceLocalError,
            match="already has a reconciliation receipt",
        ):
            runner._reconcile_stale_run(runner._EXPECTED_ROOT, run_id)
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait(timeout=10)


@pytest.mark.skipif(
    not _CANONICAL_WINDOWS_E2E,
    reason="literal-root Windows workstation supervisor contract",
)
def test_windows_job_owner_abrupt_death_kills_child_and_descendant() -> None:
    run_id = _fresh_e2e_id("jobc")
    target_root = (
        runner._EXPECTED_ROOT / "build" / "workspace-local-e2e-targets" / run_id
    )
    target_root.mkdir(parents=True, exist_ok=False)
    ready_path = target_root / "controller-ready.json"
    descendant_pid_path = target_root / "descendant-pid.txt"
    (target_root / "test_job_descendant.py").write_text(
        "import time\n\n"
        "def test_job_descendant_holds():\n"
        "    time.sleep(120)\n",
        encoding="utf-8",
        newline="\n",
    )
    (target_root / "test_job_child.py").write_text(
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n\n"
        "def test_job_child_spawns_descendant():\n"
        "    child = subprocess.Popen([\n"
        "        sys.executable, '-B', '-m', 'pytest',\n"
        "        'test_job_descendant.py', '-q', '-p', 'no:cacheprovider',\n"
        "    ])\n"
        "    Path(os.environ['PFC_TEST_DESCENDANT_PID']).write_text("
        "str(child.pid), encoding='ascii')\n"
        "    time.sleep(120)\n",
        encoding="utf-8",
        newline="\n",
    )
    (target_root / "test_job_controller.py").write_text(
        "import json\n"
        "import os\n"
        "import runpy\n"
        "import subprocess\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "def test_job_controller_owns_kill_on_close_job():\n"
        "    runner = runpy.run_path(os.environ['PFC_TEST_RUNNER_SOURCE'])\n"
        "    job = runner['_create_windows_kill_on_close_job']()\n"
        "    process = subprocess.Popen(\n"
        "        [sys.executable, '-B', '-m', 'pytest', 'test_job_child.py', "
        "'-q', '-p', 'no:cacheprovider'],\n"
        "        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000004,\n"
        "    )\n"
        "    runner['_assign_process_to_windows_job'](job, process)\n"
        "    identity = runner['_new_process_identity'](process.pid, label='job child')\n"
        "    runner['_resume_windows_process'](process)\n"
        "    Path(os.environ['PFC_TEST_CONTROLLER_READY']).write_text(\n"
        "        json.dumps(identity), encoding='utf-8'\n"
        "    )\n"
        "    process.wait()\n",
        encoding="utf-8",
        newline="\n",
    )
    environment = dict(os.environ)
    environment["PFC_TEST_RUNNER_SOURCE"] = str(
        runner._EXPECTED_ROOT / "scripts" / "run_workspace_local.py"
    )
    environment["PFC_TEST_CONTROLLER_READY"] = str(ready_path)
    environment["PFC_TEST_DESCENDANT_PID"] = str(descendant_pid_path)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["TEMP"] = str(target_root)
    environment["TMP"] = str(target_root)
    controller = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "test_job_controller.py",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=target_root,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for(
            lambda: ready_path.is_file() and descendant_pid_path.is_file(),
            timeout_seconds=15,
        )
        child_identity = _read_json(ready_path)
        descendant_identity = runner._new_process_identity(
            int(descendant_pid_path.read_text(encoding="ascii")),
            label="job descendant",
        )
        controller.kill()
        controller.wait(timeout=10)

        identities = (
            (child_identity, "job child"),
            (descendant_identity, "job descendant"),
        )

        def tree_is_inactive() -> bool:
            return all(
                runner._observe_expected_process(identity, label=label)[
                    "exact_identity_active"
                ]
                is False
                for identity, label in identities
            )

        _wait_for(tree_is_inactive, timeout_seconds=10)
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait(timeout=10)


def test_supervisor_timeout_terminalizes_existing_pending_worker_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    run_root = root / "build" / "workspace-local-runs" / "timeout"
    supervisor_root = root / "build" / "workspace-local-supervisors" / "timeout"
    run_root.mkdir(parents=True)
    supervisor_root.mkdir(parents=True)
    receipt_path = run_root / "execution-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _stale_execution_receipt(
                root,
                run_root,
                runner_process={"pid": 999_994, "start_token": "4" * 16},
                target_process=None,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    evidence = runner._terminalize_supervisor_timeout(
        root=root,
        run_id="timeout",
        supervisor_root=supervisor_root,
    )

    terminal = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert evidence is not None
    assert terminal["status"] == (
        "SUPERVISOR_WALL_TIMEOUT_TREE_TERMINATED_NO_AUTHORITY"
    )
    assert terminal["runner_exit_code"] == 124
    assert terminal["supervisor_termination"]["tree_termination_confirmed"] is True
    assert evidence["sha256"] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()


def test_sensitive_and_authority_environment_is_removed(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    run_root, paths = runner.prepare_run(root, "scrub")

    env, removed = runner.build_environment(
        {
            "PATH": "safe",
            "APPDATA": r"C:\Users\worker\AppData\Roaming",
            "API_TOKEN": "secret",
            "PFC_LT_RUNTIME_RECEIPT_PATH": "authority",
            "PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_PATH": "authority-public-key",
            "PFC_DATA_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH": "timestamp-public-key",
            "PFC_DATA_TIMESTAMP_JOURNAL_ID": "authority-journal",
            "PFC_DATA_PUBLICATION_ANCHOR_MTLS_KEY_PATH": r"H:\private.key",
            "PYTHONPATH": r"H:\untrusted",
            "PYTEST_ADDOPTS": "-p untrusted",
            "FMV_DATA_ROOT": r"H:\data",
            "PFC_DATA_ROOT": r"H:\generic",
            "PFC_LT_DATA_ROOT": r"H:\lt",
            "PFC_ENTSOE_DATA_ROOT": r"H:\entsoe",
            "PFC_CT_DATA_ROOT": r"H:\ct",
            "PFC_SHARED_DATA_ROOT": r"H:\shared",
            "PFC_REQUESTS_CA_BUNDLE": r"H:\ambient-ca.pem",
            "LOCALAPPDATA": r"C:\Users\worker\AppData\Local",
            "PROGRAMDATA": r"C:\ProgramData",
        },
        run_root,
        paths,
    )

    assert env["PATH"] == "safe"
    assert Path(env["APPDATA"]) == paths["appdata"]
    assert Path(env["LOCALAPPDATA"]) == paths["local-appdata"]
    assert Path(env["PROGRAMDATA"]) == paths["programdata"]
    assert "API_TOKEN" not in env
    assert "PFC_LT_RUNTIME_RECEIPT_PATH" not in env
    assert "PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_PATH" not in env
    assert "PFC_DATA_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH" not in env
    assert "PFC_DATA_TIMESTAMP_JOURNAL_ID" not in env
    assert "PFC_DATA_PUBLICATION_ANCHOR_MTLS_KEY_PATH" not in env
    assert "PYTHONPATH" not in env
    assert "PYTEST_ADDOPTS" not in env
    assert "FMV_DATA_ROOT" not in env
    assert "PFC_DATA_ROOT" not in env
    assert "PFC_LT_DATA_ROOT" not in env
    assert "PFC_ENTSOE_DATA_ROOT" not in env
    assert "PFC_CT_DATA_ROOT" not in env
    assert "PFC_SHARED_DATA_ROOT" not in env
    assert "PFC_REQUESTS_CA_BUNDLE" not in env
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert removed == [
        "API_TOKEN",
        "APPDATA",
        "FMV_DATA_ROOT",
        "LOCALAPPDATA",
        "PFC_CT_DATA_ROOT",
        "PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_PATH",
        "PFC_DATA_PUBLICATION_ANCHOR_MTLS_KEY_PATH",
        "PFC_DATA_ROOT",
        "PFC_DATA_TIMESTAMP_JOURNAL_ID",
        "PFC_DATA_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH",
        "PFC_ENTSOE_DATA_ROOT",
        "PFC_LT_DATA_ROOT",
        "PFC_LT_RUNTIME_RECEIPT_PATH",
        "PFC_REQUESTS_CA_BUNDLE",
        "PFC_SHARED_DATA_ROOT",
        "PROGRAMDATA",
        "PYTEST_ADDOPTS",
        "PYTHONPATH",
    ]


def test_command_redaction_does_not_persist_secret_value() -> None:
    redacted = runner._redact_command(
        ["python", "-m", "ruff", "--api-key=clear-secret", "--mtls-key", "private.pem"]
    )

    assert redacted[-3:] == ["--api-key=<redacted>", "--mtls-key", "<redacted>"]
    assert "clear-secret" not in json.dumps(redacted)
    assert "private.pem" not in json.dumps(redacted)
