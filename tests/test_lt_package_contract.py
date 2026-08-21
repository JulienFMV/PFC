from __future__ import annotations

import base64
import csv
import hashlib
import importlib
import io
import logging
import os
import subprocess
import sys
import tomllib
import uuid
import zipfile
from pathlib import Path

import pytest

from pfc_shaping.package_contract import (
    ALLOWED_RUNTIME_PYTHON_FILES,
    BUILD_IDENTITY_PLACEHOLDER,
    BUILD_IDENTITY_TEMPLATE,
    REPRODUCIBLE_BUILD_EPOCH,
)
from pfc_shaping.pipeline.production_phases import load_inputs
from pfc_shaping.version import __version__
from scripts.check_lt_wheel_contract import (
    EXPECTED_DIST_INFO,
    EXPECTED_ZIP_DATETIME,
    MAX_WHEEL_BYTES,
    MAX_WHEEL_MEMBERS,
    REQUIRED_FILES,
    audit_wheel,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tmp_path() -> Path:
    """Keep package-contract scratch space under the governed workspace."""

    selected = ROOT / "build" / "test-workspaces" / "lt-package-contract" / uuid.uuid4().hex
    selected.mkdir(parents=True, exist_ok=False)
    return selected


def test_runtime_data_import_does_not_load_excluded_acquisition_modules() -> None:
    import pfc_shaping.data as runtime_data

    excluded = {
        "pfc_shaping.calibration.monthly_curve_lambda_calibration",
        "pfc_shaping.data.databricks_client",
        "pfc_shaping.data.acquisition_signing",
        "pfc_shaping.data.electrification_acquisition",
        "pfc_shaping.data.ep2050",
        "pfc_shaping.data.forward_acquisition_databricks",
        "pfc_shaping.pipeline.short_term_orchestration",
    }
    for name in excluded:
        sys.modules.pop(name, None)

    importlib.reload(runtime_data)

    assert excluded.isdisjoint(sys.modules)
    assert "pfc_shaping/data/acquisition_signing.py" not in ALLOWED_RUNTIME_PYTHON_FILES
    assert (
        "pfc_shaping/data/forward_acquisition_databricks.py"
        not in ALLOWED_RUNTIME_PYTHON_FILES
    )


def test_runtime_forward_parser_has_no_databricks_execution_path() -> None:
    source = (ROOT / "pfc_shaping" / "data" / "ingest_forwards.py").read_text(
        encoding="utf-8"
    )

    assert "databricks_client" not in source
    assert "def load_base_prices(" not in source


def test_pyproject_forbids_generated_console_script_launchers() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "scripts" not in project["project"]
    assert "gui-scripts" not in project["project"]
    assert "entry-points" not in project["project"]
    assert project["tool"]["setuptools"]["include-package-data"] is False
    excluded = set(project["tool"]["setuptools"]["packages"]["find"]["exclude"])
    assert {"pfc_shaping.ct", "pfc_shaping.ct.*"}.issubset(excluded)
    assert (
        "pfc_shaping/pipeline/probabilistic_output_governance.py"
        in ALLOWED_RUNTIME_PYTHON_FILES
    )
    assert "pfc_shaping/pipeline/release_request_contract.py" in ALLOWED_RUNTIME_PYTHON_FILES
    assert "pfc_shaping/cli/audit_ch_lt_estimand_contract.py" in ALLOWED_RUNTIME_PYTHON_FILES
    assert (
        "pfc_shaping/cli/audit_ch_lt_origin_registry_protocol.py"
        in ALLOWED_RUNTIME_PYTHON_FILES
    )
    assert (
        "pfc_shaping/cli/audit_ch_lt_origin_target_mask_inventory.py"
        in ALLOWED_RUNTIME_PYTHON_FILES
    )
    assert (
        "pfc_shaping/cli/audit_ch_lt_successor_candidate_core.py"
        in ALLOWED_RUNTIME_PYTHON_FILES
    )
    assert (
        "pfc_shaping/cli/audit_ch_lt_dependence_power_design.py"
        in ALLOWED_RUNTIME_PYTHON_FILES
    )
    assert "pfc_shaping/cli/audit_ch_market_time_regime.py" in ALLOWED_RUNTIME_PYTHON_FILES
    assert "pfc_shaping/cli/audit_ch_lt_compute_runtime.py" in ALLOWED_RUNTIME_PYTHON_FILES
    assert (
        "pfc_shaping/cli/audit_ch_lt_compute_runtime_manifest.py"
        in ALLOWED_RUNTIME_PYTHON_FILES
    )
    assert (
        "pfc_shaping/cli/verify_ch_lt_preregistration_supersession.py"
        in ALLOWED_RUNTIME_PYTHON_FILES
    )
    assert (
        "pfc_shaping/cli/audit_legacy_provider_resolution.py"
        not in ALLOWED_RUNTIME_PYTHON_FILES
    )
    assert (
        "pfc_shaping/cli/audit_provider_acquisition_quarantine.py"
        not in ALLOWED_RUNTIME_PYTHON_FILES
    )
    assert "pfc_shaping/validation/ch_lt_estimand_contract.py" in ALLOWED_RUNTIME_PYTHON_FILES
    assert (
        "pfc_shaping/validation/ch_lt_origin_registry_protocol.py"
        in ALLOWED_RUNTIME_PYTHON_FILES
    )
    assert "pfc_shaping/validation/ch_market_time_regime.py" in ALLOWED_RUNTIME_PYTHON_FILES
    assert "pfc_shaping/validation/ch_lt_compute_runtime.py" in ALLOWED_RUNTIME_PYTHON_FILES
    assert (
        "pfc_shaping/validation/ch_lt_dependence_power_design.py"
        in ALLOWED_RUNTIME_PYTHON_FILES
    )
    assert (
        "pfc_shaping/validation/ch_lt_origin_target_mask_inventory.py"
        in ALLOWED_RUNTIME_PYTHON_FILES
    )
    assert (
        "pfc_shaping/validation/ch_lt_preregistration_supersession.py"
        in ALLOWED_RUNTIME_PYTHON_FILES
    )
    assert (
        "pfc_shaping/validation/ch_lt_successor_candidate_core.py"
        in ALLOWED_RUNTIME_PYTHON_FILES
    )
    assert (
        "pfc_shaping/validation/ch_lt_successor_candidate_core_v2.py"
        in ALLOWED_RUNTIME_PYTHON_FILES
    )
    assert (
        "pfc_shaping/validation/ch_lt_successor_candidate_core_v3.py"
        in ALLOWED_RUNTIME_PYTHON_FILES
    )
    assert (
        "pfc_shaping/validation/ch_lt_successor_readiness.py"
        in ALLOWED_RUNTIME_PYTHON_FILES
    )
    assert "pfc_shaping/data/eex_forward_vintage_intake.py" in ALLOWED_RUNTIME_PYTHON_FILES
    assert "pfc_shaping/data/eex_datasource_v2_capture.py" in ALLOWED_RUNTIME_PYTHON_FILES
    assert (
        "pfc_shaping/data/ch_lt_origin_registry_reference.py"
        not in ALLOWED_RUNTIME_PYTHON_FILES
    )
    assert (
        "pfc_shaping/cli/eex_forward_vintage_builder.py"
        in ALLOWED_RUNTIME_PYTHON_FILES
    )
    assert (
        "pfc_shaping/validation/ch_lt_compute_runtime_manifest.py"
        in ALLOWED_RUNTIME_PYTHON_FILES
    )
    assert "pfc_shaping/verifier_package_contract.py" not in ALLOWED_RUNTIME_PYTHON_FILES
    assert "pfc_shaping/verifier_runtime_admission.py" not in ALLOWED_RUNTIME_PYTHON_FILES
    prospective_runtime = {
        "pfc_shaping/cli/score_ch_lt_structural_prediction_commitment.py",
        "pfc_shaping/validation/ch_lt_local_future_origin_selection.py",
        "pfc_shaping/validation/ch_lt_native_hourly_truth_bundle.py",
        "pfc_shaping/validation/ch_lt_prospective_hourly_scoring.py",
        "pfc_shaping/validation/ch_lt_structural_prediction_commitment.py",
    }
    assert prospective_runtime.issubset(ALLOWED_RUNTIME_PYTHON_FILES)
    assert prospective_runtime.issubset(REQUIRED_FILES)


def test_sealed_runtime_contains_d147_transitive_dependencies() -> None:
    required = {
        "pfc_shaping/data/lt_input_replay.py",
        "pfc_shaping/data/lt_replay_transforms.py",
        "pfc_shaping/data/snapshot_anchor_client.py",
        "pfc_shaping/data/snapshot_publication_contract.py",
        "pfc_shaping/data/snapshot_publication_state.py",
    }

    assert required.issubset(ALLOWED_RUNTIME_PYTHON_FILES)
    assert "pfc_shaping/data/snapshot_publisher.py" not in ALLOWED_RUNTIME_PYTHON_FILES
    assert (
        "pfc_shaping/data/snapshot_bootstrap_authority.py"
        not in ALLOWED_RUNTIME_PYTHON_FILES
    )
    assert "pfc_shaping/publisher_package_contract.py" not in ALLOWED_RUNTIME_PYTHON_FILES


def test_governed_cli_import_is_checkout_and_ct_independent(tmp_path: Path) -> None:
    code = "\n".join(
        [
            "import os, sys",
            "before_path = list(sys.path)",
            "before_cwd = os.getcwd()",
            "import pfc_shaping.cli.governed_release",
            "import pfc_shaping.cli.audit_ch_lt_compute_runtime",
            "import pfc_shaping.cli.audit_ch_lt_compute_runtime_manifest",
            "import pfc_shaping.cli.audit_ch_lt_dependence_power_design",
            "import pfc_shaping.cli.audit_ch_lt_estimand_contract",
            "import pfc_shaping.cli.audit_ch_lt_origin_registry_protocol",
            "import pfc_shaping.cli.audit_ch_lt_origin_target_mask_inventory",
            "import pfc_shaping.cli.audit_ch_lt_successor_candidate_core",
            "import pfc_shaping.cli.audit_ch_market_time_regime",
            "import pfc_shaping.cli.eex_forward_vintage_builder",
            "import pfc_shaping.cli.verify_ch_lt_preregistration_supersession",
            "import pfc_shaping.cli.audit_legacy_provider_resolution",
            "import pfc_shaping.cli.audit_provider_acquisition_quarantine",
            "import pfc_shaping.cli.score_ch_lt_structural_prediction_commitment",
            "import pfc_shaping.validation.ch_lt_local_future_origin_selection",
            "import pfc_shaping.validation.ch_lt_native_hourly_truth_bundle",
            "import pfc_shaping.validation.ch_lt_prospective_hourly_scoring",
            "import pfc_shaping.validation.ch_lt_structural_prediction_commitment",
            "assert sys.path == before_path",
            "assert os.getcwd() == before_cwd",
            "assert not any(name == 'scripts' or name.startswith('scripts.') for name in sys.modules)",
            "assert not any(name == 'pfc_shaping.ct' or name.startswith('pfc_shaping.ct.') for name in sys.modules)",
            "assert 'pfc_shaping.pipeline.production_phases' not in sys.modules",
        ]
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "wrapper",
    [
        "run_governed_lt_release.py",
        "build_lt_candidate.py",
        "finalize_lt_candidate.py",
        "audit_ch_product_normalization.py",
        "check_monthly_curve_promotion_from_manifests.py",
        "assemble_lt_candidate_product_evidence.py",
    ],
)
def test_compatibility_wrappers_do_not_mutate_import_path(wrapper: str) -> None:
    source = (ROOT / "scripts" / wrapper).read_text(encoding="utf-8")

    assert "sys.path" not in source


def test_config_hash_mismatch_fails_before_data_snapshot_resolution(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("model: {}\n", encoding="utf-8")
    actual = hashlib.sha256(config.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="config SHA-256"):
        load_inputs(
            str(tmp_path),
            logging.getLogger("package-contract"),
            reference_timestamp="2026-07-14T08:00:00+00:00",
            data_root=tmp_path / "missing-data-root",
            config_path=config,
            expected_config_sha256=("a" * 64 if actual != "a" * 64 else "b" * 64),
        )


def test_wheel_auditor_rejects_non_wheel(tmp_path: Path) -> None:
    not_wheel = tmp_path / "artifact.txt"
    not_wheel.write_text("not a wheel", encoding="utf-8")

    with pytest.raises(ValueError, match=".whl"):
        audit_wheel(not_wheel)


def test_wheel_auditor_rejects_file_size_before_zip_parsing(tmp_path: Path) -> None:
    wheel = tmp_path / "oversized.whl"
    wheel.write_bytes(b"x" * (MAX_WHEEL_BYTES + 1))

    result = audit_wheel(wheel)

    assert result["status"] == "FAIL"
    assert result["wheel_sha256"] is None
    assert any("wheel size exceeds resource budget" in error for error in result["errors"])


def test_wheel_auditor_rejects_member_count_before_member_reads(tmp_path: Path) -> None:
    wheel = tmp_path / "too-many-members.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for index in range(MAX_WHEEL_MEMBERS + 1):
            archive.writestr(f"member-{index}.txt", b"x")

    result = audit_wheel(wheel)

    assert result["status"] == "FAIL"
    assert any("member count exceeds resource budget" in error for error in result["errors"])


def test_wheel_auditor_rejects_zip_bomb_ratio_before_member_reads(tmp_path: Path) -> None:
    wheel = tmp_path / "zip-bomb.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("pfc_shaping/version.py", b"0" * (1024 * 1024))

    result = audit_wheel(wheel)

    assert result["status"] == "FAIL"
    assert any("compression-ratio budget" in error for error in result["errors"])


def test_wheel_auditor_rejects_embedded_workstation_paths(tmp_path: Path) -> None:
    wheel = tmp_path / "fmv_pfc_lt-0.14.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in REQUIRED_FILES:
            payload = b'LOCAL = "H:\\\\desk\\\\input.xlsx"\n' if name.endswith("version.py") else b""
            archive.writestr(name, payload)

    result = audit_wheel(wheel)

    assert result["status"] == "FAIL"
    assert any("non-portable runtime paths" in error for error in result["errors"])
    assert any("embedded source revision" in error for error in result["errors"])


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "./pfc_shaping/version.py",
        "pfc_shaping//version.py",
        "pfc_shaping/../outside.py",
    ],
)
def test_wheel_auditor_reports_unsafe_members_without_crashing(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    wheel = _write_test_wheel(
        tmp_path / "fmv_pfc_lt-0.14.0-py3-none-any.whl",
        extra_members={unsafe_name: b'LOCAL = "C:\\\\Users\\\\operator\\\\file"\n'},
    )

    result = audit_wheel(wheel)

    assert result["status"] == "FAIL"
    assert any("unsafe wheel members" in error for error in result["errors"])


def test_wheel_auditor_does_not_scan_documentation_metadata_as_runtime(
    tmp_path: Path,
) -> None:
    metadata_name = f"{EXPECTED_DIST_INFO}/METADATA"
    wheel = _write_test_wheel(
        tmp_path / "fmv_pfc_lt-0.14.0-py3-none-any.whl",
        extra_members={
            metadata_name: _test_metadata() + b'Example: C:\\\\Users\\\\operator\\\\input.xlsx\n'
        },
    )

    result = audit_wheel(wheel)

    assert result["status"] == "PASS", result["errors"]


def test_wheel_auditor_accepts_closed_positive_inventory(tmp_path: Path) -> None:
    wheel = _write_test_wheel(tmp_path / "fmv_pfc_lt-0.14.0-py3-none-any.whl")

    result = audit_wheel(wheel)

    assert result["status"] == "PASS", result["errors"]
    assert result["schema_version"] == "fmv_pfc_lt_wheel_contract.v2"
    assert result["wheel_sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert REPRODUCIBLE_BUILD_EPOCH == 1783987200
    with zipfile.ZipFile(wheel) as archive:
        assert {info.date_time for info in archive.infolist()} == {EXPECTED_ZIP_DATETIME}


def test_path_safety_device_guard_does_not_exempt_workstation_paths(
    tmp_path: Path,
) -> None:
    source = (ROOT / "pfc_shaping/path_safety.py").read_bytes()
    wheel = _write_test_wheel(
        tmp_path / "fmv_pfc_lt-0.14.0-py3-none-any.whl",
        extra_members={
            "pfc_shaping/path_safety.py": source
            + b'\nLOCAL_WORKSTATION_PATH = "C:\\\\Users\\\\operator\\\\input.xlsx"\n'
        },
    )

    result = audit_wheel(wheel)

    assert result["status"] == "FAIL"
    assert any("non-portable runtime paths" in error for error in result["errors"])


def test_wheel_auditor_rejects_code_in_build_identity(tmp_path: Path) -> None:
    wheel = _write_test_wheel(
        tmp_path / "fmv_pfc_lt-0.14.0-py3-none-any.whl",
        identity_suffix="raise RuntimeError('executed before identity check')\n",
    )

    result = audit_wheel(wheel)

    assert result["status"] == "FAIL"
    assert any("embedded source revision" in error for error in result["errors"])


def test_wheel_auditor_rejects_pth_even_with_valid_record(tmp_path: Path) -> None:
    wheel = _write_test_wheel(
        tmp_path / "fmv_pfc_lt-0.14.0-py3-none-any.whl",
        extra_members={"bootstrap.pth": b"import os\n"},
    )

    result = audit_wheel(wheel)

    assert result["status"] == "FAIL"
    assert any("unexpected wheel members" in error for error in result["errors"])


def test_wheel_auditor_rejects_console_script_metadata_even_with_valid_record(
    tmp_path: Path,
) -> None:
    wheel = _write_test_wheel(
        tmp_path / "fmv_pfc_lt-0.14.0-py3-none-any.whl",
        extra_members={
            f"{EXPECTED_DIST_INFO}/entry_points.txt": (
                b"[console_scripts]\npfc-lt = "
                b"pfc_shaping.cli.governed_release:main\n"
            )
        },
    )

    result = audit_wheel(wheel)

    assert result["status"] == "FAIL"
    assert any("entry point metadata is forbidden" in error for error in result["errors"])


def _write_test_wheel(
    path: Path,
    *,
    identity_suffix: str = "",
    extra_members: dict[str, bytes] | None = None,
) -> Path:
    members = {
        name: (ROOT / name).read_bytes()
        for name in ALLOWED_RUNTIME_PYTHON_FILES
        if name != "pfc_shaping/build_identity.py"
    }
    digest = hashlib.sha256()
    identity_placeholder = BUILD_IDENTITY_TEMPLATE.format(
        source_revision=BUILD_IDENTITY_PLACEHOLDER
    ).encode("ascii")
    for name in sorted(ALLOWED_RUNTIME_PYTHON_FILES):
        relative = name.removeprefix("pfc_shaping/")
        payload = identity_placeholder if name.endswith("build_identity.py") else members[name]
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    revision = digest.hexdigest()
    members["pfc_shaping/build_identity.py"] = (
        BUILD_IDENTITY_TEMPLATE.format(source_revision=revision) + identity_suffix
    ).encode("ascii")
    members[f"{EXPECTED_DIST_INFO}/METADATA"] = _test_metadata()
    members[f"{EXPECTED_DIST_INFO}/WHEEL"] = (
        "Wheel-Version: 1.0\n"
        "Generator: setuptools (80.9.0)\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n\n"
    ).encode("ascii")
    members[f"{EXPECTED_DIST_INFO}/top_level.txt"] = b"pfc_shaping\n"
    members.update(extra_members or {})
    record_name = f"{EXPECTED_DIST_INFO}/RECORD"
    rows = []
    for name, payload in sorted(members.items()):
        encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
        rows.append([name, f"sha256={encoded.decode('ascii')}", str(len(payload))])
    rows.append([record_name, "", ""])
    record = io.StringIO(newline="")
    csv.writer(record, lineterminator="\n").writerows(rows)
    members[record_name] = record.getvalue().encode("utf-8")
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in sorted(members.items()):
            archive.writestr(zipfile.ZipInfo(name, date_time=EXPECTED_ZIP_DATETIME), payload)
    return path


def _test_metadata() -> bytes:
    requirements = [
        "cryptography==47.0.0",
        "holidays==0.83",
        "numpy==2.0.2",
        "openpyxl==3.1.5",
        "pandas==2.3.3",
        "pyarrow==21.0.0",
        "PyYAML==6.0.3",
        "scikit-learn==1.6.1",
        "scipy==1.13.1",
        'duckdb==1.5.0; extra == "dashboard"',
        'plotly==6.6.0; extra == "dashboard"',
        'streamlit==1.50.0; extra == "dashboard"',
        'matplotlib==3.8.4; extra == "validation"',
        'statsmodels==0.14.6; extra == "validation"',
    ]
    headers = [
        "Metadata-Version: 2.4",
        "Name: fmv-pfc-lt",
        f"Version: {__version__}",
        "Summary: Governed Swiss long-term price forward curve pipeline for FMV",
        "Author: FMV SA",
        "License-Expression: LicenseRef-Proprietary",
        "Requires-Python: <3.12,>=3.11",
        "Description-Content-Type: text/markdown",
    ]
    headers.extend(f"Requires-Dist: {value}" for value in requirements[:9])
    headers.extend(
        [
            "Provides-Extra: dashboard",
            *(f"Requires-Dist: {value}" for value in requirements[9:12]),
            "Provides-Extra: validation",
            *(f"Requires-Dist: {value}" for value in requirements[12:]),
        ]
    )
    return ("\n".join(headers) + "\n\n# FMV PFC LT\n").encode("utf-8")
