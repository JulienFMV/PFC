from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / "build" / "conda-runtime-v36-origin-mask-schema-v2-base"
RUNTIME = PREFIX / "python.exe"
RUNTIME_RECEIPT = (
    ROOT
    / "build"
    / "launcherless-runtime-receipt-20260730-v36-origin-mask-schema-v2.json"
)
RUNTIME_RECEIPT_SHA256 = (
    "a23f1af134aa374db038aba5c37935167d4fd2f6b471a64ef72fe6fdf605bbe4"
)
WHEEL_SHA256 = "14c4e44127409a30b3110d07432bcaecd4171db4e669beaf8dd4eac99162bbf0"
SOURCE_REVISION = "036219d574b2f675fd6c06c5c8fd0417228acc54a63b76abfa97002befec8024"
INVENTORY = (
    ROOT
    / "build"
    / "ch-lt-origin-target-mask-inventories"
    / "structural-dry-run-20260730T120000Z-schema-v2.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_installed_cli(
    arguments: list[str], *, env: dict[str, str], timeout: int = 600
) -> dict[str, object]:
    completed = subprocess.run(
        [
            str(RUNTIME),
            "-I",
            "-B",
            "-m",
            "pfc_shaping.cli.audit_ch_lt_origin_target_mask_inventory",
            "--evidence-root",
            str(ROOT),
            *arguments,
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    return json.loads(completed.stdout)


def test_installed_v36_builds_then_audits_schema_v2_inventory(
    tmp_path: Path,
) -> None:
    assert _sha256(RUNTIME_RECEIPT) == RUNTIME_RECEIPT_SHA256
    receipt = json.loads(RUNTIME_RECEIPT.read_text(encoding="utf-8"))
    assert receipt["project_source_revision"] == SOURCE_REVISION
    assert receipt["production_authorization"] is False
    assert receipt["sys_path"] == [
        str(PREFIX / "Lib"),
        str(PREFIX / "DLLs"),
        str(PREFIX / "governed-site-packages"),
    ]
    project = [
        source
        for source in receipt["sources"]
        if source.get("kind") == "GOVERNED_PROJECT_WHEEL"
    ]
    assert len(project) == 1
    assert project[0]["sha256"] == WHEEL_SHA256
    assert project[0]["source_revision"] == SOURCE_REVISION

    for relative in (
        "pfc_shaping/validation/ch_lt_origin_target_mask_inventory.py",
        "pfc_shaping/cli/audit_ch_lt_origin_target_mask_inventory.py",
    ):
        assert _sha256(ROOT / relative) == _sha256(
            PREFIX / "governed-site-packages" / relative
        )

    env = os.environ.copy()
    env["PFC_LT_RUNTIME_RECEIPT_PATH"] = str(RUNTIME_RECEIPT)
    env["PFC_LT_RUNTIME_RECEIPT_SHA256"] = RUNTIME_RECEIPT_SHA256
    env["TEMP"] = str(tmp_path)
    env["TMP"] = str(tmp_path)
    env["PYTHONNOUSERSITE"] = "1"
    site = subprocess.run(
        [str(RUNTIME), "-I", "-B", "-m", "site"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert site.returncode == 0, site.stderr
    match = re.search(r"sys\.path = (\[.*?\])\r?\nUSER_BASE", site.stdout, re.DOTALL)
    assert match is not None
    assert ast.literal_eval(match.group(1)) == receipt["sys_path"]

    built = _run_installed_cli(
        [
            "build",
            "--origin-as-of-utc",
            "2026-07-30T12:00:00Z",
            "--output",
            str(INVENTORY),
        ],
        env=env,
    )
    assert built["governance_bindings_verified"] is True
    assert built["inventory_path"] == str(INVENTORY)
    assert built["inventory_sha256"] == _sha256(INVENTORY)

    audited = _run_installed_cli(
        [
            "audit",
            "--inventory",
            str(INVENTORY),
            "--expected-inventory-sha256",
            str(built["inventory_sha256"]),
        ],
        env=env,
    )
    assert audited["inventory_sha256"] == built["inventory_sha256"]
    assert audited["governance_bindings_verified"] is True
    assert audited["countable_prospective_origin"] is False
    assert audited["evidence_slot_satisfied"] is False
    assert audited["outcomes_opened"] is False
    assert audited["scientific_admission"] is False
    assert audited["execution_authorized"] is False
    assert audited["production_authorization"] is False
    assert audited["promotion_gate"] is False

    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    assert inventory["schema_version"] == "ch_lt_origin_target_mask_inventory.v2"
    assert inventory["inventory_name"].endswith("_v2")
    targets = inventory["targets"]
    assert len(targets) == 36
    assert all(
        target["truth_state"] == "NOT_AVAILABLE_OR_NOT_BOUND_NOT_READ"
        for target in targets
    )
