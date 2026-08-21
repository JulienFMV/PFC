from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / "build" / "conda-runtime-v38b-origin-registry-v2-base"
RUNTIME = PREFIX / "python.exe"
RUNTIME_RECEIPT = (
    ROOT
    / "build"
    / "launcherless-runtime-receipt-20260730-v38-origin-registry-v2.json"
)
RUNTIME_RECEIPT_SHA256 = (
    "c999a5d139a1a4be1c004d2697a5fef9a262c297eab60d3b77cc4f01591e58dd"
)
WHEEL_SHA256 = "489f0437386435bfb2849155fde755ce150efdc1e6a556a5d174ab3110e6e808"
SOURCE_REVISION = "20af143fe257784a829e9b9c461cd52a3e2ef6bef39ec07c3b8c572a36964e79"
PROTOCOL = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "CH-LT-ORIGIN-REGISTRY-PROTOCOL-DRAFT-V2-20260730.json"
)
PROTOCOL_SHA256 = "6ea896ccdb35414b52237f2bcf1065755c3c10444b308ce905b60f472e68c697"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_installed_v38_audits_protocol_v2_with_exact_sealed_sys_path(
    tmp_path: Path,
) -> None:
    assert _sha256(RUNTIME_RECEIPT) == RUNTIME_RECEIPT_SHA256
    receipt = json.loads(RUNTIME_RECEIPT.read_text(encoding="utf-8"))
    expected_sys_path = [
        str(PREFIX / "Lib"),
        str(PREFIX / "DLLs"),
        str(PREFIX / "governed-site-packages"),
    ]
    assert receipt["status"] == "PASS"
    assert receipt["project_source_revision"] == SOURCE_REVISION
    assert receipt["sys_path"] == expected_sys_path
    assert receipt["local_quality_authorization"] is True
    assert receipt["production_authorization"] is False
    project = [
        source
        for source in receipt["sources"]
        if source.get("kind") == "GOVERNED_PROJECT_WHEEL"
    ]
    assert len(project) == 1
    assert project[0]["sha256"] == WHEEL_SHA256
    assert project[0]["source_revision"] == SOURCE_REVISION

    for relative in (
        "pfc_shaping/validation/ch_lt_origin_registry_protocol.py",
        "pfc_shaping/cli/audit_ch_lt_origin_registry_protocol.py",
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
    assert ast.literal_eval(match.group(1)) == expected_sys_path

    completed = subprocess.run(
        [
            str(RUNTIME),
            "-I",
            "-B",
            "-m",
            "pfc_shaping.cli.audit_ch_lt_origin_registry_protocol",
            "--evidence-root",
            str(ROOT),
            "--protocol",
            str(PROTOCOL),
            "--expected-protocol-sha256",
            PROTOCOL_SHA256,
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    result = json.loads(completed.stdout)
    assert (
        result["status"]
        == "LOCAL_OUTCOME_BLIND_ORIGIN_REGISTRY_PROTOCOL_DRAFT_INCOMPLETE_NO_GO"
    )
    assert result["schema_version"] == "ch_lt_origin_registry_protocol_assessment.v2"
    assert result["protocol_hash_closed_locally"] is True
    assert result["registry_implemented"] is False
    assert result["countable_prospective_origin_count"] == 0
    assert result["truth_open_authorized"] is False
    assert result["execution_authorized"] is False
    assert result["production_authorization"] is False
    assert result["promotion_gate"] is False
