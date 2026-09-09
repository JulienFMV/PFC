from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PREFIX = ROOT / "build" / "conda-runtime-v31-core-base"
RUNTIME = RUNTIME_PREFIX / "python.exe"
RUNTIME_RECEIPT = ROOT / "build" / "launcherless-runtime-receipt-20260730-v31.json"
RUNTIME_RECEIPT_SHA256 = (
    "4cc404d6fdceee4f1b41384ce1366cb9177b318c8074bbf740515519130e5323"
)
CORE = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "CH-LT-SUCCESSOR-CANDIDATE-CORE-V2-20260730.json"
)
CORE_SHA256 = "a2ec1d758043b7ed4bf111e99ef4f87d814ad292b7b3c115c36de30d1da4011e"
WHEEL_SHA256 = "3e3f4e24c23f36ac0f4bd43a77d46d23ed23e846c0f329ba922c8534cb89f00b"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_installed_v31_core_cli_is_isolated_and_non_authoritative(tmp_path: Path) -> None:
    assert _sha256(RUNTIME_RECEIPT) == RUNTIME_RECEIPT_SHA256
    assert _sha256(CORE) == CORE_SHA256
    runtime_receipt = json.loads(RUNTIME_RECEIPT.read_text(encoding="utf-8"))
    assert runtime_receipt["production_authorization"] is False
    project_sources = [
        source
        for source in runtime_receipt["sources"]
        if source.get("kind") == "GOVERNED_PROJECT_WHEEL"
    ]
    assert len(project_sources) == 1
    assert project_sources[0]["sha256"] == WHEEL_SHA256

    env = os.environ.copy()
    env["PFC_LT_RUNTIME_RECEIPT_PATH"] = str(RUNTIME_RECEIPT)
    env["PFC_LT_RUNTIME_RECEIPT_SHA256"] = RUNTIME_RECEIPT_SHA256
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
    assert ast.literal_eval(match.group(1)) == [
        str(RUNTIME_PREFIX / "Lib"),
        str(RUNTIME_PREFIX / "DLLs"),
        str(RUNTIME_PREFIX / "governed-site-packages"),
    ]

    audit = subprocess.run(
        [
            str(RUNTIME),
            "-I",
            "-B",
            "-m",
            "pfc_shaping.cli.audit_ch_lt_successor_candidate_core",
            "--evidence-root",
            str(ROOT),
            "--core",
            str(CORE),
            "--expected-core-sha256",
            CORE_SHA256,
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert audit.returncode == 0, audit.stderr
    result = json.loads(audit.stdout)
    assert result["status"] == (
        "LOCAL_HASH_CLOSED_OUTCOME_BLIND_CORE_NOT_EXTERNALLY_FROZEN_NOT_ADMITTED"
    )
    assert result["candidate_core_frozen"] is False
    assert result["candidate_core_admitted"] is False
    assert result["successor_exists"] is False
    assert result["execution_authorized"] is False
    assert result["production_authorization"] is False
    assert result["promotion_gate"] is False
