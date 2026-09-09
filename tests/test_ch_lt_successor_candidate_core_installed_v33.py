from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PREFIX = ROOT / "build" / "conda-runtime-v33-core-v3-base"
RUNTIME = RUNTIME_PREFIX / "python.exe"
RUNTIME_RECEIPT = ROOT / "build" / "launcherless-runtime-receipt-20260730-v33.json"
RUNTIME_RECEIPT_SHA256 = (
    "3e9af2e9b3ccbb9092e84aa26beaa7b499ba7397e1b1c1b5eaf0d4d0bbd7475d"
)
PHASE = ROOT / ".planning" / "phases" / "14-lt-audit-remediation"
CORE = PHASE / "CH-LT-SUCCESSOR-CANDIDATE-CORE-V3-20260730.json"
CORE_SHA256 = "e2c2a6f9d3ca677991f3f76f03dec1328492e2f60a4015b7796a2ff6aa6f905a"
DESIGN = PHASE / "CH-LT-DEPENDENCE-POWER-DESIGN-DRAFT-V1-20260730.json"
DESIGN_SHA256 = "005b8655b817db10e7f3c227b1c5912d545305b68e262ab82d9aa0b5817a6a91"
WHEEL_SHA256 = "7fa4e36160717bbd535f717b2f62d95098fd547f8286c6085a64c88ab8969de7"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_module(
    *, module: str, arguments: list[str], cwd: Path, env: dict[str, str]
) -> dict[str, object]:
    result = subprocess.run(
        [str(RUNTIME), "-I", "-B", "-m", module, *arguments],
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, dict)
    return parsed


def test_installed_v33_core_and_power_clis_are_isolated_and_no_go(
    tmp_path: Path,
) -> None:
    assert _sha256(RUNTIME_RECEIPT) == RUNTIME_RECEIPT_SHA256
    assert _sha256(CORE) == CORE_SHA256
    assert _sha256(DESIGN) == DESIGN_SHA256
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

    core = _run_module(
        module="pfc_shaping.cli.audit_ch_lt_successor_candidate_core",
        arguments=[
            "--evidence-root",
            str(ROOT),
            "--core",
            str(CORE),
            "--expected-core-sha256",
            CORE_SHA256,
        ],
        cwd=tmp_path,
        env=env,
    )
    assert core["status"] == (
        "LOCAL_HASH_CLOSED_OUTCOME_BLIND_CONTRAST_AWARE_CORE_"
        "NOT_EXTERNALLY_FROZEN_NOT_ADMITTED"
    )
    assert core["candidate_core_admitted"] is False
    assert core["execution_authorized"] is False
    assert core["production_authorization"] is False
    assert core["promotion_gate"] is False

    design = _run_module(
        module="pfc_shaping.cli.audit_ch_lt_dependence_power_design",
        arguments=[
            "--evidence-root",
            str(ROOT),
            "--design",
            str(DESIGN),
            "--expected-design-sha256",
            DESIGN_SHA256,
        ],
        cwd=tmp_path,
        env=env,
    )
    assert design["status"] == (
        "LOCAL_OUTCOME_BLIND_POWER_DESIGN_DRAFT_INCOMPLETE_NOT_EXTERNALLY_FROZEN_NO_GO"
    )
    assert design["power_design_complete"] is False
    assert design["execution_authorized"] is False
    assert design["production_authorization"] is False
    assert design["promotion_gate"] is False
