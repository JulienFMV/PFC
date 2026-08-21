from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / "build" / "conda-runtime-v37-origin-registry-base"
RUNTIME = PREFIX / "python.exe"
RUNTIME_RECEIPT = (
    ROOT / "build" / "launcherless-runtime-receipt-20260730-v37-origin-registry.json"
)
RUNTIME_RECEIPT_SHA256 = (
    "44a69422a06d9ee4ec9d01aa9989e1bb150b6e96b0188bad2bf906823d8a1dbb"
)
WHEEL_SHA256 = "62b1cb371f3e03e887488aebfe795c66a5b0b36332f009712e7428ef67bc6752"
SOURCE_REVISION = "317da4000afbcd0938af076659249088f648bd558ce5cbd23c0bae8db95da9ca"
PROTOCOL = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "CH-LT-ORIGIN-REGISTRY-PROTOCOL-DRAFT-V1-20260730.json"
)
PROTOCOL_SHA256 = "67e0b63a6eca5ec843847725b2f3097f7f0fbeeac615bcdb2b2c8be92e1b43fc"
INSTALLED_SOURCE_SHA256 = {
    "pfc_shaping/validation/ch_lt_origin_registry_protocol.py": (
        "66513b01d2053822fc864dfaed9b33fa9e02e6486a345dd9600b9c269941cdcb"
    ),
    "pfc_shaping/cli/audit_ch_lt_origin_registry_protocol.py": (
        "c29458f0be640b2cbac06d8053688138f6351ad0862f3a08cff293556f6eb67e"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_installed_v37_audits_protocol_with_exact_sealed_sys_path(
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

    for relative, expected_sha256 in INSTALLED_SOURCE_SHA256.items():
        assert _sha256(PREFIX / "governed-site-packages" / relative) == expected_sha256

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
    assert result["protocol_hash_closed_locally"] is True
    assert result["registry_implemented"] is False
    assert result["countable_prospective_origin_count"] == 0
    assert result["truth_open_authorized"] is False
    assert result["execution_authorized"] is False
    assert result["production_authorization"] is False
    assert result["promotion_gate"] is False
