from __future__ import annotations

import hashlib
import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PREFIX = ROOT / "build" / "conda-runtime-v34-peak-base"
RUNTIME = RUNTIME_PREFIX / "python.exe"
RUNTIME_RECEIPT = (
    ROOT / "build" / "launcherless-runtime-receipt-20260730-v34-peak.json"
)
RUNTIME_RECEIPT_SHA256 = (
    "8f6f36e36a2707b1c71ff98cce20691e8c648bdda02eeb5f6ffa1335c8c4d405"
)
WHEEL_SHA256 = "cfd9a8db3c7a05154dfac45b473201aa29e0069c7e3e005a8fa7acc2e6ae4388"
SOURCE_REVISION = "545fe6d9be82aba0069766685a0165de7b6fea20d487dd2d29cbd345feb2834c"
SHAPE_CONSTRAINTS_SHA256 = (
    "9298534b42d7fde3c867394a237c4b4665dd15273bde56568e5077eaf27798ad"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_installed_v34_peak_offpeak_runtime_is_isolated_and_exact(
    tmp_path: Path,
) -> None:
    assert _sha256(RUNTIME_RECEIPT) == RUNTIME_RECEIPT_SHA256
    runtime_receipt = json.loads(RUNTIME_RECEIPT.read_text(encoding="utf-8"))
    assert runtime_receipt["status"] == "PASS"
    assert runtime_receipt["project_source_revision"] == SOURCE_REVISION
    assert runtime_receipt["production_authorization"] is False
    project_sources = [
        source
        for source in runtime_receipt["sources"]
        if source.get("kind") == "GOVERNED_PROJECT_WHEEL"
    ]
    assert len(project_sources) == 1
    assert project_sources[0]["sha256"] == WHEEL_SHA256
    installed_shape = (
        RUNTIME_PREFIX
        / "governed-site-packages"
        / "pfc_shaping"
        / "lt"
        / "model"
        / "shape_constraints.py"
    )
    assert _sha256(installed_shape) == SHAPE_CONSTRAINTS_SHA256
    assert _sha256(ROOT / "pfc_shaping" / "lt" / "model" / "shape_constraints.py") == (
        SHAPE_CONSTRAINTS_SHA256
    )

    probe = textwrap.dedent(
        """
        import json
        import sys

        import numpy as np
        import pandas as pd

        from pfc_shaping.lt.model.shape_constraints import (
            build_base_peak_offpeak_constraint_system,
            eex_peak_mask,
        )

        start = pd.Timestamp("2030-01-01", tz="Europe/Zurich").tz_convert("UTC")
        end = pd.Timestamp("2030-04-01", tz="Europe/Zurich").tz_convert("UTC")
        index = pd.date_range(start, end, freq="h", inclusive="left")
        system = build_base_peak_offpeak_constraint_system(
            index,
            {"2030-Q1": 80.0, "2030-01": 90.0},
            {"2030-Q1": 100.0, "2030-01": 110.0},
            country="CH",
        )
        curve, *_ = np.linalg.lstsq(system.matrix, system.targets, rcond=None)
        local = index.tz_convert("Europe/Zurich")
        peak = eex_peak_mask(index, country="CH")
        q1 = local.quarter == 1
        january = local.month == 1
        print(json.dumps({
            "sys_path": sys.path,
            "hour_count": len(index),
            "constraint_max_abs_error": float(
                np.max(np.abs(system.matrix @ curve - system.targets))
            ),
            "base_q1": float(curve[q1].mean()),
            "base_january": float(curve[january].mean()),
            "peak_q1": float(curve[q1 & peak].mean()),
            "peak_january": float(curve[january & peak].mean()),
        }, sort_keys=True))
        """
    )
    env = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP"):
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [str(RUNTIME), "-I", "-B", "-c", probe],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    observed = json.loads(result.stdout)
    assert observed["sys_path"] == [
        str(RUNTIME_PREFIX / "Lib"),
        str(RUNTIME_PREFIX / "DLLs"),
        str(RUNTIME_PREFIX / "governed-site-packages"),
    ]
    assert observed["hour_count"] == 2159
    assert observed["constraint_max_abs_error"] <= 1e-9
    assert observed["base_q1"] == pytest.approx(80.0, abs=1e-9)
    assert observed["base_january"] == pytest.approx(90.0, abs=1e-9)
    assert observed["peak_q1"] == pytest.approx(100.0, abs=1e-9)
    assert observed["peak_january"] == pytest.approx(110.0, abs=1e-9)
