from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

import run_pfc_production
from pfc_shaping.pipeline import rolling_update
from pfc_shaping.pipeline.production_phases import save_long_term_outputs
from scripts import build_lt_candidate
from scripts import run_daily


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_direct_production_entrypoint_is_disabled() -> None:
    with pytest.raises(RuntimeError, match="Legacy direct publication is disabled"):
        run_pfc_production.main()


def test_direct_long_term_output_sink_is_disabled() -> None:
    with pytest.raises(RuntimeError, match="Direct LT output publication is disabled"):
        save_long_term_outputs(None, None)  # type: ignore[arg-type]


def test_candidate_builder_captures_governance_before_solver() -> None:
    source = inspect.getsource(build_lt_candidate.build_candidate)

    assert source.index("capture_pre_run_governance_evidence") < source.index(
        "run_long_term_phase"
    )
    assert "finalize_candidate_staging" not in source


def test_legacy_daily_python_publisher_is_disabled() -> None:
    with pytest.raises(RuntimeError, match="Legacy rolling_update publication is disabled"):
        run_daily.run_pfc_production()


def test_legacy_rolling_update_module_is_disabled_before_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def mark_io(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(rolling_update, "setup_logging", mark_io)

    with pytest.raises(RuntimeError, match="Legacy rolling_update publication is disabled"):
        rolling_update.run_update()
    assert called is False


def test_legacy_rolling_update_module_execution_fails(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [sys.executable, "-m", "pfc_shaping.pipeline.rolling_update"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Legacy rolling_update publication is disabled" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_legacy_rolling_update_direct_file_execution_fails(tmp_path: Path) -> None:
    source = ROOT / "pfc_shaping" / "pipeline" / "rolling_update.py"
    result = subprocess.run(
        [sys.executable, str(source)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Legacy rolling_update publication is disabled" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_retired_rolling_update_implementation_is_not_callable() -> None:
    with pytest.raises(RuntimeError, match="Retired rolling_update implementation"):
        rolling_update._run_update_legacy()


def test_rolling_update_tombstone_contains_no_ompex_or_publisher_logic() -> None:
    source = inspect.getsource(rolling_update).lower()

    forbidden = (
        "benchmark_against_hfc",
        "hfc_benchmark_dir",
        "benchmark_policy",
        "upsert_run_and_forecast",
        "export_both",
        "pfcassembler",
    )
    assert all(token not in source for token in forbidden)
    assert "candidate/audit/receipt/promotion" in source


def test_rolling_update_compatibility_symbols_are_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="logging is disabled"):
        rolling_update.setup_logging("20260714")


def test_legacy_daily_main_stops_before_ingestion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def mark_ingestion(*, days_back: int) -> dict[str, str]:
        nonlocal called
        called = True
        return {"days_back": str(days_back)}

    monkeypatch.setattr(run_daily, "ingest_data", mark_ingestion)

    with pytest.raises(RuntimeError, match="Legacy daily ingestion/publication is disabled"):
        run_daily.main()
    assert called is False


@pytest.mark.parametrize(
    "relative_path",
    [
        "pfc_shaping/tools/run_daily_pfc.ps1",
        "pfc_shaping/tools/register_daily_task.ps1",
    ],
)
def test_legacy_powershell_publishers_fail_before_execution(relative_path: str) -> None:
    content = (ROOT / relative_path).read_text(encoding="utf-8")
    throw_position = content.index("throw \"")
    execution_markers = [
        position
        for marker in ("Set-Location", "Register-ScheduledTask")
        if (position := content.find(marker)) >= 0
    ]

    assert execution_markers
    assert throw_position < min(execution_markers)
