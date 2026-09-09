from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.backtest_intraday_shape_estimator import run_backtest


def _fixture(tmp_path: Path) -> tuple[Path, str, Path, str]:
    index = pd.date_range("2025-10-01T00:00:00Z", periods=80 * 24 * 4, freq="15min")
    parent = np.repeat(np.arange(len(index) // 4), 4)
    hour_level = 50.0 + 10.0 * np.sin(parent / 24.0)
    factors = np.tile(np.array([0.8, 0.95, 1.05, 1.2]), len(index) // 4)
    panel = pd.DataFrame({"price_eur_mwh": hour_level * factors}, index=index)
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path)
    panel_sha = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "local_intraday_calibration_panel.v3",
        "status": "COMPLETE_LOCAL_MIXED_AUTHORITY_NO_GO",
        "production_authorization": False,
        "promotion_gate": False,
        "scientific_admission": False,
        "training_eligible": False,
        "model_selection_eligible": False,
        "confirmatory_rolling_origin_eligible": False,
        "future_holdout_eligible": False,
        "t057_eligible": False,
        "candidate_assembly_eligible": False,
        "local_diagnostic_allowed": True,
        "resolution_provenance": {
            "native_quarter_hour_truth_eligible": False,
        },
        "panel": {"sha256": panel_sha},
        "limitations": ["fixture"],
    }
    manifest_path = tmp_path / "panel-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return panel_path, panel_sha, manifest_path, manifest_sha


def test_run_backtest_binds_inputs_and_writes_summary_last(tmp_path: Path):
    panel, panel_sha, manifest, manifest_sha = _fixture(tmp_path)
    output = tmp_path / "out"

    summary = run_backtest(
        panel_path=panel,
        expected_panel_sha256=panel_sha,
        panel_manifest_path=manifest,
        expected_panel_manifest_sha256=manifest_sha,
        output_dir=output,
        origin_start=pd.Timestamp("2025-11-15T00:00:00Z"),
        origin_end=pd.Timestamp("2025-12-05T00:00:00Z"),
        fold_days=7,
        minimum_train_days=30,
    )

    assert summary["production_authorization"] is False
    assert summary["promotion_gate"] is False
    assert summary["scientific_admission"] is False
    assert summary["confirmatory_rolling_origin_eligible"] is False
    assert summary["local_diagnostic_allowed"] is True
    assert summary["input"]["panel_sha256"] == panel_sha
    assert summary["fold_count"] >= 1
    assert (output / "fold_metrics.csv").is_file()
    assert json.loads((output / "summary.json").read_text(encoding="ascii")) == summary


def test_run_backtest_rejects_wrong_caller_held_panel_hash(tmp_path: Path):
    panel, _, manifest, manifest_sha = _fixture(tmp_path)

    with pytest.raises(ValueError, match="panel SHA-256 mismatch"):
        run_backtest(
            panel_path=panel,
            expected_panel_sha256="0" * 64,
            panel_manifest_path=manifest,
            expected_panel_manifest_sha256=manifest_sha,
            output_dir=tmp_path / "out",
            origin_start=pd.Timestamp("2025-11-15T00:00:00Z"),
            origin_end=pd.Timestamp("2025-12-05T00:00:00Z"),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("status", "COMPLETE"),
        ("production_authorization", True),
        ("promotion_gate", True),
        ("scientific_admission", True),
        ("training_eligible", True),
        ("model_selection_eligible", True),
        ("confirmatory_rolling_origin_eligible", True),
        ("future_holdout_eligible", True),
        ("t057_eligible", True),
        ("candidate_assembly_eligible", True),
        ("local_diagnostic_allowed", False),
    ),
)
def test_run_backtest_rejects_panel_without_exact_local_diagnostic_no_go(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    panel, panel_sha, manifest_path, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="explicit local diagnostic NO_GO"):
        run_backtest(
            panel_path=panel,
            expected_panel_sha256=panel_sha,
            panel_manifest_path=manifest_path,
            expected_panel_manifest_sha256=manifest_sha,
            output_dir=tmp_path / "out",
            origin_start=pd.Timestamp("2025-11-15T00:00:00Z"),
            origin_end=pd.Timestamp("2025-12-05T00:00:00Z"),
        )
