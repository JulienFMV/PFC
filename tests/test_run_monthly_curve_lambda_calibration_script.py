from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_monthly_curve_lambda_calibration import main


def test_cli_rejects_reconstructed_latest_revision_history_without_outputs(
    tmp_path: Path,
    capsys,
) -> None:
    history = tmp_path / "legacy.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-01"]),
            "product": ["2027"],
            "load_type": ["BASE"],
            "product_type": ["Cal"],
            "price": [80.0],
            "market": ["CH"],
            "source": ["EEX"],
        }
    ).to_parquet(history, index=False)
    grid = tmp_path / "grid.yaml"
    grid.write_text(
        """
defaults:
  constraint_tolerance: 0.01
grid:
  lambda_smooth_month: [1.0]
  lambda_smooth_yoy: [0.0]
  lambda_shape: [0.0]
  neighbor_shrinkage: [0.5]
""",
        encoding="utf-8",
    )
    output = tmp_path / "output"

    rc = main(
        [
            "--forwards",
            str(history),
            "--grid",
            str(grid),
            "--output-dir",
            str(output),
            "--smoke",
        ]
    )

    stdout = capsys.readouterr().out
    assert rc == 2
    assert "final_status=FAIL_MISSING_HISTORICAL_VINTAGE_CATALOG" in stdout
    assert "production_approved=False" in stdout
    assert not output.exists()


def test_cli_rejects_manual_origin_reduction_outside_smoke(
    tmp_path: Path,
    capsys,
) -> None:
    grid = tmp_path / "grid.yaml"
    grid.write_text(
        """
defaults:
  constraint_tolerance: 0.01
grid:
  lambda_smooth_month: [1.0]
  lambda_smooth_yoy: [0.0]
  lambda_shape: [0.0]
  neighbor_shrinkage: [0.5]
""",
        encoding="utf-8",
    )

    rc = main(["--grid", str(grid), "--max-origins", "1"])

    assert rc == 2
    assert "final_status=FAIL_RESEARCH_REDUCTION_OUTSIDE_SMOKE" in (
        capsys.readouterr().out
    )
