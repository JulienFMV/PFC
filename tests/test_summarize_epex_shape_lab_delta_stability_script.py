from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.summarize_epex_shape_lab_delta_stability import summarize_delta_stability


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_case(
    tmp_path: Path,
    label: str,
    *,
    delta_shift: float = 0.0,
    weekend_intensity: float = 0.5,
) -> tuple[str, Path, Path, Path]:
    timestamps = pd.date_range("2030-01-25", "2030-03-05 23:00", freq="h", tz="Europe/Zurich")
    delta = np.sin(np.arange(len(timestamps), dtype=float) / 24.0) * 0.75 + delta_shift
    aligned = tmp_path / f"{label}_aligned.csv"
    pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%d.%m.%Y %H:%M"),
            "delta_eur_mwh": delta,
            "month": timestamps.month,
            "hour": timestamps.hour,
        }
    ).to_csv(aligned, index=False)
    summary = tmp_path / f"{label}_summary.json"
    _write_json(
        summary,
        {
            "benchmark_policy": "independent_no_ompex",
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
        },
    )
    lab = tmp_path / f"{label}_lab.json"
    _write_json(
        lab,
        {
            "activation_status": "lab_only",
            "production_approved": False,
            "config": {
                "variant": "epex_weekend_tail_peak_v1",
                "weekend_intensity": weekend_intensity,
                "low_tail_intensity": 0.25,
                "peak_subshape_intensity": 0.75,
                "max_abs_delta_eur_mwh": 3.0,
                "negative_price_floor": -10.0,
                "max_weighted_negative_hours": 0,
                "lookback_years": 5,
                "require_monthly_base_constraints": True,
            },
        },
    )
    return label, aligned, summary, lab


def test_delta_stability_passes_stable_no_ompex_cases(tmp_path: Path) -> None:
    decision = summarize_delta_stability(
        cases=[
            _write_case(tmp_path, "asof1"),
            _write_case(tmp_path, "asof2", delta_shift=0.001),
        ],
        output_dir=tmp_path / "out",
    )

    assert decision["status"] == "PASS"
    assert decision["promotion_gate"] is False
    assert decision["comparison_count"] == 1
    rows = pd.read_csv(tmp_path / "out" / "delta_stability_comparisons.csv")
    assert rows.loc[0, "timestamp_delta_diff_mae_eur_mwh"] < 0.01
    assert rows.loc[0, "config_match"] is True or bool(rows.loc[0, "config_match"])


def test_delta_stability_fails_large_timestamp_delta_diff(tmp_path: Path) -> None:
    decision = summarize_delta_stability(
        cases=[
            _write_case(tmp_path, "asof1"),
            _write_case(tmp_path, "asof2", delta_shift=0.5),
        ],
        output_dir=tmp_path / "out",
        max_timestamp_delta_abs=0.25,
    )

    assert decision["status"] == "FAIL"
    rows = pd.read_csv(tmp_path / "out" / "delta_stability_comparisons.csv")
    assert rows.loc[0, "case_pass"] is False or not bool(rows.loc[0, "case_pass"])


def test_delta_stability_fails_config_mismatch(tmp_path: Path) -> None:
    decision = summarize_delta_stability(
        cases=[
            _write_case(tmp_path, "asof1"),
            _write_case(tmp_path, "asof2", weekend_intensity=0.75),
        ],
        output_dir=tmp_path / "out",
    )

    assert decision["status"] == "FAIL"
    assert decision["config_consistent"] is False
