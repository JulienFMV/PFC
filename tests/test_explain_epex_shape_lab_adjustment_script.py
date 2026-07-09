from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.explain_epex_shape_lab_adjustment import explain_adjustment


def _write_candidate(path: Path, timestamps: pd.DatetimeIndex, delta: np.ndarray | float = 0.0) -> None:
    values = np.full(len(timestamps), 80.0) + np.asarray(delta, dtype=float)
    frame = pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%d.%m.%Y %H:%M"),
            "utc_offset_ch": timestamps.strftime("UTC%z"),
            "price_weighted_mean_eur_mwh": values,
        }
    )
    frame.to_csv(path, index=False)


def _write_templates(path: Path) -> None:
    rows = []
    for month in range(1, 13):
        for hour in range(24):
            for is_weekend in (False, True):
                rows.append(
                    {
                        "month": month,
                        "hour": hour,
                        "is_weekend": is_weekend,
                        "weekend_delta_eur_mwh": 1.0 if is_weekend else 0.0,
                        "low_tail_delta_eur_mwh": -0.5 if 3 <= month <= 10 and 10 <= hour <= 16 else 0.0,
                        "peak_subshape_delta_eur_mwh": 0.0,
                        "evening_recovery_delta_eur_mwh": 0.0,
                        "night_delta_eur_mwh": -0.25 if hour <= 5 else 0.0,
                        "ramp_delta_eur_mwh": 0.0,
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_manifest(path: Path, templates: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "config": {
                    "weekend_intensity": 2.0,
                    "low_tail_intensity": 1.0,
                    "peak_subshape_intensity": 0.0,
                    "evening_recovery_intensity": 0.0,
                    "night_intensity": 1.0,
                    "ramp_intensity": 0.0,
                },
                "outputs": {"templates_csv": str(templates)},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_explain_epex_shape_lab_adjustment_reports_component_buckets(tmp_path: Path) -> None:
    timestamps = pd.date_range("2030-01-01", "2030-01-14 23:00", freq="h", tz="Europe/Zurich")
    baseline = tmp_path / "baseline.csv"
    adjusted = tmp_path / "adjusted.csv"
    templates = tmp_path / "templates.csv"
    manifest = tmp_path / "manifest.json"
    _write_candidate(baseline, timestamps)
    _write_candidate(adjusted, timestamps)
    _write_templates(templates)
    _write_manifest(manifest, templates)

    summary = explain_adjustment(
        baseline_csv=baseline,
        adjusted_csv=adjusted,
        lab_manifest=manifest,
        output_dir=tmp_path / "out",
    )

    assert summary["status"] == "DIAGNOSTIC_PASS"
    assert summary["promotion_gate"] is False
    assert summary["benchmark_policy"] == "shape_explainability_no_ompex_diagnostic_only"
    assert summary["ompex_used_in_model"] is False
    assert summary["component_totals"]["weekend"]["nonzero_hours"] > 0
    assert summary["bucket_delta_summary"]["weekend"]["hours"] > 0
    assert summary["bucket_delta_summary"]["weekend"]["dominant_raw_component"] == "weekend"
    assert summary["bucket_delta_summary"]["weekend"]["projection_residual_to_raw_abs_ratio"] is not None
    assert summary["compression_summary"]["projection_residual_to_raw_abs_ratio"] is not None
    assert summary["compression_summary"]["high_compression_bucket_count"] > 0
    assert summary["strict_checks"]["monthly_base_delta_conserved"] is True
    assert (tmp_path / "out" / "bucket_delta_summary.csv").exists()
    written = json.loads((tmp_path / "out" / "shape_explainability_summary.json").read_text(encoding="utf-8"))
    assert written["output_hashes"] == summary["output_hashes"]


def test_explain_epex_shape_lab_adjustment_fails_on_monthly_level_drift(tmp_path: Path) -> None:
    timestamps = pd.date_range("2030-01-01", "2030-01-07 23:00", freq="h", tz="Europe/Zurich")
    baseline = tmp_path / "baseline.csv"
    adjusted = tmp_path / "adjusted.csv"
    templates = tmp_path / "templates.csv"
    manifest = tmp_path / "manifest.json"
    _write_candidate(baseline, timestamps)
    _write_candidate(adjusted, timestamps, delta=1.0)
    _write_templates(templates)
    _write_manifest(manifest, templates)

    summary = explain_adjustment(
        baseline_csv=baseline,
        adjusted_csv=adjusted,
        lab_manifest=manifest,
        output_dir=tmp_path / "out",
    )

    assert summary["status"] == "DIAGNOSTIC_FAIL"
    assert summary["strict_checks"]["monthly_base_delta_conserved"] is False
