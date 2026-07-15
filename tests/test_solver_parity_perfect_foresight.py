from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pfc_shaping.pipeline.monthly_curve_authority import monthly_solver_settings
from pfc_shaping.pipeline.strict_structured_data import load_strict_yaml
from pfc_shaping.validation.solver_parity_perfect_foresight import (
    _history_as_of_vintage,
    _validate_solver_settings,
    run_solver_parity_perfect_foresight,
)


def test_history_as_of_vintage_excludes_post_vintage_rows() -> None:
    history = pd.DataFrame(
        {
            "date": ["2024-12-30", "2024-12-31", "2025-01-02"],
            "product": ["2025", "2025", "2025"],
            "load_type": ["BASE", "BASE", "BASE"],
            "price": [90.0, 91.0, 200.0],
            "market": ["CH", "CH", "CH"],
        }
    )

    filtered, cutoff = _history_as_of_vintage(
        history,
        pd.Timestamp("2024-12-31T17:00:00Z"),
    )

    assert cutoff == pd.Timestamp("2024-12-31")
    assert filtered["price"].tolist() == [90.0, 91.0]
    assert filtered["date"].max() == pd.Timestamp("2024-12-31")


def test_solver_parity_rejects_legacy_level_authority() -> None:
    with pytest.raises(ValueError, match="production solver settings"):
        _validate_solver_settings(
            {
                "enabled": True,
                "monthly_level_authority": "legacy",
                "skip_legacy_level_cascade": True,
                "skip_legacy_base_smoothing": True,
                "constraint_tolerance": 1e-9,
            }
        )


@pytest.mark.slow
def test_cal_2025_solver_parity_golden_and_future_row_invariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pfc_shaping.lt.model.assembler import PFCAssembler

    root = Path(__file__).resolve().parents[1]
    epex = pd.read_parquet(root / "data" / "epex_hourly.parquet")["price_eur_mwh"]
    history = pd.read_parquet(root / "data" / "eex_forwards_history.parquet")
    config = load_strict_yaml(
        (root / "pfc_shaping" / "config.yaml").read_text(encoding="utf-8")
    )
    settings = monthly_solver_settings(config)
    vintage = pd.Timestamp("2024-12-31T17:00:00Z")
    counts = {"legacy_calibration": 0, "final_projection": 0}
    original_calibration = PFCAssembler._apply_calibration
    original_projection = PFCAssembler._project_final_solver_products

    def count_calibration(self, *args, **kwargs):
        counts["legacy_calibration"] += 1
        return original_calibration(self, *args, **kwargs)

    def count_projection(self, *args, **kwargs):
        counts["final_projection"] += 1
        return original_projection(self, *args, **kwargs)

    monkeypatch.setattr(PFCAssembler, "_apply_calibration", count_calibration)
    monkeypatch.setattr(
        PFCAssembler,
        "_project_final_solver_products",
        count_projection,
    )

    full = run_solver_parity_perfect_foresight(
        epex,
        history,
        target_year=2025,
        vintage=vintage,
        solver_settings=settings,
        config_name="bowl_on_floors_off",
        estimator="baseline",
    )
    history_asof, _ = _history_as_of_vintage(history, vintage)
    asof_only = run_solver_parity_perfect_foresight(
        epex,
        history_asof,
        target_year=2025,
        vintage=vintage,
        solver_settings=settings,
        config_name="bowl_on_floors_off",
        estimator="baseline",
    )
    sota = run_solver_parity_perfect_foresight(
        epex,
        history_asof,
        target_year=2025,
        vintage=vintage,
        solver_settings=settings,
        config_name="bowl_on_floors_off",
        estimator="sota",
    )

    assert counts == {"legacy_calibration": 0, "final_projection": 3}
    assert full.metadata["production_representative"] is False
    assert full.metadata["promotion_eligible"] is False
    assert full.metadata["pit_evidence_status"] == "UNAVAILABLE_ROW_LEVEL_AVAILABLE_AT"
    assert full.metadata["post_vintage_rows_excluded"] == len(history) - len(history_asof)
    assert full.metadata["pfc_price_shape_hash"] == asof_only.metadata["pfc_price_shape_hash"]
    assert full.metadata["monthly_solution_hash"] == asof_only.metadata["monthly_solution_hash"]
    assert full.metadata["diagnostic_class"] == "RESEARCH_ONLY_NON_PIT_CONSERVATION_CHECK"
    assert full.metadata["monthly_metrics_are_model_comparison_evidence"] is False
    assert full.metadata["solver_numerical_status"] == "PASS"
    assert sota.metadata["monthly_solution_hash"] == full.metadata["monthly_solution_hash"]
    assert sota.metrics["monthly_pearson_corr"] == pytest.approx(
        full.metrics["monthly_pearson_corr"],
        abs=1e-12,
    )
    assert sota.metadata["pfc_price_shape_hash"] != full.metadata["pfc_price_shape_hash"]
    assert float(
        (sota.pfc["price_shape"] - full.pfc["price_shape"]).abs().max()
    ) > 1.0
    assert full.metrics["monthly_pearson_corr"] == pytest.approx(
        0.940896826086905,
        abs=1e-10,
    )
    assert full.metrics["monthly_mae_eur_mwh"] == pytest.approx(
        9.421012002860222,
        abs=1e-9,
    )
    assert full.metrics["projection_max_abs_error_eur_mwh"] <= 1e-6
    assert full.final_projection_report["base_constraint_count"] > 0
    assert full.final_projection_report["peak_constraint_count"] > 0
    assert full.final_projection_report["offpeak_constraint_count"] == 0
    assert abs(full.metrics["cal_base_residual_eur_mwh"]) <= 1e-9
    assert abs(full.metrics["cal_peak_residual_eur_mwh"]) <= 1e-9
    assert abs(full.metrics["cal_implied_offpeak_residual_eur_mwh"]) <= 1e-9
