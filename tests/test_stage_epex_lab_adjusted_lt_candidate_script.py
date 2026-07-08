from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.stage_epex_lab_adjusted_lt_candidate import stage_candidate


def _fan_parquet(path) -> None:
    local = pd.date_range("2030-01-01", "2030-02-28 23:45", freq="15min", tz="Europe/Zurich")
    base = np.where(local.month == 1, 80.0, 90.0).astype(float)
    fan = pd.DataFrame(
        {
            "curve_slow": base - 2.0,
            "curve_central": base,
            "curve_fast": base + 2.0,
            "weighted_mean": base,
            "structural_p10": base - 8.0,
            "structural_p50": base,
            "structural_p90": base + 12.0,
            "structural_width": np.full(len(base), 20.0),
        },
        index=local.tz_convert("UTC"),
    )
    fan.to_parquet(path)


def _spot_parquet(path) -> None:
    timestamps = pd.date_range("2024-01-01", "2025-12-31 23:00", freq="h", tz="Europe/Zurich")
    hour = timestamps.hour.to_numpy(dtype=float)
    weekend_discount = np.where(timestamps.weekday >= 5, -5.0, 0.0)
    solar_tail = np.where((timestamps.month >= 3) & (timestamps.month <= 10) & (hour >= 10) & (hour <= 16), -4.0, 0.0)
    price = 70.0 + 4.0 * np.sin(hour / 24.0 * 2.0 * np.pi) + weekend_discount + solar_tail
    pd.DataFrame(
        {"price_eur_mwh": price},
        index=timestamps.tz_convert("UTC"),
    ).to_parquet(path)


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _strict_evidence(tmp_path):
    monthly = tmp_path / "monthly.json"
    product = tmp_path / "product.json"
    powerbi = tmp_path / "powerbi.csv"
    policy = tmp_path / "policy.json"
    independent = tmp_path / "independent.json"
    governance = tmp_path / "governance.json"
    _write_json(
        monthly,
        {
            "monthly_level_authority": "solver",
            "monthly_solution_hash": "m" * 64,
            "active_constraints_hash": "c" * 64,
            "active_config_hash": "a" * 64,
        },
    )
    _write_json(
        product,
        {
            "all_gates_pass": True,
            "critical_count": 0,
            "unsupported_count": 0,
            "blocking_quote_conflict_count": 0,
        },
    )
    pd.DataFrame(
        [
            {"metric": "powerbi_quality_gate_status", "value": "PASS"},
            {"metric": "weighted_negative_hours", "value": "0"},
            {"metric": "monthly_path_critical_flags", "value": "0"},
        ]
    ).to_csv(powerbi, index=False)
    _write_json(policy, {"production_approved": True})
    _write_json(
        independent,
        {
            "benchmark_policy": "independent_no_ompex",
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
        },
    )
    _write_json(governance, {"status": "PASS"})
    return monthly, product, powerbi, policy, independent, governance


def test_stage_epex_lab_adjusted_candidate_from_fan_is_no_go_and_hash_bound(tmp_path) -> None:
    fan = tmp_path / "fan.parquet"
    spot = tmp_path / "spot.parquet"
    _fan_parquet(fan)
    _spot_parquet(spot)

    manifest = stage_candidate(
        fan_parquet=fan,
        output_dir=tmp_path / "stage",
        spot_parquet=spot,
        valuation_timestamp="2026-01-01T00:00:00Z",
        local_start_date="2030-01-01",
        local_end_date="2030-02-28",
        weekend_intensity=0.5,
        low_tail_intensity=0.25,
        peak_subshape_intensity=0.75,
        max_abs_delta_eur_mwh=3.0,
    )

    assert manifest["schema_version"] == "epex_lab_adjusted_lt_candidate_stage.v1"
    assert manifest["activation_status"] == "staged_lab_only"
    assert manifest["production_approved"] is False
    assert manifest["production_promotion_approved"] is False
    assert manifest["ompex_used_in_model"] is False
    assert manifest["ompex_used_in_selection"] is False
    assert manifest["source_kind"] == "fan_parquet"
    assert manifest["source_sha256"]
    assert manifest["staged_candidate_csv_sha256"]
    assert manifest["adjusted_csv_sha256"]
    assert "baseline_monthly_manifest" in manifest["missing_production_contract_inputs"]

    stage_dir = tmp_path / "stage"
    assert (stage_dir / "lt_hourly_candidate.csv").exists()
    assert (stage_dir / "epex_lab" / "candidate_epex_shape_lab_adjusted.csv").exists()
    assert (stage_dir / "epex_lab" / "ab_lab_manifest.json").exists()
    saved = json.loads((stage_dir / "staged_lt_epex_lab_candidate_manifest.json").read_text(encoding="utf-8"))
    assert saved == manifest


def test_stage_epex_lab_adjusted_candidate_requires_single_source(tmp_path) -> None:
    spot = tmp_path / "spot.parquet"
    _spot_parquet(spot)

    with pytest.raises(ValueError, match="provide exactly one"):
        stage_candidate(
            output_dir=tmp_path / "stage",
            spot_parquet=spot,
            valuation_timestamp="2026-01-01T00:00:00Z",
        )


def test_stage_epex_lab_adjusted_candidate_writes_contract_when_evidence_is_supplied(tmp_path) -> None:
    fan = tmp_path / "fan.parquet"
    spot = tmp_path / "spot.parquet"
    _fan_parquet(fan)
    _spot_parquet(spot)
    monthly, product, powerbi, policy, independent, governance = _strict_evidence(tmp_path)

    manifest = stage_candidate(
        fan_parquet=fan,
        output_dir=tmp_path / "stage",
        spot_parquet=spot,
        valuation_timestamp="2026-01-01T00:00:00Z",
        local_start_date="2030-01-01",
        local_end_date="2030-02-28",
        baseline_monthly_manifest=monthly,
        product_summary=product,
        powerbi_summary=powerbi,
        source_hierarchy_policy=policy,
        independent_summary=independent,
        governance_audit=governance,
    )

    assert manifest["missing_production_contract_inputs"] == []
    assert manifest["adjusted_production_contract_pass"] is True
    contract_path = tmp_path / "stage" / "adjusted_production_manifest_no_go.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["schema_version"] == "epex_lab_adjusted_production_manifest.v1"
    assert contract["contract_pass"] is True
    assert contract["production_approved"] is False
    assert contract["production_promotion_approved"] is False
