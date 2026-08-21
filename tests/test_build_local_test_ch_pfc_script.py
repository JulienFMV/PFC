from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.build_local_test_ch_pfc import (
    DEFAULT_LOCAL_FINAL_PRODUCT_REPLAY_TOLERANCE,
    DEFAULT_LOCAL_MONTHLY_CONSTRAINT_TOLERANCE,
    _admit_intraday_provenance,
    _assert_fresh_output_set,
    _build_fan,
    _completion_manifest,
    _delivery_index_to_local_end_exclusive,
    _delivery_months_from_index,
    _expand_inventory,
    _final_product_replay,
    _input_path,
    _intraday_fit_diagnostics,
    _output_path,
    _peak_prices_for_delivery,
    _require_exact_monthly_constraint_tolerance,
    _require_expected_sha256,
    _require_intraday_provenance_arguments,
    _safe_output_prefix,
    _write_json,
)


def test_exact_local_delivery_window_contains_complete_months_and_cal2030() -> None:
    index = _delivery_index_to_local_end_exclusive(
        start_date="2026-07-31T22:00:00Z",
        local_end_exclusive="2031-01-01T00:00:00",
        market="CH",
    )

    local = index.tz_convert("Europe/Zurich")
    cal2030 = index[local.year == 2030]
    assert len(index) == 154_948
    assert local[0] == pd.Timestamp("2026-08-01T00:00:00", tz="Europe/Zurich")
    assert local[-1] == pd.Timestamp("2030-12-31T23:45:00", tz="Europe/Zurich")
    assert len(cal2030) == 35_040
    assert list(_delivery_months_from_index(index, market="CH")[[0, -1]].astype(str)) == [
        "2026-08",
        "2030-12",
    ]


def test_final_product_replay_requires_complete_cal2030_and_exact_price() -> None:
    index = pd.date_range(
        pd.Timestamp("2030-01-01T00:00:00", tz="Europe/Zurich").tz_convert("UTC"),
        pd.Timestamp("2031-01-01T00:00:00", tz="Europe/Zurich").tz_convert("UTC"),
        freq="15min",
        inclusive="left",
    )
    frame = pd.DataFrame({"price_shape": 69.470000000005}, index=index)
    snapshot = pd.DataFrame(
        [{"product": "2030", "load_type": "BASE", "price": 69.47}]
    )

    replay = _final_product_replay(
        {"central": frame},
        forward_snapshot=snapshot,
        forward_date=pd.Timestamp("2026-06-12"),
        market="CH",
    )

    scenario = replay["scenarios"]["central"]
    assert DEFAULT_LOCAL_FINAL_PRODUCT_REPLAY_TOLERANCE == 1e-9
    assert scenario["cal2030_base_rows"] == 8_760
    assert 1e-12 < scenario["maximum_supported_abs_residual_eur_mwh"] <= 1e-9
    base_gate = next(
        gate
        for gate in scenario["gates"]
        if gate["gate_id"] == "hard_base_product_repricing"
        and gate["product"] == "2030"
    )
    assert base_gate["abs_residual_eur_mwh"] > 1e-12
    with pytest.raises(ValueError, match="Cal2030|in-scope BASE"):
        _final_product_replay(
            {"central": frame.iloc[:-4]},
            forward_snapshot=snapshot,
            forward_date=pd.Timestamp("2026-06-12"),
            market="CH",
        )


def test_peak_prices_for_delivery_excludes_partial_front_product() -> None:
    snapshot = pd.DataFrame(
        [
            {"product": "2026", "load_type": "PEAK", "price": 100.0},
            {"product": "2026-08", "load_type": "PEAK", "price": 101.0},
            {"product": "2030", "load_type": "PEAK", "price": 90.0},
        ]
    )

    selected = _peak_prices_for_delivery(
        snapshot,
        delivery_months=pd.period_range("2026-08", "2030-12", freq="M"),
    )

    assert selected == {"2026-08-Peak": 101.0, "2030-Peak": 90.0}


def test_final_product_replay_accepts_exact_base_peak_and_implied_offpeak() -> None:
    index = pd.date_range(
        pd.Timestamp("2030-01-01T00:00:00", tz="Europe/Zurich").tz_convert("UTC"),
        pd.Timestamp("2031-01-01T00:00:00", tz="Europe/Zurich").tz_convert("UTC"),
        freq="15min",
        inclusive="left",
    )
    local = index.tz_convert("Europe/Zurich")
    peak = (local.weekday < 5) & (local.hour >= 8) & (local.hour < 20)
    peak_hours = int(peak.sum())
    offpeak_hours = len(index) - peak_hours
    offpeak_price = (70.0 * len(index) - 90.0 * peak_hours) / offpeak_hours
    frame = pd.DataFrame(
        {"price_shape": np.where(peak, 90.0, offpeak_price)},
        index=index,
    )
    snapshot = pd.DataFrame(
        [
            {"product": "2030", "load_type": "BASE", "price": 70.0},
            {"product": "2030", "load_type": "PEAK", "price": 90.0},
        ]
    )
    replay = _final_product_replay(
        {"central": frame},
        forward_snapshot=snapshot,
        forward_date=pd.Timestamp("2026-06-12"),
        market="CH",
    )

    assert replay["scenarios"]["central"]["maximum_supported_abs_residual_eur_mwh"] <= 1e-9


def test_local_runner_default_monthly_repricing_tolerance_is_exact() -> None:
    assert DEFAULT_LOCAL_MONTHLY_CONSTRAINT_TOLERANCE == 1e-9
    assert _require_exact_monthly_constraint_tolerance(1e-9) == 1e-9
    with pytest.raises(ValueError, match="exactly 1e-9"):
        _require_exact_monthly_constraint_tolerance(0.01)


def test_local_runner_confines_inputs_to_repo_and_outputs_to_build(tmp_path) -> None:
    root = tmp_path / "repo"
    build = root / "build"
    data = root / "data"
    outside = tmp_path / "outside"
    for directory in (build, data, outside):
        directory.mkdir(parents=True)

    assert _input_path(data / "input.parquet", repo_root=root) == data / "input.parquet"
    assert _output_path(build / "run" / "curve.parquet", repo_root=root) == (
        build / "run" / "curve.parquet"
    )
    with pytest.raises(ValueError, match="input escapes the repository"):
        _input_path(outside / "input.parquet", repo_root=root)
    with pytest.raises(ValueError, match="below canonical repo build"):
        _output_path(root / "output" / "curve.parquet", repo_root=root)


def _inventory():
    rows = []
    for scenario, weight, pv in [("slow", 0.25, 7.0), ("central", 0.5, 9.0), ("fast", 0.25, 12.0)]:
        rows.append(
            {
                "publication_date": pd.Timestamp("2026-01-01", tz="UTC"),
                "country": "CH",
                "scenario": scenario,
                "delivery_year": 2030,
                "delivery_month": None,
                "scenario_weight": weight,
                "quality_flag": "official_partial_proxy_neutralized_explicit",
                "source": "unit",
                "demand_twh": 70.0,
                "peak_load_gw": 14.0,
                "winter_demand_twh": 25.0,
                "pv_gw": pv,
                "pv_twh": pv,
                "wind_gw": 0.2,
                "wind_twh": 0.3,
                "battery_power_gw": 2.0,
                "battery_energy_gwh": 6.0,
                "ev_twh": 0.2,
                "heatpump_twh": 4.0,
                "managed_charging_share": 0.3,
                "hydro_twh": 40.0,
                "hydro_capacity_gw": 17.0,
                "hydro_reservoir_twh": 4.0,
                "nuclear_gw": 3.0,
                "dispatchable_gw": 3.0,
                "gas_gw": 0.1,
                "coal_gw": 0.0,
                "net_import_twh": 8.0,
                "ntc_ch_de_gw": 1.0,
                "ntc_ch_fr_gw": 1.0,
                "ntc_ch_it_gw": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_expand_inventory_clamps_single_year_for_local_test(tmp_path):
    output = tmp_path / "expanded.parquet"

    expanded = _expand_inventory(
        _inventory(),
        years=[2030, 2031],
        scenarios=["slow", "central", "fast"],
        output=output,
    )

    assert output.exists()
    assert set(expanded["delivery_year"]) == {2030, 2031}
    assert len(expanded) == 6
    assert expanded.loc[expanded["delivery_year"] == 2031, "quality_flag"].str.contains("interpolated").all()


def test_build_fan_has_ordered_scenario_bracket():
    index = pd.date_range("2030-01-01", periods=2, freq="15min", tz="UTC")
    curves = {
        "slow": pd.DataFrame({"price_shape": [40.0, 42.0]}, index=index),
        "central": pd.DataFrame({"price_shape": [45.0, 47.0]}, index=index),
        "fast": pd.DataFrame({"price_shape": [50.0, 52.0]}, index=index),
    }

    fan = _build_fan(curves, {"slow": 0.25, "central": 0.5, "fast": 0.25})

    assert (fan["structural_scenario_low"] <= fan["structural_scenario_central"]).all()
    assert (fan["structural_scenario_central"] <= fan["structural_scenario_high"]).all()
    assert (fan["structural_scenario_spread"] > 0).all()


def test_intraday_diagnostics_separate_direct_fit_from_fallback():
    index = pd.date_range("2025-10-01", periods=8, freq="15min", tz="UTC")
    model = SimpleNamespace(
        base_factors_={
            ("Automne", "Ouvrable", 0): np.array([0.9, 1.0, 1.0, 1.1]),
            ("Hiver", "Ouvrable", 0): np.array([0.9, 1.0, 1.0, 1.1]),
        },
        n_obs_={("Automne", "Ouvrable", 0): 40},
    )

    diagnostics = _intraday_fit_diagnostics(
        model,
        pd.DataFrame({"price_eur_mwh": range(8)}, index=index),
        source="intraday.parquet",
        market="DE",
    )

    assert diagnostics["surface_cells_after_fallback"] == 2
    assert diagnostics["directly_learned_cells"] == 1
    assert diagnostics["fallback_cells"] == 1
    assert diagnostics["fallback_fraction"] == 0.5
    assert diagnostics["direct_nonflat_cells"] == 1
    assert diagnostics["direct_cells_by_season"] == {
        "Hiver": 0,
        "Printemps": 0,
        "Ete": 0,
        "Automne": 1,
    }


def test_local_output_set_rejects_existing_artifact(tmp_path):
    existing = tmp_path / "already.parquet"
    existing.write_bytes(b"old")

    with pytest.raises(FileExistsError, match="refusing to mix or overwrite"):
        _assert_fresh_output_set({"curve": existing, "summary": tmp_path / "new.md"})


def test_completion_manifest_binds_outputs_and_remains_no_go(tmp_path):
    output = tmp_path / "curve.parquet"
    output.write_bytes(b"curve")
    inputs = {
        "forwards": {
            "path": "data/forwards.parquet",
            "sha256": "a" * 64,
            "size_bytes": 5,
        }
    }

    manifest = _completion_manifest(
        args=argparse.Namespace(market="CH", horizon_days=1),
        repo_root=tmp_path,
        inputs=inputs,
        outputs={"curve": output},
        monthly_manifest={
            "monthly_level_authority": "solver",
            "forward_source_kind": "TEST_FIXTURE",
            "hard_quote_eligible": False,
            "promotion_eligible": False,
            "forward_source_sha256": None,
        },
    )

    assert manifest["status"] == "COMPLETE_LOCAL_TEST_NO_GO"
    assert manifest["production_authorization"] is False
    assert manifest["promotion_gate"] is False
    assert manifest["scientific_admission"] is False
    assert manifest["confirmatory_rolling_origin_eligible"] is False
    assert manifest["local_diagnostic_allowed"] is True
    assert manifest["monthly_level_authority"] == "solver"
    assert manifest["code_execution_attestation"] == (
        "NOT_PROVIDED_SOURCE_HASHES_ARE_OBSERVED_AFTER_IMPORT_ONLY"
    )
    assert manifest["outputs"]["curve"]["sha256"] == hashlib.sha256(b"curve").hexdigest()


def test_completion_manifest_is_exclusive_and_last_written(tmp_path):
    path = tmp_path / "completion.json"
    payload = {"status": "COMPLETE_LOCAL_TEST_NO_GO", "promotion_gate": False}

    _write_json(path, payload, label="test completion")

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    with pytest.raises(ValueError, match="already exists"):
        _write_json(path, {**payload, "promotion_gate": True}, label="test completion")


def test_caller_held_intraday_provenance_hash_is_required_exactly():
    record = {"sha256": "a" * 64}

    _require_expected_sha256(record, "A" * 64, label="intraday provenance")

    with pytest.raises(ValueError, match="caller-held"):
        _require_expected_sha256(record, "b" * 64, label="intraday provenance")
    with pytest.raises(ValueError, match="invalid"):
        _require_expected_sha256(record, "not-a-hash", label="intraday provenance")


def test_explicit_intraday_panel_requires_manifest_and_caller_held_hash():
    with pytest.raises(ValueError, match="explicit intraday panel requires"):
        _require_intraday_provenance_arguments(
            argparse.Namespace(
                intraday_epex="panel.parquet",
                intraday_provenance_manifest=None,
                expected_intraday_provenance_sha256=None,
            )
        )

    _require_intraday_provenance_arguments(
        argparse.Namespace(
            intraday_epex="panel.parquet",
            intraday_provenance_manifest="panel-manifest.json",
            expected_intraday_provenance_sha256="a" * 64,
        )
    )


def test_output_prefix_rejects_absolute_or_segmented_paths():
    assert _safe_output_prefix("local_pfc_candidate-v1") == "local_pfc_candidate-v1"
    with pytest.raises(ValueError, match="unsafe path"):
        _safe_output_prefix(r"C:\outside")
    with pytest.raises(ValueError, match="unsafe path"):
        _safe_output_prefix("nested/output")


def test_intraday_provenance_binds_exact_consumed_panel(tmp_path):
    panel_path = tmp_path / "panel.parquet"
    panel_path.write_bytes(b"captured-panel")
    panel_sha256 = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    payload = json.dumps(
        {
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
            "market": "DE-LU_PROXY_FOR_CH_LOCAL_DIAGNOSTIC",
            "start_utc": "2026-01-01T00:00:00Z",
            "end_utc": "2026-01-01T00:15:00Z",
            "rows": 1,
            "frequency_seconds": 900,
            "resolution_provenance": {
                "status": "MIXED_AUTHORITY_OUTPUT_GRID_NOT_NATIVE_TRUTH",
                "native_quarter_hour_truth_eligible": False,
                "reason": "fixture remains local and mixed-authority",
            },
            "season_row_counts": {"Hiver": 1},
            "sources": [],
            "panel": {
                "path": str(panel_path),
                "sha256": panel_sha256,
                "size_bytes": panel_path.stat().st_size,
            },
            "limitations": ["fixture"],
        }
    ).encode("utf-8")

    admitted = _admit_intraday_provenance(
        manifest_payload=payload,
        intraday_path=panel_path,
        intraday_sha256=panel_sha256,
    )

    assert admitted["panel"]["sha256"] == panel_sha256
    with pytest.raises(ValueError, match="consumed panel hash"):
        _admit_intraday_provenance(
            manifest_payload=payload,
            intraday_path=panel_path,
            intraday_sha256="0" * 64,
        )

    overclaimed = json.loads(payload)
    overclaimed["scientific_admission"] = True
    overclaimed["status"] = "SCIENTIFICALLY_ADMITTED"
    with pytest.raises(ValueError, match="must remain non-promotional"):
        _admit_intraday_provenance(
            manifest_payload=json.dumps(overclaimed).encode("utf-8"),
            intraday_path=panel_path,
            intraday_sha256=panel_sha256,
        )


def test_exact_pinned_legacy_intraday_manifest_is_local_diagnostic_only() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = (
        root
        / "output"
        / "phase14"
        / "prospective_intraday_panel_20260724"
        / "panel-manifest.json"
    )
    panel_path = manifest_path.parent / "epex_de_15min_complete.parquet"
    payload = manifest_path.read_bytes()
    panel_sha256 = hashlib.sha256(panel_path.read_bytes()).hexdigest()

    admitted = _admit_intraday_provenance(
        manifest_payload=payload,
        intraday_path=panel_path,
        intraday_sha256=panel_sha256,
    )

    assert admitted["schema_version"] == "local_intraday_calibration_panel.v1"
    assert admitted["production_authorization"] is False
    assert admitted["promotion_gate"] is False
    with pytest.raises(ValueError, match="exact pinned local diagnostic"):
        _admit_intraday_provenance(
            manifest_payload=payload + b"\n",
            intraday_path=panel_path,
            intraday_sha256=panel_sha256,
        )
