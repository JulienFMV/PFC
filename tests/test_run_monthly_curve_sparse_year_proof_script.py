from __future__ import annotations

import argparse

import pandas as pd

from pfc_shaping.calibration.monthly_curve_lambda_calibration import config_hash
from pfc_shaping.calibration.monthly_forward_curve import MonthlyCurveConfig
from pfc_shaping.calibration.monthly_forward_curve import MarketQuote
from scripts.run_monthly_curve_sparse_year_proof import _active_config_payload, _build_sparse_proof_governance_gates


def test_sparse_year_proof_emits_required_governance_gates_when_hashes_are_supplied() -> None:
    args = argparse.Namespace(
        require_lambda_artifact=True,
        require_path_parity=True,
        production_monthly_solution_hash="solution",
        export_monthly_solution_hash="solution",
        production_active_constraints_hash="constraints",
        export_active_constraints_hash="constraints",
    )

    gates = _build_sparse_proof_governance_gates(
        args=args,
        run_timestamp=pd.Timestamp("2026-06-17"),
        own_quotes=(_quote("2028"),),
        neighbor_quotes=(),
        eex_history=_history(),
        active_config_hash="config",
        selected_config_hash="config",
    )

    required = gates[
        gates["gate_id"].isin(
            {
                "lambda_calibration_artifact_present",
                "production_export_path_parity",
            }
        )
    ]
    assert set(required["gate_id"]) == {
        "lambda_calibration_artifact_present",
        "production_export_path_parity",
    }
    assert set(required["status"]) == {"PASS"}


def test_sparse_year_proof_does_not_emit_optional_governance_gates_by_default() -> None:
    args = argparse.Namespace(
        require_lambda_artifact=False,
        require_path_parity=False,
        production_monthly_solution_hash="",
        export_monthly_solution_hash="",
        production_active_constraints_hash="",
        export_active_constraints_hash="",
    )

    gates = _build_sparse_proof_governance_gates(
        args=args,
        run_timestamp=pd.Timestamp("2026-06-17"),
        own_quotes=(_quote("2028"),),
        neighbor_quotes=(),
        eex_history=_history(),
        active_config_hash=None,
        selected_config_hash=None,
    )

    assert set(gates["gate_id"]) == {"point_in_time_data_contract"}


def test_sparse_year_proof_active_config_payload_includes_structural_cli_knobs() -> None:
    args = argparse.Namespace(
        history_lookback_years=6,
        min_structural_snapshots=24,
        allow_template_structural_fallback=True,
        structural_amplitude_eur_mwh=110.0,
        panel_weight=1.0,
        history_weight=0.5,
        structural_weight=1.0,
    )
    config = MonthlyCurveConfig(
        lambda_prior=1e-6,
        lambda_smooth_month=1.0,
        lambda_smooth_yoy=0.25,
        lambda_shape=1.0,
        neighbor_shrinkage=0.5,
        robust_panel_quantile=0.5,
        min_history_snapshots=24,
        constraint_tolerance=1e-9,
    )
    payload = _active_config_payload(args, config, ("DE", "FR", "AT", "IT"))

    assert config_hash(payload) != config_hash(payload | {"structural_weight": 0.5})
    assert payload["allow_template_structural_fallback"] is True
    assert payload["structural_amplitude_eur_mwh"] == 110.0
    assert payload["markets"] == ["DE", "FR", "AT", "IT"]


def _quote(product: str) -> MarketQuote:
    return MarketQuote(
        market="CH",
        product=product,
        load_type="BASE",
        price=80.0,
        snapshot_date=pd.Timestamp("2026-06-17"),
        available_at=pd.Timestamp("2026-06-17"),
    )


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-06-17"),
                "market": "CH",
                "load_type": "BASE",
                "product": "2028",
                "price": 80.0,
            }
        ]
    )
