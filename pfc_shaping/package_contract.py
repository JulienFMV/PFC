"""Positive inventory and immutable identity template for the governed LT wheel."""

BUILD_IDENTITY_TEMPLATE = (
    '"""Build-generated identity of the installed governed LT runtime."""\n\n'
    'SOURCE_REVISION = "{source_revision}"\n'
)
BUILD_IDENTITY_PLACEHOLDER = "0" * 64
REPRODUCIBLE_BUILD_EPOCH = 1783987200

REQUIRED_RUNTIME_DISTRIBUTIONS = {
    "cryptography": "47.0.0",
    "holidays": "0.83",
    "numpy": "2.0.2",
    "openpyxl": "3.1.5",
    "pandas": "2.3.3",
    "pyarrow": "21.0.0",
    "pyyaml": "6.0.3",
    "scikit-learn": "1.6.1",
    "scipy": "1.13.1",
}

SEALED_RUNTIME_DISTRIBUTIONS = {
    "cffi": "2.1.0",
    "cryptography": "47.0.0",
    "et-xmlfile": "2.0.0",
    "fmv-pfc-lt": "0.14.0",
    "holidays": "0.83",
    "joblib": "1.5.3",
    "numpy": "2.0.2",
    "openpyxl": "3.1.5",
    "pandas": "2.3.3",
    "pyarrow": "21.0.0",
    "pycparser": "3.0",
    "python-dateutil": "2.9.0.post0",
    "pytz": "2026.2",
    "pyyaml": "6.0.3",
    "scikit-learn": "1.6.1",
    "scipy": "1.13.1",
    "six": "1.17.0",
    "threadpoolctl": "3.6.0",
    "tzdata": "2026.3",
}

ALLOWED_DIST_INFO_FILES = frozenset(
    {
        "METADATA",
        "RECORD",
        "WHEEL",
        "top_level.txt",
    }
)

ALLOWED_RUNTIME_PYTHON_FILES = frozenset(
    """
pfc_shaping/__init__.py
pfc_shaping/build_identity.py
pfc_shaping/durable_artifact.py
pfc_shaping/exit_codes.py
pfc_shaping/package_contract.py
pfc_shaping/parquet_safety.py
pfc_shaping/path_safety.py
pfc_shaping/version.py
pfc_shaping/calibration/__init__.py
pfc_shaping/calibration/arbitrage_free.py
pfc_shaping/calibration/cascading.py
pfc_shaping/calibration/constraints.py
pfc_shaping/calibration/eex_contract_selection.py
pfc_shaping/calibration/monthly_curve_audit.py
pfc_shaping/calibration/monthly_curve_capstone.py
pfc_shaping/calibration/monthly_curve_config_identity.py
pfc_shaping/calibration/monthly_curve_priors.py
pfc_shaping/calibration/monthly_curve_promotion.py
pfc_shaping/calibration/monthly_forward_curve.py
pfc_shaping/cli/__init__.py
pfc_shaping/cli/audit_ch_lt_compute_runtime.py
pfc_shaping/cli/audit_ch_lt_compute_runtime_manifest.py
pfc_shaping/cli/audit_ch_lt_dependence_power_design.py
pfc_shaping/cli/audit_ch_lt_estimand_contract.py
pfc_shaping/cli/audit_ch_lt_origin_registry_protocol.py
pfc_shaping/cli/audit_ch_lt_origin_target_mask_inventory.py
pfc_shaping/cli/audit_ch_lt_successor_candidate_core.py
pfc_shaping/cli/audit_ch_market_time_regime.py
pfc_shaping/cli/build_ch_lt_prospective_capture_ledger.py
pfc_shaping/cli/eex_forward_vintage_builder.py
pfc_shaping/cli/governed_acquisition_builder.py
pfc_shaping/cli/governed_release.py
pfc_shaping/cli/score_ch_lt_structural_prediction_commitment.py
pfc_shaping/cli/verify_ch_lt_preregistration_supersession.py
pfc_shaping/data/__init__.py
pfc_shaping/data/acquisition_contract.py
pfc_shaping/data/calendar_ch.py
pfc_shaping/data/eex_datasource_v2_capture.py
pfc_shaping/data/eex_forward_vintage_intake.py
pfc_shaping/data/eex_historical_vintage.py
pfc_shaping/data/electrification_scenarios.py
pfc_shaping/data/forward_proxy.py
pfc_shaping/data/governed_lt_acquisition.py
pfc_shaping/data/ingest_forwards.py
pfc_shaping/data/lt_input_replay.py
pfc_shaping/data/lt_replay_transforms.py
pfc_shaping/data/lt_input_sources.py
pfc_shaping/data/shared_data_root.py
pfc_shaping/data/snapshot_anchor_client.py
pfc_shaping/data/snapshot_publication_contract.py
pfc_shaping/data/snapshot_publication_state.py
pfc_shaping/lt/__init__.py
pfc_shaping/lt/model/__init__.py
pfc_shaping/lt/model/assembler.py
pfc_shaping/lt/model/electrification_shape.py
pfc_shaping/lt/model/intraday_amplitude.py
pfc_shaping/lt/model/msfc_spline.py
pfc_shaping/lt/model/quant_shape_optimizer.py
pfc_shaping/lt/model/shape_constraints.py
pfc_shaping/lt/model/shape_hourly.py
pfc_shaping/lt/model/shape_hourly_mlp.py
pfc_shaping/lt/model/shape_intraday.py
pfc_shaping/lt/model/solar_modulation.py
pfc_shaping/lt/model/uncertainty.py
pfc_shaping/lt/model/water_value.py
pfc_shaping/pipeline/__init__.py
pfc_shaping/pipeline/atomic_promotion.py
pfc_shaping/pipeline/candidate_build.py
pfc_shaping/pipeline/candidate_bundle.py
pfc_shaping/pipeline/candidate_evidence.py
pfc_shaping/pipeline/candidate_evidence_assembler.py
pfc_shaping/pipeline/candidate_finalize.py
pfc_shaping/pipeline/candidate_product_evidence.py
pfc_shaping/pipeline/governed_release.py
pfc_shaping/pipeline/governed_release_cli_contract.py
pfc_shaping/pipeline/model_governance_contract.py
pfc_shaping/pipeline/monthly_curve_authority.py
pfc_shaping/pipeline/process_identity.py
pfc_shaping/pipeline/production_phases.py
pfc_shaping/pipeline/probabilistic_output_governance.py
pfc_shaping/pipeline/promotion_contract.py
pfc_shaping/pipeline/quality_gate.py
pfc_shaping/pipeline/quote_conflict_policy_contract.py
pfc_shaping/pipeline/release_request_contract.py
pfc_shaping/pipeline/strict_structured_data.py
pfc_shaping/validation/__init__.py
pfc_shaping/validation/block_masks.py
pfc_shaping/validation/ch_lt_compute_runtime.py
pfc_shaping/validation/ch_lt_compute_runtime_manifest.py
pfc_shaping/validation/ch_lt_dependence_power_design.py
pfc_shaping/validation/ch_lt_estimand_contract.py
pfc_shaping/validation/ch_lt_local_future_origin_selection.py
pfc_shaping/validation/ch_lt_native_hourly_truth_bundle.py
pfc_shaping/validation/ch_lt_origin_registry_protocol.py
pfc_shaping/validation/ch_lt_origin_target_mask_inventory.py
pfc_shaping/validation/ch_lt_preregistration_supersession.py
pfc_shaping/validation/ch_lt_prospective_capture_ledger.py
pfc_shaping/validation/ch_lt_prospective_hourly_scoring.py
pfc_shaping/validation/ch_lt_structural_prediction_commitment.py
pfc_shaping/validation/ch_lt_successor_candidate_core.py
pfc_shaping/validation/ch_lt_successor_candidate_core_v2.py
pfc_shaping/validation/ch_lt_successor_candidate_core_v3.py
pfc_shaping/validation/ch_lt_successor_readiness.py
pfc_shaping/validation/ch_market_time_regime.py
pfc_shaping/validation/product_normalization.py
""".split()
)
