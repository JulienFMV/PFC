"""Fail-closed structural readiness chain for a future CH LT preregistration.

The v1 contract makes D164 prerequisites machine-checkable.  The selected
outcome-blind candidate core supersedes its ``NOT_AUTHORED`` lifecycle and
legacy T057-access observations; the admitted successor remains uncreated and
every authority remains false.  The chain may read only the outcome-blind
T057 tombstone and never the superseded registry, outcomes, scores, or future
truth.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json
from pfc_shaping.validation.ch_lt_successor_candidate_core_v3 import (
    CORE_RELATIVE_PATH as CANDIDATE_CORE_RELATIVE_PATH,
)
from pfc_shaping.validation.ch_lt_successor_candidate_core_v3 import (
    CORE_SHA256 as CANDIDATE_CORE_SHA256,
)
from pfc_shaping.validation.ch_lt_successor_candidate_core_v3 import (
    verify_candidate_core_v3,
)

SCHEMA_VERSION = "ch_lt_pit_successor_readiness_contract.v1"
ASSESSMENT_SCHEMA_VERSION = "ch_lt_pit_successor_readiness_assessment.v1"
BASE_STATUS = "STRUCTURAL_READINESS_VALID_SUCCESSOR_NOT_CREATED"
STATUS = (
    "CANDIDATE_CORE_V3_CONTRAST_AWARE_LOCAL_HASH_CLOSED_NOT_EXTERNALLY_FROZEN_"
    "NOT_ADMITTED_SUCCESSOR_NOT_CREATED"
)
CONTRACT_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-PIT-SUCCESSOR-READINESS-CONTRACT-20260729.json"
)
CONTRACT_SHA256 = "734a7824ec747c829526774b346da441b45fff7cff5c9eb79ffc653ac78c7b8e"
CONTRACT_ID = "5e655cab1ca090cd067100dc8eb06161811b77991132e7c299883a7eb249e706"
READINESS_UPDATE_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-PIT-SUCCESSOR-READINESS-UPDATE-V3-20260730.json"
)
READINESS_UPDATE_SHA256 = (
    "07abfbe22f1211049d95333b7af5c987983230325d112f14c4299f234d6caadb"
)
READINESS_UPDATE_ID = (
    "eb69c29f1f3744c4d62d4ea2ecdb2532a8039148e1daba1dcf1f56a25c807f11"
)
READINESS_UPDATE_V2_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-PIT-SUCCESSOR-READINESS-UPDATE-V2-20260730.json"
)
READINESS_UPDATE_V2_SHA256 = (
    "93dd94ad7f2fc2734a81a73ab80fc5f373c9b9674e0895857f2ab8019c037497"
)
READINESS_UPDATE_V2_ID = (
    "8825f83e8a190198d909c85eda8142b868e937969eb29c07702c0b51493647f4"
)

PREREGISTRATION_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-PIT-PROBABILISTIC-PREREGISTRATION-DRAFT-20260724.json"
)
PREREGISTRATION_SHA256 = "aba798530084b7031a0ac38b1c48b20cff575d6082edbcf37c9a04528900ba61"
PREREGISTRATION_PLAN_ID = "ae5557fd7e58a6ee4164e7f8a949cb379fc2d8ac23766e17a1873c4de420c5f6"
ESTIMAND_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-ESTIMAND-AND-ECONOMIC-DESIGN-DRAFT-20260724.json"
)
ESTIMAND_SHA256 = "4209931e28a7c1cf2a4224d779f73648c4c9c5eac55df0a7ba1ad872226e2931"
ESTIMAND_CONTRACT_ID = "da4090073a4566f662e47fa59e206e1485d683305a1134f2089ebb13a4daa344"
COMPUTE_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/CH-LT-COMPUTE-RUNTIME-DRAFT-20260727.json"
)
COMPUTE_SHA256 = "b231345e96e7664ae02b7dbf3514af87d47ded7783034eaab1f8d449a28fe96f"
COMPUTE_CONTRACT_ID = "d06710ba8ebee2364b81930fce51d17768f206521edfe62336ff2abdef60930a"

REQUIRED_BEFORE_CREATION = (
    "GOVERNED_POINT_IN_TIME_CH_AND_EEX_EVIDENCE",
    "NATIVE_CH_LAYER_TRUTH_AND_POST_EPISODE_OUTCOMES",
    "EXACT_ORIGIN_TARGET_MASK_AND_INNER_FOLD_INVENTORIES",
    "FROZEN_DETERMINISTIC_CANDIDATE_BASELINE_COMPLEXITY_AND_MARKET_GATES",
    "FROZEN_DEPENDENCE_POWER_AND_MULTIPLICITY_DESIGN",
    "FROZEN_PROBABILISTIC_SCENARIO_AND_MONTE_CARLO_DESIGN",
    "FMV_RISK_APPROVED_ECONOMIC_MATERIALITY",
    "QUALIFIED_CPU_ORACLE_AND_CPU_GPU_RUNTIME_PARITY",
    "INDEPENDENT_SECURITY_ITOPS_AND_QUANT_REVIEWS",
    "EXTERNAL_ADMISSION_ENVELOPE_AND_ONE_SHOT_LEDGER",
)

_TOP_LEVEL_KEYS = {
    "schema_version",
    "contract_name",
    "lifecycle",
    "bound_structural_inputs",
    "successor_creation_gate",
    "holdout_non_consumption",
    "estimand_origin_and_truth_freeze",
    "deterministic_candidate_and_market_gates",
    "dependence_power_and_multiplicity",
    "probabilistic_scenario_and_monte_carlo",
    "economic_materiality",
    "compute_runtime_qualification",
    "external_admission_and_ledger",
    "independent_review",
    "scientific_admission",
    "execution_authorized",
    "production_authorization",
    "promotion_gate",
    "contract_id",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_BYTES = 2_000_000


class ChLtSuccessorReadinessError(ValueError):
    """Raised when the readiness contract is invalid or rebound."""


def compute_contract_id(document: Mapping[str, object]) -> str:
    """Return the semantic SHA-256 after removing the self-binding field."""

    core = dict(document)
    core.pop("contract_id", None)
    payload = json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_successor_readiness(
    *,
    repo_root: str | Path,
    contract_path: str | Path,
    expected_contract_sha256: str,
) -> dict[str, object]:
    """Verify the exact readiness chain without creating a successor."""

    _require_sha256(expected_contract_sha256, label="expected contract")
    if expected_contract_sha256 != CONTRACT_SHA256:
        raise ChLtSuccessorReadinessError(
            "caller-held contract SHA-256 does not match frozen canonical identity"
        )
    root = assert_absolute_path_has_no_links(_absolute(repo_root))
    if not root.is_dir():
        raise ChLtSuccessorReadinessError("repository root is unavailable")
    canonical_contract = _repo_path(root, CONTRACT_RELATIVE_PATH)
    if _path_key(_absolute(contract_path)) != _path_key(canonical_contract):
        raise ChLtSuccessorReadinessError("readiness contract path is not canonical")

    expected = {
        CONTRACT_RELATIVE_PATH: CONTRACT_SHA256,
        READINESS_UPDATE_RELATIVE_PATH: READINESS_UPDATE_SHA256,
        READINESS_UPDATE_V2_RELATIVE_PATH: READINESS_UPDATE_V2_SHA256,
        PREREGISTRATION_RELATIVE_PATH: PREREGISTRATION_SHA256,
        ESTIMAND_RELATIVE_PATH: ESTIMAND_SHA256,
        COMPUTE_RELATIVE_PATH: COMPUTE_SHA256,
    }
    payloads: dict[str, bytes] = {}
    paths: list[Path] = []
    for relative_path, expected_sha256 in expected.items():
        path = _repo_path(root, relative_path)
        payload = read_stable_single_link_file(
            path,
            label=f"CH LT successor readiness artifact {relative_path}",
            max_bytes=_MAX_BYTES,
        )
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ChLtSuccessorReadinessError(f"artifact SHA-256 mismatch: {relative_path}")
        payloads[relative_path] = payload
        paths.append(path)
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if os.path.samefile(left, right):
                raise ChLtSuccessorReadinessError(
                    "readiness roles must be physically distinct files"
                )

    contract = _strict_mapping(payloads[CONTRACT_RELATIVE_PATH], label="readiness contract")
    readiness_update = _strict_mapping(
        payloads[READINESS_UPDATE_RELATIVE_PATH], label="readiness update v3"
    )
    readiness_update_v2 = _strict_mapping(
        payloads[READINESS_UPDATE_V2_RELATIVE_PATH], label="superseded readiness update v2"
    )
    preregistration = _strict_mapping(
        payloads[PREREGISTRATION_RELATIVE_PATH], label="superseded preregistration"
    )
    estimand = _strict_mapping(payloads[ESTIMAND_RELATIVE_PATH], label="estimand contract")
    compute = _strict_mapping(payloads[COMPUTE_RELATIVE_PATH], label="compute contract")
    result = assess_successor_readiness(contract, document_bytes=payloads[CONTRACT_RELATIVE_PATH])
    _verify_bound_documents(contract, preregistration, estimand, compute)
    candidate_core = verify_candidate_core_v3(
        repo_root=root,
        core_path=_repo_path(root, CANDIDATE_CORE_RELATIVE_PATH),
        expected_core_sha256=CANDIDATE_CORE_SHA256,
    )
    _verify_readiness_update(readiness_update, readiness_update_v2, candidate_core)
    _equal(
        candidate_core.get("missing_evidence"),
        list(REQUIRED_BEFORE_CREATION),
        "candidate core/readiness blockers",
    )
    _equal(candidate_core.get("successor_exists"), False, "candidate core successor state")
    result.update(
        {
            "status": STATUS,
            "candidate_core_path": CANDIDATE_CORE_RELATIVE_PATH,
            "candidate_core_sha256": CANDIDATE_CORE_SHA256,
            "candidate_core_authored": True,
            "candidate_core_hash_closed_locally": True,
            "candidate_core_frozen": False,
            "candidate_core_externally_frozen": False,
            "candidate_core_admitted": False,
            "readiness_update_path": READINESS_UPDATE_RELATIVE_PATH,
            "readiness_update_sha256": READINESS_UPDATE_SHA256,
            "readiness_update_id": READINESS_UPDATE_ID,
        }
    )
    return result


def assess_successor_readiness(
    document: Mapping[str, object],
    *,
    document_bytes: bytes,
) -> dict[str, object]:
    """Validate exact frozen bytes while returning all readiness blockers."""

    parsed_document = _strict_mapping(document_bytes, label="readiness contract bytes")
    _equal(parsed_document, document, "document mapping/bytes")
    if hashlib.sha256(document_bytes).hexdigest() != CONTRACT_SHA256:
        raise ChLtSuccessorReadinessError("readiness contract document SHA-256 mismatch")

    _exact_keys(document, _TOP_LEVEL_KEYS, label="readiness contract")
    _equal(document.get("schema_version"), SCHEMA_VERSION, "schema version")
    _equal(
        document.get("contract_name"),
        "ch_lt_pit_successor_readiness_20260729_v1",
        "contract name",
    )
    _equal(document.get("contract_id"), CONTRACT_ID, "contract id")
    _equal(compute_contract_id(document), CONTRACT_ID, "computed contract id")
    _equal(
        document.get("lifecycle"),
        {
            "state": "STRUCTURAL_READINESS_CONTRACT_ONLY",
            "successor_candidate_core_state": "NOT_AUTHORED",
            "candidate_core_may_be_authored_without_admission": True,
            "candidate_core_is_an_admitted_successor": False,
            "admitted_successor_state": "NOT_CREATED",
            "contract_can_authorize_successor_creation": False,
            "external_admission_required": True,
        },
        "lifecycle",
    )
    _equal(document.get("bound_structural_inputs"), _expected_bindings(), "bindings")
    gate = _mapping(document.get("successor_creation_gate"), label="creation gate")
    _exact_keys(
        gate,
        {
            "required_before_creation",
            "current_requirement_evidence",
            "all_requirements_must_be_satisfied_by_distinct_hash_bound_evidence",
            "successor_creation_authorized",
        },
        label="creation gate",
    )
    _equal(gate.get("required_before_creation"), list(REQUIRED_BEFORE_CREATION), "requirements")
    _equal(
        gate.get("current_requirement_evidence"),
        [
            {"requirement": requirement, "status": "MISSING", "path": None, "sha256": None}
            for requirement in REQUIRED_BEFORE_CREATION
        ],
        "requirement evidence",
    )
    _equal(
        gate.get("all_requirements_must_be_satisfied_by_distinct_hash_bound_evidence"),
        True,
        "distinct evidence rule",
    )
    _equal(gate.get("successor_creation_authorized"), False, "successor authorization")
    _verify_policies(document)
    for field in (
        "scientific_admission",
        "execution_authorized",
        "production_authorization",
        "promotion_gate",
    ):
        _equal(document.get(field), False, field)

    return {
        "schema_version": ASSESSMENT_SCHEMA_VERSION,
        "status": BASE_STATUS,
        "contract_path": CONTRACT_RELATIVE_PATH,
        "contract_sha256": CONTRACT_SHA256,
        "contract_id": CONTRACT_ID,
        "successor_exists": False,
        "readiness_complete": False,
        "blockers": list(REQUIRED_BEFORE_CREATION),
        "future_holdout_consumed": False,
        "t057_consumed": False,
        "scientific_admission": False,
        "execution_authorized": False,
        "production_authorization": False,
        "promotion_gate": False,
    }


def _verify_policies(document: Mapping[str, object]) -> None:
    _equal(
        document.get("holdout_non_consumption"),
        {
            "allowed_t057_access": "SUPERSESSION_METADATA_ONLY_NO_OUTCOME_OR_SCORE_BYTES",
            "forbidden_inputs": [
                "T057_OUTCOME",
                "T057_SCORE",
                "FUTURE_HOLDOUT_TRUTH",
                "POST_ORIGIN_TRUTH",
                "HOLDOUT_DERIVED_HYPERPARAMETER_OR_THRESHOLD",
            ],
            "future_holdout_consumed": False,
            "t057_consumed": False,
            "post_holdout_retuning": "FORBIDDEN_NEW_PLAN_REQUIRED",
        },
        "holdout non-consumption",
    )
    _equal(
        document.get("estimand_origin_and_truth_freeze"),
        {
            "market": "CH",
            "timezone": "Europe/Zurich",
            "delivery_resolution_minutes": 15,
            "delivery_horizon": "M01_M36_FULL_LOCAL_DELIVERY_MONTHS",
            "primary_target": "FULL_DELIVERED_CH_PRICE_CURVE",
            "monthly_level_authority": "SOLVER",
            "all_fit_inputs_available_at_or_before_origin": True,
            "exact_origin_target_mask_and_inner_fold_inventories_required": True,
            "same_frozen_origin_inventory_for_all_compared_models": True,
            "direct_ch_truth_required_for_claimed_resolution": True,
            "native_truth_and_outcome_manifest_schema": (
                "ch_lt_native_truth_and_post_episode_outcomes_manifest.v1"
            ),
            "required_native_truth_layers": [
                "CH_HOURLY_DAY_AHEAD",
                "ONE_PREDEVELOPMENT_FROZEN_NATIVE_CH_15_MIN_PRODUCT",
            ],
            "claimed_resolution_without_native_truth": "UNSUPPORTED_NEVER_PASS",
            "post_episode_outcomes_availability_revisions_and_masks_required": True,
            "native_truth_manifest_required_fields": [
                "schema_version",
                "candidate_core_sha256",
                "truth_layer_id",
                "market_product_id",
                "auction_or_session_id",
                "price_unit",
                "source_manifest_sha256",
                "source_product_semantics_manifest_sha256",
                "source_product_semantics_admission_receipt_sha256",
                "source_timezone",
                "delivery_resolution_minutes",
                "publication_timestamps_sha256",
                "delivery_timestamps_sha256",
                "native_observation_values_sha256",
                "availability_lag_policy",
                "revision_policy",
                "revision_inventory_sha256",
                "missingness_mask_sha256",
                "dst_fold_and_gap_policy",
                "post_episode_outcome_inventory_sha256",
            ],
            "hourly_duplication_or_interpolation_as_native_quarter_hour_truth": "FORBIDDEN",
            "post_selection_origin_reuse_for_confirmation": "FORBIDDEN",
        },
        "estimand/origin/truth policy",
    )
    _equal(
        document.get("deterministic_candidate_and_market_gates"),
        {
            "manifest_schema": "ch_lt_deterministic_candidate_market_gate_manifest.v1",
            "candidate_and_baseline_graphs_frozen_before_outer_origin_truth": True,
            "hyperparameter_and_complexity_selection_scope": "INNER_FOLDS_ONLY",
            "market_quote_inventory": (
                "EXACT_ORIGIN_AVAILABLE_CH_EEX_VINTAGE_AND_NORMALIZED_PRODUCT_BYTES"
            ),
            "required_repricing_products": [
                "BASE",
                "PEAK",
                "OFFPEAK",
                "ALL_ORIGIN_AVAILABLE_EEX_PRODUCTS",
            ],
            "repricing_tolerance_policy": (
                "PREDECLARED_ABSOLUTE_AND_RELATIVE_TOLERANCES_PER_PRODUCT"
            ),
            "monthly_level_authority": "SOLVER",
            "ensemble_expected_curve_reprices_origin_available_products": True,
            "cascade_invariance_required": True,
            "quote_to_curve_sensitivity_diagnostics": [
                "JACOBIAN_RANK",
                "CONDITION_NUMBER",
                "PERTURBATION_STABILITY",
            ],
            "sensitivity_rank_condition_and_stability_thresholds_frozen_before_truth": True,
            "insufficient_or_unstable_complexity_evidence": "UNSUPPORTED_NEVER_PASS",
            "post_solver_individual_month_patch": "FORBIDDEN",
            "negative_price_support_required": True,
            "ompex_role": "BENCHMARK_ONLY_FORBIDDEN_AS_TRUTH_MODEL_INPUT_OR_LEVEL_AUTHORITY",
            "required_manifest_fields": [
                "schema_version",
                "candidate_core_sha256",
                "candidate_graph_sha256",
                "baseline_graphs_sha256",
                "feature_inventory_sha256",
                "hyperparameter_grid_sha256",
                "complexity_selection_rule_sha256",
                "inner_fold_inventory_sha256",
                "inner_fold_selection_results_sha256",
                "selected_complexity_by_outer_origin_sha256",
                "complexity_diagnostics_sha256",
                "eex_vintage_manifest_sha256",
                "normalized_product_inventory_sha256",
                "repricing_tolerances_sha256",
                "repricing_results_sha256",
                "monthly_solver_means_sha256",
                "cascade_invariance_result_sha256",
                "quote_curve_jacobian_sha256",
                "jacobian_rank",
                "condition_number",
                "perturbation_stability_result_sha256",
                "sensitivity_thresholds_sha256",
            ],
        },
        "deterministic candidate/market gate policy",
    )
    _equal(
        document.get("dependence_power_and_multiplicity"),
        {
            "direct_ch_truth_and_dependence_aware_clusters_required": True,
            "cross_fitting_required_for_nuisance_calibration_and_selection": True,
            "cross_fitting_substitutes_for_independent_confirmation": False,
            "cross_fitting_creates_independent_units": False,
            "insufficient_effective_clusters_or_power": "UNSUPPORTED_NEVER_PASS",
            "de_lu_pilot_required_n_reuse": "FORBIDDEN",
            "dependence_model_and_block_or_cluster_unit_frozen_before_truth": True,
            "effective_sample_size_method_frozen_before_truth": True,
            "familywise_error_or_fdr_policy_frozen_before_truth": True,
            "target_power_and_mde_frozen_before_truth": True,
            "simultaneous_inference_across_horizon_regime_metric_and_population": True,
            "plugin_variance_or_leave_one_out_value_as_confidence_bound": "FORBIDDEN",
            "overlapping_origins_treated_as_independent": "FORBIDDEN",
        },
        "dependence/power/multiplicity policy",
    )
    _equal(
        document.get("probabilistic_scenario_and_monte_carlo"),
        {
            "manifest_schema": "ch_lt_probabilistic_scenario_mc_design_manifest.v1",
            "primary_representation": "JOINT_FULL_PRICE_SCENARIO_PATHS_PLUS_MARGINAL_QUANTILES",
            "ensemble_monthly_forward_consistency": (
                "ENERGY_WEIGHTED_SCENARIO_EXPECTATION_EQUALS_MONTHLY_SOLVER_FORWARD"
            ),
            "scenario_specific_monthly_level_uncertainty": (
                "ALLOWED_ONLY_WITH_ENSEMBLE_ZERO_MEAN_AND_FROZEN_COVARIANCE_DESIGN"
            ),
            "pathwise_fixed_monthly_level": "SEPARATE_SHAPE_ONLY_DIAGNOSTIC_PRODUCT",
            "scenario_covariance_and_dependence_frozen_before_truth": True,
            "candidate_independent_cpu_generated_shock_bytes_required": True,
            "scenario_count_seed_engine_backend_chunk_and_order_frozen": True,
            "monte_carlo_error_study_and_freeze_receipt_required": True,
            "monte_carlo_error_bound_relative_to_every_scientific_and_economic_margin": True,
            "quantile_grid_frozen_before_truth": True,
            "exact_temporal_horizon_and_population_aggregation_weights_required": True,
            "energy_score_scaling_and_dimension_strategy_frozen_before_truth": True,
            "variogram_order_lags_graph_and_weights_frozen_before_truth": True,
            "dependence_and_tail_event_calibration_uses_cluster_aware_bands": True,
            "minimum_effective_calibration_cells_frozen_before_truth": True,
            "insufficient_effective_calibration_cells": "UNSUPPORTED_NEVER_PASS",
            "calibration_multiplicity_policy_frozen_before_truth": True,
            "marginal_scores": [
                "CRPS",
                "WIS",
                "PINBALL_LOSS",
                "COVERAGE",
                "PIT_OR_RANK_HISTOGRAM",
            ],
            "multivariate_scores": ["ENERGY_SCORE", "VARIOGRAM_SCORE"],
            "calibration_required_by_horizon_regime_and_population": True,
            "score_recomputation_on_cpu_float64_oracle": True,
            "required_manifest_fields": [
                "schema_version",
                "candidate_core_sha256",
                "quantile_grid",
                "temporal_aggregation_weights_sha256",
                "horizon_aggregation_weights_sha256",
                "population_aggregation_weights_sha256",
                "energy_score_scaling_strategy",
                "energy_score_dimension_strategy",
                "variogram_order",
                "variogram_lags",
                "variogram_graph_sha256",
                "variogram_weights_sha256",
                "dependence_calibration_method",
                "tail_event_definitions_sha256",
                "cluster_aware_band_method",
                "minimum_effective_calibration_cells",
                "calibration_multiplicity_policy",
                "covariance_design_sha256",
                "scenario_count",
                "seed",
                "rng_engine",
                "backend",
                "chunk_size",
                "generation_order",
                "cpu_shock_bytes_sha256",
                "mc_error_bounds_by_metric_and_population_sha256",
                "scientific_margins_sha256",
                "economic_margins_sha256",
            ],
            "post_truth_scenario_or_threshold_change": "FORBIDDEN_NEW_PLAN_REQUIRED",
        },
        "probabilistic/scenario/MC policy",
    )
    _equal(
        document.get("economic_materiality"),
        {
            "manifest_schema": "ch_lt_fmv_economic_materiality_manifest.v1",
            "risk_approved_populations_and_thresholds_required": True,
            "fil_and_acc_scored_on_full_delivered_price_with_direction": True,
            "bloc_13_uses_frozen_payoff_not_generic_profile": True,
            "hydro_models_use_same_optimizer_constraints_and_information_set": True,
            "market_value_capture_factor_and_regret_metrics_required": True,
            "fx_valuation_date_discounting_and_units_frozen": True,
            "required_simultaneous_populations": [
                "FIL",
                "ACC",
                "BLOC_13",
                "HYDRO_DISPATCH",
            ],
            "simultaneous_noninferiority_and_material_improvement_rule_frozen_before_truth": True,
            "fmv_risk_signed_decision_binds_thresholds_adverse_populations_and_tail_rules": True,
            "required_manifest_fields": [
                "schema_version",
                "candidate_core_sha256",
                "fil_population_bytes_sha256",
                "acc_population_bytes_sha256",
                "bloc_13_payoff_sha256",
                "bloc_13_settlement_rule_sha256",
                "bloc_13_discount_curve_sha256",
                "bloc_13_fx_rule_sha256",
                "hydro_policy_sha256",
                "hydro_optimizer_constraints_sha256",
                "hydro_initial_states_sha256",
                "hydro_terminal_states_sha256",
                "hydro_costs_sha256",
                "population_metrics_and_directions_sha256",
                "simultaneous_noninferiority_rule_sha256",
                "material_improvement_thresholds_sha256",
                "adverse_population_rules_sha256",
                "tail_event_rules_sha256",
                "fmv_risk_owner_identity",
                "fmv_risk_signed_decision_sha256",
            ],
            "ompex_role": "BENCHMARK_ONLY_FORBIDDEN_AS_TRUTH_MODEL_INPUT_OR_MDE_EVIDENCE",
            "default_capture_premium_as_evidence": "FORBIDDEN",
        },
        "economic materiality policy",
    )
    _equal(
        document.get("compute_runtime_qualification"),
        {
            "manifest_schema": "ch_lt_compute_runtime_qualification_manifest.v1",
            "cpu_float64_oracle_is_canonical_numerical_reference": True,
            "cpu_float64_oracle_does_not_grant_scientific_or_data_authority": True,
            "exact_wheel_runtime_input_code_and_shock_bindings_required": True,
            "minimum_repeats_per_cpu_and_gpu_backend": 3,
            "cpu_gpu_parity_required": True,
            "parity_tolerances_frozen_before_execution": True,
            "tf32_amp_and_compile_acceleration": "DISABLED_FOR_QUALIFICATION",
            "deterministic_algorithms_required": True,
            "gpu_fallback_policy_frozen_before_execution": True,
            "solver_repricing_and_hard_market_gates_on_gpu": "FORBIDDEN_CPU_FLOAT64_ONLY",
            "required_manifest_fields": [
                "schema_version",
                "candidate_core_sha256",
                "wheel_sha256",
                "runtime_manifest_sha256",
                "input_manifests_sha256",
                "source_tree_sha256",
                "cpu_shock_bytes_sha256",
                "cpu_backend_identity",
                "gpu_backend_identity",
                "cpu_repeat_receipts_sha256",
                "gpu_repeat_receipts_sha256",
                "cpu_repeat_count",
                "gpu_repeat_count",
                "determinism_settings_sha256",
                "parity_tolerances_sha256",
                "parity_results_sha256",
                "fallback_policy_sha256",
                "runtime_budget_result_sha256",
            ],
        },
        "compute runtime qualification policy",
    )
    _equal(
        document.get("external_admission_and_ledger"),
        {
            "immutable_receipt_free_plan_core_required": True,
            "independently_signed_admission_envelope_required": True,
            "monotone_one_shot_attempt_ledger_required": True,
            "seal_predictions_and_scenarios_before_truth_access": True,
            "builder_inaccessible_evidence_and_external_cas_worm_fresh_head_required": True,
            "trusted_time_and_independent_acquisition_signature_required": True,
            "admission_envelope_cannot_authorize_production": True,
        },
        "external admission/ledger policy",
    )
    _equal(
        document.get("independent_review"),
        {
            "required_roles": ["SECURITY", "IT_OPERATIONS", "QUANT_DATA", "FMV_RISK_OWNER"],
            "reviews_bind_exact_candidate_core_and_evidence_hashes": True,
            "review_after_candidate_core_freeze_before_admitted_successor": True,
            "candidate_core_cannot_execute_or_consume_holdout": True,
            "self_review_or_shared_role_manifest": "FORBIDDEN",
        },
        "independent review policy",
    )


def _verify_bound_documents(
    contract: Mapping[str, object],
    preregistration: Mapping[str, object],
    estimand: Mapping[str, object],
    compute: Mapping[str, object],
) -> None:
    _equal(preregistration.get("plan_id"), PREREGISTRATION_PLAN_ID, "superseded plan id")
    _equal(estimand.get("contract_id"), ESTIMAND_CONTRACT_ID, "estimand contract id")
    _equal(compute.get("contract_id"), COMPUTE_CONTRACT_ID, "compute contract id")
    _equal(preregistration.get("production_authorization"), False, "old production")
    _equal(compute.get("execution_authorized"), False, "compute execution")
    economic = _mapping(contract.get("economic_materiality"), label="economic policy")
    profile = _mapping(estimand.get("profile_policy"), label="estimand profile policy")
    dispatch = _mapping(estimand.get("dispatch_policy"), label="estimand dispatch policy")
    _equal(
        economic.get("required_simultaneous_populations"),
        profile.get("required_population_ids"),
        "readiness/estimand population ids",
    )
    _equal(dispatch.get("profile_id"), "HYDRO_DISPATCH", "estimand dispatch profile id")


def _verify_readiness_update(
    update: Mapping[str, object],
    update_v2: Mapping[str, object],
    candidate_core: Mapping[str, object],
) -> None:
    _exact_keys(
        update,
        {
            "schema_version",
            "update_name",
            "superseded_readiness_update",
            "base_readiness_contract",
            "selected_candidate_core",
            "lifecycle",
            "current_requirement_evidence",
            "legacy_t057_policy",
            "scientific_admission",
            "execution_authorized",
            "production_authorization",
            "promotion_gate",
            "readiness_id",
        },
        label="readiness update v3",
    )
    _equal(
        update.get("schema_version"),
        "ch_lt_pit_successor_readiness_update.v3",
        "readiness update schema",
    )
    _equal(update.get("readiness_id"), READINESS_UPDATE_ID, "readiness update id")
    semantic = dict(update)
    semantic.pop("readiness_id", None)
    payload = json.dumps(
        semantic,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    _equal(hashlib.sha256(payload).hexdigest(), READINESS_UPDATE_ID, "computed update id")
    superseded = _mapping(
        update.get("superseded_readiness_update"), label="superseded readiness update"
    )
    _equal(
        dict(superseded),
        {
            "path": READINESS_UPDATE_V2_RELATIVE_PATH,
            "sha256": READINESS_UPDATE_V2_SHA256,
            "readiness_id": READINESS_UPDATE_V2_ID,
            "reason": (
                "V2_SELECTED_CORE_GLOBALIZED_FULL_HORIZON_OVERLAP_GEOMETRY_"
                "ACROSS_PRIMARY_BUCKET_CONTRASTS"
            ),
        },
        "superseded readiness update binding",
    )
    _equal(
        update_v2.get("schema_version"),
        "ch_lt_pit_successor_readiness_update.v2",
        "superseded readiness update schema",
    )
    _equal(update_v2.get("readiness_id"), READINESS_UPDATE_V2_ID, "superseded update id")
    base = _mapping(update.get("base_readiness_contract"), label="base readiness binding")
    _equal(base.get("path"), CONTRACT_RELATIVE_PATH, "base readiness path")
    _equal(base.get("sha256"), CONTRACT_SHA256, "base readiness SHA-256")
    _equal(base.get("contract_id"), CONTRACT_ID, "base readiness id")
    _equal(
        base.get("preserved_policy_role"),
        (
            "ALL_POLICIES_PRESERVED_EXCEPT_CANDIDATE_CORE_LIFECYCLE_T057_"
            "ACCESS_AND_CONTRAST_DEPENDENCE_GEOMETRY_OBSERVATIONS"
        ),
        "base readiness preservation role",
    )
    selected = _mapping(update.get("selected_candidate_core"), label="selected core")
    _equal(selected.get("path"), CANDIDATE_CORE_RELATIVE_PATH, "selected core path")
    _equal(selected.get("sha256"), CANDIDATE_CORE_SHA256, "selected core SHA-256")
    _equal(selected.get("core_id"), candidate_core.get("core_id"), "selected core id")
    lifecycle = _mapping(update.get("lifecycle"), label="updated lifecycle")
    _equal(
        lifecycle.get("prior_candidate_core_state_superseded"),
        "V2_OUTCOME_BLIND_GLOBAL_OVERLAP_GEOMETRY_P1_NOT_ADMITTED",
        "superseded candidate core state",
    )
    _equal(
        lifecycle.get("current_candidate_core_state"),
        "AUTHORED_LOCAL_HASH_CLOSED_NOT_EXTERNALLY_FROZEN_NOT_ADMITTED",
        "updated candidate core state",
    )
    _equal(
        lifecycle.get("base_t057_access_observation_superseded"),
        (
            "/holdout_non_consumption/allowed_t057_access="
            "SUPERSESSION_METADATA_ONLY_NO_OUTCOME_OR_SCORE_BYTES"
        ),
        "superseded base T057 access observation",
    )
    _equal(
        lifecycle.get("current_t057_access_observation"),
        "OUTCOME_BLIND_TOMBSTONE_ONLY",
        "current T057 access observation",
    )
    _equal(lifecycle.get("admitted_successor_state"), "NOT_CREATED", "successor state")
    _equal(lifecycle.get("readiness_complete"), False, "updated readiness")
    _equal(
        update.get("current_requirement_evidence"),
        [
            {"requirement": item, "status": "MISSING", "path": None, "sha256": None}
            for item in REQUIRED_BEFORE_CREATION
        ],
        "updated requirement evidence",
    )
    legacy = _mapping(update.get("legacy_t057_policy"), label="legacy T057 policy")
    _equal(legacy.get("outcome_metadata_exposure_acknowledged"), True, "T057 exposure")
    _equal(
        legacy.get("legacy_t057_confirmation_reuse"),
        "PERMANENTLY_FORBIDDEN",
        "T057 reuse",
    )
    _equal(legacy.get("selected_core_access"), "OUTCOME_BLIND_TOMBSTONE_ONLY", "T057 access")
    for field in (
        "scientific_admission",
        "execution_authorized",
        "production_authorization",
        "promotion_gate",
    ):
        _equal(update.get(field), False, f"readiness update {field}")


def _expected_bindings() -> dict[str, object]:
    return {
        "superseded_preregistration": {
            "path": PREREGISTRATION_RELATIVE_PATH,
            "sha256": PREREGISTRATION_SHA256,
            "plan_id": PREREGISTRATION_PLAN_ID,
        },
        "estimand_contract": {
            "path": ESTIMAND_RELATIVE_PATH,
            "sha256": ESTIMAND_SHA256,
            "contract_id": ESTIMAND_CONTRACT_ID,
        },
        "compute_contract": {
            "path": COMPUTE_RELATIVE_PATH,
            "sha256": COMPUTE_SHA256,
            "contract_id": COMPUTE_CONTRACT_ID,
        },
    }


def _strict_mapping(payload: bytes, *, label: str) -> Mapping[str, object]:
    try:
        parsed = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChLtSuccessorReadinessError(f"{label} is invalid JSON") from exc
    return _mapping(parsed, label=label)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ChLtSuccessorReadinessError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ChLtSuccessorReadinessError(f"{label} keys are not exact")


def _equal(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ChLtSuccessorReadinessError(f"{label} is invalid")


def _repo_path(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != relative_path:
        raise ChLtSuccessorReadinessError("relative path is unsafe")
    return root.joinpath(*pure.parts)


def _absolute(path: str | Path) -> Path:
    lexical = Path(path).expanduser()
    return lexical if lexical.is_absolute() else Path(os.path.abspath(lexical))


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ChLtSuccessorReadinessError(f"{label} SHA-256 is invalid")


__all__ = [
    "ASSESSMENT_SCHEMA_VERSION",
    "BASE_STATUS",
    "CONTRACT_ID",
    "CONTRACT_RELATIVE_PATH",
    "CONTRACT_SHA256",
    "REQUIRED_BEFORE_CREATION",
    "STATUS",
    "ChLtSuccessorReadinessError",
    "assess_successor_readiness",
    "compute_contract_id",
    "verify_successor_readiness",
]
