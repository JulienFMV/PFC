"""Fail-closed structural contract for the full delivered Swiss LT estimand.

Schema v1 defines what must eventually be measured.  It cannot admit an
evaluation or production run; independent truth, profile, dispatch, risk and
origin/mask evidence remain external prerequisites.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.pipeline.strict_structured_data import load_strict_json

SCHEMA_VERSION = "ch_lt_estimand_and_economic_design.v1"
AUDIT_SCHEMA_VERSION = "ch_lt_estimand_and_economic_design_audit.v1"
STATUS = "STRUCTURAL_ESTIMAND_DRAFT_BLOCKED_NOT_EXECUTABLE"
DRAFT_STATE = "DRAFT_NOT_FROZEN"
CONTRACT_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-ESTIMAND-AND-ECONOMIC-DESIGN-DRAFT-20260724.json"
)
DOCUMENT_SHA256 = "4209931e28a7c1cf2a4224d779f73648c4c9c5eac55df0a7ba1ad872226e2931"
CONTRACT_ID = "da4090073a4566f662e47fa59e206e1485d683305a1134f2089ebb13a4daa344"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "scope",
        "market",
        "timezone",
        "lifecycle",
        "horizon_policy",
        "target_layers",
        "truth_policy",
        "origin_design_policy",
        "eligibility_policy",
        "calendar_policy",
        "metric_policy",
        "statistical_decision_policy",
        "probabilistic_policy",
        "profile_policy",
        "dispatch_policy",
        "execution_controls",
        "evidence_bindings",
        "contract_id",
    }
)
_LIFECYCLE_FIELDS = frozenset({"state", "frozen_at_utc"})

EXPECTED_HORIZON_POLICY: dict[str, object] = {
    "delivery_support": "FULL_LOCAL_DELIVERY_MONTHS_ONLY",
    "lead_month_formula": (
        "12_X_DELIVERY_YEAR_MINUS_ORIGIN_YEAR_PLUS_DELIVERY_MONTH_MINUS_ORIGIN_MONTH"
    ),
    "eligible_lead_month_min": 1,
    "eligible_lead_month_max": 36,
    "current_partial_month_policy": "EXCLUDED",
    "outside_support_policy": "UNSUPPORTED_NEVER_EXTRAPOLATED_AS_PASS",
    "buckets": [
        {"bucket_id": "M01_M06", "minimum_lead_month": 1, "maximum_lead_month": 6},
        {"bucket_id": "M07_M12", "minimum_lead_month": 7, "maximum_lead_month": 12},
        {"bucket_id": "M13_M24", "minimum_lead_month": 13, "maximum_lead_month": 24},
        {"bucket_id": "M25_M36", "minimum_lead_month": 25, "maximum_lead_month": 36},
    ],
}

EXPECTED_TARGET_LAYERS: list[dict[str, object]] = [
    {
        "layer_id": "MARKET_CONSISTENCY",
        "target": "FINAL_DELIVERED_CURVE_REPRICES_ORIGIN_AVAILABLE_CH_EEX_PRODUCTS",
        "truth_role": "ORIGIN_AVAILABLE_HARD_CH_EEX_QUOTES",
        "scoring": "BASE_PEAK_OFFPEAK_PRODUCT_REPRICING",
        "level_authority": "MONTHLY_SOLVER",
        "product_repricing_tolerance_eur_mwh": 1e-6,
        "monthly_solver_level_tolerance_eur_mwh": 1e-9,
        "cascade_invariance_tolerance_eur_mwh": 1e-8,
    },
    {
        "layer_id": "HOURLY_SHAPE",
        "target": "DIRECT_CH_DAY_AHEAD_AUCTION_PRICE_AT_NATIVE_HOURLY_OR_FINER_RESOLUTION",
        "truth_role": "SCORING_ONLY_NEVER_MONTHLY_LEVEL_AUTHORITY",
        "scoring": "MONTHLY_LEVEL_NEUTRALIZED_PRICE_ERROR",
        "maximum_native_interval_minutes": 60,
        "hourly_aggregation": "ENERGY_WEIGHTED_FROM_NATIVE_IF_FINER",
    },
    {
        "layer_id": "QUARTER_HOURLY_SHAPE",
        "target": "DIRECT_CH_NATIVE_QUARTER_HOURLY_PRICE_FOR_ONE_FROZEN_MARKET_PRODUCT",
        "truth_role": "SCORING_ONLY_NEVER_MONTHLY_LEVEL_AUTHORITY",
        "scoring": "MONTHLY_LEVEL_NEUTRALIZED_PRICE_ERROR",
        "required_native_resolution_minutes": 15,
        "hourly_duplication_or_interpolation": "FORBIDDEN",
    },
    {
        "layer_id": "PROBABILISTIC_SCENARIOS",
        "target": "JOINT_CH_DELIVERED_PRICE_PATH_CONDITIONAL_ON_ORIGIN_INFORMATION_SET",
        "truth_role": "DIRECT_CH_SCORING_TRUTH_ONLY_NEVER_MODEL_INPUT",
        "scoring": "PROPER_UNIVARIATE_AND_MULTIVARIATE_SCORES_PLUS_CALIBRATION",
        "monthly_level_authority": (
            "ENSEMBLE_ENERGY_WEIGHTED_MONTHLY_EXPECTATION_EQUALS_SOLVER_FORWARD"
        ),
        "temporal_independence_assumption": "FORBIDDEN",
    },
]

EXPECTED_TRUTH_POLICY: dict[str, object] = {
    "market": "CH",
    "timezone": "Europe/Zurich",
    "delivery_timestamp_axis": "UTC_WITH_EUROPE_ZURICH_LOCAL_MARKET_CALENDAR",
    "price_unit": "EUR_PER_MWH",
    "delivery_resolution_minutes": 15,
    "negative_and_zero_prices_retained": True,
    "spot_truth_is_model_input": False,
    "post_delivery_truth_release_allowed_for_scoring_only": True,
    "truth_product_id_freeze_boundary": (
        "BEFORE_MODEL_DEVELOPMENT_SELECTION_TUNING_OR_SCORING"
    ),
    "distinct_truth_products_must_not_be_pooled": True,
    "all_fit_inputs_available_at_or_before_origin": True,
    "later_revisions_forbidden": True,
    "proxy_or_neighbor_truth_for_ch": "FORBIDDEN",
    "ompex_role": "DIAGNOSTIC_BENCHMARK_ONLY_NEVER_TRUTH_INPUT_SELECTION_OR_GATE",
    "monthly_level_authority": "SOLVER_WITH_HARD_CH_EEX_CONSTRAINTS",
}

EXPECTED_ORIGIN_DESIGN_POLICY: dict[str, object] = {
    "primary_observation_and_weighting_unit": "UNIQUE_ORIGIN_AS_OF_UTC",
    "duplicate_origin_as_of_utc": "FORBIDDEN",
    "issuance_cadence_and_calendar_must_be_frozen": True,
    "origin_density_cannot_change_weighting_without_new_design": True,
    "overlapping_delivery_targets_require_clustered_dependence_treatment": True,
    "minimum_unique_origins_by_horizon_and_stratum_from_power_design": True,
    "same_frozen_origin_inventory_for_all_compared_models": True,
    "post_selection_origin_reuse_for_confirmation": "FORBIDDEN",
}

EXPECTED_ELIGIBILITY_POLICY: dict[str, object] = {
    "same_rows_for_all_challengers_and_baselines": True,
    "complete_case_intersection": [
        "PREDICTION",
        "BASELINE",
        "DIRECT_NATIVE_TRUTH",
        "ENERGY_WEIGHT",
        "CALENDAR_LABELS",
    ],
    "missing_or_nonfinite": "UNSUPPORTED_NEVER_ZERO_OR_WIN",
    "native_truth_resolution_mismatch": "UNSUPPORTED",
    "derived_quarter_hour_truth_from_hourly": "FORBIDDEN",
    "target_or_revision_leakage": "FAIL",
    "dst_policy": "EXPECTED_UTC_INTERVALS_FOR_EUROPE_ZURICH_LOCAL_DELIVERY_MONTH",
    "monthly_completeness_required": True,
    "partial_month_policy": "UNSUPPORTED",
    "origin_weighting": "EQUAL_WEIGHT_AFTER_WITHIN_ORIGIN_ENERGY_WEIGHTING",
    "stratification_required": [
        "LEAD_MONTH_BUCKET",
        "SEASON",
        "WEEKDAY_HOLIDAY",
        "HOUR_RAMP",
        "BASE_PEAK_OFFPEAK",
        "HYDRO_REGIME",
        "NEGATIVE_PRICE",
        "PRICE_SPIKE",
        "RENEWABLE_HIGH",
        "CROSS_BORDER_STRESS",
    ],
}

EXPECTED_CALENDAR_POLICY: dict[str, object] = {
    "definition_manifest_required_fields": [
        "calendar_id",
        "calendar_version",
        "timezone",
        "utc_interval_generation_rule",
        "dst_fold_and_gap_rule",
        "base_peak_offpeak_rule",
        "weekday_holiday_rule",
        "holiday_source_and_version",
        "season_rule",
        "stress_regime_definitions",
        "manifest_sha256",
    ],
    "calendar_and_regime_freeze_boundary": (
        "BEFORE_MODEL_DEVELOPMENT_SELECTION_TUNING_OR_SCORING"
    ),
    "data_driven_stratum_redefinition_after_results": "FORBIDDEN",
}

EXPECTED_METRIC_POLICY: dict[str, object] = {
    "monthly_shape_centering": (
        "P_MINUS_ENERGY_WEIGHTED_LOCAL_DELIVERY_MONTH_MEAN_SEPARATELY_FOR_PREDICTION_AND_TRUTH"
    ),
    "primary_metric": "MONTHLY_LEVEL_NEUTRALIZED_MAE_EUR_MWH",
    "secondary_metrics": [
        "MONTHLY_LEVEL_NEUTRALIZED_RMSE_EUR_MWH",
        "MONTHLY_LEVEL_NEUTRALIZED_BIAS_EUR_MWH",
        "P95_ABSOLUTE_ERROR_EUR_MWH",
        "BASE_PEAK_OFFPEAK_WEIGHTED_ERROR_EUR_MWH",
        "FIXED_PROFILE_CAPTURE_PRICE_ERROR_EUR_MWH",
        "BLOC_13_CONTRACT_PAYOFF_ERROR_CHF",
        "HYDRO_DISPATCH_REALIZED_NET_VALUE_OR_REGRET_CHF",
    ],
    "mape_primary_or_gate": False,
    "market_repricing_scored_separately_from_spot_shape": True,
    "quarter_hour_gate_requires_native_quarter_hour_truth": True,
    "unsupported_layer_cannot_be_hidden_by_aggregate_pass": True,
}

EXPECTED_STATISTICAL_DECISION_POLICY: dict[str, object] = {
    "design_manifest_required_fields": [
        "design_id",
        "design_version",
        "candidate_and_baseline_hypotheses",
        "estimand_contrasts_by_layer_population_and_horizon",
        "noninferiority_and_superiority_margins",
        "economic_mde_by_population_and_horizon",
        "familywise_alpha",
        "target_power",
        "multiplicity_family_and_correction",
        "confidence_interval_method",
        "dependence_cluster_and_overlap_model",
        "minimum_unique_origins_by_horizon_and_stratum",
        "effect_size_and_variance_source",
        "missingness_and_unsupported_policy",
        "model_selection_and_reuse_policy",
        "manifest_sha256",
    ],
    "market_consistency_gates": "ABSOLUTE_HARD_GATES_NEVER_COMPENSATED",
    "shape_comparison": "FROZEN_CANDIDATE_VERSUS_FROZEN_BASELINES_ON_IDENTICAL_ROWS",
    "economic_comparison": "FULL_PRICE_OR_CASHFLOW_NOT_CENTERED_SHAPE_ONLY",
    "post_hoc_threshold_or_family_change": "FORBIDDEN",
    "pilot_selected_origins_reused_for_confirmation": "FORBIDDEN",
    "all_reported_claims_require_predeclared_simultaneous_inference": True,
}

EXPECTED_PROBABILISTIC_POLICY: dict[str, object] = {
    "primary_representation": "JOINT_SCENARIO_PATHS_WITH_DERIVED_MARGINAL_QUANTILES",
    "ensemble_monthly_forward_consistency": (
        "ENERGY_WEIGHTED_SCENARIO_EXPECTATION_EQUALS_MONTHLY_SOLVER_FORWARD"
    ),
    "scenario_specific_monthly_level_uncertainty": (
        "ALLOWED_ONLY_WITH_ENSEMBLE_ZERO_MEAN_AND_FROZEN_COVARIANCE_DESIGN"
    ),
    "shape_only_fixed_level_scenarios": "SEPARATE_DIAGNOSTIC_PRODUCT_ONLY",
    "required_univariate_scores": ["CRPS", "WIS", "PINBALL_LOSS"],
    "required_calibration_diagnostics": [
        "PIT_OR_RANDOMIZED_PIT",
        "EMPIRICAL_COVERAGE_AND_INTERVAL_WIDTH",
        "TAIL_EVENT_CALIBRATION",
    ],
    "required_multivariate_scores": ["ENERGY_SCORE", "VARIOGRAM_SCORE"],
    "required_coherence_checks": [
        "SCENARIO_QUANTILE_CONSISTENCY",
        "TEMPORAL_AND_CROSS_HORIZON_DEPENDENCE",
        "BASE_PEAK_OFFPEAK_AND_MONTHLY_AGGREGATION",
    ],
    "scenario_count_seed_engine_and_backend_must_be_frozen": True,
    "cpu_gpu_reproducibility_tolerance_must_be_predeclared": True,
    "monte_carlo_error_must_be_bounded_before_comparison": True,
}

EXPECTED_PROFILE_POLICY: dict[str, object] = {
    "required_population_ids": ["FIL", "ACC", "BLOC_13", "HYDRO_DISPATCH"],
    "fixed_profile_roles": ["FIL", "ACC"],
    "fixed_profile_manifest_required_fields": [
        "profile_id",
        "profile_version",
        "timestamp_utc",
        "delivery_timezone",
        "volume_mwh",
        "direction",
        "availability_rule",
        "source_document_id",
        "row_hash",
    ],
    "fixed_profile_volume_policy": "FINITE_NONNEGATIVE_MWH_SEPARATE_DIRECTION_FIELD",
    "fixed_profile_direction_values": ["GENERATION", "CONSUMPTION"],
    "fixed_profile_capture_price_formula": "SUM_PRICE_X_VOLUME_DIV_SUM_VOLUME",
    "fixed_profile_pricing_basis": "FULL_DELIVERED_PRICE_NOT_MONTHLY_CENTERED_SHAPE",
    "fixed_profile_direction_sign": {"GENERATION": 1, "CONSUMPTION": -1},
    "fixed_profile_signed_cashflow_formula": (
        "DIRECTION_SIGN_X_SUM_FULL_PRICE_EUR_PER_MWH_X_VOLUME_MWH"
    ),
    "fixed_profile_capture_error_formula": (
        "ABS_PREDICTED_CAPTURE_PRICE_MINUS_REALIZED_CAPTURE_PRICE_EUR_MWH"
    ),
    "zero_total_volume_policy": "UNSUPPORTED",
    "capture_factor_role": "SECONDARY_ONLY_WHEN_REFERENCE_PRICE_ABS_GT_APPROVED_EPSILON",
    "contract_payoff_roles": ["BLOC_13"],
    "contract_payoff_manifest_required_fields": [
        "contract_id",
        "contract_version",
        "payoff_formula",
        "underlying_and_legs",
        "strike_or_collar_parameters",
        "settlement_convention",
        "delivery_timezone",
        "position_sign_convention",
        "transaction_and_imbalance_cost_rule",
        "source_document_id",
        "manifest_sha256",
    ],
    "bloc_13_profile_proxy_policy": "FORBIDDEN_WITHOUT_FROZEN_CONTRACT_PAYOFF_MANIFEST",
    "chf_metric_fx_policy": (
        "REQUIRES_FROZEN_ORIGIN_AVAILABLE_FX_FIXING_AND_CONVERSION_MANIFEST"
    ),
    "negative_price_policy": "RETAIN_IN_PRICE_AND_CASHFLOW",
    "pfc_flavors_default_capture_premium_role": "FORBIDDEN_AS_VALIDATION_OR_MDE_EVIDENCE",
    "ompex_profile_role": "BENCHMARK_ONLY_NEVER_PROFILE_TRUTH",
}

EXPECTED_DISPATCH_POLICY: dict[str, object] = {
    "profile_id": "HYDRO_DISPATCH",
    "policy_freeze_boundary": "BEFORE_MODEL_DEVELOPMENT_SELECTION_TUNING_OR_SCORING",
    "non_anticipative_information_set_bound_to_origin": True,
    "physical_constraints_manifest_required": True,
    "water_value_and_terminal_condition_manifest_required": True,
    "transaction_imbalance_and_operating_cost_manifest_required": True,
    "primary_economic_metric": "REALIZED_NET_VALUE_OR_REGRET_CHF",
    "price_capture_premium_shortcut": "FORBIDDEN_AS_DECISION_VALUE_EVIDENCE",
    "same_optimizer_constraints_information_and_decision_rule_class": True,
    "model_specific_actions": (
        "ALLOWED_ONLY_WHEN_NONANTICIPATIVE_AND_OPTIMIZED_FROM_EACH_ORIGIN_CURVE"
    ),
    "realized_value_recourse": "ONLY_ACTIONS_ALLOWED_BY_THE_FROZEN_POLICY",
    "regret_benchmark": (
        "FEASIBLE_CLAIRVOYANT_REALIZED_NET_VALUE_MINUS_MODEL_POLICY_REALIZED_NET_VALUE"
    ),
    "regret_benchmark_same_initial_terminal_state_and_constraints": True,
}

EXPECTED_EXECUTION_CONTROLS: dict[str, object] = {
    "schema_v1_authorizes_execution": False,
    "production_authorization": False,
    "promotion_gate": False,
    "future_holdout_consumption": False,
    "external_admission_required": True,
    "exact_origin_target_mask_inventory_required": True,
    "output_append_only": True,
}

EVIDENCE_FIELDS = (
    "eex_quote_vintage_manifest_sha256",
    "monthly_solver_configuration_manifest_sha256",
    "candidate_baseline_identity_manifest_sha256",
    "direct_ch_hourly_truth_manifest_sha256",
    "direct_ch_quarter_hour_truth_manifest_sha256",
    "statistical_design_manifest_sha256",
    "calendar_and_strata_definition_manifest_sha256",
    "probabilistic_scenario_design_manifest_sha256",
    "fmv_profile_population_manifest_sha256",
    "hydro_dispatch_policy_manifest_sha256",
    "economic_valuation_and_fx_manifest_sha256",
    "fmv_risk_mde_approval_receipt_sha256",
    "origin_target_mask_inventory_sha256",
    "market_consistency_audit_manifest_sha256",
)

EVIDENCE_BLOCKERS = {
    "eex_quote_vintage_manifest_sha256": "ORIGIN_AVAILABLE_CH_EEX_VINTAGE_MANIFEST_ABSENT",
    "monthly_solver_configuration_manifest_sha256": (
        "MONTHLY_SOLVER_CONFIGURATION_MANIFEST_ABSENT"
    ),
    "candidate_baseline_identity_manifest_sha256": (
        "FROZEN_CANDIDATE_BASELINE_IDENTITY_MANIFEST_ABSENT"
    ),
    "direct_ch_hourly_truth_manifest_sha256": "DIRECT_CH_HOURLY_TRUTH_MANIFEST_ABSENT",
    "direct_ch_quarter_hour_truth_manifest_sha256": (
        "DIRECT_CH_NATIVE_QUARTER_HOUR_TRUTH_MANIFEST_ABSENT"
    ),
    "statistical_design_manifest_sha256": "EXACT_STATISTICAL_DESIGN_MANIFEST_ABSENT",
    "calendar_and_strata_definition_manifest_sha256": (
        "VERSIONED_CALENDAR_AND_STRATA_MANIFEST_ABSENT"
    ),
    "probabilistic_scenario_design_manifest_sha256": (
        "PROBABILISTIC_SCENARIO_DESIGN_MANIFEST_ABSENT"
    ),
    "fmv_profile_population_manifest_sha256": "EXACT_FMV_PROFILE_POPULATION_MANIFEST_ABSENT",
    "hydro_dispatch_policy_manifest_sha256": "HYDRO_DISPATCH_POLICY_MANIFEST_ABSENT",
    "economic_valuation_and_fx_manifest_sha256": (
        "ECONOMIC_VALUATION_AND_FX_CONVENTION_MANIFEST_ABSENT"
    ),
    "fmv_risk_mde_approval_receipt_sha256": "FMV_RISK_ECONOMIC_MDE_APPROVAL_ABSENT",
    "origin_target_mask_inventory_sha256": "EXACT_ORIGIN_TARGET_MASK_INVENTORY_ABSENT",
    "market_consistency_audit_manifest_sha256": (
        "POST_EVALUATION_MARKET_CONSISTENCY_AUDIT_MANIFEST_ABSENT"
    ),
}


class ChLtEstimandContractError(ValueError):
    """Raised when the structural estimand contract is malformed or weakened."""


@dataclass(frozen=True)
class EstimandAssessment:
    contract_id: str
    source_document_sha256: str | None
    canonical_semantic_sha256: str
    validator_policy_sha256: str
    status: str
    execution_authorized: bool
    scientific_admission: bool
    production_authorization: bool
    promotion_gate: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "contract_id": self.contract_id,
            "source_document_sha256": self.source_document_sha256,
            "canonical_semantic_sha256": self.canonical_semantic_sha256,
            "validator_policy_sha256": self.validator_policy_sha256,
            "status": self.status,
            "execution_authorized": self.execution_authorized,
            "scientific_admission": self.scientific_admission,
            "production_authorization": self.production_authorization,
            "promotion_gate": self.promotion_gate,
            "blockers": list(self.blockers),
        }


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def compute_contract_id(document: Mapping[str, object]) -> str:
    unsigned = dict(document)
    unsigned.pop("contract_id", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def compute_validator_policy_sha256() -> str:
    policy = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "top_level_fields": sorted(_TOP_LEVEL_FIELDS),
        "lifecycle_fields": sorted(_LIFECYCLE_FIELDS),
        "horizon_policy": EXPECTED_HORIZON_POLICY,
        "target_layers": EXPECTED_TARGET_LAYERS,
        "truth_policy": EXPECTED_TRUTH_POLICY,
        "origin_design_policy": EXPECTED_ORIGIN_DESIGN_POLICY,
        "eligibility_policy": EXPECTED_ELIGIBILITY_POLICY,
        "calendar_policy": EXPECTED_CALENDAR_POLICY,
        "metric_policy": EXPECTED_METRIC_POLICY,
        "statistical_decision_policy": EXPECTED_STATISTICAL_DECISION_POLICY,
        "probabilistic_policy": EXPECTED_PROBABILISTIC_POLICY,
        "profile_policy": EXPECTED_PROFILE_POLICY,
        "dispatch_policy": EXPECTED_DISPATCH_POLICY,
        "execution_controls": EXPECTED_EXECUTION_CONTROLS,
        "evidence_fields": list(EVIDENCE_FIELDS),
        "evidence_blockers": EVIDENCE_BLOCKERS,
    }
    return hashlib.sha256(canonical_json_bytes(policy)).hexdigest()


def load_estimand_contract(
    path: str | Path, *, expected_sha256: str | None = None
) -> tuple[dict[str, object], bytes]:
    lexical = Path(path).expanduser()
    if not lexical.is_absolute():
        lexical = Path(os.path.abspath(lexical))
    payload = read_stable_single_link_file(
        lexical,
        label="CH LT estimand contract",
        max_bytes=_MAX_DOCUMENT_BYTES,
    )
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, label="expected contract")
        if actual_sha256 != expected_sha256:
            raise ChLtEstimandContractError("estimand contract SHA-256 mismatch")
    try:
        parsed = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChLtEstimandContractError("estimand contract is invalid JSON") from exc
    document = _mapping(parsed, label="estimand contract")
    return dict(document), payload


def assess_estimand_contract(
    document: Mapping[str, object], *, document_bytes: bytes | None = None
) -> EstimandAssessment:
    if not isinstance(document, Mapping):
        raise ChLtEstimandContractError("estimand contract must be an object")
    if document_bytes is None:
        source_document_sha256 = None
    else:
        try:
            rebound = load_strict_json(document_bytes.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ChLtEstimandContractError("estimand contract bytes are invalid") from exc
        if not _strict_equal(rebound, document):
            raise ChLtEstimandContractError("estimand contract mapping/bytes mismatch")
        source_document_sha256 = hashlib.sha256(document_bytes).hexdigest()

    if set(document) != _TOP_LEVEL_FIELDS:
        raise ChLtEstimandContractError("top-level field inventory is invalid")
    _require_equal(document.get("schema_version"), SCHEMA_VERSION, "schema version")
    _require_equal(document.get("scope"), "FULL_DELIVERED_CH_LT_PFC", "scope")
    _require_equal(document.get("market"), "CH", "market")
    _require_equal(document.get("timezone"), "Europe/Zurich", "timezone")
    lifecycle = _mapping(document.get("lifecycle"), label="lifecycle")
    if set(lifecycle) != _LIFECYCLE_FIELDS:
        raise ChLtEstimandContractError("lifecycle field inventory is invalid")
    _require_equal(lifecycle.get("state"), DRAFT_STATE, "lifecycle state")
    _require_equal(lifecycle.get("frozen_at_utc"), None, "frozen_at_utc")

    _require_equal(document.get("horizon_policy"), EXPECTED_HORIZON_POLICY, "horizon policy")
    _require_equal(document.get("target_layers"), EXPECTED_TARGET_LAYERS, "target layers")
    _require_equal(document.get("truth_policy"), EXPECTED_TRUTH_POLICY, "truth policy")
    _require_equal(
        document.get("origin_design_policy"),
        EXPECTED_ORIGIN_DESIGN_POLICY,
        "origin design policy",
    )
    _require_equal(
        document.get("eligibility_policy"),
        EXPECTED_ELIGIBILITY_POLICY,
        "eligibility policy",
    )
    _require_equal(document.get("calendar_policy"), EXPECTED_CALENDAR_POLICY, "calendar policy")
    _require_equal(document.get("metric_policy"), EXPECTED_METRIC_POLICY, "metric policy")
    _require_equal(
        document.get("statistical_decision_policy"),
        EXPECTED_STATISTICAL_DECISION_POLICY,
        "statistical decision policy",
    )
    _require_equal(
        document.get("probabilistic_policy"),
        EXPECTED_PROBABILISTIC_POLICY,
        "probabilistic policy",
    )
    _require_equal(document.get("profile_policy"), EXPECTED_PROFILE_POLICY, "profile policy")
    _require_equal(document.get("dispatch_policy"), EXPECTED_DISPATCH_POLICY, "dispatch policy")
    _require_equal(
        document.get("execution_controls"),
        EXPECTED_EXECUTION_CONTROLS,
        "execution controls",
    )

    evidence = _mapping(document.get("evidence_bindings"), label="evidence bindings")
    if set(evidence) != set(EVIDENCE_FIELDS):
        raise ChLtEstimandContractError("evidence binding inventory is invalid")
    blockers: list[str] = []
    for field in EVIDENCE_FIELDS:
        value = evidence.get(field)
        if value is None:
            blockers.append(EVIDENCE_BLOCKERS[field])
        else:
            _require_sha256(value, label=field)
    blockers.append("EXTERNAL_ESTIMAND_ADMISSION_ENVELOPE_NOT_IMPLEMENTED")

    declared_id = document.get("contract_id")
    _require_sha256(declared_id, label="contract_id")
    computed_id = compute_contract_id(document)
    if declared_id != computed_id:
        raise ChLtEstimandContractError("estimand contract_id mismatch")

    return EstimandAssessment(
        contract_id=computed_id,
        source_document_sha256=source_document_sha256,
        canonical_semantic_sha256=hashlib.sha256(canonical_json_bytes(document)).hexdigest(),
        validator_policy_sha256=compute_validator_policy_sha256(),
        status=STATUS,
        execution_authorized=False,
        scientific_admission=False,
        production_authorization=False,
        promotion_gate=False,
        blockers=tuple(blockers),
    )


def _strict_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(
            _strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, float):
        return math.isfinite(left) and math.isfinite(right) and left == right
    return left == right


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ChLtEstimandContractError(f"{label} must be an object")
    return value


def _require_equal(actual: object, expected: object, label: str) -> None:
    if not _strict_equal(actual, expected):
        raise ChLtEstimandContractError(f"{label} does not match the structural contract")


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ChLtEstimandContractError(f"{label} SHA-256 is invalid")


__all__ = [
    "CONTRACT_ID",
    "CONTRACT_RELATIVE_PATH",
    "DOCUMENT_SHA256",
    "AUDIT_SCHEMA_VERSION",
    "ChLtEstimandContractError",
    "DRAFT_STATE",
    "EVIDENCE_BLOCKERS",
    "EXPECTED_DISPATCH_POLICY",
    "EXPECTED_CALENDAR_POLICY",
    "EXPECTED_ELIGIBILITY_POLICY",
    "EXPECTED_EXECUTION_CONTROLS",
    "EXPECTED_HORIZON_POLICY",
    "EXPECTED_METRIC_POLICY",
    "EXPECTED_ORIGIN_DESIGN_POLICY",
    "EXPECTED_PROBABILISTIC_POLICY",
    "EXPECTED_PROFILE_POLICY",
    "EXPECTED_STATISTICAL_DECISION_POLICY",
    "EXPECTED_TARGET_LAYERS",
    "EXPECTED_TRUTH_POLICY",
    "SCHEMA_VERSION",
    "STATUS",
    "assess_estimand_contract",
    "canonical_json_bytes",
    "compute_contract_id",
    "compute_validator_policy_sha256",
    "load_estimand_contract",
]
