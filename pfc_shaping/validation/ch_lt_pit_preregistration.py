"""Fail-closed preregistration contract for full Swiss LT PFC evaluation.

The contract is deliberately separate from the Tier-2 masked-EEX-quote
evaluation.  It governs a delivered hourly/quarter-hourly CH curve, including
deterministic shape, probabilistic marginals and multivariate scenarios.

A validated document never authorizes execution by itself. A separate future
external admission envelope must verify independent authorities and issue a
one-shot non-production capability. Neither object is a production authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd

from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.pipeline.strict_structured_data import load_strict_json

SCHEMA_VERSION = "ch_lt_pit_probabilistic_preregistration.v1"
AUDIT_SCHEMA_VERSION = "ch_lt_pit_preregistration_audit.v1"
PROTOCOL_NAME = "ch_lt_pit_outer24_future4_20260724_v1"

DRAFT_STATE = "DRAFT_NOT_FROZEN"
FROZEN_STATE = "FROZEN_STRUCTURE_UNVERIFIED_EXTERNAL_ADMISSION_REQUIRED"
DRAFT_STATUS = "DRAFT_BLOCKED_NOT_EXECUTABLE"
FROZEN_BLOCKED_STATUS = "FROZEN_STRUCTURE_EXTERNAL_ADMISSION_REQUIRED"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SEASONS = ("WINTER", "SPRING", "SUMMER", "AUTUMN")
REGIMES = (
    "HYDRO_LOW",
    "HYDRO_HIGH",
    "NEGATIVE_PRICE",
    "PRICE_SPIKE",
    "RENEWABLE_HIGH",
    "CROSS_BORDER_STRESS",
)
UNCONDITIONAL_DATA_ROLES = (
    "eex_forward_source",
    "eex_forwards_history",
    "epex_ch",
)
CONDITIONAL_DATA_ROLES = (
    "commodities",
    "entso",
    "epex_at",
    "epex_de",
    "epex_fr",
    "epex_it",
    "hydro",
    "outages",
)

EXPECTED_BENCHMARK_POLICY: dict[str, object] = {
    "common_origins_targets_and_eligibility_masks": True,
    "selection_nested_inside_each_outer_origin": True,
    "benchmarks": [
        "market_constrained_seasonal_naive.v1",
        "incumbent_ch_lt.v1",
        "robust_regularized_linear_gam.v1",
        "justified_nonlinear_tree_challenger.v1",
    ],
    "ompex_role": "DIAGNOSTIC_EXTERNAL_BENCHMARK_ONLY",
    "ompex_allowed_in_fit": False,
    "ompex_allowed_in_selection": False,
    "ompex_allowed_as_truth": False,
}

EXPECTED_DETERMINISTIC_POLICY: dict[str, object] = {
    "primary_metric": "MONTHLY_LEVEL_NEUTRALIZED_MAE_EUR_MWH",
    "secondary_metrics": [
        "MONTHLY_LEVEL_NEUTRALIZED_RMSE_EUR_MWH",
        "BIAS_EUR_MWH",
        "P95_ABSOLUTE_ERROR_EUR_MWH",
        "CAPTURE_FACTOR_ERROR",
        "BASE_PEAK_OFFPEAK_WEIGHTED_ERROR",
    ],
    "aggregation_order": "WITHIN_ORIGIN_VOLUME_WEIGHTED_THEN_EQUAL_ORIGIN_WEIGHT",
    "minimum_relative_mae_improvement": 0.02,
    "minimum_relative_rmse_improvement": 0.0,
    "minimum_positive_origin_share": 0.70,
    "maximum_mandatory_stratum_relative_degradation": 0.02,
    "comparison_tolerance": 1e-12,
    "final_eex_repricing_tolerance_eur_mwh": 1e-6,
    "monthly_solver_repricing_tolerance_eur_mwh": 1e-9,
    "monthly_level_preservation_tolerance_eur_mwh": 1e-9,
    "cascade_invariance_tolerance_eur_mwh": 1e-8,
    "quote_sensitivity_policy": {
        "support_weighted_gain_max": 2.0,
        "delivery_year_gain_max": 2.0,
        "cascade_bucket_gain_max": math.sqrt(5.0),
        "monthly_row_l1_gain_max": 3.0,
        "full_horizon_gain_is_gate": False,
    },
    "complexity_policy": {
        "effective_degrees_of_freedom_reported": True,
        "active_basis_dimension_reported": True,
        "regularization_selected_inside_origin": True,
        "maximum_condition_number": 1e9,
        "ablation_by_feature_group_required": True,
        "incremental_oos_value_required": True,
    },
}

EXPECTED_STATISTICAL_POLICY: dict[str, object] = {
    "inference_unit": "ORIGIN",
    "familywise_alpha": 0.05,
    "multiplicity": "HOLM_FOR_PRIMARY_CHALLENGER_COMPARISONS",
    "confidence_interval": "TWO_SIDED_95_PERCENT",
    "bootstrap": {
        "method": "CIRCULAR_MOVING_BLOCK_PERCENTILE",
        "unit": "ORIGIN",
        "block_length_origins": 3,
        "replicates": 10_000,
        "rng": "PCG64",
        "seed": 20_260_724,
    },
    "hac_cross_check": {
        "method": "NEWEY_WEST_HLN_CORRECTED_DIEBOLD_MARIANO",
        "lag_origins": 2,
        "primary_gate": False,
    },
    "aggregate_improvement_ci_lower_bound_strictly_positive": True,
    "ties_within_comparison_tolerance": "TIE_NOT_WIN",
    "missing_or_ineligible": "UNSUPPORTED_NEVER_IMPUTED_AS_ZERO_OR_WIN",
    "future_episode_rule": ("ALL_FOUR_EPISODES_NON_REGRESSING_ON_PRIMARY_METRIC_AND_COMBINED_PASS"),
}

EXPECTED_PROBABILISTIC_POLICY: dict[str, object] = {
    "quantile_grid": [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
    "scenario_count": 1_000,
    "scenario_rng": "PCG64",
    "scenario_seed": 20_260_724,
    "marginal_scores": [
        "PINBALL_BY_QUANTILE",
        "WIS",
        "CRPS",
        "EMPIRICAL_COVERAGE",
        "SHARPNESS",
        "PIT_OR_RANDOMIZED_RANK",
    ],
    "trajectory_scores": ["ENERGY_SCORE", "VARIOGRAM_SCORE"],
    "variogram": {
        "order": 0.5,
        "lags_quarter_hours": [1, 4, 16, 48, 96],
        "lag_weight": "INVERSE_LAG",
    },
    "transparent_baselines": [
        "EMPIRICAL_BLOCK_RESIDUAL_BOOTSTRAP",
        "REGULARIZED_QUANTILE_REGRESSION",
    ],
    "quantile_crossing_tolerance_eur_mwh": 0.0,
    "scenario_monthly_mean_tolerance_eur_mwh": 1e-9,
    "quantile_market_identity_policy": (
        "MARGINAL_QUANTILES_NOT_SEPARATELY_REPRICED_SCENARIOS_PROJECTED_PATHWISE"
    ),
    "pathwise_market_product_identity_required": True,
    "minimum_relative_wis_improvement": 0.02,
    "minimum_relative_crps_improvement": 0.02,
    "energy_score_no_regression": True,
    "variogram_score_no_regression": True,
    "calibration_required_by_horizon_and_regime": True,
}

EXPECTED_EXECUTION_CONTROLS: dict[str, object] = {
    "full_refit_per_origin": [
        "ACQUISITION_ADMISSION",
        "MONTHLY_SOLVER",
        "FEATURE_TRANSFORMS",
        "SHAPING_FIT",
        "COMPLEXITY_SELECTION",
        "PROBABILISTIC_CALIBRATION",
        "SCENARIO_GENERATION",
    ],
    "available_at_rule": "EVERY_CONSUMED_ROW_AVAILABLE_AT_LE_ORIGIN",
    "revision_policy": "LATEST_REVISION_AVAILABLE_AT_ORIGIN_ONLY",
    "future_data_injection_test_required": True,
    "same_file_alias_attack_test_required": True,
    "exact_origin_inventory_required": True,
    "one_shot_future_holdout_namespace": True,
    "candidate_grid_hash_bound_before_evaluation": True,
    "output_append_only": True,
    "production_authorization": False,
    "promotion_gate": False,
}

EXPECTED_ADMISSION_CONTROLS = [
    "EXACT_PROVIDER_RAW_BYTES",
    "SOURCE_SPECIFIC_REPLAY_EQUALITY",
    "AVAILABLE_AT_COMPLETE_AND_LE_ORIGIN",
    "INDEPENDENT_TRUSTED_TIME_RECEIPT",
    "INDEPENDENT_SIGNATURE",
    "BUILDER_INACCESSIBLE_IMMUTABLE_NAMESPACE",
    "EXTERNAL_CAS_RECEIPT_AND_FRESH_HEAD",
    "NO_LINK_REPARSE_OR_HARDLINK_ALIAS",
]

UNRESOLVED_EXECUTION_BOUNDARY_BLOCKERS = (
    "PLAN_CORE_RECEIPT_FREE_EXTERNAL_ADMISSION_ENVELOPE_NOT_IMPLEMENTED",
    "DEPENDENCE_POWER_MDE_AND_EFFECTIVE_SAMPLE_SIZE_STUDY_NOT_FROZEN",
    "EXACT_LT_HORIZON_TARGET_TRUTH_MASK_AND_INNER_FOLD_INVENTORIES_ABSENT",
    "DIRECT_CH_15MIN_TRUTH_AND_POST_EPISODE_OUTCOME_RECEIPTS_ABSENT",
    "FULL_STATISTICAL_PROBABILISTIC_HYPOTHESIS_FAMILY_AND_MC_ERROR_NOT_FROZEN",
    "FMV_PROFILE_ARTIFACT_FORMULA_AND_PER_PROFILE_NONREGRESSION_NOT_FROZEN",
    "DURABLE_ONE_SHOT_ATTEMPT_SEAL_AND_LEDGER_NOT_IMPLEMENTED",
    "RUNTIME_CPU_GPU_REPRODUCIBILITY_CONTRACT_NOT_BOUND",
    "PREREGISTRATION_V1_PATHWISE_SCENARIO_LEVEL_POLICY_SUPERSEDED",
)


class ChLtPitPreregistrationError(ValueError):
    """Raised when the preregistration document is structurally invalid."""


@dataclass(frozen=True)
class PreregistrationAssessment:
    plan_id: str
    document_sha256: str
    lifecycle_state: str
    status: str
    execution_authorized: bool
    production_authorization: bool
    promotion_gate: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "document_sha256": self.document_sha256,
            "lifecycle_state": self.lifecycle_state,
            "status": self.status,
            "execution_authorized": self.execution_authorized,
            "production_authorization": self.production_authorization,
            "promotion_gate": self.promotion_gate,
            "blockers": list(self.blockers),
        }


def canonical_json_bytes(value: object) -> bytes:
    """Serialize governance JSON deterministically without non-finite values."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def compute_plan_id(document: Mapping[str, object]) -> str:
    unsigned = dict(document)
    unsigned.pop("plan_id", None)
    return canonical_sha256(unsigned)


def load_preregistration(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, object], bytes]:
    lexical = Path(path).expanduser()
    if not lexical.is_absolute():
        lexical = Path(os.path.abspath(lexical))
    payload = read_stable_single_link_file(
        lexical,
        label="CH LT PIT preregistration",
        max_bytes=2_000_000,
    )
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, label="expected plan SHA-256")
        if digest != expected_sha256:
            raise ChLtPitPreregistrationError("plan bytes differ from expected SHA-256")
    try:
        parsed = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChLtPitPreregistrationError("plan JSON is invalid") from exc
    if not isinstance(parsed, dict):
        raise ChLtPitPreregistrationError("plan JSON must be an object")
    return parsed, payload


def assess_preregistration(
    document: Mapping[str, object],
    *,
    document_bytes: bytes | None = None,
) -> PreregistrationAssessment:
    """Validate the exact policy and report whether evaluation may start."""

    if document_bytes is not None:
        try:
            captured = load_strict_json(document_bytes.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ChLtPitPreregistrationError("assessment bytes are not valid strict JSON") from exc
        if not _strict_equal(captured, document):
            raise ChLtPitPreregistrationError(
                "assessment document differs from captured document bytes"
            )

    _require_exact_keys(
        document,
        {
            "schema_version",
            "plan_id",
            "protocol_name",
            "scope",
            "market",
            "timezone",
            "delivery_resolution_minutes",
            "production_authorization",
            "promotion_gate",
            "lifecycle",
            "data_admission",
            "origin_design",
            "benchmark_policy",
            "deterministic_policy",
            "statistical_policy",
            "probabilistic_policy",
            "economic_materiality",
            "execution_controls",
            "independent_review",
        },
        label="preregistration",
    )
    _require_equal(document.get("schema_version"), SCHEMA_VERSION, "schema version")
    _require_equal(document.get("protocol_name"), PROTOCOL_NAME, "protocol name")
    _require_equal(document.get("scope"), "FULL_DELIVERED_CH_LT_PFC", "scope")
    _require_equal(document.get("market"), "CH", "market")
    _require_equal(document.get("timezone"), "Europe/Zurich", "timezone")
    _require_equal(document.get("delivery_resolution_minutes"), 15, "resolution")
    _require_equal(document.get("production_authorization"), False, "production authority")
    _require_equal(document.get("promotion_gate"), False, "promotion gate")
    _require_equal(document.get("benchmark_policy"), EXPECTED_BENCHMARK_POLICY, "benchmark policy")
    _require_equal(
        document.get("deterministic_policy"),
        EXPECTED_DETERMINISTIC_POLICY,
        "deterministic policy",
    )
    _require_equal(
        document.get("statistical_policy"),
        EXPECTED_STATISTICAL_POLICY,
        "statistical policy",
    )
    _require_equal(
        document.get("probabilistic_policy"),
        EXPECTED_PROBABILISTIC_POLICY,
        "probabilistic policy",
    )
    _require_equal(
        document.get("execution_controls"),
        EXPECTED_EXECUTION_CONTROLS,
        "execution controls",
    )

    plan_id = _require_sha256(document.get("plan_id"), label="plan_id")
    if plan_id != compute_plan_id(document):
        raise ChLtPitPreregistrationError("plan_id does not bind canonical plan content")

    lifecycle = _mapping(document.get("lifecycle"), label="lifecycle")
    _require_exact_keys(
        lifecycle,
        {
            "state",
            "frozen_at_utc",
            "trusted_time_receipt_sha256",
            "governance_receipt_sha256",
            "candidate_grid_sha256",
        },
        label="lifecycle",
    )
    state = str(lifecycle.get("state"))
    if state not in {DRAFT_STATE, FROZEN_STATE}:
        raise ChLtPitPreregistrationError("lifecycle state is unsupported")

    blockers: list[str] = []
    if state == DRAFT_STATE:
        for field in (
            "frozen_at_utc",
            "trusted_time_receipt_sha256",
            "governance_receipt_sha256",
            "candidate_grid_sha256",
        ):
            if lifecycle.get(field) is not None:
                raise ChLtPitPreregistrationError(f"draft lifecycle field {field} must be null")
        blockers.append("PLAN_NOT_FROZEN_BY_INDEPENDENT_TRUSTED_TIME_AND_GOVERNANCE")
        frozen_at = None
    else:
        frozen_at = _utc_timestamp(lifecycle.get("frozen_at_utc"), label="frozen_at_utc")
        for field in (
            "trusted_time_receipt_sha256",
            "governance_receipt_sha256",
            "candidate_grid_sha256",
        ):
            _require_sha256(lifecycle.get(field), label=field)

    _validate_data_admission(
        document.get("data_admission"), blockers=blockers, frozen=state == FROZEN_STATE
    )
    _validate_origin_design(
        document.get("origin_design"),
        blockers=blockers,
        frozen_at=frozen_at,
        frozen=state == FROZEN_STATE,
    )
    _validate_economic_materiality(
        document.get("economic_materiality"),
        blockers=blockers,
        frozen=state == FROZEN_STATE,
    )
    _validate_independent_review(
        document.get("independent_review"),
        blockers=blockers,
        frozen=state == FROZEN_STATE,
    )

    blockers.extend(UNRESOLVED_EXECUTION_BOUNDARY_BLOCKERS)

    if state == FROZEN_STATE:
        blockers.extend(
            [
                "PATH_VERIFIED_PLAN_GOVERNANCE_AND_TRUSTED_TIME_ABSENT",
                "PATH_VERIFIED_DATA_SIGNATURE_CAS_HEAD_AND_PIT_REPLAY_ABSENT",
                "PATH_VERIFIED_RISK_AND_INDEPENDENT_REVIEW_RECEIPTS_ABSENT",
                "REGIME_ASSIGNMENT_AND_POWER_STUDY_REPLAY_NOT_VERIFIED",
            ]
        )
    # A document can describe receipt hashes, but cannot authenticate them.
    # Only a future path-based external admission envelope may issue a one-shot
    # evaluation capability after verifying exact bytes, signatures, authority
    # separation, trusted time, CAS/HEAD, runtime and outcome-truth contracts.
    execution_authorized = False
    status = FROZEN_BLOCKED_STATUS if state == FROZEN_STATE else DRAFT_STATUS
    raw = document_bytes if document_bytes is not None else canonical_json_bytes(document)
    return PreregistrationAssessment(
        plan_id=plan_id,
        document_sha256=hashlib.sha256(raw).hexdigest(),
        lifecycle_state=state,
        status=status,
        execution_authorized=execution_authorized,
        production_authorization=False,
        promotion_gate=False,
        blockers=tuple(blockers),
    )


def _validate_data_admission(value: object, *, blockers: list[str], frozen: bool) -> None:
    admission = _mapping(value, label="data_admission")
    _require_exact_keys(
        admission,
        {
            "admission_controls",
            "unconditional_roles",
            "conditional_roles_if_used",
            "current_evidence",
            "admitted_role_manifests",
            "current_evidence_admissible",
        },
        label="data admission",
    )
    _require_equal(
        admission.get("admission_controls"), EXPECTED_ADMISSION_CONTROLS, "admission controls"
    )
    _require_equal(
        admission.get("unconditional_roles"), list(UNCONDITIONAL_DATA_ROLES), "unconditional roles"
    )
    _require_equal(
        admission.get("conditional_roles_if_used"),
        list(CONDITIONAL_DATA_ROLES),
        "conditional roles",
    )
    _require_equal(
        admission.get("current_evidence_admissible"), False, "current evidence authority"
    )
    evidence = _sequence(admission.get("current_evidence"), label="current evidence")
    if not evidence:
        raise ChLtPitPreregistrationError("current evidence inventory is empty")
    for raw in evidence:
        item = _mapping(raw, label="current evidence item")
        _require_exact_keys(
            item,
            {
                "evidence_id",
                "role",
                "path",
                "sha256",
                "authority_status",
                "raw_resolution_minutes",
                "admissible_for_training",
                "admissible_for_selection",
                "admissible_for_holdout",
                "limitations",
            },
            label="current evidence item",
        )
        _require_sha256(item.get("sha256"), label="current evidence SHA-256")
        for field in (
            "admissible_for_training",
            "admissible_for_selection",
            "admissible_for_holdout",
        ):
            _require_equal(item.get(field), False, f"current evidence {field}")
        resolution = item.get("raw_resolution_minutes")
        if not isinstance(resolution, int) or isinstance(resolution, bool) or resolution <= 0:
            raise ChLtPitPreregistrationError("current evidence resolution is invalid")
        if not isinstance(item.get("limitations"), list) or not item["limitations"]:
            raise ChLtPitPreregistrationError("current evidence limitations are missing")

    manifests = _sequence(admission.get("admitted_role_manifests"), label="admitted role manifests")
    if not frozen:
        if manifests:
            raise ChLtPitPreregistrationError("draft cannot claim admitted role manifests")
        blockers.append("GOVERNED_PIT_DATA_ROLE_MANIFESTS_ABSENT")
        return
    by_role: dict[str, Mapping[str, object]] = {}
    manifest_hashes: set[str] = set()
    for raw in manifests:
        item = _mapping(raw, label="admitted role manifest")
        _require_exact_keys(
            item,
            {
                "role",
                "manifest_sha256",
                "trusted_time_receipt_sha256",
                "external_cas_receipt_sha256",
                "independent_signature_verified",
                "immutable_builder_inaccessible",
                "available_at_complete",
                "point_in_time_replay_pass",
            },
            label="admitted role manifest",
        )
        role = str(item.get("role"))
        if role in by_role:
            raise ChLtPitPreregistrationError("duplicate admitted data role")
        if role not in {*UNCONDITIONAL_DATA_ROLES, *CONDITIONAL_DATA_ROLES}:
            raise ChLtPitPreregistrationError("unknown admitted data role")
        for field in (
            "manifest_sha256",
            "trusted_time_receipt_sha256",
            "external_cas_receipt_sha256",
        ):
            _require_sha256(item.get(field), label=f"{role} {field}")
        manifest_sha256 = str(item["manifest_sha256"])
        if manifest_sha256 in manifest_hashes:
            raise ChLtPitPreregistrationError(
                "distinct admitted data roles cannot share one manifest hash"
            )
        manifest_hashes.add(manifest_sha256)
        for field in (
            "independent_signature_verified",
            "immutable_builder_inaccessible",
            "available_at_complete",
            "point_in_time_replay_pass",
        ):
            _require_equal(item.get(field), True, f"{role} {field}")
        by_role[role] = item
    missing = sorted(set(UNCONDITIONAL_DATA_ROLES) - set(by_role))
    if missing:
        blockers.append("GOVERNED_PIT_DATA_ROLE_MANIFESTS_INCOMPLETE:" + ",".join(missing))


def _validate_origin_design(
    value: object,
    *,
    blockers: list[str],
    frozen_at: pd.Timestamp | None,
    frozen: bool,
) -> None:
    design = _mapping(value, label="origin design")
    _require_exact_keys(
        design,
        {"retrospective_outer", "nested_selection", "future_holdout"},
        label="origin design",
    )
    outer = _mapping(design.get("retrospective_outer"), label="retrospective outer")
    _require_exact_keys(
        outer,
        {
            "required_origin_count",
            "cadence_months",
            "minimum_per_season",
            "minimum_per_regime",
            "regime_overlap_allowed",
            "exact_origins",
            "origin_inventory_sha256",
            "regime_definition_manifest_sha256",
        },
        label="retrospective outer",
    )
    _require_equal(outer.get("required_origin_count"), 24, "outer origin count")
    _require_equal(outer.get("cadence_months"), 1, "outer cadence")
    _require_equal(outer.get("minimum_per_season"), 6, "outer season minimum")
    _require_equal(
        outer.get("minimum_per_regime"),
        {regime: 4 for regime in REGIMES},
        "outer regime minima",
    )
    _require_equal(outer.get("regime_overlap_allowed"), True, "regime overlap")
    origins = _sequence(outer.get("exact_origins"), label="exact origins")
    if not frozen:
        if (
            origins
            or outer.get("origin_inventory_sha256") is not None
            or outer.get("regime_definition_manifest_sha256") is not None
        ):
            raise ChLtPitPreregistrationError("draft origin inventory must be empty and unhashed")
        blockers.append("EXACT_24_ORIGIN_INVENTORY_AND_REGIME_DEFINITIONS_ABSENT")
    else:
        _validate_outer_origins(origins)
        inventory_sha = _require_sha256(
            outer.get("origin_inventory_sha256"), label="origin inventory SHA-256"
        )
        if inventory_sha != canonical_sha256(origins):
            raise ChLtPitPreregistrationError("origin inventory SHA-256 mismatch")
        _require_sha256(
            outer.get("regime_definition_manifest_sha256"), label="regime definition SHA-256"
        )

    nested = _mapping(design.get("nested_selection"), label="nested selection")
    _require_equal(
        nested,
        {
            "selection_scope": "INSIDE_EACH_OUTER_ORIGIN_ONLY",
            "minimum_inner_origins": 12,
            "embargo_months": 1,
            "hyperparameters_refit_per_outer_origin": True,
            "feature_groups_selected_per_outer_origin": True,
            "future_holdout_visible_to_selection": False,
        },
        "nested selection",
    )

    future = _mapping(design.get("future_holdout"), label="future holdout")
    _require_exact_keys(
        future,
        {
            "required_episode_count",
            "episode_length_days",
            "one_episode_per_season",
            "exact_episodes",
            "episode_inventory_sha256",
            "strictly_post_freeze",
            "one_shot",
        },
        label="future holdout",
    )
    _require_equal(future.get("required_episode_count"), 4, "future episode count")
    _require_equal(future.get("episode_length_days"), 14, "future episode length")
    _require_equal(future.get("one_episode_per_season"), True, "future season rule")
    _require_equal(future.get("strictly_post_freeze"), True, "future freeze order")
    _require_equal(future.get("one_shot"), True, "future one-shot rule")
    episodes = _sequence(future.get("exact_episodes"), label="future episodes")
    if not frozen:
        if episodes or future.get("episode_inventory_sha256") is not None:
            raise ChLtPitPreregistrationError("draft future inventory must be empty and unhashed")
        blockers.append("EXACT_FOUR_SEASONAL_FUTURE_EPISODES_ABSENT")
    else:
        if frozen_at is None:
            raise ChLtPitPreregistrationError("frozen plan has no freeze time")
        _validate_future_episodes(episodes, frozen_at=frozen_at)
        inventory_sha = _require_sha256(
            future.get("episode_inventory_sha256"), label="episode inventory SHA-256"
        )
        if inventory_sha != canonical_sha256(episodes):
            raise ChLtPitPreregistrationError("future episode inventory SHA-256 mismatch")


def _validate_outer_origins(origins: Sequence[object]) -> None:
    if len(origins) != 24:
        raise ChLtPitPreregistrationError("frozen outer inventory must contain 24 origins")
    ids: set[str] = set()
    timestamps: list[pd.Timestamp] = []
    season_counts: Counter[str] = Counter()
    regime_counts: Counter[str] = Counter()
    for raw in origins:
        item = _mapping(raw, label="outer origin")
        _require_exact_keys(
            item, {"origin_id", "as_of_utc", "season", "regimes"}, label="outer origin"
        )
        origin_id = _required_text(item.get("origin_id"), label="origin_id")
        if origin_id in ids:
            raise ChLtPitPreregistrationError("duplicate outer origin_id")
        ids.add(origin_id)
        timestamp = _utc_timestamp(item.get("as_of_utc"), label="origin as_of_utc")
        timestamps.append(timestamp)
        season = str(item.get("season"))
        if season != _season_for_month(timestamp.tz_convert("Europe/Zurich").month):
            raise ChLtPitPreregistrationError("outer origin season does not match local month")
        season_counts[season] += 1
        regimes = item.get("regimes")
        if not isinstance(regimes, list) or not regimes or len(regimes) != len(set(regimes)):
            raise ChLtPitPreregistrationError("outer origin regimes are invalid")
        if any(regime not in REGIMES for regime in regimes):
            raise ChLtPitPreregistrationError("outer origin uses unknown regime")
        if {"HYDRO_LOW", "HYDRO_HIGH"}.issubset(regimes):
            raise ChLtPitPreregistrationError(
                "outer origin cannot be both HYDRO_LOW and HYDRO_HIGH"
            )
        regime_counts.update(regimes)
    if timestamps != sorted(timestamps) or len(set(timestamps)) != 24:
        raise ChLtPitPreregistrationError("outer origins must be unique and chronological")
    periods = [
        timestamp.tz_convert("Europe/Zurich").tz_localize(None).to_period("M")
        for timestamp in timestamps
    ]
    if any(right.ordinal - left.ordinal != 1 for left, right in zip(periods, periods[1:])):
        raise ChLtPitPreregistrationError("outer origins must have exact monthly cadence")
    if any(season_counts[season] < 6 for season in SEASONS):
        raise ChLtPitPreregistrationError("outer origins under-cover a season")
    if any(regime_counts[regime] < 4 for regime in REGIMES):
        raise ChLtPitPreregistrationError("outer origins under-cover a mandatory regime")


def _validate_future_episodes(episodes: Sequence[object], *, frozen_at: pd.Timestamp) -> None:
    if len(episodes) != 4:
        raise ChLtPitPreregistrationError("frozen future inventory must contain four episodes")
    seasons: set[str] = set()
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    ids: set[str] = set()
    for raw in episodes:
        item = _mapping(raw, label="future episode")
        _require_exact_keys(
            item, {"episode_id", "start_utc", "end_utc", "season"}, label="future episode"
        )
        episode_id = _required_text(item.get("episode_id"), label="episode_id")
        if episode_id in ids:
            raise ChLtPitPreregistrationError("duplicate future episode_id")
        ids.add(episode_id)
        start = _utc_timestamp(item.get("start_utc"), label="future start_utc")
        end = _utc_timestamp(item.get("end_utc"), label="future end_utc")
        if start <= frozen_at:
            raise ChLtPitPreregistrationError("future episode is not strictly post-freeze")
        if end - start != timedelta(days=14):
            raise ChLtPitPreregistrationError("future episode is not exactly 14 days")
        season = str(item.get("season"))
        local_start = start.tz_convert("Europe/Zurich")
        local_last_interval = (end - timedelta(minutes=15)).tz_convert("Europe/Zurich")
        if season != _season_for_month(local_start.month):
            raise ChLtPitPreregistrationError("future episode season does not match local start")
        if season != _season_for_month(local_last_interval.month):
            raise ChLtPitPreregistrationError("future episode crosses a season boundary")
        if any(
            value.second != 0 or value.microsecond != 0 or value.minute % 15 != 0
            for value in (start, end)
        ):
            raise ChLtPitPreregistrationError("future episode bounds must align to quarter-hours")
        if season in seasons:
            raise ChLtPitPreregistrationError("future episodes repeat a season")
        seasons.add(season)
        intervals.append((start, end))
    if seasons != set(SEASONS):
        raise ChLtPitPreregistrationError("future episodes do not cover all seasons")
    intervals.sort()
    if any(
        left_end > right_start for (_, left_end), (right_start, _) in zip(intervals, intervals[1:])
    ):
        raise ChLtPitPreregistrationError("future episodes overlap")


def _validate_economic_materiality(value: object, *, blockers: list[str], frozen: bool) -> None:
    policy = _mapping(value, label="economic materiality")
    _require_exact_keys(
        policy,
        {
            "metric",
            "population",
            "minimum_capture_value_improvement_eur_mwh",
            "transaction_and_imbalance_assumptions_sha256",
            "fmv_risk_approval_receipt_sha256",
            "status",
        },
        label="economic materiality",
    )
    _require_equal(policy.get("metric"), "PROFILE_CAPTURE_VALUE_ERROR_EUR_MWH", "economic metric")
    _require_equal(
        policy.get("population"),
        "FMV_PREDEFINED_FIL_ACC_AND_DISPATCH_PROFILES",
        "economic population",
    )
    if not frozen:
        if any(
            policy.get(field) is not None
            for field in (
                "minimum_capture_value_improvement_eur_mwh",
                "transaction_and_imbalance_assumptions_sha256",
                "fmv_risk_approval_receipt_sha256",
            )
        ):
            raise ChLtPitPreregistrationError("draft cannot claim an FMV economic threshold")
        _require_equal(
            policy.get("status"), "BLOCKED_REQUIRES_FMV_RISK_APPROVAL", "economic status"
        )
        blockers.append("FMV_CAPTURE_VALUE_MATERIALITY_THRESHOLD_NOT_RISK_APPROVED")
        return
    threshold = policy.get("minimum_capture_value_improvement_eur_mwh")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
        or float(threshold) <= 0.0
    ):
        raise ChLtPitPreregistrationError("frozen economic threshold must be finite and positive")
    _require_sha256(
        policy.get("transaction_and_imbalance_assumptions_sha256"),
        label="economic assumptions SHA-256",
    )
    _require_sha256(
        policy.get("fmv_risk_approval_receipt_sha256"), label="FMV risk approval SHA-256"
    )
    _require_equal(policy.get("status"), "RISK_APPROVED_FROZEN", "economic status")


def _validate_independent_review(value: object, *, blockers: list[str], frozen: bool) -> None:
    review = _mapping(value, label="independent review")
    _require_exact_keys(
        review,
        {
            "security_receipt_sha256",
            "it_operations_receipt_sha256",
            "quant_data_receipt_sha256",
            "required_no_open_severity",
        },
        label="independent review",
    )
    _require_equal(review.get("required_no_open_severity"), ["P0", "P1"], "review severity gate")
    fields = (
        "security_receipt_sha256",
        "it_operations_receipt_sha256",
        "quant_data_receipt_sha256",
    )
    if not frozen:
        if any(review.get(field) is not None for field in fields):
            raise ChLtPitPreregistrationError("draft cannot claim independent review receipts")
        blockers.append("INDEPENDENT_SECURITY_ITOPS_QUANT_REVIEW_RECEIPTS_ABSENT")
        return
    for field in fields:
        _require_sha256(review.get(field), label=field)


def _season_for_month(month: int) -> str:
    if month in {12, 1, 2}:
        return "WINTER"
    if month in {3, 4, 5}:
        return "SPRING"
    if month in {6, 7, 8}:
        return "SUMMER"
    return "AUTUMN"


def _utc_timestamp(value: object, *, label: str) -> pd.Timestamp:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ChLtPitPreregistrationError(f"{label} must be canonical UTC text")
    try:
        timestamp = pd.Timestamp(value)
    except ValueError as exc:
        raise ChLtPitPreregistrationError(f"{label} is invalid") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        raise ChLtPitPreregistrationError(f"{label} is not UTC")
    canonical = timestamp.isoformat().replace("+00:00", "Z")
    if canonical != value:
        raise ChLtPitPreregistrationError(f"{label} is not canonical")
    return timestamp


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ChLtPitPreregistrationError(f"{label} must be an object")
    return value


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ChLtPitPreregistrationError(f"{label} must be a list")
    return value


def _require_exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ChLtPitPreregistrationError(
            f"{label} keys differ: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _require_equal(actual: object, expected: object, label: str) -> None:
    if not _strict_equal(actual, expected):
        raise ChLtPitPreregistrationError(f"{label} differs from frozen policy")


def _strict_equal(actual: object, expected: object) -> bool:
    """Compare JSON-like values without Python's bool/int equivalence."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            return False
        return all(_strict_equal(actual[key], expected[key]) for key in expected)
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            return False
        return all(_strict_equal(left, right) for left, right in zip(actual, expected))
    return bool(actual == expected)


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ChLtPitPreregistrationError(f"{label} is not canonical SHA-256")
    return value


def _required_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ChLtPitPreregistrationError(f"{label} is invalid")
    return value


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "ChLtPitPreregistrationError",
    "DRAFT_STATE",
    "DRAFT_STATUS",
    "EXPECTED_ADMISSION_CONTROLS",
    "EXPECTED_BENCHMARK_POLICY",
    "EXPECTED_DETERMINISTIC_POLICY",
    "EXPECTED_EXECUTION_CONTROLS",
    "EXPECTED_PROBABILISTIC_POLICY",
    "EXPECTED_STATISTICAL_POLICY",
    "FROZEN_BLOCKED_STATUS",
    "FROZEN_STATE",
    "PROTOCOL_NAME",
    "PreregistrationAssessment",
    "REGIMES",
    "SCHEMA_VERSION",
    "SEASONS",
    "UNCONDITIONAL_DATA_ROLES",
    "UNRESOLVED_EXECUTION_BOUNDARY_BLOCKERS",
    "assess_preregistration",
    "canonical_json_bytes",
    "canonical_sha256",
    "compute_plan_id",
    "load_preregistration",
]
