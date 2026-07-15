"""Governed Tier-2 contract for monthly EEX masked-quote evaluation.

This module deliberately does not run a calibration or select a model.  It
defines the fail-closed evidence boundary that a later runner must satisfy.
Monthly EEX vintages can support only a monthly masked-quote claim; they cannot
establish the quality of the delivered hourly PFC.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from pfc_shaping.calibration.cascading import count_hours
from pfc_shaping.calibration.monthly_forward_curve import product_periods
from pfc_shaping.pipeline.model_governance_contract import (
    MODEL_GOVERNANCE_RECEIPT_SCHEMA,
    MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV,
)


PLAN_SCHEMA = "tier2_monthly_eex_evaluation_plan.v1"
SELECTION_INPUT_PLAN_SCHEMA = "tier2_monthly_eex_evaluation_plan.v2"
FOLD_SCHEMA = "tier2_monthly_eex_fold_result.v1"
CAMPAIGN_SCHEMA = "tier2_monthly_eex_campaign.v1"
CANDIDATE_FREEZE_SCHEMA = "tier2_candidate_freeze.v1"
INDEX_SCHEMA = "tier2_monthly_eex_evaluation_index.v1"
TIMESTAMP_RECEIPT_SCHEMA = "tier2_artifact_timestamp_receipt.v1"
EXECUTION_ATTESTATION_SCHEMA = "tier2_execution_attestation.v1"

SCOPE = "MONTHLY_EEX_MASKED_QUOTE_ONLY"
FULL_TIER2_STATUS = "UNSUPPORTED_REQUIRES_HOURLY_PIT_TRUTH"
PASS_STATUS = "PASS_MONTHLY_EEX_MASKED_QUOTE_ONLY"
SELECTION_PASS_STATUS = "PASS_SELECTION_COMPLETE_CANDIDATE_FREEZE_ELIGIBLE"
LOCAL_TIMEZONE = "Europe/Zurich"
MARKET = "CH"
SIGNATURE_ALGORITHM = "Ed25519"

TIMESTAMP_PUBLIC_KEY_ENV = "PFC_TIER2_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH"
TIMESTAMP_JOURNAL_ID_ENV = "PFC_TIER2_TIMESTAMP_JOURNAL_ID"
EXECUTION_PUBLIC_KEY_ENV = "PFC_TIER2_EXECUTION_TRUSTED_PUBLIC_KEY_PATH"

MIN_SELECTION_ORIGINS = 36
MIN_SELECTION_MONTHS = 36
MIN_HOLDOUT_ORIGINS = 24
MIN_HOLDOUT_MONTHS = 12
MIN_BUCKET_ORIGINS = 12
MIN_BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_713
MIN_RELATIVE_IMPROVEMENT = 0.02
MIN_POSITIVE_FOLD_SHARE = 0.70
MAX_BUCKET_DEGRADATION = 0.02
BASELINE_ZERO_TOLERANCE = 1e-12
RANK_ATOL = 1e-12
RANK_RTOL = 1e-10
MAX_CONDITION_NUMBER = 1e9
_VERIFIED_PLAN_TOKEN = object()

SELECTION_INPUT_RESOURCE_PROFILE_SCHEMA = (
    "tier2_monthly_eex_selection_input_resource_profile.v1"
)
EXPECTED_SELECTION_INPUT_RESOURCE_PROFILE = {
    "schema_version": SELECTION_INPUT_RESOURCE_PROFILE_SCHEMA,
    "max_selection_folds": 512,
    "max_verified_frame_rows": 2_000_000,
    "max_fold_row_work": 100_000_000,
    "max_delivery_months_per_fold": 120,
    "max_total_emitted_row_hashes": 5_000_000,
    "max_selection_input_manifest_bytes": 4_000_000,
    "max_selection_input_payload_bytes": 512_000_000,
    "max_eex_catalog_bytes": 64_000_000,
    "max_eex_history_bytes": 4_000_000_000,
    "max_eex_internal_file_bytes": 1_000_000_000,
    "max_evidence_envelope_bytes": 4_000_000,
    "max_source_closure_bytes": 16_000_000,
    "max_runtime_lock_bytes": 128_000_000,
    "max_timestamp_journal_entries": 10_000,
}

MANDATORY_TENOR_HORIZON_BUCKETS = tuple(
    f"{tenor}_{horizon}"
    for tenor in ("MONTH", "QUARTER")
    for horizon in ("H0", "H1", "H2", "H3_PLUS")
)

FOLD_EVALUATED = "EVALUATED"
FOLD_FAIL_STATUSES = {
    "FAIL_PROVENANCE",
    "FAIL_CONTRACT",
    "FAIL_REPRICING",
}
FOLD_ERROR_STATUSES = {"ERROR_EXECUTION"}
FOLD_UNSUPPORTED_STATUSES = {
    "UNSUPPORTED_INSUFFICIENT_DATA",
    "UNSUPPORTED_TARGET_SPANNED_BY_RETAINED_QUOTES",
    "UNSUPPORTED_NUMERICAL_IDENTIFIABILITY",
}
FOLD_STATUSES = (
    {FOLD_EVALUATED}
    | FOLD_FAIL_STATUSES
    | FOLD_ERROR_STATUSES
    | FOLD_UNSUPPORTED_STATUSES
)

_EXPECTED_BASELINE = {
    "baseline_id": "market_constrained_seasonal.v1",
    "same_retained_quotes_and_hard_constraints": True,
    "calendar": "EEX_PEAKLOAD_WEEKDAY_08_20_HOLIDAYS_INCLUDED",
    "timezone": LOCAL_TIMEZONE,
    "shape_source": "CH_ONLY_PIT_SEASONAL_DEVIATIONS",
    "shape_energy_weighted_mean": 0.0,
    "neighbor_inputs_allowed": False,
    "proxy_inputs_allowed": False,
    "ompex_inputs_allowed": False,
    "estimator": "FIXED_ROBUST_MONTH_OF_YEAR_MEDIAN",
    "lookback_months": 72,
    "embargo_months": 1,
    "unavailable_status": "UNSUPPORTED_BASELINE",
}

_EXPECTED_COVERAGE = {
    "selection_min_effective_origins": MIN_SELECTION_ORIGINS,
    "selection_min_distinct_origin_months": MIN_SELECTION_MONTHS,
    "selection_requires_all_calendar_months": True,
    "holdout_min_effective_origins": MIN_HOLDOUT_ORIGINS,
    "holdout_min_distinct_origin_months": MIN_HOLDOUT_MONTHS,
    "mandatory_bucket_min_effective_origins": MIN_BUCKET_ORIGINS,
    "insufficient_coverage_status": "UNSUPPORTED",
}

_EXPECTED_PERFORMANCE = {
    "aggregation_order": "TARGET_WEIGHTED_WITHIN_ORIGIN_THEN_EQUAL_ORIGIN_WEIGHT",
    "population": "EXACT_COMPLETE_CASE_BASELINE_CANDIDATE",
    "holdout_only_for_promotion": True,
    "mae_min_relative_improvement": MIN_RELATIVE_IMPROVEMENT,
    "rmse_min_relative_improvement": MIN_RELATIVE_IMPROVEMENT,
    "min_positive_mae_origin_share": MIN_POSITIVE_FOLD_SHARE,
    "aggregate_improvement_ci_level": 0.95,
    "aggregate_improvement_ci_lower_bound_strictly_positive": True,
    "bootstrap_statistic": "ONE_MINUS_RATIO_OF_RESAMPLED_ORIGIN_MEANS",
    "bootstrap_metrics": ["MAE", "RMSE"],
    "mandatory_bucket_max_relative_degradation": MAX_BUCKET_DEGRADATION,
    "mandatory_bucket_metrics": ["MAE", "RMSE"],
    "repricing_tolerance": 1e-9,
    "conservation_tolerance": 1e-9,
    "baseline_zero_policy": "UNSUPPORTED_IF_BASELINE_METRIC_LE_1E-12",
    "comparison_tolerance": 1e-12,
    "rounding": "NONE_FLOAT64",
}

_EXPECTED_BOOTSTRAP = {
    "unit": "ORIGIN",
    "method": "CIRCULAR_MOVING_BLOCK_PERCENTILE",
    "rng": "PCG64",
    "seed": BOOTSTRAP_SEED,
    "replicates": MIN_BOOTSTRAP_REPLICATES,
    "confidence_level": 0.95,
    "edge_policy": "CIRCULAR",
    "block_length_selected_on": "SELECTION_ONLY",
    "block_length_binding": "CANDIDATE_FREEZE",
}

_EXPECTED_IDENTIFIABILITY = {
    "basis": "DELIVERY_MONTH_X_PEAK_OFFPEAK_SEGMENT",
    "dtype": "float64",
    "month_order": "CHRONOLOGICAL",
    "timezone": LOCAL_TIMEZONE,
    "peak_calendar": "WEEKDAY_08_20_PUBLIC_HOLIDAYS_INCLUDED",
    "dst_policy": "UTC_HOURS_MAPPED_TO_LOCAL_EEX_CALENDAR",
    "constraint_normalization": "L2",
    "rank_method": "SVD",
    "rank_atol": RANK_ATOL,
    "rank_rtol_times_smax": RANK_RTOL,
    "max_condition_number": MAX_CONDITION_NUMBER,
    "condition_number_spectrum": "NONZERO_EFFECTIVE_RANK_ONLY",
    "spanned_target_status": "UNSUPPORTED_TARGET_SPANNED_BY_RETAINED_QUOTES",
    "ill_conditioned_status": "UNSUPPORTED_NUMERICAL_IDENTIFIABILITY",
}

_EXPECTED_CANDIDATE_GRID_CONTRACT = {
    "schema_version": "tier2_monthly_eex_candidate_grid.v1",
    "artifact_role": "tier2_monthly_eex_candidate_grid",
    "market": MARKET,
    "timezone": LOCAL_TIMEZONE,
    "top_level_fields": [
        "schema_version",
        "grid_id",
        "market",
        "timezone",
        "configs",
    ],
    "config_entry_fields": ["config_sha256", "config"],
    "config_hash_formula": "SHA256_CANONICAL_JSON_CONFIG_V1",
    "grid_id_formula": "SHA256_CANONICAL_JSON_UNSIGNED_GRID_V1",
    "exact_allowlist_membership_required": True,
    "grid_bytes_must_match_plan_sha256": True,
    "config_order": "CONFIG_SHA256_ASCENDING_UNIQUE",
    "delivery_months_policy": "CONTIGUOUS_MIN_TO_MAX_FULL_PRODUCTS_PER_FOLD",
}

_EXPECTED_FOLD_SOURCE_POLICY = {
    "source": "EEX_ONLY_SIGNED_IMMUTABLE_VINTAGE_CATALOG",
    "catalog_root_hash_formula": (
        "SHA256_CANONICAL_JSON_V1_CATALOG_ID_CATALOG_SHA256_HISTORY_SHA256"
    ),
    "snapshot_selection": (
        "LATEST_UNIQUE_CH_SNAPSHOT_AVAILABLE_AT_OR_BEFORE_FOLD_AS_OF"
    ),
    "snapshot_tie_status": "UNSUPPORTED_SNAPSHOT_OR_TARGET_UNAVAILABLE",
    "target_selection": "EXACT_PLAN_PRODUCT_LOAD_TYPE_IN_SELECTED_SNAPSHOT",
    "revision_resolution": "LATEST_AVAILABLE_PER_DATE_MARKET_LOAD_PRODUCT",
    "revision_fallback_allowed": False,
    "retained_policy": (
        "SELECTED_CH_SNAPSHOT_COMPLEMENT_AFTER_REMOVE_ALL_DELIVERY_OVERLAPS"
    ),
    "revealing_overlap_policy": "ANY_DELIVERY_MONTH_INTERSECTION_ALL_LOAD_TYPES",
    "outside_grid_policy": "EXCLUDED_AND_INVENTORIED",
    "later_quote_policy": "DIAGNOSTIC_EXCLUDED_NEVER_CONSUMED_OR_SCORED",
    "diagnostic_inventory_predicate": (
        "ALL_CATALOG_ROWS_CH_TARGET_LOAD_PRODUCT_AVAILABLE_AT_GT_FOLD_AS_OF"
    ),
}

_EXPECTED_FOLD_CONTEXT_POLICY = {
    "selection_training_cutoff": "MIN_PLAN_TRUSTED_TIME_AND_FOLD_AS_OF",
    "fold_history_cutoff": "FOLD_AS_OF",
    "pit_time_axis": "available_at",
    "window_time_axis": "date",
    "window_calendar_timezone": LOCAL_TIMEZONE,
    "window_anchor_formula": (
        "LOCAL_MONTH_START_OF_CUTOFF_MINUS_DATEOFFSET_EMBARGO_MONTHS"
    ),
    "membership_formula": (
        "AVAILABLE_AT_LE_CUTOFF_AND_LOWER_DATE_LE_DATE_LT_UPPER_DATE"
    ),
    "lookback_months": 72,
    "embargo_months": 1,
    "lower_bound_formula": "UPPER_DATE_MINUS_DATEOFFSET_LOOKBACK_MONTHS",
    "lower_bound": "INCLUSIVE",
    "upper_bound": "EXCLUSIVE",
    "revision_resolution": "LATEST_AVAILABLE_PER_DATE_MARKET_LOAD_PRODUCT",
    "revision_order": "REVISION_SEQUENCE_THEN_AVAILABLE_AT_THEN_QUOTE_ID",
    "inventory_closure": "EXACT_ALL_ELIGIBLE_ROWS_AFTER_REVISION_RESOLUTION",
    "evaluation_snapshot_excluded_from_contexts": True,
    "markets": ["CH", "DE", "FR", "AT", "IT"],
    "load_types": ["BASE", "PEAK", "OFFPEAK"],
    "fallback_allowed": False,
}

_EXPECTED_FOLD_WEIGHT_POLICY = {
    "formula": "EEX_PRODUCT_DELIVERY_HOURS",
    "base": "ALL_UTC_HOURS_IN_LOCAL_DELIVERY_PERIOD",
    "peak": "WEEKDAY_LOCAL_08_20_PUBLIC_HOLIDAYS_INCLUDED",
    "offpeak": "BASE_HOURS_MINUS_PEAK_HOURS",
    "timezone": LOCAL_TIMEZONE,
    "dtype": "float64",
}

_EXPECTED_FOLD_CLAIM_POLICY = {
    "allowed_status": "DATA_LINEAGE_REPLAYED_MODEL_OUTPUT_UNVERIFIED",
    "data_lineage_verified": True,
    "model_output_verified": False,
    "campaign_eligible": False,
    "production_approved": False,
    "holdout_status_without_verified_freeze": (
        "UNSUPPORTED_HOLDOUT_REQUIRES_VERIFIED_FREEZE"
    ),
}


class Tier2MonthlyEexEvaluationError(ValueError):
    """Raised when monthly EEX evaluation evidence is not promotion-grade."""


@dataclass(frozen=True)
class ProductConstraint:
    product: str
    load_type: str


@dataclass(frozen=True)
class IdentifiabilityAssessment:
    delivery_months: tuple[str, ...]
    retained_rank: int
    augmented_rank: int
    retained_condition_number: float
    augmented_condition_number: float
    status: str


@dataclass(frozen=True)
class VerifiedArtifactAuthority:
    artifact_sha256: str
    artifact_role: str
    artifact_schema_version: str
    envelope_id: str
    authority_key_id: str
    trusted_key_path: str
    trusted_key_sha256: str
    journal_id: str | None
    event_time_utc: pd.Timestamp


@dataclass(frozen=True)
class VerifiedEvaluationPlan:
    plan_id: str
    artifact_sha256: str
    governance_key_id: str
    timestamp_key_id: str
    trusted_time_utc: pd.Timestamp
    governance_receipt_sha256: str
    timestamp_receipt_sha256: str
    payload: Mapping[str, object]
    _verification_token: InitVar[object | None] = None

    def __post_init__(self, _verification_token: object | None) -> None:
        if _verification_token is not _VERIFIED_PLAN_TOKEN:
            raise Tier2MonthlyEexEvaluationError(
                "VerifiedEvaluationPlan can only be created by path-based verification"
            )


def product_constraint_vector(
    constraint: ProductConstraint,
    *,
    delivery_months: Sequence[str | pd.Period],
) -> np.ndarray:
    """Build an EEX average-price row on month x PEAK/OFFPEAK atoms."""

    months = pd.PeriodIndex(delivery_months, freq="M").sort_values().unique()
    covered = product_periods(constraint.product)
    if not set(covered).issubset(set(months)):
        raise Tier2MonthlyEexEvaluationError(
            f"delivery grid does not fully cover {constraint.product}"
        )
    load_type = str(constraint.load_type).upper()
    if load_type not in {"BASE", "PEAK", "OFFPEAK"}:
        raise Tier2MonthlyEexEvaluationError(
            f"unsupported EEX load type: {constraint.load_type}"
        )
    vector = np.zeros(len(months) * 2, dtype=np.float64)
    denominator = 0.0
    counts: dict[pd.Period, tuple[float, float]] = {}
    for month in covered:
        _, peak_hours, offpeak_hours = count_hours(
            int(month.year),
            int(month.month),
            int(month.month),
            tz=LOCAL_TIMEZONE,
            country=MARKET,
        )
        peak = float(peak_hours)
        offpeak = float(offpeak_hours)
        counts[month] = (peak, offpeak)
        if load_type == "BASE":
            denominator += peak + offpeak
        elif load_type == "PEAK":
            denominator += peak
        else:
            denominator += offpeak
    if denominator <= 0.0:
        raise Tier2MonthlyEexEvaluationError("EEX product has no delivery hours")
    positions = {month: index for index, month in enumerate(months)}
    for month, (peak, offpeak) in counts.items():
        offset = positions[month] * 2
        if load_type in {"BASE", "PEAK"}:
            vector[offset] = peak / denominator
        if load_type in {"BASE", "OFFPEAK"}:
            vector[offset + 1] = offpeak / denominator
    return vector


def assess_target_identifiability(
    *,
    target: ProductConstraint,
    retained: Sequence[ProductConstraint],
) -> IdentifiabilityAssessment:
    """Assess whether a masked target adds independent economic information."""

    products = [target, *retained]
    months = sorted(
        {month for item in products for month in product_periods(item.product)}
    )
    month_labels = tuple(str(month) for month in months)
    target_row = _l2_normalize(
        product_constraint_vector(target, delivery_months=months)
    )
    retained_rows = _deduplicate_rows(
        [
            _l2_normalize(product_constraint_vector(item, delivery_months=months))
            for item in retained
        ]
    )
    retained_matrix = (
        np.vstack(retained_rows)
        if retained_rows
        else np.empty((0, len(months) * 2), dtype=np.float64)
    )
    augmented = np.vstack([retained_matrix, target_row])
    retained_rank, condition = _effective_rank_and_condition(retained_matrix)
    augmented_rank, augmented_condition = _effective_rank_and_condition(augmented)
    if max(condition, augmented_condition) > MAX_CONDITION_NUMBER:
        status = "UNSUPPORTED_NUMERICAL_IDENTIFIABILITY"
    elif augmented_rank == retained_rank:
        status = "UNSUPPORTED_TARGET_SPANNED_BY_RETAINED_QUOTES"
    else:
        status = "IDENTIFIABLE"
    return IdentifiabilityAssessment(
        delivery_months=month_labels,
        retained_rank=retained_rank,
        augmented_rank=augmented_rank,
        retained_condition_number=condition,
        augmented_condition_number=augmented_condition,
        status=status,
    )


def _validate_evaluation_plan(plan: Mapping[str, object]) -> dict[str, object]:
    schema = plan.get("schema_version")
    required = {
        "schema_version",
        "plan_id",
        "scope",
        "full_tier2_pfc_status",
        "production_approved",
        "market",
        "timezone",
        "estimand",
        "baseline",
        "secondary_challengers",
        "coverage",
        "performance",
        "bootstrap",
        "identifiability",
        "status_precedence",
        "candidate_grid_sha256",
        "candidate_grid_contract",
        "candidate_config_sha256_allowlist",
        "selection_rule",
        "selection_eex_root",
        "fold_source_policy",
        "fold_context_policy",
        "fold_weight_policy",
        "fold_claim_policy",
        "evaluation_source_closure_sha256",
        "runtime_lock_sha256",
        "mandatory_tenor_horizon_buckets",
        "selection_folds",
        "holdout_folds",
    }
    if schema == SELECTION_INPUT_PLAN_SCHEMA:
        required.add("selection_input_resource_profile")
    _require_exact_keys(plan, required, label="evaluation plan")
    _require_common_scope(plan)
    if schema not in {PLAN_SCHEMA, SELECTION_INPUT_PLAN_SCHEMA}:
        raise Tier2MonthlyEexEvaluationError("plan schema is unsupported")
    if schema == SELECTION_INPUT_PLAN_SCHEMA:
        _require_equal(
            plan.get("selection_input_resource_profile"),
            EXPECTED_SELECTION_INPUT_RESOURCE_PROFILE,
            label="selection input resource profile",
        )
    _require_equal(plan.get("market"), MARKET, label="plan market")
    _require_equal(plan.get("timezone"), LOCAL_TIMEZONE, label="plan timezone")
    _require_equal(plan.get("baseline"), _EXPECTED_BASELINE, label="primary baseline")
    _require_equal(plan.get("coverage"), _EXPECTED_COVERAGE, label="coverage contract")
    _require_equal(plan.get("performance"), _EXPECTED_PERFORMANCE, label="performance contract")
    _require_equal(plan.get("bootstrap"), _EXPECTED_BOOTSTRAP, label="bootstrap contract")
    _require_equal(
        plan.get("identifiability"),
        _EXPECTED_IDENTIFIABILITY,
        label="identifiability contract",
    )
    _require_equal(
        plan.get("status_precedence"),
        ["ERROR", "FAIL", "UNSUPPORTED", PASS_STATUS],
        label="status precedence",
    )
    for field in (
        "candidate_grid_sha256",
        "evaluation_source_closure_sha256",
        "runtime_lock_sha256",
    ):
        _require_sha256(plan.get(field), label=field)
    _require_equal(
        plan.get("candidate_grid_contract"),
        _EXPECTED_CANDIDATE_GRID_CONTRACT,
        label="candidate grid contract",
    )
    config_allowlist = plan.get("candidate_config_sha256_allowlist")
    if not isinstance(config_allowlist, list) or not config_allowlist:
        raise Tier2MonthlyEexEvaluationError(
            "candidate config SHA-256 allowlist must be non-empty"
        )
    normalized_configs = [
        _require_sha256(value, label="candidate config SHA-256")
        for value in config_allowlist
    ]
    if normalized_configs != sorted(set(normalized_configs)):
        raise Tier2MonthlyEexEvaluationError(
            "candidate config SHA-256 allowlist must be sorted and unique"
        )
    selection_root = _mapping(plan.get("selection_eex_root"), label="selection EEX root")
    _require_exact_keys(
        selection_root,
        {"catalog_id", "catalog_sha256", "history_sha256", "root_sha256"},
        label="selection EEX root",
    )
    for field in ("catalog_id", "catalog_sha256", "history_sha256"):
        _require_sha256(selection_root.get(field), label=f"selection EEX {field}")
    expected_root_hash = _sha256_json(
        {
            "schema_version": "tier2_monthly_eex_catalog_root.v1",
            "catalog_id": selection_root["catalog_id"],
            "catalog_sha256": selection_root["catalog_sha256"],
            "history_sha256": selection_root["history_sha256"],
        }
    )
    _require_equal(
        selection_root.get("root_sha256"),
        expected_root_hash,
        label="selection EEX root hash",
    )
    _require_equal(
        plan.get("fold_source_policy"),
        _EXPECTED_FOLD_SOURCE_POLICY,
        label="fold source policy",
    )
    _require_equal(
        plan.get("fold_context_policy"),
        _EXPECTED_FOLD_CONTEXT_POLICY,
        label="fold context policy",
    )
    _require_equal(
        plan.get("fold_weight_policy"),
        _EXPECTED_FOLD_WEIGHT_POLICY,
        label="fold weight policy",
    )
    _require_equal(
        plan.get("fold_claim_policy"),
        _EXPECTED_FOLD_CLAIM_POLICY,
        label="fold claim policy",
    )
    _require_equal(
        plan.get("selection_rule"),
        "MIN_SELECTION_MAE_THEN_RMSE_LEXICOGRAPHIC_FIXED_GRID",
        label="selection rule",
    )
    _require_equal(
        plan.get("mandatory_tenor_horizon_buckets"),
        list(MANDATORY_TENOR_HORIZON_BUCKETS),
        label="mandatory tenor-horizon buckets",
    )
    estimand = _mapping(plan.get("estimand"), label="estimand")
    _require_equal(
        estimand,
        {
            "unit": "ONE_ECONOMIC_SAME_VINTAGE_EEX_QUOTE",
            "target_removed_from_all_inputs": True,
            "algebraically_revealing_quotes_removed": True,
            "retained_hard_quotes_repriced": True,
            "later_quotes_namespace": "DIAGNOSTIC_EXCLUDED",
            "allowed_sources": ["EEX"],
            "forbidden_sources": ["OMPEX", "PROXY", "BENCHMARK"],
        },
        label="estimand",
    )
    _require_equal(
        plan.get("secondary_challengers"),
        ["parent_flat.v1", "last_visible_quote.v1"],
        label="secondary challengers",
    )
    selection = _validate_fold_specs(plan.get("selection_folds"), campaign="SELECTION")
    holdout = _validate_fold_specs(plan.get("holdout_folds"), campaign="HOLDOUT")
    if set(selection).intersection(holdout):
        raise Tier2MonthlyEexEvaluationError("selection and holdout fold IDs overlap")
    _validate_preregistered_coverage(plan["selection_folds"], campaign="SELECTION")  # type: ignore[arg-type]
    _validate_preregistered_coverage(plan["holdout_folds"], campaign="HOLDOUT")  # type: ignore[arg-type]
    selection_times = {
        str(item["fold_as_of_utc"]) for item in plan["selection_folds"]  # type: ignore[index]
    }
    holdout_times = {
        str(item["fold_as_of_utc"]) for item in plan["holdout_folds"]  # type: ignore[index]
    }
    if selection_times.intersection(holdout_times):
        raise Tier2MonthlyEexEvaluationError("selection and holdout origin times overlap")
    unsigned = dict(plan)
    plan_id = str(unsigned.pop("plan_id", ""))
    if plan_id != _sha256_json(unsigned):
        raise Tier2MonthlyEexEvaluationError("plan_id mismatch")
    return dict(plan)


def validate_candidate_freeze(freeze: Mapping[str, object]) -> dict[str, object]:
    _raise_downstream_replay_unavailable("candidate freeze")
    required = {
        "schema_version",
        "freeze_id",
        "scope",
        "full_tier2_pfc_status",
        "production_approved",
        "plan_sha256",
        "selection_campaign_sha256",
        "selected_config_sha256",
        "source_closure_sha256",
        "runtime_fingerprint_sha256",
        "bootstrap_block_length_origins",
        "frozen_at_utc",
    }
    _require_exact_keys(freeze, required, label="candidate freeze")
    _require_common_scope(freeze)
    _require_equal(freeze.get("schema_version"), CANDIDATE_FREEZE_SCHEMA, label="freeze schema")
    for field in (
        "plan_sha256",
        "selection_campaign_sha256",
        "selected_config_sha256",
        "source_closure_sha256",
        "runtime_fingerprint_sha256",
    ):
        _require_sha256(freeze.get(field), label=field)
    block_length = _positive_int(
        freeze.get("bootstrap_block_length_origins"),
        label="bootstrap block length",
    )
    if block_length > MIN_SELECTION_ORIGINS:
        raise Tier2MonthlyEexEvaluationError(
            "bootstrap block length exceeds minimum selection history"
        )
    _aware_utc(freeze.get("frozen_at_utc"), label="frozen_at_utc")
    unsigned = dict(freeze)
    freeze_id = str(unsigned.pop("freeze_id", ""))
    if freeze_id != _sha256_json(unsigned):
        raise Tier2MonthlyEexEvaluationError("freeze_id mismatch")
    return dict(freeze)


def validate_fold_result(
    fold: Mapping[str, object],
    *,
    campaign_type: str,
    freeze_time_utc: str | pd.Timestamp | None = None,
) -> dict[str, object]:
    _raise_downstream_replay_unavailable("fold result")
    required = {
        "schema_version",
        "fold_id",
        "fold_spec_hash",
        "campaign_type",
        "origin_id",
        "fold_as_of_utc",
        "tenor_horizon_bucket",
        "status",
        "included_in_complete_case",
        "target_weight",
        "candidate_abs_error",
        "candidate_squared_error",
        "baseline_abs_error",
        "baseline_squared_error",
        "repricing_max_abs_error",
        "conservation_max_abs_error",
        "fold_as_of_utc",
        "selection_training_context_max_available_at_utc",
        "fold_historical_context_max_available_at_utc",
        "evaluation_snapshot_received_at_utc",
        "masked_target_available_at_utc",
        "selection_identity_hashes",
        "evaluation_identity_hashes",
        "masked_target_quote_id",
        "retained_quote_ids",
        "revealing_quote_ids",
        "later_quote_namespace",
    }
    _require_exact_keys(fold, required, label="fold result")
    _require_equal(fold.get("schema_version"), FOLD_SCHEMA, label="fold schema")
    _require_equal(fold.get("campaign_type"), campaign_type, label="fold campaign")
    _required_text(fold.get("fold_id"), label="fold_id")
    _require_sha256(fold.get("fold_spec_hash"), label="fold_spec_hash")
    _required_text(fold.get("origin_id"), label="origin_id")
    pd.Timestamp(_required_text(fold.get("origin_date"), label="origin_date"))
    _required_text(fold.get("tenor_horizon_bucket"), label="tenor_horizon_bucket")
    status = str(fold.get("status", ""))
    if status not in FOLD_STATUSES:
        raise Tier2MonthlyEexEvaluationError(f"unsupported fold status: {status}")
    included = fold.get("included_in_complete_case")
    if not isinstance(included, bool):
        raise Tier2MonthlyEexEvaluationError("complete-case flag must be boolean")
    if (status == FOLD_EVALUATED) != included:
        raise Tier2MonthlyEexEvaluationError(
            "only EVALUATED folds belong to the complete-case population"
        )
    weight = _finite_number(fold.get("target_weight"), label="target_weight")
    if weight <= 0.0:
        raise Tier2MonthlyEexEvaluationError("target_weight must be positive")
    metric_fields = (
        "candidate_abs_error",
        "candidate_squared_error",
        "baseline_abs_error",
        "baseline_squared_error",
        "repricing_max_abs_error",
        "conservation_max_abs_error",
    )
    metrics = [_finite_number(fold.get(field), label=field) for field in metric_fields]
    if any(value < 0.0 for value in metrics):
        raise Tier2MonthlyEexEvaluationError("fold error metrics cannot be negative")
    fold_as_of = _aware_utc(fold.get("fold_as_of_utc"), label="fold_as_of_utc")
    selection_cutoff = _aware_utc(
        fold.get("selection_training_context_max_available_at_utc"),
        label="selection training context cutoff",
    )
    historical_cutoff = _aware_utc(
        fold.get("fold_historical_context_max_available_at_utc"),
        label="fold historical context cutoff",
    )
    snapshot_time = _aware_utc(
        fold.get("evaluation_snapshot_received_at_utc"),
        label="evaluation snapshot received_at",
    )
    target_time = _aware_utc(
        fold.get("masked_target_available_at_utc"),
        label="masked target available_at",
    )
    if max(selection_cutoff, historical_cutoff, snapshot_time, target_time) > fold_as_of:
        raise Tier2MonthlyEexEvaluationError("fold consumes information after fold_as_of")
    selection_ids = _identity_hashes(
        fold.get("selection_identity_hashes"), label="selection identities"
    )
    evaluation_ids = _identity_hashes(
        fold.get("evaluation_identity_hashes"), label="evaluation identities"
    )
    if campaign_type == "HOLDOUT":
        if freeze_time_utc is None:
            raise Tier2MonthlyEexEvaluationError("holdout fold requires freeze time")
        freeze_time = _aware_utc(freeze_time_utc, label="freeze time")
        if snapshot_time <= freeze_time or target_time <= freeze_time:
            raise Tier2MonthlyEexEvaluationError(
                "holdout evaluation snapshot and target must be post-freeze"
            )
        if selection_ids.intersection(evaluation_ids):
            raise Tier2MonthlyEexEvaluationError(
                "holdout evaluation identities overlap selection evidence"
            )
    _required_text(fold.get("masked_target_quote_id"), label="masked target quote id")
    retained = _unique_text_list(fold.get("retained_quote_ids"), label="retained quotes")
    revealing = _unique_text_list(fold.get("revealing_quote_ids"), label="revealing quotes")
    if revealing:
        raise Tier2MonthlyEexEvaluationError(
            "algebraically revealing quotes were not removed"
        )
    if str(fold.get("masked_target_quote_id")) in retained:
        raise Tier2MonthlyEexEvaluationError("masked target remains in retained inputs")
    _require_equal(
        fold.get("later_quote_namespace"),
        "DIAGNOSTIC_EXCLUDED",
        label="later quote namespace",
    )
    return dict(fold)


def validate_campaign(
    campaign: Mapping[str, object],
    *,
    plan: Mapping[str, object],
    freeze: Mapping[str, object] | None = None,
) -> dict[str, object]:
    _raise_downstream_replay_unavailable("campaign")
    required = {
        "schema_version",
        "campaign_id",
        "scope",
        "full_tier2_pfc_status",
        "production_approved",
        "campaign_type",
        "plan_sha256",
        "candidate_freeze_sha256",
        "expected_fold_hashes",
        "fold_results",
        "bootstrap_block_length_origins",
        "reported_effective_block_count",
        "final_status",
        "performance_summary",
    }
    _require_exact_keys(campaign, required, label="campaign")
    _require_common_scope(campaign)
    _require_equal(campaign.get("schema_version"), CAMPAIGN_SCHEMA, label="campaign schema")
    campaign_type = str(campaign.get("campaign_type", ""))
    if campaign_type not in {"SELECTION", "HOLDOUT"}:
        raise Tier2MonthlyEexEvaluationError("campaign type must be SELECTION or HOLDOUT")
    plan_valid = _validate_evaluation_plan(plan)
    _require_equal(
        campaign.get("plan_sha256"),
        _sha256_json(plan_valid),
        label="campaign plan hash",
    )
    freeze_time: pd.Timestamp | None = None
    if campaign_type == "SELECTION":
        _require_equal(
            campaign.get("candidate_freeze_sha256"), "", label="selection freeze hash"
        )
    else:
        if freeze is None:
            raise Tier2MonthlyEexEvaluationError("holdout campaign requires candidate freeze")
        freeze_valid = validate_candidate_freeze(freeze)
        _require_equal(
            campaign.get("candidate_freeze_sha256"),
            _sha256_json(freeze_valid),
            label="holdout freeze hash",
        )
        _require_equal(
            freeze_valid.get("plan_sha256"),
            _sha256_json(plan_valid),
            label="freeze plan hash",
        )
        freeze_time = _aware_utc(freeze_valid["frozen_at_utc"], label="frozen_at_utc")
    fold_key = "selection_folds" if campaign_type == "SELECTION" else "holdout_folds"
    expected = {
        str(item["fold_id"]): str(item["fold_spec_hash"])
        for item in plan_valid[fold_key]  # type: ignore[index]
    }
    _require_equal(
        campaign.get("expected_fold_hashes"), expected, label="expected fold inventory"
    )
    raw_results = campaign.get("fold_results")
    if not isinstance(raw_results, list):
        raise Tier2MonthlyEexEvaluationError("fold_results must be a list")
    results = [
        validate_fold_result(item, campaign_type=campaign_type, freeze_time_utc=freeze_time)
        for item in raw_results
        if isinstance(item, Mapping)
    ]
    if len(results) != len(raw_results):
        raise Tier2MonthlyEexEvaluationError("fold result is not an object")
    received = {str(item["fold_id"]): str(item["fold_spec_hash"]) for item in results}
    if len(received) != len(results) or received != expected:
        raise Tier2MonthlyEexEvaluationError(
            "received fold inventory differs from preregistered inventory"
        )
    block_length = _positive_int(
        campaign.get("bootstrap_block_length_origins"),
        label="bootstrap block length",
    )
    if campaign_type == "HOLDOUT":
        assert freeze is not None
        _require_equal(
            block_length,
            freeze["bootstrap_block_length_origins"],
            label="frozen bootstrap block length",
        )
    evaluated = [item for item in results if item["status"] == FOLD_EVALUATED]
    origin_count = len({str(item["origin_id"]) for item in evaluated})
    effective_blocks = math.ceil(origin_count / block_length) if origin_count else 0
    _require_equal(
        campaign.get("reported_effective_block_count"),
        effective_blocks,
        label="effective block count",
    )
    computed = _campaign_performance(
        results,
        campaign_type=campaign_type,
        block_length=block_length,
    )
    _require_summary_equal(campaign.get("performance_summary"), computed)
    expected_status = _campaign_status(
        results,
        computed,
        campaign_type=campaign_type,
    )
    _require_equal(campaign.get("final_status"), expected_status, label="campaign status")
    unsigned = dict(campaign)
    campaign_id = str(unsigned.pop("campaign_id", ""))
    if campaign_id != _sha256_json(unsigned):
        raise Tier2MonthlyEexEvaluationError("campaign_id mismatch")
    return dict(campaign)


def validate_evaluation_index(index: Mapping[str, object]) -> dict[str, object]:
    _raise_downstream_replay_unavailable("evaluation index")
    required = {
        "schema_version",
        "index_id",
        "scope",
        "full_tier2_pfc_status",
        "production_approved",
        "plan_sha256",
        "selection_campaign_sha256",
        "candidate_freeze_sha256",
        "holdout_campaign_sha256",
        "plan_governance_key_id",
        "timestamp_key_id",
        "execution_key_id",
        "monthly_claim_status",
    }
    _require_exact_keys(index, required, label="evaluation index")
    _require_common_scope(index)
    _require_equal(index.get("schema_version"), INDEX_SCHEMA, label="index schema")
    for field in (
        "plan_sha256",
        "selection_campaign_sha256",
        "candidate_freeze_sha256",
        "holdout_campaign_sha256",
        "plan_governance_key_id",
        "timestamp_key_id",
        "execution_key_id",
    ):
        _require_sha256(index.get(field), label=field)
    authority_ids = {
        str(index["plan_governance_key_id"]),
        str(index["timestamp_key_id"]),
        str(index["execution_key_id"]),
    }
    if len(authority_ids) != 3:
        raise Tier2MonthlyEexEvaluationError(
            "governance, trusted-time, and execution authorities must be distinct"
        )
    _require_equal(index.get("monthly_claim_status"), PASS_STATUS, label="monthly claim")
    unsigned = dict(index)
    index_id = str(unsigned.pop("index_id", ""))
    if index_id != _sha256_json(unsigned):
        raise Tier2MonthlyEexEvaluationError("index_id mismatch")
    return dict(index)


def verify_governed_plan_artifact(
    plan_path: str | Path,
    *,
    governance_receipt_path: str | Path,
    timestamp_receipt_path: str | Path,
) -> VerifiedEvaluationPlan:
    """Verify exact plan bytes under distinct governance and time authorities."""

    path, plan, plan_bytes = _load_json_artifact(plan_path)
    _, governance_receipt, governance_receipt_bytes = _load_json_artifact(
        governance_receipt_path
    )
    _, timestamp_receipt, timestamp_receipt_bytes = _load_json_artifact(
        timestamp_receipt_path
    )
    _validate_evaluation_plan(plan)
    plan_schema = str(plan["schema_version"])
    timestamp = _verify_exact_artifact_authority_bytes(
        timestamp_receipt,
        artifact_path=path,
        artifact_bytes=plan_bytes,
        expected_role="tier2_monthly_eex_evaluation_plan",
        expected_artifact_schema=plan_schema,
        receipt_schema=TIMESTAMP_RECEIPT_SCHEMA,
        signature_field="trusted_time_attestation",
        trusted_key_env=TIMESTAMP_PUBLIC_KEY_ENV,
        event_time_field="received_at_utc",
        authority_id_field="authority_id",
        id_field="receipt_id",
        journal_required=True,
    )
    verified_governance, governance_key_id = _verify_model_governance_receipt_bytes(
        governance_receipt,
        artifact_path=path,
        artifact_bytes=plan_bytes,
        expected_role="tier2_monthly_eex_evaluation_plan",
        valuation_timestamp=timestamp.event_time_utc,
    )
    if governance_key_id == timestamp.authority_key_id:
        raise Tier2MonthlyEexEvaluationError(
            "plan governance and trusted-time authorities are not independent"
        )
    _require_equal(
        verified_governance.get("artifact_sha256"),
        timestamp.artifact_sha256,
        label="plan authority byte binding",
    )
    holdout_times = [
        _aware_utc(item["fold_as_of_utc"], label="holdout fold_as_of_utc")
        for item in plan["holdout_folds"]  # type: ignore[index]
    ]
    if any(value <= timestamp.event_time_utc for value in holdout_times):
        raise Tier2MonthlyEexEvaluationError(
            "every holdout origin must be strictly post-preregistration trusted time"
        )
    return VerifiedEvaluationPlan(
        plan_id=str(plan["plan_id"]),
        artifact_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        governance_key_id=governance_key_id,
        timestamp_key_id=timestamp.authority_key_id,
        trusted_time_utc=timestamp.event_time_utc,
        governance_receipt_sha256=hashlib.sha256(
            governance_receipt_bytes
        ).hexdigest(),
        timestamp_receipt_sha256=hashlib.sha256(timestamp_receipt_bytes).hexdigest(),
        payload=_deep_freeze_json(plan),
        _verification_token=_VERIFIED_PLAN_TOKEN,
    )


def verify_timestamp_receipt(
    receipt: Mapping[str, object],
    *,
    artifact_path: str | Path,
    expected_role: str,
    expected_schema: str,
) -> VerifiedArtifactAuthority:
    return _verify_exact_artifact_authority(
        receipt,
        artifact_path=artifact_path,
        expected_role=expected_role,
        expected_artifact_schema=expected_schema,
        receipt_schema=TIMESTAMP_RECEIPT_SCHEMA,
        signature_field="trusted_time_attestation",
        trusted_key_env=TIMESTAMP_PUBLIC_KEY_ENV,
        event_time_field="received_at_utc",
        authority_id_field="authority_id",
        id_field="receipt_id",
        journal_required=True,
    )


def verify_execution_attestation(
    attestation: Mapping[str, object],
    *,
    artifact_path: str | Path,
    expected_role: str,
    expected_schema: str,
    forbidden_authority_key_ids: Sequence[str] = (),
) -> VerifiedArtifactAuthority:
    verified = _verify_exact_artifact_authority(
        attestation,
        artifact_path=artifact_path,
        expected_role=expected_role,
        expected_artifact_schema=expected_schema,
        receipt_schema=EXECUTION_ATTESTATION_SCHEMA,
        signature_field="execution_attestation",
        trusted_key_env=EXECUTION_PUBLIC_KEY_ENV,
        event_time_field="executed_at_utc",
        authority_id_field="authority_id",
        id_field="attestation_id",
        journal_required=False,
    )
    if verified.authority_key_id in {str(value) for value in forbidden_authority_key_ids}:
        raise Tier2MonthlyEexEvaluationError(
            "execution authority is not independent of governance/trusted time"
        )
    return verified


def _validate_fold_specs(value: object, *, campaign: str) -> dict[str, str]:
    if not isinstance(value, list) or not value:
        raise Tier2MonthlyEexEvaluationError(f"{campaign} fold plan is empty")
    output: dict[str, str] = {}
    required = {
        "fold_id",
        "campaign_type",
        "origin_id",
        "fold_as_of_utc",
        "tenor_horizon_bucket",
        "target_product",
        "target_load_type",
        "target_rule",
        "historical_context_rule",
        "fold_spec_hash",
    }
    for raw in value:
        if not isinstance(raw, Mapping):
            raise Tier2MonthlyEexEvaluationError("fold specification is not an object")
        _require_exact_keys(raw, required, label="fold specification")
        _require_equal(raw.get("campaign_type"), campaign, label="fold campaign")
        fold_id = _required_text(raw.get("fold_id"), label="fold_id")
        if fold_id in output:
            raise Tier2MonthlyEexEvaluationError("duplicate fold_id")
        fold_as_of_text = _canonical_utc_text(
            raw.get("fold_as_of_utc"), label="fold_as_of_utc"
        )
        expected_origin_id = _planned_origin_id(campaign, fold_as_of_text)
        _require_equal(raw.get("origin_id"), expected_origin_id, label="origin_id")
        bucket = _required_text(
            raw.get("tenor_horizon_bucket"), label="tenor_horizon_bucket"
        )
        if bucket not in MANDATORY_TENOR_HORIZON_BUCKETS:
            raise Tier2MonthlyEexEvaluationError(
                f"fold uses a non-governed tenor-horizon bucket: {bucket}"
            )
        product = _required_text(raw.get("target_product"), label="target_product")
        try:
            target_periods = product_periods(product)
        except ValueError as exc:
            raise Tier2MonthlyEexEvaluationError("target_product is invalid") from exc
        tenor = {1: "MONTH", 3: "QUARTER", 12: "YEAR"}.get(len(target_periods))
        if tenor is None:
            raise Tier2MonthlyEexEvaluationError("target_product tenor is invalid")
        origin = pd.Timestamp(fold_as_of_text).tz_convert(LOCAL_TIMEZONE)
        horizon_years = int(target_periods[0].year) - int(origin.year)
        if horizon_years < 0:
            raise Tier2MonthlyEexEvaluationError("target_product predates fold origin")
        horizon = f"H{horizon_years}" if horizon_years < 3 else "H3_PLUS"
        if bucket != f"{tenor}_{horizon}":
            raise Tier2MonthlyEexEvaluationError(
                "tenor_horizon_bucket does not match origin and target product"
            )
        delivery_start = (
            target_periods[0]
            .to_timestamp(how="start")
            .tz_localize(LOCAL_TIMEZONE)
            .tz_convert("UTC")
        )
        if delivery_start <= pd.Timestamp(fold_as_of_text):
            raise Tier2MonthlyEexEvaluationError(
                "target product delivery must start strictly after fold_as_of_utc"
            )
        raw_load_type = _required_text(
            raw.get("target_load_type"), label="target_load_type"
        )
        load_type = raw_load_type.upper()
        if load_type not in {"BASE", "PEAK", "OFFPEAK"}:
            raise Tier2MonthlyEexEvaluationError("target_load_type is invalid")
        if raw_load_type != load_type:
            raise Tier2MonthlyEexEvaluationError(
                "target_load_type must use canonical uppercase"
            )
        _require_equal(
            raw.get("target_rule"),
            "MASK_ONE_IDENTIFIABLE_SAME_VINTAGE_EEX_QUOTE",
            label="target rule",
        )
        _require_equal(
            raw.get("historical_context_rule"),
            "EXPANDING_PIT_AVAILABLE_AT_LE_FOLD_AS_OF_NO_RETUNING",
            label="historical context rule",
        )
        unsigned = dict(raw)
        fold_hash = str(unsigned.pop("fold_spec_hash", ""))
        if fold_hash != _sha256_json(unsigned):
            raise Tier2MonthlyEexEvaluationError("fold_spec_hash mismatch")
        output[fold_id] = fold_hash
    economic_keys = {
        (
            str(raw["origin_id"]),
            str(raw["target_product"]),
            str(raw["target_load_type"]).upper(),
        )
        for raw in value
        if isinstance(raw, Mapping)
    }
    if len(economic_keys) != len(value):
        raise Tier2MonthlyEexEvaluationError(
            f"{campaign} repeats an origin/target economic fold"
        )
    return output


def _raise_downstream_replay_unavailable(artifact: str) -> None:
    raise Tier2MonthlyEexEvaluationError(
        f"{artifact} validation is UNSUPPORTED until signed row-level EEX replay is implemented"
    )


def _validate_preregistered_coverage(
    folds: Sequence[Mapping[str, object]], *, campaign: str
) -> None:
    origin_times = {
        _canonical_utc_text(item["fold_as_of_utc"], label="fold_as_of_utc")
        for item in folds
    }
    origin_months = {
        pd.Timestamp(value).tz_convert(LOCAL_TIMEZONE).strftime("%Y-%m")
        for value in origin_times
    }
    calendar_months = {
        int(pd.Timestamp(value).tz_convert(LOCAL_TIMEZONE).month)
        for value in origin_times
    }
    min_origins = MIN_SELECTION_ORIGINS if campaign == "SELECTION" else MIN_HOLDOUT_ORIGINS
    min_months = MIN_SELECTION_MONTHS if campaign == "SELECTION" else MIN_HOLDOUT_MONTHS
    if len(origin_times) < min_origins or len(origin_months) < min_months:
        raise Tier2MonthlyEexEvaluationError(
            f"{campaign} preregistration does not meet origin/month coverage"
        )
    if campaign == "SELECTION" and calendar_months != set(range(1, 13)):
        raise Tier2MonthlyEexEvaluationError(
            "SELECTION preregistration does not cover all calendar months"
        )
    for bucket in MANDATORY_TENOR_HORIZON_BUCKETS:
        bucket_origins = {
            _canonical_utc_text(item["fold_as_of_utc"], label="fold_as_of_utc")
            for item in folds
            if str(item["tenor_horizon_bucket"]) == bucket
        }
        if len(bucket_origins) < MIN_BUCKET_ORIGINS:
            raise Tier2MonthlyEexEvaluationError(
                f"{campaign} preregistration undercovers mandatory bucket {bucket}"
            )


def _campaign_performance(
    results: Sequence[Mapping[str, object]],
    *,
    campaign_type: str,
    block_length: int,
) -> dict[str, object]:
    evaluated = [item for item in results if item["status"] == FOLD_EVALUATED]
    by_origin: dict[str, list[Mapping[str, object]]] = {}
    for item in evaluated:
        by_origin.setdefault(str(item["origin_id"]), []).append(item)
    origins: list[dict[str, object]] = []
    for origin_id, rows in sorted(by_origin.items()):
        weights = np.asarray([float(row["target_weight"]) for row in rows], dtype=np.float64)
        weights /= weights.sum()
        candidate_mae = float(
            np.dot(weights, [float(row["candidate_abs_error"]) for row in rows])
        )
        baseline_mae = float(
            np.dot(weights, [float(row["baseline_abs_error"]) for row in rows])
        )
        candidate_rmse = float(
            np.sqrt(np.dot(weights, [float(row["candidate_squared_error"]) for row in rows]))
        )
        baseline_rmse = float(
            np.sqrt(np.dot(weights, [float(row["baseline_squared_error"]) for row in rows]))
        )
        origins.append(
            {
                "origin_id": origin_id,
                "origin_date": min(str(row["origin_date"]) for row in rows),
                "candidate_mae": candidate_mae,
                "baseline_mae": baseline_mae,
                "candidate_rmse": candidate_rmse,
                "baseline_rmse": baseline_rmse,
            }
        )
    origins.sort(key=lambda item: (str(item["origin_date"]), str(item["origin_id"])))
    origin_months = {str(item["origin_date"])[:7] for item in origins}
    calendar_months = {int(str(item["origin_date"])[5:7]) for item in origins}
    bucket_origin_rows: dict[str, dict[str, list[Mapping[str, object]]]] = {}
    for item in evaluated:
        bucket = str(item["tenor_horizon_bucket"])
        origin_id = str(item["origin_id"])
        bucket_origin_rows.setdefault(bucket, {}).setdefault(origin_id, []).append(item)
    bucket_counts = {
        bucket: len(origin_rows)
        for bucket, origin_rows in sorted(bucket_origin_rows.items())
    }
    coverage_ok = (
        len(origins)
        >= (MIN_SELECTION_ORIGINS if campaign_type == "SELECTION" else MIN_HOLDOUT_ORIGINS)
        and len(origin_months)
        >= (MIN_SELECTION_MONTHS if campaign_type == "SELECTION" else MIN_HOLDOUT_MONTHS)
        and (campaign_type != "SELECTION" or calendar_months == set(range(1, 13)))
        and bool(bucket_counts)
        and min(bucket_counts.values()) >= MIN_BUCKET_ORIGINS
    )
    zero_baseline = any(
        float(item["baseline_mae"]) <= BASELINE_ZERO_TOLERANCE
        or float(item["baseline_rmse"]) <= BASELINE_ZERO_TOLERANCE
        for item in origins
    )
    if origins and not zero_baseline:
        candidate_mae = float(np.mean([float(item["candidate_mae"]) for item in origins]))
        baseline_mae = float(np.mean([float(item["baseline_mae"]) for item in origins]))
        candidate_rmse = float(np.mean([float(item["candidate_rmse"]) for item in origins]))
        baseline_rmse = float(np.mean([float(item["baseline_rmse"]) for item in origins]))
        mae_improvement = 1.0 - candidate_mae / baseline_mae
        rmse_improvement = 1.0 - candidate_rmse / baseline_rmse
        per_origin = np.asarray(
            [
                1.0 - float(item["candidate_mae"]) / float(item["baseline_mae"])
                for item in origins
            ],
            dtype=np.float64,
        )
        positive_share = float(np.mean(per_origin > 0.0))
        ci_lower, ci_upper = _moving_block_percentile_ci(per_origin, block_length)
        bucket_degradations: list[float] = []
        for bucket, origin_rows in sorted(bucket_origin_rows.items()):
            candidate_by_origin: list[float] = []
            baseline_by_origin: list[float] = []
            for rows in origin_rows.values():
                weights = np.asarray(
                    [float(row["target_weight"]) for row in rows], dtype=np.float64
                )
                weights /= weights.sum()
                candidate_by_origin.append(
                    float(
                        np.dot(
                            weights,
                            [float(row["candidate_abs_error"]) for row in rows],
                        )
                    )
                )
                baseline_by_origin.append(
                    float(
                        np.dot(
                            weights,
                            [float(row["baseline_abs_error"]) for row in rows],
                        )
                    )
                )
            candidate = float(np.mean(candidate_by_origin))
            baseline = float(np.mean(baseline_by_origin))
            bucket_degradations.append(candidate / baseline - 1.0)
        max_bucket_degradation = max(bucket_degradations)
    else:
        candidate_mae = baseline_mae = candidate_rmse = baseline_rmse = None
        mae_improvement = rmse_improvement = positive_share = None
        ci_lower = ci_upper = max_bucket_degradation = None
    repricing_ok = all(
        float(item["repricing_max_abs_error"]) <= 1e-9
        and float(item["conservation_max_abs_error"]) <= 1e-9
        for item in evaluated
    )
    performance_ok = bool(
        campaign_type == "HOLDOUT"
        and coverage_ok
        and not zero_baseline
        and mae_improvement >= MIN_RELATIVE_IMPROVEMENT
        and rmse_improvement >= MIN_RELATIVE_IMPROVEMENT
        and positive_share >= MIN_POSITIVE_FOLD_SHARE
        and ci_lower > 0.0
        and max_bucket_degradation <= MAX_BUCKET_DEGRADATION
        and repricing_ok
    )
    return {
        "effective_origin_count": len(origins),
        "distinct_origin_month_count": len(origin_months),
        "calendar_months_covered": sorted(calendar_months),
        "bucket_origin_counts": bucket_counts,
        "coverage_ok": coverage_ok,
        "baseline_zero_detected": zero_baseline,
        "candidate_mae": candidate_mae,
        "baseline_mae": baseline_mae,
        "candidate_rmse": candidate_rmse,
        "baseline_rmse": baseline_rmse,
        "mae_relative_improvement": mae_improvement,
        "rmse_relative_improvement": rmse_improvement,
        "positive_mae_origin_share": positive_share,
        "bootstrap_ci_lower": ci_lower,
        "bootstrap_ci_upper": ci_upper,
        "max_bucket_relative_degradation": max_bucket_degradation,
        "repricing_and_conservation_ok": repricing_ok,
        "holdout_performance_ok": performance_ok,
    }


def _campaign_status(
    results: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    *,
    campaign_type: str,
) -> str:
    statuses = {str(item["status"]) for item in results}
    if statuses.intersection(FOLD_ERROR_STATUSES):
        return "ERROR"
    if statuses.intersection(FOLD_FAIL_STATUSES):
        return "FAIL"
    if statuses.intersection(FOLD_UNSUPPORTED_STATUSES):
        return "UNSUPPORTED"
    if not bool(summary["coverage_ok"]) or bool(summary["baseline_zero_detected"]):
        return "UNSUPPORTED"
    if campaign_type == "SELECTION":
        if bool(summary["repricing_and_conservation_ok"]):
            return SELECTION_PASS_STATUS
        return "FAIL"
    if bool(summary["holdout_performance_ok"]):
        return PASS_STATUS
    return "FAIL"


def _moving_block_percentile_ci(
    values: np.ndarray, block_length: int
) -> tuple[float, float]:
    if values.ndim != 1 or len(values) == 0:
        return float("nan"), float("nan")
    block = min(_positive_int(block_length, label="block length"), len(values))
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    blocks_needed = math.ceil(len(values) / block)
    samples = np.empty(MIN_BOOTSTRAP_REPLICATES, dtype=np.float64)
    offsets = np.arange(block)
    for replicate in range(MIN_BOOTSTRAP_REPLICATES):
        starts = rng.integers(0, len(values), size=blocks_needed)
        indices = ((starts[:, None] + offsets[None, :]) % len(values)).ravel()[: len(values)]
        samples[replicate] = float(np.mean(values[indices]))
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return float(lower), float(upper)


def _verify_exact_artifact_authority(
    envelope: Mapping[str, object],
    *,
    artifact_path: str | Path,
    expected_role: str,
    expected_artifact_schema: str,
    receipt_schema: str,
    signature_field: str,
    trusted_key_env: str,
    event_time_field: str,
    authority_id_field: str,
    id_field: str,
    journal_required: bool,
) -> VerifiedArtifactAuthority:
    path = _regular_file(artifact_path)
    try:
        artifact_bytes = path.read_bytes()
    except OSError as exc:
        raise Tier2MonthlyEexEvaluationError("artifact bytes are unreadable") from exc
    return _verify_exact_artifact_authority_bytes(
        envelope,
        artifact_path=path,
        artifact_bytes=artifact_bytes,
        expected_role=expected_role,
        expected_artifact_schema=expected_artifact_schema,
        receipt_schema=receipt_schema,
        signature_field=signature_field,
        trusted_key_env=trusted_key_env,
        event_time_field=event_time_field,
        authority_id_field=authority_id_field,
        id_field=id_field,
        journal_required=journal_required,
    )


def _verify_exact_artifact_authority_bytes(
    envelope: Mapping[str, object],
    *,
    artifact_path: Path,
    artifact_bytes: bytes,
    expected_role: str,
    expected_artifact_schema: str,
    receipt_schema: str,
    signature_field: str,
    trusted_key_env: str,
    event_time_field: str,
    authority_id_field: str,
    id_field: str,
    journal_required: bool,
    trusted_public_key: Ed25519PublicKey | None = None,
    trusted_key_path: str | None = None,
    trusted_key_sha256: str | None = None,
    expected_journal_id: str | None = None,
) -> VerifiedArtifactAuthority:
    unsigned = dict(envelope)
    signature = unsigned.pop(signature_field, None)
    expected_fields = {
        "schema_version",
        "artifact_role",
        "artifact_schema_version",
        "artifact_sha256",
        "artifact_size_bytes",
        authority_id_field,
        event_time_field,
        id_field,
    }
    if journal_required:
        expected_fields.update(
            {"journal_id", "journal_sequence", "previous_receipt_id"}
        )
    _require_exact_keys(unsigned, expected_fields, label="authority envelope")
    _require_equal(unsigned.get("schema_version"), receipt_schema, label="authority schema")
    _require_equal(unsigned.get("artifact_role"), expected_role, label="artifact role")
    _require_equal(
        unsigned.get("artifact_schema_version"),
        expected_artifact_schema,
        label="artifact schema binding",
    )
    artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
    _require_equal(unsigned.get("artifact_sha256"), artifact_hash, label="artifact hash")
    _require_equal(
        unsigned.get("artifact_size_bytes"), len(artifact_bytes), label="artifact size"
    )
    _required_text(unsigned.get(authority_id_field), label=authority_id_field)
    event_time = _aware_utc(unsigned.get(event_time_field), label=event_time_field)
    journal_id: str | None = None
    if journal_required:
        journal_id = (
            expected_journal_id.strip()
            if expected_journal_id is not None
            else os.environ.get(TIMESTAMP_JOURNAL_ID_ENV, "").strip()
        )
        if not journal_id:
            raise Tier2MonthlyEexEvaluationError(
                f"{TIMESTAMP_JOURNAL_ID_ENV} is required"
            )
        _require_equal(unsigned.get("journal_id"), journal_id, label="timestamp journal")
        _positive_int(unsigned.get("journal_sequence"), label="journal sequence", allow_zero=True)
        previous = str(unsigned.get("previous_receipt_id", ""))
        if previous:
            _require_sha256(previous, label="previous_receipt_id")
    identity = dict(unsigned)
    envelope_id = str(identity.pop(id_field, ""))
    if envelope_id != _sha256_json(identity):
        raise Tier2MonthlyEexEvaluationError(f"{id_field} mismatch")
    if not isinstance(signature, Mapping):
        raise Tier2MonthlyEexEvaluationError(f"{signature_field} is missing")
    _require_exact_keys(
        signature,
        {"algorithm", "key_id", "payload_sha256", "value_base64"},
        label=signature_field,
    )
    _require_equal(signature.get("algorithm"), SIGNATURE_ALGORITHM, label="signature algorithm")
    if trusted_public_key is None:
        key_path_value = os.environ.get(trusted_key_env, "").strip()
        if not key_path_value:
            raise Tier2MonthlyEexEvaluationError(f"{trusted_key_env} is required")
        key_path = Path(key_path_value)
        if not key_path.is_absolute():
            raise Tier2MonthlyEexEvaluationError(
                "trusted public key path must be absolute"
            )
        key_path = _regular_file(key_path)
        try:
            key_bytes = key_path.read_bytes()
        except OSError as exc:
            raise Tier2MonthlyEexEvaluationError(
                "cannot read trusted public key"
            ) from exc
        public_key = _load_public_key_bytes(key_bytes)
        verified_key_path = str(key_path)
        verified_key_sha256 = hashlib.sha256(key_bytes).hexdigest()
    else:
        if not trusted_key_path or not trusted_key_sha256:
            raise Tier2MonthlyEexEvaluationError(
                "captured trusted key identity is incomplete"
            )
        public_key = trusted_public_key
        verified_key_path = trusted_key_path
        verified_key_sha256 = trusted_key_sha256
    key_id = _public_key_id(public_key)
    _require_equal(signature.get("key_id"), key_id, label="trusted authority key")
    signed = {**identity, id_field: envelope_id}
    payload = _canonical_json_bytes(signed)
    _require_equal(
        signature.get("payload_sha256"), hashlib.sha256(payload).hexdigest(), label="payload hash"
    )
    try:
        raw_signature = base64.b64decode(str(signature.get("value_base64", "")), validate=True)
        public_key.verify(raw_signature, payload)
    except (TypeError, ValueError, InvalidSignature) as exc:
        raise Tier2MonthlyEexEvaluationError("authority signature is invalid") from exc
    return VerifiedArtifactAuthority(
        artifact_sha256=artifact_hash,
        artifact_role=expected_role,
        artifact_schema_version=expected_artifact_schema,
        envelope_id=envelope_id,
        authority_key_id=key_id,
        trusted_key_path=verified_key_path,
        trusted_key_sha256=verified_key_sha256,
        journal_id=journal_id,
        event_time_utc=event_time,
    )


def _verify_model_governance_receipt_bytes(
    receipt: Mapping[str, object],
    *,
    artifact_path: Path,
    artifact_bytes: bytes,
    expected_role: str,
    valuation_timestamp: str | pd.Timestamp,
) -> tuple[dict[str, object], str]:
    payload = dict(receipt)
    signature = payload.pop("signature", None)
    expected_fields = {
        "schema_version",
        "artifact_role",
        "artifact_sha256",
        "artifact_size_bytes",
        "authority_id",
        "issued_at_utc",
        "data_cutoff_utc",
        "approved",
        "receipt_id",
    }
    _require_exact_keys(payload, expected_fields, label="governance receipt")
    _require_equal(
        payload.get("schema_version"),
        MODEL_GOVERNANCE_RECEIPT_SCHEMA,
        label="governance receipt schema",
    )
    _require_equal(payload.get("artifact_role"), expected_role, label="governance role")
    artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
    _require_equal(payload.get("artifact_sha256"), artifact_hash, label="governance hash")
    _require_equal(
        payload.get("artifact_size_bytes"),
        len(artifact_bytes),
        label="governance artifact size",
    )
    if payload.get("approved") is not True:
        raise Tier2MonthlyEexEvaluationError("governance artifact is not approved")
    _required_text(payload.get("authority_id"), label="governance authority_id")
    issued_at = _aware_utc(payload.get("issued_at_utc"), label="governance issued_at")
    data_cutoff = _aware_utc(payload.get("data_cutoff_utc"), label="governance data cutoff")
    valuation = _aware_utc(valuation_timestamp, label="governance valuation time")
    if issued_at > valuation or data_cutoff > issued_at:
        raise Tier2MonthlyEexEvaluationError("governance receipt is not point-in-time")
    identity = dict(payload)
    receipt_id = str(identity.pop("receipt_id", ""))
    if receipt_id != _sha256_json(identity):
        raise Tier2MonthlyEexEvaluationError("governance receipt_id mismatch")
    if not isinstance(signature, Mapping):
        raise Tier2MonthlyEexEvaluationError("governance signature is missing")
    _require_exact_keys(
        signature,
        {"algorithm", "key_id", "value_base64"},
        label="governance signature",
    )
    _require_equal(
        signature.get("algorithm"), SIGNATURE_ALGORITHM, label="governance algorithm"
    )
    key_path = os.environ.get(MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV, "").strip()
    if not key_path:
        raise Tier2MonthlyEexEvaluationError(
            f"{MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV} is required"
        )
    public_key = _load_public_key(key_path)
    key_id = _public_key_id(public_key)
    _require_equal(signature.get("key_id"), key_id, label="governance key")
    signed_payload = {**identity, "receipt_id": receipt_id}
    try:
        raw_signature = base64.b64decode(
            str(signature.get("value_base64", "")), validate=True
        )
        public_key.verify(raw_signature, _canonical_json_bytes(signed_payload))
    except (TypeError, ValueError, InvalidSignature) as exc:
        raise Tier2MonthlyEexEvaluationError("governance signature is invalid") from exc
    if hashlib.sha256(artifact_bytes).hexdigest() != artifact_hash:
        raise Tier2MonthlyEexEvaluationError(
            f"artifact snapshot changed during verification: {artifact_path}"
        )
    return signed_payload, key_id


def _effective_rank_and_condition(matrix: np.ndarray) -> tuple[int, float]:
    if matrix.size == 0 or matrix.shape[0] == 0:
        return 0, 1.0
    singular = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    if len(singular) == 0:
        return 0, 1.0
    threshold = max(RANK_ATOL, RANK_RTOL * float(singular[0]))
    nonzero = singular[singular > threshold]
    rank = int(len(nonzero))
    condition = float(nonzero[0] / nonzero[-1]) if rank else 1.0
    return rank, condition


def _deduplicate_rows(rows: Sequence[np.ndarray]) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for row in rows:
        if not any(np.array_equal(row, existing) for existing in output):
            output.append(row)
    return output


def _l2_normalize(row: np.ndarray) -> np.ndarray:
    vector = np.asarray(row, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise Tier2MonthlyEexEvaluationError("constraint vector has zero/non-finite norm")
    return vector / norm


def _require_common_scope(artifact: Mapping[str, object]) -> None:
    _require_equal(artifact.get("scope"), SCOPE, label="evaluation scope")
    _require_equal(
        artifact.get("full_tier2_pfc_status"), FULL_TIER2_STATUS, label="full Tier-2 status"
    )
    if artifact.get("production_approved") is not False:
        raise Tier2MonthlyEexEvaluationError("monthly EEX evidence cannot approve production")


def _require_summary_equal(actual: object, expected: Mapping[str, object]) -> None:
    if not isinstance(actual, Mapping) or set(actual) != set(expected):
        raise Tier2MonthlyEexEvaluationError("performance summary schema mismatch")
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if expected_value is None:
            if actual_value is not None:
                raise Tier2MonthlyEexEvaluationError(
                    f"performance summary mismatch: {key}"
                )
        elif isinstance(expected_value, float):
            if math.isnan(expected_value):
                if not isinstance(actual_value, (int, float)) or not math.isnan(float(actual_value)):
                    raise Tier2MonthlyEexEvaluationError(
                        f"performance summary mismatch: {key}"
                    )
            elif not math.isclose(
                float(actual_value), expected_value, rel_tol=0.0, abs_tol=1e-12
            ):
                raise Tier2MonthlyEexEvaluationError(f"performance summary mismatch: {key}")
        elif actual_value != expected_value:
            raise Tier2MonthlyEexEvaluationError(f"performance summary mismatch: {key}")


def _load_json_artifact(
    path_value: str | Path,
) -> tuple[Path, dict[str, object], bytes]:
    path = _regular_file(path_value)
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Tier2MonthlyEexEvaluationError("JSON artifact is unreadable") from exc
    if not isinstance(value, dict):
        raise Tier2MonthlyEexEvaluationError("JSON artifact root must be an object")
    return path, value, raw


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise Tier2MonthlyEexEvaluationError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _regular_file(path_value: str | Path) -> Path:
    path = Path(path_value).absolute()
    if _path_has_reparse_component(path):
        raise Tier2MonthlyEexEvaluationError(
            "artifact path cannot traverse a symlink or junction"
        )
    if not path.is_file():
        raise Tier2MonthlyEexEvaluationError("artifact must be a regular non-symlink file")
    return path.resolve()


def _path_has_reparse_component(path: Path) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for component in (path, *path.parents):
        if component.is_symlink():
            return True
        try:
            attributes = getattr(os.lstat(component), "st_file_attributes", 0)
        except OSError:
            continue
        if int(attributes) & int(reparse_flag):
            return True
    return False


def _load_public_key(path_value: str | Path) -> Ed25519PublicKey:
    path = Path(path_value)
    if not path.is_absolute():
        raise Tier2MonthlyEexEvaluationError("trusted public key path must be absolute")
    trusted_path = _regular_file(path)
    try:
        key_bytes = trusted_path.read_bytes()
    except OSError as exc:
        raise Tier2MonthlyEexEvaluationError("cannot load trusted public key") from exc
    return _load_public_key_bytes(key_bytes)


def _load_public_key_bytes(key_bytes: bytes) -> Ed25519PublicKey:
    try:
        key = serialization.load_pem_public_key(key_bytes)
    except (TypeError, ValueError) as exc:
        raise Tier2MonthlyEexEvaluationError("cannot load trusted public key") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise Tier2MonthlyEexEvaluationError("trusted key is not Ed25519")
    return key


def _public_key_id(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def _require_exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    missing = sorted(expected.difference(value))
    extra = sorted(set(value).difference(expected))
    if missing or extra:
        raise Tier2MonthlyEexEvaluationError(
            f"{label} fields mismatch: missing={missing}, extra={extra}"
        )


def _require_equal(actual: object, expected: object, *, label: str) -> None:
    if not _strict_json_equal(actual, expected):
        raise Tier2MonthlyEexEvaluationError(f"{label} mismatch")


def _strict_json_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            return False
        return all(_strict_json_equal(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _strict_json_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Tier2MonthlyEexEvaluationError(f"{label} must be an object")
    return value


def _required_text(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise Tier2MonthlyEexEvaluationError(f"{label} is required")
    return text


def _require_sha256(value: object, *, label: str) -> str:
    text = _required_text(value, label=label).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise Tier2MonthlyEexEvaluationError(f"{label} is not a SHA-256 digest")
    return text


def _positive_int(value: object, *, label: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Tier2MonthlyEexEvaluationError(f"{label} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise Tier2MonthlyEexEvaluationError(f"{label} is below {minimum}")
    return value


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Tier2MonthlyEexEvaluationError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise Tier2MonthlyEexEvaluationError(f"{label} must be finite")
    return number


def _unique_text_list(value: object, *, label: str) -> list[str]:
    if not isinstance(value, list):
        raise Tier2MonthlyEexEvaluationError(f"{label} must be a list")
    items = [_required_text(item, label=label) for item in value]
    if len(items) != len(set(items)):
        raise Tier2MonthlyEexEvaluationError(f"{label} contains duplicates")
    return items


def _identity_hashes(value: object, *, label: str) -> set[str]:
    identity = _mapping(value, label=label)
    expected = {"document", "snapshot", "revision", "quote"}
    _require_exact_keys(identity, expected, label=label)
    output: set[str] = set()
    for role in sorted(expected):
        values = _unique_text_list(identity[role], label=f"{label}.{role}")
        for item in values:
            digest = _require_sha256(item, label=f"{label}.{role}")
            if digest in output:
                raise Tier2MonthlyEexEvaluationError(
                    f"{label} repeats an identity across roles"
                )
            output.add(digest)
    if not output:
        raise Tier2MonthlyEexEvaluationError(f"{label} cannot be empty")
    return output


def _aware_utc(value: object, *, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise Tier2MonthlyEexEvaluationError(f"{label} is invalid") from exc
    if timestamp.tzinfo is None:
        raise Tier2MonthlyEexEvaluationError(f"{label} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _canonical_utc_text(value: object, *, label: str) -> str:
    timestamp = _aware_utc(value, label=label)
    canonical = timestamp.isoformat()
    if str(value) != canonical:
        raise Tier2MonthlyEexEvaluationError(
            f"{label} must use canonical UTC ISO-8601 form {canonical}"
        )
    return canonical


def _planned_origin_id(campaign: str, fold_as_of_utc: str) -> str:
    return _sha256_json(
        {
            "schema_version": "tier2_monthly_eex_origin.v1",
            "campaign_type": campaign,
            "fold_as_of_utc": fold_as_of_utc,
        }
    )


def _deep_freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze_json(item) for item in value)
    return value


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Tier2MonthlyEexEvaluationError("artifact is not canonical JSON data") from exc


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "FULL_TIER2_STATUS",
    "IdentifiabilityAssessment",
    "MANDATORY_TENOR_HORIZON_BUCKETS",
    "SELECTION_PASS_STATUS",
    "PLAN_SCHEMA",
    "ProductConstraint",
    "SCOPE",
    "Tier2MonthlyEexEvaluationError",
    "assess_target_identifiability",
    "product_constraint_vector",
    "verify_governed_plan_artifact",
]
