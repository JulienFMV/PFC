"""Dormant explicit combination boundary for EEX short-tenor components.

This module performs mathematical contract testing only.  It does not learn
coefficients, choose caps, authenticate training/selection receipts, discover
components, assemble a candidate, or activate the LT production pipeline.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pfc_shaping.lt.model.short_tenor_shape_contract import (
    ShortTenorShapeProjection,
    project_short_tenor_shape_delta,
)

SHORT_TENOR_COMBINATION_CONTRACT_VERSION = (
    "fmv-eex-ch-short-tenor-combination-contract-v1"
)
SHORT_TENOR_COMBINATION_SCHEMA_VERSION = (
    "fmv_eex_ch_short_tenor_combination_projection.v1"
)
MAX_LINEARITY_RESIDUAL_EUR_MWH = 1e-9
COMPONENT_AUDIT_COLUMNS = (
    "component_key",
    "coefficient",
    "max_abs_raw_component_eur_mwh",
    "max_abs_raw_contribution_eur_mwh",
    "max_abs_projected_component_eur_mwh",
    "max_abs_projected_contribution_eur_mwh",
    "max_constraint_residual_eur_mwh",
    "max_monthly_residual_eur_mwh",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

SELECTED_CONTRAST_BUNDLE_CONTENT_ID = (
    "309b9f07236d1cfb32b3d92c1ed5413bde6966fa383629882a539ccbd60e9cb5"
)
SELECTED_CONTRAST_BUNDLE_MANIFEST_SHA256 = (
    "57b571bef311cf22955edc415e650f55671ad2f243141d4f334ccac425d8bb0f"
)
SELECTED_CONTRAST_BUNDLE_DECISION = "D-20260805-225"
SELECTED_SHAPE_PROJECTION_CONTENT_ID = (
    "4da441b09c5559d772fd507214b2989489314bd9114a9328752a150f3957450a"
)
SUCCESSOR_CORE_ID = (
    "d86dab183dd95a2ca3cda0862f2c9f50cec01716a37c6cdc1c3a15763254dcc3"
)
SUCCESSOR_CORE_SHA256 = (
    "e2c2a6f9d3ca677991f3f76f03dec1328492e2f60a4015b7796a2ff6aa6f905a"
)
ALLOWED_CANDIDATE_FAMILIES = (
    "ROBUST_REGULARIZED_LINEAR_GAM_SHAPE_V1",
    "MONOTONE_HIST_GRADIENT_BOOSTING_SHAPE_V1",
)
ALLOWED_BASELINE_FAMILIES = (
    "MARKET_CONSTRAINED_SEASONAL_NAIVE_V1",
    "INCUMBENT_CH_LT_FROZEN_V1",
)
_REQUIRED_DIAGNOSTICS = (
    "BIAS_EUR_MWH",
    "RMSE_EUR_MWH",
    "MEDIAN_ABSOLUTE_ERROR_EUR_MWH",
    "RAMP_ERROR_EUR_MWH",
    "NEGATIVE_PRICE_REGIME_MAE_EUR_MWH",
    "TAIL_MAE_EUR_MWH",
    "PEAK_OFFPEAK_MAE_EUR_MWH",
    "WEEKDAY_WEEKEND_HOLIDAY_MAE_EUR_MWH",
    "SEASON_MAE_EUR_MWH",
    "DST_TRANSITION_MAE_EUR_MWH",
    "FORECAST_HORIZON_BUCKET_MAE_EUR_MWH",
    "VINTAGE_REVISION_STABILITY_EUR_MWH",
)
_PENDING_NUMERIC_FIELDS = (
    "coefficient_lattice",
    "coefficient_absolute_cap",
    "per_component_contribution_absolute_cap_eur_mwh",
    "combined_signal_absolute_cap_eur_mwh",
    "superiority_margin_eur_mwh",
    "noninferiority_margins_eur_mwh",
    "minimum_marginal_power_by_contrast",
    "required_unique_outer_origins_by_contrast",
)


@dataclass(frozen=True)
class ShortTenorCombinationProjection:
    """Explicit weighted sum plus the final D219 solver-neutral projection."""

    pre_final_projection_delta: pd.Series
    projection: ShortTenorShapeProjection
    component_audit: pd.DataFrame
    audit: Mapping[str, object]

    @property
    def delta(self) -> pd.Series:
        """Return the final solver-neutral additive delta."""

        return self.projection.delta


def validate_short_tenor_combination_policy(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate the local outcome-blind policy without authorizing execution.

    The v1 document is deliberately incomplete as an executable
    preregistration: coefficients, caps, margins, required origin counts and
    embargo lengths must remain unset until direct-CH power evidence and an
    external trusted-time freeze exist.  Any premature number or authority
    escalation fails closed.
    """

    if not isinstance(document, Mapping):
        raise TypeError("short-tenor combination policy must be a mapping")
    _require_exact_keys(
        document,
        {
            "schema_version",
            "status",
            "purpose",
            "scientific_basis",
            "selected_boundary_evidence",
            "mathematical_operation",
            "required_explicit_runtime_bindings",
            "future_empirical_selection_policy",
            "numeric_decisions_pending_external_freeze",
            "outcome_blind_firewall",
            "forbidden_inputs_and_operations",
            "authorities",
            "required_before_empirical_use",
        },
        label="combination policy",
    )
    _require_equal(
        document.get("schema_version"),
        SHORT_TENOR_COMBINATION_CONTRACT_VERSION,
        "combination policy schema",
    )
    _require_equal(
        document.get("status"),
        "PASS_LOCAL_STRUCTURE_ONLY_NO_EMPIRICAL_EXECUTION",
        "combination policy status",
    )

    references = document.get("scientific_basis")
    if not isinstance(references, list) or len(references) < 5:
        raise ValueError("combination policy requires the frozen scientific basis")
    observed_dois = {
        str(_require_mapping(item, "scientific reference").get("doi"))
        for item in references
    }
    required_dois = {
        "10.1016/j.apenergy.2021.116983",
        "10.1016/j.eneco.2017.12.016",
        "10.1111/j.1468-0262.2006.00718.x",
        "10.1111/j.1468-0262.2005.00615.x",
        "10.3982/ECTA5771",
    }
    if observed_dois != required_dois:
        raise ValueError("combination policy scientific basis has drifted")

    evidence = _require_mapping(
        document.get("selected_boundary_evidence"), "selected boundary evidence"
    )
    _require_equal(
        evidence.get("contrast_construction_content_id"),
        SELECTED_CONTRAST_BUNDLE_CONTENT_ID,
        "selected contrast bundle",
    )
    _require_equal(
        evidence.get("contrast_construction_manifest_sha256"),
        SELECTED_CONTRAST_BUNDLE_MANIFEST_SHA256,
        "selected contrast manifest",
    )
    _require_equal(
        evidence.get("contrast_construction_decision"),
        SELECTED_CONTRAST_BUNDLE_DECISION,
        "selected contrast decision",
    )
    _require_equal(
        evidence.get("shape_projection_content_id"),
        SELECTED_SHAPE_PROJECTION_CONTENT_ID,
        "selected shape projection",
    )
    _require_equal(
        evidence.get("successor_core_id"),
        SUCCESSOR_CORE_ID,
        "successor core id",
    )
    _require_equal(
        evidence.get("successor_core_sha256"),
        SUCCESSOR_CORE_SHA256,
        "successor core hash",
    )
    for field in (
        "point_in_time_availability_proven",
        "model_input_authorized",
        "production_authorized",
    ):
        _require_false(evidence.get(field), f"selected evidence {field}")

    operation = _require_mapping(
        document.get("mathematical_operation"), "mathematical operation"
    )
    _require_equal(
        operation.get("price_space"),
        "additive_eur_per_mwh",
        "combination price space",
    )
    _require_equal(
        operation.get("combination"),
        "explicit_finite_scalar_weighted_sum",
        "combination operation",
    )
    for field in (
        "component_keys_and_coefficient_keys_must_match_exactly",
        "all_provenance_content_ids_pairwise_distinct",
    ):
        _require_true(operation.get(field), f"mathematical operation {field}")
    for field in (
        "automatic_weight_default_authorized",
        "automatic_component_discovery_authorized",
        "silent_coefficient_clipping_authorized",
        "silent_signal_clipping_authorized",
        "post_projection_clipping_authorized",
    ):
        _require_false(operation.get(field), f"mathematical operation {field}")
    _require_true(
        operation.get("hard_rejection_on_explicit_cap_breach"),
        "hard cap rejection",
    )

    policy = _require_mapping(
        document.get("future_empirical_selection_policy"),
        "future empirical selection policy",
    )
    _require_false(policy.get("current_execution_authorized"), "policy execution")
    _require_equal(
        tuple(_require_string_list(policy.get("candidate_families"), "candidates")),
        ALLOWED_CANDIDATE_FAMILIES,
        "candidate families",
    )
    _require_equal(
        tuple(_require_string_list(policy.get("baseline_families"), "baselines")),
        ALLOWED_BASELINE_FAMILIES,
        "baseline families",
    )
    _require_equal(
        tuple(_require_string_list(policy.get("fixed_family_tie_order"), "tie order")),
        ALLOWED_CANDIDATE_FAMILIES,
        "fixed family tie order",
    )
    for field in (
        "minimum_inner_origins",
        "minimum_effective_clusters",
        "embargo_months",
    ):
        _require_equal(policy.get(field), None, f"unfrozen {field}")
    _require_equal(
        policy.get("primary_loss"),
        "NATIVE_REGIME_WITHIN_MONTH_SHAPE_MAE_EUR_MWH",
        "primary loss",
    )
    _require_equal(
        tuple(
            _require_string_list(
                policy.get("mandatory_diagnostics"), "mandatory diagnostics"
            )
        ),
        _REQUIRED_DIAGNOSTICS,
        "mandatory diagnostics",
    )
    _require_equal(
        policy.get("multiplicity"),
        "HOLM_FWER_OVER_EXACT_FROZEN_PRIMARY_FAMILY",
        "multiplicity policy",
    )
    for field in (
        "same_outer_origins_targets_and_eligibility_masks_required",
        "selection_nested_inside_each_outer_origin",
        "coefficients_refit_per_outer_origin",
        "caps_selected_from_prefrozen_training_only_lattice_per_outer_origin",
        "candidate_and_hyperparameter_grid_hash_required",
        "feature_group_leave_one_out_ablation_required",
        "common_origin_paired_loss_testing_required",
        "hourly_market_regime_scored_on_native_hourly_truth",
        "future_holdout_strictly_post_external_freeze",
        "future_holdout_one_shot",
        "legacy_t057_excluded",
    ):
        _require_true(policy.get(field), f"future policy {field}")
    for field in (
        "selection_holdout_origin_overlap_authorized",
        "hourly_truth_may_be_replicated_as_four_independent_quarter_hours",
        "numeric_coefficients_or_caps_selected_in_this_contract",
    ):
        _require_false(policy.get(field), f"future policy {field}")

    pending = _require_mapping(
        document.get("numeric_decisions_pending_external_freeze"),
        "pending numeric decisions",
    )
    for field in _PENDING_NUMERIC_FIELDS:
        _require_equal(pending.get(field), None, f"pending numeric field {field}")
    _require_false(
        pending.get("post_outcome_changes_authorized"), "post-outcome changes"
    )

    firewall = _require_mapping(
        document.get("outcome_blind_firewall"), "outcome-blind firewall"
    )
    _require_equal(
        firewall.get("ompex_role"),
        "POST_CANDIDATE_READ_ONLY_BENCHMARK_ONLY",
        "OMPEX role",
    )
    _require_equal(firewall.get("t057_role"), "SEALED_AND_EXCLUDED", "T057 role")
    for field in (
        "ompex_allowed_for_training_tuning_selection_caps_or_margins",
        "afry_numeric_values_allowed",
        "local_non_pit_eex_allowed_as_rolling_origin_evidence",
        "legacy_or_synthetic_entsoe_substitution_allowed",
        "outcomes_opened",
        "future_holdout_consumed",
    ):
        _require_false(firewall.get(field), f"outcome firewall {field}")

    authorities = _require_mapping(document.get("authorities"), "authorities")
    _require_true(
        authorities.get("local_mathematical_combination_testing"),
        "local mathematical testing",
    )
    _require_true(
        authorities.get("descriptive_policy_design"), "descriptive policy design"
    )
    for field in (
        "training_receipt_authentication_performed",
        "selection_receipt_authentication_performed",
        "point_in_time_availability_proven",
        "rolling_origin_selection_authorized",
        "model_input_authorized",
        "candidate_assembly_authorized",
        "production_authorized",
    ):
        _require_false(authorities.get(field), f"authority {field}")

    blockers = _require_string_list(
        document.get("required_before_empirical_use"), "empirical-use blockers"
    )
    if len(blockers) < 8:
        raise ValueError("combination policy empirical-use blockers are incomplete")
    canonical = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema_version": SHORT_TENOR_COMBINATION_CONTRACT_VERSION,
        "status": "PASS_LOCAL_STRUCTURE_ONLY_NO_EMPIRICAL_EXECUTION",
        "policy_content_id": hashlib.sha256(canonical).hexdigest(),
        "selected_contrast_bundle_content_id": SELECTED_CONTRAST_BUNDLE_CONTENT_ID,
        "successor_core_id": SUCCESSOR_CORE_ID,
        "candidate_families": list(ALLOWED_CANDIDATE_FAMILIES),
        "baseline_families": list(ALLOWED_BASELINE_FAMILIES),
        "blocker_count": len(blockers),
        "numeric_decisions_frozen": False,
        "outcomes_opened": False,
        "execution_authorized": False,
        "model_input_authorized": False,
        "production_authorized": False,
    }


def combine_short_tenor_shape_components(
    components: Mapping[str, pd.Series],
    *,
    coefficients: Mapping[str, float],
    coefficient_abs_cap: float,
    per_component_contribution_abs_cap_eur_mwh: float,
    combined_signal_abs_cap_eur_mwh: float,
    monthly_base_levels: Mapping[str, float],
    source_normalization_content_id: str,
    source_audit_content_id: str,
    source_contrast_bundle_content_id: str,
    combination_policy_content_id: str,
    training_receipt_content_id: str,
    selection_receipt_content_id: str,
    valuation_timestamp_utc: str | pd.Timestamp,
    peak_forward_prices: Mapping[str, float] | None = None,
    country: str = "CH",
) -> ShortTenorCombinationProjection:
    """Combine explicit components without selecting or silently clipping them.

    Every component is independently validated/projected through D219.  The
    weighted sum is projected once more to prove linearity and active-constraint
    preservation.  Receipt identifiers are bound for replay but deliberately
    not authenticated by this dormant local boundary.
    """

    identifiers = {
        "source_normalization_content_id": source_normalization_content_id,
        "source_audit_content_id": source_audit_content_id,
        "source_contrast_bundle_content_id": source_contrast_bundle_content_id,
        "combination_policy_content_id": combination_policy_content_id,
        "training_receipt_content_id": training_receipt_content_id,
        "selection_receipt_content_id": selection_receipt_content_id,
    }
    for field, value in identifiers.items():
        _validate_content_id(value, field)
    if len(set(identifiers.values())) != len(identifiers):
        raise ValueError("short-tenor provenance content IDs must be pairwise distinct")
    if str(country).upper() != "CH":
        raise ValueError("short-tenor combination contract currently supports CH only")

    coefficient_cap = _positive_finite(
        coefficient_abs_cap,
        "coefficient_abs_cap",
    )
    contribution_cap = _positive_finite(
        per_component_contribution_abs_cap_eur_mwh,
        "per_component_contribution_abs_cap_eur_mwh",
    )
    combined_cap = _positive_finite(
        combined_signal_abs_cap_eur_mwh,
        "combined_signal_abs_cap_eur_mwh",
    )
    ordered_components = _validate_components(components)
    ordered_coefficients = _validate_coefficients(
        coefficients,
        component_keys=tuple(ordered_components),
        coefficient_abs_cap=coefficient_cap,
    )

    first_index = next(iter(ordered_components.values())).index
    component_rows: list[dict[str, object]] = []
    projected_values: dict[str, np.ndarray] = {}
    for component_key, raw_signal in ordered_components.items():
        if not raw_signal.index.equals(first_index):
            raise ValueError("all short-tenor components must use the same delivery index")
        coefficient = ordered_coefficients[component_key]
        raw_values = raw_signal.to_numpy(dtype=float)
        max_abs_raw = float(np.max(np.abs(raw_values)))
        max_abs_raw_contribution = abs(coefficient) * max_abs_raw
        if max_abs_raw_contribution > contribution_cap:
            raise ValueError(
                "short-tenor raw component contribution breaches its explicit cap: "
                f"{component_key} {max_abs_raw_contribution:.12g} > "
                f"{contribution_cap:.12g} EUR/MWh"
            )
        component_projection = project_short_tenor_shape_delta(
            raw_signal,
            monthly_base_levels=monthly_base_levels,
            peak_forward_prices=peak_forward_prices,
            source_normalization_content_id=source_normalization_content_id,
            source_audit_content_id=source_audit_content_id,
            valuation_timestamp_utc=valuation_timestamp_utc,
            country=country,
        )
        projected = component_projection.delta.to_numpy(dtype=float)
        max_abs_projected = float(np.max(np.abs(projected)))
        max_abs_projected_contribution = abs(coefficient) * max_abs_projected
        if max_abs_projected_contribution > contribution_cap:
            raise ValueError(
                "short-tenor projected component contribution breaches its explicit cap: "
                f"{component_key} {max_abs_projected_contribution:.12g} > "
                f"{contribution_cap:.12g} EUR/MWh"
            )
        projected_values[component_key] = projected
        component_rows.append(
            {
                "component_key": component_key,
                "coefficient": coefficient,
                "max_abs_raw_component_eur_mwh": max_abs_raw,
                "max_abs_raw_contribution_eur_mwh": max_abs_raw_contribution,
                "max_abs_projected_component_eur_mwh": max_abs_projected,
                "max_abs_projected_contribution_eur_mwh": (
                    max_abs_projected_contribution
                ),
                "max_constraint_residual_eur_mwh": float(
                    component_projection.audit[
                        "max_constraint_residual_eur_mwh"
                    ]
                ),
                "max_monthly_residual_eur_mwh": float(
                    component_projection.audit[
                        "max_monthly_mean_residual_eur_mwh"
                    ]
                ),
            }
        )

    combined_values = np.zeros(len(first_index), dtype=float)
    for component_key in ordered_components:
        combined_values += (
            ordered_coefficients[component_key] * projected_values[component_key]
        )
    max_abs_combined = float(np.max(np.abs(combined_values)))
    if max_abs_combined > combined_cap:
        raise ValueError(
            "combined short-tenor signal breaches its explicit cap: "
            f"{max_abs_combined:.12g} > {combined_cap:.12g} EUR/MWh"
        )
    pre_final = pd.Series(
        combined_values,
        index=first_index,
        name="short_tenor_combined_pre_final_projection_eur_mwh",
        dtype=float,
    )
    final_projection = project_short_tenor_shape_delta(
        pre_final,
        monthly_base_levels=monthly_base_levels,
        peak_forward_prices=peak_forward_prices,
        source_normalization_content_id=source_normalization_content_id,
        source_audit_content_id=source_audit_content_id,
        valuation_timestamp_utc=valuation_timestamp_utc,
        country=country,
    )
    final_values = final_projection.delta.to_numpy(dtype=float)
    max_abs_final = float(np.max(np.abs(final_values)))
    if max_abs_final > combined_cap:
        raise ValueError(
            "final projected short-tenor signal breaches its explicit cap: "
            f"{max_abs_final:.12g} > {combined_cap:.12g} EUR/MWh"
        )
    linearity_residual = float(
        np.max(np.abs(final_values - combined_values))
    )
    if linearity_residual > MAX_LINEARITY_RESIDUAL_EUR_MWH:
        raise RuntimeError(
            "short-tenor combination is not stable under final D219 projection: "
            f"{linearity_residual:.12g} EUR/MWh"
        )

    component_audit = pd.DataFrame(component_rows, columns=COMPONENT_AUDIT_COLUMNS)
    coefficient_payload = json.dumps(
        ordered_coefficients,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    audit: dict[str, Any] = {
        "schema_version": SHORT_TENOR_COMBINATION_SCHEMA_VERSION,
        "contract_version": SHORT_TENOR_COMBINATION_CONTRACT_VERSION,
        "status": "PASS_LOCAL_MATHEMATICAL_COMBINATION_ONLY_NO_MODEL_AUTHORITY",
        "country": "CH",
        "component_count": len(ordered_components),
        "component_keys": list(ordered_components),
        "coefficient_map_sha256": hashlib.sha256(coefficient_payload).hexdigest(),
        "coefficient_abs_cap": coefficient_cap,
        "per_component_contribution_abs_cap_eur_mwh": contribution_cap,
        "combined_signal_abs_cap_eur_mwh": combined_cap,
        "max_abs_combined_pre_final_projection_eur_mwh": max_abs_combined,
        "max_abs_combined_final_projection_eur_mwh": max_abs_final,
        "max_abs_linearity_residual_eur_mwh": linearity_residual,
        "max_constraint_residual_eur_mwh": float(
            final_projection.audit["max_constraint_residual_eur_mwh"]
        ),
        "max_monthly_mean_residual_eur_mwh": float(
            final_projection.audit["max_monthly_mean_residual_eur_mwh"]
        ),
        **identifiers,
        "coefficients_selected_here": False,
        "caps_selected_here": False,
        "automatic_component_discovery_used": False,
        "silent_clipping_used": False,
        "post_projection_clipping_used": False,
        "hard_rejection_on_cap_breach": True,
        "training_receipt_authenticated": False,
        "selection_receipt_authenticated": False,
        "provenance_content_ids_pairwise_distinct": True,
        "point_in_time_availability_proven": False,
        "rolling_origin_selection_authorized": False,
        "model_input_authorized": False,
        "candidate_assembly_authorized": False,
        "production_authorized": False,
        "ompex_used": False,
        "afry_numeric_values_used": False,
        "t057_used": False,
    }
    return ShortTenorCombinationProjection(
        pre_final_projection_delta=pre_final,
        projection=final_projection,
        component_audit=component_audit,
        audit=audit,
    )


def _validate_components(
    components: Mapping[str, pd.Series],
) -> dict[str, pd.Series]:
    if not isinstance(components, Mapping) or not components:
        raise ValueError("short-tenor components must be a non-empty mapping")
    normalized: dict[str, pd.Series] = {}
    for raw_key, signal in components.items():
        if not isinstance(raw_key, str) or not raw_key or raw_key != raw_key.strip():
            raise ValueError("short-tenor component keys must be non-empty exact strings")
        if not isinstance(signal, pd.Series):
            raise TypeError(f"short-tenor component {raw_key} must be a pandas Series")
        if raw_key in normalized:
            raise ValueError(f"short-tenor component key is duplicated: {raw_key}")
        normalized[raw_key] = signal
    return {key: normalized[key] for key in sorted(normalized)}


def _validate_coefficients(
    coefficients: Mapping[str, float],
    *,
    component_keys: tuple[str, ...],
    coefficient_abs_cap: float,
) -> dict[str, float]:
    if not isinstance(coefficients, Mapping):
        raise TypeError("short-tenor coefficients must be a mapping")
    actual_keys = set(coefficients)
    expected_keys = set(component_keys)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(str(key) for key in actual_keys - expected_keys)
        raise ValueError(
            "short-tenor coefficient keys must exactly match component keys; "
            f"missing={missing}, extra={extra}"
        )
    normalized: dict[str, float] = {}
    for key in component_keys:
        raw_value = coefficients[key]
        if isinstance(raw_value, (bool, np.bool_)):
            raise ValueError(f"short-tenor coefficient is invalid: {key}")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"short-tenor coefficient is invalid: {key}") from exc
        if not math.isfinite(value):
            raise ValueError(f"short-tenor coefficient must be finite: {key}")
        if abs(value) > coefficient_abs_cap:
            raise ValueError(
                "short-tenor coefficient breaches its explicit cap: "
                f"{key} {abs(value):.12g} > {coefficient_abs_cap:.12g}"
            )
        normalized[key] = value
    return normalized


def _positive_finite(value: object, field: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be positive and finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be positive and finite") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be positive and finite")
    return result


def _validate_content_id(value: object, field: str) -> None:
    if _SHA256.fullmatch(str(value)) is None:
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256 content id")


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_equal(value: object, expected: object, label: str) -> None:
    if expected is None or isinstance(expected, bool):
        matches = value is expected
    else:
        matches = value == expected
    if not matches:
        raise ValueError(f"{label} must be {expected!r}")


def _require_true(value: object, label: str) -> None:
    _require_equal(value, True, label)


def _require_false(value: object, label: str) -> None:
    _require_equal(value, False, label)


def _require_string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{label} must contain only non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must be unique")
    return list(value)


__all__ = [
    "ALLOWED_BASELINE_FAMILIES",
    "ALLOWED_CANDIDATE_FAMILIES",
    "COMPONENT_AUDIT_COLUMNS",
    "MAX_LINEARITY_RESIDUAL_EUR_MWH",
    "SELECTED_CONTRAST_BUNDLE_CONTENT_ID",
    "SELECTED_CONTRAST_BUNDLE_DECISION",
    "SELECTED_CONTRAST_BUNDLE_MANIFEST_SHA256",
    "SELECTED_SHAPE_PROJECTION_CONTENT_ID",
    "SHORT_TENOR_COMBINATION_CONTRACT_VERSION",
    "SHORT_TENOR_COMBINATION_SCHEMA_VERSION",
    "SUCCESSOR_CORE_ID",
    "SUCCESSOR_CORE_SHA256",
    "ShortTenorCombinationProjection",
    "combine_short_tenor_shape_components",
    "validate_short_tenor_combination_policy",
]
