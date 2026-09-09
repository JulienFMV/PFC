"""Frozen feature inventory for the governed LT hourly-model comparison.

The inventory separates candidate inputs from shared downstream shaping
context. It defines no connector, materialization, fitting, scoring, or
authority path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

FEATURE_INVENTORY_SCHEMA = "fmv-lt-hourly-feature-inventory.v1"
CANONICAL_FEATURE_NAMES = (
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "is_holiday",
    "hydro_fill",
    "years_ahead",
)
CANONICAL_FEATURE_INVENTORY_SEMANTIC_SHA256 = (
    "83fabacc804de4201560877400fefa6974076f64d14f9738306a86bcf815a6fd"
)


@dataclass(frozen=True, slots=True)
class FeatureInventoryAuthority:
    """Non-overridable negative authority for the feature decision."""

    data_acquisition_authorized: bool = field(default=False, init=False)
    feature_materialization_authorized: bool = field(default=False, init=False)
    model_training_authorized: bool = field(default=False, init=False)
    truth_open_authorized: bool = field(default=False, init=False)
    model_selection_authorized: bool = field(default=False, init=False)
    monthly_level_change_authorized: bool = field(default=False, init=False)
    publication_authorized: bool = field(default=False, init=False)
    production_authorized: bool = field(default=False, init=False)

    def to_manifest(self) -> dict[str, bool]:
        return {
            "data_acquisition_authorized": self.data_acquisition_authorized,
            "feature_materialization_authorized": self.feature_materialization_authorized,
            "model_training_authorized": self.model_training_authorized,
            "truth_open_authorized": self.truth_open_authorized,
            "model_selection_authorized": self.model_selection_authorized,
            "monthly_level_change_authorized": self.monthly_level_change_authorized,
            "publication_authorized": self.publication_authorized,
            "production_authorized": self.production_authorized,
        }


@dataclass(frozen=True, slots=True)
class HourlyFeatureInventory:
    """Exact ordered candidate matrix and explicit non-candidate context."""

    authority: FeatureInventoryAuthority = field(
        default_factory=FeatureInventoryAuthority,
        init=False,
    )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return CANONICAL_FEATURE_NAMES

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": FEATURE_INVENTORY_SCHEMA,
            "status": "FROZEN_HOURLY_FEATURES_NO_EXECUTION_OR_MODEL_AUTHORITY",
            "scope": "LT_HOURLY_CANDIDATE_MATRIX_ONLY",
            "population_policy": "SAME_ORDER_AND_COMPLETE_CASE_ROWS_FOR_ALL_FIVE_CANDIDATES",
            "evidence_bindings": {
                "incumbent_source_normalized_lf_sha256": (
                    "8f199d8075d1cd0e1d0231c3999a819d2663c4ca0a6af8ebb1178b1e6bab4aef"
                ),
                "incumbent_config_normalized_lf_sha256": (
                    "f06bb9d101289e2750f72eae10cd8726aed525645455bf23b4ae524b1f1e972d"
                ),
                "entsoe_feature_availability_contract_content_id": (
                    "c7826b4ad2fa5cdb6baff5d077f00ee5fd8d98108cebef9920c147d787df2ab0"
                ),
            },
            "feature_names": list(self.feature_names),
            "features": [
                {
                    "feature_name": name,
                    "training_information_role": "CALENDAR_KNOWN",
                    "prediction_information_role": "CALENDAR_KNOWN",
                    "training_semantics": "DETERMINISTIC_FROM_DELIVERY_AND_ORIGIN",
                    "prediction_semantics": "DETERMINISTIC_FROM_DELIVERY_AND_ORIGIN",
                }
                for name in self.feature_names[:7]
            ]
            + [
                {
                    "feature_name": "hydro_fill",
                    "unit": "FRACTION_0_1",
                    "training_information_role": "REALIZED_ACTUAL",
                    "prediction_information_role": "ORIGIN_FROZEN_CLIMATOLOGY",
                    "training_semantics": "TARGET_TIME_REALIZED_VALUE_AVAILABLE_BEFORE_ORIGIN",
                    "prediction_semantics": "WEEK_OF_YEAR_CLIMATOLOGY_FIT_STRICTLY_BEFORE_ORIGIN",
                },
                {
                    "feature_name": "years_ahead",
                    "unit": "365_25_DAY_YEARS",
                    "training_information_role": "CALENDAR_KNOWN",
                    "prediction_information_role": "CALENDAR_KNOWN",
                    "training_semantics": "MAX_YEARS_BETWEEN_DELIVERY_AND_ORIGIN_AND_ZERO",
                    "prediction_semantics": "MAX_YEARS_BETWEEN_DELIVERY_AND_ORIGIN_AND_ZERO",
                },
            ],
            "missingness_policy": "PRESERVE_NULL_THEN_ONE_COMMON_COMPLETE_CASE_MASK",
            "incumbent_compatibility_constants": {
                "unavailable_nuclear": 0.0,
                "unavailable_hydro": 0.0,
                "unavailable_thermal": 0.0,
                "meaning": "FEATURE_DISABLED_BY_GOVERNED_CONFIG_NOT_MISSING_VALUE_FILL",
                "candidate_matrix_membership": False,
            },
            "excluded_candidate_features": {
                "raw_entsoe_actuals": [
                    "load_mw",
                    "solar_mw",
                    "wind_mw",
                    "cross_border_mw",
                ],
                "raw_actual_reason": "NO_SAME_TARGET_FUTURE_ACTUAL_AT_ORIGIN",
                "outage_features": [
                    "unavailable_nuclear",
                    "unavailable_hydro",
                    "unavailable_thermal",
                ],
                "outage_reason": "DISABLED_AND_NO_COMPLETE_M01_M36_ORIGIN_AVAILABLE_FORECAST",
            },
            "shared_downstream_context": {
                "candidate_matrix_membership": False,
                "layer": "QUARTER_HOUR_SHAPING_COMMON_TO_EVERY_CANDIDATE",
                "origin_frozen_climatology_features": [
                    "solar_regime",
                    "load_deviation",
                    "flow_deviation",
                ],
                "policy": "SEPARATELY_VALIDATE_IDENTICAL_CONTEXT_AND_CUTOFF_FOR_EVERY_CANDIDATE",
            },
            "monthly_level_authority": "CH_MONTHLY_BASE_SOLVER_ONLY",
            "non_calendar_effect_policy": "ZERO_MEAN_WITHIN_SOLVER_MONTH",
            "real_data_opened": False,
            "model_fit_performed": False,
            "ranking_or_selection_performed": False,
            "warehouse_start_count": 0,
            "gpu_execution_count": 0,
            "authority": self.authority.to_manifest(),
        }

    def semantic_sha256(self) -> str:
        payload = json.dumps(
            self.to_manifest(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def default_hourly_feature_inventory() -> HourlyFeatureInventory:
    """Return the canonical inventory and fail if its semantics drift."""

    inventory = HourlyFeatureInventory()
    observed = inventory.semantic_sha256()
    if observed != CANONICAL_FEATURE_INVENTORY_SEMANTIC_SHA256:
        raise RuntimeError(
            f"canonical LT hourly feature inventory semantic SHA-256 drifted: {observed}"
        )
    return inventory


__all__ = [
    "CANONICAL_FEATURE_INVENTORY_SEMANTIC_SHA256",
    "CANONICAL_FEATURE_NAMES",
    "FEATURE_INVENTORY_SCHEMA",
    "FeatureInventoryAuthority",
    "HourlyFeatureInventory",
    "default_hourly_feature_inventory",
]
