"""Authority-negative execution contract for the LT hourly comparison.

This module records how the source-bound incumbent and the four generic
challengers can share one estimand without pretending that they share one fit
interface.  It performs no I/O, target construction, fitting, prediction, or
scoring.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from pfc_shaping.lt.evaluation_feature_inventory import (
    default_hourly_feature_inventory,
)
from pfc_shaping.lt.evaluation_protocol import default_evaluation_protocol

EXECUTION_CONTRACT_SCHEMA = "fmv-lt-hourly-candidate-execution.v1"
CANONICAL_EXECUTION_CONTRACT_SEMANTIC_SHA256 = (
    "91129fd87a945050d11f724461fa6677726a56dfd8da366143ac7fe61f3cf976"
)


@dataclass(frozen=True, slots=True)
class CandidateExecutionAuthority:
    """Non-overridable negative authority for the interface decision."""

    data_acquisition_authorized: bool = field(default=False, init=False)
    target_materialization_authorized: bool = field(default=False, init=False)
    model_training_authorized: bool = field(default=False, init=False)
    truth_open_authorized: bool = field(default=False, init=False)
    model_selection_authorized: bool = field(default=False, init=False)
    monthly_level_change_authorized: bool = field(default=False, init=False)
    publication_authorized: bool = field(default=False, init=False)
    production_authorized: bool = field(default=False, init=False)

    def to_manifest(self) -> dict[str, bool]:
        return {
            "data_acquisition_authorized": self.data_acquisition_authorized,
            "target_materialization_authorized": self.target_materialization_authorized,
            "model_training_authorized": self.model_training_authorized,
            "truth_open_authorized": self.truth_open_authorized,
            "model_selection_authorized": self.model_selection_authorized,
            "monthly_level_change_authorized": self.monthly_level_change_authorized,
            "publication_authorized": self.publication_authorized,
            "production_authorized": self.production_authorized,
        }


@dataclass(frozen=True, slots=True)
class CandidateExecutionContract:
    """Exact native-incumbent/common-estimand execution decision."""

    authority: CandidateExecutionAuthority = field(
        default_factory=CandidateExecutionAuthority,
        init=False,
    )

    def to_manifest(self) -> dict[str, object]:
        protocol = default_evaluation_protocol()
        inventory = default_hourly_feature_inventory()
        return {
            "schema_version": EXECUTION_CONTRACT_SCHEMA,
            "status": "FROZEN_INTERFACE_DECISION_NO_EXECUTION_AUTHORITY",
            "execution_path": ("NATIVE_INCUMBENT_REPLAY_WITH_COMMON_CHALLENGER_OBSERVATIONS"),
            "evidence_bindings": {
                "evaluation_protocol_semantic_sha256": protocol.semantic_sha256(),
                "feature_inventory_semantic_sha256": inventory.semantic_sha256(),
                "incumbent_source_normalized_lf_sha256": (
                    protocol.bindings.incumbent_source_normalized_lf_sha256
                ),
                "incumbent_config_normalized_lf_sha256": (
                    protocol.bindings.incumbent_config_normalized_lf_sha256
                ),
            },
            "incumbent_adapter": {
                "candidate_id": "current-unweighted-mlp",
                "fit_interface": "NATIVE_SHAPE_HOURLY_MLP_FIT",
                "prediction_interface": "NATIVE_SHAPE_HOURLY_MLP_APPLY",
                "generic_surrogate_allowed": False,
                "source_hash_reverification_required": True,
            },
            "challenger_adapter": {
                "candidate_ids": [
                    "recency-weighted-mlp",
                    "ridge-linear",
                    "spline-ridge-gam",
                    "lightgbm-deterministic-cpu",
                ],
                "fit_interface": "COMMON_NINE_COLUMN_MATRIX",
                "prediction_interface": "COMMON_NINE_COLUMN_MATRIX",
                "synthetic_lab_is_real_data_runner": False,
            },
            "learning_estimand": {
                "name": "INCUMBENT_EQUIVALENT_HOURLY_F_H",
                "raw_source": "DIRECT_CH_QUARTER_HOUR_PRICE_EUR_MWH",
                "local_day_timezone": "Europe/Zurich",
                "daily_mean_rule": "ARITHMETIC_MEAN_AFTER_NATIVE_CALENDAR_JOIN",
                "eligible_day_rule": "DAILY_MEAN_STRICTLY_GREATER_THAN_5_EUR_MWH",
                "quarter_hour_transform": "PRICE_DIVIDED_BY_LOCAL_DAY_MEAN",
                "quarter_hour_clip": [0.2, 3.0],
                "training_group": "SWISS_LOCAL_DATE_AND_CLOCK_HOUR",
                "repeated_fallback_hour_policy": "MERGE_TO_MATCH_SOURCE_BOUND_INCUMBENT",
                "within_group_aggregation": "INCUMBENT_RECENCY_WEIGHTED_MEAN",
                "direct_raw_price_fit_allowed": False,
                "target_column": "target_f_h",
            },
            "prediction_estimand": {
                "raw_model_output": "F_H",
                "delivery_rows": "NATIVE_QUARTER_HOUR_UTC_GRID",
                "common_postprocessing": [
                    "POSITIVE_FLOOR_0_1",
                    "SWISS_LOCAL_DAY_ARITHMETIC_MEAN_NORMALIZATION",
                    "FINAL_CLIP_0_4_2_0",
                ],
                "postprocessing_applied_exactly_once": True,
            },
            "scoring_boundary": {
                "candidate_output_before_assembly": "F_H_NOT_EUR_MWH",
                "common_curve_assembly_required": True,
                "scoring_input": "FULL_PRICE_EUR_MWH",
                "scoring_transform": "SEPARATE_ENERGY_WEIGHTED_LOCAL_MONTH_CENTERING",
                "monthly_level_authority": "CH_MONTHLY_BASE_SOLVER_UNCHANGED",
                "common_downstream_layers_required": True,
            },
            "implementation_sequence": [
                "BUILD_AND_VERIFY_INCUMBENT_EQUIVALENT_TRAINING_OBSERVATIONS",
                "VERIFY_FEATURE_AND_TARGET_PARITY_WITH_NATIVE_INCUMBENT",
                "ADD_REAL_DATA_ADAPTER_SEPARATE_FROM_SYNTHETIC_LAB",
                "ADD_COMMON_F_H_POSTPROCESSOR_FOR_CHALLENGERS",
                "ADD_COMMON_CURVE_ASSEMBLY_AND_ONLY_THEN_SCORE_EUR_MWH",
            ],
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


def default_candidate_execution_contract() -> CandidateExecutionContract:
    """Return the canonical decision and fail closed on semantic drift."""

    contract = CandidateExecutionContract()
    observed = contract.semantic_sha256()
    if observed != CANONICAL_EXECUTION_CONTRACT_SEMANTIC_SHA256:
        raise RuntimeError(f"canonical LT execution contract semantic SHA-256 drifted: {observed}")
    return contract


__all__ = [
    "CANONICAL_EXECUTION_CONTRACT_SEMANTIC_SHA256",
    "EXECUTION_CONTRACT_SCHEMA",
    "CandidateExecutionAuthority",
    "CandidateExecutionContract",
    "default_candidate_execution_contract",
]
