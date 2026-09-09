"""Outcome-blind primary reference for the governed LT model comparison.

The reference is deliberately outside the five-model candidate inventory. It
freezes comparison semantics only; it cannot load data, fit, score, rank, or
grant authority.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from pfc_shaping.lt.evaluation_protocol import (
    LEAD_MONTH_BUCKETS,
    PRIMARY_METRIC,
    SECONDARY_METRICS,
)

REFERENCE_BENCHMARK_SCHEMA = "fmv-lt-primary-reference-benchmark.v1"
CANONICAL_REFERENCE_SEMANTIC_SHA256 = (
    "a0b1dd1f8add11086b1dba8e1b447377388b748a2221e9edd6029092001d0418"
)


@dataclass(frozen=True, slots=True)
class ReferenceAuthority:
    """Non-overridable negative authority for the local reference contract."""

    data_acquisition_authorized: bool = field(default=False, init=False)
    real_data_training_authorized: bool = field(default=False, init=False)
    real_truth_open_authorized: bool = field(default=False, init=False)
    candidate_selection_authorized: bool = field(default=False, init=False)
    model_selection_authorized: bool = field(default=False, init=False)
    monthly_level_change_authorized: bool = field(default=False, init=False)
    publication_authorized: bool = field(default=False, init=False)
    production_authorized: bool = field(default=False, init=False)

    def to_manifest(self) -> dict[str, bool]:
        return {
            "data_acquisition_authorized": self.data_acquisition_authorized,
            "real_data_training_authorized": self.real_data_training_authorized,
            "real_truth_open_authorized": self.real_truth_open_authorized,
            "candidate_selection_authorized": self.candidate_selection_authorized,
            "model_selection_authorized": self.model_selection_authorized,
            "monthly_level_change_authorized": self.monthly_level_change_authorized,
            "publication_authorized": self.publication_authorized,
            "production_authorized": self.production_authorized,
        }


@dataclass(frozen=True, slots=True)
class SeasonalReferenceBenchmark:
    """Placement and scoring contract for the transparent seasonal reference."""

    benchmark_id: str = field(default="market-constrained-seasonal-reference-v1", init=False)
    authority: ReferenceAuthority = field(default_factory=ReferenceAuthority, init=False)

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": REFERENCE_BENCHMARK_SCHEMA,
            "benchmark_id": self.benchmark_id,
            "role": "PRIMARY_PROMOTION_REFERENCE",
            "definition_status": "PLACEMENT_FROZEN_IMPLEMENTATION_PENDING",
            "candidate_inventory_membership": False,
            "candidate_selection_eligible": False,
            "comparison_population": "SAME_ORIGINS_ROWS_MASKS_AND_WEIGHTS_AS_CANDIDATES",
            "scoring_lane": "SEPARATE_REFERENCE_SCORE_BEFORE_CANDIDATE_RANKING",
            "fit_policy": "FROZEN_TRANSPARENT_FORMULA_NO_HYPERPARAMETER_TUNING",
            "feature_family": [
                "SWISS_LOCAL_SEASON",
                "CH_DAY_TYPE",
                "SWISS_LOCAL_HOUR",
            ],
            "level_authority": "CH_MONTHLY_BASE_SOLVER_ONLY",
            "shape_constraint": "ENERGY_WEIGHTED_ZERO_MEAN_WITHIN_SOLVER_MONTH",
            "primary_metric": PRIMARY_METRIC,
            "secondary_metrics": list(SECONDARY_METRICS),
            "lead_month_buckets": list(LEAD_MONTH_BUCKETS),
            "promotion_target": ("CANDIDATE_WEIGHTED_MAE_AND_RMSE_IMPROVEMENT_AT_LEAST_2_PERCENT"),
            "runtime_policy": {
                "reference_backend": "CPU_FLOAT64",
                "gpu_acceleration_applicable_to_reference": False,
                "canonical_lightgbm_backend": "CPU_DETERMINISTIC",
                "gpu_available_for_qualified_nonlinear_work": True,
                "gpu_fit_or_model_selection_status": (
                    "REQUIRES_SEPARATE_CPU_GPU_PARITY_QUALIFICATION"
                ),
                "gpu_does_not_replace_cpu_hard_gate_oracle": True,
            },
            "real_data_training_performed": False,
            "real_truth_opened": False,
            "ranking_or_selection_performed": False,
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


def default_primary_reference_benchmark() -> SeasonalReferenceBenchmark:
    """Return the canonical reference contract and verify its semantic identity."""

    reference = SeasonalReferenceBenchmark()
    observed = reference.semantic_sha256()
    if observed != CANONICAL_REFERENCE_SEMANTIC_SHA256:
        raise RuntimeError(f"canonical primary reference semantic hash changed: {observed}")
    return reference


__all__ = [
    "CANONICAL_REFERENCE_SEMANTIC_SHA256",
    "REFERENCE_BENCHMARK_SCHEMA",
    "ReferenceAuthority",
    "SeasonalReferenceBenchmark",
    "default_primary_reference_benchmark",
]
