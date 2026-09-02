"""Outcome-blind LT model-evaluation protocol.

The canonical protocol freezes metadata only: candidate families, nested
selection rules, metrics, and a prospective origin schedule.  It cannot load
truth, fit a model, count an origin, select a winner, or change solver-owned
monthly levels.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

EVALUATION_PROTOCOL_VERSION = "fmv-lt-hourly-evaluation-protocol-v1"
EVALUATION_STATUS = "LOCAL_HASH_FROZEN_NOT_EXTERNALLY_REGISTERED_NO_GO"
CANONICAL_PROTOCOL_SEMANTIC_SHA256 = (
    "c2705a8d175bfe7421e2722d316bee4a5eb5631506f284dde03363ab561cb26b"
)
PRIMARY_METRIC = "MONTHLY_LEVEL_NEUTRALIZED_MAE_EUR_MWH"
SECONDARY_METRICS = (
    "MONTHLY_LEVEL_NEUTRALIZED_RMSE_EUR_MWH",
    "MONTHLY_LEVEL_NEUTRALIZED_BIAS_EUR_MWH",
    "P95_ABSOLUTE_ERROR_EUR_MWH",
    "BASE_PEAK_OFFPEAK_WEIGHTED_ERROR_EUR_MWH",
    "FIXED_PROFILE_CAPTURE_PRICE_ERROR_EUR_MWH",
    "BLOC_13_CONTRACT_PAYOFF_ERROR_CHF",
    "HYDRO_DISPATCH_REALIZED_NET_VALUE_OR_REGRET_CHF",
)
LEAD_MONTH_BUCKETS = ("M01_M06", "M07_M12", "M13_M24", "M25_M36")

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
_MONTH = re.compile(r"^(?P<year>[0-9]{4})-(?P<month>0[1-9]|1[0-2])$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CandidateRole(str, Enum):
    INCUMBENT = "incumbent"
    CHALLENGER = "challenger"


class ModelFamily(str, Enum):
    CURRENT_UNWEIGHTED_MLP = "current_unweighted_mlp"
    RECENCY_WEIGHTED_MLP = "recency_weighted_mlp"
    RIDGE = "ridge"
    SPLINE_RIDGE_GAM = "spline_ridge_gam"
    LIGHTGBM = "lightgbm"


class FitPolicy(str, Enum):
    FROZEN_IMPLEMENTATION_PER_ORIGIN = "frozen_implementation_per_origin"
    NESTED_ORIGIN_SELECTION = "nested_origin_selection_no_holdout_tuning"


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One frozen family specification without training authority."""

    candidate_id: str
    role: CandidateRole
    family: ModelFamily
    fit_policy: FitPolicy
    fixed_parameters: tuple[tuple[str, str], ...]
    tuning_grid: tuple[tuple[str, tuple[str, ...]], ...] = ()
    implementation_normalized_lf_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.candidate_id, "candidate_id")
        if not isinstance(self.role, CandidateRole):
            raise TypeError("role must be a CandidateRole")
        if not isinstance(self.family, ModelFamily):
            raise TypeError("family must be a ModelFamily")
        if not isinstance(self.fit_policy, FitPolicy):
            raise TypeError("fit_policy must be a FitPolicy")
        _require_parameter_table(self.fixed_parameters, "fixed_parameters")
        _require_tuning_grid(self.tuning_grid)
        if self.role is CandidateRole.INCUMBENT:
            if self.family is not ModelFamily.CURRENT_UNWEIGHTED_MLP:
                raise ValueError("the incumbent must be current_unweighted_mlp")
            if self.fit_policy is not FitPolicy.FROZEN_IMPLEMENTATION_PER_ORIGIN:
                raise ValueError("the incumbent requires frozen implementation replay")
            _require_sha256(
                self.implementation_normalized_lf_sha256,
                "incumbent implementation",
            )
            if self.tuning_grid:
                raise ValueError("the incumbent cannot tune hyperparameters")
        else:
            if self.family is ModelFamily.CURRENT_UNWEIGHTED_MLP:
                raise ValueError("current_unweighted_mlp is reserved for the incumbent")
            if self.fit_policy is not FitPolicy.NESTED_ORIGIN_SELECTION:
                raise ValueError("challengers require nested-origin selection")
            if self.implementation_normalized_lf_sha256 is not None:
                raise ValueError("unimplemented challengers cannot carry an implementation hash")

    def to_manifest(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "role": self.role.value,
            "family": self.family.value,
            "fit_policy": self.fit_policy.value,
            "fixed_parameters": _parameter_manifest(self.fixed_parameters),
            "tuning_grid": [
                {"name": name, "values": list(values)} for name, values in self.tuning_grid
            ],
            "implementation_normalized_lf_sha256": (self.implementation_normalized_lf_sha256),
            "implementation_hash_basis": "UTF8_NORMALIZED_LF_SHA256",
            "implementation_status": (
                "SOURCE_BOUND_BASELINE"
                if self.implementation_normalized_lf_sha256 is not None
                else "SPECIFICATION_ONLY_IMPLEMENTATION_PENDING"
            ),
            "training_authorized": False,
        }


@dataclass(frozen=True, slots=True)
class OriginSlot:
    """One proposed monthly origin and its complete LT delivery support."""

    slot_id: str
    origin_as_of_utc: datetime
    first_delivery_month: str
    last_delivery_month: str

    def __post_init__(self) -> None:
        _require_identifier(self.slot_id, "slot_id")
        _require_utc(self.origin_as_of_utc, "origin_as_of_utc")
        origin_month = f"{self.origin_as_of_utc.year:04d}-{self.origin_as_of_utc.month:02d}"
        if self.slot_id != f"origin-{origin_month}":
            raise ValueError("slot_id must identify the UTC origin month")
        if self.first_delivery_month != _add_months(origin_month, 1):
            raise ValueError("first_delivery_month must be lead month 1")
        if self.last_delivery_month != _add_months(origin_month, 36):
            raise ValueError("last_delivery_month must be lead month 36")

    def to_manifest(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "origin_as_of_utc": self.origin_as_of_utc.isoformat(),
            "first_delivery_month": self.first_delivery_month,
            "last_delivery_month": self.last_delivery_month,
            "externally_registered": False,
            "countable_origin": False,
            "truth_open_authorized": False,
        }


@dataclass(frozen=True, slots=True)
class FutureHoldout:
    """Locally frozen prospective cohort pending external registration."""

    holdout_id: str
    locally_frozen_at_utc: datetime
    origin_slots: tuple[OriginSlot, ...]
    lead_month_buckets: tuple[str, ...] = LEAD_MONTH_BUCKETS

    def __post_init__(self) -> None:
        _require_identifier(self.holdout_id, "holdout_id")
        _require_utc(self.locally_frozen_at_utc, "locally_frozen_at_utc")
        if self.lead_month_buckets != LEAD_MONTH_BUCKETS:
            raise ValueError("lead-month buckets must match the LT estimand")
        if not isinstance(self.origin_slots, tuple) or any(
            not isinstance(slot, OriginSlot) for slot in self.origin_slots
        ):
            raise TypeError("origin_slots must be a tuple of OriginSlot values")
        if len(self.origin_slots) != 12:
            raise ValueError("the first prospective cohort must contain 12 monthly slots")
        origins = tuple(slot.origin_as_of_utc for slot in self.origin_slots)
        if origins != tuple(sorted(origins)) or len(set(origins)) != len(origins):
            raise ValueError("origin slots must be unique and strictly ordered")
        if origins[0] <= self.locally_frozen_at_utc:
            raise ValueError("every holdout origin must follow the local freeze")
        months = tuple((value.year, value.month) for value in origins)
        expected = tuple(_month_pair_add(months[0], offset) for offset in range(12))
        if months != expected:
            raise ValueError("origin slots must be consecutive calendar months")

    def to_manifest(self) -> dict[str, object]:
        return {
            "holdout_id": self.holdout_id,
            "locally_frozen_at_utc": self.locally_frozen_at_utc.isoformat(),
            "local_freeze_is_external_registration": False,
            "external_registry_protocol": "ch_lt_origin_registry_protocol.v2",
            "external_registration_status": "PENDING",
            "origin_cadence": "MONTHLY",
            "missed_slot_policy": "MISSED_NOT_SHIFTED_NOT_BACKFILLED_NOT_REWEIGHTED",
            "lead_month_buckets": list(self.lead_month_buckets),
            "origin_slots": [slot.to_manifest() for slot in self.origin_slots],
            "scheduled_origin_count": len(self.origin_slots),
            "countable_origin_count": 0,
            "truth_open_authorized": False,
            "holdout_consumed": False,
        }


@dataclass(frozen=True, slots=True)
class ProtocolBindings:
    """Exact local identities reused by the protocol."""

    estimand_sha256: str
    origin_registry_sha256: str
    dependence_power_design_sha256: str
    incumbent_source_normalized_lf_sha256: str
    incumbent_config_normalized_lf_sha256: str

    def __post_init__(self) -> None:
        for name, value in self.to_manifest().items():
            _require_sha256(value, name)

    def to_manifest(self) -> dict[str, str]:
        return {
            "estimand_sha256": self.estimand_sha256,
            "origin_registry_sha256": self.origin_registry_sha256,
            "dependence_power_design_sha256": self.dependence_power_design_sha256,
            "incumbent_source_normalized_lf_sha256": (self.incumbent_source_normalized_lf_sha256),
            "incumbent_config_normalized_lf_sha256": (self.incumbent_config_normalized_lf_sha256),
        }


@dataclass(frozen=True, slots=True)
class EvaluationAuthority:
    """Non-overridable negative authority for a locally frozen protocol."""

    status: str = field(default=EVALUATION_STATUS, init=False)
    data_acquisition_authorized: bool = field(default=False, init=False)
    model_training_authorized: bool = field(default=False, init=False)
    truth_open_authorized: bool = field(default=False, init=False)
    model_selection_authorized: bool = field(default=False, init=False)
    scientific_claim_authorized: bool = field(default=False, init=False)
    monthly_level_change_authorized: bool = field(default=False, init=False)
    publication_authorized: bool = field(default=False, init=False)
    production_authorized: bool = field(default=False, init=False)

    def to_manifest(self) -> dict[str, object]:
        return {
            "status": self.status,
            "data_acquisition_authorized": self.data_acquisition_authorized,
            "model_training_authorized": self.model_training_authorized,
            "truth_open_authorized": self.truth_open_authorized,
            "model_selection_authorized": self.model_selection_authorized,
            "scientific_claim_authorized": self.scientific_claim_authorized,
            "monthly_level_change_authorized": self.monthly_level_change_authorized,
            "publication_authorized": self.publication_authorized,
            "production_authorized": self.production_authorized,
        }


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    """Complete outcome-blind specification for the next LT comparison."""

    protocol_id: str
    locally_frozen_at_utc: datetime
    candidates: tuple[CandidateSpec, ...]
    holdout: FutureHoldout
    bindings: ProtocolBindings
    primary_metric: str = PRIMARY_METRIC
    secondary_metrics: tuple[str, ...] = SECONDARY_METRICS
    authority: EvaluationAuthority = field(default_factory=EvaluationAuthority, init=False)

    def __post_init__(self) -> None:
        _require_identifier(self.protocol_id, "protocol_id")
        _require_utc(self.locally_frozen_at_utc, "locally_frozen_at_utc")
        if self.holdout.locally_frozen_at_utc != self.locally_frozen_at_utc:
            raise ValueError("protocol and holdout local-freeze timestamps must match")
        if self.primary_metric != PRIMARY_METRIC or self.secondary_metrics != SECONDARY_METRICS:
            raise ValueError("metrics must match the frozen LT estimand")
        if not isinstance(self.candidates, tuple) or any(
            not isinstance(candidate, CandidateSpec) for candidate in self.candidates
        ):
            raise TypeError("candidates must be a tuple of CandidateSpec values")
        if len(self.candidates) != 5:
            raise ValueError("the protocol requires one incumbent and four challengers")
        ids = tuple(item.candidate_id for item in self.candidates)
        families = tuple(item.family for item in self.candidates)
        if len(set(ids)) != len(ids) or len(set(families)) != len(families):
            raise ValueError("candidate ids and families must be unique")
        if sum(item.role is CandidateRole.INCUMBENT for item in self.candidates) != 1:
            raise ValueError("the protocol requires exactly one incumbent")
        if set(families) != set(ModelFamily):
            raise ValueError("the candidate-family inventory is not exact")
        if "t057" in json.dumps(self.to_manifest(), sort_keys=True).lower():
            raise ValueError("T057 must not enter the new evaluation protocol")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "contract_version": EVALUATION_PROTOCOL_VERSION,
            "protocol_id": self.protocol_id,
            "market": "CH",
            "delivery_grain": "quarter_hour",
            "locally_frozen_at_utc": self.locally_frozen_at_utc.isoformat(),
            "monthly_level_authority": "CH_MONTHLY_BASE_SOLVER_UNCHANGED",
            "comparison_rows": "SAME_COMPLETE_CASE_INTERSECTION_FOR_ALL_CANDIDATES",
            "origin_weighting": "EQUAL_AFTER_WITHIN_ORIGIN_ENERGY_WEIGHTING",
            "selection_scope": "NESTED_REGISTERED_DEVELOPMENT_ORIGINS_ONLY",
            "future_holdout_tuning": "FORBIDDEN",
            "primary_metric": self.primary_metric,
            "secondary_metrics": list(self.secondary_metrics),
            "market_consistency_gates": "ABSOLUTE_HARD_GATES_NEVER_COMPENSATED",
            "decision_margins_status": "PENDING_FMV_RISK_MDE_AND_POWER_DESIGN",
            "insufficient_power_or_coverage": "UNSUPPORTED_NEVER_PASS",
            "candidates": [candidate.to_manifest() for candidate in self.candidates],
            "future_holdout": self.holdout.to_manifest(),
            "bindings": self.bindings.to_manifest(),
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


def default_evaluation_protocol() -> EvaluationProtocol:
    """Return the canonical local specification; construction performs validation."""

    frozen_at = datetime(2026, 9, 2, tzinfo=timezone.utc)
    candidates = (
        CandidateSpec(
            candidate_id="current-unweighted-mlp",
            role=CandidateRole.INCUMBENT,
            family=ModelFamily.CURRENT_UNWEIGHTED_MLP,
            fit_policy=FitPolicy.FROZEN_IMPLEMENTATION_PER_ORIGIN,
            fixed_parameters=(
                ("config_mode", "mlp"),
                ("source_identity", "utf8_normalized_lf_module_and_config_sha256"),
            ),
            implementation_normalized_lf_sha256=(
                "8f199d8075d1cd0e1d0231c3999a819d2663c4ca0a6af8ebb1178b1e6bab4aef"
            ),
        ),
        CandidateSpec(
            candidate_id="recency-weighted-mlp",
            role=CandidateRole.CHALLENGER,
            family=ModelFamily.RECENCY_WEIGHTED_MLP,
            fit_policy=FitPolicy.NESTED_ORIGIN_SELECTION,
            fixed_parameters=(
                ("architecture", "64,64-relu"),
                ("half_life_days", "180"),
                ("random_state", "42"),
                ("sample_weight", "observation_level_exponential_decay"),
            ),
        ),
        CandidateSpec(
            candidate_id="ridge-linear",
            role=CandidateRole.CHALLENGER,
            family=ModelFamily.RIDGE,
            fit_policy=FitPolicy.NESTED_ORIGIN_SELECTION,
            fixed_parameters=(
                ("feature_policy", "same_frozen_origin_features"),
                ("fit_intercept", "true"),
                ("scaling", "standardize_on_training_origins_only"),
            ),
            tuning_grid=(("alpha", ("0.01", "0.1", "1", "10", "100")),),
        ),
        CandidateSpec(
            candidate_id="spline-ridge-gam",
            role=CandidateRole.CHALLENGER,
            family=ModelFamily.SPLINE_RIDGE_GAM,
            fit_policy=FitPolicy.NESTED_ORIGIN_SELECTION,
            fixed_parameters=(
                ("basis", "cubic_spline_additive_no_interactions"),
                ("feature_policy", "same_frozen_origin_features"),
                ("scaling", "fit_on_training_origins_only"),
            ),
            tuning_grid=(
                ("alpha", ("0.01", "0.1", "1", "10", "100")),
                ("n_knots", ("5", "8", "12")),
            ),
        ),
        CandidateSpec(
            candidate_id="lightgbm-deterministic-cpu",
            role=CandidateRole.CHALLENGER,
            family=ModelFamily.LIGHTGBM,
            fit_policy=FitPolicy.NESTED_ORIGIN_SELECTION,
            fixed_parameters=(
                ("deterministic", "true"),
                ("device", "cpu"),
                ("feature_policy", "same_frozen_origin_features"),
                ("objective", "regression_l1"),
                ("seed", "42"),
            ),
            tuning_grid=(
                ("learning_rate", ("0.03", "0.05")),
                ("min_data_in_leaf", ("50", "100")),
                ("n_estimators", ("300", "600")),
                ("num_leaves", ("15", "31")),
            ),
        ),
    )
    origin_dates = (
        (2026, 10, 6),
        (2026, 11, 3),
        (2026, 12, 1),
        (2027, 1, 5),
        (2027, 2, 2),
        (2027, 3, 2),
        (2027, 4, 6),
        (2027, 5, 4),
        (2027, 6, 1),
        (2027, 7, 6),
        (2027, 8, 3),
        (2027, 9, 7),
    )
    slots = tuple(_origin_slot(*parts) for parts in origin_dates)
    protocol = EvaluationProtocol(
        protocol_id="ch-lt-hourly-challengers-2026-v1",
        locally_frozen_at_utc=frozen_at,
        candidates=candidates,
        holdout=FutureHoldout(
            holdout_id="ch-lt-future-cohort-2026-10-v1",
            locally_frozen_at_utc=frozen_at,
            origin_slots=slots,
        ),
        bindings=ProtocolBindings(
            estimand_sha256=("4209931e28a7c1cf2a4224d779f73648c4c9c5eac55df0a7ba1ad872226e2931"),
            origin_registry_sha256=(
                "6ea896ccdb35414b52237f2bcf1065755c3c10444b308ce905b60f472e68c697"
            ),
            dependence_power_design_sha256=(
                "005b8655b817db10e7f3c227b1c5912d545305b68e262ab82d9aa0b5817a6a91"
            ),
            incumbent_source_normalized_lf_sha256=(
                "8f199d8075d1cd0e1d0231c3999a819d2663c4ca0a6af8ebb1178b1e6bab4aef"
            ),
            incumbent_config_normalized_lf_sha256=(
                "f06bb9d101289e2750f72eae10cd8726aed525645455bf23b4ae524b1f1e972d"
            ),
        ),
    )
    observed_hash = protocol.semantic_sha256()
    if observed_hash != CANONICAL_PROTOCOL_SEMANTIC_SHA256:
        raise RuntimeError(f"canonical evaluation protocol semantic hash changed: {observed_hash}")
    return protocol


def _origin_slot(year: int, month: int, day: int) -> OriginSlot:
    origin_month = f"{year:04d}-{month:02d}"
    return OriginSlot(
        slot_id=f"origin-{origin_month}",
        origin_as_of_utc=datetime(year, month, day, 12, tzinfo=timezone.utc),
        first_delivery_month=_add_months(origin_month, 1),
        last_delivery_month=_add_months(origin_month, 36),
    )


def _require_parameter_table(value: tuple[tuple[str, str], ...], label: str) -> None:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} must be a non-empty tuple")
    if any(not isinstance(item, tuple) or len(item) != 2 for item in value):
        raise ValueError(f"{label} entries must be name/value pairs")
    names = tuple(item[0] for item in value)
    if names != tuple(sorted(names)) or len(set(names)) != len(names):
        raise ValueError(f"{label} names must be unique and sorted")
    if any(
        not isinstance(name, str)
        or not _IDENTIFIER.fullmatch(name)
        or not isinstance(setting, str)
        or not setting
        for name, setting in value
    ):
        raise ValueError(f"{label} entries must contain valid names and non-empty string values")


def _require_tuning_grid(value: tuple[tuple[str, tuple[str, ...]], ...]) -> None:
    if not isinstance(value, tuple):
        raise ValueError("tuning_grid must be a tuple")
    if any(not isinstance(item, tuple) or len(item) != 2 for item in value):
        raise ValueError("tuning-grid entries must be name/choices pairs")
    names = tuple(item[0] for item in value)
    if names != tuple(sorted(names)) or len(set(names)) != len(names):
        raise ValueError("tuning-grid names must be unique and sorted")
    for name, choices in value:
        if not _IDENTIFIER.fullmatch(name) or not isinstance(choices, tuple) or not choices:
            raise ValueError("each tuning dimension requires a valid name and choices")
        if any(not isinstance(choice, str) or not choice for choice in choices):
            raise ValueError("tuning choices must be non-empty strings")


def _parameter_manifest(value: tuple[tuple[str, str], ...]) -> list[dict[str, str]]:
    return [{"name": name, "value": setting} for name, setting in value]


def _require_identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase identifier of 3-128 characters")


def _require_sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _require_utc(value: datetime, label: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware UTC")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must use UTC")


def _add_months(value: str, count: int) -> str:
    match = _MONTH.fullmatch(value)
    if match is None:
        raise ValueError("month must use YYYY-MM")
    ordinal = int(match["year"]) * 12 + int(match["month"]) - 1 + count
    year, zero_based_month = divmod(ordinal, 12)
    return f"{year:04d}-{zero_based_month + 1:02d}"


def _month_pair_add(value: tuple[int, int], count: int) -> tuple[int, int]:
    result = _add_months(f"{value[0]:04d}-{value[1]:02d}", count)
    return int(result[:4]), int(result[5:])


__all__ = [
    "CANONICAL_PROTOCOL_SEMANTIC_SHA256",
    "CandidateRole",
    "CandidateSpec",
    "EVALUATION_PROTOCOL_VERSION",
    "EVALUATION_STATUS",
    "EvaluationAuthority",
    "EvaluationProtocol",
    "FitPolicy",
    "FutureHoldout",
    "LEAD_MONTH_BUCKETS",
    "ModelFamily",
    "OriginSlot",
    "PRIMARY_METRIC",
    "ProtocolBindings",
    "SECONDARY_METRICS",
    "default_evaluation_protocol",
]
