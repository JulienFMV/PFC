"""Authority-negative definitions for LT scenarios and CH curve products.

This module contains metadata contracts only.  It does not load data, fit a
model, assemble prices, publish artifacts, assign scenario probabilities, or
grant monthly-level authority.  A required level source is a lineage
requirement, never an authority grant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

SCENARIO_CONTRACT_VERSION = "fmv-lt-scenario-definition-v1"
CURVE_PRODUCT_CONTRACT_VERSION = "fmv-lt-curve-product-definition-v1"
AUTHORITY_STATUS = "CONTRACT_ONLY_NO_GO_MODEL_PUBLICATION_PRODUCTION_OR_TRADE"

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ScenarioAxis(str, Enum):
    """Admitted qualitative axes; no member carries a probability or value."""

    HYDRO = "hydro"
    WEATHER = "weather"
    PV_PENETRATION = "pv_penetration"
    ELECTRIFICATION = "electrification"
    CAPACITY = "capacity"
    OUTAGE = "outage"
    CONGESTION = "congestion"
    CROSS_BORDER = "cross_border"


class ScenarioLevelEffect(str, Enum):
    """Whether a scenario preserves or changes solver-owned monthly levels."""

    SHAPE_ONLY = "shape_only"
    LEVEL_CHANGING = "level_changing"


class NormalizationBucket(str, Enum):
    """Bucket in which a scenario must remain neutral or be solved separately."""

    SOLVER_MONTH = "solver_month"
    SEPARATE_UPSTREAM_LEVEL_SOLVE = "separate_upstream_level_solve"


class RequiredLevelSource(str, Enum):
    """Required source of levels; these values are references, not grants."""

    CH_MONTHLY_BASE_SOLVER = "ch_monthly_base_solver"
    SEPARATE_UPSTREAM_FUNDAMENTAL_SOLVE = "separate_upstream_fundamental_solve"


class CurveProductType(str, Enum):
    """The three price-curve products, kept separate from decision products."""

    MARKET_CENTRAL_CH = "market_central_ch"
    FUNDAMENTAL_SCENARIO_CH = "fundamental_scenario_ch"
    STOCHASTIC_SPOT_PATHS_CH = "stochastic_spot_paths_ch"


@dataclass(frozen=True, slots=True)
class AuthorityNegative:
    """Non-overridable proof that a definition grants no operational authority."""

    status: str = field(default=AUTHORITY_STATUS, init=False)
    scenario_probability_authorized: bool = field(default=False, init=False)
    calendar_mapping_authorized: bool = field(default=False, init=False)
    monthly_level_authority_granted: bool = field(default=False, init=False)
    model_input_authorized: bool = field(default=False, init=False)
    model_selection_authorized: bool = field(default=False, init=False)
    candidate_assembly_authorized: bool = field(default=False, init=False)
    publication_authorized: bool = field(default=False, init=False)
    production_authorized: bool = field(default=False, init=False)
    trade_execution_authorized: bool = field(default=False, init=False)

    def to_manifest(self) -> dict[str, object]:
        return {
            "status": self.status,
            "scenario_probability_authorized": self.scenario_probability_authorized,
            "calendar_mapping_authorized": self.calendar_mapping_authorized,
            "monthly_level_authority_granted": self.monthly_level_authority_granted,
            "model_input_authorized": self.model_input_authorized,
            "model_selection_authorized": self.model_selection_authorized,
            "candidate_assembly_authorized": self.candidate_assembly_authorized,
            "publication_authorized": self.publication_authorized,
            "production_authorized": self.production_authorized,
            "trade_execution_authorized": self.trade_execution_authorized,
        }


@dataclass(frozen=True, slots=True)
class ProvenanceReference:
    """Content-addressed, point-in-time lineage without source values."""

    source_id: str
    content_sha256: str
    available_at_utc: datetime

    def __post_init__(self) -> None:
        _require_identifier(self.source_id, "source_id")
        if not _SHA256.fullmatch(self.content_sha256):
            raise ValueError("content_sha256 must be a lowercase 64-character SHA-256")
        _require_utc(self.available_at_utc, "available_at_utc")

    def to_manifest(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "content_sha256": self.content_sha256,
            "available_at_utc": self.available_at_utc.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    """A qualitative CH scenario identity with explicit level semantics."""

    scenario_id: str
    label: str
    information_timestamp_utc: datetime
    axes: tuple[ScenarioAxis, ...]
    level_effect: ScenarioLevelEffect
    normalization_bucket: NormalizationBucket
    provenance: tuple[ProvenanceReference, ...]
    required_upstream_solve_id: str | None = None
    probability_weight: float | None = None
    authority: AuthorityNegative = field(default_factory=AuthorityNegative, init=False)

    def __post_init__(self) -> None:
        _require_identifier(self.scenario_id, "scenario_id")
        _require_label(self.label, "label")
        _require_utc(self.information_timestamp_utc, "information_timestamp_utc")
        _require_axes(self.axes)
        _require_provenance(self.provenance, self.information_timestamp_utc)
        if self.probability_weight is not None:
            raise ValueError("scenario probability weights are forbidden by this contract")

        if self.level_effect is ScenarioLevelEffect.SHAPE_ONLY:
            if self.normalization_bucket is not NormalizationBucket.SOLVER_MONTH:
                raise ValueError("shape-only scenarios must normalize within solver_month")
            if self.required_upstream_solve_id is not None:
                raise ValueError("shape-only scenarios cannot declare an upstream level solve")
        elif self.level_effect is ScenarioLevelEffect.LEVEL_CHANGING:
            if self.normalization_bucket is not NormalizationBucket.SEPARATE_UPSTREAM_LEVEL_SOLVE:
                raise ValueError("level-changing scenarios require a separate_upstream_level_solve")
            if self.required_upstream_solve_id is None:
                raise ValueError("level-changing scenarios require required_upstream_solve_id")
            _require_identifier(self.required_upstream_solve_id, "required_upstream_solve_id")
        else:  # pragma: no cover - defensive against untyped callers
            raise TypeError("level_effect must be a ScenarioLevelEffect")

    def to_manifest(self) -> dict[str, object]:
        return {
            "contract_version": SCENARIO_CONTRACT_VERSION,
            "scenario_id": self.scenario_id,
            "label": self.label,
            "market": "CH",
            "information_timestamp_utc": self.information_timestamp_utc.isoformat(),
            "axes": [axis.value for axis in self.axes],
            "level_effect": self.level_effect.value,
            "normalization_bucket": self.normalization_bucket.value,
            "required_upstream_solve_id": self.required_upstream_solve_id,
            "probability_weight": None,
            "provenance": [source.to_manifest() for source in self.provenance],
            "authority": self.authority.to_manifest(),
        }


@dataclass(frozen=True, slots=True)
class CurveProductDefinition:
    """Typed CH curve lineage with fail-closed scenario and level relationships."""

    product_id: str
    product_type: CurveProductType
    information_timestamp_utc: datetime
    required_level_source: RequiredLevelSource
    normalization_bucket: NormalizationBucket
    provenance: tuple[ProvenanceReference, ...]
    scenario: ScenarioDefinition | None = None
    parent_product: CurveProductDefinition | None = None
    authority: AuthorityNegative = field(default_factory=AuthorityNegative, init=False)

    def __post_init__(self) -> None:
        _require_identifier(self.product_id, "product_id")
        _require_utc(self.information_timestamp_utc, "information_timestamp_utc")
        _require_provenance(self.provenance, self.information_timestamp_utc)

        if self.product_type is CurveProductType.MARKET_CENTRAL_CH:
            self._validate_market_central()
        elif self.product_type is CurveProductType.FUNDAMENTAL_SCENARIO_CH:
            self._validate_fundamental_scenario()
        elif self.product_type is CurveProductType.STOCHASTIC_SPOT_PATHS_CH:
            self._validate_stochastic_paths()
        else:  # pragma: no cover - defensive against untyped callers
            raise TypeError("product_type must be a CurveProductType")

    def _validate_market_central(self) -> None:
        if self.scenario is not None or self.parent_product is not None:
            raise ValueError("market_central_ch cannot carry a scenario or parent product")
        if self.required_level_source is not RequiredLevelSource.CH_MONTHLY_BASE_SOLVER:
            raise ValueError("market_central_ch requires the CH monthly BASE solver level source")
        if self.normalization_bucket is not NormalizationBucket.SOLVER_MONTH:
            raise ValueError("market_central_ch requires solver_month normalization")

    def _validate_fundamental_scenario(self) -> None:
        scenario = self._require_scenario()
        if scenario.level_effect is ScenarioLevelEffect.SHAPE_ONLY:
            if self.required_level_source is not RequiredLevelSource.CH_MONTHLY_BASE_SOLVER:
                raise ValueError(
                    "shape-only scenario curves must retain the CH solver level source"
                )
            if self.normalization_bucket is not NormalizationBucket.SOLVER_MONTH:
                raise ValueError("shape-only scenario curves must normalize within solver_month")
            if (
                self.parent_product is None
                or self.parent_product.product_type is not CurveProductType.MARKET_CENTRAL_CH
            ):
                raise ValueError("shape-only scenario curves require a market_central_ch parent")
            self._require_parent_not_newer()
        else:
            if (
                self.required_level_source
                is not RequiredLevelSource.SEPARATE_UPSTREAM_FUNDAMENTAL_SOLVE
            ):
                raise ValueError(
                    "level-changing scenario curves require a separate upstream level solve"
                )
            if self.normalization_bucket is not NormalizationBucket.SEPARATE_UPSTREAM_LEVEL_SOLVE:
                raise ValueError(
                    "level-changing scenario curves cannot use a solver-month shape bucket"
                )
            if self.parent_product is not None:
                raise ValueError(
                    "level-changing scenario curves cannot mutate or descend from market_central_ch"
                )

    def _validate_stochastic_paths(self) -> None:
        scenario = self._require_scenario()
        parent = self.parent_product
        if parent is None or parent.product_type is not CurveProductType.FUNDAMENTAL_SCENARIO_CH:
            raise ValueError("stochastic_spot_paths_ch requires a fundamental_scenario_ch parent")
        if parent.scenario is None or parent.scenario.scenario_id != scenario.scenario_id:
            raise ValueError(
                "stochastic paths and their parent must use the same scenario identity"
            )
        if parent.required_level_source is not self.required_level_source:
            raise ValueError("stochastic paths must inherit the parent's required level source")
        if parent.normalization_bucket is not self.normalization_bucket:
            raise ValueError("stochastic paths must inherit the parent's normalization bucket")
        self._require_parent_not_newer()

    def _require_scenario(self) -> ScenarioDefinition:
        if self.scenario is None:
            raise ValueError(f"{self.product_type.value} requires a scenario definition")
        if self.scenario.information_timestamp_utc > self.information_timestamp_utc:
            raise ValueError("scenario information cannot postdate the curve product")
        return self.scenario

    def _require_parent_not_newer(self) -> None:
        if (
            self.parent_product is not None
            and self.parent_product.information_timestamp_utc > self.information_timestamp_utc
        ):
            raise ValueError("parent product information cannot postdate its child")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "contract_version": CURVE_PRODUCT_CONTRACT_VERSION,
            "product_id": self.product_id,
            "product_type": self.product_type.value,
            "market": "CH",
            "delivery_grain": "quarter_hour",
            "information_timestamp_utc": self.information_timestamp_utc.isoformat(),
            "scenario_id": self.scenario.scenario_id if self.scenario is not None else None,
            "parent_product_id": (
                self.parent_product.product_id if self.parent_product is not None else None
            ),
            "required_level_source": self.required_level_source.value,
            "required_level_source_is_authority_grant": False,
            "normalization_bucket": self.normalization_bucket.value,
            "provenance": [source.to_manifest() for source in self.provenance],
            "authority": self.authority.to_manifest(),
        }


def _require_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase identifier of 3-128 characters")


def _require_label(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty and trimmed")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field_name} must not contain control characters")


def _require_utc(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field_name} must be a timezone-aware UTC datetime")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must use UTC")


def _require_axes(axes: tuple[ScenarioAxis, ...]) -> None:
    if not isinstance(axes, tuple) or not axes:
        raise ValueError("axes must be a non-empty tuple of ScenarioAxis values")
    if any(not isinstance(axis, ScenarioAxis) for axis in axes):
        raise TypeError("axes must contain only ScenarioAxis values")
    if len(set(axes)) != len(axes):
        raise ValueError("axes must not contain duplicates")


def _require_provenance(
    provenance: tuple[ProvenanceReference, ...],
    information_timestamp_utc: datetime,
) -> None:
    if not isinstance(provenance, tuple) or not provenance:
        raise ValueError("provenance must be a non-empty tuple")
    if any(not isinstance(source, ProvenanceReference) for source in provenance):
        raise TypeError("provenance must contain only ProvenanceReference values")
    identities = {(source.source_id, source.content_sha256) for source in provenance}
    if len(identities) != len(provenance):
        raise ValueError("provenance must not contain duplicate source references")
    if any(source.available_at_utc > information_timestamp_utc for source in provenance):
        raise ValueError("provenance cannot become available after the information timestamp")
