from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from pfc_shaping.lt.curve_products import (
    AUTHORITY_STATUS,
    CURVE_PRODUCT_CONTRACT_VERSION,
    SCENARIO_CONTRACT_VERSION,
    AuthorityNegative,
    CurveProductDefinition,
    CurveProductType,
    NormalizationBucket,
    ProvenanceReference,
    RequiredLevelSource,
    ScenarioAxis,
    ScenarioDefinition,
    ScenarioLevelEffect,
)

INFO = datetime(2026, 9, 2, 10, 0, tzinfo=timezone.utc)
SOURCE = ProvenanceReference(
    source_id="synthetic-contract-fixture",
    content_sha256="a" * 64,
    available_at_utc=datetime(2026, 9, 2, 9, 0, tzinfo=timezone.utc),
)


def _shape_scenario(scenario_id: str = "dry-hydro-shape") -> ScenarioDefinition:
    return ScenarioDefinition(
        scenario_id=scenario_id,
        label="Dry hydro shape",
        information_timestamp_utc=INFO,
        axes=(ScenarioAxis.HYDRO,),
        level_effect=ScenarioLevelEffect.SHAPE_ONLY,
        normalization_bucket=NormalizationBucket.SOLVER_MONTH,
        provenance=(SOURCE,),
    )


def _central() -> CurveProductDefinition:
    return CurveProductDefinition(
        product_id="market-central-ch-20260902",
        product_type=CurveProductType.MARKET_CENTRAL_CH,
        information_timestamp_utc=INFO,
        required_level_source=RequiredLevelSource.CH_MONTHLY_BASE_SOLVER,
        normalization_bucket=NormalizationBucket.SOLVER_MONTH,
        provenance=(SOURCE,),
    )


def _shape_curve() -> CurveProductDefinition:
    return CurveProductDefinition(
        product_id="fundamental-dry-shape-20260902",
        product_type=CurveProductType.FUNDAMENTAL_SCENARIO_CH,
        information_timestamp_utc=INFO,
        required_level_source=RequiredLevelSource.CH_MONTHLY_BASE_SOLVER,
        normalization_bucket=NormalizationBucket.SOLVER_MONTH,
        provenance=(SOURCE,),
        scenario=_shape_scenario(),
        parent_product=_central(),
    )


def test_contracts_are_non_overridable_and_authority_negative() -> None:
    scenario = _shape_scenario()
    curve = _shape_curve()

    assert scenario.authority == AuthorityNegative()
    assert curve.authority.status == AUTHORITY_STATUS
    assert not any(value for key, value in curve.authority.to_manifest().items() if key != "status")
    with pytest.raises(TypeError):
        AuthorityNegative(production_authorized=True)  # type: ignore[call-arg]
    with pytest.raises(FrozenInstanceError):
        curve.authority.production_authorized = True  # type: ignore[misc]


def test_shape_only_scenario_curve_retains_solver_month_contract() -> None:
    curve = _shape_curve()
    scenario_manifest = curve.scenario.to_manifest()  # type: ignore[union-attr]
    product_manifest = curve.to_manifest()

    assert scenario_manifest["contract_version"] == SCENARIO_CONTRACT_VERSION
    assert scenario_manifest["probability_weight"] is None
    assert product_manifest["contract_version"] == CURVE_PRODUCT_CONTRACT_VERSION
    assert product_manifest["product_type"] == "fundamental_scenario_ch"
    assert product_manifest["parent_product_id"] == "market-central-ch-20260902"
    assert product_manifest["required_level_source"] == "ch_monthly_base_solver"
    assert product_manifest["required_level_source_is_authority_grant"] is False
    assert product_manifest["normalization_bucket"] == "solver_month"


def test_shape_only_scenario_rejects_level_solve_and_wrong_bucket() -> None:
    with pytest.raises(ValueError, match="shape-only scenarios must normalize"):
        ScenarioDefinition(
            scenario_id="invalid-shape-bucket",
            label="Invalid shape",
            information_timestamp_utc=INFO,
            axes=(ScenarioAxis.WEATHER,),
            level_effect=ScenarioLevelEffect.SHAPE_ONLY,
            normalization_bucket=NormalizationBucket.SEPARATE_UPSTREAM_LEVEL_SOLVE,
            provenance=(SOURCE,),
        )

    with pytest.raises(ValueError, match="cannot declare an upstream level solve"):
        ScenarioDefinition(
            scenario_id="invalid-shape-level",
            label="Invalid shape level",
            information_timestamp_utc=INFO,
            axes=(ScenarioAxis.WEATHER,),
            level_effect=ScenarioLevelEffect.SHAPE_ONLY,
            normalization_bucket=NormalizationBucket.SOLVER_MONTH,
            provenance=(SOURCE,),
            required_upstream_solve_id="forbidden-upstream-solve",
        )


def test_level_changing_scenario_requires_separate_noncentral_solve() -> None:
    scenario = ScenarioDefinition(
        scenario_id="electrification-level",
        label="Electrification level",
        information_timestamp_utc=INFO,
        axes=(ScenarioAxis.ELECTRIFICATION,),
        level_effect=ScenarioLevelEffect.LEVEL_CHANGING,
        normalization_bucket=NormalizationBucket.SEPARATE_UPSTREAM_LEVEL_SOLVE,
        provenance=(SOURCE,),
        required_upstream_solve_id="fundamental-solve-assumption-set-1",
    )
    curve = CurveProductDefinition(
        product_id="fundamental-electrification-20260902",
        product_type=CurveProductType.FUNDAMENTAL_SCENARIO_CH,
        information_timestamp_utc=INFO,
        required_level_source=RequiredLevelSource.SEPARATE_UPSTREAM_FUNDAMENTAL_SOLVE,
        normalization_bucket=NormalizationBucket.SEPARATE_UPSTREAM_LEVEL_SOLVE,
        provenance=(SOURCE,),
        scenario=scenario,
    )

    assert curve.parent_product is None
    assert curve.authority.monthly_level_authority_granted is False
    with pytest.raises(ValueError, match="cannot mutate or descend"):
        CurveProductDefinition(
            product_id="invalid-central-descendant",
            product_type=CurveProductType.FUNDAMENTAL_SCENARIO_CH,
            information_timestamp_utc=INFO,
            required_level_source=RequiredLevelSource.SEPARATE_UPSTREAM_FUNDAMENTAL_SOLVE,
            normalization_bucket=NormalizationBucket.SEPARATE_UPSTREAM_LEVEL_SOLVE,
            provenance=(SOURCE,),
            scenario=scenario,
            parent_product=_central(),
        )


def test_scenario_probability_is_always_rejected() -> None:
    with pytest.raises(ValueError, match="probability weights are forbidden"):
        ScenarioDefinition(
            scenario_id="weighted-weather",
            label="Weighted weather",
            information_timestamp_utc=INFO,
            axes=(ScenarioAxis.WEATHER,),
            level_effect=ScenarioLevelEffect.SHAPE_ONLY,
            normalization_bucket=NormalizationBucket.SOLVER_MONTH,
            provenance=(SOURCE,),
            probability_weight=0.5,
        )


def test_stochastic_paths_require_matching_typed_scenario_parent() -> None:
    parent = _shape_curve()
    paths = CurveProductDefinition(
        product_id="stochastic-dry-shape-20260902",
        product_type=CurveProductType.STOCHASTIC_SPOT_PATHS_CH,
        information_timestamp_utc=INFO,
        required_level_source=parent.required_level_source,
        normalization_bucket=parent.normalization_bucket,
        provenance=(SOURCE,),
        scenario=parent.scenario,
        parent_product=parent,
    )

    assert paths.to_manifest()["parent_product_id"] == parent.product_id
    with pytest.raises(ValueError, match="same scenario identity"):
        CurveProductDefinition(
            product_id="invalid-stochastic-parent",
            product_type=CurveProductType.STOCHASTIC_SPOT_PATHS_CH,
            information_timestamp_utc=INFO,
            required_level_source=parent.required_level_source,
            normalization_bucket=parent.normalization_bucket,
            provenance=(SOURCE,),
            scenario=_shape_scenario("wet-hydro-shape"),
            parent_product=parent,
        )


def test_market_central_rejects_scenario_and_non_solver_level_source() -> None:
    with pytest.raises(ValueError, match="cannot carry a scenario"):
        CurveProductDefinition(
            product_id="invalid-central-scenario",
            product_type=CurveProductType.MARKET_CENTRAL_CH,
            information_timestamp_utc=INFO,
            required_level_source=RequiredLevelSource.CH_MONTHLY_BASE_SOLVER,
            normalization_bucket=NormalizationBucket.SOLVER_MONTH,
            provenance=(SOURCE,),
            scenario=_shape_scenario(),
        )
    with pytest.raises(ValueError, match="requires the CH monthly BASE solver"):
        CurveProductDefinition(
            product_id="invalid-central-source",
            product_type=CurveProductType.MARKET_CENTRAL_CH,
            information_timestamp_utc=INFO,
            required_level_source=RequiredLevelSource.SEPARATE_UPSTREAM_FUNDAMENTAL_SOLVE,
            normalization_bucket=NormalizationBucket.SOLVER_MONTH,
            provenance=(SOURCE,),
        )


def test_information_timestamp_is_utc_and_bounds_provenance() -> None:
    late_source = ProvenanceReference(
        source_id="late-synthetic-source",
        content_sha256="b" * 64,
        available_at_utc=datetime(2026, 9, 2, 11, 0, tzinfo=timezone.utc),
    )
    with pytest.raises(ValueError, match="cannot become available after"):
        ScenarioDefinition(
            scenario_id="late-information",
            label="Late information",
            information_timestamp_utc=INFO,
            axes=(ScenarioAxis.OUTAGE,),
            level_effect=ScenarioLevelEffect.SHAPE_ONLY,
            normalization_bucket=NormalizationBucket.SOLVER_MONTH,
            provenance=(late_source,),
        )
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        ProvenanceReference(
            source_id="naive-source",
            content_sha256="c" * 64,
            available_at_utc=datetime(2026, 9, 2, 9, 0),
        )
