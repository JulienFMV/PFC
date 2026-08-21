from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import pandas as pd
import pytest

from pfc_shaping.validation import ch_lt_structural_prediction_commitment as builder


@pytest.fixture
def governed_tmp_path() -> Path:
    path = (
        builder.ROOT
        / "build"
        / "test-workspaces"
        / "structural-prediction-commitment"
        / uuid.uuid4().hex
    )
    path.mkdir(parents=True, exist_ok=False)
    return path


def _build(output):
    return builder.build_commitment(
        builder.ROOT,
        output,
        inventory_path=builder.ROOT / builder.SELECTED_INVENTORY_RELATIVE,
        expected_inventory_sha256=builder.SELECTED_INVENTORY_SHA256,
    )


def test_builds_exact_outcome_blind_noncountable_commitment(
    governed_tmp_path: Path,
) -> None:
    output = governed_tmp_path / "commitment"
    result = _build(output)

    assert result["status"] == "SEALED_LOCAL_STRUCTURAL_DRY_RUN_NONCOUNTABLE_NO_GO"
    assert result["target_count"] == 36
    assert result["prediction_count"] == 108
    assert result["monthly_level_authority"] == "solver"
    boundary = result["authority_boundary"]
    assert boundary["countable_prospective_origin"] is False
    assert boundary["truth_open_authorized"] is False
    assert boundary["outcomes_opened"] is False
    assert boundary["future_holdout_consumed"] is False
    assert boundary["t057_consumed"] is False
    assert boundary["scientific_admission"] is False
    assert boundary["production_authorization"] is False
    assert boundary["promotion_gate"] is False

    predictions = json.loads((output / "predictions.json").read_text(encoding="utf-8"))
    assert predictions["schema_version"] == "ch_lt_structural_prediction_set.v3"
    assert predictions["prediction_count"] == 108
    assert len(predictions["predictions"]) == 108
    assert [item["lead_month"] for item in predictions["predictions"][::3]] == list(
        range(1, 37)
    )
    assert all(
        item["probabilistic_interval_supported"] is False
        for item in predictions["predictions"]
    )
    contract = predictions["hourly_shape_functional_contract"]
    assert contract["truth_requirement"] == (
        "NATIVE_CH_DAY_AHEAD_HOURLY_PRICES_WITH_GOVERNED_AVAILABLE_AT"
    )
    assert contract["quarter_hour_truth_required"] is False
    assert contract["scenario_interpretation"] == (
        "STRUCTURAL_SENSITIVITY_NOT_PROBABILISTIC_COVERAGE"
    )
    assert contract["primary_scenario"] == "central"
    assert contract["scenario_pooling_for_quality_claims"] == "FORBIDDEN"
    assert contract["ex_post_scores"]["centering_rule"]["prediction"] == (
        "PREDICTION_MINUS_ITS_OWN_MONTHLY_HOURLY_MEAN"
    )
    first = predictions["predictions"][0]
    functionals = first["hourly_shape_functionals"]
    assert first["native_hourly_interval_count"] == 744
    assert functionals["hourly_interval_count"] == 744
    assert functionals["midday_hour_count"] == 124
    assert functionals["evening_hour_count"] == 124
    assert functionals["overnight_hour_count"] == 186
    assert float(functionals["hourly_mean_eur_mwh"]) == pytest.approx(
        float(first["monthly_mean_eur_mwh"]), abs=1e-12
    )
    assert float(functionals["evening_minus_midday_eur_mwh"]) == pytest.approx(
        float(functionals["evening_mean_eur_mwh"])
        - float(functionals["midday_mean_eur_mwh"]),
        abs=1e-12,
    )
    assert float(
        functionals["midday_premium_to_monthly_mean_eur_mwh"]
    ) == pytest.approx(
        float(functionals["midday_mean_eur_mwh"])
        - float(functionals["hourly_mean_eur_mwh"]),
        abs=1e-12,
    )
    assert functionals["negative_price_deficit_mean_all_hours_eur_mwh"] == "0"
    assert functionals["negative_hour_count"] >= 0
    assert predictions["outcomes_opened"] is False
    assert all("observed" not in item for item in predictions["predictions"])


def test_hourly_shape_commitment_is_byte_reproducible(
    governed_tmp_path: Path,
) -> None:
    first = governed_tmp_path / "first"
    second = governed_tmp_path / "second"

    _build(first)
    _build(second)

    assert (first / "predictions.json").read_bytes() == (
        second / "predictions.json"
    ).read_bytes()
    assert (first / "commitment.json").read_bytes() == (
        second / "commitment.json"
    ).read_bytes()


def test_reproduction_matches_selected_artifact_bytes(
    governed_tmp_path: Path,
) -> None:
    output = governed_tmp_path / "selected-reproduction"
    _build(output)

    assert hashlib.sha256((output / "predictions.json").read_bytes()).hexdigest() == (
        builder.SELECTED_PREDICTIONS_SHA256
    )
    assert hashlib.sha256((output / "commitment.json").read_bytes()).hexdigest() == (
        builder.SELECTED_COMMITMENT_SHA256
    )


def test_resumes_exact_partial_staging_without_rewriting(
    governed_tmp_path: Path,
) -> None:
    source = governed_tmp_path / "source"
    target = governed_tmp_path / "target"
    _build(source)
    staging = target.parent / f".{target.name}.staging"
    staging.mkdir()
    staged_prediction = staging / "predictions.json"
    staged_prediction.write_bytes((source / "predictions.json").read_bytes())

    _build(target)

    assert not staging.exists()
    assert (target / "predictions.json").read_bytes() == (
        source / "predictions.json"
    ).read_bytes()
    assert (target / "commitment.json").read_bytes() == (
        source / "commitment.json"
    ).read_bytes()


def test_rejects_noncontiguous_quarter_hour_grid() -> None:
    malformed_index = []
    for hour in range(24):
        malformed_index.extend(
            pd.Timestamp("2026-08-01T00:00:00Z")
            + pd.Timedelta(hours=hour, minutes=offset)
            for offset in (0, 1, 2, 3)
        )
    malformed = pd.Series(range(96), index=pd.DatetimeIndex(malformed_index), dtype=float)

    with pytest.raises(
        builder.StructuralPredictionCommitmentError,
        match="quarter-hour grid is not exact and contiguous",
    ):
        builder._hourly_shape_functionals(
            malformed,
            expected_hour_count=24,
            label="malformed",
        )


def test_selected_predictions_cover_dst_short_and_long_months() -> None:
    selected = json.loads(
        (
            builder.ROOT
            / "build/ch-lt-structural-prediction-commitments/"
            "structural-dry-run-20260731T072027Z-v7-quant-contract"
            / "predictions.json"
        ).read_text(encoding="utf-8")
    )
    central = {
        row["delivery_month"]: row
        for row in selected["predictions"]
        if row["scenario"] == "central"
    }

    march = central["2027-03"]["hourly_shape_functionals"]
    october = central["2027-10"]["hourly_shape_functionals"]
    assert march["hourly_interval_count"] == 743
    assert march["overnight_hour_count"] == 185
    assert october["hourly_interval_count"] == 745
    assert october["overnight_hour_count"] == 187


def test_rejects_rebound_inventory_hash(governed_tmp_path: Path) -> None:
    with pytest.raises(
        builder.StructuralPredictionCommitmentError,
        match="selected outcome-blind origin inventory",
    ):
        builder.build_commitment(
            builder.ROOT,
            governed_tmp_path / "rebound",
            inventory_path=builder.ROOT / builder.SELECTED_INVENTORY_RELATIVE,
            expected_inventory_sha256="0" * 64,
        )


def test_rejects_output_outside_build() -> None:
    outside_build = builder.ROOT / ".planning" / "forbidden-commitment-output"

    with pytest.raises(builder.StructuralPredictionCommitmentError, match="below build"):
        builder.build_commitment(
            builder.ROOT,
            outside_build,
            inventory_path=builder.ROOT / builder.SELECTED_INVENTORY_RELATIVE,
            expected_inventory_sha256=builder.SELECTED_INVENTORY_SHA256,
        )


def test_rejects_inventory_outside_governed_namespace(
    governed_tmp_path: Path,
) -> None:
    with pytest.raises(
        builder.StructuralPredictionCommitmentError,
        match="governed inventory namespace",
    ):
        builder.build_commitment(
            builder.ROOT,
            governed_tmp_path / "commitment",
            inventory_path=builder.ROOT / "build" / "wrong" / "inventory.json",
            expected_inventory_sha256=builder.SELECTED_INVENTORY_SHA256,
        )


def test_rejects_unselected_inventory_inside_governed_namespace(
    governed_tmp_path: Path,
) -> None:
    unselected = (
        builder.ROOT
        / "build"
        / "ch-lt-origin-target-mask-inventories"
        / "unselected.json"
    )
    with pytest.raises(
        builder.StructuralPredictionCommitmentError,
        match="not the selected outcome-blind origin inventory",
    ):
        builder.build_commitment(
            builder.ROOT,
            governed_tmp_path / "commitment",
            inventory_path=unselected,
            expected_inventory_sha256=builder.SELECTED_INVENTORY_SHA256,
        )


def test_binds_current_evidence_without_claiming_requalification(
    governed_tmp_path: Path,
) -> None:
    result = _build(governed_tmp_path / "commitment")

    current = result["bindings"]["current_evidence_selection"]
    assert current["sha256"] == builder.CURRENT_EVIDENCE_SHA256
    assert current["independent_rolling_origin_count"] == 0
    assert (
        "CURRENT_LOCAL_CAPTURE_LEDGER_DOES_NOT_REQUALIFY_THE_FROZEN_MODEL"
        in result["limitations"]
    )
