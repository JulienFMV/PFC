from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pfc_shaping.validation.eex_current_monthly_research_solution as subject
from pfc_shaping.validation.eex_current_monthly_research_solution import (
    AUTHORITIES,
    CONTRACT_CONTENT_ID,
    CONTRACT_PATH,
    DELIVERY_MONTHS,
    EXECUTION,
    HISTORY_PATH,
    LATEST_PATH,
    PASS_STATUS,
    EexCurrentMonthlyResearchSolutionError,
    build_selected_current_monthly_research_solution,
    canonical_sha256,
    solve_current_monthly_research_frames,
    validate_current_monthly_solution_contract,
    validate_current_monthly_solution_contract_path,
)

REFERENCE = datetime(2026, 8, 6, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def source_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_parquet(HISTORY_PATH), pd.read_parquet(LATEST_PATH)


@pytest.fixture(scope="module")
def selected_result():
    return build_selected_current_monthly_research_solution(reference_time_utc=REFERENCE)


def _solve(history: pd.DataFrame, latest: pd.DataFrame):
    return solve_current_monthly_research_frames(
        history=history,
        latest=latest,
        reference_time_utc=REFERENCE,
    )


def _mutate_terminal(
    history: pd.DataFrame,
    latest: pd.DataFrame,
    mask: pd.Series,
    column: str,
    value: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    changed_history = history.copy()
    changed_latest = latest.copy()
    latest_indices = changed_latest.index[mask]
    for index in latest_indices:
        row = changed_latest.loc[index]
        match = (
            changed_history["date"].eq(row["date"])
            & changed_history["product"].eq(row["product"])
            & changed_history["load_type"].eq(row["load_type"])
        )
        changed_history.loc[match, column] = value
    changed_latest.loc[mask, column] = value
    return changed_history, changed_latest


def test_contract_is_exact_content_addressed_and_zero_remote() -> None:
    summary = validate_current_monthly_solution_contract_path(CONTRACT_PATH)

    assert summary["contract_content_id"] == CONTRACT_CONTENT_ID
    assert summary["delivery_month_count"] == 76
    assert summary["base_only_monthly_level"] is True
    assert summary["peak_is_sidecar_only"] is True
    assert all(value == 0 for value in EXECUTION.values())


def test_contract_rejects_semantic_mutation() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["authorities"]["production"] = True

    with pytest.raises(EexCurrentMonthlyResearchSolutionError, match="authorities"):
        validate_current_monthly_solution_contract(contract)


def test_contract_rejects_duplicate_json_key(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text('{"schema_version":"x","schema_version":"y"}', encoding="utf-8")

    with pytest.raises(EexCurrentMonthlyResearchSolutionError):
        validate_current_monthly_solution_contract_path(path.resolve())


def test_selected_solution_is_complete_exact_and_non_authoritative(selected_result) -> None:
    assessment = selected_result.assessment

    assert assessment["status"] == PASS_STATUS
    assert assessment["monthly_solution_row_count"] == 76
    assert assessment["active_hard_constraint_count"] == 17
    assert assessment["redundant_consistency_quote_count"] == 2
    assert assessment["active_hard_max_abs_residual_eur_mwh"] <= 1e-9
    assert assessment["all_displayed_base_max_abs_residual_eur_mwh"] <= 0.01
    assert assessment["shape_prior_parent_max_abs_mean_eur_mwh"] <= 1e-10
    assert assessment["history_prior_status"] == "PARTIAL_HISTORY_FORWARD"
    assert assessment["structural_prior_status"] == "UNSUPPORTED"
    assert assessment["fused_prior_status"] == "HISTORY_FORWARD"
    assert assessment["panel_prior_status"] == "UNSUPPORTED"
    assert assessment["authorities"] == AUTHORITIES
    assert assessment["point_in_time_authorized"] is False
    assert assessment["model_input_authorized"] is False
    assert assessment["ompex_superiority_authorized"] is False


def test_selected_solution_grid_starts_at_first_wholly_undelivered_month(selected_result) -> None:
    monthly = selected_result.monthly_solution

    assert monthly["delivery_month"].tolist() == [str(month) for month in DELIVERY_MONTHS]
    assert monthly.iloc[0]["delivery_month"] == "2026-09"
    assert monthly.iloc[-1]["delivery_month"] == "2032-12"
    assert "2026-08" not in set(monthly["delivery_month"])
    assert set(monthly["monthly_level_authority"]) == {"CH_EEX_BASE_SOLVER"}


def test_peak_is_a_separate_non_authoritative_sidecar(selected_result) -> None:
    sidecar = selected_result.peak_sidecar

    assert len(sidecar) == 19
    assert set(sidecar["load_type"]) == {"PEAK"}
    assert not sidecar["monthly_level_authority"].any()
    assert not sidecar["shaping_input_authorized"].any()
    assert "price_eur_mwh" not in selected_result.monthly_solution.columns[2:]


def test_selected_replay_is_deterministic(selected_result) -> None:
    replay = build_selected_current_monthly_research_solution(reference_time_utc=REFERENCE)

    assert replay.assessment_content_id == selected_result.assessment_content_id
    pd.testing.assert_frame_equal(replay.monthly_solution, selected_result.monthly_solution)
    pd.testing.assert_frame_equal(replay.peak_sidecar, selected_result.peak_sidecar)
    pd.testing.assert_frame_equal(
        replay.constraint_diagnostics, selected_result.constraint_diagnostics
    )
    pd.testing.assert_frame_equal(replay.prior_diagnostics, selected_result.prior_diagnostics)


def test_assessment_is_price_free(selected_result, source_frames) -> None:
    _, latest = source_frames
    encoded = json.dumps(selected_result.assessment, sort_keys=True)

    assert selected_result.assessment["price_values_persisted_in_assessment"] is False
    for value in latest["price"].iloc[:10]:
        assert format(float(value), ".17g") not in encoded


def test_peak_price_mutation_cannot_change_monthly_level(source_frames) -> None:
    history, latest = source_frames
    changed_history = history.copy()
    changed_latest = latest.copy()
    history_peak = changed_history["date"].eq(subject.SURFACE_DATE) & changed_history[
        "load_type"
    ].eq("PEAK")
    latest_peak = changed_latest["load_type"].eq("PEAK")
    changed_history.loc[history_peak, "price"] += 10.0
    changed_latest.loc[latest_peak, "price"] += 10.0

    original = _solve(history, latest)
    changed = _solve(changed_history, changed_latest)

    pd.testing.assert_frame_equal(original.monthly_solution, changed.monthly_solution)
    pd.testing.assert_frame_equal(original.constraint_diagnostics, changed.constraint_diagnostics)
    assert (
        original.assessment["output_content_ids"]["peak_sidecar_content_id"]
        != changed.assessment["output_content_ids"]["peak_sidecar_content_id"]
    )


def test_base_price_mutation_changes_monthly_solution(source_frames) -> None:
    history, latest = source_frames
    mask = latest["load_type"].eq("BASE") & latest["product"].eq("2026-09")
    assert int(mask.sum()) == 1
    original_price = float(latest.loc[mask, "price"].iloc[0])
    changed_history, changed_latest = _mutate_terminal(
        history,
        latest,
        mask,
        "price",
        original_price + 1.0,
    )

    original = _solve(history, latest)
    changed = _solve(changed_history, changed_latest)

    assert not np.allclose(
        original.monthly_solution["base_price_eur_mwh"],
        changed.monthly_solution["base_price_eur_mwh"],
    )
    assert changed.assessment["active_hard_max_abs_residual_eur_mwh"] <= 1e-9


def test_past_history_level_shift_is_zero_mean_shape_only(source_frames) -> None:
    history, latest = source_frames
    shifted = history.copy()
    past = shifted["date"].lt(subject.SURFACE_DATE)
    shifted.loc[past, "price"] += 100.0

    original = _solve(history, latest)
    changed = _solve(shifted, latest)

    np.testing.assert_allclose(
        original.monthly_solution["base_price_eur_mwh"],
        changed.monthly_solution["base_price_eur_mwh"],
        atol=1e-9,
        rtol=0,
    )
    assert changed.assessment["shape_prior_parent_max_abs_mean_eur_mwh"] <= 1e-10


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("market", "DE", "market differs"),
        ("unit", "CHF/MWh", "unit differs"),
        ("source", "OTHER", "source differs"),
        ("source_system", "LOCAL", "source system differs"),
        ("source_snapshot_sha256", "a" * 64, "source snapshot differs"),
        ("load_type", "OFFPEAK", "load type differs"),
        ("product_type", "Day", "product type differs"),
        ("point_in_time_available_at_proven", True, "escalates authority"),
        ("rolling_origin_selection_authorized", True, "escalates authority"),
        ("production_promotion_authorized", True, "escalates authority"),
    ],
)
def test_frame_semantic_mutations_fail_closed(source_frames, column, value, message) -> None:
    history, latest = source_frames
    changed = latest.copy()
    changed.loc[changed.index[0], column] = value

    with pytest.raises(EexCurrentMonthlyResearchSolutionError, match=message):
        _solve(history, changed)


def test_duplicate_quote_identity_fails_closed(source_frames) -> None:
    history, latest = source_frames
    changed = latest.copy()
    for column in ("date", "product", "load_type"):
        changed.loc[changed.index[1], column] = changed.loc[changed.index[0], column]

    with pytest.raises(EexCurrentMonthlyResearchSolutionError, match="repeats quote identity"):
        _solve(history, changed)


def test_null_and_nonfinite_prices_fail_closed(source_frames) -> None:
    history, latest = source_frames
    for value, message in ((np.nan, "nulls"), (np.inf, "non-finite")):
        changed = latest.copy()
        changed.loc[changed.index[0], "price"] = value
        with pytest.raises(EexCurrentMonthlyResearchSolutionError, match=message):
            _solve(history, changed)


def test_dtype_and_column_drift_fail_closed(source_frames) -> None:
    history, latest = source_frames
    wrong_dtype = latest.copy()
    wrong_dtype["price"] = wrong_dtype["price"].astype("float32")
    with pytest.raises(EexCurrentMonthlyResearchSolutionError, match="dtypes differ"):
        _solve(history, wrong_dtype)

    wrong_columns = latest.rename(columns={"price": "value"})
    with pytest.raises(EexCurrentMonthlyResearchSolutionError, match="columns differ"):
        _solve(history, wrong_columns)


def test_latest_must_be_exact_terminal_history_slice(source_frames) -> None:
    history, latest = source_frames
    changed = latest.copy()
    changed.loc[changed.index[0], "price"] += 0.01

    with pytest.raises(EexCurrentMonthlyResearchSolutionError, match="terminal history slice"):
        _solve(history, changed)


def test_future_history_and_wrong_latest_date_fail_closed(source_frames) -> None:
    history, latest = source_frames
    future = history.copy()
    future.loc[future.index[-1], "date"] = pd.Timestamp("2026-08-05")
    with pytest.raises(EexCurrentMonthlyResearchSolutionError, match="maximum date differs"):
        _solve(future, latest)

    wrong_latest = latest.copy()
    wrong_latest["date"] = wrong_latest["date"] - pd.Timedelta(days=1)
    with pytest.raises(EexCurrentMonthlyResearchSolutionError, match="latest surface date differs"):
        _solve(history, wrong_latest)


def test_reference_time_cannot_predate_surface(source_frames) -> None:
    history, latest = source_frames

    with pytest.raises(EexCurrentMonthlyResearchSolutionError, match="reference predates"):
        solve_current_monthly_research_frames(
            history=history,
            latest=latest,
            reference_time_utc=datetime(2026, 8, 3, tzinfo=timezone.utc),
        )


def test_naive_reference_time_fails_closed(source_frames) -> None:
    history, latest = source_frames

    with pytest.raises(EexCurrentMonthlyResearchSolutionError, match="timezone-aware"):
        solve_current_monthly_research_frames(
            history=history,
            latest=latest,
            reference_time_utc=datetime(2026, 8, 6),
        )


def test_selected_normalization_manifest_mutation_fails_before_solve(tmp_path: Path) -> None:
    document = json.loads(subject.NORMALIZATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    document["authority"]["production_promotion_authorized"] = True
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")

    with pytest.raises(EexCurrentMonthlyResearchSolutionError, match="SHA-256 differs"):
        build_selected_current_monthly_research_solution(
            normalization_manifest_path=path.resolve(),
            reference_time_utc=REFERENCE,
        )


def test_assessment_content_id_recomputes_exactly(selected_result) -> None:
    assert canonical_sha256(selected_result.assessment) == selected_result.assessment_content_id
    assert copy.deepcopy(selected_result.assessment)["authorities"]["production"] is False


def test_module_has_no_ct_import() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "pfc_shaping.ct" not in source
