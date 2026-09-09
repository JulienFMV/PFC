from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.evaluation_engine import (
    EvaluationEngineError,
    MetricStatus,
    SyntheticEvaluationSet,
    evaluate_synthetic_predictions,
)
from pfc_shaping.lt.evaluation_protocol import (
    LEAD_MONTH_BUCKETS,
    PRIMARY_METRIC,
    SECONDARY_METRICS,
    default_evaluation_protocol,
)

ORIGIN = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)


def _candidate_ids() -> tuple[str, ...]:
    return tuple(item.candidate_id for item in default_evaluation_protocol().candidates)


def _batch(
    *,
    truth: np.ndarray | None = None,
    predictions: dict[str, np.ndarray] | None = None,
    index: pd.DatetimeIndex | None = None,
) -> SyntheticEvaluationSet:
    delivery = (
        pd.date_range("2026-11-29 00:00", periods=96, freq="h", tz="UTC")
        if index is None
        else index
    )
    signal = np.sin(np.arange(len(delivery), dtype=float) * 2.0 * np.pi / 24.0)
    actual = 70.0 + 8.0 * signal if truth is None else truth
    forecasts = (
        {
            candidate_id: actual + (position + 1) * 0.1 * np.cos(np.arange(len(delivery)))
            for position, candidate_id in enumerate(_candidate_ids())
        }
        if predictions is None
        else predictions
    )
    return SyntheticEvaluationSet(
        fixture_id="synthetic-evaluation-basic",
        origin_slot_id="origin-2026-10",
        origin_as_of_utc=ORIGIN,
        delivery_at_utc=delivery,
        truth_eur_mwh=actual,
        predictions_eur_mwh=forecasts,
    )


def test_report_uses_exact_inventory_without_selection_authority() -> None:
    report = evaluate_synthetic_predictions(_batch())
    manifest = report.to_manifest()

    assert tuple(score.candidate_id for score in report.scores) == _candidate_ids()
    assert manifest["ranking_or_selection_performed"] is False
    assert manifest["horizon_buckets"] == list(LEAD_MONTH_BUCKETS)
    assert manifest["unsupported_layer_cannot_be_hidden_by_aggregate_pass"] is True
    assert manifest["countable_origin"] is False
    assert manifest["real_truth_opened"] is False
    assert manifest["monthly_level_authority"] == "CH_MONTHLY_BASE_SOLVER_UNCHANGED"
    assert report.authority.model_training_authorized is False
    assert report.authority.model_selection_authorized is False
    assert report.authority.scientific_claim_authorized is False
    assert report.authority.production_authorized is False
    assert "winner" not in str(manifest).lower()


def test_metric_inventory_computes_only_frozen_shape_metrics() -> None:
    score = evaluate_synthetic_predictions(_batch()).scores[0]

    assert tuple(score.metrics) == (PRIMARY_METRIC, *SECONDARY_METRICS)
    for metric_id in (
        PRIMARY_METRIC,
        "MONTHLY_LEVEL_NEUTRALIZED_RMSE_EUR_MWH",
        "MONTHLY_LEVEL_NEUTRALIZED_BIAS_EUR_MWH",
        "P95_ABSOLUTE_ERROR_EUR_MWH",
    ):
        assert score.metrics[metric_id].status is MetricStatus.COMPUTED_SYNTHETIC
        assert score.metrics[metric_id].value is not None
    for metric_id in SECONDARY_METRICS[3:]:
        assert score.metrics[metric_id].status is MetricStatus.UNSUPPORTED
        assert score.metrics[metric_id].value is None
        assert score.metrics[metric_id].reason


def test_horizon_buckets_are_complete_and_absence_is_unsupported() -> None:
    report = evaluate_synthetic_predictions(_batch())

    for score in report.scores:
        assert tuple(item.bucket_id for item in score.horizon_scores) == LEAD_MONTH_BUCKETS
        near = score.horizon_scores[0]
        assert near.common_row_count == report.common_row_count
        assert near.metrics[PRIMARY_METRIC].status is MetricStatus.COMPUTED_SYNTHETIC
        for empty in score.horizon_scores[1:]:
            assert empty.common_row_count == 0
            assert empty.metrics[PRIMARY_METRIC].status is MetricStatus.UNSUPPORTED
            assert (
                empty.metrics[PRIMARY_METRIC].reason
                == "NO_COMMON_SYNTHETIC_ROWS_FOR_HORIZON_BUCKET"
            )


def test_candidate_score_rejects_non_reconciling_horizon_rows() -> None:
    score = evaluate_synthetic_predictions(_batch()).scores[0]
    corrupted = replace(
        score.horizon_scores[0],
        common_row_count=score.horizon_scores[0].common_row_count - 1,
    )

    with pytest.raises(EvaluationEngineError, match="do not reconcile"):
        replace(score, horizon_scores=(corrupted, *score.horizon_scores[1:]))


def test_monthly_level_neutralization_is_invariant_to_independent_monthly_shifts() -> None:
    baseline = _batch()
    baseline_report = evaluate_synthetic_predictions(baseline)
    local = baseline.delivery_at_utc.tz_convert("Europe/Zurich")
    month = local.strftime("%Y-%m")
    truth_shift = np.where(month == month[0], 125.0, -70.0)
    shifted_predictions = {
        candidate_id: values + np.where(month == month[0], position * 31.0, position * -17.0)
        for position, (candidate_id, values) in enumerate(
            baseline.predictions_eur_mwh.items(),
            start=1,
        )
    }
    shifted = _batch(
        truth=baseline.truth_eur_mwh + truth_shift,
        predictions=shifted_predictions,
        index=baseline.delivery_at_utc,
    )
    shifted_report = evaluate_synthetic_predictions(shifted)

    for before, after in zip(baseline_report.scores, shifted_report.scores):
        for metric_id in (
            PRIMARY_METRIC,
            "MONTHLY_LEVEL_NEUTRALIZED_RMSE_EUR_MWH",
            "MONTHLY_LEVEL_NEUTRALIZED_BIAS_EUR_MWH",
            "P95_ABSOLUTE_ERROR_EUR_MWH",
        ):
            assert after.metrics[metric_id].value == pytest.approx(
                before.metrics[metric_id].value,
                abs=1e-12,
            )


def test_all_candidates_share_one_complete_case_intersection() -> None:
    baseline = _batch()
    predictions = {key: value.copy() for key, value in baseline.predictions_eur_mwh.items()}
    predictions[_candidate_ids()[0]][3] = np.nan
    predictions[_candidate_ids()[1]][8] = np.nan
    truth = baseline.truth_eur_mwh.copy()
    truth[12] = np.nan

    report = evaluate_synthetic_predictions(
        _batch(truth=truth, predictions=predictions, index=baseline.delivery_at_utc)
    )

    assert report.common_row_count == len(truth) - 3
    assert {score.common_row_count for score in report.scores} == {len(truth) - 3}
    assert {score.common_energy_hours for score in report.scores} == {float(len(truth) - 3)}


def test_exact_prediction_is_zero_after_energy_weighted_centering() -> None:
    baseline = _batch()
    predictions = {candidate_id: baseline.truth_eur_mwh.copy() for candidate_id in _candidate_ids()}
    report = evaluate_synthetic_predictions(
        _batch(
            truth=baseline.truth_eur_mwh,
            predictions=predictions,
            index=baseline.delivery_at_utc,
        )
    )

    for score in report.scores:
        assert score.metrics[PRIMARY_METRIC].value == pytest.approx(0.0)
        assert score.metrics["MONTHLY_LEVEL_NEUTRALIZED_RMSE_EUR_MWH"].value == pytest.approx(0.0)
        assert score.metrics["MONTHLY_LEVEL_NEUTRALIZED_BIAS_EUR_MWH"].value == pytest.approx(0.0)


def test_input_arrays_are_copied_and_read_only() -> None:
    baseline = _batch()
    source_truth = baseline.truth_eur_mwh.copy()
    source_predictions = {key: value.copy() for key, value in baseline.predictions_eur_mwh.items()}
    batch = _batch(
        truth=source_truth,
        predictions=source_predictions,
        index=baseline.delivery_at_utc,
    )
    source_truth[0] = -999.0
    source_predictions[_candidate_ids()[0]][0] = -999.0

    assert batch.truth_eur_mwh[0] != -999.0
    assert batch.predictions_eur_mwh[_candidate_ids()[0]][0] != -999.0
    assert batch.truth_eur_mwh.flags.writeable is False
    with pytest.raises(ValueError):
        batch.truth_eur_mwh[0] = 1.0


def test_batch_rejects_inventory_origin_index_and_empty_intersection() -> None:
    baseline = _batch()
    missing = dict(baseline.predictions_eur_mwh)
    missing.pop(_candidate_ids()[0])
    with pytest.raises(EvaluationEngineError, match="exact frozen candidate inventory"):
        _batch(predictions=missing, index=baseline.delivery_at_utc)

    with pytest.raises(EvaluationEngineError, match="does not match"):
        SyntheticEvaluationSet(
            fixture_id="synthetic-invalid-origin",
            origin_slot_id="origin-2026-10",
            origin_as_of_utc=datetime(2026, 10, 7, 12, tzinfo=timezone.utc),
            delivery_at_utc=baseline.delivery_at_utc,
            truth_eur_mwh=baseline.truth_eur_mwh,
            predictions_eur_mwh=baseline.predictions_eur_mwh,
        )

    naive = baseline.delivery_at_utc.tz_localize(None)
    with pytest.raises(EvaluationEngineError, match="timezone-aware"):
        _batch(index=naive)

    all_missing = {
        candidate_id: np.full(len(baseline.delivery_at_utc), np.nan)
        for candidate_id in _candidate_ids()
    }
    with pytest.raises(EvaluationEngineError, match="intersection is empty"):
        evaluate_synthetic_predictions(
            _batch(predictions=all_missing, index=baseline.delivery_at_utc)
        )


def test_irregular_cadence_is_rejected() -> None:
    irregular = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-11-01 00:00", tz="UTC"),
            pd.Timestamp("2026-11-01 01:00", tz="UTC"),
            pd.Timestamp("2026-11-01 03:00", tz="UTC"),
        ]
    )
    with pytest.raises(EvaluationEngineError, match="regular UTC cadence"):
        _batch(index=irregular)


def test_delivery_outside_frozen_lead_horizon_is_rejected() -> None:
    lead_zero = pd.date_range("2026-10-07", periods=2, freq="h", tz="UTC")
    with pytest.raises(EvaluationEngineError, match="lead months 1 through 36"):
        _batch(index=lead_zero)

    lead_thirty_seven = pd.date_range("2029-11-01", periods=2, freq="h", tz="UTC")
    with pytest.raises(EvaluationEngineError, match="lead months 1 through 36"):
        _batch(index=lead_thirty_seven)
