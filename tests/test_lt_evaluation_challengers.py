from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.evaluation_challengers import (
    EXPECTED_LIGHTGBM_VERSION,
    ChallengerDependencyError,
    ChallengerError,
    SyntheticPredictionSet,
    SyntheticTrainingSet,
    _initial_mlp_parameters,
    _weighted_mlp_loss_gradient,
    fit_synthetic_challenger,
    recency_weights,
)

ORIGIN = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
FEATURE_NAMES = ("load", "solar")


def _training_set(*, rows: int = 48) -> SyntheticTrainingSet:
    observed = pd.date_range(end="2026-10-05 12:00", periods=rows, freq="h", tz="UTC")
    phase = np.linspace(-1.0, 1.0, rows)
    features = np.column_stack([phase, np.sin(np.pi * phase)])
    target = 1.5 * phase - 0.4 * np.sin(np.pi * phase) + 0.2
    return SyntheticTrainingSet(
        fixture_id="synthetic-training-basic",
        features=features,
        target=target,
        observed_at_utc=observed,
        origin_as_of_utc=ORIGIN,
        feature_names=FEATURE_NAMES,
    )


def _prediction_set(training: SyntheticTrainingSet) -> SyntheticPredictionSet:
    return SyntheticPredictionSet(
        fixture_id="synthetic-prediction-basic",
        features=training.features[:12],
        delivery_at_utc=pd.date_range("2026-11-01", periods=12, freq="h", tz="UTC"),
        origin_as_of_utc=ORIGIN,
        feature_names=FEATURE_NAMES,
    )


def test_training_fixture_is_copied_read_only_and_rejects_leakage() -> None:
    source = np.arange(12, dtype=float).reshape(6, 2)
    training = SyntheticTrainingSet(
        fixture_id="synthetic-copy-check",
        features=source,
        target=np.arange(6, dtype=float),
        observed_at_utc=pd.date_range("2026-10-01", periods=6, freq="h", tz="UTC"),
        origin_as_of_utc=ORIGIN,
        feature_names=FEATURE_NAMES,
    )
    source[0, 0] = -999.0

    assert training.features[0, 0] == 0.0
    assert training.features.flags.writeable is False
    assert training.target.flags.writeable is False
    with pytest.raises(ValueError):
        training.features[0, 0] = 1.0

    with pytest.raises(ChallengerError, match="strictly available before origin"):
        SyntheticTrainingSet(
            fixture_id="synthetic-leakage-check",
            features=np.ones((2, 1)),
            target=np.ones(2),
            observed_at_utc=pd.DatetimeIndex(
                [pd.Timestamp("2026-10-01", tz="UTC"), pd.Timestamp(ORIGIN)]
            ),
            origin_as_of_utc=ORIGIN,
            feature_names=("load",),
        )


def test_prediction_fixture_rejects_schema_origin_and_past_delivery() -> None:
    training = _training_set()
    prediction = _prediction_set(training)
    fitted = fit_synthetic_challenger(
        "ridge-linear",
        training,
        parameters={"alpha": "0.1"},
    )

    assert fitted.predict(prediction).shape == (12,)
    with pytest.raises(ChallengerError, match="feature schema"):
        fitted.predict(
            SyntheticPredictionSet(
                fixture_id="synthetic-schema-mismatch",
                features=prediction.features,
                delivery_at_utc=prediction.delivery_at_utc,
                origin_as_of_utc=ORIGIN,
                feature_names=("solar", "load"),
            )
        )
    with pytest.raises(ChallengerError, match="cannot precede"):
        SyntheticPredictionSet(
            fixture_id="synthetic-past-delivery",
            features=np.ones((1, 2)),
            delivery_at_utc=pd.DatetimeIndex([pd.Timestamp("2026-10-01", tz="UTC")]),
            origin_as_of_utc=ORIGIN,
            feature_names=FEATURE_NAMES,
        )


def test_recency_weights_use_the_frozen_origin_and_exact_half_life() -> None:
    observed = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-10-06 12:00", tz="UTC"),
            pd.Timestamp("2026-04-09 12:00", tz="UTC"),
            pd.Timestamp("2026-10-06 11:00", tz="UTC"),
        ]
    )
    weights = recency_weights(observed, ORIGIN, half_life_days=180.0)

    raw = weights / weights[-1]
    assert raw[1] / raw[0] == pytest.approx(2.0 ** ((365.0 - 180.0) / 180.0))
    assert weights.mean() == pytest.approx(1.0)
    assert weights.flags.writeable is False


def test_recency_weights_reject_numerical_underflow() -> None:
    observed = pd.DatetimeIndex([pd.Timestamp("2000-01-01", tz="UTC")])
    with pytest.raises(ChallengerError, match="numerically invalid"):
        recency_weights(observed, ORIGIN, half_life_days=1e-6)


def test_parameter_grid_and_incumbent_are_fail_closed() -> None:
    training = _training_set()
    with pytest.raises(ChallengerError, match="incumbent"):
        fit_synthetic_challenger("current-unweighted-mlp", training)
    with pytest.raises(ChallengerError, match="inventory mismatch"):
        fit_synthetic_challenger("ridge-linear", training)
    with pytest.raises(ChallengerError, match="outside the frozen tuning grid"):
        fit_synthetic_challenger(
            "ridge-linear",
            training,
            parameters={"alpha": "0.11"},
        )


@pytest.mark.parametrize(
    ("candidate_id", "parameters"),
    [
        ("ridge-linear", {"alpha": "0.1"}),
        ("spline-ridge-gam", {"alpha": "0.1", "n_knots": "5"}),
    ],
)
def test_sklearn_challengers_are_deterministic(
    candidate_id: str,
    parameters: dict[str, str],
) -> None:
    training = _training_set()
    prediction = _prediction_set(training)

    first = fit_synthetic_challenger(candidate_id, training, parameters=parameters)
    second = fit_synthetic_challenger(candidate_id, training, parameters=parameters)

    np.testing.assert_array_equal(first.predict(prediction), second.predict(prediction))
    manifest = first.to_manifest()
    assert manifest["training_fixture_id"] == training.fixture_id
    assert manifest["real_data_training_performed"] is False
    assert manifest["ranking_or_selection_performed"] is False
    assert first.authority.model_training_authorized is False
    assert first.authority.scientific_claim_authorized is False
    assert first.authority.production_authorized is False


def test_weighted_mlp_analytic_gradient_matches_central_difference() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(5, 2))
    target = rng.normal(size=5)
    normalized_weight = np.array([0.05, 0.1, 0.2, 0.25, 0.4])
    shapes = ((2, 3), (3, 2), (2, 1))
    theta = _initial_mlp_parameters(shapes, random_state=11)
    _, gradient = _weighted_mlp_loss_gradient(
        theta,
        features,
        target,
        normalized_weight,
        shapes,
        1e-5,
    )

    epsilon = 1e-6
    probe = np.array([0, 3, 7, 12, len(theta) - 1])
    numerical = []
    for position in probe:
        upper = theta.copy()
        lower = theta.copy()
        upper[position] += epsilon
        lower[position] -= epsilon
        loss_upper, _ = _weighted_mlp_loss_gradient(
            upper, features, target, normalized_weight, shapes, 1e-5
        )
        loss_lower, _ = _weighted_mlp_loss_gradient(
            lower, features, target, normalized_weight, shapes, 1e-5
        )
        numerical.append((loss_upper - loss_lower) / (2.0 * epsilon))

    np.testing.assert_allclose(gradient[probe], numerical, rtol=2e-5, atol=2e-6)


def test_recency_weighted_mlp_fits_and_predicts_deterministically() -> None:
    training = _training_set(rows=32)
    prediction = _prediction_set(training)

    first = fit_synthetic_challenger("recency-weighted-mlp", training)
    second = fit_synthetic_challenger("recency-weighted-mlp", training)

    np.testing.assert_allclose(first.predict(prediction), second.predict(prediction), atol=0.0)


def test_lightgbm_runtime_is_exact_or_fails_explicitly() -> None:
    training = _training_set()
    parameters = {
        "learning_rate": "0.03",
        "min_data_in_leaf": "50",
        "n_estimators": "300",
        "num_leaves": "15",
    }
    try:
        installed = metadata.version("lightgbm")
    except metadata.PackageNotFoundError:
        installed = None

    if installed != EXPECTED_LIGHTGBM_VERSION:
        with pytest.raises(ChallengerDependencyError):
            fit_synthetic_challenger(
                "lightgbm-deterministic-cpu",
                training,
                parameters=parameters,
            )
    else:
        prediction = _prediction_set(training)
        first = fit_synthetic_challenger(
            "lightgbm-deterministic-cpu",
            training,
            parameters=parameters,
        ).predict(prediction)
        second = fit_synthetic_challenger(
            "lightgbm-deterministic-cpu",
            training,
            parameters=parameters,
        ).predict(prediction)
        np.testing.assert_array_equal(first, second)
