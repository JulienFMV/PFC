"""Synthetic-only implementations of the frozen LT challenger families.

This module exists to qualify algorithms and interfaces without granting a
real-data training path.  Inputs are immutable synthetic fixtures, temporal
leakage is rejected, and every fitted object remains explicitly non-scientific
and non-production.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from importlib import metadata
from types import MappingProxyType
from typing import Mapping, Protocol

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from pfc_shaping.lt.evaluation_protocol import (
    CandidateRole,
    EvaluationAuthority,
    ModelFamily,
    default_evaluation_protocol,
)

SYNTHETIC_PROVENANCE = "SYNTHETIC_FIXTURE_ONLY"
EXPECTED_LIGHTGBM_VERSION = "4.6.0"
RECENCY_HALF_LIFE_DAYS = 180.0
_SECONDS_PER_DAY = 86_400.0


class ChallengerError(ValueError):
    """Raised when a synthetic challenger request violates its contract."""


class ChallengerDependencyError(RuntimeError):
    """Raised when an exact optional runtime dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class SyntheticTrainingSet:
    """Immutable point-in-time training fixture; never an empirical dataset."""

    fixture_id: str
    features: np.ndarray
    target: np.ndarray
    observed_at_utc: pd.DatetimeIndex
    origin_as_of_utc: datetime
    feature_names: tuple[str, ...]
    provenance: str = field(default=SYNTHETIC_PROVENANCE, init=False)

    def __post_init__(self) -> None:
        _require_fixture_id(self.fixture_id)
        features = _readonly_float_array(self.features, ndim=2, label="features")
        target = _readonly_float_array(self.target, ndim=1, label="target")
        if features.shape[0] != target.shape[0]:
            raise ChallengerError("features and target must have the same row count")
        if features.shape[0] < 2 or features.shape[1] < 1:
            raise ChallengerError("training fixtures require at least two rows and one feature")
        names = _validate_feature_names(self.feature_names, features.shape[1])
        observed = _canonical_utc_index(self.observed_at_utc, label="observed_at_utc")
        if len(observed) != features.shape[0]:
            raise ChallengerError("observed_at_utc must align one-to-one with training rows")
        origin = _canonical_utc_timestamp(self.origin_as_of_utc, label="origin_as_of_utc")
        if bool((observed >= origin).any()):
            raise ChallengerError(
                "every training observation must be strictly available before origin"
            )
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "observed_at_utc", observed)
        object.__setattr__(self, "origin_as_of_utc", origin.to_pydatetime())


@dataclass(frozen=True, slots=True)
class SyntheticPredictionSet:
    """Immutable future feature fixture aligned to one frozen origin."""

    fixture_id: str
    features: np.ndarray
    delivery_at_utc: pd.DatetimeIndex
    origin_as_of_utc: datetime
    feature_names: tuple[str, ...]
    provenance: str = field(default=SYNTHETIC_PROVENANCE, init=False)

    def __post_init__(self) -> None:
        _require_fixture_id(self.fixture_id)
        features = _readonly_float_array(self.features, ndim=2, label="features")
        if features.shape[0] < 1 or features.shape[1] < 1:
            raise ChallengerError("prediction fixtures require at least one row and one feature")
        names = _validate_feature_names(self.feature_names, features.shape[1])
        delivery = _canonical_utc_index(self.delivery_at_utc, label="delivery_at_utc")
        if len(delivery) != features.shape[0]:
            raise ChallengerError("delivery_at_utc must align one-to-one with prediction rows")
        origin = _canonical_utc_timestamp(self.origin_as_of_utc, label="origin_as_of_utc")
        if bool((delivery < origin).any()):
            raise ChallengerError("prediction delivery rows cannot precede the frozen origin")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "delivery_at_utc", delivery)
        object.__setattr__(self, "origin_as_of_utc", origin.to_pydatetime())


class _Predictor(Protocol):
    def predict(self, features: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class _LightGBMPredictor:
    model: object
    columns: tuple[str, ...]

    def predict(self, features: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(np.asarray(features, dtype=float), columns=self.columns)
        prediction = getattr(self.model, "predict")(frame)
        return np.asarray(prediction, dtype=float)


@dataclass(frozen=True, slots=True)
class FittedSyntheticCandidate:
    """A fitted synthetic diagnostic carrying no selection authority."""

    candidate_id: str
    family: ModelFamily
    feature_names: tuple[str, ...]
    parameters: Mapping[str, str]
    fitted_at_origin_utc: datetime
    training_fixture_id: str
    _predictor: _Predictor = field(repr=False)
    provenance: str = field(default=SYNTHETIC_PROVENANCE, init=False)
    authority: EvaluationAuthority = field(default_factory=EvaluationAuthority, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.family, ModelFamily):
            raise TypeError("family must be a ModelFamily")
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ChallengerError("fitted feature schema must be non-empty and unique")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    def to_manifest(self) -> dict[str, object]:
        return {
            "provenance": self.provenance,
            "candidate_id": self.candidate_id,
            "family": self.family.value,
            "training_fixture_id": self.training_fixture_id,
            "fitted_at_origin_utc": self.fitted_at_origin_utc.isoformat(),
            "selected_tuning_parameters": dict(self.parameters),
            "real_data_training_performed": False,
            "ranking_or_selection_performed": False,
            "authority": self.authority.to_manifest(),
        }

    def predict(self, batch: SyntheticPredictionSet) -> np.ndarray:
        if not isinstance(batch, SyntheticPredictionSet):
            raise TypeError("batch must be a SyntheticPredictionSet")
        if batch.feature_names != self.feature_names:
            raise ChallengerError("prediction feature schema differs from the fitted schema")
        if batch.origin_as_of_utc != self.fitted_at_origin_utc:
            raise ChallengerError("prediction origin differs from the fitted origin")
        prediction = np.asarray(self._predictor.predict(batch.features), dtype=float)
        if prediction.shape != (len(batch.delivery_at_utc),):
            raise ChallengerError("challenger returned an invalid prediction shape")
        if not np.isfinite(prediction).all():
            raise ChallengerError("challenger returned non-finite predictions")
        prediction.setflags(write=False)
        return prediction


def recency_weights(
    observed_at_utc: pd.DatetimeIndex,
    origin_as_of_utc: datetime,
    *,
    half_life_days: float = RECENCY_HALF_LIFE_DAYS,
) -> np.ndarray:
    """Return exact origin-relative exponential weights with mean one."""

    observed = _canonical_utc_index(observed_at_utc, label="observed_at_utc")
    origin = _canonical_utc_timestamp(origin_as_of_utc, label="origin_as_of_utc")
    if not np.isfinite(half_life_days) or half_life_days <= 0.0:
        raise ChallengerError("half_life_days must be positive and finite")
    if len(observed) == 0:
        raise ChallengerError("observed_at_utc cannot be empty")
    if bool((observed >= origin).any()):
        raise ChallengerError("recency weights require observations strictly before origin")
    age_days = (origin - observed).total_seconds().to_numpy(dtype=float) / _SECONDS_PER_DAY
    weights = np.exp2(-age_days / float(half_life_days))
    mean = float(weights.mean())
    if not np.isfinite(weights).all() or bool((weights <= 0.0).any()) or mean <= 0.0:
        raise ChallengerError("recency weights are numerically invalid")
    weights = np.asarray(weights / mean, dtype=float)
    weights.setflags(write=False)
    return weights


def fit_synthetic_challenger(
    candidate_id: str,
    training: SyntheticTrainingSet,
    *,
    parameters: Mapping[str, str] | None = None,
) -> FittedSyntheticCandidate:
    """Fit one challenger for synthetic verification only.

    This is deliberately not a real-data or governed-origin entry point.  The
    returned object's authority remains fully negative.
    """

    if not isinstance(training, SyntheticTrainingSet):
        raise TypeError("training must be a SyntheticTrainingSet")
    protocol = default_evaluation_protocol()
    candidate = next(
        (item for item in protocol.candidates if item.candidate_id == candidate_id),
        None,
    )
    if candidate is None:
        raise ChallengerError(f"unknown candidate_id {candidate_id!r}")
    if candidate.role is not CandidateRole.CHALLENGER:
        raise ChallengerError("the incumbent cannot be fitted by the synthetic challenger lab")
    selected = _validate_parameters(candidate.tuning_grid, parameters)

    if candidate.family is ModelFamily.RECENCY_WEIGHTED_MLP:
        predictor: _Predictor = _WeightedMLPRegressor()
        predictor.fit(
            training.features,
            training.target,
            sample_weight=recency_weights(
                training.observed_at_utc,
                training.origin_as_of_utc,
            ),
        )
    elif candidate.family is ModelFamily.RIDGE:
        predictor = make_pipeline(
            StandardScaler(),
            Ridge(alpha=float(selected["alpha"]), fit_intercept=True),
        )
        predictor.fit(training.features, training.target)
    elif candidate.family is ModelFamily.SPLINE_RIDGE_GAM:
        predictor = make_pipeline(
            SplineTransformer(
                degree=3,
                n_knots=int(selected["n_knots"]),
                include_bias=False,
            ),
            StandardScaler(),
            Ridge(alpha=float(selected["alpha"]), fit_intercept=True),
        )
        predictor.fit(training.features, training.target)
    elif candidate.family is ModelFamily.LIGHTGBM:
        predictor = _fit_lightgbm(training, selected)
    else:  # pragma: no cover - enum exhaustiveness guard
        raise ChallengerError(f"unsupported challenger family {candidate.family.value!r}")

    return FittedSyntheticCandidate(
        candidate_id=candidate.candidate_id,
        family=candidate.family,
        feature_names=training.feature_names,
        parameters=selected,
        fitted_at_origin_utc=training.origin_as_of_utc,
        training_fixture_id=training.fixture_id,
        _predictor=predictor,
    )


class _WeightedMLPRegressor:
    """Deterministic 64x64 ReLU MLP with an exact weighted squared loss."""

    _hidden_sizes = (64, 64)

    def __init__(
        self,
        *,
        alpha: float = 1e-5,
        max_iter: int = 500,
        tol: float = 1e-8,
        random_state: int = 42,
    ) -> None:
        if not np.isfinite(alpha) or alpha < 0.0:
            raise ChallengerError("weighted MLP alpha must be finite and non-negative")
        if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
            raise ChallengerError("weighted MLP max_iter must be a positive integer")
        if not np.isfinite(tol) or tol <= 0.0:
            raise ChallengerError("weighted MLP tolerance must be positive and finite")
        if not isinstance(random_state, int) or isinstance(random_state, bool):
            raise ChallengerError("weighted MLP random_state must be an integer")
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self._theta: np.ndarray | None = None
        self._shapes: tuple[tuple[int, int], ...] | None = None
        self._x_mean: np.ndarray | None = None
        self._x_scale: np.ndarray | None = None
        self._y_mean: float | None = None
        self._y_scale: float | None = None

    def fit(
        self,
        features: np.ndarray,
        target: np.ndarray,
        *,
        sample_weight: np.ndarray,
    ) -> _WeightedMLPRegressor:
        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float)
        weights = np.asarray(sample_weight, dtype=float)
        if x.ndim != 2 or y.shape != (x.shape[0],) or weights.shape != y.shape:
            raise ChallengerError("weighted MLP inputs have incompatible shapes")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ChallengerError("weighted MLP inputs must be finite")
        if not np.isfinite(weights).all() or bool((weights <= 0.0).any()):
            raise ChallengerError("weighted MLP weights must be positive and finite")

        weight_sum = float(weights.sum())
        self._x_mean = np.average(x, axis=0, weights=weights)
        variance = np.average((x - self._x_mean) ** 2, axis=0, weights=weights)
        self._x_scale = np.where(variance > 1e-24, np.sqrt(variance), 1.0)
        self._y_mean = float(np.average(y, weights=weights))
        y_variance = float(np.average((y - self._y_mean) ** 2, weights=weights))
        self._y_scale = float(np.sqrt(y_variance)) if y_variance > 1e-24 else 1.0
        x_scaled = (x - self._x_mean) / self._x_scale
        y_scaled = (y - self._y_mean) / self._y_scale

        layer_sizes = (x.shape[1], *self._hidden_sizes, 1)
        self._shapes = tuple(zip(layer_sizes[:-1], layer_sizes[1:]))
        theta0 = _initial_mlp_parameters(self._shapes, self.random_state)
        result = minimize(
            _weighted_mlp_loss_gradient,
            theta0,
            args=(x_scaled, y_scaled, weights / weight_sum, self._shapes, self.alpha),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": self.max_iter,
                "ftol": self.tol,
                "gtol": self.tol,
                "maxls": 50,
            },
        )
        if not result.success:
            raise ChallengerError(f"weighted MLP optimization failed: {result.message}")
        theta = np.asarray(result.x, dtype=float)
        if not np.isfinite(theta).all():
            raise ChallengerError("weighted MLP optimization returned non-finite parameters")
        theta.setflags(write=False)
        self._theta = theta
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if (
            self._theta is None
            or self._shapes is None
            or self._x_mean is None
            or self._x_scale is None
            or self._y_mean is None
            or self._y_scale is None
        ):
            raise ChallengerError("weighted MLP is not fitted")
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self._x_mean) or not np.isfinite(x).all():
            raise ChallengerError("weighted MLP prediction features are invalid")
        prediction = _mlp_forward((x - self._x_mean) / self._x_scale, self._theta, self._shapes)
        return np.asarray(prediction * self._y_scale + self._y_mean, dtype=float)


def _weighted_mlp_loss_gradient(
    theta: np.ndarray,
    features: np.ndarray,
    target: np.ndarray,
    normalized_weight: np.ndarray,
    shapes: tuple[tuple[int, int], ...],
    alpha: float,
) -> tuple[float, np.ndarray]:
    parameters = _unpack_mlp_parameters(theta, shapes)
    activations = [features]
    preactivations: list[np.ndarray] = []
    current = features
    for index, (weight, bias) in enumerate(parameters):
        linear = current @ weight + bias
        preactivations.append(linear)
        current = np.maximum(linear, 0.0) if index < len(parameters) - 1 else linear
        activations.append(current)
    prediction = current[:, 0]
    residual = prediction - target
    penalty = 0.5 * alpha * sum(float(np.square(weight).sum()) for weight, _ in parameters)
    loss = float(np.dot(normalized_weight, np.square(residual)) + penalty)

    delta = (2.0 * normalized_weight * residual)[:, None]
    gradients: list[tuple[np.ndarray, np.ndarray]] = []
    for layer in range(len(parameters) - 1, -1, -1):
        weight, _ = parameters[layer]
        grad_weight = activations[layer].T @ delta + alpha * weight
        grad_bias = delta.sum(axis=0)
        gradients.append((grad_weight, grad_bias))
        if layer:
            delta = (delta @ weight.T) * (preactivations[layer - 1] > 0.0)
    gradients.reverse()
    packed = np.concatenate(
        [part.ravel() for grad_weight, grad_bias in gradients for part in (grad_weight, grad_bias)]
    )
    return loss, packed


def _mlp_forward(
    features: np.ndarray,
    theta: np.ndarray,
    shapes: tuple[tuple[int, int], ...],
) -> np.ndarray:
    current = features
    parameters = _unpack_mlp_parameters(theta, shapes)
    for index, (weight, bias) in enumerate(parameters):
        current = current @ weight + bias
        if index < len(parameters) - 1:
            current = np.maximum(current, 0.0)
    return current[:, 0]


def _initial_mlp_parameters(
    shapes: tuple[tuple[int, int], ...],
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    parts: list[np.ndarray] = []
    for fan_in, fan_out in shapes:
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        parts.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)).ravel())
        parts.append(np.zeros(fan_out, dtype=float))
    return np.concatenate(parts)


def _unpack_mlp_parameters(
    theta: np.ndarray,
    shapes: tuple[tuple[int, int], ...],
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    offset = 0
    parameters: list[tuple[np.ndarray, np.ndarray]] = []
    for fan_in, fan_out in shapes:
        weight_size = fan_in * fan_out
        weight = theta[offset : offset + weight_size].reshape(fan_in, fan_out)
        offset += weight_size
        bias = theta[offset : offset + fan_out]
        offset += fan_out
        parameters.append((weight, bias))
    if offset != len(theta):
        raise ChallengerError("weighted MLP parameter vector has an invalid length")
    return tuple(parameters)


def _fit_lightgbm(
    training: SyntheticTrainingSet,
    parameters: Mapping[str, str],
) -> _Predictor:
    try:
        installed_version = metadata.version("lightgbm")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                module=r"matplotlib\._fontconfig_pattern",
            )
            from lightgbm import LGBMRegressor
    except (metadata.PackageNotFoundError, ImportError) as exc:
        raise ChallengerDependencyError(
            f"LightGBM {EXPECTED_LIGHTGBM_VERSION} is required for this optional challenger"
        ) from exc
    if installed_version != EXPECTED_LIGHTGBM_VERSION:
        raise ChallengerDependencyError(
            f"LightGBM runtime mismatch: expected {EXPECTED_LIGHTGBM_VERSION}, "
            f"observed {installed_version}"
        )
    model = LGBMRegressor(
        objective="regression_l1",
        device_type="cpu",
        deterministic=True,
        force_col_wise=True,
        n_jobs=1,
        random_state=42,
        bagging_seed=42,
        data_random_seed=42,
        feature_fraction_seed=42,
        learning_rate=float(parameters["learning_rate"]),
        min_data_in_leaf=int(parameters["min_data_in_leaf"]),
        n_estimators=int(parameters["n_estimators"]),
        num_leaves=int(parameters["num_leaves"]),
        verbosity=-1,
    )
    columns = tuple(f"feature_{position:04d}" for position in range(training.features.shape[1]))
    model.fit(pd.DataFrame(training.features, columns=columns), training.target)
    return _LightGBMPredictor(model=model, columns=columns)


def _validate_parameters(
    tuning_grid: tuple[tuple[str, tuple[str, ...]], ...],
    parameters: Mapping[str, str] | None,
) -> Mapping[str, str]:
    selected = {} if parameters is None else dict(parameters)
    expected = {name: values for name, values in tuning_grid}
    if set(selected) != set(expected):
        raise ChallengerError(
            f"parameter inventory mismatch: expected {sorted(expected)}, got {sorted(selected)}"
        )
    for name, value in selected.items():
        if not isinstance(value, str) or value not in expected[name]:
            raise ChallengerError(f"parameter {name!r} is outside the frozen tuning grid")
    return MappingProxyType(dict(sorted(selected.items())))


def _readonly_float_array(value: object, *, ndim: int, label: str) -> np.ndarray:
    try:
        array = np.array(value, dtype=float, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ChallengerError(f"{label} must be numeric") from exc
    if array.ndim != ndim:
        raise ChallengerError(f"{label} must be {ndim}-dimensional")
    if not np.isfinite(array).all():
        raise ChallengerError(f"{label} must contain only finite values")
    array.setflags(write=False)
    return array


def _validate_feature_names(value: object, count: int) -> tuple[str, ...]:
    if not isinstance(value, tuple) or len(value) != count:
        raise ChallengerError("feature_names must be a tuple matching the feature width")
    if len(set(value)) != len(value) or any(
        not isinstance(name, str) or not name.strip() for name in value
    ):
        raise ChallengerError("feature_names must be unique non-empty strings")
    return value


def _canonical_utc_index(value: object, *, label: str) -> pd.DatetimeIndex:
    if not isinstance(value, pd.DatetimeIndex) or value.tz is None:
        raise ChallengerError(f"{label} must be a timezone-aware DatetimeIndex")
    index = value.tz_convert("UTC")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ChallengerError(f"{label} must be unique and strictly ordered")
    return index


def _canonical_utc_timestamp(value: object, *, label: str) -> pd.Timestamp:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ChallengerError(f"{label} must be a timezone-aware datetime")
    timestamp = pd.Timestamp(value)
    if timestamp.utcoffset() != timedelta(0):
        raise ChallengerError(f"{label} must use UTC")
    return timestamp.tz_convert("UTC")


def _require_fixture_id(value: object) -> None:
    if not isinstance(value, str) or not value.startswith("synthetic-") or len(value) < 12:
        raise ChallengerError("fixture_id must start with 'synthetic-' and be descriptive")


__all__ = [
    "ChallengerDependencyError",
    "ChallengerError",
    "EXPECTED_LIGHTGBM_VERSION",
    "FittedSyntheticCandidate",
    "RECENCY_HALF_LIFE_DAYS",
    "SYNTHETIC_PROVENANCE",
    "SyntheticPredictionSet",
    "SyntheticTrainingSet",
    "fit_synthetic_challenger",
    "recency_weights",
]
