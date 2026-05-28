import warnings

import numpy as np

import pfc_shaping.model.lear_forecaster as learner_module
from pfc_shaping.model.lear_forecaster import LEARForecaster


def test_governed_elasticnet_ignores_benign_cv_path_warning(monkeypatch) -> None:
    cv_init_calls: list[tuple[float, int]] = []
    final_init_calls: list[tuple[float, float, int]] = []

    class FakeElasticNetCV:
        def __init__(self, *, l1_ratio, max_iter, cv, random_state):
            self.l1_ratio = float(l1_ratio)
            self.max_iter = int(max_iter)
            self.cv = cv
            self.random_state = random_state
            self.alpha_ = 0.123
            self.l1_ratio_ = self.l1_ratio
            cv_init_calls.append((self.l1_ratio, self.max_iter))

        def fit(self, X, y, sample_weight=None):
            warnings.warn("cv path warning", learner_module.ConvergenceWarning)
            self.coef_ = np.zeros(X.shape[1], dtype=float)
            return self

    class FakeElasticNet:
        def __init__(self, *, alpha, l1_ratio, max_iter, random_state):
            final_init_calls.append((float(alpha), float(l1_ratio), int(max_iter)))

        def fit(self, X, y, sample_weight=None):
            return self

    monkeypatch.setattr(learner_module, "ElasticNetCV", FakeElasticNetCV)
    monkeypatch.setattr(learner_module, "ElasticNet", FakeElasticNet)
    model = LEARForecaster(use_governed_forecast_features=True, max_iter=2500, random_state=7)

    fitted = model._fit_elasticnet_with_retries(
        X_train=np.ones((32, 3), dtype=float),
        y_train=np.linspace(0.0, 1.0, 32),
        sample_weight=np.ones(32, dtype=float),
        cv=model._time_series_cv(32),
        hour=5,
        window=42,
    )

    assert cv_init_calls == [(0.1, 2500)]
    assert final_init_calls == [(0.123, 0.1, 2500)]
    assert fitted.l1_ratio == 0.1
    status = model._governed_convergence_status["h05_w42"]
    assert status["recovered"] is False
    assert status["converged"] is True
    assert status["attempts"][0]["cv_path_warned"] is True
    assert status["attempts"][0]["final_model_warned"] is False


def test_governed_elasticnet_retries_only_when_final_model_warns(monkeypatch) -> None:
    cv_init_calls: list[tuple[float, int]] = []
    final_fit_calls = 0

    class FakeElasticNetCV:
        def __init__(self, *, l1_ratio, max_iter, cv, random_state):
            self.l1_ratio = float(l1_ratio)
            self.max_iter = int(max_iter)
            self.cv = cv
            self.random_state = random_state
            self.alpha_ = 0.123
            self.l1_ratio_ = self.l1_ratio
            cv_init_calls.append((self.l1_ratio, self.max_iter))

        def fit(self, X, y, sample_weight=None):
            warnings.warn("cv path warning", learner_module.ConvergenceWarning)
            self.coef_ = np.zeros(X.shape[1], dtype=float)
            return self

    class FakeElasticNet:
        def __init__(self, *, alpha, l1_ratio, max_iter, random_state):
            self.l1_ratio = float(l1_ratio)

        def fit(self, X, y, sample_weight=None):
            nonlocal final_fit_calls
            final_fit_calls += 1
            if final_fit_calls == 1:
                warnings.warn("final model warning", learner_module.ConvergenceWarning)
            return self

    monkeypatch.setattr(learner_module, "ElasticNetCV", FakeElasticNetCV)
    monkeypatch.setattr(learner_module, "ElasticNet", FakeElasticNet)
    model = LEARForecaster(use_governed_forecast_features=True, max_iter=2500, random_state=7)

    fitted = model._fit_elasticnet_with_retries(
        X_train=np.ones((32, 3), dtype=float),
        y_train=np.linspace(0.0, 1.0, 32),
        sample_weight=np.ones(32, dtype=float),
        cv=model._time_series_cv(32),
        hour=5,
        window=42,
    )

    assert cv_init_calls == [(0.1, 2500), (0.7, 10000)]
    assert fitted.l1_ratio == 0.7
    status = model._governed_convergence_status["h05_w42"]
    assert status["recovered"] is True
    assert status["converged"] is True
    assert [attempt["final_model_warned"] for attempt in status["attempts"]] == [True, False]
