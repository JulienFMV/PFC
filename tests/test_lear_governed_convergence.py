import warnings

import numpy as np

import pfc_shaping.model.lear_forecaster as learner_module
from pfc_shaping.model.lear_forecaster import LEARForecaster


def test_governed_elasticnet_retries_after_convergence_warning(monkeypatch) -> None:
    init_calls: list[tuple[float, int]] = []

    class FakeElasticNetCV:
        fit_calls = 0

        def __init__(self, *, l1_ratio, max_iter, cv, random_state):
            self.l1_ratio = float(l1_ratio)
            self.max_iter = int(max_iter)
            self.cv = cv
            self.random_state = random_state
            init_calls.append((self.l1_ratio, self.max_iter))

        def fit(self, X, y, sample_weight=None):
            type(self).fit_calls += 1
            if type(self).fit_calls == 1:
                warnings.warn("did not converge", learner_module.ConvergenceWarning)
            self.coef_ = np.zeros(X.shape[1], dtype=float)
            return self

    monkeypatch.setattr(learner_module, "ElasticNetCV", FakeElasticNetCV)
    model = LEARForecaster(use_governed_forecast_features=True, max_iter=2500, random_state=7)

    fitted = model._fit_elasticnet_with_retries(
        X_train=np.ones((32, 3), dtype=float),
        y_train=np.linspace(0.0, 1.0, 32),
        sample_weight=np.ones(32, dtype=float),
        cv=model._time_series_cv(32),
        hour=5,
        window=42,
    )

    assert init_calls == [(0.1, 2500), (0.7, 10000)]
    assert fitted.l1_ratio == 0.7
    assert fitted.max_iter == 10000
    status = model._governed_convergence_status["h05_w42"]
    assert status["recovered"] is True
    assert status["converged"] is True
    assert [attempt["converged"] for attempt in status["attempts"]] == [False, True]
