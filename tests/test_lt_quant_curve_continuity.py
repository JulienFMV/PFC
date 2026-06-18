from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.model.quant_shape_optimizer import QuantShapeOptimizer
from pfc_shaping.lt.model.shape_constraints import ConstraintRow, ConstraintSystem


def _monthly_constraint_system(
    index: pd.DatetimeIndex,
    targets: dict[int, float],
    *,
    mask_index: pd.DatetimeIndex | None = None,
) -> ConstraintSystem:
    mask_index = index if mask_index is None else mask_index
    rows = []
    for month, target in targets.items():
        mask = (mask_index.month == month).astype(float)
        rows.append(
            ConstraintRow(
                name=f"2030-{month:02d}",
                target=target,
                weights=mask / mask.sum(),
                kind="BASE",
            )
        )
    return ConstraintSystem(rows=tuple(rows), n_variables=len(index))


def _month_boundary_jumps(curve: pd.Series) -> list[float]:
    return [
        abs(float(curve.iloc[i] - curve.iloc[i - 1]))
        for i in range(1, len(curve))
        if curve.index[i].month != curve.index[i - 1].month
    ]


def test_optimizer_smoothness_penalties_remove_flat_delta_month_cliffs() -> None:
    index = pd.date_range("2030-01-01", "2030-03-31 23:00", freq="h", tz="Europe/Zurich")
    prior = pd.Series(0.0, index=index)
    constraints = _monthly_constraint_system(index, {1: 0.0, 2: 10.0, 3: 0.0})

    projection_only = QuantShapeOptimizer(epsilon_ridge=1e-9).solve(prior, constraints)
    smooth = QuantShapeOptimizer(
        lambda_smooth_h=10.0,
        lambda_smooth_m=1000.0,
        lambda_seam=1000.0,
        epsilon_ridge=1e-9,
    ).solve(prior, constraints)

    assert max(_month_boundary_jumps(projection_only.curve)) > 9.9
    assert max(_month_boundary_jumps(smooth.curve)) < 2.5
    for month, target in {1: 0.0, 2: 10.0, 3: 0.0}.items():
        month_mean = float(smooth.curve[smooth.curve.index.month == month].mean())
        np.testing.assert_allclose(month_mean, target, atol=1e-8)


def test_seam_curvature_preserves_legitimate_linear_month_trend() -> None:
    index = pd.date_range("2030-01-01", "2030-03-31 23:00", freq="h", tz="Europe/Zurich")
    prior = pd.Series(0.0, index=index)
    constraints = _monthly_constraint_system(index, {1: 10.0, 2: 20.0, 3: 30.0})

    curvature_only = QuantShapeOptimizer(lambda_smooth_h=10.0, epsilon_ridge=1e-9).solve(prior, constraints)
    seam_weighted = QuantShapeOptimizer(
        lambda_smooth_h=10.0,
        lambda_smooth_m=1000.0,
        lambda_seam=1000.0,
        epsilon_ridge=1e-9,
    ).solve(prior, constraints)

    curvature_jumps = _month_boundary_jumps(curvature_only.curve)
    seam_jumps = _month_boundary_jumps(seam_weighted.curve)
    assert min(seam_jumps) > 1.0
    np.testing.assert_allclose(seam_jumps, curvature_jumps, rtol=0.15, atol=0.0)
    for month, target in {1: 10.0, 2: 20.0, 3: 30.0}.items():
        month_mean = float(seam_weighted.curve[seam_weighted.curve.index.month == month].mean())
        np.testing.assert_allclose(month_mean, target, atol=1e-8)


def test_monthly_penalties_use_market_timezone_for_utc_prior_index() -> None:
    local_index = pd.date_range("2030-01-01", "2030-03-31 23:00", freq="h", tz="Europe/Zurich")
    utc_index = local_index.tz_convert("UTC")
    targets = {1: 0.0, 2: 10.0, 3: 0.0}

    local_result = QuantShapeOptimizer(
        lambda_smooth_h=10.0,
        lambda_smooth_m=1000.0,
        lambda_seam=1000.0,
        epsilon_ridge=1e-9,
    ).solve(
        pd.Series(0.0, index=local_index),
        _monthly_constraint_system(local_index, targets),
    )
    utc_result = QuantShapeOptimizer(
        lambda_smooth_h=10.0,
        lambda_smooth_m=1000.0,
        lambda_seam=1000.0,
        calendar_timezone="Europe/Zurich",
        epsilon_ridge=1e-9,
    ).solve(
        pd.Series(0.0, index=utc_index),
        _monthly_constraint_system(utc_index, targets, mask_index=local_index),
    )

    np.testing.assert_allclose(utc_result.curve.to_numpy(), local_result.curve.to_numpy(), atol=1e-8)


def test_monthly_smoothness_requires_datetime_index() -> None:
    prior = pd.Series([0.0, 1.0, 0.0], index=pd.RangeIndex(3))
    rows = (
        ConstraintRow(name="all", target=0.0, weights=np.ones(3) / 3.0, kind="BASE"),
    )
    constraints = ConstraintSystem(rows=rows, n_variables=3)

    with pytest.raises(TypeError, match="DatetimeIndex"):
        QuantShapeOptimizer(lambda_smooth_m=1.0).solve(prior, constraints)
