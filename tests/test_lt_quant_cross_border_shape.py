from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.model.cross_border_shape import (
    BucketProjectionSpec,
    NeighborWeightConfig,
    build_cross_border_shape_prior,
    project_neighbor_deviations,
    simplex_shrink_neighbor_weights,
)


def test_project_neighbor_deviations_preserves_bucket_mean() -> None:
    deviations = pd.DataFrame({"DE": [10.0, -3.0, 6.0, -8.0]}, index=pd.RangeIndex(4))
    spec = BucketProjectionSpec(
        bucket_names=("bucket",),
        step_weights=np.array([1.0, 2.0, 1.0, 2.0], dtype=float),
        constraint_matrix=np.array([[1.0, 2.0, 1.0, 2.0]], dtype=float),
    )

    result = project_neighbor_deviations(deviations, spec)

    weighted_mean = float(np.average(result["DE"].to_numpy(), weights=spec.step_weights))
    assert abs(weighted_mean) < 1e-12
    np.testing.assert_allclose(spec.constraint_matrix @ result["DE"].to_numpy(), [0.0], atol=1e-12)


def test_projection_is_invariant_to_neighbor_absolute_level_shift() -> None:
    deviations = pd.DataFrame({"DE": [3.0, -1.0, 2.0, -4.0]}, index=pd.RangeIndex(4))
    shifted = deviations + 50.0
    spec = BucketProjectionSpec(
        bucket_names=("bucket",),
        step_weights=np.ones(4, dtype=float),
        constraint_matrix=np.ones((1, 4), dtype=float),
    )

    base = project_neighbor_deviations(deviations, spec)
    plus_50 = project_neighbor_deviations(shifted, spec)

    pd.testing.assert_frame_equal(base, plus_50)


def test_projection_is_level_shift_invariant_with_partial_constraints() -> None:
    deviations = pd.DataFrame({"DE": [4.0, 6.0, 40.0, 45.0]}, index=pd.RangeIndex(4))
    shifted = deviations + 50.0
    spec = BucketProjectionSpec(
        bucket_names=("quoted_first_half",),
        step_weights=np.ones(4, dtype=float),
        constraint_matrix=np.array([[1.0, 1.0, 0.0, 0.0]], dtype=float),
    )

    base = project_neighbor_deviations(deviations, spec)
    plus_50 = project_neighbor_deviations(shifted, spec)

    pd.testing.assert_frame_equal(base, plus_50)


def test_projection_scales_to_hourly_year_without_dense_projector() -> None:
    index = pd.date_range("2030-01-01", periods=8760, freq="h", tz="Europe/Zurich")
    hours = np.arange(len(index), dtype=float)
    deviations = pd.DataFrame(
        {
            "DE": 80.0 + 10.0 * np.sin(2.0 * np.pi * hours / 8760.0),
            "FR": 65.0 + 6.0 * np.cos(2.0 * np.pi * hours / 720.0),
        },
        index=index,
    )
    constraint_rows = []
    for month in range(1, 13):
        constraint_rows.append((index.month == month).astype(float))
    spec = BucketProjectionSpec(
        bucket_names=tuple(f"2030-{month:02d}" for month in range(1, 13)),
        step_weights=np.ones(len(index), dtype=float),
        constraint_matrix=np.vstack(constraint_rows),
        tolerance=1e-9,
    )

    result = project_neighbor_deviations(deviations, spec)

    residual = spec.constraint_matrix @ result.to_numpy(dtype=float)
    assert float(np.max(np.abs(residual))) < 1e-9
    shifted = project_neighbor_deviations(deviations + 50.0, spec)
    pd.testing.assert_frame_equal(result, shifted)


def test_simplex_shrink_neighbor_weights_keeps_weights_non_negative_and_sums_to_one() -> None:
    scores = pd.DataFrame({"DE": [0.7], "FR": [0.2], "IT": [0.1]})
    config = NeighborWeightConfig(shrinkage_strength=0.5)

    weights = simplex_shrink_neighbor_weights(scores, config=config)

    assert {"DE", "FR", "IT", "ch_history_shrinkage"} == set(weights.columns)
    assert (weights.iloc[0] >= 0.0).all()
    assert abs(float(weights.iloc[0].sum()) - 1.0) < 1e-12
    assert weights.loc[0, "DE"] < 0.7


def test_cross_border_shape_prior_rejects_empty_weight_alignment() -> None:
    index = pd.date_range("2030-01-01", periods=4, freq="h", tz="UTC")
    neighbor = pd.DataFrame({"DE": [1.0, 2.0, 3.0, 4.0]}, index=index)
    scores = pd.DataFrame({"DE": [1.0, 1.0]}, index=pd.Index(["bucket-a", "bucket-b"]))
    spec = BucketProjectionSpec(
        bucket_names=("bucket",),
        step_weights=np.ones(4, dtype=float),
        constraint_matrix=np.ones((1, 4), dtype=float),
    )

    with pytest.raises(ValueError, match="weight index"):
        build_cross_border_shape_prior(neighbor, scores, spec)


def test_cross_border_shape_prior_uses_shapes_not_levels() -> None:
    index = pd.RangeIndex(4)
    neighbor_a = pd.DataFrame({"DE": [10.0, 12.0, 14.0, 16.0]}, index=index)
    neighbor_b = neighbor_a + 100.0
    scores = pd.DataFrame({"DE": [1.0]}, index=[0])
    spec = BucketProjectionSpec(
        bucket_names=("bucket",),
        step_weights=np.ones(4, dtype=float),
        constraint_matrix=np.ones((1, 4), dtype=float),
    )

    prior_a = build_cross_border_shape_prior(
        neighbor_a,
        scores,
        spec,
        config=NeighborWeightConfig(shrinkage_strength=0.0),
    )
    prior_b = build_cross_border_shape_prior(
        neighbor_b,
        scores,
        spec,
        config=NeighborWeightConfig(shrinkage_strength=0.0),
    )

    pd.testing.assert_series_equal(prior_a.shape_deviation, prior_b.shape_deviation)
    assert abs(float(np.average(prior_a.shape_deviation))) < 1e-12
