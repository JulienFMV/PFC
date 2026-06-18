"""Cross-border shape priors without neighbor level leakage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BucketProjectionSpec:
    constraint_matrix: np.ndarray
    bucket_names: tuple[str, ...]
    step_weights: np.ndarray
    tolerance: float = 1e-12


@dataclass(frozen=True)
class NeighborWeightConfig:
    max_total_neighbor_weight: float = 1.0
    shrinkage_strength: float = 1.0
    allow_negative_weights: bool = False


@dataclass(frozen=True)
class CrossBorderShapeResult:
    shape_deviation: pd.Series
    neighbor_deviations: pd.DataFrame
    weights: pd.DataFrame
    diagnostics: pd.DataFrame


def _validate_projection(spec: BucketProjectionSpec, n: int) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(spec.constraint_matrix, dtype=float)
    w = np.asarray(spec.step_weights, dtype=float)
    if a.ndim != 2 or a.shape[1] != n:
        raise ValueError(f"constraint_matrix shape {a.shape} incompatible with n={n}")
    if w.shape != (n,) or np.any(w <= 0.0) or not np.isfinite(w).all():
        raise ValueError("step_weights must be positive finite and match curve length")
    if len(spec.bucket_names) != a.shape[0]:
        raise ValueError("bucket_names length must match constraint rows")
    return a, w


def project_neighbor_deviations(
    neighbor_curves: pd.DataFrame,
    spec: BucketProjectionSpec,
) -> pd.DataFrame:
    """Project neighbor curves into the null space of CH level constraints."""

    if neighbor_curves.empty:
        return neighbor_curves.copy()
    a, w = _validate_projection(spec, len(neighbor_curves))
    level_row = w.reshape(1, -1)
    a_projection = np.vstack([a, level_row])
    inverse_weights = 1.0 / w
    middle = (a_projection * inverse_weights.reshape(1, -1)) @ a_projection.T
    values = neighbor_curves.to_numpy(dtype=float)
    multipliers = np.linalg.pinv(middle) @ (a_projection @ values)
    correction = inverse_weights.reshape(-1, 1) * (a_projection.T @ multipliers)
    projected = values - correction
    residual = a_projection @ projected
    max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
    if max_abs > float(spec.tolerance):
        raise ValueError(f"neighbor projection residual too large: {max_abs:.3e}")
    return pd.DataFrame(projected, index=neighbor_curves.index, columns=neighbor_curves.columns)


def simplex_shrink_neighbor_weights(
    raw_scores: pd.DataFrame,
    reliability: pd.DataFrame | None = None,
    config: NeighborWeightConfig = NeighborWeightConfig(),
) -> pd.DataFrame:
    """Convert raw neighbor scores into non-negative simplex weights."""

    scores = raw_scores.astype(float).copy()
    if not config.allow_negative_weights and (scores < 0.0).any().any():
        raise ValueError("negative neighbor scores require allow_negative_weights=True")
    scores = scores.clip(lower=0.0)
    if reliability is not None:
        scores = scores * reliability.reindex_like(scores).fillna(0.0).clip(lower=0.0, upper=1.0)
    row_sum = scores.sum(axis=1)
    denom = row_sum.replace(0.0, np.nan)
    weights = scores.div(denom, axis=0).fillna(0.0)
    mass = np.minimum(row_sum / (row_sum + float(config.shrinkage_strength)), 1.0)
    mass = mass.clip(upper=float(config.max_total_neighbor_weight))
    weights = weights.mul(mass, axis=0)
    weights["ch_history_shrinkage"] = 1.0 - weights.sum(axis=1)
    return weights


def build_cross_border_shape_prior(
    neighbor_curves: pd.DataFrame,
    raw_weight_scores: pd.DataFrame,
    projection_spec: BucketProjectionSpec,
    *,
    reliability: pd.DataFrame | None = None,
    config: NeighborWeightConfig = NeighborWeightConfig(),
) -> CrossBorderShapeResult:
    deviations = project_neighbor_deviations(neighbor_curves, projection_spec)
    weights = simplex_shrink_neighbor_weights(raw_weight_scores, reliability, config)
    common_markets = [c for c in deviations.columns if c in weights.columns]
    if not common_markets:
        shape = pd.Series(0.0, index=deviations.index, name="cross_border_shape_deviation")
    else:
        if len(weights) == 1:
            row = weights[common_markets].iloc[0].to_numpy(dtype=float)
            aligned_weights = pd.DataFrame(
                np.repeat(row.reshape(1, -1), len(deviations.index), axis=0),
                index=deviations.index,
                columns=common_markets,
            )
        else:
            aligned_weights = weights[common_markets].reindex(deviations.index).ffill()
            if aligned_weights.isna().all().all():
                raise ValueError("neighbor weight index does not overlap delivery index")
            aligned_weights = aligned_weights.fillna(0.0)
        if (aligned_weights.sum(axis=1) == 0.0).all() and weights[common_markets].sum().sum() > 0.0:
            raise ValueError("neighbor weights align to zero on the delivery index")
        shape = (deviations[common_markets] * aligned_weights).sum(axis=1)
        shape.name = "cross_border_shape_deviation"
    diagnostics = pd.DataFrame(
        {
            "market": deviations.columns,
            "max_abs_constraint_residual": [
                float(np.max(np.abs(np.asarray(projection_spec.constraint_matrix, dtype=float) @ deviations[col].to_numpy(dtype=float))))
                for col in deviations.columns
            ],
        }
    )
    return CrossBorderShapeResult(shape, deviations, weights, diagnostics)
