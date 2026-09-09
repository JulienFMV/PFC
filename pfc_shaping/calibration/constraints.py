"""Shared linear constraint primitives for calibration problems.

The classes in this module intentionally have no dependency on LT or CT model
code.  They represent hard average constraints in the form ``A x = q`` and
provide small diagnostics used by monthly and hourly calibration layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass(frozen=True)
class ConstraintRow:
    """One linear hard constraint row."""

    name: str
    target: float
    weights: np.ndarray
    kind: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=float)
        if weights.ndim != 1:
            raise ValueError("constraint weights must be a one-dimensional array")
        if not np.isfinite(weights).all():
            raise ValueError("constraint weights must be finite")
        if not np.isfinite(float(self.target)):
            raise ValueError("constraint target must be finite")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "target", float(self.target))


@dataclass(frozen=True)
class FeasibilityReport:
    """Rank and residual diagnostics for ``A x = q``."""

    rank_a: int
    rank_augmented: int
    infeasibility_inf: float
    singular_values: tuple[float, ...]
    rank_tol: float
    feasible: bool


@dataclass(frozen=True)
class ConstraintSystem:
    """A hard-constraint matrix ``A x = q``."""

    rows: tuple[ConstraintRow, ...]
    n_variables: int

    def __post_init__(self) -> None:
        if self.n_variables < 0:
            raise ValueError("n_variables must be non-negative")
        for row in self.rows:
            if row.weights.shape != (self.n_variables,):
                raise ValueError(
                    f"constraint row {row.name!r} has weights length {len(row.weights)} "
                    f"but n_variables={self.n_variables}"
                )

    @property
    def sparse_matrix(self) -> sp.csr_matrix:
        if not self.rows:
            return sp.csr_matrix((0, self.n_variables), dtype=float)
        return sp.vstack(
            [sp.csr_matrix(row.weights.reshape(1, -1)) for row in self.rows],
            format="csr",
        )

    @property
    def matrix(self) -> np.ndarray:
        if not self.rows:
            return np.zeros((0, self.n_variables), dtype=float)
        return np.vstack([row.weights for row in self.rows]).astype(float)

    @property
    def targets(self) -> np.ndarray:
        return np.array([row.target for row in self.rows], dtype=float)

    @property
    def names(self) -> list[str]:
        return [row.name for row in self.rows]

    def residuals(self, values: np.ndarray) -> pd.DataFrame:
        x = np.asarray(values, dtype=float)
        if x.shape != (self.n_variables,):
            raise ValueError(f"values shape {x.shape} != ({self.n_variables},)")
        achieved = self.sparse_matrix @ x
        target = self.targets
        return pd.DataFrame(
            {
                "name": self.names,
                "kind": [row.kind for row in self.rows],
                "target": target,
                "achieved": achieved,
                "abs_error": np.abs(achieved - target),
            }
        )

    def feasibility_report(self, *, tolerance: float = 1e-9) -> FeasibilityReport:
        a_sparse = self.sparse_matrix
        q = self.targets
        if a_sparse.shape[0] == 0:
            return FeasibilityReport(0, 0, 0.0, tuple(), 0.0, True)

        active_cols = np.unique(a_sparse.indices)
        if len(active_cols):
            a = a_sparse[:, active_cols].toarray()
        else:
            a = np.zeros((a_sparse.shape[0], 0), dtype=float)

        singular_values = np.linalg.svd(a, compute_uv=False)
        sigma_max = float(singular_values[0]) if len(singular_values) else 0.0
        rank_tol = max(a.shape) * np.finfo(float).eps * sigma_max * 100.0
        rank_a = int(np.sum(singular_values > rank_tol))

        augmented = np.column_stack([a, q])
        singular_aug = np.linalg.svd(augmented, compute_uv=False)
        sigma_aug = float(singular_aug[0]) if len(singular_aug) else 0.0
        rank_tol_aug = max(augmented.shape) * np.finfo(float).eps * sigma_aug * 100.0
        rank_aug = int(np.sum(singular_aug > rank_tol_aug))

        rcond = rank_tol / sigma_max if sigma_max > 0.0 else 1e-12
        projected = a @ (np.linalg.pinv(a, rcond=rcond) @ q)
        infeasibility = float(np.max(np.abs(projected - q))) if len(q) else 0.0

        return FeasibilityReport(
            rank_a=rank_a,
            rank_augmented=rank_aug,
            infeasibility_inf=infeasibility,
            singular_values=tuple(float(v) for v in singular_values),
            rank_tol=float(rank_tol),
            feasible=bool(rank_a == rank_aug and infeasibility <= tolerance),
        )
