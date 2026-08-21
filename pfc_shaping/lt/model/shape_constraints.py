"""Constraint primitives for the LT quant-shaping optimizer.

This module is intentionally small and explicit.  It builds average-price
constraint rows on a canonical UTC delivery index, while local market time is
used only for product masks and peak/offpeak definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pfc_shaping.calibration.eex_contract_selection import calibration_buckets

_COUNTRY_TZ = {
    "CH": "Europe/Zurich",
    "DE": "Europe/Berlin",
    "DE-LU": "Europe/Berlin",
    "AT": "Europe/Vienna",
    "FR": "Europe/Paris",
    "IT": "Europe/Rome",
    "IT_NORD": "Europe/Rome",
}



@dataclass(frozen=True)
class ConstraintRow:
    """One average-price hard constraint."""

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
    rank_a: int
    rank_augmented: int
    infeasibility_inf: float
    singular_values: tuple[float, ...]
    rank_tol: float
    feasible: bool


@dataclass(frozen=True)
class ConstraintSystem:
    """A hard-constraint matrix ``A p = q``."""

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

    def feasibility_report(self) -> FeasibilityReport:
        a_sparse = self.sparse_matrix
        q = self.targets
        if a_sparse.shape[0] == 0:
            return FeasibilityReport(0, 0, 0.0, tuple(), 0.0, True)
        active_cols = np.unique(a_sparse.indices)
        a = a_sparse[:, active_cols].toarray() if len(active_cols) else np.zeros((a_sparse.shape[0], 0))
        singular_values = np.linalg.svd(a, compute_uv=False)
        sigma_max = float(singular_values[0]) if len(singular_values) else 0.0
        rank_tol = max(a.shape) * np.finfo(float).eps * sigma_max * 100.0
        rank_a = int(np.sum(singular_values > rank_tol))
        augmented = np.column_stack([a, q])
        singular_aug = np.linalg.svd(augmented, compute_uv=False)
        sigma_aug = float(singular_aug[0]) if len(singular_aug) else 0.0
        rank_tol_aug = max(augmented.shape) * np.finfo(float).eps * sigma_aug * 100.0
        rank_aug = int(np.sum(singular_aug > rank_tol_aug))
        projected = a @ (np.linalg.pinv(a, rcond=rank_tol / sigma_max if sigma_max > 0.0 else 1e-12) @ q)
        infeasibility = float(np.max(np.abs(projected - q))) if len(q) else 0.0
        return FeasibilityReport(
            rank_a=rank_a,
            rank_augmented=rank_aug,
            infeasibility_inf=infeasibility,
            singular_values=tuple(float(v) for v in singular_values),
            rank_tol=float(rank_tol),
            feasible=bool(rank_a == rank_aug and infeasibility <= 1e-9),
        )


def validate_utc_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return a canonical sorted UTC index or raise on ambiguous inputs."""

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("delivery index must be a pandas DatetimeIndex")
    if index.tz is None:
        raise ValueError("delivery index must be timezone-aware")
    utc = index.tz_convert("UTC")
    if utc.has_duplicates:
        raise ValueError("delivery index has duplicate UTC timestamps")
    if not utc.is_monotonic_increasing:
        raise ValueError("delivery index must be sorted")
    return utc


def interval_hours(index: pd.DatetimeIndex) -> np.ndarray:
    """Infer delivery interval durations in hours on a UTC index."""

    utc = validate_utc_index(index)
    if len(utc) == 0:
        return np.array([], dtype=float)
    if len(utc) == 1:
        return np.array([1.0], dtype=float)
    ns = utc.view("int64")
    diffs = np.diff(ns).astype(float) / 3_600_000_000_000.0
    if np.any(diffs <= 0.0) or not np.isfinite(diffs).all():
        raise ValueError("delivery index must have positive finite intervals")
    if not np.allclose(diffs, diffs[0], rtol=0.0, atol=1e-12):
        raise ValueError("delivery index must have a regular UTC cadence")
    if not (np.isclose(diffs[0], 1.0, rtol=0.0, atol=1e-12) or np.isclose(diffs[0], 0.25, rtol=0.0, atol=1e-12)):
        raise ValueError("delivery index cadence must be hourly or 15-minute")
    last = diffs[-1]
    return np.concatenate([diffs, [last]]).astype(float)


def eex_peak_mask(index: pd.DatetimeIndex, *, country: str = "CH") -> np.ndarray:
    """Return an EEX-style peak mask on the given UTC delivery intervals."""

    code = country.upper()
    tz = _COUNTRY_TZ.get(code)
    if tz is None:
        raise ValueError(f"unsupported peak calendar country {country!r}")
    utc = validate_utc_index(index)
    local = utc.tz_convert(tz)
    return (
        (local.weekday < 5)
        & (local.hour >= 8)
        & (local.hour <= 19)
    )


def build_base_constraint_system(
    index: pd.DatetimeIndex,
    base_forward_prices: Mapping[str, float],
    *,
    country: str = "CH",
) -> ConstraintSystem:
    """Build quote-aware BASE average-price constraints."""

    code = country.upper()
    if code not in _COUNTRY_TZ:
        raise ValueError(f"unsupported BASE calendar country {country!r}")
    utc = validate_utc_index(index)
    local = pd.Series(utc.tz_convert(_COUNTRY_TZ[code]), index=range(len(utc)))
    buckets, targets = calibration_buckets(local, base_forward_prices)
    hours = interval_hours(utc)
    rows: list[ConstraintRow] = []
    for bucket, idx in buckets.groupby(buckets, dropna=True).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        weights = np.zeros(len(utc), dtype=float)
        denom = float(hours[idx].sum())
        if denom <= 0.0:
            raise ValueError(f"empty BASE constraint bucket {bucket}")
        weights[idx] = hours[idx] / denom
        rows.append(
            ConstraintRow(
                name=f"BASE:{bucket}",
                target=float(targets[str(bucket)]),
                weights=weights,
                kind="BASE",
                metadata={"bucket": str(bucket), "country": country.upper(), "hours": denom},
            )
        )
    return ConstraintSystem(tuple(rows), n_variables=len(utc))


def _source_quote_keys(bucket: str, prices: Mapping[str, float]) -> tuple[str, ...]:
    keys = []
    for key in prices:
        raw = str(key)
        if bucket == raw or bucket == f"{raw}-RESIDUAL":
            keys.append(raw)
    return tuple(keys)


def _raw_product_mask(local: pd.Series, product: str) -> np.ndarray:
    raw = str(product)
    if len(raw) == 4 and raw.isdigit():
        return (local.dt.year.to_numpy() == int(raw))
    if len(raw) == 7 and raw[4:6] == "-Q" and raw[6].isdigit():
        return (
            (local.dt.year.to_numpy() == int(raw[:4]))
            & (local.dt.quarter.to_numpy() == int(raw[6]))
        )
    if len(raw) == 7 and raw[4] == "-" and raw[5:].isdigit():
        return (
            (local.dt.year.to_numpy() == int(raw[:4]))
            & (local.dt.month.to_numpy() == int(raw[5:]))
        )
    return np.zeros(len(local), dtype=bool)


def build_base_peak_offpeak_constraint_system(
    index: pd.DatetimeIndex,
    base_forward_prices: Mapping[str, float],
    peak_forward_prices: Mapping[str, float],
    *,
    country: str = "CH",
) -> ConstraintSystem:
    """Build disjoint BASE-only or PEAK/OFFPEAK hard constraints.

    When a BASE bucket has a matching quoted PEAK bucket, the BASE row is
    replaced by disjoint PEAK and OFFPEAK rows.  The original BASE and PEAK
    residuals can then be audited from the resulting curve.
    """

    code = country.upper()
    if code not in _COUNTRY_TZ:
        raise ValueError(f"unsupported PEAK calendar country {country!r}")
    utc = validate_utc_index(index)
    base_system = build_base_constraint_system(utc, base_forward_prices, country=code)
    local = pd.Series(utc.tz_convert(_COUNTRY_TZ[code]), index=range(len(utc)))
    peak_mask = eex_peak_mask(utc, country=country)
    # A PEAK parent quote is an average over PEAK delivery intervals only.
    # Building its residual hierarchy on the full BASE calendar gives nested
    # month/quarter/year products the wrong energy weights, even if the final
    # constraint rows are subsequently masked to PEAK hours.  Preserve the
    # original row positions while deriving targets from the contractual
    # PEAK-only delivery set, then expand the bucket labels back to the full
    # horizon for row construction and source-coverage checks.
    peak_only_buckets, peak_targets = calibration_buckets(
        local.loc[peak_mask],
        peak_forward_prices,
    )
    peak_buckets = peak_only_buckets.reindex(local.index)
    hours = interval_hours(utc)
    rows: list[ConstraintRow] = []
    represented_peak_buckets: set[str] = set()
    for base_row in base_system.rows:
        bucket = str(base_row.metadata["bucket"])
        peak_idx = np.where((peak_buckets.to_numpy(dtype=object) == bucket) & peak_mask)[0]
        all_idx = np.where(base_row.weights > 0.0)[0]
        peak_index_set = set(peak_idx.tolist())
        offpeak_idx = np.array(
            [idx for idx in all_idx if idx not in peak_index_set],
            dtype=int,
        )
        if bucket not in peak_targets or len(peak_idx) == 0:
            rows.append(base_row)
            continue
        if len(offpeak_idx) == 0:
            raise ValueError(f"cannot build OFFPEAK constraint for {bucket}: no offpeak intervals")
        h_base = float(hours[all_idx].sum())
        h_peak = float(hours[peak_idx].sum())
        h_offpeak = float(hours[offpeak_idx].sum())
        peak_target = float(peak_targets[bucket])
        source_quotes = _source_quote_keys(bucket, peak_forward_prices)
        offpeak_target = (float(base_row.target) * h_base - peak_target * h_peak) / h_offpeak
        peak_weights = np.zeros(len(utc), dtype=float)
        peak_weights[peak_idx] = hours[peak_idx] / h_peak
        offpeak_weights = np.zeros(len(utc), dtype=float)
        offpeak_weights[offpeak_idx] = hours[offpeak_idx] / h_offpeak
        rows.extend(
            [
                ConstraintRow(
                    name=f"PEAK:{bucket}",
                    target=peak_target,
                    weights=peak_weights,
                    kind="PEAK",
                    metadata={"bucket": bucket, "country": country.upper(), "hours": h_peak, "source_quotes": source_quotes},
                ),
                ConstraintRow(
                    name=f"OFFPEAK:{bucket}",
                    target=offpeak_target,
                    weights=offpeak_weights,
                    kind="OFFPEAK",
                    metadata={"bucket": bucket, "country": country.upper(), "hours": h_offpeak},
                ),
            ]
        )
        represented_peak_buckets.add(bucket)
    for bucket, target in peak_targets.items():
        bucket = str(bucket)
        if bucket in represented_peak_buckets:
            continue
        peak_idx = np.where((peak_buckets.to_numpy(dtype=object) == bucket) & peak_mask)[0]
        if len(peak_idx) == 0:
            raise ValueError(f"PEAK quote {bucket} is not represented on the delivery horizon")
        h_peak = float(hours[peak_idx].sum())
        peak_weights = np.zeros(len(utc), dtype=float)
        peak_weights[peak_idx] = hours[peak_idx] / h_peak
        source_quotes = _source_quote_keys(bucket, peak_forward_prices)
        rows.append(
            ConstraintRow(
                name=f"PEAK:{bucket}",
                target=float(target),
                weights=peak_weights,
                kind="PEAK",
                metadata={"bucket": bucket, "country": code, "hours": h_peak, "source_quotes": source_quotes},
            )
        )
    represented_raw_quotes = {
        source
        for row in rows
        if row.kind == "PEAK"
        for source in row.metadata.get("source_quotes", ())
    }
    missing_raw_quotes = []
    peak_bucket_values = peak_buckets.to_numpy(dtype=object)
    for key in peak_forward_prices:
        raw = str(key)
        if raw in represented_raw_quotes:
            continue
        delivery_mask = _raw_product_mask(local, raw) & peak_mask
        # A coarser parent fully covered by finer quote buckets is represented
        # by the source hierarchy even though it adds no independent KKT row.
        hierarchy_covered = bool(delivery_mask.any()) and bool(
            pd.notna(peak_bucket_values[delivery_mask]).all()
        )
        if not hierarchy_covered:
            missing_raw_quotes.append(raw)
    missing_raw_quotes.sort()
    if missing_raw_quotes:
        raise ValueError(f"PEAK quotes not represented in constraint system: {missing_raw_quotes}")
    return ConstraintSystem(tuple(rows), n_variables=len(utc))
