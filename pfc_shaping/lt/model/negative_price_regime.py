"""Scenario-path helpers for negative price regime diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

Measure = Literal["P", "Q"]
Verdict = Literal["PASS", "CONDITIONAL", "NO_GO"]


@dataclass(frozen=True)
class ScenarioBundle:
    paths: pd.DataFrame
    measure: Measure
    base_curve: pd.Series
    risk_premium: pd.Series | None = None
    label: str = "production"


@dataclass(frozen=True)
class QuantileFan:
    quantiles: pd.DataFrame
    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class ReconciliationReport:
    reconciled: ScenarioBundle
    constraint_residuals: pd.DataFrame
    tail_diagnostics: pd.DataFrame
    block_vwap_diagnostics: pd.DataFrame
    verdict: Verdict


def _align_residuals(base_curve: pd.Series, residuals: pd.DataFrame) -> pd.DataFrame:
    if residuals.empty:
        raise ValueError("residual paths are empty")
    out = residuals.copy()
    out = out.reindex(index=base_curve.index)
    if out.isna().any().any():
        raise ValueError("residual paths must cover every base curve timestamp")
    return out.astype(float)


def build_physical_paths(
    p_curve: pd.Series,
    residuals: pd.DataFrame,
    *,
    risk_premium: pd.Series | None,
    allow_diagnostic_without_rp: bool = True,
) -> ScenarioBundle:
    eps = _align_residuals(p_curve, residuals)
    if risk_premium is None:
        if not allow_diagnostic_without_rp:
            raise ValueError("physical production paths require a risk premium")
        rp = pd.Series(0.0, index=p_curve.index)
        label = "diagnostic"
    else:
        rp = risk_premium.reindex(p_curve.index).astype(float)
        if rp.isna().any():
            raise ValueError("risk premium must cover every base curve timestamp")
        label = "production"
    paths = eps.add(p_curve - rp, axis=0)
    return ScenarioBundle(paths=paths, measure="P", base_curve=p_curve, risk_premium=rp, label=label)


def build_risk_neutral_paths(p_curve: pd.Series, residuals: pd.DataFrame) -> ScenarioBundle:
    eps = _align_residuals(p_curve, residuals)
    return ScenarioBundle(paths=eps.add(p_curve, axis=0), measure="Q", base_curve=p_curve, label="production")


def marginal_quantiles_from_paths(
    scenarios: ScenarioBundle,
    alphas: Sequence[float] = (0.05, 0.10, 0.50, 0.90, 0.95),
    *,
    enforce_non_crossing: bool = True,
) -> QuantileFan:
    if any(a <= 0.0 or a >= 1.0 for a in alphas):
        raise ValueError("alphas must be inside (0, 1)")
    values = scenarios.paths.quantile(list(alphas), axis=1).T
    values.columns = [f"q{int(round(a * 100)):02d}" for a in alphas]
    if enforce_non_crossing:
        arr = np.maximum.accumulate(values.to_numpy(dtype=float), axis=1)
        values = pd.DataFrame(arr, index=values.index, columns=values.columns)
    crossing = int((np.diff(values.to_numpy(dtype=float), axis=1) < -1e-12).sum())
    diagnostics = pd.DataFrame([{"measure": scenarios.measure, "quantile_crossings": crossing}])
    return QuantileFan(quantiles=values, diagnostics=diagnostics)


def _tail_stats(paths: pd.DataFrame) -> dict[str, float]:
    values = paths.to_numpy(dtype=float)
    return {
        "share_lt_0_pct": float((values < 0.0).mean() * 100.0),
        "share_lt_minus_10_pct": float((values < -10.0).mean() * 100.0),
        "share_lt_minus_30_pct": float((values < -30.0).mean() * 100.0),
    }


def _max_negative_run(row: pd.Series) -> int:
    run = best = 0
    for value in row.to_numpy(dtype=float):
        if value < 0.0:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def reconcile_q_scenario_mean_to_quotes(
    scenarios: ScenarioBundle,
    constraint_matrix: np.ndarray,
    quote_targets: np.ndarray,
    *,
    block_matrices: Mapping[str, np.ndarray] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> ReconciliationReport:
    if scenarios.measure != "Q":
        raise ValueError("only Q-measure scenarios can be reconciled to quotes")
    a = np.asarray(constraint_matrix, dtype=float)
    q = np.asarray(quote_targets, dtype=float)
    if a.ndim != 2 or a.shape[1] != scenarios.paths.shape[0]:
        raise ValueError("constraint_matrix shape incompatible with scenario paths")
    mean = scenarios.paths.mean(axis=1).to_numpy(dtype=float)
    residual = q - a @ mean
    adjustment = a.T @ (np.linalg.pinv(a @ a.T) @ residual)
    adjusted = scenarios.paths.add(pd.Series(adjustment, index=scenarios.paths.index), axis=0)
    before_tail = _tail_stats(scenarios.paths)
    after_tail = _tail_stats(adjusted)
    constraint_residuals = pd.DataFrame(
        {
            "target": q,
            "achieved": a @ adjusted.mean(axis=1).to_numpy(dtype=float),
        }
    )
    constraint_residuals["abs_error"] = (constraint_residuals["achieved"] - constraint_residuals["target"]).abs()
    tail_rows = []
    verdict: Verdict = "PASS"
    max_tail_move_pp = float((thresholds or {}).get("max_tail_move_pp", 2.0))
    max_quote_abs_error = float((thresholds or {}).get("max_quote_abs_error", 1e-9))
    if float(constraint_residuals["abs_error"].max()) > max_quote_abs_error:
        verdict = "NO_GO"
    for key, before in before_tail.items():
        after = after_tail[key]
        move = abs(after - before)
        if verdict != "NO_GO" and move > max_tail_move_pp:
            verdict = "CONDITIONAL"
        tail_rows.append({"metric": key, "before": before, "after": after, "move": move})
    runs_before = scenarios.paths.apply(_max_negative_run, axis=0)
    runs_after = adjusted.apply(_max_negative_run, axis=0)
    tail_rows.append(
        {
            "metric": "max_negative_run",
            "before": float(runs_before.max()),
            "after": float(runs_after.max()),
            "move": float(abs(runs_after.max() - runs_before.max())),
        }
    )
    block_rows = []
    if block_matrices:
        for name, matrix in block_matrices.items():
            m = np.asarray(matrix, dtype=float)
            before = scenarios.paths.to_numpy(dtype=float).T @ m.T
            after = adjusted.to_numpy(dtype=float).T @ m.T
            block_rows.append(
                {
                    "block": name,
                    "before_p05": float(np.quantile(before, 0.05)),
                    "after_p05": float(np.quantile(after, 0.05)),
                }
            )
    reconciled = ScenarioBundle(adjusted, measure="Q", base_curve=adjusted.mean(axis=1), label=scenarios.label)
    return ReconciliationReport(
        reconciled=reconciled,
        constraint_residuals=constraint_residuals,
        tail_diagnostics=pd.DataFrame(tail_rows),
        block_vwap_diagnostics=pd.DataFrame(block_rows),
        verdict=verdict,
    )
