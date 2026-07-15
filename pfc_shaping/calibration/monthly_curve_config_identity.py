"""Deterministic identity contract for governed monthly-curve configurations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np
import pandas as pd

CONFIG_IDENTITY_KEYS = (
    "lambda_prior",
    "lambda_smooth_month",
    "lambda_smooth_yoy",
    "lambda_shape",
    "neighbor_shrinkage",
    "robust_panel_quantile",
    "min_history_snapshots",
    "max_prior_residual_eur_mwh",
    "constraint_tolerance",
    "stationarity_tolerance",
    "markets",
    "min_structural_snapshots",
    "allow_template_structural_fallback",
    "structural_amplitude_eur_mwh",
    "panel_weight",
    "history_weight",
    "structural_weight",
)
MAX_SOLVER_CONDITION_NUMBER = 1e12


def solver_numerical_diagnostic_errors(
    kkt: Mapping[str, object] | object,
    *,
    solver_config: Mapping[str, object],
) -> list[str]:
    """Return fail-closed numerical errors for one governed monthly solve."""

    if not isinstance(kkt, Mapping):
        return ["solver_kkt_missing"]
    required_numeric = (
        "max_abs_constraint_residual",
        "stationarity_residual",
        "condition_number",
    )
    values: dict[str, float] = {}
    errors: list[str] = []
    for key in required_numeric:
        raw = kkt.get(key)
        if isinstance(raw, bool):
            errors.append(f"solver_kkt_{key}_invalid")
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"solver_kkt_{key}_invalid")
            continue
        if not np.isfinite(value):
            errors.append(f"solver_kkt_{key}_nonfinite")
            continue
        values[key] = value
    for key in ("ridge_used", "solved_by_lstsq"):
        if type(kkt.get(key)) is not bool:
            errors.append(f"solver_kkt_{key}_invalid")
        elif kkt.get(key) is True:
            errors.append(f"solver_kkt_{key}")
    constraint_tolerance = float(solver_config.get("constraint_tolerance", 1e-9))
    stationarity_tolerance = float(solver_config.get("stationarity_tolerance", 1e-7))
    if values.get("max_abs_constraint_residual", float("inf")) > constraint_tolerance:
        errors.append("solver_kkt_constraint_tolerance_exceeded")
    if values.get("stationarity_residual", float("inf")) > stationarity_tolerance:
        errors.append("solver_kkt_stationarity_tolerance_exceeded")
    condition = values.get("condition_number", float("inf"))
    if condition <= 0.0 or condition > MAX_SOLVER_CONDITION_NUMBER:
        errors.append("solver_kkt_condition_number_out_of_bounds")
    return errors


def config_hash(config: object | Mapping[str, object]) -> str:
    payload = config_payload(config)
    first = _sha256_text(canonical_json(payload))
    second = _sha256_text(canonical_json(dict(reversed(list(payload.items())))))
    if first != second:
        raise ValueError("non-reproducible config hash")
    return first


def lambda_grid_hash(grid: Mapping[str, object]) -> str:
    return _sha256_text(canonical_json(grid))


def canonical_json(value: object) -> str:
    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def config_payload(config: object | Mapping[str, object]) -> dict[str, object]:
    history_lookback_years: object = None
    monthly_config = getattr(config, "monthly_config", None)
    if monthly_config is not None and hasattr(config, "history_lookback_years"):
        history_lookback_years = getattr(config, "history_lookback_years")
        raw = dict(vars(monthly_config))
    elif isinstance(config, Mapping):
        raw = dict(config)
        history_lookback_years = raw.get("history_lookback_years")
    else:
        raw = dict(vars(config))
    payload = {key: raw.get(key) for key in CONFIG_IDENTITY_KEYS}
    payload["history_lookback_years"] = history_lookback_years
    return payload


def _canonicalize(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            return str(value)
        return float(f"{value:.12g}")
    if isinstance(value, np.floating):
        return _canonicalize(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, pd.Timestamp):
        timestamp = (
            value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        )
        return timestamp.isoformat()
    return value


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
