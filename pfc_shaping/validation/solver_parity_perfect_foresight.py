"""Research-only perfect-foresight diagnostic for the production monthly solver path."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from pfc_shaping.calibration.monthly_curve_config_identity import (
    solver_numerical_diagnostic_errors,
)
from pfc_shaping.data.lt_input_sources import dataframe_sha256
from pfc_shaping.pipeline.monthly_curve_authority import (
    latest_base_prices_by_market,
    solve_monthly_level_authority,
)
from pfc_shaping.pipeline.production_phases import _solver_delivery_quarter_hour_grid
from pfc_shaping.validation.scorecard import ABLATION_GRID, _fit_validation_components

TZ = "Europe/Zurich"
SCHEMA_VERSION = "solver_parity_perfect_foresight.v1"


@dataclass(frozen=True)
class SolverParityPerfectForesightResult:
    pfc: pd.DataFrame
    metrics: dict[str, float]
    metadata: dict[str, object]
    monthly_authority_manifest: dict[str, object]
    final_projection_report: dict[str, object]


def run_solver_parity_perfect_foresight(
    epex: pd.Series,
    eex_forwards_history: pd.DataFrame,
    *,
    target_year: int,
    vintage: pd.Timestamp,
    solver_settings: Mapping[str, object],
    config_name: str,
    estimator: str = "baseline",
) -> SolverParityPerfectForesightResult:
    """Score Cal delivery through the current solver/assembler authority path.

    This is deliberately non-promotional. Historical EEX rows currently expose
    a snapshot date but no authenticated row-level ``available_at``/revision
    lineage, so the result cannot establish production point-in-time evidence.
    """

    vintage = _aware_utc(vintage)
    if epex.index.tz is None:
        raise ValueError("epex must have a timezone-aware DatetimeIndex")
    if estimator not in {"baseline", "sota"}:
        raise ValueError("solver parity estimator must be 'baseline' or 'sota'")
    settings = dict(solver_settings)
    _validate_solver_settings(settings)
    config = _ablation_config(config_name)
    history_asof, cutoff = _history_as_of_vintage(eex_forwards_history, vintage)

    from pfc_shaping.validation.perfect_foresight import (
        _sota_estimator,
        monthly_signature,
        realized_window_mean,
    )

    realized_base = realized_window_mean(epex, target_year, segment="base")
    realized_peak = realized_window_mean(epex, target_year, segment="peak")
    if realized_base is None or realized_peak is None:
        raise ValueError(f"EPEX does not fully cover delivery Cal {target_year}")
    anchors = {
        str(target_year): float(realized_base),
        f"{target_year}-Peak": float(realized_peak),
    }
    delivery_months = pd.period_range(
        f"{target_year}-01",
        f"{target_year}-12",
        freq="M",
    )
    neighbor_prices: dict[str, dict[str, float]] = {}
    for market in settings.get("markets", ("DE", "FR", "AT", "IT")):
        code = str(market).upper()
        try:
            _, prices = latest_base_prices_by_market(history_asof, market=code)
        except ValueError:
            continue
        neighbor_prices[code] = prices

    source_hashes = {
        "eex_forward_history_full": dataframe_sha256(eex_forwards_history),
        "eex_forward_history_asof": dataframe_sha256(history_asof),
        "epex_full": dataframe_sha256(epex.rename("price_eur_mwh").to_frame()),
        "epex_training": dataframe_sha256(
            epex.loc[epex.index < vintage].rename("price_eur_mwh").to_frame()
        ),
        "perfect_foresight_anchors": _sha256_json(anchors),
    }
    authority = solve_monthly_level_authority(
        market="CH",
        delivery_months=delivery_months,
        own_base_prices=anchors,
        all_market_base_prices=neighbor_prices,
        eex_history=history_asof,
        run_timestamp=vintage,
        settings=settings,
        timezone=TZ,
        source_hashes=source_hashes,
        original_forward_prices=anchors,
        allow_unverified_inputs=True,
    )
    numerical_errors = solver_numerical_diagnostic_errors(
        authority.manifest.get("solver_kkt"),
        solver_config=settings,
    )
    if numerical_errors:
        raise RuntimeError(
            f"solver parity numerical diagnostics failed: {numerical_errors}"
        )

    estimator_context = _sota_estimator() if estimator == "sota" else nullcontext()
    with estimator_context:
        sh, si, calibrator, _ = _fit_validation_components(
            config,
            vintage,
            epex.rename("price_eur_mwh").to_frame(),
            with_uncertainty=False,
        )
        from pfc_shaping.lt.model.assembler import PFCAssembler

        assembler = PFCAssembler(
            shape_hourly=sh,
            shape_intraday=si,
            uncertainty=None,
            cascader=None,
            calibrator=calibrator,
            enforce_positivity=config.enforce_positivity,
            enforce_m_factor_floor=config.enforce_m_factor_floor,
            enforce_floor=config.enforce_floor,
            allow_negative_peak=config.allow_negative_peak,
            monthly_level_authority="solver",
            skip_legacy_level_cascade=True,
            skip_legacy_base_smoothing=True,
            monthly_constraint_tolerance=float(settings["constraint_tolerance"]),
        )
        delivery_index = _solver_delivery_quarter_hour_grid(
            authority.inputs.delivery_grid.months,
            timezone=TZ,
        )
        pfc = assembler.build(
            base_prices=authority.assembler_base_prices,
            quoted_keys=authority.quoted_keys,
            delivery_index=delivery_index,
            reference_date=vintage,
            country="CH",
        )

    projection = dict(getattr(assembler, "final_product_projection_report_", {}) or {})
    if not projection:
        raise RuntimeError("solver parity build did not emit final projection evidence")
    hourly = pfc["price_shape"].resample("1h").mean()
    model_signature = monthly_signature(hourly, target_year)
    realized_signature = monthly_signature(epex, target_year)
    paired = pd.concat(
        [model_signature.rename("model"), realized_signature.rename("realized")],
        axis=1,
        join="inner",
    ).dropna()
    if len(paired) != 12:
        raise RuntimeError("solver parity diagnostic must score all 12 delivery months")
    error = paired["model"] - paired["realized"]
    local_index = pfc.index.tz_convert(TZ)
    peak_mask = assembler._is_peak_timestamp(local_index, country="CH")
    delivered_base = float(pfc["price_shape"].mean())
    delivered_peak = float(pfc.loc[peak_mask, "price_shape"].mean())
    delivered_offpeak = float(pfc.loc[~peak_mask, "price_shape"].mean())
    peak_count = int(np.sum(peak_mask))
    offpeak_count = int(np.sum(~peak_mask))
    implied_offpeak = (
        float(realized_base) * len(pfc) - float(realized_peak) * peak_count
    ) / offpeak_count
    metrics = {
        "monthly_pearson_corr": float(paired["model"].corr(paired["realized"])),
        "monthly_spearman_corr": float(
            paired["model"].corr(paired["realized"], method="spearman")
        ),
        "monthly_mae_eur_mwh": float(error.abs().mean()),
        "monthly_bias_eur_mwh": float(error.mean()),
        "monthly_rmse_eur_mwh": float(np.sqrt(np.mean(np.square(error)))),
        "projection_max_abs_error_eur_mwh": float(
            projection["max_abs_error_eur_mwh"]
        ),
        "cal_base_residual_eur_mwh": delivered_base - float(realized_base),
        "cal_peak_residual_eur_mwh": delivered_peak - float(realized_peak),
        "cal_implied_offpeak_residual_eur_mwh": delivered_offpeak - implied_offpeak,
    }
    metadata: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "target_year": int(target_year),
        "vintage_utc": vintage.isoformat(),
        "config_name": config_name,
        "estimator": estimator,
        "monthly_level_authority": "solver",
        "skip_legacy_level_cascade": True,
        "skip_legacy_base_smoothing": True,
        "legacy_calibration_applied": False,
        "final_projection_count": 1,
        "production_representative": False,
        "promotion_eligible": False,
        "diagnostic_class": "RESEARCH_ONLY_NON_PIT_CONSERVATION_CHECK",
        "monthly_metrics_are_model_comparison_evidence": False,
        "solver_numerical_status": "PASS",
        "pit_evidence_status": "UNAVAILABLE_ROW_LEVEL_AVAILABLE_AT",
        "pit_cutoff_date": str(cutoff.date()),
        "history_rows_full": int(len(eex_forwards_history)),
        "history_rows_asof": int(len(history_asof)),
        "post_vintage_rows_excluded": int(len(eex_forwards_history) - len(history_asof)),
        "history_max_date_asof": str(pd.Timestamp(history_asof["date"].max()).date()),
        "ompex_used": False,
        "source_hashes": source_hashes,
        "solver_config_hash": str(authority.manifest["solver_config_hash"]),
        "active_config_hash": str(authority.manifest["active_config_hash"]),
        "monthly_solution_hash": authority.monthly_solution_hash,
        "active_constraints_hash": authority.active_constraints_hash,
        "final_projection_hash": _sha256_json(projection),
        "pfc_price_shape_hash": dataframe_sha256(pfc[["price_shape"]]),
        "anchors": anchors,
    }
    return SolverParityPerfectForesightResult(
        pfc=pfc,
        metrics=metrics,
        metadata=metadata,
        monthly_authority_manifest=dict(authority.manifest),
        final_projection_report=projection,
    )


def _history_as_of_vintage(
    history: pd.DataFrame,
    vintage: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    required = {"date", "product", "load_type", "price", "market"}
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"EEX forward history is missing columns: {missing}")
    out = history.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.tz_convert(None).dt.normalize()
    cutoff = vintage.tz_convert(TZ).tz_localize(None).normalize()
    out = out.loc[out["date"] <= cutoff].copy()
    if out.empty:
        raise ValueError(f"no EEX forward history is available by vintage {vintage}")
    return out.sort_values(["date", "market", "load_type", "product"]).reset_index(drop=True), cutoff


def _validate_solver_settings(settings: Mapping[str, object]) -> None:
    expected = {
        "enabled": True,
        "monthly_level_authority": "solver",
        "skip_legacy_level_cascade": True,
        "skip_legacy_base_smoothing": True,
    }
    mismatches = {
        key: {"expected": value, "actual": settings.get(key)}
        for key, value in expected.items()
        if settings.get(key) != value
    }
    if mismatches:
        raise ValueError(f"solver parity requires production solver settings: {mismatches}")
    if "constraint_tolerance" not in settings:
        raise ValueError("solver parity requires constraint_tolerance")


def _ablation_config(config_name: str):
    try:
        return next(config for config in ABLATION_GRID if config.name == config_name)
    except StopIteration as exc:
        raise ValueError(f"unknown ablation config: {config_name}") from exc


def _aware_utc(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("vintage must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _sha256_json(payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


__all__ = [
    "SCHEMA_VERSION",
    "SolverParityPerfectForesightResult",
    "run_solver_parity_perfect_foresight",
]
