"""Build the current CH monthly BASE research solution from selected D212 bytes.

The output is a local research layer, not PIT or model evidence. BASE CAL/Q/M
quotes alone define hard monthly levels. PEAK quotes are preserved in a
separate sidecar for a later hourly-shaping gate.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.calibration.monthly_curve_priors import (
    build_fused_shape_prior,
    build_history_shape_prior,
    build_neighbor_panel_shape_prior,
    build_structural_monthly_shape_prior_from_history,
)
from pfc_shaping.calibration.monthly_forward_curve import (
    MarketQuote,
    MonthlyCurveConfig,
    build_monthly_constraint_system,
    product_periods,
    solve_monthly_forward_curve_from_constraints,
)
from pfc_shaping.path_safety import read_stable_single_link_file

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / ".planning" / "phases" / "14-lt-audit-remediation"
CONTRACT_PATH = PHASE / "EEX-CH-CURRENT-MONTHLY-RESEARCH-SOLUTION-CONTRACT-V1.json"

NORMALIZATION_CONTENT_ID = "2837dc4849dc4b573c441059574973e0b8cc0fbb5023203509cb2929dd636a3f"
NORMALIZATION_MANIFEST_SHA256 = "f91805f3004e746ac588e19aa6745ae0dc0f490a9ef5c49aae0918e7ec3f8f53"
SURFACE_AUDIT_CONTENT_ID = "bb1a09932b4bbff31dfdbb4ada561befb02050413ee819b03bf6c28f4858ab54"
SURFACE_AUDIT_MANIFEST_SHA256 = "7f5d565ad7191b5f455e7a29280de078422cd95b3a846d06dc8cc6f29bf131a0"
HORIZON_AUDIT_CONTENT_ID = "435ecbc737f95268f03f7f347dfafc4163f5b5a4cb5b8dc9cec87e05f1645108"
HORIZON_AUDIT_MANIFEST_SHA256 = "949a1598710ad6198b569fd4cb00a99750707e162dbc0c709fefaa04f28e0ed1"

NORMALIZATION_ROOT = (
    ROOT
    / "build"
    / "databricks-eex-daily"
    / "2026-08-05"
    / "normalizations-all-products"
    / NORMALIZATION_CONTENT_ID
)
NORMALIZATION_MANIFEST_PATH = NORMALIZATION_ROOT / "manifest.json"
HISTORY_PATH = NORMALIZATION_ROOT / "eex_ch_cal_q_m_live_history.parquet"
LATEST_PATH = NORMALIZATION_ROOT / "eex_ch_cal_q_m_latest.parquet"
SURFACE_MANIFEST_PATH = (
    ROOT
    / "build"
    / "databricks-eex-daily"
    / "2026-08-05"
    / "surface-audits-live-cqm"
    / SURFACE_AUDIT_CONTENT_ID
    / "manifest.json"
)
HORIZON_MANIFEST_PATH = (
    ROOT
    / "build"
    / "databricks-eex-daily"
    / "2026-08-05"
    / "horizon-audits-live-cqm"
    / HORIZON_AUDIT_CONTENT_ID
    / "manifest.json"
)

CONTRACT_SCHEMA = "fmv_eex_ch_current_monthly_research_solution_contract.v1"
CONTRACT_STATUS = "LOCAL_CURRENT_EEX_MONTHLY_RESEARCH_SOLUTION_NO_PIT_OR_MODEL_AUTHORITY"
CONTRACT_FILE_SHA256 = "3faa859d50b136d06df29afe02e635662c1641b7321f1bb105dddc3cf95c691a"
CONTRACT_CONTENT_ID = "e93e0c31903c93ad6ba382f9f34b05f5842c6f17d7ea0f8fe4e22eaf9eee2ca3"
ASSESSMENT_SCHEMA = "fmv_eex_ch_current_monthly_research_solution_assessment.v1"
PASS_STATUS = "PASS_LOCAL_CURRENT_EEX_MONTHLY_RESEARCH_SOLUTION_NO_PIT_OR_MODEL_AUTHORITY"

HISTORY_SHA256 = "fc5de85d1870937955dcc93cbf1cea0e1d2f85d892b286f490a6de11650d7a25"
LATEST_SHA256 = "3b8c9a9831a3ff44b8b9880e914fa1bb1e60c4e03d0610d1fd40ab8b83490aaf"
SOURCE_SNAPSHOT_SHA256 = "593e916b6aa18ad83f7bd7941ff68184cd71da8882ef4eb381de46d09ce64812"
HISTORY_ROWS = 34_105
LATEST_ROWS = 38
BASE_QUOTES = 19
PEAK_QUOTES = 19
SURFACE_DATE = pd.Timestamp("2026-08-04")
RUN_TIMESTAMP = pd.Timestamp("2026-08-04T00:00:00Z")
FIRST_MONTH = pd.Period("2026-09", freq="M")
LAST_MONTH = pd.Period("2032-12", freq="M")
DELIVERY_MONTHS = pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M")

CONFIG = MonthlyCurveConfig(
    lambda_prior=0.000001,
    lambda_smooth_month=1.0,
    lambda_smooth_yoy=0.25,
    lambda_shape=1.0,
    neighbor_shrinkage=0.5,
    min_history_snapshots=24,
    constraint_tolerance=1e-9,
    quote_conflict_tolerance=0.01,
    stationarity_tolerance=1e-7,
)
PRIOR_POLICY = {
    "minimum_snapshots": 24,
    "lookback_years": 6,
    "minimum_structural_snapshots": 24,
    "template_fallback_allowed": False,
    "panel_weight": 0.0,
    "history_weight": 0.5,
    "structural_weight": 1.0,
}
AUTHORITIES = {
    "selected_local_eex_quote_bytes_bound": True,
    "monthly_solver_mechanical_repricing": True,
    "local_research_solution": True,
    "source_authenticity": False,
    "point_in_time": False,
    "predictive_value": False,
    "model_input": False,
    "model_selection": False,
    "candidate": False,
    "ompex_superiority": False,
    "promotion": False,
    "production": False,
}
EXECUTION = {
    "databricks_connection_count": 0,
    "databricks_statement_count": 0,
    "warehouse_start_count": 0,
    "databricks_write_count": 0,
    "network_call_count": 0,
    "h_drive_access_count": 0,
    "remote_write_count": 0,
}

FRAME_COLUMNS = (
    "date",
    "product",
    "load_type",
    "product_type",
    "market",
    "price",
    "unit",
    "source",
    "source_system",
    "product_id",
    "delivery_period_id",
    "fact_load_timestamp_utc",
    "source_snapshot_sha256",
    "point_in_time_available_at_proven",
    "rolling_origin_selection_authorized",
    "production_promotion_authorized",
)
FRAME_DTYPES = (
    "datetime64[ns]",
    "string",
    "string",
    "string",
    "string",
    "float64",
    "string",
    "string",
    "string",
    "string",
    "string",
    "datetime64[ns, UTC]",
    "string",
    "boolean",
    "boolean",
    "boolean",
)
_SHA = re.compile(r"^[0-9a-f]{64}$")


class EexCurrentMonthlyResearchSolutionError(ValueError):
    """Raised when selected EEX evidence or monthly mechanics drift."""


@dataclass(frozen=True)
class CurrentMonthlyResearchSolution:
    """Price-free assessment plus separate local price-bearing output frames."""

    assessment: dict[str, object]
    assessment_content_id: str
    monthly_solution: pd.DataFrame
    peak_sidecar: pd.DataFrame
    constraint_diagnostics: pd.DataFrame
    prior_diagnostics: pd.DataFrame


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_current_monthly_solution_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    contract = _mapping(document, "D286 contract")
    _exact(
        contract,
        {
            "schema_version",
            "decision",
            "status",
            "purpose",
            "predecessors",
            "source_artifacts",
            "delivery_grid",
            "solver_policy",
            "shape_prior_policy",
            "outputs",
            "execution",
            "authorities",
        },
        "D286 contract",
    )
    _equal(contract["schema_version"], CONTRACT_SCHEMA, "contract schema")
    _equal(contract["decision"], "D-20260806-286", "contract decision")
    _equal(contract["status"], CONTRACT_STATUS, "contract status")
    _clean(contract["purpose"], "contract purpose")
    _equal(
        contract["predecessors"],
        {
            "d212_normalization_content_id": NORMALIZATION_CONTENT_ID,
            "d212_normalization_manifest_sha256": NORMALIZATION_MANIFEST_SHA256,
            "d212_surface_audit_content_id": SURFACE_AUDIT_CONTENT_ID,
            "d212_surface_audit_manifest_sha256": SURFACE_AUDIT_MANIFEST_SHA256,
            "d213_horizon_audit_content_id": HORIZON_AUDIT_CONTENT_ID,
            "d213_horizon_audit_manifest_sha256": HORIZON_AUDIT_MANIFEST_SHA256,
        },
        "predecessors",
    )
    _equal(
        contract["source_artifacts"],
        {
            "normalization_manifest_relative_path": _relative(NORMALIZATION_MANIFEST_PATH),
            "solver_history_relative_path": _relative(HISTORY_PATH),
            "solver_history_sha256": HISTORY_SHA256,
            "solver_history_row_count": HISTORY_ROWS,
            "solver_latest_relative_path": _relative(LATEST_PATH),
            "solver_latest_sha256": LATEST_SHA256,
            "solver_latest_row_count": LATEST_ROWS,
            "surface_date": "2026-08-04",
            "market": "CH",
            "unit": "EUR/MWh",
            "base_quote_count": BASE_QUOTES,
            "peak_quote_count": PEAK_QUOTES,
        },
        "source artifacts",
    )
    _equal(
        contract["delivery_grid"],
        {
            "first_month": "2026-09",
            "last_month": "2032-12",
            "month_count": 76,
            "timezone": "Europe/Zurich",
            "calendar": "CH",
        },
        "delivery grid",
    )
    _equal(
        contract["solver_policy"],
        {
            "monthly_level_authority": "ONE_CH_BASE_SOLVER",
            "hard_constraint_products": ["MONTH", "QUARTER", "YEAR"],
            "hard_constraint_load_type": "BASE",
            "peak_policy": "SEPARATE_SHAPING_SIDECAR_NO_MONTHLY_LEVEL_AUTHORITY",
            "short_tenor_policy": "DAY_WEEK_WEEKEND_EXCLUDED_FROM_MONTHLY_LEVEL",
            "neighbor_level_policy": "FORBIDDEN",
            "entsoe_level_policy": "FORBIDDEN",
            "afry_level_policy": "FORBIDDEN",
            "ompex_level_policy": "FORBIDDEN",
            "lambda_prior": CONFIG.lambda_prior,
            "lambda_smooth_month": CONFIG.lambda_smooth_month,
            "lambda_smooth_yoy": CONFIG.lambda_smooth_yoy,
            "lambda_shape": CONFIG.lambda_shape,
            "constraint_tolerance_eur_mwh": CONFIG.constraint_tolerance,
            "quote_conflict_tolerance_eur_mwh": CONFIG.quote_conflict_tolerance,
            "stationarity_tolerance": CONFIG.stationarity_tolerance,
        },
        "solver policy",
    )
    _equal(
        contract["shape_prior_policy"],
        {
            "history_source": "D212_CH_BASE_CAL_Q_M_LIVE_HISTORY",
            "run_timestamp": "2026-08-04T00:00:00Z",
            **PRIOR_POLICY,
            "prior_must_be_zero_mean_within_each_active_parent": True,
            "prior_may_rewrite_parent_level": False,
        },
        "shape prior policy",
    )
    _equal(
        contract["outputs"],
        {
            "monthly_solution_role": "monthly_base_solution",
            "peak_sidecar_role": "peak_shaping_constraints",
            "constraint_diagnostics_role": "base_constraint_diagnostics",
            "prior_diagnostics_role": "shape_prior_diagnostics",
            "price_bearing_outputs_must_remain_below_build": True,
            "proof_may_persist_prices": False,
            "outputs_are_content_addressed": True,
        },
        "outputs",
    )
    _equal(contract["execution"], EXECUTION, "execution")
    _equal(contract["authorities"], AUTHORITIES, "authorities")
    _equal(canonical_sha256(contract), CONTRACT_CONTENT_ID, "contract content ID")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": CONTRACT_CONTENT_ID,
        "delivery_month_count": len(DELIVERY_MONTHS),
        "base_only_monthly_level": True,
        "peak_is_sidecar_only": True,
        "point_in_time_authorized": False,
        "model_input_authorized": False,
        **EXECUTION,
    }


def validate_current_monthly_solution_contract_path(
    path: str | Path,
) -> dict[str, object]:
    contract_path = Path(path)
    if not contract_path.is_absolute():
        raise EexCurrentMonthlyResearchSolutionError("contract path must be absolute")
    payload = _read(contract_path, "D286 contract", 200_000)
    if hashlib.sha256(payload).hexdigest() != CONTRACT_FILE_SHA256:
        raise EexCurrentMonthlyResearchSolutionError("contract SHA-256 differs")
    return validate_current_monthly_solution_contract(_strict_json(payload, "D286 contract"))


def build_selected_current_monthly_research_solution(
    *,
    contract_path: str | Path = CONTRACT_PATH,
    normalization_manifest_path: str | Path = NORMALIZATION_MANIFEST_PATH,
    history_path: str | Path = HISTORY_PATH,
    latest_path: str | Path = LATEST_PATH,
    surface_manifest_path: str | Path = SURFACE_MANIFEST_PATH,
    horizon_manifest_path: str | Path = HORIZON_MANIFEST_PATH,
    reference_time_utc: datetime,
) -> CurrentMonthlyResearchSolution:
    """Load the exact selected local D212/D213 bytes and solve D286."""

    validate_current_monthly_solution_contract_path(contract_path)
    reference = _utc_datetime(reference_time_utc, "reference time")
    if reference < RUN_TIMESTAMP.to_pydatetime():
        raise EexCurrentMonthlyResearchSolutionError("reference predates selected surface")

    normalization_raw = _read(
        Path(normalization_manifest_path),
        "D212 normalization manifest",
        2_000_000,
    )
    surface_raw = _read(Path(surface_manifest_path), "D212 surface manifest", 2_000_000)
    horizon_raw = _read(Path(horizon_manifest_path), "D213 horizon manifest", 2_000_000)
    normalization = _bound_manifest(
        normalization_raw,
        expected_sha256=NORMALIZATION_MANIFEST_SHA256,
        expected_content_id=NORMALIZATION_CONTENT_ID,
        label="D212 normalization manifest",
    )
    _validate_normalization_manifest(normalization)
    surface = _bound_manifest(
        surface_raw,
        expected_sha256=SURFACE_AUDIT_MANIFEST_SHA256,
        expected_content_id=SURFACE_AUDIT_CONTENT_ID,
        label="D212 surface manifest",
    )
    horizon = _bound_manifest(
        horizon_raw,
        expected_sha256=HORIZON_AUDIT_MANIFEST_SHA256,
        expected_content_id=HORIZON_AUDIT_CONTENT_ID,
        label="D213 horizon manifest",
    )
    _validate_predecessor_bindings(surface, horizon)

    history_raw = _read(Path(history_path), "D212 solver history", 10_000_000)
    latest_raw = _read(Path(latest_path), "D212 latest solver surface", 2_000_000)
    if hashlib.sha256(history_raw).hexdigest() != HISTORY_SHA256:
        raise EexCurrentMonthlyResearchSolutionError("D212 history SHA-256 differs")
    if hashlib.sha256(latest_raw).hexdigest() != LATEST_SHA256:
        raise EexCurrentMonthlyResearchSolutionError("D212 latest SHA-256 differs")
    try:
        history = pd.read_parquet(BytesIO(history_raw))
        latest = pd.read_parquet(BytesIO(latest_raw))
    except Exception as exc:
        raise EexCurrentMonthlyResearchSolutionError(
            "selected D212 Parquet cannot be read"
        ) from exc
    result = solve_current_monthly_research_frames(
        history=history,
        latest=latest,
        reference_time_utc=reference,
        evidence_class="SELECTED_LOCAL_D212_REAL_QUOTE_BYTES",
    )
    for path, expected, label, maximum in (
        (
            Path(normalization_manifest_path),
            normalization_raw,
            "D212 normalization manifest",
            2_000_000,
        ),
        (Path(surface_manifest_path), surface_raw, "D212 surface manifest", 2_000_000),
        (Path(horizon_manifest_path), horizon_raw, "D213 horizon manifest", 2_000_000),
        (Path(history_path), history_raw, "D212 solver history", 10_000_000),
        (Path(latest_path), latest_raw, "D212 latest solver surface", 2_000_000),
    ):
        if _read(path, label, maximum) != expected:
            raise EexCurrentMonthlyResearchSolutionError(f"{label} changed during D286")
    return result


def solve_current_monthly_research_frames(
    *,
    history: pd.DataFrame,
    latest: pd.DataFrame,
    reference_time_utc: datetime,
    evidence_class: str = "SYNTHETIC_TEST_ONLY",
) -> CurrentMonthlyResearchSolution:
    """Solve validated frames; public for mutation tests, never an authority bypass."""

    reference = _utc_datetime(reference_time_utc, "reference time")
    evidence = _clean(evidence_class, "evidence class")
    _validate_frame(
        history, "history", exact_rows=HISTORY_ROWS if evidence.startswith("SELECTED_") else None
    )
    _validate_frame(
        latest, "latest", exact_rows=LATEST_ROWS if evidence.startswith("SELECTED_") else None
    )
    _validate_frame_relationship(history, latest)
    if reference < RUN_TIMESTAMP.to_pydatetime():
        raise EexCurrentMonthlyResearchSolutionError("reference predates selected surface")

    base = latest[latest["load_type"].astype(str).eq("BASE")].copy()
    peak = latest[latest["load_type"].astype(str).eq("PEAK")].copy()
    if len(base) != BASE_QUOTES or len(peak) != PEAK_QUOTES:
        raise EexCurrentMonthlyResearchSolutionError("BASE/PEAK quote counts differ")
    if set(base["product"].astype(str)) != set(peak["product"].astype(str)):
        raise EexCurrentMonthlyResearchSolutionError("BASE/PEAK product identities differ")

    base_quotes = tuple(_market_quote(row) for _, row in base.sort_values("product").iterrows())
    constraints = build_monthly_constraint_system(
        DELIVERY_MONTHS,
        base_quotes,
        timezone="Europe/Zurich",
        market="CH",
        load_type="BASE",
        constraint_tolerance=CONFIG.constraint_tolerance,
        quote_conflict_tolerance=CONFIG.quote_conflict_tolerance,
    )
    panel = build_neighbor_panel_shape_prior(
        constraints,
        {},
        neighbor_markets=(),
        run_timestamp=SURFACE_DATE,
    )
    historical = build_history_shape_prior(
        constraints,
        history,
        market="CH",
        load_type="BASE",
        run_timestamp=SURFACE_DATE,
        min_snapshots=PRIOR_POLICY["minimum_snapshots"],
        lookback_years=PRIOR_POLICY["lookback_years"],
    )
    structural = build_structural_monthly_shape_prior_from_history(
        constraints,
        history,
        market="CH",
        load_type="BASE",
        run_timestamp=SURFACE_DATE,
        min_snapshots=PRIOR_POLICY["minimum_structural_snapshots"],
        lookback_years=PRIOR_POLICY["lookback_years"],
        fallback_to_template=False,
    )
    fused = build_fused_shape_prior(
        constraints,
        panel_prior=panel,
        history_prior=historical,
        structural_prior=structural,
        weights={
            "panel": PRIOR_POLICY["panel_weight"],
            "history": PRIOR_POLICY["history_weight"],
            "structural": PRIOR_POLICY["structural_weight"],
        },
    )
    result = solve_monthly_forward_curve_from_constraints(
        constraints,
        config=CONFIG,
        shape_prior=fused,
    )
    active_max = float(result.residuals["abs_error"].max())
    if active_max > CONFIG.constraint_tolerance:
        raise EexCurrentMonthlyResearchSolutionError("active hard repricing tolerance exceeded")
    if panel.status != "UNSUPPORTED":
        raise EexCurrentMonthlyResearchSolutionError("neighbor panel unexpectedly entered D286")
    if structural.status == "STRUCTURAL_TEMPLATE":
        raise EexCurrentMonthlyResearchSolutionError("template prior unexpectedly entered D286")
    if fused.status == "UNSUPPORTED":
        raise EexCurrentMonthlyResearchSolutionError("D212 historical shape prior is unsupported")

    prior_diagnostics = _prior_parent_diagnostics(constraints, fused.shape)
    prior_max = float(prior_diagnostics["abs_weighted_mean_eur_mwh"].max())
    if prior_max > 1e-10:
        raise EexCurrentMonthlyResearchSolutionError("shape prior rewrites a hard parent level")
    quote_diagnostics = _all_base_quote_diagnostics(
        base,
        constraints.delivery_grid.month_hours,
        result.monthly_curve,
        constraints.quote_diagnostics,
    )
    all_quote_max = float(quote_diagnostics["abs_residual_eur_mwh"].max())
    if all_quote_max > CONFIG.quote_conflict_tolerance:
        raise EexCurrentMonthlyResearchSolutionError("displayed BASE quote tolerance exceeded")

    monthly_solution = _monthly_solution_frame(
        constraints,
        result.monthly_curve,
        fused.shape,
        base,
    )
    peak_sidecar = _peak_sidecar_frame(peak)
    if len(monthly_solution) != len(DELIVERY_MONTHS) or monthly_solution.isna().any().any():
        raise EexCurrentMonthlyResearchSolutionError("monthly solution is incomplete")
    if not np.isfinite(monthly_solution["base_price_eur_mwh"]).all():
        raise EexCurrentMonthlyResearchSolutionError("monthly solution is non-finite")

    outputs = {
        "monthly_solution_content_id": _frame_content_id(monthly_solution),
        "peak_sidecar_content_id": _frame_content_id(peak_sidecar),
        "constraint_diagnostics_content_id": _frame_content_id(quote_diagnostics),
        "prior_diagnostics_content_id": _frame_content_id(prior_diagnostics),
    }
    redundant_count = int((quote_diagnostics["constraint_role"] == "REDUNDANT_CONSISTENCY").sum())
    assessment = {
        "schema_version": ASSESSMENT_SCHEMA,
        "status": PASS_STATUS,
        "contract_content_id": CONTRACT_CONTENT_ID,
        "reference_time_utc": _utc_text(reference),
        "evidence_class": evidence,
        "normalization_content_id": NORMALIZATION_CONTENT_ID,
        "surface_audit_content_id": SURFACE_AUDIT_CONTENT_ID,
        "horizon_audit_content_id": HORIZON_AUDIT_CONTENT_ID,
        "source_history_sha256": HISTORY_SHA256,
        "source_latest_sha256": LATEST_SHA256,
        "surface_date": "2026-08-04",
        "first_wholly_undelivered_month": str(FIRST_MONTH),
        "last_delivery_month": str(LAST_MONTH),
        "monthly_solution_row_count": len(monthly_solution),
        "base_quote_count": len(base),
        "peak_sidecar_quote_count": len(peak_sidecar),
        "active_hard_constraint_count": len(result.residuals),
        "redundant_consistency_quote_count": redundant_count,
        "active_hard_max_abs_residual_eur_mwh": active_max,
        "all_displayed_base_max_abs_residual_eur_mwh": all_quote_max,
        "shape_prior_parent_max_abs_mean_eur_mwh": prior_max,
        "history_prior_status": historical.status,
        "structural_prior_status": structural.status,
        "fused_prior_status": fused.status,
        "panel_prior_status": panel.status,
        "base_is_sole_monthly_level_authority": True,
        "peak_is_sidecar_only": True,
        "short_tenor_used_as_monthly_level": False,
        "neighbor_entsoe_afry_or_ompex_level_used": False,
        "price_values_persisted_in_assessment": False,
        "point_in_time_authorized": False,
        "model_input_authorized": False,
        "ompex_superiority_authorized": False,
        "output_content_ids": outputs,
        "authorities": dict(AUTHORITIES),
        **EXECUTION,
    }
    return CurrentMonthlyResearchSolution(
        assessment=assessment,
        assessment_content_id=canonical_sha256(assessment),
        monthly_solution=monthly_solution,
        peak_sidecar=peak_sidecar,
        constraint_diagnostics=quote_diagnostics,
        prior_diagnostics=prior_diagnostics,
    )


def _validate_frame(frame: pd.DataFrame, label: str, exact_rows: int | None) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise EexCurrentMonthlyResearchSolutionError(f"{label} must be a DataFrame")
    if tuple(frame.columns) != FRAME_COLUMNS:
        raise EexCurrentMonthlyResearchSolutionError(f"{label} columns differ")
    if tuple(str(frame[column].dtype) for column in frame.columns) != FRAME_DTYPES:
        raise EexCurrentMonthlyResearchSolutionError(f"{label} dtypes differ")
    if exact_rows is not None and len(frame) != exact_rows:
        raise EexCurrentMonthlyResearchSolutionError(f"{label} row count differs")
    if frame.empty or frame.isna().any().any():
        raise EexCurrentMonthlyResearchSolutionError(f"{label} is empty or contains nulls")
    if not np.isfinite(frame["price"].to_numpy(dtype=float)).all():
        raise EexCurrentMonthlyResearchSolutionError(f"{label} contains non-finite prices")
    if set(frame["market"].astype(str)) != {"CH"}:
        raise EexCurrentMonthlyResearchSolutionError(f"{label} market differs")
    if {value.upper() for value in frame["unit"].astype(str)} != {"EUR/MWH"}:
        raise EexCurrentMonthlyResearchSolutionError(f"{label} unit differs")
    if set(frame["source"].astype(str)) != {"EEX"}:
        raise EexCurrentMonthlyResearchSolutionError(f"{label} source differs")
    if set(frame["source_system"].astype(str)) != {"DATABRICKS_PRD_GOLD"}:
        raise EexCurrentMonthlyResearchSolutionError(f"{label} source system differs")
    if set(frame["source_snapshot_sha256"].astype(str)) != {SOURCE_SNAPSHOT_SHA256}:
        raise EexCurrentMonthlyResearchSolutionError(f"{label} source snapshot differs")
    if set(frame["load_type"].astype(str)) - {"BASE", "PEAK"}:
        raise EexCurrentMonthlyResearchSolutionError(f"{label} load type differs")
    if set(frame["product_type"].astype(str)) - {"Month", "Quarter", "Cal"}:
        raise EexCurrentMonthlyResearchSolutionError(f"{label} product type differs")
    for field in (
        "point_in_time_available_at_proven",
        "rolling_origin_selection_authorized",
        "production_promotion_authorized",
    ):
        if frame[field].astype(bool).any():
            raise EexCurrentMonthlyResearchSolutionError(f"{label} {field} escalates authority")
    key = ["date", "product", "load_type"]
    if frame.duplicated(key, keep=False).any():
        raise EexCurrentMonthlyResearchSolutionError(f"{label} repeats quote identity")


def _validate_frame_relationship(history: pd.DataFrame, latest: pd.DataFrame) -> None:
    if latest["date"].nunique() != 1 or pd.Timestamp(latest["date"].iloc[0]) != SURFACE_DATE:
        raise EexCurrentMonthlyResearchSolutionError("latest surface date differs")
    if pd.Timestamp(history["date"].max()) != SURFACE_DATE:
        raise EexCurrentMonthlyResearchSolutionError("history maximum date differs")
    if (pd.to_datetime(history["date"]) > SURFACE_DATE).any():
        raise EexCurrentMonthlyResearchSolutionError("history contains future quote dates")
    selected = history[pd.to_datetime(history["date"]).eq(SURFACE_DATE)].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            selected,
            latest.reset_index(drop=True),
            check_exact=True,
            check_freq=False,
        )
    except AssertionError as exc:
        raise EexCurrentMonthlyResearchSolutionError(
            "latest surface is not the exact terminal history slice"
        ) from exc


def _market_quote(row: pd.Series) -> MarketQuote:
    product = str(row["product"])
    quote_id = _row_quote_id(row)
    return MarketQuote(
        market="CH",
        product=product,
        load_type="BASE",
        price=float(row["price"]),
        snapshot_date=SURFACE_DATE,
        source=quote_id,
        available_at=None,
        quote_id=quote_id,
        snapshot_id=NORMALIZATION_CONTENT_ID,
        observation_id="",
        source_kind="EEX_DATABRICKS_LOCAL_RESEARCH",
        source_sha256=LATEST_SHA256,
    )


def _row_quote_id(row: pd.Series) -> str:
    return canonical_sha256(
        {
            "source_snapshot_sha256": str(row["source_snapshot_sha256"]),
            "quotation_date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            "product": str(row["product"]),
            "load_type": str(row["load_type"]),
            "product_id": str(row["product_id"]),
            "delivery_period_id": str(row["delivery_period_id"]),
            "price_eur_mwh": float(row["price"]),
        }
    )


def _monthly_solution_frame(
    constraints,
    monthly_curve: pd.Series,
    shape: pd.Series,
    base: pd.DataFrame,
) -> pd.DataFrame:
    direct_months = {
        str(product)
        for product, product_type in zip(base["product"], base["product_type"], strict=True)
        if str(product_type) == "Month"
    }
    return pd.DataFrame(
        {
            "delivery_month": [str(month) for month in DELIVERY_MONTHS],
            "base_price_eur_mwh": monthly_curve.reindex(DELIVERY_MONTHS).to_numpy(dtype=float),
            "delivery_hours": constraints.delivery_grid.month_hours.reindex(
                DELIVERY_MONTHS
            ).to_numpy(dtype=float),
            "parent_bucket": constraints.month_buckets.reindex(DELIVERY_MONTHS)
            .astype(str)
            .to_numpy(),
            "shape_prior_eur_mwh": shape.reindex(DELIVERY_MONTHS).to_numpy(dtype=float),
            "direct_month_quote": [str(month) in direct_months for month in DELIVERY_MONTHS],
            "monthly_level_authority": "CH_EEX_BASE_SOLVER",
        }
    )


def _peak_sidecar_frame(peak: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in peak.sort_values("product").iterrows():
        periods = product_periods(str(row["product"]))
        rows.append(
            {
                "product": str(row["product"]),
                "product_type": str(row["product_type"]),
                "load_type": "PEAK",
                "price_eur_mwh": float(row["price"]),
                "quotation_date": pd.Timestamp(row["date"]),
                "delivery_start_month": str(periods.min()),
                "delivery_end_month": str(periods.max()),
                "product_id": str(row["product_id"]),
                "delivery_period_id": str(row["delivery_period_id"]),
                "quote_id": _row_quote_id(row),
                "monthly_level_authority": False,
                "shaping_input_authorized": False,
            }
        )
    return pd.DataFrame(rows)


def _all_base_quote_diagnostics(
    base: pd.DataFrame,
    month_hours: pd.Series,
    monthly_curve: pd.Series,
    quote_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    dropped_by_product = {
        str(row["product"]): str(row["dropped_reason"]) for _, row in quote_diagnostics.iterrows()
    }
    rows: list[dict[str, object]] = []
    for _, row in base.sort_values("product").iterrows():
        product = str(row["product"])
        periods = product_periods(product)
        hours = month_hours.reindex(periods).to_numpy(dtype=float)
        values = monthly_curve.reindex(periods).to_numpy(dtype=float)
        if np.isnan(values).any() or np.isnan(hours).any():
            raise EexCurrentMonthlyResearchSolutionError(
                f"BASE quote period falls outside solution: {product}"
            )
        repriced = float(np.average(values, weights=hours))
        target = float(row["price"])
        residual = repriced - target
        dropped = dropped_by_product.get(product, "")
        rows.append(
            {
                "product": product,
                "product_type": str(row["product_type"]),
                "quote_id": _row_quote_id(row),
                "constraint_role": (
                    "REDUNDANT_CONSISTENCY" if dropped == "redundant_consistent" else "ACTIVE_HARD"
                ),
                "target_price_eur_mwh": target,
                "repriced_eur_mwh": repriced,
                "residual_eur_mwh": residual,
                "abs_residual_eur_mwh": abs(residual),
                "hard_tolerance_eur_mwh": CONFIG.constraint_tolerance,
                "consistency_tolerance_eur_mwh": CONFIG.quote_conflict_tolerance,
            }
        )
    return pd.DataFrame(rows)


def _prior_parent_diagnostics(constraints, shape: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bucket in constraints.month_buckets.dropna().drop_duplicates():
        mask = constraints.month_buckets.eq(bucket)
        months = constraints.month_buckets.index[mask]
        hours = constraints.delivery_grid.month_hours.loc[months].to_numpy(dtype=float)
        values = shape.reindex(months).to_numpy(dtype=float)
        mean = float(np.average(values, weights=hours))
        rows.append(
            {
                "parent_bucket": str(bucket),
                "month_count": len(months),
                "delivery_hours": float(hours.sum()),
                "weighted_mean_eur_mwh": mean,
                "abs_weighted_mean_eur_mwh": abs(mean),
            }
        )
    return pd.DataFrame(rows).sort_values("parent_bucket").reset_index(drop=True)


def _validate_predecessor_bindings(
    surface: Mapping[str, object],
    horizon: Mapping[str, object],
) -> None:
    source = _mapping(surface.get("source"), "D212 surface source")
    _equal(
        source.get("normalization_content_id"), NORMALIZATION_CONTENT_ID, "surface normalization"
    )
    _equal(
        source.get("normalization_manifest_sha256"),
        NORMALIZATION_MANIFEST_SHA256,
        "surface manifest binding",
    )
    _equal(source.get("history_sha256"), HISTORY_SHA256, "surface history binding")
    _equal(
        surface.get("status"), "PASS_LOCAL_INTEGRITY_NO_MARKET_QUOTE_CONFLICTS", "surface status"
    )
    _equal(
        horizon.get("source_normalization_content_id"),
        NORMALIZATION_CONTENT_ID,
        "horizon normalization",
    )
    _equal(
        horizon.get("source_normalization_manifest_sha256"),
        NORMALIZATION_MANIFEST_SHA256,
        "horizon manifest binding",
    )
    _equal(
        horizon.get("source_normalized_history_sha256"), HISTORY_SHA256, "horizon history binding"
    )
    _equal(
        horizon.get("source_surface_audit_content_id"), SURFACE_AUDIT_CONTENT_ID, "horizon surface"
    )
    _equal(
        horizon.get("source_surface_audit_manifest_sha256"),
        SURFACE_AUDIT_MANIFEST_SHA256,
        "horizon surface manifest",
    )
    _equal(
        horizon.get("status"),
        "PASS_LOCAL_DESCRIPTIVE_HORIZON_COVERAGE_WITH_GAP_ANOMALIES_NOT_PIT",
        "horizon status",
    )
    for manifest, label in ((surface, "surface"), (horizon, "horizon")):
        authority = _mapping(manifest.get("authority"), f"{label} authority")
        for field in (
            "point_in_time_availability_proven",
            "rolling_origin_selection_authorized",
            "production_promotion_authorized",
        ):
            _equal(authority.get(field), False, f"{label} {field}")


def _validate_normalization_manifest(normalization: Mapping[str, object]) -> None:
    _equal(
        normalization.get("status"),
        "PASS_LOCAL_ALL_PRODUCTS_NORMALIZATION_WITH_QUARANTINE_NOT_PIT_OR_PROMOTION_AUTHORITY",
        "D212 normalization status",
    )
    source = _mapping(normalization.get("source"), "D212 normalization source")
    _equal(source.get("source_snapshot_sha256"), SOURCE_SNAPSHOT_SHA256, "source snapshot")
    artifacts = _mapping(normalization.get("artifacts"), "D212 normalization artifacts")
    for name, path, sha256, rows in (
        ("solver_history", HISTORY_PATH, HISTORY_SHA256, HISTORY_ROWS),
        ("solver_latest", LATEST_PATH, LATEST_SHA256, LATEST_ROWS),
    ):
        artifact = _mapping(artifacts.get(name), f"D212 {name}")
        _equal(artifact.get("path"), path.name, f"D212 {name} path")
        _equal(artifact.get("sha256"), sha256, f"D212 {name} SHA-256")
        _equal(artifact.get("row_count"), rows, f"D212 {name} row count")
    authority = _mapping(normalization.get("authority"), "D212 normalization authority")
    _equal(
        authority.get("local_research_and_pipeline_development"),
        True,
        "D212 local authority",
    )
    for key in (
        "point_in_time_availability_proven",
        "rolling_origin_selection_authorized",
        "production_promotion_authorized",
    ):
        _equal(authority.get(key), False, f"D212 {key}")


def _bound_manifest(
    payload: bytes,
    *,
    expected_sha256: str,
    expected_content_id: str,
    label: str,
) -> Mapping[str, object]:
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise EexCurrentMonthlyResearchSolutionError(f"{label} SHA-256 differs")
    document = _strict_json(payload, label)
    _equal(document.get("content_id"), expected_content_id, f"{label} content ID")
    return document


def _frame_content_id(frame: pd.DataFrame) -> str:
    records: list[dict[str, object]] = []
    for raw in frame.to_dict(orient="records"):
        record: dict[str, object] = {}
        for key, value in raw.items():
            if isinstance(value, pd.Timestamp):
                record[str(key)] = value.isoformat()
            elif isinstance(value, np.generic):
                record[str(key)] = value.item()
            else:
                record[str(key)] = value
        records.append(record)
    return canonical_sha256({"columns": list(frame.columns), "records": records})


def _read(path: Path, label: str, maximum_bytes: int) -> bytes:
    if not path.is_absolute():
        raise EexCurrentMonthlyResearchSolutionError(f"{label} path must be absolute")
    try:
        return read_stable_single_link_file(path, label=label, max_bytes=maximum_bytes)
    except (OSError, ValueError) as exc:
        raise EexCurrentMonthlyResearchSolutionError(f"{label} read failed") from exc


def _strict_json(payload: bytes, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EexCurrentMonthlyResearchSolutionError(f"{label} JSON is invalid") from exc
    return _mapping(value, label)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EexCurrentMonthlyResearchSolutionError("JSON contains duplicate keys")
        result[key] = value
    return result


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EexCurrentMonthlyResearchSolutionError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise EexCurrentMonthlyResearchSolutionError(f"{label} fields differ")


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise EexCurrentMonthlyResearchSolutionError(f"{label} differs")


def _clean(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise EexCurrentMonthlyResearchSolutionError(f"{label} must be clean text")
    return value


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _utc_datetime(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EexCurrentMonthlyResearchSolutionError(f"{label} must be timezone-aware")
    normalized = value.astimezone(timezone.utc)
    if normalized.microsecond:
        raise EexCurrentMonthlyResearchSolutionError(f"{label} must have whole-second precision")
    return normalized


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "ASSESSMENT_SCHEMA",
    "AUTHORITIES",
    "CONTRACT_CONTENT_ID",
    "CONTRACT_PATH",
    "DELIVERY_MONTHS",
    "EXECUTION",
    "HISTORY_PATH",
    "LATEST_PATH",
    "PASS_STATUS",
    "CurrentMonthlyResearchSolution",
    "EexCurrentMonthlyResearchSolutionError",
    "build_selected_current_monthly_research_solution",
    "canonical_sha256",
    "solve_current_monthly_research_frames",
    "validate_current_monthly_solution_contract",
    "validate_current_monthly_solution_contract_path",
]
