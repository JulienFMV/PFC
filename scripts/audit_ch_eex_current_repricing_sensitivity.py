"""Audit current CH EEX repricing and BASE quote-to-curve sensitivity.

This is a deliberately non-authoritative local diagnostic.  It binds the
selected quarantine capture, excludes products whose delivery has started,
and separates three claims that must not be conflated:

* exact repricing of the active independent BASE constraint set;
* exact point repricing of every overlapping BASE and PEAK source quote;
* the implied OFFPEAK energy identity for shared BASE/PEAK products; and
* deterministic quote-to-month sensitivity on a fixed hierarchy.

The report never emits source quote prices or monthly curve levels.  It cannot
grant trusted point-in-time, EEX semantics, monthly-level, candidate,
promotion, or production authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.calibration.eex_contract_selection import calibration_buckets
from pfc_shaping.calibration.monthly_curve_sensitivity import (
    MonthlySensitivityPolicy,
    audit_monthly_quote_curve_sensitivity,
)
from pfc_shaping.calibration.monthly_forward_curve import (
    MarketQuote,
    MonthlyCurveConfig,
    build_monthly_constraint_system,
    product_periods,
    solve_monthly_forward_curve_from_constraints,
)
from pfc_shaping.data.ingest_forwards import (
    load_base_prices_from_eex_report_bytes,
)
from pfc_shaping.lt.model.shape_constraints import (
    build_base_peak_offpeak_constraint_system,
    eex_peak_mask,
)
from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.pipeline.strict_structured_data import load_strict_json
from scripts.audit_ch_eex_current_local_capture import (
    REGISTRY_RELATIVE_PATH,
    CurrentEexCaptureError,
    audit_current_eex_capture,
    canonical_json_bytes,
    compute_quote_commitments,
)

SCHEMA_VERSION = "ch_eex_current_repricing_sensitivity.v2"
STATUS = "LOCAL_CURRENT_EEX_BASE_PEAK_MECHANICS_PARTIAL_PASS_PRODUCTION_NO_GO"
MECHANICS_ERROR_STATUS = "LOCAL_CURRENT_EEX_BASE_PEAK_MECHANICS_FAIL_PRODUCTION_NO_GO"
ERROR_STATUS = "LOCAL_CURRENT_EEX_REPRICING_SENSITIVITY_INVALID_NO_GO"
SOURCE_POINT_REPRICING_TOLERANCE_EUR_MWH = 1e-9
REFERENCE_SHOCK_EUR_MWH = 0.1
CROSS_CHECK_SHOCK_EUR_MWH = 1.0
CROSS_SCALE_DERIVATIVE_TOLERANCE = 1e-7
_MAX_SOURCE_BYTES = 64 * 1024 * 1024


def audit_current_eex_repricing_sensitivity(
    *,
    registry_path: Path,
    expected_registry_sha256: str,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Bind the selected capture and evaluate its non-disclosing BASE mechanics."""

    root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else repository_root.expanduser()
    )
    capture_audit = audit_current_eex_capture(
        registry_path=registry_path,
        expected_registry_sha256=expected_registry_sha256,
        repository_root=root,
    )
    canonical_registry = root.joinpath(*REGISTRY_RELATIVE_PATH.split("/"))
    registry_payload = read_stable_single_link_file(
        canonical_registry,
        label="current EEX registry for repricing audit",
        max_bytes=2_000_000,
    )
    if hashlib.sha256(registry_payload).hexdigest() != expected_registry_sha256:
        raise CurrentEexCaptureError("current EEX registry changed before repricing audit")
    registry = load_strict_json(registry_payload.decode("utf-8"))
    if not isinstance(registry, Mapping):
        raise CurrentEexCaptureError("current EEX registry is not an object")
    current = registry.get("current_capture")
    if not isinstance(current, Mapping):
        raise CurrentEexCaptureError("current EEX capture reference is invalid")
    source_path = _repo_path(root, current.get("source_path"), label="current EEX source")
    source_sha256 = str(current.get("source_sha256", ""))
    source_payload = read_stable_single_link_file(
        source_path,
        label="current EEX source for repricing audit",
        max_bytes=_MAX_SOURCE_BYTES,
    )
    if hashlib.sha256(source_payload).hexdigest() != source_sha256:
        raise CurrentEexCaptureError("current EEX source changed before repricing audit")
    report = evaluate_current_eex_repricing_mechanics(
        source_payload=source_payload,
        source_sha256=source_sha256,
        source_semantics_attested=False,
    )
    report.update(
        {
            "registry_path": REGISTRY_RELATIVE_PATH,
            "registry_sha256": expected_registry_sha256,
            "selection_id": capture_audit["selection_id"],
            "capture_id": capture_audit["current_capture_id"],
        }
    )
    report["report_id"] = _report_id(report)
    return report


def evaluate_current_eex_repricing_mechanics(
    *,
    source_payload: bytes,
    source_sha256: str,
    source_semantics_attested: bool,
) -> dict[str, object]:
    """Evaluate BASE sensitivity and joint hourly mechanics without price disclosure."""

    if type(source_semantics_attested) is not bool:
        raise ValueError("source semantics attestation flag must be boolean")
    if hashlib.sha256(source_payload).hexdigest() != source_sha256:
        raise ValueError("source SHA-256 does not bind supplied EEX bytes")
    commitments = compute_quote_commitments(source_payload, market="CH")
    prices, snapshot_date, metadata = load_base_prices_from_eex_report_bytes(
        source_payload,
        market="CH",
        source_label="<current captured CH EEX bytes>",
        return_snapshot_metadata=True,
    )
    if not isinstance(prices, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("current EEX price replay is invalid")
    lineage = metadata.get("quote_lineage")
    if not isinstance(lineage, Mapping) or set(lineage) != set(prices):
        raise ValueError("current EEX quote lineage is not exact")
    snapshot = pd.Timestamp(snapshot_date).tz_localize(None).normalize()
    first_undelivered_month = pd.Period(snapshot, freq="M") + 1

    base_quotes: list[MarketQuote] = []
    future_base_prices: dict[str, float] = {}
    future_peak_prices: dict[str, float] = {}
    excluded_base: list[dict[str, str]] = []
    excluded_peak_count = 0
    for raw_key, raw_price in sorted(prices.items()):
        key = str(raw_key)
        is_peak = key.endswith("-Peak")
        product = key.removesuffix("-Peak") if is_peak else key
        periods = product_periods(product)
        before = periods < first_undelivered_month
        if bool(before.any()):
            reason = (
                "PARTIALLY_DELIVERED_AT_LOCAL_UNTRUSTED_VALUATION_BOUNDARY"
                if not bool(before.all())
                else "DELIVERY_ALREADY_STARTED_AT_LOCAL_UNTRUSTED_VALUATION_BOUNDARY"
            )
            if is_peak:
                excluded_peak_count += 1
            else:
                excluded_base.append({"product": product, "reason": reason})
            continue
        raw_lineage = lineage.get(key)
        if not isinstance(raw_lineage, Mapping):
            raise ValueError("current EEX quote lineage is invalid")
        quote_id = str(raw_lineage.get("source_product_code", "")).strip().upper()
        if not quote_id:
            raise ValueError("current EEX quote ID is missing")
        if is_peak:
            future_peak_prices[product] = float(raw_price)
            continue
        future_base_prices[product] = float(raw_price)
        base_quotes.append(
            MarketQuote(
                market="CH",
                product=product,
                load_type="BASE",
                price=float(raw_price),
                snapshot_date=snapshot,
                source="CURRENT_LOCAL_QUARANTINE_CAPTURE_NOT_PIT_AUTHORITY",
                available_at=snapshot,
                quote_id=quote_id,
                source_kind="LOCAL_UNTRUSTED_EEX_CAPTURE",
                source_sha256=source_sha256,
            )
        )
    if not base_quotes:
        raise ValueError("current EEX capture has no wholly undelivered BASE products")
    if not future_peak_prices:
        raise ValueError("current EEX capture has no wholly undelivered PEAK products")
    delivery_months = pd.period_range(
        min(
            min(product_periods(product))
            for product in set(future_base_prices) | set(future_peak_prices)
        ),
        max(
            max(product_periods(product))
            for product in set(future_base_prices) | set(future_peak_prices)
        ),
        freq="M",
    )
    joint_oracle = _evaluate_joint_hourly_oracle(
        base_prices=future_base_prices,
        peak_prices=future_peak_prices,
        delivery_months=delivery_months,
    )
    config = MonthlyCurveConfig(
        lambda_prior=1e-6,
        lambda_smooth_month=1.0,
        lambda_smooth_yoy=0.25,
        lambda_shape=1.0,
        constraint_tolerance=SOURCE_POINT_REPRICING_TOLERANCE_EUR_MWH,
        quote_conflict_tolerance=0.01,
        stationarity_tolerance=1e-7,
    )
    reference = audit_monthly_quote_curve_sensitivity(
        delivery_months=delivery_months,
        own_quotes=tuple(base_quotes),
        config=config,
        shape_prior=None,
        policy=MonthlySensitivityPolicy(
            perturbation_eur_mwh=REFERENCE_SHOCK_EUR_MWH
        ),
    )
    cross_check = audit_monthly_quote_curve_sensitivity(
        delivery_months=delivery_months,
        own_quotes=tuple(base_quotes),
        config=config,
        shape_prior=None,
        policy=MonthlySensitivityPolicy(
            perturbation_eur_mwh=CROSS_CHECK_SHOCK_EUR_MWH
        ),
    )
    cross_scale_error = _cross_scale_sensitivity_error(
        reference.sensitivities,
        cross_check.sensitivities,
    )
    constraints = build_monthly_constraint_system(
        delivery_months,
        tuple(base_quotes),
        timezone="Europe/Zurich",
        market="CH",
        load_type="BASE",
        constraint_tolerance=config.constraint_tolerance,
        quote_conflict_tolerance=config.quote_conflict_tolerance,
    )
    result = solve_monthly_forward_curve_from_constraints(
        constraints,
        config=config,
        shape_prior=None,
    )
    quote_diagnostics = constraints.quote_diagnostics.set_index("product")
    curve = result.monthly_curve.to_numpy(dtype=float)
    hours = constraints.delivery_grid.month_hours.to_numpy(dtype=float)
    residual_rows: list[dict[str, object]] = []
    for quote in base_quotes:
        support = delivery_months.isin(product_periods(quote.product))
        implied = float(np.average(curve[support], weights=hours[support]))
        residual_rows.append(
            {
                "product": quote.product,
                "active_independent_quote": bool(
                    quote_diagnostics.loc[quote.product, "active"]
                ),
                "abs_repricing_error_eur_mwh": abs(implied - float(quote.price)),
            }
        )
    active_errors = [
        float(row["abs_repricing_error_eur_mwh"])
        for row in residual_rows
        if bool(row["active_independent_quote"])
    ]
    all_errors = [float(row["abs_repricing_error_eur_mwh"]) for row in residual_rows]
    active_max = max(active_errors)
    all_max = max(all_errors)
    inactive_count = sum(
        not bool(row["active_independent_quote"]) for row in residual_rows
    )
    inactive_products = sorted(
        str(row["product"])
        for row in residual_rows
        if not bool(row["active_independent_quote"])
    )
    reference_passed = str(reference.summary.get("status", "")).startswith("PASS_")
    cross_check_passed = str(cross_check.summary.get("status", "")).startswith("PASS_")
    active_exact = active_max <= SOURCE_POINT_REPRICING_TOLERANCE_EUR_MWH
    all_points_exact = all_max <= SOURCE_POINT_REPRICING_TOLERANCE_EUR_MWH
    cross_scale_passed = (
        math.isfinite(cross_scale_error)
        and cross_scale_error <= CROSS_SCALE_DERIVATIVE_TOLERANCE
    )
    joint_feasible = bool(joint_oracle["constraint_system_feasible"])
    joint_constraints_exact = bool(joint_oracle["active_constraint_rows_exact"])
    peak_active_exact = bool(joint_oracle["peak_repricing"]["active_independent_points_exact"])
    implied_offpeak_exact = bool(
        joint_oracle["implied_offpeak_repricing"]["active_shared_points_exact"]
    )
    gates = [
        _gate("active_independent_base_repricing_exact", active_exact, active_max),
        _gate("all_future_base_source_points_reprice_exactly", all_points_exact, all_max),
        _gate("joint_hourly_constraint_system_feasible", joint_feasible, joint_feasible),
        _gate(
            "joint_hourly_active_constraint_rows_exact",
            joint_constraints_exact,
            joint_oracle["active_constraint_max_abs_error_eur_mwh"],
        ),
        _gate(
            "active_independent_peak_repricing_exact",
            peak_active_exact,
            joint_oracle["peak_repricing"]["active_max_abs_error_eur_mwh"],
        ),
        _gate(
            "active_shared_implied_offpeak_identity_exact",
            implied_offpeak_exact,
            joint_oracle["implied_offpeak_repricing"][
                "active_shared_max_abs_error_eur_mwh"
            ],
        ),
        _gate(
            "source_tick_rounding_and_settlement_semantics_attested",
            source_semantics_attested,
            source_semantics_attested,
        ),
        _gate("reference_scale_sensitivity", reference_passed, reference_passed),
        _gate("cross_check_scale_sensitivity", cross_check_passed, cross_check_passed),
        _gate("cross_scale_derivative_invariance", cross_scale_passed, cross_scale_error),
        _gate("peak_hourly_mechanical_oracle_evaluated", True, True),
        _gate("final_delivered_hourly_candidate_repricing_evaluated", False, False),
    ]
    mechanics_passed = (
        active_exact
        and joint_feasible
        and joint_constraints_exact
        and peak_active_exact
        and implied_offpeak_exact
        and reference_passed
        and cross_check_passed
        and cross_scale_passed
    )
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS if mechanics_passed else MECHANICS_ERROR_STATUS,
        "local_base_mechanics_status": (
            "PASS_ACTIVE_BASE_CONSTRAINTS_AND_SENSITIVITY"
            if mechanics_passed
            else "FAIL_LOCAL_BASE_MECHANICS_NO_GO"
        ),
        "local_joint_hourly_mechanics_status": (
            "PASS_ACTIVE_BASE_PEAK_AND_IMPLIED_OFFPEAK_MECHANICS"
            if joint_feasible
            and joint_constraints_exact
            and peak_active_exact
            and implied_offpeak_exact
            else "FAIL_LOCAL_JOINT_HOURLY_MECHANICS_NO_GO"
        ),
        "source_sha256": source_sha256,
        "snapshot_date": snapshot.strftime("%Y-%m-%d"),
        "first_undelivered_month_local_untrusted": str(first_undelivered_month),
        "delivery_domain_start": str(delivery_months[0]),
        "delivery_domain_end": str(delivery_months[-1]),
        "quote_commitments": commitments,
        "quote_values_disclosed": False,
        "source_quote_counts": {
            "total": len(prices),
            "base_total": sum(not str(key).endswith("-Peak") for key in prices),
            "peak_total": sum(str(key).endswith("-Peak") for key in prices),
            "future_base_solver_domain": len(base_quotes),
            "future_peak_joint_oracle_domain": len(future_peak_prices),
            "explicit_offpeak_total": 0,
            "excluded_started_base": len(excluded_base),
            "excluded_started_peak": excluded_peak_count,
        },
        "excluded_started_base_products": excluded_base,
        "base_repricing": {
            "tolerance_eur_mwh": SOURCE_POINT_REPRICING_TOLERANCE_EUR_MWH,
            "quote_conflict_tolerance_eur_mwh": config.quote_conflict_tolerance,
            "active_independent_quote_count": len(active_errors),
            "redundant_consistency_only_quote_count": inactive_count,
            "redundant_consistency_only_products": inactive_products,
            "active_max_abs_error_eur_mwh": active_max,
            "all_source_points_max_abs_error_eur_mwh": all_max,
            "active_independent_points_exact": active_exact,
            "all_source_points_exact": all_points_exact,
            "rounding_or_tick_interval_pass_claimed": False,
            "source_semantics_attested": source_semantics_attested,
        },
        "joint_hourly_mechanical_oracle": joint_oracle,
        "sensitivity": {
            "reference_shock_eur_mwh": REFERENCE_SHOCK_EUR_MWH,
            "cross_check_shock_eur_mwh": CROSS_CHECK_SHOCK_EUR_MWH,
            "cross_scale_max_abs_derivative_error": cross_scale_error,
            "cross_scale_derivative_tolerance": CROSS_SCALE_DERIVATIVE_TOLERANCE,
            "reference_status": reference.summary["status"],
            "cross_check_status": cross_check.summary["status"],
            "constraint_topology_sha256": reference.summary[
                "constraint_topology_sha256"
            ],
            "active_products": reference.summary["active_products"],
            "inactive_quote_reasons": reference.summary["inactive_quote_reasons"],
            "reference_metrics": _public_sensitivity_metrics(reference.summary),
            "cross_check_metrics": _public_sensitivity_metrics(cross_check.summary),
        },
        "gates": gates,
        "limitations": [
            "LOCAL_QUARANTINE_CAPTURE_NOT_TRUSTED_POINT_IN_TIME",
            "SOURCE_PRODUCT_SESSION_SETTLEMENT_TICK_AND_ROUNDING_SEMANTICS_UNATTESTED",
            "ALL_SOURCE_QUOTE_POINT_REPRICING_FAILS_EXACT_TOLERANCE",
            "FINAL_DELIVERED_HOURLY_CANDIDATE_REPRICING_NOT_EVALUATED",
            "EXPLICIT_OFFPEAK_SOURCE_QUOTES_ABSENT_IMPLIED_IDENTITY_ONLY",
            "NO_ECONOMIC_SHAPE_PRIOR_OR_PROBABILISTIC_SCENARIO_VALIDATION",
            "NO_ROLLING_ORIGIN_OR_FUTURE_HOLDOUT_EVIDENCE",
            "OMPEX_NOT_USED",
            "TARGET_CODE_EXECUTION_IDENTITY_REQUIRES_SEPARATE_RUNNER_OR_PREIMPORT_VERIFIER_RECEIPT",
            "WORKSPACE_RUNNER_IS_POST_IMPORT_LOCAL_OBSERVATION_NOT_PREIMPORT_ADMISSION",
            "DOES_NOT_AUTHORIZE_SOLVER_LEVEL_CANDIDATE_PUBLICATION_PROMOTION_OR_PRODUCTION",
        ],
        "monthly_level_authority": "solver",
        "scientific_admission": False,
        "monthly_solver_hard_level_eligible": False,
        "candidate_assembly_eligible": False,
        "publication_authorized": False,
        "promotion_gate": False,
        "production_authorization": False,
        "t057_consumed": False,
        "ompex_used": False,
    }
    return report


def _evaluate_joint_hourly_oracle(
    *,
    base_prices: Mapping[str, float],
    peak_prices: Mapping[str, float],
    delivery_months: pd.PeriodIndex,
) -> dict[str, object]:
    """Build and reprice a feasible hourly oracle without exposing price levels."""

    start_local = delivery_months[0].to_timestamp(how="start").tz_localize(
        "Europe/Zurich"
    )
    end_local = (delivery_months[-1] + 1).to_timestamp(how="start").tz_localize(
        "Europe/Zurich"
    )
    index = pd.date_range(
        start_local.tz_convert("UTC"),
        end_local.tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    system = build_base_peak_offpeak_constraint_system(
        index,
        base_prices,
        peak_prices,
        country="CH",
    )
    feasibility = system.feasibility_report()
    if not feasibility.feasible:
        raise ValueError("joint BASE/PEAK/OFFPEAK hourly oracle is infeasible")
    matrix = system.sparse_matrix
    gram = (matrix @ matrix.T).toarray()
    dual, *_ = np.linalg.lstsq(gram, system.targets, rcond=None)
    curve = np.asarray(matrix.T @ dual, dtype=float).reshape(-1)
    constraint_errors = np.abs(np.asarray(matrix @ curve).reshape(-1) - system.targets)
    constraint_max = float(constraint_errors.max()) if len(constraint_errors) else 0.0

    local = pd.Series(index.tz_convert("Europe/Zurich"), index=range(len(index)))
    peak_mask = eex_peak_mask(index, country="CH")
    base_active = _active_source_products(local, base_prices)
    peak_active = _active_source_products(local.loc[peak_mask], peak_prices)
    base_summary = _raw_quote_repricing_summary(
        curve=curve,
        local=local,
        delivery_mask=np.ones(len(index), dtype=bool),
        prices=base_prices,
        active_products=base_active,
    )
    peak_summary = _raw_quote_repricing_summary(
        curve=curve,
        local=local,
        delivery_mask=peak_mask,
        prices=peak_prices,
        active_products=peak_active,
    )
    offpeak_summary = _implied_offpeak_repricing_summary(
        curve=curve,
        local=local,
        peak_mask=peak_mask,
        base_prices=base_prices,
        peak_prices=peak_prices,
        base_active=base_active,
        peak_active=peak_active,
    )
    return {
        "basis": "HOURLY_UTC_WITH_CH_LOCAL_PRODUCT_AND_EEX_PEAK_CALENDAR",
        "hour_count": len(index),
        "delivery_domain_start": str(delivery_months[0]),
        "delivery_domain_end": str(delivery_months[-1]),
        "constraint_count": len(system.rows),
        "base_constraint_count": sum(row.kind == "BASE" for row in system.rows),
        "peak_constraint_count": sum(row.kind == "PEAK" for row in system.rows),
        "offpeak_constraint_count": sum(row.kind == "OFFPEAK" for row in system.rows),
        "constraint_rank": feasibility.rank_a,
        "constraint_system_feasible": feasibility.feasible,
        "active_constraint_max_abs_error_eur_mwh": constraint_max,
        "active_constraint_rows_exact": (
            constraint_max <= SOURCE_POINT_REPRICING_TOLERANCE_EUR_MWH
        ),
        "base_repricing": base_summary,
        "peak_repricing": peak_summary,
        "implied_offpeak_repricing": offpeak_summary,
        "curve_values_disclosed": False,
        "mechanical_oracle_only_not_delivered_candidate": True,
    }


def _active_source_products(
    local: pd.Series,
    prices: Mapping[str, float],
) -> set[str]:
    _buckets, targets = calibration_buckets(local, prices)
    return {
        str(product)
        for product in prices
        if str(product) in targets or f"{product}-RESIDUAL" in targets
    }


def _product_delivery_mask(local: pd.Series, product: str) -> np.ndarray:
    covered = {str(month) for month in product_periods(product)}
    labels = local.dt.strftime("%Y-%m").to_numpy(dtype=object)
    return np.isin(labels, tuple(sorted(covered)))


def _raw_quote_repricing_summary(
    *,
    curve: np.ndarray,
    local: pd.Series,
    delivery_mask: np.ndarray,
    prices: Mapping[str, float],
    active_products: set[str],
) -> dict[str, object]:
    rows: list[tuple[str, bool, float]] = []
    for product, target in sorted(prices.items()):
        mask = _product_delivery_mask(local, str(product)) & delivery_mask
        if not bool(mask.any()):
            raise ValueError(f"hourly oracle does not cover EEX product {product}")
        error = abs(float(curve[mask].mean()) - float(target))
        rows.append((str(product), str(product) in active_products, error))
    active_errors = [error for _product, active, error in rows if active]
    all_errors = [error for _product, _active, error in rows]
    inactive = sorted(product for product, active, _error in rows if not active)
    active_max = max(active_errors) if active_errors else math.inf
    all_max = max(all_errors) if all_errors else math.inf
    return {
        "tolerance_eur_mwh": SOURCE_POINT_REPRICING_TOLERANCE_EUR_MWH,
        "active_independent_quote_count": len(active_errors),
        "redundant_consistency_only_quote_count": len(inactive),
        "redundant_consistency_only_products": inactive,
        "active_max_abs_error_eur_mwh": active_max,
        "all_source_points_max_abs_error_eur_mwh": all_max,
        "active_independent_points_exact": (
            bool(active_errors)
            and active_max <= SOURCE_POINT_REPRICING_TOLERANCE_EUR_MWH
        ),
        "all_source_points_exact": (
            bool(all_errors)
            and all_max <= SOURCE_POINT_REPRICING_TOLERANCE_EUR_MWH
        ),
    }


def _implied_offpeak_repricing_summary(
    *,
    curve: np.ndarray,
    local: pd.Series,
    peak_mask: np.ndarray,
    base_prices: Mapping[str, float],
    peak_prices: Mapping[str, float],
    base_active: set[str],
    peak_active: set[str],
) -> dict[str, object]:
    rows: list[tuple[str, bool, float]] = []
    for product in sorted(set(base_prices) & set(peak_prices)):
        product_mask = _product_delivery_mask(local, product)
        peak_product = product_mask & peak_mask
        offpeak_product = product_mask & ~peak_mask
        total_hours = int(product_mask.sum())
        peak_hours = int(peak_product.sum())
        offpeak_hours = int(offpeak_product.sum())
        if min(total_hours, peak_hours, offpeak_hours) <= 0:
            raise ValueError(f"cannot evaluate implied OFFPEAK identity for {product}")
        implied_target = (
            float(base_prices[product]) * total_hours
            - float(peak_prices[product]) * peak_hours
        ) / offpeak_hours
        error = abs(float(curve[offpeak_product].mean()) - implied_target)
        active_shared = product in base_active and product in peak_active
        rows.append((product, active_shared, error))
    active_errors = [error for _product, active, error in rows if active]
    all_errors = [error for _product, _active, error in rows]
    inactive = sorted(product for product, active, _error in rows if not active)
    active_max = max(active_errors) if active_errors else math.inf
    all_max = max(all_errors) if all_errors else math.inf
    return {
        "basis": "IMPLIED_FROM_SHARED_BASE_AND_PEAK_QUOTES_ENERGY_IDENTITY",
        "explicit_offpeak_quote_count": 0,
        "shared_source_product_count": len(rows),
        "active_shared_product_count": len(active_errors),
        "redundant_or_mixed_shared_product_count": len(inactive),
        "redundant_or_mixed_shared_products": inactive,
        "active_shared_max_abs_error_eur_mwh": active_max,
        "all_shared_max_abs_error_eur_mwh": all_max,
        "active_shared_points_exact": (
            bool(active_errors)
            and active_max <= SOURCE_POINT_REPRICING_TOLERANCE_EUR_MWH
        ),
        "all_shared_points_exact": (
            bool(all_errors)
            and all_max <= SOURCE_POINT_REPRICING_TOLERANCE_EUR_MWH
        ),
    }


def _public_sensitivity_metrics(summary: Mapping[str, object]) -> dict[str, object]:
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("monthly sensitivity metrics are invalid")
    fields = (
        "input_quote_count",
        "independent_quote_count",
        "inactive_quote_count",
        "condition_number",
        "baseline_constraint_residual",
        "baseline_stationarity_residual",
        "max_central_linearity_error",
        "max_active_constraint_derivative_error",
        "max_quote_repricing_jacobian_error",
        "max_quote_support_weighted_gain",
        "max_delivery_year_weighted_simultaneous_gain",
        "max_cascade_bucket_weighted_simultaneous_gain",
        "monthly_bounded_quote_shock_gain",
    )
    return {field: metrics[field] for field in fields}


def _cross_scale_sensitivity_error(left: pd.DataFrame, right: pd.DataFrame) -> float:
    keys = ["quote_product", "delivery_month"]
    value = "sensitivity_eur_mwh_per_eur_mwh"
    if set(keys + [value]) - set(left) or set(keys + [value]) - set(right):
        raise ValueError("sensitivity frame schema is incomplete")
    merged = left[keys + [value]].merge(
        right[keys + [value]],
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_reference", "_cross_check"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("sensitivity scales do not share an exact response domain")
    delta = (
        merged[f"{value}_reference"].astype(float)
        - merged[f"{value}_cross_check"].astype(float)
    ).abs()
    return float(delta.max()) if not delta.empty else 0.0


def _gate(gate_id: str, passed: bool, value: object) -> dict[str, object]:
    return {
        "gate_id": gate_id,
        "status": "PASS" if passed else "FAIL",
        "value": value,
    }


def _repo_path(root: Path, raw: object, *, label: str) -> Path:
    if (
        not isinstance(raw, str)
        or not raw
        or "\\" in raw
        or Path(raw).is_absolute()
        or any(part in {"", ".", ".."} for part in raw.split("/"))
    ):
        raise CurrentEexCaptureError(f"{label} path is invalid")
    candidate = root.joinpath(*raw.split("/"))
    if os.path.commonpath((str(root), str(candidate))) != str(root):
        raise CurrentEexCaptureError(f"{label} path escapes the repository")
    return candidate


def _report_id(report: Mapping[str, object]) -> str:
    unsigned = dict(report)
    unsigned.pop("report_id", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        result = audit_current_eex_repricing_sensitivity(
            registry_path=args.registry,
            expected_registry_sha256=args.expected_registry_sha256,
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": ERROR_STATUS,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "quote_values_disclosed": False,
                    "scientific_admission": False,
                    "monthly_solver_hard_level_eligible": False,
                    "candidate_assembly_eligible": False,
                    "promotion_gate": False,
                    "production_authorization": False,
                    "t057_consumed": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
