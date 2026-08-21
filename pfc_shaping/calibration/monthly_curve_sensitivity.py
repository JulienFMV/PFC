"""Fail-closed quote-to-month sensitivity and hierarchy invariance audit.

The monthly solver is an equality-constrained quadratic problem.  For a fixed
quote hierarchy its solution must be affine in the independent hard quotes.
This module turns that structural property into auditable evidence:

* every independent quote is perturbed symmetrically;
* the full quote-to-month Jacobian is measured;
* hard-product repricing derivatives must form the identity matrix;
* redundant-parent removal and input ordering must not change the curve; and
* the response may not amplify beyond the energy-residual cascade geometry.

The audit is deliberately non-promotional.  It measures one deterministic
solver state; it does not validate the market data or the economic priors.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from pfc_shaping.calibration.monthly_curve_config_identity import (
    MAX_SOLVER_CONDITION_NUMBER,
)
from pfc_shaping.calibration.monthly_forward_curve import (
    MarketQuote,
    MonthlyCurveConfig,
    MonthlyObjectiveSystem,
    build_monthly_objective_system,
    build_monthly_constraint_system,
    product_periods,
    solve_monthly_forward_curve_from_constraints,
)

SENSITIVITY_SCHEMA_VERSION = "monthly_quote_curve_sensitivity.v3"
_ALLOWED_OBJECTIVE_TERMS = {
    "numerical_parent_flat_ridge",
    "parent_flat_prior",
    "shape_prior",
    "smooth_month",
    "smooth_yoy_shape",
}


@dataclass(frozen=True)
class MonthlySensitivityPolicy:
    # One cent is small relative to quoted power prices while remaining above
    # floating-point noise for deliberately sparse KKT systems.
    perturbation_eur_mwh: float = 0.01
    solution_tolerance_eur_mwh: float = 1e-9
    derivative_tolerance: float = 1e-7
    max_condition_number: float = MAX_SOLVER_CONDITION_NUMBER
    # Horizon-invariant local diagnostic caps. Values 2/sqrt(5)/3 encode the
    # desired geometry of one disjoint residual step with pre-covered hours no
    # greater than residual hours. The annual 2x cap is separate conservative
    # local policy. None is an empirically calibrated production risk appetite.
    max_quote_support_weighted_gain: float = 2.0
    max_delivery_year_weighted_simultaneous_gain: float = 2.0
    max_cascade_bucket_weighted_simultaneous_gain: float = math.sqrt(5.0)
    max_monthly_bounded_quote_shock_gain: float = 3.0
    require_direct_kkt_solve: bool = True


@dataclass(frozen=True)
class MonthlyQuoteSensitivityAudit:
    summary: dict[str, object]
    sensitivities: pd.DataFrame


def audit_monthly_quote_curve_sensitivity(
    *,
    delivery_months: Sequence[str | pd.Period] | pd.PeriodIndex,
    own_quotes: Sequence[MarketQuote],
    config: MonthlyCurveConfig,
    shape_prior: pd.Series | object | None,
    market: str = "CH",
    timezone: str = "Europe/Zurich",
    policy: MonthlySensitivityPolicy | None = None,
) -> MonthlyQuoteSensitivityAudit:
    """Audit the local derivative and hierarchy invariants of one monthly solve."""

    selected_policy = policy or MonthlySensitivityPolicy()
    _validate_policy(selected_policy)
    quotes = tuple(own_quotes)
    if not quotes:
        raise ValueError("monthly sensitivity audit requires own-market quotes")

    baseline_constraints = build_monthly_constraint_system(
        delivery_months,
        quotes,
        timezone=timezone,
        market=market,
        load_type="BASE",
        constraint_tolerance=config.constraint_tolerance,
        quote_conflict_tolerance=config.quote_conflict_tolerance,
    )
    diagnostics = baseline_constraints.quote_diagnostics.copy()
    if diagnostics.empty:
        raise ValueError("monthly sensitivity audit has no quote diagnostics")

    quote_by_product: dict[str, MarketQuote] = {}
    for quote in quotes:
        if quote.load_type.upper() != "BASE":
            continue
        if quote.product in quote_by_product:
            raise ValueError("monthly sensitivity audit requires unique BASE products")
        quote_by_product[quote.product] = quote
    active_products = diagnostics.loc[diagnostics["active"].astype(bool), "product"].astype(str).tolist()
    if not active_products:
        raise ValueError("monthly sensitivity audit has no independent active quotes")
    try:
        independent_quotes = tuple(quote_by_product[product] for product in active_products)
    except KeyError as exc:
        raise ValueError("active quote diagnostic cannot be bound to an input quote") from exc

    baseline = solve_monthly_forward_curve_from_constraints(
        baseline_constraints,
        config=config,
        shape_prior=shape_prior,
    )
    baseline_objective = build_monthly_objective_system(
        baseline_constraints,
        config=config,
        shape_prior=shape_prior,
    )
    baseline_objective_structure = _objective_structure_manifest(
        baseline_objective
    )
    baseline_curve = baseline.monthly_curve.to_numpy(dtype=float)

    independent_constraints = build_monthly_constraint_system(
        delivery_months,
        independent_quotes,
        timezone=timezone,
        market=market,
        load_type="BASE",
        constraint_tolerance=config.constraint_tolerance,
        quote_conflict_tolerance=config.quote_conflict_tolerance,
    )
    _require_same_topology(baseline_constraints, independent_constraints)
    independent = solve_monthly_forward_curve_from_constraints(
        independent_constraints,
        config=config,
        shape_prior=shape_prior,
    )
    independent_objective = build_monthly_objective_system(
        independent_constraints,
        config=config,
        shape_prior=shape_prior,
    )
    independent_objective_structure = _objective_structure_manifest(
        independent_objective
    )
    independent_curve = independent.monthly_curve.to_numpy(dtype=float)

    reversed_constraints = build_monthly_constraint_system(
        delivery_months,
        tuple(reversed(quotes)),
        timezone=timezone,
        market=market,
        load_type="BASE",
        constraint_tolerance=config.constraint_tolerance,
        quote_conflict_tolerance=config.quote_conflict_tolerance,
    )
    _require_same_topology(baseline_constraints, reversed_constraints)
    reversed_result = solve_monthly_forward_curve_from_constraints(
        reversed_constraints,
        config=config,
        shape_prior=shape_prior,
    )
    reversed_objective_structure = _objective_structure_manifest(
        build_monthly_objective_system(
            reversed_constraints,
            config=config,
            shape_prior=shape_prior,
        )
    )
    repeated_result = solve_monthly_forward_curve_from_constraints(
        baseline_constraints,
        config=config,
        shape_prior=shape_prior,
    )
    objective_structure_sha256 = str(
        independent_objective_structure["structure_sha256"]
    )
    observed_objective_structure_hashes = {
        str(baseline_objective_structure["structure_sha256"]),
        objective_structure_sha256,
        str(reversed_objective_structure["structure_sha256"]),
    }
    objective_structure_invariant = len(observed_objective_structure_hashes) == 1

    perturbation = float(selected_policy.perturbation_eur_mwh)
    derivative_columns: list[np.ndarray] = []
    target_derivative_columns: list[np.ndarray] = []
    column_metrics: list[dict[str, object]] = []
    topology_invariant = True
    max_linearity_error = 0.0
    max_constraint_derivative_error = 0.0
    long_rows: list[dict[str, object]] = []
    months = independent_constraints.delivery_grid.months

    for quote_position, quote in enumerate(independent_quotes):
        plus_quotes = list(independent_quotes)
        minus_quotes = list(independent_quotes)
        plus_quotes[quote_position] = replace(quote, price=float(quote.price) + perturbation)
        minus_quotes[quote_position] = replace(quote, price=float(quote.price) - perturbation)
        plus_constraints = build_monthly_constraint_system(
            delivery_months,
            tuple(plus_quotes),
            timezone=timezone,
            market=market,
            load_type="BASE",
            constraint_tolerance=config.constraint_tolerance,
            quote_conflict_tolerance=config.quote_conflict_tolerance,
        )
        minus_constraints = build_monthly_constraint_system(
            delivery_months,
            tuple(minus_quotes),
            timezone=timezone,
            market=market,
            load_type="BASE",
            constraint_tolerance=config.constraint_tolerance,
            quote_conflict_tolerance=config.quote_conflict_tolerance,
        )
        try:
            _require_same_topology(independent_constraints, plus_constraints)
            _require_same_topology(independent_constraints, minus_constraints)
        except ValueError:
            topology_invariant = False
            raise
        for perturbed_constraints in (plus_constraints, minus_constraints):
            perturbed_structure = _objective_structure_manifest(
                build_monthly_objective_system(
                    perturbed_constraints,
                    config=config,
                    shape_prior=shape_prior,
                )
            )
            observed_hash = str(perturbed_structure["structure_sha256"])
            observed_objective_structure_hashes.add(observed_hash)
            objective_structure_invariant = (
                objective_structure_invariant
                and observed_hash == objective_structure_sha256
            )
        plus = solve_monthly_forward_curve_from_constraints(
            plus_constraints,
            config=config,
            shape_prior=shape_prior,
        ).monthly_curve.to_numpy(dtype=float)
        minus = solve_monthly_forward_curve_from_constraints(
            minus_constraints,
            config=config,
            shape_prior=shape_prior,
        ).monthly_curve.to_numpy(dtype=float)
        derivative = (plus - minus) / (2.0 * perturbation)
        target_derivative = (
            plus_constraints.targets - minus_constraints.targets
        ) / (2.0 * perturbation)
        forward = (plus - independent_curve) / perturbation
        backward = (independent_curve - minus) / perturbation
        linearity_error = float(np.max(np.abs(forward - backward)))
        constraint_error = float(
            np.max(
                np.abs(
                    independent_constraints.matrix @ derivative - target_derivative
                )
            )
        )
        max_linearity_error = max(max_linearity_error, linearity_error)
        max_constraint_derivative_error = max(
            max_constraint_derivative_error,
            constraint_error,
        )
        derivative_columns.append(derivative)
        target_derivative_columns.append(target_derivative)
        in_product = months.isin(product_periods(quote.product))
        max_curve_response = float(np.max(np.abs(derivative)))
        max_target_response = float(np.max(np.abs(target_derivative)))
        column_metrics.append(
            {
                "quote_product": quote.product,
                "quote_price_eur_mwh": float(quote.price),
                "max_abs_curve_response": max_curve_response,
                "max_abs_active_target_response": max_target_response,
                "amplification_excess": max_curve_response - max_target_response,
                "linearity_error": linearity_error,
                "constraint_derivative_error": constraint_error,
                "affected_month_count": int(np.sum(np.abs(derivative) > selected_policy.derivative_tolerance)),
                "positive_month_count": int(np.sum(derivative > selected_policy.derivative_tolerance)),
                "negative_month_count": int(np.sum(derivative < -selected_policy.derivative_tolerance)),
            }
        )
        for month_position, month in enumerate(months):
            long_rows.append(
                {
                    "quote_product": quote.product,
                    "delivery_month": str(month),
                    "sensitivity_eur_mwh_per_eur_mwh": float(derivative[month_position]),
                    "abs_sensitivity": float(abs(derivative[month_position])),
                    "inside_quote_delivery": bool(in_product[month_position]),
                    "parent_bucket": str(independent_constraints.month_buckets.iloc[month_position]),
                }
            )

    jacobian = np.column_stack(derivative_columns)
    target_jacobian = np.column_stack(target_derivative_columns)
    quote_operator = _quote_mean_operator(independent_constraints, independent_quotes)
    repricing_jacobian = quote_operator @ jacobian
    identity = np.eye(len(independent_quotes), dtype=float)
    max_repricing_jacobian_error = float(np.max(np.abs(repricing_jacobian - identity)))
    max_amplification_excess = max(
        float(metric["amplification_excess"]) for metric in column_metrics
    )
    response_rank = int(np.linalg.matrix_rank(jacobian))
    expected_rank = len(independent_quotes)
    objective_complexity = _objective_complexity_metrics(
        independent_objective,
        constraints=independent_constraints.matrix,
        repricing_jacobian=repricing_jacobian,
    )
    objective_weight_binding_error = _objective_weight_binding_error(
        independent_objective,
        config=config,
    )

    objective_terms = sorted(
        str(metric).removeprefix("objective_")
        for metric in baseline.diagnostics["metric"].astype(str)
        if str(metric).startswith("objective_")
    )
    unexpected_objective_terms = sorted(set(objective_terms) - _ALLOWED_OBJECTIVE_TERMS)
    inactive = diagnostics.loc[~diagnostics["active"].astype(bool)]
    inactive_reasons = {
        str(reason): int(count)
        for reason, count in inactive["dropped_reason"].astype(str).value_counts().sort_index().items()
    }
    excluded_quote_boundaries = [
        {
            "product": str(row.product),
            "reason": str(row.dropped_reason),
            "classification": _inactive_boundary_classification(
                str(row.dropped_reason)
            ),
        }
        for row in inactive.itertuples(index=False)
    ]

    month_hours = independent_constraints.delivery_grid.month_hours.to_numpy(
        dtype=float
    )
    total_delivery_hours = float(month_hours.sum())
    if (
        not np.isfinite(month_hours).all()
        or np.any(month_hours <= 0.0)
        or not math.isfinite(total_delivery_hours)
        or total_delivery_hours <= 0.0
    ):
        raise ValueError("monthly sensitivity delivery-hour weights are invalid")
    delivery_hour_weights = month_hours / total_delivery_hours
    full_grid_weighted_gain = _weighted_spectral_gain(
        jacobian,
        delivery_hour_weights,
    )
    quote_support_gains: dict[str, float] = {}
    for quote_position, product in enumerate(active_products):
        column = jacobian[:, quote_position]
        support = np.abs(column) > selected_policy.derivative_tolerance
        if not bool(support.any()):
            raise ValueError("active quote has no measured monthly response")
        support_weights = month_hours[support] / float(month_hours[support].sum())
        quote_support_gains[product] = float(
            np.sqrt(np.sum(support_weights * np.square(column[support])))
        )

    delivery_year_gains: dict[str, float] = {}
    for year in sorted(set(int(month.year) for month in months)):
        mask = np.asarray([int(month.year) == year for month in months])
        weights = month_hours[mask] / float(month_hours[mask].sum())
        delivery_year_gains[str(year)] = _weighted_spectral_gain(
            jacobian[mask, :],
            weights,
        )

    bucket_labels = np.asarray(
        [
            str(bucket)
            if not pd.isna(bucket)
            else f"UNREPRESENTED:{int(month.year)}"
            for month, bucket in zip(
                months,
                independent_constraints.month_buckets.to_numpy(),
                strict=True,
            )
        ]
    )
    cascade_bucket_gains: dict[str, float] = {}
    for bucket in sorted(set(bucket_labels)):
        mask = bucket_labels == bucket
        weights = month_hours[mask] / float(month_hours[mask].sum())
        cascade_bucket_gains[bucket] = _weighted_spectral_gain(
            jacobian[mask, :],
            weights,
        )

    max_quote_support_gain = max(quote_support_gains.values())
    max_delivery_year_gain = max(delivery_year_gains.values())
    max_cascade_bucket_gain = max(cascade_bucket_gains.values())
    monthly_bounded_quote_shock_gain = float(
        np.max(np.sum(np.abs(jacobian), axis=1))
    )

    metrics = {
        "active_constraint_rank": int(baseline.kkt["active_constraint_rank"]),
        "condition_number": float(baseline.kkt["condition_number"]),
        "input_quote_count": int(len(diagnostics)),
        "independent_quote_count": int(len(independent_quotes)),
        "inactive_quote_count": int(len(inactive)),
        "delivery_month_count": int(len(months)),
        "nullspace_dimension": int(baseline.kkt["nullspace_dimension"]),
        "response_rank": response_rank,
        "max_abs_quote_to_month_sensitivity": float(np.max(np.abs(jacobian))),
        "max_abs_active_target_sensitivity": float(np.max(np.abs(target_jacobian))),
        "jacobian_spectral_norm": float(np.linalg.norm(jacobian, ord=2)),
        "jacobian_frobenius_norm": float(np.linalg.norm(jacobian, ord="fro")),
        "full_grid_weighted_gain_diagnostic_only": full_grid_weighted_gain,
        "max_quote_support_weighted_gain": max_quote_support_gain,
        "max_delivery_year_weighted_simultaneous_gain": max_delivery_year_gain,
        "max_cascade_bucket_weighted_simultaneous_gain": (
            max_cascade_bucket_gain
        ),
        "monthly_bounded_quote_shock_gain": monthly_bounded_quote_shock_gain,
        "max_central_linearity_error": max_linearity_error,
        "max_active_constraint_derivative_error": max_constraint_derivative_error,
        "max_quote_repricing_jacobian_error": max_repricing_jacobian_error,
        "max_unexplained_amplification": max_amplification_excess,
        "redundant_removal_curve_error": float(np.max(np.abs(baseline_curve - independent_curve))),
        "quote_order_curve_error": float(
            np.max(
                np.abs(
                    baseline_curve
                    - reversed_result.monthly_curve.to_numpy(dtype=float)
                )
            )
        ),
        "repeat_solve_curve_error": float(
            np.max(
                np.abs(
                    baseline_curve
                    - repeated_result.monthly_curve.to_numpy(dtype=float)
                )
            )
        ),
        "baseline_constraint_residual": float(baseline.kkt["max_abs_constraint_residual"]),
        "baseline_stationarity_residual": float(baseline.kkt["stationarity_residual"]),
        "objective_term_count": int(objective_complexity["objective_term_count"]),
        "objective_operator_row_count": int(
            objective_complexity["objective_operator_row_count"]
        ),
        "objective_operator_rank": int(
            objective_complexity["objective_operator_rank"]
        ),
        "feasible_direction_objective_rank": int(
            objective_complexity["feasible_direction_objective_rank"]
        ),
        "reduced_hessian_min_eigenvalue": float(
            objective_complexity["reduced_hessian_min_eigenvalue"]
        ),
        "reduced_hessian_max_eigenvalue": float(
            objective_complexity["reduced_hessian_max_eigenvalue"]
        ),
        "reduced_hessian_condition_number": float(
            objective_complexity["reduced_hessian_condition_number"]
        ),
        "quote_response_effective_degrees_of_freedom": float(
            objective_complexity["quote_response_effective_degrees_of_freedom"]
        ),
        "quote_response_effective_degrees_of_freedom_error": float(
            objective_complexity[
                "quote_response_effective_degrees_of_freedom_error"
            ]
        ),
        "objective_weight_binding_error": objective_weight_binding_error,
    }

    gates = [
        _gate("baseline_constraint_repricing", metrics["baseline_constraint_residual"], config.constraint_tolerance),
        _gate("baseline_stationarity", metrics["baseline_stationarity_residual"], config.stationarity_tolerance),
        _gate("solver_condition_number", metrics["condition_number"], selected_policy.max_condition_number, lower_exclusive=0.0),
        _gate("repeat_solve_determinism", metrics["repeat_solve_curve_error"], selected_policy.solution_tolerance_eur_mwh),
        _gate("quote_input_order_invariance", metrics["quote_order_curve_error"], selected_policy.solution_tolerance_eur_mwh),
        _gate("redundant_parent_removal_invariance", metrics["redundant_removal_curve_error"], selected_policy.solution_tolerance_eur_mwh),
        _gate("central_difference_linearity", metrics["max_central_linearity_error"], selected_policy.derivative_tolerance),
        _gate("active_constraint_derivative", metrics["max_active_constraint_derivative_error"], selected_policy.derivative_tolerance),
        _gate("source_quote_repricing_jacobian", metrics["max_quote_repricing_jacobian_error"], selected_policy.derivative_tolerance),
        _gate("cascade_geometry_amplification", metrics["max_unexplained_amplification"], selected_policy.derivative_tolerance),
        _gate(
            "quote_support_weighted_gain",
            metrics["max_quote_support_weighted_gain"],
            selected_policy.max_quote_support_weighted_gain
            + selected_policy.derivative_tolerance,
        ),
        _gate(
            "delivery_year_weighted_simultaneous_gain",
            metrics["max_delivery_year_weighted_simultaneous_gain"],
            selected_policy.max_delivery_year_weighted_simultaneous_gain
            + selected_policy.derivative_tolerance,
        ),
        _gate(
            "cascade_bucket_weighted_simultaneous_gain",
            metrics["max_cascade_bucket_weighted_simultaneous_gain"],
            selected_policy.max_cascade_bucket_weighted_simultaneous_gain
            + selected_policy.derivative_tolerance,
        ),
        _gate(
            "monthly_bounded_quote_shock_gain",
            metrics["monthly_bounded_quote_shock_gain"],
            selected_policy.max_monthly_bounded_quote_shock_gain
            + selected_policy.derivative_tolerance,
        ),
        {
            "gate_id": "quote_response_full_rank",
            "status": "PASS" if response_rank == expected_rank else "FAIL",
            "value": response_rank,
            "threshold": expected_rank,
        },
        {
            "gate_id": "constraint_topology_invariant",
            "status": "PASS" if topology_invariant else "FAIL",
            "value": topology_invariant,
            "threshold": True,
        },
        {
            "gate_id": "declared_objective_term_allowlist",
            "status": "PASS" if not unexpected_objective_terms else "FAIL",
            "value": objective_terms,
            "threshold": sorted(_ALLOWED_OBJECTIVE_TERMS),
        },
        {
            "gate_id": "objective_operator_and_weight_identity_invariant",
            "status": "PASS" if objective_structure_invariant else "FAIL",
            "value": sorted(observed_objective_structure_hashes),
            "threshold": [objective_structure_sha256],
        },
        _gate(
            "objective_weights_bound_to_solver_config",
            metrics["objective_weight_binding_error"],
            0.0,
        ),
        {
            "gate_id": "feasible_direction_objective_full_rank",
            "status": (
                "PASS"
                if metrics["feasible_direction_objective_rank"]
                == metrics["nullspace_dimension"]
                else "FAIL"
            ),
            "value": metrics["feasible_direction_objective_rank"],
            "threshold": metrics["nullspace_dimension"],
        },
        _gate(
            "reduced_hessian_condition_number",
            metrics["reduced_hessian_condition_number"],
            selected_policy.max_condition_number,
            lower_exclusive=0.0,
        ),
        _gate(
            "quote_response_effective_degrees_of_freedom",
            metrics["quote_response_effective_degrees_of_freedom_error"],
            selected_policy.derivative_tolerance,
        ),
    ]
    if selected_policy.require_direct_kkt_solve:
        direct = not bool(baseline.kkt["ridge_used"]) and not bool(
            baseline.kkt["solved_by_lstsq"]
        )
        gates.append(
            {
                "gate_id": "direct_kkt_no_ridge_or_lstsq",
                "status": "PASS" if direct else "FAIL",
                "value": direct,
                "threshold": True,
            }
        )
    passed = all(gate["status"] == "PASS" for gate in gates)
    sensitivity_frame = pd.DataFrame(long_rows).sort_values(
        ["quote_product", "delivery_month"]
    ).reset_index(drop=True)
    summary: dict[str, object] = {
        "schema_version": SENSITIVITY_SCHEMA_VERSION,
        "status": (
            "PASS_LOCAL_DETERMINISTIC_SENSITIVITY_NOT_PRODUCTION"
            if passed
            else "FAIL_LOCAL_DETERMINISTIC_SENSITIVITY_NO_GO"
        ),
        "production_authorization": False,
        "promotion_gate": False,
        "market": str(market).upper(),
        "policy": {
            "perturbation_eur_mwh": perturbation,
            "solution_tolerance_eur_mwh": selected_policy.solution_tolerance_eur_mwh,
            "derivative_tolerance": selected_policy.derivative_tolerance,
            "max_condition_number": selected_policy.max_condition_number,
            "max_quote_support_weighted_gain": (
                selected_policy.max_quote_support_weighted_gain
            ),
            "max_delivery_year_weighted_simultaneous_gain": (
                selected_policy.max_delivery_year_weighted_simultaneous_gain
            ),
            "max_cascade_bucket_weighted_simultaneous_gain": (
                selected_policy.max_cascade_bucket_weighted_simultaneous_gain
            ),
            "max_monthly_bounded_quote_shock_gain": (
                selected_policy.max_monthly_bounded_quote_shock_gain
            ),
            "stability_threshold_basis": (
                "LOCAL_HORIZON_INVARIANT_DIAGNOSTIC_CAPS_NOT_PRODUCTION_POLICY"
            ),
            "structural_cap_rationale": {
                "residual_geometry": (
                    "ONE_DISJOINT_RESIDUAL_STEP_WITH_PRE_COVERED_HOURS_LE_RESIDUAL_HOURS"
                ),
                "support_bucket_and_monthly_caps": (
                    "ALGEBRAIC_BOUNDS_2_SQRT5_3_UNDER_DECLARED_RESIDUAL_GEOMETRY"
                ),
                "delivery_year_cap": (
                    "SEPARATE_CONSERVATIVE_LOCAL_POLICY_NOT_IMPLIED_BY_RESIDUAL_ALGEBRA"
                ),
            },
            "require_direct_kkt_solve": selected_policy.require_direct_kkt_solve,
        },
        "active_products": active_products,
        "inactive_quote_reasons": inactive_reasons,
        "response_domain": {
            "basis": "ACTIVE_INDEPENDENT_QUOTES_ONLY",
            "hierarchy_selection": "FROZEN_AT_BASELINE",
            "prior_and_objective": "FROZEN_AT_BASELINE",
            "full_feed_end_to_end_claim": False,
            "differentiated_products": active_products,
            "excluded_quote_boundaries": excluded_quote_boundaries,
        },
        "stability_diagnostics": {
            "full_grid_weighted_gain_is_gated": False,
            "quote_support_weighted_gains": quote_support_gains,
            "delivery_year_weighted_simultaneous_gains": delivery_year_gains,
            "cascade_bucket_weighted_simultaneous_gains": (
                cascade_bucket_gains
            ),
        },
        "objective_terms": objective_terms,
        "unexpected_objective_terms": unexpected_objective_terms,
        "objective_structure": independent_objective_structure,
        "objective_structure_invariant": objective_structure_invariant,
        "observed_objective_structure_sha256": sorted(
            observed_objective_structure_hashes
        ),
        "metrics": metrics,
        "column_metrics": column_metrics,
        "gates": gates,
        "constraint_topology_sha256": _constraint_topology_hash(independent_constraints),
        "sensitivity_semantic_sha256": _frame_hash(sensitivity_frame),
        "limitations": [
            "LOCAL_DIAGNOSTIC_ONLY",
            "DOES_NOT_VALIDATE_QUOTE_PROVENANCE_OR_FRESHNESS",
            "DOES_NOT_VALIDATE_PRIOR_ECONOMIC_FIT",
            "REDUNDANT_AND_OUTSIDE_GRID_QUOTES_ARE_DOMAIN_BOUNDARIES_NOT_ZERO_RISK",
            "OBJECTIVE_OPERATORS_AND_WEIGHTS_ARE_HASH_BOUND_FOR_THIS_FIXED_TOPOLOGY_NOT_ECONOMICALLY_CALIBRATED",
            "QUOTE_RESPONSE_EFFECTIVE_DEGREES_OF_FREEDOM_IS_A_LINEAR_CONSTRAINED_RESPONSE_DIAGNOSTIC_NOT_FULL_STATISTICAL_MODEL_EDF",
            "STABILITY_CAPS_ARE_LOCAL_POLICY_ONLY_SUPPORT_BUCKET_MONTHLY_HAVE_DECLARED_CONDITIONAL_ALGEBRA_AND_NONE_ARE_EMPIRICALLY_CALIBRATED_PRODUCTION_LIMITS",
            "PRODUCTION_SCENARIOS_REQUIRE_VOLATILITY_COVARIANCE_AND_LIQUIDITY_NORMALIZATION",
            "STABILITY_GATES_ARE_CONDITIONAL_ON_THE_ACTIVE_BASIS_AND_FROZEN_TOPOLOGY",
            "OBSERVED_SOURCE_HASHES_DO_NOT_ATTEST_EXECUTED_CODE_BYTES",
            "DOES_NOT_AUTHORIZE_PRODUCTION_OR_PROMOTION",
        ],
    }
    return MonthlyQuoteSensitivityAudit(summary=summary, sensitivities=sensitivity_frame)


def _validate_policy(policy: MonthlySensitivityPolicy) -> None:
    for name in (
        "perturbation_eur_mwh",
        "solution_tolerance_eur_mwh",
        "derivative_tolerance",
        "max_condition_number",
        "max_quote_support_weighted_gain",
        "max_delivery_year_weighted_simultaneous_gain",
        "max_cascade_bucket_weighted_simultaneous_gain",
        "max_monthly_bounded_quote_shock_gain",
    ):
        value = float(getattr(policy, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"monthly sensitivity policy {name} must be finite and positive")


def _objective_structure_manifest(
    objective: MonthlyObjectiveSystem,
) -> dict[str, object]:
    terms: list[dict[str, object]] = []
    names: set[str] = set()
    variable_count = int(objective.q_matrix.shape[0])
    if objective.q_matrix.shape != (variable_count, variable_count):
        raise ValueError("monthly objective Hessian dimensions are invalid")
    for term in objective.terms:
        if term.name in names:
            raise ValueError("monthly objective term names are not unique")
        names.add(term.name)
        operator = np.asarray(term.operator, dtype=float)
        target = np.asarray(term.target, dtype=float)
        weight = float(term.weight)
        if (
            operator.ndim != 2
            or operator.shape[1] != variable_count
            or target.shape != (operator.shape[0],)
            or not np.isfinite(operator).all()
            or not np.isfinite(target).all()
            or not math.isfinite(weight)
            or weight <= 0.0
        ):
            raise ValueError("monthly objective term is invalid")
        terms.append(
            {
                "name": term.name,
                "weight": weight,
                "operator_rows": int(operator.shape[0]),
                "operator_columns": int(operator.shape[1]),
                "operator_rank": int(np.linalg.matrix_rank(operator)),
                "operator_sha256": _array_sha256(operator),
                "target_role": _objective_target_role(term.name),
            }
        )
    structure = {
        "variable_count": variable_count,
        "term_count": len(terms),
        "terms": terms,
        "hessian_sha256": _array_sha256(objective.q_matrix),
        "ridge_used": bool(objective.ridge_used),
        "dropped_yoy_rows": int(objective.dropped_yoy_rows),
        "target_values_hash_bound": False,
        "target_values_reason": (
            "PARENT_FLAT_TARGET_IS_AN_AFFINE_FUNCTION_OF_HARD_QUOTES_WHILE_"
            "OPERATORS_AND_WEIGHTS_MUST_REMAIN_FIXED"
        ),
    }
    return {
        **structure,
        "structure_sha256": hashlib.sha256(_canonical_json(structure)).hexdigest(),
    }


def _objective_complexity_metrics(
    objective: MonthlyObjectiveSystem,
    *,
    constraints: np.ndarray,
    repricing_jacobian: np.ndarray,
) -> dict[str, object]:
    a = np.asarray(constraints, dtype=float)
    q_matrix = np.asarray(objective.q_matrix, dtype=float)
    if a.ndim != 2 or q_matrix.shape != (a.shape[1], a.shape[1]):
        raise ValueError("monthly objective complexity dimensions are invalid")
    weighted_operators = [
        math.sqrt(float(term.weight)) * np.asarray(term.operator, dtype=float)
        for term in objective.terms
    ]
    if not weighted_operators:
        raise ValueError("monthly objective complexity has no active term")
    stacked = np.vstack(weighted_operators)
    constraint_rank = int(np.linalg.matrix_rank(a)) if a.size else 0
    variable_count = int(q_matrix.shape[0])
    _, _, vh = np.linalg.svd(a, full_matrices=True)
    nullspace = vh[constraint_rank:, :].T
    nullspace_dimension = int(nullspace.shape[1])
    if nullspace_dimension:
        reduced = nullspace.T @ (0.5 * (q_matrix + q_matrix.T)) @ nullspace
        eigenvalues = np.linalg.eigvalsh(reduced)
        minimum = float(np.min(eigenvalues))
        maximum = float(np.max(eigenvalues))
        reduced_rank = int(np.linalg.matrix_rank(reduced))
        condition = (
            float(maximum / minimum)
            if minimum > 0.0 and math.isfinite(maximum / minimum)
            else 0.0
        )
    else:
        minimum = 1.0
        maximum = 1.0
        reduced_rank = 0
        condition = 1.0
    repricing = np.asarray(repricing_jacobian, dtype=float)
    if (
        repricing.ndim != 2
        or repricing.shape[0] != repricing.shape[1]
        or not np.isfinite(repricing).all()
    ):
        raise ValueError("monthly quote-response EDF matrix is invalid")
    quote_response_edf = float(np.trace(repricing))
    expected_edf = float(repricing.shape[0])
    return {
        "objective_term_count": len(objective.terms),
        "objective_operator_row_count": int(stacked.shape[0]),
        "objective_operator_rank": int(np.linalg.matrix_rank(stacked)),
        "constraint_rank": constraint_rank,
        "variable_count": variable_count,
        "nullspace_dimension": nullspace_dimension,
        "feasible_direction_objective_rank": reduced_rank,
        "reduced_hessian_min_eigenvalue": minimum,
        "reduced_hessian_max_eigenvalue": maximum,
        "reduced_hessian_condition_number": condition,
        "quote_response_effective_degrees_of_freedom": quote_response_edf,
        "expected_quote_response_effective_degrees_of_freedom": expected_edf,
        "quote_response_effective_degrees_of_freedom_error": abs(
            quote_response_edf - expected_edf
        ),
    }


def _objective_weight_binding_error(
    objective: MonthlyObjectiveSystem,
    *,
    config: MonthlyCurveConfig,
) -> float:
    expected = {
        "parent_flat_prior": float(config.lambda_prior),
        "numerical_parent_flat_ridge": 1e-10,
        "smooth_month": float(config.lambda_smooth_month),
        "smooth_yoy_shape": float(config.lambda_smooth_yoy),
        "shape_prior": float(config.lambda_shape),
    }
    errors: list[float] = []
    for term in objective.terms:
        if term.name not in expected:
            raise ValueError("monthly objective term is not weight-governed")
        errors.append(abs(float(term.weight) - expected[term.name]))
    if not errors or not all(math.isfinite(error) for error in errors):
        raise ValueError("monthly objective weight binding is invalid")
    return max(errors)


def _objective_target_role(name: str) -> str:
    if name in {"parent_flat_prior", "numerical_parent_flat_ridge"}:
        return "QUOTE_AFFINE_PARENT_FLAT_TARGET"
    if name in {"smooth_month", "smooth_yoy_shape"}:
        return "ZERO_REGULARIZATION_TARGET"
    if name == "shape_prior":
        return "FROZEN_RECENTERED_SHAPE_PRIOR_TARGET"
    return "UNDECLARED_TARGET_ROLE"


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    header = _canonical_json(
        {
            "dtype": "float64-little-endian",
            "shape": list(array.shape),
        }
    )
    digest = hashlib.sha256(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _inactive_boundary_classification(reason: str) -> str:
    if reason == "redundant_consistent":
        return "HIERARCHY_SELECTION_BOUNDARY_NOT_DIFFERENTIATED"
    if reason == "outside_delivery_grid":
        return "OUTSIDE_DELIVERY_DOMAIN_NOT_DIFFERENTIATED"
    return "INACTIVE_INPUT_NOT_DIFFERENTIATED"


def _weighted_spectral_gain(jacobian: np.ndarray, weights: np.ndarray) -> float:
    selected_jacobian = np.asarray(jacobian, dtype=float)
    selected_weights = np.asarray(weights, dtype=float)
    if selected_jacobian.ndim != 2 or selected_weights.shape != (
        selected_jacobian.shape[0],
    ):
        raise ValueError("monthly sensitivity weighted gain dimensions are invalid")
    if (
        not np.isfinite(selected_jacobian).all()
        or not np.isfinite(selected_weights).all()
        or np.any(selected_weights <= 0.0)
        or not math.isclose(float(selected_weights.sum()), 1.0, abs_tol=1e-12)
    ):
        raise ValueError("monthly sensitivity weighted gain inputs are invalid")
    return float(
        np.linalg.norm(
            np.sqrt(selected_weights)[:, None] * selected_jacobian,
            ord=2,
        )
    )


def _require_same_topology(left: object, right: object) -> None:
    if list(left.names) != list(right.names):
        raise ValueError("monthly quote perturbation changed active constraint names")
    if left.matrix.shape != right.matrix.shape or not np.array_equal(left.matrix, right.matrix):
        raise ValueError("monthly quote perturbation changed active constraint matrix")
    left_buckets = left.month_buckets.astype("string").fillna("<NA>").tolist()
    right_buckets = right.month_buckets.astype("string").fillna("<NA>").tolist()
    if left_buckets != right_buckets:
        raise ValueError("monthly quote perturbation changed cascade bucket topology")


def _quote_mean_operator(constraints: object, quotes: Sequence[MarketQuote]) -> np.ndarray:
    months = constraints.delivery_grid.months
    hours = constraints.delivery_grid.month_hours.to_numpy(dtype=float)
    rows: list[np.ndarray] = []
    for quote in quotes:
        mask = months.isin(product_periods(quote.product))
        if not bool(mask.any()):
            raise ValueError("independent quote lies outside the delivery grid")
        row = np.zeros(len(months), dtype=float)
        positions = np.flatnonzero(mask)
        selected_hours = hours[positions]
        row[positions] = selected_hours / float(selected_hours.sum())
        rows.append(row)
    return np.vstack(rows)


def _gate(
    gate_id: str,
    value: object,
    threshold: float,
    *,
    lower_exclusive: float | None = None,
) -> dict[str, object]:
    numeric = float(value)
    passed = math.isfinite(numeric) and numeric <= float(threshold)
    if lower_exclusive is not None:
        passed = passed and numeric > float(lower_exclusive)
    return {
        "gate_id": gate_id,
        "status": "PASS" if passed else "FAIL",
        "value": numeric,
        "threshold": float(threshold),
    }


def _constraint_topology_hash(constraints: object) -> str:
    payload = {
        "names": list(constraints.names),
        "matrix": np.asarray(constraints.matrix, dtype=float).tolist(),
        "month_buckets": constraints.month_buckets.astype("string").fillna("<NA>").tolist(),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _frame_hash(frame: pd.DataFrame) -> str:
    records = frame.to_dict(orient="records")
    return hashlib.sha256(_canonical_json(records)).hexdigest()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
