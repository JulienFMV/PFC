from __future__ import annotations

import pandas as pd
import pytest

from pfc_shaping.calibration.monthly_curve_sensitivity import (
    MonthlySensitivityPolicy,
    audit_monthly_quote_curve_sensitivity,
)
from pfc_shaping.calibration.monthly_forward_curve import MarketQuote, MonthlyCurveConfig


def _quote(product: str, price: float) -> MarketQuote:
    return MarketQuote(
        market="CH",
        product=product,
        load_type="BASE",
        price=price,
        quote_id=f"quote-{product}",
    )


def test_quote_to_curve_jacobian_reprices_independent_products_exactly() -> None:
    months = pd.period_range("2028-01", "2028-12", freq="M")
    audit = audit_monthly_quote_curve_sensitivity(
        delivery_months=months,
        own_quotes=(_quote("2028", 80.40), _quote("2028-Q1", 109.97)),
        config=MonthlyCurveConfig(
            lambda_smooth_month=1.0,
            lambda_smooth_yoy=0.0,
            lambda_shape=0.0,
        ),
        shape_prior=None,
    )

    assert audit.summary["status"] == "PASS_LOCAL_DETERMINISTIC_SENSITIVITY_NOT_PRODUCTION"
    assert audit.summary["production_authorization"] is False
    assert audit.summary["promotion_gate"] is False
    metrics = audit.summary["metrics"]
    assert metrics["independent_quote_count"] == 2
    assert metrics["response_rank"] == 2
    assert metrics["max_quote_repricing_jacobian_error"] <= 1e-7
    assert metrics["max_active_constraint_derivative_error"] <= 1e-7
    assert metrics["max_unexplained_amplification"] <= 1e-7
    assert metrics["max_quote_support_weighted_gain"] <= 2.0 + 1e-7
    assert metrics["max_delivery_year_weighted_simultaneous_gain"] <= 2.0 + 1e-7
    assert metrics["max_cascade_bucket_weighted_simultaneous_gain"] <= 5**0.5 + 1e-7
    assert metrics["monthly_bounded_quote_shock_gain"] <= 3.0 + 1e-7
    stability_gates = {
        gate["gate_id"]: gate["status"]
        for gate in audit.summary["gates"]
        if gate["gate_id"]
        in {
            "quote_support_weighted_gain",
            "delivery_year_weighted_simultaneous_gain",
            "cascade_bucket_weighted_simultaneous_gain",
            "monthly_bounded_quote_shock_gain",
        }
    }
    assert stability_gates == {
        "quote_support_weighted_gain": "PASS",
        "delivery_year_weighted_simultaneous_gain": "PASS",
        "cascade_bucket_weighted_simultaneous_gain": "PASS",
        "monthly_bounded_quote_shock_gain": "PASS",
    }
    assert set(audit.sensitivities["quote_product"]) == {"2028", "2028-Q1"}


def test_consistent_redundant_calendar_is_cascade_invariant() -> None:
    months = pd.period_range("2028-01", "2028-12", freq="M")
    quotes = tuple(
        _quote(product, 80.0)
        for product in ("2028", "2028-Q1", "2028-Q2", "2028-Q3", "2028-Q4")
    )

    audit = audit_monthly_quote_curve_sensitivity(
        delivery_months=months,
        own_quotes=quotes,
        config=MonthlyCurveConfig(lambda_smooth_yoy=0.0, lambda_shape=0.0),
        shape_prior=None,
    )

    metrics = audit.summary["metrics"]
    assert metrics["input_quote_count"] == 5
    assert metrics["independent_quote_count"] == 4
    assert audit.summary["inactive_quote_reasons"] == {"redundant_consistent": 1}
    assert audit.summary["response_domain"]["excluded_quote_boundaries"] == [
        {
            "product": "2028",
            "reason": "redundant_consistent",
            "classification": "HIERARCHY_SELECTION_BOUNDARY_NOT_DIFFERENTIATED",
        }
    ]
    assert metrics["redundant_removal_curve_error"] <= 1e-9
    assert metrics["quote_order_curve_error"] <= 1e-9
    assert metrics["repeat_solve_curve_error"] <= 1e-9
    assert all(gate["status"] == "PASS" for gate in audit.summary["gates"])


def test_outside_grid_quote_is_disclosed_but_not_a_response_dimension() -> None:
    months = pd.period_range("2028-01", "2028-12", freq="M")
    audit = audit_monthly_quote_curve_sensitivity(
        delivery_months=months,
        own_quotes=(_quote("2028", 80.0), _quote("2029", 75.0)),
        config=MonthlyCurveConfig(lambda_smooth_yoy=0.0, lambda_shape=0.0),
        shape_prior=None,
    )

    assert audit.summary["inactive_quote_reasons"] == {"outside_delivery_grid": 1}
    assert audit.summary["metrics"]["independent_quote_count"] == 1
    assert audit.summary["response_domain"]["full_feed_end_to_end_claim"] is False
    assert audit.summary["response_domain"]["excluded_quote_boundaries"] == [
        {
            "product": "2029",
            "reason": "outside_delivery_grid",
            "classification": "OUTSIDE_DELIVERY_DOMAIN_NOT_DIFFERENTIATED",
        }
    ]
    assert set(audit.sensitivities["quote_product"]) == {"2028"}


def test_support_weighted_shock_limit_fails_closed() -> None:
    audit = audit_monthly_quote_curve_sensitivity(
        delivery_months=pd.period_range("2028-01", "2028-12", freq="M"),
        own_quotes=(_quote("2028", 80.0),),
        config=MonthlyCurveConfig(lambda_smooth_yoy=0.0, lambda_shape=0.0),
        shape_prior=None,
        policy=MonthlySensitivityPolicy(
            max_quote_support_weighted_gain=0.5
        ),
    )

    gate = next(
        item
        for item in audit.summary["gates"]
        if item["gate_id"] == "quote_support_weighted_gain"
    )
    assert gate["status"] == "FAIL"
    assert audit.summary["status"] == "FAIL_LOCAL_DETERMINISTIC_SENSITIVITY_NO_GO"


def test_narrow_residual_leverage_cannot_hide_in_a_long_delivery_grid() -> None:
    months = pd.period_range("2028-01", "2032-12", freq="M")
    quotes = tuple(
        [_quote(f"2028-{month:02d}", 80.0) for month in range(1, 12)]
        + [_quote(str(year), 80.0) for year in range(2028, 2033)]
    )

    audit = audit_monthly_quote_curve_sensitivity(
        delivery_months=months,
        own_quotes=quotes,
        config=MonthlyCurveConfig(lambda_smooth_yoy=0.0, lambda_shape=0.0),
        shape_prior=None,
    )

    metrics = audit.summary["metrics"]
    assert metrics["full_grid_weighted_gain_diagnostic_only"] < 2.0
    assert metrics["max_abs_quote_to_month_sensitivity"] > 10.0
    assert metrics["monthly_bounded_quote_shock_gain"] > 3.0
    assert audit.summary["status"] == "FAIL_LOCAL_DETERMINISTIC_SENSITIVITY_NO_GO"
    assert next(
        gate["status"]
        for gate in audit.summary["gates"]
        if gate["gate_id"] == "monthly_bounded_quote_shock_gain"
    ) == "FAIL"


def test_sensitivity_policy_rejects_nonpositive_perturbation() -> None:
    with pytest.raises(ValueError, match="perturbation_eur_mwh"):
        audit_monthly_quote_curve_sensitivity(
            delivery_months=pd.period_range("2028-01", "2028-12", freq="M"),
            own_quotes=(_quote("2028", 80.0),),
            config=MonthlyCurveConfig(),
            shape_prior=None,
            policy=MonthlySensitivityPolicy(perturbation_eur_mwh=0.0),
        )
