from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.model.short_tenor_combination_contract import (
    ALLOWED_BASELINE_FAMILIES,
    ALLOWED_CANDIDATE_FAMILIES,
    COMPONENT_AUDIT_COLUMNS,
    SELECTED_CONTRAST_BUNDLE_CONTENT_ID,
    SELECTED_CONTRAST_BUNDLE_MANIFEST_SHA256,
    SHORT_TENOR_COMBINATION_CONTRACT_VERSION,
    combine_short_tenor_shape_components,
    validate_short_tenor_combination_policy,
)
from pfc_shaping.lt.model.short_tenor_shape_contract import (
    project_short_tenor_shape_delta,
)
from pfc_shaping.pipeline import production_phases

TZ = "Europe/Zurich"
NORMALIZATION_ID = "2" * 64
AUDIT_ID = "b" * 64
CONTRAST_BUNDLE_ID = "c" * 64
POLICY_ID = "d" * 64
TRAINING_ID = "e" * 64
SELECTION_ID = "f" * 64
POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "EEX-CH-SHORT-TENOR-COMBINATION-CONTRACT-V1.json"
)


def _policy_document() -> dict[str, object]:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _complete_local_month(month: str, *, freq: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{month}-01", tz=TZ)
    end = start + pd.offsets.MonthBegin(1)
    return pd.date_range(
        start.tz_convert("UTC"),
        end.tz_convert("UTC"),
        freq=freq,
        inclusive="left",
    )


def _block_signal(
    index: pd.DatetimeIndex,
    *,
    day_scale: float,
    peak_shift: float,
) -> pd.Series:
    local = index.tz_convert(TZ)
    day_number = local.normalize().factorize()[0].astype(float)
    day_effect = day_scale * ((day_number % 7.0) - 3.0)
    short_peak = (local.hour >= 8) & (local.hour < 20)
    values = day_effect + np.where(short_peak, peak_shift, -peak_shift / 3.0)
    return pd.Series(values, index=index, dtype=float)


def _components(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    return {
        "calendar_day": _block_signal(index, day_scale=0.5, peak_shift=4.0),
        "weekend_block": _block_signal(index, day_scale=-0.25, peak_shift=2.5),
    }


def _combine(
    components: dict[str, pd.Series],
    *,
    coefficients: dict[str, float] | None = None,
    base_level: float = 80.0,
    peak_level: float = 95.0,
    **overrides,
):
    months = tuple(
        dict.fromkeys(
            next(iter(components.values())).index.tz_convert(TZ).strftime("%Y-%m")
        )
    )
    kwargs = {
        "components": components,
        "coefficients": coefficients or {
            "calendar_day": 0.75,
            "weekend_block": -0.4,
        },
        "coefficient_abs_cap": 2.0,
        "per_component_contribution_abs_cap_eur_mwh": 100.0,
        "combined_signal_abs_cap_eur_mwh": 100.0,
        "monthly_base_levels": {month: base_level for month in months},
        "peak_forward_prices": {month: peak_level for month in months},
        "source_normalization_content_id": NORMALIZATION_ID,
        "source_audit_content_id": AUDIT_ID,
        "source_contrast_bundle_content_id": CONTRAST_BUNDLE_ID,
        "combination_policy_content_id": POLICY_ID,
        "training_receipt_content_id": TRAINING_ID,
        "selection_receipt_content_id": SELECTION_ID,
        "valuation_timestamp_utc": (
            next(iter(components.values())).index[0] - pd.Timedelta(hours=1)
        ),
    }
    kwargs.update(overrides)
    return combine_short_tenor_shape_components(**kwargs)


def test_outcome_blind_policy_is_hash_closed_but_not_executable() -> None:
    policy = _policy_document()
    result = validate_short_tenor_combination_policy(policy)

    assert result["status"] == "PASS_LOCAL_STRUCTURE_ONLY_NO_EMPIRICAL_EXECUTION"
    assert result["selected_contrast_bundle_content_id"] == (
        SELECTED_CONTRAST_BUNDLE_CONTENT_ID
    )
    assert policy["selected_boundary_evidence"][  # type: ignore[index]
        "contrast_construction_manifest_sha256"
    ] == SELECTED_CONTRAST_BUNDLE_MANIFEST_SHA256
    assert result["candidate_families"] == list(ALLOWED_CANDIDATE_FAMILIES)
    assert result["baseline_families"] == list(ALLOWED_BASELINE_FAMILIES)
    assert result["blocker_count"] == 8
    assert result["numeric_decisions_frozen"] is False
    assert result["outcomes_opened"] is False
    assert result["execution_authorized"] is False
    assert result["model_input_authorized"] is False
    assert result["production_authorized"] is False
    assert len(str(result["policy_content_id"])) == 64


def test_policy_hash_is_independent_of_json_key_order() -> None:
    policy = _policy_document()
    reversed_policy = dict(reversed(list(policy.items())))

    first = validate_short_tenor_combination_policy(policy)
    second = validate_short_tenor_combination_policy(reversed_policy)

    assert first["policy_content_id"] == second["policy_content_id"]


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        (
            "selected_boundary_evidence",
            "contrast_construction_content_id",
            "ae7b962ca5f85ff75223ad45fd578c29b49a1d66dfbdd418b941c337dda38147",
            "selected contrast bundle",
        ),
        (
            "selected_boundary_evidence",
            "contrast_construction_decision",
            "D-20260805-223",
            "selected contrast decision",
        ),
        (
            "mathematical_operation",
            "all_provenance_content_ids_pairwise_distinct",
            False,
            "mathematical operation",
        ),
        (
            "future_empirical_selection_policy",
            "minimum_inner_origins",
            12,
            "unfrozen minimum_inner_origins",
        ),
        (
            "future_empirical_selection_policy",
            "embargo_months",
            1,
            "unfrozen embargo_months",
        ),
        (
            "numeric_decisions_pending_external_freeze",
            "coefficient_absolute_cap",
            2.0,
            "pending numeric field coefficient_absolute_cap",
        ),
        (
            "numeric_decisions_pending_external_freeze",
            "superiority_margin_eur_mwh",
            0.01,
            "pending numeric field superiority_margin_eur_mwh",
        ),
        (
            "outcome_blind_firewall",
            "ompex_allowed_for_training_tuning_selection_caps_or_margins",
            True,
            "outcome firewall",
        ),
        (
            "outcome_blind_firewall",
            "future_holdout_consumed",
            True,
            "outcome firewall",
        ),
        (
            "authorities",
            "model_input_authorized",
            True,
            "authority model_input_authorized",
        ),
        (
            "authorities",
            "production_authorized",
            True,
            "authority production_authorized",
        ),
    ],
)
def test_policy_rejects_premature_numbers_outcomes_or_authority(
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    policy = copy.deepcopy(_policy_document())
    policy[section][field] = value  # type: ignore[index]

    with pytest.raises(ValueError, match=message):
        validate_short_tenor_combination_policy(policy)


def test_policy_rejects_candidate_family_or_metric_drift() -> None:
    wrong_family = copy.deepcopy(_policy_document())
    wrong_family["future_empirical_selection_policy"][  # type: ignore[index]
        "candidate_families"
    ].append("UNPREREGISTERED_CHALLENGER")
    with pytest.raises(ValueError, match="candidate families"):
        validate_short_tenor_combination_policy(wrong_family)

    wrong_metric = copy.deepcopy(_policy_document())
    wrong_metric["future_empirical_selection_policy"][  # type: ignore[index]
        "primary_loss"
    ] = "OMPEX_DISTANCE"
    with pytest.raises(ValueError, match="primary loss"):
        validate_short_tenor_combination_policy(wrong_metric)


def test_explicit_hourly_combination_is_linear_and_solver_neutral() -> None:
    index = _complete_local_month("2027-03", freq="1h")
    components = _components(index)
    coefficients = {"calendar_day": 0.75, "weekend_block": -0.4}
    result = _combine(components, coefficients=coefficients)
    individual = {
        key: project_short_tenor_shape_delta(
            signal,
            monthly_base_levels={"2027-03": 80.0},
            peak_forward_prices={"2027-03": 95.0},
            source_normalization_content_id=NORMALIZATION_ID,
            source_audit_content_id=AUDIT_ID,
            valuation_timestamp_utc="2027-02-28T22:00:00Z",
        ).delta
        for key, signal in components.items()
    }
    expected = sum(coefficients[key] * individual[key] for key in coefficients)

    np.testing.assert_allclose(
        result.pre_final_projection_delta,
        expected,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(result.delta, expected, rtol=0.0, atol=1e-9)
    assert result.audit["contract_version"] == SHORT_TENOR_COMBINATION_CONTRACT_VERSION
    assert result.audit["max_abs_linearity_residual_eur_mwh"] <= 1e-9
    assert result.audit["max_abs_combined_final_projection_eur_mwh"] <= 100.0
    assert result.audit["max_constraint_residual_eur_mwh"] <= 1e-9
    assert result.audit["max_monthly_mean_residual_eur_mwh"] <= 1e-9
    assert tuple(result.component_audit.columns) == COMPONENT_AUDIT_COLUMNS


@pytest.mark.parametrize("month", ["2028-03", "2028-10"])
def test_quarter_hour_dst_combination_preserves_native_block_grain(month: str) -> None:
    index = _complete_local_month(month, freq="15min")
    result = _combine(_components(index))
    local = result.delta.index.tz_convert(TZ)
    frame = pd.DataFrame(
        {
            "value": result.delta.to_numpy(),
            "day": local.strftime("%Y-%m-%d"),
            "block": np.where(
                (local.hour >= 8) & (local.hour < 20),
                "PEAK",
                "OFFPEAK",
            ),
        }
    )
    dispersion = frame.groupby(["day", "block"])["value"].agg(
        lambda values: values.max() - values.min()
    )

    assert dispersion.max() <= 1e-10
    assert result.projection.audit["cadence"] == "15min"


def test_combination_is_invariant_to_forward_numeric_levels() -> None:
    index = _complete_local_month("2027-06", freq="1h")
    components = _components(index)
    positive = _combine(components, base_level=75.0, peak_level=90.0)
    extreme = _combine(components, base_level=-500.0, peak_level=1_000.0)

    np.testing.assert_allclose(positive.delta, extreme.delta, rtol=0.0, atol=1e-12)
    assert positive.audit["coefficient_map_sha256"] == extreme.audit[
        "coefficient_map_sha256"
    ]


def test_component_order_does_not_change_bytes_or_audit_order() -> None:
    index = _complete_local_month("2027-07", freq="1h")
    components = _components(index)
    reversed_components = dict(reversed(list(components.items())))
    first = _combine(components)
    second = _combine(reversed_components)

    np.testing.assert_array_equal(first.delta.to_numpy(), second.delta.to_numpy())
    pd.testing.assert_frame_equal(first.component_audit, second.component_audit)
    assert first.audit == second.audit
    assert first.audit["component_keys"] == ["calendar_day", "weekend_block"]


@pytest.mark.parametrize(
    "coefficients",
    [
        {"calendar_day": 0.5},
        {"calendar_day": 0.5, "weekend_block": 0.5, "extra": 0.1},
    ],
)
def test_rejects_missing_or_extra_coefficient_keys(coefficients) -> None:
    index = _complete_local_month("2027-08", freq="1h")
    with pytest.raises(ValueError, match="exactly match component keys"):
        _combine(_components(index), coefficients=coefficients)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, "bad", True])
def test_rejects_nonfinite_or_invalid_coefficients(bad_value) -> None:
    index = _complete_local_month("2027-09", freq="1h")
    coefficients = {"calendar_day": bad_value, "weekend_block": 0.5}
    with pytest.raises(ValueError, match="coefficient"):
        _combine(_components(index), coefficients=coefficients)


def test_rejects_coefficient_contribution_and_combined_cap_breaches() -> None:
    index = _complete_local_month("2027-11", freq="1h")
    components = _components(index)
    with pytest.raises(ValueError, match="coefficient breaches"):
        _combine(
            components,
            coefficients={"calendar_day": 2.1, "weekend_block": 0.0},
        )
    with pytest.raises(ValueError, match="raw component contribution"):
        _combine(
            components,
            coefficients={"calendar_day": 1.0, "weekend_block": 0.0},
            per_component_contribution_abs_cap_eur_mwh=0.1,
        )
    with pytest.raises(ValueError, match="combined short-tenor signal"):
        _combine(
            components,
            coefficients={"calendar_day": 1.0, "weekend_block": 1.0},
            combined_signal_abs_cap_eur_mwh=0.1,
        )


@pytest.mark.parametrize(
    "field",
    [
        "coefficient_abs_cap",
        "per_component_contribution_abs_cap_eur_mwh",
        "combined_signal_abs_cap_eur_mwh",
    ],
)
def test_rejects_nonpositive_or_nonfinite_caps(field: str) -> None:
    index = _complete_local_month("2027-12", freq="1h")
    for value in (0.0, -1.0, np.nan, np.inf, True):
        with pytest.raises(ValueError, match="positive and finite"):
            _combine(_components(index), **{field: value})


def test_rejects_partial_month_misaligned_index_and_intrablock_fabrication() -> None:
    index = _complete_local_month("2028-01", freq="1h")
    components = _components(index)
    partial = {key: value.iloc[1:] for key, value in components.items()}
    with pytest.raises(ValueError, match="complete local calendar months"):
        _combine(partial)

    misaligned = components.copy()
    misaligned["weekend_block"] = misaligned["weekend_block"].copy()
    misaligned["weekend_block"].index = misaligned["weekend_block"].index.shift(
        1,
        freq="h",
    )
    with pytest.raises(ValueError, match="same delivery index"):
        _combine(misaligned)

    fabricated = components.copy()
    fabricated["calendar_day"] = fabricated["calendar_day"].copy()
    fabricated["calendar_day"].iloc[10] += 0.01
    with pytest.raises(ValueError, match="unauthorized within-day/block structure"):
        _combine(fabricated)


def test_rejects_bad_provenance_country_and_late_valuation() -> None:
    index = _complete_local_month("2028-02", freq="1h")
    components = _components(index)
    with pytest.raises(ValueError, match="SHA-256"):
        _combine(components, source_contrast_bundle_content_id="bad")
    with pytest.raises(ValueError, match="pairwise distinct"):
        _combine(components, training_receipt_content_id=SELECTION_ID)
    with pytest.raises(ValueError, match="supports CH only"):
        _combine(components, country="DE")
    with pytest.raises(ValueError, match="must precede"):
        _combine(components, valuation_timestamp_utc=index[0])


def test_audit_has_no_forward_targets_and_grants_no_authority() -> None:
    index = _complete_local_month("2028-04", freq="1h")
    result = _combine(
        _components(index),
        coefficients={"calendar_day": 0.0, "weekend_block": -0.25},
    )
    audit_text = repr(result.audit).lower()

    assert "target" not in result.projection.constraint_residuals.columns
    assert "monthly_base_levels" not in audit_text
    assert "peak_forward_prices" not in audit_text
    assert result.audit["coefficients_selected_here"] is False
    assert result.audit["caps_selected_here"] is False
    assert result.audit["silent_clipping_used"] is False
    assert result.audit["training_receipt_authenticated"] is False
    assert result.audit["selection_receipt_authenticated"] is False
    assert result.audit["point_in_time_availability_proven"] is False
    assert result.audit["model_input_authorized"] is False
    assert result.audit["candidate_assembly_authorized"] is False
    assert result.audit["production_authorized"] is False
    assert result.audit["ompex_used"] is False
    assert result.audit["afry_numeric_values_used"] is False
    assert result.audit["t057_used"] is False


def test_combination_boundary_remains_absent_from_lt_orchestration() -> None:
    source = inspect.getsource(production_phases)
    assert "short_tenor_combination_contract" not in source
    assert "combine_short_tenor_shape_components" not in source
