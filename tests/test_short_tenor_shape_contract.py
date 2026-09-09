from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.model.shape_constraints import (
    build_base_peak_offpeak_constraint_system,
)
from pfc_shaping.lt.model.short_tenor_shape_contract import (
    SHORT_TENOR_SHAPE_CONTRACT_VERSION,
    project_short_tenor_shape_delta,
)
from pfc_shaping.pipeline import production_phases


NORMALIZATION_ID = "2" * 64
AUDIT_ID = "f" * 64
TZ = "Europe/Zurich"


def _complete_local_range(first_month: str, last_month: str, *, freq: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{first_month}-01", tz=TZ)
    end = pd.Timestamp(f"{last_month}-01", tz=TZ) + pd.offsets.MonthBegin(1)
    return pd.date_range(
        start.tz_convert("UTC"),
        end.tz_convert("UTC"),
        freq=freq,
        inclusive="left",
    )


def _block_signal(index: pd.DatetimeIndex, *, negative: bool = False) -> pd.Series:
    local = index.tz_convert(TZ)
    day_number = local.normalize().factorize()[0]
    is_short_peak = (local.hour >= 8) & (local.hour < 20)
    day_value = ((day_number % 11) - 5).astype(float)
    values = day_value + np.where(is_short_peak, 7.25, -2.75)
    if negative:
        values -= 40.0
    return pd.Series(values, index=index, dtype=float)


def _month_levels(index: pd.DatetimeIndex, *, negative_january: bool = False) -> dict[str, float]:
    keys = tuple(dict.fromkeys(index.tz_convert(TZ).strftime("%Y-%m").tolist()))
    levels = {key: 80.0 + pos for pos, key in enumerate(keys)}
    if negative_january:
        january = next((key for key in keys if key.endswith("-01")), None)
        if january is not None:
            levels[january] = -12.5
    return levels


def _project(
    signal: pd.Series,
    *,
    peak: bool = True,
    valuation: str = "2026-12-31T22:00:00Z",
):
    base = _month_levels(signal.index, negative_january=True)
    return project_short_tenor_shape_delta(
        signal,
        monthly_base_levels=base,
        peak_forward_prices=base if peak else {},
        source_normalization_content_id=NORMALIZATION_ID,
        source_audit_content_id=AUDIT_ID,
        valuation_timestamp_utc=valuation,
    )


def test_hourly_full_year_preserves_monthly_and_peak_offpeak_constraints_across_dst() -> None:
    index = _complete_local_range("2027-01", "2027-12", freq="1h")
    result = _project(_block_signal(index))

    assert len(result.delta) == 8760
    assert result.audit["contract_version"] == SHORT_TENOR_SHAPE_CONTRACT_VERSION
    assert result.audit["cadence"] == "1h"
    assert result.audit["constraint_kinds"] == ["OFFPEAK", "PEAK"]
    assert result.audit["max_constraint_residual_eur_mwh"] <= 1e-9
    assert result.audit["max_monthly_mean_residual_eur_mwh"] <= 1e-9
    assert result.constraint_residuals["abs_error"].max() <= 1e-9
    assert result.monthly_residuals["abs_error"].max() <= 1e-9
    assert "target" not in result.constraint_residuals
    assert result.audit["model_input_authorized"] is False
    assert result.audit["production_authorized"] is False
    assert result.audit["ompex_used"] is False
    assert result.audit["afry_numeric_values_used"] is False


@pytest.mark.parametrize("month", ["2028-03", "2028-10"])
def test_quarter_hour_dst_month_has_no_inferred_intraday_structure(month: str) -> None:
    index = _complete_local_range(month, month, freq="15min")
    result = _project(_block_signal(index), valuation="2027-12-31T22:00:00Z")
    local = result.delta.index.tz_convert(TZ)
    frame = pd.DataFrame(
        {
            "value": result.delta.to_numpy(),
            "day": local.strftime("%Y-%m-%d"),
            "block": np.where((local.hour >= 8) & (local.hour < 20), "PEAK", "OFFPEAK"),
        }
    )

    dispersion = frame.groupby(["day", "block"])["value"].agg(lambda x: x.max() - x.min())
    assert result.audit["cadence"] == "15min"
    assert dispersion.max() <= 1e-10
    assert result.monthly_residuals["abs_error"].max() <= 1e-9


def test_base_only_projection_is_deterministic_idempotent_and_accepts_negative_signal() -> None:
    index = _complete_local_range("2027-01", "2027-01", freq="1h")
    signal = _block_signal(index, negative=True)
    first = _project(signal, peak=False)
    second = _project(signal, peak=False)
    projected_again = _project(first.delta, peak=False)

    np.testing.assert_array_equal(first.delta.to_numpy(), second.delta.to_numpy())
    np.testing.assert_allclose(
        projected_again.delta.to_numpy(),
        first.delta.to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )
    assert first.audit["constraint_kinds"] == ["BASE"]
    assert first.audit["max_abs_raw_delta_eur_mwh"] > 0.0


def test_zero_signal_remains_exactly_zero() -> None:
    index = _complete_local_range("2027-02", "2027-02", freq="1h")
    signal = pd.Series(np.zeros(len(index)), index=index)
    result = _project(signal, valuation="2027-01-31T22:00:00Z")

    np.testing.assert_array_equal(result.delta.to_numpy(), np.zeros(len(index)))
    assert result.audit["l2_retention_ratio"] == 0.0


@pytest.mark.parametrize("freq", ["1h", "15min"])
def test_rejects_unauthorized_within_short_block_shape(freq: str) -> None:
    index = _complete_local_range("2027-03", "2027-03", freq=freq)
    signal = _block_signal(index)
    local = index.tz_convert(TZ)
    signal.loc[(local.day == 8) & (local.hour == 10)] += 0.01

    with pytest.raises(ValueError, match="unauthorized within-day/block structure"):
        _project(signal, valuation="2027-02-28T22:00:00Z")


def test_rejects_partial_month_and_wrong_month_level_coverage() -> None:
    full_index = _complete_local_range("2027-04", "2027-04", freq="1h")
    partial = _block_signal(full_index[1:])
    with pytest.raises(ValueError, match="complete local calendar months"):
        _project(partial, valuation="2027-03-31T21:00:00Z")

    signal = _block_signal(full_index)
    common = dict(
        raw_delta=signal,
        source_normalization_content_id=NORMALIZATION_ID,
        source_audit_content_id=AUDIT_ID,
        valuation_timestamp_utc="2027-03-31T21:00:00Z",
    )
    with pytest.raises(ValueError, match=r"missing=\['2027-04'\]"):
        project_short_tenor_shape_delta(monthly_base_levels={}, **common)
    with pytest.raises(ValueError, match=r"extra=\['2027-05'\]"):
        project_short_tenor_shape_delta(
            monthly_base_levels={"2027-04": 70.0, "2027-05": 71.0},
            **common,
        )


def test_rejects_bad_index_nonfinite_provenance_and_late_valuation() -> None:
    index = _complete_local_range("2027-05", "2027-05", freq="1h")
    signal = _block_signal(index)
    kwargs = dict(
        monthly_base_levels={"2027-05": 70.0},
        source_normalization_content_id=NORMALIZATION_ID,
        source_audit_content_id=AUDIT_ID,
        valuation_timestamp_utc="2027-04-30T21:00:00Z",
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        project_short_tenor_shape_delta(signal.set_axis(index.tz_localize(None)), **kwargs)
    duplicated = signal.iloc[:2].set_axis(pd.DatetimeIndex([index[0], index[0]]))
    with pytest.raises(ValueError, match="duplicate"):
        project_short_tenor_shape_delta(duplicated, **kwargs)
    irregular = signal.drop(signal.index[100])
    with pytest.raises(ValueError, match="regular UTC cadence"):
        project_short_tenor_shape_delta(irregular, **kwargs)
    nonfinite = signal.copy()
    nonfinite.iloc[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        project_short_tenor_shape_delta(nonfinite, **kwargs)
    with pytest.raises(ValueError, match="SHA-256"):
        project_short_tenor_shape_delta(
            signal,
            **{**kwargs, "source_normalization_content_id": "not-a-hash"},
        )
    with pytest.raises(ValueError, match="must precede"):
        project_short_tenor_shape_delta(
            signal,
            **{**kwargs, "valuation_timestamp_utc": str(index[0])},
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        project_short_tenor_shape_delta(
            signal,
            **{**kwargs, "valuation_timestamp_utc": "2027-04-30 21:00:00"},
        )


def test_array_input_requires_matching_explicit_index_and_ch_country() -> None:
    index = _complete_local_range("2027-06", "2027-06", freq="1h")
    values = _block_signal(index).to_numpy()
    levels = {"2027-06": 75.0}
    common = dict(
        monthly_base_levels=levels,
        source_normalization_content_id=NORMALIZATION_ID,
        source_audit_content_id=AUDIT_ID,
        valuation_timestamp_utc="2027-05-31T21:00:00Z",
    )

    with pytest.raises(ValueError, match="index is required"):
        project_short_tenor_shape_delta(values, **common)
    with pytest.raises(ValueError, match="supports CH only"):
        project_short_tenor_shape_delta(values, index=index, country="DE", **common)

    result = project_short_tenor_shape_delta(values, index=index, **common)
    assert result.audit["country"] == "CH"


def test_projection_depends_on_constraint_geometry_not_forward_price_values() -> None:
    index = _complete_local_range("2027-07", "2027-08", freq="1h")
    signal = _block_signal(index)
    common = dict(
        raw_delta=signal,
        source_normalization_content_id=NORMALIZATION_ID,
        source_audit_content_id=AUDIT_ID,
        valuation_timestamp_utc="2027-06-30T21:00:00Z",
    )
    first = project_short_tenor_shape_delta(
        monthly_base_levels={"2027-07": 45.0, "2027-08": 250.0},
        peak_forward_prices={"2027-07": 80.0, "2027-08": -5.0},
        **common,
    )
    second = project_short_tenor_shape_delta(
        monthly_base_levels={"2027-07": -100.0, "2027-08": 1.0},
        peak_forward_prices={"2027-07": 999.0, "2027-08": 500.0},
        **common,
    )

    np.testing.assert_array_equal(first.delta.to_numpy(), second.delta.to_numpy())
    assert "target" not in first.constraint_residuals.columns
    assert not any("forward_price" in key for key in first.audit)


def test_additive_delta_preserves_an_existing_solver_admissible_curve() -> None:
    index = _complete_local_range("2028-03", "2028-04", freq="15min")
    base = {"2028-03": 72.0, "2028-04": -8.0}
    peak = {"2028-03": 91.0, "2028-04": 15.0}
    result = project_short_tenor_shape_delta(
        _block_signal(index),
        monthly_base_levels=base,
        peak_forward_prices=peak,
        source_normalization_content_id=NORMALIZATION_ID,
        source_audit_content_id=AUDIT_ID,
        valuation_timestamp_utc="2028-02-29T22:00:00Z",
    )
    system = build_base_peak_offpeak_constraint_system(
        index,
        base,
        peak,
        country="CH",
    )
    matrix = system.matrix
    baseline = np.linalg.lstsq(matrix, system.targets, rcond=None)[0]

    np.testing.assert_allclose(matrix @ baseline, system.targets, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        matrix @ (baseline + result.delta.to_numpy()),
        system.targets,
        rtol=0.0,
        atol=1e-9,
    )


def test_contract_remains_dormant_and_unwired_from_lt_orchestration() -> None:
    source = inspect.getsource(production_phases)

    assert "short_tenor_shape_contract" not in source
    assert "project_short_tenor_shape_delta" not in source
    assert "short_tenor_shape_delta" not in source


def test_preserves_a_coarser_accepted_peak_constraint() -> None:
    index = _complete_local_range("2027-07", "2027-09", freq="1h")
    signal = _block_signal(index)
    result = project_short_tenor_shape_delta(
        signal,
        monthly_base_levels=_month_levels(index),
        peak_forward_prices={"2027-Q3": 92.0},
        source_normalization_content_id=NORMALIZATION_ID,
        source_audit_content_id=AUDIT_ID,
        valuation_timestamp_utc="2027-06-30T21:00:00Z",
    )

    assert set(result.constraint_residuals["kind"]) == {"BASE", "PEAK"}
    assert result.constraint_residuals["abs_error"].max() <= 1e-9
