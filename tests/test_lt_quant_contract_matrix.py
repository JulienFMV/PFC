from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.model.shape_constraints import (
    ConstraintRow,
    ConstraintSystem,
    build_base_constraint_system,
    build_base_peak_offpeak_constraint_system,
    eex_peak_mask,
)


def test_month_base_row_is_duration_weighted_average():
    idx = pd.date_range("2030-01-01", "2030-01-31 23:00", freq="h", tz="UTC")
    system = build_base_constraint_system(idx, {"2030-01": 100.0})

    assert system.names == ["BASE:2030-01"]
    assert system.matrix.shape == (1, len(idx))
    assert system.matrix.sum() == pytest.approx(1.0)
    assert system.targets[0] == 100.0


def test_consistent_redundant_constraints_pass_feasibility():
    rows = (
        ConstraintRow("a", 10.0, np.array([0.5, 0.5]), "BASE"),
        ConstraintRow("b", 10.0, np.array([0.5, 0.5]), "BASE"),
    )
    report = ConstraintSystem(rows, n_variables=2).feasibility_report()

    assert report.feasible
    assert report.rank_a == 1
    assert report.rank_augmented == 1


def test_constraint_system_rejects_wrong_row_dimension():
    rows = (ConstraintRow("bad", 10.0, np.array([1.0, 0.0, 0.0]), "BASE"),)

    with pytest.raises(ValueError, match="weights length"):
        ConstraintSystem(rows, n_variables=2)


def test_inconsistent_overlapping_constraints_fail_before_qp():
    rows = (
        ConstraintRow("a", 10.0, np.array([0.5, 0.5]), "BASE"),
        ConstraintRow("b", 11.0, np.array([0.5, 0.5]), "BASE"),
    )
    report = ConstraintSystem(rows, n_variables=2).feasibility_report()

    assert not report.feasible
    assert report.infeasibility_inf > 0.0


def test_delivery_index_with_gap_fails_instead_of_inflating_hours():
    idx = pd.DatetimeIndex(
        ["2030-01-01T00:00:00Z", "2030-01-01T01:00:00Z", "2030-01-01T03:00:00Z"]
    )

    with pytest.raises(ValueError, match="regular UTC cadence"):
        build_base_constraint_system(idx, {"2030-01": 100.0})


def test_uniform_two_hour_delivery_index_fails_explicit_cadence_gate():
    idx = pd.date_range("2030-01-01", periods=12, freq="2h", tz="UTC")

    with pytest.raises(ValueError, match="hourly or 15-minute"):
        build_base_constraint_system(idx, {"2030-01": 100.0})


def test_unknown_base_country_fails_strictly():
    idx = pd.date_range("2030-01-01", periods=24, freq="h", tz="UTC")

    with pytest.raises(ValueError, match="unsupported BASE calendar country"):
        build_base_constraint_system(idx, {"2030-01": 100.0}, country="XX")


def test_base_peak_are_transformed_to_disjoint_peak_offpeak_rows():
    idx = pd.date_range("2030-01-01", "2030-01-07 23:00", freq="h", tz="UTC")
    system = build_base_peak_offpeak_constraint_system(
        idx,
        {"2030-01": 80.0},
        {"2030-01": 100.0},
        country="CH",
    )

    assert {row.kind for row in system.rows} == {"PEAK", "OFFPEAK"}
    peak = next(row for row in system.rows if row.kind == "PEAK")
    offpeak = next(row for row in system.rows if row.kind == "OFFPEAK")
    assert np.dot(peak.weights, offpeak.weights) == pytest.approx(0.0)
    assert peak.target == 100.0


def test_peak_quote_is_represented_even_when_base_bucket_name_differs():
    idx = pd.date_range("2030-01-01", "2030-01-31 23:00", freq="h", tz="UTC")
    system = build_base_peak_offpeak_constraint_system(
        idx,
        {"2030": 80.0},
        {"2030-01": 100.0},
        country="CH",
    )

    assert "PEAK:2030-01" in system.names
    peak = next(row for row in system.rows if row.name == "PEAK:2030-01")
    assert "2030-01" in peak.metadata["source_quotes"]


def test_ch_august_1_holiday_is_offpeak():
    idx = pd.date_range("2030-08-01 00:00", "2030-08-01 23:00", freq="h", tz="Europe/Zurich").tz_convert("UTC")
    peak = eex_peak_mask(idx, country="CH")

    assert not bool(peak.any())
