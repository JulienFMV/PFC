from __future__ import annotations

import pytest
import pandas as pd

from pfc_shaping.calibration.eex_contract_selection import (
    calibration_buckets,
    finest_product_for_timestamp,
)


def test_finest_product_for_timestamp_prefers_month_then_quarter_then_year():
    ts = pd.Timestamp("2028-01-15 12:00", tz="Europe/Zurich")

    assert finest_product_for_timestamp(ts, {"2028": 80.0}) == "2028"
    assert finest_product_for_timestamp(ts, {"2028": 80.0, "2028-Q1": 100.0}) == "2028-Q1"
    assert (
        finest_product_for_timestamp(ts, {"2028": 80.0, "2028-Q1": 100.0, "2028-01": 120.0})
        == "2028-01"
    )


def test_calibration_buckets_builds_quarter_and_year_residuals():
    ts = pd.Series(pd.date_range("2028-01-01", "2028-12-31 23:00", freq="h", tz="Europe/Zurich"))

    buckets, targets = calibration_buckets(ts, {"2028": 79.98, "2028-Q1": 110.76})

    assert set(buckets.dropna().unique()) == {"2028-Q1", "2028-RESIDUAL"}
    assert targets["2028-Q1"] == 110.76
    assert targets["2028-RESIDUAL"] == pytest.approx(69.800824, abs=1e-6)
    weighted = (
        targets["2028-Q1"] * int((buckets == "2028-Q1").sum())
        + targets["2028-RESIDUAL"] * int((buckets == "2028-RESIDUAL").sum())
    ) / len(buckets)
    assert weighted == pytest.approx(79.98, abs=1e-9)


def test_calibration_buckets_builds_month_and_quarter_residuals():
    ts = pd.Series(pd.date_range("2028-01-01", "2028-03-31 23:00", freq="h", tz="Europe/Zurich"))

    buckets, targets = calibration_buckets(ts, {"2028-Q1": 100.0, "2028-01": 120.0})

    assert set(buckets.dropna().unique()) == {"2028-01", "2028-Q1-RESIDUAL"}
    assert targets["2028-01"] == 120.0
    weighted = (
        targets["2028-01"] * int((buckets == "2028-01").sum())
        + targets["2028-Q1-RESIDUAL"] * int((buckets == "2028-Q1-RESIDUAL").sum())
    ) / len(buckets)
    assert weighted == pytest.approx(100.0, abs=1e-9)
