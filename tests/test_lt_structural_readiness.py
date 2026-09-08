from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.structural_readiness import (
    audit_annual_inventory,
    center_signed_hourly_shape,
)


def _inventory(**updates):
    row = {
        "country": "CH", "scenario": "vendor_case", "delivery_year": 2030,
        "publication_date": "2026-01-01T00:00:00Z",
        "ingested_at_utc": "2026-01-02T00:00:00Z", "track": "scenario",
        "source": "test_fixture", "quality_flag": "synthetic_unit_only",
        "pv_gw": 10.0, "battery_power_gw": 0.0, "battery_energy_gwh": 0.0,
        "dsm_gw": 0.0,
    }
    row.update(updates)
    return pd.DataFrame([row])


def _audit(frame):
    return audit_annual_inventory(
        frame, as_of=pd.Timestamp("2026-09-07T12:00Z"),
        countries=("CH", "DE"), years=(2030, 2035),
    )


def _field(matrix, field, country="CH", year=2030):
    return matrix.loc[
        matrix["field"].eq(field) & matrix["country"].eq(country)
        & matrix["delivery_year"].eq(year), "field_status"
    ].item()


def test_inventory_keeps_exact_labels_gaps_and_explicit_zero_separate():
    original = _inventory()
    untouched = original.copy(deep=True)
    matrix, report = _audit(original)
    pd.testing.assert_frame_equal(original, untouched)
    assert report["scenario_labels"] == ["vendor_case"]
    assert _field(matrix, "pv_gw") == "PRESENT"
    assert _field(matrix, "wind_gw") == "MISSING"
    assert _field(matrix, "battery_power_gw") == "PRESENT_ZERO"
    assert _field(matrix, "battery_charge_efficiency") == "NOT_APPLICABLE"
    assert _field(matrix, "dsm_max_shift_hours") == "NOT_APPLICABLE"
    assert _field(matrix, "pv_gw", year=2035) == "NO_EXACT_ROW"
    assert _field(matrix, "pv_gw", country="DE") == "NO_EXACT_ROW"
    assert all(v is False for k, v in report["authority"].items() if k != "status")


def test_battery_duration_and_operational_assumptions_are_not_imputed():
    matrix, report = _audit(_inventory(battery_power_gw=2.0, battery_energy_gwh=0.0))
    assert report["battery_size_conflicts"] == 1
    assert _field(matrix, "battery_charge_efficiency") == "MISSING"
    assert _field(matrix, "battery_terminal_soc_share") == "MISSING"
    matrix, report = _audit(_inventory(battery_power_gw=2.0, battery_energy_gwh=None))
    assert _field(matrix, "battery_energy_gwh") == "MISSING"
    assert report["battery_size_conflicts"] == 0  # unknown, not an asserted conflict


@pytest.mark.parametrize("field,value", [
    ("pv_gw", -1), ("pv_gw", np.inf), ("pv_gw", True), ("pv_gw", "invalid"),
    ("managed_charging_share", 1.01), ("battery_charge_efficiency", 0),
    ("battery_charge_efficiency", 1.1), ("demand_twh", 0),
])
def test_physical_units_reject_invalid_values_without_converting_them_to_zero(field, value):
    matrix, _ = _audit(_inventory(battery_power_gw=1.0, battery_energy_gwh=2.0, **{field: value}))
    assert _field(matrix, field) == "INVALID"


@pytest.mark.parametrize("change", [
    {"publication_date": "2027-01-01T00:00Z"},
    {"ingested_at_utc": "2027-01-01T00:00Z"},
    {"publication_date": None}, {"delivery_month": 1},
])
def test_future_or_nonannual_rows_cannot_supply_an_annual_cell(change):
    matrix, _ = _audit(_inventory(**change))
    assert _field(matrix, "pv_gw") == "NO_EXACT_ROW"


def test_missing_ingestion_is_visible_but_does_not_erase_descriptive_coverage():
    matrix, report = _audit(_inventory(ingested_at_utc=None))
    assert _field(matrix, "pv_gw") == "PRESENT"
    assert report["ingestion_missing_or_invalid"] == 1
    assert matrix["ingestion_timestamp_missing"].all()


def test_duplicate_editions_are_not_silently_selected_or_merged():
    frame = pd.concat([_inventory(), _inventory(publication_date="2026-03-01T00:00Z")])
    matrix, _ = _audit(frame.reset_index(drop=True))
    assert _field(matrix, "pv_gw") == "AMBIGUOUS"


@pytest.mark.parametrize("year", [2030.5, True, np.nan, np.inf])
def test_fractional_or_missing_years_are_not_truncated(year):
    with pytest.raises(ValueError, match="delivery_year"):
        _audit(_inventory(delivery_year=year))


def _prices(start, end):
    index = pd.date_range(start, end, freq="h", inclusive="left", tz="Europe/Zurich")
    values = np.where(index.hour < 12, -40.0, 70.0) + index.month * 20
    return pd.Series(values, index=index.tz_convert("UTC"), name="price_eur_mwh")


@pytest.mark.parametrize("start,end,hours", [
    ("2030-03-01", "2030-04-01", 743),
    ("2030-10-01", "2030-11-01", 745),
    ("2028-02-01", "2028-03-01", 696),
])
def test_signed_shape_uses_real_swiss_months_and_retains_both_dst_hours(start, end, hours):
    prices = _prices(start, end)
    original = prices.copy()
    shape = center_signed_hourly_shape(prices)
    assert len(shape) == hours
    assert shape.index.equals(prices.index)
    assert shape.min() < 0 < shape.max()
    assert abs(shape.mean()) < 1e-12
    np.testing.assert_allclose(shape, prices - prices.mean(), atol=1e-12, rtol=0)
    pd.testing.assert_series_equal(prices, original)


def test_neutrality_is_monthly_not_annual_and_preserves_relative_price_differences():
    prices = _prices("2030-01-01", "2030-04-01")
    shape = center_signed_hourly_shape(prices)
    months = shape.index.tz_convert("Europe/Zurich").strftime("%Y-%m")
    assert shape.groupby(months).mean().abs().max() < 1e-12
    # Analytical recomposition only, no new assembly adapter or production curve.
    levels = pd.Series(np.where(months == "2030-01", 0.0, 80.0), index=shape.index)
    np.testing.assert_allclose((levels + shape).groupby(months).mean(),
                               levels.groupby(months).mean(), rtol=0, atol=1e-12)
    shifted = prices + pd.Series(np.where(months == "2030-01", 700, -90), index=prices.index)
    np.testing.assert_allclose(center_signed_hourly_shape(shifted), shape, rtol=0, atol=1e-12)


def test_timestamp_storage_resolution_does_not_change_shape():
    prices = _prices("2030-03-01", "2030-04-01")
    coarse = prices.copy()
    coarse.index = coarse.index.as_unit("us")
    pd.testing.assert_series_equal(center_signed_hourly_shape(prices), center_signed_hourly_shape(coarse))


@pytest.mark.parametrize("kind", ["missing", "duplicate", "unsorted", "naive", "qh", "nonfinite", "partial"])
def test_invalid_delivery_grids_and_prices_fail_closed(kind):
    prices = _prices("2030-03-01", "2030-04-01")
    if kind == "missing":
        prices = prices.drop(prices.index[42])
    elif kind == "duplicate":
        prices = pd.concat([prices, prices.iloc[-1:]])
    elif kind == "unsorted":
        prices = prices.iloc[::-1]
    elif kind == "naive":
        prices.index = prices.index.tz_localize(None)
    elif kind == "qh":
        prices = prices.resample("15min").ffill()
    elif kind == "nonfinite":
        prices.iloc[42] = np.nan
    else:
        prices = prices.iloc[1:]
    with pytest.raises((ValueError, TypeError)):
        center_signed_hourly_shape(prices)


def _captured_history(tmp_path, monkeypatch):
    import scripts.audit_lt_structural_readiness as runner

    source = tmp_path / "prepared"
    source.mkdir()
    hourly = _prices("2030-01-01", "2030-03-01")
    qh_index = pd.date_range(hourly.index[0], hourly.index[-1] + pd.Timedelta(hours=1),
                            freq="15min", inclusive="left")
    quarter = hourly.reindex(qh_index).ffill().to_frame("price_eur_mwh")
    quarter.to_parquet(source / "epex-ch.parquet")
    metadata = {"files": {"epex-ch.parquet": hashlib.sha256((source / "epex-ch.parquet").read_bytes()).hexdigest()}}
    (source / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(runner, "D300", source)
    monkeypatch.setattr(runner, "D300_MANIFEST_SHA", hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest())
    return runner, source, quarter


def test_history_diagnostic_excludes_months_not_closed_at_the_origin(tmp_path, monkeypatch):
    runner, _, _ = _captured_history(tmp_path, monkeypatch)
    result = runner._signed_history(tmp_path, pd.Timestamp("2030-02-15T00:00Z"))
    assert result["complete_months"] == 1
    assert result["hours"] == 744
    assert result["negative_raw_hours_retained"] == 372
    saved = pd.read_parquet(tmp_path / "signed-historical-targets.parquet")
    assert saved.index.max() < pd.Timestamp("2030-02-01", tz="Europe/Zurich")
    audit = pd.read_csv(tmp_path / "signed-history-months.csv")
    assert audit.loc[audit["month"].eq("2030-02"), "status"].item() == "EXCLUDED_NOT_CLOSED_AT_ORIGIN"


def test_changed_source_bytes_are_rejected_before_target_output(tmp_path, monkeypatch):
    runner, source, quarter = _captured_history(tmp_path, monkeypatch)
    quarter.iloc[0, 0] += 1.0
    quarter.to_parquet(source / "epex-ch.parquet")
    with pytest.raises(ValueError, match="source hash mismatch"):
        runner._signed_history(tmp_path, pd.Timestamp("2030-03-01T00:00Z"))
    assert not (tmp_path / "signed-historical-targets.parquet").exists()


def test_quarter_hour_changes_cannot_be_misrepresented_as_hourly_transport(tmp_path, monkeypatch):
    runner, source, quarter = _captured_history(tmp_path, monkeypatch)
    quarter.iloc[0, 0] += 1.0
    quarter.to_parquet(source / "epex-ch.parquet")
    metadata = {"files": {"epex-ch.parquet": hashlib.sha256((source / "epex-ch.parquet").read_bytes()).hexdigest()}}
    (source / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(runner, "D300_MANIFEST_SHA", hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="native hourly identity"):
        runner._signed_history(tmp_path, pd.Timestamp("2030-03-01T00:00Z"))
