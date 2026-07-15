from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from pfc_shaping.data.forward_proxy import (
    ForwardSnapshot,
    load_forward_snapshot,
    validate_forward_snapshot,
)
from pfc_shaping.lt.model.assembler import PFCAssembler
from pfc_shaping.pipeline.monthly_curve_authority import (
    delivery_months_for_local_window,
    delivery_months_for_window,
    delivery_months_from_prices,
    latest_base_prices_by_market,
    monthly_solver_enabled,
    monthly_solver_settings,
    solve_monthly_level_authority,
    solve_monthly_level_authority_from_history,
)
from scripts.export_local_test_ch_hourly_csv import main as export_hourly_main


class Contract:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _NonNeutralShapeHourly:
    _use_seasonal_hourly = False
    f_W_seasonal_: dict = {}
    f_W_: dict = {}

    def apply(self, idx, cal, reference_date=None):
        idx_local = idx.tz_convert("Europe/Zurich")
        values = pd.Series(1.0, index=idx, name="f_H")
        values.loc[(idx_local.hour >= 8) & (idx_local.hour < 20)] = 1.35
        values.loc[idx_local.hour < 6] = 0.70
        return values


class _ShapeHourlyWithoutSeasonalFlag:
    f_W_seasonal_: dict = {}
    f_W_: dict = {}

    def apply(self, idx, cal, reference_date=None):
        idx_local = idx.tz_convert("Europe/Zurich")
        values = pd.Series(1.0, index=idx, name="f_H")
        values.loc[(idx_local.hour >= 8) & (idx_local.hour < 20)] = 1.20
        values.loc[idx_local.hour < 6] = 0.80
        return values


class _NonNeutralShapeIntraday:
    def apply(self, idx, cal, entso_forecast=None, reference_date=None):
        values = pd.Series(1.0, index=idx, name="f_Q")
        values.iloc[::2] = 1.08
        values.iloc[1::2] = 0.92
        return values


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "monthly_curve_phase_e_parity_baseline.json"


def _load_parity_fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _contract_signature(contracts: list[Contract]) -> list[dict[str, object]]:
    return [
        {
            "name": str(contract.name),
            "price": round(float(contract.price), 10),
            "start": str(pd.Timestamp(contract.start)),
            "end": str(pd.Timestamp(contract.end)),
            "product_type": str(getattr(contract, "product_type", "")),
            "is_hard": bool(getattr(contract, "is_hard", True)),
            "penalty_weight": round(float(getattr(contract, "penalty_weight", 1.0)), 10),
        }
        for contract in contracts
    ]


def _boundaries(year: int, start_month: int, end_month: int, tz: str):
    start = pd.Timestamp(year=year, month=start_month, day=1, tz=tz).tz_convert("UTC")
    if end_month == 12:
        end = pd.Timestamp(year=year + 1, month=1, day=1, tz=tz).tz_convert("UTC")
    else:
        end = pd.Timestamp(year=year, month=end_month + 1, day=1, tz=tz).tz_convert("UTC")
    return start, end


def _history() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    date = pd.Timestamp("2026-06-17")
    for product, price in {
        "2028": 80.0,
        "2028-Q1": 110.0,
        "2029": 70.0,
    }.items():
        rows.append(_row(date, "CH", product, price))
    for month in range(1, 13):
        rows.append(_row(date, "DE", f"2028-{month:02d}", 70.0 + month))
    rows.append(_row(date, "DE", "2028", 76.5))
    return pd.DataFrame(rows)


def _row(date: pd.Timestamp, market: str, product: str, price: float) -> dict[str, object]:
    return {
        "date": date,
        "market": market,
        "load_type": "BASE",
        "product": product,
        "price": price,
    }


def test_monthly_solver_default_config_is_off() -> None:
    assert not monthly_solver_enabled({"forwards": {}}, market="CH")


def test_monthly_solver_defaults_include_structural_template_fallback() -> None:
    settings = monthly_solver_settings({"forwards": {"monthly_curve_solver": {"enabled": True}}})

    assert settings["allow_template_structural_fallback"] is True
    assert settings["structural_amplitude_eur_mwh"] == 110.0
    assert settings["structural_weight"] == 1.0


def test_monthly_authority_manifest_records_structural_template_summary() -> None:
    own = {"2028": 80.0, "2028-Q1": 110.0}
    months = delivery_months_from_prices(own)

    authority = solve_monthly_level_authority(
        market="CH",
        delivery_months=months,
        own_base_prices=own,
        all_market_base_prices={},
        eex_history=pd.DataFrame(columns=["date", "market", "load_type", "product", "price"]),
        run_timestamp=pd.Timestamp("2026-06-17"),
        settings={
            "enabled": True,
            "markets": [],
            "allow_template_structural_fallback": True,
            "structural_amplitude_eur_mwh": 110.0,
            "structural_weight": 1.0,
        },
        allow_unverified_inputs=True,
    )

    summary = authority.manifest["structural_prior_summary"]
    assert authority.manifest["structural_status"] == "STRUCTURAL_TEMPLATE"
    assert summary["status"] == "STRUCTURAL_TEMPLATE"
    assert summary["sources"] == ["template_structural_monthly_ratios"]
    assert summary["fallback_reasons"] == ["empty_history"]
    assert summary["amplitude_eur_mwh_min"] == 110.0
    assert summary["amplitude_eur_mwh_max"] == 110.0
    assert summary["zero_mean_parent_space_all"] is True
    assert summary["max_abs_parent_mean_residual_max"] <= 1e-10
    assert summary["n_history_max"] == 0.0


def test_monthly_authority_active_config_hash_includes_structural_prior_knobs() -> None:
    history = _history()
    own = {"2028": 80.0, "2028-Q1": 110.0}
    months = delivery_months_from_prices(own)
    base_settings = {
        "enabled": True,
        "markets": ["DE"],
        "allow_template_structural_fallback": True,
        "structural_amplitude_eur_mwh": 110.0,
        "panel_weight": 1.0,
        "history_weight": 0.5,
        "structural_weight": 1.0,
    }

    baseline = solve_monthly_level_authority(
        market="CH",
        delivery_months=months,
        own_base_prices=own,
        all_market_base_prices={"DE": {f"2028-{month:02d}": 70.0 + month for month in range(1, 13)}},
        eex_history=history,
        run_timestamp=pd.Timestamp("2026-06-17"),
        settings=base_settings,
        allow_unverified_inputs=True,
    )
    changed_amplitude = solve_monthly_level_authority(
        market="CH",
        delivery_months=months,
        own_base_prices=own,
        all_market_base_prices={"DE": {f"2028-{month:02d}": 70.0 + month for month in range(1, 13)}},
        eex_history=history,
        run_timestamp=pd.Timestamp("2026-06-17"),
        settings={**base_settings, "structural_amplitude_eur_mwh": 40.0},
        allow_unverified_inputs=True,
    )
    changed_weight = solve_monthly_level_authority(
        market="CH",
        delivery_months=months,
        own_base_prices=own,
        all_market_base_prices={"DE": {f"2028-{month:02d}": 70.0 + month for month in range(1, 13)}},
        eex_history=history,
        run_timestamp=pd.Timestamp("2026-06-17"),
        settings={**base_settings, "structural_weight": 0.25},
        allow_unverified_inputs=True,
    )

    assert baseline.manifest["active_config_hash"] != changed_amplitude.manifest["active_config_hash"]
    assert baseline.manifest["active_config_hash"] != changed_weight.manifest["active_config_hash"]


def test_delivery_months_for_window_respects_exclusive_end_boundary() -> None:
    months = delivery_months_for_window(
        start_date=pd.Timestamp("2028-01-01", tz="Europe/Zurich"),
        horizon_days=366,
        timezone="Europe/Zurich",
    )

    assert str(months[0]) == "2028-01"
    assert str(months[-1]) == "2028-12"
    assert "2029-01" not in {str(month) for month in months}


def test_delivery_months_for_local_window_uses_intended_artifact_months() -> None:
    rounded_build_months = delivery_months_for_window(
        start_date="2026-06-12 22:00:00",
        horizon_days=1664,
        timezone="Europe/Zurich",
    )
    artifact_months = delivery_months_for_local_window(
        start_date="2026-06-13",
        end_date="2030-12-31",
        timezone="Europe/Zurich",
    )

    assert "2031-01" in {str(month) for month in rounded_build_months}
    assert str(artifact_months[0]) == "2026-06"
    assert str(artifact_months[-1]) == "2030-12"
    assert "2031-01" not in {str(month) for month in artifact_months}


def test_monthly_authority_hash_parity_and_quoted_keys_exclude_synthetic_months() -> None:
    history = _history()
    own = {"2028": 80.0, "2028-Q1": 110.0}
    neighbors = {"DE": {f"2028-{month:02d}": 70.0 + month for month in range(1, 13)}}
    months = delivery_months_from_prices(own)
    settings = {
        "enabled": True,
        "markets": ["DE"],
        "lambda_prior": 1e-6,
        "lambda_smooth_month": 1.0,
        "lambda_smooth_yoy": 0.0,
        "lambda_shape": 1.0,
    }

    left = solve_monthly_level_authority(
        market="CH",
        delivery_months=months,
        own_base_prices=own,
        all_market_base_prices=neighbors,
        eex_history=history,
        run_timestamp=pd.Timestamp("2026-06-17"),
        settings=settings,
        original_forward_prices=own,
        allow_unverified_inputs=True,
    )
    right = solve_monthly_level_authority(
        market="CH",
        delivery_months=months,
        own_base_prices=own,
        all_market_base_prices=neighbors,
        eex_history=history,
        run_timestamp=pd.Timestamp("2026-06-17"),
        settings=settings,
        original_forward_prices=own,
        allow_unverified_inputs=True,
    )

    assert left.monthly_solution_hash == right.monthly_solution_hash
    assert left.active_constraints_hash == right.active_constraints_hash
    assert "2028" in left.quoted_keys
    assert "2028-Q1" in left.quoted_keys
    assert "2028-04" not in left.quoted_keys
    assert "2028-04" in left.assembler_base_prices


def test_monthly_authority_direct_and_history_paths_hash_equal(tmp_path) -> None:
    fixture = _load_parity_fixture()
    expected = fixture["monthly_authority"]
    history = _history()
    forwards_path = tmp_path / "forwards.parquet"
    history.to_parquet(forwards_path)
    run_timestamp, own = latest_base_prices_by_market(history, market="CH")
    _, de = latest_base_prices_by_market(history, market="DE")
    months = delivery_months_from_prices(own)
    settings = {
        "enabled": True,
        "markets": ["DE"],
        "lambda_prior": 1e-6,
        "lambda_smooth_month": 1.0,
        "lambda_smooth_yoy": 0.0,
        "lambda_shape": 1.0,
    }

    direct = solve_monthly_level_authority(
        market="CH",
        delivery_months=months,
        own_base_prices=own,
        all_market_base_prices={"DE": de},
        eex_history=history,
        run_timestamp=run_timestamp,
        settings=settings,
        original_forward_prices=own,
        allow_unverified_inputs=True,
    )
    from_history = solve_monthly_level_authority_from_history(
        forwards_path=forwards_path,
        market="CH",
        delivery_months=months,
        settings=settings,
        original_forward_prices=own,
    )

    assert direct.monthly_solution_hash == from_history.monthly_solution_hash
    assert direct.active_constraints_hash == from_history.active_constraints_hash
    assert direct.monthly_solution_hash == expected["monthly_solution_hash"]
    assert direct.active_constraints_hash == expected["active_constraints_hash"]
    assert sorted(direct.quoted_keys) == expected["quoted_keys"]
    assert sorted(direct.synthetic_monthly_keys) == expected["synthetic_monthly_keys"]
    pd.testing.assert_series_equal(direct.result.monthly_curve, from_history.result.monthly_curve)


def test_legacy_final_calibrator_preserves_cascaded_month_priority() -> None:
    selected = PFCAssembler._select_base_contract_key(
        key_m="2028-04",
        key_q="2028-Q2",
        key_y="2028",
        base_prices={"2028": 80.0, "2028-04": 81.0},
        quoted_keys={"2028"},
    )
    assert selected == "2028-04"


def test_final_calibrator_keeps_genuinely_quoted_month() -> None:
    selected = PFCAssembler._select_base_contract_key(
        key_m="2028-04",
        key_q="2028-Q2",
        key_y="2028",
        base_prices={"2028": 80.0, "2028-04": 81.0},
        quoted_keys={"2028", "2028-04"},
    )
    assert selected == "2028-04"


def test_solver_monthly_level_preserves_monthly_base_means_in_far_horizon() -> None:
    assembler = PFCAssembler(
        shape_hourly=_NonNeutralShapeHourly(),
        shape_intraday=_NonNeutralShapeIntraday(),
        monthly_level_authority="solver",
        skip_legacy_level_cascade=True,
        skip_legacy_base_smoothing=True,
        calibrator=None,
        water_value=None,
    )
    base_prices = {
        "2029-01": 100.0,
        "2029-02": 60.0,
        "2029-03": 90.0,
    }

    df = assembler.build(
        base_prices=base_prices,
        start_date="2029-01-01",
        horizon_days=60,
        reference_date=pd.Timestamp("2026-06-22", tz="Europe/Zurich"),
    )
    idx_local = df.index.tz_convert("Europe/Zurich")

    assert df["f_S"].eq(1.0).all()
    for month_key, group in df.groupby(idx_local.strftime("%Y-%m")):
        if month_key in base_prices:
            assert group["price_shape"].mean() == pytest.approx(group["B"].mean(), abs=1e-9)


def test_solver_monthly_level_preserves_monthly_base_means_after_near_term_rebalance() -> None:
    assembler = PFCAssembler(
        shape_hourly=_NonNeutralShapeHourly(),
        shape_intraday=_NonNeutralShapeIntraday(),
        monthly_level_authority="solver",
        skip_legacy_level_cascade=True,
        skip_legacy_base_smoothing=True,
        calibrator=None,
        water_value=None,
    )
    base_prices = {
        "2026-07": 100.0,
        "2026-08": 65.0,
    }

    df = assembler.build(
        base_prices=base_prices,
        start_date="2026-07-01",
        horizon_days=62,
        reference_date=pd.Timestamp("2026-06-22", tz="Europe/Zurich"),
    )
    idx_local = df.index.tz_convert("Europe/Zurich")

    for month_key, group in df.groupby(idx_local.strftime("%Y-%m")):
        if month_key in base_prices:
            assert group["price_shape"].mean() == pytest.approx(group["B"].mean(), abs=1e-9)


def test_solver_final_projection_preserves_base_and_reprices_peak_after_bridge() -> None:
    assembler = PFCAssembler(
        shape_hourly=_NonNeutralShapeHourly(),
        shape_intraday=_NonNeutralShapeIntraday(),
        monthly_level_authority="solver",
        skip_legacy_level_cascade=True,
        skip_legacy_base_smoothing=True,
        calibrator=None,
        water_value=None,
    )
    base_prices = {
        "2026-07": 100.0,
        "2026-07-Peak": 120.0,
    }

    df = assembler.build(
        base_prices=base_prices,
        quoted_keys=set(base_prices),
        start_date="2026-06-30 22:00:00",
        horizon_days=31,
        reference_date=pd.Timestamp("2026-06-22", tz="Europe/Zurich"),
    )
    idx_local = df.index.tz_convert("Europe/Zurich")
    july = (idx_local.year == 2026) & (idx_local.month == 7)
    peak = july & assembler._is_peak_timestamp(idx_local, country="CH")
    offpeak = july & ~assembler._is_peak_timestamp(idx_local, country="CH")
    total_hours = float(july.sum())
    peak_hours = float(peak.sum())
    offpeak_hours = float(offpeak.sum())
    implied_offpeak = (100.0 * total_hours - 120.0 * peak_hours) / offpeak_hours

    assert float(df.loc[july, "price_shape"].mean()) == pytest.approx(100.0, abs=1e-9)
    assert float(df.loc[peak, "price_shape"].mean()) == pytest.approx(120.0, abs=1e-9)
    assert float(df.loc[offpeak, "price_shape"].mean()) == pytest.approx(implied_offpeak, abs=1e-9)
    assert df["calibrated"].all()
    assert assembler.final_product_projection_report_["max_abs_error_eur_mwh"] <= 1e-9
    assert assembler.final_product_projection_report_["partial_peak_quote_ids"] == []


def test_solver_final_projection_rejects_explicit_offpeak_quote() -> None:
    assembler = PFCAssembler(
        shape_hourly=_NonNeutralShapeHourly(),
        shape_intraday=_NonNeutralShapeIntraday(),
        monthly_level_authority="solver",
        skip_legacy_level_cascade=True,
        skip_legacy_base_smoothing=True,
        calibrator=None,
        water_value=None,
    )
    prices = {"2026-07": 100.0, "2026-07-OffPeak": 80.0}

    with pytest.raises(ValueError, match="explicit OFFPEAK quote"):
        assembler.build(
            base_prices=prices,
            quoted_keys=set(prices),
            start_date="2026-06-30 22:00:00",
            horizon_days=31,
            reference_date=pd.Timestamp("2026-06-22", tz="Europe/Zurich"),
        )


def test_solver_final_projection_rejects_partially_covered_peak_quote() -> None:
    assembler = PFCAssembler(
        shape_hourly=_NonNeutralShapeHourly(),
        shape_intraday=_NonNeutralShapeIntraday(),
        monthly_level_authority="solver",
        skip_legacy_level_cascade=True,
        skip_legacy_base_smoothing=True,
        calibrator=None,
        water_value=None,
    )
    prices = {"2026-07": 100.0, "2026-07-Peak": 120.0}

    with pytest.raises(ValueError, match="partially overlap"):
        assembler.build(
            base_prices=prices,
            quoted_keys=set(prices),
            start_date="2026-07-15 00:00:00",
            horizon_days=10,
            reference_date=pd.Timestamp("2026-06-22", tz="Europe/Zurich"),
        )


def test_solver_monthly_level_accepts_hourly_model_without_seasonal_flag() -> None:
    assembler = PFCAssembler(
        shape_hourly=_ShapeHourlyWithoutSeasonalFlag(),
        shape_intraday=_NonNeutralShapeIntraday(),
        monthly_level_authority="solver",
        skip_legacy_level_cascade=True,
        skip_legacy_base_smoothing=True,
        calibrator=None,
        water_value=None,
    )
    base_prices = {
        "2029-01": 100.0,
        "2029-02": 60.0,
    }

    df = assembler.build(
        base_prices=base_prices,
        start_date="2029-01-01",
        horizon_days=60,
        reference_date=pd.Timestamp("2026-06-22", tz="Europe/Zurich"),
    )
    idx_local = df.index.tz_convert("Europe/Zurich")

    assert not df.empty
    for month_key, group in df.groupby(idx_local.strftime("%Y-%m")):
        if month_key in base_prices:
            assert group["price_shape"].mean() == pytest.approx(group["B"].mean(), abs=1e-9)


def test_solver_monthly_level_requires_legacy_cascade_and_smoothing_skips() -> None:
    assembler = PFCAssembler(
        shape_hourly=_NonNeutralShapeHourly(),
        shape_intraday=_NonNeutralShapeIntraday(),
        monthly_level_authority="solver",
        skip_legacy_level_cascade=False,
        skip_legacy_base_smoothing=True,
        calibrator=None,
        water_value=None,
    )

    with pytest.raises(ValueError, match="monthly_level_authority='solver' requires"):
        assembler.build(
            base_prices={"2029-01": 100.0},
            start_date="2029-01-01",
            horizon_days=1,
            reference_date=pd.Timestamp("2026-06-22", tz="Europe/Zurich"),
        )


def test_solver_mode_final_calibration_uses_non_overlapping_average_contracts() -> None:
    fixture = _load_parity_fixture()
    assembler = object.__new__(PFCAssembler)
    assembler.skip_legacy_level_cascade = True
    start_utc, end_utc = _boundaries(2028, 1, 12, "Europe/Zurich")
    idx = pd.date_range(start_utc, end_utc, freq="h", inclusive="left")
    contracts = assembler._build_non_overlapping_contracts(
        idx=idx,
        base_prices={
            "2028": 80.0,
            "2028-Q1": 110.0,
            **{f"2028-{month:02d}": 70.0 + month for month in range(1, 13)},
        },
        quoted_keys={"2028", "2028-Q1"},
        futures_contract_cls=Contract,
        period_boundaries_fn=_boundaries,
        country="CH",
    )
    signature = _contract_signature(contracts)

    assert [contract.name for contract in contracts] == [
        "2028-Q1<monthly_solver:2028-Q1>",
        "2028-RESIDUAL<monthly_solver:2028>",
    ]
    assert contracts[0].price == pytest.approx(110.0)
    assert contracts[1].price < 80.0
    assert signature == fixture["on_solver_contracts"]
    assert _sha256_json(signature) == fixture["on_solver_contract_signature_hash"]


def test_solver_mode_final_calibration_skips_partial_products() -> None:
    assembler = object.__new__(PFCAssembler)
    assembler.skip_legacy_level_cascade = True
    assembler.monthly_constraint_tolerance = 1e-9
    idx = pd.date_range("2028-07-01", "2029-01-01", freq="h", inclusive="left", tz="UTC")

    contracts = assembler._build_non_overlapping_contracts(
        idx=idx,
        base_prices={
            "2028": 80.0,
            **{f"2028-{month:02d}": 90.0 for month in range(7, 13)},
        },
        quoted_keys={"2028"},
        futures_contract_cls=Contract,
        period_boundaries_fn=_boundaries,
        country="CH",
    )

    assert contracts == []


def test_flag_off_final_calibration_keeps_legacy_monthly_contracts() -> None:
    fixture = _load_parity_fixture()
    assembler = object.__new__(PFCAssembler)
    assembler.skip_legacy_level_cascade = False
    assembler.peak_source_policy = "same_first"
    idx = pd.date_range("2028-01-01", "2029-01-01", freq="h", inclusive="left", tz="UTC")

    contracts = assembler._build_non_overlapping_contracts(
        idx=idx,
        base_prices={"2028": 80.0, "2028-Q1": 110.0},
        quoted_keys={"2028", "2028-Q1"},
        futures_contract_cls=Contract,
        period_boundaries_fn=_boundaries,
        country="CH",
    )
    signature = _contract_signature(contracts)

    assert len(contracts) == 12
    assert [contract.name for contract in contracts[:3]] == [
        "2028-01<2028-Q1>",
        "2028-02<2028-Q1>",
        "2028-03<2028-Q1>",
    ]
    assert contracts[-1].name == "2028-12<2028>"
    assert signature == fixture["off_legacy_contracts"]
    assert _sha256_json(signature) == fixture["off_legacy_contract_signature_hash"]


def test_solver_mode_final_calibration_ignores_peak_keys_case_insensitively() -> None:
    def boundaries(year: int, start_month: int, end_month: int, tz: str):
        start = pd.Timestamp(year=year, month=start_month, day=1, tz=tz).tz_convert("UTC")
        if end_month == 12:
            end = pd.Timestamp(year=year + 1, month=1, day=1, tz=tz).tz_convert("UTC")
        else:
            end = pd.Timestamp(year=year, month=end_month + 1, day=1, tz=tz).tz_convert("UTC")
        return start, end

    assembler = object.__new__(PFCAssembler)
    assembler.skip_legacy_level_cascade = True
    start_utc, end_utc = boundaries(2028, 1, 12, "Europe/Zurich")
    idx = pd.date_range(start_utc, end_utc, freq="h", inclusive="left")
    contracts = assembler._build_non_overlapping_contracts(
        idx=idx,
        base_prices={
            "2028": 80.0,
            "2028-Peak": 120.0,
            "2028-OffPeak": 65.0,
            **{f"2028-{month:02d}": 70.0 + month for month in range(1, 13)},
        },
        quoted_keys={"2028", "2028-Peak", "2028-OffPeak"},
        futures_contract_cls=Contract,
        period_boundaries_fn=boundaries,
        country="CH",
    )

    assert [contract.name for contract in contracts] == ["2028<monthly_solver:2028>"]


def test_export_refuses_solver_with_mutating_legacy_post_processor() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        export_hourly_main(
            [
                "--enable-monthly-forward-curve-solver",
                "--enable-quote-aware-monthly-smoothing",
            ]
        )


def _forward_provenance(tmp_path: Path) -> ForwardSnapshot:
    source = tmp_path / "eex.xlsx"
    rows = [
        [None, "Y01_2028_BASE", "Q01_2028_BASE", "M01_2028_BASE"],
        [None, "ISIN-1", "ISIN-2", "ISIN-3"],
        ["Date", None, None, None],
        ["10.07.2026", 80.0, 80.0, 80.0],
    ]
    with pd.ExcelWriter(source, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="CH", index=False, header=False)
    return load_forward_snapshot(
        pd.DataFrame(),
        eex_report_path=str(source),
        config={},
        market="CH",
        allow_spot_proxy=False,
        exact_source_only=True,
        source_available_at="2026-07-10T18:00:00Z",
    )


def test_monthly_authority_manifest_uses_real_forward_snapshot_date(tmp_path: Path) -> None:
    snapshot = _forward_provenance(tmp_path)
    valuation = pd.Timestamp("2026-07-13T20:00:00Z")
    authority = solve_monthly_level_authority(
        market="CH",
        delivery_months=pd.period_range("2028-01", "2028-12", freq="M"),
        own_base_prices=dict(snapshot.prices),
        run_timestamp=valuation,
        settings={"allow_template_structural_fallback": True},
        own_forward_snapshot=snapshot,
        forward_eligibility=validate_forward_snapshot(
            snapshot,
            reference_timestamp=valuation,
        ),
    )

    assert authority.manifest["forward_snapshot_date"] == "2026-07-10"
    assert authority.manifest["forward_source_kind"] == "EEX_XLSX"
    assert authority.manifest["hard_quote_eligible"] is True
    assert authority.manifest["promotion_eligible"] is True
    assert authority.inputs.own_quotes[0].snapshot_date == pd.Timestamp(
        "2026-07-10", tz="UTC"
    )
    quote_by_product = {quote.product: quote for quote in authority.inputs.own_quotes}
    assert quote_by_product["2028"].source == snapshot.quote_lineage["2028"]["quote_id"]
    assert quote_by_product["2028"].quote_id == snapshot.quote_lineage["2028"]["quote_id"]
    assert quote_by_product["2028"].snapshot_id == snapshot.snapshot_id
    assert quote_by_product["2028"].observation_id == snapshot.observation_id
    assert authority.constraints.rows["source_quote_ids"].str.len().gt(0).all()
    assert len(authority.manifest["constraint_provenance_hash"]) == 64
    assert len(authority.manifest["product_hierarchy_policy_sha256"]) == 64


def test_monthly_authority_rejects_prices_detached_from_snapshot(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly match"):
        solve_monthly_level_authority(
            market="CH",
            delivery_months=pd.period_range("2028-01", "2028-12", freq="M"),
            own_base_prices={"2028": 81.0},
            run_timestamp=pd.Timestamp("2026-07-13"),
            settings={"allow_template_structural_fallback": True},
            own_forward_snapshot=_forward_provenance(tmp_path),
        )


def test_monthly_authority_requires_matching_eligibility_receipt(tmp_path: Path) -> None:
    snapshot = _forward_provenance(tmp_path)
    with pytest.raises(ValueError, match="forward_eligibility.v1"):
        solve_monthly_level_authority(
            market="CH",
            delivery_months=pd.period_range("2028-01", "2028-12", freq="M"),
            own_base_prices=dict(snapshot.prices),
            run_timestamp=pd.Timestamp("2026-07-13T20:00:00Z"),
            settings={"allow_template_structural_fallback": True},
            own_forward_snapshot=snapshot,
        )


def test_monthly_authority_rejects_forged_eligibility_mapping(tmp_path: Path) -> None:
    snapshot = _forward_provenance(tmp_path)
    forged = {
        "schema_version": "forward_eligibility.v1",
        "status": "PASS",
        "requested_role": "HARD_LEVEL",
        "snapshot_id": snapshot.snapshot_id,
    }
    with pytest.raises(TypeError, match="verified ForwardEligibility"):
        solve_monthly_level_authority(
            market="CH",
            delivery_months=pd.period_range("2028-01", "2028-12", freq="M"),
            own_base_prices=dict(snapshot.prices),
            run_timestamp=pd.Timestamp("2026-07-13T20:00:00Z"),
            settings={"allow_template_structural_fallback": True},
            own_forward_snapshot=snapshot,
            forward_eligibility=forged,  # type: ignore[arg-type]
        )


def test_monthly_authority_without_provenance_is_non_promotional() -> None:
    authority = solve_monthly_level_authority(
        market="CH",
        delivery_months=pd.period_range("2028-01", "2028-12", freq="M"),
        own_base_prices={"2028": 80.0},
        run_timestamp=pd.Timestamp("2026-07-13"),
        settings={"allow_template_structural_fallback": True},
        allow_unverified_inputs=True,
    )

    assert authority.manifest["forward_source_kind"] == "TEST_FIXTURE"
    assert authority.manifest["hard_quote_eligible"] is False
    assert authority.manifest["promotion_eligible"] is False


def test_monthly_authority_rejects_market_mismatch(tmp_path: Path) -> None:
    snapshot = _forward_provenance(tmp_path)
    with pytest.raises(ValueError, match="market mismatch"):
        solve_monthly_level_authority(
            market="DE",
            delivery_months=pd.period_range("2028-01", "2028-12", freq="M"),
            own_base_prices=dict(snapshot.prices),
            run_timestamp=pd.Timestamp("2026-07-13T20:00:00Z"),
            settings={"allow_template_structural_fallback": True},
            own_forward_snapshot=snapshot,
        )
