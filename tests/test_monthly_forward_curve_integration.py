from __future__ import annotations

import pytest
import pandas as pd

from pfc_shaping.lt.model.assembler import PFCAssembler
from pfc_shaping.pipeline.monthly_curve_authority import (
    delivery_months_from_prices,
    monthly_solver_enabled,
    solve_monthly_level_authority,
)
from scripts.export_local_test_ch_hourly_csv import main as export_hourly_main


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
    )

    assert left.monthly_solution_hash == right.monthly_solution_hash
    assert left.active_constraints_hash == right.active_constraints_hash
    assert "2028" in left.quoted_keys
    assert "2028-Q1" in left.quoted_keys
    assert "2028-04" not in left.quoted_keys
    assert "2028-04" in left.assembler_base_prices


def test_final_calibrator_prefers_original_quoted_base_key_over_synthetic_month() -> None:
    selected = PFCAssembler._select_base_contract_key(
        key_m="2028-04",
        key_q="2028-Q2",
        key_y="2028",
        base_prices={"2028": 80.0, "2028-04": 81.0},
        quoted_keys={"2028"},
    )
    assert selected == "2028"


def test_final_calibrator_keeps_genuinely_quoted_month() -> None:
    selected = PFCAssembler._select_base_contract_key(
        key_m="2028-04",
        key_q="2028-Q2",
        key_y="2028",
        base_prices={"2028": 80.0, "2028-04": 81.0},
        quoted_keys={"2028", "2028-04"},
    )
    assert selected == "2028-04"


def test_solver_mode_final_calibration_uses_non_overlapping_average_contracts() -> None:
    class Contract:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def boundaries(year: int, start_month: int, end_month: int, tz: str):
        start = pd.Timestamp(year=year, month=start_month, day=1, tz=tz).tz_convert("UTC")
        if end_month == 12:
            end = pd.Timestamp(year=year + 1, month=1, day=1, tz=tz).tz_convert("UTC")
        else:
            end = pd.Timestamp(year=year, month=end_month + 1, day=1, tz=tz).tz_convert("UTC")
        return start, end

    assembler = object.__new__(PFCAssembler)
    assembler.skip_legacy_level_cascade = True
    idx = pd.date_range("2028-01-01", "2029-01-01", freq="h", inclusive="left", tz="UTC")
    contracts = assembler._build_non_overlapping_contracts(
        idx=idx,
        base_prices={
            "2028": 80.0,
            "2028-Q1": 110.0,
            **{f"2028-{month:02d}": 70.0 + month for month in range(1, 13)},
        },
        quoted_keys={"2028", "2028-Q1"},
        futures_contract_cls=Contract,
        period_boundaries_fn=boundaries,
        country="CH",
    )

    assert [contract.name for contract in contracts] == [
        "2028-Q1<monthly_solver:2028-Q1>",
        "2028-RESIDUAL<monthly_solver:2028>",
    ]
    assert contracts[0].price == pytest.approx(110.0)
    assert contracts[1].price < 80.0


def test_export_refuses_solver_with_mutating_legacy_post_processor() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        export_hourly_main(
            [
                "--enable-monthly-forward-curve-solver",
                "--enable-quote-aware-monthly-smoothing",
            ]
        )
