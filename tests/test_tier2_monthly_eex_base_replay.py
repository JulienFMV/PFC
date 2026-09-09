from __future__ import annotations

import inspect
from collections import UserDict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pfc_shaping.calibration.tier2_monthly_eex_base_replay as base_replay
from pfc_shaping.calibration.tier2_monthly_eex_base_replay import (
    BASE_REPLAY_CONFIG_SCHEMA,
    BASE_REPLAY_STATUS,
    JOINT_SEGMENT_UNSUPPORTED,
    Tier2MonthlyBaseReplayError,
    Tier2MonthlyBaseReplayNumericalError,
    _validate_solve,
    _replay_monthly_base_outputs,
    validate_base_candidate_config,
)


def test_base_replay_is_target_blind_deterministic_and_reprices_retained() -> None:
    retained = _retained_quotes()
    history = _historical_context()
    config = _candidate_config()

    first = _replay_monthly_base_outputs(
        delivery_months=_delivery_months(),
        retained_quotes=retained,
        fold_historical_context=history,
        candidate_config=config,
    )
    second = _replay_monthly_base_outputs(
        delivery_months=_delivery_months(),
        retained_quotes=retained.sample(frac=1.0, random_state=7),
        fold_historical_context=history.sample(frac=1.0, random_state=11),
        candidate_config=config,
    )

    assert first == second
    assert first.status == BASE_REPLAY_STATUS
    assert first.scope == "MONTHLY_EEX_BASE_MASKED_QUOTE_ONLY"
    assert first.delivery_months == tuple(_delivery_months())
    assert np.isfinite(first.candidate_monthly_base).all()
    assert np.isfinite(first.baseline_monthly_base).all()
    assert first.candidate_max_abs_constraint_residual <= 1e-9
    assert first.baseline_max_abs_constraint_residual <= 1e-9
    assert first.candidate_monthly_base[0] == pytest.approx(74.5, abs=1e-5)
    assert first.baseline_monthly_base[0] == pytest.approx(74.5, abs=1e-5)
    for month in range(2, 13):
        expected = 80.0 + _seasonal_deviation(month)
        assert first.candidate_monthly_base[month - 1] == pytest.approx(expected)
        assert first.baseline_monthly_base[month - 1] == pytest.approx(expected)

    parameters = inspect.signature(_replay_monthly_base_outputs).parameters
    assert "target" not in parameters
    assert "target_price" not in parameters
    assert "diagnostic" not in parameters


def test_base_replay_rejects_peak_or_offpeak_retained_quotes() -> None:
    retained = _retained_quotes()
    retained.loc[retained.index[0], "load_type"] = "PEAK"

    with pytest.raises(Tier2MonthlyBaseReplayError, match=JOINT_SEGMENT_UNSUPPORTED):
        _replay_monthly_base_outputs(
            delivery_months=_delivery_months(),
            retained_quotes=retained,
            fold_historical_context=_historical_context(),
            candidate_config=_candidate_config(),
        )


def test_base_replay_requires_complete_exact_config() -> None:
    config = _candidate_config()
    del config["monthly_curve"]["lambda_shape"]

    with pytest.raises(Tier2MonthlyBaseReplayError, match="fields mismatch"):
        validate_base_candidate_config(config)

    config = _candidate_config()
    config["monthly_curve"]["implicit_default"] = 1.0
    with pytest.raises(Tier2MonthlyBaseReplayError, match="fields mismatch"):
        validate_base_candidate_config(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("lambda_smooth_month", True, "JSON float"),
        ("lambda_smooth_yoy", float("nan"), "finite"),
        ("neighbor_shrinkage", 0.5, "forbids neighbor"),
        ("robust_panel_quantile", 0.4, "exactly 0.5"),
        ("min_history_snapshots", 23, "integer >= 24"),
        ("constraint_tolerance", 1e-8, "exactly 1e-9"),
        ("lambda_smooth_month", 1, "JSON float"),
        ("neighbor_shrinkage", -0.0, "negative zero"),
    ],
)
def test_base_replay_rejects_weakened_config(
    field: str, value: object, message: str
) -> None:
    config = _candidate_config()
    config["monthly_curve"][field] = value

    with pytest.raises(Tier2MonthlyBaseReplayError, match=message):
        validate_base_candidate_config(config)


def test_base_replay_fails_closed_when_one_seasonal_month_is_insufficient() -> None:
    history = _historical_context()
    january = history["product"].astype(str).str.endswith("-01")
    first_snapshot = history["snapshot_id"].iloc[0]
    history = history[~(january & history["snapshot_id"].eq(first_snapshot))]

    with pytest.raises(Tier2MonthlyBaseReplayError, match="UNSUPPORTED_BASELINE"):
        _replay_monthly_base_outputs(
            delivery_months=_delivery_months(),
            retained_quotes=_retained_quotes(),
            fold_historical_context=history,
            candidate_config=_candidate_config(),
        )


def test_base_replay_counts_distinct_snapshots_not_multi_year_observations() -> None:
    history = _historical_context()
    history = history[history["snapshot_id"].isin([f"snapshot-{i:02d}" for i in range(12)])]
    extra: list[dict[str, object]] = []
    for snapshot_id, group in history.groupby("snapshot_id", sort=True):
        first = group.iloc[0]
        calendar_price = 90.0
        products = [("2028", calendar_price)] + [
            (f"2028-{month:02d}", calendar_price + _seasonal_deviation(month))
            for month in range(1, 13)
        ]
        for product, price in products:
            extra.append(
                {
                    **first.to_dict(),
                    "product": product,
                    "price": price,
                    "row_hash": f"extra-{snapshot_id}-{product}",
                }
            )
    history = pd.concat([history, pd.DataFrame(extra)], ignore_index=True)

    with pytest.raises(Tier2MonthlyBaseReplayError, match="snapshots=12"):
        _replay_monthly_base_outputs(
            delivery_months=_delivery_months(),
            retained_quotes=_retained_quotes(),
            fold_historical_context=history,
            candidate_config=_candidate_config(),
        )


def test_base_replay_rejects_duplicate_retained_products() -> None:
    retained = _retained_quotes()
    duplicate = retained.iloc[[0]].copy()
    duplicate["quote_id"] = "different-id"
    duplicate["price"] = duplicate["price"] + 0.001
    retained = pd.concat([retained, duplicate], ignore_index=True)

    with pytest.raises(Tier2MonthlyBaseReplayError, match="duplicate economic"):
        _replay_monthly_base_outputs(
            delivery_months=_delivery_months(),
            retained_quotes=retained,
            fold_historical_context=_historical_context(),
            candidate_config=_candidate_config(),
        )


def test_base_replay_rejects_non_plain_or_effectful_config() -> None:
    with pytest.raises(Tier2MonthlyBaseReplayError, match="plain JSON object"):
        validate_base_candidate_config(UserDict(_candidate_config()))

    config = _candidate_config()
    config["monthly_curve"]["max_prior_residual_eur_mwh"] = 0.1
    with pytest.raises(Tier2MonthlyBaseReplayError, match="requires.*null"):
        validate_base_candidate_config(config)


def test_base_replay_rejects_unbounded_objective_weight() -> None:
    config = _candidate_config()
    config["monthly_curve"]["lambda_shape"] = 1e308

    with pytest.raises(Tier2MonthlyBaseReplayError, match="must be <="):
        validate_base_candidate_config(config)


def test_base_replay_types_solver_failure_as_numerical_error(monkeypatch) -> None:
    def fail_closed(*args, **kwargs):
        raise ValueError("synthetic numerical failure")

    monkeypatch.setattr(
        base_replay,
        "solve_monthly_forward_curve_from_constraints",
        fail_closed,
    )

    with pytest.raises(Tier2MonthlyBaseReplayNumericalError, match="candidate.*failed"):
        _replay_monthly_base_outputs(
            delivery_months=_delivery_months(),
            retained_quotes=_retained_quotes(),
            fold_historical_context=_historical_context(),
            candidate_config=_candidate_config(),
        )


def test_baseline_is_independent_of_candidate_hyperparameters() -> None:
    first_config = _candidate_config()
    second_config = _candidate_config()
    second_config["monthly_curve"]["lambda_smooth_month"] = 4.0
    second_config["monthly_curve"]["lambda_smooth_yoy"] = 0.0

    first = _replay_monthly_base_outputs(
        delivery_months=_delivery_months(),
        retained_quotes=_retained_quotes(),
        fold_historical_context=_historical_context(),
        candidate_config=first_config,
    )
    second = _replay_monthly_base_outputs(
        delivery_months=_delivery_months(),
        retained_quotes=_retained_quotes(),
        fold_historical_context=_historical_context(),
        candidate_config=second_config,
    )

    assert first.baseline_monthly_base == second.baseline_monthly_base
    assert first.candidate_config_sha256 != second.candidate_config_sha256


def test_base_replay_anchors_each_delivery_year_independently() -> None:
    delivery = [
        f"{year}-{month:02d}"
        for year in (2027, 2028)
        for month in range(1, 13)
    ]
    retained = pd.DataFrame(
        [
            {
                "market": "CH",
                "load_type": "BASE",
                "product": f"{year}-{month:02d}",
                "price": level + _seasonal_deviation(month),
                "quote_id": f"retained-{year}-{month:02d}",
            }
            for year, level in ((2027, 80.0), (2028, 180.0))
            for month in range(2, 13)
        ]
    )

    output = _replay_monthly_base_outputs(
        delivery_months=delivery,
        retained_quotes=retained,
        fold_historical_context=_historical_context(years=(2027, 2028)),
        candidate_config=_candidate_config(),
    )

    assert output.candidate_monthly_base[0] == pytest.approx(74.5, abs=0.01)
    assert output.candidate_monthly_base[12] == pytest.approx(174.5, abs=0.01)
    assert output.baseline_monthly_base[0] == pytest.approx(74.5, abs=1e-5)
    assert output.baseline_monthly_base[12] == pytest.approx(174.5, abs=1e-5)


def test_base_replay_rejects_delivery_year_without_level_anchor() -> None:
    delivery = [
        f"{year}-{month:02d}"
        for year in (2027, 2028)
        for month in range(1, 13)
    ]

    with pytest.raises(Tier2MonthlyBaseReplayError, match="year=2028"):
        _replay_monthly_base_outputs(
            delivery_months=delivery,
            retained_quotes=_retained_quotes(),
            fold_historical_context=_historical_context(years=(2027, 2028)),
            candidate_config=_candidate_config(),
        )


def test_monthly_smoothing_does_not_bridge_unrepresented_year_levels() -> None:
    delivery = [
        f"{year}-{month:02d}"
        for year in (2027, 2028)
        for month in range(1, 13)
    ]
    retained = pd.DataFrame(
        [
            {
                "market": "CH",
                "load_type": "BASE",
                "product": f"{year}-{month:02d}",
                "price": level + _seasonal_deviation(month),
                "quote_id": f"retained-{year}-{month:02d}",
            }
            for year, level in ((2027, 80.0), (2028, 180.0))
            for month in range(1, 13)
            if (year, month) not in {(2027, 11), (2027, 12), (2028, 1)}
        ]
    )

    output = _replay_monthly_base_outputs(
        delivery_months=delivery,
        retained_quotes=retained,
        fold_historical_context=_historical_context(years=(2027, 2028)),
        candidate_config=_candidate_config(),
    )

    assert output.candidate_monthly_base[10] == pytest.approx(84.5, abs=0.05)
    assert output.candidate_monthly_base[11] == pytest.approx(85.5, abs=0.05)
    assert output.candidate_monthly_base[12] == pytest.approx(174.5, abs=0.05)


@pytest.mark.parametrize("column", ["product", "quote_id"])
def test_base_replay_rejects_ambiguous_retained_identity(column: str) -> None:
    retained = _retained_quotes()
    retained.loc[retained.index[0], column] = 1

    with pytest.raises(Tier2MonthlyBaseReplayError, match="JSON strings"):
        _replay_monthly_base_outputs(
            delivery_months=_delivery_months(),
            retained_quotes=retained,
            fold_historical_context=_historical_context(),
            candidate_config=_candidate_config(),
        )


@pytest.mark.parametrize(
    ("metric", "value"),
    [
        ("max_abs_constraint_residual", float("nan")),
        ("max_abs_constraint_residual", float("inf")),
        ("max_abs_constraint_residual", float("-inf")),
        ("stationarity_residual", float("nan")),
        ("stationarity_residual", float("inf")),
        ("stationarity_residual", float("-inf")),
    ],
)
def test_base_replay_rejects_nonfinite_kkt_diagnostics(
    metric: str, value: float
) -> None:
    kkt: dict[str, object] = {
        "max_abs_constraint_residual": 0.0,
        "stationarity_residual": 0.0,
        "condition_number": 1.0,
        "solved_by_lstsq": False,
        "ridge_used": False,
    }
    kkt[metric] = value
    result = SimpleNamespace(
        monthly_curve=pd.Series([80.0], index=pd.period_range("2027-01", periods=1, freq="M")),
        kkt=kkt,
    )

    with pytest.raises(Tier2MonthlyBaseReplayError, match="non-finite"):
        _validate_solve(
            result,
            config=validate_base_candidate_config(_candidate_config()),
            label="candidate",
        )


def test_base_replay_rejects_duplicate_history_row_hashes() -> None:
    history = _historical_context()
    history.loc[history.index[1], "row_hash"] = history.loc[history.index[0], "row_hash"]

    with pytest.raises(Tier2MonthlyBaseReplayError, match="duplicate row hashes"):
        _replay_monthly_base_outputs(
            delivery_months=_delivery_months(),
            retained_quotes=_retained_quotes(),
            fold_historical_context=history,
            candidate_config=_candidate_config(),
        )


def _candidate_config() -> dict[str, object]:
    return {
        "schema_version": BASE_REPLAY_CONFIG_SCHEMA,
        "monthly_curve": {
            "lambda_prior": 1e-6,
            "lambda_smooth_month": 1.0,
            "lambda_smooth_yoy": 0.25,
            "lambda_shape": 1.0,
            "neighbor_shrinkage": 0.0,
            "robust_panel_quantile": 0.5,
            "min_history_snapshots": 24,
            "max_prior_residual_eur_mwh": None,
            "constraint_tolerance": 1e-9,
            "quote_conflict_tolerance": 0.01,
            "stationarity_tolerance": 1e-7,
        },
    }


def _delivery_months() -> list[str]:
    return [f"2027-{month:02d}" for month in range(1, 13)]


def _seasonal_deviation(month: int) -> float:
    return float(month - 6.5)


def _retained_quotes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "market": "CH",
                "load_type": "BASE",
                "product": f"2027-{month:02d}",
                "price": 80.0 + _seasonal_deviation(month),
                "quote_id": f"retained-{month:02d}",
            }
            for month in range(2, 13)
        ]
    )


def _historical_context(*, years: tuple[int, ...] = (2027,)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for snapshot in range(24):
        date = pd.Timestamp("2023-01-01T08:00:00Z") + pd.DateOffset(months=snapshot)
        snapshot_id = f"snapshot-{snapshot:02d}"
        calendar_price = 70.0 + snapshot * 0.1
        products: list[tuple[str, float]] = []
        for year in years:
            year_level = calendar_price + (year - 2027) * 100.0
            products.append((str(year), year_level))
            products.extend(
                (
                    f"{year}-{month:02d}",
                    year_level + _seasonal_deviation(month),
                )
                for month in range(1, 13)
            )
        for product, price in products:
            rows.append(
                {
                    "date": date.tz_localize(None).normalize(),
                    "available_at": date,
                    "market": "CH",
                    "load_type": "BASE",
                    "product": product,
                    "price": price,
                    "snapshot_id": snapshot_id,
                    "row_hash": f"row-{snapshot:02d}-{product}",
                }
            )
    return pd.DataFrame(rows)
