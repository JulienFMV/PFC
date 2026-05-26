from __future__ import annotations

import pandas as pd

from pfc_shaping.pipeline.swiss_short_term import (
    _merge_governed_and_baseline_forecasts,
    _overwrite_routed_day_slice,
    _resolve_governed_applied_horizon_days,
)


def _make_forecast(prices: list[float], p10_shift: float = -5.0, p90_shift: float = 5.0) -> pd.DataFrame:
    ts = pd.date_range("2026-05-01 00:00:00+00:00", periods=len(prices), freq="D")
    days = list(range(1, len(prices) + 1))
    return pd.DataFrame(
        {
            "timestamp": ts,
            "date": [t.tz_convert("Europe/Zurich").date().isoformat() for t in ts],
            "hour": [12] * len(prices),
            "days_ahead": days,
            "price_lear": prices,
            "price_p10": [p + p10_shift for p in prices],
            "price_p90": [p + p90_shift for p in prices],
            "n_windows": [5] * len(prices),
            "mlp_used": [False] * len(prices),
            "fm_used": [True] * len(prices),
            "fm_raw_price": prices,
        }
    )


def test_governed_baseline_blend_caps_boundary_jump() -> None:
    governed = _make_forecast([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    baseline = _make_forecast([100.0, 100.0, 100.0, 100.0, 100.0, 110.0, 120.0])

    merged = _merge_governed_and_baseline_forecasts(
        governed_forecast=governed,
        baseline_forecast=baseline,
        governed_horizon_days=5,
    )

    series = merged.sort_values("days_ahead")["price_lear"].to_numpy()
    max_jump = max(abs(series[i + 1] - series[i]) for i in range(len(series) - 1))
    assert max_jump <= 1.5 + 1e-9


def test_governed_baseline_blend_preserves_interval_ordering() -> None:
    governed = _make_forecast([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0], p10_shift=-3.0, p90_shift=4.0)
    baseline = _make_forecast([99.0, 100.0, 101.0, 102.0, 103.0, 109.0, 115.0], p10_shift=-8.0, p90_shift=9.0)

    merged = _merge_governed_and_baseline_forecasts(
        governed_forecast=governed,
        baseline_forecast=baseline,
        governed_horizon_days=5,
    )

    assert (merged["price_p10"] <= merged["price_lear"]).all()
    assert (merged["price_lear"] <= merged["price_p90"]).all()


def test_overwrite_routed_day_slice_replaces_j1_and_caps_jump_to_j2() -> None:
    routed = _make_forecast([110.0, 100.0, 100.0, 100.0])
    replacement = _make_forecast([90.0, 95.0, 95.0, 95.0])

    merged = _overwrite_routed_day_slice(
        base_forecast=routed,
        replacement_forecast=replacement,
        day_values=[1],
    )

    out = merged.sort_values("days_ahead").reset_index(drop=True)
    assert float(out.loc[0, "price_lear"]) == 90.0
    assert abs(float(out.loc[1, "price_lear"]) - float(out.loc[0, "price_lear"])) <= 1.5 + 1e-9


def test_resolve_governed_applied_horizon_days_caps_at_j7() -> None:
    assert _resolve_governed_applied_horizon_days(10, 10) == 7
    assert _resolve_governed_applied_horizon_days(8, 10) == 7
    assert _resolve_governed_applied_horizon_days(5, 10) == 5


def test_resolve_governed_applied_horizon_days_returns_zero_when_unsupported() -> None:
    assert _resolve_governed_applied_horizon_days(0, 10) == 0
    assert _resolve_governed_applied_horizon_days(-1, 10) == 0
