from scripts.eval_lear_governed_forecast_ab import _parse_horizons


def test_parse_horizons_prefers_single_horizon_override() -> None:
    assert _parse_horizons("1,5,10", 5) == [5]


def test_parse_horizons_defaults_to_reference_set() -> None:
    assert _parse_horizons(None, None) == [1, 5, 10]
