import importlib

import scripts.eval_lear_governed_forecast_ab as harness

from scripts.eval_lear_governed_forecast_ab import _parse_horizons
from pfc_shaping.data.ct_datasets import resolve_local_cache_path


def test_parse_horizons_prefers_single_horizon_override() -> None:
    assert _parse_horizons("1,5,10", 5) == [5]


def test_parse_horizons_defaults_to_reference_set() -> None:
    assert _parse_horizons(None, None) == [1, 5, 10]


def test_governed_harness_uses_canonical_multi_country_cache(monkeypatch) -> None:
    monkeypatch.setenv("PFC_CT_DATA_ROOT", r"C:\Users\jbattaglia\PFC_CT_DATA")
    reloaded = importlib.reload(harness)
    expected = resolve_local_cache_path(reloaded.ROOT, "ct_forecast_multi_country_15min")
    assert reloaded.DATA_FILES["multi_country_forecast"] == expected
