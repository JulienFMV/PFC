from __future__ import annotations

from pathlib import Path

from pfc_shaping.data.ct_datasets import get_ct_data_root, resolve_local_cache_path
from pfc_shaping.pipeline.rolling_update import _resolve_config_path


def test_ct_data_root_defaults_to_project_root(monkeypatch) -> None:
    monkeypatch.delenv("PFC_CT_DATA_ROOT", raising=False)
    project_root = Path(r"C:\tmp\pfc_project")
    assert get_ct_data_root(project_root) == project_root.resolve()


def test_ct_data_root_uses_environment_override(monkeypatch) -> None:
    monkeypatch.setenv("PFC_CT_DATA_ROOT", r"C:\Users\jbattaglia\PFC_CT_DATA")
    assert get_ct_data_root(Path(r"H:\repo")) == Path(r"C:\Users\jbattaglia\PFC_CT_DATA").resolve()


def test_resolve_local_cache_path_uses_ct_data_root(monkeypatch) -> None:
    monkeypatch.setenv("PFC_CT_DATA_ROOT", r"C:\Users\jbattaglia\PFC_CT_DATA")
    path = resolve_local_cache_path(Path(r"H:\repo"), "ct_forecast_multi_country_15min")
    expected = Path(r"C:\Users\jbattaglia\PFC_CT_DATA\pfc_shaping\data\multi_country_forecast_15min.parquet").resolve()
    assert path == expected


def test_resolve_config_path_routes_relative_data_paths_into_ct_cache(monkeypatch) -> None:
    monkeypatch.setenv("PFC_CT_DATA_ROOT", r"C:\Users\jbattaglia\PFC_CT_DATA")
    path = _resolve_config_path("data/entso_15min.parquet")
    expected = Path(r"C:\Users\jbattaglia\PFC_CT_DATA\pfc_shaping\data\entso_15min.parquet").resolve()
    assert path == expected
