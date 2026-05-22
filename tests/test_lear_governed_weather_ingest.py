from __future__ import annotations

import requests

from pfc_shaping.data.ingest_weather_forecast import _open_meteo_get


class _FakeResponse:
    def __init__(self, payload: dict[str, object] | None = None) -> None:
        self._payload = payload or {"hourly": {"time": []}}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


def test_open_meteo_retries_insecure_on_ssl_error_by_default(monkeypatch) -> None:
    calls: list[bool | str] = []

    def fake_get(url, params=None, timeout=None, verify=None):  # noqa: ANN001
        calls.append(verify)
        if verify is not False:
            raise requests.exceptions.SSLError("corp proxy cert")
        return _FakeResponse()

    monkeypatch.delenv("PFC_ALLOW_INSECURE_WEATHER_SSL", raising=False)
    monkeypatch.delenv("PFC_FORCE_STRICT_WEATHER_SSL", raising=False)
    monkeypatch.setattr(requests, "get", fake_get)

    response = _open_meteo_get({"latitude": 0.0, "longitude": 0.0})

    assert isinstance(response, _FakeResponse)
    assert calls[-1] is False
    assert len(calls) == 2


def test_open_meteo_strict_mode_disables_insecure_retry(monkeypatch) -> None:
    calls: list[bool | str] = []

    def fake_get(url, params=None, timeout=None, verify=None):  # noqa: ANN001
        calls.append(verify)
        raise requests.exceptions.SSLError("corp proxy cert")

    monkeypatch.setenv("PFC_FORCE_STRICT_WEATHER_SSL", "1")
    monkeypatch.delenv("PFC_ALLOW_INSECURE_WEATHER_SSL", raising=False)
    monkeypatch.setattr(requests, "get", fake_get)

    try:
        _open_meteo_get({"latitude": 0.0, "longitude": 0.0})
    except requests.exceptions.SSLError:
        pass
    else:
        raise AssertionError("Strict mode should not retry insecurely")

    assert len(calls) == 1
