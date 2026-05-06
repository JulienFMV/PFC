"""
test_country_tz_plumbing.py
---------------------------
Audit Bloc A — verify that country / timezone are propagated correctly
through the LT pipeline (assembler + arbitrage_free calibrator), so
each market's PFC uses its own holidays calendar and local tz instead
of silently defaulting to CH/Europe-Zurich.

Findings addressed:
  #1  arbitrage_free.calibrate ignored country (CH-only peak mask)
  #2  assembler._resolve_base / _compute_f_S hardcoded Europe/Zurich
  #3  assembler._check_energy_consistency hardcoded Europe/Zurich
  #10 _is_peak_timestamp Series-with-index broke on DST duplicates
  #23 cross-cutting tz inconsistency CH-only

These tests do NOT exercise the full PFC build (which needs fitted
artifacts) — they target the helpers in isolation, which is enough to
trap a regression that re-introduces a CH-only path.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.calibration.arbitrage_free import (
    _COUNTRY_TZ,
    _build_peak_mask,
    _get_country_holidays,
    _get_ch_holidays,
)
from pfc_shaping.lt.model.assembler import (
    _COUNTRY_LOCAL_TZ,
    PFCAssembler,
    _country_holidays,
    _country_local_tz,
)


# ---------------------------------------------------------------------------
# Country → timezone tables coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "country, expected_tz",
    [
        ("CH", "Europe/Zurich"),
        ("DE", "Europe/Berlin"),
        ("AT", "Europe/Vienna"),
        ("FR", "Europe/Paris"),
        ("IT", "Europe/Rome"),
    ],
)
def test_country_tz_table_covers_full_panel(country: str, expected_tz: str) -> None:
    assert _COUNTRY_LOCAL_TZ[country] == expected_tz
    assert _COUNTRY_TZ[country] == expected_tz
    assert _country_local_tz(country) == expected_tz


def test_country_tz_assembler_and_calibrator_agree() -> None:
    """The two tz tables (assembler + calibrator) must agree, otherwise
    the assembler's idx_local and the calibrator's peak mask diverge."""
    assert _COUNTRY_LOCAL_TZ == _COUNTRY_TZ


def test_country_tz_unknown_falls_back_to_zurich_with_warning(caplog) -> None:
    with caplog.at_level("WARNING"):
        tz = _country_local_tz("XX")
    assert tz == "Europe/Zurich"
    assert any("Unknown country" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# Holidays calendars
# ---------------------------------------------------------------------------


def test_holidays_calendars_are_country_specific() -> None:
    """Bundesfeier (1 August) is a CH-only national holiday; Tag der
    Deutschen Einheit (3 October) is a DE-only national holiday. Each
    calendar must contain its own and not the other's."""
    ch = _country_holidays([2024], country="CH")
    de = _country_holidays([2024], country="DE")
    bundesfeier = dt.date(2024, 8, 1)
    deutsche_einheit = dt.date(2024, 10, 3)
    assert bundesfeier in ch, "Bundesfeier missing from CH calendar"
    assert deutsche_einheit not in ch, "Deutsche Einheit must NOT be in CH"
    assert deutsche_einheit in de, "Deutsche Einheit missing from DE calendar"
    assert bundesfeier not in de, "Bundesfeier must NOT be in DE"


def test_calibrator_holidays_country_aware() -> None:
    """The calibration-side helper must mirror the assembler-side helper."""
    ch = _get_country_holidays([2024], country="CH")
    de = _get_country_holidays([2024], country="DE")
    fr = _get_country_holidays([2024], country="FR")
    it = _get_country_holidays([2024], country="IT")
    at = _get_country_holidays([2024], country="AT")

    # Bastille Day (FR), Festa della Repubblica (IT), Nationalfeiertag (AT)
    assert dt.date(2024, 7, 14) in fr
    assert dt.date(2024, 6, 2) in it
    assert dt.date(2024, 10, 26) in at
    # And confirm they don't bleed into CH/DE
    assert dt.date(2024, 7, 14) not in ch
    assert dt.date(2024, 6, 2) not in de


def test_legacy_get_ch_holidays_alias_still_works() -> None:
    """Old call sites use ``_get_ch_holidays`` directly; keep them green."""
    legacy = _get_ch_holidays([2024])
    modern = _get_country_holidays([2024], country="CH")
    assert legacy == modern


def test_calibrator_unsupported_country_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported country"):
        _get_country_holidays([2024], country="XX")


# ---------------------------------------------------------------------------
# _build_peak_mask: tz-aware behaviour
# ---------------------------------------------------------------------------


def _build_index(start: str, end: str, tz: str = "UTC") -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="15min", tz=tz, inclusive="left")


def test_peak_mask_excludes_country_specific_holidays() -> None:
    """For 2024-08-01 (Bundesfeier, a Thursday): CH peak mask = all False
    over the day; DE peak mask = full Peak (08-20) for that day."""
    idx = _build_index("2024-07-31", "2024-08-03", tz="UTC")
    ch_hols = _get_country_holidays([2024], country="CH")
    de_hols = _get_country_holidays([2024], country="DE")

    ch_mask = _build_peak_mask(idx, peak_hours=(8, 20), holiday_dates=ch_hols, tz="Europe/Zurich")
    de_mask = _build_peak_mask(idx, peak_hours=(8, 20), holiday_dates=de_hols, tz="Europe/Berlin")

    # Index in local time
    idx_zh = idx.tz_convert("Europe/Zurich")
    is_aug1 = idx_zh.date == dt.date(2024, 8, 1)
    is_peak_window = (idx_zh.hour >= 8) & (idx_zh.hour < 20)
    is_aug1_peak_window = is_aug1 & is_peak_window

    # On Aug 1 during peak hours: CH excludes (Bundesfeier), DE includes.
    assert ch_mask[is_aug1_peak_window].sum() == 0, "CH wrongly marks Bundesfeier as peak"
    assert de_mask[is_aug1_peak_window].sum() > 0, "DE wrongly marks 1 Aug as non-peak"


def test_peak_mask_excludes_de_specific_holiday() -> None:
    """3 October 2024 (Thursday) is Tag der Deutschen Einheit (DE only)."""
    idx = _build_index("2024-10-02", "2024-10-05", tz="UTC")
    ch_hols = _get_country_holidays([2024], country="CH")
    de_hols = _get_country_holidays([2024], country="DE")

    ch_mask = _build_peak_mask(idx, peak_hours=(8, 20), holiday_dates=ch_hols, tz="Europe/Zurich")
    de_mask = _build_peak_mask(idx, peak_hours=(8, 20), holiday_dates=de_hols, tz="Europe/Berlin")

    idx_be = idx.tz_convert("Europe/Berlin")
    is_oct3 = idx_be.date == dt.date(2024, 10, 3)
    is_peak_window = (idx_be.hour >= 8) & (idx_be.hour < 20)
    is_oct3_peak_window = is_oct3 & is_peak_window

    assert de_mask[is_oct3_peak_window].sum() == 0, "DE wrongly marks Deutsche Einheit as peak"
    assert ch_mask[is_oct3_peak_window].sum() > 0, "CH wrongly marks 3 Oct as non-peak"


# ---------------------------------------------------------------------------
# _is_peak_timestamp: DST robustness (finding #10)
# ---------------------------------------------------------------------------


def test_is_peak_timestamp_survives_autumnal_dst_transition() -> None:
    """At the autumnal DST transition (Europe/Zurich, last Sunday of
    October at 02:00→02:00), the local index has duplicated timestamps.
    The legacy ``pd.Series(..., index=idx_local).isin(...)`` raised on
    duplicate index in some pandas versions. The vectorised numpy
    version must run cleanly."""
    # Build a UTC index spanning the 2024 fall back transition (27 Oct).
    idx_utc = _build_index("2024-10-26", "2024-10-28", tz="UTC")
    idx_local = idx_utc.tz_convert("Europe/Zurich")

    mask = PFCAssembler._is_peak_timestamp(idx_local, country="CH")
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (len(idx_local),)
    assert mask.dtype == bool


def test_is_peak_timestamp_country_panel() -> None:
    """All 5 markets must yield a bool mask of the right shape — guards
    the FR/AT/IT activation against a regression on the staticmethod."""
    idx = _build_index("2024-07-15", "2024-07-22", tz="UTC")
    for country, tz in _COUNTRY_LOCAL_TZ.items():
        idx_local = idx.tz_convert(tz)
        mask = PFCAssembler._is_peak_timestamp(idx_local, country=country)
        assert mask.shape == (len(idx_local),), f"{country}: bad shape"
        # Sanity: peak rate should be in [0.20, 0.45] for a normal week.
        rate = mask.sum() / len(mask)
        assert 0.20 <= rate <= 0.45, f"{country}: implausible peak rate {rate:.3f}"


# ---------------------------------------------------------------------------
# Calibrator country argument (finding #1)
# ---------------------------------------------------------------------------


def test_calibrator_calibrate_signature_accepts_country() -> None:
    """``calibrate()`` must expose ``country`` as a kwarg so the
    assembler can route the market code through to the peak mask."""
    import inspect
    from pfc_shaping.calibration.arbitrage_free import ArbitrageFreeCalibrator

    sig = inspect.signature(ArbitrageFreeCalibrator.calibrate)
    assert "country" in sig.parameters, "calibrate() must accept a country kwarg"
    # Default kept as 'CH' for backward compat.
    assert sig.parameters["country"].default == "CH"


def test_calibrator_rejects_unknown_country() -> None:
    """A typo'd country must fail loudly, not silently fall back to CH."""
    from pfc_shaping.calibration.arbitrage_free import ArbitrageFreeCalibrator

    cal = ArbitrageFreeCalibrator()
    s = pd.Series(
        np.linspace(50, 80, 96),
        index=_build_index("2024-01-01", "2024-01-02", tz="UTC"),
    )
    with pytest.raises(ValueError, match="Unsupported country"):
        cal.calibrate(s, contracts=[], country="XX")
