"""Regression tests for DST-safe Zurich localization.

Both `scripts/extract_deviwa_cache.py` and `demo_deviwa/utils.py` ship a
helper that resolves the autumn DST duplicate hour with an explicit
``ambiguous`` array, instead of dropping it via ``ambiguous="NaT"``.
This test pins the contract :

  - autumn DST day → 25 physical hours preserved (100 quarter-hours)
  - duplicate naive timestamps in the duplicated hour are mapped to
    CEST (1st occurrence) then CET (2nd) so the resulting tz-aware
    index is strictly monotonic in UTC
  - normal (non-DST) days are unchanged
  - already tz-aware input is converted to Zurich without re-localizing
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo_deviwa"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from utils import _localize_zurich_safe  # noqa: E402
from extract_deviwa_cache import _localize_zurich_no_shift  # noqa: E402

# 2026 autumn DST in Europe/Zurich : Sun 25 Oct 2026, clocks go from
# 03:00 CEST → 02:00 CET (the 02:00–03:00 hour is duplicated).
AUTUMN_DST_DATE = "2026-10-25"


def _autumn_dst_quarter_hours() -> pd.DatetimeIndex:
    """The 100 naive quarter-hours of the autumn DST day, with the
    02:00-02:45 block duplicated as the source would write it."""
    base = pd.date_range(
        start=f"{AUTUMN_DST_DATE} 00:00",
        end=f"{AUTUMN_DST_DATE} 23:45",
        freq="15min",
    )
    # Duplicate the 02:00, 02:15, 02:30, 02:45 quarter-hours
    dup_block = pd.DatetimeIndex([
        pd.Timestamp(f"{AUTUMN_DST_DATE} 02:00"),
        pd.Timestamp(f"{AUTUMN_DST_DATE} 02:15"),
        pd.Timestamp(f"{AUTUMN_DST_DATE} 02:30"),
        pd.Timestamp(f"{AUTUMN_DST_DATE} 02:45"),
    ])
    # Insert the duplicates right after the original 02:00–02:45 block
    insert_at = base.searchsorted(pd.Timestamp(f"{AUTUMN_DST_DATE} 03:00"))
    return base[:insert_at].append(dup_block).append(base[insert_at:])


@pytest.mark.parametrize("helper", [_localize_zurich_safe, _localize_zurich_no_shift])
def test_autumn_dst_preserves_25_hours(helper):
    naive = _autumn_dst_quarter_hours()
    assert len(naive) == 100, "fixture sanity: 100 naive quarter-hours"

    localized = helper(naive)
    assert localized.tz is not None
    # No NaT — every quarter-hour is preserved
    assert not pd.isna(localized).any()
    # Strictly monotonic in UTC after localization
    utc = pd.DatetimeIndex(localized).tz_convert("UTC")
    assert utc.is_monotonic_increasing
    # The autumn DST day spans 25 hours = 100 × 15min — confirm via duration
    span = utc.max() - utc.min()
    assert span == pd.Timedelta(hours=24, minutes=45)
    # The CEST/CET split lands on the right quarter-hours
    cest_count = sum(1 for ts in localized if ts.utcoffset() == pd.Timedelta(hours=2))
    cet_count = sum(1 for ts in localized if ts.utcoffset() == pd.Timedelta(hours=1))
    # Before transition : 00:00 → 02:45 CEST = 12 quarter-hours
    # After transition  : 02:00 → 23:45 CET  = 88 quarter-hours
    assert cest_count == 12
    assert cet_count == 88


@pytest.mark.parametrize("helper", [_localize_zurich_safe, _localize_zurich_no_shift])
def test_normal_day_unchanged(helper):
    naive = pd.date_range("2026-06-15 00:00", "2026-06-15 23:45", freq="15min")
    localized = helper(naive)
    assert localized.tz is not None
    assert len(localized) == len(naive)
    # All summer day → CEST (UTC+2)
    assert all(ts.utcoffset() == pd.Timedelta(hours=2) for ts in localized)


@pytest.mark.parametrize("helper", [_localize_zurich_safe, _localize_zurich_no_shift])
def test_already_tz_aware_passes_through(helper):
    aware = pd.date_range(
        "2026-01-15 00:00", "2026-01-15 02:00", freq="15min", tz="UTC"
    )
    out = helper(aware)
    assert str(out.tz) == "Europe/Zurich"
    # Same instant, just a different wall clock
    assert (out.tz_convert("UTC") == aware).all()


def test_helper_accepts_series():
    """The Series overload is what `load_deviwa_programme_cached` relies on."""
    naive = _autumn_dst_quarter_hours()
    s = pd.Series(naive)
    out = _localize_zurich_safe(s)
    assert isinstance(out, pd.Series)
    assert out.dt.tz is not None
    assert len(out) == 100
    assert not out.isna().any()
