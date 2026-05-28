import pandas as pd

from scripts.eval_ct_governed_replication import _derive_default_cutoffs, _parse_cutoffs, _replication_label, _wilson_interval


def test_parse_cutoffs_adds_end_of_day_utc() -> None:
    epex = pd.DataFrame(index=pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC"))
    first_governed = pd.Timestamp("2025-10-01", tz="UTC")
    cutoffs = _parse_cutoffs("2026-01-31,2026-02-28", epex, first_governed, 2, "ME")
    assert cutoffs == [
        pd.Timestamp("2026-01-31 23:45:00+00:00"),
        pd.Timestamp("2026-02-28 23:45:00+00:00"),
    ]


def test_derive_default_cutoffs_respects_first_governed_ts() -> None:
    epex = pd.DataFrame(
        index=pd.date_range("2025-09-01", "2026-03-31 23:45:00", freq="15min", tz="UTC")
    )
    first_governed = pd.Timestamp("2025-10-01", tz="UTC")
    cutoffs = _derive_default_cutoffs(epex, first_governed, n_periods=3, freq="ME")
    assert cutoffs == [
        pd.Timestamp("2026-01-31 23:45:00+00:00"),
        pd.Timestamp("2026-02-28 23:45:00+00:00"),
        pd.Timestamp("2026-03-31 23:45:00+00:00"),
    ]


def test_replication_label_uses_local_date() -> None:
    cutoff = pd.Timestamp("2026-03-31 23:45:00+00:00")
    assert _replication_label(cutoff) == "20260401"


def test_wilson_interval_bounds_are_ordered() -> None:
    low, high = _wilson_interval(5, 8)
    assert 0.0 <= low <= high <= 1.0
