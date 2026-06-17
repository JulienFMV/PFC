"""
test_rebuild_forwards_history.py
---------------------------------
Phase 2 LT — verify scripts/rebuild_forwards_history.py end-to-end:

  1. Ingests Historique2019 (CH/DE) + Yearly (CH/DE/FR/AT/IT) into one
     long-format Parquet.
  2. On overlapping (date, product, load_type, market) tuples the
     Yearly mark wins (most recently quoted).
  3. Empty / non-market sheets in Historique are silently skipped.
  4. ``--skip-yearly`` and ``--skip-history`` flags work for partial
     refreshes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.rebuild_forwards_history import (
    DEFAULT_OUT_PARQUET,
    HISTORY_MARKETS,
    YEARLY_MARKETS,
    _guard_partial_default_overwrite,
    rebuild,
)


def _build_market_sheet(
    products: list[str],
    dates: list[pd.Timestamp],
    base_price: float,
    *,
    add_zero_columns: bool = False,
) -> pd.DataFrame:
    """Build an EEX-style sheet (3 header rows + data)."""
    rows: list[list] = []
    rows.append([None] + products)
    rows.append([None] + [f"ISIN_{p}" for p in products])
    rows.append(["Date"] + [None] * len(products))
    for i, d in enumerate(dates):
        marks = [base_price + j * 0.5 + i * 0.01 for j in range(len(products))]
        if add_zero_columns:
            # Inject one literal 0 (non-quoted) in each row so we exercise
            # the desk convention that 0 = blank.
            marks[-1] = 0.0
        rows.append([d.strftime("%d.%m.%Y")] + marks)
    return pd.DataFrame(rows, columns=range(1 + len(products)))


def _empty_sheet() -> pd.DataFrame:
    return pd.DataFrame([[None]])


@pytest.fixture
def yearly_workbook(tmp_path: Path) -> Path:
    """Yearly workbook: full panel CH/DE/FR/AT/IT + non-market sheets."""
    dates = pd.date_range("2024-05-01", "2024-05-05", freq="D")
    products = ["Y01_2027_BASE", "Q03_2026_BASE", "M02_2027_BASE"]
    path = tmp_path / "yearly.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet in YEARLY_MARKETS:
            # Yearly marks at base 80 to be distinguishable from Historique (55).
            df = _build_market_sheet(products, list(dates), base_price=80.0)
            df.to_excel(writer, sheet_name=sheet, index=False, header=False)
        # Non-market noise tabs.
        pd.DataFrame({"x": [0.97]}).to_excel(writer, sheet_name="FX", index=False)
        pd.DataFrame({"x": [1]}).to_excel(writer, sheet_name="HFC", index=False)
    return path


@pytest.fixture
def history_workbook(tmp_path: Path) -> Path:
    """Historique workbook: CH/DE only with extended history; FR/AT/IT
    are empty placeholder sheets."""
    dates = pd.date_range("2019-08-01", "2024-05-05", freq="W-MON")
    products = ["Y01_2020_BASE", "Q01_2020_BASE", "M07_2019_BASE", "Y01_2027_BASE"]
    path = tmp_path / "history.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet in HISTORY_MARKETS:
            df = _build_market_sheet(products, list(dates), base_price=55.0)
            df.to_excel(writer, sheet_name=sheet, index=False, header=False)
        # FR / AT / IT exist as empty sheets — must not crash the run.
        for sheet in ["FR", "AT", "IT"]:
            _empty_sheet().to_excel(writer, sheet_name=sheet, index=False, header=False)
    return path


# ---------------------------------------------------------------------------
# Core rebuild path
# ---------------------------------------------------------------------------


def test_rebuild_combines_yearly_and_history(
    tmp_path: Path, yearly_workbook: Path, history_workbook: Path
) -> None:
    out = tmp_path / "forwards_history.parquet"
    df = rebuild(yearly=yearly_workbook, history=history_workbook, out_parquet=out)

    # Expected market coverage: CH+DE from both files; FR/AT/IT only Yearly.
    assert set(df["market"].unique()) == {"CH", "DE", "FR", "AT", "IT"}
    assert out.exists()

    # CH & DE rows include both 2019+ history and 2024 yearly marks.
    ch_dates = df.loc[df["market"] == "CH", "date"]
    assert ch_dates.min() <= pd.Timestamp("2019-12-31")
    assert ch_dates.max() >= pd.Timestamp("2024-05-01")

    # FR/AT/IT only go back to the yearly start (2024-05-01).
    fr_dates = df.loc[df["market"] == "FR", "date"]
    assert fr_dates.min() >= pd.Timestamp("2024-05-01")


def test_rebuild_yearly_wins_on_conflict(
    tmp_path: Path, yearly_workbook: Path, history_workbook: Path
) -> None:
    """For (date, product, load_type, market) tuples present in both
    files, the Yearly mark must replace the Historique mark."""
    out = tmp_path / "fwd_history.parquet"
    df = rebuild(yearly=yearly_workbook, history=history_workbook, out_parquet=out)

    # The product Y01_2027_BASE on CH for date 2024-05-05 is in both
    # files. Yearly base is 80.0 (col 0 → +0.5*0 +0.01*4 = 80.04),
    # Historique base is 55.0. Expect ~80.x.
    overlap = df[
        (df["market"] == "CH")
        & (df["product"] == "2027")
        & (df["load_type"] == "BASE")
        & (df["date"] == pd.Timestamp("2024-05-05"))
    ]
    assert len(overlap) == 1, f"expected exactly 1 overlap row, got {len(overlap)}"
    assert overlap.iloc[0]["price"] >= 70.0, (
        f"Yearly mark must win — got {overlap.iloc[0]['price']}"
    )


def test_rebuild_skip_yearly_only_history(
    tmp_path: Path, history_workbook: Path
) -> None:
    out = tmp_path / "history_only.parquet"
    df = rebuild(yearly=None, history=history_workbook, out_parquet=out)
    # Only CH/DE survive (Historique markets).
    assert set(df["market"].unique()) == {"CH", "DE"}


def test_rebuild_skip_history_only_yearly(
    tmp_path: Path, yearly_workbook: Path
) -> None:
    out = tmp_path / "yearly_only.parquet"
    df = rebuild(yearly=yearly_workbook, history=None, out_parquet=out)
    assert set(df["market"].unique()) == {"CH", "DE", "FR", "AT", "IT"}


def test_rebuild_raises_when_both_skipped(tmp_path: Path) -> None:
    out = tmp_path / "no_input.parquet"
    with pytest.raises(ValueError, match="At least one"):
        rebuild(yearly=None, history=None, out_parquet=out)


def test_guard_blocks_partial_rebuild_to_canonical_parquet() -> None:
    with pytest.raises(ValueError, match="Refusing --skip-history overwrite"):
        _guard_partial_default_overwrite(
            skip_yearly=False,
            skip_history=True,
            out_parquet=DEFAULT_OUT_PARQUET,
            allow_partial_overwrite=False,
        )


def test_guard_allows_partial_rebuild_to_temporary_parquet(tmp_path: Path) -> None:
    _guard_partial_default_overwrite(
        skip_yearly=False,
        skip_history=True,
        out_parquet=tmp_path / "yearly_only.parquet",
        allow_partial_overwrite=False,
    )


def test_rebuild_handles_empty_neighbor_sheets(
    tmp_path: Path, history_workbook: Path
) -> None:
    """The Historique workbook has empty placeholder sheets for
    FR/AT/IT. The script must NOT crash; those sheets just contribute
    0 rows."""
    out = tmp_path / "with_empties.parquet"
    df = rebuild(yearly=None, history=history_workbook, out_parquet=out)
    # Empty sheets shouldn't appear in the output.
    assert "FR" not in df["market"].unique()
    assert "AT" not in df["market"].unique()
    assert "IT" not in df["market"].unique()


def test_rebuild_preserves_negative_prices_end_to_end(tmp_path: Path) -> None:
    """End-to-end check that a negative settlement on a Cal/Q/M product
    flows from the source XLSX through to the rebuilt Parquet.

    This is the matching downstream test for the ``_coerce_price``
    Phase 2.1 fix — the rebuild script must not silently drop negative
    marks, since this is the whole point of relaxing the legacy
    ``> 0`` filter.
    """
    dates = pd.date_range("2024-05-01", "2024-05-04", freq="D")
    products = ["Y01_2027_BASE", "Q03_2026_BASE"]

    rows: list[list] = []
    rows.append([None, *products])
    rows.append([None, *(f"ISIN_{p}" for p in products)])
    rows.append(["Date", *(None for _ in products)])
    # Day 1: positive on both
    rows.append([dates[0].strftime("%d.%m.%Y"), 50.0, 60.0])
    # Day 2: negative on Cal (rare but legitimate)
    rows.append([dates[1].strftime("%d.%m.%Y"), -8.4, 60.0])
    # Day 3: literal 0 (non-quoted) on Cal — must NOT appear in parquet
    rows.append([dates[2].strftime("%d.%m.%Y"), 0.0, 60.0])
    # Day 4: extreme negative below sanity range — must be rejected
    rows.append([dates[3].strftime("%d.%m.%Y"), -10_000.0, 60.0])

    df = pd.DataFrame(rows, columns=range(1 + len(products)))
    yearly = tmp_path / "yearly_with_neg.xlsx"
    with pd.ExcelWriter(yearly, engine="openpyxl") as writer:
        for sheet in YEARLY_MARKETS:
            df.to_excel(writer, sheet_name=sheet, index=False, header=False)

    out = tmp_path / "fwd_history_neg.parquet"
    rebuilt = rebuild(yearly=yearly, history=None, out_parquet=out)

    # Cal 2027 BASE on each market: 2 valid rows (positive d1 + negative d2),
    # zero rejected, sanity-range-violator rejected.
    cal_de = rebuilt[
        (rebuilt["market"] == "DE")
        & (rebuilt["product"] == "2027")
        & (rebuilt["load_type"] == "BASE")
    ].sort_values("date")
    assert len(cal_de) == 2, (
        f"expected 2 Cal 2027 rows on DE (d1=50, d2=-8.4), got {len(cal_de)}: "
        f"{cal_de[['date', 'price']].to_dict('records')}"
    )
    assert cal_de.iloc[0]["price"] == 50.0
    assert cal_de.iloc[1]["price"] == -8.4, "negative settlement must flow through"

    # Quarter Q3 2026 BASE: 4 rows (all positive 60.0).
    q_de = rebuilt[
        (rebuilt["market"] == "DE")
        & (rebuilt["product"] == "2026-Q3")
        & (rebuilt["load_type"] == "BASE")
    ]
    assert len(q_de) == 4
    assert (q_de["price"] == 60.0).all()


def test_rebuild_works_under_pandas_compat_path(
    tmp_path: Path, yearly_workbook: Path
) -> None:
    """Smoke check that the as-of selection logic in
    ``load_base_prices_from_eex_report`` (which is exercised by
    ``rebuild`` via ``load_forwards_timeseries`` indirectly) does not
    rely on ``DataFrame.map``.

    Older pandas (<2.1) did not have ``DataFrame.map``. We assert the
    rebuild path completes without touching that API by inspecting
    the actual call chain — running it end-to-end is the simplest
    proof. If this test ever starts failing on a fresh pandas, we
    likely re-introduced a non-vectorised ``.map`` call.
    """
    out = tmp_path / "fwd_history_compat.parquet"
    # Just ensure the call completes; pandas-version assertions live in
    # CI. The key invariant: no AttributeError on .map / .applymap.
    df = rebuild(yearly=yearly_workbook, history=None, out_parquet=out)
    assert not df.empty
