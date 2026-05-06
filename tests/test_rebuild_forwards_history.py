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
    HISTORY_MARKETS,
    YEARLY_MARKETS,
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
