"""
test_forwards_parser.py
-----------------------
Phase 1ter LT — verify the EEX forward parser:

  1. Cal / Quarter / Month BASE+PEAK products are correctly normalised
     across the 5 market sheets (CH, DE, FR, AT, IT) of a Yearly-style
     workbook.
  2. Week products (3-digit prefix codes such as 205_2026_PEAK,
     307_2018_BASE) are recognised and skipped by default.
  3. Non-market sheets (FX, Produits, HFC) are filtered out
     automatically by ``update_forwards_parquet``.
  4. The historical CH/DE workbook layout (mix of legacy + Week
     codes, 6 years of marks) parses without error.

These tests build XLSX fixtures in memory to avoid any dependency on
the H:\\ shared drive.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pfc_shaping.data.ingest_forwards import (
    _EEX_WEEK_PATTERN,
    _NON_MARKET_SHEETS,
    _PRICE_CEILING_EUR_MWH,
    _PRICE_FLOOR_EUR_MWH,
    _coerce_price,
    _normalize_product,
    load_base_prices_from_eex_report,
    load_forwards_timeseries,
    update_forwards_parquet,
)

# ---------------------------------------------------------------------------
# Pure-regex / classification tests (no Excel I/O)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code, expected",
    [
        ("Y01_2027_BASE", ("2027", "BASE", "Cal")),
        ("Y01_2030_PEAK", ("2030", "PEAK", "Cal")),
        ("Q03_2026_BASE", ("2026-Q3", "BASE", "Quarter")),
        ("Q01_2026_PEAK", ("2026-Q1", "PEAK", "Quarter")),
        ("M02_2027_BASE", ("2027-02", "BASE", "Month")),
        ("M12_2026_PEAK", ("2026-12", "PEAK", "Month")),
        ("205_2026_PEAK", ("Wk205_2026", "PEAK", "Week")),
        ("307_2018_BASE", ("Wk307_2018", "BASE", "Week")),
        ("504_2018_BASE", ("Wk504_2018", "BASE", "Week")),
        # Casing should not matter
        ("y01_2027_base", ("2027", "BASE", "Cal")),
        # Garbage stays None
        ("FX_EUR_CHF", None),
        ("HFC_BENCHMARK", None),
        ("", None),
        # Quarter out of range
        ("Q05_2026_BASE", None),
        # Month out of range
        ("M13_2027_BASE", None),
        # Week starts with 0 → reserved (not matched by [1-9]\d{2})
        ("005_2026_BASE", None),
    ],
)
def test_normalize_product(code: str, expected) -> None:
    assert _normalize_product(code) == expected


def test_week_pattern_is_disjoint_from_cal_q_m() -> None:
    """A Cal/Q/M code must never accidentally match the Week pattern."""
    for code in ["Y01_2027_BASE", "Q03_2026_BASE", "M02_2027_BASE"]:
        assert _EEX_WEEK_PATTERN.match(code) is None
    assert _EEX_WEEK_PATTERN.match("205_2026_PEAK") is not None


def test_non_market_sheets_constant_is_frozen() -> None:
    assert "FX" in _NON_MARKET_SHEETS
    assert "PRODUITS" in _NON_MARKET_SHEETS
    assert "HFC" in _NON_MARKET_SHEETS
    assert isinstance(_NON_MARKET_SHEETS, frozenset)


# ---------------------------------------------------------------------------
# Price coercion : negative prices, "0 = non-quoted" convention, sanity range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        # Empty / NaN cells
        (None, None),
        (float("nan"), None),
        ("", None),
        ("not a number", None),
        # Desk convention: literal 0 = non-quoted
        (0, None),
        (0.0, None),
        (-0.0, None),
        ("0", None),
        # Standard positive marks
        (75.5, 75.5),
        (82, 82.0),
        # Negative prices preserved (real market phenomenon)
        (-12.3, -12.3),
        (-1.0, -1.0),
        # French decimal comma tolerated
        ("75,5", 75.5),
        ("-12,3", -12.3),
        # Sanity range: outside → None
        (_PRICE_FLOOR_EUR_MWH - 1, None),
        (_PRICE_CEILING_EUR_MWH + 1, None),
        (-1_000_000.0, None),
        (1e9, None),
        # Sanity range: inclusive at the boundaries
        (_PRICE_FLOOR_EUR_MWH, _PRICE_FLOOR_EUR_MWH),
        (_PRICE_CEILING_EUR_MWH, _PRICE_CEILING_EUR_MWH),
    ],
)
def test_coerce_price(raw, expected) -> None:
    assert _coerce_price(raw) == expected


def test_load_forwards_timeseries_preserves_negative_prices(tmp_path: Path) -> None:
    """A negative settlement on a Cal/Q/M product (rare today but
    structurally possible in the future) must flow through to the
    timeseries, not be silently dropped like in the legacy ``> 0``
    filter."""
    # Build a minimal sheet where one product has a negative mark on
    # one day, a literal 0 (non-quoted) on another, and a regular
    # positive mark elsewhere.
    dates = pd.date_range("2024-05-01", "2024-05-05", freq="D")
    # 1 product column, 5 dates
    rows: list[list] = []
    rows.append([None, "Y01_2027_BASE"])
    rows.append([None, "ISIN_dummy"])
    rows.append(["Date", None])
    marks = [50.0, -10.5, 0.0, 0.0, 60.0]
    for d, m in zip(dates, marks):
        rows.append([d.strftime("%d.%m.%Y"), m])
    df = pd.DataFrame(rows, columns=range(2))
    path = tmp_path / "negative_test.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="DE", index=False, header=False)

    out = load_forwards_timeseries(path, market="DE")
    # 3 valid rows: 50.0, -10.5, 60.0 (the two zeros are non-quoted)
    assert len(out) == 3, f"expected 3 rows, got {len(out)}"
    assert sorted(out["price"].tolist()) == [-10.5, 50.0, 60.0]
    # Negative price must keep its sign and product type.
    neg_row = out[out["price"] == -10.5].iloc[0]
    assert neg_row["product_type"] == "Cal"
    assert neg_row["load_type"] == "BASE"


def test_load_base_prices_from_eex_report_treats_zero_as_non_quoted(
    tmp_path: Path,
) -> None:
    """When the latest row has 0s (non-quoted) on some products, those
    must NOT appear in base_prices, while real quotes (incl. negatives)
    do appear."""
    dates = pd.date_range("2024-05-01", "2024-05-03", freq="D")
    rows: list[list] = []
    rows.append([None, "Y01_2027_BASE", "Y01_2027_PEAK", "Q03_2026_BASE"])
    rows.append([None, "ISIN_a", "ISIN_b", "ISIN_c"])
    rows.append(["Date", None, None, None])
    # Last row has Cal BASE quoted (-3.5, valid negative), Cal PEAK at
    # 0 (non-quoted), Quarter BASE at 0 (non-quoted).
    rows.append([dates[0].strftime("%d.%m.%Y"), 50.0, 60.0, 55.0])
    rows.append([dates[1].strftime("%d.%m.%Y"), 51.0, 61.0, 56.0])
    rows.append([dates[2].strftime("%d.%m.%Y"), -3.5, 0.0, 0.0])
    df = pd.DataFrame(rows, columns=range(4))
    path = tmp_path / "zero_test.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="DE", index=False, header=False)

    prices = load_base_prices_from_eex_report(path, market="DE")
    # Only the negative Cal BASE survives the as-of selection of the
    # last row.
    assert prices == {"2027": -3.5}


def test_load_base_prices_from_eex_report_rejects_out_of_sanity_range(
    tmp_path: Path,
) -> None:
    """A typo of 9_999_999 in a price cell must NOT pollute base_prices;
    the as-of selection picks an earlier valid row instead."""
    dates = pd.date_range("2024-05-01", "2024-05-03", freq="D")
    rows: list[list] = []
    rows.append([None, "Y01_2027_BASE"])
    rows.append([None, "ISIN_x"])
    rows.append(["Date", None])
    # Latest row is a typo outside the sanity range; previous row is OK.
    rows.append([dates[0].strftime("%d.%m.%Y"), 50.0])
    rows.append([dates[1].strftime("%d.%m.%Y"), 75.0])
    rows.append([dates[2].strftime("%d.%m.%Y"), 9_999_999.0])
    df = pd.DataFrame(rows, columns=range(2))
    path = tmp_path / "typo_test.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="DE", index=False, header=False)

    prices = load_base_prices_from_eex_report(path, market="DE")
    # Typo row rejected; falls back to the latest in-range row (75.0).
    assert prices == {"2027": 75.0}


# ---------------------------------------------------------------------------
# Excel fixture builder
# ---------------------------------------------------------------------------


def _build_market_sheet(
    products: list[str],
    dates: list[pd.Timestamp],
    base_price: float,
) -> pd.DataFrame:
    """Build a sheet that matches the EEX Yearly layout.

    Layout:
        row 0 (raw[0]): blank in col A, product code in cols B..N
        row 1 (raw[1]): blank in col A, ISIN-like string in cols B..N
        row 2 (raw[2]): "Date" in col A, blank in cols B..N
        row 3+ (raw[3..]): date in col A, prices in cols B..N
    """
    n_cols = 1 + len(products)
    rows: list[list] = []
    rows.append([None] + products)
    rows.append([None] + [f"ISIN_{p}" for p in products])
    rows.append(["Date"] + [None] * len(products))
    for i, d in enumerate(dates):
        # Increment per-product so we can distinguish columns in tests.
        marks = [base_price + j * 0.5 + i * 0.01 for j in range(len(products))]
        rows.append([d.strftime("%d.%m.%Y")] + marks)
    return pd.DataFrame(rows, columns=range(n_cols))


def _build_yearly_workbook(tmp_path: Path) -> Path:
    """Build a 5-market Yearly XLSX with Cal/Q/M + Week products and
    extra non-market sheets (FX / Produits / HFC).
    """
    dates = pd.date_range("2024-05-01", "2024-05-10", freq="D")
    market_products = [
        "Y01_2027_BASE", "Y01_2027_PEAK",
        "Q03_2026_BASE", "Q03_2026_PEAK",
        "M02_2027_BASE", "M02_2027_PEAK",
        "205_2026_PEAK",  # Week — must be skipped
        "307_2026_BASE",  # Week — must be skipped
    ]
    path = tmp_path / "Price_Report_EEX_Yearly_TEST.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet in ["DE", "CH", "AT", "IT", "FR"]:
            df = _build_market_sheet(market_products, list(dates), base_price=70.0)
            df.to_excel(writer, sheet_name=sheet, index=False, header=False)
        # Non-market sheets that must be ignored when iterating markets.
        pd.DataFrame({"date": dates, "EUR_CHF": [0.97] * len(dates)}).to_excel(
            writer, sheet_name="FX", index=False
        )
        pd.DataFrame({"product": market_products}).to_excel(
            writer, sheet_name="Produits", index=False
        )
        pd.DataFrame({"hfc": [0.0]}).to_excel(writer, sheet_name="HFC", index=False)
    return path


# ---------------------------------------------------------------------------
# load_forwards_timeseries on a synthetic Yearly workbook
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def yearly_workbook(tmp_path_factory) -> Path:
    return _build_yearly_workbook(tmp_path_factory.mktemp("forwards"))


@pytest.mark.parametrize("market", ["DE", "CH", "AT", "IT", "FR"])
def test_load_forwards_timeseries_drops_week_products(
    market: str, yearly_workbook: Path
) -> None:
    df = load_forwards_timeseries(yearly_workbook, market=market)

    # Cal/Quarter/Month BASE+PEAK = 6 product columns × 10 dates = 60 rows
    assert len(df) == 60, f"{market}: expected 60 rows, got {len(df)}"
    assert set(df["product_type"].unique()) == {"Cal", "Quarter", "Month"}
    assert "Week" not in df["product_type"].unique()
    assert set(df["load_type"].unique()) == {"BASE", "PEAK"}


@pytest.mark.parametrize("market", ["DE", "CH"])
def test_load_forwards_timeseries_can_include_week_products(
    market: str, yearly_workbook: Path
) -> None:
    df = load_forwards_timeseries(yearly_workbook, market=market, include_week=True)
    assert "Week" in df["product_type"].unique()
    n_week = (df["product_type"] == "Week").sum()
    # 2 Week products × 10 dates = 20 rows
    assert n_week == 20, f"{market}: expected 20 Week rows, got {n_week}"


def test_load_base_prices_from_eex_report_skips_week(yearly_workbook: Path) -> None:
    prices = load_base_prices_from_eex_report(yearly_workbook, market="CH")
    # Should contain only Cal/Q/M keys, never Wk*_*
    assert all(not k.startswith("Wk") for k in prices), (
        f"Week products leaked into base_prices: {[k for k in prices if k.startswith('Wk')]}"
    )
    # Expected keys (BASE + PEAK):
    expected = {"2027", "2027-Peak", "2026-Q3", "2026-Q3-Peak", "2027-02", "2027-02-Peak"}
    assert expected.issubset(prices.keys()), f"missing: {expected - set(prices.keys())}"


# ---------------------------------------------------------------------------
# update_forwards_parquet end-to-end with non-market sheet filtering
# ---------------------------------------------------------------------------


def test_update_forwards_parquet_filters_non_market_sheets(
    tmp_path: Path, yearly_workbook: Path
) -> None:
    parquet_out = tmp_path / "forwards_history.parquet"
    # Caller may pass FX/Produits/HFC by mistake — they must be silently dropped.
    requested = ["CH", "DE", "FR", "AT", "IT", "FX", "Produits", "HFC"]
    df = update_forwards_parquet(
        yearly_workbook, parquet_path=parquet_out, markets=requested,
    )
    # Only the 5 real markets should appear, never the non-market tabs.
    assert set(df["market"].unique()) == {"CH", "DE", "FR", "AT", "IT"}
    # 5 markets × 60 rows = 300
    assert len(df) == 300, f"expected 300 rows, got {len(df)}"
    assert "Week" not in df["product_type"].unique()
    assert parquet_out.exists()


def test_update_forwards_parquet_default_markets_are_full_panel(
    tmp_path: Path, yearly_workbook: Path
) -> None:
    parquet_out = tmp_path / "forwards_history_default.parquet"
    df = update_forwards_parquet(yearly_workbook, parquet_path=parquet_out)
    # Default markets list must now cover the full CH/DE/FR/AT/IT panel.
    assert set(df["market"].unique()) == {"CH", "DE", "FR", "AT", "IT"}


def test_update_forwards_parquet_only_non_market_sheets_raises(
    tmp_path: Path, yearly_workbook: Path
) -> None:
    parquet_out = tmp_path / "forwards_history_invalid.parquet"
    with pytest.raises(ValueError, match="No market sheets remain"):
        update_forwards_parquet(
            yearly_workbook,
            parquet_path=parquet_out,
            markets=["FX", "Produits", "HFC"],
        )


# ---------------------------------------------------------------------------
# Historical layout (CH/DE only, mix of Cal/Q/M + Week codes)
# ---------------------------------------------------------------------------


def test_historical_layout_de_ch_with_mixed_codes(tmp_path: Path) -> None:
    """Simulate the Historique2019 workbook: DE+CH only, lots of Week
    codes alongside Cal/Q/M. AT/IT/FR exist as empty sheets."""
    dates = pd.date_range("2019-08-01", "2019-08-15", freq="D")
    history_products = [
        "Y01_2020_BASE",
        "Q01_2019_PEAK",
        "M07_2018_BASE",
        "M09_2018_BASE",
        "M10_2018_BASE",
        # Lots of weekly noise that previously polluted the parser
        "504_2018_BASE",
        "305_2018_BASE",
        "207_2018_PEAK",
        "307_2018_BASE",
    ]
    path = tmp_path / "Price_Report_EEX_CH_DE_Historique_TEST.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet in ["DE", "CH"]:
            df = _build_market_sheet(history_products, list(dates), base_price=55.0)
            df.to_excel(writer, sheet_name=sheet, index=False, header=False)
        # Empty placeholder sheets (the Historique file has them visually).
        for sheet in ["AT", "IT", "FR"]:
            pd.DataFrame([[None]]).to_excel(writer, sheet_name=sheet, index=False, header=False)

    df_de = load_forwards_timeseries(path, market="DE")
    # 5 Cal/Q/M products × 15 dates = 75 rows. 4 Week products are dropped.
    assert len(df_de) == 75, f"DE: expected 75 rows, got {len(df_de)}"
    assert "Week" not in df_de["product_type"].unique()
    assert set(df_de["product_type"].unique()) == {"Cal", "Quarter", "Month"}

    # update_forwards_parquet on the full default panel should not crash on
    # the empty AT/IT/FR sheets — they yield warnings, but the run completes
    # with CH+DE rows.
    parquet_out = tmp_path / "history_parquet.parquet"
    combined = update_forwards_parquet(path, parquet_path=parquet_out)
    assert set(combined["market"].unique()) == {"CH", "DE"}
    assert len(combined) == 150  # 75 × 2 markets
