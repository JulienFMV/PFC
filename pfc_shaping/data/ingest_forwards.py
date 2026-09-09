"""
ingest_forwards.py
------------------
Parsing déterministe des rapports EEX matérialisés en amont.

Ces forwards constituent les niveaux de base B (â‚¬/MWh) utilisÃ©s par
l'assembleur PFC. EULER les calibre sur les prix EEX liquides ; notre
modÃ¨le applique par-dessus la forme (shape) 15min.

Format de sortie :
    dict[str, float] â€” clÃ©s : '2025', '2025-Q1', '2025-03', etc.
    Directement compatible avec PFCAssembler.build(base_prices=...)

L'acquisition Databricks reste hors du runtime gouverné et vit dans
``pfc_shaping.data.forward_acquisition_databricks``.
"""

from __future__ import annotations

import logging
import re
from io import BytesIO
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


_EEX_BASE_PATTERN = re.compile(r"^(Y01|Q\d{2}|M\d{2})_(\d{4})_BASE$")
_EEX_PRODUCT_PATTERN = re.compile(r"^(Y01|Q\d{2}|M\d{2})_(\d{4})_(BASE|PEAK)$")
# Week products: 3-digit prefix not starting with 0 (e.g. 205_2026_PEAK,
# 307_2018_BASE, 504_2018_BASE). The leading digit/sub-digits encode the
# week index in EEX desk codes. The PFC long-term curve does not consume
# weekly products — they are recognised here only to be filtered out
# explicitly instead of falling silently into the "other / unknown" bucket.
_EEX_WEEK_PATTERN = re.compile(r"^([1-9]\d{2})_(\d{4})_(BASE|PEAK)$")

# Workbook tabs that are *not* market sheets in the EEX daily / yearly
# reports: FX rates, internal product catalogue, HFC benchmark. They must
# never be parsed as forward markets.
_NON_MARKET_SHEETS: frozenset[str] = frozenset({"FX", "PRODUITS", "HFC"})

# Sanity bounds on EEX forward marks (EUR/MWh).
#   - Lower bound: power forwards have never settled below ~−100 EUR/MWh on
#     a Cal/Q/M product, but Day/Week products did dip lower in stressed
#     2022/2024 conditions. We pick a permissive [-500, +10_000] window:
#     anything outside is almost certainly a data error (cell typo, unit
#     mismatch).
#   - The desk convention is to fill non-quoted cells with literal ``0``,
#     so 0.0 is treated as "not quoted today" and dropped, NOT as a valid
#     zero settlement. This matches the Phase 0 snapshot where every
#     market sheet had thousands of literal zeros.
_PRICE_FLOOR_EUR_MWH: float = -500.0
_PRICE_CEILING_EUR_MWH: float = 10_000.0


def _coerce_price(raw: object) -> float | None:
    """Convert a raw cell value into a numeric price, applying the desk
    convention (``0`` = non-quoted) and the sanity range.

    Returns ``None`` if the cell is empty / NaN / non-numeric / equals
    zero (non-quoted), or falls outside ``[_PRICE_FLOOR_EUR_MWH,
    _PRICE_CEILING_EUR_MWH]``. Otherwise returns the float price.

    A negative price (e.g. ``-12.4``) is preserved — power markets do
    settle negative on shorter products and may eventually do so on
    Cal/Q/M as renewable penetration increases.
    """
    if raw is None:
        return None
    if isinstance(raw, float) and (raw != raw):  # NaN
        return None
    try:
        # Tolerate French decimal comma in legacy XLSX cells.
        price = float(str(raw).replace(",", ".").strip())
    except (TypeError, ValueError):
        return None
    # Desk convention: literal 0 = non-quoted. We deliberately do NOT
    # widen this to include exactly +0.0 or -0.0 differently — both map
    # to "blank cell".
    if price == 0.0:
        return None
    if price < _PRICE_FLOOR_EUR_MWH or price > _PRICE_CEILING_EUR_MWH:
        return None
    return price


def _normalize_delivery_period(eex_code: str) -> str | None:
    """
    Convert EEX code format to internal delivery key format.

    Examples:
        Y01_2027_BASE -> 2027
        Q03_2026_BASE -> 2026-Q3
        M04_2026_BASE -> 2026-04
    """
    m = _EEX_BASE_PATTERN.match(eex_code.strip().upper())
    if not m:
        return None

    prefix, year_str = m.groups()
    year = int(year_str)

    if prefix == "Y01":
        return f"{year}"

    if prefix.startswith("Q"):
        quarter = int(prefix[1:])
        if 1 <= quarter <= 4:
            return f"{year}-Q{quarter}"
        return None

    if prefix.startswith("M"):
        month = int(prefix[1:])
        if 1 <= month <= 12:
            return f"{year}-{month:02d}"
        return None

    return None


def _normalize_product(eex_code: str) -> tuple[str, str, str] | None:
    """Parse an EEX desk code into (delivery_key, load_type, product_type).

    Returns ``None`` for codes that do not match any recognised template.

    Examples:
        Y01_2027_BASE -> ('2027', 'BASE', 'Cal')
        Q03_2026_PEAK -> ('2026-Q3', 'PEAK', 'Quarter')
        M04_2026_BASE -> ('2026-04', 'BASE', 'Month')
        205_2026_PEAK -> ('Wk205_2026', 'PEAK', 'Week')
        307_2018_BASE -> ('Wk307_2018', 'BASE', 'Week')
        FX_EUR_CHF    -> None
    """
    code = eex_code.strip().upper()

    m = _EEX_PRODUCT_PATTERN.match(code)
    if m is not None:
        prefix, year_str, load_type = m.groups()
        year = int(year_str)
        if prefix == "Y01":
            return (f"{year}", load_type, "Cal")
        if prefix.startswith("Q"):
            quarter = int(prefix[1:])
            if 1 <= quarter <= 4:
                return (f"{year}-Q{quarter}", load_type, "Quarter")
            return None
        if prefix.startswith("M"):
            month = int(prefix[1:])
            if 1 <= month <= 12:
                return (f"{year}-{month:02d}", load_type, "Month")
            return None
        return None

    m_wk = _EEX_WEEK_PATTERN.match(code)
    if m_wk is not None:
        prefix, year_str, load_type = m_wk.groups()
        return (f"Wk{prefix}_{year_str}", load_type, "Week")

    return None


def normalize_eex_product_code(eex_code: str) -> tuple[str, str, str] | None:
    """Return the canonical delivery, load and product type for a desk code.

    This public wrapper is the single product-identity parser used by governed
    prospective EEX intake. Week products remain recognizable but ineligible
    for the LT monthly solver.
    """

    return _normalize_product(eex_code)


def load_forwards_timeseries(
    report_path: str | Path,
    market: str = "CH",
    include_week: bool = False,
) -> pd.DataFrame:
    """Extract full timeseries from EEX report (all dates, BASE + PEAK).

    Args:
        report_path: Absolute or relative path to the EEX report XLSX.
        market: Sheet name to load (case-insensitive against workbook tabs).
        include_week: If False (default), Week products are dropped silently
            and only Cal/Quarter/Month rows are returned. The PFC long-term
            curve does not consume weekly products.

    Returns:
        DataFrame with columns: date, product, load_type, product_type, price.
        product_type ∈ {'Cal', 'Quarter', 'Month'} when ``include_week=False``,
        plus 'Week' when ``include_week=True``.
    """
    report_path = Path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(f"EEX report not found: {report_path}")

    raw = pd.read_excel(report_path, sheet_name=market, header=None)
    if raw.shape[0] < 4 or raw.shape[1] < 2:
        raise ValueError(f"Unexpected format in {report_path} (sheet={market})")

    product_codes = raw.iloc[0, 1:]
    date_series = pd.to_datetime(raw.iloc[3:, 0], dayfirst=True, errors="coerce")
    values = raw.iloc[3:, 1:]

    # Parse all valid product columns (BASE + PEAK)
    col_info: dict[int, tuple[str, str, str]] = {}  # col_idx -> (product, load_type, product_type)
    n_week_skipped = 0
    for idx, code in enumerate(product_codes):
        if pd.isna(code):
            continue
        parsed = _normalize_product(str(code))
        if parsed is None:
            continue
        delivery_key, load_type, ptype = parsed
        if ptype == "Week" and not include_week:
            n_week_skipped += 1
            continue
        col_info[idx] = (delivery_key, load_type, ptype)

    if n_week_skipped:
        logger.debug(
            "Sheet %s: skipped %d Week product columns (include_week=False)",
            market, n_week_skipped,
        )

    if not col_info:
        raise ValueError(f"No valid products found in EEX sheet {market}")

    rows = []
    for row_idx in values.index:
        dt = date_series.loc[row_idx]
        if pd.isna(dt):
            continue
        for col_idx, (product, load_type, ptype) in col_info.items():
            val = values.iloc[row_idx - values.index[0], col_idx]
            price = _coerce_price(val)
            if price is not None:
                rows.append({
                    "date": dt.normalize(),
                    "product": product,
                    "load_type": load_type,
                    "product_type": ptype,
                    "price": price,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No valid prices in EEX sheet {market}")

    df["date"] = pd.to_datetime(df["date"])
    logger.info(
        "EEX timeseries loaded (%s, sheet=%s): %d obs, %d products, %s → %s",
        report_path.name, market, len(df),
        df["product"].nunique(),
        df["date"].min().date(), df["date"].max().date(),
    )
    return df


def update_forwards_parquet(
    report_path: str | Path,
    parquet_path: str | Path = "data/eex_forwards_history.parquet",
    markets: list[str] | None = None,
    include_week: bool = False,
) -> pd.DataFrame:
    """Ingest EEX report and append to historical Parquet (dedup on date+product+load_type+market).

    Args:
        report_path: EEX XLSX (Yearly snapshot or Historique).
        parquet_path: Destination Parquet for the appended history.
        markets: Sheet names to ingest. ``None`` defaults to the full panel
            ``["CH", "DE", "FR", "AT", "IT"]``. Any sheet name in
            :data:`_NON_MARKET_SHEETS` (FX / Produits / HFC) is filtered
            out automatically and logged.
        include_week: Forward Week products to the Parquet too (default
            False — skipped because the LT PFC does not use them).

    Returns the updated full DataFrame.
    """
    if markets is None:
        markets = ["CH", "DE", "FR", "AT", "IT"]

    requested = list(markets)
    filtered: list[str] = []
    for mkt in requested:
        if str(mkt).strip().upper() in _NON_MARKET_SHEETS:
            logger.info(
                "Skipping non-market sheet %r (FX / Produits / HFC are never parsed as forwards)",
                mkt,
            )
            continue
        filtered.append(mkt)
    markets = filtered

    if not markets:
        raise ValueError(
            f"No market sheets remain after filtering non-market tabs from {requested}"
        )

    parquet_path = Path(parquet_path)
    dfs = []
    for mkt in markets:
        try:
            ts = load_forwards_timeseries(report_path, market=mkt, include_week=include_week)
            ts["market"] = mkt
            dfs.append(ts)
        except Exception as exc:
            logger.warning("Skipping market %s: %s", mkt, exc)

    if not dfs:
        raise ValueError("No market data loaded from EEX report")

    new_data = pd.concat(dfs, ignore_index=True)

    if parquet_path.exists():
        existing = pd.read_parquet(parquet_path)
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["date", "product", "load_type", "market"], keep="last"
        )
    else:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        combined = new_data

    combined = combined.sort_values(["market", "date", "product", "load_type"]).reset_index(drop=True)
    combined.to_parquet(parquet_path, index=False)
    logger.info("Forwards history saved: %s (%d rows)", parquet_path, len(combined))
    return combined


def load_base_prices_from_eex_report(
    report_path: str | Path,
    market: str = "CH",
    as_of_date: str | None = None,
    *,
    return_snapshot_date: bool = False,
    return_snapshot_metadata: bool = False,
) -> (
    dict[str, float]
    | tuple[dict[str, float], pd.Timestamp]
    | tuple[dict[str, float], pd.Timestamp, dict[str, object]]
):
    """
    Load forward prices from a daily EEX price report XLSX file (BASE + PEAK).

    Expected workbook layout:
        - One sheet per market (e.g. CH/DE/FR)
        - Row 1 contains product codes (Y01_YYYY_BASE/PEAK, QNN_YYYY_BASE/PEAK, MNN_YYYY_BASE/PEAK)
        - Row 4+ contains daily marks with a date in column A

    Week products (3-digit prefix codes such as ``205_2026_PEAK``) are
    detected and **skipped** — the long-term PFC curve does not consume them.

    Args:
        report_path: Absolute or relative path to the EEX report XLSX.
        market: Sheet name to load (default: CH).
        as_of_date: Optional date (YYYY-MM-DD). If None, latest available
                    non-zero date is selected automatically.
        return_snapshot_date: Also return the date of the workbook row used.
                              This date is source evidence and must not be
                              inferred from the pipeline run timestamp.
        return_snapshot_metadata: Return source row/column lineage as a third
                                  element. Implies ``return_snapshot_date``.

    Returns:
        dict[str, float]: {'2027': 82.9, '2027-Peak': 95.1, '2026-Q3': 74.8,
                           '2026-Q3-Peak': 88.2, '2026-04': 84.5, '2026-04-Peak': 92.0, ...}
        PEAK products use '-Peak' suffix. BASE products have no suffix.
    """
    report_path = Path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(f"EEX report not found: {report_path}")
    return load_base_prices_from_eex_report_bytes(
        report_path.read_bytes(),
        market=market,
        as_of_date=as_of_date,
        source_label=str(report_path),
        return_snapshot_date=return_snapshot_date,
        return_snapshot_metadata=return_snapshot_metadata,
    )


def load_base_prices_from_eex_report_bytes(
    report_bytes: bytes,
    *,
    market: str = "CH",
    as_of_date: str | None = None,
    source_label: str = "<EEX workbook bytes>",
    return_snapshot_date: bool = False,
    return_snapshot_metadata: bool = False,
) -> (
    dict[str, float]
    | tuple[dict[str, float], pd.Timestamp]
    | tuple[dict[str, float], pd.Timestamp, dict[str, object]]
):
    """Parse the exact workbook bytes that are bound into source evidence."""

    report_path = source_label
    raw = pd.read_excel(BytesIO(report_bytes), sheet_name=market, header=None)
    if raw.shape[0] < 4 or raw.shape[1] < 2:
        raise ValueError(f"Unexpected EEX report format in {report_path} (sheet={market})")

    product_codes = raw.iloc[0, 1:]
    date_series = pd.to_datetime(raw.iloc[3:, 0], dayfirst=True, errors="coerce")
    values = raw.iloc[3:, 1:]

    selected_cols: list[int] = []
    delivery_keys: dict[int, str] = {}
    delivery_source_codes: dict[int, str] = {}
    n_week_skipped = 0
    for idx, code in enumerate(product_codes):
        if pd.isna(code):
            continue
        parsed = _normalize_product(str(code))
        if parsed is None:
            continue
        delivery_key, load_type, ptype = parsed
        if ptype == "Week":
            n_week_skipped += 1
            continue
        if load_type == "PEAK":
            delivery_key = f"{delivery_key}-Peak"
        selected_cols.append(idx)
        delivery_keys[idx] = delivery_key
        delivery_source_codes[idx] = str(code)

    if n_week_skipped:
        logger.debug(
            "Sheet %s: skipped %d Week products from base_prices",
            market, n_week_skipped,
        )

    if not selected_cols:
        raise ValueError(f"No Cal/Quarter/Month contracts found in EEX sheet {market}")

    selected = values.iloc[:, selected_cols].copy()
    for col in selected.columns:
        selected[col] = pd.to_numeric(
            selected[col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )

    # Build a "row has at least one quoted product" mask using the desk
    # convention: 0 / NaN = non-quoted, anything else (negative or
    # positive) = a quote. We check membership against the sanity range
    # so a stray ``9999999`` typo doesn't pass the as-of selection.
    #
    # Vectorised expression: NaN comparisons return False so NaN rows
    # never pass either bound, and ``selected != 0.0`` is False on NaN
    # — both effects collaborate. Stays compatible with pandas 1.x / 2.0
    # (no ``DataFrame.map`` dependency, no per-cell Python lambda).
    quoted_mask_per_row = (
        (selected != 0.0)
        & (selected >= _PRICE_FLOOR_EUR_MWH)
        & (selected <= _PRICE_CEILING_EUR_MWH)
    ).any(axis=1)

    valid_mask = date_series.notna() & quoted_mask_per_row
    if as_of_date is not None:
        target = pd.Timestamp(as_of_date).normalize()
        valid_mask &= date_series.dt.normalize() == target

    if not valid_mask.any():
        d = f" date={as_of_date}" if as_of_date else ""
        raise ValueError(f"No valid EEX row found in {report_path} (sheet={market}{d})")

    eligible_dates = date_series.loc[valid_mask].dt.normalize()
    selected_date = pd.Timestamp(eligible_dates.max()).normalize()
    row_positions = eligible_dates[eligible_dates.eq(selected_date)].index.tolist()

    def _prices_for_row(row_pos: int) -> tuple[dict[str, float], dict[str, dict[str, object]]]:
        row_prices: dict[str, float] = {}
        row_lineage: dict[str, dict[str, object]] = {}
        for local_col, val in selected.loc[row_pos].items():
            price = _coerce_price(val)
            if price is None:
                continue
            # local_col is absolute column index in the original sheet minus
            # the date-column offset; delivery_keys is positional.
            product_idx = int(local_col) - 1
            key = delivery_keys.get(product_idx)
            if key is None:
                continue
            if key in row_prices:
                raise ValueError(
                    f"QUOTE_CONFLICT duplicate product {key} in {report_path} "
                    f"(sheet={market}, date={selected_date.date()})"
                )
            row_prices[key] = price
            row_lineage[key] = {
                "source_product_code": delivery_source_codes[product_idx],
                "source_row_index": int(row_pos) + 1,
                "source_column_index": int(local_col) + 1,
                "is_direct_market_quote": True,
            }
        return row_prices, row_lineage

    candidates = [_prices_for_row(int(row_pos)) for row_pos in row_positions]
    base_prices, quote_lineage = candidates[0]
    if any(candidate_prices != base_prices for candidate_prices, _ in candidates[1:]):
        raise ValueError(
            f"QUOTE_CONFLICT duplicate snapshot rows in {report_path} "
            f"(sheet={market}, date={selected_date.date()})"
        )
    row_date = selected_date

    if not base_prices:
        raise ValueError(
            f"EEX row has no quoted prices (sheet={market}, date={row_date.date()})"
        )

    n_base = sum(1 for k in base_prices if not k.endswith("-Peak"))
    n_peak = sum(1 for k in base_prices if k.endswith("-Peak"))
    logger.info(
        "Forwards EEX XLSX loaded (%s, sheet=%s, date=%s): %d BASE + %d PEAK products",
        report_path,
        market,
        row_date.date(),
        n_base,
        n_peak,
    )
    snapshot_date = pd.Timestamp(row_date).tz_localize(None).normalize()
    if return_snapshot_metadata:
        return base_prices, snapshot_date, {"quote_lineage": quote_lineage}
    if return_snapshot_date:
        return base_prices, snapshot_date
    return base_prices
