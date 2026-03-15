"""
ingest_forwards.py
------------------
Chargement des forwards EEX calibrÃ©s par EULER depuis Databricks.

Ces forwards constituent les niveaux de base B (â‚¬/MWh) utilisÃ©s par
l'assembleur PFC. EULER les calibre sur les prix EEX liquides ; notre
modÃ¨le applique par-dessus la forme (shape) 15min.

Table Databricks attendue (config.yaml â†’ databricks.tables.eex_forwards) :
    run_date        DATE        â€” date du run EULER (dernier run disponible)
    delivery_period STRING      â€” 'YYYY' | 'YYYY-QN' | 'YYYY-MM'
    price_eur_mwh   DOUBLE      â€” prix forward calibrÃ© [â‚¬/MWh]
    product_type    STRING      â€” 'Cal' | 'Quarter' | 'Month'

Format de sortie :
    dict[str, float] â€” clÃ©s : '2025', '2025-Q1', '2025-03', etc.
    Directement compatible avec PFCAssembler.build(base_prices=...)

Usage :
    from pfc_shaping.data.ingest_forwards import load_base_prices
    base_prices = load_base_prices(run_date='2025-03-10')
    # {'2025': 75.0, '2026': 72.0, '2025-Q2': 74.5, '2025-03': 82.0, ...}
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from pfc_shaping.data.databricks_client import query_to_df, table_fqn

logger = logging.getLogger(__name__)


_EEX_BASE_PATTERN = re.compile(r"^(Y01|Q\d{2}|M\d{2})_(\d{4})_BASE$")
_EEX_PRODUCT_PATTERN = re.compile(r"^(Y01|Q\d{2}|M\d{2})_(\d{4})_(BASE|PEAK)$")


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


def _normalize_product(eex_code: str) -> tuple[str, str] | None:
    """Parse EEX code into (delivery_key, load_type).

    Examples:
        Y01_2027_BASE -> ('2027', 'BASE')
        Q03_2026_PEAK -> ('2026-Q3', 'PEAK')
        M04_2026_BASE -> ('2026-04', 'BASE')
        303_2026_PEAK -> None  (weekly products ignored)
    """
    m = _EEX_PRODUCT_PATTERN.match(eex_code.strip().upper())
    if not m:
        return None

    prefix, year_str, load_type = m.groups()
    year = int(year_str)

    if prefix == "Y01":
        return f"{year}", load_type
    if prefix.startswith("Q"):
        quarter = int(prefix[1:])
        if 1 <= quarter <= 4:
            return f"{year}-Q{quarter}", load_type
    if prefix.startswith("M"):
        month = int(prefix[1:])
        if 1 <= month <= 12:
            return f"{year}-{month:02d}", load_type
    return None


def load_forwards_timeseries(
    report_path: str | Path,
    market: str = "CH",
) -> pd.DataFrame:
    """Extract full timeseries from EEX report (all dates, BASE + PEAK).

    Returns:
        DataFrame with columns: date, product, load_type, product_type, price
        - product: '2027', '2026-Q3', '2026-04', etc.
        - load_type: 'BASE' or 'PEAK'
        - product_type: 'Cal', 'Quarter', 'Month'
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
    for idx, code in enumerate(product_codes):
        if pd.isna(code):
            continue
        parsed = _normalize_product(str(code))
        if parsed is None:
            continue
        delivery_key, load_type = parsed
        if len(delivery_key) == 4:
            ptype = "Cal"
        elif "-Q" in delivery_key:
            ptype = "Quarter"
        else:
            ptype = "Month"
        col_info[idx] = (delivery_key, load_type, ptype)

    if not col_info:
        raise ValueError(f"No valid products found in EEX sheet {market}")

    rows = []
    for row_idx in values.index:
        dt = date_series.loc[row_idx]
        if pd.isna(dt):
            continue
        for col_idx, (product, load_type, ptype) in col_info.items():
            val = values.iloc[row_idx - values.index[0], col_idx]
            price = pd.to_numeric(str(val).replace(",", "."), errors="coerce") if not pd.isna(val) else None
            if price is not None and price > 0:
                rows.append({
                    "date": dt.normalize(),
                    "product": product,
                    "load_type": load_type,
                    "product_type": ptype,
                    "price": float(price),
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
) -> pd.DataFrame:
    """Ingest EEX report and append to historical Parquet (dedup on date+product+load_type+market).

    Returns the updated full DataFrame.
    """
    if markets is None:
        markets = ["CH", "DE"]

    parquet_path = Path(parquet_path)
    dfs = []
    for mkt in markets:
        try:
            ts = load_forwards_timeseries(report_path, market=mkt)
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
) -> dict[str, float]:
    """
    Load forward prices from a daily EEX price report XLSX file (BASE + PEAK).

    Expected workbook layout:
        - One sheet per market (e.g. CH/DE/FR)
        - Row 1 contains product codes (Y01_YYYY_BASE/PEAK, QNN_YYYY_BASE/PEAK, MNN_YYYY_BASE/PEAK)
        - Row 4+ contains daily marks with a date in column A

    Args:
        report_path: Absolute or relative path to the EEX report XLSX.
        market: Sheet name to load (default: CH).
        as_of_date: Optional date (YYYY-MM-DD). If None, latest available
                    non-zero date is selected automatically.

    Returns:
        dict[str, float]: {'2027': 82.9, '2027-Peak': 95.1, '2026-Q3': 74.8,
                           '2026-Q3-Peak': 88.2, '2026-04': 84.5, '2026-04-Peak': 92.0, ...}
        PEAK products use '-Peak' suffix. BASE products have no suffix.
    """
    report_path = Path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(f"EEX report not found: {report_path}")

    raw = pd.read_excel(report_path, sheet_name=market, header=None)
    if raw.shape[0] < 4 or raw.shape[1] < 2:
        raise ValueError(f"Unexpected EEX report format in {report_path} (sheet={market})")

    product_codes = raw.iloc[0, 1:]
    date_series = pd.to_datetime(raw.iloc[3:, 0], dayfirst=True, errors="coerce")
    values = raw.iloc[3:, 1:]

    selected_cols: list[int] = []
    delivery_keys: dict[int, str] = {}
    for idx, code in enumerate(product_codes):
        if pd.isna(code):
            continue
        parsed = _normalize_product(str(code))
        if parsed is None:
            continue
        delivery_key, load_type = parsed
        if load_type == "PEAK":
            delivery_key = f"{delivery_key}-Peak"
        selected_cols.append(idx)
        delivery_keys[idx] = delivery_key

    if not selected_cols:
        raise ValueError(f"No Cal/Quarter/Month contracts found in EEX sheet {market}")

    selected = values.iloc[:, selected_cols].copy()
    for col in selected.columns:
        selected[col] = pd.to_numeric(
            selected[col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )

    valid_mask = date_series.notna()
    if as_of_date is not None:
        target = pd.Timestamp(as_of_date).normalize()
        valid_mask &= date_series.dt.normalize() == target
    else:
        valid_mask &= (selected.fillna(0) > 0).any(axis=1)

    if not valid_mask.any():
        d = f" date={as_of_date}" if as_of_date else ""
        raise ValueError(f"No valid EEX row found in {report_path} (sheet={market}{d})")

    row_pos = valid_mask[valid_mask].index[-1]
    row_values = selected.loc[row_pos]
    row_date = date_series.loc[row_pos]

    base_prices: dict[str, float] = {}
    for local_col, val in row_values.items():
        if pd.isna(val) or float(val) <= 0:
            continue
        # local_col is absolute column index in original sheet minus 1 offset from product row,
        # while delivery_keys uses positional index over product_codes.
        product_idx = int(local_col) - 1
        key = delivery_keys.get(product_idx)
        if key is not None:
            base_prices[key] = float(val)

    if not base_prices:
        raise ValueError(
            f"EEX row has no positive prices (sheet={market}, date={row_date.date()})"
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
    return base_prices


def load_base_prices(
    run_date: str | None = None,
    db_config: dict | None = None,
) -> dict[str, float]:
    """
    Charge les forwards EEX depuis Databricks et retourne le dict base_prices.

    Args:
        run_date  : 'YYYY-MM-DD' du run EULER Ã  utiliser
                    Si None â†’ prend le run le plus rÃ©cent disponible
        db_config : config Databricks (si None, lit config.yaml)

    Returns:
        dict[str, float] â€” ex. {'2025': 75.0, '2025-Q1': 80.0, '2025-03': 82.0}
    """
    fqn = table_fqn("eex_forwards", db_config)

    if run_date is None:
        # Requête atomique : dernier run disponible + données en une seule passe
        sql = f"""
            SELECT delivery_period, price_eur_mwh, product_type
            FROM {fqn}
            WHERE run_date = (SELECT MAX(run_date) FROM {fqn})
            ORDER BY product_type, delivery_period
        """
        logger.info("Chargement forwards EEX (run le plus récent)...")
        df = query_to_df(sql, config=db_config)
        if not df.empty:
            run_date = "latest"
    else:
        # Validation format date
        import re
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", run_date):
            raise ValueError(f"run_date invalide (attendu YYYY-MM-DD) : {run_date}")

        sql = f"""
            SELECT delivery_period, price_eur_mwh, product_type
            FROM {fqn}
            WHERE run_date = ?
            ORDER BY product_type, delivery_period
        """
        logger.info("Chargement forwards EEX (run=%s)...", run_date)
        df = query_to_df(sql, params=[run_date], config=db_config)

    if df.empty:
        raise ValueError(f"Aucun forward EEX trouvÃ© pour run_date={run_date}")

    base_prices: dict[str, float] = {}
    for _, row in df.iterrows():
        key = str(row["delivery_period"]).strip()
        price = float(row["price_eur_mwh"])
        base_prices[key] = price

    logger.info(
        "Forwards EEX chargÃ©s : %d produits (Cal=%d, Quarter=%d, Month=%d)",
        len(base_prices),
        df[df["product_type"] == "Cal"].shape[0],
        df[df["product_type"] == "Quarter"].shape[0],
        df[df["product_type"] == "Month"].shape[0],
    )
    return base_prices


def latest_run_date(db_config: dict | None = None) -> str:
    """Retourne la date du dernier run EULER disponible dans Databricks."""
    fqn = table_fqn("eex_forwards", db_config)
    sql = f"SELECT MAX(run_date) AS latest FROM {fqn}"
    df = query_to_df(sql, config=db_config)
    return str(df["latest"].iloc[0])
