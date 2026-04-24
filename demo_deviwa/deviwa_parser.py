"""Parser für die Deviwa-Transaktionsdatei.

Unterstützt sowohl XLSX mit mehreren Sheets (ein Sheet je Akteur) als auch
einzelne CSV-Dateien. Die Spaltennamen werden normalisiert, damit kleine
Abweichungen (Gross-/Kleinschreibung, Leerzeichen) toleriert werden.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


ACTOR_SHEET_MAPPING = {
    "RELL": ["rell", "rell - deals"],
    "EDSH": ["edsh", "edsh - deals"],
    "EDSH (Programme & Spot)": ["edsh - programme", "edsh programme", "programme&spot"],
    "EW Binn": ["ew binn", "ew_binn", "ew binn - deals"],
    "EVTL": ["evtl", "evtl - deals"],
}

HPFC_SHEET_KEYS = ["hpfc"]


CANONICAL_COLUMNS = {
    "counterparty": ["counterparty", "gegenpartei", "partner"],
    "asset_type": ["asset type", "asset_type", "typ"],
    "custom": ["custom", "benutzerdefiniert"],
    "deal": ["deal", "deal id", "deal name", "geschäft"],
    "trade_date": ["deal trade date", "trade date", "handelsdatum", "tradedate"],
    "delivery_from": ["deal delivery from", "delivery from", "lieferung von", "from"],
    "delivery_to": ["deal delivery to", "delivery to", "lieferung bis", "to"],
    "product": ["product", "produkt"],
    "scope": ["scope", "richtung", "intake/withdrawal"],
    "month": ["date", "month", "lieferung", "lieferperiode", "period"],
    "volume_sum": ["volume (sum)", "volume_sum", "volumen"],
    "volume_mean": ["volume (mean)", "volume_mean"],
    "volume_net": ["volume (net)", "volume_net"],
    "volume_net_mean": ["volume (net mean)", "volume_net_mean"],
    "market_value_sum": ["market value (sum)", "market_value_sum", "marktwert"],
    "market_value_mean": ["market value (mean)", "market_value_mean"],
    "notional_sum": ["notional (sum)", "notional_sum"],
    "notional_mean": ["notional (mean)", "notional_mean"],
    "pnl_sum": ["pnl (sum)", "pnl_sum", "p&l", "profit"],
    "pnl_mean": ["pnl (mean)", "pnl_mean"],
}


def _normalize_col(name: str) -> str:
    return str(name).strip().lower().replace("_", " ")


def _match_col(raw_name: str) -> str | None:
    n = _normalize_col(raw_name)
    for canonical, variants in CANONICAL_COLUMNS.items():
        for v in variants:
            if _normalize_col(v) == n or _normalize_col(v) in n:
                return canonical
    return None


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in df.columns:
        match = _match_col(c)
        if match:
            rename_map[c] = match
    df = df.rename(columns=rename_map)

    # Typkonvertierungen
    for col in ("trade_date", "delivery_from", "delivery_to"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    if "month" in df.columns:
        # Deviwa speichert "2026-01" als String oder Datum
        df["month"] = pd.to_datetime(df["month"], errors="coerce")

    numeric_cols = [
        "volume_sum", "volume_mean", "volume_net", "volume_net_mean",
        "market_value_sum", "market_value_mean",
        "notional_sum", "notional_mean",
        "pnl_sum", "pnl_mean",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def load_deviwa_file(path: str | Path) -> dict[str, pd.DataFrame]:
    """Lädt Deviwa-Datei (xlsx oder csv). Gibt ein Dict {Akteur: DataFrame} zurück."""
    p = Path(path)
    if not p.exists():
        return {}

    suffix = p.suffix.lower()
    out: dict[str, pd.DataFrame] = {}

    if suffix in (".xlsx", ".xls"):
        xls = pd.ExcelFile(p)
        for sheet in xls.sheet_names:
            sn = sheet.lower()
            if any(k in sn for k in HPFC_SHEET_KEYS):
                df_h = pd.read_excel(xls, sheet_name=sheet)
                out["_HPFC"] = df_h
                continue
            actor = _assign_actor(sn)
            if actor is None:
                continue
            df = pd.read_excel(xls, sheet_name=sheet)
            df = _standardize_columns(df)
            df["_actor"] = actor
            df["_sheet"] = sheet
            if actor in out:
                out[actor] = pd.concat([out[actor], df], ignore_index=True)
            else:
                out[actor] = df
    else:
        # Reine CSV: alle Deals zusammen, ohne Akteur-Trennung
        df = pd.read_csv(p, sep=None, engine="python")
        df = _standardize_columns(df)
        if "counterparty" in df.columns:
            for cp, sub in df.groupby("counterparty"):
                out[str(cp)] = sub.copy()
        else:
            out["Alle"] = df

    return out


def _assign_actor(sheet_name_lower: str) -> str | None:
    for actor, keys in ACTOR_SHEET_MAPPING.items():
        for k in keys:
            if k in sheet_name_lower:
                return actor
    return None


# ---------------------------------------------------------------------------
# Aggregationen
# ---------------------------------------------------------------------------

def summarize_actor(df: pd.DataFrame) -> dict:
    """KPI-Zusammenfassung für einen Akteur."""
    if df is None or df.empty:
        return {"n_deals": 0, "total_volume_mwh": 0.0, "total_pnl_eur": 0.0,
                "intake_mwh": 0.0, "withdrawal_mwh": 0.0,
                "period_start": None, "period_end": None}

    n_deals = int(df["deal"].nunique()) if "deal" in df.columns else len(df)
    total_volume = float(df.get("volume_sum", pd.Series([0])).abs().sum())
    total_pnl = float(df.get("pnl_sum", pd.Series([0])).sum())

    scope_col = df.get("scope")
    if scope_col is not None:
        intake_mask = scope_col.astype(str).str.lower().str.contains("intake")
        withdrawal_mask = scope_col.astype(str).str.lower().str.contains("withdrawal")
        intake_mwh = float(df.loc[intake_mask, "volume_sum"].abs().sum()) if intake_mask.any() else 0.0
        withdrawal_mwh = float(df.loc[withdrawal_mask, "volume_sum"].abs().sum()) if withdrawal_mask.any() else 0.0
    else:
        intake_mwh = 0.0
        withdrawal_mwh = 0.0

    period_start = df["month"].min() if "month" in df.columns else None
    period_end = df["month"].max() if "month" in df.columns else None

    return {
        "n_deals": n_deals,
        "total_volume_mwh": total_volume,
        "total_pnl_eur": total_pnl,
        "intake_mwh": intake_mwh,
        "withdrawal_mwh": withdrawal_mwh,
        "period_start": period_start,
        "period_end": period_end,
    }


def monthly_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Monatliche Aggregation Volumen/PnL nach Scope."""
    if df is None or df.empty or "month" not in df.columns:
        return pd.DataFrame()
    g = df.copy()
    g["scope_clean"] = g.get("scope", "unknown").astype(str).str.title().replace({
        "Intake": "Einspeisung", "Withdrawal": "Bezug",
    })
    agg = g.groupby(["month", "scope_clean"]).agg(
        volume_mwh=("volume_sum", lambda s: float(pd.to_numeric(s, errors="coerce").abs().sum())),
        pnl_eur=("pnl_sum", lambda s: float(pd.to_numeric(s, errors="coerce").sum())),
        market_value_eur=("market_value_sum", lambda s: float(pd.to_numeric(s, errors="coerce").sum())),
    ).reset_index()
    return agg.sort_values("month")


def deals_table(df: pd.DataFrame) -> pd.DataFrame:
    """Bereinigte Deal-Übersicht für Tabelle."""
    if df is None or df.empty:
        return pd.DataFrame()
    cols = ["deal", "trade_date", "delivery_from", "delivery_to", "product",
            "scope", "month", "volume_sum", "market_value_sum", "pnl_sum"]
    cols = [c for c in cols if c in df.columns]
    show = df[cols].copy()
    rename = {
        "deal": "Deal",
        "trade_date": "Handelsdatum",
        "delivery_from": "Lieferung ab",
        "delivery_to": "Lieferung bis",
        "product": "Produkt",
        "scope": "Richtung",
        "month": "Monat",
        "volume_sum": "Volumen (MWh)",
        "market_value_sum": "Marktwert (EUR)",
        "pnl_sum": "PnL (EUR)",
    }
    return show.rename(columns=rename)
