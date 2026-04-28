"""Parser für die Deviwa-Transaktionsdatei.

Unterstützt sowohl XLSX mit mehreren Sheets (ein Sheet je Akteur) als auch
einzelne CSV-Dateien. Die Spaltennamen werden normalisiert, damit kleine
Abweichungen (Gross-/Kleinschreibung, Leerzeichen) toleriert werden.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
PROGRAMME_SHEET_KEYS = ["programme", "program", "pro_real", "pro real", "real"]


CANONICAL_COLUMNS = {
    "counterparty": ["counterparty", "gegenpartei", "partner"],
    "asset_type": ["asset type", "asset_type", "typ"],
    "custom": ["custom", "benutzerdefiniert"],
    "deal": ["deal", "deal id", "deal name", "geschäft"],
    "trade_date": ["deal trade date", "trade date", "handelsdatum", "tradedate"],
    "delivery_from": ["deal delivery start", "deal delivery from", "delivery start", "delivery from", "lieferung von", "from"],
    "delivery_to": ["deal delivery end", "deal delivery to", "delivery end", "delivery to", "lieferung bis", "to"],
    "product": ["product", "produkt"],
    "scope": ["scope", "richtung", "intake/withdrawal"],
    "month": ["date", "month", "lieferung", "lieferperiode", "period"],
    "volume_sum": ["volume (sum)", "volume_sum", "volumen"],
    "volume_mean": ["volume (mean)", "volume_mean"],
    "volume_net": ["volume (net)", "volume (net) (sum)", "volume_net", "volume_net_sum"],
    "volume_net_mean": ["volume (net mean)", "volume (net) (mean)", "volume_net_mean"],
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
        if any(_normalize_col(v) == n for v in variants):
            return canonical
    best_match: tuple[int, str] | None = None
    for canonical, variants in CANONICAL_COLUMNS.items():
        for v in variants:
            vn = _normalize_col(v)
            if vn in n:
                score = len(vn)
                if best_match is None or score > best_match[0]:
                    best_match = (score, canonical)
    if best_match is not None:
        return best_match[1]
    return None


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in df.columns:
        match = _match_col(c)
        if match:
            rename_map[c] = match
    df = df.rename(columns=rename_map)
    if df.columns.duplicated().any():
        # After tolerant matching, keep one canonical column by taking the first
        # non-null value across duplicate source columns.
        collapsed = {}
        for col in pd.Index(df.columns).unique():
            same = df.loc[:, df.columns == col]
            collapsed[col] = same.bfill(axis=1).iloc[:, 0] if same.shape[1] > 1 else same.iloc[:, 0]
        df = pd.DataFrame(collapsed, index=df.index)

    # Typkonvertierungen
    for col in ("trade_date", "delivery_from", "delivery_to"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    if "month" in df.columns:
        # Deviwa speichert "2026-01" als String oder Datum
        df["month"] = pd.to_datetime(df["month"], errors="coerce", format="mixed")

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


def _invert_to_client_perspective(df: pd.DataFrame) -> pd.DataFrame:
    """Konvertiert FMV-Sicht (Buchhaltung) in Kunden-Sicht.

    Die Deviwa-Datei enthält Deals aus FMV-Perspektive: was FMV einkauft
    (Intake für FMV) entspricht einem Verkauf aus Kundensicht und umgekehrt.
    Diese Funktion wendet zwei Transformationen an:

    1. ``scope`` wird getauscht: Intake ↔ Withdrawal
    2. Alle vorzeichenbehafteten monetären/Netto-Spalten werden negiert
       (Notional, Marktwert, PnL, Volume-Net).

    Volumen-Beträge (volume_sum, volume_mean) bleiben als Absolutwerte erhalten.
    """
    if df is None or df.empty:
        return df
    df = df.copy()

    if "scope" in df.columns:
        def _flip(s):
            if not isinstance(s, str):
                return s
            sl = s.strip().lower()
            if "intake" in sl or "einspeise" in sl or sl == "in":
                return "Withdrawal"
            if "withdraw" in sl or "bezug" in sl or sl == "out":
                return "Intake"
            return s
        df["scope"] = df["scope"].apply(_flip)

    sign_flip_cols = [
        "volume_net", "volume_net_mean",
        "market_value_sum", "market_value_mean",
        "notional_sum", "notional_mean",
        "pnl_sum", "pnl_mean",
    ]
    for c in sign_flip_cols:
        if c in df.columns:
            df[c] = -pd.to_numeric(df[c], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def load_deviwa_file(path: str | Path) -> dict[str, pd.DataFrame]:
    """Lädt Deviwa-Datei (xlsx oder csv) in **Kundensicht**.

    Die Deals werden automatisch aus der FMV-Buchhaltungssicht in die
    Kundensicht invertiert (Intake↔Withdrawal, Vorzeichen der monetären
    Felder). Gibt ein Dict {Akteur: DataFrame} zurück.
    """
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
            # Keep ProgrammeReal / PRO_REAL sheets out of the deals loader.
            # Actor names such as "RELL" also appear in programme sheets; if
            # we let the broad actor matcher below catch them, the dashboard
            # receives tens of thousands of zero-volume pseudo-deal rows.
            if any(k in sn for k in PROGRAMME_SHEET_KEYS) and "deal" not in sn:
                continue
            actor = _assign_actor(sn)
            if actor is None:
                continue
            df = pd.read_excel(xls, sheet_name=sheet)
            df = _standardize_columns(df)
            df = _invert_to_client_perspective(df)
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
        df = _invert_to_client_perspective(df)
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


def extract_programme_data(path: str | Path) -> dict[str, pd.DataFrame]:
    """Liest die ProgrammeReal-Sheets der Deviwa-Datei.

    Erkennt automatisch das Schema mit Spalten ``<Akteur>_Programm``,
    ``<Akteur>_Real`` und optional ``<Akteur>_Budget`` (neues Format) sowie
    älteres heuristisches Format mit zwei Spalten Plan/Ist pro Sheet.

    Liefert ein Dict ``{Akteur: DataFrame}`` mit Spalten:
    timestamp, programme_mw, actual_mw, budget_mw, delta_mw, delta_pct,
    month, date, hour, dow, iso_week, year.
    """
    p = Path(path)
    if not p.exists() or p.suffix.lower() not in (".xlsx", ".xls"):
        return {}

    out: dict[str, pd.DataFrame] = {}
    xls = pd.ExcelFile(p)

    for sheet in xls.sheet_names:
        sn = sheet.lower()
        if "hpfc" in sn:
            continue

        raw = pd.read_excel(xls, sheet_name=sheet)
        if raw.empty:
            continue

        date_col = next(
            (c for c in raw.columns if str(c).strip().lower() in ("date", "datum", "zeit", "timestamp")),
            None,
        )
        if date_col is None:
            continue

        # Neues Schema: <Akteur>_Programm / _Real / _Budget
        actors_in_sheet: dict[str, dict[str, str]] = {}
        for col in raw.columns:
            cs = str(col).strip()
            for suffix, key in [
                ("_Programm", "programme"),
                ("_Program", "programme"),
                ("_Real", "actual"),
                ("_Ist", "actual"),
                ("_Actual", "actual"),
                ("_Budget", "budget"),
            ]:
                if cs.endswith(suffix):
                    actor_raw = cs[: -len(suffix)].replace("_", " ").strip()
                    actors_in_sheet.setdefault(actor_raw, {})[key] = col
                    break

        if not actors_in_sheet:
            # Fallback altes heuristisches Format
            actors_in_sheet[sheet.strip()] = {}
            for c in raw.columns:
                lc = str(c).strip().lower()
                if "programm" in lc or "plan" in lc or "geplant" in lc or "soll" in lc:
                    actors_in_sheet[sheet.strip()].setdefault("programme", c)
                elif " ist" in lc or "actual" in lc or "real" in lc or "spot" in lc:
                    actors_in_sheet[sheet.strip()].setdefault("actual", c)
                elif "budget" in lc:
                    actors_in_sheet[sheet.strip()].setdefault("budget", c)

        ts_all = pd.to_datetime(raw[date_col], errors="coerce", dayfirst=True)

        for actor_raw, cols in actors_in_sheet.items():
            if "programme" not in cols and "actual" not in cols:
                continue
            df_out = pd.DataFrame({"timestamp": ts_all})
            df_out["programme_mw"] = (
                pd.to_numeric(raw[cols["programme"]], errors="coerce") if "programme" in cols else np.nan
            )
            df_out["actual_mw"] = (
                pd.to_numeric(raw[cols["actual"]], errors="coerce") if "actual" in cols else np.nan
            )
            df_out["budget_mw"] = (
                pd.to_numeric(raw[cols["budget"]], errors="coerce") if "budget" in cols else np.nan
            )
            df_out = df_out.dropna(subset=["timestamp"]).copy()
            if df_out.empty:
                continue
            if df_out["timestamp"].dt.tz is None:
                df_out["timestamp"] = df_out["timestamp"].dt.tz_localize(
                    "Europe/Zurich", nonexistent="shift_forward", ambiguous="NaT"
                )
            df_out = df_out[df_out["timestamp"].notna()].sort_values("timestamp")

            df_out["delta_mw"] = df_out["actual_mw"] - df_out["programme_mw"]
            df_out["delta_pct"] = (
                100.0 * df_out["delta_mw"] / df_out["programme_mw"].replace(0, np.nan)
            )
            df_out["month"] = df_out["timestamp"].dt.to_period("M").dt.to_timestamp()
            df_out["date"] = df_out["timestamp"].dt.date
            df_out["hour"] = df_out["timestamp"].dt.hour
            df_out["dow"] = df_out["timestamp"].dt.dayofweek
            df_out["iso_week"] = df_out["timestamp"].dt.isocalendar().week.astype(int)
            df_out["year"] = df_out["timestamp"].dt.year

            actor_norm = actor_raw.strip()
            mapped = _assign_actor(actor_norm.lower()) or _assign_actor(sheet.lower())
            if mapped:
                actor_norm = mapped

            if actor_norm in out:
                out[actor_norm] = pd.concat([out[actor_norm], df_out], ignore_index=True)
            else:
                out[actor_norm] = df_out.reset_index(drop=True)

    return out


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
