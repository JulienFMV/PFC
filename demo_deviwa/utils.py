"""Gemeinsame Hilfsfunktionen und Datenlader für die FMV Deviwa-Demo.

Alles bleibt lokal und dateibasiert, damit die Demo offline läuft.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PFC_DATA_DIR = ROOT / "pfc_shaping" / "data"
OUTPUT_DIR = ROOT / "pfc_shaping" / "output"
EEX_YEARLY_PATH = Path(r"H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX_Yearly.xlsx")
HFC_OMPEX_DIR = Path(r"H:\Energy\GeCom\MARCHE & NEGOCE\Prix\Analyse HFC\HFC test\ER -HFC_OMPEX_15min")

# Lokaler Schreib-durch-Cache für langsame H:\-Zugriffe (legacy, für Backward-
# Compatibility erhalten). Wird bei erstem erfolgreichen Lesen einer H:\-Datei
# automatisch gefüllt und bei jedem weiteren Start direkt gelesen.
H_CACHE_DIR = DATA_DIR / "_h_cache"
H_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Neuer canonical Parquet-Cache (durch scripts/extract_deviwa_cache.py befüllt).
# Alle Pages lesen primär hier — falls eine Datei fehlt, fallback auf den
# alten Pfad (legacy parquet, H:\, oder direkter Excel-Read).
CACHE_DIR = DATA_DIR / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FMV_BLUE = "#0F52CC"
FMV_NAVY = "#0E1F3D"
FMV_ACCENT = "#F5B700"
FMV_GREEN = "#1E8A4C"
FMV_RED = "#C0392B"
FMV_GREY = "#6B7A99"


# ---------------------------------------------------------------------------
# Formatierung (Schweizer Konvention)
# ---------------------------------------------------------------------------

def fmt_chf(value: float, decimals: int = 0) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    sign = "-" if value < 0 else ""
    abs_val = abs(float(value))
    whole, frac = divmod(abs_val, 1.0)
    whole_str = f"{int(whole):,}".replace(",", "'")
    if decimals == 0:
        return f"{sign}{whole_str}"
    frac_str = f"{frac:.{decimals}f}"[2:]
    return f"{sign}{whole_str}.{frac_str}"


def fmt_eur_mwh(value: float, decimals: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{fmt_chf(value, decimals=decimals)} EUR/MWh"


def fmt_mwh(value: float, decimals: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{fmt_chf(value, decimals=decimals)} MWh"


def fmt_delta(value: float, suffix: str = "") -> str:
    if value is None or not np.isfinite(value):
        return "—"
    sign = "+" if value >= 0 else ""
    return f"{sign}{fmt_chf(value, decimals=2)}{suffix}"


# ---------------------------------------------------------------------------
# Daten-Loader (mit Caching)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Lade EPEX CH Spot …")
def load_epex_ch() -> pd.DataFrame:
    path = PFC_DATA_DIR / "epex_15min.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df = df.rename(columns={"price_eur_mwh": "price"})
    df = df[["price"]]
    df.index = df.index.tz_convert("Europe/Zurich")
    return df


@st.cache_data(show_spinner=False)
def load_epex_de() -> pd.DataFrame:
    path = PFC_DATA_DIR / "epex_de_15min.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = df.index.tz_convert("Europe/Zurich")
    return df


@st.cache_data(show_spinner="Lade ENTSO-E Daten …")
def load_entso() -> pd.DataFrame:
    path = PFC_DATA_DIR / "entso_15min.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = df.index.tz_convert("Europe/Zurich")
    return df


@st.cache_data(show_spinner=False)
def load_hydro() -> pd.DataFrame:
    path = PFC_DATA_DIR / "hydro_reservoir.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = df.index.tz_convert("Europe/Zurich")
    return df


def _read_with_cache(
    src_path: Path,
    cache_name: str,
    reader,
) -> pd.DataFrame | None:
    """Write-through Cache: liest aus H:\\ nur wenn die lokale Parquet-Kopie
    älter ist als das Original (mtime-Vergleich). Spart bei jedem zweiten
    Start den kompletten Netzwerk-Download.
    """
    cache_path = H_CACHE_DIR / f"{cache_name}.parquet"
    try:
        src_mtime = src_path.stat().st_mtime
    except OSError:
        if cache_path.exists():
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                return None
        return None

    if cache_path.exists():
        try:
            if cache_path.stat().st_mtime >= src_mtime:
                return pd.read_parquet(cache_path)
        except Exception:
            pass

    try:
        df = reader(src_path)
    except Exception:
        if cache_path.exists():
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                return None
        return None

    if df is None or df.empty:
        return df
    try:
        df.to_parquet(cache_path)
    except Exception:
        pass
    return df


def _read_eex_yearly(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="CH", header=None)
    if raw.shape[0] < 3 or raw.shape[1] < 2:
        return pd.DataFrame()
    codes = raw.iloc[0].astype(str)
    labels = raw.iloc[1].astype(str)
    dates = pd.to_datetime(raw.iloc[2:, 0], errors="coerce", dayfirst=True)
    rows = []
    for col_idx in range(1, raw.shape[1]):
        code = str(codes.iloc[col_idx])
        label = str(labels.iloc[col_idx])
        if not label or label.lower() == "nan" or label.startswith("Unnamed"):
            label = code
        load_type = "peak" if "PEAK" in code.upper() or " PEAK" in label.upper() else "base"
        prices = pd.to_numeric(raw.iloc[2:, col_idx], errors="coerce")
        mask = dates.notna() & prices.notna() & (prices > 0)
        if not mask.any():
            continue
        rows.append(pd.DataFrame({
            "date": dates[mask].values,
            "market": "CH",
            "product": f"{label} {code}",
            "load_type": load_type,
            "price": prices[mask].values,
            "source": "Price_Report_EEX_Yearly.xlsx",
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


@st.cache_data(show_spinner="Lade EEX Forwards …")
def load_eex_forwards() -> pd.DataFrame:
    # 1) Nouveau cache parquet canonical (peuplé par extract_deviwa_cache.py)
    canonical = CACHE_DIR / "eex_yearly_ch.parquet"
    if canonical.exists():
        df = pd.read_parquet(canonical)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # 2) Legacy : write-through depuis H:\ Windows
    if EEX_YEARLY_PATH.exists():
        cached = _read_with_cache(EEX_YEARLY_PATH, "eex_yearly_ch", _read_eex_yearly)
        if cached is not None and not cached.empty:
            return cached

    # 3) Fallback final : ancien parquet versionné
    path = DATA_DIR / "eex_forwards_history.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_commodities() -> pd.DataFrame:
    path = DATA_DIR / "commodities_cache.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.columns = [c.split("|")[0].strip() for c in df.columns]
    return df


@st.cache_data(show_spinner="Lade LEAR-Prognose …")
def load_lear_forecast() -> pd.DataFrame:
    path = OUTPUT_DIR / "lear_forecast_latest.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/Zurich")
    return df


@st.cache_data(show_spinner=False)
def load_lear_backtest() -> pd.DataFrame:
    path = OUTPUT_DIR / "lear_backtest_latest.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["forecast_ts"] = pd.to_datetime(df["forecast_ts"], utc=True)
    df["forecast_ts"] = df["forecast_ts"].dt.tz_convert("Europe/Zurich")
    return df


def _read_hfc_xlsx(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="HFC")
    if not {"Date", "EUR/MWh"}.issubset(df.columns):
        return pd.DataFrame()
    ts = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    price = pd.to_numeric(df["EUR/MWh"], errors="coerce")
    out = pd.DataFrame({"price_shape": price.values}, index=ts)
    out = out.dropna(subset=["price_shape"])
    out = out[out.index.notna()].sort_index()
    if out.empty:
        return pd.DataFrame()
    out.index = pd.DatetimeIndex(out.index)
    if out.index.tz is None:
        out.index = out.index.tz_localize(
            "Europe/Zurich", nonexistent="shift_forward", ambiguous="NaT"
        )
        out = out[out.index.notna()]
    else:
        out.index = out.index.tz_convert("Europe/Zurich")
    out["source_file"] = path.name
    return out


@st.cache_data(show_spinner="Lade PFC …")
def load_pfc() -> pd.DataFrame:
    """Letzte verfügbare PFC 15-Minuten-Kurve.

    Reihenfolge: 1) canonical parquet cache, 2) write-through H:\\, 3) local PFC parquet output.
    """
    # 1) Canonical parquet
    canonical = CACHE_DIR / "pfc_ompex.parquet"
    if canonical.exists():
        try:
            df = pd.read_parquet(canonical)
            if not df.empty:
                if df.index.tz is None:
                    df.index = df.index.tz_localize(
                        "Europe/Zurich", nonexistent="shift_forward", ambiguous="NaT"
                    )
                else:
                    df.index = df.index.tz_convert("Europe/Zurich")
                return df
        except Exception:
            pass

    # 2) Legacy : write-through H:\
    if HFC_OMPEX_DIR.exists():
        # Engerer Glob: nur HFC_Ompex_*.xlsx und nur die 2 neuesten Kandidaten
        # testen, um die Anzahl der SMB-Roundtrips drastisch zu reduzieren.
        try:
            candidates = list(HFC_OMPEX_DIR.glob("HFC_Ompex_*.xlsx"))
            if not candidates:
                candidates = list(HFC_OMPEX_DIR.glob("HFC_OMPEX_*.xlsx"))
            if not candidates:
                candidates = list(HFC_OMPEX_DIR.glob("HFC*.xlsx"))
            candidates = sorted(candidates, key=lambda p: p.name, reverse=True)[:2]
        except OSError:
            candidates = []

        for path in candidates:
            cached = _read_with_cache(
                path,
                f"hfc_ompex_{path.stem}",
                _read_hfc_xlsx,
            )
            if cached is not None and not cached.empty:
                return cached

    candidates_local = sorted(OUTPUT_DIR.glob("pfc_15min_*.parquet"))
    if not candidates_local:
        return pd.DataFrame()
    df = pd.read_parquet(candidates_local[-1])
    df.index = pd.to_datetime(df.index).tz_convert("Europe/Zurich")
    return df


# ---------------------------------------------------------------------------
# Gemeinsame UI-Bausteine
# ---------------------------------------------------------------------------

def render_header(page_title: str) -> None:
    st.markdown(
        f"""
        <div style='display:flex; align-items:center; justify-content:space-between;
                    border-bottom: 3px solid {FMV_BLUE}; padding-bottom: 0.6rem;
                    margin-bottom: 1.2rem;'>
            <div>
                <div style='font-size:0.85rem; letter-spacing:0.12em; color:{FMV_GREY};
                            text-transform:uppercase;'>FMV – Forces Motrices Valaisannes</div>
                <div style='font-size:1.8rem; font-weight:700; color:{FMV_NAVY};'>{page_title}</div>
            </div>
            <div style='text-align:right; font-size:0.8rem; color:{FMV_GREY};'>
                Deviwa Energiepool · Demo<br/>
                Sicht: <strong style='color:{FMV_BLUE};'>Kunde</strong> · Stand: {pd.Timestamp.now(tz='Europe/Zurich').strftime('%d.%m.%Y %H:%M')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, delta: str | None = None,
             delta_color: str = FMV_NAVY, help_text: str | None = None) -> str:
    delta_html = (
        f"<div style='font-size:0.9rem; color:{delta_color}; margin-top:0.2rem;'>{delta}</div>"
        if delta else ""
    )
    tooltip = f" title='{help_text}'" if help_text else ""
    return f"""
    <div{tooltip} style='background:{"#FFFFFF"}; border:1px solid #E5EBF4;
                border-radius:10px; padding:0.9rem 1.1rem; height:100%;
                box-shadow: 0 1px 2px rgba(14,31,61,0.04);'>
        <div style='font-size:0.78rem; color:{FMV_GREY}; text-transform:uppercase;
                    letter-spacing:0.08em;'>{label}</div>
        <div style='font-size:1.55rem; font-weight:700; color:{FMV_NAVY}; margin-top:0.25rem;'>{value}</div>
        {delta_html}
    </div>
    """


# ---------------------------------------------------------------------------
# Gemeinsamer Deviwa-Loader und Akteur-Auswahl
# ---------------------------------------------------------------------------

DEVIWA_CANDIDATES = [
    "Deviwa.xlsx", "Deviwa.csv", "deviwa.xlsx", "deviwa.csv",
    "Deviwa_Deals.xlsx", "deviwa_deals.xlsx",
]


def find_deviwa_file() -> Path | None:
    for name in DEVIWA_CANDIDATES:
        p = DATA_DIR / name
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner="Lade Deviwa-Deals …")
def _load_deviwa_cached(path_str: str, mtime: float) -> dict[str, pd.DataFrame]:
    from deviwa_parser import load_deviwa_file  # type: ignore
    return load_deviwa_file(path_str)


# ---------------------------------------------------------------------------
# Canonical parquet-cache loaders (peuplé par scripts/extract_deviwa_cache.py)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_deviwa_deals_cached() -> pd.DataFrame:
    """Long-format deals frame (scope déjà flippé en vue client)."""
    p = CACHE_DIR / "deviwa_deals.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def load_deviwa_programme_cached() -> pd.DataFrame:
    """Long-format programme/real par acteur (timestamp tz-aware Europe/Zurich)."""
    p = CACHE_DIR / "deviwa_programme.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "timestamp" in df.columns and df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("Europe/Zurich", ambiguous="NaT")
    return df


@st.cache_data(show_spinner=False)
def load_deviwa_edsh_suppliers_cached() -> pd.DataFrame:
    """8-col EDSH avec décomposition par fournisseur (BKW/EnAlpin/EWZ/FMV/Spot)."""
    p = CACHE_DIR / "deviwa_edsh_suppliers.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def load_cache_meta() -> dict:
    """Lit data/_cache/_meta.json — sources, fingerprints, quality issues."""
    meta_file = CACHE_DIR / "_meta.json"
    if not meta_file.exists():
        return {}
    try:
        import json
        return json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_deviwa_auto() -> dict[str, pd.DataFrame]:
    """Lädt Deviwa-Daten automatisch.

    Reihenfolge:
    1) Parquet cache (data/_cache/deviwa_deals.parquet) — schnell
    2) Direktes Excel-Read (legacy) — langsam, fallback wenn Cache fehlt
    """
    # 1) Cache parquet
    deals = load_deviwa_deals_cached()
    if not deals.empty and "actor" in deals.columns:
        result: dict[str, pd.DataFrame] = {}
        for actor, sub in deals.groupby("actor"):
            result[str(actor)] = sub.reset_index(drop=True).copy()
        return result

    # 2) Fallback Excel direct
    p = find_deviwa_file()
    if p is None:
        return {}
    data = _load_deviwa_cached(str(p), p.stat().st_mtime)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def refresh_cache(force: bool = False) -> tuple[bool, str]:
    """Re-execute le pipeline d'extraction parquet et clear les caches Streamlit.

    Retourne (success, message). À appeler depuis un bouton sidebar.
    """
    try:
        # Import retardé pour éviter le coût au cold start si le bouton n'est pas cliqué
        import sys as _sys
        scripts_path = ROOT / "scripts"
        if str(scripts_path) not in _sys.path:
            _sys.path.insert(0, str(scripts_path))
        from extract_deviwa_cache import run_all  # type: ignore
        results = run_all(force=force, quiet=True)
        n_extracted = sum(1 for r in results if r.action == "extracted")
        n_skipped = sum(1 for r in results if r.action == "skipped_up_to_date")
        n_failed = sum(1 for r in results if r.action == "failed")
        # Vide les caches Streamlit pour que les nouvelles données soient relues
        st.cache_data.clear()
        msg = f"Cache aktualisiert · {n_extracted} extrahiert, {n_skipped} aktuell"
        if n_failed > 0:
            msg += f" · {n_failed} fehlgeschlagen"
        return (n_failed == 0, msg)
    except Exception as e:
        return (False, f"Fehler: {type(e).__name__}: {e}")


def render_cache_status() -> None:
    """Affiche un badge sidebar avec l'état du cache + bouton refresh."""
    meta = load_cache_meta()
    if not meta:
        st.sidebar.markdown(
            f"<div style='color:{FMV_RED}; font-size:0.85rem;'>"
            "● Cache leer — bitte aktualisieren</div>",
            unsafe_allow_html=True,
        )
    else:
        from datetime import datetime as _dt
        last_extracts = []
        all_issues: list[str] = []
        for k, v in meta.items():
            ts = v.get("extracted_at")
            if ts:
                last_extracts.append(_dt.fromisoformat(ts.replace("Z", "+00:00")))
            all_issues.extend(v.get("quality_issues", []) or [])
        most_recent = max(last_extracts) if last_extracts else None
        if most_recent:
            color = FMV_GREEN if (pd.Timestamp.utcnow() - pd.Timestamp(most_recent)).total_seconds() < 86400 else FMV_ACCENT
            st.sidebar.markdown(
                f"<div style='color:{color}; font-size:0.85rem;'>"
                f"● Cache: {most_recent.strftime('%d.%m.%Y %H:%M')}</div>",
                unsafe_allow_html=True,
            )
        if all_issues:
            with st.sidebar.expander(f"⚠️ {len(all_issues)} Qualitäts-Hinweis(e)"):
                for it in all_issues[:15]:
                    st.caption(f"• {it}")

    if st.sidebar.button("🔄 Daten aktualisieren", use_container_width=True):
        with st.spinner("Aktualisiere Cache..."):
            ok, msg = refresh_cache(force=False)
        if ok:
            st.sidebar.success(msg)
            st.rerun()
        else:
            st.sidebar.error(msg)


def render_actor_selector(
    data: dict[str, pd.DataFrame],
    label: str = "Akteur auswählen",
    include_pool: bool = True,
    key: str | None = None,
) -> tuple[str, pd.DataFrame]:
    """Zeigt einen konsistenten Akteur-Selector und gibt (choice, filtered_df) zurück."""
    actors = [a for a in data.keys() if not a.startswith("_")]
    options = (["Gesamter Pool"] if include_pool else []) + actors
    if not options:
        st.warning("Keine Akteure gefunden.")
        return ("", pd.DataFrame())
    choice = st.radio(label, options=options, horizontal=True, key=key)
    if choice == "Gesamter Pool":
        frames = [v.assign(_actor=k) if "_actor" not in v.columns else v
                  for k, v in data.items() if v is not None and not v.empty]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        df = data.get(choice, pd.DataFrame()).copy()
    return choice, df


def freshness_badge(df: pd.DataFrame, label: str = "Datenstand") -> str:
    if df is None or df.empty:
        return f"<span style='color:{FMV_RED};'>❌ {label}: keine Daten</span>"
    if isinstance(df.index, pd.DatetimeIndex):
        last = df.index.max()
    else:
        ts_col = df.get("timestamp", df.get("date"))
        last = pd.to_datetime(ts_col.max()) if ts_col is not None else None
    if last is None or pd.isna(last):
        return f"<span style='color:{FMV_RED};'>❌ {label}: keine Daten</span>"
    ts = pd.Timestamp(last)
    if ts.tzinfo is None:
        ts = ts.tz_localize("Europe/Zurich")
    else:
        ts = ts.tz_convert("Europe/Zurich")
    age_hours = (pd.Timestamp.now(tz="Europe/Zurich") - ts).total_seconds() / 3600
    color = FMV_GREEN if age_hours < 36 else (FMV_ACCENT if age_hours < 96 else FMV_RED)
    return (f"<span style='color:{color};'>● {label}: "
            f"{ts.strftime('%d.%m.%Y %H:%M')}</span>")
