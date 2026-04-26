"""Gemeinsame Hilfsfunktionen und Datenlader für die FMV Deviwa-Demo.

Alles bleibt lokal und dateibasiert, damit die Demo offline läuft.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger("demo_deviwa")

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
        except (FileNotFoundError, OSError, ValueError) as exc:
            # Cache parquet présent mais corrompu — log et fallback explicite
            logger.warning(
                "PFC cache parquet illisible (%s) — fallback H:\\/legacy. Détail: %s",
                canonical, exc,
            )
        else:
            if not df.empty:
                if df.index.tz is None:
                    df.index = df.index.tz_localize(
                        "Europe/Zurich", nonexistent="shift_forward", ambiguous="NaT"
                    )
                else:
                    df.index = df.index.tz_convert("Europe/Zurich")
                return df

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
    """Returns ``{actor_name: deals_DataFrame}`` for the Deviwa pool.

    **Contract (post-Phase 1 refactor)** : the per-actor DataFrames contain
    **deal rows ONLY**, with the standardized columns from
    ``deviwa_parser`` (deal, scope, volume_sum, notional_sum, ...) plus
    canonical metadata columns ``actor``, ``source_sheet``.

    For backward compatibility with callers that grepped on the legacy
    ``_actor`` / ``_sheet`` columns, those names are added as aliases.

    **This is intentionally NOT** the same shape as the legacy
    ``deviwa_parser.load_deviwa_file()`` which mixed deal rows together
    with raw programme rows under each actor key (~35k rows per actor
    instead of just 24-120 deal rows). The mixed shape was confusing and
    error-prone (it contributed to the bugged hedge_ratio_pct calculation
    in the original portfolio_analytics).

    For programme/real time-series, use ``load_deviwa_programme_cached()``.
    For EDSH supplier breakdown, use ``load_deviwa_edsh_suppliers_cached()``.

    Resolution order :
    1. Parquet cache (data/_cache/deviwa_deals.parquet) — fast
    2. Direct Excel read (legacy fallback) — slow, only if cache absent
    """
    # 1) Cache parquet
    deals = load_deviwa_deals_cached()
    if not deals.empty and "actor" in deals.columns:
        result: dict[str, pd.DataFrame] = {}
        for actor, sub in deals.groupby("actor"):
            df = sub.reset_index(drop=True).copy()
            # Backward-compat aliases for callers that grepped on legacy names
            if "_actor" not in df.columns:
                df["_actor"] = df["actor"]
            if "_sheet" not in df.columns and "source_sheet" in df.columns:
                df["_sheet"] = df["source_sheet"]
            result[str(actor)] = df
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
            now_utc = pd.Timestamp.now(tz="UTC")
            extracted_ts = pd.Timestamp(most_recent)
            if extracted_ts.tzinfo is None:
                extracted_ts = extracted_ts.tz_localize("UTC")
            color = FMV_GREEN if (now_utc - extracted_ts).total_seconds() < 86400 else FMV_ACCENT
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


# ---------------------------------------------------------------------------
# Phase 4 visual helpers : year card + load/hedge chart + EDSH supplier stack
# + hedge ladder. Imported by demo_deviwa/app.py.
# ---------------------------------------------------------------------------


def _hedge_status_color(status: str) -> str:
    """Map portfolio_yearly.YearlyHedgeMetrics.target_status → FMV palette color."""
    return {
        "in_corridor": FMV_GREEN,
        "below": FMV_RED,
        "above": FMV_BLUE,
        "no_data": FMV_GREY,
        "n/a": FMV_GREY,
    }.get(status, FMV_GREY)


def render_year_card(
    year: int,
    metrics,  # YearlyHedgeMetrics | None
    budget,  # BudgetProjection | None
    today=None,
) -> str:
    """Render a single calendar-year card as a single-line HTML string.

    Important : Streamlit's ``st.markdown(html, unsafe_allow_html=True)``
    runs Markdown rendering BEFORE HTML pass-through. Lines indented by
    four or more spaces are treated as Markdown code blocks, which would
    surface raw HTML tags as text in the rendered card. We therefore emit
    a single-line string with no leading whitespace.

    Discipline visuelle :
      - Max 5 chiffres : Hedge %, mini-gauge, Ø price vs marché, Open MWh,
        budget central ± P10/P90 band.
      - Empty cards (no programme, no deals) → calm grey "Noch keine
        Absicherung" — informative, not alarming.
      - Color is driven by target_status (in_corridor / below / above).
    """
    if today is None:
        today = pd.Timestamp.now(tz="Europe/Zurich")
    title_suffix = " (in delivery)" if year == today.year else ""

    # ── No data → empty card ────────────────────────────────────────────────
    if metrics is None or (
        metrics.programme_mwh == 0 and metrics.hedged_mwh == 0
    ):
        return (
            f"<div style='background:#FAFBFD; border:1px solid #E5EBF4;"
            f" border-radius:10px; padding:1.0rem 1.1rem; height:100%;"
            f" box-shadow: 0 1px 2px rgba(14,31,61,0.04);'>"
            f"<div style='font-size:0.78rem; color:{FMV_GREY};"
            f" text-transform:uppercase; letter-spacing:0.08em;'>"
            f"Cal-{year}{title_suffix}</div>"
            f"<div style='font-size:1.55rem; font-weight:700; color:{FMV_GREY};"
            f" margin-top:0.25rem;'>—</div>"
            f"<div style='font-size:0.85rem; color:{FMV_GREY};"
            f" margin-top:0.5rem; line-height:1.4;'>"
            f"Noch keine<br/>Absicherung</div>"
            f"</div>"
        )

    # ── Card with data ──────────────────────────────────────────────────────
    ratio = metrics.hedge_ratio_pct
    ratio_str = f"{ratio:.0f} %" if np.isfinite(ratio) else "—"
    color = _hedge_status_color(metrics.target_status)

    # Mini-gauge horizontale (clipped to [0, 120%] for the on-screen scale)
    gauge_pct = max(0.0, min(120.0, ratio if np.isfinite(ratio) else 0.0))
    gauge_width = int(round(gauge_pct / 120.0 * 100))

    # Corridor markers
    corridor_html = ""
    if metrics.target_low_pct is not None and metrics.target_high_pct is not None:
        low_pos = int(round(metrics.target_low_pct / 120.0 * 100))
        high_pos = int(round(metrics.target_high_pct / 120.0 * 100))
        corridor_html = (
            f"<div style='position:absolute; left:{low_pos}%; top:-2px;"
            f" bottom:-2px; width:1px; background:{FMV_NAVY};"
            f" opacity:0.4;'></div>"
            f"<div style='position:absolute; left:{high_pos}%; top:-2px;"
            f" bottom:-2px; width:1px; background:{FMV_NAVY};"
            f" opacity:0.4;'></div>"
        )

    target_text = ""
    if metrics.target_low_pct is not None:
        target_text = (
            f"<div style='font-size:0.72rem; color:{FMV_GREY};"
            f" margin-top:0.15rem;'>Ziel {metrics.target_low_pct}-"
            f"{metrics.target_high_pct} %</div>"
        )

    # Ø Hedge price vs Markt
    price_html = ""
    if np.isfinite(metrics.avg_hedge_price_eur_mwh):
        markt = budget.forward_price_eur_mwh if budget else float("nan")
        delta = (
            metrics.avg_hedge_price_eur_mwh - markt
            if np.isfinite(markt) else float("nan")
        )
        delta_color = (
            FMV_GREEN if np.isfinite(delta) and delta < 0
            else FMV_RED if np.isfinite(delta) and delta > 0
            else FMV_GREY
        )
        markt_str = f" · Markt {markt:.1f}" if np.isfinite(markt) else ""
        delta_str = f" ({delta:+.1f})" if np.isfinite(delta) else ""
        price_html = (
            f"<div style='font-size:0.85rem; color:{FMV_NAVY};"
            f" margin-top:0.6rem;'>Ø Hedge {metrics.avg_hedge_price_eur_mwh:.1f}"
            f"<span style='color:{delta_color};'>{markt_str}{delta_str}</span>"
            f"</div>"
        )

    # Budget projection line
    budget_html = ""
    if budget is not None and np.isfinite(budget.central_eur):
        band = abs(budget.p90_eur - budget.central_eur)
        budget_html = (
            f"<div style='font-size:0.85rem; color:{FMV_NAVY};"
            f" margin-top:0.35rem;'>Budget {fmt_chf(budget.central_eur, 0)} EUR"
            f" <span style='color:{FMV_GREY};'>± {fmt_chf(band, 0)}</span>"
            f"</div>"
        )

    open_html = (
        f"<div style='font-size:0.85rem; color:{FMV_NAVY};"
        f" margin-top:0.35rem;'>Offen {fmt_mwh(metrics.open_mwh, 0)}</div>"
    )

    return (
        f"<div style='background:#FFFFFF; border:1px solid #E5EBF4;"
        f" border-radius:10px; padding:1.0rem 1.1rem; height:100%;"
        f" box-shadow: 0 1px 2px rgba(14,31,61,0.04);'>"
        f"<div style='font-size:0.78rem; color:{FMV_GREY};"
        f" text-transform:uppercase; letter-spacing:0.08em;'>"
        f"Cal-{year}{title_suffix}</div>"
        f"<div style='font-size:1.55rem; font-weight:700; color:{color};"
        f" margin-top:0.25rem;'>{ratio_str}</div>"
        f"<div style='position:relative; height:8px; background:#EEF1F6;"
        f" border-radius:4px; margin-top:0.35rem; overflow:visible;'>"
        f"<div style='width:{gauge_width}%; height:100%; background:{color};"
        f" border-radius:4px;'></div>{corridor_html}</div>"
        f"{target_text}{price_html}{open_html}{budget_html}"
        f"</div>"
    )


def render_load_hedge_chart(
    programme: pd.DataFrame,
    deals: pd.DataFrame,
    delivery_year: int,
    actor: "str | None",
):
    """Two-panel Plotly figure for the year detail view.

    Top panel — "Was Sie verbrauchen / produzieren" :
      - Programme net (bars) : forecast monthly load.
      - Real (line+markers, when available) : the realized monthly load.
        Helps see where the forecast already proved right or wrong YTD.

    Bottom panel — "Wie Sie abgesichert sind" :
      - Programme reference (line) : the level we need to reach.
      - Hedge (bars, green) : volume already locked in via FMV deals.
      - Offene Position (bars, red, stacked on hedge) : the residual
        exposure that fills the gap up to programme.
        → Visually : hedge + offen = programme. The user sees in one
        glance how much is locked vs how much is still at market risk.

    For Withdrawal-net months (= net short, e.g. EVTL Cal-27 over-hedged)
    the hedge bar can exceed programme. We surface that explicitly with a
    grey "Sur-hedge" bar above programme rather than letting the stack
    look misleading.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.5, 0.5],
        subplot_titles=(
            "Was Sie verbrauchen / produzieren (MWh / Monat)",
            "Wie Sie abgesichert sind (MWh / Monat)",
        ),
    )

    # ── Compute monthly programme (used by both panels) ─────────────────────
    monthly_prog = pd.Series(dtype=float)
    monthly_actual = pd.Series(dtype=float)
    if not programme.empty and "timestamp" in programme.columns:
        prog_y = programme.copy()
        if actor is not None:
            prog_y = prog_y[prog_y["actor"] == actor]
        prog_y = prog_y[prog_y["timestamp"].dt.year == delivery_year]
        if not prog_y.empty:
            prog_y = prog_y.copy()
            prog_y["_month"] = prog_y["timestamp"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
            monthly_prog = prog_y.groupby("_month")["programme_mw"].sum()
            # Hide all-zero programme (means no forecast for that year yet)
            if (monthly_prog == 0).all():
                monthly_prog = pd.Series(dtype=float)
            if "actual_mw" in prog_y.columns and prog_y["actual_mw"].notna().any():
                monthly_actual = prog_y.groupby("_month")["actual_mw"].sum()
                # Drop months that are still entirely zero (= future, no real yet)
                monthly_actual = monthly_actual[monthly_actual != 0]

    # ── Compute monthly hedge from deals ────────────────────────────────────
    monthly_hedge = pd.Series(dtype=float)
    if not deals.empty:
        d = deals.copy()
        if actor is not None:
            d = d[d["actor"] == actor]
        d["_month"] = pd.to_datetime(d["month"], errors="coerce")
        d = d[d["_month"].dt.year == delivery_year]
        if not d.empty:
            vol = pd.to_numeric(d["volume_sum"], errors="coerce").fillna(0.0).abs()
            scope_lower = d["scope"].astype(str).str.lower()
            signed = vol.where(scope_lower.str.contains("intake", na=False), -vol)
            d["_signed_vol"] = signed
            monthly_hedge = d.groupby("_month")["_signed_vol"].sum()

    # ── Top panel : programme bars + real overlay ──────────────────────────
    if not monthly_prog.empty:
        fig.add_trace(
            go.Bar(
                x=monthly_prog.index, y=monthly_prog.values,
                name="Programme net",
                marker_color=FMV_BLUE, opacity=0.85,
                hovertemplate="%{x|%b %Y}<br>Programme: %{y:.0f} MWh<extra></extra>",
            ),
            row=1, col=1,
        )
    if not monthly_actual.empty:
        fig.add_trace(
            go.Scatter(
                x=monthly_actual.index, y=monthly_actual.values,
                mode="lines+markers", name="Real",
                line=dict(color=FMV_NAVY, width=2.5),
                marker=dict(size=7),
                hovertemplate="%{x|%b %Y}<br>Real: %{y:.0f} MWh<extra></extra>",
            ),
            row=1, col=1,
        )

    # ── Bottom panel : hedge stacked + open stacked + programme line ──────
    # Align all series on a common monthly axis
    all_months = monthly_prog.index.union(monthly_hedge.index)
    if not all_months.empty:
        prog_a = monthly_prog.reindex(all_months, fill_value=0.0).astype(float)
        hedge_a = monthly_hedge.reindex(all_months, fill_value=0.0).astype(float)

        # Decompose : hedge_within = clipped to [0, programme] (the part that
        # actually covers programme), over_hedge = max(hedge − programme, 0)
        # (the part that exceeds programme, = sold-short via Withdrawal).
        # offene_position = max(programme − hedge, 0) (residual gap).
        hedge_within = np.minimum(np.maximum(hedge_a, 0.0), prog_a.where(prog_a > 0, 0.0))
        offene = np.maximum(prog_a - hedge_a, 0.0)
        # Net-short component (hedge above programme, OR negative net hedge
        # = client sold more than they consume)
        over_hedge = np.maximum(hedge_a - prog_a, 0.0)
        # When hedge is negative (short position) treat its magnitude as
        # "Sur-hedge / Verkauf-Position" displayed below the zero axis
        net_short = np.minimum(hedge_a, 0.0).abs()

        # Stack 1 : hedge_within (covers part of programme)
        fig.add_trace(
            go.Bar(
                x=all_months, y=hedge_within,
                name="Hedge (FMV)",
                marker_color=FMV_GREEN, opacity=0.88,
                hovertemplate="%{x|%b %Y}<br>Hedge: %{y:.0f} MWh<extra></extra>",
            ),
            row=2, col=1,
        )
        # Stack 2 : open position (gap to programme, on top of hedge)
        fig.add_trace(
            go.Bar(
                x=all_months, y=offene,
                name="Offene Position",
                marker_color="#E08A2B", opacity=0.85,
                hovertemplate="%{x|%b %Y}<br>Offen: %{y:.0f} MWh<extra></extra>",
            ),
            row=2, col=1,
        )
        # Stack 3 : over-hedge (bars beyond programme, flagged grey)
        if (over_hedge > 0).any():
            fig.add_trace(
                go.Bar(
                    x=all_months, y=over_hedge,
                    name="Sur-Hedge",
                    marker_color="#94A3B8", opacity=0.75,
                    hovertemplate="%{x|%b %Y}<br>Sur-Hedge: %{y:.0f} MWh<extra></extra>",
                ),
                row=2, col=1,
            )
        # Net-short (negative bars below zero, when hedge is itself negative
        # = client sold short with FMV)
        if (net_short > 0).any():
            fig.add_trace(
                go.Bar(
                    x=all_months, y=-net_short,
                    name="Verkauf (Withdrawal)",
                    marker_color=FMV_RED, opacity=0.78,
                    hovertemplate="%{x|%b %Y}<br>Verkauf: %{y:.0f} MWh<extra></extra>",
                ),
                row=2, col=1,
            )
        # Programme reference line
        if not monthly_prog.empty:
            fig.add_trace(
                go.Scatter(
                    x=monthly_prog.index, y=monthly_prog.values,
                    mode="lines+markers", name="Programme (Ziel)",
                    line=dict(color=FMV_NAVY, width=2.2, dash="solid"),
                    marker=dict(size=6),
                    hovertemplate="%{x|%b %Y}<br>Programme: %{y:.0f} MWh<extra></extra>",
                ),
                row=2, col=1,
            )

    fig.update_layout(
        height=560,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.08),
        hovermode="x unified",
        bargap=0.2,
        barmode="stack",  # the bottom panel needs stack mode
    )
    fig.update_xaxes(gridcolor="#EEF1F6", row=1, col=1)
    fig.update_xaxes(gridcolor="#EEF1F6", row=2, col=1)
    fig.update_yaxes(title_text="MWh", gridcolor="#EEF1F6", row=1, col=1)
    fig.update_yaxes(
        title_text="MWh", gridcolor="#EEF1F6", row=2, col=1,
        zeroline=True, zerolinewidth=1, zerolinecolor=FMV_GREY,
    )
    return fig


def render_edsh_supplier_stack(edsh_suppliers: pd.DataFrame, delivery_year: int):
    """Stacked area chart : EDSH supplier breakdown per month for a delivery year.

    Columns expected : timestamp, programme_mw, bkw_mw, enalpin_mw, ewz_mw, fmv_mw, spot_mw.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    if edsh_suppliers.empty or "timestamp" not in edsh_suppliers.columns:
        fig.add_annotation(text="Keine EDSH-Lieferantendaten verfügbar.", showarrow=False)
        fig.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white")
        return fig

    df = edsh_suppliers[edsh_suppliers["timestamp"].dt.year == delivery_year].copy()
    if df.empty:
        fig.add_annotation(text=f"Keine EDSH-Daten für {delivery_year}.", showarrow=False)
        fig.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white")
        return fig

    df["_month"] = df["timestamp"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    supplier_cols = [
        ("bkw_mw", "BKW", "#0F52CC"),
        ("enalpin_mw", "EnAlpin", "#1E8A4C"),
        ("ewz_mw", "EWZ", "#F5B700"),
        ("fmv_mw", "FMV", FMV_NAVY),
        ("spot_mw", "Spot (offen)", "#C0392B"),
    ]
    monthly = df.groupby("_month")[[c for c, _, _ in supplier_cols]].sum()

    for col, label, color in supplier_cols:
        if col not in monthly.columns:
            continue
        fig.add_trace(
            go.Bar(
                x=monthly.index, y=monthly[col],
                name=label, marker_color=color, opacity=0.88,
                hovertemplate="%{x|%b %Y}<br>" + label + ": %{y:.0f} MWh<extra></extra>",
            )
        )

    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
        barmode="stack",
        legend=dict(orientation="h", y=-0.12),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#EEF1F6")
    fig.update_yaxes(title_text="MWh", gridcolor="#EEF1F6")
    return fig


def render_hedge_ladder(
    deals: pd.DataFrame,
    forwards: pd.DataFrame,
    delivery_year: int,
    actor: "str | None",
):
    """Cumulative hedge build-up by trade date, with Cal-Y forward overlay.

    Two y-axes :
      - Left  : cumulative hedged MWh (signed, line)
      - Right : Cal-Y forward price history (line, shaded markers at trade days)
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Use the canonical helper for the cumulative hedge curve
    from portfolio_yearly import compute_hedge_ladder

    ladder = compute_hedge_ladder(deals, delivery_year, actor=actor)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if ladder.empty:
        fig.add_annotation(text=f"Keine Deals für Cal-{delivery_year}.", showarrow=False)
        fig.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white")
        return fig

    # Cumulative hedge (left axis)
    fig.add_trace(
        go.Scatter(
            x=ladder["trade_date"], y=ladder["cum_hedge_mwh"],
            mode="lines+markers", name="Kumulierte Hedge (MWh)",
            line=dict(color=FMV_BLUE, width=3),
            marker=dict(size=10),
            hovertemplate="%{x|%d.%m.%Y}<br>Kum. Hedge: %{y:.0f} MWh<extra></extra>",
        ),
        secondary_y=False,
    )

    # Cumulative volume-weighted average price (markers on right axis)
    fig.add_trace(
        go.Scatter(
            x=ladder["trade_date"], y=ladder["cum_avg_price"],
            mode="lines+markers", name="Ø Hedge-Preis kumuliert",
            line=dict(color=FMV_NAVY, width=2, dash="dash"),
            marker=dict(size=8, symbol="diamond"),
            hovertemplate="%{x|%d.%m.%Y}<br>Ø Preis: %{y:.2f} EUR/MWh<extra></extra>",
        ),
        secondary_y=True,
    )

    # Forward history overlay : Cal-Y when liquid (future year), or the
    # available quarterlies (Q01..Q04) when the calendar contract has retired
    # (= current year case). Same fallback logic as portfolio_yearly's
    # get_forward_year_proxy, applied tenor by tenor for the time series.
    if not forwards.empty and "product" in forwards.columns:
        fwd_base = forwards[
            forwards["load_type"].astype(str).str.lower() == "base"
        ].copy()

        cal_match = fwd_base[
            fwd_base["product"].astype(str).str.contains(
                f"Y01_{delivery_year}", na=False, regex=False
            )
        ].sort_values("date")
        if not cal_match.empty:
            fig.add_trace(
                go.Scatter(
                    x=cal_match["date"], y=cal_match["price"],
                    mode="lines", name=f"Cal-{delivery_year} Forward",
                    line=dict(color="#888888", width=1.2),
                    hovertemplate="%{x|%d.%m.%Y}<br>Cal-"
                    + str(delivery_year) + ": %{y:.2f} EUR/MWh<extra></extra>",
                ),
                secondary_y=True,
            )
        else:
            # Cal-Y absent (current year) → plot each available quarterly
            # so the user still sees a market reference. Use distinguishable
            # shades of grey so the eye groups them.
            quarter_shades = {
                1: "#B5BFCB",
                2: "#9CA9BA",
                3: "#7E8DA1",
                4: "#5E6E85",
            }
            for q in (1, 2, 3, 4):
                q_match = fwd_base[
                    fwd_base["product"].astype(str).str.contains(
                        f"Q0{q}_{delivery_year}", na=False, regex=False
                    )
                ].sort_values("date")
                if q_match.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=q_match["date"], y=q_match["price"],
                        mode="lines",
                        name=f"Q{q}-{delivery_year}",
                        line=dict(color=quarter_shades[q], width=1.0, dash="dot"),
                        hovertemplate="%{x|%d.%m.%Y}<br>Q" + str(q) + "-"
                        + str(delivery_year) + ": %{y:.2f} EUR/MWh<extra></extra>",
                    ),
                    secondary_y=True,
                )

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.18),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#EEF1F6")
    fig.update_yaxes(title_text="MWh", gridcolor="#EEF1F6", secondary_y=False)
    fig.update_yaxes(title_text="EUR/MWh", showgrid=False, secondary_y=True)
    return fig
