"""
Shared data loading, caching, and chart helpers for the PFC dashboard.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import yaml

logger = logging.getLogger("pfc_dashboard")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

COLORS = {
    "bg": "#F3F6FB",
    "card": "#FFFFFF",
    "text": "#0E1F3D",
    "amber": "#C08A2B",
    "blue": "#0F52CC",
    "green": "#1F9D55",
    "red": "#C63D3D",
    "gray": "#D4DEEE",
    "muted": "#5A6B8A",
    "band": "rgba(15,82,204,0.12)",
}

PFC_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(family="Inter, sans-serif", color=COLORS["text"], size=13),
        title=dict(font=dict(size=18, color=COLORS["text"]), x=0, xanchor="left"),
        xaxis=dict(gridcolor="#D9E2F1", zerolinecolor="#D9E2F1", showgrid=True, gridwidth=1),
        yaxis=dict(
            gridcolor="#D9E2F1",
            zerolinecolor="#D9E2F1",
            showgrid=True,
            gridwidth=1,
            title_standoff=10,
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", font=dict(size=12)),
        margin=dict(l=60, r=20, t=50, b=40),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=COLORS["card"], font_color=COLORS["text"]),
    )
)
pio.templates["pfc_dashboard"] = PFC_TEMPLATE
pio.templates.default = "pfc_dashboard"


def _safe_read_parquet(path: Path, name: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            logger.warning("%s parquet exists but is empty: %s", name, path)
            return None
        return df
    except Exception as exc:
        logger.error("Failed to read %s parquet %s: %s", name, path, exc)
        return None


def _safe_read_csv(path: Path, name: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep=";")
        if df.empty:
            logger.warning("%s csv exists but is empty: %s", name, path)
            return None
        if "timestamp_local" in df.columns:
            idx = pd.to_datetime(df["timestamp_local"], errors="coerce")
            df = df.drop(columns=["timestamp_local"]).set_index(idx)
            df = df[~df.index.isna()]
            df.index.name = "timestamp_local"
        return df
    except Exception as exc:
        logger.error("Failed to read %s csv %s: %s", name, path, exc)
        return None


def _resolve_config_path(raw_path: str | Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / "pfc_shaping" / p).resolve()


@st.cache_data(ttl=86400, show_spinner=False)
def load_config() -> dict:
    cfg_path = PROJECT_ROOT / "pfc_shaping" / "config.yaml"
    try:
        with cfg_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception as exc:
        logger.error("Failed to load config.yaml: %s", exc)
        return {}


def _paths_from_config() -> dict[str, Path]:
    cfg = load_config()
    path_cfg = cfg.get("paths", {})
    defaults = {
        "epex_parquet": "data/epex_15min.parquet",
        "entso_parquet": "data/entso_15min.parquet",
        "hydro_parquet": "data/hydro_reservoir.parquet",
        "output_dir": "output",
        "duckdb_path": "data/pfc_local.duckdb",
    }
    merged = {k: path_cfg.get(k, v) for k, v in defaults.items()}
    return {k: _resolve_config_path(v) for k, v in merged.items()}


def _latest_market_file(output_dir: Path, prefix: str, suffix: str) -> Path | None:
    dated = sorted(output_dir.glob(f"{prefix}_*{suffix}"), reverse=True)
    if dated:
        return dated[0]
    static = output_dir / f"{prefix}{suffix}"
    return static if static.exists() else None


@st.cache_data(ttl=3600, show_spinner="Chargement prix EPEX...")
def load_epex() -> pd.DataFrame | None:
    return _safe_read_parquet(_paths_from_config()["epex_parquet"], "EPEX")


@st.cache_data(ttl=3600, show_spinner="Chargement load/gen...")
def load_entso() -> pd.DataFrame | None:
    return _safe_read_parquet(_paths_from_config()["entso_parquet"], "ENTSO")


@st.cache_data(ttl=3600, show_spinner="Chargement hydro...")
def load_hydro() -> pd.DataFrame | None:
    hydro_path = _paths_from_config()["hydro_parquet"]
    df = _safe_read_parquet(hydro_path, "Hydro")
    if df is not None:
        return df
    try:
        from pfc_shaping.data.ingest_hydro import build_water_value, load_from_sfoe

        return build_water_value(load_from_sfoe())
    except Exception as exc:
        logger.error("Hydro fallback failed: %s", exc)
        return None


@st.cache_data(ttl=3600, show_spinner="Chargement PFC...")
def load_pfc() -> pd.DataFrame | None:
    output_dir = _paths_from_config()["output_dir"]
    parquet_path = _latest_market_file(output_dir, "pfc_15min", ".parquet")
    csv_path = _latest_market_file(output_dir, "pfc_15min", ".csv")
    if parquet_path is not None:
        df = _safe_read_parquet(parquet_path, "PFC")
        if df is not None:
            return df
    if csv_path is not None:
        return _safe_read_csv(csv_path, "PFC")
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_pfc_metadata() -> dict[str, str]:
    output_dir = _paths_from_config()["output_dir"]
    parquet_path = _latest_market_file(output_dir, "pfc_15min", ".parquet")
    csv_path = _latest_market_file(output_dir, "pfc_15min", ".csv")
    chosen = parquet_path or csv_path
    if chosen is None:
        return {"file": "-", "updated_at": "-"}
    ts = datetime.fromtimestamp(chosen.stat().st_mtime).strftime("%d/%m/%Y %H:%M")
    return {"file": chosen.name, "updated_at": ts}


@st.cache_data(ttl=3600, show_spinner=False)
def load_pfc_market(market: str) -> pd.DataFrame | None:
    m = str(market).upper()
    output_dir = _paths_from_config()["output_dir"]
    if m == "CH":
        prefix = "pfc_15min"
    elif m == "DE":
        prefix = "pfc_de_15min"
    else:
        return None

    parquet_path = _latest_market_file(output_dir, prefix, ".parquet")
    csv_path = _latest_market_file(output_dir, prefix, ".csv")
    if parquet_path is not None:
        df = _safe_read_parquet(parquet_path, f"PFC-{m}")
        if df is not None:
            return df
    if csv_path is not None:
        return _safe_read_csv(csv_path, f"PFC-{m}")
    return None


def _read_duckdb(sql: str, params: list | None = None) -> pd.DataFrame:
    paths = _paths_from_config()
    db_path = paths["duckdb_path"]
    if not db_path.exists():
        return pd.DataFrame()
    try:
        import duckdb

        with duckdb.connect(str(db_path), read_only=True) as con:
            if params:
                return con.execute(sql, params).fetch_df()
            return con.execute(sql).fetch_df()
    except Exception as exc:
        logger.error("DuckDB query failed: %s", exc)
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_runs(limit: int = 200) -> pd.DataFrame:
    safe_limit = max(1, int(limit))
    return _read_duckdb(
        f"""
        SELECT run_id, run_ts_utc, source_forwards, row_count, calibrated, pfc_csv_path, pfc_parquet_path
        FROM runs
        ORDER BY run_id DESC
        LIMIT {safe_limit}
        """
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_benchmarks(limit: int = 200) -> pd.DataFrame:
    safe_limit = max(1, int(limit))
    return _read_duckdb(
        f"""
        SELECT run_id, hfc_file, n_points, mae, rmse, bias, p95_abs_error, window_start, window_end
        FROM benchmarks
        ORDER BY run_id DESC
        LIMIT {safe_limit}
        """
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_forecasts_hourly(run_id: str | None = None, limit: int = 5000) -> pd.DataFrame:
    safe_limit = max(1, int(limit))
    if run_id:
        sql = (
            "SELECT run_id, ts_local, price_shape, p10, p90 "
            f"FROM forecasts_hourly WHERE run_id = ? ORDER BY ts_local LIMIT {safe_limit}"
        )
        return _read_duckdb(sql, [run_id])
    else:
        sql = (
            "SELECT run_id, ts_local, price_shape, p10, p90 "
            f"FROM forecasts_hourly ORDER BY run_id DESC, ts_local LIMIT {safe_limit}"
        )
        return _read_duckdb(sql)


def data_freshness() -> dict[str, str]:
    paths = _paths_from_config()
    pfc_meta = load_pfc_metadata()
    table = {
        "EPEX": paths["epex_parquet"],
        "Load/Gen": paths["entso_parquet"],
        "Hydro": paths["hydro_parquet"],
    }
    out: dict[str, str] = {}
    for name, path in table.items():
        if path.exists():
            out[name] = datetime.fromtimestamp(path.stat().st_mtime).strftime("%d/%m/%Y %H:%M")
        else:
            out[name] = "-"
    out["PFC"] = pfc_meta["updated_at"]
    return out


def latest_run_summary() -> dict[str, str]:
    runs = load_runs(limit=1)
    if runs.empty:
        return {"run_id": "-", "source_forwards": "-", "rows": "-", "calibrated": "-"}
    row = runs.iloc[0]
    return {
        "run_id": str(row.get("run_id", "-")),
        "source_forwards": str(row.get("source_forwards", "-")),
        "rows": str(int(row.get("row_count", 0))) if pd.notna(row.get("row_count")) else "-",
        "calibrated": "yes" if bool(row.get("calibrated")) else "no",
    }


def show_freshness_sidebar() -> None:
    freshness = data_freshness()
    run = latest_run_summary()
    with st.sidebar:
        st.markdown("##### Derniere mise a jour")
        for name, ts in freshness.items():
            color = COLORS["green"] if ts != "-" else COLORS["red"]
            st.markdown(f'<span style="color:{color};">●</span> **{name}** : {ts}', unsafe_allow_html=True)
        st.caption(f"Run: {run['run_id']} | src forwards: {run['source_forwards']}")
        if st.button("Rafraichir le cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


def format_eur(v: float, decimals: int = 1) -> str:
    return f"{v:,.{decimals}f} EUR/MWh"


def format_pct(v: float, decimals: int = 1) -> str:
    return f"{v:,.{decimals}f}%"


def format_gwh(v: float, decimals: int = 0) -> str:
    return f"{v:,.{decimals}f} GWh"


def add_range_slider(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(
        rangeslider=dict(visible=True, bgcolor=COLORS["card"], thickness=0.04),
        rangeselector=dict(
            buttons=[
                dict(count=7, label="1W", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(label="All", step="all"),
            ],
            bgcolor=COLORS["card"],
            activecolor=COLORS["amber"],
            font=dict(color=COLORS["text"], size=11),
            x=1.0,
            xanchor="right",
            y=1.06,
            yanchor="top",
        ),
    )
    return fig


def no_data_warning(name: str = "donnees") -> None:
    st.warning(
        f"Aucune {name} disponible. Lance `python -m pfc_shaping.pipeline.rolling_update` pour ingerer les donnees.",
        icon="⚠️",
    )


def export_csv_button(df: pd.DataFrame, filename: str, label: str = "Export CSV") -> None:
    csv = df.to_csv()
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def load_model_quality() -> dict[str, float] | None:
    """Load latest eval metrics from eval.log (autoresearch_eval.py output).

    Returns dict of metric_name -> value, or None if no eval available.
    """
    eval_path = PROJECT_ROOT / "eval.log"
    if not eval_path.exists():
        return None
    try:
        lines = eval_path.read_text().strip().splitlines()
        metrics: dict[str, float] = {}
        in_block = False
        for line in lines:
            if line.strip() == "---":
                in_block = True
                metrics = {}  # reset to last block
                continue
            if in_block and ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                try:
                    metrics[key] = float(val)
                except ValueError:
                    metrics[key] = val  # type: ignore[assignment]
        return metrics if metrics else None
    except Exception:
        return None


# ── Commodity data (yfinance) ─────────────────────────────────────────────

COMMODITY_TICKERS = {
    "TTF Gas": {"ticker": "TTF=F", "unit": "EUR/MWh", "color": "#E97451"},
    "Brent": {"ticker": "BZ=F", "unit": "USD/bbl", "color": "#1F2937"},
    "CO2 EUA (KRBN)": {"ticker": "KRBN", "unit": "USD", "color": "#16A34A"},
}


@st.cache_data(ttl=3600, show_spinner="Chargement commodités...")
def load_commodities(period: str = "2y") -> dict[str, pd.DataFrame]:
    """Load commodity prices — cached parquet first, yfinance fallback."""
    # Try local cache first (works on Streamlit Cloud where yfinance may fail)
    cache_path = PROJECT_ROOT / "data" / "commodities_cache.parquet"
    if cache_path.exists():
        try:
            combined = pd.read_parquet(cache_path)
            results: dict[str, pd.DataFrame] = {}
            for name in COMMODITY_TICKERS:
                col = f"{name}|close"
                if col in combined.columns:
                    s = combined[[col]].dropna().rename(columns={col: "close"})
                    s["close"] = s["close"].astype(float)
                    if not s.empty:
                        results[name] = s
            if results:
                return results
        except Exception as exc:
            logger.warning("Failed to read commodity cache: %s", exc)

    # Fallback: live download via yfinance
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — commodity data unavailable")
        return {}
    import warnings

    results = {}
    for name, cfg in COMMODITY_TICKERS.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = yf.download(cfg["ticker"], period=period, progress=False)
            if data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            results[name] = data[["Close"]].rename(columns={"Close": "close"}).copy()
            results[name]["close"] = results[name]["close"].astype(float)
        except Exception as exc:
            logger.warning("Failed to download %s: %s", name, exc)
    return results


def compute_bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Compute SMA and Bollinger Bands."""
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return pd.DataFrame({
        "sma": sma,
        "upper": sma + num_std * std,
        "lower": sma - num_std * std,
    }, index=series.index)


@st.cache_data(ttl=3600, show_spinner="Chargement flows ENTSO-E...")
def load_cross_border_flows() -> pd.DataFrame | None:
    """Load per-border cross-border flows if available in ENTSO-E data."""
    entso = load_entso()
    if entso is None or "cross_border_mw" not in entso.columns:
        return None
    return entso[["cross_border_mw"]]
