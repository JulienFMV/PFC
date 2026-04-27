"""Pricing-Logik: Lastprofil gegen PFC bepreisen."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Both import styles must work : Streamlit pages run this file with
# ``demo_deviwa`` on ``sys.path`` (top-level ``utils`` resolvable),
# whereas a package-style ``import demo_deviwa.pricing`` from a test
# or another script needs the relative form.
try:
    from .utils import _localize_zurich_safe
except ImportError:
    from utils import _localize_zurich_safe  # type: ignore[no-redef]

PEAK_HOURS = set(range(8, 20))  # CH Peak Definition: 08:00 – 20:00 Mo–Fr


@dataclass
class LoadCurve:
    series: pd.Series
    unit: str = "MWh"

    @property
    def hourly(self) -> pd.Series:
        return self.series.resample("h").sum() if self.series.index.freq != "h" else self.series


def detect_load_column(df: pd.DataFrame) -> str | None:
    """Finde die Lastspalte heuristisch."""
    candidates = ["load", "last", "verbrauch", "mwh", "mw", "energy", "kwh", "value", "volumen", "volume"]
    for col in df.columns:
        lc = str(col).lower()
        for key in candidates:
            if key in lc:
                return col
    # Fallback: erste numerische Spalte
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def detect_timestamp_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        lc = str(col).lower()
        if any(key in lc for key in ["time", "date", "zeit", "datum", "stempel", "timestamp"]):
            return col
    return None


def parse_load_file(file_obj) -> tuple[pd.Series, dict]:
    """Parse eine hochgeladene Lastprofil-Datei (CSV oder XLSX).

    Returns:
        (hourly_series_mwh, metadata)
    """
    name = getattr(file_obj, "name", "load.csv").lower()
    raw: pd.DataFrame
    if name.endswith(".xlsx") or name.endswith(".xls"):
        raw = pd.read_excel(file_obj)
    else:
        # Versuche mehrere Separator- und Dezimal-Konventionen
        content = file_obj.read() if hasattr(file_obj, "read") else file_obj
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        for sep, dec in [(";", ","), (",", "."), (";", "."), ("\t", ".")]:
            try:
                raw = pd.read_csv(io.StringIO(content), sep=sep, decimal=dec)
                if raw.shape[1] >= 2:
                    break
            except Exception:
                continue
        else:
            raise ValueError("CSV konnte nicht geparst werden.")

    ts_col = detect_timestamp_column(raw)
    load_col = detect_load_column(raw)
    if ts_col is None or load_col is None:
        raise ValueError(
            "Lastprofil benötigt eine Zeit-Spalte (z.B. 'timestamp', 'datum') "
            "und eine Last-Spalte (z.B. 'load', 'MWh')."
        )

    ts = pd.to_datetime(raw[ts_col], errors="coerce", dayfirst=True)
    load = pd.to_numeric(raw[load_col], errors="coerce")
    mask = ts.notna() & load.notna()
    if mask.sum() < 24:
        raise ValueError(f"Zu wenige gültige Datenpunkte: {int(mask.sum())}.")

    s = pd.Series(load[mask].values, index=ts[mask], name="load_mwh").sort_index()
    # Tz-Handhabung — DST-safe (autumn DST hour preserved)
    s.index = _localize_zurich_safe(pd.DatetimeIndex(s.index))

    # Einheit raten
    median_val = float(np.nanmedian(s.values))
    unit = "MWh"
    if median_val > 1000:  # sieht nach kWh aus
        s = s / 1000.0
        unit = "kWh -> MWh (umgerechnet)"

    # Auf Stundenniveau aggregieren (sum falls feiner, mean falls bereits h)
    inferred_freq = pd.infer_freq(s.index[:50]) if len(s) > 50 else None
    if inferred_freq and "H" not in str(inferred_freq).upper():
        hourly = s.resample("h").sum()
    elif inferred_freq and "H" in str(inferred_freq).upper():
        hourly = s.copy()
    else:
        # Unregelmässiger Index – konservativ: Stundensumme
        hourly = s.resample("h").sum()

    meta = {
        "rows": int(mask.sum()),
        "start": s.index.min(),
        "end": s.index.max(),
        "timestamp_col": ts_col,
        "load_col": load_col,
        "unit": unit,
        "inferred_freq": str(inferred_freq) if inferred_freq else "unbekannt",
        "annual_mwh": float(hourly.sum()),
    }
    return hourly, meta


def price_load_against_pfc(
    load_hourly: pd.Series,
    pfc: pd.DataFrame,
    pfc_col: str = "price_shape",
) -> dict:
    """Bepreise Lastprofil gegen PFC.

    Returns:
        dict mit Kennzahlen + monatlichem Breakdown.
    """
    if pfc.empty or pfc_col not in pfc.columns:
        raise ValueError("PFC nicht verfügbar.")

    # PFC auf Stundenmittel
    pfc_h = pfc[pfc_col].resample("h").mean()
    pfc_h.index = pfc_h.index.tz_convert("Europe/Zurich")

    # Schnittmenge der Zeiträume
    common = load_hourly.index.intersection(pfc_h.index)
    if len(common) < 24:
        # Zirkuläre Projektion: wenn Lastprofil ein typisches Jahr ist,
        # aber in der Vergangenheit liegt, skalieren wir das Wochenprofil
        # auf den ersten PFC-Zeitraum.
        return _price_with_projection(load_hourly, pfc_h)

    load_c = load_hourly.reindex(common).fillna(0.0)
    pfc_c = pfc_h.reindex(common)

    total_mwh = float(load_c.sum())
    cost_total = float((load_c * pfc_c).sum())
    profile_price = cost_total / total_mwh if total_mwh > 0 else float("nan")
    baseload_price = float(pfc_c.mean())

    # Peak/Offpeak Split
    is_peak = (load_c.index.hour.isin(PEAK_HOURS)) & (load_c.index.dayofweek < 5)
    peak_mwh = float(load_c[is_peak].sum())
    offpeak_mwh = float(load_c[~is_peak].sum())
    peak_price = float((load_c[is_peak] * pfc_c[is_peak]).sum() / max(peak_mwh, 1e-9))
    offpeak_price = float((load_c[~is_peak] * pfc_c[~is_peak]).sum() / max(offpeak_mwh, 1e-9))

    # Monatlicher Breakdown
    monthly = pd.DataFrame({
        "load_mwh": load_c,
        "pfc_eur_mwh": pfc_c,
        "cost_eur": load_c * pfc_c,
    })
    monthly["month"] = monthly.index.to_period("M").to_timestamp()
    monthly_agg = monthly.groupby("month").agg(
        load_mwh=("load_mwh", "sum"),
        cost_eur=("cost_eur", "sum"),
        avg_pfc=("pfc_eur_mwh", "mean"),
    )
    monthly_agg["profile_price"] = (
        monthly_agg["cost_eur"] / monthly_agg["load_mwh"].replace(0, np.nan)
    )

    return {
        "total_mwh": total_mwh,
        "cost_total_eur": cost_total,
        "profile_price_eur_mwh": profile_price,
        "baseload_price_eur_mwh": baseload_price,
        "profile_vs_baseload_eur_mwh": profile_price - baseload_price,
        "peak_mwh": peak_mwh,
        "offpeak_mwh": offpeak_mwh,
        "peak_share_pct": 100.0 * peak_mwh / max(total_mwh, 1e-9),
        "peak_price_eur_mwh": peak_price,
        "offpeak_price_eur_mwh": offpeak_price,
        "monthly": monthly_agg.reset_index(),
        "period_start": common.min(),
        "period_end": common.max(),
        "coverage_hours": len(common),
        "projection_used": False,
    }


def _price_with_projection(load_hourly: pd.Series, pfc_h: pd.Series) -> dict:
    """Fallback: Lastprofil ist historisch, PFC ist Zukunft.
    Wir projizieren das Wochenprofil auf den PFC-Zeitraum.
    """
    # Typisches Wochenprofil (Stunde × Wochentag)
    df = pd.DataFrame({"load": load_hourly})
    df["dow"] = df.index.dayofweek
    df["hour"] = df.index.hour
    profile = df.groupby(["dow", "hour"])["load"].mean()

    total_hist_mwh = float(load_hourly.sum())
    n_hist_days = max(1, (load_hourly.index.max() - load_hourly.index.min()).days + 1)
    annual_factor = 365.0 / n_hist_days

    # Auf PFC-Zeitraum projizieren (erste 12 Monate)
    pfc_start = pfc_h.index.min()
    target_index = pd.date_range(
        start=pfc_start, periods=8760, freq="h", tz="Europe/Zurich"
    )
    proj = pd.Series(0.0, index=target_index)
    for ts in target_index:
        key = (ts.dayofweek, ts.hour)
        if key in profile.index:
            proj.loc[ts] = profile.loc[key]

    # Auf geschätztes Jahresvolumen skalieren
    if proj.sum() > 0:
        proj = proj * (total_hist_mwh * annual_factor / proj.sum())

    common = proj.index.intersection(pfc_h.index)
    load_c = proj.reindex(common)
    pfc_c = pfc_h.reindex(common)

    result = price_load_against_pfc(load_c, pd.DataFrame({"price_shape": pfc_c}))
    result["projection_used"] = True
    return result
