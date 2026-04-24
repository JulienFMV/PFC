"""Portfolio-Kennzahlen: Hedge-Quote, MtM, P&L, Ø Hedge-Preis, Tenor-Buckets.

Gemeinsame Berechnungen für Cockpit und VaR-Seite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


PEAK_HOURS = set(range(8, 20))


def _as_utc_naive(ts: pd.Timestamp | None) -> pd.Timestamp | None:
    if ts is None or pd.isna(ts):
        return None
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


@dataclass
class PortfolioMetrics:
    """Zentrale Kennzahlen eines Akteurs / Pools."""
    total_volume_mwh: float = 0.0
    net_volume_mwh: float = 0.0
    intake_mwh: float = 0.0
    withdrawal_mwh: float = 0.0
    n_deals: int = 0

    # Notional (Summe Volumen × Deal-Preis aller Deals)
    notional_total_eur: float = 0.0
    # Gewichteter Ø Deal-Preis (alle)
    avg_deal_price_eur_mwh: float = float("nan")
    # Gewichteter Ø Intake-Preis (Einkauf)
    avg_intake_price_eur_mwh: float = float("nan")
    # Gewichteter Ø Withdrawal-Preis (Verkauf)
    avg_withdrawal_price_eur_mwh: float = float("nan")

    # Mark-to-Market gegen aktuelle PFC
    mtm_eur: float = float("nan")
    # Unrealisierter PnL (nur noch offene Positionen)
    unrealized_pnl_eur: float = float("nan")
    # Realisierter PnL (Lieferung abgeschlossen)
    realized_pnl_eur: float = float("nan")

    # Hedge-Quote: Anteil Volumen mit Lieferung in der Zukunft
    hedge_ratio_pct: float = float("nan")
    hedged_volume_mwh: float = 0.0
    open_volume_mwh: float = 0.0

    # Ø Hedge-Preis (nur Zukunfts-Volumen)
    avg_hedge_price_eur_mwh: float = float("nan")

    # Tenor-Buckets (nächste Lieferperioden)
    tenor_buckets: pd.DataFrame = field(default_factory=pd.DataFrame)


def _attach_deal_price(df: pd.DataFrame) -> pd.DataFrame:
    """Leitet deal_price_eur_mwh aus notional_sum / volume_sum ab."""
    df = df.copy()
    vol = pd.to_numeric(df.get("volume_sum"), errors="coerce")
    notional = pd.to_numeric(df.get("notional_sum"), errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        price = notional / vol.replace(0, np.nan)
    df["deal_price_eur_mwh"] = price.abs()
    return df


def _pfc_monthly_series(pfc: pd.DataFrame) -> pd.Series:
    """Mittlerer PFC-Preis je Monat (Index = erster Monatstag, tz-naive)."""
    if pfc is None or pfc.empty or "price_shape" not in pfc.columns:
        return pd.Series(dtype=float)
    s = pfc["price_shape"].resample("h").mean()
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    s.index = s.index.tz_convert("Europe/Zurich")
    monthly = s.resample("MS").mean()
    monthly.index = monthly.index.tz_localize(None)
    return monthly


def compute_portfolio_metrics(
    df: pd.DataFrame,
    pfc: pd.DataFrame | None = None,
    today: pd.Timestamp | None = None,
) -> PortfolioMetrics:
    """Berechnet alle Kennzahlen für einen Akteur / Pool."""
    metrics = PortfolioMetrics()
    if df is None or df.empty:
        return metrics

    df = _attach_deal_price(df)
    today = _as_utc_naive(today) or pd.Timestamp.utcnow().tz_localize(None)

    vol = pd.to_numeric(df.get("volume_sum"), errors="coerce").fillna(0.0)
    notional = pd.to_numeric(df.get("notional_sum"), errors="coerce").fillna(0.0)

    scope = df.get("scope", "").astype(str).str.lower()
    intake_mask = scope.str.contains("intake")
    withdrawal_mask = scope.str.contains("withdrawal")

    metrics.total_volume_mwh = float(vol.abs().sum())
    metrics.intake_mwh = float(vol[intake_mask].abs().sum())
    metrics.withdrawal_mwh = float(vol[withdrawal_mask].abs().sum())
    metrics.net_volume_mwh = metrics.intake_mwh - metrics.withdrawal_mwh
    metrics.n_deals = int(df["deal"].nunique()) if "deal" in df.columns else int(len(df))

    # Gewichtete Ø-Preise
    def _wavg(values: pd.Series, weights: pd.Series) -> float:
        w_sum = float(weights.abs().sum())
        if w_sum <= 0:
            return float("nan")
        return float((values * weights.abs()).sum() / w_sum)

    metrics.notional_total_eur = float(notional.abs().sum())
    metrics.avg_deal_price_eur_mwh = _wavg(df["deal_price_eur_mwh"].fillna(0), vol.abs())
    metrics.avg_intake_price_eur_mwh = _wavg(
        df.loc[intake_mask, "deal_price_eur_mwh"].fillna(0),
        vol[intake_mask].abs(),
    ) if intake_mask.any() else float("nan")
    metrics.avg_withdrawal_price_eur_mwh = _wavg(
        df.loc[withdrawal_mask, "deal_price_eur_mwh"].fillna(0),
        vol[withdrawal_mask].abs(),
    ) if withdrawal_mask.any() else float("nan")

    # Hedge-Quote = Anteil Volumen mit Lieferung in Zukunft
    delivery_from = df.get("delivery_from")
    delivery_to = df.get("delivery_to")
    if delivery_to is not None:
        delivery_to = pd.to_datetime(delivery_to, errors="coerce")
        open_mask = delivery_to > today
    else:
        open_mask = pd.Series(False, index=df.index)
    metrics.hedged_volume_mwh = float(vol[open_mask].abs().sum())
    metrics.open_volume_mwh = metrics.hedged_volume_mwh
    if metrics.total_volume_mwh > 0:
        metrics.hedge_ratio_pct = 100.0 * metrics.hedged_volume_mwh / metrics.total_volume_mwh

    # Ø Hedge-Preis (nur zukünftige Lieferungen)
    if open_mask.any():
        metrics.avg_hedge_price_eur_mwh = _wavg(
            df.loc[open_mask, "deal_price_eur_mwh"].fillna(0),
            vol[open_mask].abs(),
        )

    # Mark-to-Market gegen PFC
    pfc_monthly = _pfc_monthly_series(pfc) if pfc is not None else pd.Series(dtype=float)
    pfc_avg = float(pfc_monthly.mean()) if not pfc_monthly.empty else float("nan")

    if np.isfinite(pfc_avg) and "month" in df.columns:
        months = pd.to_datetime(df["month"], errors="coerce")
        pfc_lookup = pfc_monthly.reindex(months.dt.to_period("M").dt.to_timestamp()).values
        pfc_lookup = pd.Series(pfc_lookup, index=df.index).fillna(pfc_avg)

        # Für Intake (Kauf): MtM = Volumen × aktueller Preis - Notional
        # Für Withdrawal (Verkauf): MtM = Notional - Volumen × aktueller Preis
        signed_vol = vol.abs().where(intake_mask, -vol.abs())
        market_value = signed_vol * pfc_lookup
        notional_signed = notional.where(intake_mask, -notional.abs())
        unrealized = market_value - notional_signed

        metrics.mtm_eur = float(market_value.sum())
        # Realisiert vs unrealisiert trennen
        realized_mask = delivery_to <= today if delivery_to is not None else pd.Series(False, index=df.index)
        metrics.realized_pnl_eur = float(
            pd.to_numeric(df.loc[realized_mask, "pnl_sum"], errors="coerce").fillna(0).sum()
        ) if "pnl_sum" in df.columns else 0.0
        metrics.unrealized_pnl_eur = float(unrealized[open_mask].sum())

    # Tenor-Buckets
    if delivery_from is not None:
        d_from = pd.to_datetime(delivery_from, errors="coerce")
        days_to_delivery = (d_from - today).dt.days
        bins = [-np.inf, 0, 30, 90, 365, 730, np.inf]
        labels = ["Vergangen", "M+1", "M+2 bis Q+1", "Q+2 bis Y+1", "Y+2", "Y+3+"]
        bucket = pd.cut(days_to_delivery, bins=bins, labels=labels, right=True)
        tb = pd.DataFrame({
            "bucket": bucket,
            "volume": vol.abs(),
            "notional": notional.abs(),
            "price": df["deal_price_eur_mwh"],
        })
        agg = tb.groupby("bucket", observed=True).agg(
            volume_mwh=("volume", "sum"),
            notional_eur=("notional", "sum"),
            avg_price_eur_mwh=("price", lambda s: _wavg(
                s.fillna(0), tb.loc[s.index, "volume"]
            )),
        ).reset_index()
        agg = agg[agg["volume_mwh"] > 0]
        metrics.tenor_buckets = agg

    return metrics


def monthly_position(df: pd.DataFrame) -> pd.DataFrame:
    """Monatliche Sicht: Intake, Withdrawal, Netto, Ø Preis je Richtung."""
    if df is None or df.empty or "month" not in df.columns:
        return pd.DataFrame()
    df = _attach_deal_price(df)
    months = pd.to_datetime(df["month"], errors="coerce")
    vol = pd.to_numeric(df.get("volume_sum"), errors="coerce").fillna(0.0)
    scope = df.get("scope", "").astype(str).str.lower()
    intake_mask = scope.str.contains("intake")
    withdrawal_mask = scope.str.contains("withdrawal")

    rows = []
    for month, idx in pd.Series(df.index, index=months).groupby(months).groups.items():
        if pd.isna(month):
            continue
        sub_idx = list(idx)
        in_idx = [i for i in sub_idx if intake_mask.loc[i]]
        out_idx = [i for i in sub_idx if withdrawal_mask.loc[i]]
        in_vol = float(vol.loc[in_idx].abs().sum())
        out_vol = float(vol.loc[out_idx].abs().sum())
        in_price = float(
            (df.loc[in_idx, "deal_price_eur_mwh"].fillna(0) * vol.loc[in_idx].abs()).sum()
            / max(in_vol, 1e-9)
        ) if in_vol > 0 else float("nan")
        out_price = float(
            (df.loc[out_idx, "deal_price_eur_mwh"].fillna(0) * vol.loc[out_idx].abs()).sum()
            / max(out_vol, 1e-9)
        ) if out_vol > 0 else float("nan")
        rows.append({
            "month": month,
            "intake_mwh": in_vol,
            "withdrawal_mwh": out_vol,
            "net_mwh": in_vol - out_vol,
            "avg_intake_price": in_price,
            "avg_withdrawal_price": out_price,
        })
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def hedge_ratio_by_tenor(df: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Hedge-Quote je Tenor-Bucket (Volumen mit abgeschlossenem Deal / Gesamt-Volumen im Bucket)."""
    if df is None or df.empty or "delivery_from" not in df.columns:
        return pd.DataFrame()
    today = _as_utc_naive(today) or pd.Timestamp.utcnow().tz_localize(None)
    d_from = pd.to_datetime(df["delivery_from"], errors="coerce")
    days = (d_from - today).dt.days
    bins = [-np.inf, 0, 90, 365, 730, np.inf]
    labels = ["bereits in Lieferung", "Q+1", "Y+1", "Y+2", "Y+3+"]
    bucket = pd.cut(days, bins=bins, labels=labels, right=True)
    vol = pd.to_numeric(df.get("volume_sum"), errors="coerce").fillna(0.0).abs()
    return pd.DataFrame({
        "bucket": bucket,
        "volume_mwh": vol,
    }).groupby("bucket", observed=True)["volume_mwh"].sum().reset_index()
