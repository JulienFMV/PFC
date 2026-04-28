"""FMV Deviwa Demo — Startseite : Portfolio-Cockpit.

Vue d'ensemble du portefeuille Deviwa par année de livraison. Les chiffres
viennent directement des modules :

  - ``portfolio_yearly``    : hedge ratio FMV-share par année + budget projection
  - ``pool_diversification``: VaR / ES individuel × pool + diversification benefit
  - ``utils.load_*_cached`` : parquets pré-extraits (cf. Phase 1)

Discipline visuelle pour audience GRD non-trader :
  - 4 cartes années (Cal-26 / 27 / 28 / 29) avec mini-gauge horizontale
  - Cartes vides en gris doux ("Noch keine Absicherung") — l'absence de hedge
    est un signal commercial, pas une erreur à cacher
  - Visuel principal : load curve + hedge histograms par mois, piloté par un
    sélecteur d'année unique
  - EDSH spécial : stacked area par fournisseur (BKW / EnAlpin / EWZ / FMV / Spot)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pool_diversification import compute_pool_diversification  # noqa: E402
from portfolio_yearly import (  # noqa: E402
    compute_budget_projection,
    compute_hedge_by_year,
)
from utils import (  # noqa: E402
    FMV_BLUE,
    FMV_GREY,
    FMV_NAVY,
    fmt_chf,
    fmt_eur_mwh,
    fmt_mwh,
    freshness_badge,
    load_deviwa_deals_cached,
    load_deviwa_edsh_suppliers_cached,
    load_deviwa_programme_cached,
    load_eex_forwards,
    render_actor_selector,
    render_cache_status,
    render_edsh_supplier_stack,
    render_header,
    render_hedge_ladder,
    render_load_hedge_chart,
    render_weather_strip,
    render_year_card,
)


# ---------------------------------------------------------------------------
# Page configuration & header
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FMV Deviwa · Portfolio-Cockpit",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_header("Portfolio-Cockpit")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        f"<div style='color:{FMV_NAVY}; font-weight:700; font-size:1.05rem;'>"
        "FMV · Deviwa Energiepool</div>",
        unsafe_allow_html=True,
    )
    st.caption("Demo · alle Daten lokal")
    st.divider()
    render_cache_status()
    st.divider()
    st.markdown("**Konvention**")
    extrapolate_programme = st.toggle(
        "Programm-Hochrechnung",
        value=True,
        help=(
            "Wenn ein Akteur für ein Lieferjahr noch kein Programm geliefert "
            "hat, das letzte verfügbare Jahr fortschreiben (EFET-Standard). "
            "Aus : Lücken bleiben leer."
        ),
        key="extrapolate_programme",
    )
    st.divider()
    st.markdown("**Navigation**")
    st.markdown(
        "- 🎯 Portfolio-Cockpit\n- 📊 Marktüberblick\n- 📈 Kurzfristprognose\n"
        "- 💡 Lastprofil-Pricing\n- 📒 Ihre Transaktionen\n- 🎚️ Programmqualität\n"
        "- 🛡️ VaR & Stresstest\n- 🌍 Marktdaten"
    )


# ---------------------------------------------------------------------------
# Load data (parquet cache, < 100 ms total)
# ---------------------------------------------------------------------------

deals = load_deviwa_deals_cached()
programme = load_deviwa_programme_cached()
edsh_suppliers = load_deviwa_edsh_suppliers_cached()
forwards = load_eex_forwards()

if deals.empty and programme.empty:
    st.warning(
        "Keine Deviwa-Daten gefunden. Bitte `data/Deviwa.xlsx` ablegen und "
        "den Cache via Sidebar-Button aktualisieren."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Actor selector — Pool / 4 actors. Build a {actor: deals_df} dict to reuse
# the existing `render_actor_selector` helper that returns (choice, frame).
# ---------------------------------------------------------------------------

# Actor universe = union of deals + programme. An actor with programme but
# no deals is still inspectable (the cockpit will show "Noch keine Absicherung"
# cards and the load curve still renders).
actor_set: set[str] = set()
if not deals.empty and "actor" in deals.columns:
    actor_set.update(deals["actor"].dropna().astype(str).unique())
if not programme.empty and "actor" in programme.columns:
    actor_set.update(programme["actor"].dropna().astype(str).unique())
actors_in_data: list[str] = sorted(actor_set)

data_dict: dict[str, pd.DataFrame] = {
    a: deals[deals["actor"] == a].reset_index(drop=True) if not deals.empty else pd.DataFrame()
    for a in actors_in_data
}

choice, _ = render_actor_selector(data_dict, key="cockpit_actor")
selected_actor: str | None = None if choice == "Gesamter Pool" else choice


# ---------------------------------------------------------------------------
# Personal narrative banner
# ---------------------------------------------------------------------------
# Weather strip moved to the footer so the headline KPI (Σ-VaR / hedge ratio
# / diversification benefit) takes the prime visual slot. Local-terrain
# weather is "nice to have" context, not the first thing a GRD audience
# should see when the dashboard opens.

today = pd.Timestamp.now(tz="Europe/Zurich")
display_years = list(range(today.year, today.year + 4))

# Compute hedge ratios for the selected actor (or pool when actor=None).
# include_years forces 4 cards even for actors with sparse data, and
# extrapolate_missing_programme drives the carry-forward convention.
hedge_metrics = compute_hedge_by_year(
    deals, programme, actor=selected_actor, today=today,
    extrapolate_missing_programme=extrapolate_programme,
    include_years=display_years,
)

# Headline year = next calendar year (Y-1 industry corridor target)
headline_year = today.year + 1
headline_metrics = hedge_metrics.get(headline_year)

actor_label = selected_actor if selected_actor else "Pool Deviwa"

# Headline number : when the pool is selected, the diversification benefit
# is THE selling number — the entire commercial argument of the platform
# in a single percentage. Keeping it in a dense banner buries the lede :
# now we lift it into a hero card on the right of the greeting block, so
# the eye lands on it in the first second of the demo (Bloomberg / Volue
# convention). When an actor is selected the slot is reused for that
# actor's hedge ratio, which is the equivalent personal headline number.
pool_div = None
if selected_actor is None:
    pool_div = compute_pool_diversification(
        headline_year, deals, programme, forwards, today=today,
        extrapolate_missing_programme=extrapolate_programme,
    )

hero_html = ""
if (
    selected_actor is None
    and pool_div is not None
    and np.isfinite(pool_div.diversification_benefit_eur)
    and pool_div.diversification_benefit_eur > 0
):
    hero_html = (
        f"<div style='background:linear-gradient(135deg,{FMV_BLUE},{FMV_NAVY}); "
        f"color:white; border-radius:12px; padding:0.9rem 1.2rem; "
        f"min-width:240px; text-align:right;'>"
        f"<div style='font-size:0.72rem; letter-spacing:0.10em; "
        f"text-transform:uppercase; opacity:0.85;'>"
        f"Pool-Diversifikations-Vorteil {headline_year}</div>"
        f"<div style='font-size:2.0rem; font-weight:700; line-height:1.1;'>"
        f"{pool_div.diversification_pct:.0f} %</div>"
        f"<div style='font-size:0.85rem; opacity:0.92;'>"
        f"≈ {fmt_chf(pool_div.diversification_benefit_eur, 0)} EUR weniger Risiko "
        f"als allein</div></div>"
    )
elif headline_metrics is not None and np.isfinite(headline_metrics.hedge_ratio_pct):
    ratio = headline_metrics.hedge_ratio_pct
    in_corridor = headline_metrics.target_status == "in_corridor"
    bg = (
        f"linear-gradient(135deg,{FMV_GREEN},#0F6B3B)" if in_corridor
        else f"linear-gradient(135deg,{FMV_ACCENT},#C28A00)"
    )
    hero_html = (
        f"<div style='background:{bg}; color:white; border-radius:12px; "
        f"padding:0.9rem 1.2rem; min-width:240px; text-align:right;'>"
        f"<div style='font-size:0.72rem; letter-spacing:0.10em; "
        f"text-transform:uppercase; opacity:0.85;'>"
        f"Hedge-Quote Cal-{headline_year}</div>"
        f"<div style='font-size:2.0rem; font-weight:700; line-height:1.1;'>"
        f"{ratio:.0f} %</div>"
        f"<div style='font-size:0.85rem; opacity:0.92;'>"
        f"Korridor {headline_metrics.target_low_pct}–"
        f"{headline_metrics.target_high_pct} %</div></div>"
    )

greeting_html = (
    f"<div style='flex:1;'>"
    f"<div style='font-weight:700; color:{FMV_NAVY}; font-size:1.15rem;'>"
    f"Guten Tag {actor_label}</div>"
    f"<div style='color:{FMV_GREY}; font-size:0.85rem;'>"
    f"Stand {today.strftime('%d.%m.%Y %H:%M')} · Sicht: Kunde</div>"
    f"</div>"
)

st.markdown(
    f"""
    <div style='display:flex; align-items:stretch; gap:1.1rem;
                background:#F7FAFF; border:1px solid #D8E5FF;
                border-left:6px solid {FMV_BLUE}; border-radius:12px;
                padding:1.0rem 1.25rem; margin:0.6rem 0 1.4rem 0;'>
        {greeting_html}
        {hero_html}
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 4 year cards — Cal-26 / 27 / 28 / 29
# ---------------------------------------------------------------------------

st.subheader("Absicherung pro Lieferjahr")

# We always render 4 cards (today.year .. today.year+3). Years without data
# show as empty grey cards ("Noch keine Absicherung") — the absence of a
# hedge is itself a commercial signal.
budget_metrics = compute_budget_projection(
    deals, programme, forwards, actor=selected_actor, today=today,
    extrapolate_missing_programme=extrapolate_programme,
    include_years=display_years,
)

cols = st.columns(4)
for i, y in enumerate(display_years):
    yhm = hedge_metrics.get(y)
    bp = budget_metrics.get(y)
    with cols[i]:
        st.markdown(render_year_card(y, yhm, bp, today=today), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Year detail : load curve + hedge histograms
# ---------------------------------------------------------------------------

st.markdown("")
st.subheader("Detail pro Lieferjahr")

# Year selector (pills) — default to the first year >= today.year+1 that has
# programme data (= the operationally most relevant year for hedging
# decisions). Falls back to the current year, then to the first listed year.
default_year_idx = 0
preferred_order = [y for y in display_years if y >= today.year + 1] + [today.year]
for y_pref in preferred_order:
    yhm = hedge_metrics.get(y_pref)
    if yhm is not None and yhm.programme_mwh > 0 and y_pref in display_years:
        default_year_idx = display_years.index(y_pref)
        break

selected_year = st.radio(
    "Lieferjahr",
    options=display_years,
    horizontal=True,
    index=default_year_idx,
    format_func=lambda y: f"Cal-{y}",
    key="detail_year",
)

# The chart panel
fig = render_load_hedge_chart(programme, deals, selected_year, selected_actor)
st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# EDSH-only : supplier stack visualisation
# ---------------------------------------------------------------------------

if selected_actor == "EDSH" and not edsh_suppliers.empty:
    st.markdown("")
    st.subheader("Lieferanten-Mix (EDSH)")
    st.caption(
        "Decomposition der EDSH-Lastdeckung nach Lieferant. Diese Sicht ist "
        "exklusiv für EDSH verfügbar — die anderen Akteure liefern uns nur "
        "ihre FMV-Transaktionen, nicht den Gesamt-Mix."
    )
    fig_edsh = render_edsh_supplier_stack(edsh_suppliers, selected_year)
    st.plotly_chart(fig_edsh, use_container_width=True)


# ---------------------------------------------------------------------------
# Hedge ladder (collapsible — power-user view)
# ---------------------------------------------------------------------------

with st.expander(f"📈 Hedge-Aufbau Cal-{selected_year} (Trade Dates × Forward-Preis)"):
    st.caption(
        "Wann wurden die Deals gemacht und zu welchem Preis ? Die kumulierte "
        "Hedge-Quote sollte sich gleichmässig aufbauen, damit man nicht nur "
        "hohe oder nur tiefe Preise erwischt."
    )
    fig_ladder = render_hedge_ladder(deals, forwards, selected_year, selected_actor)
    st.plotly_chart(fig_ladder, use_container_width=True)


# ---------------------------------------------------------------------------
# Footer freshness badges
# ---------------------------------------------------------------------------

st.divider()
cols = st.columns(4)
cols[0].markdown(
    freshness_badge(programme.set_index("timestamp") if not programme.empty else programme, "Programme"),
    unsafe_allow_html=True,
)
cols[1].markdown(
    freshness_badge(forwards.set_index("date") if not forwards.empty else forwards, "EEX Forwards"),
    unsafe_allow_html=True,
)
cols[2].markdown(
    freshness_badge(deals.assign(_ts=pd.to_datetime(deals.get("trade_date"))).set_index("_ts") if not deals.empty else deals, "Deals (letzter trade)"),
    unsafe_allow_html=True,
)
cols[3].caption(f"Sicht: **Kunde** · Stand {today.strftime('%d.%m.%Y %H:%M')}")

# Haut-Valais weather context — local terrain colour for the audience.
# Lives in the footer so it doesn't compete with the headline KPI for
# attention at the top of the page. Silently skipped if Open-Meteo is
# unreachable.
render_weather_strip()

st.caption(
    "Hinweis : Die Hedge-Quote zeigt den FMV-Anteil der Absicherung. "
    "Andere Lieferantenverträge (BKW, EnAlpin, EWZ …) sind in dieser Datenquelle "
    "nicht enthalten — ausser für EDSH, wo die volle Lieferanten-Decomposition verfügbar ist."
)
