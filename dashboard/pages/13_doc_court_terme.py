"""
Documentation: Prevision Court Terme (LEAR / ALPINE)
=====================================================
Page de documentation pour la direction.
"""

import streamlit as st

st.title("Prevision Court Terme - Documentation")
st.caption("Architecture ALPINE (Adaptive Learning Pipeline for INtelligent Energy pricing)")

# ── Resume executif ──
st.header("Resume executif")
st.markdown("""
Le systeme de prevision court terme de FMV predit les **prix spot EPEX CH heure par heure**
pour les 10 prochains jours (D+1 a D+10). Il est utilise pour :
- L'optimisation du dispatch des centrales hydro
- Les decisions de trading day-ahead
- L'estimation de la valeur de l'eau (water value)

**Performance actuelle** : MAE ~ 10-11 EUR/MWh sur 30 jours de backtest
(marche tres volatile, guerre Iran mars 2026).
""")

# ── Architecture ──
st.header("Architecture du modele")

st.markdown("""
```
Couche 4:  QRA Meta-Learner (combinaison optimale + intervalles calibres)
              |           |            |
Couche 3:  LEAR      Chronos-2     Regime Router
           causal    + covariates   (spike/normal/volatile)
              |           |            |
Couche 2:  Features enrichis (Swiss Market Intelligence)
              |
Couche 1:  Donnees (EPEX CH/DE, ENTSO-E, hydro, commodites)
```
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Modeles utilises")
    st.markdown("""
    | Modele | Role | Type |
    |--------|------|------|
    | **LEAR** (ElasticNet) | Modele principal | 24 regressions lineaires (1/heure) |
    | **Chronos-2** (Amazon) | Signal complementaire IA | Foundation model pre-entraine |
    | **GBM** (Gradient Boosting) | Heures de pointe | Capture les non-linearites |
    | **MLP** (Neural Network) | Ensemble member | Patterns non-lineaires |
    | **QRA** | Intervalles P10/P90 | Regression quantile |
    """)

with col2:
    st.subheader("Donnees d'entree")
    st.markdown("""
    | Source | Donnees | Frequence |
    |--------|---------|-----------|
    | EPEX Spot | Prix CH + DE | 15 min |
    | ENTSO-E | Charge, solaire, eolien CH | 15 min |
    | ENTSO-E | **Previsions eolien/solaire DE** | 15 min |
    | BFE/SFOE | Remplissage hydro CH | Hebdomadaire |
    | Yahoo/ICE | Gaz TTF, CO2 EUA, Brent | Journalier |
    | Calendrier | Feries CH (VS) + DE | Journalier |
    """)

# ── Innovations ALPINE ──
st.header("Innovations cles (ALPINE)")

st.markdown("""
### 1. Chronos-2 avec covariates europeennes
- Premier foundation model (IA) utilise avec les prix DE, charge et renouvelables
  comme variables d'entree natives
- Modele pre-entraine par Amazon sur des milliards de series temporelles
- Poids adaptatif (10-25%) selon la performance recente vs LEAR

### 2. Normalisation causale (inspiree de Toto/Datadog)
- Au lieu de normaliser les prix avec une moyenne fixe, chaque observation est
  normalisee par les statistiques de tout ce qui precede (fenetre glissante)
- S'adapte automatiquement aux changements de regime de prix

### 3. Regime Router
- Detection automatique du regime de marche : **spike** / **volatile** / **normal**
- Ajuste les parametres du modele en temps reel :
  - En mode spike : correction AR plus forte, poids FM plus eleve
  - En mode normal : parametres standards

### 4. GBM pour les heures de pointe (h07-h19)
- Gradient Boosting Machine capture les non-linearites des transitions
  matin/midi/soir que le modele lineaire ne peut pas saisir
- Poids adaptatif base sur la MAE recente GBM vs LEAR

### 5. Previsions renouvelables DE (ENTSO-E)
- Integration des previsions day-ahead eolien + solaire allemandes
- L'eolien en mer du Nord (~18 GW en moyenne) et le solaire DE (~7 GW)
  sont les principaux drivers des prix EPEX via le couplage de marche

### 6. Swiss Market Intelligence
- **Congestion CH-DE** : detection quand les prix suisses se decouplent de l'Allemagne
- **Delta hydro** : taux de remplissage/vidange des reservoirs (fonte des neiges vs hiver)
- **Feries CH+DE** : les feries allemands impactent fortement les volumes EPEX
""")

# ── Performance ──
st.header("Performance mesuree")

st.markdown("""
**Backtest rolling 30 jours (marche volatile, mars 2026) :**

| Metrique | Avant ALPINE | Apres ALPINE | Amelioration |
|----------|-------------|-------------|-------------|
| MAE (erreur moyenne) | 13.4 EUR/MWh | **10.7 EUR/MWh** | **-20%** |
| RMSE | 19.6 | 16.6 | -15% |
| Correlation | 0.78 | 0.82 | +5% |
| Peak MAE (h07-19) | 19.5 | 13.8 | **-29%** |
| Offpeak MAE | 9.3 | 6.9 | -26% |
| Biais | +1.0 | +0.1 | quasi nul |

*Note : en marche calme, la MAE serait ~7-9 EUR/MWh. La volatilite actuelle
(guerre Iran, prix gas eleves) augmente mecaniquement l'erreur de tout modele.*
""")

# ── Benchmark industrie ──
st.header("Benchmark industrie")

st.markdown("""
| Niveau | MAE typique | Qui |
|--------|-----------|-----|
| Elite (marche calme) | 3-5 EUR/MWh | Axpo/Alpiq desks internes (20-50 quants + modeles fondamentaux) |
| Tres bon (marche calme) | 5-8 EUR/MWh | KYOS, Volue SpotEx, bons academiques |
| Bon (marche calme) | 8-12 EUR/MWh | LEAR bien tune, modeles DNN |
| **Marche volatile (crise)** | **10-20 EUR/MWh** | **Tout le monde souffre** |

**Notre position** : MAE ~10.7 en marche de crise = dans le top des modeles statistiques.
Les experts d'Axpo/Alpiq utilisent en plus des modeles **fondamentaux** (simulation du merit order)
que nous n'avons pas encore.
""")

# ── Travail en cours ──
st.header("Travail en cours et a venir")

st.markdown("""
### En cours
- **Fine-tuning Chronos-2 (LoRA)** : adapter le modele IA specifiquement aux prix
  electricite CH+DE avec 3 ans de donnees historiques. Necessite un GPU.
- **Optimisation continue** des parametres (spike uplift, AR coefficients, poids FM)
  via boucle autoresearch autonome

### Prochaines etapes (par priorite)
1. **Fine-tuning Chronos-2** sur donnees CH+DE (attendu : -10 a 20% MAE du FM)
2. **Previsions de charge DE** comme covariates supplementaires
3. **Meta-learner** : XGBoost qui combine optimalement LEAR + FM + GBM
4. **Nuclear FR** : disponibilite nucleaire francaise comme feature
   (la France passe d'exportatrice a importatrice quand le nucleaire baisse < 45 GW)
5. **Modele fondamental simplifie** : merit order approxime avec capacites connues

### Recherche appliquee
- 7 articles de recherche analyses (Timer-XL, Toto, TimesFM-ICF, TiRex, etc.)
- Roadmap complet dans `docs/research/ALPINE_innovation_roadmap.md`
- 4 rapports d'experts generes (SOTA EPF, techniques avancees, foundation models, PDFs)
""")

# ── References ──
with st.expander("References academiques"):
    st.markdown("""
    - **Lago et al. (2021)** — Benchmark LEAR, International Journal of Forecasting
    - **Nowotarski & Weron (2015)** — QRA, European Journal of OR
    - **Paraschiv, Fleten & Schuerle** — Ponderation exponentielle, marche suisse
    - **Uniejewski et al. (2019)** — Features renouvelables, -4 a 8% MAE
    - **Amazon Chronos-2 (2025)** — Foundation model avec covariates
    - **Datadog Toto (2025)** — Normalisation causale, Student-T mixture
    - **Hildmann (2013)** — Criteres de qualite HPFC (ETH Zurich)
    """)
