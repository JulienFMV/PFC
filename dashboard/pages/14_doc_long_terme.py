"""
Documentation: PFC Long Terme — Shaping & Calibration
======================================================
Page de documentation pour la direction.
"""

import streamlit as st

st.title("PFC Long Terme - Documentation")
st.caption("Construction de la Price Forward Curve 15min sur 3 ans")

# ── Resume executif ──
st.header("Resume executif")
st.markdown("""
La **PFC (Price Forward Curve)** est la courbe de prix horaire/15min sur les **3 prochaines
annees**. Elle sert a :
- **Valoriser les actifs hydro** (centrales, pompages, reservoirs)
- **Evaluer les PPA** (Power Purchase Agreements)
- **Gerer le risque** de marche (VaR, stress tests)
- **Alimenter EULER/Phinergy** pour l'optimisation de portefeuille

La PFC transforme les prix forwards EEX (base, peak) en **35'000+ prix 15 minutes**
en decomposant le signal en 6 facteurs multiplicatifs.
""")

# ── Architecture ──
st.header("Architecture du modele")

st.markdown("""
### Formule de decomposition

```
P(t) = B(year) x f_S(month) x f_W(dow) x f_H(hour) x f_Q(quarter) x corrections
```

| Facteur | Description | Source |
|---------|-------------|--------|
| **B(t)** | Niveau base annuel | Forwards EEX (Cal, Q, M) |
| **f_S(month)** | Saisonnalite mensuelle | Historique EPEX |
| **f_W(dow)** | Profil jour de la semaine | Historique EPEX |
| **f_H(hour)** | Shape horaire (24h) | Historique EPEX + MLP |
| **f_Q(quarter)** | Shape intra-horaire (4x15min) | Historique EPEX |
| Corrections | Water value, uncertainty, LEAR blend | Modeles specifiques |
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Modules du modele")
    st.markdown("""
    | Module | Fonction |
    |--------|----------|
    | `shape_hourly.py` | Facteurs f_H par saison/type jour |
    | `shape_intraday.py` | Facteurs f_Q intra-horaires |
    | `msfc_spline.py` | Spline max smoothness pour B(t) |
    | `water_value.py` | Correction valeur de l'eau hydro |
    | `uncertainty.py` | Bandes de confiance (bootstrap) |
    | `assembler.py` | Assemblage final 15min |
    | `lear_forecaster.py` | Blend court terme (D+1 a D+10) |
    """)

with col2:
    st.subheader("Donnees d'entree")
    st.markdown("""
    | Source | Donnees |
    |--------|---------|
    | **EEX** | Forwards Base + Peak (M, Q, Cal) |
    | **EPEX Spot** | Historique 15min CH + DE |
    | **ENTSO-E** | Charge, solar, wind, nuclear |
    | **BFE/SFOE** | Reservoirs hydro (fill %) |
    | **Yahoo/ICE** | Gas TTF, CO2 EUA, Brent |
    | **REMIT** | Indisponibilites centrales |
    | **Calendrier** | Feries CH (VS) + DE |
    """)

# ── Calibration ──
st.header("Calibration et arbitrage-free")

st.markdown("""
### Maximum Smoothness Forward Curve (MSFC)

La methode de reference (Benth, Koekebakker & Ollmar 2007) :
- Le niveau de base B(t) est ajuste pour que la **moyenne de la PFC sur chaque
  periode de livraison = prix forward cote sur EEX**
- Utilise des splines d'ordre 4 qui minimisent la courbure (maximum smoothness)
- Garantit la **continuite** entre periodes (pas de sauts aux frontieres de mois)

### Cascading

Quand il manque des forwards intermediaires (ex: seulement Cal mais pas les Q) :
- Decomposition en cascade : Cal -> Q -> M -> Weekly -> Daily
- Chaque niveau herite les contraintes du niveau superieur
- Les forwards plus granulaires sont "leaders" (monthly > quarterly > calendar)

### Contrainte d'arbitrage-free

**Regle fondamentale** : la moyenne de la PFC horaire sur n'importe quel produit
forward doit etre egale au prix forward cote.
- Verifie automatiquement apres chaque construction
- Ajustements additifs appliques si necessaire
""")

# ── Shape factors ──
st.header("Facteurs de shape")

st.markdown("""
### Shape horaire f_H
- Calcule comme le ratio : prix moyen de l'heure h / prix moyen du jour
- Stratifie par **saison** (hiver/printemps/ete/automne) et **type de jour**
  (ouvrable/samedi/dimanche/ferie CH/ferie DE)
- Base sur l'historique EPEX CH (3 ans glissants)
- Corrige par un MLP (reseau de neurones) pour capturer les non-linearites

### Shape intra-horaire f_Q
- Le coeur du modele FMV : decompose chaque heure en 4 blocs de 15 minutes
- Capture les rampes de prix intra-horaires (ex: debut d'heure plus cher)
- Particulierement important pour le trading 15min et l'optimisation pompage

### Water Value
- Correction saisonniere basee sur le cout d'opportunite de l'eau stockee
- Si les reservoirs sont pleins -> prix spot plus bas (plus de generation hydro)
- Si les reservoirs sont vides -> prix spot plus hauts (moins de flexibilite)
- Integre le niveau de remplissage actuel vs la moyenne saisonniere
""")

# ── Blend court terme ──
st.header("Integration court terme / long terme")

st.markdown("""
La PFC fusionne le **court terme** (LEAR/ALPINE) avec le **long terme** (shape factors) :

| Horizon | Source principale | Methode |
|---------|------------------|---------|
| **D+1 a D+7** | LEAR/ALPINE (forecast) | Le forecast court terme remplace la PFC structurelle |
| **D+8 a D+10** | Blend progressif | Transition lineaire LEAR -> PFC structurelle |
| **D+11 a N+3 ans** | PFC structurelle pure | Shape factors x forwards EEX |

Le blend preserve le **shape intra-journalier** de la PFC tout en utilisant
le **niveau** predit par LEAR pour les premiers jours.
""")

# ── Qualite ──
st.header("Criteres de qualite (Hildmann 2013)")

st.markdown("""
Selon les standards IEEE (Hildmann, Caro et al., ETH Zurich) :

| Critere | Description | Statut |
|---------|-------------|--------|
| **Arbitrage-free** | Moyenne HPFC = prix forward cote | Implemente |
| **Holiday/Weekend** | Jours feries et week-ends differencies | Implemente |
| **Seasonal profile** | Saisonnalite annuelle correcte | Implemente |
| **Continuity** | Profils lisses entre periodes | En amelioration (MSFC) |
| **Peak/Offpeak** | Calibration conjointe base+peak | En cours |
""")

# ── Travail en cours ──
st.header("Travail en cours et a venir")

st.markdown("""
### Ameliorations recentes
- Ajout de la page **PFC vs HFC** dans le dashboard pour comparer avec les
  courbes de reference (KYOS, Volue)
- Integration du **LEAR blend** pour les 10 premiers jours
- Ajout des forwards EEX comme reference de calibration

### Prochaines etapes
1. **Calibration Peak** : intégrer les forwards Peak (pas seulement Base)
   pour calibrer conjointement le niveau peak/offpeak
2. **MSFC complet** : spline max smoothness d'ordre 4 pour le niveau B(t)
   au lieu de l'approche escalier actuelle
3. **Transitions saisonnieres** : blending graduel entre saisons
   (actuellement transition discrete)
4. **Variation intra-mois** : profils jour-a-jour au lieu de semaines identiques
5. **Previsions renouvelables** : ajuster le profil selon les previsions
   solaire/eolien a moyen terme (capture price effect)

### Benchmark vs KYOS/Volue
| Aspect | KYOS (reference) | Notre modele | Gap |
|--------|-----------------|--------------|-----|
| Niveau B(t) | Spline max smoothness | En cours (MSFC) | Comble bientot |
| Forwards utilises | Base + Peak + Quarters | Base + Peak (en cours) | A combler |
| Shaping | Proprietaire, backteste | Shape factors historiques | OK |
| Lissage frontieres | Proprietaire | MSFC spline | En cours |
| Renouvelables | Module capture price | Pas encore | A faire |
""")

# ── References ──
with st.expander("References academiques"):
    st.markdown("""
    - **Benth, Koekebakker & Ollmar (2007)** — Maximum Smoothness Forward Curve (reference)
    - **Fleten & Lemming (2003)** — Construction de courbes forwards electricite
    - **Caldana, Fusai & Roncoroni (2017)** — Thin Granularity Forward Curves (EJOR)
    - **Hildmann, Caro et al. (2013)** — Criteres de qualite HPFC (IEEE/ETH)
    - **Adams & Van Deventer (1994)** — Fondements mathematiques max smoothness
    - **Biegler-Konig & Pilz** — Arbitrage-free shifting de courbes forward
    """)
