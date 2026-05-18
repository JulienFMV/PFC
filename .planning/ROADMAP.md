# Roadmap: PFC Long-Term

## Overview

Du legacy "utility-grade" actuel (shape historique, plancher à zéro, peak/offpeak
écrasés) vers une PFC "trading-grade" capable de battre OMPEX sur les profile
deals industriels. Séquence pilotée par le business case profile-deal (≥250k€
P&L par 5 €/MWh d'erreur shape), pas par l'ordre alphabétique des features.

## Phases

**Phase Numbering:**
- Phases entières (1-10) : milestones planifiés.
- Phases décimales (5bis, 5ter, etc.) : urgences insérées post-audit.
- Phases ✅ : déjà livrées dans des commits référencés.
- Phases ⏸ HOLD : explicitement différées par décision business.

### Done (référence historique)

- [x] **Phase 0: Cadrage sources EEX** — `f0b4a10`. Décision Yearly + Historique2019 + parser legacy `Y01_/Q0N_/M0N_` validé sur snapshot.
- [x] **Refactor B: Split LT/CT model** — `596e3c5`. `pfc_shaping/lt/model/` + `pfc_shaping/ct/model/` + shim de compat avec `DeprecationWarning`.
- [x] **Phase 1ter: Parser EEX étendu** — `8d28b63`. Week products explicitement classifiés et filtrés, onglets FX/Produits/HFC skippés.
- [x] **Phase 1bis: `_build_long_term_branch(spec)`** — `2aa99ea`. Factory générique market-aware, prêt pour FR/AT/IT.
- [x] **Phase 2: Prix négatifs en ingestion + script rebuild** — `867c51e` + `0915f0e`. `_coerce_price` avec sanity `[-500, 10000]`, `0` = non-quoté, vectorisation compat pandas<2.1.
- [x] **Audit profond LT 25 findings** — voir `.planning/research/audit-deep-lt-2026-05.md` (à créer ; pour l'instant le rapport vit en conversation).
- [x] **Bloc A+C1+C2: audit fixes** — `28dfd65`. country/tz dans arbitrage_free + 3 helpers assembler ; water_value `center=False` causal ; backtest propage `reference_date`.

### Active (P0 — prochaine action)

- [ ] **Phase 5bis: Shape seasonal × type_jour × hour** 🔥
  - Goal: `ShapeHourly.factors_[(saison, type_jour, hour)]` → bowl midday d'été différencié du plateau hiver.
  - Plans: voir `.planning/phases/05bis/` à créer via `/gsd:plan-phase 5bis`.

### Active (P0 — successeurs immédiats)

- [ ] **Phase 5: MSFC log-prix + retire silent floors + PFC peut être négative**
  - Goal: la PFC peut descendre à -20 €/MWh aux heures structurelles (été 2027+ midi).
  - Bloque la profondeur du bowl, requis pour pricer correctement le bloc 10-15 d'été.
- [ ] **Phase 5ter: Distribution probabiliste par bloc**
  - Goal: `pfc_block_distribution(start, end, hours_mask) → (p10, p50, p90)` via Monte-Carlo shape.
  - Permet au trader de calculer une prime de risque (shape inhedgeable).
- [ ] **Phase 10 (refondu): Backtest par bloc client vs HFC OMPEX 2024-2025**
  - Goal: démontrer Δ MAE ≤ -1.5 €/MWh sur ≥ 3 blocs.
  - Argument interne décisif vs OMPEX.

### Deferred (HOLD — pas de valeur sur deal CH actuel)

- [ ] ⏸ **Phase 3: Activation FR/AT/IT en production**
- [ ] ⏸ **Phase 4: Basis cross-border long-terme** (CH = voisin − basis(t,h))

### v2 (post-shape SOTA validé)

- [ ] **Phase 5quater: Fundamentals-driven shape** (residual demand forward, 4-6 semaines)
- [ ] **Phase 6: Stochastic water value** (LSMC/SDP, lake vs RoR)
- [ ] **Phase 7: Curves gouvernées TTF/EUA/API2**
- [ ] **Phase 8: Calibration jointe multi-zone**
- [ ] **Phase 9: Dashboard PFC vs Quotes multi-marché**

---

## Phase Details

### Phase 5bis: Shape seasonal × type_jour × hour
**Goal**: Différencier le facteur horaire `f_H` selon la saison ET le type de jour ET l'heure (table 3D), au lieu d'un f_H global moyenné sur l'année.

**Depends on**: Bloc A (country/tz plumbing) — déjà livré.

**Requirements**: SHP-01, SHP-02, SHP-03, SHP-04

**Success Criteria** (what must be TRUE):
  1. `ShapeHourly.factors_` est un dict indexé par `(saison, type_jour, hour)` ; un test unitaire confirme que `factors_[("Ete","Ouvrable",12)]` ≠ `factors_[("Hiver","Ouvrable",12)]` sur des données synthétiques avec bowl injecté.
  2. `assembler.build` produit une PFC où le profil h10-h15 d'été d'un dimanche est sensiblement plus bas que celui d'un dimanche d'hiver (la différence absolue moyenne >5 €/MWh).
  3. La feature flag `PFC_LT_USE_SEASONAL_HOURLY_SHAPE=0` permet de revenir au comportement legacy bit-pour-bit.
  4. La suite tests existante (142 passed, 4 skipped) reste verte.
  5. Backward-compat : un `factors_` 2D legacy chargé depuis parquet (`ShapeHourly.load`) est promu en 3D lazily, sans crash.

**Plans**: TBD (à générer via `/gsd:plan-phase 5bis` — estimation 3-5 plans atomiques).

---

### Phase 5: MSFC log-prix + retire silent floors
**Goal**: Autoriser une PFC négative aux heures structurelles, en retirant les 4 planchers silencieux actuels.

**Depends on**: Phase 5bis.

**Requirements**: NEG-01 → NEG-05.

**Success Criteria**:
  1. Un Cal'27 forward coté -10 €/MWh produit une PFC qui moyenne à -10 €/MWh exactement sur l'année (calibration arbitrage-free respectée même en négatif).
  2. Un mois solaire (juillet) avec forward 30 €/MWh produit une PFC qui pique à -25 €/MWh à h13 dimanche.
  3. Les 4 planchers silencieux (MSFC `np.maximum(B, 1.0)`, `m_factor >= 0.1`, F_WV_FLOOR, peak ratio >= 1) sont soit retirés, soit gated par une option `enforce_positivity` désactivable.
  4. La feature flag `PFC_LT_ALLOW_NEGATIVE_PRICES=0` permet rollback.
  5. La PFC mid-market (flavor) reste positive si tous les inputs sont positifs (régression test).

**Plans**: TBD via `/gsd:plan-phase 5`.

---

### Phase 5ter: Distribution probabiliste par bloc
**Goal**: Sortir une distribution de la moyenne d'un bloc client (e.g. 10h-15h en juillet) via Monte-Carlo sur les shape factors.

**Depends on**: Phase 5bis et Phase 5.

**Requirements**: DIST-01 → DIST-03.

**Success Criteria**:
  1. Méthode `pfc_block_distribution(start, end, hours_mask) → (p10, p50, p90)` retourne des floats reproductibles vs `random_state`.
  2. Sur le bloc 10-15 juillet Cal'27, `(p90 - p50)` > 5 €/MWh (sanity prime de risque).
  3. N=500 trajectoires Monte-Carlo suffisent ; runtime < 5s pour un horizon Cal complet.

**Plans**: TBD via `/gsd:plan-phase 5ter`.

---

### Phase 10 (refondu): Backtest par bloc vs HFC OMPEX
**Goal**: Démontrer la supériorité de notre PFC vs OMPEX sur les profile deals.

**Depends on**: Phase 5bis livrée (peut être lancé en parallèle des Phases 5/5ter).

**Requirements**: BT-01 → BT-05.

**Success Criteria**:
  1. Harness produit un tableau markdown reproductible avec MAE FMV vs MAE OMPEX par bloc.
  2. Δ MAE ≤ -1.5 €/MWh sur au moins 3 blocs des 5 testés (sur 2024-2025).
  3. DM test p-value < 0.05 sur ≥ 2 blocs (significativité statistique).
  4. Le tableau peut être référencé directement dans une note interne FMV.

**Plans**: TBD via `/gsd:plan-phase 10`.

---

## Plans (legacy notation, kept for compat)

Phase 5bis est la prochaine. Les plans détaillés seront générés via
`/gsd:plan-phase 5bis` au démarrage de la phase.
