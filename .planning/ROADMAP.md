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

- [x] **Phase 5bis-A: Shape Hourly Infrastructure & Flag (no-op refactor)** ✓
  - Goal: livrer l'infra qui permettra de mesurer et reverter bit-pour-bit tout changement comportemental futur du `ShapeHourly` (factors_3d_ view pour SHP-01 littéral, save/load complet, feature flag persisté en parquet, baseline frozen `tests/fixtures/baseline_pfc_seed42.parquet`).
  - **Aucun changement numérique** — `assert_frame_equal(build(flag=OFF), build(flag=ON), atol=1e-12)`.
  - Context: `.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md`.
  - **Plans:** 5 plans
    - [x] 05B-01-PLAN.md — Freeze baseline_pfc_seed42 fixture from main@28dfd65 (separate commit ahead) — DONE `9cc959b`
    - [x] 05B-02-PLAN.md — Complete save/load roundtrip via _meta.parquet sidecar (fix pre-existing bug)
    - [x] 05B-03-PLAN.md — Feature flag PFC_LT_USE_SEASONAL_HOURLY_SHAPE: constructor arg + env-default + freeze + persist
    - [x] 05B-04-PLAN.md — factors_3d_ read-only 3D view (SHP-01 literal) + replace try/except TypeError at assembler.py:284
    - [x] 05B-05-PLAN.md — conftest autouse env hygiene + legacy fixture + 6 tests including parametrized baseline regression

### Active (P0 — successeurs immédiats)

- [x] **Phase 5bis-B: Shape Hourly Bowl-Deepening (math change)** ✓ 2026-05-19
  - Goal: creuser la duck curve via (a) fix bug `_apply_hydro_analogue_weights` (utiliser `_climatological_fill[week_of_year]` au lieu de `current_fill`), (b) split `f_H = level × anomaly` avec `shape_freedom['f_H']` damping sur level only, (c) σ paramétrable (0.5 OFF / 0.25 ON).
  - SC: `|Δ price_shape Été-h10-15 vs Hiver-h10-15| > 5 €/MWh` sur fixture EPEX-like réaliste, gated par `PFC_LT_USE_SEASONAL_HOURLY_SHAPE=1`.
  - Depends on: Phase 5bis-A livrée (baseline frozen + flag persisté).
  - Context: `.planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md`.
  - **Plans:** 3 plans
    - [x] 05C-01-PLAN.md — Lever 1 hydro kernel reformulation + bowl fixture + tests D-A4-3, D-A4-8
    - [x] 05C-02-PLAN.md — Lever 2 split f_H = level + anomaly + assembler integration + tests D-A4-4, D-A4-6
    - [x] 05C-03-PLAN.md — Lever 3 sigma paramétrisation + baseline flag=ON + PROJECT.md D-FLIP-1 + tests D-A4-5, D-A4-7, D-A4-9

- [x] **Phase 5: MSFC retire silent floors + PFC peut être négative** (completed 2026-05-20)
  - Goal: la PFC peut descendre à -20 €/MWh aux heures structurelles (été 2027+ midi).
  - Bloque la profondeur du bowl, requis pour pricer correctement le bloc 10-15 d'été.
  - Context: `.planning/phases/05-msfc-log-prix-retire-silent-floors/05-CONTEXT.md`.
  - **Plans:** 3 plans
    - [x] 05-01-PLAN.md — MSFC enforce_positivity + ArbitrageFreeCalibrator enforce_m_factor_floor + REQUIREMENTS.md NEG-05 reformulation + scaffold tests/test_phase05_negative_prices.py
    - [x] 05-02-PLAN.md — WaterValueCorrection delta-additif (compute_delta_wv) + assembler integration + telemetry + 2 NEG-03 tests
    - [x] 05-03-PLAN.md — ContractCascader spread-additif + master flag PFC_LT_ALLOW_NEGATIVE_PRICES audit-trail + fixture/baseline parquets + 6 tests + PROJECT.md D-FLIP-2
- [ ] **Phase 5ter: Distribution probabiliste par bloc**
  - Goal: `pfc_block_distribution(start, end, hours_mask) → (p10, p50, p90)` via Monte-Carlo shape.
  - Permet au trader de calculer une prime de risque (shape inhedgeable).
- [ ] **Phase 10 (refondu): Backtest par bloc client vs HFC OMPEX 2024-2025**
  - Goal: démontrer Δ MAE ≤ -1.5 €/MWh sur ≥ 3 blocs.
  - Argument interne décisif vs OMPEX.
  - Depends on: Phase 5bis-B (sinon backtest mesure un no-op).

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

### Phase 5bis-A: Shape Hourly Infrastructure & Flag (no-op refactor)
**Goal**: livrer l'infra (view 3D `factors_3d_` pour SHP-01 littéral, save/load complet sur tous attributs entraînés, feature flag `PFC_LT_USE_SEASONAL_HOURLY_SHAPE` persisté en parquet sidecar et gelé à `__init__`, baseline snapshot `tests/fixtures/baseline_pfc_seed42.parquet` committé séparément) qui permettra à 5bis-B et toutes les phases shape ultérieures d'être mesurables et réversibles bit-pour-bit. **Aucun changement numérique** dans cette phase.

**Recadrage** : Phase 5bis initiale du roadmap a été splittée en 5bis-A (infra, ce document) + 5bis-B (bowl-deepening) suite à panel d'experts adversarial (3 reviewers indépendants, verdict unanime "disagree" sur la proposition initiale). Voir `.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md` pour le détail.

**Depends on**: Bloc A (country/tz plumbing) — déjà livré.

**Requirements satisfaits** :
- SHP-01 — `ShapeHourly.factors_3d_[(saison, type_jour, hour)]` accessible (view sur dict[(s,tj)]→array[24] interne).
- SHP-02 — Déjà satisfait par Bloc A (`_country_local_tz` dans assembler). Non-régression assurée par baseline.
- SHP-03 — Invariant `mean_h(f_H | s, tj) ≈ 1.0` ; déjà satisfait par re-normalisation à `shape_hourly.py:150`. Non-régression assurée.
- SHP-04 — Feature flag `PFC_LT_USE_SEASONAL_HOURLY_SHAPE` opérationnel (constructor arg + env-default, gelé à `__init__`, persisté en parquet).

**Success Criteria** :
1. `factors_3d_[("Ete","Ouvrable",12)] == factors_[("Ete","Ouvrable")][12]` pour toutes les cellules ; lecture-seule (test).
2. `numpy.allclose(build(flag=OFF, seed=42), build(flag=ON, seed=42), atol=1e-12)` (5bis-A = no-op).
3. `assert_frame_equal(build(flag=OFF, seed=42), tests/fixtures/baseline_pfc_seed42.parquet, atol=1e-10)` (rollback bit-pour-bit testable).
4. `save → load → fit → save → load` roundtrip identique sur tous attributs (`factors_`, `factors_by_year_`, `trend_per_hour_`, `f_W_seasonal_`, `_climatological_fill`, `sigma`, `halflife_days`, `hydro_weight_sigma`, `_use_seasonal_hourly`).
5. Un parquet legacy fitté avec `main@28dfd65` se recharge sans crash (warning émis), predictions identiques modulo les attributs déjà manquants pre-5bis-A.
6. `tests/conftest.py` autouse fixture évite la fuite env-var test→test sur `PFC_LT_*`.
7. `assembler.py:284` try/except `TypeError` remplacé par capability check explicite (pas de masquage de bug).
8. Suite 142 passed / 4 skipped reste verte.

**Plans**: 5 plans (wave 1→5, sequential because plans 02/03/04 all touch `shape_hourly.py`):
1. `05B-01-PLAN.md` (wave 1) — Freeze `baseline_pfc_seed42.parquet` from main@28dfd65 (separate commit AHEAD of any logic plan).
2. `05B-02-PLAN.md` (wave 2) — Complete save/load roundtrip via `_meta.parquet` sidecar (fixes pre-existing silent attribute loss on `factors_by_year_`, `trend_per_hour_`, `f_W_seasonal_`, `_climatological_fill`, `sigma`, `halflife_days`, `hydro_weight_sigma`).
3. `05B-03-PLAN.md` (wave 3) — Feature flag mechanics: constructor arg + env-default, frozen at `__init__`, persisted in `_meta.parquet`, restored on `load()` (parquet wins over env). [SHP-04]
4. `05B-04-PLAN.md` (wave 4) — Read-only `factors_3d_` view on `ShapeHourly` [SHP-01 literal] + replace try/except `TypeError` at `assembler.py:284` with explicit signature-based capability check.
5. `05B-05-PLAN.md` (wave 5) — `tests/conftest.py` autouse env-var hygiene + legacy fixture parquets + six new tests in `tests/test_shape_hourly_infra.py` including parametrized `test_baseline_regression[False|True]` that asserts `assert_frame_equal(build(flag), baseline, atol=1e-10)` for both flag states (the no-op proof). [SHP-01, SHP-04]

---

### Phase 5bis-B: Shape Hourly Bowl-Deepening (math change)
**Goal**: creuser la duck curve réelle de la PFC pour que les profile deals GRD soient pricés au juste prix (bloc nuit 18-9 + solaire WE OP1/OP2). Trois leviers gated par `PFC_LT_USE_SEASONAL_HOURLY_SHAPE=1` :
1. **Fix bug** `_apply_hydro_analogue_weights` (shape_hourly.py:584,607) : utiliser `_climatological_fill[week_of_year(t)]` pour le weighting historique au lieu de `current_fill` global. Pour build Y+2/Y+3, idem comme cible.
2. **Split f_H = level × anomaly** : `level = mean_h(f_H_cell)`, `anomaly = f_H - level`. `shape_freedom['f_H']` damping à `assembler.py:303` damp **uniquement** le level, l'anomaly (signature saisonnière) survit à Y+2/Y+3.
3. **σ smoothing paramétrable** : default 0.5 quand flag OFF, 0.25 quand flag ON (lever mineur, bonus).

**Depends on**: Phase 5bis-A (baseline frozen + flag persisté + save/load complet).

**Requirements**: SHP-01, SHP-02, SHP-03, SHP-04 (déjà satisfaits par 5bis-A) — 5bis-B livre la VALEUR métier derrière.

**Success Criteria** :
1. Sur fixture EPEX-like réaliste avec duck curve : `np.ptp(factors_[("Ete","Ouvrable")])` strictement > baseline `main@5bis-A` (le bowl s'amplifie).
2. `assembler.build` produit une PFC où `|mean(price_shape[Dim, Été, h10-15]) − mean(price_shape[Dim, Hiver, h10-15])| > 5 €/MWh` (SC #2 original).
3. Tests par stage : assertion sur `df["f_H"]` post-damping à horizon M+30 garde une amplitude `> 0.X` (à calibrer).
4. Flag OFF reproduit baseline bit-pour-bit (régression assurée).
5. Suite 142 + nouveaux 5bis-A reste verte.

**Plans**: 3 plans (waves 1→2→3, sequential — plans 02/03 read same files modified by 01):
1. `05C-01-PLAN.md` (wave 1) — Lever 1: hydro kernel reformulation (per-timestamp `_climatological_fill[woy(t)]` gated by flag) + ctor extension `hydro_weight_sigma_off/_on` with backward-compat resolution + sidecar persistence + bowl fixture + tests D-A4-3 (kernel) and D-A4-8 (flag=OFF baseline).
2. `05C-02-PLAN.md` (wave 2) — Lever 2: module-level helper `_split_level_anomaly` + `__all__` + integration `assembler.build` at line 333 gated by flag + telemetry drift detection + tests D-A4-4 (split invariant) and D-A4-6 (f_H amplitude M+30).
3. `05C-03-PLAN.md` (wave 3) — Lever 3: ctor extension `sigma_off/_on` with unified resolution + sidecar (10 keys total) + telemetry init + new baseline `baseline_pfc_seed42_bowl.parquet` (RESEARCH Pitfall B: generated AFTER all three levers ship) + tests D-A4-5 (SC #1 ptp), D-A4-7 (SC #2 delta), D-A4-9 (new baseline convention) + PROJECT.md D-FLIP-1 entry + SUPERSEDED note on 05bis pre-doc.

---

### Phase 5: MSFC retire silent floors + PFC peut être négative
**Goal**: Autoriser une PFC négative aux heures structurelles, en retirant les 4 planchers silencieux actuels par ctor args defaults OFF (negative-ready) avec master flag `PFC_LT_ALLOW_NEGATIVE_PRICES` en audit-trail INFO log only (les 4 ctor args sont la véritable surface API, rollback opérateur explicite per D-A2-3).

**Recadrage** : Le titre historique "MSFC log-prix" est un artefact pré-audit (D-A1-1 : MSFC reste LINÉAIRE, aucune transformation log-space appliquée). CONTEXT.md autoritative pour le nouveau scope.

**Depends on**: Phase 5bis-A (baseline frozen, flag sidecar) + 5bis-B (bowl deepening shippé, requis pour SC #2 acceptance test).

**Requirements**: NEG-01, NEG-02, NEG-03, NEG-04, NEG-05 (reformulé per D-A4-7 dans Plan 05-01 : monthly forward négatif, pas Cal annuel).

**Success Criteria**:
  1. Un monthly forward négatif (e.g. July M-07'27 = -2 €/MWh, autres months positifs typiques EEX) est correctement repricé par la PFC à -2 €/MWh moyenne sur le mois (math invariance test). Couvre NEG-05 reformulé.
  2. Un mois solaire (juillet) avec forward 20 €/MWh positif produit, sous `PFC_LT_USE_SEASONAL_HOURLY_SHAPE=1` (bowl 5bis-B actif) + `PFC_LT_ALLOW_NEGATIVE_PRICES=1`, une PFC qui pique à -20 €/MWh ou moins à h13 dimanche (acceptance gated par 5bis-B bowl marker).
  3. Les 4 planchers silencieux (MSFC `np.maximum(B, 1.0)` ligne 131 ET ligne 203, `m_factor >= 0.1` ligne 517, F_WV_FLOOR ligne 394/407, peak ratio >= 1) sont rendus optionnels par 4 ctor args defaults OFF : `enforce_positivity=False`, `enforce_m_factor_floor=False`, `enforce_floor=False`, `allow_negative_peak=True`.
  4. La feature flag `PFC_LT_ALLOW_NEGATIVE_PRICES` est lue ONCE à `PFCAssembler.__init__` (audit-trail INFO log only per D-A2-2). Rollback opérateur = passer `enforce_*=True` / `allow_negative_peak=False` EXPLICITEMENT aux 4 callsites (D-A2-3).
  5. La PFC mid-market reste IDENTIQUE au baseline 5bis-A (`baseline_pfc_seed42.parquet`) à `atol=1e-12, rtol=0` quand les 4 ctor args sont passés en mode legacy (`enforce_*=True, allow_negative_peak=False`). Le baseline canonique Phase 5 (`baseline_pfc_seed42_phase05.parquet`) est généré avec defaults negative-ready et asservi à la même tolerance.

**Plans**: 3 plans séquentiels (wave 1 → 2 → 3, no parallelism — partage `assembler.py` entre Plans 02 et 03)
- `05-01-PLAN.md` (wave 1) — MSFC `enforce_positivity` + ArbitrageFreeCalibrator `enforce_m_factor_floor` + REQUIREMENTS.md NEG-05 reformulation per D-A4-7 + scaffolding `tests/test_phase05_negative_prices.py` (10 tests, 2 populated, 8 skip-stubbed). Couvre NEG-01, NEG-02, NEG-05.
- `05-02-PLAN.md` (wave 2) — WaterValueCorrection `enforce_floor` + nouveau `compute_delta_wv(B_smooth, fill_df, calendar_df) → pd.Series` + `assembler.build()` refactor `P = B × f_H × f_W + delta_wv` + telemetry INFO (D-A3-5) + 2 NEG-03 tests populés. Couvre NEG-03.
- `05-03-PLAN.md` (wave 3) — ContractCascader `allow_negative_peak` + `fit_peak_spreads` + `fit_peak_ratios` DeprecationWarning shim + master flag `PFC_LT_ALLOW_NEGATIVE_PRICES` audit-trail INFO log + fixture/baseline parquets + 4 callsite migrations (`production_phases.py:344,644` migrations explicites, doc comments aux 4 callsites) + 6 tests promus (NEG-04 ×2, master flag, SC #2 acceptance gated, 2 regressions canonique+rollback) + `.planning/PROJECT.md` D-FLIP-2 entry. Couvre NEG-04, NEG-05.

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

### Phase 10: Backtest par bloc vs HFC OMPEX
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
