# Requirements: PFC Long-Term

**Defined:** 2026-05-18
**Core Value:** Pricing trading-grade des blocs profil client (10-15 solaire,
18-9 nuit) avec ≥ 1.5 €/MWh de MAE-bloc en moins que HFC OMPEX.

## v1 Requirements

Les exigences ci-dessous constituent la cible minimale pour considérer la PFC
LT "SOTA maison FMV" et démontrer sa supériorité vs OMPEX sur les profile deals.

### Shape Quality (priorité 1)

- [x] **SHP-01** : `ShapeHourly.factors_` est indexé par `(saison, type_jour, hour)`
  au lieu de `(saison, type_jour) → array[24]`. Une recherche sur une heure
  donne une valeur conditionnelle au triplet, pas une moyenne.
- [x] **SHP-02** : `assembler.build` consomme le shape seasonal-hour sur le
  bon `country` (la signature `apply(idx, cal, country=...)` route via
  `_country_local_tz` introduit en Bloc A).
- [x] **SHP-03** : `mean(f_H)` calculé sur chaque combinaison `(saison, type_jour)`
  reste ≈ 1.0 ± 1e-3 — l'invariant énergétique mensuel est préservé.
- [x] **SHP-04** : Une feature flag `PFC_LT_USE_SEASONAL_HOURLY_SHAPE` (default
  False) permet de désactiver et revenir au comportement legacy.

### Negative Prices (priorité 1)

- [ ] **NEG-01** : Le plancher `np.maximum(B_smooth, 1.0)` dans `msfc_spline.py`
  est conditionné par une option `enforce_positivity=False` (default off pour LT).
- [ ] **NEG-02** : Le clip `m_factor = np.maximum(m_factor, 0.1)` dans
  `arbitrage_free.py` ne masque plus les résidus de calibration : `converged=False`
  est correctement propagé quand le clip est appliqué.
- [ ] **NEG-03** : `WaterValueCorrection.F_WV_FLOOR` peut être < 0 (configurable),
  ou est explicitement désactivé sur l'export LT mid-market.
- [ ] **NEG-04** : `cascading.synthesize_peak_prices` n'impose pas un ratio ≥ 1
  sur les forwards Cal négatifs.
- [ ] **NEG-05** : Un monthly forward négatif (e.g., July M-07'27 = -2 €/MWh, autres months positifs typiques EEX) est correctement repricé par la PFC à -2 €/MWh moyenne sur le mois (math invariance test, vérifie l'absence des floors silencieux ligne 131 et ligne 203 du MSFC spline). Note (2026-05-19) : reformulation post-discuss-phase Phase 5 — un Cal annuel n'est jamais négatif en pratique, ce sont des heures structurelles qui le sont (D-A4-7 / 05-CONTEXT.md).

### Block Distribution / Risk Premium (priorité 2)

- [ ] **DIST-01** : Méthode publique
  `pfc_block_distribution(start, end, hours_mask) → (p10, p50, p90)`
  retourne la distribution de la moyenne d'un bloc client via Monte-Carlo shape
  (N=500 trajectoires minimum).
- [ ] **DIST-02** : `(p90 - p50)` sur le bloc 10-15 (juillet, Cal'27 horizon)
  est > 5 €/MWh (sanity : on capture une vraie incertitude shape).
- [ ] **DIST-03** : Les distributions sont déterministes vs `random_state` fixé
  (test reproductibilité).

### Backtest by Block / PFC FMV Quality Scorecard (priorité 1)

**Pivot D-A0-2 (2026-05-20) :** BT-03 et BT-05 (vs HFC OMPEX) migrés vers
nouveau groupe BT-10B-* (Phase 10B deferred). BT-04 reformulé pour DM vs 3
naive baselines maison. BT-06..BT-10 ajoutés (1 par pilier scorecard Phase 10).

- [ ] **BT-01** : Le harness backtest accepte une définition de bloc client
  `{start_hour, end_hour, dow_filter, month_filter}` et retourne MAE par bloc.
- [ ] **BT-02** : Backtest 2024-2025 sur CH spot, **5 blocs renommés D-A2-2** :
  `block_overnight_weekday` (18-9 Lun-Ven, crosses midnight),
  `block_midday_weekday` (10-15 Lun-Ven),
  `block_weekend_midday` (11-15 Sam-Dim),
  `block_summer_solar_bowl` (11-14 mai-août toutes DOW),
  `block_winter_evening_peak` (17-21 Lun-Ven nov-fév).
- [ ] **BT-04** (reformulé Plan 10-01, D-A0-2) : Sortie : tableau markdown
  avec colonnes `bloc | baseline | horizon | MAE_PFC | MAE_baseline | Δ MAE |
  DM-stat | p-value | better_than_baseline (Y/N)`. Baselines maison =
  climatology, persistence Y-1, forwards-flat-no-shape (D-A4-1 CONTEXT).
  DM test avec Newey-West HAC lag=h-1 + HLN small-sample correction (D-A4-2 CONTEXT).
- [~] **BT-06** (Pillar 1 — Structural Hildmann) : Sous Config 4 (bowl ON +
  floors negative-ready), les 4 tests structurels Hildmann PASS :
  arb-free < 0.01 €/MWh sur chaque Cal/Q/M tradé,
  holiday/weekend ratio ∈ [0.65, 0.95] (threshold décidé Plan 10-01 NOTES,
  branche IF research default confirmé empiriquement, ratio mesuré = 0.8033),
  seasonal corr > 0.85 entre PFC monthly signature et EPEX 2019-2023 monthly signature,
  continuity max-jump < 2 €/MWh aux frontières mensuelles.
  **SC#1 UNIQUE GATE Phase 10.**
  *Plan 10-04 real-run 2026-05-21 : 2/4 PASS sous `forwards_source=fallback_diagnostic` —
  DIAGNOSTIC ONLY, not gate-eligible. D-FLIP-1 BLOCKED (PROJECT.md 2026-05-21).
  Gate-eligible run reporté Phase 10B (FMV poste H:\\).*
- [x] **BT-07** (Pillar 2 — Empirical KYOS) : Pour chaque cellule (bloc × horizon
  ∈ {M+1, M+3, M+6, Y+1, Y+2} × config), MAE, RMSE, bias absolu et
  Mincer-Zarnowitz régression `realised = α + β·pred + ε` avec test joint
  `α=0 & β=1` (Wald via `statsmodels.OLS.f_test`) reportés dans le scorecard markdown.
  *Livré Plan 10-04 : 100 rows dans `scorecard_kpis_pillar2.parquet` + tables markdown
  dans `10-VERIFICATION.md`.*
- [x] **BT-08** (Pillar 3 — Christoffersen unconditional) : IC80 source =
  `Uncertainty(n_boot=500, seed=42)` bootstrap (cols p10/p90). Per (bloc × horizon)
  sur Config 4 : test binomial LR_uc Christoffersen 1998 (`H0: observed_freq ==
  nominal`) reportée dans le scorecard.
  **IC95 (p2.5/p97.5) NON supportée par Uncertainty API actuelle → déférée
  Phase 5ter** (CONTEXT D-A3-3 amendé). Conditional coverage + reliability
  diagrams aussi déférés Phase 5ter.
  *Livré Plan 10-04 : 5 rows (1 par bloc) dans `scorecard_kpis_pillar3.parquet`,
  IC80 only, IC95 audit-trail dans 10-VERIFICATION.md §Annexes.*
- [x] **BT-09** (Pillar 4 — DM vs naive baselines) : 3 baselines maison
  (climatology = mean(EPEX hist pré-2024) par bloc, persistence Y-1 = mean(EPEX |
  bloc, vintage-1yr ± 15j), forwards-flat-no-shape = forward EEX Cal/Q/M
  as-of vintage sans shape). DM test avec loss differential MAE-cohérent et
  HAC Newey-West lag=h-1 + HLN small-sample correction. **Reformulation
  de BT-04 historique** (qui était vs OMPEX → migré vers BT-10B-*).
  *Livré Plan 10-04 : 300 rows dans `scorecard_kpis_pillar4.parquet` (4 configs ×
  5 blocs × 5 horizons × 3 baselines), DM stats + p-values + better_than_baseline.*
- [x] **BT-10** (Pillar 5 — Peer review SOTA) : Table comparative 9×6 (9
  features methodology × 6 références PFC FMV/KYOS/Volue/EULER/Benth-Koekebakker/Caldana) +
  gap analysis 3 paragraphes (~150 mots chacun : où FMV est SOTA / où il y a
  gap actionnable / où on innove) intégrés dans `10-VERIFICATION.md`.
  *Livré Plan 10-04 Task 3 : table 9×6 + 3 paragraphs + sources (6 références).*

### Backtest comparatif HFC OMPEX (Phase 10B, deferred — requires FMV poste)

- [ ] **BT-10B-01** (ex-BT-03) : Le harness compare PFC FMV vs HFC OMPEX
  (chemin `H:\Energy\GeCom\MARCHE & NEGOCE\Prix\Analyse HFC\HFC test\ER -HFC_OMPEX_15min`)
  bloc par bloc. Réutilise toute l'infra Phase 10 — c'est purement un ajout
  de baseline + un slot column dans la table comparative.
- [ ] **BT-10B-02** (ex-BT-05) : Cible de validation : Δ MAE PFC FMV vs HFC
  OMPEX ≤ -1.5 €/MWh sur au moins 3 blocs des 5 testés. Migration depuis
  BT-05 historique.

### Multi-market readiness (priorité 3, post-deal CH)

- [ ] **MULT-01** : `_build_long_term_branch(spec)` consume une `MarketSpec` FR/AT/IT
  sans crash, même si seasonal_ratios short (< 1 an d'historique spot).
- [ ] **MULT-02** : `calendar_ch.py` étendu pour `country ∈ {CH, DE, AT, FR, IT}`
  via `holidays.{Switzerland|Germany|Austria|France|Italy}`.
- [ ] **MULT-03** : Backtest par bloc fonctionne pour les 5 marchés (smoke test
  seulement, validation prix réelle dépend de la dispo HFC OMPEX par marché).

## v2 Requirements

Différés post-v1. Trackés mais hors roadmap immédiate.

### Fundamentals-driven shape

- **FUND-01** : Residual demand forecast (`load - 0.85*solar - 0.95*wind`)
  comme driver alternatif de f_H pour Y+1..Y+3.
- **FUND-02** : Trajectoire PV capacity installée (CRE FR, BNetzA DE, Pronovo CH)
  intégrée comme input forward-looking.

### Stochastic water value

- **WV-01** : Modèle de stockage type LSMC sur niveau réservoir CH.
- **WV-02** : Distinction lake (saisonnier) vs run-of-river (flux).

### Cross-border basis

- **BAS-01** : Modèle stochastique basis CH-DE, CH-FR avec régime pré/post-SDAC.
- **BAS-02** : Contraintes interzonales molles dans la calibration arbitrage-free.

### Governed commodity curves

- **COM-01** : TTF / EUA / API2 depuis source institutionnelle (ICE, EEX, ou
  Refinitiv via Databricks Gold), remplaçant le fallback Yahoo proxy actuel.

## Out of Scope

| Feature | Raison |
|---|---|
| Court-terme J+1..J+10 | Géré sur `claude/ct-worktree`, hors périmètre LT |
| TSFM (Chronos, Timer-XL, Toto) | Non-pertinents sur horizon N+3 ans |
| Dashboard Streamlit rewrite | Consumer-only, pas une dépendance LT |
| EEX OP1/OP2 forwards | Non liquides / non quotés sur CH d'après vérification utilisateur 2026-05 |
| Refactor production-wide | Approche incrémentale phase-par-phase |

## Traceability

| Requirement | Phase | Status |
|---|---|---|
| SHP-01 → SHP-04 | Phase 5bis | NEXT |
| NEG-01 → NEG-05 | Phase 5 | post 5bis |
| DIST-01 → DIST-03 | Phase 5ter | post 5 |
| BT-01, BT-02, BT-04, BT-06..BT-10 | Phase 10 | active |
| BT-10B-01, BT-10B-02 | Phase 10B | deferred (requires FMV poste H:\) |
| MULT-01 → MULT-03 | Phase 3 | HOLD |
| FUND-01 → FUND-02 | Phase 5quater | v2 |
| WV-01 → WV-02 | Phase 6 | v2 |
| BAS-01 → BAS-02 | Phase 4 + Phase 8 | HOLD |
| COM-01 | Phase 7 | v2 |
