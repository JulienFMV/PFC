# Requirements: PFC Long-Term

**Defined:** 2026-05-18
**Core Value:** Pricing trading-grade des blocs profil client (10-15 solaire,
18-9 nuit) avec ≥ 1.5 €/MWh de MAE-bloc en moins que HFC OMPEX.

## v1 Requirements

Les exigences ci-dessous constituent la cible minimale pour considérer la PFC
LT "SOTA maison FMV" et démontrer sa supériorité vs OMPEX sur les profile deals.

### Shape Quality (priorité 1)

- [ ] **SHP-01** : `ShapeHourly.factors_` est indexé par `(saison, type_jour, hour)`
  au lieu de `(saison, type_jour) → array[24]`. Une recherche sur une heure
  donne une valeur conditionnelle au triplet, pas une moyenne.
- [ ] **SHP-02** : `assembler.build` consomme le shape seasonal-hour sur le
  bon `country` (la signature `apply(idx, cal, country=...)` route via
  `_country_local_tz` introduit en Bloc A).
- [ ] **SHP-03** : `mean(f_H)` calculé sur chaque combinaison `(saison, type_jour)`
  reste ≈ 1.0 ± 1e-3 — l'invariant énergétique mensuel est préservé.
- [ ] **SHP-04** : Une feature flag `PFC_LT_USE_SEASONAL_HOURLY_SHAPE` (default
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
- [ ] **NEG-05** : Un Cal'27 forward coté -10 €/MWh est correctement repricé par
  la PFC à -10 €/MWh moyenne sur l'année (test de bout-en-bout).

### Block Distribution / Risk Premium (priorité 2)

- [ ] **DIST-01** : Méthode publique
  `pfc_block_distribution(start, end, hours_mask) → (p10, p50, p90)`
  retourne la distribution de la moyenne d'un bloc client via Monte-Carlo shape
  (N=500 trajectoires minimum).
- [ ] **DIST-02** : `(p90 - p50)` sur le bloc 10-15 (juillet, Cal'27 horizon)
  est > 5 €/MWh (sanity : on capture une vraie incertitude shape).
- [ ] **DIST-03** : Les distributions sont déterministes vs `random_state` fixé
  (test reproductibilité).

### Backtest by Block (priorité 1)

- [ ] **BT-01** : Le harness backtest accepte une définition de bloc client
  `{start_hour, end_hour, dow_filter, month_filter}` et retourne MAE par bloc.
- [ ] **BT-02** : Backtest 2024-2025 sur CH spot, blocs : `10-15 weekday`,
  `18-9 weekday`, `12-16 weekend`, `solar bowl summer`, `evening peak winter`.
- [ ] **BT-03** : Le harness compare PFC FMV vs HFC OMPEX (lu depuis
  `config.yaml → forwards.hfc_benchmark_dir`) bloc par bloc.
- [ ] **BT-04** : Sortie : tableau markdown avec colonnes
  `bloc | MAE FMV | MAE OMPEX | Δ MAE | DM test p-value`.
- [ ] **BT-05** : Cible de validation : Δ MAE ≤ -1.5 €/MWh sur au moins 3 blocs
  des 5 testés.

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
| BT-01 → BT-05 | Phase 10 (refondu) | parallèle 5bis/5/5ter |
| MULT-01 → MULT-03 | Phase 3 | HOLD |
| FUND-01 → FUND-02 | Phase 5quater | v2 |
| WV-01 → WV-02 | Phase 6 | v2 |
| BAS-01 → BAS-02 | Phase 4 + Phase 8 | HOLD |
| COM-01 | Phase 7 | v2 |
