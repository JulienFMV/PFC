# PFC Long-Term (FMV)

## What This Is

PFC = Price Forward Curve. Le module **long-terme** du projet PFC construit une
courbe 15min N+3 ans pour la Suisse et ses voisins (DE, FR, AT, IT), ancrée sur
les forwards EEX et calibrée arbitrage-free. C'est l'outil de pricing structurel
de FMV — pour le P&L management, la valorisation portefeuille, les deals
clients industriels (notamment profile deals jour/nuit), et la cohérence avec
les forwards tradables.

Le périmètre **court-terme** (J+1..J+10, LEAR + Chronos-2 + Foundation Models)
est géré séparément sur la branche `claude/ct-worktree` par un autre agent ; ce
projet GSD couvre exclusivement le long-terme.

## Core Value

Pricing **trading-grade** des blocs de profil client (e.g. bloc nuit 18-9 et
bloc solaire 10-15), avec un shape qui bat OMPEX d'au moins 1.5 €/MWh en MAE
sur le backtest 2024-2025. Sans ça, FMV perd ~250-500 k€ par grand deal sur
des erreurs de shape inhedgeables (OP1/OP2 non tradables).

## Requirements

### Validated

Aucun encore — validation passe par le backtest par bloc en Phase 10.

### Active

- [ ] **Shape seasonal × hour** sur les 5 marchés (Phase 5bis)
- [x] **Prix négatifs autorisés** par la PFC (Phase 5 livrée 2026-05-20, 4 floors retirés via ctor args defaults OFF, voir D-FLIP-2)
- [ ] **Distribution probabiliste par bloc** (Phase 5ter, Monte-Carlo shape)
- [ ] **Backtest par bloc client** vs HFC OMPEX (Phase 10 reformulé)
- [ ] **Activation FR/AT/IT** (Phase 3, HOLD — pas de valeur sur deals CH actuels)
- [ ] **Basis cross-border LT** (Phase 4, HOLD)
- [ ] **Stochastic water value** (Phase 6, plus tard)

### Out of Scope

- **Court-terme (LEAR / Chronos-2 / D+1..D+10)** — géré sur branche `claude/ct-worktree`.
  N'importer aucun module CT, ne pas modifier `pfc_shaping/ct/` ni
  `pfc_shaping/pipeline/swiss_short_term.py`.
- **Foundation Models long-terme** — aucun TSFM ne couvre N+3 ans, la PFC LT
  est dominée par la structure (forwards + shape ratios), pas par la prévision
  temporelle pure.
- **Dashboard rewriting** — la couche Streamlit est consommatrice seule.
- **Refactor production-wide** — on travaille incrémentalement, une phase à la fois.

## Context

- **FMV** est une utility hydroélectrique cantonale du Valais, exposée aux prix
  EPEX Swiss avec une production majoritairement hydro (réservoirs + RoR).
- Le **PFC fourni par OMPEX** (référence externe historique) est jugé inadapté
  sur les deals profil — d'où ce chantier.
- Le repo a été refactoré récemment pour séparer LT (`pfc_shaping/lt/*`) de
  CT (`pfc_shaping/ct/*`) avec un shim de compat pour les anciens imports.
- L'historique data EEX vient de deux fichiers Excel (Yearly + Historique2019)
  ingérés via `scripts/rebuild_forwards_history.py`.

## Constraints

- **Tech stack** : Python 3.11+, pandas, numpy, scipy, scikit-learn, holidays,
  openpyxl, pyarrow. Pas de TensorFlow/PyTorch côté LT.
- **Tz strict** : tout l'index interne est UTC ; les conversions locales (CH,
  DE, FR, AT, IT) passent par `_country_local_tz()` dans assembler.py.
- **No CT contamination** : LT n'importe jamais de `pfc_shaping.ct.*`.
- **Tests verts requis avant tout commit** : la suite `tests/` doit passer
  (142 passed, 4 skipped en env CT-deps-absentes).
- **Backward-compat** sur les anciens chemins `pfc_shaping.model.*` (shim
  émet `DeprecationWarning`, ne pas casser).
- **Branche unique** pour le LT : `claude/clean-lt-ct-integration`. Pas de
  travail sur `main` ni sur `claude/ct-worktree`.

## Key Decisions

| Date | Décision | Rationale |
|---|---|---|
| 2026-05-06 | Refactor LT/CT (commit 596e3c5) | Permettre 2 agents indépendants sans collision |
| 2026-05-06 | Phase 1ter parser EEX (commit 8d28b63) | Week products explicitement filtrés ; onglets FX/Produits/HFC skippés |
| 2026-05-06 | Phase 1bis `_build_long_term_branch(spec)` (commit 2aa99ea) | Generic market-aware, prep activation FR/AT/IT |
| 2026-05-06 | Phase 2 prix négatifs autorisés en ingestion (commits 867c51e + 0915f0e) | Sanity range `[-500, 10_000]` €/MWh, `0` = non-quoté ; vectorisation du quoted-mask (compat pandas <2.1) |
| 2026-05-18 | Audit profond LT 25 findings | Voir `.planning/research/audit-deep-lt-2026-05.md` |
| 2026-05-18 | Bloc A+C1+C2 fixes (commit 28dfd65) | country/tz plumbing + water_value causal + backtest reference_date |
| 2026-05-18 | **Hold Phase 3/4, prio Phase 5bis/5/5ter/10** | Profile deal P&L >250k€/deal écrase l'urgence FR/AT/IT |
| 2026-05-18 | **Cible block-MAE -1.5 €/MWh vs HFC OMPEX** | KPI métier mesurable pour démontrer SOTA en interne |
| 2026-05-19 | **Flag PFC_LT_USE_SEASONAL_HOURLY_SHAPE livré default OFF en Phase 5bis-B. Flip default ON gated par Phase 10 SC#1 PASS : 4 tests structurels Hildmann (arb-free, holiday/weekend, seasonal corr, continuity) PASS aux seuils SOTA-grade sous Config 4 (bowl ON + floors negative-ready) sur l'agrégat des 24 vintages 2024-2025. Le benchmark comparatif Δ MAE vs HFC OMPEX est désormais déféré Phase 10B (post-pivot 2026-05-20).** (D-FLIP-1) | EPFL/SOTA principle: no production change without empirical validation gate. Updated 2026-05-20 to reflect Phase 10 reframing (quality-absolute scorecard, not comparative). Voir .planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md D-FLIP-1 + .planning/phases/10-pfc-fmv-quality-scorecard/10-CONTEXT.md D-A0-3. |
| 2026-05-20 | **Phase 5 livré 2026-05-20, defaults negative-ready, master flag PFC_LT_ALLOW_NEGATIVE_PRICES audit-trail INFO only (D-A2-2). Rollback opérateur = construire PFCAssembler(..., enforce_positivity=True, enforce_m_factor_floor=True, enforce_floor=True, allow_negative_peak=False) au callsite production — les quatre kwargs sont sur PFCAssembler.__init__ par B1/B4 Approach B (D-A2-3).** (D-FLIP-2) | 4 planchers silencieux retirés par défaut : MSFC enforce_positivity=False, ArbitrageFreeCal enforce_m_factor_floor=False, WaterValue enforce_floor=False (→ delta-additif NEG-03), Cascading allow_negative_peak=True (spread-additif NEG-04). Test suite : 272 passed, 4 skipped (SC #2 gated-skip synth env). Voir .planning/phases/05-msfc-log-prix-retire-silent-floors/. |
| 2026-05-20 | **Phase 10 reframed as PFC FMV Quality Scorecard (5-pillar SOTA replication). OMPEX comparative benchmark deferred to new Phase 10B (requires FMV poste H:\\). SC#1 unique gate = Hildmann 4/4 PASS sous Config 4 (bowl ON + floors negative-ready). Pillars 2-5 produisent du content informational mais NE bloquent PAS la phase ni le flip D-FLIP-1.** (D-A0-1..D-A0-4) | User explicit pivot 2026-05-20 lors de /gsd:discuss-phase 10 : "oublier OMPEX, focus qualité absolue PFC FMV state-of-the-art, c'est notre mission... comment font les autres, qu'est-ce qui se fait sur le marché". EPFL principle "build it right by design, validate against principle, not just against competitor". 5 piliers = Hildmann structural (SC#1 gate) + KYOS empirical accuracy + Christoffersen unconditional + DM vs 3 naive baselines + peer review SOTA. Voir .planning/phases/10-pfc-fmv-quality-scorecard/10-CONTEXT.md. |
| 2026-05-21 | **Flip default ON de PFC_LT_USE_SEASONAL_HOURLY_SHAPE BLOQUÉ pour cette livraison Phase 10. Real-run Mac Mini 2026-05-21 (96 builds en 634.5s) sous `forwards_source=fallback_diagnostic` (snapshot EEX XLSX H:\\ inaccessible depuis Mac Mini). Verdict SC#1 Hildmann = 2/4 tests PASS sur agrégat 24 vintages 2024-2025, annoté "⚠ Diagnostic only — not gate-eligible" (preflight Check 2 + 10-VERIFICATION.md §Pillar 1 banner). Le run actuel n'est PAS gate-eligible par construction — il informe mais ne tranche pas. Le flip ON est explicitement déféré jusqu'à un real-run exécuté depuis le poste FMV avec accès H:\\ et `forwards_source=real_eex_xlsx` (Phase 10B context). D-A6-3 reproducibility contract verified (Task 5 subset 4 builds atol=1e-12 rtol=0 PASS sans fallback). D-FLIP-1 reste default OFF.** (D-FLIP-1 BLOCKED) | Phase 10 PFC FMV Quality Scorecard livré complet (scorecard.json + 4 figures matplotlib + 10-VERIFICATION.md 5 piliers + Pillar 5 peer-review SOTA 9×6 + REPRODUCIBILITY-EXCEPTIONS.md template + 349 tests verts dont reproducibility) — toutes deliverables shippées même si le verdict SC#1 demeure diagnostic-only par contrainte d'accès. Cohérent EPFL principle "no production change without empirical validation gate" : le flip ON reste verrouillé jusqu'à un real-run gate-eligible. Audit-trail : .planning/phases/10-pfc-fmv-quality-scorecard/10-VERIFICATION.md §Pillar 1 + 10-PREFLIGHT.md Check 2 + 10-04-SUMMARY.md. |
