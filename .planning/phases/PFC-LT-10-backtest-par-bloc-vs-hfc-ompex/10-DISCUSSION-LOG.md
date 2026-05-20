# Phase 10: PFC FMV Quality Scorecard - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in 10-CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-20
**Phase:** 10 — PFC FMV Quality Scorecard (refondu, OMPEX deferred)
**Areas discussed:** Gray-area selection, Area 0 (Phase 10 reframing), Area A (Empirical accuracy), Area B (Structural + Probabilistic), Area C (DM + Peer review), Area D (Infra), Success criteria

---

## Initial gray-area selection (user selected all 4)

| Option | Description | Selected |
|--------|-------------|----------|
| As-of paradigm + vintage HFC | Single snapshot Q4-2023 vs walk-forward monthly vs trimestriel | ✓ |
| Block definitions précises | Confirm BT-02 5 blocs + tz + filter schema exact | ✓ |
| Flag state matrix + DM test impl | Run avec flag bowl ON only vs 2x2 ablation, DM via arch vs Newey-West maison | ✓ |
| Exec location + reporting depth | Mac Mini vs FMV, markdown only vs +figures, output/ vs VERIFICATION.md | ✓ |

**User's choice:** All 4 (multiSelect).
**Notes:** Avant qu'on commence le deep-dive, user pivote radicalement le périmètre Phase 10 — voir Area 0 ci-dessous.

---

## Area 0 — Phase 10 reframing (PIVOT MAJEUR initié par user)

### Q1 — As-of paradigm initial (sur l'hypothèse OMPEX benchmark)

| Option | Description | Selected |
|--------|-------------|----------|
| Walk-forward monthly (24 vintages) | 1 HFC OMPEX + 1 PFC FMV chaque fin de mois 2024-2025 | (refused) |
| Vintage unique Q4-2023 | 1 HFC OMPEX + 1 PFC FMV as-of 2023-12-29 | (refused) |
| Vintages trimestriels (8 points) | 1 HFC + 1 PFC FMV par fin de trimestre | (refused) |

**User's choice (free-text pivot):** *"Moi je pense qu'il faut oublier Ompex, il faut se focaliser sur la qualité absolue de notre HFC ou PFC, un plan comme ça, et en oublier le reste, c'est vraiment notre mission, c'est de faire la PFC pour FMV, state of the art la plus efficace et la plus aboutie possible."*
**Notes:** Pivot radical du périmètre Phase 10. La comparaison OMPEX (originellement le cœur de la phase via REQUIREMENTS BT-01..BT-05) est rejetée. Claude reflète le pivot et clarifie les implications (KPI -1.5 €/MWh retiré, BT-03/BT-04 reformulés, ROADMAP/REQUIREMENTS à mettre à jour).

### Q2 — Référence pour mesurer la "qualité absolue"

| Option | Description | Selected |
|--------|-------------|----------|
| Realised EPEX 2024-2025 seul | MAE/RMSE/bias par bloc vs spot réalisé, skill vs climatology | (continuation) |
| EPEX + benchmark naive internal | + climatology, persistence Y-1, forwards-flat | (continuation) |
| EPEX + Hildmann 2013 quality criteria | + tests structurels arb-free / continuity / etc. | (continuation) |
| Refus initial — Phase 10 à redessiner | Stopper, mettre à jour ROADMAP d'abord | |

**User's choice (free-text Socratic):** *"Comment font les autres? Qu'est-ce qui se fait sur le marché entre guillemets? Comment on fait les PFL? Comment on font toutes les recherches? On va essayer de se baser là-dessus, puis répliquer ça, non? Qu'est-ce que tu en penses?"*
**Notes:** User délègue le call expert mais oriente vers "replication SOTA literature". Claude consulte `reference_pfc_state_of_art.md` (mémoire user : Hildmann 2013 ETH, Benth-Koekebakker 2007 max smoothness, Caldana 2017, Christoffersen 1998, KYOS/Volue/EULER vendors) et propose un Quality Scorecard à 4 piliers SOTA.

### Q3 — Confirmation du reframing 5 piliers

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, 4 piliers + OMPEX deferred | Structural + Empirical + Christoffersen + DM. OMPEX → Phase 10B deferred | |
| Yes, mais 3 piliers (skip Christoffersen) | Christoffersen déféré Phase 5ter | |
| Yes, mais 5 piliers (ajoute peer review) | + 5e pilier "peer review SOTA" comparaison vs KYOS/Volue/EULER/papers | ✓ |
| Non, redessine encore | Stop, précise | |

**User's choice:** 5 piliers (avec peer review SOTA literature comparison).
**Notes:** Confirme le scope EPFL-grade. Le 5e pilier ajoute la dimension méta "comment on se positionne face à la literature". Cohérent avec "qu'est-ce qui se fait sur le marché".

### Q-confirm — OMPEX deferral (user invites challenge)

**User's free-text:** *"On pourra ensuite benchmarker le shaping entre ompex et ce qu'on aura fait, mais plus tard. De nouveau, tu peux challenger cela, mais je crois qu'on peut vivre sans devoir comparer tout de suite notre HFC à celle de ompex."*

**Claude's challenge response:** Validation de la defer. 3 raisons pro-defer (mission = la PFC FMV pas la comparaison ; H:\ inaccessible Mac Mini → CI bloqué si dépendance OMPEX ; SOTA literature offre des critères de qualité absolue meilleurs qu'un benchmark concurrent). Risque résiduel (perte argument commercial 250k€/deal) mitigé par Phase 10B deferred explicit dans ROADMAP.

---

## Area A — Empirical accuracy (Pillar 2)

### A1 — Walk-forward cadence

| Option | Description | Selected |
|--------|-------------|----------|
| Monthly (24 vintages) | Re-build PFC chaque fin de mois, 24 samples/bloc/horizon | ✓ (Claude expert) |
| Quarterly (8 vintages) | Aligne reporting trimestriel FMV, 8-9 samples | |
| Single vintage Q4-2023 | 1 PFC immobile, simple mais DM faible | |
| Bi-mensuel (12 vintages) | Mid-month + end-month, ajoute complexité d'alignement EEX | |

**User's choice (free-text delegation):** *"Tu es l'expert et tu choisis."*
**Notes:** Claude choisit Monthly (24 vintages). Rationale : production-realistic, DM puissant, ~1h compute Mac Mini. End-of-month last business day, tz Europe/Zurich → UTC.

### A2 — Block definitions

| Option | Description | Selected |
|--------|-------------|----------|
| Adopter les 5 proposés tels quels | 5 masques REQUIREMENTS BT-02 + summer_solar_bowl + winter_evening_peak | (refined) |
| Ajuster un ou plusieurs masques | User précise | |
| Ajouter blocs (>5) | + peak_hours_year_round, holiday_pattern, shoulder_seasons | |
| Moins de blocs (3) | Garder seulement 3 client-relevant | |

**User's choice (free-text delegation):** *"Tu es l'expert et tu regardes s'il faut encore retravailler ça."*
**Notes:** Claude raffine les 5 blocs (correction naming "offpeak morning 10-15" qui est en fait peak EEX hours, renommé `block_midday_weekday`). Locked in 10-CONTEXT.md D-A2-2.

### A3 — Horizons évalués

| Option | Description | Selected |
|--------|-------------|----------|
| M+1, M+3, M+6, Y+1, Y+2 | 5 horizons, mix near-term + Cals | ✓ |
| Tous M+1..Y+3 (15 colonnes) | Exhaustif mais redondant | |
| Seulement Y+1 et Y+2 | Focus deals profil business | |
| M+3, Y+1, Y+2 | 3 horizons compromis | |

**User's choice:** M+1, M+3, M+6, Y+1, Y+2.
**Notes:** Aligne avec horizons forwards liquides (M+1..M+6 monthly) + Cals tradés (Y+1, Y+2). Skip M+12 (~=Y+1) et Y+3 (illiquide).

---

## Area B — Structural (Pillar 1) + Probabilistic (Pillar 3)

### B1 — Hildmann tolerances

| Option | Description | Selected |
|--------|-------------|----------|
| SOTA-grade (KYOS/Volue equivalent) | arb-free<0.01, continuity<2, holiday[0.65,0.95], seasonal corr>0.85 | ✓ |
| Trader-grade (plus lax) | arb-free<0.1, continuity<5, holiday[0.55,1.0], corr>0.7 | |
| Calibration first run | Mesurer puis lock au 90th percentile observé | |

**User's choice:** SOTA-grade.
**Notes:** Alignment direct avec literature KYOS/Volue. Plus strict que le `calibration_tol: 0.01` du modèle (test mesure après pipeline complet, pas seulement la calibration).

### B2 — Christoffersen probabilistic

| Option | Description | Selected |
|--------|-------------|----------|
| Inclure unconditional only | Binomial test obs vs nominal 80%/95%, via Uncertainty bootstrap | ✓ |
| Inclure complet (uncond + cond + reliability) | Full Christoffersen 1998 framework | |
| Déférer à Phase 5ter | Phase 5ter native scope | |

**User's choice:** Unconditional only.
**Notes:** Pragmatique. Réutilise `Uncertainty` class déjà câblé. Cond + reliability diagrams → Phase 5ter (distribution probabiliste native).

---

## Area C — DM (Pillar 4) + Peer review (Pillar 5)

### C1 — DM naive baselines

| Option | Description | Selected |
|--------|-------------|----------|
| 3 baselines (climatology + persistence + forwards-flat) | 3 angles différents, robuste | ✓ |
| 2 baselines (climatology + forwards-flat) | Skip persistence Y-1 | |
| 1 baseline (forwards-flat-no-shape) | Le plus pure, mesure shape value-add seul | |
| 4 baselines (+ HFC OMPEX deferred placeholder) | Slot vide pour Phase 10B | |

**User's choice:** 3 baselines.
**Notes:** Robuste statistiquement. La placeholder OMPEX (option 4) est documentée mais pas réservée — Phase 10B ajoutera la colonne quand exécuté.

### C2 — Peer review depth

| Option | Description | Selected |
|--------|-------------|----------|
| Light — table + gap analysis (2 pages) | 7×6 table features × vendors + 3-paragraph gap analysis | ✓ |
| Medium — +description methodology (5 pages) | Para par vendor/paper + citations | |
| Heavy — +annexes mathématiques (10-15 pages) | Working paper-grade | |

**User's choice:** Light (2 pages).
**Notes:** Suffisant pour discussion interne FMV. Si demande de présentation externe/audit post-merge → easy upgrade vers Medium.

---

## Area D — Infrastructure

### D1 — Data sourcing (Claude default, no question)

**Claude's decision:** EPEX 15-min réalisé via pipelines existants (energy_charts.info primary, SMARD/ENTSO-E fallbacks, config.yaml:33-55). Cache `data/epex_15min.parquet` gitignored, bootstrap au premier run.
**Notes:** Pas de question car pattern bien établi dans le repo. Pas de dépendance H:\ ni Databricks pour Phase 10.

### D2 — Flag matrix ablation

| Option | Description | Selected |
|--------|-------------|----------|
| 2x2 ablation full (bowl OFF/ON × floors OFF/ON) | 96 PFC builds, isole 5bis-B + Phase 5 séparément | ✓ |
| 1x2 bowl ablation seul | 48 builds, isole 5bis-B uniquement | |
| 1 config production target | 24 builds, mesure prod state | |
| 3 configs progression arc | 72 builds, storytelling legacy → 5bis-A → target | |

**User's choice:** 2x2 ablation full.
**Notes:** Plus EPFL-grade. ~4h compute Mac Mini. SC#1 gate évalué sur Config 4 (production target = bowl ON + floors negative-ready). Pillars 2-5 reportent les 4 configs pour ablation compare-effect.

### D3 — Reporting location

| Option | Description | Selected |
|--------|-------------|----------|
| VERIFICATION.md + figures embedded | `.planning/phases/.../10-VERIFICATION.md` + figures/ dans même dossier | ✓ |
| output/ versionné + lien VERIFICATION | Public dans repo, partageable hors gsd | |
| Both (DRY violation) | Risque drift | |
| HTML interactive | Plotly standalone | |

**User's choice:** VERIFICATION.md + figures embedded.
**Notes:** Convention gsd respectée. Single source of truth. Audit trail naturel via git. Si demande management de format présentation → easy add-on post-merge (Plotly HTML, PDF).

---

## Success Criteria (gate D-FLIP-1)

| Option | Description | Selected |
|--------|-------------|----------|
| SC#1 Hildmann 4/4 PASS (bowl ON config) | 4 tests structurels Hildmann PASS sur config target | ✓ |
| SC#2 DM p<0.05 vs ≥2 baselines sur ≥3 blocs | PFC FMV bat 2/3 baselines avec p<0.05 sur 3/5 blocs | |
| SC#3 Christoffersen IC80 ∈ [75%, 85%] | Couverture IC80 ± 5% sur ≥3 blocs | |
| SC#4 Ablation 2x2 bowl ON ≥ bowl OFF MAE | bowl ON MAE ≤ bowl OFF MAE sur ≥3 blocs | |

**User's choice:** SC#1 only.
**Notes:** User explicit decision : qualité absolue structurelle = unique gate. Pillars 2-5 produisent du content informational/diagnostic mais ne bloquent pas la phase ni le flip D-FLIP-1. Évite le risque "fail-on-DM-because-baseline-was-too-good-by-chance" sur 24 samples.

---

## Claude's Discretion

- Naming exact des modules/classes (`structural_tests.py`, `block_masks.py`, `scorecard.py`, `christoffersen.py`, `dm_test.py`) — alternatives acceptées au planning.
- Format figures matplotlib : scatter pred-realised par bloc (Pillar 2), bar chart MAE par horizon (Pillar 2), reliability-like overlay IC observed vs nominal (Pillar 3). Aucune figure pour Pillars 1/4/5 (tables markdown seules) — trancher au planning si plus d'utilité.
- Mincer-Zarnowitz Wald test implementation (statsmodels OLS + f_test).
- Lag DM HAC = h-1 (DM 1995 recommendation), edge case si Newey-West variance négative.
- Persistence Y-1 window : ±15 jours autour de `vintage_date - 1yr` pour suffisamment de samples.
- Format scorecard markdown final (5 sections par pillar + executive summary 1 page + TOC) — trancher au planning.
- Walk-forward cadence "Monthly (24 vintages)" — Claude expert call délégué par user A1.
- Block masks renaming + refinement (5 blocs renommés D-A2-2) — Claude expert call délégué par user A2.

## Deferred Ideas

- **Phase 10B (NEW)** : Backtest comparatif PFC FMV vs HFC OMPEX 2024-2025 depuis poste FMV avec H:\, KPI Δ MAE ≤ -1.5 €/MWh. Réutilise toute l'infra Phase 10.
- **Phase 5ter (déjà roadmap)** : hérite block_masks + scorecard harness Phase 10, ajoute Christoffersen conditional + reliability diagrams + Monte-Carlo distribution.
- **ROADMAP/REQUIREMENTS updates** (Plan 10-01) : Phase 10 title refondre, BT-03/BT-04/BT-05 → Phase 10B group, nouveaux BT-06..BT-10 pour les 5 piliers scorecard.
- **HTML interactive Plotly** : easy add-on si demande management post-merge.
- **Daily walk-forward** (504 business days × 4 configs = 2016 builds, 16h compute) : pas en scope, ajout simple si demande.
- **Multi-market scorecard** (FR/DE/AT/IT) : hérite infra Phase 10 quand Phase 3 (HOLD) sera dé-prioritisée.
- **Benchmark vs LEAR/Chronos-2 short-term** : horizons différents, pas comparable directement. Nouvelle phase dédiée si besoin de mesurer convergence LT vs CT sur overlap M+1..M+3.
- **Mincer-Zarnowitz per-horizon** vs pooled : à valider au planning si besoin de plus granularité.
- **Working paper FMV interne** : Pillar 5 Medium/Heavy version si demande post-merge.
