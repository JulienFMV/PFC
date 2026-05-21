# Phase 10: PFC FMV Quality Scorecard (refondu, OMPEX deferred) — Context

**Gathered:** 2026-05-20
**Status:** Ready for planning
**Depends on:** Phase 5bis-A livrée (baseline frozen + flag persisté), Phase 5bis-B livrée (bowl deepening shippé sous `PFC_LT_USE_SEASONAL_HOURLY_SHAPE=1`), Phase 5 livrée (4 floors retirés, defaults negative-ready, `PFC_LT_ALLOW_NEGATIVE_PRICES` audit-trail).

**Style discussion:** User a explicitement pivoté le périmètre Phase 10 en cours de discuss : *"oublier OMPEX, focus qualité absolue PFC FMV state-of-the-art, c'est notre mission... comment font les autres, qu'est-ce qui se fait sur le marché"*. La phase devient un **Quality Scorecard à 5 piliers répliquant la SOTA literature** (Hildmann 2013, Benth-Koekebakker 2007, Christoffersen 1998, Diebold-Mariano 1995, KYOS/Volue/EULER vendor benchmarks). Le benchmark comparatif vs HFC OMPEX est **déféré à une nouvelle Phase 10B** exécutable plus tard depuis le poste FMV (accès H:\).

<domain>
## Phase Boundary

**Phase 10 livre un Quality Scorecard public-FMV-grade de la PFC FMV** mesurant la qualité absolue (structurelle, empirique, probabiliste) sur la période 2024-2025 via walk-forward backtest, et la positionnant face à la SOTA literature/vendors. Le scorecard est l'argument interne décisif pour démontrer que la PFC FMV est state-of-the-art **par construction**, sans dépendre d'un benchmark concurrent. C'est aussi le gate qui autorise le flip ON par défaut de `PFC_LT_USE_SEASONAL_HOURLY_SHAPE` (D-FLIP-1 dans PROJECT.md).

**Pivot acté vs roadmap original :**
- Le KPI "Δ MAE bloc ≤ -1.5 €/MWh vs HFC OMPEX" est retiré du SC Phase 10.
- BT-01..BT-05 dans REQUIREMENTS.md doivent être reformulés (deferred immediate fix en Plan 10-01 ou en mini-fix doc pré-plan).
- Nouvelle Phase 10B (deferred) couvre le benchmark OMPEX comparatif depuis poste FMV avec accès H:\.
- ROADMAP Phase 10 title : "Backtest par bloc client vs HFC OMPEX" → "PFC FMV Quality Scorecard (5-pillar SOTA replication)" + nouveau Phase 10B "PFC FMV vs HFC OMPEX block-MAE benchmark (deferred, requires FMV poste)".

**In scope (5 piliers du scorecard) :**

### Pillar 1 — Structural Quality (Hildmann 2013, ETH Zurich)
4 tests pass/fail aux seuils SOTA-grade équivalents KYOS/Volue :
- **Arbitrage-freeness** : `|mean(PFC | period) - forward_price|` < 0.01 €/MWh sur chaque Cal/Q/M tradés.
- **Holiday/weekend pattern** : `mean(PFC | weekend) / mean(PFC | weekday)` ∈ [0.65, 0.95] (CH typical).
- **Seasonal profile** : Pearson `corr(monthly_PFC_signature, monthly_EPEX_hist_signature)` > 0.85.
- **Continuity** : pas de saut > 2 €/MWh aux frontières Cal/Q/M (max abs diff entre dernière heure du mois M et première heure de M+1).

### Pillar 2 — Empirical Accuracy (KYOS/Volue style, vs realised EPEX)
Walk-forward backtest avec **24 vintages monthly** (dernier jour ouvré chaque mois 2024-01..2025-12, reference_date tz-aware) × **5 blocs client** × **5 horizons** :
- Blocs (tz Europe/Zurich pour les masks, conversion UTC pour l'index interne) :
  - `block_overnight_weekday` : 18h-09h, Lun-Ven, all months (crosses midnight)
  - `block_midday_weekday` : 10h-15h, Lun-Ven, all months
  - `block_weekend_midday` : 11h-15h, Sam-Dim, all months
  - `block_summer_solar_bowl` : 11h-14h, all DOW, mai-août (test 5bis-B duck curve)
  - `block_winter_evening_peak` : 17h-21h, Lun-Ven, nov-fév
- Horizons : M+1, M+3, M+6, Y+1, Y+2.
- KPIs par cellule : MAE, RMSE, bias absolu (€/MWh) ET Mincer-Zarnowitz regression `realised = α + β·pred + ε` (test α=0, β=1).

### Pillar 3 — Probabilistic Quality (Christoffersen 1998, unconditional only)
- IC80/IC95 source = `Uncertainty` class bootstrap (n=500) déjà câblé dans `pfc_shaping/lt/model/uncertainty.py` + utilisé `validation/backtest.py:189-191`.
- Test **unconditional coverage seul** (binomial : `H0: observed_freq == nominal`) sur chaque (bloc × horizon).
- Reliability diagram **non inclus** (déféré Phase 5ter).
- Test conditional coverage (Markov independence) **non inclus** (déféré Phase 5ter).

### Pillar 4 — Diebold-Mariano vs Naive Baselines (NOT vs OMPEX)
3 baselines maison :
- **Climatology** : profil moyen flat = valeur moyenne du bloc sur historique pré-2024 (1 chiffre par bloc).
- **Persistence Y-1** : même bloc, année précédente (= valeur réalisée du bloc 12 mois avant).
- **Forwards-flat-no-shape** : forward EEX brut (Cal/Q/M) au reference_date, sans shape intra-mois (= valeur uniforme constante sur le bloc).

DM test config :
- Loss differential `d_t = |e_PFC_t| - |e_baseline_t|` (MAE-cohérent).
- HAC : Newey-West avec lag = h-1 où h = horizon de forecast (recommandation DM 1995).
- Per (bloc × baseline), p-value reportée dans le scorecard.

### Pillar 5 — Peer Review SOTA (Light : table + gap analysis 2 pages)
- Table comparative : lignes = features methodology (level B, shape f_H, smoothness, arb-freeness, neg prices, probabilistic, peak/offpeak, hourly granularity, multi-market), colonnes = `PFC FMV` / `KYOS KyCurve` / `Volue HPFC` / `EULER (Phinergy)` / `Benth-Koekebakker 2007` / `Caldana 2017`.
- Cellules : oui/non + commentaire bref (1 phrase).
- Section "Gap analysis" : 3 paragraphes (où PFC FMV est SOTA / où il y a un gap actionnable / où on innove vs literature).
- Sources : `reference_pfc_state_of_art.md` mémoire + PFC LT codebase.

### Infrastructure
- **Ablation grid 2x2** : `bowl OFF/ON × floors OFF/ON` = 4 configs × 24 vintages = **96 PFC FMV builds** (~4h compute Mac Mini).
- **Data sourcing EPEX 15-min réalisé 2024-2025** : via pipelines existants `energy_charts.info` (primary, config.yaml:33) + SMARD/ENTSO-E (fallbacks). Cache local `data/epex_15min.parquet` (gitignored).
- **Reporting** : tout dans `.planning/phases/PFC-LT-10-backtest-par-bloc-vs-hfc-ompex/10-VERIFICATION.md` (convention gsd) + figures matplotlib PNG dans `.planning/phases/PFC-LT-10-backtest-par-bloc-vs-hfc-ompex/figures/`. Versionnés, suit le workflow gsd.
- **CI strategy** : tests unitaires harness avec mock synthetic data (Mac Mini-friendly). Real-run end-to-end depuis Mac Mini avec EPEX cached. Pas de dépendance H:\ ni OMPEX.

**Success criteria (gate D-FLIP-1) :**
- **SC#1 — UNIQUE GATE** : Sous config target (bowl ON + floors negative-ready, = post-5bis-B+5 production state), les 4 tests structurels Hildmann (Pillar 1) PASS aux seuils SOTA-grade (arb-free < 0.01 €/MWh, continuity < 2 €/MWh, holiday ratio ∈ [0.65, 0.95], seasonal corr > 0.85) sur l'agrégat des 24 vintages.

**Pillars 2-5 produisent du contenu informational mais NE bloquent PAS la phase** ni le flip D-FLIP-1 (user explicit decision : "qualité absolue, pas comparative").

**Out of scope (déférés) :**
- **Phase 10B (nouveau, deferred)** : Backtest comparatif PFC FMV vs HFC OMPEX 2024-2025, KPI `Δ MAE bloc ≤ -1.5 €/MWh`. Exécutable depuis poste FMV avec accès H:\ uniquement. Pattern fichier `HFC_Ompex_*.xlsx` (config.yaml:71).
- **Phase 5ter** : distribution probabiliste par bloc (DIST-01..03). Hérite de l'infra Phase 10 (block masks + walk-forward) et ajoute le test Christoffersen conditional + reliability diagrams.
- **Reformulation REQUIREMENTS BT-01..BT-05** : à fixup en Plan 10-01 ou pré-plan. BT-03 (vs HFC OMPEX) déplacé vers nouveau BT-10B-* groupe. BT-04 (DM test markdown table) reformulé pour les 3 naive baselines.
- **Phase 5quater fundamentals** : pas en scope Phase 10. Le scorecard mesure ce qui est livré aujourd'hui.
- **Tout `pfc_shaping/ct/*`** : LT only.
- **Daily walk-forward** (vs monthly) : trop lourd, peu de signal additionnel. Si demande post-merge → ajout simple.
- **HTML interactive scorecard** : markdown + PNG suffisent pour ce livrable.
- **Christoffersen conditional + reliability diagrams** : déférés Phase 5ter.

</domain>

<decisions>
## Implementation Decisions

### Area 0 — Reframing Phase 10 (pivot user du 2026-05-20)

- **D-A0-1 :** Phase 10 devient **PFC FMV Quality Scorecard** à 5 piliers SOTA literature replication (Hildmann + KYOS empirical + Christoffersen uncond + DM vs naive baselines + Peer review). OMPEX comparison **déférée** à nouvelle Phase 10B exécutable depuis poste FMV. Justification user verbatim : *"oublier OMPEX, focus qualité absolue PFC FMV state-of-the-art, c'est notre mission... comment font les autres, qu'est-ce qui se fait sur le marché, on va essayer de se baser là-dessus, puis répliquer ça"*.
- **D-A0-2 :** ROADMAP/REQUIREMENTS/PROJECT.md updates en Plan 10-01 (mini-fix doc) : nouveau title Phase 10, BT-01..BT-05 reformulés (BT-03/BT-04 OMPEX → BT-10B-* group deferred), nouveau key decision PROJECT.md "2026-05-20 : Phase 10 reframed as quality scorecard, OMPEX bench → Phase 10B deferred".
- **D-A0-3 :** SC#1 Hildmann 4/4 PASS = **UNIQUE GATE** verrouillant la phase + autorisant flip D-FLIP-1 `PFC_LT_USE_SEASONAL_HOURLY_SHAPE` default ON. Pillars 2-5 = informational/diagnostic, ne bloquent pas. User explicit choice : qualité absolue structurelle prime sur statistique/comparative.
- **D-A0-4 :** Le scorecard est versionné dans `.planning/phases/PFC-LT-10-backtest-par-bloc-vs-hfc-ompex/10-VERIFICATION.md` (gsd convention), pas dans `output/`. Single source of truth, audit-trail naturel via git.

### Area 1 — Pillar 1 Structural (Hildmann 2013)

- **D-A1-1 :** **4 tests structurels Hildmann** implémentés comme pytest avec assertions + numerical output dans le scorecard markdown. Seuils SOTA-grade équivalents KYOS/Volue :
  - **Arbitrage-freeness** : `|mean(PFC over period) - forward_price|` < 0.01 €/MWh (1 centime, plus strict que le `calibration_tol: 0.01` du modèle car le test mesure après tout le pipeline).
  - **Holiday/weekend** : `mean(PFC | weekend ∪ jours_fériés_CH) / mean(PFC | weekday hors fériés)` ∈ [0.65, 0.95].
  - **Seasonal profile** : `pearsonr(monthly_PFC_signature_2024_2025, monthly_EPEX_signature_2019_2023)` > 0.85.
  - **Continuity** : `max(|PFC[end of M_i] - PFC[start of M_{i+1}]|)` < 2 €/MWh aux 36 frontières mensuelles de l'horizon Y+3.
- **D-A1-2 :** Implémentation centralisée dans `pfc_shaping/validation/structural_tests.py` (nouveau fichier). 4 fonctions `test_arb_free(pfc, forwards, tol=0.01)`, `test_holiday_weekend(pfc, calendar, range_=(0.65,0.95))`, `test_seasonal_profile(pfc, epex_hist, min_corr=0.85)`, `test_continuity(pfc, max_jump=2.0)`. Retournent `(passed: bool, observed: float, threshold: float, details: dict)`.
- **D-A1-3 :** Test pytest correspondant dans `tests/test_phase10_hildmann.py` qui calle ces fonctions sur la PFC produite par `assembler.build()` en config target (bowl ON + floors negative-ready, seed=42 fixture). Suite verte = SC#1 PASS.

### Area 2 — Pillar 2 Empirical Accuracy

- **D-A2-1 :** **Walk-forward monthly 24 vintages** : dernier jour ouvré de chaque mois 2024-01..2025-12 (dates pinnées via `calendar_ch.last_business_day(month)` ou équivalent ; tz Europe/Zurich → UTC pour reference_date).
- **D-A2-2 :** **5 blocs client renommés** (correction du naming "offpeak morning" mal placé sur 10-15 = peak EEX hours) :
  - `block_overnight_weekday` (18h-09h, Lun-Ven, all months — crosses midnight, mask `(hour >= 18) | (hour < 9)` & weekday)
  - `block_midday_weekday` (10h-15h, Lun-Ven, all months)
  - `block_weekend_midday` (11h-15h, Sam-Dim, all months)
  - `block_summer_solar_bowl` (11h-14h, all DOW, mai-août — test 5bis-B duck curve)
  - `block_winter_evening_peak` (17h-21h, Lun-Ven, nov-fév)
- **D-A2-3 :** **5 horizons scorés** : M+1, M+3, M+6, Y+1, Y+2. Skip M+12 (redondant avec Y+1) et Y+3 (très peu liquide en forwards).
- **D-A2-4 :** **KPIs par cellule (bloc × horizon × config)** : MAE, RMSE, bias (€/MWh, absolu). Mincer-Zarnowitz régression `realised = α + β·pred + ε` avec test joint `α=0 & β=1` (Wald), p-value reportée.
- **D-A2-5 :** Realised EPEX 15-min 2024-2025 source : pipeline existant via `energy_charts.info` (primary, config.yaml:33), SMARD fallback. Cache `data/epex_15min.parquet` (gitignored). Aggrégation 15-min → hourly avec mean intra-hour pour aligner avec les blocs horaires.

### Area 3 — Pillar 3 Probabilistic (Christoffersen unconditional only)

- **D-A3-1 :** **Unconditional coverage seul** : binomial test `H0 : observed_freq_in_IC == nominal (80% ou 95%)`. Per (bloc × horizon). Cellule scorecard : `obs_freq (X/Y) — p-value Z`.
- **D-A3-2 :** **Source IC80/IC95** : `Uncertainty` class bootstrap n=500 (déjà câblé `validation/backtest.py:189-191`). Aucune extension code requise — réutilisation directe de l'API existante via `assembler.build(with_uncertainty=True)`.
- **D-A3-3 :** **Conditional coverage (Markov independence) + reliability diagrams** : **déférés Phase 5ter** (sa scope native est distribution probabiliste). Sera ajouté quand Phase 5ter shippera `pfc_block_distribution`.

### Area 4 — Pillar 4 DM vs Naive Baselines

- **D-A4-1 :** **3 naive baselines** computés en pre-step du backtest :
  - `baseline_climatology(bloc) = mean(EPEX_hist[bloc filter, 2019-2023])` (1 scalar par bloc, indépendant du horizon/vintage).
  - `baseline_persistence_y1(bloc, vintage_date) = mean(EPEX_realised[bloc filter, vintage_date - 1yr ± window])`.
  - `baseline_forwards_flat(bloc, horizon, vintage) = forward_EEX(Cal/Q/M matching horizon, as-of vintage_date)` (sans shape — valeur uniforme constante sur le bloc).
- **D-A4-2 :** **Diebold-Mariano implementation** : Newey-West HAC adjusted, loss `d_t = |e_PFC_t| - |e_baseline_t|` (MAE-cohérent). Lag = h-1 où h = horizon en mois (DM 1995 recommendation). Implémentation maison ~30 lignes dans `pfc_shaping/validation/dm_test.py` (évite ajouter dep `arch`). `statsmodels.tsa.stattools.acovf` réutilisable pour la HAC variance.
- **D-A4-3 :** **Output** : per (bloc × baseline × horizon), DM statistic + p-value (two-sided). Reportée dans la table du scorecard `block | baseline | horizon | DM-stat | p-value | better_than_baseline (Y/N)`.

### Area 5 — Pillar 5 Peer Review SOTA (Light)

- **D-A5-1 :** **Table comparative 9×6** : lignes = 9 features methodology (level B / shape f_H / smoothness / arbitrage-freeness / negative prices / probabilistic / peak-offpeak / hourly granularity / multi-market), colonnes = 6 références (`PFC FMV` / `KYOS KyCurve` / `Volue HPFC` / `EULER (Phinergy)` / `Benth-Koekebakker 2007` / `Caldana 2017`).
- **D-A5-2 :** Cellules : `oui/non` + commentaire bref (≤ 15 mots). Sources : `~/.claude/projects/-Users-julienbattaglia-Desktop-PFC/memory/reference_pfc_state_of_art.md` (mémoire user) + code PFC FMV pour la colonne `PFC FMV`.
- **D-A5-3 :** **Gap analysis** : 3 paragraphes (~150 mots chacun) : (1) où PFC FMV est SOTA (level smoothness MSFC + arb-free joint calibration + neg prices + shape seasonal × type_jour × hour + bowl deepening), (2) où il y a un gap actionnable (probabilistic light only, no peer review of forwards inputs, single-market CH non multi-zone), (3) où on innove (delta-additif WV sign-invariant, ctor args negative-ready convention, master flag audit-trail pattern).
- **D-A5-4 :** **Format** : 2 pages markdown intégrées dans `10-VERIFICATION.md`, section `## Pillar 5 — Peer Review SOTA`. Pas de PDF / pas d'annexes maths (déférés à un working-paper interne FMV si besoin post-merge).

### Area 6 — Infrastructure

- **D-A6-1 :** **Ablation grid 2x2** : 4 configs × 24 vintages = 96 builds.
  - Config 1 : bowl OFF + floors ON (legacy, pre-5bis-A baseline)
  - Config 2 : bowl ON + floors ON (5bis-B livré sans Phase 5)
  - Config 3 : bowl OFF + floors OFF (Phase 5 livré sans 5bis-B)
  - Config 4 : bowl ON + floors OFF (production target post-5bis-B+5)
  - SC#1 Hildmann gate évalué uniquement sur **Config 4** (production target).
  - Pillars 2-5 reportent les 4 configs (compare-effect ablation).
- **D-A6-2 :** **Data ingestion** : pipeline existant `pfc_shaping.data.*` via `energy_charts.info` (primary, config.yaml:33-35). Cache `data/epex_15min.parquet` (gitignored, premier-run bootstrap ~2 ans × 15-min ≈ 70k rows = quelques minutes download). Pas de dépendance Databricks / H:\.
- **D-A6-3 :** **Reproducibility contract** : `assert_frame_equal(scorecard_kpis_run1, scorecard_kpis_run2, check_exact=False, atol=1e-12, rtol=0)` — convention 5bis-A/B/5 tolerance préservée. Random seed=42 pour `Uncertainty` bootstrap (déjà géré via `seed=0` dans `Uncertainty(n_boot=200, seed=0)`).
- **D-A6-4 :** **Exec location** : tout depuis Mac Mini. Pas de dépendance H:\, pas de dépendance FMV poste, pas d'OMPEX. CI tests harness avec mock synthetic data + real-run end-to-end depuis Mac Mini avec EPEX cached.
- **D-A6-5 :** **Reporting** : `10-VERIFICATION.md` dans `.planning/phases/PFC-LT-10-backtest-par-bloc-vs-hfc-ompex/` + figures PNG matplotlib dans `figures/` du même dossier. Versionnés.

### Plan decomposition (preview, à finaliser au planning)

- **D-A7-1 :** Estimation 3-4 plans séquentiels (waves) :
  - **Plan 10-01-PLAN.md** (wave 1) — Infra : data ingestion EPEX historical via pipelines existants + `pfc_shaping/validation/block_masks.py` (5 blocs class-based) + harness backtest walk-forward `pfc_shaping/validation/scorecard.py` skeleton + REQUIREMENTS/ROADMAP/PROJECT.md updates (mini-fix doc). Pas encore de pillars implémentés.
  - **Plan 10-02-PLAN.md** (wave 2) — Pillars 1 + 2 : `structural_tests.py` (4 fonctions Hildmann + tests pytest = SC#1 gate) + table empirical accuracy MAE/RMSE/bias/MZ par (bloc × horizon × config). Suite `tests/test_phase10_hildmann.py` PASS sur config target.
  - **Plan 10-03-PLAN.md** (wave 3) — Pillars 3 + 4 : Christoffersen unconditional `pfc_shaping/validation/christoffersen.py` + DM test `pfc_shaping/validation/dm_test.py` + 3 baselines + tables intégrées scorecard. Tests pytest `tests/test_phase10_probabilistic.py` et `tests/test_phase10_dm.py`.
  - **Plan 10-04-PLAN.md** (wave 4) — Pillar 5 + final scorecard assembly + ablation grid 2x2 run end-to-end + figures matplotlib + `10-VERIFICATION.md` final write + PROJECT.md update D-FLIP-1 flip ON (gated by SC#1 PASS). Final commit.
- **D-A7-2 :** Plans réutilisent les conventions des phases précédentes : `gsd-executor` mode no-worktree, atomic commits per task, atol=1e-12 reproducibility contract, baseline fixtures convention.

### Claude's Discretion
- **Naming exact** des fonctions/classes/files (`structural_tests.py`, `block_masks.py`, `scorecard.py`, `dm_test.py`, `christoffersen.py`) : suggestions, alternatives acceptées au planning si plus cohérent avec code existant.
- **Format figures matplotlib** : scatter pred-realised par bloc (Pillar 2), bar chart MAE par horizon (Pillar 2), reliability-like overlay IC observed vs nominal (Pillar 3), aucune figure pour Pillars 1/4/5 (tables markdown seules). Trancher au planning si plus de figures aident.
- **Mincer-Zarnowitz Wald test** : statsmodels.regression.linear_model.OLS + `f_test('intercept=0, x=1')`. Pattern standard.
- **Lag DM HAC** : h-1 (DM 1995) — si h=1 (M+1), lag=0 = sample variance; pour h=24 (Y+2), lag=23 typique. À calibrer si Newey-West donne variance négative (cas edge).
- **Persistence Y-1 window** : ±15 jours autour de la date vintage_date - 1yr pour avoir suffisamment de samples par bloc.
- **Format scorecard markdown final** : 5 sections (1 par pillar) + executive summary 1 page en tête + table-of-contents. Trancher au planning pour la structure exacte.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Roadmap & requirements (à mettre à jour en Plan 10-01)
- [.planning/ROADMAP.md §Phase 10](.planning/ROADMAP.md) (lignes 181-194) — Title et goal à refondre : "Backtest par bloc vs HFC OMPEX" → "PFC FMV Quality Scorecard (5-pillar SOTA replication)". Nouveau Phase 10B "PFC FMV vs HFC OMPEX block-MAE benchmark (deferred, requires FMV poste)" à insérer.
- [.planning/REQUIREMENTS.md §Backtest by Block](.planning/REQUIREMENTS.md) (BT-01..BT-05) — Reformulation : BT-03 (vs HFC OMPEX) déplacé vers BT-10B-* group ; BT-04 reformulé pour DM vs naive baselines ; nouveau BT-06..BT-08 pour les 5 piliers du scorecard.
- [.planning/PROJECT.md §Key Decisions](.planning/PROJECT.md) — Nouvelle entrée 2026-05-20 : "Phase 10 reframed as quality scorecard, OMPEX → Phase 10B deferred". Update D-FLIP-1 wording : gate SC#1 Phase 10 Hildmann 4/4 PASS.

### Prior phase context (must read for continuity)
- [.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md](.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md) — 5bis-A pattern : freeze-at-init flag, sidecar `_meta.parquet`, baseline frozen, conftest autouse env hygiene.
- [.planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md](.planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md) — 5bis-B bowl deepening, σ_off/_on, hydro_weight_sigma_off/_on, D-FLIP-1 flip strategy (gated par Phase 10 success). Phase 10 produces the data backing D-FLIP-1.
- [.planning/phases/05-msfc-log-prix-retire-silent-floors/05-CONTEXT.md](.planning/phases/05-msfc-log-prix-retire-silent-floors/05-CONTEXT.md) — Phase 5 negative-ready defaults, ctor args pattern, master flag audit-trail. Phase 10 config target = bowl ON + floors negative-ready.

### Code à utiliser / étendre (Phase 10 scope)
- [pfc_shaping/lt/model/assembler.py](pfc_shaping/lt/model/assembler.py) — `PFCAssembler.build(reference_date=...)` déjà as-of-aware (utilisé `validation/backtest.py:207`). Réutilisé tel quel pour les 24 vintages walk-forward.
- [pfc_shaping/validation/backtest.py](pfc_shaping/validation/backtest.py) — walk-forward existant (recal mensuelle, KPIs shape ratios). Architecture partiellement réutilisable mais KPIs différents (Phase 10 = €/MWh par bloc, pas ratios). Considérer refactor commun ou wrapper.
- [pfc_shaping/validation/compare_hfc.py](pfc_shaping/validation/compare_hfc.py) — quick MAE/RMSE PFC vs HFC. Pattern d'alignment + metrics réutilisable pour Pillar 2.
- [pfc_shaping/lt/model/uncertainty.py](pfc_shaping/lt/model/uncertainty.py) — `Uncertainty(n_boot=500, seed=0).fit(...)` réutilisé pour Pillar 3 IC80/IC95.
- [pfc_shaping/data/calendar_ch.py](pfc_shaping/data/calendar_ch.py) — `enrich_15min_index` fournit saison/type_jour/heure_hce → utile pour les block masks Pillar 2.
- [pfc_shaping/config.yaml](pfc_shaping/config.yaml) (lignes 28-55) — energy_charts/smard/entsoe pipelines pour bootstrap EPEX historical realised. `forwards.eex_report_path` pour les forwards as-of historiques.

### Code à créer (Phase 10 nouveau)
- `pfc_shaping/validation/block_masks.py` — 5 blocs class-based avec `apply(idx_tz_zurich) → boolean mask`.
- `pfc_shaping/validation/structural_tests.py` — 4 fonctions Hildmann (Pillar 1).
- `pfc_shaping/validation/scorecard.py` — harness walk-forward 24 vintages + ablation grid 2x2 + KPIs aggregation Pillar 2.
- `pfc_shaping/validation/christoffersen.py` — unconditional coverage test (Pillar 3).
- `pfc_shaping/validation/dm_test.py` — Diebold-Mariano avec Newey-West HAC + 3 naive baselines (Pillar 4).
- `tests/test_phase10_hildmann.py` — pytest gate SC#1 (Pillar 1 PASS sur config target).
- `tests/test_phase10_empirical.py` — pytest sanity Pillar 2.
- `tests/test_phase10_probabilistic.py` — pytest Pillar 3.
- `tests/test_phase10_dm.py` — pytest Pillar 4.
- `data/epex_15min.parquet` (gitignored) — cache EPEX historical 15-min réalisé 2024-2025, bootstrap depuis energy_charts.info au premier run.

### Convention SOTA literature (Pillar 5 sources)
- [~/.claude/projects/-Users-julienbattaglia-Desktop-PFC/memory/reference_pfc_state_of_art.md](file:///Users/julienbattaglia/.claude/projects/-Users-julienbattaglia-Desktop-PFC/memory/reference_pfc_state_of_art.md) — Fleten-Lemming 2003, Benth-Koekebakker 2007 (Max Smoothness REFERENCE), Caldana 2017 thin granularity, Hildmann 2013 (ETH Zurich, 4 quality criteria), Adams-Van Deventer 1994, Biegler-König Pilz arbitrage-free shifting, KYOS KyCurve, Volue HPFC, EULER (Phinergy). Source primaire pour Pillar 5 peer review + Pillar 1 Hildmann formalisation.
- [~/.claude/projects/-Users-julienbattaglia-Desktop-PFC/memory/project_pfc_roadmap.md](file:///Users/julienbattaglia/.claude/projects/-Users-julienbattaglia-Desktop-PFC/memory/project_pfc_roadmap.md) — Business case profile-deal 250-500k€/5€MWh, alignment KPI structure.

### Convention agentique (réutilisée)
- [.planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md §code_context "Established Patterns"](.planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md) — Patterns 5bis-A/B/5 : freeze-at-init flag, sidecar persistence, tolerance contract `atol=1e-12 rtol=0`, `gsd-executor` mode no-worktree, atomic commits per task, baseline fixtures convention.
- [.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-REVIEWS.md](.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-REVIEWS.md) §1-2 — tolerance contract numerical-equality NOT byte-equivalence parquet.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- [pfc_shaping/lt/model/assembler.py](pfc_shaping/lt/model/assembler.py) `PFCAssembler.build(reference_date=...)` — déjà as-of-aware via reference_date param utilisé en `validation/backtest.py:207`. **Réutilisé tel quel** pour les 24 vintages walk-forward (Pillar 2). Pas de modification core code, juste un harness new.
- [pfc_shaping/validation/backtest.py](pfc_shaping/validation/backtest.py) — `WalkForwardBacktest` existing. Architecture utile mais KPIs différents (shape ratios vs Phase 10 = €/MWh par bloc). Refactor commun acceptable mais pas obligatoire — Phase 10 peut shipper `scorecard.py` séparé.
- [pfc_shaping/validation/compare_hfc.py](pfc_shaping/validation/compare_hfc.py) `_align_for_comparison`, `_metrics` — pattern d'alignment + MAE/RMSE/bias metrics. **Réutilisé directement** pour Pillar 2 KPI computation, étendu avec bias + MZ regression.
- [pfc_shaping/lt/model/uncertainty.py](pfc_shaping/lt/model/uncertainty.py) `Uncertainty(n_boot=500, seed=0).fit(...).predict_intervals(...)` — bootstrap IC80/IC95 déjà câblé. **Réutilisé tel quel** pour Pillar 3 (zéro extension code requise).
- [pfc_shaping/data/calendar_ch.py](pfc_shaping/data/calendar_ch.py) `enrich_15min_index(idx)` retourne saison/type_jour/heure_hce → **réutilisé** pour les 5 block masks Pillar 2.
- [tests/conftest.py](tests/conftest.py) autouse env hygiene (`PFC_LT_*` vars) — 5bis-A D-12. **Phase 10 hérite automatiquement**. Aucun ajout requis.
- [tests/fixtures/baseline_pfc_seed42.parquet](tests/fixtures/baseline_pfc_seed42.parquet) — 5bis-A baseline frozen, déterministe seed=42. **Réutilisé** comme fixture de référence dans tests Phase 10 (sanity build).
- [pfc_shaping/data/*ingest*.py](pfc_shaping/data/) — pipelines ingestion `energy_charts.info` (primary), SMARD (fallback), ENTSO-E (fallback). **Bootstrap automatique** au premier run pour EPEX historical 2024-2025.

### Established Patterns
- **Walk-forward backtest avec recal mensuelle** (`validation/backtest.py:103`) : pattern existant. Phase 10 étend en multi-config (4 configs ablation grid) + multi-bloc (5 blocs) + multi-horizon (5 horizons) en gardant la cadence monthly.
- **Tolerance contract `atol=1e-12 rtol=0`** (5bis-A REVIEWS, Phase 5) : applicable aux KPIs scorecard pour reproducibility test (re-run = bit-identical numerical output).
- **`gsd-executor` mode no-worktree, 1 plan = 1 wave** (5bis-A/B/5 convention) : Phase 10 = 4 plans séquentiels mêmes modes.
- **Numerical-equality NOT byte-equivalence parquet** (5bis-A REVIEWS addendum) : `assert_frame_equal(check_exact=False, atol=1e-12, rtol=0)` + identical columns/dtypes/index/sort order.
- **Mock synthetic data CI + real-run local** (5bis-A baseline pattern) : Phase 10 tests CI utilisent mock data déterministe, real-run scorecard utilise EPEX cached local.
- **Master flag audit-trail INFO log only, ctor args = vraie surface API** (Phase 5 D-A2-2) : Pas applicable directement, mais la grid 2x2 ablation est buildée en passant `enforce_*` + `_use_seasonal_hourly` explicitement aux constructeurs.

### Integration Points
- `assembler.build()` est le point d'entrée unique pour les 24×4 = 96 PFC builds. Pas de modification core ; le harness scorecard appelle `PFCAssembler(sh, si, unc, **flag_config).build(reference_date=vintage)` 96 fois.
- `tests/conftest.py` autouse env hygiene : Phase 10 ajoute `PFC_LT_ALLOW_NEGATIVE_PRICES` et `PFC_LT_USE_SEASONAL_HOURLY_SHAPE` au snapshot/restore par test (~1 ligne, déjà fait Phase 5 si non).
- `pfc_shaping/validation/` reçoit 5 nouveaux modules (`block_masks.py`, `structural_tests.py`, `scorecard.py`, `christoffersen.py`, `dm_test.py`). Existing `backtest.py` et `compare_hfc.py` conservés inchangés.

</code_context>

<specifics>
## Specific Ideas

### Vision user verbatim 2026-05-20 (capture pour downstream agents)
> *"Moi je pense qu'il faut oublier Ompex, il faut se focaliser sur la qualité absolue de notre HFC ou PFC, un plan comme ça, et en oublier le reste, c'est vraiment notre mission, c'est de faire la PFC pour FMV, state of the art la plus efficace et la plus aboutie possible."*
>
> *"Comment font les autres? Qu'est-ce qui se fait sur le marché entre guillemets? Comment on fait les PFL? Comment on font toutes les recherches? On va essayer de se baser là-dessus, puis répliquer ça."*

**Interprétation** : Phase 10 est l'argument SOTA-by-construction de la PFC FMV. La comparaison externe (OMPEX, autres vendors) est valeur ajoutée mais pas nécessaire pour démontrer la qualité — la literature et les critères structurels suffisent. Cohérent avec EPFL principle "build it right by design, validate against principle, not just against competitor".

### Pourquoi 5 piliers (pas 3, pas 4)
- **3 piliers** auraient été tentants (Structural + Empirical + DM) — minimaliste, mais skip Christoffersen rate l'aspect probabiliste critique pour un trader desk qui pricer un IC pour un client.
- **4 piliers** sans peer review = scorecard mais pas d'argument vs literature → manque le "comment on se positionne face à la SOTA mondiale" qui est exactement la question business.
- **5 piliers** : le scorecard couvre les 4 dimensions techniques (structurelle / empirique / probabiliste / statistique vs baselines) + la dimension méta (peer review SOTA literature). Réplique le format d'un chapitre "validation" d'un working paper EPFL/IEEE.

### Convention quant "scorecard public-FMV-grade"
Le `10-VERIFICATION.md` doit être lisible par un trader senior FMV sans contexte gsd. Sections claires, executive summary 1 page en tête, tables markdown standalone, figures embedded inline. C'est le livrable "argument interne décisif" de la phase, le PROJECT.md le cite par référence post-merge.

### Pourquoi SC#1 unique gate (et pas SC#1+SC#2+SC#3+SC#4)
User explicit decision 2026-05-20 : SC#1 Hildmann 4/4 PASS suffit. Justification : qualité absolue se mesure structurellement (cohérence interne), pas statistiquement (DM vs baselines) ni comparativement (vs OMPEX). Les autres piliers produisent du **content informational** pour le scorecard mais ne sont pas des gates. Évite le piège "fail-on-DM-because-baseline-was-too-good-by-chance" sur un sample 24 vintages.

### Méthode scientifique (innovation EPFL angle, suite 5bis-A/B/5)
- **Replication SOTA literature first, comparison externe later** : on construit le scorecard EPFL-grade par construction. La comparaison OMPEX (Phase 10B) ajoute une couche externe quand elle sera mesurable.
- **Ablation grid 2x2 isole les contributions** : on voit séparément la value-add de 5bis-B (bowl) et de Phase 5 (negative prices). Crédibilité scientifique du discours D-FLIP-1.
- **SC#1 structural-only = "qualité interne" gate** : aligne avec le principe Hildmann 2013 "a good HPFC has 4 internal properties". Pas de dépendance bench externe.

</specifics>

<deferred>
## Deferred Ideas

### Phase 10B (NEW deferred phase) — PFC FMV vs HFC OMPEX comparative benchmark
- Exécuté depuis poste FMV avec accès H:\ (chemin `H:\Energy\GeCom\MARCHE & NEGOCE\Prix\Analyse HFC\HFC test\ER -HFC_OMPEX_15min`, config.yaml:71).
- KPI : Δ MAE bloc PFC FMV vs HFC OMPEX ≤ -1.5 €/MWh sur ≥ 3 blocs (= ex-BT-05 préservé pour 10B).
- DM test PFC FMV vs OMPEX (placement vide dans Pillar 4 Phase 10, à remplir 10B).
- Réutilise toute l'infra Phase 10 (block_masks, scorecard harness, structural_tests, etc.) — c'est purement un ajout de baseline + un slot column dans la table comparative.
- À insérer dans ROADMAP comme nouvelle phase entre 10 et 11 (ou 10.5). Title : "PFC FMV vs HFC OMPEX block-MAE benchmark (deferred, requires FMV poste)".

### Phase 5ter (déjà roadmap'd) hérite de l'infra Phase 10
- `pfc_block_distribution(start, end, hours_mask) → (p10, p50, p90)` Monte-Carlo shape (DIST-01..03).
- Christoffersen conditional coverage (Markov independence) + reliability diagrams par bloc → ajout naturel quand 5ter livrera la distribution probabiliste full.
- 5ter consume directement les block_masks + scorecard harness Phase 10.

### Vers ROADMAP/REQUIREMENTS updates (à fixup pré-planning ou Plan 10-01)
- ROADMAP §Phase 10 title + goal refondre : "Backtest par bloc vs HFC OMPEX" → "PFC FMV Quality Scorecard (5-pillar SOTA replication)".
- ROADMAP insérer Phase 10B (deferred) : "PFC FMV vs HFC OMPEX block-MAE benchmark (requires FMV poste H:\)".
- REQUIREMENTS BT-01..BT-05 reformulés :
  - BT-01 : harness backtest accepte block schema {hours, dow, months} — préservé tel quel.
  - BT-02 : 5 blocs 2024-2025 → preservé mais blocs renommés selon D-A2-2.
  - BT-03 (HFC OMPEX) → renommé BT-10B-01, déplacé vers Phase 10B group.
  - BT-04 (DM test markdown) → reformulé "DM test vs 3 naive baselines (climatology, persistence Y-1, forwards-flat)".
  - BT-05 (Δ MAE ≤ -1.5 vs OMPEX) → renommé BT-10B-02, déplacé vers Phase 10B.
  - Nouveaux BT-06..BT-10 pour les 5 piliers du scorecard Phase 10.
- PROJECT.md Key Decisions ajout 2026-05-20 entrée + update D-FLIP-1 wording (gate = SC#1 Hildmann 4/4 PASS Phase 10).

### Vers visualisations HTML interactives (post-merge si demande management)
- Plotly figures interactives + standalone HTML scorecard pour présentation interne. Pas en scope initial (markdown + PNG suffisent), mais easy add-on si demande.

### Vers daily walk-forward (vs monthly)
- 504 business days 2024-2025 × 4 configs = 2016 builds. ~16h compute Mac Mini. Pas de signal additionnel significatif vs monthly. Si demande post-merge → ajout simple en augmentant la liste `vintages` dans `scorecard.py`.

### Vers multi-market scorecard (FR, DE, AT, IT)
- Réutilise toute l'infra Phase 10 quand Phase 3 (FR/AT/IT activation, HOLD actuellement) sera dé-prioritisée. Block masks devraient être market-aware (jours fériés, tz).

### Vers benchmarking vs forecast institutionnel (LEAR / Chronos-2 short-term)
- Phase 10 mesure la PFC LT (3 ans forward). LEAR/Chronos-2 sont court-terme (D+1..D+10). Pas la même horizon → pas comparable directement. Si demande de benchmark "LT vs CT convergence" sur l'horizon overlap (M+1..M+3) → nouvelle phase dédiée.

### Vers reformulation Mincer-Zarnowitz unbiasedness test (Pillar 2 enhancement)
- Ajouter test joint `α=0 & β=1` Wald comme métrique unbiasedness — déjà inclus en D-A2-4 mais validation au planning si besoin de plus de granularité (e.g. per-horizon MZ regression vs pooled).

</deferred>

---

*Phase: 10 — PFC FMV Quality Scorecard (refondu 2026-05-20, OMPEX deferred to Phase 10B)*
*Context gathered: 2026-05-20*
*Convention quant : replication SOTA literature first (Hildmann + Benth-Koekebakker + Christoffersen + DM + peer review), comparison externe deferred. SC#1 Hildmann 4/4 PASS unique gate. Cohérent avec EPFL principle "qualité absolue, pas comparative".*
