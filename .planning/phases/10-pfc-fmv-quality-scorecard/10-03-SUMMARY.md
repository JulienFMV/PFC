---
phase: 10-pfc-fmv-quality-scorecard
plan: 03
subsystem: validation
tags: [phase10, scorecard, pillar3, pillar4, christoffersen, diebold-mariano, ic80, blocker4-resolved, autonomous-wave3]

# Dependency graph
requires:
  - phase: 10-pfc-fmv-quality-scorecard
    plan: 02
    provides: pfc_shaping.validation.scorecard (mz_test + compute_cell_kpis + run_scorecard_pillar_1) ; pfc_shaping.validation.structural_tests (4 Hildmann) ; statsmodels==0.14.6 + matplotlib>=3.7.0 pre-installed
  - phase: 10-pfc-fmv-quality-scorecard
    plan: 01
    provides: pfc_shaping.validation.block_masks (ALL_BLOCKS, BlockMask base class) ; ABLATION_GRID + build_one + FORWARDS_SOURCE_* constants
provides:
  - pfc_shaping/validation/christoffersen.py — Christoffersen 1998 unconditional coverage LR_uc (scipy.stats.chi2 closed-form, IC80 only)
  - pfc_shaping/validation/dm_test.py — diebold_mariano (statsmodels.tsa.stattools.acovf + Bartlett HAC + HLN small-sample correction) + 3 naive baselines maison (climatology, persistence_y1, forwards_flat)
  - pfc_shaping/validation/scorecard.py extended — compute_pillar3_coverage (IC80 only, ValueError on ic_level=0.95) + compute_pillar4_dm (13-key cell KPI dict)
  - tests/test_phase10_probabilistic.py — 10 tests Pillar 3 (5 unit LR_uc + 3 compute_pillar3 + 1 missing_cols + 1 integration ; incl. test_compute_pillar3_coverage_rejects_ic95)
  - tests/test_phase10_dm.py — 15 tests Pillar 4 (8 DM unit + 4 baselines + 2 compute_pillar4 + 1 imports)
affects: [10-04, 5ter]

# Tech tracking
tech-stack:
  added:
    - "scipy.stats.chi2 (Christoffersen LR_uc closed-form p-value chi2.sf(lr_stat, df=1))"
    - "scipy.stats.t (HLN small-sample correction Student-t df=n-1 vs N(0,1))"
    - "statsmodels.tsa.stattools.acovf (DM HAC long-run variance via Bartlett-weighted lag-k autocovariances)"
  patterns:
    - "Christoffersen LR_uc maison closed-form ~10 lignes : log-formulation pour stabilité numérique (log_lik_null - log_lik_alt) ; degenerate guards (n==0 OR x∈{0,n}) pour éviter log(0)/div-by-zero — pas de MCMC ni d'optim"
    - "Diebold-Mariano maison ~30 lignes : Bartlett weights `1 - k/(n_lags+1)` (positive-semidefinite par construction vs uniform DM 1995 §3.1 qui peut donner var_d<0) ; HLN adj = sqrt((n+1-2h+h(h-1)/n)/n) + Student-t df=n-1 ; fallback gammas[0] sur var_d≤0 + degenerate=True si encore négatif"
    - "Blocker #4 fix pattern : ValueError explicite avec message verbatim mentionnant `IC95`+`Phase 5ter`+`uncertainty.py` (anti-silent-skip audit-trail). Test pytest dédié `test_compute_pillar3_coverage_rejects_ic95` assert le raise."
    - "TDD strict (RED commit AVANT GREEN commit) sur les 2 tasks : 8fd4f65 RED + 0c7509a GREEN Task 1 (Pillar 3) ; 6f6c0ae RED + 0088c05 GREEN Task 2 (Pillar 4)."
    - "Zéro dep PyPI risquée : ni `arch` (5+ Mo + abandonware risk), ni `dieboldmariano==0.1.x` (slop risk RESEARCH §Don't Hand-Roll). Implémentation maison auditable line-by-line via grep."

key-files:
  created:
    - pfc_shaping/validation/christoffersen.py (121 lignes — lr_unconditional_coverage + degenerate guards + docstring complète)
    - pfc_shaping/validation/dm_test.py (420 lignes — diebold_mariano + baseline_climatology + baseline_persistence_y1 + baseline_forwards_flat + edge cases handlers)
    - tests/test_phase10_probabilistic.py (272 lignes — 10 tests Pillar 3 verts)
    - tests/test_phase10_dm.py (330 lignes — 15 tests Pillar 4 verts)
  modified:
    - pfc_shaping/validation/scorecard.py (+277 lignes : compute_pillar3_coverage + compute_pillar4_dm + 2 __all__ exports ajoutés)

key-decisions:
  - "IC80 only en Phase 10 — IC95 explicitement déférée Phase 5ter (blocker #4 résolu). Garde ValueError verbatim mentionnant `IC95`+`Phase 5ter`+`uncertainty.py` dans compute_pillar3_coverage. Aucun silent skip, aucun fallback ambigu. La classe Uncertainty (pfc_shaping/lt/model/uncertainty.py:51-194) expose UNIQUEMENT p10/p90 (percentiles 10/90 = IC80) sans paramètre level= ; étendre = scope-creep refusé."
  - "Bartlett weights (positive-semidefinite par construction) au lieu d'uniform weights DM 1995 §3.1 — Pitfall 3 RESEARCH (uniform weights peuvent produire var_d négatif sur petits échantillons avec autocov négatives dominantes ; Bartlett = R `forecast::dm.test` default)."
  - "HLN small-sample correction par défaut (hln_correction=True) — adj = sqrt((n+1-2h+h(h-1)/n)/n) + Student-t df=n-1. Recommandé par real-statistics.com pour n<50 et h>1. Notre cas Phase 10 : n=24 vintages × bloc-len, h jusqu'à 24 mois → small-sample régime."
  - "Convention sign Pillar 4 : `errors_a = realised - pred_pfc` (PFC), `errors_b = realised - pred_baseline` (baseline). `mean_d < 0` → PFC meilleur (loss différentiel négatif). `better_than_baseline = 'Y'` ssi `mean_d < 0 AND p_value < 0.05` (test bilatéral mais sens du mean_d encodé séparément)."
  - "Test integration Pillar 3 (TestPillar3Integration) court-circuite build_one+Uncertainty pour éviter coût CI (~60s+) ; construit pfc_with_ic synth directement et teste l'integration compute_pillar3_coverage. Le full integration avec n_boot=500 Uncertainty est réservé Plan 10-04 real-run (Pitfall 7 compute budget)."

patterns-established:
  - "Implémentations maison ~30-100 lignes auditables line-by-line, préférées aux deps PyPI risquées (arch, dieboldmariano). Pattern : 1 docstring complet (formule + interpretation + sources + edge cases), 1 fonction pure (in numpy out dict), 1 fichier dédié par pillar, exports __all__ explicites."
  - "Explicit guard ValueError pattern pour scope-creep refus : message verbatim mentionnant (a) la valeur rejetée, (b) le scope où sera ré-évalué, (c) le fichier à étendre. Test pytest dédié assert le raise + scan substrings (anti-silent-skip audit-trail)."
  - "Edge case dict return pattern : NaN p_value + degenerate=True flag dans tous les outputs au lieu de crash ou silent NaN propagation. Caller voit explicitement la dégénérescence et décide en aval (e.g. scorecard cellule marquée degenerate=True)."

requirements-completed: [BT-04, BT-08, BT-09]

# Metrics
duration: ~35 min
completed: 2026-05-21
---

# Phase 10 Plan 03: PFC FMV Quality Scorecard — Pillars 3 + 4 Wired Summary

**Pillar 3 (Christoffersen IC80 unconditional coverage) + Pillar 4 (Diebold-Mariano vs 3 naive baselines) livrés via implémentations maison auditables (~150 LOC core, 0 dep PyPI risquée). 25 tests nouveaux verts, 0 régression suite. Blocker #4 (IC95 silent skip) résolu via guard ValueError explicite + test pytest dédié.**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-05-21T06:55:00Z
- **Completed:** 2026-05-21T07:13:45Z
- **Tasks executed:** 2 of 2 (autonomous wave, no checkpoints)
- **Files created:** 4 (christoffersen.py, dm_test.py, test_phase10_probabilistic.py, test_phase10_dm.py)
- **Files modified:** 1 (scorecard.py +277 lignes)
- **Commits created:** 4 (1 RED + 1 GREEN par task, TDD strict)
- **Test delta:** 323 → 348 collected (+25 new : 10 Pillar 3 + 15 Pillar 4), 2 skipped préservés, 0 régression
- **Suite runtime:** 47s full (vs 46s baseline ; +1s pour les 2 nouvelles suites combinées)

## Accomplishments

- **Pillar 3 (probabilistic calibration IC80) livré end-to-end** : `christoffersen.py` (121 lignes) expose `lr_unconditional_coverage(x, n, p)` per RESEARCH §Pattern 3 canonical (scipy.stats.chi2 closed-form, log-formulation pour stabilité numérique, degenerate guards explicites n==0 et x∈{0,n}). Le scorecard expose `compute_pillar3_coverage(pfc_with_ic, realised, bloc, ic_level=0.80)` avec **garde ValueError explicite sur `ic_level=0.95`** (blocker #4 résolu).
- **Pillar 4 (forecast accuracy comparison) livré end-to-end** : `dm_test.py` (420 lignes) expose `diebold_mariano(errors_a, errors_b, h, loss, hln_correction)` ~30 lignes maison via `statsmodels.tsa.stattools.acovf` + Bartlett HAC + HLN small-sample correction + Student-t df=n-1. Les 3 baselines naïves (`baseline_climatology`, `baseline_persistence_y1`, `baseline_forwards_flat`) sont implémentées comme fonctions deterministes (`vintage_date`, `bloc`, `horizon_label`) → float. Le scorecard expose `compute_pillar4_dm(pred_pfc, pred_baseline, realised, bloc, h_months)` retournant un dict 13 keys (DM + MAE PFC/baseline + delta_mae + `better_than_baseline` Y/N).
- **25 tests neufs verts** : `tests/test_phase10_probabilistic.py` (10 tests : 5 LR_uc unit + 3 compute_pillar3 + 1 missing-cols + 1 integration end-to-end) et `tests/test_phase10_dm.py` (15 tests : 8 DM unit incluant edge cases + 4 baselines + 2 compute_pillar4 + 1 imports).
- **Blocker #4 résolu avec audit-trail** : `compute_pillar3_coverage(ic_level=0.95)` raise `ValueError` avec message verbatim contenant `"IC95"`, `"Phase 5ter"`, et `"uncertainty.py"`. Le test `TestComputePillar3Coverage::test_compute_pillar3_coverage_rejects_ic95` assert chaque substring → aucun silent skip possible. La grep audit `grep -E "IC95.*Phase 5ter|ic_level.*0\.95.*ValueError" scorecard.py` retourne ≥1 hit.
- **Zéro dep externe ajoutée** : ni `arch` (5+ MB, abandonware risk), ni `dieboldmariano==0.1.x` (slop risk). Implémentation maison ~150 LOC core auditable line-by-line.

## Verdict Pillar 3 sanity (perfect coverage)

| Cas                            | Input          | Output                                                | Verdict           |
|--------------------------------|----------------|-------------------------------------------------------|-------------------|
| Perfect coverage               | x=20, n=100, p=0.20  | lr_stat=0.0, p_value=1.0                       | H0 NOT rejected ✓ |
| Undercoverage                  | x=5,  n=100, p=0.20  | lr_stat>5, p_value<0.05                        | H0 rejected ✓     |
| Overcoverage                   | x=40, n=100, p=0.20  | lr_stat>5, p_value<0.05                        | H0 rejected ✓     |
| Degenerate n=0                 | x=0, n=0             | degenerate=True, p=NaN, no crash               | OK ✓              |
| Degenerate x=0 / x=n           | x=0 / x=100, n=100   | degenerate=True, p=NaN, no log(0) crash        | OK ✓              |
| **IC95 reject (blocker #4)**   | ic_level=0.95        | **ValueError verbatim "IC95 + Phase 5ter + uncertainty.py"** | **Audit-trail ✓** |

**Verdict canonique** : `lr_unconditional_coverage(x=20, n=100, p=0.20)` → `p_value = 1.0 > 0.9` ✓ (sanity perfect coverage du plan §verify).

## Verdict Pillar 4 sanity (A clearly better)

| Cas                           | Input                                          | Output                                       | Verdict           |
|-------------------------------|------------------------------------------------|----------------------------------------------|-------------------|
| A clearly better              | errors_a ~ N(0, 0.1), errors_b ~ N(0, 2.0), n=50, h=1 | mean_d<0, dm_stat<-2, p<0.05            | H0 rejected ✓     |
| Equal accuracy                | errors_a, errors_b ~ N(0, 1.0) iid, n=100, h=1 | mean_d≈0, |dm_stat|<2.5, p>0.05              | H0 NOT rejected ✓ |
| n too few (degenerate)        | n=3 < max(h+1, 8)                              | degenerate=True, p=NaN, no crash             | OK ✓              |
| h=24 lag=23 (Y+2 horizon)     | n=100, h=24                                    | n_lags_hac=23, no crash, p valid             | OK ✓              |
| var_d<0 fallback              | oscillating loss [-1,+1] forcé HAC<0           | fallback gammas[0] OR degenerate=True        | OK ✓              |
| MSE loss alternative          | A better, loss='mse'                           | mean_d<0, p<0.05                             | OK ✓              |
| Loss invalid                  | loss='logloss'                                 | ValueError                                   | OK ✓              |
| Length mismatch               | len(a)=20, len(b)=15                           | ValueError                                   | OK ✓              |

**Verdict canonique** : `diebold_mariano(np.array([0.1]*20), np.array([2.0]*20), h=1)` → `p_value ≈ 3.5e-304 << 0.05` ✓ (A clearly better du plan §verify).

## Verdict IC95 explicit defer (blocker #4 audit-trail)

- **Test pytest dédié** : `tests/test_phase10_probabilistic.py::TestComputePillar3Coverage::test_compute_pillar3_coverage_rejects_ic95` PASS (1/1 en 0.40s).
- **ValueError message verbatim** :
  ```
  compute_pillar3_coverage: ic_level=0.95 not supported.
  Phase 10 tests IC80 only (Uncertainty.compute returns p10/p90, no IC95 bounds).
  IC95 deferred to Phase 5ter (CONTEXT D-A3-3 amendé).
  To enable IC95: extend pfc_shaping/lt/model/uncertainty.py to support level= param,
  then re-open this guard.
  ```
- **3 substrings asserted** : "IC95" ✓ + "Phase 5ter" ✓ + "uncertainty.py" ✓.
- **Grep audit-trail** : `grep -E "IC95.*Phase 5ter|ic_level.*0\.95.*ValueError|uncertainty\.py" scorecard.py` → 5 hits (docstring + impl + message + audit).
- **Scope-creep refus documenté** : `pfc_shaping/lt/model/uncertainty.py` lignes 51-194 exposent UNIQUEMENT `compute() → DataFrame['p10', 'p90']`. Pas de paramètre `level=` ni `confidence=`. Étendre = toucher le code core LT model. Décision : IC95 attendra Phase 5ter (CONTEXT D-A3-3 amendé) quand `pfc_block_distribution` sera shipped.

## Task Commits

Each task was committed atomically via TDD strict (RED commit AVANT GREEN commit) :

1. **Task 1 RED — failing tests Pillar 3 Christoffersen + IC95 reject** : `8fd4f65` (test)
2. **Task 1 GREEN — wire Pillar 3 Christoffersen LR_uc + compute_pillar3_coverage (IC80 only)** : `0c7509a` (feat)
3. **Task 2 RED — failing tests Pillar 4 DM + 3 baselines** : `6f6c0ae` (test)
4. **Task 2 GREEN — wire Pillar 4 Diebold-Mariano + 3 baselines + compute_pillar4_dm** : `0088c05` (feat)

**Plan metadata commit:** _handled post-wave by orchestrator (Wave 3 complete)._

## Files Created/Modified

### Created

- `pfc_shaping/validation/christoffersen.py` (121 lignes) — `lr_unconditional_coverage(x, n, p) -> dict` per RESEARCH §Pattern 3 canonical, scipy.stats.chi2 closed-form (log_lik_null - log_lik_alt formula), degenerate guards explicites (n==0 OR x∈{0, n} → p=NaN, no log(0) crash). Docstring complète avec sources + interpretation + degenerate cases.
- `pfc_shaping/validation/dm_test.py` (420 lignes) — `diebold_mariano(errors_a, errors_b, h, loss, hln_correction) -> dict` per RESEARCH §Pattern 4 canonical (~30 lignes core), 3 baselines maison déterministes (`baseline_climatology`, `baseline_persistence_y1`, `baseline_forwards_flat`), edge cases handlers (n<max(h+1,8) → degenerate, var_d≤0 Bartlett → fallback gammas[0] + log.warning, var_d≤0 lag=0 → degenerate, loss invalid / length mismatch → ValueError).
- `tests/test_phase10_probabilistic.py` (272 lignes, 10 tests) — TestLrUnconditionalCoverage (5 tests : perfect/under/over coverage + 2 degenerate) + TestComputePillar3Coverage (3 tests : synth integration + missing_cols + **test_compute_pillar3_coverage_rejects_ic95** blocker #4 fix) + TestPillar3Integration (1 test end-to-end via _synth_epex_hist_for_mock) + 1 module imports smoke.
- `tests/test_phase10_dm.py` (330 lignes, 15 tests) — TestDieboldMariano (8 tests : A clearly better / equal / n too few / h=24 lag=23 / var_d<0 / MSE loss / loss invalid / length mismatch) + TestBaselines (4 tests : climatology mean / persistence_y1 window / persistence_y1 empty→NaN / forwards_flat M+1+M+3+Y+1+Y+2 + missing→NaN) + TestComputePillar4Dm (2 tests : end_to_end_synth + pred_baseline_scalar_broadcasts) + 1 module imports smoke.

### Modified

- `pfc_shaping/validation/scorecard.py` (+277 lignes net) — nouvelles fonctions `compute_pillar3_coverage(pfc_with_ic, realised, bloc, ic_level=0.80)` (avec garde ValueError IC95 + missing cols + empty alignment handling) et `compute_pillar4_dm(pred_pfc, pred_baseline, realised, bloc, h_months)` (avec pred_baseline scalar broadcast + 13-key dict output incl. better_than_baseline Y/N). 2 nouveaux exports `__all__`.

## Decisions Made

- **IC80 only en Phase 10, IC95 explicitement déférée Phase 5ter** — décision actée par blocker #4 résolution (Plan 10-03 frontmatter `notes.ic95_deferral`). La classe `Uncertainty` ne supporte pas natively IC95 (pas de paramètre `level=`/`confidence=`, percentiles 10/90 hardcoded lignes 110/116/119). Étendre = scope-creep refusé. La garde `ValueError` explicite dans `compute_pillar3_coverage` empêche toute ambiguïté future.
- **Bartlett weights pour HAC variance** — choix `1 - k/(n_lags+1)` au lieu d'uniform weights DM 1995 §3.1. Justification : positive-semidefinite par construction (cf. RESEARCH §Pattern 4 ligne 465 : "Bartlett-like weights `(1 - k/(h+1))` ensure positive semi-definite variance estimator"). R `forecast::dm.test` default identique. Évite var_d<0 dans 99% des cas.
- **HLN small-sample correction par défaut** (`hln_correction=True`) — adj = sqrt((n+1-2h+h(h-1)/n)/n) + Student-t df=n-1 au lieu de N(0,1). Justification : real-statistics.com recommandation pour n<50 et h>1. Cas Phase 10 : n=24 vintages × bloc-len, h jusqu'à 24 → small-sample régime confirmé.
- **Convention sign Pillar 4** : `errors_a = realised - pred_pfc` (PFC), `errors_b = realised - pred_baseline` (baseline). `mean_d < 0` → PFC meilleur (loss différentiel négatif). `better_than_baseline = 'Y'` ssi `mean_d < 0 AND p_value < 0.05` (test bilatéral mais le sens du mean_d encodé séparément, plus interprétable downstream que juste p<0.05).
- **Integration Pillar 3 court-circuite build_one+Uncertainty** — TestPillar3Integration construit pfc_with_ic synth directement au lieu de chainer Uncertainty(n_boot=50, seed=42) + build_one(with_uncertainty=True). Justification : Pitfall 7 RESEARCH compute budget (24 builds avec Uncertainty ≈ 36-50 min Mac Mini, même n_boot=50 réduit ≈ 5-10 min CI). Le full integration n_boot=500 est réservé Plan 10-04 real-run sur Config 4 SEULEMENT (production target).
- **pred_baseline scalar broadcast dans compute_pillar4_dm** — quand `baseline_climatology` retourne un float scalar, le wiring broadcast vers `pd.Series(float_val, index=realised.index)` automatiquement. Évite au caller de manipuler manuellement le broadcast pour les baselines scalaires (climatology) vs Series (forwards_flat indirect via time-varying lookup).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 — Bug] Float ULP comparison `0.80 == pytest.approx(0.20)` au lieu de `==`**

- **Found during:** Task 1 GREEN (premier essai test_synth_pfc_realised_in_bounds)
- **Issue:** `1.0 - 0.80 = 0.19999999999999996` (float ULP) → `assert res["nominal_p"] == 0.20` fail directement avec `0.19999999999999996 == 0.2`.
- **Fix:** Remplacer `res["nominal_p"] == 0.20` par `res["nominal_p"] == pytest.approx(0.20)` (et idem pour ic_level=0.80).
- **Files modified:** `tests/test_phase10_probabilistic.py::TestComputePillar3Coverage::test_synth_pfc_realised_in_bounds` (2 lignes)
- **Verification:** Test PASS après tweak (1.0-0.80 floating-point ULP documenté in-test).
- **Committed in:** `0c7509a` (Task 1 GREEN, scope du même commit)

---

**Total deviations:** 1 auto-fixed (1× Rule 1 — Bug ; 0 Rule 2 ; 0 Rule 3 ; 0 Rule 4)

**Impact on plan:** Pure correction technique (float comparison hygiene). Aucun scope creep, aucune décision architecturale. Le critère sémantique du test est préservé (nominal_p doit valoir 0.20, à l'ULP près).

Aucune deviation Rule 4 (architectural) n'a été déclenchée. Aucun blocker rencontré. Aucun checkpoint atteint (plan autonomous=true).

## Issues Encountered

- **Aucun**. Les 2 tasks ont passé verts du premier coup après le tweak ULP cosmétique (Task 1) et directement (Task 2). Pas de warning statsmodels nouveau ni de NaN propagation silencieux observé.
- **Mention budget integration Pillar 3** : le test `TestPillar3Integration::test_end_to_end_uncertainty_compute_pillar3` court-circuite intentionnellement build_one+Uncertainty(n_boot=50) pour rester sous 60s CI. Le full integration avec `Uncertainty(n_boot=500)` sur 4 configs × 24 vintages reste réservé Plan 10-04 real-run (estimation Pitfall 7 RESEARCH : ~36-50 min Mac Mini).

## Known Stubs

**Aucun stub silencieux non-documenté.** Toutes les fonctions retournent soit un dict complet (incluant `degenerate: True` flag explicite), soit raise ValueError explicite (IC95 reject, missing columns, loss invalid, length mismatch). Pas de placeholder, pas de TODO, pas de NotImplementedError résiduel sur les fonctions du scope Plan 10-03.

## Threat Flags

Pas de nouvelle surface de sécurité non-couverte par le threat model PLAN.md :

- **T-10-03-01 (Tampering DM impl maison vs PyPI alt)** : Maison ~30 lignes auditable line-by-line ; grep `from arch|from dieboldmariano` dm_test.py → 0 hits ✓
- **T-10-03-02 (Information disclosure NaN propagation)** : Degenerate guards explicites (n==0, x∈{0,n}, n<max(h+1,8), var_d≤0) + flag `degenerate: True` dans tous outputs ; grep `degenerate` dm_test.py = 10 hits, christoffersen.py = 7 hits ✓
- **T-10-03-03 (Repudiation DM lag choice h-1 not traced)** : Docstring `diebold_mariano` cite DM 1995 §3.1 + Bartlett = R `forecast::dm.test` default ; n_lags_hac inclus dans output dict ; test `test_h_24_lag_23_no_crash` assert `n_lags_hac == 23` ✓
- **T-10-03-04 (DoS Pillar 3 integration test timeout)** : Integration test court-circuite build_one+Uncertainty (pas de n_boot=500 en CI) ; runtime 0.40s < 60s budget ✓
- **T-10-03-05 (Repudiation IC95 silent skip — blocker #4 historique)** : Garde ValueError explicite dans `compute_pillar3_coverage` avec message verbatim mentionnant `IC95`+`Phase 5ter`+`uncertainty.py` ; test pytest dédié `test_compute_pillar3_coverage_rejects_ic95` assert le raise + 3 substrings ✓

Aucun threat flag nouveau à signaler. Le grep `grep -E "from arch|from dieboldmariano" pfc_shaping/validation/dm_test.py` retourne 0 hits (sup-chain mitigation effective).

## Pré-autorisation Plan 10-04

Toutes les primitives 5 piliers sont prêtes pour final assembly :

- **Pillar 1 (Hildmann SC#1 gate)** : `pfc_shaping.validation.scorecard.run_scorecard_pillar_1` wired (Plan 10-02) ; `pfc_shaping.validation.structural_tests` 4 fonctions canoniques.
- **Pillar 2 (KYOS empirical accuracy)** : `pfc_shaping.validation.scorecard.mz_test` + `compute_cell_kpis` (Plan 10-02).
- **Pillar 3 (Probabilistic calibration IC80)** : `pfc_shaping.validation.christoffersen.lr_unconditional_coverage` + `compute_pillar3_coverage(ic_level=0.80)` (Plan 10-03) — IC95 explicitement déférée Phase 5ter via ValueError guard.
- **Pillar 4 (Forecast accuracy DM vs 3 baselines)** : `pfc_shaping.validation.dm_test.diebold_mariano` + 3 baselines (`baseline_climatology`, `baseline_persistence_y1`, `baseline_forwards_flat`) + `compute_pillar4_dm` (Plan 10-03).
- **Pillar 5 (Peer-review structural checks)** : déjà couvert par les 4 Hildmann (Pillar 1) — pas de fonction additionnelle requise per CONTEXT.

Plan 10-04 final assembly devra :
1. Orchestrer le 96-grid loop (24 vintages × 4 configs) avec caching parquet per-vintage (déjà en place via `run_scorecard_pillar_1`).
2. Pour Config 4 SEULEMENT, ajouter le path `with_uncertainty=True` (Uncertainty n_boot=500) pour Pillar 3 — Pitfall 7 compute budget ~36-50 min Mac Mini.
3. Aggréger Pillars 1-4 dans le scorecard markdown + figures PNG matplotlib (statsmodels + matplotlib pré-installés Plan 10-01/02).
4. Test SC#1 gate REAL (Pillar 1 sur `forwards_source='real_eex_xlsx'`) bloque sur D-FLIP-1 si Mac Mini default (fallback diagnostic). User override explicit possible si statut diagnostic-only accepté.

## Compute budget mesuré

- **Suite Phase 10 Plan 10-03 isolated** : 25 tests en 0.49s (`pytest tests/test_phase10_probabilistic.py tests/test_phase10_dm.py`).
- **Suite globale** : 348 passed + 2 skipped en 47.69s (+1.0s vs 46.69s baseline Plan 10-02).
- **Integration Pillar 3 court-circuité** : test_end_to_end_uncertainty_compute_pillar3 court-circuite build_one+Uncertainty pour rester sous 1s. Projection full integration avec n_boot=50 (CI safe) ≈ 5-10 min, avec n_boot=500 (prod) ≈ 36-50 min Mac Mini (cf. Pitfall 7 RESEARCH) — réservé Plan 10-04 real-run sur Config 4 SEULEMENT.

## Self-Check: PASSED

Files asserted as created/modified :

- `[FOUND]` pfc_shaping/validation/christoffersen.py (121 lignes)
- `[FOUND]` pfc_shaping/validation/dm_test.py (420 lignes)
- `[FOUND]` tests/test_phase10_probabilistic.py (272 lignes)
- `[FOUND]` tests/test_phase10_dm.py (330 lignes)
- `[FOUND]` pfc_shaping/validation/scorecard.py (modified +277 lignes, 2 nouveaux exports)

Commits asserted in git log :

- `[FOUND]` 8fd4f65 (Task 1 RED test)
- `[FOUND]` 0c7509a (Task 1 GREEN feat)
- `[FOUND]` 6f6c0ae (Task 2 RED test)
- `[FOUND]` 0088c05 (Task 2 GREEN feat)

Test suite : 348 passed + 2 skipped (vs baseline 323+2 — net +25 nouveaux tests, 2 pre-existing skips préservés, 0 régression).

Blocker #4 audit-trail :

- `[FOUND]` grep "IC95.*Phase 5ter|ic_level.*0\.95.*ValueError|uncertainty\.py" scorecard.py → 5 hits
- `[FOUND]` test_compute_pillar3_coverage_rejects_ic95 PASS (substrings "IC95" + "Phase 5ter" + "uncertainty.py" asserted)
- `[FOUND]` grep "from arch|from dieboldmariano" dm_test.py → 0 hits (zero slop risk dep)
- `[FOUND]` grep "acovf" dm_test.py → 2 hits (statsmodels canonical confirmed)
- `[FOUND]` grep "degenerate" dm_test.py = 10 + christoffersen.py = 7 (edge cases handled explicitement)

Acceptance criteria canoniques (du plan §verify) :

- `[PASS]` lr_unconditional_coverage(x=20, n=100, p=0.20) → p_value = 1.0 > 0.9
- `[PASS]` diebold_mariano([0.1]*20, [2.0]*20, h=1) → p_value ≈ 3.5e-304 << 0.05
- `[PASS]` All imports OK : christoffersen.lr_unconditional_coverage + dm_test.{diebold_mariano, baseline_*} + scorecard.{compute_pillar3_coverage, compute_pillar4_dm}

---

*Phase: 10-pfc-fmv-quality-scorecard*
*Plan: 03 — Pillars 3 (Christoffersen IC80 unconditional coverage) + 4 (Diebold-Mariano vs 3 naive baselines) wired end-to-end*
*Completed: 2026-05-21*
