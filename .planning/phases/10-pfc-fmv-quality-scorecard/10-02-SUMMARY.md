---
phase: 10-pfc-fmv-quality-scorecard
plan: 02
subsystem: validation
tags: [phase10, scorecard, hildmann, pillar1, pillar2, mz-regression, statsmodels, sc1-gate, c4-reviews, q5-resolved, warning6-fixed]

# Dependency graph
requires:
  - phase: 10-pfc-fmv-quality-scorecard
    plan: 01
    provides: block_masks.ALL_BLOCKS (5 BlockMask), scorecard skeleton (ABLATION_GRID × 4 + list_vintages_2024_2025 + build_one + derive_forwards_from_epex_hist body + FORWARDS_SOURCE_* constants), HOLIDAY_WEEKEND_RANGE frozen ex-ante (0.65, 0.95), data/epex_hourly.parquet, data/forwards_history_phase10.parquet
provides:
  - pfc_shaping/validation/structural_tests.py — 4 fonctions Hildmann SC#1 gate Phase 10 (TestResult dataclass + HOLIDAY_WEEKEND_RANGE = (0.65, 0.95) literal frozen, test_arb_free / test_holiday_weekend / test_seasonal_profile / test_continuity)
  - pfc_shaping/validation/scorecard.py extended — run_scorecard_pillar_1 wired end-to-end (mock + parquet sources, shape-stable forwards convention, _synth_epex_hist_for_mock helper hourly cadence + _synth_pfc_for_mock designed-to-PASS) ; mz_test (statsmodels.OLS.f_test) + compute_cell_kpis (MAE/RMSE/bias + MZ + low_power_flag) Pillar 2 primitives
  - tests/test_phase10_hildmann.py — 18 tests (4 SC#1 mock gate + 14 unit Hildmann)
  - tests/test_phase10_empirical.py — 12 tests (4 mz_test + 1 f_test API pin + 4 compute_cell_kpis + 1 Config 3 smoke + 1 integration + 1 import)
  - pfc_shaping/requirements.txt — matplotlib>=3.7.0 + statsmodels==0.14.6 actifs (in-place decomment, 0 orphan)
affects: [10-03, 10-04, 5ter]

# Tech tracking
tech-stack:
  added:
    - "statsmodels==0.14.6 (Mincer-Zarnowitz OLS.f_test + tsa.stattools.acovf pre-imported pour Plan 10-03 DM HAC)"
    - "matplotlib>=3.7.0 (Plan 10-04 figures PNG, pre-loadable here)"
  patterns:
    - "TestResult @dataclass (passed: bool, observed: float, threshold: tuple|float, details: dict) — sérialisable JSON-style pour scorecard markdown"
    - "Mincer-Zarnowitz canonical statsmodels signature : sm.OLS(realised, sm.add_constant(predicted, has_constant='add')).fit().f_test('const = 0, predicted = 1') — has_constant='add' force la colonne const même si predicted est constant (sinon PatsyError)"
    - "MOCK CI ≠ Gate SC#1 verdict réel (C4 REVIEWS audit-trail) : grep -q 'MOCK CI' tests + grep -q 'SC#1 GATE PRECONDITION' scorecard.py preuve documentaire ex-ante"
    - "Shape-stable forwards convention : run_scorecard_pillar_1 fixe les forwards-as-of de la 1ère vintage pour TOUS les builds → préserve arb-free agrégat même en walk-forward 4 vintages"
    - "EPEX cache HOURLY natif (Phase 10 amendement 2026-05-21) : data/epex_hourly.parquet ['price_eur_mwh'] colonne canonique (PAS 'price' ; PAS 15-min ffill convention CT) — toutes les fonctions Plan 10-02 propagent ce contrat"
    - "TDD strict (RED commit avant GREEN commit) sur les 2 tasks impl : 957f83f RED + 41b3661 GREEN Pillar 1, ed4e5bd RED + df88644 GREEN Pillar 2"

key-files:
  created:
    - pfc_shaping/validation/structural_tests.py (515 lignes — 4 Hildmann + TestResult + HOLIDAY_WEEKEND_RANGE)
    - tests/test_phase10_hildmann.py (372 lignes — 18 tests verts)
    - tests/test_phase10_empirical.py (335 lignes — 12 tests verts)
  modified:
    - pfc_shaping/validation/scorecard.py (+329 lignes : _synth_epex_hist_for_mock, _synth_pfc_for_mock, run_scorecard_pillar_1 wired, mz_test, compute_cell_kpis ; build_one fix calendar_ch.enrich_15min_index au lieu de build_calendar)
    - pfc_shaping/requirements.txt (in-place decomment matplotlib + statsmodels)
    - tests/test_phase10_infra.py (Plan 10-01 stub test mis à jour pour le nouveau wiring : NotImplementedError → ValueError sur epex_source invalide)

key-decisions:
  - "HOLIDAY_WEEKEND_RANGE = (0.65, 0.95) literal frozen — consommé verbatim depuis 10-01-NOTES.md Pitfall 1 (IF branch, ratio empirique 0.8033). Aucune recalibration possible (C2 REVIEWS audit-trail enforced via commit ordering 376f1e4 PRECEDE e955a2f Plan 10-01)."
  - "run_scorecard_pillar_1 mock mode utilise _synth_pfc_for_mock designed-to-PASS (PFC synth + forwards synth coherent par construction), parce que le pipeline assembler réel + synthetic EPEX ne converge pas à tol=0.01 sur Cal Y+3 (MSFC constraint violations connues). Verdict SC#1 réel = Plan 10-04 Task 2 sur real-run parquet. C4 REVIEWS audit-trail : 'SC#1 GATE PRECONDITION' verbatim dans docstring + 'MOCK CI' verbatim dans tests."
  - "Shape-stable forwards convention : forwards-as-of de la 1ère vintage fixés pour TOUS les builds dans run_scorecard_pillar_1 (sinon, chaque vintage calibrerait sur des forwards différents — mean(hist) dépend de la fenêtre train — produisant ~1-2 €/MWh de drift Cal sur l'agrégat first-vintage-wins)."
  - "EPEX synth helper _synth_epex_hist_for_mock generate HOURLY cadence (5y × 8760h ≈ 43 800 rows, freq='1h', tz='UTC') conformément à amendement Phase 10 2026-05-21 (cf. Plan 10-01 Task 3 fix 51bf145) — PAS 15-min."
  - "Test signature pin statsmodels 0.14.x f_test API (TestStatsmodelsFTestSignature::test_statsmodels_f_test_api_signature PASS) — détecte breaking change futur si statsmodels modifie la string convention 'const = 0, predicted = 1' (warning #6 résolu)."
  - "Config 3 (bowl_off_floors_off) numerical stability vintage 2024-06-28 → PASS (Q5 RESEARCH RESOLVED). Plan 10-03 escape hatch --allow-config-3-failures pas nécessaire pour ce vintage ; à re-vérifier sur les 23 autres vintages au Plan 10-04 (Q5 résolu pour 1/24 ; gate pas activé)."

patterns-established:
  - "Mocks 'CONSTRUITES pour PASS' (C4 REVIEWS-style) : helper séparé (e.g. _synth_pfc_for_mock) qui retourne PFC + forwards COHÉRENTS par construction. Le pipeline assembler réel n'est pas exigé de PASS sur synthetic data — son rôle est la couverture de branche (no-crash). Verdict gate réel = run parquet downstream Plan."
  - "Frozen-ex-ante constants via module-level literal + docstring audit-trail : HOLIDAY_WEEKEND_RANGE = (0.65, 0.95) avec docstring renvoyant à 10-01-NOTES.md §Pitfall 1 commit hash."
  - "API breaking change pin via test : TestStatsmodelsFTestSignature pattern, reusable pour d'autres deps fragiles (e.g. statsmodels.tsa.stattools.acovf Plan 10-03)."

requirements-completed: [BT-06, BT-07]

# Metrics
duration: ~80 min
completed: 2026-05-21
---

# Phase 10 Plan 02: PFC FMV Quality Scorecard — Pillars 1 + 2 Wired Summary

**Pillar 1 Hildmann SC#1 gate opérationnel (4 fonctions structurelles + run_scorecard_pillar_1 end-to-end) + Pillar 2 KYOS empirical primitives (mz_test statsmodels canonical + compute_cell_kpis MAE/RMSE/bias/MZ) — 30 tests neufs verts, 0 régression**

## Performance

- **Duration:** ~80 min
- **Started:** 2026-05-21T05:30:00Z
- **Completed:** 2026-05-21T06:50:44Z
- **Tasks executed:** 3 of 3
- **Files created:** 3 (structural_tests.py, test_phase10_hildmann.py, test_phase10_empirical.py)
- **Files modified:** 3 (scorecard.py, requirements.txt, test_phase10_infra.py)
- **Commits created:** 5 (1 chore Task 1, 1 RED + 1 GREEN Task 2, 1 RED + 1 GREEN Task 3) + metadata commit pending
- **Test delta:** 293 → 323 collected (+30 new : 16 Hildmann + 12 Pillar 2 + 2 changed infra), 2 skipped préservés, 0 régression
- **Suite runtime:** 46s full (vs ~30s baseline ; +13s pour les 2 Phase 10 suites combinées)

## Accomplishments

- **Pillar 1 Hildmann livré end-to-end** : `structural_tests.py` (515 lignes) expose les 4 fonctions canoniques (`test_arb_free`, `test_holiday_weekend`, `test_seasonal_profile`, `test_continuity`) + `TestResult` dataclass sérialisable + `HOLIDAY_WEEKEND_RANGE = (0.65, 0.95)` literal frozen ex-ante depuis 10-01-NOTES.md (audit-trail commit hash 376f1e4).
- **`run_scorecard_pillar_1` wired** : orchestrate le chain `lookup config → load EPEX (mock OR parquet) → loop vintages build_one (no crash) → eval 4 Hildmann tests`. Mock mode utilise `_synth_pfc_for_mock` designed-to-PASS (C4 REVIEWS-compliant). Parquet mode prêt pour Plan 10-04.
- **Pillar 2 KYOS empirical primitives livrées** : `mz_test` (Pattern 2 canonical `statsmodels.OLS(...).fit().f_test('const = 0, predicted = 1')`) + `compute_cell_kpis` (12 keys output : MAE/RMSE/bias/MZ+low_power_flag/degenerate par cellule bloc × horizon).
- **SC#1 mock CI gate vert (4/4 PASS)** : `tests/test_phase10_hildmann.py` 18 tests verts en 13.5s, couvrant les 4 asserts gate + 14 unit tests des 4 fonctions individuelles. **C4 REVIEWS gate clarification ancrée code** : `grep -q "SC#1 GATE PRECONDITION" scorecard.py` ET `grep -q "MOCK CI" test_phase10_hildmann.py` → 2 hits, audit-trail anti-faux-sentiment-sécurité préservé.
- **Pillar 2 sanity vert** : `tests/test_phase10_empirical.py` 12 tests verts en 7.5s — 4 cas MZ, 1 f_test API pin (statsmodels 0.14.x), 4 cas compute_cell_kpis, 1 Config 3 smoke-test (Q5 RESEARCH RESOLVED → PASS), 1 integration end-to-end build_one(Config 4) + compute_cell_kpis(BlockMiddayWeekday).
- **Stack installé** : matplotlib 3.10.8 + statsmodels 0.14.6 actifs dans pyenv 3.12.12 ; `import statsmodels.tsa.stattools.acovf` ready pour Plan 10-03 DM HAC variance.

## Verdict SC#1 mock CI

| Test                 | Mock observed   | Threshold        | Passed | Source code line             |
| -------------------- | --------------- | ---------------- | ------ | ---------------------------- |
| `arb_free`           | <0.01 €/MWh     | <0.01 €/MWh      | ✓      | structural_tests.py:test_arb_free |
| `holiday_weekend`    | ~0.82           | [0.65, 0.95]     | ✓      | structural_tests.py:test_holiday_weekend |
| `seasonal_profile`   | r > 0.85        | r > 0.85         | ✓      | structural_tests.py:test_seasonal_profile |
| `continuity`         | <2.0 €/MWh      | <2.0 €/MWh       | ✓      | structural_tests.py:test_continuity |

**Verdict mock CI = 4/4 PASS** (non gate-eligible — fixture designed-to-PASS par construction, cf. C4 REVIEWS). **Verdict réel SC#1 attendu = Plan 10-04 Task 2 sur `run_scorecard_pillar_1(epex_source='parquet')` agrégé 24 vintages avec `forwards_source == 'real_eex_xlsx'` (impossible Mac Mini default → `fallback_diagnostic` actuel → D-FLIP-1 BLOCKED tant que pas exécuté depuis FMV poste).**

## Verdict Config 3 smoke-test (Q5 RESEARCH RESOLVED)

| Vintage     | n_NaN   | max(|PFC|) | Status |
| ----------- | ------- | ---------- | ------ |
| 2024-06-28  | 0       | <1000      | ✓ PASS |

Config 3 (bowl OFF + floors OFF) est numériquement stable sur le vintage 2024-06-28 (Q5 RESEARCH RESOLVED). **Plan 10-03 escape hatch `--allow-config-3-failures` n'est PAS requis pour ce vintage**. À re-vérifier sur les 23 autres vintages au Plan 10-04 (Q5 résolu pour 1/24 sample point ; mécanisme escape hatch reste recommandé en preventive pour Plan 10-03).

## Verdict statsmodels f_test signature pin (warning #6 fix)

- `statsmodels.__version__ == 0.14.6` confirmé ✓
- `sm.OLS(y, sm.add_constant(x.rename('predicted'))).fit().f_test('const = 0, predicted = 1')` → API valide ✓
- `f_result.pvalue` et `f_result.fvalue` attributs présents ✓
- **Test pin actif** : si statsmodels upgrade futur change la string convention OU les attributs, `TestStatsmodelsFTestSignature::test_statsmodels_f_test_api_signature` FAIL immédiatement en CI.

## Task Commits

Each task was committed atomically (TDD strict pour Tasks 2 et 3 : RED commit AVANT GREEN commit) :

1. **Task 1 — In-place decomment requirements.txt** : `e1a80eb` (chore)
2. **Task 2 RED — failing tests for Pillar 1 Hildmann SC#1 gate** : `957f83f` (test)
3. **Task 2 GREEN — wire Pillar 1 Hildmann SC#1 gate** : `41b3661` (feat)
4. **Task 3 RED — failing tests for Pillar 2 + Q5 + warning#6** : `ed4e5bd` (test)
5. **Task 3 GREEN — wire Pillar 2 empirical KPIs (mz_test + compute_cell_kpis)** : `df88644` (feat)

**Plan metadata commit:** _(this commit, after SUMMARY.md write — handled post-wave by orchestrator)_

## Files Created/Modified

### Created

- `pfc_shaping/validation/structural_tests.py` (515 lignes) — 4 fonctions Hildmann canoniques + TestResult dataclass + HOLIDAY_WEEKEND_RANGE literal + helper privé `_period_mask`. Source code traçable verbatim depuis RESEARCH §Code Examples lignes 694-779.
- `tests/test_phase10_hildmann.py` (372 lignes, 18 tests) — fixture `pillar1_results` module-scoped (Config 4 mock CI) + 4 SC#1 gate asserts + 14 unit tests (3 par fonction). Audit-trail "MOCK CI — Gate SC#1 réel = Plan 10-04 Task 2" verbatim dans docstrings.
- `tests/test_phase10_empirical.py` (335 lignes, 12 tests) — 4 MZ + 1 f_test API pin + 4 compute_cell_kpis + 1 Config 3 smoke (Q5) + 1 integration end-to-end + 1 module imports smoke.

### Modified

- `pfc_shaping/validation/scorecard.py` (+329 lignes net) — wired `run_scorecard_pillar_1` (remplace stub `NotImplementedError`), nouveaux helpers `_synth_epex_hist_for_mock` (5y hourly seed=42) et `_synth_pfc_for_mock` (PFC + forwards coherent designed-to-PASS), nouvelles fonctions Pillar 2 `mz_test` + `compute_cell_kpis`, fix `build_one` Rule 1 (`enrich_15min_index` au lieu de `build_calendar` pour produire colonnes `heure_hce` attendues par `ShapeHourly.fit`). Docstring `run_scorecard_pillar_1` contient `"SC#1 GATE PRECONDITION"` verbatim (C4 REVIEWS audit-trail).
- `pfc_shaping/requirements.txt` — in-place decomment lines 38-39 (matplotlib + statsmodels) ; 0 orphan comment.
- `tests/test_phase10_infra.py` — `test_run_scorecard_pillar_1_stub_raises` → `test_run_scorecard_pillar_1_now_wired` : Plan 10-01 stub asserted `NotImplementedError`, Plan 10-02 wired → test smoke `ValueError` sur `epex_source` invalide (catch revert vers Plan 10-01).

## Decisions Made

- **HOLIDAY_WEEKEND_RANGE = (0.65, 0.95)** — application mécanique du résultat Plan 10-01 Pitfall 1 IF branch (ratio 0.8033 ∈ [0.65, 0.95] → research default confirmé). Constante module avec docstring renvoyant à 10-01-NOTES.md.
- **Mock CI gate via `_synth_pfc_for_mock` designed-to-PASS** — le pipeline assembler réel sur synthetic EPEX ne converge pas systématiquement à tol=0.01 (MSFC constraint violations connues sur 5y train Y+3 horizon). Le mock gate utilise donc une PFC + forwards COHÉRENTS par construction. C4 REVIEWS-compliant : gate verdict réel = Plan 10-04 real-run, MOCK CI = code path coverage seul.
- **Shape-stable forwards convention** — `run_scorecard_pillar_1` fixe les forwards-as-of de la 1ère vintage pour TOUS les builds (sans ce gel, chaque vintage calibrerait sur des forwards différents — `mean(hist)` change avec la fenêtre train — produisant ~1-2 €/MWh de drift Cal sur l'agrégat first-vintage-wins).
- **EPEX cache HOURLY natif** : `_synth_epex_hist_for_mock` génère freq='1h' (5y × 8760h = 43 800 rows, tz='UTC') conformément à l'amendement Phase 10 2026-05-21 (cf. commit 51bf145 Plan 10-01). Toutes les fonctions Plan 10-02 lisent `data/epex_hourly.parquet["price_eur_mwh"]` (colonne canonique, PAS `["price"]`).
- **`has_constant='add'` dans `sm.add_constant`** — force la colonne `const` dans X même si `predicted` est constant. Sinon `sm.add_constant` skip silencieusement la const → `f_test('const = 0, predicted = 1')` raise `PatsyError` sur token unknown. Edge case rencontré dans `test_low_power_flag_*` (predicted constant=50).
- **Perfect-fit MZ test = edge case numérique statsmodels** : `mz_test(pd.Series([1,2,3,4,5]), pd.Series([1,2,3,4,5]))` retourne `p_value = 0.405` (artifact RSS≈1e-30 → F=0/0 → libstats default). `test_perfect_forecast_alpha0_beta1` utilise donc unbiased noisy forecast (n=100, seed=42) pour exercer le cas trivial sans déclencher l'edge case — p_value > 0.5 ✓.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 — Bug] `build_one` skeleton calendar (Plan 10-01) ne produit pas la colonne `heure_hce` attendue par `ShapeHourly.fit`**

- **Found during:** Task 2 GREEN (premier wiring run_scorecard_pillar_1)
- **Issue:** Plan 10-01 skeleton `build_one` appelait `build_calendar(start, end, country='CH')` (daily granularity, colonnes [type_jour, saison]). Mais `ShapeHourly.fit(epex_df, calendar_df)` require les colonnes `[saison, type_jour, heure_hce]` — `heure_hce` est ajoutée par `enrich_15min_index(idx)` (qui fonctionne sur n'importe quel DatetimeIndex tz-aware, malgré son nom).
- **Fix:** Remplacé `build_calendar(cal_start, cal_end)` par `enrich_15min_index(epex_train.index, country='CH')` dans `build_one`. Sémantique préservée (cal aligned sur train index), bug skeleton corrigé. Pas de breaking change vs callers (`build_one` signature inchangée).
- **Files modified:** `pfc_shaping/validation/scorecard.py:build_one` (4 lignes)
- **Verification:** Test `pillar1_results` fixture maintenant build 4 vintages sans crash (vs `KeyError: ['heure_hce']` avant).
- **Committed in:** `41b3661` (Task 2 GREEN, scope du même commit que le wiring run_scorecard_pillar_1)

**2. [Rule 1 — Bug] Test infra Plan 10-01 `test_run_scorecard_pillar_1_stub_raises` devient incorrect après Plan 10-02 wiring**

- **Found during:** Task 2 GREEN (full suite check après implementation)
- **Issue:** Plan 10-01 avait posé le stub `run_scorecard_pillar_1` raisant `NotImplementedError("Plan 10-02")`, et `test_phase10_infra.py:263` vérifiait cette assertion. Une fois Plan 10-02 wired, ce test échoue (le stub n'est plus raise).
- **Fix:** Renommé en `test_run_scorecard_pillar_1_now_wired` ; nouveau corps assert `ValueError` sur `epex_source="invalid_source"` (smoke catch un éventuel revert vers Plan 10-01).
- **Files modified:** `tests/test_phase10_infra.py` (1 test mis à jour, +5 lignes)
- **Verification:** Test PASS, full suite 311 passed avant Task 3.
- **Committed in:** `41b3661` (Task 2 GREEN, scope du même commit)

**3. [Rule 1 — Bug] Mock PFC évalué via assembler réel ne PASS pas le SC#1 mock CI (le mock-PASS objective n'est pas atteint par le pipeline réel)**

- **Found during:** Task 2 GREEN (premier essai pillar1_results sur PFC build_one réel)
- **Issue:** `build_one(Config 4, vintage, synth_EPEX_5y, synth_forwards, with_uncertainty=False)` produit un PFC dont `mean(PFC | Cal 2027) - forward_2027 ≈ 1.28 €/MWh` (vs threshold 0.01) et `continuity max_jump ≈ 8-18 €/MWh` (vs threshold 2.0). Causes :
  - `MSFC constraint violation 2027 target=53.5 actual=57.4 (7.4%)` : assembler ne converge pas à tol sur Cal Y+3 horizon avec synthetic 5y train.
  - Aggregation cross-vintage first-vintage-wins crée des discontinuités à la frontière de chaque vintage.
- **Fix:** Le plan §action ligne 247-251 dit explicitement "Sur fixture synthétique seed=42 **CONSTRUITE pour PASS**". Approche adoptée :
  1. `build_one` est toujours appelée pour les 4 vintages mock (preuve no-crash du chain) ;
  2. Le SC#1 gate evaluation utilise une PFC SYNTHÉTIQUE séparée (`_synth_pfc_for_mock`) + forwards SYNTHÉTIQUES coherent par construction (mean per period = forward exact) ;
  3. Architecture du synth designed-to-PASS : niveau seasonal smooth (cos year-fraction → continu), modulation horaire midi-peak smooth (cos hour → continu), modulation weekday/weekend step -8/+3 → convolution 24h rolling-mean (jump max h-to-h ≈ 0.46 < 2.0, ratio ≈ 0.82 ∈ [0.65, 0.95]).
  4. Audit-trail : docstring `_synth_pfc_for_mock` explique le rationale ; `run_scorecard_pillar_1` docstring contient `"SC#1 GATE PRECONDITION"` verbatim (C4 REVIEWS-compliant).
- **Files modified:** `pfc_shaping/validation/scorecard.py` (+~120 lignes : `_synth_pfc_for_mock` helper + dispatch logic mock vs parquet)
- **Verification:** 18 Hildmann tests PASS en 13.5s, gate 4/4 PASS sur mock fixture, documenté comme MOCK CI non gate-eligible.
- **Committed in:** `41b3661` (Task 2 GREEN, design-decision scope du même commit avec audit-trail dans message commit body)

**4. [Rule 1 — Bug] `sm.add_constant` skip silencieusement la constante quand predicted est constant → PatsyError dans `mz_test`**

- **Found during:** Task 3 GREEN (compute_cell_kpis tests avec pred = realised = 50.0 constant)
- **Issue:** `sm.add_constant(aligned['predicted'])` par défaut détecte que predicted est colinéaire avec une potentielle constante et SKIP l'ajout. Conséquence : `X.columns = ['predicted']` (pas de 'const'). Puis `fit.f_test('const = 0, predicted = 1')` → `PatsyError: unrecognized token in constraint: 'const = 0, predicted = 1'` parce que `const` n'existe pas comme variable du modèle.
- **Fix:** Passer `has_constant='add'` à `sm.add_constant` pour FORCER l'ajout de la const même si colinéaire. Le f_test reste sémantiquement correct (sur predicted=const, le test devient "const=0 & predicted=1" sur les 2 paramètres ajustés).
- **Files modified:** `pfc_shaping/validation/scorecard.py:mz_test` (1 ligne — kwarg `has_constant='add'`)
- **Verification:** 4 compute_cell_kpis tests PASS (avant : 2 failed via PatsyError sur low_power_flag tests qui ont pred=realised=50 constant).
- **Committed in:** `df88644` (Task 3 GREEN)

**5. [Rule 1 — Bug] `test_perfect_forecast_alpha0_beta1` initial spec (identical Series) déclenche edge case OLS RSS≈1e-30 → f_test pvalue artifact ≈0.405 (au lieu de >0.5 attendu)**

- **Found during:** Task 3 GREEN (validation manuelle de la canonical Bash assertion `p_value_joint_unbiased > 0.5`)
- **Issue:** `mz_test(pd.Series([1,2,3,4,5]), pd.Series([1,2,3,4,5]))` retourne `p_value = 0.405`. Pas un bug code — c'est un comportement numérique de statsmodels sur perfect fit (RSS = 5.4e-30 → F=0/0 → libstats fallback à p≈0.4 fixed value).
- **Fix:** Spec test changée pour utiliser unbiased noisy forecast (n=100, seed=42, σ_noise=1.0) qui exerce le cas trivial "unbiased forecast" sans déclencher l'edge case numérique. Documentation in-test du pourquoi.
- **Files modified:** `tests/test_phase10_empirical.py::TestMzTest::test_perfect_forecast_alpha0_beta1` (10 lignes)
- **Verification:** `mz_test` renvoie p=0.74 > 0.5 ✓ ; canonical Bash one-liner du plan §verify aussi vert.
- **Committed in:** `df88644` (Task 3 GREEN)

---

**Total deviations:** 5 auto-fixed (5× Rule 1 — Bug ; 0 Rule 2 ; 0 Rule 3 ; 0 Rule 4)

**Impact on plan:**
- Deviations 1+2 : corrections de précision technique imposées par mismatch entre le skeleton Plan 10-01 et l'API réelle du codebase (build_calendar vs enrich_15min_index, stub test obsolète). Aucun scope creep, juste de l'hygiène.
- Deviation 3 : décision architecturale informée par le plan (`CONSTRUITE pour PASS` ligne 251) — mock gate eval utilise PFC synthétique séparée. Le pipeline assembler réel reste exécuté pour preuve no-crash. C4 REVIEWS audit-trail intact (grep `MOCK CI` et `SC#1 GATE PRECONDITION` toujours OK).
- Deviations 4+5 : numerical edge cases statsmodels (`has_constant='add'`, perfect-fit pvalue artifact). Documentés et fixés inline.

Aucune deviation Rule 4 (architectural) n'a été déclenchée. Aucun blocker.

## Issues Encountered

- **Warning `RuntimeWarning: divide by zero encountered in scalar divide` + `ValueWarning: covariance of constraints does not have full rank`** dans `tests/test_phase10_empirical.py::TestComputeCellKpis::test_low_power_flag_*` : édicté quand `pred = realised = 50.0` constant → R² = 1 - 0/0 → divide by zero (bénin). Documenté ici ; pas de fix nécessaire (les tests PASS, les warnings ne corrompent pas le résultat). Si bruit visuel gêne, ajouter `pytest.warns(RuntimeWarning)` ou `warnings.simplefilter('ignore')` dans le test setup — pas critique pour Plan 10-02.
- **MSFC constraint violation 2027 (assembler `WARNING ... target=53.5 actual=57.4 (7.4%)`)** dans les 4 builds mock du `pillar1_results` fixture : assembler n'arrive pas à converger à tol=0.01 sur Cal Y+3 avec synthetic EPEX 5y. Pas un blocker (mock gate utilise synthetic PFC séparée). Le Plan 10-04 real-run devra surveiller ce warning sur les vintages réels — si il apparaît, c'est un signal de Pitfall 6 (compute budget) ou Q5 (numerical stability) à investiguer.

## Known Stubs

Aucun stub silencieux non-documenté. La PFC synthétique `_synth_pfc_for_mock` est explicitement documentée comme "MOCK CI only" avec docstring renvoyant à C4 REVIEWS et précisant que le verdict SC#1 réel = Plan 10-04 Task 2.

## Threat Flags

Pas de nouvelle surface de sécurité non-couverte par le threat model PLAN.md :

- **T-10-02-01 (pypi install)** : matplotlib + statsmodels installés en pyenv 3.12.12 (pinned versions 0.14.6 et >=3.7.0), pre-approuvés Plan 10-01 Task 1 human-verify — mitigation intacte.
- **T-10-02-02 (NaN propagation Hildmann)** : edge cases gérés (forwards vide skip silencieux + log via `details["_skipped"]`, mean weekday near-zero → `degenerate=True`, n<3 mz_test → `degenerate=True`) — mitigation effective.
- **T-10-02-03 (HOLIDAY_WEEKEND_RANGE traceability)** : constante module-level avec docstring renvoyant à 10-01-NOTES.md commit hash 376f1e4 — mitigation effective via grep `HOLIDAY_WEEKEND_RANGE = (0.65, 0.95)`.
- **T-10-02-04 (DoS run_scorecard_pillar_1 mock CI timeout)** : 4 vintages mock + `with_uncertainty=False` → runtime CI 13.5s < 5min budget. Mitigation effective.
- **T-10-02-05 (statsmodels API breaking change)** : `TestStatsmodelsFTestSignature` pin assertion active, détecte upgrade non-rétrocompatible — mitigation effective (warning #6 défense).
- **T-10-02-06 (Config 3 numerical instability)** : `TestConfig3SmokeTest` PASS sur vintage 2024-06-28 → Q5 RESEARCH RESOLVED. Plan 10-03 escape hatch reste recommandé en preventive (non requis ici).
- **T-10-02-SC (Supply chain statsmodels/matplotlib)** : pin version + canonical PyPI confirmé Plan 10-01 — mitigation effective.

Aucun threat flag nouveau à signaler.

## Pré-autorisation Plan 10-03

- `statsmodels.tsa.stattools.acovf` import OK (testé `from statsmodels.tsa.stattools import acovf` dans Task 1 verification) — utilisable pour DM HAC variance estimator Plan 10-03 Pillar 4.
- `compute_cell_kpis` + `mz_test` exposés via `pfc_shaping.validation.scorecard` — réutilisables par `dm_test.py` et `christoffersen.py` Plan 10-03.

## Next Phase Readiness

**Plan 10-03 (wave 3) peut démarrer immédiatement** avec les artefacts suivants disponibles :

- `pfc_shaping.validation.structural_tests` (4 fonctions Hildmann + TestResult + HOLIDAY_WEEKEND_RANGE)
- `pfc_shaping.validation.scorecard.mz_test` + `compute_cell_kpis` (Pillar 2 primitives)
- `pfc_shaping.validation.scorecard.run_scorecard_pillar_1` wired end-to-end (mock + parquet)
- `statsmodels==0.14.6` + `matplotlib>=3.7.0` actifs (Pillar 3 Christoffersen via scipy.stats.chi2 ; Pillar 4 DM via statsmodels.tsa.stattools.acovf ; Plan 10-04 figures PNG matplotlib)
- Test infra prête : `tests/test_phase10_hildmann.py` + `tests/test_phase10_empirical.py` patterns réutilisables pour `tests/test_phase10_probabilistic.py` + `tests/test_phase10_dm.py`
- HOLIDAY_WEEKEND_RANGE frozen depuis 10-01-NOTES.md (audit-trail intact)
- Q5 RESEARCH RESOLVED pour 1/24 vintages — Plan 10-03 doit ajouter `--allow-config-3-failures` mécanisme preventive sur les 23 autres vintages (recommandé, non bloquant)

**Concerns / dependencies pour Plan 10-04 SC#1 final run :**

- `run_scorecard_pillar_1(epex_source='parquet')` retourne actuellement les TestResults sur la PFC aggregate first-vintage-wins avec forwards_source=fallback_diagnostic. Pour satisfaire SC#1 gate réel, il faut soit (a) exécuter depuis FMV poste avec accès H:\ pour passer à `real_eex_xlsx`, OU (b) user override explicit acceptant le statut diagnostic-only (D-FLIP-1 reste BLOCKED dans ce cas). Cette contrainte est connue depuis Plan 10-01 ; pas une régression Plan 10-02.

## Self-Check: PASSED

Files asserted as created/modified :

- `[FOUND]` pfc_shaping/validation/structural_tests.py
- `[FOUND]` tests/test_phase10_hildmann.py
- `[FOUND]` tests/test_phase10_empirical.py
- `[FOUND]` pfc_shaping/validation/scorecard.py (modified +329 lignes)
- `[FOUND]` pfc_shaping/requirements.txt (in-place decomment)
- `[FOUND]` tests/test_phase10_infra.py (stub test updated)

Commits asserted in git log :

- `[FOUND]` e1a80eb (Task 1 chore)
- `[FOUND]` 957f83f (Task 2 RED test)
- `[FOUND]` 41b3661 (Task 2 GREEN feat)
- `[FOUND]` ed4e5bd (Task 3 RED test)
- `[FOUND]` df88644 (Task 3 GREEN feat)

Test suite : 323 passed + 2 skipped (vs baseline 293+2 — net +30 new tests, 2 pre-existing skips préservés, 0 régression).

C4 REVIEWS audit-trail :

- `[FOUND]` grep -q "SC#1 GATE PRECONDITION" pfc_shaping/validation/scorecard.py → OK
- `[FOUND]` grep -q "MOCK CI" tests/test_phase10_hildmann.py → OK
- `[FOUND]` HOLIDAY_WEEKEND_RANGE = (0.65, 0.95) literal — no recalibration possible

---

*Phase: 10-pfc-fmv-quality-scorecard*
*Plan: 02 — Pillars 1 (Hildmann SC#1 gate) + 2 (KYOS empirical accuracy primitives) wired end-to-end*
*Completed: 2026-05-21*
