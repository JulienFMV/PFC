---
phase: 10-pfc-fmv-quality-scorecard
plan: 04
subsystem: validation
tags: [phase10, scorecard, wave4, final-assembly, sc1-diagnostic-only, d-flip-1-blocked, d-a6-3-verified, pillar5-peer-review, reproducibility]

# Dependency graph
requires:
  - phase: 10-pfc-fmv-quality-scorecard
    plan: 03
    provides: "pfc_shaping.validation.scorecard (compute_pillar3_coverage IC80 + compute_pillar4_dm 13-key) + christoffersen.py LR_uc + dm_test.py DM+3 baselines + 25 tests verts"
  - phase: 10-pfc-fmv-quality-scorecard
    plan: 02
    provides: "pfc_shaping.validation.scorecard (mz_test + compute_cell_kpis + run_scorecard_pillar_1) + structural_tests.py 4 Hildmann + matplotlib + statsmodels"
  - phase: 10-pfc-fmv-quality-scorecard
    plan: 01
    provides: "block_masks.py (ALL_BLOCKS) + ABLATION_GRID + build_one + FORWARDS_SOURCE_* + data/epex_hourly.parquet + data/forwards_history_phase10.parquet"
provides:
  - "pfc_shaping/validation/scorecard.py extended (+540 lignes) — run_scorecard_full + render_figures + render_markdown_report + _build_pred_baseline_for_vintage + HORIZONS_PILLAR2 + helpers"
  - "scripts/run_phase10_scorecard.py — CLI Mac Mini runner avec --reproducibility-subset"
  - "scripts/preflight_phase10.py — CLI preflight Task 0 (déjà livré commit 7fc3ff5)"
  - "tests/test_phase10_reproducibility.py — D-A6-3 contract assert (subset 4 builds atol=1e-12 rtol=0)"
  - ".planning/phases/10-pfc-fmv-quality-scorecard/10-VERIFICATION.md — scorecard final 5 piliers + 4 figures + Pillar 5 peer-review 9×6"
  - ".planning/phases/10-pfc-fmv-quality-scorecard/figures/*.png — 4 figures matplotlib publication-grade ≥150 dpi"
  - ".planning/phases/10-pfc-fmv-quality-scorecard/REPRODUCIBILITY-EXCEPTIONS.md — template documenté [C5 REVIEWS]"
  - ".planning/phases/10-pfc-fmv-quality-scorecard/10-PREFLIGHT.md — verdict file go [C6 REVIEWS] (Task 0)"
  - ".planning/phases/10-pfc-fmv-quality-scorecard/scorecard_kpis_pillar{2,3,4}.parquet — KPIs versionnés audit-trail"
  - ".planning/PROJECT.md updated — D-FLIP-1 BLOCKED entry timestamped 2026-05-21"
  - ".planning/REQUIREMENTS.md updated — BT-06..BT-10 status reflected (BT-06 partial diagnostic-only, BT-07..10 completed)"
affects: [Phase 10B (real-run gate-eligible), Phase 5ter (IC95 + Christoffersen conditional)]

# Tech tracking
tech-stack:
  added:
    - "matplotlib.pyplot + matplotlib.use('Agg') (figures PNG ≥150 dpi)"
    - "shutil.disk_usage (preflight Check 3)"
  patterns:
    - "Per-vintage parquet cache (96 PFC × ~4 MB = ~400 MB local Mac Mini) gitignored via .planning/phases/*/cache/ — re-run idempotent via cache_intermediate=True ; --reproducibility-subset N flag pour comparaison déterministe"
    - "Progress callback explicit (run_scorecard_full(progress_callback=...)) — log INFO ligne par milestone sans tool calls, orchestrator stall detection compatible"
    - "Pillar 3 ONLY on Config 4 (Pitfall 7 RESEARCH) — Uncertainty(n_boot=500) overhead 1.4x sur Config 4 seul ; les 3 autres configs build with_uncertainty=False"
    - "C3 REVIEWS gate-eligibility banner pattern : Pillar 1 header markdown ('✓ Gate-eligible run' OR '⚠ Diagnostic only — not gate-eligible') ; colonne forwards_source propagée dans Pillar 2/4 parquets + suffix '(diagnostic)' sur cellules fallback ; SC#1 ne peut être satisfait par run fallback"
    - "Tolerance double-niveau D-A6-3 [C5 REVIEWS] : primary atol=1e-12 rtol=0, fallback atol=1e-10 ONLY via entrée signée dans REPRODUCIBILITY-EXCEPTIONS.md ; _load_documented_exceptions parser strict ; aucune relaxation silencieuse possible"
    - "D-FLIP-1 BLOCKED audit-trail : entrée PROJECT.md timestamped + référence 10-VERIFICATION.md §Pillar 1 banner + 10-PREFLIGHT.md Check 2 ; le flag PFC_LT_USE_SEASONAL_HOURLY_SHAPE reste default OFF en code (zero modif shape_hourly.py:107)"

key-files:
  created:
    - "scripts/run_phase10_scorecard.py (~160 lignes) — CLI runner avec sys.path bootstrap"
    - "tests/test_phase10_reproducibility.py (~170 lignes) — D-A6-3 contract test (1 test, runtime 202s)"
    - ".planning/phases/10-pfc-fmv-quality-scorecard/10-VERIFICATION.md — scorecard final ~490 lignes"
    - ".planning/phases/10-pfc-fmv-quality-scorecard/REPRODUCIBILITY-EXCEPTIONS.md — template [C5 REVIEWS]"
    - ".planning/phases/10-pfc-fmv-quality-scorecard/figures/pillar1_seasonal_correlation_scatter.png"
    - ".planning/phases/10-pfc-fmv-quality-scorecard/figures/pillar2_mae_per_horizon_bar.png"
    - ".planning/phases/10-pfc-fmv-quality-scorecard/figures/pillar2_scatter_pred_vs_realised.png"
    - ".planning/phases/10-pfc-fmv-quality-scorecard/figures/pillar3_ic80_observed_vs_nominal.png"
    - ".planning/phases/10-pfc-fmv-quality-scorecard/scorecard_kpis_pillar2.parquet (100 rows)"
    - ".planning/phases/10-pfc-fmv-quality-scorecard/scorecard_kpis_pillar3.parquet (5 rows)"
    - ".planning/phases/10-pfc-fmv-quality-scorecard/scorecard_kpis_pillar4.parquet (300 rows)"
  modified:
    - "pfc_shaping/validation/scorecard.py (+540 lignes — 4 new exports __all__)"
    - ".gitignore (+4 lignes .planning/phases/*/cache/ + run_phase10.log)"
    - ".planning/PROJECT.md (+1 ligne D-FLIP-1 BLOCKED entry timestamped 2026-05-21)"
    - ".planning/REQUIREMENTS.md (+10 lignes — BT-06..BT-10 status notes)"

key-decisions:
  - "D-FLIP-1 BLOCKED (pas FLIPPED) — Le real-run Mac Mini 2026-05-21 a été exécuté sous `forwards_source=fallback_diagnostic` (snapshot EEX XLSX H:\\ inaccessible depuis Mac Mini). Verdict SC#1 = 2/4 PASS mais annoté DIAGNOSTIC ONLY par construction. Le flip ON est explicitement déféré jusqu'à un real-run depuis le poste FMV avec `forwards_source=real_eex_xlsx` (Phase 10B context). PROJECT.md entry timestamped 2026-05-21 documente l'audit-trail complet."
  - "Phase 10 quand même livrée `completed` (per user authorization 2026-05-21) — tous les deliverables shippés : scorecard_kpis_pillar{2,3,4}.parquet + 4 figures + 10-VERIFICATION.md complete (5 piliers + Pillar 5 peer-review 9×6 + 3 paragraphs + sources) + REPRODUCIBILITY-EXCEPTIONS.md + 349 tests verts dont test_phase10_reproducibility.py D-A6-3 verified. Le seul item non-shipped est le verdict SC#1 gate-eligible, qui requiert l'accès H:\\."
  - "D-A6-3 reproducibility verified sans fallback — `tests/test_phase10_reproducibility.py::test_reproducibility_subset_4_builds_config_4` PASS en 202s sur primary `atol=1e-12 rtol=0` ; REPRODUCIBILITY-EXCEPTIONS.md reste vide (état initial attendu). Subset 4 builds (Config 4 × 4 premières vintages 2024 avec cache_intermediate=False) ; les deux runs produisent des KPIs DataFrames bit-numerical-identical sur Mac Mini Python 3.12.12."
  - "Per-vintage construction baselines explicit (warning #8 fix) — helper `_build_pred_baseline_for_vintage(baseline_name, bloc, vintage_date, horizon_label, epex_hist, epex_realised, forwards_asof, realised_window) -> pd.Series` clarifie la construction des 3 baselines (climatology scalar(bloc), persistence_y1 scalar(bloc, vintage_date), forwards_flat scalar(bloc, vintage_date, horizon_label)). Docstring D-A4-1 référence ; ValueError sur unknown baseline_name."
  - "PFC 15-min → hourly resample mean pour aligner avec EPEX realised hourly — convention Phase 10 amendement 2026-05-21 (cache EPEX horaire natif). Le scorecard utilise un pas horaire honnête sans gonfler artificiellement N (pas de 15-min ffill)."
  - "C3 REVIEWS gate-eligibility wired end-to-end — colonne `forwards_source` propagée dans Pillar 2/4 KPIs parquets (validé : `df['forwards_source'].unique() == ['fallback_diagnostic']` × 3 parquets) + banner gate-eligibility en tête de Pillar 1 markdown (`grep -E 'Gate-eligible run|Diagnostic only — not gate-eligible' 10-VERIFICATION.md` = 2 hits) + suffix '(diagnostic)' sur les cellules fallback Pillar 2/4."

patterns-established:
  - "Helper preflight CLI pattern (5 checks, exit code 0/1, verdict file go/no-go) — `scripts/preflight_phase10.py` + `10-PREFLIGHT.md` ; verdict no-go BLOQUE Task 1 cost confirmation."
  - "Helper progress_callback pattern pour orchestrator-friendly long runs — `run_scorecard_full(progress_callback=fn)` émet INFO logs `[progress] ...` que l'orchestrator monitore pour stall detection sans tool calls."
  - "Tolerance double-niveau D-A6-3 [C5 REVIEWS] : primary 1e-12 (target SOTA), fallback 1e-10 ONLY via REPRODUCIBILITY-EXCEPTIONS.md entry (parseable strict format 6 colonnes) ; FAIL message guide la création d'entrée légitime (jamais silent relaxation)."
  - "BLOCKED audit-trail pattern : décision flippe-OR-bloque tracée comme une entrée Key Decisions full-form (justification + verdict + next step + reference cross-document) ; pas de modification code production (le flag reste à son default antérieur)."

requirements-completed: [BT-07, BT-08, BT-09, BT-10]
requirements-partial: [BT-06]  # diagnostic-only, not gate-eligible

# Metrics
duration: ~30 min effective work (post-checkpoint) + 10.6 min 96-build compute + 3.4 min reproducibility test = ~45 min total
completed: 2026-05-21
---

# Phase 10 Plan 04: PFC FMV Quality Scorecard — Final Assembly (DIAGNOSTIC-ONLY) Summary

**Scorecard 5-pillar Phase 10 livré end-to-end : 96-build ablation grid réel Mac Mini (634.5s),
4 figures matplotlib publication-grade, Pillar 5 peer-review SOTA 9×6 + 3 gap paragraphs +
sources, D-A6-3 reproducibility verified sans fallback. Verdict SC#1 Hildmann = 2/4 PASS sous
Config 4 (bowl ON + floors negative-ready) — annoté DIAGNOSTIC ONLY par construction
(`forwards_source=fallback_diagnostic`, H:\\ inaccessible Mac Mini). D-FLIP-1 explicitement
BLOCKED : flip ON déféré Phase 10B (real-run depuis FMV poste). Toutes deliverables shippées,
zero régression suite (349 passed + 2 skipped).**

## Performance

- **Duration (effective work, post Tasks 0-1)** : ~30 min agent work
- **Compute Task 2 (full 96-build)** : 634.5 s ≈ 10.6 min Mac Mini (Pillar 3 Uncertainty
  bootstrap n_boot=500 seed=42 seulement sur Config 4)
- **Compute Task 5 (reproducibility 2×4 builds)** : 202 s ≈ 3.4 min
- **Started Task 2** : 2026-05-21T12:34:48Z
- **Completed Task 7 (this SUMMARY)** : 2026-05-21T13:05:00Z
- **Tasks executed** : 6 of 7 (Tasks 2-7 ; Tasks 0/1 livrées pré-spawn — Task 4 pre-authorized,
  skipped without halt per user directive)
- **Files created** : 11 (scripts + tests + markdown + 4 figures + 3 KPIs parquets + 1 template)
- **Files modified** : 4 (scorecard.py + .gitignore + PROJECT.md + REQUIREMENTS.md)
- **Commits created** : 4 atomiques (Tasks 2-6) — `5e654f3` + `0edb692` + `714bb2d` + `2bd431d`
- **Test delta** : 348 → 349 (1 nouveau reproducibility test), 2 skipped préservés, 0 régression
- **Suite runtime** : 47s full hors reproducibility ; 202s avec reproducibility

## Accomplishments

- **Task 2 — `run_scorecard_full` + CLI + 96-build real-run** : `pfc_shaping/validation/scorecard.py`
  étendu (+540 lignes) avec :
  - `_build_pred_baseline_for_vintage` helper (warning #8 fix per D-A4-1) — 3 baselines
    explicites avec docstring sourcée.
  - `_horizon_to_window` / `_horizon_to_h_months` — convert 'M+N'/'Y+N' → UTC half-open
    windows + DM h= arg (12*N pour years).
  - `HORIZONS_PILLAR2 = ['M+1','M+3','M+6','Y+1','Y+2']`.
  - `run_scorecard_full(epex_source, output_dir, vintages_limit, cache_intermediate,
    progress_callback)` — orchestrate 96-build grid avec Pillar 3 ONLY Config 4 (Pitfall 7),
    per-vintage parquet cache (gitignored), pooling cross-vintage (Pitfall 5), KPIs
    deterministic column order pour D-A6-3.
  - `render_figures(scorecard_results, output_dir)` — 4 PNG matplotlib ≥150 dpi.
  - `render_markdown_report(scorecard_results, output_dir, holiday_weekend_range)` —
    `10-VERIFICATION.md` executive summary + 5 sections + figures embedded + **C3 REVIEWS
    gate-eligibility banner**.
  - `scripts/run_phase10_scorecard.py` (~160 lignes) — CLI argparse + sys.path bootstrap
    + INFO logger to `run_phase10.log` + stdout ; émet `SC#1 verdict: ...` + `SC#1 gate-eligible: ...`.

  **Real-run Mac Mini 2026-05-21** : 96 builds en **634.5 s ≈ 10.6 min** (preflight Task 0
  extrapolait 521s = 8.7 min, ordre de grandeur confirmé sans excès). 96 PFC parquets
  cachés (~400 MB total, gitignored). 3 KPIs parquets versionnés (`scorecard_kpis_pillar{2,3,4}`).
  4 figures PNG. `10-VERIFICATION.md` skeleton (Pillars 1-4 + placeholder Pillar 5).

- **Task 3 — Pillar 5 peer-review SOTA rédigé** : `10-VERIFICATION.md` placeholder
  remplacé par :
  - Table comparative 9 features × 6 implémentations (PFC FMV + KYOS + Volue + EULER +
    Benth-Koekebakker 2007 + Caldana 2017).
  - 3 gap analysis paragraphs : (a) où PFC FMV est SOTA (6/9 features), (b) où il y a gap
    actionnable (IC95 deferred Phase 5ter + peer review forwards Phase 7 + multi-market
    Phase 3 HOLD), (c) où on innove vs literature (delta-additif WaterValueCorrection
    sign-invariant + ctor args negative-ready convention + master flag audit-trail
    INFO-log-only).
  - Sous-section Sources : 6 références (KYOS KyCurve, Volue HPFC, EULER Phinergy,
    Benth-Koekebakker 2007 WP, Caldana 2017 Wilmott, Hildmann 2013 ETH PhD).

- **Task 4 — Pre-authorized review (no halt)** : Le user a explicitement préautorisé le
  passage sans nouveau checkpoint au verdict SC#1 (diagnostic-only) + D-FLIP-1 BLOCKED.
  Aucun halt émis ; passage direct Task 5.

- **Task 5 — D-A6-3 reproducibility verified** : `tests/test_phase10_reproducibility.py`
  livré (1 test, runtime 202 s) :
  - Subset 4 builds (Config 4 × 4 vintages) dans `tmp_path_factory` deux runs séparés
    avec `cache_intermediate=False`.
  - **[C5 REVIEWS] Tolerance double-niveau** : `REPRODUCIBILITY_TOLERANCE_PRIMARY`
    (atol=1e-12, rtol=0) + `REPRODUCIBILITY_TOLERANCE_FALLBACK` (atol=1e-10) + helper
    `_load_documented_exceptions` parsant `REPRODUCIBILITY-EXCEPTIONS.md` (format strict
    6 colonnes ; ligne absente → FAIL avec message guidant la création).
  - **Test PASS sur primary sans fallback** : `REPRODUCIBILITY-EXCEPTIONS.md` reste vide
    (état initial attendu, le contrat 1e-12 tient sur Mac Mini Python 3.12.12).

- **Task 6 — D-FLIP-1 BLOCKED audit-trail + REQUIREMENTS update** :
  - `.planning/PROJECT.md` : nouvelle entrée Key Decisions timestamped 2026-05-21
    documentant flip ON BLOCKED, justification fallback_diagnostic, défer Phase 10B,
    D-A6-3 verified, tag `(D-FLIP-1 BLOCKED)`.
  - `.planning/REQUIREMENTS.md` : BT-06 marqué `[~]` (partial diagnostic-only) +
    BT-07/08/09/10 marqués `[x]` (complétés avec notes).
  - **Skipped per user directive** : ROADMAP.md + STATE.md (handled by orchestrator
    post-wave 4).

## Verdict SC#1 Hildmann (real-run Mac Mini 2026-05-21)

| Test | Observed | Threshold | Passed | Gate-eligible |
|------|----------|-----------|--------|---------------|
| arb_free | 0.001 €/MWh | <0.01 €/MWh | ✓ | DIAGNOSTIC ONLY |
| holiday_weekend | (ratio observed) | [0.65, 0.95] | (per markdown) | DIAGNOSTIC ONLY |
| seasonal_profile | (pearson r) | >0.85 | (per markdown) | DIAGNOSTIC ONLY |
| continuity | (max jump €/MWh) | <2.0 €/MWh | (per markdown) | DIAGNOSTIC ONLY |

**Verdict** : 2/4 tests PASS (cf. `run_phase10.log` ligne `SC#1 verdict: FAIL 2/4`).
**Gate eligibility** : ⚠ Diagnostic only — `forwards_source=fallback_diagnostic` ; SC#1
ne peut PAS être satisfait par un run fallback (C3 REVIEWS marker hardcoded). Le verdict
informe mais ne tranche pas le D-FLIP-1.

## Verdict D-A6-3 reproducibility (Task 5 real-run Mac Mini 2026-05-21)

- **Subset** : Config 4 × 4 vintages (jan/fév/mar/avr 2024) ×2 runs (tmp_path_factory séparés)
- **Tolerance utilisée** : primary `atol=1e-12 rtol=0` (PASS sans fallback)
- **Pillars comparés** : Pillar 1 (TestResult.observed scalars) + Pillar 2/3/4 DataFrames
  (after sort_index axis 0+1 + reset_index)
- **Runtime** : 202 s
- **REPRODUCIBILITY-EXCEPTIONS.md** : état initial = vide (le contrat 1e-12 tient sur
  Mac Mini Python 3.12.12 + pandas 2.x + statsmodels 0.14.x + numpy 2.x).

## Task Commits

Each task committed atomically :

1. **Task 2 — `feat(10-04): wire run_scorecard_full + render_figures + render_markdown_report + _build_pred_baseline_for_vintage`** : `5e654f3`
2. **Task 2 (real-run) + Task 3 (Pillar 5) — `feat(10-04): full 96-build scorecard run + Pillar 5 peer-review SOTA (diagnostic-only)`** : `0edb692`
3. **Task 5 — `test(10-04): D-A6-3 reproducibility contract assert + REPRODUCIBILITY-EXCEPTIONS template`** : `714bb2d`
4. **Task 6 — `docs(10-04): D-FLIP-1 BLOCKED audit-trail + BT-06..BT-10 status update (diagnostic-only)`** : `2bd431d`

Plus le commit Task 0 pré-spawn :
- **Task 0 (preflight, pre-spawn) — `feat(10-04): preflight go/no-go script + verdict file [C6 REVIEWS]`** : `7fc3ff5`

## Files Created/Modified

### Created

- `scripts/run_phase10_scorecard.py` (~160 lignes) — CLI Mac Mini runner avec
  `--epex-source`, `--output-dir`, `--vintages-limit`, `--reproducibility-subset`, `--no-figures`,
  `--no-report` ; sys.path bootstrap (no pip install -e required) ; INFO logger to
  `run_phase10.log` + stdout ; émet `SC#1 verdict: PASS/FAIL X/N` + `SC#1 gate-eligible: Y/N`.
- `tests/test_phase10_reproducibility.py` (~170 lignes, 1 test) — D-A6-3 contract assert
  avec tolerance double-niveau [C5 REVIEWS] + helper `_load_documented_exceptions` parsing
  strict ; test PASS en 202s sur primary 1e-12 sans fallback.
- `.planning/phases/10-pfc-fmv-quality-scorecard/10-VERIFICATION.md` (~490 lignes) — scorecard
  final 5 piliers + executive summary + 4 figures embedded + Pillar 5 peer-review 9×6 + 3
  gap paragraphs + sources + annexes.
- `.planning/phases/10-pfc-fmv-quality-scorecard/REPRODUCIBILITY-EXCEPTIONS.md` — template
  [C5 REVIEWS] (purpose + convention + format strict + état initial vide).
- `.planning/phases/10-pfc-fmv-quality-scorecard/figures/pillar1_seasonal_correlation_scatter.png`
- `.planning/phases/10-pfc-fmv-quality-scorecard/figures/pillar2_mae_per_horizon_bar.png`
- `.planning/phases/10-pfc-fmv-quality-scorecard/figures/pillar2_scatter_pred_vs_realised.png`
- `.planning/phases/10-pfc-fmv-quality-scorecard/figures/pillar3_ic80_observed_vs_nominal.png`
- `.planning/phases/10-pfc-fmv-quality-scorecard/scorecard_kpis_pillar2.parquet` (100 rows × 14 cols)
- `.planning/phases/10-pfc-fmv-quality-scorecard/scorecard_kpis_pillar3.parquet` (5 rows × 11 cols)
- `.planning/phases/10-pfc-fmv-quality-scorecard/scorecard_kpis_pillar4.parquet` (300 rows × 17 cols)

### Modified

- `pfc_shaping/validation/scorecard.py` (+540 lignes) — 4 new exports `__all__` :
  `_build_pred_baseline_for_vintage`, `run_scorecard_full`, `render_figures`,
  `render_markdown_report`, `HORIZONS_PILLAR2` + helpers internes
  (`_horizon_to_window`, `_horizon_to_h_months`, `_detect_forwards_source`,
  `_forwards_for_vintage`).
- `.gitignore` (+4 lignes) — `.planning/phases/*/cache/` + `.planning/phases/*/run_phase10.log`
  (T-10-04-06 — PFC caches locaux Mac Mini).
- `.planning/PROJECT.md` (+1 ligne) — D-FLIP-1 BLOCKED entry timestamped 2026-05-21 dans
  Key Decisions table.
- `.planning/REQUIREMENTS.md` (+10 lignes) — BT-06 partial diagnostic-only + BT-07/08/09/10
  completed avec notes per-pillar.

## Decisions Made

- **D-FLIP-1 BLOCKED (pas FLIPPED) pour cette livraison Phase 10** — pre-authorized by user
  2026-05-21. Le real-run Mac Mini 2026-05-21 a été exécuté sous `forwards_source=fallback_diagnostic`
  (snapshot EEX XLSX H:\\ inaccessible). Verdict SC#1 = 2/4 PASS mais DIAGNOSTIC ONLY par
  construction. Flip ON déféré jusqu'à un real-run depuis FMV poste avec `forwards_source=real_eex_xlsx`
  (Phase 10B). PROJECT.md entry timestamped 2026-05-21 documente l'audit-trail.
- **Phase 10 marquée `completed` malgré SC#1 diagnostic-only** — toutes deliverables shippées
  (scorecard JSON parquets + 4 figures + 10-VERIFICATION.md 5 piliers + Pillar 5 peer-review +
  REPRODUCIBILITY-EXCEPTIONS template + 349 tests verts). Le seul item non-shipped = verdict
  SC#1 gate-eligible (requiert H:\\). Cohérent EPFL principle "build it right by design,
  validate against principle" — le scorecard est livré ; seule la validation gate-eligible
  est différée.
- **D-A6-3 reproducibility verified sans fallback** — `tests/test_phase10_reproducibility.py`
  PASS en 202s sur primary `atol=1e-12 rtol=0` ; REPRODUCIBILITY-EXCEPTIONS.md reste vide
  (état initial attendu). Pas de blocker côté determinism — confirme l'environnement Mac Mini
  Python 3.12.12 produit des KPIs bit-numerical-identical à chaque run.
- **PFC 15-min → hourly resample mean pour alignement** — le scorecard utilise un pas horaire
  honnête sans gonfler artificiellement N (pas de 15-min ffill). Convention Phase 10 amendement
  2026-05-21 ; cohérent avec le cache EPEX horaire natif.
- **C3 REVIEWS gate-eligibility wired end-to-end** — colonne `forwards_source` propagée dans
  les 3 KPIs parquets Pillar 2/3/4 + banner gate-eligibility en tête Pillar 1 markdown +
  suffix '(diagnostic)' sur les cellules fallback Pillar 2/4 ; SC#1 ne peut être satisfait
  par un run fallback.
- **ROADMAP.md + STATE.md non modifiés par cet agent continuation** — user directive
  explicite ("Do NOT update STATE.md or ROADMAP.md") → handled by orchestrator post-wave 4.

## Deviations from Plan

Aucune deviation Rule 4 (architectural) déclenchée. Le plan a été exécuté tel que rédigé.

### Auto-fixed Issues

**1. [Rule 3 — Blocking] sys.path bootstrap in `scripts/run_phase10_scorecard.py`**

- **Found during:** Task 2 dry-run smoke test (`/tmp/phase10_dryrun`)
- **Issue:** `ModuleNotFoundError: No module named 'pfc_shaping'` au lancement direct
  du script (pas de `pip install -e .` ni de `pyproject.toml` dans le repo).
- **Fix:** Ajout d'un bloc `sys.path.insert(0, str(_REPO_ROOT))` au header du script
  pour permettre exécution standalone depuis n'importe quel cwd.
- **Files modified:** `scripts/run_phase10_scorecard.py` (+5 lignes header)
- **Verification:** Dry-run 1 vintage post-fix PASS (28s compute + 10s figures/report).
- **Committed in:** `5e654f3` (same commit as Task 2 implementation).

---

**Total deviations:** 1 auto-fixed (1× Rule 3 — Blocking ; 0 Rule 1 ; 0 Rule 2 ; 0 Rule 4)

**Impact on plan:** Pure correction technique d'exécution (path bootstrap). Aucune décision
architecturale, aucun scope creep. Le script est désormais utilisable sans installation
package.

## Authentication Gates

Aucun. Tous les outputs sont locaux Mac Mini ; pas d'API call ni de credentials requis.

## Pre-authorized checkpoint skipped

- **Task 4 (`checkpoint:human-verify` — review scorecard + D-FLIP-1 decision)** : skipped
  per user directive 2026-05-21 ("Do NOT halt for an additional checkpoint at SC#1 verdict
  or D-FLIP-1 BLOCKED decision — both are pre-authorized by this approval"). La décision
  D-FLIP-1 BLOCKED a été appliquée directement en Task 6.

## Issues Encountered

- **Aucun blocker rencontré**. La chaîne `run_scorecard_full` → `render_figures` →
  `render_markdown_report` a passé dry-run 1 vintage + full 96-build + reproducibility test
  sans crash.
- **Warning baseline_persistence_y1 empty window sur `block_summer_solar_bowl`** — observation
  normale : Q1 2024 vintage cherche une persistance Y-1 = Q1 2023 avec une fenêtre ±15j
  autour du dernier jour ouvré ; le bloc summer_solar_bowl (mai-août) tombe hors de cette
  fenêtre → mask vide → NaN. Le DM cellule correspondant porte `degenerate=True` flag
  explicit (pas un crash).
- **Warning matplotlib font cache build (first-run only)** — normal pour environnement
  Python 3.12.12 freshly initialized ; aucune incidence sur les figures générées.

## Known Stubs

**Aucun stub silencieux non-documenté**. Tous les outputs sont des dicts complets (incluant
`degenerate: True` flag explicit pour les cellules dégénérées), Series broadcast NaN (avec
flag downstream), ou pd.DataFrame avec colonnes deterministic. Pas de placeholder, pas de
TODO, pas de NotImplementedError résiduel sur les fonctions du scope Plan 10-04.

Le seul "stub" volontaire est la **Pillar 5 placeholder** dans
`render_markdown_report` — c'est intentionnel car Task 3 réécrit cette section avec le
contenu peer-review final.

## Threat Flags

Tous les threats du `<threat_model>` Plan 10-04 sont mitigés :

- **T-10-04-01 (Tampering — 96-build seed determinism re-run)** : ASSERTED via
  `tests/test_phase10_reproducibility.py` (subset 4 builds atol=1e-12 rtol=0 PASS en 202s
  sans fallback) ✓
- **T-10-04-02 (Repudiation — 10-VERIFICATION.md verdict SC#1 not traceable)** : verdict
  inscrit dans 10-VERIFICATION.md committed git + PROJECT.md entry timestamped 2026-05-21 +
  cross-reference 10-PREFLIGHT.md Check 2 + 10-VERIFICATION.md §Pillar 1 banner ✓
- **T-10-04-03 (Tampering — Pillar 5 sources non vérifiables)** : section Sources explicite
  avec 6 références (KYOS + Volue + EULER + Benth-Koekebakker 2007 + Caldana 2017 + Hildmann
  2013) sourcées vs `reference_pfc_state_of_art.md` (user memory) ✓
- **T-10-04-04 (DoS — 96-build run >2h)** : real-run 634.5s = 10.6 min Mac Mini (bien sous
  budget 2.5h hard cap) ; intermediate cache + vintages_limit pour dry-run préalable ;
  reproducibility subset 4 builds = 202s ✓
- **T-10-04-05 (Elevation of privilege — CLI script)** : pure read-only sur data caches +
  write only à output_dir ; pas de modification code production (le flag
  PFC_LT_USE_SEASONAL_HOURLY_SHAPE reste default OFF dans shape_hourly.py:107) ✓
- **T-10-04-06 (Information disclosure — Cache PFC parquets ~400 MB)** : `.gitignore`
  amended avec `.planning/phases/*/cache/` + `run_phase10.log` ; verified via
  `git check-ignore -v` (les 96 caches sont ignorés) ✓
- **T-10-04-07 (Repudiation — `_build_pred_baseline_for_vintage` logic implicit warning #8)** :
  helper explicite avec docstring D-A4-1 + 3 cases couverts (climatology/persistence_y1/
  forwards_flat) + ValueError sur unknown baseline_name ✓

Aucun threat flag nouveau à signaler.

## Self-Check: PASSED

Files asserted as created/modified :

- `[FOUND]` scripts/run_phase10_scorecard.py
- `[FOUND]` tests/test_phase10_reproducibility.py
- `[FOUND]` .planning/phases/10-pfc-fmv-quality-scorecard/10-VERIFICATION.md
- `[FOUND]` .planning/phases/10-pfc-fmv-quality-scorecard/REPRODUCIBILITY-EXCEPTIONS.md
- `[FOUND]` .planning/phases/10-pfc-fmv-quality-scorecard/figures/pillar1_seasonal_correlation_scatter.png
- `[FOUND]` .planning/phases/10-pfc-fmv-quality-scorecard/figures/pillar2_mae_per_horizon_bar.png
- `[FOUND]` .planning/phases/10-pfc-fmv-quality-scorecard/figures/pillar2_scatter_pred_vs_realised.png
- `[FOUND]` .planning/phases/10-pfc-fmv-quality-scorecard/figures/pillar3_ic80_observed_vs_nominal.png
- `[FOUND]` .planning/phases/10-pfc-fmv-quality-scorecard/scorecard_kpis_pillar2.parquet
- `[FOUND]` .planning/phases/10-pfc-fmv-quality-scorecard/scorecard_kpis_pillar3.parquet
- `[FOUND]` .planning/phases/10-pfc-fmv-quality-scorecard/scorecard_kpis_pillar4.parquet
- `[FOUND]` pfc_shaping/validation/scorecard.py (modified +540 lignes)
- `[FOUND]` .planning/PROJECT.md (D-FLIP-1 BLOCKED entry)
- `[FOUND]` .planning/REQUIREMENTS.md (BT-06..BT-10 status updated)
- `[FOUND]` .gitignore (cache patterns added)

Commits asserted in git log :

- `[FOUND]` 7fc3ff5 (Task 0 preflight, pre-spawn)
- `[FOUND]` 5e654f3 (Task 2 implementation)
- `[FOUND]` 0edb692 (Task 2 real-run + Task 3 Pillar 5)
- `[FOUND]` 714bb2d (Task 5 reproducibility)
- `[FOUND]` 2bd431d (Task 6 D-FLIP-1 BLOCKED + REQUIREMENTS)

Test suite : 349 passed + 2 skipped (vs baseline 348 + 2 — net +1 nouveau test
reproducibility, 0 régression).

Acceptance criteria canoniques (du plan §verification + §success_criteria) :

- `[PASS]` `python -c "from pfc_shaping.validation.scorecard import run_scorecard_full, render_figures, render_markdown_report, _build_pred_baseline_for_vintage"` (all 4 exports)
- `[PASS]` `scripts/run_phase10_scorecard.py --help` affiche argparse options incl. `--reproducibility-subset`
- `[PASS]` 96 PFC parquets dans `cache/` (réécriture full)
- `[PASS]` 4 figures PNG dans `figures/`
- `[PASS]` 10-VERIFICATION.md complete (5 piliers + Pillar 5 9×6 + 3 gap paragraphs)
- `[PASS]` grep "Gate-eligible run|Diagnostic only — not gate-eligible" 10-VERIFICATION.md = 2 hits
- `[PASS]` grep "delta-additif|ctor args negative-ready|master flag audit-trail" 10-VERIFICATION.md = 4 hits
- `[PASS]` grep "KYOS|Volue|EULER|Benth-Koekebakker|Caldana|Hildmann" 10-VERIFICATION.md = 23 hits
- `[PASS]` grep "PLACEHOLDER|TBD|TODO" 10-VERIFICATION.md = 0 hits
- `[PASS]` forwards_source colonne dans pillar2/3/4 parquets (3/3 unique values = ['fallback_diagnostic'])
- `[PASS]` SC#1 verdict line dans run_phase10.log : "SC#1 verdict: FAIL 2/4"
- `[PASS]` SC#1 gate-eligible line dans run_phase10.log : "SC#1 gate-eligible: N (forwards_source=fallback_diagnostic)"
- `[PASS]` pytest tests/test_phase10_reproducibility.py = 1 passed en 202s
- `[PASS]` REPRODUCIBILITY-EXCEPTIONS.md exists + REPRODUCIBILITY_TOLERANCE_FALLBACK in test file
- `[PASS]` _load_documented_exceptions helper exposed in test file
- `[PASS]` PROJECT.md "D-FLIP-1 BLOCKED" entry exists
- `[PASS]` REQUIREMENTS.md BT-06..BT-10 status updated

---

*Phase: 10-pfc-fmv-quality-scorecard*
*Plan: 04 — Final Assembly + Release (5-pillar scorecard livré, SC#1 DIAGNOSTIC-ONLY, D-FLIP-1 BLOCKED)*
*Completed: 2026-05-21*
