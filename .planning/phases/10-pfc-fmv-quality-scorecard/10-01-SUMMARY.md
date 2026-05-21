---
phase: 10-pfc-fmv-quality-scorecard
plan: 01
subsystem: validation
tags: [phase10, scorecard, block-masks, ablation-grid, hildmann, epex-bootstrap, fallback-forwards, c2-reviews, c3-reviews]

# Dependency graph
requires:
  - phase: 05B-shape-hourly-infrastructure-flag-no-op-refactor
    provides: baseline_pfc_seed42 fixture frozen, PFC_LT_USE_SEASONAL_HOURLY_SHAPE flag, conftest autouse env hygiene PFC_LT_* prefix
  - phase: 05C-shape-hourly-bowl-deepening
    provides: bowl deepening shippé, σ paramétrable, D-FLIP-1 entry, baseline_pfc_seed42_bowl fixture
  - phase: 05-msfc-log-prix-retire-silent-floors
    provides: 4 ctor args PFCAssembler (enforce_positivity, enforce_m_factor_floor, enforce_floor, allow_negative_peak), PFC_LT_ALLOW_NEGATIVE_PRICES audit-trail INFO, defaults negative-ready
provides:
  - pfc_shaping/validation/block_masks.py with 5 BlockMask classes (D-A2-2) + ALL_BLOCKS
  - pfc_shaping/validation/scorecard.py skeleton (AblationConfig × 4, list_vintages_2024_2025 × 24, last_business_day_of_month, build_one, derive_forwards_from_epex_hist body, FORWARDS_SOURCE_* constants, run_scorecard_pillar_1 stub)
  - data/epex_hourly.parquet (61,368 rows, 2019-2025 tz-UTC horaire, gitignored)
  - data/forwards_history_phase10.parquet (1,188 records, 24 vintages × ~49 keys, forwards_source=fallback_diagnostic, gitignored)
  - 10-01-NOTES.md decisions empiriques (C2 REVIEWS frozen method + ratio 0.8033 + threshold (0.65, 0.95) IF branch + Q2 fallback path + C3 REVIEWS marker convention)
  - ROADMAP.md / REQUIREMENTS.md / PROJECT.md updates reflecting D-A0-2 pivot
  - tests/test_phase10_infra.py (16 tests verts, runtime <0.5s)
affects: [10-02, 10-03, 10-04, 10B, 5ter]

# Tech tracking
tech-stack:
  added:
    - "pre-authorized (Plan 10-02 install): matplotlib>=3.7.0 + statsmodels==0.14.6 (human-verified Plan 10-01 Task 1)"
  patterns:
    - "BlockMask class hierarchy: tz-naive raises ValueError, tz_convert UTC→Europe/Zurich for local-hour semantics, np.ndarray[bool] aligned on input idx_utc"
    - "AblationConfig as @dataclass(frozen=True) — 4 configs D-A6-1 explored in 2×2 grid (bowl OFF/ON × floors OFF/ON)"
    - "list_vintages_2024_2025: BMonthEnd(0) on Europe/Zurich + replace(hour=18, minute=0) + tz_convert(UTC) — post-market close convention"
    - "Fallback-by-design pattern: derive_forwards_from_epex_hist marked structurally via FORWARDS_SOURCE_FALLBACK_DIAGNOSTIC constant + parquet column forwards_source, gate-eligibility filtered downstream (Plan 10-04 SC#1 reads marker)"
    - "TDD gate sequence: test commit → feat commit → data/docs commit per task (Phase 10 honored)"
    - "C2 REVIEWS ex-ante freeze pattern: write the recalibration formula in NOTES.md + separate commit BEFORE any empirical measurement to mechanically forbid post-hoc tweaks"

key-files:
  created:
    - pfc_shaping/validation/block_masks.py
    - pfc_shaping/validation/scorecard.py
    - tests/test_phase10_infra.py
    - .planning/phases/10-pfc-fmv-quality-scorecard/10-01-NOTES.md
    - data/epex_hourly.parquet (gitignored)
    - data/forwards_history_phase10.parquet (gitignored)
  modified:
    - pfc_shaping/validation/__init__.py (was 0-byte → enriched docstring + __all__)
    - pfc_shaping/requirements.txt (in-place comment update for matplotlib + new statsmodels line)
    - .gitignore (added data/*.parquet + data/_cache/ + data/_h_cache/)
    - .planning/ROADMAP.md (Phase 10 reframed + Phase 10B added)
    - .planning/REQUIREMENTS.md (BT-01..BT-05 → BT-01/BT-02/BT-04 + BT-06..BT-10 + BT-10B-01/02 new section)
    - .planning/PROJECT.md (new 2026-05-20 D-A0-* entry + D-FLIP-1 wording update)

key-decisions:
  - "Pillar 1.2 holiday/weekend threshold = (0.65, 0.95) confirmed empirically (branch IF, ratio mesuré 0.8033 ∈ [0.65, 0.95])"
  - "Forwards-as-of-vintage path Mac Mini = fallback_diagnostic via derive_forwards_from_epex_hist (H:\\ inaccessible) — gate-eligibility blocks SC#1 unless real_eex_xlsx source used"
  - "5 blocs client renommés D-A2-2 (block_overnight_weekday, block_midday_weekday, block_weekend_midday, block_summer_solar_bowl, block_winter_evening_peak)"
  - "AblationConfig grid = 4 configs (D-A6-1): bowl × floors 2×2, Config 4 (bowl_on_floors_off) = production target SC#1 gate"
  - "Vintages calendar = BMonthEnd(0) on Europe/Zurich pinned 18:00 local → tz_convert UTC (24 timestamps Jan 2024..Dec 2025)"
  - "C2 REVIEWS frozen recalibration formula written EX-ANTE in separate commit (376f1e4) before empirical measurement (e955a2f) — audit-trail enforced by commit ordering"
  - "C3 REVIEWS forwards_source as structured column (not log line): FORWARDS_SOURCE_REAL=real_eex_xlsx vs FORWARDS_SOURCE_FALLBACK_DIAGNOSTIC=fallback_diagnostic, propagated via parquet"

patterns-established:
  - "BlockMask base class with tz-safety contract: raise ValueError on tz-naive idx, internal tz_convert UTC→Europe/Zurich for local-hour semantics, return np.ndarray[bool] aligned on input"
  - "Cross-midnight semantics for overnight blocks: (hour >= 18) | (hour < 9) — no wrap-around special case needed since mask is index-by-index"
  - "build_one fit-from-scratch with strict no-leakage filter (epex_hist.index < vintage), 1 cellule of the 96-grid (4 configs × 24 vintages), with_uncertainty kwarg optional"
  - "C2 REVIEWS-style ex-ante freeze: forbidden-pattern enumeration + verbatim-formula audit trail to prevent post-hoc threshold tweaks on politically-sensitive metrics"

requirements-completed: [BT-01, BT-02, BT-06, BT-07, BT-08, BT-09, BT-10]

# Metrics
duration: ~25min (continuation agent post-Task-1 human-verify)
completed: 2026-05-21
---

# Phase 10 Plan 01: PFC FMV Quality Scorecard Infrastructure Scaffold Summary

**5 BlockMask tz-aware classes + 4-config ablation grid + 24-vintage calendar + fallback forwards proxy + EPEX 2019-2025 bootstrap + 5-pillar scorecard docs pivot (D-A0-2), ratio Hildmann mesuré 0.8033 → threshold (0.65, 0.95) confirmé**

## Performance

- **Duration:** ~25 min (continuation agent — Task 1 verified pre-spawn)
- **Started:** 2026-05-21T07:30:00Z (continuation spawn after Task 1 human-verify "approved")
- **Completed:** 2026-05-21T07:55:00Z
- **Tasks executed:** 3 of 4 (Task 1 verified pre-spawn, Tasks 2/3/4 executed here)
- **Files modified:** 10 (4 created + 4 modified + 2 data caches gitignored)
- **Commits created:** 5 atomiques (1 RED + 1 GREEN Task 2, 1 C2-frozen-ex-ante + 1 Task 3, 1 Task 4)
- **Test delta:** 279 → 295 collected (291 passed + 4 skipped, +16 new infra tests, 0 regression)

## Accomplishments

- **Scorecard infra livré (skeleton-only)** : `pfc_shaping/validation/{block_masks,scorecard}.py` importables, `ALL_BLOCKS` (5 items) + `ABLATION_GRID` (4 items) + `list_vintages_2024_2025()` (24 timestamps) + `derive_forwards_from_epex_hist` (body implémenté) + `FORWARDS_SOURCE_REAL/_FALLBACK_DIAGNOSTIC` constants exportés.
- **Cache EPEX 2019-2025 prêt** : 61,368 rows tz-UTC horaire dans `data/epex_hourly.parquet` (gitignored), bootstrap via `energy-charts.info` pipeline existant. Couvre la totalité de la fenêtre Phase 10 walk-forward.
- **Décision empirique Hildmann Pillar 1.2 tranchée mécaniquement** : ratio mesuré 0.8033 sur 2019-2023 CH/VS → branche IF de la formule frozen → threshold (0.65, 0.95) research default confirmé. Aucune déviation vs la formule (C2 REVIEWS audit-trail green).
- **Q2 RESEARCH forwards-as-of-vintage tranchée** : test H:\ FAIL → fallback `derive_forwards_from_epex_hist` body implémenté + 1,188 records (24 vintages × ~49 keys) cachés dans `data/forwards_history_phase10.parquet` avec colonne `forwards_source=fallback_diagnostic` (C3 REVIEWS marker structuré).
- **Pivot D-A0-2 acté en doc** : ROADMAP/REQUIREMENTS/PROJECT.md cohérents avec Phase 10 = scorecard absolu (SC#1 Hildmann gate) + Phase 10B = OMPEX deferred.
- **Requirements.txt pré-autorisé** : matplotlib + statsmodels==0.14.6 annotés human-verified pour install Plan 10-02 (in-place comment update, pas de duplicate orphan).

## Task Commits

Each task was committed atomically (commit hashes show TDD ordering — RED before GREEN, ex-ante C2 before empirical measure) :

1. **Task 1 — Package legitimacy human-verify** : verified pre-spawn by user ("approved" — statsmodels + matplotlib canonical PyPI confirmed)
2. **Task 2 RED — failing tests for block_masks + scorecard skeleton + env hygiene** : `ea46ca4` (test)
3. **Task 2 GREEN — scaffold block_masks + scorecard skeleton + requirements pre-auth** : `a02ca10` (feat)
4. **Task 3 sub-step 3a-bis — freeze C2 REVIEWS recalibration method EX-ANTE before empirical measure** : `376f1e4` (docs)
5. **Task 3 sub-steps 3a/3b/3c — bootstrap EPEX 2019-2025 + derive_forwards body + empirical decisions** : `e955a2f` (feat)
6. **Task 4 — pivot D-A0-2 ROADMAP/REQUIREMENTS/PROJECT** : `e2672b4` (docs)

**Plan metadata commit:** _(this commit, after SUMMARY.md write)_

_Note: Task 2 followed TDD (RED → GREEN). Task 3 followed C2 REVIEWS ex-ante freeze (docs commit BEFORE empirical measure commit) to mechanically prevent threshold tweaks._

## Files Created/Modified

### Created

- `pfc_shaping/validation/block_masks.py` (141 lignes) — 5 BlockMask classes (D-A2-2) avec tz-safety + cross-midnight + ALL_BLOCKS canonique
- `pfc_shaping/validation/scorecard.py` (~280 lignes) — AblationConfig + ABLATION_GRID (4 configs) + list_vintages_2024_2025 + last_business_day_of_month + build_one + derive_forwards_from_epex_hist (body) + FORWARDS_SOURCE_* constants + run_scorecard_pillar_1 stub
- `tests/test_phase10_infra.py` (312 lignes, 16 tests) — block_masks + scorecard + conftest env hygiene
- `.planning/phases/10-pfc-fmv-quality-scorecard/10-01-NOTES.md` (~200 lignes) — 4 sections (C2 REVIEWS frozen formula, Pitfall 1 mesure, Q2 forwards path, C3 REVIEWS marker convention)
- `data/epex_hourly.parquet` (gitignored) — 61,368 rows tz-UTC horaire, range 2019-01-01..2025-12-31
- `data/forwards_history_phase10.parquet` (gitignored) — 1,188 records, 24 vintages × ~49 keys, forwards_source=fallback_diagnostic

### Modified

- `pfc_shaping/validation/__init__.py` — was 0-byte, enriched with docstring + `__all__`
- `pfc_shaping/requirements.txt` — in-place comment update line 38 matplotlib (human-verified Plan 10-01 Task 1) + new statsmodels==0.14.6 line, no orphan duplicate
- `.gitignore` — added `data/*.parquet` + `data/_cache/` + `data/_h_cache/` (Phase 10 root-level data caches), tracked-file rule preserved
- `.planning/ROADMAP.md` — Phase 10 reframed "PFC FMV Quality Scorecard (5-pillar SOTA replication)", new Phase 10B (deferred), Plans listed explicitly
- `.planning/REQUIREMENTS.md` — section renamed, BT-01/02 preserved, BT-04 reformulated, BT-03/05 migrated to BT-10B-01/02, BT-06..BT-10 added, Traceability split
- `.planning/PROJECT.md` — new 2026-05-20 D-A0-* entry, D-FLIP-1 wording updated (gate = SC#1 Hildmann 4/4 PASS Config 4)

## Decisions Made

- **Threshold Hildmann Pillar 1.2 = (0.65, 0.95)** — application mécanique de la formule frozen Sub-step 3a-bis. Ratio empirique mesuré 0.8033 ∈ [0.65, 0.95] → branche IF. Aucune déviation. Le P50 monthly 0.8002 confirme la stabilité du signal.
- **Forwards-as-of-vintage path Mac Mini = fallback diagnostic** — H:\\ inaccessible (attendu) → body `derive_forwards_from_epex_hist` implémenté avec proxies same-period mean (Cal/Q/M depuis EPEX hist). Marker `forwards_source=fallback_diagnostic` propagé. SC#1 ne pourra PAS être satisfait par un run fallback (Plan 10-04 gate-filter).
- **5 blocs renommés D-A2-2** retenus tels quels (block_overnight_weekday, midday_weekday, weekend_midday, summer_solar_bowl, winter_evening_peak) — convention finale Phase 10.
- **AblationConfig.allow_negative_peak True/False inverse de enforce_*** — décidé en cohérence avec D-A6-1 CONTEXT : mode legacy = enforce_*=True + allow_negative_peak=False ; mode negative-ready = inverse.
- **C2 REVIEWS audit-trail via commit ordering** — frozen formula commit (376f1e4) PRÉCÈDE le commit empirical measure (e955a2f) ; preuve git que la formule n'a pas été dérivée des chiffres.
- **C3 REVIEWS marker as structured parquet column** (pas log line) — colonne `forwards_source` obligatoire dans `data/forwards_history_phase10.parquet`, nunique()==1 sur Mac Mini default.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] Canonical EPEX column name `price_eur_mwh` (not `price`)**
- **Found during:** Task 3 sub-step 3a EPEX bootstrap
- **Issue:** Le plan §verify suggère `df['price']` mais `load_prices` retourne en réalité `df[['price_eur_mwh', 'spike_flag']]` (column name canonique dans ingest_energy_charts.py:113).
- **Fix:** Tous mes appels (`prices = df['price_eur_mwh']`, parquet readback) utilisent le nom canonique. La signature de `derive_forwards_from_epex_hist` prend `pd.Series` (caller-controlled) donc agnostique au nom — pas de patch source nécessaire.
- **Files modified:** N/A (decision propagée dans les Bash calls + tests qui se basent déjà sur la Series passée, pas sur le DataFrame).
- **Verification:** `data/epex_hourly.parquet` re-loaded successfully + derive_forwards returns 49 keys on vintage 2024-06-28.
- **Committed in:** `e955a2f` (Task 3) — documenté dans le commit body.

**2. [Rule 3 — Blocking] `.gitignore` extension for root-level data/*.parquet**
- **Found during:** Task 3 sub-step 3a (gitignore audit)
- **Issue:** `.gitignore` couvrait uniquement `pfc_shaping/data/*` pour les caches parquet, pas `data/*.parquet` au repo root. Sans patch, `data/epex_hourly.parquet` (245k rows = ~20 MB) aurait été untracked-mais-visible et risque de commit accidentel par `git add -A` futur.
- **Fix:** Ajouté `data/*.parquet`, `data/_cache/`, `data/_h_cache/` au `.gitignore`. Vérifié que `data/eex_forwards_history.parquet` + `data/commodities_cache.parquet` (déjà trackés) ne sont pas masqués (git ignore ne s'applique pas aux fichiers déjà trackés).
- **Files modified:** `.gitignore`
- **Verification:** `git check-ignore -v data/epex_hourly.parquet` → 21:data/*.parquet match. `git ls-files data/*.parquet` → 2 trackés (eex_forwards_history, commodities_cache) toujours présents.
- **Committed in:** `e955a2f` (Task 3)

**3. [Rule 3 — Blocking] `last_business_day_of_month` returns Europe/Zurich tz (not UTC)**
- **Found during:** Task 2 (writing tests)
- **Issue:** Le plan §action ligne 265 dit "tz_convert('UTC')" inside `last_business_day_of_month`, mais c'est plus propre de garder la valeur en tz-Europe/Zurich et de laisser `list_vintages_2024_2025` faire le `tz_convert("UTC")` final. Cela permet aux callers de tester `replace(hour=18, minute=0)` en local time directement.
- **Fix:** `last_business_day_of_month` retourne tz-aware Europe/Zurich ; `list_vintages_2024_2025` appelle `.tz_convert("UTC")` au bout. Sémantique identique au plan, juste plus testable (test `test_last_business_day_of_month_march_2024` vérifie `.tz_convert("UTC").hour == 17` = 18 local CET).
- **Files modified:** `pfc_shaping/validation/scorecard.py`
- **Verification:** Test `test_last_business_day_of_month_march_2024` PASS (29 mars 2024, 18:00 CET = 17:00 UTC).
- **Committed in:** `a02ca10` (Task 2 GREEN)

---

**Total deviations:** 3 auto-fixed (Rule 3 — Blocking)
**Impact on plan:** All 3 auto-fixes étaient des corrections de précision technique imposées par le code réel (column naming) ou la propreté git (cache leakage prevention). Aucun scope creep ; les 3 sont documentés dans les commits.

## Issues Encountered

- **Aucun blocker pendant l'exécution.** Le seul "issue" notable a été le warning pandas `Converting to PeriodArray/Index representation will drop timezone information` pendant le bootstrap EPEX et la mesure empirique — c'est un warning bénin de `groupby(.to_period())` qui ne corrompt pas le calcul (groupby est itéré par period independent of tz, et nous re-projeté via `.dt.to_period().values`). Documenté ici ; pas une correction nécessaire.

## Known Stubs

Les stubs intentionnels suivants sont **par design** (skeleton-only Plan 10-01 ; bodies implémentés Plans 10-02/03/04) :

| Stub | File | Line | Reason | Resolved by |
|------|------|------|--------|-------------|
| `run_scorecard_pillar_1` raises `NotImplementedError("Pillar 1 wiring deferred to Plan 10-02")` | `pfc_shaping/validation/scorecard.py` | ~285 | Skeleton — Hildmann 4 tests body in Plan 10-02 (`structural_tests.py`) | Plan 10-02 |
| `pfc_shaping/validation/__init__.py` lists `block_masks` + `scorecard` but no eager-imported wirings to Plans 10-02/03/04 modules (`structural_tests`, `christoffersen`, `dm_test`) | `pfc_shaping/validation/__init__.py` | docstring | Plan 10-01 ne crée pas ces modules ; ils seront ajoutés au `__all__` quand wired | Plans 10-02/03 |

Aucun stub silencieux non-documenté (ni dans le data layer, ni dans l'UI — il n'y a pas d'UI dans Phase 10).

## Threat Flags

Pas de nouvelle surface de sécurité non-couverte par le threat model du PLAN.md :

- **T-10-01 / T-10-SC (pypi install)** : human-verify Task 1 effectué (approved) ; install effective reportée à Plan 10-02 — mitigation intacte.
- **T-10-02 (energy-charts.info JSON)** : 245k rows téléchargés + cache parquet local, retry logic + spike_flag ingest existants. Pas de PII, pas d'auth, CC BY 4.0 — accept disposition préservée.
- **T-10-03 (DoS EPEX bootstrap)** : 7 ans téléchargés ~~4 minutes wall-time, pas de timeout — cache-aside fonctionne. SMARD/ENTSO-E fallback non déclenché.
- **T-10-04 (repudiation NOTES decisions)** : audit-trail garantie par commit ordering (376f1e4 ex-ante PRÉCÈDE e955a2f empirical).

Aucun threat flag nouveau à signaler.

## User Setup Required

None — pas de configuration de service externe. La Phase 10 reste 100% Mac Mini-local. L'install effective de `matplotlib + statsmodels` est reportée à Plan 10-02 (pré-autorisée Task 1 du présent plan).

## Next Phase Readiness

**Plan 10-02 peut démarrer immédiatement** avec les artefacts suivants disponibles :

- `pfc_shaping.validation.block_masks.ALL_BLOCKS` (5 BlockMask tz-aware itérables)
- `pfc_shaping.validation.scorecard.ABLATION_GRID` (4 AblationConfig)
- `pfc_shaping.validation.scorecard.list_vintages_2024_2025()` (24 timestamps UTC)
- `pfc_shaping.validation.scorecard.build_one(config, vintage, epex_hist, forwards_asof)` (no-leakage fit + PFCAssembler kwargs câblé)
- `data/epex_hourly.parquet` (cache 2019-2025 tz-UTC 15-min)
- `data/forwards_history_phase10.parquet` (24 vintages × 49 keys forwards proxy, forwards_source=fallback_diagnostic)
- Threshold Hildmann Pillar 1.2 = (0.65, 0.95) tranché empiriquement (cf. NOTES §Pitfall 1)
- requirements.txt prêt à recevoir matplotlib + statsmodels (in-place decomment, 2 lignes annotées human-verified)

**Concerns / dependencies pour Plan 10-04 SC#1 final run :**

- `forwards_source` actuel = `fallback_diagnostic` (Mac Mini) → SC#1 ne peut PAS être satisfait en l'état (gate-filter Plan 10-04). Décision requise avant Plan 10-04 : (a) exécuter depuis FMV poste avec accès H:\ pour passer à `real_eex_xlsx`, OU (b) user override explicit acceptant le statut diagnostic-only (auquel cas D-FLIP-1 reste BLOCKED).
- Tous les autres concerns sont résolus (threshold Pillar 1.2 frozen, vintages calendar deterministic, ablation grid 4 configs prêts, conftest env hygiene déjà couvrant les 2 flags par préfixe).

## Self-Check: PASSED

Files asserted as created/modified :

- `[FOUND]` pfc_shaping/validation/block_masks.py
- `[FOUND]` pfc_shaping/validation/scorecard.py
- `[FOUND]` tests/test_phase10_infra.py
- `[FOUND]` .planning/phases/10-pfc-fmv-quality-scorecard/10-01-NOTES.md
- `[FOUND]` data/epex_hourly.parquet (61,368 rows, tz=UTC)
- `[FOUND]` data/forwards_history_phase10.parquet (1,188 records, forwards_source unique)
- `[FOUND]` pfc_shaping/validation/__init__.py (non-empty)
- `[FOUND]` pfc_shaping/requirements.txt (matplotlib line + statsmodels line, 1 occurrence each)
- `[FOUND]` .gitignore (data/*.parquet pattern)
- `[FOUND]` .planning/ROADMAP.md (PFC FMV Quality Scorecard + Phase 10B)
- `[FOUND]` .planning/REQUIREMENTS.md (BT-06..BT-10 + BT-10B-01/02)
- `[FOUND]` .planning/PROJECT.md (D-A0-* entry + D-FLIP-1 updated)

Commits asserted in git log :

- `[FOUND]` ea46ca4 (Task 2 RED test)
- `[FOUND]` a02ca10 (Task 2 GREEN feat)
- `[FOUND]` 376f1e4 (Task 3a-bis docs frozen ex-ante)
- `[FOUND]` e955a2f (Task 3a/3b/3c feat)
- `[FOUND]` e2672b4 (Task 4 docs pivot)

Test suite : 291 passed + 4 skipped (no regression vs baseline 279+4=283 — net +16 new infra tests, 4 pre-existing skips preserved).

---

*Phase: 10-pfc-fmv-quality-scorecard*
*Plan: 01 — Infrastructure scaffold + ROADMAP/REQ/PROJECT pivot D-A0-2*
*Completed: 2026-05-21*
