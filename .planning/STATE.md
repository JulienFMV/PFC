---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: ready_to_plan
stopped_at: Phase 10 complete (4/4) — ready to discuss Phase 10B
last_updated: 2026-05-21T11:12:28.288Z
last_activity: 2026-05-21 -- Phase 10 execution started
progress:
  total_phases: 5
  completed_phases: 1
  total_plans: 7
  completed_plans: 15
  percent: 20
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-18)

**Core value:** Pricing trading-grade des blocs profil client (10-15 / 18-9)
avec ≥ 1.5 €/MWh de MAE-bloc en moins que HFC OMPEX.

**Current focus:** Phase 10B — pfc fmv vs hfc ompex block mae benchmark (deferred)

## Current Position

Phase: 10B
Plan: Not started
Status: Ready to plan
Last activity: 2026-05-21

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**

- Phases livrées : 7 (Phase 0, Refactor B, 1ter, 1bis, 2, Audit, Bloc A+C1+C2)
- Durée moyenne par phase : ~½ journée
- Tests : 239 passing, 3 skipped (32 tests infra 05B-02 + 26 tests flag 05B-03 + 38 tests view+capability-check 05B-04 + 8 tests mises à jour)

**By Phase:**

| Phase | Commits | Files touched | Tests added |
|-------|---------|---------------|-------------|
| 0 Cadrage EEX | f0b4a10 | 2 | 0 (research) |
| Refactor B | 596e3c5 | 40 (12 git mv) | 17 |
| 1ter Parser | 8d28b63 | 4 | 30 |
| 1bis Generic build | 2aa99ea | 2 | 13 |
| 2 Negative + script | 867c51e + 0915f0e | 4 + 2 | 23 + 2 |
| Bloc A+C1+C2 | 28dfd65 | 5 | 17 |
| Phase 05B P02 | 18m | 2 tasks | 2 files |
| Phase 05B P03 | 12m | 2 tasks | 2 files |
| Phase 05B P04 | 18m | 2 tasks | 3 files |

## Accumulated Context

### Decisions

Voir `.planning/PROJECT.md` section "Key Decisions" pour la liste consolidée.
Décisions récentes affectant la phase courante :

- **2026-05-18 (05B-01)** : Contract régression = numerical-equality (`atol=1e-12, rtol=0`) pas byte-équivalence parquet — documenté dans README.md et script docstring.
- **2026-05-18 (05B-01)** : `ShapeIntraday` fitté sur données synthétiques (seed=42) car `PFCAssembler.build()` appelle `self.si.apply()` sans None guard — workaround documenté dans le script.
- **2026-05-18 (05B-02)** : Meta sidecar `shape_hourly.meta.parquet` — toutes les valeurs stockées en string (`repr(float)` / JSON) pour éviter `pyarrow.ArrowInvalid` sur colonnes à types mixtes. Parsées à la lecture via `.apply(float)` / `json.loads()`.
- **2026-05-18 (05B-02)** : `global_factors_` non persisté dans le sidecar — reconstruit déterministiquement depuis `factors_` via `_compute_global_fallback()` à la lecture. Élimine le risque de drift state dupliqué.
- **2026-05-18 (05B-03)** : `_FLAG_ENV_VAR` exporté au niveau module (importable dans les tests sans hardcoder la chaîne). `_resolve_flag()` private helper centralise la logique de précédence — seul callsite `os.getenv`.
- **2026-05-18 (05B-04)** : `_Factors3DView(Mapping)` facade stocke une référence (pas une copie) à `factors_` — vue live, zéro arithmétique. `factors_3d_` property construite à la demande (thin wrapper, pas de cache sur l'instance).
- **2026-05-18 (05B-04)** : `inspect.signature(type(self.sh).apply)` à `PFCAssembler.__init__` remplace `try/except TypeError` (D-13). TypeError de bugs internes à `sh.apply()` propagent maintenant correctement. Cache `_sh_accepts_outages` une seule fois à l'init.
- **2026-05-18 (05B-04)** : One-shot `logger.info("Detected sh=... — outages_forecast passed/skipped")` à `PFCAssembler.__init__` — audit opérateur, pas de spam par `build()` call.
- **2026-05-18 (05B-03)** : `use_seasonal_hourly` ajouté aux hyperparams JSON du sidecar (sort_keys=True) — schema étendu depuis 05B-02. Tests 05B-02 mis à jour pour inclure la nouvelle clé.

- **2026-05-18** : Hold Phase 3 (FR/AT/IT) et Phase 4 (basis cross-border)
  pour prioriser Phase 5bis/5/5ter/10. Justification : business case profile
  deal (>250k€/5€MWh d'erreur) écrase l'urgence multi-marché.

- **2026-05-18** : Cible KPI = Δ MAE bloc ≤ -1.5 €/MWh vs HFC OMPEX sur 2024-2025.
  KPI métier directement parlant au desk FMV.

- **2026-05-18** : Adoption GSD framework pour le workflow LT (alignement avec
  Mint chez FMV). `.claude/` gitignored (machine-specific), `.planning/` versionné.

### Pending Todos

Discussion 5bis-A close (2026-05-18). Décisions verrouillées dans `.planning/phases/PFC-LT-05B-shape-seasonal-type-jour-hour/05B-CONTEXT.md` D-01..D-20. Résumé :

- Indexation `factors_` : nested dict status quo + view `factors_3d_` lecture-seule pour SHP-01 littéral.
- Feature flag : constructor arg + env-default, gelé à `__init__`, persisté en parquet sidecar `_meta.parquet`.
- Backward-compat : roundtrip complet (bug pré-existant `factors_by_year_`, `trend_per_hour_`, etc. lost au load → fixé).
- Stratégie test : synthétique deterministe + baseline frozen `tests/fixtures/baseline_pfc_seed42.parquet` committée AVANT 5bis-A, test param OFF vs ON `numpy.allclose(atol=1e-12)`, conftest autouse env-var hygiene.

Prochaine action : `/gsd:plan-phase 5bis-A` (estimation 3-5 plans atomiques + 1 PR séparée pour baseline snapshot).

Après livraison 5bis-A : `/gsd:discuss-phase 5bis-B` pour le bowl-deepening (hydro_weight bug fix + split f_H level/anomaly + σ paramétrable).

### Blockers/Concerns

- Pas d'accès au HFC OMPEX historique depuis l'env Cloud Desktop (chemin `H:\…`). Le backtest réel (Phase 10) doit tourner depuis le poste FMV avec accès `H:\`.
- L'agent sur Mac Mini peut développer Phase 5bis localement avec données synthétiques ; le backtest acceptance doit tourner sur FMV ou être déplacé sur le worktree FMV.

## Deferred Items

| Category | Item | Status | Deferred At |
|---|---|---|---|
| Multi-market | Phase 3 (FR/AT/IT activation) | HOLD | 2026-05-18 |
| Cross-border | Phase 4 (basis CH-DE) | HOLD | 2026-05-18 |
| Hydro | Phase 6 (stochastic water value) | v2 | 2026-05-18 |
| Data governance | Phase 7 (TTF/EUA/API2 governed) | v2 | 2026-05-18 |
| Multi-market calib | Phase 8 (joint multi-zone) | v2 | 2026-05-18 |

## Session Continuity

Last session: 2026-05-20T18:48:51.992Z

Stopped at: Phase 10 context gathered (reframed scorecard, OMPEX deferred to 10B)

Pour reprendre sur Mac Mini :

```bash
cd ~/PFC   # ou chemin réel du repo local
git fetch origin
git checkout claude/clean-lt-ct-integration
git pull origin claude/clean-lt-ct-integration
pytest tests/   # sanity, doit être vert
npx get-shit-done-cc@latest --local --claude --profile=standard

# Puis dans Claude Code :

/gsd:resume-work

# Ou directement :

/gsd:discuss-phase 5bis
```
