---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: État frais sur la branche `claude/clean-lt-ct-integration` @ `28dfd65`.
last_updated: "2026-05-18T19:14:45.560Z"
last_activity: 2026-05-18 -- Phase 5bis-A planning complete
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-18)

**Core value:** Pricing trading-grade des blocs profil client (10-15 / 18-9)
avec ≥ 1.5 €/MWh de MAE-bloc en moins que HFC OMPEX.

**Current focus:** Phase 5bis-A — Shape Hourly Infrastructure & Flag (no-op refactor).

## Current Position

Phase: 5bis-A of 12 (split de 5bis en 5bis-A infra + 5bis-B bowl, post panel d'experts 2026-05-18)
Plan: 0 of TBD (à générer via `/gsd:plan-phase 5bis-A`)
Status: Ready to execute
Last activity: 2026-05-18 -- Phase 5bis-A planning complete

Progress: [██████░░░░] 60% (7 phases done sur ~11)

## Performance Metrics

**Velocity:**

- Phases livrées : 7 (Phase 0, Refactor B, 1ter, 1bis, 2, Audit, Bloc A+C1+C2)
- Durée moyenne par phase : ~½ journée
- Tests : 142 passing, 4 skipped (CT-only deps)

**By Phase:**

| Phase | Commits | Files touched | Tests added |
|-------|---------|---------------|-------------|
| 0 Cadrage EEX | f0b4a10 | 2 | 0 (research) |
| Refactor B | 596e3c5 | 40 (12 git mv) | 17 |
| 1ter Parser | 8d28b63 | 4 | 30 |
| 1bis Generic build | 2aa99ea | 2 | 13 |
| 2 Negative + script | 867c51e + 0915f0e | 4 + 2 | 23 + 2 |
| Bloc A+C1+C2 | 28dfd65 | 5 | 17 |

## Accumulated Context

### Decisions

Voir `.planning/PROJECT.md` section "Key Decisions" pour la liste consolidée.
Décisions récentes affectant la phase courante :

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

Last session: 2026-05-18 (Cloud Desktop) — seeded .planning/ + committed audit fixes Bloc A+C1+C2.

Stopped at: État frais sur la branche `claude/clean-lt-ct-integration` @ `28dfd65`.
Tests verts. `.planning/` complet. Prêt pour `/gsd:discuss-phase 5bis` sur Mac Mini.

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
