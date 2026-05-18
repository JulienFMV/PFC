# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-18)

**Core value:** Pricing trading-grade des blocs profil client (10-15 / 18-9)
avec ≥ 1.5 €/MWh de MAE-bloc en moins que HFC OMPEX.

**Current focus:** Phase 5bis — Shape seasonal × type_jour × hour.

## Current Position

Phase: 5bis of 11 (Phase 5bis — Shape seasonal × type_jour × hour)
Plan: 0 of TBD (à générer via `/gsd:plan-phase 5bis`)
Status: **Ready to discuss** — appeler `/gsd:discuss-phase 5bis` pour verrouiller les décisions techniques avant planning.
Last activity: 2026-05-18 — Bloc A+C1+C2 fixes commité (`28dfd65`) ; .planning/ seedée depuis Cloud Desktop.

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

- Décider lors de `/gsd:discuss-phase 5bis` :
  - Type d'indexation de `factors_` : nested dict `{(saison, type_jour): np.array[24]}` (compatible avec le format actuel + lazy upgrade) vs flat dict `{(saison, type_jour, hour): float}` ?
  - Feature flag par variable d'environnement ou paramètre constructeur `ShapeHourly(seasonal_hourly=True)` ?
  - Backward-compat : strict (legacy + nouveau coexistent) ou breaking change ?
  - Stratégie test : injection synthétique de bowl (déterministe) ou fixture EPEX réelle ?

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
