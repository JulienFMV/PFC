# AGENTS.md

## Permanent handoff and context hygiene rules

This file is the canonical root contract for Codex and local agents. If
`CLAUDE.md` exists, it must point here instead of duplicating rules.

Permanent project facts:

- LT code must remain independent from CT code. Do not import
  `pfc_shaping.ct.*` from LT modules.
- The CH LT monthly reform uses one monthly BASE solver with hard CH EEX
  constraints. Neighbor/history information may enter only as zero-mean shape.
- When `monthly_level_authority="solver"`, the monthly solver is the level
  authority. Hourly layers may shape within a month but must not rewrite solver
  monthly means.
- Promotion evidence must come from independent real manifests: production,
  local export, and the selected lambda artifact.
- Far-horizon `UNSUPPORTED` can be accepted only when documented and when it
  does not hide a `CRITICAL` gate or known-bad fixture failure.

Standard-user workstation execution contract:

- Before every shell action, verify that both the current directory and the
  Git top-level are exactly `C:\Users\jbattaglia\PFC_LT`. Never execute from
  the legacy `H:` checkout.
- The workstation has no administrator rights. Never request elevation, an
  ASR/Defender exception, an ACL takeover, or installation into a system
  location.
- Keep every mutable build input and output under the canonical workspace.
  In particular, place Conda prefixes, wheelhouses, pip/Conda caches,
  `TEMP`/`TMP`, pytest basetemps and runtime staging below `build/` (or another
  explicitly governed directory inside the repo), never under `AppData`,
  `ProgramData`, another user directory, or `H:`.
- Never construct or submit a shell command that names a mutable path outside
  the canonical workspace, even when the target program would reject that
  path. The VS Code/tool boundary can request approval before the program's
  own fail-closed validation runs. Negative path tests must synthesize their
  fixtures below a fresh repo-local `build/` basetemp.
- Do not request a shell sandbox override or approval for this workspace.
  Ordinary commands must use the canonical root as their working directory
  and repo-local mutable arguments. If that is impossible, record the external
  blocker instead of submitting an approval-triggering command.
- Existing interpreters, Conda executables, archives and caches outside the
  workspace (including preinstalled tools under `ProgramData`) may be read
  without mutation only to capture or copy verified bytes below `build/`.
  Subsequent mutable consumption must use the repo-local copy or prefix; never
  create or mutate an environment outside the workspace.
- Do not build or launch project `.exe` files or Playwright/browser runtimes on
  this managed workstation. Use Python module entry points and library-level
  tests. Route any required executable/browser E2E qualification to an
  independently governed standard-user CI runner.
- Ordinary read/write/test operations inside the canonical workspace must run
  without an elevation request. If an essential action truly requires an
  external writable authority, network entitlement, administrator right, or
  security-policy exception, record it as an external blocker and continue
  with safe local work; do not repeatedly ask the workstation user to approve
  it.
- Never submit a shell command that names a mutable path outside the canonical
  workspace merely to demonstrate that a guard rejects it. The host may ask
  for permission before the project guard can run. Exercise those negative
  cases only in library-level tests with synthetic ``tmp_path`` values. Every
  real command submitted on this workstation must keep its mutable paths below
  ``C:\Users\jbattaglia\PFC_LT\build`` so ordinary execution remains
  non-elevated and approval-free.
- Use `scripts.run_workspace_local` only for its explicitly allowlisted Python
  build/audit/test modules on this laptop. It is a non-authoritative convenience
  boundary, not a generic shell, a filesystem sandbox, a production-admission
  runtime or the CI runner. Conda exact-prefix creation, wheel construction and
  installed v19 admission must use their dedicated governed recipes with
  repo-local mutable paths. Independent CI uses its own checkout and policy;
  it must not reuse the laptop's literal-root harness.

Do not touch without explicit request:

- `pfc_shaping/ct/*` during LT work.
- `powerbi/data/*` or `powerbi/PFC_QA.*` unless the task is explicitly Power BI.
- Heavy desk data files such as `data/eex_forwards_history.parquet`,
  `data/epex_hourly.parquet`, `pfc_shaping/data/*.parquet`, or
  `pfc_shaping/data/*.duckdb`.
- Monthly solver production flag promotion without manifest-backed gates.
- Individual month patches after the solver. Fix specification, priors,
  objective weights, or audit gates, then regenerate.

Restricted AFRY scenario evidence contract:

- Before any AFRY, long-term scenario or AFRY-derived shaping work, read
  `.planning/phases/14-lt-audit-remediation/AFRY-CH-2026-Q2-AGENT-DATA-CONTEXT.md`
  and its referenced source, semantic and shape-diagnostic contracts.
- Access restricted AFRY values only through the current hash-verified local
  catalog or diagnostic interfaces described in that context. Never copy raw
  or derived values into Git, documentation, a wiki, RAG, embeddings or an
  external prompt.
- AFRY is descriptive benchmark/teacher-candidate evidence only. It has no
  scenario probability, Swiss calendar, monthly-level, model-input or
  production authority. The monthly solver remains the sole level authority.
- The current empirical gate is
  `BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS`. Do not start AFRY-driven
  rolling-origin selection or Batch 4 until governed EEX and ENTSO-E
  Databricks inputs and a new independently frozen future holdout exist.
  Legacy local and synthetic EEX/ENTSO-E substitution are forbidden; T057
  remains sealed.
- Durable AFRY decisions are D-20260803-206 and D-20260803-207 in the Phase 14
  decision log. Superseded catalog or diagnostic bundles must never be used
  merely because their files still exist below `build/`.

Handoff rules:

- Aim to keep active session input below about 120k tokens.
- Prepare a handoff around 60% context, not near 90%.
- Always produce or update a handoff before closing a session or phase.
- Handoffs must record exact changed files, commands, outputs, artifact paths,
  config values, hashes/manifests when available, tests, failures, and risks.
- Durable decisions must be recorded as decision / reason / rejected
  alternatives / invariants not to break.
- For Phase 14 monthly reform, use
  `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md` and
  `SESSION-HANDOFF-YYYYMMDD-*.md`.

Conventions pour les agents (Claude Code, agents locaux, contributeurs humains)
travaillant sur ce repo. Lecture obligatoire avant toute PR.

## Découpage long-terme (LT) vs court-terme (CT)

Le repo est désormais structuré pour qu'un agent puisse intervenir
exclusivement sur le LT ou sur le CT sans toucher l'autre.

### Modules

| Périmètre | Localisation | Responsabilité |
|---|---|---|
| **LT (long-terme)** | `pfc_shaping/lt/model/` | construction de la PFC 15min N+3 ans : assemblage, shaping horaire/15min, water value, MSFC spline, uncertainty, flavors |
| **LT calibration** | `pfc_shaping/calibration/` | cascading des forwards EEX, calibration arbitrage-free |
| **CT (court-terme)** | `pfc_shaping/ct/model/` | overlay J+1..J+10 : LEAR, foundation models (Chronos), expérimental PriceFM/FutureBoost |
| **Pipeline LT** | `pfc_shaping/pipeline/production_phases.py` | orchestration LT : ingestion, fits, build CH/DE |
| **Pipeline CT** | `pfc_shaping/pipeline/swiss_short_term.py` | overlay LEAR sur la base PFC CH, blends expérimentaux |
| **Partagé** | `pfc_shaping/data/`, `pfc_shaping/calendar_ch.py`, `pfc_shaping/storage/`, `pfc_shaping/tools/` | ingestion, calendrier, persistance, outils |
| **Validation** | `pfc_shaping/validation/` | backtest LT (`backtest.py`), comparaison HFC OMPEX (`compare_hfc.py`) |

### Règles d'imports

- **Code LT** importe **librement** depuis `pfc_shaping.lt.*` et depuis le
  partagé (`pfc_shaping.data.*`, `pfc_shaping.calendar_ch`,
  `pfc_shaping.storage.*`, `pfc_shaping.calibration.*`).
- **Code LT** ne doit **jamais** importer `pfc_shaping.ct.*`. La PFC long-terme
  est indépendante du modèle court-terme. Le seul point de rencontre est
  l'orchestration top-level (`run_pfc_production.py` ou
  `production_phases.py` → `swiss_short_term.py`).
- **Code CT** peut importer `pfc_shaping.ct.*` et le partagé. Il peut
  consommer une PFC LT en sortie (`base_pfc_ch`) mais ne doit pas appeler
  le pipeline LT.
- Les anciens chemins `pfc_shaping.model.<X>` continuent de fonctionner via
  un shim de compat (`pfc_shaping/model/__init__.py`) qui émet une
  `DeprecationWarning`. **Tout nouveau code doit utiliser les chemins
  `pfc_shaping.lt.model.X` ou `pfc_shaping.ct.model.X`.**

### Cartographie de migration

| Ancien chemin (déprécié) | Nouveau chemin |
|---|---|
| `pfc_shaping.model.assembler` | `pfc_shaping.lt.model.assembler` |
| `pfc_shaping.model.shape_hourly` | `pfc_shaping.lt.model.shape_hourly` |
| `pfc_shaping.model.shape_hourly_mlp` | `pfc_shaping.lt.model.shape_hourly_mlp` |
| `pfc_shaping.model.shape_intraday` | `pfc_shaping.lt.model.shape_intraday` |
| `pfc_shaping.model.water_value` | `pfc_shaping.lt.model.water_value` |
| `pfc_shaping.model.msfc_spline` | `pfc_shaping.lt.model.msfc_spline` |
| `pfc_shaping.model.uncertainty` | `pfc_shaping.lt.model.uncertainty` |
| `pfc_shaping.model.pfc_flavors` | `pfc_shaping.lt.model.pfc_flavors` |
| `pfc_shaping.model.lear_forecaster` | `pfc_shaping.ct.model.lear_forecaster` |
| `pfc_shaping.model.foundation_forecaster` | `pfc_shaping.ct.model.foundation_forecaster` |
| `pfc_shaping.model.futureboost_experimental` | `pfc_shaping.ct.model.futureboost_experimental` |
| `pfc_shaping.model.pricefm_experimental` | `pfc_shaping.ct.model.pricefm_experimental` |

## Branches actives

- `claude/audit-pfc-forwards-q73iC` — audit forwards EEX + roadmap LT
  (Phase 0 → Phase 10 dans `docs/research/forwards_sources.md`).
- `claude/refactor-lt-ct-models` — refactor LT/CT (cette PR).
- Branches CT (à venir) — séparation reporting/export/summary, hardening LEAR.

## Workflow recommandé

1. Sur le laptop FMV, travailler uniquement dans le worktree canonique
   `C:\Users\jbattaglia\PFC_LT`; le contrat standard-user ci-dessus remplace la
   recommandation générale de créer un worktree dédié.
2. Une PR = un périmètre clair. Pas de PR mixant LT et CT sauf orchestration.
3. Tests minimum :
   - LT : `pytest tests/test_arbitrage_free.py tests/test_cascading.py tests/test_water_value.py tests/test_lt_ct_imports.py`
   - CT : tests propres au CT (à enrichir).
4. Tout ajout d'un module suit le découpage : `pfc_shaping/lt/...` ou
   `pfc_shaping/ct/...`. Pas de nouveau fichier dans `pfc_shaping/model/`.

## Outils

- `scripts/phase0_sniff_forwards.py` — cadrage des sources EEX (Phase 0 LT).
- `run_pfc_production.py` — entrée production (orchestration top-level).
- `dashboard/app.py` — dashboard Streamlit (lecture seule des artefacts LT et CT).
