# AGENTS.md

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

1. Travailler **dans un worktree dédié** par périmètre (cf. `git worktree add`).
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
