# Phase 5bis-A: Shape Hourly Infrastructure - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-18
**Phase:** 5bis-A — Shape Hourly Infrastructure & Flag (no-op refactor)
**Areas discussed:** Cible comportementale, Format de stockage factors_, Feature flag + backward-compat, Tests + interactions f_W_seasonal/shape_freedom, Recadrage post-panel d'experts

---

## Préambule — context loading

- Phase initiale (avant discussion) : "Phase 5bis: Shape seasonal × type_jour × hour" en ROADMAP.md ligne 30-32.
- Prior CONTEXT.md trouvé à `.planning/phases/05bis-shape-seasonal-hourly/CONTEXT.md` (predoc D-01..D-06).
- Code scout : `pfc_shaping/lt/model/shape_hourly.py` (645 lignes), `pfc_shaping/lt/model/assembler.py` (1022 lignes).
- **Critical finding au scout** : le code actuel a déjà `factors_: dict[(saison, type_jour)] → np.ndarray[24]`, donc `factors_[("Ete","Ouvrable")][12] != factors_[("Hiver","Ouvrable")][12]` est déjà vrai. Le predoc D-02 ("stopper smoothing across cells") repose sur une lecture incorrecte du code — le smoothing gaussien est déjà intra-cell only.

---

## Axe 1 — Cible comportementale réelle

| Option | Description | Selected |
|--------|-------------|----------|
| Refactor d'API uniquement | factors_ flat 3D, comportement bit-pour-bit | initialement écarté |
| Refactor + creuser le bowl | σ↓ + disable shape_freedom['f_H'] | ✓ (initialement) |
| Refactor + retirer floors | Scope creep Phase 5 | rejeté |
| Discuter live | Aucune des trois | |

**User's choice (initial) :** "J'en sais rien très clairement, c'est toi l'expert, donc c'est toi qui va prendre les bons choix."

**Notes:** Context business critique fourni par le user — FMV vend profile deals GRD (bloc nuit 18h-9h) + rachète solaire WE (blocs OP1/OP2 EEX). PFC OMPEX = sous-estime systématique (~250k€/deal/5€MWh shape error). Cible 5bis = creuser duck curve réellement, pas juste cosmétique.

**Décision experte initiale (Claude)** : Option 2 (refactor + creuser le bowl, sans toucher floors).

---

## Axe 2 — Format de stockage `factors_`

| Option | Description | Selected |
|--------|-------------|----------|
| dict[(s,tj)] + vue 3D | Status quo interne + property factors_3d_ | ✓ |
| Flat dict pur | {(s,tj,h): float}, refactor save/load | |
| 3D numpy array | factors_arr[s,tj,h], plus rapide | |

**User's choice :** OK — dict[(s,tj)] + vue 3D.

**Notes:** Préserve smoothing array-natif, mean=1 invariant trivial, save/load existant inchangé. Confirmé final dans 5bis-A (D-01/D-02).

---

## Axe 3 — Feature flag + backward-compat parquet

| Option | Description | Selected |
|--------|-------------|----------|
| Constructor arg + env default | use_seasonal_hourly=None lit env si None | (révisé après panel) |
| Env var seule | Plus minimal, moins test-friendly | |
| Pas de flag | Refactor direct, rollback = git revert | |

**User's choice :** "J'aimerais que tu prennes un panel d'experts pour questionner ses choix et challenger ses choix, s'il te plaît."

**Notes:** Le user a demandé un panel adversarial avant de figer cette décision. Trois agents general-purpose spawnés en parallèle (energy quant, production safety, test design). Verdict unanime: disagree avec la proposition initiale. Voir section "Recadrage post-panel" ci-dessous.

---

## Axe 4 — Interaction f_W_seasonal_ + shape_freedom + tests

Cet axe devient mostly N/A pour 5bis-A après le split (ni `f_W_seasonal_` ni `shape_freedom['f_H']` ne sont touchés). Seule la stratégie test reste pertinente :

| Sous-décision | Choix retenu |
|--------|-------------|
| Synthétique seul vs + EPEX réel | Synthétique seul (Mac Mini, no HFC OMPEX access) + baseline snapshot |
| Baseline frozen | Oui — committée AVANT 5bis-A, fixture `tests/fixtures/baseline_pfc_seed42.parquet` |
| Test param OFF vs ON | Oui — `numpy.allclose(atol=1e-12)` car 5bis-A no-op |
| Save/load roundtrip | Oui — complet sur tous attributs (D-15) |
| Hygiene env var | autouse conftest fixture `monkeypatch.setenv` PFC_LT_* |

**Notes:** Tests par stage (assertions intermédiaires) déférés à 5bis-B où il y aura du comportement à attribuer.

---

## Recadrage post-panel d'experts

### Panel adversarial (3 agents, parallèle, 2026-05-18)

**Energy quant expert — verdict : disagree.**
> Top finding: σ=0.5→0.25 est mineur (0.5-1 €/MWh), pas suffisant pour SC #2. Le vrai coupable = `_apply_hydro_analogue_weights` à `shape_hourly.py:584,607` utilise `current_fill` au lieu de `_climatological_fill[week_of_year]` calculé à la ligne 579. Disabling `shape_freedom['f_H']` brut = trop crude ; splitter f_H = level × anomaly, damp level only.

**Production safety expert — verdict : disagree.**
> Top finding: `__init__` est appelé sans args dans `load()` (shape_hourly.py:331) → train/serve skew silencieux. Save/load ne roundtrip pas σ, halflife, f_W_seasonal_, factors_by_year_, trend_per_hour_, _climatological_fill, etc. → bug pré-existant. "Bit-pour-bit rollback" non-testable depuis 142 tests actuels (aucun ne touche env var).

**Test design expert — verdict : disagree.**
> Top finding: synthetic bowl test (h12=40/h12=100) passe avec σ ∈ {0.25, 0.5, 1.0} → tautologique, pas une regression gate. SC #3 (rollback bit-pour-bit) non-falsifiable sans baseline frozen committée AVANT 5bis-A. End-to-end test masque attribution: besoin d'assertions par stage (factors_, f_H pre-damping, f_H post-damping, price_shape complet).

### Synthèse + nouvelle option présentée au user

| Option recadrage | Description | Selected |
|--------|-------------|----------|
| Adopter tout le panel, 5bis monolithique | Scope x2 (1-2 jours sup) | |
| Split 5bis-A (infra) + 5bis-B (bowl) | Convention "no-op refactor first" | ✓ |
| Min viable + spike 5quinquies | Punt encore le vrai problème | |
| Discuter live | Point précis avant choix | |

**User's choice :** "a" (procède avec split).

**Notes:** Décision finale = split 5bis-A / 5bis-B. Justification: les findings panel s'ordonnent logiquement — impossible de faire le bowl-deepening (modeling expert) sans avoir d'abord les outils de mesure et de rollback bit-pour-bit (production safety + test design experts). Convention quant standard "no-op refactor first, math change second".

---

## Claude's Discretion

Confirmé par le user :
- Format exact sidecar (`_meta.parquet` vs JSON) — à trancher en planning
- `factors_3d_` property vs Mapping subclass — implémentation
- Capability check exact pour `assembler.py:284` (`hasattr` vs `inspect.signature`) — implémentation

## Deferred Ideas

Voir CONTEXT.md section `<deferred>` pour la liste complète. Résumé :
- **5bis-B** : fix hydro_weight bug, split f_H level/anomaly, σ paramétrable, tests par stage, SC #2 sur EPEX réel
- **Phase 5** : retirer floors silencieux (NEG-01..NEG-05)
- **Phase 5ter** : distribution probabiliste par bloc Monte-Carlo
- **Phase 10** : backtest par bloc vs HFC OMPEX (nécessite accès `H:\` poste FMV)
- **ROADMAP backlog** : cleanup pre-doc legacy `.planning/phases/05bis-shape-seasonal-hourly/` ; inscrire date de flip flag dans PROJECT.md
