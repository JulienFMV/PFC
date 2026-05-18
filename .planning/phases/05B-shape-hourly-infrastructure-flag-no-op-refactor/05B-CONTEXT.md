# Phase 5bis-A: Shape Hourly — Infrastructure & Flag (no-op refactor) — Context

**Gathered:** 2026-05-18
**Status:** Ready for planning
**Scope recadrage:** Suite à panel d'experts adversarial (3 reviewers indépendants : energy quant, production safety, test design — verdict unanime "disagree" sur la proposition initiale), la Phase 5bis monolithique est splittée en **5bis-A (infrastructure, ce document)** + **5bis-B (bowl-deepening, phase suivante)**. Rationale : convention quant standard "no-op refactor first, math change second" + dépendances de testing (baseline frozen, save/load complet, flag persisté) qui doivent exister AVANT toute modif comportementale du modèle.

<domain>
## Phase Boundary

**5bis-A livre l'infrastructure qui permettra de mesurer et reverter bit-pour-bit tout changement comportemental futur du `ShapeHourly`.**

Concrètement :
1. Satisfaire littéralement SHP-01 (`factors_` indexé par `(saison, type_jour, hour)`) via une **view 3D** non-intrusive (`factors_3d_` property ou `__getitem__` 3-tuple), sans changer le stockage interne (`dict[(s,tj)] → np.ndarray[24]`).
2. Compléter `ShapeHourly.save/load` pour roundtrip **tous** les attributs entraînés (bug pré-existant : `factors_by_year_`, `trend_per_hour_`, `f_W_seasonal_`, `_climatological_fill`, `sigma`, `halflife_days`, `hydro_weight_sigma` sont silently lost au reload aujourd'hui).
3. Introduire le feature flag `PFC_LT_USE_SEASONAL_HOURLY_SHAPE` correctement : constructor arg + env-default, **gelé** à `__init__`, **persisté** en sidecar parquet. Dans 5bis-A le flag existe mais ne gate aucun comportement (flag ON ≡ flag OFF, numériquement identiques).
4. Committer une baseline frozen `tests/fixtures/baseline_pfc_seed42.parquet` qui sert de référence de régression bit-pour-bit pour 5bis-A et toutes les phases shape ultérieures.
5. Améliorer le test infrastructure : `conftest.py` autouse `monkeypatch.setenv` pour `PFC_LT_*` (anti-fuite test→test), remplacer le `try/except TypeError` à `assembler.py:284` par un capability check explicite.

**In scope** :
- `pfc_shaping/lt/model/shape_hourly.py` (ajout `factors_3d_`, save/load complet, flag dans `__init__`)
- `pfc_shaping/lt/model/assembler.py` (capability check, plumbing du flag pour préparer 5bis-B — mais aucune utilisation comportementale dans 5bis-A)
- Nouveau sidecar `_meta.parquet` à côté de `f_W.parquet` / `shape_hourly.parquet`
- Nouveau `tests/fixtures/baseline_pfc_seed42.parquet` (PR séparée AVANT 5bis-A pour figer le baseline depuis `main`)
- Nouveau `tests/test_shape_hourly_infra.py` (regression bit-pour-bit, save/load roundtrip, flag freeze, view 3D)
- Nouveau ou modifié `tests/conftest.py` (env var hygiene)

**Out of scope (déférés à 5bis-B puis 5)** :
- Toute modif numérique du comportement : σ smoothing, `_apply_hydro_analogue_weights` bug fix, `shape_freedom['f_H']` damping, split f_H en level/anomaly → **Phase 5bis-B**
- Floors silencieux (MSFC `max(B,1)`, m_factor floor 0.1, F_WV_FLOOR, peak ratio ≥1) → **Phase 5**
- Backtest réel vs HFC OMPEX → **Phase 10**
- Distribution probabiliste par bloc → **Phase 5ter**
- Activation FR/AT/IT → **Phase 3 (HOLD)**
- Anything in `pfc_shaping/ct/*`

</domain>

<decisions>
## Implementation Decisions

### API surface — view 3D pour SHP-01
- **D-01** : Garder `factors_: dict[(saison, type_jour), np.ndarray[24]]` interne (status quo). Ajouter une vue lecture-seule `factors_3d_: Mapping[(saison, type_jour, hour), float]` exposée via property ou Mapping subclass. Aucun changement de structure on-disk. Justification : SHP-01 est satisfait littéralement, l'invariant `mean_h(f_H | cell) ≈ 1.0` reste trivial à vérifier sur array, le smoothing gaussien intra-cell continue d'opérer sur array natif, save/load existant continue de fonctionner identiquement.
- **D-02** : Optionnellement supporter `factors_[("Ete","Ouvrable",12)]` (clé 3-tuple) via `__getitem__` surchargé qui dispatche : 2-tuple → renvoie l'array natif, 3-tuple → renvoie le float `array[hour]`. À implémenter si ergonomique, sinon `factors_3d_` seule suffit.

### Save/load complet — fix bug pré-existant
- **D-03** : `ShapeHourly.save(path)` écrit en plus de `factors_` et `f_W_` un sidecar `_meta.parquet` (ou `*.meta.json` si plus simple) contenant **tous** les attributs entraînés non encore persistés : `factors_by_year_` (long format `saison, type_jour, year, heure, f_H`), `trend_per_hour_` (long format `saison, type_jour, heure, slope`), `f_W_seasonal_` (long format `saison, type_jour, f_W`), `_climatological_fill` (Series week→fill_pct), hyperparams scalaires (`sigma`, `halflife_days`, `hydro_weight_sigma`), et le `use_seasonal_hourly` flag effectif au moment du fit.
- **D-04** : `ShapeHourly.load(path)` détecte la présence du sidecar et restaure tous les attributs ; si absent (fichier legacy pré-5bis-A), reconstruit avec defaults + warning explicite (`logger.warning("Loading legacy shape_hourly.parquet without _meta sidecar — trend_per_hour_, factors_by_year_, f_W_seasonal_ unavailable")`).
- **D-05** : Test de non-régression : un parquet legacy fitté avec `main` actuel se recharge sans crash et produit identique sur `apply()` pour les cas où trend/seasonal_f_W n'étaient de toute façon pas appliqués (parce que vides en mémoire post-load aujourd'hui — c'est précisément le bug).

### Feature flag — mécanisme propre
- **D-06** : `ShapeHourly.__init__(self, ..., use_seasonal_hourly: bool | None = None)`. Si `None`, lit `os.getenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", "0") == "1"`. Si bool explicite (`True` ou `False`), gagne sur l'env. **L'env est lu une seule fois dans `__init__`** et stocké dans `self._use_seasonal_hourly: bool`. Jamais re-lu dans `fit()`, `apply()`, `save()`.
- **D-07** : Le flag effectif (`self._use_seasonal_hourly`) est **persisté** dans `_meta.parquet` (D-03) et **restauré** par `load()`. Empêche train/serve skew (modèle fitté avec flag=ON rechargé avec env différent en prod).
- **D-08** : Pour 5bis-A, le flag existe mais **ne gate aucun comportement numérique**. Tests assertent `ShapeHourly(use_seasonal_hourly=True).fit(...).apply(...)` ≡ `ShapeHourly(use_seasonal_hourly=False).fit(...).apply(...)` à `numpy.allclose(atol=1e-12)` près. C'est précisément ce qui permettra à 5bis-B d'ajouter des changements gated et de prouver le rollback bit-pour-bit.
- **D-09** : Le flag est documenté avec une **date de flip prévue** dans `PROJECT.md` (T+1 release cycle max, ~30 jours après le merge de 5bis-B), pour éviter la dette de flag permanente.

### Baseline snapshot — référence de régression
- **D-10** : Commit séparé **AVANT** 5bis-A (PR de 1 fichier + 1 script) : `tests/fixtures/baseline_pfc_seed42.parquet` = sortie de `assembler.build(...)` sur un fixture minimal (Cal'27, 1 mois, seed=42, forwards synthétiques fixes). Script de génération `tests/fixtures/_generate_baseline.py` documenté et reproductible.
- **D-11** : Test de régression dans 5bis-A : `assert_frame_equal(build(flag=OFF, seed=42), pd.read_parquet("tests/fixtures/baseline_pfc_seed42.parquet"), check_exact=False, atol=1e-10)`. Test parametrized OFF/ON tous deux égalent baseline. **Toute** PR future qui veut changer le comportement devra explicitement régénérer ce fixture ET justifier dans la PR pourquoi.

### Test hygiene + capability check
- **D-12** : `tests/conftest.py` (existant ou nouveau) gagne un autouse fixture qui snapshot `os.environ` pour toutes les clés `PFC_LT_*` au début de chaque test et restaure à la fin. Évite la fuite test→test sur env vars.
- **D-13** : `assembler.py:284` actuel `try: f_H = self.sh.apply(idx, cal, reference_date=reference_date, outages_forecast=outages_forecast) except TypeError:` → remplacer par un capability check explicite, e.g. `hasattr(self.sh, "accepts_outages_forecast")` ou check de signature via `inspect.signature(self.sh.apply).parameters`. Évite de masquer une vraie `TypeError` issue d'un bug interne.

### Test cases ciblés 5bis-A (tous synthétiques, fast, no real-data dep)
- **D-14** : `test_factors_3d_view_consistency` : `factors_3d_[(s,tj,h)] == factors_[(s,tj)][h]` pour tous (s,tj,h). Lecture-seule (tentative d'assignation lève).
- **D-15** : `test_save_load_full_roundtrip` : fit sur fixture synth → save → load → reload identique sur ALL attributs (factors_, factors_by_year_, trend_per_hour_, f_W_seasonal_, _climatological_fill, sigma, halflife_days, _use_seasonal_hourly, hydro_weight_sigma).
- **D-16** : `test_save_load_legacy_compat` : un parquet écrit par `main` actuel (committé comme fixture binaire) se recharge sans crash, warning émis, predictions identiques (modulo les attributs déjà manquants aujourd'hui).
- **D-17** : `test_flag_freeze_at_init` : init avec `use_seasonal_hourly=True` + env=`"0"` → `self._use_seasonal_hourly == True`. Modifier `os.environ` après init ne change rien. Init avec `use_seasonal_hourly=None` + env=`"1"` → True. Idem `"0"` → False.
- **D-18** : `test_flag_persisted_in_parquet` : fit avec flag=True → save → re-load avec env="0" → `_use_seasonal_hourly == True` (le parquet wins).
- **D-19** : `test_baseline_regression` : `build(flag=OFF) ≈ baseline_pfc_seed42.parquet` à 1e-10. Param: `build(flag=ON) ≈ baseline_pfc_seed42.parquet` aussi (5bis-A no-op).
- **D-20** : 142 tests existants restent verts (4 skipped CT-only).

### Claude's Discretion
- Format exact du sidecar (parquet sidecar vs JSON sidecar) : implémentation. Si parquet le plus naturel (déjà la convention `f_W.parquet`), pas de raison de mixer. JSON acceptable si les attributs non-tabulaires (scalaires) dominent. À trancher dans `/gsd:plan-phase`.
- Choix `factors_3d_` property pure-Python vs Mapping subclass : implémentation, pas critique.
- Capability check exact pour assembler.py:284 : `hasattr` vs `inspect.signature` vs duck-typing via try/except plus précis. Implémentation.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Roadmap & requirements
- `.planning/ROADMAP.md` §Phase 5bis-A & §Phase Details — délimitation du scope + Success Criteria.
- `.planning/REQUIREMENTS.md` SHP-01 → SHP-04 — exigences techniques explicites (note : 5bis-A satisfait SHP-01 littéralement via view 3D, SHP-04 via flag persisté ; SHP-02 et SHP-03 sont déjà satisfaits par le code existant et resteront verts).
- `.planning/PROJECT.md` Constraints — `pfc_shaping.ct.*` interdit côté LT, branch unique `claude/clean-lt-ct-integration`, 142 tests + 4 skipped baseline.

### Pre-existing predoc (superseded by this CONTEXT.md but kept for traceability)
- `.planning/phases/05bis-shape-seasonal-hourly/CONTEXT.md` — predoc D-01..D-06 proposé avant discussion. **Superseded** par ce document après panel d'experts (D-02 du predoc reposait sur une lecture incorrecte du code : "stopper smoothing across cells" mais le smoothing est déjà intra-cell only).

### Code à modifier
- `pfc_shaping/lt/model/shape_hourly.py:55-72` — `__init__` (ajout flag), :308-343 (save/load à compléter).
- `pfc_shaping/lt/model/assembler.py:280-286` — try/except `TypeError` à remplacer par capability check.

### Audit context
- `.planning/STATE.md` §Pending Todos — liste les 4 questions ouvertes que cette discussion tranche (indexation flat vs nested, mécanisme flag, backward-compat, stratégie test). Toutes répondues dans D-01..D-20 ci-dessus.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ShapeHourly.save/load` à `shape_hourly.py:308-343` — pattern parquet long-format déjà en place pour `factors_` et `f_W_`. Le sidecar `_meta.parquet` réutilise la même convention (`Path(path).with_name("_meta.parquet")`).
- `f_W_seasonal_` à `shape_hourly.py:67` — déjà calculé dans `_fit_f_W` (ligne 456-489), mais **non sauvegardé** aujourd'hui. 5bis-A le persiste sans changer le calcul.
- `_climatological_fill` à `shape_hourly.py:579` — déjà calculé dans `_apply_hydro_analogue_weights`, mais (a) jamais utilisé pour le weighting historique (bug 5bis-B), (b) non sauvegardé. 5bis-A le persiste, 5bis-B fixera l'usage.
- Pattern `try/except` autour de `self.sh.apply()` à `assembler.py:284` — sera remplacé par capability check ; signal qu'il existe un autre implémenteur (`ShapeHourlyMLP`) avec une signature différente. Vérifier `pfc_shaping/lt/model/shape_hourly_mlp.py` pour le contract.

### Established Patterns
- **Feature flag via env var** : convention déjà en place dans `pfc_shaping` (e.g., probablement quelque part dans `assembler.py` ou `pfc_flavors.py` — à confirmer en planning). Si oui, suivre la même convention pour cohérence.
- **Backward-compat shim** : `pfc_shaping.model.*` émet `DeprecationWarning`. 5bis-A doit préserver ce shim.
- **Long-format parquet** : `factors_` et `f_W_` sont déjà écrits en long format. `_meta.parquet` suit (chaque attribut a son propre groupe de lignes typées).
- **Tests synthétiques avec calendar_df** : `tests/test_long_term_branch.py` et `tests/test_country_tz_plumbing.py` exhibent le pattern de fabrication d'un `calendar_df` enrichi sans toucher au vrai parquet EPEX. Réutiliser.

### Integration Points
- `assembler.build` → `self.sh.apply()` à `assembler.py:282/286` — l'unique point d'entrée par lequel le flag pourrait gater du comportement (en 5bis-B). En 5bis-A on prépare le plumbing : `self.sh._use_seasonal_hourly` accessible en lecture depuis l'assembler, mais aucun branchement.
- `ShapeHourly.load(...)` à `shape_hourly.py:327` — invoqué par : (à vérifier) un loader dans `pfc_shaping/lt/pipeline/` ou similaire. Le bug actuel (`obj = cls()` sans args) fait que les artefacts prod fittés avec un sigma custom se rechargent en sigma=default. 5bis-A le fixe.
- `tests/conftest.py` — si n'existe pas, à créer. Si existe, ajouter l'autouse fixture sans casser les fixtures existantes.

</code_context>

<specifics>
## Specific Ideas

### Panel d'experts (3 reviewers adversarial, 2026-05-18)

Le scope de cette phase est directement issu d'une revue contradictoire spawned via `Agent(subagent_type="general-purpose")` x3 en parallèle :

1. **Energy quant expert** (verdict: disagree) — top finding: la proposition initiale "σ↓0.25 + disable shape_freedom['f_H']" est insuffisante. Le vrai coupable de l'aplatissement du bowl = `_apply_hydro_analogue_weights` qui pondère par `current_fill` au lieu de `_climatological_fill[week_of_year]` ; et `shape_freedom['f_H']` damp à 0.42 à 36 mois doit être splitté (level vs anomaly). **→ Déféré à 5bis-B.**

2. **Production safety expert** (verdict: disagree) — top finding: `__init__` est aussi appelé sans args dans `load()`, donc un modèle fitté flag=ON puis rechargé en prod avec env différent = train/serve skew silencieux. Save/load ne roundtrip pas `factors_by_year_`, `trend_per_hour_`, `f_W_seasonal_`, `sigma`, etc. → bug pré-existant. "Rollback bit-pour-bit" non testable sans baseline frozen. **→ Adressé intégralement dans 5bis-A (D-03..D-13).**

3. **Test design expert** (verdict: disagree) — top finding: synthetic bowl test avec `h12=40/h12=100` est tautologique (passe avec σ ∈ {0.25, 0.5, 1.0}). Besoin d'une baseline snapshot frozen + tests par stage (factors_ amplitude, f_H pre-damping, f_H post-damping, price_shape complet). **→ Baseline snapshot adressée dans 5bis-A (D-10..D-11) ; tests par stage seront ajoutés dans 5bis-B où il y aura du comportement à attribuer.**

### Convention quant assumed
- "No-op refactor first, math change second" — la justification du split 5bis-A/5bis-B repose sur cette convention quant standard. Le planner doit assumer qu'aucune valeur numérique de PFC ne change entre `main@28dfd65` et `main+5bis-A`. C'est testable explicitement (D-19).

### Use case business à préserver dans le mental model
- FMV vend des profile deals GRD (bloc nuit 18h-9h) + rachète production solaire (souvent WE, blocs OP1/OP2 EEX). Pricing actuel basé sur HFC OMPEX = sous-estime systématique. 250k€/deal/5€MWh d'erreur shape. **5bis-A ne corrige pas ce problème** (no-op) ; c'est 5bis-B qui le fera. Mais 5bis-A est le prérequis non-négociable.

</specifics>

<deferred>
## Deferred Ideas

### Vers 5bis-B (bowl-deepening, math change)
- **Fix bug hydro_weight** : `_apply_hydro_analogue_weights` doit utiliser `_climatological_fill[week_of_year(t)]` pour chaque date historique, pas `current_fill` global. Pour build Y+2/Y+3, idem pour la cible.
- **Split f_H = level × anomaly** : `level = mean_h(f_H_cell)`, `anomaly = f_H - level`. `shape_freedom['f_H']` damp **uniquement** le level, l'anomaly (= la signature saisonnière) survit à Y+2/Y+3.
- **σ paramétrable** : default 0.5 quand flag OFF, 0.25 quand flag ON. Lever mineur (0.5-1 €/MWh) mais bonus.
- **Tests par stage** : amplitude `np.ptp(factors_)` pré et post-fix sur fixture EPEX-like réaliste (extraire ~3 mois depuis legacy parquet si disponible localement). Assertions intermédiaires par étage du pipeline (f_H pre-damping, f_H post-damping, price_shape).
- **SC #2 du ROADMAP original** (`|Δ price_shape Été-h10-15 vs Hiver-h10-15| > 5 €/MWh`) déféré à 5bis-B sur EPEX réel + Phase 10 sur HFC OMPEX.

### Vers 5 (PFC peut être négative)
- Tous les floors silencieux : MSFC `max(B,1)`, m_factor floor 0.1, F_WV_FLOOR, peak ratio ≥1.

### Vers 5ter (distribution prob)
- `pfc_block_distribution(...)` Monte-Carlo shape, N=500.

### Vers 10 (refondu — backtest)
- Harness backtest par bloc client vs HFC OMPEX. Cible Δ MAE ≤ -1.5 €/MWh. Nécessite accès `H:\` (poste FMV, pas Mac Mini Cloud Desktop).

### Vers ROADMAP backlog (pas de phase assignée)
- Cleanup pre-doc `.planning/phases/05bis-shape-seasonal-hourly/` — soit supprimer (info historique perdue), soit ajouter une note "SUPERSEDED, see PFC-LT-05B-shape-seasonal-type-jour-hour/". À trancher en planning ou commit final.
- Date de flip prévue du flag `PFC_LT_USE_SEASONAL_HOURLY_SHAPE` : à inscrire dans `PROJECT.md` lors du merge de 5bis-B, default T+30 jours.

</deferred>

---

*Phase: 5bis-A — Shape Hourly Infrastructure & Flag (no-op refactor)*
*Context gathered: 2026-05-18*
