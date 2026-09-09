---
phase: 10-pfc-fmv-quality-scorecard
reviewer: codex-cli-0.125.0
reviewed: 2026-05-21T00:00:00+02:00
depth: standard
files_reviewed: 12
findings:
  critical: 1
  warning: 5
  info: 1
  total: 7
status: issues_found
---

# Phase 10 Independent Code Review (Codex)

## Summary
La base est solide (guards dégénérés présents, seeds fixés, tests nombreux), mais j’ai relevé un défaut quantitatif bloquant sur la définition des périodes d’arbitrage-free en UTC au lieu du calendrier local trader, plus plusieurs modes de faux positif/sous-couverture (skips silencieux, validation Pillar 1 partielle, code mort, test-oracle faible). En l’état, le scorecard peut afficher des PASS sans couvrir correctement tout le périmètre annoncé.

## Findings

### CR-01: Arbitrage-Free Computed On UTC Calendar Instead Of Local Trading Calendar
**File:** `pfc_shaping/validation/structural_tests.py:130`
**Issue:** `_period_mask()` découpe `Cal/Quarter/Month` via `idx_utc.year/month/quarter` (`test_arb_free` l’utilise ensuite à `:196-202`). Pour une sémantique forward marché CH/DE, les frontières de période sont locales (Europe/Zurich), pas UTC. Cela crée un décalage d’1–2h aux changements de mois/trimestre/année (DST/CET/CEST) et peut fausser les moyennes de période, particulièrement critique avec `tol=0.01`.
**Fix:** Convertir l’index en `Europe/Zurich` dans `_period_mask()` avant extraction `year/month/quarter`, puis appliquer le masque résultant sur la série UTC d’origine.

### WR-01: Arb-Free Test Can Pass While Skipping Most/All Forward Keys
**File:** `pfc_shaping/validation/structural_tests.py:191`
**Issue:** `test_arb_free()` fait `continue` sur clés non parsables (`:191-194`) et masques vides (`:197-199`), puis conclut `passed=max_dev<tol` (`:215`) même si aucune clé n’a été réellement testée. `forwards={}` est explicitement PASS (`:177`). C’est un mode de faux vert silencieux.
**Fix:** Ajouter une contrainte de couverture minimale (ex. `n_tested >= 1` et/ou `%tested >= threshold`), et rendre le test FAIL/degenerate si trop de clés sont skipped.

### WR-02: Pillar 1 In Full Run Uses Only First Vintage Forwards
**File:** `pfc_shaping/validation/scorecard.py:1481`
**Issue:** Dans `run_scorecard_full()`, Pillar 1 agrège les PFC de 24 vintages (`:1476-1479`) mais teste l’arb-free uniquement contre `fwds_first` (forwards du premier vintage, `:1482-1493`). Les périodes couvertes par les vintages suivants ne sont pas validées contre leurs forwards as-of respectifs; combiné avec les skips de `test_arb_free`, cela peut masquer des violations.
**Fix:** Évaluer l’arb-free par vintage avec ses propres forwards puis agréger le max dev / taux de pass, ou reconstruire une table période→forward cohérente par timestamp de vintage.

### WR-03: Dead/No-Op Loop In Pillar 4 Baseline Build Path
**File:** `pfc_shaping/validation/scorecard.py:1568`
**Issue:** Le bloc `baseline_pieces` + boucles imbriquées `for bname ... for bloc ... pass` (`:1568-1586`) ne produit aucune donnée et n’est jamais utilisé. C’est du code mort/no-op introduisant complexité et coût inutile.
**Fix:** Supprimer ce bloc, ou l’implémenter réellement si c’était une étape de pré-calcul attendue.

### WR-04: DM Implementation Accepts Invalid Horizon Values
**File:** `pfc_shaping/validation/dm_test.py:125`
**Issue:** `diebold_mariano()` ne valide pas `h>=1`. Avec `h<=0`, `n_lags=max(h-1,0)` force un chemin artificiel et la correction HLN (`:198`) devient mathématiquement hors contrat, donnant des stats/p-values sans sens économique.
**Fix:** Ajouter `if h < 1: raise ValueError("h must be >= 1")`.

### WR-05: Claimed “Conditional Coverage” Not Implemented
**File:** `pfc_shaping/validation/christoffersen.py:4`
**Issue:** Le module n’implémente que `LR_uc` (unconditional coverage). Aucune statistique de couverture conditionnelle / indépendance des violations (LR_ind/LR_cc) n’est présente, alors que Phase 10 est présentée comme incluant la conditional coverage.
**Fix:** Soit implémenter LR_ind + LR_cc, soit corriger explicitement le scope/documentation Phase 10 pour éviter un gap d’audit.

### INFO-01: One DM Test Does Not Actually Exercise The Intended Branch
**File:** `tests/test_phase10_dm.py:125`
**Issue:** `test_var_d_negative_fallback` construit des erreurs où `|errors_a|` et `|errors_b|` sont constants (`:129-131`), ce qui tend vers variance nulle/dégénérée plutôt qu’un vrai cas de HAC négative dominant le fallback ciblé. Le test valide “no crash” mais pas la branche mathématique visée.
**Fix:** Construire explicitement une série `d_t` à autocovariances négatives contrôlées (ou mocker `acovf`) pour forcer `var_d<=0` puis vérifier la logique de fallback.
