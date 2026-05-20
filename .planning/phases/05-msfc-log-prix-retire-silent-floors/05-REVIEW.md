---
phase: 05-msfc-log-prix-retire-silent-floors
reviewed: 2026-05-20T12:00:00Z
depth: standard
files_reviewed: 13
files_reviewed_list:
  - pfc_shaping/calibration/arbitrage_free.py
  - pfc_shaping/calibration/cascading.py
  - pfc_shaping/lt/model/assembler.py
  - pfc_shaping/lt/model/msfc_spline.py
  - pfc_shaping/lt/model/water_value.py
  - pfc_shaping/pipeline/autoresearch.py
  - pfc_shaping/pipeline/production_phases.py
  - pfc_shaping/pipeline/rolling_update.py
  - tests/conftest.py
  - tests/fixtures/_generate_phase05_fixture.py
  - tests/test_phase05_negative_prices.py
  - tests/test_shape_hourly_bowl.py
  - tests/test_shape_hourly_infra.py
findings:
  critical: 2
  warning: 9
  info: 6
  total: 17
status: issues_found
---

# Phase 5 : Code Review Report

**Reviewed:** 2026-05-20T12:00:00Z
**Depth:** standard
**Files Reviewed:** 13
**Status:** issues_found

## Summary

Phase 5 refactorise PFC LT pour autoriser les prix négatifs : retrait des quatre planchers silencieux (MSFC double floor, ArbitrageFreeCalibrator `m_factor` clamp, WaterValueCorrection inversion multiplicative, BlockCascading inversion ratio) au profit de variantes ctor-gated et d'un chemin delta-additif sign-invariant pour WV et spread-additif pour Peak. Le master flag `PFC_LT_ALLOW_NEGATIVE_PRICES` est correctement audit-log only.

Globalement, l'implémentation est rigoureuse, bien documentée, traçable et bien testée. La discipline « default OFF + opérateur rollback explicite » est cohérente d'un module à l'autre.

Cependant la revue adversariale a relevé :
- **2 BLOCKER** : (1) un Floor #2 résiduel **toujours actif** dans `_enforce_mean_constraints` qui contredit le contrat NEG-01 / Pitfall 1, (2) un bug logique dans `autoresearch.evolve()` qui marque systématiquement les itérations comme « keep » dans l'historique.
- **9 WARNING** principalement autour de la robustesse (NaN propagation, divisions par très petits diviseurs, claims de docstring non vérifiés, gestion `warnings.resetwarnings()` qui efface les filtres globaux, mutation silencieuse des sub-components, etc.).
- **6 INFO** sur la qualité (logs corrompus mojibake, asserts en code de production, alias non utilisé, etc.).

## Critical Issues

### CR-01 (BLOCKER): Floor #2 résiduel toujours appliqué dans `_enforce_mean_constraints` à l'étape itérative

**File:** `pfc_shaping/lt/model/msfc_spline.py:205-236`
**Issue:**
Le contrat Phase 5 D-A2-1 / NEG-01 (RESEARCH Pitfall 1) impose que **les deux** floors (Floor #1 ligne 159 dans `smooth_base_prices` et Floor #2 ligne 245 dans `_enforce_mean_constraints`) soient désactivés sous `enforce_positivity=False`. La docstring de `_enforce_mean_constraints` (lignes 196-198) affirme : « The iterative correction `error * 0.8` is sign-invariant by construction (D-A1-3) and does NOT require this flag. »

Cette affirmation est correcte pour la correction additive elle-même, mais le **résultat retourné** à la dernière itération du boucle (lignes 243-245) est :
```python
interpolator = PchipInterpolator(x_knots, y_adjusted, extrapolate=True)
result = interpolator(x_target)
return np.maximum(result, 1.0) if enforce_positivity else result
```
ce qui est bien désactivé pour le retour final — OK.

**Mais à chaque itération interne** (lignes 207-208) :
```python
interpolator = PchipInterpolator(x_knots, y_adjusted, extrapolate=True)
B_current = interpolator(x_target)
```
le `B_current` utilisé pour calculer `actual_mean` (ligne 222) ne reçoit aucun floor — c'est correct.

**Cependant le commentaire ligne 206** affirme « (floor applied AFTER mean computation) » ce qui est faux : `B_current` n'est **jamais** flooré à l'intérieur de la boucle. Ce commentaire trompeur masque la conformité réelle.

Plus inquiétant — vérification croisée avec `test_msfc_signed_monthly_repricing` (lignes 99-149 de `test_phase05_negative_prices.py`) qui teste précisément que `mean(B_smooth[July 2027]) ≈ -2.0 EUR/MWh` :  le test exige tolérance ≤ 0.01. Or à la **convergence** de la boucle itérative, `max_error < 0.01` (ligne 233). Avec un knot dont la valeur converge vers ~ -2.0 et un knot voisin à 22 EUR/MWh, la sortie PCHIP du timestamp moyen de juillet est susceptible de varier autour de -2.0 ; le test passe car la boucle ajuste `y_adjusted[i]` directement (ligne 231). Néanmoins **l'absence de floor à l'intérieur de la boucle ET sur le retour final est correcte**.

Le vrai défaut : la **double formulation des deux floors** ne couvre pas un **3e clip implicite** introduit à la ligne 143 par `np.clip(B_smooth_raw, y_knots.min() - margin, y_knots.max() + margin)`. Pour les knots `{-30, -28, ...}` ce clip [-35, -15] borne effectivement le résultat **sans qu'aucun flag ne le gate**. Ce clip n'est pas dans le périmètre des « quatre planchers silencieux » de la phase, mais il fonctionne comme un floor implicite : pour des knots tous positifs, le `y_knots.min() - margin` peut empêcher des valeurs négatives en extrapolation, contredisant le contrat « negative-ready ». Aucun warning n'est émis si le clip mute des valeurs.

**Fix:**
1. Corriger le commentaire mensonger ligne 206 (`floor applied AFTER mean computation`).
2. Ajouter télémétrie sur le `np.clip` ligne 143 : compter le nombre de timestamps mutés et logger un WARNING si > 0 quand `enforce_positivity=False`, pour aligner le contrat audit-trail de la phase.
3. Documenter explicitement dans la docstring que le clip signed-aware d'extrapolation est un **3e plancher implicite** (non listé dans les quatre floors retirés) et qu'il est par construction sign-invariant — mais pas gated par le flag.

```python
# msfc_spline.py:142-143 — ajouter télémétrie
margin = max(0.5 * float(np.ptp(y_knots)), 1.0)
lo, hi = y_knots.min() - margin, y_knots.max() + margin
B_smooth_raw_pre = B_smooth_raw.copy()
B_smooth_raw = np.clip(B_smooth_raw, lo, hi)
n_clipped = int(np.sum(B_smooth_raw_pre != B_smooth_raw))
if n_clipped > 0 and not enforce_positivity:
    logger.info(
        "MSFC extrapolation clamp muted %d timestamps to [%.2f, %.2f] "
        "(third implicit clamp, not part of the four Phase 5 floors)",
        n_clipped, lo, hi,
    )
```

---

### CR-02 (BLOCKER): Bug logique `autoresearch.evolve` — toutes les itérations sont marquées « keep » dans l'historique

**File:** `pfc_shaping/pipeline/autoresearch.py:480-524`
**Issue:**
À la ligne 480, le test `if trial.rmse < current_rmse:` détermine si on garde le trial. **Si on le garde**, ligne 482, `current_rmse = trial.rmse`.

Ensuite, ligne 519-520, l'historique est enregistré :
```python
"action": "keep" if trial.rmse < current_rmse + 0.001 else "revert",
"rmse_before": float(current_rmse if trial.rmse < current_rmse + 0.001 else current_rmse),
```

**Problème** : après l'assignation ligne 482, `current_rmse == trial.rmse`, donc `trial.rmse < current_rmse + 0.001` est **toujours vrai** (puisque `trial.rmse < trial.rmse + 0.001`). Dans la branche `revert` (lignes 492-500), `current_rmse` est inchangé, mais la condition `trial.rmse < current_rmse + 0.001` peut TOUJOURS être vraie pour des valeurs proches (différence < 1e-3), donc on enregistrera également « keep » alors qu'on a effectivement revert.

Conséquence : l'historique JSON enregistre l'« action » comme « keep » pour pratiquement toutes les itérations, ce qui :
1. fausse l'audit de Darwinian evolution (sur lequel dashboards/monitoring s'appuient),
2. fait que `rmse_before` retourne `current_rmse` (le nouveau, post-update) au lieu du **vrai** rmse_before, faussant le delta affiché.

Ceci est un bug existant ET un défaut dans le travail Phase 5 si l'autoresearch est utilisé en CI/operation.

**Fix:**
Stocker `rmse_before` avant la branche keep/revert et déterminer `action` à partir d'un drapeau explicite :
```python
# autoresearch.py:479-524 — refactor
rmse_before_iter = current_rmse
action = "revert"
if trial.rmse < current_rmse:
    action = "keep"
    delta = current_rmse - trial.rmse
    current_rmse = trial.rmse
    baseline = trial
    improvements += 1
    worst_agent.n_improvements += 1
    worst_agent.rmse_contribution = trial.per_agent_rmse.get(worst_name, 0)
    logger.info(
        "KEEP: RMSE %.2f -> %.2f (delta=%.3f) [%s]",
        rmse_before_iter, current_rmse, delta, worst_name,
    )
else:
    worst_agent.params = old_params
    reverts += 1
    worst_agent.n_reverts += 1
    logger.info(
        "REVERT: trial RMSE %.2f >= current %.2f [%s]",
        trial.rmse, current_rmse, worst_name,
    )

# ...
self.history.append({
    "iteration": iteration,
    "target_agent": worst_name,
    "action": action,
    "rmse_before": float(rmse_before_iter),
    "rmse_after": float(trial.rmse),
    # ...
})
```

## Warnings

### WR-01: `warnings.resetwarnings()` dans le `finally` efface les filtres globaux du processus

**File:** `pfc_shaping/calibration/arbitrage_free.py:702-789`, `816-884`
**Issue:**
Aux lignes 702 et 823, le code installe un filter `warnings.filterwarnings("ignore", category=RuntimeWarning)` puis dans le `finally` (lignes 788-789, 883-884) appelle `warnings.resetwarnings()`. Cette dernière fonction **purge tous les filtres globaux**, y compris ceux installés par d'autres modules (pytest, urllib3, deprecations utilisateur, etc.). Cela peut faire échouer les tests `pytest.warns(DeprecationWarning, match="...")` (par ex. `test_fit_peak_ratios_deprecated`) si le solveur arbitrage-free est appelé entre les deux.

Conséquences observables :
- DeprecationWarning silenced après un appel à `calibrate` → tests sensibles aux warnings deviennent flaky.
- Filtres utilisateur (par exemple `-W error::ResourceWarning`) écrasés.

**Fix:** utiliser `warnings.catch_warnings()` comme context manager, qui sauvegarde et restaure correctement l'état :
```python
import warnings as _w
with _w.catch_warnings():
    _w.filterwarnings("ignore", category=RuntimeWarning)
    # ... solve ...
# état précédent restauré automatiquement à la sortie du with
```

---

### WR-02: Précondition assert sur `delta_wv.index` exécutée en prod

**File:** `pfc_shaping/lt/model/assembler.py:559-565`
**Issue:**
```python
assert delta_wv.index.equals(B.index), (
    f"compute_delta_wv returned mismatched index: ..."
)
```
Un `assert` est **éliminé** quand Python est exécuté avec `-O` (optimisation, fréquent en environnement de production conteneurisé). La précondition disparaît silencieusement et un bug d'index misalignment ne sera plus détecté.

**Fix:** remplacer par une exception explicite :
```python
if not delta_wv.index.equals(B.index):
    raise RuntimeError(
        f"compute_delta_wv returned mismatched index: "
        f"delta_wv.index has {len(delta_wv.index)} entries, "
        f"B.index has {len(B.index)} entries; "
        f"first 3 delta_wv: {list(delta_wv.index[:3])}; "
        f"first 3 B: {list(B.index[:3])}"
    )
```

---

### WR-03: Mutation silencieuse des sub-components dans `PFCAssembler.__init__`

**File:** `pfc_shaping/lt/model/assembler.py:287-292`
**Issue:**
```python
if self.calibrator is not None and self.enforce_m_factor_floor:
    self.calibrator.enforce_m_factor_floor = True  # override sub-component default per D-A2-3
if self.wv is not None and self.enforce_floor:
    self.wv.enforce_floor = True
if self.cascader is not None and not self.allow_negative_peak:
    self.cascader.allow_negative_peak = False
```
Le constructeur PFCAssembler **mute les attributs publics** des sous-composants passés par référence. Si le même `cascader` (ou `wv`, `calibrator`) est partagé entre **plusieurs** PFCAssembler (cas standard dans `production_phases.py` où `cascader_ch` est partagé entre branches CH/DE), la mutation du dernier assembler instantié écrasera l'état des assemblers précédents. Particulièrement dangereux : la mutation est **one-way** — un PFCAssembler avec `enforce_m_factor_floor=False` ne réinitialise PAS l'attribut à False sur le sub-component.

Plus subtil : un opérateur qui passe `cascader.allow_negative_peak=False` puis instantie PFCAssembler avec le defaut `allow_negative_peak=True` ne déclenche **pas** la mutation (parce que `not True` est False), donc le cascader garde son `allow_negative_peak=False`, mais l'audit log ligne 316 affiche `cascading_neg_peak_on = self.allow_negative_peak and bool(getattr(self.cascader, "allow_negative_peak", True))` = `True and False = False`. L'audit log est correct mais la sémantique « default-on-PFCAssembler » est cassée — l'opérateur croit avoir activé le path negative-ready mais le sub-component reste en legacy.

**Fix:** rendre la propagation bidirectionnelle ET idempotente, ou supprimer la mutation silencieuse et exiger que la cohérence soit garantie au site d'appel :
```python
# Option A — propagation bidirectionnelle
if self.calibrator is not None:
    self.calibrator.enforce_m_factor_floor = self.enforce_m_factor_floor
if self.wv is not None:
    self.wv.enforce_floor = self.enforce_floor
if self.cascader is not None:
    self.cascader.allow_negative_peak = self.allow_negative_peak

# Option B — n'autoriser que la mutation upward (rollback explicit) et logger un warning
#   si sub-component a une valeur différente sans override actif.
```

---

### WR-04: La docstring « ratio >= 1 constraint » pour le path legacy est trompeuse

**File:** `pfc_shaping/calibration/cascading.py:506-509, 574-575`
**Issue:**
La docstring annonce :
> **`allow_negative_peak=False`** (legacy rollback per D-A4-3): historical multiplicative `peak = base × ratio` with implicit `Peak >= Base` (ratio >= 1 constraint).

Aucune assertion `ratio >= 1` n'est appliquée dans `synthesize_peak_prices` (lignes 595-605). L'invariant n'est respecté que **si les ratios viennent du shim `fit_peak_ratios`** (formule `1.0 + spread/max(base,1.0)` qui peut donner un ratio < 1 si le spread fitté est négatif — possible pour certains marchés). Si l'opérateur set manuellement `cascader.peak_base_ratios_ = {m: 0.8}` (ce qui n'est pas interdit), le code marche silencieusement.

De plus, dans la branche `allow_negative_peak=False` avec **base négatif**, `peak = base × ratio` est sign-invertant : pour `base=-10, ratio=1.05` → `peak=-10.5`, donc Peak < Base — **violation** du contrat « ratio >= 1 → Peak >= Base ». Cette formule est fondamentalement incompatible avec des prix de base négatifs.

**Fix:**
1. Soit raise `ValueError` quand `allow_negative_peak=False` et au moins un `base_price` est négatif (cohérent avec l'esprit du rollback legacy).
2. Soit clamp le résultat : `result[peak_key] = max(price * ratio, price)` (préserve le contrat Peak >= Base même quand base < 0).
3. Soit retirer la mention « ratio >= 1 constraint » de la docstring si l'invariant n'est pas réellement enforcé.

---

### WR-05: Default `enforce_positivity=False` mais kwarg pas validé en bool

**File:** `pfc_shaping/lt/model/msfc_spline.py:39-44, 173-183`
**Issue:**
Le kwarg `enforce_positivity: bool = False` n'est pas validé ; si quelqu'un passe accidentellement `enforce_positivity=None`, `enforce_positivity="True"`, ou tout objet truthy non-bool, l'évaluation `if enforce_positivity:` peut produire des résultats inattendus (`"False"` est truthy en Python). Le test `test_msfc_signed_monthly_repricing` ne couvre que le cas `enforce_positivity=False` et `True`.

Comparez avec assembler.py:278-281 où la coercion `bool(enforce_positivity)` est faite explicitement — pas le cas dans `smooth_base_prices`.

**Fix:** coercer en bool en début de fonction :
```python
def smooth_base_prices(idx, base_prices, B_flat, enforce_positivity: bool = False) -> pd.Series:
    enforce_positivity = bool(enforce_positivity)
    # ... reste de la fonction
```

---

### WR-06: Division potentielle par très petite valeur dans `compute_delta_wv` indirectement via `apply` → `mean_f`

**File:** `pfc_shaping/lt/model/water_value.py:416-418`
**Issue:**
```python
mean_f = float(raw_f_wv.loc[year_mask].mean())
if abs(mean_f) > 1e-8:
    f_wv.loc[year_mask] = raw_f_wv.loc[year_mask] / mean_f
```
Le guard `abs(mean_f) > 1e-8` est correct pour `mean_f` exactement zéro, mais pour `mean_f ≈ 1e-7` la division reste effectuée et `raw_f_wv / 1e-7` produit des facteurs de l'ordre de 1e7 → catastrophique pour la suite (delta_wv = (f_wv - 1) × |B| explose). Aucun WARNING n'est émis.

Le cas pathologique correspond à : `enforce_floor=False` (le nouveau default), `f_wv` calibré pour produire des valeurs autour de 0 (par exemple `beta × fill_dev × season_sens` négatif et exactement compensé par `+1.0` dans `raw_f_wv = 1.0 + beta × ...`). Improbable en pratique mais le seuil 1e-8 est trop strict.

**Fix:** durcir le seuil et logger un warning quand on est dans la zone à risque :
```python
mean_f = float(raw_f_wv.loc[year_mask].mean())
if abs(mean_f) < 0.1:  # below 10% of nominal, suspect
    logger.warning(
        "WV renormalisation: mean(f_wv) for year %d is %.6f — close to zero, "
        "renormalised factor may be unstable. Skipping renorm for this year.",
        int(year), mean_f,
    )
    # leave f_wv == raw_f_wv for that year, no division
else:
    f_wv.loc[year_mask] = raw_f_wv.loc[year_mask] / mean_f
```

---

### WR-07: `_apply_calibration` retourne `True` (calibrated) même quand la calibration n'a pas convergé

**File:** `pfc_shaping/lt/model/assembler.py:706-725`
**Issue:**
Lignes 706-725 : si `result.converged is False` mais `max_abs_residual <= 50.0`, le code log un message d'avertissement puis `return result.calibrated_curve, True`. Le second élément du tuple — `calibrated=True` — est ensuite stocké dans le DataFrame de sortie ligne 615. Pour le path Phase 5 où `enforce_m_factor_floor=True` force `converged=False` lorsque le floor est appliqué (NEG-02 littéral, ligne 596-609 de arbitrage_free.py), cela signifie que la sortie peut être marquée `calibrated=True` alors que la non-convergence a été masquée par le floor — exactement le comportement que NEG-02 voulait empêcher.

La docstring de `CalibrationResult.converged` annonce : « True if the KKT system was solved and the maximum residual is below tolerance. »  Le contrat est cassé en aval par cette logique.

**Fix:** propager fidèlement `result.converged` :
```python
return result.calibrated_curve, bool(result.converged)
```
ou au minimum, ajouter une colonne `calibration_residual` au DataFrame pour traçabilité.

---

### WR-08: `fit_peak_ratios` shim — fragile mutation order

**File:** `pfc_shaping/calibration/cascading.py:472-494`
**Issue:**
Le shim de dépréciation populate `peak_base_spreads_`, `_base_price_per_month_`, **puis** dérive `peak_base_ratios_`. Si l'appelant a précédemment appelé `fit_peak_spreads(empty_df)` (qui set les attributes aux defaults), puis `fit_peak_ratios(real_df)`, le `_base_price_per_month_` du premier appel sera écrasé. C'est probablement le comportement voulu, mais le shim **assume** que `fit_peak_spreads` cache `_base_price_per_month_` ; si une future version de `fit_peak_spreads` cesse de le faire (refactor innocent), la dérivation ligne 490-493 lèvera AttributeError silencieusement non capturé.

Aucun test ne couvre la robustesse du shim aux variations d'implémentation de `fit_peak_spreads`.

**Fix:** garantir explicitement la précondition dans `fit_peak_ratios` :
```python
self.fit_peak_spreads(spot_history)
if not hasattr(self, "_base_price_per_month_") or not self._base_price_per_month_:
    # Defensive fallback if fit_peak_spreads contract changes
    self._base_price_per_month_ = {m: 50.0 for m in range(1, 13)}
self.peak_base_ratios_ = {
    m: 1.0 + spread / max(self._base_price_per_month_.get(m, 50.0), 1.0)
    for m, spread in self.peak_base_spreads_.items()
}
```

---

### WR-09: `test_phase05_summer_bowl_negative_acceptance` skip silencieux trompeur

**File:** `tests/test_phase05_negative_prices.py:955-971`
**Issue:**
La gate 2 (lignes 960-971) skip le test quand `baseline.parquet` n'a pas de prix négatifs. Le message de skip explique correctement la cause (synthetic environment, ShapeHourly [0.4, 2.0] clip). **Mais** ce skip cache potentiellement une régression réelle : si quelqu'un modifie une étape upstream et que le baseline (regenerated) cesse d'avoir des prix négatifs alors qu'il en avait avant, le test skip silencieusement au lieu d'échouer. SC #2 ROADMAP devient un acceptance criteria « zombie » qui ne signale plus jamais d'erreur.

**Fix:** capturer le baseline state initial (sha256 + min) dans le calibration report ou dans `conftest.py` et assertir explicitement la non-régression du baseline avant de skip. Au minimum, ajouter un test `test_phase05_baseline_has_negative_prices` qui xfail/skip si la condition n'est pas réunie ET échoue si elle l'a été dans la passe précédente (utiliser pytest.mark.xfail avec strict=False et un FAQ pour les opérateurs).

## Info

### IN-01: Logs corrompus mojibake dans assembler.py

**File:** `pfc_shaping/lt/model/assembler.py:5-7, 14-37, 431-635 (multiple lines)`
**Issue:** Les commentaires et f-strings contiennent des séquences mojibake (`ÃƒÂ©`, `Ã¢â€â‚¬Ã¢â€â‚¬`, `ÃŽÂ´`, `Ã¢â‚¬â€`) qui indiquent une corruption d'encodage (UTF-8 lu comme Latin-1 et re-encodé). Cela rend les logs et docstrings illisibles. Pas un bug fonctionnel mais dégrade gravement la maintenabilité.
**Fix:** ré-encoder le fichier en UTF-8 propre. Outil suggéré : `iconv -f UTF-8 -t UTF-8//IGNORE` après identification de la transformation appliquée, ou simplement recréer les sections corrompues à la main (les caractères concernés sont identifiables : `→`, `É`, `δ`, `—`, `é`, etc.).

---

### IN-02: Dataclass `field` importé mais inutilisé

**File:** `pfc_shaping/calibration/cascading.py:32`
**Issue:** `from dataclasses import dataclass, field` — `field` n'est jamais utilisé dans ce module.
**Fix:** supprimer l'import inutile : `from dataclasses import dataclass`.

---

### IN-03: `unused variable` dans `synthesize_peak_prices` legacy path

**File:** `pfc_shaping/calibration/cascading.py:591`
**Issue:** `try: ptype, year, sub = parse_key(key)` — `year` n'est jamais utilisé dans le bloc, contrairement à la branche spread-additive (ligne 554) qui ne l'utilise pas non plus. Cohérent par contrat mais améliorable.
**Fix:** utiliser `_` : `ptype, _, sub = parse_key(key)` pour les deux branches.

---

### IN-04: `from pfc_shaping.lt.model.water_value import WaterValueCorrection` ré-importé inline dans pipeline

**File:** `pfc_shaping/pipeline/rolling_update.py:206`, `pfc_shaping/pipeline/production_phases.py:307`
**Issue:** WaterValueCorrection est importé top-level dans `rolling_update.py` (ligne 206) **et** importé inline dans `production_phases.py:307`. Pas un bug mais incohérent ; les imports inline étaient probablement justifiés (lazy load) à un moment donné mais ne le sont plus.
**Fix:** harmoniser le pattern (idéalement lazy load tous les imports lourds, ou tous top-level).

---

### IN-05: Magic number `0.95 * median_count` pour la définition de "full years"

**File:** `pfc_shaping/calibration/cascading.py:382, 658`
**Issue:** Le seuil `0.95 * median_count` pour distinguer une année complète d'une année partielle est répété dans `fit_peak_spreads` (ligne 382) et `fit_seasonal_ratios` (ligne 658). Magic number sans constante nommée, fait à deux endroits pour deux fits.
**Fix:** introduire une constante module-level :
```python
_FULL_YEAR_COVERAGE_THRESHOLD = 0.95  # fraction of median yearly row count
```

---

### IN-06: Test `test_assembler_delta_additive` contient un pattern obfusqué inutile

**File:** `tests/test_phase05_negative_prices.py:547`
**Issue:**
```python
with pytest.raises(Exception) if False else __import__("contextlib").nullcontext():
```
`if False else` rend le code mort et difficile à lire. La construction `pytest.raises(...) if False else ...` est un vestige de refactor ; aucun raise n'est attendu.
**Fix:** retirer le context manager inutile :
```python
# Capture INFO log records during build (no exception expected)
log_records: list[logging.LogRecord] = []
# ... reste du test
```

## Structural / Cross-cutting Observations (non-blocking)

1. **PFCAssembler ctor surface** : quatre kwargs explicit + un master flag `allow_negative_prices` audit-only. La discipline est bonne mais le naming est asymétrique : trois kwargs sont des opt-out (`enforce_*=False` désactive) et un est un opt-in (`allow_negative_peak=True` autorise). Risque de confusion à l'usage. Documenter explicitement dans le docstring de classe que `allow_negative_peak=False == enforce_positive_peak=True` et envisager un renommage en `enforce_positive_peak` pour homogénéité.

2. **Tests dépendants de paths fixture** : plusieurs tests fail-skip si une fixture est absente (`_BOWL_MARKER_PATH`, `_PHASE05_BASELINE_PATH`). Pas un défaut mais l'absence des fixtures peut faire passer la CI verte alors que la couverture Phase 5 n'est en réalité que partielle. Le calibration_report mécanisme avec sha256 (test_calibration_report_matches_fixture) est un bon pattern à étendre aux fixtures Phase 5 elles-mêmes.

3. **`_resolve_allow_negative` traite "0" et "1" mais raise warning silencieux pour "true"/"false"/"yes"** (assembler.py:80-87) — comportement raisonnable mais le warning est `logger.warning` (lvl WARN), ce qui peut être trop verbose en CI ; les autres flags PFC_LT_* sont gérés via `_resolve_flag` dans shape_hourly. Harmoniser.

---

_Reviewed: 2026-05-20T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
