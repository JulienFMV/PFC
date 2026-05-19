# Phase 5: MSFC retire silent floors + PFC peut être négative — Context

**Gathered:** 2026-05-19
**Status:** Ready for planning
**Depends on:** Phase 5bis-A livrée (baseline frozen, flag persisté, factors_3d_ view, sidecar JSON hyperparams), Phase 5bis-B livrée (bowl deepening shippé sous PFC_LT_USE_SEASONAL_HOURLY_SHAPE=1, σ_on et hydro_weight_sigma_on calibrés via research dry-run).

**Style discussion:** User a délégué les calls techniques (mode EPFL/SOTA quant) avec un cross-check vigoureux sur le réalisme métier. **Catch important** : NEG-05 dans REQUIREMENTS.md ("Cal'27 = -10 €/MWh") a été flaggé comme non-réaliste — un Cal annuel n'est jamais négatif en pratique, ce sont des heures précises qui le sont. Reformulation des tests : 4 unit tests math sign-invariance + 1 system acceptance = SC #2 ROADMAP (h13 Sunday July < -20 €/MWh sur forwards positifs). NEG-05 wording sera reformulé dans REQUIREMENTS.md (deferred item).

<domain>
## Phase Boundary

**Phase 5 livre l'autorisation de prix négatifs sur la PFC LT** gated par flag `PFC_LT_ALLOW_NEGATIVE_PRICES` (env-var + ctor args). Les 4 planchers silencieux actuels sont retirés ou rendus optionnels par ctor args avec defaults OFF (negative-ready par défaut). Le pipeline LT mid-market (`PFCAssembler.build()`) peut produire des prix négatifs à l'heure quand les forwards / shape les induisent — typiquement h13 Sunday d'été sous bowl deepening 5bis-B activé.

**Drop du "log-prix" du titre ROADMAP** : Le titre historique "MSFC log-prix" est un artefact pré-audit. Aucune transformation log-space n'est appliquée. Justification : NEG-05 (math invariance sous signe) garanti par construction via PCHIP linéaire ; asinh/log ajoute une transformation inverse qui risque d'accumuler erreur de repricing. Le smoothness proportionnel (TODO P1-01 sur `arbitrage_free.py:455-468`) reste hors scope, pour une phase ultérieure.

**In scope** :
- `pfc_shaping/lt/model/msfc_spline.py` : ajout `enforce_positivity: bool = False` à `smooth_base_prices(...)` ; retire les 2 `np.maximum(B, 1.0)` aux lignes 131 et 203 quand `enforce_positivity=False`. Le clamp d'extrapolation ligne 120 (`np.clip(B_smooth_raw, y_knots.min()*0.5, y_knots.max()*2.0)`) corrigé pour fonctionner avec `y_knots.min() < 0` : utiliser `np.clip(B_smooth_raw, y_knots.min() - margin, y_knots.max() + margin)` avec margin = `0.5 * (y_knots.max() - y_knots.min())` (préserve la sémantique "bornes raisonnables autour des knots").
- `pfc_shaping/calibration/arbitrage_free.py` : ajout `enforce_m_factor_floor: bool = False` à `ArbitrageFreeCalibrator.__init__` ; retire le `m_factor = np.maximum(m_factor, 0.1)` ligne 517 quand `enforce_m_factor_floor=False`. Ajoute propagation explicite de `converged=False` quand le floor frappe (NEG-02 littéral) si `enforce_m_factor_floor=True`. Logue WARN à chaque clip.
- `pfc_shaping/lt/model/water_value.py` : ajout `enforce_floor: bool = False` à `WaterValueCorrection.__init__` ; retire `F_WV_FLOOR=0.80` clip aux lignes 394, 407 quand `enforce_floor=False`. **Refactor f_wv application en delta additif** : nouvelle API publique `compute_delta_wv(B_smooth, fill_df, calendar_df) → pd.Series` retournant `delta_wv = (f_wv - 1) × |B_smooth|`. `assembler.build()` applique `P = B × f_H × f_W + delta_wv` au lieu de `P = B × f_H × f_W × f_wv`.
- `pfc_shaping/calibration/cascading.py` : nouveau `fit_peak_spreads(spot_history)` qui calibre `peak_base_spreads_: dict[int, float]` (€/MWh, agrégé mensuellement). `synthesize_peak_prices` utilise `result[peak_key] = base_price + peak_base_spreads_[month]`. `fit_peak_ratios` deprecated (DeprecationWarning, backward-compat shim qui réécrit ratios→spreads via spot_history fallback ou raise si pas accessible).
- `pfc_shaping/lt/model/assembler.py` : adaptation `build()` pour delta-additif WV + master flag `PFC_LT_ALLOW_NEGATIVE_PRICES` lu une fois à `PFCAssembler.__init__` et logué INFO (audit-trail, pas un override automatique).
- Nouveau `tests/fixtures/_generate_phase05_fixture.py` : génère `tests/fixtures/forwards_phase05_seed42.parquet` (Cal'27=30 €/MWh, July M-07'27=20 €/MWh dépressé, autres months positifs typiques EEX, seed=42, déterministe).
- Nouveau `tests/fixtures/baseline_pfc_seed42_phase05.parquet` : output frozen `assembler.build(forwards_phase05, seed=42, PFC_LT_USE_SEASONAL_HOURLY_SHAPE=1, PFC_LT_ALLOW_NEGATIVE_PRICES=1)`. Convention D-A4-9 5bis-B "new baseline per math change atomique".
- Nouveau `tests/test_phase05_negative_prices.py` : 4 unit tests math + 1 system acceptance (5 tests total).

**Out of scope (déférés)** :
- **Phase 5ter** : distribution probabiliste par bloc (Monte-Carlo shape N=500). Réutilisera l'infra Phase 5 (negative-ready) + 5bis-B (split level/anomaly).
- **Phase 10 (refondu)** : backtest réel HFC OMPEX 2024-2025, cible Δ MAE bloc ≤ -1.5 €/MWh. Real-data validation gate.
- **TODO P1-01** : smoothness proportionnelle sur `arbitrage_free.py` (`H_scaled = diag(1/|S|) @ H @ diag(1/|S|)` déjà partielle, log-space full reste à faire). Phase ultérieure dédiée.
- **REQUIREMENTS.md NEG-05 reformulation** : sera fait dans Plan 05-01 ou en mini-fix doc pré-plan. Reformulation : "Un monthly forward négatif (e.g. July M-07 = -2 €/MWh) est correctement repricé par la PFC à -2 €/MWh moyenne sur le mois (math invariance test)".
- **2020-Q2 historique real-data slice** : pas en CI strict ; possible validation manuelle post-merge en VERIFICATION.md.
- **`PFC_LT_FORCE_LEGACY_FLOORS=1` env-var de hard rollback** : si demande opérateur post-merge, ajout simple (~10 lignes assembler.__init__). Pas en scope initial.
- Tout `pfc_shaping/ct/*`.

</domain>

<decisions>
## Implementation Decisions

### Area 1 — MSFC methodology

- **D-A1-1 :** MSFC reste LINÉAIRE (PCHIP sur prix bruts, pas de transformation log/asinh/log-prix). Le titre ROADMAP "MSFC log-prix" est documenté comme historique non-binding ; le requirements doc (NEG-01..05) est autoritative.
- **D-A1-2 :** Le clamp d'extrapolation ligne 120 (`np.clip(B_smooth_raw, y_knots.min()*0.5, y_knots.max()*2.0)`) corrigé pour signed knots : `np.clip(B_smooth_raw, y_knots.min() - margin, y_knots.max() + margin)` avec `margin = 0.5 * np.ptp(y_knots)`. Préserve "bornes raisonnables" en signed sans inverser les bornes si `y_knots.min() < 0`.
- **D-A1-3 :** NEG-05 invariance garanti par construction. Mean constraint `mean(B_smooth over period) == forward_price` reste exact en signed (`_enforce_mean_constraints` iterative est sign-invariante : `correction = error * 0.8` marche identique en negative).

### Area 2 — Floor strategy : ctor args defaults OFF + master flag audit-trail

- **D-A2-1 :** ctor args defaults False (negative-ready par défaut). NEG-01 littéral : "default off for LT" interprété comme "option `enforce_positivity` default False = floor désactivé par défaut".
  - `MSFCSmoother` n'est pas une classe (functional `smooth_base_prices(...)`), passer `enforce_positivity=False` en kwarg avec default False.
  - `ArbitrageFreeCalibrator(enforce_m_factor_floor=False)` (default).
  - `WaterValueCorrection(enforce_floor=False)` (default) — combiné avec D-A3-1 refactor delta-additif.
  - `BlockCascading(allow_negative_peak=True)` (default) — combiné avec D-A4-1 spread additif.
- **D-A2-2 :** Master flag `PFC_LT_ALLOW_NEGATIVE_PRICES` lu UNE fois à `PFCAssembler.__init__`. **Audit-trail INFO log only**, pas un override automatique. Format log : `"PFC_LT_ALLOW_NEGATIVE_PRICES={state}, floors_disabled={mscf:enforce_positivity, af:m_factor_floor, wv:floor, cascading:allow_neg_peak}"`.
- **D-A2-3 :** Rollback opérateur = passer `enforce_*=True` / `allow_negative_peak=False` aux 4 callsites explicitement. Documenté dans docstring `PFCAssembler.__init__` + un exemple dans `pfc_shaping/pipeline/autoresearch.py` commenté. Pas de hot-rollback automatique via env-var dans cette phase (deferred si demande opérateur).
- **D-A2-4 :** Cohérence avec `PFC_LT_USE_SEASONAL_HOURLY_SHAPE` (5bis-A pattern) : le nouveau master flag suit la MÊME convention (freeze-at-init, persisté sidecar via existing `shape_hourly.meta.parquet` extension OU via un nouveau metadata location à clarifier au planning). **Question planning** : `PFC_LT_ALLOW_NEGATIVE_PRICES` est-il un attribut Assembler ou ShapeHourly ? Probablement Assembler (touche tout le pipeline). Sidecar : étendre `shape_hourly.meta.parquet` JSON hyperparams ou créer `assembler.meta.parquet` séparé. À trancher au planning research.
- **D-A2-5 :** Baseline `baseline_pfc_seed42.parquet` (5bis-A frozen) **reste verte** en defaults OFF. Justification : sur forwards positifs synth (seed=42 standard), les 4 floors ne mordent pas → output identique. Test régression 5bis-A préservé sans changement. À VÉRIFIER en research : si l'écart > atol=1e-12 à cause de la composition multiplicative→additive sur f_wv, on commit 2 baselines (legacy + phase05).

### Area 3 — F_WV multiplicatif → delta additif

- **D-A3-1 :** Refactor `WaterValueCorrection` en delta additif. Nouvelle API publique `compute_delta_wv(B_smooth: pd.Series, fill_df, calendar_df) → pd.Series` retournant `delta_wv = (f_wv - 1) × |B_smooth|`. Le coefficient calibré `beta_wv_` et `season_sensitivity_` restent valides — pas de re-fit.
- **D-A3-2 :** `assembler.build()` consume `delta_wv` additivement : remplacer `P = B × f_H × f_W × f_wv` par `P = B × f_H × f_W + delta_wv`. Sign-invariant : `delta_wv = (f_wv-1) × |B|` donne le bon sens en B>0 et B<0.
- **D-A3-3 :** Équivalence baseline_5bisA en régime tout-positif **NON exacte** par construction (multiplicatif `B×f_H×f_W×f_wv` ≠ additif `B×f_H×f_W + (f_wv-1)×|B|`). Écart d'ordre `(f_wv-1) × B × (f_H × f_W - 1)`, typiquement <<1e-3 mais >1e-12. **Research dry-run mesure l'écart** : si <1e-12 sur baseline_5bisA → baseline réutilisée sans changement ; si >1e-12 → on commit nouvelle baseline `baseline_pfc_seed42_phase05.parquet` (canonique post-Phase 5) ET on garde `baseline_pfc_seed42.parquet` pour test legacy via `enforce_*=True`. **Le pattern 5bis-B D-A4-9 "new baseline per math change" couvre exactement ce cas**.
- **D-A3-4 :** `F_WV_FLOOR=0.80` retiré quand `enforce_floor=False`. Quand `enforce_floor=True` (legacy), comportement multiplicatif legacy + clip [0.80, 1.20] (backward-compat). API `compute_delta_wv` raise si `enforce_floor=True` (incompatible avec delta-additif sémantique).
- **D-A3-5 :** Telemetry. À chaque `assembler.build()`, log INFO `"WV delta_wv: min=%.2f, max=%.2f, mean=%.2f €/MWh, sign(B) flips: %d"`. Aide diagnostic si f_wv comportement inattendu post-flip.

### Area 4 — Peak synthesis spread additif + tests

- **D-A4-1 :** `cascading.py` ajoute `fit_peak_spreads(spot_history)` agrégeant `peak_avg - base_avg` par mois. Persiste `peak_base_spreads_: dict[int, float]` (€/MWh). `synthesize_peak_prices` utilise `result[peak_key] = base_price + peak_base_spreads_[month]` quand `allow_negative_peak=True` (default).
- **D-A4-2 :** `fit_peak_ratios` DEPRECATED avec `DeprecationWarning`. Backward-compat shim : raise NotImplementedError si appelé sans spot_history (impossible de réécrire ratios→spreads sans data). Logue WARN à l'init si `peak_base_ratios_` chargé depuis cache pre-Phase 5 sans `peak_base_spreads_` co-existant.
- **D-A4-3 :** En `allow_negative_peak=False` (legacy), `synthesize_peak_prices` garde le multiplicateur `ratio*price` historique + clip `Peak >= Base` (le "≥1 ratio" implicite). Backward-compat préservé pour tout caller legacy passant explicitement `allow_negative_peak=False`.
- **D-A4-4 :** Tests sign-invariance math (4 unit tests, in-test inputs, pas de fixture parquet) :
  - `test_msfc_signed_monthly_repricing` : `smooth_base_prices` avec `base_prices = {..., '2027-07': -2.0, ...}` (autres months positifs) → `mean(B_smooth['2027-07']) ≈ -2.0` atol=0.01.
  - `test_arbitrage_free_signed_target` : `ArbitrageFreeCalibrator` avec contraintes targetant un monthly négatif → `converged=True` + `max_abs_residual < tol`.
  - `test_water_value_delta_sign_invariant` : `compute_delta_wv(B_smooth=-10, f_wv=1.20)` → `delta_wv = +2` ; `compute_delta_wv(B_smooth=-10, f_wv=0.80)` → `delta_wv = -2`. Vs comportement legacy multiplicatif qui donnerait `f_wv × B = -12` (scarcity) ou `-8` (abundance) — sémantique inversée.
  - `test_cascading_spread_signed_base` : `BlockCascading.synthesize_peak_prices({'2027': -10}, peak_base_spreads_={1..12: +5})` → `result['2027-Peak'] = -5` (pas -10.5 du legacy `ratio=1.05`).
- **D-A4-5 :** Test system acceptance unique (SC #2 ROADMAP) :
  - `test_phase05_summer_bowl_negative_acceptance` : fixture `forwards_phase05_seed42.parquet` (Cal'27=30, July M-07'27=20, autres months positifs typiques), `PFC_LT_USE_SEASONAL_HOURLY_SHAPE=1` (5bis-B bowl actif), `PFC_LT_ALLOW_NEGATIVE_PRICES=1` (Phase 5 defaults OFF), `seed=42` → `assert pfc[Sunday, h13, July 2027].mean() < -20.0`.
  - **Gated** : skip si `5bis-B bowl calibration not verified` (ENV ou fixture check). Si fixture `baseline_pfc_seed42_bowl.parquet` (5bis-B baseline) prouve bowl actif avec amplitude attendue → test acceptance Phase 5 court. Sinon → skipped avec message "Phase 5 SC #2 requires 5bis-B bowl deepening calibrated first".
- **D-A4-6 :** Nouvelle baseline frozen `baseline_pfc_seed42_phase05.parquet` (convention 5bis-B D-A4-9). Generated avec defaults OFF + delta additif WV + spread additif Peak, `PFC_LT_USE_SEASONAL_HOURLY_SHAPE=1`, `PFC_LT_ALLOW_NEGATIVE_PRICES=1`. Test régression : `assert_frame_equal(build(forwards_phase05), baseline_pfc_seed42_phase05, atol=1e-12, rtol=0)`.
- **D-A4-7 :** Reformulation NEG-05 dans REQUIREMENTS.md = item deferred immédiat (à fixup soit pré-planning soit dans Plan 05-01). Wording cible : "NEG-05 : Un monthly forward négatif (e.g., July M-07'27 = -2 €/MWh, autres months positifs) est correctement repricé par la PFC à -2 €/MWh moyenne sur le mois (math invariance test, vérifie l'absence des floors silencieux)".

### Plan decomposition

- **D-A5-1 :** 3 plans séquentiels (waves, aligned 5bis-A/B convention) :
  - **Plan 05-01-PLAN.md (wave 1)** — MSFC `enforce_positivity` + ArbitrageFreeCalibrator `enforce_m_factor_floor` (Areas 1+2 partiels). Ajout ctor args, retrait conditionnels des 3 floors (msfc:131, msfc:203, arbitrage:517), clamp signed-aware ligne 120, propagation `converged=False`, telemetry. Tests math : `test_msfc_signed_monthly_repricing`, `test_arbitrage_free_signed_target`. Mini-fix REQUIREMENTS.md NEG-05 wording dans le même commit.
  - **Plan 05-02-PLAN.md (wave 2)** — WaterValueCorrection delta additif + assembler integration (Area 3). `compute_delta_wv` API, `enforce_floor=False` default, refactor `assembler.build()` `P = B×f_H×f_W + delta_wv`, telemetry. Test math `test_water_value_delta_sign_invariant`. **Research dry-run mesure écart baseline_5bisA** ; si >1e-12, ajoute nouvelle baseline ; sinon préserve l'ancienne.
  - **Plan 05-03-PLAN.md (wave 3)** — BlockCascading spread additif (Area 4 math) + master flag `PFC_LT_ALLOW_NEGATIVE_PRICES` audit-trail + fixture generator `_generate_phase05_fixture.py` + nouvelle baseline `baseline_pfc_seed42_phase05.parquet` + tests `test_cascading_spread_signed_base` + `test_phase05_summer_bowl_negative_acceptance` (SC #2 system acceptance, gated par 5bis-B). Update PROJECT.md `Key Decisions` à la livraison (D-FLIP-1 pattern : flag Phase 5 livré defaults OFF, rollback opérateur documenté).

### Claude's Discretion
- Format exact du log telemetry (D-A2-2, D-A3-5) : INFO vs DEBUG, format string exact : implémentation.
- Sidecar persistence pour `PFC_LT_ALLOW_NEGATIVE_PRICES` : étendre `shape_hourly.meta.parquet` JSON hyperparams (cohérence 5bis-A/B) OU créer `assembler.meta.parquet` séparé : trancher au planning research (D-A2-4 question ouverte).
- Pattern Python pour backward-compat shim `fit_peak_ratios` deprecated → spreads (D-A4-2) : choix entre raise NotImplementedError, shim de réécriture, ou alias avec WARN : implémentation, à trancher au planning.
- Variable de margin pour le clamp signed-aware ligne 120 (D-A1-2) : `0.5 * np.ptp(y_knots)` proposé, alternatives possibles (e.g. `min(abs(y_knots).max() * 0.5, 100)`). À calibrer en research si edge cases observés.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Roadmap & requirements
- [.planning/ROADMAP.md §Phase 5](/.planning/ROADMAP.md) (lignes 137-152) — Goal, Depends, Requirements NEG-01..05, Success Criteria #1-5.
- [.planning/REQUIREMENTS.md §Negative Prices](/.planning/REQUIREMENTS.md) — NEG-01..NEG-05. **Note** : NEG-05 wording "Cal'27=-10" à reformuler en deferred (D-A4-7). Cal annuels ne sont jamais négatifs en pratique — c'est des heures qui le sont. Reformulation cible : monthly forward négatif (e.g. July=-2).
- [.planning/PROJECT.md §Key Decisions](/.planning/PROJECT.md) — D-FLIP-1 (5bis-B flag flip strategy), pattern à étendre pour Phase 5 flag.

### Prior phase context (must read for continuity)
- [.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md](/.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md) — 5bis-A locked decisions (flag freeze-at-init, sidecar pattern, baseline frozen, conftest autouse env hygiene). Phase 5 ré-utilise le pattern sidecar pour `PFC_LT_ALLOW_NEGATIVE_PRICES`.
- [.planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md](/.planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md) — 5bis-B locked decisions (bowl deepening, σ_off/_on, hydro_weight_sigma_off/_on, baseline_pfc_seed42_bowl.parquet pattern). Phase 5 system acceptance test SC #2 dépend de la calibration 5bis-B.
- [.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-REVIEWS.md](/.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-REVIEWS.md) §1-2 — tolerance contract `atol=1e-12, rtol=0` (NOT byte equivalence). Phase 5 baseline régression suit la même convention.

### Code à modifier (Phase 5 scope)
- [pfc_shaping/lt/model/msfc_spline.py:120](pfc_shaping/lt/model/msfc_spline.py#L120) — clamp d'extrapolation signed-aware (D-A1-2).
- [pfc_shaping/lt/model/msfc_spline.py:131](pfc_shaping/lt/model/msfc_spline.py#L131) — floor 1 (mid-iteration), retirer conditionnel.
- [pfc_shaping/lt/model/msfc_spline.py:203](pfc_shaping/lt/model/msfc_spline.py#L203) — floor 2 (post-iteration), retirer conditionnel.
- [pfc_shaping/calibration/arbitrage_free.py:517](pfc_shaping/calibration/arbitrage_free.py#L517) — m_factor floor, retirer conditionnel + propage converged=False.
- [pfc_shaping/calibration/arbitrage_free.py:464](pfc_shaping/calibration/arbitrage_free.py#L464) — `price_scale = np.maximum(np.abs(S), 1.0)` smoothness scaling, OUT OF SCOPE Phase 5 (TODO P1-01).
- [pfc_shaping/lt/model/water_value.py:60](pfc_shaping/lt/model/water_value.py#L60) — F_WV_FLOOR constant, conserver mais désactiver par defaults.
- [pfc_shaping/lt/model/water_value.py:394,407](pfc_shaping/lt/model/water_value.py#L394) — clip lines, retirer conditionnel.
- [pfc_shaping/lt/model/water_value.py] — nouvelle méthode publique `compute_delta_wv(B_smooth, fill_df, calendar_df)`.
- [pfc_shaping/calibration/cascading.py:279](pfc_shaping/calibration/cascading.py#L279) — `fit_peak_ratios` deprecated.
- [pfc_shaping/calibration/cascading.py:342](pfc_shaping/calibration/cascading.py#L342) — `synthesize_peak_prices` refactor spread additif.
- [pfc_shaping/calibration/cascading.py] — nouvelle méthode `fit_peak_spreads(spot_history)`.
- [pfc_shaping/lt/model/assembler.py:175-200](pfc_shaping/lt/model/assembler.py#L175) — `PFCAssembler.__init__` lit master flag `PFC_LT_ALLOW_NEGATIVE_PRICES`, audit log.
- [pfc_shaping/lt/model/assembler.py] — `assembler.build()` consume `compute_delta_wv` additivement.

### Callsites legacy à préserver (backward-compat audit)
- [pfc_shaping/pipeline/autoresearch.py:234](pfc_shaping/pipeline/autoresearch.py#L234) — ShapeHourly pattern ; doit aussi adopter `enforce_*=True` explicit si l'utilisateur veut le legacy floor behaviour. Default = nouveau Phase 5 negative-ready.
- [pfc_shaping/pipeline/rolling_update.py:365](pfc_shaping/pipeline/rolling_update.py#L365) — idem.
- [tests/fixtures/_generate_baseline.py](tests/fixtures/_generate_baseline.py) — script qui a généré baseline_pfc_seed42.parquet (5bis-A). Vérifier qu'il continue à produire l'identique avec defaults Phase 5 OFF — sinon ajouter `enforce_*=True` explicit pour preserve baseline 5bis-A.

### Convention quant state-of-the-art
- [.claude/projects/-Users-julienbattaglia-Desktop-PFC/memory/reference_pfc_state_of_art.md](/Users/julienbattaglia/.claude/projects/-Users-julienbattaglia-Desktop-PFC/memory/reference_pfc_state_of_art.md) — Benth-Koekebakker max smoothness (linéaire), Fleten-Lemming. Le drop "log-prix" aligne avec Benth-Koekebakker-Ollmar 2007 qui est explicitement linéaire ; le log-prix serait Kiesel-Paraschiv (multiplicative), déjà partiellement appliqué dans arbitrage_free "multiplicative" mode (HORS scope Phase 5).

### Convention agentique
- [.planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md §code_context "Established Patterns"](/.planning/phases/05C-shape-hourly-bowl-deepening/05C-CONTEXT.md) — Pattern "freeze-at-init + persist-in-sidecar + parquet-wins-over-env" à appliquer pour `PFC_LT_ALLOW_NEGATIVE_PRICES`. Pattern "new baseline per math change atomique" (D-A4-9) appliqué pour `baseline_pfc_seed42_phase05.parquet`.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Pattern `_resolve_flag` à [shape_hourly.py:60-77](pfc_shaping/lt/model/shape_hourly.py#L60) (5bis-A) — modèle de resolution explicit-arg-vs-env-default. À transposer pour `PFC_LT_ALLOW_NEGATIVE_PRICES` à `PFCAssembler.__init__`. Helper `_resolve_allow_negative(explicit_arg, env_var_name, default)` sur le même pattern.
- Sidecar `shape_hourly.meta.parquet` hyperparams JSON ([shape_hourly.py:517-525](pfc_shaping/lt/model/shape_hourly.py#L517), 5bis-A D-03 + 5bis-B D-A3-3 extension) — schema déjà extensible (sort_keys=True). Phase 5 peut ajouter `allow_negative_prices_resolved` au même sidecar (si décision sidecar partagé) OU créer `assembler.meta.parquet` séparé.
- Fixture pattern `tests/fixtures/_generate_baseline.py` (5bis-A) + `tests/fixtures/_generate_bowl_fixture.py` (5bis-B) — modèles pour `tests/fixtures/_generate_phase05_fixture.py`. Reuse synthetic forwards + seed=42 convention.
- `_enforce_mean_constraints` ([msfc_spline.py:145](pfc_shaping/lt/model/msfc_spline.py#L145)) — iterative correction sign-invariante par construction (`correction = error * 0.8` marche identique en signed). Le seul fix nécessaire est le `np.maximum(result, 1.0)` ligne 203 final.
- Conftest autouse env-var hygiene ([tests/conftest.py](tests/conftest.py), 5bis-A D-12) — déjà en place pour `PFC_LT_*` env vars. Phase 5 hérite automatiquement.

### Established Patterns
- **Freeze-at-init + persist-in-sidecar + parquet-wins-over-env** (5bis-A D-06..D-08, 5bis-B D-A3-3) : appliqué à `_use_seasonal_hourly`, `sigma_off/_on`, `hydro_weight_sigma_off/_on`. Phase 5 étend le pattern à `allow_negative_prices`.
- **Tolerance contract atol=1e-12, rtol=0** (5bis-A REVIEWS addendum) : MÊME contract pour `baseline_pfc_seed42_phase05.parquet`. Pas byte equivalence parquet — `assert_frame_equal(check_exact=False, atol=1e-12, rtol=0)` + identical columns/dtypes/index/sort order.
- **"New baseline per math change atomique"** (5bis-B D-A4-9) : Phase 5 livre `baseline_pfc_seed42_phase05.parquet`. Pattern : `baseline_pfc_seed42_{feature_name}.parquet`.
- **ctor args defaults + sidecar persistence + master flag audit-trail** : nouvelle convention introduite par Phase 5. Master flag = info-only, ctor args sont la véritable surface API. Si Phase 5ter / future ont aussi un master flag, suivre ce pattern.
- **`gsd-executor` mode no-worktree, 1 plan = 1 wave** (5bis-A/B convention) : Phase 5 = 3 plans séquentiels mêmes modes.

### Integration Points
- `assembler.build()` ([assembler.py](pfc_shaping/lt/model/assembler.py)) — point d'entrée unique. 5bis-A préparait le plumbing pour shape, 5bis-B a livré le bowl, Phase 5 ajoute la sémantique signed (P = B×f_H×f_W + delta_wv). Pas d'autre callsite à toucher pour le pipeline d'export.
- `ShapeHourly.factors_3d_` (5bis-A D-A4-4) — view qui ne dépend pas du signe. Pas de modification Phase 5.
- `BlockCascading.fit_peak_ratios` → `fit_peak_spreads` : si appelé depuis un pipeline upstream (e.g. `rebuild_forwards_history.py`), ce script doit aussi adopter `fit_peak_spreads`. Audit à faire en research.
- `tests/conftest.py` autouse env hygiene — Phase 5 ajoute `PFC_LT_ALLOW_NEGATIVE_PRICES` à la liste snapshot/restore par test (~1 ligne).

</code_context>

<specifics>
## Specific Ideas

### Use case business à préserver dans le mental model
FMV pricing des profile deals GRD (bloc nuit 18-9 acheté à FMV par le client industriel + bloc solaire WE OP1/OP2 racheté par FMV à la production GRD). En été 2027+, le bowl midday (h10-15) devient suffisamment profond pour franchir zéro localement (sous-jacent : excès de PV ailleurs en CH/DE/AT). **Sans Phase 5**, la PFC ne reflète pas cette réalité (plancher à 1 €/MWh) → FMV sur-achète le bloc solaire racheté (paie trop cher pour de l'énergie qui sera valorisée à -X €/MWh sur le spot). **Avec Phase 5**, la PFC mid-market peut atteindre -20/-25 €/MWh à h13 dimanche d'été → le bloc solaire racheté est valorisé au juste prix.

### Catch métier important en discuss-phase
User a flag que NEG-05 dans REQUIREMENTS.md ("Cal'27 = -10 €/MWh") est non-réaliste. **Apprentissage** : downstream agents (researcher, planner) DOIVENT lire ce CONTEXT.md d'abord avant REQUIREMENTS.md sur NEG-05 spécifiquement, sinon ils vont strawman le test sur un scenario impossible. Reformulation actée : monthly forward négatif (e.g., July M-07'27 = -2 €/MWh).

### Méthode scientifique (innovation EPFL angle, suite 5bis-A/B)
- **Sign-invariance par construction** : la math de PCHIP (lin), iterative mean correction, et delta-additif WV est toute sign-invariante naturellement quand on retire les floors. Pas de "monkey-patch en negative" — c'est la math qui marche des deux côtés.
- **Master flag audit-trail, pas override** : pattern délibéré pour éviter le silent-revert. Si un opérateur veut le legacy, il doit toucher les 4 callsites explicitement → audit-able dans git history. Pas de "env-var magique qui change tout".
- **Test acceptance gated par dépendance** : SC #2 ROADMAP (h13 Sunday < -20) gated par "5bis-B bowl calibrated". Si le test rate, on sait immédiatement si c'est Phase 5 (math) ou 5bis-B (bowl pas assez profond) qui a regressé. Diagnostic non-ambigu.

### Convention quant "no-op refactor first, math change second" (suite de 5bis-A/B)
- 5bis-A : infra no-op refactor ✓
- 5bis-B : math change shape (bowl deepening) ✓
- Phase 5 : math change negative prices (defaults OFF par classe + master flag) — **PREMIÈRE phase à introduire les ctor args defaults OFF comme negative-ready primitive**, suivie par Phase 5ter (distribution probabiliste) et Phase 10 (backtest réel).
- Nouvelle convention Phase 5 : **flag master `audit-trail INFO log` n'est pas un override**. Les ctor args sont la véritable surface API. Cohérent avec EPFL principe "explicit > implicit".

</specifics>

<deferred>
## Deferred Ideas

### Immédiat (à fixup pré-planning ou dans Plan 05-01)
- **Reformulation NEG-05 dans REQUIREMENTS.md** (D-A4-7) : remplacer "Cal'27 = -10 €/MWh" par "monthly forward négatif (e.g. July M-07'27 = -2 €/MWh) repricé exactement". User caught the unrealistic wording.

### Vers Phase 5ter (distribution probabiliste par bloc)
- `pfc_block_distribution(start, end, hours_mask) → (p10, p50, p90)` Monte-Carlo shape N=500. Réutilise infra Phase 5 (negative-ready) — les trajectoires Monte-Carlo peuvent franchir zéro proprement.
- Le `prime de risque shape inhedgeable` calculé sur le bloc 10-15 d'été 2027 dépend directement de Phase 5 (sinon les p10/p50 sont écrasés au plancher 1€/MWh).

### Vers Phase 10 (refondu, real-data validation gate)
- Backtest réel HFC OMPEX 2024-2025 par bloc client (10-15 weekday, 18-9 weekday, 12-14 weekend summer).
- Cible KPI : Δ MAE bloc ≤ -1.5 €/MWh vs HFC OMPEX (PROJECT.md).
- **Gating flip flag defaults** : Phase 5 et 5bis-B sont livrés defaults negative-ready / bowl OFF respectivement. Le flip ON de `PFC_LT_USE_SEASONAL_HOURLY_SHAPE` (5bis-B D-FLIP-1) et le "no-rollback" Phase 5 (defaults OFF restent) sont gated par Phase 10 success.

### Vers TODO P1-01 (smoothness proportionnelle log-space)
- `arbitrage_free.py:455-468` : `H_scaled = diag(1/|S|) @ H @ diag(1/|S|)` est une mitigation partielle. La formulation log-space full reste à faire (Kiesel-Paraschiv consistent). Phase ultérieure dédiée si trader-grade smoothness en absolu vs proportionnel devient un signal mesurable post-Phase 10.

### Vers hot-rollback opérateur (si demande post-merge)
- Env-var `PFC_LT_FORCE_LEGACY_FLOORS=1` qui force `enforce_*=True` aux 4 callsites à l'init de `PFCAssembler`. ~10 lignes. Pas en scope initial mais facile à ajouter si demande opérateur. Documenter dans docstring `PFCAssembler.__init__` comme escape hatch.

### Vers 2020-Q2 historical real-data validation
- Slice mai-juin 2020 (covid negative spot hours) lu depuis EEX historical. Pas en CI strict ; possible validation manuelle post-merge en VERIFICATION.md (similar to 5bis-B post-merge validation pattern).

### Vers cleanup ROADMAP title
- "MSFC log-prix + retire silent floors" → "MSFC retire silent floors + PFC peut être négative". À trancher en planning ou commit final Phase 5. Le titre actuel reste comme aliasing historique mais documenté SUPERSEDED dans CONTEXT.md.

</deferred>

---

*Phase: 5 — MSFC retire silent floors + PFC peut être négative*
*Context gathered: 2026-05-19*
*Convention quant: ctor args defaults OFF (negative-ready) + master flag audit-trail (Phase 5 introduit ce pattern, Phase 5ter et future phases negative-related l'étendront)*
