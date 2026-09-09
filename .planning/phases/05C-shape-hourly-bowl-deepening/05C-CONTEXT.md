# Phase 5bis-B: Shape Hourly Bowl-Deepening (math change) — Context

**Gathered:** 2026-05-19
**Status:** Ready for planning
**Depends on:** Phase 5bis-A livrée (baseline `tests/fixtures/baseline_pfc_seed42.parquet` frozen, flag `PFC_LT_USE_SEASONAL_HOURLY_SHAPE` persisté en sidecar `shape_hourly.meta.parquet`, save/load complet sur tous attributs entraînés, `factors_3d_` view, conftest autouse env hygiene, capability check assembler).

**Style discussion:** User (énergy quant FMV, EPFL-level) a explicitement délégué les 4 calls techniques majeurs ("vous êtes les experts EPFL, à vous de prendre les bons choix… docteur, doctorant EPFL et QI de 200") — chaque décision ci-dessous est verrouillée au niveau SOTA quant + innovation, en s'appuyant sur le code base existant, l'audit deep LT 2026-05, et les conventions 5bis-A.

<domain>
## Phase Boundary

**5bis-B livre la VALEUR MÉTIER derrière le flag `PFC_LT_USE_SEASONAL_HOURLY_SHAPE=1`** : creuser la duck curve réelle de la PFC pour que les profile deals GRD (bloc nuit 18h-9h + bloc solaire WE OP1/OP2 racheté à la production) soient pricés au juste prix. **Aucune valeur numérique ne change quand flag=OFF** (baseline 5bis-A préservée bit-pour-bit, atol=1e-12).

Trois leviers gated par le flag (ROADMAP §Phase 5bis-B locked) :
1. **Bug fix `_apply_hydro_analogue_weights`** ([shape_hourly.py:839-911](pfc_shaping/lt/model/shape_hourly.py#L839-L911)) : kernel target = `_climatological_fill[week_of_year(t)]` par sample historique (au lieu de `current_fill` scalar global). Préserve la diversité saisonnière du bowl.
2. **Split f_H = level + anomaly** : helper `_split_level_anomaly(f_H_series, cal_df)` extrait `level := mean_h(f_H | cell)` et `anomaly := f_H - level`. À [assembler.py:333](pfc_shaping/lt/model/assembler.py#L333), sous flag=ON, `shape_freedom['f_H']` damping s'applique **uniquement au level** ; l'anomaly (signature saisonnière) survit à 100% jusqu'à Y+2/Y+3.
3. **σ smoothing paramétrable** : `__init__(sigma_off=0.5, sigma_on=0.25)` exposés en ctor (innovation EPFL : tout hyperparam flag-gated est explicite, persisté sidecar, MLflow-traceable). Default flag=OFF garde 0.5 (legacy), default flag=ON utilise 0.25 (lever mineur 0.5-1 €/MWh, bonus).

**In scope**:
- `pfc_shaping/lt/model/shape_hourly.py` : ajout helper module-level `_split_level_anomaly`, refactor ctor `__init__` (3 nouveaux args + backward-compat `sigma`), refactor `_apply_hydro_analogue_weights` (per-timestamp climato target, gated par flag), extension sidecar hyperparams JSON (`sigma_off`, `sigma_on`, `hydro_weight_sigma_off`, `hydro_weight_sigma_on`).
- `pfc_shaping/lt/model/assembler.py` :305-345 — intégration split level/anomaly sous flag=ON.
- Nouveau fixture déterministe `tests/fixtures/_generate_bowl_fixture.py` + `tests/fixtures/bowl_seed42.parquet` (~50KB).
- Nouvelle baseline frozen flag=ON : `tests/fixtures/baseline_pfc_seed42_bowl.parquet`.
- Nouveau `tests/test_shape_hourly_bowl.py` (7 tests : kernel reformulation, split invariant, SC #1 ptp deepening, SC #3 amplitude M+30, SC #2 seasonal solar/evening delta sur synth, baseline flag=OFF bit-pour-bit, nouvelle baseline flag=ON).
- Update PROJECT.md `Key Decisions` à la livraison : date de flip flag default OFF→ON = **post Phase 10 success** (real-data gate).

**Out of scope (déférés)** :
- **Phase 10** (refondu) : backtest réel HFC OMPEX 2024-2025, cible Δ MAE bloc client ≤ -1.5 €/MWh. Le real-data validation gate qui autorise le flip flag.
- **Phase 5** : floors silencieux (MSFC `max(B,1)`, m_factor floor 0.1, F_WV_FLOOR, peak ratio ≥1) — la PFC peut être négative.
- **Phase 5ter** : distribution probabiliste par bloc (Monte-Carlo shape N=500).
- **Phase 3** (HOLD) : activation FR/AT/IT.
- Anything in `pfc_shaping/ct/*`.
- Recalibration empirique de `sigma_on`, `hydro_weight_sigma_on`, `np.ptp` thresholds : c'est research/calibration (dry-run par `gsd-phase-researcher`), pas discussion.

</domain>

<decisions>
## Implementation Decisions

### Lever 1 — Hydro kernel reformulation (Area 1)
- **D-A1-1** : Kernel target = `self._climatological_fill[week_of_year(t)]` per-sample historique. Formulation : `hydro_weight[i] = exp(-0.5 * ((fill_values[i] - clim_target[i]) / hydro_weight_sigma)**2)` où `clim_target[i] = self._climatological_fill[woy(df.index[i])]`. Mesure "anomalie de fill vs norme climatologique à cette même semaine de l'année" — préserve la diversité saisonnière par construction.
- **D-A1-2** : Gating bit-pour-bit. Si `self._use_seasonal_hourly == False` → kernel garde `current_fill = float(fill.iloc[-1])` (legacy). Si `True` → per-timestamp clim target. Le contrat 5bis-A baseline régression reste vert à atol=1e-12.
- **D-A1-3** : Floor 0.3 ([shape_hourly.py:902](pfc_shaping/lt/model/shape_hourly.py#L902)) préservé sous les deux flags. Safety net no-op dans régime normal (anomalies typiques ±10pp ≪ σ=0.07), protège des cas pathologiques (drought 2022 / wet 2021).
- **D-A1-4** : `hydro_weight_sigma` devient flag-aware via ctor : `__init__(..., hydro_weight_sigma_off: float = 0.25, hydro_weight_sigma_on: float = TBD, ...)`. Persisté JSON sidecar. Default `hydro_weight_sigma_on` à calibrer par `gsd-phase-researcher` via dry-run sur fixture EPEX synth+real — anomalie typique (fill - clim[woy]) ≈ ±10pp vs ±30pp legacy → σ=0.25 deviendrait quasi-uniforme, valeur cible probablement ~0.05-0.10. À fermer en RESEARCH.md.
- **D-A1-5** : Backward-compat sur callsite legacy `ShapeHourly(hydro_weight_sigma=X)` : interprété comme `hydro_weight_sigma_off = hydro_weight_sigma_on = X` (single-σ pour les deux flag states, bit-pour-bit préservé). Voir D-A3-2 pour le pattern de résolution unifié.

### Lever 2 — Split f_H level/anomaly (Area 2)
- **D-A2-1** : Helper module-level `_split_level_anomaly(f_H_series: pd.Series, cal_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]` dans `pfc_shaping/lt/model/shape_hourly.py`. Exposé en `__all__`, testable indépendamment. Vit dans le module shape_hourly car la sémantique math (decomposition) appartient au modèle, pas à l'assembler.
- **D-A2-2** : Math additive. `level[t] = mean_h(f_H | saison(t), type_jour(t))` (per-cell mean of all f_H values in that cell for the current call's timestamps — groupby (saison, type_jour) du cal_df). `anomaly[t] = f_H[t] - level[t]`. Invariants : `level + anomaly ≡ f_H` (ulp exact) ; `mean_h(anomaly | cell) ≡ 0` (zero-mean per cell). Sum-preserving.
- **D-A2-3** : Damping sous flag=ON ([assembler.py:333](pfc_shaping/lt/model/assembler.py#L333)) : `level, anomaly = _split_level_anomaly(f_H, cal); level_damped = 1 + (level - 1) * sf['f_H']; f_H = level_damped + anomaly`. L'anomaly survit 100% à Y+2/Y+3, le level (qui dérive avec `trend_per_hour_` à far horizon) est shrinké vers 1.0.
- **D-A2-4** : Damping sous flag=OFF inchangé : `f_H = 1 + (f_H - 1) * sf['f_H']` (legacy). Baseline 5bis-A préservée bit-pour-bit.
- **D-A2-5** : Telemetry innovation. À chaque appel `assembler.build`, log INFO `"f_H split: max |level - 1.0| = {value:.2e}"`. Warning si > 1e-6 → détecte drift silencieux de l'invariant SHP-03 (mean-preservation), future-proof si Phase 5 MSFC log-prix ou un fit ultérieur changeait la normalisation.
- **D-A2-6** : Knot schedule pour level damping (flag=ON) = identique à `shape_freedom['f_H']` actuel `[(0,1.00), (6,0.98), (12,0.88), (24,0.62), (36,0.42)]` ([assembler.py:831-834](pfc_shaping/lt/model/assembler.py#L831-L834)). `gsd-phase-researcher` peut proposer recalibration après mesure du bowl amplitude sur fixture EPEX. Anomaly = damping 1.0 partout (pass-through total).

### Lever 3 — σ smoothing paramétrisation (Area 3)
- **D-A3-1** : Signature ctor étendue : `ShapeHourly.__init__(sigma: float | None = None, sigma_off: float = 0.5, sigma_on: float = 0.25, hydro_weight_sigma: float | None = None, hydro_weight_sigma_off: float = 0.25, hydro_weight_sigma_on: float = TBD, ...)`. Backward-compat préservée pour tous les callsites legacy ([autoresearch.py:234](pfc_shaping/pipeline/autoresearch.py#L234), [rolling_update.py:365](pfc_shaping/pipeline/rolling_update.py#L365), tests).
- **D-A3-2** : Resolution precedence unifié (sigma ET hydro_weight_sigma) :
  ```python
  if sigma is not None:
      # Legacy single-σ caller: applies to both flag states (bit-pour-bit preservation)
      self._sigma_off = sigma
      self._sigma_on = sigma
  else:
      self._sigma_off = sigma_off
      self._sigma_on = sigma_on
  self.sigma = self._sigma_on if self._use_seasonal_hourly else self._sigma_off
  ```
  Idem pour hydro_weight_sigma. Garantit `ShapeHourly(sigma=0.5)` (callsite legacy) produit le baseline 5bis-A à 1e-12, indépendamment du flag.
- **D-A3-3** : Persistence sidecar étendue. Hyperparams JSON contient `sigma_off`, `sigma_on`, `sigma_resolved`, `hydro_weight_sigma_off`, `hydro_weight_sigma_on`, `hydro_weight_sigma_resolved`. Reload : si keys manquants (sidecar pré-5bis-B, écrit par 5bis-A), fallback `sigma_off = sigma_on = legacy_sigma_from_hyperparams` (cross-plan compat, identique au pattern D-7 de 5bis-A pour `use_seasonal_hourly`).
- **D-A3-4** : Default values. `sigma_off=0.5` (GAUSSIAN_SIGMA legacy default constant, preserved), `sigma_on=0.25` (ROADMAP-proposed). `gsd-phase-researcher` valide empiriquement σ_on=0.25 contre fixture EPEX synth via mesure du bowl FWHM (σ=0.25h → FWHM≈0.59h sur grille 15min, ~1 quantum smoothing).
- **D-A3-5** : Legacy callsites continuent à fonctionner identique. **Zero migration** required pour `ShapeHourly(sigma=X)` patterns existants.
- **D-A3-6** : Telemetry init. Log INFO à `__init__` : `"ShapeHourly init: σ_resolved={self.sigma}, σ_off={self._sigma_off}, σ_on={self._sigma_on}, flag={self._use_seasonal_hourly}, hydro_σ_resolved={self.hydro_weight_sigma}"`. EPFL traceability.

### Test design (Area 4)
- **D-A4-1** : Fixture déterministe synthétique. `tests/fixtures/_generate_bowl_fixture.py` (seed=42) génère 3 mois 15min de prix EPEX-like avec duck curve injectée explicitement : weekday h10-15 dépressé (solar drop), weekday h17-20 boosté (evening peak), weekend h12-14 deep bowl (WE solar racheté), saison été plus marquée que hiver. Output : `tests/fixtures/bowl_seed42.parquet` (~50KB). **Méthode scientifique** : ground truth analytique → `np.ptp` attendu calculable, repro garantie cross-CI, ~50KB vs ~10MB real slice.
- **D-A4-2** : Nouveau fichier test isolé `tests/test_shape_hourly_bowl.py`. `test_shape_hourly_infra.py` (5bis-A) reste inchangé.
- **D-A4-3** : `test_hydro_kernel_uses_per_timestamp_climatological_target` — mock fill hydro_df avec saisonnalité connue, verify kernel target == clim[woy(t)] et non `current_fill`. Direct verification D-A1-1.
- **D-A4-4** : `test_split_level_anomaly_invariant` — sur f_H synth varié, assert `level + anomaly == f_H` (numpy.allclose atol=1e-15), `np.abs(anomaly.groupby([saison, type_jour]).mean()).max() < 1e-12` (zero-mean per cell). Direct verification D-A2-2.
- **D-A4-5** : `test_factors_ptp_deepens_under_flag` (SC #1 ROADMAP). Sur fixture bowl_seed42, fit `sh_off` (flag=False, σ=0.5) et `sh_on` (flag=True, σ=0.25, hydro reformulé). Assert `np.ptp(sh_on.factors_[("Ete","Ouvrable")]) > np.ptp(sh_off.factors_[("Ete","Ouvrable")]) * 1.X`. **X (multiplicative gain target ≥ 1.15)** : measure-then-assert pattern — `gsd-phase-researcher` exécute dry-run, capture amplitude observée (e.g. 1.32), commit threshold avec safety margin (e.g. > 1.15).
- **D-A4-6** : `test_f_H_amplitude_preserved_at_M30` (SC #3 ROADMAP). Build PFC avec horizon M+30 (years_ahead ~2.5), assert `np.ptp(f_H_series_post_damping)` > 0.X (calibre via measure-then-assert). **Innovation** : prouve que `shape_freedom['f_H']` damping sous flag=ON ne tue plus le bowl à far horizon — le test 5bis-A baseline ne capture pas ça (M+1 horizon).
- **D-A4-7** : `test_seasonal_solar_winter_evening_delta` (SC #2 ROADMAP, **sur synth**). Sur fixture bowl_seed42, assert `|mean(price_shape[Dim, Été, h10-15]) − mean(price_shape[Dim, Hiver, h10-15])| > 5 €/MWh`. **Innovation gating** : 5bis-B passe SC #2 sur synth = condition *nécessaire* (math correcte) ; Phase 10 validera sur HFC OMPEX réel = condition *suffisante* (data fit). Failure synth = math broken, ship-blocker 5bis-B. Pass synth + fail real (Phase 10) = fixture-real gap, informe future fixture design.
- **D-A4-8** : `test_flag_off_bit_for_bit_baseline` — extends 5bis-A D-19, atol=1e-12, rtol=0 contre `tests/fixtures/baseline_pfc_seed42.parquet` (5bis-A frozen). Confirme reverse-binary safety.
- **D-A4-9** : `test_flag_on_bowl_baseline` (NOUVELLE convention pattern). Commit `tests/fixtures/baseline_pfc_seed42_bowl.parquet` = sortie de `assembler.build(...)` avec flag=ON, seed=42, mêmes inputs que baseline_pfc_seed42. Régression à atol=1e-12, rtol=0 contre cette nouvelle baseline. **Convention établie** : chaque flag transition / math change atomique = nouvelle baseline frozen séparée (Phase 5 / 5ter / future phases shape suivront ce pattern).

### Plan decomposition (Area 4 conclusion)
- **D-A4-10** : 3 plans séquentiels (waves d'exécution, aligned 5bis-A convention) :
  - **Plan 05C-01-PLAN.md** (wave 1) — Lever 1 : refactor `_apply_hydro_analogue_weights` (per-timestamp clim target gated par flag) + ctor extension `hydro_weight_sigma_off/_on` avec backward-compat resolution + persistence sidecar étendue + tests D-A4-3, D-A4-8.
  - **Plan 05C-02-PLAN.md** (wave 2) — Lever 2 : helper module `_split_level_anomaly` + intégration `assembler.build` gated par flag + telemetry drift detection + tests D-A4-4, D-A4-6.
  - **Plan 05C-03-PLAN.md** (wave 3) — Lever 3 : ctor extension `sigma_off/_on` avec resolution unifiée + persistence + telemetry init + fixture generator `_generate_bowl_fixture.py` + nouvelle baseline `baseline_pfc_seed42_bowl.parquet` + tests D-A4-5, D-A4-7, D-A4-9 + update PROJECT.md Key Decisions (flag flip date).
- Justification : bisection facile. Si SC #1 (ptp deepening) régresse, on sait quel plan a cassé. Aligne convention 5bis-A (5 plans séquentiels, mode no-worktree par plan).

### Flag flip strategy
- **D-FLIP-1** : Date de flip default `PFC_LT_USE_SEASONAL_HOURLY_SHAPE` OFF → ON = **post Phase 10 success** (real-data validation gate). Pas T+30j auto post-merge. EPFL/SOTA principle : "no production change without empirical validation". Inscrire dans PROJECT.md `Key Decisions` à la livraison 5bis-B sous la forme : `2026-MM-DD | Flag PFC_LT_USE_SEASONAL_HOURLY_SHAPE livré default OFF | Flip default ON gated par Phase 10 Δ MAE bloc ≤ -1.5 €/MWh validation`.

### Claude's Discretion
- Format exact du synthetic bowl fixture (long-format DataFrame avec colonnes `[datetime, price]` vs wide avec colonnes par jour) : implémentation, peu critique.
- Choix exact des knots pour `np.ptp` threshold (D-A4-5) et f_H amplitude M+30 (D-A4-6) : calibration empirique par researcher, pattern measure-then-assert documenté en test docstring.
- Pattern Python pour la cross-plan persistence compat sur sigma_off/sigma_on (le pattern legacy fallback D-A3-3) : choix entre lire `sigma` solo et propager vs lire les 3 keys avec defaults : implémentation, à trancher en planning.
- Niveau de granularité du log telemetry (D-A2-5, D-A3-6) : INFO vs DEBUG, format exact du message : implémentation.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Roadmap & requirements
- [.planning/ROADMAP.md §Phase 5bis-B](/.planning/ROADMAP.md) (lignes 43-47, 109-126) — délimitation scope + Success Criteria (SC #1 ptp, SC #2 €5/MWh, SC #3 amplitude M+30, SC #4 flag OFF bit-pour-bit, SC #5 142+5bis-A green).
- [.planning/REQUIREMENTS.md SHP-01..SHP-04](/.planning/REQUIREMENTS.md) — SHP-01..SHP-04 déjà satisfaits littéralement par 5bis-A (view 3D, flag persisté). 5bis-B livre la VALEUR métier derrière sans modifier les requirements.
- [.planning/PROJECT.md Constraints + Core Value](/.planning/PROJECT.md) — block-MAE -1.5 €/MWh KPI vs HFC OMPEX, 142 tests + 5bis-A baseline, branch unique `claude/clean-lt-ct-integration`, `pfc_shaping.ct.*` interdit.

### Prior phase context (must read for continuity)
- [.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md](/.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md) — 5bis-A locked decisions D-01..D-20 (flag mechanics freeze-at-init, sidecar `shape_hourly.meta.parquet`, baseline frozen, conftest autouse env hygiene, capability check assembler). 5bis-B étend le sidecar (D-A3-3) et préserve toutes les autres décisions 5bis-A.
- [.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-REVIEWS.md](/.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-REVIEWS.md) §1-2 — addendum tolerance contract `atol=1e-12, rtol=0` (NOT byte equivalence). 5bis-B baseline régression suit la même convention.
- [.planning/phases/05bis-shape-seasonal-hourly/CONTEXT.md](/.planning/phases/05bis-shape-seasonal-hourly/CONTEXT.md) — predoc 5bis originale (SUPERSEDED par split A/B post panel d'experts 2026-05-18). Référence historique seulement.

### Audit context (driver of split + lever 1 specifically)
- [.planning/research/audit-deep-lt-2026-05.md](/.planning/research/audit-deep-lt-2026-05.md) — Audit deep LT 25 findings 2026-05-18 ; finding sur `_apply_hydro_analogue_weights` `current_fill` bias = source du Lever 1.

### Code à modifier
- [pfc_shaping/lt/model/shape_hourly.py:44 GAUSSIAN_SIGMA](pfc_shaping/lt/model/shape_hourly.py#L44) — legacy default 0.5 ; deviendra `sigma_off`.
- [pfc_shaping/lt/model/shape_hourly.py:168-195 __init__](pfc_shaping/lt/model/shape_hourly.py#L168-L195) — extension signature D-A3-1.
- [pfc_shaping/lt/model/shape_hourly.py:243 _apply_hydro_analogue_weights call](pfc_shaping/lt/model/shape_hourly.py#L243) — invocation depuis fit().
- [pfc_shaping/lt/model/shape_hourly.py:839-911 _apply_hydro_analogue_weights body](pfc_shaping/lt/model/shape_hourly.py#L839-L911) — kernel refactor D-A1-1..3.
- [pfc_shaping/lt/model/shape_hourly.py:441-630 save/load](pfc_shaping/lt/model/shape_hourly.py#L441-L630) — extension sidecar hyperparams JSON D-A3-3.
- [pfc_shaping/lt/model/assembler.py:307-345 f_H consumption + shape_freedom damping](pfc_shaping/lt/model/assembler.py#L307-L345) — split level/anomaly integration D-A2-3..4.
- [pfc_shaping/lt/model/assembler.py:803-844 _shape_freedom knots](pfc_shaping/lt/model/assembler.py#L803-L844) — knot table préservée (level damping uses same).

### Callsites legacy à préserver (backward-compat audit)
- [pfc_shaping/pipeline/autoresearch.py:234](pfc_shaping/pipeline/autoresearch.py#L234) — `ShapeHourly(sigma=sigma)` pattern, doit produire baseline 5bis-A à 1e-12 indépendamment du flag.
- [pfc_shaping/pipeline/rolling_update.py:365](pfc_shaping/pipeline/rolling_update.py#L365) — `ShapeHourly(sigma=params.get("gaussian_sigma", 0.5)).fit(...)` idem.
- [tests/test_shape_hourly_infra.py:56,239,250,628](tests/test_shape_hourly_infra.py) — quatre callsites explicit `sigma=X` ; tous doivent rester verts à atol=1e-12.

### Convention quant repository state-of-the-art
- [.claude/projects/-Users-julienbattaglia-Desktop-PFC/memory/reference_pfc_state_of_art.md](/Users/julienbattaglia/.claude/projects/-Users-julienbattaglia-Desktop-PFC/memory/reference_pfc_state_of_art.md) — Benth-Koekebakker max smoothness, Fleten-Lemming, KYOS/Volue analogue-day methodology, quality criteria (Hildmann 2013). Lever 1 hydro kernel reformulation aligne avec KYOS analogue-day literature (per-timestamp similarity-weighted KDE).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_climatological_fill` (Series week→fill_pct) calculé à [shape_hourly.py:872-874](pfc_shaping/lt/model/shape_hourly.py#L872-L874) lors de `_apply_hydro_analogue_weights` — **déjà disponible** ; le bug actuel est qu'il est calculé mais pas utilisé pour le kernel target. 5bis-B le branche enfin.
- Pattern `_resolve_flag` à [shape_hourly.py:60-77](pfc_shaping/lt/model/shape_hourly.py#L60-L77) (5bis-A) — modèle de resolution explicit-arg-vs-env-default à étendre pour `sigma_off/sigma_on` et `hydro_weight_sigma_off/_on` (D-A3-2). Helper `_resolve_sigma_pair(explicit_single, explicit_off, explicit_on, default_off, default_on, flag)` à créer sur le même pattern.
- Sidecar `shape_hourly.meta.parquet` hyperparams JSON ([shape_hourly.py:517-525](pfc_shaping/lt/model/shape_hourly.py#L517-L525), 5bis-A D-03) — schema déjà extensible (sort_keys=True). 5bis-B ajoute 5 nouvelles keys (`sigma_off`, `sigma_on`, `hydro_weight_sigma_off`, `hydro_weight_sigma_on`, `sigma_resolved`, `hydro_weight_sigma_resolved`) ; reload code [shape_hourly.py:560-573](pfc_shaping/lt/model/shape_hourly.py#L560-L573) doit gérer la cross-plan compat (sidecar pré-5bis-B avec seulement `sigma` legacy → fallback `sigma_off = sigma_on = sigma`).
- Fixture pattern `tests/fixtures/_generate_baseline.py` (5bis-A D-10) — modèle pour `tests/fixtures/_generate_bowl_fixture.py` (D-A4-1). Reuse synthetic forwards + seed=42 convention.
- Convention long-format parquet ([shape_hourly.py:441-525](pfc_shaping/lt/model/shape_hourly.py#L441-L525)) — fixture bowl_seed42 suit le même pattern (colonnes typées explicites).

### Established Patterns
- **Freeze-at-init + persist-in-sidecar + parquet-wins-over-env** (5bis-A D-06..D-08) : appliqué à `use_seasonal_hourly`. 5bis-B étend le même pattern à `sigma_off`, `sigma_on`, `hydro_weight_sigma_off`, `hydro_weight_sigma_on`. Anti train/serve skew.
- **Tolerance contract atol=1e-12, rtol=0** (5bis-A REVIEWS addendum) : MÊME contract pour les nouvelles baselines flag=ON (`baseline_pfc_seed42_bowl.parquet`). Pas byte equivalence parquet — `assert_frame_equal(check_exact=False, atol=1e-12, rtol=0)` + identical columns/dtypes/index/sort order.
- **Convention de naming sidecar `${stem}.meta.parquet`** (5bis-A post-REVIEWS rename depuis `_meta.parquet`) : préservée. Les nouvelles keys hyperparams JSON vivent dans le même sidecar `shape_hourly.meta.parquet`.
- **Conftest autouse env-var hygiene** ([tests/conftest.py](tests/conftest.py), 5bis-A D-12) : déjà en place, snapshot/restore `PFC_LT_*` env vars per test. 5bis-B hérite automatiquement.
- **`gsd-executor` mode no-worktree, 1 plan = 1 wave** (5bis-A): même mode d'exécution pour 5bis-B (3 plans séquentiels).

### Integration Points
- `assembler.build()` à [assembler.py:307-345](pfc_shaping/lt/model/assembler.py#L307-L345) — point d'entrée unique où le flag gate le comportement numérique. 5bis-A préparait le plumbing (`self.sh._use_seasonal_hourly` accessible), 5bis-B l'utilise enfin via la branche `if self.sh._use_seasonal_hourly: ... split level/anomaly ...`.
- `ShapeHourly._apply_hydro_analogue_weights` à [shape_hourly.py:839](pfc_shaping/lt/model/shape_hourly.py#L839) — méthode privée, invoquée à un seul callsite ([shape_hourly.py:243](pfc_shaping/lt/model/shape_hourly.py#L243)) depuis `fit()`. Refactor local, surface contenue.
- `ShapeHourly.__init__` extension : tous les callsites legacy (autoresearch, rolling_update, tests) doivent continuer à fonctionner sans changement de signature externe. Backward-compat via resolution precedence D-A3-2.

</code_context>

<specifics>
## Specific Ideas

### Use case business à préserver dans le mental model
FMV vend des **profile deals GRD** (bloc nuit 18h-9h) + rachète production solaire (souvent WE, blocs OP1/OP2 EEX non tradables). Pricing actuel basé sur HFC OMPEX = **sous-estime systématique** du bowl solaire et de l'evening peak hivernal. Erreur typique : ~250k€/deal/5€/MWh shape. **5bis-B livre la valeur métier** : creuser ce bowl pour que les profile deals soient pricés au juste prix. Phase 10 (refondu) le valide empiriquement vs HFC OMPEX.

### Méthode scientifique (innovation EPFL angle)
- **Ground truth analytique** : fixture bowl_seed42 a un bowl injecté explicitement, `np.ptp` attendu calculable a priori. Les tests SC #1/#2/#3 sont des **expériences contrôlées**, pas des mesures sur données incontrôlées. Contraste avec audit-deep-lt qui mesurait sur EPEX réel — utile pour orienter le scope, mais pas pour valider la math en CI.
- **Measure-then-assert pattern** pour les thresholds (D-A4-5, D-A4-6) : researcher exécute dry-run, capture amplitude observée, commit threshold avec safety margin documenté. Évite "magic numbers" non sourcés.
- **Telemetry-driven invariant detection** (D-A2-5) : log + warning sur drift `|level - 1.0|`, future-proof si Phase 5 ou 5ter modifient la normalisation post-MSFC log-prix.

### Convention quant "no-op refactor first, math change second" (suite de 5bis-A)
- 5bis-A a livré l'infra no-op. **5bis-B est la première phase math change** qui exploite cette infra. Le contrat clé : flag=OFF reproduit baseline 5bis-A bit-pour-bit (atol=1e-12). Toute régression de ce test = ship-blocker.
- Nouvelle convention établie par 5bis-B (pour Phase 5 / 5ter / phases shape futures) : **chaque flag transition ou math change atomique = nouvelle baseline frozen séparée** (D-A4-9). Pattern : `baseline_pfc_seed42_{feature_name}.parquet`.

### Innovation gating pour SC #2 (€5/MWh delta)
- 5bis-B passe SC #2 sur fixture SYNTH = condition **nécessaire** (math correcte).
- Phase 10 valide sur HFC OMPEX RÉEL = condition **suffisante** (data fit).
- Failure synth en 5bis-B = math broken, ship-blocker immédiat.
- Pass synth (5bis-B) + fail real (Phase 10) = fixture-real gap, informe future fixture design (pas un rollback 5bis-B).
- C'est l'innovation : séparer math-validation et data-validation en gates distincts.

</specifics>

<deferred>
## Deferred Ideas

### Vers Phase 10 refondu (real-data validation gate)
- Backtest réel HFC OMPEX 2024-2025 par bloc client (10-15 weekday, 18-9 weekday, 12-14 weekend summer).
- Cible KPI : Δ MAE bloc ≤ -1.5 €/MWh vs HFC OMPEX.
- **Gating flip flag default OFF→ON** : décision verrouillée D-FLIP-1.
- Nécessite accès `H:\` (poste FMV ou Databricks Gold, pas Mac Mini Cloud Desktop actuel).

### Vers Phase 5 (PFC peut être négative)
- Tous les 4 floors silencieux : MSFC `np.maximum(B, 1.0)`, m_factor floor 0.1, F_WV_FLOOR, peak ratio ≥ 1.
- Convention baseline pattern (D-A4-9) suivie : nouvelle baseline frozen `baseline_pfc_seed42_negative.parquet` à la livraison de Phase 5.

### Vers Phase 5ter (distribution probabiliste)
- `pfc_block_distribution(start, end, hours_mask) → (p10, p50, p90)` Monte-Carlo shape N=500.
- Réutilise infra 5bis-B (split level/anomaly, σ_on tuning) pour générer trajectoires shape stochastiques cohérentes avec le bowl deepening.

### Vers ROADMAP backlog (pas de phase assignée)
- Cleanup pre-doc `.planning/phases/05bis-shape-seasonal-hourly/CONTEXT.md` — SUPERSEDED, soit supprimer soit ajouter une note "SUPERSEDED, see 05B + 05C". À trancher en planning ou commit final 5bis-B.
- Calibration empirique de `sigma_on` (D-A3-4) et `hydro_weight_sigma_on` (D-A1-4) au-delà des defaults ROADMAP-proposed : research possible post Phase 10 pour A/B online tuning sur HFC OMPEX réel.
- Recalibration éventuelle des knots `shape_freedom['f_H']` pour level damping (D-A2-6) post-mesure du bowl amplitude sur fixture EPEX réelle.

### Non-deferral notes
- **σ_on=0.25 et hydro_weight_sigma_on TBD** : NE PAS reporter à plus tard les defaults — `gsd-phase-researcher` les calibre dans le cadre du Plan 05C-01-PLAN.md (Lever 1 wave 1) via dry-run sur fixture bowl_seed42, et les commit comme defaults dans la signature ctor. Pas de magic-number-deferred.

</deferred>

---

*Phase: 5bis-B — Shape Hourly Bowl-Deepening (math change)*
*Context gathered: 2026-05-19*
*Convention quant: no-op refactor first (5bis-A ✓), math change second (5bis-B, this phase)*
