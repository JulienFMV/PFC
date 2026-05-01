# PFC 2026 — Feuille de route d'amélioration structurée

Synthèse des recommandations issues de la revue d'experts (panels DA SOTA, ID/CH, PFC/calibration) sur la base des publications 2024–2026. Plan séquencé par **phases**, avec **gates de validation**, dépendances et cibles code précises.

Branche de travail : `claude/review-power-forecast-model-R3Ru4`.

---

## Principes directeurs

1. **Hippocrate** : ne pas dégrader la PFC long-terme en optimisant le DA. Toute modification = backtest comparatif obligatoire (RMSE, RMSE_shape, MAE, biais, IC80 coverage, energy-consistency vs forwards quotés).
2. **Scinder par horizon** : le DA n'est pas du shaping. Le 15-min ID n'est pas du Y+3.
3. **No-regret first** : commencer par les correctifs structurels gratuits (renormalisation, suppression `f_bridge`).
4. **Innovation contrôlée** : chaque innovation passe par un A/B harness et un gate quantitatif avant promotion en prod.
5. **Auditabilité** : chaque facteur doit rester explicable (CH, marché peu liquide, audit régulatoire).

---

## Architecture cible (vue d'ensemble)

```
Horizon       │ Modèle principal              │ Ré-anchor / niveau           │ Probabiliste
──────────────┼───────────────────────────────┼──────────────────────────────┼──────────────────────
J+1 (DA)      │ LEAR / DDNN (existe déjà:     │ —                            │ IDR + Conformal EnbPI
              │  lear_forecaster.py)          │                              │
IDA1→IDA3     │ shape_intraday v2             │ price DA prédit              │ pinball loss native
M+1..M+3      │ PFC actuelle                  │ niveau LEAR rolling          │ bootstrap + IDR
M+3..Y+3      │ PFC actuelle, sans f_bridge,  │ EEX forwards + drivers       │ bootstrap horizon-aware
              │ avec renormalisation post-    │ fundamentaux gas/EUA         │
              │ damping et calibration jointe │ (Hirsch-Ziel 2024)           │
              │ spot+forward (Kiesel-         │                              │
              │ Paraschiv)                    │                              │
```

---

## Phase 0 — Fondations (semaine 0)

**But** : avoir un harness de validation fiable avant toute modif.

| Tâche | Cible | DoD |
|---|---|---|
| 0.1 Réinstaller env (numpy/pandas/scipy/sklearn/lightgbm) | `requirements.txt` | `python3 autoresearch_eval.py` retourne `rmse:` non-vide |
| 0.2 Tag baseline | `git tag baseline-2026-05` | Tag créé, métriques figées dans `results.tsv` |
| 0.3 Étendre `autoresearch_eval.py` (read-only) ou ajouter `eval_extended.py` avec : RMSE par horizon (DA / M+1..6 / Y+2..3), RMSE peak-only, RMSE_negprice, IC80 par horizon | `pfc_shaping/eval/` (nouveau) | Tableau de métriques exporté en JSON et `results.tsv` enrichi |
| 0.4 Snapshot test data : window OOS Jan–Mar 2026 figée, hors training | `data/oos_window.json` | Window stable réutilisée par toutes les phases |

**Gate** : baseline reproductible à ±0.01 RMSE entre 2 runs.

---

## Phase 1 — Correctifs structurels gratuits (semaine 1)

**But** : retirer les biais connus, sans nouveau modèle. Tout est local au code existant.

### 1.1 Supprimer `f_bridge` post-calibration (panel PFC)

- **Pourquoi** : `_rebalance_near_term_bridge` (assembler.py:796) ré-injecte une prime peak après la calibration arbitrage-free → arbitrage statique 0.3–0.8 €/MWh sur les peak D+10..M+3 quotés.
- **Action** :
  - Retirer l'appel `_rebalance_near_term_bridge` (assembler.py:281–287).
  - Conserver `_near_term_bridge_factor` (avant cal) car il sert de prior peak ; le déplacer comme **contrainte molle** (penalty L2 sur deviation peak/prior) dans `calibration/arbitrage_free.py`.
- **Gate** : RMSE DA inchangé ±0.5 €/MWh, energy-consistency Peak/Off-peak strictement respectée (arbitrage_check.py).
- **Réf** : Caldana-Fusai-Roncoroni 2017 ; Fleten-Lemming 2003.

### 1.2 Renormaliser après `shape_freedom` damping (panel PFC)

- **Pourquoi** : `f_S = 1 + (f_S − 1) × shape_freedom["f_S"]` (assembler.py:247–251) écrase la moyenne de chaque facteur sous 1 → biais niveau 1–3% Y+2/Y+3 absorbé par la calibration mais polluant la forme.
- **Action** : après chaque damping, renormaliser le facteur par bucket *natif* : f_S à mean=1 par mois, f_W à mean=1 par ISO-week, f_H à mean=1 par jour, f_Q à mean=1 par heure. Ajouter un test unitaire `tests/test_shape_freedom.py`.
- **Gate** : `mean(f_S over month) == 1.0` dans 1e-9, RMSE_shape ≤ baseline.
- **Réf** : Latini-Piccirilli-Vargiolu 2019.

### 1.3 Espace additif sur les heures à risque négatif (panel ID/CH)

- **Pourquoi** : `_fit_base` (shape_intraday.py:368) utilise `ratio = price/mean_hour_price` ; explose quand `mean_hour_price → 0` (28% des heures PV en H1 2025 en DE selon pv-magazine).
- **Action** : pour les heures solaires (10–16h UTC, été), basculer en **espace additif** : `delta_q = price_q − mean_hour_price`, puis renormaliser pour garantir mean(delta) = 0 et |sum| = 0. Exposer un flag `use_additive_for_low_price_hours: true` dans `config.yaml`.
- **Gate** : RMSE_shape sur heures négatives −10 à −20 %, pas de dégradation sur autres heures.
- **Réf** : Latini-Piccirilli-Vargiolu 2019.

### 1.4 Recalibrer table CH `_SEASONAL_RATIOS_CH` (panel PFC)

- **Pourquoi** : table hardcodée 2010-style (Jul:0.90), incompatible avec la réalité 2024–2025 (Jul ≈ 0.78–0.85, inversion peak/off-peak certains week-ends solaires).
- **Action** : remplacer par une régression saisonnière glissante sur 3–5 dernières années CH+DE (avec dummy week-end PV) calculée à `fit` time. Conserver la table hardcodée comme fallback ultime. Cible : `forward_proxy.py` + `assembler._compute_f_S`.
- **Gate** : MAE forme été −5 à −10 %.
- **Réf** : Maciejowska et al. 2025.

**Effort total Phase 1** : ~3–5 jours dev. Risque faible, gains 1–3 % niveau, 5–20 % shape sur sous-régimes.

---

## Phase 2 — Hybride DA = LEAR rolling (semaines 2–3)

**But** : remplacer l'estimation DA par le SOTA reconnu (Lago 2021).

### 2.1 Audit de l'existant `lear_forecaster.py`

- 2012 lignes existent déjà : vérifier le périmètre (lags ? exogènes ? rolling ?), couverture tests.
- Cible : produire `docs/lear_audit.md` (gap analysis vs Lago 2021 : variantes LEAR-RM, LEAR-RW, exogènes load+wind+solar+gas+CO2, recalibrage glissant).

### 2.2 Brancher LEAR comme source de niveau DA dans `assembler.build`

- Pour `months_ahead == 0` (J+1) **uniquement** : remplacer `B(year/qm)` par la prédiction LEAR du jour.
- Garder le shaping `f_W × f_H × f_Q × f_WV` au-dessus de LEAR pour atteindre le 15-min.
- Cible : `assembler.py:_resolve_base` + nouveau `model/da_anchor.py` qui choisit la source de niveau (LEAR > forward > spot proxy).

### 2.3 Drivers fondamentaux (gas TTF, EUA, charbon API2) intégrés à LEAR

- Étendre `data/ingest_*` pour récupérer TTF + EUA quotidiens (sources publiques EEX, ICE).
- Cible : `data/ingest_fundamentals.py` (nouveau).

### 2.4 Validation A/B

- Backtest 2024–2026 OOS, comparaison RMSE DA vs (a) baseline PFC, (b) LEAR seul, (c) PFC × LEAR niveau.
- **Gate** : −30 % RMSE DA minimum vs baseline ; sinon retour Phase 1 et debug.
- **Réf** : Lago 2021 ; Hirsch-Ziel 2024 (drivers fondamentaux).

**Effort** : ~2 semaines (l'essentiel du LEAR existe).

---

## Phase 3 — Refonte intraday (semaines 4–6)

**But** : passer le 15-min sous régime DA-15-min (post Oct-2025) avec features RES-gradient.

### 3.1 Régime-switch 2025-10-01

- **Pourquoi** : DA 15-min go-live FfE → sawtooth pattern, half-life 180 j mémorise un pré-régime mort.
- **Action** : dans `shape_intraday.fit`, séparer training en deux époques : `pre_2025_10_01` (poids 0 ou très faible) et `post_2025_10_01` (poids exponentiel). Ajouter dummy `is_da15min_regime` dans Layer 2.
- **Gate** : biais 15-min Oct-Dec 2025 −10 à −20 %.

### 3.2 Layer 2 v2 — features RES-gradient

- **Remplacer** `solar_regime ∈ {0,1,2}` par :
  - `dPV/dt` : gradient PV intra-heure (pente `solar(q+1) − solar(q-1)`)
  - `PV_forecast_error` IDA1→IDA2 (si données ENTSO-E IC disponibles)
  - `dLoad/dt` : gradient charge intra-heure
  - `is_neg_price_regime` : indicateur prix négatif lag-1
  - `spread_DE_CH_lag1` : couplage frontière (CH price-taker DE 70% du temps)
- Cible : `data/ingest_entso.py` (calcul gradients) + `shape_intraday.py:_fit_correction`.

### 3.3 LightGBM monotone-constrained à la place de Ridge

- Remplacer `RidgeCV` (clip ±0.05) par **LightGBM** avec contraintes monotones (ex : monotone décroissant en `dPV/dt` pour quart 1 de l'heure 11h).
- Sortie en **softmax sur 4 QH** garantissant nativement mean(f_Q) = 1 (panel ID/CH).
- Loss **pinball** pour passer probabiliste dès la sortie (10 quantiles).
- Cible : nouveau `shape_intraday_v2.py` parallèle, switchable via `config.yaml`.
- **Gate** : −15 à −25 % RMSE_shape sur fenêtre OOS post Oct-2025.
- **Réf** : Hirsch-Ziel 2024 ; Janke et al. 2024.

**Effort** : ~3 semaines (entraînement LightGBM + features ENTSO-E enrichies).

---

## Phase 4 — Long-horizon fondamentaux & calibration jointe (semaines 7–9)

**But** : fonder Y+2/Y+3 sur des drivers structurels au lieu d'extrapoler le spot.

### 4.1 Drivers fondamentaux dans `forward_proxy`

- Régresser `B(year)` sur (gas_forward_calY, EUA_forward_calY, RES_buildout_pipeline, hydro_storage_z) au lieu d'un mean-reversion empirique.
- Cible : `forward_proxy.py` + `data/ingest_fundamentals.py` (étendu).
- **Réf** : Hirsch-Ziel 2024 ; Maciejowska et al. 2025.

### 4.2 Calibration jointe spot+forward (Kiesel-Paraschiv)

- Au lieu de calibrer la PFC uniquement contre les forwards, ajouter une contrainte molle d'ajustement aux derniers 24 mois de spot historique (terme L2 sur résidu).
- Cible : `calibration/arbitrage_free.py` (nouveau mode `joint_calibration=True`).
- **Réf** : Kiesel-Paraschiv-Sætherø 2019.

### 4.3 Interpolateur monotone-convex (Hagan-West)

- Ajouter en complément du B-spline d'ordre 4 dans `msfc_spline.py` pour traiter le "swelling" documenté post-2022.
- Mode sélectionnable via config (`interpolator: bspline | hagan_west`).
- **Réf** : Caldana-Fusai-Roncoroni 2017.

**Effort** : ~3 semaines.

---

## Phase 5 — Probabiliste end-to-end (semaines 10–11)

**But** : intervalles distribution-free valides à tous horizons.

### 5.1 IDR post-processing (Lipiecki-Uniejewski-Weron 2024)

- Wrapper isotonic distributional regression sur `uncertainty.compute`.
- Cible : `model/uncertainty.py` (nouveau mode `idr=True`).

### 5.2 Conformal EnbPI / SPCI

- Wrapper conformal prediction sur sortie point pour intervalles 90/95 % distribution-free.
- Cible : `model/uncertainty.py` (nouveau mode `conformal=True`).
- **Gate** : IC80 coverage entre 78 % et 82 % sur fenêtre OOS de toutes phases.
- **Réf** : O'Connor 2025 ; Lipiecki et al. 2024.

**Effort** : ~2 semaines.

---

## Phase 6 — Innovation (semaines 12+)

**But** : hisser le modèle au niveau frontière 2026.

### 6.1 Foundation models en challenger zero-shot

- Utiliser `foundation_forecaster.py` existant (Chronos / TimesFM / Moirai / TiRex) en challenger pour le DA et le M+1..M+3.
- A/B obligatoire : si TSFM bat LEAR de >5 % stable, promouvoir.
- **Réf** : TSFM-EPF benchmark 2025 ; Toto / Timer-XL (déjà dans `docs/research/`).

### 6.2 DDNN / NBEATSx en deuxième challenger

- Évaluer en parallèle de LEAR (Marcjasz 2023, Olivares 2023) pour DA pur.

### 6.3 Diffusion / generative pour les queues ID

- Schwenk-Nebbe & Pinson 2024 : utile pour la VaR et la calibration des queues 5/95.
- Travail de recherche, sortie 2027.

### 6.4 Hierarchical Bayesian QH (Janke 2024)

- Si #5 (Phase 5) probabiliste s'avère insuffisant en queues, ajouter un layer hiérarchique bayésien sur QH dans heure (PyMC ou NumPyro).

**Effort** : open-ended, gouverné par les gates A/B.

---

## Tableau récapitulatif — gates de validation

| Phase | Métrique pivot | Seuil | Source |
|---|---|---|---|
| 1 | RMSE_shape sur heures négatives | −10 à −20 % | OOS Apr–May 2025 |
| 1 | Energy-consistency Peak/OffPeak | erreur < 0.1 % | `_check_energy_consistency` |
| 2 | RMSE DA J+1 | −30 % vs baseline | OOS Jan–Mar 2026 |
| 3 | RMSE_shape Oct-Dec 2025 | −15 à −25 % | OOS post-rupture |
| 4 | RMSE Y+2/Y+3 vs benchmark HFC | −15 à −20 % | window 2024–2025 |
| 5 | IC80 coverage | ∈ [78 %, 82 %] | OOS toutes phases |

---

## Risques & mitigations

| Risque | Mitigation |
|---|---|
| Régression silencieuse Y+3 lors des changements DA | Backtest multi-horizon obligatoire à chaque PR |
| Sur-fit Layer 2 v2 sur fenêtre courte post-Oct-2025 | Régularisation forte + walk-forward CV |
| Données ENTSO-E indisponibles offline | Tous les nouveaux features doivent avoir un fallback neutre |
| Foundation models gourmands GPU | Mode CPU only avec batch limité ; A/B optionnel |
| Calibration jointe instable | Schur smoothing + fallback raw + alerte si non-convergence |

---

## Bibliographie de référence (à archiver dans `docs/research/`)

1. Lago, J. et al. (2021) — *Applied Energy* — EPF benchmark/LEAR
2. Hirsch, S. & Ziel, F. (2024) — *The Energy Journal* + *ASMBI* — ID GAMLSS multivariate ★
3. Marcjasz, G., Narajewski, M., Weron, R., Ziel, F. (2023) — *Energy Economics* — DDNN
4. Olivares, K. et al. (2023) — *IJF* — NBEATSx
5. Lipiecki, M., Uniejewski, B., Weron, R. (2024) — *Energy Economics* — IDR
6. Janke, P., Steinke, F. et al. (2024) — *Applied Energy* — Bayesian hierarchical ID
7. Maciejowska, K., Nitka, W., Weron, R. (2025) — *RSER* — DA→mid/long-term
8. Caldana, R., Fusai, G., Roncoroni, A. (2017) — *EJOR* — monotone-convex
9. Kiesel, R., Paraschiv, F., Sætherø, A. (2019, rev. 2023) — *CMS* — joint spot+fwd HPFC
10. Latini, L., Piccirilli, M., Vargiolu, T. (2019) — *Energy Economics* — additive no-arbitrage
11. Borovkova, S., Ladokhin, S., Schmeck, M. (2024) — SSRN — business-time seasonality
12. FfE (2025) — Sawtooth pattern post DA 15-min go-live ★
13. O'Connor et al. (2025) — Conformal Prediction EPF
14. Uniejewski, B. & Weron, R. (2025) — *J. Commodity Markets* — SQRA

---

## Suivi opérationnel

- Chaque phase = une PR distincte, label `phase-N`, base `claude/review-power-forecast-model-R3Ru4`.
- `results.tsv` étendu avec une ligne par PR mergée (commit, RMSE, RMSE_shape, RMSE_DA, IC80, status).
- Décisions importantes consignées dans `docs/decisions/` (ADR léger).
- Revue d'expert externe (panel actuel) à mi-parcours (post Phase 3) pour valider la direction.
