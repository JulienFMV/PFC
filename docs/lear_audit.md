# Phase 2.1 — Audit `lear_forecaster.py` vs Lago 2021 SOTA

Date : 2026-05-01
Branche : `claude/review-power-forecast-model-R3Ru4`

## Constat

`pfc_shaping/model/lear_forecaster.py` (2012 lignes) **dépasse déjà** le LEAR canonique de Lago et al. (2021).
C'est un **ensemble multi-modèle** déjà branché en production via `LEARForecaster.blend_with_pfc`
(D+1..D+10 overlay) appelé depuis `run_pfc_production.py` lignes 388–518.

## Composants présents (mapping vs littérature)

| Brique | Présence | Référence |
|---|---|---|
| 24 régressions LASSO indépendantes (1 par heure) | ✓ `_fit_lasso_for_hour` (l.1106) | Lago et al. 2021 |
| Asinh variance-stabilizing transform | ✓ `_asinh_transform` / `_causal_asinh_transform` (l.68, 82) | Uniejewski-Weron 2018 |
| Multi-window calibration averaging | ✓ `CALIBRATION_WINDOWS = [42, 56, 84, 180, 365]` | Marcjasz-Uniejewski-Weron 2018 |
| Lag structure J-1, J-2, J-3, J-7, J-14 | ✓ `LAGS_DAYS = [1, 2, 3, 7, 14]` | Lago 2021 |
| Cross-border DE prices comme exogène | ✓ `prices_de_h_`, `de_price` feature | Hirsch-Ziel 2024 |
| ENTSO-E load/solar/wind CH+DE | ✓ `entso_cols` (l.181) | Lago 2021 |
| Outages, hydro fill | ✓ `outages_mw`, `hydro_fill` | Maciejowska 2025 |
| Commodities gas/CO2/coal | ✓ via `commodities` argument | Hirsch-Ziel 2024 |
| DE renewable forecasts (DA wind+solar) | ✓ auto-loaded `de_renewable_forecast.parquet` | Janke et al. 2024 |
| MLP ensemble member | ✓ `_fit_mlp_ensemble` (l.1163) | Lago 2021 (DNN-EPF) |
| Foundation model (Chronos) ensemble | ✓ `FoundationForecaster` import | Toto/Chronos 2025 |
| QRA (Quantile Regression Averaging) | ✓ `_fit_qra` (l.848) | Marcjasz-Nowotarski-Weron 2020 |
| Conformal prediction intervals | ✓ `_compute_conformal_residuals` (l.1200) | O'Connor 2025 |
| AR error correction (lag-1 ≈ 0.50) | ✓ `_recent_bias_by_regime` (l.911) | Uniejewski 2024 |
| Per-hour variance recalibration | ✓ `_recalibrate_variance` (l.1236) | epftoolbox |
| Production blend D+1..D+10 | ✓ `blend_with_pfc` (l.1945) | — |

Conclusion : **aucun élément majeur de Lago 2021 n'est manquant**. La couche est en réalité plus
sophistiquée (foundation models + QRA + conformal en plus).

## Gap réel identifié

Le LEAR existe et tourne en production, mais **le harness d'évaluation ne le mesure pas** :

* `autoresearch_eval.py` (figé) construit la PFC sans appeler `blend_with_pfc`.
* `eval_extended.py` (Phase 0.3) avait la même limite avant cette PR.

Conséquence : toute "amélioration" de PFC mesurée jusqu'ici l'était PFC-only, alors qu'en production
le DA J+1..J+10 est en réalité driven par LEAR.

## Mesure du delta (Phase 2.2)

Après ajout du flag `--with-lear` à `eval_extended.py`, sur la fenêtre `winter_2026q1` restreinte
aux 10 premiers jours (territoire effectif LEAR) :

| Mode | RMSE | MAE | bias | RMSE_shape | IC80 |
|---|---:|---:|---:|---:|---:|
| PFC-only | 20.45 | 14.91 | −7.71 | 19.33 | 0.922 |
| PFC + LEAR overlay | 20.40 | 14.63 | −11.32 | **17.23** | 0.922 |
| Δ | −0.05 | −0.28 | −3.61 | **−2.10** (−10.9 %) | — |

**Lecture** : LEAR fait son métier — il améliore très significativement la **forme intra-jour**
(RMSE_shape −10.9 %) et le MAE. Il aggrave néanmoins le biais de niveau (déjà négatif dans la
PFC-only, encore plus négatif après LEAR), parce qu'à la fois la PFC et LEAR sous-estiment le
niveau réel sur ce window. Le RMSE total reste plat car le gain de forme compense la dégradation
de niveau.

## Implications pour la roadmap

1. **Phase 2 originale ("LEAR rolling DA")** est de fait **déjà implémentée**. Le travail futur
   sur LEAR est marginal (tuning hyperparamètres, peut-être recalibrage du blend window 8-11 jours
   → 5-7 jours).
2. Le **biais de niveau résiduel** (−7 à −11 EUR/MWh) confirme que la priorité bascule sur la
   **Phase 4** : drivers fondamentaux gas/EUA ancrant le niveau de B(year). Une LEAR seule ne le
   résorbera pas.
3. Le harness `eval_extended.py --with-lear` doit devenir le **benchmark de référence DA** pour
   toutes les modifications ultérieures touchant la fenêtre M+0..M+1.

## Améliorations LEAR mineures envisageables (low priority)

| Idée | Bénéfice attendu | Complexité |
|---|---|---|
| Recalibrer le horizon de blend 8-11 → 5-7 jours (LEAR plus dégradé après J+5) | RMSE −0.5 sur J+6..J+10 | Faible |
| Ajouter LightGBM monotone-constrained comme 4ᵉ membre d'ensemble | RMSE −0.3 si bien calibré | Moyenne |
| IDR post-processing sur quantiles LEAR (Lipiecki-Uniejewski-Weron 2024) | IC80 mieux calibré | Faible |
| Régularisation L2 supplémentaire sur le DE residual feature (multicolinéarité) | — | Faible |

À reprendre uniquement après Phase 4 (drivers fondamentaux) qui adresse la cause racine du biais
de niveau.
