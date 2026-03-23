# AUDIT PROFOND DU PROJET PFC -- 2026-03-23

**6 agents experts** | **71 fichiers Python audites** | **~15,000 lignes de code**

---

## SYNTHESE EXECUTIVE

| Severite | Nb findings | Impact |
|----------|-------------|--------|
| **P0 CRITICAL** | 14 | Resultats incorrects, crashes production, data leakage |
| **P1 HIGH** | 22 | Degradation precision, biais systematiques, fiabilite |
| **P2 MEDIUM** | 30 | Sous-optimalites, edge cases, performance |
| **P3 LOW** | 25 | Code smells, securite mineure, documentation |
| **TOTAL** | **91** | |

Les issues ont ete identifiees par 6 agents specialises (Bug Tracking, Modelisation, Mathematiques/Statistiques, ML/LLM, Python Quality, Dashboard/Pipeline) puis dedupliquees et consolidees.

---

## P0 -- CRITICAL (14 issues)

### P0-01: LEAR -- Data leakage: features exogenes d+0 sans lag [ML]
**Fichier:** `pfc_shaping/model/lear_forecaster.py:459-462`
**Confirme par:** 1 agent (ML/LLM)

Le feature builder LEAR utilise les valeurs exogenes du **meme jour** (load_mw, solar_mw, wind_mw, outages_mw) a l'heure cible sans lag. En entrainement, ce sont les valeurs realisees. En prediction, on utilise une moyenne 28j. Ceci cree un **mismatch train/inference** qui gonfle la performance in-sample et degrade les predictions production, surtout aux heures solaires (h11-h14).

**Fix:** Supprimer les features d+0 pour les variables CH exogenes. Garder uniquement d-1 et d-7.

### P0-02: reference_date non propage a ShapeHourly.apply() [MODELING+BUG]
**Fichier:** `pfc_shaping/model/assembler.py:158-162`
**Confirme par:** 3 agents (Modeling, Bug Tracker, ML)

L'assembleur ne passe jamais `reference_date` a `self.sh.apply()`. En backtest, les tendances horizon-dependantes sont calculees par rapport a `now()` au lieu de la date as-of. Ceci introduit un **look-ahead bias** dans toutes les valeurs f_H ajustees par tendance.

**Fix:** `self.sh.apply(idx, cal, reference_date=reference_date)`

### P0-03: LEAR -- Validation in-sample pour les poids de fenetres [ML]
**Fichier:** `pfc_shaping/model/lear_forecaster.py:878-887`

La MAE de validation pour le weighting inverse-MAE est calculee sur les **14 dernieres observations du training window** -- qui sont incluses dans les donnees d'entrainement. C'est une validation in-sample, pas out-of-time.

**Fix:** Exclure les derniers `n_val` observations du fit LASSO: `model.fit(X_w.iloc[:-n_val], y_w.iloc[:-n_val])`.

### P0-04: MLP/GBM -- Random validation split sur donnees temporelles [ML]
**Fichier:** `lear_forecaster.py:919-920`, `shape_hourly_mlp.py:186-187`

sklearn `MLPRegressor(early_stopping=True, validation_fraction=0.15)` fait un **split aleatoire** pour le validation set. Sur des series temporelles, des points futurs fuient dans le training, faussant l'early stopping.

**Fix:** Split chronologique (derniers 15%) ou desactiver early stopping avec max_iter fixe.

### P0-05: MSFC -- Tolerance de convergence trop lache (0.5 EUR/MWh) [MODELING+MATH]
**Fichier:** `pfc_shaping/model/msfc_spline.py:161-197`
**Confirme par:** 2 agents (Modeling, Math/Stats)

L'enforcement iteratif des contraintes de moyenne utilise un seuil de 0.5 EUR/MWh avec seulement 10 iterations et damping 0.8. Sur un prix de 75 EUR/MWh, c'est 0.67% d'erreur -- suffisant pour creer un arbitrage detectable au niveau mensuel. De plus, le floor de positivite `np.maximum(B_current, 1.0)` a l'interieur de la boucle corrompt le calcul de la moyenne.

**Fix:** Reduire tolerance a 0.01 EUR/MWh, augmenter max_iter a 50, deplacer le floor hors de la boucle.

### P0-06: Water value -- Regression sur prix non-stationnaires [MODELING+MATH]
**Fichier:** `pfc_shaping/model/water_value.py:162-196`
**Confirme par:** 2 agents (Modeling, Math/Stats)

La regression `prix_mensuel ~ trend + fill_deviation * season` utilise les **prix absolus** comme variable dependante. Les prix sont massivement non-stationnaires (doublement/triplement 2021-2022). Le trend absorbe la variance, et les coefficients fill_deviation sont biaises par les changements de niveau.

**Fix:** Regresser sur les deviations relatives (prix / moyenne glissante 12 mois) ou ajouter des effets fixes par annee.

### P0-07: Forward proxy -- Moyenne simple au lieu de moyenne ponderee par heures [MODELING+MATH+BUG]
**Fichier:** `pfc_shaping/data/forward_proxy.py:129-133`
**Confirme par:** 3 agents (Modeling, Math/Stats, Bug Tracker)

Les prix trimestriels sont calcules comme `np.mean(q_prices)` au lieu de la moyenne ponderee par heures de livraison: `sum(F_m * h_m) / sum(h_m)`. Erreur de 0.5-1.5 EUR/MWh par trimestre, creant une inconsistance avec le module cascading qui utilise la bonne convention.

**Fix:** Utiliser `count_hours()` du module cascading pour les moyennes ponderees.

### P0-08: Autoresearch eval -- Look-ahead ENTSO-E (donnees realisees comme "forecast") [ML]
**Fichier:** `autoresearch_eval.py:142-145`, `scripts/ab_test_mlp.py:94-96`

L'evaluation autoresearch utilise les **donnees ENTSO-E realisees** de la periode de test comme "perfect forecast". Ceci biaise a la hausse les corrections f_Q layer 2, produisant des parametres sur-optimises pour la production.

**Fix:** Utiliser le forecast climatologique comme en production, ou documenter comme borne superieure.

### P0-09: Multiplicative calibration -- Division par quasi-zero [MATH]
**Fichier:** `pfc_shaping/calibration/arbitrage_free.py:417-420`

Quand S est dans [-0.1, 0.1] (episodes de prix negatifs), la conversion multiplicative `m_factor = P_add / safe_S` produit des valeurs aberrantes. Le clamp a 0.1 detruit la correction additive originale, introduisant des erreurs de plusieurs EUR/MWh.

**Fix:** Utiliser directement le resultat additif quand `|S| < seuil`: `P[mask] = P_add[mask]`.

### P0-10: _map_outages mute le DataFrame d'entree [BUG]
**Fichier:** `pfc_shaping/model/shape_hourly_mlp.py:394-398`

`_map_outages` ajoute des colonnes directement au DataFrame passe en argument. Si le meme DataFrame est reutilise (CH puis DE dans run_pfc_production), les colonnes ajoutees lors du premier appel masquent les vraies donnees.

**Fix:** `outages_df = outages_df.copy()` en debut de methode.

### P0-11: Fonctions _retry retournent None implicitement [BUG]
**Fichiers:** `ingest_epex.py:56`, `ingest_entso.py:58`, `ingest_outages.py:56`, `ingest_smard.py:50`, `ingest_energy_charts.py:52`

Si `max_retries=0`, la boucle for ne s'execute jamais, la fonction retourne `None` implicitement. Les appelants crashent ensuite sur `.to_frame()` ou `.json()` sur None.

**Fix:** `if max_retries < 1: raise ValueError("max_retries must be >= 1")`

### P0-12: Cursor leak Databricks [BUG+PYTHON]
**Fichier:** `pfc_shaping/data/databricks_client.py:64`
**Confirme par:** 2 agents

`_connection.cursor().execute("SELECT 1")` cree un curseur jamais ferme a chaque appel `get_connection()`, epuisant le pool de curseurs du warehouse.

**Fix:** `with _connection.cursor() as cur: cur.execute("SELECT 1")`

### P0-13: DuckDB writes sans transaction [DASHBOARD]
**Fichier:** `pfc_shaping/storage/local_duckdb.py:123-134`

`upsert_run_and_forecast` fait DELETE + INSERT sans transaction. Si le processus crashe entre les deux, les donnees sont perdues.

**Fix:** Wrapper dans `BEGIN TRANSACTION ... COMMIT`.

### P0-14: Stale lock file bloque indefiniment la production [DASHBOARD+BUG]
**Fichier:** `pfc_shaping/pipeline/rolling_update.py:104-115`
**Confirme par:** 3 agents

Le mecanisme de lock ne verifie pas si le processus createur tourne encore. Un crash sans cleanup (OOM, SIGKILL) laisse le lock file indefiniment, bloquant toutes les executions futures. De plus, la verification `exists()` + `write_text()` n'est pas atomique (race condition TOCTOU).

**Fix:** Utiliser `os.open(path, os.O_CREAT | os.O_EXCL)` + PID check + TTL expiry.

---

## P1 -- HIGH (22 issues)

### P1-01: Multiplicative calibration minimise la courbure additive [MODELING]
`arbitrage_free.py:409-421` -- Le mode "multiplicatif" minimise ||delta''(t)||^2 au lieu de ||m''(t)||^2. Sur-penalise les corrections aux heures de pointe.

### P1-02: PCHIP extrapolation aux bornes de la courbe [MODELING]
`msfc_spline.py:106-114` -- `extrapolate=True` produit des prix potentiellement aberrants avant le premier et apres le dernier midpoint mensuel.

### P1-03: f_W normalise par mois au lieu de par semaine [MODELING]
`assembler.py:527-533` -- La normalisation mensuelle distord les relativites jour-type dans les mois avec ratios weekday/weekend inhabituels. Biais ~0.5-1%.

### P1-04: Uncertainty -- Pas d'entree M+0 dans HORIZON_WIDENING [MODELING]
`uncertainty.py:42-47` -- Les CI proche terme sont trop larges (facteur minimum 2.50x applique meme pour M+0).

### P1-05: ShapeHourly trend non pondere temporellement [MODELING]
`shape_hourly.py:337-405` -- Le trend est calcule avec np.polyfit non-pondere, inconsistant avec le halflife=180j du profil de base.

### P1-06: LEAR ElasticNet l1_ratio=0.1 est Ridge, pas LASSO [ML]
`lear_forecaster.py:868` -- 10% L1 / 90% L2 ne fait presque pas de selection de features, contrairement a ce que le nom et la doc indiquent.

### P1-07: LEAR variance recalibration sur predictions in-sample [ML]
`lear_forecaster.py:986-1013` -- `lasso_std` sous-estime la vraie variance OOS, causant une sur-expansion vers les extremes.

### P1-08: GBM re-entraine a chaque iteration de backtest (720x) [ML]
`lear_forecaster.py:1513-1534` -- Gaspillage computationnel massif + biais de poids GBM car MAE in-sample.

### P1-09: Chronos-2 timestamps synthetiques [ML]
`foundation_forecaster.py:178-179` -- Timestamps faux (2020-01-01 au lieu des vrais) cassent potentiellement la detection saisonniere.

### P1-10: Autoresearch fenetre de test fixe [ML]
`autoresearch.py:211-214` -- Iterations multiples sur la meme fenetre de test = overfitting aux conditions recentes.

### P1-11: ShapeHourlyMLP perd les poids temporels [ML]
`shape_hourly_mlp.py:158-192` -- Les observations moyennees ont toutes le meme poids dans le MLP, perdant la ponderation decay.

### P1-12: LEAR AR error correction asymetrique backtest/production [ML]
`lear_forecaster.py:1538-1556` -- Le backtest inclut une correction AR non disponible en production, biaisant la MAE a la baisse.

### P1-13: Chronos-2 Bolt indices de quantiles hardcodes [ML]
`foundation_forecaster.py:264-266` -- Indices [0,4,8] hardcodes sans validation de la shape.

### P1-14: File lock race condition TOCTOU [BUG+PYTHON]
`rolling_update.py:104-115` -- exists() + write_text() non atomique.

### P1-15: _compute_f_S retourne 1.0 sans fallback saisonnier [BUG]
`assembler.py:477-493` -- Si MSFC et calibration echouent, Y+2/Y+3 ont un prix plat sans differenciation saisonniere.

### P1-16: pd.Timestamp.utcnow() deprece [BUG]
`quality_gate.py:69` -- Cassera dans les futures versions de pandas.

### P1-17: DST misalignement dans autoresearch RMSE [BUG]
`autoresearch.py:321` -- `np.repeat(..., 4)[:len()]` desaligne les tableaux pendant les transitions DST.

### P1-18: Pas de retry pour le download hydro SFOE [DASHBOARD]
`ingest_hydro.py:73` -- Contrairement aux autres ingesteurs, pas de retry logic.

### P1-19: Deux pipelines dupliquees et divergentes [DASHBOARD+PYTHON]
`run_pfc_production.py` vs `rolling_update.py` -- Logiques divergentes, bugs fixes dans l'une pas dans l'autre.

### P1-20: HuberRegressor sur-parametre [MODELING+MATH]
`shape_intraday.py:383-386` -- X=ones + intercept = 2 parametres pour 1 valeur.

### P1-21: Structural break utilise Welch t-test au lieu de Chow/Hotelling [MATH]
`structural_break.py:114` -- Quasi zero pouvoir de detection des changements de forme qui preservent la moyenne.

### P1-22: Version pinning absent dans requirements.txt [PYTHON]
3 fichiers requirements.txt avec `>=` uniquement. Builds non-reproductibles pour un systeme de trading.

---

## P2 -- MEDIUM (30 issues)

| # | Fichier | Description | Agent |
|---|---------|-------------|-------|
| 01 | `arbitrage_free.py:550` | warnings.filterwarnings("ignore") sans context manager | MATH+PY |
| 02 | `shape_intraday.py:350-396` | Huber base factors sur-compliques | MODEL |
| 03 | `shape_hourly.py:280` | DST 23h/25h jours: normalisation biaisee | MODEL |
| 04 | `calendar_ch.py:23-36` | Saison Automne = 1 mois, statistiquement fragile | MODEL |
| 05 | `backtest.py:194-199` | Fixed base_price, calibration jamais testee | MODEL |
| 06 | `shape_hourly_mlp.py:8-13` | Docstring/architecture mismatch (9 vs 12 features) | MODEL |
| 07 | `msfc_spline.py:200-235` | _verify_constraints log-only, jamais fail | MODEL |
| 08 | `assembler.py:572-573` | Leap year bug dans le comptage de jours trimestriel | MODEL+BUG |
| 09 | `uncertainty.py:107-120` | Bootstrap vacuous (mean of bootstrapped quantiles = quantile empirique) | MATH |
| 10 | `cascading.py:589-603` | Ratios saisonniers non-renormalises apres extrapolation trend | MATH |
| 11 | `lear_forecaster.py:1019-1020` | Variance recalibration amplifie signal et bruit | MATH |
| 12 | `error_analysis.py:341-343` | ACF Ljung-Box invalide (pas la bonne formule Q) | MATH |
| 13 | `finetune_chronos2.py:97-98` | Pas de gap temporel entre train et test | ML |
| 14 | `ab_test_mlp.py:157-184` | Pas de test de significativite statistique | ML |
| 15 | `backtest.py:93-97` | base_price=70.0 fixe pour toutes les periodes (2021-2024) | ML |
| 16 | `lear_forecaster.py:1257-1369` | Feature builder par string parsing, fragile | ML |
| 17 | `lear_forecaster.py:1606` | MAPE clamp a 1 au lieu de sMAPE | ML |
| 18 | `foundation_forecaster.py:228,264` | "mean" = "median" pour les deux backends | ML |
| 19 | `ingest_hydro.py:216-243` | O(n^2) loop pour z-score (vectorisable) | BUG+PY |
| 20 | `shape_hourly.py:280-281` | Python loop lent pour day_key string creation | BUG |
| 21 | `shape_hourly_mlp.py:259` | Meme pattern lent que ci-dessus | BUG |
| 22 | `forward_proxy.py:50,63` | Timezone Zurich inconsistant | DASHBOARD |
| 23 | `2_pfc_vs_forwards.py:44` | tz_localize unsafe sur index deja tz-aware | DASHBOARD |
| 24 | `utils.py:545-547` | st.cache_data.clear() global (tous users) | DASHBOARD |
| 25 | `1_overview.py:37` | read_parquet sans @st.cache_data | DASHBOARD |
| 26 | `quality_gate.py:32-80` | Pas de check pour timestamps dupliques | DASHBOARD |
| 27 | `11_lear_forecast.py:304` | Lambda correlation reference outer bt DataFrame | DASHBOARD |
| 28 | `run_pfc_production.py` | Script monolithique, execute a l'import | PYTHON |
| 29 | `tests/` | 3 fichiers test pour 71 fichiers source (~4% couverture) | PYTHON |
| 30 | `databricks_client.py:40` | Singleton global non thread-safe | PYTHON |

---

## P3 -- LOW (25 issues)

| # | Fichier | Description | Agent |
|---|---------|-------------|-------|
| 01 | `shape_hourly_mlp.py:324` | Pickle deserialization = risque securite | MODEL+ML+PY |
| 02 | `assembler.py:178,266,381` | Imports circulaires a l'interieur des methodes | MODEL |
| 03 | `compare_hfc.py:73-92` | Pas de gestion timezone dans alignement | MODEL |
| 04 | `shape_hourly_mlp.py:406-421` | Indexation fragile du poids dans _fit_f_W | MODEL |
| 05 | `cascading.py:811-818` | Double normalisation intermediaire redundante | MODEL |
| 06 | `config.yaml:104` | heures_rampe ignore par ShapeIntraday | MODEL |
| 07 | `msfc_spline.py:110` | PCHIP extrapolate=True sans garde-fou plafond | MATH |
| 08 | `uncertainty.py:37-38` | Bootstrap seed hardcode (SEED=42) | MATH |
| 09 | `shape_hourly.py:603-613` | Fallback global non pondere par nb observations | MATH |
| 10 | `lear_forecaster.py:780-798` | spike_uplift constantes magiques | ML |
| 11 | `lear_forecaster.py:81-111` | _causal_asinh_transform code mort | ML+MATH |
| 12 | `lear_forecaster.py:50-60` | holidays package absent = features silencieusement zero | ML |
| 13 | `ingest_epex.py:61` + 4 fichiers | Broad except Exception dans retry | BUG |
| 14 | `databricks_client.py:39` | Singleton non thread-safe | BUG |
| 15 | `calendar_ch.py:114` | Pas de validation frequence index | BUG |
| 16 | `shape_intraday.py:469-470` | except Exception: pass silencieux | BUG |
| 17 | `ingest_forwards.py:158` | Prix zero/negatifs filtres (pourrait etre valide) | BUG |
| 18 | `run_pfc_production.py:522` | Date hardcodee "2026-03-14" | BUG+DASHBOARD |
| 19 | `rolling_update.py:549-558` | NameError possible dans except block | BUG+DASHBOARD |
| 20 | `notify_teams.py:24-36` | Format MessageCard deprece par Microsoft | DASHBOARD |
| 21 | `check_ssl_bundle.py:21` | URL de test avec date hardcodee | DASHBOARD |
| 22 | `run_pfc_production.py:521-679` | print() au lieu de logger.info() | DASHBOARD |
| 23 | `ingest_forwards.py` (multiple) | Mojibake UTF-8 dans docstrings | PYTHON |
| 24 | `tests/test_*.py:20-21` | sys.path.insert hack au lieu de packaging | PYTHON |
| 25 | `pfc_flavors.py:138,174` | np.vectorize pour hot-path (pas vraiment vectorise) | PYTHON |

---

## TOP 10 FIXES PRIORITAIRES

Ces 10 corrections auront le plus grand impact sur la qualite des predictions et la fiabilite du systeme :

### 1. LEAR: Supprimer les features exogenes d+0 (P0-01)
**Impact estime:** -2 a -5 EUR/MWh MAE aux heures solaires
**Effort:** 1h
```python
# lear_forecaster.py:459 -- NE PLUS inclure d0 pour load, solar, wind, outages CH
# Garder uniquement d-1 et d-7
```

### 2. Propager reference_date dans l'assembleur (P0-02)
**Impact:** Elimine le look-ahead bias en backtest, stabilise les courbes production
**Effort:** 30min
```python
# assembler.py:159
self.sh.apply(idx, cal, reference_date=reference_date)
```

### 3. MSFC convergence: tolerance 0.01, max_iter 50 (P0-05)
**Impact:** Garantit le repricing exact des forwards, elimine l'arbitrage residuel
**Effort:** 30min

### 4. Water value: regresser sur prix relatifs (P0-06)
**Impact:** Coefficients beta_wv corrects, meilleure capture de l'effet hydro
**Effort:** 2h

### 5. LEAR validation: hold-out pour les poids de fenetres (P0-03)
**Impact:** Poids de fenetres non biaises, meilleure selection de modele
**Effort:** 1h

### 6. MLP/GBM: split chronologique pour early stopping (P0-04)
**Impact:** Early stopping fiable, pas de data leakage temporel
**Effort:** 2h

### 7. Forward proxy: moyenne ponderee par heures (P0-07)
**Impact:** Coherence energetique avec le module cascading
**Effort:** 30min

### 8. Lock file: atomique + PID check + TTL (P0-14)
**Impact:** Plus de blocage indefini de la production apres crash
**Effort:** 1h

### 9. DuckDB: transactions explicites (P0-13)
**Impact:** Plus de perte de donnees en cas de crash mid-write
**Effort:** 30min

### 10. Autoresearch eval: forecast climatologique (P0-08)
**Impact:** Parametres autoresearch representatifs des conditions production
**Effort:** 2h

---

## ISSUES VALIDEES PAR PLUSIEURS AGENTS (haute confiance)

Ces issues ont ete detectees independamment par 2+ agents, confirmant leur validite :

| Issue | Agents | Confiance |
|-------|--------|-----------|
| Forward proxy moyenne simple | Modeling + Math + Bug | 3/6 |
| reference_date non propage | Modeling + Bug + ML | 3/6 |
| Stale lock file | Dashboard + Bug + Python | 3/6 |
| MSFC convergence tolerance | Modeling + Math | 2/6 |
| Water value non-stationnarite | Modeling + Math | 2/6 |
| Cursor leak Databricks | Bug + Python | 2/6 |
| Leap year jour count | Modeling + Bug | 2/6 |
| Pickle security | Bug + ML + Python | 3/6 |
| HuberRegressor overparametre | Modeling + Math | 2/6 |
| Race condition lock | Bug + Python + Dashboard | 3/6 |

---

## POINTS FORTS DU CODEBASE

Les agents ont egalement identifie des forces notables :
- **Architecture propre** : separation claire data/model/calibration/pipeline/storage
- **Fallback chains** : energy-charts -> SMARD -> ENTSO-E -> cache local
- **Arbitrage-free calibration** avec MSFC methodology (SOTA)
- **Quality gates** avec hard fail sur les donnees critiques
- **Conformal prediction** pour les intervalles LEAR
- **TimeSeriesSplit** pour le CV LASSO
- **Multi-window averaging** pour robustesse
- **DuckDB audit trail** des runs

---

## PLAN D'ACTION RECOMMANDE

**Sprint 1 (cette semaine):** P0-01 a P0-07 (modelisation/ML)
- Impact maximal sur la precision MAE
- Effort total estime: ~10h

**Sprint 2 (semaine prochaine):** P0-08 a P0-14 (infrastructure)
- Fiabilite production
- Effort total estime: ~6h

**Sprint 3:** P1-01 a P1-10 (ameliorations methodologiques)
- Precision incrementale

**Continu:** P2/P3 par ordre d'opportunite

---

*Rapport genere par 6 agents experts Opus 4.6, audit de 71 fichiers Python, ~15,000 lignes de code.*
*Duree totale des audits: ~20 minutes (agents paralleles).*
