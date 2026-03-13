# Audit du commit b52f790 — PFC Pipeline Production-Ready

**Date d'audit** : 2026-03-13
**Commit** : `b52f790eaf1d1e3fe15a27b9ab8c08e8dc781298`
**Message** : "Prod-ready PFC pipeline + CH/DE calibration and dashboard upgrades"
**Scope** : 29 fichiers, +2018 / -330 lignes

---

## 1. Synthese globale

Ce commit constitue une etape majeure de mise en production du pipeline PFC 15min. Il ajoute :
- Le support multi-marche CH/DE avec lecture de rapports EEX XLSX
- Un systeme de contrats non-overlapping pour la calibration arbitrage-free
- Des quality gates en entree/sortie
- Un stockage analytique local (DuckDB)
- Un dashboard de monitoring ("Control Tower") et comparaison CH/DE
- L'outillage operationnel (Teams, SSL, Task Scheduler)

**Verdict global** : Architecture solide, modelisation coherente avec les standards marche power. Plusieurs points d'attention identifies ci-dessous.

---

## 2. Modelisation — Points forts

### 2.1 Construction non-overlapping des contrats de calibration (EXCELLENT)

**Fichier** : `pfc_shaping/model/assembler.py` — `_build_non_overlapping_contracts()`

L'ancienne approche passait tous les contrats (Cal + Quarter + Month) simultanement au calibrateur, ce qui pouvait creer un systeme sur-contraint ou rang-deficient (un mois etant contraint a la fois par son prix mensuel, son prix trimestriel ET son prix annuel).

La nouvelle approche decompose en **contrats mensuels non-overlapping** avec priorite `Month > Quarter > Calendar`. C'est la bonne pratique : chaque mois n'a qu'une seule contrainte de prix, issue du niveau le plus granulaire disponible. Cela garantit un systeme bien pose pour le Maximum Smoothness Spline.

### 2.2 Fallback calibration vers P_raw (BIEN)

**Fichier** : `pfc_shaping/model/assembler.py` — parametre `calibration_fallback_to_raw`

En production, si le calibrateur ne converge pas, il vaut mieux publier la courbe brute (P_raw) que de publier une courbe mal calibree. Ce fallback est configurable et active par defaut (`config.yaml: calibration_fallback_to_raw: true`). Approche pragmatique et correcte pour un pipeline de production.

### 2.3 Cascading energy-conservation (CORRECT)

**Fichier** : `pfc_shaping/calibration/cascading.py`

La decomposition Cal -> Quarter -> Month respecte strictement la contrainte de conservation d'energie :

```
F_parent = sum(F_child_i * h_i) / sum(h_i)
```

Les heures sont comptees correctement avec gestion DST (CET/CEST) et jours feries suisses (canton VS). La verification `_verify_conservation()` avec tolerance 0.001 EUR/MWh est appropriee.

### 2.4 Shape horaire — fallback robuste (AMELIORATION)

**Fichier** : `pfc_shaping/model/shape_hourly.py`

La strategie de fallback a ete considerablement amelioree :
1. Fallback intra-saison (meme saison, autre type de jour)
2. Fallback inter-saison (meme type de jour, autre saison)
3. Fallback global (profil moyen normalise)
4. Fallback NaN residuel dans `apply()` avec warning

Cela empeche le pipeline de crasher sur des cellules vides du tableau (saison x type_jour), ce qui est critique en production avec des historiques partiels.

### 2.5 Correction `prov` -> `subdiv` pour holidays (BUGFIX)

**Fichier** : `pfc_shaping/calibration/cascading.py`

Le parametre `prov` est deprecie dans la lib `holidays` au profit de `subdiv`. Ce fix est necessaire pour la compatibilite avec holidays >= 0.40.

---

## 3. Modelisation — Points d'attention

### 3.1 CRITIQUE — Prix du contrat source applique tel quel a chaque mois decompose

**Fichier** : `pfc_shaping/model/assembler.py` — `_build_non_overlapping_contracts()`, ligne ~310

```python
contracts.append(
    futures_contract_cls(
        name=f"{year}-{month:02d}<{source_key}>",
        price=float(base_prices[source_key]),  # <-- PROBLEME
        start=start_utc,
        end=end_utc,
        product_type="Base",
    )
)
```

Quand un mois est couvert par un prix **trimestriel** ou **annuel** (pas de prix mensuel disponible), le prix du contrat parent est applique directement comme contrainte du mois individuel. Cela signifie que **chaque mois d'un trimestre sera contraint a etre egal au prix trimestriel moyen**, ignorant completement la saisonnalite intra-trimestrielle.

**Exemple concret** : Si Q1-2027 = 80 EUR/MWh, les 3 mois janvier, fevrier, mars seront chacun contraints a 80 EUR/MWh, alors que l'hiver suisse a typiquement janvier > fevrier > mars.

**Recommandation** : Apres le cascading, `base_prices` contient deja les prix mensuels derives. Il faudrait s'assurer que `_build_non_overlapping_contracts()` est appele **apres** le cascading et utilise les prix mensuels generes. Actuellement dans `build()`, le cascading est fait d'abord (etape 0), donc `base_prices` devrait contenir les monthly, **mais** `_apply_calibration()` recoit le `base_prices` deja cascade. A verifier que le flow est bien `cascade -> build_non_overlapping -> calibrate` et non l'inverse.

**Verification** : Dans `build()`, le cascading est fait a la ligne "0. Cascading des forwards manquants", et le `base_prices` est mis a jour. Ensuite `_apply_calibration(price_raw, idx, base_prices)` recoit le base_prices cascade. **Donc le flow est correct** — mais seulement si le cascader est present. Si `self.cascader is None`, les contrats Cal/Quarter seront passes directement avec le probleme decrit. Ce scenario est possible si on instancie `PFCAssembler` sans cascader.

**Severite** : Moyenne (le pipeline prod passe toujours un cascader).

### 3.2 ATTENTION — `_resolve_base()` itere timestamp par timestamp

**Fichier** : `pfc_shaping/model/assembler.py` — `_resolve_base()`

```python
for i, ts in enumerate(idx_zurich):
    key_m = ts.strftime("%Y-%m")
    ...
    B.iloc[i] = base_prices[key_m]
```

Sur un horizon 3 ans en 15min, `idx` contient ~105 000 timestamps. L'iteration Python est O(n) avec n=105k, ce qui est lent. Idem pour `_compute_f_S()`.

**Recommandation** : Vectoriser avec `idx_zurich.strftime("%Y-%m")` et `pd.Series.map()`.

### 3.3 ATTENTION — Verification de coherence energetique partielle

**Fichier** : `pfc_shaping/model/assembler.py` — `_check_energy_consistency()`

La verification ne porte que sur les annees completes (>95% des intervalles). C'est correct pour eviter les faux positifs, mais les trimestres et mois ne sont pas verifies. Apres calibration no-arbitrage, l'erreur devrait etre <0.5% par contrat. Sans verification trimestrielle/mensuelle, un bug de calibration intra-annuel pourrait passer inapercu.

### 3.4 INFO — Seuils de confiance hardcodes

**Fichier** : `pfc_shaping/model/assembler.py` — `_confidence_score()`

Les seuils 0.85/0.65/0.45 sont arbitraires et non configures dans `config.yaml`. Pour un pipeline de production, ces parametres devraient etre externalises ou au minimum documentes dans leur justification (historique de backtest, etc.).

---

## 4. Pipeline — Points forts

### 4.1 Quality gates input/output (EXCELLENT)

**Fichier** : `pfc_shaping/pipeline/quality_gate.py`

Verifications solides :
- NaN dans `price_shape` -> hard fail
- Prix > 10 000 EUR/MWh -> hard fail
- `p10 > p90` inversion bande -> hard fail
- Fraicheur des donnees (staleness) -> warning
- Index monotone croissant -> hard fail

Les seuils sont raisonnables pour le marche power suisse/allemand.

### 4.2 Multi-source data ingestion avec fallback en cascade

**Fichier** : `pfc_shaping/pipeline/rolling_update.py`

La chaine `energy-charts -> SMARD -> ENTSO-E -> cache local` est bien pensee et resiliente. Chaque source a son try/except avec fallback au niveau suivant.

### 4.3 Run lock (CORRECT)

Le mecanisme de lock fichier empeche les executions concurrentes. Simple et efficace pour un pipeline single-instance.

### 4.4 Benchmark gate configurable

Le benchmark PFC vs HFC avec seuils MAE/RMSE/bias configurables est une bonne pratique pour detecter les regressions. Le mode strict (`fail_on_benchmark: true`) permet de bloquer la publication si la qualite se degrade.

---

## 5. Pipeline — Points d'attention

### 5.1 CRITIQUE — Indentation incorrecte dans `run_update()` : code hors du `with _run_lock`

**Fichier** : `pfc_shaping/pipeline/rolling_update.py`

```python
try:
    with _run_lock("rolling_update"):
        paths = config["paths"]
        params = config["model"]
        db_cfg = config.get("databricks", {})
        entsoe_cfg = config.get("entsoe", {})
    forwards_cfg = config.get("forwards", {})    # <-- HORS DU WITH!
    quality_cfg = config.get("quality", {})       # <-- HORS DU WITH!
    markets_cfg = forwards_cfg.get("eex_markets")
    ...
```

Tout le code apres la ligne `entsoe_cfg = ...` est **en dehors du bloc `with _run_lock()`**. Le lock est relache immediatement apres avoir lu 4 variables de config. Tout le pipeline (ingestion, calibration, assemblage, export) tourne **sans protection du lock**.

**Consequence** : Deux instances peuvent tourner en parallele, ecrire des fichiers simultanement, et corrompre les exports.

**Recommandation** : Indenter tout le corps du pipeline sous le `with _run_lock`.

**Severite** : Critique.

### 5.2 ATTENTION — `datetime.utcnow()` deprecie

**Fichier** : `pfc_shaping/pipeline/rolling_update.py`

`datetime.utcnow()` est deprecie depuis Python 3.12. Utiliser `datetime.now(timezone.utc)` a la place.

### 5.3 ATTENTION — Pas de retry sur l'ingestion EEX XLSX

Si le fichier EEX XLSX est sur un partage reseau (`H:\...`) et que le lecteur n'est pas monte, le pipeline echoue pour le marche DE sans possibilite de fallback. Le fallback Databricks n'est disponible que pour CH.

### 5.4 INFO — Variable `teams_webhook` lue mais potentiellement None

L'appel `send_teams_alert(teams_webhook, ...)` dans le bloc `except` est safe (la fonction teste `if not webhook_url: return False`), mais le warning-on-failure est silencieux. En production, si Teams ne fonctionne pas, il faudrait un second canal d'alerte.

---

## 6. Ingestion forwards EEX — Points d'attention

### 6.1 ATTENTION — Parsing XLSX fragile

**Fichier** : `pfc_shaping/data/ingest_forwards.py` — `load_base_prices_from_eex_report()`

Le parsing repose sur des conventions positionnelles :
- Row 0 = product codes
- Row 3+ = prix avec date en colonne A
- Mapping `local_col -> product_idx` avec `int(local_col) - 1`

Ce mapping est fragile. Si le format du rapport EEX change (colonnes ajoutees, header modifie), le parsing echouera silencieusement ou produira des mappings incorrects.

**Recommandation** : Ajouter une validation du header (verifier que les codes produits reconnus couvrent au moins Cal + quelques Quarters/Months) et un test unitaire avec un fichier XLSX de reference.

### 6.2 INFO — Regex `_EEX_BASE_PATTERN` ne supporte que BASE

Le pattern `^(Y01|Q\d{2}|M\d{2})_(\d{4})_BASE$` filtre uniquement les produits Base. Les produits Peak ne sont pas ingeres. C'est correct si le modele ne gere que du Base (ce qui est le cas actuellement), mais limiterait une future extension Peak/OffPeak.

---

## 7. Stockage DuckDB — Points d'attention

### 7.1 BIEN — Schema clair et benchmark historise

Les 3 tables (runs, forecasts_hourly, benchmarks) sont bien structurees. La cle primaire sur `run_id` et le DELETE before INSERT simulent un UPSERT correct.

### 7.2 ATTENTION — Pas d'index sur `ts_local` dans `forecasts_hourly`

Pour des requetes de type "forecast entre date A et date B", un index sur `ts_local` serait benefique avec un historique croissant.

### 7.3 ATTENTION — `benchmark_against_hfc` fait un `groupby(level=0).mean()` sur les doublons

En cas de transitions DST (2h du matin apparaissant 2 fois), la moyenne des prix n'est pas le bon traitement. Pour l'heure "repliee" CET, il faudrait garder les 2 heures distinctes.

---

## 8. Dashboard — Points d'attention

### 8.1 ATTENTION — Injection SQL dans `load_forecasts_hourly()`

**Fichier** : `dashboard/utils.py`

```python
rid = run_id.replace("'", "''")
sql = f"...WHERE run_id = '{rid}'..."
```

L'echappement par simple remplacement de `'` par `''` est fragile. Meme si DuckDB est local et le `run_id` est genere en interne, il est preferable d'utiliser des requetes parametrees.

### 8.2 BIEN — Page CH vs DE

**Fichier** : `dashboard/pages/7_ch_de_spread.py`

La comparaison CH/DE est bien implementee avec spread horaire et term structure mensuelle. Le dual-axis (prix + spread) est pertinent pour le desk trading.

### 8.3 BIEN — Control Tower

Le monitoring des runs, benchmarks et snapshots horaires est complet et actionnable.

---

## 9. Outillage operationnel

### 9.1 BIEN — Script Task Scheduler

Le `register_daily_task.ps1` a 06:15 est raisonnable (avant ouverture du marche, apres cloture des forwards J-1).

### 9.2 BIEN — Check SSL

L'outil de diagnostic SSL est utile dans un environnement corporate avec proxy/CA custom.

### 9.3 ATTENTION — Chemin Python hardcode

Les scripts PowerShell ont `C:\Users\jbattaglia\.conda\pfc311\python.exe` en parametre par defaut. A parametriser ou documenter pour d'autres postes.

---

## 10. Resume des findings

| # | Severite | Composant | Description |
|---|----------|-----------|-------------|
| 1 | **CRITIQUE** | rolling_update.py | Code pipeline hors du `with _run_lock()` — lock inefficace |
| 2 | **MOYENNE** | assembler.py | Prix parent applique directement si pas de cascader |
| 3 | **MOYENNE** | assembler.py | `_resolve_base()` et `_compute_f_S()` non vectorises (perf) |
| 4 | **MOYENNE** | ingest_forwards.py | Parsing XLSX positionnel fragile |
| 5 | **FAIBLE** | utils.py | Injection SQL potentielle dans DuckDB |
| 6 | **FAIBLE** | rolling_update.py | `datetime.utcnow()` deprecie |
| 7 | **FAIBLE** | local_duckdb.py | Pas d'index `ts_local`, gestion DST doublons |
| 8 | **INFO** | assembler.py | Seuils confiance hardcodes |
| 9 | **INFO** | assembler.py | Coherence energetique seulement annuelle |

---

## 11. Recommandations prioritaires

1. **Corriger l'indentation du `_run_lock`** dans `rolling_update.py` pour proteger tout le pipeline
2. **Ajouter un test unitaire** pour `load_base_prices_from_eex_report()` avec un fichier XLSX de reference
3. **Vectoriser** `_resolve_base()` et `_compute_f_S()` pour de meilleures performances
4. **Ajouter une verification de coherence trimestrielle** dans `_check_energy_consistency()`
5. Utiliser des **requetes DuckDB parametrees** dans le dashboard

---

## 12. Conclusion

Ce commit represente un travail substantiel et globalement bien execute pour la mise en production du pipeline PFC. L'architecture du modele (6 facteurs multiplicatifs + calibration no-arbitrage + confidence intervals) est coherente avec les standards du marche power europeen. La gestion multi-source (energy-charts / SMARD / ENTSO-E / Databricks) avec fallback en cascade est robuste.

Le point critique a corriger en priorite est le **lock de concurrence** dans `rolling_update.py`. Les autres findings sont des ameliorations de robustesse et de performance qui peuvent etre traitees iterativement.

La nouvelle fonctionnalite CH/DE avec spread tracking est un ajout pertinent pour le desk de FMV, permettant de suivre le differentiel Suisse/Allemagne qui est structurellement positif (hydro CH + imports, congestion transfrontaliere).
