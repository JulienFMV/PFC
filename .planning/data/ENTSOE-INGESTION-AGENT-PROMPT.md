# Prompt — Pipeline d'ingestion ENTSO-E + scénarios d'électrification → Databricks

> À coller dans Codex / Claude. Tu es un **data engineer senior**. Tu construis un
> pipeline d'ingestion **idempotent, vintage-safe et auditable** pour alimenter le
> modèle de courbe forward long-terme (PFC LT) d'une desk d'énergie.

## 0. Source de vérité (à lire AVANT de coder)
Le contrat de données complet est dans **`ENTSOE-INGESTION-SPEC.html`** (fourni
séparément — ouvre-le). Il contient : les colonnes cibles, le mapping
appel-API ↔ `doc_type` ENTSO-E, les codes zones EIC, le schéma Databricks, et la
section G (scénarios d'électrification). **Ce prompt ne remplace pas la spec : il
te dit comment l'implémenter.** Si une info manque, suis la spec ; si la spec et
ce prompt se contredisent, **arrête-toi et signale-le** (ne devine pas).

## 0bis. Noms de colonnes — ne PAS inventer
Les **64 noms de colonnes exacts** de la priorité 1 sont la cible contractuelle.
Extrais-les des tableaux §A de `ENTSOE-INGESTION-SPEC.html` (ou du jeu de
référence si on te le fournit) et reproduis-les **au caractère près** (ex.
`scheduled_net_export_ch_de_mw`, `ntc_total_ch_it_mw`, `residual_load_de_mw`).
Toute colonne renommée doit être justifiée et listée dans le README.

## 1. Stack & hypothèses (confirme-les, adapte si besoin)
- **Plateforme** : Databricks (Unity Catalog), tables **Delta**, **PySpark** +
  pandas pour les appels API.
- **Source ENTSO-E** : API REST `https://web-api.tp.entsoe.eu/api`, via la lib
  Python **`entsoe-py`** (`EntsoePandasClient`). Token = secret Databricks
  (`dbutils.secrets.get(scope="entsoe", key="api_key")`), JAMAIS en clair.
- **Sources scénarios** (section G) : fichiers téléchargés (OFEN Perspectives
  2050+, ENTSO-E TYNDP 2024) + registres (Pronovo, Bundesnetzagentur MaStR).
  Ces sources sont des **fichiers/exports**, pas une API temps-réel → ingestion
  par fichier déposé dans un volume Unity Catalog, avec métadonnées.
- **Catalog/schema cibles** : à confirmer avec l'équipe (placeholder
  `energy.entso` et `energy.scenarios`). Ne crée rien hors de ces schémas.

## 2. Conventions NON négociables (valables partout)
- **Tout en UTC.** L'API renvoie l'heure locale de zone → convertir en UTC,
  gérer le **DST** (jours à 92/100 pas de 15 min). Index = début de pas.
- **Granularité 15 min.** Les séries natives horaires (souvent la charge CH) sont
  ré-échantillonnées 15 min par **forward-fill**. Documenter le resampling.
- **Vintage-safety.** Une donnée n'est utilisable à une date de valorisation `V`
  que si elle était connue avant : réalisé → `measurement_date ≤ V` ; scénario →
  `publication_date ≤ V`. Conséquence d'implémentation : **toute série de
  prévision porte une colonne `as_of_utc`** (heure de publication, ~J-1 18:00
  CET) distincte du timestamp cible. Les réalisés ont `as_of_utc = NULL`.
- **Idempotence.** Réexécuter le pipeline sur la même fenêtre ne crée pas de
  doublons : `MERGE` (upsert) sur les clés définies au §4, jamais d'`append` nu.
- **Provenance.** Chaque ligne porte `source`, `doc_type` (ENTSO-E),
  `ingested_at_utc`, et un `quality_flag` ∈ {official, actual, fallback,
  internal}. Aucune donnée sans provenance n'entre en production.

## 3. Périmètre — endpoints ENTSO-E (méthodes `entsoe-py` exactes)
Zones EIC (voir spec §E) : CH `10YCH-SWISSGRIDZ`, DE_LU `10Y1001A1001A82H`,
FR `10YFR-RTE------C`, AT `10YAT-APG------L`, IT-NORD `10Y1001A1001A73I`.
Profondeur historique cible : **2017 → aujourd'hui** (idéalement 2015).

### 3a. Priorité 1 — réalisés (méthodes confirmées dans le code de référence)
- `query_load(zone)` → `load_*_mw` (CH + DE/FR/IT/AT).
- `query_generation(zone)` → par type : Solar, Wind Onshore+Offshore, Nuclear,
  Hydro Run-of-river/poundage, Hydro Water Reservoir, Hydro Pumped Storage.
- `query_scheduled_exchanges(a,b)` → `scheduled_net_export_ch_<b>_mw` (CH↔b, 2 sens).
- `query_crossborder_flows(a,b)` → `flow_net_export_ch_<b>_mw`.
- `query_net_transfer_capacity_dayahead(a,b)` → `ntc_export/import/net/total_ch_<b>_mw`.
- `query_unavailability_of_generation_units("FR", …)` → outages nucléaires FR.
> Cible : les **64 colonnes** du jeu de référence (fenêtre actuelle 2021→2026,
> 15 min). Reproduis-les à l'identique (noms inclus).

### 3b. Priorité 2 — à ajouter (voir spec §B) — **avec `as_of_utc`**
`query_load_forecast`, `query_wind_and_solar_forecast`,
`query_installed_generation_capacity` (capacité par type/an = driver
électrification), `query_unavailability_of_generation_units`/`_production_units`
(toutes zones), `query_unavailability_transmission`,
`query_aggregate_water_reservoirs_and_hydro_storage`, NTC long-terme
(`query_offered_capacity` / yearly / monthly). Prix DA `query_day_ahead_prices`
= fallback seulement.

### 3c. Colonnes DÉRIVÉES — calculées en ETL, PAS via l'API (spec §C)
`cross_border_mw` (somme flux nets), `solar_regime` (tertiles mensuels),
`load_deviation`, `flow_deviation`, les `*_zscore` (z-score mensuel
`(x-mean_mois)/std_mois`), `ntc_balance_ch_<b>`. Implémente les formules telles
qu'écrites dans la spec.

### 3d. Section G — scénarios d'électrification (fichiers, pas API)
Construire `energy.scenarios.electrification_scenarios` selon le schéma spec §G4
(`track ∈ {scenario, actual}`, `publication_date`, `measurement_date`,
`scenario`, `scenario_edition`, `scenario_weight`, `country`, `delivery_year`,
`delivery_month`, `variable`, `value`, `unit`, `quality_flag`). **Ne jamais
mélanger réalisé et scénario** dans une même ligne. Inclure les variables
manquantes listées en §G3 (CO₂/gaz/charbon, électrolyse, peak_demand_gw, profils
horaires, hydro LT, ntc_ch_at_gw, etc.) quand la source les fournit.

## 4. Modèle de tables Delta (schéma long + vue wide)
Implémente exactement les 3 tables long de la spec §D + la table scénarios §G4 :
- `energy.entso.timeseries` — PK upsert **`(zone, variable, ts_utc, as_of_utc)`**,
  partition `(zone, variable, date(ts_utc))`.
- `energy.entso.exchanges` — PK **`(from_zone, to_zone, kind, ts_utc, as_of_utc)`**.
- `energy.entso.capacity` — PK **`(zone, fuel_type, year, as_of_utc)`**.
- `energy.scenarios.electrification_scenarios` — PK
  **`(country, scenario_edition, scenario, delivery_year, delivery_month, variable, track)`**,
  partition `(country, scenario_edition, delivery_year)`.
Puis une **vue wide** `energy.entso.v_features_15min` qui pivote
`(zone, variable) → colonnes` reproduisant les 64 colonnes nommées du jeu de
référence (c'est ce que le modèle consomme).

## 4bis. Estime le volume d'appels AVANT le backfill
Avant de lancer un backfill complet, calcule et affiche l'estimation :
`#zones × #variables × #années × (req/an)` vs le quota ~400 req/min, et le temps
attendu. Si > quelques heures ou proche du quota, propose un découpage
(par année/zone) et demande validation avant de tout lancer. Ne déclenche pas un
backfill massif sans cette estimation.

## 5. Robustesse & exploitation
- **Retry + backoff exponentiel** (base 5 s, ≥3 essais) ; l'API tombe souvent.
- **Chunking** : ≤ 1 an par requête de génération ; paralléliser par
  (zone, variable, année) sans dépasser **~400 req/min**.
- **Dégradation gracieuse** : une zone/frontière indisponible n'arrête pas le
  run ; logguer et continuer.
- **Backfill vs incrémental** : un mode `--backfill start end` (historique) et un
  mode incrémental quotidien (J-2 → J pour absorber les révisions ENTSO-E), tous
  deux idempotents via `MERGE`.
- **Job Databricks** : paramétrable (zones, fenêtre, mode), planifiable.

## 6. Qualité des données (gates avant publication)
Écris des checks qui **bloquent** la publication si :
- couverture < 98 % des pas attendus sur un mois donné (trous) ;
- un timestamp de prévision a `as_of_utc ≥ ts_utc` (fuite) → rejet ;
- une `value` négative sur une variable de génération/charge (hors flux nets) ;
- doublon sur la PK ; `quality_flag` ou `source` manquant.
Émets un rapport de qualité (table `energy.entso._dq_report`) par run.

## 7. Livrables attendus
1. Code du pipeline (notebooks/Python modulaires + un `README`).
2. DDL Delta des 4 tables + la vue wide.
3. Job(s) Databricks (backfill + incrémental) avec paramètres.
4. Tests : (a) unitaires sur le parsing/UTC/DST et le calcul des colonnes
   dérivées ; (b) un test d'idempotence (2 runs → 0 doublon) ; (c) un test
   anti-fuite (`as_of < ts` garanti sur les forecasts).
5. Le rapport de qualité §6.

## 8. Définition de « terminé » (acceptance)
- [ ] Les **64 colonnes** priorité 1 reproduites (mêmes noms) sur ≥ 2017.
- [ ] Priorité 2 ingérée avec `as_of_utc` non nul sur tous les forecasts.
- [ ] Section G (scénarios) chargée avec `track`/`publication_date`/provenance.
- [ ] Tables Delta + vue wide créées ; partitions & PK conformes §4.
- [ ] Idempotence prouvée (re-run = 0 doublon) ; anti-fuite prouvé.
- [ ] DQ gates actifs ; rapport généré ; secrets via secret scope.
- [ ] README : comment lancer backfill/incrémental, et limites connues.

## 8bis. Réalité d'exécution (lis avant de commencer)
Tu n'as peut-être PAS d'accès live au workspace Databricks, au secret scope, ni à
l'API ENTSO-E (token, réseau). Dans ce cas :
- **Produis tout le code, le DDL, les jobs et les tests** prêts à l'emploi, et
  rends-les exécutables tels quels une fois les accès fournis.
- Remplace les appels live par une couche d'**abstraction testable** (un client
  ENTSO-E injectable) + des **fixtures** d'échantillons, pour que les tests
  unitaires passent **hors-ligne**.
- Indique précisément, dans le README, les variables/secrets/droits requis pour
  passer en réel. N'invente pas de credentials ; ne fais pas semblant d'avoir
  exécuté un run live que tu n'as pas pu lancer — dis ce que tu as réellement
  exécuté vs ce qui reste à lancer côté workspace.

## 9. Garde-fous
- Token UNIQUEMENT via secret scope ; ne jamais le logger ni le committer.
- Ne crée/écris rien hors des schémas convenus (§1).
- Pour toute ambiguïté de schéma, de zone, de droit d'accès ou de coût (volume
  d'appels), **STOP et demande** — ne devine pas dans un sens qui créerait une
  fuite, un doublon, ou un dépassement de quota.
