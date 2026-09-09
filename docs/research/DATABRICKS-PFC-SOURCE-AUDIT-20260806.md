# Databricks — audit des sources candidates pour la PFC CH

Date : 6 août 2026  
Statut : modélisation suspendue ; audit des données prioritaire.

## Synthèse

| Famille | Environnement | Présence | Ce qui est prouvé | Ce qui reste à prouver |
|---|---|---|---|---|
| EEX forwards | `prd` | Oui | Historique local déjà capturé, produits BASE/PEAK DAY à YEAR, prix de règlement en EUR/MWh | Vintages signées et admission production |
| Spot | `dev` seulement | Oui, candidate | Prix granulaire, intervalles UTC, fréquence, courbe, quotation et unité dans `ge_market_euler_spot` | Source EPEX/licence, courbes CH/voisins, historique, trous et fraîcheur |
| Météo | `prd` | Oui | Mesures et forecasts MeteoSwiss/Open-Meteo, variables riches et historiques de forecast | Lieux/coordonnées, couverture nationale, origine des forecasts, vintages et PIT |
| Swissgrid balancing | `prd` | Oui | Quart d'heure : aFRR, mFRR, RR, FRCE, déséquilibre système et prix long/short/one-price | Historique, unités, signes, trous, révisions et caractère indicatif vs règlement |
| Swissgrid SDL tenders | `prd` | Oui | Offres/résultats, volumes, prix de capacité, pays, produit et snapshots annuels | Produits exacts FCR/aFRR/mFRR, unités, complétude et chronologie de publication |
| ENTSO-E | `dev` | Oui | Dimension, latest et vintages ; schéma et macro-familles partiellement inventoriés | Livraison gouvernée, familles exactes, zones, résolutions, trous et PIT |
| NTC/flux/échanges | `dev` ENTSO-E seulement | Partiel | Macro-familles NTC day/month/year-ahead, flux physiques et échanges programmés observées le 3 août | Mapping exact, intraday, 4 frontières × 2 directions, vintages et promotion `prd.gold` |
| Swissgrid NTC/redispatch | `prd` | Non observé | Aucun objet dédié trouvé dans les 406 tables/vues accessibles | Ingestion officielle Swissgrid/ENTSO-E et réconciliation Gold |

## Verdict d'admission

| Source | Verdict actuel | Risque principal | Sévérité |
|---|---|---|---|
| Spot Euler | `CANDIDAT_NON_ADMIS` | Identité EPEX/licence, zones et qualité des lignes non prouvées ; table uniquement dans `dev` | Haute |
| Météo mesurée | `CANDIDAT_NON_ADMIS` | Couverture spatiale et trous inconnus ; les observations ne sont disponibles qu'après leur cible | Haute |
| Météo forecast | `NO_GO_PIT` | Aucun timestamp d'émission/vintage explicite ; `lead_time` ne peut pas être transformé en origine sans règle propriétaire | Critique |
| Swissgrid balancing | `CANDIDAT_LAGGED_NON_ADMIS` | Données annoncées indicatives, unités/signes/révisions et publication finale inconnus | Haute |
| Swissgrid tenders | `CANDIDAT_NON_ADMIS` | Chronologie de publication et conservation des snapshots/vintages non prouvées | Haute |
| Swissgrid réactif/nodal | `EXCLU_BASELINE` | Grain actif/nodal et puissance réactive sans hypothèse économique wholesale démontrée | Moyenne |

Ces verdicts sont des conclusions de schéma et de gouvernance. Ils ne mesurent
ni complétude, ni profondeur historique, ni valeur prédictive : aucune ligne
métier n'a été ouverte.

## Verdict Gold et export local

L'objectif n'est pas de faire travailler la PFC directement sur les tables
Silver. Silver sert à vérifier la fidélité de la source ; le contrat de
consommation est Gold, suivi d'un unique export Parquet immuable sur `C:`.

Écarts Gold actuellement établis :

- aucun fait spot intervalle dans `prd.gold` ;
- météo Gold présente mais origine/vintages de forecast non gouvernées ;
- balancing Swissgrid Gold réduit au déséquilibre et au one-price ;
- aucune table ENTSO-E/NTC en `prd.gold` ; les trois tables ENTSO-E sont
  seulement dans `dev.gold`.

Le fichier utilisateur `NTC-202609.pdf` (SHA-256
`3b690c5f281321dd16609e2db168fc67f986d05432b864a82886dd6439b6de36`)
est un forecast mensuel Swissgrid de septembre 2026, publié le 28 juillet,
version 1. Il contient des valeurs horaires CH-DE et CH-AT dans les deux
directions, mais ni CH-FR ni CH-IT. Swissgrid qualifie ces valeurs mensuelles
d'indicatives et publie séparément le D-2 et l'intraday ; CH-IT/IT-CH est publié
sur ENTSO-E. Ce fichier est donc un vintage month-ahead utile, pas la vérité NTC
unique et pas un substitut aux autres timeframes.

Le collecteur local historique ne ferme pas ce trou :
`pfc_shaping.data.ingest_entso._load_swiss_border_features` appelle uniquement
`query_net_transfer_capacity_dayahead` pour les quatre frontières. Il ne
collecte ni le month-ahead versionné du PDF ni les snapshots NTC intraday. Il
reste legacy et ne peut pas remplacer la future Gold gouvernée.

Sources officielles à réconcilier :

- [Swissgrid — forecast NTC mensuel](https://www.swissgrid.ch/en/home/customers/topics/congestion-mgmt/ntc/monthly-ntc.html) ;
- [Swissgrid — NTC D-2](https://www.swissgrid.ch/en/home/customers/topics/congestion-mgmt/ntc/d-2-ntc.html) ;
- [Swissgrid — NTC intraday](https://www.swissgrid.ch/en/home/customers/topics/congestion-mgmt/ntc/intraday-ntc.html) ;
- [Swissgrid — NTC, schedules et flux transfrontaliers](https://www.swissgrid.ch/en/home/operation/grid-data/cross-border-load-flows.html) ;
- ENTSO-E Transparency Platform, notamment le document `A61` pour l'Estimated
  Net Transfer Capacity et les publications de capacité intraday.

## Swissgrid : lecture du schéma

### `prd.silver.ge_power_swissgrid_cab`

La table est la source Swissgrid la plus utile pour la forme de prix :

- `afrr_plus`, `afrr_moins` ;
- `sa_mfrr_plus`, `sa_mfrr_moins` ;
- `da_mfrr_plus`, `da_mfrr_moins` ;
- `rr_plus`, `rr_moins` ;
- `nrv_plus_import`, `nrv_moins_export` ;
- `frce_plus_import`, `frce_moins_export` ;
- `total_system_imbalance` ;
- `ae_price_long`, `ae_price_short`, `ae_price_annual` ;
- cible `ts_start_utc`, fichier source et timestamps d'ingestion.

Le commentaire du catalogue précise que ces données temps réel sont
**indicatives et non contraignantes** et que la facturation peut différer. Elles
peuvent donc devenir des variables de régime ou de forme après validation, mais
pas une vérité de règlement sans réconciliation avec la publication finale.

La table Gold `prd.gold.factswissgridbalancingquarterhourly` ne conserve que le
déséquilibre système et le prix one-price. Pour l'audit, la table Silver est
préférable car elle préserve les composantes et la traçabilité fichier.

### `prd.silver.ge_market_swissgrid_sdl_tenders`

La table conserve le code et la description de l'appel d'offres, les volumes
offerts/attribués, le prix de puissance, les unités, le pays, le statut attribué,
le snapshot source et des contrôles DQ. Elle est adaptée à l'inventaire des
marchés de capacité de réserve, sous réserve de confirmer les produits exacts
dans les lignes.

### Tables Swissgrid hors baseline PFC

`factreactiveswissgrid` et les tables `res_reactivepower_swissgrid_*` décrivent
la tension, la puissance réactive, la conformité et la facturation par nœud.
Elles sont pertinentes pour l'exploitation d'actifs/réseau, mais pas comme
fondamentaux de première ligne de la PFC wholesale CH. Elles restent exclues du
baseline tant qu'une hypothèse économique et un gain hors échantillon ne sont
pas démontrés.

## Blocages avant reprise de la modélisation

1. Spot : confirmer la source/licence et profiler courbes, zones, fréquence,
   couverture, doublons, trous et fraîcheur.
2. Météo : obtenir lieux/coordonnées et `forecast_issued_at_utc` gouverné ; le
   simple `lead_time` ne suffit pas à prouver le PIT.
3. Swissgrid balancing : confirmer unités, conventions de signe, résolution,
   publication finale/révisions et profondeur historique.
4. Swissgrid tenders : inventorier les produits et réconcilier les unités et
   snapshots annuels.
5. ENTSO-E : recevoir l'export gouverné demandé au data engineer, incluant
   vintages, résolutions, EIC/zones, gaps et réconciliation source.
6. NTC/flux/échanges : mapper ce qui existe déjà dans `dev.gold`, compléter
   l'intraday et les couples frontière-direction absents depuis Swissgrid ou
   ENTSO-E, puis publier une surface gouvernée en `prd.gold`.

Aucun entraînement, choix de feature, calibration AFRY ou comparaison de
candidat à OMPEX ne reprend avant fermeture de ces contrôles et gel d'un nouveau
holdout indépendant.

## Requêtes préparées, non exécutées

- `docs/data/sql/databricks_dev_spot_profile.sql` ;
- `docs/data/sql/databricks_prd_weather_profile.sql` ;
- `docs/data/sql/databricks_prd_swissgrid_balancing_profile.sql` ;
- `docs/data/sql/databricks_prd_swissgrid_tender_profile.sql`.

Elles sont `SELECT`-only, bornent leurs sorties et n'inventent ni unité, ni
origine de forecast, ni disponibilité PIT. Les contrôles lourds au grain seront
faits localement après rapatriement.

La demande data engineer correspondante est :
`.planning/phases/14-lt-audit-remediation/DATABRICKS-SPOT-WEATHER-SWISSGRID-DATA-ENGINEER-REQUEST-20260806.md`.
Elle impose une phase A limitée aux quatre profils et interdit tout export lourd
avant notre GO.

Ces fichiers ne constituent pas une autorisation d'exécution. Leur `LIMIT`
borne le résultat retourné, **pas les octets lus** par les agrégations. Avant
toute exécution, il faut donc confirmer localement la taille Delta, les colonnes
de partition et le plan estimé. Si le volume lu ne peut pas être plafonné ou si
le Warehouse n'est pas déjà actif pour une charge métier autorisée, la requête
reste interdite. Le profilage exhaustif sera effectué après un export local
borné, jamais par répétition de scans Databricks.

### Reçu Phase 0 contrôlé localement

D290 rend le préflight de coût vérifiable avant toute décision humaine. Le reçu
est content-addressé et lie exactement la demande enrichie ainsi que les quatre
versions SQL qualifiées par D289. Il conserve uniquement des métadonnées de
coût : taille et nombre de fichiers Delta, partitions, bornes d'octets lus,
méthode d'estimation, état déjà actif ou non du Warehouse, DBU et runtime
proposés.

Un reçu périmé, substitué, dupliqué, non plafonné, lié à une autre requête ou
déclarant une ligne métier, un retry, un démarrage ou une écriture échoue. Un
Warehouse arrêté produit `STOP_NO_ACTIVE_WAREHOUSE`; une estimation non bornée
produit `STOP_UNCAPPED_SCAN`. Même le meilleur verdict possible est seulement
`READY_FOR_HUMAN_COST_REVIEW_NO_EXECUTION_AUTHORITY` : il ne vaut jamais GO.

Le roast synthétique donne `35 passed`; la matrice adjacente
`63 passed, 1 skipped`. La preuve locale porte l'identifiant
`e3d3d7643bce35137cd8f9dd1c537e4d640021ba7fe86da59ed9f577a8f50cfc`.
Aucun identifiant de Warehouse, reçu réel ou valeur métier n'est conservé dans
la preuve.

## Preuves locales

- surface `dev`/`prd`, 940 objets visibles : SHA-256
  `d2532463f25ca29dd1b868ee300389fa9cc62a495395331b81ae780596ff9aa9` ;
- neuf schémas spot/météo : SHA-256
  `27a40f03373900229488ab699187265a40c7581773248057f2fdecc2f6b3be5f` ;
- seize schémas Swissgrid : SHA-256
  `d5c351ad5eae71ba6863105c9c00a2759039bc6bc8a281833729d9ffe78d47b0`.

Les trois captures déclarent ensemble 43 lectures du plan de contrôle, zéro
instruction SQL, zéro ligne métier ouverte, zéro démarrage de Warehouse et zéro
écriture. Une lecture séparée de l'état des Warehouses avait porté le reçu
global de la séquence à 44 appels de plan de contrôle.

## Coût et autorité

Les deux Warehouses visibles étaient arrêtés lors du dernier contrôle. Le
présent audit local n'a émis aucun nouvel appel : aucune instruction SQL, aucune
ligne métier, aucun démarrage de Warehouse et aucune écriture Databricks. La
présence et le schéma sont confirmés ; la qualité du contenu ne l'est pas encore.
