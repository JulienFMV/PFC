# Databricks — demande spot, météo et Swissgrid

## Objectif

Qualifier les données disponibles avant tout export lourd, sans démarrer de
Warehouse pour cette demande et sans scan dont le coût n'est pas préalablement
plafonné. Les quatre profils ci-dessous sont préparés, mais **non autorisés en
l'état** : leur `LIMIT` borne le résultat retourné, pas les octets lus.

La Silver peut être lue pour diagnostiquer la source, mais la livraison du data
engineer s'arrête à des tables **Gold gouvernées**, complètes et documentées.
Après acceptation de ces Gold, **notre équipe** réalisera elle-même un snapshot
Parquet immuable sous
`C:\Users\jbattaglia\PFC_LT\build\databricks-exports\<snapshot_id>\`.
Le data engineer ne doit donc pas produire ni déposer de fichier sur notre
`C:`.

## Verdict déjà établi — ne pas redemander l'inventaire

- `prd.gold` ne contient pas de prix spot au grain horaire/15 minutes. Le seul
  fait spot observé est `dev.gold.factspotpricemonthly`, trop agrégé pour la
  forme de prix ; les points existent dans `dev.silver.ge_market_euler_spot`.
- Les trois faits météo existent dans `prd.gold`, mais les deux historiques de
  forecast n'exposent pas de date d'émission ni de clé de vintage explicite.
- `prd.gold.factswissgridbalancingquarterhourly` ne conserve que le
  déséquilibre et le prix one-price ; les composantes aFRR/mFRR/RR/NRV/FRCE et
  les prix long/short restent seulement en Silver.
- Aucun objet NTC, flux transfrontalier, échange programmé, redispatch ou outage
  dédié n'a été trouvé dans les 406 objets visibles de `prd`.
- `dev.gold` contient les trois tables génériques ENTSO-E. L'audit du 3 août y
  a observé les macro-familles NTC day/month/year-ahead, flux physiques et
  échanges programmés. Leur mapping exact par frontière, direction, produit et
  vintage reste à produire ; la famille NTC intraday n'est pas prouvée.

## Phase 0 — préflight de coût, sans lecture métier

Avant toute exécution des profils, retourner pour chacune des quatre tables :

- taille Delta totale en octets et nombre de fichiers ;
- colonnes de partition et possibilité réelle de pruning ;
- volume estimé lu par la requête ou plan estimé équivalent ;
- Warehouse qui serait utilisé, en confirmant qu'il est déjà actif pour une
  charge métier autorisée ;
- timeout et plafond de coût applicables au batch.

Cette phase ne doit ouvrir aucune ligne métier et ne doit pas démarrer un
Warehouse. Si le volume lu ne peut pas être estimé ou plafonné, arrêter ici et
proposer un export gouverné produit par la plateforme data sous sa propre
autorité de coût.

## Phase A — profils uniquement, après notre GO

Requêtes préparées dans le repo :

1. `docs/data/sql/databricks_dev_spot_profile.sql` ;
2. `docs/data/sql/databricks_prd_weather_profile.sql` ;
3. `docs/data/sql/databricks_prd_swissgrid_balancing_profile.sql` ;
4. `docs/data/sql/databricks_prd_swissgrid_tender_profile.sql`.

Livrer les quatre résultats sans modification :

- `spot_profile.parquet` ;
- `weather_profile.parquet` ;
- `swissgrid_balancing_profile.parquet` ;
- `swissgrid_tender_profile.parquet`.

Ajouter :

- `source_semantics.md` avec les confirmations demandées ci-dessous ;
- `cost_receipt.json` avec taille des tables, partitions, octets estimés/lus,
  Warehouse, durée, statut de démarrage initial/final et nombre de statements ;
- `manifest.json` indiquant pour chaque fichier le nom, le nombre de lignes, la
  taille, le SHA-256 et l'heure UTC de création.

## Confirmations nécessaires

### Spot — `dev.silver.ge_market_euler_spot`

- source primaire réelle derrière « Euler » : EPEX, autre fournisseur ou
  courbe interne ;
- droit/licence autorisant l'utilisation analytique et l'export local ;
- mapping des courbes vers `CH`, `DE_LU`, `FR`, `IT_NORTH` et `AT` ;
- sens exact de `quotation_date`, `quotation_ts` et `_ingest_ts` ;
- clé officielle d'une observation et politique de correction/révision.

Ne pas présenter `prd.gold.facteexpricedaily` comme du spot : cette table
contient des produits à terme EEX.

### Météo — tables `prd.silver.weather_*`

- dictionnaire des lieux : identifiant, nom, latitude, longitude, altitude,
  pays/zone et fuseau ;
- distinction mesure/forecast et source MeteoSwiss/Open-Meteo ;
- définition de `Date_Time_UTC`, `lead_time`, `Is_Historical` et
  `Meta_Load_Timestamp` ;
- règle officielle donnant la date d'émission du forecast. Si elle n'existe
  pas, l'indiquer explicitement : ne pas la reconstruire par supposition ;
- disponibilité des anciennes vintages de forecast et politique de correction.

### Swissgrid balancing — `prd.silver.ge_power_swissgrid_cab`

- unité et convention de signe de chaque colonne aFRR, mFRR, RR, NRV, FRCE et
  déséquilibre ;
- unité et sémantique des prix long, short et one-price ;
- granularité native et sens de `ts_start_utc` ;
- caractère indicatif ou final, calendrier de publication et corrections ;
- source/document Swissgrid et identifiant de fichier/version.

### Swissgrid tenders — `prd.silver.ge_market_swissgrid_sdl_tenders`

- mapping de chaque produit vers FCR, aFRR, mFRR, RR ou autre SDL ;
- unités des volumes et prix de capacité ;
- grain officiel d'une offre et clé de dédoublonnage ;
- sens de `source_snapshot_ts` et politique de remplacement des snapshots.

### NTC et autres données transfrontalières — décision de source

Ne pas répondre par un nouvel inventaire `présent / absent`. Procéder ainsi :

1. mapper les séries déjà présentes dans `dev.gold.dimentsoeseries` vers
   `NTC_MONTH_AHEAD`, `NTC_DAY_AHEAD_D2`, `NTC_INTRADAY`,
   `SCHEDULED_EXCHANGE` ou `PHYSICAL_FLOW` ;
2. démontrer les deux directions de CH-DE, CH-AT, CH-FR et CH-IT, ou déclarer
   précisément chaque couple absent ;
3. pour les séries absentes, charger la publication officielle Swissgrid ou
   ENTSO-E en conservant document, version et timestamp de publication ;
4. réconcilier les doublons de source sans les fusionner silencieusement.

Le PDF Swissgrid `NTC-202609.pdf` est un forecast **mensuel indicatif** publié
le 28 juillet 2026, version 1. Il couvre seulement CH-DE et CH-AT dans les deux
directions, au grain horaire. Il est utile comme vintage month-ahead mais ne
remplace ni le D-2/day-ahead ni l'intraday. Swissgrid publie ces deux dernières
familles séparément et indique que les NTC CH-IT/IT-CH sont publiées sur la
Transparency Platform ENTSO-E.

Endpoints de référence :

- `https://www.swissgrid.ch/en/home/customers/topics/congestion-mgmt/ntc/monthly-ntc.html` ;
- `https://www.swissgrid.ch/en/home/customers/topics/congestion-mgmt/ntc/d-2-ntc.html` ;
- `https://www.swissgrid.ch/en/home/customers/topics/congestion-mgmt/ntc/intraday-ntc.html` ;
- `https://www.swissgrid.ch/en/home/operation/grid-data/cross-border-load-flows.html` ;
- ENTSO-E Transparency Platform pour les séries normalisées et CH-IT/IT-CH.

Champs Gold minimaux pour toute capacité transfrontalière : `source_system`,
`metric_type`, `timeframe`, `from_zone`, `to_zone`, `unit`,
`native_resolution`, `target_start_utc`, `target_end_utc`, `value`,
`published_at_utc`, `first_observed_at_utc`, `source_document_id`,
`source_revision` et `ingested_at_utc`.

## Format Gold attendu avant l'export local

Le data engineer doit publier ou enrichir les surfaces suivantes :

1. Spot : `prd.gold.dimspotproduct` et un nouveau
   `prd.gold.factspotpriceinterval`, au grain produit × intervalle de livraison,
   sans moyenne mensuelle comme substitut.
2. Météo : conserver `prd.gold.factweather`,
   `prd.gold.factweatherforecasthistms` et
   `prd.gold.factweatherforecasthistom`, mais ajouter une origine de forecast et
   des vintages gouvernées ; `Meta_Load_Timestamp` seul ne suffit pas.
3. Swissgrid : enrichir `prd.gold.factswissgridbalancingquarterhourly` avec les
   composantes et la traçabilité aujourd'hui perdues entre Silver et Gold ;
   conserver unités, snapshots et source dans
   `prd.gold.factswissgridtenderofferresult`.
4. ENTSO-E : promouvoir et enrichir en `prd.gold` les trois surfaces
   `dimentsoeseries`, `factentsoetimeserieslatest` et
   `factentsoetimeseriesvintages`, ainsi que les historiques gouvernés de
   résolution des séries et de zones/codes EIC, conformément au contrat
   ENTSO-E détaillé.

La Gold est le contrat de consommation. La Silver reste une preuve de
réconciliation et ne doit pas être le dataset final utilisé par la PFC.

## Règles du batch

Les règles de lecture seule ci-dessous s'appliquent à nos profils de
diagnostic de Phase A. Elles n'interdisent pas au data engineer de créer ou
d'enrichir les tables Gold demandées en Phase B dans son processus plateforme
gouverné.

- lecture seule ; aucune création ou modification de table/vue ;
- aucun démarrage de Warehouse pour cette demande ;
- aucune requête de Phase A sans notre GO écrit après lecture du préflight ;
- au maximum quatre statements de profilage, une exécution chacun, sans retry
  automatique ;
- aucune interpolation, imputation, resampling, `forward-fill`, dédoublonnage
  ou remplacement des `NULL` par zéro ;
- conserver les valeurs négatives et les timestamps source ;
- ne pas mélanger `dev` et `prd` dans une même série ;
- ne pas lancer de backfill ni de dump des faits avant notre GO après lecture
  des profils ;
- si une requête dépasse le budget opérationnel habituel, l'arrêter et le
  signaler plutôt que l'élargir.

Les résultats de Phase A prouvent uniquement la qualité descriptive des tables.
Ils ne donnent aucune autorité PIT, modèle, sélection, promotion ou production.

## Phase B — publication Gold, puis export local par notre équipe

Après validation des profils, le data engineer publiera les Gold demandées avec
les courbes, lieux, produits, périodes et métadonnées convenus. Notre équipe
effectuera ensuite une lecture sélective et produira elle-même un snapshot
local immuable, idéalement depuis 2019 lorsque l'historique officiel le permet,
avec les timestamps d'émission/réception et les vintages disponibles. Après
contrôle du manifeste et des hash, tous les profils, joins, features et
backtests seront exécutés sur `C:` ; Databricks ne sera plus rescanné pour les
itérations de modélisation.

Les Gold doivent permettre à notre exporteur de produire les preuves ENTSO-E
prévues dans le contrat détaillé : historique des résolutions, historique des
zones/EIC, qualité par série, inventaire des familles, rapport de trous,
réconciliation source/livraison et exclusions. Ces informations ne sont pas
optionnelles : elles permettent de contrôler la couverture CH, les quatre
frontières, les deux directions et les vintages avant toute utilisation par la
PFC.
