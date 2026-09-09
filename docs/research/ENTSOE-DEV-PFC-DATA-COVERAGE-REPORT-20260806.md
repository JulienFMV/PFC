# Couverture ENTSO-E `dev` pour la PFC suisse

État au 6 août 2026 — audit hors ligne D252, sans requête Databricks.

## Résumé technique

**Non, nous ne pouvons pas encore affirmer que tous les types de données
ENTSO-E nécessaires sont présents dans `dev.gold`.** Les trois tables attendues
existent et l'audit du 3 août a observé 11 macro-signaux utiles. En revanche,
l'inventaire exact et courant de `GroupName` / `FieldName` n'a pas été capturé :
la tentative du 6 août s'est arrêtée avant SQL parce que le Warehouse était
arrêté.

Le point le plus important ajouté par D252 est la couverture **des zones
couplées**. Pour une PFC CH supérieure à une courbe purement calendaire, il ne
suffit pas de vérifier les fondamentaux suisses et les quatre frontières. Il
faut aussi qualifier prix, charge, forecasts et production de `DE_LU`, `FR`,
`IT_NORTH` et `AT`, avec une correspondance EIC/versionnée approuvée.

La documentation ENTSO-E récente fait aussi apparaître plusieurs candidats que
notre première checklist ne rendait pas assez explicites : prix intraday,
batteries/stockage, FCR, impact des indisponibilités réseau sur les positions
nettes, paramètres flow-based long terme, prévisions d'adéquation court terme,
indisponibilités de grandes unités de consommation et historique des zones/EIC.
Seul ce dernier élément est un prérequis de métadonnées ; les autres doivent
rester des features candidates, soumises à un rolling-origin indépendant.

## Ce qui est réellement confirmé dans `dev`

La preuve disponible confirme la structure réelle suivante :

- `dev.gold.dimentsoeseries` : 11 colonnes ;
- `dev.gold.factentsoetimeserieslatest` : 8 colonnes ;
- `dev.gold.factentsoetimeseriesvintages` : 10 colonnes.

L'audit du 3 août a observé les macro-signaux suivants. « Observé » ne signifie
pas que toutes les zones, technologies, directions, périodes et vintages sont
complètes.

| Macro-signal | État actuel | Ce qui reste à prouver |
|---|---|---|
| Charge réelle | Observé | zones CH et voisines, cadence, trous, fraîcheur |
| Prix day-ahead | Observé | toutes les zones couplées et métadonnée canonique `EUR/MWh` |
| Production réelle | Observé | solaire, éolien, nucléaire, fil de l'eau, réservoir, pompage par zone |
| Forecast de production | Observé | technologies, horizons et timestamp de disponibilité |
| Forecasts solaire/éolien | Observés mais non séparés | day-ahead versus intraday, zones et PIT |
| Stockage des réservoirs | Observé | CH/AT/FR/IT, unité, trous et calendrier de publication |
| Flux physiques | Observés | CH-DE/FR/IT/AT, directions et signe positif |
| Échanges programmés | Observés | quatre frontières, directions et signe |
| NTC day/month/year-ahead | Observés | quatre frontières, directions, révisions et disponibilité PIT |
| Forecast de charge | Non prouvé comme famille distincte | inventaire exact requis |

## La baseline exige encore des preuves exactes

Le premier export admissible doit contenir les 13 groupes du contrat : charge
réelle et forecast, prix day-ahead, production réelle et forecast, stockage
hydro, forecasts renouvelables day-ahead et intraday, flux physiques, échanges
programmés et NTC day/month/year-ahead.

Il doit également démontrer :

- six technologies de production : solaire, éolien, nucléaire, fil de l'eau,
  réservoir et pompage ;
- CH-DE, CH-FR, CH-IT et CH-AT avec directions et signes documentés ;
- cadence native par série et par régime historique, sans faux upsampling ;
- prix explicitement en `EUR/MWh` ;
- identifiant immuable du document, endpoint, qualité et révision source ;
- `PublicationTimestampUtc`, `PullTimestampUtc` et `Meta_Load_Timestamp`
  séparés, puis une règle `as_of_utc` approuvée ;
- début d'intervalle calculé depuis la borne droite `DateTimeUtc` en retirant la
  cadence native ;
- historique de configuration des zones et codes EIC.

Une série suisse historiquement horaire n'est pas un défaut. La grille attendue
est celle de sa cadence native et du régime de marché applicable.

## Couverture régionale à demander explicitement

| Type | Zones logiques | Priorité | Utilité PFC |
|---|---|---|---|
| Prix day-ahead | CH, DE_LU, FR, IT_NORTH, AT | Baseline | prix couplés et spreads horaires |
| Charge réelle + forecast | mêmes cinq zones | Haute | charge résiduelle et régimes import/export |
| Production réelle + forecast | mêmes cinq zones | Haute | nucléaire FR, renouvelables DE, hydro alpine |
| Forecasts renouvelables DA/ID | mêmes cinq zones | Haute | erreurs de forecast, rampes et rareté |
| Stockage hydro | CH, AT, FR, IT_NORTH | Haute | valeur d'eau et flexibilité alpine |

Les labels ci-dessus sont des zones logiques du modèle. Le data engineer doit
fournir la correspondance exacte avec les EIC ENTSO-E et ses fenêtres de
validité ; aucune correspondance ne doit être déduite d'un simple nom.

## Données additionnelles à forte valeur

### Priorité haute déjà identifiée

- indisponibilités de production et de réseau, planifiées et fortuites ;
- capacité installée et disponible par technologie ;
- énergie d'équilibrage activée : prix et volumes aFRR/mFRR/RR, hausse/baisse ;
- prix de déséquilibre et déséquilibre système ;
- capacité de réserve achetée et prix de capacité ;
- redispatch, countertrading et congestion ;
- positions nettes ;
- évolution complète de la capacité cross-zonale intraday, pas seulement le
  dernier snapshot.

### Ajouts issus de la revue ENTSO-E

| Famille | Priorité | Pourquoi elle peut améliorer la PFC |
|---|---|---|
| Prix intraday | Haute | mesure le repricing des erreurs de forecast et de la rareté intraday |
| Production/charge et capacité des batteries | Haute | explique creux solaires et rampes du soir croissants |
| FCR : capacité achetée et prix | Haute | complète explicitement aFRR, mFRR et RR |
| Impact des outages réseau sur les positions nettes | Haute | plus informatif qu'un NTC révisé isolé |
| Paramètres flow-based long terme | Haute | décrit des contraintes régionales qui affectent les voisins de la Suisse |
| Prévisions d'adéquation court terme | Exploratoire | signal direct de régime de rareté lorsqu'il est horodaté PIT |
| Indisponibilités de grandes unités de consommation | Exploratoire | explique certains chocs de charge résiduelle |
| Historique zones/EIC | Métadonnée requise | évite les jointures historiques silencieusement fausses |
| Demandes élastiques et sélection de produits balancing | Exploratoire | affine les régimes de rareté de réserve |

La révision v3.4 du Manual of Procedures mentionne notamment les prix intraday,
le type de production « Energy Storage » et de nouveaux paramètres flow-based.
Les révisions v3.2/v3.3 détaillent l'évolution de capacité intraday et la
séparation des produits/directions de balancing. La documentation v3.5 ajoute
les paramètres flow-based long terme, FCR et des modes de publication des
indisponibilités réseau fondés sur capacité ou impact sur positions nettes.
La bibliothèque EDI documente aussi les interfaces de prévision d'adéquation et
de coordination des outages.

Sources officielles :

- [ENTSO-E — Manual of Procedures](https://www.entsoe.eu/data/transparency-platform/mop/)
- [ENTSO-E — EDI Library](https://www.entsoe.eu/publications/electronic-data-interchange-edi-library/)
- [ENTSO-E — consultation MoP v3.5](https://consultations.entsoe.eu/markets/amendments-to-the-manual-of-procedures-mop-v3r5-of/)

## Méthode et robustesse

L'audit sépare quatre niveaux : existence des tables, découverte de
macro-familles, mapping exact de la dimension et admission des valeurs/PIT. Les
deux premiers seulement ont une preuve partielle. D243 préparera un inventaire
borné de la petite dimension et D251 exigera un mapping propriétaire exact,
sans fuzzy matching. D250 garde les transformations temporelles et PIT fermées
tant que cadence et sémantique ne sont pas approuvées.

L'audit D252 est lié par hash aux états D247/D251 et aux contrats D243/D250. Son
roast refuse toute auto-déclaration d'inventaire complet, toute élévation
d'autorité modèle et toute disparition d'une zone couplée.

## Limites actuelles

- Aucun inventaire SQL courant n'a été ouvert le 6 août ; « non prouvé » ne
  signifie donc pas « absent ».
- Les 11 macro-signaux observés ne sont pas un dénominateur permettant de
  calculer un taux de couverture.
- Aucune valeur réelle n'a été ouverte dans D252 : couverture temporelle,
  fraîcheur, trous, distributions et PIT restent inconnus.
- Les familles additionnelles sont des candidates de shaping. Elles ne peuvent
  ni réécrire les moyennes mensuelles du solveur, ni être sélectionnées sans
  rolling-origin indépendant.

## Prochaine action recommandée

Sur un prochain jour autorisé, et uniquement si le Warehouse est déjà en cours
d'exécution, lancer une fois la requête D243 sur la seule dimension. Demander
ensuite au data engineer de mapper chaque signature exacte D251 et de fournir :

1. le groupe logique, la zone/EIC et sa période de validité ;
2. le produit/horizon/technologie/direction ;
3. l'unité et la cadence native ;
4. le signe, la qualité, la révision, le document et l'endpoint ;
5. le statut `présent`, `absent`, `ambigu` ou `non applicable` pour chaque ligne
   des matrices ci-dessus.

Jusqu'à cette capture, statut : **couverture macro partielle, inventaire exact
non prouvé, aucune autorité modèle ou production**.

## Coût et accès

D252 : zéro connexion Databricks, zéro SQL, zéro démarrage de Warehouse, zéro
écriture Databricks, zéro ligne ENTSO-E réelle ouverte et zéro accès `H:`. La
consultation de la documentation publique ENTSO-E n'a aucun effet sur le coût
Databricks.
