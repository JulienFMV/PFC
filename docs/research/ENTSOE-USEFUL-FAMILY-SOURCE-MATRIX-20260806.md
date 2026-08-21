# ENTSO-E — matrice des familles utiles et de leurs sources

État au 6 août 2026. Revue documentaire officielle, sans requête Databricks.

## Verdict

La plupart des fondamentaux nécessaires à la PFC sont bien des publications
prévues par la Transparency Platform ENTSO-E. Cela ne prouve toutefois pas
qu'elles sont présentes et complètes dans `dev.gold` pour la Suisse et ses
voisins.

Trois exceptions importantes :

- les prix intraday sont une publication volontaire sur ENTSO-E ; EPEX SPOT
  est la source de marché primaire ;
- les données suisses de balancing, redispatch et NTC doivent être contrôlées
  en priorité chez Swissgrid, même lorsqu'un standard ENTSO-E existe ;
- météo, apports hydro et combustibles/CO2 ne sont pas des familles ENTSO-E.

## Matrice de sourcing

| Famille | Utilité PFC | Sur ENTSO-E ? | Source à privilégier | Statut Databricks observé |
|---|---|---|---|---|
| Charge réelle et forecast day-ahead | Très haute | Oui, publication réglementaire | ENTSO-E TP | Réel observé ; forecast distinct non prouvé |
| Prix day-ahead | Très haute | Oui, publication réglementaire | EPEX/EEX pour le prix primaire ; ENTSO-E pour contrôle | Macro-famille observée, couverture zones à prouver |
| Production réelle par technologie | Très haute | Oui | ENTSO-E TP | Observée, détail technologique à prouver |
| Production programmée day-ahead | Haute | Oui | ENTSO-E TP | Forecast agrégé observé, sémantique exacte à prouver |
| Forecasts solaire/éolien DA, intraday et courant | Très haute | Oui | ENTSO-E TP | Observés sans séparation complète des horizons |
| Stockage des réservoirs hydro | Haute | Oui | ENTSO-E TP ; OFEN pour contrôle suisse | Observé, couverture et unité à prouver |
| Flux physiques et échanges programmés | Très haute | Oui | ENTSO-E TP ; Swissgrid pour les frontières CH | Observés, trous/directions à fermer |
| NTC day/month/year et capacité intraday | Très haute | Oui | Swissgrid pour CH ; ENTSO-E en contrôle | NTC observée ; évolution intraday non prouvée |
| Positions nettes | Haute | Oui | ENTSO-E TP ; Swissgrid pour les flux commerciaux CH | Non observées dans les preuves `dev` |
| Capacité installée et disponible | Haute | Oui | ENTSO-E TP ; OFEN pour contrôle d'actifs suisses | Non observée dans les preuves `dev` |
| Outages production et réseau | Très haute | Oui | ENTSO-E TP ; Swissgrid/REMIT pour contrôle CH | Non observés dans les preuves `dev` |
| Redispatch et countertrading | Haute | Oui | Swissgrid pour CH ; ENTSO-E TP en contrôle | Non observés dans les preuves `dev` |
| Énergie d'équilibrage aFRR/mFRR/RR, prix et volumes | Haute | Oui | Swissgrid pour CH ; ENTSO-E TP pour voisins | Schéma présent dans `prd.silver.ge_power_swissgrid_cab` ; contenu, unités, signes, révisions et PIT non prouvés |
| Déséquilibre système et prix d'imbalance | Haute | Oui | Swissgrid pour CH ; ENTSO-E TP pour voisins | Schémas présents dans `prd.silver.ge_power_swissgrid_cab` et `prd.gold.factswissgridbalancingquarterhourly` ; source indicative et contenu non qualifié |
| Capacité de réserve FCR/aFRR/mFRR/RR et prix | Haute | Oui | Swissgrid pour CH ; ENTSO-E TP pour voisins | Tenders Swissgrid présents dans `prd` avec volumes/prix/unités ; produits exacts, snapshots et publication non qualifiés |
| Prix intraday | Haute | Oui, mais volontaire | EPEX SPOT ; ENTSO-E seulement si publié | Non observés dans les preuves `dev` |
| Batteries / « Energy Storage » | Haute à moyen terme | Type ENTSO-E supporté depuis MoP v3.4 | ENTSO-E, puis registre OFEN/Swissgrid en contrôle | Non observées dans les preuves `dev` |
| Curtailment renouvelable | Exploratoire | Pas de série harmonisée directe identifiée | Événements Swissgrid/REMIT et données opérateurs | Non observé ; ne pas assimiler à forecast moins actual |
| Paramètres flow-based | Haute pour les contraintes voisines | Oui | JAO pour Core/Italy North ; ENTSO-E en contrôle | Non observés dans les preuves `dev` |
| Prévision d'adéquation court terme | Exploratoire | Outil ENTSO-E OPC/STA, accès public brut non prouvé | ENTSO-E/RCC ; Swissgrid/ElCom pour la Suisse | Non observée dans les preuves `dev` |
| Historique des zones et codes EIC | Bloquant de métadonnées | Oui | Registre EIC ENTSO-E et propriétaires de zone | Absent du schéma `dev` actuel |

La FCR doit rester séparée de l'énergie d'équilibrage : la documentation
ENTSO-E indique que la FCR a des offres de **capacité**, mais pas d'offres
d'énergie activée analogues à l'aFRR ou la mFRR.

## Sources officielles

- [ENTSO-E — Manual of Procedures](https://www.entsoe.eu/data/transparency-platform/mop/)
  et [Detailed Data Descriptions v3.4](https://eepublicdownloads.entsoe.eu/clean-documents/Transparency/MoP_Ref2_DDD_v3r4.pdf) : charge, génération,
  forecasts, capacités, outages, transmission, redispatch, balancing,
  stockage et nouvelles publications.
- [ENTSO-E — Transparency Platform](https://www.entsoe.eu/data/transparency-platform/) : périmètre réglementaire et historique disponible depuis le lancement de 2015, avec archives antérieures.
- [Swissgrid — énergie de réglage et bilan système](https://www.swissgrid.ch/en/home/operation/grid-data/control-energy-system-balance.html) : aFRR, mFRR, déséquilibre et prix suisses en téléchargement.
- [Swissgrid — flux transfrontaliers](https://www.swissgrid.ch/en/home/operation/grid-data/cross-border-load-flows.html) et [NTC intraday](https://www.swissgrid.ch/en/home/customers/topics/congestion-mgmt/ntc/intraday-ntc.html) : flux commerciaux et capacités aux quatre frontières suisses.
- [Swissgrid — redispatch](https://www.swissgrid.ch/en/home/customers/topics/redispatch.html) : événements détaillés suisses.
- [Swissgrid — Balancing Roadmap 2026–2030](https://www.swissgrid.ch/dam/jcr%3Aeaa2aa50-deb1-4579-92c6-dacb66429480/balancing-roadmap-en.pdf) : calendrier public des produits suisses à 15 minutes.
- [JAO — publication flow-based long terme](https://www.jao.eu/LTFBA-external-parallel-run) : résultats Core long terme ; les données Italy North sont également publiées par JAO.
- [ENTSO-E — OPC/STA](https://www.entsoe.eu/data/opcsta/) : outil paneuropéen de prévision d'adéquation à une semaine.
- [OFEN — statistiques électriques](https://www.bfe.admin.ch/bfe/en/home/supply/statistics-and-geodata/energy-statistics/electricity-statistics.html/) : contrôle suisse des bilans et réservoirs.
- [EPEX SPOT — produits 15 minutes](https://www.epexspot.com/en/new-15-minute-products-market-coupling) : calendrier et nature des produits spot ; les données de marché restent soumises aux conditions de licence EPEX.

## Données utiles hors ENTSO-E

Pour une PFC réellement meilleure, les familles suivantes doivent être
sourcées ailleurs et évaluées en rolling-origin :

- météo observée et forecasts archivés : température, irradiation, vent,
  précipitations et neige — MeteoSwiss/ECMWF ;
- apports hydro, neige et niveaux hydrologiques — OFEN et OFEV ;
- gaz, charbon et CO2 — EEX/ICE selon les licences ;
- prix intraday exécutables et indices CH — EPEX SPOT ;
- données détaillées d'actifs et REMIT — opérateurs, Swissgrid et sources REMIT.

AFRY reste un benchmark descriptif et un candidat de forme. Il ne remplace ni
ces observations, ni leurs vintages, ni le calendrier suisse, et ne possède
aucune autorité sur le niveau mensuel du solveur.

## Audit de la réponse du data engineer

La réponse est positive mais incomplète sur quatre risques importants :

1. un backfill jusqu'en 2019 fournit un historique révisé, pas les vintages
   réellement disponibles aux anciennes dates ;
2. « compléter les trous » doit signifier recharger une valeur officielle,
   jamais interpoler ou recopier ;
3. la date de réception et chaque correction future doivent être conservées
   dès maintenant ; ce point n'est pas encore confirmé dans sa réponse ;
4. l'alignement doit conserver `DateTimeUtc` comme borne droite et produire
   séparément le début d'intervalle à partir de la cadence effective.

La date du `3 novembre 2026` n'a pas été retrouvée dans une annonce publique
officielle. La feuille de route Swissgrid 2026–2030 disponible au 6 août 2026
parle d'un lancement des produits 15 minutes dans les enchères explicites
day-ahead et intraday au troisième trimestre 2026, sans confirmer de jour
précis. Elle indique également que l'allocation continue utilise déjà des
produits 15 minutes aux frontières CH-DE et CH-AT, avec CH-IT prévu en 2027 et
CH-FR en 2029. En outre, plusieurs données physiques suisses ont leur propre
cadence. Il ne faut donc jamais appliquer une bascule globale à toutes les
séries. Une éventuelle communication non publique fixant le 3 novembre devra
être référencée et rattachée uniquement aux produits et séries qu'elle couvre.

Statut : inventaire documentaire terminé ; présence exacte dans `dev.gold`
toujours non prouvée tant que la petite dimension n'a pas été inventoriée.
