# PFC CH — Proposition de registre des événements structurels

État : proposition et premier relevé documentaire, 8 septembre 2026.
Aucun collecteur récurrent ni connecteur au modèle n'est activé.

## Projets vérifiés

Swissgrid recense 31 projets dans son Réseau stratégique 2040. Le programme
comprend des remplacements, renforcements et transformateurs, autant que de
nouvelles lignes. Un projet ne représente donc pas automatiquement un ajout
net de capacité. [Programme officiel](https://www.swissgrid.ch/fr/home/projects/future-grid/grid-development-requirements.html).

| Projet | Échéance publiée consultée | Intérêt pour la PFC |
|---|---|---|
| Bickigen–Chippis | 2031, procédure avec recours | Renforcement intérieur 220→380 kV ; accès à l'hydro et aux importations |
| Beznau–Tiengen (DE) | Programme 2040 ; année de mise en service non établie ici | Renforcement direct de la liaison CH–DE |
| Transformateurs déphaseurs en Suisse romande | Programme 2040 ; année précise non établie ici | Pilotage des flux et possibilités d'échange |
| Câble du Saint-Gothard | 2030 | Remplacement d'une ligne 220 kV ; gain de capacité commerciale non quantifié |

Sources : [Bickigen–Chippis](https://www.swissgrid.ch/fr/home/projects/project-overview/bickigen-chippis.html),
[liste des projets 2040](https://www.swissgrid.ch/fr/home/projects/future-grid/grid-development-requirements.html),
[Saint-Gothard](https://www.swissgrid.ch/fr/home/projects/project-overview/gotthard.html).
Ces pages ont été consultées aujourd'hui ; leurs versions historiques ne sont
pas reconstruites. Une échéance affichée est une prévision, pas une date ferme.

Les mises en service 2030/2031 se situent au-delà des livraisons 2026–2029 du
candidat actuel. Elles restent en veille. Des travaux, consignations ou mesures
temporaires pendant 2026–2029 peuvent en revanche concerner ce candidat, avec
une preuve et un intervalle d'effet propres. Ne jamais avancer une mise en
service pour qu'elle entre dans l'horizon du modèle.

## Mécanisme économique à tester

La chaîne utile est : ouvrage et disponibilité → capacité effectivement
offerte aux échanges → équilibre offre/demande par heure → prix CH et écarts
avec les voisins. La capacité physique nominale et la capacité commerciale
ne sont pas interchangeables ; la configuration du réseau et les règles
d'allocation interviennent aussi. [Analyse ACER](https://www.acer.europa.eu/monitoring/MMR/crosszonal_electricity_trade_capacities_2024).
Swissgrid sépare elle-même simulations de marché et de réseau dans sa
[méthode de planification](https://www.swissgrid.ch/fr/home/projects/future-grid/grid-development.html).

Les effets ci-dessous sont des hypothèses économiques conditionnelles, pas
des effets chiffrés établis pour les quatre ouvrages :

- Lorsque CH est plus chère et qu'une frontière limite l'importation, une
  capacité supplémentaire peut réduire la prime suisse et les pointes.
- Lorsque CH est moins chère, de nouvelles possibilités d'exportation peuvent
  relever son prix. Une meilleure interconnexion peut également transmettre
  des épisodes très bon marché ou négatifs venant des voisins.
- Hydro, météo, consommation et indisponibilités modifient le sens et
  l'intensité de ces effets selon l'heure et la saison.

Un soulagement de congestion intérieure peut améliorer la sécurité ou réduire
les coûts de redispatch sans modifier fortement le prix zonal CH. Les coûts
d'investissement et tarifs de réseau ne sont pas des suppléments mécaniques
à une PFC de prix de gros. Aucun coefficient universel EUR/MWh par ligne,
par kilomètre ou par kV ne doit être créé.

## Intégration dans le modèle existant

Le registre alimente d'abord des hypothèses de capacités et de disponibilité,
avec origine et validité explicites. Pour commencer, conserver une approche
CH + voisins avec capacités commerciales directionnelles et régimes de
congestion. Un modèle complet de flux électriques demanderait la topologie,
les paramètres et les contraintes de sécurité correspondants ; il n'est pas
nécessaire de le reconstruire pour tenir le premier registre.

Comparer, à météo/charge/parc identiques, quelques variantes fixées à l'avance :
réseau existant, projet au calendrier documenté, projet retardé. Une plage de
sensibilité n'est pas une distribution de probabilité. Les MW inconnus restent
inconnus ; une annonce descriptive seule ne déclenche aucun ajustement de prix.
Documenter aussi les projets déjà inclus dans une trajectoire de référence et
leurs dépendances, pour éviter de les ajouter deux fois.

Séparer explicitement les deux sorties :

1. **Niveau mensuel.** Le solveur et ses contraintes CH EEX restent seuls
   compétents. Une annonce peut déjà être reflétée dans les forwards. Aucun
   supplément événementiel ni réécriture des moyennes après le solveur. Un
   écart fondamental mensuel reste un diagnostic séparé ; toute évolution du
   solveur demanderait sa propre spécification et ses preuves.
2. **Forme horaire.** Une variante physique peut proposer une déformation
   additive, centrée avec les durées réelles dans chaque mois suisse, puis
   passée dans l'assembleur et la projection EEX existants. Vérifier à la fin
   les moyennes ET tous les contrats conservés. Le centrage seul ne garantit
   pas les contraintes PEAK ; la projection peut retirer une partie du signal.

D304 reste la référence et D305–D307 les comparaisons. Figer les candidats et
seuils avant résultats, sans sélection par mois. Examiner séparément les
années, saisons, horizons, prix négatifs/proches de zéro, rampes et pointes.
Un épisode historique unique ne suffit pas à identifier un effet causal :
contrôler météo, demande, autres capacités et indisponibilités ; utiliser un
holdout futur archivé. Les références et les exports existants restent intacts.

## Registre minimal et suivi proposés

Commencer par des tables versionnées et un journal immuable des observations,
puis utiliser Delta dans Databricks pour un suivi partagé lorsqu'il est utile.
Une base applicative distincte ou une base vectorielle n'est pas requise.

| Objet | Champs utiles |
|---|---|
| Événement | Identifiant stable, type, ouvrage, pays/frontière, direction, statut source et statut normalisé |
| Observation | URL/éditeur, date de publication si connue, première observation réelle UTC, dernière vérification, référence au document conservé et SHA-256, version remplacée |
| Effet physique | Date ou fenêtre de mise en service, période de travaux, unité, MW documentés ou inconnus, dépendances, actifs remplacés |
| Hypothèse modèle | Horizon, scénario, variable concernée, provenance de la conversion vers capacité commerciale, preuve de validation ; effet prix inconnu par défaut |

Conserver séparément le moment où l'information était connue et celui où
l'événement doit produire un effet. Un report crée une nouvelle observation ;
il n'efface pas l'ancienne échéance. Si la date exacte n'est pas connue,
conserver une précision annuelle ou une fenêtre sans inventer un 1er janvier.
Le relevé initial associé utilise seulement les pages consultées : il ne
possède pas encore les copies sources nécessaires à une admission historique.

Cadence proposée : revue hebdomadaire des projets, collecte quotidienne des
indisponibilités/capacités opérationnelles déjà structurées, revue périodique
de l'univers des projets, et alerte sur mise en service, retard, abandon ou
changement matériel de capacité. Signaler aussi une source inaccessible ou
non vérifiée depuis le délai fixé. Dédupliquer les annonces et conserver les
désaccords entre sources. Extraction assistée possible ; conversion physique
et admission du signal doivent être contrôlées. La veille ne réentraîne pas
automatiquement le modèle à chaque article.

Sources prioritaires : Swissgrid/OFEN/ESTI pour les projets CH, gestionnaires
voisins et TYNDP pour les liaisons, ENTSO-E/JAO pour les capacités disponibles
et indisponibilités. Étendre ensuite le même registre aux mises en service et
retraits de centrales, stockage, grandes charges et changements de marché.
Les prix de combustibles/CO2 et la météo restent des séries structurées ; les
articles ne les remplacent pas. Aucun scénario AFRY n'est utilisé.

Livrables présents : ce cadrage et quatre entrées dans
`CH-STRUCTURAL-EVENT-WATCHLIST-20260908.json`. État documentaire seulement :
toutes les autorités restent `false`, aucun modèle ni ordonnanceur modifié.
