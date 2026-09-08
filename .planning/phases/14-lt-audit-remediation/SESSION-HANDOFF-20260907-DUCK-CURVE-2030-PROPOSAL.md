# Reprise après compact — duck curve 2030 et suivantes

Date : 7 septembre 2026. Statut : audit explicatif et proposition, sans fit,
scoring, modification de modèle, activation de scénario ni nouveau protocole.

## Demande et priorité proposée

L'utilisateur demande s'il faut compacter ou changer de session et comment
la future forme horaire intégrera solaire, BESS, smart grid et électrification,
en référence aux mécanismes examinés dans AFRY et à l'état de l'art août 2026.
Recommandation : `/compact`, puis poursuivre dans ce même workspace ; D300,
D301 et les propositions sont déjà documentés. L'utilisateur doit déclencher
la commande ; cette réponse n'affirme pas l'avoir exécutée.

La proposition précédente de benchmark v2 reste utile, mais elle ne suffit
pas à qualifier une duck curve 2030+. La priorité recommandée est de spécifier
la couche de trajectoires physiques et sa traduction en forme horaire, puis
de comparer les régressions/réseaux sur ce problème correctement défini.
Il ne s'agit pas de prétendre qu'un choix de scénario a été approuvé.

## Ce que fait effectivement le projet aujourd'hui

- D300 : `build/local-pfc-source-preflight-20260907/generate_curve.py`, lignes
  46–69, construit une climatologie ENTSO-E et une trajectoire de retour de
  l'écart hydro vers zéro. Il charge le MLP corrigé, l'intraday et water value,
  puis appelle le même `PFCAssembler` avec l'autorité mensuelle du solveur.
  Les options solaire et électrification ne sont pas activées.
- `pfc_shaping/pipeline/production_phases.py::_build_entsoe_climatology_forecast`
  reproduit des médianes par mois, heure et quart d'heure suisses. Ce n'est pas
  une projection de capacités, demande et flexibilité propres à 2030 ou 2035.
- `pfc_shaping/lt/model/shape_hourly_mlp.py::fit` apprend les ratios historiques
  avec calendrier/hydro et emplacements d'indisponibilités ; `years_ahead` est
  nul à l'entraînement. D300 n'apporte pas de trajectoire BESS/smart charging.
- `assembler.py::_shape_freedom` atténue des effets suivant l'échéance avec des
  coefficients configurés dans le code. Cela ne simule pas l'évolution du
  parc électrique. Le vocabulaire « structural base » n'établit pas un moteur
  d'équilibre énergétique futur.
- `solar_modulation.py` est expérimental, OFF par défaut ; sa projection repose
  sur climatologie/tendance historique bornée. Son commentaire d'audit signale
  une dégradation passée de sa métrique de creux solaire : aucune activation
  automatique recommandée.
- `electrification_shape.py::StructuralDriverProjector` expose déjà solaire,
  batteries puissance/énergie, véhicules, PAC, charge pilotée, hydro, thermique
  et interconnexions. Mais `ElectrificationFHCorrection.fit` est un no-op et
  `_adjustment` utilise des coefficients fixes par bloc horaire. Ce prototype
  n'optimise pas la charge des batteries ni l'équilibre des pays voisins.
  Ses contrôles d'inventaire ne constituent pas une validation économique.
- D301 compare la couche horaire historique, avec physique, intraday et water
  value neutralisés. Son gagnant global ne valide donc pas le shaping 2030+.

Ne pas modifier ces composants pendant une explication. Ne pas confondre
une grille générée jusqu'en 2032 avec une preuve sur son économie future.

## Construction cible proposée, à spécifier avant implémentation

1. Trajectoires versionnées CH/DE/FR/IT/AT par année : capacités et production
   renouvelables, demande par usage, puissance ET énergie BESS, rendement,
   contraintes de charge, part pilotable EV/PAC/industrie, parc pilotable,
   réservoirs/apports hydro, capacité d'échange et coûts combustibles/carbone.
   « Smart grid » doit devenir des paramètres mesurables : MW déplaçables,
   délais, disponibilité, limites réseau et participation. Les capacités
   installées peuvent être exogènes au premier lot ; pas besoin de construire
   immédiatement un modèle endogène d'investissement paneuropéen complet.
2. Plusieurs chronologies météo cohérentes entre pays et technologies,
   reconstruites sur le calendrier de livraison. Charge résiduelle puis
   fonctionnement intertemporel des batteries/hydro/demande flexible et
   échanges contraints. Une pile statique d'offre/demande ignore la valeur
   de reporter de l'énergie ; des journées-types isolées peuvent perdre
   les épisodes pluri-journaliers et la contrainte saisonnière hydro.
3. Modèle économique réduit ou sorties externes admises pour obtenir des
   formes conditionnelles ; modèles statistiques pour calibrer les biais
   et les résidus. Documenter offres/comportements/prix négatifs et limites
   de prévision parfaite : prix marginal d'une optimisation et prix réellement
   négocié ne sont pas interchangeables. Une IA récente peut accélérer ou
   corriger ce moteur, sans inventer la trajectoire d'investissement.
4. Recentrer la forme signée par mois suisse et conserver le niveau mensuel
   du solveur, puis utiliser la projection BASE/PEAK existante. La PFC centrale
   est conditionnée au marché ; des stress fondamentaux séparés peuvent
   étudier les effets de niveau, sans les injecter dans cette PFC. Aux échéances
   non cotées, le niveau dépend de priors/extrapolation explicites du solveur,
   pas d'un prétendu prix EEX observé. Aucun nouvel adaptateur d'assemblage.
5. Comparer historique seul, structure seule et hybride sur les mêmes entrées
   et contraintes. Tester creux de midi, écart soir/midi, capture solaire,
   hydro/flexibilité, négatifs, météo et stabilité. Vérifier séparément l'effet
   de la projection de marché. L'absence actuelle de vérité 2030 implique
   reconstruction du passé et stress conditionnels, pas validation anticipée
   de 2030. Les scénarios ne deviennent pas des probabilités par défaut.

Un creux solaire peut se renforcer puis se réduire si la flexibilité absorbe
les surplus ; ce n'est ni monotone ni garanti. Une batterie de même puissance
avec une durée différente ne produit pas le même effet. L'électrification peut
renforcer le pic du soir ou absorber le solaire selon son pilotage. Le climat,
les flux et les règles de marché empêchent une conversion universelle entre
GW solaires supplémentaires et baisse de prix en EUR/MWh.

## Sources publiques primaires consultées

- [IEA Electricity 2026 — Prices](https://www.iea.org/reports/electricity-2026/prices) :
  prix négatifs et lien avec manque de flexibilité ; évolutions différentes
  selon marchés et absorption des surplus par batteries/demande réactive.
- [IEA Electricity 2026 — Flexibility](https://www.iea.org/reports/electricity-2026/flexibility) :
  stockage, réseaux et demande flexible dans les systèmes à fort renouvelable.
- [REMIND–PyPSA-Eur, v1 du 5 octobre 2025](https://arxiv.org/abs/2510.04388v1),
  publication Progress in Energy 2026 : exemple de couplage de trajectoires
  sectorielles et d'opération horaire avec stockage et prix différenciés.
  Cela soutient l'approche proposée, sans prouver sa supériorité pour FMV.
- [BID3 — fonctionnalités publiques](https://bid3.afry.com/publicpages/key-functionality.html) :
  dispatch, stockage avec rendement et énergie, demande flexible, hydro,
  météo et interconnexions. Page actuelle non figée historiquement ; utilisée
  pour expliquer la méthode AFRY, pas comme preuve de performance août 2026.
- [TYNDP 2024 — scénarios](https://tyndp.entsoe.eu/resources/tyndp2024-scenarios-report) :
  source candidate de trajectoires publiques à réconcilier, pas des entrées
  déjà acquises ou validées par cette session.

## AFRY : contexte lu, valeurs non consommées

Lecture de `AFRY-CH-2026-Q2-AGENT-DATA-CONTEXT.md` et des contrats source,
sémantique et diagnostic V1. Aucun catalogue numérique, Parquet, workbook,
PDF restreint ou dérivé AFRY chargé. Aucun résultat fournisseur transmis au web.

AFRY reste benchmark/teacher-candidate, sans autorité modèle/calendrier/niveau
ou probabilités. Le rapprochement de ses 8760 créneaux représentatifs avec
le calendrier suisse n'est pas admis. Une forme annuellement centrée n'est pas
automatiquement mensuellement neutre. Les scénarios vendor ne sont pas un
jeu FMV « plus ou moins de transition » interchangeable. Le benchmark local
CPU autorisé ne lève pas le gate AFRY ni ne rouvre T057.

## Reprise concrète et vérifications

Plus petit lot recommandé : inventaire des variables 2030+ disponibles/manquantes,
contrat de forme signée et comparaison du prototype actuel avec les exigences
intertemporelles. Définir les sorties de test avant un nouvel entraînement.
Modélisation/product owner : hypothèses, mécanismes, calibration, valeur FMV.
Data engineering : provenance, dates, unités, couverture et révisions sources.

Fichiers changés : ce document, `.planning/HANDOFF.md`, et un pointeur de nuance
dans `SESSION-HANDOFF-20260907-SOTA-AUGUST-NEXT-STEP-PROPOSAL.md`.
Commandes : gardes cwd/racine Git, `Get-Content -Encoding utf8`, `rg -n` ciblés,
`git diff --check` et contrôle d'espaces des deux notes. Aucun test modèle
relancé pour ces seuls changements documentaires. Pas de nouveau hash numérique
de modèle ni d'artefact de calcul. D300/D301 et leurs manifests restent inchangés.
Pas de Warehouse, GPU, package, données ou publication externe ; toutes les
autorités précédemment false le restent.
