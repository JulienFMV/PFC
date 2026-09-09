# Proposition après D301 — état de l'art au 31 août 2026

Date de recherche : 7 septembre 2026.
Statut : recommandation documentée en réponse à la question de l'utilisateur,
pas un nouveau protocole gelé ni une sélection de modèle exécutée.
L'autorisation antérieure du travail local CPU reste acquise.

Nuance demandée ensuite sur 2030+ : lire
`SESSION-HANDOFF-20260907-DUCK-CURVE-2030-PROPOSAL.md`. Le benchmark de nouveaux
modèles ne remplace pas la couche de trajectoires physiques et de fonctionnement
horaire nécessaire pour expliquer l'évolution future de la duck curve.

## Conclusion et point de départ

D301 est un premier benchmark contrôlé, pas une couverture complète de l'état
de l'art. Le MLP corrigé reste premier au global (MAE 22.817767 EUR/MWh), devant
LightGBM (22.915155). Le gain LightGBM de 2.511676% sur les douze premiers mois
de trois origines est une sensibilité observée après les résultats, pas une
preuve indépendante permettant de choisir une bascule par échéance.

Preuves locales : `build/local-lt-benchmark-20260907/rapport-comparatif.html`,
`comparative-summary.csv`, `sensitivity-common-horizons.csv`, `final-review.json`.
Plan D301 SHA-256 :
`d0ea1b8fc21725e74da9cdf1234e4e473c3f877f0b9c39c8a1856028a74f56e1`.
Lire aussi `SESSION-HANDOFF-20260907-LOCAL-CPU-BENCHMARK.md` et la spécification
inchangée `docs/model/LT-LOCAL-CPU-BENCHMARK.md`.

## Sources primaires et portée des résultats

Les versions datées ci-dessous précèdent la borne du 31 août 2026. Les pages
de dépôt ou fiches de modèle consultées aujourd'hui sont des vérifications
techniques actuelles ; leurs branches mobiles ne constituent pas des versions
historiquement figées. Une exécution devra épingler code et poids exacts.

| Source | Version/date vérifiée | Ce qu'elle apporte à la proposition |
| --- | --- | --- |
| [FETS](https://arxiv.org/abs/2604.22328v2) | v2, 17 juillet 2026 | Chronos-2 et TiRex-2 avec covariables sont les meilleurs en agrégé sur 54 jeux énergétiques. Horizons principaux 96/192/288 pas, généralement 1–3 jours à 15 minutes ; certaines séries sont horaires. Un essai CPU réduit existe. Aucune preuve de supériorité sur une PFC CH à trois ans. |
| [Étude EPF transfrontalière](https://arxiv.org/abs/2608.17091v1) | v1, 17 août 2026 | N-HiTS et NBEATSx compétitifs lorsque les données sont limitées ; Transformers sensibles à l'adaptation et au réglage. La tâche reste la prévision du lendemain, test DE-LU 2024. |
| [Chronos-2](https://arxiv.org/abs/2510.15821v1) | v1, 17 octobre 2025 | Modèle préentraîné multivarié avec covariables. Justifie un candidat, pas un gain FMV présumé. |
| [TiRex-2](https://arxiv.org/abs/2607.01204v1) | v1, 1 juillet 2026 | Alternative récurrente avec covariables passées et futures connues ; sa propriété de contexte en flux ne démontre pas une précision à échéance arbitraire. |
| [TabPFN-TS](https://arxiv.org/abs/2501.02945v4) | v4, 26 janvier 2026 | Transformation de la prévision en régression tabulaire, intéressante pour notre jeu de caractéristiques. Résultats généraux, pas qualification LT CH. |
| [Benchmark EPF de Hornek et al.](https://arxiv.org/abs/2506.08113v2) | v2, 20 août 2025 | Sur cette comparaison du lendemain, aucun modèle préentraîné ne surpasse statistiquement la référence MSTL. Ne pas extrapoler ce résultat aux versions 2026. |

Limites de déploiement vérifiées : la [fiche Chronos-2](https://huggingface.co/amazon/chronos-2)
annonce 1024 pas de prévision, très loin de trois années horaires en une sortie.
Le [dépôt TabPFN](https://github.com/PriorLabs/TabPFN) recommande le GPU et limite
l'usage CPU aux tailles modérées ; les conditions des poids varient selon la
version. La [fiche TabPFN-3](https://huggingface.co/Prior-Labs/tabpfn_3) expose un
checkpoint temporel de mai 2026. Sa présence ne vaut pas admission à l'usage FMV.
Pas d'envoi de données FMV vers une API pour contourner une limite locale.

Un [préprint de Lipiecki et Weron](https://arxiv.org/abs/2609.00089v1), soumis
le 31 août mais indexé 2609, a aussi été lu : résultats favorables à TabPFN pour
le lendemain, sans domination économique universelle. La disponibilité publique
avant la borne stricte n'a pas été établie ; il reste hors du noyau daté de la
recommandation. La sélection ne dépend pas de ce papier frontière.

## Prochain lot proposé : benchmark local v2

1. **Représentation de la forme.** Comparer, séparément du choix de modèle,
   la cible positive actuelle et une forme signée en EUR/MWh, de moyenne
   mensuelle nulle. Conserver les jours à prix négatif dans l'apprentissage
   de cette nouvelle variante et dans toute évaluation. Vérifier le passage
   dans les interfaces existantes avant de figer le protocole. Aucun nouvel
   adaptateur d'assemblage ; aucune correction manuelle de mois après solveur.
   Une courbe centrale n'a pas à reproduire artificiellement les extrêmes spot.
2. **Échéance et équité du test.** Construire les exemples depuis des origines
   historiques avec des variables accessibles à chaque origine, afin de tester
   un effet d'échéance effectivement appris. D301 avait `years_ahead=0` partout
   à l'entraînement. Comparer une ablation sans cette variable aux candidats
   par échéance ; aucun sélecteur MLP/LightGBM choisi sur les scores D301.
   Pour étudier la récence, comparer MLP pondéré/non pondéré avec le même
   optimiseur et un budget de convergence fixé ; D301 n'a pas mesuré l'effet
   causal de la pondération, puisque ses deux fits pondérés ont échoué.
3. **Challengers bornés.** Garder saisonnier, MLP corrigé et LightGBM comme
   témoins. Introduire NBEATSx comme premier nouveau challenger neuronal.
   Réaliser ensuite une qualification locale CPU de Chronos-2, puis TiRex-2 :
   dépendances, poids, mémoire, temps et formulation compatible avec les
   échéances LT. Des profils agrégés peuvent être étudiés, mais cela change la
   représentation et exige son propre témoin ; ne pas étirer silencieusement
   une prévision de quelques jours à 36 mois. TabPFN-TS reste un candidat de
   deuxième vague, après vérification des poids utilisables et de la taille CPU.
4. **Rapport comparable.** Séparer effet de cible, effet de variables et effet
   du modèle. Même population complète, mêmes quotes, mêmes couches communes ;
   MAE/RMSE/P95 par échéance, saison et régime, coûts CPU/mémoire, conservation
   mensuelle et audits BASE/PEAK. Mesurer l'erreur de forme avant/après projection
   pour comprendre ce que les contraintes de marché absorbent. Fixer toutes
   les règles et budgets avant les nouveaux scores ; conserver les échecs.

Les années 2023–2026 de D301 sont maintenant connues de la conception : leur
réutilisation est du développement rétrospectif, même avec de nouveaux plis.
Un modèle préentraîné récent peut aussi avoir vu des séries publiques couvrant
ces années : vérifier les corpus/date des poids, déclarer tout recouvrement
inconnu et ne pas présenter le rejeu comme une simulation historique PIT.
Le futur holdout indépendant sert à confirmer le progrès ; son absence ne
bloque pas la construction locale autorisée. Ne pas rouvrir T057.

Après stabilisation de la forme horaire : qualification séparée de la couche
15 minutes et des distributions d'incertitude. D301 neutralisait intraday,
water value et physique ; il ne qualifie pas ces couches. Les séries physiques
futures doivent être disponibles à l'origine ou produites explicitement par
un modèle validé ; jamais remplacées par leurs réalisations ultérieures.

## Responsabilités et invariants

- Équipe modèle : cible, variables, protocoles, prévention des fuites, modèles,
  contrôles consommateurs et rapports. Le propriétaire produit fixe les pertes
  économiques FMV avant une sélection liée à l'exploitation hydro/couverture.
- Data engineering : disponibilité historique, révisions, trous et couverture
  PEAK/indisponibilités. Ces preuves ne sont pas fabriquées par le benchmark.
- Solveur mensuel seul détenteur du niveau ; préserver les contraintes de
  marché et montrer les conflits. Scientific v6 et D300/D301 restent inchangés.
- Proposition locale CPU, aucune activation GPU/Warehouse, aucun travail AFRY,
  aucun accès CT ou T057, toutes autorités opérationnelles/scientifiques false.

## Actions effectuées dans cette réponse et reprise

Recherche web en lecture seule et lecture des contrats/résultats locaux.
Aucun fit, scoring, téléchargement de poids, extraction de données ou runtime
de navigateur. Aucun nouvel artefact numérique ni nouvelle décision de promotion.

Fichiers changés : ce handoff et `.planning/HANDOFF.md` (pointeur uniquement).
Contrôles : cwd et racine Git canoniques avant chaque commande ; lecture ciblée
avec `Get-Content -Encoding utf8`, `rg -n`, état `git status --short`, puis
`git diff --check` et contrôle d'espaces du nouveau document. Pas de tests
modèle relancés pour cette modification documentaire ; résultat D301 conservé
à 108 tests réussis et quatre skips optionnels CT.

Limite de recherche : page éditeur FETS inaccessible (403), remplacée par le
préprint primaire v2. Il s'agit d'une revue ciblée, pas d'une revue exhaustive
de toutes les publications existantes au 31 août 2026.

Premier acte de reprise recommandé : spécifier et tester la cible signée dans
les interfaces existantes, puis figer le plan v2 et son budget avant fitting.
La proposition ne modifie pas rétroactivement le plan, les résultats ou les
critères de D301. Une adoption ultérieure du plan v2 sera consignée au journal
des décisions ; ce document ne prétend pas que cette adoption a déjà eu lieu.
