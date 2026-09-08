# Contre-relecture Claude de la réponse D317 — `c5b80a80f3` et `d07aca6c17`

Date : 8 septembre 2026. Base : branche `fix/lt-audit-remediation` au commit
`d07aca6c1745e4c052fc51e0df3fed3834d6c2e4`, réponse
`docs/model/PFC-CH-AUDIT-RESPONSE-20260908.md`, handoff
`SESSION-HANDOFF-20260908-AUDIT-RESPONSE.md`, décision D-20260908-317.
Rapport d'origine : `docs/model/PFC-CH-AUDIT-REPORT-20260908.md` (33 constats,
importé sans modification en `c2680d7`).

Méthode : lecture intégrale du diff `e8313f771c..d07aca6c17` (26 fichiers,
+2 541 / −55), sondes sur les fonctions corrigées, rejeu des tests sous Linux,
lecture des journaux GitHub Actions. Aucun artefact `build/` n'est disponible
ici ; le rejeu « 75 niveaux mensuels, écart 0,0 » et les cinq sondes locales
`independent-v1` sont rapportés comme **non rejoués indépendamment**.

## 1. Verdict

Les défauts que l'audit avait démontrés sont corrigés dans le code, testés par
des régressions rejouables depuis Git, et la CI distante est verte sur les deux
workflows. Le périmètre annoncé est respecté : aucun fichier de
`pfc_shaping/lt/`, `pfc_shaping/calibration/` ni `pfc_shaping/pipeline/` n'est
touché par le diff, ce qui garantit structurellement que D304, le solveur,
l'assembleur et la projection EEX sont inchangés, indépendamment du rejeu
local des 75 niveaux. Les 33 dispositions sont classées honnêtement ; aucune
n'est présentée comme fermée au-delà de ce que le code montre.

Quatre observations nouvelles (§4), toutes mineures, dont une décision
d'exploitation à prendre avant le jour 2 du pilote (N-1).

## 2. Vérification exécutée

| Vérification | Résultat |
|---|---|
| Diff `e8313f771c..d07aca6c17` | 26 fichiers ; code touché limité à `pfc_shaping/data/` (4 fichiers), `pfc_shaping/validation/lt_benchmark_snapshots.py`, deux scripts collecteur, six fichiers de tests, un workflow, `.gitattributes` |
| Matrice checkpoint (34 fichiers) + `test_databricks_lt_snapshot.py`, `test_snapshot_publisher_container_contract.py`, `test_lt_input_sources.py` sous Linux | **681 passed / 0 failed / 20 skipped** (43 s) ; les 20 skips sont Windows-only, `torch`/`tensorflow` absents, et le nouveau skip explicite F-13 |
| CI GitHub sur `d07aca6c17` (PR #3) | `lt-model` [run 34237033961](https://github.com/JulienFMV/PFC/actions/runs/34237033961) succès ; `publisher-runtime-v6` [run 34237033948](https://github.com/JulienFMV/PFC/actions/runs/34237033948) succès. L'échec intermédiaire CRLF sur `c5b80a8` ([run 34236508625](https://github.com/JulienFMV/PFC/actions/runs/34236508625)) est conforme au récit du handoff |
| `ruff` par fichier, avant/après, sur les 13 fichiers `.py` modifiés | aucune violation nouvelle (comptes identiques fichier par fichier) ; la dette F-15 reste celle du checkpoint |
| Sondes `merge_observed_eex_dates` (fonction réelle) | date renvoyée entièrement quarantainée → rejet ; date la plus récente non acceptée → rejet ; historique hors réponse préservé |
| Attribut réel utilisé par le nouveau reçu solveur | `MonthlyLevelAuthority.constraints` (`pfc_shaping/pipeline/monthly_curve_authority.py:77`) → `MonthlyConstraintSystem.quote_diagnostics` (`pfc_shaping/calibration/monthly_forward_curve.py:97`) : le chemin `solved.constraints.quote_diagnostics` de `solver_receipt` résout sur l'objet réel, pas seulement sur le `SimpleNamespace` du test |

## 3. Constat par constat

| Constat | Statut vérifié | Preuve dans `d07aca6c17` |
|---|---|---|
| F-13 | corrigé | `tests/test_lt_source_acquisition_outage_plan.py:165-166`, skip observé sous Linux |
| F-14 | corrigé | `.github/workflows/publisher-runtime-v6.yml:67` (`-e ".[test]"`), `.gitattributes:2` (LF), deux runs verts |
| F-16 | corrigé | `databricks_lt_materialization.py:403-406` rejette `SOURCE_DOCUMENT_CREATED` dans la lane PIT ; test `test_document_creation_is_not_causal_pit_availability` |
| F-17 | affirmation corrigée | `README.md:58`, `DATABRICKS-LT-MATERIALIZATION.md:9-13`, `DATABRICKS-LT-SNAPSHOT-INTAKE.md:8-13` |
| F-18 | corrigé | `_utc_series` `:1146-1170` : numériques et naïfs rejetés ; appliqué aux sept colonnes Silver `:751-758` ; tests 6 colonnes × 2 formes |
| F-19 | corrigé | alignement en nanosecondes `:1019-1020`, `:1056-1057`, `:1097-1099` ; test paramétré `pit/gold/spot` |
| F-20 | corrigé aux frontières visées | `databricks_lt_snapshot.py:293-296`, `lt_input_replay.py:281-287`, hash lié `:354-355` |
| F-21 | corrigé | ordre `[Availability, Revision, FirstObserved]` `:781`, égalité sémantique exigée sur les ex æquo `:782-795` |
| F-23 | partiel, comme déclaré | provenance `fmv_databricks_spot_resolution.v1` `:697-709`, validateur partagé rejeu/qualité `lt_input_replay.py:299-340` |
| F-25 | corrigé | `collect_lt_benchmark_day.py:166-183` ; quarantaine et audit écrits avant la fusion `:224-226` |
| F-26 | affirmation corrigée, politique ouverte | `run_lt_benchmark_day.py:118-141` ; tolérance non signée 0,01 exposée, `audit_conflicts_accepted=False`, `signed_policy=False` |
| F-27 | corrigé pour les nouveaux jours | `lt_benchmark_snapshots.py:38-56`, `run_lt_benchmark_day.py:41-54`, schéma v2 `:107` |
| F-28 | partiel, comme déclaré | libellé explicatif `run_lt_benchmark_day.py:138` ; `TEST_FIXTURE` inchangé |
| F-29 | corrigé | `collect_lt_benchmark_day.py:143` |
| F-30 | corrigé | `:247-255` : exception primaire propagée, échec secondaire consigné |
| F-31 | corrigé | `main` restructuré `:270-301` ; test paramétré `registry`/`config` |
| F-32 | corrigé | `:192` (`if records`), recontrôle `bound_file(root, history_ref)` `:265` |
| F-01, F-05, F-06, F-10 | reportés au prochain benchmark | conforme : aucun changement de `pfc_shaping/lt/` dans D317 |
| F-02–F-04, F-07–F-09, F-11, F-12 | retenus / requalifiés | conforme au tableau D317 |
| F-15, F-22, F-24, F-33 | ouverts | conforme ; dette lint inchangée (vérifié) |

## 4. Observations nouvelles

- **N-1 (MINEUR, décision d'exploitation avant le jour 2).** La fusion refuse
  désormais toute réponse dont la date de cotation la plus récente n'a aucune
  ligne acceptée (`collect_lt_benchmark_day.py:177-178`). Or la requête inclut
  le jour courant (`QuotationDateID <= {day}`, `:219`). Si Gold contient un jour
  des lignes partielles du jour même à 09:45 (règlement non encore chargé,
  lignes quarantainées), le jour de pilote entier échoue au lieu d'enregistrer
  la surface de la veille. C'est cohérent avec « pas de repli silencieux »,
  mais c'est une perte de jour évitable. Recommandation : borner la requête à
  `QuotationDateID < {day}` (la surface visée est le règlement de la veille),
  ou documenter explicitement que l'abandon du jour est voulu.
- **N-2 (NOTE, sémantique PIT).** L'éligibilité PIT est `known ∧ availability ≤
  origine ∧ ¬dq` (`:411`) sans exiger `FirstObservedAtUtc ≤ origine`. Avec les
  seules bases restantes (`FMV_FIRST_SEEN`, où availability = première
  observation, et `UNKNOWN_BACKFILL`, exclue), c'est sans effet. Si une base
  « publication originale » est un jour admise, il faudra choisir et documenter
  entre « disponible sur le marché » et « connu de FMV » ; les deux ne donnent
  pas la même rétro-validation.
- **N-3 (NOTE, performance).** `_utc_series` valide désormais élément par
  élément via `Series.map` (`:1156-1170`). Sur un profil Silver de 3,1 millions
  de lignes de versions (D316), cela peut coûter plusieurs minutes hors ligne.
  Sans incidence sur la correction ; une variante vectorisée (rejet des dtypes
  numériques en bloc, puis contrôle de fuseau sur le dtype) serait équivalente.
- **N-4 (NOTE, lisibilité).** Deux chaînes distinctes désignent le mode PIT :
  `"SILVER_POINT_IN_TIME"` dans les métadonnées de matérialisation
  (`databricks_lt_materialization.py:427`, contrôlée par `lt_input_replay.py:286`)
  et `"SILVER_ENTSOE_POINT_IN_TIME"` dans la configuration de rejeu
  (`databricks_lt_replay.py:42`, contrôlée par `databricks_lt_snapshot.py:293`).
  Chaque contrôle est cohérent avec son producteur ; un test croisé qui
  affirme que la métadonnée PIT n'apparaît que sous la configuration PIT
  fermerait la porte à une divergence future.

## 5. Réponse au « challenge du modèle proposé »

Le point sur l'écart entraînement/usage est juste : apprendre `b_c` sur la
moyenne mensuelle réalisée puis substituer le niveau solveur en production
introduit un biais si les deux diffèrent systématiquement (prime de risque,
erreur de prévision). Proposition ajustée, à figer avant tout résultat :

1. **Variable de conditionnement connue à l'origine, en apprentissage aussi.**
   Pour chaque mois d'entraînement `m`, utiliser le niveau BASE de `m` impliqué
   par la surface EEX observée à une échéance fixe avant livraison (par
   exemple la dernière date de cotation avant le premier jour de `m`, depuis
   `eex-normalized-history`), ou la reconstruction solveur à cette date quand
   elle existe. C'est la même nature d'information qu'en production.
2. **Lane oracle séparée** avec la moyenne réalisée, publiée comme diagnostic
   de borne supérieure, jamais comme candidat.
3. **Ablation exacte** : `b_c = 0` doit reproduire D304 à 1e-9 ; une seule
   recette globale, pente régularisée, centrage sur les durées réelles du mois
   suisse, mesure avant et après projection BASE/PEAK.
4. **Gates** : celles de F-05/F-06/F-10 corrigées, populations communes,
   veto par origine, seuils pré-origine ; aucune sélection sur 2024.

Sur le témoin Git : accord, un commit public horodate un engagement, il ne
remplace ni la custody de la vérité ni un registre de sélection figé avant
ouverture. Sur l'accord CH–UE : « dominant » est une hypothèse de rang, pas
une mesure ; le point d'ancrage quantitatif disponible est l'étude d'adéquation
ElCom citée dans le rapport, à confronter aux capacités commerciales
observées avant toute utilisation dans la forme.

## 6. Ce qui reste ouvert, dans l'ordre que je recommande

1. Décider N-1, puis lancer le jour 2 réel avec les contrôles v2.
2. Politique signée des conflits EEX (F-26) : le reçu expose maintenant les
   diagnostics ; il manque la règle métier qui dit quel parent l'emporte et
   pourquoi, signée par le propriétaire produit.
3. Qualification PIT réelle sur PRD (F-17/F-20) : blocs A03, base de
   disponibilité, et la lane latest-observed versus SQL realized (F-22).
4. Garde-fous F-01/F-05/F-06/F-10, puis le benchmark restreint « forme signée
   conditionnée au niveau » avec la variable de §5.1.
5. Dette lint F-15, à traiter séparément et sans mélange avec un lot modèle.

Autorités : les six restent `false`. Aucun code, donnée, artefact ni autorité
n'a été modifié par cette contre-relecture ; seuls ce document et son handoff
sont ajoutés sur `claude/pfc-ch-audit-15kn4o`.
