# Audit indépendant Claude — checkpoint `dee652bc91` (branche `fix/lt-audit-remediation`)

Date de l'audit : 8 septembre 2026. Auditeur : Claude (session distante, checkout
public GitHub `JulienFMV/PFC`, Linux, CPython 3.11). Point d'entrée suivi :
`docs/model/PFC-CH-AUDIT-ENTRYPOINT-20260908.md`, puis `AGENTS.md`,
`.planning/HANDOFF.md`, `DECISION-LOG.md` (D300–D315) et les handoffs D304–D315.

Commit audité : `dee652bc91` (code, tests, protocoles). Le commit suivant
`e8313f771c` (docs/lt : D316, `CH-NATIONAL-INPUT-READINESS-20260908.md`,
`CH-STRUCTURAL-EVENT-REGISTRY-PROPOSAL.md`, watchlist JSON) ne touche que la
documentation et la planification ; il a été lu et pris en compte au §10, et la
branche d'audit a été posée sur lui pour que les références de fichiers soient
valides. Les résultats de tests ci-dessous ont été obtenus sur `dee652bc91` ;
`e8313f771c` ne modifie aucun fichier `.py`.

Périmètre respecté : aucun run de production, AFRY, T057, mutation de données
protégées ni promotion de modèle. Aucun accès Databricks/Warehouse. Tout ce qui
est sous `build/` (artefacts D300–D314) est **absent du checkout public** et
n'a donc pas été rejoué : chaque chiffre de performance cité dans les handoffs
(MAE 20.345234, gains 1.737 %, 9 QUOTE_CONFLICT, etc.) est rapporté ici comme
**non rejoué indépendamment**, jamais comme confirmé.

## 0. Verdict en bref

- Le checkpoint est **cohérent avec ce qu'il annonce** : c'est un instantané
  de code, tests et protocoles ; aucune des six autorités (production,
  promotion, scientific_admission, trading, externally_registered,
  countable_origin) n'est accordée par le code, et les invariants solveur /
  assembleur / projection EEX sont réellement appliqués dans le code (voir §3).
- La construction horaire signée D304 est numériquement saine : neutralité
  mensuelle, grilles DST/bissextiles, conservation des moyennes solveur et
  projection dure BASE/PEAK sont vérifiées par le code et par des tests que j'ai
  rejoués. En revanche, aucune de ses **affirmations de performance** n'est
  reproductible depuis Git (artefacts locaux), et les origines 2023–2026 sont
  des années de développement exposées, pas un holdout.
- Points à corriger (détail §§3–7) : deux défauts majeurs de
  code (F-25 : effacement silencieux de l'historique EEX d'une date renvoyée
  mais entièrement quarantainée ; F-18/F-19 : parsing d'horodatages et contrôle
  de grille de la matérialisation qui échouent « ouvert »), trois défauts
  majeurs de sémantique/affirmation (F-26 : « aucun conflit accepté » n'est
  vrai qu'à la gate d'audit, le solveur tranche à 0.01 EUR/MWh ; F-16/F-17 : la
  lane PIT Silver utilise un horodatage non causal et n'a jamais tourné sur
  PRD ; F-20 : l'étiquette PIT/latest n'est lue par aucun consommateur), un
  défaut de CI (F-14) et une série de points mineurs. Aucun de ces défauts
  n'invalide les résultats D304–D310 tels qu'ils sont qualifiés (développement
  local exposé), car la lane réellement utilisée pour D300 est la lane
  latest-observed, la plus défensive.
- Bloqueurs production (inchangés et correctement documentés par le dépôt) :
  absence de vintages PIT réels, holdout futur non enregistré, un seul jour
  réel de pilote, pas d'ordonnanceur, custody de la vérité future absente.
  Rien dans ce checkpoint ne les lève, et rien ne prétend les lever.

## 1. Méthode et ce qui a été réellement exécuté

| Étape | Résultat |
|---|---|
| `git fetch origin fix/lt-audit-remediation` ; `git cat-file -t dee652bc91` | commit présent, tip de la branche, 404 commits devant `main` (`2c9bd5c`), base commune = `main` |
| Branche d'audit `claude/pfc-ch-audit-15kn4o` | rebasée sur `dee652bc91` (aucun commit propre avant l'audit) |
| Env : `uv venv` + `pip install -e ".[test,validation,ingest]" lightgbm==4.6.0 duckdb` | pandas 2.3.3 / numpy 2.0.2 / scipy 1.13.1 / lightgbm 4.6.0 / pytest 8.4.2 / ruff 0.15.12 |
| Matrice checkpoint (34 fichiers de `PFC-CH-AUDIT-CHECKPOINT-VERIFICATION-20260908.json`) | **576 passed / 1 failed / 19 skipped** (23.5 s) |
| Suite complète `tests/` (4 996 tests, plafond 1700 s) | **4 638 passed / 220 failed / 73 errors / 65 skipped** (600 s) ; échecs très majoritairement liés à l'environnement (voir §2.2) |
| `ruff check` (config du projet : E4, E7, E9, F, I) sur les 41 fichiers `.py` changés par le checkpoint | 264 violations : 170 E701, 34 E402, 31 I001, 25 E702, 4 F401 |
| CI GitHub sur `dee652bc91` (PR #3) | `lt-model` : succès ; `publisher-runtime-v6` : **échec** (voir §2.3) |
| Lecture de code | assembler, signed_benchmark, evaluation_curve_assembly, local_benchmark, signed_intraday, shape_intraday (residual), shape_constraints, quant_shape_optimizer, water_value (delta), collecteur, scripts D304–D310, matérialisation Databricks, tests associés |

Réconciliation avec le JSON de vérification du checkpoint (591 passed / 0 failed /
5 skipped sous Windows) : 576 + 1 + 14 = 591. Les 14 skips supplémentaires sont
des contrats « Windows-only » de `tests/test_governed_lt_acquisition.py`
(device names, path grammar, handle-share), légitimement sautés sous Linux. Le
seul échec est `tests/test_lt_source_acquisition_outage_plan.py::test_existing_eex_capture_matches_the_bound_local_bytes`
(ligne 157-165) : il lit `build/databricks-eex-daily/2026-08-05/eex_ch_power.ndjson`,
qui n'existe que sur le poste local. Le chiffre « 591 pass » est donc
**crédible mais non reproductible depuis le dépôt public** ; ce test devrait
être un `skip` explicite quand les octets locaux sont absents (finding F-13).

## 2. Vérification du checkpoint lui-même

### 2.1 Git

Le commit `dee652bc919d06345f71304d1f1eaacf0edf7bc7` (auteur Julien Battaglia,
2026-09-08 14:52:29 +0200, parent `e4bfd45`) modifie 139 fichiers
(+22 673 / −118). Côté code : 41 fichiers `pfc_shaping/`, `scripts/` et `tests/`
(+8 157 / −60). Les modifications de code de production stricto sensu sont
limitées : `assembler.py` (+134, lane signée), `shape_intraday.py` (+51,
`price_conditioned_residual`), `shape_hourly_mlp_hydro.py` (+29),
`databricks_lt_materialization.py` (+131), `databricks_eex_daily_snapshot.py`
(+10), `governed_lt_acquisition.py` (+12), `spot_source_reconciliation.py`,
`entsoe_day_ahead_export.py`. Le reste est constitué de nouveaux modules
d'évaluation/expérimentation, de scripts bornés, de tests et de documentation.
Aucune donnée brute, aucun `.parquet`/`.pkl` de modèle, aucun secret évident
n'est versionné (contrôle `git show --stat` + `.gitignore`).

### 2.2 Tests

Matrice du checkpoint : voir §1. Détail des skips Linux : 3× path grammar, 3×
filename grammar, 5× device name, 1× control character, 1× DOS device, 2× handle
share, 1× handle-relative rename, 1× wheel audité, 2× `torch`/`tensorflow` absents.

Suite complète (`tests/`, deux exécutions identiques, 4 996 tests) : 4 638
réussites, 220 échecs, 73 erreurs, 65 skips. Le dépôt ne prétend pas que la
suite complète passe (« record full-suite timeout » `d743d24`, « 2 known Phase5
failures » dans chaque handoff). Classification des causes à partir des
traces une ligne (`--tb=line`) :

- **Contrat poste de travail** (cwd canonique `C:\Users\jbattaglia\PFC_LT`,
  `TEMP` « not configured » / « must be repo-local », espace de travail AFRY
  exact, grammaire de chemins Windows `\\?\GLOBAL…`, « path must be absolute »)
  : ~70 échecs, dont les 38 `EntsoeCadencePackageBindingError: TEMP is not
  configured` et la plupart des 49 « Regex pattern did not match » (le message
  attendu n'est jamais atteint parce qu'une garde d'environnement lève avant).
- **Octets locaux absents** (`build/databricks-eex-daily/2026-08-05/…`,
  `build/market-time-regime-evidence-20260730-v3`, manifestes de preuve
  D233/D235/D240/D241, ledger v10, run15, inventaires d'origines) : 73 erreurs
  de setup (`test_tier2_monthly_eex_fold_evidence.py`,
  `test_eex_current_monthly_research_solution.py`,
  `test_ch_lt_origin_*`) et ~30 échecs « … is unavailable ».
- **Environnement d'audit** : interpréteur de venv symlinké (« publisher runtime
  file is linked or non-regular », « artifact path cannot traverse a symlink »),
  bundle CA FMV épinglé absent (6), largeur de placeholder Conda (1).
- **Échecs documentés** : `tests/test_phase05_negative_prices.py` (2, « 5bis-A
  rollback regression FAILED », goldens Phase 5), exactement ceux annoncés.
- **Non attribués individuellement** (≈ 10) :
  `test_check_monthly_curve_promotion_from_manifests.py` (6, dont trois
  « product_replay_uses_captured_bytes »), `test_ch_lt_origin_registry_protocol.py`
  (3), `test_archive_entso_dataset_script.py` (1). Ils dépendent
  vraisemblablement d'octets capturés localement, mais je ne l'ai pas prouvé
  un par un.

Conclusion : aucun échec de la suite complète n'a été rattaché à un défaut du
code du checkpoint ; la suite n'est simplement pas exécutable hors du poste
canonique, ce qui est une limite de portabilité (voir F-13), pas un défaut de
calcul. Les 34 modules du checkpoint et les quatre modules D307–D310, cinq
modules collecteur et le module de matérialisation passent tous ici.

### 2.3 CI GitHub (PR [#3](https://github.com/JulienFMV/PFC/pull/3))

- `lt-model` (contrats LT bornés + contrats données/PIT) : **succès** sur
  `dee652bc91` ([run 34228620940](https://github.com/JulienFMV/PFC/actions/runs/34228620940)).
- `publisher-runtime-v6` : **échec** sur `dee652bc91`
  ([run 34228620773](https://github.com/JulienFMV/PFC/actions/runs/34228620773))
  et déjà sur `d743d24` le 21 août. Cause lue dans le journal du job :
  `tests/conftest.py:23` importe `pfc_shaping.cli.governed_release`, dont la
  chaîne d'import charge `pandas`, alors que le workflow n'installe que
  `pytest==8.4.2 ruff==0.15.12` (`.github/workflows/publisher-runtime-v6.yml:65-67`).
  Résultat : `ModuleNotFoundError: No module named 'pandas'` avant toute
  collecte. Défaut de configuration CI, antérieur au checkpoint, mais non
  mentionné dans le handoff public (« A checkpoint is not a statement that
  every … external CI gate passed » couvre le cas, sans le nommer). → F-14.

### 2.4 Lint

La configuration `ruff` du projet (`pyproject.toml`, `select = ["E4","E7","E9","F","I"]`)
n'est pas respectée par les fichiers ajoutés : 170 E701/25 E702 (plusieurs
instructions par ligne, surtout dans les tests paramétrés et les scripts
`verify_*`), 34 E402 (imports après manipulation de `sys.path` dans
`run_lt_local_benchmark.py`, `run_lt_signed_benchmark.py`), 31 I001, 4 F401
(`numpy` inutilisé dans `scripts/run_lt_maturity.py:9`,
`FEATURES` inutilisé dans `scripts/run_lt_signed_benchmark.py:30`). Aucun
E9/F8 (pas d'erreur de syntaxe ni de nom indéfini). → F-15 (mineur, mais
contradictoire avec le principe « match the surrounding style » d'AGENTS.md).

### 2.5 Findings transverses

- **F-13 (MINEUR, portabilité des tests).**
  `tests/test_lt_source_acquisition_outage_plan.py:157-165` lit des octets
  sous `build/databricks-eex-daily/2026-08-05/` sans garde : échec assuré sur
  tout checkout public. Le transformer en `pytest.skip("local bound bytes
  absent")` explicite, ou déplacer le contrôle dans un script d'audit local.
- **F-14 (MINEUR, CI).** `.github/workflows/publisher-runtime-v6.yml:65-67`
  n'installe que `pytest` et `ruff`, alors que `tests/conftest.py:23` importe
  la chaîne `governed_release → atomic_promotion → candidate_evidence → pandas`.
  Le job « Fail-closed contract tests » échoue avant collecte sur chaque PR
  (runs du 21 août et du 8 septembre). Installer `-e ".[test]"` ou isoler ces
  deux tests d'un `conftest` sans dépendance pandas.
- **F-15 (MINEUR, style).** 264 violations de la configuration `ruff` du
  projet dans les fichiers du checkpoint (voir §2.4) ; `ruff check --fix`
  corrige 35 d'entre elles, le reste (E701/E702/E402) demande une réécriture
  mécanique. Aucun impact fonctionnel.

## 3. Point 1 — construction horaire signée (signed_benchmark → evaluation_curve_assembly → assembler)

### 3.1 Ce que le code garantit (vérifié)

1. **Cibles fermées avant l'origine.** `closed_month_targets`
   (`pfc_shaping/lt/signed_benchmark.py:12-40`) ne garde que les mois suisses
   complets dont la fin est ≤ origine, construit la grille attendue depuis les
   bornes de mois `Europe/Zurich` (donc 743/745 h en mars/octobre, 696 h en
   février bissextile) et exige l'égalité exacte des index ; les trous
   intérieurs font échouer `center_signed_hourly_shape`
   (`structural_readiness.py:170-203`). Tests rejoués :
   `tests/test_signed_benchmark.py` (deux heures 02:00 d'octobre distinctes,
   mois ouvert exclu, valeurs négatives conservées).
2. **Référence calendaire strictement pré-livraison.**
   `calendar_cell_reference` (`signed_benchmark.py:42-88`) refuse
   `index[-1] >= delivery[0]`, moyenne non tronquée par cellule
   (saison, type_jour, heure_hce) avec repli saison/heure → heure → global, et
   poids optionnels finis et positifs.
3. **Lane signée de l'assembleur.** `PFCAssembler.build(..., signed_hourly_shape=)`
   (`assembler.py:489-516`, `581-585`, `1434-1521`) exige : solveur mensuel
   comme autorité de niveau, `country="CH"`, deux couches legacy sautées,
   aucune couche annexe/floor active, intraday exactement neutre, une clé BASE
   mensuelle **pour chaque mois** livré, neutralité mensuelle du shape ≤ 1e-9,
   neutralité par heure-mère du résidu intra-horaire, neutralité mensuelle du
   delta hydro. Le shape horaire est transporté ×4 (`np.repeat`), recentré par
   `_preserve_monthly_base_means`, puis projeté par
   `_project_final_solver_products` (`assembler.py:1523-1632`) : QP à
   `lambda_prior=1`, sans lissage, contraintes dures BASE mensuelles et
   PEAK/OFFPEAK disjointes pour les clés PEAK cotées à couverture complète ;
   résidu > 1e-6 → exception ; couverture PEAK partielle → exception.
4. **Définition PEAK cohérente** entre `shape_constraints.eex_peak_mask`
   (`:186-195`, `hour <= 19`), `assembler._is_peak_timestamp` (`:421`,
   `hour < 20`) et `local_benchmark.score_curves` (`:274`) : lundi–vendredi
   08:00–20:00 heure locale, jours fériés inclus, conforme à la définition EEX
   Swiss Peak.
5. **Scoring niveau/forme séparé.** `score_curves` (`local_benchmark.py:232-294`)
   recentre vérité et prédiction par mois suisse (`_monthly_center`,
   `evaluation_engine.py:411-429`) avant toute erreur ; la population
   d'évaluation est la même pour tous les candidats (`run_lt_signed_benchmark.py`
   lève si `common_population` change) ; les mois de vérité incomplets sont
   exclus pour tous.
6. **Contrôles de non-régression** dans `run_lt_signed_benchmark.py` : rejeu
   exact du chemin natif D301 (≤ 1e-10) et aller-retour signé (≤ 1e-9). Ce
   dernier est mathématiquement attendu (les corrections KKT sont constantes
   dans chaque heure car toutes les contraintes sont des moyennes d'heures
   entières) ; il valide l'implémentation, pas la compétence du modèle.
7. Tests rejoués avec succès : `test_signed_hourly_assembly.py` (mois
   2024-02/03/10 × base −50/0/50, PEAK repricé à 1e-9, 12 entrées invalides
   rejetées), `test_signed_composition.py`, `test_price_conditioned_intraday.py`
   (dont 2032-10, prix −40/0/0.01/80).

### 3.2 Findings du point 1

- **F-01 (MINEUR, robustesse) — niveaux de repli silencieux dans la lane
  multiplicative commune.** `evaluation_curve_assembly._validated_prices`
  (`:347-362`) n'exige qu'*au moins une* clé mensuelle, pas une clé par mois
  livré. Or `_resolve_base` (`assembler.py:1239-1252`) remplit un mois sans clé
  par la cote trimestrielle/annuelle, puis par les *années précédentes*, puis
  par **interpolation linéaire** avec un simple `logger.warning`. Comme
  `calibration_buckets` laisse ce mois sans contrainte (`eex_contract_selection.py:71-80`),
  la vérification « moyennes mensuelles = B » (`evaluation_curve_assembly.py:337`)
  passerait avec un B qui n'est pas un niveau solveur. Non atteignable avec
  les entrées D300/D301 (tous les mois ont une clé), mais la lane signée fait
  déjà le bon contrôle (`assembler.py:1470-1472`) ; l'aligner ici coûte trois
  lignes. Recommandation : exiger dans `_validated_prices` une clé `YYYY-MM`
  pour chaque mois de `delivery`, et faire échouer `_resolve_base` (au lieu de
  `warning`) quand `monthly_level_authority == "solver"`.
- **F-02 (NOTE, équité de comparaison D304).** Les candidats « ratio »
  utilisent une table `f_W` **refittée** sur les mois fermés
  (`run_lt_signed_benchmark.py:160-166`, demi-vie 180 j de
  `ShapeHourlyMLP`), alors que l'incumbent `d301-current-mlp` garde le `f_W_`
  de son pickle D301 (jours complets, pas mois fermés). La comparaison
  « ratio-seasonal vs d301-current-mlp » n'est donc pas à couche `f_W`
  strictement commune. Sans effet sur la conclusion principale (le candidat
  signé n'utilise pas `f_W`), mais le handoff D304 devrait le dire.
- **F-03 (NOTE, sémantique du candidat retenu).** La « signed seasonal
  reference » est une climatologie EUR/MWh de cellules calendaires moyennée
  sur toute l'historique disponible, indépendante du **niveau** de prix. Elle
  injecte donc l'amplitude EUR/MWh moyenne des régimes 2021–2023 (crise gaz)
  dans des mois de niveau bas et inversement. C'est exactement le mécanisme
  qui explique, à mon sens, les régressions « delivery 2024 » et « ramps »
  observées dès qu'on pondère le passé récent (D307/D308). Ce n'est pas un
  défaut de code, c'est une limite de représentation à traiter en priorité
  (§9.1).
- **F-04 (NOTE).** `_check_energy_consistency` (`assembler.py:1816-1864`)
  calcule la couverture trimestrielle avec 28 jours pour février quelle que
  soit l'année ; sans effet (seuil de couverture 0.90), mais trompeur.

## 4. Point 2 — D307–D310 (recency, stabilité, révisions, maturité)

Revue déléguée à un sous-agent (lecture intégrale des scripts `run_*`,
`verify_*`, `audit_lt_hourly_revisions.py`, `lt_maturity_experiment.py`, des
quatre fichiers de tests et des docs/handoffs), puis contre-vérifiée par
sondage aux lignes citées. Les quatre fichiers de tests passent (61 tests).

### 4.1 Verdict par critère

| Critère | Verdict | Base dans le code |
|---|---|---|
| (a) Configurations globales gelées | conforme | demi-vies 365.25/730.5 en constantes (`run_lt_hourly_recency.py:24`), blends 25 %/50 % (`run_lt_hourly_stability.py:24-25`), ridge/base fixés (`lt_maturity_experiment.py:9,64,68-78`) ; aucune sélection par mois/horizon ; `adoption=False` codé en dur |
| (b) Seuils pré-origine, pas de fuite | conforme | labels via `closed_month_targets` (mois ouverts exclus) ; seuils = quantiles des mêmes frames pré-origine (`run_lt_hourly_recency.py:132-134`, `run_lt_hourly_stability.py:37-54,102-103`, `audit_lt_hourly_revisions.py:40-64`) ; paires D310 strictement causales (`lt_maturity_experiment.py:40-53`) |
| (c) Populations communes | conforme, une assertion manquante en D310 | `score_curves` refuse les NaN et choisit les mois complets indépendamment du candidat ; D307/D308 assertent l'égalité des populations (`run_lt_hourly_recency.py:186-187`, `run_lt_hourly_stability.py:152-155`) |
| (d) Séparation niveau/forme | conforme | erreur de forme = prédiction recentrée − vérité recentrée ; identité `MSE(full) = MSE(shape) + MSE(level)` recalculée par les vérificateurs ; `save_curve` lève si résidu solveur > 1e-9 |
| (e) Rapport fidèle des gates échouées | conforme avec réserves | aucun chemin ne transforme un échec en succès ; réserves ci-dessous (F-05, F-06) |
| (f) Résultats exposés ≠ holdout | conforme | rôles hérités de D301 ; toutes les docs étiquettent 2023–2026 « exposed development » ; holdout D309 `DRAFT_NOT_INDEPENDENTLY_REGISTERED` (`audit_lt_hourly_revisions.py:299-306`) |

### 4.2 Findings du point 2

- **F-05 (MINEUR) — la règle de support rend invisibles au veto les régimes
  à origine unique.** `verify_lt_hourly_recency.py:227-231` :
  `supported = hours >= 168 and origins >= 2`, puis seuls les statuts
  `REGRESSION` comptent (`:242-244`). `YEAR_2023` n'existe que pour l'origine
  2023 et ne peut donc jamais opposer de veto ; D307 n'avait pas de veto par
  origine (la perte de `signed-hl730` sur l'origine 2023, ≈ +7.3 % d'après le
  handoff, n'était pas « gate-relevant »). D308 a ajouté le veto par origine
  (`verify_lt_hourly_stability.py:33-38,49,54`), réutilisé par D310, mais
  `screening.json` n'enregistre toujours pas le nombre de ratios défavorables
  écartés comme UNSUPPORTED. Recommandation : ajouter
  `unsupported_adverse_segments` (compte + liste) à `screening.json` /
  `decision.json`, et rendre le veto par origine obligatoire.
- **F-06 (MINEUR) — D310 n'asserte pas l'identité des populations.**
  `run_lt_maturity.py:117-122` appelle `score_curves` par candidat sans le
  contrôle `population.equals(errors.index)` de D307/D308 ; idem
  `verify_lt_maturity.py:131-133`. Structurellement identiques ici (même
  `baseline.index`, même vérité), mais l'invariant n'est pas appliqué.
- **F-07 (NOTE) — bord NaN du gate D307.** `a = mae/ref.mae if ref.mae > 0
  else nan` puis `max(a, b) > 1.05` : `max(nan, 1.5) > 1.05` vaut `False`,
  donc PASS. Inatteignable en pratique (MAE de référence > 0) et corrigé en
  D308 (`verify_lt_hourly_stability.py:27-28`).
- **F-08 (NOTE) — seuils D307 calculés pendant le run**, écrits dans
  `thresholds.json` après `plan.json` (`run_lt_hourly_recency.py:132-134`) ;
  déterministes et pré-origine, recalculés par le vérificateur, mais pas
  « gelés dans le plan » comme l'écrit la doc ; D308 les a mis dans le plan.
- **F-09 (NOTE) — plans D307/D308 sans champ de politique explicite** ;
  l'étiquette nue `assessment` se propage dans les CSV/JSON alors que D304 et
  D309/D310 portent `comparison_policy`/`source_policy`. Ajouter
  `evidence_class = EXPOSED_DEVELOPMENT_NO_HOLDOUT` à chaque JSON de screening.
- **F-10 (NOTE, à sécuriser) — le `hourly.pkl` de production (ajusté jusqu'en
  2026) est passé à l'assembleur pour des origines 2021–2026**
  (`run_lt_hourly_recency.py:120,151`, `run_lt_hourly_stability.py:113,136`,
  `audit_lt_hourly_revisions.py:245,273`, `run_lt_maturity.py:72,109`). Inerte
  aujourd'hui : `_build_signed_hourly_shape` ne lit jamais `self.sh`, et D307
  vérifie l'égalité à 1e-9 avec la courbe D305 construite avec les modèles par
  origine. Mais rien n'empêche une future évolution de la lane signée d'utiliser
  `self.sh`, ce qui injecterait silencieusement un fit post-origine. Passer un
  objet sentinelle dont `apply` lève, ou asserter `frame.f_H.eq(1).all()`.
- **F-11 (NOTE) — âges de récence mesurés depuis la dernière heure
  d'entraînement**, pas depuis l'origine (`run_lt_hourly_recency.py:148-149`) ;
  équivalent pour des moyennes pondérées (facteur constant), mais différent de
  la convention `recency_weights` du dépôt (`evaluation_challengers.py:176-199`),
  et les `weights.parquet` sauvés ne sont pas relatifs à l'origine.
- **F-12 (NOTE) — asymétries mineures** : origine courante codée en dur dans
  `verify_lt_hourly_recency.py:76` ; p95 calculé différemment (`stats()` vs
  `_weighted_quantile`, exclu du cross-check) ; compteurs d'activité écrits en
  littéraux (`seasonal_calculations=21`, `run_lt_hourly_recency.py:212` ;
  `fits=14`, `run_lt_maturity.py:156`, celui-ci au moins gardé par
  `assert len(receipts) == 14`).
- **Multiplicité (design, pas code).** Au moins huit candidats horaires (plus
  D305/D306) ont été criblés sur les mêmes quatre origines exposées avec les
  mêmes gates 2 %/5 %, chaque lot étant choisi en connaissance du précédent.
  Toutes les docs le disent ; un futur `local_screen_pass = True` ne peut être
  lu que comme générateur d'hypothèse. Le journal de décisions devrait tenir
  le compte cumulé des candidats criblés.

### 4.3 Affirmations D307–D310 : supportées ou non par le code

Supportées (structure et formules, valeurs non rejouées) : gates ≥ 2 % MAE et
RMSE, ≥ 3/4 origines, aucune régression supportée > 5 % ; déclencheur D309
« ratio > 1.10 sur les deux erreurs, ≥ 8 événements, ≥ 3 origines » ; veto
seam D310 ≤ 1.05 ; pourcentages du journal (1.737363 %/3.198581 %,
0.931579 … 1.155386 %, −5.447341 %, ratios 0.962204 = 18.264819/18.982272 et
0.797615 = 13.446529/16.858420) arithmétiquement cohérents avec la formule
`1 − cand/ref` ; comptes de lignes (3 808/960, 8 768/1 920, 59 120/11 136,
72 337 h appariées, 28 513/114 052/219 268 lignes d'export) reproductibles
depuis la structure des boucles et les fenêtres ; 136 617 paires D310 pour
l'origine 2023 reproduites par un rejeu synthétique de `pairs()`.

Non vérifiables depuis ce checkout : toutes les valeurs absolues et hashes ;
« protocole gelé avant résultats » (scripts, docs, tests et handoffs D307–D310
sont tous entrés dans Git dans le seul commit `dee652b` ; le seul lien de
gel est le hash de la doc épinglé dans `plan.json`, sous `build/`).

## 5. Point 3 — `databricks_lt_materialization.py`

Revue déléguée à un sous-agent (lecture du module, de ses tests, de
`databricks_lt_replay.py`, `databricks_lt_snapshot.py`, `lt_input_replay.py`,
`lt_replay_transforms.py`, `governed_lt_acquisition.py`, des docs
`DATABRICKS-LT-MATERIALIZATION.md`, `ENTSOE-DAY-AHEAD-EXPORT-V2.md`,
`LT-SOURCE-ACQUISITION-OUTAGE-RUNBOOK.md` et des SQL sous `docs/data/sql/`),
avec 13 sondes exécutées contre les vraies fonctions ; contre-vérifiée par
sondage aux lignes citées. Les 45 tests du module passent. Aucune donnée
Databricks n'est dans le checkout ; le script qui a appelé la lane
« latest-observed » pour D300 n'est pas public.

### 5.1 Ce qui est conforme (vérifié)

- Les trois lanes ENTSO-E portent des étiquettes honnêtes et distinctes :
  `SILVER_POINT_IN_TIME` (`:420`), `GOLD_CURRENT_SERVING` (`:502`),
  `SILVER_LATEST_OBSERVED_LOCAL_ONLY` avec `historical_pit_authorized: False`
  (`:605,616`) ; toutes les autorités sont fausses. **Aucun chemin de ce module
  ne réétiquette du latest-observed en PIT.**
- Aucun `SeriesKey`/`FieldName` deviné : contrat explicite lié par hash à la
  dimension (`:284-343`, `:775-784`) ; séquences de classification AT/DE-LU
  passées en paramètres SQL, sans défaut.
- Expansion d'intervalles en arithmétique UTC demi-ouverte : jours suisses de
  23/25 h et 29 février corrects (sondes : 92/100/96 lignes) ;
  `UNKNOWN_BACKFILL` ne peut pas remplir une grille PIT ; égalités exactes de
  vintages fail-closed (`:754-767`) ; blocs superposés/contradictoires rejetés
  dans la lane latest-observed ; alignement modulo-résolution vérifié dans
  cette lane (`:563-568`).
- Hachage déterministe et invariant à l'ordre là où la clé de tri est unique ;
  `allow_nan=False, sort_keys=True` ; correction µs→ns de D300 en place (`:1051`).

### 5.2 Findings du point 3

- **F-16 (MAJEUR) — la lane PIT accepte `publication_timestamp_utc` (le
  `createdDateTime` de la réponse XML) comme instant causal pour les lignes
  `SOURCE_DOCUMENT_CREATED`, sans preuve de publication originale.**
  `databricks_lt_materialization.py:1155-1158` exige `availability ==
  publication` pour cette base, et la sélection PIT l'utilise comme coupure
  (`:404`, mode `SILVER_POINT_IN_TIME` `:420`). Le dépôt lui-même dit que ce
  champ « n'est pas l'heure de publication historique »
  (`.planning/HANDOFF.md:367-371`, `ENTSOE-DAY-AHEAD-EXPORT-V2.md:11-16`), et
  la lane day-ahead l'impose (`entsoe_day_ahead_consumption.py:540-546`,
  `ORIGINAL_PUBLICATION_UNPROVEN`). Sonde : une révision publiée à 03:00Z mais
  vue par FMV à 03:30Z est sélectionnée à l'origine 03:00Z
  (`excluded_after_origin_rows: 0`). Atténuations : fixtures livrées en
  `FMV_FIRST_SEEN`, autorités fausses, et F-17 fait échouer les vraies lignes.
  Correction : rejeter `SOURCE_DOCUMENT_CREATED` dans
  `materialize_entsoe_pit_features` ou exiger `original_publication_proven`
  par ligne ; la disponibilité PIT ne doit jamais précéder `FirstObservedAtUtc`.
- **F-17 (MAJEUR, affirmation) — la lane PIT Silver n'a jamais tourné sur des
  données PRD et les rejetterait.** `:1163` `invalid |= publication_utc.gt(first_utc)`
  est inconditionnel, alors que le dépôt enregistre que **toutes** les 17 925
  lignes PRD ont `publication > first_seen` (`.planning/HANDOFF.md:366-367`) et
  que le test du checkpoint modélise cette réalité
  (`tests/test_databricks_lt_materialization.py:301-302`) ; la lane PIT
  n'accepte en outre que des lignes atomiques (`:986-994`) alors que le Silver
  PRD contient des blocs de longueur variable. La seule lane exercée sur PRD
  est `materialize_entsoe_latest_observed_features` (ajoutée dans `dee652b`),
  qui n'appelle jamais `_validate_availability_semantics`. Les phrases
  « Silver vintages → point-in-time entso raw and derived features »
  (`DATABRICKS-LT-MATERIALIZATION.md:24`) et « Stage 2 is implemented
  end-to-end for Gold spot and Gold/Silver ENTSO-E »
  (`DATABRICKS-LT-SNAPSHOT-INTAKE.md:45-46`) décrivent du code validé sur
  fixtures synthétiques seulement. À écrire tel quel dans les docs.
- **F-18 (MAJEUR) — parsing des séries temporelles sans contrôle d'unité ni
  de fuseau : une erreur d'unité échoue « ouvert » dans une frame étiquetée
  PIT.** `_utc_series` (`:1117`) fait `pd.to_datetime(values, errors="coerce",
  utc=True)` sans `unit` ni contrôle de dtype (contraste `_utc_scalar`
  `:1104-1108` qui rejette le naïf). Sondes : epochs int64 en **secondes** →
  interprétés en ns (1970) → toutes les lignes « disponibles » → la révision
  post-origine fuit (`load_mw[0] = 999.0`, mode PIT) ; horodatages naïfs
  Europe/Zurich → décalage de deux heures de la grille. Le projet a déjà eu un
  incident d'unité dans ce fichier (µs pris pour ns, D300). Correction :
  exiger `is_datetime64_any_dtype` avec `tz is not None` (ou ISO tz-aware),
  rejeter int/float/naïf ; tests dédiés.
- **F-19 (MAJEUR) — un décalage sub-seconde cohérent passe les lanes PIT,
  Gold-current et spot ; le contrôle de grille tronque les ns aux secondes.**
  `:1066-1067` `np.diff(asi8) // 1_000_000_000` puis `== 900` : 900.5 s → 900.
  `_expand_entsoe_intervals` (`:986-994`) et `_expand_spot_intervals`
  (`:1016-1025`) ne valident que la durée propre de chaque ligne ; le contrôle
  modulo n'existe que dans la lane latest-observed (`:565-566`). Sondes A/B :
  index `… 01:00:00.500000+00:00` acceptés. Contredit
  `DATABRICKS-LT-MATERIALIZATION.md:111-112` (« sub-minute interval drift are
  rejected »). Correction : `np.diff(asi8) == 900_000_000_000` exact et
  contrôle `value % resolution_ns == 0` dans les deux helpers.
- **F-20 (MAJEUR) — l'étiquette sémantique n'est lue par aucun consommateur ;
  le validateur de snapshot v4 ne distingue pas un rejeu Gold-current d'un
  rejeu Silver-PIT.** `MATERIALIZATION_METADATA_ATTR` n'est lu nulle part hors
  du module (grep) ; `databricks_lt_replay.py:179` lie `mode` dans la config
  de rejeu, mais `databricks_lt_snapshot.py` ne l'inspecte jamais (seulement
  `export_mode == FULL_SNAPSHOT`, `:27/:301`) ; `_verify_export_manifest`
  (`:292-312`) et le contrôle de watermark (`:404-412`) sont satisfaits à
  l'identique par un rejeu `GOLD_ENTSOE_CURRENT`. Les trois lanes émettent des
  frames `entso` de schéma identique via `build_entso_features` (`:956`) et la
  grille perd la disponibilité par ligne (`:937-940`). Réponse à la question du
  point d'entrée : pas de réétiquetage dans le module, mais un consommateur
  peut obtenir de l'historique révisé sous le rôle `entso` en choisissant le
  mauvais mode, sans qu'aucun aval ne le remarque. Correction : exiger
  `replay_config["mode"] == "SILVER_ENTSOE_POINT_IN_TIME"` pour les rôles
  ENTSO-E éligibles à la calibration ; faire lire/enregistrer `mode` par
  `lt_input_replay._validate_raw_frame`.
- **F-21 (MINEUR) — le départage PIT utilise une information post-origine** :
  `order = ["AvailabilityTimestampUtc", "RevisionNumber", "LastObservedAtUtc"]`
  (`:753`) ; sonde : deux vintages à disponibilité et révision égales, celui
  vu pour la dernière fois **après** l'origine gagne. Le SQL PIT
  (`databricks_prd_entsoe_day_ahead_pit_extract.sql:52-58`) fait de même, alors
  que l'export causal v2 (`:71-76`) départage par `vintage_id` : deux
  classements « PIT » différents. Départager par `FirstObservedAtUtc`/`VintageID`.
- **F-22 (MINEUR) — classement latest-observed différent du SQL réalisé
  documenté** (`:589` vs `databricks_prd_entsoe_day_ahead_realized_export_v2.sql:76-83`) ;
  déterministes tous deux, mais divergents si une révision inférieure est
  ré-observée après une supérieure.
- **F-23 (MINEUR) — valeurs horaires répétées sur la grille 15 min sans le
  marqueur de provenance de résolution** : `:587` écrase `Resolution = "PT15M"` ;
  aucune lane ne pose `OBSERVATION_RESOLUTION_PROVENANCE_ATTR`
  (`governed_lt_acquisition.py:39`) utilisé par `lt_input_replay.py:183-187` et
  `lt_input_sources.py:1500-1504` pour `native_quarter_hour_truth_eligible` ;
  une frame `epex_ch` Databricks horaire-répétée passe donc la validation de
  rejeu sans refus de vérité 15 min. Exposition bornée (les scripts de panel
  échouent sur champ de manifeste manquant), mais à corriger ; noter aussi que
  `build_entso_features` donne un poids ×4 à chaque valeur horaire dans ses
  quantiles glissants.
- **F-24 (NOTE)** : ancrage CH faible pour charge/production (`"CH" in
  {FromZone, ToZone}`, `:862-866`) ; `dataframe_semantic_sha256` hache les
  flottants par motif binaire (`0.0 ≠ -0.0`, `-0.0` possible via `component *
  -1.0`, `:920`) ; `as_of_utc` latest-observed est une coupure d'évaluation,
  pas un état antérieur reconstructible (`:548`).
- **Couverture de tests** : 45 tests, bons cas couverts (fall-back 2026-10-25,
  supersession de blocs, µs, ambiguïté exacte) ; **non couverts** : passage à
  l'heure d'été et jour bissextile dans les lanes ENTSO-E, blocs A03 dans la
  lane PIT, décalage sub-seconde cohérent (F-19), epoch/naïf (F-18),
  `SOURCE_DOCUMENT_CREATED` en PIT (F-16), `publication > first_seen` PRD-like
  (F-17), départage post-origine (F-21).

## 6. Point 4 — collecteur quotidien, garde-fous et neuf conflits de cotes

Revue déléguée à un sous-agent (lecture de `scripts/collect_lt_benchmark_day.py`,
`run_lt_benchmark_day.py`, `register_lt_benchmark_snapshot.py`,
`pfc_shaping/validation/lt_benchmark_snapshots.py`, `lt_source_quality.py`,
`product_normalization.py`, `pfc_shaping/data/databricks_eex_daily_snapshot.py`,
`pfc_shaping/calibration/monthly_forward_curve.py`, `eex_contract_selection.py`,
`pfc_shaping/pipeline/monthly_curve_authority.py`, tests et handoffs), avec
démonstrations sur les vraies fonctions ; contre-vérifiée par sondage. Les cinq
modules de tests de la chaîne passent (70 tests). Le collecteur lui-même ne
peut pas s'exécuter ici (racine `C:\Users\jbattaglia\PFC_LT` codée en dur).

### 6.1 Ce qui est conforme (vérifié)

- **Date réelle** : `now = pd.Timestamp.now(tz='UTC')`, jour suisse dérivé
  (`collect_lt_benchmark_day.py:175-176`), fenêtre SQL J−7..J (`:206-207`),
  `as_of_date` suisse (`:211`) ; aucun argument de date dans les deux CLI ;
  rejet des faits chargés après observation (`:209-210`) et des dates de
  cotation > date de chargement (`databricks_eex_daily_snapshot.py:211-214`) ;
  chronologie observé ≤ valorisation ≤ commit ≤ enregistrement vérifiée
  (`lt_benchmark_snapshots.py:46-59`). La garde de doublon bascule à minuit
  **suisse** (démontré : `ALREADY_CAPTURED_TODAY` à 21:59Z, `DUE` à 22:30Z).
- **Garde de doublon** : trois contrôles indépendants (`due_day` avant
  identifiants, à nouveau sous verrou, `preflight` du builder exigeant un jour
  suisse strictement postérieur) plus création exclusive `'xb'` du fichier de
  registre `NNNN-YYYY-MM-DD.json` (`lt_benchmark_snapshots.py:96-108`).
- **Budget Warehouse** réellement appliqué dans `BoundedQueries`
  (`collect_lt_benchmark_day.py:96-137`) : ≤ 6 SELECT, `row_limit` sentinelle
  ≤ 30 000, 180 s par instruction, pagination bornée (20 chunks, liens de sa
  propre instruction), annulation sur échec, démarrage uniquement depuis
  STOPPED avec borne 600 s, auto-stop exigé dans (0, 45] avant toute action ;
  aucun appel stop/config/écriture ; exactement six SELECT dans `capture()`.
- **Tentatives immuables** : `write()` en mode `'x'`, sortie fraîche sous
  `build/`, manifeste calculé dans `finally` après `status.json`/`failure.json`,
  ré-hachage des entrées après construction.
- **Dates absentes d'une réponse** : préservées par `merge_observed_eex_dates`
  (`:166-171`), test de non-régression présent
  (`tests/test_collect_lt_benchmark_day.py:128-132`).
- **Gate d'audit des conflits** : `QUOTE_CONFLICT` n'est produit que par
  reclassification d'un CRITICAL dont le parent est entièrement couvert par des
  enfants qui passent (`product_normalization.py:820-930`) ; l'acceptation
  exige une politique signée Ed25519, liée au hash d'identité exact et
  `production_approved` (`:140-194`, `quote_conflict_policy_contract.py:50-76`) ;
  `reconstruct_quote_conflicts` (`lt_source_quality.py:25-90`) ne fait que
  classer (`accepted=False`, `vendor_rounding_confirmed=False`). Aucune
  hypothèse d'arrondi ni politique non signée n'accepte un conflit **à la
  gate**. Conforme au point d'entrée.

### 6.2 Findings du point 4

- **F-25 (MAJEUR) — une date de cotation renvoyée mais dont toutes les lignes
  CAL/Q/M sont mises en quarantaine efface silencieusement l'historique accepté
  de cette date.** `merge_observed_eex_dates` (`collect_lt_benchmark_day.py:166-171`)
  calcule `dates` sur la réponse **brute** (avant quarantaine, DAY/WEEK/WEEKEND
  compris), supprime les lignes sauvées sur ces dates, puis ajoute le sous-ensemble
  CAL/Q/M **après** quarantaine. La vacuité totale du delta est fail-closed
  (`databricks_eex_daily_snapshot.py:242,254`), la vacuité **par date** ne l'est
  pas. Démontré sur la vraie fonction : historique avec 2026-09-04 et
  2026-09-07, `normalized` vide, `quotation_dates=['20260907']` → 0 ligne
  restante pour 2026-09-07. Déclencheurs plausibles : correction d'une
  dimension entraînant `DELIVERY_BOUNDARY_MISMATCH` sur toutes les lignes d'une
  date, `QUOTATION_NOT_BEFORE_DELIVERY_START`, ou date présente uniquement via
  des produits courts. En aval, `select_latest_quote_surface` choisit alors la
  date précédente **sans trace** (`:354,367`), et le registre ne porte aucune
  date de cotation (F-27). C'est précisément le scénario « corrected collector
  history erasure on absent quote dates » de D313, mais pour le cas
  *présent-mais-quarantiné*. Correction : dériver les dates de remplacement de
  `normalized.history.date` ; pour toute date brute qui avait des lignes
  sauvées et n'en a plus, échouer (ou exiger un reçu explicite
  `saved_rows_removed`/`rows_added` par date) ; asserter que
  `max(raw.QuotationDateID)` égale la date de surface fusionnée.
- **F-26 (MAJEUR, précision d'affirmation) — « aucun conflit accepté » n'est
  vrai qu'à la gate d'audit ; le solveur et la projection PEAK tranchent
  chaque conflit en faveur des produits fins, sous une tolérance non signée de
  0.01 EUR/MWh.** `monthly_forward_curve.py:184-236` : priorité Month > Quarter
  > Calendar, parent entièrement couvert écarté avec
  `dropped_reason="redundant_consistent"` (`:235`) ; `_validate_all_quotes`
  (`:804-830`) ne lève que si `|implied − quote| > quote_conflict_tolerance`,
  défaut `0.01` (`:47` ; `monthly_curve_authority.py:68,119`). Démontré :
  écart parent 0.004 et 0.0099 → parent écarté « redundant_consistent » ;
  0.0101 → exception. La projection PEAK représente un parent PEAK couvert
  uniquement par ses enfants (`shape_constraints.py:373-380`) sans contrôle de
  cohérence du parent. La courbe applique donc les enfants à 1e-6 et les six
  parents seulement à ≤ 0.0041 EUR/MWh : **les parents sont les conflits, et
  ils perdent**. 0.01 est exactement le double de la demi-largeur d'arrondi au
  demi-centime, codé en dur ; il absorberait tout écart parent/enfant jusqu'à
  un centime quelle qu'en soit la cause. Le `solver.json` quotidien n'expose
  que `quote_diagnostics_hash` (`monthly_curve_authority.py:801`), pas les
  lignes écartées. Recommandation : écrire dans les handoffs que « non accepté »
  vise la gate d'audit tandis que le solveur applique priorité + 0.01 ;
  sortir les lignes `quote_diagnostics` dans `solver.json` ; renommer
  `redundant_consistent` en `redundant_within_tolerance` avec l'écart ; et si
  l'intention est « aucune partie ne gagne », abaisser la tolérance du solveur
  à 1e-6 pour le pilote et traiter la divergence parent/enfant comme un échec
  de source.
- **F-27 (MINEUR) — le registre ne porte aucune date de cotation EEX** :
  `('EEX', out/'EEX.parquet', eex_seen, None)` (`collect_lt_benchmark_day.py:232`)
  → `issue_at_utc = None` ; le schéma d'entrée (`run_lt_benchmark_day.py:103-106`,
  `lt_benchmark_snapshots.py:39-40`) n'a pas de champ de date de cotation ;
  `eex-normalization.json` ne décrit que le delta. Écrire `max(raw.QuotationDateID)`,
  la date de surface du delta et celle de la surface fusionnée dans
  `request.json`, et les asserter égales dans `preflight`.
- **F-28 (MINEUR) — le `solver.json` quotidien étiquette de vraies cotes EEX
  comme fixture de test** : `run_lt_benchmark_day.py:86-88` appelle
  `solve_monthly_level_authority(..., allow_unverified_inputs=True)` sans
  `own_forward_snapshot`, d'où `source_kind = "TEST_FIXTURE"`
  (`monthly_curve_authority.py:655-668`) et `hard_quotes=[]`. Passer une
  provenance explicite (`DATABRICKS_PRD_GOLD`, `source_snapshot_sha256`).
- **F-29 (MINEUR) — champ de reçu codé en dur et trompeur** :
  `manual_stop='PRIOR_HTTP403_NOT_RETRIED'` est écrit à chaque run
  (`collect_lt_benchmark_day.py:141-143`) alors que ce collecteur n'émet jamais
  de stop ; remplacer par `stop_requests=0`, `manual_stop_attempted=False`.
- **F-30 (MINEUR) — `finish()` dans `finally` peut masquer l'échec primaire**
  (`:227-228`) : si le GET final lève, `failure.json` enregistre cette
  exception au lieu de celle de la capture.
- **F-31 (MINEUR) — les échecs précoces ne laissent aucune trace durable** :
  tout ce qui échoue entre `:247` et `:268` (cwd, sortie, registre illisible,
  schéma de config, identifiants, Warehouse différent) précède le `try` de
  `:271` ; ni `failure.json` ni `manifest.json`. Ouvrir le `try/finally` juste
  après `output.mkdir`.
- **F-32 (MINEUR) — base d'historique du jour 2 asymétrique, non liée, non
  testée** : `history_ref = records[-1]…EEX.artifact if len(records) > 1 else
  config['initial_eex_history']` (`:180`) ; au jour 2 la base est le fichier de
  config, pas l'artefact du jour 1, et `initial_eex_history` n'est jamais lié à
  `records[0]…EEX.artifact` ; `:182` n'exige qu'une colonne `date`. Aucun test
  n'exerce `capture()`. Correction : `records[-1] if records else config[...]`,
  égalité fail-closed config/record 0001, contrôle d'égalité des colonnes, test.
- **F-33 (NOTE)** : verrou sans identité de propriétaire (PID/hôte) et non pris
  par les chemins manuels `run_lt_benchmark_day`/`register_lt_benchmark_snapshot`
  (`'xb'` seul) ; `read_registry` ré-hache tous les artefacts de tous les
  enregistrements à chaque appel (fail-closed mais fragile) ; dépassement
  possible des délais de ~33 s (sommeil 3 s + HTTP 30 s après le test) ; une
  panne de plus de sept jours laisse des dates de cotation définitivement
  absentes de la fenêtre J−7.

### 6.3 Les neuf conflits tels que documentés

Identités reconstituées depuis `SESSION-HANDOFF-20260907-LOCAL-PFC-SOURCE-INTEGRATION.md`
(cotes du 4 septembre) et les handoffs D312/D313 (cotes du 7 septembre : 6
directs / 3 dérivés, résidu direct max 0.0040625, OFFPEAK max 0.0045363). Les
identités exactes du 7 septembre sont dans `build/…/conflict-register`, absent
ici. Les enfants sont déduits de `calibration_buckets`
(`eex_contract_selection.py:71-121`), pas lus dans un artefact.

| # | Parent | Charge | En conflit avec (enfants qui passent) | Gate | Ampleur documentée | Politique d'audit | Comportement solveur |
|---|---|---|---|---|---|---|---|
| 1 | 2026-Q4 | BASE | 2026-10, 2026-11, 2026-12 | hard_base_product_repricing | ≤ 0.0041 | non accepté, bloquant | parent écarté `redundant_consistent` |
| 2 | 2026-Q4 | PEAK | mêmes mois, PEAK | hard_peak_product_repricing | ≤ 0.0041 | non accepté | parent représenté via enfants |
| 3 | 2027-Q1 | BASE | 2027-01, 02, 03 | hard_base_product_repricing | ≤ 0.0041 | non accepté | écarté |
| 4 | 2027-Q1 | PEAK | mêmes mois, PEAK | hard_peak_product_repricing | ≤ 0.0041 | non accepté | via enfants |
| 5 | 2027 (CAL) | BASE | 2027-01..03, 2027-Q2, Q3, Q4 | hard_base_product_repricing | ≤ 0.0041 | non accepté | écarté |
| 6 | 2027 (CAL) | PEAK | idem, PEAK | hard_peak_product_repricing | ≤ 0.0041 | non accepté | via enfants |
| 7 | 2026-Q4 | OFFPEAK | impliqué par 1/2 | implied_offpeak_identity | ≤ 0.0045 (0.00584 le 4 sept.) | non accepté | conséquence |
| 8 | 2027-Q1 | OFFPEAK | impliqué par 3/4 | implied_offpeak_identity | ≤ 0.0045 | non accepté | conséquence |
| 9 | 2027 (CAL) | OFFPEAK | impliqué par 5/6 | implied_offpeak_identity | ≤ 0.0045 | non accepté | conséquence |

Verdict de cohérence : la politique de **gate** est bien « aucun conflit
accepté » ; la politique **solveur/projection** n'est pas « aucune partie ne
gagne » (F-26).

## 7. Affirmations non supportées ou à requalifier

| Affirmation (source) | Statut après audit |
|---|---|
| « 591 passed / 0 failed / 5 skipped » (`PFC-CH-AUDIT-CHECKPOINT-VERIFICATION-20260908.json`) | Cohérent (576 + 1 échec local + 14 skips Windows sous Linux) ; non reproductible à l'identique hors du poste local (F-13). |
| « All nine conflicts remain visible and none is accepted » (D313, D315) | Vrai à la gate d'audit ; **incomplet** : le solveur écarte les six parents comme `redundant_consistent` sous une tolérance non signée de 0.01 EUR/MWh et la courbe applique les enfants (F-26). |
| « Corrected collector history erasure on absent quote dates » (D313) | Vrai pour les dates **absentes** ; le cas « date présente mais entièrement quarantainée » efface encore (F-25). |
| « Silver vintages are the PIT authority » / « Stage 2 implemented end-to-end for Gold/Silver ENTSO-E » (README, `DATABRICKS-LT-SNAPSHOT-INTAKE.md:45-46`) | Non supporté par des données réelles : la lane PIT est validée sur fixtures synthétiques et rejetterait les lignes PRD documentées (F-17). |
| « sub-minute interval drift are rejected » (`DATABRICKS-LT-MATERIALIZATION.md:111-112`) | Faux pour un décalage sub-seconde cohérent (F-19). |
| « Protocol fixed before results » (docs D307–D310) | Non vérifiable depuis Git : scripts, docs et handoffs sont entrés dans le seul commit `dee652b` ; seul le hash de doc épinglé dans `plan.json` (local) le lie. |
| MAE 20.345234 vs 22.817767, gains 10.836 %/10.191 % (D304) ; 1.737 %/3.199 % (D307) ; 0.93–1.40 % (D308) ; −5.447 % (D310) ; ratios 0.962/0.798 (D309) | Formules et arithmétique cohérentes avec le code ; **valeurs non rejouées** (artefacts `build/` absents). |
| « signed round-trip hourly error ≤ 6.2528e-13 » (D304) | Contrôle attendu mathématiquement (corrections KKT constantes par heure) ; valide l'intégration, pas la compétence. |
| « Four regime regressions » (D307), « 2 / 2 » (D308) | Comptage par ligne conforme au code des vérificateurs ; mais les régimes à origine unique ne peuvent jamais compter (F-05). |
| Reçu `manual_stop='PRIOR_HTTP403_NOT_RETRIED'` de chaque run du collecteur | Littéral codé en dur, pas une observation (F-29). |
| `solver.json` quotidien : `forward_source_kind = TEST_FIXTURE` | Étiquette fausse pour de vraies cotes PRD (F-28). |
| « No scheduler exists », « pilot 1/20 », « all six authorities false » | Confirmé par le code : aucune tâche planifiée dans le dépôt, `AUTHORITIES` tous `False` (`local_benchmark.py:40-41`), `CurveAssemblyAuthority` non surchargeable. |

## 8. Bloqueurs production versus travail exploratoire

**Bloqueurs production (à lever avant toute admission ; aucun n'est levé par
ce checkpoint, et le dépôt ne prétend pas le contraire)** :

1. Vérité PIT : lane PIT Silver non validée sur PRD, avec deux défauts fail-open
   (F-16, F-18, F-19) et une étiquette non consommée (F-20). Tant que cela
   n'est pas corrigé, aucune rétro-validation « historique » ne peut être
   qualifiée de PIT.
2. Intégrité de l'historique EEX du pilote quotidien (F-25) et traçabilité de
   la date de cotation dans le registre (F-27) : à corriger **avant le jour 2
   du pilote**, sinon les 19 captures restantes peuvent hériter d'une base
   erronée sans trace.
3. Politique des conflits de cotes : documenter et signer explicitement la
   règle « priorité aux produits fins sous 0.01 EUR/MWh » ou la ramener à 1e-6
   (F-26). En l'état, l'affirmation « aucun conflit accepté » est incomplète.
4. Holdout futur non enregistré, custody de la vérité absente, un seul jour de
   pilote, aucun ordonnanceur : inchangés ; voir §9.4 pour un témoin public
   immédiat.
5. CI : `publisher-runtime-v6` rouge sur la PR pour une raison de configuration
   (F-14) ; à corriger pour que la CI redevienne un signal.

**Travail exploratoire localement autorisé, correctement borné** : D304–D310
(représentation signée, composition, intraday conditionné, récence, blends,
audit des révisions, maturité) et D311–D312 (distances externes descriptives,
premier snapshot). Leur code respecte les invariants (niveau solveur, pas de
sélection par mois, populations communes, gates gelées) ; leurs limites sont
déclarées (années exposées, multiplicité). Ils peuvent continuer sans attendre
les bloqueurs 1–5, à condition d'appliquer les corrections mineures F-01,
F-05, F-06, F-10 avant le prochain lot.

## 9. Recommandations de modélisation (état de l'art 2026, PFC horaire CH)

Cadre : l'architecture actuelle (niveau mensuel = solveur EEX, forme = couches à
moyenne mensuelle nulle, projection dure BASE/PEAK) est la bonne architecture
de HPFC ; l'état de l'art 2026 ne la remplace pas, il l'enrichit sur la forme.
Les recommandations ci-dessous respectent les invariants d'AGENTS.md (pas de
patch mensuel, pas de sélection par mois, protocole gelé avant résultats).

### 9.1 Représentation : forme signée **conditionnée au niveau** (priorité 1)

Le candidat D304 est une climatologie EUR/MWh indépendante du niveau (F-03).
Le ratio (multiplicatif) casse près de zéro et sous zéro. La solution standard
est intermédiaire : par cellule calendaire `c` (saison, type de jour, heure),
`shape_h = a_c + b_c · (B_m − B_ref)` où `B_m` est le niveau BASE mensuel du
solveur (connu à l'origine, donc sans fuite) et `B_ref` une constante gelée
(médiane des niveaux d'entraînement). En apprentissage, `B_m` est la moyenne
mensuelle réalisée du mois fermé ; en prédiction, le niveau solveur. Recentrage
mensuel exact ensuite (la lane signée existante l'exige déjà). Effet attendu :
amplitude cohérente avec le régime de prix, prix négatifs préservés, et
disparition du compromis « récence contre années 2024 » de D307/D308.
Test frozen : mêmes six origines, mêmes populations, mêmes gates 2 %/5 % ; un
seul recipe global ; ablation `b_c = 0` = D304 exactement (contrôle exact).

### 9.2 Dérive structurelle photovoltaïque (priorité 2)

Une climatologie calendaire retarde structurellement l'approfondissement du
creux de midi (duck curve). Ajouter un terme à moyenne mensuelle nulle piloté
par la capacité PV installée **connue à l'origine** (statistiques OFEN/Swissolar
et Pronovo publiées avant l'origine ; trajectoire future gelée à l'origine, par
exemple la trajectoire de la loi sur l'électricité, 35 TWh en 2035), estimé
comme une pente par cellule calendaire sur l'historique. C'est la première
brique concrète du contrat D302, sans modèle de dispatch complet.

### 9.3 Vérité PIT : distinguer prix et covariables

Le dépôt qualifie toute l'historique de « latest-observed, not PIT ». Pour les
**prix day-ahead** EPEX/ENTSO-E, la valeur est définitive à la publication
(vers 13:00 J−1) et n'est pratiquement jamais révisée : la vérité prix est de
fait PIT dès que l'horodatage de première observation est ≥ publication. Le
vrai risque PIT concerne le remplissage hydro OFEN (révisé), les actuals
ENTSO-E charge/production (révisés) et les corrections de règlement EEX. Le
protocole gagnerait à formaliser cette distinction : cela lève une réserve
excessive sur les scores de forme et concentre l'effort là où il compte.

### 9.4 Holdout futur : témoin public déjà disponible

Le commit public `dee652bc91` contient le SHA-256 de la courbe D312
(`13bca031…`) dans `SESSION-HANDOFF-20260908-POST-D312-QUALITY-ROADMAP.md`.
L'horodatage GitHub de ce commit constitue un témoin tiers de l'engagement
prospectif Oct 2026 → Sep 2027, à condition de conserver les octets localement
et de les divulguer au moment du scoring. Recommandation : formaliser ce
mécanisme (fichier `commitments/` avec hash + fenêtre + règle de vérité,
poussé avant le début de la livraison) plutôt que d'attendre un « custodian »
externe qui n'existe pas encore. Cela ne remplace pas la custody de la vérité,
mais rend le holdout inaltérable.

### 9.5 Ce qu'il ne faut pas faire ensuite

- Ne pas continuer à élargir les grilles fixes (demi-vies, blends, résidu de
  maturité ridge) : D307–D310 montrent que ce n'est pas la bonne classe
  d'hypothèse ; le dépôt l'a correctement conclu.
- Ne pas engager Chronos-2 / TiRex-2 / TabPFN-TS sur la forme à 36 mois : ce
  sont des modèles d'horizon court (≤ 1024 pas), avec un risque de contamination
  des années 2023–2025 par leurs corpus publics ; le handoff SOTA d'août le dit
  déjà. Un usage borné reste envisageable pour l'overlay CT.
- Ne pas laisser l'information externe (OMPEX, LSEG) entrer autrement que comme
  distance descriptive (D311/D312) : bon choix, à maintenir.

### 9.6 Quart d'heure et incertitude

CH day-ahead reste horaire (doc `CH-DAY-AHEAD-RESOLUTION-20260908.md`) : garder
le transport ×4 comme représentation officielle, et préparer sur données
natives DE-LU/AT un résidu quart-horaire **conditionné aux rampes solaires**
(pas seulement au prix-mère), à activer uniquement à la date d'entrée en
vigueur du 15 min CH. L'incertitude doit rester un produit distinct : scénarios
joints (années météo × fondamentaux) recentrés sur les niveaux solveur, scorés
CRPS/WIS **après** projection, comme le prévoit la feuille de route post-D312.

## 10. Lignes, interconnexions et base d'événements structurels

### 10.0 Convergence avec la proposition Codex (`e8313f771c`)

`docs/data/CH-STRUCTURAL-EVENT-REGISTRY-PROPOSAL.md` et
`docs/data/CH-STRUCTURAL-EVENT-WATCHLIST-20260908.json` (D316) proposent déjà un
registre versionné avec quatre entrées (Bickigen–Chippis, Beznau–Tiengen,
déphaseurs romands, câble du Gothard), la séparation « date où l'information
est connue » / « date d'effet », l'interdiction d'avancer une mise en service
dans l'horizon, la distinction ACER capacité physique / commerciale, et la
règle « niveau au solveur, forme additive recentrée puis projetée ». **J'y
souscris intégralement** ; les points ci-dessous ne contredisent rien, ils
complètent :

1. **Le pilote dominant manque dans la watchlist** : l'accord sur l'électricité
   Suisse–UE (couplage SDAC/SIDC, flow-based, fin de la pénalité des 70 %) et
   le passage au 15 min MTU aux frontières (JAO, 2027). Ce sont des événements
   de *règles de marché*, pas d'ouvrages, mais ils changent la capacité
   commerciale bien plus que n'importe quelle ligne du programme 2040 dans
   l'horizon 2027–2029. Ajouter une catégorie `market_coupling_rule` avec les
   jalons parlementaires datés (§10.1) et un scénario « pas d'accord ».
2. **Deux événements de parc à horizon compatible** : arrêt de Beznau 2 (2032)
   et Beznau 1 (2033) — hors 2026–2029 mais dans l'horizon des origines ≥ 2029
   — et la trajectoire PV (loi sur l'électricité), qui est le seul pilote
   structurel déjà mesurable historiquement (§9.2).
3. **Première implémentation testable** de la conversion « capacité commerciale
   → forme » : une régression de la forme signée historique sur les capacités
   commerciales *observées* (NTC/ATC JAO et ENTSO-E, familles déjà identifiées
   dans `CH-NATIONAL-INPUT-READINESS-20260908.md`) et la PV installée, plutôt
   qu'un modèle de flux ; c'est la seule voie qui produit une preuve empirique
   avant 2028. Le petit modèle de formation des prix vient ensuite, pour les
   régimes non observés (couplage complet, 15 min).
4. **Hygiène PIT de la watchlist** : les quatre entrées ont
   `first_archived_source_capture_utc: null` et `source_snapshot_sha256: null` ;
   avant toute admission, archiver les octets sources (comme pour les
   inventaires D302), et faire de `recorded_at_utc` la date « connu à
   l'origine » utilisée par les rétro-validations.
5. **Cadence** : hebdomadaire pour les projets et quotidienne pour les
   indisponibilités structurées, comme proposé ; j'ajoute une revue à chaque
   jalon parlementaire de l'accord UE et à chaque publication TYNDP/plan
   régional, car ce sont des changements discrets, pas des dérives lentes.


### 10.1 Projets de lignes et d'interconnexions : état daté (sources publiques)

- **Réseau stratégique 2040 (Swissgrid)** : 31 projets, ≈ 5,5 Mrd CHF, 1 300 km
  de lignes optimisées, 400 km de montée en tension, 790 km de constructions
  nouvelles/démontages, 21 transformateurs-déphaseurs ; objectif déclaré :
  « increase exchange capacities with other countries, reduce redispatching
  costs » ([page Swissgrid](https://www.swissgrid.ch/en/home/projects/future-grid/grid-development-requirements.html)).
- **Bickigen–Chippis 220 → 380 kV (106 km, projet 9)** : approbation OFEN
  février 2022, Tribunal administratif fédéral janvier 2024 (renvoi bruit),
  nouvelle approbation OFEN septembre 2025, **nouveaux recours pendants** ;
  mise en service initialement 2031. Sans cette montée en tension, « seuls
  environ deux tiers » de la future production hydraulique valaisanne
  pourraient être évacués ([Swissgrid](https://www.swissgrid.ch/en/home/projects/project-overview/bickigen-chippis.html)).
- **Airolo–Lavorgo 380 kV (23 km)** : procédure suspendue début 2024, dossier
  redéposé automne 2025, republication attendue au premier semestre 2026 ;
  deuxième axe 380 kV vers le Tessin ([Swissgrid](https://www.swissgrid.ch/en/home/projects/project-overview/airolo-lavorgo.html)).
- **Mörel–Lavorgo (Nufenen, projet 23)** et **câble 220 kV du tunnel routier du
  Gothard (projet 22)** : axe nord-sud Allemagne → Suisse centrale → Tessin/Italie.
- **Greenconnector (HVDC Italie–Suisse, 1 000 MW, ±400 kV, ~165 km en câble)** :
  projet d'intérêt commun européen ; **statut 2026 non confirmé** par mes
  sources ([fiche projet](http://www.greenconnector.it/en/about.html),
  [CESI](https://www.cesi.it/projects/switzerland-italy-interconnector-gf/)).
- **Accord sur l'électricité Suisse–UE** : signé le 2 mars 2026, message du
  Conseil fédéral au Parlement le 13 mars 2026, Conseil des États attendu à la
  session d'automne 2026, entrée en vigueur envisagée « à l'horizon 2028–2030 »
  sous réserve du Parlement et d'un éventuel référendum
  ([admin.ch](https://www.admin.ch/fr/paquet-suisse-ue),
  [DETEC](https://www.uvek.admin.ch/fr/negociations-suisse-ue),
  [AES](https://www.strom.ch/fr/grands-axes/accord-sur-lelectricite)). Sans
  accord, la règle des 70 % du paquet « énergie propre » réduit la capacité
  disponible aux frontières suisses ; l'ElCom estime que les capacités
  d'importation pourraient chuter de plus de 50 % dans certains scénarios
  ([EY, citant ElCom](https://www.ey.com/en_ch/insights/energy-resources/how-eu-power-agreement-boosts-swiss-supply-security)).
- **15 min MTU aux frontières suisses** : reporté à 2027 par JAO (déjà tracé
  dans `docs/data/CH-DAY-AHEAD-RESOLUTION-20260908.md`).
- **Nucléaire** : Axpo a annoncé le 4 décembre 2024 l'arrêt définitif de
  Beznau 2 en 2032 et Beznau 1 en 2033
  ([Axpo](https://www.axpo.com/ch/en/newsroom/media-releases/2024/axpo-will-operate-the-beznau-nuclear-power-plant-until-2033--inv.html)).
  Hors horizon N+3 aujourd'hui, mais dans l'horizon des origines ≥ 2029.

### 10.2 Influence sur les prix suisses

Oui, mais par des canaux différents qu'il faut séparer :

1. **Capacité transfrontalière et couplage** (accord UE, flow-based, 15 min,
   nouvelles interconnexions) : c'est le facteur dominant. Plus de capacité
   d'échange rapproche le prix CH de ses voisins heure par heure : en hiver
   (import), la prime CH sur DE-LU/FR se comprime ; en été/mi-journée
   (export vers l'Italie), les heures de pointe suivent davantage l'Italie ; le
   creux solaire de midi importé d'Allemagne se creuse. Le **niveau** mensuel
   est déjà arbitré par les forwards EEX CH (le marché intègre l'anticipation
   de l'accord), la **forme horaire** ne l'est pas explicitement.
2. **Renforcements internes** (Valais → Plateau, axe Gothard) : ils lèvent des
   contraintes d'évacuation hydraulique et du redispatch. La Suisse étant une
   zone de prix unique, l'effet sur le prix day-ahead est indirect (moins de
   redispatch, meilleure valeur de capture hydro), donc secondaire pour la
   forme de la PFC, mais important pour la valorisation FMV des actifs.
3. **Événements de parc** (Beznau 2032/33, PV) : effet de forme (base de
   nuit/hiver, creux de midi) et effet de niveau, ce dernier restant du
   ressort des forwards.

### 10.3 Comment la PFC doit intégrer ce type d'information

- **Jamais au niveau mensuel** : l'invariant « solveur = seule autorité de
  niveau » est correct. Les forwards EEX CH contiennent déjà l'anticipation du
  marché ; superposer une vue fondamentale doublerait l'information et casserait
  l'absence d'arbitrage. Une vue fondamentale du niveau est un **autre produit**
  (scénarios), pas la PFC.
- **Sur la forme, à moyenne mensuelle nulle, et connue à l'origine** : chaque
  événement entre comme un pilote structurel daté (capacité d'échange effective
  par frontière, capacité PV, capacité nucléaire) dans la couche « structural
  shaping » du contrat D302. Deux implémentations compatibles avec le dépôt :
  (a) régression empirique de la forme signée historique sur ces pilotes
  (NTC/capacités disponibles historiques JAO, PV installée), puis extrapolation
  avec la trajectoire gelée à l'origine ; (b) plus tard, un petit modèle de
  formation des prix (dispatch réduit CH + voisins) pour les régimes non
  observés historiquement (couplage complet, 15 min). Dans les deux cas, la
  sortie est recentrée et projetée par l'assembleur existant ; aucun patch
  de mois.
- **Testabilité** : un pilote n'est admissible que s'il a une date « connu à
  l'origine » ; sinon la rétro-validation est contaminée. C'est le même
  principe que pour l'hydro.

### 10.4 Faut-il une base d'événements ? Oui, sous forme de registre gouverné

Pas un flux de nouvelles, mais un **registre structurel versionné**, cohérent
avec la discipline du dépôt (sources épinglées par hash, statuts explicites) :

| Champ | Contenu |
|---|---|
| `id`, `type` | ligne interne, interconnexion, couplage/régulation, centrale, demande |
| `geography`, `border` | CH, CH-DE, CH-FR, CH-IT, CH-AT |
| `capacity_mw`, `direction` | valeur et sens (import/export/bidirectionnel) |
| `status` | annoncé, autorisé, en recours, en construction, en service, retardé, annulé |
| `dates` | annonce, mise en service prévue **avec fenêtre**, mise en service réelle |
| `known_at_utc` | date PIT à partir de laquelle l'entrée est utilisable par une origine |
| `source_url`, `retrieved_at`, `sha256` | preuve, comme pour les inventaires D302 |
| `channel`, `expected_shape_effect` | NTC, couplage, PV, nucléaire ; signe attendu par cellule (hiver/nuit, midi, pointe) |
| `used_by` | version de modèle/recette qui consomme l'entrée |

Veille associée (revue trimestrielle, chaque entrée datée) : rapports réseau
Swissgrid et procédures OFEN/TAF, communiqués JAO/ENTSO-E (TYNDP bisannuel,
plans régionaux), ElCom (adéquation), étapes parlementaires de l'accord UE
(sessions d'automne/hiver 2026, référendum éventuel), décisions Axpo/Alpiq sur
le parc, notices EPEX/SDAC. Deux règles : une entrée n'est jamais une
probabilité déguisée (on garde des scénarios de dates), et aucune entrée ne
sert à ajuster un mois après coup.

## Annexe A — commandes exactes

```text
git fetch origin fix/lt-audit-remediation && git checkout -B claude/pfc-ch-audit-15kn4o dee652bc91
uv venv <scratch>/venv --python 3.11
uv pip install -e ".[test,validation,ingest]" lightgbm==4.6.0 duckdb
python -m pytest -q -p no:cacheprovider --basetemp=<scratch>/pytest-tmp $(python -c "import json;print(' '.join(json.load(open('.planning/phases/14-lt-audit-remediation/PFC-CH-AUDIT-CHECKPOINT-VERIFICATION-20260908.json'))['test_files']))")
  -> 1 failed, 576 passed, 19 skipped in 23.54s
python -m pytest -q -p no:cacheprovider -rfE --tb=line --basetemp=<scratch>/pytest-full2 tests
  -> 220 failed, 4638 passed, 65 skipped, 73 errors in 599.76s
python -m pytest -q -p no:cacheprovider tests/test_snapshot_publisher_runtime_closure.py tests/test_snapshot_publisher_container_contract.py
  -> 8 failed, 21 passed (échecs « publisher runtime file is linked or non-regular » : le python du venv de l'audit est un lien symbolique ; propre à l'environnement)
ruff check $(git diff --name-only dee652bc91~1 dee652bc91 -- 'pfc_shaping/*.py' 'scripts/*.py' 'tests/*.py')
  -> Found 264 errors (170 E701, 34 E402, 31 I001, 25 E702, 4 F401)
GitHub API : PR #3, runs 34228620940 (lt-model, success) et 34228620773 (publisher-runtime-v6, failure), job 102068831266 (ModuleNotFoundError: pandas)
```

Sous-agents (lecture seule, sondes sur les vraies fonctions, sans modification
du dépôt) : D307–D310 (61 tests des quatre modules passés), collecteur (70
tests des cinq modules passés), matérialisation (45 tests passés, 13 sondes).
Aucun artefact `build/` n'a été rejoué ; aucun appel Databricks, Warehouse,
AFRY ou T057 ; aucune donnée cliente consultée.
