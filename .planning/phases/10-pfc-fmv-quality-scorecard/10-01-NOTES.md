# Plan 10-01 — Notes empiriques

Document de décision pour Plan 10-01 (Phase 10 PFC FMV Quality Scorecard).
Capture les 2 résolutions empiriques requises par le RESEARCH (Pitfall 1
holiday/weekend ratio threshold + Q2 forwards historiques as-of vintage)
et les 2 sections REVIEWS bakées surgicalement (C2 recalibration frozen
ex-ante + C3 forwards_source structured flag convention).

**Convention d'ordre d'écriture (audit-trail) :** la section C2 REVIEWS
ci-dessous est rédigée AVANT toute mesure empirique de Sub-step 3b, dans
un commit séparé. Le but : interdire mécaniquement tout tweak du seuil
Hildmann Pillar 1.2 post-mesure. Cf. REVIEWS row 2 (HIGH severity).

---

## C2 REVIEWS — Recalibration Method (FROZEN EX-ANTE)

**Codex review (HIGH severity, REVIEWS row 2) :** *"Pitfall 1 = holiday/weekend
ratio politiquement sensible si recalibré ad-hoc post-mesure."*

**Mitigation = figer la règle de recalibration AVANT de regarder les chiffres.**
Cette section est écrite AVANT Sub-step 3b (mesure empirique), dans un commit
séparé. Toute application post-mesure consiste à exécuter mécaniquement la
formule ci-dessous — aucun tweak autorisé.

### Formule unique de recalibration

À appliquer ssi le ratio empirique 2019-2023 sort de la fenêtre
[0.65, 0.95] définie par la SOTA literature (Hildmann 2013 / CH typical) :

```
IF empirical_ratio ∈ [0.65, 0.95]:
    threshold := (0.65, 0.95)        # research default confirmé empiriquement
ELSE:
    threshold := (max(0.50, P10_monthly_ratios),
                  min(0.99, P90_monthly_ratios))
    # data-driven envelope sur les ratios mensuels bootstrap 2019-2023
    # où P10 / P90 sont les 10e / 90e percentiles des 60 ratios MENSUELS
    # (5 ans × 12 mois) calculés sur 2019-2023.
```

### Scope

- **Applicable UNIQUEMENT à la mesure 2019-2023 holidays_VS** (CH/VS,
  `holidays.country_holidays("CH", subdiv="VS", years=range(2019, 2024))`).
- Tout nouveau scope (autre période, autre subdiv, autre métrique) nécessite
  un amendement plan SIGNÉ PAR L'UTILISATEUR — pas de "tweak silencieux".

### Trigger condition

Déclenché ssi `empirical_ratio ∉ [0.65, 0.95]`. **Aucun autre trigger autorisé :**

- ❌ "le ratio paraît trop bas / trop haut intuitivement" → reject
- ❌ "exclude PV hours feels right" → reject
- ❌ "on a remarqué que 2020 COVID biaise le sample" → reject (le data-driven
   envelope absorbe ce biais via les percentiles mensuels)

Toute déviation = BLOCKER Plan 10-01 nécessitant ré-ouverture user (commit
explicit `chore(10-01): re-open recalibration method amendment`).

### Forbidden patterns (énumérés explicitement pour éviter les regrets post-hoc)

- ❌ *"On exclut les heures solaires bowl pour gonfler artificiellement le
  ratio numérateur"* (Option 2 KYOS variant) → **REJECT EX-ANTE**
- ❌ *"On élargit la fenêtre à [0.40, 0.99] parce que ça arrange le verdict
  SC#1"* → **REJECT EX-ANTE**
- ❌ *"On utilise un subset 2021-2023 plus court pour éviter le COVID 2020"*
  → **REJECT EX-ANTE** (la formule data-driven envelope absorbe la variance
  inter-annuelle via les percentiles mensuels)
- ❌ *"On switche de holidays VS vers holidays CH-national parce que ça donne
  un meilleur ratio"* → **REJECT EX-ANTE** (subdiv VS est convention fixe
  pour FMV depuis Phase 5bis-B)

Seule la formule ci-dessus s'applique. Si elle donne un threshold
"inconfortable", c'est un signal qu'il faut investiguer le bowl OU
ré-ouvrir la phase, **pas tweaker le seuil**.

### Audit-trail convention

L'application de la formule (cas par cas IF/ELSE) doit être écrite VERBATIM
dans Sub-step 3b ci-dessous avec :

1. La valeur numérique de `empirical_ratio` (4 décimales).
2. La branche prise (`IF` / `ELSE`) explicitement nommée.
3. Si branche `ELSE` : les valeurs numériques de `P10_monthly_ratios` et
   `P90_monthly_ratios` (60 ratios mensuels sur 2019-2023).
4. Le `threshold` final retenu sous forme `(low, high)`.

Toute déviation vs cette formule = BLOCKER (commit `git revert` immédiat).

---

<!-- ## RESEARCH Pitfall 1 — Hildmann holiday/weekend ratio threshold (Pillar 1.2)         -->
<!-- Section ci-dessous écrite par Sub-step 3b APRÈS bootstrap EPEX et mesure empirique.    -->
<!-- Format imposé par la section "Audit-trail convention" ci-dessus.                       -->

## RESEARCH Pitfall 1 — Hildmann holiday/weekend ratio threshold (Pillar 1.2)

**Measurement (EPEX CH 15-min, source `energy-charts.info`, période strict
`2019-01-01..2023-12-31` exclusif, holidays subdiv VS via
`holidays.country_holidays("CH", subdiv="VS", years=range(2019, 2024))`) :**

- Sample size : `175 296` rows 15-min (≈ 5 × 35 040)
- VS holidays uniques 2019-2023 : `45` dates
- N(weekend ∪ holidays_VS) = `53 276` rows
- N(weekday hors holidays_VS) = `122 020` rows
- mean(price | weekend ∪ holidays_VS) = `98.8969 €/MWh`
- mean(price | weekday hors holidays) = `123.1085 €/MWh`
- **ratio empirique = 0.8033** (4 décimales)

**Ratios mensuels (60 = 5y × 12m, tous computables, agrégat 15-min) :**

| Quantile | Valeur |
|----------|--------|
| min      | 0.5132 |
| P10      | 0.6916 |
| P50      | 0.8002 |
| P90      | 0.8850 |
| max      | 0.9402 |

**Threshold decision Pillar 1.2 (Plan 10-02 SC#1 gate) — APPLICATION
MÉCANIQUE de la formule frozen Sub-step 3a-bis [C2 REVIEWS] :**

```
IF empirical_ratio ∈ [0.65, 0.95]: threshold := (0.65, 0.95)  # research default confirmé
ELSE: threshold := (max(0.50, P10_monthly_ratios), min(0.99, P90_monthly_ratios))
```

- **empirical_ratio = 0.8033**
- **0.8033 ∈ [0.65, 0.95]** → vrai
- **Branch retenue : IF**
- **threshold final Pillar 1.2 retenu = (0.65, 0.95)** (research default
  confirmé empiriquement sur 2019-2023 CH/VS)

Aucune déviation vs la formule. Aucune mention de "Option 2 KYOS" ou
autre forbidden pattern. Audit-trail clean : la branche `IF` est prise
mécaniquement parce que `0.65 ≤ 0.8033 ≤ 0.95`.

**Sanity-check qualitatif :** 0.8033 est compatible avec la SOTA literature
CH (Hildmann 2013 reporte 0.75-0.85 pour le marché suisse) ; le P50 monthly
0.8002 confirme que la moyenne agrégée n'est pas distordue par un mois
outlier ; le P90 0.8850 reste sous le seuil supérieur 0.95 (no overlap
weekend-weekday). Convention adoptée Plan 10-02.

---

## RESEARCH Q2 — Forwards historiques as-of vintage (Mac Mini)

**Test accès `H:\\Energy\\GeCom\\MARCHE & NEGOCE\\Prix\\EEX - ER\\Price_Report_EEX.xlsx` :
FAIL** (cas attendu sur Mac Mini Sion ; le share réseau Windows H:\\ n'est
pas monté sur macOS).

**Path retenu (Mac Mini default) : fallback `derive_forwards_from_epex_hist`**
(body implémenté Plan 10-01 Task 3 sub-step 3c dans
`pfc_shaping/validation/scorecard.py`).

**Spot-check 1-2 quotes vs forwards XLSX :** non-testable (XLSX inaccessible).
Sera fait sur FMV poste si Phase 10B exécutée plus tard.

**Convention parser keys** (identique à `assembler.build(base_prices=...)`) :

- Année      : `"YYYY"` → e.g. `"2025"`, `"2026"`, `"2027"`
- Trimestre  : `"YYYY-QN"` avec `N ∈ {1, 2, 3, 4}` → e.g. `"2025-Q1"`, `"2025-Q4"`
- Mois       : `"YYYY-MM"` avec zero-padding → e.g. `"2024-08"`, `"2025-12"`

**Body `derive_forwards_from_epex_hist` (résumé impl) :**

1. Strict filter `epex_hist.loc[epex_hist.index < vintage]` (no leakage).
2. Yearly proxy `Y+1, Y+2, Y+3` = `mean(hist)` (uniform shape proxy ; le
   true shape est rétabli par `ShapeHourly` downstream).
3. Quarterly proxy `YYYY-QN` = `mean(hist | quarter==N)` sur chacune des
   3 calendar years futures couvertes par `horizon_days` (default 3×365).
4. Monthly proxy `YYYY-MM` = `mean(hist | month==MM)` sur tous les mois
   contenus dans `[vintage, vintage + horizon_days]`.

**Persistance :** les 24 vintages × ~49 keys ont été cachés dans
`data/forwards_history_phase10.parquet` (long format : `vintage | key |
price | forwards_source`). 1 188 records totaux, `forwards_source.nunique()
== 1` avec valeur `"fallback_diagnostic"`.

**Sanity-check qualitatif vintage 2024-06-28 :** Cal 2025-2027 ≈ 111.41 €/MWh
(moyenne historique 2019-mi-2024, dominée par les pics 2022 énergie), Q1
2025 = 100.94 (≈ hiver), Q2 2025 = 84.05 (≈ printemps doux), Q3 2025 =
138.22 (≈ été haute demande). Ordre de grandeur plausible pour un proxy
hist-based — pas un vrai forward, mais utilisable pour le benchmark Pillar 2
au sens C3 REVIEWS (diagnostic only).

---

## C3 REVIEWS — forwards_source structured flag (NOT just a log line)

**Marker convention** (constantes module-level exportées depuis
`pfc_shaping/validation/scorecard.py`) :

```python
FORWARDS_SOURCE_REAL                 = "real_eex_xlsx"
FORWARDS_SOURCE_FALLBACK_DIAGNOSTIC  = "fallback_diagnostic"
```

**Propagation :**

1. **Parquet level (`data/forwards_history_phase10.parquet`)** : colonne
   obligatoire `forwards_source` à chaque row. Sur Mac Mini default cette
   colonne porte uniformément `"fallback_diagnostic"` (vérifié :
   `nunique() == 1`).
2. **Build level (`build_one(..., forwards_asof=...)`)** : la métadonnée
   est héritée du parquet source des forwards consommés.
3. **Scorecard parquet output (Plan 10-04)** : la métadonnée est propagée
   à chaque cellule (bloc × horizon × config) du scorecard final.

**Gate impact (Plan 10-04 SC#1 evaluator) :**

- Toute cellule avec `forwards_source == "fallback_diagnostic"` est
  annotée explicitement `"Diagnostic only — not gate-eligible"` dans
  `10-VERIFICATION.md` (cf. Plan 10-04 Task 2/3).
- **SC#1 Hildmann 4/4 PASS ne peut être satisfait QUE par un run avec
  `forwards_source == "real_eex_xlsx"`** agrégé sur Config 4
  (production target). Un run fallback ne flippe pas D-FLIP-1.
- Conséquence opérationnelle : si Plan 10-04 est exécuté depuis Mac
  Mini (path actuel), SC#1 ne peut pas être validé sans override
  user explicit acceptant le statut diagnostic-only (auquel cas
  D-FLIP-1 reste BLOCKED).

**Path retenu Plan 10-01 (Mac Mini default) :** `fallback_diagnostic` —
le run final SC#1 nécessitera soit l'accès H:\\ (FMV poste), soit une
override user.

**derive_forwards_from_epex_hist body implémenté :**
oui (Plan 10-01 Task 3 sub-step 3c) dans
`pfc_shaping/validation/scorecard.py`, conformément à la signature
exposée Task 2.
