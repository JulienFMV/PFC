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

_(à remplir par Sub-step 3b après bootstrap EPEX 2019-2025)_

---

## RESEARCH Q2 — Forwards historiques as-of vintage (Mac Mini)

_(à remplir par Sub-step 3c après test H:\\ + implémentation
 derive_forwards_from_epex_hist body)_

---

## C3 REVIEWS — forwards_source structured flag (NOT just a log line)

_(à remplir par Sub-step 3c — convention valeurs FORWARDS_SOURCE_REAL /
 FORWARDS_SOURCE_FALLBACK_DIAGNOSTIC + propagation parquet + gate impact)_
