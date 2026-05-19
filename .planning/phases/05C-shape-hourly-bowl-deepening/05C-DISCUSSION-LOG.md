# Phase 5bis-B: Shape Hourly Bowl-Deepening — Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-19
**Phase:** 5bis-B — shape-hourly-bowl-deepening (padded `05C`)
**Areas discussed:** Hydro climato kernel, Split f_H level/anomaly, σ paramétrisation, Fixture bowl + tests par étage, Flag flip strategy

**Discussion style :** User a explicitement délégué les 4 calls techniques majeurs ("vous êtes les experts EPFL", "QI de 200", "innovation level docteur EPFL") — Claude a verrouillé chaque décision au niveau SOTA quant + innovation, en s'appuyant sur le code base existant et l'audit deep LT 2026-05.

---

## Area 1 — Sémantique hydro climato (Lever 1)

**Question asked :** Comment formuler la cible (centre) du kernel hydro quand flag=ON ?

| Option | Description | Selected |
|--------|-------------|----------|
| Per-timestamp clim[woy(t)] | Pour chaque sample historique t, cible kernel = `_climatological_fill[woy(t)]`. Mesure 'anomalie de fill vs norme saisonnière'. Horizon-agnostic : fit unique, valide Y+0/Y+1/Y+2/Y+3. | ✓ |
| Horizon-dependent (apply-time) | Garde current_fill pour Y+0 (≤6mo), bascule sur clim[woy(t_future)] pour Y+1+. Plus précis mais demande refactor fit→apply. | |
| Clim scalaire (mean) | Cible = scalar mean(_climatological_fill) (perd la signature WoY). | |

**User's choice :** Délégation experte ("EPFL level, à vous de prendre les bons choix par rapport au code base et state of the art").

**Expert/EPFL lock :** Option 1 (Per-timestamp clim[woy(t)]). Justification SOTA :
1. Bug actuel = target time-independent → traite toute déviation comme également informative, indépendamment de la saison.
2. CH-reservoirs : 70% en mars (refilling) ≠ 70% en octobre (post-summer drawdown).
3. Per-timestamp clim[woy(t)] = chaque sample jugé contre sa propre norme saisonnière → préserve la diversité saisonnière par construction.
4. Horizon-invariance gratuite : pas d'apply-time complexity.
5. Aligne avec KYOS analogue-day literature (similarity-weighted KDE).

**Question follow-up asked :** `hydro_weight_sigma` sous flag=ON : valeur ciblée délivrée comme ?

| Option | Description | Selected |
|--------|-------------|----------|
| Constante hardcodée calibrée | self.hydro_weight_sigma = 0.25 si OFF, 0.0X si ON (branch à __init__). | |
| Param ctor exposé sigma_on/sigma_off | __init__(hydro_weight_sigma_off=0.25, hydro_weight_sigma_on=0.07). Future-proof pour A/B tuning. | ✓ |
| Tunable via env var | PFC_LT_HYDRO_WEIGHT_SIGMA_ON=0.07. Cassé par 5bis-A freeze-at-init philosophy. | |

**Expert/EPFL lock :** Option 2. Justification innovation :
- Hyperparameters flag-dependent doivent être **explicites dans la signature** (MLflow/W&B traceability)
- Future-proof : A/B online tuning sans red-deploy
- Coheres avec 5bis-A D-06 (explicit ctor + env default)
- `hydro_weight_sigma_on` persisté dans `shape_hourly.meta.parquet` hyperparams JSON → freeze-at-init train/serve skew protection

---

## Area 2 — Contrat split f_H level/anomaly (Lever 2)

**No AskUserQuestion necessary** — user delegation explicit, full SOTA lock par Claude :

**Math forensique :**
- `mean_h(f_H_cell) ≈ 1.0` strictement par construction (shape_hourly.py:270 fit-time + :345 re-normalisation post-trend).
- Donc `level := mean_h(f_H_cell) ≡ 1.0` (numérique), `anomaly := f_H - 1.0` (zero-mean per cell).
- Le damping `f_H = 1 + (f_H - 1) * sf['f_H']` (assembler.py:333) compresse l'anomaly (signature saisonnière) parce qu'il pull tous les f_H[t] vers 1.0 per timestamp.
- "Damper uniquement le level" + "level ≡ 1" ⇔ **bypass total du damping sur f_H sous flag=ON**.

**Architecture lock :**
- Helper module-level `_split_level_anomaly(f_H_series, cal_df)` dans shape_hourly.py
- Additive split (sum-preserving)
- Damping flag=ON : level damped via sf['f_H'], anomaly pass-through 100%
- Damping flag=OFF : legacy preserved
- Telemetry innovation : log `max |level - 1.0|`, warning si > 1e-6 (drift detection)
- Knot schedule level damping = identique à shape_freedom['f_H'] actuel

**Why split même si level≈1 numériquement :**
1. Telemetry traceable (drift d'invariant détecté en prod)
2. Future-proof si Phase 5 MSFC log-prix ou un fit ultérieur changeait la normalisation
3. Documente explicitement l'intent dans le code

---

## Area 3 — Paramétrisation σ smoothing (Lever 3)

**Question asked :** σ smoothing sous flag : un seul param branché, ou exposé explicitement comme sigma_off/sigma_on ?

| Option | Description | Selected |
|--------|-------------|----------|
| Param ctor exposé sigma_off/sigma_on | Même convention que hydro Area 1. __init__(sigma_off=0.5, sigma_on=0.25). Branch à __init__. Persisté sidecar. | ✓ |
| Single sigma arg, default branché | __init__(sigma=None). Plus compact mais cache la logique flag. | |
| Tunable env var | PFC_LT_SIGMA_ON. Cassé par 5bis-A freeze-at-init. | |

**User's choice :** Délégation experte ("docteur, doctorant EPFL et QI de 200").

**Expert/PhD lock :** Option 1. Justification :
1. **Consistency principle** : match le pattern hydro_weight_sigma_off/_on Area 1.
2. **Information theory** : σ=0.25 vs σ=0.5 change la bandwidth de smoothing (FWHM 0.59h vs 1.18h sur grille 15min Nyquist). Auditable en MLflow.
3. **Reproducibility hardness** : sidecar persiste les DEUX valeurs.
4. **Sub-Nyquist validation** : researcher validera empiriquement σ_on=0.25 contre fixture EPEX synth.

**Verification breaking-change :** 4 callsites passent `sigma=X` explicitement (autoresearch:234, rolling_update:365, tests:56/239/250/628). Backward-compat préservée via resolution precedence D-A3-2 : `sigma=X` legacy callsite → `_sigma_off = _sigma_on = X`, bit-pour-bit identique.

---

## Area 4 — Fixture bowl-deepening + assertions par étage

**No AskUserQuestion necessary pour le contenu** — Claude lock full SOTA pour fixture + 7 tests. Une question stratégique demandée sur la décomposition :

**Question asked :** Décomposition en plans atomiques pour 5bis-B (waves d'exécution) ?

| Option | Description | Selected |
|--------|-------------|----------|
| 3 plans séquentiels (1 lever/plan) | Aligné 5bis-A convention (5 plans waves séquentielles). Bisection facile. | ✓ |
| 2 plans (math + tests) | Plan 1 math complet, Plan 2 tests. Plus rapide mais bisection plus dure. | |
| 1 plan monolithique | Un commit feat(LT). Non bisectable. | |

**User's choice :** 3 plans séquentiels.

**Lock Area 4 (test design synthétisé EPFL SOTA) :**
- Fixture déterministe synthétique seed=42 (~50KB) — méthode scientifique : ground truth analytique, repro garantie, np.ptp attendu calculable a priori.
- Nouveau fichier `tests/test_shape_hourly_bowl.py` (séparé de test_shape_hourly_infra.py 5bis-A).
- 7 tests : kernel reformulation (D-A4-3), split invariant (D-A4-4), SC #1 ptp deepening (D-A4-5), SC #3 amplitude M+30 (D-A4-6), SC #2 seasonal solar/evening delta sur synth (D-A4-7), baseline flag=OFF bit-pour-bit (D-A4-8), nouvelle baseline flag=ON (D-A4-9).
- Innovation gating SC #2 : 5bis-B passe SC #2 sur synth = condition nécessaire (math) ; Phase 10 valide sur HFC OMPEX réel = condition suffisante (data fit).
- Nouvelle convention pattern : chaque flag transition / math change atomique = nouvelle baseline frozen séparée.

---

## Final question — Flag flip strategy

**Question asked :** Date de flip du flag PFC_LT_USE_SEASONAL_HOURLY_SHAPE de default OFF → default ON ?

| Option | Description | Selected |
|--------|-------------|----------|
| Après Phase 10 (real-data gate) | No production change without empirical validation. Flip = post Phase 10 success. Zéro risk production. | ✓ |
| T+30j après merge 5bis-B | Convention 5bis-A CONTEXT proposait ça. Hypothèse implicite : math correcte sur synth ⇒ OK en prod. Production risk. | |
| Manuel via prod ops | Conservative mais crée dette permanente. | |

**User's choice :** Après Phase 10 (real-data gate).

**Lock :** D-FLIP-1 — Inscrit dans PROJECT.md `Key Decisions` à la livraison 5bis-B.

---

## Claude's Discretion

User a explicitement délégué les 4 calls techniques majeurs aux experts EPFL/PhD. Les décisions D-A1-1..5, D-A2-1..6, D-A3-1..6, D-A4-1..10 sont prises au niveau SOTA quant + innovation, avec justification écrite dans CONTEXT.md pour traçabilité.

**Implementation details laissés au planner :**
- Format exact synthetic bowl fixture (long vs wide DataFrame)
- Pattern Python pour cross-plan persistence compat sur sigma_off/sigma_on
- Niveau de granularité telemetry log (INFO vs DEBUG)
- Format exact des messages telemetry

**Calibration empirique laissée au researcher (gsd-phase-researcher) :**
- Valeur exacte `hydro_weight_sigma_on` (probable ~0.05-0.10)
- Valeur exacte threshold multiplicatif np.ptp ratio (D-A4-5)
- Valeur exacte threshold amplitude f_H post-damping à M+30 (D-A4-6)

---

## Deferred Ideas

Voir CONTEXT.md `<deferred>` section pour la liste complète. Highlights :
- Vers Phase 10 : backtest réel HFC OMPEX, cible Δ MAE bloc ≤ -1.5 €/MWh, gate flip flag.
- Vers Phase 5 : floors silencieux (MSFC, m_factor, F_WV).
- Vers Phase 5ter : distribution probabiliste par bloc Monte-Carlo.
- Cleanup `.planning/phases/05bis-shape-seasonal-hourly/CONTEXT.md` predoc SUPERSEDED.
- Calibration online σ_on et hydro_weight_sigma_on post Phase 10.

**Non-deferral :** σ_on=0.25 et hydro_weight_sigma_on TBD doivent être calibrés dans le cadre du Plan 05C-01-PLAN.md (pas reportés indéfiniment). Pas de magic-number-deferred.
