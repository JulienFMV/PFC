# Implementation Context — Phase 5bis

# Current Context Notice - Phase 14 Supersedes This File

This file is historical Phase 5bis context and is not the active Phase 14
handoff. For current LT audit/remediation work, read these files first:

1. `AGENTS.md`
2. `.planning/HANDOFF.md`
3. `.planning/phases/14-lt-audit-remediation/DECISION-LOG.md`
4. the latest `SESSION-HANDOFF-YYYYMMDD-*.md` named in `.planning/HANDOFF.md`

Do not treat the 2026-06-18 residual-anchor prompt as the target architecture
for Phase 14. The active target is the 2026-06-19 monthly solver reform:
one monthly BASE solver with hard CH EEX constraints, external information only
as zero-mean shape, and no individual month patches after the solver.

## Why this phase, why now

FMV n'arrive pas à pricer correctement les profile deals industriels — typiquement
un client veut vendre à FMV ses heures solaires (10-15) et acheter à FMV ses heures
non-solaires (18-9). La PFC OMPEX échoue parce que son `f_H` est moyenné sur toute
l'année, donc le bowl midday d'été (~-30% vs moyenne) et le plateau midday d'hiver
(~+5%) sont écrasés en un seul facteur. Résultat : OMPEX over-price le bloc 10-15
été (et donc FMV over-paie en achat client) ou under-price le bloc 18-9 hiver (et
donc FMV under-charge en vente client). Soit FMV perd, soit FMV n'est plus
compétitif.

**Notre PFC actuelle souffre exactement du même biais** : `ShapeHourly.factors_`
est indexé par `(saison, type_jour)` retournant un array[24], donc l'heure 12
est moyennée entre été et hiver avant d'être utilisée.

**Phase 5bis** étend l'indexation à `(saison, type_jour, hour)`, donc chaque
combinaison a son propre facteur. Le bowl d'été midday devient distinct du
plateau d'hiver midday.

## Architecture cible

### Aujourd'hui

```python
# pfc_shaping/lt/model/shape_hourly.py
self.factors_: dict[tuple[str, str], np.ndarray] = {
    ("Hiver", "Ouvrable"): np.array([f_h0, f_h1, ..., f_h23]),
    ("Hiver", "Samedi"):   ...,
    ("Hiver", "Dimanche"): ...,
    ("Hiver", "Ferie_CH"): ...,
    ("Ete",   "Ouvrable"): ...,
    ...
}

# apply()
def apply(self, idx, cal, ...):
    for ts in idx:
        f_H[ts] = self.factors_[(saison(ts), type_jour(ts))][hour(ts)]
```

Le `[hour]` indexe l'array, donc une combinaison `(Hiver, Ouvrable)` a 24 valeurs
indépendantes. **Mais ces 24 valeurs sont moyennées sur tout l'hiver** — donc
le facteur h12 d'un janvier vs un mars est confondu.

C'est moins grave que ce que je décrivais initialement (l'indexation EST déjà
en partie seasonal) — la vraie limitation est que la **granularité saisonnière
est trop large** (4 saisons cardinales). Le bowl d'été varie fortement entre
juin/juillet/août vs septembre. Le plateau d'hiver h13 varie entre janvier vs mars.

### Cible Phase 5bis

```python
# Option A : nested dict (backward-compat-friendly)
self.factors_: dict[tuple[str, str], dict[int, np.ndarray]] = {
    ("Hiver", "Ouvrable"): {1: arr24, 2: arr24, 3: arr24, ...},  # par mois
    ...
}

# Option B : flat dict (plus granulaire mais index plus complexe)
self.factors_: dict[tuple[str, str, int, int], float] = {
    ("Hiver", "Ouvrable", 1, 12): 1.05,  # (saison, type_jour, mois, heure)
    ...
}
```

→ **À discuter en `/gsd:discuss-phase 5bis`** : option A (nested, simple migration)
ou option B (flat, plus granulaire, plus de risque d'overfit).

**Recommandation provisoire** : Option A avec backward-compat lazy promotion.
Un `factors_` legacy (2D) est promu en 3D au load via réplication identique sur
les 12 mois — pas de perte d'info, pas de breaking change.

### Granularité saisonnière révisée

Aujourd'hui `calendar_ch.py` mappe mois → saison via :
- Hiver = {11, 12, 1, 2, 3}
- Printemps = {4, 5}
- Été = {6, 7, 8, 9}
- Automne = {10}

→ **Granularité fixe et arbitraire**. Pour Phase 5bis, on pourrait **soit** :
- (a) Garder ces 4 saisons mais ajouter une dimension `hour` (factors_ devient 3D)
- (b) Passer à 12 mois (factors_ devient `(month, type_jour) → arr24`)
- (c) Combinaison hybride (factors_ devient `(saison, type_jour, hour, month_within_saison)`)

→ Recommandation : **option (b)** pour la Phase 5bis. 12 mois × 5 types_jour × 24h
= 1440 facteurs, calibrés sur ~6 ans × 12 mois × 5 types × 24h × ~28 jours = ~26000
observations en moyenne par cellule. Largement suffisant statistiquement et bien
plus interprétable que 4 saisons artificielles.

**À acter dans `/gsd:discuss-phase`**.

## Fichiers à toucher

- `pfc_shaping/lt/model/shape_hourly.py` (fit, apply, save, load)
- `pfc_shaping/lt/model/assembler.py` (signature `apply()` propage `country` déjà OK depuis Bloc A — vérifier que la nouvelle indexation est consommée correctement)
- Aucun autre fichier LT côté code source

## Tests à ajouter

`tests/test_shape_hourly_seasonal.py` :

1. **`test_fit_produces_seasonal_hourly_factors`** : fit synthétique sur 2 ans
   avec bowl injecté (`price[h12, month=7] = 50 * 0.4 = 20`, `price[h12, month=1] = 50 * 1.05 = 52.5`) → vérifier que `factors_` reflète ces ratios.

2. **`test_apply_differentiates_seasons`** : sur un index 15min couvrant un dimanche d'été et un dimanche d'hiver, `mean(f_H[10-15] été) < mean(f_H[10-15] hiver)` avec écart > 15%.

3. **`test_energy_invariant`** : pour chaque (saison, type_jour), `np.mean(factors_[(s,t)])` ≈ 1.0 ± 1e-3.

4. **`test_backward_compat_legacy_factors_2d`** : créer un parquet legacy au format 2D, charger via `ShapeHourly.load()`, vérifier que la promotion 2D→3D n'altère pas le comportement bit-pour-bit avec le legacy code path.

5. **`test_feature_flag_off_falls_back_to_legacy`** : `PFC_LT_USE_SEASONAL_HOURLY_SHAPE=0` ramène au comportement 2D historique exactement.

## Backtest acceptance criteria (gated by Phase 10)

Le backtest réel par bloc vs HFC OMPEX est livré en Phase 10. Mais on peut
**spot-check** dès Phase 5bis :

```bash
# Quick sanity sur 2024-2025
python scripts/backtest_block_quickcheck.py \
    --pfc-mode seasonal_hourly \
    --baseline-mode legacy \
    --blocks "10-15_weekday_summer,18-9_weekday_winter,12-16_weekend_summer" \
    --period "2024-01-01:2025-12-31"
```

Attendu : Δ MAE négatif sur les blocs solaires d'été (la cible où le bowl
historique est sous-estimé par le f_H global).

## Out of scope strict

- ❌ Ne pas toucher `pfc_shaping/ct/*` ni `swiss_short_term.py` (CT)
- ❌ Ne pas activer FR/AT/IT (Phase 3 HOLD)
- ❌ Ne pas retirer les floors silencieux (Phase 5 séparée)
- ❌ Ne pas ajouter de Monte-Carlo shape (Phase 5ter séparée)
- ❌ Ne pas refondre Phase 10 (parallèle, plan séparé)

## Premier acte attendu

Lancer `/gsd:discuss-phase 5bis` qui posera (au minimum) :
1. Option (a), (b) ou (c) pour la granularité saisonnière ?
2. Backward-compat strict via feature flag ou breaking change ?
3. Stratégie de test : synthétique seul ou + fixture EPEX réelle ?
4. Estimation du delta MAE attendu : quel est le seuil "réussite" pour cette phase isolée (avant Phase 10) ?
