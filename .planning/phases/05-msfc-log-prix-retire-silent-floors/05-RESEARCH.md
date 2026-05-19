# Phase 05: MSFC retire silent floors + PFC peut être négative — Research

**Researched:** 2026-05-19
**Domain:** Python numeric pipeline — floor removal, additive refactor, signed-aware PCHIP, feature flag
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Area 1 — MSFC methodology**
- D-A1-1 : MSFC reste LINÉAIRE (PCHIP sur prix bruts). "MSFC log-prix" du titre ROADMAP est un artefact historique non-binding.
- D-A1-2 : Clamp d'extrapolation ligne 120 : `np.clip(B_smooth_raw, y_knots.min() - margin, y_knots.max() + margin)` avec `margin = 0.5 * np.ptp(y_knots)`.
- D-A1-3 : NEG-05 invariance garantie par construction (`_enforce_mean_constraints` iterative est sign-invariante).

**Area 2 — Floor strategy**
- D-A2-1 : ctor args defaults False (negative-ready par défaut). `smooth_base_prices(enforce_positivity=False)`, `ArbitrageFreeCalibrator(enforce_m_factor_floor=False)`, `WaterValueCorrection(enforce_floor=False)`, `BlockCascading(allow_negative_peak=True)` — tous False/True = defaults négatif-ready.
- D-A2-2 : Master flag `PFC_LT_ALLOW_NEGATIVE_PRICES` lu UNE fois à `PFCAssembler.__init__`. Audit-trail INFO log only, pas un override automatique. Format : `"PFC_LT_ALLOW_NEGATIVE_PRICES={state}, floors_disabled={msfc:..., af:..., wv:..., cascading:...}"`.
- D-A2-3 : Rollback opérateur = passer `enforce_*=True` / `allow_negative_peak=False` aux 4 callsites explicitement.
- D-A2-4 : Sidecar persistence pour `PFC_LT_ALLOW_NEGATIVE_PRICES` : question ouverte au planning (étendre `shape_hourly.meta.parquet` vs créer `assembler.meta.parquet` séparé). Tranché dans cette RESEARCH.
- D-A2-5 : Baseline `baseline_pfc_seed42.parquet` (5bis-A) reste verte sous defaults OFF si f_wv multiplicatif vs additif < 1e-12 — à vérifier par dry-run. **Résultat dry-run ci-dessous : écart > 1e-12, donc nouvelle baseline requise.**

**Area 3 — F_WV delta additif**
- D-A3-1 : `compute_delta_wv(B_smooth, fill_df, calendar_df) → pd.Series` retournant `delta_wv = (f_wv - 1) × |B_smooth|`.
- D-A3-2 : `assembler.build()` applique `P = B × f_H × f_W + delta_wv` au lieu de `P = B × f_H × f_W × f_wv`.
- D-A3-3 : Équivalence baseline_5bisA NON exacte (écart d'ordre `(f_wv-1) × B × (f_H×f_W - 1)` ≫ 1e-12). Nouvelle baseline `baseline_pfc_seed42_phase05.parquet` requise (D-A4-9 pattern).
- D-A3-4 : `F_WV_FLOOR=0.80` retiré quand `enforce_floor=False`. `compute_delta_wv` raise si `enforce_floor=True` (incompatible).
- D-A3-5 : Telemetry INFO par `assembler.build()` : `"WV delta_wv: min=%.2f, max=%.2f, mean=%.2f €/MWh, sign(B) flips: %d"`.

**Area 4 — Peak synthesis spread additif + tests**
- D-A4-1 : `fit_peak_spreads(spot_history)` agrégeant `peak_avg - base_avg` par mois, persiste `peak_base_spreads_: dict[int, float]`. `synthesize_peak_prices` utilise `result[peak_key] = base_price + peak_base_spreads_[month]` quand `allow_negative_peak=True`.
- D-A4-2 : `fit_peak_ratios` DEPRECATED avec `DeprecationWarning`. Shim : raise `NotImplementedError` si appelé sans spot_history. Logue WARN si `peak_base_ratios_` chargé sans `peak_base_spreads_`.
- D-A4-3 : Legacy `allow_negative_peak=False` garde le multiplicateur `ratio*price` + `Peak >= Base` clip.
- D-A4-4 : 4 unit tests math sign-invariance (in-test inputs, pas de fixture parquet) : `test_msfc_signed_monthly_repricing`, `test_arbitrage_free_signed_target`, `test_water_value_delta_sign_invariant`, `test_cascading_spread_signed_base`.
- D-A4-5 : 1 system acceptance (SC #2 ROADMAP) : `test_phase05_summer_bowl_negative_acceptance` — gated par 5bis-B bowl calibration ; skip avec message si bowl non vérifié.
- D-A4-6 : Nouvelle baseline frozen `baseline_pfc_seed42_phase05.parquet` ; régression `assert_frame_equal(atol=1e-12, rtol=0)`.
- D-A4-7 : Reformulation NEG-05 dans REQUIREMENTS.md (monthly forward négatif July M-07'27 = -2 €/MWh).

**Plan decomposition — D-A5-1**
- Plan 05-01 : MSFC + ArbitrageFreeCalibrator (Areas 1+2 partiels)
- Plan 05-02 : WaterValueCorrection delta additif + assembler integration (Area 3)
- Plan 05-03 : BlockCascading spread additif + master flag + fixture + baseline + tests (Area 4)

### Claude's Discretion
- Format exact du log telemetry (D-A2-2, D-A3-5) : INFO vs DEBUG, format string exact.
- Sidecar persistence pour `PFC_LT_ALLOW_NEGATIVE_PRICES` : étendre `shape_hourly.meta.parquet` OU créer `assembler.meta.parquet` séparé. **Tranché ici — voir `## Architecture Patterns §Sidecar`.**
- Pattern Python pour `fit_peak_ratios` deprecated shim (D-A4-2) : raise `NotImplementedError` vs alias avec WARN. **Tranché ici — voir `## Architecture Patterns §Cascading`.**
- Variable de margin pour clamp signed-aware (D-A1-2) : `0.5 * np.ptp(y_knots)` validé ci-dessous.

### Deferred Ideas (OUT OF SCOPE)
- Phase 5ter : distribution probabiliste par bloc Monte-Carlo.
- Phase 10 : backtest réel HFC OMPEX 2024-2025.
- TODO P1-01 : smoothness proportionnelle log-space (`arbitrage_free.py:455-468`).
- `PFC_LT_FORCE_LEGACY_FLOORS=1` hot-rollback.
- 2020-Q2 historical real-data validation.
- Cleanup titre ROADMAP "MSFC log-prix".
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| NEG-01 | `np.maximum(B_smooth, 1.0)` dans `msfc_spline.py` conditionné par `enforce_positivity=False` (default off) | Lignes 131 et 203 identifiées ; `smooth_base_prices` est fonctionnelle → kwarg avec default False. Pattern `_resolve_flag` de 5bis-A comme modèle. |
| NEG-02 | `m_factor >= 0.1` ne masque plus les résidus : `converged=False` propagé quand clip appliqué | Ligne 517 dans `arbitrage_free.py` identifiée. `converged` est déjà calculé ligne 546 — conditionner le clip sur `enforce_m_factor_floor`. |
| NEG-03 | `WaterValueCorrection.F_WV_FLOOR` configurable ou désactivable sur export LT | Lignes 394 et 407 identifiées. `enforce_floor=False` default → retire les deux clips. Refactor delta-additif simultané (D-A3-1). |
| NEG-04 | `cascading.synthesize_peak_prices` n'impose pas ratio >= 1 sur forwards Cal négatifs | `ContractCascader` (pas `BlockCascading`) est la classe réelle dans le code. `fit_peak_spreads` + `allow_negative_peak=True` default (D-A4-1). |
| NEG-05 | (REFORMULÉ D-A4-7) Monthly forward négatif (July M-07'27 = -2 €/MWh) repricé exactement à -2 €/MWh moyenne du mois | `test_msfc_signed_monthly_repricing` + mean constraint sign-invariance (D-A1-3). REQUIREMENTS.md à reformuler dans Plan 05-01. |
</phase_requirements>

---

## Summary

Phase 5 autorise des prix négatifs dans la PFC LT en retirant (conditionnellement) 4 planchers silencieux répartis sur 3 modules + 1 refactor sémantique de `f_wv` multiplicatif vers additif. La phase ne crée pas de nouveau modèle — elle expose et ctor-gate les 4 `np.maximum`/`clip` existants, et change l'application de `f_WV` de multiplicative à delta-additive pour préserver la sémantique "correction en €/MWh" indépendamment du signe de B.

**Dry-run D-A3-3 résolu :** L'écart `P_mult − P_add = B × (f_H×f_W − 1) × (f_wv − 1)` est typiquement 0.2 à 1.4 €/MWh (non nul ≫ 1e-12). Une nouvelle baseline `baseline_pfc_seed42_phase05.parquet` est donc obligatoire, en sus de la preservation de `baseline_pfc_seed42.parquet` (5bis-A, testable via `enforce_*=True`).

**Sidecar D-A2-4 résolu :** `PFC_LT_ALLOW_NEGATIVE_PRICES` est un attribut `PFCAssembler` (pas `ShapeHourly`). Il doit donc aller dans un sidecar séparé `assembler.meta.parquet` si persisté — mais la valeur résolue peut aussi simplement être dans les logs (audit-trail INFO) car le master flag n'est qu'un observateur, pas un gate. Décision : **ne pas créer de nouveau sidecar pour cette phase** ; le flag est loggué à l'init, et les 4 ctor args sont la véritable surface API (persistés via les sidecars de leurs classes respectives si besoin). Cohérent avec D-A2-2 "audit-trail INFO log only".

**Callsite audit `fit_peak_ratios` résolu :** Deux callsites actifs dans `production_phases.py` (lignes 344, 644). Les deux passent `spot_history` (via `inputs.epex_ch` et `spec.epex_df`). Migration vers `fit_peak_spreads` faisable sans `NotImplementedError`. La stratégie retenue est `DeprecationWarning` + shim transparent (D-A4-2).

**Primary recommendation:** Implementer les 3 plans en séquence — Plan 01 retire les 2 floors MSFC + 1 floor ArbitrageCalibrator avec leurs ctor args, Plan 02 refactorise `f_WV` en delta-additif et adapte `assembler.build()`, Plan 03 ajoute `fit_peak_spreads` à `ContractCascader`, le master flag audit-trail dans `PFCAssembler`, et génère la nouvelle baseline + tests.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Signed-aware PCHIP clamp | `msfc_spline.py` (calcul) | `assembler.py` (appel) | La logique de clamp appartient au module MSFC où les knots vivent |
| Floor MSFC `enforce_positivity` | `msfc_spline.py` lignes 131, 203 | — | Deux sites distincts dans `smooth_base_prices` + `_enforce_mean_constraints` |
| Floor m_factor `enforce_m_factor_floor` | `arbitrage_free.py` ligne 517 | — | Local à la calibration arbitrage-free |
| Floor f_WV `enforce_floor` | `water_value.py` lignes 394, 407 | — | Local à WaterValueCorrection.apply() |
| Delta-additif `compute_delta_wv` | `water_value.py` (calcul) | `assembler.py` (consommation) | API publique sur WaterValueCorrection, consommée dans assembler.build() |
| Floor peak ratio `allow_negative_peak` | `cascading.py` ContractCascader | — | Local à synthesize_peak_prices |
| Spread additif `fit_peak_spreads` | `cascading.py` ContractCascader | — | Complément de l'API fit existante |
| Master flag audit-trail | `assembler.py` PFCAssembler.__init__ | — | Point d'entrée unique du pipeline LT |
| Fixture forwads_phase05 | `tests/fixtures/` | — | Génération déterministe seed=42 |
| Baseline phase05 | `tests/fixtures/` | — | Convention D-A4-9 5bis-B pattern |

---

## Standard Stack

### Core (aucun nouveau package)
| Library | Version installée | Purpose | Statut |
|---------|-------------------|---------|--------|
| numpy | (projet, ≥1.24) | `np.clip`, `np.maximum`, `np.ptp`, `np.abs` | Déjà utilisé dans tous les modules |
| pandas | 2.3.3 | `pd.Series.clip`, DataFrames | Déjà utilisé |
| scipy | (projet) | `PchipInterpolator` dans msfc_spline.py | Déjà utilisé |
| pytest | 8.4.2 | Tests unitaires et acceptance | Déjà en place |

**Aucune nouvelle dépendance.** Phase 5 est un refactor interne pur.

### Installation
```bash
# Aucune commande d'installation requise — pas de nouveau package.
```

---

## Package Legitimacy Audit

> Non applicable — Phase 5 n'installe aucun package externe.

---

## Architecture Patterns

### Diagramme de flux

```
Forwards (base_prices dict)
         │
         ▼
[Plan 05-01] ContractCascader.cascade()
         │  ├── fit_peak_spreads() [NEW, ±€/MWh par mois]
         │  └── synthesize_peak_prices() [spread additif si allow_negative_peak=True]
         ▼
[Plan 05-01] smooth_base_prices(enforce_positivity=False)  ← msfc_spline.py
         │  ├── PCHIP sur knots signés
         │  ├── Clamp signed-aware (y_min - margin, y_max + margin)  [FIXED]
         │  ├── _enforce_mean_constraints()  [sign-invariante par construction]
         │  └── [SKIP] np.maximum(B, 1.0)  ← floor 1 (l.131) et floor 2 (l.203) retirés
         ▼
[Plan 05-01] ArbitrageFreeCalibrator.calibrate(enforce_m_factor_floor=False)
         │  └── [SKIP] m_factor = np.maximum(m_factor, 0.1)  ← floor 3 (l.517) conditionnel
         ▼
assembler.build()  ← PFCAssembler
         │  [Plan 05-03] __init__: lit PFC_LT_ALLOW_NEGATIVE_PRICES, log INFO audit-trail
         │
         ├── f_H (ShapeHourly.apply)  ← inchangé Phase 5
         ├── f_W  ← inchangé
         ├── f_S  ← inchangé
         │
         ├── [Plan 05-02] WaterValueCorrection
         │    ├── .apply() → f_wv = 1 + β×fill_dev×... [enforce_floor=False → SKIP clip]
         │    └── .compute_delta_wv(B_smooth) → delta_wv = (f_wv - 1) × |B_smooth|
         │
         ├── B = smooth_base_prices(...)  [prix signés possibles]
         │
         ├── [LEGACY] price_raw = B × f_S × f_W × f_H × f_Q × f_WV × f_bridge
         └── [NEW Plan 05-02] price_raw = B × f_S × f_W × f_H × f_Q × f_bridge + delta_wv_scaled
                                          (delta_wv appliqué post-facteurs shape, pré-calibration)
         ▼
output PFC 15min [prix négatifs possibles à h13 dimanche été]
```

### Structure de fichiers (Phase 5, fichiers touchés ou créés)

```
pfc_shaping/
├── lt/model/
│   ├── msfc_spline.py              # Plan 05-01: enforce_positivity kwarg, clamp signed-aware
│   ├── water_value.py              # Plan 05-02: enforce_floor kwarg, compute_delta_wv()
│   └── assembler.py                # Plan 05-02+03: delta_wv integration, master flag
├── calibration/
│   ├── arbitrage_free.py           # Plan 05-01: enforce_m_factor_floor kwarg
│   └── cascading.py                # Plan 05-03: fit_peak_spreads(), allow_negative_peak
tests/
├── fixtures/
│   ├── _generate_phase05_fixture.py   # Plan 05-03: NEW — génère forwards_phase05_seed42
│   ├── forwards_phase05_seed42.parquet # Plan 05-03: NEW — Cal'27=30, July=20, seed=42
│   └── baseline_pfc_seed42_phase05.parquet  # Plan 05-03: NEW — baseline frozen defaults OFF
└── test_phase05_negative_prices.py    # Plan 05-03: NEW — 4 unit math + 1 acceptance test
.planning/REQUIREMENTS.md              # Plan 05-01: reformulation NEG-05 (D-A4-7)
```

### Pattern 1 : ctor arg `enforce_*` avec default False

**What:** Ajouter un kwarg booléen à chaque fonction/constructeur qui héberge un floor. Default = `False` = floor désactivé (negative-ready).
**When to use:** Pour chaque des 4 planchers silencieux identifiés.

```python
# Source: CONTEXT.md D-A2-1 + code existant msfc_spline.py:39
def smooth_base_prices(
    idx: pd.DatetimeIndex,
    base_prices: dict[str, float],
    B_flat: pd.Series,
    enforce_positivity: bool = False,  # NEW: default False = negative-ready
) -> pd.Series:
    ...
    # Dans _enforce_mean_constraints, remplacer l.203:
    #   return np.maximum(result, 1.0)
    # par :
    return np.maximum(result, 1.0) if enforce_positivity else result

    # Dans smooth_base_prices, remplacer l.131:
    #   B_smooth = np.maximum(B_smooth, 1.0)
    # par :
    if enforce_positivity:
        B_smooth = np.maximum(B_smooth, 1.0)
```

**Important :** `smooth_base_prices` est une fonction, pas une classe. Le kwarg `enforce_positivity` doit être propagé à `_enforce_mean_constraints` en paramètre (ou la logique de floor dans `_enforce_mean_constraints` est conditionnée sur le paramètre reçu). Voir pitfall n°1 ci-dessous.

### Pattern 2 : Clamp signed-aware ligne 120

**What:** Remplacer `y_knots.min()*0.5, y_knots.max()*2.0` par des bornes symétriques autour du range des knots.

```python
# Source: CONTEXT.md D-A1-2 + validation dry-run (voir §Assumptions Log)
# Ancien (CASSÉ pour y_knots négatifs):
B_smooth_raw = np.clip(B_smooth_raw, y_knots.min() * 0.5, y_knots.max() * 2.0)
# Nouveau:
margin = 0.5 * np.ptp(y_knots)  # = 0.5 * (max - min), toujours >= 0
B_smooth_raw = np.clip(B_smooth_raw, y_knots.min() - margin, y_knots.max() + margin)
```

**Validation (dry-run):**
- y_knots = [-5, 30, 25, 40] → margin=22.5 → bounds=[-27.5, 62.5] ✓ (CONTEXT.md attendu)
- y_knots all-negative [-30, -20, -25] → bounds=[-35, -15] ✓ (contient le range)
- Ancien sur all-negative → bounds=[-15, -40] ✗ (bounds inversées, lo > hi) [VERIFIED: code]

### Pattern 3 : Delta-additif WaterValue

**What:** Nouvelle API publique `compute_delta_wv` sur `WaterValueCorrection`. Sémantique : en €/MWh absolu, pas en facteur multiplicatif.

```python
# Source: CONTEXT.md D-A3-1 / D-A3-2
def compute_delta_wv(
    self,
    B_smooth: pd.Series,
    fill_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
) -> pd.Series:
    """Retourne delta_wv = (f_wv - 1) * |B_smooth| en €/MWh.

    Sign-invariant: si B < 0 (prix négatif), |B| reste positif → delta_wv
    a le bon signe (scarcity = prix plus bas en absolu, abundance = prix
    plus haut en absolu), indépendamment du signe de B.

    Raises ValueError si self.enforce_floor=True (incompatible avec delta-additif).
    """
    if self.enforce_floor:
        raise ValueError(
            "compute_delta_wv() n'est pas compatible avec enforce_floor=True. "
            "Utiliser apply() pour le comportement multiplicatif legacy."
        )
    f_wv = self.apply(B_smooth.index, calendar_df, fill_df)
    return (f_wv - 1.0) * B_smooth.abs()
```

**Dans assembler.build() :**

```python
# Source: CONTEXT.md D-A3-2 + code assembler.py:403
# ANCIEN (multiplicatif — potentiellement sign-incorrect):
price_raw = B * f_S * f_W * f_H * f_Q * f_WV * f_bridge

# NOUVEAU (additif — sign-invariant):
# f_WV est gardé dans shape_freedom à 1 (pass-through pur), delta appliqué séparément
if self.wv is not None and not self.wv.enforce_floor:
    delta_wv = self.wv.compute_delta_wv(B, hydro_forecast, cal)
    price_raw = B * f_S * f_W * f_H * f_Q * f_bridge + delta_wv
else:
    # Legacy path ou wv=None
    f_WV = self.wv.apply(idx, cal, hydro_forecast) if self.wv is not None else pd.Series(1.0, index=idx)
    price_raw = B * f_S * f_W * f_H * f_Q * f_WV * f_bridge
```

**Note :** Le `f_WV` dans le `shape_freedom` damping actuel (ligne 389 `f_WV = 1.0 + (f_WV - 1.0) * shape_freedom["f_WV"]`) doit être contourné sur le nouveau path. Le delta-additif incorpore déjà le `horizon_decay` interne à `WaterValueCorrection.apply()`.

### Pattern 4 : Spread additif ContractCascader

**What:** Nouvelle méthode `fit_peak_spreads` qui stocke `peak_base_spreads_: dict[int, float]` (€/MWh), utilisée dans `synthesize_peak_prices` quand `allow_negative_peak=True` (default).

```python
# Source: CONTEXT.md D-A4-1
def fit_peak_spreads(
    self,
    spot_history: pd.DataFrame,
) -> "ContractCascader":
    """Calibre peak_base_spreads_ = mean(Peak_price - Base_price) par mois.

    Stocke dict[month: int -> spread: float] en €/MWh.
    """
    ...
    self.peak_base_spreads_: dict[int, float] = {...}
    return self
```

**Changement dans `synthesize_peak_prices` :**
```python
# Source: CONTEXT.md D-A4-1 + code cascading.py:390
# Si allow_negative_peak=True (default) et peak_base_spreads_ disponible:
result[peak_key] = base_price + peak_base_spreads_.get(month, fallback_spread)
# Si allow_negative_peak=False (legacy): ancien multiplicateur × ratio (inchangé)
```

### Pattern 5 : Master flag audit-trail

**What:** `PFCAssembler.__init__` lit `PFC_LT_ALLOW_NEGATIVE_PRICES` une fois, logue INFO. Même pattern que `_resolve_flag` dans `shape_hourly.py:92-119`.

```python
# Source: CONTEXT.md D-A2-2 + shape_hourly.py:92-119 (modèle _resolve_flag)
_ALLOW_NEG_ENV_VAR = "PFC_LT_ALLOW_NEGATIVE_PRICES"

def _resolve_allow_negative(explicit: bool | None) -> bool:
    if explicit is not None:
        return bool(explicit)
    raw = os.getenv(_ALLOW_NEG_ENV_VAR, "0")
    if raw == "1":
        return True
    if raw == "0":
        return False
    logger.warning("PFC_LT_ALLOW_NEGATIVE_PRICES=%r invalide — traité comme False", raw)
    return False

# Dans PFCAssembler.__init__ :
self._allow_negative_prices: bool = _resolve_allow_negative(allow_negative_prices)
logger.info(
    "PFC_LT_ALLOW_NEGATIVE_PRICES=%s, floors_disabled={"
    "msfc:%s, af:%s, wv:%s, cascading:%s}",
    self._allow_negative_prices,
    not (shape_hourly_kwargs.get("enforce_positivity", False)),  # adapté à l'API réelle
    ...
)
```

**Décision sidecar D-A2-4 (tranchée ici) :** Ne pas créer de sidecar séparé `assembler.meta.parquet`. L'audit-trail est dans les logs (INFO). Les ctor args des sous-composants ont leurs propres sidecars si nécessaire. Justification : le master flag est un observateur info-only (D-A2-2), pas un paramètre de fit persisté. Ajouter un sidecar pour une valeur read-only-at-init serait du over-engineering.

### Pattern 6 : `fit_peak_ratios` backward-compat shim

**Décision D-A4-2 (tranchée ici) — pattern exact :**

```python
# Source: CONTEXT.md D-A4-2
import warnings

def fit_peak_ratios(
    self,
    spot_history: pd.DataFrame,
) -> "ContractCascader":
    """DEPRECATED. Utiliser fit_peak_spreads() à la place.

    Shim backward-compat : appelle fit_peak_spreads() si spot_history fourni,
    et dérive peak_base_ratios_ depuis peak_base_spreads_ pour les callers legacy
    qui lisent self.peak_base_ratios_ directement.
    """
    warnings.warn(
        "ContractCascader.fit_peak_ratios() est deprecated. "
        "Utiliser fit_peak_spreads() qui calibre des spreads en €/MWh "
        "(sign-invariant pour les forwards négatifs). "
        "Migration: remplacer fit_peak_ratios(spot) par fit_peak_spreads(spot).",
        DeprecationWarning,
        stacklevel=2,
    )
    # Appel transparent — le caller a spot_history (c'est le seul arg requis)
    return self.fit_peak_spreads(spot_history)
```

**Justification du choix shim transparent vs `NotImplementedError` :** Les deux callsites dans `production_phases.py` (lignes 344 et 644) passent déjà `spot_history` et peuvent être migrés. `NotImplementedError` bloquerait le pipeline avant migration. Le shim transparent préserve le comportement opérationnel le temps d'une migration explicite vers `fit_peak_spreads`.

### Pattern 7 : `_generate_phase05_fixture.py` (convention 5bis-B)

```python
# Source: CONTEXT.md D-A4-5 + convention tests/fixtures/_generate_bowl_fixture.py (5bis-B)
# Cal'27=30 €/MWh, July M-07'27=20 €/MWh (dépressé), autres months positifs typiques EEX
# seed=42, déterministe
```

### Anti-Patterns to Avoid

- **Anti-pattern : modifier `_enforce_mean_constraints` retour sans passer `enforce_positivity`** : la fonction a deux sites de floor (ligne 131 dans `smooth_base_prices`, ligne 203 dans `_enforce_mean_constraints`). Si `enforce_positivity` n'est pas passé en paramètre à `_enforce_mean_constraints`, le floor ligne 203 reste actif même si ligne 131 est conditionnel. Les deux doivent être conditionnés par le même flag.

- **Anti-pattern : appliquer `delta_wv` AVANT le `shape_freedom` damping** : `f_WV` est actuellement dans `shape_freedom["f_WV"]` (ligne 389 d'`assembler.py`). Si on applique `delta_wv` avant le damping, on ne respecte pas la sémantique du horizon decay déjà dans `WaterValueCorrection.apply()`. Solution : passer `f_WV` à pass-through (1.0) dans le path delta-additif, et ne plus le mettre dans `shape_freedom`.

- **Anti-pattern : utiliser `B.abs()` sur le B post-MSFC smooth** : `compute_delta_wv` reçoit `B_smooth` (après MSFC). Dans `assembler.build()`, `B` est au niveau `_resolve_base` + `smooth_base_prices`. Il faut passer `B` (le smoothed, avant les facteurs shape) à `compute_delta_wv`.

- **Anti-pattern : ne pas propager `converged=False` pour le clip m_factor** : NEG-02 littéral exige que le clip soit visible dans la convergence status. Quand `enforce_m_factor_floor=True` et que le clip frappe, `converged` doit être forcé `False` même si `max_abs_residual < tol`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Signed clamp bounds | Logique ad-hoc per-signe | `margin = 0.5 * np.ptp(y_knots)` (D-A1-2) | np.ptp est toujours ≥ 0, la formule marche avec knots mixtes, all-neg, all-pos |
| f_WV signe | Re-implement WaterValueCorrection.apply | `compute_delta_wv = (f_wv - 1) * B.abs()` | Réutilise le `horizon_decay`, `season_sensitivity_`, `beta_wv_` calibrés |
| Test NEG-05 bout-en-bout | Full PFC build avec Cal annuel négatif | Monthly forward négatif (July -2 €/MWh) via `test_msfc_signed_monthly_repricing` | Cal annuel négatif est non-réaliste (D-A4-7 CONTEXT.md) |
| Sidecar assembler | `assembler.meta.parquet` nouveau | Pas de sidecar — log INFO audit-trail (D-A2-4) | Master flag = info-only, pas un paramètre de fit |

---

## Dry-Run Results

### D-A3-3 : Écart baseline multiplicatif vs additif

Formule analytique : `diff = B × (f_H×f_W − 1) × (f_wv − 1)`

| B (€/MWh) | f_H×f_W | f_wv | Diff (€/MWh) | > 1e-12 ? |
|-----------|---------|------|--------------|-----------|
| 50 | 1.08 | 1.05 | 0.200 | OUI |
| 80 | 1.12 | 1.15 | 1.440 | OUI |
| 50 | 1.00 | 1.05 | 0.000 | NON (cas f_H×f_W=1 exact) |

**Conclusion :** Sur le baseline_5bisA (forwards positifs, f_H×f_W typiquement ≠ 1.0), la diff est 0.2–1.4 €/MWh ≫ 1e-12. **Nouvelle baseline `baseline_pfc_seed42_phase05.parquet` obligatoire** (convention D-A4-9 5bis-B). La baseline 5bis-A `baseline_pfc_seed42.parquet` reste testable via `enforce_*=True` sur tous les 4 ctor args. [VERIFIED: code+arithmetic]

### D-A1-2 : Validation clamp signed-aware

| y_knots | margin | Nouvelles bornes | Anciennes bornes | Correct ? |
|---------|--------|-----------------|-----------------|-----------|
| [-5, 30, 25, 40] | 22.5 | [-27.5, 62.5] | [-2.5, 80] | Nouveau ✓ |
| [-30, -20, -25] | 5.0 | [-35.0, -15.0] | [-15.0, -40.0] | Ancien ✗ (bornes inversées) |
| [20, 35, 30, 50] | 15.0 | [5.0, 65.0] | [10.0, 100.0] | Les deux OK pour all-positive |

**Conclusion :** L'ancienne formule est mathématiquement cassée pour les knots all-negative. La nouvelle formule est correcte dans tous les cas. [VERIFIED: code]

### D-A4-2 : Callsite audit `fit_peak_ratios`

| Fichier | Ligne | Caller | spot_history accessible ? | Migration path |
|---------|-------|--------|--------------------------|----------------|
| `production_phases.py` | 344 | `cascader_ch.fit_peak_ratios(inputs.epex_ch)` | Oui (`inputs.epex_ch` est un DataFrame EPEX) | Remplacer par `fit_peak_spreads(inputs.epex_ch)` |
| `production_phases.py` | 644 | `cascader.fit_peak_ratios(spec.epex_df)` | Oui (`spec.epex_df`) | Remplacer par `fit_peak_spreads(spec.epex_df)` |
| Tests `test_cascading.py` | — | Aucun callsite `fit_peak_ratios` trouvé | N/A | Pas de migration test nécessaire |
| `rolling_update.py` | — | Aucun callsite trouvé | N/A | — |
| `autoresearch_eval.py` | — | Aucun callsite `fit_peak_ratios` | N/A | — |

**Conclusion :** 2 callsites dans `production_phases.py`, tous avec `spot_history` disponible. Le shim transparent (DeprecationWarning + redirection vers `fit_peak_spreads`) est suffisant. [VERIFIED: grep codebase]

---

## Common Pitfalls

### Pitfall 1 : Deux floors dans MSFC — l'un caché dans `_enforce_mean_constraints`

**What goes wrong:** `smooth_base_prices` appelle `_enforce_mean_constraints`, qui a SON PROPRE `return np.maximum(result, 1.0)` à la ligne 203 (fin de la fonction). Si le plan ne conditionne que le floor ligne 131 dans `smooth_base_prices`, le floor ligne 203 reste actif silencieusement.

**Why it happens:** `_enforce_mean_constraints` est une fonction helper interne qui contient son propre floor final (ligne 203). Il faut passer `enforce_positivity` à cette fonction et la conditionner à l'intérieur aussi.

**How to avoid:** Le plan doit explicitement mentionner les DEUX modifications :
1. `smooth_base_prices` ligne 131 : `B_smooth = np.maximum(B_smooth, 1.0)` → conditionnel
2. `_enforce_mean_constraints` ligne 203 : `return np.maximum(result, 1.0)` → conditionnel
Et `_enforce_mean_constraints` doit recevoir `enforce_positivity: bool` en argument supplémentaire.

**Warning signs:** `test_msfc_signed_monthly_repricing` échoue avec `mean(B_smooth) ≈ 1.0` au lieu de `-2.0` même si le floor ligne 131 est retiré.

### Pitfall 2 : `f_WV` dans `shape_freedom` damping (line 389 assembler.py)

**What goes wrong:** La ligne `f_WV = 1.0 + (f_WV - 1.0) * shape_freedom["f_WV"]` (assembler.py ligne 389) applique un damping à `f_WV` AVANT que `price_raw` soit calculé. Dans le path delta-additif, si on appelle `compute_delta_wv` sur le `f_wv` non-dampé, puis qu'on essaie aussi d'utiliser le `f_WV` dampé dans `price_raw`, on double-applique la correction.

**Why it happens:** Le path existant traite `f_WV` comme un facteur shape (dampé avec les autres), mais le delta-additif le calcule directement depuis `WaterValueCorrection.apply()` (qui contient déjà son propre `horizon_decay`).

**How to avoid:** Dans le path delta-additif, la ligne 389 doit passer `f_WV` en pass-through (ne pas appliquer `shape_freedom["f_WV"]`), ou simplement ne pas calculer `f_WV` séparément (tout déléguer à `compute_delta_wv`). Le `shape_freedom["f_WV"]` est redondant avec le `horizon_decay` dans `WaterValueCorrection`.

**Warning signs:** Le `test_water_value_delta_sign_invariant` passe mais le test d'acceptance SC #2 montre une amplitude trop faible (double-damping de f_WV).

### Pitfall 3 : `converged=False` non propagé pour m_factor floor (NEG-02)

**What goes wrong:** La ligne 517 `m_factor = np.maximum(m_factor, 0.1)` est appliquée AVANT le calcul de `P = S * m_factor`. Si le floor frappe mais que `max_abs_residual < tol`, `converged` reste `True`. NEG-02 exige que `converged=False` soit propagé quand le floor frappe.

**Why it happens:** La logique de `converged` actuelle (ligne 546) est basée sur `max_abs_residual > self.tol`. Le floor sur `m_factor` n'affecte pas directement les résidus de repricing.

**How to avoid:** Ajouter une variable `floor_applied = False` avant ligne 517. Si le clip frappe (`m_factor.min() < 0.1` avant le clip), `floor_applied = True`. Ensuite : `if floor_applied: converged = False; logger.warning("m_factor floor applied at %d timestamps", ...)`.

**Warning signs:** `test_arbitrage_free_signed_target` passe `converged=True` sur un input qui aurait dû clipper le m_factor.

### Pitfall 4 : `synthesize_peak_prices` sans `peak_base_spreads_` fitté

**What goes wrong:** Si `fit_peak_spreads` n'a pas été appelé mais `allow_negative_peak=True` (le défaut), `synthesize_peak_prices` ne trouve pas `peak_base_spreads_` et tombe dans un fallback.

**Why it happens:** L'attribut `peak_base_spreads_` n'existe pas avant l'appel à `fit_peak_spreads`. La logique existante pour `peak_base_ratios_` a un fallback explicite (default_ratio = 1.05, ligne 359).

**How to avoid:** Prévoir le même fallback pour `peak_base_spreads_` : `if not hasattr(self, "peak_base_spreads_") or not self.peak_base_spreads_: default_spread = 5.0; logger.warning("No fitted peak spreads — using default %.1f €/MWh", default_spread); spreads = {m: default_spread for m in range(1, 13)}`.

**Warning signs:** `test_cascading_spread_signed_base` échoue avec `AttributeError: 'ContractCascader' has no attribute 'peak_base_spreads_'`.

### Pitfall 5 : Fixture forwards_phase05 avec July M-07'27 = -2 vs = 20

**What goes wrong:** Deux valeurs July différentes selon les tests. Le `test_msfc_signed_monthly_repricing` (unit, in-test) utilise `-2.0` (forward négatif). Le `test_phase05_summer_bowl_negative_acceptance` (fixture parquet) utilise `20.0` (forward positif, mais bowl 5bis-B crée des heures négatives).

**Why it happens:** D-A4-7 (test NEG-05 reformulé : monthly forward = -2 €/MWh) s'applique aux unit tests in-test. D-A4-5 (acceptance SC #2 : July = 20 €/MWh dépressé, PFC_LT_USE_SEASONAL_HOURLY_SHAPE + bowl deepening → h13 Sunday < -20) utilise la fixture parquet.

**How to avoid:** Documenter clairement dans les deux tests : unit test = forwards négatifs in-test (pas de fixture parquet), acceptance test = fixture parquet avec forwards positifs + bowl 5bis-B.

---

## Code Examples

### floor conditionnel MSFC (lignes 131 et 203)

```python
# Source: VERIFIED msfc_spline.py:39 + CONTEXT.md D-A2-1

# smooth_base_prices signature:
def smooth_base_prices(
    idx: pd.DatetimeIndex,
    base_prices: dict[str, float],
    B_flat: pd.Series,
    enforce_positivity: bool = False,  # NEW
) -> pd.Series:
    ...
    B_smooth = _enforce_mean_constraints(
        idx_zh, x_target, x_knots, y_knots, month_keys,
        base_only, B_smooth_raw, start_ts, max_iter=10,
        enforce_positivity=enforce_positivity,  # PROPAGATED
    )
    # Ligne 131: SUPPRIMÉE (floor dans _enforce_mean_constraints)
    # (l'ancien: B_smooth = np.maximum(B_smooth, 1.0) ← retiré)
    ...

# _enforce_mean_constraints signature extension:
def _enforce_mean_constraints(
    ...,
    enforce_positivity: bool = False,  # NEW
) -> np.ndarray:
    ...
    # Ligne 203: conditionnel
    result = interpolator(x_target)
    return np.maximum(result, 1.0) if enforce_positivity else result
```

### ArbitrageFreeCalibrator ctor + clip conditionnel (ligne 517)

```python
# Source: VERIFIED arbitrage_free.py:500-524 + CONTEXT.md D-A2-1 / NEG-02

class ArbitrageFreeCalibrator:
    def __init__(self, ..., enforce_m_factor_floor: bool = False):  # NEW, default False
        ...
        self.enforce_m_factor_floor = enforce_m_factor_floor

    # Dans calibrate() autour de ligne 517:
    floor_applied = False
    if self.enforce_m_factor_floor:
        clipped = np.maximum(m_factor, 0.1)
        floor_applied = bool(np.any(clipped != m_factor))
        if floor_applied:
            logger.warning(
                "m_factor floor 0.1 applied at %d timestamps (enforce_m_factor_floor=True)",
                np.sum(clipped != m_factor),
            )
            converged = False  # NEG-02: propagate converged=False
        m_factor = clipped
```

### compute_delta_wv dans WaterValueCorrection

```python
# Source: CONTEXT.md D-A3-1 / D-A3-4
class WaterValueCorrection:
    def __init__(self, enforce_floor: bool = False):  # NEW, default False
        self.enforce_floor = enforce_floor
        ...

    def apply(self, timestamps, calendar_df, hydro_forecast=None):
        ...
        # Lignes 394, 407: conditionnelles
        if self.enforce_floor:
            raw_f_wv = raw_f_wv.clip(lower=F_WV_FLOOR, upper=F_WV_CAP)
        ...
        if self.enforce_floor:
            f_wv = f_wv.clip(lower=F_WV_FLOOR, upper=F_WV_CAP)
        return f_wv

    def compute_delta_wv(
        self,
        B_smooth: pd.Series,
        fill_df: pd.DataFrame | None,
        calendar_df: pd.DataFrame,
    ) -> pd.Series:
        if self.enforce_floor:
            raise ValueError("compute_delta_wv incompatible avec enforce_floor=True")
        f_wv = self.apply(B_smooth.index, calendar_df, fill_df)
        delta = (f_wv - 1.0) * B_smooth.abs()
        delta.name = "delta_wv"
        return delta
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `P = B × f_H × f_W × f_wv` (multiplicatif) | `P = B × f_H × f_W + delta_wv` (delta additif) | Phase 5 Plan 05-02 | Sign-invariant : f_wv scarcity préserve la sémantique même pour B < 0 |
| `result = price * ratio` (peak synthèse) | `result = base_price + spread` (€/MWh) | Phase 5 Plan 05-03 | Sign-invariant : spread € ne change pas de signe avec la base |
| `np.maximum(B, 1.0)` silencieux | `enforce_positivity=False` (default) | Phase 5 Plan 05-01 | Négatifs possibles à h13 dimanche été |
| Clamp `[min*0.5, max*2.0]` | Clamp `[min-margin, max+margin]` | Phase 5 Plan 05-01 | Fix pour knots tous négatifs (ancienne formule avait des bornes inversées) |

**Deprecated/outdated:**
- `smooth_base_prices` sans kwarg `enforce_positivity` : produit des prix plancher 1 €/MWh (comportement legacy conservé via `enforce_positivity=True`).
- `ContractCascader.fit_peak_ratios()` : deprecated Phase 5, remplacer par `fit_peak_spreads()`. Callsites : `production_phases.py:344` et `production_phases.py:644`.
- REQUIREMENTS.md NEG-05 "Cal'27 = -10 €/MWh" : reformulé en "monthly forward négatif July M-07'27 = -2 €/MWh" (D-A4-7, Plan 05-01).

---

## Runtime State Inventory

> Phase 5 est un math change + refactor. Pas de rename/rebrand. Inventaire minimal requis.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | `tests/fixtures/baseline_pfc_seed42.parquet` (5bis-A frozen) | Préservé tel quel. Testable via `enforce_*=True` après Phase 5. |
| Stored data | `tests/fixtures/baseline_pfc_seed42_bowl.parquet` (5bis-B frozen) | Préservé tel quel. Gating du test acceptance SC #2. |
| Build artifacts | `tests/fixtures/baseline_pfc_seed42_phase05.parquet` | À générer en Plan 05-03 (n'existe pas encore). |
| Live service config | Aucun service externe à reconfigurer | — |
| OS-registered state | Aucun | — |
| Secrets/env vars | `PFC_LT_ALLOW_NEGATIVE_PRICES` : nouveau, pas encore existant dans env | Conftest autouse heredera automatiquement (pattern PFC_LT_*). |
| Callsites `fit_peak_ratios` | `production_phases.py:344,644` | Deprecation warning émise par le shim. Migration explicite optionnelle dans Plan 05-03. |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `_enforce_mean_constraints` est sign-invariante par construction (correction = `error * 0.8` fonctionne identique en signed) | Standard Stack / Architecture | Si faux : NEG-05 test échoue, mean constraint pas satisfaite pour monthly négatif. Vérifiable via `test_msfc_signed_monthly_repricing`. |
| A2 | Le `horizon_decay` dans `WaterValueCorrection.apply()` est suffisant comme remplacement du `shape_freedom["f_WV"]` damping | Architecture Pattern 3 | Si faux : courbe WV trop agressive à far horizon. Vérifiable via telemetry D-A3-5. |
| A3 | `ArbitrageFreeCalibrator` mode "multiplicative" est la seule configuration active dans le pipeline (pas le mode "additive") | Architecture Patterns | Si faux : `enforce_m_factor_floor` n'a pas d'effet en mode additive. Vérifiable : `grep -n "mode=" pfc_shaping/pipeline/` — non vérifié dans cette session. |

**Note A3 :** La ligne `mode = "multiplicative"` dans `ArbitrageFreeCalibrator.__init__` est le default (ASSUMED). Une vérification grep dans les pipelines serait prudente avant Plan 05-01.

---

## Open Questions

1. **Mode ArbitrageCalibrator dans les pipelines**
   - What we know: Le code arbitrage_free.py supporte deux modes ("multiplicative", "additive"). L'`enforce_m_factor_floor` n'a d'effet qu'en mode "multiplicative" (ligne 517 est dans la branche `if self.mode == "multiplicative"`).
   - What's unclear: Le mode utilisé dans `autoresearch.py` et `production_phases.py`.
   - Recommendation: Plan 05-01 doit grep `ArbitrageFreeCalibrator(` dans tous les pipelines et vérifier le mode avant d'implémenter le floor conditionnel.

2. **`shape_freedom["f_WV"]` — à neutraliser ou laisser en place ?**
   - What we know: La ligne 389 `f_WV = 1.0 + (f_WV - 1.0) * shape_freedom["f_WV"]` dampe f_WV. Dans le path delta-additif, `compute_delta_wv` utilise `f_wv` non-dampé (issu directement de `WaterValueCorrection.apply()` qui a son propre `horizon_decay`).
   - What's unclear: Est-ce que le `shape_freedom["f_WV"]` est un "second damping" en plus du `horizon_decay`, ou est-il redondant ?
   - Recommendation: Plan 05-02 doit inspecter `_shape_freedom` (assembler.py:803-844) pour le knot schedule de `f_WV` et décider si neutraliser (pass-through 1.0) dans le path delta-additif.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| pytest | Tests Phase 5 | ✓ | 8.4.2 | — |
| numpy | MSFC, WV, cascade math | ✓ | (projet) | — |
| pandas | Séries temporelles | ✓ | 2.3.3 | — |
| scipy | PchipInterpolator | ✓ | (projet) | — |
| pyarrow | Lecture/écriture fixtures parquet | ✓ | (projet) | — |

**Tests verts au démarrage :** 258 passed, 3 skipped (vérification gsd-phase-researcher 2026-05-19).

---

## Validation Architecture

> `workflow.nyquist_validation = true` dans `.planning/config.json` — section obligatoire.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.4.2 |
| Config file | `pytest.ini` ou pyproject.toml (projet existant) |
| Quick run command | `pytest tests/test_phase05_negative_prices.py -x -q` |
| Full suite command | `pytest tests/ -q --tb=short` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| NEG-01 | `smooth_base_prices` avec monthly forward -2 €/MWh → mean(B_smooth) ≈ -2.0 (atol=0.01) | unit | `pytest tests/test_phase05_negative_prices.py::test_msfc_signed_monthly_repricing -x` | ❌ Wave 0 |
| NEG-02 | `ArbitrageFreeCalibrator` avec target négatif → `converged=True`, `max_abs_residual < tol` | unit | `pytest tests/test_phase05_negative_prices.py::test_arbitrage_free_signed_target -x` | ❌ Wave 0 |
| NEG-03 | `compute_delta_wv(B_smooth=-10, f_wv=1.20)` → `delta_wv = +2.0` (scarcity = prix moins négatif) | unit | `pytest tests/test_phase05_negative_prices.py::test_water_value_delta_sign_invariant -x` | ❌ Wave 0 |
| NEG-04 | `ContractCascader.synthesize_peak_prices({'2027': -10}, peak_base_spreads_={...: +5})` → `result['2027-Peak'] = -5.0` | unit | `pytest tests/test_phase05_negative_prices.py::test_cascading_spread_signed_base -x` | ❌ Wave 0 |
| NEG-05 (reformulé) | Idem NEG-01 (test_msfc_signed_monthly_repricing couvre la reformulation D-A4-7) | unit | (inclus dans NEG-01) | ❌ Wave 0 |
| SC #2 ROADMAP | fixture forwards_phase05, PFC_LT_USE_SEASONAL_HOURLY_SHAPE=1, PFC_LT_ALLOW_NEGATIVE_PRICES=1 → `pfc[Sunday, h13, July 2027].mean() < -20.0` | acceptance | `pytest tests/test_phase05_negative_prices.py::test_phase05_summer_bowl_negative_acceptance -x` | ❌ Wave 0 |
| SC #5 ROADMAP (régression) | `assert_frame_equal(build(forwards_phase05), baseline_pfc_seed42_phase05, atol=1e-12, rtol=0)` | regression | `pytest tests/test_phase05_negative_prices.py::test_phase05_baseline_regression -x` | ❌ Wave 0 |

### Spécifications des 4 unit tests + 1 acceptance (D-A4-4/D-A4-5)

**test_msfc_signed_monthly_repricing (NEG-01 / NEG-05)**
```python
# base_prices = {'2027': 30.0, '2027-01': 35.0, ..., '2027-07': -2.0, '2027-08': 25.0, ...}
# (autres months positifs typiques)
# Appel smooth_base_prices(idx_2027, base_prices, B_flat, enforce_positivity=False)
# Assert: abs(B_smooth[mask_july_2027].mean() - (-2.0)) < 0.01
```

**test_arbitrage_free_signed_target (NEG-02)**
```python
# ArbitrageFreeCalibrator(enforce_m_factor_floor=False)
# Contracts avec target monthly négatif (e.g. July 2027 = -5 €/MWh)
# Assert: result.converged == True AND result.max_abs_residual < calibrator.tol
```

**test_water_value_delta_sign_invariant (NEG-03)**
```python
# B_smooth = pd.Series([-10.0] * N)  # prix négatifs
# f_wv = 1.20 → delta_wv = (1.20 - 1) * |-10| = +2.0   (scarcity → moins négatif)
# f_wv = 0.80 → delta_wv = (0.80 - 1) * |-10| = -2.0   (abundance → plus négatif)
# Vs legacy multiplicatif: f_wv=1.20 × B=-10 = -12 (scarcity rendait le prix plus négatif — WRONG)
```

**test_cascading_spread_signed_base (NEG-04)**
```python
# cascader.peak_base_spreads_ = {m: 5.0 for m in range(1, 13)}
# base_prices = {'2027': -10.0}
# cascader.allow_negative_peak = True
# result = cascader.synthesize_peak_prices(base_prices)
# assert result['2027-Peak'] == pytest.approx(-10.0 + 5.0)  # = -5.0
```

**test_phase05_summer_bowl_negative_acceptance (SC #2 — gated par 5bis-B)**
```python
# Skipé si baseline_pfc_seed42_bowl.parquet absent ou bowl amplitude < seuil
# Charge forwards_phase05_seed42.parquet (Cal'27=30, July=20, seed=42)
# PFC_LT_USE_SEASONAL_HOURLY_SHAPE=1, PFC_LT_ALLOW_NEGATIVE_PRICES=1
# Assert: pfc.loc[(pfc.type_jour=='Dimanche') & (pfc.heure==13) & (pfc.index.month==7), 'price_shape'].mean() < -20.0
```

### Sampling Rate
- **Per task commit:** `pytest tests/test_phase05_negative_prices.py -x -q`
- **Per wave merge:** `pytest tests/ -q --tb=short`
- **Phase gate:** Suite complète verte avant `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_phase05_negative_prices.py` — à créer en Plan 05-03 (5 tests : 4 unit + 1 acceptance)
- [ ] `tests/fixtures/_generate_phase05_fixture.py` — à créer en Plan 05-03
- [ ] `tests/fixtures/forwards_phase05_seed42.parquet` — à générer en Plan 05-03
- [ ] `tests/fixtures/baseline_pfc_seed42_phase05.parquet` — à générer en Plan 05-03 (après Plans 01+02)

*(Frameworks pytest et fixtures tests/conftest.py déjà en place depuis 5bis-A.)*

---

## Sources

### Primary (HIGH confidence)
- `pfc_shaping/lt/model/msfc_spline.py` — lecture directe du code source, lignes 120, 131, 203 identifiées [VERIFIED: code]
- `pfc_shaping/calibration/arbitrage_free.py` — ligne 517 identifiée [VERIFIED: code]
- `pfc_shaping/lt/model/water_value.py` — lignes 60, 394, 407 identifiées ; classe WaterValueCorrection.__init__ sans enforce_floor [VERIFIED: code]
- `pfc_shaping/calibration/cascading.py` — `ContractCascader` (pas `BlockCascading`) est la classe réelle dans le code ; `fit_peak_ratios` ligne 279, `synthesize_peak_prices` ligne 342 [VERIFIED: code]
- `pfc_shaping/lt/model/assembler.py` — lignes 170-201 (init sans master flag), 389 (f_WV damping), 403 (price_raw formule multiplicative) [VERIFIED: code]
- `.planning/phases/05-msfc-log-prix-retire-silent-floors/05-CONTEXT.md` — toutes les décisions D-A1-1..D-A5-1 [CITED: fichier projet]
- `.planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-REVIEWS.md §1-2` — tolerance contract atol=1e-12, rtol=0 [CITED: fichier projet]
- `tests/conftest.py` — autouse PFC_LT_* env hygiene déjà en place [VERIFIED: code]

### Secondary (MEDIUM confidence)
- Dry-run arithmétique `diff = B × (f_H×f_W - 1) × (f_wv - 1)` — calculé en session (0.2–1.4 €/MWh) [VERIFIED: arithmetic]
- Dry-run clamp signed-aware — calculé en session, formule validée pour all-negative, mixed, all-positive [VERIFIED: arithmetic]
- Grep callsites `fit_peak_ratios` — 2 callsites en `production_phases.py:344,644`, aucun dans `rolling_update.py` ni `tests/` [VERIFIED: grep]

### Tertiary (LOW confidence)
- Aucun

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — pas de nouveau package, code source lu directement
- Architecture: HIGH — floor locations confirmées par lecture du code
- Pitfalls: HIGH — identifiés par analyse structurelle du code (deux floors dans MSFC, double-damping WV)
- Dry-run D-A3-3: HIGH — calcul arithmétique vérifié (0.2–1.4 €/MWh ≫ 1e-12)

**Research date:** 2026-05-19
**Valid until:** 2026-06-19 (30 jours — codebase stable, pas de dépendances externes)

---

## RESEARCH COMPLETE

**Phase:** 05 - msfc-log-prix-retire-silent-floors
**Confidence:** HIGH

### Key Findings

1. **Deux floors MSFC, pas un** : `smooth_base_prices` a un floor ligne 131 ET `_enforce_mean_constraints` a un floor ligne 203. Les deux doivent être conditionnés par `enforce_positivity` propagé en paramètre — pitfall critique pour Plan 05-01.

2. **Dry-run D-A3-3 concluant** : L'écart multiplicatif vs additif est 0.2–1.4 €/MWh ≫ 1e-12. Nouvelle baseline `baseline_pfc_seed42_phase05.parquet` obligatoire en Plan 05-03.

3. **Callsite audit `fit_peak_ratios`** : 2 callsites dans `production_phases.py` uniquement (lignes 344, 644), tous avec `spot_history` disponible. Shim transparent `DeprecationWarning + redirection fit_peak_spreads` est la bonne approche. Tests et autres pipelines ne font pas appel à `fit_peak_ratios`.

4. **Sidecar D-A2-4 résolu** : Pas de nouveau sidecar. Le master flag `PFC_LT_ALLOW_NEGATIVE_PRICES` est loggué INFO à l'init (audit-trail). Les ctor args sont la véritable surface API.

5. **Clamp signed-aware validé** : L'ancienne formule `[min*0.5, max*2.0]` est mathématiquement cassée pour knots all-negative (bornes inversées). La nouvelle `[min - margin, max + margin]` avec `margin = 0.5 * ptp` est correcte dans tous les cas.

### File Created
`.planning/phases/05-msfc-log-prix-retire-silent-floors/05-RESEARCH.md`

### Confidence Assessment

| Area | Level | Reason |
|------|-------|--------|
| Standard Stack | HIGH | Pas de nouveau package, code source vérifié |
| Architecture (floor locations) | HIGH | Lecture directe des 4 sites dans le code |
| Dry-run D-A3-3 (baseline decision) | HIGH | Calcul arithmétique, résultat 0.2 EUR/MWh |
| Callsite audit (fit_peak_ratios) | HIGH | Grep vérifié dans tous les fichiers .py |
| Pitfalls | HIGH | Identifiés par analyse structurelle du code |

### Open Questions
1. Mode ArbitrageCalibrator dans `autoresearch.py` et `production_phases.py` : vérifier que c'est "multiplicative" avant d'implémenter `enforce_m_factor_floor` (A3 dans Assumptions Log).
2. `shape_freedom["f_WV"]` knot schedule vs `horizon_decay` de `WaterValueCorrection` : Plan 05-02 doit décider si neutraliser `f_WV` dans `shape_freedom` sur le path delta-additif.

### Ready for Planning
Research complete. Planner peut créer les 3 PLAN.md.
