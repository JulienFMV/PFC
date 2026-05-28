"""
structural_tests.py
-------------------
Pillar 1 — Hildmann 2013 (ETH Zurich) structural quality tests for HPFC.

**SC#1 UNIQUE GATE Phase 10** (D-A0-3) : sous Config 4 (bowl ON + floors OFF =
production target post-5bis-B+5), les 4 tests structurels ci-dessous doivent
PASS sur l'agrégat des 24 vintages 2024-01..2025-12. C'est le verrou qui
autorise le flip ON par défaut de `PFC_LT_USE_SEASONAL_HOURLY_SHAPE`
(D-FLIP-1 dans PROJECT.md).

Les 4 tests
-----------
1. **Arbitrage-freeness** (`test_arb_free`)
   `|mean(PFC over period) - forward| < tol €/MWh` pour chaque Cal/Q/M tradé.
2. **Holiday/weekend pattern** (`test_holiday_weekend`)
   `mean(PFC | weekend ∪ holidays_VS) / mean(PFC | weekday) ∈ HOLIDAY_WEEKEND_RANGE`
   où `HOLIDAY_WEEKEND_RANGE = (0.65, 0.95)` (frozen Plan 10-01 NOTES Pitfall 1
   IF branch, ratio mesuré 0.8033 sur EPEX CH 2019-2023).
3. **Seasonal profile** (`test_seasonal_profile`)
   `pearsonr(monthly_PFC_signature, monthly_EPEX_hist_signature) > 0.85`.
4. **Continuity** (`test_continuity`)
   `max(|PFC[end of M] - PFC[start of M+1]|) < 2.0 €/MWh` sur toutes
   les frontières mensuelles.

Convention
----------
Chaque fonction retourne un `TestResult(passed, observed, threshold, details)`
sérialisable pour le scorecard markdown. Aucune fonction ne crash sur edge
case : forwards vide → max_dev=0 (passed=True), mean weekday near-zero →
degenerate=True (passed=False), single-month → max_jump=0 (passed=True).

Traçabilité
-----------
- `HOLIDAY_WEEKEND_RANGE` source : `.planning/phases/10-pfc-fmv-quality-scorecard/10-01-NOTES.md`
  §RESEARCH Pitfall 1 (branch IF, ratio empirique 0.8033 ∈ [0.65, 0.95]).
- Formule frozen ex-ante (C2 REVIEWS, commit 376f1e4 PRECEDE 0e955a2f).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import holidays
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from pfc_shaping.calibration.cascading import parse_key


# ---------------------------------------------------------------------------
# Module-level constants (sourced from Plan 10-01 NOTES Pitfall 1 decision)
# ---------------------------------------------------------------------------

#: Threshold Hildmann Pillar 1.2 holiday/weekend ratio range.
#:
#: Frozen ex-ante par C2 REVIEWS Plan 10-01 (commit 376f1e4) AVANT mesure
#: empirique (commit 0e955a2f → ratio mesuré 0.8033 ∈ [0.65, 0.95] → branch IF
#: → research default confirmé). Traçabilité complète dans
#: `.planning/phases/10-pfc-fmv-quality-scorecard/10-01-NOTES.md` §Pitfall 1.
#:
#: **Ne JAMAIS recalibrer post-mesure**. Si un test fail sur cette range,
#: investiguer la PFC (bowl, floors) — pas tweaker la range.
HOLIDAY_WEEKEND_RANGE: tuple[float, float] = (0.65, 0.95)


# ---------------------------------------------------------------------------
# TestResult dataclass — output sérialisable des 4 fonctions Hildmann
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    """Résultat d'un test Hildmann.

    Attributes
    ----------
    passed
        True si le test passe le seuil SOTA-grade.
    observed
        Valeur empirique mesurée (max_dev pour arb-free, ratio pour
        holiday-weekend, pearson_r pour seasonal, max_jump_obs pour continuity).
    threshold
        Seuil/range cible. `float` si seuil simple (arb-free tol, max_jump,
        min_corr), `tuple[float,float]` si range (holiday-weekend range).
    details
        Dict de diagnostic (n_obs, per-key breakdown, degenerate flag, etc.).
        Sérialisable JSON-style pour le scorecard markdown.
    """

    passed: bool
    observed: float
    threshold: tuple[float, float] | float
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper privé : masque pour une période (Cal/Q/M) sur un DatetimeIndex UTC
# ---------------------------------------------------------------------------


def _period_mask(
    idx_utc: pd.DatetimeIndex,
    product_type: str,
    year: int,
    sub: int | None,
) -> np.ndarray:
    """Booléen aligné sur `idx_utc` (True = dans la période demandée).

    Parameters
    ----------
    idx_utc
        DatetimeIndex tz-aware (UTC en pratique).
    product_type
        Output de `parse_key` : `"Cal"` / `"Quarter"` / `"Month"` / variantes
        suffixed `_Peak` / `_Offpeak` (ces dernières retournent un mask vide
        ici — Pillar 1.1 arb-free ne traite que Base, pas Peak/Offpeak).
    year
        Calendar year (e.g. 2025).
    sub
        Quarter index 1-4 si Quarter, month index 1-12 si Month, None si Cal.

    Returns
    -------
    np.ndarray[bool]
        Mask aligné sur idx_utc.

    Notes
    -----
    Forwards (Cal/Q/M base) sont tradés contre des **périodes de livraison
    en calendrier local** (Europe/Zurich convention trader EEX/EPEX), pas
    UTC. Une PFC indexée UTC doit être convertie en heure locale AVANT
    bucketing : sinon ~1 h par frontière Cal/Q/M tombe dans le mauvais
    bucket (e.g. 2024-12-31 23:00 UTC = 2025-01-01 00:00 CET appartient
    au Cal 2025, pas Cal 2024). Avec `tol = 0.01 €/MWh` de
    `test_arb_free`, ce biais ~`Δ_price / 720` €/MWh par mois suffit à
    fabriquer un faux fail de la gate SC#1. Cf. CR-01 (REVIEW 2026-05-21)
    + `block_masks.py` (même convention) + `test_continuity`
    (`tz_convert("Europe/Zurich")` ligne 456).
    """
    if idx_utc.tz is None:
        raise ValueError("_period_mask: idx must be tz-aware")
    # Forwards bucket by LOCAL calendar (Europe/Zurich trader convention).
    idx_local = idx_utc.tz_convert("Europe/Zurich")
    if product_type == "Cal":
        return np.asarray(idx_local.year == year, dtype=bool)
    if product_type == "Quarter":
        return np.asarray(
            (idx_local.year == year) & (idx_local.quarter == sub), dtype=bool
        )
    if product_type == "Month":
        return np.asarray(
            (idx_local.year == year) & (idx_local.month == sub), dtype=bool
        )
    # Peak/Offpeak ou type inconnu → mask vide (skip silencieux)
    return np.zeros(len(idx_utc), dtype=bool)


def _expected_period_points(
    idx_utc: pd.DatetimeIndex,
    product_type: str,
    year: int,
    sub: int | None,
) -> int | None:
    """Expected number of timestamps for a fully covered local delivery period.

    Returns ``None`` when the frequency cannot be inferred robustly or when the
    product type is unsupported by Pillar 1.1.
    """
    if len(idx_utc) < 2:
        return None

    diffs = idx_utc.to_series().diff().dropna()
    if diffs.empty:
        return None
    step = diffs.mode().iloc[0]
    if step <= pd.Timedelta(0):
        return None

    start_month = 1
    end_month = 12
    if product_type == "Quarter":
        if sub is None:
            return None
        start_month = (sub - 1) * 3 + 1
        end_month = start_month + 2
    elif product_type == "Month":
        if sub is None:
            return None
        start_month = sub
        end_month = sub
    elif product_type != "Cal":
        return None

    start_local = pd.Timestamp(year=year, month=start_month, day=1, tz="Europe/Zurich")
    if end_month == 12:
        end_local = pd.Timestamp(year=year + 1, month=1, day=1, tz="Europe/Zurich")
    else:
        end_local = pd.Timestamp(year=year, month=end_month + 1, day=1, tz="Europe/Zurich")

    expected_idx = pd.date_range(
        start_local.tz_convert("UTC"),
        end_local.tz_convert("UTC"),
        freq=step,
        inclusive="left",
    )
    return int(len(expected_idx))


# ---------------------------------------------------------------------------
# Pillar 1.1 — Arbitrage-freeness
# ---------------------------------------------------------------------------


def test_arb_free(
    pfc: pd.Series,
    forwards: dict[str, float],
    tol: float = 0.01,
    min_coverage: float = 0.95,
) -> TestResult:
    """Pillar 1.1 Hildmann arb-free : `|mean(PFC | period) - forward| < tol €/MWh`.

    Parameters
    ----------
    pfc
        Série PFC tz-aware (UTC) à valeurs €/MWh.
    forwards
        Mapping `{key: forward_price}` avec keys au format
        `"YYYY"` / `"YYYY-QN"` / `"YYYY-MM"` (cf. `parse_key`).
    tol
        Tolérance en €/MWh (default 0.01 = 1 centime, plus strict que le
        `calibration_tol: 0.01` du modèle car teste le pipeline complet).

    Returns
    -------
    TestResult
        passed=True ssi tous les forwards sont reproduits à `tol` près.
        observed = `max_dev` (la plus grande déviation observée).
        details = `{key: {observed, forward, dev}}` per key (skip silencieux
        keys pour lesquels le mask est vide — pas une "violation").

    Notes
    -----
    Edge case : `forwards = {}` (vide) → `max_dev = 0.0`, `passed = True`.
    Edge case : key avec mask vide → silently skipped + listé dans
    `details["_skipped"]` pour audit.
    """
    if pfc.index.tz is None:
        raise ValueError("pfc must be tz-aware (UTC expected).")

    max_dev = 0.0
    details: dict[str, Any] = {}
    skipped: list[str] = []
    partial: list[str] = []
    evaluated: list[str] = []

    for key, fwd_price in forwards.items():
        try:
            product_type, year, sub = parse_key(key)
        except ValueError:
            # Key non parseable → skip (e.g. "weird_key") au lieu de crash
            skipped.append(key)
            continue

        mask = _period_mask(pfc.index, product_type, year, sub)
        if mask.sum() == 0:
            skipped.append(key)
            continue

        expected_points = _expected_period_points(pfc.index, product_type, year, sub)
        if expected_points is not None and expected_points > 0:
            coverage = float(mask.sum()) / float(expected_points)
            if coverage < min_coverage:
                partial.append(key)
                details[key] = {
                    "coverage": coverage,
                    "expected_points": expected_points,
                    "observed_points": int(mask.sum()),
                    "reason": "partial_period_coverage",
                }
                continue

        observed_mean = float(pfc.iloc[mask].mean())
        dev = abs(observed_mean - float(fwd_price))
        details[key] = {
            "observed": observed_mean,
            "forward": float(fwd_price),
            "dev": dev,
        }
        evaluated.append(key)
        if dev > max_dev:
            max_dev = dev

    if skipped:
        details["_skipped"] = skipped
    if partial:
        details["_partial"] = partial

    # WR-06 : coverage-aware gating. Si forwards non-vide mais AUCUNE key
    # n'est évaluable (toutes skipées par mask vide ou parse-error), c'est
    # un trou de couverture silencieux — pas un PASS triviale (max_dev=0).
    # Le caller doit savoir pour distinguer "PFC reproduit parfaitement
    # ses forwards" de "PFC ne couvre AUCUN delivery period référencé".
    n_keys = len(forwards)
    n_evaluated = len(evaluated)
    if n_keys > 0 and n_evaluated == 0:
        return TestResult(
            passed=False,
            observed=float("nan"),
            threshold=tol,
            details={
                **details,
                "reason": "no_keys_evaluable",
                "degenerate": True,
                "n_keys": n_keys,
                "n_evaluated": 0,
                "min_coverage": min_coverage,
            },
        )

    return TestResult(
        passed=max_dev < tol,
        observed=max_dev,
        threshold=tol,
        details=details,
    )


# ---------------------------------------------------------------------------
# Pillar 1.2 — Holiday/weekend pattern
# ---------------------------------------------------------------------------


def test_holiday_weekend(
    pfc: pd.Series,
    country: str = "CH",
    subdiv: str = "VS",
    range_: tuple[float, float] = HOLIDAY_WEEKEND_RANGE,
) -> TestResult:
    """Pillar 1.2 Hildmann holiday/weekend ratio.

    `ratio = mean(PFC | weekend ∪ holidays) / mean(PFC | weekday hors holidays)`
    doit être ∈ `range_` (default `HOLIDAY_WEEKEND_RANGE = (0.65, 0.95)`).

    Parameters
    ----------
    pfc
        Série PFC tz-aware (UTC) à valeurs €/MWh.
    country
        Code ISO holidays (default "CH" = Suisse).
    subdiv
        Subdivision holidays (default "VS" = Valais — convention FMV depuis
        Phase 5bis-B). Détermine les jours fériés cantonaux (St-Joseph 19/3,
        Fête-Dieu, Assomption 15/8, Toussaint 1/11).
    range_
        Range cible `(low, high)`. Default = `HOLIDAY_WEEKEND_RANGE` frozen
        Plan 10-01 NOTES Pitfall 1 (C2 REVIEWS audit-trail).

    Returns
    -------
    TestResult
        passed = `range_[0] <= ratio <= range_[1]`.
        observed = ratio (NaN si dégeneré).
        details = `{mean_off, mean_on, n_off, n_on, degenerate}`.

    Notes
    -----
    Edge case `|mean(weekday)| < 0.5 €/MWh` (PFC quasi-flat à zéro) → ratio
    undefined → return `passed=False`, `degenerate=True`, `observed=NaN`.
    Évite la division par quasi-zéro qui amplifierait du bruit.
    """
    if pfc.index.tz is None:
        raise ValueError("pfc must be tz-aware (UTC expected).")

    idx_local = pfc.index.tz_convert("Europe/Zurich")
    years = sorted({d.year for d in idx_local})
    hols = holidays.country_holidays(country, subdiv=subdiv, years=years)
    is_weekend = idx_local.weekday >= 5
    is_holiday = np.array([d.date() in hols for d in idx_local], dtype=bool)
    is_off = np.asarray(is_weekend | is_holiday, dtype=bool)
    is_on = ~is_off

    n_off = int(is_off.sum())
    n_on = int(is_on.sum())

    if n_off == 0 or n_on == 0:
        return TestResult(
            passed=False,
            observed=float("nan"),
            threshold=range_,
            details={
                "n_off": n_off,
                "n_on": n_on,
                "degenerate": True,
                "reason": "empty_weekday_or_weekend_mask",
            },
        )

    mean_off = float(pfc.iloc[is_off].mean())
    mean_on = float(pfc.iloc[is_on].mean())

    if abs(mean_on) < 0.5:
        return TestResult(
            passed=False,
            observed=float("nan"),
            threshold=range_,
            details={
                "mean_off": mean_off,
                "mean_on": mean_on,
                "n_off": n_off,
                "n_on": n_on,
                "degenerate": True,
                "reason": "mean_weekday_near_zero",
            },
        )

    ratio = mean_off / mean_on
    passed = (range_[0] <= ratio) and (ratio <= range_[1])
    return TestResult(
        passed=passed,
        observed=float(ratio),
        threshold=range_,
        details={
            "mean_off": mean_off,
            "mean_on": mean_on,
            "n_off": n_off,
            "n_on": n_on,
            "degenerate": False,
        },
    )


# ---------------------------------------------------------------------------
# Pillar 1.3 — Seasonal profile correlation
# ---------------------------------------------------------------------------


def test_seasonal_profile(
    pfc: pd.Series,
    epex_hist: pd.Series,
    period_pfc_years: tuple[int, int] = (2024, 2025),
    period_hist_years: tuple[int, int] = (2019, 2023),
    min_corr: float = 0.85,
) -> TestResult:
    """Pillar 1.3 Hildmann seasonal profile correlation.

    Pearson correlation entre la signature mensuelle PFC et la signature
    mensuelle EPEX historique. `corr > min_corr` (default 0.85).

    Parameters
    ----------
    pfc
        Série PFC tz-aware (UTC), valeurs €/MWh, idéalement 2024-2025.
    epex_hist
        Série EPEX historique tz-aware (UTC), valeurs €/MWh, idéalement 2019-2023.
    period_pfc_years
        Range années pour filtre PFC (inclusive, default (2024, 2025)).
    period_hist_years
        Range années pour filtre EPEX (inclusive, default (2019, 2023)).
    min_corr
        Seuil pearson_r minimum (default 0.85).

    Returns
    -------
    TestResult
        passed = `r > min_corr`.
        observed = pearson_r (float).
        details = `{pearson_r, pearson_pvalue, n_months}`.

    Notes
    -----
    Si moins de 3 mois communs, pearsonr renvoie NaN → passed=False.
    """
    if pfc.index.tz is None:
        raise ValueError("pfc must be tz-aware (UTC expected).")
    if epex_hist.index.tz is None:
        raise ValueError("epex_hist must be tz-aware (UTC expected).")

    pfc_filt = pfc[
        (pfc.index.year >= period_pfc_years[0])
        & (pfc.index.year <= period_pfc_years[1])
    ]
    hist_filt = epex_hist[
        (epex_hist.index.year >= period_hist_years[0])
        & (epex_hist.index.year <= period_hist_years[1])
    ]

    if pfc_filt.empty or hist_filt.empty:
        return TestResult(
            passed=False,
            observed=float("nan"),
            threshold=min_corr,
            details={
                "n_months": 0,
                "degenerate": True,
                "reason": "empty_pfc_or_hist_after_year_filter",
            },
        )

    sig_pfc = pfc_filt.groupby(pfc_filt.index.month).mean()
    sig_hist = hist_filt.groupby(hist_filt.index.month).mean()
    common = sig_pfc.index.intersection(sig_hist.index)

    if len(common) < 3:
        return TestResult(
            passed=False,
            observed=float("nan"),
            threshold=min_corr,
            details={
                "n_months": int(len(common)),
                "degenerate": True,
                "reason": "fewer_than_3_common_months",
            },
        )

    r, p_val = pearsonr(sig_pfc.loc[common].values, sig_hist.loc[common].values)
    r = float(r)
    return TestResult(
        passed=bool(r > min_corr),
        observed=r,
        threshold=min_corr,
        details={
            "pearson_r": r,
            "pearson_pvalue": float(p_val),
            "n_months": int(len(common)),
        },
    )


# ---------------------------------------------------------------------------
# Pillar 1.4 — Continuity at month boundaries
# ---------------------------------------------------------------------------


def test_continuity(pfc: pd.Series, max_jump: float = 2.0) -> TestResult:
    """Pillar 1.4 Hildmann continuity at month boundaries.

    `max |PFC[last point of M_i] - PFC[first point of M_{i+1}]| < max_jump €/MWh`.

    Parameters
    ----------
    pfc
        Série PFC tz-aware (UTC), valeurs €/MWh.
    max_jump
        Seuil max en €/MWh (default 2.0).

    Returns
    -------
    TestResult
        passed = `max_jump_obs < max_jump`.
        observed = max abs jump (€/MWh).
        details = `{n_boundaries, all_jumps_by_period}`.

    Notes
    -----
    - Frontières calculées en heure locale Europe/Zurich pour cohérence trader
      (le "mois" = mois local CH, pas UTC).
    - Edge case : 1 seul mois → 0 frontières → max_jump_obs = 0.0 → passed=True.
    """
    if pfc.index.tz is None:
        raise ValueError("pfc must be tz-aware (UTC expected).")

    idx_local = pfc.index.tz_convert("Europe/Zurich")
    # Compute month-period in local time pour la sémantique "mois CH".
    # Drop tz pour to_period (pandas ne supporte pas to_period sur DatetimeIndex tz-aware).
    month_period = idx_local.tz_localize(None).to_period("M")
    df = pd.DataFrame(
        {"price": pfc.values, "month": month_period}, index=pfc.index
    )
    df = df.sort_index()
    # Last point of each month + first point of each month
    last_of_month = df.groupby("month", sort=True).tail(1).set_index("month")[
        "price"
    ]
    first_of_month = df.groupby("month", sort=True).head(1).set_index("month")[
        "price"
    ]
    # last_of_month[M_i] vs first_of_month[M_{i+1}]
    last_sorted = last_of_month.sort_index()
    first_sorted = first_of_month.sort_index()

    if len(last_sorted) < 2:
        # 1 seul mois → 0 frontières
        return TestResult(
            passed=True,
            observed=0.0,
            threshold=max_jump,
            details={"n_boundaries": 0, "all_jumps": {}},
        )

    # M_i → M_{i+1} : on prend last_sorted[:-1] et first_sorted[1:]
    last_arr = last_sorted.values[:-1]
    first_arr = first_sorted.values[1:]
    jumps = np.abs(first_arr - last_arr)
    max_jump_obs = float(jumps.max()) if len(jumps) > 0 else 0.0

    # Sérialisation lisible des jumps (period.strftime → str)
    boundary_labels = [
        f"{last_sorted.index[i]} -> {first_sorted.index[i + 1]}"
        for i in range(len(jumps))
    ]
    all_jumps = dict(zip(boundary_labels, [float(j) for j in jumps]))

    return TestResult(
        passed=bool(max_jump_obs < max_jump),
        observed=max_jump_obs,
        threshold=max_jump,
        details={
            "n_boundaries": int(len(jumps)),
            "all_jumps": all_jumps,
        },
    )


__all__ = [
    "HOLIDAY_WEEKEND_RANGE",
    "TestResult",
    "test_arb_free",
    "test_holiday_weekend",
    "test_seasonal_profile",
    "test_continuity",
]
