"""
dm_test.py
----------
Pillar 4 — Diebold-Mariano test + 3 naive baselines maison.

Implémentation maison ~30 lignes via `statsmodels.tsa.stattools.acovf` +
Bartlett HAC weights + HLN small-sample correction (Harvey-Leybourne-Newbold
1997). **Aucune dep arch ou dieboldmariano PyPI** (RESEARCH §Don't Hand-Roll
reject `arch` et `dieboldmariano==0.1.x` slop risk).

Le test
-------
H0 : `equal accuracy` between forecast A and forecast B.
Loss differential `d_t = L(e_a_t) - L(e_b_t)` (MAE ou MSE).
Mean(d) < 0 → A is better.

HAC variance : Bartlett-weighted sum `gamma_0 + 2 * sum(w_k * gamma_k)` où
`w_k = 1 - k/(n_lags+1)` (R `forecast::dm.test` default, positive-semidefinite
par construction).

HLN small-sample correction : `dm_stat *= sqrt((n+1-2h+h(h-1)/n)/n)` puis
p-value via Student-t df=n-1 (au lieu de N(0,1)).

3 baselines
-----------
- `baseline_climatology(epex_hist, bloc, years)` : mean(EPEX | bloc filter, years)
- `baseline_persistence_y1(epex_realised, vintage_date, bloc, window_days)` :
  mean(EPEX_realised | bloc filter, vintage-1yr ± window_days)
- `baseline_forwards_flat(forwards_asof, horizon_label, vintage_date)` :
  forward Cal/Q/M correspondant à horizon (M+1 → "YYYY-MM", Y+1 → "YYYY")

Source
------
- Diebold-Mariano 1995 §3.1 + Harvey-Leybourne-Newbold 1997
- R reference impl : pkg.robjhyndman.com/forecast/reference/dm.test.html
- Plan 10-RESEARCH.md §Pattern 4 canonical (lignes 359-462)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from statsmodels.tsa.stattools import acovf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# diebold_mariano — DM 1995 + HLN 1997 small-sample correction
# ---------------------------------------------------------------------------


def diebold_mariano(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    h: int = 1,
    loss: str = "mae",
    hln_correction: bool = True,
) -> dict:
    """Diebold-Mariano test : H0 = equal accuracy between forecast A and B.

    Parameters
    ----------
    errors_a
        Forecast errors of A (e.g. realised - pred_a). Same sign convention as B.
    errors_b
        Forecast errors of B, same sign convention as A.
    h
        Forecast horizon in periods. HAC lag = h - 1 (DM 1995 recommendation).
    loss
        `"mae"` (loss = |e|) or `"mse"` (loss = e^2).
    hln_correction
        Apply Harvey-Leybourne-Newbold (1997) small-sample correction
        + Student-t df=n-1 p-value (vs N(0,1) without correction).

    Returns
    -------
    dict
        Mapping avec :
        - `dm_stat` : float — DM statistic (NaN if degenerate)
        - `p_value` : float — two-sided p-value (NaN if degenerate)
        - `n` : int — number of observations
        - `mean_d` : float — mean(d) (NaN if degenerate path)
        - `var_d` : float — HAC long-run variance estimate (NaN if degenerate)
        - `n_lags_hac` : int — h - 1
        - `degenerate` : bool — True if n < max(h+1, 8) OR var_d <= 0 at lag=0

    Raises
    ------
    ValueError
        - If `len(errors_a) != len(errors_b)`
        - If `loss not in {"mae", "mse"}`

    Notes
    -----
    - Loss differential `d_t = L(e_a_t) - L(e_b_t)`. Mean negative → A better.
    - HAC weights : Bartlett `1 - k/(n_lags+1)` (positive-semidefinite par
      construction, vs uniform weights DM 1995 §3.1 qui peuvent produire
      var_d < 0).
    - Edge case `var_d <= 0` (rare avec Bartlett mais documenté DM 1995 §3.2) :
      fallback à sample variance `gammas[0]` + emit log.warning ; if still
      <= 0 → return degenerate=True.
    - n < max(h+1, 8) → too few obs for meaningful HAC → degenerate=True.

    Examples
    --------
    >>> import numpy as np
    >>> # A clearly better : small errors A, large errors B
    >>> a = np.array([0.1] * 20)
    >>> b = np.array([2.0] * 20)
    >>> r = diebold_mariano(a, b, h=1)
    >>> r["p_value"] < 0.05
    True
    """
    if len(errors_a) != len(errors_b):
        raise ValueError(
            f"diebold_mariano: length mismatch — a={len(errors_a)}, b={len(errors_b)}"
        )

    # WR-11 : DM/HLN est mathématiquement défini uniquement pour h >= 1
    # (n_lags = h-1 >= 0 ET la correction HLN ligne 198 suppose h >= 1).
    # Avec h <= 0, n_lags = 0 force une fenêtre Bartlett dégénérée et la
    # formule HLN sort de son contrat ; le résultat (dm_stat, p_value) est
    # syntaxiquement valide mais sans signification économétrique. Raise
    # explicit pour bloquer ce silent-nonsense (e.g. nowcast comparison
    # miswritten as "horizon zero").
    if int(h) < 1:
        raise ValueError(
            f"diebold_mariano: h must be >= 1 (got h={h}). "
            "DM/HLN is undefined for h<=0."
        )

    n = len(errors_a)
    n_lags = max(h - 1, 0)

    # --------------------------------------------------------------
    # Degenerate : n too few obs for meaningful HAC variance
    # --------------------------------------------------------------
    if n < max(h + 1, 8):
        return {
            "dm_stat": float("nan"),
            "p_value": float("nan"),
            "n": int(n),
            "mean_d": float("nan"),
            "var_d": float("nan"),
            "n_lags_hac": int(n_lags),
            "degenerate": True,
        }

    # --------------------------------------------------------------
    # Loss differential d_t = L(e_a_t) - L(e_b_t)
    # --------------------------------------------------------------
    if loss == "mae":
        d = np.abs(errors_a) - np.abs(errors_b)
    elif loss == "mse":
        d = errors_a ** 2 - errors_b ** 2
    else:
        raise ValueError(
            f"diebold_mariano: loss must be 'mae' or 'mse', got {loss!r}"
        )

    d_mean = float(np.mean(d))

    # --------------------------------------------------------------
    # Bartlett HAC variance (gamma_0 + 2*sum(weights * gammas[1:]))
    # --------------------------------------------------------------
    gammas = acovf(d, nlag=n_lags, fft=True, demean=True)  # length n_lags+1
    if n_lags == 0:
        var_d = float(gammas[0])
    else:
        # Bartlett weights : 1 - k/(n_lags+1), positive-semidefinite par construction
        weights = 1.0 - np.arange(1, n_lags + 1) / float(n_lags + 1)
        var_d = float(gammas[0] + 2.0 * np.sum(weights * gammas[1:]))

    # --------------------------------------------------------------
    # Edge case : var_d <= 0 (rare, Pitfall 3 RESEARCH)
    # --------------------------------------------------------------
    if var_d <= 0:
        logger.warning(
            "DM var_d <= 0 after Bartlett HAC weighting (n=%d, h=%d, var_d=%.6g) "
            "— falling back to sample variance lag=0",
            n,
            h,
            var_d,
        )
        var_d = float(gammas[0])
        if var_d <= 0:
            return {
                "dm_stat": float("nan"),
                "p_value": float("nan"),
                "n": int(n),
                "mean_d": d_mean,
                "var_d": var_d,
                "n_lags_hac": int(n_lags),
                "degenerate": True,
            }

    # --------------------------------------------------------------
    # DM stat + HLN small-sample correction
    # --------------------------------------------------------------
    dm_stat = d_mean / np.sqrt(var_d / n)

    if hln_correction:
        # HLN 1997 : adj = sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
        # Clip à 1e-12 pour éviter sqrt(neg) si h très grand vs n (rare ; le
        # guard n < max(h+1, 8) couvre le cas standard).
        adj = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
        dm_stat = dm_stat * adj
        # Student-t df=n-1
        p_value = float(2 * student_t.sf(abs(dm_stat), df=n - 1))
    else:
        from scipy.stats import norm

        p_value = float(2 * norm.sf(abs(dm_stat)))

    return {
        "dm_stat": float(dm_stat),
        "p_value": p_value,
        "n": int(n),
        "mean_d": d_mean,
        "var_d": var_d,
        "n_lags_hac": int(n_lags),
        "degenerate": False,
    }


# ---------------------------------------------------------------------------
# 3 naive baselines (D-A4-1 CONTEXT Phase 10)
# ---------------------------------------------------------------------------


def baseline_climatology(
    epex_hist: pd.Series,
    bloc: Any,
    years: tuple[int, int] = (2019, 2023),
) -> float:
    """Baseline 1 — `mean(EPEX_hist | bloc filter, years range)`.

    Scalar baseline, indépendant du horizon/vintage. Représente la moyenne
    historique pour le bloc considéré (e.g. midday weekday).

    Parameters
    ----------
    epex_hist
        Série EPEX historique tz-aware (UTC).
    bloc
        Instance `BlockMask`.
    years
        Tuple `(year_start, year_end)` inclusive (default (2019, 2023)).

    Returns
    -------
    float
        Mean €/MWh sur le bloc filter dans le range d'années. NaN si vide.
    """
    if epex_hist.index.tz is None:
        raise ValueError("epex_hist must be tz-aware (UTC expected).")
    year_mask = (epex_hist.index.year >= years[0]) & (epex_hist.index.year <= years[1])
    sub = epex_hist.loc[year_mask]
    if sub.empty:
        logger.warning(
            "baseline_climatology: no EPEX data in years range %s — returning NaN",
            years,
        )
        return float("nan")
    bloc_mask = bloc.apply(sub.index)
    if bloc_mask.sum() == 0:
        logger.warning(
            "baseline_climatology: bloc %s mask empty in years range %s — returning NaN",
            bloc.name,
            years,
        )
        return float("nan")
    return float(sub.values[bloc_mask].mean())


def baseline_persistence_y1(
    epex_realised: pd.Series,
    vintage_date: pd.Timestamp,
    bloc: Any,
    window_days: int = 15,
) -> float:
    """Baseline 2 — `mean(EPEX_realised | bloc filter, vintage-1yr ± window_days)`.

    Persistance Y-1 : on suppose que le prix futur ressemble au prix réalisé
    un an plus tôt, dans une fenêtre ± window_days autour de la date vintage
    décalée d'un an.

    Parameters
    ----------
    epex_realised
        Série EPEX realised tz-aware (UTC).
    vintage_date
        Date as-of de la vintage (tz-aware).
    bloc
        Instance `BlockMask`.
    window_days
        Demi-largeur de la fenêtre en jours (default 15).

    Returns
    -------
    float
        Mean €/MWh dans la fenêtre. NaN si fenêtre vide.
    """
    if epex_realised.index.tz is None:
        raise ValueError("epex_realised must be tz-aware (UTC expected).")
    target = vintage_date - pd.DateOffset(years=1)
    window_lo = target - pd.Timedelta(days=window_days)
    window_hi = target + pd.Timedelta(days=window_days)
    # Convert window bounds to same tz as epex_realised index
    if window_lo.tz is None:
        window_lo = window_lo.tz_localize("UTC")
        window_hi = window_hi.tz_localize("UTC")
    else:
        window_lo = window_lo.tz_convert(epex_realised.index.tz)
        window_hi = window_hi.tz_convert(epex_realised.index.tz)
    sub = epex_realised.loc[
        (epex_realised.index >= window_lo) & (epex_realised.index <= window_hi)
    ]
    if sub.empty:
        logger.warning(
            "baseline_persistence_y1: empty window [%s, %s] — returning NaN",
            window_lo,
            window_hi,
        )
        return float("nan")
    bloc_mask = bloc.apply(sub.index)
    if bloc_mask.sum() == 0:
        logger.warning(
            "baseline_persistence_y1: bloc %s mask empty in window — returning NaN",
            bloc.name,
        )
        return float("nan")
    return float(sub.values[bloc_mask].mean())


def baseline_forwards_flat(
    forwards_asof: dict[str, float],
    horizon_label: str,
    vintage_date: pd.Timestamp,
) -> float:
    """Baseline 3 — forward Cal/Q/M flat (sans intra-period shape).

    Pour un horizon donné (M+1, M+3, M+6, Q+1, Y+1, Y+2), retourne la
    valeur du forward correspondant dans `forwards_asof`, sans aucune
    modulation horaire / hebdomadaire / saisonnière (= valeur uniforme
    constante sur le bloc).

    Convention horizon_label → key :
    - `"M+1"`, `"M+3"`, `"M+6"` → format `"YYYY-MM"` du mois ciblé
    - `"Q+1"`, `"Q+2"`, etc. → format `"YYYY-QN"`
    - `"Y+1"`, `"Y+2"` → format `"YYYY"` de l'année ciblée

    Parameters
    ----------
    forwards_asof
        Mapping `{key: forward_price}` issu de
        `derive_forwards_from_epex_hist` ou du snapshot EEX XLSX.
    horizon_label
        Étiquette horizon (e.g. "M+1", "Y+2").
    vintage_date
        Date as-of de la vintage (tz-aware).

    Returns
    -------
    float
        Forward price €/MWh ; NaN si key absente.
    """
    horizon_label = horizon_label.strip()
    # Month horizon
    if horizon_label.startswith("M+"):
        try:
            n_months = int(horizon_label[2:])
        except ValueError:
            logger.warning(
                "baseline_forwards_flat: invalid horizon_label %r — returning NaN",
                horizon_label,
            )
            return float("nan")
        target = vintage_date + pd.DateOffset(months=n_months)
        key = f"{target.year:04d}-{target.month:02d}"
    # Quarter horizon
    elif horizon_label.startswith("Q+"):
        try:
            n_quarters = int(horizon_label[2:])
        except ValueError:
            logger.warning(
                "baseline_forwards_flat: invalid horizon_label %r — returning NaN",
                horizon_label,
            )
            return float("nan")
        target = vintage_date + pd.DateOffset(months=3 * n_quarters)
        q_num = ((target.month - 1) // 3) + 1
        key = f"{target.year:04d}-Q{q_num}"
    # Year horizon
    elif horizon_label.startswith("Y+"):
        try:
            n_years = int(horizon_label[2:])
        except ValueError:
            logger.warning(
                "baseline_forwards_flat: invalid horizon_label %r — returning NaN",
                horizon_label,
            )
            return float("nan")
        target_year = vintage_date.year + n_years
        key = f"{target_year:04d}"
    else:
        logger.warning(
            "baseline_forwards_flat: unrecognized horizon_label %r "
            "(expected 'M+N', 'Q+N', or 'Y+N') — returning NaN",
            horizon_label,
        )
        return float("nan")

    if key not in forwards_asof:
        logger.warning(
            "baseline_forwards_flat: key %r not in forwards_asof — returning NaN",
            key,
        )
        return float("nan")
    return float(forwards_asof[key])


__all__ = [
    "diebold_mariano",
    "baseline_climatology",
    "baseline_persistence_y1",
    "baseline_forwards_flat",
]
