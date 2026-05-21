"""
christoffersen.py
-----------------
Pillar 3 — Christoffersen 1998 unconditional coverage LR test.

**IC80 only in Phase 10.** IC95 + Conditional coverage + reliability diagrams
deferred Phase 5ter (D-A3-3 amendé Plan 10-03 revision iter 1, raison:
`pfc_shaping/lt/model/uncertainty.py:51-194` expose `p10`/`p90` only — pas de
paramètre `level=` ni `confidence=`).

Le test
-------
H0 : `observed_freq == nominal p`
LR_uc = -2 * log( (p^x * (1-p)^(n-x)) / (pi_hat^x * (1-pi_hat)^(n-x)) )
Distribution sous H0 : chi-squared with 1 df.

Pour IC80 : `nominal_p = 0.20` (probabilité de violation = 1 - 0.80).
Violation = realised < p10 OR realised > p90.

Interpretation
--------------
- `p_value > 0.05` → H0 NOT rejected → unconditional coverage est correcte
  (la fréquence empirique est statistiquement compatible avec le nominal).
- `p_value < 0.05` → H0 rejected → mauvaise calibration (under/over coverage).

Degenerate cases
----------------
- `n == 0` : aucune obs → observed_freq=NaN, p_value=NaN, degenerate=True
- `x == 0` ou `x == n` : pi_hat ∈ {0, 1} → log(0) crash sans guard ;
  → degenerate=True, p_value=NaN (per RESEARCH §Pattern 3 lignes 325-335).

Source
------
- Christoffersen 1998 LR_uc formula : value-at-risk.net/backtesting-coverage-tests/
- Plan 10-RESEARCH.md §Pattern 3 canonical (lignes 305-351).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def lr_unconditional_coverage(x: int, n: int, p: float) -> dict:
    """Christoffersen (1998) unconditional coverage LR test.

    H0 : `observed_freq == nominal p` (la fréquence empirique des violations
    matche la probabilité nominale).

    Parameters
    ----------
    x
        Count of observations OUTSIDE the IC (violations).
    n
        Total observations (in-bloc after masking).
    p
        Nominal violation probability (e.g. 0.20 for IC80, 0.05 for IC95).

    Returns
    -------
    dict
        Mapping avec :
        - `lr_stat` : float — LR test statistic (NaN if degenerate)
        - `p_value` : float — chi2(df=1).sf(lr_stat) ; NaN if degenerate
        - `observed_freq` : float — x/n (NaN if n==0)
        - `nominal_p` : float — p (echo input)
        - `n` : int — total observations
        - `x` : int — violations count
        - `degenerate` : bool — True if n==0 OR x∈{0, n}

    Notes
    -----
    Degenerate guards (per RESEARCH §Pattern 3 lignes 325-335) :
    - `n == 0` → division-by-zero would crash pi_hat = x/n
    - `x == 0` or `x == n` → log(0) crash in log_lik computation
      (pi_hat=0 → log(pi_hat) = -inf, ou (1-pi_hat)=0 → log(0)=-inf)

    Use log-formulation for numerical stability (large n).

    Examples
    --------
    >>> # Perfect coverage : 20% violations vs 20% nominal → H0 not rejected
    >>> r = lr_unconditional_coverage(x=20, n=100, p=0.20)
    >>> r["p_value"] > 0.9
    True
    >>> # Undercoverage : 5% observed → H0 rejected
    >>> r = lr_unconditional_coverage(x=5, n=100, p=0.20)
    >>> r["p_value"] < 0.05
    True
    """
    if n == 0 or x == 0 or x == n:
        # Degenerate case — return NaN flags, caller must handle
        observed_freq = float("nan") if n == 0 else float(x) / float(n)
        return {
            "lr_stat": float("nan"),
            "p_value": float("nan"),
            "observed_freq": observed_freq,
            "nominal_p": float(p),
            "n": int(n),
            "x": int(x),
            "degenerate": True,
        }

    pi_hat = float(x) / float(n)
    # Use log-formulation for numerical stability
    log_lik_null = x * np.log(p) + (n - x) * np.log(1.0 - p)
    log_lik_alt = x * np.log(pi_hat) + (n - x) * np.log(1.0 - pi_hat)
    lr_stat = -2.0 * (log_lik_null - log_lik_alt)
    p_value = float(chi2.sf(lr_stat, df=1))
    return {
        "lr_stat": float(lr_stat),
        "p_value": p_value,
        "observed_freq": pi_hat,
        "nominal_p": float(p),
        "n": int(n),
        "x": int(x),
        "degenerate": False,
    }


__all__ = ["lr_unconditional_coverage"]
