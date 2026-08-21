"""
christoffersen.py
-----------------
Pillar 3 — Christoffersen 1998 coverage tests (unconditional + independence
+ combined conditional).

**IC80 only in Phase 10.** IC95 + reliability diagrams deferred Phase 5ter
(D-A3-3 amendé Plan 10-03 revision iter 1, raison:
`pfc_shaping/lt/model/uncertainty.py:51-194` expose `p10`/`p90` only — pas de
paramètre `level=` ni `confidence=`).

Les 3 tests
-----------
1. **LR_uc** — unconditional coverage (`lr_unconditional_coverage`)
   H0 : `observed_freq == nominal p` (fréquence empirique matche le nominal).
   Statistique : `LR_uc = -2 log( (p^x (1-p)^(n-x)) / (pi^x (1-pi)^(n-x)) )`.
   Distribution sous H0 : chi-squared df=1.

2. **LR_ind** — independence of violations (`lr_independence_coverage`, WR-12)
   H0 : `Prob(violation_t | violation_{t-1}) == Prob(violation_t)`
   (les violations ne sont PAS clusterées dans le temps).
   Statistique : LR test sur la chaîne de Markov à 2 états (0=no-viol,
   1=viol) avec transitions `(n_00, n_01, n_10, n_11)`. Distribution sous
   H0 : chi-squared df=1.

3. **LR_cc** — combined conditional coverage (`lr_conditional_coverage`, WR-12)
   `LR_cc = LR_uc + LR_ind`. Distribution sous H0 : chi-squared df=2.
   C'est le test global Christoffersen 1998 §IV.

Pour IC80 : `nominal_p = 0.20` (probabilité de violation = 1 - 0.80).
Violation = realised < p10 OR realised > p90.

Pourquoi LR_ind matters (WR-12 motivation)
-------------------------------------------
LR_uc seul ne détecte PAS les clusters de violations : un PFC qui produit
ses 20 violations CONSÉCUTIVES en Q4 (faute de calibration saisonnière)
passe le test LR_uc tant que x/n ≈ 0.20 global. LR_ind est sensible à ces
clusters (corr lag-1 de la séquence de violations) et permet de détecter
la mauvaise spécification saisonnière, qui est le mode de défaillance
dominant pour une PFC long-terme.

Interpretation
--------------
- `p_value > 0.05` → H0 NOT rejected (calibration correcte pour ce test)
- `p_value < 0.05` → H0 rejected → mauvaise calibration

Degenerate cases
----------------
- `n == 0` : aucune obs → degenerate=True, p_value=NaN
- `x == 0` ou `x == n` : LR_uc utilise binomial exact (WR-07). LR_ind est
  trivialement degenerate (transitions impossibles) → p_value=NaN.

Sources
-------
- Christoffersen 1998 §III (LR_uc) + §IV (LR_ind, LR_cc).
- value-at-risk.net/backtesting-coverage-tests/
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
        - `lr_stat` : float — LR test statistic (NaN if degenerate or
          boundary `x ∈ {0, n}` — LR explose à la frontière).
        - `p_value` : float — chi2(df=1).sf(lr_stat) ; pour `x ∈ {0, n}`
          (WR-07), p_value est calculé par binomial exact :
          `P(X = x | n, p)` au lieu de NaN.
        - `observed_freq` : float — x/n (NaN if n==0)
        - `nominal_p` : float — p (echo input)
        - `n` : int — total observations
        - `x` : int — violations count
        - `degenerate` : bool — True uniquement si n==0 (incalculable).
          Pour x ∈ {0, n}, degenerate=False et p_value est finite via
          binomial exact (WR-07 contract).
        - `method` : str — "lr_chi2" (cas standard) ou "binomial_exact"
          (cas boundary x ∈ {0, n}).

    Notes
    -----
    **WR-07** : pour `x == 0` ou `x == n`, l'ancien comportement
    (degenerate=True, p_value=NaN) jetait l'information ; pourtant
    `x = 0` avec `n` grand est un signal très fort d'overcoverage (e.g.
    pour `n=14, p=0.20`, `(1-p)^n ≈ 0.044` → significatif à 5%) et
    `x = n` un signal de miscalibration totale. Le remède canonique est
    de remplacer le LR (qui diverge) par le p-value binomial exact :
        - x = 0 → p_value = P(X = 0 | n, p) = (1-p)^n
        - x = n → p_value = P(X = n | n, p) = p^n
    On flag `method="binomial_exact"` pour audit downstream.

    Degenerate guard restant :
    - `n == 0` → division-by-zero would crash pi_hat = x/n → degenerate

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
    >>> # x=0 boundary : exact binomial, finite p_value
    >>> r = lr_unconditional_coverage(x=0, n=100, p=0.20)
    >>> r["method"]
    'binomial_exact'
    >>> r["p_value"] < 1e-9  # (0.8)^100 ≈ 2e-10
    True
    """
    if n == 0:
        # Truly incalculable — keep degenerate flag
        return {
            "lr_stat": float("nan"),
            "p_value": float("nan"),
            "observed_freq": float("nan"),
            "nominal_p": float(p),
            "n": int(n),
            "x": int(x),
            "degenerate": True,
            "method": "degenerate_n_zero",
        }

    if x == 0 or x == n:
        # WR-07 : LR_uc diverge à la frontière (log 0). Remplacement par
        # p-value binomial exact : P(X = x | n, p). Conserve l'information
        # discriminante au lieu de la jeter avec p_value=NaN.
        if x == 0:
            p_exact = float((1.0 - p) ** n)
        else:  # x == n
            p_exact = float(p ** n)
        return {
            "lr_stat": float("nan"),  # LR undefined at the boundary
            "p_value": p_exact,
            "observed_freq": float(x) / float(n),
            "nominal_p": float(p),
            "n": int(n),
            "x": int(x),
            "degenerate": False,
            "method": "binomial_exact",
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
        "method": "lr_chi2",
    }


def lr_independence_coverage(violations: np.ndarray) -> dict:
    """Christoffersen (1998) §IV.A LR test on consecutive violations (WR-12).

    H0 : `Prob(violation_t | violation_{t-1}) == Prob(violation_t)`
    (les violations ne sont PAS auto-corrélées — pas de clusters).

    Construction de la chaîne de Markov à 2 états :
      - état 0 = pas de violation
      - état 1 = violation
      - n_ij = nombre de transitions de l'état i vers l'état j
      - pi_01 = n_01 / (n_00 + n_01) = Prob(viol | not_viol_prev)
      - pi_11 = n_11 / (n_10 + n_11) = Prob(viol | viol_prev)
      - pi_2  = (n_01 + n_11) / (n_00 + n_01 + n_10 + n_11) = unconditional rate

    Statistique : `LR_ind = -2 log( L0 / L1 )` où
      L0 = pi_2^(n_01+n_11) * (1-pi_2)^(n_00+n_10)
      L1 = pi_01^n_01 * (1-pi_01)^n_00 * pi_11^n_11 * (1-pi_11)^n_10
    Distribution sous H0 : chi-squared df=1.

    Parameters
    ----------
    violations
        Séquence booléenne (np.ndarray[bool] ou int 0/1) des violations dans
        l'ordre temporel. La fonction calcule les transitions consécutives
        violation[t-1] → violation[t].

    Returns
    -------
    dict
        Mapping avec :
        - `lr_stat` : float — LR_ind test stat (NaN si degenerate)
        - `p_value` : float — chi2(df=1).sf(lr_stat) (NaN si degenerate)
        - `n_00`, `n_01`, `n_10`, `n_11` : int — transition counts
        - `pi_01`, `pi_11` : float — conditional rates (NaN si denom=0)
        - `n_transitions` : int — total transitions = len(violations) - 1
        - `degenerate` : bool — True ssi :
            (a) len(violations) < 2 → no transitions calculable
            (b) une des conditional rates a denom=0 (e.g. aucun "viol prev")
            (c) pi_01 == pi_11 exactement → L0 == L1 → LR_ind = 0 dégeneré
        - `method` : str — `"lr_chi2"` (cas standard) ou `"degenerate_*"`

    Notes
    -----
    Si `x == 0` (aucune violation) : n_01 = n_11 = 0 → pi_01 = pi_11 = 0 →
    L0 = L1 = (1-0)^total → LR_ind = 0 → degenerate=True (pas
    d'information sur l'indépendance car aucune violation observée).

    Si `x == n` (toutes les obs sont violations) : même argument
    symétrique → degenerate=True.

    Examples
    --------
    >>> # Violations indépendantes ~uniform → H0 not rejected
    >>> import numpy as np
    >>> rng = np.random.RandomState(0)
    >>> viol = rng.binomial(1, 0.20, 500).astype(bool)
    >>> r = lr_independence_coverage(viol)
    >>> r["p_value"] > 0.05
    True
    """
    v = np.asarray(violations).astype(bool)
    n_total = len(v)

    if n_total < 2:
        return {
            "lr_stat": float("nan"),
            "p_value": float("nan"),
            "n_00": 0,
            "n_01": 0,
            "n_10": 0,
            "n_11": 0,
            "pi_01": float("nan"),
            "pi_11": float("nan"),
            "n_transitions": 0,
            "degenerate": True,
            "method": "degenerate_too_few_obs",
        }

    # Transitions consécutives
    prev = v[:-1]
    curr = v[1:]
    n_00 = int(np.sum((~prev) & (~curr)))
    n_01 = int(np.sum((~prev) & curr))
    n_10 = int(np.sum(prev & (~curr)))
    n_11 = int(np.sum(prev & curr))
    n_transitions = n_00 + n_01 + n_10 + n_11

    denom_0 = n_00 + n_01  # transitions FROM state 0
    denom_1 = n_10 + n_11  # transitions FROM state 1

    # Si une des conditional rates est calculée sur un denom 0, le test
    # dégeénre (e.g. aucune violation, ou que des violations consécutives).
    if denom_0 == 0 or denom_1 == 0:
        pi_01 = float("nan") if denom_0 == 0 else float(n_01) / float(denom_0)
        pi_11 = float("nan") if denom_1 == 0 else float(n_11) / float(denom_1)
        return {
            "lr_stat": float("nan"),
            "p_value": float("nan"),
            "n_00": n_00,
            "n_01": n_01,
            "n_10": n_10,
            "n_11": n_11,
            "pi_01": pi_01,
            "pi_11": pi_11,
            "n_transitions": n_transitions,
            "degenerate": True,
            "method": "degenerate_empty_state",
        }

    pi_01 = float(n_01) / float(denom_0)
    pi_11 = float(n_11) / float(denom_1)
    # Uncon. rate sur les transitions (cohérent avec la pondération du LR)
    pi_2 = float(n_01 + n_11) / float(n_transitions)

    # Si pi_01 == pi_11 exactement → la décomposition est triviale, LR_ind = 0
    # (mais on retourne quand même les counts pour l'audit).
    if pi_01 == pi_11:
        return {
            "lr_stat": 0.0,
            "p_value": 1.0,
            "n_00": n_00,
            "n_01": n_01,
            "n_10": n_10,
            "n_11": n_11,
            "pi_01": pi_01,
            "pi_11": pi_11,
            "n_transitions": n_transitions,
            "degenerate": False,
            "method": "lr_chi2",
        }

    # Numerical safety : tout pi ∈ (0, 1) strict ici (denom > 0 et pi_01 ≠ pi_11
    # exclut les 4 coins triviaux 0/0/1/1). Mais pi_01 ou pi_11 peut valoir
    # 0 (e.g. n_01=0 alors que denom_0 > 0). On log-clamp à 1e-300 pour
    # éviter log(0) → -inf qui survit dans le LR.
    def _safe_log(x: float) -> float:
        return float(np.log(max(x, 1e-300)))

    log_L0 = (n_01 + n_11) * _safe_log(pi_2) + (n_00 + n_10) * _safe_log(1.0 - pi_2)
    log_L1 = (
        n_01 * _safe_log(pi_01)
        + n_00 * _safe_log(1.0 - pi_01)
        + n_11 * _safe_log(pi_11)
        + n_10 * _safe_log(1.0 - pi_11)
    )
    lr_stat = -2.0 * (log_L0 - log_L1)
    # LR doit être >= 0 par construction ; petite tolérance pour les
    # arrondis flottants.
    if lr_stat < 0 and abs(lr_stat) < 1e-9:
        lr_stat = 0.0
    p_value = float(chi2.sf(lr_stat, df=1))

    return {
        "lr_stat": float(lr_stat),
        "p_value": p_value,
        "n_00": n_00,
        "n_01": n_01,
        "n_10": n_10,
        "n_11": n_11,
        "pi_01": pi_01,
        "pi_11": pi_11,
        "n_transitions": n_transitions,
        "degenerate": False,
        "method": "lr_chi2",
    }


def lr_conditional_coverage(
    uc_result: dict,
    ind_result: dict,
) -> dict:
    """Christoffersen (1998) §IV.B combined conditional coverage (WR-12).

    `LR_cc = LR_uc + LR_ind` sous chi-squared df=2.

    Cf. Christoffersen 1998 §IV.B Proposition 1 : si les 2 tests sont
    indépendants asymptotiquement (ce qui est le cas sous H0 globale),
    leur somme suit une chi-squared(2).

    Parameters
    ----------
    uc_result
        Output de `lr_unconditional_coverage`.
    ind_result
        Output de `lr_independence_coverage`.

    Returns
    -------
    dict
        - `lr_stat` : `lr_uc + lr_ind` (NaN si l'un est NaN)
        - `p_value` : `chi2(df=2).sf(lr_stat)` (NaN si lr_stat NaN)
        - `lr_uc_stat`, `lr_ind_stat` : echo des composants
        - `degenerate` : True ssi l'un OU l'autre des composants est degenerate
        - `method` : `"lr_chi2_df2"` ou `"degenerate_uc"` ou `"degenerate_ind"`
        ou `"degenerate_both"`

    Notes
    -----
    Lorsque `uc_result["method"] == "binomial_exact"` (WR-07 boundary x=0
    ou x=n), `lr_uc_stat = NaN` (boundary LR diverge). Dans ce cas LR_cc
    est marqué degenerate=True avec method="degenerate_uc_boundary".
    """
    lr_uc = float(uc_result.get("lr_stat", float("nan")))
    lr_ind = float(ind_result.get("lr_stat", float("nan")))
    uc_degen = bool(uc_result.get("degenerate", False))
    ind_degen = bool(ind_result.get("degenerate", False))
    uc_method = str(uc_result.get("method", ""))

    if uc_method == "binomial_exact":
        return {
            "lr_stat": float("nan"),
            "p_value": float("nan"),
            "lr_uc_stat": lr_uc,
            "lr_ind_stat": lr_ind,
            "degenerate": True,
            "method": "degenerate_uc_boundary",
        }
    if uc_degen and ind_degen:
        return {
            "lr_stat": float("nan"),
            "p_value": float("nan"),
            "lr_uc_stat": lr_uc,
            "lr_ind_stat": lr_ind,
            "degenerate": True,
            "method": "degenerate_both",
        }
    if uc_degen:
        return {
            "lr_stat": float("nan"),
            "p_value": float("nan"),
            "lr_uc_stat": lr_uc,
            "lr_ind_stat": lr_ind,
            "degenerate": True,
            "method": "degenerate_uc",
        }
    if ind_degen:
        return {
            "lr_stat": float("nan"),
            "p_value": float("nan"),
            "lr_uc_stat": lr_uc,
            "lr_ind_stat": lr_ind,
            "degenerate": True,
            "method": "degenerate_ind",
        }
    if not np.isfinite(lr_uc) or not np.isfinite(lr_ind):
        return {
            "lr_stat": float("nan"),
            "p_value": float("nan"),
            "lr_uc_stat": lr_uc,
            "lr_ind_stat": lr_ind,
            "degenerate": True,
            "method": "degenerate_non_finite",
        }

    lr_cc = lr_uc + lr_ind
    return {
        "lr_stat": float(lr_cc),
        "p_value": float(chi2.sf(lr_cc, df=2)),
        "lr_uc_stat": lr_uc,
        "lr_ind_stat": lr_ind,
        "degenerate": False,
        "method": "lr_chi2_df2",
    }


__all__ = [
    "lr_unconditional_coverage",
    "lr_independence_coverage",
    "lr_conditional_coverage",
]
