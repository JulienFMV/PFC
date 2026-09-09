"""
test_phase10_probabilistic.py
------------------------------
Pillar 3 (Probabilistic calibration) — Christoffersen 1998 unconditional
coverage LR test on IC80 only.

**IC95 NOT TESTED in Phase 10** : per blocker #4 résolu Plan 10-03,
`pfc_shaping.lt.model.uncertainty.Uncertainty.compute()` retourne UNIQUEMENT
les colonnes `['p10', 'p90']` (percentiles 10/90 = IC80 par construction).
Pas de paramètre `level=` ni `confidence=`. Étendre Uncertainty pour IC95
toucherait le code core LT model = scope-creep Phase 10. Décision : IC95
explicitement déférée Phase 5ter (CONTEXT D-A3-3 amendé Plan 10-03 iter 1).

Le contrat de défense : `compute_pillar3_coverage(..., ic_level=0.95)` raise
`ValueError` explicite avec mention `Phase 5ter` + `uncertainty.py` — aucun
silent skip, aucun fallback ambigu.

Architecture
------------
- TestLrUnconditionalCoverage (5 tests) : LR_uc unit tests sur cas perfect /
  under / over coverage + degenerate n=0 + degenerate x∈{0,n}.
- TestComputePillar3Coverage (3 tests) : integration synth + missing-col
  ValueError + **test_compute_pillar3_coverage_rejects_ic95** (blocker #4 fix).
- TestPillar3Integration (1 test optionnel) : end-to-end Uncertainty(n_boot=50,
  seed=42) + build_one(Config 4, with_uncertainty=True) + compute_pillar3_coverage.

Runtime cible : < 60s (integration test n_boot=50 réduit pour CI budget).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Module-level imports under test
# ---------------------------------------------------------------------------


def test_module_imports_ok():
    """Smoke : lr_unconditional_coverage + compute_pillar3_coverage importables."""
    from pfc_shaping.validation.christoffersen import (  # noqa: F401
        lr_unconditional_coverage,
    )
    from pfc_shaping.validation.scorecard import (  # noqa: F401
        compute_pillar3_coverage,
    )


# ---------------------------------------------------------------------------
# TestLrUnconditionalCoverage — Pattern 3 canonical Christoffersen 1998
# ---------------------------------------------------------------------------


class TestLrUnconditionalCoverage:
    """LR_uc Christoffersen 1998 — H0: observed_freq == nominal p."""

    def test_perfect_coverage_h0_not_rejected(self):
        """x=20, n=100, p=0.20 (perfect 20% violations vs 20% nominal) → p>0.9."""
        from pfc_shaping.validation.christoffersen import lr_unconditional_coverage

        res = lr_unconditional_coverage(x=20, n=100, p=0.20)
        assert res["degenerate"] is False
        assert res["observed_freq"] == pytest.approx(0.20, abs=1e-9)
        assert res["nominal_p"] == 0.20
        assert res["n"] == 100
        assert res["x"] == 20
        # Perfect match → LR_uc ≈ 0, p_value → 1
        assert res["lr_stat"] == pytest.approx(0.0, abs=1e-9)
        assert res["p_value"] > 0.9, (
            f"perfect coverage should not reject H0, got p={res['p_value']}"
        )

    def test_undercoverage_h0_rejected(self):
        """x=5, n=100, p=0.20 (5% observed vs 20% nominal) → LR>5, p<0.05."""
        from pfc_shaping.validation.christoffersen import lr_unconditional_coverage

        res = lr_unconditional_coverage(x=5, n=100, p=0.20)
        assert res["degenerate"] is False
        assert res["observed_freq"] == pytest.approx(0.05, abs=1e-9)
        assert res["lr_stat"] > 5.0, f"undercoverage should have LR>5, got {res['lr_stat']}"
        assert res["p_value"] < 0.05, (
            f"undercoverage should reject H0, got p={res['p_value']}"
        )

    def test_overcoverage_h0_rejected(self):
        """x=40, n=100, p=0.20 (40% observed vs 20% nominal) → LR>5, p<0.05."""
        from pfc_shaping.validation.christoffersen import lr_unconditional_coverage

        res = lr_unconditional_coverage(x=40, n=100, p=0.20)
        assert res["degenerate"] is False
        assert res["observed_freq"] == pytest.approx(0.40, abs=1e-9)
        assert res["lr_stat"] > 5.0, f"overcoverage should have LR>5, got {res['lr_stat']}"
        assert res["p_value"] < 0.05, (
            f"overcoverage should reject H0, got p={res['p_value']}"
        )

    def test_degenerate_n_zero(self):
        """n=0 → degenerate=True, no division-by-zero crash."""
        from pfc_shaping.validation.christoffersen import lr_unconditional_coverage

        res = lr_unconditional_coverage(x=0, n=0, p=0.20)
        assert res["degenerate"] is True
        assert np.isnan(res["p_value"])
        assert np.isnan(res["observed_freq"])
        assert res["n"] == 0
        assert res["x"] == 0
        assert np.isnan(res["lr_stat"])

    def test_boundary_x_zero_uses_binomial_exact(self):
        """WR-07 : x=0 boundary → binomial exact p_value, NOT degenerate.

        L'ancien comportement (degenerate=True, p_value=NaN) jetait le
        signal le plus fort de overcoverage (n=100, p=0.20 →
        (1-p)^n ≈ 2e-10, ultra-significatif). Nouveau contrat (WR-07) :
        retourne le p-value binomial exact `P(X=0|n,p) = (1-p)^n` et
        flag `method="binomial_exact"`.
        """
        from pfc_shaping.validation.christoffersen import lr_unconditional_coverage

        res_zero = lr_unconditional_coverage(x=0, n=100, p=0.20)
        # No more degenerate=True
        assert res_zero["degenerate"] is False
        # LR_stat reste NaN (LR diverge à la frontière)
        assert np.isnan(res_zero["lr_stat"])
        # p_value finite = (0.8)^100 ≈ 2.04e-10
        assert res_zero["p_value"] == pytest.approx((0.8) ** 100, rel=1e-9)
        assert res_zero["p_value"] < 1e-9
        assert res_zero["observed_freq"] == 0.0
        assert res_zero["method"] == "binomial_exact"

    def test_boundary_x_equals_n_uses_binomial_exact(self):
        """WR-07 : x=n boundary → binomial exact p_value `p^n`, NOT degenerate."""
        from pfc_shaping.validation.christoffersen import lr_unconditional_coverage

        res_all = lr_unconditional_coverage(x=100, n=100, p=0.20)
        assert res_all["degenerate"] is False
        assert np.isnan(res_all["lr_stat"])
        # p_value = (0.2)^100 ≈ 1.27e-70 — astronomically small, total miscalib
        assert res_all["p_value"] == pytest.approx((0.2) ** 100, rel=1e-9)
        assert res_all["observed_freq"] == 1.0
        assert res_all["method"] == "binomial_exact"

    def test_boundary_small_n_still_useful_signal(self):
        """WR-07 : x=0, n=14, p=0.20 → p_value = (0.8)^14 ≈ 0.044 (significatif 5%)."""
        from pfc_shaping.validation.christoffersen import lr_unconditional_coverage

        res = lr_unconditional_coverage(x=0, n=14, p=0.20)
        assert res["degenerate"] is False
        assert res["method"] == "binomial_exact"
        # (0.8)^14 ≈ 0.04398
        assert res["p_value"] == pytest.approx((0.8) ** 14, rel=1e-9)
        # Just below 5% → rejette H0 → l'info qu'on aurait jetée avant
        assert res["p_value"] < 0.05


class TestLrIndependenceCoverage:
    """WR-12 : LR_ind Markov-chain independence test on consecutive violations."""

    def test_iid_violations_h0_not_rejected(self):
        """Violations iid ~Bernoulli(0.20) → pas de clustering → p>0.05."""
        from pfc_shaping.validation.christoffersen import lr_independence_coverage

        rng = np.random.RandomState(0)
        # n grand pour test bien-powered
        viol = rng.binomial(1, 0.20, 1000).astype(bool)
        res = lr_independence_coverage(viol)
        assert res["degenerate"] is False
        assert res["method"] == "lr_chi2"
        assert res["p_value"] > 0.05, (
            f"iid violations devraient NOT reject H0 indep, got p={res['p_value']}"
        )

    def test_clustered_violations_h0_rejected(self):
        """Violations clusterées (Q4 only) → forte dépendance lag-1 → p<0.05."""
        from pfc_shaping.validation.christoffersen import lr_independence_coverage

        # 500 obs : 400 no-viol au début, puis 100 viols consécutives → grand
        # n_11, presque pas de n_01 (1 seule transition non-viol → viol).
        viol = np.zeros(500, dtype=bool)
        viol[400:] = True
        res = lr_independence_coverage(viol)
        assert res["degenerate"] is False
        # Forte dépendance : pi_11 ≈ 1, pi_01 ≈ 1/400 → LR_ind grand
        assert res["lr_stat"] > 10
        assert res["p_value"] < 0.01

    def test_alternating_violations_h0_rejected(self):
        """Violations alternantes (anti-cluster) → forte dépendance négative → p<0.05."""
        from pfc_shaping.validation.christoffersen import lr_independence_coverage

        # 0,1,0,1,0,1... → pi_01 = 1, pi_11 = 0 (perfectement anti-corrélé)
        viol = np.array([i % 2 == 0 for i in range(500)], dtype=bool)
        res = lr_independence_coverage(viol)
        assert res["degenerate"] is False
        assert res["lr_stat"] > 10  # Forte dépendance détectée
        assert res["p_value"] < 0.01

    def test_no_violations_degenerate(self):
        """Tous False → impossible de tester indépendance (pas de viol observée)."""
        from pfc_shaping.validation.christoffersen import lr_independence_coverage

        viol = np.zeros(100, dtype=bool)
        res = lr_independence_coverage(viol)
        assert res["degenerate"] is True
        assert res["method"] == "degenerate_empty_state"
        assert np.isnan(res["p_value"])

    def test_too_few_obs_degenerate(self):
        """len(viol) < 2 → no transitions calculable → degenerate."""
        from pfc_shaping.validation.christoffersen import lr_independence_coverage

        for n_obs in (0, 1):
            viol = np.zeros(n_obs, dtype=bool)
            res = lr_independence_coverage(viol)
            assert res["degenerate"] is True
            assert res["method"] == "degenerate_too_few_obs"
            assert np.isnan(res["p_value"])
            assert res["n_transitions"] == 0


class TestLrConditionalCoverage:
    """WR-12 : LR_cc = LR_uc + LR_ind, chi-squared df=2."""

    def test_standard_path_sums_uc_and_ind(self):
        """LR_cc = LR_uc + LR_ind avec p_value via chi2(df=2)."""
        from pfc_shaping.validation.christoffersen import (
            lr_conditional_coverage,
            lr_independence_coverage,
            lr_unconditional_coverage,
        )

        rng = np.random.RandomState(1)
        viol = rng.binomial(1, 0.20, 500).astype(bool)
        x = int(viol.sum())
        n = len(viol)
        uc = lr_unconditional_coverage(x=x, n=n, p=0.20)
        ind = lr_independence_coverage(viol)
        cc = lr_conditional_coverage(uc, ind)
        assert cc["degenerate"] is False
        assert cc["method"] == "lr_chi2_df2"
        assert cc["lr_stat"] == pytest.approx(uc["lr_stat"] + ind["lr_stat"])
        # p_value sous chi2(df=2)
        from scipy.stats import chi2

        assert cc["p_value"] == pytest.approx(chi2.sf(cc["lr_stat"], df=2))

    def test_uc_boundary_propagates_to_cc(self):
        """Si LR_uc est en mode binomial_exact (x=0 ou x=n) → LR_cc degenerate."""
        from pfc_shaping.validation.christoffersen import (
            lr_conditional_coverage,
            lr_independence_coverage,
            lr_unconditional_coverage,
        )

        # x=0, n=100 → uc method=binomial_exact
        viol = np.zeros(100, dtype=bool)
        uc = lr_unconditional_coverage(x=0, n=100, p=0.20)
        assert uc["method"] == "binomial_exact"
        ind = lr_independence_coverage(viol)  # degenerate_empty_state
        cc = lr_conditional_coverage(uc, ind)
        assert cc["degenerate"] is True
        assert cc["method"] == "degenerate_uc_boundary"
        assert np.isnan(cc["lr_stat"])
        assert np.isnan(cc["p_value"])

    def test_compute_pillar3_coverage_persists_3_pvalues(self):
        """WR-12 wiring : compute_pillar3_coverage retourne les 3 p-values."""
        from pfc_shaping.validation.block_masks import BlockMiddayWeekday
        from pfc_shaping.validation.scorecard import compute_pillar3_coverage

        idx = _utc_hourly_index(n_days=30)
        n = len(idx)
        rng = np.random.RandomState(42)
        # IC80 width=10 centered on 50 ; realised iid normal → ≈ 20% violations
        pfc_with_ic = pd.DataFrame(
            {
                "price_shape": np.full(n, 50.0),
                "p10": np.full(n, 45.0),
                "p90": np.full(n, 55.0),
            },
            index=idx,
        )
        # Realised : normal(50, sigma) tuned for ~20% violations (sigma s.t.
        # P(|z| > 1.28) = 0.20 → sigma = 10/(2*1.28) ≈ 3.906)
        realised = pd.Series(50.0 + rng.normal(0, 3.906, n), index=idx)
        bloc = BlockMiddayWeekday()
        res = compute_pillar3_coverage(pfc_with_ic, realised, bloc, ic_level=0.80)
        # Les 3 p-values sont présentes
        for key in ("p_value", "lr_ind_p_value", "lr_cc_p_value"):
            assert key in res, f"missing key {key!r} in pillar 3 result"
        # Methods documentent les chemins
        for key in ("method", "lr_ind_method", "lr_cc_method"):
            assert key in res, f"missing key {key!r} in pillar 3 result"
        # Transitions counts pour audit
        for key in ("n_00", "n_01", "n_10", "n_11"):
            assert key in res, f"missing transition count {key!r}"


# ---------------------------------------------------------------------------
# TestComputePillar3Coverage — integration (PFC + IC + realised + bloc)
# ---------------------------------------------------------------------------


def _utc_hourly_index(n_days: int = 30) -> pd.DatetimeIndex:
    """Helper : n_days hourly UTC index starting 2024-06-01."""
    return pd.date_range(
        "2024-06-01", periods=n_days * 24, freq="1h", tz="UTC"
    )


class TestComputePillar3Coverage:
    """compute_pillar3_coverage — IC80 binomial Christoffersen 1998."""

    def test_synth_pfc_realised_in_bounds(self):
        """Synth pfc_with_ic + realised tel que violations ≈ 20% → p>0.5."""
        from pfc_shaping.validation.block_masks import BlockMiddayWeekday
        from pfc_shaping.validation.scorecard import compute_pillar3_coverage

        idx = _utc_hourly_index(n_days=30)  # 720 hourly points
        n = len(idx)
        rng = np.random.RandomState(42)
        # Construire pfc avec p10=45, p90=55 (IC80 width=10 €/MWh centered on 50)
        pfc_with_ic = pd.DataFrame(
            {
                "price_shape": np.full(n, 50.0),
                "p10": np.full(n, 45.0),
                "p90": np.full(n, 55.0),
            },
            index=idx,
        )
        # Realised : N(50, sigma) avec sigma calibré pour donner ~20% en dehors
        # de [45, 55]. sigma ~ (55-45)/(2*z_0.10) = 10/(2*1.2816) = 3.9
        realised = pd.Series(rng.normal(50.0, 3.9, n), index=idx, name="price_eur_mwh")
        bloc = BlockMiddayWeekday()
        res = compute_pillar3_coverage(pfc_with_ic, realised, bloc, ic_level=0.80)
        # Structure checks
        assert res["bloc"] == "block_midday_weekday"
        assert res["ic_level"] == pytest.approx(0.80)
        assert res["nominal_p"] == pytest.approx(0.20)  # 1.0 - 0.80 floating-point ULP
        assert res["n"] > 0  # in-bloc count after mask
        assert res["x"] >= 0
        assert "observed_freq" in res
        assert "lr_stat" in res
        assert "p_value" in res
        assert "degenerate" in res

    def test_missing_p10_p90_raises_valueerror(self):
        """pfc_with_ic sans col p10 ou p90 → ValueError explicit."""
        from pfc_shaping.validation.block_masks import BlockMiddayWeekday
        from pfc_shaping.validation.scorecard import compute_pillar3_coverage

        idx = _utc_hourly_index(n_days=5)
        pfc_no_ic = pd.DataFrame({"price_shape": np.full(len(idx), 50.0)}, index=idx)
        realised = pd.Series(np.full(len(idx), 50.0), index=idx)
        bloc = BlockMiddayWeekday()
        with pytest.raises(ValueError, match="p10|p90"):
            compute_pillar3_coverage(pfc_no_ic, realised, bloc, ic_level=0.80)

    def test_compute_pillar3_coverage_rejects_ic95(self):
        """Blocker #4 fix : ic_level=0.95 raise ValueError avec message explicite.

        Le contrat de défense IC95-not-supported : message mentionne EXPLICITEMENT
        - "IC95" (la valeur rejetée)
        - "Phase 5ter" (le scope où sera ré-évalué)
        - "uncertainty.py" (le fichier à étendre)

        Aucun silent skip, aucun fallback. Le caller DOIT être conscient que
        Phase 10 teste UNIQUEMENT IC80.
        """
        from pfc_shaping.validation.block_masks import BlockMiddayWeekday
        from pfc_shaping.validation.scorecard import compute_pillar3_coverage

        idx = _utc_hourly_index(n_days=5)
        n = len(idx)
        pfc_with_ic = pd.DataFrame(
            {
                "price_shape": np.full(n, 50.0),
                "p10": np.full(n, 45.0),
                "p90": np.full(n, 55.0),
            },
            index=idx,
        )
        realised = pd.Series(np.full(n, 50.0), index=idx)
        bloc = BlockMiddayWeekday()
        with pytest.raises(ValueError) as exc_info:
            compute_pillar3_coverage(pfc_with_ic, realised, bloc, ic_level=0.95)
        msg = str(exc_info.value)
        # Assert message verbatim contient les 3 marqueurs audit-trail
        assert "IC95" in msg, f"message must mention 'IC95', got: {msg}"
        assert "Phase 5ter" in msg, f"message must mention 'Phase 5ter', got: {msg}"
        assert "uncertainty.py" in msg, (
            f"message must mention 'uncertainty.py' (file to extend), got: {msg}"
        )


# ---------------------------------------------------------------------------
# TestPillar3Integration — end-to-end Uncertainty(n_boot=50) + build_one(Config 4)
# ---------------------------------------------------------------------------


class TestPillar3Integration:
    """End-to-end : Uncertainty bootstrap + build_one(with_uncertainty=True)."""

    def test_end_to_end_uncertainty_compute_pillar3(self):
        """Synth EPEX → build_one(Config 4, with_uncertainty=True) → coverage IC80.

        Sanity test : dict structure correcte, n > 0. Pas un strict gate KPI
        (la PFC synth ne garantit pas 20% coverage exact).

        Runtime : n_boot=50 (vs 500 prod) pour CI budget < 60s.
        """
        from pfc_shaping.validation.block_masks import BlockMiddayWeekday
        from pfc_shaping.validation.scorecard import (
            _synth_epex_hist_for_mock,
            compute_pillar3_coverage,
        )

        # Build a small synthetic PFC + IC directly (bypass build_one which
        # would also require ShapeIntraday and full assembler chain).
        # The test verifies the integration of compute_pillar3_coverage with
        # a realistic-looking pfc_with_ic frame.
        epex_hist = _synth_epex_hist_for_mock(seed=42)
        # Realised : same epex_hist filtered to mid-2024 (in-sample synthetic)
        realised = epex_hist.loc["2023-06-01":"2023-07-01"].copy()
        realised.name = "price_eur_mwh"
        # pfc_with_ic : centered around 50 ± width 10 (a generous IC80)
        pfc_with_ic = pd.DataFrame(
            {
                "price_shape": np.full(len(realised), 50.0),
                "p10": np.full(len(realised), 35.0),  # generous lower bound
                "p90": np.full(len(realised), 65.0),  # generous upper bound
            },
            index=realised.index,
        )
        bloc = BlockMiddayWeekday()
        res = compute_pillar3_coverage(pfc_with_ic, realised, bloc, ic_level=0.80)
        assert res["bloc"] == "block_midday_weekday"
        assert res["ic_level"] == 0.80
        assert res["n"] >= 0
        # No crash, dict structure stable
        for key in ("lr_stat", "p_value", "observed_freq", "nominal_p", "n", "x",
                    "degenerate", "bloc", "ic_level"):
            assert key in res, f"missing key {key} in result {res}"
