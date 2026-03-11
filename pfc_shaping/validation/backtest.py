"""
backtest.py
-----------
Walk-forward backtest du modÃ¨le de shaping 15min.

MÃ©thodologie :
    - PÃ©riode de test : configurable (ex. 2021-01-01 â†’ 2024-12-31)
    - Recalibration mensuelle sur donnÃ©es passÃ©es (lookback = 24 mois)
    - PrÃ©diction out-of-sample du mois suivant
    - Comparaison prix_prÃ©dit vs prix_EPEX_rÃ©el

KPIs calculÃ©s :
    - RMSE intra-horaire (shape seul, normalisÃ©)
    - MAE
    - Biais moyen par heure
    - Couverture IC 80% (si uncertainty calibrÃ©e)
    - Skill score vs climatologie (profil moyen flat)

Sortie :
    - DataFrame de rÃ©sultats mensuels
    - Figures de diagnostic (optionnel avec matplotlib)

Usage :
    from validation.backtest import WalkForwardBacktest
    bt = WalkForwardBacktest(start='2021-01-01', end='2024-12-31')
    results = bt.run(epex_df, entso_df)
    bt.report(results)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.model.shape_hourly import ShapeHourly
from pfc_shaping.model.shape_intraday import ShapeIntraday
from pfc_shaping.model.uncertainty import Uncertainty
from pfc_shaping.model.assembler import PFCAssembler

logger = logging.getLogger(__name__)

LOOKBACK_MONTHS = 24  # mois de donnÃ©es pour chaque recalibration


@dataclass
class BacktestResult:
    period: str          # 'YYYY-MM'
    rmse_shape: float    # RMSE sur les facteurs f_Q (shape seul)
    mae_shape: float
    bias_mean: float     # biais moyen (prÃ©dit - rÃ©el) / rÃ©el
    ic80_coverage: float # fraction des rÃ©els dans [p10, p90]
    skill_score: float   # skill vs flat (0 = pas mieux, 1 = parfait)
    n_obs: int


@dataclass
class BacktestReport:
    results: list[BacktestResult] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(r) for r in self.results]).set_index("period")

    def summary(self) -> dict:
        df = self.to_dataframe()
        return {
            "RMSE_shape_mean":   df["rmse_shape"].mean(),
            "MAE_shape_mean":    df["mae_shape"].mean(),
            "Bias_mean":         df["bias_mean"].mean(),
            "IC80_coverage":     df["ic80_coverage"].mean(),
            "Skill_score_mean":  df["skill_score"].mean(),
            "N_periods":         len(df),
        }


class WalkForwardBacktest:
    """
    Walk-forward backtest avec recalibration mensuelle.

    Args:
        start         : dÃ©but de la pÃ©riode de test 'YYYY-MM-DD'
        end           : fin de la pÃ©riode de test 'YYYY-MM-DD'
        lookback_months: fenÃªtre de calibration (mois)
        base_price    : niveau de base fixe en â‚¬/MWh (pour reconstruire le prix)
                        Si None, le prix est prÃ©dit en shape pur (ratios f_Q)
    """

    def __init__(
        self,
        start: str = "2021-01-01",
        end: str = "2024-12-31",
        lookback_months: int = LOOKBACK_MONTHS,
        base_price: float = 70.0,
    ) -> None:
        self.start = pd.Timestamp(start, tz="UTC")
        self.end = pd.Timestamp(end, tz="UTC")
        self.lookback_months = lookback_months
        self.base_price = base_price

    def run(
        self,
        epex_df: pd.DataFrame,
        entso_df: pd.DataFrame | None = None,
        with_uncertainty: bool = True,
    ) -> BacktestReport:
        """
        Lance le backtest walk-forward.

        Args:
            epex_df        : historique EPEX 15min complet
            entso_df       : historique ENTSO-E (optionnel)
            with_uncertainty: calibrer et Ã©valuer les IC bootstrap

        Returns:
            BacktestReport
        """
        report = BacktestReport()

        # ItÃ©ration mensuelle sur la pÃ©riode de test
        periods = pd.period_range(
            self.start.to_period("M"),
            self.end.to_period("M"),
            freq="M"
        )

        for period in periods:
            period_start = period.to_timestamp(how="start").tz_localize("UTC")
            period_end = period.to_timestamp(how="end").tz_localize("UTC")
            train_start = period_start - pd.DateOffset(months=self.lookback_months)

            # DonnÃ©es de calibration (train)
            train_epex = epex_df[(epex_df.index >= train_start) & (epex_df.index < period_start)]
            train_entso = (
                entso_df[(entso_df.index >= train_start) & (entso_df.index < period_start)]
                if entso_df is not None else None
            )

            # DonnÃ©es de test (out-of-sample)
            test_epex = epex_df[(epex_df.index >= period_start) & (epex_df.index <= period_end)]

            if len(train_epex) < 96 * 30 or len(test_epex) < 96:
                logger.debug("PÃ©riode %s : donnÃ©es insuffisantes â€” ignorÃ©e", period)
                continue

            logger.info("Backtest %s : train=%d obs, test=%d obs", period, len(train_epex), len(test_epex))

            try:
                result = self._backtest_period(
                    period_str=str(period),
                    train_epex=train_epex,
                    train_entso=train_entso,
                    test_epex=test_epex,
                    test_entso=(
                        entso_df[(entso_df.index >= period_start) & (entso_df.index <= period_end)]
                        if entso_df is not None else None
                    ),
                    with_uncertainty=with_uncertainty,
                )
                report.results.append(result)
            except Exception as e:
                logger.warning("Backtest %s Ã©chouÃ© : %s", period, e)

        logger.info("Backtest terminÃ© : %d pÃ©riodes. RÃ©sumÃ© :", len(report.results))
        for k, v in report.summary().items():
            logger.info("  %s = %.4f", k, v)

        return report

    def _backtest_period(
        self,
        period_str: str,
        train_epex: pd.DataFrame,
        train_entso: pd.DataFrame | None,
        test_epex: pd.DataFrame,
        test_entso: pd.DataFrame | None,
        with_uncertainty: bool,
    ) -> BacktestResult:
        """Calibre et Ã©value sur une pÃ©riode mensuelle."""

        # Calibration
        cal_train = enrich_15min_index(train_epex.index)
        sh = ShapeHourly().fit(train_epex, cal_train)
        si = ShapeIntraday().fit(train_epex, train_entso, cal_train)

        unc = None
        if with_uncertainty:
            unc = Uncertainty(n_boot=200, seed=0).fit(train_epex, cal_train)

        # PrÃ©diction sur pÃ©riode de test
        assembler = PFCAssembler(sh, si, unc)
        base_prices = {period_str[:4]: self.base_price}
        pfc_pred = assembler.build(
            base_prices=base_prices,
            start_date=test_epex.index.min().strftime("%Y-%m-%d"),
            horizon_days=35,
        )

        # Alignement avec donnÃ©es rÃ©elles
        common_idx = pfc_pred.index.intersection(test_epex.index)
        if len(common_idx) == 0:
            raise ValueError("Aucun index commun entre prÃ©diction et test")

        pred = pfc_pred.loc[common_idx, "price_shape"]
        real = test_epex.loc[common_idx, "price_eur_mwh"]

        # â”€â”€ KPIs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        # RMSE et MAE sur les ratios de shape (facteur f_Q)
        # On compare les ratios plutÃ´t que les prix absolus (le niveau est fixÃ© par EULER)
        cal_test = enrich_15min_index(common_idx)
        real_ratios = _compute_intraday_ratios(real, cal_test)
        pred_ratios = _compute_intraday_ratios(pred, cal_test)

        valid = ~(np.isnan(real_ratios) | np.isnan(pred_ratios))
        if valid.sum() == 0:
            raise ValueError("Pas de ratios valides pour l'Ã©valuation")

        err = pred_ratios[valid] - real_ratios[valid]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        bias = float(np.mean(err))

        # Skill score vs flat (ratio = 1 partout)
        err_flat = 1.0 - real_ratios[valid]
        mse_flat = np.mean(err_flat ** 2)
        skill = float(1.0 - np.mean(err ** 2) / max(mse_flat, 1e-10))

        # Couverture IC 80%
        ic80_cov = 0.0
        if unc is not None and "p10" in pfc_pred.columns and "p90" in pfc_pred.columns:
            p10 = pfc_pred.loc[common_idx, "p10"]
            p90 = pfc_pred.loc[common_idx, "p90"]
            in_ic = (real >= p10) & (real <= p90)
            ic80_cov = float(in_ic.mean())

        return BacktestResult(
            period=period_str,
            rmse_shape=rmse,
            mae_shape=mae,
            bias_mean=bias,
            ic80_coverage=ic80_cov,
            skill_score=skill,
            n_obs=int(valid.sum()),
        )

    def report(self, report: BacktestReport) -> None:
        """Affiche un rÃ©sumÃ© console du backtest."""
        print("\n=== BACKTEST WALK-FORWARD â€” PFC Shaping 15min ===")
        df = report.to_dataframe()
        print(df.to_string(float_format="{:.4f}".format))
        print("\n--- RÃ©sumÃ© ---")
        for k, v in report.summary().items():
            print(f"  {k:<25} = {v:.4f}")
        print("=" * 50)


# ---------------------------------------------------------------------------
# Utilitaire
# ---------------------------------------------------------------------------

def _compute_intraday_ratios(prices: pd.Series, cal: pd.DataFrame) -> np.ndarray:
    """
    Calcule le ratio prix_quart / prix_moyen_heure pour chaque point 15min.
    Retourne un array alignÃ© sur prices.index.
    """
    df = pd.DataFrame({"price": prices})
    df = df.join(cal[["heure_hce"]])
    df["hour_key"] = df.index.floor("h")
    hour_means = df.groupby("hour_key")["price"].transform("mean")
    ratios = np.where(hour_means.abs() > 0.1, df["price"] / hour_means, np.nan)
    return ratios
