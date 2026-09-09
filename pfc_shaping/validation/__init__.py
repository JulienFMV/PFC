"""
pfc_shaping.validation
----------------------
Validation harness package : backtest walk-forward, comparaison vs HFC,
et — depuis Phase 10 — scorecard SOTA 5-pillar + block masks tz-aware.

Modules
-------
backtest
    WalkForwardBacktest legacy (recalibration mensuelle, KPIs shape ratios).
compare_hfc
    Quick MAE/RMSE/bias PFC vs HFC OMPEX (alignment helpers).
block_masks  [Phase 10]
    5 BlockMask classes D-A2-2 (overnight_weekday, midday_weekday,
    weekend_midday, summer_solar_bowl, winter_evening_peak) avec
    `apply(idx_utc: pd.DatetimeIndex) -> np.ndarray[bool]` tz-aware Europe/Zurich.
scorecard    [Phase 10]
    Harness PFC FMV Quality Scorecard (24 vintages × 4 configs ablation grid),
    skeleton seul Plan 10-01 ; pillars 1-5 wirés Plans 10-02/03/04.
"""

__all__ = ["backtest", "compare_hfc", "block_masks", "scorecard"]
