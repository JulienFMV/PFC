"""
_generate_phase05_fixture.py
----------------------------
Deterministic Phase-5 forwards-fixture generator.

Produces:
  - ``tests/fixtures/forwards_phase05_seed42.parquet`` (synthetic EEX-like forward
    quotes with Cal'27=30 EUR/MWh, July M-07'27=20 EUR/MWh dépressé, autres months
    positifs typiques EEX)
  - ``tests/fixtures/baseline_pfc_seed42_phase05.parquet`` (canonical Phase-5 baseline
    with floors OFF + delta-additive WV + spread-additive Peak, PFC_LT_USE_SEASONAL_HOURLY_SHAPE=1
    and PFC_LT_ALLOW_NEGATIVE_PRICES=1)

Used by:
  - ``tests/test_phase05_negative_prices.py::test_phase05_summer_bowl_negative_acceptance``
    (SC #2 ROADMAP)
  - ``tests/test_phase05_negative_prices.py::test_phase05_baseline_regression``
    (SC #5 ROADMAP)

Phase 5 D-A4-5 / D-A4-6.

REGENERATION POLICY
-------------------
CLI: ``python tests/fixtures/_generate_phase05_fixture.py``
     Generates both the forwards fixture and the Phase-5 baseline.

CLI for forwards only: ``python tests/fixtures/_generate_phase05_fixture.py --forwards-only``
CLI for baseline only: ``python tests/fixtures/_generate_phase05_fixture.py --generate-baseline``

WARNING: Regenerating the baseline after a math change requires updating the committed
parquet AND justifying the change. Convention D-A4-9 5bis-B "new baseline per math change
atomique" — this is the canonical Phase-5 baseline post-floors-off + delta-additif WV +
spread-additif Peak.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the repo root is importable when run as a script from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pfc_shaping.data.calendar_ch import enrich_15min_index  # noqa: E402
from pfc_shaping.lt.model.assembler import PFCAssembler  # noqa: E402
from pfc_shaping.lt.model.shape_hourly import ShapeHourly  # noqa: E402
from pfc_shaping.lt.model.shape_intraday import ShapeIntraday  # noqa: E402

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
_FORWARDS_OUT_PATH = Path(__file__).resolve().parent / "forwards_phase05_seed42.parquet"
_BASELINE_OUT_PATH = Path(__file__).resolve().parent / "baseline_pfc_seed42_phase05.parquet"
_REFERENCE_DATE = pd.Timestamp("2026-12-15", tz="UTC")


def build_phase05_forwards(seed: int = 42) -> dict[str, float]:
    """Build deterministic Phase-5 forward quotes (seed=42).

    Contract set: Cal'26, Cal'27, Cal'28, 4 quarters of 2027, 12 months of 2027,
    plus Peak counterparts. July M-07'27 is dépressé at 20 EUR/MWh — the bowl
    deepening from Phase 5bis-B pushes h13 Sunday below -20 EUR/MWh per SC #2.

    Key contracts:
    - Cal'27 = 30 EUR/MWh (D-A4-5)
    - July M-07'27 = 20 EUR/MWh dépressé (D-A4-5, RESEARCH Pitfall 5)
    - Q3'27 = 22 EUR/MWh (summer quarter, dépressé)
    - Other months = positive typical EEX

    Args:
        seed: Random seed for Peak jitter (default 42 → matches committed fixture).

    Returns:
        dict[str, float] — forward prices keyed by EEX convention (YYYY, YYYY-QN, YYYY-MM).
    """
    rng = np.random.default_rng(seed)

    # Base (flat) prices
    base_forwards: dict[str, float] = {
        # Calendar year products
        "2026": 28.0,
        "2027": 30.0,
        "2028": 32.0,
        # Quarterly 2027
        "2027-Q1": 35.0,
        "2027-Q2": 23.0,
        "2027-Q3": 22.0,   # summer quarter, dépressé
        "2027-Q4": 33.0,
        # Monthly 2027 — all positive EXCEPT July which is dépressé
        "2027-01": 33.0,
        "2027-02": 31.0,
        "2027-03": 28.0,
        "2027-04": 25.0,
        "2027-05": 24.0,
        "2027-06": 22.0,
        "2027-07": 20.0,   # <-- dépressé (RESEARCH Pitfall 5 — D-A4-5)
        "2027-08": 23.0,
        "2027-09": 25.0,
        "2027-10": 28.0,
        "2027-11": 32.0,
        "2027-12": 36.0,
    }

    # Peak counterparts: Base + jittered spread (EEX-typical ~5-8 EUR/MWh)
    # Use seed=42 for deterministic jitter
    forwards = dict(base_forwards)
    peak_spread_base = 6.0
    for key, base_price in list(base_forwards.items()):
        peak_key = f"{key}-Peak"
        jitter = float(rng.normal(0.0, 0.5))
        forwards[peak_key] = round(base_price + peak_spread_base + jitter, 4)

    return forwards


def _build_synthetic_epex(n_15min: int, rng: np.random.Generator) -> pd.DataFrame:
    """Return a synthetic 15-min EPEX DataFrame with duck-curve bowl deepening.

    Bowl calibration: summer Sunday/Saturday h10-15 prices go negative (driven by
    solar PV saturation) — this is required so ShapeHourly learns a negative f_H
    for that (saison='Ete', type_jour='Weekend', h=10-14) cell, which in turn
    produces negative price_shape when the base forward B is low (July=20 €/MWh).
    Without negative training prices, f_H = hourly_mean/daily_mean stays positive
    and price_shape = B * f_H stays positive regardless of how low B is.

    Design (Phase 5 SC #2 RESEARCH Pitfall 5):
    - Base price: 30 €/MWh (representative Swiss forward level)
    - Summer weekend h10-15 depression: -55 €/MWh → raw price ≈ -25 €/MWh
      (matches the –25 €/MWh minimum seen in the EEX negative price data)
    - Daily_mean(Ete, Weekend) ≈ 28 €/MWh after depression; f_H[h13] ≈ -25/28 ≈ -0.89
    - Resulting price_shape at B=20: f_H * B ≈ -0.89 * 20 ≈ -17.8 €/MWh

    UTC→Europe/Zurich: hour masks applied in local time (codex action #4) so the
    bowl sits at h10-15 Zurich time, not UTC.
    """
    idx = pd.date_range(
        start="2022-01-01",
        periods=n_15min,
        freq="15min",
        tz="UTC",
    )
    # Use local time (Europe/Zurich) for all seasonal/hour masks — codex action #4
    idx_local = idx.tz_convert("Europe/Zurich")
    hours = idx_local.hour + idx_local.minute / 60.0
    doy = idx_local.dayofyear.values
    dow = idx_local.dayofweek.values  # 0=Mon, 6=Sun

    # Base price + annual sinusoidal cycle
    price = (
        30.0
        + 5.0 * np.sin(2 * np.pi * doy / 365.0)
        + rng.standard_normal(n_15min) * 2.0
    )

    # Summer / weekend masks
    is_summer = (doy >= 152) & (doy <= 244)   # doy 152-244 ≈ June–Aug
    is_weekend = dow >= 5                      # Sat=5, Sun=6

    # Duck-curve bowl: summer weekend h10-15 deep solar depression
    # Depression must be strong enough to produce NEGATIVE prices at h10-15 on
    # summer weekends so ShapeHourly learns f_H < 0 for those cells.
    h10_15 = (hours >= 10) & (hours < 15)
    solar_depression = np.zeros(n_15min)
    solar_depression[h10_15 & is_summer & ~is_weekend] = -20.0   # summer weekday bowl
    solar_depression[h10_15 & is_summer & is_weekend] = -55.0    # deep WE bowl → raw ≈ -25

    price = price + solar_depression
    price = np.clip(price, -200.0, 200.0)
    return pd.DataFrame({"price_eur_mwh": price}, index=idx)


def build_phase05_baseline_pfc(seed: int = 42) -> pd.DataFrame:
    """Build the canonical Phase-5 baseline PFC (floors OFF + Phase 5 math).

    Produces the reference output for:
    - test_phase05_baseline_regression (SC #5)
    - Convention D-A4-9 5bis-B "new baseline per math change atomique"

    Env vars activated:
    - PFC_LT_USE_SEASONAL_HOURLY_SHAPE=1  (5bis-B bowl deepening)
    - PFC_LT_ALLOW_NEGATIVE_PRICES=1      (Phase 5 audit-trail INFO only)

    PFCAssembler ctor args: ALL DEFAULTS (negative-ready):
    - enforce_positivity=False
    - enforce_m_factor_floor=False
    - enforce_floor=False
    - allow_negative_peak=True

    Args:
        seed: Random seed for full reproducibility (default 42).

    Returns:
        pd.DataFrame — PFC output from assembler.build(), same schema as other baselines.
    """
    # Save and set env vars
    _saved = {}
    for var in ("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", "PFC_LT_ALLOW_NEGATIVE_PRICES"):
        _saved[var] = os.environ.get(var)
    os.environ["PFC_LT_USE_SEASONAL_HOURLY_SHAPE"] = "1"
    os.environ["PFC_LT_ALLOW_NEGATIVE_PRICES"] = "1"

    try:
        # Deterministic seeds
        random.seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        # Synthetic EPEX: 3 calendar years at 15-min resolution
        n_15min = 3 * 365 * 96
        epex_df = _build_synthetic_epex(n_15min, rng)
        calendar_df = enrich_15min_index(epex_df.index, country="CH")

        # Fit ShapeHourly with use_seasonal_hourly=True (5bis-B bowl deepening)
        sh = ShapeHourly(use_seasonal_hourly=True).fit(epex_df, calendar_df)
        si = ShapeIntraday().fit(epex_df, entso_df=None, calendar_df=calendar_df)

        # Build PFCAssembler with Phase 5 defaults (all floors OFF)
        assembler = PFCAssembler(
            shape_hourly=sh,
            shape_intraday=si,
            uncertainty=None,
            water_value=None,
            cascader=None,
            calibrator=None,
            # Phase 5 defaults (negative-ready — all floors OFF per D-A2-1)
            enforce_positivity=False,
            enforce_m_factor_floor=False,
            enforce_floor=False,
            allow_negative_peak=True,
        )

        # Load forwards from committed fixture
        forwards_path = Path(__file__).resolve().parent / "forwards_phase05_seed42.parquet"
        if forwards_path.exists():
            fwd_df = pd.read_parquet(str(forwards_path))
            # Convert DataFrame to dict[str, float]
            if "contract_key" in fwd_df.columns and "price_eur_mwh" in fwd_df.columns:
                forwards = dict(zip(fwd_df["contract_key"], fwd_df["price_eur_mwh"]))
            elif fwd_df.index.name == "contract_key" or fwd_df.index.dtype == object:
                forwards = fwd_df.iloc[:, 0].to_dict()
            else:
                # Fallback: use build_phase05_forwards()
                forwards = build_phase05_forwards(seed=seed)
        else:
            forwards = build_phase05_forwards(seed=seed)

        # Build PFC: 31-day horizon starting July 2027 (the dépressé month)
        return assembler.build(
            base_prices=forwards,
            start_date="2027-07-01",
            horizon_days=31,
            reference_date=_REFERENCE_DATE,
            country="CH",
        )
    finally:
        # Restore env vars
        for var, val in _saved.items():
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = val


def main() -> None:
    """Generate Phase-5 forwards fixture and optionally the canonical baseline."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate Phase-5 test fixtures")
    parser.add_argument("--forwards-only", action="store_true",
                        help="Generate only the forwards fixture, not the baseline")
    parser.add_argument("--generate-baseline", action="store_true",
                        help="Also generate the Phase-5 baseline PFC parquet")
    args, _ = parser.parse_known_args()

    generate_baseline = args.generate_baseline or (not args.forwards_only)

    # ── Generate forwards fixture ───────────────────────────────────────
    forwards = build_phase05_forwards(seed=42)
    fwd_df = pd.DataFrame({
        "contract_key": list(forwards.keys()),
        "price_eur_mwh": list(forwards.values()),
    })
    _FORWARDS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fwd_df.to_parquet(str(_FORWARDS_OUT_PATH), index=True)

    import hashlib
    sha_fwd = hashlib.sha256(open(_FORWARDS_OUT_PATH, "rb").read()).hexdigest()
    print(
        f"Forwards fixture: rows={len(fwd_df)} path={_FORWARDS_OUT_PATH.relative_to(_REPO_ROOT)} "
        f"sha256={sha_fwd[:16]}... (D-A4-5)"
    )

    # ── Generate Phase-5 baseline PFC ──────────────────────────────────
    if generate_baseline:
        print("Generating Phase-5 baseline PFC (floors OFF + PFC_LT_USE_SEASONAL_HOURLY_SHAPE=1)...")
        df_pfc = build_phase05_baseline_pfc(seed=42)
        _BASELINE_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_pfc.to_parquet(str(_BASELINE_OUT_PATH), index=True)

        sha_bl = hashlib.sha256(open(_BASELINE_OUT_PATH, "rb").read()).hexdigest()
        price_col = "price_shape" if "price_shape" in df_pfc.columns else df_pfc.columns[0]
        print(
            f"Phase-5 baseline: rows={len(df_pfc)}"
            f" cols={list(df_pfc.columns)}"
            f" min={df_pfc[price_col].min():.2f}"
            f" max={df_pfc[price_col].max():.2f}"
            f" mean={df_pfc[price_col].mean():.2f}"
            f" sha256={sha_bl[:16]}... (D-A4-6)"
        )


if __name__ == "__main__":
    main()
