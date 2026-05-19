"""
_generate_bowl_fixture.py
-------------------------
Deterministic bowl-fixture generator for Phase 5bis-B.

PURPOSE
-------
Produces ``tests/fixtures/bowl_seed42.parquet`` — a synthetic 15-min EPEX
DataFrame with an analytically-controlled duck curve — used by
``tests/test_shape_hourly_bowl.py`` for SC #1, SC #2, SC #3 validation on
synthetic data.

Innovation gating (RESEARCH §Innovation gating, D-A4-1):
    Pass SC #1/#2/#3 on SYNTH = condition *nécessaire* (math correcte).
    Phase 10 validates on HFC OMPEX RÉEL = condition *suffisante* (data fit).
    Failure synth in 5bis-B = math broken (ship-blocker immédiat).
    Pass synth + failure Phase 10 = fixture-real gap (informe future fixture
    design, NOT a rollback 5bis-B).

Decision reference: D-A4-1 (CONTEXT.md §Test design Area 4).

REGENERATION POLICY
-------------------
CLI: ``python tests/fixtures/_generate_bowl_fixture.py``

Re-running this script produces a byte-identical ``bowl_seed42.parquet``
(seed=42 is fixed). Do NOT regenerate without re-running
``python scripts/calibrate_bowl_thresholds.py`` afterwards to refresh the
calibration report (the sha256 link between report and fixture will break).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the repo root is importable when run as a script from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Output path (relative to repo root)
# ---------------------------------------------------------------------------
_OUT_PATH = Path(__file__).resolve().parent / "bowl_seed42.parquet"


def _build_bowl_epex(n_15min: int, rng: np.random.Generator) -> pd.DataFrame:
    """Return a synthetic 15-min EPEX DataFrame with analytically-controlled duck curve.

    Bowl is injected as a deterministic signal modulated by:
      - Hour of day (solar depression h10-15, evening peak h17-20, night h22-6)
      - Season (summer stronger than winter for solar depression)
      - Day type (weekend summer solar bowl deeper)

    Base level: 80 EUR/MWh (higher than baseline fixture to have headroom for bowl).
    Noise: N(0, 2) EUR/MWh (smaller than baseline to preserve signal/noise ratio).

    Args:
        n_15min: Number of 15-min slots to generate.
        rng: NumPy random generator (seed-controlled by caller).

    Returns:
        Long-format DataFrame with column ``price_eur_mwh``, DatetimeIndex UTC 15min.
    """
    idx = pd.date_range("2022-01-01", periods=n_15min, freq="15min", tz="UTC")
    # Convert to Zurich local time for hour/season extraction
    idx_local = idx.tz_convert("Europe/Zurich")
    hours = idx_local.hour + idx_local.minute / 60.0
    doy = idx_local.dayofyear.values
    dow = idx_local.dayofweek.values  # 0=Mon, 6=Sun

    # Base price + annual seasonal cycle
    base = 80.0 + 8.0 * np.sin(2 * np.pi * doy / 365.0)

    # Summer signal (doy 152-244 approx = June-Aug)
    is_summer = (doy >= 152) & (doy <= 244)
    is_weekend = dow >= 5  # Sat/Sun

    # Duck curve components (analytically controlled)

    # Solar depression h10-15: strongest in summer
    solar_depression = np.zeros(n_15min)
    h10_15 = (hours >= 10) & (hours < 15)
    solar_depression[h10_15 & is_summer] = -18.0     # summer solar: -18 EUR/MWh
    solar_depression[h10_15 & ~is_summer] = -2.0     # winter h10-15: mild

    # Weekend solar deeper (WE solar racheté, profile deals GRD)
    solar_depression[h10_15 & is_summer & is_weekend] = -25.0  # deep WE bowl

    # Evening peak h17-20 (universal, higher demand)
    evening_peak = np.zeros(n_15min)
    h17_20 = (hours >= 17) & (hours < 20)
    evening_peak[h17_20] = 22.0           # universal evening peak

    # Night baseline h22-6 (low demand)
    night_discount = np.zeros(n_15min)
    h_night = (hours >= 22) | (hours < 6)
    night_discount[h_night] = -8.0

    price = base + solar_depression + evening_peak + night_discount
    price += rng.standard_normal(n_15min) * 2.0
    price = np.clip(price, -50.0, 200.0)
    return pd.DataFrame({"price_eur_mwh": price}, index=idx)


def _build_hydro_df(rng: np.random.Generator) -> pd.DataFrame:
    """Return a synthetic Swiss-like weekly hydro reservoir DataFrame.

    Swiss reservoir fill profile: low in January (~30%), peak in August (~80%).
    Covers 104 weeks (2 years) starting 2022-01-01 UTC.

    RESEARCH Pitfall 1: hydro must cover ≥52 weeks so ``_climatological_fill``
    covers all WOY via nearest-neighbor interpolation. 104 weeks is the safe
    upper bound used here.

    Args:
        rng: NumPy random generator (seed-controlled by caller).

    Returns:
        DataFrame with column ``fill_pct`` (0–100 scale), weekly DatetimeIndex UTC.
    """
    weeks = pd.date_range("2022-01-01", periods=104, freq="W", tz="UTC")
    doy = weeks.dayofyear.values
    # Sinusoidal: min en janvier (~30%), max en août (~80%)
    seasonal = 0.55 + 0.25 * np.sin(2 * np.pi * (doy - 30) / 365.0)
    noise = rng.standard_normal(len(weeks)) * 0.05
    fill = np.clip(seasonal + noise, 0.05, 0.98)
    return pd.DataFrame({"fill_pct": fill * 100}, index=weeks)


def build_bowl_fixture(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the deterministic bowl fixture (reusable entry point for tests).

    Generates both the EPEX 15-min DataFrame and the weekly hydro DataFrame
    from a fixed seed. Both DataFrames are fully deterministic across Python
    runs (seed=42 convention from 5bis-A, D-A4-1).

    Args:
        seed: Random seed. Default 42 matches the committed ``bowl_seed42.parquet``.

    Returns:
        Tuple ``(epex_df, hydro_df)`` where:
          - ``epex_df``: long-format 15-min EPEX with analytically-controlled duck curve,
            ~3 months (92 days × 96 slots = 8832 rows).
          - ``hydro_df``: weekly Swiss-like reservoir fill, 104 weeks (2 years).

    Decision reference: D-A4-1 (CONTEXT.md §Test design Area 4).
    """
    rng = np.random.default_rng(seed)
    # 3 months × 96 slots/day = 8832 timestamps
    n_15min = 3 * 31 * 96  # 8928 (conservative 31-day month × 3)
    epex_df = _build_bowl_epex(n_15min, rng)
    hydro_df = _build_hydro_df(rng)
    return epex_df, hydro_df


def main() -> None:
    """Generate bowl_seed42.parquet and print acceptance info."""
    epex_df, _hydro_df = build_bowl_fixture(seed=42)

    # Write ONLY the EPEX DataFrame (hydro is deterministic, rebuilt in-process by tests)
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    epex_df.to_parquet(str(_OUT_PATH), index=True)

    # Confirmation line (used by acceptance criterion grep)
    hash_price = hash(epex_df["price_eur_mwh"].values.tobytes())
    import os
    size_kb = os.path.getsize(_OUT_PATH) / 1024
    print(
        f"Wrote bowl fixture: rows={len(epex_df)}"
        f" cols={list(epex_df.columns)}"
        f" price_range=[{epex_df.price_eur_mwh.min():.1f}, {epex_df.price_eur_mwh.max():.1f}]"
        f" size={size_kb:.1f}KB"
        f" hash_price={hash_price}"
    )


if __name__ == "__main__":
    main()
