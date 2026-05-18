"""
_generate_baseline.py
---------------------
Deterministic baseline generator for Phase 5bis-A no-op refactor regression testing.

PURPOSE
-------
Produces ``tests/fixtures/baseline_pfc_seed42.parquet`` — a frozen reference
for ``test_baseline_regression`` (added in plan 05B-05).  The regression contract
is "numerically identical":
    pandas.testing.assert_frame_equal(
        build(flag=OFF), baseline,
        check_exact=False, atol=1e-12, rtol=0
    )
plus identical columns, dtypes, index, and sort order.

This is NOT a parquet byte-equivalence contract (parquet byte output is not
guaranteed across pandas/pyarrow/python versions — only the numerical content
and schema are stable).

INTENDED GIT SHA
----------------
This script is intended to be run against:
    HEAD == 28dfd65 for pfc_shaping/* per
    ``git diff --stat 28dfd65..HEAD -- pfc_shaping/ tests/`` = empty

Verify before regenerating:
    git diff --stat 28dfd65..HEAD -- pfc_shaping/ tests/

REGENERATION POLICY
-------------------
CLI: ``python tests/fixtures/_generate_baseline.py``

WARNING: This file is INPUT-ONLY for the regression test in plan 05B-05.
Modifying behavior of pfc_shaping requires explicitly regenerating AND
committing both this script and the resulting .parquet in a single PR, with a
justification block in the PR description explaining why the baseline changed.
Do NOT regenerate lightly.

WORKAROUND NOTE
---------------
PFCAssembler.__init__ requires a fitted ShapeIntraday instance (not None).
To keep this script synthetic-only, a minimal ShapeIntraday is fitted on the
same synthetic EPEX DataFrame as ShapeHourly.  Both fits use seed=42 and the
same 3-year synthetic index, so the result is fully deterministic.
"""

from __future__ import annotations

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
# Output path (relative to repo root)
# ---------------------------------------------------------------------------
_OUT_PATH = Path(__file__).resolve().parent / "baseline_pfc_seed42.parquet"


def _build_synthetic_epex(n_15min: int, rng: np.random.Generator) -> pd.DataFrame:
    """Return a synthetic 15-min EPEX DataFrame (price_eur_mwh, UTC, ~3 years)."""
    idx = pd.date_range(
        start="2022-01-01",
        periods=n_15min,
        freq="15min",
        tz="UTC",
    )
    t = np.arange(n_15min)
    hours = (t / 4) % 24  # 15-min slot → fractional hour-of-day
    doy = idx.dayofyear.values  # day-of-year (1..365)

    price = (
        30.0
        + 10.0 * np.sin(2 * np.pi * hours / 24.0)
        + 5.0 * np.sin(2 * np.pi * doy / 365.0)
        + rng.standard_normal(n_15min) * 2.0
    )
    price = np.clip(price, -50.0, 200.0)
    return pd.DataFrame({"price_eur_mwh": price}, index=idx)


def main() -> None:
    # ── Deterministic seeds ────────────────────────────────────────────
    random.seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)

    # ── Synthetic EPEX: 3 calendar years at 15-min resolution ────────
    # 2022-01-01 .. 2024-12-31 UTC  (3 * 365 * 96 ≈ 105 120 slots)
    n_15min = 3 * 365 * 96  # 105 120 (ignores leap days for simplicity)
    epex_df = _build_synthetic_epex(n_15min, rng)

    # ── Calendar enrichment ───────────────────────────────────────────
    calendar_df = enrich_15min_index(epex_df.index, country="CH")

    # ── Fit ShapeHourly (no hydro_df → _climatological_fill = None) ──
    sh = ShapeHourly().fit(epex_df, calendar_df)

    # ── Fit ShapeIntraday on the same synthetic EPEX ─────────────────
    # PFCAssembler requires a fitted ShapeIntraday; fitting on synthetic
    # data with entso_df=None calibrates only the deterministic Layer 1
    # (base_factors_), which is sufficient for a deterministic baseline.
    si = ShapeIntraday().fit(epex_df, entso_df=None, calendar_df=calendar_df)

    # ── Build PFCAssembler (all optional components = None) ──────────
    assembler = PFCAssembler(
        shape_hourly=sh,
        shape_intraday=si,
        uncertainty=None,
        water_value=None,
        cascader=None,
        calibrator=None,
    )

    # ── Run build: Cal'27, 1-month horizon, seed=42 reference date ───
    df_pfc = assembler.build(
        base_prices={"2027": 80.0},
        start_date="2027-01-01",
        horizon_days=31,
        reference_date=pd.Timestamp("2026-05-18", tz="UTC"),
        country="CH",
    )

    # ── Write parquet ─────────────────────────────────────────────────
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_pfc.to_parquet(str(_OUT_PATH), index=True)

    # ── Confirmation line (used by acceptance criterion grep) ─────────
    hash_price_shape = hash(df_pfc["price_shape"].values.tobytes())
    print(
        f"Wrote baseline: rows={len(df_pfc)}"
        f" cols={list(df_pfc.columns)}"
        f" hash_price_shape={hash_price_shape}"
    )


if __name__ == "__main__":
    main()
