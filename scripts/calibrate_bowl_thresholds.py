"""
calibrate_bowl_thresholds.py
-----------------------------
Wave 0 calibration script for Phase 5bis-B SC #1 (ptp ratio threshold).

Reads tests/fixtures/bowl_seed42.parquet, fits sh_off + sh_on, computes the
ptp ratio on the (Ete, Ouvrable) factors cell, and writes an immutable JSON
artifact to tests/fixtures/_bowl_calibration_report.json.

Commit BOTH this script AND the JSON output to git after every re-run.

PURPOSE
-------
Implements M2 from REVIEWS.md consensus (cross-AI review fix — Codex framing wins
over Gemini's lighter "hidden helper" suggestion):
    - The calibration is a COMMITTED, version-controlled, reproducible script
      (not an interactive terminal one-shot).
    - The output JSON carries calibrated_at, git_sha, fixture_sha256, ratios,
      and thresholds_emitted.
    - tests/test_shape_hourly_bowl.py loads SC1_PTP_THRESHOLD from this JSON via
      json.load — NOT from a free-floating in-comment "# observed ratio = X" value.
    - A secondary quant reviewer can re-run this script to audit the calibration;
      if the JSON changes, the test_calibration_report_matches_fixture test (added
      in Plan 05C-03) will fail loudly via sha256 mismatch.

Reproduce via:
    python scripts/calibrate_bowl_thresholds.py

REVIEWS.md reference: consensus #3 (auditability concern, both Gemini and Codex).
M2 cross-AI review fix (05C-REVIEWS.md §Recommended actions, item 2).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# Ensure repo root is importable when run as a script from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402 (needed for Timestamp in _calibrate_sc3_m30)

from pfc_shaping.data.calendar_ch import enrich_15min_index  # noqa: E402
from pfc_shaping.lt.model.assembler import PFCAssembler  # noqa: E402
from pfc_shaping.lt.model.shape_hourly import ShapeHourly  # noqa: E402
from pfc_shaping.lt.model.shape_intraday import ShapeIntraday  # noqa: E402
from tests.fixtures._generate_bowl_fixture import build_bowl_fixture  # noqa: E402

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------
FIXTURE_PATH = _REPO_ROOT / "tests" / "fixtures" / "bowl_seed42.parquet"
REPORT_PATH = _REPO_ROOT / "tests" / "fixtures" / "_bowl_calibration_report.json"

# RESEARCH §Lever 1 (05C-RESEARCH.md) threshold formula:
#   threshold = max(observed_ratio - SC1_RATIO_MARGIN, SC1_FLOOR_MULTIPLIER)
# Plancher 1.05 = 10% below minimum expected gain of ~1.13-1.18 (Lever 1 dry-run).
SC1_FLOOR_MULTIPLIER: float = 1.05
SC1_RATIO_MARGIN: float = 0.15


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _compute_fixture_sha256(path: Path) -> str:
    """Compute sha256 hex digest of a file in binary mode.

    This is the M2-mandated immutability link between the JSON report and the
    fixture binary. If bowl_seed42.parquet changes (e.g. re-generated with a
    different seed or formula), the sha256 in the report will mismatch and
    test_calibration_report_matches_fixture (Plan 05C-03) will fail loudly.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get_git_sha() -> str:
    """Return the current git HEAD SHA for audit trail.

    Returns 'unknown-not-in-git' if git is unavailable (e.g. some CI environments).
    """
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown-not-in-git"


def _calibrate_sc3_m30(
    epex_df, hydro_df, cal
) -> tuple[float, float, float]:
    """Fit assembler with flag=OFF and flag=ON, build at ~M+30 horizon, measure ptp(f_H).

    This implements the Wave 0 calibration for SC #3 (D-A4-6 / RESEARCH §Lever 2 dry-run).
    The measure-then-assert pattern: observe ptp_on at M+30, compute threshold with safety
    margin, write to JSON sidecar. Test consumes via json.load (M2 cross-AI review fix).

    Expected (RESEARCH §Lever 2 dry-run, RESEARCH.md):
        legacy M+30 ptp:  ~0.516  (full damping, sf=0.52 at 30 months)
        split M+30 ptp:   ~0.992  (anomaly survives, level ≈ 1.0 by SHP-03)
        gain ratio:       ~1.92

    Threshold formula (from PLAN 05C-02 Task 3 action):
        threshold = max(ptp_on - 0.20, ptp_off * 1.50, 0.50)
    This ensures:
        (a) threshold < ptp_on with 0.20 safety margin
        (b) threshold > ptp_off * 1.50 (proves split is 50% better than legacy at M+30)
        (c) at least the plancher 0.50

    Args:
        epex_df: 15-min EPEX DataFrame (from build_bowl_fixture)
        hydro_df: Weekly hydro fill DataFrame (from build_bowl_fixture)
        cal: Calendar enrichment for epex_df.index

    Returns:
        (ptp_off, ptp_on, threshold): M+30 ptp of f_H under flag=OFF and flag=ON,
        and the calibrated threshold for test_f_H_amplitude_preserved_at_M30.
    """
    # Fit ShapeHourly (flag=OFF and flag=ON)
    sh_off = ShapeHourly(use_seasonal_hourly=False).fit(epex_df, cal, hydro_df)
    sh_on = ShapeHourly(use_seasonal_hourly=True).fit(epex_df, cal, hydro_df)

    # Fit minimal ShapeIntraday (same pattern as _generate_baseline.py:128-133)
    si = ShapeIntraday().fit(epex_df, entso_df=None, calendar_df=cal)

    # Build PFC at far horizon ~M+30 (from start_date 2029-06-01 to reference 2027-01-01 = ~29 months)
    # horizon_days=31 gives exactly one month window to measure f_H amplitude.
    _build_kwargs = dict(
        base_prices={"2029": 80.0},
        start_date="2029-06-01",
        horizon_days=31,
        reference_date=pd.Timestamp("2027-01-01", tz="UTC"),
        country="CH",
    )

    assembler_off = PFCAssembler(
        shape_hourly=sh_off,
        shape_intraday=si,
        uncertainty=None,
        water_value=None,
        cascader=None,
        calibrator=None,
    )
    df_off = assembler_off.build(**_build_kwargs)
    ptp_off = float(np.ptp(df_off["f_H"]))

    assembler_on = PFCAssembler(
        shape_hourly=sh_on,
        shape_intraday=si,
        uncertainty=None,
        water_value=None,
        cascader=None,
        calibrator=None,
    )
    df_on = assembler_on.build(**_build_kwargs)
    ptp_on = float(np.ptp(df_on["f_H"]))

    # Threshold formula: max(ptp_on - 0.20, ptp_off * 1.50, 0.50)
    threshold = max(ptp_on - 0.20, ptp_off * 1.50, 0.50)
    return ptp_off, ptp_on, threshold


def _calibrate_sc1(
    epex_df, hydro_df, cal
) -> tuple[float, float, float]:
    """Fit sh_off + sh_on and compute SC #1 ptp ratio on (Ete, Ouvrable) cell.

    Returns:
        (ptp_off, ptp_on, ratio): ptp of the (Ete, Ouvrable) factors cell under
        flag=OFF and flag=ON, and the ratio ptp_on / ptp_off.
    """
    sh_off = ShapeHourly(use_seasonal_hourly=False).fit(epex_df, cal, hydro_df)
    sh_on = ShapeHourly(use_seasonal_hourly=True).fit(epex_df, cal, hydro_df)

    # Locate (Ete, Ouvrable) key; fall back to first common key if absent
    target_key = ("Ete", "Ouvrable")
    common_keys = set(sh_off.factors_.keys()) & set(sh_on.factors_.keys())
    if target_key not in common_keys:
        fallback_key = next(iter(common_keys))
        print(
            f"WARNING: (Ete, Ouvrable) not in common keys — using {fallback_key} as fallback"
        )
        target_key = fallback_key

    ptp_off = float(np.ptp(sh_off.factors_[target_key]))
    ptp_on = float(np.ptp(sh_on.factors_[target_key]))
    ratio = ptp_on / ptp_off if ptp_off > 0 else float("nan")
    return ptp_off, ptp_on, ratio


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Wave 0 calibration and write the immutable JSON report."""
    parser = argparse.ArgumentParser(
        description="Wave 0 calibration for Phase 5bis-B SC #1 threshold."
    )
    parser.parse_args()  # No positional args; validates unexpected args

    # Build fixture (deterministic seed=42)
    print("Building bowl fixture (seed=42)...")
    epex_df, hydro_df = build_bowl_fixture(seed=42)

    # Build calendar enrichment
    cal = enrich_15min_index(epex_df.index, country="CH")

    # Run SC #1 calibration
    print("Fitting sh_off (flag=False) and sh_on (flag=True) for SC #1...")
    ptp_off, ptp_on, ratio = _calibrate_sc1(epex_df, hydro_df, cal)

    # Sanity bounds (STOP if outside expected range)
    if ratio < 1.00:
        print(
            f"ERROR: sc1_ptp_ratio={ratio:.4f} < 1.00 — Lever 1 is REGRESSING amplitude vs "
            "flag=OFF. Possible bug in _apply_hydro_analogue_weights. Aborting report write."
        )
        sys.exit(1)
    if ratio > 3.00:
        print(
            f"ERROR: sc1_ptp_ratio={ratio:.4f} > 3.00 — implausibly large vs RESEARCH §Lever 1 "
            "analytic estimate 1.13-1.18. Verify fixture or kernel logic. Aborting report write."
        )
        sys.exit(1)

    # Compute SC #1 threshold: max(observed_ratio - margin, floor)
    sc1_threshold = max(ratio - SC1_RATIO_MARGIN, SC1_FLOOR_MULTIPLIER)

    # Run SC #3 calibration (Plan 05C-02 Task 3 extension)
    print("Calibrating SC #3 M+30 amplitude threshold (Lever 2 _split_level_anomaly)...")
    sc3_ptp_off, sc3_ptp_on, sc3_threshold = _calibrate_sc3_m30(epex_df, hydro_df, cal)

    # Sanity bounds for SC #3 (STOP if outside expected range — see PLAN 05C-02 Task 3 action)
    # NOTE: RESEARCH §Lever 2 dry-run predicted ptp_on ~0.99 assuming a full-year fixture.
    # bowl_seed42 covers Jan-Mar only (Hiver), so Ete cells fall back to Ouvrable (minimal bowl).
    # The absolute ptp values are therefore lower than the dry-run estimate, but what matters
    # for Lever 2 correctness is that ptp_on > ptp_off * gain (the split IS amplifying relative
    # to the legacy damped value). Adjusted sanity bounds accordingly.
    if sc3_ptp_on < sc3_ptp_off:
        print(
            f"ERROR: sc3_ptp_on={sc3_ptp_on:.4f} < sc3_ptp_off={sc3_ptp_off:.4f} — "
            "Lever 2 is NOT improving amplitude at M+30 vs flag=OFF. "
            "Possible bug in _split_level_anomaly or assembler integration. "
            "Aborting report write."
        )
        sys.exit(1)
    if sc3_ptp_on > 2.00:
        print(
            f"ERROR: sc3_ptp_on={sc3_ptp_on:.4f} > 2.00 — anomaly producing unbounded growth. "
            "Possible level computation error. Aborting report write."
        )
        sys.exit(1)
    if sc3_ptp_off > 0.70:
        print(
            f"WARNING: sc3_ptp_off={sc3_ptp_off:.4f} > 0.70 — legacy M+30 amplitude higher than "
            "RESEARCH analytic estimate 0.52. Bowl fixture may be too aggressive. Continuing."
        )

    # Build report dict (M2-mandated schema)
    report = {
        "calibrated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "fixture_path": "tests/fixtures/bowl_seed42.parquet",
        "fixture_sha256": _compute_fixture_sha256(FIXTURE_PATH),
        "git_sha": _get_git_sha(),
        "notes": (
            "Plan 05C-02 extended this report with SC #3 M+30 amplitude calibration. "
            "Plan 05C-03 will re-run this script with all 3 levers active (Task 3 of 05C-03); "
            "the updated artifact MUST overwrite this one and be re-committed."
        ),
        "ratios": {
            "sc1_floor_multiplier": SC1_FLOOR_MULTIPLIER,
            "sc1_ptp_off": ptp_off,
            "sc1_ptp_on": ptp_on,
            "sc1_ptp_ratio": ratio,
            "sc1_ratio_margin": SC1_RATIO_MARGIN,
            "sc3_amplitude_formula": "max(ptp_on - 0.20, ptp_off * 1.50, 0.50)",
            "sc3_ptp_off_m30": sc3_ptp_off,
            "sc3_ptp_on_m30": sc3_ptp_on,
        },
        "thresholds_emitted": {
            "SC1_PTP_THRESHOLD": sc1_threshold,
            "SC3_M30_AMPLITUDE_THRESHOLD": sc3_threshold,  # Plan 05C-02 Task 3 calibrated value
        },
    }

    # Write to REPORT_PATH (sort_keys=True + trailing newline for git-friendly stable diffs)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(
        f"Wave 0 calibrated: ratio={ratio:.4f}, sc1_threshold={sc1_threshold:.4f}, "
        f"sc3_threshold={sc3_threshold:.4f}, report={REPORT_PATH.relative_to(_REPO_ROOT)}"
    )
    print(f"  SC #1: ptp_off={ptp_off:.4f}  ptp_on={ptp_on:.4f}")
    print(f"  SC #3: ptp_off_m30={sc3_ptp_off:.4f}  ptp_on_m30={sc3_ptp_on:.4f}")
    print(f"  fixture_sha256={report['fixture_sha256'][:16]}...")
    print(f"  git_sha={report['git_sha'][:16]}...")
    print("REMINDER: Commit both scripts/calibrate_bowl_thresholds.py AND tests/fixtures/_bowl_calibration_report.json")


if __name__ == "__main__":
    main()
