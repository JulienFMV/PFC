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

from pfc_shaping.data.calendar_ch import enrich_15min_index  # noqa: E402
from pfc_shaping.lt.model.shape_hourly import ShapeHourly  # noqa: E402
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
    print("Fitting sh_off (flag=False) and sh_on (flag=True)...")
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

    # Compute threshold: max(observed_ratio - margin, floor)
    sc1_threshold = max(ratio - SC1_RATIO_MARGIN, SC1_FLOOR_MULTIPLIER)

    # Build report dict (M2-mandated schema)
    report = {
        "calibrated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "fixture_path": "tests/fixtures/bowl_seed42.parquet",
        "fixture_sha256": _compute_fixture_sha256(FIXTURE_PATH),
        "git_sha": _get_git_sha(),
        "notes": (
            "Plan 05C-01 ships Lever 1 only — sc1_ptp_ratio is the Lever-1-only gain. "
            "Plan 05C-03 will re-run this script with all 3 levers active (Plan 05C-03 "
            "Task 3); the updated artifact MUST overwrite this one and be re-committed."
        ),
        "ratios": {
            "sc1_floor_multiplier": SC1_FLOOR_MULTIPLIER,
            "sc1_ptp_off": ptp_off,
            "sc1_ptp_on": ptp_on,
            "sc1_ptp_ratio": ratio,
            "sc1_ratio_margin": SC1_RATIO_MARGIN,
        },
        "thresholds_emitted": {
            "SC1_PTP_THRESHOLD": sc1_threshold,
            "SC3_M30_AMPLITUDE_THRESHOLD_PLACEHOLDER": 0.50,  # Plan 05C-02 Task 3 updates this
        },
    }

    # Write to REPORT_PATH (sort_keys=True + trailing newline for git-friendly stable diffs)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(
        f"Wave 0 calibrated: ratio={ratio:.4f}, threshold={sc1_threshold:.4f}, "
        f"report={REPORT_PATH.relative_to(_REPO_ROOT)}"
    )
    print(f"  ptp_off={ptp_off:.4f}  ptp_on={ptp_on:.4f}")
    print(f"  fixture_sha256={report['fixture_sha256'][:16]}...")
    print(f"  git_sha={report['git_sha'][:16]}...")
    print("REMINDER: Commit both scripts/calibrate_bowl_thresholds.py AND tests/fixtures/_bowl_calibration_report.json")


if __name__ == "__main__":
    main()
