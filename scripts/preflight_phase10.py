#!/usr/bin/env python3
"""Plan 10-04 Task 0 — Preflight Go/No-Go [C6 REVIEWS].

Exécute 5 checks avant le 96-build ablation grid de Plan 10-04 :

  1. Data freshness (epex_hourly.parquet + forwards_history_phase10.parquet
     mtime <= 14 days OR --allow-stale-data).
  2. Forwards path : forwards_source unique value ∈ {real_eex_xlsx,
     fallback_diagnostic}. fallback => DIAGNOSTIC ONLY unless
     --accept-fallback-diagnostic explicit.
  3. Disk space ≥ 5 GB at output_dir mount.
  4. Runtime benchmark : 1 micro-build (Config 4, vintage 2024-06-28,
     no uncertainty) ; extrapolation to 96 × 1.4 (Pillar 3 uncertainty
     overhead) ≤ 2.5h hard cap (soft target 2h).
  5. Python environment : verify python3 -c 'import statsmodels, matplotlib'
     works AND warns if shell python3 is the Xcode 3.9 binary that doesn't
     have these deps (silent shadowing issue).

Écrit le verdict file `10-PREFLIGHT.md` avec le format documenté
plan section et exit 0 (go) ou 1 (no-go).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_FRESHNESS_THRESHOLD_DAYS = 14
DISK_SPACE_THRESHOLD_GB = 5.0
DISK_SPACE_WARN_GB = 10.0
RUNTIME_HARD_CAP_S = 9000.0  # 2.5 h
RUNTIME_SOFT_TARGET_S = 7200.0  # 2 h
UNCERTAINTY_OVERHEAD_FACTOR = 1.4  # Pillar 3 Uncertainty(n_boot=500) Config 4
TOTAL_BUILDS = 96

EPEX_PARQUET_REL = "data/epex_hourly.parquet"
FORWARDS_PARQUET_REL = "data/forwards_history_phase10.parquet"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mtime_age_days(path: Path) -> tuple[str, float]:
    mtime = path.stat().st_mtime
    age_days = (time.time() - mtime) / 86400.0
    mtime_iso = datetime.fromtimestamp(mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return mtime_iso, age_days


def check_python_env() -> dict:
    """Check Python env can import statsmodels + matplotlib.

    Critical : detects the Xcode 3.9 shadowing where shell `python3` resolves
    to /Applications/Xcode.app/.../Versions/3.9/bin/python3 which lacks
    these deps even though they're installed in pyenv 3.12.12.
    """
    executable = sys.executable
    is_xcode_shadowed = "Xcode.app" in executable or "Versions/3.9" in executable
    try:
        import statsmodels  # noqa: F401
        import matplotlib  # noqa: F401
        deps_ok = True
        err = None
    except ImportError as e:
        deps_ok = False
        err = str(e)
    return {
        "executable": executable,
        "is_xcode_shadowed": is_xcode_shadowed,
        "deps_ok": deps_ok,
        "error": err,
    }


def check_data_freshness(repo_root: Path, allow_stale: bool) -> dict:
    """Check 1 — EPEX + forwards parquet mtime ≤ 14 days (unless override)."""
    epex_path = repo_root / EPEX_PARQUET_REL
    forwards_path = repo_root / FORWARDS_PARQUET_REL
    result = {
        "epex_present": epex_path.exists(),
        "forwards_present": forwards_path.exists(),
        "epex_mtime": None,
        "epex_age_days": None,
        "forwards_mtime": None,
        "forwards_age_days": None,
        "verdict": "PASS",
        "reason": None,
    }
    if not result["epex_present"]:
        result["verdict"] = "FAIL"
        result["reason"] = f"missing {EPEX_PARQUET_REL}"
        return result
    if not result["forwards_present"]:
        result["verdict"] = "FAIL"
        result["reason"] = f"missing {FORWARDS_PARQUET_REL}"
        return result
    epex_mtime, epex_age = _mtime_age_days(epex_path)
    fw_mtime, fw_age = _mtime_age_days(forwards_path)
    result["epex_mtime"] = epex_mtime
    result["epex_age_days"] = round(epex_age, 2)
    result["forwards_mtime"] = fw_mtime
    result["forwards_age_days"] = round(fw_age, 2)
    stale = max(epex_age, fw_age) > DATA_FRESHNESS_THRESHOLD_DAYS
    if stale and not allow_stale:
        result["verdict"] = "FAIL"
        result["reason"] = (
            f"data stale (max age {max(epex_age, fw_age):.1f}d > "
            f"{DATA_FRESHNESS_THRESHOLD_DAYS}d) ; re-run Plan 10-01 Task 3 "
            f"OR pass --allow-stale-data"
        )
    elif stale and allow_stale:
        result["verdict"] = "PASS (stale OK via --allow-stale-data)"
    return result


def check_forwards_path(repo_root: Path, accept_fallback: bool) -> dict:
    """Check 2 — forwards_source column unique value (real_eex_xlsx OR fallback_diagnostic)."""
    import pandas as pd
    forwards_path = repo_root / FORWARDS_PARQUET_REL
    result = {
        "forwards_source_unique": None,
        "gate_eligible": False,
        "verdict": "PASS",
        "reason": None,
        "decision": None,
    }
    if not forwards_path.exists():
        result["verdict"] = "FAIL"
        result["reason"] = f"missing {FORWARDS_PARQUET_REL} (cannot read forwards_source column)"
        return result
    df = pd.read_parquet(forwards_path)
    if "forwards_source" not in df.columns:
        result["verdict"] = "FAIL"
        result["reason"] = (
            "Plan 10-01 C3 contract violated : forwards_source column absent. "
            "Re-run Plan 10-01 Task 3 to refresh the parquet with the marker."
        )
        return result
    unique_vals = df["forwards_source"].unique()
    if len(unique_vals) != 1:
        result["verdict"] = "FAIL"
        result["reason"] = (
            f"forwards_source has {len(unique_vals)} unique values "
            f"({list(unique_vals)}) ; expected exactly 1"
        )
        return result
    val = str(unique_vals[0])
    result["forwards_source_unique"] = val
    if val == "real_eex_xlsx":
        result["gate_eligible"] = True
        result["decision"] = "continue (gate-eligible)"
    elif val == "fallback_diagnostic":
        result["gate_eligible"] = False
        if accept_fallback:
            result["verdict"] = "PASS (DIAGNOSTIC ONLY via --accept-fallback-diagnostic)"
            result["decision"] = (
                "continue with WARNING: Plan 10-04 final run will be DIAGNOSTIC "
                "ONLY ; SC#1 cannot be satisfied — D-FLIP-1 mechanically BLOCKED"
            )
        else:
            result["verdict"] = "FAIL"
            result["reason"] = (
                "forwards_source=fallback_diagnostic ; SC#1 NOT gate-eligible. "
                "Run from FMV poste with H:\\ access for real_eex_xlsx, "
                "OR pass --accept-fallback-diagnostic to accept diagnostic-only mode."
            )
    else:
        result["verdict"] = "FAIL"
        result["reason"] = (
            f"forwards_source value {val!r} unexpected ; expected one of "
            "{real_eex_xlsx, fallback_diagnostic}"
        )
    return result


def check_disk_space(output_dir: Path) -> dict:
    """Check 3 — disk space ≥ 5 GB at output_dir mount."""
    mount = output_dir.parent if output_dir.exists() else output_dir.parent
    if not mount.exists():
        mount = Path.cwd()
    usage = shutil.disk_usage(mount)
    free_gb = usage.free / 1e9
    result = {
        "free_gb": round(free_gb, 2),
        "threshold_gb": DISK_SPACE_THRESHOLD_GB,
        "warn_gb": DISK_SPACE_WARN_GB,
        "verdict": "PASS",
        "reason": None,
    }
    if free_gb < DISK_SPACE_THRESHOLD_GB:
        result["verdict"] = "FAIL"
        result["reason"] = (
            f"free space {free_gb:.1f} GB < {DISK_SPACE_THRESHOLD_GB} GB threshold"
        )
    elif free_gb < DISK_SPACE_WARN_GB:
        result["verdict"] = "PASS (WARNING)"
        result["reason"] = (
            f"free space {free_gb:.1f} GB is below {DISK_SPACE_WARN_GB} GB warn level"
        )
    return result


def check_runtime_benchmark(repo_root: Path) -> dict:
    """Check 4 — 1 micro-build extrapolated to full 96 × 1.4 must ≤ 2.5h."""
    import pandas as pd
    sys.path.insert(0, str(repo_root))
    try:
        from pfc_shaping.validation.scorecard import (
            ABLATION_GRID,
            build_one,
            derive_forwards_from_epex_hist,
            last_business_day_of_month,
        )
    except Exception as e:
        return {
            "verdict": "FAIL",
            "reason": f"could not import pfc_shaping.validation.scorecard: {e}",
            "t_micro_s": None,
            "t_full_estimate_s": None,
            "t_full_estimate_h": None,
        }

    epex_path = repo_root / EPEX_PARQUET_REL
    if not epex_path.exists():
        return {
            "verdict": "FAIL",
            "reason": f"missing {EPEX_PARQUET_REL} ; cannot run benchmark",
            "t_micro_s": None,
            "t_full_estimate_s": None,
            "t_full_estimate_h": None,
        }

    epex_df = pd.read_parquet(epex_path)
    if "price_eur_mwh" not in epex_df.columns:
        return {
            "verdict": "FAIL",
            "reason": f"{EPEX_PARQUET_REL} missing 'price_eur_mwh' column",
            "t_micro_s": None,
            "t_full_estimate_s": None,
            "t_full_estimate_h": None,
        }

    epex_series = epex_df["price_eur_mwh"]
    epex_hist_df = epex_series.to_frame(name="price_eur_mwh")

    config4 = next(c for c in ABLATION_GRID if c.name == "bowl_on_floors_off")
    vintage = last_business_day_of_month(2024, 6).tz_convert("UTC")

    # Forwards as-of vintage : derive from EPEX hist (fallback)
    fwds = derive_forwards_from_epex_hist(epex_series, vintage)

    t0 = time.perf_counter()
    pfc = build_one(
        config=config4,
        vintage=vintage,
        epex_hist=epex_hist_df,
        forwards_asof=fwds,
        with_uncertainty=False,
    )
    t_micro = time.perf_counter() - t0
    _ = pfc  # used (no warning)

    t_full = t_micro * TOTAL_BUILDS * UNCERTAINTY_OVERHEAD_FACTOR
    t_full_h = t_full / 3600.0
    result = {
        "t_micro_s": round(t_micro, 2),
        "t_full_estimate_s": round(t_full, 1),
        "t_full_estimate_h": round(t_full_h, 2),
        "soft_target_s": RUNTIME_SOFT_TARGET_S,
        "hard_cap_s": RUNTIME_HARD_CAP_S,
        "verdict": "PASS",
        "reason": None,
    }
    if t_full > RUNTIME_HARD_CAP_S:
        result["verdict"] = "FAIL"
        result["reason"] = (
            f"extrapolated runtime {t_full_h:.2f}h > "
            f"{RUNTIME_HARD_CAP_S/3600:.1f}h hard cap"
        )
    elif t_full > RUNTIME_SOFT_TARGET_S:
        result["verdict"] = "PASS (WARNING)"
        result["reason"] = (
            f"extrapolated runtime {t_full_h:.2f}h > "
            f"{RUNTIME_SOFT_TARGET_S/3600:.1f}h soft target (still within hard cap)"
        )
    return result


def write_verdict_file(
    output_dir: Path,
    verdict: str,
    env: dict,
    c1: dict,
    c2: dict,
    c3: dict,
    c4: dict,
    next_steps_go: str,
    next_steps_no_go: str,
) -> Path:
    """Write 10-PREFLIGHT.md per Plan 10-04 Task 0 format spec."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "10-PREFLIGHT.md"

    def fmt_pass(v: str) -> str:
        if v.startswith("PASS"):
            return "PASS"
        return "FAIL"

    lines = []
    lines.append("# Plan 10-04 — Preflight Go/No-Go [C6 REVIEWS]")
    lines.append("")
    lines.append(f"**Generated** : {_now_iso()}")
    lines.append("**Script** : scripts/preflight_phase10.py")
    lines.append(f"**Verdict** : **{verdict}**")
    lines.append("")
    lines.append("## Python environment")
    lines.append(f"- sys.executable : `{env['executable']}`")
    lines.append(f"- Xcode 3.9 shadowing detected : {env['is_xcode_shadowed']}")
    lines.append(f"- statsmodels + matplotlib importable : {env['deps_ok']}")
    if env["error"]:
        lines.append(f"- import error : `{env['error']}`")
    if env["is_xcode_shadowed"] or not env["deps_ok"]:
        lines.append(
            "- **WARNING** : shell `python3` resolves to a Python that lacks "
            "Phase 10 deps. Use `/Users/julienbattaglia/.pyenv/versions/3.12.12/bin/python3` "
            "for the actual ablation run AND pytest."
        )
    lines.append("")
    lines.append("## Check 1 — Data freshness")
    lines.append(f"- epex_hourly.parquet present : {c1['epex_present']}")
    if c1.get("epex_mtime"):
        lines.append(
            f"- epex_hourly.parquet : mtime {c1['epex_mtime']}, "
            f"age {c1['epex_age_days']} days, threshold ≤ {DATA_FRESHNESS_THRESHOLD_DAYS} days"
        )
    lines.append(f"- forwards_history_phase10.parquet present : {c1['forwards_present']}")
    if c1.get("forwards_mtime"):
        lines.append(
            f"- forwards_history_phase10.parquet : mtime {c1['forwards_mtime']}, "
            f"age {c1['forwards_age_days']} days, threshold ≤ {DATA_FRESHNESS_THRESHOLD_DAYS} days"
        )
    lines.append(f"- **Status** : {fmt_pass(c1['verdict'])} ({c1['verdict']})")
    if c1.get("reason"):
        lines.append(f"- Reason : {c1['reason']}")
    lines.append("")
    lines.append("## Check 2 — Forwards path")
    lines.append(f"- forwards_source unique value : `{c2.get('forwards_source_unique')}`")
    gate_str = "Y" if c2.get("gate_eligible") else "N (DIAGNOSTIC ONLY)"
    lines.append(f"- Gate-eligible : {gate_str}")
    if c2.get("decision"):
        lines.append(f"- Decision : {c2['decision']}")
    lines.append(f"- **Status** : {fmt_pass(c2['verdict'])} ({c2['verdict']})")
    if c2.get("reason"):
        lines.append(f"- Reason : {c2['reason']}")
    lines.append("")
    lines.append("## Check 3 — Disk space")
    lines.append(f"- Free space at output_dir mount : {c3['free_gb']} GB")
    lines.append(f"- Threshold ≥ {DISK_SPACE_THRESHOLD_GB} GB → {fmt_pass(c3['verdict'])}")
    lines.append("- Budget output total : ~2.5 GB (96 PFC caches × 25 MB + KPIs + figures)")
    lines.append(f"- **Status** : {fmt_pass(c3['verdict'])} ({c3['verdict']})")
    if c3.get("reason"):
        lines.append(f"- Reason : {c3['reason']}")
    lines.append("")
    lines.append("## Check 4 — Runtime benchmark")
    if c4.get("t_micro_s") is not None:
        lines.append(
            f"- Micro-run (Config 4, vintage 2024-06-28, no uncertainty) : "
            f"{c4['t_micro_s']} sec"
        )
        lines.append(
            f"- Extrapolation full 96-build × {UNCERTAINTY_OVERHEAD_FACTOR} "
            f"(Pillar 3 overhead) : {c4['t_full_estimate_s']} sec = "
            f"{c4['t_full_estimate_h']:.2f} h"
        )
        lines.append(
            f"- Budget hard cap : {RUNTIME_HARD_CAP_S/3600:.1f}h ; "
            f"soft target : {RUNTIME_SOFT_TARGET_S/3600:.1f}h"
        )
    lines.append(f"- **Status** : {fmt_pass(c4['verdict'])} ({c4['verdict']})")
    if c4.get("reason"):
        lines.append(f"- Reason : {c4['reason']}")
    lines.append("")
    lines.append("## Final Verdict")
    lines.append(f"- **{verdict}**")
    if verdict == "go":
        lines.append(f"- Next step : {next_steps_go}")
    else:
        lines.append(f"- Next step : {next_steps_no_go}")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan 10-04 Task 0 — preflight go/no-go.")
    parser.add_argument(
        "--output-dir",
        default=".planning/phases/10-pfc-fmv-quality-scorecard",
        help="Output dir for 10-PREFLIGHT.md",
    )
    parser.add_argument(
        "--allow-stale-data",
        action="store_true",
        help="Accept data older than 14 days (override Check 1).",
    )
    parser.add_argument(
        "--accept-fallback-diagnostic",
        action="store_true",
        help="Accept forwards_source=fallback_diagnostic (DIAGNOSTIC ONLY ; "
        "Plan 10-04 D-FLIP-1 mechanically BLOCKED).",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repo root (default = cwd).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd().resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if Path(args.output_dir).is_absolute()
        else (repo_root / args.output_dir).resolve()
    )

    env = check_python_env()
    c1 = check_data_freshness(repo_root, args.allow_stale_data)
    c2 = check_forwards_path(repo_root, args.accept_fallback_diagnostic)
    c3 = check_disk_space(output_dir)
    c4 = check_runtime_benchmark(repo_root)

    fails = []
    for label, c in [("data_freshness", c1), ("forwards_path", c2),
                      ("disk_space", c3), ("runtime", c4)]:
        if c["verdict"].startswith("FAIL"):
            fails.append((label, c.get("reason")))
    if not env["deps_ok"]:
        fails.append(("python_env", env["error"]))

    verdict = "go" if not fails else "no-go"

    next_steps_go = "`/gsd:execute-plan 10-04 Task 1` (cost confirmation human-verify)"
    next_steps_no_go = (
        "addresser le(s) check(s) FAIL avant re-run preflight : "
        + " ; ".join(f"{label}: {reason}" for label, reason in fails)
    )

    out_path = write_verdict_file(
        output_dir, verdict, env, c1, c2, c3, c4,
        next_steps_go, next_steps_no_go,
    )

    print(f"Preflight verdict : {verdict}")
    print(f"Wrote {out_path}")
    if fails:
        for label, reason in fails:
            print(f"  FAIL [{label}] {reason}")
    return 0 if verdict == "go" else 1


if __name__ == "__main__":
    sys.exit(main())
