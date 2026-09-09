#!/usr/bin/env python3
"""One-command real-run launcher for the Phase 10 PFC FMV Quality Scorecard.

Wraps the three manual steps of a real (parquet) scorecard run into a single
terminal entrypoint:

    1. Bootstrap ``data/epex_hourly.parquet`` (CH day-ahead, hourly native) from
       the local ignored ``pfc_shaping/data/epex_15min.parquet`` if it is
       available.
       The 15min cache is the hourly price ffill-upsampled, so a downsampling
       ``resample('1h').mean()`` recovers the native hourly series exactly
       (without the N x4 inflation that would bias the Plan 10-03 stat tests).
    2. Run the 96-build scorecard via ``run_scorecard_full`` (optionally on a
       cold cache with ``--fresh``).
    3. Print the Pillar 1 (Hildmann SC#1 gate) per-test table + the seasonal
       shape diagnostics, and render figures / markdown report.

Usage (PowerShell or bash)
--------------------------
    python scripts/run_phase10_real.py                 # full real run
    python scripts/run_phase10_real.py --fresh         # cold cache (forwards changed)
    python scripts/run_phase10_real.py --vintages-limit 1   # quick dry-run
    python scripts/run_phase10_real.py --epex-source mock   # no real data needed

Real (parquet) runs additionally require ``data/forwards_history_phase10.parquet``
(produced by ``scripts/import_fmv_forwards.py``). Both parquet inputs are
gitignored and never committed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

EPEX_15MIN_REL = "pfc_shaping/data/epex_15min.parquet"
EPEX_HOURLY_REL = "data/epex_hourly.parquet"
FORWARDS_REL = "data/forwards_history_phase10.parquet"


def bootstrap_epex_hourly(force: bool = False) -> bool:
    """Reconstruct data/epex_hourly.parquet from the local 15min CH cache.

    Returns True if the file was (re)written, False if it already existed and
    ``force`` was not set.
    """
    import pandas as pd

    out = _REPO_ROOT / EPEX_HOURLY_REL
    if out.exists() and not force:
        print(f"[epex] {EPEX_HOURLY_REL} already present — skipping bootstrap "
              f"(use --bootstrap-epex to force).")
        return False

    src = _REPO_ROOT / EPEX_15MIN_REL
    if not src.exists():
        raise FileNotFoundError(
            f"Cannot bootstrap {EPEX_HOURLY_REL}: governed local source "
            f"{EPEX_15MIN_REL} is missing."
        )

    series = pd.read_parquet(src)["price_eur_mwh"]
    hourly = series.resample("1h").mean()
    hourly = hourly.interpolate("time").ffill().bfill()
    out.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_frame("price_eur_mwh").to_parquet(out)
    print(
        f"[epex] wrote {EPEX_HOURLY_REL}: {len(hourly)} rows, "
        f"{hourly.index.min()} -> {hourly.index.max()}, "
        f"mean {hourly.mean():.1f} EUR/MWh, NaN {int(hourly.isna().sum())}"
    )
    return True


def _preflight(epex_source: str) -> None:
    print("[preflight]")
    epex_h = _REPO_ROOT / EPEX_HOURLY_REL
    fwds = _REPO_ROOT / FORWARDS_REL
    print(f"  {EPEX_HOURLY_REL:<40} present={epex_h.exists()}")
    print(f"  {FORWARDS_REL:<40} present={fwds.exists()}")
    if epex_source == "parquet" and not fwds.exists():
        print(
            f"  WARNING: {FORWARDS_REL} missing — a parquet run will fail. "
            f"Run scripts/import_fmv_forwards.py first (or use --epex-source mock)."
        )


def _print_pillar1(results: dict) -> None:
    pillar1 = results["pillar1"]
    n_pass = sum(1 for v in pillar1.values() if v.passed)
    n_total = len(pillar1)
    gate = "Y" if results.get("sc1_gate_eligible") else "N"
    print("\n=== Pillar 1 — Hildmann SC#1 gate ===")
    print(f"forwards_source : {results.get('forwards_source')}")
    print(f"gate-eligible   : {gate}")
    print(f"{'test':<18}{'observed':>14}{'threshold':>16}  pass")
    for name, res in pillar1.items():
        thr = res.threshold
        thr_s = f"[{thr[0]}, {thr[1]}]" if isinstance(thr, tuple) else f"{thr}"
        obs = res.observed
        obs_s = f"{obs:.4f}" if isinstance(obs, float) and obs == obs else str(obs)
        print(f"{name:<18}{obs_s:>14}{thr_s:>16}  {'OK' if res.passed else 'X'}")
    print(f"verdict         : {n_pass}/{n_total} PASS — "
          f"{'PASS' if n_pass == n_total else 'FAIL'}")

    seasonal = pillar1.get("seasonal_profile")
    if seasonal is not None and isinstance(seasonal.details, dict):
        d = seasonal.details
        # Per-vintage aggregated result: dig the worst (min pearson) vintage row.
        inner = d
        if "per_vintage" in d and d["per_vintage"]:
            rows = [r for r in d["per_vintage"] if not r.get("degenerate")]
            rows = rows or d["per_vintage"]
            worst = min(rows, key=lambda r: r.get("observed", float("inf")))
            inner = worst.get("details", {})
            print("\n--- seasonal_profile shape diagnostics (worst vintage) ---")
            print(f"  pass_rate           : {d.get('n_pass')}/{d.get('n_vintages')}")
            print(f"  worst vintage       : {worst.get('vintage')} "
                  f"(pearson {worst.get('observed')})")
        elif "spearman_r" in d:
            print("\n--- seasonal_profile shape diagnostics ---")
        if isinstance(inner, dict) and "spearman_r" in inner:
            print(f"  spearman_r          : {inner['spearman_r']}")
            print(f"  top divergent months: {inner.get('top_divergent_months')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="One-command real-run launcher for the Phase 10 scorecard."
    )
    parser.add_argument("--epex-source", choices=("parquet", "mock"), default="parquet")
    parser.add_argument(
        "--output-dir",
        default=".planning/phases/10-pfc-fmv-quality-scorecard",
    )
    parser.add_argument("--vintages-limit", type=int, default=None)
    parser.add_argument(
        "--bootstrap-epex",
        action="store_true",
        help="Force-rebuild data/epex_hourly.parquet even if it exists.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Remove the per-vintage cache/ dir before running (cold cache).",
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _REPO_ROOT / output_dir

    if args.epex_source == "parquet":
        bootstrap_epex_hourly(force=args.bootstrap_epex)
    _preflight(args.epex_source)

    if args.fresh:
        cache_dir = output_dir / "cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"[fresh] removed {cache_dir}")

    from pfc_shaping.validation.scorecard import (
        render_figures,
        render_markdown_report,
        run_scorecard_full,
    )

    print(f"[run] run_scorecard_full(epex_source={args.epex_source}) — this is slow "
          f"on a cold cache (~30 min for 24 vintages).")
    results = run_scorecard_full(
        epex_source=args.epex_source,
        output_dir=output_dir,
        vintages_limit=args.vintages_limit,
        cache_intermediate=True,
        progress_callback=lambda m: print(m),
    )

    if not args.no_figures:
        render_figures(results, output_dir)
    if not args.no_report:
        md = render_markdown_report(results, output_dir)
        print(f"[report] {md}")

    _print_pillar1(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
