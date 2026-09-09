#!/usr/bin/env python3
"""Plan 10-04 Task 2 — CLI runner for the 96-build PFC FMV Quality Scorecard.

Orchestrates the 5-pillar Phase 10 scorecard end-to-end on Mac Mini local
execution (~8-10 min full run with cache, ~1.5-2h cold cache):

    python scripts/run_phase10_scorecard.py \\
        --epex-source parquet \\
        --output-dir .planning/phases/10-pfc-fmv-quality-scorecard

Flags
-----
- ``--vintages-limit N``       : dry-run subset (1 = ~5 min).
- ``--reproducibility-subset N`` : run Config 4 only × N vintages (used by
  ``tests/test_phase10_reproducibility.py``).
- ``--no-figures`` / ``--no-report`` : skip render_figures / render_markdown_report.
- ``--epex-source mock``       : use deterministic synth EPEX (seed=42) instead
  of ``data/epex_hourly.parquet``.

Outputs (under ``--output-dir``)
--------------------------------
- ``cache/pfc_{config}_{vintage}.parquet`` (96 files, ~25 MB each, gitignored)
- ``scorecard_kpis_pillar{2,3,4}.parquet`` (deterministic column order)
- ``figures/pillar{1,2,2,3}_*.png`` (4 figures publication-grade, ≥150 dpi)
- ``10-VERIFICATION.md`` (skeleton — Pillar 5 placeholder filled by Plan 10-04 Task 3)
- ``run_phase10.log`` (INFO logging redirect)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running this script directly without `pip install -e .` — prepend
# the repository root (the script lives in `repo/scripts/`).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("run_phase10_scorecard")
    logger.setLevel(logging.INFO)
    # Strip pre-existing handlers (idempotent reload)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan 10-04 Task 2 — 96-build PFC FMV Quality Scorecard runner."
    )
    parser.add_argument(
        "--epex-source",
        choices=("parquet", "mock"),
        default="parquet",
        help="EPEX source ('parquet' real cache / 'mock' synth seed=42).",
    )
    parser.add_argument(
        "--output-dir",
        default=".planning/phases/10-pfc-fmv-quality-scorecard",
        help="Output directory (cache, KPIs, figures, markdown report).",
    )
    parser.add_argument(
        "--vintages-limit",
        type=int,
        default=None,
        help="Truncate vintages to N (dry-run, default = all 24).",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip render_figures step.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip render_markdown_report step (10-VERIFICATION.md).",
    )
    parser.add_argument(
        "--reproducibility-subset",
        type=int,
        default=None,
        help="Reproducibility test subset (Config 4 × N vintages).",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "run_phase10.log"
    logger = _setup_logger(log_file)

    # Late import (so --help doesn't pay the cost)
    from pfc_shaping.validation.scorecard import (
        render_figures,
        render_markdown_report,
        run_scorecard_full,
    )

    # Progress callback : just logs ; never raises
    def progress(msg: str) -> None:
        logger.info(msg)

    t0 = time.perf_counter()
    logger.info(
        "[Plan 10-04 Task 2] run_scorecard_full epex_source=%s output_dir=%s "
        "vintages_limit=%s reproducibility_subset=%s",
        args.epex_source,
        output_dir,
        args.vintages_limit,
        args.reproducibility_subset,
    )

    vintages_limit = args.vintages_limit
    if args.reproducibility_subset is not None:
        vintages_limit = args.reproducibility_subset

    results = run_scorecard_full(
        epex_source=args.epex_source,
        output_dir=output_dir,
        vintages_limit=vintages_limit,
        cache_intermediate=True,
        progress_callback=progress,
    )

    sc1_passed = sum(1 for v in results["pillar1"].values() if v.passed)
    sc1_total = len(results["pillar1"])
    forwards_source = results["forwards_source"]
    gate_eligible = results["sc1_gate_eligible"]
    gate_str = "Y" if gate_eligible else f"N (forwards_source={forwards_source})"
    logger.info(
        "SC#1 verdict: %s %d/%d",
        "PASS" if sc1_passed == sc1_total else "FAIL",
        sc1_passed,
        sc1_total,
    )
    logger.info("SC#1 gate-eligible: %s", gate_str)

    if not args.no_figures:
        logger.info("[Plan 10-04 Task 2] render_figures")
        figs = render_figures(results, output_dir)
        logger.info("Figures written: %s", [str(f) for f in figs])
    if not args.no_report:
        logger.info("[Plan 10-04 Task 2] render_markdown_report")
        md = render_markdown_report(results, output_dir)
        logger.info("Markdown report written: %s", md)

    elapsed = time.perf_counter() - t0
    logger.info(
        "[Plan 10-04 Task 2] DONE in %.1fs (compute=%s, configs=%s, vintages=%s)",
        elapsed,
        results["compute_seconds"],
        len(results["configs"]),
        len(results["vintages"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
