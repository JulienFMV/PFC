#!/usr/bin/env python3
"""
PFC 15min Production Run
========================
Thin entrypoint that orchestrates the long-term structural phase and the
short-term LEAR overlay phase through dedicated pipeline helpers.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

PEAK_SOURCE_POLICY = os.environ.get("PFC_PEAK_SOURCE_POLICY", "same_first").strip() or "same_first"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PFC_PROD")

from pfc_shaping.pipeline.production_phases import (
    load_inputs,
    print_pipeline_summary,
    run_long_term_phase,
    run_short_term_phase,
    save_long_term_outputs,
)


def main() -> None:
    logger.info("Peak source policy: %s", PEAK_SOURCE_POLICY)
    started_at = time.time()

    inputs = load_inputs(PROJECT_ROOT, logger)
    long_term = run_long_term_phase(
        project_root=PROJECT_ROOT,
        inputs=inputs,
        peak_source_policy=PEAK_SOURCE_POLICY,
        logger=logger,
    )

    try:
        short_term = run_short_term_phase(
            project_root=PROJECT_ROOT,
            inputs=inputs,
            long_term=long_term,
            logger=logger,
        )
        long_term.swiss.pfc = short_term.pfc_ch
    except Exception as exc:
        logger.exception("  LEAR overlay failed. PFC output is not a validated CT forecast.")
        if os.getenv("PFC_ALLOW_LEAR_FAILURE", "0") != "1":
            raise RuntimeError(
                "LEAR overlay failed; set PFC_ALLOW_LEAR_FAILURE=1 only for explicit fallback runs."
            ) from exc

    save_long_term_outputs(long_term, logger)
    logger.info("=" * 70)
    logger.info("STEP 11: Comprehensive Statistics")
    logger.info("=" * 70)
    print_pipeline_summary(long_term, total_time=time.time() - started_at)


if __name__ == "__main__":
    main()
