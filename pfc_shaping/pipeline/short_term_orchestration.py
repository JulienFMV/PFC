"""Top-level CT orchestration kept outside the governed LT wheel."""

from __future__ import annotations

import logging
import os

from pfc_shaping.pipeline.production_phases import LoadedInputs, LongTermArtifacts


def run_short_term_phase(
    project_root: str,
    inputs: LoadedInputs,
    long_term: LongTermArtifacts,
    logger: logging.Logger,
):
    from pfc_shaping.pipeline.swiss_short_term import (
        SwissShortTermInputs,
        run_swiss_short_term_overlay,
    )

    st_inputs = SwissShortTermInputs(
        epex_ch=inputs.epex_ch,
        epex_de=inputs.epex_de,
        neighbor_prices_15min=inputs.neighbor_prices_15min,
        entso=inputs.entso,
        hydro=inputs.hydro,
        commodities=inputs.commodities,
        outages_all=inputs.outages_all,
        base_pfc_ch=long_term.swiss.pfc,
        require_de_exogenous=os.getenv("PFC_CT_REQUIRE_DE_EXOGENOUS", "1") == "1",
        required_neighbor_codes=("de",),
    )
    return run_swiss_short_term_overlay(
        project_root=project_root,
        inputs=st_inputs,
        logger=logger,
    )
