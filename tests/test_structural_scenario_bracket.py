from __future__ import annotations

import pandas as pd
import pytest

from pfc_shaping.lt.model.structural_scenario_bracket import recompute_structural_scenario_bracket


def test_structural_scenario_bracket_is_ordered_and_non_probabilistic_alias() -> None:
    frame = pd.DataFrame(
        {
            "price_slow_eur_mwh": [120.0],
            "price_central_eur_mwh": [100.0],
            "price_fast_eur_mwh": [90.0],
        }
    )

    out = recompute_structural_scenario_bracket(
        frame,
        {"slow": 0.25, "central": 0.50, "fast": 0.25},
        include_legacy_aliases=True,
    )

    assert out.loc[0, "price_weighted_mean_eur_mwh"] == pytest.approx(102.5)
    assert out.loc[0, "structural_scenario_low_eur_mwh"] == pytest.approx(90.0)
    assert out.loc[0, "structural_scenario_central_eur_mwh"] == pytest.approx(100.0)
    assert out.loc[0, "structural_scenario_high_eur_mwh"] == pytest.approx(120.0)
    assert out.loc[0, "structural_scenario_spread_eur_mwh"] == pytest.approx(30.0)
    assert out.loc[0, "structural_p10_eur_mwh"] == pytest.approx(90.0)
    assert out.loc[0, "structural_p50_eur_mwh"] == pytest.approx(100.0)
    assert out.loc[0, "structural_p90_eur_mwh"] == pytest.approx(120.0)
    assert out.loc[0, "structural_width_eur_mwh"] == pytest.approx(30.0)
