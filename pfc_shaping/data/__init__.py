"""Data loading helpers with lazy optional scenario exports."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ELECTRIFICATION_SCENARIO_REQUIRED_COLUMNS",
    "ELECTRIFICATION_SCENARIO_RECOMMENDED_COLUMNS",
    "HPFC_SCENARIO_FEATURE_COLUMNS",
    "asof_electrification_scenarios",
    "assert_scenario_coverage",
    "derive_hpfc_scenario_features",
    "interpolate_electrification_scenario_years",
    "load_electrification_scenarios",
    "validate_electrification_scenarios",
    "write_hpfc_scenario_features",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from pfc_shaping.data import electrification_scenarios

    aliases = {
        "ELECTRIFICATION_SCENARIO_REQUIRED_COLUMNS": "REQUIRED_COLUMNS",
        "ELECTRIFICATION_SCENARIO_RECOMMENDED_COLUMNS": "RECOMMENDED_COLUMNS",
    }
    return getattr(electrification_scenarios, aliases.get(name, name))
