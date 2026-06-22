"""
test_long_term_branch.py
------------------------
Phase 1bis LT — verify the factored ``_build_long_term_branch(spec, ...)``
in pfc_shaping/pipeline/production_phases.py.

These tests don't run the full LT pipeline (which needs heavy fitted
artifacts and full EPEX history). They focus on:

  1. The new MarketSpec / MarketBranchArtifacts dataclasses.
  2. Backward-compat properties (.base_prices_ch / _de etc.).
  3. LongTermArtifacts.swiss / .german aliases over a markets dict.
  4. Type aliasing of SwissLongTermArtifacts and
     GermanLongTermArtifacts onto MarketBranchArtifacts.
  5. Per-market artifact suffix mapping used by save_long_term_outputs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pfc_shaping.pipeline.production_phases import (
    GermanLongTermArtifacts,
    LongTermArtifacts,
    MarketBranchArtifacts,
    MarketSpec,
    SharedStructuralArtifacts,
    SwissLongTermArtifacts,
    _ARTIFACT_SUFFIX,
    _save_monthly_curve_manifests,
)


def _make_branch(code: str = "CH", **overrides) -> MarketBranchArtifacts:
    defaults = dict(
        code=code,
        pfc=pd.DataFrame({"price_shape": [10.0, 11.0]}),
        sh=None,
        base_prices={"2026": 80.0, "2026-Peak": 95.0},
        cascaded_prices={"2026": 80.0, "2026-Q1": 90.0, "2026-Peak": 95.0},
        fwd_source="test-source",
        out_base=f"/tmp/pfc_15min_{code.lower()}_2026",
    )
    defaults.update(overrides)
    return MarketBranchArtifacts(**defaults)


# ---------------------------------------------------------------------------
# MarketSpec
# ---------------------------------------------------------------------------


def test_market_spec_minimal_construction() -> None:
    spec = MarketSpec(
        code="DE",
        sheet="DE",
        tz="Europe/Berlin",
        country="DE",
        epex_df=pd.DataFrame(),
        cal_df=pd.DataFrame(),
    )
    # Default fields must be falsy / None — only DE-like minimal markets.
    assert spec.pre_fitted_sh is None
    assert spec.water_value is None
    assert spec.hydro_forecast is None
    assert spec.outages_forecast is None
    assert spec.out_base == ""


def test_market_spec_swiss_full_construction() -> None:
    spec = MarketSpec(
        code="CH",
        sheet="CH",
        tz="Europe/Zurich",
        country="CH",
        epex_df=pd.DataFrame(),
        cal_df=pd.DataFrame(),
        pre_fitted_sh=object(),
        water_value=object(),
        hydro_forecast=pd.DataFrame({"fill_deviation": [0.0]}),
        outages_forecast=pd.DataFrame({"unavailable_mw": [0.0]}),
        out_base="/tmp/pfc_15min_ch",
    )
    assert spec.pre_fitted_sh is not None
    assert spec.water_value is not None
    assert spec.hydro_forecast is not None


# ---------------------------------------------------------------------------
# MarketBranchArtifacts + legacy property aliases
# ---------------------------------------------------------------------------


def test_legacy_aliases_are_market_branch_artifacts() -> None:
    """SwissLongTermArtifacts / GermanLongTermArtifacts must be the same
    class object as MarketBranchArtifacts (cheap aliasing for any legacy
    isinstance check)."""
    assert SwissLongTermArtifacts is MarketBranchArtifacts
    assert GermanLongTermArtifacts is MarketBranchArtifacts


def test_legacy_property_base_prices_ch_returns_base_prices() -> None:
    art = _make_branch("CH")
    assert art.base_prices_ch is art.base_prices
    assert art.cascaded_prices_ch is art.cascaded_prices


def test_legacy_property_base_prices_de_returns_base_prices_on_any_branch() -> None:
    """The _ch / _de aliases are pure name aliases — they don't filter
    by code. A consumer reading ``art.base_prices_de`` on a German branch
    gets the German prices; reading the same alias on a Swiss branch
    gets the Swiss prices. This is intentional backward-compat semantics.
    """
    de = _make_branch("DE")
    assert de.base_prices_de is de.base_prices
    assert de.cascaded_prices_de is de.cascaded_prices


# ---------------------------------------------------------------------------
# LongTermArtifacts.markets + swiss / german properties
# ---------------------------------------------------------------------------


def test_long_term_artifacts_swiss_german_properties() -> None:
    swiss_art = _make_branch("CH")
    german_art = _make_branch("DE")
    shared = SharedStructuralArtifacts(
        si=None, unc=None, calibrator=None,
        entso_forecast=pd.DataFrame(),
        start_date="2026-05-06",
        horizon_days=1095,
    )
    lt = LongTermArtifacts(
        shared=shared,
        markets={"CH": swiss_art, "DE": german_art},
        out_dir="/tmp",
        artifacts_dir="/tmp",
        today="2026-05-06",
    )
    assert lt.swiss is swiss_art
    assert lt.german is german_art


def test_long_term_artifacts_supports_extra_markets_without_field_change() -> None:
    """Adding a new market (e.g. FR) must NOT require touching the
    LongTermArtifacts dataclass — only inserting the entry into
    ``markets`` should suffice."""
    swiss = _make_branch("CH")
    german = _make_branch("DE")
    french = _make_branch("FR")
    austrian = _make_branch("AT")
    italian = _make_branch("IT")

    shared = SharedStructuralArtifacts(
        si=None, unc=None, calibrator=None,
        entso_forecast=pd.DataFrame(),
        start_date="",
        horizon_days=0,
    )
    lt = LongTermArtifacts(
        shared=shared,
        markets={"CH": swiss, "DE": german, "FR": french, "AT": austrian, "IT": italian},
        out_dir="/tmp",
        artifacts_dir="/tmp",
        today="",
    )
    # Backward-compat aliases keep working.
    assert lt.swiss is swiss
    assert lt.german is german
    # New markets are addressable via the dict.
    assert lt.markets["FR"] is french
    assert lt.markets["AT"] is austrian
    assert lt.markets["IT"] is italian


# ---------------------------------------------------------------------------
# Artifact suffix mapping
# ---------------------------------------------------------------------------


def test_artifact_suffix_legacy_ch_is_empty() -> None:
    """CH must keep the unsuffixed legacy artifact paths
    ('shape_hourly.parquet', 'water_value.parquet') so the dashboard
    and downstream consumers keep reading the same paths."""
    assert _ARTIFACT_SUFFIX["CH"] == ""


@pytest.mark.parametrize(
    "code, expected",
    [("DE", "_de"), ("AT", "_at"), ("FR", "_fr"), ("IT", "_it")],
)
def test_artifact_suffix_per_market(code: str, expected: str) -> None:
    assert _ARTIFACT_SUFFIX[code] == expected


def test_artifact_suffix_covers_full_panel() -> None:
    """The mapping must contain at least the 5 markets we plan to
    activate in Phase 3. Adding more later is fine."""
    assert {"CH", "DE", "FR", "AT", "IT"}.issubset(_ARTIFACT_SUFFIX.keys())


def test_save_monthly_curve_manifests_writes_production_ch_manifest(tmp_path) -> None:
    manifest = {
        "monthly_solution_hash": "solution",
        "active_constraints_hash": "constraints",
        "active_config_hash": "config",
    }

    _save_monthly_curve_manifests({"CH": manifest}, str(tmp_path), logger=_NullLogger())

    path = tmp_path / "production_monthly_curve_manifest.json"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert '"monthly_solution_hash": "solution"' in text
    assert '"active_constraints_hash": "constraints"' in text


class _NullLogger:
    def info(self, *args, **kwargs) -> None:
        return None
