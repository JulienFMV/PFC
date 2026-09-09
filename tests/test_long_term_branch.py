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
  5. Governed monthly-manifest and solver-grid helpers.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from pfc_shaping.pipeline.production_phases import (
    GermanLongTermArtifacts,
    LongTermArtifacts,
    MarketBranchArtifacts,
    MarketSpec,
    SharedStructuralArtifacts,
    SwissLongTermArtifacts,
    _build_long_term_branch,
    _save_monthly_curve_manifests,
    _solver_delivery_quarter_hour_grid,
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


def test_long_term_branch_passes_governed_reference_timestamp_to_assembler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Assembler:
        def __init__(self, **kwargs) -> None:
            self.final_product_projection_report_ = {}

        def build(self, **kwargs) -> pd.DataFrame:
            captured.update(kwargs)
            return pd.DataFrame(
                {"price_shape": [80.0]},
                index=pd.DatetimeIndex([pd.Timestamp("2027-01-01T00:00:00Z")]),
            )

    monkeypatch.setattr("pfc_shaping.lt.model.assembler.PFCAssembler", _Assembler)
    reference = pd.Timestamp("2026-07-13T12:34:56+02:00")
    inputs = SimpleNamespace(
        reference_timestamp=reference,
        sh_mode="seasonal",
        config={},
        eex_report_path=None,
    )
    shared = SharedStructuralArtifacts(
        si=object(),
        unc=object(),
        calibrator=None,
        entso_forecast=pd.DataFrame(),
        start_date="2027-01-01",
        horizon_days=1,
    )
    spec = MarketSpec(
        code="CH",
        sheet="CH",
        tz="Europe/Zurich",
        country="CH",
        epex_df=pd.DataFrame(),
        cal_df=pd.DataFrame(),
        pre_fitted_sh=object(),
    )

    _build_long_term_branch(
        spec=spec,
        inputs=inputs,
        shared=shared,
        peak_source_policy="same_first",
        use_seasonal_hourly_shape=True,
        logger=_NullLogger(),
        pre_loaded_base_prices={"2027": 80.0},
        pre_loaded_fwd_source="fixture",
        pre_loaded_cascaded_prices={"2027": 80.0},
        pre_loaded_cascader=object(),
    )

    assert captured["reference_date"] == reference.tz_convert("UTC")


def test_solver_delivery_grid_rejects_noncontiguous_months() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        _solver_delivery_quarter_hour_grid(
            pd.PeriodIndex(["2027-01", "2027-03"], freq="M"),
            timezone="Europe/Zurich",
        )


class _NullLogger:
    def info(self, *args, **kwargs) -> None:
        return None
