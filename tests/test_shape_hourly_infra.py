"""
test_shape_hourly_infra.py
--------------------------
Tests for ShapeHourly infrastructure: save/load sidecar roundtrip (Plan 05B-02),
feature flag PFC_LT_USE_SEASONAL_HOURLY_SHAPE (Plan 05B-03),
read-only 3D view factors_3d_ (Plan 05B-04 Task 1),
assembler capability check (Plan 05B-04 Task 2), and Phase 5bis-A no-op proof
(Plan 05B-05 Task 3: seven standalone tests D-14..D-19 + Codex review §8).

Plans covered: 05B-02, 05B-03, 05B-04, 05B-05
Requirements: SHP-01 (D-01/D-14), SHP-04 (D-07/D-18)
Context ref: .planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md

Tests verify:
- Task 1: save() writes ${stem}.meta.parquet sidecar with correct schema and content
- Task 2: load() restores all attributes from sidecar; legacy compat warning on missing sidecar
- Plan 05B-03 Task 1: feature flag constructor arg + env-default + freeze-at-init
- Plan 05B-03 Task 2: flag persisted in meta sidecar and restored on load (parquet wins over env)
- Plan 05B-04 Task 1: factors_3d_ read-only Mapping view on ShapeHourly (SHP-01 literal)
- Plan 05B-04 Task 2: assembler _sh_apply_accepts_outages helper + _sh_accepts_outages instance cache
- Plan 05B-05 D-14: test_factors_3d_view_consistency — all cells × 24 hours consistent
- Plan 05B-05 D-15: test_save_load_full_roundtrip — all 10 attributes roundtrip
- Plan 05B-05 D-16: test_save_load_legacy_compat — legacy fixture from fixtures dir
- Plan 05B-05 D-17: test_flag_freeze_at_init — four combinatorial freeze assertions
- Plan 05B-05 D-18: test_flag_persisted_in_parquet — parquet wins both directions
- Plan 05B-05 D-19: test_baseline_regression — THE proof of 5bis-A no-op
- Plan 05B-05 Codex §8: test_no_hidden_behavior_branch — AST guard on _use_seasonal_hourly
"""

import ast
import inspect
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.model.shape_hourly import ShapeHourly, _META_SIDECAR_SUFFIX, _meta_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_fitted_sh(
    sigma: float = 0.3,
    halflife_days: float = 90.0,
    hydro_weight_sigma: float = 0.7,
) -> ShapeHourly:
    """Create a ShapeHourly with realistic attributes set (no real data needed)."""
    sh = ShapeHourly(sigma=sigma, halflife_days=halflife_days, hydro_weight_sigma=hydro_weight_sigma)
    # factors_
    arr = np.linspace(0.8, 1.2, 24)
    arr = arr / arr.mean()
    sh.factors_[("Hiver", "Ouvrable")] = arr.copy()
    sh.n_obs_[("Hiver", "Ouvrable")] = 100
    # f_W_
    sh.f_W_["Ouvrable"] = 1.05
    # f_W_seasonal_
    sh.f_W_seasonal_[("Hiver", "Ouvrable")] = 1.08
    # factors_by_year_
    arr_y = np.linspace(0.7, 1.3, 24)
    arr_y = arr_y / arr_y.mean()
    sh.factors_by_year_[("Hiver", "Ouvrable", 2023)] = arr_y.copy()
    # trend_per_hour_
    sh.trend_per_hour_[("Hiver", "Ouvrable")] = np.linspace(-0.01, 0.01, 24)
    # _climatological_fill
    sh._climatological_fill = pd.Series([0.5, 0.6, 0.7], index=[1, 2, 3])
    # global_factors_
    sh.global_factors_ = sh._compute_global_fallback()
    return sh


# ===========================================================================
# Task 1: save() sidecar tests
# ===========================================================================

class TestMetaPathHelper:
    """Test 1 acceptance: module-level constants and _meta_path helper."""

    def test_meta_sidecar_suffix_constant(self):
        assert _META_SIDECAR_SUFFIX == ".meta.parquet"

    def test_meta_path_from_parquet(self):
        result = _meta_path("shape_hourly.parquet")
        assert result.name == "shape_hourly.meta.parquet"

    def test_meta_path_preserves_directory(self):
        result = _meta_path("/some/dir/shape_hourly.parquet")
        assert result == Path("/some/dir/shape_hourly.meta.parquet")

    def test_meta_path_different_stem(self):
        result = _meta_path("/tmp/my_model.parquet")
        assert result.name == "my_model.meta.parquet"


class TestSaveSidecarExists:
    """Test 1: sidecar file is created next to main parquet."""

    def test_sidecar_created_on_save(self):
        sh = _minimal_fitted_sh()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "shape_hourly.parquet")
            sh.save(p)
            meta_p = os.path.join(d, "shape_hourly.meta.parquet")
            assert os.path.exists(meta_p), f"Expected {meta_p} to exist"

    def test_sidecar_is_parquet(self):
        sh = _minimal_fitted_sh()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "shape_hourly.parquet")
            sh.save(p)
            meta_p = os.path.join(d, "shape_hourly.meta.parquet")
            meta = pd.read_parquet(meta_p)
            assert isinstance(meta, pd.DataFrame)

    def test_sidecar_has_attr_column(self):
        sh = _minimal_fitted_sh()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "shape_hourly.parquet")
            sh.save(p)
            meta = pd.read_parquet(os.path.join(d, "shape_hourly.meta.parquet"))
            assert "attr" in meta.columns


class TestSaveSidecarSchema:
    """Tests 2-8: correct rows and schema in sidecar."""

    def setup_method(self):
        self.sh = _minimal_fitted_sh()
        self.tmpdir = tempfile.mkdtemp()
        self.p = os.path.join(self.tmpdir, "shape_hourly.parquet")
        self.sh.save(self.p)
        self.meta = pd.read_parquet(os.path.join(self.tmpdir, "shape_hourly.meta.parquet"))

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_discriminator_attrs_present(self):
        """Test 2: discriminator column `attr` covers expected values."""
        present = set(self.meta["attr"].unique())
        assert "factors_by_year_" in present
        assert "trend_per_hour_" in present
        assert "f_W_seasonal_" in present
        assert "_climatological_fill" in present
        assert "hyperparams" in present

    def test_factors_by_year_schema(self):
        """Test 3: factors_by_year_ — 24 rows per (saison, type_jour, year) with correct columns."""
        rows = self.meta[self.meta["attr"] == "factors_by_year_"]
        assert len(rows) == 24, f"Expected 24 rows, got {len(rows)}"
        assert {"saison", "type_jour", "year", "heure", "value"}.issubset(rows.columns)
        grp = rows.groupby(["saison", "type_jour", "year"])
        assert len(grp) == 1  # one cell: (Hiver, Ouvrable, 2023)
        arr_loaded = grp.get_group(("Hiver", "Ouvrable", 2023)).sort_values("heure")["value"].apply(float).to_numpy()
        np.testing.assert_allclose(
            arr_loaded,
            self.sh.factors_by_year_[("Hiver", "Ouvrable", 2023)],
            atol=1e-12, rtol=0
        )

    def test_trend_per_hour_schema(self):
        """Test 4: trend_per_hour_ — 24 rows per (saison, type_jour)."""
        rows = self.meta[self.meta["attr"] == "trend_per_hour_"]
        assert len(rows) == 24
        assert {"saison", "type_jour", "heure", "value"}.issubset(rows.columns)
        grp = rows.groupby(["saison", "type_jour"])
        arr_loaded = grp.get_group(("Hiver", "Ouvrable")).sort_values("heure")["value"].apply(float).to_numpy()
        np.testing.assert_allclose(
            arr_loaded,
            self.sh.trend_per_hour_[("Hiver", "Ouvrable")],
            atol=1e-12, rtol=0
        )

    def test_f_W_seasonal_schema(self):
        """Test 5: f_W_seasonal_ — 1 row per (saison, type_jour)."""
        rows = self.meta[self.meta["attr"] == "f_W_seasonal_"]
        assert len(rows) == 1
        assert {"saison", "type_jour", "value"}.issubset(rows.columns)
        val = float(rows.iloc[0]["value"])
        assert val == pytest.approx(1.08)

    def test_climatological_fill_schema(self):
        """Test 6: _climatological_fill — rows with week + value columns."""
        rows = self.meta[self.meta["attr"] == "_climatological_fill"]
        assert len(rows) == 3
        assert {"week", "value"}.issubset(rows.columns)
        loaded = rows.sort_values("week")["value"].apply(float).to_numpy()
        np.testing.assert_allclose(loaded, [0.5, 0.6, 0.7], atol=1e-12, rtol=0)

    def test_hyperparams_row(self):
        """Test 8: hyperparams — single row with JSON value.
        After Plan 05B-03, use_seasonal_hourly is also persisted in hyperparams.
        Updated by Plan 05C-01 (D-A3-3 / RESEARCH Pitfall 4): hyperparams JSON gains
        hydro_weight_sigma_off/_on/_resolved keys when 5bis-B Lever 1 ships.
        Updated by Plan 05C-03 (D-A3-3 / RESEARCH Pitfall 4 second wave): hyperparams JSON gains
        the sigma_off/_on/_resolved triplet. Total schema = 10 keys.
        The constructor _minimal_fitted_sh(sigma=0.3, hydro_weight_sigma=0.7) passes the legacy
        single-sigma args → D-A3-2 / D-A1-5 backward-compat: off=on=0.3/0.7 (legacy wins).
        """
        hp_rows = self.meta[self.meta["attr"] == "hyperparams"]
        assert len(hp_rows) == 1
        obj = json.loads(hp_rows["value"].iloc[0])
        # use_seasonal_hourly added by Plan 05B-03 — must be present
        # hydro_weight_sigma_off/_on/_resolved added by Plan 05C-01 (D-A3-3)
        # sigma_off/_on/_resolved added by Plan 05C-03 (D-A3-3 / RESEARCH Pitfall 4 second wave)
        assert obj == {
            "halflife_days": 90.0,
            "hydro_weight_sigma": 0.7,
            "hydro_weight_sigma_off": 0.7,    # legacy single-sigma wins → off = 0.7
            "hydro_weight_sigma_on": 0.7,     # legacy single-sigma wins → on = 0.7
            "hydro_weight_sigma_resolved": 0.7,
            "sigma": 0.3,
            "sigma_off": 0.3,                  # legacy sigma wins → off = 0.3 (D-A3-2)
            "sigma_on": 0.3,                   # legacy sigma wins → on = 0.3 (D-A3-2)
            "sigma_resolved": 0.3,
            "use_seasonal_hourly": False,  # default: _minimal_fitted_sh() uses no flag
        }

    def test_global_factors_not_persisted(self):
        """Test 10: global_factors_ must NOT be written to the meta sidecar."""
        count = (self.meta["attr"] == "global_factors_").sum()
        assert count == 0, "global_factors_ must NOT be persisted to the meta sidecar"


class TestSaveClimatologicalFillNone:
    """Test 6 (None branch): when _climatological_fill is None, no rows for that attr."""

    def test_no_climatological_fill_rows_when_none(self):
        sh = ShapeHourly()
        sh.factors_[("Hiver", "Ouvrable")] = np.ones(24)
        sh.n_obs_[("Hiver", "Ouvrable")] = 100
        sh.f_W_["Ouvrable"] = 1.0
        # _climatological_fill is None by default
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "shape_hourly.parquet")
            sh.save(p)
            meta = pd.read_parquet(os.path.join(d, "shape_hourly.meta.parquet"))
            count = (meta["attr"] == "_climatological_fill").sum()
            assert count == 0


class TestSaveUnfitted:
    """Test 9: saving an unfitted ShapeHourly (empty factors_) does not crash."""

    def test_save_unfitted_no_crash(self):
        sh = ShapeHourly(sigma=0.5, halflife_days=180.0, hydro_weight_sigma=0.25)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "shape_hourly.parquet")
            sh.save(p)
            meta_p = os.path.join(d, "shape_hourly.meta.parquet")
            assert os.path.exists(meta_p)
            meta = pd.read_parquet(meta_p)
            hp = meta[meta["attr"] == "hyperparams"]
            assert len(hp) == 1  # hyperparams row always present

    def test_save_unfitted_hyperparams_correct(self):
        """Updated by Plan 05C-01 (D-A3-3 / RESEARCH Pitfall 4): hyperparams JSON gains
        hydro_weight_sigma_off/_on/_resolved keys when 5bis-B Lever 1 ships.
        Updated by Plan 05C-03 (D-A3-3 / RESEARCH Pitfall 4 second wave): hyperparams JSON gains
        the sigma_off/_on/_resolved triplet alongside hydro_weight_sigma_off/_on/_resolved
        added by Plan 05C-01. Total schema = 10 keys.
        Constructor ShapeHourly(sigma=0.5, hydro_weight_sigma=0.25) passes legacy single-sigma →
        D-A3-2 / D-A1-5 backward-compat: off=on=0.5/0.25 (legacy wins for both).
        """
        sh = ShapeHourly(sigma=0.5, halflife_days=180.0, hydro_weight_sigma=0.25)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "shape_hourly.parquet")
            sh.save(p)
            meta = pd.read_parquet(os.path.join(d, "shape_hourly.meta.parquet"))
            hp = meta[meta["attr"] == "hyperparams"]
            obj = json.loads(hp["value"].iloc[0])
            # Plan 05B-03: use_seasonal_hourly now included in hyperparams
            # Plan 05C-01 (D-A3-3): hydro_weight_sigma_off/_on/_resolved added
            # Plan 05C-03 (D-A3-3 / RESEARCH Pitfall 4 second wave): sigma_off/_on/_resolved added
            assert obj == {
                "halflife_days": 180.0,
                "hydro_weight_sigma": 0.25,
                "hydro_weight_sigma_off": 0.25,    # legacy single-sigma wins → off = 0.25
                "hydro_weight_sigma_on": 0.25,     # legacy single-sigma wins → on = 0.25
                "hydro_weight_sigma_resolved": 0.25,
                "sigma": 0.5,
                "sigma_off": 0.5,                  # legacy sigma wins → off = 0.5 (D-A3-2)
                "sigma_on": 0.5,                   # legacy sigma wins → on = 0.5 (D-A3-2)
                "sigma_resolved": 0.5,
                "use_seasonal_hourly": False,
            }


# ===========================================================================
# Task 2: load() restore tests
# ===========================================================================

class TestLoadFullRoundtrip:
    """Tests 1-6, 8: full roundtrip via save/load restores all attributes."""

    def setup_method(self):
        self.sh1 = _minimal_fitted_sh(sigma=0.3, halflife_days=90.0, hydro_weight_sigma=0.7)
        self.tmpdir = tempfile.mkdtemp()
        self.p = os.path.join(self.tmpdir, "shape_hourly.parquet")
        self.sh1.save(self.p)
        self.sh2 = ShapeHourly.load(self.p)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_hyperparams_restored(self):
        """Test 4: scalar hyperparams match."""
        assert self.sh2.sigma == self.sh1.sigma
        assert self.sh2.halflife_days == self.sh1.halflife_days
        assert self.sh2.hydro_weight_sigma == self.sh1.hydro_weight_sigma

    def test_factors_by_year_restored(self):
        """Test 1: factors_by_year_ matches (allclose atol=1e-12)."""
        assert len(self.sh2.factors_by_year_) == len(self.sh1.factors_by_year_)
        for key, arr in self.sh1.factors_by_year_.items():
            assert key in self.sh2.factors_by_year_, f"Missing key {key}"
            np.testing.assert_allclose(
                self.sh2.factors_by_year_[key], arr, atol=1e-12, rtol=0
            )

    def test_trend_per_hour_restored(self):
        """Test 2: trend_per_hour_ matches (allclose atol=1e-12)."""
        assert len(self.sh2.trend_per_hour_) == len(self.sh1.trend_per_hour_)
        for key, arr in self.sh1.trend_per_hour_.items():
            assert key in self.sh2.trend_per_hour_, f"Missing key {key}"
            np.testing.assert_allclose(
                self.sh2.trend_per_hour_[key], arr, atol=1e-12, rtol=0
            )

    def test_f_W_seasonal_restored(self):
        """Test 3: f_W_seasonal_ matches (exact float comparison)."""
        assert self.sh2.f_W_seasonal_ == self.sh1.f_W_seasonal_

    def test_climatological_fill_restored(self):
        """Test 5: _climatological_fill is restored as pd.Series."""
        assert self.sh2._climatological_fill is not None
        assert list(self.sh2._climatological_fill.index) == [1, 2, 3]
        np.testing.assert_allclose(
            self.sh2._climatological_fill.values,
            self.sh1._climatological_fill.values,
            atol=1e-12, rtol=0
        )

    def test_global_factors_reconstructed(self):
        """Test 8: global_factors_ is reconstructed, not persisted, and equals pre-save value."""
        assert self.sh2.global_factors_ is not None
        np.testing.assert_allclose(
            self.sh2.global_factors_,
            self.sh1.global_factors_,
            atol=1e-12, rtol=0
        )

    def test_factors_still_restored(self):
        """Existing behavior: factors_ must still roundtrip (no regression)."""
        assert len(self.sh2.factors_) == len(self.sh1.factors_)
        for key, arr in self.sh1.factors_.items():
            np.testing.assert_allclose(
                self.sh2.factors_[key], arr, atol=1e-12, rtol=0
            )

    def test_f_W_still_restored(self):
        """Existing behavior: f_W_ must still roundtrip."""
        assert self.sh2.f_W_ == self.sh1.f_W_


class TestLoadClimatologicalFillNoneRoundtrip:
    """Test 5 (None branch): _climatological_fill is None after roundtrip if not set."""

    def test_none_roundtrip(self):
        sh1 = ShapeHourly()
        sh1.factors_[("Hiver", "Ouvrable")] = np.ones(24)
        sh1.n_obs_[("Hiver", "Ouvrable")] = 100
        sh1.f_W_["Ouvrable"] = 1.0
        assert sh1._climatological_fill is None
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "shape_hourly.parquet")
            sh1.save(p)
            sh2 = ShapeHourly.load(p)
            assert sh2._climatological_fill is None


class TestLoadLegacyCompat:
    """Test 7: loading without meta sidecar emits warning and uses defaults."""

    def test_legacy_load_no_crash(self, caplog):
        sh = _minimal_fitted_sh()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "shape_hourly.parquet")
            sh.save(p)
            # Simulate legacy: delete the meta sidecar
            os.remove(os.path.join(d, "shape_hourly.meta.parquet"))
            with caplog.at_level(logging.WARNING):
                sh3 = ShapeHourly.load(p)
            assert sh3 is not None

    def test_legacy_load_uses_defaults(self, caplog):
        sh = _minimal_fitted_sh(sigma=0.3, halflife_days=90.0, hydro_weight_sigma=0.7)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "shape_hourly.parquet")
            sh.save(p)
            os.remove(os.path.join(d, "shape_hourly.meta.parquet"))
            with caplog.at_level(logging.WARNING):
                sh3 = ShapeHourly.load(p)
            # Scalar hyperparams revert to constructor defaults
            assert sh3.sigma == 0.5
            assert sh3.halflife_days == 180.0
            assert sh3.hydro_weight_sigma == 0.25
            assert sh3.factors_by_year_ == {}
            assert sh3.trend_per_hour_ == {}
            assert sh3._climatological_fill is None

    def test_legacy_load_emits_exactly_one_warning(self, caplog):
        sh = _minimal_fitted_sh()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "shape_hourly.parquet")
            sh.save(p)
            os.remove(os.path.join(d, "shape_hourly.meta.parquet"))
            with caplog.at_level(logging.WARNING):
                ShapeHourly.load(p)
            # Count warnings containing "legacy" or "sidecar"
            warnings = [r for r in caplog.records
                        if r.levelno == logging.WARNING
                        and ("legacy" in r.message.lower() or "sidecar" in r.message.lower())]
            assert len(warnings) == 1, f"Expected exactly 1 legacy warning, got: {[r.message for r in warnings]}"

    def test_legacy_load_factors_still_populated(self):
        sh = _minimal_fitted_sh()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "shape_hourly.parquet")
            sh.save(p)
            os.remove(os.path.join(d, "shape_hourly.meta.parquet"))
            sh3 = ShapeHourly.load(p)
            # factors_ and f_W_ must still be populated
            assert len(sh3.factors_) > 0
            assert len(sh3.f_W_) > 0


class TestLoadSignatureUnchanged:
    """Verify load() signature is unchanged (only `path` parameter)."""

    def test_load_signature(self):
        import inspect
        sig = inspect.signature(ShapeHourly.load)
        params = list(sig.parameters)
        assert params == ["path"], f"Expected ['path'], got {params}"


class TestDoubleRoundtrip:
    """fit → save → load → save → load must yield identical results."""

    def test_double_roundtrip(self):
        sh1 = _minimal_fitted_sh(sigma=0.3, halflife_days=90.0, hydro_weight_sigma=0.7)
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "round1.parquet")
            p2 = os.path.join(d, "round2.parquet")
            sh1.save(p1)
            sh2 = ShapeHourly.load(p1)
            sh2.save(p2)
            sh3 = ShapeHourly.load(p2)

            # Compare sh1 vs sh3 (double roundtrip)
            assert sh3.sigma == sh1.sigma
            assert sh3.halflife_days == sh1.halflife_days
            assert sh3.hydro_weight_sigma == sh1.hydro_weight_sigma
            for key, arr in sh1.factors_by_year_.items():
                np.testing.assert_allclose(sh3.factors_by_year_[key], arr, atol=1e-12, rtol=0)
            for key, arr in sh1.trend_per_hour_.items():
                np.testing.assert_allclose(sh3.trend_per_hour_[key], arr, atol=1e-12, rtol=0)
            assert sh3.f_W_seasonal_ == sh1.f_W_seasonal_


# ===========================================================================
# Plan 05B-03 Task 1: Feature flag constructor arg + env-default + freeze-at-init
# ===========================================================================

class TestFlagEnvVarConstant:
    """_FLAG_ENV_VAR constant is exported and has the expected value."""

    def test_flag_env_var_constant_value(self):
        from pfc_shaping.lt.model.shape_hourly import _FLAG_ENV_VAR
        assert _FLAG_ENV_VAR == "PFC_LT_USE_SEASONAL_HOURLY_SHAPE"


class TestFlagSignature:
    """use_seasonal_hourly kwarg is in ShapeHourly.__init__ with default None (additive)."""

    def test_kwarg_present_in_signature(self):
        sig = inspect.signature(ShapeHourly.__init__)
        assert "use_seasonal_hourly" in sig.parameters, (
            "use_seasonal_hourly must be a kwarg of ShapeHourly.__init__"
        )

    def test_kwarg_default_is_none(self):
        sig = inspect.signature(ShapeHourly.__init__)
        param = sig.parameters["use_seasonal_hourly"]
        assert param.default is None, (
            f"use_seasonal_hourly default must be None, got {param.default!r}"
        )


class TestFlagEnvDefault:
    """Flag reads env var when constructor arg is None."""

    def test_env_unset_flag_is_false(self, monkeypatch):
        monkeypatch.delenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", raising=False)
        sh = ShapeHourly()
        assert sh._use_seasonal_hourly is False

    def test_env_one_flag_is_true(self, monkeypatch):
        monkeypatch.setenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", "1")
        sh = ShapeHourly()
        assert sh._use_seasonal_hourly is True

    def test_env_zero_flag_is_false(self, monkeypatch):
        monkeypatch.setenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", "0")
        sh = ShapeHourly()
        assert sh._use_seasonal_hourly is False

    def test_env_invalid_flag_is_false_with_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", "yes")
        with caplog.at_level(logging.WARNING):
            sh = ShapeHourly()
        assert sh._use_seasonal_hourly is False
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING
                    and "PFC_LT_USE_SEASONAL_HOURLY_SHAPE" in r.message]
        assert len(warnings) >= 1, "Expected a warning for invalid env var value"


class TestFlagConstructorWins:
    """Constructor kwarg wins over env var (D-06)."""

    def test_constructor_true_wins_over_env_zero(self, monkeypatch):
        """Constructor True beats env='0'."""
        monkeypatch.setenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", "0")
        sh = ShapeHourly(use_seasonal_hourly=True)
        assert sh._use_seasonal_hourly is True

    def test_constructor_false_wins_over_env_one(self, monkeypatch):
        """Constructor False beats env='1'."""
        monkeypatch.setenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", "1")
        sh = ShapeHourly(use_seasonal_hourly=False)
        assert sh._use_seasonal_hourly is False

    def test_constructor_true_without_env(self, monkeypatch):
        monkeypatch.delenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", raising=False)
        sh = ShapeHourly(use_seasonal_hourly=True)
        assert sh._use_seasonal_hourly is True

    def test_constructor_false_without_env(self, monkeypatch):
        monkeypatch.delenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", raising=False)
        sh = ShapeHourly(use_seasonal_hourly=False)
        assert sh._use_seasonal_hourly is False


class TestFlagFreezeAtInit:
    """Mutating os.environ after construction does NOT change _use_seasonal_hourly (D-06)."""

    def test_env_mutation_after_init_has_no_effect_true_to_false(self, monkeypatch):
        monkeypatch.setenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", "1")
        sh = ShapeHourly()
        assert sh._use_seasonal_hourly is True
        # Mutate env AFTER construction
        monkeypatch.setenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", "0")
        # Flag must NOT change
        assert sh._use_seasonal_hourly is True

    def test_env_mutation_after_init_has_no_effect_false_to_true(self, monkeypatch):
        monkeypatch.setenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", "0")
        sh = ShapeHourly()
        assert sh._use_seasonal_hourly is False
        # Mutate env AFTER construction
        monkeypatch.setenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", "1")
        # Flag must NOT change
        assert sh._use_seasonal_hourly is False

    def test_env_delete_after_init_has_no_effect(self, monkeypatch):
        monkeypatch.setenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", "1")
        sh = ShapeHourly()
        assert sh._use_seasonal_hourly is True
        monkeypatch.delenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", raising=False)
        assert sh._use_seasonal_hourly is True

    def test_new_instance_reads_updated_env(self, monkeypatch):
        """A new ShapeHourly() call DOES read updated env — only the old instance is frozen."""
        monkeypatch.setenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", "1")
        sh1 = ShapeHourly()
        assert sh1._use_seasonal_hourly is True
        monkeypatch.setenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", "0")
        sh2 = ShapeHourly()
        assert sh2._use_seasonal_hourly is False
        # sh1 still frozen
        assert sh1._use_seasonal_hourly is True


class TestFlagAttributeIsBool:
    """_use_seasonal_hourly attribute must be a bool (not a str, int, or None)."""

    def test_attribute_is_bool_true(self, monkeypatch):
        monkeypatch.delenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", raising=False)
        sh = ShapeHourly(use_seasonal_hourly=True)
        assert type(sh._use_seasonal_hourly) is bool
        assert sh._use_seasonal_hourly is True

    def test_attribute_is_bool_false(self, monkeypatch):
        monkeypatch.delenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", raising=False)
        sh = ShapeHourly(use_seasonal_hourly=False)
        assert type(sh._use_seasonal_hourly) is bool
        assert sh._use_seasonal_hourly is False

    def test_attribute_is_bool_from_env(self, monkeypatch):
        monkeypatch.setenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", "1")
        sh = ShapeHourly()
        assert type(sh._use_seasonal_hourly) is bool


# ===========================================================================
# Plan 05B-03 Task 2: Flag persisted in meta sidecar, restored on load (parquet wins)
# ===========================================================================

class TestFlagPersistenceInSidecar:
    """Flag is written to meta sidecar hyperparams JSON cell."""

    def test_flag_true_persisted(self, monkeypatch):
        monkeypatch.delenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", raising=False)
        sh = ShapeHourly(use_seasonal_hourly=True)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "shape_hourly.parquet")
            sh.save(p)
            meta = pd.read_parquet(os.path.join(d, "shape_hourly.meta.parquet"))
            hp_row = meta[meta["attr"] == "hyperparams"].iloc[0]
            hp = json.loads(hp_row["value"])
            assert "use_seasonal_hourly" in hp, f"Expected 'use_seasonal_hourly' in hyperparams, got {hp}"
            assert hp["use_seasonal_hourly"] is True

    def test_flag_false_persisted(self, monkeypatch):
        monkeypatch.delenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", raising=False)
        sh = ShapeHourly(use_seasonal_hourly=False)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "shape_hourly.parquet")
            sh.save(p)
            meta = pd.read_parquet(os.path.join(d, "shape_hourly.meta.parquet"))
            hp_row = meta[meta["attr"] == "hyperparams"].iloc[0]
            hp = json.loads(hp_row["value"])
            assert "use_seasonal_hourly" in hp
            assert hp["use_seasonal_hourly"] is False

    def test_hyperparams_json_has_all_keys(self, monkeypatch):
        """After 05B-03, hyperparams must include sigma, halflife_days, hydro_weight_sigma, use_seasonal_hourly.
        Updated by Plan 05C-01 (D-A3-3 / RESEARCH Pitfall 4): also includes
        hydro_weight_sigma_off/_on/_resolved.
        Updated by Plan 05C-03 (D-A3-3 / RESEARCH Pitfall 4 second wave): also includes
        sigma_off/_on/_resolved. Total schema = 10 keys.
        """
        monkeypatch.delenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", raising=False)
        sh = ShapeHourly(sigma=0.3, halflife_days=90.0, hydro_weight_sigma=0.7, use_seasonal_hourly=True)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "shape_hourly.parquet")
            sh.save(p)
            meta = pd.read_parquet(os.path.join(d, "shape_hourly.meta.parquet"))
            hp = json.loads(meta[meta["attr"] == "hyperparams"].iloc[0]["value"])
            # After Plan 05C-03: 10 keys (full 5bis-B schema)
            assert set(hp.keys()) == {
                "halflife_days",
                "hydro_weight_sigma",
                "hydro_weight_sigma_off",
                "hydro_weight_sigma_on",
                "hydro_weight_sigma_resolved",
                "sigma",
                "sigma_off",
                "sigma_on",
                "sigma_resolved",
                "use_seasonal_hourly",
            }


class TestFlagRestoredOnLoad:
    """load() restores _use_seasonal_hourly from sidecar; parquet wins over env (D-07)."""

    def test_parquet_wins_over_env_true_over_zero(self, monkeypatch):
        """Saved flag=True restored even when env='0'."""
        monkeypatch.delenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", raising=False)
        sh = ShapeHourly(use_seasonal_hourly=True)
        sh.factors_[("Hiver", "Ouvrable")] = np.ones(24)
        sh.n_obs_[("Hiver", "Ouvrable")] = 100
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "shape_hourly.parquet")
            sh.save(p)
            # Env says flag=False, but parquet says flag=True
            monkeypatch.setenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", "0")
            sh2 = ShapeHourly.load(p)
            assert sh2._use_seasonal_hourly is True, (
                f"Parquet value (True) must win over env '0', got {sh2._use_seasonal_hourly}"
            )

    def test_parquet_wins_over_env_false_over_one(self, monkeypatch):
        """Saved flag=False restored even when env='1'."""
        monkeypatch.delenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", raising=False)
        sh = ShapeHourly(use_seasonal_hourly=False)
        sh.factors_[("Hiver", "Ouvrable")] = np.ones(24)
        sh.n_obs_[("Hiver", "Ouvrable")] = 100
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "shape_hourly.parquet")
            sh.save(p)
            # Env says flag=True, but parquet says flag=False
            monkeypatch.setenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", "1")
            sh2 = ShapeHourly.load(p)
            assert sh2._use_seasonal_hourly is False, (
                f"Parquet value (False) must win over env '1', got {sh2._use_seasonal_hourly}"
            )

    def test_load_without_flag_in_hyperparams_uses_env(self, monkeypatch):
        """If meta sidecar exists but hyperparams has no use_seasonal_hourly key
        (legacy 05B-02 sidecar), fall back to _resolve_flag(None) — no crash."""
        monkeypatch.setenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", "0")
        sh = ShapeHourly(use_seasonal_hourly=False)
        sh.factors_[("Hiver", "Ouvrable")] = np.ones(24)
        sh.n_obs_[("Hiver", "Ouvrable")] = 100
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "shape_hourly.parquet")
            sh.save(p)
            # Patch sidecar to remove use_seasonal_hourly from hyperparams
            meta_p = os.path.join(d, "shape_hourly.meta.parquet")
            meta_df = pd.read_parquet(meta_p)
            hp_mask = meta_df["attr"] == "hyperparams"
            hp = json.loads(meta_df.loc[hp_mask, "value"].iloc[0])
            hp.pop("use_seasonal_hourly", None)
            meta_df.loc[hp_mask, "value"] = json.dumps(hp, sort_keys=True)
            meta_df.to_parquet(meta_p, index=False)
            # Load must not crash
            sh2 = ShapeHourly.load(p)
            # Must use _resolve_flag(None) = env '0' → False
            assert sh2._use_seasonal_hourly is False

    def test_load_without_sidecar_uses_env(self, monkeypatch):
        """Legacy parquet (no sidecar at all) falls back to env/default for _use_seasonal_hourly."""
        monkeypatch.setenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", "1")
        sh = ShapeHourly(use_seasonal_hourly=False)
        sh.factors_[("Hiver", "Ouvrable")] = np.ones(24)
        sh.n_obs_[("Hiver", "Ouvrable")] = 100
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "shape_hourly.parquet")
            sh.save(p)
            # Delete sidecar to simulate legacy
            os.remove(os.path.join(d, "shape_hourly.meta.parquet"))
            sh2 = ShapeHourly.load(p)
            # No sidecar → cls() called → env='1' → True
            assert sh2._use_seasonal_hourly is True


class TestFlagNoOpContract:
    """In Phase 5bis-A, flag ON ≡ flag OFF numerically (no behavior gated)."""

    def test_flag_on_off_same_attribute_set(self, monkeypatch):
        """ShapeHourly with flag=True and flag=False have the same attribute structure."""
        monkeypatch.delenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", raising=False)
        sh_on = ShapeHourly(use_seasonal_hourly=True)
        sh_off = ShapeHourly(use_seasonal_hourly=False)
        # Both should have _use_seasonal_hourly as a bool
        assert type(sh_on._use_seasonal_hourly) is bool
        assert type(sh_off._use_seasonal_hourly) is bool
        # Attribute sets (excluding _use_seasonal_hourly) must be identical
        attrs_on = {k: v for k, v in vars(sh_on).items() if k != "_use_seasonal_hourly"}
        attrs_off = {k: v for k, v in vars(sh_off).items() if k != "_use_seasonal_hourly"}
        assert set(attrs_on.keys()) == set(attrs_off.keys())


# ============================================================================
# Plan 05B-04 Task 1: factors_3d_ read-only Mapping view (SHP-01 literal)
# ============================================================================

class TestFactors3DViewExists:
    """factors_3d_ property must be importable and exist on ShapeHourly."""

    def test_class_has_factors_3d_(self):
        """ShapeHourly has a factors_3d_ attribute (property)."""
        assert hasattr(ShapeHourly, "factors_3d_"), "ShapeHourly must have factors_3d_ property"

    def test_factors_3d_is_property(self):
        """factors_3d_ is a Python property on the class."""
        assert isinstance(ShapeHourly.__dict__.get("factors_3d_"), property), (
            "factors_3d_ must be a @property"
        )

    def test_private_view_class_importable(self):
        """_Factors3DView class is importable from shape_hourly module."""
        from pfc_shaping.lt.model.shape_hourly import _Factors3DView  # noqa: F401


class TestFactors3DViewEmptyBeforeFit:
    """factors_3d_ is accessible before fit(), returns empty mapping (len==0)."""

    def test_accessible_before_fit(self):
        """factors_3d_ does not raise before fit()."""
        sh = ShapeHourly()
        v = sh.factors_3d_
        assert v is not None

    def test_len_zero_before_fit(self):
        """len(factors_3d_) == 0 when factors_ is empty."""
        sh = ShapeHourly()
        assert len(sh.factors_3d_) == 0

    def test_is_mapping_before_fit(self):
        """factors_3d_ is a collections.abc.Mapping instance."""
        from collections.abc import Mapping
        sh = ShapeHourly()
        assert isinstance(sh.factors_3d_, Mapping)


class TestFactors3DViewGetItem:
    """factors_3d_[(s, tj, h)] returns exact float from underlying factors_."""

    def setup_method(self):
        self.sh = ShapeHourly()
        arr = np.linspace(0.8, 1.2, 24)
        arr = arr / arr.mean()
        self.arr = arr
        self.sh.factors_[("Hiver", "Ouvrable")] = arr

    def test_getitem_returns_correct_float(self):
        """factors_3d_[('Hiver', 'Ouvrable', 12)] == factors_[('Hiver','Ouvrable')][12]."""
        v = self.sh.factors_3d_
        expected = float(self.arr[12])
        assert v[("Hiver", "Ouvrable", 12)] == expected, (
            f"Expected {expected}, got {v[('Hiver', 'Ouvrable', 12)]}"
        )

    def test_getitem_first_hour(self):
        """Hour 0 indexing works."""
        v = self.sh.factors_3d_
        assert v[("Hiver", "Ouvrable", 0)] == float(self.arr[0])

    def test_getitem_last_valid_hour(self):
        """Hour 23 (last valid) works."""
        v = self.sh.factors_3d_
        assert v[("Hiver", "Ouvrable", 23)] == float(self.arr[23])

    def test_keyerror_missing_cell(self):
        """KeyError for (saison, type_jour) not in factors_."""
        v = self.sh.factors_3d_
        with pytest.raises(KeyError):
            _ = v[("nope", "nope", 0)]

    def test_keyerror_hour_out_of_range_high(self):
        """KeyError for hour == 24 (must be in [0, 24))."""
        v = self.sh.factors_3d_
        with pytest.raises(KeyError):
            _ = v[("Hiver", "Ouvrable", 24)]

    def test_keyerror_hour_negative(self):
        """KeyError for negative hour."""
        v = self.sh.factors_3d_
        with pytest.raises(KeyError):
            _ = v[("Hiver", "Ouvrable", -1)]

    def test_keyerror_wrong_key_type(self):
        """KeyError for non-3-tuple key (e.g. 2-tuple)."""
        v = self.sh.factors_3d_
        with pytest.raises(KeyError):
            _ = v[("Hiver", "Ouvrable")]


class TestFactors3DViewLen:
    """len(factors_3d_) == len(factors_) * 24."""

    def test_len_single_cell(self):
        """One cell → len == 24."""
        sh = ShapeHourly()
        arr = np.ones(24)
        sh.factors_[("Hiver", "Ouvrable")] = arr
        assert len(sh.factors_3d_) == 24

    def test_len_two_cells(self):
        """Two cells → len == 48."""
        sh = ShapeHourly()
        arr = np.ones(24)
        sh.factors_[("Hiver", "Ouvrable")] = arr
        sh.factors_[("Ete", "Samedi")] = arr
        assert len(sh.factors_3d_) == 48

    def test_len_proportional(self):
        """len(factors_3d_) == len(factors_) * 24 always."""
        sh = ShapeHourly()
        arr = np.ones(24)
        for s in ["Hiver", "Ete", "Printemps"]:
            sh.factors_[(s, "Ouvrable")] = arr
        assert len(sh.factors_3d_) == len(sh.factors_) * 24


class TestFactors3DViewIteration:
    """iter(factors_3d_) yields all (s, tj, h) 3-tuples."""

    def test_iter_yields_3tuples(self):
        """All items from iter() are 3-tuples (str, str, int)."""
        sh = ShapeHourly()
        arr = np.ones(24)
        sh.factors_[("Hiver", "Ouvrable")] = arr
        v = sh.factors_3d_
        keys = list(iter(v))
        assert all(isinstance(k, tuple) and len(k) == 3 for k in keys)

    def test_iter_correct_count(self):
        """Exactly len(factors_) * 24 items yielded."""
        sh = ShapeHourly()
        arr = np.ones(24)
        sh.factors_[("Hiver", "Ouvrable")] = arr
        sh.factors_[("Ete", "Samedi")] = arr
        keys = list(sh.factors_3d_)
        assert len(keys) == 2 * 24

    def test_iter_covers_all_hours(self):
        """For each (s, tj) in factors_, all hours 0..23 appear."""
        sh = ShapeHourly()
        arr = np.ones(24)
        sh.factors_[("Hiver", "Ouvrable")] = arr
        keys = list(sh.factors_3d_)
        hours = [k[2] for k in keys if k[0] == "Hiver" and k[1] == "Ouvrable"]
        assert sorted(hours) == list(range(24))


class TestFactors3DViewReadOnly:
    """factors_3d_ raises TypeError on write attempt (read-only)."""

    def test_setitem_raises_typeerror(self):
        """Assigning to factors_3d_[(s, tj, h)] raises TypeError."""
        sh = ShapeHourly()
        arr = np.ones(24)
        sh.factors_[("Hiver", "Ouvrable")] = arr
        v = sh.factors_3d_
        with pytest.raises(TypeError):
            v[("Hiver", "Ouvrable", 12)] = 99.0

    def test_underlying_factors_unchanged_after_set_attempt(self):
        """After TypeError on set, factors_ is unchanged."""
        sh = ShapeHourly()
        arr = np.ones(24)
        sh.factors_[("Hiver", "Ouvrable")] = arr.copy()
        v = sh.factors_3d_
        try:
            v[("Hiver", "Ouvrable", 12)] = 99.0
        except TypeError:
            pass
        assert float(sh.factors_[("Hiver", "Ouvrable")][12]) == 1.0


class TestFactors3DViewLiveness:
    """factors_3d_ is a live facade — reflects mutations of factors_ dict."""

    def test_new_cell_visible_after_first_access(self):
        """Cell added AFTER first access of factors_3d_ is visible."""
        sh = ShapeHourly()
        arr = np.ones(24)
        sh.factors_[("Hiver", "Ouvrable")] = arr
        v = sh.factors_3d_  # first access
        assert len(v) == 24
        # Add a second cell AFTER getting the view
        sh.factors_[("Ete", "Samedi")] = arr * 2
        assert len(v) == 48  # live view reflects the addition
        assert v[("Ete", "Samedi", 5)] == float(arr[5] * 2)

    def test_view_reflects_mutation_of_existing_cell(self):
        """Mutating an existing array in factors_ is reflected in the view."""
        sh = ShapeHourly()
        arr = np.ones(24)
        sh.factors_[("Hiver", "Ouvrable")] = arr
        v = sh.factors_3d_
        # Mutate the underlying array
        sh.factors_[("Hiver", "Ouvrable")][12] = 9.0
        assert v[("Hiver", "Ouvrable", 12)] == 9.0


class TestFactors3DViewContains:
    """'in' operator works without raising on invalid keys."""

    def test_valid_key_in_view(self):
        """(s, tj, h) in view returns True for existing key."""
        sh = ShapeHourly()
        sh.factors_[("Hiver", "Ouvrable")] = np.ones(24)
        assert ("Hiver", "Ouvrable", 12) in sh.factors_3d_

    def test_missing_cell_not_in_view(self):
        """(s, tj, h) for missing (s, tj) returns False."""
        sh = ShapeHourly()
        sh.factors_[("Hiver", "Ouvrable")] = np.ones(24)
        assert ("Ete", "Samedi", 0) not in sh.factors_3d_

    def test_hour_out_of_range_not_in_view(self):
        """(s, tj, 24) returns False (not in view)."""
        sh = ShapeHourly()
        sh.factors_[("Hiver", "Ouvrable")] = np.ones(24)
        assert ("Hiver", "Ouvrable", 24) not in sh.factors_3d_


# ============================================================================
# Plan 05B-04 Task 2: assembler _sh_apply_accepts_outages + capability check
# ============================================================================

class TestShApplyAcceptsOutagesHelper:
    """_sh_apply_accepts_outages(cls) returns bool based on signature."""

    def test_helper_importable(self):
        """_sh_apply_accepts_outages is importable from assembler module."""
        from pfc_shaping.lt.model.assembler import _sh_apply_accepts_outages  # noqa: F401

    def test_returns_false_for_shape_hourly(self):
        """ShapeHourly.apply does NOT have outages_forecast → returns False."""
        from pfc_shaping.lt.model.assembler import _sh_apply_accepts_outages
        assert _sh_apply_accepts_outages(ShapeHourly) is False

    def test_returns_true_for_shape_hourly_mlp(self):
        """ShapeHourlyMLP.apply HAS outages_forecast → returns True."""
        from pfc_shaping.lt.model.assembler import _sh_apply_accepts_outages
        from pfc_shaping.lt.model.shape_hourly_mlp import ShapeHourlyMLP
        assert _sh_apply_accepts_outages(ShapeHourlyMLP) is True

    def test_raises_typeerror_for_stub_without_reference_date(self):
        """Class whose apply() lacks reference_date raises TypeError (minimum contract)."""
        from pfc_shaping.lt.model.assembler import _sh_apply_accepts_outages

        class LegacyStub:
            def apply(self, timestamps, calendar_df):
                pass

        with pytest.raises(TypeError, match="reference_date"):
            _sh_apply_accepts_outages(LegacyStub)


class TestAssemblerCapabilityCache:
    """PFCAssembler caches _sh_accepts_outages at __init__ time."""

    def _make_assembler(self, sh=None):
        """Create a minimal PFCAssembler without calibrator/wv/etc."""
        from pfc_shaping.lt.model.assembler import PFCAssembler
        from pfc_shaping.lt.model.shape_intraday import ShapeIntraday
        if sh is None:
            sh = ShapeHourly()
        si = ShapeIntraday()
        return PFCAssembler(shape_hourly=sh, shape_intraday=si)

    def test_instance_has_sh_accepts_outages_attr(self):
        """PFCAssembler instance has _sh_accepts_outages attribute after __init__."""
        asm = self._make_assembler()
        assert hasattr(asm, "_sh_accepts_outages"), (
            "PFCAssembler must cache _sh_accepts_outages at __init__"
        )

    def test_sh_accepts_outages_false_for_shape_hourly(self):
        """_sh_accepts_outages is False when sh is ShapeHourly."""
        asm = self._make_assembler(sh=ShapeHourly())
        assert asm._sh_accepts_outages is False

    def test_sh_accepts_outages_true_for_shape_hourly_mlp(self):
        """_sh_accepts_outages is True when sh is ShapeHourlyMLP."""
        from pfc_shaping.lt.model.assembler import PFCAssembler
        from pfc_shaping.lt.model.shape_hourly_mlp import ShapeHourlyMLP
        from pfc_shaping.lt.model.shape_intraday import ShapeIntraday
        sh = ShapeHourlyMLP()
        si = ShapeIntraday()
        asm = PFCAssembler(shape_hourly=sh, shape_intraday=si)
        assert asm._sh_accepts_outages is True

    def test_sh_accepts_outages_is_bool(self):
        """_sh_accepts_outages is a Python bool (not numpy.bool_ etc.)."""
        asm = self._make_assembler()
        assert type(asm._sh_accepts_outages) is bool


class TestAssemblerNoTryExceptTypeError:
    """assembler.py no longer has try/except TypeError around self.sh.apply."""

    def test_no_try_except_around_sh_apply(self):
        """The try/except TypeError pattern around self.sh.apply is gone from assembler.py."""
        import ast
        import pathlib
        src = pathlib.Path("pfc_shaping/lt/model/assembler.py").read_text()
        tree = ast.parse(src)

        # Walk AST and collect all ExceptHandler nodes that catch TypeError
        type_error_handlers = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                exc = node.type
                if exc is None:
                    continue
                names = []
                if isinstance(exc, ast.Name):
                    names = [exc.id]
                elif isinstance(exc, ast.Tuple):
                    names = [e.id for e in exc.elts if isinstance(e, ast.Name)]
                if "TypeError" in names:
                    # Check if any name in the handler body references sh.apply
                    body_src = ast.unparse(node).lower()
                    if "sh.apply" in body_src or "sh_apply" in body_src:
                        type_error_handlers.append(node)

        assert len(type_error_handlers) == 0, (
            f"Found {len(type_error_handlers)} try/except TypeError block(s) around "
            f"sh.apply — D-13 requires replacing with explicit signature check"
        )


class TestAssemblerOperatorLog:
    """PFCAssembler emits exactly one operator log line at __init__ naming the sh class."""

    def _make_assembler_and_capture_log(self, sh=None, caplog=None):
        from pfc_shaping.lt.model.assembler import PFCAssembler
        from pfc_shaping.lt.model.shape_intraday import ShapeIntraday
        if sh is None:
            sh = ShapeHourly()
        si = ShapeIntraday()
        with caplog.at_level(logging.INFO, logger="pfc_shaping.lt.model.assembler"):
            asm = PFCAssembler(shape_hourly=sh, shape_intraday=si)
        return asm

    def test_operator_log_emitted_at_init(self, caplog):
        """logger.info line mentioning 'Detected sh=' is emitted during __init__."""
        self._make_assembler_and_capture_log(caplog=caplog)
        messages = [r.message for r in caplog.records]
        matching = [m for m in messages if "Detected sh=" in m]
        assert len(matching) >= 1, (
            f"Expected at least one log line with 'Detected sh=', got: {messages}"
        )

    def test_operator_log_names_shape_hourly(self, caplog):
        """Log line for ShapeHourly contains 'ShapeHourly' and 'skipped'."""
        self._make_assembler_and_capture_log(sh=ShapeHourly(), caplog=caplog)
        messages = [r.message for r in caplog.records]
        matching = [m for m in messages if "Detected sh=" in m]
        assert any("ShapeHourly" in m and "skipped" in m for m in matching), (
            f"Expected log containing 'ShapeHourly' and 'skipped', got: {matching}"
        )

    def test_operator_log_names_shape_hourly_mlp(self, caplog):
        """Log line for ShapeHourlyMLP contains 'ShapeHourlyMLP' and 'passed'."""
        from pfc_shaping.lt.model.assembler import PFCAssembler
        from pfc_shaping.lt.model.shape_hourly_mlp import ShapeHourlyMLP
        from pfc_shaping.lt.model.shape_intraday import ShapeIntraday
        sh = ShapeHourlyMLP()
        si = ShapeIntraday()
        with caplog.at_level(logging.INFO, logger="pfc_shaping.lt.model.assembler"):
            PFCAssembler(shape_hourly=sh, shape_intraday=si)
        messages = [r.message for r in caplog.records]
        matching = [m for m in messages if "Detected sh=" in m]
        assert any("ShapeHourlyMLP" in m and "passed" in m for m in matching), (
            f"Expected log containing 'ShapeHourlyMLP' and 'passed', got: {matching}"
        )


# ============================================================================
# Plan 05B-05 Task 3 — Seven standalone tests D-14..D-19 + Codex review §8
# These are the definitive proof tests for Phase 5bis-A no-op contract.
# Context: .planning/phases/05B-shape-hourly-infrastructure-flag-no-op-refactor/05B-CONTEXT.md
# ============================================================================

# ---------------------------------------------------------------------------
# Fixtures directory path (absolute, via this file's location)
# ---------------------------------------------------------------------------
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# ---------------------------------------------------------------------------
# D-14 — test_factors_3d_view_consistency
# ---------------------------------------------------------------------------

def test_factors_3d_view_consistency():
    """D-14 (Plan 05B-05): factors_3d_[(s,tj,h)] == factors_[(s,tj)][h] over all cells x 24h.

    Fits on minimal synthetic data, walks all populated cells x hours and asserts
    exact float equality between the 3D view and the underlying array.
    Assignment into the view must raise TypeError (read-only contract).
    """
    sh = ShapeHourly()
    # Populate two cells with distinct arrays
    for (s, tj), base in [
        (("Hiver", "Ouvrable"), np.linspace(0.8, 1.2, 24)),
        (("Ete", "Samedi"), np.linspace(0.9, 1.1, 24)),
    ]:
        arr = base / base.mean()
        sh.factors_[(s, tj)] = arr

    view = sh.factors_3d_

    # Assert equality over ALL populated cells × 24 hours
    for (s, tj), arr in sh.factors_.items():
        for h in range(24):
            expected = float(arr[h])
            got = view[(s, tj, h)]
            assert got == expected, (
                f"Mismatch at ({s},{tj},{h}): view={got} != factors_={expected}"
            )

    # Assert mutation raises TypeError (read-only)
    with pytest.raises(TypeError):
        view[("Hiver", "Ouvrable", 12)] = 99.0


# ---------------------------------------------------------------------------
# D-15 — test_save_load_full_roundtrip
# ---------------------------------------------------------------------------

def test_save_load_full_roundtrip(tmp_path):
    """D-15 (Plan 05B-05): all 10 trained attributes roundtrip through save→load.

    Verifies: factors_, factors_by_year_, trend_per_hour_, f_W_seasonal_, f_W_,
    _climatological_fill, sigma, halflife_days, hydro_weight_sigma,
    _use_seasonal_hourly, and global_factors_ (reconstructed, not persisted).
    """
    sh1 = _minimal_fitted_sh(sigma=0.3, halflife_days=90.0, hydro_weight_sigma=0.7)
    sh1._use_seasonal_hourly = True  # exercise non-default flag
    sh1.global_factors_ = sh1._compute_global_fallback()

    p = tmp_path / "shape_hourly.parquet"
    sh1.save(str(p))
    sh2 = ShapeHourly.load(str(p))

    # ── factors_ ──
    assert set(sh2.factors_.keys()) == set(sh1.factors_.keys())
    for key in sh1.factors_:
        np.testing.assert_allclose(sh2.factors_[key], sh1.factors_[key], atol=1e-12, rtol=0)

    # ── factors_by_year_ ──
    assert set(sh2.factors_by_year_.keys()) == set(sh1.factors_by_year_.keys())
    for key in sh1.factors_by_year_:
        np.testing.assert_allclose(
            sh2.factors_by_year_[key], sh1.factors_by_year_[key], atol=1e-12, rtol=0
        )

    # ── trend_per_hour_ ──
    assert set(sh2.trend_per_hour_.keys()) == set(sh1.trend_per_hour_.keys())
    for key in sh1.trend_per_hour_:
        np.testing.assert_allclose(
            sh2.trend_per_hour_[key], sh1.trend_per_hour_[key], atol=1e-12, rtol=0
        )

    # ── f_W_seasonal_ ──
    assert sh2.f_W_seasonal_ == sh1.f_W_seasonal_

    # ── f_W_ ──
    assert sh2.f_W_ == sh1.f_W_

    # ── _climatological_fill ──
    assert sh2._climatological_fill is not None
    np.testing.assert_allclose(
        sh2._climatological_fill.sort_index().values,
        sh1._climatological_fill.sort_index().values,
        atol=1e-12, rtol=0,
    )
    assert list(sh2._climatological_fill.sort_index().index) == list(sh1._climatological_fill.sort_index().index)

    # ── Scalar hyperparams ──
    assert sh2.sigma == sh1.sigma
    assert sh2.halflife_days == sh1.halflife_days
    assert sh2.hydro_weight_sigma == sh1.hydro_weight_sigma

    # ── _use_seasonal_hourly ──
    assert sh2._use_seasonal_hourly is True  # was set to True above

    # ── global_factors_ (reconstructed via _compute_global_fallback, not persisted) ──
    assert sh2.global_factors_ is not None
    np.testing.assert_allclose(sh2.global_factors_, sh1.global_factors_, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# D-16 — test_save_load_legacy_compat
# ---------------------------------------------------------------------------

def test_save_load_legacy_compat(tmp_path, caplog):
    """D-16 (Plan 05B-05): loading legacy parquets (no .meta.parquet sidecar) does not crash.

    Uses committed frozen fixtures from tests/fixtures/shape_hourly_legacy.parquet
    and tests/fixtures/f_W_legacy.parquet (produced by _generate_legacy_fixture.py,
    which writes NO .meta.parquet sidecar — that is the whole point).

    Asserts:
    - No exception on load
    - Exactly one logger.WARNING emitted (containing 'legacy' or 'sidecar')
    - sh.factors_ is non-empty
    - sh.factors_by_year_ == {} (legacy: metadata unavailable)
    - sh.sigma == 0.5 (constructor default, sidecar absent)
    """
    # Copy legacy fixtures to a temp dir with the expected filenames
    shutil.copy(str(_FIXTURES_DIR / "shape_hourly_legacy.parquet"),
                str(tmp_path / "shape_hourly.parquet"))
    shutil.copy(str(_FIXTURES_DIR / "f_W_legacy.parquet"),
                str(tmp_path / "f_W.parquet"))

    # Confirm no sidecar was copied (legacy contract)
    assert not (tmp_path / "shape_hourly.meta.parquet").exists()

    with caplog.at_level(logging.WARNING):
        sh = ShapeHourly.load(str(tmp_path / "shape_hourly.parquet"))

    # No exception
    assert sh is not None

    # Exactly one warning mentioning 'legacy' or 'sidecar'
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and ("legacy" in r.message.lower() or "sidecar" in r.message.lower())
    ]
    assert len(warnings) == 1, (
        f"Expected exactly 1 legacy warning, got {len(warnings)}: {[w.message for w in warnings]}"
    )

    # factors_ is populated from the legacy parquet
    assert len(sh.factors_) > 0, "factors_ must be non-empty after legacy load"

    # Legacy metadata unavailable — reverts to constructor defaults
    assert sh.factors_by_year_ == {}, "factors_by_year_ must be empty for legacy loads"
    assert sh.sigma == 0.5, f"sigma must be constructor default (0.5), got {sh.sigma}"


# ---------------------------------------------------------------------------
# D-17 — test_flag_freeze_at_init
# ---------------------------------------------------------------------------

def test_flag_freeze_at_init(monkeypatch):
    """D-17 (Plan 05B-05): post-construction env-var mutations do NOT change _use_seasonal_hourly.

    Four combinatorial sub-assertions (all directions of freeze: True→ try False, False→try True,
    env=1 then delenv, env=0 then set 1).
    """
    env_key = "PFC_LT_USE_SEASONAL_HOURLY_SHAPE"

    # --- Sub-assertion 1: constructor True wins; then env changed to "0" → still True ---
    monkeypatch.setenv(env_key, "0")
    sh1 = ShapeHourly(use_seasonal_hourly=True)
    assert sh1._use_seasonal_hourly is True
    monkeypatch.setenv(env_key, "1")
    assert sh1._use_seasonal_hourly is True  # frozen

    # --- Sub-assertion 2: env=None→default False; then env set to "1" → still False ---
    monkeypatch.delenv(env_key, raising=False)
    sh2 = ShapeHourly(use_seasonal_hourly=None)
    assert sh2._use_seasonal_hourly is False
    monkeypatch.setenv(env_key, "1")
    assert sh2._use_seasonal_hourly is False  # frozen at first read

    # --- Sub-assertion 3: env="1" at init → True; then delenv → still True ---
    monkeypatch.setenv(env_key, "1")
    sh3 = ShapeHourly()
    assert sh3._use_seasonal_hourly is True
    monkeypatch.delenv(env_key, raising=False)
    assert sh3._use_seasonal_hourly is True  # frozen

    # --- Sub-assertion 4: env="0" at init → False; then set env to "1" → still False ---
    monkeypatch.setenv(env_key, "0")
    sh4 = ShapeHourly()
    assert sh4._use_seasonal_hourly is False
    monkeypatch.setenv(env_key, "1")
    assert sh4._use_seasonal_hourly is False  # frozen


# ---------------------------------------------------------------------------
# D-18 — test_flag_persisted_in_parquet
# ---------------------------------------------------------------------------

def test_flag_persisted_in_parquet(tmp_path, monkeypatch):
    """D-18 (Plan 05B-05): parquet value wins over env at load time, both directions.

    Scenario A: fit with flag=True, save, then load with env='0' → must yield True.
    Scenario B: fit with flag=False, save, then load with env='1' → must yield False.
    """
    env_key = "PFC_LT_USE_SEASONAL_HOURLY_SHAPE"

    def _make_minimal_sh(flag: bool) -> ShapeHourly:
        sh = ShapeHourly(use_seasonal_hourly=flag)
        sh.factors_[("Hiver", "Ouvrable")] = np.ones(24)
        sh.n_obs_[("Hiver", "Ouvrable")] = 100
        sh.f_W_["Ouvrable"] = 1.0
        return sh

    # ── Scenario A: flag=True persisted, env='0' at load ──
    p_a = tmp_path / "sh_a.parquet"
    monkeypatch.delenv(env_key, raising=False)
    sh_a = _make_minimal_sh(flag=True)
    sh_a.save(str(p_a))

    monkeypatch.setenv(env_key, "0")
    sh_a2 = ShapeHourly.load(str(p_a))
    assert sh_a2._use_seasonal_hourly is True, (
        f"Parquet (True) must win over env '0', got {sh_a2._use_seasonal_hourly}"
    )

    # ── Scenario B: flag=False persisted, env='1' at load ──
    p_b = tmp_path / "sh_b.parquet"
    monkeypatch.delenv(env_key, raising=False)
    sh_b = _make_minimal_sh(flag=False)
    sh_b.save(str(p_b))

    monkeypatch.setenv(env_key, "1")
    sh_b2 = ShapeHourly.load(str(p_b))
    assert sh_b2._use_seasonal_hourly is False, (
        f"Parquet (False) must win over env '1', got {sh_b2._use_seasonal_hourly}"
    )


# ---------------------------------------------------------------------------
# D-19 — test_baseline_regression (parametrized: THE no-op proof)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", [False])
def test_baseline_regression(flag):
    """D-19 (Plan 05B-05 / updated Plan 05C-02): build_pfc(flag=False) equals frozen baseline.

    SCOPE UPDATE (Plan 05C-02, D-A2-3..D-A2-4):
    Phase 5bis-A originally tested BOTH flag=False and flag=True as a no-op proof (5bis-A
    was pure plumbing — the flag had zero numerical effect). Phase 5bis-B (Plan 05C-02)
    ships Lever 2, which introduces a genuine math change under flag=True: the f_H damping
    now splits level + anomaly so the duck curve survives at far horizon. As a result:

    - flag=False (this test): MUST still equal the pre-5bis-A baseline at atol=1e-12, rtol=0.
      The else-branch in assembler.py executes the legacy single-line damping UNCHANGED.
    - flag=True: intentionally produces DIFFERENT output (Lever 2 math change). Validated
      by test_f_H_amplitude_preserved_at_M30 (D-A4-6 / SC #3) and will be covered by
      test_flag_on_bowl_baseline (D-A4-9, Plan 05C-03) — a new frozen baseline for flag=ON.

    The bit-pour-bit contract (atol=1e-12, rtol=0) applies to flag=OFF only from this plan.
    [Rule 1 - Bug] Auto-fix: removed flag=True parametrization; it was correct for 5bis-A
    no-op proof but incorrect now that 5bis-B ships a math change. D-A2-3/D-A2-4 verouillé.

    Tolerance: atol=1e-12, rtol=0 is the default contract (REVIEWS.md consensus §1).
    Fallback policy: if cross-version pandas/pyarrow patch-level drift breaks this in CI,
    the test MAY relax to atol=1e-10 ONLY with an inline comment:
    # CI-drift fallback: atol relaxed to 1e-10 due to <specific drift annotation>
    and a corresponding entry in tests/fixtures/README.md. Default is atol=1e-12, rtol=0.
    """
    from tests.fixtures._generate_baseline import build_pfc

    df = build_pfc(seed=42, flag=flag)
    baseline = pd.read_parquet(str(_FIXTURES_DIR / "baseline_pfc_seed42.parquet"))

    # ── Identical schema (columns, dtypes) ──
    assert list(df.columns) == list(baseline.columns), (
        f"Column mismatch: {list(df.columns)} != {list(baseline.columns)}"
    )
    assert df.dtypes.to_dict() == baseline.dtypes.to_dict(), (
        f"Dtype mismatch for flag={flag}"
    )

    # ── Identical index values and tz (NOT freq: parquet drops DatetimeIndex.freq) ──
    assert len(df.index) == len(baseline.index), "Index length mismatch"
    assert (df.index == baseline.index).all(), "Index values mismatch"
    assert str(df.index.tz) == str(baseline.index.tz), "Index tz mismatch"

    # ── Numerical equality: atol=1e-12, rtol=0 (default contract) ──
    # Parquet does not preserve DatetimeIndex.freq — the fresh DataFrame has
    # freq=<15 * Minutes> while the loaded baseline has freq=None. Reset to None
    # on the fresh frame before comparing so assert_frame_equal does not fail on freq.
    df_cmp = df.copy()
    df_cmp.index.freq = None
    pd.testing.assert_frame_equal(
        df_cmp,
        baseline,
        check_exact=False,
        atol=1e-12,
        rtol=0,
    )


# ---------------------------------------------------------------------------
# Codex review §8 — test_no_hidden_behavior_branch (AST scan)
# ---------------------------------------------------------------------------

# Functions in shape_hourly.py that are ALLOWED to reference _use_seasonal_hourly.
# Extend deliberately in future plans when behavior gating is intentionally added.
# Plan 05C-01 (D-A1-2): _apply_hydro_analogue_weights now gates the kernel target on
# _use_seasonal_hourly (Lever 1 of Phase 5bis-B — per-timestamp climatological fill
# target vs legacy scalar current_fill). This is intentional behavior gating, NOT a
# 5bis-A violation. See 05C-CONTEXT.md D-A1-1/D-A1-2 and 05C-REVIEWS.md.
ALLOWED_FUNCTIONS = {"__init__", "save", "load", "_resolve_flag", "_apply_hydro_analogue_weights"}


def test_no_hidden_behavior_branch():
    """Codex review §8 (Plan 05B-05): _use_seasonal_hourly is plumbing only in Phase 5bis-A.

    AST-scans pfc_shaping/lt/model/shape_hourly.py and asserts that the attribute
    _use_seasonal_hourly (as an Attribute AST node) is accessed ONLY inside functions
    listed in ALLOWED_FUNCTIONS = {"__init__", "save", "load", "_resolve_flag"}.

    If _use_seasonal_hourly appears in any other function (e.g. fit, apply, _fit_*,
    _apply_*), this test fails — guarding against silent leakage of behavior gating
    into Phase 5bis-A (where the flag must be plumbing only, no numerical branch).

    To allow a new function to reference the flag (e.g., when Phase 5bis-B adds
    behavioral gating), extend ALLOWED_FUNCTIONS deliberately in this file.
    """
    src_path = Path(__file__).resolve().parent.parent / "pfc_shaping" / "lt" / "model" / "shape_hourly.py"
    src = src_path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    violations: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        fn_name = node.name
        if fn_name in ALLOWED_FUNCTIONS:
            continue

        # Walk function body looking for Attribute nodes referencing _use_seasonal_hourly
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Attribute)
                and child.attr == "_use_seasonal_hourly"
            ):
                violations.append(
                    f"Function '{fn_name}' (line {node.lineno}) references "
                    f"_use_seasonal_hourly — not in ALLOWED_FUNCTIONS {ALLOWED_FUNCTIONS}"
                )
                break  # one violation per function is enough

    assert not violations, (
        "Phase 5bis-A violation: _use_seasonal_hourly leaked into math-path function(s).\n"
        + "\n".join(violations)
        + "\n\nTo intentionally gate behavior on this flag (Phase 5bis-B), "
        "add the function name to ALLOWED_FUNCTIONS in this test."
    )
