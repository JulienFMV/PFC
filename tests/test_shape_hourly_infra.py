"""
test_shape_hourly_infra.py
--------------------------
Tests for ShapeHourly infrastructure: save/load sidecar roundtrip (Plan 05B-02)
and feature flag PFC_LT_USE_SEASONAL_HOURLY_SHAPE (Plan 05B-03).

TDD RED phase: all tests here should FAIL before implementation.

Tests verify:
- Task 1: save() writes ${stem}.meta.parquet sidecar with correct schema and content
- Task 2: load() restores all attributes from sidecar; legacy compat warning on missing sidecar
- Plan 05B-03 Task 1: feature flag constructor arg + env-default + freeze-at-init
- Plan 05B-03 Task 2: flag persisted in meta sidecar and restored on load (parquet wins over env)
"""

import inspect
import json
import logging
import os
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
        """
        hp_rows = self.meta[self.meta["attr"] == "hyperparams"]
        assert len(hp_rows) == 1
        obj = json.loads(hp_rows["value"].iloc[0])
        # use_seasonal_hourly added by Plan 05B-03 — must be present
        assert obj == {
            "halflife_days": 90.0,
            "hydro_weight_sigma": 0.7,
            "sigma": 0.3,
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
        sh = ShapeHourly(sigma=0.5, halflife_days=180.0, hydro_weight_sigma=0.25)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "shape_hourly.parquet")
            sh.save(p)
            meta = pd.read_parquet(os.path.join(d, "shape_hourly.meta.parquet"))
            hp = meta[meta["attr"] == "hyperparams"]
            obj = json.loads(hp["value"].iloc[0])
            # Plan 05B-03: use_seasonal_hourly now included in hyperparams
            assert obj == {
                "halflife_days": 180.0,
                "hydro_weight_sigma": 0.25,
                "sigma": 0.5,
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
        """After 05B-03, hyperparams must include sigma, halflife_days, hydro_weight_sigma, use_seasonal_hourly."""
        monkeypatch.delenv("PFC_LT_USE_SEASONAL_HOURLY_SHAPE", raising=False)
        sh = ShapeHourly(sigma=0.3, halflife_days=90.0, hydro_weight_sigma=0.7, use_seasonal_hourly=True)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "shape_hourly.parquet")
            sh.save(p)
            meta = pd.read_parquet(os.path.join(d, "shape_hourly.meta.parquet"))
            hp = json.loads(meta[meta["attr"] == "hyperparams"].iloc[0]["value"])
            assert set(hp.keys()) == {"sigma", "halflife_days", "hydro_weight_sigma", "use_seasonal_hourly"}


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
