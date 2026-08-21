"""
test_lt_ct_imports.py
---------------------
Verify the LT/CT model split (pfc_shaping.lt.model + pfc_shaping.ct.model)
and the backward-compatibility shim under pfc_shaping.model.

These tests intentionally avoid loading pandas/numpy-heavy code at the
package level: they only assert that:

  1. pfc_shaping.model is the deprecation shim, not a real submodule list.
  2. Importing pfc_shaping.model registers lazy proxies in sys.modules
     for every legacy submodule name.
  3. The lazy proxies materialize to the new location on first attribute
     access and emit a DeprecationWarning exactly once.
  4. New paths (pfc_shaping.lt.model.X / pfc_shaping.ct.model.Y) are
     importable from a clean state and reach the same module identity
     as the legacy alias.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

LT_MODULES = (
    "assembler",
    "shape_hourly",
    "shape_hourly_mlp",
    "shape_intraday",
    "water_value",
    "msfc_spline",
    "uncertainty",
    "pfc_flavors",
)
CT_MODULES = (
    "lear_forecaster",
    "foundation_forecaster",
    "futureboost_experimental",
    "pricefm_experimental",
)


def _purge(prefixes: tuple[str, ...]) -> None:
    for key in [k for k in list(sys.modules) if k.startswith(prefixes)]:
        del sys.modules[key]


@pytest.fixture(autouse=True)
def fresh_modules():
    """Drop cached pfc_shaping modules between tests so import paths are exercised."""
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name.startswith("pfc_shaping")
    }
    _purge(("pfc_shaping",))
    try:
        yield
    finally:
        _purge(("pfc_shaping",))
        sys.modules.update(original_modules)


def test_shim_registers_lazy_proxies_in_sys_modules() -> None:
    """Importing pfc_shaping.model should register every legacy submodule alias."""
    import pfc_shaping.model  # noqa: F401

    for name in LT_MODULES + CT_MODULES:
        key = f"pfc_shaping.model.{name}"
        assert key in sys.modules, f"missing legacy alias for {key}"


def test_shim_dir_lists_all_legacy_modules() -> None:
    import pfc_shaping.model

    listed = set(dir(pfc_shaping.model))
    expected = set(LT_MODULES + CT_MODULES)
    assert expected.issubset(listed), f"missing {expected - listed} in dir()"


def test_lazy_proxy_does_not_force_eager_load() -> None:
    """Importing pfc_shaping.model alone must NOT load heavy LT/CT modules."""
    import pfc_shaping.model  # noqa: F401

    assert "pfc_shaping.lt.model.assembler" not in sys.modules
    assert "pfc_shaping.ct.model.lear_forecaster" not in sys.modules


def test_legacy_attribute_access_emits_warning_and_materialises() -> None:
    """`pfc_shaping.model.assembler` triggers exactly one DeprecationWarning."""
    pytest.importorskip("pandas")  # assembler depends on pandas

    import pfc_shaping.model

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Access twice; warning should fire only once.
        mod_a = pfc_shaping.model.assembler  # noqa: F841
        mod_b = pfc_shaping.model.assembler  # noqa: F841

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, f"expected 1 DeprecationWarning, got {len(deprecations)}"
    assert "pfc_shaping.lt.model.assembler" in str(deprecations[0].message)


def test_legacy_from_import_resolves_to_new_module() -> None:
    """`from pfc_shaping.model.assembler import PFCAssembler` must succeed."""
    pytest.importorskip("pandas")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from pfc_shaping.model.assembler import PFCAssembler as Legacy

        from pfc_shaping.lt.model.assembler import PFCAssembler as Modern

    assert Legacy is Modern, "legacy alias must reach the same class object"


@pytest.mark.parametrize("module_name", LT_MODULES)
def test_new_lt_model_path_is_importable(module_name: str) -> None:
    pytest.importorskip("pandas")
    new = importlib.import_module(f"pfc_shaping.lt.model.{module_name}")
    assert new.__name__ == f"pfc_shaping.lt.model.{module_name}"


# Some CT modules pull heavy optional ML deps that may be absent in lean
# CI / dev environments. Map each module to its hardest extra dep so we
# can skip cleanly instead of failing.
_CT_HEAVY_DEPS: dict[str, str] = {
    "lear_forecaster": "lightgbm",
    "foundation_forecaster": "torch",
    "futureboost_experimental": "lightgbm",
    "pricefm_experimental": "tensorflow",
}


@pytest.mark.parametrize("module_name", CT_MODULES)
def test_new_ct_model_path_is_importable(module_name: str) -> None:
    pytest.importorskip("pandas")
    extra = _CT_HEAVY_DEPS.get(module_name)
    if extra is not None:
        pytest.importorskip(extra)
    new = importlib.import_module(f"pfc_shaping.ct.model.{module_name}")
    assert new.__name__ == f"pfc_shaping.ct.model.{module_name}"


def test_unknown_legacy_attribute_raises_attribute_error() -> None:
    import pfc_shaping.model

    with pytest.raises(AttributeError):
        pfc_shaping.model.does_not_exist  # noqa: B018
