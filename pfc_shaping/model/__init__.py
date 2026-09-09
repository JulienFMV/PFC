"""
Compatibility shim — re-exports from pfc_shaping.lt.model and pfc_shaping.ct.model.

The model layer has been split between long-term (LT) and short-term (CT)
sub-packages:

    pfc_shaping.lt.model   — assembler, shape_hourly[_mlp], shape_intraday,
                             water_value, msfc_spline, uncertainty, pfc_flavors
    pfc_shaping.ct.model   — lear_forecaster, foundation_forecaster,
                             futureboost_experimental, pricefm_experimental

Old import paths (`pfc_shaping.model.<module>`) keep working but emit a
DeprecationWarning. Update call sites to the new paths; treat any external
or notebook reference as a migration item.

Mechanism: when this package is first imported, we lazy-bind each old
submodule name in ``sys.modules`` so that subsequent
``from pfc_shaping.model.<module> import X`` resolves to the new module.
A DeprecationWarning is emitted exactly once per old module on first access.
"""

from __future__ import annotations

import importlib
import sys
import warnings


_LT_MODULES = (
    "assembler",
    "shape_hourly",
    "shape_hourly_mlp",
    "shape_intraday",
    "water_value",
    "msfc_spline",
    "uncertainty",
    "pfc_flavors",
)
_CT_MODULES = (
    "lear_forecaster",
    "foundation_forecaster",
    "futureboost_experimental",
    "pricefm_experimental",
)
_MODULE_TO_TARGET: dict[str, str] = {
    name: f"pfc_shaping.lt.model.{name}" for name in _LT_MODULES
}
_MODULE_TO_TARGET.update(
    {name: f"pfc_shaping.ct.model.{name}" for name in _CT_MODULES}
)
_warned: set[str] = set()


def _bind_legacy(name: str) -> object:
    """Resolve an old submodule name to its new location and cache it.

    Emits a DeprecationWarning once per name. Re-uses the already-loaded
    new module so identity equality (``is``) holds with code that imports
    via the new path.
    """
    new_path = _MODULE_TO_TARGET[name]
    if name not in _warned:
        warnings.warn(
            f"pfc_shaping.model.{name} is deprecated; import from {new_path} instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        _warned.add(name)
    module = importlib.import_module(new_path)
    sys.modules[f"pfc_shaping.model.{name}"] = module
    sys.modules[__name__].__dict__[name] = module
    return module


def __getattr__(name: str):
    """Handle ``from pfc_shaping.model import <module>`` (attribute access)."""
    if name in _MODULE_TO_TARGET:
        return _bind_legacy(name)
    raise AttributeError(f"module 'pfc_shaping.model' has no attribute {name!r}")


# Pre-register submodule aliases so ``from pfc_shaping.model.X import Y``
# also works (Python's import machinery checks sys.modules before the file
# system, so a registered key short-circuits the missing-file lookup).
# We do NOT eagerly import the modules — we register a lazy import-on-access
# proxy so that simply importing this package stays cheap.
class _LazyLegacyModule:
    """A stand-in module that resolves on first attribute access."""

    __slots__ = ("_legacy_name", "_resolved")

    def __init__(self, legacy_name: str) -> None:
        object.__setattr__(self, "_legacy_name", legacy_name)
        object.__setattr__(self, "_resolved", None)

    def _materialize(self):
        if self._resolved is None:
            object.__setattr__(self, "_resolved", _bind_legacy(self._legacy_name))
        return self._resolved

    def __getattr__(self, attr: str):
        return getattr(self._materialize(), attr)

    def __setattr__(self, attr: str, value) -> None:
        setattr(self._materialize(), attr, value)

    def __repr__(self) -> str:
        return f"<lazy legacy shim for pfc_shaping.model.{self._legacy_name}>"


for _name in _MODULE_TO_TARGET:
    _key = f"pfc_shaping.model.{_name}"
    sys.modules.setdefault(_key, _LazyLegacyModule(_name))


def __dir__():
    return sorted(_MODULE_TO_TARGET)
