"""Narrow the wheel to the governed LT runtime closure."""

from __future__ import annotations

import hashlib
import os
import runpy
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

_PACKAGE_CONTRACT = runpy.run_path(Path("pfc_shaping") / "package_contract.py")
ALLOWED_RUNTIME_PYTHON_FILES = _PACKAGE_CONTRACT["ALLOWED_RUNTIME_PYTHON_FILES"]
BUILD_IDENTITY_PLACEHOLDER = _PACKAGE_CONTRACT["BUILD_IDENTITY_PLACEHOLDER"]
BUILD_IDENTITY_TEMPLATE = _PACKAGE_CONTRACT["BUILD_IDENTITY_TEMPLATE"]
REPRODUCIBLE_BUILD_EPOCH = _PACKAGE_CONTRACT["REPRODUCIBLE_BUILD_EPOCH"]

_configured_epoch = os.environ.get("SOURCE_DATE_EPOCH")
if _configured_epoch not in (None, str(REPRODUCIBLE_BUILD_EPOCH)):
    raise RuntimeError(
        "SOURCE_DATE_EPOCH must match the governed LT package contract: "
        f"{REPRODUCIBLE_BUILD_EPOCH}"
    )
os.environ["SOURCE_DATE_EPOCH"] = str(REPRODUCIBLE_BUILD_EPOCH)


class GovernedLtBuildPy(build_py):
    """Exclude non-LT and research modules from the production wheel."""

    def find_package_modules(self, package: str, package_dir: str):
        discovered = super().find_package_modules(package, package_dir)
        return [
            entry
            for entry in discovered
            if _module_file(entry[0], entry[1]) in ALLOWED_RUNTIME_PYTHON_FILES
        ]

    def run(self) -> None:
        super().run()
        package_root = Path(self.build_lib) / "pfc_shaping"
        _remove_stale_runtime_sources(package_root)
        revision = _runtime_source_revision(package_root)
        identity = package_root / "build_identity.py"
        identity.write_text(
            BUILD_IDENTITY_TEMPLATE.format(source_revision=revision),
            encoding="ascii",
            newline="\n",
        )


def _module_file(package: str, module: str) -> str:
    root = package.replace(".", "/")
    return f"{root}/__init__.py" if module == "__init__" else f"{root}/{module}.py"


def _runtime_source_revision(package_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root).as_posix()
        payload = path.read_bytes()
        if relative == "build_identity.py":
            payload = BUILD_IDENTITY_TEMPLATE.format(
                source_revision=BUILD_IDENTITY_PLACEHOLDER
            ).encode("ascii")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def _remove_stale_runtime_sources(package_root: Path) -> None:
    """Prevent incremental setuptools state from bypassing the positive inventory."""

    for path in sorted(package_root.rglob("*.py")):
        member = f"pfc_shaping/{path.relative_to(package_root).as_posix()}"
        if member not in ALLOWED_RUNTIME_PYTHON_FILES:
            path.unlink()
    for directory in sorted(
        (path for path in package_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass


setup(cmdclass={"build_py": GovernedLtBuildPy})
