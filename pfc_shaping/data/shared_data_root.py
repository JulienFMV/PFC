"""Resolve the shared FMV data platform without consumer-owned defaults."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import re

from pfc_shaping.path_safety import path_is_link

FMV_DATA_ROOT_ENV = "FMV_DATA_ROOT"
PFC_SHARED_DATA_ROOT_ENV = "PFC_SHARED_DATA_ROOT"
PFC_LT_DATA_ROOT_ENV = "PFC_LT_DATA_ROOT"
PFC_ENTSOE_DATA_ROOT_ENV = "PFC_ENTSOE_DATA_ROOT"
LT_DATA_VIEW_ID = "fmv_data_view:pfc_lt"

_DOMAIN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def configured_lt_data_root(
    *,
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Return the configured shared root for LT, including its legacy alias."""

    environment = os.environ if environ is None else environ
    names = (
        FMV_DATA_ROOT_ENV,
        PFC_LT_DATA_ROOT_ENV,
        PFC_SHARED_DATA_ROOT_ENV,
    )
    present = [(name, environment[name]) for name in names if name in environment]
    if not present:
        return None
    nonempty = [(name, value) for name, value in present if str(value).strip()]
    if not nonempty:
        return None
    if len(nonempty) != len(present):
        empty = [name for name, value in present if not str(value).strip()]
        raise ValueError(f"configured FMV data root aliases are empty: {empty}")
    selected_name, selected_value = nonempty[0]
    selected = Path(str(selected_value).strip()).expanduser()
    for name, value in nonempty[1:]:
        candidate = Path(str(value).strip()).expanduser()
        if candidate.resolve() != selected.resolve():
            raise ValueError(
                "configured FMV data roots diverge: "
                f"{selected_name}={selected}, {name}={candidate}"
            )
    return selected


def resolve_fmv_data_root(
    explicit: str | Path | None = None,
    *,
    compatibility_envs: Sequence[str] = (PFC_SHARED_DATA_ROOT_ENV,),
    environ: Mapping[str, str] | None = None,
    require_exists: bool = False,
) -> Path:
    """Resolve one absolute shared root and reject ambiguous configured aliases."""

    environment = os.environ if environ is None else environ
    if explicit is not None:
        return _absolute_root(explicit, label="FMV data_root", require_exists=require_exists)

    names = (FMV_DATA_ROOT_ENV, *compatibility_envs)
    configured: list[tuple[str, Path]] = []
    for name in names:
        if name not in environment:
            continue
        value = environment[name]
        if not str(value).strip():
            raise ValueError(f"{name} cannot be empty")
        configured.append(
            (name, _absolute_root(value, label=name, require_exists=require_exists))
        )
    if not configured:
        accepted = " / ".join(names)
        raise ValueError(f"explicit data_root or {accepted} is required")

    selected_name, selected = configured[0]
    divergent = [(name, path) for name, path in configured[1:] if path != selected]
    if divergent:
        detail = ", ".join(
            [f"{selected_name}={selected}", *(f"{name}={path}" for name, path in divergent)]
        )
        raise ValueError(f"configured FMV data roots diverge: {detail}")
    return selected


def resolve_data_domain_root(
    domain: str,
    *,
    data_root: str | Path | None = None,
    domain_root: str | Path | None = None,
    compatibility_envs: Sequence[str] = (PFC_SHARED_DATA_ROOT_ENV,),
    domain_env: str | None = None,
    environ: Mapping[str, str] | None = None,
    require_exists: bool = True,
) -> Path:
    """Resolve a confined domain directory under the shared root."""

    if not _DOMAIN.fullmatch(domain):
        raise ValueError(f"invalid shared data domain: {domain}")
    environment = os.environ if environ is None else environ
    explicit_roots = data_root is not None or domain_root is not None
    direct_value: str | Path | None = domain_root
    if not explicit_roots and domain_env and domain_env in environment:
        direct_value = environment[domain_env]

    parent: Path | None = None
    parent_configured = data_root is not None or (
        not explicit_roots
        and any(name in environment for name in (FMV_DATA_ROOT_ENV, *compatibility_envs))
    )
    if parent_configured:
        parent = resolve_fmv_data_root(
            data_root,
            compatibility_envs=compatibility_envs,
            environ=environment,
            require_exists=require_exists,
        )

    direct: Path | None = None
    if direct_value is not None:
        direct = _absolute_root(
            direct_value,
            label=f"{domain} domain_root",
            require_exists=require_exists,
        )
    if parent is None and direct is None:
        accepted = FMV_DATA_ROOT_ENV
        if compatibility_envs:
            accepted += " / " + " / ".join(compatibility_envs)
        if domain_env:
            accepted += f" / {domain_env}"
        raise ValueError(f"explicit data_root/domain_root or {accepted} is required")

    if parent is not None:
        expected = (parent / domain).resolve()
        if not expected.is_relative_to(parent):
            raise ValueError(f"{domain} domain root resolves outside data_root")
        if direct is not None and direct != expected:
            raise ValueError(
                f"{domain} domain_root and data_root do not identify the same store"
            )
        direct = expected
    assert direct is not None
    if direct.name.casefold() != domain.casefold():
        raise ValueError(f"direct {domain} domain_root must end with '{domain}'")
    if require_exists and not direct.is_dir():
        raise FileNotFoundError(f"{domain} domain_root must already exist: {direct}")
    return direct


def resolve_confined_path(
    root: str | Path,
    candidate: str | Path,
    *,
    label: str,
    require_exists: bool = False,
    require_file: bool = False,
) -> Path:
    """Resolve one path below root while rejecting linked ancestors."""

    root_path = Path(root).resolve()
    raw = Path(candidate)
    path = raw if raw.is_absolute() else root_path / raw
    try:
        relative = path.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"{label} is outside data_root") from exc
    current = root_path
    for part in relative.parts:
        current = current / part
        if is_link_or_junction(current):
            raise ValueError(f"{label} cannot contain a link or junction: {current}")
    resolved = path.resolve()
    if not resolved.is_relative_to(root_path):
        raise ValueError(f"{label} resolves outside data_root")
    if require_exists and not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if require_file and not resolved.is_file():
        raise FileNotFoundError(f"{label} must be a regular file: {resolved}")
    return resolved


def is_link_or_junction(path: str | Path) -> bool:
    return path_is_link(Path(path))


def _absolute_root(
    value: str | Path,
    *,
    label: str,
    require_exists: bool,
) -> Path:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} cannot be empty")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        raise ValueError(f"{label} must be absolute: {candidate}")
    resolved = candidate.resolve()
    if require_exists and not resolved.is_dir():
        raise FileNotFoundError(f"{label} must already exist: {resolved}")
    return resolved
