"""Opposable selection gate for the local dependence/power diagnostic."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json

SCHEMA_VERSION = "ch_lt_dependence_power_preflight_selection.v1"
REGISTRY_SCHEMA_VERSION = "ch_lt_dependence_power_preflight_supersession.v1"
REGISTRY_STATUS = "LOCAL_DIAGNOSTIC_V1_SUPERSEDED_BY_V2_NO_GO"
CURRENT_STATUS = "CURRENT_LOCAL_DIAGNOSTIC_SELECTION_VERIFIED_NO_GO"
STALE_STATUS = "SUPERSEDED_DEPENDENCE_POWER_PREFLIGHT_REJECTED"

REGISTRY_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-DEPENDENCE-POWER-PREFLIGHT-SUPERSESSION-20260724.json"
)
SUPERSEDED_RELATIVE_PATH = (
    "output/phase14/ch_lt_dependence_power_preflight/intraday_shape_rolling_origin_20260724_v3.json"
)
REPLACEMENT_RELATIVE_PATH = (
    "output/phase14/ch_lt_dependence_power_preflight/"
    "intraday_shape_rolling_origin_20260724_v3_v2.json"
)
FOLD_RELATIVE_PATH = "output/phase14/intraday_shape_rolling_origin_20260724_v3/fold_metrics.csv"
SOURCE_SUMMARY_RELATIVE_PATH = (
    "output/phase14/intraday_shape_rolling_origin_20260724_v3/summary.json"
)

SUPERSEDED_SHA256 = "614a7a79cd2d22c1e7abe8303900f861a7c660ca199ce1e656268a291157cadd"
REPLACEMENT_SHA256 = "34fc3621ea9082ac6c4c0306c4f7f77e333a33fef83a7f673e62c462ec6d5907"
FOLD_SHA256 = "910dd2dbe0a2c87f345f05431be14769d45e4113e1f54301176ba25fc7ecfe45"
SOURCE_SUMMARY_SHA256 = "523169da91a3ec3ab46f341608274a449f035488bf671d2af82481112a3d8f09"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_FOLD_BYTES = 64 * 1024 * 1024


class DependencePowerSupersessionError(ValueError):
    """Raised when the registry, artifacts, or selected path are not exact."""


def verify_dependence_power_selection(
    *,
    repo_root: str | Path,
    registry_path: str | Path,
    expected_registry_sha256: str,
    selected_artifact_path: str | Path,
    expected_selected_artifact_sha256: str,
) -> dict[str, object]:
    """Reject v1 selection and verify only the exact local v2 diagnostic."""

    _require_sha256(expected_registry_sha256, label="registry")
    _require_sha256(expected_selected_artifact_sha256, label="selected artifact")
    root = assert_absolute_path_has_no_links(_lexical_absolute(repo_root))
    if not root.is_dir():
        raise DependencePowerSupersessionError("repository root is unavailable")

    expected_registry = _repo_path(root, REGISTRY_RELATIVE_PATH)
    supplied_registry = _lexical_absolute(registry_path)
    if _path_key(supplied_registry) != _path_key(expected_registry):
        raise DependencePowerSupersessionError("registry path is not canonical")
    registry_payload = read_stable_single_link_file(
        supplied_registry,
        label="dependence/power supersession registry",
        max_bytes=_MAX_JSON_BYTES,
    )
    registry_sha256 = hashlib.sha256(registry_payload).hexdigest()
    if registry_sha256 != expected_registry_sha256:
        raise DependencePowerSupersessionError("registry SHA-256 mismatch")
    registry = _strict_mapping(registry_payload, label="registry")
    _verify_registry_contract(registry)

    payloads = {
        SUPERSEDED_RELATIVE_PATH: _read_bound_repo_file(
            root,
            SUPERSEDED_RELATIVE_PATH,
            SUPERSEDED_SHA256,
            max_bytes=_MAX_JSON_BYTES,
            label="superseded v1 diagnostic",
        ),
        REPLACEMENT_RELATIVE_PATH: _read_bound_repo_file(
            root,
            REPLACEMENT_RELATIVE_PATH,
            REPLACEMENT_SHA256,
            max_bytes=_MAX_JSON_BYTES,
            label="replacement v2 diagnostic",
        ),
        FOLD_RELATIVE_PATH: _read_bound_repo_file(
            root,
            FOLD_RELATIVE_PATH,
            FOLD_SHA256,
            max_bytes=_MAX_FOLD_BYTES,
            label="forbidden confirmatory fold CSV",
        ),
        SOURCE_SUMMARY_RELATIVE_PATH: _read_bound_repo_file(
            root,
            SOURCE_SUMMARY_RELATIVE_PATH,
            SOURCE_SUMMARY_SHA256,
            max_bytes=_MAX_JSON_BYTES,
            label="forbidden confirmatory source summary",
        ),
    }
    _verify_artifact_payloads(payloads)

    selected = _lexical_absolute(selected_artifact_path)
    superseded_path = _repo_path(root, SUPERSEDED_RELATIVE_PATH)
    replacement_path = _repo_path(root, REPLACEMENT_RELATIVE_PATH)
    if _path_key(selected) == _path_key(superseded_path):
        expected_selected = SUPERSEDED_SHA256
        status = STALE_STATUS
        current = False
        reason = "SELECTED_SCHEMA_AND_HASH_ARE_EXPLICITLY_SUPERSEDED"
    elif _path_key(selected) == _path_key(replacement_path):
        expected_selected = REPLACEMENT_SHA256
        status = CURRENT_STATUS
        current = True
        reason = None
    else:
        raise DependencePowerSupersessionError(
            "selected diagnostic path is neither canonical v1 nor canonical v2"
        )
    if expected_selected_artifact_sha256 != expected_selected:
        raise DependencePowerSupersessionError("selected artifact SHA-256 is not canonical")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "local_diagnostic_selection_current": current,
        "rejection_reason": reason,
        "scientific_admission": False,
        "execution_authorized": False,
        "production_authorization": False,
        "promotion_gate": False,
        "registry_path": REGISTRY_RELATIVE_PATH,
        "registry_sha256": registry_sha256,
        "selected_artifact_path": (
            REPLACEMENT_RELATIVE_PATH if current else SUPERSEDED_RELATIVE_PATH
        ),
        "selected_artifact_sha256": expected_selected,
        "superseded_artifact_sha256": SUPERSEDED_SHA256,
        "replacement_artifact_sha256": REPLACEMENT_SHA256,
        "forbidden_confirmatory_reuse": [
            {"role": "paired_fold_csv", "sha256": FOLD_SHA256},
            {"role": "source_summary", "sha256": SOURCE_SUMMARY_SHA256},
        ],
    }


def _verify_registry_contract(registry: Mapping[str, object]) -> None:
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise DependencePowerSupersessionError("registry schema is unsupported")
    if registry.get("status") != REGISTRY_STATUS:
        raise DependencePowerSupersessionError("registry status is invalid")
    for field in (
        "scientific_admission",
        "execution_authorized",
        "production_authorization",
        "promotion_gate",
        "future_holdout_consumed",
        "t057_consumed",
    ):
        if registry.get(field) is not False:
            raise DependencePowerSupersessionError(f"registry {field} must be exactly false")
    superseded = _mapping(registry.get("superseded"), label="superseded")
    replacement = _mapping(registry.get("replacement"), label="replacement")
    expected_superseded = {
        "path": SUPERSEDED_RELATIVE_PATH,
        "sha256": SUPERSEDED_SHA256,
        "schema_version": "ch_lt_dependence_power_preflight.v1",
    }
    expected_replacement = {
        "path": REPLACEMENT_RELATIVE_PATH,
        "sha256": REPLACEMENT_SHA256,
        "schema_version": "ch_lt_dependence_power_preflight.v2",
        "status": "LOCAL_DE_LU_INTRAHOUR_PLUGIN_SENSITIVITY_NO_GO",
    }
    for field, expected in expected_superseded.items():
        if superseded.get(field) != expected:
            raise DependencePowerSupersessionError(f"registry superseded {field} is invalid")
    for field, expected in expected_replacement.items():
        if replacement.get(field) != expected:
            raise DependencePowerSupersessionError(f"registry replacement {field} is invalid")
    forbidden = registry.get("forbidden_confirmatory_reuse")
    expected_forbidden = [
        {"role": "paired_fold_csv", "path": FOLD_RELATIVE_PATH, "sha256": FOLD_SHA256},
        {
            "role": "source_summary",
            "path": SOURCE_SUMMARY_RELATIVE_PATH,
            "sha256": SOURCE_SUMMARY_SHA256,
        },
    ]
    if forbidden != expected_forbidden:
        raise DependencePowerSupersessionError("registry confirmatory-reuse denylist is invalid")


def _verify_artifact_payloads(payloads: Mapping[str, bytes]) -> None:
    superseded = _strict_mapping(payloads[SUPERSEDED_RELATIVE_PATH], label="superseded diagnostic")
    replacement = _strict_mapping(
        payloads[REPLACEMENT_RELATIVE_PATH], label="replacement diagnostic"
    )
    source_summary = _strict_mapping(payloads[SOURCE_SUMMARY_RELATIVE_PATH], label="source summary")
    if superseded.get("schema_version") != "ch_lt_dependence_power_preflight.v1":
        raise DependencePowerSupersessionError("superseded artifact schema mismatch")
    if replacement.get("schema_version") != "ch_lt_dependence_power_preflight.v2":
        raise DependencePowerSupersessionError("replacement artifact schema mismatch")
    if replacement.get("status") != "LOCAL_DE_LU_INTRAHOUR_PLUGIN_SENSITIVITY_NO_GO":
        raise DependencePowerSupersessionError("replacement artifact status mismatch")
    if source_summary.get("schema_version") != "intraday_shape_estimator_rolling_origin.v2":
        raise DependencePowerSupersessionError("source summary schema mismatch")
    for document, label in (
        (superseded, "superseded artifact"),
        (replacement, "replacement artifact"),
    ):
        for field in (
            "scientific_admission",
            "execution_authorized",
            "production_authorization",
            "promotion_gate",
        ):
            if field in document and document.get(field) is not False:
                raise DependencePowerSupersessionError(f"{label} {field} must be exactly false")


def _read_bound_repo_file(
    root: Path,
    relative_path: str,
    expected_sha256: str,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    payload = read_stable_single_link_file(
        _repo_path(root, relative_path),
        label=label,
        max_bytes=max_bytes,
    )
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise DependencePowerSupersessionError(f"{label} SHA-256 mismatch")
    return payload


def _repo_path(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != relative_path:
        raise DependencePowerSupersessionError("registry relative path is unsafe")
    return root.joinpath(*pure.parts)


def _strict_mapping(payload: bytes, *, label: str) -> Mapping[str, object]:
    try:
        parsed = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise DependencePowerSupersessionError(f"{label} is invalid JSON") from exc
    return _mapping(parsed, label=label)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise DependencePowerSupersessionError(f"{label} must be an object")
    return value


def _lexical_absolute(path: str | Path) -> Path:
    selected = Path(path).expanduser()
    if not selected.is_absolute():
        selected = Path(os.path.abspath(selected))
    return selected


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DependencePowerSupersessionError(f"{label} SHA-256 is invalid")


__all__ = [
    "CURRENT_STATUS",
    "DependencePowerSupersessionError",
    "REGISTRY_RELATIVE_PATH",
    "REPLACEMENT_RELATIVE_PATH",
    "REPLACEMENT_SHA256",
    "SCHEMA_VERSION",
    "STALE_STATUS",
    "SUPERSEDED_RELATIVE_PATH",
    "SUPERSEDED_SHA256",
    "verify_dependence_power_selection",
]
