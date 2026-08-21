"""Deterministic, allow-listed replay of governed LT core inputs."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from pfc_shaping.data import lt_replay_transforms
from pfc_shaping.data.acquisition_contract import REPLAY_GOVERNED_LT_INPUT_ROLES
from pfc_shaping.data.governed_lt_acquisition import (
    ENERGY_CHARTS_SOURCE_SYSTEM,
    OBSERVATION_RESOLUTION_PROVENANCE_ATTR,
    SFOE_SOURCE_SYSTEM,
    energy_price_resolution_provenance,
)
from pfc_shaping.parquet_safety import validate_parquet_allocation_budget
from pfc_shaping.path_safety import read_stable_single_link_file

REPLAY_CONFIG_SCHEMA = "lt_replay_config.v1"


class LTInputReplayError(ValueError):
    """Raised when a governed raw artifact cannot reproduce its derived frame."""


@dataclass(frozen=True)
class _ReplaySpec:
    roles: frozenset[str]
    source_systems: frozenset[str]
    parser_module_path: Path
    transform: Callable[[pd.DataFrame], pd.DataFrame]


_REPLAY_REGISTRY = {
    "pfc.epex_spot_features.v1": _ReplaySpec(
        roles=frozenset({"epex_ch", "epex_de", "epex_at", "epex_fr", "epex_it"}),
        source_systems=frozenset({"EPEX_SPOT", ENERGY_CHARTS_SOURCE_SYSTEM}),
        parser_module_path=Path(lt_replay_transforms.__file__).resolve(),
        transform=lt_replay_transforms.clean_epex,
    ),
    "pfc.entso_features.v1": _ReplaySpec(
        roles=frozenset({"entso"}),
        source_systems=frozenset({"ENTSOE_TRANSPARENCY_PLATFORM"}),
        parser_module_path=Path(lt_replay_transforms.__file__).resolve(),
        transform=lt_replay_transforms.build_entso_features,
    ),
    "pfc.hydro_water_value.v1": _ReplaySpec(
        roles=frozenset({"hydro"}),
        source_systems=frozenset({"FMV_HYDRO_CURATED", SFOE_SOURCE_SYSTEM}),
        parser_module_path=Path(lt_replay_transforms.__file__).resolve(),
        transform=lt_replay_transforms.build_hydro_water_value,
    ),
}


def replay_transform_id_for_role(role: str) -> str:
    mapping = {
        "epex_ch": "pfc.epex_spot_features.v1",
        "epex_de": "pfc.epex_spot_features.v1",
        "epex_at": "pfc.epex_spot_features.v1",
        "epex_fr": "pfc.epex_spot_features.v1",
        "epex_it": "pfc.epex_spot_features.v1",
        "entso": "pfc.entso_features.v1",
        "hydro": "pfc.hydro_water_value.v1",
    }
    try:
        return mapping[str(role)]
    except KeyError as exc:
        raise LTInputReplayError(f"LT replay role is not allow-listed: {role}") from exc


def approved_parser_path_for_role(role: str) -> Path:
    transform_id = replay_transform_id_for_role(role)
    return _REPLAY_REGISTRY[transform_id].parser_module_path


def approved_parser_payload_for_role(role: str) -> bytes:
    """Read the installed feature parser from filesystem or zipimport package."""

    replay_transform_id_for_role(role)
    path = Path(str(lt_replay_transforms.__file__))
    try:
        if path.is_file():
            payload = read_stable_single_link_file(
                path,
                label=f"LT replay parser for {role}",
            )
        else:
            payload = (
                importlib.resources.files(lt_replay_transforms.__package__)
                .joinpath(Path(str(lt_replay_transforms.__file__)).name)
                .read_bytes()
            )
    except (AttributeError, ImportError, OSError, TypeError, ValueError) as exc:
        raise LTInputReplayError(f"LT replay parser cannot be read: {role}") from exc
    if not payload or len(payload) > 2 * 1024 * 1024:
        raise LTInputReplayError(f"LT replay parser size is invalid: {role}")
    return payload


def build_replay_config(
    *,
    role: str,
    source_system: str,
    raw_frame: pd.DataFrame,
    derived_frame: pd.DataFrame,
) -> dict[str, object]:
    """Build the strict semantic replay binding written by a trusted collector."""

    transform_id = replay_transform_id_for_role(role)
    return {
        "schema_version": REPLAY_CONFIG_SCHEMA,
        "transform_id": transform_id,
        "role": str(role),
        "source_system": str(source_system).upper(),
        "raw_format": "PARQUET_CANONICAL_FRAME",
        "raw_frame_sha256": _frame_sha256(raw_frame),
        "derived_frame_sha256": _frame_sha256(derived_frame),
    }


def verify_core_lt_role_replay(
    *,
    role: str,
    source_system: str,
    raw_payload: bytes,
    derived_payload: bytes,
    parser_payload: bytes,
    parser_code_sha256: str,
    parser_config: Mapping[str, object],
) -> dict[str, object]:
    """Replay one core role through installed allow-listed code and compare frames."""

    transform_id = str(parser_config.get("transform_id", ""))
    spec = _REPLAY_REGISTRY.get(transform_id)
    if spec is None:
        raise LTInputReplayError(f"LT replay transform is not allow-listed: {transform_id}")
    expected_keys = {
        "schema_version",
        "transform_id",
        "role",
        "source_system",
        "raw_format",
        "raw_frame_sha256",
        "derived_frame_sha256",
    }
    if set(parser_config) != expected_keys:
        raise LTInputReplayError(f"LT replay config fields are not exact: {role}")
    if parser_config.get("schema_version") != REPLAY_CONFIG_SCHEMA:
        raise LTInputReplayError(f"LT replay config schema is unsupported: {role}")
    if str(role) not in spec.roles or str(parser_config.get("role", "")) != str(role):
        raise LTInputReplayError(f"LT replay transform/role mismatch: {role}")
    normalized_source = str(source_system).upper()
    if (
        normalized_source not in spec.source_systems
        or str(parser_config.get("source_system", "")).upper() != normalized_source
    ):
        raise LTInputReplayError(f"LT replay source system mismatch: {role}")
    if parser_config.get("raw_format") != "PARQUET_CANONICAL_FRAME":
        raise LTInputReplayError(f"LT replay raw format is unsupported: {role}")

    installed_parser = approved_parser_payload_for_role(role)
    installed_hash = hashlib.sha256(installed_parser).hexdigest()
    archived_hash = hashlib.sha256(parser_payload).hexdigest()
    if installed_hash != archived_hash or installed_hash != str(parser_code_sha256):
        raise LTInputReplayError(
            f"LT replay parser is not the installed allow-listed version: {role}"
        )

    raw = _read_parquet(raw_payload, role=role, label="raw")
    derived = _read_parquet(derived_payload, role=role, label="derived")
    _validate_raw_frame(
        role,
        raw,
        require_resolution_provenance=(
            normalized_source == ENERGY_CHARTS_SOURCE_SYSTEM
            and str(role).startswith("epex_")
        ),
    )
    raw_hash = _frame_sha256(raw)
    derived_hash = _frame_sha256(derived)
    if raw_hash != str(parser_config.get("raw_frame_sha256", "")):
        raise LTInputReplayError(f"LT replay raw semantic hash mismatch: {role}")
    if derived_hash != str(parser_config.get("derived_frame_sha256", "")):
        raise LTInputReplayError(f"LT replay derived semantic hash mismatch: {role}")
    try:
        replayed = spec.transform(raw.copy())
    except Exception as exc:
        raise LTInputReplayError(f"LT replay transform failed: {role}") from exc
    try:
        pd.testing.assert_frame_equal(
            replayed,
            derived,
            check_dtype=True,
            check_exact=True,
            check_like=False,
            check_freq=False,
        )
    except AssertionError as exc:
        raise LTInputReplayError(
            f"LT raw artifact does not reproduce derived frame: {role}"
        ) from exc
    if replayed.attrs != derived.attrs:
        raise LTInputReplayError(
            f"LT raw artifact does not reproduce derived metadata: {role}"
        )
    return {
        "schema_version": REPLAY_CONFIG_SCHEMA,
        "transform_id": transform_id,
        "role": str(role),
        "raw_frame_sha256": raw_hash,
        "derived_frame_sha256": derived_hash,
        "status": "VERIFIED_DETERMINISTIC_REPLAY",
    }


def _read_parquet(payload: bytes, *, role: str, label: str) -> pd.DataFrame:
    try:
        validate_parquet_allocation_budget(
            payload,
            label=f"LT replay {label} artifact {role}",
            max_rows=5_000 if role == "hydro" else 2_000_000,
            max_columns=128,
            max_cells=40_000_000,
            max_row_groups=8_192,
            allowed_physical_types={
                "BOOLEAN",
                "INT32",
                "INT64",
                "FLOAT",
                "DOUBLE",
            },
        )
    except ValueError as exc:
        raise LTInputReplayError(str(exc)) from exc
    try:
        frame = pd.read_parquet(BytesIO(payload))
    except (OSError, ValueError) as exc:
        raise LTInputReplayError(f"LT replay {label} artifact is not parquet: {role}") from exc
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise LTInputReplayError(f"LT replay {label} frame is empty: {role}")
    return frame


def _validate_raw_frame(
    role: str,
    frame: pd.DataFrame,
    *,
    require_resolution_provenance: bool,
) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise LTInputReplayError(f"LT replay raw index is not datetime: {role}")
    if frame.index.tz is None:
        raise LTInputReplayError(f"LT replay raw index is not timezone-aware: {role}")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise LTInputReplayError(f"LT replay raw index is not sorted and unique: {role}")
    if role in REPLAY_GOVERNED_LT_INPUT_ROLES and role.startswith("epex_"):
        if list(frame.columns) != ["price_eur_mwh"]:
            raise LTInputReplayError(f"LT replay EPEX bronze schema is invalid: {role}")
        if require_resolution_provenance:
            try:
                energy_price_resolution_provenance(frame, role=role)
            except ValueError as exc:
                raise LTInputReplayError(
                    f"LT replay EPEX resolution provenance is invalid: {role}"
                ) from exc
    elif role == "entso":
        required = {"load_mw", "solar_mw", "wind_mw"}
        forbidden = {"solar_regime", "load_deviation", "flow_deviation"}
        if not required.issubset(frame.columns) or forbidden.intersection(frame.columns):
            raise LTInputReplayError("LT replay ENTSO-E bronze schema is invalid")
    elif role == "hydro":
        required = {"fill_pct", "fill_gwh", "max_capacity_gwh"}
        forbidden = {"fill_deviation", "water_value_proxy", "iso_week"}
        if not required.issubset(frame.columns) or forbidden.intersection(frame.columns):
            raise LTInputReplayError("LT replay hydro bronze schema is invalid")
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise LTInputReplayError(f"LT replay raw numeric values are not finite: {role}")


def _frame_sha256(frame: pd.DataFrame) -> str:
    metadata = {
        "columns": [str(value) for value in frame.columns],
        "dtypes": [str(value) for value in frame.dtypes],
        "index_names": [str(value) for value in frame.index.names],
        "index_dtype": str(frame.index.dtype),
        "rows": len(frame),
    }
    if OBSERVATION_RESOLUTION_PROVENANCE_ATTR in frame.attrs:
        metadata["resolution_provenance"] = frame.attrs[
            OBSERVATION_RESOLUTION_PROVENANCE_ATTR
        ]
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()
