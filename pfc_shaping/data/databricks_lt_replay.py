"""Exact local replay from bounded Databricks exports to LT model frames.

This module is the Databricks-specific source boundary that the generic
``lt_input_snapshot.v3`` API-response envelope cannot represent.  It accepts
only already exported Parquet bytes, binds their exact bytes and source-table
identities, and proves that the installed materializer reproduces the archived
raw and derived LT frames.  It has no connector, SQL, signing or publication
authority.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Mapping

import pandas as pd

from pfc_shaping.data import databricks_lt_materialization
from pfc_shaping.data.databricks_lt_materialization import (
    DatabricksMaterialization,
    entsoe_feature_mapping_from_contract,
    materialize_entsoe_current_features,
    materialize_entsoe_pit_features,
    materialize_spot_price_history,
)
from pfc_shaping.data.governed_lt_acquisition import (
    dataframe_semantic_sha256,
    provider_runtime_fingerprint,
)
from pfc_shaping.parquet_safety import validate_parquet_allocation_budget
from pfc_shaping.path_safety import read_stable_single_link_file

DATABRICKS_REPLAY_CONFIG_SCHEMA = "fmv_databricks_lt_replay_config.v1"
DATABRICKS_REPLAY_BUILD_SCHEMA = "fmv_databricks_lt_replay_build.v1"
GOLD_SPOT_MODE = "GOLD_SPOT_INTERVAL"
GOLD_ENTSOE_CURRENT_MODE = "GOLD_ENTSOE_CURRENT"
SILVER_ENTSOE_PIT_MODE = "SILVER_ENTSOE_POINT_IN_TIME"

_MODE_ROLES = {
    GOLD_SPOT_MODE: frozenset(
        {"epex_ch", "epex_de", "epex_at", "epex_fr", "epex_it"}
    ),
    GOLD_ENTSOE_CURRENT_MODE: frozenset({"entso"}),
    SILVER_ENTSOE_PIT_MODE: frozenset({"entso"}),
}
_MODE_SOURCE_ROLES = {
    GOLD_SPOT_MODE: ("spot_price_interval",),
    GOLD_ENTSOE_CURRENT_MODE: ("entsoe_series_dimension", "entsoe_latest"),
    SILVER_ENTSOE_PIT_MODE: ("entsoe_series_dimension", "entsoe_vintages"),
}
_SOURCE_TABLE_SUFFIXES = {
    "spot_price_interval": "gold.factspotpriceinterval",
    "entsoe_series_dimension": "gold.dimentsoeseries",
    "entsoe_latest": "gold.factentsoetimeserieslatest",
    "entsoe_vintages": "silver.ge_power_entsoe_time_series_vintages",
}
_ROLE_SOURCE_SYSTEM = {
    "epex_ch": "EPEX_SPOT",
    "epex_de": "EPEX_SPOT",
    "epex_at": "EPEX_SPOT",
    "epex_fr": "EPEX_SPOT",
    "epex_it": "EPEX_SPOT",
    "entso": "ENTSOE_TRANSPARENCY_PLATFORM",
}
_AUTHORITIES = {
    "source_authenticity_verified": False,
    "model_input_authorized": False,
    "calibration_authorized": False,
    "publication_authorized": False,
    "production_authorized": False,
}
_PARQUET_PHYSICAL_TYPES = {
    "BOOLEAN",
    "INT32",
    "INT64",
    "FLOAT",
    "DOUBLE",
    "BYTE_ARRAY",
    "FIXED_LEN_BYTE_ARRAY",
}


class DatabricksLTReplayError(ValueError):
    """Raised when exported bytes cannot exactly replay a model-facing role."""


@dataclass(frozen=True)
class DatabricksRoleReplay:
    """One locally replayed role and its exact non-authoritative config."""

    config: Mapping[str, object]
    materialization: DatabricksMaterialization


@dataclass(frozen=True)
class DatabricksReplayPackage:
    """Self-contained exact-byte package awaiting independent authorities."""

    artifacts: Mapping[str, bytes]
    manifest_payload: bytes
    replay: DatabricksRoleReplay


def approved_databricks_materializer_payload() -> bytes:
    """Read the installed allow-listed Databricks materializer source bytes."""

    path = Path(str(databricks_lt_materialization.__file__))
    try:
        if path.is_file():
            payload = read_stable_single_link_file(
                path,
                label="Databricks LT materializer",
                max_bytes=2 * 1024 * 1024,
            )
        else:
            payload = (
                importlib.resources.files(databricks_lt_materialization.__package__)
                .joinpath(path.name)
                .read_bytes()
            )
    except (AttributeError, ImportError, OSError, TypeError, ValueError) as exc:
        raise DatabricksLTReplayError(
            "Databricks LT materializer source cannot be read"
        ) from exc
    if not payload or len(payload) > 2 * 1024 * 1024:
        raise DatabricksLTReplayError(
            "Databricks LT materializer source size is invalid"
        )
    return payload


def build_databricks_role_replay(
    *,
    role: str,
    mode: str,
    as_of_utc: str | pd.Timestamp,
    source_payloads: Mapping[str, bytes],
    source_tables: Mapping[str, str],
    selection: Mapping[str, object],
) -> DatabricksRoleReplay:
    """Materialize one role from exact local Parquet payloads and bind it."""

    normalized_role, normalized_mode = _validate_role_mode(role, mode)
    expected_sources = _MODE_SOURCE_ROLES[normalized_mode]
    if set(source_payloads) != set(expected_sources):
        raise DatabricksLTReplayError(
            "Databricks replay source payload inventory is not exact"
        )
    normalized_tables, environment = _validate_source_tables(
        source_tables,
        expected_sources=expected_sources,
    )
    frames = {
        source_role: _read_source_parquet(payload, source_role=source_role)
        for source_role, payload in source_payloads.items()
    }
    origin = _utc_scalar(as_of_utc)
    normalized_selection = _validate_selection(
        selection,
        mode=normalized_mode,
        dimension=frames.get("entsoe_series_dimension"),
    )
    result = _materialize(
        mode=normalized_mode,
        origin=origin,
        frames=frames,
        selection=normalized_selection,
    )
    materializer_payload = approved_databricks_materializer_payload()
    config = {
        "schema_version": DATABRICKS_REPLAY_CONFIG_SCHEMA,
        "role": normalized_role,
        "source_system": _ROLE_SOURCE_SYSTEM[normalized_role],
        "mode": normalized_mode,
        "source_environment": environment,
        "as_of_utc": origin.isoformat(),
        "source_artifacts": {
            source_role: {
                "source_table": normalized_tables[source_role],
                "sha256": hashlib.sha256(source_payloads[source_role]).hexdigest(),
                "size_bytes": len(source_payloads[source_role]),
            }
            for source_role in expected_sources
        },
        "selection": normalized_selection,
        "materializer_code_sha256": hashlib.sha256(materializer_payload).hexdigest(),
        "runtime_fingerprint": provider_runtime_fingerprint(),
        "raw_frame_sha256": dataframe_semantic_sha256(result.raw_frame),
        "derived_frame_sha256": dataframe_semantic_sha256(result.derived_frame),
        "materialization_audit_sha256": _canonical_sha256(result.audit),
        "authorities": dict(_AUTHORITIES),
    }
    return DatabricksRoleReplay(config=config, materialization=result)


def verify_databricks_role_replay(
    *,
    config: Mapping[str, object],
    source_payloads: Mapping[str, bytes],
    raw_payload: bytes,
    derived_payload: bytes,
    materializer_payload: bytes,
    materializer_code_sha256: str,
) -> dict[str, object]:
    """Prove exact source exports reproduce archived LT raw/derived frames."""

    normalized = _validate_config(config)
    installed_payload = approved_databricks_materializer_payload()
    installed_hash = hashlib.sha256(installed_payload).hexdigest()
    archived_hash = hashlib.sha256(materializer_payload).hexdigest()
    if (
        installed_hash != archived_hash
        or installed_hash != str(materializer_code_sha256)
        or installed_hash != normalized["materializer_code_sha256"]
    ):
        raise DatabricksLTReplayError(
            "Databricks materializer is not the installed allow-listed version"
        )

    source_artifacts = normalized["source_artifacts"]
    if set(source_payloads) != set(source_artifacts):
        raise DatabricksLTReplayError(
            "Databricks replay source payload inventory differs from config"
        )
    for source_role, declaration in source_artifacts.items():
        payload = source_payloads[source_role]
        if (
            hashlib.sha256(payload).hexdigest() != declaration["sha256"]
            or len(payload) != declaration["size_bytes"]
        ):
            raise DatabricksLTReplayError(
                f"Databricks replay source artifact changed: {source_role}"
            )

    replay = build_databricks_role_replay(
        role=str(normalized["role"]),
        mode=str(normalized["mode"]),
        as_of_utc=str(normalized["as_of_utc"]),
        source_payloads=source_payloads,
        source_tables={
            source_role: str(declaration["source_table"])
            for source_role, declaration in source_artifacts.items()
        },
        selection=normalized["selection"],
    )
    if dict(replay.config) != normalized:
        raise DatabricksLTReplayError(
            "Databricks replay config does not match reproduced semantics"
        )

    archived_raw = _read_model_parquet(raw_payload, label="raw")
    archived_derived = _read_model_parquet(derived_payload, label="derived")
    _assert_frame_equal(
        replay.materialization.raw_frame,
        archived_raw,
        label="raw",
    )
    _assert_frame_equal(
        replay.materialization.derived_frame,
        archived_derived,
        label="derived",
    )
    return {
        "schema_version": DATABRICKS_REPLAY_CONFIG_SCHEMA,
        "role": normalized["role"],
        "mode": normalized["mode"],
        "source_environment": normalized["source_environment"],
        "source_artifact_count": len(source_artifacts),
        "raw_frame_sha256": normalized["raw_frame_sha256"],
        "derived_frame_sha256": normalized["derived_frame_sha256"],
        "status": "VERIFIED_EXACT_DATABRICKS_EXPORT_REPLAY",
        "authorities": dict(_AUTHORITIES),
    }


def build_databricks_replay_package(
    *,
    role: str,
    mode: str,
    as_of_utc: str | pd.Timestamp,
    source_payloads: Mapping[str, bytes],
    source_tables: Mapping[str, str],
    selection: Mapping[str, object],
) -> DatabricksReplayPackage:
    """Build a self-contained unsigned replay package entirely in memory."""

    replay = build_databricks_role_replay(
        role=role,
        mode=mode,
        as_of_utc=as_of_utc,
        source_payloads=source_payloads,
        source_tables=source_tables,
        selection=selection,
    )
    config_payload = _canonical_json_bytes(replay.config)
    audit_payload = _canonical_json_bytes(replay.materialization.audit)
    materializer_payload = approved_databricks_materializer_payload()
    artifacts: dict[str, bytes] = {
        **{
            f"sources/{source_role}.parquet": source_payloads[source_role]
            for source_role in _MODE_SOURCE_ROLES[str(replay.config["mode"])]
        },
        "model/raw.parquet": _parquet_payload(replay.materialization.raw_frame),
        "model/derived.parquet": _parquet_payload(
            replay.materialization.derived_frame
        ),
        "evidence/materializer.py": materializer_payload,
        "evidence/replay-config.json": config_payload,
        "evidence/materialization-audit.json": audit_payload,
    }
    manifest_without_id = {
        "schema_version": DATABRICKS_REPLAY_BUILD_SCHEMA,
        "role": replay.config["role"],
        "mode": replay.config["mode"],
        "source_environment": replay.config["source_environment"],
        "artifacts": {
            path: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for path, payload in sorted(artifacts.items())
        },
        "status": "UNSIGNED_LOCAL_REPLAY_VERIFIED_NOT_PUBLISHED",
        "execution": {
            "databricks_connection_count": 0,
            "databricks_statement_count": 0,
            "warehouse_start_count": 0,
            "network_call_count": 0,
            "remote_write_count": 0,
        },
        "authorities": dict(_AUTHORITIES),
    }
    build_id = _canonical_sha256(manifest_without_id)
    manifest = {**manifest_without_id, "build_id": build_id}
    return DatabricksReplayPackage(
        artifacts=artifacts,
        manifest_payload=_canonical_json_bytes(manifest),
        replay=replay,
    )


def verify_databricks_replay_package(
    *,
    artifacts: Mapping[str, bytes],
    manifest_payload: bytes,
) -> dict[str, object]:
    """Verify exact package inventory and replay without filesystem authority."""

    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatabricksLTReplayError(
            "Databricks replay package manifest is invalid"
        ) from exc
    if not isinstance(manifest, Mapping) or _canonical_json_bytes(
        manifest
    ) != manifest_payload:
        raise DatabricksLTReplayError(
            "Databricks replay package manifest is not canonical JSON"
        )
    expected_manifest_fields = {
        "schema_version",
        "build_id",
        "role",
        "mode",
        "source_environment",
        "artifacts",
        "status",
        "execution",
        "authorities",
    }
    if set(manifest) != expected_manifest_fields or manifest.get(
        "schema_version"
    ) != DATABRICKS_REPLAY_BUILD_SCHEMA:
        raise DatabricksLTReplayError(
            "Databricks replay package manifest fields are not exact"
        )
    identity = dict(manifest)
    build_id = _require_sha256(identity.pop("build_id", None), label="build_id")
    if _canonical_sha256(identity) != build_id:
        raise DatabricksLTReplayError(
            "Databricks replay package build_id is invalid"
        )
    if manifest.get("status") != "UNSIGNED_LOCAL_REPLAY_VERIFIED_NOT_PUBLISHED":
        raise DatabricksLTReplayError(
            "Databricks replay package status is invalid"
        )
    if manifest.get("authorities") != _AUTHORITIES:
        raise DatabricksLTReplayError(
            "Databricks replay package grants unsupported authority"
        )
    if manifest.get("execution") != {
        "databricks_connection_count": 0,
        "databricks_statement_count": 0,
        "warehouse_start_count": 0,
        "network_call_count": 0,
        "remote_write_count": 0,
    }:
        raise DatabricksLTReplayError(
            "Databricks replay package execution declaration is invalid"
        )
    declarations = manifest.get("artifacts")
    if not isinstance(declarations, Mapping) or set(declarations) != set(artifacts):
        raise DatabricksLTReplayError(
            "Databricks replay package artifact inventory differs"
        )
    for path, declaration in declarations.items():
        if (
            not isinstance(declaration, Mapping)
            or set(declaration) != {"sha256", "size_bytes"}
        ):
            raise DatabricksLTReplayError(
                f"Databricks replay package artifact declaration is invalid: {path}"
            )
        payload = artifacts[path]
        if (
            hashlib.sha256(payload).hexdigest() != declaration.get("sha256")
            or len(payload) != declaration.get("size_bytes")
        ):
            raise DatabricksLTReplayError(
                f"Databricks replay package artifact changed: {path}"
            )
    required_fixed = {
        "model/raw.parquet",
        "model/derived.parquet",
        "evidence/materializer.py",
        "evidence/replay-config.json",
        "evidence/materialization-audit.json",
    }
    source_paths = {
        path for path in artifacts if path.startswith("sources/") and path.endswith(".parquet")
    }
    if set(artifacts) != required_fixed | source_paths or not source_paths:
        raise DatabricksLTReplayError(
            "Databricks replay package paths are not exact"
        )
    config = _strict_json_mapping(
        artifacts["evidence/replay-config.json"],
        label="replay config",
    )
    audit = _strict_json_mapping(
        artifacts["evidence/materialization-audit.json"],
        label="materialization audit",
    )
    normalized_config = _validate_config(config)
    expected_sources = _MODE_SOURCE_ROLES[str(normalized_config["mode"])]
    expected_source_paths = {
        f"sources/{source_role}.parquet" for source_role in expected_sources
    }
    if source_paths != expected_source_paths:
        raise DatabricksLTReplayError(
            "Databricks replay package source paths differ from mode"
        )
    if (
        manifest.get("role") != normalized_config["role"]
        or manifest.get("mode") != normalized_config["mode"]
        or manifest.get("source_environment")
        != normalized_config["source_environment"]
        or _canonical_sha256(audit)
        != normalized_config["materialization_audit_sha256"]
    ):
        raise DatabricksLTReplayError(
            "Databricks replay package manifest/config/audit binding differs"
        )
    result = verify_databricks_role_replay(
        config=normalized_config,
        source_payloads={
            source_role: artifacts[f"sources/{source_role}.parquet"]
            for source_role in expected_sources
        },
        raw_payload=artifacts["model/raw.parquet"],
        derived_payload=artifacts["model/derived.parquet"],
        materializer_payload=artifacts["evidence/materializer.py"],
        materializer_code_sha256=str(normalized_config["materializer_code_sha256"]),
    )
    return {
        **result,
        "schema_version": DATABRICKS_REPLAY_BUILD_SCHEMA,
        "build_id": build_id,
        "artifact_count": len(artifacts),
        "status": "VERIFIED_SELF_CONTAINED_DATABRICKS_REPLAY_PACKAGE",
    }


def _materialize(
    *,
    mode: str,
    origin: pd.Timestamp,
    frames: Mapping[str, pd.DataFrame],
    selection: Mapping[str, object],
) -> DatabricksMaterialization:
    if mode == GOLD_SPOT_MODE:
        return materialize_spot_price_history(
            frames["spot_price_interval"],
            market_zone=str(selection["market_zone"]),
            source_product=str(selection["source_product"]),
            as_of_utc=origin,
        )
    dimension = frames["entsoe_series_dimension"]
    mapping_contract = selection["mapping_contract"]
    assert isinstance(mapping_contract, Mapping)
    mapping = entsoe_feature_mapping_from_contract(
        mapping_contract,
        dimension=dimension,
    )
    if mode == GOLD_ENTSOE_CURRENT_MODE:
        return materialize_entsoe_current_features(
            dimension=dimension,
            gold_latest=frames["entsoe_latest"],
            mapping=mapping,
            as_of_utc=origin,
        )
    return materialize_entsoe_pit_features(
        dimension=dimension,
        silver_vintages=frames["entsoe_vintages"],
        mapping=mapping,
        as_of_utc=origin,
    )


def _validate_config(config: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "schema_version",
        "role",
        "source_system",
        "mode",
        "source_environment",
        "as_of_utc",
        "source_artifacts",
        "selection",
        "materializer_code_sha256",
        "runtime_fingerprint",
        "raw_frame_sha256",
        "derived_frame_sha256",
        "materialization_audit_sha256",
        "authorities",
    }
    if not isinstance(config, Mapping) or set(config) != expected:
        raise DatabricksLTReplayError("Databricks replay config fields are not exact")
    if config.get("schema_version") != DATABRICKS_REPLAY_CONFIG_SCHEMA:
        raise DatabricksLTReplayError("Databricks replay config schema is unsupported")
    role, mode = _validate_role_mode(
        str(config.get("role", "")),
        str(config.get("mode", "")),
    )
    if config.get("source_system") != _ROLE_SOURCE_SYSTEM[role]:
        raise DatabricksLTReplayError("Databricks replay source system is invalid")
    if config.get("runtime_fingerprint") != provider_runtime_fingerprint():
        raise DatabricksLTReplayError("Databricks replay runtime fingerprint changed")
    if config.get("authorities") != _AUTHORITIES:
        raise DatabricksLTReplayError("Databricks replay grants unsupported authority")
    for field in (
        "materializer_code_sha256",
        "raw_frame_sha256",
        "derived_frame_sha256",
        "materialization_audit_sha256",
    ):
        _require_sha256(config.get(field), label=field)
    origin = _utc_scalar(config.get("as_of_utc"))
    source_artifacts = config.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping):
        raise DatabricksLTReplayError("Databricks replay source artifacts are invalid")
    source_tables: dict[str, str] = {}
    for source_role, declaration in source_artifacts.items():
        if not isinstance(declaration, Mapping) or set(declaration) != {
            "source_table",
            "sha256",
            "size_bytes",
        }:
            raise DatabricksLTReplayError(
                f"Databricks replay source declaration is invalid: {source_role}"
            )
        _require_sha256(declaration.get("sha256"), label=f"{source_role} sha256")
        size = declaration.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise DatabricksLTReplayError(
                f"Databricks replay source size is invalid: {source_role}"
            )
        source_tables[str(source_role)] = str(declaration.get("source_table", ""))
    normalized_tables, environment = _validate_source_tables(
        source_tables,
        expected_sources=_MODE_SOURCE_ROLES[mode],
    )
    if config.get("source_environment") != environment:
        raise DatabricksLTReplayError("Databricks replay source environment differs")
    selection = config.get("selection")
    if not isinstance(selection, Mapping):
        raise DatabricksLTReplayError("Databricks replay selection is invalid")
    normalized = dict(config)
    normalized["role"] = role
    normalized["mode"] = mode
    normalized["as_of_utc"] = origin.isoformat()
    normalized["source_artifacts"] = {
        source_role: {
            **dict(source_artifacts[source_role]),
            "source_table": normalized_tables[source_role],
        }
        for source_role in _MODE_SOURCE_ROLES[mode]
    }
    return normalized


def _validate_role_mode(role: str, mode: str) -> tuple[str, str]:
    normalized_role = str(role).strip()
    normalized_mode = str(mode).strip()
    allowed_roles = _MODE_ROLES.get(normalized_mode)
    if allowed_roles is None or normalized_role not in allowed_roles:
        raise DatabricksLTReplayError("Databricks replay role/mode is invalid")
    return normalized_role, normalized_mode


def _validate_source_tables(
    source_tables: Mapping[str, str],
    *,
    expected_sources: tuple[str, ...],
) -> tuple[dict[str, str], str]:
    if not isinstance(source_tables, Mapping) or set(source_tables) != set(
        expected_sources
    ):
        raise DatabricksLTReplayError(
            "Databricks replay source-table inventory is not exact"
        )
    normalized: dict[str, str] = {}
    environments: set[str] = set()
    for source_role in expected_sources:
        table = str(source_tables[source_role]).strip().lower()
        parts = table.split(".")
        expected_suffix = _SOURCE_TABLE_SUFFIXES[source_role]
        if (
            len(parts) != 3
            or parts[0] not in {"dev", "prd"}
            or ".".join(parts[1:]) != expected_suffix
        ):
            raise DatabricksLTReplayError(
                f"Databricks replay source table is invalid: {source_role}"
            )
        normalized[source_role] = table
        environments.add(parts[0])
    if len(environments) != 1:
        raise DatabricksLTReplayError(
            "Databricks replay cannot mix DEV and PRD source tables"
        )
    return normalized, environments.pop().upper()


def _validate_selection(
    selection: Mapping[str, object],
    *,
    mode: str,
    dimension: pd.DataFrame | None,
) -> dict[str, object]:
    if not isinstance(selection, Mapping):
        raise DatabricksLTReplayError("Databricks replay selection must be a mapping")
    if mode == GOLD_SPOT_MODE:
        if set(selection) != {"market_zone", "source_product"}:
            raise DatabricksLTReplayError(
                "Databricks spot replay selection fields are not exact"
            )
        market_zone = str(selection.get("market_zone", "")).strip().upper()
        source_product = str(selection.get("source_product", "")).strip()
        if not market_zone or not source_product:
            raise DatabricksLTReplayError(
                "Databricks spot replay selection is incomplete"
            )
        return {"market_zone": market_zone, "source_product": source_product}
    if set(selection) != {"mapping_contract"} or dimension is None:
        raise DatabricksLTReplayError(
            "Databricks ENTSO-E replay selection fields are not exact"
        )
    contract = selection.get("mapping_contract")
    if not isinstance(contract, Mapping):
        raise DatabricksLTReplayError(
            "Databricks ENTSO-E mapping contract is missing"
        )
    entsoe_feature_mapping_from_contract(contract, dimension=dimension)
    return {"mapping_contract": dict(contract)}


def _read_source_parquet(payload: bytes, *, source_role: str) -> pd.DataFrame:
    try:
        validate_parquet_allocation_budget(
            payload,
            label=f"Databricks replay source {source_role}",
            max_rows=2_000_000,
            max_columns=96,
            max_cells=120_000_000,
            max_row_groups=8_192,
            allowed_physical_types=_PARQUET_PHYSICAL_TYPES,
        )
    except ValueError as exc:
        raise DatabricksLTReplayError(str(exc)) from exc
    try:
        frame = pd.read_parquet(BytesIO(payload))
    except (OSError, ValueError) as exc:
        raise DatabricksLTReplayError(
            f"Databricks replay source is not readable Parquet: {source_role}"
        ) from exc
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DatabricksLTReplayError(
            f"Databricks replay source frame is empty: {source_role}"
        )
    return frame


def _read_model_parquet(payload: bytes, *, label: str) -> pd.DataFrame:
    try:
        validate_parquet_allocation_budget(
            payload,
            label=f"Databricks replay {label} frame",
            max_rows=2_000_000,
            max_columns=128,
            max_cells=120_000_000,
            max_row_groups=8_192,
            allowed_physical_types=_PARQUET_PHYSICAL_TYPES,
        )
        frame = pd.read_parquet(BytesIO(payload))
    except (OSError, ValueError) as exc:
        raise DatabricksLTReplayError(
            f"Databricks replay {label} frame is not readable Parquet"
        ) from exc
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DatabricksLTReplayError(f"Databricks replay {label} frame is empty")
    return frame


def _assert_frame_equal(
    expected: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    label: str,
) -> None:
    try:
        pd.testing.assert_frame_equal(
            expected,
            observed,
            check_dtype=True,
            check_exact=True,
            check_like=False,
            check_freq=False,
        )
    except AssertionError as exc:
        raise DatabricksLTReplayError(
            f"Databricks exports do not reproduce archived {label} frame"
        ) from exc
    if expected.attrs != observed.attrs:
        raise DatabricksLTReplayError(
            f"Databricks exports do not reproduce archived {label} metadata"
        )


def _utc_scalar(value: object) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DatabricksLTReplayError("Databricks replay as_of_utc is invalid") from exc
    if timestamp.tzinfo is None:
        raise DatabricksLTReplayError(
            "Databricks replay as_of_utc must be timezone-aware"
        )
    return timestamp.tz_convert("UTC")


def _require_sha256(value: object, *, label: str) -> str:
    normalized = str(value or "")
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise DatabricksLTReplayError(f"Databricks replay {label} is invalid")
    return normalized


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _strict_json_mapping(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatabricksLTReplayError(
            f"Databricks {label} is invalid JSON"
        ) from exc
    if not isinstance(value, dict) or _canonical_json_bytes(value) != payload:
        raise DatabricksLTReplayError(
            f"Databricks {label} is not a canonical mapping"
        )
    return value


def _parquet_payload(frame: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    frame.to_parquet(buffer, index=True)
    return buffer.getvalue()


__all__ = [
    "DATABRICKS_REPLAY_CONFIG_SCHEMA",
    "DATABRICKS_REPLAY_BUILD_SCHEMA",
    "DatabricksLTReplayError",
    "DatabricksReplayPackage",
    "DatabricksRoleReplay",
    "GOLD_ENTSOE_CURRENT_MODE",
    "GOLD_SPOT_MODE",
    "SILVER_ENTSOE_PIT_MODE",
    "approved_databricks_materializer_payload",
    "build_databricks_replay_package",
    "build_databricks_role_replay",
    "verify_databricks_replay_package",
    "verify_databricks_role_replay",
]
