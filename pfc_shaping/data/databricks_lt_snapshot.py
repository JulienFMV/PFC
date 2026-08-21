"""Governed binding of Databricks export evidence to an LT replay package.

The verifier is deliberately read-only and in-memory.  It neither connects to
Databricks nor grants source, calibration, publication or production authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from io import BytesIO

import pandas as pd

from pfc_shaping.data.acquisition_contract import (
    DATABRICKS_UPSTREAM_REPLAY_KIND,
)
from pfc_shaping.data.databricks_lt_replay import (
    verify_databricks_replay_package,
)

DATABRICKS_EXPORT_MANIFEST_SCHEMA = "fmv_databricks_lt_export_manifest.v1"
DATABRICKS_QUALITY_SCHEMA = "lt_source_quality.v3"
DATABRICKS_EXPORT_MODE = "FULL_SNAPSHOT"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_READ_ONLY_FORBIDDEN_SQL = re.compile(
    r"\b(?:ALTER|COPY|CREATE|DELETE|DROP|GRANT|INSERT|MERGE|OPTIMIZE|REPLACE|"
    r"REVOKE|TRUNCATE|UPDATE|VACUUM)\b",
    flags=re.IGNORECASE,
)
_AUTHORITIES = {
    "source_authenticity_verified": False,
    "model_input_authorized": False,
    "calibration_authorized": False,
    "publication_authorized": False,
    "production_authorized": False,
}


class DatabricksLTSnapshotError(ValueError):
    """Raised when signed snapshot declarations do not prove exact replay."""


def canonical_json_bytes(value: object) -> bytes:
    """Encode deterministic JSON used by the export evidence contract."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def databricks_export_id(unsigned_manifest: Mapping[str, object]) -> str:
    """Return the content identity of an export manifest without export_id."""

    return hashlib.sha256(canonical_json_bytes(unsigned_manifest)).hexdigest()


def databricks_replay_bindings(
    entry: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    """Return the exact physical files declared by a Databricks replay role."""

    upstream = entry.get("upstream_replay")
    if not isinstance(upstream, Mapping) or set(upstream) != {
        "kind",
        "replayed_at_utc",
        "manifest",
        "artifacts",
        "export_manifest",
    }:
        raise DatabricksLTSnapshotError("Databricks upstream replay declaration is not exact")
    if upstream.get("kind") != DATABRICKS_UPSTREAM_REPLAY_KIND:
        raise DatabricksLTSnapshotError("Databricks upstream replay kind is invalid")
    result: dict[str, Mapping[str, object]] = {}
    for label in ("manifest", "export_manifest"):
        binding = upstream.get(label)
        if not isinstance(binding, Mapping):
            raise DatabricksLTSnapshotError(f"Databricks upstream {label} binding is missing")
        _validate_binding(binding, label=label)
        result[label] = binding
    artifacts = upstream.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise DatabricksLTSnapshotError("Databricks replay artifacts are missing")
    for artifact_role, binding in artifacts.items():
        if not isinstance(artifact_role, str) or not artifact_role:
            raise DatabricksLTSnapshotError("Databricks artifact role is invalid")
        if not isinstance(binding, Mapping):
            raise DatabricksLTSnapshotError(
                f"Databricks artifact binding is invalid: {artifact_role}"
            )
        _validate_binding(binding, label=f"artifact {artifact_role}")
        result[f"artifact:{artifact_role}"] = binding
    _utc(upstream.get("replayed_at_utc"), label="Databricks replayed_at_utc")
    return result


def verify_databricks_snapshot_replay(
    *,
    role: str,
    source_system: str,
    entry: Mapping[str, object],
    manifest_payload: bytes,
    artifact_payloads: Mapping[str, bytes],
    export_manifest_payload: bytes,
    raw_payload: bytes,
    derived_payload: bytes,
) -> dict[str, object]:
    """Verify a complete PRD export-to-model replay chain for one LT role."""

    bindings = databricks_replay_bindings(entry)
    manifest_binding = bindings["manifest"]
    export_binding = bindings["export_manifest"]
    _assert_payload_binding(
        manifest_payload,
        manifest_binding,
        label="Databricks replay manifest",
    )
    _assert_payload_binding(
        export_manifest_payload,
        export_binding,
        label="Databricks export manifest",
    )
    for artifact_role, payload in artifact_payloads.items():
        binding = bindings.get(f"artifact:{artifact_role}")
        if binding is None:
            raise DatabricksLTSnapshotError(
                f"Databricks replay artifact is undeclared: {artifact_role}"
            )
        _assert_payload_binding(
            payload,
            binding,
            label=f"Databricks replay artifact {artifact_role}",
        )
    declared_roles = {
        key.removeprefix("artifact:") for key in bindings if key.startswith("artifact:")
    }
    if set(artifact_payloads) != declared_roles:
        raise DatabricksLTSnapshotError("Databricks replay artifact inventory is incomplete")
    if artifact_payloads.get("model/raw.parquet") != raw_payload:
        raise DatabricksLTSnapshotError(
            "Databricks replay raw frame differs from the canonical LT raw artifact"
        )
    if artifact_payloads.get("model/derived.parquet") != derived_payload:
        raise DatabricksLTSnapshotError(
            "Databricks replay derived frame differs from the LT role artifact"
        )

    try:
        replay_result = verify_databricks_replay_package(
            artifacts=artifact_payloads,
            manifest_payload=manifest_payload,
        )
    except ValueError as exc:
        raise DatabricksLTSnapshotError(str(exc)) from exc
    export = _strict_json_mapping(
        export_manifest_payload,
        label="Databricks export manifest",
    )
    replay_manifest = _strict_json_mapping(
        manifest_payload,
        label="Databricks replay manifest",
    )
    replay_config = _strict_json_mapping(
        artifact_payloads["evidence/replay-config.json"],
        label="Databricks replay config",
    )
    _verify_export_manifest(
        export,
        role=role,
        replay_manifest=replay_manifest,
        replay_config=replay_config,
        artifact_payloads=artifact_payloads,
    )
    if str(replay_config.get("source_system", "")) != source_system:
        raise DatabricksLTSnapshotError(
            "Databricks replay source system differs from the governed role"
        )
    upstream = entry["upstream_replay"]
    assert isinstance(upstream, Mapping)
    replayed_at = _utc(
        upstream.get("replayed_at_utc"),
        label="Databricks replayed_at_utc",
    )
    exported_at = _utc(export.get("exported_at_utc"), label="exported_at_utc")
    receipt = entry.get("source_receipt")
    if not isinstance(receipt, Mapping):
        raise DatabricksLTSnapshotError("Databricks source receipt is missing")
    received_at = _utc(
        receipt.get("received_at_utc"),
        label="source receipt received_at_utc",
    )
    if not (exported_at <= received_at <= replayed_at):
        raise DatabricksLTSnapshotError(
            "Databricks export, source receipt and replay times are not ordered"
        )
    source_artifacts = replay_config.get("source_artifacts")
    assert isinstance(source_artifacts, Mapping)
    expected_locator = "databricks://" + ",".join(
        sorted(str(value["source_table"]) for value in source_artifacts.values())
    )
    if str(receipt.get("source_locator", "")).lower() != expected_locator:
        raise DatabricksLTSnapshotError(
            "Databricks source receipt locator differs from replay tables"
        )
    return {
        **replay_result,
        "export_id": export["export_id"],
        "export_manifest_sha256": hashlib.sha256(export_manifest_payload).hexdigest(),
        "source_environment": "PRD",
        "status": "VERIFIED_SIGNABLE_PRD_DATABRICKS_REPLAY_CHAIN",
        "authorities": dict(_AUTHORITIES),
    }


def expected_databricks_quality_bindings(
    *,
    entry: Mapping[str, object],
    raw: Mapping[str, object],
    derivation: Mapping[str, object],
) -> dict[str, str]:
    """Return the exact v3 quality bindings for a Databricks-backed role."""

    bindings = databricks_replay_bindings(entry)
    artifact = {
        key.removeprefix("artifact:"): value
        for key, value in bindings.items()
        if key.startswith("artifact:")
    }
    required = {
        "evidence/materializer.py",
        "evidence/replay-config.json",
        "evidence/materialization-audit.json",
    }
    if not required.issubset(artifact):
        raise DatabricksLTSnapshotError(
            "Databricks replay evidence artifact inventory is incomplete"
        )
    return {
        "databricks_replay_manifest_sha256": str(bindings["manifest"]["sha256"]),
        "databricks_export_manifest_sha256": str(bindings["export_manifest"]["sha256"]),
        "databricks_materializer_code_sha256": str(artifact["evidence/materializer.py"]["sha256"]),
        "databricks_replay_config_sha256": str(artifact["evidence/replay-config.json"]["sha256"]),
        "databricks_materialization_audit_sha256": str(
            artifact["evidence/materialization-audit.json"]["sha256"]
        ),
        "bronze_sha256": str(raw.get("sha256", "")),
        "feature_parser_code_sha256": str(derivation.get("parser_code_sha256", "")),
        "feature_parser_config_sha256": str(derivation.get("parser_config_sha256", "")),
        "derived_sha256": str(entry.get("sha256", "")),
    }


def _verify_export_manifest(
    manifest: Mapping[str, object],
    *,
    role: str,
    replay_manifest: Mapping[str, object],
    replay_config: Mapping[str, object],
    artifact_payloads: Mapping[str, bytes],
) -> None:
    expected_fields = {
        "schema_version",
        "export_id",
        "role",
        "source_environment",
        "export_mode",
        "predecessor_generation_id",
        "as_of_utc",
        "exported_at_utc",
        "source_queries",
        "cost_evidence",
        "authorities",
    }
    if set(manifest) != expected_fields:
        raise DatabricksLTSnapshotError("Databricks export manifest fields are not exact")
    if manifest.get("schema_version") != DATABRICKS_EXPORT_MANIFEST_SCHEMA:
        raise DatabricksLTSnapshotError("Databricks export schema is unsupported")
    identity = dict(manifest)
    export_id = _sha256(identity.pop("export_id", None), label="export_id")
    if databricks_export_id(identity) != export_id:
        raise DatabricksLTSnapshotError("Databricks export_id is invalid")
    if manifest.get("role") != role or replay_manifest.get("role") != role:
        raise DatabricksLTSnapshotError("Databricks export role differs")
    if (
        manifest.get("source_environment") != "PRD"
        or replay_manifest.get("source_environment") != "PRD"
        or replay_config.get("source_environment") != "PRD"
    ):
        raise DatabricksLTSnapshotError(
            "calibration-eligible Databricks replay requires PRD sources"
        )
    if (
        manifest.get("export_mode") != DATABRICKS_EXPORT_MODE
        or manifest.get("predecessor_generation_id") is not None
    ):
        raise DatabricksLTSnapshotError(
            "v4 admits only an explicit full snapshot; incremental composition is unproved"
        )
    as_of = _utc(manifest.get("as_of_utc"), label="as_of_utc")
    exported = _utc(manifest.get("exported_at_utc"), label="exported_at_utc")
    if as_of.isoformat() != str(replay_config.get("as_of_utc", "")) or exported < as_of:
        raise DatabricksLTSnapshotError("Databricks export/replay point-in-time ordering differs")
    if manifest.get("authorities") != _AUTHORITIES:
        raise DatabricksLTSnapshotError("Databricks export grants unsupported authority")
    source_queries = manifest.get("source_queries")
    source_artifacts = replay_config.get("source_artifacts")
    if not isinstance(source_queries, Mapping) or not isinstance(source_artifacts, Mapping):
        raise DatabricksLTSnapshotError("Databricks export source inventory is invalid")
    if set(source_queries) != set(source_artifacts):
        raise DatabricksLTSnapshotError("Databricks export source inventory differs")
    total_rows = 0
    total_bytes = 0
    for source_role, declaration in source_queries.items():
        if not isinstance(declaration, Mapping):
            raise DatabricksLTSnapshotError(f"Databricks source query is invalid: {source_role}")
        expected = source_artifacts[source_role]
        assert isinstance(expected, Mapping)
        source_path = f"sources/{source_role}.parquet"
        payload = artifact_payloads.get(source_path)
        if payload is None:
            raise DatabricksLTSnapshotError(f"Databricks source payload is missing: {source_role}")
        row_count, size_bytes = _verify_source_query(
            declaration,
            expected=expected,
            payload=payload,
            source_role=str(source_role),
            as_of=as_of,
        )
        total_rows += row_count
        total_bytes += size_bytes
    _verify_cost_evidence(
        manifest.get("cost_evidence"),
        statement_count=len(source_queries),
        rows_exported=total_rows,
        bytes_exported=total_bytes,
    )


def _verify_source_query(
    declaration: Mapping[str, object],
    *,
    expected: Mapping[str, object],
    payload: bytes,
    source_role: str,
    as_of: datetime,
) -> tuple[int, int]:
    fields = {
        "source_table",
        "selected_columns",
        "predicate_sql",
        "watermark_column",
        "lower_watermark_exclusive_utc",
        "upper_watermark_inclusive_utc",
        "row_count",
        "artifact_sha256",
        "artifact_size_bytes",
    }
    if set(declaration) != fields:
        raise DatabricksLTSnapshotError(
            f"Databricks source query fields are not exact: {source_role}"
        )
    table = str(declaration.get("source_table", "")).strip().lower()
    if table != expected.get("source_table") or not table.startswith("prd."):
        raise DatabricksLTSnapshotError(f"Databricks source table differs: {source_role}")
    columns = declaration.get("selected_columns")
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(value, str) or not value.strip() for value in columns)
        or len(columns) != len(set(columns))
    ):
        raise DatabricksLTSnapshotError(
            f"Databricks selected-column inventory is invalid: {source_role}"
        )
    predicate = str(declaration.get("predicate_sql", "")).strip()
    if (
        not predicate
        or ";" in predicate
        or _READ_ONLY_FORBIDDEN_SQL.search(predicate)
        or table not in predicate.lower()
        or not predicate.lstrip().upper().startswith(("SELECT ", "WITH "))
    ):
        raise DatabricksLTSnapshotError(
            f"Databricks query is not a bounded read-only statement: {source_role}"
        )
    if declaration.get("lower_watermark_exclusive_utc") is not None:
        raise DatabricksLTSnapshotError(
            "full-snapshot Databricks export cannot claim an incremental lower watermark"
        )
    watermark_column = declaration.get("watermark_column")
    upper_value = declaration.get("upper_watermark_inclusive_utc")
    if (watermark_column is None) != (upper_value is None):
        raise DatabricksLTSnapshotError(
            f"Databricks watermark declaration is incomplete: {source_role}"
        )
    if watermark_column is not None:
        if not isinstance(watermark_column, str) or not watermark_column.strip():
            raise DatabricksLTSnapshotError(
                f"Databricks watermark column is invalid: {source_role}"
            )
        if _utc(upper_value, label=f"{source_role} upper watermark") > as_of:
            raise DatabricksLTSnapshotError(
                f"Databricks upper watermark exceeds the PIT origin: {source_role}"
            )
    digest = hashlib.sha256(payload).hexdigest()
    size = len(payload)
    row_count = declaration.get("row_count")
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 1:
        raise DatabricksLTSnapshotError(f"Databricks source row count is invalid: {source_role}")
    try:
        observed_frame = pd.read_parquet(BytesIO(payload))
    except (OSError, ValueError) as exc:
        raise DatabricksLTSnapshotError(
            f"Databricks source payload is unreadable: {source_role}"
        ) from exc
    if row_count != len(observed_frame):
        raise DatabricksLTSnapshotError(f"Databricks source row count differs: {source_role}")
    if columns != [str(column) for column in observed_frame.columns]:
        raise DatabricksLTSnapshotError(
            f"Databricks selected columns differ from payload: {source_role}"
        )
    if (
        digest != declaration.get("artifact_sha256")
        or digest != expected.get("sha256")
        or size != declaration.get("artifact_size_bytes")
        or size != expected.get("size_bytes")
    ):
        raise DatabricksLTSnapshotError(
            f"Databricks source artifact binding differs: {source_role}"
        )
    return row_count, size


def _verify_cost_evidence(
    value: object,
    *,
    statement_count: int,
    rows_exported: int,
    bytes_exported: int,
) -> None:
    fields = {
        "schema_version",
        "evidence_basis",
        "statement_count",
        "warehouse_start_count",
        "rows_exported",
        "bytes_exported",
        "billing_usage_quantity_dbcu",
        "estimated_cost_chf",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise DatabricksLTSnapshotError("Databricks cost evidence fields are not exact")
    if value.get("schema_version") != "fmv_databricks_lt_export_cost.v1":
        raise DatabricksLTSnapshotError("Databricks cost evidence schema is unsupported")
    basis = value.get("evidence_basis")
    if basis not in {"CLIENT_OBSERVATION", "DATABRICKS_QUERY_HISTORY"}:
        raise DatabricksLTSnapshotError("Databricks cost evidence basis is invalid")
    expected = {
        "statement_count": statement_count,
        "rows_exported": rows_exported,
        "bytes_exported": bytes_exported,
    }
    if any(value.get(field) != expected_value for field, expected_value in expected.items()):
        raise DatabricksLTSnapshotError("Databricks cost counters differ from export")
    starts = value.get("warehouse_start_count")
    if not isinstance(starts, int) or isinstance(starts, bool) or starts not in {0, 1}:
        raise DatabricksLTSnapshotError("Databricks warehouse-start count is invalid")
    for field in ("billing_usage_quantity_dbcu", "estimated_cost_chf"):
        amount = value.get(field)
        if amount is not None and (
            not isinstance(amount, (int, float)) or isinstance(amount, bool) or amount < 0
        ):
            raise DatabricksLTSnapshotError(f"Databricks {field} is invalid")
    if basis == "CLIENT_OBSERVATION" and any(
        value.get(field) is not None
        for field in ("billing_usage_quantity_dbcu", "estimated_cost_chf")
    ):
        raise DatabricksLTSnapshotError(
            "client observation cannot claim Databricks billing quantities"
        )


def _validate_binding(binding: Mapping[str, object], *, label: str) -> None:
    if set(binding) != {"path", "sha256", "size_bytes"}:
        raise DatabricksLTSnapshotError(f"Databricks {label} binding is not exact")
    path = str(binding.get("path", ""))
    if not path or path.startswith(("/", "\\")) or ".." in path.replace("\\", "/").split("/"):
        raise DatabricksLTSnapshotError(f"Databricks {label} path is not portable")
    _sha256(binding.get("sha256"), label=f"{label} SHA-256")
    size = binding.get("size_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise DatabricksLTSnapshotError(f"Databricks {label} size is invalid")


def _assert_payload_binding(
    payload: bytes,
    binding: Mapping[str, object],
    *,
    label: str,
) -> None:
    if hashlib.sha256(payload).hexdigest() != binding.get("sha256") or len(payload) != binding.get(
        "size_bytes"
    ):
        raise DatabricksLTSnapshotError(f"{label} bytes differ from binding")


def _strict_json_mapping(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatabricksLTSnapshotError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise DatabricksLTSnapshotError(f"{label} is not canonical JSON")
    return value


def _sha256(value: object, *, label: str) -> str:
    normalized = str(value or "")
    if not _SHA256.fullmatch(normalized):
        raise DatabricksLTSnapshotError(f"Databricks {label} is invalid")
    return normalized


def _utc(value: object, *, label: str) -> datetime:
    raw = str(value or "")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DatabricksLTSnapshotError(f"Databricks {label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DatabricksLTSnapshotError(f"Databricks {label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


__all__ = [
    "DATABRICKS_EXPORT_MANIFEST_SCHEMA",
    "DATABRICKS_EXPORT_MODE",
    "DATABRICKS_QUALITY_SCHEMA",
    "DatabricksLTSnapshotError",
    "canonical_json_bytes",
    "databricks_export_id",
    "databricks_replay_bindings",
    "expected_databricks_quality_bindings",
    "verify_databricks_snapshot_replay",
]
