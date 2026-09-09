from __future__ import annotations

import copy
import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pfc_shaping.validation import databricks_gold_snapshot_intake as module
from pfc_shaping.validation.databricks_gold_snapshot_intake import (
    DatabricksGoldSnapshotIntakeError,
    derive_parquet_schema_sha256,
    derive_snapshot_id,
    validate_gold_snapshot_intake_contract,
    validate_gold_snapshot_intake_contract_path,
    validate_gold_snapshot_package,
)

REFERENCE = datetime(2026, 8, 6, 18, 0, tzinfo=timezone.utc)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class PackageFixture:
    package: Path
    manifest: dict[str, object]
    external_paths: list[Path] = field(default_factory=list)

    @property
    def manifest_path(self) -> Path:
        return self.package / "manifest.json"

    def write_manifest(self, *, rebind: bool) -> None:
        if rebind:
            derivation = copy.deepcopy(self.manifest)
            derivation.pop("snapshot_id", None)
            snapshot_id = derive_snapshot_id(derivation)
            self.manifest["snapshot_id"] = snapshot_id
            destination = self.package.parent / snapshot_id
            if destination != self.package:
                destination.mkdir()
                for entry in tuple(self.package.iterdir()):
                    if entry.name != "manifest.json":
                        (destination / entry.name).write_bytes(entry.read_bytes())
                for entry in tuple(self.package.iterdir()):
                    entry.unlink()
                self.package.rmdir()
                self.package = destination
        self.manifest_path.write_bytes(_json_bytes(self.manifest))

    def refresh_total(self) -> None:
        self.manifest["total_size_bytes"] = sum(
            int(item["size_bytes"])
            for item in (*self.manifest["companions"], *self.manifest["artifacts"])
        )

    def refresh_companion(self, role: str) -> None:
        declaration = next(item for item in self.manifest["companions"] if item["role"] == role)
        path = self.package / str(declaration["relative_path"])
        declaration["sha256"] = _hash(path)
        declaration["size_bytes"] = path.stat().st_size
        self.refresh_total()

    def refresh_artifact(self, role: str, *, schema_hash: str | None = None) -> None:
        declaration = next(item for item in self.manifest["artifacts"] if item["role"] == role)
        path = self.package / str(declaration["relative_path"])
        declaration["sha256"] = _hash(path)
        declaration["size_bytes"] = path.stat().st_size
        parquet = pq.ParquetFile(path)
        try:
            declaration["row_count"] = parquet.metadata.num_rows
            declaration["column_count"] = parquet.metadata.num_columns
            declaration["row_group_count"] = parquet.metadata.num_row_groups
            declaration["schema_sha256"] = (
                derive_parquet_schema_sha256(parquet.schema_arrow)
                if schema_hash is None
                else schema_hash
            )
        finally:
            parquet.close()
        self.refresh_total()


@pytest.fixture
def package_factory():
    module.EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    created: list[PackageFixture] = []

    def build() -> PackageFixture:
        artifacts = []
        artifact_payloads = {}
        for role, source_tables, filename, allow_empty in module.ARTIFACT_SPECS:
            rows = 0 if allow_empty else 2
            table = pa.table(
                {
                    "entity_id": pa.array(
                        [f"{role}-{index}" for index in range(rows)],
                        type=pa.string(),
                    ),
                    "target_time_utc": pa.array(
                        (
                            [
                                datetime(2026, 1, 1, tzinfo=timezone.utc),
                                datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
                            ]
                            if rows
                            else []
                        ),
                        type=pa.timestamp("us", tz="UTC"),
                    ),
                    "value": pa.array([1.0, -1.0] if rows else [], type=pa.float64()),
                }
            )
            sink = pa.BufferOutputStream()
            pq.write_table(table, sink, compression="NONE")
            payload = sink.getvalue().to_pybytes()
            artifact_payloads[filename] = payload
            parquet = pq.ParquetFile(pa.BufferReader(payload))
            try:
                artifacts.append(
                    {
                        "role": role,
                        "source_tables": list(source_tables),
                        "format": "PARQUET",
                        "relative_path": filename,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": len(payload),
                        "row_count": parquet.metadata.num_rows,
                        "column_count": parquet.metadata.num_columns,
                        "row_group_count": parquet.metadata.num_row_groups,
                        "schema_sha256": derive_parquet_schema_sha256(parquet.schema_arrow),
                    }
                )
            finally:
                parquet.close()
        companion_payloads = {
            "cost_receipt": b'{"evidence_class":"SYNTHETIC_TEST_ONLY"}\n',
            "source_semantics": b"# Synthetic D292 fixture\n",
            "entsoe_quality_summary": b'{"series_count":2}\n',
            "entsoe_family_inventory": b'{"families":[]}\n',
        }
        declarations = []
        for role, filename, media_type in module.COMPANION_SPECS:
            payload = companion_payloads[role]
            declarations.append(
                {
                    "role": role,
                    "relative_path": filename,
                    "media_type": media_type,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
            )
        manifest_without_id = {
            "schema_version": module.MANIFEST_SCHEMA,
            "evidence_class": module.EVIDENCE_CLASS,
            "export_batch_id": str(uuid.uuid4()),
            "created_at_utc": "2026-08-06T17:00:00Z",
            "source_catalog": "prd",
            "source_schema": "gold",
            "request_sha256": module.REQUEST_SHA256,
            "entsoe_request_sha256": module.ENTSOE_REQUEST_SHA256,
            "cost_preflight_contract_content_id": module.COST_CONTRACT_CONTENT_ID,
            "total_size_bytes": sum(
                int(item["size_bytes"]) for item in (*declarations, *artifacts)
            ),
            "companions": declarations,
            "artifacts": artifacts,
            "execution": dict(module.MANIFEST_EXECUTION),
            "authorities": dict(module.MANIFEST_AUTHORITIES),
        }
        snapshot_id = derive_snapshot_id(manifest_without_id)
        manifest = {
            "schema_version": manifest_without_id["schema_version"],
            "evidence_class": manifest_without_id["evidence_class"],
            "snapshot_id": snapshot_id,
            **{
                key: value
                for key, value in manifest_without_id.items()
                if key not in {"schema_version", "evidence_class"}
            },
        }
        package = module.EXPORT_ROOT / snapshot_id
        package.mkdir()
        for filename, payload in artifact_payloads.items():
            (package / filename).write_bytes(payload)
        for role, filename, _ in module.COMPANION_SPECS:
            payload = companion_payloads[role]
            (package / filename).write_bytes(payload)
        fixture = PackageFixture(package=package, manifest=manifest)
        fixture.write_manifest(rebind=False)
        created.append(fixture)
        return fixture

    yield build

    for fixture in reversed(created):
        for external in fixture.external_paths:
            external.unlink(missing_ok=True)
        if fixture.package.exists():
            for entry in tuple(fixture.package.iterdir()):
                if entry.is_file():
                    entry.unlink()
            fixture.package.rmdir()


def _validate(fixture: PackageFixture):
    return validate_gold_snapshot_package(
        fixture.manifest_path,
        reference_time_utc=REFERENCE,
    )


def test_contract_is_exact_and_has_no_remote_or_model_authority() -> None:
    result = validate_gold_snapshot_intake_contract_path()
    assert result["contract_content_id"] == module.CONTRACT_CONTENT_ID
    assert result["exact_artifact_count"] == 16
    assert result["exact_companion_count"] == 4
    assert result["databricks_connection_count"] == 0
    assert result["business_value_row_count_opened"] == 0
    assert result["model_input_authorized"] is False
    assert result["production_authorized"] is False


@pytest.mark.parametrize(
    "mutator",
    [
        lambda document: document["input_bindings"].pop("data_engineer_request_sha256"),
        lambda document: document["gold_source_contract"].update({"unexpected": "field"}),
        lambda document: document["manifest_policy"].pop("unexpected_package_entries_allowed"),
        lambda document: document["authorities"].update({"unexpected_authority": False}),
    ],
)
def test_contract_nested_field_drift_is_rejected(mutator) -> None:
    document = json.loads(module.CONTRACT_PATH.read_text(encoding="utf-8"))
    mutator(document)
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="fields differ"):
        validate_gold_snapshot_intake_contract(document)


def test_valid_package_verifies_bytes_and_footers_only(package_factory) -> None:
    fixture = package_factory()
    result = _validate(fixture)
    assert result.summary["status"] == module.RESULT_STATUS
    assert result.summary["artifact_count"] == 16
    assert result.summary["artifact_row_count"] == 28
    assert result.summary["execution"] == module.LOCAL_EXECUTION
    assert result.summary["authorities"]["local_byte_integrity"] is True
    assert result.summary["authorities"]["parquet_footer_integrity"] is True
    assert result.summary["authorities"]["semantic_quality"] is False
    assert result.summary["authorities"]["model_input"] is False
    assert result.summary["authorities"]["production"] is False


def test_artifact_byte_tamper_is_rejected(package_factory) -> None:
    fixture = package_factory()
    artifact = fixture.package / module.ARTIFACT_SPECS[0][2]
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="size"):
        _validate(fixture)


@pytest.mark.parametrize(
    "role",
    [
        "cost_receipt",
        "source_semantics",
        "entsoe_quality_summary",
        "entsoe_family_inventory",
    ],
)
def test_companion_byte_tamper_is_rejected(package_factory, role) -> None:
    fixture = package_factory()
    declaration = next(item for item in fixture.manifest["companions"] if item["role"] == role)
    path = fixture.package / declaration["relative_path"]
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="size"):
        _validate(fixture)


def test_missing_artifact_is_rejected(package_factory) -> None:
    fixture = package_factory()
    (fixture.package / module.ARTIFACT_SPECS[0][2]).unlink()
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="inventory"):
        _validate(fixture)


def test_unexpected_package_entry_is_rejected(package_factory) -> None:
    fixture = package_factory()
    (fixture.package / "unexpected.txt").write_text("x", encoding="utf-8")
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="inventory"):
        _validate(fixture)


def test_artifact_reordering_is_rejected(package_factory) -> None:
    fixture = package_factory()
    fixture.manifest["artifacts"][0], fixture.manifest["artifacts"][1] = (
        fixture.manifest["artifacts"][1],
        fixture.manifest["artifacts"][0],
    )
    fixture.write_manifest(rebind=False)
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="role"):
        _validate(fixture)


def test_silver_source_substitution_is_rejected(package_factory) -> None:
    fixture = package_factory()
    fixture.manifest["artifacts"][0]["source_tables"] = ["dev.silver.dimspotproduct"]
    fixture.write_manifest(rebind=False)
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="source tables"):
        _validate(fixture)


def test_artifact_path_substitution_is_rejected(package_factory) -> None:
    fixture = package_factory()
    fixture.manifest["artifacts"][0]["relative_path"] = "../escape.parquet"
    fixture.write_manifest(rebind=False)
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="path"):
        _validate(fixture)


@pytest.mark.parametrize(
    ("field", "delta", "message"),
    [
        ("row_count", 1, "row count"),
        ("column_count", 1, "column count"),
        ("row_group_count", 1, "row-group count"),
    ],
)
def test_footer_metadata_mismatch_is_rejected(package_factory, field, delta, message) -> None:
    fixture = package_factory()
    fixture.manifest["artifacts"][0][field] += delta
    fixture.write_manifest(rebind=True)
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match=message):
        _validate(fixture)


def test_schema_fingerprint_mismatch_is_rejected(package_factory) -> None:
    fixture = package_factory()
    fixture.manifest["artifacts"][0]["schema_sha256"] = "0" * 64
    fixture.write_manifest(rebind=True)
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="schema hash"):
        _validate(fixture)


def test_total_size_mismatch_is_rejected(package_factory) -> None:
    fixture = package_factory()
    fixture.manifest["total_size_bytes"] += 1
    fixture.write_manifest(rebind=True)
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="total size"):
        _validate(fixture)


def test_corrupt_parquet_is_rejected_after_exact_rebinding(package_factory) -> None:
    fixture = package_factory()
    declaration = fixture.manifest["artifacts"][0]
    path = fixture.package / declaration["relative_path"]
    path.write_bytes(b"PAR1brokenPAR1")
    declaration["sha256"] = _hash(path)
    declaration["size_bytes"] = path.stat().st_size
    fixture.refresh_total()
    fixture.write_manifest(rebind=True)
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="not valid Parquet"):
        _validate(fixture)


def test_nested_parquet_schema_is_rejected(package_factory) -> None:
    fixture = package_factory()
    role = module.ARTIFACT_SPECS[0][0]
    declaration = fixture.manifest["artifacts"][0]
    path = fixture.package / declaration["relative_path"]
    pq.write_table(pa.table({"nested": pa.array([[1], [2]])}), path, compression="NONE")
    fixture.refresh_artifact(role, schema_hash="0" * 64)
    fixture.write_manifest(rebind=True)
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="schema must be flat"):
        _validate(fixture)


def test_empty_artifact_declaration_is_rejected(package_factory) -> None:
    fixture = package_factory()
    fixture.manifest["artifacts"][0]["row_count"] = 0
    fixture.write_manifest(rebind=False)
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="non-empty"):
        _validate(fixture)


def test_snapshot_id_substitution_is_rejected(package_factory) -> None:
    fixture = package_factory()
    fixture.manifest["snapshot_id"] = "0" * 64
    fixture.write_manifest(rebind=False)
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="derivation"):
        _validate(fixture)


def test_noncanonical_export_batch_id_is_rejected(package_factory) -> None:
    fixture = package_factory()
    fixture.manifest["export_batch_id"] = str(uuid.uuid4()).upper()
    fixture.write_manifest(rebind=False)
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="UUID4"):
        _validate(fixture)


def test_future_manifest_is_rejected(package_factory) -> None:
    fixture = package_factory()
    fixture.manifest["created_at_utc"] = "2026-08-06T18:06:00Z"
    fixture.write_manifest(rebind=False)
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="future"):
        _validate(fixture)


def test_manifest_authority_elevation_is_rejected(package_factory) -> None:
    fixture = package_factory()
    fixture.manifest["authorities"]["model_input_authorized"] = True
    fixture.write_manifest(rebind=False)
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="authorities"):
        _validate(fixture)


def test_databricks_write_claim_is_rejected(package_factory) -> None:
    fixture = package_factory()
    fixture.manifest["execution"]["databricks_write_count"] = 1
    fixture.write_manifest(rebind=False)
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="execution"):
        _validate(fixture)


def test_empty_source_semantics_is_rejected_after_exact_rebinding(package_factory) -> None:
    fixture = package_factory()
    path = fixture.package / "source_semantics.md"
    path.write_bytes(b" \n")
    fixture.refresh_companion("source_semantics")
    fixture.write_manifest(rebind=True)
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="empty"):
        _validate(fixture)


def test_invalid_cost_receipt_json_is_rejected_after_exact_rebinding(
    package_factory,
) -> None:
    fixture = package_factory()
    path = fixture.package / "cost_receipt.json"
    path.write_bytes(b"not-json\n")
    fixture.refresh_companion("cost_receipt")
    fixture.write_manifest(rebind=True)
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="strict JSON"):
        _validate(fixture)


def test_duplicate_manifest_json_key_is_rejected(package_factory) -> None:
    fixture = package_factory()
    payload = fixture.manifest_path.read_bytes()
    fixture.manifest_path.write_bytes(
        payload.replace(b'{"artifacts":', b'{"artifacts":[],"artifacts":', 1)
    )
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="duplicate JSON keys"):
        _validate(fixture)


def test_hardlinked_artifact_is_rejected(package_factory) -> None:
    fixture = package_factory()
    artifact = fixture.package / module.ARTIFACT_SPECS[0][2]
    outside = module.ROOT / "build" / f"d292-hardlink-{uuid.uuid4()}.parquet"
    os.link(artifact, outside)
    fixture.external_paths.append(outside)
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="single-link"):
        _validate(fixture)


def test_relative_manifest_path_is_rejected(package_factory) -> None:
    fixture = package_factory()
    relative = fixture.manifest_path.relative_to(module.ROOT)
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="absolute"):
        validate_gold_snapshot_package(relative, reference_time_utc=REFERENCE)


def test_contract_byte_drift_is_rejected(tmp_path) -> None:
    copy_path = tmp_path / "contract.json"
    copy_path.write_bytes(module.CONTRACT_PATH.read_bytes() + b" ")
    with pytest.raises(DatabricksGoldSnapshotIntakeError, match="contract hash"):
        validate_gold_snapshot_intake_contract_path(copy_path)
