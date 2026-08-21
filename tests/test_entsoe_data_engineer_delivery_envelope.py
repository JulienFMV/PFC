from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pfc_shaping.validation import entsoe_data_engineer_delivery_envelope as module
from pfc_shaping.validation.entsoe_data_engineer_delivery_envelope import (
    EntsoeDataEngineerDeliveryEnvelopeError,
    derive_snapshot_id,
    validate_delivery_envelope_contract,
    validate_delivery_envelope_contract_path,
    verify_delivery_envelope,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = module.CONTRACT_PATH
REFERENCE = datetime(2026, 8, 6, 3, 0, tzinfo=timezone.utc)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bounds(role: str) -> dict[str, dict[str, str | None]]:
    result: dict[str, dict[str, str | None]] = {}
    for field in module.TIMESTAMP_FIELDS_BY_ROLE[role]:
        result[field] = {
            "minimum_utc": "2019-01-01T00:00:00Z",
            "maximum_utc": "2026-08-06T02:00:00Z",
        }
    return result


def _package_document(
    *,
    excluded_series_count: int = 0,
    artifact_mutator=None,
    top_level_mutator=None,
) -> tuple[dict[str, object], dict[str, bytes]]:
    specifications = list(module.REQUIRED_ARTIFACTS)
    if excluded_series_count:
        specifications.append(module.CONDITIONAL_ARTIFACT)
    payloads: dict[str, bytes] = {}
    declarations: list[dict[str, object]] = []
    for role, relative_path, media_type, maximum_size in specifications:
        payload = b"{}" if media_type == module.JSON_MEDIA_TYPE else f"{role}-bytes".encode()
        payloads[relative_path] = payload
        declaration = {
            "role": role,
            "relative_path": relative_path,
            "media_type": media_type,
            "sha256": _sha(payload),
            "size_bytes": len(payload),
            "logical_record_count": (excluded_series_count if role == "excluded_series" else 1),
            "schema_content_id": _sha(f"schema-{role}".encode()),
            "timestamp_bounds": _bounds(role),
        }
        if artifact_mutator is not None:
            artifact_mutator(declaration, maximum_size)
        declarations.append(declaration)
    manifest_without_snapshot = {
        "schema_version": module.MANIFEST_SCHEMA,
        "evidence_class": module.EVIDENCE_CLASS,
        "created_at_utc": "2026-08-06T02:30:00Z",
        "source_catalog": "dev",
        "source_schema": "gold",
        "source_tables": list(module.SOURCE_TABLES),
        "extraction_window": {
            "start_utc": "2019-01-01T00:00:00Z",
            "end_utc": "2026-08-06T02:00:00Z",
        },
        "filters": [],
        "producer": {"job_version": "entsoe-export-v1", "query_version": "q1"},
        "artifacts": declarations,
        "exclusion_summary": {
            "excluded_series_count": excluded_series_count,
            "excluded_series_artifact_present": excluded_series_count > 0,
        },
        "authorities": dict(module.MANIFEST_AUTHORITIES),
    }
    if top_level_mutator is not None:
        top_level_mutator(manifest_without_snapshot)
    manifest = {
        **manifest_without_snapshot,
        "snapshot_id": derive_snapshot_id(manifest_without_snapshot),
    }
    return manifest, payloads


def _write_package(
    root: Path,
    *,
    excluded_series_count: int = 0,
    artifact_mutator=None,
    top_level_mutator=None,
) -> Path:
    manifest, payloads = _package_document(
        excluded_series_count=excluded_series_count,
        artifact_mutator=artifact_mutator,
        top_level_mutator=top_level_mutator,
    )
    package = root / str(manifest["snapshot_id"])
    package.mkdir(parents=True)
    for relative_path, payload in payloads.items():
        (package / relative_path).write_bytes(payload)
    manifest_path = package / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _execute(manifest_path: Path):
    return verify_delivery_envelope(
        contract_path=CONTRACT_PATH,
        manifest_path=manifest_path,
        reference_time_utc=REFERENCE,
    )


@pytest.fixture
def package(tmp_path: Path) -> Path:
    return _write_package(tmp_path)


def test_contract_is_exact_value_blind_and_non_authoritative() -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    direct = validate_delivery_envelope_contract(document)
    stable = validate_delivery_envelope_contract_path(CONTRACT_PATH)

    assert direct == stable
    assert direct["contract_content_id"] == module.CONTRACT_CONTENT_ID
    assert direct["required_artifact_count"] == 10
    assert direct["maximum_artifact_count"] == 11
    assert direct["content_decoded"] is False
    assert direct["model_input_authorized"] is False
    assert all(value == 0 for value in module.EXECUTION.values())


def test_complete_zero_exclusion_envelope_passes_without_decoding(package: Path) -> None:
    result = _execute(package)

    assert result.assessment["status"] == module.PASS_STATUS
    assert result.assessment["artifact_count"] == 10
    assert result.assessment["excluded_series_count"] == 0
    assert result.assessment["artifact_content_decoded"] is False
    assert result.assessment["fixed_inventory_verified"] is True
    assert result.assessment["artifact_hashes_verified_twice"] is True
    assert result.assessment["authorities"] == module.AUTHORITIES
    assert result.assessment["model_input_authorized"] is False


def test_positive_exclusion_requires_and_binds_eleventh_artifact(tmp_path: Path) -> None:
    manifest_path = _write_package(tmp_path, excluded_series_count=3)

    result = _execute(manifest_path)

    assert result.assessment["artifact_count"] == 11
    assert result.assessment["excluded_series_count"] == 3
    assert result.assessment["artifact_receipts"][-1]["role"] == "excluded_series"
    assert result.assessment["artifact_receipts"][-1]["logical_record_count"] == 3


def test_replay_is_content_deterministic(package: Path) -> None:
    first = _execute(package)
    second = _execute(package)

    assert first.assessment_content_id == second.assessment_content_id
    assert first.assessment == second.assessment


def test_invalid_json_and_parquet_payloads_still_prove_value_blind_boundary(package: Path) -> None:
    result = _execute(package)

    assert result.assessment["decoded_parquet_row_count"] == 0
    assert result.assessment["decoded_json_artifact_count"] == 0
    assert result.assessment["opened_real_value_row_count"] == 0


def test_assessment_contains_no_paths_or_artifact_bytes(package: Path) -> None:
    result = _execute(package)
    encoded = json.dumps(result.assessment, sort_keys=True)

    assert "series_dimension.parquet" not in encoded
    assert "series_dimension-bytes" not in encoded
    assert str(package.parent) not in encoded
    assert result.assessment["source_authenticity_authorized"] is False
    assert result.assessment["schema_or_data_quality_authorized"] is False


def test_artifact_byte_tampering_is_rejected(package: Path) -> None:
    artifact = package.parent / "latest_values.parquet"
    artifact.write_bytes(artifact.read_bytes() + b"tamper")

    with pytest.raises(EntsoeDataEngineerDeliveryEnvelopeError, match="artifact hash"):
        _execute(package)


def test_missing_required_file_is_rejected(package: Path) -> None:
    (package.parent / "gap_report.parquet").unlink()

    with pytest.raises(EntsoeDataEngineerDeliveryEnvelopeError, match="inventory differs"):
        _execute(package)


def test_unexpected_file_is_rejected(package: Path) -> None:
    (package.parent / "README.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(EntsoeDataEngineerDeliveryEnvelopeError, match="inventory differs"):
        _execute(package)


def test_wrong_package_directory_name_is_rejected(tmp_path: Path) -> None:
    manifest, payloads = _package_document()
    package = tmp_path / "not-the-snapshot"
    package.mkdir()
    for relative_path, payload in payloads.items():
        (package / relative_path).write_bytes(payload)
    manifest_path = package / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(EntsoeDataEngineerDeliveryEnvelopeError, match="directory name"):
        _execute(manifest_path)


def test_relative_manifest_path_is_rejected(package: Path) -> None:
    with pytest.raises(EntsoeDataEngineerDeliveryEnvelopeError, match="path is unsafe"):
        verify_delivery_envelope(
            contract_path=CONTRACT_PATH,
            manifest_path=Path("manifest.json"),
            reference_time_utc=REFERENCE,
        )


def test_hardlinked_artifact_is_rejected(package: Path) -> None:
    artifact = package.parent / "series_dimension.parquet"
    os.link(artifact, package.parent.parent / "series-dimension-alias.parquet")

    with pytest.raises(EntsoeDataEngineerDeliveryEnvelopeError, match="single-link"):
        _execute(package)


def test_duplicate_manifest_key_is_rejected(package: Path) -> None:
    payload = package.read_text(encoding="utf-8")
    package.write_text(
        payload.replace("{", '{"schema_version":"duplicate",', 1),
        encoding="utf-8",
    )

    with pytest.raises(EntsoeDataEngineerDeliveryEnvelopeError, match="duplicate keys"):
        _execute(package)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda document: document.update(source_catalog="prd"), "source catalog"),
        (
            lambda document: document["source_tables"].reverse(),
            "source tables",
        ),
        (
            lambda document: document.update(created_at_utc="2026-08-06T03:00:01Z"),
            "future-dated",
        ),
        (
            lambda document: document["authorities"].update(model_input=True),
            "manifest authorities",
        ),
    ],
    ids=["catalog", "source-order", "future", "authority-escalation"],
)
def test_top_level_manifest_drift_fails_closed(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    manifest_path = _write_package(tmp_path, top_level_mutator=mutation)

    with pytest.raises(EntsoeDataEngineerDeliveryEnvelopeError, match=match):
        _execute(manifest_path)


def test_artifact_role_order_is_exact(tmp_path: Path) -> None:
    def mutate(document: dict[str, object]) -> None:
        document["artifacts"][0], document["artifacts"][1] = (
            document["artifacts"][1],
            document["artifacts"][0],
        )

    manifest_path = _write_package(tmp_path, top_level_mutator=mutate)

    with pytest.raises(EntsoeDataEngineerDeliveryEnvelopeError, match="artifact role"):
        _execute(manifest_path)


def test_traversal_artifact_path_is_rejected_before_file_access(tmp_path: Path) -> None:
    def mutate(declaration: dict[str, object], maximum_size: int) -> None:
        if declaration["role"] == "series_dimension":
            declaration["relative_path"] = "../series_dimension.parquet"

    manifest_path = _write_package(tmp_path, artifact_mutator=mutate)

    with pytest.raises(EntsoeDataEngineerDeliveryEnvelopeError, match="artifact path"):
        _execute(manifest_path)


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered"])
def test_required_timestamp_bound_fields_are_exact(tmp_path: Path, mutation: str) -> None:
    def mutate(declaration: dict[str, object], maximum_size: int) -> None:
        if declaration["role"] != "latest_values":
            return
        bounds = declaration["timestamp_bounds"]
        if mutation == "missing":
            del bounds["publication_timestamp_utc"]
        elif mutation == "extra":
            bounds["invented_at_utc"] = copy.deepcopy(bounds["target_start_utc"])
        else:
            reordered = list(bounds.items())[::-1]
            declaration["timestamp_bounds"] = dict(reordered)

    manifest_path = _write_package(tmp_path, artifact_mutator=mutate)

    with pytest.raises(EntsoeDataEngineerDeliveryEnvelopeError, match="fields differ"):
        _execute(manifest_path)


@pytest.mark.parametrize("mutation", ["one-sided", "inverted", "noncanonical"])
def test_timestamp_bound_values_fail_closed(tmp_path: Path, mutation: str) -> None:
    def mutate(declaration: dict[str, object], maximum_size: int) -> None:
        if declaration["role"] != "series_dimension":
            return
        bound = declaration["timestamp_bounds"]["meta_load_timestamp_utc"]
        if mutation == "one-sided":
            bound["minimum_utc"] = None
        elif mutation == "inverted":
            bound["minimum_utc"] = "2026-08-06T02:00:00Z"
            bound["maximum_utc"] = "2019-01-01T00:00:00Z"
        else:
            bound["minimum_utc"] = "2019-01-01T00:00:00+00:00"

    manifest_path = _write_package(tmp_path, artifact_mutator=mutate)

    with pytest.raises(EntsoeDataEngineerDeliveryEnvelopeError):
        _execute(manifest_path)


@pytest.mark.parametrize(
    "count,present",
    [(1, False), (0, True)],
    ids=["positive-without-artifact", "zero-with-artifact-flag"],
)
def test_exclusion_count_and_artifact_presence_are_bijective(
    tmp_path: Path,
    count: int,
    present: bool,
) -> None:
    def mutate(document: dict[str, object]) -> None:
        document["exclusion_summary"] = {
            "excluded_series_count": count,
            "excluded_series_artifact_present": present,
        }

    manifest_path = _write_package(tmp_path, top_level_mutator=mutate)

    with pytest.raises(EntsoeDataEngineerDeliveryEnvelopeError, match="presence"):
        _execute(manifest_path)


def test_excluded_series_count_must_match_artifact_record_count(tmp_path: Path) -> None:
    def mutate(declaration: dict[str, object], maximum_size: int) -> None:
        if declaration["role"] == "excluded_series":
            declaration["logical_record_count"] = 2

    manifest_path = _write_package(
        tmp_path,
        excluded_series_count=3,
        artifact_mutator=mutate,
    )

    with pytest.raises(EntsoeDataEngineerDeliveryEnvelopeError, match="record count"):
        _execute(manifest_path)


def test_mid_stream_mutation_signal_is_rejected(package: Path, monkeypatch) -> None:
    original = module._stream_hash_once
    call_count = 0

    def unstable(path: Path, *, label: str, maximum_size: int):
        nonlocal call_count
        call_count += 1
        result = original(path, label=label, maximum_size=maximum_size)
        if call_count == len(module.REQUIRED_ARTIFACTS) + 1:
            return _sha(b"different"), result[1]
        return result

    monkeypatch.setattr(module, "_stream_hash_once", unstable)

    with pytest.raises(EntsoeDataEngineerDeliveryEnvelopeError, match="post-envelope"):
        _execute(package)


def test_mid_execution_manifest_mutation_is_rejected(package: Path, monkeypatch) -> None:
    original = module._read_small
    manifest_reads = 0

    def unstable(path: Path, label: str, maximum_bytes: int) -> bytes:
        nonlocal manifest_reads
        payload = original(path, label, maximum_bytes)
        if Path(path) == package:
            manifest_reads += 1
            if manifest_reads == 2:
                return payload + b" "
        return payload

    monkeypatch.setattr(module, "_read_small", unstable)

    with pytest.raises(EntsoeDataEngineerDeliveryEnvelopeError, match="manifest changed"):
        _execute(package)


def test_mid_execution_inventory_mutation_is_rejected(package: Path, monkeypatch) -> None:
    original = module._validate_package_inventory
    inventory_reads = 0

    def unstable(path: Path, expected_names: set[str]) -> None:
        nonlocal inventory_reads
        inventory_reads += 1
        if inventory_reads == 2:
            raise EntsoeDataEngineerDeliveryEnvelopeError("package inventory differs")
        original(path, expected_names)

    monkeypatch.setattr(module, "_validate_package_inventory", unstable)

    with pytest.raises(EntsoeDataEngineerDeliveryEnvelopeError, match="inventory differs"):
        _execute(package)


def test_snapshot_id_is_content_derived_and_manifest_bound() -> None:
    manifest, _ = _package_document()
    snapshot_id = manifest.pop("snapshot_id")

    assert derive_snapshot_id(manifest) == snapshot_id
    mutated = copy.deepcopy(manifest)
    mutated["producer"]["query_version"] = "q2"
    assert derive_snapshot_id(mutated) != snapshot_id


def test_contract_hash_rejects_mutated_copy(tmp_path: Path) -> None:
    mutated = tmp_path / "contract.json"
    mutated.write_bytes(CONTRACT_PATH.read_bytes() + b" ")

    with pytest.raises(EntsoeDataEngineerDeliveryEnvelopeError, match="SHA-256 differs"):
        validate_delivery_envelope_contract_path(mutated)


def test_module_has_no_ct_or_remote_execution_dependency() -> None:
    source = (ROOT / "pfc_shaping/validation/entsoe_data_engineer_delivery_envelope.py").read_text(
        encoding="utf-8"
    )

    assert "pfc_shaping.ct" not in source
    assert "databricks.sql" not in source
    assert "requests." not in source
    assert "H:\\" not in source
