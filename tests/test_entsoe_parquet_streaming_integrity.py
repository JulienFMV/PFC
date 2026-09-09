from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pfc_shaping.validation import entsoe_parquet_streaming_integrity as module
from pfc_shaping.validation.entsoe_parquet_streaming_integrity import (
    EntsoeParquetStreamingIntegrityError,
    derive_arrow_schema_sha256,
    derive_normalized_schema_sha256,
    derive_snapshot_id,
    expected_arrow_schema,
    validate_parquet_streaming_contract,
    validate_parquet_streaming_receipt,
    verify_synthetic_entsoe_parquet_package,
)

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / ".planning" / "phases" / "14-lt-audit-remediation"
CONTRACT_PATH = (
    PHASE / "ENTSOE-PARQUET-STREAMING-INTEGRITY-PREFLIGHT-CONTRACT-V1.json"
)
REFERENCE = datetime(2026, 8, 6, 1, 0, tzinfo=timezone.utc)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _table(role: str, *, defect: str | None = None) -> pa.Table:
    schema = expected_arrow_schema(role)
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    if role == "series_dimension":
        rows: dict[str, list[object]] = {
            "series_id": ["price_ch", "load_ch", "solar_ch"],
            "group_name": ["day_ahead_price", "actual_load", "generation"],
            "field_name": ["price", "load", "solar"],
            "from_zone": ["CH", "CH", "CH"],
            "to_zone": ["CH", "CH", "CH"],
            "unit": ["EUR/MWh", "MW", "MW"],
            "native_resolution": ["PT1H", "PT1H", "PT15M"],
            "source_endpoint": ["synthetic"] * 3,
            "document_id": ["doc-price", "doc-load", "doc-solar"],
        }
    else:
        count = 8_193
        targets = [base + timedelta(minutes=15 * index) for index in range(count)]
        common: dict[str, list[object]] = {
            "series_id": ["solar_ch", "load_ch", "price_ch"] * (count // 3)
            + ["solar_ch"] * (count % 3),
            "target_ts_utc": targets,
            "value": [float(index % 997) / 10.0 for index in range(count)],
            "quality_flag": ["SYNTHETIC"] * count,
            "revision_number": [index % 3 for index in range(count)],
            "meta_load_timestamp_utc": [target + timedelta(hours=1) for target in targets],
        }
        if role == "latest_values":
            rows = {
                "series_id": common["series_id"],
                "target_ts_utc": common["target_ts_utc"],
                "value": common["value"],
                "is_historical": [True] * count,
                "quality_flag": common["quality_flag"],
                "revision_number": common["revision_number"],
                "meta_load_timestamp_utc": common["meta_load_timestamp_utc"],
            }
        else:
            rows = {
                "series_id": common["series_id"],
                "target_ts_utc": common["target_ts_utc"],
                "as_of_utc": [target - timedelta(hours=1) for target in targets],
                "value": common["value"],
                "quality_flag": common["quality_flag"],
                "revision_number": common["revision_number"],
                "meta_load_timestamp_utc": common["meta_load_timestamp_utc"],
            }
    if defect == "null":
        rows["series_id"][0] = None
    if defect == "nonfinite":
        rows["value"][0] = math.inf
    if defect == "negative_revision":
        rows["revision_number"][0] = -1
    arrays = [pa.array(rows[field.name], type=field.type) for field in schema]
    table = pa.Table.from_arrays(arrays, schema=schema)
    if defect == "float32":
        index = table.schema.get_field_index("value")
        table = table.set_column(index, "value", table.column(index).cast(pa.float32()))
    return table


@pytest.fixture
def package_factory():
    created: list[Path] = []

    def build(
        *,
        table_defects: dict[str, str] | None = None,
        compression: str = "zstd",
        file_mutator=None,
        declaration_mutator=None,
        evidence_class: str = "SYNTHETIC_TEST_ONLY",
    ) -> tuple[Path, dict[str, object]]:
        table_defects = {} if table_defects is None else table_defects
        staging = ROOT / "build" / "d244-fixture-staging" / str(uuid.uuid4())
        staging.mkdir(parents=True)
        created.append(staging)
        paths: dict[str, Path] = {}
        for role in module.ROLE_ORDER:
            path = staging / module.ROLE_FILE_NAMES[role]
            pq.write_table(
                _table(role, defect=table_defects.get(role)),
                path,
                version="2.6",
                compression=compression,
                row_group_size=2_048,
                use_dictionary=True,
                write_statistics=True,
                write_page_checksum=True,
            )
            paths[role] = path
        metadata_by_role: dict[str, tuple[int, int, list[str], str, str]] = {}
        for role, path in paths.items():
            parquet = pq.ParquetFile(path)
            try:
                metadata_by_role[role] = (
                    parquet.metadata.num_rows,
                    parquet.metadata.num_row_groups,
                    sorted(
                        {
                            str(
                                parquet.metadata.row_group(group)
                                .column(column)
                                .compression
                            ).upper()
                            for group in range(parquet.metadata.num_row_groups)
                            for column in range(parquet.metadata.num_columns)
                        }
                    ),
                    str(parquet.metadata.format_version),
                    str(parquet.metadata.created_by),
                )
            finally:
                parquet.close()
        if file_mutator is not None:
            file_mutator(paths)
        declarations: list[dict[str, object]] = []
        for role in module.ROLE_ORDER:
            path = paths[role]
            row_count, row_group_count, codecs, format_version, created_by = (
                metadata_by_role[role]
            )
            declarations.append(
                {
                    "role": role,
                    "format": "PARQUET",
                    "relative_path": module.ROLE_FILE_NAMES[role],
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "row_count": row_count,
                    "row_group_count": row_group_count,
                    "columns": list(module.ROLE_COLUMNS[role]),
                    "column_types": module.NORMALIZED_TYPES[role],
                    "normalized_schema_sha256": derive_normalized_schema_sha256(role),
                    "arrow_schema_sha256": derive_arrow_schema_sha256(role),
                    "compression": codecs,
                    "parquet_format_version": format_version,
                    "created_by": created_by,
                }
            )
        if declaration_mutator is not None:
            declaration_mutator(declarations)
        capture_batch_id = str(uuid.uuid4())
        reservation_id = str(uuid.uuid4())
        proposal_content_id = "2e52466d59c0f9c69b422815dd9103227522389bf63098912530cd8f8d92ec7c"
        derivation = {
            "capture_batch_id": capture_batch_id,
            "proposal_content_id": proposal_content_id,
            "reservation_id": reservation_id,
            "snapshot_reference_time_utc": "2026-08-02T03:00:00Z",
            "artifacts": declarations,
        }
        snapshot_id = derive_snapshot_id(derivation)
        manifest = {
            "schema_version": module.MANIFEST_SCHEMA,
            "evidence_class": evidence_class,
            "snapshot_id": snapshot_id,
            "capture_batch_id": capture_batch_id,
            "proposal_content_id": proposal_content_id,
            "reservation_id": reservation_id,
            "snapshot_reference_time_utc": "2026-08-02T03:00:00Z",
            "created_at_utc": "2026-08-06T00:00:00Z",
            "artifacts": declarations,
            "authorities": dict(module.MANIFEST_AUTHORITIES),
        }
        package = module.SYNTHETIC_PARQUET_ROOT / snapshot_id
        package.mkdir(parents=True, exist_ok=False)
        created.append(package)
        for role, source in paths.items():
            shutil.copyfile(source, package / module.ROLE_FILE_NAMES[role])
        manifest_path = package / "manifest.json"
        manifest_path.write_bytes(_json_bytes(manifest))
        return manifest_path, manifest

    yield build

    for directory in reversed(created):
        if directory.exists():
            for entry in tuple(directory.iterdir()):
                if entry.is_file():
                    entry.unlink()
            directory.rmdir()


def _verify(manifest_path: Path):
    return verify_synthetic_entsoe_parquet_package(
        contract_path=CONTRACT_PATH,
        package_manifest_path=manifest_path,
        reference_time_utc=REFERENCE,
    )


def test_contract_is_exact_zero_cost_and_synthetic_only() -> None:
    result = validate_parquet_streaming_contract(
        json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    )
    assert result["contract_content_id"] == module.CONTRACT_CONTENT_ID
    assert result["synthetic_streaming_implemented"] is True
    assert result["full_file_materialization_authorized"] is False
    assert result["real_artifact_admission_implemented"] is False
    assert result["databricks_connection_budget"] == 0
    assert result["warehouse_start_budget"] == 0


def test_exact_parquet_package_is_streamed_without_real_authority(
    package_factory,
    monkeypatch,
) -> None:
    manifest_path, _ = package_factory()
    original = module.read_stable_single_link_file

    def json_only_reader(path, **kwargs):
        assert Path(path).suffix == ".json"
        return original(path, **kwargs)

    monkeypatch.setattr(module, "read_stable_single_link_file", json_only_reader)
    result = _verify(manifest_path)
    assert result.total_rows == 16_389
    assert result.total_record_batches == 7
    assert result.total_uncompressed_bytes > 0
    assert result.receipt["authorities"][
        "synthetic_parquet_stream_integrity_verified"
    ] is True
    assert result.receipt["authorities"]["real_entsoe_values_opened"] is False
    assert result.receipt["authorities"]["incremental_quality_profile_verified"] is False
    assert all(item["record_batch_count"] > 0 for item in result.receipt["artifacts"])


def test_hash_tamper_is_rejected_before_parquet_parse(package_factory) -> None:
    manifest_path, _ = package_factory()
    artifact = manifest_path.parent / module.ROLE_FILE_NAMES["latest_values"]
    payload = bytearray(artifact.read_bytes())
    payload[len(payload) // 2] ^= 1
    artifact.write_bytes(payload)
    with pytest.raises(EntsoeParquetStreamingIntegrityError, match="artifact hash"):
        _verify(manifest_path)


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("null", "null value"),
        ("nonfinite", "non-finite value"),
        ("negative_revision", "negative revision"),
        ("float32", "Arrow schema"),
    ],
)
def test_batch_value_and_schema_defects_fail_closed(
    package_factory,
    defect: str,
    message: str,
) -> None:
    manifest_path, _ = package_factory(table_defects={"latest_values": defect})
    with pytest.raises(EntsoeParquetStreamingIntegrityError, match=message):
        _verify(manifest_path)


def test_unsupported_codec_is_rejected(package_factory) -> None:
    manifest_path, _ = package_factory(compression="gzip")
    with pytest.raises(EntsoeParquetStreamingIntegrityError, match="compression declaration"):
        _verify(manifest_path)


def test_footer_magic_is_checked_after_declared_hash(package_factory) -> None:
    def corrupt(paths: dict[str, Path]) -> None:
        path = paths["series_dimension"]
        payload = bytearray(path.read_bytes())
        payload[0:4] = b"NOPE"
        path.write_bytes(payload)

    manifest_path, _ = package_factory(file_mutator=corrupt)
    with pytest.raises(EntsoeParquetStreamingIntegrityError, match="leading magic"):
        _verify(manifest_path)


def test_declared_rows_row_groups_caps_path_and_inventory_fail_closed(
    package_factory,
) -> None:
    def wrong_rows(declarations: list[dict[str, object]]) -> None:
        declarations[0]["row_count"] = int(declarations[0]["row_count"]) + 1

    manifest_path, _ = package_factory(declaration_mutator=wrong_rows)
    with pytest.raises(EntsoeParquetStreamingIntegrityError, match="Parquet row count"):
        _verify(manifest_path)

    def wrong_groups(declarations: list[dict[str, object]]) -> None:
        declarations[1]["row_group_count"] = int(declarations[1]["row_group_count"]) + 1

    manifest_path, _ = package_factory(declaration_mutator=wrong_groups)
    with pytest.raises(EntsoeParquetStreamingIntegrityError, match="row-group count"):
        _verify(manifest_path)

    def wrong_version(declarations: list[dict[str, object]]) -> None:
        declarations[1]["parquet_format_version"] = "1.0"

    manifest_path, _ = package_factory(declaration_mutator=wrong_version)
    with pytest.raises(
        EntsoeParquetStreamingIntegrityError,
        match="Parquet format version",
    ):
        _verify(manifest_path)

    def empty_created_by(declarations: list[dict[str, object]]) -> None:
        declarations[1]["created_by"] = ""

    manifest_path, _ = package_factory(declaration_mutator=empty_created_by)
    with pytest.raises(EntsoeParquetStreamingIntegrityError, match="created_by"):
        _verify(manifest_path)

    def cap_breach(declarations: list[dict[str, object]]) -> None:
        declarations[0]["size_bytes"] = module.ROLE_MAX_FILE_BYTES["series_dimension"] + 1

    manifest_path, _ = package_factory(declaration_mutator=cap_breach)
    with pytest.raises(EntsoeParquetStreamingIntegrityError, match="declaration cap"):
        _verify(manifest_path)

    manifest_path, manifest = package_factory()
    manifest["artifacts"][0]["relative_path"] = "../escape.parquet"
    manifest_path.write_bytes(_json_bytes(manifest))
    with pytest.raises(EntsoeParquetStreamingIntegrityError, match="path is unsafe"):
        _verify(manifest_path)

    manifest_path, _ = package_factory()
    (manifest_path.parent / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(EntsoeParquetStreamingIntegrityError, match="package inventory"):
        _verify(manifest_path)


def test_manifest_real_class_and_fake_authority_are_rejected(package_factory) -> None:
    manifest_path, _ = package_factory(evidence_class="REAL_CAPTURE")
    with pytest.raises(EntsoeParquetStreamingIntegrityError, match="manifest evidence"):
        _verify(manifest_path)

    manifest_path, manifest = package_factory()
    manifest["authorities"]["real_artifact_receipt_admitted"] = True
    manifest_path.write_bytes(_json_bytes(manifest))
    with pytest.raises(EntsoeParquetStreamingIntegrityError, match="manifest authorities"):
        _verify(manifest_path)


def test_snapshot_directory_source_proof_and_post_scan_hash_are_bound(
    package_factory,
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path, manifest = package_factory()
    manifest["snapshot_id"] = "0" * 64
    manifest_path.write_bytes(_json_bytes(manifest))
    with pytest.raises(EntsoeParquetStreamingIntegrityError, match="snapshot ID derivation"):
        _verify(manifest_path)

    manifest_path, _ = package_factory()
    tampered = json.loads(module.D240_PROOF_MANIFEST_PATH.read_text(encoding="utf-8"))
    tampered["execution"]["network_call_count"] = 1
    tampered_path = tmp_path / "d240-tampered.json"
    tampered_path.write_bytes(_json_bytes(tampered))
    with pytest.raises(EntsoeParquetStreamingIntegrityError, match="D240 proof hash"):
        verify_synthetic_entsoe_parquet_package(
            contract_path=CONTRACT_PATH,
            package_manifest_path=manifest_path,
            reference_time_utc=REFERENCE,
            d240_proof_manifest_path=tampered_path,
        )

    manifest_path, _ = package_factory()
    tampered = json.loads(
        module.D241_REAL_SCHEMA_PROOF_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    tampered["execution"]["sql_statement_count"] = 1
    tampered_path = tmp_path / "d241-real-schema-tampered.json"
    tampered_path.write_bytes(_json_bytes(tampered))
    with pytest.raises(
        EntsoeParquetStreamingIntegrityError,
        match="D241 real schema proof hash",
    ):
        verify_synthetic_entsoe_parquet_package(
            contract_path=CONTRACT_PATH,
            package_manifest_path=manifest_path,
            reference_time_utc=REFERENCE,
            d241_real_schema_proof_manifest_path=tampered_path,
        )

    manifest_path, _ = package_factory()
    original_hash = module._hash_stream
    calls = 0

    def changed_second_hash(stream):
        nonlocal calls
        calls += 1
        digest, size = original_hash(stream)
        if calls == 2:
            return "0" * 64, size
        return digest, size

    monkeypatch.setattr(module, "_hash_stream", changed_second_hash)
    with pytest.raises(EntsoeParquetStreamingIntegrityError, match="post-scan hash"):
        _verify(manifest_path)


def test_hardlink_and_outside_package_are_rejected(
    package_factory,
    tmp_path: Path,
) -> None:
    manifest_path, _ = package_factory()
    artifact = manifest_path.parent / module.ROLE_FILE_NAMES["series_dimension"]
    link_root = ROOT / "build" / "d244-hardlink-test"
    link_root.mkdir(exist_ok=True)
    link = link_root / f"{uuid.uuid4()}.parquet"
    try:
        try:
            os.link(artifact, link)
        except OSError:
            pytest.skip("hard links unavailable on this workstation")
        with pytest.raises(EntsoeParquetStreamingIntegrityError, match="single-link"):
            _verify(manifest_path)
    finally:
        if link.exists():
            link.unlink()
        if link_root.exists() and not any(link_root.iterdir()):
            link_root.rmdir()

    manifest_path, _ = package_factory()
    outside = tmp_path / manifest_path.parent.name
    outside.mkdir()
    for source in manifest_path.parent.iterdir():
        shutil.copyfile(source, outside / source.name)
    with pytest.raises(EntsoeParquetStreamingIntegrityError, match="package path differs"):
        _verify(outside / "manifest.json")


def test_receipt_tamper_is_rejected(package_factory) -> None:
    manifest_path, manifest = package_factory()
    result = _verify(manifest_path)
    receipt = copy.deepcopy(result.receipt)
    receipt["artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(EntsoeParquetStreamingIntegrityError, match="receipt series_dimension.sha256"):
        validate_parquet_streaming_receipt(
            receipt,
            manifest,
            reference_time_utc=REFERENCE,
        )


def test_uncompressed_string_and_batch_byte_caps_fail_before_authority(
    package_factory,
    monkeypatch,
) -> None:
    manifest_path, manifest = package_factory()
    declaration = manifest["artifacts"][0]
    artifact = manifest_path.parent / module.ROLE_FILE_NAMES["series_dimension"]
    with pytest.raises(EntsoeParquetStreamingIntegrityError, match="uncompressed-byte budget"):
        module._scan_parquet_artifact(
            artifact,
            declaration,
            "series_dimension",
            uncompressed_budget=1,
        )

    schema = expected_arrow_schema("series_dimension")
    values = {
        field.name: ["x" * (module.MAX_STRING_VALUE_BYTES + 1)]
        for field in schema
    }
    batch = pa.RecordBatch.from_arrays(
        [pa.array(values[field.name], type=field.type) for field in schema],
        schema=schema,
    )
    with pytest.raises(EntsoeParquetStreamingIntegrityError, match="string value byte cap"):
        module._validate_batch_values(batch, "series_dimension")

    valid_batch = _table("series_dimension").to_batches()[0]
    monkeypatch.setattr(module, "MAX_RECORD_BATCH_BYTES", 1)
    parquet = pq.ParquetFile(artifact)
    try:
        with pytest.raises(
            EntsoeParquetStreamingIntegrityError,
            match="record-batch byte cap",
        ):
            module._inspect_and_scan_parquet(
                parquet,
                declaration,
                "series_dimension",
                footer_bytes=parquet.metadata.serialized_size,
                uncompressed_budget=module.MAX_TOTAL_UNCOMPRESSED_BYTES,
            )
    finally:
        parquet.close()
    assert valid_batch.num_rows == 3


def test_verifier_source_has_no_pandas_or_full_parquet_byte_read() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "import pandas" not in source
    assert ".to_pandas(" not in source
    assert "pq.read_table(" not in source
    assert "read_stable_single_link_file(artifact" not in source
