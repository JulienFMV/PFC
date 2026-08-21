from __future__ import annotations

import copy
import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pfc_shaping.validation import entsoe_incremental_quality as module
from pfc_shaping.validation import entsoe_parquet_streaming_integrity as d244
from pfc_shaping.validation.entsoe_incremental_quality import (
    EntsoeIncrementalQualityError,
    derive_quality_context_content_id,
    profile_synthetic_entsoe_incremental_quality,
    validate_incremental_quality_contract,
)
from tests.test_entsoe_normalized_quality import SNAPSHOT, _fixture

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / ".planning" / "phases" / "14-lt-audit-remediation"
CONTRACT_PATH = PHASE / "ENTSOE-INCREMENTAL-QUALITY-PROFILE-CONTRACT-V1.json"
REFERENCE = datetime(2026, 8, 6, 2, 0, tzinfo=timezone.utc)


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
        while chunk := stream.read(d244.HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _arrow_table(frame: pd.DataFrame, role: str) -> pa.Table:
    return pa.Table.from_pandas(
        frame,
        schema=d244.expected_arrow_schema(role),
        preserve_index=False,
        safe=True,
    ).replace_schema_metadata(None)


@pytest.fixture
def quality_package_factory():
    created: list[Path] = []

    def build(
        *,
        fixture_mutator=None,
        context_mutator=None,
        evidence_class: str = "SYNTHETIC_TEST_ONLY",
    ) -> tuple[Path, Path, dict[str, object]]:
        fixture = copy.deepcopy(_fixture())
        if fixture_mutator is not None:
            fixture_mutator(fixture)
        staging = ROOT / "build" / "d245-fixture-staging" / str(uuid.uuid4())
        staging.mkdir(parents=True)
        created.append(staging)
        declarations: list[dict[str, object]] = []
        paths: dict[str, Path] = {}
        for role in d244.ROLE_ORDER:
            path = staging / d244.ROLE_FILE_NAMES[role]
            pq.write_table(
                _arrow_table(fixture[role], role),
                path,
                version="2.6",
                compression="zstd",
                row_group_size=257,
                use_dictionary=True,
                write_statistics=True,
                write_page_checksum=True,
            )
            paths[role] = path
            parquet = pq.ParquetFile(path)
            try:
                codecs = sorted(
                    {
                        str(
                            parquet.metadata.row_group(group)
                            .column(column)
                            .compression
                        ).upper()
                        for group in range(parquet.metadata.num_row_groups)
                        for column in range(parquet.metadata.num_columns)
                    }
                )
                declarations.append(
                    {
                        "role": role,
                        "format": "PARQUET",
                        "relative_path": d244.ROLE_FILE_NAMES[role],
                        "sha256": _sha256_file(path),
                        "size_bytes": path.stat().st_size,
                        "row_count": parquet.metadata.num_rows,
                        "row_group_count": parquet.metadata.num_row_groups,
                        "columns": list(d244.ROLE_COLUMNS[role]),
                        "column_types": d244.NORMALIZED_TYPES[role],
                        "normalized_schema_sha256": d244.derive_normalized_schema_sha256(role),
                        "arrow_schema_sha256": d244.derive_arrow_schema_sha256(role),
                        "compression": codecs,
                        "parquet_format_version": str(parquet.metadata.format_version),
                        "created_by": str(parquet.metadata.created_by),
                    }
                )
            finally:
                parquet.close()
        capture_batch_id = str(uuid.uuid4())
        reservation_id = str(uuid.uuid4())
        proposal_content_id = "4fdb6aa976c5dc81b6f773bd614486e2e688380190829491fcbc8974957bcda6"
        snapshot_text = SNAPSHOT.strftime("%Y-%m-%dT%H:%M:%SZ")
        derivation = {
            "capture_batch_id": capture_batch_id,
            "proposal_content_id": proposal_content_id,
            "reservation_id": reservation_id,
            "snapshot_reference_time_utc": snapshot_text,
            "artifacts": declarations,
        }
        snapshot_id = d244.derive_snapshot_id(derivation)
        manifest = {
            "schema_version": d244.MANIFEST_SCHEMA,
            "evidence_class": "SYNTHETIC_TEST_ONLY",
            "snapshot_id": snapshot_id,
            "capture_batch_id": capture_batch_id,
            "proposal_content_id": proposal_content_id,
            "reservation_id": reservation_id,
            "snapshot_reference_time_utc": snapshot_text,
            "created_at_utc": snapshot_text,
            "artifacts": declarations,
            "authorities": dict(d244.MANIFEST_AUTHORITIES),
        }
        package = d244.SYNTHETIC_PARQUET_ROOT / snapshot_id
        package.mkdir(parents=True, exist_ok=False)
        created.append(package)
        for role, source in paths.items():
            shutil.copyfile(source, package / d244.ROLE_FILE_NAMES[role])
        manifest_path = package / "manifest.json"
        manifest_path.write_bytes(_json_bytes(manifest))
        context = {
            "schema_version": module.CONTEXT_SCHEMA,
            "evidence_class": evidence_class,
            "snapshot_id": snapshot_id,
            "snapshot_reference_time_utc": snapshot_text,
            "sign_semantics_by_series": fixture["sign_semantics_by_series"],
            "backfill_status_by_series": fixture["backfill_status_by_series"],
            "authorities": dict(module.CONTEXT_AUTHORITIES),
        }
        if context_mutator is not None:
            context_mutator(context)
        context_id = derive_quality_context_content_id(context)
        context_dir = module.QUALITY_CONTEXT_ROOT / context_id
        context_dir.mkdir(parents=True, exist_ok=False)
        created.append(context_dir)
        context_path = context_dir / "context.json"
        context_path.write_bytes(_json_bytes(context))
        return manifest_path, context_path, fixture

    yield build

    for directory in reversed(created):
        if directory.exists():
            for entry in tuple(directory.iterdir()):
                if entry.is_file():
                    entry.unlink()
            directory.rmdir()


def _profile(manifest_path: Path, context_path: Path):
    return profile_synthetic_entsoe_incremental_quality(
        contract_path=CONTRACT_PATH,
        package_manifest_path=manifest_path,
        quality_context_path=context_path,
        reference_time_utc=REFERENCE,
    )


def test_contract_is_exact_bounded_zero_cost_and_no_real_authority() -> None:
    result = validate_incremental_quality_contract(
        json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    )
    assert result["contract_content_id"] == module.CONTRACT_CONTENT_ID
    assert result["bounded_sqlite_spill_implemented"] is True
    assert result["pandas_authorized"] is False
    assert result["real_artifact_admission_implemented"] is False
    assert result["databricks_connection_budget"] == 0


def test_incremental_profile_matches_d239_fixture_without_persisting_values(
    quality_package_factory,
) -> None:
    manifest_path, context_path, _ = quality_package_factory()
    before = set(module.STAGING_ROOT.iterdir()) if module.STAGING_ROOT.exists() else set()
    result = _profile(manifest_path, context_path)
    after = set(module.STAGING_ROOT.iterdir()) if module.STAGING_ROOT.exists() else set()
    assert before == after
    assert result.staging_cleaned is True
    assert result.summary["status"] == "PASS_SYNTHETIC_INCREMENTAL_QUALITY_MODEL_NO_GO"
    assert result.summary["dataset"]["dimension_row_count"] == 33
    assert result.summary["dataset"]["latest_row_count"] == 792
    assert result.summary["dataset"]["vintage_row_count"] == 792
    assert result.summary["checks"]["primary_keys_unique"] is True
    assert result.summary["checks"]["dimension_join_coverage"] == 1.0
    assert result.summary["checks"]["required_groups_present"] is True
    assert result.summary["checks"]["required_borders_by_crossborder_group_present"] is True
    assert result.summary["checks"]["required_generation_breakdown_present"] is True
    assert result.summary["rates"]["latest_to_vintage_overlap_coverage_ratio"] == 1.0
    assert result.summary["rates"]["minimum_grid_completeness_ratio"] == 1.0
    assert result.summary["coverage"]["minimum_complete_utc_days"] == 1
    assert result.summary["sqlite"]["raw_numeric_values_stored"] is False
    assert result.sqlite_peak_bytes <= module.MAX_DATABASE_BYTES
    assert all(value == 0 for value in result.summary["execution"].values())
    assert result.summary["authorities"]["model_input_authorized"] is False
    assert len(result.series_profile) == 33
    assert all("series_id" not in row and "value" not in row for row in result.series_profile)
    codes = {finding["code"] for finding in result.summary["findings"]}
    assert "SYNTHETIC_FIXTURE_NOT_EMPIRICAL_EVIDENCE" in codes
    assert "D241_REAL_SCHEMA_MAPPING_REMAINS_INCOMPATIBLE" in codes
    assert "D243_REAL_SERIES_INVENTORY_NOT_EXECUTED" in codes
    assert "SEASONAL_COMPLETE_DAY_THRESHOLD_NOT_MET" in codes
    assert "BACKFILLED_SERIES_NOT_RETROACTIVE_PIT_EVIDENCE" in codes


def test_duplicate_grains_are_counted_with_rates_and_fail(quality_package_factory) -> None:
    def mutate(fixture: dict[str, object]) -> None:
        latest = fixture["latest_values"]
        fixture["latest_values"] = pd.concat([latest, latest.iloc[[0]]], ignore_index=True)

    manifest_path, context_path, _ = quality_package_factory(fixture_mutator=mutate)
    result = _profile(manifest_path, context_path)
    assert result.summary["status"].startswith("FAIL_")
    assert result.summary["failure_counts"]["latest_duplicate_key_count"] == 1
    assert result.summary["rates"]["latest_duplicate_key_rate"] == pytest.approx(1 / 793)


def test_orphan_and_missing_dimension_series_fail_with_exact_counts(
    quality_package_factory,
) -> None:
    def mutate(fixture: dict[str, object]) -> None:
        latest = fixture["latest_values"].copy()
        latest.loc[0, "series_id"] = "orphan_series"
        fixture["latest_values"] = latest

    manifest_path, context_path, _ = quality_package_factory(fixture_mutator=mutate)
    result = _profile(manifest_path, context_path)
    assert result.summary["status"].startswith("FAIL_")
    assert result.summary["failure_counts"]["latest_orphan_row_count"] == 1
    assert result.summary["rates"]["latest_dimension_join_coverage_ratio"] == pytest.approx(791 / 792)


def test_gap_is_reported_as_rate_without_becoming_empirical_authority(
    quality_package_factory,
) -> None:
    def mutate(fixture: dict[str, object]) -> None:
        fixture["latest_values"] = fixture["latest_values"].drop(index=1).reset_index(drop=True)

    manifest_path, context_path, _ = quality_package_factory(fixture_mutator=mutate)
    result = _profile(manifest_path, context_path)
    assert result.summary["rates"]["minimum_grid_completeness_ratio"] < 1.0
    codes = {finding["code"] for finding in result.summary["findings"]}
    assert "NATIVE_GRID_COMPLETENESS_GAPS_PRESENT" in codes
    assert result.summary["authorities"]["model_input_authorized"] is False


def test_latest_last_vintage_mismatch_is_hash_detected(quality_package_factory) -> None:
    def mutate(fixture: dict[str, object]) -> None:
        latest = fixture["latest_values"].copy()
        latest.loc[0, "value"] = float(latest.loc[0, "value"]) + 1.0
        fixture["latest_values"] = latest

    manifest_path, context_path, _ = quality_package_factory(fixture_mutator=mutate)
    result = _profile(manifest_path, context_path)
    assert result.summary["status"].startswith("FAIL_")
    assert result.summary["failure_counts"]["latest_last_vintage_mismatch_count"] == 1


def test_revision_and_load_chronology_violation_is_detected(quality_package_factory) -> None:
    def mutate(fixture: dict[str, object]) -> None:
        vintage = fixture["vintage_values"].copy()
        row = vintage.iloc[0].copy()
        row["as_of_utc"] = row["as_of_utc"] + pd.Timedelta(minutes=30)
        row["revision_number"] = 0
        row["meta_load_timestamp_utc"] = row["as_of_utc"]
        fixture["vintage_values"] = pd.concat(
            [vintage, row.to_frame().T],
            ignore_index=True,
        )
        for column in ("target_ts_utc", "as_of_utc", "meta_load_timestamp_utc"):
            fixture["vintage_values"][column] = pd.to_datetime(
                fixture["vintage_values"][column],
                utc=True,
            )
        fixture["vintage_values"]["revision_number"] = fixture["vintage_values"][
            "revision_number"
        ].astype("int64")

    manifest_path, context_path, _ = quality_package_factory(fixture_mutator=mutate)
    result = _profile(manifest_path, context_path)
    assert result.summary["failure_counts"]["vintage_chronology_violation_count"] == 1
    assert result.summary["status"].startswith("FAIL_")


def test_availability_sign_and_snapshot_violations_fail_closed(
    quality_package_factory,
) -> None:
    def mutate(fixture: dict[str, object]) -> None:
        dimension = fixture["series_dimension"]
        pre_series = dimension.loc[
            dimension["group_name"].eq("day_ahead_prices"), "series_id"
        ].iloc[0]
        vintage = fixture["vintage_values"].copy()
        index = vintage.index[vintage["series_id"].eq(pre_series)][0]
        vintage.loc[index, "as_of_utc"] = vintage.loc[index, "target_ts_utc"] + pd.Timedelta(hours=1)
        vintage.loc[index, "meta_load_timestamp_utc"] = vintage.loc[index, "as_of_utc"]
        fixture["vintage_values"] = vintage
        load_series = dimension.loc[
            dimension["group_name"].eq("actual_load"), "series_id"
        ].iloc[0]
        latest = fixture["latest_values"].copy()
        load_index = latest.index[latest["series_id"].eq(load_series)][0]
        latest.loc[load_index, "value"] = -1.0
        latest.loc[load_index, "meta_load_timestamp_utc"] = SNAPSHOT + pd.Timedelta(hours=1)
        fixture["latest_values"] = latest

    manifest_path, context_path, _ = quality_package_factory(fixture_mutator=mutate)
    result = _profile(manifest_path, context_path)
    assert result.summary["failure_counts"]["vintage_pre_target_availability_violation_count"] == 1
    assert result.summary["failure_counts"]["latest_sign_violation_count"] == 1
    assert result.summary["failure_counts"]["latest_load_after_snapshot_count"] == 1
    assert result.summary["status"].startswith("FAIL_")


def test_context_mapping_real_class_authority_and_path_fail_closed(
    quality_package_factory,
    tmp_path: Path,
) -> None:
    def remove_mapping(context: dict[str, object]) -> None:
        context["sign_semantics_by_series"].pop(next(iter(context["sign_semantics_by_series"])))

    manifest_path, context_path, _ = quality_package_factory(context_mutator=remove_mapping)
    result = _profile(manifest_path, context_path)
    assert result.summary["failure_counts"]["context_mapping_mismatch_count"] == 1
    assert result.summary["status"].startswith("FAIL_")

    manifest_path, context_path, _ = quality_package_factory(evidence_class="REAL_CAPTURE")
    with pytest.raises(EntsoeIncrementalQualityError, match="context evidence"):
        _profile(manifest_path, context_path)

    def escalate(context: dict[str, object]) -> None:
        context["authorities"]["model_input_authorized"] = True

    manifest_path, context_path, _ = quality_package_factory(context_mutator=escalate)
    with pytest.raises(EntsoeIncrementalQualityError, match="context authorities"):
        _profile(manifest_path, context_path)

    manifest_path, context_path, _ = quality_package_factory()
    outside = tmp_path / "context.json"
    shutil.copyfile(context_path, outside)
    with pytest.raises(EntsoeIncrementalQualityError, match="content-addressed path"):
        _profile(manifest_path, outside)


def test_bound_source_tamper_is_rejected(quality_package_factory, tmp_path: Path, monkeypatch) -> None:
    manifest_path, context_path, _ = quality_package_factory()
    proof = json.loads(module.D244_PROOF_MANIFEST_PATH.read_text(encoding="utf-8"))
    proof["execution"]["network_call_count"] = 1
    tampered = tmp_path / "d244-proof.json"
    tampered.write_bytes(_json_bytes(proof))
    monkeypatch.setattr(module, "D244_PROOF_MANIFEST_PATH", tampered)
    with pytest.raises(EntsoeIncrementalQualityError, match="D244 proof hash"):
        _profile(manifest_path, context_path)


def test_sqlite_staging_is_cleaned_on_mid_scan_failure(
    quality_package_factory,
    monkeypatch,
) -> None:
    manifest_path, context_path, _ = quality_package_factory()
    before = set(module.STAGING_ROOT.iterdir()) if module.STAGING_ROOT.exists() else set()

    def fail(*args, **kwargs):
        raise EntsoeIncrementalQualityError("injected consumer failure")

    monkeypatch.setattr(module, "_consume_fact", fail)
    with pytest.raises(EntsoeIncrementalQualityError, match="injected consumer failure"):
        _profile(manifest_path, context_path)
    after = set(module.STAGING_ROOT.iterdir()) if module.STAGING_ROOT.exists() else set()
    assert before == after


def test_verifier_source_has_no_pandas_or_whole_table_materialization() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "import pandas" not in source
    assert ".to_pandas(" not in source
    assert "pq.read_table(" not in source
    assert "read_bytes()" not in source
    assert "row_fingerprint BLOB" in source
    assert "value REAL" not in source
