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

from pfc_shaping.validation import entsoe_cadence_package_binding as module
from pfc_shaping.validation import entsoe_incremental_quality as d245
from pfc_shaping.validation import entsoe_parquet_streaming_integrity as d244
from pfc_shaping.validation.entsoe_cadence_package_binding import (
    EntsoeCadencePackageBindingError,
    derive_sidecar_manifest_content_id,
    validate_cadence_package_binding_contract,
    verify_synthetic_entsoe_cadence_package,
)
from tests.test_entsoe_incremental_quality import (
    quality_package_factory as _d245_quality_package_factory,
)

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / ".planning" / "phases" / "14-lt-audit-remediation"
CONTRACT_PATH = PHASE / "ENTSOE-CADENCE-PACKAGE-BINDING-CONTRACT-V1.json"
REFERENCE = datetime(2026, 8, 6, 2, 0, tzinfo=timezone.utc)
START = "2026-08-01T00:00:00Z"
END = "2026-08-02T00:00:00Z"
METADATA_AS_OF = "2026-08-02T03:00:00Z"


@pytest.fixture
def quality_package_factory():
    yield from _d245_quality_package_factory.__wrapped__()


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
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture
def cadence_sidecar_factory():
    created: list[Path] = []

    def build(
        package_manifest_path: Path,
        quality_context_path: Path,
        fixture: dict[str, object],
        *,
        assessment_start: str = START,
        assessment_end: str = END,
        metadata_as_of: str = METADATA_AS_OF,
        regime_mutator=None,
        manifest_mutator=None,
    ) -> Path:
        dimension = fixture["series_dimension"]
        regimes = pd.DataFrame(
            [
                {
                    "series_id": row.series_id,
                    "valid_from_utc": pd.Timestamp("2026-07-01T00:00:00Z"),
                    "valid_to_utc": pd.NaT,
                    "native_resolution": row.native_resolution,
                    "source_document_id": f"synthetic/{index}",
                    "source_locator": f"https://example.com/synthetic/{index}",
                    "owner_confirmed_at_utc": pd.Timestamp(metadata_as_of),
                }
                for index, row in enumerate(dimension.itertuples(index=False), start=1)
            ],
            columns=list(module.SIDECAR_COLUMNS),
        )
        regimes["valid_from_utc"] = pd.to_datetime(regimes["valid_from_utc"], utc=True)
        regimes["valid_to_utc"] = pd.to_datetime(regimes["valid_to_utc"], utc=True)
        regimes["owner_confirmed_at_utc"] = pd.to_datetime(
            regimes["owner_confirmed_at_utc"], utc=True
        )
        if regime_mutator is not None:
            regime_mutator(regimes)

        staging = ROOT / "build" / "d270-fixture-staging" / str(uuid.uuid4())
        staging.mkdir(parents=True)
        created.append(staging)
        artifact = staging / module.FILE_NAME
        table = pa.Table.from_pandas(
            regimes,
            schema=module.expected_sidecar_schema(),
            preserve_index=False,
            safe=True,
        ).replace_schema_metadata(None)
        pq.write_table(
            table,
            artifact,
            version="2.6",
            compression="zstd",
            row_group_size=17,
            use_dictionary=True,
            write_statistics=True,
            write_page_checksum=True,
        )
        parquet = pq.ParquetFile(artifact)
        try:
            declaration = {
                "role": module.ROLE,
                "format": "PARQUET",
                "relative_path": module.FILE_NAME,
                "sha256": _sha256_file(artifact),
                "size_bytes": artifact.stat().st_size,
                "row_count": parquet.metadata.num_rows,
                "row_group_count": parquet.metadata.num_row_groups,
                "columns": list(module.SIDECAR_COLUMNS),
                "column_types": module.SIDECAR_TYPES,
                "arrow_schema_sha256": module.derive_sidecar_arrow_schema_sha256(),
                "compression": sorted(
                    {
                        str(parquet.metadata.row_group(group).column(column).compression).upper()
                        for group in range(parquet.metadata.num_row_groups)
                        for column in range(parquet.metadata.num_columns)
                    }
                ),
                "parquet_format_version": str(parquet.metadata.format_version),
                "created_by": str(parquet.metadata.created_by),
            }
        finally:
            parquet.close()
        package = json.loads(package_manifest_path.read_text(encoding="utf-8"))
        context = json.loads(quality_context_path.read_text(encoding="utf-8"))
        manifest = {
            "schema_version": module.MANIFEST_SCHEMA,
            "evidence_class": module.EVIDENCE_CLASS,
            "snapshot_id": package["snapshot_id"],
            "base_package_manifest_content_id": d244.canonical_sha256(package),
            "quality_context_content_id": d245.derive_quality_context_content_id(context),
            "assessment_start_utc": assessment_start,
            "assessment_end_utc": assessment_end,
            "metadata_as_of_utc": metadata_as_of,
            "created_at_utc": metadata_as_of,
            "artifact": declaration,
            "authorities": dict(module.AUTHORITIES),
        }
        if manifest_mutator is not None:
            manifest_mutator(manifest)
        content_id = derive_sidecar_manifest_content_id(manifest)
        sidecar = module.SIDECAR_ROOT / content_id
        sidecar.mkdir(parents=True, exist_ok=False)
        created.append(sidecar)
        shutil.copyfile(artifact, sidecar / module.FILE_NAME)
        manifest_path = sidecar / "manifest.json"
        manifest_path.write_bytes(_json_bytes(manifest))
        return manifest_path

    yield build

    for directory in reversed(created):
        if directory.exists():
            for entry in tuple(directory.iterdir()):
                if entry.is_file():
                    entry.unlink()
            directory.rmdir()


def _verify(package: Path, context: Path, sidecar: Path):
    return verify_synthetic_entsoe_cadence_package(
        contract_path=CONTRACT_PATH,
        package_manifest_path=package,
        quality_context_path=context,
        cadence_manifest_path=sidecar,
        reference_time_utc=REFERENCE,
    )


def test_contract_is_exact_streamed_zero_cost_and_no_authority() -> None:
    result = validate_cadence_package_binding_contract(
        json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    )
    assert result["contract_content_id"] == module.CONTRACT_CONTENT_ID
    assert result["bounded_streaming_implemented"] is True
    assert result["base_package_role_count"] == 3
    assert result["cadence_value_columns_opened"] is False
    assert result["databricks_connection_budget"] == 0
    assert result["production_authorized"] is False


def test_exact_cadence_sidecar_is_bound_to_the_passing_d245_package(
    quality_package_factory,
    cadence_sidecar_factory,
) -> None:
    package, context, fixture = quality_package_factory()
    sidecar = cadence_sidecar_factory(package, context, fixture)
    before = set(module.STAGING_ROOT.iterdir()) if module.STAGING_ROOT.exists() else set()
    result = _verify(package, context, sidecar)
    after = set(module.STAGING_ROOT.iterdir()) if module.STAGING_ROOT.exists() else set()
    assert before == after
    assert result.summary["status"] == module.PASS_STATUS
    assert result.summary["series_count"] == 33
    assert result.summary["regime_count"] == 33
    assert result.summary["expected_target_count"] == 792
    assert result.summary["observed_target_count"] == 792
    assert result.summary["missing_native_slot_count"] == 0
    assert result.summary["base_d245_incremental_quality_passed"] is True
    assert result.summary["base_value_columns_opened_by_cadence_pass"] is False
    assert result.summary["authorities"] == module.AUTHORITIES
    assert result.staging_cleaned is True


def test_explicit_window_reports_leading_and_trailing_missing_slots(
    quality_package_factory,
    cadence_sidecar_factory,
) -> None:
    package, context, fixture = quality_package_factory()
    sidecar = cadence_sidecar_factory(
        package,
        context,
        fixture,
        assessment_start="2026-07-31T23:00:00Z",
        assessment_end="2026-08-02T01:00:00Z",
    )
    result = _verify(package, context, sidecar)
    assert result.summary["status"] == module.GAP_STATUS
    assert result.summary["expected_target_count"] == 858
    assert result.summary["observed_target_count"] == 792
    assert result.summary["missing_native_slot_count"] == 66


def test_metadata_as_of_may_be_off_delivery_grid(
    quality_package_factory,
    cadence_sidecar_factory,
) -> None:
    package, context, fixture = quality_package_factory()
    sidecar = cadence_sidecar_factory(
        package,
        context,
        fixture,
        metadata_as_of="2026-08-02T03:07:00Z",
    )
    result = _verify(package, context, sidecar)
    assert result.summary["status"] == module.PASS_STATUS
    assert result.summary["metadata_as_of_utc"] == "2026-08-02T03:07:00Z"


def test_sidecar_cannot_bind_another_base_manifest(
    quality_package_factory,
    cadence_sidecar_factory,
) -> None:
    package, context, fixture = quality_package_factory()
    sidecar = cadence_sidecar_factory(
        package,
        context,
        fixture,
        manifest_mutator=lambda manifest: manifest.update(
            {"base_package_manifest_content_id": "0" * 64}
        ),
    )
    with pytest.raises(EntsoeCadencePackageBindingError, match="base manifest binding"):
        _verify(package, context, sidecar)


def test_fake_real_authority_is_rejected(
    quality_package_factory,
    cadence_sidecar_factory,
) -> None:
    package, context, fixture = quality_package_factory()

    def mutate(manifest):
        manifest["authorities"]["model_input"] = True

    sidecar = cadence_sidecar_factory(package, context, fixture, manifest_mutator=mutate)
    with pytest.raises(EntsoeCadencePackageBindingError, match="sidecar authorities"):
        _verify(package, context, sidecar)


def test_sidecar_artifact_tamper_is_rejected_before_use(
    quality_package_factory,
    cadence_sidecar_factory,
) -> None:
    package, context, fixture = quality_package_factory()
    sidecar = cadence_sidecar_factory(package, context, fixture)
    artifact = sidecar.parent / module.FILE_NAME
    payload = bytearray(artifact.read_bytes())
    payload[len(payload) // 2] ^= 1
    artifact.write_bytes(bytes(payload))
    with pytest.raises(EntsoeCadencePackageBindingError, match="artifact hash"):
        _verify(package, context, sidecar)


def test_every_dimension_series_requires_its_own_regime(
    quality_package_factory,
    cadence_sidecar_factory,
) -> None:
    package, context, fixture = quality_package_factory()
    sidecar = cadence_sidecar_factory(
        package,
        context,
        fixture,
        regime_mutator=lambda regimes: regimes.drop(index=regimes.index[-1], inplace=True),
    )
    with pytest.raises(EntsoeCadencePackageBindingError, match="series coverage differs"):
        _verify(package, context, sidecar)


def test_source_locator_must_be_clean_https(
    quality_package_factory,
    cadence_sidecar_factory,
) -> None:
    package, context, fixture = quality_package_factory()

    def mutate(regimes: pd.DataFrame) -> None:
        regimes.loc[0, "source_locator"] = "https://example.com/source#fragment"

    sidecar = cadence_sidecar_factory(
        package,
        context,
        fixture,
        regime_mutator=mutate,
    )
    with pytest.raises(EntsoeCadencePackageBindingError, match="clean HTTPS"):
        _verify(package, context, sidecar)


def test_regime_transition_must_close_previous_native_grid(
    quality_package_factory,
    cadence_sidecar_factory,
) -> None:
    package, context, fixture = quality_package_factory()

    def mutate(regimes: pd.DataFrame) -> None:
        first = regimes.iloc[0].copy()
        regimes.loc[0, "valid_to_utc"] = pd.Timestamp("2026-08-01T12:30:00Z")
        first["valid_from_utc"] = pd.Timestamp("2026-08-01T12:30:00Z")
        first["valid_to_utc"] = pd.Timestamp("2026-09-01T00:00:00Z")
        regimes.loc[len(regimes)] = first

    sidecar = cadence_sidecar_factory(
        package,
        context,
        fixture,
        regime_mutator=mutate,
    )
    with pytest.raises(EntsoeCadencePackageBindingError, match="transition does not close"):
        _verify(package, context, sidecar)


def test_source_contains_no_pandas_and_cadence_scan_selects_no_value_column() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "import pandas" not in source
    assert '("series_id", "target_ts_utc")' in source
    assert '("series_id", "group_name", "native_resolution")' in source
    assert 'columns=("value"' not in source


def test_contract_mutations_cannot_relax_zero_cost_or_upsampling() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    for mutator, match in (
        (
            lambda value: value["execution"].update({"databricks_statement_budget": 1}),
            "execution budgets",
        ),
        (
            lambda value: value["cadence_rules"].update({"implicit_upsampling_authorized": True}),
            "cadence rules",
        ),
    ):
        mutated = copy.deepcopy(contract)
        mutator(mutated)
        with pytest.raises(EntsoeCadencePackageBindingError, match=match):
            validate_cadence_package_binding_contract(mutated)
