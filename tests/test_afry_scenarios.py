from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pfc_shaping.data.afry_scenarios import (
    ANNUAL_FACT_SCHEMA,
    CATALOG_DECISION_STATE,
    CATALOG_INTEGRITY_GATE_NAMES,
    CATALOG_INTEGRITY_PASS,
    CATALOG_SCHEMA,
    HOUR_SLOT_SHAPE_SCHEMA,
    HOURLY_SCHEMA,
    REGISTRATION_SCHEMA,
    RELEASE_ID,
    SCENARIOS,
    STRUCTURAL_DECLARED_NULLABLE_FIELDS,
    STRUCTURAL_REQUIRED_NON_NULL_FIELDS,
    STRUCTURAL_SCHEMA,
    AfryAnnualContract,
    AfryDataContractError,
    AfrySource,
    AfrySupportingDocument,
    _write_frame,
    build_structural_benchmark,
    canonical_json_bytes,
    capture_verified_source,
    catalog_content_id,
    file_sha256,
    load_registration,
    read_verified_artifact_bytes,
    runtime_tree_receipt,
    validate_annual_semantics,
    validate_structural_contract,
    verify_catalog_bundle,
    verify_semantic_contract,
    verify_source,
    verify_supporting_document,
)
from scripts.materialize_afry_scenario_catalog import _require_canonical_workspace
from tests.test_afry_scenario_extraction import _semantic_fixture

REGISTRATION = Path(
    ".planning/phases/14-lt-audit-remediation/AFRY-CH-2026-Q2-SOURCE-REGISTRATION.json"
)


def _registration_payload() -> dict[str, object]:
    return json.loads(REGISTRATION.read_text(encoding="utf-8"))


def _write_registration(workspace: Path, payload: dict[str, object]) -> Path:
    path = workspace / "registration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _zip_source(path: Path, members: dict[str, bytes]) -> AfrySource:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    return AfrySource(
        role="annual_scenarios",
        path=path,
        sha256=file_sha256(path),
        publisher_release_date="2026-06-26",
    )


def test_real_registration_is_restricted_repo_local_and_annual() -> None:
    registration = load_registration(REGISTRATION, workspace=Path.cwd())

    assert registration.release_id == RELEASE_ID
    assert registration.cadence == "annual"
    assert registration.expected_arrival_month == 7
    assert registration.schedule_timezone == "Europe/Zurich"
    assert registration.hourly_contract.expected_total_rows == 3_854_400
    assert registration.hourly_contract.negative_zero_tolerance_eur_mwh == pytest.approx(5e-6)
    assert registration.annual_contract.delivery_years == tuple(range(2027, 2061))
    assert len(registration.hourly_contract.sheets) == 8
    assert {source.role for source in registration.sources} == {
        "annual_scenarios",
        "hourly_prices",
    }
    for source in registration.sources:
        source.path.resolve().relative_to(Path.cwd().resolve())
        assert source.path.suffix.lower() == ".xlsx"
        assert source.path.is_file()
        assert file_sha256(source.path) == source.sha256
        assert source.declared_money_basis
    assert {document.role for document in registration.supporting_documents} == {
        "commodity_and_modelling_annex",
        "quarterly_update_note",
    }
    for document in registration.supporting_documents:
        document.path.resolve().relative_to(Path.cwd().resolve())
        assert document.path.suffix.lower() == ".pdf"
        assert document.path.is_file()
        assert file_sha256(document.path) == document.sha256
    assert registration.semantic_contract_path.is_file()
    assert file_sha256(registration.semantic_contract_path) == registration.semantic_contract_sha256
    assert verify_semantic_contract(registration)["model_input_authorized"] is False


def test_materializer_cli_requires_exact_canonical_workspace() -> None:
    assert _require_canonical_workspace() == Path(r"C:\Users\jbattaglia\PFC_LT")


def test_materializer_cli_rejects_noncanonical_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(AfryDataContractError, match="cwd must be exactly"):
        _require_canonical_workspace()


@pytest.mark.parametrize(
    "field",
    [
        "raw_or_derived_values_allowed_in_git",
        "rag_indexing_allowed",
        "monthly_level_authority",
        "scenario_mapping_approved",
        "model_input_authorized",
        "production_authorized",
    ],
)
def test_registration_rejects_authority_flip(tmp_path: Path, field: str) -> None:
    payload = _registration_payload()
    payload[field] = True
    path = _write_registration(tmp_path, payload)

    with pytest.raises(AfryDataContractError, match=field):
        load_registration(path, workspace=tmp_path)


def test_registration_rejects_non_july_schedule(tmp_path: Path) -> None:
    payload = _registration_payload()
    payload["release_schedule"]["expected_arrival_month"] = 8  # type: ignore[index]
    path = _write_registration(tmp_path, payload)

    with pytest.raises(AfryDataContractError, match="July"):
        load_registration(path, workspace=tmp_path)


def test_registration_rejects_missing_hourly_content_contract(tmp_path: Path) -> None:
    payload = _registration_payload()
    payload.pop("content_contract")
    path = _write_registration(tmp_path, payload)

    with pytest.raises(AfryDataContractError, match="missing hourly content contract"):
        load_registration(path, workspace=tmp_path)


def test_registration_rejects_missing_annual_content_contract(tmp_path: Path) -> None:
    payload = _registration_payload()
    payload["content_contract"].pop("annual_scenarios")  # type: ignore[index]
    path = _write_registration(tmp_path, payload)

    with pytest.raises(AfryDataContractError, match="missing annual content contract"):
        load_registration(path, workspace=tmp_path)


def test_registration_rejects_missing_supporting_documents(tmp_path: Path) -> None:
    payload = _registration_payload()
    payload["supporting_documents"] = []
    path = _write_registration(tmp_path, payload)

    with pytest.raises(AfryDataContractError, match="QUN and commodity/modelling annex"):
        load_registration(path, workspace=tmp_path)


def test_registration_rejects_missing_semantic_contract_declaration(tmp_path: Path) -> None:
    payload = _registration_payload()
    payload.pop("semantic_contract")
    path = _write_registration(tmp_path, payload)

    with pytest.raises(AfryDataContractError, match="contain an AFRY semantic contract"):
        load_registration(path, workspace=tmp_path)


def test_registration_drives_a_future_annual_release_without_code_change(tmp_path: Path) -> None:
    payload = _registration_payload()
    payload["release_id"] = "AFRY_CH_2027_Q2_V100"
    payload["first_observed_at_utc"] = "2027-07-30T10:00:00Z"
    for source in payload["sources"]:  # type: ignore[union-attr]
        source["publisher_release_date"] = "2027-07-24"
    payload["content_contract"]["hourly_prices"]["sheets"] = [  # type: ignore[index]
        {
            "title": f"Hourly WP {scenario} (2028-2061 anchors)",
            "scenario_vendor": scenario,
            "delivery_years": [2028, 2061],
        }
        for scenario in ("High", "Central", "Low", "Delayed Transition")
    ]
    payload["content_contract"]["annual_scenarios"]["delivery_years"] = list(  # type: ignore[index]
        range(2028, 2062)
    )
    path = _write_registration(tmp_path, payload)

    registration = load_registration(path, workspace=tmp_path)

    assert registration.release_id == "AFRY_CH_2027_Q2_V100"
    assert registration.hourly_contract.expected_total_rows == 350_400
    assert registration.annual_contract.delivery_years == tuple(range(2028, 2062))


def test_registration_rejects_annual_horizon_not_anchored_to_release(tmp_path: Path) -> None:
    payload = _registration_payload()
    payload["release_id"] = "AFRY_CH_2027_Q2_V100"
    path = _write_registration(tmp_path, payload)

    with pytest.raises(AfryDataContractError, match="release-anchored horizon"):
        load_registration(path, workspace=tmp_path)


def test_registration_rejects_hourly_horizon_without_annual_end_anchor(
    tmp_path: Path,
) -> None:
    payload = _registration_payload()
    for sheet in payload["content_contract"]["hourly_prices"]["sheets"]:  # type: ignore[index]
        if 2060 in sheet["delivery_years"]:
            sheet["delivery_years"].remove(2060)
    path = _write_registration(tmp_path, payload)

    with pytest.raises(AfryDataContractError, match="share first and last years"):
        load_registration(path, workspace=tmp_path)


def test_runtime_tree_receipt_binds_every_plain_file(tmp_path: Path) -> None:
    prefix = tmp_path / "runtime"
    (prefix / "Lib").mkdir(parents=True)
    (prefix / "python.exe").write_bytes(b"python")
    (prefix / "Lib" / "dependency.py").write_bytes(b"dependency")

    first = runtime_tree_receipt(prefix)
    (prefix / "Lib" / "dependency.py").write_bytes(b"changed")
    second = runtime_tree_receipt(prefix)

    assert first["file_count"] == 2
    assert first["tree_sha256"] != second["tree_sha256"]


def test_registration_rejects_absolute_source_path(tmp_path: Path) -> None:
    payload = _registration_payload()
    payload["sources"][0]["path"] = str(tmp_path / "source.xlsx")  # type: ignore[index]
    path = _write_registration(tmp_path, payload)

    with pytest.raises(AfryDataContractError, match="relative"):
        load_registration(path, workspace=tmp_path)


def test_verify_source_accepts_safe_xlsx_container(tmp_path: Path) -> None:
    source = _zip_source(
        tmp_path / "safe.xlsx",
        {
            "[Content_Types].xml": b"<Types/>",
            "xl/workbook.xml": b"<workbook/>",
        },
    )

    receipt = verify_source(source)

    assert receipt["sha256"] == source.sha256
    assert receipt["macro_member_count"] == 0
    assert receipt["encrypted_member_count"] == 0


def test_capture_verified_source_captures_immutable_verified_bytes(
    tmp_path: Path,
) -> None:
    source = _zip_source(
        tmp_path / "safe.xlsx",
        {
            "[Content_Types].xml": b"<Types/>",
            "xl/workbook.xml": b"<workbook/>",
        },
    )
    captured, receipt = capture_verified_source(source)

    assert captured.payload == source.path.read_bytes()
    assert captured.source_name == source.path.name
    assert receipt["path"] == source.path.name
    assert receipt["capture_sha256"] == source.sha256
    assert receipt["captured_bytes"] == source.path.stat().st_size
    assert receipt["capture_location"] == "immutable_process_memory"


def test_capture_verified_source_rejects_unsafe_payload_without_disk_copy(tmp_path: Path) -> None:
    source = _zip_source(
        tmp_path / "macro.xlsx",
        {
            "xl/workbook.xml": b"<workbook/>",
            "xl/vbaProject.bin": b"not-a-real-macro",
        },
    )
    with pytest.raises(AfryDataContractError, match="macros=1"):
        capture_verified_source(source)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["macro.xlsx"]


def test_capture_verified_source_rejects_unregistered_role(tmp_path: Path) -> None:
    source = _zip_source(tmp_path / "safe.xlsx", {"xl/workbook.xml": b"<workbook/>"})

    with pytest.raises(AfryDataContractError, match="unsupported AFRY source role"):
        capture_verified_source(replace(source, role="../escape"))

    assert sorted(path.name for path in tmp_path.iterdir()) == ["safe.xlsx"]


def test_verify_source_rejects_hash_mismatch(tmp_path: Path) -> None:
    source = _zip_source(tmp_path / "hash.xlsx", {"xl/workbook.xml": b"<workbook/>"})
    bad = AfrySource(
        role=source.role,
        path=source.path,
        sha256="0" * 64,
        publisher_release_date=source.publisher_release_date,
    )

    with pytest.raises(AfryDataContractError, match="sha256 mismatch"):
        verify_source(bad)


def test_verify_source_rejects_macro_member(tmp_path: Path) -> None:
    source = _zip_source(
        tmp_path / "macro.xlsx",
        {
            "xl/workbook.xml": b"<workbook/>",
            "xl/vbaProject.bin": b"not-a-real-macro",
        },
    )

    with pytest.raises(AfryDataContractError, match="macros=1"):
        verify_source(source)


def test_verify_source_rejects_embedded_active_content(tmp_path: Path) -> None:
    source = _zip_source(
        tmp_path / "embedded.xlsx",
        {
            "xl/workbook.xml": b"<workbook/>",
            "xl/embeddings/oleObject1.bin": b"embedded executable content",
        },
    )

    with pytest.raises(AfryDataContractError, match="active_content=1"):
        verify_source(source)


def test_verify_source_rejects_worksheet_formula(tmp_path: Path) -> None:
    source = _zip_source(
        tmp_path / "formula.xlsx",
        {
            "xl/workbook.xml": b"<workbook/>",
            "xl/worksheets/sheet1.xml": b'<worksheet><c r="A1"><f>1+1</f><v>2</v></c></worksheet>',
        },
    )

    with pytest.raises(AfryDataContractError, match="formulas are forbidden"):
        verify_source(source)


def test_verify_source_rejects_case_and_backslash_obfuscated_formula(tmp_path: Path) -> None:
    source = _zip_source(
        tmp_path / "formula-obfuscated.xlsx",
        {"XL\\WORKSHEETS\\sheet1.XML": b"<worksheet><c><F>1+1</F></c></worksheet>"},
    )

    with pytest.raises(AfryDataContractError, match="formulas are forbidden"):
        verify_source(source)


def test_verify_source_allows_inert_external_link_metadata_for_static_values(
    tmp_path: Path,
) -> None:
    source = _zip_source(
        tmp_path / "inert-link.xlsx",
        {
            "xl/workbook.xml": b"<workbook/>",
            "xl/worksheets/sheet1.xml": b'<worksheet><c r="A1"><v>2</v></c></worksheet>',
            "xl/externalLinks/externalLink1.xml": b"<externalLink/>",
        },
    )

    receipt = verify_source(source)

    assert receipt["external_link_member_count"] == 1
    assert receipt["worksheet_formula_member_count"] == 0
    assert receipt["external_links_inert_static_values"] is True


def test_verify_source_counts_case_and_backslash_external_link_metadata(
    tmp_path: Path,
) -> None:
    source = _zip_source(
        tmp_path / "external-obfuscated.xlsx",
        {
            "xl/workbook.xml": b"<workbook/>",
            "XL\\EXTERNALLINKS\\externalLink1.xml": b"<externalLink/>",
        },
    )

    receipt = verify_source(source)

    assert receipt["external_link_member_count"] == 1


def test_verify_source_rejects_zip_traversal_member(tmp_path: Path) -> None:
    source = _zip_source(
        tmp_path / "traversal.xlsx",
        {
            "xl/workbook.xml": b"<workbook/>",
            "xl/../escape.xml": b"escape",
        },
    )

    with pytest.raises(AfryDataContractError, match="unsafe_paths=1"):
        verify_source(source)


def test_verify_source_rejects_windows_style_zip_traversal_member(tmp_path: Path) -> None:
    source = _zip_source(
        tmp_path / "windows-traversal.xlsx",
        {
            "xl/workbook.xml": b"<workbook/>",
            "xl\\..\\escape.xml": b"escape",
        },
    )

    with pytest.raises(AfryDataContractError, match="unsafe_paths=1"):
        verify_source(source)


def test_verify_source_rejects_excessive_compression_ratio(tmp_path: Path) -> None:
    source = _zip_source(
        tmp_path / "compression.xlsx",
        {"xl/workbook.xml": b"0" * 1_000_000},
    )

    with pytest.raises(AfryDataContractError, match="compression ratio"):
        verify_source(source)


def test_verify_source_rejects_diluted_per_member_zip_bomb(tmp_path: Path) -> None:
    path = tmp_path / "diluted-compression.xlsx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            b"0" * 200_000,
            compress_type=zipfile.ZIP_DEFLATED,
        )
        archive.writestr(
            "xl/media/padding.bin",
            b"padding" * 50_000,
            compress_type=zipfile.ZIP_STORED,
        )
    source = AfrySource(
        role="annual_scenarios",
        path=path,
        sha256=file_sha256(path),
        publisher_release_date="2026-06-26",
    )

    with pytest.raises(AfryDataContractError, match="member compression ratio"):
        verify_source(source)


def test_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "bytes.bin"
    payload = b"AFRY governed source bytes"
    path.write_bytes(payload)

    assert file_sha256(path) == hashlib.sha256(payload).hexdigest()


def test_catalog_content_id_covers_semantic_and_agent_access_fields() -> None:
    first = {
        "schema_version": "catalog-test-v1",
        "artifacts": [{"name": "data.parquet", "sha256": "a" * 64}],
        "admission": {"production_authorized": False},
        "agent_access": {"recommended_engine": "duckdb"},
    }
    second = json.loads(json.dumps(first))
    second["agent_access"]["recommended_engine"] = "pyarrow"

    assert catalog_content_id(first) != catalog_content_id(second)


def test_catalog_content_id_rejects_self_referential_bundle_id() -> None:
    with pytest.raises(AfryDataContractError, match="must not contain bundle_id"):
        catalog_content_id({"bundle_id": "not-allowed"})


def _write_catalog_bundle(root: Path) -> Path:
    annual_contract = AfryAnnualContract(
        release_year=2026,
        first_delivery_year_offset=1,
        delivery_year_count=34,
        delivery_years=tuple(range(2027, 2061)),
        structural_required_non_null_fields=STRUCTURAL_REQUIRED_NON_NULL_FIELDS,
        structural_declared_nullable_fields=STRUCTURAL_DECLARED_NULLABLE_FIELDS,
    )
    annual_facts = _semantic_fixture().copy()
    annual_facts["release_id"] = RELEASE_ID
    annual_facts["table_header_row"] = 1
    for field in (
        "dimension_1_raw",
        "dimension_2_raw",
        "dimension_3_raw",
        "dimension_2",
        "dimension_3",
    ):
        if field not in annual_facts:
            annual_facts[field] = None
    annual_facts["dimension_1_raw"] = annual_facts["dimension_1"]
    annual_facts["source_cell"] = [
        f"Fixture!A{index}" for index in range(1, len(annual_facts) + 1)
    ]
    annual_facts = annual_facts[ANNUAL_FACT_SCHEMA.names]
    structural = build_structural_benchmark(
        annual_facts,
        observed_at_utc="2026-08-03T09:22:51Z",
        release_id=RELEASE_ID,
        delivery_years=annual_contract.delivery_years,
    )
    for field in STRUCTURAL_REQUIRED_NON_NULL_FIELDS:
        structural[field] = 0.0
    structural["field_value_statuses_json"] = json.dumps(
        {field: ["modelled"] for field in STRUCTURAL_REQUIRED_NON_NULL_FIELDS},
        sort_keys=True,
        separators=(",", ":"),
    )
    structural["source_value_statuses"] = "modelled"
    structural["contains_interpolated_values"] = False
    annual_semantics = validate_annual_semantics(
        annual_facts,
        delivery_years=annual_contract.delivery_years,
        release_year=annual_contract.release_year,
    )
    structural_audit = validate_structural_contract(structural, annual_contract)
    parquet_schemas = {
        "hourly_prices.parquet": HOURLY_SCHEMA,
        "annual_facts.parquet": ANNUAL_FACT_SCHEMA,
        "structural_scenarios_benchmark.parquet": STRUCTURAL_SCHEMA,
        "hour_slot_shape_benchmark.parquet": HOUR_SLOT_SHAPE_SCHEMA,
    }
    artifacts = []
    root.mkdir(exist_ok=True)
    staging = root / "staging"
    staging.mkdir()
    for name, schema in parquet_schemas.items():
        path = staging / name
        if name == "annual_facts.parquet":
            _write_frame(annual_facts, path, schema)
            row_count = len(annual_facts)
        elif name == "structural_scenarios_benchmark.parquet":
            _write_frame(structural, path, schema)
            row_count = len(structural)
        else:
            values: dict[str, list[object]] = {}
            for field in schema:
                if field.name in {
                    "model_input_authorized",
                    "monthly_level_authority",
                    "production_authorized",
                }:
                    value: object = False
                elif pa.types.is_boolean(field.type):
                    value = False
                elif pa.types.is_integer(field.type):
                    value = 1
                elif pa.types.is_floating(field.type):
                    value = 0.0
                elif pa.types.is_timestamp(field.type):
                    value = datetime(2026, 1, 1, tzinfo=timezone.utc)
                else:
                    value = "fixture"
                values[field.name] = [value]
            pq.write_table(pa.Table.from_pydict(values, schema=schema), path)
            row_count = 1
        artifacts.append(
            {
                "name": name,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "rows": row_count,
            }
        )
    json_payloads = {
        "hourly_profile.json": {},
        "annual_semantic_profile.json": annual_semantics,
        "structural_contract_audit.json": structural_audit,
        "swiss_2035_policy_gap_profile.json": {
            "model_input_authorized": False,
            "monthly_level_authority": False,
            "production_authorized": False,
        },
        "hour_slot_shape_audit.json": {
            "model_input_authorized": False,
            "monthly_level_authority": False,
            "production_authorized": False,
        },
    }
    for name, payload in json_payloads.items():
        path = staging / name
        path.write_bytes(canonical_json_bytes(payload))
        artifacts.append(
            {
                "name": name,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "rows": None,
            }
        )
    body: dict[str, object] = {
        "schema_version": CATALOG_SCHEMA,
        "release_id": RELEASE_ID,
        "artifacts": artifacts,
        "decision_state": CATALOG_DECISION_STATE,
        "integrity_verdict": CATALOG_INTEGRITY_PASS,
        "gates": {
            **{name: "PASS" for name in CATALOG_INTEGRITY_GATE_NAMES},
            "swiss_2035_policy_target_reconciliation": "NO_GO_UNRESOLVED",
            "model_input_authorized": "NO_GO",
            "production_authorized": "NO_GO",
        },
        "admission": {
            "license_classification": "restricted_client_engagement",
            "raw_or_derived_values_allowed_in_git": False,
            "rag_indexing_allowed": False,
            "monthly_level_authority": False,
            "shaping_benchmark_only": True,
            "scenario_mapping_approved": False,
            "model_input_authorized": False,
            "production_authorized": False,
        },
        "registration": {
            "path": ".planning/registration.json",
            "schema_version": REGISTRATION_SCHEMA,
            "sha256": "c" * 64,
            "bytes": 1,
            "stable_during_materialization": True,
        },
        "annual_contract": {
            "release_year": annual_contract.release_year,
            "first_delivery_year_offset": annual_contract.first_delivery_year_offset,
            "delivery_year_count": annual_contract.delivery_year_count,
            "delivery_years": list(annual_contract.delivery_years),
            "scenarios": list(SCENARIOS),
        },
        "producer": {
            "files": [
                {
                    "role": "ingestion_module",
                    "path": "pfc_shaping/data/afry_scenarios.py",
                    "sha256": "a" * 64,
                },
                {
                    "role": "materializer_entrypoint",
                    "path": "scripts/materialize_afry_scenario_catalog.py",
                    "sha256": "b" * 64,
                },
            ],
            "runtime": {
                "python": "3.11.0",
                "implementation": "cpython",
                "cache_tag": "cpython-311",
                "platform_system": "Windows",
                "platform_machine": "AMD64",
                "executable_path": "build/runtime/python.exe",
                "executable_sha256": "d" * 64,
                "prefix_path": "build/runtime",
                "dependency_closure": {
                    "schema_version": "fmv-afry-runtime-tree-receipt-v1",
                    "tree_sha256": "e" * 64,
                    "file_count": 1,
                    "total_bytes": 1,
                    "plain_file_tree": True,
                    "stable_during_materialization": True,
                },
                "pandas": "test",
                "pyarrow": "test",
                "openpyxl": "test",
            },
            "source_stable_during_materialization": True,
        },
    }
    bundle_id = catalog_content_id(body)
    bundle = root / bundle_id
    shutil.copytree(staging, bundle)
    catalog_path = bundle / "catalog.json"
    catalog_path.write_bytes(canonical_json_bytes({**body, "bundle_id": bundle_id}))
    (bundle / "catalog.sha256").write_bytes(
        f"{file_sha256(catalog_path)}\n".encode("ascii")
    )
    return bundle


def test_verify_catalog_bundle_checks_exact_member_and_hash_closure(tmp_path: Path) -> None:
    bundle = _write_catalog_bundle(tmp_path)

    catalog = verify_catalog_bundle(bundle)

    assert catalog["bundle_id"] == bundle.name


def test_read_verified_artifact_bytes_rebinds_consumed_payload(tmp_path: Path) -> None:
    bundle = _write_catalog_bundle(tmp_path)

    catalog, payload = read_verified_artifact_bytes(bundle, "hourly_profile.json")

    artifact = next(
        item for item in catalog["artifacts"] if item["name"] == "hourly_profile.json"
    )
    assert hashlib.sha256(payload).hexdigest() == artifact["sha256"]


def test_verify_catalog_bundle_rejects_corrupt_artifact(tmp_path: Path) -> None:
    bundle = _write_catalog_bundle(tmp_path)
    (bundle / "hourly_profile.json").write_bytes(b"corrupt")

    with pytest.raises(AfryDataContractError, match="size mismatch"):
        verify_catalog_bundle(bundle)


def test_verify_catalog_bundle_rejects_extra_member(tmp_path: Path) -> None:
    bundle = _write_catalog_bundle(tmp_path)
    (bundle / "unexpected.bin").write_bytes(b"unexpected")

    with pytest.raises(AfryDataContractError, match="member closure mismatch"):
        verify_catalog_bundle(bundle)


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("monthly_level_authority", True),
        ("scenario_mapping_approved", True),
        ("model_input_authorized", True),
        ("production_authorized", True),
        ("rag_indexing_allowed", True),
        ("raw_or_derived_values_allowed_in_git", True),
    ],
)
def test_verify_catalog_bundle_rejects_self_consistent_authority_flip(
    tmp_path: Path, field: str, unsafe_value: bool
) -> None:
    bundle = _write_catalog_bundle(tmp_path)
    catalog_path = bundle / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["admission"][field] = unsafe_value
    body = dict(catalog)
    body.pop("bundle_id")
    new_id = catalog_content_id(body)
    catalog["bundle_id"] = new_id
    catalog_path.write_bytes(canonical_json_bytes(catalog))
    (bundle / "catalog.sha256").write_bytes(
        f"{file_sha256(catalog_path)}\n".encode("ascii")
    )
    rewritten = bundle.with_name(new_id)
    shutil.copytree(bundle, rewritten)

    with pytest.raises(AfryDataContractError, match=field):
        verify_catalog_bundle(rewritten)


def test_verify_catalog_bundle_rejects_self_consistent_parquet_authority_flip(
    tmp_path: Path,
) -> None:
    bundle = _write_catalog_bundle(tmp_path)
    parquet_path = bundle / "structural_scenarios_benchmark.parquet"
    table = pq.read_table(parquet_path)
    column_index = table.schema.get_field_index("model_input_authorized")
    table = table.set_column(
        column_index,
        STRUCTURAL_SCHEMA.field("model_input_authorized"),
        pa.array([True, *([False] * (table.num_rows - 1))], type=pa.bool_()),
    ).replace_schema_metadata(STRUCTURAL_SCHEMA.metadata)
    pq.write_table(table, parquet_path)
    catalog_path = bundle / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    artifact = next(
        item
        for item in catalog["artifacts"]
        if item["name"] == "structural_scenarios_benchmark.parquet"
    )
    artifact["bytes"] = parquet_path.stat().st_size
    artifact["sha256"] = file_sha256(parquet_path)
    body = dict(catalog)
    body.pop("bundle_id")
    new_id = catalog_content_id(body)
    catalog["bundle_id"] = new_id
    catalog_path.write_bytes(canonical_json_bytes(catalog))
    (bundle / "catalog.sha256").write_bytes(
        f"{file_sha256(catalog_path)}\n".encode("ascii")
    )
    rewritten = bundle.with_name(new_id)
    shutil.copytree(bundle, rewritten)

    with pytest.raises(AfryDataContractError, match="Parquet authority"):
        verify_catalog_bundle(rewritten)


def test_verify_catalog_bundle_recomputes_structural_invariants(tmp_path: Path) -> None:
    bundle = _write_catalog_bundle(tmp_path)
    parquet_path = bundle / "structural_scenarios_benchmark.parquet"
    table = pq.read_table(parquet_path)
    column_index = table.schema.get_field_index("scenario_weight")
    weights = table["scenario_weight"].to_pylist()
    weights[0] = 0.25
    table = table.set_column(
        column_index,
        STRUCTURAL_SCHEMA.field("scenario_weight"),
        pa.array(weights, type=pa.float64()),
    ).replace_schema_metadata(STRUCTURAL_SCHEMA.metadata)
    pq.write_table(table, parquet_path)
    catalog_path = bundle / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    artifact = next(
        item
        for item in catalog["artifacts"]
        if item["name"] == "structural_scenarios_benchmark.parquet"
    )
    artifact["bytes"] = parquet_path.stat().st_size
    artifact["sha256"] = file_sha256(parquet_path)
    body = dict(catalog)
    body.pop("bundle_id")
    new_id = catalog_content_id(body)
    catalog["bundle_id"] = new_id
    catalog_path.write_bytes(canonical_json_bytes(catalog))
    (bundle / "catalog.sha256").write_bytes(
        f"{file_sha256(catalog_path)}\n".encode("ascii")
    )
    rewritten = bundle.with_name(new_id)
    shutil.copytree(bundle, rewritten)

    with pytest.raises(AfryDataContractError, match="structural audit does not match"):
        verify_catalog_bundle(rewritten)


def test_verify_catalog_bundle_recomputes_annual_domain(tmp_path: Path) -> None:
    bundle = _write_catalog_bundle(tmp_path)
    parquet_path = bundle / "annual_facts.parquet"
    table = pq.read_table(parquet_path)
    column_index = table.schema.get_field_index("delivery_year")
    years = table["delivery_year"].to_pylist()
    years[0] = 2061
    table = table.set_column(
        column_index,
        ANNUAL_FACT_SCHEMA.field("delivery_year"),
        pa.array(years, type=pa.int16()),
    ).replace_schema_metadata(ANNUAL_FACT_SCHEMA.metadata)
    pq.write_table(table, parquet_path)
    catalog_path = bundle / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    artifact = next(
        item for item in catalog["artifacts"] if item["name"] == "annual_facts.parquet"
    )
    artifact["bytes"] = parquet_path.stat().st_size
    artifact["sha256"] = file_sha256(parquet_path)
    body = dict(catalog)
    body.pop("bundle_id")
    new_id = catalog_content_id(body)
    catalog["bundle_id"] = new_id
    catalog_path.write_bytes(canonical_json_bytes(catalog))
    (bundle / "catalog.sha256").write_bytes(
        f"{file_sha256(catalog_path)}\n".encode("ascii")
    )
    rewritten = bundle.with_name(new_id)
    shutil.copytree(bundle, rewritten)

    with pytest.raises(AfryDataContractError, match="annual semantic audit does not match"):
        verify_catalog_bundle(rewritten)


def test_verify_catalog_bundle_rejects_ambiguous_plain_pass(tmp_path: Path) -> None:
    bundle = _write_catalog_bundle(tmp_path)
    catalog_path = bundle / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["qa_verdict"] = "PASS"
    body = dict(catalog)
    body.pop("bundle_id")
    new_id = catalog_content_id(body)
    catalog["bundle_id"] = new_id
    catalog_path.write_bytes(canonical_json_bytes(catalog))
    (bundle / "catalog.sha256").write_bytes(
        f"{file_sha256(catalog_path)}\n".encode("ascii")
    )
    rewritten = bundle.with_name(new_id)
    shutil.copytree(bundle, rewritten)

    with pytest.raises(AfryDataContractError, match="qa_verdict"):
        verify_catalog_bundle(rewritten)


def test_verify_catalog_bundle_rejects_missing_producer_provenance(tmp_path: Path) -> None:
    bundle = _write_catalog_bundle(tmp_path)
    catalog_path = bundle / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog.pop("producer")
    body = dict(catalog)
    body.pop("bundle_id")
    new_id = catalog_content_id(body)
    catalog["bundle_id"] = new_id
    catalog_path.write_bytes(canonical_json_bytes(catalog))
    (bundle / "catalog.sha256").write_bytes(
        f"{file_sha256(catalog_path)}\n".encode("ascii")
    )
    rewritten = bundle.with_name(new_id)
    shutil.copytree(bundle, rewritten)

    with pytest.raises(AfryDataContractError, match="producer provenance"):
        verify_catalog_bundle(rewritten)


def test_verify_supporting_document_accepts_exact_static_pdf_bytes(tmp_path: Path) -> None:
    path = tmp_path / "note.pdf"
    path.write_bytes(b"%PDF-1.7\nstatic fixture")
    document = AfrySupportingDocument(
        role="quarterly_update_note",
        path=path,
        sha256=file_sha256(path),
        publisher_release_date="2026-07-16",
    )

    receipt = verify_supporting_document(document)

    assert receipt["sha256"] == document.sha256
    assert receipt["content_use"] == "semantic_context_only"
    assert receipt["parsed_by_model_pipeline"] is False


def test_verify_supporting_document_rejects_non_pdf_header(tmp_path: Path) -> None:
    path = tmp_path / "note.pdf"
    path.write_bytes(b"not a PDF")
    document = AfrySupportingDocument(
        role="quarterly_update_note",
        path=path,
        sha256=file_sha256(path),
        publisher_release_date="2026-07-16",
    )

    with pytest.raises(AfryDataContractError, match="lacks a PDF header"):
        verify_supporting_document(document)


def test_verify_supporting_document_reports_non_authoritative_active_token_hint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "active.pdf"
    path.write_bytes(b"%PDF-1.7\n1 0 obj << /OpenAction 2 0 R >> endobj")
    document = AfrySupportingDocument(
        role="quarterly_update_note",
        path=path,
        sha256=file_sha256(path),
        publisher_release_date="2026-07-16",
    )

    receipt = verify_supporting_document(document)

    assert receipt["active_content_lexical_hints_non_authoritative"] == ["open_action"]
    assert receipt["rendering_or_execution_authorized"] is False


def test_verify_semantic_contract_rejects_authority_flip(tmp_path: Path) -> None:
    registration = load_registration(REGISTRATION, workspace=Path.cwd())
    payload = json.loads(registration.semantic_contract_path.read_text(encoding="utf-8"))
    payload["shaping_admission"]["production_authorized"] = True
    path = tmp_path / "semantic-contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    tampered = replace(
        registration,
        semantic_contract_path=path,
        semantic_contract_sha256=file_sha256(path),
    )

    with pytest.raises(AfryDataContractError, match="production_authorized"):
        verify_semantic_contract(tampered)


def test_verify_semantic_contract_rejects_vendor_model_as_observed_truth(
    tmp_path: Path,
) -> None:
    registration = load_registration(REGISTRATION, workspace=Path.cwd())
    payload = json.loads(registration.semantic_contract_path.read_text(encoding="utf-8"))
    payload["methodological_evidence"]["observed_swiss_truth"] = True
    path = tmp_path / "semantic-contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    tampered = replace(
        registration,
        semantic_contract_path=path,
        semantic_contract_sha256=file_sha256(path),
    )

    with pytest.raises(AfryDataContractError, match="observed_swiss_truth"):
        verify_semantic_contract(tampered)
