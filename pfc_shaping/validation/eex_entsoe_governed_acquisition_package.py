"""Validate the D233 zero-execution EEX/ENTSO-E evidence-package request.

The validator opens policy metadata only. It never connects to Databricks,
opens market values, or grants model authority.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)

ACQUISITION_PACKAGE_CONTRACT_SCHEMA = (
    "fmv-eex-entsoe-governed-acquisition-package-contract-v1"
)
ACQUISITION_PACKAGE_CONTRACT_STATUS = (
    "PREPARED_D233_LOCAL_ZERO_EXECUTION_NO_MODEL_AUTHORITY"
)
ACQUISITION_PACKAGE_CONTRACT_CONTENT_ID = (
    "4f6e644b15593e8d4ae2ca196e1d52a002d4255bdf42e7a137982edb854f0fa0"
)
ACQUISITION_PACKAGE_CONTRACT_FILE_SHA256 = (
    "97edaf93c639953365d20d7539380a706f0f8fdbec9476af94262fb1d4b2204d"
)
TOP_LEVEL_FIELDS = (
    "schema_version",
    "status",
    "purpose",
    "selected_boundaries",
    "delivery_boundary",
    "artifact_inventory",
    "source_scope",
    "execution_and_cost_fence",
    "manifest_contract",
    "source_envelope_contract",
    "external_time_batch_contract",
    "standards_basis",
    "authorities",
    "required_before_authoritative_use",
)
ZERO_QUERY_PLAN_CONTENT_ID = (
    "3395f45bd1d22663386aa7cd4e93cfe2bc02079fa7cb8b103825b9c5650dc1af"
)
ZERO_QUERY_PLAN_FILE_SHA256 = (
    "f62bd9e0a9ffe6f0daa2b02917b47763b64a340f28ab59cdd664cbdd8ec58999"
)
REGISTERED_SIGNER_BINDING_CONTENT_ID = (
    "2fa6d60542f513faf132bc33dbf6ea17c98693b7658c37ad9e92195dcdd053f0"
)
REGISTERED_SIGNER_BINDING_FILE_SHA256 = (
    "849cb6cea28913b87b69cea463d54ea9f110150467343cbd8f27f602a07fca23"
)
ENTSOE_INTAKE_CONTRACT_FILE_SHA256 = (
    "7ede1698099390babfa1d130bfecae61fd1e090888a3d6e6e4f892119db52b87"
)
ENTSOE_INTAKE_CONTRACT_CONTENT_ID = (
    "75cb7e9f5d4cedb55e31fbd26b0bdcdf4ff629caf641c984cfec3fc6510b2a60"
)
RECEIPT_CONTRACT_CONTENT_ID = (
    "d4f7e4e87b8e059255e1a85a589f41d54596756d731ce4a9781b2dbc46d61068"
)
RECEIPT_CONTRACT_FILE_SHA256 = (
    "3168d75e245f033daa44f4c6dab9061f51e06f2b993797b7fb7d9c46dbe6b665"
)
TRUST_REGISTRY_CONTRACT_CONTENT_ID = (
    "3ecf96f83be71c5beb24160793d5fcbce2d771edbda74e190bda3d5b851b54fa"
)
TRUST_REGISTRY_CONTRACT_FILE_SHA256 = (
    "e7fbf4f099bcc355073d0f5b591d988100df4168c22b172ce8fc08e4c4a0ed9c"
)
EEX_NORMALIZER_MODULE_SHA256 = (
    "f1469df05c0f4ed8e25c515c6bb62b92f479ac4ad1081a6a3be7c9aeea7bb086"
)
ENTSOE_METADATA_ADMISSION_CONTENT_ID = (
    "ac0ccb6b5a94e0dee46050b5a765f92efcf800f078b93645917247a3be206564"
)
ENTSOE_METADATA_ADMISSION_FILE_SHA256 = (
    "19033d5e24800abc240d3de4974cc44ec7a3fc7afc6f69db07ee1cf0a6191c43"
)
REQUIRED_ROLES = (
    "eex_acquisition",
    "eex_source_trusted_time",
    "entsoe_acquisition",
    "entsoe_source_trusted_time",
)
ARTIFACT_ROLES = (
    "eex_daily_snapshot",
    "entsoe_series_dimension",
    "entsoe_latest_values",
    "entsoe_vintage_values",
)
EEX_TABLES = (
    "facteexpricedaily",
    "dimeexproduct",
    "dimeexdeliveryperiod",
)
EEX_PRODUCT_FAMILIES = ("DAY", "WEEK", "WEEKEND", "MONTH", "QUARTER", "YEAR")
EEX_GRAIN = ("ProductID", "DeliveryPeriodID", "QuotationDateID")
EEX_FIELDS = (
    "ProductID",
    "DeliveryPeriodID",
    "QuotationDateID",
    "SettlementPriceEurMWh",
    "LastPriceEurMWh",
    "FactLoadTimestampUtc",
    "Country",
    "Commodity",
    "ProductType",
    "DeliveryPeriodType",
    "DeliveryStartDate",
    "DeliveryEndDate",
)
ENTSOE_TABLES = (
    "dimentsoeseries",
    "factentsoetimeserieslatest",
    "factentsoetimeseriesvintages",
)
ENTSOE_FIELDS = {
    "entsoe_series_dimension": (
        "series_id",
        "group_name",
        "field_name",
        "from_zone",
        "to_zone",
        "unit",
        "native_resolution",
        "source_endpoint",
        "document_id",
    ),
    "entsoe_latest_values": (
        "series_id",
        "target_ts_utc",
        "value",
        "is_historical",
        "quality_flag",
        "revision_number",
        "meta_load_timestamp_utc",
    ),
    "entsoe_vintage_values": (
        "series_id",
        "target_ts_utc",
        "as_of_utc",
        "value",
        "quality_flag",
        "revision_number",
        "meta_load_timestamp_utc",
    ),
}
BOUNDARY_FILES: dict[str, tuple[str, str, str | None]] = {
    "zero_query_plan": (
        ".planning/phases/14-lt-audit-remediation/CH-LT-DATABRICKS-ZERO-QUERY-ACQUISITION-PLAN-V1.json",
        ZERO_QUERY_PLAN_FILE_SHA256,
        ZERO_QUERY_PLAN_CONTENT_ID,
    ),
    "entsoe_metadata_admission_contract": (
        ".planning/phases/14-lt-audit-remediation/ENTSOE-DATABRICKS-METADATA-ADMISSION-CONTRACT-V1.json",
        ENTSOE_METADATA_ADMISSION_FILE_SHA256,
        ENTSOE_METADATA_ADMISSION_CONTENT_ID,
    ),
    "receipt_contract": (
        ".planning/phases/14-lt-audit-remediation/EEX-CH-SHORT-TENOR-RECEIPT-CONTRACT-V1.json",
        RECEIPT_CONTRACT_FILE_SHA256,
        RECEIPT_CONTRACT_CONTENT_ID,
    ),
    "trust_registry_contract": (
        ".planning/phases/14-lt-audit-remediation/EEX-CH-SHORT-TENOR-TRUST-REGISTRY-CONTRACT-V1.json",
        TRUST_REGISTRY_CONTRACT_FILE_SHA256,
        TRUST_REGISTRY_CONTRACT_CONTENT_ID,
    ),
    "registered_signer_binding_contract": (
        ".planning/phases/14-lt-audit-remediation/EEX-CH-SHORT-TENOR-REGISTERED-SIGNER-BINDING-CONTRACT-V1.json",
        REGISTERED_SIGNER_BINDING_FILE_SHA256,
        REGISTERED_SIGNER_BINDING_CONTENT_ID,
    ),
    "entsoe_intake_contract": (
        ".planning/phases/14-lt-audit-remediation/ENTSOE-DATABRICKS-INTAKE-CONTRACT-V1.json",
        ENTSOE_INTAKE_CONTRACT_FILE_SHA256,
        ENTSOE_INTAKE_CONTRACT_CONTENT_ID,
    ),
    "eex_normalizer_module": (
        "pfc_shaping/data/databricks_eex_daily_snapshot.py",
        EEX_NORMALIZER_MODULE_SHA256,
        None,
    ),
}
EEX_LOCAL_FILES = {
    "eex_source_manifest": (
        "build/databricks-eex-daily/2026-08-05/manifest.json",
        "f8ec096be43851d85b16ec2b678d4a695fb0521c2c651e8bcf7c2491a29b50c1",
    ),
    "eex_artifact": (
        "build/databricks-eex-daily/2026-08-05/eex_ch_power.ndjson",
        "593e916b6aa18ad83f7bd7941ff68184cd71da8882ef4eb381de46d09ce64812",
    ),
}
FALSE_AUTHORITY_FIELDS = (
    "real_source_artifact_package_present",
    "real_source_envelopes_present",
    "real_external_time_evidence_present",
    "entsoe_schema_verified",
    "entsoe_values_opened",
    "same_snapshot_point_in_time_availability_proven",
    "actual_signature_generation_time_verified",
    "training_authorized",
    "selection_authorized",
    "model_input_authorized",
    "candidate_assembly_authorized",
    "promotion_authorized",
    "production_authorized",
)


class GovernedAcquisitionPackageError(ValueError):
    """Raised when the package weakens a governed boundary."""


def validate_governed_acquisition_package_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact value-free D233 preparation contract."""

    contract = _mapping(document, "acquisition package contract")
    _exact_keys(contract, set(TOP_LEVEL_FIELDS), "contract")
    _equal(contract.get("schema_version"), ACQUISITION_PACKAGE_CONTRACT_SCHEMA, "schema")
    _equal(contract.get("status"), ACQUISITION_PACKAGE_CONTRACT_STATUS, "status")
    _validate_boundaries(_mapping(contract.get("selected_boundaries"), "boundaries"))
    _validate_delivery(_mapping(contract.get("delivery_boundary"), "delivery boundary"))
    artifacts = _artifact_map(contract.get("artifact_inventory"))
    _validate_eex_artifact(artifacts["eex_daily_snapshot"])
    _validate_entsoe_artifacts(artifacts)
    _validate_source_scope(_mapping(contract.get("source_scope"), "source scope"))
    _validate_cost_fence(
        _mapping(contract.get("execution_and_cost_fence"), "execution and cost fence")
    )
    _validate_manifest(_mapping(contract.get("manifest_contract"), "manifest contract"))
    _validate_envelopes(
        _mapping(contract.get("source_envelope_contract"), "source envelope contract")
    )
    _validate_external_time(
        _mapping(contract.get("external_time_batch_contract"), "external time contract")
    )
    authorities = _mapping(contract.get("authorities"), "authorities")
    _exact_keys(authorities, set(FALSE_AUTHORITY_FIELDS), "authorities")
    for field in FALSE_AUTHORITY_FIELDS:
        _equal(authorities.get(field), False, f"authority {field}")
    _exact_sequence(
        contract.get("standards_basis"),
        (
            "https://www.rfc-editor.org/rfc/rfc3161",
            "https://www.rfc-editor.org/rfc/rfc9162",
            "https://in-toto.io/Statement/v1",
            "https://github.com/secure-systems-lab/dsse/blob/master/envelope.md",
            "https://slsa.dev/provenance/v1",
            "https://www.w3.org/TR/prov-o/",
        ),
        "standards basis",
    )
    required = contract.get("required_before_authoritative_use")
    if not isinstance(required, list) or len(required) != 10:
        raise GovernedAcquisitionPackageError(
            "required-before-authoritative-use inventory must contain ten gates"
        )
    if not all(isinstance(item, str) and item.strip() for item in required):
        raise GovernedAcquisitionPackageError("required-use gates must be non-empty text")
    content_id = _sha256_json(contract)
    _equal(content_id, ACQUISITION_PACKAGE_CONTRACT_CONTENT_ID, "contract content ID")
    return {
        "schema_version": ACQUISITION_PACKAGE_CONTRACT_SCHEMA,
        "status": ACQUISITION_PACKAGE_CONTRACT_STATUS,
        "contract_content_id": content_id,
        "artifact_roles": list(ARTIFACT_ROLES),
        "required_envelope_roles": list(REQUIRED_ROLES),
        "eex_product_families": list(EEX_PRODUCT_FAMILIES),
        "eex_price_unit": "EUR/MWh",
        "entsoe_day_ahead_price_unit": "EUR/MWh",
        "historical_hourly_swiss_prices_are_not_a_missing_15_minute_data_defect": True,
        "this_contract_databricks_statements": 0,
        "this_contract_warehouse_starts": 0,
        "future_max_capture_batches_per_europe_zurich_day": 1,
        "maximum_future_entsoe_statement_count": 3,
        "maximum_total_local_export_bytes": 68_719_476_736,
        "model_input_authorized": False,
        "production_authorized": False,
    }


def build_zero_execution_preflight(
    contract: Mapping[str, object],
    *,
    local_validation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build deterministic local D233 evidence without acquiring data."""

    assessment = validate_governed_acquisition_package_contract(contract)
    local_verified = local_validation is not None
    if local_validation is not None:
        _equal(
            local_validation.get("status"),
            "PASS_LOCAL_HASHES_ONLY_ZERO_EXECUTION",
            "local validation status",
        )
        for field in (
            "existing_eex_rows_parsed",
            "databricks_connections",
            "databricks_statements",
            "warehouse_starts",
            "network_calls",
            "h_drive_accesses",
            "remote_writes",
        ):
            _integer_zero(local_validation.get(field), f"local validation {field}")
    return {
        "schema_version": "fmv_eex_entsoe_governed_acquisition_preflight.v1",
        "status": assessment["status"],
        "contract_content_id": assessment["contract_content_id"],
        "artifact_roles": list(ARTIFACT_ROLES),
        "source_roles": list(REQUIRED_ROLES),
        "source_inventory": {
            "eex": {
                "catalog": "prd",
                "schema": "gold",
                "tables": list(EEX_TABLES),
                "product_families": list(EEX_PRODUCT_FAMILIES),
                "price_unit": "EUR/MWh",
            },
            "entsoe": {
                "catalog": "dev",
                "schema": "gold",
                "tables": list(ENTSOE_TABLES),
                "day_ahead_price_unit": "EUR/MWh",
            },
        },
        "execution": {
            "databricks_connections": 0,
            "databricks_statements": 0,
            "warehouse_starts": 0,
            "network_calls": 0,
            "h_drive_accesses": 0,
            "remote_writes": 0,
            "market_value_rows_opened": 0,
        },
        "future_daily_budget": {
            "max_capture_batches_per_europe_zurich_day": 1,
            "max_warehouse_starts_per_capture_batch": 1,
            "max_entsoe_read_only_statements_per_capture_batch": 3,
            "ceiling_active": False,
            "new_explicit_user_authorization_required": True,
        },
        "local_boundary_integrity_verified": local_verified,
        "existing_eex_bytes_hash_verified": local_verified,
        "evidence": {
            "real_source_artifact_package_present": False,
            "real_source_envelopes_present": False,
            "real_external_time_evidence_present": False,
            "same_snapshot_point_in_time_availability_proven": False,
        },
        "authorities": {
            "model_input_authorized": False,
            "candidate_assembly_authorized": False,
            "promotion_authorized": False,
            "production_authorized": False,
        },
    }


def validate_zero_execution_preflight(document: Mapping[str, object]) -> dict[str, object]:
    """Fail closed if a purported preflight reports execution or authority."""

    preflight = _mapping(document, "zero-execution preflight")
    _exact_keys(
        preflight,
        {
            "schema_version",
            "status",
            "contract_content_id",
            "artifact_roles",
            "source_roles",
            "source_inventory",
            "execution",
            "future_daily_budget",
            "local_boundary_integrity_verified",
            "existing_eex_bytes_hash_verified",
            "evidence",
            "authorities",
        },
        "preflight",
    )
    _equal(
        preflight.get("schema_version"),
        "fmv_eex_entsoe_governed_acquisition_preflight.v1",
        "preflight schema",
    )
    _equal(preflight.get("status"), ACQUISITION_PACKAGE_CONTRACT_STATUS, "preflight status")
    _equal(
        preflight.get("contract_content_id"),
        ACQUISITION_PACKAGE_CONTRACT_CONTENT_ID,
        "preflight contract content ID",
    )
    _exact_sequence(preflight.get("artifact_roles"), ARTIFACT_ROLES, "artifact roles")
    _exact_sequence(preflight.get("source_roles"), REQUIRED_ROLES, "source roles")
    execution = _mapping(preflight.get("execution"), "preflight execution")
    _exact_keys(
        execution,
        {
            "databricks_connections",
            "databricks_statements",
            "warehouse_starts",
            "network_calls",
            "h_drive_accesses",
            "remote_writes",
            "market_value_rows_opened",
        },
        "preflight execution",
    )
    for field, value in execution.items():
        _integer_zero(value, f"preflight execution {field}")
    evidence = _mapping(preflight.get("evidence"), "preflight evidence")
    for field, value in evidence.items():
        _equal(value, False, f"preflight evidence {field}")
    authorities = _mapping(preflight.get("authorities"), "preflight authorities")
    for field, value in authorities.items():
        _equal(value, False, f"preflight authority {field}")
    budget = _mapping(preflight.get("future_daily_budget"), "future daily budget")
    _equal(budget.get("ceiling_active"), False, "future ceiling active")
    _equal(
        budget.get("new_explicit_user_authorization_required"),
        True,
        "future explicit authorization",
    )
    return dict(preflight)


def validate_local_boundaries(workspace_root: Path) -> dict[str, object]:
    """Hash frozen local contracts and reused EEX bytes without parsing prices."""

    root = workspace_root.resolve(strict=True)
    boundary_sha256s: dict[str, str] = {}
    boundary_content_ids: dict[str, str] = {}
    for role, (relative, expected_sha, expected_content_id) in BOUNDARY_FILES.items():
        payload = _read_stable_workspace_file(root, relative)
        actual_sha = hashlib.sha256(payload).hexdigest()
        _equal(actual_sha, expected_sha, f"{role} file SHA-256")
        boundary_sha256s[role] = actual_sha
        if expected_content_id is not None:
            actual_content_id = _sha256_json(load_strict_json(payload))
            _equal(actual_content_id, expected_content_id, f"{role} content ID")
            boundary_content_ids[role] = actual_content_id

    existing_eex_sha256s: dict[str, str] = {}
    for role, (relative, expected_sha) in EEX_LOCAL_FILES.items():
        payload = _read_stable_workspace_file(root, relative, maximum_bytes=70_000_000)
        actual_sha = hashlib.sha256(payload).hexdigest()
        _equal(actual_sha, expected_sha, f"{role} SHA-256")
        existing_eex_sha256s[role] = actual_sha

    manifest_payload = _read_stable_workspace_file(
        root, EEX_LOCAL_FILES["eex_source_manifest"][0]
    )
    # The already-hashed D231 PowerShell manifest is legacy UTF-8-with-BOM.
    # Accept that exact frozen byte artifact only; future package JSON remains
    # strict UTF-8 without this compatibility normalization.
    manifest = load_strict_json(manifest_payload.removeprefix(b"\xef\xbb\xbf"))
    _equal(manifest.get("row_count"), 82_552, "reused EEX row count")
    _equal(manifest.get("statement_count"), 1, "historical EEX statement count")
    _equal(manifest.get("read_only_sql"), True, "historical EEX read-only flag")
    _equal(manifest.get("databricks_data_mutation"), False, "historical EEX mutation flag")
    artifact = _mapping(manifest.get("artifact"), "historical EEX artifact metadata")
    _equal(artifact.get("sha256"), EEX_LOCAL_FILES["eex_artifact"][1], "EEX artifact link")
    return {
        "schema_version": "fmv_eex_entsoe_d233_local_boundary_validation.v1",
        "status": "PASS_LOCAL_HASHES_ONLY_ZERO_EXECUTION",
        "boundary_sha256s": boundary_sha256s,
        "boundary_content_ids": boundary_content_ids,
        "existing_eex_file_sha256s": existing_eex_sha256s,
        "existing_eex_rows_parsed": 0,
        "databricks_connections": 0,
        "databricks_statements": 0,
        "warehouse_starts": 0,
        "network_calls": 0,
        "h_drive_accesses": 0,
        "remote_writes": 0,
    }


def load_strict_json(payload: bytes) -> Mapping[str, object]:
    """Load bounded UTF-8 JSON and reject duplicate fields or non-finite tokens."""

    if not payload or len(payload) > 1_048_576:
        raise GovernedAcquisitionPackageError("JSON payload must be non-empty and <= 1 MiB")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GovernedAcquisitionPackageError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise GovernedAcquisitionPackageError(f"non-finite JSON token forbidden: {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GovernedAcquisitionPackageError("invalid strict UTF-8 JSON") from exc
    return _mapping(value, "JSON payload")


def _read_stable_workspace_file(
    workspace_root: Path,
    relative_path: str,
    *,
    maximum_bytes: int = 2_000_000,
) -> bytes:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise GovernedAcquisitionPackageError(
            "local boundary path must be a repo-relative descendant"
        )
    root = assert_absolute_path_has_no_links(workspace_root.resolve(strict=True))
    selected = assert_absolute_path_has_no_links((root / relative).resolve(strict=True))
    try:
        selected.relative_to(root)
    except ValueError as exc:
        raise GovernedAcquisitionPackageError(
            "local boundary path escapes the workspace"
        ) from exc
    return read_stable_single_link_file(
        selected,
        label=f"D233 local boundary {relative_path}",
        max_bytes=maximum_bytes,
    )


def _validate_boundaries(boundaries: Mapping[str, object]) -> None:
    expected = {
        "zero_query_plan_schema": "fmv-ch-lt-databricks-zero-query-acquisition-plan-v1",
        "zero_query_plan_content_id": ZERO_QUERY_PLAN_CONTENT_ID,
        "zero_query_plan_sha256": ZERO_QUERY_PLAN_FILE_SHA256,
        "zero_query_plan_decision": "D-20260805-231",
        "entsoe_metadata_admission_contract_schema": "fmv_entsoe_databricks_metadata_admission_contract.v1",
        "entsoe_metadata_admission_contract_content_id": ENTSOE_METADATA_ADMISSION_CONTENT_ID,
        "entsoe_metadata_admission_contract_sha256": ENTSOE_METADATA_ADMISSION_FILE_SHA256,
        "entsoe_metadata_admission_decision": "D-20260805-232",
        "receipt_contract_schema": "fmv-eex-ch-short-tenor-receipt-contract-v1",
        "receipt_contract_content_id": RECEIPT_CONTRACT_CONTENT_ID,
        "receipt_contract_sha256": RECEIPT_CONTRACT_FILE_SHA256,
        "receipt_contract_decision": "D-20260805-227",
        "trust_registry_contract_schema": "fmv-eex-ch-short-tenor-trust-registry-contract-v1",
        "trust_registry_contract_content_id": TRUST_REGISTRY_CONTRACT_CONTENT_ID,
        "trust_registry_contract_sha256": TRUST_REGISTRY_CONTRACT_FILE_SHA256,
        "trust_registry_contract_decision": "D-20260805-229",
        "registered_signer_binding_contract_schema": "fmv-eex-ch-short-tenor-registered-signer-binding-contract-v1",
        "registered_signer_binding_contract_content_id": REGISTERED_SIGNER_BINDING_CONTENT_ID,
        "registered_signer_binding_contract_sha256": REGISTERED_SIGNER_BINDING_FILE_SHA256,
        "registered_signer_binding_decision": "D-20260805-230",
        "entsoe_intake_contract_schema": "fmv_entsoe_databricks_intake_contract.v1",
        "entsoe_intake_contract_content_id": ENTSOE_INTAKE_CONTRACT_CONTENT_ID,
        "entsoe_intake_contract_sha256": ENTSOE_INTAKE_CONTRACT_FILE_SHA256,
        "eex_normalizer_schema": "fmv_databricks_eex_daily_normalization.v1",
        "eex_normalizer_module_sha256": EEX_NORMALIZER_MODULE_SHA256,
    }
    _exact_keys(boundaries, set(expected), "boundaries")
    for field, expected_value in expected.items():
        _equal(boundaries.get(field), expected_value, f"boundary {field}")


def _validate_delivery(delivery: Mapping[str, object]) -> None:
    _equal(
        delivery.get("preferred_mode"),
        "DATA_ENGINEER_PUBLISHES_EXISTING_READ_ONLY_EXPORT_TO_REPO_LOCAL_INTAKE",
        "delivery mode",
    )
    _equal(
        delivery.get("intake_root_template"),
        "build/governed-source-intake/<snapshot_id>",
        "intake root",
    )
    for field in (
        "pfc_databricks_connector_must_not_be_called",
        "pfc_databricks_cost_is_zero_only_while_all_execution_counters_remain_zero",
        "future_external_or_pfc_execution_requires_new_explicit_user_authorization",
        "external_data_engineer_cost_is_not_claimed_zero",
        "all_mutable_paths_must_remain_below_canonical_workspace_build",
        "stable_single_link_regular_files_required",
        "content_addressed_snapshot_directory_required",
    ):
        _equal(delivery.get(field), True, f"delivery {field}")
    for field in ("h_drive_access_authorized", "remote_write_authorized"):
        _equal(delivery.get(field), False, f"delivery {field}")
    _exact_sequence(
        delivery.get("accepted_data_formats"),
        ("PARQUET", "NDJSON"),
        "formats",
    )
    _equal(
        delivery.get("manifest_format"),
        "STRICT_UTF8_CANONICAL_JSON",
        "manifest format",
    )


def _artifact_map(value: object) -> dict[str, Mapping[str, object]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise GovernedAcquisitionPackageError("artifact inventory must be a sequence")
    artifacts: dict[str, Mapping[str, object]] = {}
    ordered_roles: list[str] = []
    for item in value:
        selected = _mapping(item, "artifact")
        role = selected.get("role")
        if not isinstance(role, str) or role in artifacts:
            raise GovernedAcquisitionPackageError("artifact roles must be unique strings")
        artifacts[role] = selected
        ordered_roles.append(role)
    _exact_keys(artifacts, set(ARTIFACT_ROLES), "artifact inventory")
    _equal(tuple(ordered_roles), ARTIFACT_ROLES, "artifact role order")
    return artifacts


def _validate_eex_artifact(eex: Mapping[str, object]) -> None:
    _equal(eex.get("source_mode"), "REUSE_D231_LOCAL_ARTIFACT_NO_NEW_QUERY", "EEX mode")
    _equal(eex.get("catalog"), "prd", "EEX catalog")
    _equal(eex.get("schema"), "gold", "EEX schema")
    _exact_sequence(eex.get("source_tables"), EEX_TABLES, "EEX tables")
    _equal(
        eex.get("artifact_path"),
        EEX_LOCAL_FILES["eex_artifact"][0],
        "EEX artifact path",
    )
    _equal(
        eex.get("artifact_sha256"),
        EEX_LOCAL_FILES["eex_artifact"][1],
        "EEX artifact SHA-256",
    )
    _equal(
        eex.get("source_manifest_path"),
        EEX_LOCAL_FILES["eex_source_manifest"][0],
        "EEX manifest path",
    )
    _equal(
        eex.get("source_manifest_sha256"),
        EEX_LOCAL_FILES["eex_source_manifest"][1],
        "EEX manifest SHA-256",
    )
    _equal(eex.get("row_count"), 82_552, "EEX row count")
    _exact_sequence(eex.get("grain"), EEX_GRAIN, "EEX grain")
    _exact_sequence(eex.get("required_fields"), EEX_FIELDS, "EEX fields")
    _exact_sequence(
        eex.get("required_product_families"), EEX_PRODUCT_FAMILIES, "EEX products"
    )
    _exact_sequence(
        eex.get("required_product_types"), ("BASE", "PEAK"), "EEX product types"
    )
    _equal(eex.get("price_unit"), "EUR/MWh", "EEX price unit")
    _equal(eex.get("new_databricks_statement_count"), 0, "EEX new statements")
    _equal(
        eex.get("historical_point_in_time_reconstruction_proven"),
        False,
        "EEX PIT",
    )


def _validate_entsoe_artifacts(artifacts: Mapping[str, Mapping[str, object]]) -> None:
    expected = {
        "entsoe_series_dimension": ("dimentsoeseries", ("series_id",)),
        "entsoe_latest_values": (
            "factentsoetimeserieslatest",
            ("series_id", "target_ts_utc"),
        ),
        "entsoe_vintage_values": (
            "factentsoetimeseriesvintages",
            ("series_id", "target_ts_utc", "as_of_utc"),
        ),
    }
    for role, (table, grain) in expected.items():
        artifact = artifacts[role]
        _equal(
            artifact.get("source_mode"),
            "FUTURE_NORMALIZED_EXPORT_AFTER_SCHEMA_ADMISSION",
            f"{role} mode",
        )
        _equal(artifact.get("catalog"), "dev", f"{role} catalog")
        _equal(artifact.get("schema"), "gold", f"{role} schema")
        _equal(artifact.get("source_table"), table, f"{role} table")
        _exact_sequence(artifact.get("grain"), grain, f"{role} grain")
        _exact_sequence(
            artifact.get("required_fields"), ENTSOE_FIELDS[role], f"{role} fields"
        )
        expected_path = {
            "entsoe_series_dimension": "data/entsoe_series_dimension.parquet",
            "entsoe_latest_values": "data/entsoe_latest_values.parquet",
            "entsoe_vintage_values": "data/entsoe_vintage_values.parquet",
        }[role]
        _equal(artifact.get("artifact_path"), expected_path, f"{role} artifact path")


def _validate_source_scope(scope: Mapping[str, object]) -> None:
    _equal(scope.get("eex_country"), "CH", "EEX country")
    _equal(scope.get("eex_commodity"), "POWER", "EEX commodity")
    _exact_sequence(
        scope.get("entsoe_required_borders"),
        ("CH-DE", "CH-FR", "CH-IT", "CH-AT"),
        "ENTSO-E borders",
    )
    for field in (
        "entsoe_required_swiss_groups_are_controlled_by_exact_intake_contract",
        "entsoe_generation_breakdown_is_controlled_by_exact_intake_contract",
        "all_available_in_scope_history_requested",
        "native_resolution_required_per_series",
        "historical_backfill_must_be_flagged",
        "swiss_day_ahead_price_grid_must_follow_applicable_market_time_regime",
        "historical_hourly_swiss_prices_are_not_a_missing_15_minute_data_defect",
        "short_tenor_shape_must_be_zero_mean_within_solver_month",
    ):
        _equal(scope.get(field), True, f"source scope {field}")
    for field in (
        "silent_truncation_or_sampling_authorized",
        "implicit_upsampling_or_forward_fill_authorized",
        "historical_backfill_may_reconstruct_past_information_sets",
    ):
        _equal(scope.get(field), False, f"source scope {field}")
    _equal(scope.get("monthly_level_authority"), "solver", "monthly level authority")


def _validate_cost_fence(fence: Mapping[str, object]) -> None:
    for field in (
        "this_contract_databricks_connections",
        "this_contract_databricks_statements",
        "this_contract_warehouse_starts",
        "this_contract_network_calls",
        "this_contract_h_drive_accesses",
        "this_contract_remote_writes",
        "this_contract_market_value_rows_opened",
    ):
        _integer_zero(fence.get(field), f"cost fence {field}")
    for field in (
        "this_contract_execution_enabled",
        "entsoe_physical_schema_admitted",
        "entsoe_value_query_generation_authorized",
        "select_star_authorized",
        "ddl_dml_merge_copy_unload_or_temp_objects_authorized",
        "partial_or_truncated_snapshot_admission_authorized",
    ):
        _equal(fence.get(field), False, f"cost fence {field}")
    _equal(
        fence.get(
            "future_capture_ceiling_is_inactive_until_schema_admission_and_new_authorization"
        ),
        True,
        "inactive future ceiling",
    )
    _equal(
        fence.get("future_max_capture_batches_per_europe_zurich_day"),
        1,
        "daily capture ceiling",
    )
    _equal(
        fence.get("future_max_warehouse_starts_per_capture_batch"),
        1,
        "warehouse ceiling",
    )
    _equal(
        fence.get("future_max_entsoe_read_only_statements_per_capture_batch"),
        3,
        "statement ceiling",
    )
    _equal(fence.get("future_max_statement_runtime_seconds"), 300, "statement runtime")
    _equal(fence.get("future_max_batch_runtime_seconds"), 900, "batch runtime")
    _equal(fence.get("future_max_retries"), 0, "retry ceiling")
    _equal(fence.get("future_warehouse_auto_stop_minutes_max"), 5, "auto-stop ceiling")
    _exact_sequence(
        fence.get("future_allowed_statement_classes"),
        ("SELECT", "WITH_SELECT"),
        "allowed statement classes",
    )
    for field in (
        "monetary_estimate_and_currency_cap_required_before_authorization",
        "query_sha256_and_schema_fingerprint_required_before_authorization",
        "failure_on_any_cap_exceedance_required",
    ):
        _equal(fence.get(field), True, f"cost fence {field}")
    total_cap = 68_719_476_736
    _equal(fence.get("max_total_local_export_bytes"), total_cap, "total byte cap")
    caps = _mapping(fence.get("per_artifact_hard_caps"), "artifact caps")
    _exact_keys(caps, set(ARTIFACT_ROLES), "artifact caps")
    expected_caps = {
        "eex_daily_snapshot": (100_000, 67_108_864),
        "entsoe_series_dimension": (1_000_000, 268_435_456),
        "entsoe_latest_values": (100_000_000, 17_179_869_184),
        "entsoe_vintage_values": (300_000_000, 50_465_865_728),
    }
    byte_total = 0
    for role, (rows, size) in expected_caps.items():
        cap = _mapping(caps.get(role), f"{role} cap")
        _exact_keys(cap, {"max_rows", "max_bytes"}, f"{role} cap")
        _equal(cap.get("max_rows"), rows, f"{role} row cap")
        _equal(cap.get("max_bytes"), size, f"{role} byte cap")
        byte_total += size
    if byte_total > total_cap:
        raise GovernedAcquisitionPackageError("per-artifact byte caps exceed total cap")


def _validate_manifest(manifest: Mapping[str, object]) -> None:
    _exact_sequence(
        manifest.get("batch_required_fields"),
        (
            "schema_version",
            "snapshot_id",
            "capture_batch_id",
            "snapshot_reference_time_utc",
            "challenge_id",
            "challenge_nonce_sha256",
            "capture_started_at_utc",
            "capture_completed_at_utc",
            "workspace_host_sha256",
            "warehouse_id_sha256",
            "warehouse_size",
            "warehouse_start_count",
            "read_statement_count",
            "retry_count",
            "total_runtime_seconds",
            "total_local_export_bytes",
            "artifact_manifest_sha256s",
            "source_envelope_sha256s",
        ),
        "batch manifest fields",
    )
    _exact_sequence(
        manifest.get("artifact_required_fields"),
        (
            "role",
            "catalog",
            "schema",
            "source_table_names",
            "source_watermarks",
            "physical_to_normalized_column_mapping",
            "normalized_schema_sha256",
            "query_text_sha256",
            "statement_id_sha256",
            "statement_class",
            "statement_started_at_utc",
            "statement_completed_at_utc",
            "row_count",
            "size_bytes",
            "minimum_timestamps_by_field",
            "maximum_timestamps_by_field",
            "artifact_relative_path",
            "artifact_sha256",
            "read_only_statement_verified",
            "backfill_status",
        ),
        "artifact manifest fields",
    )
    _equal(
        manifest.get("source_to_dimension_join_coverage_required"),
        1.0,
        "join coverage",
    )
    _equal(
        manifest.get("minimum_complete_days_for_seasonal_cross_market_diagnostic"),
        730,
        "complete days",
    )
    for field in (
        "primary_key_uniqueness_required",
        "required_fields_non_null_required",
        "finite_values_required",
        "expected_vs_observed_native_grid_required",
        "as_of_not_after_origin_replay_required",
        "artifact_hashes_must_verify_before_any_values_are_opened",
        "unknown_manifest_fields_fail_closed",
    ):
        _equal(manifest.get(field), True, f"manifest {field}")


def _validate_envelopes(envelopes: Mapping[str, object]) -> None:
    _exact_sequence(envelopes.get("required_roles"), REQUIRED_ROLES, "envelope roles")
    _equal(
        envelopes.get("exactly_one_dsse_envelope_per_role"),
        True,
        "one envelope per role",
    )
    _equal(
        envelopes.get("source_and_source_time_signers_must_be_pairwise_distinct"),
        True,
        "distinct signers",
    )
    _equal(
        envelopes.get("source_envelope_signatures_are_not_self_declared_authority"),
        True,
        "self-declared authority",
    )
    _equal(
        envelopes.get("payload_type"),
        "application/vnd.in-toto+json",
        "DSSE payload type",
    )
    _equal(
        envelopes.get("statement_schema"),
        "https://in-toto.io/Statement/v1",
        "in-toto statement schema",
    )
    for field in (
        "subject_digests_bind_exact_batch_and_artifact_manifests",
        "predicate_binds_capture_batch_challenge_query_schema_artifact_and_source_watermarks",
        "all_payloads_bind_same_capture_batch_id_and_challenge_nonce_sha256",
        "signer_key_must_resolve_to_same_named_d229_registry_role",
        "all_envelope_sha256s_must_be_computed_over_exact_dsse_bytes",
    ):
        _equal(envelopes.get(field), True, f"source envelopes {field}")


def _validate_external_time(external: Mapping[str, object]) -> None:
    _exact_sequence(
        external.get("allowed_profiles"),
        ("RFC3161_BATCH_ROOT", "TRANSPARENCY_LOG_BATCH_ROOT"),
        "external time profiles",
    )
    _exact_sequence(
        external.get("leaf_roles_in_canonical_order"),
        REQUIRED_ROLES,
        "timestamp leaf roles",
    )
    _equal(
        external.get("leaf_payload_schema"),
        "fmv_source_envelope_timestamp_leaf.v1",
        "timestamp leaf schema",
    )
    _exact_sequence(
        external.get("leaf_payload_fields"),
        (
            "schema_version",
            "capture_batch_id",
            "challenge_nonce_sha256",
            "envelope_role",
            "envelope_sha256",
        ),
        "timestamp leaf fields",
    )
    _equal(
        external.get("leaf_and_node_hashing"),
        "RFC9162_SHA256_0X00_LEAF_0X01_NODE",
        "Merkle hashing",
    )
    _equal(external.get("challenge_nonce_minimum_bits"), 128, "challenge nonce bits")
    _equal(
        external.get("maximum_external_time_lag_after_last_envelope_seconds"),
        3600,
        "external time lag",
    )
    for field in (
        "exactly_one_profile_per_capture_batch",
        "common_unpredictable_governed_challenge_required_before_capture",
        "challenge_issuance_time_and_issuer_signature_required",
        "one_inclusion_proof_per_exact_envelope_leaf_required",
        "rfc3161_message_imprint_must_equal_exact_batch_merkle_root",
        "rfc3161_nonce_policy_certificate_chain_gen_time_accuracy_and_validation_receipt_required",
        "transparency_entry_must_bind_exact_batch_merkle_root",
        "transparency_inclusion_proof_signed_checkpoint_pinned_log_key_integrated_time_and_consistency_evidence_required",
        "source_role_registry_keys_must_be_valid_for_entire_challenge_to_external_time_bracket",
        "proves_existence_no_later_than_external_time_not_exact_signature_creation_time",
        "governed_challenge_provides_only_a_verified_not-before-bracket",
        "historical_rows_preceding_challenge_do_not_gain_retroactive_point_in_time_proof",
        "future_training_and_selection_envelopes_require_a_separate_causally_later_external_time_batch",
    ):
        _equal(external.get(field), True, f"external time {field}")
    _equal(external.get("profile_mixing_authorized"), False, "profile mixing")


def _read_stable_workspace_file(
    root: Path,
    relative: str,
    *,
    maximum_bytes: int = 1_048_576,
) -> bytes:
    candidate = root / Path(relative)
    if candidate.is_symlink():
        raise GovernedAcquisitionPackageError(f"symlink input forbidden: {relative}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise GovernedAcquisitionPackageError(f"input escapes workspace: {relative}") from exc
    if not resolved.is_file():
        raise GovernedAcquisitionPackageError(f"input is not a regular file: {relative}")
    stat_before = resolved.stat()
    if stat_before.st_nlink != 1 or stat_before.st_size > maximum_bytes:
        raise GovernedAcquisitionPackageError(f"unstable or oversized input: {relative}")
    payload = resolved.read_bytes()
    stat_after = resolved.stat()
    if (
        stat_before.st_size != stat_after.st_size
        or stat_before.st_mtime_ns != stat_after.st_mtime_ns
        or stat_before.st_ino != stat_after.st_ino
    ):
        raise GovernedAcquisitionPackageError(f"input changed while hashing: {relative}")
    return payload


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise GovernedAcquisitionPackageError(f"{label} must be a string-keyed mapping")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise GovernedAcquisitionPackageError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _exact_sequence(value: object, expected: Sequence[str], label: str) -> None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise GovernedAcquisitionPackageError(f"{label} must be a sequence")
    if tuple(value) != tuple(expected):
        raise GovernedAcquisitionPackageError(
            f"{label} differs: expected {list(expected)!r}, received {list(value)!r}"
        )


def _integer_zero(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value != 0:
        raise GovernedAcquisitionPackageError(f"{label} must be integer zero")


def _integer_zero(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value != 0:
        raise GovernedAcquisitionPackageError(f"{label} must be integer zero")


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise GovernedAcquisitionPackageError(
            f"{label} differs: expected {expected!r}, received {value!r}"
        )


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
