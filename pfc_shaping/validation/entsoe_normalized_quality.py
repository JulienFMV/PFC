"""Offline quality profiling for normalized governed ENTSO-E frames."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pfc_shaping.path_safety import read_stable_single_link_file

CONTRACT_SCHEMA = "fmv_entsoe_normalized_quality_profile_contract.v1"
CONTRACT_STATUS = "STATIC_OFFLINE_NORMALIZED_QUALITY_CONTRACT_NO_MODEL_AUTHORITY"
CONTRACT_FILE_SHA256 = (
    "3cfe59d8700e8b29662918de3dc2941678267520353fc271c852874321f345e1"
)
CONTRACT_CONTENT_ID = (
    "7ff2d5b7808f636f2c181fa92b80cc4ae86dccdcaad84f6da8dc3d13499736ab"
)
INTAKE_CONTRACT_CONTENT_ID = (
    "75cb7e9f5d4cedb55e31fbd26b0bdcdf4ff629caf641c984cfec3fc6510b2a60"
)
INTAKE_CONTRACT_SHA256 = (
    "7ede1698099390babfa1d130bfecae61fd1e090888a3d6e6e4f892119db52b87"
)
SOURCE_PROOFS = {
    "d233": {
        "content_id": (
            "314ec85590c787874e2844d7db085236144c601681a5d1722f2735e6b1219d53"
        ),
        "manifest_sha256": (
            "d7f6ad3af60d2efd087718e0515c8b73a6534387ab12c7c7a5f88f42aeadf2b4"
        ),
    },
    "d234": {
        "content_id": (
            "bb7ea1894463cb2c5fc30287d2239f0285cd6a74e901ab51a2f9de7e6794b766"
        ),
        "manifest_sha256": (
            "7d86330e2143e2009947b486eb1e8428dd6022735ebc33c9aa4cfc4b871c5268"
        ),
    },
    "d235": {
        "content_id": (
            "93504d69834b299ce361352adce433509cf84bce0fbbb940be00a9bd59616ff1"
        ),
        "manifest_sha256": (
            "a10bbfa380652365684213197862139bf958dc33e706d81635cc2a6ae4582f02"
        ),
    },
    "d236": {
        "content_id": (
            "772da4e0b22540bf22e1715ca146cc0a59adf0c7a9ec508e5a893d8495539247"
        ),
        "manifest_sha256": (
            "9d76631395176718b600ca0de8f7303ad8faea931c24298f514a54354da3d6e0"
        ),
    },
    "d237": {
        "content_id": (
            "53e2222392f71541d28e05e1dfc912361c02003b4278f2770109827319d4e9c1"
        ),
        "manifest_sha256": (
            "c5d8be620f1c2f5012aa7174a5949adbc0d74708a162ac215c44f40f1f8dcdcf"
        ),
    },
    "d238": {
        "content_id": (
            "2b9f6c513e0382e685bee78fb02d6c071a6954a26e715c5784fb28efec878aa8"
        ),
        "manifest_sha256": (
            "922a31ce0bf54b28e00b352bae1e4fa1d66fa357adf687c9a5f456e94d4511f6"
        ),
    },
}

EVIDENCE_CLASSES = frozenset(
    {"SYNTHETIC_TEST_ONLY", "REAL_HASH_ADMITTED_ARTIFACTS"}
)
ROLE_FIELDS = {
    "series_dimension": (
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
    "latest_values": (
        "series_id",
        "target_ts_utc",
        "value",
        "is_historical",
        "quality_flag",
        "revision_number",
        "meta_load_timestamp_utc",
    ),
    "vintage_values": (
        "series_id",
        "target_ts_utc",
        "as_of_utc",
        "value",
        "quality_flag",
        "revision_number",
        "meta_load_timestamp_utc",
    ),
}
ROLE_GRAINS = {
    "series_dimension": ("series_id",),
    "latest_values": ("series_id", "target_ts_utc"),
    "vintage_values": ("series_id", "target_ts_utc", "as_of_utc"),
}
REQUIRED_GROUPS = (
    "actual_load",
    "day_ahead_prices",
    "generation_actual",
    "generation_forecast",
    "hydro_reservoir_storage",
    "load_forecast",
    "renewables_forecast_day_ahead",
    "renewables_forecast_intraday",
    "crossborder_physical_flows",
    "scheduled_exchanges",
    "ntc_day_ahead",
    "ntc_month_ahead",
    "ntc_year_ahead",
)
REQUIRED_BORDERS = ("CH-DE", "CH-FR", "CH-IT", "CH-AT")
REQUIRED_NEIGHBORS = frozenset({"DE", "FR", "IT", "AT"})
CROSSBORDER_GROUPS = frozenset(
    {
        "crossborder_physical_flows",
        "scheduled_exchanges",
        "ntc_day_ahead",
        "ntc_month_ahead",
        "ntc_year_ahead",
    }
)
PRE_TARGET_GROUPS = frozenset(
    {
        "day_ahead_prices",
        "generation_forecast",
        "load_forecast",
        "renewables_forecast_day_ahead",
        "renewables_forecast_intraday",
        "scheduled_exchanges",
        "ntc_day_ahead",
        "ntc_month_ahead",
        "ntc_year_ahead",
    }
)
POST_TARGET_GROUPS = frozenset(
    {
        "actual_load",
        "generation_actual",
        "hydro_reservoir_storage",
        "crossborder_physical_flows",
    }
)
REQUIRED_GENERATION_FIELDS = frozenset(
    {
        "solar",
        "wind",
        "nuclear",
        "hydro_run_of_river",
        "hydro_reservoir",
        "hydro_pumped_storage",
    }
)
ALLOWED_UNITS = frozenset({"MW", "MWh", "EUR/MWh"})
RESOLUTION_SECONDS = {
    "PT15M": 900,
    "PT30M": 1_800,
    "PT60M": 3_600,
    "PT1H": 3_600,
    "P1D": 86_400,
}
SIGN_SEMANTICS = frozenset(
    {
        "NON_NEGATIVE_PHYSICAL_QUANTITY",
        "SIGNED_POSITIVE_CH_TO_NEIGHBOR",
        "SIGNED_POSITIVE_NEIGHBOR_TO_CH",
        "SIGNED_SOURCE_DEFINED_DOCUMENTED",
        "PRICE_CAN_BE_NEGATIVE",
    }
)
SIGNED_SEMANTICS = frozenset(
    {
        "SIGNED_POSITIVE_CH_TO_NEIGHBOR",
        "SIGNED_POSITIVE_NEIGHBOR_TO_CH",
        "SIGNED_SOURCE_DEFINED_DOCUMENTED",
    }
)
BACKFILL_STATUSES = frozenset(
    {"NATIVE_VINTAGES", "HISTORICAL_BACKFILL", "MIXED_NATIVE_AND_BACKFILL"}
)
MIN_SEASONAL_COMPLETE_DAYS = 730


class EntsoeNormalizedQualityError(ValueError):
    """Raised when normalized ENTSO-E evidence fails a hard quality gate."""


@dataclass(frozen=True)
class EntsoeNormalizedQualityProfile:
    """Quality summary plus value-free per-series and overlap profiles."""

    summary: Mapping[str, object]
    series_profile: pd.DataFrame
    overlap_profile: pd.DataFrame


def validate_normalized_quality_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact D239 static contract and its zero-execution boundary."""

    contract = _mapping(document, "normalized quality contract")
    _equal(contract.get("schema_version"), CONTRACT_SCHEMA, "contract schema")
    _equal(contract.get("status"), CONTRACT_STATUS, "contract status")
    bindings = _mapping(contract.get("input_bindings"), "input bindings")
    _equal(
        bindings.get("governed_acquisition_package_proof_content_id"),
        SOURCE_PROOFS["d233"]["content_id"],
        "D233 proof content ID",
    )
    _equal(
        bindings.get("physical_mapping_compiler_proof_content_id"),
        SOURCE_PROOFS["d234"]["content_id"],
        "D234 proof content ID",
    )
    _equal(
        bindings.get("external_time_batch_proof_content_id"),
        SOURCE_PROOFS["d235"]["content_id"],
        "D235 proof content ID",
    )
    _equal(
        bindings.get("bounded_execution_receipt_proof_content_id"),
        SOURCE_PROOFS["d236"]["content_id"],
        "D236 proof content ID",
    )
    _equal(
        bindings.get("rfc3161_request_der_proof_content_id"),
        SOURCE_PROOFS["d237"]["content_id"],
        "D237 proof content ID",
    )
    _equal(
        bindings.get("daily_capture_reservation_proof_content_id"),
        SOURCE_PROOFS["d238"]["content_id"],
        "D238 proof content ID",
    )
    _equal(
        bindings.get("entsoe_intake_contract_content_id"),
        INTAKE_CONTRACT_CONTENT_ID,
        "ENTSO-E intake contract content ID",
    )
    roles = _mapping(contract.get("normalized_roles"), "normalized roles")
    _equal(set(roles), set(ROLE_FIELDS), "normalized role names")
    for role in ROLE_FIELDS:
        spec = _mapping(roles.get(role), f"role {role}")
        _equal(spec.get("grain"), list(ROLE_GRAINS[role]), f"grain {role}")
        _equal(spec.get("exact_fields"), list(ROLE_FIELDS[role]), f"fields {role}")
    execution = _mapping(contract.get("execution_policy"), "execution policy")
    for field in (
        "current_databricks_connection_budget",
        "current_databricks_statement_budget",
        "current_warehouse_start_budget",
        "current_network_call_budget",
        "current_h_drive_access_budget",
        "current_remote_write_budget",
    ):
        _equal(execution.get(field), 0, f"execution policy {field}")
    _equal(execution.get("connector_code_allowed"), False, "connector authority")
    _equal(
        execution.get("artifact_path_opening_authorized_by_static_contract"),
        False,
        "artifact opening authority",
    )
    authorities = _mapping(contract.get("authorities"), "authorities")
    if not authorities or any(value is not False for value in authorities.values()):
        raise EntsoeNormalizedQualityError("all contract authorities must be false")
    content_id = canonical_sha256(contract)
    _equal(content_id, CONTRACT_CONTENT_ID, "contract content ID")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": content_id,
        "databricks_connection_budget": 0,
        "databricks_statement_budget": 0,
        "warehouse_start_budget": 0,
        "artifact_opening_authorized": False,
        "model_input_authorized": False,
        "production_authorized": False,
    }


def profile_entsoe_normalized_quality(
    series_dimension: pd.DataFrame,
    latest_values: pd.DataFrame,
    vintage_values: pd.DataFrame,
    *,
    evidence_class: str,
    snapshot_reference_time_utc: datetime,
    sign_semantics_by_series: Mapping[str, str],
    backfill_status_by_series: Mapping[str, str],
    artifact_hashes_verified: bool,
    real_receipt_admitted: bool,
) -> EntsoeNormalizedQualityProfile:
    """Profile normalized frames while preserving strict model non-authority."""

    if evidence_class not in EVIDENCE_CLASSES:
        raise EntsoeNormalizedQualityError("evidence class is invalid")
    real = evidence_class == "REAL_HASH_ADMITTED_ARTIFACTS"
    if real:
        raise EntsoeNormalizedQualityError(
            "real evidence admission is unavailable until an independent "
            "receipt and artifact verifier is composed"
        )
    if artifact_hashes_verified is not False or real_receipt_admitted is not False:
        raise EntsoeNormalizedQualityError(
            "caller booleans cannot bootstrap hash or receipt admission"
        )
    snapshot = _utc_datetime(
        snapshot_reference_time_utc,
        "snapshot_reference_time_utc",
    )
    frames = {
        "series_dimension": _copy_frame(series_dimension, "series_dimension"),
        "latest_values": _copy_frame(latest_values, "latest_values"),
        "vintage_values": _copy_frame(vintage_values, "vintage_values"),
    }
    for role, frame in frames.items():
        _validate_frame_structure(role, frame)
    dimension = frames["series_dimension"]
    latest = frames["latest_values"]
    vintage = frames["vintage_values"]

    series_ids = _validate_dimension(dimension)
    semantics = _validate_exact_series_mapping(
        sign_semantics_by_series,
        series_ids,
        SIGN_SEMANTICS,
        "sign semantics",
    )
    backfill = _validate_exact_series_mapping(
        backfill_status_by_series,
        series_ids,
        BACKFILL_STATUSES,
        "backfill status",
    )
    _validate_semantics(dimension, semantics)
    _validate_value_frame("latest_values", latest, series_ids, snapshot)
    _validate_value_frame("vintage_values", vintage, series_ids, snapshot)
    _validate_dimension_value_coverage(series_ids, latest, vintage)
    _validate_value_signs(dimension, latest, vintage, semantics)
    _validate_group_availability(dimension, vintage)
    _validate_vintage_temporal_order(vintage)
    overlap_profile = _latest_vintage_overlap(latest, vintage)
    if bool(overlap_profile["value_equal"].eq(False).any()):
        raise EntsoeNormalizedQualityError(
            "latest values differ from last vintage on overlapping keys"
        )

    series_profile = _series_grid_profile(dimension, latest, vintage, backfill)
    minimum_complete_days = int(series_profile["complete_utc_days"].min())
    seasonal_coverage_met = minimum_complete_days >= MIN_SEASONAL_COMPLETE_DAYS
    overlap_keys = len(overlap_profile)
    overlap_coverage = float(overlap_keys / len(latest))
    backfill_series = sorted(
        series_id
        for series_id, status in backfill.items()
        if status != "NATIVE_VINTAGES"
    )
    findings: list[dict[str, object]] = []
    if not real:
        findings.append(
            {
                "code": "SYNTHETIC_FIXTURE_NOT_EMPIRICAL_EVIDENCE",
                "severity": "CRITICAL",
                "evidence": {"evidence_class": evidence_class},
                "impact": "The profile cannot support model choice or validation.",
            }
        )
    if not seasonal_coverage_met:
        findings.append(
            {
                "code": "SEASONAL_COMPLETE_DAY_THRESHOLD_NOT_MET",
                "severity": "HIGH",
                "evidence": {
                    "required_complete_days": MIN_SEASONAL_COMPLETE_DAYS,
                    "minimum_observed_complete_days": minimum_complete_days,
                },
                "impact": (
                    "Cross-market seasonal diagnostics and rolling-origin "
                    "selection remain unsupported."
                ),
            }
        )
    if backfill_series:
        findings.append(
            {
                "code": "BACKFILLED_SERIES_NOT_RETROACTIVE_PIT_EVIDENCE",
                "severity": "HIGH",
                "evidence": {"series_count": len(backfill_series)},
                "impact": (
                    "Historical rows before the earliest trustworthy as-of "
                    "cannot enter historical origin replays."
                ),
            }
        )
    if overlap_coverage < 1.0:
        findings.append(
            {
                "code": "LATEST_TO_VINTAGE_OVERLAP_INCOMPLETE",
                "severity": "HIGH",
                "evidence": {"overlap_coverage_ratio": overlap_coverage},
                "impact": (
                    "Latest history without a vintage counterpart cannot "
                    "reconstruct earlier information sets."
                ),
            }
        )
    findings.append(
        {
            "code": "NEW_INDEPENDENT_FUTURE_HOLDOUT_MISSING",
            "severity": "CRITICAL",
            "evidence": {"prospective_holdout_count": 0},
            "impact": "Training, selection and production admission remain forbidden.",
        }
    )

    summary = {
        "schema_version": "fmv_entsoe_normalized_quality_profile.v1",
        "status": "PASS_SYNTHETIC_STRUCTURAL_PROFILE_MODEL_NO_GO",
        "evidence_class": evidence_class,
        "snapshot_reference_time_utc": _format_utc(snapshot),
        "dataset": {
            "dimension_row_count": len(dimension),
            "latest_row_count": len(latest),
            "vintage_row_count": len(vintage),
            "series_count": len(series_ids),
            "latest_vintage_overlap_key_count": overlap_keys,
        },
        "checks": {
            "exact_normalized_fields": True,
            "primary_keys_unique": True,
            "required_fields_non_null": True,
            "finite_values": True,
            "dimension_join_coverage": 1.0,
            "dimension_to_latest_series_coverage": 1.0,
            "dimension_to_vintage_series_coverage": 1.0,
            "required_groups_present": True,
            "required_borders_by_crossborder_group_present": True,
            "required_generation_breakdown_present": True,
            "units_valid": True,
            "native_grid_alignment_valid": True,
            "latest_vintage_overlap_mismatch_count": 0,
            "seasonal_complete_day_threshold_met": seasonal_coverage_met,
        },
        "coverage": {
            "minimum_complete_utc_days": minimum_complete_days,
            "minimum_grid_completeness_ratio": float(
                series_profile["grid_completeness_ratio"].min()
            ),
            "latest_to_vintage_overlap_coverage_ratio": overlap_coverage,
            "backfill_or_mixed_series_count": len(backfill_series),
        },
        "findings": findings,
        "authorities": {
            "real_artifact_receipt_admitted": False,
            "real_artifact_hashes_verified": False,
            "real_entsoe_values_opened": False,
            "same_snapshot_point_in_time_availability_proven": False,
            "seasonal_cross_market_diagnostic_authorized": False,
            "training_authorized": False,
            "selection_authorized": False,
            "model_input_authorized": False,
            "candidate_assembly_authorized": False,
            "promotion_authorized": False,
            "production_authorized": False,
        },
        "execution": {
            "databricks_connection_count": 0,
            "databricks_statement_count": 0,
            "warehouse_start_count": 0,
            "network_call_count": 0,
            "h_drive_access_count": 0,
            "remote_write_count": 0,
        },
    }
    return EntsoeNormalizedQualityProfile(
        summary=summary,
        series_profile=series_profile,
        overlap_profile=overlap_profile,
    )


def replay_vintages_as_of(
    vintage_values: pd.DataFrame,
    *,
    origin_utc: datetime,
) -> pd.DataFrame:
    """Select the last eligible vintage per target without future leakage."""

    origin = _utc_datetime(origin_utc, "origin_utc")
    vintage = _copy_frame(vintage_values, "vintage_values")
    _validate_frame_structure("vintage_values", vintage)
    _validate_utc_column(vintage, "as_of_utc", "vintage_values")
    eligible = vintage.loc[vintage["as_of_utc"].le(origin)].copy()
    if eligible.empty:
        return eligible
    eligible = eligible.sort_values(
        ["series_id", "target_ts_utc", "as_of_utc", "revision_number"],
        kind="mergesort",
    )
    result = eligible.drop_duplicates(
        ["series_id", "target_ts_utc"],
        keep="last",
    ).reset_index(drop=True)
    if bool(result["as_of_utc"].gt(origin).any()):
        raise EntsoeNormalizedQualityError("rolling-origin replay leaked future rows")
    return result


def verify_normalized_quality_source_paths(
    *,
    contract_path: str | Path,
    intake_contract_path: str | Path,
    source_proof_manifest_paths: Mapping[str, str | Path],
) -> dict[str, object]:
    """Bind the exact local D233-D239 contracts and proofs without artifacts."""

    contract_raw = _read_path(contract_path, "D239 quality contract", 100_000)
    if hashlib.sha256(contract_raw).hexdigest() != CONTRACT_FILE_SHA256:
        raise EntsoeNormalizedQualityError("D239 quality contract SHA-256 differs")
    contract = _strict_json(contract_raw, "D239 quality contract")
    assessment = validate_normalized_quality_contract(contract)
    intake_raw = _read_path(intake_contract_path, "ENTSO-E intake contract", 100_000)
    if hashlib.sha256(intake_raw).hexdigest() != INTAKE_CONTRACT_SHA256:
        raise EntsoeNormalizedQualityError("ENTSO-E intake contract SHA-256 differs")
    intake = _strict_json(intake_raw, "ENTSO-E intake contract")
    _equal(
        canonical_sha256(intake),
        INTAKE_CONTRACT_CONTENT_ID,
        "ENTSO-E intake contract content ID",
    )
    if set(source_proof_manifest_paths) != set(SOURCE_PROOFS):
        raise EntsoeNormalizedQualityError("source proof manifest roles differ")
    for role, expected in SOURCE_PROOFS.items():
        raw = _read_path(
            source_proof_manifest_paths[role],
            f"{role.upper()} proof manifest",
            100_000,
        )
        if hashlib.sha256(raw).hexdigest() != expected["manifest_sha256"]:
            raise EntsoeNormalizedQualityError(f"{role.upper()} proof hash differs")
        manifest = _strict_json(raw, f"{role.upper()} proof manifest")
        _equal(manifest.get("content_id"), expected["content_id"], f"{role} content ID")
        execution = _mapping(manifest.get("execution"), f"{role} execution")
        if any(value != 0 for value in execution.values()):
            raise EntsoeNormalizedQualityError(
                f"{role.upper()} proof execution counters are not zero"
            )
    return {
        **assessment,
        "source_proof_content_ids": {
            role: value["content_id"] for role, value in SOURCE_PROOFS.items()
        },
    }


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _copy_frame(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise EntsoeNormalizedQualityError(f"{role} must be a non-empty DataFrame")
    return frame.copy(deep=True)


def _validate_frame_structure(role: str, frame: pd.DataFrame) -> None:
    expected = list(ROLE_FIELDS[role])
    if list(frame.columns) != expected or frame.columns.has_duplicates:
        raise EntsoeNormalizedQualityError(f"{role} fields differ")
    if bool(frame.isna().any().any()):
        raise EntsoeNormalizedQualityError(f"{role} contains null required fields")
    if bool(frame.duplicated(list(ROLE_GRAINS[role]), keep=False).any()):
        raise EntsoeNormalizedQualityError(f"{role} primary key is duplicated")


def _validate_dimension(dimension: pd.DataFrame) -> frozenset[str]:
    string_columns = ROLE_FIELDS["series_dimension"]
    for column in string_columns:
        _validate_nonempty_strings(dimension, column, "series_dimension")
    normalized_ids = dimension["series_id"].str.strip().str.casefold()
    if bool(normalized_ids.duplicated(keep=False).any()):
        raise EntsoeNormalizedQualityError(
            "series identifiers are duplicated after trim and casefold"
        )
    groups = set(dimension["group_name"])
    missing_groups = sorted(set(REQUIRED_GROUPS).difference(groups))
    if missing_groups:
        raise EntsoeNormalizedQualityError(f"required groups missing: {missing_groups}")
    units = set(dimension["unit"])
    if not units.issubset(ALLOWED_UNITS) or "EUR" in units:
        raise EntsoeNormalizedQualityError("dimension unit is invalid")
    resolutions = set(dimension["native_resolution"])
    if not resolutions.issubset(RESOLUTION_SECONDS):
        raise EntsoeNormalizedQualityError("native resolution is invalid")
    zones = set(dimension["from_zone"]) | set(dimension["to_zone"])
    if not zones.issubset({"CH", *REQUIRED_NEIGHBORS}):
        raise EntsoeNormalizedQualityError("dimension zone is invalid")
    for row in dimension.itertuples(index=False):
        expected_unit = (
            "EUR/MWh"
            if row.group_name == "day_ahead_prices"
            else "MWh"
            if row.group_name == "hydro_reservoir_storage"
            else "MW"
        )
        if row.unit != expected_unit:
            raise EntsoeNormalizedQualityError(
                f"unit differs for group {row.group_name}: {row.unit}"
            )
    generation = set(
        dimension.loc[
            dimension["group_name"].eq("generation_actual"),
            "field_name",
        ]
    )
    if not REQUIRED_GENERATION_FIELDS.issubset(generation):
        raise EntsoeNormalizedQualityError("generation actual breakdown is incomplete")
    for group in CROSSBORDER_GROUPS:
        rows = dimension.loc[dimension["group_name"].eq(group)]
        observed = {
            _border(row.from_zone, row.to_zone)
            for row in rows.itertuples(index=False)
        }
        if observed != set(REQUIRED_BORDERS):
            raise EntsoeNormalizedQualityError(
                f"required border coverage differs for {group}: {sorted(observed)}"
            )
    return frozenset(str(value) for value in dimension["series_id"])


def _validate_semantics(
    dimension: pd.DataFrame,
    semantics: Mapping[str, str],
) -> None:
    metadata = dimension.set_index("series_id").to_dict(orient="index")
    for series_id, row in metadata.items():
        group = row["group_name"]
        semantic = semantics[str(series_id)]
        if group == "day_ahead_prices":
            _equal(semantic, "PRICE_CAN_BE_NEGATIVE", f"price semantic {series_id}")
        elif group in CROSSBORDER_GROUPS:
            if semantic not in SIGNED_SEMANTICS:
                raise EntsoeNormalizedQualityError(
                    f"cross-border series requires signed semantics: {series_id}"
                )
            if (
                semantic == "SIGNED_POSITIVE_CH_TO_NEIGHBOR"
                and row["from_zone"] != "CH"
            ):
                raise EntsoeNormalizedQualityError(
                    f"cross-border sign direction differs: {series_id}"
                )
            if (
                semantic == "SIGNED_POSITIVE_NEIGHBOR_TO_CH"
                and row["to_zone"] != "CH"
            ):
                raise EntsoeNormalizedQualityError(
                    f"cross-border sign direction differs: {series_id}"
                )
        elif semantic == "PRICE_CAN_BE_NEGATIVE":
            raise EntsoeNormalizedQualityError(
                f"non-price series uses price semantics: {series_id}"
            )


def _validate_value_frame(
    role: str,
    frame: pd.DataFrame,
    series_ids: frozenset[str],
    snapshot: datetime,
) -> None:
    _validate_nonempty_strings(frame, "series_id", role)
    _validate_nonempty_strings(frame, "quality_flag", role)
    unknown = sorted(set(frame["series_id"]).difference(series_ids))
    if unknown:
        raise EntsoeNormalizedQualityError(f"{role} has orphan series: {unknown}")
    _validate_utc_column(frame, "target_ts_utc", role)
    _validate_utc_column(frame, "meta_load_timestamp_utc", role)
    if role == "vintage_values":
        _validate_utc_column(frame, "as_of_utc", role)
        if bool(frame["as_of_utc"].gt(snapshot).any()):
            raise EntsoeNormalizedQualityError("vintage as_of exceeds snapshot")
        if bool(frame["as_of_utc"].gt(frame["meta_load_timestamp_utc"]).any()):
            raise EntsoeNormalizedQualityError("vintage as_of exceeds meta load")
    if bool(frame["meta_load_timestamp_utc"].gt(snapshot).any()):
        raise EntsoeNormalizedQualityError(f"{role} meta load exceeds snapshot")
    if not pd.api.types.is_numeric_dtype(frame["value"].dtype):
        raise EntsoeNormalizedQualityError(f"{role} value must be numeric")
    numeric = frame["value"].astype(float)
    if not bool(numeric.map(math.isfinite).all()):
        raise EntsoeNormalizedQualityError(f"{role} value is non-finite")
    revision = frame["revision_number"]
    if not pd.api.types.is_integer_dtype(revision.dtype) or bool(revision.lt(0).any()):
        raise EntsoeNormalizedQualityError(
            f"{role} revision_number must be a non-negative integer"
        )
    if role == "latest_values" and not pd.api.types.is_bool_dtype(
        frame["is_historical"].dtype
    ):
        raise EntsoeNormalizedQualityError("latest is_historical must be boolean")


def _validate_dimension_value_coverage(
    series_ids: frozenset[str],
    latest: pd.DataFrame,
    vintage: pd.DataFrame,
) -> None:
    latest_ids = frozenset(str(value) for value in latest["series_id"])
    vintage_ids = frozenset(str(value) for value in vintage["series_id"])
    if latest_ids != series_ids:
        raise EntsoeNormalizedQualityError("dimension to latest series coverage differs")
    if vintage_ids != series_ids:
        raise EntsoeNormalizedQualityError("dimension to vintage series coverage differs")


def _validate_value_signs(
    dimension: pd.DataFrame,
    latest: pd.DataFrame,
    vintage: pd.DataFrame,
    semantics: Mapping[str, str],
) -> None:
    for series_id in dimension["series_id"]:
        semantic = semantics[str(series_id)]
        if semantic != "NON_NEGATIVE_PHYSICAL_QUANTITY":
            continue
        values = pd.concat(
            [
                latest.loc[latest["series_id"].eq(series_id), "value"],
                vintage.loc[vintage["series_id"].eq(series_id), "value"],
            ],
            ignore_index=True,
        )
        if bool(values.lt(0).any()):
            raise EntsoeNormalizedQualityError(
                f"negative value violates sign semantics: {series_id}"
            )


def _validate_vintage_temporal_order(vintage: pd.DataFrame) -> None:
    ordered = vintage.sort_values(
        ["series_id", "target_ts_utc", "as_of_utc"],
        kind="mergesort",
    )
    for _, group in ordered.groupby(["series_id", "target_ts_utc"], sort=False):
        if bool(group["revision_number"].diff().dropna().lt(0).any()):
            raise EntsoeNormalizedQualityError(
                "vintage revisions decrease as availability time advances"
            )
        if bool(group["meta_load_timestamp_utc"].diff().dropna().lt(pd.Timedelta(0)).any()):
            raise EntsoeNormalizedQualityError(
                "vintage meta-load time decreases as availability time advances"
            )


def _validate_group_availability(
    dimension: pd.DataFrame,
    vintage: pd.DataFrame,
) -> None:
    groups = dimension[["series_id", "group_name"]]
    enriched = vintage.merge(groups, on="series_id", how="left", validate="many_to_one")
    pre_target = enriched["group_name"].isin(PRE_TARGET_GROUPS)
    if bool(enriched.loc[pre_target, "as_of_utc"].gt(
        enriched.loc[pre_target, "target_ts_utc"]
    ).any()):
        raise EntsoeNormalizedQualityError(
            "forecast, price, schedule or capacity vintage is available after target"
        )
    post_target = enriched["group_name"].isin(POST_TARGET_GROUPS)
    if bool(enriched.loc[post_target, "as_of_utc"].lt(
        enriched.loc[post_target, "target_ts_utc"]
    ).any()):
        raise EntsoeNormalizedQualityError(
            "actual or storage vintage is available before target"
        )


def _latest_vintage_overlap(
    latest: pd.DataFrame,
    vintage: pd.DataFrame,
) -> pd.DataFrame:
    last = (
        vintage.sort_values(
            ["series_id", "target_ts_utc", "as_of_utc", "revision_number"],
            kind="mergesort",
        )
        .drop_duplicates(["series_id", "target_ts_utc"], keep="last")
        .rename(
            columns={
                "value": "vintage_value",
                "quality_flag": "vintage_quality_flag",
                "revision_number": "vintage_revision_number",
                "meta_load_timestamp_utc": "vintage_meta_load_timestamp_utc",
            }
        )
    )
    overlap = latest.merge(
        last[
            [
                "series_id",
                "target_ts_utc",
                "as_of_utc",
                "vintage_value",
                "vintage_quality_flag",
                "vintage_revision_number",
                "vintage_meta_load_timestamp_utc",
            ]
        ],
        on=["series_id", "target_ts_utc"],
        how="inner",
        validate="one_to_one",
    )
    overlap["value_equal"] = (
        overlap["value"].astype(float).eq(overlap["vintage_value"].astype(float))
        & overlap["quality_flag"].eq(overlap["vintage_quality_flag"])
        & overlap["revision_number"].eq(overlap["vintage_revision_number"])
        & overlap["meta_load_timestamp_utc"].eq(
            overlap["vintage_meta_load_timestamp_utc"]
        )
    )
    return overlap[
        ["series_id", "target_ts_utc", "as_of_utc", "value_equal"]
    ].sort_values(["series_id", "target_ts_utc"]).reset_index(drop=True)


def _series_grid_profile(
    dimension: pd.DataFrame,
    latest: pd.DataFrame,
    vintage: pd.DataFrame,
    backfill: Mapping[str, str],
) -> pd.DataFrame:
    metadata = dimension.set_index("series_id").to_dict(orient="index")
    rows: list[dict[str, object]] = []
    for series_id in sorted(metadata):
        role_latest = latest.loc[latest["series_id"].eq(series_id)].sort_values(
            "target_ts_utc"
        )
        targets = role_latest["target_ts_utc"]
        resolution = str(metadata[series_id]["native_resolution"])
        resolution_seconds = RESOLUTION_SECONDS[resolution]
        resolution_ns = resolution_seconds * 1_000_000_000
        target_ns = targets.astype("int64")
        if bool(target_ns.mod(resolution_ns).ne(0).any()):
            raise EntsoeNormalizedQualityError(
                f"target timestamp is off native grid: {series_id}"
            )
        differences = target_ns.diff().dropna()
        if bool(differences.le(0).any() or differences.mod(resolution_ns).ne(0).any()):
            raise EntsoeNormalizedQualityError(
                f"target grid step differs from native resolution: {series_id}"
            )
        minimum = targets.min()
        maximum = targets.max()
        expected_count = int((maximum - minimum).total_seconds() / resolution_seconds) + 1
        observed_count = len(targets)
        missing_count = expected_count - observed_count
        expected_per_day = 86_400 // resolution_seconds
        daily_counts = targets.groupby(targets.dt.floor("D")).nunique()
        complete_days = int(daily_counts.eq(expected_per_day).sum())
        vintage_rows = vintage.loc[vintage["series_id"].eq(series_id)]
        earliest_as_of = vintage_rows["as_of_utc"].min()
        rows.append(
            {
                "series_id": series_id,
                "group_name": metadata[series_id]["group_name"],
                "field_name": metadata[series_id]["field_name"],
                "border": _border(
                    str(metadata[series_id]["from_zone"]),
                    str(metadata[series_id]["to_zone"]),
                    allow_domestic=True,
                ),
                "unit": metadata[series_id]["unit"],
                "native_resolution": resolution,
                "latest_observed_count": observed_count,
                "latest_expected_count": expected_count,
                "missing_native_slot_count": missing_count,
                "grid_completeness_ratio": float(observed_count / expected_count),
                "complete_utc_days": complete_days,
                "minimum_target_utc": _format_utc(minimum.to_pydatetime()),
                "maximum_target_utc": _format_utc(maximum.to_pydatetime()),
                "vintage_row_count": len(vintage_rows),
                "earliest_trustworthy_as_of_utc": _format_utc(
                    earliest_as_of.to_pydatetime()
                ),
                "backfill_status": backfill[series_id],
                "raw_values_persisted_in_profile": False,
            }
        )
    return pd.DataFrame(rows).sort_values("series_id").reset_index(drop=True)


def _validate_exact_series_mapping(
    raw: Mapping[str, str],
    series_ids: frozenset[str],
    allowed: frozenset[str],
    label: str,
) -> dict[str, str]:
    mapping = _mapping(raw, label)
    if set(mapping) != set(series_ids):
        raise EntsoeNormalizedQualityError(f"{label} series coverage differs")
    result: dict[str, str] = {}
    for series_id, value in mapping.items():
        if not isinstance(value, str) or value not in allowed:
            raise EntsoeNormalizedQualityError(f"{label} is invalid: {series_id}")
        result[series_id] = value
    return result


def _validate_nonempty_strings(frame: pd.DataFrame, column: str, role: str) -> None:
    series = frame[column]
    if not pd.api.types.is_string_dtype(series.dtype) and series.dtype != object:
        raise EntsoeNormalizedQualityError(f"{role}.{column} must contain strings")
    if not bool(series.map(lambda value: isinstance(value, str) and value == value.strip() and bool(value)).all()):
        raise EntsoeNormalizedQualityError(
            f"{role}.{column} contains an empty or untrimmed string"
        )


def _validate_utc_column(frame: pd.DataFrame, column: str, role: str) -> None:
    dtype = frame[column].dtype
    if not isinstance(dtype, pd.DatetimeTZDtype) or str(frame[column].dt.tz) != "UTC":
        raise EntsoeNormalizedQualityError(f"{role}.{column} must be timezone-aware UTC")


def _border(from_zone: str, to_zone: str, *, allow_domestic: bool = False) -> str:
    zones = {from_zone, to_zone}
    if zones == {"CH"} and allow_domestic:
        return "CH-CH"
    if "CH" not in zones or len(zones) != 2:
        raise EntsoeNormalizedQualityError(
            f"border must pair CH with one neighbor: {from_zone}/{to_zone}"
        )
    neighbor = next(zone for zone in zones if zone != "CH")
    if neighbor not in REQUIRED_NEIGHBORS:
        raise EntsoeNormalizedQualityError(f"border neighbor is invalid: {neighbor}")
    return f"CH-{neighbor}"


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_datetime(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EntsoeNormalizedQualityError(f"{label} must be timezone-aware UTC")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise EntsoeNormalizedQualityError(f"{label} must be timezone-aware UTC")
    return value


def _read_path(raw_path: str | Path, label: str, max_bytes: int) -> bytes:
    path = Path(raw_path)
    if not path.is_absolute():
        raise EntsoeNormalizedQualityError(f"{label} path must be absolute")
    try:
        return read_stable_single_link_file(path, label=label, max_bytes=max_bytes)
    except (OSError, ValueError) as exc:
        raise EntsoeNormalizedQualityError(f"{label} path read failed") from exc


def _strict_json(payload: bytes, label: str) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise EntsoeNormalizedQualityError(f"{label} duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntsoeNormalizedQualityError(f"{label} JSON is invalid") from exc
    return _mapping(value, label)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EntsoeNormalizedQualityError(f"{label} must be a mapping")
    return value


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise EntsoeNormalizedQualityError(
            f"{label} differs: expected {expected!r}, received {value!r}"
        )


__all__ = [
    "CONTRACT_CONTENT_ID",
    "EntsoeNormalizedQualityError",
    "EntsoeNormalizedQualityProfile",
    "canonical_sha256",
    "profile_entsoe_normalized_quality",
    "replay_vintages_as_of",
    "validate_normalized_quality_contract",
    "verify_normalized_quality_source_paths",
]
