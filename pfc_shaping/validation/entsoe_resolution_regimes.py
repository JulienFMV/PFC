"""Piecewise native-resolution validation for normalized ENTSO-E series.

The algorithm is deliberately synthetic-only.  It prevents a later 15-minute
market regime from making valid hourly history look incomplete and prevents a
caller from manufacturing quarter-hourly observations by resampling.  An
explicit UTC window prevents leading or trailing truncation from shrinking the
completeness question to whatever rows happened to arrive.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

import pandas as pd

from pfc_shaping.path_safety import read_stable_single_link_file

CONTRACT_SCHEMA = "fmv_entsoe_resolution_regime_contract.v2"
CONTRACT_STATUS = "ALGORITHM_ONLY_EXPLICIT_WINDOW_NO_REAL_DATA_AUTHORITY"
CONTRACT_FILE_SHA256 = "a661cd047ab498e2286ce4f082bca345b75318a2e94aebf8f54d325279fd0e7d"
CONTRACT_CONTENT_ID = "450d3628bf7dc83e16d630fe001709b58561f74bdad66d73b155b199f5d96821"
PROFILE_SCHEMA = "fmv_entsoe_resolution_regime_profile.v1"
PASS_STATUS = "PASS_SYNTHETIC_RESOLUTION_REGIME_ALGORITHM"
GAP_STATUS = "FAIL_SYNTHETIC_RESOLUTION_REGIME_GAPS"
EVIDENCE_CLASS = "SYNTHETIC_TEST_ONLY"
MAX_EXPECTED_SLOTS = 5_000_000
RESOLUTION_SECONDS = {
    "PT15M": 15 * 60,
    "PT30M": 30 * 60,
    "PT1H": 60 * 60,
}
DIMENSION_COLUMNS = ("series_id", "group_name", "native_resolution")
TARGET_COLUMNS = ("series_id", "target_ts_utc")
REGIME_COLUMNS = (
    "series_id",
    "valid_from_utc",
    "valid_to_utc",
    "native_resolution",
    "source_document_id",
    "source_locator",
    "owner_confirmed_at_utc",
)
_SHA = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,254}$")


class EntsoeResolutionRegimeError(ValueError):
    """Raised when a cadence schedule is ambiguous, unsafe, or inconsistent."""


@dataclass(frozen=True)
class EntsoeResolutionRegimeProfile:
    """Value-free resolution profile with hashed physical series identities."""

    summary: dict[str, object]
    series_profile: tuple[dict[str, object], ...]


def canonical_sha256(value: Mapping[str, object]) -> str:
    """Return the canonical JSON identity of a mapping."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def validate_resolution_regime_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact execution-free D261 cadence-window contract."""

    contract = _mapping(document, "resolution regime contract")
    _exact(
        contract,
        {
            "schema_version",
            "decision",
            "status",
            "purpose",
            "supersession_scope",
            "source_findings",
            "normalized_sidecar",
            "quality_rules",
            "execution",
            "authorities",
        },
        "resolution regime contract",
    )
    _equal(contract["schema_version"], CONTRACT_SCHEMA, "contract schema")
    _equal(contract["decision"], "D-20260806-261", "contract decision")
    _equal(contract["status"], CONTRACT_STATUS, "contract status")
    _clean_text(contract["purpose"], "contract purpose")

    supersession = _mapping(contract["supersession_scope"], "supersession scope")
    _exact(
        supersession,
        {
            "supersedes_single_resolution_as_historical_truth",
            "supersedes_observed_bounds_as_completeness_window",
            "single_native_resolution_remains_current_or_single_regime_metadata_only",
            "does_not_supersede_family_unit_sign_lineage_quality_or_pit_gates",
        },
        "supersession scope",
    )
    if any(value is not True for value in supersession.values()):
        raise EntsoeResolutionRegimeError("supersession scope must be exact and true")

    findings = contract["source_findings"]
    if not isinstance(findings, list) or len(findings) != 2:
        raise EntsoeResolutionRegimeError("source findings must contain two sources")
    expected_sources = {
        "EPEX_SPOT_15_MINUTE_MARKET_TIME_UNIT",
        "SWISSGRID_BALANCING_ROADMAP_2026_2030",
    }
    observed_sources: set[str] = set()
    for position, raw in enumerate(findings, start=1):
        finding = _mapping(raw, f"source finding {position}")
        _exact(
            finding,
            {
                "source_id",
                "source_locator",
                "scope",
                "finding",
                "may_define_an_observed_go_live_date",
            },
            f"source finding {position}",
        )
        source_id = _clean_text(finding["source_id"], f"source {position} ID")
        observed_sources.add(source_id)
        _https(finding["source_locator"], f"source {position} locator")
        _clean_text(finding["scope"], f"source {position} scope")
        _clean_text(finding["finding"], f"source {position} finding")
        _equal(
            finding["may_define_an_observed_go_live_date"],
            False,
            f"source {position} go-live authority",
        )
    _equal(observed_sources, expected_sources, "source finding identities")

    sidecar = _mapping(contract["normalized_sidecar"], "normalized sidecar")
    _exact(
        sidecar,
        {
            "role",
            "grain",
            "required_fields",
            "interval_semantics",
            "timestamp_semantics",
            "supported_resolutions",
            "open_ended_final_regime_allowed",
            "gaps_or_overlaps_allowed",
            "one_country_wide_regime_substitution_allowed",
        },
        "normalized sidecar",
    )
    _equal(sidecar["role"], "series_resolution_regimes", "sidecar role")
    _equal(sidecar["grain"], ["series_id", "valid_from_utc"], "sidecar grain")
    _equal(sidecar["required_fields"], list(REGIME_COLUMNS), "sidecar fields")
    _equal(sidecar["interval_semantics"], "LEFT_CLOSED_RIGHT_OPEN", "interval semantics")
    _equal(
        sidecar["timestamp_semantics"],
        "DELIVERY_INTERVAL_START_UTC",
        "timestamp semantics",
    )
    _equal(
        sidecar["supported_resolutions"],
        list(RESOLUTION_SECONDS),
        "supported resolutions",
    )
    _equal(sidecar["open_ended_final_regime_allowed"], True, "open final regime")
    _equal(sidecar["gaps_or_overlaps_allowed"], False, "regime gap authority")
    _equal(
        sidecar["one_country_wide_regime_substitution_allowed"],
        False,
        "global regime authority",
    )

    rules = _mapping(contract["quality_rules"], "quality rules")
    _exact(
        rules,
        {
            "exact_series_coverage_required",
            "transition_boundary_must_close_previous_grid",
            "observed_target_must_match_active_regime_grid",
            "expected_slots_are_built_piecewise",
            "explicit_assessment_window_required",
            "leading_and_trailing_missing_slots_are_reported",
            "metadata_as_of_not_before_assessment_end",
            "dimension_resolution_checked_at_metadata_as_of",
            "metadata_as_of_may_be_off_delivery_grid",
            "owner_confirmation_not_after_metadata_as_of",
            "clean_https_source_locator_required",
            "missing_slots_are_reported_not_filled",
            "implicit_upsampling_authorized",
            "implicit_downsampling_authorized",
            "implicit_forward_fill_authorized",
            "historical_hourly_series_is_invalid_only_because_later_market_is_15_minute",
            "roadmap_or_calendar_date_may_replace_series_evidence",
            "maximum_expected_slots_per_assessment",
        },
        "quality rules",
    )
    for key in (
        "exact_series_coverage_required",
        "transition_boundary_must_close_previous_grid",
        "observed_target_must_match_active_regime_grid",
        "expected_slots_are_built_piecewise",
        "explicit_assessment_window_required",
        "leading_and_trailing_missing_slots_are_reported",
        "metadata_as_of_not_before_assessment_end",
        "dimension_resolution_checked_at_metadata_as_of",
        "metadata_as_of_may_be_off_delivery_grid",
        "owner_confirmation_not_after_metadata_as_of",
        "clean_https_source_locator_required",
        "missing_slots_are_reported_not_filled",
    ):
        _equal(rules[key], True, key)
    for key in (
        "implicit_upsampling_authorized",
        "implicit_downsampling_authorized",
        "implicit_forward_fill_authorized",
        "historical_hourly_series_is_invalid_only_because_later_market_is_15_minute",
        "roadmap_or_calendar_date_may_replace_series_evidence",
    ):
        _equal(rules[key], False, key)
    _equal(
        rules["maximum_expected_slots_per_assessment"],
        MAX_EXPECTED_SLOTS,
        "expected-slot cap",
    )

    execution = _mapping(contract["execution"], "execution counters")
    _exact(
        execution,
        {
            "databricks_connection_count",
            "databricks_statement_count",
            "warehouse_start_count",
            "databricks_write_count",
            "network_call_count",
            "h_drive_access_count",
            "opened_real_value_row_count",
        },
        "execution counters",
    )
    if not execution or any(value != 0 or isinstance(value, bool) for value in execution.values()):
        raise EntsoeResolutionRegimeError("execution counters must be exact zeros")
    authorities = _mapping(contract["authorities"], "authorities")
    _exact(
        authorities,
        {
            "real_source_evidence",
            "external_owner_identity",
            "point_in_time_availability",
            "value_quality",
            "model_input",
            "model_selection",
            "candidate",
            "promotion",
            "production",
        },
        "authorities",
    )
    if not authorities or any(value is not False for value in authorities.values()):
        raise EntsoeResolutionRegimeError("authorities must be exact and false")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": canonical_sha256(contract),
        "databricks_statement_count": 0,
        "warehouse_start_count": 0,
        "network_call_count": 0,
        "h_drive_access_count": 0,
        "model_input_authorized": False,
        "production_authorized": False,
    }


def validate_resolution_regime_contract_path(path: str | Path) -> dict[str, object]:
    """Read and bind the exact repo-local D260 JSON contract."""

    contract_path = Path(path)
    if not contract_path.is_absolute():
        raise EntsoeResolutionRegimeError("resolution regime contract path must be absolute")
    try:
        payload = read_stable_single_link_file(
            contract_path,
            label="D260 ENTSO-E resolution regime contract",
            max_bytes=100_000,
        )
    except (OSError, ValueError) as exc:
        raise EntsoeResolutionRegimeError("resolution regime contract read failed") from exc
    if sha256(payload).hexdigest() != CONTRACT_FILE_SHA256:
        raise EntsoeResolutionRegimeError("resolution regime contract SHA-256 differs")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntsoeResolutionRegimeError("resolution regime contract JSON is invalid") from exc
    return validate_resolution_regime_contract(_mapping(document, "resolution regime contract"))


def profile_resolution_regimes(
    series_dimension: pd.DataFrame,
    target_timestamps: pd.DataFrame,
    resolution_regimes: pd.DataFrame,
    *,
    evidence_class: str,
    assessment_start_utc: pd.Timestamp,
    assessment_end_utc: pd.Timestamp,
    metadata_as_of_utc: pd.Timestamp,
) -> EntsoeResolutionRegimeProfile:
    """Profile a piecewise cadence schedule without values or empirical authority."""

    if evidence_class != EVIDENCE_CLASS:
        raise EntsoeResolutionRegimeError(
            "real regime evidence requires an independent artifact and owner verifier"
        )
    assessment_start = _utc_instant(assessment_start_utc, "assessment start")
    assessment_end = _utc_instant(assessment_end_utc, "assessment end")
    metadata_as_of = _utc_instant(metadata_as_of_utc, "metadata as-of")
    if assessment_start >= assessment_end:
        raise EntsoeResolutionRegimeError("assessment window is empty or reversed")
    if metadata_as_of < assessment_end:
        raise EntsoeResolutionRegimeError("metadata as-of precedes the assessment window end")
    dimension = _copy_exact_frame(series_dimension, DIMENSION_COLUMNS, "series dimension")
    targets = _copy_exact_frame(
        target_timestamps,
        TARGET_COLUMNS,
        "target timestamps",
        allow_empty=True,
    )
    regimes = _copy_exact_frame(resolution_regimes, REGIME_COLUMNS, "resolution regimes")
    series_ids = _validate_dimension(dimension)
    _validate_targets(
        targets,
        series_ids,
        assessment_start=assessment_start,
        assessment_end=assessment_end,
    )
    _validate_regime_structure(
        regimes,
        series_ids,
        metadata_as_of=metadata_as_of,
    )

    metadata = dimension.set_index("series_id").to_dict(orient="index")
    profiles: list[dict[str, object]] = []
    total_expected = 0
    total_observed = 0
    total_missing = 0
    total_transitions = 0
    for series_id in sorted(series_ids):
        observed = targets.loc[targets["series_id"].eq(series_id), "target_ts_utc"].sort_values(
            kind="mergesort"
        )
        schedule = regimes.loc[regimes["series_id"].eq(series_id)].sort_values(
            "valid_from_utc",
            kind="mergesort",
        )
        profile = _profile_one_series(
            series_id,
            str(metadata[series_id]["group_name"]),
            str(metadata[series_id]["native_resolution"]),
            observed,
            schedule,
            assessment_start_ns=int(assessment_start.value),
            assessment_end_ns=int(assessment_end.value),
            metadata_as_of_ns=int(metadata_as_of.value),
            remaining_slot_budget=MAX_EXPECTED_SLOTS - total_expected,
        )
        profiles.append(profile)
        total_expected += int(profile["expected_target_count"])
        total_observed += int(profile["observed_target_count"])
        total_missing += int(profile["missing_native_slot_count"])
        total_transitions += int(profile["transition_count"])

    all_complete = total_missing == 0
    authorities = {
        "real_source_evidence": False,
        "external_owner_identity": False,
        "point_in_time_availability": False,
        "value_quality": False,
        "model_input": False,
        "model_selection": False,
        "candidate": False,
        "promotion": False,
        "production": False,
    }
    summary: dict[str, object] = {
        "schema_version": PROFILE_SCHEMA,
        "status": PASS_STATUS if all_complete else GAP_STATUS,
        "evidence_class": evidence_class,
        "timestamp_semantics": "DELIVERY_INTERVAL_START_UTC",
        "interval_semantics": "LEFT_CLOSED_RIGHT_OPEN",
        "assessment_start_utc": assessment_start.isoformat().replace("+00:00", "Z"),
        "assessment_end_utc": assessment_end.isoformat().replace("+00:00", "Z"),
        "metadata_as_of_utc": metadata_as_of.isoformat().replace("+00:00", "Z"),
        "series_count": len(series_ids),
        "regime_count": len(regimes),
        "transition_count": total_transitions,
        "expected_target_count": total_expected,
        "observed_target_count": total_observed,
        "missing_native_slot_count": total_missing,
        "all_series_piecewise_grid_complete": all_complete,
        "implicit_upsampling_performed": False,
        "implicit_downsampling_performed": False,
        "implicit_forward_fill_performed": False,
        "raw_series_ids_persisted_in_profile": False,
        "opened_value_column_count": 0,
        "databricks_connection_count": 0,
        "databricks_statement_count": 0,
        "warehouse_start_count": 0,
        "network_call_count": 0,
        "h_drive_access_count": 0,
        "authorities": authorities,
    }
    return EntsoeResolutionRegimeProfile(
        summary=summary,
        series_profile=tuple(sorted(profiles, key=lambda row: str(row["series_id_sha256"]))),
    )


def _profile_one_series(
    series_id: str,
    group_name: str,
    declared_current_resolution: str,
    observed: pd.Series,
    schedule: pd.DataFrame,
    *,
    assessment_start_ns: int,
    assessment_end_ns: int,
    metadata_as_of_ns: int,
    remaining_slot_budget: int,
) -> dict[str, object]:
    observed_ns = set(int(value) for value in observed.astype("int64"))
    schedule_rows = _schedule_rows(schedule)
    _active_regime(assessment_start_ns, schedule_rows, series_id)
    final_window_regime = _containing_regime(
        assessment_end_ns - 1,
        schedule_rows,
        series_id,
    )
    final_window_resolution_ns = RESOLUTION_SECONDS[final_window_regime[2]] * 1_000_000_000
    if (assessment_end_ns - final_window_regime[0]) % final_window_resolution_ns != 0:
        raise EntsoeResolutionRegimeError(
            f"assessment end does not close the active resolution grid: {series_id}"
        )
    metadata_regime = _containing_regime(metadata_as_of_ns, schedule_rows, series_id)
    if declared_current_resolution != metadata_regime[2]:
        raise EntsoeResolutionRegimeError(
            f"declared current resolution differs at metadata as-of: {series_id}"
        )
    expected_ns: set[int] = set()
    resolutions: list[str] = []
    for start_ns, end_ns, resolution in schedule_rows:
        overlap_start = max(assessment_start_ns, start_ns)
        effective_end = assessment_end_ns if end_ns is None else min(assessment_end_ns, end_ns)
        if effective_end <= overlap_start:
            continue
        resolution_ns = RESOLUTION_SECONDS[resolution] * 1_000_000_000
        offset = overlap_start - start_ns
        first_step = max(0, (offset + resolution_ns - 1) // resolution_ns)
        current = start_ns + first_step * resolution_ns
        while current < effective_end:
            if end_ns is not None and current + resolution_ns > end_ns:
                raise EntsoeResolutionRegimeError(
                    f"resolution regime ends inside a delivery interval: {series_id}"
                )
            if len(expected_ns) >= remaining_slot_budget:
                raise EntsoeResolutionRegimeError("expected target slot cap exceeded")
            expected_ns.add(current)
            current += resolution_ns
        resolutions.append(resolution)

    unexpected = sorted(observed_ns.difference(expected_ns))
    if unexpected:
        raise EntsoeResolutionRegimeError(
            f"observed target is outside the active native-resolution grid: {series_id}"
        )
    missing = expected_ns.difference(observed_ns)
    complete_days = _complete_utc_days(
        expected_ns,
        observed_ns,
        minimum_ns=assessment_start_ns,
        window_end_ns=assessment_end_ns,
    )
    expected_count = len(expected_ns)
    observed_count = len(observed_ns)
    return {
        "series_id_sha256": sha256(series_id.encode("utf-8")).hexdigest(),
        "group_name": group_name,
        "regime_count": len(resolutions),
        "transition_count": max(0, len(resolutions) - 1),
        "resolution_sequence": resolutions,
        "declared_current_resolution": declared_current_resolution,
        "observed_target_count": observed_count,
        "expected_target_count": expected_count,
        "missing_native_slot_count": len(missing),
        "piecewise_grid_completeness_ratio": float(observed_count / expected_count),
        "complete_utc_days": complete_days,
        "raw_series_id_persisted": False,
        "raw_values_opened_or_persisted": False,
    }


def _schedule_rows(schedule: pd.DataFrame) -> list[tuple[int, int | None, str]]:
    rows: list[tuple[int, int | None, str]] = []
    for row in schedule.itertuples(index=False):
        start_ns = int(row.valid_from_utc.value)
        end_ns = None if pd.isna(row.valid_to_utc) else int(row.valid_to_utc.value)
        rows.append((start_ns, end_ns, str(row.native_resolution)))
    return rows


def _active_regime(
    target_ns: int,
    schedule: list[tuple[int, int | None, str]],
    series_id: str,
) -> tuple[int, int | None, str]:
    match = _containing_regime(target_ns, schedule, series_id)
    start_ns, _, resolution = match
    resolution_ns = RESOLUTION_SECONDS[resolution] * 1_000_000_000
    if (target_ns - start_ns) % resolution_ns != 0:
        raise EntsoeResolutionRegimeError(
            f"target timestamp is off its active resolution grid: {series_id}"
        )
    return match


def _containing_regime(
    target_ns: int,
    schedule: list[tuple[int, int | None, str]],
    series_id: str,
) -> tuple[int, int | None, str]:
    matches = [
        row for row in schedule if row[0] <= target_ns and (row[1] is None or target_ns < row[1])
    ]
    if len(matches) != 1:
        raise EntsoeResolutionRegimeError(
            f"target timestamp has no unique active resolution regime: {series_id}"
        )
    return matches[0]


def _validate_dimension(dimension: pd.DataFrame) -> frozenset[str]:
    _nonempty_strings(dimension, "series_id", "series dimension")
    _nonempty_strings(dimension, "group_name", "series dimension")
    _nonempty_strings(dimension, "native_resolution", "series dimension")
    normalized = dimension["series_id"].str.strip().str.casefold()
    if bool(normalized.duplicated(keep=False).any()):
        raise EntsoeResolutionRegimeError("series identifiers are duplicated")
    if not set(dimension["native_resolution"]).issubset(RESOLUTION_SECONDS):
        raise EntsoeResolutionRegimeError("declared current resolution is invalid")
    return frozenset(str(value) for value in dimension["series_id"])


def _validate_targets(
    targets: pd.DataFrame,
    series_ids: frozenset[str],
    *,
    assessment_start: pd.Timestamp,
    assessment_end: pd.Timestamp,
) -> None:
    _nonempty_strings(targets, "series_id", "target timestamps")
    _utc_column(targets, "target_ts_utc", "target timestamps", allow_null=False)
    if bool(targets.duplicated(list(TARGET_COLUMNS), keep=False).any()):
        raise EntsoeResolutionRegimeError("target timestamp key is duplicated")
    target_ids = frozenset(str(value) for value in targets["series_id"])
    if not target_ids.issubset(series_ids):
        raise EntsoeResolutionRegimeError("target timestamps contain an unknown series")
    if bool(
        (
            targets["target_ts_utc"].lt(assessment_start)
            | targets["target_ts_utc"].ge(assessment_end)
        ).any()
    ):
        raise EntsoeResolutionRegimeError(
            "target timestamp lies outside the explicit assessment window"
        )


def _validate_regime_structure(
    regimes: pd.DataFrame,
    series_ids: frozenset[str],
    *,
    metadata_as_of: pd.Timestamp,
) -> None:
    for column in (
        "series_id",
        "native_resolution",
        "source_document_id",
        "source_locator",
    ):
        _nonempty_strings(regimes, column, "resolution regimes")
    _utc_column(regimes, "valid_from_utc", "resolution regimes", allow_null=False)
    _utc_column(regimes, "valid_to_utc", "resolution regimes", allow_null=True)
    _utc_column(regimes, "owner_confirmed_at_utc", "resolution regimes", allow_null=False)
    if bool(regimes["owner_confirmed_at_utc"].gt(metadata_as_of).any()):
        raise EntsoeResolutionRegimeError("owner confirmation is later than metadata as-of")
    if bool(regimes.duplicated(["series_id", "valid_from_utc"], keep=False).any()):
        raise EntsoeResolutionRegimeError("resolution regime key is duplicated")
    regime_ids = frozenset(str(value) for value in regimes["series_id"])
    if regime_ids != series_ids:
        raise EntsoeResolutionRegimeError("resolution regime series coverage differs")
    if not set(regimes["native_resolution"]).issubset(RESOLUTION_SECONDS):
        raise EntsoeResolutionRegimeError("resolution regime value is invalid")
    for value in regimes["source_document_id"]:
        if _IDENTIFIER.fullmatch(str(value)) is None:
            raise EntsoeResolutionRegimeError("source document ID is invalid")
    for value in regimes["source_locator"]:
        _https(value, "source locator")

    for series_id, raw_schedule in regimes.groupby("series_id", sort=False):
        schedule = raw_schedule.sort_values("valid_from_utc", kind="mergesort")
        previous_end: pd.Timestamp | None = None
        previous_start: pd.Timestamp | None = None
        previous_resolution: str | None = None
        for position, row in enumerate(schedule.itertuples(index=False), start=1):
            start = row.valid_from_utc
            end = row.valid_to_utc
            if not pd.isna(end) and end <= start:
                raise EntsoeResolutionRegimeError(
                    f"resolution regime interval is empty or reversed: {series_id}"
                )
            if position > 1:
                if previous_end is None or pd.isna(previous_end):
                    raise EntsoeResolutionRegimeError(
                        f"open-ended resolution regime is not final: {series_id}"
                    )
                if start != previous_end:
                    raise EntsoeResolutionRegimeError(
                        f"resolution regimes have a gap or overlap: {series_id}"
                    )
                assert previous_start is not None
                assert previous_resolution is not None
                previous_ns = RESOLUTION_SECONDS[previous_resolution] * 1_000_000_000
                if (int(previous_end.value) - int(previous_start.value)) % previous_ns != 0:
                    raise EntsoeResolutionRegimeError(
                        f"transition boundary does not close the previous grid: {series_id}"
                    )
            previous_start = start
            previous_end = None if pd.isna(end) else end
            previous_resolution = str(row.native_resolution)


def _complete_utc_days(
    expected_ns: set[int],
    observed_ns: set[int],
    *,
    minimum_ns: int,
    window_end_ns: int,
) -> int:
    day_ns = 86_400 * 1_000_000_000
    first_full_day = -(-minimum_ns // day_ns) * day_ns
    last_full_day_end = (window_end_ns // day_ns) * day_ns
    expected_by_day: dict[int, set[int]] = {}
    observed_by_day: dict[int, set[int]] = {}
    for value in expected_ns:
        expected_by_day.setdefault(value // day_ns, set()).add(value)
    for value in observed_ns:
        observed_by_day.setdefault(value // day_ns, set()).add(value)
    complete = 0
    day_start = first_full_day
    while day_start < last_full_day_end:
        day_key = day_start // day_ns
        expected_day = expected_by_day.get(day_key, set())
        observed_day = observed_by_day.get(day_key, set())
        if expected_day and observed_day == expected_day:
            complete += 1
        day_start += day_ns
    return complete


def _copy_exact_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    label: str,
    *,
    allow_empty: bool = False,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise EntsoeResolutionRegimeError(f"{label} must be a pandas DataFrame")
    if list(frame.columns) != list(columns):
        raise EntsoeResolutionRegimeError(f"{label} columns differ")
    if frame.empty and not allow_empty:
        raise EntsoeResolutionRegimeError(f"{label} must be non-empty")
    return frame.copy(deep=True)


def _utc_instant(value: object, label: str) -> pd.Timestamp:
    if (
        not isinstance(value, pd.Timestamp)
        or pd.isna(value)
        or value.tz is None
        or str(value.tz) != "UTC"
    ):
        raise EntsoeResolutionRegimeError(f"{label} must be a timezone-aware UTC Timestamp")
    return value


def _utc_column(
    frame: pd.DataFrame,
    column: str,
    label: str,
    *,
    allow_null: bool,
) -> None:
    series = frame[column]
    if not isinstance(series.dtype, pd.DatetimeTZDtype) or str(series.dt.tz) != "UTC":
        raise EntsoeResolutionRegimeError(f"{label} {column} must be timezone-aware UTC")
    if not allow_null and bool(series.isna().any()):
        raise EntsoeResolutionRegimeError(f"{label} {column} contains nulls")


def _nonempty_strings(frame: pd.DataFrame, column: str, label: str) -> None:
    series = frame[column]
    if not pd.api.types.is_string_dtype(series.dtype) or not bool(
        series.map(lambda value: isinstance(value, str)).all()
    ):
        raise EntsoeResolutionRegimeError(f"{label} {column} must contain strings")
    if bool(series.isna().any()):
        raise EntsoeResolutionRegimeError(f"{label} {column} contains nulls")
    if bool(series.map(lambda value: value != value.strip() or not value).any()):
        raise EntsoeResolutionRegimeError(f"{label} {column} is not canonical")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EntsoeResolutionRegimeError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise EntsoeResolutionRegimeError(f"{label} fields differ")


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise EntsoeResolutionRegimeError(f"{label} differs")


def _clean_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise EntsoeResolutionRegimeError(f"{label} is invalid")
    return value


def _https(value: object, label: str) -> str:
    text = _clean_text(value, label)
    parsed = urlsplit(text)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(char.isspace() for char in text)
    ):
        raise EntsoeResolutionRegimeError(f"{label} must be an HTTPS locator")
    return text


__all__ = [
    "CONTRACT_FILE_SHA256",
    "CONTRACT_CONTENT_ID",
    "CONTRACT_SCHEMA",
    "CONTRACT_STATUS",
    "DIMENSION_COLUMNS",
    "EVIDENCE_CLASS",
    "EntsoeResolutionRegimeError",
    "EntsoeResolutionRegimeProfile",
    "GAP_STATUS",
    "PASS_STATUS",
    "PROFILE_SCHEMA",
    "REGIME_COLUMNS",
    "RESOLUTION_SECONDS",
    "TARGET_COLUMNS",
    "canonical_sha256",
    "profile_resolution_regimes",
    "validate_resolution_regime_contract",
    "validate_resolution_regime_contract_path",
]
