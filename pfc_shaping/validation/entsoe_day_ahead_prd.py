"""Fail-closed admission for PRD ENTSO-E day-ahead price extracts.

The module is deliberately narrow: seven observed source series across five
coupled markets, one value-blind profile query and one explicitly selected
five-series point-in-time monthly extract.  It never connects to Databricks
and never grants model or production authority.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import pandas as pd

from pfc_shaping.data.governed_lt_acquisition import dataframe_semantic_sha256
from pfc_shaping.path_safety import read_stable_single_link_file

ROOT = Path(__file__).resolve().parents[2]
PROFILE_SQL_PATH = ROOT / "docs/data/sql/databricks_prd_entsoe_day_ahead_profile.sql"
PIT_SQL_PATH = ROOT / "docs/data/sql/databricks_prd_entsoe_day_ahead_pit_extract.sql"
PROFILE_SQL_SHA256 = "e48bc8b09d6f3676616ed42966d50a3f44d9eaf3649f9ce9c9543a0bc024259e"
PIT_SQL_SHA256 = "7b444430db6b62a7bcd9f0bc6e5f05858a56b9756be68f70016c107a4b271ebf"

REPORT_SCHEMA = "fmv_entsoe_day_ahead_prd_profile.v1"
PIT_AUDIT_SCHEMA = "fmv_entsoe_day_ahead_prd_pit_extract.v2"
PASS_STATUS = "PASS_ENTSOE_DAY_AHEAD_PRD_PROFILE"
BLOCKED_STATUS = "BLOCKED_ENTSOE_DAY_AHEAD_PRD_PROFILE"
REQUIRED_CHANGE_COMMIT = "db3a93316cd431a95b4e096d8482e482fda3491e"

EXPECTED_FIELDS = (
    "ch_price",
    "at_price",
    "de_lu_price",
    "fr_price",
    "it_nord_price",
)
SERIES_INVENTORY_SPEC = (
    ("ch_price", "ch_price", None),
    ("at_price_seq1", "at_price", "1"),
    ("at_price_seq2", "at_price", "2"),
    ("de_lu_price_seq1", "de_lu_price", "1"),
    ("de_lu_price_seq2", "de_lu_price", "2"),
    ("fr_price", "fr_price", None),
    ("it_nord_price", "it_nord_price", None),
)
EXPECTED_SERIES_SLOTS = tuple(slot for slot, _, _ in SERIES_INVENTORY_SPEC)
EXPECTED_SERIES_IDENTITIES = frozenset(
    (field, classification) for _, field, classification in SERIES_INVENTORY_SPEC
)
REBUILD_GROUPS = frozenset(
    {
        "day_ahead_prices",
        "production_unit_unavailability",
        "generation_unit_unavailability",
        "transmission_unavailability",
        "installed_capacity_per_unit",
        "generation_forecast",
    }
)
PROFILE_RESULT_ROW_LIMIT = 101
PIT_RESULT_ROW_LIMIT = 20_001
MAX_MONTHLY_WINDOW_DAYS = 31

PROFILE_COLUMNS = (
    "field_name",
    "series_key",
    "classification_sequence",
    "unit",
    "document_type",
    "series_id",
    "resolution",
    "vintage_row_count",
    "distinct_interval_count",
    "min_interval_start_utc",
    "max_interval_end_utc",
    "null_value_count",
    "dq_failed_count",
    "unknown_availability_count",
    "invalid_availability_order_count",
    "invalid_interval_count",
    "canonical_series_key_mismatch_count",
    "profile_row_count",
    "duplicate_vintage_key_count",
    "orphan_series_key_count",
    "gold_series_key_duplicate_count",
    "latest_grain_duplicate_count",
    "legacy_new_overlap_interval_count",
)
PIT_COLUMNS = (
    "field_name",
    "series_key",
    "interval_start_utc",
    "interval_end_utc",
    "resolution",
    "price_eur_per_mwh",
    "availability_timestamp_utc",
    "source_document_mrid",
    "source_document_revision_number",
)
GLOBAL_COUNT_COLUMNS = (
    "profile_row_count",
    "duplicate_vintage_key_count",
    "orphan_series_key_count",
    "gold_series_key_duplicate_count",
    "latest_grain_duplicate_count",
    "legacy_new_overlap_interval_count",
)
ROW_FAILURE_COLUMNS = {
    "null_value_count": "DAY_AHEAD_VALUE_NULL",
    "dq_failed_count": "DAY_AHEAD_DQ_FAILED",
    "unknown_availability_count": "DAY_AHEAD_AVAILABILITY_UNKNOWN",
    "invalid_availability_order_count": "DAY_AHEAD_AVAILABILITY_ORDER_INVALID",
    "invalid_interval_count": "DAY_AHEAD_INTERVAL_INVALID",
    "canonical_series_key_mismatch_count": "DAY_AHEAD_SERIES_KEY_NONCANONICAL",
}
GLOBAL_FAILURE_COLUMNS = {
    "duplicate_vintage_key_count": "DAY_AHEAD_VINTAGE_KEY_DUPLICATED",
    "orphan_series_key_count": "DAY_AHEAD_SERIES_KEY_ORPHANED",
    "gold_series_key_duplicate_count": "DAY_AHEAD_GOLD_SERIES_KEY_DUPLICATED",
    "latest_grain_duplicate_count": "DAY_AHEAD_LATEST_GRAIN_DUPLICATED",
    "legacy_new_overlap_interval_count": "DAY_AHEAD_LEGACY_NEW_KEY_OVERLAP",
}
REBUILD_EVIDENCE_FIELDS = frozenset(
    {
        "environment",
        "run_id",
        "manifest_sha256",
        "deployed_commit",
        "required_change_commit",
        "required_change_ancestor_proven",
        "mode",
        "groups_rebuilt",
        "run_status",
        "post_backfill_validation_status",
        "old_new_coexistence_count",
        "completed_at_utc",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_CLEAN_TEXT = re.compile(r"^[^\x00-\x1f\x7f]+$")
_RESOLUTION_SECONDS = {"PT15M": 900, "PT30M": 1_800, "PT60M": 3_600, "PT1H": 3_600}


class EntsoeDayAheadPrdError(ValueError):
    """Raised when an input is structurally unsafe or not hash-bound."""


@dataclass(frozen=True)
class DayAheadFinding:
    """One actionable source-quality failure."""

    code: str
    message: str
    affected_count: int | None = None
    severity: str = "CRITICAL"

    def as_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "affected_count": self.affected_count,
        }


@dataclass(frozen=True)
class DayAheadProfileReport:
    """Value-blind assessment; a pass only authorizes the bounded PIT query."""

    status: str
    findings: tuple[DayAheadFinding, ...]
    metrics: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        passed = self.status == PASS_STATUS
        return {
            "schema_version": REPORT_SCHEMA,
            "status": self.status,
            "findings": [finding.as_dict() for finding in self.findings],
            "metrics": dict(self.metrics),
            "layer_policy": {
                "current_series_authority": "PRD_GOLD_DIMENTSOESERIES",
                "point_in_time_value_authority": (
                    "PRD_SILVER_GE_POWER_ENTSOE_TIME_SERIES_VINTAGES"
                ),
                "multi_auction_inventory": "AT_AND_DE_LU_CLASSIFICATION_SEQUENCE_1_AND_2",
                "pit_selection_policy": "EXPLICIT_ONE_SERIES_PER_FIELD_NO_DEFAULT_CHOICE",
                "lseg_epex_actuals_role": "REALIZED_SPOT_CROSSCHECK_CH_AT_DE_LU_FR",
                "euler_spot_role": "INDEPENDENT_CROSSCHECK_ONLY",
            },
            "authorities": {
                "bounded_pit_extraction_authorized": passed,
                "cadence_completeness_authorized": False,
                "model_input_authorized": False,
                "model_selection_authorized": False,
                "production_authorized": False,
            },
        }


@dataclass(frozen=True)
class DayAheadPitExtract:
    """Validated local PIT rows and a non-authoritative audit payload."""

    frame: pd.DataFrame
    audit: Mapping[str, object]


def verify_sql_bindings() -> dict[str, str]:
    """Verify the two immutable SQL templates used by this gate."""

    observed = {
        "profile_sql_sha256": _file_sha256(PROFILE_SQL_PATH, "profile SQL"),
        "pit_sql_sha256": _file_sha256(PIT_SQL_PATH, "PIT SQL"),
    }
    expected = {
        "profile_sql_sha256": PROFILE_SQL_SHA256,
        "pit_sql_sha256": PIT_SQL_SHA256,
    }
    if observed != expected:
        raise EntsoeDayAheadPrdError("ENTSO-E day-ahead SQL binding differs")
    return observed


def assess_day_ahead_prd_profile(
    profile: pd.DataFrame,
    *,
    window_start_utc: str | pd.Timestamp,
    window_end_utc: str | pd.Timestamp,
    assessed_at_utc: str | pd.Timestamp,
    profile_query_sha256: str,
    series_inventory_binding: Mapping[str, str] | None = None,
    rebuild_evidence: Mapping[str, object] | None = None,
) -> DayAheadProfileReport:
    """Assess one value-blind PRD profile and fail closed on missing evidence."""

    verify_sql_bindings()
    _expected_sha(profile_query_sha256, PROFILE_SQL_SHA256, "profile query")
    frame = _exact_frame(profile, PROFILE_COLUMNS, "day-ahead profile")
    if frame.empty:
        raise EntsoeDayAheadPrdError("day-ahead profile is empty")

    parameters = _monthly_window_parameters(
        window_start_utc=window_start_utc,
        window_end_utc=window_end_utc,
        label="profile",
    )
    start = _utc_timestamp(parameters["start_utc"], "profile window start")
    end = _utc_timestamp(parameters["end_utc"], "profile window end")
    assessed_at = _utc_timestamp(assessed_at_utc, "profile assessment time")
    if assessed_at < end:
        raise EntsoeDayAheadPrdError("profile assessment time precedes the window end")

    findings: list[DayAheadFinding] = []
    global_counts = _global_counts(frame)
    if global_counts["profile_row_count"] != len(frame):
        findings.append(
            DayAheadFinding(
                "DAY_AHEAD_PROFILE_ROW_COUNT_MISMATCH",
                "The query-reported profile row count differs from the received rows.",
                abs(global_counts["profile_row_count"] - len(frame)),
            )
        )
    if len(frame) >= PROFILE_RESULT_ROW_LIMIT or global_counts["profile_row_count"] >= (
        PROFILE_RESULT_ROW_LIMIT
    ):
        findings.append(
            DayAheadFinding(
                "DAY_AHEAD_PROFILE_LIMIT_HIT",
                "The 101-row rejection sentinel was reached.",
                len(frame),
            )
        )
    for column, code in GLOBAL_FAILURE_COLUMNS.items():
        count = global_counts[column]
        if count:
            findings.append(
                DayAheadFinding(code, f"Blocking aggregate is non-zero: {column}.", count)
            )

    normalized = _normalize_profile(frame, findings)
    _assess_profile_inventory(normalized, findings)
    binding = _assess_inventory_binding(series_inventory_binding, normalized, findings)
    rebuild = _assess_rebuild_evidence(rebuild_evidence, assessed_at, findings)

    status = BLOCKED_STATUS if findings else PASS_STATUS
    metrics = {
        "profile_query_sha256": PROFILE_SQL_SHA256,
        "profile_rows": len(normalized),
        "series_count": int(normalized["series_key"].nunique()),
        "field_count": int(normalized["field_name"].nunique()),
        "window_start_utc": _utc_text(start),
        "window_end_utc": _utc_text(end),
        "assessed_at_utc": _utc_text(assessed_at),
        "bound_series_inventory": dict(binding) if binding is not None else None,
        "series_profiles": _series_profile_metrics(normalized),
        "rebuild_run_id": rebuild.get("run_id") if rebuild is not None else None,
        "severity_counts": {
            severity: sum(item.severity == severity for item in findings)
            for severity in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
        },
        **global_counts,
    }
    return DayAheadProfileReport(status, tuple(findings), metrics)


def build_day_ahead_profile_parameters(
    *,
    window_start_utc: str | pd.Timestamp,
    window_end_utc: str | pd.Timestamp,
) -> dict[str, object]:
    """Build typed parameters for one value-blind, partition-pruned profile."""

    verify_sql_bindings()
    return _monthly_window_parameters(
        window_start_utc=window_start_utc,
        window_end_utc=window_end_utc,
        label="profile",
    )


def build_day_ahead_pit_parameters(
    *,
    series_selection: Mapping[str, str],
    window_start_utc: str | pd.Timestamp,
    window_end_utc: str | pd.Timestamp,
    as_of_utc: str | pd.Timestamp,
) -> dict[str, object]:
    """Build typed parameters for one partition-pruned monthly PIT query."""

    verify_sql_bindings()
    selection = _series_selection(series_selection)
    start = _utc_timestamp(window_start_utc, "PIT window start")
    end = _utc_timestamp(window_end_utc, "PIT window end")
    as_of = _utc_timestamp(as_of_utc, "PIT as-of")
    window = _monthly_window_parameters(
        window_start_utc=start,
        window_end_utc=end,
        label="PIT",
    )
    return {
        **window,
        "as_of_utc": _utc_text(as_of),
        **{f"{field.removesuffix('_price')}_series_key": key for field, key in selection.items()},
    }


def validate_day_ahead_pit_extract(
    frame: pd.DataFrame,
    *,
    series_selection: Mapping[str, str],
    window_start_utc: str | pd.Timestamp,
    window_end_utc: str | pd.Timestamp,
    as_of_utc: str | pd.Timestamp,
    pit_query_sha256: str,
) -> DayAheadPitExtract:
    """Validate normalized Silver blocks without asserting cadence completeness."""

    _expected_sha(pit_query_sha256, PIT_SQL_SHA256, "PIT query")
    parameters = build_day_ahead_pit_parameters(
        series_selection=series_selection,
        window_start_utc=window_start_utc,
        window_end_utc=window_end_utc,
        as_of_utc=as_of_utc,
    )
    source = _exact_frame(frame, PIT_COLUMNS, "day-ahead PIT extract")
    if source.empty:
        raise EntsoeDayAheadPrdError("day-ahead PIT extract is empty")
    if len(source) >= PIT_RESULT_ROW_LIMIT:
        raise EntsoeDayAheadPrdError("day-ahead PIT extract hit the rejection sentinel")

    selection = _series_selection(series_selection)
    result = source.copy()
    for column in ("field_name", "series_key", "resolution", "source_document_mrid"):
        result[column] = result[column].map(lambda value: _text(value, column))
    for column in ("interval_start_utc", "interval_end_utc", "availability_timestamp_utc"):
        parsed = pd.to_datetime(result[column], errors="coerce", utc=True)
        if parsed.isna().any():
            raise EntsoeDayAheadPrdError(f"PIT extract contains invalid {column}")
        result[column] = parsed

    expected_keys = result["field_name"].map(selection)
    if expected_keys.isna().any() or result["series_key"].ne(expected_keys).any():
        raise EntsoeDayAheadPrdError("PIT extract differs from the explicit five-series selection")
    if set(result["field_name"]) != set(EXPECTED_FIELDS):
        raise EntsoeDayAheadPrdError("PIT extract does not contain all five fields")

    start = _utc_timestamp(parameters["start_utc"], "PIT start")
    end = _utc_timestamp(parameters["end_utc"], "PIT end")
    as_of = _utc_timestamp(parameters["as_of_utc"], "PIT as-of")
    if result["interval_start_utc"].lt(start).any() or result["interval_start_utc"].ge(end).any():
        raise EntsoeDayAheadPrdError("PIT extract contains rows outside the delivery window")
    if result["interval_end_utc"].gt(end).any():
        raise EntsoeDayAheadPrdError("PIT extract contains rows outside the delivery window")
    if result["availability_timestamp_utc"].gt(as_of).any():
        raise EntsoeDayAheadPrdError("PIT extract leaks a value unavailable at the as-of")

    seconds = result["resolution"].map(_resolution_seconds)
    if seconds.isna().any():
        raise EntsoeDayAheadPrdError("PIT extract resolution is unsupported")
    resolution_ns = seconds.astype("int64") * 1_000_000_000
    start_ns = result["interval_start_utc"].map(lambda value: value.value)
    end_ns = result["interval_end_utc"].map(lambda value: value.value)
    duration_ns = end_ns - start_ns
    if duration_ns.le(0).any():
        raise EntsoeDayAheadPrdError("PIT extract interval is empty or inverted")
    if duration_ns.mod(resolution_ns).ne(0).any():
        raise EntsoeDayAheadPrdError(
            "PIT extract interval duration is not a native-resolution multiple"
        )
    if start_ns.mod(resolution_ns).ne(0).any() or end_ns.mod(resolution_ns).ne(0).any():
        raise EntsoeDayAheadPrdError("PIT extract interval is off its native resolution grid")
    prices = pd.to_numeric(result["price_eur_per_mwh"], errors="coerce")
    if prices.isna().any() or not all(math.isfinite(float(value)) for value in prices):
        raise EntsoeDayAheadPrdError("PIT extract contains a non-finite price")
    result["price_eur_per_mwh"] = prices.astype(float)
    if result.duplicated(
        ["field_name", "series_key", "interval_start_utc", "interval_end_utc"],
        keep=False,
    ).any():
        raise EntsoeDayAheadPrdError("PIT extract grain is duplicated")

    result = result.sort_values(
        ["field_name", "interval_start_utc", "interval_end_utc"], kind="mergesort"
    ).reset_index(drop=True)
    previous_end = result.groupby(["field_name", "series_key"], sort=False)[
        "interval_end_utc"
    ].shift()
    if result["interval_start_utc"].lt(previous_end).any():
        raise EntsoeDayAheadPrdError("PIT extract intervals overlap")
    audit = {
        "schema_version": PIT_AUDIT_SCHEMA,
        "status": "PASS_BOUNDED_PIT_EXTRACT_NOT_MODEL_INPUT_AUTHORITY",
        "pit_query_sha256": PIT_SQL_SHA256,
        "row_count": len(result),
        "frame_semantic_sha256": dataframe_semantic_sha256(result),
        "window_start_utc": parameters["start_utc"],
        "window_end_utc": parameters["end_utc"],
        "as_of_utc": parameters["as_of_utc"],
        "series_selection": dict(selection),
        "interval_policy": {
            "source_contract": "producer_normalized_half_open_utc_interval",
            "duration": "positive_integer_multiple_of_native_resolution",
            "coverage": "not_asserted_by_bounded_pit_validator",
        },
        "authorities": {
            "point_in_time_filter_validated": True,
            "cadence_completeness_authorized": False,
            "model_input_authorized": False,
            "model_selection_authorized": False,
            "production_authorized": False,
        },
    }
    return DayAheadPitExtract(result, audit)


def _normalize_profile(frame: pd.DataFrame, findings: list[DayAheadFinding]) -> pd.DataFrame:
    result = frame.copy()
    for column in ("field_name", "series_key", "unit", "document_type"):
        result[column] = result[column].map(lambda value: _text(value, column))
    result["classification_sequence"] = result["classification_sequence"].map(
        lambda value: _optional_text(value, "classification_sequence")
    )
    result["resolution"] = result["resolution"].map(
        lambda value: _optional_text(value, "resolution")
    )
    for column in ("series_id", "vintage_row_count", "distinct_interval_count"):
        result[column] = result[column].map(lambda value: _nonnegative_int(value, column))
    for column in ROW_FAILURE_COLUMNS:
        result[column] = result[column].map(lambda value: _nonnegative_int(value, column))

    duplicate_profiles = int(
        result.duplicated(["field_name", "series_key", "resolution"], keep=False).sum()
    )
    if duplicate_profiles:
        findings.append(
            DayAheadFinding(
                "DAY_AHEAD_PROFILE_GRAIN_DUPLICATED",
                "Profile rows repeat field, SeriesKey and resolution.",
                duplicate_profiles,
            )
        )
    return result


def _assess_profile_inventory(frame: pd.DataFrame, findings: list[DayAheadFinding]) -> None:
    observed_fields = set(frame["field_name"])
    missing = sorted(set(EXPECTED_FIELDS) - observed_fields)
    unexpected = sorted(observed_fields - set(EXPECTED_FIELDS))
    if missing:
        findings.append(
            DayAheadFinding(
                "DAY_AHEAD_FIELD_MISSING", f"Required fields are missing: {missing}.", len(missing)
            )
        )
    if unexpected:
        findings.append(
            DayAheadFinding(
                "DAY_AHEAD_FIELD_UNEXPECTED",
                f"Unexpected fields are present: {unexpected}.",
                len(unexpected),
            )
        )
    observed_identities = set(
        zip(frame["field_name"], frame["classification_sequence"], strict=True)
    )
    missing_identities = sorted(
        EXPECTED_SERIES_IDENTITIES - observed_identities,
        key=lambda item: (item[0], item[1] or ""),
    )
    unexpected_identities = sorted(
        observed_identities - EXPECTED_SERIES_IDENTITIES,
        key=lambda item: (item[0], item[1] or ""),
    )
    if missing_identities:
        findings.append(
            DayAheadFinding(
                "DAY_AHEAD_SERIES_IDENTITY_MISSING",
                f"Required field/classification identities are missing: {missing_identities}.",
                len(missing_identities),
            )
        )
    if unexpected_identities:
        findings.append(
            DayAheadFinding(
                "DAY_AHEAD_SERIES_IDENTITY_UNEXPECTED",
                f"Unexpected field/classification identities are present: {unexpected_identities}.",
                len(unexpected_identities),
            )
        )
    if frame.groupby("series_key")["field_name"].nunique().gt(1).any():
        findings.append(
            DayAheadFinding(
                "DAY_AHEAD_SERIES_KEY_REUSED",
                "One SeriesKey is attached to more than one price field.",
            )
        )

    stable_columns = ("classification_sequence", "unit", "document_type", "series_id")
    unstable = 0
    for _, rows in frame.groupby("series_key", sort=False):
        unstable += int(any(rows[column].nunique(dropna=False) != 1 for column in stable_columns))
    if unstable:
        findings.append(
            DayAheadFinding(
                "DAY_AHEAD_SERIES_METADATA_UNSTABLE",
                "Series metadata changes across resolution rows.",
                unstable,
            )
        )

    for row in frame.itertuples(index=False):
        expected_key = _canonical_series_key(row.field_name, row.classification_sequence)
        if row.series_key != expected_key:
            findings.append(
                DayAheadFinding(
                    "DAY_AHEAD_GOLD_SERIES_KEY_NONCANONICAL",
                    f"Gold SeriesKey is not canonical for {row.field_name}.",
                    1,
                )
            )
        if row.unit != "EUR/MWh":
            findings.append(
                DayAheadFinding("DAY_AHEAD_UNIT_INVALID", f"{row.field_name} is not EUR/MWh.", 1)
            )
        if row.document_type != "A44":
            findings.append(
                DayAheadFinding(
                    "DAY_AHEAD_DOCUMENT_TYPE_INVALID", f"{row.field_name} is not A44.", 1
                )
            )
        if row.vintage_row_count == 0 or row.distinct_interval_count == 0:
            findings.append(
                DayAheadFinding(
                    "DAY_AHEAD_SERIES_EMPTY", f"{row.series_key} has no vintage interval.", 1
                )
            )
        if row.distinct_interval_count > row.vintage_row_count:
            findings.append(
                DayAheadFinding(
                    "DAY_AHEAD_INTERVAL_COUNT_INVALID",
                    f"{row.series_key} has more intervals than vintage rows.",
                    row.distinct_interval_count,
                )
            )
        if row.resolution is not None and _resolution_seconds(row.resolution) is None:
            findings.append(
                DayAheadFinding(
                    "DAY_AHEAD_RESOLUTION_UNSUPPORTED",
                    f"{row.series_key} has unsupported resolution {row.resolution}.",
                    1,
                )
            )
        for column, code in ROW_FAILURE_COLUMNS.items():
            count = int(getattr(row, column))
            if count:
                findings.append(DayAheadFinding(code, f"{row.series_key} fails {column}.", count))


def _assess_inventory_binding(
    value: Mapping[str, str] | None,
    profile: pd.DataFrame,
    findings: list[DayAheadFinding],
) -> dict[str, str] | None:
    if value is None:
        findings.append(
            DayAheadFinding(
                "DAY_AHEAD_EXACT_SERIES_INVENTORY_BINDING_MISSING",
                "An explicit SeriesKey is required for each of the seven source slots.",
                len(EXPECTED_SERIES_SLOTS),
            )
        )
        return None
    binding = _series_inventory_binding(value)
    for slot, field, _ in SERIES_INVENTORY_SPEC:
        key = binding[slot]
        rows = profile.loc[profile["field_name"].eq(field) & profile["series_key"].eq(key)]
        if rows.empty:
            findings.append(
                DayAheadFinding(
                    "DAY_AHEAD_BOUND_SERIES_ABSENT",
                    f"Bound SeriesKey is absent from the profile: {field}.",
                    1,
                )
            )
            continue
    return binding


def _series_profile_metrics(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for slot, field, classification in SERIES_INVENTORY_SPEC:
        key = _canonical_series_key(field, classification)
        rows = frame.loc[frame["field_name"].eq(field) & frame["series_key"].eq(key)]
        observed_start = _timestamp_min(rows["min_interval_start_utc"])
        observed_end = _timestamp_max(rows["max_interval_end_utc"])
        result[slot] = {
            "field_name": field,
            "classification_sequence": classification,
            "series_key": key,
            "resolutions": sorted(rows["resolution"].dropna().unique().tolist()),
            "vintage_row_count": int(rows["vintage_row_count"].sum()),
            "distinct_interval_count_by_resolution_sum": int(rows["distinct_interval_count"].sum()),
            "min_interval_start_utc": (
                _utc_text(observed_start) if observed_start is not None else None
            ),
            "max_interval_end_utc": (_utc_text(observed_end) if observed_end is not None else None),
        }
    return result


def _assess_rebuild_evidence(
    value: Mapping[str, object] | None,
    assessed_at: pd.Timestamp,
    findings: list[DayAheadFinding],
) -> Mapping[str, object] | None:
    if value is None:
        findings.append(
            DayAheadFinding(
                "ENTSOE_PRD_REBUILD_MANIFEST_MISSING",
                "The independently supplied PRD full-rebuild manifest is missing.",
            )
        )
        return None
    if not isinstance(value, Mapping) or set(value) != REBUILD_EVIDENCE_FIELDS:
        raise EntsoeDayAheadPrdError("rebuild evidence fields are not exact")
    _text(value["run_id"], "rebuild run ID")
    _sha256(value["manifest_sha256"], "rebuild manifest SHA-256")
    _git_commit(value["deployed_commit"], "deployed commit")
    _expected_commit(value["required_change_commit"], REQUIRED_CHANGE_COMMIT, "required change")
    groups = value["groups_rebuilt"]
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
        raise EntsoeDayAheadPrdError("rebuilt groups must be a sequence")
    cleaned_groups = [_text(item, "rebuilt group") for item in groups]
    if len(cleaned_groups) != len(set(cleaned_groups)):
        raise EntsoeDayAheadPrdError("rebuilt groups repeat")
    coexistence = _nonnegative_int(value["old_new_coexistence_count"], "manifest coexistence count")
    completed = _utc_timestamp(value["completed_at_utc"], "rebuild completion")
    if completed > assessed_at:
        raise EntsoeDayAheadPrdError("rebuild completion is after the assessment time")

    claims = (
        (value["environment"] == "prd", "ENTSOE_REBUILD_ENVIRONMENT_INVALID"),
        (value["mode"] == "full", "ENTSOE_REBUILD_MODE_INVALID"),
        (value["run_status"] == "SUCCESS", "ENTSOE_REBUILD_RUN_FAILED"),
        (
            value["post_backfill_validation_status"] == "PASS",
            "ENTSOE_REBUILD_VALIDATION_FAILED",
        ),
        (
            type(value["required_change_ancestor_proven"]) is bool
            and value["required_change_ancestor_proven"],
            "ENTSOE_REQUIRED_CHANGE_ANCESTRY_UNPROVEN",
        ),
        (REBUILD_GROUPS.issubset(cleaned_groups), "ENTSOE_REBUILD_GROUPS_INCOMPLETE"),
        (coexistence == 0, "ENTSOE_REBUILD_MANIFEST_REPORTS_KEY_COEXISTENCE"),
    )
    for passed, code in claims:
        if not passed:
            findings.append(DayAheadFinding(code, "The rebuild manifest claim is not satisfied."))
    return value


def _series_inventory_binding(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(EXPECTED_SERIES_SLOTS):
        raise EntsoeDayAheadPrdError(
            "series inventory binding must contain exactly the seven source slots"
        )
    result: dict[str, str] = {}
    for slot, field, classification in SERIES_INVENTORY_SPEC:
        key = _text(value[slot], f"SeriesKey for {slot}")
        if key != _canonical_series_key(field, classification):
            raise EntsoeDayAheadPrdError(f"SeriesKey inventory binding is noncanonical: {slot}")
        result[slot] = key
    if len(set(result.values())) != len(result):
        raise EntsoeDayAheadPrdError("series inventory binding reuses a SeriesKey")
    return result


def _series_selection(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(EXPECTED_FIELDS):
        raise EntsoeDayAheadPrdError("series selection must contain exactly the five market fields")
    allowed = {
        field: {
            _canonical_series_key(candidate_field, classification)
            for _, candidate_field, classification in SERIES_INVENTORY_SPEC
            if candidate_field == field
        }
        for field in EXPECTED_FIELDS
    }
    result: dict[str, str] = {}
    for field in EXPECTED_FIELDS:
        key = _text(value[field], f"selected SeriesKey for {field}")
        if key not in allowed[field]:
            raise EntsoeDayAheadPrdError(
                f"selected SeriesKey is not an admitted candidate: {field}"
            )
        result[field] = key
    if len(set(result.values())) != len(result):
        raise EntsoeDayAheadPrdError("series selection reuses a SeriesKey")
    return result


def _global_counts(frame: pd.DataFrame) -> dict[str, int]:
    result: dict[str, int] = {}
    for column in GLOBAL_COUNT_COLUMNS:
        values = frame[column].map(lambda value: _nonnegative_int(value, column))
        if values.nunique(dropna=False) != 1:
            raise EntsoeDayAheadPrdError(f"global profile count is inconsistent: {column}")
        result[column] = int(values.iloc[0])
    return result


def _exact_frame(frame: object, columns: tuple[str, ...], label: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise EntsoeDayAheadPrdError(f"{label} must be a DataFrame")
    if frame.columns.has_duplicates or tuple(frame.columns) != columns:
        raise EntsoeDayAheadPrdError(f"{label} columns are not exact")
    return frame.copy()


def _canonical_series_key(field: str, classification: str | None) -> str:
    base = f"day_ahead_prices||{field}"
    return f"{base}||{classification}" if classification is not None else base


def _timestamp_min(values: pd.Series) -> pd.Timestamp | None:
    if values.empty:
        return None
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return None if parsed.isna().any() else parsed.min()


def _timestamp_max(values: pd.Series) -> pd.Timestamp | None:
    if values.empty:
        return None
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return None if parsed.isna().any() else parsed.max()


def _resolution_seconds(value: object) -> int | None:
    return _RESOLUTION_SECONDS.get(value) if isinstance(value, str) else None


def _file_sha256(path: Path, label: str) -> str:
    try:
        payload = read_stable_single_link_file(path, label=label, max_bytes=200_000)
    except (OSError, ValueError) as exc:
        raise EntsoeDayAheadPrdError(f"{label} read failed") from exc
    return hashlib.sha256(payload).hexdigest()


def _expected_sha(value: object, expected: str, label: str) -> str:
    observed = _sha256(value, label)
    if observed != expected:
        raise EntsoeDayAheadPrdError(f"{label} SHA-256 differs")
    return observed


def _expected_commit(value: object, expected: str, label: str) -> str:
    observed = _git_commit(value, label)
    if observed != expected:
        raise EntsoeDayAheadPrdError(f"{label} Git commit differs")
    return observed


def _git_commit(value: object, label: str) -> str:
    text = _text(value, label)
    if not _GIT_COMMIT.fullmatch(text):
        raise EntsoeDayAheadPrdError(f"{label} must be a lowercase Git SHA-1")
    return text


def _sha256(value: object, label: str) -> str:
    text = _text(value, label)
    if not _SHA256.fullmatch(text):
        raise EntsoeDayAheadPrdError(f"{label} must be lowercase SHA-256")
    return text


def _text(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or _CLEAN_TEXT.fullmatch(value) is None
    ):
        raise EntsoeDayAheadPrdError(f"{label} must be clean text")
    return value


def _optional_text(value: object, label: str) -> str | None:
    if value is None or pd.isna(value):
        return None
    return _text(value, label)


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise EntsoeDayAheadPrdError(f"{label} must be a nonnegative integer")
    return int(value)


def _utc_timestamp(value: object, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise EntsoeDayAheadPrdError(f"{label} is invalid") from exc
    if timestamp.tzinfo is None:
        raise EntsoeDayAheadPrdError(f"{label} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _ordered_window(start: pd.Timestamp, end: pd.Timestamp, label: str) -> None:
    if start >= end:
        raise EntsoeDayAheadPrdError(f"{label} window is empty or inverted")


def _monthly_window_parameters(
    *,
    window_start_utc: object,
    window_end_utc: object,
    label: str,
) -> dict[str, object]:
    start = _utc_timestamp(window_start_utc, f"{label} window start")
    end = _utc_timestamp(window_end_utc, f"{label} window end")
    _ordered_window(start, end, label)
    if end - start > pd.Timedelta(days=MAX_MONTHLY_WINDOW_DAYS):
        raise EntsoeDayAheadPrdError(f"{label} window exceeds 31 days")
    if any(value.second or value.microsecond or value.minute % 15 for value in (start, end)):
        raise EntsoeDayAheadPrdError(f"{label} bounds must lie on a 15-minute UTC grid")
    month_start = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    next_month = month_start + pd.offsets.MonthBegin(1)
    if start < month_start or end > next_month:
        raise EntsoeDayAheadPrdError(f"{label} window must stay inside one UTC calendar month")
    return {
        "start_utc": _utc_text(start),
        "end_utc": _utc_text(end),
        "delivery_year": start.year,
        "delivery_month": start.month,
    }


def _utc_text(value: pd.Timestamp) -> str:
    return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")


__all__ = [
    "BLOCKED_STATUS",
    "DayAheadFinding",
    "DayAheadPitExtract",
    "DayAheadProfileReport",
    "EXPECTED_FIELDS",
    "EXPECTED_SERIES_IDENTITIES",
    "EXPECTED_SERIES_SLOTS",
    "EntsoeDayAheadPrdError",
    "PASS_STATUS",
    "PIT_COLUMNS",
    "PIT_RESULT_ROW_LIMIT",
    "PIT_SQL_SHA256",
    "PROFILE_COLUMNS",
    "PROFILE_RESULT_ROW_LIMIT",
    "PROFILE_SQL_SHA256",
    "REQUIRED_CHANGE_COMMIT",
    "SERIES_INVENTORY_SPEC",
    "assess_day_ahead_prd_profile",
    "build_day_ahead_profile_parameters",
    "build_day_ahead_pit_parameters",
    "validate_day_ahead_pit_extract",
    "verify_sql_bindings",
]
