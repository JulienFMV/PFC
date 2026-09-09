"""Offline readiness audit for a hash-verified reusable ENTSO-E import.

The legacy reusable import is useful for schema and pipeline development, but
it is not point-in-time evidence.  This module profiles its timestamp/series
grain, coverage and cross-file consistency without granting calibration,
training, selection or production authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping

import pandas as pd


READINESS_SCHEMA_VERSION = "fmv_entsoe_local_readiness.v1"
EXPECTED_IMPORT_SCHEMA = "entso_reusable_import.v2"
EXPECTED_MODEL_ACTIVATION = "FORBIDDEN_UNTIL_GOVERNED_IMPORT"
EXPECTED_SOURCE_COHERENCE = "UNVERIFIED_MULTI_FILE_IMPORT"
EXPECTED_CADENCE = pd.Timedelta(minutes=15)
MIN_SEASONAL_HISTORY_DAYS = 730

REQUIRED_FILES = (
    "de_renewable_forecast.parquet",
    "entso_15min.parquet",
    "entso_border_15min.parquet",
    "entso_fundamentals_15min.parquet",
    "fr_nuclear_forecast_15min.parquet",
    "multi_country_forecast_15min.parquet",
    "outages_15min.parquet",
)
FORECAST_FILES = (
    "de_renewable_forecast.parquet",
    "fr_nuclear_forecast_15min.parquet",
    "multi_country_forecast_15min.parquet",
)
REQUIRED_POINT_IN_TIME_COLUMNS = frozenset({"as_of_utc"})
RAW_LINEAGE_COLUMNS = frozenset(
    {
        "series_id",
        "source_endpoint",
        "document_id",
        "revision_number",
        "unit",
        "native_resolution",
        "from_zone",
        "to_zone",
        "quality_flag",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CH_ACTUALS = frozenset(
    {
        "load_mw",
        "solar_mw",
        "wind_mw",
        "nuclear_mw",
        "hydro_ror_mw",
        "hydro_reservoir_mw",
        "hydro_pumped_mw",
    }
)


class EntsoeLocalReadinessError(ValueError):
    """Raised when an import cannot be profiled without ambiguity."""


@dataclass(frozen=True)
class EntsoeLocalReadinessAudit:
    """Aggregate audit, per-file profile and per-series profile."""

    summary: Mapping[str, object]
    file_profile: pd.DataFrame
    series_profile: pd.DataFrame
    projection_profile: pd.DataFrame


def audit_entsoe_local_readiness(
    manifest: Mapping[str, object],
    frames: Mapping[str, pd.DataFrame],
    *,
    manifest_sha256: str,
    archive_integrity_verified: bool,
) -> EntsoeLocalReadinessAudit:
    """Profile a verified legacy import while preserving its strict NO_GO."""

    manifest_hash = str(manifest_sha256).strip().lower()
    if _SHA256.fullmatch(manifest_hash) is None:
        raise EntsoeLocalReadinessError("ENTSO-E manifest SHA-256 is invalid")
    if archive_integrity_verified is not True:
        raise EntsoeLocalReadinessError(
            "ENTSO-E archive integrity must be verified before profiling"
        )
    _validate_manifest_authority(manifest)

    declared = manifest.get("files")
    if not isinstance(declared, Mapping) or not declared:
        raise EntsoeLocalReadinessError("ENTSO-E manifest file map is empty")
    if set(frames) != set(declared):
        raise EntsoeLocalReadinessError(
            "ENTSO-E frames do not exactly match the manifest"
        )

    file_rows: list[dict[str, object]] = []
    series_rows: list[dict[str, object]] = []
    for file_name in sorted(frames):
        frame = frames[file_name]
        entry = declared[file_name]
        if not isinstance(entry, Mapping):
            raise EntsoeLocalReadinessError(
                f"ENTSO-E manifest entry is invalid: {file_name}"
            )
        _validate_frame(file_name, frame, entry)
        file_row, file_series = _profile_frame(file_name, frame)
        file_rows.append(file_row)
        series_rows.extend(file_series)

    file_profile = pd.DataFrame(file_rows).sort_values("file").reset_index(drop=True)
    series_profile = (
        pd.DataFrame(series_rows)
        .sort_values(["role", "file", "series"])
        .reset_index(drop=True)
    )

    missing_required = sorted(set(REQUIRED_FILES).difference(frames))
    projection_profile = _projection_profile(frames)
    projection_checks = {
        str(projection): {
            "passed": bool(group["exact_match"].all()),
            "compared_series_count": len(group),
            "divergent_series_count": int(group["exact_match"].eq(False).sum()),
            "total_mismatch_cell_count": int(group["mismatch_count"].sum()),
        }
        for projection, group in projection_profile.groupby(
            "projection_file", sort=True
        )
    }
    missing_pit_files = sorted(
        file_name
        for file_name in FORECAST_FILES
        if file_name in frames
        and not REQUIRED_POINT_IN_TIME_COLUMNS.issubset(
            {str(column).lower() for column in frames[file_name].columns}
        )
    )
    all_columns = {
        str(column).lower()
        for frame in frames.values()
        for column in frame.columns
    }
    missing_lineage = sorted(RAW_LINEAGE_COLUMNS.difference(all_columns))
    role_coverage = _role_coverage(series_profile)

    findings: list[dict[str, object]] = [
        {
            "code": "ARCHIVE_AUTHORITY_FORBIDS_EMPIRICAL_USE",
            "severity": "CRITICAL",
            "evidence": {
                "calibration_eligible": manifest["calibration_eligible"],
                "model_activation": manifest["model_activation"],
                "source_coherence": manifest["source_coherence"],
            },
            "impact": (
                "The import cannot support training, rolling-origin selection "
                "or production evidence."
            ),
        }
    ]
    if missing_pit_files:
        findings.append(
            {
                "code": "FORECAST_POINT_IN_TIME_COLUMNS_MISSING",
                "severity": "CRITICAL",
                "evidence": {"files": missing_pit_files},
                "impact": (
                    "Forecast rows can leak later revisions into historical "
                    "origins because their availability time is unknown."
                ),
            }
        )
    if missing_lineage:
        findings.append(
            {
                "code": "RAW_PROVIDER_LINEAGE_NOT_REPLAYABLE",
                "severity": "HIGH",
                "evidence": {"missing_columns": missing_lineage},
                "impact": (
                    "Units, native cadence, series identity, revisions and "
                    "source responses cannot be independently reconstructed."
                ),
            }
        )
    insufficient_roles = sorted(
        role
        for role, coverage in role_coverage.items()
        if role
        in {
            "neighbor_actual",
            "physical_flow",
            "scheduled_exchange",
            "ntc",
        }
        and float(coverage["minimum_observed_span_days"])
        < MIN_SEASONAL_HISTORY_DAYS
    )
    if insufficient_roles:
        findings.append(
            {
                "code": "CROSS_MARKET_SEASONAL_HISTORY_INSUFFICIENT",
                "severity": "HIGH",
                "evidence": {
                    "minimum_required_days": MIN_SEASONAL_HISTORY_DAYS,
                    "roles": insufficient_roles,
                },
                "impact": (
                    "The archive cannot identify stable seasonal or regime "
                    "effects for Swiss cross-market shaping."
                ),
            }
        )
    gap_files = file_profile.loc[
        file_profile["non_15min_index_gap_count"].gt(0), "file"
    ].tolist()
    if gap_files:
        findings.append(
            {
                "code": "FILE_INDEX_GAPS_PRESENT",
                "severity": "MEDIUM",
                "evidence": {"files": gap_files},
                "impact": (
                    "Gap-aware folds and explicit missingness handling are "
                    "required even for local tooling."
                ),
            }
        )
    if missing_required:
        findings.append(
            {
                "code": "REQUIRED_REUSABLE_FILES_MISSING",
                "severity": "HIGH",
                "evidence": {"files": missing_required},
                "impact": "The expected local schema-development surface is incomplete.",
            }
        )
    failed_projections = sorted(
        name
        for name, check in projection_checks.items()
        if check["passed"] is not True
    )
    if failed_projections:
        findings.append(
            {
                "code": "DUPLICATE_PROJECTION_DIVERGENCE",
                "severity": "HIGH",
                "evidence": {"checks": failed_projections},
                "impact": (
                    "Dedicated and combined views disagree on nominally "
                    "identical timestamp/series values."
                ),
            }
        )

    local_integrity_pass = not missing_required and not failed_projections
    summary = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "status": (
            "PASS_LOCAL_SCHEMA_AND_QUALITY_PROFILE_NO_GO_EMPIRICAL_USE"
            if local_integrity_pass
            else "FAIL_LOCAL_SCHEMA_OR_CONSISTENCY_NO_GO_EMPIRICAL_USE"
        ),
        "dataset": {
            "import_id": manifest["import_id"],
            "manifest_sha256": manifest_hash,
            "grain": "timestamp_utc_x_wide_derived_series_column",
            "expected_cadence": "PT15M",
            "file_count": len(frames),
            "series_profile_count": len(series_profile),
        },
        "checks": {
            "archive_integrity_verified": True,
            "required_files_present": not missing_required,
            "duplicate_projection_checks": projection_checks,
            "all_indexes_unique": bool(file_profile["index_unique"].all()),
            "all_indexes_monotonic": bool(file_profile["index_monotonic"].all()),
            "all_indexes_utc": bool(file_profile["index_timezone"].eq("UTC").all()),
        },
        "role_coverage": role_coverage,
        "findings": findings,
        "authority": {
            "local_schema_and_quality_tooling": True,
            "historical_empirical_validation": False,
            "rolling_origin_selection": False,
            "model_training": False,
            "model_selection": False,
            "candidate_assembly": False,
            "production_promotion": False,
        },
    }
    return EntsoeLocalReadinessAudit(
        summary=summary,
        file_profile=file_profile,
        series_profile=series_profile,
        projection_profile=projection_profile,
    )


def _validate_manifest_authority(manifest: Mapping[str, object]) -> None:
    expected = {
        "schema_version": EXPECTED_IMPORT_SCHEMA,
        "calibration_eligible": False,
        "model_activation": EXPECTED_MODEL_ACTIVATION,
        "source_coherence": EXPECTED_SOURCE_COHERENCE,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise EntsoeLocalReadinessError(
                f"ENTSO-E reusable import authority differs: {key}"
            )
    import_id = str(manifest.get("import_id", "")).strip()
    if not import_id:
        raise EntsoeLocalReadinessError("ENTSO-E import_id is missing")


def _validate_frame(
    file_name: str,
    frame: pd.DataFrame,
    entry: Mapping[str, object],
) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise EntsoeLocalReadinessError(f"ENTSO-E frame is empty: {file_name}")
    if len(frame) != int(entry.get("rows", -1)):
        raise EntsoeLocalReadinessError(f"ENTSO-E row count differs: {file_name}")
    if [str(column) for column in frame.columns] != list(entry.get("columns", [])):
        raise EntsoeLocalReadinessError(f"ENTSO-E columns differ: {file_name}")
    if frame.columns.has_duplicates:
        raise EntsoeLocalReadinessError(
            f"ENTSO-E frame repeats a column: {file_name}"
        )
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise EntsoeLocalReadinessError(
            f"ENTSO-E index is not DatetimeIndex: {file_name}"
        )
    if index.hasnans or str(index.tz) != "UTC":
        raise EntsoeLocalReadinessError(
            f"ENTSO-E index is not complete UTC: {file_name}"
        )
    if not index.is_unique or not index.is_monotonic_increasing:
        raise EntsoeLocalReadinessError(
            f"ENTSO-E index is duplicated or unsorted: {file_name}"
        )
    differences_ns = (
        index.to_series()
        .diff()
        .dropna()
        .astype("timedelta64[ns]")
        .astype("int64")
    )
    expected_ns = int(EXPECTED_CADENCE.value)
    if bool(
        differences_ns.lt(expected_ns).any()
        or differences_ns.mod(expected_ns).ne(0).any()
    ):
        raise EntsoeLocalReadinessError(
            f"ENTSO-E index is not aligned to the 15-minute grid: {file_name}"
        )
    for column in frame.select_dtypes(include="number").columns:
        values = frame[column].dropna().to_numpy(dtype=float)
        if not all(math.isfinite(value) for value in values):
            raise EntsoeLocalReadinessError(
                f"ENTSO-E numeric series is non-finite: {file_name}/{column}"
            )


def _profile_frame(
    file_name: str,
    frame: pd.DataFrame,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    index = frame.index
    gaps = index.to_series().diff().dropna()
    completeness = frame.notna().mean()
    all_zero: list[str] = []
    series_rows: list[dict[str, object]] = []
    for column in frame.columns:
        values = frame[column].dropna()
        is_all_zero = False
        if pd.api.types.is_numeric_dtype(frame[column].dtype) and not values.empty:
            is_all_zero = bool(values.eq(0).all())
        if is_all_zero:
            all_zero.append(str(column))
        first_valid = values.index.min() if not values.empty else None
        last_valid = values.index.max() if not values.empty else None
        if first_valid is None or last_valid is None:
            expected_internal = 0
            observed_span_days = 0.0
        else:
            expected_internal = int(
                (last_valid - first_valid) / EXPECTED_CADENCE
            ) + 1
            observed_span_days = (
                (last_valid - first_valid).total_seconds() / 86_400.0
            )
        series_rows.append(
            {
                "file": file_name,
                "series": str(column),
                "role": _series_role(file_name, str(column)),
                "row_count": len(frame),
                "valid_count": len(values),
                "missing_count": len(frame) - len(values),
                "global_completeness": float(len(values) / len(frame)),
                "first_valid_utc": first_valid,
                "last_valid_utc": last_valid,
                "observed_span_days": observed_span_days,
                "internal_expected_count": expected_internal,
                "internal_completeness": (
                    float(len(values) / expected_internal)
                    if expected_internal
                    else 0.0
                ),
                "all_zero_non_null": is_all_zero,
                "point_in_time_proven": False,
                "native_resolution_proven": False,
            }
        )
    return (
        {
            "file": file_name,
            "row_count": len(frame),
            "column_count": len(frame.columns),
            "index_min_utc": index.min(),
            "index_max_utc": index.max(),
            "index_timezone": str(index.tz),
            "index_unique": bool(index.is_unique),
            "index_monotonic": bool(index.is_monotonic_increasing),
            "non_15min_index_gap_count": int((gaps != EXPECTED_CADENCE).sum()),
            "maximum_index_gap_seconds": (
                int(gaps.max().total_seconds()) if not gaps.empty else 0
            ),
            "minimum_column_completeness": float(completeness.min()),
            "median_column_completeness": float(completeness.median()),
            "columns_below_95pct_complete": int(completeness.lt(0.95).sum()),
            "all_null_column_count": int(completeness.eq(0).sum()),
            "all_zero_non_null_column_count": len(all_zero),
        },
        series_rows,
    )


def _series_role(file_name: str, column: str) -> str:
    if file_name in FORECAST_FILES or column.startswith("forecast_"):
        return "forecast"
    if file_name == "outages_15min.parquet" or column.startswith("unavailable_"):
        return "outage"
    if column in _CH_ACTUALS:
        return "ch_actual"
    if re.fullmatch(r"(?:load|solar|wind)_(?:de|fr|at|it)_mw", column):
        return "neighbor_actual"
    if re.fullmatch(r"flow_net_export_ch_(?:de|fr|at|it)_mw", column):
        return "physical_flow"
    if re.fullmatch(r"scheduled_net_export_ch_(?:de|fr|at|it)_mw", column):
        return "scheduled_exchange"
    if re.fullmatch(
        r"ntc_(?:export|import|net|total)_ch_(?:de|fr|at|it)_mw",
        column,
    ):
        return "ntc"
    return "derived"


def _projection_profile(
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    combined = frames.get("entso_15min.parquet")
    rows: list[dict[str, object]] = []
    for dedicated_name in (
        "entso_fundamentals_15min.parquet",
        "entso_border_15min.parquet",
    ):
        dedicated = frames.get(dedicated_name)
        if combined is None or dedicated is None:
            rows.append(
                _structural_projection_row(
                    dedicated_name,
                    "COMBINED_OR_DEDICATED_FILE_MISSING",
                )
            )
            continue
        if not dedicated.index.equals(combined.index):
            rows.append(
                _structural_projection_row(
                    dedicated_name,
                    "TIMESTAMP_INDEX_DIFFERS",
                )
            )
            continue
        for column in dedicated.columns:
            series = str(column)
            if series not in combined.columns:
                rows.append(
                    _structural_projection_row(
                        dedicated_name,
                        f"COMBINED_COLUMN_MISSING:{series}",
                    )
                )
                continue
            left = combined[series]
            right = dedicated[series]
            null_asymmetry = left.isna() ^ right.isna()
            both_present = left.notna() & right.notna()
            if pd.api.types.is_numeric_dtype(
                left.dtype
            ) and pd.api.types.is_numeric_dtype(right.dtype):
                absolute_difference = (
                    left[both_present].astype(float)
                    - right[both_present].astype(float)
                ).abs()
                value_mismatch = absolute_difference.ne(0)
                max_abs_difference = (
                    float(absolute_difference.max())
                    if not absolute_difference.empty
                    else 0.0
                )
            else:
                value_mismatch = left[both_present].astype("string").ne(
                    right[both_present].astype("string")
                )
                max_abs_difference = None
            mismatch = null_asymmetry.copy()
            mismatch.loc[value_mismatch.index] |= value_mismatch
            mismatch_index = mismatch[mismatch].index
            mismatch_gaps = mismatch_index.to_series().diff().dropna()
            rows.append(
                {
                    "projection_file": dedicated_name,
                    "series": series,
                    "structural_error": None,
                    "combined_dtype": str(left.dtype),
                    "dedicated_dtype": str(right.dtype),
                    "null_asymmetry_count": int(null_asymmetry.sum()),
                    "value_mismatch_count": int(value_mismatch.sum()),
                    "mismatch_count": len(mismatch_index),
                    "first_mismatch_utc": (
                        mismatch_index.min() if len(mismatch_index) else None
                    ),
                    "last_mismatch_utc": (
                        mismatch_index.max() if len(mismatch_index) else None
                    ),
                    "non_contiguous_break_count": int(
                        (mismatch_gaps != EXPECTED_CADENCE).sum()
                    ),
                    "max_abs_difference": max_abs_difference,
                    "exact_match": len(mismatch_index) == 0
                    and str(left.dtype) == str(right.dtype),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["projection_file", "series"])
        .reset_index(drop=True)
    )


def _structural_projection_row(
    projection_file: str,
    error: str,
) -> dict[str, object]:
    return {
        "projection_file": projection_file,
        "series": "__STRUCTURE__",
        "structural_error": error,
        "combined_dtype": None,
        "dedicated_dtype": None,
        "null_asymmetry_count": 0,
        "value_mismatch_count": 0,
        "mismatch_count": 1,
        "first_mismatch_utc": None,
        "last_mismatch_utc": None,
        "non_contiguous_break_count": 0,
        "max_abs_difference": None,
        "exact_match": False,
    }


def _role_coverage(series_profile: pd.DataFrame) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for role, group in series_profile.groupby("role", sort=True):
        first_valid = group["first_valid_utc"].dropna()
        last_valid = group["last_valid_utc"].dropna()
        result[str(role)] = {
            "series_count": len(group),
            "minimum_global_completeness": float(group["global_completeness"].min()),
            "median_global_completeness": float(
                group["global_completeness"].median()
            ),
            "maximum_global_completeness": float(group["global_completeness"].max()),
            "minimum_internal_completeness": float(
                group["internal_completeness"].min()
            ),
            "minimum_observed_span_days": float(group["observed_span_days"].min()),
            "maximum_observed_span_days": float(group["observed_span_days"].max()),
            "earliest_valid_utc": (
                first_valid.min().isoformat() if not first_valid.empty else None
            ),
            "latest_valid_utc": (
                last_valid.max().isoformat() if not last_valid.empty else None
            ),
        }
    return result


__all__ = [
    "EntsoeLocalReadinessAudit",
    "EntsoeLocalReadinessError",
    "MIN_SEASONAL_HISTORY_DAYS",
    "READINESS_SCHEMA_VERSION",
    "REQUIRED_FILES",
    "audit_entsoe_local_readiness",
]
