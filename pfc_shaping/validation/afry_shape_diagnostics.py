"""Calendar-free diagnostics for the restricted AFRY Swiss scenario catalog.

The outputs are local vendor-benchmark evidence only. Representative-hour
slots are deliberately not interpreted as UTC, Europe/Zurich, monthly or
quarter-hour timestamps, and no output has model, monthly-level or production
authority.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path, PurePosixPath

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pfc_shaping.data import afry_scenarios as afry_scenarios_module
from pfc_shaping.data.afry_scenarios import (
    canonical_json_bytes,
    file_sha256,
    runtime_tree_receipt,
    verify_catalog_bundle,
)

DIAGNOSTIC_SCHEMA = "fmv-afry-shape-diagnostic-bundle-v1"
DIAGNOSTIC_STATUS = "PASS_LOCAL_DIAGNOSTIC_ONLY"
DECISION_STATE = "NO_GO_MODEL_AND_PRODUCTION"
CENTRAL_BLOCK_SLOTS = (10, 11, 12, 13, 14)
LATE_BLOCK_SLOTS = (18, 19, 20, 21)
EXPECTED_INPUT_ARTIFACTS = {
    "hour_slot_shape_benchmark.parquet",
    "hourly_profile.json",
}
EXPECTED_OUTPUT_ARTIFACTS = {
    "group_shape_diagnostics.parquet",
    "term_shape_changes.parquet",
    "weather_shape_sensitivity.parquet",
    "diagnostic-audit.json",
}
CONTRACT_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "AFRY-CH-2026-Q2-SHAPE-DIAGNOSTIC-CONTRACT-V1.json"
)
ZERO_MEAN_TOLERANCE_EUR_MWH = 1e-12
CANONICAL_WORKSPACE = Path(r"C:\Users\jbattaglia\PFC_LT")
MAX_MANIFEST_BYTES = 1_000_000
MAX_JSON_ARTIFACT_BYTES = 5_000_000
MAX_PARQUET_ARTIFACT_BYTES = 100_000_000
MAX_DIAGNOSTIC_ROWS = 100_000


class AfryShapeDiagnosticError(ValueError):
    """Raised when a diagnostic input or output violates the local contract."""


def _file_identity(path_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(path_stat.st_dev),
        int(path_stat.st_ino),
        int(path_stat.st_size),
        int(path_stat.st_mtime_ns),
        int(path_stat.st_nlink),
    )


def _directory_identity(path_stat: os.stat_result) -> tuple[int, int]:
    return (int(path_stat.st_dev), int(path_stat.st_ino))


def _is_link_or_reparse(path: Path, path_stat: os.stat_result) -> bool:
    attributes = int(getattr(path_stat, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return path.is_symlink() or bool(attributes & reparse_flag)


def _require_plain_directory(path: Path, *, label: str) -> os.stat_result:
    try:
        path_stat = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise AfryShapeDiagnosticError(f"{label} is unavailable: {path}") from exc
    if _is_link_or_reparse(path, path_stat) or not stat.S_ISDIR(path_stat.st_mode):
        raise AfryShapeDiagnosticError(f"{label} is not a plain directory: {path}")
    return path_stat


def _stable_file_bytes(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
) -> tuple[bytes, str, int, tuple[int, int, int, int, int]]:
    try:
        before = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise AfryShapeDiagnosticError(f"{label} is unavailable: {path}") from exc
    if _is_link_or_reparse(path, before) or not stat.S_ISREG(before.st_mode):
        raise AfryShapeDiagnosticError(f"{label} is not a plain file: {path}")
    if before.st_size <= 0 or before.st_size > maximum_bytes:
        raise AfryShapeDiagnosticError(f"{label} size is outside the governed bound")
    identity_before = _file_identity(before)
    try:
        payload = path.read_bytes()
        after = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise AfryShapeDiagnosticError(f"{label} changed or became unreadable") from exc
    identity_after = _file_identity(after)
    if (
        identity_before != identity_after
        or _is_link_or_reparse(path, after)
        or len(payload) != before.st_size
    ):
        raise AfryShapeDiagnosticError(f"{label} changed while read")
    return payload, hashlib.sha256(payload).hexdigest(), len(payload), identity_before


def _assert_file_identity(
    path: Path,
    expected: tuple[int, int, int, int, int],
    *,
    label: str,
) -> None:
    try:
        observed = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise AfryShapeDiagnosticError(f"{label} changed after integrity capture") from exc
    if _is_link_or_reparse(path, observed) or _file_identity(observed) != expected:
        raise AfryShapeDiagnosticError(f"{label} changed after integrity capture")


def _require_canonical_workspace(workspace: Path) -> Path:
    resolved = workspace.resolve()
    if str(resolved) != str(CANONICAL_WORKSPACE):
        raise AfryShapeDiagnosticError(
            f"diagnostic workspace must be exactly {CANONICAL_WORKSPACE}; got {resolved}"
        )
    observed_cwd = _current_directory()
    if str(observed_cwd) != str(CANONICAL_WORKSPACE):
        raise AfryShapeDiagnosticError(
            f"diagnostic cwd must be exactly {CANONICAL_WORKSPACE}; got {observed_cwd}"
        )
    observed_git_root = _git_top_level(resolved)
    if str(observed_git_root) != str(CANONICAL_WORKSPACE):
        raise AfryShapeDiagnosticError(
            "diagnostic Git root must be exactly "
            f"{CANONICAL_WORKSPACE}; got {observed_git_root}"
        )
    _require_plain_directory(resolved, label="diagnostic workspace")
    return resolved


def _current_directory() -> Path:
    return Path.cwd().resolve()


def _git_top_level(workspace: Path) -> Path:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AfryShapeDiagnosticError("diagnostic Git root is unavailable") from exc
    return Path(result.stdout.strip()).resolve()


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _require_plain_workspace_ancestry(
    workspace: Path,
    target: Path,
    *,
    label: str,
) -> Path:
    absolute = Path(os.path.abspath(str(target)))
    try:
        relative = absolute.relative_to(workspace)
    except ValueError as exc:
        raise AfryShapeDiagnosticError(f"{label} must remain inside workspace") from exc
    current = workspace
    for part in relative.parts:
        current = current / part
        try:
            current_stat = current.stat(follow_symlinks=False)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise AfryShapeDiagnosticError(f"{label} ancestry is unreadable: {current}") from exc
        if _is_link_or_reparse(current, current_stat):
            raise AfryShapeDiagnosticError(f"{label} ancestry contains a reparse point: {current}")
    return absolute


_AUTHORITY_METADATA = {
    b"usage": b"restricted_vendor_benchmark_only",
    b"calendar_mapping_authorized": b"false",
    b"monthly_level_authority": b"false",
    b"model_input_authorized": b"false",
    b"production_authorized": b"false",
}

GROUP_SCHEMA = pa.schema(
    [
        pa.field("release_id", pa.string(), nullable=False),
        pa.field("scenario_vendor", pa.string(), nullable=False),
        pa.field("delivery_year", pa.int16(), nullable=False),
        pa.field("weather_year", pa.int16(), nullable=False),
        pa.field("central_block_additive_mean_eur_mwh", pa.float64(), nullable=False),
        pa.field("late_block_additive_mean_eur_mwh", pa.float64(), nullable=False),
        pa.field("late_minus_central_block_eur_mwh", pa.float64(), nullable=False),
        pa.field("shape_amplitude_eur_mwh", pa.float64(), nullable=False),
        pa.field("cyclic_slot_roughness_eur_mwh", pa.float64(), nullable=False),
        pa.field("negative_price_share", pa.float64(), nullable=False),
        pa.field("strict_negative_price_share", pa.float64(), nullable=False),
        pa.field("near_zero_negative_noise_share", pa.float64(), nullable=False),
        pa.field("calendar_mapping_authorized", pa.bool_(), nullable=False),
        pa.field("monthly_level_authority", pa.bool_(), nullable=False),
        pa.field("model_input_authorized", pa.bool_(), nullable=False),
        pa.field("production_authorized", pa.bool_(), nullable=False),
    ],
    metadata={
        **_AUTHORITY_METADATA,
        b"slot_semantics": b"ordered_representative_slots_not_clock_hours",
        b"normalization": b"input_additive_shape_zero_mean_across_24_slots",
    },
)

WEATHER_SCHEMA = pa.schema(
    [
        pa.field("release_id", pa.string(), nullable=False),
        pa.field("scenario_vendor", pa.string(), nullable=False),
        pa.field("delivery_year", pa.int16(), nullable=False),
        pa.field("weather_pattern_count", pa.int8(), nullable=False),
        pa.field("duck_spread_min_eur_mwh", pa.float64(), nullable=False),
        pa.field("duck_spread_median_eur_mwh", pa.float64(), nullable=False),
        pa.field("duck_spread_max_eur_mwh", pa.float64(), nullable=False),
        pa.field("shape_amplitude_min_eur_mwh", pa.float64(), nullable=False),
        pa.field("shape_amplitude_median_eur_mwh", pa.float64(), nullable=False),
        pa.field("shape_amplitude_max_eur_mwh", pa.float64(), nullable=False),
        pa.field("negative_share_min", pa.float64(), nullable=False),
        pa.field("negative_share_median", pa.float64(), nullable=False),
        pa.field("negative_share_max", pa.float64(), nullable=False),
        pa.field("probability_weighting_authorized", pa.bool_(), nullable=False),
        pa.field("calendar_mapping_authorized", pa.bool_(), nullable=False),
        pa.field("monthly_level_authority", pa.bool_(), nullable=False),
        pa.field("model_input_authorized", pa.bool_(), nullable=False),
        pa.field("production_authorized", pa.bool_(), nullable=False),
    ],
    metadata={
        **_AUTHORITY_METADATA,
        b"aggregation": b"unweighted_min_median_max_sensitivity_not_probability",
    },
)

TERM_SCHEMA = pa.schema(
    [
        pa.field("release_id", pa.string(), nullable=False),
        pa.field("scenario_vendor", pa.string(), nullable=False),
        pa.field("weather_year", pa.int16(), nullable=False),
        pa.field("from_delivery_year", pa.int16(), nullable=False),
        pa.field("to_delivery_year", pa.int16(), nullable=False),
        pa.field("delivery_year_gap", pa.int16(), nullable=False),
        pa.field("duck_spread_change_eur_mwh", pa.float64(), nullable=False),
        pa.field("duck_spread_change_per_year_eur_mwh", pa.float64(), nullable=False),
        pa.field("shape_amplitude_change_eur_mwh", pa.float64(), nullable=False),
        pa.field("negative_share_change", pa.float64(), nullable=False),
        pa.field("calendar_mapping_authorized", pa.bool_(), nullable=False),
        pa.field("monthly_level_authority", pa.bool_(), nullable=False),
        pa.field("model_input_authorized", pa.bool_(), nullable=False),
        pa.field("production_authorized", pa.bool_(), nullable=False),
    ],
    metadata={
        **_AUTHORITY_METADATA,
        b"term_semantics": b"adjacent_available_delivery_years_no_interpolation",
    },
)


def _profile_groups(profile: dict[str, object]) -> dict[tuple[str, int, int], dict[str, object]]:
    groups: dict[tuple[str, int, int], dict[str, object]] = {}
    sheets = profile.get("sheets")
    if not isinstance(sheets, list):
        raise AfryShapeDiagnosticError("AFRY hourly profile sheets must be a list")
    for sheet in sheets:
        if not isinstance(sheet, dict):
            raise AfryShapeDiagnosticError("AFRY hourly profile sheet must be an object")
        scenario = str(sheet.get("scenario_vendor", ""))
        sheet_groups = sheet.get("groups")
        if not scenario or not isinstance(sheet_groups, list):
            raise AfryShapeDiagnosticError("AFRY hourly profile sheet semantics are incomplete")
        for group in sheet_groups:
            if not isinstance(group, dict):
                raise AfryShapeDiagnosticError("AFRY hourly profile group must be an object")
            key = (scenario, int(group["delivery_year"]), int(group["weather_year"]))
            if key in groups:
                raise AfryShapeDiagnosticError(f"duplicate AFRY hourly profile group: {key}")
            rows = int(group["rows"])
            strict_negative = int(group["strict_negative_prices"])
            negative = int(group["negative_prices"])
            noise = int(group["near_zero_negative_noise_count"])
            if (
                rows <= 0
                or not 0 <= negative <= strict_negative <= rows
                or noise != strict_negative - negative
            ):
                raise AfryShapeDiagnosticError(
                    f"invalid AFRY negative-price counts for group {key}"
                )
            groups[key] = group
    if not groups:
        raise AfryShapeDiagnosticError("AFRY hourly profile contains no groups")
    return groups


def _assert_false_authorities(frame: pd.DataFrame) -> None:
    for field in (
        "calendar_mapping_authorized",
        "monthly_level_authority",
        "model_input_authorized",
        "production_authorized",
    ):
        if field in frame and not frame[field].eq(False).all():
            raise AfryShapeDiagnosticError(f"AFRY diagnostic authority {field} must be false")


def build_shape_diagnostic_frames(
    shape: pd.DataFrame,
    hourly_profile: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build calendar-free, non-authoritative group, weather and term diagnostics."""

    required = {
        "release_id",
        "scenario_vendor",
        "delivery_year",
        "weather_year",
        "representative_hour_slot",
        "annual_zero_mean_additive_shape_eur_mwh",
        "model_input_authorized",
        "monthly_level_authority",
        "production_authorized",
    }
    missing = required - set(shape.columns)
    if missing:
        raise AfryShapeDiagnosticError(f"AFRY shape input columns missing: {sorted(missing)}")
    if shape.empty or shape[list(required)].isna().any().any():
        raise AfryShapeDiagnosticError("AFRY shape input must be nonempty and complete")
    for field in ("model_input_authorized", "monthly_level_authority", "production_authorized"):
        if not shape[field].eq(False).all():
            raise AfryShapeDiagnosticError(f"AFRY input authority {field} must be false")
    release_ids = {str(value) for value in shape["release_id"].unique()}
    profile_release = str(hourly_profile.get("release_id", ""))
    if len(release_ids) != 1 or release_ids != {profile_release}:
        raise AfryShapeDiagnosticError("AFRY shape/profile release IDs do not match exactly")
    profile_groups = _profile_groups(hourly_profile)
    declared_weather_years = hourly_profile.get("weather_years")
    representative_hours = hourly_profile.get("representative_hours_per_group")
    if (
        not isinstance(declared_weather_years, list)
        or not declared_weather_years
        or any(type(value) is not int for value in declared_weather_years)
        or len(set(declared_weather_years)) != len(declared_weather_years)
    ):
        raise AfryShapeDiagnosticError("AFRY profile weather-year declaration is invalid")
    if representative_hours != 8760:
        raise AfryShapeDiagnosticError("AFRY profile must retain 8760 representative hours")
    expected_weather_years = set(declared_weather_years)
    for key, group in profile_groups.items():
        if int(group["rows"]) != representative_hours:
            raise AfryShapeDiagnosticError(f"AFRY profile row count is not 8760: {key}")
    group_columns = ["scenario_vendor", "delivery_year", "weather_year"]
    grouped = shape.groupby(group_columns, sort=True, observed=True)
    shape_keys = {
        (str(scenario), int(delivery), int(weather))
        for scenario, delivery, weather in grouped.groups
    }
    if shape_keys != set(profile_groups):
        raise AfryShapeDiagnosticError("AFRY shape/profile group coverage does not match exactly")
    weather_by_scenario_delivery: dict[tuple[str, int], set[int]] = {}
    for scenario, delivery, weather in shape_keys:
        weather_by_scenario_delivery.setdefault((scenario, delivery), set()).add(weather)
    if any(values != expected_weather_years for values in weather_by_scenario_delivery.values()):
        raise AfryShapeDiagnosticError(
            "AFRY diagnostic groups must preserve every declared weather pattern"
        )

    rows: list[dict[str, object]] = []
    zero_mean_errors: list[float] = []
    for raw_key, group in grouped:
        key = (str(raw_key[0]), int(raw_key[1]), int(raw_key[2]))
        ordered = group.sort_values("representative_hour_slot")
        slots = [int(value) for value in ordered["representative_hour_slot"]]
        if slots != list(range(24)):
            raise AfryShapeDiagnosticError(f"AFRY group lacks exact slots 0..23: {key}")
        values = [
            float(value)
            for value in ordered["annual_zero_mean_additive_shape_eur_mwh"]
        ]
        if not all(math.isfinite(value) for value in values):
            raise AfryShapeDiagnosticError(f"AFRY shape contains non-finite values: {key}")
        zero_mean_error = math.fsum(values) / 24.0
        zero_mean_errors.append(zero_mean_error)
        if abs(zero_mean_error) > ZERO_MEAN_TOLERANCE_EUR_MWH:
            raise AfryShapeDiagnosticError(f"AFRY additive shape is not zero mean: {key}")
        central = math.fsum(values[slot] for slot in CENTRAL_BLOCK_SLOTS) / len(
            CENTRAL_BLOCK_SLOTS
        )
        late = math.fsum(values[slot] for slot in LATE_BLOCK_SLOTS) / len(
            LATE_BLOCK_SLOTS
        )
        cyclic_differences = [
            values[(slot + 1) % 24] - values[slot] for slot in range(24)
        ]
        roughness = math.sqrt(
            math.fsum(value * value for value in cyclic_differences) / 24.0
        )
        profile_group = profile_groups[key]
        count = int(profile_group["rows"])
        negative = int(profile_group["negative_prices"])
        strict_negative = int(profile_group["strict_negative_prices"])
        noise = int(profile_group["near_zero_negative_noise_count"])
        rows.append(
            {
                "release_id": profile_release,
                "scenario_vendor": key[0],
                "delivery_year": key[1],
                "weather_year": key[2],
                "central_block_additive_mean_eur_mwh": central,
                "late_block_additive_mean_eur_mwh": late,
                "late_minus_central_block_eur_mwh": late - central,
                "shape_amplitude_eur_mwh": max(values) - min(values),
                "cyclic_slot_roughness_eur_mwh": roughness,
                "negative_price_share": negative / count,
                "strict_negative_price_share": strict_negative / count,
                "near_zero_negative_noise_share": noise / count,
                "calendar_mapping_authorized": False,
                "monthly_level_authority": False,
                "model_input_authorized": False,
                "production_authorized": False,
            }
        )
    group_frame = pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)
    _assert_false_authorities(group_frame)

    weather_rows: list[dict[str, object]] = []
    for (scenario, delivery), group in group_frame.groupby(
        ["scenario_vendor", "delivery_year"], sort=True, observed=True
    ):
        duck = group["late_minus_central_block_eur_mwh"]
        amplitude = group["shape_amplitude_eur_mwh"]
        negative = group["negative_price_share"]
        weather_rows.append(
            {
                "release_id": profile_release,
                "scenario_vendor": str(scenario),
                "delivery_year": int(delivery),
                "weather_pattern_count": len(group),
                "duck_spread_min_eur_mwh": float(duck.min()),
                "duck_spread_median_eur_mwh": float(duck.median()),
                "duck_spread_max_eur_mwh": float(duck.max()),
                "shape_amplitude_min_eur_mwh": float(amplitude.min()),
                "shape_amplitude_median_eur_mwh": float(amplitude.median()),
                "shape_amplitude_max_eur_mwh": float(amplitude.max()),
                "negative_share_min": float(negative.min()),
                "negative_share_median": float(negative.median()),
                "negative_share_max": float(negative.max()),
                "probability_weighting_authorized": False,
                "calendar_mapping_authorized": False,
                "monthly_level_authority": False,
                "model_input_authorized": False,
                "production_authorized": False,
            }
        )
    weather_frame = pd.DataFrame(weather_rows).sort_values(
        ["scenario_vendor", "delivery_year"]
    ).reset_index(drop=True)
    _assert_false_authorities(weather_frame)
    if not weather_frame["probability_weighting_authorized"].eq(False).all():
        raise AfryShapeDiagnosticError("AFRY weather probabilities must remain unauthorized")

    term_rows: list[dict[str, object]] = []
    for (scenario, weather), group in group_frame.groupby(
        ["scenario_vendor", "weather_year"], sort=True, observed=True
    ):
        ordered = group.sort_values("delivery_year").reset_index(drop=True)
        for index in range(1, len(ordered)):
            previous = ordered.iloc[index - 1]
            current = ordered.iloc[index]
            gap = int(current["delivery_year"]) - int(previous["delivery_year"])
            if gap <= 0:
                raise AfryShapeDiagnosticError("AFRY delivery-year sequence must increase")
            duck_change = float(current["late_minus_central_block_eur_mwh"]) - float(
                previous["late_minus_central_block_eur_mwh"]
            )
            term_rows.append(
                {
                    "release_id": profile_release,
                    "scenario_vendor": str(scenario),
                    "weather_year": int(weather),
                    "from_delivery_year": int(previous["delivery_year"]),
                    "to_delivery_year": int(current["delivery_year"]),
                    "delivery_year_gap": gap,
                    "duck_spread_change_eur_mwh": duck_change,
                    "duck_spread_change_per_year_eur_mwh": duck_change / gap,
                    "shape_amplitude_change_eur_mwh": float(
                        current["shape_amplitude_eur_mwh"]
                    )
                    - float(previous["shape_amplitude_eur_mwh"]),
                    "negative_share_change": float(current["negative_price_share"])
                    - float(previous["negative_price_share"]),
                    "calendar_mapping_authorized": False,
                    "monthly_level_authority": False,
                    "model_input_authorized": False,
                    "production_authorized": False,
                }
            )
    term_frame = pd.DataFrame(term_rows).sort_values(
        ["scenario_vendor", "weather_year", "from_delivery_year"]
    ).reset_index(drop=True)
    _assert_false_authorities(term_frame)
    qualitative_by_scenario: dict[str, dict[str, bool]] = {}
    for scenario, scenario_frame in term_frame.groupby(
        "scenario_vendor", sort=True, observed=True
    ):
        deepening_then_recompression = []
        negative_share_nonmonotone = []
        for _, path in scenario_frame.groupby("weather_year", sort=True, observed=True):
            ordered = path.sort_values("from_delivery_year")
            duck_changes = list(ordered["duck_spread_change_per_year_eur_mwh"])
            negative_changes = list(ordered["negative_share_change"])
            deepening_then_recompression.append(
                any(
                    change > 0.0
                    and any(later < 0.0 for later in duck_changes[index + 1 :])
                    for index, change in enumerate(duck_changes[:-1])
                )
            )
            negative_share_nonmonotone.append(
                any(change > 0.0 for change in negative_changes)
                and any(change < 0.0 for change in negative_changes)
            )
        qualitative_by_scenario[str(scenario)] = {
            "deepening_then_recompression_all_weather_patterns": all(
                deepening_then_recompression
            ),
            "negative_share_nonmonotone_all_weather_patterns": all(
                negative_share_nonmonotone
            ),
        }
    audit = {
        "schema_version": "fmv-afry-shape-diagnostic-audit-v1",
        "status": DIAGNOSTIC_STATUS,
        "decision_state": DECISION_STATE,
        "release_id": profile_release,
        "group_count": len(group_frame),
        "weather_summary_count": len(weather_frame),
        "term_change_count": len(term_frame),
        "central_block_slots": list(CENTRAL_BLOCK_SLOTS),
        "late_block_slots": list(LATE_BLOCK_SLOTS),
        "slot_labels_are_clock_hours": False,
        "max_abs_input_zero_mean_error_eur_mwh": max(
            abs(value) for value in zero_mean_errors
        ),
        "zero_mean_tolerance_eur_mwh": ZERO_MEAN_TOLERANCE_EUR_MWH,
        "weather_aggregation": "UNWEIGHTED_SENSITIVITY_NOT_PROBABILITY",
        "qualitative_vendor_output": {
            "interpretation": "DESCRIPTIVE_VENDOR_OUTPUT_NOT_EMPIRICAL_OR_CAUSAL_VALIDATION",
            "by_scenario": qualitative_by_scenario,
            "weather_dispersion": {
                "duck_spread_nonzero_for_every_scenario_year": bool(
                    (
                        weather_frame["duck_spread_max_eur_mwh"]
                        > weather_frame["duck_spread_min_eur_mwh"]
                    ).all()
                ),
                "negative_share_nonzero_for_every_scenario_year": bool(
                    (weather_frame["negative_share_max"] > weather_frame["negative_share_min"]).all()
                ),
            },
        },
        "external_inputs_consumed": [],
        "empirical_validation_status": "BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS",
        "authorities": {
            "calendar_mapping_authorized": False,
            "probability_weighting_authorized": False,
            "monthly_level_authority": False,
            "model_input_authorized": False,
            "production_authorized": False,
        },
    }
    return group_frame, weather_frame, term_frame, audit


def _table_for_frame(frame: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    records = [
        {key: None if pd.isna(value) else value for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]
    return pa.Table.from_pylist(records, schema=schema)


def _write_frame(frame: pd.DataFrame, path: Path, schema: pa.Schema) -> None:
    table = _table_for_frame(frame, schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
        row_group_size=65_536,
    )


def _capture_artifact(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> None:
    before = source.stat(follow_symlinks=False)
    if _is_link_or_reparse(source, before) or not stat.S_ISREG(before.st_mode):
        raise AfryShapeDiagnosticError(f"AFRY diagnostic input is not plain: {source.name}")
    identity_before = _file_identity(before)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            writer.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    after = source.stat(follow_symlinks=False)
    if (
        _file_identity(after) != identity_before
        or _is_link_or_reparse(source, after)
        or size != expected_bytes
        or digest.hexdigest() != expected_sha256
    ):
        destination.unlink(missing_ok=True)
        raise AfryShapeDiagnosticError(f"AFRY diagnostic input capture mismatch: {source.name}")


def _replace_directory_once(source: Path, target: Path) -> None:
    source.replace(target)


def _publish_directory(source: Path, target: Path) -> None:
    """Bound transient Windows handle contention without weakening no-clobber."""

    delays = (0.05, 0.1, 0.2, 0.4, 0.8)
    for attempt in range(len(delays) + 1):
        try:
            _replace_directory_once(source, target)
            return
        except PermissionError:
            if target.exists() or attempt == len(delays):
                raise
            gc.collect()
            time.sleep(delays[attempt])


def _quarantine_invalid_publication(path: Path) -> Path:
    quarantine = path.parent / f".quarantine-{path.name}-{uuid.uuid4().hex}"
    _publish_directory(path, quarantine)
    if path.exists() or not quarantine.exists():
        raise AfryShapeDiagnosticError("invalid diagnostic publication quarantine failed")
    return quarantine


def _producer_files(workspace: Path) -> list[dict[str, object]]:
    sources = {
        "diagnostic_module": Path(__file__).resolve(),
        "diagnostic_entrypoint": (
            workspace / "scripts" / "build_afry_shape_diagnostics.py"
        ).resolve(),
        "source_catalog_module": Path(afry_scenarios_module.__file__).resolve(),
    }
    result = []
    for role, path in sources.items():
        try:
            relative = path.relative_to(workspace)
        except ValueError as exc:
            raise AfryShapeDiagnosticError(f"producer source outside workspace: {path}") from exc
        _, sha256, byte_count, _ = _stable_file_bytes(
            path,
            label=f"diagnostic producer {role}",
            maximum_bytes=5_000_000,
        )
        result.append(
            {
                "role": role,
                "path": relative.as_posix(),
                "sha256": sha256,
                "bytes": byte_count,
            }
        )
    return result


def _verify_deterministic_replay(
    *,
    bundle: Path,
    source_catalog_bundle: Path,
    source_catalog: dict[str, object],
    source_inputs: dict[str, dict[str, object]],
    output_artifacts: dict[str, dict[str, object]],
) -> None:
    """Recompute every derived output from byte-stable in-memory captures."""

    captured_sources: dict[str, bytes] = {}
    for name, item in source_inputs.items():
        maximum = (
            MAX_PARQUET_ARTIFACT_BYTES if name.endswith(".parquet") else MAX_JSON_ARTIFACT_BYTES
        )
        payload, sha256, byte_count, _ = _stable_file_bytes(
            source_catalog_bundle / name,
            label=f"diagnostic replay source {name}",
            maximum_bytes=maximum,
        )
        if sha256 != item["sha256"] or byte_count != item["bytes"]:
            raise AfryShapeDiagnosticError(f"diagnostic replay source mismatch: {name}")
        captured_sources[name] = payload
    captured_outputs: dict[str, bytes] = {}
    for name, item in output_artifacts.items():
        maximum = (
            MAX_PARQUET_ARTIFACT_BYTES if name.endswith(".parquet") else MAX_JSON_ARTIFACT_BYTES
        )
        payload, sha256, byte_count, _ = _stable_file_bytes(
            bundle / name,
            label=f"diagnostic replay output {name}",
            maximum_bytes=maximum,
        )
        if sha256 != item["sha256"] or byte_count != item["bytes"]:
            raise AfryShapeDiagnosticError(f"diagnostic replay output mismatch: {name}")
        captured_outputs[name] = payload
    source_shape = pq.read_table(
        pa.BufferReader(captured_sources["hour_slot_shape_benchmark.parquet"])
    ).to_pandas()
    source_profile = json.loads(captured_sources["hourly_profile.json"])
    expected_group, expected_weather, expected_term, expected_audit = (
        build_shape_diagnostic_frames(source_shape, source_profile)
    )
    expected_audit["source_catalog_bundle_id"] = source_catalog["bundle_id"]
    actual_frames = {
        name: pq.read_table(pa.BufferReader(captured_outputs[name])).to_pandas()
        for name in (
            "group_shape_diagnostics.parquet",
            "weather_shape_sensitivity.parquet",
            "term_shape_changes.parquet",
        )
    }
    for name, expected in (
        (
            "group_shape_diagnostics.parquet",
            _table_for_frame(expected_group, GROUP_SCHEMA).to_pandas(),
        ),
        (
            "weather_shape_sensitivity.parquet",
            _table_for_frame(expected_weather, WEATHER_SCHEMA).to_pandas(),
        ),
        (
            "term_shape_changes.parquet",
            _table_for_frame(expected_term, TERM_SCHEMA).to_pandas(),
        ),
    ):
        try:
            pd.testing.assert_frame_equal(
                actual_frames[name],
                expected,
                check_exact=True,
                check_dtype=True,
                check_like=False,
            )
        except AssertionError as exc:
            raise AfryShapeDiagnosticError(
                f"diagnostic deterministic replay mismatch: {name}"
            ) from exc
    actual_audit = json.loads(captured_outputs["diagnostic-audit.json"])
    if actual_audit != expected_audit:
        raise AfryShapeDiagnosticError("diagnostic deterministic replay mismatch: audit")


def verify_shape_diagnostic_bundle(
    bundle: Path,
    *,
    workspace: Path | None = None,
    source_catalog_bundle: Path | None = None,
) -> dict[str, object]:
    """Verify content identity, exact closure, schemas and negative authority."""

    workspace = _require_canonical_workspace(workspace or Path.cwd())
    build_root = _require_plain_workspace_ancestry(
        workspace,
        workspace / "build",
        label="diagnostic build root",
    )
    bundle = _require_plain_workspace_ancestry(
        workspace,
        bundle,
        label="diagnostic bundle",
    )
    try:
        bundle.relative_to(build_root)
    except ValueError as exc:
        raise AfryShapeDiagnosticError("diagnostic bundle must remain below workspace build/") from exc
    build_root_identity = _directory_identity(
        _require_plain_directory(build_root, label="diagnostic build root")
    )
    bundle_identity = _file_identity(
        _require_plain_directory(bundle, label="diagnostic bundle")
    )
    manifest_path = bundle / "diagnostic-manifest.json"
    checksum_path = bundle / "diagnostic-manifest.sha256"
    manifest_bytes, manifest_sha256, _, manifest_identity = _stable_file_bytes(
        manifest_path,
        label="diagnostic manifest",
        maximum_bytes=MAX_MANIFEST_BYTES,
    )
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AfryShapeDiagnosticError("diagnostic manifest is unreadable") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != DIAGNOSTIC_SCHEMA:
        raise AfryShapeDiagnosticError("unsupported diagnostic manifest schema")
    bundle_id = manifest.get("bundle_id")
    body = dict(manifest)
    body.pop("bundle_id", None)
    expected_id = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if bundle_id != expected_id or bundle.name != bundle_id:
        raise AfryShapeDiagnosticError("diagnostic content identity mismatch")
    if manifest.get("status") != DIAGNOSTIC_STATUS:
        raise AfryShapeDiagnosticError("diagnostic status is not local-only PASS")
    if manifest.get("decision_state") != DECISION_STATE:
        raise AfryShapeDiagnosticError("diagnostic decision state must remain NO_GO")
    if "qa_verdict" in manifest:
        raise AfryShapeDiagnosticError("diagnostic bundle forbids ambiguous qa_verdict")
    configuration = manifest.get("configuration")
    if configuration != {
        "central_block_slots": list(CENTRAL_BLOCK_SLOTS),
        "late_block_slots": list(LATE_BLOCK_SLOTS),
        "slot_labels_are_clock_hours": False,
        "weather_aggregation": "UNWEIGHTED_MIN_MEDIAN_MAX_NOT_PROBABILITY",
        "term_interpolation": False,
    }:
        raise AfryShapeDiagnosticError("diagnostic configuration is not exact")
    admission = manifest.get("admission")
    required_false = (
        "calendar_mapping_authorized",
        "probability_weighting_authorized",
        "monthly_level_authority",
        "model_input_authorized",
        "production_authorized",
    )
    if not isinstance(admission, dict) or any(admission.get(field) is not False for field in required_false):
        raise AfryShapeDiagnosticError("diagnostic admission invariants must remain false")
    if (
        admission.get("restricted_vendor_derived_values") is not True
        or admission.get("empirical_validation_status")
        != "BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS"
    ):
        raise AfryShapeDiagnosticError("diagnostic restricted-data or external blocker mismatch")
    contract = manifest.get("contract")
    if (
        not isinstance(contract, dict)
        or contract.get("path") != CONTRACT_RELATIVE_PATH
        or not isinstance(contract.get("bytes"), int)
        or contract["bytes"] <= 0
        or re.fullmatch(r"[0-9a-f]{64}", str(contract.get("sha256", ""))) is None
        or contract.get("stable_during_build") is not True
    ):
        raise AfryShapeDiagnosticError("diagnostic contract provenance is invalid")
    contract_path = (workspace / CONTRACT_RELATIVE_PATH).resolve()
    contract_bytes, contract_sha256, contract_size, _ = _stable_file_bytes(
        contract_path,
        label="canonical diagnostic contract",
        maximum_bytes=MAX_JSON_ARTIFACT_BYTES,
    )
    if (
        contract.get("sha256") != contract_sha256
        or contract.get("bytes") != contract_size
        or not contract_bytes
    ):
        raise AfryShapeDiagnosticError("diagnostic contract does not match canonical workspace")
    source_catalog = manifest.get("source_catalog")
    captured_inputs = (
        source_catalog.get("captured_artifacts") if isinstance(source_catalog, dict) else None
    )
    if (
        not isinstance(source_catalog, dict)
        or re.fullmatch(r"[0-9a-f]{64}", str(source_catalog.get("bundle_id", ""))) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(source_catalog.get("catalog_sha256", "")))
        is None
        or not isinstance(captured_inputs, list)
    ):
        raise AfryShapeDiagnosticError("diagnostic source-catalog provenance is invalid")
    if source_catalog_bundle is None:
        raise AfryShapeDiagnosticError("diagnostic source catalog caller anchor is required")
    manifest_source_bundle = _require_plain_workspace_ancestry(
        workspace,
        source_catalog_bundle,
        label="diagnostic source catalog",
    )
    try:
        manifest_source_bundle.relative_to(build_root)
    except ValueError as exc:
        raise AfryShapeDiagnosticError("diagnostic source catalog must remain below build/") from exc
    _require_plain_directory(manifest_source_bundle, label="diagnostic source catalog")
    source_catalog_path = manifest_source_bundle / "catalog.json"
    _, observed_catalog_sha256, _, source_catalog_identity = _stable_file_bytes(
        source_catalog_path,
        label="diagnostic source catalog manifest",
        maximum_bytes=MAX_MANIFEST_BYTES,
    )
    observed_source_catalog = verify_catalog_bundle(manifest_source_bundle)
    _assert_file_identity(
        source_catalog_path,
        source_catalog_identity,
        label="diagnostic source catalog manifest",
    )
    if (
        observed_source_catalog.get("bundle_id") != source_catalog.get("bundle_id")
        or observed_catalog_sha256 != source_catalog.get("catalog_sha256")
    ):
        raise AfryShapeDiagnosticError("diagnostic source catalog anchor mismatch")
    input_by_name = {
        str(item.get("name")): item for item in captured_inputs if isinstance(item, dict)
    }
    if set(input_by_name) != EXPECTED_INPUT_ARTIFACTS or len(input_by_name) != len(
        captured_inputs
    ):
        raise AfryShapeDiagnosticError("diagnostic captured-input inventory is not exact")
    for name, item in input_by_name.items():
        if (
            re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))) is None
            or not isinstance(item.get("bytes"), int)
            or item["bytes"] <= 0
        ):
            raise AfryShapeDiagnosticError(f"diagnostic captured input is invalid: {name}")
    observed_source_artifacts = {
        str(item.get("name")): item
        for item in observed_source_catalog.get("artifacts", [])
        if isinstance(item, dict)
    }
    for name, item in input_by_name.items():
        observed = observed_source_artifacts.get(name)
        if (
            not isinstance(observed, dict)
            or observed.get("sha256") != item.get("sha256")
            or observed.get("bytes") != item.get("bytes")
        ):
            raise AfryShapeDiagnosticError(
                f"diagnostic captured input does not match source catalog: {name}"
            )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise AfryShapeDiagnosticError("diagnostic artifact inventory must be a list")
    by_name = {
        str(item.get("name")): item for item in artifacts if isinstance(item, dict)
    }
    if set(by_name) != EXPECTED_OUTPUT_ARTIFACTS or len(by_name) != len(artifacts):
        raise AfryShapeDiagnosticError("diagnostic artifact inventory is not exact")
    expected_members = EXPECTED_OUTPUT_ARTIFACTS | {
        "diagnostic-manifest.json",
        "diagnostic-manifest.sha256",
    }
    entries = list(bundle.iterdir())
    if {entry.name for entry in entries} != expected_members:
        raise AfryShapeDiagnosticError("diagnostic bundle member closure is not exact")
    for entry in entries:
        entry_stat = entry.stat(follow_symlinks=False)
        if _is_link_or_reparse(entry, entry_stat) or not stat.S_ISREG(entry_stat.st_mode):
            raise AfryShapeDiagnosticError("diagnostic bundle member closure is not plain")
    checksum_bytes, _, _, checksum_identity = _stable_file_bytes(
        checksum_path,
        label="diagnostic manifest checksum",
        maximum_bytes=256,
    )
    expected_sidecar = f"{manifest_sha256}\n".encode("ascii")
    if checksum_bytes != expected_sidecar:
        raise AfryShapeDiagnosticError("diagnostic manifest checksum mismatch")
    artifact_identities: dict[str, tuple[int, int, int, int, int]] = {}
    for name, item in by_name.items():
        path = bundle / name
        maximum_bytes = (
            MAX_PARQUET_ARTIFACT_BYTES if name.endswith(".parquet") else MAX_JSON_ARTIFACT_BYTES
        )
        _, observed_sha256, observed_bytes, observed_identity = _stable_file_bytes(
            path,
            label=f"diagnostic artifact {name}",
            maximum_bytes=maximum_bytes,
        )
        if (
            observed_bytes != item.get("bytes")
            or observed_sha256 != item.get("sha256")
            or (
                item.get("rows") is not None
                and (
                    not isinstance(item.get("rows"), int)
                    or item["rows"] <= 0
                    or item["rows"] > MAX_DIAGNOSTIC_ROWS
                )
            )
        ):
            raise AfryShapeDiagnosticError(f"diagnostic artifact mismatch: {name}")
        artifact_identities[name] = observed_identity
    verified_frames: dict[str, pd.DataFrame] = {}
    for name, schema in (
        ("group_shape_diagnostics.parquet", GROUP_SCHEMA),
        ("weather_shape_sensitivity.parquet", WEATHER_SCHEMA),
        ("term_shape_changes.parquet", TERM_SCHEMA),
    ):
        path = bundle / name
        if not pq.read_schema(path).equals(schema, check_metadata=True):
            raise AfryShapeDiagnosticError(f"diagnostic Parquet schema mismatch: {name}")
        if pq.read_metadata(path).num_rows != by_name[name].get("rows"):
            raise AfryShapeDiagnosticError(f"diagnostic Parquet row count mismatch: {name}")
        table = pq.read_table(path)
        authority_fields = [field for field in required_false if field in schema.names]
        for field in authority_fields:
            if any(value.as_py() is not False for value in table[field]):
                raise AfryShapeDiagnosticError(
                    f"diagnostic artifact authority {field} must remain false"
                )
        frame = table.to_pandas()
        numeric = frame.select_dtypes(include="number")
        if not numeric.empty and not numeric.apply(lambda column: column.map(math.isfinite)).all().all():
            raise AfryShapeDiagnosticError(f"diagnostic numeric values must be finite: {name}")
        verified_frames[name] = frame
        _assert_file_identity(
            path,
            artifact_identities[name],
            label=f"diagnostic artifact {name}",
        )
    group_frame = verified_frames["group_shape_diagnostics.parquet"]
    weather_frame = verified_frames["weather_shape_sensitivity.parquet"]
    term_frame = verified_frames["term_shape_changes.parquet"]
    if group_frame.duplicated(["scenario_vendor", "delivery_year", "weather_year"]).any():
        raise AfryShapeDiagnosticError("diagnostic group keys must be unique")
    if weather_frame.duplicated(["scenario_vendor", "delivery_year"]).any():
        raise AfryShapeDiagnosticError("diagnostic weather keys must be unique")
    if term_frame.duplicated(
        ["scenario_vendor", "weather_year", "from_delivery_year", "to_delivery_year"]
    ).any():
        raise AfryShapeDiagnosticError("diagnostic term keys must be unique")
    share_columns = {
        "group_shape_diagnostics.parquet": [
            "negative_price_share",
            "strict_negative_price_share",
            "near_zero_negative_noise_share",
        ],
        "weather_shape_sensitivity.parquet": [
            "negative_share_min",
            "negative_share_median",
            "negative_share_max",
        ],
    }
    for name, columns in share_columns.items():
        frame = verified_frames[name]
        if any(((frame[column] < 0.0) | (frame[column] > 1.0)).any() for column in columns):
            raise AfryShapeDiagnosticError(f"diagnostic shares are outside [0, 1]: {name}")
    if not (
        (term_frame["delivery_year_gap"] > 0).all()
        and (
            term_frame["to_delivery_year"] - term_frame["from_delivery_year"]
            == term_frame["delivery_year_gap"]
        ).all()
        and (
            (
                term_frame["duck_spread_change_eur_mwh"]
                / term_frame["delivery_year_gap"]
                - term_frame["duck_spread_change_per_year_eur_mwh"]
            ).abs()
            <= 1e-12
        ).all()
    ):
        raise AfryShapeDiagnosticError("diagnostic term-change arithmetic is inconsistent")
    audit_bytes, _, _, audit_identity = _stable_file_bytes(
        bundle / "diagnostic-audit.json",
        label="diagnostic audit",
        maximum_bytes=MAX_JSON_ARTIFACT_BYTES,
    )
    audit = json.loads(audit_bytes)
    if (
        audit.get("status") != DIAGNOSTIC_STATUS
        or audit.get("decision_state") != DECISION_STATE
        or audit.get("empirical_validation_status")
        != "BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS"
    ):
        raise AfryShapeDiagnosticError("diagnostic audit authority or blocker mismatch")
    audit_authorities = audit.get("authorities")
    if (
        not isinstance(audit_authorities, dict)
        or any(audit_authorities.get(field) is not False for field in required_false)
        or audit.get("slot_labels_are_clock_hours") is not False
        or audit.get("external_inputs_consumed") != []
        or audit.get("source_catalog_bundle_id") != source_catalog["bundle_id"]
        or audit.get("group_count")
        != by_name["group_shape_diagnostics.parquet"].get("rows")
        or audit.get("weather_summary_count")
        != by_name["weather_shape_sensitivity.parquet"].get("rows")
        or audit.get("term_change_count") != by_name["term_shape_changes.parquet"].get("rows")
    ):
        raise AfryShapeDiagnosticError("diagnostic audit closure is incomplete")
    _verify_deterministic_replay(
        bundle=bundle,
        source_catalog_bundle=manifest_source_bundle,
        source_catalog=observed_source_catalog,
        source_inputs=input_by_name,
        output_artifacts=by_name,
    )
    producer = manifest.get("producer")
    files = producer.get("files") if isinstance(producer, dict) else None
    if (
        not isinstance(files, list)
        or {item.get("role") for item in files if isinstance(item, dict)}
        != {"diagnostic_module", "diagnostic_entrypoint", "source_catalog_module"}
    ):
        raise AfryShapeDiagnosticError("diagnostic producer provenance is incomplete")
    for item in files:
        path_value = item.get("path") if isinstance(item, dict) else None
        path_posix = PurePosixPath(path_value) if isinstance(path_value, str) else None
        if (
            not isinstance(path_value, str)
            or "\\" in path_value
            or path_posix is None
            or path_posix.is_absolute()
            or ".." in path_posix.parts
            or not isinstance(item.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
            or not isinstance(item.get("bytes"), int)
            or item["bytes"] <= 0
        ):
            raise AfryShapeDiagnosticError("diagnostic producer provenance is invalid")
        producer_path = (workspace / Path(*path_posix.parts)).resolve()
        try:
            producer_path.relative_to(workspace)
        except ValueError as exc:
            raise AfryShapeDiagnosticError("diagnostic producer escaped workspace") from exc
        _, producer_sha256, producer_bytes, _ = _stable_file_bytes(
            producer_path,
            label=f"diagnostic producer {item.get('role')}",
            maximum_bytes=5_000_000,
        )
        if producer_sha256 != item["sha256"] or producer_bytes != item["bytes"]:
            raise AfryShapeDiagnosticError("diagnostic producer does not match workspace")
    runtime = producer.get("runtime") if isinstance(producer, dict) else None
    runtime_executable = Path(sys.executable).resolve()
    try:
        executable_relative = runtime_executable.relative_to(workspace).as_posix()
    except ValueError as exc:
        raise AfryShapeDiagnosticError("diagnostic runtime executable must be repo-local") from exc
    _, executable_sha256, _, _ = _stable_file_bytes(
        runtime_executable,
        label="diagnostic runtime executable",
        maximum_bytes=100_000_000,
    )
    expected_runtime = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "implementation": platform.python_implementation(),
        "cache_tag": str(sys.implementation.cache_tag),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "executable_path": executable_relative,
        "executable_sha256": executable_sha256,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "dependency_closure": runtime_tree_receipt(runtime_executable.parent),
    }
    if (
        runtime != expected_runtime
        or producer.get("source_stable_during_build") is not True
    ):
        raise AfryShapeDiagnosticError("diagnostic producer runtime is incomplete")
    _assert_file_identity(
        bundle / "diagnostic-audit.json",
        audit_identity,
        label="diagnostic audit",
    )
    _assert_file_identity(manifest_path, manifest_identity, label="diagnostic manifest")
    _assert_file_identity(checksum_path, checksum_identity, label="diagnostic checksum")
    if _file_identity(_require_plain_directory(bundle, label="diagnostic bundle")) != bundle_identity:
        raise AfryShapeDiagnosticError("diagnostic bundle changed during verification")
    if (
        _directory_identity(
            _require_plain_directory(build_root, label="diagnostic build root")
        )
        != build_root_identity
    ):
        raise AfryShapeDiagnosticError("diagnostic build root changed during verification")
    if {entry.name for entry in bundle.iterdir()} != expected_members:
        raise AfryShapeDiagnosticError("diagnostic bundle closure changed during verification")
    return manifest


def build_shape_diagnostic_bundle(
    *,
    workspace: Path,
    source_catalog_bundle: Path,
    output_root: Path,
) -> Path:
    """Build one deterministic local diagnostic bundle from a verified AFRY catalog."""

    workspace = _require_canonical_workspace(workspace)
    build_root = _require_plain_workspace_ancestry(
        workspace,
        workspace / "build",
        label="diagnostic build root",
    )
    source_catalog_bundle = _require_plain_workspace_ancestry(
        workspace,
        source_catalog_bundle,
        label="diagnostic source catalog",
    )
    output_root = _require_plain_workspace_ancestry(
        workspace,
        output_root,
        label="diagnostic output root",
    )
    build_root_identity = _directory_identity(
        _require_plain_directory(build_root, label="diagnostic build root")
    )
    for path, label in (
        (source_catalog_bundle, "source catalog"),
        (output_root, "output root"),
    ):
        try:
            path.relative_to(build_root)
        except ValueError as exc:
            raise AfryShapeDiagnosticError(f"{label} must remain below workspace build/") from exc
    if _paths_overlap(source_catalog_bundle, output_root):
        raise AfryShapeDiagnosticError("source catalog and diagnostic output root must not overlap")
    _require_plain_directory(source_catalog_bundle, label="diagnostic source catalog")
    source_catalog_path = source_catalog_bundle / "catalog.json"
    _, source_catalog_sha256, _, source_catalog_identity = _stable_file_bytes(
        source_catalog_path,
        label="diagnostic source catalog manifest",
        maximum_bytes=MAX_MANIFEST_BYTES,
    )
    source_catalog = verify_catalog_bundle(source_catalog_bundle)
    _assert_file_identity(
        source_catalog_path,
        source_catalog_identity,
        label="diagnostic source catalog manifest",
    )
    if (
        source_catalog.get("integrity_verdict") != "PASS_LOCAL_INTEGRITY_ONLY"
        or source_catalog.get("decision_state") != DECISION_STATE
        or source_catalog.get("admission", {}).get("model_input_authorized") is not False
    ):
        raise AfryShapeDiagnosticError("source AFRY catalog is not admitted for local diagnostics")
    source_artifacts = {
        str(item["name"]): item for item in source_catalog.get("artifacts", [])
    }
    if not EXPECTED_INPUT_ARTIFACTS <= set(source_artifacts):
        raise AfryShapeDiagnosticError("source AFRY catalog lacks diagnostic inputs")
    producers = _producer_files(workspace)
    runtime_executable = Path(sys.executable).resolve()
    try:
        runtime_executable_relative = runtime_executable.relative_to(workspace).as_posix()
    except ValueError as exc:
        raise AfryShapeDiagnosticError("diagnostic runtime executable must be repo-local") from exc
    _, runtime_executable_sha256, _, runtime_executable_identity = _stable_file_bytes(
        runtime_executable,
        label="diagnostic runtime executable",
        maximum_bytes=100_000_000,
    )
    runtime_dependency_closure = runtime_tree_receipt(runtime_executable.parent)
    contract_path = (workspace / CONTRACT_RELATIVE_PATH).resolve()
    try:
        contract_path.relative_to(workspace)
    except ValueError as exc:
        raise AfryShapeDiagnosticError("diagnostic contract must remain inside workspace") from exc
    _, contract_sha256, contract_bytes, _ = _stable_file_bytes(
        contract_path,
        label="diagnostic contract",
        maximum_bytes=MAX_JSON_ARTIFACT_BYTES,
    )
    contract_receipt = {
        "path": CONTRACT_RELATIVE_PATH,
        "sha256": contract_sha256,
        "bytes": contract_bytes,
        "stable_during_build": True,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    output_root_identity = _directory_identity(
        _require_plain_directory(output_root, label="diagnostic output root")
    )
    staging = output_root / f".staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        captures: dict[str, Path] = {}
        for name in sorted(EXPECTED_INPUT_ARTIFACTS):
            item = source_artifacts[name]
            destination = staging / f"captured-{name}"
            _capture_artifact(
                source_catalog_bundle / name,
                destination,
                expected_sha256=str(item["sha256"]),
                expected_bytes=int(item["bytes"]),
            )
            captures[name] = destination
        shape = pq.read_table(captures["hour_slot_shape_benchmark.parquet"]).to_pandas()
        profile = json.loads(captures["hourly_profile.json"].read_text(encoding="utf-8"))
        group_frame, weather_frame, term_frame, audit = build_shape_diagnostic_frames(
            shape, profile
        )
        audit["source_catalog_bundle_id"] = source_catalog["bundle_id"]
        group_path = staging / "group_shape_diagnostics.parquet"
        weather_path = staging / "weather_shape_sensitivity.parquet"
        term_path = staging / "term_shape_changes.parquet"
        audit_path = staging / "diagnostic-audit.json"
        _write_frame(group_frame, group_path, GROUP_SCHEMA)
        _write_frame(weather_frame, weather_path, WEATHER_SCHEMA)
        _write_frame(term_frame, term_path, TERM_SCHEMA)
        audit_path.write_bytes(canonical_json_bytes(audit))
        for name, path in captures.items():
            item = source_artifacts[name]
            if file_sha256(path) != item["sha256"]:
                raise AfryShapeDiagnosticError(f"captured diagnostic input changed: {name}")
            path.unlink()
        for item in producers:
            path = workspace / item["path"]
            _, observed_sha256, observed_bytes, _ = _stable_file_bytes(
                path,
                label=f"diagnostic producer {item['role']}",
                maximum_bytes=5_000_000,
            )
            if observed_sha256 != item["sha256"] or observed_bytes != item["bytes"]:
                raise AfryShapeDiagnosticError(
                    f"diagnostic producer changed during build: {item['path']}"
                )
        _, observed_contract_sha256, observed_contract_bytes, _ = _stable_file_bytes(
            contract_path,
            label="diagnostic contract",
            maximum_bytes=MAX_JSON_ARTIFACT_BYTES,
        )
        if observed_contract_sha256 != contract_sha256 or observed_contract_bytes != contract_bytes:
            raise AfryShapeDiagnosticError("diagnostic contract changed during build")
        _assert_file_identity(
            runtime_executable,
            runtime_executable_identity,
            label="diagnostic runtime executable",
        )
        if runtime_tree_receipt(runtime_executable.parent) != runtime_dependency_closure:
            raise AfryShapeDiagnosticError("diagnostic runtime closure changed during build")
        source_catalog_after = verify_catalog_bundle(source_catalog_bundle)
        _assert_file_identity(
            source_catalog_path,
            source_catalog_identity,
            label="diagnostic source catalog manifest",
        )
        if source_catalog_after != source_catalog:
            raise AfryShapeDiagnosticError("diagnostic source catalog changed during build")
        output_artifacts = []
        for path, rows in (
            (group_path, len(group_frame)),
            (weather_path, len(weather_frame)),
            (term_path, len(term_frame)),
            (audit_path, None),
        ):
            output_artifacts.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                    "rows": rows,
                }
            )
        body = {
            "schema_version": DIAGNOSTIC_SCHEMA,
            "status": DIAGNOSTIC_STATUS,
            "decision_state": DECISION_STATE,
            "source_catalog": {
                "bundle_id": source_catalog["bundle_id"],
                "catalog_sha256": source_catalog_sha256,
                "captured_artifacts": [
                    {
                        "name": name,
                        "sha256": source_artifacts[name]["sha256"],
                        "bytes": source_artifacts[name]["bytes"],
                    }
                    for name in sorted(EXPECTED_INPUT_ARTIFACTS)
                ],
            },
            "contract": contract_receipt,
            "configuration": {
                "central_block_slots": list(CENTRAL_BLOCK_SLOTS),
                "late_block_slots": list(LATE_BLOCK_SLOTS),
                "slot_labels_are_clock_hours": False,
                "weather_aggregation": "UNWEIGHTED_MIN_MEDIAN_MAX_NOT_PROBABILITY",
                "term_interpolation": False,
            },
            "artifacts": output_artifacts,
            "producer": {
                "files": producers,
                "runtime": {
                    "python": ".".join(str(value) for value in sys.version_info[:3]),
                    "implementation": platform.python_implementation(),
                    "cache_tag": str(sys.implementation.cache_tag),
                    "platform_system": platform.system(),
                    "platform_machine": platform.machine(),
                    "executable_path": runtime_executable_relative,
                    "executable_sha256": runtime_executable_sha256,
                    "pandas": pd.__version__,
                    "pyarrow": pa.__version__,
                    "dependency_closure": runtime_dependency_closure,
                },
                "source_stable_during_build": True,
            },
            "admission": {
                "restricted_vendor_derived_values": True,
                "calendar_mapping_authorized": False,
                "probability_weighting_authorized": False,
                "monthly_level_authority": False,
                "model_input_authorized": False,
                "production_authorized": False,
                "empirical_validation_status": "BLOCKED_PENDING_GOVERNED_EEX_ENTSOE_DATABRICKS",
            },
        }
        bundle_id = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
        manifest = {**body, "bundle_id": bundle_id}
        manifest_path = staging / "diagnostic-manifest.json"
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        (staging / "diagnostic-manifest.sha256").write_bytes(
            f"{file_sha256(manifest_path)}\n".encode("ascii")
        )
        if (
            _directory_identity(
                _require_plain_directory(build_root, label="diagnostic build root")
            )
            != build_root_identity
            or _directory_identity(
                _require_plain_directory(output_root, label="diagnostic output root")
            )
            != output_root_identity
        ):
            raise AfryShapeDiagnosticError("diagnostic publication ancestry changed during build")
        final = output_root / bundle_id
        for publish_attempt in range(3):
            if final.exists():
                try:
                    existing = verify_shape_diagnostic_bundle(
                        final,
                        workspace=workspace,
                        source_catalog_bundle=source_catalog_bundle,
                    )
                except Exception:
                    _quarantine_invalid_publication(final)
                else:
                    if existing != manifest:
                        raise AfryShapeDiagnosticError("diagnostic bundle ID collision")
                    shutil.rmtree(staging)
                    return final
            try:
                _publish_directory(staging, final)
                break
            except (FileExistsError, PermissionError):
                if not final.exists() or publish_attempt == 2:
                    raise
        else:  # pragma: no cover - defensive loop exhaustiveness
            raise AfryShapeDiagnosticError("diagnostic publication race did not converge")
        try:
            verify_shape_diagnostic_bundle(
                final,
                workspace=workspace,
                source_catalog_bundle=source_catalog_bundle,
            )
        except Exception as exc:
            try:
                quarantine = _quarantine_invalid_publication(final)
            except Exception as quarantine_exc:
                raise AfryShapeDiagnosticError(
                    "diagnostic post-publication verification failed and final slot "
                    "could not be quarantined"
                ) from quarantine_exc
            raise AfryShapeDiagnosticError(
                f"diagnostic post-publication verification failed; quarantined as {quarantine.name}"
            ) from exc
        return final
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "AfryShapeDiagnosticError",
    "CENTRAL_BLOCK_SLOTS",
    "DECISION_STATE",
    "DIAGNOSTIC_SCHEMA",
    "DIAGNOSTIC_STATUS",
    "GROUP_SCHEMA",
    "LATE_BLOCK_SLOTS",
    "TERM_SCHEMA",
    "WEATHER_SCHEMA",
    "build_shape_diagnostic_bundle",
    "build_shape_diagnostic_frames",
    "verify_shape_diagnostic_bundle",
]
