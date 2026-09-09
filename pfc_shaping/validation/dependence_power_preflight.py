"""Dependence-aware power preflight for paired rolling-origin LT losses.

This module quantifies how weak a retrospective pilot is.  It deliberately
cannot authorize a candidate, consume a holdout, or freeze a statistical
design.  Its output is a post-hoc diagnostic for planning the next independent
CH point-in-time evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from statistics import NormalDist
from zoneinfo import ZoneInfo

from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.pipeline.strict_structured_data import load_strict_json

SCHEMA_VERSION = "ch_lt_dependence_power_preflight.v2"
SUPPORTED_SOURCE_SCHEMA = "intraday_shape_estimator_rolling_origin.v2"
SUPPORTED_SOURCE_TARGET = "quarter_hour_price_given_realized_parent_hour_mean"
STATUS = "LOCAL_DE_LU_INTRAHOUR_PLUGIN_SENSITIVITY_NO_GO"
PRIMARY_METRIC = "mae_eur_mwh"
FULL_CH_LT_TARGET = "DELIVERED_CH_LT_QUARTER_HOURLY_PRICE"
TARGET_RELATIVE_IMPROVEMENT = 0.02
FAMILYWISE_ALPHA = 0.05
TARGET_POWER = 0.80
HYPOTHESIS_FAMILY_SIZES = (1, 4, 8)
MAX_HAC_LAG = 12
COMPARISON_TOLERANCE = 1e-12
MAX_CSV_BYTES = 64 * 1024 * 1024
MAX_SUMMARY_BYTES = 4 * 1024 * 1024
MAX_ABSOLUTE_LOSS_EUR_MWH = 1_000_000.0

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_COLUMNS = {
    "origin_utc",
    "test_end_utc",
    "model",
    "n_quarter_hours",
    PRIMARY_METRIC,
}
_SEASON_BY_MONTH = {
    12: "WINTER",
    1: "WINTER",
    2: "WINTER",
    3: "SPRING",
    4: "SPRING",
    5: "SPRING",
    6: "SUMMER",
    7: "SUMMER",
    8: "SUMMER",
    9: "AUTUMN",
    10: "AUTUMN",
    11: "AUTUMN",
}
_ALL_SEASONS = ("WINTER", "SPRING", "SUMMER", "AUTUMN")


class DependencePowerPreflightError(ValueError):
    """Raised when pilot evidence is malformed, unbound, or overstated."""


def audit_dependence_power_pilot(
    *,
    fold_csv_path: str | Path,
    expected_fold_csv_sha256: str,
    source_summary_path: str | Path,
    expected_source_summary_sha256: str,
    candidate_model: str,
    benchmark_model: str,
) -> dict[str, object]:
    """Audit exact paired-origin losses and return a non-authorizing power pilot."""

    _require_sha256(expected_fold_csv_sha256, label="fold CSV")
    _require_sha256(expected_source_summary_sha256, label="source summary")
    _require_model_name(candidate_model, label="candidate model")
    _require_model_name(benchmark_model, label="benchmark model")
    if candidate_model == benchmark_model:
        raise DependencePowerPreflightError("candidate and benchmark must differ")

    fold_path = _lexical_absolute(fold_csv_path)
    summary_path = _lexical_absolute(source_summary_path)
    fold_payload = read_stable_single_link_file(
        fold_path,
        label="paired-origin fold CSV",
        max_bytes=MAX_CSV_BYTES,
    )
    summary_payload = read_stable_single_link_file(
        summary_path,
        label="paired-origin source summary",
        max_bytes=MAX_SUMMARY_BYTES,
    )
    fold_sha256 = hashlib.sha256(fold_payload).hexdigest()
    summary_sha256 = hashlib.sha256(summary_payload).hexdigest()
    if fold_sha256 != expected_fold_csv_sha256:
        raise DependencePowerPreflightError("fold CSV SHA-256 mismatch")
    if summary_sha256 != expected_source_summary_sha256:
        raise DependencePowerPreflightError("source summary SHA-256 mismatch")

    summary = _load_summary(summary_payload)
    source_design = _source_design(summary)
    rows = _load_fold_rows(
        fold_payload,
        candidate_model=candidate_model,
        benchmark_model=benchmark_model,
        expected_fold_days=int(source_design["fold_days"]),
    )
    _verify_source_summary(
        summary,
        rows=rows,
        fold_sha256=fold_sha256,
        candidate_model=candidate_model,
        benchmark_model=benchmark_model,
    )

    origins = tuple(sorted(rows))
    candidate_losses = tuple(rows[origin][candidate_model][1] for origin in origins)
    benchmark_losses = tuple(rows[origin][benchmark_model][1] for origin in origins)
    deltas = tuple(
        benchmark - candidate for benchmark, candidate in zip(benchmark_losses, candidate_losses)
    )
    benchmark_mean = _mean(benchmark_losses)
    candidate_mean = _mean(candidate_losses)
    target_delta = benchmark_mean * TARGET_RELATIVE_IMPROVEMENT
    lag_grid = tuple(range(0, min(MAX_HAC_LAG, len(deltas) // 4) + 1))
    hac = _hac_grid(deltas, lag_grid=lag_grid)
    conservative_lrv = max(item["long_run_variance"] for item in hac)
    sample_variance = _sample_variance(deltas)
    if sample_variance == 0.0 or conservative_lrv == 0.0:
        raise DependencePowerPreflightError(
            "paired loss variance is degenerate and cannot support power sensitivity"
        )
    effective_n = _effective_sample_size(
        n=len(deltas),
        sample_variance=sample_variance,
        conservative_lrv=conservative_lrv,
    )
    power_grid = _required_origin_grid(
        conservative_lrv=conservative_lrv,
        target_delta=target_delta,
    )
    leave_one_out = _leave_one_out_required_origin_ranges(
        deltas=deltas,
        benchmark_losses=benchmark_losses,
        lag_grid=lag_grid,
    )
    point_estimate_required = max(item["required_origins"] for item in power_grid)
    delete_one_plugin_max = max(
        point_estimate_required,
        max(item["maximum_required_origins"] for item in leave_one_out),
    )
    observed_relative_improvement = _mean(deltas) / benchmark_mean

    window_seasons = [
        season
        for origin in origins
        for season in _window_seasons(origin, rows[origin][candidate_model][0])
    ]
    season_counts = {season: window_seasons.count(season) for season in _ALL_SEASONS}
    missing_seasons = [season for season, count in season_counts.items() if count == 0]
    wins = sum(delta > COMPARISON_TOLERANCE for delta in deltas)
    losses = sum(delta < -COMPARISON_TOLERANCE for delta in deltas)
    ties = len(deltas) - wins - losses

    blockers = [
        "POST_HOC_SELECTED_CANDIDATE_PILOT",
        "MIXED_AUTHORITY_DE_PROXY_NOT_DIRECT_CH_TRUTH",
        "PRE_REGISTERED_HYPOTHESIS_FAMILY_NOT_FROZEN",
        "BOOTSTRAP_BLOCK_SELECTION_NOT_FROZEN",
        "POWER_MDE_UNCERTAINTY_NOT_INDEPENDENTLY_VALIDATED",
        "TARGET_NOT_FULL_CH_LT",
        "REQUIRED_N_UNSUPPORTED_FOR_CH_ACQUISITION",
        "PILOT_MUST_NOT_BE_REUSED_FOR_CONFIRMATORY_TEST",
    ]
    if missing_seasons:
        blockers.append("ALL_FOUR_SEASONS_NOT_COVERED")
    if observed_relative_improvement + COMPARISON_TOLERANCE < TARGET_RELATIVE_IMPROVEMENT:
        blockers.append("OBSERVED_IMPROVEMENT_BELOW_TWO_PERCENT_TARGET")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "scientific_admission": False,
        "execution_authorized": False,
        "production_authorization": False,
        "promotion_gate": False,
        "future_holdout_consumed": False,
        "t057_consumed": False,
        "source": {
            "fold_csv_path": str(fold_path),
            "fold_csv_sha256": fold_sha256,
            "source_summary_path": str(summary_path),
            "source_summary_sha256": summary_sha256,
            "source_schema_version": summary["schema_version"],
            "source_status": summary["status"],
            "source_authority_status": _mapping(summary["input"], label="input")[
                "authority_status"
            ],
            "source_target": source_design["target"],
            "source_fold_days": source_design["fold_days"],
            "target_is_full_ch_lt": source_design["target"] == FULL_CH_LT_TARGET,
            "ompex_used": False,
        },
        "comparison": {
            "candidate_model": candidate_model,
            "benchmark_model": benchmark_model,
            "primary_metric": PRIMARY_METRIC,
            "aggregation_order": "EQUAL_WEIGHT_PER_NON_OVERLAPPING_ORIGIN",
            "origin_count": len(origins),
            "origin_start_utc": origins[0].isoformat(),
            "origin_end_utc": origins[-1].isoformat(),
            "candidate_mean_loss": candidate_mean,
            "benchmark_mean_loss": benchmark_mean,
            "mean_loss_improvement": _mean(deltas),
            "observed_relative_improvement": observed_relative_improvement,
            "observed_relative_improvement_meets_target": (
                observed_relative_improvement + COMPARISON_TOLERANCE >= TARGET_RELATIVE_IMPROVEMENT
            ),
            "wins": wins,
            "losses": losses,
            "ties": ties,
        },
        "dependence": {
            "paired_loss_difference_definition": "BENCHMARK_MINUS_CANDIDATE",
            "sample_variance": sample_variance,
            "hac_kernel": "BARTLETT",
            "hac_lag_grid": list(lag_grid),
            "hac_long_run_variance_grid": hac,
            "observed_plugin_long_run_variance_max_over_lag_grid": conservative_lrv,
            "observed_plugin_effective_origin_count": effective_n,
            "observed_plugin_effective_origin_fraction": effective_n / len(deltas),
            "is_upper_confidence_bound": False,
        },
        "power_design": {
            "target_relative_improvement": TARGET_RELATIVE_IMPROVEMENT,
            "target_absolute_loss_improvement": target_delta,
            "familywise_alpha": FAMILYWISE_ALPHA,
            "target_power": TARGET_POWER,
            "multiplicity_bound": "BONFERRONI_WORST_CASE_FOR_HOLM_FAMILY",
            "hypothesis_family_sizes": list(HYPOTHESIS_FAMILY_SIZES),
            "observed_plugin_required_origin_grid": power_grid,
            "observed_plugin_required_origins_max": point_estimate_required,
            "delete_one_plugin_required_origin_ranges": leave_one_out,
            "delete_one_plugin_required_origins_max": delete_one_plugin_max,
            "direct_sequential_collection_years_if_misused": (
                delete_one_plugin_max * int(source_design["fold_days"]) / 365.2425
            ),
            "ch_acquisition_requirement_supported": False,
            "ch_acquisition_requirement_origins": None,
            "pilot_reuse_for_confirmatory_test_forbidden": True,
            "interpretation": (
                "EXPLORATORY_POST_SELECTION_DE_LU_PLUGIN_SENSITIVITY_NOT_CH_ACQUISITION_PLAN"
            ),
        },
        "coverage": {
            "window_season_touch_counts": season_counts,
            "missing_seasons": missing_seasons,
            "all_test_windows_non_overlapping": True,
        },
        "blockers": blockers,
        "limitations": [
            "The candidate was selected before this power calculation.",
            "The source is a local mixed-authority DE-LU proxy, not direct governed CH truth.",
            "The target uses the realized parent-hour mean and is not a full CH LT estimand.",
            "Long-run variance, ESS and required-n values are post-selection plug-in sensitivities, not confidence bounds.",
            "The delete-one maximum must not be used as a CH acquisition target.",
            "These pilot origins are forbidden for reuse in a confirmatory evaluation.",
        ],
    }


def _load_summary(payload: bytes) -> Mapping[str, object]:
    try:
        parsed = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise DependencePowerPreflightError("source summary is invalid JSON") from exc
    return _mapping(parsed, label="source summary")


def _source_design(summary: Mapping[str, object]) -> dict[str, object]:
    design = _mapping(summary.get("design"), label="source design")
    fold_days = _strict_int(design.get("fold_days"), label="source fold days")
    if fold_days != 14:
        raise DependencePowerPreflightError("source fold duration must be exactly 14 days")
    target = design.get("target")
    if target != SUPPORTED_SOURCE_TARGET:
        raise DependencePowerPreflightError("source target is unsupported or relabelled")
    return {"fold_days": fold_days, "target": target}


def _load_fold_rows(
    payload: bytes,
    *,
    candidate_model: str,
    benchmark_model: str,
    expected_fold_days: int,
) -> dict[datetime, dict[str, tuple[datetime, float, int]]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DependencePowerPreflightError("fold CSV is not UTF-8") from exc
    if "\x00" in text:
        raise DependencePowerPreflightError("fold CSV contains NUL bytes")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    headers = reader.fieldnames
    if headers is None or len(headers) != len(set(headers)):
        raise DependencePowerPreflightError("fold CSV header is absent or duplicated")
    missing = sorted(_REQUIRED_COLUMNS - set(headers))
    if missing:
        raise DependencePowerPreflightError("fold CSV is missing columns: " + ",".join(missing))

    expected_models = {candidate_model, benchmark_model}
    rows: dict[datetime, dict[str, tuple[datetime, float, int]]] = {}
    for line_number, raw in enumerate(reader, start=2):
        if None in raw or any(value is None for value in raw.values()):
            raise DependencePowerPreflightError(
                f"fold CSV row {line_number} has inconsistent fields"
            )
        model = raw["model"]
        if model not in expected_models:
            raise DependencePowerPreflightError(
                f"fold CSV row {line_number} has an unexpected model"
            )
        origin = _utc_datetime(raw["origin_utc"], label=f"row {line_number} origin")
        test_end = _utc_datetime(raw["test_end_utc"], label=f"row {line_number} test end")
        if test_end <= origin:
            raise DependencePowerPreflightError(
                f"fold CSV row {line_number} has a non-positive test window"
            )
        if not _quarter_hour_aligned(origin) or not _quarter_hour_aligned(test_end):
            raise DependencePowerPreflightError(
                f"fold CSV row {line_number} is not aligned to 15 minutes"
            )
        duration = test_end - origin
        if duration != timedelta(days=expected_fold_days):
            raise DependencePowerPreflightError(
                f"fold CSV row {line_number} does not match source fold duration"
            )
        loss = _finite_float(raw[PRIMARY_METRIC], label=f"row {line_number} loss")
        if loss < 0.0:
            raise DependencePowerPreflightError(f"fold CSV row {line_number} has a negative loss")
        quarter_hours = _strict_positive_integer(
            raw["n_quarter_hours"], label=f"row {line_number} quarter-hours"
        )
        expected_quarter_hours = int(duration.total_seconds() // (15 * 60))
        if quarter_hours != expected_quarter_hours:
            raise DependencePowerPreflightError(
                f"fold CSV row {line_number} observation count does not match its window"
            )
        by_model = rows.setdefault(origin, {})
        if model in by_model:
            raise DependencePowerPreflightError(
                f"fold CSV duplicates origin/model at row {line_number}"
            )
        by_model[model] = (test_end, loss, quarter_hours)
    if len(rows) < 8:
        raise DependencePowerPreflightError(
            "at least eight distinct origins are required for a diagnostic pilot"
        )

    previous_end: datetime | None = None
    for origin in sorted(rows):
        by_model = rows[origin]
        if set(by_model) != expected_models:
            raise DependencePowerPreflightError(
                "every origin must contain exactly candidate and benchmark"
            )
        ends = {value[0] for value in by_model.values()}
        quarter_hour_counts = {value[2] for value in by_model.values()}
        if len(ends) != 1 or len(quarter_hour_counts) != 1:
            raise DependencePowerPreflightError(
                "paired rows must share test window and observation count"
            )
        if previous_end is not None and origin < previous_end:
            raise DependencePowerPreflightError(
                "rolling-origin test windows overlap and are not independent units"
            )
        previous_end = next(iter(ends))
    return rows


def _verify_source_summary(
    summary: Mapping[str, object],
    *,
    rows: Mapping[datetime, Mapping[str, tuple[datetime, float, int]]],
    fold_sha256: str,
    candidate_model: str,
    benchmark_model: str,
) -> None:
    if summary.get("schema_version") != SUPPORTED_SOURCE_SCHEMA:
        raise DependencePowerPreflightError("source summary schema is unsupported")
    if summary.get("production_authorization") is not False:
        raise DependencePowerPreflightError("source summary authorizes production")
    if summary.get("promotion_gate") is not False:
        raise DependencePowerPreflightError("source summary authorizes promotion")
    if summary.get("ompex_used") is not False:
        raise DependencePowerPreflightError("OMPEX-contaminated source is forbidden")
    status = summary.get("status")
    if not isinstance(status, str) or not status.endswith("_NOT_PRODUCTION"):
        raise DependencePowerPreflightError("source summary status is not NO-GO")
    artifacts = _mapping(summary.get("artifacts"), label="artifacts")
    if artifacts.get("fold_metrics_csv_sha256") != fold_sha256:
        raise DependencePowerPreflightError("source summary does not bind fold CSV")
    if _strict_int(summary.get("fold_count"), label="fold count") != len(rows):
        raise DependencePowerPreflightError("source summary fold count mismatch")
    source_input = _mapping(summary.get("input"), label="input")
    authority_status = source_input.get("authority_status")
    if not isinstance(authority_status, str) or "NO_GO" not in authority_status:
        raise DependencePowerPreflightError("source authority is overstated")
    metrics = _mapping(summary.get("metrics"), label="metrics")
    for model in (candidate_model, benchmark_model):
        declared = _mapping(metrics.get(model), label=f"{model} metrics")
        declared_loss = _finite_number(declared.get(PRIMARY_METRIC), label=f"{model} declared loss")
        weighted_loss = _weighted_loss(rows, model=model)
        if not math.isclose(
            declared_loss,
            weighted_loss,
            rel_tol=0.0,
            abs_tol=COMPARISON_TOLERANCE,
        ):
            raise DependencePowerPreflightError(f"source summary {model} loss does not reconstruct")


def _weighted_loss(
    rows: Mapping[datetime, Mapping[str, tuple[datetime, float, int]]],
    *,
    model: str,
) -> float:
    total_weight = sum(by_model[model][2] for by_model in rows.values())
    weighted_sum = math.fsum(by_model[model][1] * by_model[model][2] for by_model in rows.values())
    result = weighted_sum / total_weight
    if not math.isfinite(result):
        raise DependencePowerPreflightError("weighted source loss is not finite")
    return result


def _hac_grid(values: Sequence[float], *, lag_grid: Sequence[int]) -> list[dict[str, object]]:
    mean = _mean(values)
    centered = [value - mean for value in values]
    n = len(centered)
    gamma_zero = sum(value * value for value in centered) / n
    result: list[dict[str, object]] = []
    for lag in lag_grid:
        long_run_variance = gamma_zero
        for offset in range(1, lag + 1):
            covariance = (
                sum(centered[index] * centered[index - offset] for index in range(offset, n)) / n
            )
            weight = 1.0 - offset / (lag + 1.0)
            long_run_variance += 2.0 * weight * covariance
        result.append(
            {
                "lag_origins": lag,
                "long_run_variance": max(0.0, long_run_variance),
            }
        )
    return result


def _required_origin_grid(
    *, conservative_lrv: float, target_delta: float
) -> list[dict[str, object]]:
    if target_delta <= 0.0:
        raise DependencePowerPreflightError("power target delta must be positive")
    normal = NormalDist()
    z_power = normal.inv_cdf(TARGET_POWER)
    result: list[dict[str, object]] = []
    for family_size in HYPOTHESIS_FAMILY_SIZES:
        alpha_per_comparison = FAMILYWISE_ALPHA / family_size
        z_alpha = normal.inv_cdf(1.0 - alpha_per_comparison / 2.0)
        required = max(
            2,
            math.ceil(((z_alpha + z_power) ** 2 * conservative_lrv) / (target_delta**2)),
        )
        result.append(
            {
                "hypothesis_family_size": family_size,
                "alpha_per_comparison": alpha_per_comparison,
                "required_origins": required,
            }
        )
    return result


def _leave_one_out_required_origin_ranges(
    *,
    deltas: Sequence[float],
    benchmark_losses: Sequence[float],
    lag_grid: Sequence[int],
) -> list[dict[str, object]]:
    by_family: dict[int, list[int]] = {family_size: [] for family_size in HYPOTHESIS_FAMILY_SIZES}
    for omitted in range(len(deltas)):
        subset_deltas = tuple(value for index, value in enumerate(deltas) if index != omitted)
        subset_benchmark = tuple(
            value for index, value in enumerate(benchmark_losses) if index != omitted
        )
        subset_lags = tuple(lag for lag in lag_grid if lag <= len(subset_deltas) // 4)
        hac = _hac_grid(subset_deltas, lag_grid=subset_lags)
        lrv = max(item["long_run_variance"] for item in hac)
        grid = _required_origin_grid(
            conservative_lrv=float(lrv),
            target_delta=_mean(subset_benchmark) * TARGET_RELATIVE_IMPROVEMENT,
        )
        for item in grid:
            by_family[int(item["hypothesis_family_size"])].append(int(item["required_origins"]))
    return [
        {
            "hypothesis_family_size": family_size,
            "minimum_required_origins": min(values),
            "maximum_required_origins": max(values),
        }
        for family_size, values in by_family.items()
    ]


def _effective_sample_size(*, n: int, sample_variance: float, conservative_lrv: float) -> float:
    if sample_variance == 0.0:
        return 1.0
    gamma_zero = sample_variance * (n - 1) / n
    if conservative_lrv <= 0.0:
        return float(n)
    return max(1.0, min(float(n), n * gamma_zero / conservative_lrv))


def _season(origin: datetime) -> str:
    local = origin.astimezone(ZoneInfo("Europe/Zurich"))
    return _SEASON_BY_MONTH[local.month]


def _window_seasons(start: datetime, end: datetime) -> tuple[str, ...]:
    touched: set[str] = set()
    current = start
    while current < end:
        touched.add(_season(current))
        current += timedelta(days=1)
    touched.add(_season(end - timedelta(microseconds=1)))
    return tuple(season for season in _ALL_SEASONS if season in touched)


def _quarter_hour_aligned(value: datetime) -> bool:
    return value.second == 0 and value.microsecond == 0 and value.minute % 15 == 0


def _utc_datetime(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise DependencePowerPreflightError(f"{label} must be a timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DependencePowerPreflightError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DependencePowerPreflightError(f"{label} must be timezone-aware")
    if parsed.utcoffset().total_seconds() != 0:
        raise DependencePowerPreflightError(f"{label} must be UTC")
    return parsed


def _strict_positive_integer(value: object, *, label: str) -> int:
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise DependencePowerPreflightError(f"{label} must be an exact integer")
    parsed = int(value)
    if parsed <= 0:
        raise DependencePowerPreflightError(f"{label} must be positive")
    return parsed


def _strict_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DependencePowerPreflightError(f"{label} must be an integer")
    return value


def _finite_float(value: object, *, label: str) -> float:
    if not isinstance(value, str):
        raise DependencePowerPreflightError(f"{label} must be numeric text")
    try:
        parsed = float(value)
    except (OverflowError, ValueError) as exc:
        raise DependencePowerPreflightError(f"{label} is invalid") from exc
    if not math.isfinite(parsed) or abs(parsed) > MAX_ABSOLUTE_LOSS_EUR_MWH:
        raise DependencePowerPreflightError(f"{label} must be finite and bounded")
    return parsed


def _finite_number(value: object, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DependencePowerPreflightError(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (OverflowError, ValueError) as exc:
        raise DependencePowerPreflightError(f"{label} is invalid") from exc
    if not math.isfinite(parsed) or abs(parsed) > MAX_ABSOLUTE_LOSS_EUR_MWH:
        raise DependencePowerPreflightError(f"{label} must be finite and bounded")
    return parsed


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise DependencePowerPreflightError(f"{label} must be an object")
    return value


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DependencePowerPreflightError(f"{label} SHA-256 is invalid")


def _require_model_name(value: object, *, label: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 160:
        raise DependencePowerPreflightError(f"{label} is invalid")


def _lexical_absolute(path: str | Path) -> Path:
    selected = Path(path).expanduser()
    if not selected.is_absolute():
        selected = Path(os.path.abspath(selected))
    return selected


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _sample_variance(values: Sequence[float]) -> float:
    mean = _mean(values)
    return math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)


__all__ = [
    "DependencePowerPreflightError",
    "SCHEMA_VERSION",
    "STATUS",
    "audit_dependence_power_pilot",
]
