"""Fail-closed scoring primitives for sealed CH LT hourly predictions."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json
from pfc_shaping.validation import (
    ch_lt_local_future_origin_selection as origin_selection_auditor,
)
from pfc_shaping.validation.ch_lt_prospective_capture_ledger import (
    build_prospective_capture_ledger,
)

_LOCAL_TIMEZONE = "Europe/Zurich"
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_PFC_BYTES = 64 * 1024 * 1024
_SCENARIOS = ("slow", "central", "fast")
_FLOAT_MATCH_ABS_TOLERANCE = 1e-10
_TRUTH_COMMON_KEYS = {
    "schema_version",
    "status",
    "test_fixture",
    "market",
    "target_id",
    "delivery_month",
    "delivery_end_utc",
    "native_hourly_cadence",
    "quarter_hour_proxy",
    "native_hourly_interval_count",
    "truth_opened_at_utc_untrusted",
    "observations",
    "outcomes_opened",
    "truth_publication_receipt_present",
    "prospective_truth_consumed",
    "future_holdout_consumed",
    "t057_consumed",
    "training_eligible",
    "model_selection_eligible",
    "scientific_admission",
    "production_authorization",
    "promotion_gate",
}
_TRUTH_REAL_EXTRA_KEYS = {
    "truth_bundle_id",
    "source_ledger",
    "prediction_commitment",
    "future_origin_selection",
}
_TRUTH_PUBLICATION_RECEIPT_KEYS = {
    "schema_version",
    "receipt_id",
    "status",
    "test_fixture",
    "target_id",
    "delivery_month",
    "delivery_end_utc",
    "truth_opened_at_utc_untrusted",
    "local_diagnostic_truth_read_occurred",
    "truth_bundle",
    "prediction_commitment",
    "hourly_scoring_contract",
    "future_origin_selection",
    "authority_boundary",
    "future_holdout_consumed",
    "t057_consumed",
    "scientific_admission",
    "production_authorization",
    "promotion_gate",
}
_TRUTH_OPEN_AUTHORITY_KEYS = {
    "independent_trusted_time_receipt_present",
    "independent_signature_present",
    "builder_inaccessible_external_cas_receipt_present",
    "independent_registry_receipt_present",
    "scientific_truth_open_authorized",
    "production_authorization",
    "promotion_gate",
}


class ProspectiveHourlyScoringError(ValueError):
    """Raised when prospective scoring evidence is incomplete or rebound."""


def _local_wall_clock_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProspectiveHourlyScoringError(f"{label} must be an object")
    return value


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _lower_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProspectiveHourlyScoringError(f"{label} is not lowercase SHA-256")
    return value


def _read_bound_bytes(
    root: Path,
    path: Path,
    expected_sha256: str,
    *,
    label: str,
    max_bytes: int,
) -> tuple[Path, bytes]:
    expected_sha256 = _lower_sha256(expected_sha256, label=f"{label} SHA-256")
    selected = path if path.is_absolute() else root / path
    selected = Path(os.path.abspath(selected))
    if not _inside(selected, root):
        raise ProspectiveHourlyScoringError(f"{label} escapes the repository")
    payload = read_stable_single_link_file(selected, label=label, max_bytes=max_bytes)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ProspectiveHourlyScoringError(f"{label} SHA-256 mismatch")
    return selected, payload


def _read_bound_json(
    root: Path,
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[Path, Mapping[str, object]]:
    selected, payload = _read_bound_bytes(
        root,
        path,
        expected_sha256,
        label=label,
        max_bytes=_MAX_JSON_BYTES,
    )
    try:
        parsed = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProspectiveHourlyScoringError(f"{label} is not strict UTF-8 JSON") from exc
    return selected, _mapping(parsed, label=label)


def _utc_timestamp(value: object, *, label: str) -> pd.Timestamp:
    try:
        selected = pd.Timestamp(str(value))
    except (TypeError, ValueError) as exc:
        raise ProspectiveHourlyScoringError(f"{label} is not a timestamp") from exc
    if selected.tzinfo is None:
        raise ProspectiveHourlyScoringError(f"{label} must be timezone-aware")
    return selected.tz_convert("UTC")


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ProspectiveHourlyScoringError(f"{label} is not numeric")
    try:
        selected = float(value)
    except (TypeError, ValueError) as exc:
        raise ProspectiveHourlyScoringError(f"{label} is not numeric") from exc
    if not math.isfinite(selected):
        raise ProspectiveHourlyScoringError(f"{label} is not finite")
    return selected


def _finite_text(value: float, *, label: str) -> str:
    if not math.isfinite(value):
        raise ProspectiveHourlyScoringError(f"{label} is not finite")
    return format(value, ".17g")


def _exact_hourly_series(
    series: pd.Series,
    *,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    expected_count: int,
    label: str,
) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex) or series.index.tz is None:
        raise ProspectiveHourlyScoringError(f"{label} index is not timezone-aware")
    selected = series.copy()
    selected.index = selected.index.tz_convert("UTC")
    selected = selected.sort_index()
    expected = pd.date_range(start_utc, end_utc, freq="h", inclusive="left")
    if len(expected) != expected_count:
        raise ProspectiveHourlyScoringError(f"{label} expected-hour contract is inconsistent")
    if not selected.index.equals(expected):
        raise ProspectiveHourlyScoringError(
            f"{label} does not exactly cover the committed UTC hours"
        )
    values = pd.to_numeric(selected, errors="coerce").astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ProspectiveHourlyScoringError(f"{label} contains non-finite prices")
    return values


def aggregate_exact_quarter_hours(
    series: pd.Series,
    *,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    expected_hour_count: int,
    label: str,
) -> pd.Series:
    """Aggregate four exact UTC quarter-hours to each committed native hour."""

    if not isinstance(series.index, pd.DatetimeIndex) or series.index.tz is None:
        raise ProspectiveHourlyScoringError(f"{label} index is not timezone-aware")
    selected = series.copy()
    selected.index = selected.index.tz_convert("UTC")
    selected = selected.sort_index()
    expected_quarters = pd.date_range(start_utc, end_utc, freq="15min", inclusive="left")
    if len(expected_quarters) != expected_hour_count * 4:
        raise ProspectiveHourlyScoringError(
            f"{label} expected-quarter-hour contract is inconsistent"
        )
    if not selected.index.equals(expected_quarters):
        raise ProspectiveHourlyScoringError(
            f"{label} does not exactly cover the committed UTC quarter-hours"
        )
    values = pd.to_numeric(selected, errors="coerce").astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ProspectiveHourlyScoringError(f"{label} contains non-finite prices")
    hour_key = values.index.floor("h")
    counts = values.groupby(hour_key, sort=True).count()
    hourly = values.groupby(hour_key, sort=True).mean()
    if len(hourly) != expected_hour_count or not (counts == 4).all():
        raise ProspectiveHourlyScoringError(f"{label} cannot form exact UTC hours")
    return _exact_hourly_series(
        hourly,
        start_utc=start_utc,
        end_utc=end_utc,
        expected_count=expected_hour_count,
        label=f"{label} hourly aggregation",
    )


def hourly_shape_functionals(hourly: pd.Series, *, label: str) -> dict[str, object]:
    """Recompute the functionals sealed by the structural prediction contract."""

    local_hours = hourly.index.tz_convert(_LOCAL_TIMEZONE).hour
    midday = hourly[(local_hours >= 11) & (local_hours < 15)]
    evening = hourly[(local_hours >= 18) & (local_hours < 22)]
    overnight = hourly[(local_hours >= 0) & (local_hours < 6)]
    ramps = hourly.diff().dropna().abs()
    if midday.empty or evening.empty or overnight.empty or ramps.empty:
        raise ProspectiveHourlyScoringError(f"{label} has an empty functional window")
    midday_mean = float(midday.mean())
    evening_mean = float(evening.mean())
    overnight_mean = float(overnight.mean())
    hourly_mean = float(hourly.mean())
    negative_count = int((hourly < 0.0).sum())
    return {
        "hourly_interval_count": len(hourly),
        "hourly_mean_eur_mwh": _finite_text(hourly_mean, label=f"{label} mean"),
        "midday_hour_count": len(midday),
        "midday_mean_eur_mwh": _finite_text(midday_mean, label=f"{label} midday"),
        "midday_premium_to_monthly_mean_eur_mwh": _finite_text(
            midday_mean - hourly_mean, label=f"{label} midday premium"
        ),
        "evening_hour_count": len(evening),
        "evening_mean_eur_mwh": _finite_text(evening_mean, label=f"{label} evening"),
        "evening_premium_to_monthly_mean_eur_mwh": _finite_text(
            evening_mean - hourly_mean, label=f"{label} evening premium"
        ),
        "evening_minus_midday_eur_mwh": _finite_text(
            evening_mean - midday_mean,
            label=f"{label} evening minus midday",
        ),
        "overnight_hour_count": len(overnight),
        "overnight_mean_eur_mwh": _finite_text(
            overnight_mean, label=f"{label} overnight"
        ),
        "overnight_premium_to_monthly_mean_eur_mwh": _finite_text(
            overnight_mean - hourly_mean, label=f"{label} overnight premium"
        ),
        "hourly_standard_deviation_eur_mwh_population": _finite_text(
            float(hourly.std(ddof=0)), label=f"{label} standard deviation"
        ),
        "hourly_p05_eur_mwh_linear": _finite_text(
            float(hourly.quantile(0.05, interpolation="linear")), label=f"{label} p05"
        ),
        "hourly_p50_eur_mwh_linear": _finite_text(
            float(hourly.quantile(0.50, interpolation="linear")), label=f"{label} p50"
        ),
        "hourly_p95_eur_mwh_linear": _finite_text(
            float(hourly.quantile(0.95, interpolation="linear")), label=f"{label} p95"
        ),
        "hourly_min_eur_mwh": _finite_text(float(hourly.min()), label=f"{label} minimum"),
        "hourly_max_eur_mwh": _finite_text(float(hourly.max()), label=f"{label} maximum"),
        "mean_absolute_hourly_ramp_eur_mwh_utc": _finite_text(
            float(ramps.mean()), label=f"{label} mean absolute ramp"
        ),
        "negative_hour_count": negative_count,
        "negative_hour_fraction": _finite_text(
            negative_count / len(hourly), label=f"{label} negative fraction"
        ),
        "negative_price_deficit_mean_all_hours_eur_mwh": _finite_text(
            float((-hourly.clip(upper=0.0)).mean()),
            label=f"{label} negative price deficit mean",
        ),
    }


def _assert_sealed_functionals(
    recomputed: Mapping[str, object], sealed: Mapping[str, object], *, label: str
) -> None:
    if set(recomputed) != set(sealed):
        raise ProspectiveHourlyScoringError(f"{label} functional fields differ")
    for key, value in recomputed.items():
        if key.endswith("_count"):
            if sealed.get(key) != value:
                raise ProspectiveHourlyScoringError(f"{label} {key} differs")
            continue
        expected = _finite_float(sealed.get(key), label=f"{label} sealed {key}")
        observed = _finite_float(value, label=f"{label} recomputed {key}")
        if not math.isclose(
            expected,
            observed,
            rel_tol=0.0,
            abs_tol=_FLOAT_MATCH_ABS_TOLERANCE,
        ):
            raise ProspectiveHourlyScoringError(f"{label} {key} differs")


def score_hourly_prediction(
    prediction: pd.Series, truth: pd.Series, *, label: str
) -> dict[str, object]:
    """Score level and zero-mean shape without conflating the two estimands."""

    if not prediction.index.equals(truth.index):
        raise ProspectiveHourlyScoringError(f"{label} prediction/truth grid differs")
    level_error = float(prediction.mean() - truth.mean())
    prediction_shape = prediction - float(prediction.mean())
    truth_shape = truth - float(truth.mean())
    residual = prediction_shape - truth_shape
    prediction_std = float(prediction_shape.std(ddof=0))
    truth_std = float(truth_shape.std(ddof=0))
    correlation: str | None
    if prediction_std == 0.0 or truth_std == 0.0:
        correlation = None
        correlation_status = "UNSUPPORTED_ZERO_VARIANCE"
    else:
        correlation = _finite_text(
            float(prediction_shape.corr(truth_shape)), label=f"{label} correlation"
        )
        correlation_status = "SUPPORTED"
    prediction_functionals = hourly_shape_functionals(prediction, label=f"{label} prediction")
    truth_functionals = hourly_shape_functionals(truth, label=f"{label} truth")
    raw_window_diagnostics = {
        "midday_mean_error_eur_mwh": _finite_text(
            _finite_float(
                prediction_functionals["midday_mean_eur_mwh"], label="prediction midday"
            )
            - _finite_float(truth_functionals["midday_mean_eur_mwh"], label="truth midday"),
            label=f"{label} midday error",
        ),
        "evening_mean_error_eur_mwh": _finite_text(
            _finite_float(
                prediction_functionals["evening_mean_eur_mwh"], label="prediction evening"
            )
            - _finite_float(truth_functionals["evening_mean_eur_mwh"], label="truth evening"),
            label=f"{label} evening error",
        ),
        "evening_minus_midday_error_eur_mwh": _finite_text(
            _finite_float(
                prediction_functionals["evening_minus_midday_eur_mwh"],
                label="prediction evening minus midday",
            )
            - _finite_float(
                truth_functionals["evening_minus_midday_eur_mwh"],
                label="truth evening minus midday",
            ),
            label=f"{label} evening-minus-midday error",
        ),
    }
    negative_count_error = int(prediction_functionals["negative_hour_count"]) - int(
        truth_functionals["negative_hour_count"]
    )
    negative_fraction_error = _finite_float(
        prediction_functionals["negative_hour_fraction"], label="prediction negative fraction"
    ) - _finite_float(
        truth_functionals["negative_hour_fraction"], label="truth negative fraction"
    )
    negative_deficit_error = _finite_float(
        prediction_functionals["negative_price_deficit_mean_all_hours_eur_mwh"],
        label="prediction negative deficit",
    ) - _finite_float(
        truth_functionals["negative_price_deficit_mean_all_hours_eur_mwh"],
        label="truth negative deficit",
    )
    minimum_error = _finite_float(
        prediction_functionals["hourly_min_eur_mwh"], label="prediction minimum"
    ) - _finite_float(truth_functionals["hourly_min_eur_mwh"], label="truth minimum")
    functional_errors = {
        "midday_premium_error_eur_mwh": _finite_text(
            _finite_float(
                prediction_functionals["midday_premium_to_monthly_mean_eur_mwh"],
                label="prediction midday premium",
            )
            - _finite_float(
                truth_functionals["midday_premium_to_monthly_mean_eur_mwh"],
                label="truth midday premium",
            ),
            label=f"{label} midday premium error",
        ),
        "evening_premium_error_eur_mwh": _finite_text(
            _finite_float(
                prediction_functionals["evening_premium_to_monthly_mean_eur_mwh"],
                label="prediction evening premium",
            )
            - _finite_float(
                truth_functionals["evening_premium_to_monthly_mean_eur_mwh"],
                label="truth evening premium",
            ),
            label=f"{label} evening premium error",
        ),
        "evening_minus_midday_error_eur_mwh": raw_window_diagnostics[
            "evening_minus_midday_error_eur_mwh"
        ],
        "negative_hour_count_error": negative_count_error,
        "absolute_negative_hour_count_error": abs(negative_count_error),
        "negative_hour_fraction_error": _finite_text(
            negative_fraction_error, label=f"{label} negative fraction error"
        ),
        "absolute_negative_hour_fraction_error": _finite_text(
            abs(negative_fraction_error),
            label=f"{label} absolute negative fraction error",
        ),
        "negative_price_deficit_mean_error_eur_mwh": _finite_text(
            negative_deficit_error, label=f"{label} negative deficit error"
        ),
        "absolute_negative_price_deficit_mean_error_eur_mwh": _finite_text(
            abs(negative_deficit_error),
            label=f"{label} absolute negative deficit error",
        ),
        "minimum_price_error_eur_mwh": _finite_text(
            minimum_error, label=f"{label} minimum price error"
        ),
        "absolute_minimum_price_error_eur_mwh": _finite_text(
            abs(minimum_error), label=f"{label} absolute minimum price error"
        ),
    }
    return {
        "monthly_mean_error_eur_mwh": _finite_text(level_error, label=f"{label} level"),
        "monthly_mean_absolute_error_eur_mwh": _finite_text(
            abs(level_error), label=f"{label} absolute level"
        ),
        "zero_mean_hourly_shape": {
            "mae_eur_mwh": _finite_text(
                float(residual.abs().mean()), label=f"{label} shape MAE"
            ),
            "rmse_eur_mwh": _finite_text(
                float(np.sqrt(np.mean(np.square(residual.to_numpy())))),
                label=f"{label} shape RMSE",
            ),
            "pearson_correlation": correlation,
            "pearson_correlation_status": correlation_status,
        },
        "functional_errors": functional_errors,
        "level_contaminated_window_diagnostics": raw_window_diagnostics,
        "truth_functionals": truth_functionals,
    }


def _truth_series(
    document: Mapping[str, object],
    *,
    target_id: str,
    delivery_month: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    expected_count: int,
    allow_test_fixture: bool,
) -> pd.Series:
    status = document.get("status")
    is_test_fixture = document.get("test_fixture") is True
    expected_status = (
        "TEST_FIXTURE_COMPLETE_NATIVE_HOURLY_TRUTH_NO_GO"
        if is_test_fixture
        else "COMPLETE_LOCAL_NATIVE_HOURLY_TRUTH_NO_GO"
    )
    expected_keys = set(_TRUTH_COMMON_KEYS)
    if not is_test_fixture:
        expected_keys.update(_TRUTH_REAL_EXTRA_KEYS)
    if set(document) != expected_keys or status != expected_status:
        raise ProspectiveHourlyScoringError("truth fixture/status/schema fields differ")
    if is_test_fixture and not allow_test_fixture:
        raise ProspectiveHourlyScoringError("truth test fixture is forbidden")
    if (
        document.get("schema_version") != "ch_lt_native_hourly_truth_bundle.v1"
        or document.get("market") != "CH"
        or document.get("target_id") != target_id
        or document.get("delivery_month") != delivery_month
        or _utc_timestamp(
            document.get("delivery_end_utc"), label="truth target end"
        )
        != end_utc
        or document.get("native_hourly_cadence") is not True
        or document.get("quarter_hour_proxy") is not False
        or document.get("outcomes_opened") is not True
        or document.get("truth_publication_receipt_present") is not True
        or document.get("prospective_truth_consumed") is not (not is_test_fixture)
        or document.get("future_holdout_consumed") is not (not is_test_fixture)
        or document.get("t057_consumed") is not False
        or document.get("training_eligible") is not False
        or document.get("model_selection_eligible") is not False
        or document.get("scientific_admission") is not False
        or document.get("production_authorization") is not False
        or document.get("promotion_gate") is not False
    ):
        raise ProspectiveHourlyScoringError("truth authority or target boundary differs")
    if document.get("native_hourly_interval_count") != expected_count:
        raise ProspectiveHourlyScoringError("truth interval count differs")
    opened_at = _utc_timestamp(
        document.get("truth_opened_at_utc_untrusted"), label="truth opened at"
    )
    if opened_at < end_utc:
        raise ProspectiveHourlyScoringError("truth was opened before target maturity")
    observations = document.get("observations")
    if not isinstance(observations, list) or len(observations) != expected_count:
        raise ProspectiveHourlyScoringError("truth observations are incomplete")
    timestamps: list[pd.Timestamp] = []
    prices: list[float] = []
    available_at_values: list[pd.Timestamp] = []
    for position, raw in enumerate(observations):
        observation = _mapping(raw, label=f"truth observation {position}")
        expected_fields = {
            "delivery_start_utc",
            "price_eur_mwh",
            "available_at_utc",
        }
        if not is_test_fixture:
            expected_fields.add("acquisition_id")
        if set(observation) != expected_fields:
            raise ProspectiveHourlyScoringError(
                f"truth observation {position} fields differ"
            )
        if not is_test_fixture and (
            not isinstance(observation.get("acquisition_id"), str)
            or not observation.get("acquisition_id")
        ):
            raise ProspectiveHourlyScoringError(
                f"truth observation {position} acquisition ID differs"
            )
        timestamps.append(
            _utc_timestamp(
                observation.get("delivery_start_utc"),
                label=f"truth observation {position} start",
            )
        )
        prices.append(
            _finite_float(
                observation.get("price_eur_mwh"),
                label=f"truth observation {position} price",
            )
        )
        available_at_values.append(
            _utc_timestamp(
                observation.get("available_at_utc"),
                label=f"truth observation {position} available at",
            )
        )
    if any(value > opened_at for value in available_at_values):
        raise ProspectiveHourlyScoringError(
            "truth observation became available after the truth-open event"
        )
    return _exact_hourly_series(
        pd.Series(prices, index=pd.DatetimeIndex(timestamps), dtype=float),
        start_utc=start_utc,
        end_utc=end_utc,
        expected_count=expected_count,
        label="truth",
    )


def score_structural_commitment(
    root: Path,
    *,
    commitment_path: Path,
    expected_commitment_sha256: str,
    predictions_path: Path,
    expected_predictions_sha256: str,
    scoring_contract_path: Path,
    expected_scoring_contract_sha256: str,
    truth_publication_receipt_path: Path,
    expected_truth_publication_receipt_sha256: str,
    delivery_month: str,
    allow_test_fixture: bool = False,
) -> dict[str, object]:
    """Verify sealed inputs and score one fully matured local delivery month."""

    root = assert_absolute_path_has_no_links(root)
    if Path.cwd() != root:
        raise ProspectiveHourlyScoringError("cwd and root differ")
    selection_audit = origin_selection_auditor.audit_selection(
        registry_path=root / origin_selection_auditor.REGISTRY_RELATIVE,
        expected_registry_sha256=origin_selection_auditor.REGISTRY_SHA256,
    )
    if (
        selection_audit.get("status")
        != (
            "VALID_LOCAL_FUTURE_ORIGIN_REHEARSAL_V7_PACKAGED_MODULE_"
            "SOURCES_BOUND_NONCOUNTABLE_NO_GO"
        )
        or selection_audit.get("outcome_blind") is not True
        or selection_audit.get("future_holdout_consumed") is not False
        or selection_audit.get("t057_consumed") is not False
        or selection_audit.get("scientific_admission") is not False
        or selection_audit.get("production_authorization") is not False
    ):
        raise ProspectiveHourlyScoringError("future origin selection audit differs")
    selection_selected, future_selection = _read_bound_json(
        root,
        origin_selection_auditor.REGISTRY_RELATIVE,
        origin_selection_auditor.REGISTRY_SHA256,
        label="future origin selection",
    )
    selection_bindings = _mapping(
        future_selection.get("bindings"), label="future selection bindings"
    )
    expected_commitment_binding = _mapping(
        selection_bindings.get("prediction_commitment"),
        label="selected prediction commitment",
    )
    expected_predictions_binding = _mapping(
        selection_bindings.get("sealed_predictions"), label="selected predictions"
    )
    expected_scoring_binding = _mapping(
        selection_bindings.get("hourly_scoring_contract"),
        label="selected scoring contract",
    )
    caller_bindings = (
        (commitment_path, expected_commitment_sha256, expected_commitment_binding),
        (predictions_path, expected_predictions_sha256, expected_predictions_binding),
        (
            scoring_contract_path,
            expected_scoring_contract_sha256,
            expected_scoring_binding,
        ),
    )
    for caller_path, caller_sha256, selected_binding in caller_bindings:
        selected_path = Path(os.path.abspath(root / Path(str(selected_binding.get("path")))))
        supplied_path = caller_path if caller_path.is_absolute() else root / caller_path
        supplied_path = Path(os.path.abspath(supplied_path))
        if (
            supplied_path != selected_path
            or caller_sha256 != selected_binding.get("sha256")
        ):
            raise ProspectiveHourlyScoringError(
                "caller evidence is not the canonically selected future origin chain"
            )
    commitment_selected, commitment = _read_bound_json(
        root,
        commitment_path,
        expected_commitment_sha256,
        label="prediction commitment",
    )
    predictions_selected, predictions = _read_bound_json(
        root,
        predictions_path,
        expected_predictions_sha256,
        label="sealed predictions",
    )
    scoring_contract_selected, scoring_contract = _read_bound_json(
        root,
        scoring_contract_path,
        expected_scoring_contract_sha256,
        label="hourly scoring contract",
    )
    sealed_payload = _mapping(commitment.get("sealed_payload"), label="sealed payload")
    if (
        commitment.get("schema_version") != "ch_lt_structural_prediction_commitment.v3"
        or commitment.get("status")
        != "SEALED_LOCAL_STRUCTURAL_DRY_RUN_NONCOUNTABLE_NO_GO"
        or commitment.get("future_holdout_consumed") is not None
        and commitment.get("future_holdout_consumed") is not False
        or sealed_payload.get("sha256") != expected_predictions_sha256
        or predictions.get("schema_version") != "ch_lt_structural_prediction_set.v3"
        or predictions.get("outcomes_opened") is not False
        or predictions.get("future_holdout_consumed") is not False
        or predictions.get("t057_consumed") is not False
    ):
        raise ProspectiveHourlyScoringError("prediction seal boundary differs")

    selected_prediction_binding = _mapping(
        scoring_contract.get("selected_prediction_binding"),
        label="scoring selected prediction binding",
    )
    supersedes = _mapping(scoring_contract.get("supersedes"), label="scoring supersession")
    _, formula_contract = _read_bound_json(
        root,
        Path(str(supersedes.get("path"))),
        str(supersedes.get("sha256")),
        label="normative scoring formula contract",
    )
    evaluation_unit = _mapping(
        formula_contract.get("evaluation_unit"), label="scoring evaluation unit"
    )
    point_scores = _mapping(formula_contract.get("point_scores"), label="point scores")
    hourly_shape_score = _mapping(
        point_scores.get("zero_mean_hourly_shape"), label="hourly shape score"
    )
    functional_score = _mapping(
        point_scores.get("shape_functionals"), label="shape functional score"
    )
    aggregation = _mapping(formula_contract.get("aggregation"), label="score aggregation")
    authority = _mapping(
        scoring_contract.get("authority_boundary"), label="scoring authority boundary"
    )
    if (
        scoring_contract.get("schema_version")
        != "ch_lt_future_hourly_scoring_contract.v2"
        or scoring_contract.get("status")
        != "OUTCOME_BLIND_LOCAL_SCORING_SPECIFICATION_V7_NO_GO"
        or selected_prediction_binding.get("commitment_sha256")
        != expected_commitment_sha256
        or selected_prediction_binding.get("predictions_sha256")
        != expected_predictions_sha256
        or selected_prediction_binding.get("commitment_id")
        != commitment.get("commitment_id")
        or selected_prediction_binding.get("prediction_set_id")
        != predictions.get("prediction_set_id")
        or supersedes.get("contract_id") != formula_contract.get("contract_id")
        or formula_contract.get("schema_version")
        != "ch_lt_future_hourly_scoring_contract.v1"
        or evaluation_unit.get("primary_scenario") != "central"
        or evaluation_unit.get("sensitivity_scenarios") != ["slow", "fast"]
        or evaluation_unit.get("scenario_pooling") != "FORBIDDEN"
        or evaluation_unit.get("same_origin_same_mask_baseline_required") is not True
        or point_scores.get("residual_sign") != "PREDICTION_MINUS_TRUTH"
        or hourly_shape_score.get("prediction_centering")
        != "PREDICTION_HOUR_MINUS_TARGET_PREDICTION_HOURLY_MEAN"
        or hourly_shape_score.get("truth_centering")
        != "TRUTH_HOUR_MINUS_TARGET_TRUTH_HOURLY_MEAN"
        or hourly_shape_score.get("pearson_zero_variance")
        != "UNSUPPORTED_NOT_ZERO_NOT_WIN"
        or functional_score.get("aggregation_input")
        != "ABS_FUNCTIONAL_RESIDUAL_ONLY"
        or aggregation.get("missing_or_nonfinite")
        != "UNSUPPORTED_NEVER_IMPUTED_NEVER_ZERO_NEVER_WIN"
        or aggregation.get("anti_cancellation")
        != "ABSOLUTE_ERRORS_REQUIRED_BEFORE_ANY_AGGREGATION"
        or not authority
        or any(value is not False for value in authority.values())
    ):
        raise ProspectiveHourlyScoringError("hourly scoring contract boundary differs")

    bindings = _mapping(commitment.get("bindings"), label="commitment bindings")
    inventory_binding = _mapping(
        bindings.get("origin_target_mask_inventory"), label="inventory binding"
    )
    _, inventory = _read_bound_json(
        root,
        Path(str(inventory_binding.get("path"))),
        str(inventory_binding.get("sha256")),
        label="origin target inventory",
    )
    targets = inventory.get("targets")
    if not isinstance(targets, list):
        raise ProspectiveHourlyScoringError("inventory targets are missing")
    matching_targets = [
        _mapping(item, label="inventory target")
        for item in targets
        if isinstance(item, Mapping) and item.get("delivery_month") == delivery_month
    ]
    if len(matching_targets) != 1:
        raise ProspectiveHourlyScoringError("delivery month is not unique in inventory")
    target = matching_targets[0]
    target_id = str(target.get("target_id"))
    start_utc = _utc_timestamp(target.get("delivery_start_utc"), label="target start")
    end_utc = _utc_timestamp(target.get("delivery_end_utc"), label="target end")
    expected_hour_count = int(target.get("expected_native_hourly_interval_count"))

    raw_prediction_rows = predictions.get("predictions")
    if not isinstance(raw_prediction_rows, list):
        raise ProspectiveHourlyScoringError("prediction rows are missing")
    prediction_rows = [
        _mapping(item, label="prediction row")
        for item in raw_prediction_rows
        if isinstance(item, Mapping) and item.get("target_id") == target_id
    ]
    if [item.get("scenario") for item in prediction_rows] != list(_SCENARIOS):
        raise ProspectiveHourlyScoringError("target scenario order differs")

    pfc_bindings = _mapping(bindings.get("pfc_outputs"), label="PFC bindings")
    verified_hourly_predictions: list[tuple[str, pd.Series]] = []
    for scenario, sealed_row in zip(_SCENARIOS, prediction_rows, strict=True):
        role = f"pfc_{scenario}"
        pfc_binding = _mapping(pfc_bindings.get(role), label=f"{role} binding")
        _, pfc_payload = _read_bound_bytes(
            root,
            Path(str(pfc_binding.get("path"))),
            str(pfc_binding.get("sha256")),
            label=role,
            max_bytes=_MAX_PFC_BYTES,
        )
        frame = pd.read_parquet(io.BytesIO(pfc_payload))
        if "price_shape" not in frame.columns:
            raise ProspectiveHourlyScoringError(f"{role} price_shape is missing")
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ProspectiveHourlyScoringError(f"{role} index is not timezone-aware")
        selected = frame.loc[
            (frame.index >= start_utc) & (frame.index < end_utc), "price_shape"
        ]
        hourly = aggregate_exact_quarter_hours(
            selected,
            start_utc=start_utc,
            end_utc=end_utc,
            expected_hour_count=expected_hour_count,
            label=role,
        )
        recomputed_functionals = hourly_shape_functionals(hourly, label=role)
        sealed_functionals = _mapping(
            sealed_row.get("hourly_shape_functionals"),
            label=f"{scenario} sealed functionals",
        )
        _assert_sealed_functionals(
            recomputed_functionals,
            sealed_functionals,
            label=f"{scenario} sealed prediction",
        )
        verified_hourly_predictions.append((scenario, hourly))

    truth_publication_receipt_selected, truth_publication_receipt = _read_bound_json(
        root,
        truth_publication_receipt_path,
        expected_truth_publication_receipt_sha256,
        label="truth publication receipt",
    )
    if set(truth_publication_receipt) != _TRUTH_PUBLICATION_RECEIPT_KEYS:
        raise ProspectiveHourlyScoringError("truth publication receipt fields differ")
    receipt_core = dict(truth_publication_receipt)
    receipt_id = _lower_sha256(
        receipt_core.pop("receipt_id", None), label="truth publication receipt ID"
    )
    if hashlib.sha256(canonical_score_bytes(receipt_core)).hexdigest() != receipt_id:
        raise ProspectiveHourlyScoringError("truth publication receipt ID mismatch")
    receipt_authority = _mapping(
        truth_publication_receipt.get("authority_boundary"),
        label="truth publication receipt authority",
    )
    receipt_commitment = _mapping(
        truth_publication_receipt.get("prediction_commitment"),
        label="truth publication receipt commitment",
    )
    receipt_scoring = _mapping(
        truth_publication_receipt.get("hourly_scoring_contract"),
        label="truth publication receipt scoring contract",
    )
    receipt_selection = _mapping(
        truth_publication_receipt.get("future_origin_selection"),
        label="truth publication receipt future selection",
    )
    receipt_end = _utc_timestamp(
        truth_publication_receipt.get("delivery_end_utc"), label="receipt target end"
    )
    receipt_opened_at = _utc_timestamp(
        truth_publication_receipt.get("truth_opened_at_utc_untrusted"),
        label="receipt truth opened at",
    )
    receipt_is_test_fixture = truth_publication_receipt.get("test_fixture") is True
    expected_receipt_status = (
        "TEST_FIXTURE_LOCAL_DIAGNOSTIC_TRUTH_PUBLICATION_RECEIPT_NO_GO"
        if receipt_is_test_fixture
        else "LOCAL_DIAGNOSTIC_TRUTH_PUBLICATION_RECEIPT_NO_GO"
    )
    if receipt_is_test_fixture and not allow_test_fixture:
        raise ProspectiveHourlyScoringError(
            "truth publication receipt fixture is forbidden"
        )
    if (
        truth_publication_receipt.get("schema_version")
        != "ch_lt_local_truth_publication_receipt.v1"
        or truth_publication_receipt.get("status") != expected_receipt_status
        or truth_publication_receipt.get("target_id") != target_id
        or truth_publication_receipt.get("delivery_month") != delivery_month
        or receipt_end != end_utc
        or receipt_opened_at < end_utc
        or truth_publication_receipt.get("local_diagnostic_truth_read_occurred")
        is not True
        or receipt_commitment.get("sha256") != expected_commitment_sha256
        or receipt_commitment.get("commitment_id") != commitment.get("commitment_id")
        or receipt_scoring.get("sha256") != expected_scoring_contract_sha256
        or receipt_scoring.get("contract_id") != scoring_contract.get("contract_id")
        or receipt_selection.get("sha256")
        != origin_selection_auditor.REGISTRY_SHA256
        or receipt_selection.get("selection_id") != future_selection.get("selection_id")
        or set(receipt_authority) != _TRUTH_OPEN_AUTHORITY_KEYS
        or any(value is not False for value in receipt_authority.values())
        or truth_publication_receipt.get("future_holdout_consumed") is not False
        or truth_publication_receipt.get("t057_consumed") is not False
        or truth_publication_receipt.get("scientific_admission") is not False
        or truth_publication_receipt.get("production_authorization") is not False
        or truth_publication_receipt.get("promotion_gate") is not False
    ):
        raise ProspectiveHourlyScoringError("truth publication receipt boundary differs")
    if _local_wall_clock_utc() < end_utc:
        raise ProspectiveHourlyScoringError(
            "local wall clock has not reached target maturity"
        )

    truth_binding = _mapping(
        truth_publication_receipt.get("truth_bundle"),
        label="truth publication bundle binding",
    )
    expected_truth_sha256 = _lower_sha256(
        truth_binding.get("sha256"), label="native hourly truth bundle SHA-256"
    )
    truth_selected, truth_document = _read_bound_json(
        root,
        Path(str(truth_binding.get("path"))),
        expected_truth_sha256,
        label="native hourly truth bundle",
    )
    if (truth_document.get("test_fixture") is True) is not receipt_is_test_fixture:
        raise ProspectiveHourlyScoringError(
            "truth receipt and bundle fixture classifications differ"
        )
    if truth_document.get("test_fixture") is not True:
        truth_bundle_id = _lower_sha256(
            truth_document.get("truth_bundle_id"), label="truth bundle ID"
        )
        truth_core = {
            key: value for key, value in truth_document.items() if key != "truth_bundle_id"
        }
        if hashlib.sha256(canonical_score_bytes(truth_core)).hexdigest() != truth_bundle_id:
            raise ProspectiveHourlyScoringError("truth bundle ID mismatch")
        source_ledger = _mapping(
            truth_document.get("source_ledger"), label="truth source ledger"
        )
        _, ledger = _read_bound_json(
            root,
            Path(str(source_ledger.get("path"))),
            str(source_ledger.get("sha256")),
            label="truth source ledger",
        )
        if (
            ledger.get("ledger_id") != source_ledger.get("ledger_id")
            or source_ledger.get("recursive_reconstruction_verified") is not True
        ):
            raise ProspectiveHourlyScoringError("truth source ledger binding differs")
        construction_request = _mapping(
            ledger.get("construction_request"), label="ledger construction request"
        )
        reconstructed_ledger = build_prospective_capture_ledger(
            repo_root=root,
            request_path=root / Path(str(construction_request.get("path"))),
            expected_request_sha256=str(construction_request.get("sha256")),
        )
        if dict(reconstructed_ledger) != dict(ledger):
            raise ProspectiveHourlyScoringError(
                "truth source ledger recursive reconstruction differs"
            )
        truth_commitment = _mapping(
            truth_document.get("prediction_commitment"),
            label="truth prediction commitment",
        )
        if (
            truth_commitment.get("sha256") != expected_commitment_sha256
            or truth_commitment.get("commitment_id") != commitment.get("commitment_id")
        ):
            raise ProspectiveHourlyScoringError(
                "truth prediction commitment binding differs"
            )
        truth_selection = _mapping(
            truth_document.get("future_origin_selection"),
            label="truth future origin selection",
        )
        if (
            truth_selection.get("path")
            != selection_selected.relative_to(root).as_posix()
            or truth_selection.get("sha256")
            != origin_selection_auditor.REGISTRY_SHA256
            or truth_selection.get("selection_id")
            != future_selection.get("selection_id")
        ):
            raise ProspectiveHourlyScoringError(
                "truth future origin selection binding differs"
            )
    truth = _truth_series(
        truth_document,
        target_id=target_id,
        delivery_month=delivery_month,
        start_utc=start_utc,
        end_utc=end_utc,
        expected_count=expected_hour_count,
        allow_test_fixture=allow_test_fixture,
    )
    scores = [
        {
            "scenario": scenario,
            **score_hourly_prediction(hourly, truth, label=scenario),
        }
        for scenario, hourly in verified_hourly_predictions
    ]

    return {
        "schema_version": "ch_lt_prospective_hourly_score.v1",
        "status": "LOCAL_PROSPECTIVE_SCORING_DIAGNOSTIC_NO_GO",
        "market": "CH",
        "delivery_month": delivery_month,
        "target_id": target_id,
        "native_hourly_interval_count": expected_hour_count,
        "bindings": {
            "future_origin_selection": {
                "path": selection_selected.relative_to(root).as_posix(),
                "sha256": origin_selection_auditor.REGISTRY_SHA256,
                "selection_id": future_selection.get("selection_id"),
            },
            "prediction_commitment": {
                "path": commitment_selected.relative_to(root).as_posix(),
                "sha256": expected_commitment_sha256,
                "commitment_id": commitment.get("commitment_id"),
            },
            "sealed_predictions": {
                "path": predictions_selected.relative_to(root).as_posix(),
                "sha256": expected_predictions_sha256,
                "prediction_set_id": predictions.get("prediction_set_id"),
            },
            "hourly_scoring_contract": {
                "path": scoring_contract_selected.relative_to(root).as_posix(),
                "sha256": expected_scoring_contract_sha256,
                "contract_id": scoring_contract.get("contract_id"),
            },
            "normative_formula_contract": {
                "path": str(supersedes.get("path")),
                "sha256": str(supersedes.get("sha256")),
                "contract_id": formula_contract.get("contract_id"),
            },
            "native_hourly_truth_bundle": {
                "path": truth_selected.relative_to(root).as_posix(),
                "sha256": expected_truth_sha256,
                "status": truth_document.get("status"),
            },
            "truth_publication_receipt": {
                "path": truth_publication_receipt_selected.relative_to(root).as_posix(),
                "sha256": expected_truth_publication_receipt_sha256,
                "receipt_id": truth_publication_receipt.get("receipt_id"),
            },
        },
        "estimand_separation": {
            "level": "MONTHLY_MEAN_ERROR_PREDICTION_MINUS_TRUTH",
            "shape": "ERROR_AFTER_INDEPENDENT_MONTHLY_DEMEANING",
            "functionals": "PRECOMMITTED_CENTERED_PREMIUM_AND_ABSOLUTE_TAIL_ERRORS",
        },
        "scenario_interpretation": "STRUCTURAL_SENSITIVITY_NOT_PROBABILISTIC_COVERAGE",
        "primary_scenario": "central",
        "sensitivity_scenarios": ["slow", "fast"],
        "scenario_pooling_for_quality_claims": "FORBIDDEN",
        "same_origin_same_mask_baseline_attached": False,
        "quality_or_noninferiority_claim_authorized": False,
        "scores": scores,
        "outcomes_opened": True,
        "prospective_truth_consumed": truth_document.get("test_fixture") is not True,
        "test_fixture_consumed": truth_document.get("test_fixture") is True,
        "future_holdout_consumed": truth_document.get("test_fixture") is not True,
        "t057_consumed": False,
        "training_eligible": False,
        "model_selection_eligible": False,
        "scientific_admission": False,
        "production_authorization": False,
        "promotion_gate": False,
    }


def canonical_score_bytes(document: Mapping[str, object]) -> bytes:
    """Return deterministic exact bytes for a score document."""

    return (
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
