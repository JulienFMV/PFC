"""Fail-closed reconciliation of LSEG and ENTSO-E day-ahead actual prices.

LSEG is an independent realized-price cross-check for CH, AT, DE-LU and FR.
ENTSO-E remains the homogeneous five-zone candidate panel, including IT-NORD.
The module never queries Databricks, silently substitutes a source, or grants
model, monthly-level, model-selection or production authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pfc_shaping.data.governed_lt_acquisition import dataframe_semantic_sha256
from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.validation.entsoe_day_ahead_export import (
    DayAheadConsumerExport,
    EntsoeDayAheadExportError,
    validate_realized_latest_candidate,
)
from pfc_shaping.validation.entsoe_day_ahead_prd import (
    PIT_COLUMNS,
    DayAheadPitExtract,
    EntsoeDayAheadPrdError,
    validate_day_ahead_pit_extract,
)

ROOT = Path(__file__).resolve().parents[2]
LSEG_PIT_SQL_PATH = ROOT / "docs/data/sql/databricks_prd_lseg_epex_actuals_pit_extract.sql"
LSEG_PIT_SQL_SHA256 = "a4cc587d9f8116dd3f20c295ce8edbf291bfcf06a3f70f859d6fc9816c336bed"
LSEG_LATEST_SQL_PATH = ROOT / "docs/data/sql/databricks_prd_lseg_epex_actuals_latest_extract.sql"
LSEG_LATEST_SQL_SHA256 = "be9e94de41c9e0c65f1814be617e3abd591103d6b869c6c80444761e6de22fff"

LSEG_PIT_AUDIT_SCHEMA = "fmv_lseg_epex_actuals_prd_pit_extract.v1"
LSEG_LATEST_AUDIT_SCHEMA = "fmv_lseg_epex_actuals_prd_latest_extract.v1"
RECONCILIATION_SCHEMA = "fmv_lseg_entsoe_spot_reconciliation.v2"
PASS_STATUS = "PASS_SPOT_SOURCE_RECONCILIATION_NOT_MODEL_AUTHORITY"
BLOCKED_STATUS = "BLOCKED_SPOT_SOURCE_RECONCILIATION"
PIT_RESULT_ROW_LIMIT = 20_001
MAX_PIT_WINDOW_DAYS = 31

LSEG_CURVES: Mapping[str, tuple[str, str]] = {
    "CH": ("115688058", "PT1H"),
    "AT": ("165444048", "PT15M"),
    "DE_LU": ("165349556", "PT15M"),
    "FR": ("165442712", "PT15M"),
}
ENTSOE_FIELDS: Mapping[str, str] = {
    "ch_price": "CH",
    "at_price": "AT",
    "de_lu_price": "DE_LU",
    "fr_price": "FR",
    "it_nord_price": "IT_NORD",
}
LSEG_PIT_COLUMNS = (
    "market_zone",
    "curve_id",
    "interval_start_utc",
    "interval_end_utc",
    "resolution",
    "price_eur_per_mwh",
    "pipeline_first_seen_at_utc",
    "pull_ts_utc",
    "curve_value_vintage_id",
)

_RESOLUTION_SECONDS = {"PT15M": 900, "PT30M": 1_800, "PT60M": 3_600, "PT1H": 3_600}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CLEAN_TEXT = re.compile(r"^[^\x00-\x1f\x7f]+$")


class SpotSourceReconciliationError(ValueError):
    """Raised when an input or audit binding is structurally unsafe."""


@dataclass(frozen=True)
class LsegActualsPitExtract:
    """Validated local LSEG actual-price rows plus their audit binding."""

    frame: pd.DataFrame
    audit: Mapping[str, object]


@dataclass(frozen=True)
class LsegActualsLatestExtract:
    """Validated latest LSEG actual-price rows without historical PIT authority."""

    frame: pd.DataFrame
    audit: Mapping[str, object]


@dataclass(frozen=True)
class SpotFinding:
    """One aggregate reconciliation failure; raw prices are never embedded."""

    code: str
    message: str
    market_zone: str | None = None
    affected_hours: int | None = None
    severity: str = "CRITICAL"

    def as_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "market_zone": self.market_zone,
            "affected_hours": self.affected_hours,
        }


@dataclass(frozen=True)
class SpotReconciliationPolicy:
    """Explicit, hashable admission thresholds; there are no hidden defaults."""

    policy_id: str
    minimum_matched_hours_per_zone: int
    minimum_hourly_overlap_ratio: float
    maximum_p95_abs_difference_eur_per_mwh: float
    maximum_absolute_bias_eur_per_mwh: float
    maximum_single_hour_abs_difference_eur_per_mwh: float

    def __post_init__(self) -> None:
        _text(self.policy_id, "policy ID")
        if (
            isinstance(self.minimum_matched_hours_per_zone, bool)
            or not isinstance(self.minimum_matched_hours_per_zone, int)
            or self.minimum_matched_hours_per_zone < 1
        ):
            raise SpotSourceReconciliationError(
                "minimum matched hours per zone must be a positive integer"
            )
        ratio = _finite_float(self.minimum_hourly_overlap_ratio, "minimum overlap ratio")
        if not 0 < ratio <= 1:
            raise SpotSourceReconciliationError("minimum overlap ratio must be in (0, 1]")
        thresholds = (
            _finite_nonnegative(
                self.maximum_p95_abs_difference_eur_per_mwh,
                "maximum p95 absolute difference",
            ),
            _finite_nonnegative(
                self.maximum_absolute_bias_eur_per_mwh,
                "maximum absolute bias",
            ),
            _finite_nonnegative(
                self.maximum_single_hour_abs_difference_eur_per_mwh,
                "maximum single-hour absolute difference",
            ),
        )
        if thresholds[2] < thresholds[0]:
            raise SpotSourceReconciliationError(
                "single-hour threshold cannot be below the p95 threshold"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "minimum_matched_hours_per_zone": self.minimum_matched_hours_per_zone,
            "minimum_hourly_overlap_ratio": self.minimum_hourly_overlap_ratio,
            "maximum_p95_abs_difference_eur_per_mwh": (self.maximum_p95_abs_difference_eur_per_mwh),
            "maximum_absolute_bias_eur_per_mwh": self.maximum_absolute_bias_eur_per_mwh,
            "maximum_single_hour_abs_difference_eur_per_mwh": (
                self.maximum_single_hour_abs_difference_eur_per_mwh
            ),
        }

    @property
    def semantic_sha256(self) -> str:
        payload = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SpotReconciliationReport:
    """Aggregate cross-source evidence with deliberately limited authority."""

    status: str
    findings: tuple[SpotFinding, ...]
    metrics: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        passed = self.status == PASS_STATUS
        return {
            "schema_version": RECONCILIATION_SCHEMA,
            "status": self.status,
            "findings": [finding.as_dict() for finding in self.findings],
            "metrics": dict(self.metrics),
            "layer_policy": {
                "entsoe_role": "HOMOGENEOUS_FIVE_ZONE_MODELLING_PANEL_CANDIDATE",
                "lseg_epex_role": "REALIZED_SPOT_CROSSCHECK_CH_AT_DE_LU_FR",
                "it_nord_role": "ENTSOE_ONLY_NO_ACTIVE_LSEG_EPEX_CURVE",
                "mismatch_action": "BLOCK_NO_SILENT_SOURCE_SUBSTITUTION",
                "monthly_level_authority": "EEX_CONSTRAINED_MONTHLY_SOLVER_ONLY",
            },
            "authorities": {
                "source_reconciliation_validated": passed,
                "cadence_completeness_authorized": False,
                "model_input_authorized": False,
                "monthly_level_authorized": False,
                "model_selection_authorized": False,
                "production_authorized": False,
            },
        }


def verify_lseg_sql_binding() -> str:
    """Verify the immutable, bounded LSEG extraction template."""

    try:
        payload = read_stable_single_link_file(
            LSEG_PIT_SQL_PATH, label="LSEG EPEX PIT SQL", max_bytes=200_000
        )
    except (OSError, ValueError) as exc:
        raise SpotSourceReconciliationError("LSEG EPEX PIT SQL read failed") from exc
    observed = hashlib.sha256(payload).hexdigest()
    if observed != LSEG_PIT_SQL_SHA256:
        raise SpotSourceReconciliationError("LSEG EPEX PIT SQL binding differs")
    return observed


def verify_lseg_latest_sql_binding() -> str:
    """Verify the immutable, bounded LSEG latest-observation template."""

    try:
        payload = read_stable_single_link_file(
            LSEG_LATEST_SQL_PATH, label="LSEG EPEX latest SQL", max_bytes=200_000
        )
    except (OSError, ValueError) as exc:
        raise SpotSourceReconciliationError("LSEG EPEX latest SQL read failed") from exc
    observed = hashlib.sha256(payload).hexdigest()
    if observed != LSEG_LATEST_SQL_SHA256:
        raise SpotSourceReconciliationError("LSEG EPEX latest SQL binding differs")
    return observed


def build_lseg_epex_actuals_pit_parameters(
    *,
    window_start_utc: str | pd.Timestamp,
    window_end_utc: str | pd.Timestamp,
    as_of_utc: str | pd.Timestamp,
) -> dict[str, object]:
    """Build one UTC-calendar-month parameters for the LSEG PIT template."""

    verify_lseg_sql_binding()
    return _build_lseg_window_parameters(
        window_start_utc=window_start_utc,
        window_end_utc=window_end_utc,
        cutoff_utc=as_of_utc,
        label="LSEG PIT",
        allow_two_utc_months=False,
    )


def build_lseg_epex_actuals_latest_parameters(
    *,
    window_start_utc: str | pd.Timestamp,
    window_end_utc: str | pd.Timestamp,
    assessed_at_utc: str | pd.Timestamp,
) -> dict[str, object]:
    """Build bounded latest-observation parameters without claiming historical PIT."""

    verify_lseg_latest_sql_binding()
    bounded = _build_lseg_window_parameters(
        window_start_utc=window_start_utc,
        window_end_utc=window_end_utc,
        cutoff_utc=assessed_at_utc,
        label="LSEG latest",
        allow_two_utc_months=True,
    )
    return {
        "start_utc": bounded["start_utc"],
        "end_utc": bounded["end_utc"],
        "assessed_at_utc": bounded["as_of_utc"],
        "start_value_date": bounded["start_value_date"],
        "end_value_date": bounded["end_value_date"],
    }


def _build_lseg_window_parameters(
    *,
    window_start_utc: str | pd.Timestamp,
    window_end_utc: str | pd.Timestamp,
    cutoff_utc: str | pd.Timestamp,
    label: str,
    allow_two_utc_months: bool,
) -> dict[str, object]:
    start = _utc_timestamp(window_start_utc, f"{label} window start")
    end = _utc_timestamp(window_end_utc, f"{label} window end")
    cutoff = _utc_timestamp(cutoff_utc, f"{label} cutoff")
    _ordered_window(start, end, label)
    if end - start > pd.Timedelta(days=MAX_PIT_WINDOW_DAYS):
        raise SpotSourceReconciliationError(f"{label} window exceeds 31 days")
    if any(value.second or value.microsecond or value.minute % 15 for value in (start, end)):
        raise SpotSourceReconciliationError(f"{label} bounds must lie on a 15-minute UTC grid")
    last_instant = end - pd.Timedelta(nanoseconds=1)
    start_month = start.year * 12 + start.month
    end_month = last_instant.year * 12 + last_instant.month
    month_span = end_month - start_month
    allowed_spans = {0, 1} if allow_two_utc_months else {0}
    if month_span not in allowed_spans:
        scope = "at most two" if allow_two_utc_months else "one"
        raise SpotSourceReconciliationError(
            f"{label} window must stay inside {scope} UTC calendar month(s)"
        )
    if cutoff < end:
        raise SpotSourceReconciliationError(
            "LSEG actuals reconciliation cutoff precedes the delivery window end"
        )
    return {
        "start_utc": _utc_text(start),
        "end_utc": _utc_text(end),
        "as_of_utc": _utc_text(cutoff),
        "start_value_date": start.date().isoformat(),
        "end_value_date": end.date().isoformat(),
    }


def _validate_lseg_extract_rows(
    frame: pd.DataFrame,
    *,
    window_start_utc: str,
    window_end_utc: str,
    cutoff_utc: str,
    label: str,
) -> pd.DataFrame:
    result = _exact_frame(frame, LSEG_PIT_COLUMNS, label)
    if result.empty:
        raise SpotSourceReconciliationError(f"{label} is empty")
    if len(result) >= PIT_RESULT_ROW_LIMIT:
        raise SpotSourceReconciliationError(f"{label} hit the rejection sentinel")

    for column in ("market_zone", "curve_id", "resolution", "curve_value_vintage_id"):
        result[column] = result[column].map(lambda value: _text(value, column))
    for column in (
        "interval_start_utc",
        "interval_end_utc",
        "pipeline_first_seen_at_utc",
        "pull_ts_utc",
    ):
        values = pd.to_datetime(result[column], errors="coerce", utc=True)
        if values.isna().any():
            raise SpotSourceReconciliationError(f"LSEG extract contains invalid {column}")
        result[column] = values

    expected_curve = result["market_zone"].map(
        {zone: curve for zone, (curve, _) in LSEG_CURVES.items()}
    )
    expected_resolution = result["market_zone"].map(
        {zone: resolution for zone, (_, resolution) in LSEG_CURVES.items()}
    )
    if expected_curve.isna().any() or result["curve_id"].ne(expected_curve).any():
        raise SpotSourceReconciliationError(
            "LSEG extract differs from the exact four-price-curve binding"
        )
    if result["resolution"].ne(expected_resolution).any():
        raise SpotSourceReconciliationError(
            "LSEG extract resolution differs from its curve binding"
        )
    if set(result["market_zone"]) != set(LSEG_CURVES):
        raise SpotSourceReconciliationError("LSEG extract does not contain all four market zones")

    start = _utc_timestamp(window_start_utc, "LSEG extract start")
    end = _utc_timestamp(window_end_utc, "LSEG extract end")
    cutoff = _utc_timestamp(cutoff_utc, "LSEG extract cutoff")
    if result["interval_start_utc"].lt(start).any() or result["interval_start_utc"].ge(end).any():
        raise SpotSourceReconciliationError(
            "LSEG extract contains rows outside the delivery window"
        )
    if result["interval_end_utc"].gt(end).any():
        raise SpotSourceReconciliationError("LSEG extract ends after the delivery window")
    if result["pipeline_first_seen_at_utc"].gt(cutoff).any():
        raise SpotSourceReconciliationError("LSEG extract leaks a value unavailable at the cutoff")
    if result["pull_ts_utc"].lt(result["pipeline_first_seen_at_utc"]).any():
        raise SpotSourceReconciliationError("LSEG pull time precedes pipeline first-seen time")

    seconds = result["resolution"].map(_RESOLUTION_SECONDS)
    duration = (result["interval_end_utc"] - result["interval_start_utc"]).dt.total_seconds()
    if seconds.isna().any() or duration.ne(seconds).any():
        raise SpotSourceReconciliationError("LSEG extract interval duration is inconsistent")
    prices = pd.to_numeric(result["price_eur_per_mwh"], errors="coerce")
    if prices.isna().any() or not all(math.isfinite(float(value)) for value in prices):
        raise SpotSourceReconciliationError("LSEG extract contains a non-finite price")
    result["price_eur_per_mwh"] = prices.astype(float)
    if result["curve_value_vintage_id"].duplicated(keep=False).any():
        raise SpotSourceReconciliationError("LSEG curve vintage ID is duplicated")
    if result.duplicated(
        ["market_zone", "curve_id", "interval_start_utc", "interval_end_utc"], keep=False
    ).any():
        raise SpotSourceReconciliationError("LSEG extract interval grain is duplicated")

    return result.sort_values(
        ["market_zone", "interval_start_utc", "interval_end_utc"], kind="mergesort"
    ).reset_index(drop=True)


def validate_lseg_epex_actuals_pit_extract(
    frame: pd.DataFrame,
    *,
    window_start_utc: str | pd.Timestamp,
    window_end_utc: str | pd.Timestamp,
    as_of_utc: str | pd.Timestamp,
    pit_query_sha256: str,
) -> LsegActualsPitExtract:
    """Validate exact LSEG price curves and their pipeline-first-seen PIT cut."""

    if _sha256(pit_query_sha256, "LSEG PIT query") != LSEG_PIT_SQL_SHA256:
        raise SpotSourceReconciliationError("LSEG PIT query SHA-256 differs")
    parameters = build_lseg_epex_actuals_pit_parameters(
        window_start_utc=window_start_utc,
        window_end_utc=window_end_utc,
        as_of_utc=as_of_utc,
    )
    result = _validate_lseg_extract_rows(
        frame,
        window_start_utc=parameters["start_utc"],
        window_end_utc=parameters["end_utc"],
        cutoff_utc=parameters["as_of_utc"],
        label="LSEG EPEX PIT extract",
    )
    audit = {
        "schema_version": LSEG_PIT_AUDIT_SCHEMA,
        "status": "PASS_LSEG_BOUNDED_PIT_EXTRACT_NOT_MODEL_INPUT_AUTHORITY",
        "pit_query_sha256": LSEG_PIT_SQL_SHA256,
        "row_count": len(result),
        "frame_semantic_sha256": dataframe_semantic_sha256(result),
        "window_start_utc": parameters["start_utc"],
        "window_end_utc": parameters["end_utc"],
        "as_of_utc": parameters["as_of_utc"],
        "curve_binding": {
            zone: {"curve_id": curve, "resolution": resolution}
            for zone, (curve, resolution) in LSEG_CURVES.items()
        },
        "authorities": {
            "point_in_time_filter_validated": True,
            "cadence_completeness_authorized": False,
            "model_input_authorized": False,
            "model_selection_authorized": False,
            "production_authorized": False,
        },
    }
    return LsegActualsPitExtract(result, audit)


def validate_lseg_epex_actuals_latest_extract(
    frame: pd.DataFrame,
    *,
    window_start_utc: str | pd.Timestamp,
    window_end_utc: str | pd.Timestamp,
    assessed_at_utc: str | pd.Timestamp,
    latest_query_sha256: str,
) -> LsegActualsLatestExtract:
    """Validate an exact latest LSEG snapshot without granting PIT authority."""

    if _sha256(latest_query_sha256, "LSEG latest query") != LSEG_LATEST_SQL_SHA256:
        raise SpotSourceReconciliationError("LSEG latest query SHA-256 differs")
    parameters = build_lseg_epex_actuals_latest_parameters(
        window_start_utc=window_start_utc,
        window_end_utc=window_end_utc,
        assessed_at_utc=assessed_at_utc,
    )
    result = _validate_lseg_extract_rows(
        frame,
        window_start_utc=parameters["start_utc"],
        window_end_utc=parameters["end_utc"],
        cutoff_utc=parameters["assessed_at_utc"],
        label="LSEG EPEX latest extract",
    )
    audit = {
        "schema_version": LSEG_LATEST_AUDIT_SCHEMA,
        "status": "PASS_LSEG_BOUNDED_LATEST_EXTRACT_NOT_PIT_OR_MODEL_AUTHORITY",
        "latest_query_sha256": LSEG_LATEST_SQL_SHA256,
        "row_count": len(result),
        "frame_semantic_sha256": dataframe_semantic_sha256(result),
        "window_start_utc": parameters["start_utc"],
        "window_end_utc": parameters["end_utc"],
        "assessed_at_utc": parameters["assessed_at_utc"],
        "curve_binding": {
            zone: {"curve_id": curve, "resolution": resolution}
            for zone, (curve, resolution) in LSEG_CURVES.items()
        },
        "authorities": {
            "latest_observation_validated": True,
            "point_in_time_filter_validated": False,
            "cadence_completeness_authorized": False,
            "model_input_authorized": False,
            "model_selection_authorized": False,
            "production_authorized": False,
        },
    }
    return LsegActualsLatestExtract(result, audit)


def reconcile_lseg_entsoe_spot(
    *,
    entsoe: DayAheadPitExtract,
    lseg: LsegActualsPitExtract,
    policy: SpotReconciliationPolicy,
) -> SpotReconciliationReport:
    """Compare independently admitted extracts at complete UTC-hour grain."""

    if not isinstance(policy, SpotReconciliationPolicy):
        raise SpotSourceReconciliationError("reconciliation policy has the wrong type")
    entsoe_frame, entsoe_audit = _validated_entsoe_artifact(entsoe)
    lseg_frame, lseg_audit = _validated_lseg_artifact(lseg)
    return _reconcile_validated_frames(
        entsoe_frame=entsoe_frame,
        entsoe_audit=entsoe_audit,
        lseg_frame=lseg_frame,
        lseg_audit=lseg_audit,
        policy=policy,
    )


def reconcile_lseg_entsoe_latest_candidate(
    *,
    entsoe_raw_frame: pd.DataFrame,
    entsoe: DayAheadConsumerExport,
    lseg: LsegActualsLatestExtract,
    policy: SpotReconciliationPolicy,
) -> SpotReconciliationReport:
    """Compare LSEG with an exactly replayed, explicitly non-final ENTSO-E candidate."""

    if not isinstance(policy, SpotReconciliationPolicy):
        raise SpotSourceReconciliationError("reconciliation policy has the wrong type")
    entsoe_frame, entsoe_audit = _validated_entsoe_latest_candidate(entsoe_raw_frame, entsoe)
    lseg_frame, lseg_audit = _validated_lseg_latest_artifact(lseg)
    return _reconcile_validated_frames(
        entsoe_frame=entsoe_frame,
        entsoe_audit=entsoe_audit,
        lseg_frame=lseg_frame,
        lseg_audit=lseg_audit,
        policy=policy,
    )


def _reconcile_validated_frames(
    *,
    entsoe_frame: pd.DataFrame,
    entsoe_audit: Mapping[str, object],
    lseg_frame: pd.DataFrame,
    lseg_audit: Mapping[str, object],
    policy: SpotReconciliationPolicy,
) -> SpotReconciliationReport:
    audit_keys = ("window_start_utc", "window_end_utc", "as_of_utc")
    if any(entsoe_audit.get(key) != lseg_audit.get(key) for key in audit_keys):
        raise SpotSourceReconciliationError("LSEG and ENTSO-E audit windows do not match")

    start = _utc_timestamp(entsoe_audit["window_start_utc"], "reconciliation start")
    end = _utc_timestamp(entsoe_audit["window_end_utc"], "reconciliation end")
    if start.floor("h") != start or end.floor("h") != end:
        raise SpotSourceReconciliationError("reconciliation bounds must be whole UTC hours")
    expected_hours = int((end - start) / pd.Timedelta(hours=1))
    if expected_hours < 1:
        raise SpotSourceReconciliationError("reconciliation window contains no UTC hour")

    entsoe_native = entsoe_frame.loc[
        :,
        [
            "field_name",
            "interval_start_utc",
            "interval_end_utc",
            "resolution",
            "price_eur_per_mwh",
        ],
    ].rename(columns={"field_name": "market_zone"})
    entsoe_native["market_zone"] = entsoe_native["market_zone"].map(ENTSOE_FIELDS)
    entsoe_native = _expand_normalized_entsoe_blocks(entsoe_native)
    lseg_native = lseg_frame.loc[
        :,
        [
            "market_zone",
            "interval_start_utc",
            "interval_end_utc",
            "price_eur_per_mwh",
        ],
    ]

    findings: list[SpotFinding] = []
    entsoe_hourly, entsoe_invalid = _complete_hourly(entsoe_native, "ENTSOE")
    lseg_hourly, lseg_invalid = _complete_hourly(lseg_native, "LSEG")
    _append_incomplete_findings(findings, entsoe_invalid, "ENTSOE")
    _append_incomplete_findings(findings, lseg_invalid, "LSEG")

    expected_index = pd.date_range(start=start, end=end, freq="h", inclusive="left")
    per_zone: dict[str, dict[str, object]] = {}
    for zone in LSEG_CURVES:
        entsoe_zone = entsoe_hourly.loc[
            entsoe_hourly["market_zone"].eq(zone),
            [
                "hour_start_utc",
                "hourly_price_eur_per_mwh",
            ],
        ].rename(columns={"hourly_price_eur_per_mwh": "entsoe_price"})
        lseg_zone = lseg_hourly.loc[
            lseg_hourly["market_zone"].eq(zone),
            [
                "hour_start_utc",
                "hourly_price_eur_per_mwh",
            ],
        ].rename(columns={"hourly_price_eur_per_mwh": "lseg_price"})
        aligned = (
            pd.DataFrame({"hour_start_utc": expected_index})
            .merge(entsoe_zone, on="hour_start_utc", how="left", validate="one_to_one")
            .merge(lseg_zone, on="hour_start_utc", how="left", validate="one_to_one")
        )
        matched = aligned["entsoe_price"].notna() & aligned["lseg_price"].notna()
        differences = aligned.loc[matched, "lseg_price"] - aligned.loc[matched, "entsoe_price"]
        absolute = differences.abs()
        matched_hours = int(matched.sum())
        overlap_ratio = matched_hours / expected_hours
        metrics = {
            "expected_hours": expected_hours,
            "matched_hours": matched_hours,
            "entsoe_missing_hours": int(aligned["entsoe_price"].isna().sum()),
            "lseg_missing_hours": int(aligned["lseg_price"].isna().sum()),
            "hourly_overlap_ratio": overlap_ratio,
            "mean_difference_lseg_minus_entsoe_eur_per_mwh": _mean_or_none(differences),
            "median_abs_difference_eur_per_mwh": _quantile_or_none(absolute, 0.5),
            "p95_abs_difference_eur_per_mwh": _quantile_or_none(absolute, 0.95),
            "maximum_abs_difference_eur_per_mwh": _max_or_none(absolute),
            "rmse_eur_per_mwh": _rmse_or_none(differences),
        }
        per_zone[zone] = metrics
        _assess_zone_metrics(zone, metrics, policy, findings)

    it_nord_hours = int(
        entsoe_hourly.loc[entsoe_hourly["market_zone"].eq("IT_NORD"), "hour_start_utc"]
        .isin(expected_index)
        .sum()
    )
    if it_nord_hours != expected_hours:
        findings.append(
            SpotFinding(
                "ENTSOE_IT_NORD_HOURLY_COVERAGE_INCOMPLETE",
                "IT-NORD has no LSEG substitute and its ENTSO-E UTC-hour coverage is incomplete.",
                "IT_NORD",
                expected_hours - it_nord_hours,
            )
        )

    metrics = {
        "policy": policy.as_dict(),
        "policy_semantic_sha256": policy.semantic_sha256,
        "window_start_utc": _utc_text(start),
        "window_end_utc": _utc_text(end),
        "as_of_utc": entsoe_audit["as_of_utc"],
        "expected_utc_hours_per_zone": expected_hours,
        "entsoe_frame_semantic_sha256": entsoe_audit["frame_semantic_sha256"],
        "lseg_frame_semantic_sha256": lseg_audit["frame_semantic_sha256"],
        "per_zone": per_zone,
        "it_nord": {
            "expected_hours": expected_hours,
            "entsoe_complete_hours": it_nord_hours,
            "lseg_crosscheck_available": False,
            "status": "ENTSOE_ONLY_NO_ACTIVE_LSEG_EPEX_CURVE",
        },
        "severity_counts": {
            severity: sum(item.severity == severity for item in findings)
            for severity in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
        },
    }
    status = BLOCKED_STATUS if findings else PASS_STATUS
    return SpotReconciliationReport(status, tuple(findings), metrics)


def _validated_entsoe_artifact(
    value: object,
) -> tuple[pd.DataFrame, Mapping[str, object]]:
    if not isinstance(value, DayAheadPitExtract) or not isinstance(value.audit, Mapping):
        raise SpotSourceReconciliationError("ENTSO-E artifact has the wrong type")
    frame = _exact_frame(value.frame, PIT_COLUMNS, "ENTSO-E PIT artifact")
    audit = value.audit
    try:
        revalidated = validate_day_ahead_pit_extract(
            frame,
            series_selection=audit.get("series_selection"),
            window_start_utc=audit.get("window_start_utc"),
            window_end_utc=audit.get("window_end_utc"),
            as_of_utc=audit.get("as_of_utc"),
            pit_query_sha256=audit.get("pit_query_sha256"),
        )
    except (EntsoeDayAheadPrdError, TypeError) as exc:
        raise SpotSourceReconciliationError("ENTSO-E PIT audit binding is invalid") from exc
    if dict(revalidated.audit) != dict(audit):
        raise SpotSourceReconciliationError("ENTSO-E PIT audit binding is invalid")
    return revalidated.frame, revalidated.audit


def _validated_entsoe_latest_candidate(
    raw_frame: pd.DataFrame,
    value: object,
) -> tuple[pd.DataFrame, Mapping[str, object]]:
    if not isinstance(value, DayAheadConsumerExport) or not isinstance(value.audit, Mapping):
        raise SpotSourceReconciliationError("ENTSO-E latest candidate has the wrong type")
    audit = value.audit
    try:
        revalidated = validate_realized_latest_candidate(
            raw_frame,
            series_selection=audit.get("series_selection"),
            market_uses=audit.get("market_uses"),
            window_start_utc=audit.get("window_start_utc"),
            window_end_utc=audit.get("window_end_utc"),
            assessed_at_utc=audit.get("temporal_cutoff_utc"),
            query_sha256=audit.get("query_sha256"),
        )
    except (EntsoeDayAheadExportError, TypeError) as exc:
        raise SpotSourceReconciliationError(
            "ENTSO-E latest candidate audit binding is invalid"
        ) from exc
    try:
        pd.testing.assert_frame_equal(
            revalidated.frame,
            value.frame,
            check_dtype=True,
            check_exact=True,
            check_like=False,
            check_freq=False,
        )
    except AssertionError as exc:
        raise SpotSourceReconciliationError(
            "ENTSO-E latest candidate audit binding is invalid"
        ) from exc
    if dict(revalidated.audit) != dict(audit):
        raise SpotSourceReconciliationError("ENTSO-E latest candidate audit binding is invalid")
    frame = revalidated.frame.rename(columns={"native_resolution": "resolution"})
    reconciliation_audit = {
        "window_start_utc": audit["window_start_utc"],
        "window_end_utc": audit["window_end_utc"],
        "as_of_utc": audit["temporal_cutoff_utc"],
        "frame_semantic_sha256": audit["consumer_frame_semantic_sha256"],
    }
    return frame, reconciliation_audit


def _validated_lseg_artifact(
    value: object,
) -> tuple[pd.DataFrame, Mapping[str, object]]:
    if not isinstance(value, LsegActualsPitExtract) or not isinstance(value.audit, Mapping):
        raise SpotSourceReconciliationError("LSEG artifact has the wrong type")
    frame = _exact_frame(value.frame, LSEG_PIT_COLUMNS, "LSEG PIT artifact")
    audit = value.audit
    try:
        revalidated = validate_lseg_epex_actuals_pit_extract(
            frame,
            window_start_utc=audit.get("window_start_utc"),
            window_end_utc=audit.get("window_end_utc"),
            as_of_utc=audit.get("as_of_utc"),
            pit_query_sha256=audit.get("pit_query_sha256"),
        )
    except (SpotSourceReconciliationError, TypeError) as exc:
        raise SpotSourceReconciliationError("LSEG PIT audit binding is invalid") from exc
    if dict(revalidated.audit) != dict(audit):
        raise SpotSourceReconciliationError("LSEG PIT audit binding is invalid")
    return revalidated.frame, revalidated.audit


def _validated_lseg_latest_artifact(
    value: object,
) -> tuple[pd.DataFrame, Mapping[str, object]]:
    if not isinstance(value, LsegActualsLatestExtract) or not isinstance(value.audit, Mapping):
        raise SpotSourceReconciliationError("LSEG latest artifact has the wrong type")
    frame = _exact_frame(value.frame, LSEG_PIT_COLUMNS, "LSEG latest artifact")
    audit = value.audit
    try:
        revalidated = validate_lseg_epex_actuals_latest_extract(
            frame,
            window_start_utc=audit.get("window_start_utc"),
            window_end_utc=audit.get("window_end_utc"),
            assessed_at_utc=audit.get("assessed_at_utc"),
            latest_query_sha256=audit.get("latest_query_sha256"),
        )
    except (SpotSourceReconciliationError, TypeError) as exc:
        raise SpotSourceReconciliationError("LSEG latest audit binding is invalid") from exc
    if dict(revalidated.audit) != dict(audit):
        raise SpotSourceReconciliationError("LSEG latest audit binding is invalid")
    reconciliation_audit = {
        "window_start_utc": audit["window_start_utc"],
        "window_end_utc": audit["window_end_utc"],
        "as_of_utc": audit["assessed_at_utc"],
        "frame_semantic_sha256": audit["frame_semantic_sha256"],
    }
    return revalidated.frame, reconciliation_audit


def _expand_normalized_entsoe_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    """Repeat normalized Silver price blocks at their declared native cadence."""

    expanded: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        resolution_seconds = _RESOLUTION_SECONDS.get(row.resolution)
        if resolution_seconds is None:
            raise SpotSourceReconciliationError(
                "ENTSO-E normalized block resolution is unsupported"
            )
        resolution = pd.Timedelta(seconds=resolution_seconds)
        duration = row.interval_end_utc - row.interval_start_utc
        if duration <= pd.Timedelta(0) or duration % resolution:
            raise SpotSourceReconciliationError(
                "ENTSO-E normalized block is not a native-resolution multiple"
            )
        for offset in range(int(duration / resolution)):
            item = row._asdict()
            point_start = row.interval_start_utc + offset * resolution
            item["interval_start_utc"] = point_start
            item["interval_end_utc"] = point_start + resolution
            expanded.append(item)
    if not expanded:
        raise SpotSourceReconciliationError("ENTSO-E normalized-block expansion is empty")
    return pd.DataFrame(expanded, columns=frame.columns)


def _complete_hourly(frame: pd.DataFrame, source: str) -> tuple[pd.DataFrame, Mapping[str, int]]:
    result = frame.copy()
    result["hour_start_utc"] = result["interval_start_utc"].dt.floor("h")
    result["duration_seconds"] = (
        result["interval_end_utc"] - result["interval_start_utc"]
    ).dt.total_seconds()
    valid_grid = (
        result["interval_start_utc"].dt.minute.mod(15).eq(0)
        & result["interval_start_utc"].dt.second.eq(0)
        & result["interval_start_utc"].dt.microsecond.eq(0)
        & result["interval_end_utc"].le(result["hour_start_utc"] + pd.Timedelta(hours=1))
    )
    invalid_groups: dict[str, int] = {}
    for zone, rows in result.groupby("market_zone", sort=False):
        invalid_hours: set[pd.Timestamp] = set(
            rows.loc[~valid_grid.loc[rows.index], "hour_start_utc"].tolist()
        )
        ordered = rows.sort_values(
            ["hour_start_utc", "interval_start_utc", "interval_end_utc"], kind="mergesort"
        )
        previous_end = ordered.groupby("hour_start_utc")["interval_end_utc"].shift()
        overlap = ordered["interval_start_utc"].lt(previous_end)
        invalid_hours.update(ordered.loc[overlap, "hour_start_utc"].tolist())
        coverage = rows.groupby("hour_start_utc")["duration_seconds"].sum()
        invalid_hours.update(coverage.index[coverage.ne(3_600)].tolist())
        if invalid_hours:
            invalid_groups[str(zone)] = len(invalid_hours)
    invalid_pairs = {
        (zone, hour)
        for zone, rows in result.groupby("market_zone", sort=False)
        for hour in rows["hour_start_utc"].unique()
        if str(zone) in invalid_groups
        and _hour_is_invalid(rows.loc[rows["hour_start_utc"].eq(hour)], hour)
    }
    keep = [
        (row.market_zone, row.hour_start_utc) not in invalid_pairs
        for row in result.itertuples(index=False)
    ]
    complete = result.loc[keep].copy()
    complete["weighted_price"] = complete["price_eur_per_mwh"] * complete["duration_seconds"]
    hourly = (
        complete.groupby(["market_zone", "hour_start_utc"], as_index=False, sort=True)
        .agg(weighted_price=("weighted_price", "sum"))
        .assign(hourly_price_eur_per_mwh=lambda values: values["weighted_price"] / 3_600)
        .drop(columns="weighted_price")
    )
    if hourly.duplicated(["market_zone", "hour_start_utc"]).any():
        raise SpotSourceReconciliationError(f"{source} hourly aggregation is duplicated")
    return hourly, invalid_groups


def _hour_is_invalid(rows: pd.DataFrame, hour: object) -> bool:
    if rows.empty:
        return False
    hour_start = pd.Timestamp(hour)
    valid_grid = (
        rows["interval_start_utc"].dt.minute.mod(15).eq(0)
        & rows["interval_start_utc"].dt.second.eq(0)
        & rows["interval_start_utc"].dt.microsecond.eq(0)
        & rows["interval_end_utc"].le(hour_start + pd.Timedelta(hours=1))
    )
    ordered = rows.sort_values(["interval_start_utc", "interval_end_utc"], kind="mergesort")
    overlaps = ordered["interval_start_utc"].lt(ordered["interval_end_utc"].shift()).any()
    return not valid_grid.all() or overlaps or rows["duration_seconds"].sum() != 3_600


def _append_incomplete_findings(
    findings: list[SpotFinding], invalid: Mapping[str, int], source: str
) -> None:
    for zone, count in invalid.items():
        findings.append(
            SpotFinding(
                f"{source}_NATIVE_INTERVALS_DO_NOT_FORM_COMPLETE_UTC_HOURS",
                f"{source} intervals contain gaps, overlaps or cross-hour rows.",
                zone,
                count,
            )
        )


def _assess_zone_metrics(
    zone: str,
    metrics: Mapping[str, object],
    policy: SpotReconciliationPolicy,
    findings: list[SpotFinding],
) -> None:
    matched = int(metrics["matched_hours"])
    if matched < policy.minimum_matched_hours_per_zone:
        findings.append(
            SpotFinding(
                "CROSS_SOURCE_MATCHED_HOURS_BELOW_POLICY",
                "Matched UTC hours are below the explicitly frozen policy minimum.",
                zone,
                policy.minimum_matched_hours_per_zone - matched,
            )
        )
    if float(metrics["hourly_overlap_ratio"]) < policy.minimum_hourly_overlap_ratio:
        findings.append(
            SpotFinding(
                "CROSS_SOURCE_HOURLY_OVERLAP_BELOW_POLICY",
                "Cross-source UTC-hour overlap is below policy.",
                zone,
                int(metrics["expected_hours"]) - matched,
                "HIGH",
            )
        )
    checks = (
        (
            "p95_abs_difference_eur_per_mwh",
            policy.maximum_p95_abs_difference_eur_per_mwh,
            "CROSS_SOURCE_P95_ABS_DIFFERENCE_ABOVE_POLICY",
        ),
        (
            "maximum_abs_difference_eur_per_mwh",
            policy.maximum_single_hour_abs_difference_eur_per_mwh,
            "CROSS_SOURCE_SINGLE_HOUR_ABS_DIFFERENCE_ABOVE_POLICY",
        ),
    )
    for metric, threshold, code in checks:
        value = metrics[metric]
        if value is not None and float(value) > threshold:
            findings.append(
                SpotFinding(
                    code, f"{metric} exceeds the frozen policy threshold.", zone, severity="HIGH"
                )
            )
    bias = metrics["mean_difference_lseg_minus_entsoe_eur_per_mwh"]
    if bias is not None and abs(float(bias)) > policy.maximum_absolute_bias_eur_per_mwh:
        findings.append(
            SpotFinding(
                "CROSS_SOURCE_ABSOLUTE_BIAS_ABOVE_POLICY",
                "Absolute LSEG-minus-ENTSO-E hourly bias exceeds policy.",
                zone,
                severity="HIGH",
            )
        )


def _mean_or_none(values: pd.Series) -> float | None:
    return None if values.empty else float(values.mean())


def _quantile_or_none(values: pd.Series, quantile: float) -> float | None:
    return None if values.empty else float(values.quantile(quantile))


def _max_or_none(values: pd.Series) -> float | None:
    return None if values.empty else float(values.max())


def _rmse_or_none(values: pd.Series) -> float | None:
    return None if values.empty else math.sqrt(float((values**2).mean()))


def _exact_frame(frame: object, columns: tuple[str, ...], label: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise SpotSourceReconciliationError(f"{label} must be a DataFrame")
    if frame.columns.has_duplicates or tuple(frame.columns) != columns:
        raise SpotSourceReconciliationError(f"{label} columns are not exact")
    return frame.copy()


def _sha256(value: object, label: str) -> str:
    text = _text(value, label)
    if _SHA256.fullmatch(text) is None:
        raise SpotSourceReconciliationError(f"{label} must be lowercase SHA-256")
    return text


def _text(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or _CLEAN_TEXT.fullmatch(value) is None
    ):
        raise SpotSourceReconciliationError(f"{label} must be clean text")
    return value


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpotSourceReconciliationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SpotSourceReconciliationError(f"{label} must be finite")
    return result


def _finite_nonnegative(value: object, label: str) -> float:
    result = _finite_float(value, label)
    if result < 0:
        raise SpotSourceReconciliationError(f"{label} must be nonnegative")
    return result


def _utc_timestamp(value: object, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise SpotSourceReconciliationError(f"{label} is invalid") from exc
    if timestamp.tzinfo is None:
        raise SpotSourceReconciliationError(f"{label} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _ordered_window(start: pd.Timestamp, end: pd.Timestamp, label: str) -> None:
    if start >= end:
        raise SpotSourceReconciliationError(f"{label} window is empty or inverted")


def _utc_text(value: pd.Timestamp) -> str:
    return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")


__all__ = [
    "BLOCKED_STATUS",
    "ENTSOE_FIELDS",
    "LSEG_CURVES",
    "LSEG_LATEST_SQL_PATH",
    "LSEG_LATEST_SQL_SHA256",
    "LSEG_PIT_COLUMNS",
    "LSEG_PIT_SQL_PATH",
    "LSEG_PIT_SQL_SHA256",
    "LsegActualsPitExtract",
    "LsegActualsLatestExtract",
    "PASS_STATUS",
    "SpotFinding",
    "SpotReconciliationPolicy",
    "SpotReconciliationReport",
    "SpotSourceReconciliationError",
    "build_lseg_epex_actuals_pit_parameters",
    "build_lseg_epex_actuals_latest_parameters",
    "reconcile_lseg_entsoe_latest_candidate",
    "reconcile_lseg_entsoe_spot",
    "validate_lseg_epex_actuals_pit_extract",
    "validate_lseg_epex_actuals_latest_extract",
    "verify_lseg_latest_sql_binding",
    "verify_lseg_sql_binding",
]
