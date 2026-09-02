"""Deterministic LT consumption contract for ENTSO-E day-ahead prices.

This module is deliberately downstream of acquisition.  It does not query a
lakehouse and it does not choose an auction by convention.  It turns an
explicitly selected, provenance-rich interval inventory into either final
realized truth or causally available observations, expands producer-normalized
blocks at their native cadence, and can derive only a duration-weighted
zero-mean monthly shape.  None of these operations has monthly-level or
production authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from pfc_shaping.data.governed_lt_acquisition import dataframe_semantic_sha256


class SpotUsage(str, Enum):
    """Economically distinct uses of a day-ahead observation."""

    REALIZED_FINAL = "realized_final"
    CAUSAL_ASOF = "causal_asof"


class AvailabilityBasis(str, Enum):
    """Upstream ENTSO-E availability bases retained without substitution."""

    FMV_FIRST_SEEN = "FMV_FIRST_SEEN"
    SOURCE_DOCUMENT_CREATED = "SOURCE_DOCUMENT_CREATED"
    UNKNOWN_BACKFILL = "UNKNOWN_BACKFILL"


class IndependentControl(str, Enum):
    """Independent selection evidence for one effective-dated series rule."""

    LSEG_RECONCILIATION_PASSED = "LSEG_RECONCILIATION_PASSED"
    ENTSOE_ONLY_NO_LSEG_CURVE = "ENTSOE_ONLY_NO_LSEG_CURVE"


FIELD_TO_ZONE: Mapping[str, str] = {
    "ch_price": "CH",
    "at_price": "AT",
    "de_lu_price": "DE_LU",
    "fr_price": "FR",
    "it_nord_price": "IT_NORD",
}
MARKET_TIMEZONES: Mapping[str, str] = {
    "CH": "Europe/Zurich",
    "AT": "Europe/Vienna",
    "DE_LU": "Europe/Berlin",
    "FR": "Europe/Paris",
    "IT_NORD": "Europe/Rome",
}
LSEG_CONTROL_ZONES = frozenset({"CH", "AT", "DE_LU", "FR"})
RESOLUTION_SECONDS: Mapping[str, int] = {
    "PT15M": 900,
    "PT30M": 1_800,
    "PT60M": 3_600,
    "PT1H": 3_600,
}

SOURCE_COLUMNS = (
    "field_name",
    "market_zone",
    "market_timezone",
    "series_key",
    "classification_sequence",
    "interval_start_utc",
    "interval_end_utc",
    "native_resolution",
    "price_eur_per_mwh",
    "publication_timestamp_utc",
    "first_seen_pull_ts_utc",
    "availability_basis",
    "original_publication_proven",
    "is_historical",
    "is_final",
    "dq_failed",
    "quality_status",
    "source_time_series_id",
    "source_document_mrid",
    "source_document_revision_number",
    "source_snapshot_id",
    "source_file_path",
)

OUTPUT_COLUMNS = (
    "field_name",
    "market_zone",
    "market_timezone",
    "series_key",
    "classification_sequence",
    "interval_start_utc",
    "interval_end_utc",
    "interval_start_market_time",
    "interval_end_market_time",
    "native_resolution",
    "source_interval_start_utc",
    "source_interval_end_utc",
    "price_eur_per_mwh",
    "publication_timestamp_utc",
    "first_seen_pull_ts_utc",
    "availability_basis",
    "availability_timestamp_utc",
    "availability_mode",
    "original_publication_proven",
    "is_historical",
    "is_final",
    "dq_failed",
    "quality_status",
    "source_time_series_id",
    "source_document_mrid",
    "source_document_revision_number",
    "source_snapshot_id",
    "source_file_path",
)

_CLEAN_TEXT = re.compile(r"^[^\x00-\x1f\x7f]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DayAheadConsumptionError(ValueError):
    """Typed, audit-friendly refusal of an unsafe consumption request."""

    def __init__(self, code: str, message: str, **context: object) -> None:
        self.code = code
        self.context = dict(context)
        suffix = "" if not context else f"; context={self.context}"
        super().__init__(f"{code}: {message}{suffix}")


@dataclass(frozen=True)
class EffectiveDatedSeriesRule:
    """One proven source-semantic series choice over a UTC validity interval."""

    field_name: str
    market_zone: str
    series_key: str
    classification_sequence: str | None
    effective_start_utc: str | pd.Timestamp
    effective_end_utc: str | pd.Timestamp
    source_semantics: str
    selection_evidence_id: str
    selection_evidence_sha256: str
    independent_control: IndependentControl

    def normalized(self) -> dict[str, object]:
        field = _text(self.field_name, "field name")
        zone = _text(self.market_zone, "market zone")
        if FIELD_TO_ZONE.get(field) != zone:
            _fail("SERIES_RULE_ZONE_MISMATCH", "field and market zone do not match", field=field)
        sequence = _optional_text(self.classification_sequence, "classification sequence")
        if field in {"at_price", "de_lu_price"} and sequence not in {"1", "2"}:
            _fail(
                "MULTI_AUCTION_SEQUENCE_UNPROVEN",
                "AT and DE-LU require an explicit classification sequence 1 or 2",
                field=field,
            )
        if field not in {"at_price", "de_lu_price"} and sequence is not None:
            _fail(
                "UNEXPECTED_CLASSIFICATION_SEQUENCE",
                "this market field must not carry a classification sequence",
                field=field,
            )
        expected_key = _canonical_series_key(field, sequence)
        if self.series_key != expected_key:
            _fail(
                "SERIES_RULE_KEY_NONCANONICAL",
                "effective-dated rule does not bind the canonical SeriesKey",
                field=field,
            )
        start = _utc_timestamp(self.effective_start_utc, "rule start")
        end = _utc_timestamp(self.effective_end_utc, "rule end")
        if start >= end:
            _fail("SERIES_RULE_INTERVAL_INVALID", "rule interval is empty or inverted")
        semantics = _text(self.source_semantics, "source semantics")
        evidence_id = _text(self.selection_evidence_id, "selection evidence ID")
        evidence_sha = _sha256(self.selection_evidence_sha256, "selection evidence SHA-256")
        try:
            control = IndependentControl(self.independent_control)
        except ValueError as exc:
            raise DayAheadConsumptionError(
                "SERIES_CONTROL_INVALID", "independent control status is not recognized"
            ) from exc
        expected_control = (
            IndependentControl.LSEG_RECONCILIATION_PASSED
            if zone in LSEG_CONTROL_ZONES
            else IndependentControl.ENTSOE_ONLY_NO_LSEG_CURVE
        )
        if control is not expected_control:
            _fail(
                "SERIES_CONTROL_UNPROVEN",
                "the required independent LSEG control status is absent",
                field=field,
                expected=expected_control.value,
            )
        return {
            "field_name": field,
            "market_zone": zone,
            "series_key": expected_key,
            "classification_sequence": sequence,
            "effective_start_utc": start,
            "effective_end_utc": end,
            "source_semantics": semantics,
            "selection_evidence_id": evidence_id,
            "selection_evidence_sha256": evidence_sha,
            "independent_control": control.value,
        }


@dataclass(frozen=True)
class DayAheadConsumption:
    """Expanded day-ahead observations plus non-authoritative audit evidence."""

    frame: pd.DataFrame
    audit: Mapping[str, object]


def materialize_day_ahead_consumption(
    source: pd.DataFrame,
    *,
    usage: SpotUsage | str,
    series_rules: Sequence[EffectiveDatedSeriesRule],
    window_start_utc: str | pd.Timestamp,
    window_end_utc: str | pd.Timestamp,
    as_of_utc: str | pd.Timestamp | None = None,
    required_fields: Sequence[str] = tuple(FIELD_TO_ZONE),
) -> DayAheadConsumption:
    """Select, validate and expand normalized Silver intervals fail closed."""

    try:
        selected_usage = SpotUsage(usage)
    except ValueError as exc:
        raise DayAheadConsumptionError(
            "USAGE_MODE_INVALID", "spot usage mode is not recognized"
        ) from exc
    start = _utc_timestamp(window_start_utc, "window start")
    end = _utc_timestamp(window_end_utc, "window end")
    if start >= end:
        _fail("WINDOW_INVALID", "delivery window is empty or inverted")
    if any(value.second or value.microsecond or value.minute % 15 for value in (start, end)):
        _fail("WINDOW_OFF_GRID", "delivery bounds must lie on a 15-minute UTC grid")
    required = _required_fields(required_fields)
    rules = _normalize_rules(series_rules, required, start, end)
    frame = _normalize_source(source)
    selected = _apply_series_rules(frame, rules, start, end)
    selected = _apply_usage(selected, selected_usage, as_of_utc)
    expanded = _expand_intervals(selected, start, end, required)
    expanded = (
        expanded.loc[:, OUTPUT_COLUMNS]
        .sort_values(["field_name", "interval_start_utc", "interval_end_utc"], kind="mergesort")
        .reset_index(drop=True)
    )
    rule_payload = [_json_rule(rule) for rule in rules]
    audit = {
        "schema_version": "fmv_entsoe_day_ahead_consumption.v2",
        "status": f"PASS_{selected_usage.value.upper()}_NOT_MONTHLY_LEVEL_AUTHORITY",
        "usage": selected_usage.value,
        "window_start_utc": _utc_text(start),
        "window_end_utc": _utc_text(end),
        "as_of_utc": (
            _utc_text(_utc_timestamp(as_of_utc, "as-of")) if as_of_utc is not None else None
        ),
        "required_fields": list(required),
        "source_row_count": len(selected),
        "expanded_row_count": len(expanded),
        "frame_semantic_sha256": dataframe_semantic_sha256(expanded),
        "series_rules": rule_payload,
        "series_rules_sha256": hashlib.sha256(
            json.dumps(rule_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "interval_policy": {
            "source_contract": "producer_normalized_half_open_utc_interval",
            "duration": "positive_integer_multiple_of_native_resolution",
            "expansion": "repeat_price_at_native_resolution_without_curve_type_inference",
            "coverage": "no_overlap_and_no_gap_inside_requested_utc_window",
            "dst": "UTC_INTERVALS_WITH_EXPLICIT_MARKET_TIMEZONE_NO_24_HOUR_ASSUMPTION",
        },
        "authorities": {
            "realized_truth_authorized": selected_usage is SpotUsage.REALIZED_FINAL,
            "point_in_time_filter_validated": selected_usage is SpotUsage.CAUSAL_ASOF,
            "monthly_level_authorized": False,
            "model_input_authorized": False,
            "model_selection_authorized": False,
            "production_authorized": False,
        },
    }
    return DayAheadConsumption(expanded, audit)


def build_monthly_zero_mean_spot_shape(
    consumption: DayAheadConsumption,
    *,
    monthly_level_authority: str,
    tolerance_eur_per_mwh: float = 1e-10,
) -> pd.DataFrame:
    """Return a duration-weighted local-month shape without changing solver means."""

    if monthly_level_authority != "solver":
        _fail(
            "MONTHLY_LEVEL_AUTHORITY_INVALID",
            "spot shape construction requires monthly_level_authority='solver'",
        )
    if not isinstance(consumption, DayAheadConsumption):
        _fail("CONSUMPTION_TYPE_INVALID", "consumption artifact has the wrong type")
    if consumption.audit.get("frame_semantic_sha256") != dataframe_semantic_sha256(
        consumption.frame
    ):
        _fail("CONSUMPTION_AUDIT_MISMATCH", "consumption frame does not match its audit")
    if (
        isinstance(tolerance_eur_per_mwh, bool)
        or not isinstance(tolerance_eur_per_mwh, (int, float))
        or not math.isfinite(float(tolerance_eur_per_mwh))
        or tolerance_eur_per_mwh <= 0
    ):
        _fail("ZERO_MEAN_TOLERANCE_INVALID", "zero-mean tolerance must be positive and finite")
    frame = consumption.frame.copy()
    frame["duration_seconds"] = (
        frame["interval_end_utc"] - frame["interval_start_utc"]
    ).dt.total_seconds()
    frame["solver_month_local"] = [
        timestamp.tz_convert(timezone).strftime("%Y-%m")
        for timestamp, timezone in zip(
            frame["interval_start_utc"], frame["market_timezone"], strict=True
        )
    ]
    group_columns = ["field_name", "market_zone", "solver_month_local"]
    weighted = frame["price_eur_per_mwh"] * frame["duration_seconds"]
    numerator = weighted.groupby([frame[column] for column in group_columns]).transform("sum")
    denominator = (
        frame["duration_seconds"]
        .groupby([frame[column] for column in group_columns])
        .transform("sum")
    )
    frame["shape_eur_per_mwh"] = frame["price_eur_per_mwh"] - numerator / denominator
    residual = (frame["shape_eur_per_mwh"] * frame["duration_seconds"]).groupby(
        [frame[column] for column in group_columns]
    ).sum() / denominator.groupby([frame[column] for column in group_columns]).first()
    if residual.abs().max() > float(tolerance_eur_per_mwh):
        _fail("MONTHLY_ZERO_MEAN_FAILED", "spot shape is not zero mean inside solver month")
    return frame


def _normalize_source(source: object) -> pd.DataFrame:
    if not isinstance(source, pd.DataFrame):
        _fail("SOURCE_TYPE_INVALID", "source must be a DataFrame")
    if source.columns.has_duplicates or tuple(source.columns) != SOURCE_COLUMNS:
        _fail("SOURCE_SCHEMA_INVALID", "source columns are not exact")
    if source.empty:
        _fail("SOURCE_EMPTY", "source contains no day-ahead rows")
    frame = source.copy()
    text_columns = (
        "field_name",
        "market_zone",
        "market_timezone",
        "series_key",
        "native_resolution",
        "availability_basis",
        "quality_status",
        "source_time_series_id",
        "source_document_mrid",
        "source_snapshot_id",
        "source_file_path",
    )
    for column in text_columns:
        frame[column] = frame[column].map(lambda value: _text(value, column))
    frame["classification_sequence"] = frame["classification_sequence"].map(
        lambda value: _optional_text(value, "classification sequence")
    )
    for column in (
        "interval_start_utc",
        "interval_end_utc",
        "publication_timestamp_utc",
        "first_seen_pull_ts_utc",
    ):
        frame[column] = frame[column].map(lambda value: _optional_utc_timestamp(value, column))
    for column in ("original_publication_proven", "is_historical", "is_final", "dq_failed"):
        if not frame[column].map(lambda value: type(value) is bool).all():
            _fail("SOURCE_BOOLEAN_INVALID", "source boolean is not exact", column=column)
    revisions = pd.to_numeric(frame["source_document_revision_number"], errors="coerce")
    if revisions.isna().any() or (revisions < 0).any() or (revisions % 1 != 0).any():
        _fail("SOURCE_REVISION_INVALID", "document revision must be a nonnegative integer")
    frame["source_document_revision_number"] = revisions.astype(int)
    prices = pd.to_numeric(frame["price_eur_per_mwh"], errors="coerce")
    if prices.isna().any() or not prices.map(lambda value: math.isfinite(float(value))).all():
        _fail("SOURCE_PRICE_INVALID", "day-ahead price must be finite")
    frame["price_eur_per_mwh"] = prices.astype(float)
    for row in frame.itertuples(index=False):
        if FIELD_TO_ZONE.get(row.field_name) != row.market_zone:
            _fail("SOURCE_ZONE_MISMATCH", "source field and zone do not match")
        if MARKET_TIMEZONES[row.market_zone] != row.market_timezone:
            _fail("SOURCE_TIMEZONE_MISMATCH", "market timezone is not canonical")
        if row.series_key != _canonical_series_key(row.field_name, row.classification_sequence):
            _fail("SOURCE_SERIES_KEY_NONCANONICAL", "source SeriesKey is not canonical")
        if row.native_resolution not in RESOLUTION_SECONDS:
            _fail("RESOLUTION_UNSUPPORTED", "native resolution is unsupported")
        if row.interval_start_utc is None or row.interval_end_utc is None:
            _fail("INTERVAL_BOUND_MISSING", "interval bounds must be present")
        if row.interval_end_utc <= row.interval_start_utc:
            _fail("INTERVAL_NONPOSITIVE", "interval duration must be strictly positive")
        if row.dq_failed or row.quality_status != "PASSED":
            _fail("SOURCE_QUALITY_FAILED", "day-ahead source row failed quality")
        try:
            AvailabilityBasis(row.availability_basis)
        except ValueError as exc:
            raise DayAheadConsumptionError(
                "AVAILABILITY_BASIS_INVALID", "availability basis is not recognized"
            ) from exc
    duplicate = frame.duplicated(
        [
            "field_name",
            "series_key",
            "interval_start_utc",
            "interval_end_utc",
            "source_document_revision_number",
        ],
        keep=False,
    )
    if duplicate.any():
        _fail("SOURCE_VINTAGE_DUPLICATED", "source contains duplicate vintage grain")
    return frame


def _normalize_rules(
    series_rules: Sequence[EffectiveDatedSeriesRule],
    required: tuple[str, ...],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[dict[str, object]]:
    if isinstance(series_rules, (str, bytes)) or not isinstance(series_rules, Sequence):
        _fail("SERIES_RULES_INVALID", "series rules must be a sequence")
    rules: list[dict[str, object]] = []
    for value in series_rules:
        if not isinstance(value, EffectiveDatedSeriesRule):
            _fail("SERIES_RULE_TYPE_INVALID", "series rule has the wrong type")
        rules.append(value.normalized())
    for field in required:
        field_rules = sorted(
            [rule for rule in rules if rule["field_name"] == field],
            key=lambda rule: rule["effective_start_utc"],
        )
        cursor = start
        for rule in field_rules:
            rule_start = max(start, rule["effective_start_utc"])
            rule_end = min(end, rule["effective_end_utc"])
            if rule_start >= rule_end:
                continue
            if rule_start < cursor:
                _fail("SERIES_RULE_OVERLAP", "effective-dated series rules overlap", field=field)
            if rule_start > cursor:
                _fail("SERIES_RULE_GAP", "effective-dated series rules leave a gap", field=field)
            cursor = rule_end
        if cursor != end:
            _fail(
                "SERIES_RULE_GAP",
                "effective-dated series rules do not cover the window",
                field=field,
            )
    unexpected = sorted(set(rule["field_name"] for rule in rules) - set(required))
    if unexpected:
        _fail("SERIES_RULE_FIELD_UNEXPECTED", "rules contain unexpected fields", fields=unexpected)
    return rules


def _apply_series_rules(
    frame: pd.DataFrame,
    rules: Sequence[Mapping[str, object]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    chosen: list[pd.DataFrame] = []
    for rule in rules:
        lower = max(start, rule["effective_start_utc"])
        upper = min(end, rule["effective_end_utc"])
        if lower >= upper:
            continue
        candidates = frame.loc[
            frame["field_name"].eq(rule["field_name"])
            & frame["series_key"].eq(rule["series_key"])
            & frame["interval_start_utc"].lt(upper)
            & frame["interval_end_utc"].gt(lower)
        ]
        contained = candidates["interval_start_utc"].ge(lower) & candidates["interval_end_utc"].le(
            upper
        )
        if (~contained).any():
            _fail(
                "SERIES_INTERVAL_CROSSES_RULE_BOUNDARY",
                "source interval crosses an effective-dated series boundary",
                field=rule["field_name"],
            )
        rows = candidates.loc[contained]
        chosen.append(rows)
    if not chosen:
        _fail("SERIES_SELECTION_EMPTY", "no source row matches the proven series rules")
    result = pd.concat(chosen, ignore_index=True)
    if result.empty:
        _fail("SERIES_SELECTION_EMPTY", "no source row matches the proven series rules")
    return result


def _apply_usage(
    frame: pd.DataFrame,
    usage: SpotUsage,
    as_of_utc: str | pd.Timestamp | None,
) -> pd.DataFrame:
    result = frame.copy()
    if usage is SpotUsage.REALIZED_FINAL:
        if as_of_utc is not None:
            _fail("REALIZED_FINAL_ASOF_FORBIDDEN", "realized_final must not carry an as-of")
        if not result["is_final"].all():
            _fail("REALIZED_FINAL_UNPROVEN", "realized_final contains a non-final source row")
        result["availability_timestamp_utc"] = pd.NaT
        result["availability_mode"] = SpotUsage.REALIZED_FINAL.value
        return result
    if as_of_utc is None:
        _fail("CAUSAL_ASOF_REQUIRED", "causal_asof requires an explicit as-of timestamp")
    as_of = _utc_timestamp(as_of_utc, "as-of")
    availability: list[pd.Timestamp] = []
    for row in result.itertuples(index=False):
        basis = AvailabilityBasis(row.availability_basis)
        if basis is AvailabilityBasis.UNKNOWN_BACKFILL:
            _fail(
                "CAUSAL_AVAILABILITY_UNPROVEN",
                "unknown historical backfill availability cannot become a PIT feature",
                series_key=row.series_key,
            )
        if basis is AvailabilityBasis.FMV_FIRST_SEEN:
            timestamp = row.first_seen_pull_ts_utc
            if timestamp is None:
                _fail("FIRST_SEEN_MISSING", "FMV_FIRST_SEEN basis has no first-seen timestamp")
        else:
            if not row.original_publication_proven:
                _fail(
                    "ORIGINAL_PUBLICATION_UNPROVEN",
                    "createdDateTime is not proven original day-ahead publication time",
                    series_key=row.series_key,
                )
            timestamp = row.publication_timestamp_utc
            if timestamp is None:
                _fail("PUBLICATION_TIMESTAMP_MISSING", "publication basis has no timestamp")
            if timestamp >= row.interval_start_utc:
                _fail(
                    "PUBLICATION_NOT_DAY_AHEAD",
                    "proven day-ahead publication must precede delivery",
                    series_key=row.series_key,
                )
        if timestamp > as_of:
            _fail(
                "VALUE_NOT_AVAILABLE_AT_ASOF",
                "selected value was not available at the observation time",
                series_key=row.series_key,
            )
        availability.append(timestamp)
    result["availability_timestamp_utc"] = availability
    result["availability_mode"] = SpotUsage.CAUSAL_ASOF.value
    return result


def _expand_intervals(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    required: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_row in frame.itertuples(index=False):
        interval_start = source_row.interval_start_utc
        interval_end = source_row.interval_end_utc
        duration_ns = interval_end.value - interval_start.value
        resolution = RESOLUTION_SECONDS[source_row.native_resolution]
        resolution_ns = resolution * 1_000_000_000
        if duration_ns <= 0:
            _fail("INTERVAL_NONPOSITIVE", "interval duration must be strictly positive")
        if duration_ns % resolution_ns:
            _fail(
                "INTERVAL_NOT_RESOLUTION_MULTIPLE",
                "interval duration is not an integer multiple of native resolution",
                series_key=source_row.series_key,
            )
        if interval_start.value % resolution_ns or interval_end.value % resolution_ns:
            _fail(
                "INTERVAL_OFF_NATIVE_GRID",
                "interval bounds are not aligned to native resolution",
                series_key=source_row.series_key,
            )
        periods = duration_ns // resolution_ns
        for offset in range(periods):
            point_start = interval_start + pd.Timedelta(seconds=resolution * offset)
            point_end = point_start + pd.Timedelta(seconds=resolution)
            if point_end <= start or point_start >= end:
                continue
            if point_start < start or point_end > end:
                _fail("WINDOW_PARTIAL_INTERVAL", "delivery window cuts a native interval")
            item = source_row._asdict()
            item.update(
                {
                    "interval_start_utc": point_start,
                    "interval_end_utc": point_end,
                    "source_interval_start_utc": interval_start,
                    "source_interval_end_utc": interval_end,
                    "interval_start_market_time": point_start.tz_convert(
                        source_row.market_timezone
                    ).isoformat(),
                    "interval_end_market_time": point_end.tz_convert(
                        source_row.market_timezone
                    ).isoformat(),
                }
            )
            rows.append(item)
    if not rows:
        _fail("EXPANSION_EMPTY", "no expanded interval lies inside the delivery window")
    result = pd.DataFrame(rows)
    for field in required:
        ordered = result.loc[result["field_name"].eq(field)].sort_values(
            ["interval_start_utc", "interval_end_utc"], kind="mergesort"
        )
        if ordered.empty:
            _fail("FIELD_COVERAGE_MISSING", "selected field has no interval", field=field)
        cursor = start
        for row in ordered.itertuples(index=False):
            if row.interval_start_utc < cursor:
                _fail("INTERVAL_OVERLAP", "expanded intervals overlap", field=field)
            if row.interval_start_utc > cursor:
                _fail(
                    "INTERVAL_GAP",
                    "true gap remains after normalized-block expansion",
                    field=field,
                )
            cursor = row.interval_end_utc
        if cursor != end:
            _fail(
                "INTERVAL_GAP", "expanded intervals do not cover the delivery window", field=field
            )
    return result


def _required_fields(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        _fail("REQUIRED_FIELDS_INVALID", "required fields must be a sequence")
    result = tuple(_text(value, "required field") for value in values)
    if not result or len(result) != len(set(result)) or not set(result).issubset(FIELD_TO_ZONE):
        _fail("REQUIRED_FIELDS_INVALID", "required fields are empty, duplicate or unknown")
    return result


def _canonical_series_key(field: str, classification: str | None) -> str:
    base = f"day_ahead_prices||{field}"
    return f"{base}||{classification}" if classification is not None else base


def _json_rule(rule: Mapping[str, object]) -> dict[str, object]:
    return {
        key: _utc_text(value) if isinstance(value, pd.Timestamp) else value
        for key, value in rule.items()
    }


def _utc_timestamp(value: object, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DayAheadConsumptionError("TIMESTAMP_INVALID", f"{label} is invalid") from exc
    if timestamp.tzinfo is None:
        _fail("TIMESTAMP_NAIVE", f"{label} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _optional_utc_timestamp(value: object, label: str) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    return _utc_timestamp(value, label)


def _text(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or _CLEAN_TEXT.fullmatch(value) is None
    ):
        _fail("TEXT_INVALID", f"{label} must be clean text")
    return value


def _optional_text(value: object, label: str) -> str | None:
    if value is None or pd.isna(value):
        return None
    return _text(value, label)


def _sha256(value: object, label: str) -> str:
    text = _text(value, label)
    if _SHA256.fullmatch(text) is None:
        _fail("SHA256_INVALID", f"{label} must be lowercase SHA-256")
    return text


def _utc_text(value: pd.Timestamp) -> str:
    return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _fail(code: str, message: str, **context: object) -> None:
    raise DayAheadConsumptionError(code, message, **context)


__all__ = [
    "AvailabilityBasis",
    "DayAheadConsumption",
    "DayAheadConsumptionError",
    "EffectiveDatedSeriesRule",
    "FIELD_TO_ZONE",
    "IndependentControl",
    "MARKET_TIMEZONES",
    "OUTPUT_COLUMNS",
    "SOURCE_COLUMNS",
    "SpotUsage",
    "build_monthly_zero_mean_spot_shape",
    "materialize_day_ahead_consumption",
]
