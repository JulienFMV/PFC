"""Semantic acceptance for governed Databricks LT extracts.

Gold is the serving authority for current data. ENTSO-E Silver vintages are
the point-in-time authority because the optimized Gold model deliberately no
longer duplicates the historical vintage fact. Bronze remains diagnostic only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

REPORT_SCHEMA = "fmv_databricks_pfc_layer_acceptance.v2"

DEFAULT_REQUIRED_GROUPS = (
    "day_ahead_prices",
    "actual_load",
    "load_forecast",
    "renewables_forecast_day_ahead",
    "renewables_forecast_intraday",
    "generation_actual",
    "generation_forecast",
    "hydro_reservoir_storage",
    "scheduled_exchanges",
    "crossborder_physical_flows",
    "ntc_day_ahead",
    "ntc_month_ahead",
    "ntc_year_ahead",
    "ntc_intraday",
)

REQUIRED_BORDER_GROUPS = (
    "scheduled_exchanges",
    "crossborder_physical_flows",
    "ntc_day_ahead",
    "ntc_month_ahead",
    "ntc_year_ahead",
    "ntc_intraday",
)

CH_NEIGHBORS = ("AT", "DE_LU", "FR", "IT_NORD")

PRICE_GROUPS = frozenset(("day_ahead_prices", "imbalance_prices"))
POWER_GROUPS = frozenset(
    (
        "actual_load",
        "load_forecast",
        "renewables_forecast_day_ahead",
        "renewables_forecast_intraday",
        "generation_actual",
        "generation_forecast",
        "scheduled_exchanges",
        "net_position_day_ahead",
        "crossborder_physical_flows",
        "ntc_day_ahead",
        "ntc_week_ahead",
        "ntc_month_ahead",
        "ntc_year_ahead",
        "ntc_intraday",
        "production_unit_unavailability",
        "generation_unit_unavailability",
        "transmission_unavailability",
        "installed_capacity_by_type",
        "installed_capacity_per_unit",
    )
)
ENERGY_GROUPS = frozenset(("hydro_reservoir_storage", "imbalance_volumes"))
OUTAGE_GROUPS = frozenset(
    (
        "production_unit_unavailability",
        "generation_unit_unavailability",
        "transmission_unavailability",
    )
)

GOLD_REQUIRED_COLUMNS = {
    "entsoe_series_dimension": frozenset(
        (
            "SeriesID",
            "SeriesKey",
            "FieldName",
            "SourceTimeSeriesId",
            "GroupName",
            "DocumentType",
            "BusinessType",
            "ProcessType",
            "PsrType",
            "FromZone",
            "ToZone",
            "Unit",
        )
    ),
    "entsoe_latest": frozenset(
        (
            "SeriesID",
            "IntervalStartUtc",
            "DateTimeUtc",
            "Resolution",
            "FieldValue",
            "PublicationTimestampUtc",
            "AvailabilityBasis",
            "AvailabilityKnown",
            "AvailabilityTimestampUtc",
            "IsHistorical",
        )
    ),
    "spot_price_interval": frozenset(
        (
            "SpotProductID",
            "SourceProduct",
            "MarketZone",
            "DeliveryStartUtc",
            "DeliveryEndUtc",
            "FrequencyMinutes",
            "Price",
            "PriceUnit",
            "ObservedAtUtc",
        )
    ),
}

SILVER_VINTAGE_COLUMNS = frozenset(
    (
        "SK_ge_power_entsoe_time_series_vintages",
        "series_key",
        "field_value",
        "IntervalStartUtc",
        "Date_Time_UTC",
        "IntervalEndUtc",
        "resolution",
        "publication_timestamp_utc",
        "Is_Historical",
        "resource_details",
        "first_seen_pull_ts_utc",
        "last_seen_pull_ts_utc",
        "availability_basis",
        "availability_known",
        "availability_timestamp_utc",
        "source_document_mrid",
        "source_document_revision_number",
        "dq_failed",
    )
)

BRIDGE_COLUMNS = frozenset(
    ("BridgeID", "SeriesID", "ResourceRole", "ResourceMrid")
)

_SILVER_CANONICAL_NAMES = {
    "SK_ge_power_entsoe_time_series_vintages": "VintageID",
    "series_key": "SeriesKey",
    "field_value": "FieldValue",
    "Date_Time_UTC": "DateTimeUtc",
    "resolution": "Resolution",
    "publication_timestamp_utc": "PublicationTimestampUtc",
    "Is_Historical": "IsHistorical",
    "first_seen_pull_ts_utc": "FirstObservedAtUtc",
    "last_seen_pull_ts_utc": "LastObservedAtUtc",
    "availability_basis": "AvailabilityBasis",
    "availability_known": "AvailabilityKnown",
    "availability_timestamp_utc": "AvailabilityTimestampUtc",
    "source_document_mrid": "SourceDocumentMRID",
    "source_document_revision_number": "SourceDocumentRevisionNumber",
}

_RESOLUTION = re.compile(
    r"^PT(?:(?P<hours>[1-9]\d*)H)?(?:(?P<minutes>[1-9]\d*)M)?$"
)


class DatabricksPfcLayerAcceptanceError(ValueError):
    """Raised when local assessment inputs are structurally malformed."""


@dataclass(frozen=True)
class LayerFinding:
    severity: str
    code: str
    role: str
    layer: str
    message: str
    affected_rows: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "code": self.code,
            "role": self.role,
            "layer": self.layer,
            "message": self.message,
            "affected_rows": self.affected_rows,
        }


@dataclass(frozen=True)
class PfcLayerAcceptanceReport:
    status: str
    findings: tuple[LayerFinding, ...]
    metrics: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": REPORT_SCHEMA,
            "status": self.status,
            "findings": [finding.as_dict() for finding in self.findings],
            "metrics": dict(self.metrics),
            "layer_policy": {
                "current_serving_authority": "GOLD",
                "entsoe_point_in_time_authority": (
                    "SILVER_GE_POWER_ENTSOE_TIME_SERIES_VINTAGES"
                ),
                "entsoe_resource_bridge_role": "OPTIONAL_CURRENT_ENRICHMENT",
                "bronze_role": "SOURCE_ADJUDICATION_ONLY",
                "lseg_role": "BENCHMARK_ONLY",
            },
            "authorities": {
                "semantic_acceptance_passed": self.status
                == "PASS_GOVERNED_LAYERS",
                "model_input_authorized": False,
                "production_authorized": False,
            },
        }


def assess_pfc_data_layers(
    *,
    gold: Mapping[str, pd.DataFrame],
    silver: Mapping[str, pd.DataFrame] | None = None,
    bronze: Mapping[str, pd.DataFrame] | None = None,
    required_start_utc: str | pd.Timestamp = "2019-01-01T00:00:00Z",
    required_groups: Sequence[str] = DEFAULT_REQUIRED_GROUPS,
) -> PfcLayerAcceptanceReport:
    """Assess the mixed Gold-serving and ENTSO-E-Silver-PIT contract."""

    gold_frames = _frames(gold, "Gold")
    silver_frames = _frames(silver or {}, "Silver")
    bronze_frames = _frames(bronze or {}, "Bronze")
    required_start = _utc_scalar(required_start_utc, "required_start_utc")
    findings: list[LayerFinding] = []

    for role, required in GOLD_REQUIRED_COLUMNS.items():
        frame = gold_frames.get(role)
        if frame is None:
            findings.append(
                LayerFinding(
                    "CRITICAL",
                    "GOLD_ROLE_MISSING",
                    role,
                    "gold",
                    f"Required Gold role is missing: {role}",
                )
            )
            continue
        _require_columns(findings, frame, required, role=role, layer="gold")

    silver_vintages = silver_frames.get("entsoe_vintages")
    if silver_vintages is None:
        findings.append(
            LayerFinding(
                "CRITICAL",
                "ENTSOE_SILVER_VINTAGES_MISSING",
                "entsoe_vintages",
                "silver",
                "Canonical ENTSO-E Silver vintages are missing.",
            )
        )
    else:
        _require_columns(
            findings,
            silver_vintages,
            SILVER_VINTAGE_COLUMNS,
            role="entsoe_vintages",
            layer="silver",
        )

    dimension = gold_frames.get("entsoe_series_dimension")
    latest = gold_frames.get("entsoe_latest")
    spot = gold_frames.get("spot_price_interval")
    bridge = gold_frames.get("entsoe_series_resources_current")

    if dimension is not None:
        _assess_dimension(findings, dimension, required_groups)
    if latest is not None:
        _assess_latest(findings, latest, dimension, required_start)
    if silver_vintages is not None and SILVER_VINTAGE_COLUMNS.issubset(
        silver_vintages.columns
    ):
        _assess_vintages(
            findings,
            _canonical_silver_vintages(silver_vintages),
            dimension,
            required_start,
        )
    if spot is not None:
        _assess_spot(findings, spot)
    if bridge is not None:
        _require_columns(
            findings,
            bridge,
            BRIDGE_COLUMNS,
            role="entsoe_series_resources_current",
            layer="gold",
        )
        _assess_bridge(findings, bridge, dimension)
    elif dimension is not None and "GroupName" in dimension:
        outage_rows = int(dimension["GroupName"].isin(OUTAGE_GROUPS).sum())
        if outage_rows:
            findings.append(
                LayerFinding(
                    "LOW",
                    "ENTSOE_CURRENT_RESOURCE_BRIDGE_MISSING",
                    "entsoe_series_resources_current",
                    "gold",
                    "Current outage resource enrichment is unavailable; the core PFC remains usable.",
                    affected_rows=outage_rows,
                )
            )

    severity_counts = {
        severity: sum(finding.severity == severity for finding in findings)
        for severity in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
    }
    if severity_counts["CRITICAL"]:
        status = "BLOCKED_GOVERNED_LAYERS"
    elif severity_counts["HIGH"]:
        status = "REVIEW_GOVERNED_LAYERS"
    else:
        status = "PASS_GOVERNED_LAYERS"
    metrics = {
        "gold_role_count": len(gold_frames),
        "silver_role_count": len(silver_frames),
        "bronze_diagnostic_role_count": len(bronze_frames),
        "row_count_by_gold_role": {
            role: len(frame) for role, frame in sorted(gold_frames.items())
        },
        "row_count_by_silver_role": {
            role: len(frame) for role, frame in sorted(silver_frames.items())
        },
        "severity_counts": severity_counts,
        "required_start_utc": required_start.isoformat().replace("+00:00", "Z"),
        "required_groups": list(required_groups),
    }
    return PfcLayerAcceptanceReport(status, tuple(findings), metrics)


def _frames(
    value: Mapping[str, pd.DataFrame], label: str
) -> dict[str, pd.DataFrame]:
    if not isinstance(value, Mapping):
        raise DatabricksPfcLayerAcceptanceError(f"{label} roles must be a mapping")
    result: dict[str, pd.DataFrame] = {}
    for role, frame in value.items():
        if not isinstance(role, str) or not role.strip() or role != role.strip():
            raise DatabricksPfcLayerAcceptanceError(f"{label} role name is invalid")
        if not isinstance(frame, pd.DataFrame):
            raise DatabricksPfcLayerAcceptanceError(
                f"{label} role is not a DataFrame: {role}"
            )
        if frame.columns.has_duplicates:
            raise DatabricksPfcLayerAcceptanceError(
                f"{label} role has duplicate columns: {role}"
            )
        result[role] = frame.copy()
    return result


def _require_columns(
    findings: list[LayerFinding],
    frame: pd.DataFrame,
    required: frozenset[str],
    *,
    role: str,
    layer: str,
) -> None:
    for column in sorted(required.difference(frame.columns)):
        findings.append(
            LayerFinding(
                "CRITICAL",
                f"{layer.upper()}_REQUIRED_COLUMN_MISSING",
                role,
                layer,
                f"{layer.title()} column is missing: {role}.{column}",
            )
        )


def _canonical_silver_vintages(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.rename(columns=_SILVER_CANONICAL_NAMES).copy()
    if result.columns.has_duplicates:
        raise DatabricksPfcLayerAcceptanceError(
            "Silver vintage canonical projection has duplicate columns"
        )
    return result


def _assess_dimension(
    findings: list[LayerFinding],
    frame: pd.DataFrame,
    required_groups: Sequence[str],
) -> None:
    role = "entsoe_series_dimension"
    if "SeriesID" in frame:
        _duplicates(
            findings,
            frame,
            role,
            ("SeriesID",),
            "SERIES_ID_DUPLICATED",
            "gold",
        )
        _nulls(findings, frame, role, "SeriesID", "gold")
    if "SeriesKey" in frame:
        _duplicates(
            findings,
            frame,
            role,
            ("SeriesKey",),
            "SERIES_KEY_DUPLICATED",
            "gold",
        )
        _nulls(findings, frame, role, "SeriesKey", "gold")
    if "GroupName" not in frame:
        return
    groups = set(frame["GroupName"].dropna().astype(str))
    missing_groups = sorted(set(required_groups).difference(groups))
    if missing_groups:
        findings.append(
            LayerFinding(
                "CRITICAL",
                "ENTSOE_REQUIRED_GROUPS_MISSING",
                role,
                "gold",
                f"Required ENTSO-E groups are missing: {missing_groups}",
                affected_rows=len(missing_groups),
            )
        )
    if {"GroupName", "Unit"}.issubset(frame.columns):
        expected_units = {
            **{group: "EUR/MWh" for group in PRICE_GROUPS},
            **{group: "MW" for group in POWER_GROUPS},
            **{group: "MWh" for group in ENERGY_GROUPS},
        }
        expected = frame["GroupName"].map(expected_units)
        invalid = expected.notna() & frame["Unit"].ne(expected)
        count = int(invalid.sum())
        if count:
            findings.append(
                LayerFinding(
                    "CRITICAL",
                    "ENTSOE_UNIT_MISMATCH",
                    role,
                    "gold",
                    "Gold units differ from configured ENTSO-E family semantics.",
                    affected_rows=count,
                )
            )
    if {"GroupName", "FromZone", "ToZone"}.issubset(frame.columns):
        expected_directions = {
            direction
            for neighbor in CH_NEIGHBORS
            for direction in (("CH", neighbor), (neighbor, "CH"))
        }
        for group in REQUIRED_BORDER_GROUPS:
            rows = frame.loc[frame["GroupName"].eq(group)]
            observed = set(
                zip(
                    rows["FromZone"].astype("string"),
                    rows["ToZone"].astype("string"),
                    strict=True,
                )
            )
            missing = expected_directions.difference(observed)
            if missing:
                findings.append(
                    LayerFinding(
                        "CRITICAL",
                        "ENTSOE_BORDER_DIRECTIONS_MISSING",
                        role,
                        "gold",
                        f"{group} misses CH border directions: {sorted(missing)}",
                        affected_rows=len(missing),
                    )
                )
    if "installed_capacity_per_unit" in groups:
        rows = frame.loc[frame["GroupName"].eq("installed_capacity_per_unit")]
        if "RegisteredResourceMrid" not in rows or rows[
            "RegisteredResourceMrid"
        ].isna().any():
            findings.append(
                LayerFinding(
                    "CRITICAL",
                    "ENTSOE_REGISTERED_RESOURCE_ID_MISSING",
                    role,
                    "gold",
                    "Installed capacity per unit requires RegisteredResourceMrid.",
                )
            )


def _assess_latest(
    findings: list[LayerFinding],
    frame: pd.DataFrame,
    dimension: pd.DataFrame | None,
    required_start: pd.Timestamp,
) -> None:
    role = "entsoe_latest"
    if {"SeriesID", "IntervalStartUtc", "DateTimeUtc"}.issubset(frame.columns):
        _duplicates(
            findings,
            frame,
            role,
            ("SeriesID", "IntervalStartUtc", "DateTimeUtc"),
            "ENTSOE_LATEST_GRAIN_DUPLICATED",
            "gold",
        )
    _referential_integrity(
        findings,
        frame,
        dimension,
        role=role,
        layer="gold",
        fact_key="SeriesID",
        dimension_key="SeriesID",
    )
    _assess_entsoe_intervals(findings, frame, role, "gold")
    _coverage_start(findings, frame, role, required_start, "gold")


def _assess_vintages(
    findings: list[LayerFinding],
    frame: pd.DataFrame,
    dimension: pd.DataFrame | None,
    required_start: pd.Timestamp,
) -> None:
    role = "entsoe_vintages"
    _duplicates(
        findings,
        frame,
        role,
        ("VintageID",),
        "VINTAGE_ID_DUPLICATED",
        "silver",
    )
    _referential_integrity(
        findings,
        frame,
        dimension,
        role=role,
        layer="silver",
        fact_key="SeriesKey",
        dimension_key="SeriesKey",
    )
    _assess_entsoe_intervals(findings, frame, role, "silver")
    _coverage_start(findings, frame, role, required_start, "silver")

    timestamp_columns = (
        "PublicationTimestampUtc",
        "FirstObservedAtUtc",
        "LastObservedAtUtc",
    )
    parsed = {
        column: _utc_series(findings, frame, role, column, "silver")
        for column in timestamp_columns
    }
    invalid = (
        parsed["PublicationTimestampUtc"] > parsed["FirstObservedAtUtc"]
    ) | (parsed["FirstObservedAtUtc"] > parsed["LastObservedAtUtc"])
    count = int(invalid.fillna(False).sum())
    if count:
        findings.append(
            LayerFinding(
                "CRITICAL",
                "VINTAGE_TIMESTAMP_ORDER_INVALID",
                role,
                "silver",
                "Expected Publication <= FirstObserved <= LastObserved is violated.",
                affected_rows=count,
            )
        )

    basis = frame["AvailabilityBasis"].astype("string")
    known = frame["AvailabilityKnown"].astype("boolean")
    availability = _utc_series(
        findings, frame, role, "AvailabilityTimestampUtc", "silver"
    )
    unknown = basis.eq("UNKNOWN_BACKFILL")
    invalid_unknown = unknown & (known.fillna(False) | availability.notna())
    recognized = basis.isin(
        ("UNKNOWN_BACKFILL", "FMV_FIRST_SEEN", "SOURCE_DOCUMENT_CREATED")
    )
    invalid_known = (~unknown) & (~known.fillna(False) | availability.isna())
    invalid_semantics = (~recognized) | invalid_unknown | invalid_known
    count_invalid = int(invalid_semantics.fillna(True).sum())
    if count_invalid:
        findings.append(
            LayerFinding(
                "CRITICAL",
                "VINTAGE_AVAILABILITY_SEMANTICS_INVALID",
                role,
                "silver",
                "Availability basis, known flag and timestamp are inconsistent.",
                affected_rows=count_invalid,
            )
        )
    count_unknown = int(unknown.sum())
    if count_unknown:
        findings.append(
            LayerFinding(
                "HIGH",
                "VINTAGE_PIT_AVAILABILITY_UNKNOWN",
                role,
                "silver",
                "Historical backfill rows cannot support known-at backtests.",
                affected_rows=count_unknown,
            )
        )
    if "dq_failed" in frame:
        count_dq = int(frame["dq_failed"].fillna(True).astype(bool).sum())
        if count_dq:
            findings.append(
                LayerFinding(
                    "CRITICAL",
                    "ENTSOE_SILVER_DQ_FAILED",
                    role,
                    "silver",
                    "Silver vintages contain rows flagged as blocking DQ failures.",
                    affected_rows=count_dq,
                )
            )


def _assess_bridge(
    findings: list[LayerFinding],
    frame: pd.DataFrame,
    dimension: pd.DataFrame | None,
) -> None:
    role = "entsoe_series_resources_current"
    if BRIDGE_COLUMNS.issubset(frame.columns):
        _duplicates(
            findings,
            frame,
            role,
            ("BridgeID",),
            "ENTSOE_BRIDGE_ID_DUPLICATED",
            "gold",
        )
        _referential_integrity(
            findings,
            frame,
            dimension,
            role=role,
            layer="gold",
            fact_key="SeriesID",
            dimension_key="SeriesID",
        )


def _assess_entsoe_intervals(
    findings: list[LayerFinding],
    frame: pd.DataFrame,
    role: str,
    layer: str,
) -> None:
    required = {"DateTimeUtc", "IntervalStartUtc", "IntervalEndUtc", "Resolution"}
    if not required.issubset(frame.columns):
        return
    right_edge = _utc_series(findings, frame, role, "DateTimeUtc", layer)
    start = _utc_series(findings, frame, role, "IntervalStartUtc", layer)
    end = _utc_series(findings, frame, role, "IntervalEndUtc", layer)
    seconds = frame["Resolution"].map(_resolution_seconds)
    invalid_resolution = seconds.isna()
    if int(invalid_resolution.sum()):
        findings.append(
            LayerFinding(
                "CRITICAL",
                "ENTSOE_RESOLUTION_INVALID",
                role,
                layer,
                "Resolution must be a positive ISO-8601 hour/minute duration.",
                affected_rows=int(invalid_resolution.sum()),
            )
        )
    duration = (end - start).dt.total_seconds()
    invalid = end.ne(right_edge) | start.ge(end) | duration.ne(seconds)
    count = int(invalid.fillna(True).sum())
    if count:
        findings.append(
            LayerFinding(
                "CRITICAL",
                "ENTSOE_INTERVAL_INCONSISTENT",
                role,
                layer,
                "Interval start/end, right edge and resolution are inconsistent.",
                affected_rows=count,
            )
        )


def _assess_spot(findings: list[LayerFinding], frame: pd.DataFrame) -> None:
    role = "spot_price_interval"
    if {"SpotProductID", "DeliveryStartUtc"}.issubset(frame.columns):
        _duplicates(
            findings,
            frame,
            role,
            ("SpotProductID", "DeliveryStartUtc"),
            "SPOT_INTERVAL_GRAIN_DUPLICATED",
            "gold",
        )
    if "PriceUnit" in frame:
        count = int(frame["PriceUnit"].ne("EUR/MWh").sum())
        if count:
            findings.append(
                LayerFinding(
                    "CRITICAL",
                    "SPOT_PRICE_UNIT_INVALID",
                    role,
                    "gold",
                    "Spot prices must be published as EUR/MWh.",
                    affected_rows=count,
                )
            )
    required = {"DeliveryStartUtc", "DeliveryEndUtc", "FrequencyMinutes"}
    if required.issubset(frame.columns):
        start = _utc_series(findings, frame, role, "DeliveryStartUtc", "gold")
        end = _utc_series(findings, frame, role, "DeliveryEndUtc", "gold")
        minutes = pd.to_numeric(frame["FrequencyMinutes"], errors="coerce")
        invalid = start.ge(end) | (end - start).dt.total_seconds().ne(minutes * 60)
        count = int(invalid.fillna(True).sum())
        if count:
            findings.append(
                LayerFinding(
                    "CRITICAL",
                    "SPOT_INTERVAL_INCONSISTENT",
                    role,
                    "gold",
                    "Spot interval duration differs from FrequencyMinutes.",
                    affected_rows=count,
                )
            )
    if {"QuotationTimestampUtc", "DeliveryStartUtc"}.issubset(frame.columns):
        quotation = _utc_series(
            findings, frame, role, "QuotationTimestampUtc", "gold"
        )
        delivery = _utc_series(findings, frame, role, "DeliveryStartUtc", "gold")
        non_null = quotation.notna() & delivery.notna()
        if bool(non_null.any()) and bool(quotation[non_null].eq(delivery[non_null]).all()):
            findings.append(
                LayerFinding(
                    "CRITICAL",
                    "SPOT_QUOTATION_EQUALS_DELIVERY_START",
                    role,
                    "gold",
                    "QuotationTimestampUtc is populated with the delivery start.",
                    affected_rows=int(non_null.sum()),
                )
            )


def _referential_integrity(
    findings: list[LayerFinding],
    frame: pd.DataFrame,
    dimension: pd.DataFrame | None,
    *,
    role: str,
    layer: str,
    fact_key: str,
    dimension_key: str,
) -> None:
    if (
        dimension is None
        or fact_key not in frame
        or dimension_key not in dimension
    ):
        return
    known = set(dimension[dimension_key].dropna())
    count = int((~frame[fact_key].isin(known)).sum())
    if count:
        findings.append(
            LayerFinding(
                "CRITICAL",
                "ENTSOE_ORPHAN_SERIES_KEY",
                role,
                layer,
                f"Rows reference {fact_key} values absent from Gold dimension {dimension_key}.",
                affected_rows=count,
            )
        )


def _coverage_start(
    findings: list[LayerFinding],
    frame: pd.DataFrame,
    role: str,
    required_start: pd.Timestamp,
    layer: str,
) -> None:
    if "DateTimeUtc" not in frame or frame.empty:
        return
    timestamps = _utc_series(findings, frame, role, "DateTimeUtc", layer)
    observed_start = timestamps.min()
    if pd.notna(observed_start) and observed_start > required_start + pd.Timedelta(days=1):
        findings.append(
            LayerFinding(
                "HIGH",
                "ENTSOE_HISTORY_STARTS_AFTER_2019",
                role,
                layer,
                f"Observed history starts at {observed_start.isoformat()}, after 2019-01-01.",
            )
        )


def _duplicates(
    findings: list[LayerFinding],
    frame: pd.DataFrame,
    role: str,
    columns: Sequence[str],
    code: str,
    layer: str,
) -> None:
    if not set(columns).issubset(frame.columns):
        return
    count = int(frame.duplicated(list(columns), keep=False).sum())
    if count:
        findings.append(
            LayerFinding(
                "CRITICAL",
                code,
                role,
                layer,
                f"{layer.title()} grain is duplicated on {list(columns)}.",
                affected_rows=count,
            )
        )


def _nulls(
    findings: list[LayerFinding],
    frame: pd.DataFrame,
    role: str,
    column: str,
    layer: str,
) -> None:
    count = int(frame[column].isna().sum())
    if count:
        findings.append(
            LayerFinding(
                "CRITICAL",
                f"{layer.upper()}_REQUIRED_VALUE_NULL",
                role,
                layer,
                f"Required values are null: {role}.{column}",
                affected_rows=count,
            )
        )


def _utc_series(
    findings: list[LayerFinding],
    frame: pd.DataFrame,
    role: str,
    column: str,
    layer: str,
) -> pd.Series:
    parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
    invalid = frame[column].notna() & parsed.isna()
    count = int(invalid.sum())
    if count:
        findings.append(
            LayerFinding(
                "CRITICAL",
                "TIMESTAMP_INVALID",
                role,
                layer,
                f"Timestamp values are invalid: {role}.{column}",
                affected_rows=count,
            )
        )
    return parsed


def _utc_scalar(value: str | pd.Timestamp, label: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DatabricksPfcLayerAcceptanceError(f"{label} is invalid") from exc
    if result.tzinfo is None:
        raise DatabricksPfcLayerAcceptanceError(f"{label} must be timezone-aware")
    return result.tz_convert("UTC")


def _resolution_seconds(value: object) -> float:
    if not isinstance(value, str):
        return float("nan")
    match = _RESOLUTION.fullmatch(value)
    if match is None or not any(match.groupdict().values()):
        return float("nan")
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    return float(hours * 3600 + minutes * 60)


__all__ = [
    "DEFAULT_REQUIRED_GROUPS",
    "DatabricksPfcLayerAcceptanceError",
    "LayerFinding",
    "PfcLayerAcceptanceReport",
    "REPORT_SCHEMA",
    "assess_pfc_data_layers",
]
