"""Deterministic local materialization of admitted Databricks LT extracts.

The functions in this module are deliberately offline. They accept already
exported DataFrames, require explicit ENTSO-E ``SeriesKey`` selections and
produce the canonical raw/derived frames consumed by the existing LT replay
layer. They do not connect to Databricks, publish snapshots or grant model
authority.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from pfc_shaping.data.governed_lt_acquisition import dataframe_semantic_sha256
from pfc_shaping.data.lt_replay_transforms import build_entso_features, clean_epex

MATERIALIZATION_SCHEMA_VERSION = "fmv_databricks_lt_materialization.v1"
MATERIALIZATION_METADATA_ATTR = "fmv_databricks_lt_materialization"
ENTSOE_FEATURE_MAPPING_SCHEMA_VERSION = "fmv_entsoe_feature_mapping.v1"

_DIMENSION_COLUMNS = frozenset(
    {
        "SeriesID",
        "SeriesKey",
        "SourceTimeSeriesId",
        "GroupName",
        "FieldName",
        "DocumentType",
        "BusinessType",
        "ProcessType",
        "PsrType",
        "FromZone",
        "ToZone",
        "Unit",
    }
)
_GOLD_LATEST_COLUMNS = frozenset(
    {
        "SeriesID",
        "IntervalStartUtc",
        "DateTimeUtc",
        "IntervalEndUtc",
        "Resolution",
        "FieldValue",
        "PublicationTimestampUtc",
        "AvailabilityBasis",
        "AvailabilityKnown",
        "AvailabilityTimestampUtc",
    }
)
_SILVER_VINTAGE_COLUMNS = frozenset(
    {
        "SK_ge_power_entsoe_time_series_vintages",
        "series_key",
        "field_value",
        "IntervalStartUtc",
        "Date_Time_UTC",
        "IntervalEndUtc",
        "resolution",
        "publication_timestamp_utc",
        "first_seen_pull_ts_utc",
        "last_seen_pull_ts_utc",
        "availability_basis",
        "availability_known",
        "availability_timestamp_utc",
        "source_document_revision_number",
        "source_document_mrid",
        "dq_failed",
    }
)
_SPOT_COLUMNS = frozenset(
    {
        "SpotProductID",
        "SourceProduct",
        "MarketZone",
        "DeliveryStartUtc",
        "DeliveryEndUtc",
        "FrequencyMinutes",
        "Price",
        "PriceUnit",
        "ObservedAtUtc",
    }
)
_RESOLUTION = re.compile(
    r"^PT(?:(?P<hours>[1-9]\d*)H)?(?:(?P<minutes>[1-9]\d*)M)?$"
)
_CORE_FEATURES = ("load_mw", "solar_mw", "wind_mw")
_FEATURE_POLICY = {
    "load_mw": {
        "groups": frozenset({"actual_load"}),
        "documents": frozenset({"A65"}),
        "processes": frozenset({"A16"}),
        "business_types": None,
        "psr_types": None,
    },
    "solar_mw": {
        "groups": frozenset({"generation_actual"}),
        "documents": frozenset({"A75"}),
        "processes": frozenset({"A16"}),
        "business_types": frozenset({"A01"}),
        "psr_types": frozenset({"B16"}),
    },
    "wind_mw": {
        "groups": frozenset({"generation_actual"}),
        "documents": frozenset({"A75"}),
        "processes": frozenset({"A16"}),
        "business_types": frozenset({"A01"}),
        "psr_types": frozenset({"B18", "B19"}),
    },
    "cross_border_mw": {
        "groups": frozenset({"crossborder_physical_flows"}),
        "documents": frozenset({"A11"}),
        "processes": frozenset({"A16"}),
        "business_types": None,
        "psr_types": None,
    },
}


class DatabricksLTMaterializationError(ValueError):
    """Raised when a local Databricks extract is ambiguous or incomplete."""


@dataclass(frozen=True)
class EntsoeSeriesTerm:
    """One explicitly selected ENTSO-E series and its aggregation sign."""

    series_key: str
    weight: float = 1.0


@dataclass(frozen=True)
class EntsoeFeatureMapping:
    """Exact SeriesKey mapping for the canonical ENTSO-E feature frame."""

    load_mw: tuple[EntsoeSeriesTerm, ...]
    solar_mw: tuple[EntsoeSeriesTerm, ...]
    wind_mw: tuple[EntsoeSeriesTerm, ...]
    cross_border_mw: tuple[EntsoeSeriesTerm, ...] = ()

    def as_dict(self) -> dict[str, list[dict[str, object]]]:
        return {
            feature: [
                {"series_key": term.series_key, "weight": float(term.weight)}
                for term in getattr(self, feature)
            ]
            for feature in (*_CORE_FEATURES, "cross_border_mw")
            if getattr(self, feature)
        }


@dataclass(frozen=True)
class DatabricksMaterialization:
    """Canonical model frames and their non-authoritative audit payload."""

    raw_frame: pd.DataFrame
    derived_frame: pd.DataFrame
    audit: Mapping[str, object]


def entsoe_dimension_semantic_sha256(dimension: pd.DataFrame) -> str:
    """Hash the exact dimension fields that give a SeriesKey its meaning."""

    _require_columns(dimension, _DIMENSION_COLUMNS, label="Gold ENTSO-E dimension")
    projection = (
        dimension.loc[:, sorted(_DIMENSION_COLUMNS)]
        .sort_values("SeriesKey", kind="mergesort")
        .reset_index(drop=True)
    )
    return dataframe_semantic_sha256(projection)


def entsoe_feature_mapping_from_contract(
    contract: Mapping[str, object],
    *,
    dimension: pd.DataFrame,
) -> EntsoeFeatureMapping:
    """Parse an exact mapping contract bound to one dimension semantics hash."""

    if not isinstance(contract, Mapping) or set(contract) != {
        "schema_version",
        "market",
        "dimension_semantic_sha256",
        "features",
    }:
        raise DatabricksLTMaterializationError(
            "ENTSO-E feature mapping contract fields are not exact"
        )
    if contract.get("schema_version") != ENTSOE_FEATURE_MAPPING_SCHEMA_VERSION:
        raise DatabricksLTMaterializationError(
            "ENTSO-E feature mapping contract schema is unsupported"
        )
    if contract.get("market") != "CH":
        raise DatabricksLTMaterializationError(
            "ENTSO-E feature mapping contract market must be CH"
        )
    expected_dimension_hash = entsoe_dimension_semantic_sha256(dimension)
    if contract.get("dimension_semantic_sha256") != expected_dimension_hash:
        raise DatabricksLTMaterializationError(
            "ENTSO-E feature mapping dimension hash mismatch"
        )
    features = contract.get("features")
    expected_features = {*_CORE_FEATURES, "cross_border_mw"}
    if not isinstance(features, Mapping) or set(features) != expected_features:
        raise DatabricksLTMaterializationError(
            "ENTSO-E feature mapping feature inventory is not exact"
        )

    parsed: dict[str, tuple[EntsoeSeriesTerm, ...]] = {}
    for feature in (*_CORE_FEATURES, "cross_border_mw"):
        raw_terms = features.get(feature)
        if not isinstance(raw_terms, list):
            raise DatabricksLTMaterializationError(
                f"ENTSO-E feature mapping terms must be a list: {feature}"
            )
        terms: list[EntsoeSeriesTerm] = []
        for raw_term in raw_terms:
            if not isinstance(raw_term, Mapping) or set(raw_term) != {
                "series_key",
                "weight",
            }:
                raise DatabricksLTMaterializationError(
                    f"ENTSO-E feature mapping term fields are not exact: {feature}"
                )
            try:
                term = EntsoeSeriesTerm(
                    series_key=str(raw_term["series_key"]),
                    weight=float(raw_term["weight"]),
                )
            except (TypeError, ValueError) as exc:
                raise DatabricksLTMaterializationError(
                    f"ENTSO-E feature mapping term is invalid: {feature}"
                ) from exc
            terms.append(term)
        parsed[feature] = tuple(terms)
    mapping = EntsoeFeatureMapping(**parsed)
    _validate_entsoe_mapping(dimension, mapping)
    return mapping


def materialize_entsoe_pit_features(
    *,
    dimension: pd.DataFrame,
    silver_vintages: pd.DataFrame,
    mapping: EntsoeFeatureMapping,
    as_of_utc: str | pd.Timestamp,
) -> DatabricksMaterialization:
    """Build an origin-safe ENTSO-E feature frame from Silver vintages."""

    origin = _utc_scalar(as_of_utc, label="ENTSO-E PIT origin")
    dimension_projection = _validate_entsoe_mapping(dimension, mapping)
    full_source = _canonical_silver_vintages(silver_vintages)
    selected_keys = set(dimension_projection["SeriesKey"].astype(str))
    source = full_source.loc[
        full_source["SeriesKey"].astype(str).isin(selected_keys)
    ].copy()
    if source.empty:
        raise DatabricksLTMaterializationError(
            "Silver vintages contain none of the explicitly mapped SeriesKeys"
        )
    source_projection_hash = dataframe_semantic_sha256(source)

    known = source["AvailabilityKnown"].astype("boolean")
    availability = _utc_series(
        source["AvailabilityTimestampUtc"],
        label="Silver availability",
        allow_missing=True,
    )
    _validate_availability_semantics(
        basis=source["AvailabilityBasis"],
        known=known,
        availability=availability,
        publication=source["PublicationTimestampUtc"],
        first_observed=source["FirstObservedAtUtc"],
        last_observed=source["LastObservedAtUtc"],
        label="Silver vintages",
    )
    dq_failed = source["dq_failed"].fillna(True).astype(bool)
    eligible_mask = known.fillna(False) & availability.le(origin) & ~dq_failed
    eligible = source.loc[eligible_mask].copy()
    eligible["AvailabilityTimestampUtc"] = availability.loc[eligible_mask]
    excluded_unknown = int((~known.fillna(False)).sum())
    excluded_future = int((known.fillna(False) & availability.gt(origin)).sum())
    excluded_dq = int(dq_failed.sum())
    if eligible.empty:
        raise DatabricksLTMaterializationError(
            "Silver vintages contain no known, non-DQ row available at the PIT origin"
        )

    eligible = _select_latest_vintage_state(eligible)
    return _materialize_entsoe(
        dimension_projection=dimension_projection,
        facts=eligible,
        mapping=mapping,
        mode="SILVER_POINT_IN_TIME",
        as_of_utc=origin,
        source_projection_sha256=source_projection_hash,
        selection_metrics={
            "source_rows": len(source),
            "unmapped_source_rows": len(full_source) - len(source),
            "eligible_rows": int(eligible.shape[0]),
            "excluded_unknown_availability_rows": excluded_unknown,
            "excluded_after_origin_rows": excluded_future,
            "excluded_dq_rows": excluded_dq,
        },
    )


def materialize_entsoe_current_features(
    *,
    dimension: pd.DataFrame,
    gold_latest: pd.DataFrame,
    mapping: EntsoeFeatureMapping,
    as_of_utc: str | pd.Timestamp,
) -> DatabricksMaterialization:
    """Build a current-serving feature frame from Gold Latest.

    This path still requires known availability at or before ``as_of_utc`` but
    does not claim historical PIT replay authority.
    """

    origin = _utc_scalar(as_of_utc, label="ENTSO-E current origin")
    dimension_projection = _validate_entsoe_mapping(dimension, mapping)
    _require_columns(gold_latest, _GOLD_LATEST_COLUMNS, label="Gold Latest")
    if gold_latest.columns.has_duplicates:
        raise DatabricksLTMaterializationError("Gold Latest has duplicate columns")
    facts = gold_latest.loc[:, sorted(_GOLD_LATEST_COLUMNS)].copy()
    facts = facts.merge(
        dimension_projection[["SeriesID", "SeriesKey"]],
        on="SeriesID",
        how="inner",
        validate="many_to_one",
    )
    facts = facts.rename(
        columns={
            "FieldValue": "FieldValue",
            "DateTimeUtc": "DateTimeUtc",
        }
    )
    known = facts["AvailabilityKnown"].astype("boolean")
    availability = _utc_series(
        facts["AvailabilityTimestampUtc"],
        label="Gold Latest availability",
        allow_missing=True,
    )
    _validate_availability_semantics(
        basis=facts["AvailabilityBasis"],
        known=known,
        availability=availability,
        publication=facts["PublicationTimestampUtc"],
        label="Gold Latest",
    )
    eligible_mask = known.fillna(False) & availability.le(origin)
    eligible = facts.loc[eligible_mask].copy()
    eligible["AvailabilityTimestampUtc"] = availability.loc[eligible_mask]
    if eligible.empty:
        raise DatabricksLTMaterializationError(
            "Gold Latest contains no known row available at the current origin"
        )
    identity = ["SeriesKey", "IntervalStartUtc", "DateTimeUtc"]
    if eligible.duplicated(identity).any():
        raise DatabricksLTMaterializationError(
            "Gold Latest repeats a selected series interval"
        )
    source_hash = dataframe_semantic_sha256(
        facts.drop(columns=["AvailabilityKnown"]).reset_index(drop=True)
    )
    return _materialize_entsoe(
        dimension_projection=dimension_projection,
        facts=eligible,
        mapping=mapping,
        mode="GOLD_CURRENT_SERVING",
        as_of_utc=origin,
        source_projection_sha256=source_hash,
        selection_metrics={
            "source_rows": len(facts),
            "eligible_rows": len(eligible),
            "excluded_unknown_availability_rows": int((~known.fillna(False)).sum()),
            "excluded_after_origin_rows": int(
                (known.fillna(False) & availability.gt(origin)).sum()
            ),
        },
    )


def materialize_spot_price_history(
    frame: pd.DataFrame,
    *,
    market_zone: str,
    source_product: str,
    as_of_utc: str | pd.Timestamp,
) -> DatabricksMaterialization:
    """Build one causal 15-minute spot history from a Gold interval fact."""

    _require_columns(frame, _SPOT_COLUMNS, label="Gold spot")
    if frame.columns.has_duplicates:
        raise DatabricksLTMaterializationError("Gold spot has duplicate columns")
    origin = _utc_scalar(as_of_utc, label="spot origin")
    zone = str(market_zone).strip().upper()
    product = str(source_product).strip()
    if not zone or not product:
        raise DatabricksLTMaterializationError(
            "spot market_zone and source_product must be explicit"
        )
    source = frame.loc[
        frame["MarketZone"].astype("string").str.upper().eq(zone)
        & frame["SourceProduct"].astype("string").eq(product)
    ].copy()
    if source.empty:
        raise DatabricksLTMaterializationError(
            "Gold spot contains no row for the selected market/product"
        )
    if not source["PriceUnit"].eq("EUR/MWh").all():
        raise DatabricksLTMaterializationError("Gold spot price unit must be EUR/MWh")
    observed = _utc_series(source["ObservedAtUtc"], label="spot observed-at")
    delivery_end = _utc_series(source["DeliveryEndUtc"], label="spot delivery end")
    eligible_mask = observed.le(origin) & delivery_end.le(origin)
    eligible = source.loc[eligible_mask].copy()
    if eligible.empty:
        raise DatabricksLTMaterializationError(
            "Gold spot contains no delivered row known at the requested origin"
        )
    if eligible.duplicated(["SpotProductID", "DeliveryStartUtc"]).any():
        raise DatabricksLTMaterializationError(
            "Gold spot repeats the selected product/interval grain"
        )
    raw, cadence_counts = _expand_spot_intervals(eligible)
    metadata = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "source_layer": "GOLD",
        "source_role": "spot_price_interval",
        "market_zone": zone,
        "source_product": product,
        "as_of_utc": origin.isoformat(),
        "source_cadence_minutes": cadence_counts,
        "output_cadence_seconds": 900,
        "resampling_method": (
            "NONE"
            if set(cadence_counts) == {"15"}
            else "FORWARD_FILL_WITHIN_DECLARED_SOURCE_INTERVAL"
        ),
    }
    raw.attrs[MATERIALIZATION_METADATA_ATTR] = metadata
    derived = clean_epex(raw)
    audit = {
        **metadata,
        "status": "PASS_LOCAL_MATERIALIZATION_NO_MODEL_AUTHORITY",
        "source_rows": len(source),
        "eligible_rows": len(eligible),
        "excluded_not_known_or_delivered_rows": int((~eligible_mask).sum()),
        "output_rows": len(raw),
        "output_start_utc": raw.index.min().isoformat(),
        "output_end_utc": raw.index.max().isoformat(),
        "raw_frame_sha256": dataframe_semantic_sha256(raw),
        "derived_frame_sha256": dataframe_semantic_sha256(derived),
        "authorities": _false_authorities(),
    }
    return DatabricksMaterialization(raw, derived, audit)


def _canonical_silver_vintages(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, _SILVER_VINTAGE_COLUMNS, label="Silver vintages")
    if frame.columns.has_duplicates:
        raise DatabricksLTMaterializationError("Silver vintages has duplicate columns")
    projection = frame.loc[:, sorted(_SILVER_VINTAGE_COLUMNS)].rename(
        columns={
            "SK_ge_power_entsoe_time_series_vintages": "VintageID",
            "series_key": "SeriesKey",
            "field_value": "FieldValue",
            "Date_Time_UTC": "DateTimeUtc",
            "resolution": "Resolution",
            "publication_timestamp_utc": "PublicationTimestampUtc",
            "first_seen_pull_ts_utc": "FirstObservedAtUtc",
            "last_seen_pull_ts_utc": "LastObservedAtUtc",
            "availability_basis": "AvailabilityBasis",
            "availability_known": "AvailabilityKnown",
            "availability_timestamp_utc": "AvailabilityTimestampUtc",
            "source_document_revision_number": "RevisionNumber",
            "source_document_mrid": "SourceDocumentMRID",
        }
    )
    if projection["VintageID"].isna().any() or projection["VintageID"].duplicated().any():
        raise DatabricksLTMaterializationError("Silver VintageID must be non-null and unique")
    for column in ("SeriesKey", "SourceDocumentMRID"):
        values = projection[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise DatabricksLTMaterializationError(
                f"Silver {column} must be non-null and non-empty"
            )
    return projection


def _select_latest_vintage_state(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["LastObservedAtUtc"] = _utc_series(
        working["LastObservedAtUtc"], label="Silver last observed"
    )
    working["RevisionNumber"] = pd.to_numeric(
        working["RevisionNumber"], errors="coerce"
    )
    if working["RevisionNumber"].isna().any():
        raise DatabricksLTMaterializationError(
            "Silver source document revision number is invalid"
        )
    identity = ["SeriesKey", "IntervalStartUtc", "DateTimeUtc"]
    order = ["AvailabilityTimestampUtc", "RevisionNumber", "LastObservedAtUtc"]
    tied = working.duplicated([*identity, *order], keep=False)
    if tied.any():
        semantic = ["FieldValue", "IntervalEndUtc", "Resolution"]
        ambiguous = (
            working.loc[tied]
            .groupby([*identity, *order], dropna=False)[semantic]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if ambiguous.any():
            raise DatabricksLTMaterializationError(
                "Silver vintages contain an ambiguous latest-state tie"
            )
    return (
        working.sort_values([*identity, *order, "VintageID"], kind="mergesort")
        .drop_duplicates(identity, keep="last")
        .reset_index(drop=True)
    )


def _validate_entsoe_mapping(
    dimension: pd.DataFrame,
    mapping: EntsoeFeatureMapping,
) -> pd.DataFrame:
    _require_columns(dimension, _DIMENSION_COLUMNS, label="Gold ENTSO-E dimension")
    if dimension.columns.has_duplicates:
        raise DatabricksLTMaterializationError(
            "Gold ENTSO-E dimension has duplicate columns"
        )
    projection = dimension.loc[:, sorted(_DIMENSION_COLUMNS)].copy()
    if projection["SeriesID"].isna().any() or projection["SeriesID"].duplicated().any():
        raise DatabricksLTMaterializationError("Gold SeriesID must be non-null and unique")
    if projection["SeriesKey"].isna().any() or projection["SeriesKey"].duplicated().any():
        raise DatabricksLTMaterializationError("Gold SeriesKey must be non-null and unique")

    all_keys: set[str] = set()
    for feature in _CORE_FEATURES:
        if not getattr(mapping, feature):
            raise DatabricksLTMaterializationError(
                f"ENTSO-E mapping requires at least one {feature} SeriesKey"
            )
    by_key = projection.set_index("SeriesKey", drop=False)
    for feature in (*_CORE_FEATURES, "cross_border_mw"):
        policy = _FEATURE_POLICY[feature]
        for term in getattr(mapping, feature):
            key = str(term.series_key).strip()
            weight = float(term.weight)
            if not key or not math.isfinite(weight) or weight == 0.0:
                raise DatabricksLTMaterializationError(
                    f"ENTSO-E mapping term is invalid for {feature}"
                )
            if key in all_keys:
                raise DatabricksLTMaterializationError(
                    f"ENTSO-E SeriesKey is mapped more than once: {key}"
                )
            all_keys.add(key)
            if key not in by_key.index:
                raise DatabricksLTMaterializationError(
                    f"ENTSO-E SeriesKey is absent from Gold dimension: {key}"
                )
            row = by_key.loc[key]
            if str(row["Unit"]) != "MW":
                raise DatabricksLTMaterializationError(
                    f"ENTSO-E mapped series is not MW: {key}"
                )
            if str(row["GroupName"]) not in policy["groups"]:
                raise DatabricksLTMaterializationError(
                    f"ENTSO-E mapped series has the wrong group for {feature}: {key}"
                )
            if str(row["FieldName"]).lower() != "quantity":
                raise DatabricksLTMaterializationError(
                    f"ENTSO-E mapped series is not a quantity field: {key}"
                )
            for column, policy_key, label in (
                ("DocumentType", "documents", "DocumentType"),
                ("ProcessType", "processes", "ProcessType"),
                ("BusinessType", "business_types", "BusinessType"),
            ):
                allowed = policy[policy_key]
                if allowed is not None and str(row[column]) not in allowed:
                    raise DatabricksLTMaterializationError(
                        f"ENTSO-E mapped series has the wrong {label} for {feature}: {key}"
                    )
            psr_types = policy["psr_types"]
            if psr_types is not None and str(row["PsrType"]) not in psr_types:
                raise DatabricksLTMaterializationError(
                    f"ENTSO-E mapped series has the wrong PsrType for {feature}: {key}"
                )
            zones = {str(row["FromZone"]), str(row["ToZone"])}
            if "CH" not in zones:
                raise DatabricksLTMaterializationError(
                    f"ENTSO-E mapped series is not connected to CH: {key}"
                )
    return projection.loc[projection["SeriesKey"].isin(all_keys)].reset_index(drop=True)


def _materialize_entsoe(
    *,
    dimension_projection: pd.DataFrame,
    facts: pd.DataFrame,
    mapping: EntsoeFeatureMapping,
    mode: str,
    as_of_utc: pd.Timestamp,
    source_projection_sha256: str,
    selection_metrics: Mapping[str, object],
) -> DatabricksMaterialization:
    selected_keys = set(dimension_projection["SeriesKey"].astype(str))
    selected = facts.loc[facts["SeriesKey"].astype(str).isin(selected_keys)].copy()
    if selected.empty:
        raise DatabricksLTMaterializationError(
            "ENTSO-E facts contain none of the explicitly mapped SeriesKeys"
        )

    feature_series: dict[str, pd.Series] = {}
    for feature in (*_CORE_FEATURES, "cross_border_mw"):
        terms = getattr(mapping, feature)
        if not terms:
            continue
        components: list[pd.Series] = []
        expected_index: pd.DatetimeIndex | None = None
        for term in terms:
            rows = selected.loc[selected["SeriesKey"].astype(str).eq(term.series_key)]
            if rows.empty:
                raise DatabricksLTMaterializationError(
                    f"ENTSO-E mapped series has no eligible values: {term.series_key}"
                )
            component = _expand_entsoe_intervals(rows) * float(term.weight)
            if expected_index is None:
                expected_index = component.index
            elif not component.index.equals(expected_index):
                raise DatabricksLTMaterializationError(
                    f"ENTSO-E component coverage differs inside {feature}"
                )
            components.append(component)
        feature_series[feature] = sum(components[1:], start=components[0].copy())

    core_index = feature_series[_CORE_FEATURES[0]].index
    for feature, series in feature_series.items():
        if not series.index.equals(core_index):
            raise DatabricksLTMaterializationError(
                f"ENTSO-E feature coverage differs from load_mw: {feature}"
            )
    _require_complete_quarter_hour_grid(core_index, label="ENTSO-E materialization")
    raw = pd.DataFrame(
        {feature: feature_series[feature].to_numpy(dtype=float) for feature in feature_series},
        index=core_index,
    )
    raw.index.name = None
    if not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise DatabricksLTMaterializationError(
            "ENTSO-E materialization produced non-finite values"
        )
    metadata = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "mode": mode,
        "as_of_utc": as_of_utc.isoformat(),
        "mapping": mapping.as_dict(),
        "output_cadence_seconds": 900,
    }
    raw.attrs[MATERIALIZATION_METADATA_ATTR] = metadata
    derived = build_entso_features(raw)
    audit = {
        **metadata,
        "status": "PASS_LOCAL_MATERIALIZATION_NO_MODEL_AUTHORITY",
        "source_projection_sha256": source_projection_sha256,
        "dimension_projection_sha256": dataframe_semantic_sha256(
            dimension_projection.reset_index(drop=True)
        ),
        "selection_metrics": dict(selection_metrics),
        "output_rows": len(raw),
        "output_start_utc": raw.index.min().isoformat(),
        "output_end_utc": raw.index.max().isoformat(),
        "raw_frame_sha256": dataframe_semantic_sha256(raw),
        "derived_frame_sha256": dataframe_semantic_sha256(derived),
        "authorities": _false_authorities(),
    }
    return DatabricksMaterialization(raw, derived, audit)


def _expand_entsoe_intervals(frame: pd.DataFrame) -> pd.Series:
    values: list[float] = []
    timestamps: list[pd.Timestamp] = []
    working = frame.copy()
    start = _utc_series(working["IntervalStartUtc"], label="ENTSO-E interval start")
    end = _utc_series(working["IntervalEndUtc"], label="ENTSO-E interval end")
    right_edge = _utc_series(working["DateTimeUtc"], label="ENTSO-E right edge")
    field_value = pd.to_numeric(working["FieldValue"], errors="coerce")
    if field_value.isna().any() or not np.isfinite(field_value.to_numpy(dtype=float)).all():
        raise DatabricksLTMaterializationError("ENTSO-E values must be finite")
    for position in range(len(working)):
        resolution_seconds = _resolution_seconds(working["Resolution"].iloc[position])
        duration_seconds = int((end.iloc[position] - start.iloc[position]).total_seconds())
        if (
            end.iloc[position] != right_edge.iloc[position]
            or duration_seconds != resolution_seconds
            or duration_seconds % 900 != 0
            or duration_seconds > 3600
        ):
            raise DatabricksLTMaterializationError(
                "ENTSO-E interval/resolution is not an atomic 15/30/60-minute value"
            )
        interval_index = pd.date_range(
            start.iloc[position],
            end.iloc[position],
            freq="15min",
            inclusive="left",
        )
        timestamps.extend(interval_index)
        values.extend([float(field_value.iloc[position])] * len(interval_index))
    result = pd.Series(values, index=pd.DatetimeIndex(timestamps), dtype=float)
    result = result.sort_index(kind="mergesort")
    if result.index.has_duplicates:
        raise DatabricksLTMaterializationError(
            "ENTSO-E selected intervals overlap after 15-minute expansion"
        )
    return result


def _expand_spot_intervals(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    timestamps: list[pd.Timestamp] = []
    values: list[float] = []
    cadence_counts: dict[str, int] = {}
    start = _utc_series(frame["DeliveryStartUtc"], label="spot delivery start")
    end = _utc_series(frame["DeliveryEndUtc"], label="spot delivery end")
    minutes = pd.to_numeric(frame["FrequencyMinutes"], errors="coerce")
    prices = pd.to_numeric(frame["Price"], errors="coerce")
    if minutes.isna().any() or prices.isna().any():
        raise DatabricksLTMaterializationError("Gold spot cadence/price is invalid")
    if not np.isfinite(prices.to_numpy(dtype=float)).all():
        raise DatabricksLTMaterializationError("Gold spot price is non-finite")
    for position in range(len(frame)):
        cadence = int(minutes.iloc[position])
        duration = int((end.iloc[position] - start.iloc[position]).total_seconds() / 60)
        if cadence not in {15, 30, 60} or duration != cadence:
            raise DatabricksLTMaterializationError(
                "Gold spot interval is not an atomic 15/30/60-minute value"
            )
        interval_index = pd.date_range(
            start.iloc[position], end.iloc[position], freq="15min", inclusive="left"
        )
        timestamps.extend(interval_index)
        values.extend([float(prices.iloc[position])] * len(interval_index))
        key = str(cadence)
        cadence_counts[key] = cadence_counts.get(key, 0) + 1
    result = pd.DataFrame(
        {"price_eur_mwh": values}, index=pd.DatetimeIndex(timestamps)
    ).sort_index(kind="mergesort")
    if result.index.has_duplicates:
        raise DatabricksLTMaterializationError(
            "Gold spot intervals overlap after 15-minute expansion"
        )
    _require_complete_quarter_hour_grid(result.index, label="Gold spot materialization")
    return result, cadence_counts


def _require_complete_quarter_hour_grid(
    index: pd.DatetimeIndex,
    *,
    label: str,
) -> None:
    if index.empty or index.tz is None:
        raise DatabricksLTMaterializationError(f"{label} index must be non-empty UTC")
    utc_index = index.tz_convert("UTC")
    if not utc_index.is_monotonic_increasing or utc_index.has_duplicates:
        raise DatabricksLTMaterializationError(f"{label} index is not sorted and unique")
    if len(utc_index) > 1:
        differences = np.diff(utc_index.asi8) // 1_000_000_000
        if not bool(np.equal(differences, 900).all()):
            raise DatabricksLTMaterializationError(
                f"{label} contains a 15-minute grid gap"
            )


def _resolution_seconds(value: object) -> int:
    match = _RESOLUTION.fullmatch(str(value))
    if match is None:
        raise DatabricksLTMaterializationError(
            f"ENTSO-E resolution is unsupported: {value}"
        )
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = hours * 3600 + minutes * 60
    if seconds <= 0:
        raise DatabricksLTMaterializationError(
            f"ENTSO-E resolution is unsupported: {value}"
        )
    return seconds


def _require_columns(
    frame: pd.DataFrame,
    required: frozenset[str],
    *,
    label: str,
) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DatabricksLTMaterializationError(f"{label} frame is empty")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DatabricksLTMaterializationError(
            f"{label} frame is missing columns: {missing}"
        )


def _utc_scalar(value: str | pd.Timestamp, *, label: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise DatabricksLTMaterializationError(f"{label} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _utc_series(
    values: pd.Series,
    *,
    label: str,
    allow_missing: bool = False,
) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    invalid = parsed.isna() & (values.notna() if allow_missing else True)
    if invalid.any():
        raise DatabricksLTMaterializationError(f"{label} contains invalid timestamps")
    return parsed


def _validate_availability_semantics(
    *,
    basis: pd.Series,
    known: pd.Series,
    availability: pd.Series,
    publication: pd.Series,
    label: str,
    first_observed: pd.Series | None = None,
    last_observed: pd.Series | None = None,
) -> None:
    normalized_basis = basis.astype("string")
    recognized = normalized_basis.isin(
        {"UNKNOWN_BACKFILL", "FMV_FIRST_SEEN", "SOURCE_DOCUMENT_CREATED"}
    )
    publication_utc = _utc_series(publication, label=f"{label} publication")
    unknown = normalized_basis.eq("UNKNOWN_BACKFILL")
    invalid = (~recognized) | (unknown & (known.fillna(False) | availability.notna()))
    invalid |= (~unknown) & (~known.fillna(False) | availability.isna())
    source_created = normalized_basis.eq("SOURCE_DOCUMENT_CREATED")
    invalid |= source_created & availability.ne(publication_utc)
    if first_observed is not None:
        first_utc = _utc_series(first_observed, label=f"{label} first observed")
        fmv_first_seen = normalized_basis.eq("FMV_FIRST_SEEN")
        invalid |= fmv_first_seen & availability.ne(first_utc)
        invalid |= publication_utc.gt(first_utc)
        if last_observed is None:
            raise DatabricksLTMaterializationError(
                f"{label} last observed timestamp is missing"
            )
        last_utc = _utc_series(last_observed, label=f"{label} last observed")
        invalid |= first_utc.gt(last_utc)
    if invalid.fillna(True).any():
        raise DatabricksLTMaterializationError(
            f"{label} availability semantics are inconsistent"
        )


def _false_authorities() -> dict[str, bool]:
    return {
        "source_layer_semantic_acceptance": False,
        "model_input_authorized": False,
        "calibration_authorized": False,
        "production_authorized": False,
    }


__all__ = [
    "DatabricksLTMaterializationError",
    "DatabricksMaterialization",
    "EntsoeFeatureMapping",
    "EntsoeSeriesTerm",
    "ENTSOE_FEATURE_MAPPING_SCHEMA_VERSION",
    "MATERIALIZATION_SCHEMA_VERSION",
    "entsoe_dimension_semantic_sha256",
    "entsoe_feature_mapping_from_contract",
    "materialize_entsoe_current_features",
    "materialize_entsoe_pit_features",
    "materialize_spot_price_history",
]
