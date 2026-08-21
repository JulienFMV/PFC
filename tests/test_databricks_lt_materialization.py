from __future__ import annotations

import pandas as pd
import pytest

from pfc_shaping.data.databricks_lt_materialization import (
    ENTSOE_FEATURE_MAPPING_SCHEMA_VERSION,
    DatabricksLTMaterializationError,
    EntsoeFeatureMapping,
    EntsoeSeriesTerm,
    entsoe_dimension_semantic_sha256,
    entsoe_feature_mapping_from_contract,
    materialize_entsoe_current_features,
    materialize_entsoe_pit_features,
    materialize_spot_price_history,
)


def _dimension() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _dimension_row(1, "LOAD", "actual_load", None, "CH", None),
            _dimension_row(2, "SOLAR", "generation_actual", "B16", "CH", None),
            _dimension_row(3, "WIND_ON", "generation_actual", "B19", "CH", None),
            _dimension_row(4, "WIND_OFF", "generation_actual", "B18", "CH", None),
            _dimension_row(
                5,
                "FLOW_EXPORT",
                "crossborder_physical_flows",
                None,
                "CH",
                "DE_LU",
            ),
            _dimension_row(
                6,
                "FLOW_IMPORT",
                "crossborder_physical_flows",
                None,
                "DE_LU",
                "CH",
            ),
        ]
    )


def _dimension_row(
    series_id: int,
    series_key: str,
    group: str,
    psr_type: str | None,
    from_zone: str,
    to_zone: str | None,
) -> dict[str, object]:
    return {
        "SeriesID": series_id,
        "SeriesKey": series_key,
        "SourceTimeSeriesId": f"TS-{series_id}",
        "GroupName": group,
        "FieldName": "quantity",
        "DocumentType": {
            "actual_load": "A65",
            "generation_actual": "A75",
            "crossborder_physical_flows": "A11",
        }[group],
        "BusinessType": "A01" if group == "generation_actual" else None,
        "ProcessType": "A16",
        "PsrType": psr_type,
        "FromZone": from_zone,
        "ToZone": to_zone,
        "Unit": "MW",
    }


def _mapping() -> EntsoeFeatureMapping:
    return EntsoeFeatureMapping(
        load_mw=(EntsoeSeriesTerm("LOAD"),),
        solar_mw=(EntsoeSeriesTerm("SOLAR"),),
        wind_mw=(EntsoeSeriesTerm("WIND_ON"), EntsoeSeriesTerm("WIND_OFF")),
        cross_border_mw=(
            EntsoeSeriesTerm("FLOW_EXPORT", 1.0),
            EntsoeSeriesTerm("FLOW_IMPORT", -1.0),
        ),
    )


def _silver_vintages() -> pd.DataFrame:
    values = {
        "LOAD": (100.0, 110.0),
        "SOLAR": (10.0, 20.0),
        "WIND_ON": (30.0, 40.0),
        "WIND_OFF": (5.0, 6.0),
        "FLOW_EXPORT": (50.0, 55.0),
        "FLOW_IMPORT": (20.0, 25.0),
    }
    rows: list[dict[str, object]] = []
    vintage_number = 0
    for series_key, series_values in values.items():
        for hour, value in enumerate(series_values):
            vintage_number += 1
            rows.append(
                _vintage_row(
                    vintage_id=f"V-{vintage_number}",
                    series_key=series_key,
                    start=f"2026-10-24T{hour:02d}:00:00Z",
                    value=value,
                    availability="2026-10-24T02:30:00Z",
                    revision=1,
                )
            )
    rows.append(
        _vintage_row(
            vintage_id="V-LATE-REVISION",
            series_key="LOAD",
            start="2026-10-24T00:00:00Z",
            value=999.0,
            availability="2026-10-24T03:30:00Z",
            revision=2,
        )
    )
    return pd.DataFrame(rows)


def _vintage_row(
    *,
    vintage_id: str,
    series_key: str,
    start: str,
    value: float,
    availability: str | None,
    revision: int,
    availability_known: bool = True,
    dq_failed: bool = False,
    last_seen: str | None = None,
) -> dict[str, object]:
    start_ts = pd.Timestamp(start)
    end_ts = start_ts + pd.Timedelta(hours=1)
    return {
        "SK_ge_power_entsoe_time_series_vintages": vintage_id,
        "series_key": series_key,
        "field_value": value,
        "IntervalStartUtc": start_ts,
        "Date_Time_UTC": end_ts,
        "IntervalEndUtc": end_ts,
        "resolution": "PT60M",
        "publication_timestamp_utc": (
            pd.Timestamp(availability) - pd.Timedelta(minutes=30)
            if availability is not None
            else pd.Timestamp("2026-10-24T02:00:00Z")
        ),
        "first_seen_pull_ts_utc": (
            pd.Timestamp(availability)
            if availability is not None
            else pd.Timestamp("2026-10-24T02:30:00Z")
        ),
        "last_seen_pull_ts_utc": pd.Timestamp(last_seen or availability),
        "availability_basis": (
            "FMV_FIRST_SEEN" if availability_known else "UNKNOWN_BACKFILL"
        ),
        "availability_known": availability_known,
        "availability_timestamp_utc": (
            pd.Timestamp(availability) if availability is not None else pd.NaT
        ),
        "source_document_revision_number": revision,
        "source_document_mrid": f"DOC-{series_key}",
        "dq_failed": dq_failed,
    }


def _gold_latest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    values = {1: 100.0, 2: 10.0, 3: 30.0, 4: 5.0, 5: 50.0, 6: 20.0}
    for series_id, value in values.items():
        for hour in range(2):
            start = pd.Timestamp(f"2026-10-24T{hour:02d}:00:00Z")
            rows.append(
                {
                    "SeriesID": series_id,
                    "IntervalStartUtc": start,
                    "DateTimeUtc": start + pd.Timedelta(hours=1),
                    "IntervalEndUtc": start + pd.Timedelta(hours=1),
                    "Resolution": "PT60M",
                    "FieldValue": value + hour,
                    "PublicationTimestampUtc": pd.Timestamp(
                        "2026-10-24T02:30:00Z"
                    ),
                    "AvailabilityBasis": "SOURCE_DOCUMENT_CREATED",
                    "AvailabilityKnown": True,
                    "AvailabilityTimestampUtc": pd.Timestamp(
                        "2026-10-24T02:30:00Z"
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_silver_pit_materialization_excludes_later_revision_and_aggregates() -> None:
    result = materialize_entsoe_pit_features(
        dimension=_dimension(),
        silver_vintages=_silver_vintages(),
        mapping=_mapping(),
        as_of_utc="2026-10-24T03:00:00Z",
    )

    assert len(result.raw_frame) == 8
    assert result.raw_frame.index.tz is not None
    assert result.raw_frame.iloc[0].to_dict() == {
        "load_mw": 100.0,
        "solar_mw": 10.0,
        "wind_mw": 35.0,
        "cross_border_mw": 30.0,
    }
    assert result.raw_frame.iloc[-1].to_dict() == {
        "load_mw": 110.0,
        "solar_mw": 20.0,
        "wind_mw": 46.0,
        "cross_border_mw": 30.0,
    }
    assert result.audit["mode"] == "SILVER_POINT_IN_TIME"
    assert result.audit["selection_metrics"]["excluded_after_origin_rows"] == 1
    assert result.audit["authorities"]["model_input_authorized"] is False
    assert {"solar_regime", "load_deviation", "flow_deviation"}.issubset(
        result.derived_frame.columns
    )


def test_silver_unknown_backfill_cannot_fill_pit_grid() -> None:
    vintages = _silver_vintages()
    mask = vintages["series_key"].eq("SOLAR") & vintages["IntervalStartUtc"].eq(
        pd.Timestamp("2026-10-24T01:00:00Z")
    )
    vintages.loc[mask, "availability_basis"] = "UNKNOWN_BACKFILL"
    vintages.loc[mask, "availability_known"] = False
    vintages.loc[mask, "availability_timestamp_utc"] = pd.NaT

    with pytest.raises(
        DatabricksLTMaterializationError,
        match="coverage differs",
    ):
        materialize_entsoe_pit_features(
            dimension=_dimension(),
            silver_vintages=vintages,
            mapping=_mapping(),
            as_of_utc="2026-10-24T03:00:00Z",
        )


def test_silver_ambiguous_latest_state_tie_fails_closed() -> None:
    vintages = _silver_vintages()
    duplicate = vintages.loc[
        vintages["SK_ge_power_entsoe_time_series_vintages"].eq("V-1")
    ].copy()
    duplicate["SK_ge_power_entsoe_time_series_vintages"] = "V-AMBIGUOUS"
    duplicate["field_value"] = 777.0
    vintages = pd.concat([vintages, duplicate], ignore_index=True)

    with pytest.raises(
        DatabricksLTMaterializationError,
        match="ambiguous latest-state tie",
    ):
        materialize_entsoe_pit_features(
            dimension=_dimension(),
            silver_vintages=vintages,
            mapping=_mapping(),
            as_of_utc="2026-10-24T03:00:00Z",
        )


def test_entsoe_mapping_rejects_wrong_psr_type() -> None:
    dimension = _dimension()
    dimension.loc[dimension["SeriesKey"].eq("SOLAR"), "PsrType"] = "B19"

    with pytest.raises(DatabricksLTMaterializationError, match="wrong PsrType"):
        materialize_entsoe_pit_features(
            dimension=dimension,
            silver_vintages=_silver_vintages(),
            mapping=_mapping(),
            as_of_utc="2026-10-24T03:00:00Z",
        )


def test_entsoe_mapping_rejects_generation_consumption_series() -> None:
    dimension = _dimension()
    dimension.loc[dimension["SeriesKey"].eq("SOLAR"), "BusinessType"] = "A04"

    with pytest.raises(DatabricksLTMaterializationError, match="wrong BusinessType"):
        materialize_entsoe_pit_features(
            dimension=dimension,
            silver_vintages=_silver_vintages(),
            mapping=_mapping(),
            as_of_utc="2026-10-24T03:00:00Z",
        )


def test_entsoe_mapping_contract_is_bound_to_dimension_semantics() -> None:
    dimension = _dimension()
    expected = _mapping()
    contract = {
        "schema_version": ENTSOE_FEATURE_MAPPING_SCHEMA_VERSION,
        "market": "CH",
        "dimension_semantic_sha256": entsoe_dimension_semantic_sha256(dimension),
        "features": {
            feature: [
                {"series_key": term.series_key, "weight": term.weight}
                for term in getattr(expected, feature)
            ]
            for feature in (
                "load_mw",
                "solar_mw",
                "wind_mw",
                "cross_border_mw",
            )
        },
    }

    observed = entsoe_feature_mapping_from_contract(contract, dimension=dimension)

    assert observed == expected
    changed = dimension.copy()
    changed.loc[changed["SeriesKey"].eq("SOLAR"), "FieldName"] = "other_quantity"
    with pytest.raises(DatabricksLTMaterializationError, match="hash mismatch"):
        entsoe_feature_mapping_from_contract(contract, dimension=changed)


def test_silver_inconsistent_availability_basis_fails_closed() -> None:
    vintages = _silver_vintages()
    vintages.loc[
        vintages["SK_ge_power_entsoe_time_series_vintages"].eq("V-1"),
        "availability_basis",
    ] = "SOURCE_DOCUMENT_CREATED"

    with pytest.raises(
        DatabricksLTMaterializationError,
        match="availability semantics are inconsistent",
    ):
        materialize_entsoe_pit_features(
            dimension=_dimension(),
            silver_vintages=vintages,
            mapping=_mapping(),
            as_of_utc="2026-10-24T03:00:00Z",
        )


def test_gold_latest_current_materialization_is_explicitly_non_authoritative() -> None:
    result = materialize_entsoe_current_features(
        dimension=_dimension(),
        gold_latest=_gold_latest(),
        mapping=_mapping(),
        as_of_utc="2026-10-24T03:00:00Z",
    )

    assert len(result.raw_frame) == 8
    assert result.audit["mode"] == "GOLD_CURRENT_SERVING"
    assert result.audit["authorities"] == {
        "source_layer_semantic_acceptance": False,
        "model_input_authorized": False,
        "calibration_authorized": False,
        "production_authorized": False,
    }


def _spot_rows(*, omit_hour: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for hour in range(4):
        if hour == omit_hour:
            continue
        start = pd.Timestamp("2026-10-25T00:00:00Z") + pd.Timedelta(hours=hour)
        rows.append(
            {
                "SpotProductID": 1,
                "SourceProduct": "CH_DAY_AHEAD",
                "MarketZone": "CH",
                "DeliveryStartUtc": start,
                "DeliveryEndUtc": start + pd.Timedelta(hours=1),
                "FrequencyMinutes": 60,
                "Price": 50.0 + hour,
                "PriceUnit": "EUR/MWh",
                "ObservedAtUtc": pd.Timestamp("2026-10-24T12:00:00Z"),
            }
        )
    return pd.DataFrame(rows)


def test_spot_materialization_preserves_swiss_fall_back_in_utc() -> None:
    result = materialize_spot_price_history(
        _spot_rows(),
        market_zone="CH",
        source_product="CH_DAY_AHEAD",
        as_of_utc="2026-10-25T05:00:00Z",
    )

    assert len(result.raw_frame) == 16
    assert result.raw_frame.index.is_unique
    assert result.raw_frame.index.tz_convert("Europe/Zurich").hour.tolist().count(2) == 8
    assert result.raw_frame.iloc[:4, 0].eq(50.0).all()
    assert result.audit["resampling_method"] == (
        "FORWARD_FILL_WITHIN_DECLARED_SOURCE_INTERVAL"
    )


def test_spot_materialization_rejects_a_grid_gap() -> None:
    with pytest.raises(DatabricksLTMaterializationError, match="grid gap"):
        materialize_spot_price_history(
            _spot_rows(omit_hour=2),
            market_zone="CH",
            source_product="CH_DAY_AHEAD",
            as_of_utc="2026-10-25T05:00:00Z",
        )
