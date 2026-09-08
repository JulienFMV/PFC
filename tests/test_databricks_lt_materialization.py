from __future__ import annotations

import hashlib
from io import BytesIO

import pandas as pd
import pytest

from pfc_shaping.data.databricks_eex_daily_snapshot import EXPECTED_COLUMNS
from pfc_shaping.data.databricks_lt_materialization import (
    ENTSOE_FEATURE_MAPPING_SCHEMA_VERSION,
    DatabricksLTMaterializationError,
    EntsoeFeatureMapping,
    EntsoeSeriesTerm,
    entsoe_dimension_semantic_sha256,
    entsoe_feature_mapping_from_contract,
    materialize_eex_forward_history,
    materialize_entsoe_current_features,
    materialize_entsoe_latest_observed_features,
    materialize_entsoe_pit_features,
    materialize_spot_price_history,
)
from pfc_shaping.data.databricks_lt_replay import (
    GOLD_ENTSOE_CURRENT_MODE,
    GOLD_SPOT_MODE,
    SILVER_ENTSOE_PIT_MODE,
    DatabricksLTReplayError,
    approved_databricks_materializer_payload,
    build_databricks_replay_package,
    build_databricks_role_replay,
    verify_databricks_replay_package,
    verify_databricks_role_replay,
)


@pytest.mark.parametrize("column", [
    "availability_timestamp_utc", "first_seen_pull_ts_utc",
    "last_seen_pull_ts_utc", "IntervalStartUtc", "IntervalEndUtc", "Date_Time_UTC",
])
@pytest.mark.parametrize("kind", ["epoch_seconds", "naive"])
def test_pit_rejects_ambiguous_timestamp_representation(column, kind):
    source = _silver_vintages()
    stamps = pd.to_datetime(source[column], utc=True)
    source[column] = (stamps.map(lambda value: value.timestamp()).astype("int64")
                      if kind == "epoch_seconds" else stamps.dt.tz_localize(None))
    with pytest.raises(DatabricksLTMaterializationError, match="timestamp|timezone"):
        materialize_entsoe_pit_features(
            dimension=_dimension(), silver_vintages=source, mapping=_mapping(),
            as_of_utc="2026-10-24T03:00:00Z",
        )


def test_document_creation_is_not_causal_pit_availability():
    source = _silver_vintages()
    late = source["SK_ge_power_entsoe_time_series_vintages"].eq("V-LATE-REVISION")
    source.loc[late, "availability_basis"] = "SOURCE_DOCUMENT_CREATED"
    source.loc[late, ["availability_timestamp_utc", "publication_timestamp_utc"]] = pd.Timestamp("2026-10-24T03:00:00Z")
    with pytest.raises(DatabricksLTMaterializationError, match="availability semantics|publication.*unproven"):
        materialize_entsoe_pit_features(
            dimension=_dimension(), silver_vintages=source, mapping=_mapping(),
            as_of_utc="2026-10-24T03:00:00Z",
        )


@pytest.mark.parametrize("lane", ["pit", "gold", "spot"])
def test_materialization_rejects_consistently_shifted_subsecond_grid(lane):
    shift = pd.Timedelta(milliseconds=500)
    with pytest.raises(DatabricksLTMaterializationError, match="align|grid"):
        if lane == "spot":
            source = _spot_rows()
            for column in ("DeliveryStartUtc", "DeliveryEndUtc"):
                source[column] += shift
            materialize_spot_price_history(source, market_zone="CH", source_product="CH_DAY_AHEAD", as_of_utc="2026-10-25T05:00Z")
        elif lane == "gold":
            source = _gold_latest()
            for column in ("IntervalStartUtc", "IntervalEndUtc", "DateTimeUtc"):
                source[column] += shift
            materialize_entsoe_current_features(dimension=_dimension(), gold_latest=source, mapping=_mapping(), as_of_utc="2026-10-24T03:00Z")
        else:
            source = _silver_vintages()
            for column in ("IntervalStartUtc", "IntervalEndUtc", "Date_Time_UTC"):
                source[column] += shift
            materialize_entsoe_pit_features(dimension=_dimension(), silver_vintages=source, mapping=_mapping(), as_of_utc="2026-10-24T03:00Z")


def test_pit_tie_cannot_be_resolved_by_future_last_observation():
    source = _silver_vintages().iloc[:-1].copy()
    rival = source.iloc[[0]].copy()
    rival["SK_ge_power_entsoe_time_series_vintages"] = "V-CONFLICT"
    rival["field_value"] = 999.0
    rival["last_seen_pull_ts_utc"] = pd.Timestamp("2026-10-25T05:00Z")
    source = pd.concat([source, rival], ignore_index=True)
    with pytest.raises(DatabricksLTMaterializationError, match="ambiguous"):
        materialize_entsoe_pit_features(dimension=_dimension(), silver_vintages=source, mapping=_mapping(), as_of_utc="2026-10-24T03:00Z")


def test_correlated_epoch_units_cannot_admit_a_post_origin_revision():
    source = _silver_vintages()
    for column in ("availability_timestamp_utc", "publication_timestamp_utc",
                   "first_seen_pull_ts_utc", "last_seen_pull_ts_utc"):
        source[column] = source[column].map(lambda value: value.timestamp()).astype('int64')
    with pytest.raises(DatabricksLTMaterializationError, match='numeric timestamps'):
        materialize_entsoe_pit_features(dimension=_dimension(), silver_vintages=source, mapping=_mapping(), as_of_utc="2026-10-24T03:00Z")


@pytest.mark.parametrize('mode', ['gold', 'latest', 'pit'])
def test_replay_consumes_entsoe_materialization_semantics(mode):
    from pfc_shaping.data.lt_input_replay import LTInputReplayError, _validate_raw_frame

    if mode == 'gold':
        result = materialize_entsoe_current_features(dimension=_dimension(), gold_latest=_gold_latest(), mapping=_mapping(), as_of_utc="2026-10-24T03:00Z")
    elif mode == 'latest':
        result = _local_observed(_silver_vintages())
    else:
        result = materialize_entsoe_pit_features(dimension=_dimension(), silver_vintages=_silver_vintages(), mapping=_mapping(), as_of_utc="2026-10-24T03:00Z")
    if mode == 'pit':
        _validate_raw_frame('entso', result.raw_frame, require_resolution_provenance=False)
    else:
        with pytest.raises(LTInputReplayError, match='not PIT'):
            _validate_raw_frame('entso', result.raw_frame, require_resolution_provenance=False)


@pytest.mark.parametrize('attack', ['none', 'missing', 'claim_native_truth'])
def test_replay_enforces_databricks_hourly_price_provenance(attack):
    from pfc_shaping.data.governed_lt_acquisition import OBSERVATION_RESOLUTION_PROVENANCE_ATTR
    from pfc_shaping.data.lt_input_replay import LTInputReplayError, _validate_raw_frame
    from pfc_shaping.data.lt_input_sources import _quality_frame_metrics

    result = materialize_spot_price_history(_spot_rows(), market_zone='CH', source_product='CH_DAY_AHEAD', as_of_utc='2026-10-25T05:00Z')
    frame = pd.read_parquet(BytesIO(_parquet_payload(result.raw_frame)))
    provenance = frame.attrs[OBSERVATION_RESOLUTION_PROVENANCE_ATTR]
    assert provenance['native_quarter_hour_truth_eligible'] is False
    assert provenance['source_observation_count'] * 4 == len(frame)
    assert result.derived_frame.attrs == result.raw_frame.attrs
    if attack == 'missing':
        del frame.attrs[OBSERVATION_RESOLUTION_PROVENANCE_ATTR]
    elif attack == 'claim_native_truth':
        provenance['native_quarter_hour_truth_eligible'] = True
    if attack == 'none':
        _validate_raw_frame('epex_ch', frame, require_resolution_provenance=False)
        quality = _quality_frame_metrics(_parquet_payload(frame), role='epex_ch', label='bronze')
        assert quality['resolution_provenance'] == provenance
    else:
        with pytest.raises(LTInputReplayError, match='provenance'):
            _validate_raw_frame('epex_ch', frame, require_resolution_provenance=False)
        with pytest.raises(LTInputReplayError, match='provenance'):
            _quality_frame_metrics(_parquet_payload(frame), role='epex_ch', label='bronze')


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
        "GenerationDirection": (
            "GENERATION" if group == "generation_actual" else None
        ),
        "FromZone": from_zone,
        "ToZone": to_zone,
        "Unit": "MW",
    }


def _mapping(dimension: pd.DataFrame | None = None) -> EntsoeFeatureMapping:
    selected_dimension = _dimension() if dimension is None else dimension
    return EntsoeFeatureMapping(
        dimension_semantic_sha256=entsoe_dimension_semantic_sha256(
            selected_dimension
        ),
        load_mw=(EntsoeSeriesTerm("LOAD"),),
        solar_mw=(EntsoeSeriesTerm("SOLAR"),),
        wind_mw=(EntsoeSeriesTerm("WIND_ON"), EntsoeSeriesTerm("WIND_OFF")),
        cross_border_mw=(
            EntsoeSeriesTerm("FLOW_EXPORT", 1.0),
            EntsoeSeriesTerm("FLOW_IMPORT", -1.0),
        ),
    )


def _mapping_contract(dimension: pd.DataFrame) -> dict[str, object]:
    mapping = _mapping(dimension)
    return {
        "schema_version": ENTSOE_FEATURE_MAPPING_SCHEMA_VERSION,
        "market": "CH",
        "dimension_semantic_sha256": entsoe_dimension_semantic_sha256(dimension),
        "features": {
            feature: [
                {"series_key": term.series_key, "weight": term.weight}
                for term in getattr(mapping, feature)
            ]
            for feature in (
                "load_mw",
                "solar_mw",
                "wind_mw",
                "cross_border_mw",
            )
        },
    }


def _parquet_payload(frame: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    frame.to_parquet(buffer, index=True)
    return buffer.getvalue()


def _eex_daily_rows() -> pd.DataFrame:
    rows = [
        {
            "ProductID": "month-base",
            "DeliveryPeriodID": "2026-09-base",
            "QuotationDateID": "2026-08-20",
            "SettlementPriceEurMWh": 71.0,
            "LastPriceEurMWh": 72.0,
            "FactLoadTimestampUtc": "2026-08-20T22:15:00Z",
            "Country": "CH",
            "Commodity": "POWER",
            "ProductType": "BASE",
            "DeliveryPeriodType": "MONTH",
            "DeliveryStartDate": "2026-09-01",
            "DeliveryEndDate": "2026-09-30",
        },
        {
            "ProductID": "quarter-base",
            "DeliveryPeriodID": "2026-q4-base",
            "QuotationDateID": "2026-08-21",
            "SettlementPriceEurMWh": 74.0,
            "LastPriceEurMWh": None,
            "FactLoadTimestampUtc": "2026-08-21T06:15:00Z",
            "Country": "CH",
            "Commodity": "POWER",
            "ProductType": "BASE",
            "DeliveryPeriodType": "QUARTER",
            "DeliveryStartDate": "2026-10-01",
            "DeliveryEndDate": "2026-12-31",
        },
        {
            "ProductID": "year-base",
            "DeliveryPeriodID": "2027-base",
            "QuotationDateID": "2026-08-22",
            "SettlementPriceEurMWh": 78.0,
            "LastPriceEurMWh": None,
            "FactLoadTimestampUtc": "2026-08-22T06:15:00Z",
            "Country": "CH",
            "Commodity": "POWER",
            "ProductType": "BASE",
            "DeliveryPeriodType": "YEAR",
            "DeliveryStartDate": "2027-01-01",
            "DeliveryEndDate": "2027-12-31",
        },
    ]
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def test_eex_materialization_enforces_observation_origin_and_solver_scope() -> None:
    result = materialize_eex_forward_history(
        _eex_daily_rows(),
        as_of_utc="2026-08-21T12:00:00Z",
    )

    assert set(result.derived_frame["product_type"]) == {"Month", "Quarter"}
    assert result.audit["source_rows"] == 3
    assert result.audit["eligible_rows"] == 2
    assert result.audit["excluded_after_origin_rows"] == 1
    assert result.audit["authorities"]["model_input_authorized"] is False
    assert result.audit["normalization_audit"]["authority"][
        "point_in_time_availability_proven"
    ] is False


def test_eex_materialization_is_stable_under_source_row_order() -> None:
    source = _eex_daily_rows()
    first = materialize_eex_forward_history(
        source,
        as_of_utc="2026-08-23T12:00:00Z",
    )
    second = materialize_eex_forward_history(
        source.sample(frac=1.0, random_state=17),
        as_of_utc="2026-08-23T12:00:00Z",
    )

    assert first.audit["source_projection_sha256"] == second.audit[
        "source_projection_sha256"
    ]
    pd.testing.assert_frame_equal(first.raw_frame, second.raw_frame)
    pd.testing.assert_frame_equal(first.derived_frame, second.derived_frame)


@pytest.mark.parametrize("date_encoding", ["integer", "compact_string", "mixed"])
def test_eex_prd_date_keys_preserve_calendar_dates_and_cutoff(date_encoding: str) -> None:
    source = _eex_daily_rows()
    expected = materialize_eex_forward_history(source, as_of_utc="2026-08-21T12:00:00Z")
    keys = [20260820, 20260821, 20260822]
    if date_encoding == "compact_string":
        keys = [str(key) for key in keys]
    elif date_encoding == "mixed":
        keys = [20260820, "2026-08-21", "20260822"]
    source["QuotationDateID"] = keys
    actual = materialize_eex_forward_history(source, as_of_utc="2026-08-21T12:00:00Z")
    pd.testing.assert_frame_equal(actual.raw_frame, expected.raw_frame)
    pd.testing.assert_frame_equal(actual.derived_frame, expected.derived_frame)
    assert actual.audit == expected.audit


@pytest.mark.parametrize("date_key", [20260230, 20261301, 0])
def test_eex_materialization_rejects_invalid_prd_date_keys(date_key: int) -> None:
    source = _eex_daily_rows()
    source.loc[0, "QuotationDateID"] = date_key
    with pytest.raises(DatabricksLTMaterializationError, match="valid timezone-naive dates"):
        materialize_eex_forward_history(source, as_of_utc="2026-08-23T12:00:00Z")


def test_eex_materialization_rejects_naive_origin() -> None:
    with pytest.raises(DatabricksLTMaterializationError, match="timezone-aware"):
        materialize_eex_forward_history(
            _eex_daily_rows(),
            as_of_utc="2026-08-21T12:00:00",
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


def _local_observed(vintages: pd.DataFrame, *, as_of: str = "2026-10-24T04:00:00Z"):
    return materialize_entsoe_latest_observed_features(
        dimension=_dimension(), silver_vintages=vintages, mapping=_mapping(),
        window_start_utc="2026-10-24T00:00:00Z",
        window_end_utc="2026-10-24T02:00:00Z", as_of_utc=as_of,
    )


def test_local_observed_replays_changed_blocks_without_claiming_pit() -> None:
    source = _silver_vintages()
    first_load = source["SK_ge_power_entsoe_time_series_vintages"].eq("V-1")
    source.loc[first_load, ["IntervalEndUtc", "Date_Time_UTC"]] = pd.Timestamp("2026-10-24T02:00:00Z")
    source.loc[first_load, ["first_seen_pull_ts_utc", "last_seen_pull_ts_utc"]] = pd.Timestamp("2026-10-24T02:20:00Z")
    # PRD response creation follows the pull start; it is not historical publication.
    source["publication_timestamp_utc"] = source["first_seen_pull_ts_utc"] + pd.Timedelta(minutes=3)
    result = _local_observed(source)
    assert result.raw_frame["load_mw"].tolist() == [999.0] * 4 + [110.0] * 4
    assert result.audit["mode"] == "SILVER_LATEST_OBSERVED_LOCAL_ONLY"
    assert result.audit["selection_metrics"]["superseded_or_repeated_transport_rows"] == 8
    assert not any(result.audit["authorities"].values())
    shuffled = _local_observed(source.sample(frac=1, random_state=9))
    pd.testing.assert_frame_equal(result.raw_frame, shuffled.raw_frame)
    assert result.audit == shuffled.audit


def test_local_observed_excludes_later_observations() -> None:
    result = _local_observed(_silver_vintages(), as_of="2026-10-24T03:00:00Z")
    assert result.raw_frame["load_mw"].tolist() == [100.0] * 4 + [110.0] * 4
    assert result.audit["selection_metrics"]["excluded_source_rows"] == 1


def test_local_observed_rejects_conflicting_equal_order_overlap() -> None:
    source = _silver_vintages().iloc[:-1].copy()
    source.loc[0, ["IntervalEndUtc", "Date_Time_UTC"]] = pd.Timestamp("2026-10-24T02:00:00Z")
    with pytest.raises(DatabricksLTMaterializationError, match="equal-order overlap"):
        _local_observed(source)


@pytest.mark.parametrize("defect", ["gap", "first_after_last", "off_grid", "string_dq", "nonfinite"])
def test_local_observed_rejects_invalid_inputs(defect: str) -> None:
    source = _silver_vintages()
    if defect == "gap":
        source = source.loc[~source["SK_ge_power_entsoe_time_series_vintages"].eq("V-2")]
    elif defect == "first_after_last":
        source.loc[0, "first_seen_pull_ts_utc"] = pd.Timestamp("2026-10-24T05:00:00Z")
    elif defect == "off_grid":
        source.loc[0, "IntervalStartUtc"] += pd.Timedelta(milliseconds=1)
    elif defect == "string_dq":
        source["dq_failed"] = source["dq_failed"].astype(object)
        source.loc[0, "dq_failed"] = "false"
    else:
        source.loc[0, "field_value"] = float("inf")
    with pytest.raises(DatabricksLTMaterializationError):
        _local_observed(source)


def test_entsoe_accepts_observed_prd_quantity_names_and_a11_null_process() -> None:
    dimension = _dimension()
    dimension["FieldName"] = ["ch_actual_load", "ch_solar_actual", "ch_wind_onshore_actual",
                              "ch_wind_offshore_actual", "ch_to_de_lu_crossborder_flow", "de_lu_to_ch_crossborder_flow"]
    dimension.loc[dimension["GroupName"].eq("crossborder_physical_flows"), "ProcessType"] = None
    observed = materialize_entsoe_latest_observed_features(
        dimension=dimension, silver_vintages=_silver_vintages(), mapping=_mapping(dimension),
        window_start_utc="2026-10-24T00:00:00Z", window_end_utc="2026-10-24T02:00:00Z",
        as_of_utc="2026-10-24T04:00:00Z",
    )
    assert observed.raw_frame["cross_border_mw"].eq(30.0).all()
    dimension.loc[dimension["SeriesKey"].eq("FLOW_EXPORT"), "FieldName"] = "de_lu_to_ch_crossborder_flow"
    with pytest.raises(DatabricksLTMaterializationError, match="quantity field"):
        materialize_entsoe_latest_observed_features(
            dimension=dimension, silver_vintages=_silver_vintages(), mapping=_mapping(dimension),
            window_start_utc="2026-10-24T00:00:00Z", window_end_utc="2026-10-24T02:00:00Z",
            as_of_utc="2026-10-24T04:00:00Z",
        )


def test_local_observed_preserves_arrow_microsecond_timestamps() -> None:
    source = _silver_vintages()
    expected = _local_observed(source)
    for column in ("IntervalStartUtc", "IntervalEndUtc", "Date_Time_UTC"):
        source[column] = source[column].astype("datetime64[us, UTC]")
    actual = _local_observed(source)
    pd.testing.assert_frame_equal(actual.raw_frame, expected.raw_frame)


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
            mapping=_mapping(dimension),
            as_of_utc="2026-10-24T03:00:00Z",
        )


def test_entsoe_mapping_rejects_generation_consumption_series() -> None:
    dimension = _dimension()
    dimension.loc[dimension["SeriesKey"].eq("SOLAR"), "BusinessType"] = "A04"

    with pytest.raises(DatabricksLTMaterializationError, match="wrong BusinessType"):
        materialize_entsoe_pit_features(
            dimension=dimension,
            silver_vintages=_silver_vintages(),
            mapping=_mapping(dimension),
            as_of_utc="2026-10-24T03:00:00Z",
        )


def test_entsoe_mapping_contract_is_bound_to_dimension_semantics() -> None:
    dimension = _dimension()
    expected = _mapping(dimension)
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


def test_silver_unknown_backfill_accepts_missing_publication_but_not_for_pit() -> None:
    vintages = _silver_vintages()
    row = vintages["SK_ge_power_entsoe_time_series_vintages"].eq("V-1")
    vintages.loc[row, "publication_timestamp_utc"] = pd.NaT
    vintages.loc[row, "availability_basis"] = "UNKNOWN_BACKFILL"
    vintages.loc[row, "availability_known"] = False
    vintages.loc[row, "availability_timestamp_utc"] = pd.NaT

    with pytest.raises(DatabricksLTMaterializationError, match="coverage differs"):
        materialize_entsoe_pit_features(
            dimension=_dimension(),
            silver_vintages=vintages,
            mapping=_mapping(),
            as_of_utc="2026-10-24T03:00:00Z",
        )


def test_entsoe_mapping_requires_dimension_binding_even_without_contract_parser() -> None:
    dimension = _dimension()
    mapping = _mapping(dimension)
    changed = dimension.copy()
    changed.loc[changed["SeriesKey"].eq("LOAD"), "SourceTimeSeriesId"] = "OTHER"

    with pytest.raises(DatabricksLTMaterializationError, match="not bound"):
        materialize_entsoe_pit_features(
            dimension=changed,
            silver_vintages=_silver_vintages(),
            mapping=mapping,
            as_of_utc="2026-10-24T03:00:00Z",
        )


def test_entsoe_mapping_rejects_wrong_generation_direction() -> None:
    dimension = _dimension()
    dimension.loc[
        dimension["SeriesKey"].eq("SOLAR"), "GenerationDirection"
    ] = "CONSUMPTION"

    with pytest.raises(DatabricksLTMaterializationError, match="GenerationDirection"):
        materialize_entsoe_pit_features(
            dimension=dimension,
            silver_vintages=_silver_vintages(),
            mapping=_mapping(dimension),
            as_of_utc="2026-10-24T03:00:00Z",
        )


def test_entsoe_mapping_rejects_wrong_cross_border_sign() -> None:
    dimension = _dimension()
    mapping = EntsoeFeatureMapping(
        dimension_semantic_sha256=entsoe_dimension_semantic_sha256(dimension),
        load_mw=(EntsoeSeriesTerm("LOAD"),),
        solar_mw=(EntsoeSeriesTerm("SOLAR"),),
        wind_mw=(EntsoeSeriesTerm("WIND_ON"), EntsoeSeriesTerm("WIND_OFF")),
        cross_border_mw=(
            EntsoeSeriesTerm("FLOW_EXPORT", -1.0),
            EntsoeSeriesTerm("FLOW_IMPORT", 1.0),
        ),
    )

    with pytest.raises(DatabricksLTMaterializationError, match="NET_EXPORT_FROM_CH"):
        materialize_entsoe_pit_features(
            dimension=dimension,
            silver_vintages=_silver_vintages(),
            mapping=mapping,
            as_of_utc="2026-10-24T03:00:00Z",
        )


def test_silver_source_hash_is_stable_under_export_row_order() -> None:
    vintages = _silver_vintages()
    first = materialize_entsoe_pit_features(
        dimension=_dimension(),
        silver_vintages=vintages,
        mapping=_mapping(),
        as_of_utc="2026-10-24T03:00:00Z",
    )
    second = materialize_entsoe_pit_features(
        dimension=_dimension().sample(frac=1.0, random_state=4),
        silver_vintages=vintages.sample(frac=1.0, random_state=7),
        mapping=_mapping(),
        as_of_utc="2026-10-24T03:00:00Z",
    )

    assert first.audit["source_projection_sha256"] == second.audit[
        "source_projection_sha256"
    ]
    assert first.audit["dimension_projection_sha256"] == second.audit[
        "dimension_projection_sha256"
    ]
    pd.testing.assert_frame_equal(first.raw_frame, second.raw_frame)


def test_silver_rejects_string_boolean_dq_flag() -> None:
    vintages = _silver_vintages()
    vintages["dq_failed"] = "False"

    with pytest.raises(DatabricksLTMaterializationError, match="non-null booleans"):
        materialize_entsoe_pit_features(
            dimension=_dimension(),
            silver_vintages=vintages,
            mapping=_mapping(),
            as_of_utc="2026-10-24T03:00:00Z",
        )


def test_silver_rejects_subsecond_interval_drift() -> None:
    vintages = _silver_vintages()
    row = vintages["SK_ge_power_entsoe_time_series_vintages"].eq("V-1")
    vintages.loc[row, "IntervalEndUtc"] = (
        vintages.loc[row, "IntervalEndUtc"] + pd.Timedelta(milliseconds=500)
    )
    vintages.loc[row, "Date_Time_UTC"] = vintages.loc[row, "IntervalEndUtc"]

    with pytest.raises(DatabricksLTMaterializationError, match="atomic"):
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


def test_gold_latest_source_hash_is_stable_under_export_row_order() -> None:
    first = materialize_entsoe_current_features(
        dimension=_dimension(),
        gold_latest=_gold_latest(),
        mapping=_mapping(),
        as_of_utc="2026-10-24T03:00:00Z",
    )
    second = materialize_entsoe_current_features(
        dimension=_dimension().sample(frac=1.0, random_state=8),
        gold_latest=_gold_latest().sample(frac=1.0, random_state=9),
        mapping=_mapping(),
        as_of_utc="2026-10-24T03:00:00Z",
    )

    assert first.audit["source_projection_sha256"] == second.audit[
        "source_projection_sha256"
    ]
    pd.testing.assert_frame_equal(first.raw_frame, second.raw_frame)


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


@pytest.mark.parametrize(
    ("frequency_minutes", "extra_seconds"),
    [(15.5, 0), (15.0, 30)],
)
def test_spot_materialization_rejects_fractional_interval_declarations(
    frequency_minutes: float,
    extra_seconds: int,
) -> None:
    rows = _spot_rows().iloc[[0]].copy()
    rows["FrequencyMinutes"] = frequency_minutes
    rows["DeliveryEndUtc"] = (
        rows["DeliveryStartUtc"]
        + pd.Timedelta(minutes=15)
        + pd.Timedelta(seconds=extra_seconds)
    )

    with pytest.raises(DatabricksLTMaterializationError, match="atomic"):
        materialize_spot_price_history(
            rows,
            market_zone="CH",
            source_product="CH_DAY_AHEAD",
            as_of_utc="2026-10-25T05:00:00Z",
        )


def test_databricks_spot_export_replays_exact_archived_frames() -> None:
    source_payloads = {"spot_price_interval": _parquet_payload(_spot_rows())}
    replay = build_databricks_role_replay(
        role="epex_ch",
        mode=GOLD_SPOT_MODE,
        as_of_utc="2026-10-25T05:00:00Z",
        source_payloads=source_payloads,
        source_tables={"spot_price_interval": "dev.gold.factspotpriceinterval"},
        selection={
            "market_zone": "CH",
            "source_product": "CH_DAY_AHEAD",
        },
    )
    materializer = approved_databricks_materializer_payload()

    result = verify_databricks_role_replay(
        config=replay.config,
        source_payloads=source_payloads,
        raw_payload=_parquet_payload(replay.materialization.raw_frame),
        derived_payload=_parquet_payload(replay.materialization.derived_frame),
        materializer_payload=materializer,
        materializer_code_sha256=hashlib.sha256(materializer).hexdigest(),
    )

    assert result["status"] == "VERIFIED_EXACT_DATABRICKS_EXPORT_REPLAY"
    assert result["source_environment"] == "DEV"
    assert result["authorities"]["model_input_authorized"] is False


def test_databricks_silver_pit_export_replays_exact_archived_frames() -> None:
    dimension = _dimension()
    source_payloads = {
        "entsoe_series_dimension": _parquet_payload(dimension),
        "entsoe_vintages": _parquet_payload(_silver_vintages()),
    }
    replay = build_databricks_role_replay(
        role="entso",
        mode=SILVER_ENTSOE_PIT_MODE,
        as_of_utc="2026-10-24T03:00:00Z",
        source_payloads=source_payloads,
        source_tables={
            "entsoe_series_dimension": "prd.gold.dimentsoeseries",
            "entsoe_vintages": (
                "prd.silver.ge_power_entsoe_time_series_vintages"
            ),
        },
        selection={"mapping_contract": _mapping_contract(dimension)},
    )
    materializer = approved_databricks_materializer_payload()

    result = verify_databricks_role_replay(
        config=replay.config,
        source_payloads=source_payloads,
        raw_payload=_parquet_payload(replay.materialization.raw_frame),
        derived_payload=_parquet_payload(replay.materialization.derived_frame),
        materializer_payload=materializer,
        materializer_code_sha256=hashlib.sha256(materializer).hexdigest(),
    )

    assert result["source_environment"] == "PRD"
    assert result["source_artifact_count"] == 2


def test_databricks_gold_current_export_builds_replay_config() -> None:
    dimension = _dimension()
    replay = build_databricks_role_replay(
        role="entso",
        mode=GOLD_ENTSOE_CURRENT_MODE,
        as_of_utc="2026-10-24T03:00:00Z",
        source_payloads={
            "entsoe_series_dimension": _parquet_payload(dimension),
            "entsoe_latest": _parquet_payload(_gold_latest()),
        },
        source_tables={
            "entsoe_series_dimension": "dev.gold.dimentsoeseries",
            "entsoe_latest": "dev.gold.factentsoetimeserieslatest",
        },
        selection={"mapping_contract": _mapping_contract(dimension)},
    )

    assert replay.config["mode"] == GOLD_ENTSOE_CURRENT_MODE
    assert replay.materialization.audit["mode"] == "GOLD_CURRENT_SERVING"


def test_databricks_replay_rejects_changed_source_export() -> None:
    source_payloads = {"spot_price_interval": _parquet_payload(_spot_rows())}
    replay = build_databricks_role_replay(
        role="epex_ch",
        mode=GOLD_SPOT_MODE,
        as_of_utc="2026-10-25T05:00:00Z",
        source_payloads=source_payloads,
        source_tables={"spot_price_interval": "dev.gold.factspotpriceinterval"},
        selection={
            "market_zone": "CH",
            "source_product": "CH_DAY_AHEAD",
        },
    )
    changed = _spot_rows()
    changed.loc[0, "Price"] = 999.0
    materializer = approved_databricks_materializer_payload()

    with pytest.raises(DatabricksLTReplayError, match="source artifact changed"):
        verify_databricks_role_replay(
            config=replay.config,
            source_payloads={"spot_price_interval": _parquet_payload(changed)},
            raw_payload=_parquet_payload(replay.materialization.raw_frame),
            derived_payload=_parquet_payload(replay.materialization.derived_frame),
            materializer_payload=materializer,
            materializer_code_sha256=hashlib.sha256(materializer).hexdigest(),
        )


def test_databricks_replay_rejects_mixed_dev_prd_sources() -> None:
    dimension = _dimension()
    with pytest.raises(DatabricksLTReplayError, match="mix DEV and PRD"):
        build_databricks_role_replay(
            role="entso",
            mode=SILVER_ENTSOE_PIT_MODE,
            as_of_utc="2026-10-24T03:00:00Z",
            source_payloads={
                "entsoe_series_dimension": _parquet_payload(dimension),
                "entsoe_vintages": _parquet_payload(_silver_vintages()),
            },
            source_tables={
                "entsoe_series_dimension": "prd.gold.dimentsoeseries",
                "entsoe_vintages": (
                    "dev.silver.ge_power_entsoe_time_series_vintages"
                ),
            },
            selection={"mapping_contract": _mapping_contract(dimension)},
        )


def test_databricks_replay_package_is_self_contained_and_tamper_evident() -> None:
    package = build_databricks_replay_package(
        role="epex_ch",
        mode=GOLD_SPOT_MODE,
        as_of_utc="2026-10-25T05:00:00Z",
        source_payloads={"spot_price_interval": _parquet_payload(_spot_rows())},
        source_tables={"spot_price_interval": "dev.gold.factspotpriceinterval"},
        selection={
            "market_zone": "CH",
            "source_product": "CH_DAY_AHEAD",
        },
    )

    verified = verify_databricks_replay_package(
        artifacts=package.artifacts,
        manifest_payload=package.manifest_payload,
    )

    assert verified["status"] == (
        "VERIFIED_SELF_CONTAINED_DATABRICKS_REPLAY_PACKAGE"
    )
    assert verified["artifact_count"] == 6
    changed = dict(package.artifacts)
    changed["model/raw.parquet"] = changed["model/raw.parquet"] + b"tampered"
    with pytest.raises(DatabricksLTReplayError, match="artifact changed"):
        verify_databricks_replay_package(
            artifacts=changed,
            manifest_payload=package.manifest_payload,
        )
