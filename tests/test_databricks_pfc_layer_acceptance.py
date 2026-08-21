from __future__ import annotations

import pandas as pd

from pfc_shaping.validation.databricks_pfc_layer_acceptance import (
    DEFAULT_REQUIRED_GROUPS,
    assess_pfc_data_layers,
)


def _unit(group: str) -> str:
    if group in {"day_ahead_prices", "imbalance_prices"}:
        return "EUR/MWh"
    if group in {"hydro_reservoir_storage", "imbalance_volumes"}:
        return "MWh"
    return "MW"


def _fixtures() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    border_groups = {
        "scheduled_exchanges",
        "crossborder_physical_flows",
        "ntc_day_ahead",
        "ntc_month_ahead",
        "ntc_year_ahead",
        "ntc_intraday",
    }
    neighbors = ("AT", "DE_LU", "FR", "IT_NORD")
    rows: list[dict[str, object]] = []
    series_id = 1
    for group in DEFAULT_REQUIRED_GROUPS:
        directions = (
            [("CH", neighbor) for neighbor in neighbors]
            + [(neighbor, "CH") for neighbor in neighbors]
            if group in border_groups
            else [("CH", None)]
        )
        for from_zone, to_zone in directions:
            rows.append(
                {
                    "SeriesID": series_id,
                    "SeriesKey": f"SERIES-{series_id}",
                    "FieldName": f"{group}_{from_zone}_{to_zone or 'NA'}",
                    "SourceTimeSeriesId": f"TS-{series_id}",
                    "GroupName": group,
                    "DocumentType": "A00",
                    "BusinessType": None,
                    "ProcessType": None,
                    "PsrType": None,
                    "FromZone": from_zone,
                    "ToZone": to_zone,
                    "Unit": _unit(group),
                }
            )
            series_id += 1
    dimension = pd.DataFrame(rows)
    target_start = pd.Timestamp("2019-01-01T00:00:00Z")
    target_end = pd.Timestamp("2019-01-01T01:00:00Z")
    publication = pd.Timestamp("2018-12-31T12:00:00Z")
    first_observed = pd.Timestamp("2018-12-31T12:01:00Z")
    latest = pd.DataFrame(
        {
            "SeriesID": dimension["SeriesID"],
            "IntervalStartUtc": target_start,
            "DateTimeUtc": target_end,
            "IntervalEndUtc": target_end,
            "Resolution": "PT60M",
            "FieldValue": 1.0,
            "PublicationTimestampUtc": publication,
            "AvailabilityBasis": "SOURCE_DOCUMENT_CREATED",
            "AvailabilityKnown": True,
            "AvailabilityTimestampUtc": publication,
            "IsHistorical": True,
        }
    )
    vintages = pd.DataFrame(
        {
            "SK_ge_power_entsoe_time_series_vintages": [
                f"V-{value}" for value in dimension["SeriesID"]
            ],
            "series_key": dimension["SeriesKey"],
            "field_value": 1.0,
            "IntervalStartUtc": target_start,
            "Date_Time_UTC": target_end,
            "IntervalEndUtc": target_end,
            "resolution": "PT60M",
            "publication_timestamp_utc": publication,
            "Is_Historical": True,
            "resource_details": [[] for _ in range(len(dimension))],
            "first_seen_pull_ts_utc": first_observed,
            "last_seen_pull_ts_utc": first_observed,
            "availability_basis": "SOURCE_DOCUMENT_CREATED",
            "availability_known": True,
            "availability_timestamp_utc": publication,
            "source_document_mrid": [
                f"DOC-{value}" for value in dimension["SeriesID"]
            ],
            "source_document_revision_number": 1,
            "dq_failed": False,
        }
    )
    spot = pd.DataFrame(
        {
            "SpotProductID": [1],
            "SourceProduct": ["CH_DA"],
            "MarketZone": ["CH"],
            "DeliveryStartUtc": [target_start],
            "DeliveryEndUtc": [target_end],
            "FrequencyMinutes": [60],
            "Price": [75.0],
            "PriceUnit": ["EUR/MWh"],
            "ObservedAtUtc": [publication],
        }
    )
    return (
        {
            "entsoe_series_dimension": dimension,
            "entsoe_latest": latest,
            "spot_price_interval": spot,
        },
        {"entsoe_vintages": vintages},
    )


def _codes(report) -> set[str]:
    return {finding.code for finding in report.findings}


def test_gold_serving_and_silver_vintages_pass() -> None:
    gold, silver = _fixtures()
    report = assess_pfc_data_layers(gold=gold, silver=silver)

    assert report.status == "PASS_GOVERNED_LAYERS"
    assert report.findings == ()
    policy = report.as_dict()["layer_policy"]
    assert policy["current_serving_authority"] == "GOLD"
    assert policy["entsoe_point_in_time_authority"].startswith("SILVER_")
    assert report.as_dict()["authorities"] == {
        "semantic_acceptance_passed": True,
        "model_input_authorized": False,
        "production_authorized": False,
    }


def test_missing_silver_vintages_is_blocking() -> None:
    gold, _ = _fixtures()
    report = assess_pfc_data_layers(gold=gold)

    assert report.status == "BLOCKED_GOVERNED_LAYERS"
    assert "ENTSOE_SILVER_VINTAGES_MISSING" in _codes(report)


def test_price_units_and_spot_quotation_misuse_are_blocking() -> None:
    gold, silver = _fixtures()
    dimension = gold["entsoe_series_dimension"]
    dimension.loc[dimension["GroupName"].eq("day_ahead_prices"), "Unit"] = "EUR"
    spot = gold["spot_price_interval"]
    spot["QuotationTimestampUtc"] = spot["DeliveryStartUtc"]

    report = assess_pfc_data_layers(gold=gold, silver=silver)

    assert report.status == "BLOCKED_GOVERNED_LAYERS"
    assert "ENTSOE_UNIT_MISMATCH" in _codes(report)
    assert "SPOT_QUOTATION_EQUALS_DELIVERY_START" in _codes(report)


def test_silver_interval_and_timestamp_failures_are_detected() -> None:
    gold, silver = _fixtures()
    gold["entsoe_latest"] = pd.concat(
        [gold["entsoe_latest"], gold["entsoe_latest"].iloc[[0]]],
        ignore_index=True,
    )
    vintages = silver["entsoe_vintages"]
    vintages.loc[0, "IntervalStartUtc"] = pd.Timestamp("2019-01-01T00:30:00Z")
    vintages.loc[0, "first_seen_pull_ts_utc"] = pd.Timestamp(
        "2018-12-31T11:59:00Z"
    )

    report = assess_pfc_data_layers(gold=gold, silver=silver)

    assert report.status == "BLOCKED_GOVERNED_LAYERS"
    assert "ENTSOE_LATEST_GRAIN_DUPLICATED" in _codes(report)
    assert "ENTSOE_INTERVAL_INCONSISTENT" in _codes(report)
    assert "VINTAGE_TIMESTAMP_ORDER_INVALID" in _codes(report)


def test_unknown_backfill_is_not_accepted_as_pit_truth() -> None:
    gold, silver = _fixtures()
    vintages = silver["entsoe_vintages"]
    vintages.loc[0, "availability_basis"] = "UNKNOWN_BACKFILL"
    vintages.loc[0, "availability_known"] = False
    vintages.loc[0, "availability_timestamp_utc"] = pd.NaT

    report = assess_pfc_data_layers(gold=gold, silver=silver)

    assert report.status == "REVIEW_GOVERNED_LAYERS"
    assert "VINTAGE_PIT_AVAILABILITY_UNKNOWN" in _codes(report)


def test_current_resource_bridge_is_optional_but_validated_when_present() -> None:
    gold, silver = _fixtures()
    dimension = gold["entsoe_series_dimension"]
    outage = dimension.iloc[[0]].copy()
    outage["SeriesID"] = int(dimension["SeriesID"].max()) + 1
    outage["SeriesKey"] = "SERIES-OUTAGE"
    outage["GroupName"] = "production_unit_unavailability"
    outage["Unit"] = "MW"
    gold["entsoe_series_dimension"] = pd.concat(
        [dimension, outage], ignore_index=True
    )
    vintage = silver["entsoe_vintages"].iloc[[0]].copy()
    vintage["SK_ge_power_entsoe_time_series_vintages"] = "V-OUTAGE"
    vintage["series_key"] = "SERIES-OUTAGE"
    vintage["resource_details"] = [[{"resource_mrid": "R-1"}]]
    silver["entsoe_vintages"] = pd.concat(
        [silver["entsoe_vintages"], vintage], ignore_index=True
    )

    without_bridge = assess_pfc_data_layers(gold=gold, silver=silver)
    assert without_bridge.status == "PASS_GOVERNED_LAYERS"
    assert "ENTSOE_CURRENT_RESOURCE_BRIDGE_MISSING" in _codes(without_bridge)

    gold["entsoe_series_resources_current"] = pd.DataFrame(
        {
            "BridgeID": ["B-1"],
            "SeriesID": [outage["SeriesID"].iloc[0]],
            "ResourceRole": ["PRODUCTION_UNIT"],
            "ResourceMrid": ["R-1"],
        }
    )
    with_bridge = assess_pfc_data_layers(gold=gold, silver=silver)
    assert with_bridge.status == "PASS_GOVERNED_LAYERS"
    assert "ENTSOE_CURRENT_RESOURCE_BRIDGE_MISSING" not in _codes(with_bridge)
