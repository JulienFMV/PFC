from __future__ import annotations

import re

import pandas as pd
import pytest

from pfc_shaping.validation.entsoe_day_ahead_export import (
    RAW_COLUMNS,
    REALIZED_SQL_SHA256,
    DayAheadMarketUse,
    validate_realized_latest_candidate,
)
from pfc_shaping.validation.entsoe_day_ahead_prd import (
    EXPECTED_FIELDS,
    PIT_COLUMNS,
    PIT_SQL_SHA256,
    validate_day_ahead_pit_extract,
)
from pfc_shaping.validation.spot_source_reconciliation import (
    BLOCKED_STATUS,
    LSEG_CURVES,
    LSEG_LATEST_SQL_PATH,
    LSEG_LATEST_SQL_SHA256,
    LSEG_PIT_COLUMNS,
    LSEG_PIT_SQL_PATH,
    LSEG_PIT_SQL_SHA256,
    PASS_STATUS,
    SpotReconciliationPolicy,
    SpotSourceReconciliationError,
    build_lseg_epex_actuals_latest_parameters,
    build_lseg_epex_actuals_pit_parameters,
    reconcile_lseg_entsoe_latest_candidate,
    reconcile_lseg_entsoe_spot,
    validate_lseg_epex_actuals_latest_extract,
    validate_lseg_epex_actuals_pit_extract,
    verify_lseg_latest_sql_binding,
    verify_lseg_sql_binding,
)

START = "2026-01-01T00:00:00Z"
END = "2026-01-01T02:00:00Z"
AS_OF = "2026-01-02T00:00:00Z"

FIELD_TO_ZONE = {
    "ch_price": "CH",
    "at_price": "AT",
    "de_lu_price": "DE_LU",
    "fr_price": "FR",
    "it_nord_price": "IT_NORD",
}
ZONE_BASE = {"CH": 10.0, "AT": 20.0, "DE_LU": 30.0, "FR": 40.0, "IT_NORD": 50.0}


def _binding() -> dict[str, str]:
    return {
        "ch_price": "day_ahead_prices||ch_price",
        "at_price": "day_ahead_prices||at_price||1",
        "de_lu_price": "day_ahead_prices||de_lu_price||1",
        "fr_price": "day_ahead_prices||fr_price",
        "it_nord_price": "day_ahead_prices||it_nord_price",
    }


def _entsoe_frame(
    *,
    start: str = START,
    end: str = END,
    frequency: str = "15min",
) -> pd.DataFrame:
    resolution = {"15min": "PT15M", "30min": "PT30M", "h": "PT1H"}[frequency]
    duration = pd.Timedelta(frequency)
    rows: list[dict[str, object]] = []
    for field in EXPECTED_FIELDS:
        zone = FIELD_TO_ZONE[field]
        for index, interval_start in enumerate(
            pd.date_range(start=start, end=end, freq=frequency, inclusive="left")
        ):
            hour_offset = int((interval_start - pd.Timestamp(start)) / pd.Timedelta(hours=1))
            rows.append(
                {
                    "field_name": field,
                    "series_key": _binding()[field],
                    "interval_start_utc": interval_start,
                    "interval_end_utc": interval_start + duration,
                    "resolution": resolution,
                    "price_eur_per_mwh": ZONE_BASE[zone] + hour_offset,
                    "availability_timestamp_utc": "2025-12-31T12:00:00Z",
                    "source_document_mrid": f"DOC-{field}-{index}",
                    "source_document_revision_number": 1,
                }
            )
    return pd.DataFrame(rows, columns=PIT_COLUMNS)


def _lseg_frame(
    *,
    start: str = START,
    end: str = END,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for zone, (curve_id, resolution) in LSEG_CURVES.items():
        frequency = "h" if resolution == "PT1H" else "15min"
        duration = pd.Timedelta(hours=1) if frequency == "h" else pd.Timedelta(minutes=15)
        for index, interval_start in enumerate(
            pd.date_range(start=start, end=end, freq=frequency, inclusive="left")
        ):
            hour_offset = int((interval_start - pd.Timestamp(start)) / pd.Timedelta(hours=1))
            rows.append(
                {
                    "market_zone": zone,
                    "curve_id": curve_id,
                    "interval_start_utc": interval_start,
                    "interval_end_utc": interval_start + duration,
                    "resolution": resolution,
                    "price_eur_per_mwh": ZONE_BASE[zone] + hour_offset,
                    "pipeline_first_seen_at_utc": "2026-01-01T12:00:00Z",
                    "pull_ts_utc": "2026-01-01T13:00:00Z",
                    "curve_value_vintage_id": f"V-{zone}-{index}",
                }
            )
    return pd.DataFrame(rows, columns=LSEG_PIT_COLUMNS)


def _candidate_raw_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, source in _entsoe_frame().iterrows():
        series_key = str(source["series_key"])
        classification = series_key.split("||")[2:]
        rows.append(
            {
                "field_name": source["field_name"],
                "series_key": series_key,
                "classification_sequence": classification[0] if classification else None,
                "interval_start_utc": source["interval_start_utc"],
                "date_time_utc": source["interval_end_utc"],
                "interval_end_utc": source["interval_end_utc"],
                "resolution": source["resolution"],
                "price_eur_per_mwh": source["price_eur_per_mwh"],
                "publication_timestamp_utc": "2025-12-31T12:00:00Z",
                "first_seen_pull_ts_utc": "2025-12-31T12:00:00Z",
                "availability_basis": "FMV_FIRST_SEEN",
                "availability_known": True,
                "availability_timestamp_utc": "2025-12-31T12:00:00Z",
                "is_historical": False,
                "dq_failed": False,
                "source_time_series_id": f"TS-{index}",
                "source_document_mrid": f"DOC-{index}",
                "source_document_revision_number": 1,
                "source_snapshot_id": f"SNAP-{index}",
                "source_file_path": f"entsoe/{index}.xml",
                "vintage_id": f"VINTAGE-{index}",
            }
        )
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def _candidate_artifact(raw: pd.DataFrame):
    return validate_realized_latest_candidate(
        raw,
        series_selection=_binding(),
        market_uses={
            "ch_price": DayAheadMarketUse.VALUATION_HEDGE_SCOPE,
            "at_price": DayAheadMarketUse.OBSERVATION_RISK,
            "de_lu_price": DayAheadMarketUse.VALUATION_HEDGE_SCOPE,
            "fr_price": DayAheadMarketUse.OBSERVATION_RISK,
            "it_nord_price": DayAheadMarketUse.OBSERVATION_RISK,
        },
        window_start_utc=START,
        window_end_utc=END,
        assessed_at_utc=AS_OF,
        query_sha256=REALIZED_SQL_SHA256,
    )


def _entsoe_artifact(
    frame: pd.DataFrame | None = None,
    *,
    start: str = START,
    end: str = END,
    as_of: str = AS_OF,
):
    return validate_day_ahead_pit_extract(
        _entsoe_frame(start=start, end=end) if frame is None else frame,
        series_selection=_binding(),
        window_start_utc=start,
        window_end_utc=end,
        as_of_utc=as_of,
        pit_query_sha256=PIT_SQL_SHA256,
    )


def _lseg_artifact(
    frame: pd.DataFrame | None = None,
    *,
    start: str = START,
    end: str = END,
    as_of: str = AS_OF,
):
    return validate_lseg_epex_actuals_pit_extract(
        _lseg_frame(start=start, end=end) if frame is None else frame,
        window_start_utc=start,
        window_end_utc=end,
        as_of_utc=as_of,
        pit_query_sha256=LSEG_PIT_SQL_SHA256,
    )


def _lseg_latest_artifact(
    frame: pd.DataFrame | None = None,
    *,
    start: str = START,
    end: str = END,
    assessed_at: str = AS_OF,
):
    return validate_lseg_epex_actuals_latest_extract(
        _lseg_frame(start=start, end=end) if frame is None else frame,
        window_start_utc=start,
        window_end_utc=end,
        assessed_at_utc=assessed_at,
        latest_query_sha256=LSEG_LATEST_SQL_SHA256,
    )


def _policy(**overrides: object) -> SpotReconciliationPolicy:
    values: dict[str, object] = {
        "policy_id": "spot-reconciliation-test-v1",
        "minimum_matched_hours_per_zone": 2,
        "minimum_hourly_overlap_ratio": 1.0,
        "maximum_p95_abs_difference_eur_per_mwh": 0.0,
        "maximum_absolute_bias_eur_per_mwh": 0.0,
        "maximum_single_hour_abs_difference_eur_per_mwh": 0.0,
    }
    values.update(overrides)
    return SpotReconciliationPolicy(**values)  # type: ignore[arg-type]


def _codes(report) -> set[str]:
    return {finding.code for finding in report.findings}


def test_sql_is_hash_bound_bounded_partition_pruned_and_price_only() -> None:
    assert verify_lseg_sql_binding() == LSEG_PIT_SQL_SHA256
    sql = LSEG_PIT_SQL_PATH.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", sql.lower())

    assert "prd.silver.ge_market_lseg_curve_value_vintages" in normalized
    assert "v.value_date >= p.start_value_date" in normalized
    assert "v.value_date <= p.end_value_date" in normalized
    assert "v.pipeline_first_seen_at_utc <= p.as_of_utc" in normalized
    assert "limit 20001" in normalized
    assert "value_type = 'price'" in normalized
    assert set(re.findall(r"'([0-9]{9})'", sql)) == {
        "115688058",
        "165444048",
        "165349556",
        "165442712",
    }
    for forbidden_curve in ("115689883", "165442711", "165444047", "110181967"):
        assert forbidden_curve not in sql
    assert not re.search(
        r"\b(insert|update|delete|merge|create|replace|alter|drop|truncate)\b", normalized
    )

    assert verify_lseg_latest_sql_binding() == LSEG_LATEST_SQL_SHA256
    latest_sql = LSEG_LATEST_SQL_PATH.read_text(encoding="utf-8")
    latest_normalized = re.sub(r"\s+", " ", latest_sql.lower())
    assert "prd.silver.ge_market_lseg_curve_values" in latest_normalized
    assert "ge_market_lseg_curve_value_vintages" not in latest_normalized
    assert "v._silver_updated_ts <= p.assessed_at_utc" in latest_normalized
    assert "limit 20001" in latest_normalized
    assert not re.search(
        r"\b(insert|update|delete|merge|create|replace|alter|drop|truncate)\b",
        latest_normalized,
    )


def test_latest_parameter_builder_accepts_one_market_month_across_two_utc_months() -> None:
    assert build_lseg_epex_actuals_pit_parameters(
        window_start_utc=START,
        window_end_utc=END,
        as_of_utc=AS_OF,
    ) == {
        "start_utc": START,
        "end_utc": END,
        "as_of_utc": AS_OF,
        "start_value_date": "2026-01-01",
        "end_value_date": "2026-01-01",
    }
    with pytest.raises(SpotSourceReconciliationError, match="one UTC calendar month"):
        build_lseg_epex_actuals_pit_parameters(
            window_start_utc="2026-06-30T22:00:00Z",
            window_end_utc="2026-07-31T22:00:00Z",
            as_of_utc="2026-09-04T08:33:46.008Z",
        )
    assert build_lseg_epex_actuals_latest_parameters(
        window_start_utc="2026-06-30T22:00:00Z",
        window_end_utc="2026-07-31T22:00:00Z",
        assessed_at_utc="2026-09-04T08:33:46.008Z",
    ) == {
        "start_utc": "2026-06-30T22:00:00Z",
        "end_utc": "2026-07-31T22:00:00Z",
        "assessed_at_utc": "2026-09-04T08:33:46.008000Z",
        "start_value_date": "2026-06-30",
        "end_value_date": "2026-07-31",
    }
    with pytest.raises(SpotSourceReconciliationError, match="precedes"):
        build_lseg_epex_actuals_pit_parameters(
            window_start_utc=START,
            window_end_utc=END,
            as_of_utc="2026-01-01T01:00:00Z",
        )


def test_clean_lseg_extract_passes_without_model_authority() -> None:
    artifact = _lseg_artifact()

    assert artifact.audit["row_count"] == 26
    assert artifact.audit["curve_binding"]["CH"] == {
        "curve_id": "115688058",
        "resolution": "PT1H",
    }
    assert artifact.audit["authorities"] == {
        "point_in_time_filter_validated": True,
        "cadence_completeness_authorized": False,
        "model_input_authorized": False,
        "model_selection_authorized": False,
        "production_authorized": False,
    }


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("curve_id", "110181967", "four-price-curve binding"),
        ("resolution", "PT1H", "resolution differs"),
        ("price_eur_per_mwh", float("nan"), "non-finite price"),
        ("pipeline_first_seen_at_utc", "2026-01-03T00:00:00Z", "leaks"),
    ],
)
def test_lseg_extract_rejects_wrong_semantics_or_leakage(
    column: str, value: object, match: str
) -> None:
    frame = _lseg_frame()
    row = frame.index[frame["market_zone"].eq("AT")][0]
    frame.loc[row, column] = value

    with pytest.raises(SpotSourceReconciliationError, match=match):
        _lseg_artifact(frame)


def test_lseg_extract_rejects_missing_zone_duplicate_grain_and_bad_hash() -> None:
    frame = _lseg_frame()
    with pytest.raises(SpotSourceReconciliationError, match="all four"):
        _lseg_artifact(frame.loc[~frame["market_zone"].eq("FR")].reset_index(drop=True))

    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(SpotSourceReconciliationError, match="vintage ID is duplicated"):
        _lseg_artifact(duplicated)

    with pytest.raises(SpotSourceReconciliationError, match="SHA-256 differs"):
        validate_lseg_epex_actuals_pit_extract(
            frame,
            window_start_utc=START,
            window_end_utc=END,
            as_of_utc=AS_OF,
            pit_query_sha256="a" * 64,
        )


def test_lseg_validator_defensively_copies_the_input() -> None:
    frame = _lseg_frame()
    artifact = _lseg_artifact(frame)
    frame.loc[0, "price_eur_per_mwh"] = 999.0

    assert artifact.frame.loc[0, "price_eur_per_mwh"] != 999.0


def test_exact_cross_source_reconciliation_passes_but_grants_no_model_authority() -> None:
    report = reconcile_lseg_entsoe_spot(
        entsoe=_entsoe_artifact(), lseg=_lseg_artifact(), policy=_policy()
    )

    assert report.status == PASS_STATUS
    assert report.findings == ()
    payload = report.as_dict()
    assert payload["metrics"]["per_zone"]["CH"]["matched_hours"] == 2
    assert payload["metrics"]["per_zone"]["FR"]["p95_abs_difference_eur_per_mwh"] == 0.0
    assert payload["metrics"]["it_nord"] == {
        "expected_hours": 2,
        "entsoe_complete_hours": 2,
        "lseg_crosscheck_available": False,
        "status": "ENTSOE_ONLY_NO_ACTIVE_LSEG_EPEX_CURVE",
    }
    assert payload["authorities"]["source_reconciliation_validated"] is True
    assert payload["authorities"]["model_input_authorized"] is False
    assert payload["authorities"]["monthly_level_authorized"] is False
    assert payload["layer_policy"]["mismatch_action"] == "BLOCK_NO_SILENT_SOURCE_SUBSTITUTION"


def test_latest_candidate_reconciliation_passes_without_promoting_the_candidate() -> None:
    raw = _candidate_raw_frame()
    candidate = _candidate_artifact(raw)
    report = reconcile_lseg_entsoe_latest_candidate(
        entsoe_raw_frame=raw,
        entsoe=candidate,
        lseg=_lseg_latest_artifact(),
        policy=_policy(),
    )

    assert report.status == PASS_STATUS
    assert not candidate.frame["is_final"].any()
    assert candidate.audit["authorities"]["consumer_contract_authorized"] is False
    assert report.as_dict()["authorities"]["model_input_authorized"] is False

    changed = raw.copy()
    changed.loc[0, "price_eur_per_mwh"] += 1.0
    with pytest.raises(SpotSourceReconciliationError, match="candidate audit binding"):
        reconcile_lseg_entsoe_latest_candidate(
            entsoe_raw_frame=changed,
            entsoe=candidate,
            lseg=_lseg_latest_artifact(),
            policy=_policy(),
        )


def test_normalized_entsoe_blocks_expand_before_hourly_reconciliation() -> None:
    wide = _entsoe_frame().groupby("field_name", as_index=False, sort=False).first()
    wide["interval_start_utc"] = pd.Timestamp(START)
    wide["interval_end_utc"] = pd.Timestamp(END)
    wide["price_eur_per_mwh"] = wide["field_name"].map(
        lambda field: ZONE_BASE[FIELD_TO_ZONE[field]]
    )
    wide = wide.loc[:, PIT_COLUMNS]
    lseg = _lseg_frame()
    lseg["price_eur_per_mwh"] = lseg["market_zone"].map(ZONE_BASE)

    report = reconcile_lseg_entsoe_spot(
        entsoe=_entsoe_artifact(wide),
        lseg=_lseg_artifact(lseg),
        policy=_policy(),
    )

    assert report.status == PASS_STATUS
    assert report.as_dict()["schema_version"] == "fmv_lseg_entsoe_spot_reconciliation.v2"
    assert report.metrics["per_zone"]["AT"]["matched_hours"] == 2


def test_quarter_hours_are_duration_weighted_down_to_hour_without_upsampling() -> None:
    entsoe_frame = _entsoe_frame()
    ch_first_hour = entsoe_frame["field_name"].eq("ch_price") & entsoe_frame[
        "interval_start_utc"
    ].lt(pd.Timestamp("2026-01-01T01:00:00Z"))
    entsoe_frame.loc[ch_first_hour, "price_eur_per_mwh"] = [10.0, 20.0, 30.0, 40.0]
    lseg_frame = _lseg_frame()
    ch_first = lseg_frame["market_zone"].eq("CH") & lseg_frame["interval_start_utc"].eq(
        pd.Timestamp(START)
    )
    lseg_frame.loc[ch_first, "price_eur_per_mwh"] = 25.0

    report = reconcile_lseg_entsoe_spot(
        entsoe=_entsoe_artifact(entsoe_frame),
        lseg=_lseg_artifact(lseg_frame),
        policy=_policy(),
    )

    assert report.status == PASS_STATUS
    assert report.metrics["per_zone"]["CH"]["maximum_abs_difference_eur_per_mwh"] == 0.0


def test_statistical_threshold_breaches_block_without_selecting_a_replacement_source() -> None:
    lseg_frame = _lseg_frame()
    lseg_frame.loc[lseg_frame["market_zone"].eq("FR"), "price_eur_per_mwh"] += 4.0
    report = reconcile_lseg_entsoe_spot(
        entsoe=_entsoe_artifact(),
        lseg=_lseg_artifact(lseg_frame),
        policy=_policy(
            maximum_p95_abs_difference_eur_per_mwh=1.0,
            maximum_absolute_bias_eur_per_mwh=1.0,
            maximum_single_hour_abs_difference_eur_per_mwh=2.0,
        ),
    )

    assert report.status == BLOCKED_STATUS
    assert {
        "CROSS_SOURCE_P95_ABS_DIFFERENCE_ABOVE_POLICY",
        "CROSS_SOURCE_SINGLE_HOUR_ABS_DIFFERENCE_ABOVE_POLICY",
        "CROSS_SOURCE_ABSOLUTE_BIAS_ABOVE_POLICY",
    }.issubset(_codes(report))
    assert report.as_dict()["layer_policy"]["mismatch_action"] == (
        "BLOCK_NO_SILENT_SOURCE_SUBSTITUTION"
    )


def test_missing_quarter_hour_blocks_coverage_and_is_not_silently_averaged() -> None:
    frame = _lseg_frame()
    remove = frame["market_zone"].eq("AT") & frame["interval_start_utc"].eq(
        pd.Timestamp("2026-01-01T00:15:00Z")
    )
    report = reconcile_lseg_entsoe_spot(
        entsoe=_entsoe_artifact(),
        lseg=_lseg_artifact(frame.loc[~remove].reset_index(drop=True)),
        policy=_policy(),
    )

    assert report.status == BLOCKED_STATUS
    assert "LSEG_NATIVE_INTERVALS_DO_NOT_FORM_COMPLETE_UTC_HOURS" in _codes(report)
    assert report.metrics["per_zone"]["AT"]["matched_hours"] == 1
    assert report.metrics["per_zone"]["AT"]["lseg_missing_hours"] == 1


def test_cross_hour_or_off_grid_native_intervals_block_reconciliation() -> None:
    frame = _lseg_frame()
    at_rows = frame.index[frame["market_zone"].eq("AT")]
    frame.loc[at_rows[1], "interval_start_utc"] = pd.Timestamp("2026-01-01T00:10:00Z")
    frame.loc[at_rows[1], "interval_end_utc"] = pd.Timestamp("2026-01-01T00:25:00Z")
    report = reconcile_lseg_entsoe_spot(
        entsoe=_entsoe_artifact(), lseg=_lseg_artifact(frame), policy=_policy()
    )

    assert report.status == BLOCKED_STATUS
    assert "LSEG_NATIVE_INTERVALS_DO_NOT_FORM_COMPLETE_UTC_HOURS" in _codes(report)


def test_it_nord_is_entsoe_only_and_incomplete_coverage_blocks() -> None:
    frame = _entsoe_frame()
    remove = frame["field_name"].eq("it_nord_price") & frame["interval_start_utc"].eq(
        pd.Timestamp("2026-01-01T00:15:00Z")
    )
    report = reconcile_lseg_entsoe_spot(
        entsoe=_entsoe_artifact(frame.loc[~remove].reset_index(drop=True)),
        lseg=_lseg_artifact(),
        policy=_policy(),
    )

    assert report.status == BLOCKED_STATUS
    assert "ENTSOE_IT_NORD_HOURLY_COVERAGE_INCOMPLETE" in _codes(report)
    assert report.metrics["it_nord"]["lseg_crosscheck_available"] is False


def test_negative_prices_are_valid_market_values() -> None:
    entsoe_frame = _entsoe_frame()
    lseg_frame = _lseg_frame()
    entsoe_frame["price_eur_per_mwh"] -= 100.0
    lseg_frame["price_eur_per_mwh"] -= 100.0

    report = reconcile_lseg_entsoe_spot(
        entsoe=_entsoe_artifact(entsoe_frame),
        lseg=_lseg_artifact(lseg_frame),
        policy=_policy(),
    )
    assert report.status == PASS_STATUS


def test_reconciliation_rejects_mutated_artifact_and_mismatched_as_of() -> None:
    entsoe = _entsoe_artifact()
    entsoe.frame.loc[0, "price_eur_per_mwh"] += 1.0
    with pytest.raises(SpotSourceReconciliationError, match="audit binding"):
        reconcile_lseg_entsoe_spot(entsoe=entsoe, lseg=_lseg_artifact(), policy=_policy())

    lseg = _lseg_artifact()
    lseg.audit["window_end_utc"] = "2026-01-01T01:00:00Z"  # type: ignore[index]
    with pytest.raises(SpotSourceReconciliationError, match="audit binding"):
        reconcile_lseg_entsoe_spot(entsoe=_entsoe_artifact(), lseg=lseg, policy=_policy())

    with pytest.raises(SpotSourceReconciliationError, match="windows do not match"):
        reconcile_lseg_entsoe_spot(
            entsoe=_entsoe_artifact(),
            lseg=_lseg_artifact(as_of="2026-01-03T00:00:00Z"),
            policy=_policy(),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"minimum_matched_hours_per_zone": 0},
        {"minimum_hourly_overlap_ratio": 0.0},
        {"minimum_hourly_overlap_ratio": 1.01},
        {"maximum_p95_abs_difference_eur_per_mwh": -1.0},
        {
            "maximum_p95_abs_difference_eur_per_mwh": 2.0,
            "maximum_single_hour_abs_difference_eur_per_mwh": 1.0,
        },
    ],
)
def test_policy_rejects_unsafe_thresholds(overrides: dict[str, object]) -> None:
    with pytest.raises(SpotSourceReconciliationError):
        _policy(**overrides)


def test_policy_is_explicit_and_hash_stable() -> None:
    policy = _policy()
    assert re.fullmatch(r"[0-9a-f]{64}", policy.semantic_sha256)
    assert policy.semantic_sha256 == _policy().semantic_sha256
    assert policy.semantic_sha256 != _policy(maximum_absolute_bias_eur_per_mwh=0.1).semantic_sha256


def test_utc_hour_grid_does_not_assume_a_24_hour_local_dst_day() -> None:
    start = "2026-03-29T00:00:00Z"
    end = "2026-03-29T03:00:00Z"
    as_of = "2026-03-30T00:00:00Z"
    policy = _policy(minimum_matched_hours_per_zone=3)
    report = reconcile_lseg_entsoe_spot(
        entsoe=_entsoe_artifact(
            _entsoe_frame(start=start, end=end), start=start, end=end, as_of=as_of
        ),
        lseg=_lseg_artifact(_lseg_frame(start=start, end=end), start=start, end=end, as_of=as_of),
        policy=policy,
    )

    assert report.status == PASS_STATUS
    assert report.metrics["expected_utc_hours_per_zone"] == 3


def test_report_contains_aggregate_metrics_not_raw_aligned_prices() -> None:
    payload = reconcile_lseg_entsoe_spot(
        entsoe=_entsoe_artifact(), lseg=_lseg_artifact(), policy=_policy()
    ).as_dict()
    serialized = str(payload)

    assert "entsoe_price" not in serialized
    assert "lseg_price" not in serialized
    assert "hour_start_utc" not in serialized
