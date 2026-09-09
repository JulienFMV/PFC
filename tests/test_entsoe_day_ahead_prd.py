from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from pfc_shaping.validation.entsoe_day_ahead_prd import (
    BLOCKED_STATUS,
    EXPECTED_FIELDS,
    EXPECTED_SERIES_SLOTS,
    PASS_STATUS,
    PIT_COLUMNS,
    PIT_RESULT_ROW_LIMIT,
    PIT_SQL_PATH,
    PIT_SQL_SHA256,
    PROFILE_COLUMNS,
    PROFILE_SQL_PATH,
    PROFILE_SQL_SHA256,
    REQUIRED_CHANGE_COMMIT,
    SERIES_INVENTORY_SPEC,
    EntsoeDayAheadPrdError,
    assess_day_ahead_prd_profile,
    build_day_ahead_pit_parameters,
    build_day_ahead_profile_parameters,
    validate_day_ahead_pit_extract,
    verify_sql_bindings,
)

WINDOW_START = "2026-01-01T00:00:00Z"
WINDOW_END = "2026-02-01T00:00:00Z"
ASSESSED_AT = "2026-02-02T00:00:00Z"


def _series_key(field: str, classification: str | None) -> str:
    base = f"day_ahead_prices||{field}"
    return f"{base}||{classification}" if classification is not None else base


def _inventory_binding() -> dict[str, str]:
    return {
        slot: _series_key(field, classification)
        for slot, field, classification in SERIES_INVENTORY_SPEC
    }


def _selection(*, at_sequence: str = "1", de_lu_sequence: str = "1") -> dict[str, str]:
    return {
        "ch_price": "day_ahead_prices||ch_price",
        "at_price": f"day_ahead_prices||at_price||{at_sequence}",
        "de_lu_price": f"day_ahead_prices||de_lu_price||{de_lu_sequence}",
        "fr_price": "day_ahead_prices||fr_price",
        "it_nord_price": "day_ahead_prices||it_nord_price",
    }


def _profile() -> pd.DataFrame:
    binding = _inventory_binding()
    rows = []
    for series_id, (slot, field, classification) in enumerate(SERIES_INVENTORY_SPEC, start=1):
        rows.append(
            {
                "field_name": field,
                "series_key": binding[slot],
                "classification_sequence": classification,
                "unit": "EUR/MWh",
                "document_type": "A44",
                "series_id": series_id,
                "resolution": "PT60M",
                "vintage_row_count": 744,
                "distinct_interval_count": 744,
                "min_interval_start_utc": WINDOW_START,
                "max_interval_end_utc": WINDOW_END,
                "null_value_count": 0,
                "dq_failed_count": 0,
                "unknown_availability_count": 0,
                "invalid_availability_order_count": 0,
                "invalid_interval_count": 0,
                "canonical_series_key_mismatch_count": 0,
                "profile_row_count": len(EXPECTED_SERIES_SLOTS),
                "duplicate_vintage_key_count": 0,
                "orphan_series_key_count": 0,
                "gold_series_key_duplicate_count": 0,
                "latest_grain_duplicate_count": 0,
                "legacy_new_overlap_interval_count": 0,
            }
        )
    return pd.DataFrame(rows, columns=PROFILE_COLUMNS)


def _rebuild_evidence(**overrides: object) -> dict[str, object]:
    evidence: dict[str, object] = {
        "environment": "prd",
        "run_id": "entsoe-full-20260824",
        "manifest_sha256": "a" * 64,
        "deployed_commit": "b" * 40,
        "required_change_commit": REQUIRED_CHANGE_COMMIT,
        "required_change_ancestor_proven": True,
        "mode": "full",
        "groups_rebuilt": [
            "day_ahead_prices",
            "production_unit_unavailability",
            "generation_unit_unavailability",
            "transmission_unavailability",
            "installed_capacity_per_unit",
            "generation_forecast",
        ],
        "run_status": "SUCCESS",
        "post_backfill_validation_status": "PASS",
        "old_new_coexistence_count": 0,
        "completed_at_utc": "2026-01-31T23:00:00Z",
    }
    evidence.update(overrides)
    return evidence


def _assess(
    profile: pd.DataFrame | None = None,
    *,
    binding: dict[str, str] | None = None,
    evidence: dict[str, object] | None = None,
):
    return assess_day_ahead_prd_profile(
        _profile() if profile is None else profile,
        window_start_utc=WINDOW_START,
        window_end_utc=WINDOW_END,
        assessed_at_utc=ASSESSED_AT,
        profile_query_sha256=PROFILE_SQL_SHA256,
        series_inventory_binding=_inventory_binding() if binding is None else binding,
        rebuild_evidence=_rebuild_evidence() if evidence is None else evidence,
    )


def _codes(report) -> set[str]:
    return {finding.code for finding in report.findings}


def _pit_frame() -> pd.DataFrame:
    rows = []
    selection = _selection()
    for revision, field in enumerate(EXPECTED_FIELDS, start=1):
        rows.append(
            {
                "field_name": field,
                "series_key": selection[field],
                "interval_start_utc": "2026-01-01T00:00:00Z",
                "interval_end_utc": "2026-01-01T01:00:00Z",
                "resolution": "PT60M",
                "price_eur_per_mwh": float(50 + revision),
                "availability_timestamp_utc": "2025-12-31T12:00:00Z",
                "source_document_mrid": f"DOC-{revision}",
                "source_document_revision_number": revision,
            }
        )
    return pd.DataFrame(rows, columns=PIT_COLUMNS)


def test_clean_profile_passes_but_does_not_grant_model_authority() -> None:
    report = _assess()

    assert report.status == PASS_STATUS
    assert report.findings == ()
    payload = report.as_dict()
    assert payload["authorities"] == {
        "bounded_pit_extraction_authorized": True,
        "cadence_completeness_authorized": False,
        "model_input_authorized": False,
        "model_selection_authorized": False,
        "production_authorized": False,
    }
    assert payload["metrics"]["series_count"] == 7
    assert payload["metrics"]["field_count"] == 5
    assert payload["layer_policy"]["multi_auction_inventory"] == (
        "AT_AND_DE_LU_CLASSIFICATION_SEQUENCE_1_AND_2"
    )
    assert payload["layer_policy"]["pit_selection_policy"] == (
        "EXPLICIT_ONE_SERIES_PER_FIELD_NO_DEFAULT_CHOICE"
    )
    assert payload["layer_policy"]["lseg_epex_actuals_role"] == (
        "REALIZED_SPOT_CROSSCHECK_CH_AT_DE_LU_FR"
    )
    assert payload["layer_policy"]["euler_spot_role"] == "INDEPENDENT_CROSSCHECK_ONLY"


def test_missing_binding_and_rebuild_manifest_fail_closed() -> None:
    report = assess_day_ahead_prd_profile(
        _profile(),
        window_start_utc=WINDOW_START,
        window_end_utc=WINDOW_END,
        assessed_at_utc=ASSESSED_AT,
        profile_query_sha256=PROFILE_SQL_SHA256,
    )

    assert report.status == BLOCKED_STATUS
    assert _codes(report) == {
        "DAY_AHEAD_EXACT_SERIES_INVENTORY_BINDING_MISSING",
        "ENTSOE_PRD_REBUILD_MANIFEST_MISSING",
    }


@pytest.mark.parametrize(
    ("column", "value", "code"),
    [
        ("null_value_count", 3, "DAY_AHEAD_VALUE_NULL"),
        ("dq_failed_count", 2, "DAY_AHEAD_DQ_FAILED"),
        ("unknown_availability_count", 4, "DAY_AHEAD_AVAILABILITY_UNKNOWN"),
        ("invalid_interval_count", 1, "DAY_AHEAD_INTERVAL_INVALID"),
        (
            "canonical_series_key_mismatch_count",
            1,
            "DAY_AHEAD_SERIES_KEY_NONCANONICAL",
        ),
    ],
)
def test_each_silver_quality_counter_is_blocking(column: str, value: int, code: str) -> None:
    profile = _profile()
    profile.loc[0, column] = value

    report = _assess(profile)

    assert report.status == BLOCKED_STATUS
    assert code in _codes(report)


@pytest.mark.parametrize(
    ("column", "value", "code"),
    [
        (
            "duplicate_vintage_key_count",
            1,
            "DAY_AHEAD_VINTAGE_KEY_DUPLICATED",
        ),
        ("orphan_series_key_count", 1, "DAY_AHEAD_SERIES_KEY_ORPHANED"),
        (
            "gold_series_key_duplicate_count",
            1,
            "DAY_AHEAD_GOLD_SERIES_KEY_DUPLICATED",
        ),
        ("latest_grain_duplicate_count", 1, "DAY_AHEAD_LATEST_GRAIN_DUPLICATED"),
        (
            "legacy_new_overlap_interval_count",
            1,
            "DAY_AHEAD_LEGACY_NEW_KEY_OVERLAP",
        ),
    ],
)
def test_each_cross_layer_quality_counter_is_blocking(column: str, value: int, code: str) -> None:
    profile = _profile()
    profile[column] = value

    report = _assess(profile)

    assert report.status == BLOCKED_STATUS
    assert code in _codes(report)


def test_gold_semantics_and_empty_source_series_are_blocking() -> None:
    profile = _profile()
    profile.loc[0, "unit"] = "EUR"
    profile.loc[1, "document_type"] = "A00"
    profile.loc[2, "vintage_row_count"] = 0
    profile.loc[2, "distinct_interval_count"] = 0
    profile.loc[2, "min_interval_start_utc"] = None
    profile.loc[2, "max_interval_end_utc"] = None

    report = _assess(profile)

    assert {
        "DAY_AHEAD_UNIT_INVALID",
        "DAY_AHEAD_DOCUMENT_TYPE_INVALID",
        "DAY_AHEAD_SERIES_EMPTY",
    }.issubset(_codes(report))


def test_missing_field_and_noncanonical_gold_key_are_blocking() -> None:
    profile = _profile().iloc[:-1].copy()
    profile["profile_row_count"] = len(profile)
    profile.loc[0, "series_key"] = "day_ahead_prices||wrong"

    report = _assess(profile)

    assert "DAY_AHEAD_FIELD_MISSING" in _codes(report)
    assert "DAY_AHEAD_SERIES_IDENTITY_MISSING" in _codes(report)
    assert "DAY_AHEAD_GOLD_SERIES_KEY_NONCANONICAL" in _codes(report)
    assert "DAY_AHEAD_BOUND_SERIES_ABSENT" in _codes(report)


def test_rebuild_claims_must_all_hold() -> None:
    evidence = _rebuild_evidence(
        environment="dev",
        mode="incremental",
        run_status="FAILED",
        post_backfill_validation_status="FAIL",
        required_change_ancestor_proven=False,
        groups_rebuilt=["day_ahead_prices"],
        old_new_coexistence_count=7,
    )

    report = _assess(evidence=evidence)

    assert {
        "ENTSOE_REBUILD_ENVIRONMENT_INVALID",
        "ENTSOE_REBUILD_MODE_INVALID",
        "ENTSOE_REBUILD_RUN_FAILED",
        "ENTSOE_REBUILD_VALIDATION_FAILED",
        "ENTSOE_REQUIRED_CHANGE_ANCESTRY_UNPROVEN",
        "ENTSOE_REBUILD_GROUPS_INCOMPLETE",
        "ENTSOE_REBUILD_MANIFEST_REPORTS_KEY_COEXISTENCE",
    } == _codes(report)


def test_profile_schema_query_hash_and_count_types_are_strict() -> None:
    with pytest.raises(EntsoeDayAheadPrdError, match="columns are not exact"):
        _assess(_profile().drop(columns="series_id"))
    with pytest.raises(EntsoeDayAheadPrdError, match="SHA-256 differs"):
        assess_day_ahead_prd_profile(
            _profile(),
            window_start_utc=WINDOW_START,
            window_end_utc=WINDOW_END,
            assessed_at_utc=ASSESSED_AT,
            profile_query_sha256="0" * 64,
        )
    profile = _profile()
    profile["vintage_row_count"] = profile["vintage_row_count"].astype(object)
    profile.loc[0, "vintage_row_count"] = True
    with pytest.raises(EntsoeDayAheadPrdError, match="nonnegative integer"):
        _assess(profile)


def test_profile_accepts_both_real_multi_auction_sequences() -> None:
    report = _assess()

    assert report.status == PASS_STATUS
    identities = set(
        zip(_profile()["field_name"], _profile()["classification_sequence"], strict=True)
    )
    assert ("at_price", "1") in identities
    assert ("at_price", "2") in identities
    assert ("de_lu_price", "1") in identities
    assert ("de_lu_price", "2") in identities


def test_profile_requires_both_sequences_even_when_all_five_fields_exist() -> None:
    profile = _profile()
    profile = profile.loc[
        ~(profile["field_name"].eq("at_price") & profile["classification_sequence"].eq("2"))
    ].reset_index(drop=True)
    profile["profile_row_count"] = len(profile)

    report = _assess(profile)

    assert report.status == BLOCKED_STATUS
    assert "DAY_AHEAD_FIELD_MISSING" not in _codes(report)
    assert "DAY_AHEAD_SERIES_IDENTITY_MISSING" in _codes(report)
    assert "DAY_AHEAD_BOUND_SERIES_ABSENT" in _codes(report)


def test_one_series_may_have_effective_dated_resolution_rows() -> None:
    profile = _profile()
    source_index = profile.index[profile["series_key"].eq("day_ahead_prices||at_price||1")][0]
    profile.loc[source_index, "max_interval_end_utc"] = "2026-01-16T00:00:00Z"
    second_regime = profile.loc[[source_index]].copy()
    second_regime["resolution"] = "PT15M"
    second_regime["min_interval_start_utc"] = "2026-01-16T00:00:00Z"
    second_regime["max_interval_end_utc"] = WINDOW_END
    second_regime["vintage_row_count"] = 1_536
    second_regime["distinct_interval_count"] = 1_536
    profile = pd.concat([profile, second_regime], ignore_index=True)
    profile["profile_row_count"] = len(profile)

    report = _assess(profile)

    assert report.status == PASS_STATUS
    assert report.metrics["series_count"] == 7
    assert report.metrics["profile_rows"] == 8
    assert report.metrics["series_profiles"]["at_price_seq1"]["resolutions"] == [
        "PT15M",
        "PT60M",
    ]
    assert (
        report.metrics["series_profiles"]["at_price_seq1"]["min_interval_start_utc"] == WINDOW_START
    )
    assert report.metrics["series_profiles"]["at_price_seq1"]["max_interval_end_utc"] == WINDOW_END


def test_profile_rejects_an_unexpected_multi_auction_sequence() -> None:
    profile = _profile()
    extra = profile.loc[profile["field_name"].eq("at_price")].iloc[[0]].copy()
    extra["series_key"] = "day_ahead_prices||at_price||3"
    extra["classification_sequence"] = "3"
    extra["series_id"] = 99
    profile = pd.concat([profile, extra], ignore_index=True)
    profile["profile_row_count"] = len(profile)

    report = _assess(profile)

    assert report.status == BLOCKED_STATUS
    assert "DAY_AHEAD_SERIES_IDENTITY_UNEXPECTED" in _codes(report)


def test_pit_parameters_are_exact_and_partition_pruned() -> None:
    parameters = build_day_ahead_pit_parameters(
        series_selection=_selection(),
        window_start_utc=WINDOW_START,
        window_end_utc=WINDOW_END,
        as_of_utc=ASSESSED_AT,
    )

    assert parameters == {
        "start_utc": WINDOW_START,
        "end_utc": WINDOW_END,
        "as_of_utc": ASSESSED_AT,
        "delivery_year": 2026,
        "delivery_month": 1,
        "ch_series_key": "day_ahead_prices||ch_price",
        "at_series_key": "day_ahead_prices||at_price||1",
        "de_lu_series_key": "day_ahead_prices||de_lu_price||1",
        "fr_series_key": "day_ahead_prices||fr_price",
        "it_nord_series_key": "day_ahead_prices||it_nord_price",
    }


def test_profile_parameters_are_exact_and_partition_pruned() -> None:
    parameters = build_day_ahead_profile_parameters(
        window_start_utc=WINDOW_START,
        window_end_utc=WINDOW_END,
    )

    assert parameters == {
        "start_utc": WINDOW_START,
        "end_utc": WINDOW_END,
        "delivery_year": 2026,
        "delivery_month": 1,
    }


def test_pit_selection_accepts_sequence_two_only_when_explicit() -> None:
    parameters = build_day_ahead_pit_parameters(
        series_selection=_selection(at_sequence="2", de_lu_sequence="2"),
        window_start_utc=WINDOW_START,
        window_end_utc=WINDOW_END,
        as_of_utc=ASSESSED_AT,
    )

    assert parameters["at_series_key"] == "day_ahead_prices||at_price||2"
    assert parameters["de_lu_series_key"] == "day_ahead_prices||de_lu_price||2"


@pytest.mark.parametrize(
    "selection",
    [
        {**_selection(), "at_price": "day_ahead_prices||at_price"},
        {**_selection(), "de_lu_price": "day_ahead_prices||de_lu_price||3"},
    ],
)
def test_pit_selection_rejects_base_or_unknown_multi_auction_keys(
    selection: dict[str, str],
) -> None:
    with pytest.raises(EntsoeDayAheadPrdError, match="not an admitted candidate"):
        build_day_ahead_pit_parameters(
            series_selection=selection,
            window_start_utc=WINDOW_START,
            window_end_utc=WINDOW_END,
            as_of_utc=ASSESSED_AT,
        )


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        ("2026-01-01T00:01:00Z", WINDOW_END, "15-minute"),
        (WINDOW_START, "2026-02-02T00:00:00Z", "31 days"),
        (WINDOW_END, WINDOW_START, "empty or inverted"),
    ],
)
def test_pit_parameters_reject_unsafe_windows(start: str, end: str, message: str) -> None:
    with pytest.raises(EntsoeDayAheadPrdError, match=message):
        build_day_ahead_pit_parameters(
            series_selection=_selection(),
            window_start_utc=start,
            window_end_utc=end,
            as_of_utc=ASSESSED_AT,
        )


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        ("2026-01-01T00:01:00Z", WINDOW_END, "15-minute"),
        (WINDOW_START, "2026-02-02T00:00:00Z", "31 days"),
        ("2026-01-15T00:00:00Z", "2026-02-01T00:15:00Z", "calendar month"),
        (WINDOW_END, WINDOW_START, "empty or inverted"),
    ],
)
def test_profile_parameters_reject_unsafe_windows(start: str, end: str, message: str) -> None:
    with pytest.raises(EntsoeDayAheadPrdError, match=message):
        build_day_ahead_profile_parameters(
            window_start_utc=start,
            window_end_utc=end,
        )


def test_clean_pit_extract_is_sorted_hashed_and_non_authoritative() -> None:
    source = _pit_frame().iloc[::-1].reset_index(drop=True)

    result = validate_day_ahead_pit_extract(
        source,
        series_selection=_selection(),
        window_start_utc=WINDOW_START,
        window_end_utc=WINDOW_END,
        as_of_utc=ASSESSED_AT,
        pit_query_sha256=PIT_SQL_SHA256,
    )

    assert tuple(result.frame.columns) == PIT_COLUMNS
    assert result.audit["row_count"] == 5
    assert result.audit["schema_version"] == "fmv_entsoe_day_ahead_prd_pit_extract.v2"
    assert re.fullmatch(r"[0-9a-f]{64}", result.audit["frame_semantic_sha256"])
    assert result.audit["authorities"]["point_in_time_filter_validated"] is True
    assert result.audit["authorities"]["model_input_authorized"] is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda frame: frame.assign(availability_timestamp_utc="2026-02-03T00:00:00Z"),
            "leaks",
        ),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "duplicated",
        ),
        (lambda frame: frame.assign(price_eur_per_mwh=float("nan")), "non-finite"),
        (lambda frame: frame.assign(resolution="PT5M"), "resolution is unsupported"),
    ],
)
def test_pit_extract_rejects_leakage_duplicates_and_invalid_values(mutator, message: str) -> None:
    with pytest.raises(EntsoeDayAheadPrdError, match=message):
        validate_day_ahead_pit_extract(
            mutator(_pit_frame()),
            series_selection=_selection(),
            window_start_utc=WINDOW_START,
            window_end_utc=WINDOW_END,
            as_of_utc=ASSESSED_AT,
            pit_query_sha256=PIT_SQL_SHA256,
        )


def test_pit_extract_accepts_a_normalized_multi_cadence_block() -> None:
    source = _pit_frame()
    source.loc[source["field_name"].eq("ch_price"), "interval_end_utc"] = "2026-01-01T04:00:00Z"

    result = validate_day_ahead_pit_extract(
        source,
        series_selection=_selection(),
        window_start_utc=WINDOW_START,
        window_end_utc=WINDOW_END,
        as_of_utc=ASSESSED_AT,
        pit_query_sha256=PIT_SQL_SHA256,
    )

    assert result.frame.loc[
        result.frame["field_name"].eq("ch_price"), "interval_end_utc"
    ].item() == pd.Timestamp("2026-01-01T04:00:00Z")
    assert result.audit["interval_policy"]["duration"] == (
        "positive_integer_multiple_of_native_resolution"
    )


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        ("2026-01-01T00:00:00Z", "2026-01-01T01:30:00Z", "not a native-resolution multiple"),
        ("2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "empty or inverted"),
        ("2026-01-01T01:00:00Z", "2026-01-01T00:00:00Z", "empty or inverted"),
        ("2026-01-01T00:15:00Z", "2026-01-01T01:15:00Z", "off its native resolution grid"),
    ],
)
def test_pit_extract_rejects_invalid_normalized_bounds(start: str, end: str, message: str) -> None:
    source = _pit_frame()
    mask = source["field_name"].eq("ch_price")
    source.loc[mask, "interval_start_utc"] = start
    source.loc[mask, "interval_end_utc"] = end

    with pytest.raises(EntsoeDayAheadPrdError, match=message):
        validate_day_ahead_pit_extract(
            source,
            series_selection=_selection(),
            window_start_utc=WINDOW_START,
            window_end_utc=WINDOW_END,
            as_of_utc=ASSESSED_AT,
            pit_query_sha256=PIT_SQL_SHA256,
        )


def test_pit_extract_rejects_overlapping_normalized_blocks() -> None:
    source = _pit_frame()
    ch = source.loc[source["field_name"].eq("ch_price")].copy()
    source.loc[source["field_name"].eq("ch_price"), "resolution"] = "PT30M"
    ch["resolution"] = "PT30M"
    ch["interval_start_utc"] = "2026-01-01T00:30:00Z"
    ch["interval_end_utc"] = "2026-01-01T01:30:00Z"
    ch["source_document_revision_number"] = 99
    source = pd.concat([source, ch], ignore_index=True)

    with pytest.raises(EntsoeDayAheadPrdError, match="intervals overlap"):
        validate_day_ahead_pit_extract(
            source,
            series_selection=_selection(),
            window_start_utc=WINDOW_START,
            window_end_utc=WINDOW_END,
            as_of_utc=ASSESSED_AT,
            pit_query_sha256=PIT_SQL_SHA256,
        )


def test_pit_extract_rejects_a_block_ending_beyond_the_bounded_window() -> None:
    source = _pit_frame()
    mask = source["field_name"].eq("ch_price")
    source.loc[mask, "interval_start_utc"] = "2026-01-31T23:00:00Z"
    source.loc[mask, "interval_end_utc"] = "2026-02-01T01:00:00Z"

    with pytest.raises(EntsoeDayAheadPrdError, match="outside the delivery window"):
        validate_day_ahead_pit_extract(
            source,
            series_selection=_selection(),
            window_start_utc=WINDOW_START,
            window_end_utc=WINDOW_END,
            as_of_utc=ASSESSED_AT,
            pit_query_sha256=PIT_SQL_SHA256,
        )


def test_pit_extract_rejects_the_result_limit_sentinel() -> None:
    source = pd.concat([_pit_frame().iloc[[0]]] * PIT_RESULT_ROW_LIMIT, ignore_index=True)

    with pytest.raises(EntsoeDayAheadPrdError, match="rejection sentinel"):
        validate_day_ahead_pit_extract(
            source,
            series_selection=_selection(),
            window_start_utc=WINDOW_START,
            window_end_utc=WINDOW_END,
            as_of_utc=ASSESSED_AT,
            pit_query_sha256=PIT_SQL_SHA256,
        )


def test_sql_templates_are_hash_bound_read_only_and_cost_fenced() -> None:
    assert verify_sql_bindings() == {
        "profile_sql_sha256": PROFILE_SQL_SHA256,
        "pit_sql_sha256": PIT_SQL_SHA256,
    }
    profile_sql = Path(PROFILE_SQL_PATH).read_text(encoding="utf-8")
    pit_sql = Path(PIT_SQL_PATH).read_text(encoding="utf-8")
    combined = f"{profile_sql}\n{pit_sql}".lower()
    assert not re.search(
        r"\b(insert|update|delete|merge|create|alter|drop|truncate|optimize)\b", combined
    )
    assert "v._year = p.delivery_year" in profile_sql.lower()
    assert "v._month = p.delivery_month" in profile_sql.lower()
    assert "v._year = p.delivery_year" in pit_sql.lower()
    assert "v._month = p.delivery_month" in pit_sql.lower()
    assert "v.availability_timestamp_utc <= p.as_of_utc" in pit_sql.lower()
    assert "limit 101" in profile_sql.lower()
    assert "limit 20001" in pit_sql.lower()
    assert "price_eur_per_mwh" not in profile_sql.lower()
