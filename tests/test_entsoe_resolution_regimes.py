from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from pfc_shaping.validation.entsoe_resolution_regimes import (
    CONTRACT_CONTENT_ID,
    CONTRACT_STATUS,
    DIMENSION_COLUMNS,
    EVIDENCE_CLASS,
    GAP_STATUS,
    PASS_STATUS,
    REGIME_COLUMNS,
    TARGET_COLUMNS,
    EntsoeResolutionRegimeError,
    profile_resolution_regimes,
    validate_resolution_regime_contract,
    validate_resolution_regime_contract_path,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "ENTSOE-RESOLUTION-REGIME-CONTRACT-V2.json"
)

ASSESSMENT_START = pd.Timestamp("2026-01-01T00:00:00Z")
ASSESSMENT_END = pd.Timestamp("2026-01-03T00:00:00Z")
METADATA_AS_OF = pd.Timestamp("2026-02-01T00:00:00Z")


def _dimension() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "series_id": "price_ch_da",
                "group_name": "day_ahead_prices",
                "native_resolution": "PT15M",
            },
            {
                "series_id": "ntc_ch_fr",
                "group_name": "ntc_day_ahead",
                "native_resolution": "PT1H",
            },
        ],
        columns=DIMENSION_COLUMNS,
    )


def _targets() -> pd.DataFrame:
    price_hourly = pd.date_range(
        "2026-01-01T00:00:00Z",
        "2026-01-02T00:00:00Z",
        freq="1h",
        inclusive="left",
    )
    price_quarter_hourly = pd.date_range(
        "2026-01-02T00:00:00Z",
        "2026-01-03T00:00:00Z",
        freq="15min",
        inclusive="left",
    )
    ntc_hourly = pd.date_range(
        "2026-01-01T00:00:00Z",
        "2026-01-03T00:00:00Z",
        freq="1h",
        inclusive="left",
    )
    rows = [
        {"series_id": "price_ch_da", "target_ts_utc": timestamp}
        for timestamp in price_hourly.append(price_quarter_hourly)
    ] + [{"series_id": "ntc_ch_fr", "target_ts_utc": timestamp} for timestamp in ntc_hourly]
    return pd.DataFrame(rows, columns=TARGET_COLUMNS)


def _regime(
    series_id: str,
    start: str,
    end: str | None,
    resolution: str,
    *,
    document: str,
) -> dict[str, object]:
    return {
        "series_id": series_id,
        "valid_from_utc": pd.Timestamp(start),
        "valid_to_utc": pd.NaT if end is None else pd.Timestamp(end),
        "native_resolution": resolution,
        "source_document_id": document,
        "source_locator": f"https://example.test/{document}",
        "owner_confirmed_at_utc": pd.Timestamp("2026-02-01T00:00:00Z"),
    }


def _regimes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _regime(
                "price_ch_da",
                "2020-01-01T00:00:00Z",
                "2026-01-02T00:00:00Z",
                "PT1H",
                document="price-hourly-v1",
            ),
            _regime(
                "price_ch_da",
                "2026-01-02T00:00:00Z",
                None,
                "PT15M",
                document="price-quarter-hourly-v1",
            ),
            _regime(
                "ntc_ch_fr",
                "2020-01-01T00:00:00Z",
                None,
                "PT1H",
                document="ntc-hourly-v1",
            ),
        ],
        columns=REGIME_COLUMNS,
    )


def _profile(
    *,
    dimension: pd.DataFrame | None = None,
    targets: pd.DataFrame | None = None,
    regimes: pd.DataFrame | None = None,
    assessment_start: pd.Timestamp = ASSESSMENT_START,
    assessment_end: pd.Timestamp = ASSESSMENT_END,
    metadata_as_of: pd.Timestamp = METADATA_AS_OF,
):
    return profile_resolution_regimes(
        _dimension() if dimension is None else dimension,
        _targets() if targets is None else targets,
        _regimes() if regimes is None else regimes,
        evidence_class=EVIDENCE_CLASS,
        assessment_start_utc=assessment_start,
        assessment_end_utc=assessment_end,
        metadata_as_of_utc=metadata_as_of,
    )


def test_exact_contract_and_path_are_bound_without_authority() -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    direct = validate_resolution_regime_contract(document)
    bound = validate_resolution_regime_contract_path(CONTRACT_PATH)

    assert direct == bound
    assert direct["status"] == CONTRACT_STATUS
    assert direct["contract_content_id"] == CONTRACT_CONTENT_ID
    assert direct["databricks_statement_count"] == 0
    assert direct["model_input_authorized"] is False


def test_contract_rejects_global_calendar_go_live_authority() -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(document)
    mutated["quality_rules"]["roadmap_or_calendar_date_may_replace_series_evidence"] = True

    with pytest.raises(EntsoeResolutionRegimeError, match="roadmap_or_calendar"):
        validate_resolution_regime_contract(mutated)


def test_contract_cannot_drop_the_explicit_window_requirement() -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(document)
    mutated["quality_rules"]["explicit_assessment_window_required"] = False

    with pytest.raises(EntsoeResolutionRegimeError, match="explicit_assessment_window"):
        validate_resolution_regime_contract(mutated)


def test_piecewise_price_and_constant_ntc_cadences_are_both_complete() -> None:
    result = _profile()

    assert result.summary["status"] == PASS_STATUS
    assert result.summary["series_count"] == 2
    assert result.summary["regime_count"] == 3
    assert result.summary["transition_count"] == 1
    assert result.summary["observed_target_count"] == 168
    assert result.summary["expected_target_count"] == 168
    assert result.summary["missing_native_slot_count"] == 0
    assert result.summary["implicit_upsampling_performed"] is False

    price = next(row for row in result.series_profile if row["group_name"] == "day_ahead_prices")
    assert price["resolution_sequence"] == ["PT1H", "PT15M"]
    assert price["observed_target_count"] == 120
    assert price["missing_native_slot_count"] == 0


def test_historical_hourly_price_is_not_missing_quarter_hours() -> None:
    result = _profile()
    price = next(row for row in result.series_profile if row["group_name"] == "day_ahead_prices")

    assert price["expected_target_count"] == 24 + 96
    assert price["piecewise_grid_completeness_ratio"] == 1.0


def test_missing_post_transition_quarter_hour_is_reported_without_fill() -> None:
    targets = _targets()
    missing = pd.Timestamp("2026-01-02T12:15:00Z")
    targets = targets.loc[
        ~(targets["series_id"].eq("price_ch_da") & targets["target_ts_utc"].eq(missing))
    ].reset_index(drop=True)

    result = _profile(targets=targets)

    assert result.summary["status"] == GAP_STATUS
    assert result.summary["missing_native_slot_count"] == 1
    assert result.summary["observed_target_count"] == 167
    assert result.summary["expected_target_count"] == 168
    assert result.summary["implicit_forward_fill_performed"] is False


def test_missing_leading_and_trailing_slots_cannot_shrink_the_window() -> None:
    targets = _targets()
    removed = {
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-02T23:45:00Z"),
    }
    targets = targets.loc[
        ~(targets["series_id"].eq("price_ch_da") & targets["target_ts_utc"].isin(removed))
    ].reset_index(drop=True)

    result = _profile(targets=targets)
    price = next(row for row in result.series_profile if row["group_name"] == "day_ahead_prices")

    assert result.summary["status"] == GAP_STATUS
    assert price["expected_target_count"] == 120
    assert price["observed_target_count"] == 118
    assert price["missing_native_slot_count"] == 2


def test_series_with_no_observation_reports_the_full_window_missing() -> None:
    targets = _targets().loc[lambda frame: frame["series_id"].ne("ntc_ch_fr")]

    result = _profile(targets=targets)
    ntc = next(row for row in result.series_profile if row["group_name"] == "ntc_day_ahead")

    assert result.summary["status"] == GAP_STATUS
    assert ntc["observed_target_count"] == 0
    assert ntc["expected_target_count"] == 48
    assert ntc["missing_native_slot_count"] == 48


def test_observation_outside_explicit_window_is_rejected() -> None:
    targets = _targets()
    targets.loc[len(targets)] = {
        "series_id": "ntc_ch_fr",
        "target_ts_utc": pd.Timestamp("2025-12-31T23:00:00Z"),
    }

    with pytest.raises(EntsoeResolutionRegimeError, match="outside.*assessment window"):
        _profile(targets=targets)


def test_metadata_resolution_is_checked_at_explicit_as_of_not_last_observation() -> None:
    targets = _targets().loc[
        lambda frame: (
            ~(
                frame["series_id"].eq("price_ch_da")
                & frame["target_ts_utc"].ge("2026-01-02T00:00:00Z")
            )
        )
    ]

    result = _profile(targets=targets)
    price = next(row for row in result.series_profile if row["group_name"] == "day_ahead_prices")

    assert price["declared_current_resolution"] == "PT15M"
    assert price["missing_native_slot_count"] == 96


def test_metadata_as_of_is_an_evidence_instant_not_a_delivery_grid_point() -> None:
    result = _profile(
        metadata_as_of=pd.Timestamp("2026-02-01T00:07:00Z"),
    )

    assert result.summary["metadata_as_of_utc"] == "2026-02-01T00:07:00Z"


def test_future_owner_confirmation_cannot_support_the_metadata_as_of() -> None:
    regimes = _regimes()
    regimes.loc[0, "owner_confirmed_at_utc"] = pd.Timestamp("2026-02-01T00:00:01Z")

    with pytest.raises(EntsoeResolutionRegimeError, match="owner confirmation"):
        _profile(regimes=regimes)


def test_assessment_window_and_metadata_as_of_are_fail_closed() -> None:
    with pytest.raises(EntsoeResolutionRegimeError, match="timezone-aware UTC"):
        _profile(assessment_start=pd.Timestamp("2026-01-01T00:00:00"))
    with pytest.raises(EntsoeResolutionRegimeError, match="empty or reversed"):
        _profile(assessment_end=ASSESSMENT_START)
    with pytest.raises(EntsoeResolutionRegimeError, match="metadata as-of precedes"):
        _profile(metadata_as_of=ASSESSMENT_END - pd.Timedelta(seconds=1))


def test_fake_quarter_hour_before_transition_is_rejected() -> None:
    targets = _targets()
    targets.loc[len(targets)] = {
        "series_id": "price_ch_da",
        "target_ts_utc": pd.Timestamp("2026-01-01T00:15:00Z"),
    }

    with pytest.raises(EntsoeResolutionRegimeError, match="outside the active"):
        _profile(targets=targets)


@pytest.mark.parametrize(
    ("first_end", "second_start"),
    [
        ("2026-01-01T23:00:00Z", "2026-01-02T00:00:00Z"),
        ("2026-01-02T01:00:00Z", "2026-01-02T00:00:00Z"),
    ],
)
def test_regime_gaps_and_overlaps_are_rejected(
    first_end: str,
    second_start: str,
) -> None:
    regimes = _regimes()
    price = regimes["series_id"].eq("price_ch_da")
    indices = regimes.index[price].tolist()
    regimes.loc[indices[0], "valid_to_utc"] = pd.Timestamp(first_end)
    regimes.loc[indices[1], "valid_from_utc"] = pd.Timestamp(second_start)

    with pytest.raises(EntsoeResolutionRegimeError, match="gap or overlap"):
        _profile(regimes=regimes)


def test_transition_must_close_previous_hourly_interval() -> None:
    regimes = _regimes()
    price = regimes["series_id"].eq("price_ch_da")
    indices = regimes.index[price].tolist()
    boundary = pd.Timestamp("2026-01-02T00:30:00Z")
    regimes.loc[indices[0], "valid_to_utc"] = boundary
    regimes.loc[indices[1], "valid_from_utc"] = boundary

    with pytest.raises(EntsoeResolutionRegimeError, match="close the previous grid"):
        _profile(regimes=regimes)


def test_open_ended_regime_must_be_final() -> None:
    extra = _regime(
        "ntc_ch_fr",
        "2027-01-01T00:00:00Z",
        None,
        "PT15M",
        document="ntc-future-quarter-hourly",
    )
    regimes = pd.DataFrame(
        [*_regimes().to_dict(orient="records"), extra],
        columns=REGIME_COLUMNS,
    )

    with pytest.raises(EntsoeResolutionRegimeError, match="open-ended.*not final"):
        _profile(regimes=regimes)


def test_dimension_resolution_must_match_final_observed_regime() -> None:
    dimension = _dimension()
    dimension.loc[dimension["series_id"].eq("price_ch_da"), "native_resolution"] = "PT1H"

    with pytest.raises(EntsoeResolutionRegimeError, match="declared current resolution"):
        _profile(dimension=dimension)


def test_every_physical_series_requires_its_own_regime_evidence() -> None:
    regimes = _regimes().loc[lambda frame: frame["series_id"].ne("ntc_ch_fr")]

    with pytest.raises(EntsoeResolutionRegimeError, match="series coverage differs"):
        _profile(regimes=regimes)


def test_duplicate_target_key_is_rejected() -> None:
    targets = _targets()
    targets.loc[len(targets)] = targets.iloc[0]

    with pytest.raises(EntsoeResolutionRegimeError, match="key is duplicated"):
        _profile(targets=targets)


def test_naive_timestamps_are_rejected_instead_of_assumed_utc() -> None:
    targets = _targets()
    targets["target_ts_utc"] = targets["target_ts_utc"].dt.tz_localize(None)

    with pytest.raises(EntsoeResolutionRegimeError, match="timezone-aware UTC"):
        _profile(targets=targets)


def test_non_https_source_locator_is_rejected() -> None:
    regimes = _regimes()
    regimes.loc[0, "source_locator"] = "http://example.test/not-governed"

    with pytest.raises(EntsoeResolutionRegimeError, match="HTTPS locator"):
        _profile(regimes=regimes)


@pytest.mark.parametrize(
    "locator",
    [
        "https://",
        "https://user:secret@example.test/source",
        "https://example.test/source?token=secret",
        "https://example.test/source#fragment",
    ],
)
def test_unsafe_https_source_locator_is_rejected(locator: str) -> None:
    regimes = _regimes()
    regimes.loc[0, "source_locator"] = locator

    with pytest.raises(EntsoeResolutionRegimeError, match="HTTPS locator"):
        _profile(regimes=regimes)


def test_real_evidence_cannot_be_bootstrapped_by_a_caller_label() -> None:
    with pytest.raises(EntsoeResolutionRegimeError, match="independent artifact"):
        profile_resolution_regimes(
            _dimension(),
            _targets(),
            _regimes(),
            evidence_class="REAL_OWNER_REVIEWED",
            assessment_start_utc=ASSESSMENT_START,
            assessment_end_utc=ASSESSMENT_END,
            metadata_as_of_utc=METADATA_AS_OF,
        )


def test_profile_persists_hashes_not_raw_series_ids_and_no_authority() -> None:
    result = _profile()

    assert all("series_id" not in row for row in result.series_profile)
    assert all(len(str(row["series_id_sha256"])) == 64 for row in result.series_profile)
    assert result.summary["raw_series_ids_persisted_in_profile"] is False
    assert result.summary["opened_value_column_count"] == 0
    assert all(value is False for value in result.summary["authorities"].values())


def test_spring_dst_delivery_window_is_valid_at_its_native_hourly_cadence() -> None:
    dimension = pd.DataFrame(
        [
            {
                "series_id": "price_ch_dst",
                "group_name": "day_ahead_prices",
                "native_resolution": "PT1H",
            }
        ],
        columns=DIMENSION_COLUMNS,
    )
    targets = pd.DataFrame(
        {
            "series_id": ["price_ch_dst"] * 23,
            "target_ts_utc": pd.date_range(
                "2026-03-28T23:00:00Z",
                periods=23,
                freq="1h",
            ),
        },
        columns=TARGET_COLUMNS,
    )
    regimes = pd.DataFrame(
        [
            _regime(
                "price_ch_dst",
                "2020-01-01T00:00:00Z",
                None,
                "PT1H",
                document="price-hourly-dst",
            )
        ],
        columns=REGIME_COLUMNS,
    )
    regimes["valid_to_utc"] = pd.to_datetime(regimes["valid_to_utc"], utc=True)

    result = profile_resolution_regimes(
        dimension,
        targets,
        regimes,
        evidence_class=EVIDENCE_CLASS,
        assessment_start_utc=pd.Timestamp("2026-03-28T23:00:00Z"),
        assessment_end_utc=pd.Timestamp("2026-03-29T22:00:00Z"),
        metadata_as_of_utc=pd.Timestamp("2026-03-29T22:00:00Z"),
    )

    assert result.summary["status"] == PASS_STATUS
    assert result.summary["expected_target_count"] == 23
    assert result.summary["missing_native_slot_count"] == 0
