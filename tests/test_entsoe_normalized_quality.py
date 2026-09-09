from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from pfc_shaping.validation.entsoe_normalized_quality import (
    CONTRACT_CONTENT_ID,
    EntsoeNormalizedQualityError,
    profile_entsoe_normalized_quality,
    replay_vintages_as_of,
    validate_normalized_quality_contract,
    verify_normalized_quality_source_paths,
)

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / ".planning" / "phases" / "14-lt-audit-remediation"
CONTRACT_PATH = PHASE / "ENTSOE-NORMALIZED-QUALITY-PROFILE-CONTRACT-V1.json"
INTAKE_CONTRACT_PATH = PHASE / "ENTSOE-DATABRICKS-INTAKE-CONTRACT-V1.json"
PROOF_ROOT = ROOT / "build" / "databricks-eex-daily" / "2026-08-05"
SOURCE_PROOF_PATHS = {
    "d233": (
        PROOF_ROOT
        / "eex-entsoe-governed-acquisition-package-proofs"
        / "314ec85590c787874e2844d7db085236144c601681a5d1722f2735e6b1219d53"
        / "manifest.json"
    ),
    "d234": (
        PROOF_ROOT
        / "entsoe-physical-mapping-compiler-proofs"
        / "bb7ea1894463cb2c5fc30287d2239f0285cd6a74e901ab51a2f9de7e6794b766"
        / "manifest.json"
    ),
    "d235": (
        PROOF_ROOT
        / "eex-entsoe-external-time-batch-proofs"
        / "93504d69834b299ce361352adce433509cf84bce0fbbb940be00a9bd59616ff1"
        / "manifest.json"
    ),
    "d236": (
        PROOF_ROOT
        / "entsoe-bounded-execution-receipt-proofs"
        / "772da4e0b22540bf22e1715ca146cc0a59adf0c7a9ec508e5a893d8495539247"
        / "manifest.json"
    ),
    "d237": (
        PROOF_ROOT
        / "eex-entsoe-rfc3161-request-der-proofs"
        / "53e2222392f71541d28e05e1dfc912361c02003b4278f2770109827319d4e9c1"
        / "manifest.json"
    ),
    "d238": (
        PROOF_ROOT
        / "entsoe-daily-capture-reservation-ledger-proofs"
        / "2b9f6c513e0382e685bee78fb02d6c071a6954a26e715c5784fb28efec878aa8"
        / "manifest.json"
    ),
}
SNAPSHOT = datetime(2026, 8, 2, 3, 0, tzinfo=timezone.utc)

PRE_TARGET_GROUPS = {
    "day_ahead_prices",
    "generation_forecast",
    "load_forecast",
    "renewables_forecast_day_ahead",
    "renewables_forecast_intraday",
    "scheduled_exchanges",
    "ntc_day_ahead",
    "ntc_month_ahead",
    "ntc_year_ahead",
}
CROSSBORDER_GROUPS = {
    "crossborder_physical_flows",
    "scheduled_exchanges",
    "ntc_day_ahead",
    "ntc_month_ahead",
    "ntc_year_ahead",
}


def _dimension() -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(
        group: str,
        field: str,
        *,
        unit: str = "MW",
        from_zone: str = "CH",
        to_zone: str = "CH",
    ) -> None:
        suffix = "domestic" if to_zone == "CH" else to_zone.lower()
        rows.append(
            {
                "series_id": f"{group}__{field}__{suffix}",
                "group_name": group,
                "field_name": field,
                "from_zone": from_zone,
                "to_zone": to_zone,
                "unit": unit,
                "native_resolution": "PT1H",
                "source_endpoint": f"entsoe/{group}",
                "document_id": f"document/{group}/{field}/{suffix}",
            }
        )

    add("actual_load", "load")
    add("day_ahead_prices", "price", unit="EUR/MWh")
    for field in (
        "solar",
        "wind",
        "nuclear",
        "hydro_run_of_river",
        "hydro_reservoir",
        "hydro_pumped_storage",
    ):
        add("generation_actual", field)
    add("generation_forecast", "generation")
    add("hydro_reservoir_storage", "reservoir", unit="MWh")
    add("load_forecast", "load")
    add("renewables_forecast_day_ahead", "renewables")
    add("renewables_forecast_intraday", "renewables")
    for group in sorted(CROSSBORDER_GROUPS):
        for neighbor in ("DE", "FR", "IT", "AT"):
            add(group, f"{group}_{neighbor.lower()}", to_zone=neighbor)
    return pd.DataFrame(rows)


def _fixture() -> dict[str, object]:
    dimension = _dimension()
    targets = pd.date_range("2026-08-01", periods=24, freq="1h", tz="UTC")
    latest_rows: list[dict[str, object]] = []
    vintage_rows: list[dict[str, object]] = []
    semantics: dict[str, str] = {}
    backfill: dict[str, str] = {}
    for metadata in dimension.itertuples(index=False):
        if metadata.group_name == "day_ahead_prices":
            semantics[metadata.series_id] = "PRICE_CAN_BE_NEGATIVE"
        elif metadata.group_name in CROSSBORDER_GROUPS:
            semantics[metadata.series_id] = "SIGNED_POSITIVE_CH_TO_NEIGHBOR"
        elif metadata.field_name == "hydro_pumped_storage":
            semantics[metadata.series_id] = "SIGNED_SOURCE_DEFINED_DOCUMENTED"
        else:
            semantics[metadata.series_id] = "NON_NEGATIVE_PHYSICAL_QUANTITY"
        backfill[metadata.series_id] = (
            "HISTORICAL_BACKFILL"
            if metadata.group_name == "actual_load"
            else "NATIVE_VINTAGES"
        )
        for index, target in enumerate(targets):
            is_pre_target = metadata.group_name in PRE_TARGET_GROUPS
            as_of = target - pd.Timedelta(hours=1) if is_pre_target else target + pd.Timedelta(hours=1)
            value = float(index + 1)
            if index == 0 and (
                metadata.group_name in CROSSBORDER_GROUPS
                or metadata.group_name == "day_ahead_prices"
                or metadata.field_name == "hydro_pumped_storage"
            ):
                value = -1.0
            common = {
                "series_id": metadata.series_id,
                "target_ts_utc": target,
                "value": value,
                "quality_flag": "OK",
                "revision_number": 1,
                "meta_load_timestamp_utc": as_of,
            }
            latest_rows.append({**common, "is_historical": True})
            vintage_rows.append({**common, "as_of_utc": as_of})
    latest = pd.DataFrame(latest_rows)[
        [
            "series_id",
            "target_ts_utc",
            "value",
            "is_historical",
            "quality_flag",
            "revision_number",
            "meta_load_timestamp_utc",
        ]
    ]
    vintage = pd.DataFrame(vintage_rows)[
        [
            "series_id",
            "target_ts_utc",
            "as_of_utc",
            "value",
            "quality_flag",
            "revision_number",
            "meta_load_timestamp_utc",
        ]
    ]
    return {
        "series_dimension": dimension,
        "latest_values": latest,
        "vintage_values": vintage,
        "sign_semantics_by_series": semantics,
        "backfill_status_by_series": backfill,
    }


def _profile(fixture: dict[str, object], **overrides):
    arguments = {
        "evidence_class": "SYNTHETIC_TEST_ONLY",
        "snapshot_reference_time_utc": SNAPSHOT,
        "artifact_hashes_verified": False,
        "real_receipt_admitted": False,
    }
    arguments.update(overrides)
    return profile_entsoe_normalized_quality(
        fixture["series_dimension"],
        fixture["latest_values"],
        fixture["vintage_values"],
        sign_semantics_by_series=fixture["sign_semantics_by_series"],
        backfill_status_by_series=fixture["backfill_status_by_series"],
        **arguments,
    )


def _drop_series(fixture: dict[str, object], series_id: str, *, role: str) -> None:
    frame = fixture[role]
    fixture[role] = frame.loc[frame["series_id"].ne(series_id)].reset_index(drop=True)


def test_contract_and_source_proofs_are_zero_execution_bound() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    result = validate_normalized_quality_contract(contract)
    assert result["contract_content_id"] == CONTRACT_CONTENT_ID
    assert result["databricks_statement_budget"] == 0
    assert result["artifact_opening_authorized"] is False
    bound = verify_normalized_quality_source_paths(
        contract_path=CONTRACT_PATH,
        intake_contract_path=INTAKE_CONTRACT_PATH,
        source_proof_manifest_paths=SOURCE_PROOF_PATHS,
    )
    assert bound["contract_content_id"] == CONTRACT_CONTENT_ID
    assert set(bound["source_proof_content_ids"]) == {
        "d233",
        "d234",
        "d235",
        "d236",
        "d237",
        "d238",
    }


def test_profiles_complete_normalized_fixture_but_preserves_no_go() -> None:
    result = _profile(_fixture())
    assert result.summary["status"] == "PASS_SYNTHETIC_STRUCTURAL_PROFILE_MODEL_NO_GO"
    assert result.summary["checks"]["primary_keys_unique"] is True
    assert result.summary["checks"]["required_borders_by_crossborder_group_present"] is True
    assert result.summary["checks"]["latest_vintage_overlap_mismatch_count"] == 0
    assert result.summary["checks"]["seasonal_complete_day_threshold_met"] is False
    assert result.summary["coverage"]["minimum_complete_utc_days"] == 1
    assert result.summary["coverage"]["minimum_grid_completeness_ratio"] == 1.0
    assert result.summary["coverage"]["latest_to_vintage_overlap_coverage_ratio"] == 1.0
    assert all(value == 0 for value in result.summary["execution"].values())
    assert result.summary["authorities"]["model_input_authorized"] is False
    assert result.summary["authorities"]["production_authorized"] is False
    codes = {finding["code"] for finding in result.summary["findings"]}
    assert "SYNTHETIC_FIXTURE_NOT_EMPIRICAL_EVIDENCE" in codes
    assert "SEASONAL_COMPLETE_DAY_THRESHOLD_NOT_MET" in codes
    assert "BACKFILLED_SERIES_NOT_RETROACTIVE_PIT_EVIDENCE" in codes
    assert "NEW_INDEPENDENT_FUTURE_HOLDOUT_MISSING" in codes


def test_profiles_are_value_free_and_exactly_reconcile_latest_vintage() -> None:
    result = _profile(_fixture())
    assert "value" not in result.series_profile.columns
    assert result.series_profile["raw_values_persisted_in_profile"].eq(False).all()
    assert result.overlap_profile["value_equal"].all()
    assert len(result.overlap_profile) == result.summary["dataset"]["latest_row_count"]


def test_replay_vintages_as_of_never_uses_later_revision() -> None:
    fixture = _fixture()
    vintage = fixture["vintage_values"].copy()
    row = vintage.iloc[0].copy()
    first_as_of = row["as_of_utc"]
    row["as_of_utc"] = first_as_of + pd.Timedelta(hours=1)
    row["meta_load_timestamp_utc"] = row["as_of_utc"]
    row["revision_number"] = 2
    row["value"] = 999.0
    vintage = pd.concat([vintage, row.to_frame().T], ignore_index=True)
    for column in ("target_ts_utc", "as_of_utc", "meta_load_timestamp_utc"):
        vintage[column] = pd.to_datetime(vintage[column], utc=True)
    vintage["revision_number"] = vintage["revision_number"].astype("int64")
    before = replay_vintages_as_of(vintage, origin_utc=first_as_of.to_pydatetime())
    selected = before.loc[
        before["series_id"].eq(row["series_id"])
        & before["target_ts_utc"].eq(row["target_ts_utc"])
    ].iloc[0]
    assert selected["revision_number"] == 1
    after = replay_vintages_as_of(
        vintage,
        origin_utc=(first_as_of + pd.Timedelta(hours=1)).to_pydatetime(),
    )
    selected_after = after.loc[
        after["series_id"].eq(row["series_id"])
        & after["target_ts_utc"].eq(row["target_ts_utc"])
    ].iloc[0]
    assert selected_after["revision_number"] == 2
    assert selected_after["as_of_utc"] <= first_as_of + pd.Timedelta(hours=1)


def test_real_class_cannot_be_bootstrapped_by_caller_booleans() -> None:
    fixture = _fixture()
    with pytest.raises(EntsoeNormalizedQualityError, match="independent receipt"):
        _profile(fixture, evidence_class="REAL_HASH_ADMITTED_ARTIFACTS")
    with pytest.raises(EntsoeNormalizedQualityError, match="independent receipt"):
        _profile(
            fixture,
            evidence_class="REAL_HASH_ADMITTED_ARTIFACTS",
            artifact_hashes_verified=True,
            real_receipt_admitted=True,
        )


def test_synthetic_class_rejects_real_admission_claims() -> None:
    with pytest.raises(EntsoeNormalizedQualityError, match="cannot bootstrap"):
        _profile(_fixture(), artifact_hashes_verified=True)


def test_duplicate_key_null_or_orphan_is_rejected() -> None:
    fixture = _fixture()
    fixture["latest_values"] = pd.concat(
        [fixture["latest_values"], fixture["latest_values"].iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(EntsoeNormalizedQualityError, match="primary key"):
        _profile(fixture)

    fixture = _fixture()
    fixture["latest_values"].loc[0, "quality_flag"] = None
    with pytest.raises(EntsoeNormalizedQualityError, match="null required"):
        _profile(fixture)

    fixture = _fixture()
    fixture["latest_values"].loc[0, "series_id"] = "unknown-series"
    with pytest.raises(EntsoeNormalizedQualityError, match="orphan series"):
        _profile(fixture)


def test_dimension_semantics_units_and_coverage_are_hard_gates() -> None:
    fixture = _fixture()
    fixture["series_dimension"].loc[
        fixture["series_dimension"]["group_name"].eq("day_ahead_prices"), "unit"
    ] = "EUR"
    with pytest.raises(EntsoeNormalizedQualityError, match="unit is invalid"):
        _profile(fixture)

    fixture = _fixture()
    series_id = next(iter(fixture["sign_semantics_by_series"]))
    fixture["sign_semantics_by_series"].pop(series_id)
    with pytest.raises(EntsoeNormalizedQualityError, match="series coverage"):
        _profile(fixture)

    fixture = _fixture()
    border_series = fixture["series_dimension"].loc[
        fixture["series_dimension"]["group_name"].eq("ntc_day_ahead")
        & fixture["series_dimension"]["to_zone"].eq("AT"),
        "series_id",
    ].iloc[0]
    fixture["series_dimension"] = fixture["series_dimension"].loc[
        fixture["series_dimension"]["series_id"].ne(border_series)
    ].reset_index(drop=True)
    _drop_series(fixture, border_series, role="latest_values")
    _drop_series(fixture, border_series, role="vintage_values")
    fixture["sign_semantics_by_series"].pop(border_series)
    fixture["backfill_status_by_series"].pop(border_series)
    with pytest.raises(EntsoeNormalizedQualityError, match="required border coverage"):
        _profile(fixture)


def test_identifier_resolution_zone_and_sign_direction_are_validated() -> None:
    fixture = _fixture()
    fixture["series_dimension"].loc[0, "series_id"] += " "
    with pytest.raises(EntsoeNormalizedQualityError, match="untrimmed string"):
        _profile(fixture)

    fixture = _fixture()
    fixture["series_dimension"].loc[0, "native_resolution"] = "PT45M"
    with pytest.raises(EntsoeNormalizedQualityError, match="native resolution"):
        _profile(fixture)

    fixture = _fixture()
    fixture["series_dimension"].loc[0, "from_zone"] = "GB"
    with pytest.raises(EntsoeNormalizedQualityError, match="zone is invalid"):
        _profile(fixture)

    fixture = _fixture()
    border_id = fixture["series_dimension"].loc[
        fixture["series_dimension"]["group_name"].eq("scheduled_exchanges"),
        "series_id",
    ].iloc[0]
    fixture["sign_semantics_by_series"][border_id] = (
        "SIGNED_POSITIVE_NEIGHBOR_TO_CH"
    )
    with pytest.raises(EntsoeNormalizedQualityError, match="sign direction"):
        _profile(fixture)


def test_native_grid_nonfinite_revision_and_future_load_are_rejected() -> None:
    fixture = _fixture()
    fixture["latest_values"].loc[0, "target_ts_utc"] += pd.Timedelta(minutes=30)
    with pytest.raises(EntsoeNormalizedQualityError, match="off native grid"):
        _profile(fixture)

    fixture = _fixture()
    fixture["latest_values"].loc[0, "value"] = float("inf")
    with pytest.raises(EntsoeNormalizedQualityError, match="non-finite"):
        _profile(fixture)

    fixture = _fixture()
    fixture["vintage_values"].loc[0, "revision_number"] = -1
    with pytest.raises(EntsoeNormalizedQualityError, match="non-negative integer"):
        _profile(fixture)

    fixture = _fixture()
    fixture["latest_values"].loc[0, "meta_load_timestamp_utc"] = pd.Timestamp(
        "2026-08-03", tz="UTC"
    )
    with pytest.raises(EntsoeNormalizedQualityError, match="meta load exceeds snapshot"):
        _profile(fixture)


def test_timezone_asof_meta_load_and_revision_chronology_are_enforced() -> None:
    fixture = _fixture()
    fixture["latest_values"]["target_ts_utc"] = fixture["latest_values"][
        "target_ts_utc"
    ].dt.tz_localize(None)
    with pytest.raises(EntsoeNormalizedQualityError, match="timezone-aware UTC"):
        _profile(fixture)

    fixture = _fixture()
    fixture["vintage_values"].loc[0, "as_of_utc"] = (
        fixture["vintage_values"].loc[0, "meta_load_timestamp_utc"]
        + pd.Timedelta(seconds=1)
    )
    with pytest.raises(EntsoeNormalizedQualityError, match="as_of exceeds meta load"):
        _profile(fixture)

    fixture = _fixture()
    vintage = fixture["vintage_values"]
    row = vintage.iloc[0].copy()
    row["as_of_utc"] += pd.Timedelta(minutes=30)
    row["meta_load_timestamp_utc"] = row["as_of_utc"]
    row["revision_number"] = 0
    fixture["vintage_values"] = pd.concat(
        [vintage, row.to_frame().T],
        ignore_index=True,
    )
    for column in ("target_ts_utc", "as_of_utc", "meta_load_timestamp_utc"):
        fixture["vintage_values"][column] = pd.to_datetime(
            fixture["vintage_values"][column],
            utc=True,
        )
    fixture["vintage_values"]["revision_number"] = fixture["vintage_values"][
        "revision_number"
    ].astype("int64")
    fixture["vintage_values"]["value"] = fixture["vintage_values"]["value"].astype(
        "float64"
    )
    with pytest.raises(EntsoeNormalizedQualityError, match="revisions decrease"):
        _profile(fixture)


def test_generation_breakdown_and_required_group_cannot_disappear() -> None:
    fixture = _fixture()
    series_id = fixture["series_dimension"].loc[
        fixture["series_dimension"]["field_name"].eq("solar"), "series_id"
    ].iloc[0]
    fixture["series_dimension"] = fixture["series_dimension"].loc[
        fixture["series_dimension"]["series_id"].ne(series_id)
    ].reset_index(drop=True)
    _drop_series(fixture, series_id, role="latest_values")
    _drop_series(fixture, series_id, role="vintage_values")
    fixture["sign_semantics_by_series"].pop(series_id)
    fixture["backfill_status_by_series"].pop(series_id)
    with pytest.raises(EntsoeNormalizedQualityError, match="generation actual"):
        _profile(fixture)

    fixture = _fixture()
    group = "load_forecast"
    series_id = fixture["series_dimension"].loc[
        fixture["series_dimension"]["group_name"].eq(group), "series_id"
    ].iloc[0]
    fixture["series_dimension"] = fixture["series_dimension"].loc[
        fixture["series_dimension"]["series_id"].ne(series_id)
    ].reset_index(drop=True)
    _drop_series(fixture, series_id, role="latest_values")
    _drop_series(fixture, series_id, role="vintage_values")
    fixture["sign_semantics_by_series"].pop(series_id)
    fixture["backfill_status_by_series"].pop(series_id)
    with pytest.raises(EntsoeNormalizedQualityError, match="required groups missing"):
        _profile(fixture)


def test_forecast_and_actual_availability_direction_is_enforced() -> None:
    fixture = _fixture()
    forecast_id = fixture["series_dimension"].loc[
        fixture["series_dimension"]["group_name"].eq("load_forecast"), "series_id"
    ].iloc[0]
    row = fixture["vintage_values"]["series_id"].eq(forecast_id)
    fixture["vintage_values"].loc[row, "as_of_utc"] = (
        fixture["vintage_values"].loc[row, "target_ts_utc"] + pd.Timedelta(hours=1)
    )
    fixture["vintage_values"].loc[row, "meta_load_timestamp_utc"] = fixture[
        "vintage_values"
    ].loc[row, "as_of_utc"]
    with pytest.raises(EntsoeNormalizedQualityError, match="available after target"):
        _profile(fixture)

    fixture = _fixture()
    actual_id = fixture["series_dimension"].loc[
        fixture["series_dimension"]["group_name"].eq("actual_load"), "series_id"
    ].iloc[0]
    row = fixture["vintage_values"]["series_id"].eq(actual_id)
    fixture["vintage_values"].loc[row, "as_of_utc"] = (
        fixture["vintage_values"].loc[row, "target_ts_utc"] - pd.Timedelta(hours=1)
    )
    fixture["vintage_values"].loc[row, "meta_load_timestamp_utc"] = fixture[
        "vintage_values"
    ].loc[row, "target_ts_utc"]
    with pytest.raises(EntsoeNormalizedQualityError, match="available before target"):
        _profile(fixture)


def test_negative_values_and_latest_vintage_divergence_are_rejected() -> None:
    fixture = _fixture()
    nonnegative_id = fixture["series_dimension"].loc[
        fixture["series_dimension"]["group_name"].eq("actual_load"), "series_id"
    ].iloc[0]
    row = fixture["latest_values"]["series_id"].eq(nonnegative_id)
    fixture["latest_values"].loc[row.idxmax(), "value"] = -1.0
    with pytest.raises(EntsoeNormalizedQualityError, match="sign semantics"):
        _profile(fixture)

    fixture = _fixture()
    fixture["latest_values"].loc[0, "value"] = 12345.0
    with pytest.raises(EntsoeNormalizedQualityError, match="differ from last vintage"):
        _profile(fixture)


def test_dimension_to_fact_series_coverage_is_exact() -> None:
    fixture = _fixture()
    series_id = fixture["series_dimension"]["series_id"].iloc[0]
    _drop_series(fixture, series_id, role="latest_values")
    with pytest.raises(EntsoeNormalizedQualityError, match="dimension to latest"):
        _profile(fixture)

    fixture = _fixture()
    series_id = fixture["series_dimension"]["series_id"].iloc[0]
    _drop_series(fixture, series_id, role="vintage_values")
    with pytest.raises(EntsoeNormalizedQualityError, match="dimension to vintage"):
        _profile(fixture)


def test_source_proof_tamper_is_rejected(tmp_path: Path) -> None:
    tampered = json.loads(SOURCE_PROOF_PATHS["d236"].read_text(encoding="utf-8"))
    tampered["execution"]["warehouse_start_count"] = 1
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    paths = dict(SOURCE_PROOF_PATHS)
    paths["d236"] = tampered_path
    with pytest.raises(EntsoeNormalizedQualityError, match="proof hash differs"):
        verify_normalized_quality_source_paths(
            contract_path=CONTRACT_PATH,
            intake_contract_path=INTAKE_CONTRACT_PATH,
            source_proof_manifest_paths=paths,
        )
