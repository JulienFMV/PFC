from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from pfc_shaping.validation.entsoe_local_readiness import (
    EntsoeLocalReadinessError,
    REQUIRED_FILES,
    audit_entsoe_local_readiness,
)


def _frames() -> dict[str, pd.DataFrame]:
    index = pd.date_range("2026-01-01", periods=8, freq="15min", tz="UTC")
    fundamentals = pd.DataFrame(
        {
            "load_mw": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0],
            "solar_mw": [0.0] * 8,
            "load_de_mw": [None, None, None, None, 1.0, 2.0, 3.0, 4.0],
        },
        index=index,
    )
    border = pd.DataFrame(
        {
            "flow_net_export_ch_de_mw": [None] * 4 + [1.0] * 4,
            "scheduled_net_export_ch_de_mw": [None] * 4 + [2.0] * 4,
            "ntc_export_ch_de_mw": [None] * 4 + [3.0] * 4,
        },
        index=index,
    )
    combined = pd.concat([fundamentals, border], axis=1)
    return {
        "entso_15min.parquet": combined,
        "entso_fundamentals_15min.parquet": fundamentals,
        "entso_border_15min.parquet": border,
        "de_renewable_forecast.parquet": pd.DataFrame(
            {"forecast_wind_de_mw": range(8)}, index=index
        ),
        "fr_nuclear_forecast_15min.parquet": pd.DataFrame(
            {"forecast_fr_nuclear_unavailable_mw": range(8)}, index=index
        ),
        "multi_country_forecast_15min.parquet": pd.DataFrame(
            {"forecast_load_ch_mw": range(8)}, index=index
        ),
        "outages_15min.parquet": pd.DataFrame(
            {"unavailable_mw": range(8)}, index=index
        ),
    }


def _manifest(frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    return {
        "schema_version": "entso_reusable_import.v2",
        "import_id": "fixture-import-v1",
        "calibration_eligible": False,
        "model_activation": "FORBIDDEN_UNTIL_GOVERNED_IMPORT",
        "source_coherence": "UNVERIFIED_MULTI_FILE_IMPORT",
        "files": {
            name: {
                "rows": len(frame),
                "columns": [str(column) for column in frame.columns],
            }
            for name, frame in frames.items()
        },
    }


def _hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_profiles_complete_local_surface_but_preserves_no_go() -> None:
    frames = _frames()
    manifest = _manifest(frames)

    result = audit_entsoe_local_readiness(
        manifest,
        frames,
        manifest_sha256=_hash(manifest),
        archive_integrity_verified=True,
    )

    assert result.summary["status"] == (
        "PASS_LOCAL_SCHEMA_AND_QUALITY_PROFILE_NO_GO_EMPIRICAL_USE"
    )
    assert result.summary["checks"]["required_files_present"] is True
    assert all(
        check["passed"]
        for check in result.summary["checks"]["duplicate_projection_checks"].values()
    )
    assert result.summary["authority"] == {
        "local_schema_and_quality_tooling": True,
        "historical_empirical_validation": False,
        "rolling_origin_selection": False,
        "model_training": False,
        "model_selection": False,
        "candidate_assembly": False,
        "production_promotion": False,
    }
    codes = {finding["code"] for finding in result.summary["findings"]}
    assert "FORECAST_POINT_IN_TIME_COLUMNS_MISSING" in codes
    assert "RAW_PROVIDER_LINEAGE_NOT_REPLAYABLE" in codes
    assert "CROSS_MARKET_SEASONAL_HISTORY_INSUFFICIENT" in codes


def test_series_profile_keeps_missingness_and_zero_information() -> None:
    frames = _frames()
    manifest = _manifest(frames)
    result = audit_entsoe_local_readiness(
        manifest,
        frames,
        manifest_sha256=_hash(manifest),
        archive_integrity_verified=True,
    )

    solar = result.series_profile.loc[
        (result.series_profile["file"] == "entso_fundamentals_15min.parquet")
        & (result.series_profile["series"] == "solar_mw")
    ].iloc[0]
    assert solar["all_zero_non_null"]
    assert solar["global_completeness"] == 1.0
    load = result.series_profile.loc[
        (result.series_profile["file"] == "entso_fundamentals_15min.parquet")
        & (result.series_profile["series"] == "load_mw")
    ].iloc[0]
    assert load["missing_count"] == 1
    assert load["point_in_time_proven"] == False  # noqa: E712
    assert load["native_resolution_proven"] == False  # noqa: E712


def test_projection_divergence_fails_local_consistency() -> None:
    frames = _frames()
    frames["entso_border_15min.parquet"] = frames[
        "entso_border_15min.parquet"
    ].copy()
    frames["entso_border_15min.parquet"].iloc[-1, 0] = 999.0
    manifest = _manifest(frames)

    result = audit_entsoe_local_readiness(
        manifest,
        frames,
        manifest_sha256=_hash(manifest),
        archive_integrity_verified=True,
    )

    assert result.summary["status"].startswith("FAIL_LOCAL_SCHEMA_OR_CONSISTENCY")
    assert "DUPLICATE_PROJECTION_DIVERGENCE" in {
        finding["code"] for finding in result.summary["findings"]
    }
    divergence = result.projection_profile.loc[
        (result.projection_profile["projection_file"] == "entso_border_15min.parquet")
        & (result.projection_profile["series"] == "flow_net_export_ch_de_mw")
    ].iloc[0]
    assert divergence["mismatch_count"] == 1
    assert divergence["max_abs_difference"] == 998.0


def test_missing_required_file_is_reported_without_granting_authority() -> None:
    frames = _frames()
    frames.pop(REQUIRED_FILES[0])
    manifest = _manifest(frames)

    result = audit_entsoe_local_readiness(
        manifest,
        frames,
        manifest_sha256=_hash(manifest),
        archive_integrity_verified=True,
    )

    assert result.summary["checks"]["required_files_present"] is False
    assert result.summary["authority"]["model_training"] is False


def test_rejects_authority_escalation() -> None:
    frames = _frames()
    manifest = _manifest(frames)
    manifest["calibration_eligible"] = True

    with pytest.raises(EntsoeLocalReadinessError, match="authority differs"):
        audit_entsoe_local_readiness(
            manifest,
            frames,
            manifest_sha256=_hash(manifest),
            archive_integrity_verified=True,
        )


def test_rejects_unverified_archive_or_bad_index() -> None:
    frames = _frames()
    manifest = _manifest(frames)
    with pytest.raises(EntsoeLocalReadinessError, match="integrity must be verified"):
        audit_entsoe_local_readiness(
            manifest,
            frames,
            manifest_sha256=_hash(manifest),
            archive_integrity_verified=False,
        )

    bad = _frames()
    bad["outages_15min.parquet"] = bad["outages_15min.parquet"].iloc[::-1]
    bad_manifest = _manifest(bad)
    with pytest.raises(EntsoeLocalReadinessError, match="duplicated or unsorted"):
        audit_entsoe_local_readiness(
            bad_manifest,
            bad,
            manifest_sha256=_hash(bad_manifest),
            archive_integrity_verified=True,
        )


def test_all_null_series_is_profiled_and_sub_quarter_hour_grid_is_rejected() -> None:
    frames = _frames()
    frames["outages_15min.parquet"]["unavailable_thermal"] = None
    manifest = _manifest(frames)
    result = audit_entsoe_local_readiness(
        manifest,
        frames,
        manifest_sha256=_hash(manifest),
        archive_integrity_verified=True,
    )
    thermal = result.series_profile.loc[
        result.series_profile["series"].eq("unavailable_thermal")
    ].iloc[0]
    assert thermal["valid_count"] == 0
    assert thermal["internal_completeness"] == 0.0

    bad = _frames()
    five_minute_index = pd.date_range(
        "2026-01-01", periods=8, freq="5min", tz="UTC"
    )
    bad["outages_15min.parquet"] = bad["outages_15min.parquet"].set_axis(
        five_minute_index
    )
    bad_manifest = _manifest(bad)
    with pytest.raises(EntsoeLocalReadinessError, match="15-minute grid"):
        audit_entsoe_local_readiness(
            bad_manifest,
            bad,
            manifest_sha256=_hash(bad_manifest),
            archive_integrity_verified=True,
        )


def test_databricks_intake_contract_preserves_native_pit_and_no_go() -> None:
    contract_path = Path(
        ".planning/phases/14-lt-audit-remediation/"
        "ENTSOE-DATABRICKS-INTAKE-CONTRACT-V1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert contract["schema_version"] == (
        "fmv_entsoe_databricks_intake_contract.v1"
    )
    assert contract["normalized_export"]["vintage_values"]["grain"] == [
        "series_id",
        "target_ts_utc",
        "as_of_utc",
    ]
    assert contract["time_contract"]["implicit_upsampling_authorized"] is False
    assert contract["time_contract"]["implicit_forward_fill_authorized"] is False
    assert contract["unit_contract"]["day_ahead_price"] == "EUR/MWh"
    assert contract["point_in_time_contract"][
        "backfill_may_reconstruct_historical_information_set"
    ] is False
    assert contract["quality_gates"][
        "minimum_complete_days_for_seasonal_cross_market_diagnostic"
    ] == 730
    assert set(contract["required_borders"]) == {
        "CH-DE",
        "CH-FR",
        "CH-IT",
        "CH-AT",
    }
    assert contract["authority"]["legacy_local_substitution_authorized"] is False
    assert contract["authority"]["model_training"] is False
    assert contract["authority"]["production_promotion"] is False
