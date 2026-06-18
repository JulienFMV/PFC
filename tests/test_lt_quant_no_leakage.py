from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pfc_shaping.data.lt_data_contract import (
    CoverageRequirement,
    FeatureSpec,
    artifact_record,
    asof_feature_frame,
    assert_no_future_available_at,
    coverage_matrix,
    manifest_skeleton,
    validate_run_manifest_schema,
)


def test_asof_feature_frame_filters_by_available_at_and_records_latest_only() -> None:
    raw = pd.DataFrame(
        {
            "feature": ["solar", "solar", "hydro"],
            "market": ["CH", "CH", "CH"],
            "value": [1.0, 9.0, 2.0],
            "available_at": [
                "2026-06-16T10:00:00Z",
                "2026-06-18T10:00:00Z",
                "2026-06-16T09:00:00Z",
            ],
        }
    )
    spec = FeatureSpec(
        feature_group="fundamentals",
        required_columns=("value",),
        key_columns=("feature", "market"),
    )

    frame = asof_feature_frame(raw, spec, valuation_timestamp="2026-06-17T00:00:00Z")

    assert set(frame["feature"]) == {"solar", "hydro"}
    assert frame.loc[frame["feature"].eq("solar"), "value"].item() == 1.0
    assert frame["available_at"].max() <= pd.Timestamp("2026-06-17T00:00:00Z")


def test_assert_no_future_available_at_rejects_leakage() -> None:
    raw = pd.DataFrame({"available_at": ["2026-06-18T10:00:00Z"]})
    spec = FeatureSpec(feature_group="test", required_columns=(), key_columns=())

    with pytest.raises(ValueError, match="after valuation_timestamp"):
        assert_no_future_available_at(raw, spec, valuation_timestamp="2026-06-17T00:00:00Z")


def test_coverage_matrix_reports_missing_required_feature() -> None:
    frame = pd.DataFrame({"feature": ["solar"], "market": ["CH"], "value": [1.0]})
    spec = FeatureSpec(feature_group="solar", required_columns=("value",), key_columns=("feature", "market"))
    requirement = CoverageRequirement(
        feature_group="solar",
        required_keys=pd.DataFrame({"feature": ["solar", "solar"], "market": ["CH", "DE"]}),
        min_coverage=1.0,
    )

    report = coverage_matrix(frame, spec, requirement)

    assert report.loc[0, "status"] == "NO_GO"
    assert report.loc[0, "present_rows"] == 1
    assert report.loc[0, "required_rows"] == 2


def test_manifest_schema_requires_auditable_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "curve.csv"
    artifact.write_text("timestamp,price\n2027-01-01T00:00:00Z,1\n", encoding="utf-8")
    as_of = "2026-06-17T00:00:00Z"
    record = artifact_record(artifact, feature_group="output_curve", available_at="2026-06-17T00:00:00Z")
    manifest = manifest_skeleton(
        run_id="run-1",
        git_sha="abcdef1",
        git_branch="fix/lt-audit-remediation",
        valuation_timestamp=as_of,
        input_artifacts=[record],
        data_vintages=[],
        qa_verdict="PASS",
    )

    validate_run_manifest_schema(manifest)

    manifest["input_artifacts"][0]["sha256"] = ""
    with pytest.raises(ValueError, match="sha256"):
        validate_run_manifest_schema(manifest)


def test_manifest_schema_rejects_future_data_vintage(tmp_path: Path) -> None:
    artifact = tmp_path / "source.csv"
    artifact.write_text("x\n1\n", encoding="utf-8")
    record = artifact_record(artifact, feature_group="source", available_at="2026-06-18T00:00:00Z")
    manifest = manifest_skeleton(
        run_id="run-2",
        git_sha="abcdef1",
        git_branch="fix/lt-audit-remediation",
        valuation_timestamp="2026-06-17T00:00:00Z",
        input_artifacts=[],
        data_vintages=[],
        qa_verdict="NO_GO",
    )
    manifest["data_vintages"] = [record]

    with pytest.raises(ValueError, match="data_vintages artifact available_at"):
        validate_run_manifest_schema(manifest)
