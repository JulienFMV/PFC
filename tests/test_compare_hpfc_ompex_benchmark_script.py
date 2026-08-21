from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook

from pfc_shaping.validation.ompex_truth_comparison import (
    OmpexTruthComparisonError,
    compare_hourly_forecasts_against_truth,
)
from scripts.compare_hpfc_ompex_benchmark import _series_from_frame, compare, main


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_ompex(
    path: Path, index: pd.DatetimeIndex, values: list[float]
) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "HFC"
    sheet.append(["Date", "EUR/MWh"])
    for timestamp, value in zip(index.tz_convert("Europe/Zurich"), values, strict=True):
        sheet.append(
            [
                (timestamp.tz_localize(None) + pd.Timedelta(hours=1)).to_pydatetime(),
                value,
            ]
        )
    buffer = BytesIO()
    workbook.save(buffer)
    workbook.close()
    path.write_bytes(buffer.getvalue())


def _fixtures(tmp_path: Path) -> tuple[Path, Path, Path, pd.DatetimeIndex]:
    index = pd.date_range("2026-07-01 00:00", periods=48, freq="h", tz="UTC")
    truth_values = [float(20 + position) for position in range(len(index))]
    candidate_values = [value + (-1.0 if position % 2 else 1.0) for position, value in enumerate(truth_values)]
    ompex_values = [value + 2.0 for value in truth_values]

    candidate = tmp_path / "candidate.csv"
    pd.DataFrame(
        {
            "delivery_start_utc": index,
            "price_weighted_mean_eur_mwh": candidate_values,
        }
    ).to_csv(candidate, index=False)
    truth = tmp_path / "truth.csv"
    pd.DataFrame(
        {"delivery_start_utc": index, "price_eur_mwh": truth_values}
    ).to_csv(truth, index=False)
    ompex = tmp_path / "HFC_Ompex_20260630_101700.xlsx"
    _write_ompex(ompex, index, ompex_values)
    return candidate, ompex, truth, index


def test_compare_requires_truth_and_reports_only_descriptive_paired_errors(
    tmp_path: Path,
) -> None:
    candidate, ompex, truth, index = _fixtures(tmp_path)

    summary = compare(
        hpfc_csv=candidate,
        hpfc_sha256=_sha256(candidate),
        ompex_xlsx=ompex,
        ompex_sha256=_sha256(ompex),
        truth_csv=truth,
        truth_sha256=_sha256(truth),
        output_dir=tmp_path / "benchmark",
        origin_utc=pd.Timestamp("2026-06-30 10:17", tz="UTC"),
        target_start_utc=index[0],
        target_end_utc=index[-1] + pd.Timedelta(hours=1),
    )

    assert summary["status"] == "DESCRIPTIVE_ONLY_NO_SUPERIORITY_DECISION"
    assert summary["candidate"]["mae_eur_mwh"] == 1.0
    assert summary["ompex"]["mae_eur_mwh"] == 2.0
    assert summary["paired"]["mae_delta_candidate_minus_ompex_eur_mwh"] == -1.0
    assert summary["paired"]["decision"] == "NOT_EVALUATED"
    assert summary["superiority_claim_authorized"] is False
    assert summary["automatic_alignment_selection_used"] is False
    assert summary["timestamp_semantics_authenticated"] is False
    assert summary["truth_independence_authenticated"] is False
    assert summary["same_missingness_mask"] is True
    assert summary["benchmark_policy"] == "advisory"
    assert summary["ompex_used_in_model"] is False
    assert summary["ompex_used_in_selection"] is False
    assert (tmp_path / "benchmark" / "paired_hourly_against_truth.csv").exists()
    assert (tmp_path / "benchmark" / "subgroup_metrics.csv").exists()
    assert (tmp_path / "benchmark" / "benchmark_metrics.json").exists()
    assert not (tmp_path / "benchmark" / "alignment_sensitivity.csv").exists()


def test_exact_common_grid_forbids_missingness_advantage(tmp_path: Path) -> None:
    candidate, ompex, truth, index = _fixtures(tmp_path)
    truth_frame = pd.read_csv(truth).iloc[:-1]
    truth_frame.to_csv(truth, index=False)

    with pytest.raises(OmpexTruthComparisonError, match="truth does not exactly cover"):
        compare(
            hpfc_csv=candidate,
            hpfc_sha256=_sha256(candidate),
            ompex_xlsx=ompex,
            ompex_sha256=_sha256(ompex),
            truth_csv=truth,
            truth_sha256=_sha256(truth),
            output_dir=tmp_path / "benchmark",
            origin_utc=pd.Timestamp("2026-06-30 10:17", tz="UTC"),
            target_start_utc=index[0],
            target_end_utc=index[-1] + pd.Timedelta(hours=1),
        )


def test_hash_binding_and_post_origin_window_fail_closed(tmp_path: Path) -> None:
    candidate, ompex, truth, index = _fixtures(tmp_path)
    with pytest.raises(ValueError, match="HPFC candidate SHA-256 mismatch"):
        compare(
            hpfc_csv=candidate,
            hpfc_sha256="0" * 64,
            ompex_xlsx=ompex,
            ompex_sha256=_sha256(ompex),
            truth_csv=truth,
            truth_sha256=_sha256(truth),
            output_dir=tmp_path / "benchmark",
            origin_utc=pd.Timestamp("2026-06-30 10:17", tz="UTC"),
            target_start_utc=index[0],
            target_end_utc=index[-1] + pd.Timedelta(hours=1),
        )

    series = pd.Series(range(len(index)), index=index, dtype="float64")
    with pytest.raises(OmpexTruthComparisonError, match="target starts before"):
        compare_hourly_forecasts_against_truth(
            candidate=series,
            ompex=series,
            truth=series,
            origin_utc=index[1],
            target_start_utc=index[0],
            target_end_utc=index[-1] + pd.Timedelta(hours=1),
        )


def test_cli_has_no_auto_alignment_or_two_curve_mode() -> None:
    with pytest.raises(SystemExit):
        main(["--alignment", "auto"])

    source = Path("scripts/compare_hpfc_ompex_benchmark.py").read_text(
        encoding="utf-8"
    )
    assert "H:" + "\\" not in source
    assert "alignment_sensitivity" not in source
    assert "--truth-csv" in source


def test_diagnostic_contract_cannot_claim_superiority() -> None:
    path = Path(
        ".planning/phases/14-lt-audit-remediation/"
        "OMPEX-PAIRED-TRUTH-DIAGNOSTIC-CONTRACT-V1.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))

    assert contract["status"] == "DESCRIPTIVE_ONLY_NO_SUPERIORITY_DECISION"
    assert contract["alignment"]["automatic_alignment_selection_authorized"] is False
    assert contract["alignment"]["same_missingness_mask_required"] is True
    assert contract["decision_boundary"]["superiority_claim_authorized"] is False
    assert contract["decision_boundary"]["countable_origin_count"] == 0
    assert contract["local_execution"]["h_drive_runtime_dependency"] is False
    assert contract["local_execution"]["databricks_request_required"] is False


@pytest.mark.parametrize(
    "start",
    ["2026-03-28 22:00:00+00:00", "2026-10-24 22:00:00+00:00"],
)
def test_candidate_local_timestamp_and_explicit_offset_round_trip_dst(start: str) -> None:
    expected = pd.date_range(start, periods=36, freq="h")
    local = expected.tz_convert("Europe/Zurich")
    frame = pd.DataFrame(
        {
            "timestamp_ch": local.tz_localize(None).strftime("%d.%m.%Y %H:%M"),
            "utc_offset_ch": [
                f"UTC{timestamp.strftime('%z')[:3]}:{timestamp.strftime('%z')[3:]}"
                for timestamp in local
            ],
            "price_weighted_mean_eur_mwh": range(len(expected)),
        }
    )

    parsed = _series_from_frame(
        frame,
        price_column="price_weighted_mean_eur_mwh",
        label="candidate",
    )

    pd.testing.assert_index_equal(parsed.index, expected.rename("delivery_start_utc"))
