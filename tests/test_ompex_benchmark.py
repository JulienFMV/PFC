from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook

from pfc_shaping.validation.ompex_benchmark import (
    OmpexBenchmarkError,
    hourly_transport_to_quarter_hour,
    parse_ompex_workbook,
    select_post_origin_rows,
)


def _workbook_bytes(
    delivery_starts_utc: pd.DatetimeIndex,
    *,
    formula: bool = False,
    headers: tuple[str, str] = ("Date", "EUR/MWh"),
) -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "HFC"
    sheet.append(headers)
    local_starts = delivery_starts_utc.tz_convert("Europe/Zurich")
    for position, local_start in enumerate(local_starts):
        hour_end_label = (local_start.tz_localize(None) + pd.Timedelta(hours=1)).to_pydatetime()
        price: object = f"=40+{position}" if formula and position == 0 else 40 + position
        sheet.append([hour_end_label, price])
    buffer = BytesIO()
    workbook.save(buffer)
    workbook.close()
    return buffer.getvalue()


@pytest.mark.parametrize(
    "start",
    ["2026-03-28 22:00:00+00:00", "2026-10-24 22:00:00+00:00"],
)
def test_parses_dst_as_local_hour_end_and_produces_contiguous_utc(start: str) -> None:
    expected = pd.date_range(start, periods=36, freq="h")
    payload = _workbook_bytes(expected)

    parsed = parse_ompex_workbook(
        payload,
        source_name="HFC_Ompex_20260805_101700.xlsx",
        accept_unconfirmed_hour_end_semantics=True,
    )

    pd.testing.assert_index_equal(
        pd.DatetimeIndex(parsed.data["delivery_start_utc"]),
        expected.rename("delivery_start_utc"),
    )
    assert parsed.receipt["native_resolution"] == "PT1H"
    assert parsed.receipt["native_quarter_hour"] is False
    assert parsed.receipt["hour_end_interpretation_vendor_authenticated"] is False
    assert parsed.receipt["price_statistics_emitted"] is False
    assert parsed.receipt["authority"]["benchmark_only"] is True
    assert all(
        parsed.receipt["authority"][key] is False
        for key in ("model_input", "training", "tuning", "selection", "promotion")
    )


def test_refuses_implicit_timestamp_semantics_and_formula() -> None:
    expected = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    payload = _workbook_bytes(expected)
    with pytest.raises(OmpexBenchmarkError, match="explicit acceptance"):
        parse_ompex_workbook(payload, source_name="HFC_Ompex_test.xlsx")

    formula_payload = _workbook_bytes(expected, formula=True)
    with pytest.raises(OmpexBenchmarkError, match="formulas are forbidden"):
        parse_ompex_workbook(
            formula_payload,
            source_name="HFC_Ompex_test.xlsx",
            accept_unconfirmed_hour_end_semantics=True,
        )


def test_refuses_wrong_schema_nonfinite_and_nonhourly_curve() -> None:
    expected = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    wrong_headers = _workbook_bytes(expected, headers=("timestamp", "price"))
    with pytest.raises(OmpexBenchmarkError, match="headers must be exactly"):
        parse_ompex_workbook(
            wrong_headers,
            source_name="HFC_Ompex_test.xlsx",
            accept_unconfirmed_hour_end_semantics=True,
        )

    nonhourly = expected.copy()
    nonhourly = nonhourly.delete(2).insert(2, expected[2] + pd.Timedelta(minutes=15))
    with pytest.raises(OmpexBenchmarkError, match="not native hourly"):
        parse_ompex_workbook(
            _workbook_bytes(nonhourly),
            source_name="HFC_Ompex_test.xlsx",
            accept_unconfirmed_hour_end_semantics=True,
        )


def test_post_origin_filter_excludes_rewritten_history() -> None:
    expected = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    parsed = parse_ompex_workbook(
        _workbook_bytes(expected),
        source_name="HFC_Ompex_20260101_101700.xlsx",
        accept_unconfirmed_hour_end_semantics=True,
    )

    selected = select_post_origin_rows(
        parsed, origin_utc=pd.Timestamp("2026-01-01 04:00", tz="UTC")
    )

    assert len(selected) == 4
    assert selected["delivery_start_utc"].min() == pd.Timestamp(
        "2026-01-01 04:00", tz="UTC"
    )
    with pytest.raises(OmpexBenchmarkError, match="timezone-aware"):
        select_post_origin_rows(parsed, origin_utc=pd.Timestamp("2026-01-01"))


def test_quarter_hour_expansion_is_transport_and_keeps_hour_clusters() -> None:
    expected = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    parsed = parse_ompex_workbook(
        _workbook_bytes(expected),
        source_name="HFC_Ompex_test.xlsx",
        accept_unconfirmed_hour_end_semantics=True,
    )

    transported = hourly_transport_to_quarter_hour(parsed.data)

    assert len(transported) == 12
    assert transported["native_quarter_hour"].eq(False).all()
    assert transported["effective_hour_cluster"].nunique() == 3
    assert transported["price_eur_mwh"].nunique() == 3


def test_contract_is_outcome_blind_and_does_not_fake_quarter_hour_truth() -> None:
    contract_path = Path(
        ".planning/phases/14-lt-audit-remediation/"
        "OMPEX-INDEPENDENT-BENCHMARK-CONTRACT-V1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    role = contract["benchmark_role"]
    assert role["ompex_is_read_only_post_freeze_benchmark"] is True
    assert role["candidate_and_ompex_are_scored_against_independent_truth"] is True
    assert role["ompex_is_model_input"] is False
    assert role["ompex_is_tuning_target"] is False
    assert role["ompex_is_selection_criterion"] is False
    alignment = contract["alignment"]
    assert alignment["ompex_native_resolution"] == "PT1H"
    assert alignment["automatic_alignment_selected_on_lowest_error_authorized"] is False
    regimes = contract["cadence_regimes"]
    assert regimes["hourly_truth_may_be_counted_as_four_independent_quarter_hours"] is False
    assert regimes["stepwise_transport_may_be_labelled_native_quarter_hour"] is False
    assert regimes["native_quarter_hour_accuracy_gate"].startswith("UNSUPPORTED")
    rule = contract["decision_rule"]
    assert rule["literal_win_at_every_timestamp_required"] is False
    assert rule["all_integrity_pit_calendar_and_provenance_gates_must_pass"] is True
    assert rule["post_outcome_metric_or_margin_changes_authorized"] is False
