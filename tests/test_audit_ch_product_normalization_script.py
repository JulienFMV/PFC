from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.calibration.cascading import count_hours
from pfc_shaping.data.forward_proxy import ForwardEligibility, ForwardSnapshot
from pfc_shaping.data.ingest_forwards import load_base_prices_from_eex_report_bytes
from scripts.audit_ch_product_normalization import (
    _sha256_file,
    eex_peak_mask,
    load_forward_snapshot,
    load_source_hierarchy_policy,
    main,
    run_audit,
)


@pytest.mark.parametrize(
    ("suffix", "raw", "message"),
    [
        (
            ".json",
            '{"production_approved":false,"production_approved":true}',
            "duplicate JSON key",
        ),
        (
            ".yaml",
            (
                "defaults: &defaults\n"
                "  production_approved: false\n"
                "policy:\n"
                "  <<: *defaults\n"
                "  production_approved: true\n"
            ),
            "duplicate YAML key",
        ),
    ],
)
def test_source_hierarchy_policy_rejects_ambiguous_structured_bytes(
    tmp_path: Path,
    suffix: str,
    raw: str,
    message: str,
) -> None:
    policy = tmp_path / f"policy{suffix}"
    policy.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_source_hierarchy_policy(policy)


def test_eex_peak_mask_matches_contractual_2027_hour_count() -> None:
    timestamps = pd.Series(
        pd.date_range(
            "2027-01-01T00:00:00",
            "2028-01-01T00:00:00",
            freq="1h",
            inclusive="left",
            tz="Europe/Zurich",
        )
    )

    assert int(eex_peak_mask(timestamps, country="CH").sum()) == 3132


LOCAL_TZ = "Europe/Zurich"
PRICE = "price_weighted_mean_eur_mwh"


def _hourly_index(start: str, end: str) -> pd.DatetimeIndex:
    start_utc = pd.Timestamp(start, tz=LOCAL_TZ).tz_convert("UTC")
    end_utc = pd.Timestamp(end, tz=LOCAL_TZ).tz_convert("UTC")
    return pd.date_range(start_utc, end_utc, freq="h", inclusive="left", tz="UTC")


def _write_delivered_csv(
    path: Path,
    *,
    start: str = "2027-01-01",
    end: str = "2027-02-01",
    base_target: float = 100.0,
    peak_target: float = 120.0,
    peak_delta: float = 0.0,
) -> pd.DataFrame:
    idx_utc = _hourly_index(start, end)
    ts_ch = idx_utc.tz_convert(LOCAL_TZ)
    peak_mask = eex_peak_mask(pd.Series(ts_ch), country="CH").to_numpy(dtype=bool)
    total_hours = len(idx_utc)
    peak_hours = int(peak_mask.sum())
    offpeak_hours = total_hours - peak_hours
    offpeak_target = (base_target * total_hours - peak_target * peak_hours) / offpeak_hours
    price = np.where(peak_mask, peak_target + peak_delta, offpeak_target)
    offset = pd.Index(ts_ch.strftime("%z"))
    frame = pd.DataFrame(
        {
            "timestamp_ch": ts_ch.strftime("%d.%m.%Y %H:%M"),
            "utc_offset_ch": "UTC" + offset.str.slice(0, 3) + ":" + offset.str.slice(3, 5),
            "timestamp_utc": idx_utc.strftime("%d.%m.%Y %H:%M"),
            PRICE: price,
        }
    )
    frame.to_csv(path, index=False)
    return frame


def _write_forwards(path: Path, *, date: str = "2026-06-22") -> None:
    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(date),
                "product": "2027-01",
                "load_type": "BASE",
                "product_type": "Month",
                "price": 100.0,
                "market": "CH",
                "source": "synthetic-test",
            },
            {
                "date": pd.Timestamp(date),
                "product": "2027-01",
                "load_type": "PEAK",
                "product_type": "Month",
                "price": 120.0,
                "market": "CH",
                "source": "synthetic-test",
            },
        ]
    )
    frame.to_parquet(path)


def _write_monthly_candidate(path: Path, forwards_path: Path) -> None:
    forward_date, snapshot = load_forward_snapshot(
        forwards_path,
        market="CH",
        required_forward_date="2026-06-22",
    )
    assert forward_date == pd.Timestamp("2026-06-22")
    report_path = path.with_suffix(".xlsx")
    workbook_rows = [
        [None, "M01_2027_BASE", "M01_2027_PEAK"],
        [None, "ISIN-BASE", "ISIN-PEAK"],
        ["Date", None, None],
        ["22.06.2026", 100.0, 120.0],
    ]
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        pd.DataFrame(workbook_rows).to_excel(writer, sheet_name="CH", index=False, header=False)
    report_bytes = report_path.read_bytes()
    prices, parsed_date, metadata = load_base_prices_from_eex_report_bytes(
        report_bytes,
        market="CH",
        source_label=str(report_path.resolve()),
        return_snapshot_metadata=True,
    )
    raw_snapshot = ForwardSnapshot(
        prices=prices,
        market="CH",
        source_kind="EEX_XLSX",
        source_description="synthetic manifest identity fixture",
        snapshot_date=parsed_date,
        available_at="2026-06-22T12:00:00Z",
        source_path=str(report_path.resolve()),
        source_sha256=hashlib.sha256(report_bytes).hexdigest(),
        quote_lineage=metadata["quote_lineage"],
        source_sheet="CH",
    )
    snapshot_manifest = raw_snapshot.to_manifest()
    snapshot_manifest["hard_quote_eligible"] = True
    eligibility = ForwardEligibility(
        snapshot_id=raw_snapshot.snapshot_id,
        observation_id=raw_snapshot.observation_id,
        source_sha256=str(raw_snapshot.source_sha256),
        quote_set_sha256=raw_snapshot.quote_set_sha256,
        quote_lineage_sha256=raw_snapshot.quote_lineage_sha256,
        reference_timestamp=pd.Timestamp("2026-06-22T20:00:00Z"),
        available_at=pd.Timestamp("2026-06-22T12:00:00Z"),
        business_age_days=0,
        max_age_business_days=1,
    )
    payload = {
        "promotion_eligible": True,
        "forward_snapshot": snapshot_manifest,
        "forward_eligibility": eligibility.to_manifest(),
        "monthly_solution_hash": "solution-hash",
        "active_constraints_hash": "constraints-hash",
        "active_config_hash": "config-hash",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_forwards_rows(path: Path, rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows)
    if "source" not in frame.columns:
        frame["source"] = "synthetic-test"
    frame.to_parquet(path)


def _quote_row(product: str, load_type: str, price: float, *, date: str = "2026-06-22") -> dict[str, object]:
    product_type = "Cal"
    if "-Q" in product:
        product_type = "Quarter"
    elif "-" in product:
        product_type = "Month"
    return {
        "date": pd.Timestamp(date),
        "product": product,
        "load_type": load_type,
        "product_type": product_type,
        "price": price,
        "market": "CH",
        "source": "synthetic-test",
    }


def _write_2028_cal_q1_curve(csv_path: Path, forwards_path: Path) -> None:
    idx_utc = _hourly_index("2028-01-01", "2029-01-01")
    ts_ch = idx_utc.tz_convert(LOCAL_TZ)
    ts_series = pd.Series(ts_ch)
    q1_mask = ts_series.dt.month.le(3).to_numpy(dtype=bool)
    peak_mask = eex_peak_mask(ts_series, country="CH").to_numpy(dtype=bool)

    total_year, peak_year, offpeak_year = count_hours(2028, 1, 12, tz=LOCAL_TZ, country="CH")
    total_q1, peak_q1_hours, offpeak_q1_hours = count_hours(2028, 1, 3, tz=LOCAL_TZ, country="CH")
    peak_residual = peak_year - peak_q1_hours
    offpeak_residual = offpeak_year - offpeak_q1_hours

    base_cal = 100.0
    base_q1 = 120.0
    peak_cal = 110.0
    peak_q1_target = 130.0

    q1_peak_price = peak_q1_target
    q1_offpeak_price = (base_q1 * total_q1 - q1_peak_price * peak_q1_hours) / offpeak_q1_hours
    residual_peak_price = (peak_cal * peak_year - q1_peak_price * peak_q1_hours) / peak_residual
    residual_offpeak_price = (
        base_cal * total_year
        - base_q1 * total_q1
        - residual_peak_price * peak_residual
    ) / offpeak_residual

    price = np.empty(len(idx_utc), dtype=float)
    price[q1_mask & peak_mask] = q1_peak_price
    price[q1_mask & ~peak_mask] = q1_offpeak_price
    price[~q1_mask & peak_mask] = residual_peak_price
    price[~q1_mask & ~peak_mask] = residual_offpeak_price

    offset = pd.Index(ts_ch.strftime("%z"))
    frame = pd.DataFrame(
        {
            "timestamp_ch": ts_ch.strftime("%d.%m.%Y %H:%M"),
            "utc_offset_ch": "UTC" + offset.str.slice(0, 3) + ":" + offset.str.slice(3, 5),
            "timestamp_utc": idx_utc.strftime("%d.%m.%Y %H:%M"),
            PRICE: price,
        }
    )
    frame.to_csv(csv_path, index=False)
    _write_forwards_rows(
        forwards_path,
        [
            _quote_row("2028", "BASE", base_cal),
            _quote_row("2028-Q1", "BASE", base_q1),
            _quote_row("2028", "PEAK", peak_cal),
            _quote_row("2028-Q1", "PEAK", peak_q1_target),
        ],
    )


def _write_2027_q3_redundant_conflict_curve(
    csv_path: Path,
    forwards_path: Path,
    *,
    drift_monthly_bucket: bool = False,
) -> None:
    idx_utc = _hourly_index("2027-07-01", "2027-10-01")
    ts_ch = idx_utc.tz_convert(LOCAL_TZ)
    ts_series = pd.Series(ts_ch)
    peak_mask = eex_peak_mask(ts_series, country="CH").to_numpy(dtype=bool)
    base_by_month = {7: 100.0, 8: 110.0, 9: 120.0}
    peak_by_month = {7: 130.0, 8: 140.0, 9: 150.0}
    price = np.empty(len(idx_utc), dtype=float)

    for month in (7, 8, 9):
        total_hours, peak_hours, offpeak_hours = count_hours(
            2027,
            month,
            month,
            tz=LOCAL_TZ,
            country="CH",
        )
        offpeak = (
            base_by_month[month] * total_hours
            - peak_by_month[month] * peak_hours
        ) / offpeak_hours
        month_mask = ts_series.dt.month.eq(month).to_numpy(dtype=bool)
        price[month_mask & peak_mask] = peak_by_month[month]
        price[month_mask & ~peak_mask] = offpeak

    if drift_monthly_bucket:
        price[ts_series.dt.month.eq(7).to_numpy(dtype=bool)] += 5.0

    offset = pd.Index(ts_ch.strftime("%z"))
    frame = pd.DataFrame(
        {
            "timestamp_ch": ts_ch.strftime("%d.%m.%Y %H:%M"),
            "utc_offset_ch": "UTC" + offset.str.slice(0, 3) + ":" + offset.str.slice(3, 5),
            "timestamp_utc": idx_utc.strftime("%d.%m.%Y %H:%M"),
            PRICE: price,
        }
    )
    frame.to_csv(csv_path, index=False)

    total_q3, peak_q3, _offpeak_q3 = count_hours(2027, 7, 9, tz=LOCAL_TZ, country="CH")
    month_hours = {
        month: count_hours(2027, month, month, tz=LOCAL_TZ, country="CH")
        for month in (7, 8, 9)
    }
    implied_base_q3 = sum(
        base_by_month[month] * month_hours[month][0]
        for month in (7, 8, 9)
    ) / total_q3
    implied_peak_q3 = sum(
        peak_by_month[month] * month_hours[month][1]
        for month in (7, 8, 9)
    ) / peak_q3
    rows = []
    for month in (7, 8, 9):
        rows.append(_quote_row(f"2027-{month:02d}", "BASE", base_by_month[month]))
        rows.append(_quote_row(f"2027-{month:02d}", "PEAK", peak_by_month[month]))
    rows.append(_quote_row("2027-Q3", "BASE", implied_base_q3 + 1.0))
    rows.append(_quote_row("2027-Q3", "PEAK", implied_peak_q3 - 1.0))
    _write_forwards_rows(forwards_path, rows)


def _write_partial_january_full_february_curve(csv_path: Path, forwards_path: Path) -> None:
    idx_utc = _hourly_index("2027-01-15", "2027-03-01")
    ts_ch = idx_utc.tz_convert(LOCAL_TZ)
    ts_series = pd.Series(ts_ch)
    peak_mask = eex_peak_mask(ts_series, country="CH").to_numpy(dtype=bool)
    feb_mask = ts_series.dt.month.eq(2).to_numpy(dtype=bool)

    base_target = 100.0
    peak_target = 120.0
    total_feb, peak_feb, offpeak_feb = count_hours(2027, 2, 2, tz=LOCAL_TZ, country="CH")
    offpeak_target = (base_target * total_feb - peak_target * peak_feb) / offpeak_feb

    price = np.full(len(idx_utc), 99.0, dtype=float)
    price[feb_mask & peak_mask] = peak_target
    price[feb_mask & ~peak_mask] = offpeak_target

    offset = pd.Index(ts_ch.strftime("%z"))
    frame = pd.DataFrame(
        {
            "timestamp_ch": ts_ch.strftime("%d.%m.%Y %H:%M"),
            "utc_offset_ch": "UTC" + offset.str.slice(0, 3) + ":" + offset.str.slice(3, 5),
            "timestamp_utc": idx_utc.strftime("%d.%m.%Y %H:%M"),
            PRICE: price,
        }
    )
    frame.to_csv(csv_path, index=False)
    _write_forwards_rows(
        forwards_path,
        [
            _quote_row("2027-01", "BASE", base_target),
            _quote_row("2027-01", "PEAK", peak_target),
            _quote_row("2027-02", "BASE", base_target),
            _quote_row("2027-02", "PEAK", peak_target),
        ],
    )


def _approved_quote_conflict_policy(
    csv_path: Path,
    forwards_path: Path,
    initial_summary: dict[str, object],
    *,
    expected_count: int = 3,
    decision: str = "test-only approved hierarchy",
) -> dict[str, object]:
    return {
        "schema_version": "ch_quote_conflict_source_hierarchy_policy.v1",
        "market": "CH",
        "forward_snapshot_date": "2026-06-22",
        "source_hierarchy": "quote_aware_finer_buckets_over_redundant_parent",
        "accept_quote_conflict": True,
        "expected_quote_conflict_count": expected_count,
        "input_csv_sha256": _sha256_file(csv_path),
        "forwards_sha256": _sha256_file(forwards_path),
        "quote_conflict_identity_hash": initial_summary["quote_conflict_identity_hash"],
        "production_approved": True,
        "decision": decision,
    }


def test_product_normalization_audit_passes_base_peak_and_implied_offpeak(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered.csv"
    forwards_path = tmp_path / "forwards.parquet"
    _write_delivered_csv(csv_path)
    _write_forwards(forwards_path)

    gates, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )

    assert summary["critical_count"] == 0
    assert set(gates["status"]) == {"PASS"}
    direct = gates.set_index(["gate_id", "load_type", "product"])
    assert direct.loc[("hard_base_product_repricing", "BASE", "2027-01"), "abs_residual_eur_mwh"] <= 1e-9
    assert direct.loc[("hard_peak_product_repricing", "PEAK", "2027-01"), "abs_residual_eur_mwh"] <= 1e-9
    assert direct.loc[("implied_offpeak_identity", "OFFPEAK", "2027-01"), "abs_residual_eur_mwh"] <= 1e-9
    total_hours, peak_hours, offpeak_hours = count_hours(2027, 1, 1, tz=LOCAL_TZ, country="CH")
    assert direct.loc[("hard_base_product_repricing", "BASE", "2027-01"), "rows"] == total_hours
    assert direct.loc[("hard_peak_product_repricing", "PEAK", "2027-01"), "rows"] == peak_hours
    assert direct.loc[("implied_offpeak_identity", "OFFPEAK", "2027-01"), "rows"] == offpeak_hours


def test_product_audit_binds_exact_monthly_candidate_forward_snapshot(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered.csv"
    forwards_path = tmp_path / "forwards.parquet"
    candidate_path = tmp_path / "monthly_candidate.json"
    _write_delivered_csv(csv_path)
    _write_forwards(forwards_path)
    _write_monthly_candidate(candidate_path, forwards_path)

    _, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
        monthly_candidate_manifest_path=candidate_path,
    )

    assert summary["monthly_candidate_binding_status"] == "BOUND"
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert summary["forward_snapshot_id"] == payload["forward_snapshot"]["snapshot_id"]
    assert summary["forward_eligibility_id"] == payload["forward_eligibility"]["eligibility_id"]
    assert summary["monthly_candidate_manifest_sha256"] == _sha256_file(candidate_path)


def test_product_audit_rejects_different_forward_quote_set(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered.csv"
    forwards_path = tmp_path / "forwards.parquet"
    candidate_path = tmp_path / "monthly_candidate.json"
    _write_delivered_csv(csv_path)
    _write_forwards(forwards_path)
    _write_monthly_candidate(candidate_path, forwards_path)
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    payload["forward_snapshot"]["quote_set_sha256"] = "different"
    candidate_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="quote_set_sha256"):
        run_audit(
            csv_path=csv_path,
            forwards_path=forwards_path,
            required_forward_date="2026-06-22",
            monthly_candidate_manifest_path=candidate_path,
        )


def test_product_audit_rejects_substituted_forward_eligibility(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered.csv"
    forwards_path = tmp_path / "forwards.parquet"
    candidate_path = tmp_path / "monthly_candidate.json"
    _write_delivered_csv(csv_path)
    _write_forwards(forwards_path)
    _write_monthly_candidate(candidate_path, forwards_path)
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    payload["forward_eligibility"]["observation_id"] = "0" * 64
    candidate_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="observation_id mismatch"):
        run_audit(
            csv_path=csv_path,
            forwards_path=forwards_path,
            required_forward_date="2026-06-22",
            monthly_candidate_manifest_path=candidate_path,
        )


def test_product_normalization_audit_flags_peak_repricing_break(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered.csv"
    forwards_path = tmp_path / "forwards.parquet"
    _write_delivered_csv(csv_path, peak_delta=5.0)
    _write_forwards(forwards_path)

    gates, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )

    peak_rows = gates[
        (gates["gate_id"] == "hard_peak_product_repricing")
        & (gates["load_type"] == "PEAK")
        & (gates["product"] == "2027-01")
    ]
    assert summary["critical_count"] >= 1
    assert peak_rows.iloc[0]["status"] == "CRITICAL"
    assert peak_rows.iloc[0]["abs_residual_eur_mwh"] == pytest.approx(5.0)


def test_product_normalization_audit_checks_quote_aware_residual_buckets(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered_2028.csv"
    forwards_path = tmp_path / "forwards_2028.parquet"
    _write_2028_cal_q1_curve(csv_path, forwards_path)

    gates, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )

    assert summary["all_gates_pass"] is True
    indexed = gates.set_index(["gate_id", "load_type", "product"])
    assert indexed.loc[("quote_aware_base_bucket_repricing", "BASE", "2028-RESIDUAL"), "status"] == "PASS"
    assert indexed.loc[("quote_aware_peak_bucket_repricing", "PEAK", "2028-RESIDUAL"), "status"] == "PASS"
    assert indexed.loc[("hard_base_product_repricing", "BASE", "2028"), "status"] == "PASS"
    assert indexed.loc[("hard_peak_product_repricing", "PEAK", "2028"), "status"] == "PASS"


def test_product_normalization_audit_reclassifies_redundant_parent_quote_conflict(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered_q3.csv"
    forwards_path = tmp_path / "forwards_q3.parquet"
    _write_2027_q3_redundant_conflict_curve(csv_path, forwards_path)

    gates, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )

    indexed = gates.set_index(["gate_id", "load_type", "product"])
    assert indexed.loc[("hard_base_product_repricing", "BASE", "2027-Q3"), "status"] == "QUOTE_CONFLICT"
    assert indexed.loc[("hard_peak_product_repricing", "PEAK", "2027-Q3"), "status"] == "QUOTE_CONFLICT"
    assert indexed.loc[("implied_offpeak_identity", "OFFPEAK", "2027-Q3"), "status"] == "QUOTE_CONFLICT"
    assert indexed.loc[("quote_aware_base_bucket_repricing", "BASE", "2027-07"), "status"] == "PASS"
    assert indexed.loc[("quote_aware_peak_bucket_repricing", "PEAK", "2027-07"), "status"] == "PASS"
    assert summary["critical_count"] == 0
    assert summary["quote_conflict_count"] == 3
    assert summary["delivered_curve_drift_count"] == 0
    assert summary["all_gates_pass"] is False


def test_product_normalization_source_hierarchy_policy_requires_production_approval(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered_q3.csv"
    forwards_path = tmp_path / "forwards_q3.parquet"
    policy_path = tmp_path / "policy.json"
    _write_2027_q3_redundant_conflict_curve(csv_path, forwards_path)
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "ch_quote_conflict_source_hierarchy_policy.v1",
                "market": "CH",
                "forward_snapshot_date": "2026-06-22",
                "source_hierarchy": "quote_aware_finer_buckets_over_redundant_parent",
                "accept_quote_conflict": True,
                "expected_quote_conflict_count": 3,
                "production_approved": False,
                "decision": "draft policy only",
            }
        ),
        encoding="utf-8",
    )

    _, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
        source_hierarchy_policy_path=policy_path,
    )

    assert summary["quote_conflict_count"] == 3
    assert summary["accepted_quote_conflict_count"] == 0
    assert summary["blocking_quote_conflict_count"] == 3
    assert summary["source_hierarchy_policy"]["status"] == "VALID_NOT_PRODUCTION_APPROVED"
    assert summary["all_gates_pass"] is False


def test_product_normalization_source_hierarchy_policy_can_accept_quote_conflicts(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered_q3.csv"
    forwards_path = tmp_path / "forwards_q3.parquet"
    policy_path = tmp_path / "policy.json"
    _write_2027_q3_redundant_conflict_curve(csv_path, forwards_path)
    _, initial_summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "ch_quote_conflict_source_hierarchy_policy.v1",
                "market": "CH",
                "forward_snapshot_date": "2026-06-22",
                "source_hierarchy": "quote_aware_finer_buckets_over_redundant_parent",
                "accept_quote_conflict": True,
                "expected_quote_conflict_count": 3,
                "input_csv_sha256": _sha256_file(csv_path),
                "forwards_sha256": _sha256_file(forwards_path),
                "quote_conflict_identity_hash": initial_summary["quote_conflict_identity_hash"],
                "production_approved": True,
                "decision": "test-only approved hierarchy",
            }
        ),
        encoding="utf-8",
    )

    _, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
        source_hierarchy_policy_path=policy_path,
    )

    assert summary["quote_conflict_count"] == 3
    assert summary["accepted_quote_conflict_count"] == 3
    assert summary["blocking_quote_conflict_count"] == 0
    assert summary["source_hierarchy_policy"]["status"] == "ACCEPTED_PRODUCTION_APPROVED"
    assert summary["all_gates_pass"] is True


def test_product_normalization_source_hierarchy_policy_can_accept_by_conflict_identity_hash(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered_q3.csv"
    forwards_path = tmp_path / "forwards_q3.parquet"
    policy_path = tmp_path / "policy.json"
    _write_2027_q3_redundant_conflict_curve(csv_path, forwards_path)
    _, initial_summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "ch_quote_conflict_source_hierarchy_policy.v1",
                "market": "CH",
                "forward_snapshot_date": "2026-06-22",
                "source_hierarchy": "quote_aware_finer_buckets_over_redundant_parent",
                "accept_quote_conflict": True,
                "expected_quote_conflict_count": 3,
                "input_csv_sha256": _sha256_file(csv_path),
                "forwards_sha256": _sha256_file(forwards_path),
                "quote_conflict_identity_hash": initial_summary["quote_conflict_identity_hash"],
                "production_approved": True,
                "decision": "test-only approved hierarchy by conflict identity hash",
            }
        ),
        encoding="utf-8",
    )

    _, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
        source_hierarchy_policy_path=policy_path,
    )

    assert summary["quote_conflict_count"] == 3
    assert summary["accepted_quote_conflict_count"] == 3
    assert summary["blocking_quote_conflict_count"] == 0
    assert summary["source_hierarchy_policy"]["status"] == "ACCEPTED_PRODUCTION_APPROVED"
    assert summary["all_gates_pass"] is True


def test_product_normalization_source_hierarchy_policy_can_accept_by_expected_conflicts(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered_q3.csv"
    forwards_path = tmp_path / "forwards_q3.parquet"
    policy_path = tmp_path / "policy.json"
    _write_2027_q3_redundant_conflict_curve(csv_path, forwards_path)
    _, initial_summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "ch_quote_conflict_source_hierarchy_policy.v1",
                "market": "CH",
                "forward_snapshot_date": "2026-06-22",
                "source_hierarchy": "quote_aware_finer_buckets_over_redundant_parent",
                "accept_quote_conflict": True,
                "expected_quote_conflict_count": 3,
                "input_csv_sha256": _sha256_file(csv_path),
                "forwards_sha256": _sha256_file(forwards_path),
                "expected_quote_conflicts": initial_summary["quote_conflict_identities"],
                "production_approved": True,
                "decision": "test-only approved hierarchy by exact identities",
            }
        ),
        encoding="utf-8",
    )

    _, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
        source_hierarchy_policy_path=policy_path,
    )

    assert summary["accepted_quote_conflict_count"] == 3
    assert summary["blocking_quote_conflict_count"] == 0
    assert summary["source_hierarchy_policy"]["status"] == "ACCEPTED_PRODUCTION_APPROVED"
    assert summary["all_gates_pass"] is True


def test_product_normalization_source_hierarchy_policy_rejects_conflict_identity_mismatch(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered_q3.csv"
    forwards_path = tmp_path / "forwards_q3.parquet"
    policy_path = tmp_path / "policy.json"
    _write_2027_q3_redundant_conflict_curve(csv_path, forwards_path)
    _, initial_summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )
    expected_conflicts = list(initial_summary["quote_conflict_identities"])
    expected_conflicts[0] = {**expected_conflicts[0], "product": "WRONG"}
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "ch_quote_conflict_source_hierarchy_policy.v1",
                "market": "CH",
                "forward_snapshot_date": "2026-06-22",
                "source_hierarchy": "quote_aware_finer_buckets_over_redundant_parent",
                "accept_quote_conflict": True,
                "expected_quote_conflict_count": 3,
                "input_csv_sha256": _sha256_file(csv_path),
                "forwards_sha256": _sha256_file(forwards_path),
                "expected_quote_conflicts": expected_conflicts,
                "production_approved": True,
                "decision": "wrong conflict identities",
            }
        ),
        encoding="utf-8",
    )

    _, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
        source_hierarchy_policy_path=policy_path,
    )

    assert summary["accepted_quote_conflict_count"] == 0
    assert summary["blocking_quote_conflict_count"] == 3
    assert summary["source_hierarchy_policy"]["status"] == "INVALID"
    assert "expected_quote_conflicts_mismatch" in summary["source_hierarchy_policy"]["reason"]
    assert summary["all_gates_pass"] is False


def test_product_normalization_source_hierarchy_policy_rejects_identity_hash_mismatch(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered_q3.csv"
    forwards_path = tmp_path / "forwards_q3.parquet"
    policy_path = tmp_path / "policy.json"
    _write_2027_q3_redundant_conflict_curve(csv_path, forwards_path)
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "ch_quote_conflict_source_hierarchy_policy.v1",
                "market": "CH",
                "forward_snapshot_date": "2026-06-22",
                "source_hierarchy": "quote_aware_finer_buckets_over_redundant_parent",
                "accept_quote_conflict": True,
                "expected_quote_conflict_count": 3,
                "input_csv_sha256": _sha256_file(csv_path),
                "forwards_sha256": _sha256_file(forwards_path),
                "quote_conflict_identity_hash": "0" * 64,
                "production_approved": True,
                "decision": "wrong identity hash",
            }
        ),
        encoding="utf-8",
    )

    _, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
        source_hierarchy_policy_path=policy_path,
    )

    assert summary["accepted_quote_conflict_count"] == 0
    assert summary["blocking_quote_conflict_count"] == 3
    assert summary["source_hierarchy_policy"]["status"] == "INVALID"
    assert "quote_conflict_identity_hash_mismatch" in summary["source_hierarchy_policy"]["reason"]
    assert summary["all_gates_pass"] is False


def test_product_normalization_source_hierarchy_policy_rejects_any_bad_binding(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered_q3.csv"
    forwards_path = tmp_path / "forwards_q3.parquet"
    policy_path = tmp_path / "policy.json"
    _write_2027_q3_redundant_conflict_curve(csv_path, forwards_path)
    _, initial_summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "ch_quote_conflict_source_hierarchy_policy.v1",
                "market": "CH",
                "forward_snapshot_date": "2026-06-22",
                "source_hierarchy": "quote_aware_finer_buckets_over_redundant_parent",
                "accept_quote_conflict": True,
                "expected_quote_conflict_count": 3,
                "input_csv_sha256": _sha256_file(csv_path),
                "forwards_sha256": "0" * 64,
                "quote_conflict_identity_hash": initial_summary["quote_conflict_identity_hash"],
                "production_approved": True,
                "decision": "one good binding and one bad binding",
            }
        ),
        encoding="utf-8",
    )

    _, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
        source_hierarchy_policy_path=policy_path,
    )

    assert summary["accepted_quote_conflict_count"] == 0
    assert summary["blocking_quote_conflict_count"] == 3
    assert summary["source_hierarchy_policy"]["status"] == "INVALID"
    assert "forwards_sha256_mismatch" in summary["source_hierarchy_policy"]["reason"]
    assert summary["all_gates_pass"] is False


def test_product_normalization_source_hierarchy_policy_rejects_noncanonical_expected_conflicts(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "delivered_q3.csv"
    forwards_path = tmp_path / "forwards_q3.parquet"
    policy_path = tmp_path / "policy.json"
    _write_2027_q3_redundant_conflict_curve(csv_path, forwards_path)
    _, initial_summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )
    expected_conflicts = list(initial_summary["quote_conflict_identities"])
    expected_conflicts[0] = {**expected_conflicts[0], "extra": "not-canonical"}
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "ch_quote_conflict_source_hierarchy_policy.v1",
                "market": "CH",
                "forward_snapshot_date": "2026-06-22",
                "source_hierarchy": "quote_aware_finer_buckets_over_redundant_parent",
                "accept_quote_conflict": True,
                "expected_quote_conflict_count": 3,
                "input_csv_sha256": _sha256_file(csv_path),
                "forwards_sha256": _sha256_file(forwards_path),
                "expected_quote_conflicts": expected_conflicts,
                "production_approved": True,
                "decision": "noncanonical conflict identities",
            }
        ),
        encoding="utf-8",
    )

    _, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
        source_hierarchy_policy_path=policy_path,
    )

    assert summary["accepted_quote_conflict_count"] == 0
    assert summary["blocking_quote_conflict_count"] == 3
    assert summary["source_hierarchy_policy"]["status"] == "INVALID"
    assert "expected_quote_conflicts_mismatch" in summary["source_hierarchy_policy"]["reason"]
    assert summary["all_gates_pass"] is False


def test_product_normalization_source_hierarchy_policy_requires_binding_for_prod_approval(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered_q3.csv"
    forwards_path = tmp_path / "forwards_q3.parquet"
    policy_path = tmp_path / "policy.json"
    _write_2027_q3_redundant_conflict_curve(csv_path, forwards_path)
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "ch_quote_conflict_source_hierarchy_policy.v1",
                "market": "CH",
                "forward_snapshot_date": "2026-06-22",
                "source_hierarchy": "quote_aware_finer_buckets_over_redundant_parent",
                "accept_quote_conflict": True,
                "expected_quote_conflict_count": 3,
                "production_approved": True,
                "decision": "missing conflict binding",
            }
        ),
        encoding="utf-8",
    )

    _, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
        source_hierarchy_policy_path=policy_path,
    )

    assert summary["accepted_quote_conflict_count"] == 0
    assert summary["blocking_quote_conflict_count"] == 3
    assert summary["source_hierarchy_policy"]["status"] == "INVALID"
    assert "input_csv_sha256_missing" in summary["source_hierarchy_policy"]["reason"]
    assert "forwards_sha256_missing" in summary["source_hierarchy_policy"]["reason"]
    assert "quote_conflict_identity_binding_missing" in summary["source_hierarchy_policy"]["reason"]
    assert summary["all_gates_pass"] is False


@pytest.mark.parametrize(
    ("missing_key", "expected_reason"),
    [
        ("input_csv_sha256", "input_csv_sha256_missing"),
        ("forwards_sha256", "forwards_sha256_missing"),
        ("quote_conflict_identity_hash", "quote_conflict_identity_binding_missing"),
    ],
)
def test_product_normalization_source_hierarchy_policy_requires_each_prod_binding(
    tmp_path: Path,
    missing_key: str,
    expected_reason: str,
) -> None:
    csv_path = tmp_path / "delivered_q3.csv"
    forwards_path = tmp_path / "forwards_q3.parquet"
    policy_path = tmp_path / "policy.json"
    _write_2027_q3_redundant_conflict_curve(csv_path, forwards_path)
    _, initial_summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )
    policy = _approved_quote_conflict_policy(csv_path, forwards_path, initial_summary)
    policy.pop(missing_key)
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    _, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
        source_hierarchy_policy_path=policy_path,
    )

    assert summary["accepted_quote_conflict_count"] == 0
    assert summary["blocking_quote_conflict_count"] == 3
    assert summary["source_hierarchy_policy"]["status"] == "INVALID"
    assert expected_reason in summary["source_hierarchy_policy"]["reason"]
    assert summary["all_gates_pass"] is False


def test_product_normalization_source_hierarchy_policy_does_not_override_critical(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered.csv"
    forwards_path = tmp_path / "forwards.parquet"
    policy_path = tmp_path / "policy.json"
    _write_delivered_csv(csv_path, peak_delta=5.0)
    _write_forwards(forwards_path)
    _, initial_summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "ch_quote_conflict_source_hierarchy_policy.v1",
                "market": "CH",
                "forward_snapshot_date": "2026-06-22",
                "source_hierarchy": "quote_aware_finer_buckets_over_redundant_parent",
                "accept_quote_conflict": True,
                "expected_quote_conflict_count": 0,
                "input_csv_sha256": _sha256_file(csv_path),
                "forwards_sha256": _sha256_file(forwards_path),
                "quote_conflict_identity_hash": initial_summary["quote_conflict_identity_hash"],
                "production_approved": True,
                "decision": "approved but no critical override",
            }
        ),
        encoding="utf-8",
    )

    _, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
        source_hierarchy_policy_path=policy_path,
    )

    assert summary["source_hierarchy_policy"]["status"] == "ACCEPTED_PRODUCTION_APPROVED"
    assert summary["critical_count"] >= 1
    assert summary["accepted_quote_conflict_count"] == 0
    assert summary["all_gates_pass"] is False


def test_product_normalization_source_hierarchy_policy_rejects_input_csv_hash_mismatch(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered_q3.csv"
    forwards_path = tmp_path / "forwards_q3.parquet"
    policy_path = tmp_path / "policy.json"
    _write_2027_q3_redundant_conflict_curve(csv_path, forwards_path)
    _, initial_summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )
    policy = _approved_quote_conflict_policy(
        csv_path,
        forwards_path,
        initial_summary,
        decision="wrong csv hash",
    )
    policy["input_csv_sha256"] = "0" * 64
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    _, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
        source_hierarchy_policy_path=policy_path,
    )

    assert summary["accepted_quote_conflict_count"] == 0
    assert summary["blocking_quote_conflict_count"] == 3
    assert summary["source_hierarchy_policy"]["status"] == "INVALID"
    assert "input_csv_sha256_mismatch" in summary["source_hierarchy_policy"]["reason"]
    assert summary["all_gates_pass"] is False


def test_product_normalization_source_hierarchy_policy_does_not_override_unsupported(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered.csv"
    forwards_path = tmp_path / "forwards.parquet"
    policy_path = tmp_path / "policy.json"
    _write_delivered_csv(csv_path)
    _write_forwards_rows(forwards_path, [_quote_row("2027-01", "BASE", 100.0)])
    _, initial_summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )
    policy = _approved_quote_conflict_policy(
        csv_path,
        forwards_path,
        initial_summary,
        expected_count=0,
        decision="approved but no unsupported override",
    )
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    _, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
        source_hierarchy_policy_path=policy_path,
    )

    assert summary["source_hierarchy_policy"]["status"] == "ACCEPTED_PRODUCTION_APPROVED"
    assert summary["unsupported_count"] >= 1
    assert summary["accepted_quote_conflict_count"] == 0
    assert summary["all_gates_pass"] is False


def test_product_normalization_source_hierarchy_policy_does_not_mask_out_of_scope_counts(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered_q3.csv"
    forwards_path = tmp_path / "forwards_q3_with_far_quote.parquet"
    policy_path = tmp_path / "policy.json"
    _write_2027_q3_redundant_conflict_curve(csv_path, forwards_path)
    forwards = pd.read_parquet(forwards_path)
    forwards = pd.concat(
        [
            forwards,
            pd.DataFrame(
                [
                    _quote_row("2031", "BASE", 100.0),
                    _quote_row("2031", "PEAK", 120.0),
                ]
            ),
        ],
        ignore_index=True,
    )
    forwards.to_parquet(forwards_path)
    _, initial_summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )
    policy = _approved_quote_conflict_policy(
        csv_path,
        forwards_path,
        initial_summary,
        decision="approved conflict policy with far out-of-scope quote",
    )
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    gates, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
        source_hierarchy_policy_path=policy_path,
    )

    assert summary["source_hierarchy_policy"]["status"] == "ACCEPTED_PRODUCTION_APPROVED"
    assert summary["accepted_quote_conflict_count"] == 3
    assert summary["blocking_quote_conflict_count"] == 0
    assert summary["out_of_scope_count"] == 3
    assert summary["unsupported_count"] == 0
    assert summary["all_gates_pass"] is True
    assert set(gates.loc[gates["status"] == "OUT_OF_SCOPE", "product"]) == {"2031"}


def test_product_normalization_source_hierarchy_policy_rejects_string_booleans(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered_q3.csv"
    forwards_path = tmp_path / "forwards_q3.parquet"
    policy_path = tmp_path / "policy.json"
    _write_2027_q3_redundant_conflict_curve(csv_path, forwards_path)
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "ch_quote_conflict_source_hierarchy_policy.v1",
                "market": "CH",
                "forward_snapshot_date": "2026-06-22",
                "source_hierarchy": "quote_aware_finer_buckets_over_redundant_parent",
                "accept_quote_conflict": "false",
                "expected_quote_conflict_count": 3,
                "production_approved": "false",
                "decision": "badly typed policy",
            }
        ),
        encoding="utf-8",
    )

    _, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
        source_hierarchy_policy_path=policy_path,
    )

    assert summary["accepted_quote_conflict_count"] == 0
    assert summary["blocking_quote_conflict_count"] == 3
    assert summary["source_hierarchy_policy"]["status"] == "INVALID"
    assert "accept_quote_conflict_false" in summary["source_hierarchy_policy"]["reason"]
    assert "production_approved_not_boolean" in summary["source_hierarchy_policy"]["reason"]
    assert summary["all_gates_pass"] is False


def test_product_normalization_source_hierarchy_policy_requires_snapshot_date(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered_q3.csv"
    forwards_path = tmp_path / "forwards_q3.parquet"
    policy_path = tmp_path / "policy.json"
    _write_2027_q3_redundant_conflict_curve(csv_path, forwards_path)
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "ch_quote_conflict_source_hierarchy_policy.v1",
                "market": "CH",
                "source_hierarchy": "quote_aware_finer_buckets_over_redundant_parent",
                "accept_quote_conflict": True,
                "expected_quote_conflict_count": 3,
                "input_csv_sha256": _sha256_file(csv_path),
                "production_approved": True,
                "decision": "missing date",
            }
        ),
        encoding="utf-8",
    )

    _, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
        source_hierarchy_policy_path=policy_path,
    )

    assert summary["accepted_quote_conflict_count"] == 0
    assert summary["blocking_quote_conflict_count"] == 3
    assert summary["source_hierarchy_policy"]["status"] == "INVALID"
    assert "forward_snapshot_date_missing" in summary["source_hierarchy_policy"]["reason"]
    assert summary["all_gates_pass"] is False


def test_product_normalization_source_hierarchy_policy_requires_expected_conflict_count(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered_q3.csv"
    forwards_path = tmp_path / "forwards_q3.parquet"
    policy_path = tmp_path / "policy.json"
    _write_2027_q3_redundant_conflict_curve(csv_path, forwards_path)
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "ch_quote_conflict_source_hierarchy_policy.v1",
                "market": "CH",
                "forward_snapshot_date": "2026-06-22",
                "source_hierarchy": "quote_aware_finer_buckets_over_redundant_parent",
                "accept_quote_conflict": True,
                "expected_quote_conflict_count": 2,
                "input_csv_sha256": _sha256_file(csv_path),
                "production_approved": True,
                "decision": "wrong count",
            }
        ),
        encoding="utf-8",
    )

    _, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
        source_hierarchy_policy_path=policy_path,
    )

    assert summary["accepted_quote_conflict_count"] == 0
    assert summary["blocking_quote_conflict_count"] == 3
    assert summary["source_hierarchy_policy"]["status"] == "INVALID"
    assert "expected_quote_conflict_count_mismatch" in summary["source_hierarchy_policy"]["reason"]
    assert summary["all_gates_pass"] is False


@pytest.mark.parametrize("bad_count", ["3", 3.0, 3.9, True])
def test_product_normalization_source_hierarchy_policy_rejects_non_integer_conflict_count(
    tmp_path: Path,
    bad_count: object,
) -> None:
    csv_path = tmp_path / "delivered_q3.csv"
    forwards_path = tmp_path / "forwards_q3.parquet"
    policy_path = tmp_path / "policy.json"
    _write_2027_q3_redundant_conflict_curve(csv_path, forwards_path)
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "ch_quote_conflict_source_hierarchy_policy.v1",
                "market": "CH",
                "forward_snapshot_date": "2026-06-22",
                "source_hierarchy": "quote_aware_finer_buckets_over_redundant_parent",
                "accept_quote_conflict": True,
                "expected_quote_conflict_count": bad_count,
                "input_csv_sha256": _sha256_file(csv_path),
                "production_approved": True,
                "decision": "bad count type",
            }
        ),
        encoding="utf-8",
    )

    _, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
        source_hierarchy_policy_path=policy_path,
    )

    assert summary["accepted_quote_conflict_count"] == 0
    assert summary["blocking_quote_conflict_count"] == 3
    assert summary["source_hierarchy_policy"]["status"] == "INVALID"
    assert "expected_quote_conflict_count_invalid" in summary["source_hierarchy_policy"]["reason"]
    assert summary["all_gates_pass"] is False


def test_product_normalization_audit_keeps_parent_critical_when_fine_bucket_fails(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered_q3_drift.csv"
    forwards_path = tmp_path / "forwards_q3.parquet"
    _write_2027_q3_redundant_conflict_curve(csv_path, forwards_path, drift_monthly_bucket=True)

    gates, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )

    indexed = gates.set_index(["gate_id", "load_type", "product"])
    assert indexed.loc[("quote_aware_base_bucket_repricing", "BASE", "2027-07"), "status"] == "CRITICAL"
    assert indexed.loc[("hard_base_product_repricing", "BASE", "2027-Q3"), "status"] == "CRITICAL"
    assert summary["critical_count"] >= 1
    assert summary["all_gates_pass"] is False


def test_product_normalization_cli_fails_closed_on_quote_conflict(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered_q3.csv"
    forwards_path = tmp_path / "forwards_q3.parquet"
    gates_path = tmp_path / "gates.csv"
    summary_path = tmp_path / "summary.json"
    _write_2027_q3_redundant_conflict_curve(csv_path, forwards_path)

    exit_code = main(
        [
            "--csv",
            str(csv_path),
            "--forwards",
            str(forwards_path),
            "--required-forward-date",
            "2026-06-22",
            "--output-csv",
            str(gates_path),
            "--summary-json",
            str(summary_path),
        ]
    )

    assert exit_code == 1
    summary = pd.read_json(summary_path, typ="series").to_dict()
    assert summary["quote_conflict_count"] == 3
    assert summary["all_gates_pass"] is False


def test_product_normalization_cli_rejects_unsigned_production_policy(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "delivered_q3.csv"
    forwards_path = tmp_path / "forwards_q3.parquet"
    gates_path = tmp_path / "gates.csv"
    summary_path = tmp_path / "summary.json"
    policy_path = tmp_path / "unsigned-policy.json"
    _write_2027_q3_redundant_conflict_curve(csv_path, forwards_path)
    _, initial = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )
    policy_path.write_text(
        json.dumps(
            _approved_quote_conflict_policy(csv_path, forwards_path, initial)
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="authentically signed"):
        main(
            [
                "--csv",
                str(csv_path),
                "--forwards",
                str(forwards_path),
                "--required-forward-date",
                "2026-06-22",
                "--source-hierarchy-policy",
                str(policy_path),
                "--output-csv",
                str(gates_path),
                "--summary-json",
                str(summary_path),
            ]
        )


def test_product_normalization_audit_marks_in_scope_missing_required_quote_unsupported(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered.csv"
    forwards_path = tmp_path / "forwards.parquet"
    _write_delivered_csv(csv_path)
    _write_forwards_rows(forwards_path, [_quote_row("2027-01", "BASE", 100.0)])

    gates, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )

    assert summary["critical_count"] == 0
    assert summary["covered_hard_gates_pass"] is False
    assert summary["all_gates_pass"] is False
    assert summary["unsupported_count"] >= 1
    assert summary["out_of_scope_count"] == 0
    missing = gates[
        (gates["gate_id"] == "required_forward_quote_present")
        & (gates["load_type"] == "PEAK")
        & (gates["product"] == "2027-01")
    ]
    assert missing.iloc[0]["status"] == "UNSUPPORTED"
    assert missing.iloc[0]["evidence"] == "missing_required_forward_quote"


def test_product_normalization_audit_marks_out_of_scope_products_non_blocking(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered.csv"
    forwards_path = tmp_path / "forwards_with_far_quotes.parquet"
    _write_delivered_csv(csv_path)
    _write_forwards_rows(
        forwards_path,
        [
            _quote_row("2027-01", "BASE", 100.0),
            _quote_row("2027-01", "PEAK", 120.0),
            _quote_row("2031", "BASE", 100.0),
            _quote_row("2031", "PEAK", 120.0),
        ],
    )

    gates, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )

    out_of_scope = gates[gates["status"] == "OUT_OF_SCOPE"]
    assert summary["critical_count"] == 0
    assert summary["unsupported_count"] == 0
    assert summary["out_of_scope_count"] == 3
    assert summary["all_gates_pass"] is True
    assert set(out_of_scope["product"]) == {"2031"}
    assert set(out_of_scope["load_type"]) == {"BASE", "PEAK", "OFFPEAK"}
    assert set(out_of_scope["evidence"]) == {"outside_delivered_artifact_window"}


def test_product_normalization_audit_marks_partial_boundary_month_out_of_scope(tmp_path: Path) -> None:
    csv_path = tmp_path / "partial_boundary.csv"
    forwards_path = tmp_path / "forwards_boundary.parquet"
    _write_partial_january_full_february_curve(csv_path, forwards_path)

    gates, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )

    boundary = gates[gates["product"].eq("2027-01") & gates["status"].eq("OUT_OF_SCOPE")]
    feb_hard = gates[
        gates["product"].eq("2027-02")
        & gates["gate_id"].isin(["hard_base_product_repricing", "hard_peak_product_repricing"])
    ]
    assert summary["critical_count"] == 0
    assert summary["unsupported_count"] == 0
    assert summary["out_of_scope_count"] == 3
    assert summary["all_gates_pass"] is True
    assert set(boundary["load_type"]) == {"BASE", "PEAK", "OFFPEAK"}
    assert set(boundary["evidence"]) == {"outside_delivered_artifact_window"}
    assert set(feb_hard["status"]) == {"PASS"}


def test_product_normalization_audit_fails_when_only_out_of_scope_products_exist(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered.csv"
    forwards_path = tmp_path / "forwards_far_only.parquet"
    _write_delivered_csv(csv_path)
    _write_forwards_rows(
        forwards_path,
        [
            _quote_row("2031", "BASE", 100.0),
            _quote_row("2031", "PEAK", 120.0),
        ],
    )

    gates, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )

    assert summary["critical_count"] == 1
    assert summary["out_of_scope_count"] == 3
    assert summary["all_gates_pass"] is False
    assert "audit_evidence_present" in set(gates["gate_id"])
    assert "no_in_scope_product_gates_emitted" in set(gates["evidence"])


def test_product_normalization_audit_rejects_only_invalid_product_evidence(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered.csv"
    forwards_path = tmp_path / "forwards.parquet"
    _write_delivered_csv(csv_path)
    _write_forwards_rows(
        forwards_path,
        [
            {
                "date": pd.Timestamp("2026-06-22"),
                "product": "NOT-A-PRODUCT",
                "load_type": "BASE",
                "product_type": "Bad",
                "price": 1.0,
                "market": "CH",
                "source": "synthetic-test",
            }
        ],
    )

    with pytest.raises(ValueError, match="unsupported forward product"):
        run_audit(
            csv_path=csv_path,
            forwards_path=forwards_path,
            required_forward_date="2026-06-22",
        )


def test_product_normalization_audit_flags_missing_required_peak_quote(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered.csv"
    forwards_path = tmp_path / "base_only_forwards.parquet"
    _write_delivered_csv(csv_path)
    _write_forwards_rows(forwards_path, [_quote_row("2027-01", "BASE", 100.0)])

    gates, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )

    missing = gates[
        (gates["gate_id"] == "required_forward_quote_present")
        & (gates["load_type"] == "PEAK")
        & (gates["product"] == "2027-01")
    ]
    assert summary["critical_count"] == 0
    assert summary["unsupported_count"] == 1
    assert summary["all_gates_pass"] is False
    assert missing.iloc[0]["status"] == "UNSUPPORTED"
    assert missing.iloc[0]["evidence"] == "missing_required_forward_quote"


def test_product_normalization_audit_rejects_timestamp_utc_mismatch(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad_timestamp.csv"
    forwards_path = tmp_path / "forwards.parquet"
    frame = _write_delivered_csv(csv_path)
    frame.loc[0, "timestamp_utc"] = "01.01.2027 12:00"
    frame.to_csv(csv_path, index=False)
    _write_forwards(forwards_path)

    with pytest.raises(ValueError, match="round-trip"):
        run_audit(
            csv_path=csv_path,
            forwards_path=forwards_path,
            required_forward_date="2026-06-22",
        )


def test_product_normalization_audit_rejects_duplicate_forward_quote(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered.csv"
    forwards_path = tmp_path / "forwards.parquet"
    _write_delivered_csv(csv_path)
    _write_forwards(forwards_path)
    frame = pd.read_parquet(forwards_path)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame.to_parquet(forwards_path)

    with pytest.raises(ValueError, match="duplicate forward quote"):
        run_audit(
            csv_path=csv_path,
            forwards_path=forwards_path,
            required_forward_date="2026-06-22",
        )


@pytest.mark.parametrize("bad_price", [np.nan, np.inf, -np.inf])
def test_product_normalization_rejects_nonfinite_quote_even_out_of_scope(
    tmp_path: Path,
    bad_price: float,
) -> None:
    csv_path = tmp_path / "delivered.csv"
    forwards_path = tmp_path / "forwards.parquet"
    _write_delivered_csv(csv_path)
    _write_forwards(forwards_path)
    frame = pd.read_parquet(forwards_path)
    invalid = frame.iloc[[0]].copy()
    invalid["product"] = "2035"
    invalid["price"] = bad_price
    pd.concat([frame, invalid], ignore_index=True).to_parquet(forwards_path, index=False)

    with pytest.raises(ValueError, match="non-finite"):
        run_audit(
            csv_path=csv_path,
            forwards_path=forwards_path,
            required_forward_date="2026-06-22",
        )


@pytest.mark.parametrize(
    ("product", "load_type", "message"),
    [
        ("NOT-A-PRODUCT", "BASE", "unsupported forward product"),
        ("2027", "UNKNOWN", "unsupported forward load_type"),
    ],
)
def test_product_normalization_rejects_unrecognized_quote_rows(
    tmp_path: Path,
    product: str,
    load_type: str,
    message: str,
) -> None:
    csv_path = tmp_path / "delivered.csv"
    forwards_path = tmp_path / "forwards.parquet"
    _write_delivered_csv(csv_path)
    _write_forwards(forwards_path)
    frame = pd.read_parquet(forwards_path)
    invalid = frame.iloc[[0]].copy()
    invalid["product"] = product
    invalid["load_type"] = load_type
    pd.concat([frame, invalid], ignore_index=True).to_parquet(forwards_path, index=False)

    with pytest.raises(ValueError, match=message):
        run_audit(
            csv_path=csv_path,
            forwards_path=forwards_path,
            required_forward_date="2026-06-22",
        )


def test_product_normalization_cli_is_fail_closed_and_writes_outputs(tmp_path: Path) -> None:
    csv_path = tmp_path / "delivered.csv"
    forwards_path = tmp_path / "forwards.parquet"
    gates_path = tmp_path / "gates.csv"
    summary_path = tmp_path / "summary.json"
    _write_delivered_csv(csv_path, peak_delta=5.0)
    _write_forwards(forwards_path)

    exit_code = main(
        [
            "--csv",
            str(csv_path),
            "--forwards",
            str(forwards_path),
            "--required-forward-date",
            "2026-06-22",
            "--output-csv",
            str(gates_path),
            "--summary-json",
            str(summary_path),
        ]
    )

    assert exit_code == 1
    assert gates_path.exists()
    assert summary_path.exists()
    summary = pd.read_json(summary_path, typ="series").to_dict()
    assert summary["critical_count"] >= 1
    assert "audit_script_sha256" in summary
    assert "command_argv" in summary
