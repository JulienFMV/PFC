from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.calibration.cascading import count_hours
from scripts.audit_ch_product_normalization import eex_peak_mask, main, run_audit


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


def test_product_normalization_audit_marks_partial_product_unsupported(tmp_path: Path) -> None:
    csv_path = tmp_path / "partial.csv"
    forwards_path = tmp_path / "forwards.parquet"
    _write_delivered_csv(csv_path, start="2027-01-01", end="2027-01-11")
    _write_forwards(forwards_path)

    gates, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )

    assert summary["critical_count"] == 0
    assert summary["covered_hard_gates_pass"] is False
    assert summary["all_gates_pass"] is False
    assert set(gates["status"]) == {"UNSUPPORTED"}


def test_product_normalization_audit_flags_empty_evidence_as_critical(tmp_path: Path) -> None:
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

    gates, summary = run_audit(
        csv_path=csv_path,
        forwards_path=forwards_path,
        required_forward_date="2026-06-22",
    )

    assert summary["critical_count"] == 1
    assert summary["all_gates_pass"] is False
    assert gates.iloc[0]["gate_id"] == "audit_evidence_present"
    assert gates.iloc[0]["status"] == "CRITICAL"


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
