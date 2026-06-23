from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.audit_ch_product_normalization import (
    audit,
    eex_peak_mask,
    main,
    parse_product,
)


PRICE = "price_weighted_mean_eur_mwh"


def _index_for(product_key: str) -> pd.DatetimeIndex:
    product = parse_product(product_key)
    assert product is not None
    return product.expected_utc_index


def _write_curve(tmp_path: Path, index: pd.DatetimeIndex, prices) -> Path:
    frame = pd.DataFrame(
        {
            "timestamp_utc": [ts.isoformat() for ts in index],
            PRICE: prices,
        }
    )
    path = tmp_path / "curve.csv"
    frame.to_csv(path, index=False)
    return path


def _write_forwards(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "forwards.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text('{"artifact":"synthetic-test-fixture"}\n', encoding="utf-8")
    return path


def _base_peak_curve(index: pd.DatetimeIndex, *, base: float, peak: float) -> list[float]:
    mask = eex_peak_mask(index, country="CH")
    peak_hours = int(mask.sum())
    offpeak_hours = len(index) - peak_hours
    offpeak = (base * len(index) - peak * peak_hours) / offpeak_hours
    return [peak if is_peak else offpeak for is_peak in mask]


def _calendar_with_q1_curve(index: pd.DatetimeIndex, *, calendar: float, q1_value: float) -> list[float]:
    q1 = parse_product("2028-Q1")
    assert q1 is not None
    q1_mask = (index >= q1.start_utc) & (index < q1.end_utc)
    peak = eex_peak_mask(index, country="CH")
    q1_peak_hours = int((q1_mask & peak).sum())
    q1_offpeak_hours = int((q1_mask & ~peak).sum())
    year_peak_hours = int(peak.sum())
    year_offpeak_hours = len(index) - year_peak_hours
    residual_peak = (calendar * year_peak_hours - q1_value * q1_peak_hours) / (year_peak_hours - q1_peak_hours)
    residual_offpeak = (calendar * year_offpeak_hours - q1_value * q1_offpeak_hours) / (year_offpeak_hours - q1_offpeak_hours)
    return [
        q1_value if in_q1 else (residual_peak if is_peak else residual_offpeak)
        for in_q1, is_peak in zip(q1_mask, peak)
    ]


def _jan_base_peak_forwards(base: float = 100.0, peak: float = 120.0) -> list[dict[str, object]]:
    return [
        {"market": "CH", "load_type": "BASE", "date": "2026-06-19", "product": "2028-01", "price": base},
        {"market": "CH", "load_type": "PEAK", "date": "2026-06-19", "product": "2028-01", "price": peak},
    ]


def _statuses(result) -> set[str]:
    return set(result["gates"]["status"].astype(str))


def test_product_windows_have_fixed_dst_and_leap_hour_counts():
    feb = parse_product("2028-02")
    march = parse_product("2028-03")
    october = parse_product("2028-10")
    year = parse_product("2028")
    assert feb is not None and march is not None and october is not None and year is not None

    assert len(feb.expected_utc_index) == 29 * 24
    assert len(march.expected_utc_index) == 31 * 24 - 1
    assert len(october.expected_utc_index) == 31 * 24 + 1
    assert len(year.expected_utc_index) == 366 * 24
    assert march.start_utc == pd.Timestamp("2028-02-29 23:00:00+00:00")
    assert march.end_utc == pd.Timestamp("2028-03-31 22:00:00+00:00")
    assert october.start_utc == pd.Timestamp("2028-09-30 22:00:00+00:00")
    assert october.end_utc == pd.Timestamp("2028-10-31 23:00:00+00:00")


def test_eex_peak_mask_excludes_ch_holiday_and_respects_local_hours():
    local = pd.DatetimeIndex(
        [
            "2024-08-01 10:00",  # CH National Day, Thursday.
            "2024-08-02 07:00",
            "2024-08-02 08:00",
            "2024-08-02 19:00",
            "2024-08-02 20:00",
        ],
        tz="Europe/Zurich",
    )
    mask = eex_peak_mask(local.tz_convert("UTC"), country="CH")

    assert mask.tolist() == [False, False, True, True, False]


def test_pass_complete_base_peak_offpeak(tmp_path):
    index = _index_for("2028-01")
    csv = _write_curve(tmp_path, index, _base_peak_curve(index, base=100.0, peak=120.0))
    forwards = _write_forwards(tmp_path, _jan_base_peak_forwards())

    result = audit(csv, forwards, tolerance_eur_mwh=1e-9, manifest_path=_write_manifest(tmp_path))

    assert "CRITICAL" not in _statuses(result)
    assert "UNSUPPORTED" not in _statuses(result)
    gates = result["gates"]
    offpeak = gates[(gates["gate"] == "direct_quote_mean") & (gates["load_type"] == "OFFPEAK")]
    assert offpeak["status"].iloc[0] == "PASS"


def test_peak_broken_is_critical(tmp_path):
    index = _index_for("2028-01")
    prices = _base_peak_curve(index, base=100.0, peak=121.0)
    csv = _write_curve(tmp_path, index, prices)
    forwards = _write_forwards(tmp_path, _jan_base_peak_forwards(base=100.0, peak=120.0))

    result = audit(csv, forwards, tolerance_eur_mwh=1e-9, manifest_path=_write_manifest(tmp_path))

    peak = result["gates"][
        (result["gates"]["gate"] == "direct_quote_mean")
        & (result["gates"]["load_type"] == "PEAK")
        & (result["gates"]["product"] == "2028-01")
    ]
    assert peak["status"].iloc[0] == "CRITICAL"


def test_parent_direct_quote_failure_is_not_hidden_by_residual_bucket(tmp_path):
    index = _index_for("2028")
    q1 = parse_product("2028-Q1")
    assert q1 is not None
    q1_mask = (index >= q1.start_utc) & (index < q1.end_utc)
    q1_hours = int(q1_mask.sum())
    year_hours = len(index)
    residual_target = (100.0 * year_hours - 90.0 * q1_hours) / (year_hours - q1_hours)
    prices = [110.0 if in_q1 else residual_target for in_q1 in q1_mask]
    csv = _write_curve(tmp_path, index, prices)
    forwards = _write_forwards(
        tmp_path,
        [
            {"market": "CH", "load_type": "BASE", "date": "2026-06-19", "product": "2028", "price": 100.0},
            {"market": "CH", "load_type": "BASE", "date": "2026-06-19", "product": "2028-Q1", "price": 90.0},
            {"market": "CH", "load_type": "PEAK", "date": "2026-06-19", "product": "2028", "price": 100.0},
            {"market": "CH", "load_type": "PEAK", "date": "2026-06-19", "product": "2028-Q1", "price": 110.0},
        ],
    )

    result = audit(csv, forwards, tolerance_eur_mwh=1e-9, manifest_path=_write_manifest(tmp_path))

    gates = result["gates"]
    base_residual = gates[
        (gates["gate"] == "quote_aware_bucket_mean")
        & (gates["load_type"] == "BASE")
        & (gates["bucket"] == "2028-RESIDUAL")
    ]
    cal_direct = gates[
        (gates["gate"] == "direct_quote_mean")
        & (gates["load_type"] == "BASE")
        & (gates["product"] == "2028")
    ]
    assert base_residual["status"].iloc[0] == "PASS"
    assert cal_direct["status"].iloc[0] == "CRITICAL"


def test_calendar_plus_q1_reports_residual_bucket(tmp_path):
    index = _index_for("2028")
    prices = _calendar_with_q1_curve(index, calendar=100.0, q1_value=90.0)
    csv = _write_curve(tmp_path, index, prices)
    forwards = _write_forwards(
        tmp_path,
        [
            {"market": "CH", "load_type": "BASE", "date": "2026-06-19", "product": "2028", "price": 100.0},
            {"market": "CH", "load_type": "BASE", "date": "2026-06-19", "product": "2028-Q1", "price": 90.0},
            {"market": "CH", "load_type": "PEAK", "date": "2026-06-19", "product": "2028", "price": 100.0},
            {"market": "CH", "load_type": "PEAK", "date": "2026-06-19", "product": "2028-Q1", "price": 90.0},
        ],
    )

    result = audit(csv, forwards, tolerance_eur_mwh=1e-9, manifest_path=_write_manifest(tmp_path))

    gates = result["gates"]
    residual = gates[
        (gates["gate"] == "quote_aware_bucket_mean")
        & (gates["load_type"] == "BASE")
        & (gates["bucket"] == "2028-RESIDUAL")
    ]
    assert residual["status"].iloc[0] == "PASS"
    assert "CRITICAL" not in _statuses(result)
    assert "UNSUPPORTED" not in _statuses(result)


def test_partial_window_is_unsupported(tmp_path):
    index = _index_for("2028-01")[:-1]
    csv = _write_curve(tmp_path, index, _base_peak_curve(index, base=100.0, peak=120.0))
    forwards = _write_forwards(tmp_path, _jan_base_peak_forwards())

    result = audit(csv, forwards, tolerance_eur_mwh=1e-9, manifest_path=_write_manifest(tmp_path))

    partial = result["gates"][result["gates"]["gate"] == "partial_product_window"]
    assert "UNSUPPORTED" in set(partial["status"])
    partial_buckets = result["gates"][result["gates"]["gate"] == "partial_bucket_window"]
    assert "UNSUPPORTED" in set(partial_buckets["status"])
    assert "CRITICAL" not in _statuses(result)


def test_source_quote_parent_child_inconsistency_is_critical(tmp_path):
    index = _index_for("2028-Q1")
    csv = _write_curve(tmp_path, index, [100.0] * len(index))
    forwards = _write_forwards(
        tmp_path,
        [
            {"market": "CH", "load_type": "BASE", "date": "2026-06-19", "product": "2028-01", "price": 100.0},
            {"market": "CH", "load_type": "BASE", "date": "2026-06-19", "product": "2028-02", "price": 100.0},
            {"market": "CH", "load_type": "BASE", "date": "2026-06-19", "product": "2028-03", "price": 100.0},
            {"market": "CH", "load_type": "BASE", "date": "2026-06-19", "product": "2028-Q1", "price": 101.0},
        ],
    )

    result = audit(csv, forwards, tolerance_eur_mwh=1e-9, manifest_path=_write_manifest(tmp_path))

    source = result["gates"][result["gates"]["gate"] == "active_quote_set_parent_dropped"]
    assert source["status"].iloc[0] == "CRITICAL"
    assert source["product"].iloc[0] == "2028-Q1"
    direct_parent = result["gates"][
        (result["gates"]["gate"] == "direct_quote_mean")
        & (result["gates"]["load_type"] == "BASE")
        & (result["gates"]["product"] == "2028-Q1")
    ]
    assert direct_parent.empty
    direct_months = result["gates"][
        (result["gates"]["gate"] == "direct_quote_mean")
        & (result["gates"]["load_type"] == "BASE")
        & (result["gates"]["product"].isin(["2028-01", "2028-02", "2028-03"]))
    ]
    assert set(direct_months["status"]) == {"PASS"}


def test_empty_evidence_is_critical(tmp_path):
    index = _index_for("2028-01")
    csv = _write_curve(tmp_path, index, [100.0] * len(index))
    forwards = _write_forwards(
        tmp_path,
        [
            {"market": "DE", "load_type": "BASE", "date": "2026-06-19", "product": "2028-01", "price": 100.0},
        ],
    )

    result = audit(csv, forwards, tolerance_eur_mwh=1e-9, manifest_path=_write_manifest(tmp_path))

    assert "CRITICAL" in _statuses(result)
    assert "forward_evidence_empty" in set(result["gates"]["gate"])


def test_missing_peak_quote_on_covered_product_is_unsupported(tmp_path):
    index = _index_for("2028-01")
    csv = _write_curve(tmp_path, index, [100.0] * len(index))
    forwards = _write_forwards(
        tmp_path,
        [
            {"market": "CH", "load_type": "BASE", "date": "2026-06-19", "product": "2028-01", "price": 100.0},
        ],
    )

    result = audit(csv, forwards, tolerance_eur_mwh=1e-9, manifest_path=_write_manifest(tmp_path))

    missing = result["gates"][result["gates"]["gate"] == "missing_peak_quote"]
    assert missing["status"].iloc[0] == "UNSUPPORTED"


def test_explicit_csv_window_scoping_excludes_absent_products(tmp_path):
    index = _index_for("2028-01")
    csv = _write_curve(tmp_path, index, _base_peak_curve(index, base=100.0, peak=120.0))
    forwards = _write_forwards(
        tmp_path,
        _jan_base_peak_forwards()
        + [
            {"market": "CH", "load_type": "BASE", "date": "2026-06-19", "product": "2027-12", "price": 90.0},
            {"market": "CH", "load_type": "PEAK", "date": "2026-06-19", "product": "2027-12", "price": 95.0},
        ],
    )
    manifest = _write_manifest(tmp_path)

    unscoped = audit(csv, forwards, tolerance_eur_mwh=1e-9, manifest_path=manifest)
    assert "quoted_product_absent" in set(unscoped["gates"]["gate"])
    assert "UNSUPPORTED" in _statuses(unscoped)

    scoped = audit(csv, forwards, tolerance_eur_mwh=1e-9, manifest_path=manifest, scope_to_csv_window=True)
    out_of_scope = scoped["gates"][scoped["gates"]["gate"] == "out_of_scope_quote"]
    assert len(out_of_scope) == 2
    assert set(out_of_scope["status"]) == {"INFO"}
    assert "CRITICAL" not in _statuses(scoped)
    assert "UNSUPPORTED" not in _statuses(scoped)
    assert scoped["metadata"]["out_of_scope_forward_quote_rows"] == 2


def test_timestamp_utc_mismatch_duplicate_and_gap_are_critical(tmp_path):
    forwards = _write_forwards(tmp_path, _jan_base_peak_forwards())
    index = _index_for("2028-01")[:4]

    naive_csv = tmp_path / "naive.csv"
    pd.DataFrame({"timestamp_utc": [ts.tz_localize(None).isoformat() for ts in index], PRICE: [100.0] * 4}).to_csv(naive_csv, index=False)
    manifest = _write_manifest(tmp_path)
    naive = audit(naive_csv, forwards, tolerance_eur_mwh=1e-9, manifest_path=manifest)
    assert "timestamp_utc_timezone" in set(naive["gates"]["gate"])
    assert "CRITICAL" in _statuses(naive)

    duplicate_csv = _write_curve(tmp_path, pd.DatetimeIndex([index[0], index[1], index[1], index[2]]), [100.0] * 4)
    duplicate = audit(duplicate_csv, forwards, tolerance_eur_mwh=1e-9, manifest_path=manifest)
    assert "timestamp_duplicate" in set(duplicate["gates"]["gate"])

    gap_csv = _write_curve(tmp_path, pd.DatetimeIndex([index[0], index[1], index[3]]), [100.0] * 3)
    gap = audit(gap_csv, forwards, tolerance_eur_mwh=1e-9, manifest_path=manifest)
    assert "timestamp_hourly_frequency" in set(gap["gates"]["gate"])


def test_duplicate_conflicting_quote_is_critical(tmp_path):
    index = _index_for("2028-01")
    csv = _write_curve(tmp_path, index, _base_peak_curve(index, base=100.0, peak=120.0))
    forwards = _write_forwards(
        tmp_path,
        _jan_base_peak_forwards()
        + [
            {"market": "CH", "load_type": "PEAK", "date": "2026-06-19", "product": "2028-01", "price": 121.0},
        ],
    )

    result = audit(csv, forwards, tolerance_eur_mwh=1e-9, manifest_path=_write_manifest(tmp_path))

    conflicts = result["gates"][result["gates"]["gate"] == "duplicate_conflicting_quote"]
    assert conflicts["status"].iloc[0] == "CRITICAL"


def test_cli_fail_closed_writes_outputs_then_returns_nonzero(tmp_path):
    index = _index_for("2028-01")
    csv = _write_curve(tmp_path, index, _base_peak_curve(index, base=100.0, peak=121.0))
    forwards = _write_forwards(tmp_path, _jan_base_peak_forwards(base=100.0, peak=120.0))
    report = tmp_path / "report.md"
    gates = tmp_path / "gates.csv"
    json_output = tmp_path / "report.json"
    manifest = _write_manifest(tmp_path)

    code = main(
        [
            "--csv",
            str(csv),
            "--forwards",
            str(forwards),
            "--manifest",
            str(manifest),
            "--tolerance-eur-mwh",
            "0.000000001",
            "--report",
            str(report),
            "--gates-output",
            str(gates),
            "--json-output",
            str(json_output),
        ]
    )

    assert code == 2
    assert report.exists()
    assert gates.exists()
    assert json_output.exists()
    written = pd.read_csv(gates)
    assert "CRITICAL" in set(written["status"])


def test_cli_missing_forwards_writes_outputs_then_returns_nonzero(tmp_path):
    index = _index_for("2028-01")
    csv = _write_curve(tmp_path, index, [100.0] * len(index))
    missing_forwards = tmp_path / "missing_forwards.csv"
    report = tmp_path / "missing_report.md"
    gates = tmp_path / "missing_gates.csv"
    json_output = tmp_path / "missing_report.json"
    manifest = _write_manifest(tmp_path)

    code = main(
        [
            "--csv",
            str(csv),
            "--forwards",
            str(missing_forwards),
            "--manifest",
            str(manifest),
            "--tolerance-eur-mwh",
            "0.000001",
            "--report",
            str(report),
            "--gates-output",
            str(gates),
            "--json-output",
            str(json_output),
        ]
    )

    assert code == 2
    assert report.exists()
    assert gates.exists()
    assert json_output.exists()
    written = pd.read_csv(gates)
    assert "forwards_missing" in set(written["gate"])
    assert "CRITICAL" in set(written["status"])


def test_missing_manifest_is_critical(tmp_path):
    index = _index_for("2028-01")
    csv = _write_curve(tmp_path, index, _base_peak_curve(index, base=100.0, peak=120.0))
    forwards = _write_forwards(tmp_path, _jan_base_peak_forwards())

    result = audit(csv, forwards, tolerance_eur_mwh=1e-9)

    missing = result["gates"][result["gates"]["gate"] == "manifest_missing"]
    assert missing["status"].iloc[0] == "CRITICAL"
