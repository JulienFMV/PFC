from __future__ import annotations

import pandas as pd

from scripts.audit_ntc_baseline_inputs import audit_ntc_inputs


def test_audit_ntc_inputs_reports_available_columns():
    idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    data = {}
    for border in ["de", "fr", "it", "at"]:
        for kind in ["import", "export", "net", "total"]:
            data[f"ntc_{kind}_ch_{border}_mw"] = [1000.0, 1100.0]
    audit = audit_ntc_inputs(pd.DataFrame(data, index=idx))

    assert len(audit) == 16
    assert audit["available"].all()
    assert set(audit["median_mw"]) == {1050.0}


def test_audit_ntc_inputs_reports_empty_columns_as_unusable():
    idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    data = {"ntc_import_ch_de_mw": [pd.NA, pd.NA]}
    audit = audit_ntc_inputs(pd.DataFrame(data, index=idx))

    row = audit[audit["column"] == "ntc_import_ch_de_mw"].iloc[0]
    missing = audit[audit["column"] == "ntc_export_ch_de_mw"].iloc[0]
    assert bool(row["available"]) is False
    assert int(row["non_null_rows"]) == 0
    assert bool(missing["available"]) is False
