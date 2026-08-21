from __future__ import annotations

import json
from pathlib import Path


def test_external_share_is_explicit_read_only_benchmark_refresh_only() -> None:
    contract_path = Path(
        ".planning/phases/14-lt-audit-remediation/"
        "OMPEX-EXTERNAL-BENCHMARK-ACCESS-CONTRACT-V1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    source = contract["external_source"]
    local = contract["local_execution"]

    assert contract["canonical_workspace"] == r"C:\Users\jbattaglia\PFC_LT"
    assert source["role"] == "OMPEX benchmark source only"
    assert source["explicit_manual_refresh_required"] is True
    assert source["read_only"] is True
    assert source["routine_pipeline_dependency"] is False
    assert source["writes_authorized"] is False
    assert local["all_mutable_inputs_and_outputs_under_canonical_workspace"] is True
    assert local["scoring_may_depend_on_live_H_path"] is False
    assert local["source_path_has_no_application_default"] is True
    assert local["H_unavailable_blocks_routine_PFC_work"] is False
    assert contract["authority"]["model_input"] is False
    assert contract["authority"]["selection"] is False

    runner = Path("scripts/audit_ompex_archive_inventory.py").read_text(
        encoding="utf-8"
    )
    assert "H:\\" not in runner
    assert 'parser.add_argument("--source-dir", type=Path, required=True)' in runner
