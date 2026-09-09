from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest

from pfc_shaping.lt.model.short_tenor_evidence_selection import (
    SELECTION_CONTENT_ID,
    SELECTION_STATUS,
    SELECTED_COMBINATION_CONTENT_ID,
    SELECTED_CONTRAST_CONTENT_ID,
    validate_short_tenor_evidence_manifest_chain,
    validate_short_tenor_evidence_selection,
)
from pfc_shaping.pipeline import production_phases


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_PATH = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "EEX-CH-SHORT-TENOR-EVIDENCE-SELECTION-V1.json"
)
DAY_ROOT = ROOT / "build" / "databricks-eex-daily" / "2026-08-05"


def _document() -> dict[str, object]:
    return json.loads(DOCUMENT_PATH.read_text(encoding="utf-8"))


def _manifest_payloads() -> dict[str, bytes]:
    locations = {
        "309b9f07236d1cfb32b3d92c1ed5413bde6966fa383629882a539ccbd60e9cb5": (
            "short-tenor-contrast-panels"
        ),
        "ae7b962ca5f85ff75223ad45fd578c29b49a1d66dfbdd418b941c337dda38147": (
            "short-tenor-contrast-panels"
        ),
        "7a4c2a601e169d8c471e0939919f12cf56b03d132d8691a82d6bfef5b392bd03": (
            "short-tenor-contrast-panels"
        ),
        "122afd7292549b00a946a2893487d344e459c420a8bbdf99c11714c7308a8cfd": (
            "short-tenor-combination-contract-proofs"
        ),
        "21c557df75260ce162c15af6fbe0c91de4c8a6fcd233564403a04e7b1fc53ad0": (
            "short-tenor-combination-contract-proofs"
        ),
        "c06dfcf5fbda16cd5bab04005d4581e35216116c30c6c9df6cef94f2d949f67b": (
            "short-tenor-combination-contract-proofs"
        ),
    }
    paths = {
        content_id: DAY_ROOT / family / content_id / "manifest.json"
        for content_id, family in locations.items()
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        pytest.skip("local D225 evidence is not installed: " + ", ".join(missing))
    return {content_id: path.read_bytes() for content_id, path in paths.items()}


def test_frozen_selection_document_is_valid_but_non_authoritative() -> None:
    result = validate_short_tenor_evidence_selection(_document())

    assert result["status"] == SELECTION_STATUS
    assert result["selection_content_id"] == SELECTION_CONTENT_ID
    assert result["selected_contrast_content_id"] == SELECTED_CONTRAST_CONTENT_ID
    assert result["selected_combination_content_id"] == (
        SELECTED_COMBINATION_CONTENT_ID
    )
    assert result["superseded_artifact_count"] == 4
    assert result["model_input_authorized"] is False
    assert result["production_authorized"] is False


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        (
            "selected_contrast",
            "content_id",
            "7a4c2a601e169d8c471e0939919f12cf56b03d132d8691a82d6bfef5b392bd03",
            "selected contrast content id",
        ),
        (
            "selected_proof",
            "content_id",
            "c06dfcf5fbda16cd5bab04005d4581e35216116c30c6c9df6cef94f2d949f67b",
            "selected combination content id",
        ),
        (
            "selected_policy",
            "content_id",
            "00a2b2087589d14ef1330cfa0de109fe9dfcce81436e0a0265fb96067d10fbb6",
            "selected policy content id",
        ),
        ("execution", "repo_local_replay_count", 3, "replay count"),
        ("execution", "databricks_request_count", 1, "databricks_request_count"),
        ("authorities", "model_input_authorized", True, "model_input_authorized"),
        (
            "identity",
            "selected_and_superseded_contrast_payloads_identical",
            False,
            "scientific payload equivalence",
        ),
        (
            "identity",
            "selection_used_market_performance",
            True,
            "market-performance selection",
        ),
    ],
)
def test_selection_rejects_stale_identity_cost_or_authority(
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    document = copy.deepcopy(_document())
    if section == "selected_contrast":
        target = document["selected_chain"]["contrast_bundle"]  # type: ignore[index]
    elif section == "selected_proof":
        target = document["selected_chain"]["combination_proof"]  # type: ignore[index]
    elif section == "selected_policy":
        target = document["selected_chain"]["combination_policy"]  # type: ignore[index]
    elif section == "identity":
        target = document["scientific_payload_identity"]  # type: ignore[index]
    else:
        target = document[section]  # type: ignore[index]
    target[field] = value

    with pytest.raises(ValueError, match=message):
        validate_short_tenor_evidence_selection(document)


def test_selection_rejects_incomplete_or_relabelled_supersession() -> None:
    missing = copy.deepcopy(_document())
    missing["superseded_evidence"].pop()  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="exactly four"):
        validate_short_tenor_evidence_selection(missing)

    relabelled = copy.deepcopy(_document())
    relabelled["superseded_evidence"][0][  # type: ignore[index]
        "reason"
    ] = "PERFORMANCE_WORSE"
    with pytest.raises(ValueError, match="superseded contrast reason"):
        validate_short_tenor_evidence_selection(relabelled)


def test_exact_local_manifest_chain_is_verified_without_authority() -> None:
    result = validate_short_tenor_evidence_manifest_chain(
        _document(), _manifest_payloads()
    )

    assert result["manifest_chain_verified"] is True
    assert result["manifest_count"] == 6
    assert result["databricks_request_count"] == 0
    assert result["warehouse_start_count"] == 0
    assert result["model_input_authorized"] is False
    assert result["production_authorized"] is False


def test_manifest_chain_rejects_missing_or_mutated_bytes() -> None:
    payloads = _manifest_payloads()
    missing = dict(payloads)
    missing.pop(SELECTED_COMBINATION_CONTENT_ID)
    with pytest.raises(ValueError, match="manifest identity set"):
        validate_short_tenor_evidence_manifest_chain(_document(), missing)

    mutated = dict(payloads)
    mutated[SELECTED_CONTRAST_CONTENT_ID] += b" "
    with pytest.raises(ValueError, match="manifest hash"):
        validate_short_tenor_evidence_manifest_chain(_document(), mutated)


def test_selection_module_is_not_wired_into_lt_production() -> None:
    source = inspect.getsource(production_phases)

    assert "short_tenor_evidence_selection" not in source
    assert "SELECTED_CONTRAST_CONTENT_ID" not in source
    assert "SELECTED_COMBINATION_CONTENT_ID" not in source
