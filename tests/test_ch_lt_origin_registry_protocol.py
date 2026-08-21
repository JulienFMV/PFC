from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import pfc_shaping.cli.audit_ch_lt_origin_registry_protocol as packaged_cli
import pfc_shaping.validation.ch_lt_dependence_power_design as power_design
import pfc_shaping.validation.ch_lt_estimand_contract as estimand
import pfc_shaping.validation.ch_lt_origin_registry_protocol as protocol
import pfc_shaping.validation.ch_lt_origin_target_mask_inventory as inventory
import pfc_shaping.validation.ch_lt_successor_candidate_core_v2 as core_v2
import pfc_shaping.validation.ch_lt_successor_candidate_core_v3 as core_v3
import pfc_shaping.validation.ch_market_time_regime as market_time
from pfc_shaping.pipeline.governed_release_cli_contract import ReleaseCliIdentityError
from pfc_shaping.validation.ch_lt_origin_registry_protocol import (
    PROTOCOL_ID,
    PROTOCOL_RELATIVE_PATH,
    PROTOCOL_SHA256,
    REQUIRED_EXTERNAL_EVIDENCE,
    STATUS,
    ChLtOriginRegistryProtocolError,
    assess_origin_registry_protocol,
    compute_protocol_id,
    verify_origin_registry_protocol,
)
from scripts import audit_ch_lt_origin_registry_protocol as audit_script

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.joinpath(*PROTOCOL_RELATIVE_PATH.split("/"))


def _document() -> dict[str, object]:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def _mutated_bytes(
    document: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> bytes:
    mutated_id = compute_protocol_id(document)
    document["protocol_id"] = mutated_id
    monkeypatch.setattr(protocol, "PROTOCOL_ID", mutated_id)
    payload = json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    monkeypatch.setattr(protocol, "PROTOCOL_SHA256", hashlib.sha256(payload).hexdigest())
    return payload


def test_canonical_protocol_is_exact_outcome_blind_and_incomplete() -> None:
    result = verify_origin_registry_protocol(
        repo_root=ROOT,
        protocol_path=PROTOCOL,
        expected_protocol_sha256=PROTOCOL_SHA256,
    )

    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == PROTOCOL_SHA256
    assert compute_protocol_id(_document()) == PROTOCOL_ID
    assert result["status"] == STATUS
    assert result["protocol_hash_closed_locally"] is True
    assert result["protocol_externally_frozen"] is False
    assert result["fmv_cadence_approved"] is False
    assert result["statistical_cadence"] == "MONTHLY"
    assert result["exact_schedule_frozen"] is False
    assert result["registry_implemented"] is False
    assert result["global_origin_uniqueness_enforced"] is False
    assert result["countable_prospective_origin_count"] == 0
    assert result["missing_external_evidence"] == list(REQUIRED_EXTERNAL_EVIDENCE)
    assert result["governance_bindings_verified"] is True
    assert result["structural_origin_countable"] is False
    assert result["power_design_complete"] is False
    for field in (
        "evidence_slot_satisfied",
        "truth_open_authorized",
        "scientific_admission",
        "execution_authorized",
        "production_authorization",
        "promotion_gate",
        "future_holdout_consumed",
        "t057_consumed",
    ):
        assert result[field] is False


def test_protocol_freezes_monthly_slot_and_no_backfill_semantics() -> None:
    document = _document()
    cadence = document["cadence_proposal"]
    assert isinstance(cadence, dict)
    assert cadence["statistical_cadence"] == "MONTHLY"
    assert cadence["maximum_countable_origins_per_slot"] == 1
    assert cadence["fmv_approval_state"] == "MISSING"
    assert cadence["exact_schedule_manifest_path"] is None
    assert cadence["schedule_materialization"].endswith("NO_RUNTIME_CALENDAR_INFERENCE")
    assert (
        cadence["origin_time_window_invariant"]
        == "CAPTURE_WINDOW_OPEN_UTC_LE_ORIGIN_AS_OF_UTC_LE_CAPTURE_WINDOW_CLOSE_UTC"
    )
    assert (
        cadence["registry_commit_window_invariant"]
        == "ORIGIN_AS_OF_UTC_LE_COMMITTED_AT_UTC_LE_LATEST_EXTERNAL_REGISTRY_COMMIT_UTC_LT_FIRST_TARGET_DELIVERY_START_UTC"
    )
    assert cadence["cet_cest_abbreviation_inference"] == "FORBIDDEN"
    assert cadence["missed_slot_policy"] == "MISSED_NOT_SHIFTED_NOT_BACKFILLED_NOT_REWEIGHTED"
    sources = cadence["official_planning_sources"]
    assert isinstance(sources, list) and len(sources) == 2
    assert all("PLANNING_ONLY" in item["evidence_role"] for item in sources)


def test_protocol_requires_remote_cas_exact_retry_and_complete_commitments() -> None:
    document = _document()
    registry = document["external_registry_domain"]
    request = document["registration_request_contract"]
    receipt = document["registration_receipt_contract"]
    assert isinstance(registry, dict)
    assert isinstance(request, dict)
    assert isinstance(receipt, dict)
    assert registry["local_filesystem_or_sqlite_role"].startswith("TEST_REFERENCE_ONLY")
    assert registry["duplicate_origin_or_slot"].startswith("COMPARE_AND_APPEND_REJECT")
    assert registry["exact_retry"].endswith(
        "IDENTICAL_REQUEST_ID_AND_REQUEST_SHA256"
    )
    assert registry["divergent_retry_or_alternate_branch"] == "FAIL_CLOSED"
    fields = set(request["required_fields"])
    assert {
        "origin_target_mask_inventory_sha256",
        "origin_available_eex_product_inventory_sha256",
        "prediction_commitment_sha256",
        "scenario_commitment_sha256",
        "structural_target_universe_commitment_sha256",
        "ex_ante_evaluation_mask_rule_commitment_sha256",
        "runtime_receipt_sha256",
        "trusted_origin_time_receipt_sha256",
    }.issubset(fields)
    assert "request_id" in fields
    assert "protocol_sha256" in fields
    assert "exact_schedule_entry_sha256" in fields
    assert "final_mask_commitment_sha256" not in fields
    assert request["request_id_derivation"].startswith("SHA256_DOMAIN_SEPARATED")
    assert request["truth_opened_required_value"] is False
    assert request["commit_deadline"].startswith(
        "COMMITTED_AT_UTC_LE_SELECTED_SCHEDULE_LATEST_EXTERNAL_REGISTRY_COMMIT_UTC"
    )
    request_invariants = set(request["cross_field_invariants"])
    assert "CADENCE_SLOT_ID_SELECTS_EXACTLY_ONE_HASH_MATCHING_SCHEDULE_ENTRY" in request_invariants
    assert "REALIZED_MATURITY_OR_TRUTH_MASK_IS_ABSENT_AT_ORIGIN" in request_invariants
    receipt_fields = set(receipt["required_fields"])
    assert {"request_id", "request_sha256", "protocol_sha256"}.issubset(receipt_fields)
    assert any(
        invariant.startswith("ORIGIN_AS_OF_UTC_LE_COMMITTED_AT_UTC")
        for invariant in receipt["cross_field_invariants"]
    )
    assert receipt["receipt_without_fresh_head_observation"].startswith("NOT_CURRENT")


def test_protocol_chain_reads_only_outcome_blind_t057_tombstone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules = (protocol, inventory, estimand, power_design, core_v3, core_v2, market_time)
    readers = {
        module: module.read_stable_single_link_file
        for module in modules
        if hasattr(module, "read_stable_single_link_file")
    }
    observed: list[str] = []

    def _reader_for(module: object):
        original = readers[module]

        def _reader(path: Path, **kwargs: object) -> bytes:
            observed.append(path.as_posix())
            return original(path, **kwargs)

        return _reader

    for module in readers:
        monkeypatch.setattr(module, "read_stable_single_link_file", _reader_for(module))

    verify_origin_registry_protocol(
        repo_root=ROOT,
        protocol_path=PROTOCOL,
        expected_protocol_sha256=PROTOCOL_SHA256,
    )

    t057_reads = [path for path in observed if "T057" in path.upper()]
    assert t057_reads
    assert all(path.endswith("T057-OUTCOME-BLIND-TOMBSTONE-20260730.json") for path in t057_reads)
    assert all("T057-EVIDENCE-SUPERSESSION-REGISTRY.json" not in path for path in observed)


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("lifecycle", "externally_frozen", True),
        ("cadence_proposal", "maximum_countable_origins_per_slot", 2),
        ("cadence_proposal", "fmv_approval_state", "APPROVED"),
        ("cadence_proposal", "missed_slot_policy", "SHIFT_TO_NEXT_DAY"),
        ("external_registry_domain", "local_filesystem_or_sqlite_role", "AUTHORITY"),
        ("external_registry_domain", "duplicate_origin_or_slot", "ALLOW"),
        ("registration_request_contract", "truth_opened_required_value", True),
        ("truth_firewall", "outcome_bearing_t057_registry", "ALLOWED"),
    ],
)
def test_rehashed_semantic_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    replacement: object,
) -> None:
    document = _document()
    policy = document[section]
    assert isinstance(policy, dict)
    policy[field] = replacement
    payload = _mutated_bytes(document, monkeypatch)
    with pytest.raises(ChLtOriginRegistryProtocolError, match="semantic section mismatch"):
        assess_origin_registry_protocol(document, document_bytes=payload)


def test_rehashed_authority_and_external_evidence_claims_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document()
    document["registry_implemented"] = True
    document["countable_prospective_origin_count"] = 1
    requirements = document["required_external_evidence"]
    assert isinstance(requirements, list) and isinstance(requirements[0], dict)
    requirements[0].update({"status": "SATISFIED", "path": "forged.json", "sha256": "a" * 64})
    payload = _mutated_bytes(document, monkeypatch)
    with pytest.raises(ChLtOriginRegistryProtocolError):
        assess_origin_registry_protocol(document, document_bytes=payload)


def test_wrong_hash_and_noncanonical_path_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ChLtOriginRegistryProtocolError, match="canonical identity"):
        verify_origin_registry_protocol(
            repo_root=ROOT,
            protocol_path=PROTOCOL,
            expected_protocol_sha256="0" * 64,
        )
    shadow = tmp_path / PROTOCOL.name
    shadow.write_bytes(PROTOCOL.read_bytes())
    with pytest.raises(ChLtOriginRegistryProtocolError, match="path is not canonical"):
        verify_origin_registry_protocol(
            repo_root=ROOT,
            protocol_path=shadow,
            expected_protocol_sha256=PROTOCOL_SHA256,
        )

    historical_v1 = PROTOCOL.with_name(
        "CH-LT-ORIGIN-REGISTRY-PROTOCOL-DRAFT-V1-20260730.json"
    )
    with pytest.raises(ChLtOriginRegistryProtocolError, match="path is not canonical"):
        verify_origin_registry_protocol(
            repo_root=ROOT,
            protocol_path=historical_v1,
            expected_protocol_sha256=PROTOCOL_SHA256,
        )


def test_local_cli_reports_valid_incomplete_protocol(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert audit_script.main(
        ["--protocol", str(PROTOCOL), "--expected-protocol-sha256", PROTOCOL_SHA256]
    ) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert captured.err == ""
    assert result["status"] == STATUS
    assert result["registry_implemented"] is False
    assert result["countable_prospective_origin_count"] == 0


def test_packaged_cli_is_sealed_and_failures_are_structured(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(packaged_cli, "assert_installed_runtime_sealed", lambda: None)
    assert packaged_cli.main(
        [
            "--evidence-root",
            str(ROOT),
            "--protocol",
            str(PROTOCOL),
            "--expected-protocol-sha256",
            "0" * 64,
        ]
    ) == 2
    captured = capsys.readouterr()
    result = json.loads(captured.err)
    assert captured.out == ""
    assert result["status"] == "INVALID_CH_LT_ORIGIN_REGISTRY_PROTOCOL_NO_GO"
    assert result["truth_open_authorized"] is False
    assert result["promotion_gate"] is False


def test_packaged_cli_rejects_unsealed_checkout_before_parsing(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _reject() -> None:
        raise ReleaseCliIdentityError("checkout runtime is not sealed")

    monkeypatch.setattr(packaged_cli, "assert_installed_runtime_sealed", _reject)
    assert packaged_cli.main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["status"] == "INVALID_GOVERNED_LT_RUNTIME"
