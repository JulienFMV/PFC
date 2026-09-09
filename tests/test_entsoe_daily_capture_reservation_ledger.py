from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pfc_shaping.validation import entsoe_daily_capture_reservation_ledger as module
from pfc_shaping.validation.entsoe_bounded_execution_receipt import canonical_sha256 as d236_sha
from pfc_shaping.validation.entsoe_daily_capture_reservation_ledger import (
    EntsoeDailyReservationError,
    validate_daily_capture_candidate,
    validate_daily_reservation_contract,
    validate_daily_reservation_ledger,
    verify_daily_reservation_paths,
)
from tests.test_entsoe_bounded_execution_receipt import _proposal

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / ".planning" / "phases" / "14-lt-audit-remediation"
CONTRACT = PHASE / "ENTSOE-DAILY-CAPTURE-RESERVATION-LEDGER-CONTRACT-V1.json"
D233 = PHASE / "EEX-ENTSOE-GOVERNED-ACQUISITION-PACKAGE-CONTRACT-V1.json"
PROOF_ROOT = ROOT / "build" / "databricks-eex-daily" / "2026-08-05"
D235 = PROOF_ROOT / "eex-entsoe-external-time-batch-proofs" / module.D235_PROOF_CONTENT_ID / "manifest.json"
D236 = PROOF_ROOT / "entsoe-bounded-execution-receipt-proofs" / module.D236_PROOF_CONTENT_ID / "manifest.json"
REFERENCE = datetime(2026, 8, 5, 13, 0, tzinfo=timezone.utc)


def _ledger(entries: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "schema_version": module.LEDGER_SCHEMA,
        "evidence_class": "SYNTHETIC_TEST_ONLY",
        "created_at_utc": "2026-08-05T12:30:00Z",
        "entries": [] if entries is None else entries,
        "authorities": dict(module.AUTHORITIES),
    }


def _candidate() -> dict[str, object]:
    return {
        "schema_version": module.CANDIDATE_SCHEMA,
        "evidence_class": "SYNTHETIC_TEST_ONLY",
        "capture_day_europe_zurich": "2026-08-05",
        "reservation_id": "44444444-4444-4444-8444-444444444444",
        "proposal": _proposal(),
        "authorities": dict(module.AUTHORITIES),
    }


def _entry(
    *,
    day: str = "2026-08-04",
    reservation: str = "55555555-5555-4555-8555-555555555555",
    batch: str = "66666666-6666-4666-8666-666666666666",
    proposal: str = "1" * 64,
    reserved_at: str = "2026-08-04T10:00:00Z",
    state: str = "SUCCEEDED",
) -> dict[str, object]:
    return {
        "capture_day_europe_zurich": day,
        "reservation_id": reservation,
        "capture_batch_id": batch,
        "proposal_content_id": proposal,
        "reserved_at_utc": reserved_at,
        "state": state,
    }


def test_contract_is_exact_and_zero_authority() -> None:
    result = validate_daily_reservation_contract(json.loads(CONTRACT.read_text(encoding="utf-8")))
    assert result["contract_content_id"] == module.CONTRACT_CONTENT_ID
    assert result["current_reservation_budget"] == 0
    assert result["capture_execution_authorized"] is False


def test_empty_ledger_and_candidate_are_structurally_eligible_only() -> None:
    result = validate_daily_capture_candidate(_candidate(), _ledger(), reference_time_utc=REFERENCE)
    assert result["status"].endswith("EXECUTION_NOT_AUTHORIZED")
    assert result["cost_fence_validated"] is True
    assert result["reservation_write_authorized"] is False


def test_existing_other_day_is_accepted() -> None:
    result = validate_daily_capture_candidate(_candidate(), _ledger([_entry()]), reference_time_utc=REFERENCE)
    assert result["capture_day_europe_zurich"] == "2026-08-05"


@pytest.mark.parametrize("state", module.COUNTED_STATES)
def test_every_counted_state_consumes_the_day(state: str) -> None:
    proposal = _proposal()
    entry = _entry(
        day="2026-08-05",
        reserved_at="2026-08-05T10:00:00Z",
        state=state,
        proposal=d236_sha(proposal),
    )
    with pytest.raises(EntsoeDailyReservationError, match="already consumed"):
        validate_daily_capture_candidate(_candidate(), _ledger([entry]), reference_time_utc=REFERENCE)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_day", "daily capture ceiling exceeded"),
        ("duplicate_reservation", "must be unique"),
        ("duplicate_batch", "must be unique"),
        ("duplicate_proposal", "must be unique"),
        ("unordered_time", "strictly increasing"),
        ("bad_state", "state is invalid"),
        ("day_time_mismatch", "day or reservation time differs"),
        ("future_reservation", "day or reservation time differs"),
    ],
)
def test_ledger_rejects_reusable_or_inconsistent_entries(mutation: str, message: str) -> None:
    first = _entry()
    second = _entry(
        day="2026-08-05",
        reservation="77777777-7777-4777-8777-777777777777",
        batch="88888888-8888-4888-8888-888888888888",
        proposal="2" * 64,
        reserved_at="2026-08-05T10:00:00Z",
    )
    if mutation == "duplicate_day":
        second["capture_day_europe_zurich"] = "2026-08-04"
        second["reserved_at_utc"] = "2026-08-04T11:00:00Z"
    elif mutation == "duplicate_reservation":
        second["reservation_id"] = first["reservation_id"]
    elif mutation == "duplicate_batch":
        second["capture_batch_id"] = first["capture_batch_id"]
    elif mutation == "duplicate_proposal":
        second["proposal_content_id"] = first["proposal_content_id"]
    elif mutation == "unordered_time":
        second["reserved_at_utc"] = "2026-08-04T09:00:00Z"
        second["capture_day_europe_zurich"] = "2026-08-04"
    elif mutation == "bad_state":
        second["state"] = "CANCELLED_REUSABLE"
    elif mutation == "day_time_mismatch":
        second["capture_day_europe_zurich"] = "2026-08-06"
    elif mutation == "future_reservation":
        second["reserved_at_utc"] = "2026-08-05T12:31:00Z"
    else:
        raise AssertionError(mutation)
    with pytest.raises(EntsoeDailyReservationError, match=message):
        validate_daily_reservation_ledger(_ledger([first, second]), reference_time_utc=REFERENCE)


def test_candidate_uses_zurich_day_not_utc_day() -> None:
    candidate = _candidate()
    proposal = candidate["proposal"]
    assert isinstance(proposal, dict)
    proposal["created_at_utc"] = "2026-08-04T22:30:00Z"
    proposal["snapshot_reference_time_utc"] = "2026-08-04T22:00:00Z"
    result = validate_daily_capture_candidate(candidate, _ledger(), reference_time_utc=REFERENCE)
    assert result["capture_day_europe_zurich"] == "2026-08-05"


def test_candidate_day_mismatch_and_changed_proposal_fail() -> None:
    candidate = _candidate()
    candidate["capture_day_europe_zurich"] = "2026-08-04"
    with pytest.raises(EntsoeDailyReservationError, match="differs from proposal Zurich day"):
        validate_daily_capture_candidate(candidate, _ledger(), reference_time_utc=REFERENCE)


def test_path_verifier_binds_d233_d235_d236(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    candidate_path = tmp_path / "candidate.json"
    ledger_path.write_text(json.dumps(_ledger()), encoding="utf-8")
    candidate_path.write_text(json.dumps(_candidate()), encoding="utf-8")
    result = verify_daily_reservation_paths(
        contract_path=CONTRACT.resolve(),
        d233_contract_path=D233.resolve(),
        d235_manifest_path=D235.resolve(),
        d236_manifest_path=D236.resolve(),
        ledger_path=ledger_path.resolve(),
        candidate_path=candidate_path.resolve(),
        reference_time_utc=REFERENCE,
    )
    assert result["capture_execution_authorized"] is False


def test_path_drift_duplicate_json_and_relative_path_fail(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    candidate_path = tmp_path / "candidate.json"
    changed_contract = tmp_path / "contract.json"
    ledger_path.write_text(json.dumps(_ledger()), encoding="utf-8")
    candidate_path.write_text('{"schema_version":"v1","schema_version":"v2"}', encoding="utf-8")
    changed_contract.write_bytes(CONTRACT.read_bytes() + b" ")
    kwargs = {
        "d233_contract_path": D233.resolve(),
        "d235_manifest_path": D235.resolve(),
        "d236_manifest_path": D236.resolve(),
        "ledger_path": ledger_path.resolve(),
        "candidate_path": candidate_path.resolve(),
        "reference_time_utc": REFERENCE,
    }
    with pytest.raises(EntsoeDailyReservationError, match="contract SHA-256 differs"):
        verify_daily_reservation_paths(contract_path=changed_contract.resolve(), **kwargs)
    with pytest.raises(EntsoeDailyReservationError, match="duplicate key"):
        verify_daily_reservation_paths(contract_path=CONTRACT.resolve(), **kwargs)
    with pytest.raises(EntsoeDailyReservationError, match="path must be absolute"):
        verify_daily_reservation_paths(contract_path=Path("contract.json"), **kwargs)


def test_mutated_authority_fails_closed() -> None:
    candidate = copy.deepcopy(_candidate())
    candidate["authorities"]["capture_execution_authorized"] = True
    with pytest.raises(EntsoeDailyReservationError, match="candidate authorities differs"):
        validate_daily_capture_candidate(candidate, _ledger(), reference_time_utc=REFERENCE)
