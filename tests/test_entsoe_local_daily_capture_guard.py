from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pfc_shaping.data import entsoe_local_daily_capture_guard as module
from pfc_shaping.data.entsoe_local_daily_capture_guard import (
    AUTHORITIES,
    CONTRACT_CONTENT_ID,
    CONTRACT_STATUS,
    PURPOSE,
    RESERVATION_SCHEMA,
    STATUS,
    EntsoeCaptureDayConsumedError,
    EntsoeLocalDailyCaptureGuardError,
    reserve_entsoe_capture_day,
    validate_local_daily_guard_contract,
    validate_local_daily_guard_contract_path,
    validate_local_reservation_document,
)
from pfc_shaping.validation.entsoe_series_inventory import (
    CONTRACT_CONTENT_ID as D243_CONTRACT_CONTENT_ID,
)
from pfc_shaping.validation.entsoe_series_inventory import (
    QUERY_SHA256 as D243_QUERY_SHA256,
)

REFERENCE = datetime(2026, 8, 6, 12, 0, 30, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "ENTSOE-LOCAL-DAILY-CAPTURE-GUARD-CONTRACT-V1.json"
)


def reservation(*, reservation_id: str | None = None) -> dict[str, object]:
    return {
        "schema_version": RESERVATION_SCHEMA,
        "evidence_class": "REAL_LOCAL_WORKSTATION_COST_GUARD",
        "purpose": PURPOSE,
        "query_sha256": D243_QUERY_SHA256,
        "d243_contract_content_id": D243_CONTRACT_CONTENT_ID,
        "capture_day_europe_zurich": "2026-08-06",
        "reservation_id": reservation_id or "123e4567-e89b-12d3-a456-426614174000",
        "reserved_at_utc": "2026-08-06T12:00:00Z",
        "warehouse_id_sha256": "b" * 64,
        "clock_source": "LOCAL_SYSTEM_UTC_UNATTESTED",
        "state": "AUTHORIZED_RESERVED",
        "failure_consumes_day": True,
        "retry_same_day_authorized": False,
        "authorities": dict(AUTHORITIES),
    }


def set_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "reservations"
    monkeypatch.setattr(module, "CANONICAL_RESERVATION_ROOT", root)
    return root


def test_exact_contract_binds_real_local_guard_without_execution_authority() -> None:
    result = validate_local_daily_guard_contract_path(CONTRACT)

    assert result["status"] == CONTRACT_STATUS
    assert result["contract_content_id"] == CONTRACT_CONTENT_ID
    assert result["local_daily_cost_guard_implemented"] is True
    assert result["cross_host_uniqueness_proven"] is False
    assert result["capture_execution_authorized"] is False


def test_contract_cannot_escalate_warehouse_execution() -> None:
    document = json.loads(CONTRACT.read_text(encoding="utf-8"))
    document["authorities"]["warehouse_execution"] = True

    with pytest.raises(
        EntsoeLocalDailyCaptureGuardError,
        match="contract authorities differs",
    ):
        validate_local_daily_guard_contract(document)


def test_real_local_reservation_consumes_day_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = set_root(monkeypatch, tmp_path)

    result = reserve_entsoe_capture_day(reservation(), reference_time_utc=REFERENCE)

    assert result["status"] == STATUS
    assert result["day_consumed"] is True
    assert result["same_day_retry_authorized"] is False
    assert result["warehouse_state_checked"] is False
    assert result["warehouse_start_count"] == 0
    assert result["sql_statement_count"] == 0
    assert result["databricks_write_count"] == 0
    assert result["capture_execution_authorized"] is False
    marker = json.loads((root / "2026-08-06.json").read_text(encoding="utf-8"))
    assert marker["reservation_content_id"] == result["reservation_content_id"]
    assert marker["failure_consumes_day"] is True
    assert "token" not in json.dumps(marker).casefold()
    assert "host" not in json.dumps(marker).casefold()


def test_any_existing_marker_consumes_day_even_if_corrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = set_root(monkeypatch, tmp_path)
    root.mkdir(parents=True)
    (root / "2026-08-06.json").write_bytes(b"partial")

    with pytest.raises(EntsoeCaptureDayConsumedError, match="already consumed"):
        reserve_entsoe_capture_day(reservation(), reference_time_utc=REFERENCE)


def test_same_reservation_cannot_be_replayed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_root(monkeypatch, tmp_path)
    reserve_entsoe_capture_day(reservation(), reference_time_utc=REFERENCE)

    with pytest.raises(EntsoeCaptureDayConsumedError, match="already consumed"):
        reserve_entsoe_capture_day(reservation(), reference_time_utc=REFERENCE)


def test_concurrent_reservations_have_exactly_one_winner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = set_root(monkeypatch, tmp_path)
    reservation_ids = [str(uuid.uuid4()) for _ in range(32)]

    def attempt(index: int) -> str:
        value = reservation(reservation_id=reservation_ids[index])
        try:
            reserve_entsoe_capture_day(value, reference_time_utc=REFERENCE)
        except EntsoeCaptureDayConsumedError:
            return "consumed"
        return "reserved"

    with ThreadPoolExecutor(max_workers=16) as pool:
        outcomes = list(pool.map(attempt, range(32)))

    assert outcomes.count("reserved") == 1
    assert outcomes.count("consumed") == 31
    assert [path.name for path in root.iterdir()] == ["2026-08-06.json"]


def test_fsync_failure_still_consumes_day(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_root(monkeypatch, tmp_path)

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(module.os, "fsync", fail_fsync)
    with pytest.raises(EntsoeLocalDailyCaptureGuardError, match="day remains consumed"):
        reserve_entsoe_capture_day(reservation(), reference_time_utc=REFERENCE)
    with pytest.raises(EntsoeCaptureDayConsumedError, match="already consumed"):
        reserve_entsoe_capture_day(reservation(), reference_time_utc=REFERENCE)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value.update(query_sha256="c" * 64),
            "D243 query SHA-256 differs",
        ),
        (
            lambda value: value.update(d243_contract_content_id="c" * 64),
            "D243 contract content ID differs",
        ),
        (
            lambda value: value.update(capture_day_europe_zurich="2026-08-07"),
            "Europe/Zurich reservation day differs",
        ),
        (
            lambda value: value.update(retry_same_day_authorized=True),
            "same-day retry authority differs",
        ),
        (
            lambda value: value.update(authorities={**AUTHORITIES, "model_input_authorized": True}),
            "reservation authorities differs",
        ),
        (
            lambda value: value.update(warehouse_id_sha256="not-a-hash"),
            "warehouse ID SHA-256 must be lowercase SHA-256",
        ),
    ],
)
def test_reservation_document_fails_closed(mutation, match: str) -> None:
    value = reservation()
    mutation(value)

    with pytest.raises(EntsoeLocalDailyCaptureGuardError, match=match):
        validate_local_reservation_document(value, reference_time_utc=REFERENCE)


def test_stale_and_future_reservations_are_rejected() -> None:
    value = reservation()

    with pytest.raises(EntsoeLocalDailyCaptureGuardError, match="stale"):
        validate_local_reservation_document(
            value,
            reference_time_utc=REFERENCE + timedelta(minutes=2),
        )
    with pytest.raises(EntsoeLocalDailyCaptureGuardError, match="after reference"):
        validate_local_reservation_document(
            value,
            reference_time_utc=REFERENCE - timedelta(minutes=2),
        )


def test_reservation_root_cannot_escape_workspace_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "CANONICAL_RESERVATION_ROOT", module.WORKSPACE_ROOT)

    with pytest.raises(EntsoeLocalDailyCaptureGuardError, match="below workspace build"):
        reserve_entsoe_capture_day(reservation(), reference_time_utc=REFERENCE)


def test_reservation_root_must_be_absolute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "CANONICAL_RESERVATION_ROOT", Path("build/escape"))

    with pytest.raises(EntsoeLocalDailyCaptureGuardError, match="path safety check failed"):
        reserve_entsoe_capture_day(reservation(), reference_time_utc=REFERENCE)
