from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from pfc_shaping.data.ch_lt_origin_registry_reference import (
    PROTOCOL_REGISTRY_DOMAIN_SHA256,
    RECEIPT_SCHEMA,
    REFERENCE_AUTHORITY_CLASSIFICATION,
    REGISTRY_DOMAIN_SHA256,
    REQUEST_SCHEMA,
    SCHEDULE_MANIFEST_SCHEMA,
    TRUSTED_TIME_RECEIPT_SCHEMA,
    ChLtOriginRegistryReferenceAuthority,
    ChLtOriginRegistryReferenceConfig,
    ChLtOriginRegistryReferenceConflict,
    ChLtOriginRegistryReferenceError,
    canonical_json,
    compute_origin_id,
    semantic_sha256,
    sign_exact_schedule_entry,
    sign_origin_registration_receipt,
    sign_origin_registration_request,
    sign_trusted_time_receipt,
    verify_head_observation,
    verify_origin_registration_receipt,
)
from pfc_shaping.validation.ch_lt_origin_registry_protocol import (
    PROTOCOL_ID,
    PROTOCOL_RELATIVE_PATH,
    PROTOCOL_SHA256,
)
from pfc_shaping.validation.ch_lt_origin_target_mask_inventory import (
    build_structural_inventory,
)
from pfc_shaping.validation.ch_lt_origin_target_mask_inventory import (
    canonical_json_bytes as inventory_json_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.joinpath(*PROTOCOL_RELATIVE_PATH.split("/"))
UTC = timezone.utc

ARTIFACT_FIELDS = (
    "origin_target_mask_inventory_sha256",
    "origin_available_eex_product_inventory_sha256",
    "eex_vintage_manifest_sha256",
    "monthly_solver_configuration_sha256",
    "candidate_baseline_identity_manifest_sha256",
    "candidate_hypothesis_and_procedure_manifest_sha256",
    "prediction_commitment_sha256",
    "scenario_commitment_sha256",
    "structural_target_universe_commitment_sha256",
    "ex_ante_evaluation_mask_rule_commitment_sha256",
    "calendar_and_strata_manifest_sha256",
    "runtime_receipt_sha256",
    "project_wheel_sha256",
)


@dataclass
class ReferenceSetup:
    authority: ChLtOriginRegistryReferenceAuthority
    config: ChLtOriginRegistryReferenceConfig
    request_key: Ed25519PrivateKey
    request_public_key: Ed25519PublicKey
    registry_public_key: Ed25519PublicKey
    registry_key: Ed25519PrivateKey
    schedule_payload: bytes
    artifacts: dict[str, bytes]
    entries: dict[str, dict[str, object]]
    inventories: dict[str, dict[str, object]]
    inventory_sha_by_slot: dict[str, str]
    time_by_slot: dict[str, str]
    now: datetime

    def request(
        self,
        slot: str,
        *,
        expected_sequence: int = 1,
        previous_receipt_id: str | None = None,
        operation_id: str | None = None,
    ) -> bytes:
        entry = self.entries[slot]
        origin_as_of = str(entry["capture_window_open_utc"])
        selected_inventory_id = str(self.inventories[slot]["inventory_id"])
        inventory_sha256 = self.inventory_sha_by_slot[slot]
        origin_id = compute_origin_id(
            origin_as_of_utc=origin_as_of,
            inventory_id=selected_inventory_id,
            inventory_sha256=inventory_sha256,
        )
        core: dict[str, object] = {
            "schema_version": REQUEST_SCHEMA,
            "protocol_sha256": PROTOCOL_SHA256,
            "protocol_id": PROTOCOL_ID,
            "registry_domain_sha256": REGISTRY_DOMAIN_SHA256,
            "registration_operation_id": operation_id or str(uuid.uuid4()),
            "expected_sequence": expected_sequence,
            "expected_previous_receipt_id": previous_receipt_id,
            "cadence_slot_id": slot,
            "cadence_contract_sha256": semantic_sha256(
                _protocol_document()["cadence_proposal"]
            ),
            "exact_schedule_manifest_sha256": hashlib.sha256(
                self.schedule_payload
            ).hexdigest(),
            "exact_schedule_entry_id": entry["schedule_entry_id"],
            "exact_schedule_entry_sha256": hashlib.sha256(
                canonical_json(entry)
            ).hexdigest(),
            "capture_window_open_utc": entry["capture_window_open_utc"],
            "capture_window_close_utc": entry["capture_window_close_utc"],
            "latest_external_registry_commit_utc": entry[
                "latest_external_registry_commit_utc"
            ],
            "origin_as_of_utc": origin_as_of,
            "origin_id": origin_id,
            "origin_target_mask_inventory_id": selected_inventory_id,
            "project_source_revision": hashlib.sha256(b"source revision").hexdigest(),
            "trusted_origin_time_utc": origin_as_of,
            "trusted_origin_time_receipt_sha256": self._trusted_time_sha(slot),
            "first_target_delivery_start_utc": self.inventories[slot]["targets"][0][
                "delivery_start_utc"
            ],
            "truth_opened": False,
            "future_holdout_consumed": False,
            "t057_consumed": False,
        }
        for field in ARTIFACT_FIELDS:
            core[field] = (
                inventory_sha256
                if field == "origin_target_mask_inventory_sha256"
                else self._artifact_sha(field)
            )
        return canonical_json(
            sign_origin_registration_request(core, private_key=self.request_key)
        )

    def _artifact_sha(self, field: str) -> str:
        payload = canonical_json({"field": field, "reference_fixture": True})
        digest = hashlib.sha256(payload).hexdigest()
        assert self.artifacts[digest] == payload
        return digest

    def _trusted_time_sha(self, slot: str) -> str:
        return self.time_by_slot[slot]


def _protocol_document() -> dict[str, object]:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def _write_keypair(
    root: Path, name: str
) -> tuple[Ed25519PrivateKey, Path, Path]:
    private_key = Ed25519PrivateKey.generate()
    private_path = root / f"{name}-private.pem"
    public_path = root / f"{name}-public.pem"
    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_key, private_path, public_path


def _schedule_entry(
    *,
    slot: str,
    trading_day: str,
    open_utc: str,
    close_utc: str,
    deadline_utc: str,
    schedule_key: Ed25519PrivateKey,
) -> dict[str, object]:
    return sign_exact_schedule_entry(
        {
            "protocol_sha256": PROTOCOL_SHA256,
            "cadence_slot_id": slot,
            "eex_trading_day": trading_day,
            "capture_window_open_utc": open_utc,
            "capture_window_close_utc": close_utc,
            "latest_external_registry_commit_utc": deadline_utc,
            "official_eex_calendar_document_sha256": hashlib.sha256(
                b"official calendar"
            ).hexdigest(),
            "official_settlement_event_definition_sha256": hashlib.sha256(
                b"official settlement semantics"
            ).hexdigest(),
        },
        private_key=schedule_key,
    )


@pytest.fixture
def reference(tmp_path: Path) -> ReferenceSetup:
    request_key, _, request_public = _write_keypair(tmp_path, "request")
    schedule_key, _, schedule_public = _write_keypair(tmp_path, "schedule")
    time_key, _, time_public = _write_keypair(tmp_path, "time")
    registry_key, registry_private, _ = _write_keypair(tmp_path, "registry")
    entries = {
        "2026-07": _schedule_entry(
            slot="2026-07",
            trading_day="2026-07-31",
            open_utc="2026-07-31T18:05:00Z",
            close_utc="2026-07-31T18:10:00Z",
            deadline_utc="2026-07-31T18:15:00Z",
            schedule_key=schedule_key,
        ),
        "2026-08": _schedule_entry(
            slot="2026-08",
            trading_day="2026-08-31",
            open_utc="2026-08-31T18:05:00Z",
            close_utc="2026-08-31T18:10:00Z",
            deadline_utc="2026-08-31T18:15:00Z",
            schedule_key=schedule_key,
        ),
    }
    schedule_payload = canonical_json(
        {
            "schema_version": SCHEDULE_MANIFEST_SCHEMA,
            "protocol_sha256": PROTOCOL_SHA256,
            "entries": [entries["2026-07"], entries["2026-08"]],
        }
    )
    artifacts: dict[str, bytes] = {}
    for field in ARTIFACT_FIELDS:
        if field == "origin_target_mask_inventory_sha256":
            continue
        payload = canonical_json({"field": field, "reference_fixture": True})
        artifacts[hashlib.sha256(payload).hexdigest()] = payload
    inventories = {
        slot: build_structural_inventory(
            repo_root=ROOT,
            origin_as_of_utc=str(entry["capture_window_open_utc"]),
        )
        for slot, entry in entries.items()
    }
    inventory_sha_by_slot: dict[str, str] = {}
    for slot, document in inventories.items():
        payload = inventory_json_bytes(document) + b"\n"
        digest = hashlib.sha256(payload).hexdigest()
        artifacts[digest] = payload
        inventory_sha_by_slot[slot] = digest
    for slot, entry in entries.items():
        subject = str(entry["capture_window_open_utc"])
        time_payload = canonical_json(
            sign_trusted_time_receipt(
                {
                    "schema_version": TRUSTED_TIME_RECEIPT_SCHEMA,
                    "subject_utc": subject,
                    "issued_at_utc": subject,
                    "purpose": "CH_LT_ORIGIN_AS_OF_V2",
                    "nonce": hashlib.sha256(f"time:{slot}".encode("ascii")).hexdigest(),
                },
                private_key=time_key,
            )
        )
        artifacts[hashlib.sha256(time_payload).hexdigest()] = time_payload

    time_by_slot = {
        slot: next(
            digest
            for digest, payload in artifacts.items()
            if b'"schema_version":"ch_lt_origin_trusted_time_receipt.reference.v1"'
            in payload
            and f'"subject_utc":"{entry["capture_window_open_utc"]}"'.encode("ascii")
            in payload
        )
        for slot, entry in entries.items()
    }

    now = datetime(2026, 7, 31, 18, 7, tzinfo=UTC)
    config = ChLtOriginRegistryReferenceConfig(
        database_path=(tmp_path / "origin-registry.sqlite").absolute(),
        request_public_key_path=request_public,
        schedule_public_key_path=schedule_public,
        trusted_time_public_key_path=time_public,
        registry_private_key_path=registry_private,
        protocol_payload=PROTOCOL.read_bytes(),
        schedule_manifest_payload=schedule_payload,
        artifact_loader=lambda digest: artifacts[digest],
    )
    setup = ReferenceSetup(
        authority=ChLtOriginRegistryReferenceAuthority(config, clock=lambda: now),
        config=config,
        request_key=request_key,
        request_public_key=request_key.public_key(),
        registry_public_key=registry_key.public_key(),
        registry_key=registry_key,
        schedule_payload=schedule_payload,
        artifacts=artifacts,
        entries=entries,
        inventories=inventories,
        inventory_sha_by_slot=inventory_sha_by_slot,
        time_by_slot=time_by_slot,
        now=now,
    )
    return setup


def test_reference_compare_append_retry_head_and_local_non_authority(
    reference: ReferenceSetup,
) -> None:
    request_payload = reference.request("2026-07")
    receipt_payload = reference.authority.compare_and_append(request_payload)

    assert reference.authority.compare_and_append(request_payload) == receipt_payload
    assert reference.authority.get_head() == receipt_payload
    request = json.loads(request_payload)
    assert (
        reference.authority.lookup_operation(
            request["registration_operation_id"],
            expected_request_payload=request_payload,
        )
        == receipt_payload
    )
    assessment = verify_origin_registration_receipt(
        request_payload,
        receipt_payload,
        request_public_key=reference.request_public_key,
        registry_public_key=reference.registry_public_key,
    )
    assert assessment["reference_authority_classification"] == REFERENCE_AUTHORITY_CLASSIFICATION
    raw_receipt = json.loads(receipt_payload)
    assert raw_receipt["schema_version"] == RECEIPT_SCHEMA
    assert raw_receipt["schema_version"] != "ch_lt_origin_registration_receipt.v2"
    assert raw_receipt["countable_prospective_origin"] is False
    assert assessment["receipt_claimed_countable_prospective_origin"] is False
    assert assessment["external_authority_verified"] is False
    assert assessment["countable_prospective_origin"] is False
    assert assessment["scientific_admission"] is False
    assert assessment["production_authorization"] is False

    nonce = hashlib.sha256(b"fresh head challenge").hexdigest()
    observation_payload = reference.authority.observe_head(challenge_nonce=nonce)
    observation = verify_head_observation(
        observation_payload,
        expected_receipt_payload=receipt_payload,
        expected_challenge_nonce=nonce,
        registry_public_key=reference.registry_public_key,
        reference_time=reference.now,
    )
    assert observation["countable_prospective_origin"] is False
    assert observation["promotion_gate"] is False
    with pytest.raises(ChLtOriginRegistryReferenceConflict, match="already consumed"):
        reference.authority.observe_head(challenge_nonce=nonce)


def test_reference_rejects_duplicate_slot_and_retains_noncountable_result(
    reference: ReferenceSetup,
) -> None:
    first_request = reference.request("2026-07")
    first_receipt = reference.authority.compare_and_append(first_request)
    first = verify_origin_registration_receipt(
        first_request,
        first_receipt,
        request_public_key=reference.request_public_key,
        registry_public_key=reference.registry_public_key,
    )
    duplicate = reference.request(
        "2026-07",
        expected_sequence=2,
        previous_receipt_id=str(first["receipt_id"]),
        operation_id=str(uuid.uuid4()),
    )
    with pytest.raises(ChLtOriginRegistryReferenceConflict, match="uniqueness conflict"):
        reference.authority.compare_and_append(duplicate)
    assert reference.authority.rejection_count() == 1
    assert reference.authority.get_head() == first_receipt


def test_rejection_audit_never_persists_untrusted_payload_or_error_text(
    reference: ReferenceSetup,
) -> None:
    sentinel = b"FMV_SECRET_SENTINEL_MUST_NOT_ENTER_SQLITE_20260731"
    rejected_payload = b'{"credential":"' + sentinel + b'"}'

    with pytest.raises(ChLtOriginRegistryReferenceError, match="fields are not exact"):
        reference.authority.compare_and_append(rejected_payload)

    with sqlite3.connect(reference.config.database_path) as connection:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(rejected_operations)")
        }
        row = connection.execute(
            """
            SELECT request_sha256, operation_id, request_id, error_type,
                   error_code, request_size_bytes
            FROM rejected_operations
            """
        ).fetchone()
    assert columns == {
        "rejection_id",
        "request_sha256",
        "operation_id",
        "request_id",
        "error_type",
        "error_code",
        "request_size_bytes",
    }
    assert row == (
        hashlib.sha256(rejected_payload).hexdigest(),
        None,
        None,
        "ChLtOriginRegistryReferenceError",
        "VALIDATION_ERROR",
        len(rejected_payload),
    )
    for suffix in ("", "-wal", "-shm"):
        selected = Path(f"{reference.config.database_path}{suffix}")
        if selected.exists():
            assert sentinel not in selected.read_bytes()


def test_reference_receipt_signer_rejects_production_schema_and_true_authority(
    reference: ReferenceSetup,
) -> None:
    request_payload = reference.request("2026-07")
    receipt_payload = reference.authority.compare_and_append(request_payload)
    core = json.loads(receipt_payload)
    core.pop("receipt_id")
    core.pop("registry_signature")

    with pytest.raises(ChLtOriginRegistryReferenceError, match="countability"):
        sign_origin_registration_receipt(
            {**core, "countable_prospective_origin": True},
            private_key=reference.registry_key,
        )
    with pytest.raises(ChLtOriginRegistryReferenceError, match="countability"):
        sign_origin_registration_receipt(
            {**core, "schema_version": "ch_lt_origin_registration_receipt.v2"},
            private_key=reference.registry_key,
        )


def test_reference_restart_verifies_chain_identity_and_sqlite_schema(
    reference: ReferenceSetup,
) -> None:
    request_payload = reference.request("2026-07")
    receipt_payload = reference.authority.compare_and_append(request_payload)

    reopened = ChLtOriginRegistryReferenceAuthority(
        reference.config,
        clock=lambda: reference.now,
    )
    assert reopened.get_head() == receipt_payload
    with sqlite3.connect(reference.config.database_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (2,)


def test_concurrent_stale_head_has_exactly_one_winner(reference: ReferenceSetup) -> None:
    first = reference.request("2026-07")
    second = reference.request("2026-08")

    def submit(payload: bytes) -> str:
        try:
            reference.authority.compare_and_append(payload)
        except ChLtOriginRegistryReferenceConflict:
            return "conflict"
        return "committed"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, (first, second)))
    assert sorted(results) == ["committed", "conflict"]
    assert reference.authority.rejection_count() == 1


def test_tamper_deadline_and_stale_observation_fail_closed(
    reference: ReferenceSetup,
) -> None:
    request_payload = reference.request("2026-07")
    tampered = bytearray(request_payload)
    tampered[-2] = ord("0") if tampered[-2] != ord("0") else ord("1")
    with pytest.raises(ChLtOriginRegistryReferenceError):
        reference.authority.compare_and_append(bytes(tampered))

    receipt_payload = reference.authority.compare_and_append(request_payload)
    nonce = hashlib.sha256(b"stale challenge").hexdigest()
    observation_payload = reference.authority.observe_head(challenge_nonce=nonce)
    with pytest.raises(ChLtOriginRegistryReferenceError, match="not fresh"):
        verify_head_observation(
            observation_payload,
            expected_receipt_payload=receipt_payload,
            expected_challenge_nonce=nonce,
            registry_public_key=reference.registry_public_key,
            reference_time=reference.now + timedelta(minutes=2),
        )


def test_reference_rejects_overlapping_signer_roles(tmp_path: Path) -> None:
    key, private_path, public_path = _write_keypair(tmp_path, "shared")
    entry = _schedule_entry(
        slot="2026-07",
        trading_day="2026-07-31",
        open_utc="2026-07-31T18:05:00Z",
        close_utc="2026-07-31T18:10:00Z",
        deadline_utc="2026-07-31T18:15:00Z",
        schedule_key=key,
    )
    schedule_payload = canonical_json(
        {
            "schema_version": SCHEDULE_MANIFEST_SCHEMA,
            "protocol_sha256": PROTOCOL_SHA256,
            "entries": [entry],
        }
    )
    config = ChLtOriginRegistryReferenceConfig(
        database_path=(tmp_path / "overlap.sqlite").absolute(),
        request_public_key_path=public_path,
        schedule_public_key_path=public_path,
        trusted_time_public_key_path=public_path,
        registry_private_key_path=private_path,
        protocol_payload=PROTOCOL.read_bytes(),
        schedule_manifest_payload=schedule_payload,
        artifact_loader=lambda _digest: b"",
    )
    with pytest.raises(ChLtOriginRegistryReferenceError, match="roles must be disjoint"):
        ChLtOriginRegistryReferenceAuthority(config)


def test_reference_source_is_outcome_blind_and_never_claims_authority() -> None:
    source = (
        ROOT / "pfc_shaping" / "data" / "ch_lt_origin_registry_reference.py"
    ).read_text(encoding="utf-8")
    assert "T057-EVIDENCE-SUPERSESSION" not in source
    assert "COUNTABLE_AUTHORITY = False" in source
    assert "NON_PRODUCTION = True" in source
    assert "local SQLite database" in source
    assert REGISTRY_DOMAIN_SHA256 != PROTOCOL_REGISTRY_DOMAIN_SHA256
