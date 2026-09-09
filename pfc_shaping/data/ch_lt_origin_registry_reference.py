"""NON-PRODUCTION reference authority for CH LT origin registration.

This module exercises the outcome-blind protocol-v2 request, schedule, and
chronology semantics through a deliberately incompatible reference receipt
schema and signing domain.  It is not an exact production wire
implementation and is intentionally excluded from the governed LT wheel: a
local SQLite database controlled by the model operator is not an independent
registry, trusted clock, WORM store, or production authority and can never
make an origin countable.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sqlite3
import stat
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    freeze_process_immutable_file,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json
from pfc_shaping.validation import ch_lt_origin_target_mask_inventory as origin_inventory
from pfc_shaping.validation.ch_lt_origin_registry_protocol import (
    PROTOCOL_ID,
    PROTOCOL_SHA256,
)
from pfc_shaping.validation.ch_lt_origin_registry_protocol import (
    SCHEMA_VERSION as PROTOCOL_SCHEMA_VERSION,
)

REFERENCE_AUTHORITY_CLASSIFICATION = "NON_PRODUCTION_TEST_ONLY_NEVER_COUNTABLE"
REFERENCE_SCHEMA_VERSION = "ch_lt_origin_registry_reference.v2"
REQUEST_SCHEMA = "ch_lt_origin_registration_request.v2"
RECEIPT_SCHEMA = "ch_lt_origin_registration_receipt.non_production_reference.v1"
SCHEDULE_MANIFEST_SCHEMA = "ch_lt_origin_exact_schedule_manifest.reference.v1"
TRUSTED_TIME_RECEIPT_SCHEMA = "ch_lt_origin_trusted_time_receipt.reference.v1"
HEAD_OBSERVATION_SCHEMA = "ch_lt_origin_registry_head_observation.reference.v1"
SIGNATURE_ALGORITHM = "ED25519"
PROTOCOL_LOGICAL_DOMAIN = "FMV_CH_LT_CONFIRMATORY_ORIGINS_V2"
LOGICAL_DOMAIN = "FMV_CH_LT_CONFIRMATORY_ORIGINS_V2_NON_PRODUCTION_REFERENCE_V1"
REGISTRY_DOMAIN_SHA256 = hashlib.sha256(LOGICAL_DOMAIN.encode("ascii")).hexdigest()
PROTOCOL_REGISTRY_DOMAIN_SHA256 = hashlib.sha256(
    PROTOCOL_LOGICAL_DOMAIN.encode("ascii")
).hexdigest()

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CADENCE_SLOT = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_MAX_BYTES = 16 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
_MAX_SEQUENCE = 10_000_000
_MAX_HEAD_TTL = timedelta(minutes=5)
_SQLITE_USER_VERSION = 2
_REFERENCE_BUILD_ROOT = Path(__file__).absolute().parents[2] / "build"
_REFERENCE_RECEIPT_DOMAIN = (
    b"FMV_CH_LT_ORIGIN_REGISTRATION_RECEIPT_NON_PRODUCTION_REFERENCE_V1"
)

_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "request_id",
        "request_signature",
        "protocol_sha256",
        "protocol_id",
        "registry_domain_sha256",
        "registration_operation_id",
        "expected_sequence",
        "expected_previous_receipt_id",
        "cadence_slot_id",
        "cadence_contract_sha256",
        "exact_schedule_manifest_sha256",
        "exact_schedule_entry_id",
        "exact_schedule_entry_sha256",
        "capture_window_open_utc",
        "capture_window_close_utc",
        "latest_external_registry_commit_utc",
        "origin_as_of_utc",
        "origin_id",
        "origin_target_mask_inventory_sha256",
        "origin_target_mask_inventory_id",
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
        "project_source_revision",
        "trusted_origin_time_utc",
        "trusted_origin_time_receipt_sha256",
        "first_target_delivery_start_utc",
        "truth_opened",
        "future_holdout_consumed",
        "t057_consumed",
    }
)
_REQUEST_CORE_FIELDS = _REQUEST_FIELDS - {"request_id", "request_signature"}
_ARTIFACT_SHA_FIELDS = frozenset(
    {
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
    }
)
_SCHEDULE_ENTRY_FIELDS = frozenset(
    {
        "schedule_entry_id",
        "protocol_sha256",
        "cadence_slot_id",
        "eex_trading_day",
        "capture_window_open_utc",
        "capture_window_close_utc",
        "latest_external_registry_commit_utc",
        "official_eex_calendar_document_sha256",
        "official_settlement_event_definition_sha256",
        "signature",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_sha256",
        "protocol_id",
        "registry_domain_sha256",
        "receipt_id",
        "sequence",
        "previous_receipt_id",
        "registration_operation_id",
        "request_id",
        "request_sha256",
        "cadence_slot_id",
        "exact_schedule_entry_id",
        "exact_schedule_entry_sha256",
        "capture_window_open_utc",
        "capture_window_close_utc",
        "latest_external_registry_commit_utc",
        "origin_id",
        "origin_as_of_utc",
        "committed_at_utc",
        "countable_prospective_origin",
        "registry_signature",
    }
)
_HEAD_OBSERVATION_FIELDS = frozenset(
    {
        "schema_version",
        "registry_domain_sha256",
        "protocol_sha256",
        "sequence",
        "receipt_id",
        "receipt_sha256",
        "challenge_nonce",
        "observed_at_utc",
        "expires_at_utc",
        "reference_authority_classification",
        "observation_id",
        "registry_signature",
    }
)


class ChLtOriginRegistryReferenceError(RuntimeError):
    """Raised when reference conformance or persistence fails closed."""


class ChLtOriginRegistryReferenceConflict(ChLtOriginRegistryReferenceError):
    """Raised for compare-and-append, uniqueness, or idempotency conflicts."""


def _rejection_error_code(error: Exception) -> str:
    """Return a bounded code without persisting attacker-controlled text."""

    if isinstance(error, ChLtOriginRegistryReferenceConflict):
        return "CONFLICT"
    if isinstance(error, ChLtOriginRegistryReferenceError):
        return "VALIDATION_ERROR"
    if isinstance(error, OSError):
        return "IO_ERROR"
    if isinstance(error, ValueError):
        return "VALUE_ERROR"
    return "UNEXPECTED_ERROR"


@dataclass(frozen=True)
class ChLtOriginRegistryReferenceConfig:
    """Immutable startup configuration for the test-only authority."""

    database_path: Path
    request_public_key_path: Path
    schedule_public_key_path: Path
    trusted_time_public_key_path: Path
    registry_private_key_path: Path
    protocol_payload: bytes
    schedule_manifest_payload: bytes
    artifact_loader: Callable[[str], bytes]
    sqlite_timeout_seconds: float = 30.0
    head_observation_ttl: timedelta = timedelta(minutes=1)


class ChLtOriginRegistryReferenceAuthority:
    """Linearizable SQLite drill for selected protocol-v2 semantics."""

    NON_PRODUCTION = True
    COUNTABLE_AUTHORITY = False

    def __init__(
        self,
        config: ChLtOriginRegistryReferenceConfig,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = _validate_config(config)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._request_key = _load_public_key(self._config.request_public_key_path)
        self._schedule_key = _load_public_key(self._config.schedule_public_key_path)
        self._time_key = _load_public_key(self._config.trusted_time_public_key_path)
        self._registry_key = _load_private_key(self._config.registry_private_key_path)
        self._key_ids = {
            "request": _public_key_id(self._request_key),
            "schedule": _public_key_id(self._schedule_key),
            "trusted_time": _public_key_id(self._time_key),
            "registry": _public_key_id(self._registry_key.public_key()),
        }
        if len(set(self._key_ids.values())) != 4:
            raise ChLtOriginRegistryReferenceError(
                "origin request, schedule, time, and registry signer roles must be disjoint"
            )
        self._protocol = _strict_mapping(
            self._config.protocol_payload, label="origin registry protocol"
        )
        if (
            hashlib.sha256(self._config.protocol_payload).hexdigest() != PROTOCOL_SHA256
            or self._protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION
            or self._protocol.get("protocol_id") != PROTOCOL_ID
        ):
            raise ChLtOriginRegistryReferenceError(
                "reference authority protocol bytes are not canonical protocol v2"
            )
        self._cadence_contract_sha256 = semantic_sha256(
            self._protocol.get("cadence_proposal")
        )
        self._schedule = verify_exact_schedule_manifest(
            self._config.schedule_manifest_payload,
            protocol_sha256=PROTOCOL_SHA256,
            trusted_public_key=self._schedule_key,
        )
        self._initialize_database()

    def compare_and_append(self, request_payload: bytes) -> bytes:
        """Append one exact verified request or return an exact idempotent retry."""

        try:
            request = verify_origin_registration_request(
                request_payload,
                protocol_payload=self._config.protocol_payload,
                schedule_manifest_payload=self._config.schedule_manifest_payload,
                request_public_key=self._request_key,
                schedule_public_key=self._schedule_key,
                trusted_time_public_key=self._time_key,
                artifact_loader=self._config.artifact_loader,
            )
            return self._compare_verified(request_payload, request)
        except (ChLtOriginRegistryReferenceError, OSError, ValueError) as exc:
            self._record_rejection(request_payload, exc)
            if isinstance(exc, ChLtOriginRegistryReferenceError):
                raise
            raise ChLtOriginRegistryReferenceError(
                "origin registration request verification failed"
            ) from exc

    def _compare_verified(
        self, request_payload: bytes, request: Mapping[str, object]
    ) -> bytes:
        request_id = str(request["request_id"])
        operation_id = str(request["registration_operation_id"])
        request_sha256 = hashlib.sha256(request_payload).hexdigest()
        with self._transaction(immediate=True) as connection:
            existing = connection.execute(
                "SELECT request_payload, receipt_payload FROM receipts WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if existing is not None:
                if bytes(existing[0]) != request_payload:
                    raise ChLtOriginRegistryReferenceConflict(
                        "request_id was already committed with different exact bytes"
                    )
                return bytes(existing[1])

            for field, column in (
                ("registration_operation_id", "operation_id"),
                ("cadence_slot_id", "cadence_slot_id"),
                ("origin_id", "origin_id"),
                ("origin_as_of_utc", "origin_as_of_utc"),
            ):
                conflict = connection.execute(
                    f"SELECT request_id FROM receipts WHERE {column} = ?",  # noqa: S608
                    (str(request[field]),),
                ).fetchone()
                if conflict is not None:
                    raise ChLtOriginRegistryReferenceConflict(
                        f"origin registry uniqueness conflict: {field}"
                    )

            head = connection.execute(
                "SELECT sequence, receipt_id FROM receipts ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            current_sequence = 0 if head is None else int(head[0])
            current_receipt_id = None if head is None else str(head[1])
            if (
                request["expected_sequence"] != current_sequence + 1
                or request["expected_previous_receipt_id"] != current_receipt_id
            ):
                raise ChLtOriginRegistryReferenceConflict(
                    "compare-and-append predecessor is not the current registry HEAD"
                )

            committed_at = _canonical_clock(self._clock())
            origin_at = _utc(str(request["origin_as_of_utc"]), label="origin_as_of_utc")
            deadline = _utc(
                str(request["latest_external_registry_commit_utc"]),
                label="latest_external_registry_commit_utc",
            )
            first_target = _utc(
                str(request["first_target_delivery_start_utc"]),
                label="first_target_delivery_start_utc",
            )
            if not origin_at <= committed_at <= deadline < first_target:
                raise ChLtOriginRegistryReferenceConflict(
                    "registry trusted commit time is outside the frozen admission window"
                )

            receipt = sign_origin_registration_receipt(
                {
                    "schema_version": RECEIPT_SCHEMA,
                    "protocol_sha256": PROTOCOL_SHA256,
                    "protocol_id": PROTOCOL_ID,
                    "registry_domain_sha256": REGISTRY_DOMAIN_SHA256,
                    "sequence": current_sequence + 1,
                    "previous_receipt_id": current_receipt_id,
                    "registration_operation_id": operation_id,
                    "request_id": request_id,
                    "request_sha256": request_sha256,
                    "cadence_slot_id": request["cadence_slot_id"],
                    "exact_schedule_entry_id": request["exact_schedule_entry_id"],
                    "exact_schedule_entry_sha256": request[
                        "exact_schedule_entry_sha256"
                    ],
                    "capture_window_open_utc": request["capture_window_open_utc"],
                    "capture_window_close_utc": request["capture_window_close_utc"],
                    "latest_external_registry_commit_utc": request[
                        "latest_external_registry_commit_utc"
                    ],
                    "origin_id": request["origin_id"],
                    "origin_as_of_utc": request["origin_as_of_utc"],
                    "committed_at_utc": _render_utc(committed_at),
                    "countable_prospective_origin": False,
                },
                private_key=self._registry_key,
            )
            receipt_payload = canonical_json(receipt)
            connection.execute(
                """
                INSERT INTO receipts (
                    sequence, receipt_id, previous_receipt_id, operation_id,
                    request_id, request_sha256, cadence_slot_id, origin_id,
                    origin_as_of_utc, request_payload, receipt_payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt["sequence"],
                    receipt["receipt_id"],
                    receipt["previous_receipt_id"],
                    operation_id,
                    request_id,
                    request_sha256,
                    request["cadence_slot_id"],
                    request["origin_id"],
                    request["origin_as_of_utc"],
                    request_payload,
                    receipt_payload,
                ),
            )
            return receipt_payload

    def get_head(self) -> bytes | None:
        with self._transaction(immediate=False) as connection:
            row = connection.execute(
                "SELECT receipt_payload FROM receipts ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
        return None if row is None else bytes(row[0])

    def lookup_operation(
        self,
        operation_id: str,
        *,
        expected_request_payload: bytes,
    ) -> bytes:
        canonical_operation = _canonical_uuid(operation_id, label="operation_id")
        with self._transaction(immediate=False) as connection:
            row = connection.execute(
                "SELECT request_payload, receipt_payload FROM receipts WHERE operation_id = ?",
                (canonical_operation,),
            ).fetchone()
        if row is None:
            raise ChLtOriginRegistryReferenceError("registration operation is unknown")
        if bytes(row[0]) != expected_request_payload:
            raise ChLtOriginRegistryReferenceConflict(
                "operation lookup request differs from committed exact bytes"
            )
        return bytes(row[1])

    def observe_head(self, *, challenge_nonce: str) -> bytes:
        if not re.fullmatch(r"[0-9a-f]{64}", challenge_nonce):
            raise ChLtOriginRegistryReferenceError("head challenge nonce is invalid")
        with self._transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT receipt_payload FROM receipts ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            if row is None:
                raise ChLtOriginRegistryReferenceError("origin registry HEAD is empty")
            receipt_payload = bytes(row[0])
            receipt = _strict_canonical_mapping(receipt_payload, label="origin receipt")
            observed_at = _canonical_clock(self._clock())
            observation = sign_reference_head_observation(
                {
                    "schema_version": HEAD_OBSERVATION_SCHEMA,
                    "registry_domain_sha256": REGISTRY_DOMAIN_SHA256,
                    "protocol_sha256": PROTOCOL_SHA256,
                    "sequence": receipt["sequence"],
                    "receipt_id": receipt["receipt_id"],
                    "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
                    "challenge_nonce": challenge_nonce,
                    "observed_at_utc": _render_utc(observed_at),
                    "expires_at_utc": _render_utc(
                        observed_at + self._config.head_observation_ttl
                    ),
                    "reference_authority_classification": (
                        REFERENCE_AUTHORITY_CLASSIFICATION
                    ),
                },
                private_key=self._registry_key,
            )
            payload = canonical_json(observation)
            try:
                connection.execute(
                    """
                    INSERT INTO observations (
                        observation_id, receipt_id, challenge_nonce, observation_payload
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        observation["observation_id"],
                        observation["receipt_id"],
                        challenge_nonce,
                        payload,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ChLtOriginRegistryReferenceConflict(
                    "head challenge nonce was already consumed"
                ) from exc
            return payload

    def rejection_count(self) -> int:
        with self._transaction(immediate=False) as connection:
            row = connection.execute("SELECT COUNT(*) FROM rejected_operations").fetchone()
        return int(row[0])

    def _record_rejection(self, request_payload: bytes, error: Exception) -> None:
        request_sha256 = hashlib.sha256(request_payload).hexdigest()
        operation_id: str | None = None
        request_id: str | None = None
        try:
            document = _strict_canonical_mapping(request_payload, label="rejected request")
            _verify_request_signature_and_id(
                document,
                request_public_key=self._request_key,
            )
            operation_id = _canonical_uuid(
                document.get("registration_operation_id"),
                label="registration_operation_id",
            )
            request_id = str(document["request_id"])
        except ChLtOriginRegistryReferenceError:
            pass
        error_type = type(error).__name__
        error_code = _rejection_error_code(error)
        rejection_id = _domain_hash(
            b"FMV_CH_LT_ORIGIN_REJECTION_REFERENCE_V2",
            {
                "request_sha256": request_sha256,
                "error_type": error_type,
                "error_code": error_code,
            },
        )
        with self._transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO rejected_operations (
                    rejection_id, request_sha256, operation_id, request_id,
                    error_type, error_code, request_size_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rejection_id,
                    request_sha256,
                    operation_id,
                    request_id,
                    error_type,
                    error_code,
                    len(request_payload),
                ),
            )

    def _initialize_database(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS authority_metadata (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_version TEXT NOT NULL,
                    classification TEXT NOT NULL,
                    protocol_sha256 TEXT NOT NULL,
                    schedule_manifest_sha256 TEXT NOT NULL,
                    registry_domain_sha256 TEXT NOT NULL,
                    request_key_id TEXT NOT NULL,
                    schedule_key_id TEXT NOT NULL,
                    trusted_time_key_id TEXT NOT NULL,
                    registry_key_id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS receipts (
                    sequence INTEGER PRIMARY KEY CHECK (sequence >= 1),
                    receipt_id TEXT NOT NULL UNIQUE,
                    previous_receipt_id TEXT UNIQUE,
                    operation_id TEXT NOT NULL UNIQUE,
                    request_id TEXT NOT NULL UNIQUE,
                    request_sha256 TEXT NOT NULL UNIQUE,
                    cadence_slot_id TEXT NOT NULL UNIQUE,
                    origin_id TEXT NOT NULL UNIQUE,
                    origin_as_of_utc TEXT NOT NULL UNIQUE,
                    request_payload BLOB NOT NULL,
                    receipt_payload BLOB NOT NULL,
                    FOREIGN KEY (previous_receipt_id) REFERENCES receipts(receipt_id)
                );
                CREATE TABLE IF NOT EXISTS rejected_operations (
                    rejection_id TEXT PRIMARY KEY,
                    request_sha256 TEXT NOT NULL,
                    operation_id TEXT,
                    request_id TEXT,
                    error_type TEXT NOT NULL,
                    error_code TEXT NOT NULL,
                    request_size_bytes INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS observations (
                    observation_id TEXT PRIMARY KEY,
                    receipt_id TEXT NOT NULL,
                    challenge_nonce TEXT NOT NULL UNIQUE,
                    observation_payload BLOB NOT NULL,
                    FOREIGN KEY (receipt_id) REFERENCES receipts(receipt_id)
                );
                """
            )
            expected = (
                REFERENCE_SCHEMA_VERSION,
                REFERENCE_AUTHORITY_CLASSIFICATION,
                PROTOCOL_SHA256,
                hashlib.sha256(self._config.schedule_manifest_payload).hexdigest(),
                REGISTRY_DOMAIN_SHA256,
                self._key_ids["request"],
                self._key_ids["schedule"],
                self._key_ids["trusted_time"],
                self._key_ids["registry"],
            )
            row = connection.execute(
                """
                SELECT schema_version, classification, protocol_sha256,
                       schedule_manifest_sha256, registry_domain_sha256,
                       request_key_id, schedule_key_id, trusted_time_key_id,
                       registry_key_id
                FROM authority_metadata WHERE singleton = 1
                """
            ).fetchone()
            user_version_row = connection.execute("PRAGMA user_version").fetchone()
            user_version = 0 if user_version_row is None else int(user_version_row[0])
            if row is None:
                if user_version != 0:
                    raise ChLtOriginRegistryReferenceError(
                        "reference registry schema version has no matching identity"
                    )
                connection.execute(
                    """
                    INSERT INTO authority_metadata (
                        singleton, schema_version, classification, protocol_sha256,
                        schedule_manifest_sha256, registry_domain_sha256,
                        request_key_id, schedule_key_id, trusted_time_key_id,
                        registry_key_id
                    ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    expected,
                )
                connection.execute(f"PRAGMA user_version = {_SQLITE_USER_VERSION}")
            elif tuple(row) != expected or user_version != _SQLITE_USER_VERSION:
                raise ChLtOriginRegistryReferenceError(
                    "reference registry startup identity differs from persisted identity"
                )
            self._verify_persisted_chain(connection)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _verify_persisted_chain(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT sequence, receipt_id, previous_receipt_id, operation_id,
                   request_id, request_sha256, cadence_slot_id, origin_id,
                   origin_as_of_utc, request_payload, receipt_payload
            FROM receipts ORDER BY sequence
            """
        ).fetchall()
        previous_receipt_id: str | None = None
        for expected_sequence, row in enumerate(rows, start=1):
            (
                sequence,
                receipt_id,
                stored_previous,
                operation_id,
                request_id,
                request_sha256,
                cadence_slot_id,
                origin_id,
                origin_as_of_utc,
                request_payload,
                receipt_payload,
            ) = row
            if sequence != expected_sequence or stored_previous != previous_receipt_id:
                raise ChLtOriginRegistryReferenceError(
                    "persisted origin receipt chain is not contiguous"
                )
            exact_request = bytes(request_payload)
            exact_receipt = bytes(receipt_payload)
            request = verify_origin_registration_request(
                exact_request,
                protocol_payload=self._config.protocol_payload,
                schedule_manifest_payload=self._config.schedule_manifest_payload,
                request_public_key=self._request_key,
                schedule_public_key=self._schedule_key,
                trusted_time_public_key=self._time_key,
                artifact_loader=self._config.artifact_loader,
            )
            verified = verify_origin_registration_receipt(
                exact_request,
                exact_receipt,
                request_public_key=self._request_key,
                registry_public_key=self._registry_key.public_key(),
            )
            expected_row = (
                verified["sequence"],
                verified["receipt_id"],
                verified["previous_receipt_id"],
                request["registration_operation_id"],
                request["request_id"],
                hashlib.sha256(exact_request).hexdigest(),
                request["cadence_slot_id"],
                request["origin_id"],
                request["origin_as_of_utc"],
            )
            if tuple(row[:9]) != expected_row:
                raise ChLtOriginRegistryReferenceError(
                    "persisted origin receipt row differs from signed bytes"
                )
            if request_sha256 != hashlib.sha256(exact_request).hexdigest():
                raise ChLtOriginRegistryReferenceError(
                    "persisted origin request hash differs from exact bytes"
                )
            previous_receipt_id = str(receipt_id)

    @contextmanager
    def _transaction(self, *, immediate: bool) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        try:
            assert_absolute_path_has_no_links(self._config.database_path)
            for selected in _sqlite_paths(self._config.database_path):
                _assert_single_link_regular_if_exists(selected)
        except ValueError as exc:
            raise ChLtOriginRegistryReferenceError(
                "reference database path contains a link or reparse point"
            ) from exc
        connection = sqlite3.connect(
            self._config.database_path,
            timeout=self._config.sqlite_timeout_seconds,
            isolation_level=None,
        )
        try:
            timeout_ms = int(self._config.sqlite_timeout_seconds * 1000)
            connection.execute(f"PRAGMA busy_timeout = {timeout_ms}")
            connection.execute("PRAGMA foreign_keys = ON")
            journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()
            connection.execute("PRAGMA synchronous = FULL")
            if (
                journal_mode is None
                or str(journal_mode[0]).lower() != "wal"
                or connection.execute("PRAGMA foreign_keys").fetchone() != (1,)
                or connection.execute("PRAGMA synchronous").fetchone() != (2,)
                or connection.execute("PRAGMA busy_timeout").fetchone() != (timeout_ms,)
            ):
                raise ChLtOriginRegistryReferenceError(
                    "reference SQLite safety PRAGMAs were not applied"
                )
            for selected in _sqlite_paths(self._config.database_path):
                _assert_single_link_regular_if_exists(selected)
        except BaseException:
            connection.close()
            raise
        return connection


def sign_exact_schedule_entry(
    core: Mapping[str, object], *, private_key: Ed25519PrivateKey
) -> dict[str, object]:
    expected = _SCHEDULE_ENTRY_FIELDS - {"schedule_entry_id", "signature"}
    _exact_fields(core, expected, label="schedule entry core")
    return _sign_identity_document(
        core,
        identity_field="schedule_entry_id",
        signature_field="signature",
        private_key=private_key,
        domain=b"FMV_CH_LT_ORIGIN_SCHEDULE_ENTRY_V2",
    )


def sign_trusted_time_receipt(
    core: Mapping[str, object], *, private_key: Ed25519PrivateKey
) -> dict[str, object]:
    expected = {
        "schema_version",
        "subject_utc",
        "issued_at_utc",
        "purpose",
        "nonce",
    }
    _exact_fields(core, expected, label="trusted time core")
    if core.get("schema_version") != TRUSTED_TIME_RECEIPT_SCHEMA:
        raise ChLtOriginRegistryReferenceError("trusted time schema is invalid")
    return _sign_identity_document(
        core,
        identity_field="receipt_id",
        signature_field="signature",
        private_key=private_key,
        domain=b"FMV_CH_LT_ORIGIN_TRUSTED_TIME_REFERENCE_V1",
    )


def sign_origin_registration_request(
    core: Mapping[str, object], *, private_key: Ed25519PrivateKey
) -> dict[str, object]:
    _exact_fields(core, _REQUEST_CORE_FIELDS, label="origin request core")
    if core.get("schema_version") != REQUEST_SCHEMA:
        raise ChLtOriginRegistryReferenceError("origin request schema is invalid")
    _validate_no_float(core, label="origin request")
    request_id = _domain_hash(
        b"FMV_CH_LT_ORIGIN_REGISTRATION_REQUEST_V2", core
    )
    payload = {**core, "request_id": request_id}
    payload["request_signature"] = _signature(private_key, canonical_json(payload))
    _exact_fields(payload, _REQUEST_FIELDS, label="signed origin request")
    return payload


def sign_origin_registration_receipt(
    core: Mapping[str, object], *, private_key: Ed25519PrivateKey
) -> dict[str, object]:
    expected = _RECEIPT_FIELDS - {"receipt_id", "registry_signature"}
    _exact_fields(core, expected, label="origin receipt core")
    if (
        core.get("schema_version") != RECEIPT_SCHEMA
        or core.get("registry_domain_sha256") != REGISTRY_DOMAIN_SHA256
        or core.get("countable_prospective_origin") is not False
    ):
        raise ChLtOriginRegistryReferenceError(
            "reference receipt identity/countability is invalid"
        )
    return _sign_identity_document(
        core,
        identity_field="receipt_id",
        signature_field="registry_signature",
        private_key=private_key,
        domain=_REFERENCE_RECEIPT_DOMAIN,
    )


def sign_reference_head_observation(
    core: Mapping[str, object], *, private_key: Ed25519PrivateKey
) -> dict[str, object]:
    expected = _HEAD_OBSERVATION_FIELDS - {"observation_id", "registry_signature"}
    _exact_fields(core, expected, label="origin HEAD observation core")
    return _sign_identity_document(
        core,
        identity_field="observation_id",
        signature_field="registry_signature",
        private_key=private_key,
        domain=b"FMV_CH_LT_ORIGIN_HEAD_OBSERVATION_REFERENCE_V1",
    )


def verify_origin_registration_request(
    request_payload: bytes,
    *,
    protocol_payload: bytes,
    schedule_manifest_payload: bytes,
    request_public_key: Ed25519PublicKey,
    schedule_public_key: Ed25519PublicKey,
    trusted_time_public_key: Ed25519PublicKey,
    artifact_loader: Callable[[str], bytes],
) -> dict[str, object]:
    request = _strict_canonical_mapping(request_payload, label="origin request")
    _verify_request_signature_and_id(request, request_public_key=request_public_key)

    protocol = _strict_mapping(protocol_payload, label="origin protocol")
    if (
        hashlib.sha256(protocol_payload).hexdigest() != PROTOCOL_SHA256
        or protocol.get("protocol_id") != PROTOCOL_ID
        or request.get("protocol_sha256") != PROTOCOL_SHA256
        or request.get("protocol_id") != PROTOCOL_ID
        or request.get("registry_domain_sha256") != REGISTRY_DOMAIN_SHA256
    ):
        raise ChLtOriginRegistryReferenceError("origin request protocol/domain mismatch")
    cadence_sha256 = semantic_sha256(protocol.get("cadence_proposal"))
    if request.get("cadence_contract_sha256") != cadence_sha256:
        raise ChLtOriginRegistryReferenceError("origin cadence contract mismatch")

    schedule = verify_exact_schedule_manifest(
        schedule_manifest_payload,
        protocol_sha256=PROTOCOL_SHA256,
        trusted_public_key=schedule_public_key,
    )
    if request.get("exact_schedule_manifest_sha256") != hashlib.sha256(
        schedule_manifest_payload
    ).hexdigest():
        raise ChLtOriginRegistryReferenceError("origin schedule manifest mismatch")
    entries = schedule["entries"]
    matching = [
        entry
        for entry in entries
        if entry["cadence_slot_id"] == request.get("cadence_slot_id")
    ]
    if len(matching) != 1:
        raise ChLtOriginRegistryReferenceError(
            "cadence slot does not select exactly one schedule entry"
        )
    entry = matching[0]
    entry_payload = canonical_json(entry)
    for request_field, entry_field in (
        ("exact_schedule_entry_id", "schedule_entry_id"),
        ("capture_window_open_utc", "capture_window_open_utc"),
        ("capture_window_close_utc", "capture_window_close_utc"),
        ("latest_external_registry_commit_utc", "latest_external_registry_commit_utc"),
    ):
        if request.get(request_field) != entry.get(entry_field):
            raise ChLtOriginRegistryReferenceError(
                f"origin request schedule binding mismatch: {request_field}"
            )
    if request.get("exact_schedule_entry_sha256") != hashlib.sha256(
        entry_payload
    ).hexdigest():
        raise ChLtOriginRegistryReferenceError("origin schedule entry hash mismatch")

    expected_sequence = request.get("expected_sequence")
    if (
        type(expected_sequence) is not int
        or expected_sequence < 1
        or expected_sequence > _MAX_SEQUENCE
    ):
        raise ChLtOriginRegistryReferenceError("origin expected sequence is invalid")
    previous = request.get("expected_previous_receipt_id")
    if previous is not None:
        _require_sha256(previous, label="expected_previous_receipt_id")
    _canonical_uuid(
        request.get("registration_operation_id"),
        label="registration_operation_id",
    )
    if not isinstance(request.get("cadence_slot_id"), str) or not _CADENCE_SLOT.fullmatch(
        str(request["cadence_slot_id"])
    ):
        raise ChLtOriginRegistryReferenceError("cadence_slot_id is invalid")

    open_at = _utc(str(request["capture_window_open_utc"]), label="capture open")
    close_at = _utc(str(request["capture_window_close_utc"]), label="capture close")
    origin_at = _utc(str(request["origin_as_of_utc"]), label="origin as-of")
    trusted_at = _utc(str(request["trusted_origin_time_utc"]), label="trusted origin")
    deadline = _utc(
        str(request["latest_external_registry_commit_utc"]), label="registry deadline"
    )
    first_target = _utc(
        str(request["first_target_delivery_start_utc"]), label="first target"
    )
    if not open_at <= origin_at == trusted_at <= close_at <= deadline < first_target:
        raise ChLtOriginRegistryReferenceError(
            "origin request chronology violates its frozen windows"
        )

    artifact_payloads: dict[str, bytes] = {}
    for field in _ARTIFACT_SHA_FIELDS:
        digest = request.get(field)
        _require_sha256(digest, label=field)
        try:
            payload = artifact_loader(str(digest))
        except Exception as exc:
            raise ChLtOriginRegistryReferenceError(
                f"origin bound artifact is unavailable: {field}"
            ) from exc
        if (
            not isinstance(payload, bytes)
            or len(payload) > _MAX_ARTIFACT_BYTES
            or hashlib.sha256(payload).hexdigest() != digest
        ):
            raise ChLtOriginRegistryReferenceError(
                f"origin bound artifact hash mismatch: {field}"
            )
        artifact_payloads[field] = payload
    _require_sha256(request.get("project_source_revision"), label="project source revision")
    _require_sha256(
        request.get("origin_target_mask_inventory_id"), label="origin inventory id"
    )
    expected_origin_id = compute_origin_id(
        origin_as_of_utc=str(request["origin_as_of_utc"]),
        inventory_id=str(request["origin_target_mask_inventory_id"]),
        inventory_sha256=str(request["origin_target_mask_inventory_sha256"]),
    )
    if request.get("origin_id") != expected_origin_id:
        raise ChLtOriginRegistryReferenceError("origin_id binding mismatch")

    inventory_payload = artifact_payloads["origin_target_mask_inventory_sha256"]
    inventory_document = _strict_mapping(
        inventory_payload, label="origin target/mask inventory"
    )
    try:
        inventory_assessment = origin_inventory.assess_structural_inventory(
            inventory_document,
            document_bytes=inventory_payload,
        )
    except ValueError as exc:
        raise ChLtOriginRegistryReferenceError(
            "origin target/mask inventory semantics are invalid"
        ) from exc
    if (
        inventory_assessment.get("inventory_id")
        != request.get("origin_target_mask_inventory_id")
        or inventory_assessment.get("origin_as_of_utc")
        != request.get("origin_as_of_utc")
        or inventory_assessment.get("outcome_blind") is not True
        or inventory_assessment.get("outcomes_opened") is not False
        or inventory_assessment.get("t057_consumed") is not False
    ):
        raise ChLtOriginRegistryReferenceError(
            "origin target/mask inventory does not bind the outcome-blind request"
        )
    targets = inventory_document.get("targets")
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes, bytearray)):
        raise ChLtOriginRegistryReferenceError("origin target inventory is unavailable")
    delivery_starts = [
        str(target.get("delivery_start_utc"))
        for target in targets
        if isinstance(target, Mapping)
    ]
    if len(delivery_starts) != len(targets) or not delivery_starts:
        raise ChLtOriginRegistryReferenceError("origin target delivery starts are invalid")
    derived_first_target = min(
        delivery_starts,
        key=lambda value: _utc(value, label="inventory delivery start"),
    )
    if request.get("first_target_delivery_start_utc") != derived_first_target:
        raise ChLtOriginRegistryReferenceError(
            "first target delivery start is not derived from the bound inventory"
        )

    time_sha256 = request.get("trusted_origin_time_receipt_sha256")
    _require_sha256(time_sha256, label="trusted time receipt")
    try:
        time_payload = artifact_loader(str(time_sha256))
    except Exception as exc:
        raise ChLtOriginRegistryReferenceError(
            "trusted time receipt is unavailable"
        ) from exc
    if (
        not isinstance(time_payload, bytes)
        or len(time_payload) > _MAX_BYTES
        or hashlib.sha256(time_payload).hexdigest() != time_sha256
    ):
        raise ChLtOriginRegistryReferenceError("trusted time receipt hash mismatch")
    time_receipt = verify_trusted_time_receipt(
        time_payload, trusted_public_key=trusted_time_public_key
    )
    issued_at = _utc(str(time_receipt["issued_at_utc"]), label="time issued-at")
    if (
        time_receipt.get("subject_utc") != request.get("origin_as_of_utc")
        or time_receipt.get("purpose") != "CH_LT_ORIGIN_AS_OF_V2"
        or not origin_at <= issued_at <= close_at
    ):
        raise ChLtOriginRegistryReferenceError(
            "trusted time receipt does not attest the exact origin window"
        )
    for field in ("truth_opened", "future_holdout_consumed", "t057_consumed"):
        if request.get(field) is not False:
            raise ChLtOriginRegistryReferenceError(f"origin truth firewall failed: {field}")
    return dict(request)


def verify_origin_registration_receipt(
    request_payload: bytes,
    receipt_payload: bytes,
    *,
    request_public_key: Ed25519PublicKey,
    registry_public_key: Ed25519PublicKey,
) -> dict[str, object]:
    request = _strict_canonical_mapping(request_payload, label="origin request")
    receipt = _strict_canonical_mapping(receipt_payload, label="origin receipt")
    _verify_request_signature_and_id(request, request_public_key=request_public_key)
    _verify_receipt_signature_and_id(receipt, registry_public_key=registry_public_key)
    bindings = {
        "protocol_sha256": "protocol_sha256",
        "protocol_id": "protocol_id",
        "registry_domain_sha256": "registry_domain_sha256",
        "registration_operation_id": "registration_operation_id",
        "request_id": "request_id",
        "cadence_slot_id": "cadence_slot_id",
        "exact_schedule_entry_id": "exact_schedule_entry_id",
        "exact_schedule_entry_sha256": "exact_schedule_entry_sha256",
        "capture_window_open_utc": "capture_window_open_utc",
        "capture_window_close_utc": "capture_window_close_utc",
        "latest_external_registry_commit_utc": "latest_external_registry_commit_utc",
        "origin_id": "origin_id",
        "origin_as_of_utc": "origin_as_of_utc",
    }
    for receipt_field, request_field in bindings.items():
        if receipt.get(receipt_field) != request.get(request_field):
            raise ChLtOriginRegistryReferenceError(
                f"origin receipt/request binding mismatch: {receipt_field}"
            )
    if (
        receipt.get("sequence") != request.get("expected_sequence")
        or receipt.get("previous_receipt_id")
        != request.get("expected_previous_receipt_id")
    ):
        raise ChLtOriginRegistryReferenceError(
            "origin receipt/request predecessor binding mismatch"
        )
    if receipt.get("request_sha256") != hashlib.sha256(request_payload).hexdigest():
        raise ChLtOriginRegistryReferenceError("origin receipt request hash mismatch")
    committed = _utc(str(receipt["committed_at_utc"]), label="receipt committed-at")
    origin = _utc(str(receipt["origin_as_of_utc"]), label="receipt origin")
    deadline = _utc(
        str(receipt["latest_external_registry_commit_utc"]), label="receipt deadline"
    )
    first_target = _utc(
        str(request["first_target_delivery_start_utc"]), label="receipt first target"
    )
    if not origin <= committed <= deadline < first_target:
        raise ChLtOriginRegistryReferenceError("origin receipt chronology is invalid")
    if receipt.get("countable_prospective_origin") is not False:
        raise ChLtOriginRegistryReferenceError("origin receipt countability value is invalid")
    return {
        **receipt,
        "receipt_claimed_countable_prospective_origin": False,
        "reference_authority_classification": REFERENCE_AUTHORITY_CLASSIFICATION,
        "external_authority_verified": False,
        "countable_prospective_origin": False,
        "scientific_admission": False,
        "production_authorization": False,
        "promotion_gate": False,
    }


def verify_head_observation(
    observation_payload: bytes,
    *,
    expected_receipt_payload: bytes,
    expected_challenge_nonce: str,
    registry_public_key: Ed25519PublicKey,
    reference_time: datetime,
) -> dict[str, object]:
    observation = _strict_canonical_mapping(
        observation_payload, label="origin HEAD observation"
    )
    if not re.fullmatch(r"[0-9a-f]{64}", expected_challenge_nonce):
        raise ChLtOriginRegistryReferenceError("expected HEAD challenge nonce is invalid")
    _exact_fields(
        observation,
        _HEAD_OBSERVATION_FIELDS,
        label="origin HEAD observation",
    )
    signature = observation.get("registry_signature")
    unsigned = dict(observation)
    unsigned.pop("registry_signature", None)
    _verify_signature(signature, canonical_json(unsigned), registry_public_key)
    core = dict(unsigned)
    observation_id = core.pop("observation_id", None)
    if observation_id != _domain_hash(
        b"FMV_CH_LT_ORIGIN_HEAD_OBSERVATION_REFERENCE_V1", core
    ):
        raise ChLtOriginRegistryReferenceError("HEAD observation id mismatch")
    receipt = _strict_canonical_mapping(expected_receipt_payload, label="HEAD receipt")
    _verify_receipt_signature_and_id(
        receipt,
        registry_public_key=registry_public_key,
    )
    if (
        observation.get("schema_version") != HEAD_OBSERVATION_SCHEMA
        or observation.get("registry_domain_sha256") != REGISTRY_DOMAIN_SHA256
        or observation.get("protocol_sha256") != PROTOCOL_SHA256
        or observation.get("sequence") != receipt.get("sequence")
        or observation.get("receipt_id") != receipt.get("receipt_id")
        or observation.get("receipt_sha256")
        != hashlib.sha256(expected_receipt_payload).hexdigest()
        or observation.get("challenge_nonce") != expected_challenge_nonce
        or observation.get("reference_authority_classification")
        != REFERENCE_AUTHORITY_CLASSIFICATION
    ):
        raise ChLtOriginRegistryReferenceError("HEAD observation binding mismatch")
    observed_at = _utc(str(observation["observed_at_utc"]), label="HEAD observed-at")
    expires_at = _utc(str(observation["expires_at_utc"]), label="HEAD expires-at")
    committed_at = _utc(str(receipt["committed_at_utc"]), label="HEAD committed-at")
    selected_time = _canonical_clock(reference_time)
    if not (
        committed_at <= observed_at <= selected_time <= expires_at
        and timedelta(0) < expires_at - observed_at <= _MAX_HEAD_TTL
    ):
        raise ChLtOriginRegistryReferenceError("HEAD observation is not fresh")
    return {
        **observation,
        "external_authority_verified": False,
        "countable_prospective_origin": False,
        "scientific_admission": False,
        "production_authorization": False,
        "promotion_gate": False,
    }


def verify_exact_schedule_manifest(
    payload: bytes,
    *,
    protocol_sha256: str,
    trusted_public_key: Ed25519PublicKey,
) -> dict[str, object]:
    document = _strict_canonical_mapping(payload, label="exact origin schedule")
    _exact_fields(
        document,
        {"schema_version", "protocol_sha256", "entries"},
        label="exact origin schedule",
    )
    if (
        document.get("schema_version") != SCHEDULE_MANIFEST_SCHEMA
        or document.get("protocol_sha256") != protocol_sha256
    ):
        raise ChLtOriginRegistryReferenceError("exact schedule identity is invalid")
    entries = document.get("entries")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
        raise ChLtOriginRegistryReferenceError("exact schedule entries are invalid")
    selected: list[dict[str, object]] = []
    seen_slots: set[str] = set()
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise ChLtOriginRegistryReferenceError("exact schedule entry is invalid")
        entry = dict(raw)
        _exact_fields(entry, _SCHEDULE_ENTRY_FIELDS, label="exact schedule entry")
        signature = entry.pop("signature")
        _verify_signature(signature, canonical_json(entry), trusted_public_key)
        entry_id = entry.pop("schedule_entry_id", None)
        if entry_id != _domain_hash(b"FMV_CH_LT_ORIGIN_SCHEDULE_ENTRY_V2", entry):
            raise ChLtOriginRegistryReferenceError("exact schedule entry id mismatch")
        slot = str(raw.get("cadence_slot_id"))
        if not _CADENCE_SLOT.fullmatch(slot) or slot in seen_slots:
            raise ChLtOriginRegistryReferenceError("exact schedule slot is invalid or duplicate")
        seen_slots.add(slot)
        if raw.get("protocol_sha256") != protocol_sha256:
            raise ChLtOriginRegistryReferenceError("exact schedule protocol mismatch")
        open_at = _utc(str(raw["capture_window_open_utc"]), label="schedule open")
        close_at = _utc(str(raw["capture_window_close_utc"]), label="schedule close")
        deadline = _utc(
            str(raw["latest_external_registry_commit_utc"]), label="schedule deadline"
        )
        _utc(str(raw["eex_trading_day"] + "T00:00:00Z"), label="EEX trading day")
        if not open_at <= close_at <= deadline:
            raise ChLtOriginRegistryReferenceError("exact schedule chronology is invalid")
        _require_sha256(
            raw.get("official_eex_calendar_document_sha256"),
            label="official EEX calendar",
        )
        _require_sha256(
            raw.get("official_settlement_event_definition_sha256"),
            label="official settlement event",
        )
        selected.append(dict(raw))
    if not selected:
        raise ChLtOriginRegistryReferenceError("exact schedule is empty")
    return {**document, "entries": selected}


def verify_trusted_time_receipt(
    payload: bytes, *, trusted_public_key: Ed25519PublicKey
) -> dict[str, object]:
    document = _strict_canonical_mapping(payload, label="trusted time receipt")
    expected = {
        "schema_version",
        "subject_utc",
        "issued_at_utc",
        "purpose",
        "nonce",
        "receipt_id",
        "signature",
    }
    _exact_fields(document, expected, label="trusted time receipt")
    signature = document.get("signature")
    unsigned = dict(document)
    unsigned.pop("signature")
    _verify_signature(signature, canonical_json(unsigned), trusted_public_key)
    core = dict(unsigned)
    receipt_id = core.pop("receipt_id", None)
    if receipt_id != _domain_hash(
        b"FMV_CH_LT_ORIGIN_TRUSTED_TIME_REFERENCE_V1", core
    ):
        raise ChLtOriginRegistryReferenceError("trusted time receipt id mismatch")
    if document.get("schema_version") != TRUSTED_TIME_RECEIPT_SCHEMA:
        raise ChLtOriginRegistryReferenceError("trusted time schema is invalid")
    _utc(str(document["subject_utc"]), label="trusted time subject")
    _utc(str(document["issued_at_utc"]), label="trusted time issued-at")
    if not re.fullmatch(r"[0-9a-f]{64}", str(document.get("nonce"))):
        raise ChLtOriginRegistryReferenceError("trusted time nonce is invalid")
    return dict(document)


def compute_origin_id(
    *, origin_as_of_utc: str, inventory_id: str, inventory_sha256: str
) -> str:
    _utc(origin_as_of_utc, label="origin id as-of")
    _require_sha256(inventory_id, label="origin inventory id")
    _require_sha256(inventory_sha256, label="origin inventory SHA-256")
    return _domain_hash(
        b"FMV_CH_LT_ORIGIN_ID_V2",
        {
            "origin_as_of_utc": origin_as_of_utc,
            "origin_target_mask_inventory_id": inventory_id,
            "origin_target_mask_inventory_sha256": inventory_sha256,
        },
    )


def semantic_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def canonical_json(value: object) -> bytes:
    _validate_no_float(value, label="canonical JSON")
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ChLtOriginRegistryReferenceError("canonical JSON value is invalid") from exc


def _validate_config(
    config: ChLtOriginRegistryReferenceConfig,
) -> ChLtOriginRegistryReferenceConfig:
    database_path = Path(config.database_path).expanduser()
    if not database_path.is_absolute():
        raise ChLtOriginRegistryReferenceError("reference database path must be absolute")
    try:
        database_path = assert_absolute_path_has_no_links(database_path)
    except ValueError as exc:
        raise ChLtOriginRegistryReferenceError(
            "reference database path contains a link or reparse point"
        ) from exc
    if not database_path.parent.is_dir():
        raise ChLtOriginRegistryReferenceError(
            "reference database parent must already exist"
        )
    try:
        build_root = assert_absolute_path_has_no_links(_REFERENCE_BUILD_ROOT)
    except ValueError as exc:
        raise ChLtOriginRegistryReferenceError(
            "canonical reference build root is unavailable"
        ) from exc
    governed_paths = (
        database_path,
        Path(config.request_public_key_path).expanduser().absolute(),
        Path(config.schedule_public_key_path).expanduser().absolute(),
        Path(config.trusted_time_public_key_path).expanduser().absolute(),
        Path(config.registry_private_key_path).expanduser().absolute(),
    )
    if any(not _is_within(path, build_root) for path in governed_paths):
        raise ChLtOriginRegistryReferenceError(
            "reference database and trust material must remain below repo build/"
        )
    if (
        type(config.sqlite_timeout_seconds) not in (int, float)
        or not 0 < float(config.sqlite_timeout_seconds) <= 120
    ):
        raise ChLtOriginRegistryReferenceError("reference SQLite timeout is invalid")
    if not timedelta(0) < config.head_observation_ttl <= _MAX_HEAD_TTL:
        raise ChLtOriginRegistryReferenceError("reference HEAD TTL is invalid")
    if not callable(config.artifact_loader):
        raise ChLtOriginRegistryReferenceError("reference artifact loader is invalid")
    try:
        request_public = freeze_process_immutable_file(
            config.request_public_key_path,
            label="reference origin request public key",
        )
        schedule_public = freeze_process_immutable_file(
            config.schedule_public_key_path,
            label="reference origin schedule public key",
        )
        trusted_time_public = freeze_process_immutable_file(
            config.trusted_time_public_key_path,
            label="reference origin trusted-time public key",
        )
        registry_private = freeze_process_immutable_file(
            config.registry_private_key_path,
            label="reference origin registry private key",
        )
    except ValueError as exc:
        raise ChLtOriginRegistryReferenceError(
            "reference origin trust material cannot be frozen"
        ) from exc
    return ChLtOriginRegistryReferenceConfig(
        database_path=database_path,
        request_public_key_path=request_public,
        schedule_public_key_path=schedule_public,
        trusted_time_public_key_path=trusted_time_public,
        registry_private_key_path=registry_private,
        protocol_payload=bytes(config.protocol_payload),
        schedule_manifest_payload=bytes(config.schedule_manifest_payload),
        artifact_loader=config.artifact_loader,
        sqlite_timeout_seconds=float(config.sqlite_timeout_seconds),
        head_observation_ttl=config.head_observation_ttl,
    )


def _sign_identity_document(
    core: Mapping[str, object],
    *,
    identity_field: str,
    signature_field: str,
    private_key: Ed25519PrivateKey,
    domain: bytes,
) -> dict[str, object]:
    _validate_no_float(core, label="signed document")
    identity = _domain_hash(domain, core)
    payload = {**core, identity_field: identity}
    payload[signature_field] = _signature(private_key, canonical_json(payload))
    return payload


def _signature(private_key: Ed25519PrivateKey, payload: bytes) -> dict[str, str]:
    return {
        "algorithm": SIGNATURE_ALGORITHM,
        "key_id": _public_key_id(private_key.public_key()),
        "value_base64": base64.b64encode(private_key.sign(payload)).decode("ascii"),
    }


def _verify_request_signature_and_id(
    request: Mapping[str, object], *, request_public_key: Ed25519PublicKey
) -> None:
    _exact_fields(request, _REQUEST_FIELDS, label="origin request")
    signature = request.get("request_signature")
    unsigned = dict(request)
    unsigned.pop("request_signature")
    _verify_signature(signature, canonical_json(unsigned), request_public_key)
    core = dict(unsigned)
    request_id = core.pop("request_id", None)
    expected_id = _domain_hash(b"FMV_CH_LT_ORIGIN_REGISTRATION_REQUEST_V2", core)
    if request_id != expected_id:
        raise ChLtOriginRegistryReferenceError("origin request_id mismatch")
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise ChLtOriginRegistryReferenceError("origin request schema is invalid")
    _validate_no_float(request, label="origin request")


def _verify_receipt_signature_and_id(
    receipt: Mapping[str, object], *, registry_public_key: Ed25519PublicKey
) -> None:
    _exact_fields(receipt, _RECEIPT_FIELDS, label="origin receipt")
    signature = receipt.get("registry_signature")
    unsigned = dict(receipt)
    unsigned.pop("registry_signature")
    _verify_signature(signature, canonical_json(unsigned), registry_public_key)
    core = dict(unsigned)
    receipt_id = core.pop("receipt_id", None)
    expected_receipt_id = _domain_hash(
        _REFERENCE_RECEIPT_DOMAIN, core
    )
    if receipt_id != expected_receipt_id:
        raise ChLtOriginRegistryReferenceError("origin receipt_id mismatch")
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("protocol_sha256") != PROTOCOL_SHA256
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("registry_domain_sha256") != REGISTRY_DOMAIN_SHA256
    ):
        raise ChLtOriginRegistryReferenceError("origin receipt identity is invalid")
    sequence = receipt.get("sequence")
    if type(sequence) is not int or sequence < 1 or sequence > _MAX_SEQUENCE:
        raise ChLtOriginRegistryReferenceError("origin receipt sequence is invalid")
    previous = receipt.get("previous_receipt_id")
    if sequence == 1:
        if previous is not None:
            raise ChLtOriginRegistryReferenceError("origin genesis predecessor is invalid")
    else:
        _require_sha256(previous, label="previous_receipt_id")


def _verify_signature(
    signature: object, payload: bytes, trusted_public_key: Ed25519PublicKey
) -> None:
    if not isinstance(signature, Mapping) or set(signature) != {
        "algorithm",
        "key_id",
        "value_base64",
    }:
        raise ChLtOriginRegistryReferenceError("origin signature block is invalid")
    if (
        signature.get("algorithm") != SIGNATURE_ALGORITHM
        or signature.get("key_id") != _public_key_id(trusted_public_key)
    ):
        raise ChLtOriginRegistryReferenceError("origin signature trust binding is invalid")
    try:
        value = base64.b64decode(str(signature.get("value_base64")), validate=True)
        trusted_public_key.verify(value, payload)
    except (TypeError, ValueError, InvalidSignature) as exc:
        raise ChLtOriginRegistryReferenceError("origin signature is invalid") from exc


def _domain_hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + b"\x00" + canonical_json(value)).hexdigest()


def _strict_canonical_mapping(payload: bytes, *, label: str) -> dict[str, object]:
    parsed = _strict_mapping(payload, label=label)
    if canonical_json(parsed) != payload:
        raise ChLtOriginRegistryReferenceError(f"{label} is not exact canonical JSON")
    return parsed


def _strict_mapping(payload: bytes, *, label: str) -> dict[str, object]:
    if len(payload) > _MAX_BYTES:
        raise ChLtOriginRegistryReferenceError(f"{label} exceeds byte budget")
    try:
        parsed = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChLtOriginRegistryReferenceError(f"{label} is not strict JSON") from exc
    if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
        raise ChLtOriginRegistryReferenceError(f"{label} must be a JSON object")
    return parsed


def _exact_fields(value: Mapping[str, object], expected: set[str] | frozenset[str], *, label: str) -> None:
    if set(value) != set(expected):
        raise ChLtOriginRegistryReferenceError(f"{label} fields are not exact")


def _validate_no_float(value: object, *, label: str) -> None:
    if isinstance(value, float):
        raise ChLtOriginRegistryReferenceError(f"{label} contains a float")
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ChLtOriginRegistryReferenceError(f"{label} contains a non-string key")
        for child in value.values():
            _validate_no_float(child, label=label)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _validate_no_float(child, label=label)


def _load_private_key(path: Path) -> Ed25519PrivateKey:
    try:
        payload = read_stable_single_link_file(path.absolute(), label="origin private key")
        key = serialization.load_pem_private_key(payload, password=None)
    except (OSError, TypeError, ValueError) as exc:
        raise ChLtOriginRegistryReferenceError("origin private key is invalid") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ChLtOriginRegistryReferenceError("origin private key is not Ed25519")
    return key


def _load_public_key(path: Path) -> Ed25519PublicKey:
    try:
        payload = read_stable_single_link_file(path.absolute(), label="origin public key")
        key = serialization.load_pem_public_key(payload)
    except (OSError, TypeError, ValueError) as exc:
        raise ChLtOriginRegistryReferenceError("origin public key is invalid") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ChLtOriginRegistryReferenceError("origin public key is not Ed25519")
    return key


def _public_key_id(key: Ed25519PublicKey) -> str:
    raw = key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def _sqlite_paths(database_path: Path) -> tuple[Path, ...]:
    raw = str(database_path)
    return (
        database_path,
        Path(raw + "-journal"),
        Path(raw + "-wal"),
        Path(raw + "-shm"),
    )


def _assert_single_link_regular_if_exists(path: Path) -> None:
    try:
        metadata = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    if (
        not stat.S_ISREG(metadata.st_mode)
        or int(metadata.st_nlink) != 1
        or (os.name == "nt" and attributes & 0x400)
    ):
        raise ValueError(f"reference SQLite path is not a single-link regular file: {path}")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _utc(value: str, *, label: str) -> datetime:
    if not _UTC.fullmatch(value):
        raise ChLtOriginRegistryReferenceError(f"{label} is not canonical UTC")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ChLtOriginRegistryReferenceError(f"{label} is invalid") from exc


def _render_utc(value: datetime) -> str:
    selected = _canonical_clock(value)
    if selected.microsecond:
        raise ChLtOriginRegistryReferenceError("reference clock must have whole seconds")
    return selected.strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_clock(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ChLtOriginRegistryReferenceError("reference clock must be UTC")
    return value.astimezone(timezone.utc)


def _canonical_uuid(value: object, *, label: str) -> str:
    raw = str(value or "").lower()
    try:
        selected = str(uuid.UUID(raw))
    except ValueError as exc:
        raise ChLtOriginRegistryReferenceError(f"{label} is invalid") from exc
    if raw != selected:
        raise ChLtOriginRegistryReferenceError(f"{label} is not canonical")
    return selected


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ChLtOriginRegistryReferenceError(f"{label} SHA-256 is invalid")


__all__ = [
    "HEAD_OBSERVATION_SCHEMA",
    "LOGICAL_DOMAIN",
    "PROTOCOL_LOGICAL_DOMAIN",
    "PROTOCOL_REGISTRY_DOMAIN_SHA256",
    "REFERENCE_AUTHORITY_CLASSIFICATION",
    "REGISTRY_DOMAIN_SHA256",
    "REQUEST_SCHEMA",
    "RECEIPT_SCHEMA",
    "SCHEDULE_MANIFEST_SCHEMA",
    "TRUSTED_TIME_RECEIPT_SCHEMA",
    "ChLtOriginRegistryReferenceAuthority",
    "ChLtOriginRegistryReferenceConfig",
    "ChLtOriginRegistryReferenceConflict",
    "ChLtOriginRegistryReferenceError",
    "canonical_json",
    "compute_origin_id",
    "semantic_sha256",
    "sign_exact_schedule_entry",
    "sign_origin_registration_request",
    "sign_origin_registration_receipt",
    "sign_reference_head_observation",
    "sign_trusted_time_receipt",
    "verify_exact_schedule_manifest",
    "verify_head_observation",
    "verify_origin_registration_receipt",
    "verify_origin_registration_request",
    "verify_trusted_time_receipt",
]
