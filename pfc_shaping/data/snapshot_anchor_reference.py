"""NON-PRODUCTION SQLite reference authority for the D147 external CAS protocol.

This implementation exists only to exercise the protocol and attack matrix. A
SQLite file controlled by the model operator is not an independent production
authority, WORM storage, or a substitute for the IT-owned service.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from pfc_shaping.data.snapshot_anchor_signing import (
    sign_snapshot_anchor_head_observation,
    sign_snapshot_anchor_receipt,
)
from pfc_shaping.data.snapshot_publication_contract import (
    BOOTSTRAP_TRANSITION_TYPE,
    PUBLICATION_HEAD_OBSERVATION_SCHEMA,
    PUBLICATION_RECEIPT_SCHEMA,
    assert_disjoint_publication_authority_ids,
    publication_domain_sha256,
    publication_keyring_key_ids,
    publication_private_key_id,
    publication_signature_key_id,
    publication_trusted_key_ids,
    verify_snapshot_bootstrap_intent_authorization,
    verify_snapshot_publication_intent,
)
from pfc_shaping.data.snapshot_publication_state import (
    canonical_json,
    external_head_challenge_sha256,
    pointer_mapping_from_external_receipt,
)
from pfc_shaping.path_safety import (
    freeze_process_immutable_directory,
    freeze_process_immutable_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json

REFERENCE_AUTHORITY_CLASSIFICATION = "NON_PRODUCTION_TEST_ONLY"
_SCHEMA_VERSION = "d147_snapshot_anchor_reference.v2"
_MAX_HEAD_TTL = timedelta(minutes=5)
GENESIS_POLICY_BOOTSTRAP_REQUIRED = "BOOTSTRAP_REQUIRED"
GENESIS_POLICY_ALLOW_PUBLISH_FOR_TESTS = "ALLOW_PUBLISH_FOR_NON_PRODUCTION_TESTS"


class SnapshotAnchorReferenceError(RuntimeError):
    """Raised when the reference authority cannot perform an operation."""


class SnapshotAnchorReferenceConflict(SnapshotAnchorReferenceError):
    """Raised when compare-and-append or operation idempotency conflicts."""


@dataclass(frozen=True)
class SnapshotAnchorReferenceConfig:
    """Configuration for the test-only reference authority."""

    database_path: Path
    request_public_key_path: Path
    anchor_private_key_path: Path
    request_public_key_dir: Path | None = None
    anchor_public_key_dir: Path | None = None
    bootstrap_public_key_path: Path | None = None
    bootstrap_public_key_dir: Path | None = None
    genesis_policy: str = GENESIS_POLICY_BOOTSTRAP_REQUIRED
    head_observation_ttl: timedelta = timedelta(minutes=1)
    sqlite_timeout_seconds: float = 30.0
    authority_key_ids: tuple[tuple[str, tuple[str, ...]], ...] = field(
        default=(),
        repr=False,
    )


class SnapshotAnchorReferenceAuthority:
    """Durable, linearizable reference implementation of the D147 CAS API."""

    NON_PRODUCTION = True

    def __init__(
        self,
        config: SnapshotAnchorReferenceConfig,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = _validate_config(config)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._domain_sha256 = publication_domain_sha256()
        self._anchor_key_id = publication_private_key_id(self._config.anchor_private_key_path)
        self._initialize_database()

    def compare_and_append(self, intent_payload: bytes) -> bytes:
        """Atomically append an exact signed intent or return its exact retry."""

        intent = _strict_canonical_mapping(intent_payload, label="publication intent")
        operation_id = _canonical_operation_id(intent.get("operation_id"))
        with self._transaction(immediate=True) as connection:
            existing = connection.execute(
                "SELECT intent_payload, receipt_payload FROM receipts WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if existing is not None:
                if bytes(existing[0]) != intent_payload:
                    raise SnapshotAnchorReferenceConflict(
                        "operation_id was already committed with different input"
                    )
                self._verify_intent_authority(intent, allow_historical=True)
                return bytes(existing[1])

            verified = self._verify_intent_authority(intent, allow_historical=False)
            transition = str(verified["transition_type"])
            bootstrap_authorization = None
            if transition == BOOTSTRAP_TRANSITION_TYPE:
                if self._config.bootstrap_public_key_path is None:
                    raise SnapshotAnchorReferenceConflict(
                        "reference authority has no bootstrap trust binding"
                    )
                bootstrap_authorization = verify_snapshot_bootstrap_intent_authorization(
                    intent,
                    trusted_public_key_path=self._config.bootstrap_public_key_path,
                    trusted_public_key_dir=None,
                )
            intent_sha256 = hashlib.sha256(intent_payload).hexdigest()

            head = connection.execute(
                "SELECT sequence, receipt_id FROM receipts ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            current_sequence = 0 if head is None else int(head[0])
            current_receipt_id = None if head is None else str(head[1])
            if current_sequence == 0:
                if (
                    self._config.genesis_policy == GENESIS_POLICY_BOOTSTRAP_REQUIRED
                    and transition != BOOTSTRAP_TRANSITION_TYPE
                ):
                    raise SnapshotAnchorReferenceConflict(
                        "authority genesis is BOOTSTRAP_REQUIRED"
                    )
            elif transition == BOOTSTRAP_TRANSITION_TYPE:
                raise SnapshotAnchorReferenceConflict(
                    "bootstrap is permitted only for an empty authority"
                )
            if (
                verified["expected_anchor_sequence"] != current_sequence
                or verified["expected_anchor_receipt_id"] != current_receipt_id
            ):
                raise SnapshotAnchorReferenceConflict(
                    "compare-and-append predecessor is not the current authority HEAD"
                )

            committed_at = self._now()
            operation_created_at = datetime.fromisoformat(str(verified["operation_created_at_utc"]))
            if committed_at < operation_created_at:
                raise SnapshotAnchorReferenceError(
                    "publication intent creation time is after the authority clock"
                )
            if bootstrap_authorization is not None:
                bootstrap_authorization = verify_snapshot_bootstrap_intent_authorization(
                    intent,
                    trusted_public_key_path=self._config.bootstrap_public_key_path,
                    trusted_public_key_dir=None,
                    admitted_at_utc=committed_at,
                )
                consumed = connection.execute(
                    "SELECT receipt_id FROM bootstrap_authorizations WHERE authorization_id = ?",
                    (str(bootstrap_authorization["authorization_id"]),),
                ).fetchone()
                if consumed is not None:
                    raise SnapshotAnchorReferenceConflict(
                        "bootstrap authorization was already consumed"
                    )
            receipt = sign_snapshot_anchor_receipt(
                {
                    "schema_version": PUBLICATION_RECEIPT_SCHEMA,
                    "publication_domain_sha256": self._domain_sha256,
                    "sequence": current_sequence + 1,
                    "previous_receipt_id": current_receipt_id,
                    "operation_id": operation_id,
                    "intent_id": verified["intent_id"],
                    "intent_sha256": intent_sha256,
                    "transition_type": verified["transition_type"],
                    "generation_id": verified["generation_id"],
                    "contract_sha256": verified["contract_sha256"],
                    "generation_inventory_sha256": verified["generation_inventory_sha256"],
                    "legacy_pointer_sha256": verified["legacy_pointer_sha256"],
                    "bootstrap_authorization_id": verified[
                        "bootstrap_authorization_id"
                    ],
                    "bootstrap_authorization_sha256": verified[
                        "bootstrap_authorization_sha256"
                    ],
                    "committed_at_utc": committed_at.isoformat(),
                },
                private_key_path=self._config.anchor_private_key_path,
            )
            self._assert_captured_signer("anchor", publication_signature_key_id(receipt))
            receipt_payload = canonical_json(receipt)
            connection.execute(
                """
                INSERT INTO receipts (
                    sequence, receipt_id, previous_receipt_id, operation_id,
                    intent_sha256, intent_payload, receipt_payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt["sequence"],
                    receipt["receipt_id"],
                    receipt["previous_receipt_id"],
                    operation_id,
                    intent_sha256,
                    intent_payload,
                    receipt_payload,
                ),
            )
            if bootstrap_authorization is not None:
                connection.execute(
                    """
                    INSERT INTO bootstrap_authorizations (
                        authorization_id, receipt_id, authorization_payload
                    ) VALUES (?, ?, ?)
                    """,
                    (
                        bootstrap_authorization["authorization_id"],
                        receipt["receipt_id"],
                        canonical_json(verified["bootstrap_authorization"]),
                    ),
                )
            return receipt_payload

    def _verify_intent_authority(
        self,
        intent: Mapping[str, object],
        *,
        allow_historical: bool,
    ) -> dict[str, object]:
        request_keyring = self._config.request_public_key_dir if allow_historical else None
        bootstrap_keyring = (
            self._config.bootstrap_public_key_dir if allow_historical else None
        )
        try:
            verified = verify_snapshot_publication_intent(
                intent,
                trusted_public_key_path=self._config.request_public_key_path,
                trusted_public_key_dir=request_keyring,
                bootstrap_trusted_public_key_path=self._config.bootstrap_public_key_path,
                bootstrap_trusted_public_key_dir=bootstrap_keyring,
            )
        except ValueError as exc:
            scope = "replay" if allow_historical else "active admission"
            raise SnapshotAnchorReferenceError(
                f"publication intent is not authorized by the {scope} trust set"
            ) from exc
        request_key_id = publication_signature_key_id(intent)
        self._assert_captured_signer("request", request_key_id)
        if request_key_id == self._anchor_key_id:
            raise SnapshotAnchorReferenceError(
                "publisher request and reference anchor signer identities overlap"
            )
        if verified["transition_type"] == BOOTSTRAP_TRANSITION_TYPE:
            bootstrap = verified["bootstrap_authorization"]
            if not isinstance(bootstrap, Mapping):
                raise SnapshotAnchorReferenceError(
                    "verified bootstrap intent has no authorization mapping"
                )
            self._assert_captured_signer(
                "bootstrap",
                publication_signature_key_id(bootstrap),
            )
        return verified

    def _assert_captured_signer(self, role: str, key_id: str) -> None:
        registry = {name: frozenset(ids) for name, ids in self._config.authority_key_ids}
        if key_id not in registry.get(role, frozenset()):
            raise SnapshotAnchorReferenceError(
                f"reference {role} signer is outside the startup trust registry"
            )

    def get_head(self) -> bytes | None:
        """Return the exact latest receipt, or ``None`` for an empty authority."""

        with self._transaction(immediate=False) as connection:
            head = connection.execute(
                "SELECT receipt_payload FROM receipts ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
        return None if head is None else bytes(head[0])

    def lookup_operation(
        self,
        operation_id: str,
        *,
        expected_receipt_payload: bytes | None = None,
        expected_intent_payload: bytes | None = None,
    ) -> bytes:
        """Return immutable receipt bytes only when caller expectations match."""

        if expected_receipt_payload is None and expected_intent_payload is None:
            raise SnapshotAnchorReferenceError(
                "operation lookup requires an expected receipt or intent"
            )
        canonical_operation = _canonical_operation_id(operation_id)
        with self._transaction(immediate=False) as connection:
            row = connection.execute(
                "SELECT intent_payload, receipt_payload FROM receipts WHERE operation_id = ?",
                (canonical_operation,),
            ).fetchone()
        if row is None:
            raise SnapshotAnchorReferenceError("publication operation is unknown")
        intent_payload = bytes(row[0])
        receipt_payload = bytes(row[1])
        if expected_intent_payload is not None and expected_intent_payload != intent_payload:
            raise SnapshotAnchorReferenceConflict(
                "operation lookup intent differs from committed input"
            )
        if expected_receipt_payload is not None and expected_receipt_payload != receipt_payload:
            raise SnapshotAnchorReferenceConflict(
                "operation lookup receipt differs from committed output"
            )
        return receipt_payload

    def observe_head(
        self,
        *,
        intent_payload: bytes,
        receipt_payload: bytes,
        pointer: Mapping[str, object],
        challenge_nonce: str | None = None,
    ) -> bytes:
        """Return a fresh signed observation of the linearized current HEAD."""

        selected_nonce = challenge_nonce or secrets.token_hex(32)
        intent = _strict_canonical_mapping(intent_payload, label="publication intent")
        receipt = _strict_canonical_mapping(receipt_payload, label="anchor receipt")
        receipt_sha256 = hashlib.sha256(receipt_payload).hexdigest()
        expected_pointer = pointer_mapping_from_external_receipt(
            intent,
            receipt,
            receipt_sha256=receipt_sha256,
        )
        if dict(pointer) != expected_pointer:
            raise SnapshotAnchorReferenceConflict(
                "HEAD observation pointer is not the exact receipt projection"
            )

        with self._transaction(immediate=True) as connection:
            head = connection.execute(
                """
                SELECT intent_payload, receipt_payload
                FROM receipts ORDER BY sequence DESC LIMIT 1
                """
            ).fetchone()
            if (
                head is None
                or bytes(head[0]) != intent_payload
                or bytes(head[1]) != receipt_payload
            ):
                raise SnapshotAnchorReferenceConflict(
                    "requested receipt is not the current authority HEAD"
                )
            observed_at = self._now()
            committed_at = datetime.fromisoformat(str(receipt["committed_at_utc"]))
            if observed_at < committed_at:
                raise SnapshotAnchorReferenceError(
                    "authority clock precedes the current receipt commit"
                )
            observation = sign_snapshot_anchor_head_observation(
                {
                    "schema_version": PUBLICATION_HEAD_OBSERVATION_SCHEMA,
                    "publication_domain_sha256": self._domain_sha256,
                    "receipt_id": receipt["receipt_id"],
                    "receipt_sha256": receipt_sha256,
                    "sequence": receipt["sequence"],
                    "challenge_nonce": selected_nonce,
                    "challenge_sha256": external_head_challenge_sha256(
                        pointer,
                        challenge_nonce=selected_nonce,
                    ),
                    "observed_at_utc": observed_at.isoformat(),
                    "expires_at_utc": (observed_at + self._config.head_observation_ttl).isoformat(),
                },
                private_key_path=self._config.anchor_private_key_path,
            )
            self._assert_captured_signer("anchor", publication_signature_key_id(observation))
            observation_payload = canonical_json(observation)
            connection.execute(
                """
                INSERT INTO observations (
                    observation_id, receipt_id, challenge_nonce, observation_payload
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    observation["observation_id"],
                    observation["receipt_id"],
                    selected_nonce,
                    observation_payload,
                ),
            )
            return observation_payload

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS authority_metadata (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_version TEXT NOT NULL,
                    publication_domain_sha256 TEXT NOT NULL,
                    genesis_policy TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS receipts (
                    sequence INTEGER PRIMARY KEY CHECK (sequence >= 1),
                    receipt_id TEXT NOT NULL UNIQUE,
                    previous_receipt_id TEXT,
                    operation_id TEXT NOT NULL UNIQUE,
                    intent_sha256 TEXT NOT NULL,
                    intent_payload BLOB NOT NULL,
                    receipt_payload BLOB NOT NULL,
                    FOREIGN KEY (previous_receipt_id) REFERENCES receipts(receipt_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_successor_per_receipt
                    ON receipts(previous_receipt_id)
                    WHERE previous_receipt_id IS NOT NULL;
                CREATE TABLE IF NOT EXISTS observations (
                    observation_id TEXT PRIMARY KEY,
                    receipt_id TEXT NOT NULL,
                    challenge_nonce TEXT NOT NULL,
                    observation_payload BLOB NOT NULL,
                    FOREIGN KEY (receipt_id) REFERENCES receipts(receipt_id)
                );
                CREATE TABLE IF NOT EXISTS bootstrap_authorizations (
                    authorization_id TEXT PRIMARY KEY,
                    receipt_id TEXT NOT NULL UNIQUE,
                    authorization_payload BLOB NOT NULL,
                    FOREIGN KEY (receipt_id) REFERENCES receipts(receipt_id)
                );
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO authority_metadata (
                    singleton, schema_version, publication_domain_sha256, genesis_policy
                ) VALUES (1, ?, ?, ?)
                """,
                (
                    _SCHEMA_VERSION,
                    self._domain_sha256,
                    self._config.genesis_policy,
                ),
            )
            metadata = connection.execute(
                """
                SELECT schema_version, publication_domain_sha256, genesis_policy
                FROM authority_metadata WHERE singleton = 1
                """
            ).fetchone()
            if metadata != (
                _SCHEMA_VERSION,
                self._domain_sha256,
                self._config.genesis_policy,
            ):
                raise SnapshotAnchorReferenceError(
                    "SQLite authority belongs to another publication domain"
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._config.database_path,
            timeout=self._config.sqlite_timeout_seconds,
            isolation_level=None,
        )
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute(
            f"PRAGMA busy_timeout = {int(self._config.sqlite_timeout_seconds * 1000)}"
        )
        return connection

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

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise SnapshotAnchorReferenceError("reference authority clock must return UTC")
        return value.astimezone(timezone.utc)


def _validate_config(
    config: SnapshotAnchorReferenceConfig,
) -> SnapshotAnchorReferenceConfig:
    database_path = Path(config.database_path).expanduser()
    if not database_path.is_absolute():
        raise SnapshotAnchorReferenceError("reference SQLite path must be absolute")
    if not database_path.parent.is_dir():
        raise SnapshotAnchorReferenceError("reference SQLite parent is unavailable")
    if config.head_observation_ttl <= timedelta(0) or config.head_observation_ttl > _MAX_HEAD_TTL:
        raise SnapshotAnchorReferenceError("reference HEAD observation TTL is invalid")
    if config.sqlite_timeout_seconds <= 0:
        raise SnapshotAnchorReferenceError("reference SQLite timeout is invalid")
    if config.genesis_policy not in {
        GENESIS_POLICY_BOOTSTRAP_REQUIRED,
        GENESIS_POLICY_ALLOW_PUBLISH_FOR_TESTS,
    }:
        raise SnapshotAnchorReferenceError("reference genesis policy is invalid")
    request_keyring = (
        None if config.request_public_key_dir is None else Path(config.request_public_key_dir)
    )
    anchor_keyring = (
        None if config.anchor_public_key_dir is None else Path(config.anchor_public_key_dir)
    )
    bootstrap_public = (
        None
        if config.bootstrap_public_key_path is None
        else Path(config.bootstrap_public_key_path)
    )
    bootstrap_keyring = (
        None if config.bootstrap_public_key_dir is None else Path(config.bootstrap_public_key_dir)
    )
    if (
        config.genesis_policy == GENESIS_POLICY_BOOTSTRAP_REQUIRED
        and bootstrap_public is None
    ):
        raise SnapshotAnchorReferenceError(
            "BOOTSTRAP_REQUIRED authority requires an IT bootstrap public key"
        )
    try:
        request_public = freeze_process_immutable_file(
            config.request_public_key_path,
            label="reference active request public key",
        )
        anchor_private = freeze_process_immutable_file(
            config.anchor_private_key_path,
            label="reference active anchor private key",
        )
        if request_keyring is not None:
            freeze_process_immutable_directory(
                request_keyring,
                label="reference historical request keyring",
            )
        if anchor_keyring is not None:
            freeze_process_immutable_directory(
                anchor_keyring,
                label="reference historical anchor keyring",
            )
        if bootstrap_public is not None:
            bootstrap_public = freeze_process_immutable_file(
                bootstrap_public,
                label="reference active bootstrap public key",
            )
        if bootstrap_keyring is not None:
            freeze_process_immutable_directory(
                bootstrap_keyring,
                label="reference historical bootstrap keyring",
            )
    except ValueError as exc:
        raise SnapshotAnchorReferenceError(
            "reference authority trust material cannot be frozen"
        ) from exc
    request_ids = publication_trusted_key_ids(
        request_public,
        keyring_directory=request_keyring,
    )
    anchor_id = publication_private_key_id(anchor_private)
    anchor_ids = frozenset({anchor_id}) | publication_keyring_key_ids(anchor_keyring)
    bootstrap_ids: frozenset[str] = frozenset()
    if bootstrap_public is not None:
        bootstrap_ids = publication_trusted_key_ids(
            bootstrap_public,
            keyring_directory=bootstrap_keyring,
        )
    authority_ids = {
        "anchor": anchor_ids,
        "bootstrap": bootstrap_ids,
        "request": request_ids,
    }
    try:
        assert_disjoint_publication_authority_ids(authority_ids)
    except ValueError as exc:
        raise SnapshotAnchorReferenceError(
            "reference publication authority trust roles must be globally disjoint"
        ) from exc
    return SnapshotAnchorReferenceConfig(
        database_path=database_path,
        request_public_key_path=request_public,
        anchor_private_key_path=anchor_private,
        request_public_key_dir=request_keyring,
        anchor_public_key_dir=anchor_keyring,
        bootstrap_public_key_path=bootstrap_public,
        bootstrap_public_key_dir=bootstrap_keyring,
        genesis_policy=config.genesis_policy,
        head_observation_ttl=config.head_observation_ttl,
        sqlite_timeout_seconds=float(config.sqlite_timeout_seconds),
        authority_key_ids=tuple(
            (role, tuple(sorted(ids))) for role, ids in sorted(authority_ids.items())
        ),
    )


def _strict_canonical_mapping(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SnapshotAnchorReferenceError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict) or canonical_json(value) != payload:
        raise SnapshotAnchorReferenceError(f"{label} is not canonical JSON")
    return value


def _canonical_operation_id(value: object) -> str:
    raw = str(value or "").strip().lower()
    try:
        canonical = str(uuid.UUID(raw))
    except ValueError as exc:
        raise SnapshotAnchorReferenceError("operation_id is not a canonical UUID") from exc
    if raw != canonical:
        raise SnapshotAnchorReferenceError("operation_id is not a canonical UUID")
    return canonical
