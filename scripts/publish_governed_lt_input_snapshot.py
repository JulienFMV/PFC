#!/usr/bin/env python3
"""Publish a governed LT snapshot through the external CAS authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.data.shared_data_root import (  # noqa: E402
    configured_lt_data_root,
    resolve_fmv_data_root,
)
from pfc_shaping.data.snapshot_anchor_client import (  # noqa: E402
    SnapshotAnchorAuthenticationError,
    SnapshotAnchorClient,
    SnapshotAnchorClientError,
    SnapshotAnchorConflictError,
    SnapshotAnchorIndeterminateError,
    SnapshotAnchorProtocolError,
    SnapshotAnchorUnavailableError,
    load_anchor_client_config_from_environment,
)
from pfc_shaping.data.snapshot_publication_state import (  # noqa: E402
    canonical_json,
    pointer_mapping_from_external_receipt,
)
from pfc_shaping.data.snapshot_publisher import (  # noqa: E402
    SnapshotPublicationBusy,
    SnapshotPublicationConflict,
    SnapshotPublicationRepairRequired,
    assert_snapshot_publisher_runtime_identity,
    committed_snapshot_repair_result,
    finalize_governed_lt_snapshot,
    prepare_governed_lt_snapshot,
)
from pfc_shaping.durable_artifact import (  # noqa: E402
    DurableArtifactRepairRequired,
    durable_artifact_durability_classification,
    validated_durable_target,
    write_durable_exact_bytes,
)
from pfc_shaping.exit_codes import (  # noqa: E402
    EXIT_CAS_CONFLICT,
    EXIT_INDETERMINATE,
    EXIT_INTEGRITY_FAILURE,
    EXIT_PROJECTION_REPAIR_REQUIRED,
    EXIT_TRANSITION_BUSY,
)
from pfc_shaping.path_safety import (  # noqa: E402
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    try:
        for name in ("FMV_DATA_ROOT", "PFC_LT_DATA_ROOT", "PFC_SHARED_DATA_ROOT"):
            value = str(os.environ.get(name, "")).strip()
            if value:
                assert_absolute_path_has_no_links(Path(value).expanduser())
        default_root = configured_lt_data_root()
    except ValueError as exc:
        parser.error(str(exc))

    prepare = subparsers.add_parser("prepare", help="Install generation and sign intent")
    _add_data_root(prepare, default_root)
    prepare.add_argument("--source-bundle", type=Path)
    prepare.add_argument(
        "--bootstrap-authorization",
        type=Path,
        help="Offline IT authorization for the one-shot migrated legacy bootstrap",
    )
    prepare.add_argument("--operation-id", required=True)
    prepare.add_argument("--operation-created-at-utc", required=True)
    prepare.add_argument("--expected-anchor-sequence", type=int)
    prepare.add_argument("--expected-anchor-receipt-id")

    cas = subparsers.add_parser("cas", help="Commit intent through external mTLS CAS")
    cas.add_argument("--intent", type=Path, required=True)
    cas.add_argument("--receipt-output", type=Path, required=True)

    finalize = subparsers.add_parser(
        "finalize",
        help="Acquire a fresh HEAD proof and project current.json",
    )
    _add_data_root(finalize, default_root)
    finalize.add_argument("--intent", type=Path, required=True)
    finalize.add_argument("--anchor-receipt", type=Path, required=True)
    finalize.add_argument("--observation-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in {"prepare", "finalize"}:
        try:
            required_value = str(os.environ.get("FMV_DATA_ROOT", "")).strip()
            if not required_value:
                raise ValueError("FMV_DATA_ROOT is required")
            required_lexical = assert_absolute_path_has_no_links(
                Path(required_value).expanduser()
            )
            selected_lexical = assert_absolute_path_has_no_links(
                Path(args.data_root).expanduser()
            )
            required_root = resolve_fmv_data_root(
                required_lexical,
                compatibility_envs=(),
                environ={},
                require_exists=False,
            )
            selected_root = resolve_fmv_data_root(
                selected_lexical,
                compatibility_envs=(),
                environ={},
                require_exists=False,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if selected_root.resolve() != required_root.resolve():
            parser.error("--data-root must match the authoritative FMV_DATA_ROOT")
        # Preserve the checked authoritative lexical path. Resolving it here
        # would erase junction/symlink evidence before the publication boundary
        # performs its own fail-closed ancestry check.
        args.data_root = required_lexical
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        assert_snapshot_publisher_runtime_identity(phase=args.command)
        if args.command == "prepare":
            expected_receipt_id, expected_sequence = _prepare_head_expectation(args)
            result = prepare_governed_lt_snapshot(
                source_bundle=args.source_bundle,
                data_root=args.data_root,
                operation_id=args.operation_id,
                operation_created_at_utc=args.operation_created_at_utc,
                expected_anchor_receipt_id=expected_receipt_id,
                expected_anchor_sequence=expected_sequence,
                adopt_current_legacy=False,
                expected_legacy_pointer_sha256=None,
                allow_empty_genesis=False,
                bootstrap_authorization_path=args.bootstrap_authorization,
            )
        elif args.command == "cas":
            result = _commit_external_cas(args)
        else:
            result = _finalize_external_projection(args)
    except SnapshotPublicationRepairRequired as exc:
        print(json.dumps(exc.result, sort_keys=True), file=sys.stderr)
        return EXIT_PROJECTION_REPAIR_REQUIRED
    except DurableArtifactRepairRequired as exc:
        return _emit_local_repair_failure(args.command, exc)
    except (SnapshotPublicationConflict, SnapshotAnchorConflictError) as exc:
        return _emit_failure(args.command, "CAS_CONFLICT", exc, EXIT_CAS_CONFLICT)
    except SnapshotPublicationBusy as exc:
        return _emit_failure(args.command, "TRANSITION_BUSY", exc, EXIT_TRANSITION_BUSY)
    except (SnapshotAnchorIndeterminateError, SnapshotAnchorUnavailableError) as exc:
        return _emit_failure(args.command, "INDETERMINATE", exc, EXIT_INDETERMINATE)
    except (
        SnapshotAnchorAuthenticationError,
        SnapshotAnchorProtocolError,
        SnapshotAnchorClientError,
        OSError,
        ValueError,
    ) as exc:
        return _emit_failure(args.command, "INTEGRITY_FAILURE", exc, EXIT_INTEGRITY_FAILURE)
    print(json.dumps(result, sort_keys=True))
    return 0


def _emit_failure(command: str, status: str, exc: Exception, exit_code: int) -> int:
    payload = {
        "schema_version": "lt_snapshot_publication_cli_result.v1",
        "status": status,
        "command": command,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "production_authorization": False,
    }
    print(json.dumps(payload, sort_keys=True), file=sys.stderr)
    return exit_code


def _emit_local_repair_failure(
    command: str,
    exc: DurableArtifactRepairRequired,
) -> int:
    payload = {
        "schema_version": "lt_snapshot_publication_cli_result.v1",
        "status": "LOCAL_ARTIFACT_REPAIR_REQUIRED",
        "command": command,
        "commit_status": "NOT_COMMITTED" if command == "prepare" else "UNPROVEN",
        "projection_status": "LOCAL_ARTIFACT_REPAIR_REQUIRED",
        "retry_disposition": "FOLLOW_OPERATOR_ACTION_THEN_RETRY_SAME_COMMAND",
        "local_repair": exc.result,
        "production_authorization": False,
    }
    print(json.dumps(payload, sort_keys=True), file=sys.stderr)
    return EXIT_INTEGRITY_FAILURE


def _prepare_head_expectation(args: argparse.Namespace) -> tuple[str | None, int]:
    if getattr(args, "bootstrap_authorization", None) is not None:
        if args.expected_anchor_sequence not in {None, 0}:
            raise ValueError("bootstrap authorization requires expected anchor sequence 0")
        if args.expected_anchor_receipt_id is not None:
            raise ValueError("bootstrap authorization requires an empty external HEAD")
        return None, 0
    if args.expected_anchor_sequence is not None:
        return args.expected_anchor_receipt_id, int(args.expected_anchor_sequence)
    if args.expected_anchor_receipt_id is not None:
        raise ValueError(
            "--expected-anchor-receipt-id requires --expected-anchor-sequence"
        )
    client = SnapshotAnchorClient(load_anchor_client_config_from_environment(os.environ))
    head_payload = client.get_head()
    head = _strict_mapping(head_payload, label="external anchor HEAD receipt")
    return str(head["receipt_id"]), int(head["sequence"])


def _commit_external_cas(args: argparse.Namespace) -> dict[str, object]:
    intent_path = assert_absolute_path_has_no_links(args.intent.expanduser())
    intent_payload = read_stable_single_link_file(
        intent_path,
        label="publication intent",
    )
    intent = _strict_mapping(intent_payload, label="publication intent")
    receipt_output = _validated_output_target(args.receipt_output)
    client = SnapshotAnchorClient(load_anchor_client_config_from_environment(os.environ))
    if receipt_output.is_file():
        try:
            local_receipt_payload = read_stable_single_link_file(
                receipt_output,
                label="persisted anchor receipt",
            )
            _strict_mapping(local_receipt_payload, label="persisted anchor receipt")
        except (OSError, ValueError):
            receipt_payload = client.lookup_operation(
                str(intent["operation_id"]),
                expected_intent_payload=intent_payload,
            )
        else:
            receipt_payload = client.lookup_operation(
                str(intent["operation_id"]),
                expected_receipt_payload=local_receipt_payload,
                expected_intent_payload=intent_payload,
            )
    else:
        try:
            receipt_payload = client.compare_and_append(intent_payload)
        except SnapshotAnchorClientError as commit_error:
            try:
                receipt_payload = client.lookup_operation(
                    str(intent["operation_id"]),
                    expected_intent_payload=intent_payload,
                )
            except SnapshotAnchorClientError as recovery_error:
                if isinstance(commit_error, SnapshotAnchorConflictError):
                    raise commit_error
                if isinstance(
                    commit_error,
                    (SnapshotAnchorIndeterminateError, SnapshotAnchorUnavailableError),
                ):
                    raise SnapshotAnchorIndeterminateError(
                        "external CAS outcome is indeterminate and exact recovery is unavailable"
                    ) from recovery_error
                if isinstance(
                    commit_error,
                    (
                        SnapshotAnchorAuthenticationError,
                        SnapshotAnchorProtocolError,
                    ),
                ):
                    raise commit_error
                raise SnapshotAnchorProtocolError(
                    "external CAS failed and exact operation recovery was invalid"
                ) from recovery_error
    receipt = _strict_mapping(receipt_payload, label="anchor receipt")
    try:
        _write_exact_output(receipt_output, receipt_payload, repair_from_authority=True)
    except Exception as exc:
        raise SnapshotPublicationRepairRequired(
            committed_snapshot_repair_result(
                command="cas",
                intent=intent,
                receipt=receipt,
                receipt_payload=receipt_payload,
                receipt_path=receipt_output,
                projection_status="RECEIPT_ARCHIVE_REPAIR_REQUIRED",
                retry_disposition="RETRY_EXACT_OPERATION_LOOKUP",
                error=exc,
            )
        ) from exc
    return {
        "schema_version": "lt_snapshot_publication_cli_result.v1",
        "status": "COMMITTED_EXTERNAL_CAS_AWAITING_FRESH_HEAD",
        "command": "cas",
        "commit_status": "COMMITTED",
        "projection_status": "AWAITING_FRESH_HEAD",
        "retry_disposition": "RUN_FINALIZE_WITH_FRESH_HEAD",
        "operation_id": intent["operation_id"],
        "intent_id": intent["intent_id"],
        "anchor_receipt_id": receipt["receipt_id"],
        "anchor_sequence": receipt["sequence"],
        "receipt_path": str(receipt_output),
        "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
        "local_durability": durable_artifact_durability_classification(),
        "production_authorization": False,
    }


def _finalize_external_projection(args: argparse.Namespace) -> dict[str, object]:
    intent_path = assert_absolute_path_has_no_links(args.intent.expanduser())
    intent_payload = read_stable_single_link_file(intent_path, label="publication intent")
    intent = _strict_mapping(intent_payload, label="publication intent")
    receipt_path = _validated_output_target(args.anchor_receipt)
    client = SnapshotAnchorClient(load_anchor_client_config_from_environment(os.environ))
    receipt_payload = _recover_exact_receipt(
        client=client,
        intent=intent,
        intent_payload=intent_payload,
        receipt_path=receipt_path,
    )
    receipt = _strict_mapping(receipt_payload, label="anchor receipt")
    try:
        pointer = pointer_mapping_from_external_receipt(
            intent,
            receipt,
            receipt_sha256=hashlib.sha256(receipt_payload).hexdigest(),
        )
        challenge_nonce = os.urandom(32).hex()
        observation_payload = client.observe_head(
            intent_payload=intent_payload,
            receipt_payload=receipt_payload,
            pointer=pointer,
            challenge_nonce=challenge_nonce,
        )
        observation = _strict_mapping(
            observation_payload,
            label="fresh anchor HEAD observation",
        )
        observation_directory = _validated_output_directory(args.observation_directory)
        observation_path = _write_exact_output(
            observation_directory / f"{observation['observation_id']}.json",
            observation_payload,
            repair_from_authority=True,
        )
        result = finalize_governed_lt_snapshot(
            data_root=args.data_root,
            intent_path=intent_path,
            anchor_receipt_path=receipt_path,
            head_observation_path=observation_path,
            expected_head_challenge_nonce=challenge_nonce,
        )
    except SnapshotPublicationRepairRequired:
        raise
    except Exception as exc:
        raise SnapshotPublicationRepairRequired(
            committed_snapshot_repair_result(
                command="finalize",
                intent=intent,
                receipt=receipt,
                receipt_payload=receipt_payload,
                receipt_path=receipt_path,
                projection_status="LOCAL_PROJECTION_REPAIR_REQUIRED",
                retry_disposition="RETRY_FINALIZE_WITH_FRESH_HEAD",
                error=exc,
            )
        ) from exc
    return {
        **result,
        "head_observation_path": str(observation_path),
        "head_observation_sha256": hashlib.sha256(observation_payload).hexdigest(),
        "head_challenge_nonce": challenge_nonce,
    }


def _recover_exact_receipt(
    *,
    client: SnapshotAnchorClient,
    intent: dict[str, object],
    intent_payload: bytes,
    receipt_path: Path,
) -> bytes:
    local_payload: bytes | None = None
    if receipt_path.is_file():
        try:
            local_payload = read_stable_single_link_file(
                receipt_path,
                label="persisted anchor receipt",
            )
            _strict_mapping(local_payload, label="persisted anchor receipt")
        except (OSError, ValueError):
            local_payload = None
    if local_payload is not None:
        return client.lookup_operation(
            str(intent["operation_id"]),
            expected_receipt_payload=local_payload,
            expected_intent_payload=intent_payload,
        )
    authoritative_payload = client.lookup_operation(
        str(intent["operation_id"]),
        expected_intent_payload=intent_payload,
    )
    authoritative_receipt = _strict_mapping(
        authoritative_payload,
        label="authoritative anchor receipt",
    )
    try:
        _write_exact_output(
            receipt_path,
            authoritative_payload,
            repair_from_authority=True,
        )
    except Exception as exc:
        raise SnapshotPublicationRepairRequired(
            committed_snapshot_repair_result(
                command="finalize",
                intent=intent,
                receipt=authoritative_receipt,
                receipt_payload=authoritative_payload,
                receipt_path=receipt_path,
                projection_status="RECEIPT_ARCHIVE_REPAIR_REQUIRED",
                retry_disposition="RETRY_EXACT_OPERATION_LOOKUP",
                error=exc,
            )
        ) from exc
    return authoritative_payload


def _write_exact_output(
    path: Path,
    payload: bytes,
    *,
    repair_from_authority: bool = False,
) -> Path:
    return write_durable_exact_bytes(
        path,
        payload,
        label="publication evidence output",
        replace_existing_if=(
            _publication_evidence_is_noncanonical if repair_from_authority else None
        ),
    )


def _publication_evidence_is_noncanonical(payload: bytes) -> bool:
    try:
        _strict_mapping(payload, label="existing publication evidence")
    except (UnicodeDecodeError, ValueError):
        return True
    return False


def _validated_output_target(path: Path) -> Path:
    return validated_durable_target(path, label="publication evidence output")


def _validated_output_directory(path: Path) -> Path:
    directory = path.expanduser()
    if not directory.is_absolute():
        raise ValueError("publication evidence directory must be absolute")
    parent = assert_absolute_path_has_no_links(directory.parent)
    if not parent.is_dir():
        raise ValueError("publication evidence parent directory is unavailable")
    directory.mkdir(exist_ok=True)
    directory = assert_absolute_path_has_no_links(directory)
    if not directory.is_dir():
        raise ValueError("publication evidence directory is unavailable")
    return directory


def _strict_mapping(payload: bytes, *, label: str) -> dict[str, object]:
    value = load_strict_json(payload.decode("utf-8"))
    if not isinstance(value, dict) or canonical_json(value) != payload:
        raise ValueError(f"{label} must be canonical JSON")
    return value


def _add_data_root(
    parser: argparse.ArgumentParser,
    default_root: Path | None,
) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_root,
        required=default_root is None,
        help="Shared root (or FMV_DATA_ROOT)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
