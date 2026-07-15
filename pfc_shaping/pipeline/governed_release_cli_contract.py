"""Lightweight, shared CLI contract for phase-separated LT release commands."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from pfc_shaping.build_identity import SOURCE_REVISION
from pfc_shaping.package_contract import (
    ALLOWED_RUNTIME_PYTHON_FILES,
    BUILD_IDENTITY_PLACEHOLDER,
    BUILD_IDENTITY_TEMPLATE,
)
from pfc_shaping.path_safety import path_is_link
from pfc_shaping.pipeline.promotion_contract import (
    public_key_id_from_private_key_path,
    public_key_id_from_public_key_path,
)
from pfc_shaping.pipeline.release_request_contract import (
    REGISTER_SIGNING_PRIVATE_KEY_ENV,
    REGISTER_TRUSTED_PUBLIC_KEY_DIR_ENV,
    REGISTER_TRUSTED_PUBLIC_KEY_ENV,
)

EXIT_SUCCESS = 0
EXIT_USAGE = 2
EXIT_GOVERNANCE_PENDING = 30
EXIT_CAS_CONFLICT = 40
EXIT_TRANSITION_BUSY = 41
EXIT_INTEGRITY_FAILURE = 50
EXIT_PROJECTION_REPAIR_REQUIRED = 51

MODEL_GOVERNANCE_PRIVATE_KEY_ENV = "PFC_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH"
ACQUISITION_PRIVATE_KEY_ENV = "PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH"
QUOTE_CONFLICT_PRIVATE_KEY_ENV = (
    "PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH"
)
RECEIPT_PRIVATE_KEY_ENV = "PFC_PROMOTION_SIGNING_PRIVATE_KEY_PATH"
EVENT_PRIVATE_KEY_ENV = "PFC_PROMOTION_EVENT_SIGNING_PRIVATE_KEY_PATH"
ROLLBACK_AUTHORIZATION_PRIVATE_KEY_ENV = (
    "PFC_ROLLBACK_AUTHORIZATION_SIGNING_PRIVATE_KEY_PATH"
)
EVENT_PUBLIC_KEY_ENV = "PFC_PROMOTION_EVENT_TRUSTED_PUBLIC_KEY_PATH"
RECEIPT_PUBLIC_KEY_ENV = "PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH"
EVENT_PUBLIC_KEY_DIR_ENV = "PFC_PROMOTION_EVENT_TRUSTED_PUBLIC_KEY_DIR"
RECEIPT_PUBLIC_KEY_DIR_ENV = "PFC_PROMOTION_TRUSTED_PUBLIC_KEY_DIR"
ROLLBACK_AUTHORIZATION_PUBLIC_KEY_ENV = (
    "PFC_ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_PATH"
)
ROLLBACK_AUTHORIZATION_PUBLIC_KEY_DIR_ENV = (
    "PFC_ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_DIR"
)
PRIVATE_KEY_ENVIRONMENTS = frozenset(
    {
        MODEL_GOVERNANCE_PRIVATE_KEY_ENV,
        ACQUISITION_PRIVATE_KEY_ENV,
        QUOTE_CONFLICT_PRIVATE_KEY_ENV,
        RECEIPT_PRIVATE_KEY_ENV,
        EVENT_PRIVATE_KEY_ENV,
        ROLLBACK_AUTHORIZATION_PRIVATE_KEY_ENV,
        REGISTER_SIGNING_PRIVATE_KEY_ENV,
    }
)
ALLOWED_PRIVATE_KEYS_BY_COMMAND = {
    "build": frozenset(),
    "finalize": frozenset(),
    "register": frozenset({REGISTER_SIGNING_PRIVATE_KEY_ENV}),
    "audit": frozenset({RECEIPT_PRIVATE_KEY_ENV}),
    "promote": frozenset({EVENT_PRIVATE_KEY_ENV}),
    "rollback": frozenset({EVENT_PRIVATE_KEY_ENV}),
    "status": frozenset(),
}


class ReleaseCliIdentityError(RuntimeError):
    """Raised when one process exposes private keys from another authority."""


def canonical_sha256(value: str) -> str:
    selected = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", selected):
        raise argparse.ArgumentTypeError("value must be a canonical SHA-256")
    return selected


def canonical_generation_id(value: str) -> str:
    selected = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", selected):
        raise argparse.ArgumentTypeError("generation id must be a portable identifier")
    if selected.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        raise argparse.ArgumentTypeError("generation id is reserved on Windows")
    return selected


def canonical_source_revision(value: str) -> str:
    selected = str(value).strip().lower()
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", selected):
        raise argparse.ArgumentTypeError("source revision must be a 40- or 64-hex digest")
    return selected


def absolute_path(value: str) -> Path:
    selected = Path(value).expanduser()
    if not selected.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return selected


def governed_absolute_path(value: str | Path, *, label: str) -> Path:
    """Return one absolute lexical path after rejecting linked components."""

    try:
        selected = Path(value).expanduser()
    except TypeError as exc:
        raise ReleaseCliIdentityError(f"{label} path is missing") from exc
    if not selected.is_absolute():
        raise ReleaseCliIdentityError(f"{label} path must be absolute")
    lexical = Path(os.path.abspath(selected))
    for component in (lexical, *lexical.parents):
        try:
            is_link = path_is_link(component)
        except OSError as exc:
            raise ReleaseCliIdentityError(f"cannot validate {label} path") from exc
        if is_link:
            raise ReleaseCliIdentityError(f"{label} path cannot contain a link")
    return lexical


def assert_namespace_absolute_paths(
    args: argparse.Namespace,
    names: tuple[str, ...],
) -> None:
    for name in names:
        try:
            value = getattr(args, name)
        except AttributeError as exc:
            raise ReleaseCliIdentityError(f"{name} path is missing") from exc
        setattr(args, name, governed_absolute_path(value, label=name))


def canonical_utc_timestamp(value: str) -> str:
    raw = str(value).strip().replace("Z", "+00:00")
    try:
        selected = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timestamp must be ISO-8601") from exc
    if selected.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must be timezone-aware")
    return selected.astimezone(timezone.utc).isoformat()


def assert_phase_private_key_scope(command: str) -> None:
    """Reject authority-key co-location before any phase performs I/O."""

    allowed = ALLOWED_PRIVATE_KEYS_BY_COMMAND.get(command)
    if allowed is None:
        raise ReleaseCliIdentityError(f"unknown governed release command: {command}")
    exposed = {
        name
        for name in PRIVATE_KEY_ENVIRONMENTS
        if str(os.environ.get(name, "")).strip()
    }
    forbidden = sorted(exposed.difference(allowed))
    if forbidden:
        raise ReleaseCliIdentityError(
            f"{command} identity exposes forbidden private keys: {forbidden}"
        )
    if command in {"promote", "rollback"}:
        _assert_transition_authority_separation(command)


def assert_governed_runtime_source(source_revision: str) -> None:
    assert_installed_runtime_sealed()
    if str(source_revision) != SOURCE_REVISION:
        raise ReleaseCliIdentityError(
            "source revision does not match the installed LT runtime"
        )


def assert_installed_runtime_sealed() -> None:
    if not isinstance(SOURCE_REVISION, str) or not re.fullmatch(
        r"[0-9a-f]{64}", SOURCE_REVISION
    ):
        raise ReleaseCliIdentityError(
            "governed mutation requires an installed runtime with a sealed identity"
        )
    if _installed_runtime_source_revision() != SOURCE_REVISION:
        raise ReleaseCliIdentityError(
            "installed LT runtime sources do not match the sealed identity"
        )


def _installed_runtime_source_revision() -> str:
    package_root = Path(__file__).resolve().parents[1]
    expected = {
        path.removeprefix("pfc_shaping/") for path in ALLOWED_RUNTIME_PYTHON_FILES
    }
    actual = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*.py")
        if path.is_file()
    }
    if actual != expected:
        raise ReleaseCliIdentityError(
            "installed LT runtime source inventory does not match the package contract"
        )
    digest = hashlib.sha256()
    identity_placeholder = BUILD_IDENTITY_TEMPLATE.format(
        source_revision=BUILD_IDENTITY_PLACEHOLDER
    ).encode("ascii")
    for relative in sorted(expected):
        path = package_root / relative
        if path_is_link(path):
            raise ReleaseCliIdentityError(
                f"installed LT runtime source cannot be linked: {relative}"
            )
        try:
            stat = path.stat()
            payload = path.read_bytes()
        except OSError as exc:
            raise ReleaseCliIdentityError(
                f"cannot verify installed LT runtime source: {relative}"
            ) from exc
        if stat.st_nlink != 1:
            raise ReleaseCliIdentityError(
                f"installed LT runtime source cannot be hardlinked: {relative}"
            )
        if relative == "build_identity.py":
            payload = identity_placeholder
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def _assert_transition_authority_separation(command: str) -> None:
    event_private = str(os.environ.get(EVENT_PRIVATE_KEY_ENV, "")).strip()
    if not event_private:
        return
    event_public = str(os.environ.get(EVENT_PUBLIC_KEY_ENV, "")).strip()
    receipt_public = str(os.environ.get(RECEIPT_PUBLIC_KEY_ENV, "")).strip()
    rollback_public = str(
        os.environ.get(ROLLBACK_AUTHORIZATION_PUBLIC_KEY_ENV, "")
    ).strip()
    if not event_public and not receipt_public and not rollback_public:
        return
    try:
        event_private_path = _absolute_lexical_path(event_private)
        _assert_no_key_path_links(
            event_private_path,
            label="transition signing private key",
        )
        event_id = public_key_id_from_private_key_path(event_private_path)
        if event_public and public_key_id_from_public_key_path(event_public) != event_id:
            raise ReleaseCliIdentityError(
                "transition signing private key does not match its trusted public key"
            )
        authority_ids: dict[str, set[str]] = {"event": {event_id}}
        if event_public:
            authority_ids["event"] = _public_key_identities(
                event_public,
                directory_env=EVENT_PUBLIC_KEY_DIR_ENV,
            )
        if receipt_public:
            authority_ids["receipt"] = _public_key_identities(
                receipt_public,
                directory_env=RECEIPT_PUBLIC_KEY_DIR_ENV,
            )
        if rollback_public:
            authority_ids["rollback_authorization"] = _public_key_identities(
                rollback_public,
                directory_env=ROLLBACK_AUTHORIZATION_PUBLIC_KEY_DIR_ENV,
            )
        register_public = str(os.environ.get(REGISTER_TRUSTED_PUBLIC_KEY_ENV, "")).strip()
        if register_public:
            authority_ids["register"] = _public_key_identities(
                register_public,
                directory_env=REGISTER_TRUSTED_PUBLIC_KEY_DIR_ENV,
            )
        labels = sorted(authority_ids)
        collisions = [
            f"{left}/{right}"
            for index, left in enumerate(labels)
            for right in labels[index + 1 :]
            if authority_ids[left].intersection(authority_ids[right])
        ]
    except ReleaseCliIdentityError:
        raise
    except (OSError, ValueError) as exc:
        raise ReleaseCliIdentityError(
            f"cannot validate {command} authority key identities"
        ) from exc
    if collisions:
        raise ReleaseCliIdentityError(
            f"transition authority keyrings overlap: {collisions}"
        )


def _public_key_identities(primary: str, *, directory_env: str) -> set[str]:
    paths = [_absolute_lexical_path(primary)]
    configured_directory = str(os.environ.get(directory_env, "")).strip()
    if configured_directory:
        directory = _absolute_lexical_path(configured_directory)
        _assert_no_key_path_links(directory, label=directory_env)
        if not directory.is_absolute() or not directory.is_dir():
            raise ReleaseCliIdentityError(f"{directory_env} is unavailable")
        paths.extend(sorted(directory.glob("*.pem")))
    for path in paths:
        _assert_no_key_path_links(path, label="authority public key")
        if not path.is_file():
            raise ReleaseCliIdentityError(f"authority public key is unavailable: {path}")
    return {public_key_id_from_public_key_path(path) for path in paths}


def _absolute_lexical_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ReleaseCliIdentityError("authority key paths must be absolute")
    return Path(os.path.abspath(path))


def _assert_no_key_path_links(path: Path, *, label: str) -> None:
    for component in (path, *path.parents):
        try:
            is_link = path_is_link(component)
        except OSError as exc:
            raise ReleaseCliIdentityError(f"cannot validate {label}") from exc
        if is_link:
            raise ReleaseCliIdentityError(f"{label} cannot contain a link")


def add_build_arguments(parser: argparse.ArgumentParser) -> None:
    """Add canonical builder arguments without importing the model runtime."""

    parser.add_argument("--run-id", required=True)
    parser.add_argument("--release-root", type=absolute_path, required=True)
    parser.add_argument(
        "--failure-root",
        type=absolute_path,
        required=True,
        help="External durable root for immutable operational failure manifests",
    )
    parser.add_argument(
        "--source-revision",
        type=canonical_source_revision,
        required=True,
        help="Exact source revision used to build the installed runtime",
    )
    parser.add_argument(
        "--as-of",
        "--reference-timestamp",
        dest="reference_timestamp",
        type=canonical_utc_timestamp,
        required=True,
        help="Explicit timezone-aware valuation timestamp",
    )
    parser.add_argument(
        "--build-timestamp",
        type=canonical_utc_timestamp,
        required=True,
        help="Deterministic logical build timestamp, at or after --as-of",
    )
    parser.add_argument(
        "--config",
        type=absolute_path,
        required=True,
        help="Exact external LT configuration snapshot",
    )
    parser.add_argument(
        "--config-sha256",
        type=canonical_sha256,
        required=True,
        help="Expected SHA-256 of the external LT configuration snapshot",
    )
    parser.add_argument(
        "--input-snapshot-contract",
        type=absolute_path,
        required=True,
        help="Exact immutable lt_input_snapshot.json selected for this build",
    )
    parser.add_argument(
        "--input-snapshot-sha256",
        type=canonical_sha256,
        required=True,
        help="Expected SHA-256 of the selected LT input snapshot contract",
    )
    parser.add_argument(
        "--input-pointer-contract",
        type=absolute_path,
        required=True,
        help="Exact current.json bytes selecting the snapshot for this build or replay",
    )
    parser.add_argument(
        "--input-pointer-sha256",
        type=canonical_sha256,
        required=True,
        help="Expected SHA-256 of the canonical pfc_lt pointer binding the snapshot",
    )
    parser.add_argument(
        "--input-generation-id",
        type=canonical_generation_id,
        required=True,
        help="Exact generation_id bound by pointer and snapshot contract",
    )
    parser.add_argument(
        "--data-root",
        type=absolute_path,
        required=True,
        help="Exact shared FMV data root containing the selected snapshot",
    )
    parser.add_argument(
        "--historical-thresholds",
        type=absolute_path,
        required=True,
    )
    parser.add_argument(
        "--historical-thresholds-receipt",
        type=absolute_path,
        required=True,
    )
    parser.add_argument(
        "--selected-lambda-decision",
        type=absolute_path,
        required=True,
    )
    parser.add_argument(
        "--selected-lambda-decision-receipt",
        type=absolute_path,
        required=True,
    )
    parser.add_argument(
        "--eex-report-path",
        type=absolute_path,
        required=True,
        help="Exact EEX workbook mounted for this run",
    )
    parser.add_argument(
        "--eex-acquisition-contract",
        type=absolute_path,
        required=True,
        help="IT-signed acquisition contract binding the exact EEX workbook",
    )
    parser.add_argument(
        "--peak-source-policy",
        choices=("same_first", "strict_same", "any"),
        required=True,
    )
    parser.add_argument(
        "--use-seasonal-hourly-shape",
        action=argparse.BooleanOptionalAction,
        required=True,
        help="Explicitly enable or disable the governed seasonal hourly shape",
    )


def resolve_build_data_root(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> argparse.Namespace:
    """Retain a shared parser hook while requiring an explicit build root."""

    if args.data_root is None:
        parser.error("--data-root is required")
    try:
        assert_governed_runtime_source(args.source_revision)
    except ReleaseCliIdentityError as exc:
        parser.error(str(exc))
    if datetime.fromisoformat(args.build_timestamp) < datetime.fromisoformat(
        args.reference_timestamp
    ):
        parser.error("--build-timestamp must be at or after --as-of")
    return args


def add_finalize_arguments(parser: argparse.ArgumentParser) -> None:
    """Add canonical finalization arguments without importing finalizer code."""

    parser.add_argument("--release-root", type=absolute_path, required=True)
    parser.add_argument(
        "--failure-root",
        type=absolute_path,
        required=True,
        help="External durable root for immutable operational failure manifests",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--source-hierarchy-policy",
        type=absolute_path,
        required=True,
    )
