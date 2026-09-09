"""Authenticated REGISTER authority contract for governed LT releases."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from pfc_shaping.path_safety import (
    path_is_link,
    process_immutable_directory_entries,
    process_immutable_file_bytes,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import (
    StrictStructuredDataError,
    load_strict_json,
)

RELEASE_REQUEST_SCHEMA = "governed_lt_release_request.v3"
WORKFLOW_DOMAIN_SCHEMA = "governed_lt_workflow_domain.v1"
WORKFLOW_DOMAIN_MARKER = "workflow_domain.json"
WORKFLOW_DOMAIN_ID_ENV = "PFC_RELEASE_WORKFLOW_DOMAIN_ID"
REGISTER_SIGNING_PRIVATE_KEY_ENV = "PFC_RELEASE_REQUEST_SIGNING_PRIVATE_KEY_PATH"
REGISTER_TRUSTED_PUBLIC_KEY_ENV = "PFC_RELEASE_REQUEST_TRUSTED_PUBLIC_KEY_PATH"
REGISTER_TRUSTED_PUBLIC_KEY_DIR_ENV = "PFC_RELEASE_REQUEST_TRUSTED_PUBLIC_KEY_DIR"
SIGNATURE_ALGORITHM = "Ed25519"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ReleaseRequestAuthenticationError(RuntimeError):
    """Raised when REGISTER authority or workflow-domain evidence is invalid."""


def ensure_workflow_domain(workflow_root: str | Path, *, create: bool) -> tuple[Path, str]:
    """Resolve and authenticate the immutable workflow domain marker."""

    root = _absolute_unlinked_path(workflow_root, label="workflow_root")
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ReleaseRequestAuthenticationError(f"workflow_root is unavailable: {root}")
    selected = str(os.environ.get(WORKFLOW_DOMAIN_ID_ENV, "")).strip().lower()
    if not selected:
        raise ReleaseRequestAuthenticationError(f"{WORKFLOW_DOMAIN_ID_ENV} is required")
    try:
        domain_id = str(uuid.UUID(selected))
    except ValueError as exc:
        raise ReleaseRequestAuthenticationError(
            f"{WORKFLOW_DOMAIN_ID_ENV} must be a canonical UUID"
        ) from exc
    if selected != domain_id:
        raise ReleaseRequestAuthenticationError(
            f"{WORKFLOW_DOMAIN_ID_ENV} must be a canonical UUID"
        )
    domain_sha256 = hashlib.sha256(
        f"{WORKFLOW_DOMAIN_SCHEMA}:{domain_id}".encode("ascii")
    ).hexdigest()
    expected = {
        "schema_version": WORKFLOW_DOMAIN_SCHEMA,
        "workflow_domain_id": domain_id,
        "workflow_domain_sha256": domain_sha256,
    }
    marker = root / WORKFLOW_DOMAIN_MARKER
    if create and not marker.exists() and not path_is_link(marker):
        if any(root.iterdir()):
            raise ReleaseRequestAuthenticationError(
                "workflow domain marker is missing after workflow activity"
            )
        raw = (json.dumps(expected, indent=2, sort_keys=True) + "\n").encode("utf-8")
        try:
            with marker.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(root)
        except FileExistsError:
            pass
        except OSError as exc:
            raise ReleaseRequestAuthenticationError(
                "cannot persist workflow domain marker"
            ) from exc
    _assert_single_link_file(marker, label="workflow domain marker")
    if _load_strict_object(marker) != expected:
        raise ReleaseRequestAuthenticationError(
            "configured workflow domain does not match pinned marker"
        )
    return root, domain_sha256


def sign_release_request(
    payload: Mapping[str, object],
    *,
    private_key_path: str | Path,
) -> dict[str, object]:
    """Sign one content-addressed release request as the REGISTER authority."""

    unsigned = dict(payload)
    unsigned.pop("request_id", None)
    unsigned.pop("register_signature", None)
    if unsigned.get("schema_version") != RELEASE_REQUEST_SCHEMA:
        raise ReleaseRequestAuthenticationError(
            f"only {RELEASE_REQUEST_SCHEMA} can be signed"
        )
    _canonical_sha256(unsigned.get("workflow_domain_sha256"), "workflow domain")
    request_id = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    signed = {**unsigned, "request_id": request_id}
    private_key = _load_private_key(private_key_path)
    signed["register_signature"] = {
        "algorithm": SIGNATURE_ALGORITHM,
        "key_id": _public_key_id(private_key.public_key()),
        "value_base64": base64.b64encode(
            private_key.sign(_canonical_json_bytes(signed))
        ).decode("ascii"),
    }
    return signed


def verify_release_request_signature(
    request: Mapping[str, object],
    *,
    trusted_public_key_path: str | Path,
) -> dict[str, object]:
    """Authenticate one signed release request and return its signed payload."""

    payload = dict(request)
    signature = payload.pop("register_signature", None)
    if not isinstance(signature, Mapping) or set(signature) != {
        "algorithm",
        "key_id",
        "value_base64",
    }:
        raise ReleaseRequestAuthenticationError("release request signature is missing or invalid")
    if signature.get("algorithm") != SIGNATURE_ALGORITHM:
        raise ReleaseRequestAuthenticationError("release request signature algorithm is unsupported")
    if payload.get("schema_version") != RELEASE_REQUEST_SCHEMA:
        raise ReleaseRequestAuthenticationError("release request schema mismatch")
    request_id = _canonical_sha256(payload.get("request_id"), "request_id")
    unsigned = dict(payload)
    unsigned.pop("request_id", None)
    if hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() != request_id:
        raise ReleaseRequestAuthenticationError("release request_id mismatch")
    _canonical_sha256(payload.get("workflow_domain_sha256"), "workflow domain")
    public_key = _load_public_key(trusted_public_key_path)
    if signature.get("key_id") != _public_key_id(public_key):
        raise ReleaseRequestAuthenticationError("release request signing key is not trusted")
    try:
        signature_bytes = base64.b64decode(
            str(signature.get("value_base64", "")), validate=True
        )
    except (TypeError, ValueError) as exc:
        raise ReleaseRequestAuthenticationError(
            "release request signature is not valid base64"
        ) from exc
    try:
        public_key.verify(signature_bytes, _canonical_json_bytes(payload))
    except InvalidSignature as exc:
        raise ReleaseRequestAuthenticationError("release request signature is invalid") from exc
    return payload


def load_registered_release_request(
    request_path: str | Path,
    *,
    workflow_root: str | Path,
    expected_run_id: str,
    allow_historical_key: bool = False,
) -> dict[str, object]:
    """Load a signed request from its exact canonical workflow-domain location."""

    request, _ = load_registered_release_request_evidence(
        request_path,
        workflow_root=workflow_root,
        expected_run_id=expected_run_id,
        allow_historical_key=allow_historical_key,
    )
    return request


def load_registered_release_request_evidence(
    request_path: str | Path,
    *,
    workflow_root: str | Path,
    expected_run_id: str,
    allow_historical_key: bool = False,
) -> tuple[dict[str, object], str]:
    """Authenticate a request and hash the same immutable bytes that were parsed."""

    workflow, workflow_domain_sha256 = ensure_workflow_domain(workflow_root, create=False)
    path = _absolute_unlinked_path(request_path, label="registered release request")
    _assert_single_link_file(path, label="registered release request")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReleaseRequestAuthenticationError(
            "cannot read registered release request"
        ) from exc
    document = _load_strict_object_bytes(raw, path=path)
    public_key = _matching_trusted_public_key(
        document,
        allow_historical=allow_historical_key,
    )
    request = verify_release_request_signature(
        document,
        trusted_public_key_path=public_key,
    )
    if str(request.get("run_id", "")) != expected_run_id:
        raise ReleaseRequestAuthenticationError("release request run_id mismatch")
    if request.get("workflow_domain_sha256") != workflow_domain_sha256:
        raise ReleaseRequestAuthenticationError("release request workflow domain mismatch")
    expected_path = (
        workflow
        / expected_run_id
        / "requests"
        / str(request["request_id"])[:32]
        / "release_request.json"
    ).resolve()
    if path != expected_path:
        raise ReleaseRequestAuthenticationError(
            "release request is outside its canonical workflow-domain location"
        )
    return dict(document), hashlib.sha256(raw).hexdigest()


def release_request_document_sha256(request_path: str | Path) -> str:
    """Hash the exact signed REGISTER document bytes."""

    try:
        path = _absolute_unlinked_path(request_path, label="registered release request")
        _assert_single_link_file(path, label="registered release request")
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ReleaseRequestAuthenticationError("cannot read release request document") from exc


def _matching_trusted_public_key(
    document: Mapping[str, object],
    *,
    allow_historical: bool,
) -> Path:
    signature = document.get("register_signature")
    if not isinstance(signature, Mapping):
        raise ReleaseRequestAuthenticationError("release request signature is missing")
    key_id = _canonical_sha256(signature.get("key_id"), "release request signing key_id")
    selected = str(os.environ.get(REGISTER_TRUSTED_PUBLIC_KEY_ENV, "")).strip()
    if not selected:
        raise ReleaseRequestAuthenticationError(f"{REGISTER_TRUSTED_PUBLIC_KEY_ENV} is required")
    candidates = [_absolute_unlinked_path(selected, label="trusted REGISTER public key")]
    if allow_historical:
        keyring = str(os.environ.get(REGISTER_TRUSTED_PUBLIC_KEY_DIR_ENV, "")).strip()
        if keyring:
            directory = _absolute_unlinked_path(keyring, label="REGISTER public keyring")
            immutable_entries = process_immutable_directory_entries(directory)
            if immutable_entries is None:
                if not directory.is_dir():
                    raise ReleaseRequestAuthenticationError(
                        "REGISTER public keyring is unavailable"
                    )
                candidates.extend(sorted(directory.glob("*.pem")))
            else:
                candidates.extend(immutable_entries)
    for candidate in candidates:
        if process_immutable_file_bytes(candidate) is not None or (
            candidate.is_file() and not path_is_link(candidate)
        ):
            try:
                if _public_key_id(_load_public_key(candidate)) == key_id:
                    return candidate
            except ReleaseRequestAuthenticationError:
                continue
    raise ReleaseRequestAuthenticationError("release request signing key is not trusted")


def _absolute_unlinked_path(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ReleaseRequestAuthenticationError(f"{label} must be absolute")
    lexical = Path(os.path.abspath(path))
    for component in (lexical, *lexical.parents):
        try:
            linked = path_is_link(component)
        except OSError as exc:
            raise ReleaseRequestAuthenticationError(f"cannot validate {label} path") from exc
        if linked:
            raise ReleaseRequestAuthenticationError(f"{label} path cannot contain a link")
    return lexical.resolve()


def _load_strict_object(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReleaseRequestAuthenticationError(f"cannot read strict JSON: {path}") from exc
    return _load_strict_object_bytes(raw, path=path)


def _load_strict_object_bytes(raw: bytes, *, path: Path) -> dict[str, object]:
    try:
        payload = load_strict_json(raw.decode("utf-8"))
    except (OSError, UnicodeError, StrictStructuredDataError, RecursionError) as exc:
        raise ReleaseRequestAuthenticationError(f"cannot read strict JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ReleaseRequestAuthenticationError(f"JSON document must be an object: {path}")
    return payload


def _load_private_key(path: str | Path) -> Ed25519PrivateKey:
    selected = _absolute_unlinked_path(path, label="REGISTER private key")
    _assert_single_link_file(selected, label="REGISTER private key")
    try:
        key = serialization.load_pem_private_key(
            read_stable_single_link_file(selected, label="REGISTER private key"),
            password=None,
        )
    except (OSError, ValueError, TypeError) as exc:
        raise ReleaseRequestAuthenticationError("cannot load REGISTER private key") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ReleaseRequestAuthenticationError("REGISTER private key is not Ed25519")
    return key


def _load_public_key(path: str | Path) -> Ed25519PublicKey:
    selected = _absolute_unlinked_path(path, label="REGISTER public key")
    _assert_single_link_file(selected, label="REGISTER public key")
    try:
        key = serialization.load_pem_public_key(
            read_stable_single_link_file(selected, label="REGISTER public key")
        )
    except (OSError, ValueError, TypeError) as exc:
        raise ReleaseRequestAuthenticationError("cannot load REGISTER public key") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ReleaseRequestAuthenticationError("REGISTER public key is not Ed25519")
    return key


def _public_key_id(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha256(value: object, label: str) -> str:
    selected = str(value or "")
    if not _SHA256_RE.fullmatch(selected):
        raise ReleaseRequestAuthenticationError(f"{label} is not a canonical SHA-256")
    return selected


def _assert_single_link_file(path: Path, *, label: str) -> None:
    if path_is_link(path) or not path.is_file():
        raise ReleaseRequestAuthenticationError(f"{label} is not a regular file")
    try:
        links = int(path.stat().st_nlink)
    except (AttributeError, OSError) as exc:
        raise ReleaseRequestAuthenticationError(f"cannot validate {label} link count") from exc
    if links != 1:
        raise ReleaseRequestAuthenticationError(f"{label} hardlinks are forbidden")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError as exc:
        raise ReleaseRequestAuthenticationError(
            "cannot open workflow domain directory for fsync"
        ) from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise ReleaseRequestAuthenticationError(
            "cannot fsync workflow domain directory"
        ) from exc
    finally:
        os.close(descriptor)


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReleaseRequestAuthenticationError("release request is not canonical JSON") from exc
