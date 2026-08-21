"""Lightweight, shared CLI contract for phase-separated LT release commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from pfc_shaping.build_identity import SOURCE_REVISION
from pfc_shaping.exit_codes import EXIT_CAS_CONFLICT as EXIT_CAS_CONFLICT
from pfc_shaping.exit_codes import EXIT_GOVERNANCE_PENDING as EXIT_GOVERNANCE_PENDING
from pfc_shaping.exit_codes import EXIT_INTEGRITY_FAILURE as EXIT_INTEGRITY_FAILURE
from pfc_shaping.exit_codes import (
    EXIT_PROJECTION_REPAIR_REQUIRED as EXIT_PROJECTION_REPAIR_REQUIRED,
)
from pfc_shaping.exit_codes import EXIT_SUCCESS as EXIT_SUCCESS
from pfc_shaping.exit_codes import EXIT_TRANSITION_BUSY as EXIT_TRANSITION_BUSY
from pfc_shaping.package_contract import (
    ALLOWED_RUNTIME_PYTHON_FILES,
    BUILD_IDENTITY_PLACEHOLDER,
    BUILD_IDENTITY_TEMPLATE,
    REQUIRED_RUNTIME_DISTRIBUTIONS,
    SEALED_RUNTIME_DISTRIBUTIONS,
)
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    path_is_link,
    paths_overlap_by_identity,
    process_immutable_directory_entries,
    process_immutable_file_bytes,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.promotion_contract import (
    public_key_id_from_private_key_path,
    public_key_id_from_public_key_path,
)
from pfc_shaping.pipeline.release_request_contract import (
    REGISTER_SIGNING_PRIVATE_KEY_ENV,
    REGISTER_TRUSTED_PUBLIC_KEY_DIR_ENV,
    REGISTER_TRUSTED_PUBLIC_KEY_ENV,
)

RUNTIME_RECEIPT_PATH_ENV = "PFC_LT_RUNTIME_RECEIPT_PATH"
RUNTIME_RECEIPT_SHA256_ENV = "PFC_LT_RUNTIME_RECEIPT_SHA256"
_RUNTIME_RECEIPT_SCHEMA = "fmv_lt_launcherless_local_runtime.v5"
_PYTHON_RUNTIME_MANIFEST_SCHEMA = (
    "fmv_lt_launcherless_python_runtime_manifest.v1"
)
_CONDA_PREFIX_RECEIPT_SCHEMA = (
    "fmv_lt_launcherless_conda_prefix_build_receipt.v3"
)
_REPO_LOCAL_CONDA_BUILD_SCHEMA = "fmv_lt_repo_local_conda_prefix_build.v1"
_REPO_LOCAL_CONDA_BUILD_STATUS = "PASS_REPO_LOCAL_MUTABLE_PATHS_NOT_PRODUCTION"
_CONDA_ARCHIVE_LOCK_SCHEMA = "fmv_lt_launcherless_conda_archive_lock.v2"
_CONDA_PREFIX_RECEIPT_STATUS = (
    "PASS_LOCAL_EXPLICIT_ARCHIVE_REPLAY_NOT_PRODUCTION"
)
_RUNTIME_CLOSURE_NAME = "governed-site-packages"
_RUNTIME_PTH_NAME = "python311._pth"
_RUNTIME_PTH_PAYLOAD = b"Lib\nDLLs\ngoverned-site-packages\n"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MD5 = re.compile(r"^[0-9a-f]{32}$")
_CONDA_ARCHIVE_PACKAGE_KEYS = {
    "name",
    "version",
    "build",
    "build_number",
    "subdir",
    "channel",
    "source_url",
    "filename",
    "md5",
    "sha256",
    "size",
    "depends",
    "constrains",
    "conda_meta_filename",
    "conda_meta_sha256",
    "cache_root",
    "archive_path",
    "archive_uri",
    "archive_payload_tree_sha256",
    "archive_payload_file_count",
    "prefix_replacement_file_count",
    "noarch_type",
}
_CONDA_TARGET_META_KEYS = {
    "filename",
    "conda_meta_filename",
    "conda_meta_sha256",
    "installed_file_count",
    "source_record_depends",
    "target_record_depends",
    "source_record_constrains",
    "target_record_constrains",
    "dependency_metadata_matches_source_record",
    "archive_verified_file_count",
    "archive_verified_tree_sha256",
    "generated_nonruntime_file_count",
    "generated_nonruntime_tree_sha256",
}

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


def assert_installed_runtime_sealed(
    *,
    runtime_receipt_path: str | Path | None = None,
    expected_runtime_receipt_sha256: str | None = None,
) -> None:
    _assert_isolated_runtime_flags()
    if runtime_receipt_path is None and expected_runtime_receipt_sha256 is None:
        _assert_launcherless_runtime_admission()
    else:
        _assert_launcherless_runtime_admission(
            runtime_receipt_path=runtime_receipt_path,
            expected_runtime_receipt_sha256=expected_runtime_receipt_sha256,
        )
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
    installed = _installed_runtime_dependency_versions()
    if installed != REQUIRED_RUNTIME_DISTRIBUTIONS:
        raise ReleaseCliIdentityError(
            "installed LT runtime dependency versions do not match the package contract"
        )


def assert_production_transition_runtime_authorized() -> None:
    """Reject production transitions from the launcherless local runtime.

    The local runtime receipt and its hash are caller-held observations. They
    support deterministic local execution, but they are not an independently
    signed production capability. Keep this guard separate so promote and
    rollback cannot accidentally inherit local admission.
    """

    assert_installed_runtime_sealed()
    raise ReleaseCliIdentityError(
        "promote and rollback require an independently signed, IT-admitted "
        "production runtime attestation; the launcherless local runtime is "
        "not production authority"
    )


def installed_runtime_dependency_versions() -> dict[str, str]:
    """Return exact direct runtime dependency versions after sealed admission."""

    assert_installed_runtime_sealed()
    return _installed_runtime_dependency_versions()


def _assert_isolated_runtime_flags() -> None:
    if not (
        sys.flags.isolated
        and sys.flags.ignore_environment
        and sys.flags.no_user_site
        and sys.flags.safe_path
        and sys.flags.dont_write_bytecode
    ):
        raise ReleaseCliIdentityError(
            "governed LT runtime requires an isolated -I -B launcher"
        )


def _assert_launcherless_runtime_admission(
    *,
    runtime_receipt_path: str | Path | None = None,
    expected_runtime_receipt_sha256: str | None = None,
) -> None:
    explicit = runtime_receipt_path is not None or expected_runtime_receipt_sha256 is not None
    if explicit and (
        runtime_receipt_path is None or expected_runtime_receipt_sha256 is None
    ):
        raise ReleaseCliIdentityError(
            "launcherless runtime receipt path and SHA-256 must be supplied together"
        )
    if explicit:
        receipt_path = Path(runtime_receipt_path)
        if not receipt_path.is_absolute():
            raise ReleaseCliIdentityError(
                "launcherless runtime receipt path must be absolute"
            )
        receipt_sha256 = str(expected_runtime_receipt_sha256).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", receipt_sha256):
            raise ReleaseCliIdentityError(
                "launcherless runtime receipt SHA-256 must be canonical"
            )
    else:
        receipt_path = _required_runtime_path_environment(RUNTIME_RECEIPT_PATH_ENV)
        receipt_sha256 = _required_runtime_sha256_environment(
            RUNTIME_RECEIPT_SHA256_ENV
        )
    receipt_payload = _stable_runtime_file(
        receipt_path,
        label="launcherless runtime receipt",
    )
    if hashlib.sha256(receipt_payload).hexdigest() != receipt_sha256:
        raise ReleaseCliIdentityError(
            "launcherless runtime receipt differs from the caller-held SHA-256"
        )
    receipt = _canonical_runtime_json(
        receipt_payload,
        label="launcherless runtime receipt",
    )
    expected_receipt_keys = {
        "schema_version",
        "status",
        "runtime_prefix",
        "python",
        "python_runtime_manifest",
        "conda_prefix_build_receipt",
        "python_dot_pth_sha256",
        "project_source_revision",
        "source_lock",
        "closure",
        "sys_path",
        "module_origins",
        "sources",
        "launcher_policy",
        "local_quality_authorization",
        "production_authorization",
    }
    if set(receipt) != expected_receipt_keys or (
        receipt.get("schema_version") != _RUNTIME_RECEIPT_SCHEMA
        or receipt.get("status") != "PASS"
        or receipt.get("local_quality_authorization") is not True
        or receipt.get("production_authorization") is not False
        or receipt.get("project_source_revision") != SOURCE_REVISION
    ):
        raise ReleaseCliIdentityError("launcherless runtime receipt policy is invalid")

    prefix = _runtime_prefix_from_receipt(receipt)
    if _same_or_nested_runtime_path(receipt_path, prefix):
        raise ReleaseCliIdentityError(
            "caller-held launcherless runtime receipt cannot be inside the prefix"
        )
    executable = Path(os.path.abspath(sys.executable))
    if os.path.normcase(str(executable)) != os.path.normcase(
        str(prefix / "python.exe")
    ):
        raise ReleaseCliIdentityError(
            "launcherless runtime executable differs from the admitted prefix"
        )

    python_record = receipt.get("python")
    if not isinstance(python_record, dict) or (
        python_record.get("version") != "3.11.13"
        or python_record.get("implementation") != "CPython"
        or python_record.get("pointer_bits") != 64
        or os.path.normcase(str(python_record.get("executable", "")))
        != os.path.normcase(str(executable))
        or not _is_sha256(python_record.get("python_sha256"))
    ):
        raise ReleaseCliIdentityError("launcherless runtime Python policy is invalid")
    executable_payload = _stable_runtime_file(
        executable,
        label="launcherless runtime Python executable",
    )
    if hashlib.sha256(executable_payload).hexdigest() != python_record[
        "python_sha256"
    ]:
        raise ReleaseCliIdentityError("launcherless runtime Python identity diverges")

    closure = prefix / _RUNTIME_CLOSURE_NAME
    expected_sys_path = [
        str(prefix / "Lib"),
        str(prefix / "DLLs"),
        str(closure),
    ]
    if not isinstance(receipt.get("sys_path"), list) or not _same_path_list(
        receipt["sys_path"], expected_sys_path
    ) or not _same_path_list(sys.path, expected_sys_path):
        raise ReleaseCliIdentityError("launcherless runtime sys.path is not exact")

    pth_payload = _stable_runtime_file(
        prefix / _RUNTIME_PTH_NAME,
        label="launcherless runtime python311._pth",
    )
    if pth_payload != _RUNTIME_PTH_PAYLOAD or (
        not _is_sha256(receipt.get("python_dot_pth_sha256"))
        or hashlib.sha256(pth_payload).hexdigest()
        != receipt["python_dot_pth_sha256"]
    ):
        raise ReleaseCliIdentityError("launcherless runtime python311._pth diverges")

    manifest_record = receipt.get("python_runtime_manifest")
    if not isinstance(manifest_record, dict) or set(manifest_record) != {
        "path",
        "sha256",
        "tree_sha256",
        "file_count",
    } or not _is_sha256(manifest_record.get("sha256")):
        raise ReleaseCliIdentityError("Python runtime manifest receipt is invalid")
    manifest_path = _absolute_runtime_path(
        manifest_record["path"],
        label="caller-held Python runtime manifest",
    )
    if _same_or_nested_runtime_path(manifest_path, prefix):
        raise ReleaseCliIdentityError(
            "caller-held Python runtime manifest cannot be inside the runtime prefix"
        )
    manifest_payload = _stable_runtime_file(
        manifest_path,
        label="caller-held Python runtime manifest",
    )
    if hashlib.sha256(manifest_payload).hexdigest() != manifest_record["sha256"]:
        raise ReleaseCliIdentityError(
            "Python runtime manifest differs from its caller-held SHA-256"
        )
    manifest = _canonical_runtime_json(
        manifest_payload,
        label="caller-held Python runtime manifest",
    )
    _validate_runtime_prefix_manifest(
        prefix=prefix,
        manifest=manifest,
        receipt_record=manifest_record,
    )
    conda_receipt_record = receipt.get("conda_prefix_build_receipt")
    _validate_conda_prefix_build_receipt(
        prefix=prefix,
        receipt_record=conda_receipt_record,
        python_manifest=manifest,
        python_manifest_record=manifest_record,
    )

    closure_record = receipt.get("closure")
    if not isinstance(closure_record, dict) or set(closure_record) != {
        "path",
        "tree_sha256",
        "file_count",
        "distribution_count",
        "distributions",
    } or os.path.normcase(str(closure_record.get("path", ""))) != os.path.normcase(
        str(closure)
    ):
        raise ReleaseCliIdentityError("launcherless runtime closure receipt is invalid")
    closure_tree = _runtime_tree_inventory(closure)
    if closure_tree != {
        "tree_sha256": closure_record.get("tree_sha256"),
        "file_count": closure_record.get("file_count"),
    }:
        raise ReleaseCliIdentityError("launcherless runtime closure bytes diverge")
    distributions = _sealed_distribution_inventory(closure)
    if distributions != SEALED_RUNTIME_DISTRIBUTIONS or (
        closure_record.get("distributions") != distributions
        or closure_record.get("distribution_count") != len(distributions)
    ):
        raise ReleaseCliIdentityError(
            "launcherless runtime distribution inventory diverges"
        )

    _validate_runtime_prefix_manifest(
        prefix=prefix,
        manifest=manifest,
        receipt_record=manifest_record,
    )
    _validate_conda_prefix_build_receipt(
        prefix=prefix,
        receipt_record=conda_receipt_record,
        python_manifest=manifest,
        python_manifest_record=manifest_record,
    )
    if _runtime_tree_inventory(closure) != closure_tree:
        raise ReleaseCliIdentityError(
            "launcherless runtime changed during launch-time admission"
        )


def _required_runtime_path_environment(name: str) -> Path:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        raise ReleaseCliIdentityError(f"{name} is required")
    return _absolute_runtime_path(raw, label=name)


def _required_runtime_sha256_environment(name: str) -> str:
    value = str(os.environ.get(name, "")).strip().lower()
    if _SHA256.fullmatch(value) is None:
        raise ReleaseCliIdentityError(f"{name} must be a canonical SHA-256")
    return value


def _absolute_runtime_path(value: object, *, label: str) -> Path:
    try:
        return assert_absolute_path_has_no_links(Path(str(value)))
    except (OSError, TypeError, ValueError) as exc:
        raise ReleaseCliIdentityError(f"{label} path is invalid") from exc


def _stable_runtime_file(path: Path, *, label: str) -> bytes:
    try:
        return read_stable_single_link_file(path, label=label)
    except (OSError, ValueError) as exc:
        raise ReleaseCliIdentityError(f"{label} cannot be read safely") from exc


def _canonical_runtime_json(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseCliIdentityError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict) or json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii") != payload:
        raise ReleaseCliIdentityError(f"{label} is not canonical JSON")
    return value


def _canonical_runtime_jsonl(payload: bytes, *, label: str) -> dict[str, object]:
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ReleaseCliIdentityError(f"{label} is not canonical JSONL")
    value = _canonical_runtime_json(payload[:-1], label=label)
    return value


def _runtime_prefix_from_receipt(receipt: dict[str, object]) -> Path:
    prefix = _absolute_runtime_path(
        receipt.get("runtime_prefix"),
        label="launcherless runtime prefix",
    )
    if not prefix.is_dir():
        raise ReleaseCliIdentityError("launcherless runtime prefix is unavailable")
    return prefix


def _same_path_list(actual: object, expected: list[str]) -> bool:
    return isinstance(actual, list) and len(actual) == len(expected) and all(
        isinstance(item, str)
        and os.path.normcase(os.path.abspath(item))
        == os.path.normcase(os.path.abspath(expected[index]))
        for index, item in enumerate(actual)
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _same_or_nested_runtime_path(path: Path, parent: Path) -> bool:
    return paths_overlap_by_identity(path, parent)


def _validate_runtime_prefix_manifest(
    *,
    prefix: Path,
    manifest: dict[str, object],
    receipt_record: dict[str, object],
) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "runtime_prefix",
        "python_version",
        "python_executable_sha256",
        "python_dll_sha256",
        "tree_sha256",
        "file_count",
        "files",
        "capture_policy",
        "production_authorization",
    }
    if set(manifest) != expected_keys or (
        manifest.get("schema_version") != _PYTHON_RUNTIME_MANIFEST_SCHEMA
        or manifest.get("status") != "PASS"
        or os.path.normcase(str(manifest.get("runtime_prefix", "")))
        != os.path.normcase(str(prefix))
        or manifest.get("python_version") != "3.11.13"
        or manifest.get("capture_policy")
        != "CALLER_HELD_BEFORE_FIRST_TARGET_EXECUTION"
        or manifest.get("production_authorization") is not False
        or receipt_record.get("tree_sha256") != manifest.get("tree_sha256")
        or receipt_record.get("file_count") != manifest.get("file_count")
    ):
        raise ReleaseCliIdentityError("Python runtime manifest policy is invalid")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ReleaseCliIdentityError("Python runtime manifest inventory is invalid")
    expected: list[tuple[str, str, int]] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size"}:
            raise ReleaseCliIdentityError(
                "Python runtime manifest file record is invalid"
            )
        relative = record.get("path")
        sha256 = record.get("sha256")
        size = record.get("size")
        if not _portable_runtime_relative_path(relative) or not _is_sha256(
            sha256
        ) or not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ReleaseCliIdentityError(
                "Python runtime manifest file record is invalid"
            )
        expected.append((relative, sha256, size))
    if expected != sorted(expected) or len(
        {item[0].casefold() for item in expected}
    ) != len(expected):
        raise ReleaseCliIdentityError("Python runtime manifest inventory is ambiguous")
    actual, tree = _runtime_prefix_base_inventory(prefix)
    if actual != expected or tree != {
        "tree_sha256": manifest.get("tree_sha256"),
        "file_count": manifest.get("file_count"),
    }:
        raise ReleaseCliIdentityError(
            "Python runtime prefix differs from its caller-held manifest"
        )
    by_path = {path: sha256 for path, sha256, _ in actual}
    if by_path.get("python.exe") != manifest.get(
        "python_executable_sha256"
    ) or by_path.get("python311.dll") != manifest.get("python_dll_sha256"):
        raise ReleaseCliIdentityError("Python runtime executable or ABI DLL diverges")


def _validate_conda_prefix_build_receipt(
    *,
    prefix: Path,
    receipt_record: object,
    python_manifest: dict[str, object],
    python_manifest_record: dict[str, object],
) -> None:
    expected_record_keys = {
        "path",
        "sha256",
        "prefix_receipt_id",
        "archive_lock_sha256",
        "archive_lock_id",
        "archive_set_id",
        "package_count",
        "file_count",
        "archive_verified_file_count",
        "generated_nonruntime_file_count",
        "repo_local_build_receipt",
    }
    if (
        not isinstance(receipt_record, dict)
        or set(receipt_record) != expected_record_keys
        or not _is_sha256(receipt_record.get("sha256"))
        or not _is_sha256(receipt_record.get("prefix_receipt_id"))
        or not _is_sha256(receipt_record.get("archive_lock_sha256"))
        or not _is_sha256(receipt_record.get("archive_lock_id"))
        or not _is_sha256(receipt_record.get("archive_set_id"))
    ):
        raise ReleaseCliIdentityError("Conda prefix build receipt record is invalid")
    receipt_path = _absolute_runtime_path(
        receipt_record["path"],
        label="caller-held Conda prefix build receipt",
    )
    if _same_or_nested_runtime_path(receipt_path, prefix):
        raise ReleaseCliIdentityError(
            "caller-held Conda prefix build receipt cannot be inside the prefix"
        )
    payload = _stable_runtime_file(
        receipt_path,
        label="caller-held Conda prefix build receipt",
    )
    if hashlib.sha256(payload).hexdigest() != receipt_record["sha256"]:
        raise ReleaseCliIdentityError(
            "Conda prefix build receipt differs from its caller-held SHA-256"
        )
    receipt = _canonical_runtime_jsonl(
        payload,
        label="caller-held Conda prefix build receipt",
    )
    expected_keys = {
        "schema_version",
        "status",
        "archive_lock",
        "explicit_spec",
        "runtime_prefix",
        "conda_history",
        "python_runtime_manifest",
        "target_conda_meta",
        "repo_local_build_receipt",
        "package_count",
        "file_count",
        "archive_verified_file_count",
        "generated_nonruntime_file_count",
        "build_policy",
        "prefix_receipt_id",
        "production_authorization",
        "promotion_gate",
    }
    expected_policy = {
        "solver_used": False,
        "network_required": False,
        "copy_only": True,
        "new_prefix_required": True,
        "target_python_executed_before_manifest": False,
        "dependency_metadata_authority": (
            "ARCHIVE_PATHS_BYTES_EXCEPT_DECLARED_GENERATED_NONRUNTIME"
        ),
    }
    archive_lock = receipt.get("archive_lock")
    repo_local_build_receipt = receipt.get("repo_local_build_receipt")
    manifest_binding = receipt.get("python_runtime_manifest")
    target_meta = receipt.get("target_conda_meta")
    package_count = receipt.get("package_count")
    if set(receipt) != expected_keys or (
        receipt.get("schema_version") != _CONDA_PREFIX_RECEIPT_SCHEMA
        or receipt.get("status") != _CONDA_PREFIX_RECEIPT_STATUS
        or os.path.normcase(str(receipt.get("runtime_prefix", "")))
        != os.path.normcase(str(prefix))
        or receipt.get("build_policy") != expected_policy
        or receipt.get("production_authorization") is not False
        or receipt.get("promotion_gate") is not False
        or receipt.get("prefix_receipt_id")
        != receipt_record["prefix_receipt_id"]
        or receipt.get("file_count") != receipt_record["file_count"]
        or receipt.get("file_count") != python_manifest.get("file_count")
        or receipt.get("archive_verified_file_count")
        != receipt_record["archive_verified_file_count"]
        or receipt.get("generated_nonruntime_file_count")
        != receipt_record["generated_nonruntime_file_count"]
        or package_count != receipt_record["package_count"]
        or not isinstance(package_count, int)
        or isinstance(package_count, bool)
        or package_count <= 0
        or not isinstance(target_meta, list)
        or len(target_meta) != package_count
    ):
        raise ReleaseCliIdentityError("Conda prefix build receipt policy is invalid")
    if not isinstance(archive_lock, dict) or set(archive_lock) != {
        "path",
        "sha256",
        "archive_lock_id",
        "archive_set_id",
    } or (
        archive_lock.get("archive_set_id") != receipt_record["archive_set_id"]
        or archive_lock.get("archive_lock_id")
        != receipt_record["archive_lock_id"]
        or archive_lock.get("sha256")
        != receipt_record["archive_lock_sha256"]
        or not _is_sha256(archive_lock.get("sha256"))
        or not _is_sha256(archive_lock.get("archive_lock_id"))
    ):
        raise ReleaseCliIdentityError("Conda prefix archive lock identity is invalid")
    _validate_repo_local_conda_build_receipt(
        prefix=prefix,
        identity=repo_local_build_receipt,
        explicit_spec=receipt.get("explicit_spec"),
    )
    if repo_local_build_receipt != receipt_record["repo_local_build_receipt"]:
        raise ReleaseCliIdentityError("repo-local Conda receipt record diverges")
    archive_lock_document = _validate_launch_conda_archive_lock(
        prefix=prefix,
        record=archive_lock,
    )
    if not isinstance(manifest_binding, dict) or set(manifest_binding) != {
        "path",
        "sha256",
        "tree_sha256",
        "file_count",
        "python_executable_sha256",
        "python_dll_sha256",
    } or (
        manifest_binding.get("path") != python_manifest_record.get("path")
        or manifest_binding.get("sha256") != python_manifest_record.get("sha256")
        or manifest_binding.get("tree_sha256")
        != python_manifest.get("tree_sha256")
        or manifest_binding.get("file_count") != python_manifest.get("file_count")
        or manifest_binding.get("python_executable_sha256")
        != python_manifest.get("python_executable_sha256")
        or manifest_binding.get("python_dll_sha256")
        != python_manifest.get("python_dll_sha256")
    ):
        raise ReleaseCliIdentityError("Conda prefix Python manifest binding is invalid")
    _validate_launch_conda_prefix_derivations(
        prefix=prefix,
        receipt=receipt,
        receipt_record=receipt_record,
        archive_lock_document=archive_lock_document,
        python_manifest=python_manifest,
        python_manifest_record=python_manifest_record,
    )


def _validate_repo_local_conda_build_receipt(
    *,
    prefix: Path,
    identity: object,
    explicit_spec: object,
) -> None:
    expected_identity_keys = {
        "path",
        "sha256",
        "schema_version",
        "status",
        "command_sha256",
        "external_guard_sha256",
        "external_guard_unchanged",
    }
    if (
        not isinstance(identity, dict)
        or set(identity) != expected_identity_keys
        or identity.get("schema_version") != _REPO_LOCAL_CONDA_BUILD_SCHEMA
        or identity.get("status") != _REPO_LOCAL_CONDA_BUILD_STATUS
        or not _is_sha256(identity.get("sha256"))
        or not _is_sha256(identity.get("command_sha256"))
        or not _is_sha256(identity.get("external_guard_sha256"))
        or identity.get("external_guard_unchanged") is not True
    ):
        raise ReleaseCliIdentityError("repo-local Conda receipt identity is invalid")
    path = _absolute_runtime_path(
        identity["path"], label="repo-local Conda prefix build receipt"
    )
    build_root = _absolute_runtime_path(prefix.parent, label="runtime build root")
    if not _path_is_within(path, build_root) or _same_or_nested_runtime_path(
        path, prefix
    ):
        raise ReleaseCliIdentityError("repo-local Conda receipt path is invalid")
    payload = _stable_runtime_file(path, label="repo-local Conda prefix build receipt")
    if hashlib.sha256(payload).hexdigest() != identity["sha256"]:
        raise ReleaseCliIdentityError("repo-local Conda receipt bytes diverge")
    document = _canonical_runtime_jsonl(
        payload, label="repo-local Conda prefix build receipt"
    )
    expected_keys = {
        "admin_required",
        "command_sha256",
        "conda_python",
        "explicit_spec",
        "external_guard_after",
        "external_guard_before",
        "external_guard_unchanged",
        "mutable_paths",
        "network_required",
        "prefix_ready",
        "production_authorization",
        "promotion_gate",
        "runtime_prefix",
        "schema_version",
        "status",
        "target_exit_code",
        "target_failure_class",
        "target_stderr_sha256",
        "target_stdout_sha256",
        "work_root",
    }
    before = document.get("external_guard_before")
    spec_evidence = document.get("explicit_spec")
    if (
        set(document) != expected_keys
        or document.get("schema_version") != _REPO_LOCAL_CONDA_BUILD_SCHEMA
        or document.get("status") != _REPO_LOCAL_CONDA_BUILD_STATUS
        or document.get("runtime_prefix") != str(prefix)
        or document.get("command_sha256") != identity["command_sha256"]
        or document.get("target_exit_code") != 0
        or document.get("target_failure_class") is not None
        or document.get("prefix_ready") is not True
        or document.get("external_guard_unchanged") is not True
        or not isinstance(before, dict)
        or not before
        or before != document.get("external_guard_after")
        or document.get("network_required") is not False
        or document.get("admin_required") is not False
        or document.get("production_authorization") is not False
        or document.get("promotion_gate") is not False
        or not _is_sha256(document.get("target_stdout_sha256"))
        or not _is_sha256(document.get("target_stderr_sha256"))
        or not isinstance(explicit_spec, dict)
        or not isinstance(spec_evidence, dict)
        or spec_evidence.get("path") != explicit_spec.get("path")
        or spec_evidence.get("sha256") != explicit_spec.get("sha256")
        or not _is_sha256(spec_evidence.get("sha256"))
    ):
        raise ReleaseCliIdentityError("repo-local Conda receipt policy is invalid")
    guard_sha256 = hashlib.sha256(
        json.dumps(before, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if guard_sha256 != identity["external_guard_sha256"]:
        raise ReleaseCliIdentityError("repo-local Conda guard identity diverges")
    for guard_path, evidence in before.items():
        _absolute_runtime_path(guard_path, label="repo-local Conda external guard")
        if not isinstance(evidence, dict) or evidence.get("exists") not in {
            True,
            False,
        }:
            raise ReleaseCliIdentityError("repo-local Conda guard evidence is invalid")
        if evidence["exists"] is True and (
            set(evidence) != {"exists", "mtime_ns", "sha256", "size_bytes"}
            or not _is_sha256(evidence.get("sha256"))
            or not _is_nonnegative_int(evidence.get("size_bytes"))
            or not _is_nonnegative_int(evidence.get("mtime_ns"))
        ):
            raise ReleaseCliIdentityError("repo-local Conda guard evidence is invalid")
        if evidence["exists"] is False and set(evidence) != {"exists"}:
            raise ReleaseCliIdentityError("repo-local Conda guard evidence is invalid")
    work_root = _absolute_runtime_path(
        document.get("work_root"), label="repo-local Conda work root"
    )
    mutable = document.get("mutable_paths")
    if not _path_is_within(work_root, build_root) or not isinstance(mutable, dict):
        raise ReleaseCliIdentityError("repo-local Conda mutable root is invalid")
    if set(mutable) != {"home", "conda_pkgs", "conda_envs", "temp", "condarc"}:
        raise ReleaseCliIdentityError("repo-local Conda mutable inventory is invalid")
    for value in mutable.values():
        mutable_path = _absolute_runtime_path(
            value, label="repo-local Conda mutable path"
        )
        if not _path_is_within(mutable_path, work_root):
            raise ReleaseCliIdentityError("repo-local Conda mutable path escapes")


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath(
            (os.path.normcase(str(path)), os.path.normcase(str(root)))
        ) == os.path.normcase(str(root))
    except ValueError:
        return False


def _validate_launch_conda_archive_lock(
    *,
    prefix: Path,
    record: dict[str, object],
) -> dict[str, object]:
    path = _absolute_runtime_path(record["path"], label="Conda archive lock")
    if _same_or_nested_runtime_path(path, prefix):
        raise ReleaseCliIdentityError("Conda archive lock cannot be inside the prefix")
    payload = _stable_runtime_file(path, label="Conda archive lock")
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ReleaseCliIdentityError("Conda archive lock bytes diverge")
    lock = _canonical_runtime_jsonl(payload, label="Conda archive lock")
    expected_keys = {
        "schema_version",
        "status",
        "source_runtime_prefix",
        "package_cache_roots",
        "conda_history",
        "packages",
        "package_count",
        "total_archive_bytes",
        "archive_set_id",
        "archive_lock_id",
        "explicit_spec_lines",
        "network_required",
        "admin_required",
        "production_authorization",
        "promotion_gate",
    }
    packages = lock.get("packages")
    if set(lock) != expected_keys or (
        lock.get("schema_version") != _CONDA_ARCHIVE_LOCK_SCHEMA
        or lock.get("status") != "PASS_LOCAL_ARCHIVE_LOCK_NOT_PRODUCTION"
        or lock.get("network_required") is not False
        or lock.get("admin_required") is not False
        or lock.get("production_authorization") is not False
        or lock.get("promotion_gate") is not False
        or lock.get("archive_lock_id") != record["archive_lock_id"]
        or lock.get("archive_set_id") != record["archive_set_id"]
        or not isinstance(packages, list)
        or not packages
        or lock.get("package_count") != len(packages)
        or not _is_positive_int(lock.get("total_archive_bytes"))
    ):
        raise ReleaseCliIdentityError("Conda archive lock policy is invalid")
    portable_packages: list[dict[str, object]] = []
    total_archive_bytes = 0
    filenames: set[str] = set()
    for package in packages:
        if (
            not isinstance(package, dict)
            or set(package) != _CONDA_ARCHIVE_PACKAGE_KEYS
            or not all(
                isinstance(package.get(key), str) and package.get(key)
                for key in ("name", "version", "build", "channel", "source_url")
            )
            or package.get("subdir") not in {"win-64", "noarch"}
            or package.get("noarch_type") not in {"none", "generic", "python"}
            or not _is_nonnegative_int(package.get("build_number"))
            or not _is_positive_int(package.get("size"))
            or not _is_nonnegative_int(package.get("archive_payload_file_count"))
            or not _is_nonnegative_int(package.get("prefix_replacement_file_count"))
            or not _is_sha256(package.get("sha256"))
            or not _is_sha256(package.get("conda_meta_sha256"))
            or not _is_sha256(package.get("archive_payload_tree_sha256"))
            or not isinstance(package.get("md5"), str)
            or _MD5.fullmatch(package["md5"]) is None
            or not _is_string_list(package.get("depends"))
            or not _is_string_list(package.get("constrains"))
        ):
            raise ReleaseCliIdentityError("Conda archive package record is invalid")
        filename = package.get("filename")
        expected_meta = (
            f"{package['name']}-{package['version']}-{package['build']}.json"
        )
        if (
            not isinstance(filename, str)
            or not filename.endswith((".conda", ".tar.bz2"))
            or Path(filename).name != filename
            or filename in filenames
            or package.get("conda_meta_filename") != expected_meta
        ):
            raise ReleaseCliIdentityError("Conda archive package identity is invalid")
        filenames.add(filename)
        archive_path = _absolute_runtime_path(
            package.get("archive_path"),
            label="retained Conda archive",
        )
        if _same_or_nested_runtime_path(archive_path, prefix):
            raise ReleaseCliIdentityError(
                "retained Conda archive cannot be inside the prefix"
            )
        archive_payload = _stable_runtime_file(
            archive_path,
            label="retained Conda archive",
        )
        if (
            archive_path.name != package.get("filename")
            or len(archive_payload) != package.get("size")
            or hashlib.sha256(archive_payload).hexdigest()
            != package.get("sha256")
        ):
            raise ReleaseCliIdentityError("retained Conda archive bytes diverge")
        total_archive_bytes += len(archive_payload)
        portable_packages.append(
            {
                key: value
                for key, value in package.items()
                if key not in {"cache_root", "archive_path", "archive_uri"}
            }
        )
    canonical_portable = json.dumps(
        portable_packages,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    archive_set_id = hashlib.sha256(canonical_portable).hexdigest()
    history = lock.get("conda_history")
    if (
        not isinstance(history, dict)
        or set(history) != {"path", "sha256", "size"}
        or not _is_sha256(history.get("sha256"))
        or not _is_positive_int(history.get("size"))
    ):
        raise ReleaseCliIdentityError("Conda source history identity is invalid")
    explicit_lines = lock.get("explicit_spec_lines")
    expected_explicit_lines = ["@EXPLICIT"] + [
        f"{package['archive_uri']}#{package['md5']}" for package in packages
    ]
    if explicit_lines != expected_explicit_lines:
        raise ReleaseCliIdentityError("Conda explicit spec lines are not derived")
    lock_identity = {
        "schema_version": _CONDA_ARCHIVE_LOCK_SCHEMA,
        "archive_set_id": archive_set_id,
        "conda_history_sha256": history["sha256"],
        "package_count": len(packages),
        "total_archive_bytes": total_archive_bytes,
    }
    lock_id = hashlib.sha256(
        json.dumps(lock_identity, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()
    if (
        archive_set_id != lock.get("archive_set_id")
        or lock_id != lock.get("archive_lock_id")
        or total_archive_bytes != lock.get("total_archive_bytes")
    ):
        raise ReleaseCliIdentityError("Conda archive lock IDs are not derived")
    return lock


def _validate_launch_conda_prefix_derivations(
    *,
    prefix: Path,
    receipt: dict[str, object],
    receipt_record: dict[str, object],
    archive_lock_document: dict[str, object],
    python_manifest: dict[str, object],
    python_manifest_record: dict[str, object],
) -> None:
    explicit = receipt.get("explicit_spec")
    if not isinstance(explicit, dict) or set(explicit) != {
        "path",
        "sha256",
        "size",
        "line_count",
    }:
        raise ReleaseCliIdentityError("Conda explicit spec record is invalid")
    spec_path = _absolute_runtime_path(explicit["path"], label="Conda explicit spec")
    if _same_or_nested_runtime_path(spec_path, prefix):
        raise ReleaseCliIdentityError("Conda explicit spec cannot be inside the prefix")
    spec_payload = _stable_runtime_file(spec_path, label="Conda explicit spec")
    expected_spec = (
        "\n".join(
            str(item) for item in archive_lock_document["explicit_spec_lines"]
        )
        + "\n"
    ).encode("ascii")
    if (
        spec_payload != expected_spec
        or hashlib.sha256(spec_payload).hexdigest() != explicit.get("sha256")
        or len(spec_payload) != explicit.get("size")
        or len(archive_lock_document["explicit_spec_lines"])
        != explicit.get("line_count")
    ):
        raise ReleaseCliIdentityError("Conda explicit spec derivation is invalid")
    history = receipt.get("conda_history")
    if not isinstance(history, dict) or set(history) != {"path", "sha256", "size"}:
        raise ReleaseCliIdentityError("Conda replay history record is invalid")
    history_path = _absolute_runtime_path(history["path"], label="Conda replay history")
    history_payload = _stable_runtime_file(history_path, label="Conda replay history")
    if (
        os.path.normcase(str(history_path))
        != os.path.normcase(str(prefix / "conda-meta" / "history"))
        or hashlib.sha256(history_payload).hexdigest() != history.get("sha256")
        or len(history_payload) != history.get("size")
    ):
        raise ReleaseCliIdentityError("Conda replay history binding is invalid")
    target_meta = receipt.get("target_conda_meta")
    if not isinstance(target_meta, list):
        raise ReleaseCliIdentityError("Conda target metadata is invalid")
    locked_by_filename = {
        item["filename"]: item for item in archive_lock_document["packages"]
    }
    archive_count = 0
    generated_count = 0
    seen: set[str] = set()
    for item in target_meta:
        if (
            not isinstance(item, dict)
            or set(item) != _CONDA_TARGET_META_KEYS
            or not isinstance(item.get("filename"), str)
            or item["filename"] in seen
            or item["filename"] not in locked_by_filename
            or not _is_sha256(item.get("conda_meta_sha256"))
            or not _is_sha256(item.get("archive_verified_tree_sha256"))
            or not _is_sha256(item.get("generated_nonruntime_tree_sha256"))
            or not _is_nonnegative_int(item.get("installed_file_count"))
            or not _is_nonnegative_int(item.get("archive_verified_file_count"))
            or not _is_nonnegative_int(item.get("generated_nonruntime_file_count"))
            or not _is_string_list(item.get("source_record_depends"))
            or not _is_string_list(item.get("target_record_depends"))
            or not _is_string_list(item.get("source_record_constrains"))
            or not _is_string_list(item.get("target_record_constrains"))
            or not isinstance(
                item.get("dependency_metadata_matches_source_record"), bool
            )
        ):
            raise ReleaseCliIdentityError("Conda target metadata record is invalid")
        locked = locked_by_filename[item["filename"]]
        if (
            item.get("conda_meta_filename") != locked["conda_meta_filename"]
            or item.get("source_record_depends") != locked["depends"]
            or item.get("source_record_constrains") != locked["constrains"]
            or item.get("dependency_metadata_matches_source_record")
            != (
                item.get("target_record_depends") == locked["depends"]
                and item.get("target_record_constrains") == locked["constrains"]
            )
            or item["archive_verified_file_count"]
            != locked["archive_payload_file_count"]
            or item["installed_file_count"]
            != item["archive_verified_file_count"]
            + item["generated_nonruntime_file_count"]
        ):
            raise ReleaseCliIdentityError("Conda target metadata diverges from lock")
        metadata_path = _absolute_runtime_path(
            prefix / "conda-meta" / str(item["conda_meta_filename"]),
            label="Conda target package metadata",
        )
        metadata_payload = _stable_runtime_file(
            metadata_path,
            label="Conda target package metadata",
        )
        if hashlib.sha256(metadata_payload).hexdigest() != item["conda_meta_sha256"]:
            raise ReleaseCliIdentityError("Conda target package metadata bytes diverge")
        try:
            metadata_document = json.loads(metadata_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReleaseCliIdentityError(
                "Conda target package metadata is invalid"
            ) from exc
        if not isinstance(metadata_document, dict):
            raise ReleaseCliIdentityError(
                "Conda target package metadata is invalid"
            )
        actual_depends = metadata_document.get("depends", [])
        actual_constrains = metadata_document.get("constrains", [])
        if (
            not isinstance(actual_depends, list)
            or not all(isinstance(value, str) for value in actual_depends)
            or not isinstance(actual_constrains, list)
            or not all(isinstance(value, str) for value in actual_constrains)
            or sorted(actual_depends) != item["target_record_depends"]
            or sorted(actual_constrains) != item["target_record_constrains"]
            or metadata_document.get("fn") != locked["filename"]
            or metadata_document.get("sha256") != locked["sha256"]
        ):
            raise ReleaseCliIdentityError("Conda target package record diverges")
        seen.add(item["filename"])
        archive_count += item["archive_verified_file_count"]
        generated_count += item["generated_nonruntime_file_count"]
    if (
        seen != set(locked_by_filename)
        or len(target_meta) != receipt["package_count"]
        or archive_count != receipt["archive_verified_file_count"]
        or generated_count != receipt["generated_nonruntime_file_count"]
    ):
        raise ReleaseCliIdentityError("Conda target metadata aggregates diverge")
    core = {
        "archive_lock_sha256": receipt_record["archive_lock_sha256"],
        "archive_lock_id": receipt_record["archive_lock_id"],
        "archive_set_id": receipt_record["archive_set_id"],
        "explicit_spec_sha256": explicit["sha256"],
        "python_runtime_manifest_sha256": python_manifest_record["sha256"],
        "python_runtime_tree_sha256": python_manifest["tree_sha256"],
        "target_conda_meta_sha256": hashlib.sha256(
            json.dumps(target_meta, sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            )
        ).hexdigest(),
        "package_count": receipt["package_count"],
        "file_count": receipt["file_count"],
        "archive_verified_file_count": archive_count,
        "generated_nonruntime_file_count": generated_count,
        "repo_local_build_receipt_sha256": receipt_record[
            "repo_local_build_receipt"
        ]["sha256"],
        "repo_local_external_guard_sha256": receipt_record[
            "repo_local_build_receipt"
        ]["external_guard_sha256"],
    }
    prefix_id = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    if prefix_id != receipt_record["prefix_receipt_id"]:
        raise ReleaseCliIdentityError("Conda prefix receipt ID is not derived")


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_string_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and value == sorted(value)
        and len(value) == len(set(value))
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def _portable_runtime_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = Path(value)
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


def _runtime_prefix_base_inventory(
    prefix: Path,
) -> tuple[list[tuple[str, str, int]], dict[str, object]]:
    files: list[tuple[str, str, int]] = []
    for path in sorted(prefix.rglob("*")):
        relative = path.relative_to(prefix).as_posix()
        first = relative.split("/", maxsplit=1)[0].casefold()
        if first == _RUNTIME_CLOSURE_NAME.casefold() or relative.casefold() == _RUNTIME_PTH_NAME.casefold():
            continue
        path_stat = path.stat(follow_symlinks=False)
        if path_is_link(path):
            raise ReleaseCliIdentityError(
                f"Python runtime prefix contains a link: {relative}"
            )
        if path.is_dir():
            continue
        if not stat.S_ISREG(path_stat.st_mode) or int(path_stat.st_nlink) != 1:
            raise ReleaseCliIdentityError(
                f"Python runtime prefix file is not mono-linked: {relative}"
            )
        payload = _stable_runtime_file(path, label=f"Python runtime file {relative}")
        files.append((relative, hashlib.sha256(payload).hexdigest(), len(payload)))
    files.sort()
    return files, _runtime_inventory_digest(files)


def _runtime_tree_inventory(root: Path) -> dict[str, object]:
    if not root.is_dir() or path_is_link(root):
        raise ReleaseCliIdentityError("launcherless runtime closure is unavailable")
    files: list[tuple[str, str, int]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        path_stat = path.stat(follow_symlinks=False)
        if path_is_link(path):
            raise ReleaseCliIdentityError(
                f"launcherless runtime closure contains a link: {relative}"
            )
        if path.is_dir():
            continue
        lowered = relative.lower()
        if not stat.S_ISREG(path_stat.st_mode) or int(path_stat.st_nlink) != 1 or (
            lowered.endswith((".pth", ".pyc", ".pyo", ".exe", ".cmd", ".bat", ".ps1"))
            or "__pycache__" in {part.lower() for part in Path(relative).parts}
        ):
            raise ReleaseCliIdentityError(
                f"launcherless runtime closure contains a forbidden file: {relative}"
            )
        payload = _stable_runtime_file(
            path,
            label=f"launcherless runtime closure file {relative}",
        )
        files.append((relative, hashlib.sha256(payload).hexdigest(), len(payload)))
    files.sort()
    return _runtime_inventory_digest(files)


def _runtime_inventory_digest(
    files: list[tuple[str, str, int]],
) -> dict[str, object]:
    digest = hashlib.sha256()
    for relative, sha256, _ in files:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\0")
    return {"tree_sha256": digest.hexdigest(), "file_count": len(files)}


def _sealed_distribution_inventory(root: Path) -> dict[str, str]:
    installed: dict[str, str] = {}
    for distribution in metadata.distributions(path=[str(root)]):
        name = distribution.metadata.get("Name")
        if not name:
            raise ReleaseCliIdentityError(
                "launcherless runtime distribution has no canonical name"
            )
        normalized = re.sub(r"[-_.]+", "-", name.strip().lower())
        if normalized in installed:
            raise ReleaseCliIdentityError(
                "launcherless runtime distribution inventory is ambiguous"
            )
        installed[normalized] = distribution.version
    return installed


def _installed_runtime_dependency_versions() -> dict[str, str]:
    installed: dict[str, str] = {}
    for name in REQUIRED_RUNTIME_DISTRIBUTIONS:
        try:
            installed[name] = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise ReleaseCliIdentityError(
                f"installed LT runtime dependency is missing: {name}"
            ) from exc
    return installed


def _installed_runtime_source_revision() -> str:
    package_root = Path(__file__).resolve().parents[1]
    for path in package_root.rglob("*"):
        if path.is_file() and (
            path.suffix.lower() in {".pyc", ".pyo"}
            or "__pycache__" in {part.lower() for part in path.parts}
        ):
            raise ReleaseCliIdentityError(
                "installed LT runtime contains forbidden bytecode"
            )
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
        immutable_entries = process_immutable_directory_entries(directory)
        if immutable_entries is None:
            _assert_no_key_path_links(directory, label=directory_env)
            if not directory.is_absolute() or not directory.is_dir():
                raise ReleaseCliIdentityError(f"{directory_env} is unavailable")
            paths.extend(sorted(directory.glob("*.pem")))
        else:
            paths.extend(immutable_entries)
    for path in paths:
        if process_immutable_file_bytes(path) is None:
            _assert_no_key_path_links(path, label="authority public key")
            if not path.is_file():
                raise ReleaseCliIdentityError(
                    f"authority public key is unavailable: {path}"
                )
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
        "--publication-head-observation",
        type=absolute_path,
        required=True,
        help="Fresh signed external anchor HEAD observation captured for this build",
    )
    parser.add_argument(
        "--publication-head-observation-sha256",
        type=canonical_sha256,
        required=True,
        help="Expected SHA-256 of the signed external anchor HEAD observation",
    )
    parser.add_argument(
        "--publication-head-challenge-nonce",
        type=canonical_sha256,
        required=True,
        help="256-bit caller nonce bound by the signed HEAD observation",
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
