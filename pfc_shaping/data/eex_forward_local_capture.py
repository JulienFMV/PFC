"""One-shot local capture of an external EEX forward workbook.

This module preserves exact bytes for a later independent trusted-time and
external-CAS handoff.  A successful local capture is deliberately not PIT,
scientific, solver, candidate, promotion, or production authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from pfc_shaping.data.eex_forward_vintage_intake import (
    EexForwardVintageIntakeError,
    inspect_eex_forward_workbook_latest_market_row,
    preflight_eex_forward_workbook_bytes,
)
from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)

CAPTURE_SCHEMA = "eex_forward_local_capture.v1"
ATTEMPT_SCHEMA = "eex_forward_local_capture_attempt.v1"
STATUS = "LOCAL_UNTRUSTED_CAPTURE_NOT_PIT_AUTHORITY"
ATTEMPT_STATUS = "ATTEMPTED_LOCAL_UNTRUSTED_CAPTURE"
_PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_SOURCE_BYTES = 64 * 1024 * 1024
_EXPECTED_SOURCE_NAME = "Price_Report_EEX.xlsx"
_EXPECTED_SOURCE_PATH = Path(
    r"\\fmvfs1\data\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX.xlsx"
)
_NEXT_AUTHORITIES = [
    "INDEPENDENT_TRUSTED_TIME_RECEIPT",
    "INDEPENDENT_ACQUISITION_SIGNATURE",
    "BUILDER_INACCESSIBLE_IMMUTABLE_COPY",
    "EXTERNAL_CAS_WORM_RECEIPT_AND_FRESH_HEAD",
    "SOURCE_PRODUCT_SESSION_AND_SETTLEMENT_ATTESTATION",
]


class EexForwardLocalCaptureError(ValueError):
    """Raised when exact local capture cannot be completed safely."""


def canonical_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def capture_eex_forward_workbook(
    *,
    source_document: str | Path,
    expected_source_sha256: str,
    capture_id: str,
    repo_root: str | Path | None = None,
    clock: Callable[[], datetime] | None = None,
) -> Path:
    """Capture exact workbook bytes and return the immutable manifest path."""

    if _PORTABLE_ID.fullmatch(capture_id) is None:
        raise EexForwardLocalCaptureError("capture_id must be a portable identifier")
    if _SHA256.fullmatch(expected_source_sha256) is None:
        raise EexForwardLocalCaptureError("expected source SHA-256 is invalid")

    root = assert_absolute_path_has_no_links(
        Path(repo_root) if repo_root is not None else canonical_repo_root()
    )
    if not root.is_dir():
        raise EexForwardLocalCaptureError("canonical repository root is unavailable")
    build_root = assert_absolute_path_has_no_links(root / "build")
    if not build_root.is_dir():
        raise EexForwardLocalCaptureError("repository build root is unavailable")

    requested_source = Path(source_document)
    if (
        not requested_source.is_absolute()
        or _lexical(requested_source) != _lexical(_EXPECTED_SOURCE_PATH)
    ):
        raise EexForwardLocalCaptureError(
            "source document must be the canonical read-only FMV EEX workbook"
        )
    source = assert_absolute_path_has_no_links(requested_source)
    if source.name != _EXPECTED_SOURCE_NAME:
        raise EexForwardLocalCaptureError("source workbook name is not canonical")
    if _is_within(source, root):
        raise EexForwardLocalCaptureError("source workbook must be external to the workspace")

    capture_root = build_root / "eex-forward-local-captures"
    capture_root.mkdir(exist_ok=True)
    capture_root = assert_absolute_path_has_no_links(capture_root)
    output = capture_root / capture_id
    output.mkdir(exist_ok=False)

    now = clock or (lambda: datetime.now(timezone.utc))
    started = _utc_now(now, label="capture start")
    attempt = {
        "schema_version": ATTEMPT_SCHEMA,
        "status": ATTEMPT_STATUS,
        "capture_id": capture_id,
        "source_document_name": source.name,
        "expected_source_sha256": expected_source_sha256,
        "capture_started_utc": _iso(started),
        "workstation_clock_trusted": False,
        "scientific_admission": False,
        "monthly_solver_hard_level_eligible": False,
        "production_authorization": False,
        "promotion_gate": False,
    }
    _write_json(output / "capture-attempt.json", attempt, label="EEX capture attempt")

    try:
        metadata_before = source.stat(follow_symlinks=False)
        payload = read_stable_single_link_file(
            source,
            label="external EEX workbook",
            max_bytes=_MAX_SOURCE_BYTES,
        )
        observed_hash = hashlib.sha256(payload).hexdigest()
        if observed_hash != expected_source_sha256:
            raise EexForwardLocalCaptureError("external EEX workbook SHA-256 mismatch")
        try:
            preflight_eex_forward_workbook_bytes(payload)
            snapshot_date, quotes = inspect_eex_forward_workbook_latest_market_row(
                payload,
                market="CH",
            )
        except EexForwardVintageIntakeError as exc:
            raise EexForwardLocalCaptureError(str(exc)) from exc
        metadata = source.stat(follow_symlinks=False)
        if _file_identity(metadata_before) != _file_identity(metadata):
            raise EexForwardLocalCaptureError(
                "external EEX workbook metadata changed during capture"
            )

        captured_source = output / "source.xlsx"
        write_durable_exact_bytes(
            captured_source,
            payload,
            label="captured EEX workbook",
        )
        copied = read_stable_single_link_file(
            captured_source,
            label="captured EEX workbook",
            max_bytes=_MAX_SOURCE_BYTES,
        )
        if copied != payload:
            raise EexForwardLocalCaptureError("captured EEX workbook bytes differ")

        completed = _utc_now(now, label="capture completion")
        manifest = {
            "schema_version": CAPTURE_SCHEMA,
            "status": STATUS,
            "capture_id": capture_id,
            "source_role": "eex_forward_source",
            "source_system": "FMV_EEX_DESK_WORKBOOK_SHARE",
            "source_document_name": source.name,
            "source_path_kind": "UNC_NETWORK_SHARE"
            if os.fspath(source).startswith("\\\\")
            else "EXTERNAL_LOCAL_PATH",
            "source_document_sha256": observed_hash,
            "source_document_size_bytes": len(payload),
            "source_filesystem_last_write_utc": _iso(
                datetime.fromtimestamp(metadata.st_mtime, tz=timezone.utc)
            ),
            "source_filesystem_time_trusted": False,
            "capture_started_utc": _iso(started),
            "capture_completed_utc": _iso(completed),
            "workstation_clock_trusted": False,
            "captured_source": {
                "path": "source.xlsx",
                "sha256": observed_hash,
                "size_bytes": len(payload),
            },
            "market_inventory": {
                "CH": {
                    "latest_snapshot_date": snapshot_date,
                    "quote_count": len(quotes),
                    "quotes": quotes,
                }
            },
            "quote_values_disclosed_in_manifest": False,
            "trusted_time_receipt_present": False,
            "independent_acquisition_signature_present": False,
            "builder_inaccessible_immutable_copy_present": False,
            "external_cas_admitted": False,
            "source_product_semantics_attested": False,
            "scientific_admission": False,
            "monthly_solver_hard_level_eligible": False,
            "training_eligible": False,
            "model_selection_eligible": False,
            "holdout_eligible": False,
            "candidate_assembly_eligible": False,
            "production_authorization": False,
            "promotion_gate": False,
            "next_required_authorities": list(_NEXT_AUTHORITIES),
        }
        manifest_path = output / "capture-manifest.json"
        _write_json(manifest_path, manifest, label="EEX capture manifest")
        return manifest_path
    except Exception:
        # The exclusive one-shot directory and attempt record are negative
        # evidence. Never repair, delete, or retry this capture_id in place.
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-document", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--capture-id", required=True)
    args = parser.parse_args(argv)
    try:
        manifest = capture_eex_forward_workbook(
            source_document=args.source_document,
            expected_source_sha256=args.expected_source_sha256,
            capture_id=args.capture_id,
        )
    except (EexForwardLocalCaptureError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "LOCAL_CAPTURE_FAILED_NOT_AUTHORITY",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "production_authorization": False,
                    "promotion_gate": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(str(manifest))
    return 0


def _is_within(path: Path, root: Path) -> bool:
    try:
        Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(root)))
    except ValueError:
        return False
    return True


def _lexical(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
    )


def _utc_now(clock: Callable[[], datetime], *, label: str) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EexForwardLocalCaptureError(f"{label} clock is not timezone-aware")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object, *, label: str) -> None:
    write_durable_exact_bytes(path, _canonical_json(value), label=label)


__all__ = [
    "ATTEMPT_SCHEMA",
    "CAPTURE_SCHEMA",
    "EexForwardLocalCaptureError",
    "STATUS",
    "capture_eex_forward_workbook",
    "main",
]
