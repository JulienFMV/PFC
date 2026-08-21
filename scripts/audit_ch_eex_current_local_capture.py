"""Audit the selected local CH EEX capture without granting PIT authority.

The audit binds exact capture bytes, the workspace-run receipt, the parser
sources and value commitments for the latest CH row.  Price values are never
printed or written to the registry.  A successful result is local replay
evidence only and is ineligible for the monthly solver, T057 or production.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

from pfc_shaping.data import ingest_forwards
from pfc_shaping.data.eex_forward_local_capture import (
    CAPTURE_SCHEMA,
)
from pfc_shaping.data.eex_forward_local_capture import (
    STATUS as CAPTURE_STATUS,
)
from pfc_shaping.data.eex_forward_vintage_intake import (
    inspect_eex_forward_workbook_latest_market_row,
    preflight_eex_forward_workbook_bytes,
)
from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.pipeline.strict_structured_data import load_strict_json

REGISTRY_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-EEX-CURRENT-LOCAL-CAPTURE-SELECTION-20260730.json"
)
REGISTRY_SHA256 = "5f0b99aa04fabcb8219cfa34f20ea262a705940cbc3db3ab2e114ba99bb4a778"
REGISTRY_ID = "3be51903d7ed2774d464f8bfd49b20fe283fb5d1b2c0bb93f677e99fe4884667"
SCHEMA_VERSION = "ch_eex_current_local_capture_selection.v1"
STATUS = "CURRENT_LOCAL_QUARANTINE_SELECTION_NOT_PIT_AUTHORITY"
AUDIT_SCHEMA_VERSION = "ch_eex_current_local_capture_audit.v1"
AUDIT_STATUS = "CURRENT_EEX_LOCAL_CAPTURE_VALID_NOT_PIT_NO_GO"
QUOTE_COMMITMENT_SCHEMA = "ch_eex_quote_value_commitment.v1"
QUOTE_COMMITMENT_ALGORITHM = "SHA256_CANONICAL_JSON_DECIMAL_STRING_V1"
INTAKE_PARSER_SHA256 = (
    "0cde82c272705d90eb8f23837c96cb2769cf2885fe2549618e759e987d3e8854"
)
PRICE_PARSER_SHA256 = (
    "39ec58287abaf945e4d5537190057cb0592e8a177f8d6f3197d9f8d5fd25e42a"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_BYTES = 2_000_000
_MAX_SOURCE_BYTES = 64 * 1024 * 1024
_CANONICAL_UNC_SOURCE = (
    r"\\fmvfs1\data\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX.xlsx"
)
_REQUIRED_EXTERNAL_AUTHORITIES = [
    "INDEPENDENT_TRUSTED_TIME_RECEIPT",
    "INDEPENDENT_ACQUISITION_SIGNATURE",
    "BUILDER_INACCESSIBLE_IMMUTABLE_COPY",
    "EXTERNAL_CAS_WORM_RECEIPT_AND_FRESH_HEAD",
    "SOURCE_PRODUCT_SESSION_AND_SETTLEMENT_ATTESTATION",
]
_TOP_LEVEL_KEYS = {
    "schema_version",
    "selection_id",
    "status",
    "as_of_utc_untrusted",
    "precedence",
    "current_capture",
    "superseded_capture",
    "parser_binding",
    "external_authority_requirements",
    "quote_values_disclosed",
    "monthly_level_authority",
    "ompex_used",
    "scientific_admission",
    "monthly_solver_hard_level_eligible",
    "training_eligible",
    "model_selection_eligible",
    "t057_consumed",
    "candidate_assembly_eligible",
    "publication_authorized",
    "promotion_gate",
    "production_authorization",
}
_CURRENT_KEYS = {
    "capture_id",
    "capture_directory",
    "capture_manifest_path",
    "capture_manifest_sha256",
    "source_path",
    "source_sha256",
    "source_size_bytes",
    "snapshot_date",
    "quote_count",
    "quote_identity_commitment_sha256",
    "quote_value_commitment_sha256",
    "quote_commitment_algorithm",
    "workspace_receipt_path",
    "workspace_receipt_sha256",
    "current_for_local_diagnostic",
}
_HISTORICAL_KEYS = {
    "capture_id",
    "capture_manifest_path",
    "capture_manifest_sha256",
    "source_path",
    "source_sha256",
    "source_size_bytes",
    "snapshot_date",
    "workspace_receipt_path",
    "workspace_receipt_sha256",
    "may_be_used_as_current_capture",
}
_PARSER_BINDING_KEYS = {
    "intake_parser_path",
    "intake_parser_sha256",
    "intake_parser_size_bytes",
    "price_parser_path",
    "price_parser_sha256",
    "price_parser_size_bytes",
}


class CurrentEexCaptureError(ValueError):
    """Raised when the selected EEX capture chain is ambiguous or rebound."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def compute_selection_id(document: Mapping[str, object]) -> str:
    unsigned = dict(document)
    unsigned.pop("selection_id", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def compute_quote_commitments(
    source_bytes: bytes,
    *,
    market: str = "CH",
) -> dict[str, object]:
    """Return non-disclosing commitments for the latest physical market row."""

    preflight_eex_forward_workbook_bytes(source_bytes)
    inspected_date, inspected_quotes = inspect_eex_forward_workbook_latest_market_row(
        source_bytes,
        market=market,
    )
    try:
        prices, snapshot_date, metadata = (
            ingest_forwards.load_base_prices_from_eex_report_bytes(
                source_bytes,
                market=market,
                source_label="<captured CH EEX bytes>",
                return_snapshot_metadata=True,
            )
        )
    except Exception as exc:
        raise CurrentEexCaptureError("captured EEX price replay failed") from exc
    snapshot_iso = pd.Timestamp(snapshot_date).strftime("%Y-%m-%d")
    if snapshot_iso != inspected_date:
        raise CurrentEexCaptureError("EEX inventory and price snapshot dates differ")
    if not isinstance(prices, Mapping) or not prices:
        raise CurrentEexCaptureError("captured EEX price inventory is empty")
    lineage = metadata.get("quote_lineage") if isinstance(metadata, Mapping) else None
    if not isinstance(lineage, Mapping) or set(lineage) != set(prices):
        raise CurrentEexCaptureError("captured EEX price lineage is not exact")

    identity_rows: list[dict[str, str]] = []
    value_rows: list[dict[str, str]] = []
    for price_key in sorted(prices):
        raw_lineage = lineage[price_key]
        if not isinstance(raw_lineage, Mapping):
            raise CurrentEexCaptureError("captured EEX quote lineage is invalid")
        code = str(raw_lineage.get("source_product_code", "")).strip().upper()
        normalized = ingest_forwards.normalize_eex_product_code(code)
        if normalized is None:
            raise CurrentEexCaptureError("captured EEX product identity is unsupported")
        product, load_type, product_type = normalized
        if product_type == "Week" or product_type.upper() not in {
            "CAL",
            "QUARTER",
            "MONTH",
        }:
            raise CurrentEexCaptureError("captured EEX product is not LT admissible")
        expected_key = product if load_type == "BASE" else f"{product}-Peak"
        if price_key != expected_key:
            raise CurrentEexCaptureError("captured EEX price key is inconsistent")
        identity = {
            "market": market,
            "source_product_code": code,
            "product": product,
            "load_type": load_type,
            "product_type": product_type.upper(),
        }
        identity_rows.append(identity)
        value_rows.append(
            {
                **identity,
                "price_eur_mwh": _canonical_decimal(prices[price_key]),
            }
        )

    identity_rows.sort(key=_quote_sort_key)
    value_rows.sort(key=_quote_sort_key)
    if identity_rows != inspected_quotes:
        raise CurrentEexCaptureError("captured EEX identity and value inventories differ")
    identity_payload = {
        "schema_version": "ch_eex_quote_identity_commitment.v1",
        "market": market,
        "snapshot_date": snapshot_iso,
        "quotes": identity_rows,
    }
    value_payload = {
        "schema_version": QUOTE_COMMITMENT_SCHEMA,
        "algorithm": QUOTE_COMMITMENT_ALGORITHM,
        "market": market,
        "snapshot_date": snapshot_iso,
        "unit": "EUR/MWH",
        "quotes": value_rows,
    }
    return {
        "snapshot_date": snapshot_iso,
        "quote_count": len(value_rows),
        "quote_identity_commitment_sha256": hashlib.sha256(
            canonical_json_bytes(identity_payload)
        ).hexdigest(),
        "quote_value_commitment_sha256": hashlib.sha256(
            canonical_json_bytes(value_payload)
        ).hexdigest(),
        "quote_commitment_algorithm": QUOTE_COMMITMENT_ALGORITHM,
        "quote_values_disclosed": False,
    }


def audit_current_eex_capture(
    *,
    registry_path: Path,
    expected_registry_sha256: str,
    repository_root: Path | None = None,
) -> dict[str, object]:
    root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else repository_root.expanduser()
    )
    if not root.is_absolute():
        raise CurrentEexCaptureError("repository root must be absolute")
    if _path_key(Path.cwd()) != _path_key(root):
        raise CurrentEexCaptureError("cwd is not the canonical repository root")
    if not (root / ".git").exists() or not (root / "AGENTS.md").is_file():
        raise CurrentEexCaptureError("canonical repository markers are absent")
    canonical_registry = root.joinpath(*REGISTRY_RELATIVE_PATH.split("/"))
    if _path_key(registry_path) != _path_key(canonical_registry):
        raise CurrentEexCaptureError("registry path is not canonical")
    _require_sha256(expected_registry_sha256, label="expected registry")
    if expected_registry_sha256 != REGISTRY_SHA256:
        raise CurrentEexCaptureError(
            "caller-held registry SHA-256 does not match frozen identity"
        )
    registry_payload = _read_exact(
        canonical_registry,
        REGISTRY_SHA256,
        label="EEX current-capture registry",
        max_bytes=_MAX_JSON_BYTES,
    )
    registry = _strict_mapping(registry_payload, label="EEX current-capture registry")
    _exact_keys(registry, _TOP_LEVEL_KEYS, label="EEX current-capture registry")
    _equal(registry.get("schema_version"), SCHEMA_VERSION, "registry schema")
    _equal(registry.get("status"), STATUS, "registry status")
    selection_id = _sha256(registry.get("selection_id"), label="selection ID")
    if selection_id != REGISTRY_ID or selection_id != compute_selection_id(registry):
        raise CurrentEexCaptureError("selection ID does not bind exact content")
    _equal(
        registry.get("precedence"),
        {
            "current_local_capture": "EEX_CH_20260730_V1_ONLY",
            "superseded_local_capture": "EEX_CH_20260729_V2_HISTORICAL_ONLY",
            "scientific_rule": (
                "LOCAL_CAPTURE_SELECTION_NEVER_CREATES_TRUSTED_PIT_OR_SOLVER_AUTHORITY"
            ),
        },
        "selection precedence",
    )

    current = _mapping(registry.get("current_capture"), label="current capture")
    historical = _mapping(
        registry.get("superseded_capture"), label="superseded capture"
    )
    parser = _mapping(registry.get("parser_binding"), label="parser binding")
    _exact_keys(current, _CURRENT_KEYS, label="current capture")
    _exact_keys(historical, _HISTORICAL_KEYS, label="superseded capture")
    _exact_keys(parser, _PARSER_BINDING_KEYS, label="parser binding")

    current_manifest, current_source = _validate_capture_reference(
        root,
        current,
        current=True,
    )
    historical_manifest, _ = _validate_capture_reference(
        root,
        historical,
        current=False,
    )
    commitments = compute_quote_commitments(current_source, market="CH")
    for field in (
        "snapshot_date",
        "quote_count",
        "quote_identity_commitment_sha256",
        "quote_value_commitment_sha256",
        "quote_commitment_algorithm",
    ):
        _equal(commitments[field], current.get(field), f"current {field}")
    market_inventory = _mapping(
        _mapping(current_manifest.get("market_inventory"), label="market inventory").get(
            "CH"
        ),
        label="CH market inventory",
    )
    _equal(market_inventory.get("latest_snapshot_date"), commitments["snapshot_date"], "manifest snapshot date")
    _equal(market_inventory.get("quote_count"), commitments["quote_count"], "manifest quote count")

    _validate_parser_binding(root, parser)
    _validate_runner_receipt(
        root,
        current,
        source_sha256=str(current["source_sha256"]),
        label="current workspace receipt",
    )
    _validate_runner_receipt(
        root,
        historical,
        source_sha256=str(historical["source_sha256"]),
        label="historical workspace receipt",
    )
    if str(current_manifest.get("capture_id")) == str(
        historical_manifest.get("capture_id")
    ):
        raise CurrentEexCaptureError("current and historical capture IDs collide")
    if current.get("source_sha256") == historical.get("source_sha256"):
        raise CurrentEexCaptureError("current capture does not advance source bytes")
    if pd.Timestamp(str(current.get("snapshot_date"))) <= pd.Timestamp(
        str(historical.get("snapshot_date"))
    ):
        raise CurrentEexCaptureError("current EEX snapshot date is not monotone")
    _equal(current.get("current_for_local_diagnostic"), True, "current local use")
    _equal(
        historical.get("may_be_used_as_current_capture"),
        False,
        "historical current eligibility",
    )
    _equal(
        registry.get("external_authority_requirements"),
        _REQUIRED_EXTERNAL_AUTHORITIES,
        "external authority requirements",
    )
    _equal(registry.get("quote_values_disclosed"), False, "quote disclosure")
    _equal(registry.get("monthly_level_authority"), "solver", "monthly authority")
    _equal(registry.get("ompex_used"), False, "OMPEX use")
    for field in (
        "scientific_admission",
        "monthly_solver_hard_level_eligible",
        "training_eligible",
        "model_selection_eligible",
        "t057_consumed",
        "candidate_assembly_eligible",
        "publication_authorized",
        "promotion_gate",
        "production_authorization",
    ):
        _equal(registry.get(field), False, field)

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": AUDIT_STATUS,
        "selection_id": selection_id,
        "repository_root": str(root),
        "registry_path": str(canonical_registry),
        "registry_sha256": REGISTRY_SHA256,
        "current_capture_id": current.get("capture_id"),
        "current_source_sha256": current.get("source_sha256"),
        "current_snapshot_date": commitments["snapshot_date"],
        "quote_count": commitments["quote_count"],
        "quote_identity_commitment_sha256": commitments[
            "quote_identity_commitment_sha256"
        ],
        "quote_value_commitment_sha256": commitments[
            "quote_value_commitment_sha256"
        ],
        "quote_values_disclosed": False,
        "superseded_capture_id": historical.get("capture_id"),
        "scientific_admission": False,
        "monthly_solver_hard_level_eligible": False,
        "t057_consumed": False,
        "publication_authorized": False,
        "promotion_gate": False,
        "production_authorization": False,
    }


def _validate_capture_reference(
    root: Path,
    reference: Mapping[str, object],
    *,
    current: bool,
) -> tuple[Mapping[str, object], bytes]:
    manifest_path = _resolve_repo_path(
        root,
        reference.get("capture_manifest_path"),
        label="capture manifest path",
    )
    source_path = _resolve_repo_path(
        root,
        reference.get("source_path"),
        label="capture source path",
    )
    receipt_path = _resolve_repo_path(
        root,
        reference.get("workspace_receipt_path"),
        label="workspace receipt path",
    )
    if current:
        capture_directory = _resolve_repo_path(
            root,
            reference.get("capture_directory"),
            label="capture directory",
        )
        if _path_key(manifest_path.parent) != _path_key(capture_directory):
            raise CurrentEexCaptureError("current capture directory is inconsistent")
        inventory = sorted(path.name for path in capture_directory.iterdir())
        if inventory != ["capture-attempt.json", "capture-manifest.json", "source.xlsx"]:
            raise CurrentEexCaptureError("current capture directory inventory is not exact")
    manifest_payload = _read_exact(
        manifest_path,
        _sha256(reference.get("capture_manifest_sha256"), label="capture manifest"),
        label="capture manifest",
        max_bytes=_MAX_JSON_BYTES,
    )
    manifest = _strict_mapping(manifest_payload, label="capture manifest")
    _equal(manifest.get("schema_version"), CAPTURE_SCHEMA, "capture schema")
    _equal(manifest.get("status"), CAPTURE_STATUS, "capture status")
    _equal(manifest.get("capture_id"), reference.get("capture_id"), "capture ID")
    _equal(
        manifest.get("source_document_sha256"),
        reference.get("source_sha256"),
        "capture source hash",
    )
    _equal(
        manifest.get("source_document_size_bytes"),
        reference.get("source_size_bytes"),
        "capture source size",
    )
    captured_source = _mapping(
        manifest.get("captured_source"), label="captured source"
    )
    _equal(captured_source.get("path"), source_path.name, "captured source name")
    _equal(captured_source.get("sha256"), reference.get("source_sha256"), "captured source SHA-256")
    _equal(captured_source.get("size_bytes"), reference.get("source_size_bytes"), "captured source bytes")
    _equal(manifest.get("quote_values_disclosed_in_manifest"), False, "manifest quote disclosure")
    _equal(manifest.get("next_required_authorities"), _REQUIRED_EXTERNAL_AUTHORITIES, "manifest authorities")
    for field in (
        "trusted_time_receipt_present",
        "independent_acquisition_signature_present",
        "builder_inaccessible_immutable_copy_present",
        "external_cas_admitted",
        "source_product_semantics_attested",
        "scientific_admission",
        "monthly_solver_hard_level_eligible",
        "training_eligible",
        "model_selection_eligible",
        "holdout_eligible",
        "candidate_assembly_eligible",
        "production_authorization",
        "promotion_gate",
    ):
        _equal(manifest.get(field), False, f"manifest {field}")
    if not source_path.is_file() or not receipt_path.is_file():
        raise CurrentEexCaptureError("capture source or receipt is unavailable")
    source_payload = _read_exact(
        source_path,
        _sha256(reference.get("source_sha256"), label="captured source"),
        label="captured workbook",
        max_bytes=_MAX_SOURCE_BYTES,
    )
    _equal(len(source_payload), reference.get("source_size_bytes"), "captured source size")
    snapshot_date, quote_inventory = inspect_eex_forward_workbook_latest_market_row(
        source_payload,
        market="CH",
    )
    market_inventory = _mapping(
        _mapping(manifest.get("market_inventory"), label="market inventory").get(
            "CH"
        ),
        label="CH market inventory",
    )
    _exact_keys(
        market_inventory,
        {"latest_snapshot_date", "quote_count", "quotes"},
        label="CH market inventory",
    )
    _equal(snapshot_date, reference.get("snapshot_date"), "capture snapshot date")
    _equal(
        market_inventory.get("latest_snapshot_date"),
        snapshot_date,
        "manifest snapshot date",
    )
    _equal(
        market_inventory.get("quote_count"),
        len(quote_inventory),
        "manifest quote count",
    )
    _equal(market_inventory.get("quotes"), quote_inventory, "manifest quote inventory")
    return manifest, source_payload


def _validate_parser_binding(root: Path, parser: Mapping[str, object]) -> None:
    bindings = (
        (
            "intake_parser",
            "pfc_shaping/data/eex_forward_vintage_intake.py",
            INTAKE_PARSER_SHA256,
        ),
        (
            "price_parser",
            "pfc_shaping/data/ingest_forwards.py",
            PRICE_PARSER_SHA256,
        ),
    )
    for prefix, expected_relative, frozen_sha256 in bindings:
        _equal(parser.get(f"{prefix}_path"), expected_relative, f"{prefix} path")
        _equal(
            parser.get(f"{prefix}_sha256"),
            frozen_sha256,
            f"{prefix} frozen SHA-256",
        )
        payload = _read_repo_file(
            root,
            parser.get(f"{prefix}_path"),
            expected_sha256=_sha256(
                parser.get(f"{prefix}_sha256"), label=f"{prefix}"
            ),
            label=prefix,
            max_bytes=2 * 1024 * 1024,
        )
        _equal(len(payload), parser.get(f"{prefix}_size_bytes"), f"{prefix} size")


def _validate_runner_receipt(
    root: Path,
    reference: Mapping[str, object],
    *,
    source_sha256: str,
    label: str,
) -> None:
    receipt_payload = _read_repo_file(
        root,
        reference.get("workspace_receipt_path"),
        expected_sha256=_sha256(
            reference.get("workspace_receipt_sha256"), label=label
        ),
        label=label,
        max_bytes=_MAX_JSON_BYTES,
    )
    receipt = _strict_mapping(receipt_payload, label=label)
    _equal(receipt.get("schema_version"), "pfc_lt_workspace_local_execution.v4", f"{label} schema")
    _equal(receipt.get("status"), "TARGET_EXIT_ZERO_NOT_AUTHORITY", f"{label} status")
    _equal(receipt.get("target_exit_code"), 0, f"{label} target exit")
    _equal(receipt.get("runner_exit_code"), 0, f"{label} runner exit")
    _equal(receipt.get("workspace"), str(root), f"{label} workspace")
    for field in (
        "runtime_authority",
        "scientific_authorization",
        "evaluation_authorization",
        "promotion_authorization",
        "production_authorization",
    ):
        _equal(receipt.get(field), False, f"{label} {field}")
    command = receipt.get("redacted_command")
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
        raise CurrentEexCaptureError(f"{label} command is invalid")
    command_values = [str(value) for value in command]
    expected_tail = [
        "-B",
        "-m",
        "scripts.capture_eex_forward_local",
        "--source-document",
        _CANONICAL_UNC_SOURCE,
        "--expected-source-sha256",
        source_sha256,
        "--capture-id",
        str(reference.get("capture_id")),
    ]
    if len(command_values) != len(expected_tail) + 1 or command_values[1:] != expected_tail:
        raise CurrentEexCaptureError(f"{label} command does not bind the capture")
    interpreter = Path(command_values[0])
    if not interpreter.is_absolute() or interpreter.name.lower() != "python.exe":
        raise CurrentEexCaptureError(f"{label} interpreter is invalid")
    mutable_paths = _mapping(receipt.get("mutable_paths"), label=f"{label} mutable paths")
    build_root = root / "build"
    for value in mutable_paths.values():
        if not isinstance(value, str) or not _is_within(Path(value), build_root):
            raise CurrentEexCaptureError(f"{label} mutable path escapes repo build")
    run_root = receipt.get("run_root")
    if not isinstance(run_root, str) or not _is_within(Path(run_root), build_root):
        raise CurrentEexCaptureError(f"{label} run root escapes repo build")
    expected_run_root = _resolve_repo_path(
        root,
        str(reference.get("workspace_receipt_path")).rsplit("/", 1)[0],
        label=f"{label} run root",
    )
    if _path_key(Path(run_root)) != _path_key(expected_run_root):
        raise CurrentEexCaptureError(f"{label} run root is inconsistent")
    _validate_execution_identity(
        receipt.get("execution_identity"),
        root=root,
        interpreter=interpreter,
        label=label,
    )
    _validate_target_output(
        receipt.get("target_output"),
        run_root=expected_run_root,
        label=label,
    )


def _validate_execution_identity(
    raw_identity: object,
    *,
    root: Path,
    interpreter: Path,
    label: str,
) -> None:
    identity = _mapping(raw_identity, label=f"{label} execution identity")
    _exact_keys(
        identity,
        {"git_branch", "git_head", "runner_source", "source_tree", "target_interpreter"},
        label=f"{label} execution identity",
    )
    if identity.get("git_branch") != "fix/lt-audit-remediation":
        raise CurrentEexCaptureError(f"{label} git branch is invalid")
    if re.fullmatch(r"[0-9a-f]{40}", str(identity.get("git_head", ""))) is None:
        raise CurrentEexCaptureError(f"{label} git head is invalid")
    runner = _mapping(identity.get("runner_source"), label=f"{label} runner source")
    target = _mapping(
        identity.get("target_interpreter"), label=f"{label} target interpreter"
    )
    metadata_keys = {"device", "inode", "link_count", "path", "sha256", "size_bytes"}
    _exact_keys(runner, metadata_keys, label=f"{label} runner source")
    _exact_keys(target, metadata_keys, label=f"{label} target interpreter")
    expected_runner = root / "scripts" / "run_workspace_local.py"
    if _path_key(Path(str(runner.get("path", "")))) != _path_key(expected_runner):
        raise CurrentEexCaptureError(f"{label} runner source path is invalid")
    if _path_key(Path(str(target.get("path", "")))) != _path_key(interpreter):
        raise CurrentEexCaptureError(f"{label} target interpreter path is invalid")
    for name, record in (("runner", runner), ("target interpreter", target)):
        _sha256(record.get("sha256"), label=f"{label} {name}")
        for field in ("device", "inode", "link_count", "size_bytes"):
            value = record.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise CurrentEexCaptureError(
                    f"{label} {name} {field} is invalid"
                )
    source_tree = _mapping(identity.get("source_tree"), label=f"{label} source tree")
    _exact_keys(
        source_tree,
        {"schema_version", "file_count", "total_bytes", "tree_sha256"},
        label=f"{label} source tree",
    )
    _equal(
        source_tree.get("schema_version"),
        "pfc_lt_workspace_source_tree.v1",
        f"{label} source tree schema",
    )
    _sha256(source_tree.get("tree_sha256"), label=f"{label} source tree")
    for field in ("file_count", "total_bytes"):
        value = source_tree.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise CurrentEexCaptureError(f"{label} source tree {field} is invalid")


def _validate_target_output(
    raw_output: object,
    *,
    run_root: Path,
    label: str,
) -> None:
    output = _mapping(raw_output, label=f"{label} target output")
    _exact_keys(
        output,
        {"capture_limit_bytes_per_stream", "complete", "streams"},
        label=f"{label} target output",
    )
    _equal(output.get("complete"), True, f"{label} output completeness")
    limit = output.get("capture_limit_bytes_per_stream")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise CurrentEexCaptureError(f"{label} output limit is invalid")
    streams = _mapping(output.get("streams"), label=f"{label} streams")
    _exact_keys(streams, {"stdout", "stderr"}, label=f"{label} streams")
    for stream_name in ("stdout", "stderr"):
        stream = _mapping(
            streams.get(stream_name), label=f"{label} {stream_name} stream"
        )
        _exact_keys(
            stream,
            {
                "captured_bytes",
                "full_stream_sha256",
                "path",
                "sha256",
                "total_bytes",
                "truncated",
            },
            label=f"{label} {stream_name} stream",
        )
        _equal(stream.get("truncated"), False, f"{label} {stream_name} truncation")
        captured = stream.get("captured_bytes")
        total = stream.get("total_bytes")
        if (
            not isinstance(captured, int)
            or isinstance(captured, bool)
            or captured < 0
            or captured != total
            or captured > limit
        ):
            raise CurrentEexCaptureError(f"{label} {stream_name} size is invalid")
        sha256 = _sha256(stream.get("sha256"), label=f"{label} {stream_name}")
        _equal(
            stream.get("full_stream_sha256"),
            sha256,
            f"{label} {stream_name} full hash",
        )
        expected_path = run_root / f"target-{stream_name}.log"
        if _path_key(Path(str(stream.get("path", "")))) != _path_key(expected_path):
            raise CurrentEexCaptureError(f"{label} {stream_name} path is invalid")
        payload = _read_exact(
            expected_path,
            sha256,
            label=f"{label} {stream_name} log",
            max_bytes=limit,
        )
        _equal(len(payload), captured, f"{label} {stream_name} captured bytes")
        if stream_name == "stderr" and payload:
            raise CurrentEexCaptureError(f"{label} stderr is not empty")


def _canonical_decimal(value: object) -> str:
    try:
        numeric = float(value)
        decimal = Decimal(str(numeric))
    except (InvalidOperation, TypeError, ValueError, OverflowError) as exc:
        raise CurrentEexCaptureError("captured EEX price is not numeric") from exc
    if not math.isfinite(numeric) or not decimal.is_finite() or numeric == 0.0:
        raise CurrentEexCaptureError("captured EEX price is not admissible")
    rendered = format(decimal.normalize(), "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    if rendered in {"", "-0", "+0", "0"}:
        raise CurrentEexCaptureError("captured EEX price is ambiguous")
    return rendered


def _quote_sort_key(value: Mapping[str, str]) -> tuple[str, ...]:
    return (
        value["market"],
        value["product_type"],
        value["product"],
        value["load_type"],
        value["source_product_code"],
    )


def _read_repo_file(
    root: Path,
    raw_path: object,
    *,
    expected_sha256: str,
    label: str,
    max_bytes: int,
) -> bytes:
    return _read_exact(
        _resolve_repo_path(root, raw_path, label=f"{label} path"),
        expected_sha256,
        label=label,
        max_bytes=max_bytes,
    )


def _resolve_repo_path(root: Path, raw_path: object, *, label: str) -> Path:
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or "\\" in raw_path
        or Path(raw_path).is_absolute()
        or any(part in {"", ".", ".."} for part in raw_path.split("/"))
    ):
        raise CurrentEexCaptureError(f"{label} is invalid")
    candidate = root.joinpath(*raw_path.split("/"))
    if os.path.commonpath((str(root), str(candidate))) != str(root):
        raise CurrentEexCaptureError(f"{label} escapes the repository")
    return candidate


def _read_exact(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    payload = read_stable_single_link_file(path, label=label, max_bytes=max_bytes)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise CurrentEexCaptureError(f"{label} SHA-256 mismatch")
    return payload


def _strict_mapping(payload: bytes, *, label: str) -> Mapping[str, object]:
    try:
        value = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CurrentEexCaptureError(f"{label} is invalid JSON") from exc
    return _mapping(value, label=label)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise CurrentEexCaptureError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise CurrentEexCaptureError(f"{label} keys are not exact")


def _equal(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise CurrentEexCaptureError(f"{label} is invalid")


def _sha256(value: object, *, label: str) -> str:
    _require_sha256(value, label=label)
    assert isinstance(value, str)
    return value


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CurrentEexCaptureError(f"{label} SHA-256 is invalid")


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _is_within(path: Path, root: Path) -> bool:
    try:
        Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(root)))
    except ValueError:
        return False
    return True


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        result = audit_current_eex_capture(
            registry_path=args.registry,
            expected_registry_sha256=args.expected_registry_sha256,
        )
    except (OSError, CurrentEexCaptureError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": AUDIT_SCHEMA_VERSION,
                    "status": "CURRENT_EEX_LOCAL_CAPTURE_INVALID_NO_GO",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "quote_values_disclosed": False,
                    "scientific_admission": False,
                    "monthly_solver_hard_level_eligible": False,
                    "t057_consumed": False,
                    "promotion_gate": False,
                    "production_authorization": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
