"""Audit the canonical cross-domain CH LT evidence selection registry.

This checkout audit resolves the current prospective ledger separately from
the frozen ledger snapshot embedded in the market-time contract.  It grants no
scientific, publication, promotion, or production authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path

from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.pipeline.strict_structured_data import load_strict_json
from pfc_shaping.validation.ch_lt_prospective_capture_ledger import (
    ChLtProspectiveCaptureLedgerError,
    build_prospective_capture_ledger,
)

REGISTRY_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-CURRENT-EVIDENCE-SELECTION-V3-20260731.json"
)
REGISTRY_SHA256 = "6518f7e876ce1c233fc055d3d20ad213088d361d784afbe5aa16d4f165e744f7"
REGISTRY_ID = "748aba2e2f85711d0a5dcdb07e0acacbf8dbce7a76ab4a4b07ef48371ec25488"
SCHEMA_VERSION = "ch_lt_current_evidence_selection.v3"
STATUS = "CURRENT_LOCAL_DIAGNOSTIC_SELECTION_ONLY_NOT_AUTHORITY"
AUDIT_STATUS = "CURRENT_SELECTION_LINKS_VALID_LOCAL_ONLY_NO_GO"
_MAX_BYTES = 2_000_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_INVALID_PATH_CHARS_RE = re.compile(r'[<>:"|?*\x00-\x1f\x7f]')
_WINDOWS_RESERVED_BASENAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}
_TOP_LEVEL_KEYS = {
    "schema_version",
    "registry_id",
    "status",
    "as_of_utc_untrusted",
    "precedence",
    "current_prospective_capture_selection",
    "superseded_prospective_capture_selection",
    "market_time_embedded_historical_selection",
    "current_market_time_regime",
    "operator_resolution",
    "independent_rolling_origin_count",
    "future_holdout_consumed",
    "t057_consumed",
    "scientific_admission",
    "candidate_assembly_eligible",
    "publication_authorized",
    "promotion_gate",
    "production_authorization",
}


class CurrentEvidenceSelectionError(ValueError):
    """Raised when the current-selection chain is ambiguous or rebound."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def compute_registry_id(document: Mapping[str, object]) -> str:
    unsigned = dict(document)
    unsigned.pop("registry_id", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def audit_current_selection(
    *, registry_path: Path, expected_registry_sha256: str
) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    if _path_key(Path.cwd()) != _path_key(root):
        raise CurrentEvidenceSelectionError("cwd is not the canonical repository root")
    if not (root / ".git").exists() or not (root / "AGENTS.md").is_file():
        raise CurrentEvidenceSelectionError("canonical repository markers are absent")
    canonical_registry = root.joinpath(*REGISTRY_RELATIVE_PATH.split("/"))
    if _path_key(registry_path) != _path_key(canonical_registry):
        raise CurrentEvidenceSelectionError("registry path is not canonical")
    _require_sha256(expected_registry_sha256, label="expected registry")
    if expected_registry_sha256 != REGISTRY_SHA256:
        raise CurrentEvidenceSelectionError(
            "caller-held registry SHA-256 does not match frozen identity"
        )
    payload = _read_exact(canonical_registry, REGISTRY_SHA256, label="registry")
    document = _strict_mapping(payload, label="registry")
    _exact_keys(document, _TOP_LEVEL_KEYS, label="registry")
    _equal(document.get("schema_version"), SCHEMA_VERSION, "schema version")
    _equal(document.get("status"), STATUS, "registry status")
    registry_id = _sha256(document.get("registry_id"), label="registry ID")
    if registry_id != REGISTRY_ID or registry_id != compute_registry_id(document):
        raise CurrentEvidenceSelectionError("registry ID does not bind exact content")

    precedence = _mapping(document.get("precedence"), label="precedence")
    _equal(
        precedence,
        {
            "prospective_capture_ledger": "CURRENT_PROSPECTIVE_SELECTION_V4_ONLY",
            "superseded_prospective_capture_ledger": (
                "SELECTION_V3_HISTORICAL_ONLY"
            ),
            "market_time_regime": "MARKET_TIME_CONTRACT_V1_ONLY",
            "cross_domain_rule": (
                "MARKET_TIME_EMBEDDED_LEDGER_IS_A_FROZEN_AUDIT_SNAPSHOT_NOT_THE_"
                "CURRENT_PROSPECTIVE_LEDGER_SELECTION"
            ),
        },
        "precedence",
    )
    current = _mapping(
        document.get("current_prospective_capture_selection"),
        label="current prospective selection",
    )
    historical = _mapping(
        document.get("superseded_prospective_capture_selection"),
        label="immediate superseded prospective selection",
    )
    market_historical = _mapping(
        document.get("market_time_embedded_historical_selection"),
        label="market-time embedded historical selection",
    )
    market_time = _mapping(
        document.get("current_market_time_regime"), label="market-time selection"
    )
    operator = _mapping(document.get("operator_resolution"), label="operator rules")

    current_document = _read_linked_json(root, current, label="current selection")
    historical_document = _read_linked_json(
        root, historical, label="immediate superseded selection"
    )
    market_historical_document = _read_linked_json(
        root,
        market_historical,
        label="market-time embedded historical selection",
    )
    market_document = _read_linked_json(root, market_time, label="market-time contract")

    current_supersedes = _mapping(
        current_document.get("supersedes"), label="current selection supersession"
    )
    _equal(
        current_supersedes.get("path"),
        historical.get("path"),
        "current selection superseded path",
    )
    _equal(
        current_supersedes.get("sha256"),
        historical.get("sha256"),
        "current selection superseded hash",
    )
    _equal(
        current_supersedes.get("selected_ledger_version"),
        historical.get("selected_ledger_version"),
        "current selection superseded version",
    )
    _equal(
        current_supersedes.get("reason"),
        "V10_EXTENDS_THE_CONTIGUOUS_LOCAL_PREFIX_WITH_A_STRICTLY_POST_DELIVERY_CAPTURE",
        "current selection supersession reason",
    )

    historical_supersedes = _mapping(
        historical_document.get("supersedes"),
        label="historical selection supersession",
    )
    prior_document = _read_linked_json(
        root,
        historical_supersedes,
        label="prior prospective selection",
    )
    _equal(
        historical_supersedes.get("selected_ledger_version"),
        "v8",
        "historical superseded version",
    )
    _equal(
        historical_supersedes.get("reason"),
        "V9_EXTENDS_THE_CONTIGUOUS_LOCAL_PREFIX_WITH_A_STRICTLY_POST_DELIVERY_CAPTURE",
        "historical supersession reason",
    )
    _equal(
        prior_document.get("schema_version"),
        "ch_lt_prospective_capture_ledger_selection.v2",
        "prior selection schema",
    )
    prior_supersedes = _mapping(
        prior_document.get("supersedes"), label="prior selection supersession"
    )
    _equal(
        prior_supersedes.get("path"),
        market_historical.get("path"),
        "prior selection superseded path",
    )
    _equal(
        prior_supersedes.get("sha256"),
        market_historical.get("sha256"),
        "prior selection superseded hash",
    )
    _equal(
        prior_supersedes.get("reason"),
        "V8_EXTENDS_THE_CONTIGUOUS_LOCAL_PREFIX_WITH_A_POST_DELIVERY_CAPTURE",
        "prior selection supersession reason",
    )

    _equal(current.get("selected_ledger_version"), "v10", "current ledger version")
    _equal(
        current.get("selected_ledger_sha256"),
        "3805b71f1368a0e742c8627d4995a85c791de1360257f97b78be0b1665723140",
        "current ledger SHA-256",
    )
    _equal(current.get("current_for_local_structural_diagnostics"), True, "local selection")
    for field in ("scientific_admission", "production_authorization"):
        _equal(current.get(field), False, f"current selection {field}")
    _equal(
        current_document.get("schema_version"),
        "ch_lt_prospective_capture_ledger_selection.v3",
        "current selection schema",
    )
    selected = _mapping(current_document.get("selected"), label="current selected ledger")
    _equal(selected.get("version"), "v10", "selected ledger version")
    _equal(selected.get("sha256"), current.get("selected_ledger_sha256"), "selected ledger hash")
    _equal(
        selected.get("execution_receipt_sha256"),
        current.get("selected_execution_receipt_sha256"),
        "selected execution receipt",
    )
    _audit_selected_v10(
        root=root,
        current_reference=current,
        selection_document=current_document,
        selected=selected,
    )

    _equal(
        historical.get("selected_ledger_version"),
        "v9",
        "immediate superseded ledger version",
    )
    _equal(
        historical.get("selected_ledger_sha256"),
        "62cbdb981e5c5f98cc26dd2f09d08c7b70383e24d7b2e6a62c2bd0212583909d",
        "immediate superseded ledger hash",
    )
    _equal(
        historical.get("may_be_used_as_current_prospective_selection"),
        False,
        "immediate superseded current-selection eligibility",
    )
    _equal(
        historical_document.get("schema_version"),
        "ch_lt_prospective_capture_ledger_selection.v3",
        "immediate superseded selection schema",
    )
    historical_selected = _mapping(
        historical_document.get("selected"),
        label="immediate superseded selected ledger",
    )
    _equal(historical_selected.get("version"), "v9", "superseded selected version")
    _equal(
        historical_selected.get("sha256"),
        historical.get("selected_ledger_sha256"),
        "superseded selected ledger hash",
    )

    _equal(
        market_historical.get("selected_ledger_version"),
        "v5",
        "market-time historical ledger version",
    )
    _equal(
        market_historical.get("may_be_used_as_current_prospective_selection"),
        False,
        "market-time historical current-selection eligibility",
    )
    _equal(
        market_historical_document.get("schema_version"),
        "ch_lt_prospective_capture_ledger_selection.v1",
        "market-time historical selection schema",
    )
    market_historical_selected = _mapping(
        market_historical_document.get("selected"),
        label="market-time historical selected ledger",
    )
    _equal(
        market_historical_selected.get("version"),
        "v5",
        "market-time historical selected version",
    )

    _equal(market_time.get("contract_id"), market_document.get("contract_id"), "market contract ID")
    _equal(
        market_time.get("embedded_local_hourly_selection_is_frozen_historical_snapshot"),
        True,
        "embedded snapshot classification",
    )
    _equal(
        market_time.get("controls_current_prospective_capture_selection"),
        False,
        "market contract prospective control",
    )
    _equal(market_time.get("controls_market_time_regime_only"), True, "market contract scope")
    for field in ("transition_admitted", "production_authorization"):
        _equal(market_time.get(field), False, f"market-time {field}")
    evidence = _mapping(market_document.get("evidence"), label="market evidence")
    embedded = _mapping(
        evidence.get("local_hourly_observation"), label="embedded hourly snapshot"
    )
    _equal(
        embedded.get("selection_manifest_sha256"),
        market_historical.get("sha256"),
        "embedded historical selection hash",
    )
    _equal(
        embedded.get("selected_ledger_sha256"),
        market_historical_selected.get("sha256"),
        "embedded historical ledger hash",
    )

    for field in (
        "must_resolve_current_evidence_through_this_registry",
        "must_not_infer_current_ledger_from_market_time_embedded_snapshot",
        "divergent_or_missing_link_fails_closed",
    ):
        _equal(operator.get(field), True, f"operator rule {field}")
    for field in (
        "v9_may_be_used_as_current_prospective_selection",
        "v8_may_be_used_as_current_prospective_selection",
        "v5_may_be_used_as_current_prospective_selection",
        "v6_timeout_artifact_selectable",
        "v7_future_chronology_rejection_selectable",
        "full_day_v2_capture_prospective_ledger_eligible",
    ):
        _equal(operator.get(field), False, f"operator rule {field}")
    _equal(
        operator.get("v10_current_for_local_structural_diagnostics"),
        True,
        "operator rule v10 current local diagnostics",
    )
    _equal(document.get("independent_rolling_origin_count"), 0, "rolling origins")
    for field in (
        "future_holdout_consumed",
        "t057_consumed",
        "scientific_admission",
        "candidate_assembly_eligible",
        "publication_authorized",
        "promotion_gate",
        "production_authorization",
    ):
        _equal(document.get(field), False, field)

    return {
        "schema_version": "ch_lt_current_evidence_selection_audit.v1",
        "status": AUDIT_STATUS,
        "registry_id": registry_id,
        "registry_sha256": REGISTRY_SHA256,
        "current_prospective_ledger_version": "v10",
        "current_prospective_ledger_sha256": current.get("selected_ledger_sha256"),
        "superseded_prospective_ledger_version": "v9",
        "market_time_embedded_ledger_version": "v5",
        "market_time_embedded_ledger_is_historical_snapshot": True,
        "independent_rolling_origin_count": 0,
        "future_holdout_consumed": False,
        "t057_consumed": False,
        "scientific_admission": False,
        "candidate_assembly_eligible": False,
        "publication_authorized": False,
        "promotion_gate": False,
        "production_authorization": False,
    }


def _audit_selected_v10(
    *,
    root: Path,
    current_reference: Mapping[str, object],
    selection_document: Mapping[str, object],
    selected: Mapping[str, object],
) -> None:
    """Reopen the selected ledger, request, receipts, and installed runtime."""

    _equal(
        selected.get("path"),
        current_reference.get("selected_ledger_path"),
        "selected ledger path",
    )
    ledger = _read_linked_json(
        root,
        {"path": selected.get("path"), "sha256": selected.get("sha256")},
        label="selected v10 ledger",
    )
    execution = _read_linked_json(
        root,
        {
            "path": selected.get("execution_receipt_path"),
            "sha256": selected.get("execution_receipt_sha256"),
        },
        label="selected v10 execution receipt",
    )
    request = _read_linked_json(
        root,
        {
            "path": selected.get("request_path"),
            "sha256": selected.get("request_sha256"),
        },
        label="selected v10 construction request",
    )
    runtime = _read_linked_json(
        root,
        {
            "path": selected.get("runtime_receipt_path"),
            "sha256": selected.get("runtime_receipt_sha256"),
        },
        label="selected v10 runtime receipt",
    )

    _equal(ledger.get("schema_version"), "ch_lt_prospective_capture_ledger.v1", "ledger schema")
    _equal(
        ledger.get("status"),
        "CONTIGUOUS_LOCAL_QUARANTINE_NOT_PIT_AUTHORITY",
        "ledger status",
    )
    _equal(ledger.get("ledger_id"), selected.get("ledger_id"), "ledger ID")
    _equal(ledger.get("market"), "CH", "ledger market")
    _equal(ledger.get("source_role"), "epex_ch", "ledger source role")
    construction = _mapping(
        ledger.get("construction_request"), label="ledger construction request"
    )
    _equal(construction.get("path"), selected.get("request_path"), "ledger request path")
    _equal(construction.get("sha256"), selected.get("request_sha256"), "ledger request hash")
    window = _mapping(ledger.get("window"), label="ledger window")
    for field in ("start_utc", "end_utc_exclusive", "duration_hours"):
        selected_field = "window_" + field if field != "duration_hours" else field
        _equal(window.get(field), selected.get(selected_field), f"ledger window {field}")
    inventory = _mapping(ledger.get("inventory"), label="ledger inventory")
    for field in (
        "capture_count",
        "native_hourly_observation_count",
        "transport_quarter_hour_row_count",
        "proxy_rows_add_independent_information",
    ):
        _equal(inventory.get(field), selected.get(field), f"ledger inventory {field}")
    _equal(inventory.get("native_cadence_seconds"), 3600, "native cadence")
    _equal(inventory.get("transport_cadence_seconds"), 900, "transport cadence")
    _equal(
        inventory.get("transport_sampling_kind"),
        "UPSAMPLED_STEPWISE_PROXY",
        "transport sampling kind",
    )
    scope = _mapping(ledger.get("scientific_scope"), label="ledger scientific scope")
    _equal(scope.get("independent_rolling_origin_count"), 0, "ledger origin count")
    _equal(scope.get("model_selection_eligible"), False, "ledger model selection")
    _equal(scope.get("future_holdout_eligible"), False, "ledger future holdout")
    _equal(
        scope.get("price_level_shaping_or_economic_inference_allowed"),
        False,
        "ledger economic inference",
    )
    _equal(
        scope.get("structural_cadence_and_exact_replay_check_allowed"),
        True,
        "ledger structural diagnostic",
    )
    ledger_entries = ledger.get("entries")
    if not isinstance(ledger_entries, list) or len(ledger_entries) != selected.get(
        "capture_count"
    ):
        raise CurrentEvidenceSelectionError("selected ledger entry count is invalid")
    last_entry = _mapping(ledger_entries[-1], label="selected ledger final entry")
    new_capture = _mapping(
        selection_document.get("new_capture"), label="selected new capture"
    )
    for entry_field, capture_field in (
        ("start_utc", "start_utc"),
        ("end_utc", "end_utc_exclusive"),
        ("received_at_utc_untrusted", "received_at_utc_untrusted"),
        ("source_observation_count", "source_observation_count"),
        ("output_observation_count", "transport_quarter_hour_row_count"),
    ):
        _equal(
            last_entry.get(entry_field),
            new_capture.get(capture_field),
            f"selected new capture {capture_field}",
        )
    for field in (
        "future_holdout_consumed",
        "t057_consumed",
        "scientific_admission",
        "candidate_assembly_eligible",
        "publication_authorized",
        "promotion_gate",
        "production_authorization",
        "training_eligible",
    ):
        _equal(ledger.get(field), False, f"selected ledger {field}")

    _equal(
        request.get("schema_version"),
        "ch_lt_prospective_capture_ledger_request.v1",
        "request schema",
    )
    _equal(request.get("market"), "CH", "request market")
    _equal(request.get("source_role"), "epex_ch", "request source role")
    request_entries = request.get("entries")
    if not isinstance(request_entries, list) or len(request_entries) != 5:
        raise CurrentEvidenceSelectionError("selected request entry count is invalid")
    request_path = _repository_relative_path(
        root,
        selected.get("request_path"),
        label="selected v10 construction request",
    )
    try:
        reconstructed_ledger = build_prospective_capture_ledger(
            repo_root=root,
            request_path=request_path,
            expected_request_sha256=str(selected.get("request_sha256")),
        )
    except (OSError, ChLtProspectiveCaptureLedgerError) as exc:
        raise CurrentEvidenceSelectionError(
            "selected ledger recursive evidence reconstruction failed"
        ) from exc
    _equal(ledger, reconstructed_ledger, "selected ledger exact reconstruction")

    _equal(
        execution.get("schema_version"),
        "ch_lt_prospective_capture_ledger_execution_receipt.v1",
        "execution receipt schema",
    )
    _equal(
        execution.get("status"),
        "PASS_LOCAL_CONSTRUCTION_OBSERVATION_NOT_AUTHORITY",
        "execution receipt status",
    )
    execution_ledger = _mapping(execution.get("ledger"), label="execution ledger")
    _equal(execution_ledger.get("sha256"), selected.get("sha256"), "execution ledger hash")
    _equal(execution_ledger.get("ledger_id"), selected.get("ledger_id"), "execution ledger ID")
    execution_request = _mapping(execution.get("request"), label="execution request")
    _equal(execution_request.get("sha256"), selected.get("request_sha256"), "execution request hash")
    execution_runtime = _mapping(
        execution.get("constructor_runtime_receipt"), label="execution runtime"
    )
    _equal(
        execution_runtime.get("sha256"),
        selected.get("runtime_receipt_sha256"),
        "execution runtime hash",
    )
    _equal(execution.get("sys_path"), runtime.get("sys_path"), "execution sys.path")
    for field in (
        "future_holdout_consumed",
        "t057_consumed",
        "runtime_authority",
        "scientific_admission",
        "promotion_gate",
        "production_authorization",
    ):
        _equal(execution.get(field), False, f"execution receipt {field}")

    _equal(runtime.get("schema_version"), "fmv_lt_launcherless_local_runtime.v5", "runtime schema")
    _equal(runtime.get("status"), "PASS", "runtime status")
    _equal(runtime.get("local_quality_authorization"), True, "runtime local quality")
    _equal(runtime.get("production_authorization"), False, "runtime production")
    _equal(
        runtime.get("project_source_revision"),
        selected.get("project_source_revision"),
        "runtime project source revision",
    )
    sys_path = runtime.get("sys_path")
    if not isinstance(sys_path, list) or len(sys_path) != 3 or any(
        str(root).casefold() == str(path).casefold() for path in sys_path
    ):
        raise CurrentEvidenceSelectionError("selected runtime sys.path is invalid")
    sources = runtime.get("sources")
    if not isinstance(sources, list):
        raise CurrentEvidenceSelectionError("selected runtime sources are invalid")
    project_wheels = [
        source
        for source in sources
        if isinstance(source, Mapping) and source.get("kind") == "GOVERNED_PROJECT_WHEEL"
    ]
    if len(project_wheels) != 1:
        raise CurrentEvidenceSelectionError("selected runtime project wheel is ambiguous")
    _equal(
        project_wheels[0].get("sha256"),
        selected.get("project_wheel_sha256"),
        "selected project wheel hash",
    )


def _read_linked_json(
    root: Path, reference: Mapping[str, object], *, label: str
) -> Mapping[str, object]:
    candidate = _repository_relative_path(root, reference.get("path"), label=label)
    expected = _sha256(reference.get("sha256"), label=label)
    return _strict_mapping(_read_exact(candidate, expected, label=label), label=label)


def _repository_relative_path(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CurrentEvidenceSelectionError(f"{label} path is invalid")
    parts = value.split("/")
    if Path(value).is_absolute() or any(
        part in {"", ".", ".."}
        or _WINDOWS_INVALID_PATH_CHARS_RE.search(part) is not None
        or part.endswith((" ", "."))
        or part.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_BASENAMES
        for part in parts
    ):
        raise CurrentEvidenceSelectionError(f"{label} path is invalid")
    normalized_root = Path(os.path.abspath(root))
    candidate = Path(os.path.abspath(normalized_root.joinpath(*parts)))
    try:
        common = Path(os.path.commonpath((str(normalized_root), str(candidate))))
    except ValueError as exc:
        raise CurrentEvidenceSelectionError(f"{label} path is invalid") from exc
    if _path_key(common) != _path_key(normalized_root):
        raise CurrentEvidenceSelectionError(f"{label} escapes the repository")
    return candidate


def _read_exact(path: Path, expected_sha256: str, *, label: str) -> bytes:
    payload = read_stable_single_link_file(path, label=label, max_bytes=_MAX_BYTES)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise CurrentEvidenceSelectionError(f"{label} SHA-256 mismatch")
    return payload


def _strict_mapping(payload: bytes, *, label: str) -> Mapping[str, object]:
    try:
        value = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CurrentEvidenceSelectionError(f"{label} is invalid JSON") from exc
    return _mapping(value, label=label)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise CurrentEvidenceSelectionError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise CurrentEvidenceSelectionError(f"{label} keys are not exact")


def _equal(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise CurrentEvidenceSelectionError(f"{label} is invalid")


def _sha256(value: object, *, label: str) -> str:
    _require_sha256(value, label=label)
    assert isinstance(value, str)
    return value


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CurrentEvidenceSelectionError(f"{label} SHA-256 is invalid")


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        result = audit_current_selection(
            registry_path=args.registry,
            expected_registry_sha256=args.expected_registry_sha256,
        )
    except (OSError, CurrentEvidenceSelectionError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "ch_lt_current_evidence_selection_audit.v1",
                    "status": "CURRENT_SELECTION_INVALID_NO_GO",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "future_holdout_consumed": False,
                    "t057_consumed": False,
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
