"""Audit the selected local future CH origin without opening any outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import pandas as pd

from pfc_shaping.build_identity import SOURCE_REVISION
from pfc_shaping.package_contract import (
    ALLOWED_RUNTIME_PYTHON_FILES,
    BUILD_IDENTITY_PLACEHOLDER,
    BUILD_IDENTITY_TEMPLATE,
)
from pfc_shaping.path_safety import path_is_link, read_stable_single_link_file
from pfc_shaping.pipeline.strict_structured_data import load_strict_json
from pfc_shaping.validation import (
    ch_lt_structural_prediction_commitment as prediction_builder,
)

# Capture the governed invocation root once without embedding a workstation path.
ROOT = Path.cwd()
REGISTRY_RELATIVE = Path(
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-LOCAL-FUTURE-ORIGIN-SELECTION-V6-20260731.json"
)
REGISTRY_SHA256 = "11966d1ee85ace46e97006fa74f8aab4789c71f7438de909ed71541aab480df7"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_BYTES = 64 * 1024 * 1024
_CURRENT_EVIDENCE_AUDIT_EXECUTION = (
    "build/workspace-local-runs/cev31v3h/execution-receipt.json",
    "c0c63129c36b2e6c25070a3851defa822e060ab89ca7817c9e7021fc43f26994",
)
_CURRENT_EVIDENCE_AUDIT_SUPERVISOR = (
    "build/workspace-local-supervisors/cev31v3h/supervisor-receipt.json",
    "6a5663c6b24a7b68effbedfe10638da2e95ca36f2667c36a30e469a25f364a04",
)
_TOP_LEVEL_KEYS = {
    "schema_version",
    "selection_id",
    "status",
    "market",
    "supersession",
    "selected_origin",
    "bindings",
    "implementation_binding",
    "evaluation_contract",
    "truth_lifecycle",
    "authority_boundary",
    "next_external_requirements",
    "future_holdout_consumed",
    "t057_consumed",
    "scientific_admission",
    "production_authorization",
    "promotion_gate",
}
_ORIGIN_KEYS = {
    "origin_as_of_utc",
    "origin_id",
    "first_target_delivery_start_utc",
    "last_target_delivery_end_utc",
    "origin_precedes_first_target",
    "local_clock_trusted",
    "duplicate_origin_as_of_utc",
}
_AUTHORITY_BOUNDARY = {
    "independent_registry_receipt_present": False,
    "independent_trusted_time_receipt_present": False,
    "independent_signature_present": False,
    "builder_inaccessible_external_cas_receipt_present": False,
    "fresh_governed_ch_eex_level_present": False,
    "direct_ch_shape_authority_present": False,
    "countable_prospective_origin": False,
    "outcomes_opened": False,
    "future_holdout_consumed": False,
    "t057_consumed": False,
    "scientific_admission": False,
    "candidate_assembly_eligible": False,
    "publication_authorized": False,
    "promotion_gate": False,
    "production_authorization": False,
}
_EVALUATION_CONTRACT = {
    "source": "BOUND_HOURLY_SCORING_CONTRACT_ONLY_NO_DUPLICATED_FORMULAS",
    "hourly_shape_functional_contract_sha256": "a8cb845385d7da16b0e486b2fa16330d312c2d2f6a8fba2028e2e0d08a11fbbe",
    "hourly_scoring_contract_sha256": "d84d7ebc2f3c6aa4b820fab26f0a5de18940a6d35af8e796323025eab694ea4d",
    "target_count": 36,
    "prediction_count": 108,
    "primary_scenario": "central",
    "scenario_pooling": "FORBIDDEN",
    "quarter_hour_scoring": "FORBIDDEN_UNTIL_MARKET_TRANSITION_IS_INDEPENDENTLY_ADMITTED",
}
_TRUTH_LIFECYCLE = {
    "native_hourly_collector": (
        "LEDGER_BOUND_CAPTURE_PIPELINE_PRESENT_NOT_OPERATIONALLY_SCHEDULED"
    ),
    "truth_manifest_builder": "PACKAGE_NAMESPACE_SOURCE_BOUND_LOCAL_LINTED_NO_GO",
    "truth_open_guard": (
        "LOCAL_WALL_CLOCK_ONLY_SOURCE_BOUND_NOT_TRUSTED_TIME_NO_GO"
    ),
    "truth_publication_receipt": (
        "IMPLEMENTED_POST_READ_SOURCE_BOUND_NOT_AUTHORIZATION_NO_GO"
    ),
    "precommitted_scorer": "PACKAGE_NAMESPACE_SOURCE_BOUND_LOCAL_LINTED_NO_GO",
    "provider_lag_sla": "UNSET_REQUIRES_MEASUREMENT_AND_OWNER_APPROVAL",
    "operational_owner": "UNSET_EXTERNAL_FMV_DECISION",
    "partial_month_truth_open": "FORBIDDEN",
    "post_truth_retuning": "FORBIDDEN_FOR_THIS_ORIGIN",
}
_EXTERNAL_REQUIREMENTS = [
    "GENERATE_AND_RUNTIME_CONSUME_AN_EXHAUSTIVE_IMPORT_CLOSURE_BINDING_V6_CANONICAL_AUDITOR_EXACT_WHEEL_AND_LOADED_CODE",
    "SUPERVISE_THE_INSTALLED_TRUTH_AND_SCORING_MODULES_WITH_EXACT_RUNTIME_RECEIPTS",
    "OPERATE_AND_MONITOR_NATIVE_HOURLY_CAPTURE_WITH_OWNER_APPROVED_PROVIDER_LAG_SLA",
    "REGISTER_A_FUTURE_ORIGIN_THROUGH_AN_INDEPENDENT_LINEARIZABLE_AUTHORITY_BEFORE_ITS_FIRST_TARGET",
    "BIND_INDEPENDENT_TRUSTED_TIME_SIGNATURE_AND_BUILDER_INACCESSIBLE_EXTERNAL_CAS_RECEIPT",
    "BIND_FRESH_POINT_IN_TIME_CH_EEX_LEVEL_AND_ORIGIN_AVAILABLE_PRODUCT_INVENTORY",
    "ACCUMULATE_THE_PREDECLARED_NUMBER_OF_INDEPENDENT_ORIGINS_BEFORE_ANY_QUALITY_CLAIM",
]
_IMPLEMENTATION_BINDING = {
    "status": (
        "PRE_DELIVERY_PACKAGE_NAMESPACE_SOURCE_BYTES_BOUND_LOCAL_LINTED_"
        "NOT_INSTALLED_NOT_OPERATIONAL_NO_GO"
    ),
    "source_binding_completed_at_utc_untrusted": "2026-07-31T09:11:09Z",
    "completed_before_first_target_by_untrusted_local_clock": True,
    "sources": {
        "prospective_hourly_scoring_core": {
            "path": "pfc_shaping/validation/ch_lt_prospective_hourly_scoring.py",
            "sha256": (
                "7faf157e6e83c78c5de3b1419a8418ea1ad7fc8886297f0e7322ee08e37c1cac"
            ),
        },
        "prospective_hourly_scoring_cli": {
            "path": "pfc_shaping/cli/score_ch_lt_structural_prediction_commitment.py",
            "sha256": (
                "481a7c21dfeefbf7a7f998d634a4924632320b12d21b620fe55fb30170992cb0"
            ),
        },
        "native_hourly_truth_bundle_builder": {
            "path": "pfc_shaping/validation/ch_lt_native_hourly_truth_bundle.py",
            "sha256": (
                "698a06b698277d1fe542337c93df5f77831e7f3f74f7767ffa58dd2d05caaedf"
            ),
        },
        "structural_prediction_commitment_builder": {
            "path": (
                "pfc_shaping/validation/"
                "ch_lt_structural_prediction_commitment.py"
            ),
            "sha256": (
                "9949f8e818aba79d198853a03208e526fe90ef4c6ef937917bfe676aacbf8bef"
            ),
        },
    },
    "implemented_fail_closed_boundaries": {
        "selection_audited_before_outcome_path_read": True,
        "selected_commitment_predictions_and_scoring_contract_rehashed_before_truth_read": True,
        "local_wall_clock_maturity_checked_before_ledger_or_bronze_read": True,
        "caller_declared_truth_open_cannot_exceed_local_wall_clock": True,
        "truth_publication_receipt_required_before_scorer_truth_bundle_read": True,
        "truth_publication_receipt_is_post_read_evidence_not_open_authorization": True,
        "receipt_and_bundle_fixture_classification_biconditional": True,
        "complete_month_native_hourly_grid_required": True,
        "recursive_capture_ledger_reconstruction_required": True,
        "atomic_truth_bundle_and_publication_receipt": True,
        "atomic_score_publication_with_empty_staging_resume": True,
    },
    "execution_boundary": {
        "checkout_source_only": True,
        "package_namespace_sources": True,
        "wheel_contract_declared": True,
        "installed_package_bound": False,
        "supervised_runtime_receipt_present": False,
        "operational_scheduler_present": False,
        "independent_trusted_time_present": False,
        "canonical_auditor_bound_by_independent_closure_manifest": False,
        "scientific_authority": False,
        "production_authority": False,
    },
}
_EXPECTED_BINDINGS = {
    "origin_target_mask_inventory": (
        "build/ch-lt-origin-target-mask-inventories/"
        "structural-dry-run-20260731T072027Z-schema-v2.json",
        "a4d5722841df549ba24ca2dd190fa0f5b1d79738fc63680acf93034e148419b7",
    ),
    "prediction_commitment": (
        "build/ch-lt-structural-prediction-commitments/"
        "structural-dry-run-20260731T072027Z-v7-quant-contract/commitment.json",
        "67cc6fe722586cd4a02149252d1af7fbbb35e90a89e8bc7965f8d0bf87eda9b4",
    ),
    "sealed_predictions": (
        "build/ch-lt-structural-prediction-commitments/"
        "structural-dry-run-20260731T072027Z-v7-quant-contract/predictions.json",
        "859c4622eb5b699f22242af0121141729d6c00472b6b9070bdb87c2ba9132302",
    ),
    "hourly_scoring_contract": (
        ".planning/phases/14-lt-audit-remediation/"
        "CH-LT-FUTURE-HOURLY-SCORING-CONTRACT-V2-20260731.json",
        "d84d7ebc2f3c6aa4b820fab26f0a5de18940a6d35af8e796323025eab694ea4d",
    ),
    "current_evidence_selection": (
        ".planning/phases/14-lt-audit-remediation/"
        "CH-LT-CURRENT-EVIDENCE-SELECTION-V3-20260731.json",
        "6518f7e876ce1c233fc055d3d20ad213088d361d784afbe5aa16d4f165e744f7",
    ),
    "inventory_execution_receipt": (
        "build/workspace-local-runs/org31fresh/execution-receipt.json",
        "5553023b0605f7cee0de3981dbe4f025a2ea6c26be645a8852ee427facb13ce0",
    ),
    "inventory_supervisor_receipt": (
        "build/workspace-local-supervisors/org31fresh/supervisor-receipt.json",
        "62c3efeb9cc142d6c4071798d738d7b7bf12c4d01b060822f909541d9baccac8",
    ),
    "prediction_execution_receipt": (
        "build/workspace-local-runs/pred31v7b/execution-receipt.json",
        "ef4d711c4bea9301d4285df08daa8a8893226368453ab5c474ae48db19235f46",
    ),
    "prediction_supervisor_receipt": (
        "build/workspace-local-supervisors/pred31v7b/supervisor-receipt.json",
        "7edc535bf40bd04337483df09606aee2a36a71480d7469480d9099d5bc061b05",
    ),
}


class LocalFutureOriginSelectionError(ValueError):
    """Raised when the selected rehearsal is not exact and outcome-blind."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _verify_runtime_import_closure() -> dict[str, object]:
    """Consume the build-generated full source closure before registry bytes."""

    if SOURCE_REVISION is None:
        return {
            "status": "UNSEALED_CHECKOUT_NOT_RUNTIME_CLOSURE",
            "sealed_installed_runtime": False,
            "source_revision": None,
            "source_count": None,
            "package_root_occurrences_in_sys_path": None,
            "checkout_root_occurrences_in_sys_path": None,
        }
    if _SHA256_RE.fullmatch(SOURCE_REVISION) is None:
        raise LocalFutureOriginSelectionError("installed source revision is invalid")
    package_root = Path(__file__).resolve().parents[1]
    package_parent = package_root.parent
    expected = {
        path.removeprefix("pfc_shaping/") for path in ALLOWED_RUNTIME_PYTHON_FILES
    }
    actual = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*.py")
        if path.is_file()
    }
    if actual != expected:
        raise LocalFutureOriginSelectionError(
            "installed source inventory differs from the package contract"
        )
    forbidden_bytecode = [
        path
        for path in package_root.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in {".pyc", ".pyo"}
            or "__pycache__" in {part.lower() for part in path.parts}
        )
    ]
    if forbidden_bytecode:
        raise LocalFutureOriginSelectionError(
            "installed source closure contains forbidden bytecode"
        )
    identity_placeholder = BUILD_IDENTITY_TEMPLATE.format(
        source_revision=BUILD_IDENTITY_PLACEHOLDER
    ).encode("ascii")
    digest = hashlib.sha256()
    for relative in sorted(expected):
        path = package_root / relative
        if path_is_link(path):
            raise LocalFutureOriginSelectionError(
                f"installed source closure contains a link: {relative}"
            )
        metadata = path.stat(follow_symlinks=False)
        if metadata.st_nlink != 1:
            raise LocalFutureOriginSelectionError(
                f"installed source closure contains a hardlink: {relative}"
            )
        payload = read_stable_single_link_file(
            path,
            label=f"installed runtime source {relative}",
            max_bytes=_MAX_BYTES,
        )
        if relative == "build_identity.py":
            payload = identity_placeholder
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    if digest.hexdigest() != SOURCE_REVISION:
        raise LocalFutureOriginSelectionError(
            "installed source bytes differ from the build-generated closure"
        )
    normalized_parent = os.path.normcase(os.path.abspath(package_parent))
    package_root_occurrences = sum(
        1
        for item in sys.path
        if isinstance(item, str)
        and item
        and os.path.normcase(os.path.abspath(item)) == normalized_parent
    )
    if package_root_occurrences != 1:
        raise LocalFutureOriginSelectionError(
            "installed package root must occur exactly once in sys.path"
        )
    normalized_checkout_root = os.path.normcase(os.path.abspath(ROOT))
    checkout_root_occurrences = sum(
        1
        for item in sys.path
        if isinstance(item, str)
        and item
        and os.path.normcase(os.path.abspath(item)) == normalized_checkout_root
    )
    if normalized_checkout_root != normalized_parent and checkout_root_occurrences != 0:
        raise LocalFutureOriginSelectionError(
            "installed prospective audit sys.path contains the source checkout"
        )
    for name, module in tuple(sys.modules.items()):
        if name == "scripts" or name.startswith("scripts."):
            raise LocalFutureOriginSelectionError(
                "installed prospective audit loaded a scripts module"
            )
        if name == "pfc_shaping.ct" or name.startswith("pfc_shaping.ct."):
            raise LocalFutureOriginSelectionError(
                "installed prospective audit loaded a CT module"
            )
        if name != "pfc_shaping" and not name.startswith("pfc_shaping."):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        try:
            Path(origin).resolve().relative_to(package_root)
        except ValueError as exc:
            raise LocalFutureOriginSelectionError(
                f"loaded project module escaped installed package root: {name}"
            ) from exc
    return {
        "status": "SEALED_INSTALLED_FULL_SOURCE_IMPORT_CLOSURE_VERIFIED",
        "sealed_installed_runtime": True,
        "source_revision": SOURCE_REVISION,
        "source_count": len(expected),
        "package_root_occurrences_in_sys_path": package_root_occurrences,
        "checkout_root_occurrences_in_sys_path": checkout_root_occurrences,
    }


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise LocalFutureOriginSelectionError(f"{label} must be an object")
    return value


def _equal(actual: object, expected: object, *, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise LocalFutureOriginSelectionError(f"{label} is invalid")


def _safe_path(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise LocalFutureOriginSelectionError(f"{label} path is invalid")
    parts = relative.split("/")
    if (
        Path(relative).is_absolute()
        or any(part in {"", ".", ".."} for part in parts)
        or any(":" in part or part.endswith((" ", ".")) for part in parts)
    ):
        raise LocalFutureOriginSelectionError(f"{label} path is invalid")
    candidate = Path(os.path.abspath(root.joinpath(*parts)))
    try:
        common = os.path.commonpath((os.path.abspath(root), os.path.abspath(candidate)))
    except ValueError as exc:
        raise LocalFutureOriginSelectionError(f"{label} path is invalid") from exc
    if os.path.normcase(common) != os.path.normcase(os.path.abspath(root)):
        raise LocalFutureOriginSelectionError(f"{label} escapes the repository")
    return candidate


def _read_json(path: Path, expected_sha256: str, *, label: str) -> Mapping[str, object]:
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise LocalFutureOriginSelectionError(f"{label} SHA-256 is invalid")
    payload = read_stable_single_link_file(path, label=label, max_bytes=_MAX_BYTES)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise LocalFutureOriginSelectionError(f"{label} SHA-256 mismatch")
    try:
        decoded = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise LocalFutureOriginSelectionError(f"{label} is not strict JSON") from exc
    return _mapping(decoded, label=label)


def _verify_implementation_binding(value: Mapping[str, object]) -> int:
    _equal(value, _IMPLEMENTATION_BINDING, label="implementation binding")
    sources = _mapping(value.get("sources"), label="implementation sources")
    for name, raw_reference in sources.items():
        reference = _mapping(raw_reference, label=f"implementation source {name}")
        expected_sha256 = str(reference.get("sha256"))
        if _SHA256_RE.fullmatch(expected_sha256) is None:
            raise LocalFutureOriginSelectionError(
                f"implementation source {name} SHA-256 is invalid"
            )
        payload = read_stable_single_link_file(
            _safe_path(
                ROOT,
                reference.get("path"),
                label=f"implementation source {name}",
            ),
            label=f"implementation source {name}",
            max_bytes=_MAX_BYTES,
        )
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise LocalFutureOriginSelectionError(
                f"implementation source {name} SHA-256 mismatch"
            )
    return len(sources)


def _parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise LocalFutureOriginSelectionError(f"{label} is invalid")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise LocalFutureOriginSelectionError(f"{label} is invalid") from exc
    return parsed


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise LocalFutureOriginSelectionError(f"{label} keys are not exact")


def _verify_prediction_chain(
    *,
    inventory: Mapping[str, object],
    commitment: Mapping[str, object],
    predictions: Mapping[str, object],
) -> int:
    _equal(
        predictions.get("schema_version"),
        "ch_lt_structural_prediction_set.v3",
        label="prediction schema",
    )
    prediction_core = dict(predictions)
    prediction_set_id = prediction_core.pop("prediction_set_id", None)
    _equal(
        prediction_set_id,
        hashlib.sha256(_canonical(prediction_core)).hexdigest(),
        label="prediction set id",
    )
    _equal(
        predictions.get("hourly_shape_functional_contract"),
        prediction_builder._HOURLY_SHAPE_FUNCTIONAL_CONTRACT,
        label="hourly functional contract",
    )
    raw_targets = inventory.get("targets")
    raw_predictions = predictions.get("predictions")
    if not isinstance(raw_targets, list) or not isinstance(raw_predictions, list):
        raise LocalFutureOriginSelectionError("prediction grid is invalid")

    commitment_bindings = _mapping(commitment.get("bindings"), label="commitment bindings")
    completion_reference = _mapping(
        commitment_bindings.get("model_completion"), label="model completion binding"
    )
    completion = _read_json(
        _safe_path(ROOT, completion_reference.get("path"), label="model completion"),
        str(completion_reference.get("sha256")),
        label="model completion",
    )
    nested_count = 1
    for name in (
        "candidate_core",
        "local_quality_selection",
        "model_execution_receipt",
    ):
        reference = _mapping(commitment_bindings.get(name), label=f"{name} binding")
        _read_json(
            _safe_path(ROOT, reference.get("path"), label=name),
            str(reference.get("sha256")),
            label=name,
        )
        nested_count += 1

    frames: dict[str, pd.DataFrame] = {}
    pfc_bindings: dict[str, object] = {}
    for role in prediction_builder._PFC_ROLES:
        frames[role], pfc_bindings[role] = prediction_builder._bound_pfc(
            ROOT, completion, role
        )
        nested_count += 1
    _equal(
        commitment_bindings.get("pfc_outputs"),
        pfc_bindings,
        label="PFC output bindings",
    )

    expected_predictions: list[dict[str, object]] = []
    universe: list[dict[str, object]] = []
    for raw_target in raw_targets:
        target = _mapping(raw_target, label="inventory target")
        start = pd.Timestamp(str(target.get("delivery_start_utc")))
        end = pd.Timestamp(str(target.get("delivery_end_utc")))
        expected_qh = int(target.get("expected_quarter_hour_interval_count"))
        expected_hours = int(target.get("expected_native_hourly_interval_count"))
        universe.append(
            {
                "target_id": target.get("target_id"),
                "lead_month": target.get("lead_month"),
                "delivery_month": target.get("delivery_month"),
                "delivery_start_utc": target.get("delivery_start_utc"),
                "delivery_end_utc": target.get("delivery_end_utc"),
                "expected_quarter_hour_interval_count": expected_qh,
                "expected_native_hourly_interval_count": expected_hours,
            }
        )
        for role in prediction_builder._PFC_ROLES:
            selected = frames[role].loc[
                (frames[role].index >= start) & (frames[role].index < end),
                "price_shape",
            ]
            if len(selected) != expected_qh or not selected.notna().all():
                raise LocalFutureOriginSelectionError("selected PFC target is incomplete")
            functionals = prediction_builder._hourly_shape_functionals(
                selected,
                expected_hour_count=expected_hours,
                label=f"{role} target {target.get('target_id')}",
            )
            expected_predictions.append(
                {
                    "target_id": target.get("target_id"),
                    "lead_month": target.get("lead_month"),
                    "delivery_month": target.get("delivery_month"),
                    "scenario": role.removeprefix("pfc_"),
                    "quarter_hour_interval_count": len(selected),
                    "native_hourly_interval_count": expected_hours,
                    "monthly_mean_eur_mwh": format(float(selected.mean()), ".17g"),
                    "hourly_shape_functionals": functionals,
                    "probabilistic_interval_supported": False,
                }
            )
    _equal(raw_predictions, expected_predictions, label="reconstructed predictions")

    sealed = _mapping(commitment.get("sealed_payload"), label="sealed payload")
    prediction_bytes = read_stable_single_link_file(
        ROOT / _EXPECTED_BINDINGS["sealed_predictions"][0],
        label="sealed predictions",
        max_bytes=_MAX_BYTES,
    )
    _equal(
        sealed.get("sha256"),
        hashlib.sha256(prediction_bytes).hexdigest(),
        label="sealed prediction hash",
    )
    commitments = _mapping(commitment.get("commitments"), label="commitments")
    scenario_payload = {
        "scenario_order": ["slow", "central", "fast"],
        "primary_scenario": "central",
        "sensitivity_scenarios": ["slow", "fast"],
        "scenario_pooling_for_quality_claims": "FORBIDDEN",
        "structural_weights": ["0.25", "0.50", "0.25"],
        "structural_weights_use": "DISPLAY_ONLY_NOT_SCORING_NOT_PROBABILITY",
        "probabilistic_interpretation": False,
        "hourly_shape_functional_contract": prediction_builder._HOURLY_SHAPE_FUNCTIONAL_CONTRACT,
        "pfc_bindings": pfc_bindings,
    }
    mask_policy = _mapping(inventory.get("mask_policy"), label="mask policy")
    expected_commitments = {
        "prediction_commitment_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "scenario_commitment_sha256": hashlib.sha256(_canonical(scenario_payload)).hexdigest(),
        "structural_target_universe_commitment_sha256": hashlib.sha256(
            _canonical(universe)
        ).hexdigest(),
        "ex_ante_evaluation_mask_rule_commitment_sha256": hashlib.sha256(
            _canonical(mask_policy)
        ).hexdigest(),
        "candidate_hypothesis_and_procedure_commitment_sha256": prediction_builder.CANDIDATE_SHA256,
        "hourly_shape_functional_contract_sha256": hashlib.sha256(
            _canonical(prediction_builder._HOURLY_SHAPE_FUNCTIONAL_CONTRACT)
        ).hexdigest(),
    }
    _equal(commitments, expected_commitments, label="reconstructed commitments")
    commitment_core = dict(commitment)
    commitment_id = commitment_core.pop("commitment_id", None)
    _equal(
        commitment_id,
        hashlib.sha256(_canonical(commitment_core)).hexdigest(),
        label="commitment id",
    )
    return nested_count


def _verify_scoring_formula_v1(contract: Mapping[str, object]) -> int:
    _equal(
        contract.get("schema_version"),
        "ch_lt_future_hourly_scoring_contract.v1",
        label="scoring contract schema",
    )
    _equal(
        contract.get("status"),
        "OUTCOME_BLIND_LOCAL_SCORING_SPECIFICATION_NO_GO",
        label="scoring contract status",
    )
    contract_core = dict(contract)
    contract_id = contract_core.pop("contract_id", None)
    _equal(
        contract_id,
        hashlib.sha256(_canonical(contract_core)).hexdigest(),
        label="scoring contract id",
    )
    bindings = _mapping(contract.get("bindings"), label="scoring bindings")
    _mapping(
        bindings.get("prediction_functional_contract"),
        label="prediction functional binding",
    )
    nested_count = 0
    for name in ("estimand_contract", "dependence_and_power_design"):
        reference = _mapping(bindings.get(name), label=f"scoring {name}")
        _read_json(
            _safe_path(ROOT, reference.get("path"), label=f"scoring {name}"),
            str(reference.get("sha256")),
            label=f"scoring {name}",
        )
        nested_count += 1
    point_scores = _mapping(contract.get("point_scores"), label="point scores")
    zero_mean = _mapping(
        point_scores.get("zero_mean_hourly_shape"), label="zero-mean scores"
    )
    _equal(zero_mean.get("primary_metric"), "MAE_EUR_MWH", label="primary metric")
    _equal(
        zero_mean.get("pearson_zero_variance"),
        "UNSUPPORTED_NOT_ZERO_NOT_WIN",
        label="Pearson zero variance",
    )
    aggregation = _mapping(contract.get("aggregation"), label="aggregation")
    _equal(
        aggregation.get("cross_origin_weight"),
        "EQUAL_WEIGHT_PER_UNIQUE_ORIGIN_AS_OF_UTC",
        label="origin weighting",
    )
    _equal(
        aggregation.get("missing_or_nonfinite"),
        "UNSUPPORTED_NEVER_IMPUTED_NEVER_ZERO_NEVER_WIN",
        label="missing data",
    )
    negative = _mapping(
        contract.get("negative_price_boundary"), label="negative price boundary"
    )
    _equal(
        negative.get("negative_hour_count_is_proper_score"),
        False,
        label="negative count score",
    )
    authority = _mapping(
        contract.get("authority_boundary"), label="scoring authority boundary"
    )
    if not authority or any(value is not False for value in authority.values()):
        raise LocalFutureOriginSelectionError(
            "scoring authority boundary must be nonempty and all false"
        )
    return nested_count


def _verify_scoring_contract(contract: Mapping[str, object]) -> int:
    _equal(
        contract.get("schema_version"),
        "ch_lt_future_hourly_scoring_contract.v2",
        label="selected scoring contract schema",
    )
    _equal(
        contract.get("status"),
        "OUTCOME_BLIND_LOCAL_SCORING_SPECIFICATION_V7_NO_GO",
        label="selected scoring contract status",
    )
    contract_core = dict(contract)
    contract_id = contract_core.pop("contract_id", None)
    _equal(
        contract_id,
        hashlib.sha256(_canonical(contract_core)).hexdigest(),
        label="selected scoring contract id",
    )
    supersedes = _mapping(contract.get("supersedes"), label="scoring supersession")
    _equal(
        supersedes.get("sha256"),
        "e6f0767de168c02fefa2a26cb90b147e8491d5a16cdd008ed137e1cd430a237a",
        label="scoring formula source hash",
    )
    formula_source = _read_json(
        _safe_path(ROOT, supersedes.get("path"), label="scoring formula source"),
        str(supersedes.get("sha256")),
        label="scoring formula source",
    )
    nested_count = 1 + _verify_scoring_formula_v1(formula_source)
    selected = _mapping(
        contract.get("selected_prediction_binding"),
        label="selected prediction scoring binding",
    )
    _equal(
        selected.get("commitment_sha256"),
        _EXPECTED_BINDINGS["prediction_commitment"][1],
        label="selected scoring commitment",
    )
    _equal(
        selected.get("predictions_sha256"),
        _EXPECTED_BINDINGS["sealed_predictions"][1],
        label="selected scoring predictions",
    )
    _equal(
        selected.get("hourly_shape_functional_contract_sha256"),
        hashlib.sha256(
            _canonical(prediction_builder._HOURLY_SHAPE_FUNCTIONAL_CONTRACT)
        ).hexdigest(),
        label="selected scoring functional contract",
    )
    clarifications = _mapping(
        contract.get("clarifications"), label="scoring clarifications"
    )
    _equal(
        clarifications.get("negative_price_support_observed"),
        False,
        label="negative price support observation",
    )
    _equal(
        clarifications.get("single_origin_role"),
        "DIAGNOSTIC_ONLY_NO_MODEL_SELECTION_NO_INFERENCE",
        label="single-origin role",
    )
    authority = _mapping(
        contract.get("authority_boundary"), label="selected scoring authority"
    )
    if not authority or any(value is not False for value in authority.values()):
        raise LocalFutureOriginSelectionError(
            "selected scoring authority boundary must be nonempty and all false"
        )
    return nested_count


def _verify_execution_receipt(
    receipt: Mapping[str, object], *, expected_command_tail: list[str], label: str
) -> None:
    _equal(
        receipt.get("schema_version"),
        "pfc_lt_workspace_local_execution.v6",
        label=f"{label} schema",
    )
    _equal(receipt.get("workspace"), str(ROOT), label=f"{label} workspace")
    _equal(receipt.get("status"), "TARGET_EXIT_ZERO_NOT_AUTHORITY", label=f"{label} status")
    _equal(receipt.get("target_exit_code"), 0, label=f"{label} target exit")
    for field in (
        "scientific_authorization",
        "evaluation_authorization",
        "runtime_authority",
        "promotion_authorization",
        "production_authorization",
    ):
        _equal(receipt.get(field), False, label=f"{label} {field}")
    _equal(
        receipt.get("ambient_credential_isolation"),
        "ENV_DENYLIST_ONLY_NOT_FILESYSTEM_SANDBOX",
        label=f"{label} ambient isolation",
    )
    identity = _mapping(receipt.get("execution_identity"), label=f"{label} identity")
    interpreter = _mapping(
        identity.get("target_interpreter"), label=f"{label} interpreter"
    )
    command = receipt.get("redacted_command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise LocalFutureOriginSelectionError(f"{label} command is invalid")
    _equal(command[0], interpreter.get("path"), label=f"{label} interpreter command")
    _equal(command[1:], expected_command_tail, label=f"{label} command tail")
    source_tree = _mapping(identity.get("source_tree"), label=f"{label} source tree")
    if _SHA256_RE.fullmatch(str(source_tree.get("tree_sha256"))) is None:
        raise LocalFutureOriginSelectionError(f"{label} source tree is invalid")
    _equal(
        _mapping(identity.get("target_import_closure"), label=f"{label} closure").get(
            "status"
        ),
        "UNBOUND_EXTERNAL_INTERPRETER",
        label=f"{label} local-only interpreter boundary",
    )
    usage = _mapping(receipt.get("resource_usage"), label=f"{label} resource usage")
    _equal(usage.get("available"), True, label=f"{label} resource accounting")
    _equal(usage.get("active_processes"), 0, label=f"{label} active processes")
    supervision = _mapping(receipt.get("supervision"), label=f"{label} supervision")
    _equal(supervision.get("timed_out"), False, label=f"{label} timed out")
    _equal(supervision.get("interrupted"), False, label=f"{label} interrupted")
    _equal(
        supervision.get("tree_termination_confirmed"),
        True,
        label=f"{label} tree termination",
    )
    _equal(supervision.get("descendant_cleanup_triggered"), False, label=f"{label} leak")
    target_output = _mapping(receipt.get("target_output"), label=f"{label} output")
    _equal(target_output.get("complete"), True, label=f"{label} complete output")
    streams = _mapping(target_output.get("streams"), label=f"{label} streams")
    _equal(set(streams), {"stdout", "stderr"}, label=f"{label} stream names")
    for stream_name, raw_stream in streams.items():
        stream = _mapping(raw_stream, label=f"{label} {stream_name}")
        _equal(stream.get("truncated"), False, label=f"{label} {stream_name} truncated")
        _equal(
            stream.get("captured_bytes"),
            stream.get("total_bytes"),
            label=f"{label} {stream_name} byte count",
        )
        _equal(
            stream.get("sha256"),
            stream.get("full_stream_sha256"),
            label=f"{label} {stream_name} hash",
        )
    mutable_paths = _mapping(receipt.get("mutable_paths"), label=f"{label} mutable paths")
    build_root = ROOT / "build"
    for name, value in mutable_paths.items():
        if not isinstance(value, str):
            raise LocalFutureOriginSelectionError(f"{label} mutable path {name} is invalid")
        candidate = Path(os.path.abspath(value))
        if os.path.commonpath((str(build_root), str(candidate))) != str(build_root):
            raise LocalFutureOriginSelectionError(
                f"{label} mutable path {name} escapes build"
            )


def _verify_supervisor_receipt(
    receipt: Mapping[str, object], *, execution_sha256: str, label: str
) -> None:
    _equal(
        receipt.get("schema_version"),
        "pfc_lt_workspace_local_supervisor.v1",
        label=f"{label} schema",
    )
    _equal(receipt.get("workspace"), str(ROOT), label=f"{label} workspace")
    _equal(receipt.get("status"), "WORKER_EXIT_ZERO_NOT_AUTHORITY", label=f"{label} status")
    _equal(receipt.get("worker_exit_code"), 0, label=f"{label} worker exit")
    _equal(receipt.get("scientific_authorization"), False, label=f"{label} science")
    _equal(receipt.get("production_authorization"), False, label=f"{label} production")
    _equal(receipt.get("terminal_receipt_write_attempts"), 1, label=f"{label} writes")
    for field in (
        "scientific_authorization",
        "evaluation_authorization",
        "runtime_authority",
        "promotion_authorization",
        "production_authorization",
    ):
        _equal(receipt.get(field), False, label=f"{label} {field}")
    deadline = _mapping(receipt.get("deadline"), label=f"{label} deadline")
    _equal(deadline.get("deadline_exceeded"), False, label=f"{label} deadline exceeded")
    execution = _mapping(receipt.get("execution_receipt"), label=f"{label} execution")
    _equal(execution.get("sha256"), execution_sha256, label=f"{label} execution hash")
    capability = _mapping(
        receipt.get("worker_capability"), label=f"{label} worker capability"
    )
    _equal(capability.get("one_shot"), True, label=f"{label} one-shot capability")
    _equal(capability.get("consumed"), True, label=f"{label} consumed capability")
    _equal(
        capability.get("absent_after_worker"),
        True,
        label=f"{label} absent capability",
    )
    usage = _mapping(receipt.get("resource_usage"), label=f"{label} resource usage")
    _equal(usage.get("active_processes"), 0, label=f"{label} active processes")


def _verify_bound_current_evidence_recursive_audit() -> int:
    execution = _read_json(
        _safe_path(ROOT, _CURRENT_EVIDENCE_AUDIT_EXECUTION[0], label="current audit"),
        _CURRENT_EVIDENCE_AUDIT_EXECUTION[1],
        label="current evidence recursive audit execution",
    )
    supervisor = _read_json(
        _safe_path(
            ROOT,
            _CURRENT_EVIDENCE_AUDIT_SUPERVISOR[0],
            label="current audit supervisor",
        ),
        _CURRENT_EVIDENCE_AUDIT_SUPERVISOR[1],
        label="current evidence recursive audit supervisor",
    )
    _equal(
        execution.get("status"),
        "TARGET_EXIT_ZERO_NOT_AUTHORITY",
        label="current audit status",
    )
    _equal(execution.get("target_exit_code"), 0, label="current audit exit")
    pytest_result = _mapping(
        execution.get("pytest_result"), label="current audit pytest result"
    )
    _equal(pytest_result.get("tests"), 32, label="current audit test count")
    _equal(pytest_result.get("passed"), 32, label="current audit passed count")
    _equal(pytest_result.get("failures"), 0, label="current audit failures")
    _equal(pytest_result.get("errors"), 0, label="current audit errors")
    command = execution.get("redacted_command")
    if not isinstance(command, list):
        raise LocalFutureOriginSelectionError("current audit command is invalid")
    required_arguments = {
        "tests\\test_audit_ch_lt_current_evidence_selection_script.py",
        "tests\\test_ch_lt_prospective_capture_ledger.py",
    }
    if not required_arguments.issubset(set(command)):
        raise LocalFutureOriginSelectionError("current recursive audit scope is invalid")
    closure = _mapping(
        _mapping(
            execution.get("execution_identity"), label="current audit identity"
        ).get("target_import_closure"),
        label="current audit import closure",
    )
    _equal(
        closure.get("status"),
        "BOUND_REPO_LOCAL_PTH",
        label="current audit import closure status",
    )
    sys_path = closure.get("sys_path")
    if not isinstance(sys_path, list) or sys_path.count(str(ROOT)) != 1:
        raise LocalFutureOriginSelectionError(
            "current audit must contain the repository root exactly once in sys.path"
        )
    _equal(
        supervisor.get("status"),
        "WORKER_EXIT_ZERO_NOT_AUTHORITY",
        label="current audit supervisor status",
    )
    _equal(
        _mapping(supervisor.get("execution_receipt"), label="current audit receipt").get(
            "sha256"
        ),
        _CURRENT_EVIDENCE_AUDIT_EXECUTION[1],
        label="current audit execution receipt hash",
    )
    _equal(
        _mapping(supervisor.get("deadline"), label="current audit deadline").get(
            "deadline_exceeded"
        ),
        False,
        label="current audit deadline exceeded",
    )
    _equal(
        supervisor.get("terminal_receipt_write_attempts"),
        1,
        label="current audit terminal writes",
    )
    return 2


def audit_selection(
    *, registry_path: Path, expected_registry_sha256: str
) -> dict[str, object]:
    runtime_import_closure = _verify_runtime_import_closure()
    if Path.cwd() != ROOT:
        raise LocalFutureOriginSelectionError("cwd must be the canonical repository")
    selected_registry = Path(os.path.abspath(registry_path))
    if selected_registry != ROOT / REGISTRY_RELATIVE:
        raise LocalFutureOriginSelectionError("registry must be the canonical selection")
    if expected_registry_sha256 != REGISTRY_SHA256:
        raise LocalFutureOriginSelectionError(
            "caller-held registry SHA-256 does not match frozen identity"
        )
    registry = _read_json(selected_registry, REGISTRY_SHA256, label="selection")
    _exact_keys(registry, _TOP_LEVEL_KEYS, label="selection")
    _equal(
        registry.get("schema_version"),
        "ch_lt_local_future_origin_selection.v6",
        label="schema version",
    )
    _equal(
        registry.get("status"),
        (
            "SEALED_LOCAL_FUTURE_ORIGIN_REHEARSAL_V7_PACKAGED_MODULE_"
            "SOURCES_BOUND_NONCOUNTABLE_NO_GO"
        ),
        label="status",
    )
    _equal(registry.get("market"), "CH", label="market")
    _equal(
        registry.get("supersession"),
        {
            "supersedes_selection_path": (
                ".planning/phases/14-lt-audit-remediation/"
                "CH-LT-LOCAL-FUTURE-ORIGIN-SELECTION-V5-20260731.json"
            ),
            "supersedes_selection_sha256": (
                "77e8480a563cf64d095ade55282af64e9b569fa9e77c003084d639adf94d35f5"
            ),
            "supersedes_selection_id": (
                "4f13d9fe274087449cca92a5ed234f997aefb9fb3bc8ac0e799cf42fc0fbf5e7"
            ),
            "v5_sealed_byte_identity_preserved": True,
            "same_origin_not_an_additional_origin": True,
            "reason": (
                "MOVE_PROSPECTIVE_RUNTIME_FROM_SCRIPTS_INTO_PACKAGED_NAMESPACE_"
                "WITHOUT_CHANGING_SEALED_PREDICTIONS"
            ),
            "v5_retention_role": (
                "SUPERSEDED_IMMUTABLE_LOCAL_REHEARSAL_AUDIT_ONLY"
            ),
        },
        label="supersession",
    )
    selection_id = registry.get("selection_id")
    unsigned = dict(registry)
    unsigned.pop("selection_id", None)
    _equal(
        selection_id,
        hashlib.sha256(_canonical(unsigned)).hexdigest(),
        label="selection id",
    )

    bindings = _mapping(registry.get("bindings"), label="bindings")
    _equal(set(bindings), set(_EXPECTED_BINDINGS), label="binding names")
    linked: dict[str, Mapping[str, object]] = {}
    for name, (expected_path, expected_hash) in _EXPECTED_BINDINGS.items():
        reference = _mapping(bindings.get(name), label=name)
        expected_reference_keys = {"path", "sha256"}
        if name == "origin_target_mask_inventory":
            expected_reference_keys.add("inventory_id")
        elif name == "prediction_commitment":
            expected_reference_keys.add("commitment_id")
        elif name == "sealed_predictions":
            expected_reference_keys.add("prediction_set_id")
        elif name == "current_evidence_selection":
            expected_reference_keys.update(
                {"registry_id", "independent_rolling_origin_count"}
            )
        elif name == "hourly_scoring_contract":
            expected_reference_keys.add("contract_id")
        _exact_keys(reference, expected_reference_keys, label=name)
        _equal(reference.get("path"), expected_path, label=f"{name} path")
        _equal(reference.get("sha256"), expected_hash, label=f"{name} hash")
        linked[name] = _read_json(
            _safe_path(ROOT, expected_path, label=name), expected_hash, label=name
        )

    origin = _mapping(registry.get("selected_origin"), label="selected origin")
    _exact_keys(origin, _ORIGIN_KEYS, label="selected origin")
    _equal(origin.get("origin_precedes_first_target"), True, label="origin precedence")
    _equal(origin.get("local_clock_trusted"), False, label="local clock trust")
    _equal(origin.get("duplicate_origin_as_of_utc"), "FORBIDDEN", label="duplicate origin")
    inventory = linked["origin_target_mask_inventory"]
    inventory_origin = _mapping(inventory.get("origin"), label="inventory origin")
    targets = inventory.get("targets")
    if not isinstance(targets, list) or len(targets) != 36:
        raise LocalFutureOriginSelectionError("inventory target grid is invalid")
    first_target = _mapping(targets[0], label="first target")
    last_target = _mapping(targets[-1], label="last target")
    _equal(origin.get("origin_as_of_utc"), inventory_origin.get("as_of_utc"), label="origin time")
    _equal(origin.get("origin_id"), inventory_origin.get("origin_id"), label="origin id")
    _equal(
        origin.get("first_target_delivery_start_utc"),
        first_target.get("delivery_start_utc"),
        label="first target start",
    )
    _equal(
        origin.get("last_target_delivery_end_utc"),
        last_target.get("delivery_end_utc"),
        label="last target end",
    )
    if _parse_utc(origin.get("origin_as_of_utc"), label="origin time") >= _parse_utc(
        origin.get("first_target_delivery_start_utc"), label="first target start"
    ):
        raise LocalFutureOriginSelectionError("origin does not precede first delivery")
    implementation = _mapping(
        registry.get("implementation_binding"), label="implementation binding"
    )
    if _parse_utc(
        implementation.get("source_binding_completed_at_utc_untrusted"),
        label="implementation binding time",
    ) >= _parse_utc(
        origin.get("first_target_delivery_start_utc"), label="first target start"
    ):
        raise LocalFutureOriginSelectionError(
            "implementation binding does not precede first delivery"
        )

    commitment = linked["prediction_commitment"]
    predictions = linked["sealed_predictions"]
    _equal(commitment.get("origin_id"), origin.get("origin_id"), label="commitment origin")
    _equal(predictions.get("origin_id"), origin.get("origin_id"), label="prediction origin")
    _equal(commitment.get("target_count"), 36, label="commitment targets")
    _equal(predictions.get("prediction_count"), 108, label="prediction count")
    sealed = _mapping(commitment.get("sealed_payload"), label="sealed payload")
    _equal(
        sealed.get("sha256"),
        _EXPECTED_BINDINGS["sealed_predictions"][1],
        label="sealed prediction hash",
    )
    commitment_inventory = _mapping(
        _mapping(commitment.get("bindings"), label="commitment bindings").get(
            "origin_target_mask_inventory"
        ),
        label="commitment inventory",
    )
    _equal(
        commitment_inventory.get("sha256"),
        _EXPECTED_BINDINGS["origin_target_mask_inventory"][1],
        label="commitment inventory hash",
    )
    nested_binding_count = _verify_prediction_chain(
        inventory=inventory,
        commitment=commitment,
        predictions=predictions,
    )
    nested_binding_count += _verify_scoring_contract(
        linked["hourly_scoring_contract"]
    )
    nested_binding_count += _verify_implementation_binding(implementation)

    current = linked["current_evidence_selection"]
    nested_binding_count += _verify_bound_current_evidence_recursive_audit()
    _equal(current.get("independent_rolling_origin_count"), 0, label="rolling origins")
    for field in (
        "future_holdout_consumed",
        "t057_consumed",
        "scientific_admission",
        "production_authorization",
    ):
        _equal(current.get(field), False, label=f"current evidence {field}")

    _verify_execution_receipt(
        linked["inventory_execution_receipt"],
        expected_command_tail=[
            "-B",
            "-m",
            "scripts.audit_ch_lt_origin_target_mask_inventory",
            "build",
            "--origin-as-of-utc",
            "2026-07-31T07:20:27Z",
            "--output",
            str(ROOT / _EXPECTED_BINDINGS["origin_target_mask_inventory"][0]),
        ],
        label="inventory execution",
    )
    _verify_execution_receipt(
        linked["prediction_execution_receipt"],
        expected_command_tail=[
            "-B",
            "-m",
            "scripts.build_ch_lt_structural_prediction_commitment",
            "--inventory",
            str(ROOT / _EXPECTED_BINDINGS["origin_target_mask_inventory"][0]),
            "--expected-inventory-sha256",
            _EXPECTED_BINDINGS["origin_target_mask_inventory"][1],
            "--output-directory",
            str((ROOT / _EXPECTED_BINDINGS["prediction_commitment"][0]).parent),
        ],
        label="prediction execution",
    )
    _verify_supervisor_receipt(
        linked["inventory_supervisor_receipt"],
        execution_sha256=_EXPECTED_BINDINGS["inventory_execution_receipt"][1],
        label="inventory supervisor",
    )
    _verify_supervisor_receipt(
        linked["prediction_supervisor_receipt"],
        execution_sha256=_EXPECTED_BINDINGS["prediction_execution_receipt"][1],
        label="prediction supervisor",
    )

    boundary = _mapping(registry.get("authority_boundary"), label="authority boundary")
    _equal(boundary, _AUTHORITY_BOUNDARY, label="authority boundary")
    for document, label in ((inventory, "inventory"), (commitment, "commitment"), (predictions, "predictions")):
        for field in ("future_holdout_consumed", "t057_consumed", "scientific_admission", "production_authorization"):
            value = (
                _mapping(document.get("authority_boundary"), label=f"{label} boundary").get(field)
                if label == "commitment"
                else document.get(field)
            )
            _equal(value, False, label=f"{label} {field}")

    evaluation = _mapping(
        registry.get("evaluation_contract"), label="evaluation contract"
    )
    _equal(evaluation, _EVALUATION_CONTRACT, label="evaluation contract")
    lifecycle = _mapping(registry.get("truth_lifecycle"), label="truth lifecycle")
    _equal(lifecycle, _TRUTH_LIFECYCLE, label="truth lifecycle")
    _equal(
        registry.get("next_external_requirements"),
        _EXTERNAL_REQUIREMENTS,
        label="external requirements",
    )
    for field in (
        "future_holdout_consumed",
        "t057_consumed",
        "scientific_admission",
        "production_authorization",
        "promotion_gate",
    ):
        _equal(registry.get(field), False, label=f"selection {field}")

    return {
        "schema_version": "ch_lt_local_future_origin_selection_assessment.v6",
        "status": (
            "VALID_LOCAL_FUTURE_ORIGIN_REHEARSAL_V7_PACKAGED_MODULE_"
            "SOURCES_BOUND_NONCOUNTABLE_NO_GO"
        ),
        "selection_id": selection_id,
        "origin_id": origin.get("origin_id"),
        "origin_as_of_utc": origin.get("origin_as_of_utc"),
        "first_target_delivery_start_utc": origin.get("first_target_delivery_start_utc"),
        "target_count": 36,
        "prediction_count": 108,
        "recursive_binding_count": len(linked) + nested_binding_count,
        "runtime_import_closure": runtime_import_closure,
        "origin_precedes_first_target": True,
        "outcome_blind": True,
        "countable_prospective_origin": False,
        "future_holdout_consumed": False,
        "t057_consumed": False,
        "scientific_admission": False,
        "production_authorization": False,
        "promotion_gate": False,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = audit_selection(
            registry_path=args.registry,
            expected_registry_sha256=args.expected_registry_sha256,
        )
    except (OSError, LocalFutureOriginSelectionError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "ch_lt_local_future_origin_selection_error.v1",
                    "status": "INVALID_LOCAL_FUTURE_ORIGIN_SELECTION_NO_GO",
                    "error": str(exc),
                    "scientific_admission": False,
                    "production_authorization": False,
                    "promotion_gate": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
