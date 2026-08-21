"""Fail-closed structural receipts for future CH EEX short-tenor selection.

The validators in this module prove schema, hash, time-order and outcome-blind
consistency only.  They deliberately cannot authenticate a signature, trusted
time, source availability or execution authority.  Future empirical use must
wrap the accepted bytes in the existing path-based signed replay boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from pfc_shaping.pipeline.strict_structured_data import (
    StrictStructuredDataError,
    load_strict_json,
)

RECEIPT_CONTRACT_SCHEMA = "fmv-eex-ch-short-tenor-receipt-contract-v1"
TRAINING_RECEIPT_SCHEMA = "fmv_eex_ch_short_tenor_training_receipt.v1"
SELECTION_RECEIPT_SCHEMA = "fmv_eex_ch_short_tenor_selection_receipt.v1"
CONTRACT_STATUS = "PASS_LOCAL_RECEIPT_SCHEMA_ONLY_NO_AUTHENTICATED_RECEIPTS"
RECEIPT_STATUS = "STRUCTURAL_ONLY_UNAUTHENTICATED_NO_EXECUTION"
CONTRACT_CONTENT_ID = (
    "d4f7e4e87b8e059255e1a85a589f41d54596756d731ce4a9781b2dbc46d61068"
)

POLICY_CONTENT_ID = (
    "189e7ab9d460daefbaa689f1dc8ae4c3bb269f7b5bcb5eae95e77b6d0d4d3e43"
)
POLICY_DOCUMENT_SHA256 = (
    "367ae861ecb6273dbd72ff308fdd0523b8640e57f7dba122fade4d127e7b1aae"
)
CONTRAST_BUNDLE_CONTENT_ID = (
    "309b9f07236d1cfb32b3d92c1ed5413bde6966fa383629882a539ccbd60e9cb5"
)
CONTRAST_BUNDLE_MANIFEST_SHA256 = (
    "57b571bef311cf22955edc415e650f55671ad2f243141d4f334ccac425d8bb0f"
)
COMBINATION_PROOF_CONTENT_ID = (
    "122afd7292549b00a946a2893487d344e459c420a8bbdf99c11714c7308a8cfd"
)
COMBINATION_PROOF_MANIFEST_SHA256 = (
    "d9772a5e18eea9adb3f6aa9a085558fe0bf91534b03d2e172c4d3c16b0626dab"
)
EVIDENCE_SELECTION_CONTENT_ID = (
    "792149a7d46834b17507ea4ec6a15dbfeb2ab9233b058b9783673a58a57c26aa"
)
EVIDENCE_SELECTION_DOCUMENT_SHA256 = (
    "2517c106e5cfa9e6ab2dc39df301cb620df53aeb91948c652e58ef40015ac643"
)
EVIDENCE_SELECTION_PROOF_CONTENT_ID = (
    "ca099abf04153724254663a880152ba9ed6979638b89eb3462b716a0af8fb041"
)
EVIDENCE_SELECTION_PROOF_MANIFEST_SHA256 = (
    "6f0212a01490104c0e5fde40d3b4004ad0e9589424ca0766404eaff7c71b036d"
)
SHAPE_PROJECTION_CONTENT_ID = (
    "4da441b09c5559d772fd507214b2989489314bd9114a9328752a150f3957450a"
)
SUCCESSOR_CORE_ID = (
    "d86dab183dd95a2ca3cda0862f2c9f50cec01716a37c6cdc1c3a15763254dcc3"
)
SUCCESSOR_CORE_SHA256 = (
    "e2c2a6f9d3ca677991f3f76f03dec1328492e2f60a4015b7796a2ff6aa6f905a"
)

REQUIRED_MATERIAL_ROLES = (
    "EEX_PIT_CATALOG",
    "ENTSOE_PIT_CATALOG",
    "D225_SELECTION_DOCUMENT",
    "D225_SELECTION_PROOF",
    "D220_CONTRAST_BUNDLE",
    "D225_COMBINATION_PROOF",
    "D219_PROJECTION_PROOF",
    "D225_COMBINATION_POLICY",
    "ORIGIN_TARGET_MASK",
    "CANDIDATE_GRID",
    "FOLD_INVENTORY",
    "IMPLEMENTATION_CLOSURE",
    "RUNTIME_LOCK",
)
TRAINING_BYPRODUCT_ROLES = (
    "access_log",
    "environment_manifest",
    "inner_loss_commitment",
    "training_log",
)
SELECTION_BYPRODUCT_ROLES = (
    "access_log",
    "environment_manifest",
    "selection_log",
)
SELECTION_RULE = (
    "ONE_STANDARD_ERROR_THEN_LOWEST_LATTICE_RANK_WITH_FIXED_FAMILY_TIE_ORDER"
)
FORBIDDEN_INPUTS_AND_CLAIMS = (
    "OMPEX as training tuning selection cap margin target or truth",
    "AFRY restricted numeric values",
    "sealed T057 outcomes",
    "future holdout truth",
    "local non-PIT EEX chronology as empirical rolling-origin evidence",
    "legacy or synthetic ENTSO-E substitution",
    "outer target truth before candidate freeze",
    "self-declared authority or authentication",
    "unbound source runtime grid fold or mask bytes",
    "unbound outer-origin fold PIT-cutoff or inner-loss commitment",
    "numeric prices coefficients caps losses or margins in structural receipts",
    "candidate assembly promotion or production authority",
)
REQUIRED_BEFORE_AUTHENTICATED_USE = (
    "externally frozen D225 numeric successor with direct-CH power decisions",
    "signed EEX provider-vintage availability catalog",
    "governed ENTSO-E point-in-time catalog",
    "frozen exact origin-target mask and inner-fold inventory",
    "hash-bound nonempty candidate and hyperparameter grid",
    "independent governance execution trusted-time and acquisition trust anchors",
    "path-based same-snapshot signature and byte replay",
    "new independently frozen future holdout distinct from sealed T057",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_MAX_RECEIPT_BYTES = 256 * 1024


class ShortTenorReceiptError(ValueError):
    """Raised when structural receipt evidence is ambiguous or overclaims."""


def validate_short_tenor_receipt_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate the receipt schema contract without granting execution."""

    _require_mapping(document, "receipt contract")
    _require_exact_keys(
        document,
        {
            "schema_version",
            "status",
            "purpose",
            "selected_boundary",
            "attestation_model",
            "receipt_schemas",
            "required_material_roles",
            "common_receipt_contract",
            "inner_fold_contract",
            "training_receipt_contract",
            "selection_receipt_contract",
            "authority_separation",
            "forbidden_inputs_and_claims",
            "authorities",
            "required_before_authenticated_use",
        },
        label="receipt contract",
    )
    _equal(document.get("schema_version"), RECEIPT_CONTRACT_SCHEMA, "contract schema")
    _equal(document.get("status"), CONTRACT_STATUS, "contract status")

    boundary = _require_mapping(document.get("selected_boundary"), "selected boundary")
    expected_boundary = {
        "evidence_selection_content_id": EVIDENCE_SELECTION_CONTENT_ID,
        "evidence_selection_document_sha256": EVIDENCE_SELECTION_DOCUMENT_SHA256,
        "evidence_selection_proof_content_id": EVIDENCE_SELECTION_PROOF_CONTENT_ID,
        "evidence_selection_proof_manifest_sha256": (
            EVIDENCE_SELECTION_PROOF_MANIFEST_SHA256
        ),
        "policy_content_id": POLICY_CONTENT_ID,
        "policy_document_sha256": POLICY_DOCUMENT_SHA256,
        "policy_decision": "D-20260805-225",
        "contrast_bundle_content_id": CONTRAST_BUNDLE_CONTENT_ID,
        "contrast_bundle_manifest_sha256": CONTRAST_BUNDLE_MANIFEST_SHA256,
        "combination_proof_content_id": COMBINATION_PROOF_CONTENT_ID,
        "combination_proof_manifest_sha256": COMBINATION_PROOF_MANIFEST_SHA256,
        "shape_projection_content_id": SHAPE_PROJECTION_CONTENT_ID,
        "successor_core_id": SUCCESSOR_CORE_ID,
        "successor_core_sha256": SUCCESSOR_CORE_SHA256,
    }
    _equal(dict(boundary), expected_boundary, "selected boundary")

    model = _require_mapping(document.get("attestation_model"), "attestation model")
    _require_exact_keys(
        model,
        {
            "in_toto_attestation_framework_profile",
            "in_toto_statement",
            "slsa_provenance_predicate",
            "dsse_payload_type",
            "adopted_subject_rule",
            "adopted_material_rule",
            "adopted_run_rule",
            "statement_subject_digest_sha256_required",
            "slsa_build_definition_and_run_details_required",
            "external_parameters_and_resolved_dependencies_hash_bound",
            "local_profile_rejects_unknown_fields",
            "generic_slsa_unknown_field_rule_acknowledged",
            "local_signed_replay_reference",
            "path_based_signature_and_trusted_time_verification_required",
            "self_declared_hash_or_boolean_grants_authority",
            "current_cryptographic_authentication_performed",
        },
        label="attestation model",
    )
    _equal(
        model.get("in_toto_attestation_framework_profile"),
        "v1.2",
        "in-toto framework profile",
    )
    _equal(model.get("in_toto_statement"), "https://in-toto.io/Statement/v1", "in-toto schema")
    _equal(
        model.get("slsa_provenance_predicate"),
        "https://slsa.dev/provenance/v1",
        "SLSA predicate",
    )
    _equal(
        model.get("dsse_payload_type"),
        "application/vnd.in-toto+json",
        "DSSE payload type",
    )
    _equal(
        model.get("adopted_subject_rule"),
        "outputs_are_named_and_bound_by_sha256",
        "attestation subject rule",
    )
    _equal(
        model.get("adopted_material_rule"),
        "every_resolved_dependency_is_named_and_bound_by_sha256",
        "attestation material rule",
    )
    _equal(
        model.get("adopted_run_rule"),
        "builder_invocation_start_finish_and_byproduct_hashes_are_bound",
        "attestation run rule",
    )
    for field in (
        "statement_subject_digest_sha256_required",
        "slsa_build_definition_and_run_details_required",
        "external_parameters_and_resolved_dependencies_hash_bound",
        "local_profile_rejects_unknown_fields",
    ):
        _true(model.get(field), f"attestation model {field}")
    _equal(
        model.get("generic_slsa_unknown_field_rule_acknowledged"),
        "IGNORE_UNRECOGNIZED_FIELDS",
        "generic SLSA unknown-field rule",
    )
    _equal(
        model.get("local_signed_replay_reference"),
        "pfc_shaping.calibration.tier2_monthly_eex_fold_evidence."
        "verify_signed_selection_fold_evidence",
        "signed replay reference",
    )
    _true(
        model.get("path_based_signature_and_trusted_time_verification_required"),
        "path-based verification",
    )
    _false(
        model.get("self_declared_hash_or_boolean_grants_authority"),
        "self-declared authority",
    )
    _false(
        model.get("current_cryptographic_authentication_performed"),
        "current cryptographic authentication",
    )

    schemas = _require_mapping(document.get("receipt_schemas"), "receipt schemas")
    _equal(
        dict(schemas),
        {"training": TRAINING_RECEIPT_SCHEMA, "selection": SELECTION_RECEIPT_SCHEMA},
        "receipt schemas",
    )
    _equal(
        tuple(_string_list(document.get("required_material_roles"), "material roles")),
        REQUIRED_MATERIAL_ROLES,
        "required material roles",
    )

    common = _require_mapping(document.get("common_receipt_contract"), "common receipt")
    common_true_fields = {
        "outer_origin_must_use_canonical_utc_z",
        "all_materials_must_be_available_not_after_outer_origin",
        "subject_and_material_names_unique",
        "all_sha256_values_lowercase_64_hex",
        "candidate_grid_and_fold_inventory_hashes_must_match_materials",
        "outer_origin_entry_hash_required",
        "material_and_subject_names_are_safe_relative_posix",
        "duplicate_json_keys_rejected_by_future_path_replay",
        "training_and_selection_receipt_ids_are_hashes_of_unsigned_canonical_json",
        "receipt_ids_are_not_signatures",
        "training_selection_and_trusted_time_order_required",
        "unknown_fields_fail_closed",
    }
    _require_exact_keys(
        common,
        {"market", "currency_unit", "campaign_type", *common_true_fields},
        label="common receipt contract",
    )
    _equal(common.get("market"), "CH", "contract market")
    _equal(common.get("currency_unit"), "EUR/MWh", "contract unit")
    _equal(common.get("campaign_type"), "SELECTION", "contract campaign")
    for field in common_true_fields:
        _true(common.get(field), f"common contract {field}")

    fold = _require_mapping(document.get("inner_fold_contract"), "inner-fold contract")
    fold_true_fields = {
        "fold_ids_sorted_and_unique",
        "validation_origin_ids_unique",
        "validation_origin_excluded_from_its_training_origins",
        "training_data_max_available_at_not_after_fold_as_of",
        "validation_delivery_end_not_after_truth_available_at",
        "validation_truth_available_at_not_after_outer_origin",
        "fold_as_of_strictly_before_outer_origin",
        "outer_origin_excluded_from_all_inner_train_and_validation_origins",
        "fold_inventory_entry_hash_required",
        "eex_pit_cutoff_receipt_hash_required",
        "entsoe_pit_cutoff_receipt_hash_required",
        "fold_input_snapshot_hash_required",
    }
    _require_exact_keys(
        fold,
        {*fold_true_fields, "minimum_fold_count_selected_here", "minimum_fold_count"},
        label="inner-fold contract",
    )
    for field in fold_true_fields:
        _true(fold.get(field), f"inner-fold contract {field}")
    _false(fold.get("minimum_fold_count_selected_here"), "minimum fold selection")
    _equal(fold.get("minimum_fold_count"), None, "minimum fold count")

    training = _require_mapping(
        document.get("training_receipt_contract"), "training receipt contract"
    )
    training_true_fields = {
        "candidate_config_allowlist_sorted_and_unique",
        "inner_folds_nonempty",
        "training_starts_not_before_plan_freeze",
        "training_finishes_not_after_outer_origin",
        "training_data_max_available_at_not_after_outer_origin",
        "outer_origin_entry_hash_required",
        "inner_loss_commitment_hash_required",
        "inner_loss_commitment_bound_as_training_byproduct",
    }
    training_false_fields = {
        "outer_target_truth_opened",
        "ompex_opened",
        "afry_numeric_values_accessed",
        "t057_accessed",
        "future_holdout_consumed",
        "numeric_hyperparameters_or_prices_emitted",
    }
    _require_exact_keys(
        training,
        {"subject_role", *training_true_fields, *training_false_fields},
        label="training receipt contract",
    )
    _equal(training.get("subject_role"), "TRAINED_CANDIDATE_SET", "training subject")
    for field in training_true_fields:
        _true(training.get(field), f"training contract {field}")
    for field in training_false_fields:
        _false(training.get(field), f"training contract {field}")

    selection = _require_mapping(
        document.get("selection_receipt_contract"), "selection receipt contract"
    )
    selection_true_fields = {
        "training_receipt_id_and_sha256_required",
        "selected_candidate_must_belong_to_training_allowlist",
        "candidate_grid_and_fold_inventory_must_match_training",
        "outer_origin_entry_must_match_training",
        "inner_loss_commitment_must_match_training",
        "selection_starts_not_before_training_finishes",
        "selection_finishes_not_after_outer_origin",
        "inner_loss_commitment_hash_required",
    }
    selection_false_fields = {
        "outer_target_truth_opened",
        "ompex_opened",
        "afry_numeric_values_accessed",
        "t057_accessed",
        "future_holdout_consumed",
        "performance_values_emitted",
    }
    _require_exact_keys(
        selection,
        {
            "subject_role",
            "selection_rule",
            *selection_true_fields,
            *selection_false_fields,
        },
        label="selection receipt contract",
    )
    _equal(
        selection.get("subject_role"),
        "SELECTED_SHORT_TENOR_CANDIDATE",
        "selection subject",
    )
    _equal(selection.get("selection_rule"), SELECTION_RULE, "selection rule")
    for field in selection_true_fields:
        _true(selection.get(field), f"selection contract {field}")
    for field in selection_false_fields:
        _false(selection.get(field), f"selection contract {field}")

    separation = _require_mapping(
        document.get("authority_separation"), "authority separation"
    )
    separation_true_fields = {
        "governance_execution_trusted_time_key_ids_required",
        "governance_execution_and_trusted_time_keys_pairwise_distinct",
        "source_acquisition_key_ids_required",
        "source_trusted_time_key_ids_required",
        "non_time_and_trusted_time_key_sets_disjoint",
        "independent_signature_verification_required_before_execution",
    }
    separation_false_fields = {
        "current_independent_signature_verification_performed",
        "current_training_receipt_authenticated",
        "current_selection_receipt_authenticated",
    }
    _require_exact_keys(
        separation,
        {*separation_true_fields, *separation_false_fields},
        label="authority separation",
    )
    for field in separation_true_fields:
        _true(separation.get(field), f"authority separation {field}")
    for field in separation_false_fields:
        _false(separation.get(field), f"authority separation {field}")

    authorities = _require_mapping(document.get("authorities"), "authorities")
    authority_false_fields = {
        "cryptographic_authentication",
        "point_in_time_availability_proven",
        "training_authorized",
        "selection_authorized",
        "model_input_authorized",
        "candidate_assembly_authorized",
        "promotion_authorized",
        "production_authorized",
    }
    _require_exact_keys(
        authorities,
        {"local_schema_validation", "synthetic_fixture_testing", *authority_false_fields},
        label="authorities",
    )
    _true(authorities.get("local_schema_validation"), "local schema validation")
    _true(authorities.get("synthetic_fixture_testing"), "synthetic fixture testing")
    for field in authority_false_fields:
        _false(authorities.get(field), f"authority {field}")

    forbidden = tuple(
        _string_list(
            document.get("forbidden_inputs_and_claims"), "forbidden inputs and claims"
        )
    )
    _equal(forbidden, FORBIDDEN_INPUTS_AND_CLAIMS, "forbidden inputs and claims")
    blockers = _string_list(
        document.get("required_before_authenticated_use"), "authentication blockers"
    )
    _equal(
        tuple(blockers),
        REQUIRED_BEFORE_AUTHENTICATED_USE,
        "authentication blockers",
    )
    contract_content_id = _sha256_json(document)
    _equal(contract_content_id, CONTRACT_CONTENT_ID, "receipt contract content id")
    return {
        "schema_version": RECEIPT_CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": contract_content_id,
        "evidence_selection_content_id": EVIDENCE_SELECTION_CONTENT_ID,
        "evidence_selection_proof_content_id": EVIDENCE_SELECTION_PROOF_CONTENT_ID,
        "policy_content_id": POLICY_CONTENT_ID,
        "combination_proof_content_id": COMBINATION_PROOF_CONTENT_ID,
        "required_material_roles": list(REQUIRED_MATERIAL_ROLES),
        "blocker_count": len(blockers),
        "cryptographic_authentication_performed": False,
        "training_authorized": False,
        "selection_authorized": False,
        "production_authorized": False,
    }


def validate_structural_training_receipt(
    receipt: Mapping[str, object],
) -> dict[str, object]:
    """Validate one unsigned training receipt as non-authoritative structure."""

    _require_mapping(receipt, "training receipt")
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "receipt_id",
            "receipt_role",
            "status",
            "market",
            "currency_unit",
            "campaign_type",
            "outer_origin_id",
            "outer_origin_as_of_utc",
            "outer_origin_entry_sha256",
            "plan_frozen_at_utc",
            "policy_content_id",
            "successor_core_id",
            "candidate_grid_sha256",
            "fold_inventory_sha256",
            "inner_loss_commitment_sha256",
            "materials",
            "candidate_config_sha256_allowlist",
            "inner_folds",
            "subject",
            "run_details",
            "access_log",
            "authority",
            "claims",
        },
        label="training receipt",
    )
    _reject_numbers(receipt, path="training receipt")
    _common_receipt_header(receipt, schema=TRAINING_RECEIPT_SCHEMA, role="TRAINING")
    outer_id = _text(receipt.get("outer_origin_id"), "outer origin id")
    outer = _utc(receipt.get("outer_origin_as_of_utc"), "outer origin")
    outer_entry_hash = _sha(
        receipt.get("outer_origin_entry_sha256"), "outer origin entry"
    )
    frozen = _utc(receipt.get("plan_frozen_at_utc"), "plan freeze")
    if frozen > outer:
        raise ShortTenorReceiptError("plan freeze must not follow outer origin")

    grid_hash = _sha(receipt.get("candidate_grid_sha256"), "candidate grid")
    fold_inventory_hash = _sha(receipt.get("fold_inventory_sha256"), "fold inventory")
    inner_loss_hash = _sha(
        receipt.get("inner_loss_commitment_sha256"), "inner loss commitment"
    )
    materials = _materials(receipt.get("materials"), outer=outer)
    _bind_selected_materials(materials)
    _equal(materials["CANDIDATE_GRID"]["sha256"], grid_hash, "grid material hash")
    _equal(
        materials["FOLD_INVENTORY"]["sha256"],
        fold_inventory_hash,
        "fold inventory material hash",
    )

    allowlist = tuple(
        _sha_list(
            receipt.get("candidate_config_sha256_allowlist"),
            "candidate config allowlist",
        )
    )
    if tuple(sorted(set(allowlist))) != allowlist:
        raise ShortTenorReceiptError("candidate config allowlist must be sorted and unique")

    folds = _inner_folds(receipt.get("inner_folds"), outer=outer, outer_id=outer_id)
    subject = _subject(
        receipt.get("subject"),
        expected_role="TRAINED_CANDIDATE_SET",
        label="training subject",
    )
    access = _access_log(receipt.get("access_log"), label="training access log")
    run = _run_details(
        receipt.get("run_details"),
        expected_byproducts=TRAINING_BYPRODUCT_ROLES,
        access_log=access,
        label="training run",
    )
    _equal(
        run["byproduct_sha256s"]["inner_loss_commitment"],
        inner_loss_hash,
        "training inner-loss byproduct binding",
    )
    if not (frozen <= run["started_at"] <= run["finished_at"] <= outer):
        raise ShortTenorReceiptError(
            "training run must occur after plan freeze and not after outer origin"
        )
    authority = _authority(receipt.get("authority"), label="training authority")
    _claims(receipt.get("claims"), label="training claims")
    _receipt_id(receipt)
    return {
        "schema_version": TRAINING_RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "receipt_id": str(receipt["receipt_id"]),
        "receipt_sha256": _sha256_json(receipt),
        "outer_origin_id": outer_id,
        "outer_origin_as_of_utc": str(receipt["outer_origin_as_of_utc"]),
        "outer_origin_entry_sha256": outer_entry_hash,
        "candidate_grid_sha256": grid_hash,
        "fold_inventory_sha256": fold_inventory_hash,
        "inner_loss_commitment_sha256": inner_loss_hash,
        "candidate_config_sha256_allowlist": list(allowlist),
        "inner_fold_ids": [item["fold_id"] for item in folds],
        "subject_sha256": subject["sha256"],
        "training_finished_at_utc": run["finished_text"],
        "governance_key_id": authority["governance_key_id"],
        "execution_key_id": authority["execution_key_id"],
        "trusted_time_key_id": authority["trusted_time_key_id"],
        "cryptographic_authentication_performed": False,
        "point_in_time_availability_proven": False,
        "training_authorized": False,
        "model_input_authorized": False,
        "production_authorized": False,
    }


def validate_structural_selection_receipt(
    receipt: Mapping[str, object],
    *,
    training_receipt: Mapping[str, object],
) -> dict[str, object]:
    """Validate selection linkage without authenticating either receipt."""

    training = validate_structural_training_receipt(training_receipt)
    _require_mapping(receipt, "selection receipt")
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "receipt_id",
            "receipt_role",
            "status",
            "market",
            "currency_unit",
            "campaign_type",
            "outer_origin_id",
            "outer_origin_as_of_utc",
            "outer_origin_entry_sha256",
            "policy_content_id",
            "successor_core_id",
            "training_receipt_id",
            "training_receipt_sha256",
            "candidate_grid_sha256",
            "fold_inventory_sha256",
            "selected_candidate_config_sha256",
            "selection_rule",
            "inner_loss_commitment_sha256",
            "subject",
            "run_details",
            "access_log",
            "authority",
            "claims",
        },
        label="selection receipt",
    )
    _reject_numbers(receipt, path="selection receipt")
    _common_receipt_header(receipt, schema=SELECTION_RECEIPT_SCHEMA, role="SELECTION")
    _equal(receipt.get("outer_origin_id"), training["outer_origin_id"], "outer origin id")
    _equal(
        receipt.get("outer_origin_as_of_utc"),
        training["outer_origin_as_of_utc"],
        "outer origin time",
    )
    _equal(
        receipt.get("outer_origin_entry_sha256"),
        training["outer_origin_entry_sha256"],
        "outer origin entry hash",
    )
    _equal(receipt.get("training_receipt_id"), training["receipt_id"], "training id")
    _equal(
        receipt.get("training_receipt_sha256"),
        training["receipt_sha256"],
        "training receipt hash",
    )
    _equal(
        receipt.get("candidate_grid_sha256"),
        training["candidate_grid_sha256"],
        "candidate grid hash",
    )
    _equal(
        receipt.get("fold_inventory_sha256"),
        training["fold_inventory_sha256"],
        "fold inventory hash",
    )
    selected = _sha(
        receipt.get("selected_candidate_config_sha256"), "selected candidate config"
    )
    if selected not in training["candidate_config_sha256_allowlist"]:
        raise ShortTenorReceiptError("selected candidate is outside training allowlist")
    _equal(receipt.get("selection_rule"), SELECTION_RULE, "selection rule")
    _equal(
        receipt.get("inner_loss_commitment_sha256"),
        training["inner_loss_commitment_sha256"],
        "inner loss commitment hash",
    )
    subject = _selection_subject(receipt.get("subject"), selected_config=selected)
    access = _access_log(receipt.get("access_log"), label="selection access log")
    run = _run_details(
        receipt.get("run_details"),
        expected_byproducts=SELECTION_BYPRODUCT_ROLES,
        access_log=access,
        label="selection run",
    )
    outer = _utc(receipt.get("outer_origin_as_of_utc"), "outer origin")
    training_finish = _utc(training["training_finished_at_utc"], "training finish")
    if not (training_finish <= run["started_at"] <= run["finished_at"] <= outer):
        raise ShortTenorReceiptError(
            "selection run must follow training and finish not after outer origin"
        )
    authority = _authority(receipt.get("authority"), label="selection authority")
    _claims(receipt.get("claims"), label="selection claims")
    _receipt_id(receipt)
    return {
        "schema_version": SELECTION_RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "receipt_id": str(receipt["receipt_id"]),
        "receipt_sha256": _sha256_json(receipt),
        "training_receipt_id": training["receipt_id"],
        "outer_origin_id": training["outer_origin_id"],
        "outer_origin_entry_sha256": training["outer_origin_entry_sha256"],
        "inner_loss_commitment_sha256": training[
            "inner_loss_commitment_sha256"
        ],
        "selected_candidate_config_sha256": selected,
        "selected_subject_sha256": subject["sha256"],
        "selection_finished_at_utc": run["finished_text"],
        "governance_key_id": authority["governance_key_id"],
        "execution_key_id": authority["execution_key_id"],
        "trusted_time_key_id": authority["trusted_time_key_id"],
        "cryptographic_authentication_performed": False,
        "point_in_time_availability_proven": False,
        "selection_authorized": False,
        "model_input_authorized": False,
        "candidate_assembly_authorized": False,
        "production_authorized": False,
    }


def with_receipt_id(document: Mapping[str, object]) -> dict[str, object]:
    """Return a copy with its canonical unsigned receipt identifier."""

    output = dict(document)
    output.pop("receipt_id", None)
    output["receipt_id"] = _sha256_json(output)
    return output


def load_strict_receipt_json(payload: bytes) -> Mapping[str, object]:
    """Parse bounded UTF-8 JSON while rejecting duplicate keys."""

    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_RECEIPT_BYTES:
        raise ShortTenorReceiptError("receipt payload must be non-empty bounded bytes")
    try:
        value = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, StrictStructuredDataError) as exc:
        raise ShortTenorReceiptError("receipt payload is not strict UTF-8 JSON") from exc
    return _require_mapping(value, "receipt payload")


def _common_receipt_header(
    receipt: Mapping[str, object], *, schema: str, role: str
) -> None:
    _equal(receipt.get("schema_version"), schema, "receipt schema")
    _equal(receipt.get("receipt_role"), role, "receipt role")
    _equal(receipt.get("status"), RECEIPT_STATUS, "receipt status")
    _equal(receipt.get("market"), "CH", "receipt market")
    _equal(receipt.get("currency_unit"), "EUR/MWh", "receipt unit")
    _equal(receipt.get("campaign_type"), "SELECTION", "receipt campaign")
    _text(receipt.get("outer_origin_id"), "outer origin id")
    _equal(receipt.get("policy_content_id"), POLICY_CONTENT_ID, "policy content id")
    _equal(receipt.get("successor_core_id"), SUCCESSOR_CORE_ID, "successor core id")


def _materials(value: object, *, outer: datetime) -> dict[str, dict[str, str]]:
    if not isinstance(value, list) or len(value) != len(REQUIRED_MATERIAL_ROLES):
        raise ShortTenorReceiptError("materials must contain the exact role inventory")
    output: dict[str, dict[str, str]] = {}
    names: set[str] = set()
    content_ids: set[str] = set()
    for expected_role, raw in zip(REQUIRED_MATERIAL_ROLES, value, strict=True):
        item = _require_mapping(raw, "material")
        _require_exact_keys(
            item,
            {
                "role",
                "name",
                "sha256",
                "content_id",
                "max_available_at_utc",
                "trusted_time_receipt_sha256",
            },
            label="material",
        )
        _equal(item.get("role"), expected_role, "material role order")
        name = _logical_name(item.get("name"), "material name")
        if name in names:
            raise ShortTenorReceiptError("material names must be unique")
        names.add(name)
        digest = _sha(item.get("sha256"), f"{expected_role} sha256")
        content_id = _sha(item.get("content_id"), f"{expected_role} content id")
        if content_id in content_ids:
            raise ShortTenorReceiptError("material content IDs must be unique")
        content_ids.add(content_id)
        available_text = str(item.get("max_available_at_utc", ""))
        if _utc(available_text, f"{expected_role} availability") > outer:
            raise ShortTenorReceiptError("material became available after outer origin")
        output[expected_role] = {
            "name": name,
            "sha256": digest,
            "content_id": content_id,
            "max_available_at_utc": available_text,
            "trusted_time_receipt_sha256": _sha(
                item.get("trusted_time_receipt_sha256"),
                f"{expected_role} trusted-time receipt",
            ),
        }
    return output


def _bind_selected_materials(materials: Mapping[str, Mapping[str, str]]) -> None:
    _equal(
        materials["D225_SELECTION_DOCUMENT"]["content_id"],
        EVIDENCE_SELECTION_CONTENT_ID,
        "D225 selection document content id",
    )
    _equal(
        materials["D225_SELECTION_DOCUMENT"]["sha256"],
        EVIDENCE_SELECTION_DOCUMENT_SHA256,
        "D225 selection document hash",
    )
    _equal(
        materials["D225_SELECTION_PROOF"]["content_id"],
        EVIDENCE_SELECTION_PROOF_CONTENT_ID,
        "D225 selection proof content id",
    )
    _equal(
        materials["D225_SELECTION_PROOF"]["sha256"],
        EVIDENCE_SELECTION_PROOF_MANIFEST_SHA256,
        "D225 selection proof manifest hash",
    )
    _equal(
        materials["D220_CONTRAST_BUNDLE"]["content_id"],
        CONTRAST_BUNDLE_CONTENT_ID,
        "D220 material content id",
    )
    _equal(
        materials["D220_CONTRAST_BUNDLE"]["sha256"],
        CONTRAST_BUNDLE_MANIFEST_SHA256,
        "D220 material manifest hash",
    )
    _equal(
        materials["D225_COMBINATION_PROOF"]["content_id"],
        COMBINATION_PROOF_CONTENT_ID,
        "D225 combination proof content id",
    )
    _equal(
        materials["D225_COMBINATION_PROOF"]["sha256"],
        COMBINATION_PROOF_MANIFEST_SHA256,
        "D225 combination proof manifest hash",
    )
    _equal(
        materials["D219_PROJECTION_PROOF"]["content_id"],
        SHAPE_PROJECTION_CONTENT_ID,
        "D219 material content id",
    )
    _equal(
        materials["D225_COMBINATION_POLICY"]["content_id"],
        POLICY_CONTENT_ID,
        "D225 policy material content id",
    )
    _equal(
        materials["D225_COMBINATION_POLICY"]["sha256"],
        POLICY_DOCUMENT_SHA256,
        "D225 policy material document hash",
    )


def _inner_folds(
    value: object, *, outer: datetime, outer_id: str
) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ShortTenorReceiptError("inner folds must be non-empty")
    output: list[dict[str, str]] = []
    validation_origins: set[str] = set()
    for raw in value:
        fold = _require_mapping(raw, "inner fold")
        _require_exact_keys(
            fold,
            {
                "fold_id",
                "fold_as_of_utc",
                "training_origin_ids",
                "validation_origin_id",
                "training_data_max_available_at_utc",
                "validation_target_delivery_start_utc",
                "validation_target_delivery_end_utc",
                "validation_truth_available_at_utc",
                "fold_inventory_entry_sha256",
                "eex_pit_cutoff_receipt_sha256",
                "entsoe_pit_cutoff_receipt_sha256",
                "fold_input_snapshot_sha256",
            },
            label="inner fold",
        )
        fold_id = _text(fold.get("fold_id"), "fold id")
        fold_as_of = _utc(fold.get("fold_as_of_utc"), "fold as-of")
        training_max = _utc(
            fold.get("training_data_max_available_at_utc"), "fold training cutoff"
        )
        delivery_start = _utc(
            fold.get("validation_target_delivery_start_utc"),
            "validation delivery start",
        )
        delivery_end = _utc(
            fold.get("validation_target_delivery_end_utc"), "validation delivery end"
        )
        truth_available = _utc(
            fold.get("validation_truth_available_at_utc"), "validation truth availability"
        )
        if not (training_max <= fold_as_of < delivery_start <= delivery_end):
            raise ShortTenorReceiptError("inner-fold PIT and delivery order is invalid")
        if not (delivery_end <= truth_available <= outer and fold_as_of < outer):
            raise ShortTenorReceiptError("inner-fold truth or origin exceeds outer origin")
        training_origins = _string_list(
            fold.get("training_origin_ids"), "fold training origin ids"
        )
        if training_origins != sorted(set(training_origins)):
            raise ShortTenorReceiptError("fold training origin IDs must be sorted and unique")
        if outer_id in training_origins:
            raise ShortTenorReceiptError("outer origin appears in inner-fold training")
        validation_origin = _text(
            fold.get("validation_origin_id"), "validation origin id"
        )
        if validation_origin in training_origins:
            raise ShortTenorReceiptError("validation origin appears in fold training origins")
        if validation_origin == outer_id:
            raise ShortTenorReceiptError("outer origin appears as inner-fold validation")
        if validation_origin in validation_origins:
            raise ShortTenorReceiptError("validation origin IDs must be unique")
        validation_origins.add(validation_origin)
        _sha(fold.get("fold_inventory_entry_sha256"), "fold inventory entry")
        _sha(fold.get("eex_pit_cutoff_receipt_sha256"), "EEX PIT cutoff receipt")
        _sha(
            fold.get("entsoe_pit_cutoff_receipt_sha256"),
            "ENTSO-E PIT cutoff receipt",
        )
        _sha(fold.get("fold_input_snapshot_sha256"), "fold input snapshot")
        output.append({"fold_id": fold_id, "validation_origin_id": validation_origin})
    fold_ids = [item["fold_id"] for item in output]
    if fold_ids != sorted(set(fold_ids)):
        raise ShortTenorReceiptError("inner fold IDs must be sorted and unique")
    return output


def _subject(value: object, *, expected_role: str, label: str) -> dict[str, str]:
    subject = _require_mapping(value, label)
    _require_exact_keys(subject, {"role", "name", "sha256"}, label=label)
    _equal(subject.get("role"), expected_role, f"{label} role")
    return {
        "role": expected_role,
        "name": _logical_name(subject.get("name"), f"{label} name"),
        "sha256": _sha(subject.get("sha256"), f"{label} hash"),
    }


def _selection_subject(value: object, *, selected_config: str) -> dict[str, str]:
    subject = _require_mapping(value, "selection subject")
    _require_exact_keys(
        subject,
        {"role", "name", "sha256", "candidate_config_sha256"},
        label="selection subject",
    )
    _equal(
        subject.get("role"),
        "SELECTED_SHORT_TENOR_CANDIDATE",
        "selection subject role",
    )
    _equal(
        subject.get("candidate_config_sha256"),
        selected_config,
        "selection subject config",
    )
    return {
        "role": "SELECTED_SHORT_TENOR_CANDIDATE",
        "name": _logical_name(subject.get("name"), "selection subject name"),
        "sha256": _sha(subject.get("sha256"), "selection subject hash"),
    }


def _access_log(value: object, *, label: str) -> Mapping[str, object]:
    access = _require_mapping(value, label)
    expected = {
        "outer_target_truth_opened",
        "ompex_opened",
        "afry_numeric_values_accessed",
        "t057_accessed",
        "future_holdout_consumed",
        "post_origin_data_accessed",
    }
    _require_exact_keys(access, expected, label=label)
    for field in expected:
        _false(access.get(field), f"{label} {field}")
    return access


def _run_details(
    value: object,
    *,
    expected_byproducts: tuple[str, ...],
    access_log: Mapping[str, object],
    label: str,
) -> dict[str, Any]:
    run = _require_mapping(value, label)
    _require_exact_keys(
        run,
        {
            "builder_id",
            "invocation_id",
            "started_at_utc",
            "finished_at_utc",
            "execution_attestation_sha256",
            "trusted_time_receipt_sha256",
            "byproduct_sha256s",
        },
        label=label,
    )
    builder = _text(run.get("builder_id"), f"{label} builder")
    if "://" not in builder:
        raise ShortTenorReceiptError(f"{label} builder_id must be a URI")
    _text(run.get("invocation_id"), f"{label} invocation")
    started_text = str(run.get("started_at_utc", ""))
    finished_text = str(run.get("finished_at_utc", ""))
    started = _utc(started_text, f"{label} start")
    finished = _utc(finished_text, f"{label} finish")
    if started > finished:
        raise ShortTenorReceiptError(f"{label} starts after it finishes")
    _sha(run.get("execution_attestation_sha256"), f"{label} execution attestation")
    _sha(run.get("trusted_time_receipt_sha256"), f"{label} trusted time receipt")
    byproducts = _require_mapping(run.get("byproduct_sha256s"), f"{label} byproducts")
    _require_exact_keys(byproducts, set(expected_byproducts), label=f"{label} byproducts")
    for role in expected_byproducts:
        _sha(byproducts.get(role), f"{label} byproduct {role}")
    _equal(
        byproducts.get("access_log"),
        _sha256_json(access_log),
        f"{label} access-log binding",
    )
    return {
        "started_at": started,
        "finished_at": finished,
        "started_text": started_text,
        "finished_text": finished_text,
        "byproduct_sha256s": dict(byproducts),
    }


def _authority(value: object, *, label: str) -> dict[str, str]:
    authority = _require_mapping(value, label)
    _require_exact_keys(
        authority,
        {
            "governance_key_id",
            "execution_key_id",
            "trusted_time_key_id",
            "source_acquisition_key_ids",
            "source_trusted_time_key_ids",
            "cryptographic_authentication_performed",
            "independent_authorities_proven",
        },
        label=label,
    )
    governance = _sha(authority.get("governance_key_id"), f"{label} governance key")
    execution = _sha(authority.get("execution_key_id"), f"{label} execution key")
    trusted_time = _sha(
        authority.get("trusted_time_key_id"), f"{label} trusted-time key"
    )
    if len({governance, execution, trusted_time}) != 3:
        raise ShortTenorReceiptError(f"{label} primary key IDs must be pairwise distinct")
    acquisition = set(
        _sha_list(
            authority.get("source_acquisition_key_ids"),
            f"{label} source acquisition keys",
        )
    )
    source_time = set(
        _sha_list(
            authority.get("source_trusted_time_key_ids"),
            f"{label} source trusted-time keys",
        )
    )
    non_time = {governance, execution, *acquisition}
    time_keys = {trusted_time, *source_time}
    if non_time.intersection(time_keys):
        raise ShortTenorReceiptError(f"{label} non-time and trusted-time keys overlap")
    _false(
        authority.get("cryptographic_authentication_performed"),
        f"{label} cryptographic authentication",
    )
    _false(
        authority.get("independent_authorities_proven"),
        f"{label} independent authorities",
    )
    return {
        "governance_key_id": governance,
        "execution_key_id": execution,
        "trusted_time_key_id": trusted_time,
    }


def _claims(value: object, *, label: str) -> None:
    claims = _require_mapping(value, label)
    expected = {
        "receipt_authenticated",
        "point_in_time_availability_proven",
        "training_authorized",
        "selection_authorized",
        "model_input_authorized",
        "candidate_assembly_authorized",
        "promotion_authorized",
        "production_authorized",
    }
    _require_exact_keys(claims, expected, label=label)
    for field in expected:
        _false(claims.get(field), f"{label} {field}")


def _receipt_id(receipt: Mapping[str, object]) -> None:
    unsigned = dict(receipt)
    receipt_id = _sha(unsigned.pop("receipt_id", None), "receipt id")
    _equal(receipt_id, _sha256_json(unsigned), "canonical unsigned receipt id")


def _reject_numbers(value: object, *, path: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        raise ShortTenorReceiptError(f"{path} contains a forbidden numeric value")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_numbers(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_numbers(item, path=f"{path}[{index}]")
        return
    raise ShortTenorReceiptError(f"{path} contains an unsupported value type")


def _utc(value: object, label: str) -> datetime:
    text = str(value)
    if _UTC_Z.fullmatch(text) is None:
        raise ShortTenorReceiptError(f"{label} must use canonical UTC seconds and Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ShortTenorReceiptError(f"{label} is not a valid UTC timestamp") from exc
    if parsed.tzinfo != timezone.utc:
        raise ShortTenorReceiptError(f"{label} must be UTC")
    return parsed


def _sha(value: object, label: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise ShortTenorReceiptError(f"{label} must be lowercase SHA-256")
    return text


def _sha_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ShortTenorReceiptError(f"{label} must be non-empty")
    output = [_sha(item, label) for item in value]
    if output != sorted(set(output)):
        raise ShortTenorReceiptError(f"{label} must be sorted and unique")
    return output


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ShortTenorReceiptError(f"{label} must be non-empty")
    output = [_text(item, label) for item in value]
    if len(set(output)) != len(output):
        raise ShortTenorReceiptError(f"{label} must be unique")
    return output


def _logical_name(value: object, label: str) -> str:
    text = _text(value, label)
    segments = text.split("/")
    if (
        text.startswith("/")
        or "\\" in text
        or ":" in text
        or any(segment in {"", ".", ".."} for segment in segments)
        or any(ord(character) < 32 for character in text)
    ):
        raise ShortTenorReceiptError(
            f"{label} must be a safe relative POSIX logical name"
        )
    return text


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ShortTenorReceiptError(f"{label} must be non-empty exact text")
    return value


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ShortTenorReceiptError(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ShortTenorReceiptError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _equal(value: object, expected: object, label: str) -> None:
    if expected is None or isinstance(expected, bool):
        matches = value is expected
    else:
        matches = value == expected
    if not matches:
        raise ShortTenorReceiptError(f"{label} must be {expected!r}")


def _true(value: object, label: str) -> None:
    _equal(value, True, label)


def _false(value: object, label: str) -> None:
    _equal(value, False, label)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ShortTenorReceiptError("document is not canonical-JSON compatible") from exc


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


__all__ = [
    "COMBINATION_PROOF_CONTENT_ID",
    "COMBINATION_PROOF_MANIFEST_SHA256",
    "CONTRACT_CONTENT_ID",
    "CONTRACT_STATUS",
    "CONTRAST_BUNDLE_CONTENT_ID",
    "CONTRAST_BUNDLE_MANIFEST_SHA256",
    "EVIDENCE_SELECTION_CONTENT_ID",
    "EVIDENCE_SELECTION_DOCUMENT_SHA256",
    "EVIDENCE_SELECTION_PROOF_CONTENT_ID",
    "EVIDENCE_SELECTION_PROOF_MANIFEST_SHA256",
    "POLICY_CONTENT_ID",
    "POLICY_DOCUMENT_SHA256",
    "RECEIPT_CONTRACT_SCHEMA",
    "RECEIPT_STATUS",
    "REQUIRED_MATERIAL_ROLES",
    "SELECTION_RECEIPT_SCHEMA",
    "SELECTION_RULE",
    "SHAPE_PROJECTION_CONTENT_ID",
    "ShortTenorReceiptError",
    "SUCCESSOR_CORE_ID",
    "SUCCESSOR_CORE_SHA256",
    "TRAINING_RECEIPT_SCHEMA",
    "load_strict_receipt_json",
    "validate_short_tenor_receipt_contract",
    "validate_structural_selection_receipt",
    "validate_structural_training_receipt",
    "with_receipt_id",
]
